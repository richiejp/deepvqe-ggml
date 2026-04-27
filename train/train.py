"""DeepVQE AEC training script.

Features:
- Schedule-Free AdamW (default) or SOAP optimizer
- Mixed precision (BF16 autocast, TF32 for remaining FP32 ops)
- Gradient accumulation for large effective batch sizes
- Gradient clipping
- TensorBoard logging (loss, lr, grad norms, audio, spectrograms, delay heatmaps)
- Checkpointing (last N + best)
"""

import argparse
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import schedulefree
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from collections import defaultdict

from data.dataset import AECDataset, DummyAECDataset, FixedSynthDataset
from src.config import load_config
from src.losses import DeepVQELoss, mask_magnitude_regularizer
from src.metrics import compute_aecmos, compute_dnsmos, compute_pesq, compute_stoi
from src.model import DeepVQEAEC
from src.stft import istft
from src.viz import (
    add_scalar_with_help,
    log_loss_ratios,
    log_per_layer_grad_norms,
    log_weight_histograms,
    plot_activation_stats,
    plot_ccm_mask,
    plot_delay_with_gt,
    plot_encoder_activations,
    plot_spectrogram_comparison,
    ActivationCapture,
)
from utils import (
    _unwrap,
    collate_fn,
    delay_samples_to_frame,
    load_checkpoint,
    save_checkpoint,
)


def compute_delay_loss(delay_dist, delay_samples, hop_length, dmax):
    """Cross-entropy loss between predicted delay distribution and ground truth.

    Args:
        delay_dist: (B, T, dmax) — predicted delay distribution from AlignBlock
        delay_samples: (B,) — ground truth delay in samples
        hop_length: int
        dmax: int

    Returns:
        loss: scalar cross-entropy
        accuracy: fraction of examples where peak is within ±1 frame of truth
    """
    B, T, D = delay_dist.shape
    target_frames = delay_samples_to_frame(delay_samples, hop_length, dmax)  # (B,)

    # Cross-entropy: average over all frames (same target for all frames in an example)
    # delay_dist is already a probability distribution, use NLL
    log_probs = torch.log(delay_dist + 1e-10)  # (B, T, dmax)
    # Gather the log-prob at the target frame for each example, averaged over frames
    target_expanded = target_frames[:, None, None].expand(B, T, 1)  # (B, T, 1)
    nll = -log_probs.gather(dim=-1, index=target_expanded).squeeze(-1)  # (B, T)
    loss = nll.mean()

    # Accuracy: peak of mean attention over frames, within ±1 frame of target
    avg_dist = delay_dist.mean(dim=1)  # (B, dmax)
    peak_frames = avg_dist.argmax(dim=-1)  # (B,)
    correct = (peak_frames - target_frames).abs() <= 1
    accuracy = correct.float().mean()

    return loss, accuracy


def compute_attention_entropy(delay_dist):
    """Entropy of the delay attention distribution: H = -sum(p * log(p))."""
    return -(delay_dist * torch.log(delay_dist + 1e-10)).sum(dim=-1).mean()


def compute_erle(mic_wav, enhanced_wav, clean_wav):
    """Compute Echo Return Loss Enhancement in dB.

    ERLE = 10 * log10(||mic - clean||^2 / ||enhanced - clean||^2)
    Positive means echo was reduced.
    """
    echo_plus_noise = mic_wav - clean_wav
    residual = enhanced_wav - clean_wav
    echo_power = (echo_plus_noise ** 2).sum(dim=-1)
    residual_power = (residual ** 2).sum(dim=-1)
    erle = 10 * torch.log10(echo_power / (residual_power + 1e-10))
    return erle.mean()


def create_optimizer(cfg, model_params, warmup_steps):
    """Create optimizer based on config.

    Returns (optimizer, scheduler).
    Schedule-Free manages its own LR internally; SOAP needs an external warmup.
    """
    if cfg.training.optimizer == "soap":
        from soap import SOAP
        optimizer = SOAP(
            model_params,
            lr=cfg.training.lr,
            weight_decay=cfg.training.weight_decay,
            precondition_frequency=10,
        )
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-6, total_iters=warmup_steps,
        ) if warmup_steps > 0 else None
        return optimizer, scheduler
    elif cfg.training.optimizer == "schedulefree":
        optimizer = schedulefree.AdamWScheduleFree(
            model_params,
            lr=cfg.training.lr,
            weight_decay=cfg.training.weight_decay,
            warmup_steps=warmup_steps,
        )
        return optimizer, None
    else:
        raise ValueError(f"Unknown optimizer: {cfg.training.optimizer!r}")


def manage_checkpoints(ckpt_dir, keep_n):
    """Keep only the last N checkpoints (excluding best)."""
    ckpts = sorted(ckpt_dir.glob("epoch_*.pt"), key=lambda p: p.stat().st_mtime)
    while len(ckpts) > keep_n:
        ckpts.pop(0).unlink()


def log_audio_and_spectrograms(writer, model, val_batches, epoch, cfg, device,
                               act_capture):
    """Log audio samples, spectrograms, and diagnostic figures to TensorBoard.

    Args:
        val_batches: list of (batch, sample_idx) tuples — diverse validation
            examples chosen randomly each epoch.
        act_capture: ActivationCapture instance (persistent hooks, avoids
            torch.compile recompilation).
    """
    import matplotlib.pyplot as plt

    model.eval()

    # Use first sample for diagnostic figures (activations, CCM, delay)
    diag_batch, diag_idx = val_batches[0]

    act_capture.enable()

    diag_delay_dist = None
    diag_mic_stft = None

    use_amp = cfg.training.amp and device.type == "cuda"
    autocast_ctx = torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp)

    with torch.no_grad(), autocast_ctx:
        # Log multiple audio samples for review
        for i, (batch, sample_idx) in enumerate(val_batches):
            mic_stft = batch["mic_stft"][sample_idx:sample_idx+1].to(device)
            ref_stft = batch["ref_stft"][sample_idx:sample_idx+1].to(device)
            clean_stft = batch["clean_stft"][sample_idx:sample_idx+1].to(device)

            enhanced, delay_dist, mask_raw = model(mic_stft, ref_stft, return_delay=True)
            length = batch["clean_wav"].shape[-1]

            mic_wav = istft(mic_stft, cfg.audio.n_fft, cfg.audio.hop_length, length=length)
            enh_wav = istft(enhanced, cfg.audio.n_fft, cfg.audio.hop_length, length=length)
            clean_wav = batch["clean_wav"][sample_idx:sample_idx+1].to(device)

            sr = cfg.audio.sample_rate
            tag = f"audio_{i}" if i > 0 else "audio"
            writer.add_audio(f"{tag}/mic", mic_wav[0].cpu(), epoch, sample_rate=sr)
            writer.add_audio(f"{tag}/enhanced", enh_wav[0].cpu(), epoch, sample_rate=sr)
            writer.add_audio(f"{tag}/clean", clean_wav[0].cpu(), epoch, sample_rate=sr)

            # Spectrogram comparison for each sample
            fig = plot_spectrogram_comparison(
                mic_stft, enhanced, clean_stft, sr, cfg.audio.hop_length,
            )
            writer.add_figure(f"spectrograms/comparison_{i}", fig, epoch)
            plt.close(fig)

            # Capture activations/diagnostics from first sample only;
            # disable capture for remaining samples to avoid overhead
            if i == 0:
                diag_delay_dist = delay_dist
                diag_mic_stft = mic_stft
                act_capture.disable()

        activation_store = act_capture.store

        # Delay distribution with ground truth (first sample)
        if diag_delay_dist is not None:
            gt_delay = diag_batch["delay_samples"][diag_idx]
            fig = plot_delay_with_gt(
                diag_delay_dist, gt_delay, cfg.audio.hop_length, cfg.model.dmax,
            )
            writer.add_figure("delay/distribution_with_gt", fig, epoch)
            plt.close(fig)

        # CCM mask analysis (dec1 output = 27ch mask, no BN)
        if "dec1" in activation_store:
            fig = plot_ccm_mask(activation_store["dec1"], diag_mic_stft)
            writer.add_figure("ccm/mask_magnitude", fig, epoch)
            plt.close(fig)

        # Encoder activations
        if activation_store:
            fig = plot_encoder_activations(activation_store)
            writer.add_figure("activations/encoder_stages", fig, epoch)
            plt.close(fig)

            fig = plot_activation_stats(activation_store)
            writer.add_figure("activations/statistics", fig, epoch)
            plt.close(fig)

    model.train()


def train(cfg, resume=None, dummy=False, overfit_real=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    print(f"Device: {device}")

    # Create timestamped run directories so each run is separate in TensorBoard
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_dir = Path(cfg.paths.checkpoint_dir) / run_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(cfg.paths.log_dir) / run_id

    writer = SummaryWriter(log_dir)

    # Model — register hooks before torch.compile so dynamo guards are stable
    model = DeepVQEAEC.from_config(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    act_capture = ActivationCapture(model)
    if device.type == "cuda":
        # Allow enough recompiles for BFloat16/Float32 dtype variants + hook states
        torch._dynamo.config.recompile_limit = 256
        model = torch.compile(model)
        print("Model compiled with torch.compile")

    # Loss
    criterion = DeepVQELoss.from_config(cfg).to(device)

    # Dataset
    if overfit_real:
        target_len = int(cfg.training.clip_length_sec * cfg.audio.sample_rate)
        train_ds = FixedSynthDataset(
            clean_dir=cfg.data.clean_dir,
            noise_dir=cfg.data.noise_dir,
            farend_dir=cfg.data.farend_dir,
            rir_dir=cfg.data.rir_dir,
            delays_ms=cfg.data.overfit_delays_ms,
            sr=cfg.audio.sample_rate,
            target_len=target_len,
            n_fft=cfg.audio.n_fft,
            hop_length=cfg.audio.hop_length,
            snr_db=cfg.data.overfit_snr_db,
            ser_db=cfg.data.overfit_ser_db,
            repeat=cfg.data.overfit_repeat,
            max_rir_length_ms=cfg.data.max_rir_length_ms,
            drr_db=cfg.data.drr_range[0],
        )
        val_ds = train_ds  # same examples for overfit
    elif dummy:
        train_ds = DummyAECDataset(
            length=cfg.data.num_train,
            target_len=int(cfg.training.clip_length_sec * cfg.audio.sample_rate),
            n_fft=cfg.audio.n_fft,
            hop_length=cfg.audio.hop_length,
            delay_range=tuple(int(x) for x in cfg.data.delay_range),
        )
        val_ds = DummyAECDataset(
            length=cfg.data.num_val,
            target_len=int(cfg.training.clip_length_sec * cfg.audio.sample_rate),
            n_fft=cfg.audio.n_fft,
            hop_length=cfg.audio.hop_length,
            delay_range=tuple(int(x) for x in cfg.data.delay_range),
        )
    else:
        train_ds = AECDataset(cfg, split="train")
        val_ds = AECDataset(cfg, split="val")
        print(f"Train: {len(train_ds)} clean files, Val: {len(val_ds)} examples")

    num_workers = cfg.training.num_workers
    pin = device.type == "cuda"

    def worker_init_fn(worker_id):
        """Seed Python random and numpy per worker to avoid duplicate examples."""
        worker_seed = torch.initial_seed() % 2**32
        random.seed(worker_seed + worker_id)
        np.random.seed(worker_seed + worker_id)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=True,
        pin_memory=pin,
        persistent_workers=num_workers > 0,
        worker_init_fn=worker_init_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin,
        persistent_workers=num_workers > 0,
        worker_init_fn=worker_init_fn,
    )

    steps_per_epoch = len(train_loader) // cfg.training.grad_accum_steps
    warmup_steps = cfg.training.warmup_epochs * steps_per_epoch
    optimizer, scheduler = create_optimizer(
        cfg, model.parameters(), warmup_steps,
    )
    print(f"Optimizer: {cfg.training.optimizer} "
          f"(lr={cfg.training.lr}, warmup={warmup_steps} steps)")

    # AMP — BF16 on CUDA (no GradScaler needed), disabled on CPU
    use_amp = cfg.training.amp and device.type == "cuda"
    autocast_ctx = lambda: torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=use_amp,
    )

    # Resume
    start_epoch = 0
    best_val_loss = float("inf")
    if resume:
        start_epoch, saved_loss = load_checkpoint(resume, model, optimizer, scheduler)
        if saved_loss is not None:
            best_val_loss = saved_loss
            print(f"Resumed from epoch {start_epoch}, best_val_loss={best_val_loss:.4f}")
        else:
            print(f"Resumed from epoch {start_epoch}")

    global_step = start_epoch * steps_per_epoch
    accum_steps = cfg.training.grad_accum_steps
    patience = cfg.training.early_stop_patience
    min_delta = cfg.training.early_stop_min_delta
    epochs_without_improvement = 0

    for epoch in range(start_epoch, cfg.training.epochs):
        model.train()
        if isinstance(optimizer, schedulefree.AdamWScheduleFree):
            optimizer.train()

        # Anneal AlignBlock temperature: linear decay from start to end
        t_start = cfg.model.align_temp_start
        t_end = cfg.model.align_temp_end
        t_epochs = cfg.model.align_temp_epochs
        if t_epochs > 0 and epoch < t_epochs:
            temperature = t_start + (t_end - t_start) * epoch / t_epochs
        else:
            temperature = t_end
        align = _unwrap(model).align
        # In-place fill of the registered buffer; mutating .temperature directly
        # would shadow the buffer with a Python attribute and break state_dict.
        align.temperature.fill_(temperature)
        add_scalar_with_help(writer, "train/temperature", temperature, epoch)

        epoch_losses = {"total": 0, "plcmse": 0}
        if cfg.loss.mag_l1_weight > 0:
            epoch_losses["mag_l1"] = 0
        if cfg.loss.time_l1_weight > 0:
            epoch_losses["time_l1"] = 0
        if cfg.loss.sisdr_weight > 0:
            epoch_losses["sisdr"] = 0
        if cfg.loss.energy_preservation_weight > 0:
            epoch_losses["energy_pres"] = 0
        if cfg.loss.delay_weight > 0:
            epoch_losses["delay"] = 0
        if cfg.loss.entropy_weight > 0:
            epoch_losses["entropy"] = 0
        if cfg.loss.mask_reg_weight > 0:
            epoch_losses["mask_reg"] = 0
        epoch_delay_acc = 0
        n_batches = 0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.training.epochs}")
        for batch_idx, batch in enumerate(pbar):
            mic_stft = batch["mic_stft"].to(device, non_blocking=True)
            ref_stft = batch["ref_stft"].to(device, non_blocking=True)
            clean_stft = batch["clean_stft"].to(device, non_blocking=True)
            clean_wav = batch["clean_wav"].to(device, non_blocking=True)
            delay_samp = batch["delay_samples"].to(device, non_blocking=True)

            with autocast_ctx():
                enhanced, delay_dist, mask_raw = model(mic_stft, ref_stft, return_delay=True)
                loss, components = criterion(enhanced, clean_stft, clean_wav)

                # Delay supervision loss
                delay_loss, delay_acc = compute_delay_loss(
                    delay_dist, delay_samp, cfg.audio.hop_length, cfg.model.dmax,
                )
                if cfg.loss.delay_weight > 0:
                    components["delay"] = delay_loss
                    loss = loss + cfg.loss.delay_weight * delay_loss

                # Entropy penalty on delay attention
                if cfg.loss.entropy_weight > 0:
                    entropy = compute_attention_entropy(delay_dist)
                    components["entropy"] = entropy
                    loss = loss + cfg.loss.entropy_weight * entropy

                # Mask magnitude regularizer
                if cfg.loss.mask_reg_weight > 0:
                    mask_reg = mask_magnitude_regularizer(mask_raw)
                    components["mask_reg"] = mask_reg
                    loss = loss + cfg.loss.mask_reg_weight * mask_reg

                # Update total so loss_ratio/* metrics use the true total
                components["total"] = loss

                loss = loss / accum_steps

            loss.backward()

            if (batch_idx + 1) % accum_steps == 0:
                raw_gn = nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)

                # Bail on NaN gradients (check every 10 steps to avoid sync overhead)
                if global_step % 10 == 0 and not torch.isfinite(raw_gn):
                    print(f"\n  FATAL: NaN/Inf gradient norm at step {global_step}, "
                          f"epoch {epoch+1}. Stopping.")
                    optimizer.zero_grad()
                    writer.close()
                    return

                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # Log to TensorBoard every 10 optimizer steps (reduces GPU sync overhead)
                log_this_step = (global_step % 10 == 0)
                if log_this_step:
                    cur_lr = optimizer.param_groups[0]["lr"]
                    add_scalar_with_help(writer, "train/loss", components["total"].item(), global_step)
                    add_scalar_with_help(writer, "train/plcmse", components["plcmse"].item(), global_step)
                    add_scalar_with_help(writer, "train/delay_acc", delay_acc.item(), global_step)
                    add_scalar_with_help(writer, "train/lr", cur_lr, global_step)
                    add_scalar_with_help(writer, "train/grad_norm", raw_gn.item(), global_step)
                    # Log active auxiliary components
                    loss_weights = {"plcmse": cfg.loss.plcmse_weight}
                    for key, w in [("mag_l1", cfg.loss.mag_l1_weight),
                                   ("time_l1", cfg.loss.time_l1_weight),
                                   ("sisdr", cfg.loss.sisdr_weight),
                                   ("smooth_l1", cfg.loss.smooth_l1_weight),
                                   ("energy_pres", cfg.loss.energy_preservation_weight),
                                   ("delay", cfg.loss.delay_weight),
                                   ("entropy", cfg.loss.entropy_weight),
                                   ("mask_reg", cfg.loss.mask_reg_weight)]:
                        if w > 0 and key in components:
                            add_scalar_with_help(writer, f"train/{key}", components[key].item(), global_step)
                            loss_weights[key] = w
                    log_loss_ratios(writer, components, global_step, weights=loss_weights)

                # Per-layer grad norms — expensive, log every 100 steps
                if global_step % 100 == 0:
                    log_per_layer_grad_norms(writer, _unwrap(model), global_step)

                # Update progress bar at logging steps only (reuse synced values)
                if log_this_step:
                    pbar.set_postfix(
                        loss=f"{components['total'].item():.4f}",
                        dacc=f"{delay_acc.item():.0%}",
                        lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                    )

            # Accumulate epoch stats on GPU (no sync until epoch end)
            for k in epoch_losses:
                epoch_losses[k] = epoch_losses[k] + components[k].detach()
            epoch_delay_acc = epoch_delay_acc + delay_acc.detach()
            n_batches += 1

        # Epoch averages (single GPU sync point for all accumulated tensors)
        for k in epoch_losses:
            avg = epoch_losses[k] / max(n_batches, 1)
            epoch_losses[k] = avg.item() if torch.is_tensor(avg) else avg
            add_scalar_with_help(writer, f"train_epoch/{k}", epoch_losses[k], epoch)
        if torch.is_tensor(epoch_delay_acc):
            epoch_delay_acc = (epoch_delay_acc / max(n_batches, 1)).item()
        else:
            epoch_delay_acc /= max(n_batches, 1)
        add_scalar_with_help(writer, "train_epoch/delay_acc", epoch_delay_acc, epoch)

        # Weight histograms every 5 epochs
        if (epoch + 1) % 5 == 0:
            log_weight_histograms(writer, _unwrap(model), epoch)

        # Validation
        if isinstance(optimizer, schedulefree.AdamWScheduleFree):
            optimizer.eval()
        model.eval()
        val_losses = {"total": 0, "plcmse": 0}
        if cfg.loss.mag_l1_weight > 0:
            val_losses["mag_l1"] = 0
        if cfg.loss.time_l1_weight > 0:
            val_losses["time_l1"] = 0
        if cfg.loss.sisdr_weight > 0:
            val_losses["sisdr"] = 0
        if cfg.loss.energy_preservation_weight > 0:
            val_losses["energy_pres"] = 0
        if cfg.loss.delay_weight > 0:
            val_losses["delay"] = 0
        if cfg.loss.entropy_weight > 0:
            val_losses["entropy"] = 0
        if cfg.loss.mask_reg_weight > 0:
            val_losses["mask_reg"] = 0
        val_delay_acc = 0
        n_val = 0
        scenario_erle = defaultdict(list)
        scenario_pesq = defaultdict(list)
        scenario_stoi = defaultdict(list)
        scenario_dnsmos_ovrl = defaultdict(list)
        scenario_echo_mos = defaultdict(list)
        scenario_deg_mos = defaultdict(list)
        pesq_count = 0
        dnsmos_count = 0
        aecmos_count = 0
        pesq_budget = cfg.eval.pesq_subset
        dnsmos_budget = cfg.eval.dnsmos_subset
        aecmos_budget = cfg.eval.aecmos_subset
        # Collect diverse samples for TensorBoard: pick random batches/indices
        n_tb_samples = min(cfg.eval.audio_samples, len(val_loader))
        tb_batch_indices = set(random.sample(range(len(val_loader)), n_tb_samples))
        val_sample_batches = []

        with torch.no_grad(), autocast_ctx():
            for batch_idx, batch in enumerate(val_loader):
                mic_stft = batch["mic_stft"].to(device, non_blocking=True)
                ref_stft = batch["ref_stft"].to(device, non_blocking=True)
                clean_stft = batch["clean_stft"].to(device, non_blocking=True)
                clean_wav = batch["clean_wav"].to(device, non_blocking=True)
                delay_samp = batch["delay_samples"].to(device, non_blocking=True)

                enhanced, delay_dist, mask_raw = model(mic_stft, ref_stft, return_delay=True)
                _, components = criterion(enhanced, clean_stft, clean_wav)

                # Delay accuracy (always computed as diagnostic)
                delay_loss, delay_acc = compute_delay_loss(
                    delay_dist, delay_samp, cfg.audio.hop_length, cfg.model.dmax,
                )

                # Add active auxiliary losses to total
                if cfg.loss.delay_weight > 0:
                    components["delay"] = delay_loss
                    components["total"] = components["total"] + cfg.loss.delay_weight * delay_loss
                if cfg.loss.entropy_weight > 0:
                    entropy = compute_attention_entropy(delay_dist)
                    components["entropy"] = entropy
                    components["total"] = components["total"] + cfg.loss.entropy_weight * entropy
                if cfg.loss.mask_reg_weight > 0:
                    mask_reg = mask_magnitude_regularizer(mask_raw)
                    components["mask_reg"] = mask_reg
                    components["total"] = components["total"] + cfg.loss.mask_reg_weight * mask_reg

                # ERLE + per-scenario metrics
                length = clean_wav.shape[-1]
                mic_wav = batch["mic_wav"].to(device, non_blocking=True)
                enh_wav = istft(enhanced, cfg.audio.n_fft, cfg.audio.hop_length, length=length)

                # Per-sample scenario breakdown
                batch_metadata = batch["metadata"]
                sr = cfg.audio.sample_rate
                for i in range(mic_wav.shape[0]):
                    sc = batch_metadata[i]["scenario"]
                    # Per-sample ERLE
                    echo_i = mic_wav[i] - clean_wav[i]
                    res_i = enh_wav[i] - clean_wav[i]
                    # eps on numerator too — NE-ST / raneo scenarios synthesise
                    # echo ≈ 0 by design, so without it log10(0) → -inf poisons mean
                    erle_i = 10 * torch.log10(
                        ((echo_i ** 2).sum() + 1e-10) / ((res_i ** 2).sum() + 1e-10)
                    ).item()
                    scenario_erle[sc].append(erle_i)

                    # Convert to numpy once for all CPU metrics
                    need_np = (
                        (sc == "double_talk" and pesq_count < pesq_budget)
                        or dnsmos_count < dnsmos_budget
                        or aecmos_count < aecmos_budget
                    )
                    if need_np:
                        enh_np = enh_wav[i].float().cpu().numpy()

                    # PESQ/STOI for double-talk only (budget-limited)
                    if sc == "double_talk" and pesq_count < pesq_budget:
                        cln_np = clean_wav[i].float().cpu().numpy()
                        pesq_score = compute_pesq(cln_np, enh_np, sr=sr)
                        stoi_score = compute_stoi(cln_np, enh_np, sr=sr)
                        if pesq_score is not None:
                            scenario_pesq[sc].append(pesq_score)
                        if stoi_score is not None:
                            scenario_stoi[sc].append(stoi_score)
                        pesq_count += 1

                    # DNSMOS (budget-limited, all scenarios)
                    if dnsmos_count < dnsmos_budget:
                        dnsmos = compute_dnsmos(enh_np)
                        if dnsmos is not None:
                            scenario_dnsmos_ovrl[sc].append(dnsmos["OVRL"])
                        dnsmos_count += 1

                    # AECMOS (budget-limited, all scenarios)
                    if aecmos_count < aecmos_budget:
                        mic_np = mic_wav[i].float().cpu().numpy()
                        ref_np = batch["ref_wav"][i].float().numpy()
                        aecmos = compute_aecmos(ref_np, mic_np, enh_np)
                        if aecmos is not None:
                            scenario_echo_mos[sc].append(aecmos["echo_mos"])
                            scenario_deg_mos[sc].append(aecmos["deg_mos"])
                        aecmos_count += 1

                for k in val_losses:
                    val_losses[k] = val_losses[k] + components[k].detach()
                val_delay_acc = val_delay_acc + delay_acc.detach()
                n_val += 1

                if batch_idx in tb_batch_indices:
                    # Pick a random example within this batch
                    bs = batch["mic_stft"].shape[0]
                    sample_idx = random.randint(0, bs - 1)
                    val_sample_batches.append((batch, sample_idx))

        for k in val_losses:
            avg = val_losses[k] / max(n_val, 1)
            val_losses[k] = avg.item() if torch.is_tensor(avg) else avg
            add_scalar_with_help(writer, f"val/{k}", val_losses[k], epoch)
        if torch.is_tensor(val_delay_acc):
            val_delay_acc = (val_delay_acc / max(n_val, 1)).item()
        else:
            val_delay_acc /= max(n_val, 1)
        add_scalar_with_help(writer, "val/delay_acc", val_delay_acc, epoch)

        # Per-scenario ERLE (skip non-finite from eps-padded log10)
        for sc, values in scenario_erle.items():
            finite = [v for v in values if np.isfinite(v)]
            mean_val = float(np.mean(finite)) if finite else 0.0
            add_scalar_with_help(writer, f"val/erle_{sc}", mean_val, epoch)
        # Overall ERLE: only aggregate scenarios with real echo; NE-ST and
        # raneo have echo=0 by construction so their ERLE is ill-defined.
        echo_scenarios = ("single_talk_farend", "double_talk")
        echo_erle = [v for sc in echo_scenarios for v in scenario_erle.get(sc, [])
                     if np.isfinite(v)]
        val_erle = float(np.mean(echo_erle)) if echo_erle else 0.0
        add_scalar_with_help(writer, "val/erle_db", val_erle, epoch)
        # PESQ/STOI for double-talk
        for sc, values in scenario_pesq.items():
            add_scalar_with_help(writer, f"val/pesq_{sc}", np.mean(values), epoch)
        for sc, values in scenario_stoi.items():
            add_scalar_with_help(writer, f"val/stoi_{sc}", np.mean(values), epoch)
        # DNSMOS
        for sc, values in scenario_dnsmos_ovrl.items():
            add_scalar_with_help(writer, f"val/dnsmos_ovrl_{sc}", np.mean(values), epoch)
        all_dnsmos = [v for vals in scenario_dnsmos_ovrl.values() for v in vals]
        if all_dnsmos:
            add_scalar_with_help(writer, "val/dnsmos_ovrl", np.mean(all_dnsmos), epoch)
        # AECMOS
        for sc, values in scenario_echo_mos.items():
            add_scalar_with_help(writer, f"val/echo_mos_{sc}", np.mean(values), epoch)
        for sc, values in scenario_deg_mos.items():
            add_scalar_with_help(writer, f"val/deg_mos_{sc}", np.mean(values), epoch)
        all_echo_mos = [v for vals in scenario_echo_mos.values() for v in vals]
        if all_echo_mos:
            add_scalar_with_help(writer, "val/echo_mos", np.mean(all_echo_mos), epoch)
        all_deg_mos = [v for vals in scenario_deg_mos.values() for v in vals]
        if all_deg_mos:
            add_scalar_with_help(writer, "val/deg_mos", np.mean(all_deg_mos), epoch)

        # Build per-scenario metric summary
        metric_parts = [
            f"Epoch {epoch+1}: train_loss={epoch_losses['total']:.4f}, "
            f"val_loss={val_losses['total']:.4f}, "
            f"delay_acc={val_delay_acc:.1%}, erle={val_erle:+.1f}dB, "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        ]
        for sc, values in scenario_erle.items():
            part = f"  {sc}: erle={np.mean(values):+.1f}dB (n={len(values)})"
            if sc in scenario_pesq and scenario_pesq[sc]:
                part += f", pesq={np.mean(scenario_pesq[sc]):.2f}"
            if sc in scenario_stoi and scenario_stoi[sc]:
                part += f", stoi={np.mean(scenario_stoi[sc]):.3f}"
            if sc in scenario_echo_mos and scenario_echo_mos[sc]:
                part += f", echo_mos={np.mean(scenario_echo_mos[sc]):.2f}"
            if sc in scenario_deg_mos and scenario_deg_mos[sc]:
                part += f", deg_mos={np.mean(scenario_deg_mos[sc]):.2f}"
            if sc in scenario_dnsmos_ovrl and scenario_dnsmos_ovrl[sc]:
                part += f", dnsmos={np.mean(scenario_dnsmos_ovrl[sc]):.2f}"
            metric_parts.append(part)
        print("\n".join(metric_parts))

        # Log audio/spectrograms
        if val_sample_batches:
            log_audio_and_spectrograms(writer, model, val_sample_batches, epoch, cfg, device,
                                       act_capture=act_capture)
            print(f"  Logged {len(val_sample_batches)} audio sample(s) to TensorBoard")

        # Checkpointing
        if (epoch + 1) % cfg.training.checkpoint_every == 0:
            save_checkpoint(
                model, optimizer, epoch + 1,
                val_losses["total"],
                ckpt_dir / f"epoch_{epoch+1:04d}.pt",
                scheduler=scheduler,
            )
            manage_checkpoints(ckpt_dir, cfg.training.keep_checkpoints)

        # Gate: check delay accuracy and ERLE minimums after warmup
        delay_acc_ok = val_delay_acc >= cfg.training.delay_acc_min
        erle_ok = val_erle >= cfg.training.erle_min_db
        gate_epoch = max(cfg.training.warmup_epochs + 10, 20)  # grace period

        if (epoch + 1) >= gate_epoch and not delay_acc_ok:
            print(
                f"  FAIL: delay accuracy {val_delay_acc:.1%} < "
                f"{cfg.training.delay_acc_min:.0%} after {epoch+1} epochs. Stopping."
            )
            break

        if (epoch + 1) >= gate_epoch and not erle_ok:
            print(
                f"  FAIL: ERLE {val_erle:+.1f}dB < "
                f"{cfg.training.erle_min_db:+.1f}dB after {epoch+1} epochs. Stopping."
            )
            break

        if val_losses["total"] < best_val_loss - min_delta:
            best_val_loss = val_losses["total"]
            epochs_without_improvement = 0
            save_checkpoint(
                model, optimizer, epoch + 1,
                val_losses["total"],
                ckpt_dir / "best.pt",
                scheduler=scheduler,
            )
            print(f"  New best val loss: {best_val_loss:.4f} "
                  f"(delay_acc={val_delay_acc:.1%}, erle={val_erle:+.1f}dB)")
        else:
            epochs_without_improvement += 1
            print(
                f"  No improvement for {epochs_without_improvement}/{patience} epochs"
            )
            if patience > 0 and epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch+1}.")
                break

    writer.close()
    print("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DeepVQE AEC")
    parser.add_argument("--config", default="configs/default.yaml", help="Config file")
    parser.add_argument("--resume", default=None, help="Checkpoint to resume from")
    parser.add_argument("--dummy", action="store_true", help="Use dummy dataset for testing")
    parser.add_argument("--overfit-real", action="store_true",
                        help="Use FixedSynthDataset (real audio, controlled delays)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train(cfg, resume=args.resume, dummy=args.dummy,
          overfit_real=args.overfit_real)
