"""DeepVQE AEC training script.

Features:
- AdamW optimizer with linear warmup + ReduceLROnPlateau
- Mixed precision (BF16 autocast, TF32 for remaining FP32 ops)
- Gradient accumulation for large effective batch sizes
- Gradient clipping
- TensorBoard logging (loss, lr, grad norms, audio, spectrograms, delay heatmaps)
- Checkpointing (last N + best)
"""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data.dataset import AECDataset, DummyAECDataset
from src.config import load_config
from src.losses import DeepVQELoss
from src.model import DeepVQEAEC
from src.stft import istft


def collate_fn(batch):
    """Custom collate that handles metadata dicts."""
    mic_stft = torch.stack([b["mic_stft"] for b in batch])
    ref_stft = torch.stack([b["ref_stft"] for b in batch])
    clean_stft = torch.stack([b["clean_stft"] for b in batch])
    mic_wav = torch.stack([b["mic_wav"] for b in batch])
    clean_wav = torch.stack([b["clean_wav"] for b in batch])
    delay_samples = torch.tensor([b["delay_samples"] for b in batch], dtype=torch.long)
    metadata = [b["metadata"] for b in batch]
    return {
        "mic_stft": mic_stft,
        "ref_stft": ref_stft,
        "clean_stft": clean_stft,
        "mic_wav": mic_wav,
        "clean_wav": clean_wav,
        "delay_samples": delay_samples,
        "metadata": metadata,
    }


def delay_samples_to_frame(delay_samples, hop_length, dmax):
    """Convert delay in samples to AlignBlock frame index.

    AlignBlock unfold convention: index d corresponds to ref[t - (dmax-1) + d].
    So d = dmax-1 means no delay, d = 0 means max delay.
    target_frame = dmax - 1 - round(delay_samples / hop_length)
    """
    delay_frames = torch.round(delay_samples.float() / hop_length).long()
    target_frames = (dmax - 1) - delay_frames
    return target_frames.clamp(0, dmax - 1)


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


def compute_erle(mic_wav, enhanced_wav, clean_wav):
    """Compute Echo Return Loss Enhancement in dB.

    ERLE = 10 * log10(||mic - clean||^2 / ||enhanced - clean||^2)
    Positive means echo was reduced.
    """
    echo_plus_noise = mic_wav - clean_wav
    residual = enhanced_wav - clean_wav
    echo_power = (echo_plus_noise ** 2).sum(dim=-1)
    residual_power = (residual ** 2).sum(dim=-1)
    erle = 10 * torch.log10(echo_power / (residual_power + 1e-10) + 1e-10)
    return erle.mean()


def get_warmup_scheduler(optimizer, cfg, steps_per_epoch):
    """Linear warmup scheduler (per-step) for the first few epochs."""
    warmup_steps = cfg.training.warmup_epochs * steps_per_epoch

    def lr_lambda(step):
        if warmup_steps <= 0:
            return 1.0
        return min(1.0, step / warmup_steps)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def get_plateau_scheduler(optimizer, cfg):
    """ReduceLROnPlateau (per-epoch) after warmup completes."""
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=cfg.training.lr_factor,
        patience=cfg.training.lr_patience,
        min_lr=cfg.training.lr_min,
    )


def log_health(writer, model, global_step):
    """Log per-module gradient norms, weight stats, and AlignBlock attention entropy."""
    for name, param in model.named_parameters():
        if param.grad is not None:
            writer.add_scalar(f"grad_norm/{name}", param.grad.data.norm(2).item(), global_step)
        writer.add_scalar(f"weight_mean/{name}", param.data.mean().item(), global_step)
        writer.add_scalar(f"weight_std/{name}", param.data.std().item(), global_step)
        absmax = param.data.abs().max().item()
        writer.add_scalar(f"weight_absmax/{name}", absmax, global_step)


def _unwrap(model):
    """Get the raw model, unwrapping torch.compile's OptimizedModule."""
    return getattr(model, "_orig_mod", model)


def save_checkpoint(model, optimizer, schedulers, epoch, loss, path):
    sched_states = {k: s.state_dict() for k, s in schedulers.items()}
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": _unwrap(model).state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_states": sched_states,
            "loss": loss,
        },
        path,
    )


def load_checkpoint(path, model, optimizer=None, schedulers=None):
    ckpt = torch.load(path, weights_only=False)
    state = ckpt["model_state_dict"]
    # Strip _orig_mod. prefix for backward compat with old checkpoints
    state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    _unwrap(model).load_state_dict(state)
    if optimizer and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if schedulers and "scheduler_states" in ckpt:
        for k, s in schedulers.items():
            if k in ckpt["scheduler_states"]:
                s.load_state_dict(ckpt["scheduler_states"][k])
    return ckpt["epoch"]


def manage_checkpoints(ckpt_dir, keep_n):
    """Keep only the last N checkpoints (excluding best)."""
    ckpts = sorted(ckpt_dir.glob("epoch_*.pt"), key=lambda p: p.stat().st_mtime)
    while len(ckpts) > keep_n:
        ckpts.pop(0).unlink()


def log_audio_and_spectrograms(writer, model, val_batch, epoch, cfg, device):
    """Log audio samples and spectrograms to TensorBoard."""
    model.eval()
    with torch.no_grad():
        mic_stft = val_batch["mic_stft"][:1].to(device)
        ref_stft = val_batch["ref_stft"][:1].to(device)
        clean_stft = val_batch["clean_stft"][:1].to(device)

        enhanced, delay_dist = model(mic_stft, ref_stft, return_delay=True)
        length = val_batch["clean_wav"].shape[-1]

        mic_wav = istft(mic_stft, cfg.audio.n_fft, cfg.audio.hop_length, length=length)
        enh_wav = istft(enhanced, cfg.audio.n_fft, cfg.audio.hop_length, length=length)
        clean_wav = val_batch["clean_wav"][:1].to(device)

        sr = cfg.audio.sample_rate
        writer.add_audio("audio/mic", mic_wav[0].cpu(), epoch, sample_rate=sr)
        writer.add_audio("audio/enhanced", enh_wav[0].cpu(), epoch, sample_rate=sr)
        writer.add_audio("audio/clean", clean_wav[0].cpu(), epoch, sample_rate=sr)

        # Log delay distribution heatmap
        if delay_dist is not None:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(1, 1, figsize=(10, 4))
            ax.imshow(
                delay_dist[0].cpu().numpy().T,
                aspect="auto",
                origin="lower",
                cmap="viridis",
            )
            ax.set_xlabel("Frame")
            ax.set_ylabel("Delay (frames)")
            ax.set_title("Delay Distribution")
            writer.add_figure("delay_distribution", fig, epoch)
            plt.close(fig)
    model.train()


def train(cfg, resume=None, dummy=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    print(f"Device: {device}")

    # Create directories
    ckpt_dir = Path(cfg.paths.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(cfg.paths.log_dir)

    writer = SummaryWriter(log_dir)

    # Model
    model = DeepVQEAEC.from_config(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    if device.type == "cuda":
        model = torch.compile(model)
        print("Model compiled with torch.compile")

    # Loss
    criterion = DeepVQELoss.from_config(cfg).to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
    )

    # Dataset
    if dummy:
        train_ds = DummyAECDataset(
            length=cfg.data.num_train,
            target_len=int(cfg.training.clip_length_sec * cfg.audio.sample_rate),
            n_fft=cfg.audio.n_fft,
            hop_length=cfg.audio.hop_length,
            delay_samples=160,
        )
        val_ds = DummyAECDataset(
            length=cfg.data.num_val,
            target_len=int(cfg.training.clip_length_sec * cfg.audio.sample_rate),
            n_fft=cfg.audio.n_fft,
            hop_length=cfg.audio.hop_length,
            delay_samples=160,
        )
    else:
        train_ds = AECDataset(cfg, split="train")
        val_ds = AECDataset(cfg, split="val")

    num_workers = getattr(cfg.training, "num_workers", 4)
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
    warmup_scheduler = get_warmup_scheduler(optimizer, cfg, steps_per_epoch)
    plateau_scheduler = get_plateau_scheduler(optimizer, cfg)

    # AMP — BF16 on CUDA (no GradScaler needed), disabled on CPU
    use_amp = cfg.training.amp and device.type == "cuda"
    autocast_ctx = lambda: torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=use_amp,
    )

    schedulers = {"warmup": warmup_scheduler, "plateau": plateau_scheduler}

    # Resume
    start_epoch = 0
    best_val_loss = float("inf")
    if resume:
        start_epoch = load_checkpoint(resume, model, optimizer, schedulers)
        print(f"Resumed from epoch {start_epoch}")

    global_step = start_epoch * steps_per_epoch
    warmup_done = start_epoch >= cfg.training.warmup_epochs
    accum_steps = cfg.training.grad_accum_steps
    patience = getattr(cfg.training, "early_stop_patience", 0)
    min_delta = getattr(cfg.training, "early_stop_min_delta", 1e-3)
    epochs_without_improvement = 0

    for epoch in range(start_epoch, cfg.training.epochs):
        model.train()

        # Anneal AlignBlock temperature: linear decay from start to end
        t_start = cfg.model.align_temp_start
        t_end = cfg.model.align_temp_end
        t_epochs = cfg.model.align_temp_epochs
        if t_epochs > 0 and epoch < t_epochs:
            temperature = t_start + (t_end - t_start) * epoch / t_epochs
        else:
            temperature = t_end
        align = _unwrap(model).align
        align.temperature = temperature
        writer.add_scalar("train/temperature", temperature, epoch)

        epoch_losses = {
            "total": 0, "plcmse": 0, "mag_l1": 0, "time_l1": 0,
            "sisdr": 0, "delay": 0, "entropy": 0,
        }
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
                enhanced, delay_dist = model(mic_stft, ref_stft, return_delay=True)
                loss, components = criterion(enhanced, clean_stft, clean_wav)

                # Delay supervision loss
                delay_loss, delay_acc = compute_delay_loss(
                    delay_dist, delay_samp, cfg.audio.hop_length, cfg.model.dmax,
                )
                components["delay"] = delay_loss
                loss = loss + cfg.loss.delay_weight * delay_loss

                # Entropy penalty on delay attention
                if cfg.loss.entropy_weight > 0:
                    entropy = -(delay_dist * torch.log(delay_dist + 1e-10)).sum(dim=-1).mean()
                    components["entropy"] = entropy
                    loss = loss + cfg.loss.entropy_weight * entropy
                else:
                    components["entropy"] = torch.tensor(0.0, device=device)

                loss = loss / accum_steps

            loss.backward()

            if (batch_idx + 1) % accum_steps == 0:
                raw_gn = nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)
                gn = raw_gn.item()
                optimizer.step()
                if not warmup_done:
                    warmup_scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # Log per-step
                cur_lr = optimizer.param_groups[0]["lr"]
                writer.add_scalar("train/loss", components["total"].item(), global_step)
                writer.add_scalar("train/plcmse", components["plcmse"].item(), global_step)
                writer.add_scalar("train/mag_l1", components["mag_l1"].item(), global_step)
                writer.add_scalar("train/time_l1", components["time_l1"].item(), global_step)
                writer.add_scalar("train/sisdr", components["sisdr"].item(), global_step)
                writer.add_scalar("train/delay_loss", components["delay"].item(), global_step)
                writer.add_scalar("train/delay_acc", delay_acc.item(), global_step)
                writer.add_scalar("train/entropy", components["entropy"].item(), global_step)
                writer.add_scalar("train/lr", cur_lr, global_step)
                writer.add_scalar("train/grad_norm", gn, global_step)

            for k in epoch_losses:
                epoch_losses[k] += components[k].item()
            epoch_delay_acc += delay_acc.item()
            n_batches += 1

            pbar.set_postfix(
                loss=f"{components['total'].item():.4f}",
                dacc=f"{delay_acc.item():.0%}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )

        # Epoch averages
        for k in epoch_losses:
            epoch_losses[k] /= max(n_batches, 1)
            writer.add_scalar(f"train_epoch/{k}", epoch_losses[k], epoch)
        epoch_delay_acc /= max(n_batches, 1)
        writer.add_scalar("train_epoch/delay_acc", epoch_delay_acc, epoch)

        # Validation
        model.eval()
        val_losses = {
            "total": 0, "plcmse": 0, "mag_l1": 0, "time_l1": 0,
            "sisdr": 0, "delay": 0, "entropy": 0,
        }
        val_delay_acc = 0
        val_erle = 0
        n_val = 0
        val_sample_batch = None

        with torch.no_grad(), autocast_ctx():
            for batch in val_loader:
                mic_stft = batch["mic_stft"].to(device, non_blocking=True)
                ref_stft = batch["ref_stft"].to(device, non_blocking=True)
                clean_stft = batch["clean_stft"].to(device, non_blocking=True)
                clean_wav = batch["clean_wav"].to(device, non_blocking=True)
                delay_samp = batch["delay_samples"].to(device, non_blocking=True)

                enhanced, delay_dist = model(mic_stft, ref_stft, return_delay=True)
                _, components = criterion(enhanced, clean_stft, clean_wav)

                # Delay loss + accuracy
                delay_loss, delay_acc = compute_delay_loss(
                    delay_dist, delay_samp, cfg.audio.hop_length, cfg.model.dmax,
                )
                components["delay"] = delay_loss

                entropy = -(delay_dist * torch.log(delay_dist + 1e-10)).sum(dim=-1).mean()
                components["entropy"] = entropy

                # ERLE
                length = clean_wav.shape[-1]
                mic_wav = batch["mic_wav"].to(device, non_blocking=True)
                enh_wav = istft(enhanced, cfg.audio.n_fft, cfg.audio.hop_length, length=length)
                erle = compute_erle(mic_wav, enh_wav, clean_wav)

                for k in val_losses:
                    val_losses[k] += components[k].item()
                val_delay_acc += delay_acc.item()
                val_erle += erle.item()
                n_val += 1

                if val_sample_batch is None:
                    val_sample_batch = batch

        for k in val_losses:
            val_losses[k] /= max(n_val, 1)
            writer.add_scalar(f"val/{k}", val_losses[k], epoch)
        val_delay_acc /= max(n_val, 1)
        val_erle /= max(n_val, 1)
        writer.add_scalar("val/delay_acc", val_delay_acc, epoch)
        writer.add_scalar("val/erle_db", val_erle, epoch)

        # Step plateau scheduler on val loss (only after warmup)
        if warmup_done:
            plateau_scheduler.step(val_losses["total"])
        if not warmup_done and (epoch + 1) >= cfg.training.warmup_epochs:
            warmup_done = True

        print(
            f"Epoch {epoch+1}: train_loss={epoch_losses['total']:.4f}, "
            f"val_loss={val_losses['total']:.4f}, "
            f"delay_acc={val_delay_acc:.1%}, erle={val_erle:+.1f}dB, "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )

        # Log audio/spectrograms
        if val_sample_batch:
            log_audio_and_spectrograms(writer, model, val_sample_batch, epoch, cfg, device)

        # Checkpointing
        if (epoch + 1) % cfg.training.checkpoint_every == 0:
            save_checkpoint(
                model, optimizer, schedulers, epoch + 1,
                val_losses["total"],
                ckpt_dir / f"epoch_{epoch+1:04d}.pt",
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
                model, optimizer, schedulers, epoch + 1,
                val_losses["total"],
                ckpt_dir / "best.pt",
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
    args = parser.parse_args()

    cfg = load_config(args.config)
    train(cfg, resume=args.resume, dummy=args.dummy)
