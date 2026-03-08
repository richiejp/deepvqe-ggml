"""Visualization utilities for DeepVQE training and analysis.

Pure functions that take tensors and return matplotlib figures,
plus hook registration and TensorBoard logging helpers.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Spectrogram comparison
# ---------------------------------------------------------------------------

def plot_spectrogram_comparison(mic_stft, enh_stft, clean_stft, sr, hop_length):
    """2x2 grid: mic | enhanced | clean | residual (enhanced - clean).

    Args:
        mic_stft: (F, T, 2) or (B, F, T, 2) — microphone STFT
        enh_stft: same shape — enhanced STFT
        clean_stft: same shape — clean STFT
        sr: sample rate
        hop_length: hop size

    Returns:
        matplotlib Figure
    """
    def _mag_db(x):
        if x.dim() == 4:
            x = x[0]
        mag = torch.sqrt(x[..., 0] ** 2 + x[..., 1] ** 2 + 1e-12)
        return 20 * torch.log10(mag + 1e-12).cpu().numpy()

    mic_db = _mag_db(mic_stft)
    enh_db = _mag_db(enh_stft)
    clean_db = _mag_db(clean_stft)
    res_db = enh_db - clean_db

    vmin = min(mic_db.min(), enh_db.min(), clean_db.min())
    vmax = max(mic_db.max(), enh_db.max(), clean_db.max())

    F, T = mic_db.shape
    t_axis = np.arange(T) * hop_length / sr
    f_axis = np.arange(F) * sr / ((F - 1) * 2)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    titles = ["Mic", "Enhanced", "Clean", "Residual (enh - clean)"]
    data = [mic_db, enh_db, clean_db, res_db]

    for ax, title, d in zip(axes.flat, titles, data):
        if title == "Residual (enh - clean)":
            im = ax.pcolormesh(t_axis, f_axis / 1000, d, cmap="RdBu_r", shading="auto")
        else:
            im = ax.pcolormesh(t_axis, f_axis / 1000, d, vmin=vmin, vmax=vmax,
                               cmap="magma", shading="auto")
        ax.set_title(title)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Freq (kHz)")
        fig.colorbar(im, ax=ax, label="dB")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Delay distribution with ground truth
# ---------------------------------------------------------------------------

def plot_delay_with_gt(delay_dist, gt_delay_samples, hop_length, dmax):
    """Delay heatmap (T x dmax) with ground-truth overlay.

    Args:
        delay_dist: (T, dmax) or (B, T, dmax) — predicted delay distribution
        gt_delay_samples: int or scalar tensor — ground truth delay in samples
        hop_length: int
        dmax: int

    Returns:
        matplotlib Figure
    """
    if delay_dist.dim() == 3:
        delay_dist = delay_dist[0]
    dd = delay_dist.detach().cpu().numpy()  # (T, dmax)
    T, D = dd.shape

    if isinstance(gt_delay_samples, torch.Tensor):
        gt_delay_samples = gt_delay_samples.item()

    # GT frame index (same convention as compute_delay_loss)
    gt_frame = dmax - 1 - round(gt_delay_samples / hop_length)
    gt_frame = max(0, min(dmax - 1, gt_frame))

    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    im = ax.imshow(dd.T, aspect="auto", origin="lower", cmap="viridis",
                   interpolation="nearest")
    fig.colorbar(im, ax=ax, label="Probability")

    # GT line
    ax.axhline(gt_frame, color="red", linestyle="--", linewidth=1.5,
               label=f"GT delay={gt_delay_samples} samp (frame {gt_frame})")

    # Per-frame correctness dots
    peak_frames = dd.argmax(axis=1)
    for t in range(T):
        correct = abs(int(peak_frames[t]) - gt_frame) <= 1
        color = "lime" if correct else "red"
        ax.plot(t, peak_frames[t], "o", color=color, markersize=2)

    ax.set_xlabel("Frame")
    ax.set_ylabel("Delay index")
    ax.set_title("Delay Distribution")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# CCM mask analysis
# ---------------------------------------------------------------------------

def plot_ccm_mask(mask_27ch, mic_stft):
    """Decompose 27ch mask and compare to ideal ratio mask.

    Args:
        mask_27ch: (27, T, F) or (B, 27, T, F) — raw mask from decoder
        mic_stft: (F, T, 2) or (B, F, T, 2) — microphone STFT

    Returns:
        matplotlib Figure
    """
    if mask_27ch.dim() == 4:
        mask_27ch = mask_27ch[0]
    if mic_stft.dim() == 4:
        mic_stft = mic_stft[0]

    mask = mask_27ch.detach().cpu().float()  # (27, T, F)

    # Cube-root basis vectors
    v_real = torch.tensor([1.0, -0.5, -0.5])
    v_imag = torch.tensor([0.0, np.sqrt(3) / 2, -np.sqrt(3) / 2])

    # Reshape: (3, 9, T, F)
    m = mask.view(3, 9, mask.shape[1], mask.shape[2])
    H_real = (v_real[:, None, None, None] * m).sum(dim=0)  # (9, T, F)
    H_imag = (v_imag[:, None, None, None] * m).sum(dim=0)  # (9, T, F)

    # Mean mask magnitude over the 9 kernel elements
    H_mag = torch.sqrt(H_real ** 2 + H_imag ** 2 + 1e-12).mean(dim=0)  # (T, F)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    im0 = axes[0].imshow(H_mag.numpy().T, aspect="auto", origin="lower",
                         cmap="viridis", interpolation="nearest")
    axes[0].set_title("Mean Mask Magnitude |H|")
    axes[0].set_xlabel("Frame")
    axes[0].set_ylabel("Freq bin")
    fig.colorbar(im0, ax=axes[0])

    # Ideal ratio mask: |clean|/|mic| — we don't have clean here, so show |mic| magnitude
    mic_mag = torch.sqrt(mic_stft[..., 0] ** 2 + mic_stft[..., 1] ** 2 + 1e-12)
    mic_mag_db = 20 * torch.log10(mic_mag.cpu() + 1e-12)
    im1 = axes[1].imshow(mic_mag_db.numpy(), aspect="auto", origin="lower",
                         cmap="magma", interpolation="nearest")
    axes[1].set_title("Mic Magnitude (dB) — for reference")
    axes[1].set_xlabel("Frame")
    axes[1].set_ylabel("Freq bin")
    fig.colorbar(im1, ax=axes[1], label="dB")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Encoder activation heatmaps
# ---------------------------------------------------------------------------

def plot_encoder_activations(activation_dict):
    """Channel-mean heatmaps for each encoder stage.

    Args:
        activation_dict: dict mapping layer names to tensors (B, C, T, F)

    Returns:
        matplotlib Figure
    """
    enc_keys = [k for k in activation_dict
                if k.startswith(("mic_enc", "far_enc"))]
    if not enc_keys:
        enc_keys = list(activation_dict.keys())[:7]

    n = len(enc_keys)
    if n == 0:
        fig, ax = plt.subplots(1, 1, figsize=(6, 3))
        ax.text(0.5, 0.5, "No encoder activations captured", ha="center")
        return fig

    cols = min(n, 4)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flat

    for i, key in enumerate(enc_keys):
        act = activation_dict[key]
        if act.dim() == 4:
            act = act[0]
        # Channel mean: (T, F)
        hm = act.float().mean(dim=0).cpu().numpy()
        im = axes[i].imshow(hm.T, aspect="auto", origin="lower",
                            cmap="viridis", interpolation="nearest")
        axes[i].set_title(f"{key} ({act.shape[0]}ch)")
        axes[i].set_xlabel("Frame")
        axes[i].set_ylabel("Freq bin")
        fig.colorbar(im, ax=axes[i])

    # Hide unused axes
    for j in range(i + 1, len(list(axes))):
        axes[j].set_visible(False)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Activation statistics
# ---------------------------------------------------------------------------

def plot_activation_stats(activation_dict):
    """Bar charts: mean, std, dead fraction (==0), max per layer.

    Args:
        activation_dict: dict mapping layer names to tensors

    Returns:
        matplotlib Figure
    """
    names = list(activation_dict.keys())
    if not names:
        fig, ax = plt.subplots(1, 1, figsize=(6, 3))
        ax.text(0.5, 0.5, "No activations captured", ha="center")
        return fig

    means, stds, deads, maxes = [], [], [], []
    for k in names:
        act = activation_dict[k].float()
        means.append(act.mean().item())
        stds.append(act.std().item())
        deads.append((act == 0).float().mean().item())
        maxes.append(act.abs().max().item())

    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    titles = ["Mean", "Std", "Dead Fraction", "Max |act|"]
    data = [means, stds, deads, maxes]
    colors = ["steelblue", "darkorange", "crimson", "seagreen"]

    for ax, title, vals, c in zip(axes, titles, data, colors):
        ax.bar(x, vals, color=c)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
        ax.set_title(title)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Hook registration
# ---------------------------------------------------------------------------

def register_hooks(model):
    """Register forward hooks on key layers to capture activations.

    Args:
        model: DeepVQEAEC instance (unwrapped)

    Returns:
        (activation_store, hook_handles)
    """
    activation_store = {}
    hook_handles = []

    target_names = [
        "fe_mic", "fe_ref",
        "mic_enc1", "mic_enc2", "mic_enc3", "mic_enc4", "mic_enc5",
        "far_enc1", "far_enc2",
        "align", "bottleneck",
        "dec5", "dec4", "dec3", "dec2", "dec1",
        "ccm",
    ]

    for name in target_names:
        module = getattr(model, name, None)
        if module is None:
            continue

        def _make_hook(n):
            def hook_fn(module, input, output):
                # AlignBlock returns tuple when return_delay=True
                if isinstance(output, tuple):
                    activation_store[n] = output[0].detach().cpu()
                else:
                    activation_store[n] = output.detach().cpu()
            return hook_fn

        h = module.register_forward_hook(_make_hook(name))
        hook_handles.append(h)

    return activation_store, hook_handles


def remove_hooks(hook_handles):
    """Remove all registered hooks."""
    for h in hook_handles:
        h.remove()


# ---------------------------------------------------------------------------
# TensorBoard logging helpers
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Metric descriptions for TensorBoard help tooltips
# ---------------------------------------------------------------------------
# Shown as an info icon next to each chart. Only sent on first add_scalar call.

METRIC_HELP = {
    # -- Per-step training metrics --
    "train/loss": (
        "**Weighted total loss** = plcmse + 0.5*mag_l1 + 0.5*time_l1 "
        "+ delay_weight*delay + entropy_weight*entropy.  \n"
        "Should decrease steadily over training. Typical progression:\n"
        "- Early: 1.0-2.0 (random model)\n"
        "- Mid: 0.1-0.3 (learning spectral structure)\n"
        "- Converged: 0.05-0.10\n\n"
        "**Watch for:** Sudden spikes (bad batch or LR too high), "
        "flat plateau (LR too low or stuck in local minimum), "
        "val loss diverging from train (overfitting)."
    ),
    "train/plcmse": (
        "**Power-Law Compressed MSE** — primary reconstruction loss.  \n"
        "Computes MSE on |X|^0.5 * e^(j*angle(X)), which emphasizes "
        "quiet spectral regions over loud ones. Weight: 1.0.  \n\n"
        "**Healthy range:** 0.01-0.05 when converged.  \n"
        "**Watch for:** If this stalls while other losses decrease, the model "
        "may be optimizing delay/entropy at the expense of audio quality."
    ),
    "train/mag_l1": (
        "**Magnitude L1** — L1 distance between predicted and target "
        "magnitude spectrograms. Weight: 0.5.  \n"
        "Encourages overall spectral shape accuracy without phase sensitivity.  \n\n"
        "**Healthy range:** 0.01-0.05 when converged.  \n"
        "**Watch for:** Should track plcmse trends. If mag_l1 is low but "
        "audio sounds bad, phase reconstruction may be the bottleneck."
    ),
    "train/time_l1": (
        "**Time-domain L1** — mean absolute error on the waveform. Weight: 0.5.  \n"
        "Provides a direct signal-level loss that complements spectral losses.  \n\n"
        "**Healthy range:** 0.005-0.02 when converged.  \n"
        "**Watch for:** If time_l1 is high but spectral losses are low, "
        "the model may have phase issues (spectral magnitude is right but "
        "phase is wrong, causing waveform mismatch)."
    ),
    "train/sisdr": (
        "**Negative SI-SDR** (scale-invariant signal-to-distortion ratio).  \n"
        "Currently disabled (weight=0.0). When enabled, lower = better "
        "(it's negated so it can be minimized).  \n"
        "A value of -15 means 15 dB SI-SDR, which is good.  \n\n"
        "**Note:** SI-SDR can dominate training and cause instability with "
        "AEC tasks, which is why it's disabled in favor of spectral losses."
    ),
    "train/delay_loss": (
        "**Delay cross-entropy** — supervision for the AlignBlock.  \n"
        "Measures how well the attention distribution peaks at the correct "
        "delay frame. The ground-truth delay (in samples) is converted to "
        "a frame index: `target = dmax-1 - round(delay_samples/hop_length)`.  \n\n"
        "**Healthy range:** <0.5 after warmup, <0.1 when well-trained.  \n"
        "**Watch for:** If this stays >1.0, the AlignBlock is not learning "
        "to estimate delay — check data pipeline delay labels, temperature "
        "annealing schedule, or increase delay_weight."
    ),
    "train/delay_acc": (
        "**Delay accuracy** — fraction of frames where the predicted delay "
        "peak is within +/-1 frame of ground truth.  \n\n"
        "**Target:** >95% for reliable echo cancellation.  \n"
        "**Progression:**\n"
        "- Random: ~3% (1/dmax chance)\n"
        "- After a few epochs: 50-80%\n"
        "- Converged: 95-99%\n\n"
        "**Watch for:** Accuracy jumping between high and low values may "
        "indicate the temperature is too high (attention too diffuse). "
        "If accuracy is high but ERLE is low, the alignment is correct "
        "but the mask/decoder isn't using it effectively."
    ),
    "train/entropy": (
        "**Attention entropy** in the AlignBlock delay distribution.  \n"
        "H = -sum(p * log(p)). Measures how spread out the attention is.  \n\n"
        "**Range:** 0 (delta spike, perfect certainty) to ln(64)=4.16 "
        "(uniform over all 64 delay taps).  \n"
        "**Target:** <1.0 for sharp, confident delay estimates.  \n"
        "**Progression:**\n"
        "- Start: ~4.0 (near-uniform, no delay knowledge)\n"
        "- Mid: 2.5-3.0 (learning but still diffuse)\n"
        "- Converged: 0.5-1.5 (focused on correct delay)\n\n"
        "**Watch for:** Entropy not decreasing despite good delay_acc means "
        "attention has multiple modes — the model hedges its bets. "
        "Lower the temperature faster or increase entropy_weight."
    ),
    "train/mask_reg": (
        "**Mask magnitude regularizer** — MSE between mean CCM mask magnitude "
        "and 1.0 (identity/passthrough).  \n"
        "Prevents the mask from collapsing to zero (over-suppression) or "
        "exploding (amplification).  \n\n"
        "**Healthy range:** 0.1-0.5 early, <0.1 when converged.  \n"
        "**Watch for:** If this stays high (>1.0), the model is struggling to "
        "produce unit-magnitude masks. If it drops to 0 immediately, the "
        "weight may be too high (model just outputs identity, ignoring other losses)."
    ),
    "train/lr": (
        "**Learning rate** from the optimizer.  \n"
        "Uses linear warmup for the first N epochs, then ReduceLROnPlateau "
        "(reduces by `factor` when val loss plateaus for `patience` epochs).  \n\n"
        "**Watch for:** If LR drops to min_lr early, training may have "
        "plateaued prematurely. If LR never decreases after warmup, "
        "val loss is consistently improving (good sign)."
    ),
    "train/grad_norm": (
        "**Global L2 gradient norm** (after clipping to max_norm).  \n"
        "Shows the magnitude of parameter updates.  \n\n"
        "**Healthy range:** 0.5-5.0 for this model.  \n"
        "**Watch for:**\n"
        "- Consistently at clip value: gradients are being clipped every step, "
        "consider raising clip threshold or lowering LR\n"
        "- Sudden spikes: bad batch or numerical instability\n"
        "- Near zero: vanishing gradients, model may be stuck\n"
        "- Gradually increasing: potential training instability"
    ),
    "train/temperature": (
        "**AlignBlock softmax temperature** — controls sharpness of the "
        "delay attention distribution.  \n"
        "Annealed linearly from `align_temp_start` to `align_temp_end` "
        "over `align_temp_epochs`.  \n\n"
        "Higher temperature = softer/more uniform attention (exploration).  \n"
        "Lower temperature = sharper/peakier attention (exploitation).  \n\n"
        "**Watch for:** If delay_acc drops when temperature decreases, "
        "the model learned a diffuse representation that doesn't survive "
        "sharpening — may need slower annealing or more delay supervision."
    ),

    # -- Per-epoch training averages --
    "train_epoch/total": "Epoch-averaged total loss. Smoother view of `train/loss`.",
    "train_epoch/plcmse": "Epoch-averaged power-law compressed MSE.",
    "train_epoch/mag_l1": "Epoch-averaged magnitude L1.",
    "train_epoch/time_l1": "Epoch-averaged time-domain L1.",
    "train_epoch/sisdr": "Epoch-averaged SI-SDR loss (currently disabled, weight=0).",
    "train_epoch/delay": "Epoch-averaged delay cross-entropy.",
    "train_epoch/entropy": "Epoch-averaged attention entropy.",
    "train_epoch/delay_acc": (
        "Epoch-averaged delay accuracy. Compare with `val/delay_acc` to check "
        "for overfitting on delay estimation."
    ),

    # -- Validation metrics --
    "val/total": (
        "**Validation total loss.** Compare with `train_epoch/total`.  \n"
        "If val >> train, the model is overfitting. If both are high, "
        "the model is underfitting (needs more capacity or longer training)."
    ),
    "val/plcmse": "Validation power-law compressed MSE.",
    "val/mag_l1": "Validation magnitude L1.",
    "val/time_l1": "Validation time-domain L1.",
    "val/sisdr": "Validation SI-SDR loss.",
    "val/delay": "Validation delay cross-entropy.",
    "val/entropy": (
        "Validation attention entropy. Should be close to train value. "
        "If much higher, the model overfits delay patterns in training data."
    ),
    "val/delay_acc": (
        "**Validation delay accuracy.**  \n"
        "Should track `train_epoch/delay_acc` closely. A large gap (train >> val) "
        "means the model memorized training delay patterns rather than learning "
        "general echo delay estimation."
    ),
    "val/erle_db": (
        "**Echo Return Loss Enhancement** in dB.  \n"
        "ERLE = 10*log10(|mic-clean|^2 / |enhanced-clean|^2).  \n"
        "Measures how much echo+noise power the model removes.  \n\n"
        "**Interpretation:**\n"
        "- 0 dB: no improvement over input\n"
        "- 5 dB: modest echo reduction\n"
        "- 10 dB: good echo cancellation\n"
        "- 15+ dB: strong echo cancellation\n"
        "- Negative: model is making things *worse*\n\n"
        "**This is the most important quality metric.** If loss decreases "
        "but ERLE doesn't improve, the model is optimizing the wrong thing.  \n"
        "**Watch for:** ERLE plateau while loss still drops — may indicate "
        "the loss function doesn't correlate well with perceptual quality."
    ),

    # -- Per-layer gradient norms --
    "grad_norm/fe_mic": "Grad norm for mic front-end (real+imag to 2ch). Should be small and stable.",
    "grad_norm/fe_ref": "Grad norm for far-end front-end. Similar magnitude to fe_mic.",
    "grad_norm/mic_enc1": "Grad norm for mic encoder stage 1 (2->64ch).",
    "grad_norm/mic_enc2": "Grad norm for mic encoder stage 2 (64->128ch). Feeds into AlignBlock.",
    "grad_norm/mic_enc3": "Grad norm for mic encoder stage 3 (256->128ch, post-alignment concat).",
    "grad_norm/mic_enc4": "Grad norm for mic encoder stage 4 (128->128ch).",
    "grad_norm/mic_enc5": "Grad norm for mic encoder stage 5 (128->128ch).",
    "grad_norm/far_enc1": "Grad norm for far-end encoder stage 1 (2->32ch).",
    "grad_norm/far_enc2": "Grad norm for far-end encoder stage 2 (32->128ch). Feeds into AlignBlock.",
    "grad_norm/align": (
        "Grad norm for AlignBlock. If much larger than encoder norms, "
        "delay_weight may be too high. If near zero, delay supervision "
        "isn't reaching the alignment layer."
    ),
    "grad_norm/bottleneck": (
        "Grad norm for GRU bottleneck. Should be moderate. "
        "Very large norms here can indicate GRU instability — "
        "consider gradient clipping or reducing bottleneck size."
    ),
    "grad_norm/dec5": "Grad norm for decoder stage 5 (deepest).",
    "grad_norm/dec4": "Grad norm for decoder stage 4.",
    "grad_norm/dec3": "Grad norm for decoder stage 3.",
    "grad_norm/dec2": "Grad norm for decoder stage 2.",
    "grad_norm/dec1": "Grad norm for decoder stage 1 (outputs 27ch CCM mask).",
    "grad_norm/ccm": "Grad norm for CCM (complex convolving mask) output layer.",

    # -- Loss ratios --
    "loss_ratio/plcmse": (
        "Fraction of total loss from plcmse. Should be the dominant term "
        "(0.4-0.7) since it has weight=1.0."
    ),
    "loss_ratio/mag_l1": "Fraction of total loss from mag_l1. Typically 0.1-0.3.",
    "loss_ratio/time_l1": "Fraction of total loss from time_l1. Typically 0.05-0.2.",
    "loss_ratio/sisdr": "Fraction from SI-SDR (0 when disabled).",
    "loss_ratio/delay": (
        "Fraction of total loss from delay supervision. If this dominates "
        "(>0.5), the model focuses on delay at the expense of audio quality. "
        "Consider lowering delay_weight."
    ),
    "loss_ratio/entropy": (
        "Fraction from entropy penalty. Should be small (<0.1). "
        "If it dominates, lower entropy_weight."
    ),
}

# Tags for which we've already sent the description this run.
_described_tags: set[str] = set()


def add_scalar_with_help(writer, tag, value, step):
    """Like writer.add_scalar but logs description via add_text on first call."""
    if tag not in _described_tags and tag in METRIC_HELP:
        writer.add_text(f"metric_help/{tag}", METRIC_HELP[tag], step)
        _described_tags.add(tag)
    writer.add_scalar(tag, value, step)


def log_weight_histograms(writer, model, epoch):
    """Log weight and gradient histograms to TensorBoard.

    Args:
        writer: SummaryWriter
        model: nn.Module
        epoch: current epoch
    """
    for name, param in model.named_parameters():
        writer.add_histogram(f"weights/{name}", param.detach().cpu(), epoch)
        if param.grad is not None:
            writer.add_histogram(f"gradients/{name}", param.grad.detach().cpu(), epoch)


def log_per_layer_grad_norms(writer, model, global_step):
    """Log L2 gradient norm per top-level module.

    Args:
        writer: SummaryWriter
        model: nn.Module (unwrapped)
        global_step: int
    """
    for name, module in model.named_children():
        params = [p for p in module.parameters() if p.grad is not None]
        if not params:
            continue
        total_norm = torch.sqrt(
            sum(p.grad.detach().float().pow(2).sum() for p in params)
        ).item()
        add_scalar_with_help(writer, f"grad_norm/{name}", total_norm, global_step)


def log_loss_ratios(writer, components, global_step, weights=None):
    """Log fraction each loss term contributes to total.

    Args:
        writer: SummaryWriter
        components: dict of loss component name -> value (tensor or float).
            Values are unweighted raw losses.
        global_step: int
        weights: optional dict of component name -> weight multiplier.
            When provided, each component is multiplied by its weight before
            computing the ratio, so ratios reflect weighted contributions and
            sum to 1.
    """
    total = components.get("total", None)
    if total is None:
        return
    total_val = total.item() if isinstance(total, torch.Tensor) else total
    if abs(total_val) < 1e-12:
        return

    for name, val in components.items():
        if name == "total":
            continue
        v = val.item() if isinstance(val, torch.Tensor) else val
        w = weights.get(name, 1.0) if weights else 1.0
        add_scalar_with_help(writer, f"loss_ratio/{name}", abs(v * w) / abs(total_val), global_step)
