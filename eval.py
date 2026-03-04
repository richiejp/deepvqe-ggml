"""DeepVQE AEC evaluation script.

Features:
- Compute ERLE, PESQ, STOI, segmental SNR on validation set
- Generate spectrogram comparisons (mic vs enhanced vs clean)
- Delay distribution heatmaps
- Audio sample output
- TensorBoard and/or console output
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from data.dataset import AECDataset, DummyAECDataset
from src.config import load_config
from src.metrics import evaluate_sample
from src.model import DeepVQEAEC
from src.stft import istft
from train import collate_fn, load_checkpoint


def plot_spectrograms(mic_wav, enh_wav, clean_wav, sr, title="", save_path=None):
    """Plot side-by-side spectrograms of mic, enhanced, and clean."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    for ax, wav, label in zip(
        axes,
        [mic_wav, enh_wav, clean_wav],
        ["Microphone", "Enhanced", "Clean"],
    ):
        ax.specgram(wav, NFFT=512, Fs=sr, noverlap=256, cmap="magma")
        ax.set_ylabel(f"{label}\nFreq (Hz)")
        ax.set_ylim(0, sr // 2)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(title)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    return fig


def plot_delay_heatmap(delay_dist, save_path=None):
    """Plot delay distribution heatmap.

    delay_dist: (T, dmax) numpy array
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.imshow(delay_dist.T, aspect="auto", origin="lower", cmap="viridis")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Delay (frames)")
    ax.set_title("Delay Distribution")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    return fig


def evaluate(cfg, checkpoint_path, dummy=False, output_dir="eval_output", max_samples=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "spectrograms").mkdir(exist_ok=True)
    (out_dir / "audio").mkdir(exist_ok=True)
    (out_dir / "delays").mkdir(exist_ok=True)

    # Model
    model = DeepVQEAEC.from_config(cfg).to(device)
    epoch = load_checkpoint(checkpoint_path, model)
    model.eval()
    print(f"Loaded checkpoint from epoch {epoch}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    # Dataset
    if dummy:
        val_ds = DummyAECDataset(
            length=cfg.data.num_val,
            target_len=int(cfg.training.clip_length_sec * cfg.audio.sample_rate),
            n_fft=cfg.audio.n_fft,
            hop_length=cfg.audio.hop_length,
            delay_samples=160,
        )
    else:
        val_ds = AECDataset(cfg, split="val")

    n_samples = len(val_ds)
    if max_samples:
        n_samples = min(n_samples, max_samples)

    sr = cfg.audio.sample_rate
    target_len = int(cfg.training.clip_length_sec * sr)

    all_metrics = []

    with torch.no_grad():
        for i in range(n_samples):
            sample = val_ds[i]
            mic_stft = sample["mic_stft"].unsqueeze(0).to(device)
            ref_stft = sample["ref_stft"].unsqueeze(0).to(device)
            clean_stft = sample["clean_stft"].unsqueeze(0).to(device)

            enhanced, delay_dist = model(mic_stft, ref_stft, return_delay=True)

            # Convert to waveforms
            mic_wav = istft(mic_stft, cfg.audio.n_fft, cfg.audio.hop_length, length=target_len)
            enh_wav = istft(enhanced, cfg.audio.n_fft, cfg.audio.hop_length, length=target_len)
            clean_wav = sample["clean_wav"].unsqueeze(0).to(device)

            mic_np = mic_wav[0].cpu().numpy()
            enh_np = enh_wav[0].cpu().numpy()
            clean_np = clean_wav[0].cpu().numpy()

            # Compute metrics
            metrics = evaluate_sample(mic_np, enh_np, clean_np, sr)
            metrics["sample_idx"] = i
            if "delay_samples" in sample.get("metadata", {}):
                metrics["true_delay"] = sample["metadata"]["delay_samples"]
            all_metrics.append(metrics)

            # Save visualizations for first N samples
            if i < cfg.eval.audio_samples:
                # Spectrograms
                plot_spectrograms(
                    mic_np, enh_np, clean_np, sr,
                    title=f"Sample {i} (ERLE={metrics['erle_db']:.1f} dB)",
                    save_path=out_dir / "spectrograms" / f"sample_{i:04d}.png",
                )

                # Delay heatmap
                if delay_dist is not None:
                    plot_delay_heatmap(
                        delay_dist[0].cpu().numpy(),
                        save_path=out_dir / "delays" / f"delay_{i:04d}.png",
                    )

                # Save audio (as raw float32 .npy for simplicity)
                np.save(out_dir / "audio" / f"mic_{i:04d}.npy", mic_np)
                np.save(out_dir / "audio" / f"enhanced_{i:04d}.npy", enh_np)
                np.save(out_dir / "audio" / f"clean_{i:04d}.npy", clean_np)

            if (i + 1) % 10 == 0 or i == n_samples - 1:
                print(f"  Evaluated {i+1}/{n_samples}")

    # Aggregate metrics
    erle_values = [m["erle_db"] for m in all_metrics]
    seg_snr_values = [m["seg_snr"] for m in all_metrics]

    print(f"\n{'='*50}")
    print(f"Evaluation Results ({n_samples} samples)")
    print(f"{'='*50}")
    print(f"ERLE:        {np.mean(erle_values):.2f} dB (std={np.std(erle_values):.2f})")
    print(f"Seg SNR:     {np.mean(seg_snr_values):.2f} dB (std={np.std(seg_snr_values):.2f})")

    pesq_values = [m["pesq"] for m in all_metrics if "pesq" in m]
    if pesq_values:
        print(f"PESQ:        {np.mean(pesq_values):.3f} (std={np.std(pesq_values):.3f})")

    stoi_values = [m["stoi"] for m in all_metrics if "stoi" in m]
    if stoi_values:
        print(f"STOI:        {np.mean(stoi_values):.3f} (std={np.std(stoi_values):.3f})")

    print(f"\nOutput saved to: {out_dir}")

    # Save summary
    summary = {
        "epoch": epoch,
        "n_samples": n_samples,
        "erle_mean": float(np.mean(erle_values)),
        "erle_std": float(np.std(erle_values)),
        "seg_snr_mean": float(np.mean(seg_snr_values)),
        "seg_snr_std": float(np.std(seg_snr_values)),
    }
    if pesq_values:
        summary["pesq_mean"] = float(np.mean(pesq_values))
    if stoi_values:
        summary["stoi_mean"] = float(np.mean(stoi_values))

    np.save(out_dir / "summary.npy", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate DeepVQE AEC")
    parser.add_argument("--config", default="configs/default.yaml", help="Config file")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path")
    parser.add_argument("--dummy", action="store_true", help="Use dummy dataset")
    parser.add_argument("--output-dir", default="eval_output", help="Output directory")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples to evaluate")
    args = parser.parse_args()

    cfg = load_config(args.config)
    evaluate(cfg, args.checkpoint, dummy=args.dummy,
             output_dir=args.output_dir, max_samples=args.max_samples)
