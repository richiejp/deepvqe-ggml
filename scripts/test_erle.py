"""Test ERLE and delay estimation with real speech signals."""

import glob
import os
import numpy as np
import torch
import scipy.signal
from src.config import load_config
from src.model import DeepVQEAEC
from src.stft import stft, istft
from train import load_checkpoint


def load_wav_16k(path, target_len, rng):
    """Load a wav file, resample to 16kHz, crop/pad to target_len."""
    import soundfile as sf
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio[:, 0]
    if sr != 16000:
        audio = scipy.signal.resample_poly(audio, 16000, sr).astype(np.float32)
    # Random crop or zero-pad
    if len(audio) >= target_len:
        start = rng.randint(0, len(audio) - target_len + 1)
        audio = audio[start : start + target_len]
    else:
        pad = np.zeros(target_len, dtype=np.float32)
        pad[: len(audio)] = audio
        audio = pad
    return audio


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--temperature", type=float, default=None,
                        help="Override AlignBlock temperature at inference")
    args = parser.parse_args()

    cfg = load_config("configs/default.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DeepVQEAEC.from_config(cfg).to(device)
    epoch = load_checkpoint("checkpoints/best.pt", model)
    model.eval()

    if args.temperature is not None:
        model.align.temperature = args.temperature
        print(f"Loaded best checkpoint from epoch {epoch} (temperature override: {args.temperature})")
    else:
        print(f"Loaded best checkpoint from epoch {epoch} (temperature: {model.align.temperature})")

    sr = 16000
    hop = 256
    target_len = int(3.0 * sr)

    # Find clean speech files
    clean_dirs = [
        "/data/clean_fullband",  # Docker sqsh mount
        "/workspace/deepvqe/datasets_fullband/clean_fullband",
    ]
    clean_files = []
    for d in clean_dirs:
        clean_files.extend(glob.glob(os.path.join(d, "**", "*.wav"), recursive=True))
    if not clean_files:
        print("ERROR: No clean speech files found. Run inside Docker with sqsh mounts.")
        return

    print(f"Found {len(clean_files)} clean speech files")

    rng = np.random.RandomState(123)

    # Pick 10 random clean files for near-end, 10 for far-end
    n_trials = 10
    near_idxs = rng.choice(len(clean_files), n_trials, replace=False)
    far_idxs = rng.choice(len(clean_files), n_trials, replace=False)

    delay_ms_list = [0, 50, 100, 125, 150, 200]

    # Results: delay_ms -> list of ERLE values
    results = {d: [] for d in delay_ms_list}
    delay_results = {d: [] for d in delay_ms_list}

    for trial in range(n_trials):
        near = load_wav_16k(clean_files[near_idxs[trial]], target_len, rng)
        far = load_wav_16k(clean_files[far_idxs[trial]], target_len, rng)

        # Normalize
        near = near / (np.max(np.abs(near)) + 1e-8) * 0.3
        far = far / (np.max(np.abs(far)) + 1e-8) * 0.3

        for delay_ms in delay_ms_list:
            delay_samples = int(delay_ms / 1000 * sr)

            # Create echo
            echo = np.zeros_like(far)
            if delay_samples > 0 and delay_samples < target_len:
                echo[delay_samples:] = far[:-delay_samples] * 0.7
            elif delay_samples == 0:
                echo[:] = far * 0.7

            # Add small noise
            noise = rng.randn(target_len).astype(np.float32) * 0.005
            mic = near + echo + noise

            mic_stft = stft(torch.from_numpy(mic).unsqueeze(0), cfg.audio.n_fft, hop).to(device)
            ref_stft = stft(torch.from_numpy(far).unsqueeze(0), cfg.audio.n_fft, hop).to(device)

            with torch.no_grad():
                enhanced, delay_dist = model(mic_stft, ref_stft, return_delay=True)

            enh_wav = istft(enhanced, cfg.audio.n_fft, hop, length=target_len)[0].cpu().numpy()

            # ERLE: ratio of echo power in mic vs residual echo in enhanced
            echo_pow = np.sum(echo**2)
            residual = enh_wav - near
            residual_pow = np.sum(residual**2)
            erle = 10 * np.log10(echo_pow / (residual_pow + 1e-10))

            # Delay estimation
            dd = delay_dist[0].cpu().numpy()
            avg_dist = dd.mean(axis=0)
            peak = np.argmax(avg_dist)
            peak_w = avg_dist[peak]
            frame0_w = avg_dist[0]

            results[delay_ms].append(erle)
            delay_results[delay_ms].append((peak, peak_w, frame0_w))

    print(f"\n{'Delay':>8s}  {'ERLE mean':>10s} {'ERLE std':>9s}  {'Peak frame':>10s} {'Peak wt':>8s} {'Frame0 wt':>9s}")
    print("-" * 70)
    for delay_ms in delay_ms_list:
        erles = results[delay_ms]
        peaks = [d[0] for d in delay_results[delay_ms]]
        peak_ws = [d[1] for d in delay_results[delay_ms]]
        f0_ws = [d[2] for d in delay_results[delay_ms]]
        expected_frame = 31 - delay_ms / 1000 * sr / hop if delay_ms > 0 else 0
        print(
            f"{delay_ms:>6d}ms  {np.mean(erles):>+10.1f} {np.std(erles):>8.1f}dB  "
            f"peak={np.median(peaks):>4.0f} (exp={expected_frame:>4.1f})  "
            f"{np.mean(peak_ws):>7.3f}  {np.mean(f0_ws):>8.3f}"
        )


if __name__ == "__main__":
    main()
