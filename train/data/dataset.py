"""AEC Dataset for DeepVQE training.

Supports two modes:
1. Online synthesis: generates examples on-the-fly from clean/noise/RIR files
2. Pre-existing: loads pre-synthesized AEC challenge data
"""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from data.synth import (
    _load_audio,
    _load_random_rir,
    _scale_to_ser,
    _scale_to_snr,
    synthesize_example,
)
from src.stft import stft

_CACHE_DIR = Path(".cache/file_lists")
_DNSMOS_DIR = Path(".cache/dnsmos")


def dir_cache_key(directory: str) -> str:
    """MD5 hash of the resolved absolute path, used as a cache filename."""
    return hashlib.md5(str(Path(directory).resolve()).encode()).hexdigest()


def dnsmos_scores_path(clean_dir: str) -> Path:
    """Return the path to the DNSMOS scores cache file for a clean directory."""
    return _DNSMOS_DIR / f"{dir_cache_key(clean_dir)}.json"


def _filter_by_dnsmos(file_list, clean_dir, ovrl_min):
    """Filter file list by DNSMOS OVRL score.

    Loads pre-computed scores from .cache/dnsmos/<hash>.json.
    Returns unfiltered list with a warning if scores file is missing.
    """
    scores_file = dnsmos_scores_path(clean_dir)

    if not scores_file.exists():
        print(f"WARNING: DNSMOS scores not found at {scores_file}")
        print(f"  Run: python scripts/score_dnsmos.py --clean-dir {clean_dir}")
        print(f"  Proceeding without quality filtering.")
        return file_list

    data = json.loads(scores_file.read_text())
    scores = data.get("scores", {})

    kept = []
    skipped = 0
    unscored = 0
    errors = 0
    for f in file_list:
        s = scores.get(f)
        if s is None:
            unscored += 1
            kept.append(f)  # fail-open: keep files without scores
        elif "error" in s:
            errors += 1
            kept.append(f)
        elif s["OVRL"] >= ovrl_min:
            kept.append(f)
        else:
            skipped += 1

    print(f"DNSMOS filter: {len(kept)}/{len(file_list)} clean files "
          f"(OVRL >= {ovrl_min}, skipped {skipped}, "
          f"unscored {unscored}, errors {errors})")
    return kept


def collect_audio_files(directory):
    """Recursively collect .wav and .flac files, with disk caching.

    On large datasets (500k+ files), rglob + sort can take minutes.
    Results are cached to .cache/file_lists/ keyed by directory path.
    Cache is invalidated when file count changes (top-level mtime is
    unreliable for nested directories on most filesystems).
    """
    if not directory:
        return []
    d = Path(directory)
    if not d.exists():
        return []

    abs_path = str(d.resolve())
    cache_file = _CACHE_DIR / f"{dir_cache_key(directory)}.json"

    # Try loading from cache
    if cache_file.exists():
        try:
            cached = json.loads(cache_file.read_text())
            if cached.get("path") == abs_path:
                # Quick validation: spot-check that first and last files still exist
                files = cached["files"]
                if files and os.path.exists(files[0]) and os.path.exists(files[-1]):
                    return files
        except (json.JSONDecodeError, KeyError):
            pass

    # Scan directory
    files = []
    for ext in ("*.wav", "*.flac"):
        files.extend(str(f) for f in d.rglob(ext))
    files.sort()

    # Write cache
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps({
        "path": abs_path,
        "count": len(files),
        "files": files,
    }))

    return files


class AECDataset(Dataset):
    """Online-synthesis AEC dataset.

    Each __getitem__ call synthesizes a fresh example.
    """

    def __init__(self, cfg, split="train"):
        """
        Args:
            cfg: Config object with data and audio sections
            split: 'train' or 'val'
        """
        self.sr = cfg.audio.sample_rate
        self.n_fft = cfg.audio.n_fft
        self.hop_length = cfg.audio.hop_length
        self.target_len = int(cfg.training.clip_length_sec * self.sr)
        self.snr_range = tuple(cfg.data.snr_range)
        self.ser_range = tuple(cfg.data.ser_range)
        self.delay_range = tuple(cfg.data.delay_range)
        self.single_talk_prob = cfg.data.single_talk_prob
        self.single_talk_nearend_prob = cfg.data.single_talk_nearend_prob
        self.ref_active_no_echo_prob = cfg.data.ref_active_no_echo_prob
        self.silent_nearend_prob = cfg.data.silent_nearend_prob
        self.nonspeech_ref_prob = cfg.data.nonspeech_ref_prob
        self.rir_skip_prob = cfg.data.rir_skip_prob
        self.max_rir_length_ms = cfg.data.max_rir_length_ms
        self.drr_range = tuple(cfg.data.drr_range)
        self.farend_aug = cfg.data.farend_aug
        self.nearend_aug = cfg.data.nearend_aug
        self.mic_aug = cfg.data.mic_aug

        all_clean = collect_audio_files(cfg.data.clean_dir)
        if cfg.data.dnsmos_ovrl_min > 0:
            all_clean = _filter_by_dnsmos(all_clean, cfg.data.clean_dir,
                                          cfg.data.dnsmos_ovrl_min)
        self.noise_files = collect_audio_files(cfg.data.noise_dir)
        self.farend_files = collect_audio_files(cfg.data.farend_dir)
        self.rir_files = collect_audio_files(cfg.data.rir_dir) or None

        # If farend_dir not specified, use clean_dir (full pool, no split)
        if not self.farend_files:
            self.farend_files = all_clean

        # Hold out num_val clean files for validation (deterministic split)
        num_val = cfg.data.num_val
        if split == "train":
            self.clean_files = all_clean[num_val:]
            self.length = len(self.clean_files)
        else:
            self.clean_files = all_clean[:num_val]
            self.length = len(self.clean_files)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        clean_path = self.clean_files[idx]

        mic, ref, clean, metadata = synthesize_example(
            clean_path=clean_path,
            noise_files=self.noise_files,
            farend_files=self.farend_files,
            rir_files=self.rir_files,
            target_len=self.target_len,
            sr=self.sr,
            snr_range=self.snr_range,
            ser_range=self.ser_range,
            delay_range=self.delay_range,
            single_talk_prob=self.single_talk_prob,
            single_talk_nearend_prob=self.single_talk_nearend_prob,
            ref_active_no_echo_prob=self.ref_active_no_echo_prob,
            silent_nearend_prob=self.silent_nearend_prob,
            nonspeech_ref_prob=self.nonspeech_ref_prob,
            max_rir_length_ms=self.max_rir_length_ms,
            drr_range=self.drr_range,
            rir_skip_prob=self.rir_skip_prob,
            farend_aug=self.farend_aug,
            nearend_aug=self.nearend_aug,
            mic_aug=self.mic_aug,
        )

        # Convert to tensors and compute STFTs
        mic_t = torch.from_numpy(mic).unsqueeze(0)  # (1, N)
        ref_t = torch.from_numpy(ref).unsqueeze(0)
        clean_t = torch.from_numpy(clean).unsqueeze(0)

        mic_stft = stft(mic_t, self.n_fft, self.hop_length).squeeze(0)  # (F, T, 2)
        ref_stft = stft(ref_t, self.n_fft, self.hop_length).squeeze(0)
        clean_stft = stft(clean_t, self.n_fft, self.hop_length).squeeze(0)

        return {
            "mic_stft": mic_stft,
            "ref_stft": ref_stft,
            "clean_stft": clean_stft,
            "mic_wav": mic_t.squeeze(0),
            "ref_wav": ref_t.squeeze(0),
            "clean_wav": clean_t.squeeze(0),
            "delay_samples": metadata["delay_samples"],
            "metadata": metadata,
        }


class DummyAECDataset(Dataset):
    """Synthetic dataset for testing (no audio files needed).

    Generates amplitude-modulated tonal signals with per-example random
    delay for verification.  The amplitude envelope gives temporal structure
    so the AlignBlock can detect the delay from STFT magnitude patterns
    (stationary tones have identical frames, making delay undetectable).
    """

    def __init__(self, length=100, target_len=48000, n_fft=512, hop_length=256,
                 delay_range=(0, 0), sr=16000):
        self.length = length
        self.target_len = target_len
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.delay_range = delay_range  # (min_samples, max_samples)
        self.sr = sr

    def __len__(self):
        return self.length

    @staticmethod
    def _tonal_signal(rng, length, sr, n_harmonics=15, amplitude=0.1):
        """Generate an amplitude-modulated tonal signal.

        Sum of harmonics (smooth spectral structure like speech) multiplied
        by a random smooth envelope (3-6 bumps over the signal duration).
        The envelope gives temporal variation so delay estimation can work.
        """
        t = np.arange(length, dtype=np.float32) / sr
        f0 = rng.uniform(80, 300)  # fundamental frequency
        signal = np.zeros(length, dtype=np.float32)
        for h in range(1, n_harmonics + 1):
            freq = f0 * h
            if freq > sr / 2:
                break
            amp = amplitude * rng.uniform(0.2, 1.0) / h  # harmonic roll-off
            phase = rng.uniform(0, 2 * np.pi)
            signal += amp * np.sin(2 * np.pi * freq * t + phase).astype(np.float32)

        # Amplitude modulation: smooth random envelope with 3-6 bumps
        n_bumps = rng.randint(3, 7)
        env_freqs = rng.uniform(0.5, 3.0, size=n_bumps)
        env_phases = rng.uniform(0, 2 * np.pi, size=n_bumps)
        envelope = np.ones(length, dtype=np.float32) * 0.3  # baseline
        for ef, ep in zip(env_freqs, env_phases):
            envelope += (0.7 / n_bumps) * (1 + np.sin(2 * np.pi * ef * t + ep).astype(np.float32))
        envelope = np.clip(envelope, 0.0, 1.0)
        signal *= envelope
        return signal

    def __getitem__(self, idx):
        rng = np.random.RandomState(idx)

        # Generate amplitude-modulated tonal signals
        clean = self._tonal_signal(rng, self.target_len, self.sr)
        farend = self._tonal_signal(rng, self.target_len, self.sr)
        noise = rng.randn(self.target_len).astype(np.float32) * 0.01

        # Per-example random delay (quantized to hop_length for clean frame alignment)
        delay_lo, delay_hi = self.delay_range
        if delay_hi > delay_lo:
            delay_samples = rng.randint(delay_lo, delay_hi + 1)
            # Quantize to hop_length so ground truth aligns to STFT frames
            delay_samples = int(round(delay_samples / self.hop_length) * self.hop_length)
            delay_samples = max(delay_lo, min(delay_hi, delay_samples))
        else:
            delay_samples = delay_lo

        # Echo with delay
        echo = np.zeros_like(farend)
        if delay_samples < self.target_len:
            end = min(self.target_len, self.target_len - delay_samples)
            echo[delay_samples:] = farend[:end] * 0.5

        mic = clean + echo + noise

        mic_t = torch.from_numpy(mic).unsqueeze(0)
        ref_t = torch.from_numpy(farend).unsqueeze(0)
        clean_t = torch.from_numpy(clean).unsqueeze(0)

        mic_stft = stft(mic_t, self.n_fft, self.hop_length).squeeze(0)
        ref_stft = stft(ref_t, self.n_fft, self.hop_length).squeeze(0)
        clean_stft = stft(clean_t, self.n_fft, self.hop_length).squeeze(0)

        return {
            "mic_stft": mic_stft,
            "ref_stft": ref_stft,
            "clean_stft": clean_stft,
            "mic_wav": mic_t.squeeze(0),
            "ref_wav": ref_t.squeeze(0),
            "clean_wav": clean_t.squeeze(0),
            "delay_samples": delay_samples,
            "metadata": {
                "delay_ms": delay_samples / self.sr * 1000,
                "delay_samples": delay_samples,
                "snr_db": 20.0,
                "ser_db": 6.0,
                "scenario": "double_talk",
            },
        }


class FixedSynthDataset(Dataset):
    """Pre-synthesized fixed dataset for overfit testing with real audio.

    Loads one clean, one noise, one far-end, and one RIR file, then creates
    N examples varying only the echo delay. All examples share the same audio
    content, so the model must learn delay-dependent echo cancellation.
    """

    def __init__(self, clean_dir, noise_dir, farend_dir, rir_dir,
                 delays_ms, sr=16000, target_len=48000, n_fft=512,
                 hop_length=256, snr_db=20.0, ser_db=0.0, repeat=1,
                 max_rir_length_ms=None, drr_db=None):
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.repeat = repeat

        # Collect file lists and pick deterministic first files
        clean_files = collect_audio_files(clean_dir)
        noise_files = collect_audio_files(noise_dir)
        farend_files = collect_audio_files(farend_dir) or clean_files
        rir_files = collect_audio_files(rir_dir)

        assert clean_files, f"No audio files found in {clean_dir}"
        assert noise_files, f"No audio files found in {noise_dir}"

        # Load audio once (deterministic: always first file, full length)
        import random as _random
        state = _random.getstate()
        _random.seed(42)
        nearend = _load_audio(clean_files[0], target_len, sr)
        farend = _load_audio(farend_files[1 % len(farend_files)], target_len, sr)
        noise = _load_audio(noise_files[0], target_len, sr)
        max_rir_samples = int(max_rir_length_ms * sr / 1000) if max_rir_length_ms else None
        rir = _load_random_rir(rir_files, max_rir_samples) if rir_files else None
        _random.setstate(state)

        # Apply RIR to near-end
        if rir is not None:
            from scipy.signal import fftconvolve
            nearend_reverbed = fftconvolve(nearend, rir)[:target_len].astype(np.float32)
            echo_base = fftconvolve(farend, rir)[:target_len].astype(np.float32)
        else:
            nearend_reverbed = nearend.copy()
            echo_base = farend.copy()

        # DRR mixing for mic near-end component
        if drr_db is not None and rir is not None:
            drr_linear = 10 ** (drr_db / 10)
            alpha = drr_linear / (1 + drr_linear)
            nearend_in_mic = (alpha * nearend + (1 - alpha) * nearend_reverbed).astype(np.float32)
        else:
            nearend_in_mic = nearend_reverbed

        # Pre-synthesize one example per delay
        self.examples = []
        for delay_ms in delays_ms:
            delay_samples = int(delay_ms * sr / 1000)

            # Apply delay to echo
            echo = echo_base.copy()
            if delay_samples > 0:
                echo = np.pad(echo, (delay_samples, 0))[:target_len]

            # Scale echo and noise; target is dry nearend
            clean = nearend.copy()
            echo_scaled = _scale_to_ser(nearend_in_mic, echo, ser_db)
            noise_scaled = _scale_to_snr(nearend_in_mic, noise, snr_db)

            mic = nearend_in_mic + echo_scaled + noise_scaled

            # Normalize to prevent clipping
            peak = max(np.abs(mic).max(), np.abs(farend).max(), 1e-6)
            if peak > 0.95:
                scale = 0.9 / peak
                mic = mic * scale
                farend_out = farend * scale
                clean = clean * scale
            else:
                farend_out = farend

            # Compute STFTs
            mic_t = torch.from_numpy(mic).unsqueeze(0)
            ref_t = torch.from_numpy(farend_out).unsqueeze(0)
            clean_t = torch.from_numpy(clean).unsqueeze(0)

            self.examples.append({
                "mic_stft": stft(mic_t, n_fft, hop_length).squeeze(0),
                "ref_stft": stft(ref_t, n_fft, hop_length).squeeze(0),
                "clean_stft": stft(clean_t, n_fft, hop_length).squeeze(0),
                "mic_wav": mic_t.squeeze(0),
                "ref_wav": ref_t.squeeze(0),
                "clean_wav": clean_t.squeeze(0),
                "delay_samples": delay_samples,
                "metadata": {
                    "delay_ms": delay_ms,
                    "delay_samples": delay_samples,
                    "snr_db": snr_db,
                    "ser_db": ser_db,
                    "drr_db": drr_db,
                    "scenario": "double_talk",
                },
            })

        print(f"FixedSynthDataset: {len(self.examples)} examples × {repeat} repeat "
              f"= {len(self.examples) * repeat} virtual, delays={delays_ms} ms")

    def __len__(self):
        return len(self.examples) * self.repeat

    def __getitem__(self, idx):
        return self.examples[idx % len(self.examples)]
