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

from data.synth import synthesize_example
from src.stft import stft

_CACHE_DIR = Path(".cache/file_lists")


def _collect_audio_files(directory):
    """Recursively collect .wav and .flac files, with disk caching.

    On large datasets (500k+ files), rglob + sort can take minutes.
    Results are cached to .cache/file_lists/ keyed by directory path.
    The cache is invalidated when the directory mtime changes.
    """
    if not directory:
        return []
    d = Path(directory)
    if not d.exists():
        return []

    # Cache key based on absolute path
    abs_path = str(d.resolve())
    cache_key = hashlib.md5(abs_path.encode()).hexdigest()
    cache_file = _CACHE_DIR / f"{cache_key}.json"
    dir_mtime = os.path.getmtime(abs_path)

    # Try loading from cache
    if cache_file.exists():
        try:
            cached = json.loads(cache_file.read_text())
            if cached.get("mtime") == dir_mtime and cached.get("path") == abs_path:
                return cached["files"]
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
        "mtime": dir_mtime,
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
        self.cfg = cfg
        self.sr = cfg.audio.sample_rate
        self.n_fft = cfg.audio.n_fft
        self.hop_length = cfg.audio.hop_length
        self.target_len = int(cfg.training.clip_length_sec * self.sr)

        self.clean_files = _collect_audio_files(cfg.data.clean_dir)
        self.noise_files = _collect_audio_files(cfg.data.noise_dir)
        self.farend_files = _collect_audio_files(cfg.data.farend_dir)
        self.rir_files = _collect_audio_files(cfg.data.rir_dir) or None

        # If farend_dir not specified, use clean_dir
        if not self.farend_files:
            self.farend_files = self.clean_files

        self.length = cfg.data.num_train if split == "train" else cfg.data.num_val

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        mic, ref, clean, metadata = synthesize_example(
            clean_files=self.clean_files,
            noise_files=self.noise_files,
            farend_files=self.farend_files,
            rir_files=self.rir_files,
            target_len=self.target_len,
            sr=self.sr,
            snr_range=tuple(self.cfg.data.snr_range),
            ser_range=tuple(self.cfg.data.ser_range),
            delay_range=tuple(self.cfg.data.delay_range),
            single_talk_prob=self.cfg.data.single_talk_prob,
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
            "clean_wav": clean_t.squeeze(0),
            "delay_samples": metadata["delay_samples"],
            "metadata": metadata,
        }


class DummyAECDataset(Dataset):
    """Synthetic dataset for testing (no audio files needed).

    Generates random signals with known delay for verification.
    """

    def __init__(self, length=100, target_len=48000, n_fft=512, hop_length=256,
                 delay_samples=0, sr=16000):
        self.length = length
        self.target_len = target_len
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.delay_samples = delay_samples
        self.sr = sr

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        rng = np.random.RandomState(idx)

        # Generate signals
        clean = rng.randn(self.target_len).astype(np.float32) * 0.1
        farend = rng.randn(self.target_len).astype(np.float32) * 0.1
        noise = rng.randn(self.target_len).astype(np.float32) * 0.01

        # Echo with delay
        echo = np.zeros_like(farend)
        if self.delay_samples < self.target_len:
            end = min(self.target_len, self.target_len - self.delay_samples)
            echo[self.delay_samples:] = farend[:end] * 0.5

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
            "clean_wav": clean_t.squeeze(0),
            "delay_samples": self.delay_samples,
            "metadata": {
                "delay_ms": self.delay_samples / self.sr * 1000,
                "delay_samples": self.delay_samples,
                "snr_db": 20.0,
                "ser_db": 6.0,
                "scenario": "double_talk",
            },
        }
