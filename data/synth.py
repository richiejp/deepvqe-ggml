"""Online audio synthesis for AEC training.

Generates training examples on-the-fly:
1. Sample clean near-end speech
2. Sample far-end speech (different speaker)
3. Apply near-end RIR for reverberation
4. Create echo: convolve far-end with far-end RIR, apply delay, scale by SER
5. Add background noise at random SNR
6. Mix: mic = reverbed_nearend + echo + noise
7. Handle single-talk scenarios (zero out near-end with probability)
"""

import random

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve, resample_poly


def _load_random_audio(file_list, target_len, sr=16000):
    """Load a random audio file and extract a segment of target_len samples.

    Uses partial reads via sf.read(start, stop) to avoid loading entire files
    into memory, which matters for datasets with long recordings.
    """
    path = random.choice(file_list)
    info = sf.info(path)
    file_sr = info.samplerate
    total_frames = info.frames

    if file_sr != sr:
        # Need full read for resampling (can't partial-read then resample cleanly)
        audio, _ = sf.read(path, dtype="float32")
        if audio.ndim > 1:
            audio = audio[:, 0]
        # Use polyphase anti-aliasing filter for proper downsampling
        from math import gcd
        g = gcd(sr, file_sr)
        audio = resample_poly(audio, sr // g, file_sr // g).astype(np.float32)
    elif total_frames <= target_len:
        # File is shorter than target: read entire file, will pad below
        audio, _ = sf.read(path, dtype="float32")
        if audio.ndim > 1:
            audio = audio[:, 0]
    else:
        # Partial read: pick a random offset and read only target_len frames
        start = random.randint(0, total_frames - target_len)
        audio, _ = sf.read(path, dtype="float32", start=start, stop=start + target_len)
        if audio.ndim > 1:
            audio = audio[:, 0]

    # Pad or crop to target length
    if len(audio) < target_len:
        audio = np.pad(audio, (0, target_len - len(audio)))
    elif len(audio) > target_len:
        start = random.randint(0, len(audio) - target_len)
        audio = audio[start : start + target_len]

    return audio.astype(np.float32)


def _load_random_rir(rir_list):
    """Load a random RIR."""
    path = random.choice(rir_list)
    rir, _ = sf.read(path, dtype="float32")
    if rir.ndim > 1:
        rir = rir[:, 0]
    # Normalize RIR
    rir = rir / (np.abs(rir).max() + 1e-12)
    return rir


def _rms(x):
    return np.sqrt(np.mean(x**2) + 1e-12)


def _scale_to_snr(signal, noise, snr_db):
    """Scale noise to achieve target SNR relative to signal."""
    sig_rms = _rms(signal)
    noise_rms = _rms(noise)
    target_noise_rms = sig_rms / (10 ** (snr_db / 20))
    return noise * (target_noise_rms / (noise_rms + 1e-12))


def _scale_to_ser(nearend, echo, ser_db):
    """Scale echo to achieve target signal-to-echo ratio."""
    ne_rms = _rms(nearend)
    echo_rms = _rms(echo)
    target_echo_rms = ne_rms / (10 ** (ser_db / 20))
    return echo * (target_echo_rms / (echo_rms + 1e-12))


def synthesize_example(
    clean_files,
    noise_files,
    farend_files,
    rir_files=None,
    target_len=48000,
    sr=16000,
    snr_range=(5, 40),
    ser_range=(-10, 10),
    delay_range=(0, 320),
    single_talk_prob=0.2,
):
    """Synthesize a single AEC training example.

    Returns:
        mic: (N,) microphone signal
        ref: (N,) far-end reference signal (before RIR)
        clean: (N,) clean near-end speech (reverbed)
        metadata: dict with delay_ms, snr_db, ser_db, scenario
    """
    # Sample audio
    nearend = _load_random_audio(clean_files, target_len, sr)
    farend = _load_random_audio(farend_files, target_len, sr)
    noise = _load_random_audio(noise_files, target_len, sr)

    # Decide scenario
    is_single_talk = random.random() < single_talk_prob

    # Apply near-end RIR (reverberation)
    if rir_files:
        nearend_rir = _load_random_rir(rir_files)
        nearend_reverbed = fftconvolve(nearend, nearend_rir)[:target_len].astype(
            np.float32
        )
    else:
        nearend_reverbed = nearend.copy()

    # Create echo path
    if rir_files:
        farend_rir = _load_random_rir(rir_files)
        echo = fftconvolve(farend, farend_rir)[:target_len].astype(np.float32)
    else:
        echo = farend.copy()

    # Apply delay (in samples)
    delay_ms = random.uniform(*delay_range)
    delay_samples = int(delay_ms * sr / 1000)
    if delay_samples > 0:
        echo = np.pad(echo, (delay_samples, 0))[:target_len]

    # Sample levels
    snr_db = random.uniform(*snr_range)
    ser_db = random.uniform(*ser_range)

    # Scale components
    if is_single_talk:
        nearend_reverbed = np.zeros_like(nearend_reverbed)
        clean = np.zeros_like(nearend)
        scenario = "single_talk_farend"
    else:
        clean = nearend_reverbed.copy()
        scenario = "double_talk"

    # Scale echo and noise relative to near-end (or to echo if single-talk)
    if not is_single_talk and _rms(nearend_reverbed) > 1e-6:
        echo = _scale_to_ser(nearend_reverbed, echo, ser_db)
        noise = _scale_to_snr(nearend_reverbed, noise, snr_db)
    else:
        # Single talk: noise relative to echo
        noise = _scale_to_snr(echo, noise, snr_db)

    # Mix
    mic = nearend_reverbed + echo + noise

    # Normalize to prevent clipping
    peak = max(np.abs(mic).max(), np.abs(farend).max(), 1e-6)
    if peak > 0.95:
        scale = 0.9 / peak
        mic *= scale
        farend_out = farend * scale
        clean *= scale
    else:
        farend_out = farend

    metadata = {
        "delay_ms": delay_ms,
        "delay_samples": delay_samples,
        "snr_db": snr_db,
        "ser_db": ser_db,
        "scenario": scenario,
    }

    return mic, farend_out, clean, metadata
