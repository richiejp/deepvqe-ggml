"""Evaluation metrics for DeepVQE AEC.

Metrics:
- ERLE: Echo Return Loss Enhancement
- PESQ: Perceptual Evaluation of Speech Quality (wideband)
- STOI: Short-Time Objective Intelligibility (extended)
- Segmental SNR improvement
"""

import numpy as np
import torch


def erle(mic_wav, enhanced_wav, clean_wav, frame_len=512, eps=1e-10):
    """Compute Echo Return Loss Enhancement (ERLE) in dB.

    ERLE = 10 * log10(E[echo^2] / E[residual_echo^2])
    where echo = mic - clean, residual_echo = enhanced - clean.

    Args:
        mic_wav: (N,) numpy array, microphone signal
        enhanced_wav: (N,) numpy array, enhanced signal
        clean_wav: (N,) numpy array, clean near-end signal
        frame_len: frame length for segmental computation

    Returns:
        erle_db: scalar, overall ERLE in dB
        erle_frames: (num_frames,) per-frame ERLE in dB
    """
    echo = mic_wav - clean_wav
    residual = enhanced_wav - clean_wav

    # Overall ERLE
    echo_power = np.mean(echo**2) + eps
    residual_power = np.mean(residual**2) + eps
    erle_db = 10 * np.log10(echo_power / residual_power)

    # Per-frame ERLE
    n_frames = len(echo) // frame_len
    erle_frames = np.zeros(n_frames)
    for i in range(n_frames):
        s = i * frame_len
        e = s + frame_len
        ep = np.mean(echo[s:e] ** 2) + eps
        rp = np.mean(residual[s:e] ** 2) + eps
        erle_frames[i] = 10 * np.log10(ep / rp)

    return erle_db, erle_frames


def segmental_snr(clean_wav, enhanced_wav, frame_len=512, eps=1e-10):
    """Compute segmental SNR improvement in dB.

    Args:
        clean_wav: (N,) numpy array
        enhanced_wav: (N,) numpy array
        frame_len: frame length

    Returns:
        seg_snr: scalar, average segmental SNR in dB
    """
    noise = enhanced_wav - clean_wav
    n_frames = len(clean_wav) // frame_len
    snrs = np.zeros(n_frames)
    for i in range(n_frames):
        s = i * frame_len
        e = s + frame_len
        sig_pow = np.mean(clean_wav[s:e] ** 2) + eps
        noise_pow = np.mean(noise[s:e] ** 2) + eps
        snrs[i] = 10 * np.log10(sig_pow / noise_pow)
    # Clamp to [-10, 35] dB as is standard
    snrs = np.clip(snrs, -10, 35)
    return float(np.mean(snrs))


def compute_pesq(clean_wav, enhanced_wav, sr=16000):
    """Compute wideband PESQ score.

    Returns None if pesq library is not installed.
    """
    try:
        from pesq import pesq as pesq_fn

        score = pesq_fn(sr, clean_wav, enhanced_wav, "wb")
        return float(score)
    except ImportError:
        return None
    except Exception:
        return None


def compute_stoi(clean_wav, enhanced_wav, sr=16000):
    """Compute extended STOI score.

    Returns None if pystoi library is not installed.
    """
    try:
        from pystoi import stoi

        score = stoi(clean_wav, enhanced_wav, sr, extended=True)
        return float(score)
    except ImportError:
        return None
    except Exception:
        return None


def evaluate_sample(mic_wav, enhanced_wav, clean_wav, sr=16000):
    """Compute all metrics for a single sample.

    Args:
        mic_wav, enhanced_wav, clean_wav: (N,) numpy arrays

    Returns:
        dict of metric name -> value
    """
    results = {}

    erle_db, erle_frames = erle(mic_wav, enhanced_wav, clean_wav)
    results["erle_db"] = erle_db
    results["erle_frames"] = erle_frames

    results["seg_snr"] = segmental_snr(clean_wav, enhanced_wav)

    pesq_score = compute_pesq(clean_wav, enhanced_wav, sr)
    if pesq_score is not None:
        results["pesq"] = pesq_score

    stoi_score = compute_stoi(clean_wav, enhanced_wav, sr)
    if stoi_score is not None:
        results["stoi"] = stoi_score

    return results
