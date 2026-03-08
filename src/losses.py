"""Loss functions for DeepVQE training.

Multi-component loss:
1. Power-law compressed MSE (weight=1.0)
2. Magnitude L1 (weight=0.5)
3. Time-domain L1 (weight=0.5)
4. SI-SDR (weight=0.0, disabled — spectral losses drive AEC training)
5. Mask magnitude regularizer (penalizes deviation from unity)
"""

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange

from src.stft import istft


def mask_mag_from_raw(d1):
    """Compute per-element mask magnitude from raw 27-channel decoder output.

    The 27 channels encode 3 cube-root-of-unity basis × 9 kernel elements.
    This decomposes into H_real and H_imag (each 9 elements), then computes
    the magnitude per kernel tap: sqrt(H_real^2 + H_imag^2).

    Args:
        d1: (B, 27, T, F) raw mask channels from decoder

    Returns:
        mask_mag: (B, 9, T, F) magnitude per kernel element
    """
    v_real = torch.tensor([1, -0.5, -0.5], device=d1.device, dtype=d1.dtype)
    v_imag = torch.tensor(
        [0, np.sqrt(3) / 2, -np.sqrt(3) / 2], device=d1.device, dtype=d1.dtype
    )
    m = rearrange(d1, "b (r c) t f -> b r c t f", r=3)
    H_real = torch.sum(v_real[None, :, None, None, None] * m, dim=1)  # (B,9,T,F)
    H_imag = torch.sum(v_imag[None, :, None, None, None] * m, dim=1)  # (B,9,T,F)
    return torch.sqrt(H_real ** 2 + H_imag ** 2 + 1e-12)


def si_sdr(pred, target):
    """Scale-invariant signal-to-distortion ratio (higher is better).

    Returns negative SI-SDR so it can be minimized.
    """
    # Remove mean
    pred = pred - pred.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)
    # s_target projection
    dot = torch.sum(pred * target, dim=-1, keepdim=True)
    s_target = dot * target / (torch.sum(target**2, dim=-1, keepdim=True) + 1e-8)
    e_noise = pred - s_target
    si_sdr_val = 10 * torch.log10(
        torch.sum(s_target**2, dim=-1) / (torch.sum(e_noise**2, dim=-1) + 1e-8) + 1e-8
    )
    return -si_sdr_val.mean()  # negative so we minimize


def mask_magnitude_regularizer(d1):
    """Penalize CCM mask magnitude deviating from 1 (identity/passthrough).

    For each of the 9 kernel taps, the ideal identity mask has magnitude 1
    at the center tap and 0 elsewhere.  We use a simpler formulation: pull
    the *mean* magnitude (across all 9 taps) toward 1.  This prevents both
    over-suppression (mag→0) and explosion (mag→∞) without micro-managing
    individual taps.

    Args:
        d1: (B, 27, T, F) raw 27-channel decoder output (before CCM)

    Returns:
        Scalar regularization loss.
    """
    mag = mask_mag_from_raw(d1)  # (B, 9, T, F)
    mean_mag = mag.mean(dim=1)  # (B, T, F) — average over 9 kernel taps
    return torch.mean((mean_mag - 1.0) ** 2)


class DeepVQELoss(nn.Module):
    """Combined loss for DeepVQE AEC training."""

    def __init__(
        self,
        plcmse_weight=1.0,
        mag_l1_weight=0.5,
        time_l1_weight=0.5,
        sisdr_weight=0.5,
        power_law_c=0.5,
        n_fft=512,
        hop_length=256,
    ):
        super().__init__()
        self.plcmse_weight = plcmse_weight
        self.mag_l1_weight = mag_l1_weight
        self.time_l1_weight = time_l1_weight
        self.sisdr_weight = sisdr_weight
        self.c = power_law_c
        self.n_fft = n_fft
        self.hop_length = hop_length

    def _compress(self, x):
        """Power-law compression: |X|^c * exp(j*angle(X)), as real-valued."""
        mag = torch.sqrt(x[..., 0] ** 2 + x[..., 1] ** 2 + 1e-12)
        mag_c = mag.pow(self.c)
        # Scale real/imag by |X|^(c-1)
        scale = mag_c / (mag + 1e-12)
        return x * scale.unsqueeze(-1)

    def _magnitude(self, x):
        """Compute magnitude from (B,F,T,2)."""
        return torch.sqrt(x[..., 0] ** 2 + x[..., 1] ** 2 + 1e-12)

    def forward(self, pred_stft, target_stft, target_wav=None):
        """
        pred_stft: (B, F, T, 2) — predicted enhanced STFT
        target_stft: (B, F, T, 2) — target clean STFT
        target_wav: (B, N) — target clean waveform (optional, for time-domain loss)

        Returns:
            total_loss, dict of component losses
        """
        components = {}

        # 1. Power-law compressed MSE
        pred_c = self._compress(pred_stft)
        target_c = self._compress(target_stft)
        plcmse = torch.mean((pred_c - target_c) ** 2)
        components["plcmse"] = plcmse

        # 2. Magnitude L1
        pred_mag = self._magnitude(pred_stft)
        target_mag = self._magnitude(target_stft)
        mag_l1 = torch.mean(torch.abs(pred_mag - target_mag))
        components["mag_l1"] = mag_l1

        # 3. Time-domain L1 and 4. SI-SDR
        if target_wav is not None and (self.time_l1_weight > 0 or self.sisdr_weight > 0):
            pred_wav = istft(
                pred_stft, self.n_fft, self.hop_length, length=target_wav.shape[-1]
            )
            time_l1 = torch.mean(torch.abs(pred_wav - target_wav)) if self.time_l1_weight > 0 else torch.tensor(0.0, device=pred_stft.device)
            sisdr_loss = si_sdr(pred_wav, target_wav) if self.sisdr_weight > 0 else torch.tensor(0.0, device=pred_stft.device)
            components["time_l1"] = time_l1
            components["sisdr"] = sisdr_loss
        else:
            components["time_l1"] = torch.tensor(0.0, device=pred_stft.device)
            components["sisdr"] = torch.tensor(0.0, device=pred_stft.device)
            time_l1 = components["time_l1"]
            sisdr_loss = components["sisdr"]

        total = (
            self.plcmse_weight * plcmse
            + self.mag_l1_weight * mag_l1
            + self.time_l1_weight * time_l1
            + self.sisdr_weight * sisdr_loss
        )
        components["total"] = total
        return total, components

    @classmethod
    def from_config(cls, cfg):
        return cls(
            plcmse_weight=cfg.loss.plcmse_weight,
            mag_l1_weight=cfg.loss.mag_l1_weight,
            time_l1_weight=cfg.loss.time_l1_weight,
            sisdr_weight=cfg.loss.sisdr_weight,
            power_law_c=cfg.loss.power_law_c,
            n_fft=cfg.audio.n_fft,
            hop_length=cfg.audio.hop_length,
        )
