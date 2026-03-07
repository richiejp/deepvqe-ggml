"""Loss functions for DeepVQE training.

Multi-component loss:
1. Power-law compressed MSE (weight=1.0)
2. Magnitude L1 (weight=0.5)
3. Time-domain L1 (weight=0.5)
4. SI-SDR (weight=0.0, disabled — spectral losses drive AEC training)
"""

import torch
import torch.nn as nn

from src.stft import istft


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
            time_l1 = torch.mean(torch.abs(pred_wav - target_wav))
            sisdr_loss = si_sdr(pred_wav, target_wav)
            components["time_l1"] = time_l1
            components["sisdr"] = sisdr_loss
        else:
            time_l1 = torch.tensor(0.0, device=pred_stft.device)
            sisdr_loss = torch.tensor(0.0, device=pred_stft.device)
            components["time_l1"] = time_l1
            components["sisdr"] = sisdr_loss

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
