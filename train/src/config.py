from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import yaml


@dataclass
class ModelConfig:
    mic_channels: List[int] = field(default_factory=lambda: [2, 64, 128, 128, 128, 128])
    far_channels: List[int] = field(default_factory=lambda: [2, 32, 128])
    align_hidden: int = 32
    dmax: int = 32
    align_temp_start: float = 1.0
    align_temp_end: float = 0.1
    align_temp_epochs: int = 20
    power_law_c: float = 0.3


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    n_fft: int = 512
    hop_length: int = 256
    n_freqs: int = 257


@dataclass
class TrainingConfig:
    # NOTE: YAML keys must match fields here — add new fields to BOTH places.
    batch_size: int = 8
    grad_accum_steps: int = 12
    num_workers: int = 4
    lr: float = 1.2e-3
    weight_decay: float = 5e-7
    epochs: int = 250
    clip_length_sec: float = 3.0
    amp: bool = True
    grad_clip: float = 5.0
    optimizer: str = "schedulefree"  # "schedulefree" or "soap"
    warmup_epochs: int = 5
    checkpoint_every: int = 5
    keep_checkpoints: int = 5
    early_stop_patience: int = 10
    early_stop_min_delta: float = 1e-3
    delay_acc_min: float = 0.7
    erle_min_db: float = 3.0


@dataclass
class LossConfig:
    plcmse_weight: float = 1.0
    mag_l1_weight: float = 0.5
    time_l1_weight: float = 0.5
    sisdr_weight: float = 0.5
    smooth_l1_weight: float = 0.0
    smooth_l1_beta: float = 1.0
    energy_preservation_weight: float = 0.0
    energy_pres_mode: str = "relative"  # "absolute" or "relative"
    delay_weight: float = 1.0
    entropy_weight: float = 0.01
    mask_reg_weight: float = 0.1
    power_law_c: float = 0.5


@dataclass
class FarEndAugConfig:
    """Realism augmentations on the far-end / echo path.

    Defaults mirror LocalVQE's main-pretraining values so the pipeline
    produces the anti-shortcut distribution out-of-the-box. Set any prob
    to 0 to disable that stage. Every stage is independently prob-gated —
    the identity baseline is always in the training distribution so the
    model can't fingerprint an always-on pipeline.
    """
    # Amplifier saturation on farend pre-RIR.
    saturation_prob: float = 0.20
    drive_range: Tuple[float, float] = (1.0, 2.5)
    nonlin_family: str = "random"  # "tanh" | "arctan" | "softsign" | "random"
    hardclip_prob: float = 0.03
    hardclip_thresh_range: Tuple[float, float] = (0.88, 0.98)

    # Speaker body EQ applied to echo post-RIR.
    speaker_eq_prob: float = 0.4
    eq_hpf_range: Tuple[float, float] = (80.0, 200.0)
    eq_lpf_range: Tuple[float, float] = (5000.0, 8000.0)
    eq_q_range: Tuple[float, float] = (0.5, 1.2)
    eq_bump_prob: float = 0.5
    eq_bump_freq_range: Tuple[float, float] = (200.0, 2500.0)
    eq_bump_gain_db_range: Tuple[float, float] = (-3.0, 5.0)

    # Feedforward compressor applied to farend before the ref/echo branch.
    ref_agc_prob: float = 0.20
    agc_threshold_db_range: Tuple[float, float] = (-24.0, -9.0)
    agc_ratio_range: Tuple[float, float] = (2.0, 4.0)
    agc_attack_ms: float = 5.0
    agc_release_ms: float = 50.0

    # Reference level mismatch applied to farend_out post-peak-norm.
    ref_gain_prob: float = 0.3
    ref_gain_db_range: Tuple[float, float] = (-20.0, 3.0)

    # Time-varying RIR on echo path.
    rir_crossfade_prob: float = 0.10

    # Clock drift between reference and mic stream.
    ref_drift_prob: float = 0.15
    ref_drift_ppm_range: Tuple[float, float] = (-100.0, 100.0)

    # Ref-burst gate: zeroes contiguous chunks of ref AND echo (they're
    # physically coupled — speaker off → no echo). Off by default to
    # match LocalVQE main pretraining (finetune-only).
    ref_burst_prob: float = 0.0
    ref_burst_zero_frac_range: Tuple[float, float] = (0.3, 0.7)
    ref_burst_block_ms_range: Tuple[float, float] = (200.0, 1500.0)

    # Pre-cancelled echo: upstream AEC has already attenuated echo by
    # 15-40 dB before it reaches our model. Off by default.
    pre_cancelled_prob: float = 0.0
    pre_cancelled_db_range: Tuple[float, float] = (-40.0, -15.0)

    # Ref fade envelope: cosine fade-in/out/dip applied jointly to ref
    # and echo. Off by default.
    ref_fade_prob: float = 0.0
    ref_fade_duration_ms_range: Tuple[float, float] = (300.0, 1500.0)

    # Ref dither floor: replace NE-ST zeros with low-dBFS white noise.
    # Off by default.
    ref_dither_prob: float = 0.0
    ref_dither_db_range: Tuple[float, float] = (-75.0, -55.0)

    # Ambient pickup: blend a low-level copy of mic-side noise into ref.
    # Off by default.
    ref_ambient_prob: float = 0.0
    ref_ambient_db_range: Tuple[float, float] = (-40.0, -20.0)

    def is_active(self) -> bool:
        return any(p > 0 for p in (
            self.saturation_prob, self.hardclip_prob, self.speaker_eq_prob,
            self.ref_agc_prob, self.rir_crossfade_prob,
            self.ref_gain_prob, self.ref_drift_prob,
            self.ref_burst_prob, self.pre_cancelled_prob, self.ref_fade_prob,
            self.ref_dither_prob, self.ref_ambient_prob,
        ))


@dataclass
class NearEndAugConfig:
    """Realism augmentations on the near-end speaker's voice.

    Applied to the near-end signal BEFORE the room RIR so the effects
    propagate through reverberation realistically. The same transforms
    are applied to the clean target — the model's job stays "suppress
    echo/noise + preserve this voice", not "invert the speaker's AGC".
    """
    # Speaker's own preamp AGC (their laptop/phone mic's gain ride).
    agc_prob: float = 0.20
    agc_threshold_db_range: Tuple[float, float] = (-24.0, -9.0)
    agc_ratio_range: Tuple[float, float] = (2.0, 4.0)
    agc_attack_ms: float = 5.0
    agc_release_ms: float = 50.0

    # Occasional shouting into the mic / preamp saturation.
    saturation_prob: float = 0.05
    drive_range: Tuple[float, float] = (1.0, 2.0)
    nonlin_family: str = "random"  # "tanh" | "arctan" | "softsign" | "random"

    def is_active(self) -> bool:
        return any(p > 0 for p in (self.agc_prob, self.saturation_prob))


@dataclass
class MicAugConfig:
    """Realism augmentations on the mic-capture path.

    Applied post-mix (near-end + echo + noise) to model the ADC / mic
    response chain. Every stage is independently prob-gated.
    """
    # Capture EQ (HPF bass-cut + LPF bandwidth-limit, optional presence bump).
    mic_eq_prob: float = 0.4
    mic_hpf_range: Tuple[float, float] = (80.0, 250.0)
    mic_lpf_range: Tuple[float, float] = (5000.0, 8000.0)
    mic_eq_q_range: Tuple[float, float] = (0.5, 1.2)
    mic_eq_bump_prob: float = 0.5
    mic_eq_bump_freq_range: Tuple[float, float] = (1500.0, 5000.0)
    mic_eq_bump_gain_db_range: Tuple[float, float] = (-3.0, 5.0)

    # Additive self-noise (mic preamp hiss).
    mic_hiss_prob: float = 0.15
    mic_hiss_snr_range: Tuple[float, float] = (35.0, 60.0)
    mic_hiss_pink_prob: float = 0.5  # otherwise white

    # ADC / envelope clipping.
    mic_clip_prob: float = 0.03
    mic_clip_thresh_range: Tuple[float, float] = (0.85, 0.97)

    def is_active(self) -> bool:
        return any(p > 0 for p in (
            self.mic_eq_prob, self.mic_hiss_prob, self.mic_clip_prob,
        ))


@dataclass
class DataConfig:
    clean_dir: str = ""
    noise_dir: str = ""
    rir_dir: str = ""
    farend_dir: str = ""
    snr_range: Tuple[float, float] = (5, 40)
    ser_range: Tuple[float, float] = (-10, 10)
    delay_range: Tuple[float, float] = (0, 320)
    max_delay_frames: int = 32
    # Far-end single-talk (ref active, mic silent).
    single_talk_prob: float = 0.2
    # Near-end single-talk (mic active, ref is near-silence / zeros).
    single_talk_nearend_prob: float = 0.12
    # Ref has content but no echo in mix (headphones / muted speaker / far-field).
    ref_active_no_echo_prob: float = 0.10
    # In no-echo scenarios, zero near-end with this probability.
    silent_nearend_prob: float = 0.15
    # Draw ref from noise pool instead of farend pool (music / game audio).
    nonspeech_ref_prob: float = 0.20
    max_rir_length_ms: float = 500.0
    drr_range: Tuple[float, float] = (0, 20)  # DRR dB, uniform
    # Skip both near-end and far-end RIR for a clip — forces the model to
    # align + cancel from ref directly, not from an RIR fingerprint.
    rir_skip_prob: float = 0.05
    dnsmos_ovrl_min: float = 0.0  # DNSMOS quality filter (0 = disabled)
    num_train: int = 10000  # only used for DummyAECDataset
    num_val: int = 1000
    # FixedSynthDataset settings (used with --overfit-real)
    overfit_delays_ms: List[float] = field(default_factory=lambda: [0, 40, 80, 120, 160, 200, 240, 300])
    overfit_snr_db: float = 20.0
    overfit_ser_db: float = 0.0
    overfit_repeat: int = 1
    farend_aug: FarEndAugConfig = field(default_factory=FarEndAugConfig)
    nearend_aug: NearEndAugConfig = field(default_factory=NearEndAugConfig)
    mic_aug: MicAugConfig = field(default_factory=MicAugConfig)


@dataclass
class EvalConfig:
    pesq_subset: int = 200
    dnsmos_subset: int = 100
    aecmos_subset: int = 100
    audio_samples: int = 10


@dataclass
class PathsConfig:
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    data: DataConfig = field(default_factory=DataConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)


def _apply_dict(dc, d):
    """Recursively apply dict values to a dataclass instance."""
    for k, v in d.items():
        if not hasattr(dc, k):
            raise ValueError(f"Unknown config key: {k}")
        current = getattr(dc, k)
        if hasattr(current, "__dataclass_fields__") and isinstance(v, dict):
            _apply_dict(current, v)
        else:
            # Convert lists to tuples for tuple-typed fields
            if isinstance(v, list) and isinstance(current, tuple):
                v = tuple(v)
            setattr(dc, k, v)


def load_config(path: str | Path = "configs/default.yaml") -> Config:
    cfg = Config()
    with open(path) as f:
        d = yaml.safe_load(f)
    if d:
        _apply_dict(cfg, d)
    return cfg


if __name__ == "__main__":
    cfg = load_config()
    print(cfg)
