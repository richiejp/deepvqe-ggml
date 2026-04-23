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
from math import gcd

import numpy as np
import soundfile as sf
from scipy.signal import (
    fftconvolve,
    iirfilter,
    lfilter,
    resample as scipy_resample,
    resample_poly,
    sosfilt,
)


def _load_audio(path, target_len, sr=16000):
    """Load an audio file and extract a random segment of target_len samples.

    Uses partial reads via sf.read(start, stop) to avoid loading entire files
    into memory, which matters for datasets with long recordings.
    """
    info = sf.info(path)
    file_sr = info.samplerate
    total_frames = info.frames

    if file_sr != sr:
        # Need full read for resampling (can't partial-read then resample cleanly)
        audio, _ = sf.read(path, dtype="float32")
        if audio.ndim > 1:
            audio = audio[:, 0]
        # Use polyphase anti-aliasing filter for proper downsampling
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
        pad_total = target_len - len(audio)
        pad_left = random.randint(0, pad_total)
        audio = np.pad(audio, (pad_left, pad_total - pad_left))
    elif len(audio) > target_len:
        start = random.randint(0, len(audio) - target_len)
        audio = audio[start : start + target_len]

    return audio.astype(np.float32)


def _load_random_audio(file_list, target_len, sr=16000):
    """Load a random audio file from the list."""
    return _load_audio(random.choice(file_list), target_len, sr)


def _load_random_rir(rir_list, max_length_samples=None):
    """Load a random RIR, optionally truncated with fade-out."""
    path = random.choice(rir_list)
    rir, _ = sf.read(path, dtype="float32")
    if rir.ndim > 1:
        rir = rir[:, 0]
    # Normalize RIR
    rir = rir / (np.abs(rir).max() + 1e-12)
    # Truncate if requested
    if max_length_samples is not None and len(rir) > max_length_samples:
        rir = rir[:max_length_samples]
        # Cosine fade-out over last 10ms (160 samples at 16kHz)
        fade_len = min(160, max_length_samples)
        fade = np.cos(np.linspace(0, np.pi / 2, fade_len)).astype(np.float32) ** 2
        rir[-fade_len:] *= fade
    return rir


def _rms(x):
    return np.sqrt(np.mean(x**2) + 1e-12)


def _scale_to_dbfs(x, target_db, eps=1e-12):
    """Scale x so its RMS matches target_db (dBFS)."""
    rms = float(np.sqrt(np.mean(x ** 2) + eps))
    return (x * (10.0 ** (target_db / 20.0) / rms)).astype(np.float32)


def _frame_rms_db(x, frame_len=512, eps=1e-10):
    """Per-frame RMS in dB. Returns empty array if x is shorter than one frame."""
    n = len(x) // frame_len
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    f = x[: n * frame_len].reshape(n, frame_len)
    return (10.0 * np.log10(np.mean(f * f, axis=1) + eps)).astype(np.float32)


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


# ── Far-end realism helpers ──────────────────────────────────────────────
# Model the physical path from digital far-end to mic echo:
#   sender AGC -> DAC -> amp saturation -> speaker body -> room -> mic


_NONLIN_FAMILIES = ("tanh", "arctan", "softsign")


def _nonlinearity(x, drive, hardclip_thresh=None, family="tanh"):
    """Saturation at `drive`; drive=1 is near-identity. Optional hardclip.

    family: one of _NONLIN_FAMILIES, or "random" (picks one per call).
    Different families have different spectral signatures — randomising them
    prevents the model from keying on a single nonlinearity fingerprint.
    """
    if drive <= 1.0 + 1e-6 and hardclip_thresh is None:
        return x
    if family == "random":
        family = random.choice(_NONLIN_FAMILIES)
    elif family not in _NONLIN_FAMILIES:
        raise ValueError(
            f"Unknown nonlin_family {family!r}; expected one of "
            f"{_NONLIN_FAMILIES + ('random',)}"
        )
    if drive > 1.0 + 1e-6:
        if family == "arctan":
            x = np.arctan(drive * x) / np.arctan(drive)
        elif family == "softsign":
            x = (drive * x) / (1.0 + np.abs(drive * x)) * (1.0 + drive) / drive
        else:
            x = np.tanh(drive * x) / np.tanh(drive)
    if hardclip_thresh is not None:
        x = np.clip(x, -hardclip_thresh, hardclip_thresh)
    return x.astype(np.float32)


def _peaking_biquad_sos(f0_hz, sr, gain_db, q):
    """RBJ audio EQ cookbook peaking filter. Stable by construction for
    Q > 0 and any gain_db. Replaces an earlier buggy iirpeak-based hack
    that could produce non-biquad frequency responses for |gain_db| > ~6."""
    w0 = 2.0 * np.pi * f0_hz / sr
    A = 10.0 ** (gain_db / 40.0)
    alpha = np.sin(w0) / (2.0 * q)
    cos_w0 = np.cos(w0)
    a0 = 1.0 + alpha / A
    b = np.array([1.0 + alpha * A, -2.0 * cos_w0, 1.0 - alpha * A]) / a0
    a = np.array([1.0, (-2.0 * cos_w0) / a0, (1.0 - alpha / A) / a0])
    return np.concatenate([b, a]).reshape(1, 6)


def _build_speaker_eq(sr, hpf_hz, lpf_hz, q, bump_freq_hz=None, bump_gain_db=0.0):
    """Build SOS cascade approximating a consumer speaker body response:
    HPF (bass rolloff) -> optional resonance bump -> LPF (treble rolloff).
    """
    nyq = sr / 2
    sections = [iirfilter(2, hpf_hz / nyq, btype="high", ftype="butter", output="sos")]
    if bump_freq_hz is not None and abs(bump_gain_db) > 1e-3:
        sections.append(_peaking_biquad_sos(bump_freq_hz, sr, bump_gain_db, q))
    sections.append(iirfilter(2, lpf_hz / nyq, btype="low", ftype="butter", output="sos"))
    return np.concatenate(sections, axis=0)


def _apply_speaker_eq(x, sos):
    """Apply SOS in float64 and clip to finite range.

    Some (HPF + resonance-bump + LPF) combinations hit numerically unstable
    corners of the biquad parameter space when stacked with a subsequent
    filter (e.g. mic EQ after echo EQ via the mic mix) — float32 state can
    overflow to inf on ~1% of random draws, poisoning downstream scaling.
    Running in float64 + clipping to [-64, 64] is cheap insurance; legit
    filter outputs never approach this bound for unit-peak inputs, so the
    distribution is unchanged while pathological draws stay finite.
    """
    y = sosfilt(sos, x.astype(np.float64))
    y = np.nan_to_num(y, nan=0.0, posinf=64.0, neginf=-64.0)
    return np.clip(y, -64.0, 64.0).astype(np.float32)


def _envelope_compressor(x, sr, thresh_db, ratio, atk_ms, rel_ms, eps=1e-10):
    """One-pole envelope follower + static soft-knee compression curve.

    Vectorised via `scipy.signal.lfilter`. Uses a single time constant (the
    geometric mean of atk_ms and rel_ms) — we lose asymmetric attack/release,
    but this is an augmentation, not a mastering tool: the gain-reduction
    distribution is what matters, and the symmetric follower gives a very
    similar envelope for speech-like inputs at ~10x the throughput of a
    Python recurrence loop.
    """
    tau_ms = (atk_ms * rel_ms) ** 0.5
    alpha = float(np.exp(-1.0 / (tau_ms * 1e-3 * sr + 1.0)))
    abs_x = np.abs(x).astype(np.float32)
    env = lfilter([1.0 - alpha], [1.0, -alpha], abs_x).astype(np.float32)
    env_db = 20.0 * np.log10(env + eps)
    over = np.maximum(env_db - thresh_db, 0.0)
    gain_db = -over * (1.0 - 1.0 / ratio)
    gain = 10.0 ** (gain_db / 20.0)
    return (x * gain).astype(np.float32)


def _pink_noise(n, rng=None):
    """1/f magnitude spectrum, random phase, unit RMS. O(N log N)."""
    if rng is None:
        rng = np.random
    n_fft = 1 << (n - 1).bit_length()
    freqs = np.fft.rfftfreq(n_fft, 1.0)
    mag = np.zeros(len(freqs), dtype=np.float32)
    mag[1:] = 1.0 / np.sqrt(freqs[1:])
    phase = rng.uniform(-np.pi, np.pi, size=len(freqs)).astype(np.float32)
    spec = mag * (np.cos(phase) + 1j * np.sin(phase))
    y = np.fft.irfft(spec, n=n_fft).astype(np.float32)[:n]
    y /= (np.sqrt(np.mean(y ** 2)) + 1e-12)
    return y


def _resample_ppm(x, ppm, target_len):
    """Simulate tiny clock drift by resampling to len*(1 + ppm/1e6).

    Uses FFT-based resample (O(N log N)) — far cheaper than rational
    resample_poly for sub-1000-ppm ratios, where polyphase would allocate
    a ~1M-tap FIR. ppm=0 returns x unchanged.
    """
    if abs(ppm) < 1.0:
        return x.astype(np.float32)
    new_len = max(1, int(round(len(x) * (1.0 + ppm / 1e6))))
    y = scipy_resample(x, new_len).astype(np.float32)
    if len(y) >= target_len:
        return y[:target_len]
    out = np.zeros(target_len, dtype=np.float32)
    out[: len(y)] = y
    return out


def _burst_mask(n, sr, zero_frac, block_ms_range):
    """Random-placement contiguous-block zero mask.

    Repeatedly drops blocks of random length (within block_ms_range) at
    random positions until the target zero fraction is met or an
    iteration guard trips. Returns (float32 mask, achieved_frac).
    """
    mask = np.ones(n, dtype=np.float32)
    min_samp = max(1, int(block_ms_range[0] * sr / 1000))
    max_samp = max(min_samp, int(block_ms_range[1] * sr / 1000))
    max_samp = min(max_samp, n)
    target = int(zero_frac * n)
    zeroed = 0   # running count avoids O(n) mask.sum() per iter
    guard = 0
    while zeroed < target and guard < 100:
        block = random.randint(min_samp, max_samp)
        start = random.randint(0, max(0, n - block))
        region = mask[start:start + block]
        zeroed += int(region.sum())   # count only freshly-zeroed samples
        region[:] = 0.0
        guard += 1
    return mask, zeroed / max(n, 1)


def _fade_envelope(n, sr, fade_type, duration_ms):
    """Raised-cosine fade envelope of length n.

    fade_type ∈ {"in", "out", "dip"}. "in" ramps 0→1 over duration_ms at
    the start; "out" ramps 1→0 at the end; "dip" goes 1→0→1 centred at
    a random interior position, total dip span = duration_ms.
    """
    env = np.ones(n, dtype=np.float32)
    d = min(n, max(2, int(duration_ms * sr / 1000)))
    ramp_up = (0.5 - 0.5 * np.cos(np.linspace(0, np.pi, d))).astype(np.float32)
    if fade_type == "in":
        env[:d] = ramp_up
    elif fade_type == "out":
        env[-d:] = ramp_up[::-1]
    else:  # "dip"
        d = min(d, n - 2)
        half = max(1, d // 2)
        centre = random.randint(half, n - half - 1)
        env[centre - half:centre] = ramp_up[:half][::-1]
        env[centre:centre + half] = ramp_up[:half]
    return env


def synthesize_example(
    clean_path,
    noise_files,
    farend_files,
    rir_files=None,
    target_len=48000,
    sr=16000,
    snr_range=(5, 40),
    ser_range=(-10, 10),
    delay_range=(0, 320),
    single_talk_prob=0.2,
    single_talk_nearend_prob=0.0,
    ref_active_no_echo_prob=0.0,
    silent_nearend_prob=0.0,
    nonspeech_ref_prob=0.0,
    max_rir_length_ms=None,
    drr_range=None,
    rir_skip_prob=0.0,
    farend_aug=None,
    nearend_aug=None,
    mic_aug=None,
    intermediates=None,
):
    """Synthesize a single AEC training example.

    Args:
        clean_path: path to the clean near-end speech file
        noise_files: list of noise file paths (one chosen randomly)
        farend_files: list of far-end file paths (one chosen randomly)
        max_rir_length_ms: truncate RIRs to this length (None = no truncation)
        drr_range: (min_db, max_db) direct-to-reverberant ratio for mic near-end
        farend_aug: FarEndAugConfig (or None) controlling realistic far-end
            distortion (speaker saturation, EQ, AGC, clock drift, etc.).
        intermediates: if a dict is passed in, it is populated in-place with
            per-stage audio snapshots (`farend_raw`, `farend_compressed`,
            `echo_pre_nonlin`, `echo_post_eq`, `echo_final`,
            `farend_out_pre_drift`) for TB diagnostic logging.

    Returns:
        mic: (N,) microphone signal
        ref: (N,) far-end reference signal (before RIR)
        clean: (N,) clean near-end speech (dry, anechoic)
        metadata: dict with delay_ms, snr_db, ser_db, drr_db, scenario, and
            any realised far-end-aug parameters
    """
    # Sample audio
    nearend = _load_audio(clean_path, target_len, sr)
    # Ref content: speech (default) or from the noise pool (nonspeech_ref
    # case — models loopback carrying music / game / system audio). Noise
    # pool is the same one used for mic-side noise; that's fine, it
    # still gives the model non-speech spectra to cancel.
    use_nonspeech_ref = (nonspeech_ref_prob > 0.0
                         and random.random() < nonspeech_ref_prob)
    if use_nonspeech_ref:
        farend = _load_random_audio(noise_files, target_len, sr)
    else:
        farend = _load_random_audio(farend_files, target_len, sr)
    noise = _load_random_audio(noise_files, target_len, sr)

    if intermediates is not None:
        intermediates["farend_raw"] = farend.copy()
        intermediates["nearend_raw"] = nearend.copy()

    # Decide scenario. 4-way roll, in order of the cumulative probability
    # interval. Remainder after all three specialised scenarios is double_talk.
    roll = random.random()
    is_st_farend = roll < single_talk_prob
    is_st_nearend = (not is_st_farend) and (
        roll < single_talk_prob + single_talk_nearend_prob
    )
    is_ref_active_no_echo = (
        not is_st_farend and not is_st_nearend
        and roll < single_talk_prob + single_talk_nearend_prob + ref_active_no_echo_prob
    )
    # silent_nearend sub-flag: zero the near-end audio. Applied to the
    # scenarios where near-end would otherwise be active (everything
    # except single_talk_farend, which already has silent near-end).
    silent_nearend = (silent_nearend_prob > 0.0
                      and not is_st_farend
                      and random.random() < silent_nearend_prob)
    if silent_nearend:
        nearend = np.zeros_like(nearend)

    max_rir_samples = int(max_rir_length_ms * sr / 1000) if max_rir_length_ms else None
    aug_meta = {}

    # Gate RIR application for the whole clip (both nearend and farend paths
    # share the decision). A dry pass forces the model to align + cancel
    # from the ref signal directly, not from an RIR fingerprint — important
    # because real-inference RIRs are nothing like the training RIR pool.
    use_rir = bool(rir_files) and random.random() >= rir_skip_prob
    if rir_files and not use_rir:
        aug_meta["rir_skipped"] = True

    # Near-end speaker augs (AGC / saturation) — applied BEFORE RIR so the
    # effects propagate realistically through reverberation. Applied to the
    # single `nearend` variable so both the mic contribution AND the clean
    # target (which is copied from `nearend` below) see the same transform.
    if nearend_aug is not None and not nearend_aug.is_active():
        nearend_aug = None
    if nearend_aug is not None:
        if random.random() < nearend_aug.agc_prob:
            thresh_db = random.uniform(*nearend_aug.agc_threshold_db_range)
            ratio = random.uniform(*nearend_aug.agc_ratio_range)
            nearend = _envelope_compressor(
                nearend, sr, thresh_db, ratio,
                nearend_aug.agc_attack_ms, nearend_aug.agc_release_ms,
            )
            aug_meta["ne_agc_threshold_db"] = thresh_db
            aug_meta["ne_agc_ratio"] = ratio
        if random.random() < nearend_aug.saturation_prob:
            drive = random.uniform(*nearend_aug.drive_range)
            if drive > 1.0 + 1e-6:
                nearend = _nonlinearity(
                    nearend, drive, family=nearend_aug.nonlin_family,
                )
                aug_meta["ne_drive"] = drive

    if intermediates is not None:
        intermediates["nearend_post_aug"] = nearend.copy()

    # Apply near-end RIR (reverberation)
    if use_rir:
        nearend_rir = _load_random_rir(rir_files, max_rir_samples)
        nearend_reverbed = fftconvolve(nearend, nearend_rir)[:target_len].astype(
            np.float32
        )
    else:
        nearend_reverbed = nearend.copy()

    # DRR mixing: blend dry and reverbed near-end for mic signal
    drr_db = None
    if drr_range is not None and use_rir:
        drr_db = random.uniform(*drr_range)
        drr_linear = 10 ** (drr_db / 10)
        alpha = drr_linear / (1 + drr_linear)
        nearend_in_mic = (alpha * nearend + (1 - alpha) * nearend_reverbed).astype(
            np.float32
        )
    else:
        nearend_in_mic = nearend_reverbed

    # Treat an inactive farend_aug as None so the pre-refactor baseline stays
    # byte-identical — the aug branches must not consume RNG when disabled.
    if farend_aug is not None and not farend_aug.is_active():
        farend_aug = None
    if mic_aug is not None and not mic_aug.is_active():
        mic_aug = None
    farend_c = farend
    if farend_aug is not None and random.random() < farend_aug.ref_agc_prob:
        thresh_db = random.uniform(*farend_aug.agc_threshold_db_range)
        ratio = random.uniform(*farend_aug.agc_ratio_range)
        farend_c = _envelope_compressor(
            farend, sr, thresh_db, ratio,
            farend_aug.agc_attack_ms, farend_aug.agc_release_ms,
        )
        aug_meta["agc_threshold_db"] = thresh_db
        aug_meta["agc_ratio"] = ratio

    if intermediates is not None:
        intermediates["farend_compressed"] = farend_c.copy()

    farend_nl = farend_c
    if farend_aug is not None:
        fire_sat = random.random() < farend_aug.saturation_prob
        drive = random.uniform(*farend_aug.drive_range) if fire_sat else 1.0
        hardclip = None
        if random.random() < farend_aug.hardclip_prob:
            hardclip = random.uniform(*farend_aug.hardclip_thresh_range)
            aug_meta["hardclip_thresh"] = hardclip
        if fire_sat or hardclip is not None:
            farend_nl = _nonlinearity(
                farend_c, drive, hardclip, family=farend_aug.nonlin_family,
            )
        if fire_sat:
            aug_meta["drive"] = drive
            aug_meta["nonlin_family"] = farend_aug.nonlin_family

    if intermediates is not None:
        intermediates["echo_post_nonlin"] = farend_nl.copy()

    if use_rir:
        farend_rir = _load_random_rir(rir_files, max_rir_samples)
        echo = fftconvolve(farend_nl, farend_rir)[:target_len].astype(np.float32)
        if farend_aug is not None and random.random() < farend_aug.rir_crossfade_prob:
            farend_rir_b = _load_random_rir(rir_files, max_rir_samples)
            echo_b = fftconvolve(farend_nl, farend_rir_b)[:target_len].astype(np.float32)
            ramp = np.linspace(0.0, 1.0, target_len, dtype=np.float32)
            echo = (1.0 - ramp) * echo + ramp * echo_b
            aug_meta["rir_crossfade"] = True
    else:
        echo = farend_nl.copy()

    if farend_aug is not None and random.random() < farend_aug.speaker_eq_prob:
        hpf = random.uniform(*farend_aug.eq_hpf_range)
        lpf = random.uniform(*farend_aug.eq_lpf_range)
        q = random.uniform(*farend_aug.eq_q_range)
        bump_f = None
        bump_db = 0.0
        if random.random() < farend_aug.eq_bump_prob:
            bump_f = random.uniform(*farend_aug.eq_bump_freq_range)
            bump_db = random.uniform(*farend_aug.eq_bump_gain_db_range)
        sos = _build_speaker_eq(sr, hpf, lpf, q, bump_f, bump_db)
        echo = _apply_speaker_eq(echo, sos)
        aug_meta["eq_hpf_hz"] = hpf
        aug_meta["eq_lpf_hz"] = lpf
        if bump_f is not None:
            aug_meta["eq_bump_hz"] = bump_f
            aug_meta["eq_bump_db"] = bump_db

    if intermediates is not None:
        intermediates["echo_post_eq"] = echo.copy()

    # Burst/fade mutate farend_c AND echo jointly so they stay physically
    # coupled — a real speaker mute implies no echo. Pre-cancelled
    # attenuates echo only (upstream AEC scenario).
    pre_cancelled_fired = False
    if farend_aug is not None and not is_st_nearend:
        if random.random() < farend_aug.ref_burst_prob:
            zero_frac = random.uniform(*farend_aug.ref_burst_zero_frac_range)
            burst_mask, actual_frac = _burst_mask(
                target_len, sr, zero_frac, farend_aug.ref_burst_block_ms_range,
            )
            farend_c = (farend_c * burst_mask).astype(np.float32)
            farend_nl = (farend_nl * burst_mask).astype(np.float32)
            echo = (echo * burst_mask).astype(np.float32)
            aug_meta["ref_burst_frac"] = actual_frac
        if random.random() < farend_aug.ref_fade_prob:
            fade_type = random.choice(("in", "out", "dip"))
            fade_dur_ms = random.uniform(*farend_aug.ref_fade_duration_ms_range)
            env = _fade_envelope(target_len, sr, fade_type, fade_dur_ms)
            farend_c = (farend_c * env).astype(np.float32)
            farend_nl = (farend_nl * env).astype(np.float32)
            echo = (echo * env).astype(np.float32)
            aug_meta["ref_fade_type"] = fade_type
            aug_meta["ref_fade_duration_ms"] = fade_dur_ms
        # Pre-cancelled echo: only meaningful when echo actually mixes in
        # (ST-FE or DT; zeroed scenarios skip).
        if (is_st_farend or (not is_ref_active_no_echo)) \
                and random.random() < farend_aug.pre_cancelled_prob:
            atten_db = random.uniform(*farend_aug.pre_cancelled_db_range)
            echo = (echo * (10.0 ** (atten_db / 20.0))).astype(np.float32)
            aug_meta["pre_cancelled_db"] = atten_db
            pre_cancelled_fired = True

    # Apply delay (in samples)
    delay_ms = random.uniform(*delay_range)
    delay_samples = int(delay_ms * sr / 1000)
    if delay_samples > 0:
        echo = np.pad(echo, (delay_samples, 0))[:target_len]

    # Sample levels
    snr_db = random.uniform(*snr_range)
    ser_db = random.uniform(*ser_range)

    # Scale components per scenario. silent_nearend propagates through
    # nearend being zero already (zeroed at scenario-roll time).
    if is_st_farend:
        nearend_in_mic = np.zeros_like(nearend)
        clean = np.zeros_like(nearend)
        scenario = "single_talk_farend"
    elif is_st_nearend:
        echo = np.zeros_like(echo)
        clean = nearend.copy()
        scenario = "single_talk_nearend"
    elif is_ref_active_no_echo:
        # Ref carries content but no echo couples into the mic (headphones /
        # muted / far-field). Echo contribution to mic is zero; ref
        # returned later from farend_c.
        echo = np.zeros_like(echo)
        clean = nearend.copy()
        scenario = "ref_active_no_echo"
    else:
        clean = nearend.copy()
        scenario = "double_talk"

    # Scale echo/noise. Scenarios where nearend_in_mic is zero but echo is
    # non-zero (single_talk_farend) key noise off echo; no-echo scenarios
    # with silent nearend fall through to a fixed low noise level so the
    # clip isn't bit-identical silence.
    if scenario == "single_talk_farend":
        noise = _scale_to_snr(echo, noise, snr_db)
    elif _rms(nearend_in_mic) > 1e-6:
        # Skip SER normalisation when pre_cancelled fired — otherwise it
        # would restore echo to the SER target, defeating the attenuation.
        if scenario == "double_talk" and not pre_cancelled_fired:
            echo = _scale_to_ser(nearend_in_mic, echo, ser_db)
        noise = _scale_to_snr(nearend_in_mic, noise, snr_db)
    else:
        # Silent-nearend variant of an echo-less scenario: fix noise at
        # −40 dBFS so mic isn't identically zero.
        noise = _scale_to_dbfs(noise, -40.0)

    if intermediates is not None:
        intermediates["echo_final"] = echo.copy()

    # Mix
    mic = nearend_in_mic + echo + noise

    # Don't co-scale farend_out with mic peak: a real loopback has an
    # independent gain relative to what the mic records.
    peak = max(np.abs(mic).max(), 1e-6)
    if peak > 0.95:
        scale = 0.9 / peak
        mic *= scale
        clean *= scale

    if intermediates is not None:
        intermediates["mic_pre_aug"] = mic.copy()

    if mic_aug is not None:
        # Capture EQ: applied to mic AND clean so the target tracks the mic's
        # tonal balance (model suppresses echo/noise, doesn't invert EQ).
        if random.random() < mic_aug.mic_eq_prob:
            hpf = random.uniform(*mic_aug.mic_hpf_range)
            lpf = random.uniform(*mic_aug.mic_lpf_range)
            q = random.uniform(*mic_aug.mic_eq_q_range)
            bump_f = None
            bump_db = 0.0
            if random.random() < mic_aug.mic_eq_bump_prob:
                bump_f = random.uniform(*mic_aug.mic_eq_bump_freq_range)
                bump_db = random.uniform(*mic_aug.mic_eq_bump_gain_db_range)
            sos = _build_speaker_eq(sr, hpf, lpf, q, bump_f, bump_db)
            mic = _apply_speaker_eq(mic, sos)
            clean = _apply_speaker_eq(clean, sos)
            aug_meta["mic_hpf_hz"] = hpf
            aug_meta["mic_lpf_hz"] = lpf
            if bump_f is not None:
                aug_meta["mic_bump_hz"] = bump_f
                aug_meta["mic_bump_db"] = bump_db
        # Preamp hiss: mic-only (clean target is noise-free — model suppresses).
        if random.random() < mic_aug.mic_hiss_prob:
            snr = random.uniform(*mic_aug.mic_hiss_snr_range)
            if random.random() < mic_aug.mic_hiss_pink_prob:
                hiss = _pink_noise(len(mic))
                aug_meta["mic_hiss_color"] = "pink"
            else:
                hiss = np.random.randn(len(mic)).astype(np.float32)
                aug_meta["mic_hiss_color"] = "white"
            ref_sig = mic if _rms(mic) > 1e-6 else np.ones_like(mic) * 1e-3
            hiss = _scale_to_snr(ref_sig, hiss, snr)
            mic = (mic + hiss).astype(np.float32)
            aug_meta["mic_hiss_snr_db"] = snr
        # ADC / envelope clipping: mic only.
        if random.random() < mic_aug.mic_clip_prob:
            thresh = random.uniform(*mic_aug.mic_clip_thresh_range)
            mic = np.clip(mic, -thresh, thresh).astype(np.float32)
            aug_meta["mic_clip_thresh"] = thresh

    if is_st_nearend:
        # Real loopback is literal zeros when the far-end pipeline is idle
        # (per user's recording: ref = data sent to speaker; no playback →
        # zeros). An optional dither-floor knob replaces zeros with
        # low-dBFS white noise for a minority of clips to cover deployments
        # with sub-optimal ADC loopback (non-zero quiet state).
        if (farend_aug is not None
                and random.random() < farend_aug.ref_dither_prob):
            dither_db = random.uniform(*farend_aug.ref_dither_db_range)
            farend_out = _scale_to_dbfs(
                np.random.randn(target_len).astype(np.float32), dither_db,
            )
            aug_meta["ref_content"] = "dither"
            aug_meta["ref_dither_db"] = dither_db
        else:
            farend_out = np.zeros(target_len, dtype=np.float32)
            aug_meta["ref_content"] = "zeros"
    else:
        farend_out = farend_c.copy()
        aug_meta["ref_content"] = "nonspeech" if use_nonspeech_ref else "speech"

    if silent_nearend:
        aug_meta["silent_nearend"] = True

    if intermediates is not None:
        intermediates["farend_out_pre_drift"] = farend_out.copy()

    if farend_aug is not None and not is_st_nearend:
        # Skip ref-side augs when ref is designed to be near-silent.
        if random.random() < farend_aug.ref_drift_prob:
            ppm = random.uniform(*farend_aug.ref_drift_ppm_range)
            if abs(ppm) >= 1.0:
                farend_out = _resample_ppm(farend_out, ppm, target_len)
                aug_meta["ref_drift_ppm"] = ppm
        if random.random() < farend_aug.ref_gain_prob:
            ref_gain_db = random.uniform(*farend_aug.ref_gain_db_range)
            if abs(ref_gain_db) > 1e-3:
                farend_out = farend_out * (10.0 ** (ref_gain_db / 20.0))
                aug_meta["ref_gain_db"] = ref_gain_db
        # Ambient-pickup: blend a low-level copy of mic-side noise into
        # ref, modelling a monitor mic that picks up room audio alongside
        # the playback. Uses `noise` (already resampled from the same
        # pool as mic-side noise) at an absolute dBFS floor.
        if random.random() < farend_aug.ref_ambient_prob:
            amb_db = random.uniform(*farend_aug.ref_ambient_db_range)
            amb = _scale_to_dbfs(noise.astype(np.float32, copy=False), amb_db)
            farend_out = (farend_out + amb).astype(np.float32)
            aug_meta["ref_ambient_db"] = amb_db

    metadata = {
        "delay_ms": delay_ms,
        "delay_samples": delay_samples,
        "snr_db": snr_db,
        "ser_db": ser_db,
        "drr_db": drr_db,
        "scenario": scenario,
        **aug_meta,
    }

    return mic, farend_out.astype(np.float32), clean, metadata
