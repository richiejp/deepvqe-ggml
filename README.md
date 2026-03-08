# DeepVQE — AEC with Soft Delay Estimation

Training, evaluation, and GGML inference implementation of DeepVQE
(Indenbom et al., Interspeech 2023) for joint acoustic echo cancellation,
noise suppression, and dereverberation.

**Paper**: [DeepVQE: Real Time Deep Voice Quality Enhancement](https://arxiv.org/abs/2306.03177)

**Focus**: AEC with soft delay estimation for cases with significant echo lag.

## Status

Phases 0-4 complete, Phase 5 (GGML) has export + skeleton C++ + comparison
infrastructure. Model verified (7.97M params, causality OK, all gradients flow,
AMP works). Data pipeline verified with DNS5 real data (157K clean, 64K noise,
60K RIR files). All training data packed into a single squashfs image
(dns5.sqsh), mounted via Docker entrypoint.
Evaluation script produces ERLE/PESQ/STOI/segSNR metrics, spectrograms, and
delay heatmaps. GGUF export with BN folding verified (max error 3.9e-6).

Uses Docker for training (`make build && make train-minimal`).

## Architecture

| Component | Details |
|-----------|---------|
| Sample rate | 16 kHz |
| STFT | 512 FFT, 256 hop, sqrt-Hann window, 257 freq bins |
| Mic encoder | 5 blocks: 2→64→128→128→128→128 channels |
| Far-end encoder | 2 blocks: 2→32→128 channels |
| AlignBlock | Cross-attention soft delay, dmax=32 (320ms), h=32 similarity channels |
| Encoder block 3 | 256→128 (concatenated mic + aligned far-end) |
| Bottleneck | GRU(1152→576) + Linear(576→1152) |
| Decoder | 5 blocks with sub-pixel conv: 128→128→128→64→27 |
| CCM | 27ch → 3×3 complex convolving mask (real-valued arithmetic) |
| Parameters | ~7.5M (full model) |

## Hardware

- Training: RTX 5070 16GB, ~2GB VRAM estimated for B=8 dmax=32 T=188 with AMP
- Datasets: ICASSP 2022 AEC + DNS challenge data

## Project Structure

```
deepvqe/
  configs/default.yaml          # Training configuration
  src/
    model.py                    # DeepVQEAEC (full model)
    blocks.py                   # FE, ResidualBlock, EncoderBlock, etc.
    align.py                    # AlignBlock (soft delay estimation)
    ccm.py                      # Complex convolving mask (real-valued)
    losses.py                   # Loss functions
    metrics.py                  # ERLE, PESQ, STOI
    stft.py                     # STFT/iSTFT helpers
    viz.py                      # Visualization helpers (spectrograms, delays, activations)
  data/
    dataset.py                  # AECDataset class
    synth.py                    # Online audio synthesis
  scripts/
    entrypoint.sh               # Docker entrypoint (mounts .sqsh datasets)
    download_dns5_minimal.sh    # Download DNS5 subset
  notebooks/
    explore_training.ipynb      # Interactive pipeline exploration notebook
  train.py                      # Training script
  eval.py                       # Evaluation script
  test_model.py                 # Model verification tests
  test_data.py                  # Data pipeline verification
  export_ggml.py                # Weight export with BN folding
  Makefile                      # Build, train, eval targets
  Dockerfile                    # NGC PyTorch + squashfuse
  reference/
    deepvqe_xr.py               # Xiaobin-Rong NS-only impl (real-valued CCM)
    deepvqe_xr_v1.py            # Xiaobin-Rong NS-only impl (complex CCM)
  ggml/
    deepvqe.cpp                 # C++ GGML inference
    compare.py                  # Layer-by-layer comparison
  pyproject.toml                # uv project config
```

## Implementation Checklist

### Phase 0: Project Setup
- [x] Directory structure created
- [x] YAML config system with dataclass validation
- [x] pyproject.toml with uv dependency management
- [x] Config loads and prints correctly

### Phase 1: Model Implementation
- [x] `src/blocks.py` — FE, ResidualBlock, EncoderBlock, Bottleneck, SubpixelConv2d, DecoderBlock
- [x] `src/ccm.py` — Real-valued CCM (GGML-friendly, no torch.complex)
- [x] `src/align.py` — AlignBlock with reshape bug fix and return_delay option
- [x] `src/model.py` — DeepVQEAEC with far-end branch and alignment
- [x] `test_model.py` — Verification script (7/7 tests pass)
- [x] Forward pass produces correct shape (B,257,T,2)
- [x] Parameter count: 7,975,063
- [x] Causality verified (future frames don't affect past)
- [x] All 116 parameters receive non-zero gradients
- [x] AlignBlock delay distribution sums to 1.0
- [x] Works with AMP (bfloat16 on CPU, float16 on CUDA)
- [x] Peak memory: B=1 T=188 runs on CPU

### Phase 2: Data Pipeline
- [x] `src/stft.py` — STFT/iSTFT with round-trip error 7.15e-7
- [x] `data/synth.py` — Online synthesis with anti-aliased resampling (scipy resample_poly)
- [x] `data/dataset.py` — AECDataset + DummyAECDataset, squashfuse-compatible
- [x] `test_data.py` — Verification script (5/5 tests pass)
- [x] Shapes correct (B,257,T,2)
- [x] Cross-correlation peak matches specified delay exactly
- [x] DNS5 data: 157K clean, 64K noise, 60K RIR (48kHz → 16kHz resampled)
- [ ] SER matches within 1 dB
- [ ] Spectrograms visualized for random examples
- [ ] Delay distribution uniform (not concentrated near zero)

### Phase 3: Training
- [x] `src/losses.py` — Power-law compressed MSE + magnitude L1 + time-domain L1
- [x] `train.py` — AdamW, AMP, gradient accumulation, cosine warmup
- [x] Per-step logging: loss, lr, grad norm
- [x] Per-epoch eval: validation loss, audio samples, delay heatmaps
- [x] Checkpointing (last 5 + best)
- [x] Loss decreases over first 3 epochs (CPU dummy: 4.01→2.28→1.36)
- [x] No NaN/Inf
- [x] Checkpoint save/load round-trip verified
- [ ] GPU memory < 14GB throughout (needs GPU test)

### Phase 4: Evaluation
- [x] `src/metrics.py` — ERLE, PESQ, STOI, segmental SNR
- [x] `eval.py` — Full evaluation with visualizations
- [x] Spectrogram comparisons (mic vs enhanced vs clean)
- [x] Delay distribution heatmaps
- [ ] ERLE > 10 dB at epoch 50+ (needs trained model)
- [ ] ERLE > 20 dB single-talk at epoch 200+
- [ ] PESQ > 3.0 double-talk at epoch 200+
- [ ] Add differentiable STOI proxy loss (e.g. torch-stoi or correlation-based approx)
- [ ] Add PESQ proxy loss (e.g. PESQ-Net or perceptual weighting)

### Phase 5: GGML Conversion
- [x] `export_ggml.py` — BN folding + GGUF export via gguf package
- [x] `ggml/deepvqe.cpp` — C++ skeleton (ELU, GRU, softmax, FE helpers)
- [x] `ggml/compare.py` — Layer-by-layer comparison infrastructure (146 layers captured)
- [x] BN folding verified (max error 3.9e-6)
- [ ] Full C++ GGML compute graph implementation
- [ ] Layer-by-layer max error < 1e-4 (f32)
- [ ] End-to-end max error < 1e-3 (f32)
- [ ] Runs faster than real-time
- [ ] PTQ q8_0: PESQ drop < 0.2

## Data Setup

```bash
# Download DNS5 minimal subset (~25GB download, ~50GB unpacked)
./scripts/download_dns5_minimal.sh datasets_fullband

# Pack all training data into a single squashfs image.
# Create a staging directory with hardlinks (no extra disk space):
mkdir -p datasets_fullband/_dns5_staging
cp -rl datasets_fullband/clean_fullband datasets_fullband/_dns5_staging/clean
cp -rl datasets_fullband/datasets_fullband/noise_fullband datasets_fullband/_dns5_staging/noise
cp -rl datasets_fullband/datasets_fullband/impulse_responses datasets_fullband/_dns5_staging/impulse_responses

mksquashfs datasets_fullband/_dns5_staging datasets_fullband/sqsh/dns5.sqsh \
    -comp zstd -Xcompression-level 3
rm -rf datasets_fullband/_dns5_staging

# The Docker entrypoint auto-mounts .sqsh files to /data/<name>.
# dns5.sqsh → /data/dns5/{clean,noise,impulse_responses}
```

## Paper Reference Results

From [Indenbom et al., Interspeech 2023](https://arxiv.org/abs/2306.03177):

| Metric | LD-M | LD-H | AEC-FEST | AEC-DT |
|--------|------|------|----------|--------|
| ERLE (dB) | 61.22 | 55.51 | 65.70 | — |

Paper hyperparameters (loss function not disclosed):

| Parameter | Paper | Ours |
|-----------|-------|------|
| Optimizer | AdamW | AdamW |
| Learning rate | 1.2e-3 | 1.2e-3 |
| Weight decay | 5e-7 | 5e-7 |
| Batch size | 400 | 64 (32×2 accum) |
| Epochs | 250 | 250 |
| Parameters | 7.5M | 7.97M |
| dmax (max delay frames) | 100 (~1s) | 32 (~0.5s) |

Notes:
- Paper ERLE 65.70 dB is on far-end single-talk (FEST) — easiest scenario, no near-end speech.
- LD-M / LD-H are low-delay medium/high difficulty test sets.
- DSP-aligned baseline achieved 41.76 / 33.18 / 54.12 dB on the same sets.
- Paper's batch size is ~6x ours — significantly more gradient steps per epoch.
- Paper does not disclose loss function components or weights.

## Training Details

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning rate | 1.2e-3 |
| Weight decay | 5e-7 |
| Batch size (physical) | 32 |
| Gradient accumulation | 2 (effective batch 64) |
| Epochs | 250 |
| Clip length | 3.0 seconds (~188 frames) |
| Mixed precision | Yes (AMP) |
| Gradient clipping | 5.0 |
| Scheduler | ReduceLROnPlateau (patience=3, factor=0.5) |

## Loss Function

Multi-component (paper does not disclose, this is our design):
1. **Power-law compressed MSE** (weight=1.0): Compress pred/target STFT with c=0.3, MSE
2. **Magnitude L1** (weight=0.5): Direct magnitude accuracy
3. **Time-domain L1** (weight=0.5): Waveform reconstruction
4. **Delay cross-entropy** (weight=1.0): Supervises AlignBlock attention with ground truth delay
5. **Entropy regularization** (weight=0.01): Sharpens attention distribution

Hard training gates (after epoch 20): delay accuracy ≥ 70%, ERLE ≥ 3 dB.

## Visualization & Debugging

`src/viz.py` provides helpers used during training and interactive exploration:

- **Spectrogram comparison** — mic vs enhanced vs clean spectrograms
- **Delay distribution** — AlignBlock attention heatmaps with ground-truth overlay
- **CCM mask decomposition** — visualize the 3×3 complex convolving mask channels
- **Encoder activations** — per-block activation heatmaps and statistics
- **TensorBoard integration** — weight histograms, per-layer gradient norms, loss ratios

`notebooks/explore_training.ipynb` walks through the full pipeline interactively
(STFT, encoder, AlignBlock, bottleneck, decoder, CCM, loss). Works with
`DummyAECDataset` (no data needed) or a trained checkpoint.

## References

- [DeepVQE paper](https://arxiv.org/abs/2306.03177) (Indenbom et al., 2023)
- [Xiaobin-Rong implementation](https://github.com/Xiaobin-Rong/deepvqe) (NS-only, clean code)
- [Okrio implementation](https://github.com/Okrio/deepvqe) (AEC path, reference)

### Reference Code (`reference/`)

Local copies of third-party implementations used as starting points:

- **`deepvqe_xr.py`** — Xiaobin-Rong's NS-only DeepVQE with **real-valued CCM** (no `torch.complex`). This is the variant our CCM is based on for GGML compatibility.
- **`deepvqe_xr_v1.py`** — Same architecture but uses **complex-valued CCM** (`torch.complex`). Kept for comparison.

Both implement: FE → 5 EncoderBlocks → GRU Bottleneck → 5 DecoderBlocks → CCM (single-input NS, no AEC/far-end branch).

#### Known Bugs in Reference Code

- **Xiaobin-Rong AlignBlock** (line 57): uses `K.shape[1]` (hidden=32) instead of `x_ref.shape[1]` (in_channels=128) for weighted sum reshape — fixed in our `src/align.py`.
- **Okrio AlignBlock**: `torch.zeros()` without `.to(device)` — fails on GPU. Fixed in our implementation.
