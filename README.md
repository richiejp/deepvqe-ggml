# DeepVQE — AEC with Soft Delay Estimation

Training, evaluation, and GGML inference implementation of DeepVQE
(Indenbom et al., Interspeech 2023) for joint acoustic echo cancellation,
noise suppression, and dereverberation.

**Paper**: [DeepVQE: Real Time Deep Voice Quality Enhancement](https://arxiv.org/abs/2306.03177)

**Focus**: AEC with soft delay estimation for cases with significant echo lag.

## Status

Phases 0-4 complete, Phase 5 (GGML) has export + skeleton C++ + comparison
infrastructure. Model verified (7.97M params, causality OK, all gradients flow,
AMP works). Data pipeline verified (STFT round-trip < 1e-6, delay
cross-correlation exact). Training loop validated on CPU (loss decreases,
checkpoints work). Evaluation script produces ERLE/PESQ/STOI/segSNR metrics,
spectrograms, and delay heatmaps. GGUF export with BN folding verified (max
error 3.9e-6).

Uses `uv` for dependency management (`uv sync` to install).

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
  data/
    dataset.py                  # AECDataset class
    synth.py                    # Online audio synthesis
  train.py                      # Training script
  eval.py                       # Evaluation script
  test_model.py                 # Model verification tests
  test_data.py                  # Data pipeline verification
  export_ggml.py                # Weight export with BN folding
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
- [x] `data/synth.py` — Online synthesis (nearend + echo + noise with configurable delay)
- [x] `data/dataset.py` — AECDataset + DummyAECDataset for testing
- [x] `test_data.py` — Verification script (5/5 tests pass)
- [x] Shapes correct (B,257,T,2)
- [x] Cross-correlation peak matches specified delay exactly
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

## Training Details

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning rate | 1.2e-3 |
| Weight decay | 5e-7 |
| Batch size (physical) | 8 |
| Gradient accumulation | 12 (effective batch 96) |
| Epochs | 250 |
| Clip length | 3.0 seconds (~188 frames) |
| Mixed precision | Yes (AMP) |
| Gradient clipping | 5.0 |
| Scheduler | Cosine annealing with 5-epoch warmup |

## Loss Function

Multi-component (paper does not specify, this is our design):
1. **Power-law compressed MSE** (weight=1.0): Compress pred/target STFT with c=0.3, MSE
2. **Magnitude L1** (weight=0.5): Direct magnitude accuracy
3. **Time-domain L1** (weight=0.1): Waveform reconstruction

## References

- [DeepVQE paper](https://arxiv.org/abs/2306.03177) (Indenbom et al., 2023)
- [Xiaobin-Rong implementation](https://github.com/Xiaobin-Rong/deepvqe) (NS-only, clean code)
- [Okrio implementation](https://github.com/Okrio/deepvqe) (AEC path, reference)
