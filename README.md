# DeepVQE-GGML

C/C++ inference engine for [DeepVQE](https://arxiv.org/abs/2306.03177)
(Indenbom et al., Interspeech 2023) — real-time acoustic echo cancellation
with soft delay estimation, built on [GGML](https://github.com/ggerganov/ggml).

> **Looking for something smaller?** We've since released
> [LocalVQE](https://github.com/localai-org/LocalVQE), a ~0.9M-parameter
> (~3.5 MB F32) derivative of DeepVQE with an in-graph DCT-II filterbank
> and an S4D bottleneck — roughly 9× smaller than the model in this repo,
> with streaming C++ inference and a Vulkan backend. This repository
> remains the full-width ~8M-parameter DeepVQE re-implementation.

## Building

Requires cmake and a C/C++17 compiler. A [Nix](https://nixos.org/) flake is
provided for reproducible builds:

```bash
# Enter dev shell (provides cmake, gcc, pkg-config)
nix develop

# Build the CLI inference binary
make build-ggml

# Or build the shared library (libdeepvqe.so) for embedding
make build-shared
```

Without Nix:

```bash
cd ggml
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

## Usage

### CLI

```bash
# Run inference on numpy STFT arrays (mic + far-end reference)
ggml/build/deepvqe model.gguf --input-npy mic.npy ref.npy

# Dump intermediate activations for debugging
ggml/build/deepvqe model.gguf --input-npy mic.npy ref.npy --dump-intermediates
```

### Shared Library (C API)

Build with `-DDEEPVQE_BUILD_SHARED=ON` to get `libdeepvqe.so` with a C API
defined in `ggml/deepvqe_api.h`. This can be loaded via `dlopen`, Go's
`purego`, or any FFI.

See `ggml/example_purego_test.go` for a Go integration example.

### Block Verification

Verify C++ blocks against PyTorch reference outputs:

```bash
# First, export PyTorch intermediates (requires Docker, see train/)
make compare-pt
make compare-block

# Then run C++ block tests
make test-ggml
```

## Architecture

| Component | Details |
|-----------|---------|
| Sample rate | 16 kHz |
| STFT | 512 FFT, 256 hop, sqrt-Hann window, 257 freq bins |
| Mic encoder | 5 blocks: 2->64->128->128->128->128 channels |
| Far-end encoder | 2 blocks: 2->32->128 channels |
| AlignBlock | Cross-attention soft delay, dmax=32 (320ms) |
| Bottleneck | GRU(1152->576) + Linear(576->1152) |
| Decoder | 5 blocks with sub-pixel conv |
| CCM | 27ch -> 3x3 complex convolving mask |
| Parameters | ~8.0M |

## Model Weights

Pre-trained weights are available on Hugging Face:
[richiejp/deepvqe-aec-gguf](https://huggingface.co/richiejp/deepvqe-aec-gguf).

| Variant | File | Size | Description |
|---------|------|------|-------------|
| F32 | `deepvqe.gguf` | 31 MB | Full precision (reference) |
| Q8_0 | `deepvqe_q8.gguf` | 8.5 MB | 8-bit quantized (73% smaller) |

### Quantization

The Q8_0 variant quantizes encoder, decoder (2-5), and bottleneck weights to
8-bit while keeping precision-sensitive layers at F32: AlignBlock (attention),
dec1 (mask output), and all biases. End-to-end output divergence from F32 is
max 5e-2 / mean 7e-4.

To export your own quantized model:

```bash
make -C train export-q8  # or:
./train/scripts/docker-run.sh python export_ggml.py \
    --checkpoint <path> --quantize --output deepvqe_q8.gguf
```

Compare quantized vs full-precision outputs:

```bash
make test-quantize
```

**Safety note:** Training data was filtered by DNSMOS perceived quality scores,
which can misclassify distressed speech (e.g. screaming, crying) as noise. This
model may attenuate or distort such signals and should not be relied upon for
emergency call or safety-critical applications.

To train your own model and export weights, see [train/](train/).

## Training

All training code lives in [`train/`](train/). It uses Docker with an NVIDIA
NGC PyTorch container. Quick start:

```bash
# Build Docker image and run smoke test
make -C train build
make -C train test

# Train on DNS5 data
make -C train train

# Export trained checkpoint to GGUF
make -C train export
```

See [`train/Makefile`](train/Makefile) for all available targets.

## Dataset Attribution

Model weights are trained on data from the
[ICASSP 2023 Deep Noise Suppression Challenge](https://github.com/microsoft/DNS-Challenge)
(Microsoft, CC BY 4.0).

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE).

## References

- [DeepVQE: Real Time Deep Voice Quality Enhancement](https://arxiv.org/abs/2306.03177) (Indenbom et al., Interspeech 2023)
- [GGML](https://github.com/ggerganov/ggml) tensor library
- [Xiaobin-Rong implementation](https://github.com/Xiaobin-Rong/deepvqe) (NS-only reference)
- [Okrio implementation](https://github.com/Okrio/deepvqe) (AEC reference)
