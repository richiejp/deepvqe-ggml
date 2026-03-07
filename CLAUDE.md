# DeepVQE Project Guidelines

## Project Goal
Train DeepVQE (Indenbom et al., Interspeech 2023) for joint AEC/NS/DR with
a focus on acoustic echo cancellation with soft delay estimation. Deploy via
GGML C++ inference.

## Important Rules
- **Keep README.md up to date** whenever you make structural changes, complete
  checklist items, add new files, or change the architecture/approach.
- Use the real-valued CCM implementation (no `torch.complex`) for GGML compatibility.
- All convolutions must be causal (pad top/left only, no look-ahead).
- Prefer `einops` for tensor reshaping.
- Target 16 kHz, 512 FFT, 256 hop, 257 freq bins.
- Use mixed precision (AMP) for training.
- Log extensively: loss components, gradient norms, delay distributions, audio samples.

## Architecture Quick Reference
- Encoder: 5 mic blocks (2→64→128→128→128→128), 2 far-end (2→32→128)
- AlignBlock: cross-attention soft delay after encoder stage 2
- Encoder block 3 takes 256 input channels (concat of mic + aligned far-end)
- Bottleneck: GRU(1152→576) + Linear(576→1152)
- Decoder: 5 blocks with sub-pixel conv (128,128,128,64,27)
- CCM: 27 channels → 3×3 complex convolving mask (real-valued arithmetic)

## Code Sources
See `reference/` directory and README.md "References" section for details.
Third-party reference implementations and known bugs are documented there.
