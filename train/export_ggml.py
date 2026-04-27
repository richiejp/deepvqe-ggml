"""Export DeepVQE model weights to GGUF format for GGML inference.

Features:
- BatchNorm folding into preceding Conv2d
- GRU weight decomposition for manual recurrence
- GGUF format via gguf Python package
- Optional verification of BN folding correctness
"""

import argparse
import copy

import numpy as np
import torch
import torch.nn as nn

import gguf
from gguf.constants import GGMLQuantizationType
from gguf.quants import quantize as gguf_quantize

from src.config import load_config
from src.model import DeepVQEAEC
from utils import load_checkpoint


class _ChannelAffine(nn.Module):
    """Channel-wise affine transform: y = x * scale + bias.

    Used to replace BatchNorm after SubpixelConv2d where BN can't be
    folded into the preceding conv due to the channel reshape.
    """

    def __init__(self, scale, bias):
        super().__init__()
        self.register_buffer("scale", scale)
        self.register_buffer("bias", bias)

    def forward(self, x):
        return x * self.scale[None, :, None, None] + self.bias[None, :, None, None]


def should_quantize(name: str) -> bool:
    """Decide if a tensor should be Q8_0 quantized.

    Keep FP32: biases, ChannelAffine (bn.scale/bn.bias), AlignBlock, dec1.
    Quantize: encoder/decoder conv weights, bottleneck GRU/FC weights.
    """
    if name.endswith(".bias"):
        return False
    if ".bn." in name:
        return False
    if name.startswith("align."):
        return False
    if name.startswith("dec1."):
        return False
    if name.endswith(".weight") or "weight_ih" in name or "weight_hh" in name:
        return True
    return False


def fold_bn_into_conv(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    """Fold BatchNorm parameters into Conv2d weights and bias.

    Mathematically equivalent in eval mode:
        BN(Conv(x)) = gamma/sigma * (Conv(x) - mu) + beta
                     = gamma/sigma * Conv(x) + (beta - gamma*mu/sigma)
                     = Conv'(x)  where W' = gamma/sigma * W, b' = gamma/sigma * b + beta - gamma*mu/sigma

    Returns a new Conv2d with folded weights.
    """
    bn.eval()
    conv.eval()

    # Extract BN parameters
    gamma = bn.weight.data  # (C,)
    beta = bn.bias.data  # (C,)
    mu = bn.running_mean  # (C,)
    var = bn.running_var  # (C,)
    eps = bn.eps

    sigma = torch.sqrt(var + eps)  # (C,)
    scale = gamma / sigma  # (C,)

    # Create new conv with folded parameters
    folded = copy.deepcopy(conv)
    folded.eval()

    # W' = scale[:, None, None, None] * W
    folded.weight.data = conv.weight.data * scale[:, None, None, None]

    # b' = scale * (b - mu) + beta
    if conv.bias is not None:
        folded.bias.data = scale * (conv.bias.data - mu) + beta
    else:
        folded.bias = nn.Parameter(scale * (-mu) + beta)

    return folded


def fold_model_batchnorms(model: DeepVQEAEC) -> dict:
    """Fold all BatchNorm layers into their preceding Conv2d layers.

    Returns a state dict with BN-folded weights (no BN keys).
    """
    model = copy.deepcopy(model)
    model.eval()

    # EncoderBlock: conv + bn, resblock.conv + resblock.bn
    encoder_blocks = [
        model.mic_enc1, model.mic_enc2,
        model.mic_enc3, model.mic_enc4, model.mic_enc5,
        model.far_enc1, model.far_enc2,
    ]

    for enc in encoder_blocks:
        # Fold main conv + bn
        enc.conv = fold_bn_into_conv(enc.conv, enc.bn)
        enc.bn = nn.Identity()

        # Fold residual block conv + bn
        enc.resblock.conv = fold_bn_into_conv(enc.resblock.conv, enc.resblock.bn)
        enc.resblock.bn = nn.Identity()

    # DecoderBlocks: resblock.conv + resblock.bn, and optional bn after deconv
    decoder_blocks = [model.dec5, model.dec4, model.dec3, model.dec2, model.dec1]
    for dec in decoder_blocks:
        # Fold residual block
        dec.resblock.conv = fold_bn_into_conv(dec.resblock.conv, dec.resblock.bn)
        dec.resblock.bn = nn.Identity()

        # Decoder BN follows SubpixelConv2d (which reshapes channels),
        # so we can't fold it into the preceding conv. Instead, collapse
        # the BN into a channel-wise scale+bias (affine transform).
        if hasattr(dec, "bn"):
            bn = dec.bn
            bn.eval()
            sigma = torch.sqrt(bn.running_var + bn.eps)
            scale = bn.weight.data / sigma
            bias = bn.bias.data - bn.weight.data * bn.running_mean / sigma
            affine = _ChannelAffine(scale, bias)
            dec.bn = affine

    return model


def verify_bn_folding(original_model, folded_model, device="cpu"):
    """Verify that BN-folded model produces identical output."""
    original_model.eval().to(device)
    folded_model.eval().to(device)

    with torch.no_grad():
        mic = torch.randn(1, 257, 20, 2, device=device)
        ref = torch.randn(1, 257, 20, 2, device=device)

        out_orig = original_model(mic, ref)
        out_folded = folded_model(mic, ref)

        max_err = (out_orig - out_folded).abs().max().item()
        return max_err


def export_gguf(model: DeepVQEAEC, cfg, output_path: str, fold_bn=True,
                quantize=False):
    """Export model to GGUF format.

    Args:
        model: Trained DeepVQEAEC model
        cfg: Config object
        output_path: Output .gguf file path
        fold_bn: Whether to fold BatchNorm into Conv2d
        quantize: Whether to apply Q8_0 quantization to robust layers
    """
    model.eval()

    if fold_bn:
        export_model = fold_model_batchnorms(model)
        max_err = verify_bn_folding(model, export_model)
        print(f"BN folding verification: max error = {max_err:.2e}")
        if max_err > 1e-4:
            print("WARNING: BN folding error is high!")
    else:
        export_model = copy.deepcopy(model)
        export_model.eval()

    writer = gguf.GGUFWriter(output_path, arch="deepvqe")

    # Metadata
    writer.add_uint32("deepvqe.version", 1)
    writer.add_uint32("deepvqe.n_fft", cfg.audio.n_fft)
    writer.add_uint32("deepvqe.hop_length", cfg.audio.hop_length)
    writer.add_uint32("deepvqe.n_freq_bins", cfg.audio.n_freqs)
    writer.add_uint32("deepvqe.sample_rate", cfg.audio.sample_rate)
    writer.add_uint32("deepvqe.dmax", cfg.model.dmax)
    writer.add_float32("deepvqe.power_law_c", cfg.model.power_law_c)
    writer.add_uint32("deepvqe.align_hidden", cfg.model.align_hidden)
    writer.add_bool("deepvqe.bn_folded", fold_bn)
    writer.add_bool("deepvqe.quantized", quantize)
    # AlignBlock softmax temperature — read from the live model so the
    # exported value reflects whatever the buffer holds (annealed end value
    # after training, or the constructor default for a fresh model).
    align_t = float(export_model.align.temperature.detach().cpu().item())
    writer.add_float32("deepvqe.align_temperature", align_t)

    # Channel configs as arrays
    for i, ch in enumerate(cfg.model.mic_channels):
        writer.add_uint32(f"deepvqe.mic_channels.{i}", ch)
    writer.add_uint32("deepvqe.mic_channels.count", len(cfg.model.mic_channels))

    for i, ch in enumerate(cfg.model.far_channels):
        writer.add_uint32(f"deepvqe.far_channels.{i}", ch)
    writer.add_uint32("deepvqe.far_channels.count", len(cfg.model.far_channels))

    # Export tensors
    state_dict = export_model.state_dict()
    n_skipped = 0
    n_exported = 0
    n_quantized = 0

    for name, tensor in state_dict.items():
        # Skip Identity (folded BN) remnants and non-parameter buffers
        # that are BN running stats
        if "running_mean" in name or "running_var" in name or "num_batches_tracked" in name:
            n_skipped += 1
            continue
        # align.temperature is a 0-d buffer — exported as GGUF metadata above
        # (deepvqe.align_temperature), not as a tensor.
        if name == "align.temperature":
            n_skipped += 1
            continue

        np_tensor = tensor.detach().cpu().to(torch.float32).numpy()

        if quantize and should_quantize(name):
            # Q8_0 requires innermost dim (ne[0] in GGML) to be a multiple of
            # block size 32.  Conv weights have ne[0]=kW=3, so we flatten to
            # 1D before quantizing and store the original shape as metadata.
            orig_shape = list(np_tensor.shape)
            n_elem = np_tensor.size
            assert n_elem % 32 == 0, (
                f"Tensor {name} has {n_elem} elements, not a multiple of 32"
            )
            flat = np_tensor.reshape(-1)
            q8_data = gguf_quantize(
                flat.reshape(-1, 32), GGMLQuantizationType.Q8_0
            )
            writer.add_tensor(
                name, q8_data.reshape(-1),
                raw_dtype=GGMLQuantizationType.Q8_0,
            )
            # Store original shape so C++ loader can restore it
            for d, s in enumerate(orig_shape):
                writer.add_uint32(f"deepvqe.shape.{name}.{d}", s)
            writer.add_uint32(
                f"deepvqe.shape.{name}.ndim", len(orig_shape)
            )
            n_quantized += 1
        else:
            writer.add_tensor(name, np_tensor)
            n_exported += 1

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    print(f"Exported {n_exported} tensors (F32), {n_quantized} tensors (Q8_0), "
          f"skipped {n_skipped} BN running stats")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export DeepVQE to GGUF")
    parser.add_argument("--config", default="configs/default.yaml", help="Config file")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint path")
    parser.add_argument("--output", default="deepvqe.gguf", help="Output GGUF file")
    parser.add_argument("--no-fold-bn", action="store_true", help="Skip BN folding")
    parser.add_argument("--quantize", action="store_true",
                        help="Quantize robust layers to Q8_0 (keeps align/dec1/biases as F32)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model = DeepVQEAEC.from_config(cfg)
    load_checkpoint(args.checkpoint, model)
    print(f"Loaded checkpoint: {args.checkpoint}")

    export_gguf(model, cfg, args.output, fold_bn=not args.no_fold_bn,
                quantize=args.quantize)
