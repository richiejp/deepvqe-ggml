"""Layer-by-layer comparison between PyTorch and GGML inference.

Registers forward hooks on all PyTorch layers to capture intermediate
activations, then compares against GGML binary dumps.

Usage:
    # 1. Generate PyTorch intermediates
    python ggml/compare.py --checkpoint best.pt --mode pytorch --output intermediates/

    # 2. Run GGML inference with dumps
    ./ggml/deepvqe model.gguf --dump-intermediates

    # 3. Compare
    python ggml/compare.py --mode compare --pytorch-dir intermediates/ --ggml-dir ggml_intermediates/
"""

import argparse
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

# Ensure project root is on the path when running from ggml/ subdir
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.model import DeepVQEAEC
from train import load_checkpoint


def capture_intermediates(model, mic_stft, ref_stft):
    """Run forward pass and capture all intermediate activations.

    Returns:
        output: model output tensor
        intermediates: OrderedDict of {layer_name: tensor}
    """
    intermediates = OrderedDict()
    hooks = []

    def make_hook(name):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                # Store first element for modules that return tuples
                intermediates[name] = output[0].detach().cpu()
            else:
                intermediates[name] = output.detach().cpu()
        return hook_fn

    # Register hooks on all named modules
    for name, module in model.named_modules():
        if name == "":
            continue
        hooks.append(module.register_forward_hook(make_hook(name)))

    model.eval()
    with torch.no_grad():
        output = model(mic_stft, ref_stft)

    # Remove hooks
    for h in hooks:
        h.remove()

    return output, intermediates


def save_intermediates(intermediates, output_dir):
    """Save intermediate activations as .npy files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, tensor in intermediates.items():
        safe_name = name.replace(".", "_")
        np.save(output_dir / f"{safe_name}.npy", tensor.numpy())

    print(f"Saved {len(intermediates)} intermediate activations to {output_dir}")


def load_intermediates(dir_path):
    """Load intermediate activations from .npy files."""
    dir_path = Path(dir_path)
    intermediates = {}
    for f in sorted(dir_path.glob("*.npy")):
        name = f.stem
        intermediates[name] = np.load(f)
    return intermediates


def compare_intermediates(pytorch_dir, ggml_dir):
    """Compare PyTorch and GGML intermediate activations."""
    pt_data = load_intermediates(pytorch_dir)
    ggml_data = load_intermediates(ggml_dir)

    print(f"PyTorch layers: {len(pt_data)}")
    print(f"GGML layers: {len(ggml_data)}")
    print()

    # Find matching layers
    matched = 0
    max_errors = []

    for name in pt_data:
        if name not in ggml_data:
            continue

        pt_arr = pt_data[name]
        ggml_arr = ggml_data[name]

        if pt_arr.shape != ggml_arr.shape:
            print(f"  {name}: SHAPE MISMATCH pt={pt_arr.shape} ggml={ggml_arr.shape}")
            continue

        max_err = np.max(np.abs(pt_arr - ggml_arr))
        mean_err = np.mean(np.abs(pt_arr - ggml_arr))
        max_errors.append((name, max_err, mean_err))
        matched += 1

        status = "OK" if max_err < 1e-4 else "WARN" if max_err < 1e-2 else "FAIL"
        print(f"  [{status}] {name}: max={max_err:.2e} mean={mean_err:.2e}")

    print(f"\nMatched {matched} layers")
    if max_errors:
        worst = max(max_errors, key=lambda x: x[1])
        print(f"Worst layer: {worst[0]} (max error: {worst[1]:.2e})")
        overall_max = max(e[1] for e in max_errors)
        print(f"Overall max error: {overall_max:.2e}")
        if overall_max < 1e-4:
            print("PASS: All layers within f32 tolerance (1e-4)")
        elif overall_max < 1e-2:
            print("WARN: Some layers exceed f32 tolerance, acceptable for f16")
        else:
            print("FAIL: Errors exceed acceptable tolerance")


def generate_pytorch_intermediates(cfg, checkpoint_path, output_dir):
    """Generate and save PyTorch intermediate activations."""
    device = torch.device("cpu")  # Use CPU for deterministic comparison
    model = DeepVQEAEC.from_config(cfg).to(device)
    load_checkpoint(checkpoint_path, model)
    model.eval()

    # Use fixed input for reproducibility
    torch.manual_seed(42)
    mic_stft = torch.randn(1, 257, 20, 2)
    ref_stft = torch.randn(1, 257, 20, 2)

    # Save inputs
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "input_mic_stft.npy", mic_stft.numpy())
    np.save(out_dir / "input_ref_stft.npy", ref_stft.numpy())

    output, intermediates = capture_intermediates(model, mic_stft, ref_stft)

    # Save output
    np.save(out_dir / "output.npy", output.detach().cpu().numpy())

    # Save intermediates
    save_intermediates(intermediates, out_dir)

    print(f"Output shape: {output.shape}")
    print(f"Saved inputs, output, and {len(intermediates)} intermediates to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare PyTorch vs GGML inference")
    parser.add_argument("--mode", choices=["pytorch", "compare"], required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint (for pytorch mode)")
    parser.add_argument("--output", default="intermediates/pytorch",
                        help="Output dir (for pytorch mode)")
    parser.add_argument("--pytorch-dir", default="intermediates/pytorch",
                        help="PyTorch intermediates dir (for compare mode)")
    parser.add_argument("--ggml-dir", default="intermediates/ggml",
                        help="GGML intermediates dir (for compare mode)")
    args = parser.parse_args()

    if args.mode == "pytorch":
        if not args.checkpoint:
            parser.error("--checkpoint required for pytorch mode")
        cfg = load_config(args.config)
        generate_pytorch_intermediates(cfg, args.checkpoint, args.output)
    elif args.mode == "compare":
        compare_intermediates(args.pytorch_dir, args.ggml_dir)
