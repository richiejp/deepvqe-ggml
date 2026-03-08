"""Model verification tests for DeepVQEAEC.

Checks:
- Forward pass shape
- Parameter count
- Causality (future frames don't affect past output)
- All parameters receive gradients
- AlignBlock delay distribution sums to 1.0
- AMP compatibility
- Peak memory estimate
"""

import torch

from src.model import DeepVQEAEC


def test_forward_shape():
    model = DeepVQEAEC().eval()
    B, F, T = 2, 257, 63
    mic = torch.randn(B, F, T, 2)
    ref = torch.randn(B, F, T, 2)
    out = model(mic, ref)
    assert out.shape == (B, F, T, 2), f"Expected ({B},{F},{T},2), got {out.shape}"
    print(f"[PASS] Forward shape: {out.shape}")


def test_param_count():
    model = DeepVQEAEC()
    n = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Parameters: {n:,}")
    assert 7_000_000 < n < 9_000_000, f"Expected 7-9M params, got {n:,}"
    print(f"[PASS] Parameter count: {n:,}")


def test_causality():
    model = DeepVQEAEC().eval()
    a_mic = torch.randn(1, 257, 50, 2)
    b_mic = torch.randn(1, 257, 50, 2)
    c_mic = torch.randn(1, 257, 50, 2)
    a_ref = torch.randn(1, 257, 50, 2)
    b_ref = torch.randn(1, 257, 50, 2)
    c_ref = torch.randn(1, 257, 50, 2)

    x1_mic = torch.cat([a_mic, b_mic], dim=2)
    x2_mic = torch.cat([a_mic, c_mic], dim=2)
    x1_ref = torch.cat([a_ref, b_ref], dim=2)
    x2_ref = torch.cat([a_ref, c_ref], dim=2)

    y1 = model(x1_mic, x1_ref)
    y2 = model(x2_mic, x2_ref)

    causal_diff = (y1[:, :, :50, :] - y2[:, :, :50, :]).abs().max().item()
    future_diff = (y1[:, :, 50:, :] - y2[:, :, 50:, :]).abs().max().item()

    assert causal_diff == 0.0, f"Causality violated: past diff = {causal_diff:.2e}"
    assert future_diff > 0.0, "Future frames should differ"
    print(f"[PASS] Causality: past diff={causal_diff:.2e}, future diff={future_diff:.2e}")


def test_gradients():
    model = DeepVQEAEC().train()
    mic = torch.randn(1, 257, 32, 2)
    ref = torch.randn(1, 257, 32, 2)
    out = model(mic, ref)
    loss = out.sum()
    loss.backward()

    no_grad = []
    for name, p in model.named_parameters():
        if p.grad is None or p.grad.abs().max() == 0:
            no_grad.append(name)

    if no_grad:
        print(f"[FAIL] Parameters with no gradient: {no_grad}")
    else:
        print(f"[PASS] All {sum(1 for _ in model.parameters())} parameters receive gradients")
    assert len(no_grad) == 0, f"Parameters without gradients: {no_grad}"


def test_delay_distribution():
    model = DeepVQEAEC().eval()
    mic = torch.randn(1, 257, 32, 2)
    ref = torch.randn(1, 257, 32, 2)
    _, delay, mask_raw = model(mic, ref, return_delay=True)

    assert delay.shape == (1, 32, 32), f"Expected (1,32,32), got {delay.shape}"
    sums = delay.sum(-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5), \
        f"Delay distribution doesn't sum to 1: {sums}"
    assert (delay >= 0).all(), "Delay distribution has negative values"
    print(f"[PASS] Delay distribution: shape={delay.shape}, sums to 1, non-negative")


def test_amp():
    model = DeepVQEAEC().train()
    mic = torch.randn(1, 257, 32, 2)
    ref = torch.randn(1, 257, 32, 2)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        out = model(mic, ref)
        loss = out.sum()

    loss.backward()
    assert not torch.isnan(out).any(), "NaN in AMP output"
    assert not torch.isinf(out).any(), "Inf in AMP output"
    print(f"[PASS] AMP: no NaN/Inf, output dtype={out.dtype}")


def test_peak_memory():
    """Estimate peak memory for B=1 T=188 (CPU-only, just check it runs)."""
    model = DeepVQEAEC().eval()
    mic = torch.randn(1, 257, 188, 2)
    ref = torch.randn(1, 257, 188, 2)
    out = model(mic, ref)
    assert out.shape == (1, 257, 188, 2)
    print(f"[PASS] Peak memory test: B=1 T=188 runs successfully")


if __name__ == "__main__":
    tests = [
        test_forward_shape,
        test_param_count,
        test_causality,
        test_gradients,
        test_delay_distribution,
        test_amp,
        test_peak_memory,
    ]
    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"[FAIL] {test.__name__}: {e}")
            failed += 1
    print(f"\n{'='*40}")
    print(f"Results: {passed} passed, {failed} failed")
