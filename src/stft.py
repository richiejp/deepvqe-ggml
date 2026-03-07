import torch


def make_window(n_fft, device=None):
    """Create sqrt-Hann window for perfect reconstruction."""
    return torch.sqrt(torch.hann_window(n_fft, device=device) + 1e-12)


def stft(x, n_fft=512, hop_length=256):
    """Compute STFT with sqrt-Hann window.

    Args:
        x: (B, N) waveform
        n_fft: FFT size
        hop_length: hop size

    Returns:
        (B, F, T, 2) where F = n_fft//2 + 1, last dim is (real, imag)
    """
    window = make_window(n_fft, device=x.device)
    X = torch.stft(x, n_fft, hop_length, window=window, return_complex=True)
    return torch.view_as_real(X)  # (B, F, T, 2)


def istft(X, n_fft=512, hop_length=256, length=None):
    """Compute inverse STFT with sqrt-Hann window.

    Args:
        X: (B, F, T, 2) STFT with (real, imag)
        n_fft: FFT size
        hop_length: hop size
        length: desired output length

    Returns:
        (B, N) waveform
    """
    window = make_window(n_fft, device=X.device)
    X_complex = torch.complex(X[..., 0], X[..., 1])
    return torch.istft(X_complex, n_fft, hop_length, window=window, length=length)
