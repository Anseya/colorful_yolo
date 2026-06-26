# Colorful-yolo: RGB → HSV Polar Decomposition
# ==============================================
# Image preprocessing that decomposes RGB into orthogonal complex channels.
#
# Core idea:
#   Real part = RGB (spatial structure)
#   Imag part = HSV in polar coordinates (color/hue information)
#
# The HSV polar decomposition maps:
#   S (saturation) → radius r
#   H (hue)        → angle θ = H · 2π
#   → imag = [S·cos(θ), S·sin(θ), S]
#
# Why this works:
#   - Hue is a periodic variable (0 ≡ 2π). The cos/sin transform
#     naturally handles the circular boundary without discontinuity.
#   - Saturation acts as the "confidence" of hue — gray pixels (S≈0)
#     naturally map to imag≈0, preventing noise injection.
#   - V (value/brightness) is already encoded in Real[RGB],
#     so it stays out of the complex channel to avoid redundancy.
#
# Mathematical justification:
#   For a pixel with color c = (R,G,B) in [0,1]³:
#     V = max(R,G,B)
#     S = (V - min(R,G,B)) / V     (with V>0 guard)
#     H = piecewise function of (R,G,B) differences
#   The imaginary component is:
#     imag = [S·cos(2πH), S·sin(2πH), S]
#   This is an embedding of the hue circle into ℝ³, with S as confidence weight.
#
# References:
#   - Smith, "Color Gamut Transform Pairs" (SIGGRAPH 1978) — HSV color space
#   - Trabelsi et al., "Deep Complex Networks" (ICLR 2018)

import torch
import numpy as np


def rgb_to_hsv_polar_imag(x: torch.Tensor) -> torch.Tensor:
    """Convert RGB tensor to HSV polar imaginary component.

    Args:
        x: RGB tensor of shape (B, 3, H, W) with values in [0, 1]

    Returns:
        imag tensor of shape (B, 3, H, W):
            channel 0 = S·cos(H·2π)
            channel 1 = S·sin(H·2π)
            channel 2 = S
    """
    # Extract RGB channels
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]

    # Compute Value (max) and chroma range (delta)
    mx, _ = torch.max(torch.cat([r, g, b], dim=1), dim=1, keepdim=True)
    mn, _ = torch.min(torch.cat([r, g, b], dim=1), dim=1, keepdim=True)
    delta = mx - mn + 1e-8  # epsilon guard for gray pixels

    # ── Hue computation (piecewise) ──
    # H ∈ [0, 1] mapping
    h = torch.zeros_like(r)

    # R is max
    mask_r = (mx == r)
    h = torch.where(mask_r, ((g - b) / delta) % 6, h)

    # G is max
    mask_g = (mx == g)
    h = torch.where(mask_g, (b - r) / delta + 2, h)

    # B is max
    mask_b = (mx == b)
    h = torch.where(mask_b, (r - g) / delta + 4, h)

    h = h / 6.0  # normalize to [0, 1]

    # ── Saturation computation ──
    # S = 0 for black/gray pixels (V ≈ 0)
    s = torch.where(mx < 1e-8, torch.zeros_like(mx), delta / (mx + 1e-8))

    # ── Polar coordinate embedding ──
    # z = S · exp(j · 2πH)  →  real = S·cos(2πH), imag = S·sin(2πH)
    angle = h * 2 * np.pi

    imag_ch0 = s * torch.cos(angle)   # S · cos(2πH)
    imag_ch1 = s * torch.sin(angle)   # S · sin(2πH)
    imag_ch2 = s                      # S itself

    return torch.cat([imag_ch0, imag_ch1, imag_ch2], dim=1)


def rgb_to_polar_complex(x: torch.Tensor, scale_real: float = 0.25, scale_imag: float = 0.6):
    """Convert normalized RGB to full complex tensor.

    This variant returns torch.complex directly, suitable for feeding
    into ComplexConv that expects complex input.

    Args:
        x: RGB tensor of shape (B, 3, H, W), ImageNet-normalized
        scale_real: divisor for real part normalization (default 0.25)
        scale_imag: divisor for imag part normalization (default 0.6)

    Returns:
        complex tensor of shape (B, 3, H, W)
            real = (RGB - 0.5) / scale_real
            imag = hsv_polar_imag / scale_imag
    """
    # De-normalize to [0, 1]
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
    rgb = torch.clamp(x * std + mean, 0, 1)

    # Real part: centered RGB
    x_real = (rgb - 0.5) / scale_real

    # Imag part: HSV polar decomposition
    x_imag = rgb_to_hsv_polar_imag(rgb) / scale_imag

    return torch.complex(x_real, x_imag)
