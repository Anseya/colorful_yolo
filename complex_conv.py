# Colorful-yolo: Complex Convolution Layers
# ==========================================
# Extends standard Conv2d into the complex domain W = a + jb, X = c + jd
# Complex convolution: Y = (ac - bd) + j(ad + bc)
#
# Design decisions:
#   - imag weights initialized via per-channel random rotation (+ random scale & sign)
#     to avoid the channel homogenization inherent in fixed rot90(·)*0.3
#   - SiLU activation applied *separately* to real & imag parts,
#     preserving phase information through the network
#   - output_real=True returns Euclidean magnitude √(real² + imag²)
#     with safe clamp(1e-12) to avoid division-by-zero
#
# References:
#   - Trabelsi et al., "Deep Complex Networks" (ICLR 2018)
#   - Ultralytics YOLO (AGPL-3.0)

import math
import torch
import torch.nn as nn
import numpy as np


def autopad(k, p=None, d=1):
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class ComplexConv(nn.Module):
    """Complex Convolution Layer.

    Extends real-valued Conv2d into the complex domain.

    Mathematical formulation:
        Let W = Wr + j·Wi  (complex weight)
        Let X = Xr + j·Xi  (complex input)
        Y = W * X = (Wr*Xr - Wi*Xi) + j(Wr*Xi + Wi*Xr)

    Key design features:
        1. Per-channel random-rotation imag-weight initialization
           - Each output channel gets a random rotation (0°/90°/180°/270°)
           - Random scale factor [0.15, 0.5] and random sign (±1)
           - This avoids the channel homogenization problem where all
             channels share the same rot90(-)·0.3 transformation

        2. Activation applied separately to real & imag components
           - SiLU(real) and SiLU(imag) independently
           - Preserves phase information: phase = atan2(imag, real)
           - Unlike magnitude-only activation which truncates phase

        3. Safe magnitude output
           - √(max(real² + imag², 1e-12)) avoids div-by-zero
           - No ad-hoc scaling (e.g., /1.414) to match Conv2d range

    Args:
        c1: input channels
        c2: output channels
        k: kernel size
        s: stride
        p: padding (None = auto)
        g: groups
        d: dilation
        act: activation (True = SiLU, nn.Module, or False = Identity)
        init_imag: imag weight init mode ('phase_shift', 'random', 'zero')
        output_real: if True, return magnitude; if False, return complex tensor
    """

    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True,
                 init_imag='phase_shift', output_real=True):
        super().__init__()

        pad = autopad(k, p, d)
        self.conv_real = nn.Conv2d(c1, c2, k, s, pad, groups=g, dilation=d, bias=False)
        self.conv_imag = nn.Conv2d(c1, c2, k, s, pad, groups=g, dilation=d, bias=False)

        self.bn_real = nn.BatchNorm2d(c2)
        self.bn_imag = nn.BatchNorm2d(c2)

        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        self.output_real = output_real
        self.init_imag = init_imag
        self._init_imag_weights(init_imag)

    @property
    def bias(self):
        """Expose conv_real.bias for detector head compatibility."""
        return self.conv_real.bias

    # ─── Imag-weight initialization ───────────────────────────────────────

    def _init_imag_weights(self, mode='phase_shift'):
        """Initialize imag-part weights.

        'phase_shift' (default):
            Each output channel gets a random rotation angle from {0°, 90°, 180°, 270°},
            a random scale factor ∈ [0.15, 0.5], and a random sign (±1).
            This creates diverse channel-wise phase relationships,
            avoiding the homogeneous rot90(·)*0.3 pattern.

        'random':   Independent kaiming_normal_ init (may cause magnitude explosion).
        'zero':     All zeros → degrades to real-valued convolution.
        """
        with torch.no_grad():
            nn.init.kaiming_normal_(self.conv_real.weight, mode='fan_out', nonlinearity='relu')

            if mode == 'phase_shift':
                w_real = self.conv_real.weight.data  # (c2, c1, kH, kW)
                c2, c1, kh, kw = w_real.shape

                # Per-channel random rotation {0, 1, 2, 3} * 90°
                k_vals = torch.randint(0, 4, (c2,), device=w_real.device)
                # Per-channel random scale ∈ [0.15, 0.5]
                scale_vals = 0.15 + 0.35 * torch.rand(c2, device=w_real.device)
                # Per-channel random sign ±1
                sign_vals = torch.where(
                    torch.rand(c2, device=w_real.device) > 0.5,
                    torch.ones(c2, device=w_real.device),
                    -torch.ones(c2, device=w_real.device),
                )

                w_imag = torch.zeros_like(w_real)
                for i in range(c2):
                    k = k_vals[i].item()
                    s = scale_vals[i].item()
                    sg = sign_vals[i].item()
                    w_ch = w_real[i] * s * sg
                    if k == 0:
                        w_imag[i] = w_ch
                    else:
                        w_imag[i] = torch.rot90(w_ch, k=k, dims=(-2, -1))

                self.conv_imag.weight.data = w_imag

            elif mode == 'random':
                nn.init.kaiming_normal_(self.conv_imag.weight, mode='fan_out', nonlinearity='relu')

            elif mode == 'zero':
                nn.init.zeros_(self.conv_imag.weight)

            else:
                # Fallback: rot90 * 0.3
                w_real = self.conv_real.weight.data
                if kh == kw and kh % 2 == 1:
                    w_rotated = torch.rot90(w_real, k=1, dims=(-2, -1))
                else:
                    w_rotated = w_real.flip([-2])
                self.conv_imag.weight.data = w_rotated * 0.3

    # ─── Forward ──────────────────────────────────────────────────────────

    def forward(self, x):
        """Apply complex convolution.

        Args:
            x: real tensor [B, C, H, W] or complex tensor

        Returns:
            magnitude (if output_real=True) or complex tensor
        """
        if x.is_complex():
            x_real, x_imag = x.real, x.imag
        else:
            x_real, x_imag = x, torch.zeros_like(x)

        # Complex convolution: (a+jb)*(c+jd) = (ac-bd) + j(ad+bc)
        real_out = self.conv_real(x_real) - self.conv_imag(x_imag)
        imag_out = self.conv_real(x_imag) + self.conv_imag(x_real)

        # BatchNorm
        real_out = self.bn_real(real_out)
        imag_out = self.bn_imag(imag_out)

        # Activation applied to real & imag separately (preserves phase)
        if not isinstance(self.act, nn.Identity):
            real_out = self.act(real_out)
            imag_out = self.act(imag_out)

        # Build complex output
        out_complex = torch.complex(real_out, imag_out)

        if self.output_real:
            # Safe magnitude with clamp to avoid div-by-zero in downstream
            magnitude = torch.sqrt(torch.clamp(real_out.pow(2) + imag_out.pow(2), min=1e-12))
            return magnitude

        return out_complex

    def forward_fuse(self, x):
        """Forward without BatchNorm (for model fusion / export)."""
        if x.is_complex():
            x_real, x_imag = x.real, x.imag
        else:
            x_real, x_imag = x, torch.zeros_like(x)

        real_out = self.conv_real(x_real) - self.conv_imag(x_imag)
        imag_out = self.conv_real(x_imag) + self.conv_imag(x_real)

        if not isinstance(self.act, nn.Identity):
            real_out = self.act(real_out)
            imag_out = self.act(imag_out)

        return torch.complex(real_out, imag_out)

    def get_phase_info(self):
        """Return phase statistics for diagnostic purposes."""
        with torch.no_grad():
            w_real = self.conv_real.weight.data.flatten()
            w_imag = self.conv_imag.weight.data.flatten()
            phase = torch.atan2(w_imag, w_real)
            return {
                'phase_mean': phase.mean().item(),
                'phase_std': phase.std().item(),
                'magnitude_mean': torch.sqrt(w_real**2 + w_imag**2).mean().item(),
                'real_norm': w_real.norm().item(),
                'imag_norm': w_imag.norm().item(),
            }


# ═══════════════════════════════════════════════════════════════════════════
# Complex Conv Variants
# ═══════════════════════════════════════════════════════════════════════════

class ComplexConv2(ComplexConv):
    """Complex Conv2: parallel 3×3 + 1×1 branches.

    Inherits from ComplexConv and adds a 1×1 complex convolution branch cv2.
    Output = act(main_branch + cv2_branch).
    """

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True):
        super().__init__(c1, c2, k, s, p, g, d, act)
        self.cv2 = ComplexConv(c1, c2, 1, s, autopad(1, p, d), g=g, d=d)

    def forward(self, x):
        main_out = super().forward(x)

        if x.is_complex():
            x_real, x_imag = x.real, x.imag
        else:
            x_real, x_imag = x, torch.zeros_like(x)

        real_cv2 = self.cv2.conv_real(x_real) - self.cv2.conv_imag(x_imag)
        imag_cv2 = self.cv2.conv_real(x_imag) + self.cv2.conv_imag(x_real)
        real_cv2 = self.cv2.bn_real(real_cv2)
        imag_cv2 = self.cv2.bn_imag(imag_cv2)

        if not isinstance(self.cv2.act, nn.Identity):
            real_cv2 = self.cv2.act(real_cv2)
            imag_cv2 = self.cv2.act(imag_cv2)

        cv2_out = torch.sqrt(torch.clamp(real_cv2.pow(2) + imag_cv2.pow(2), min=1e-12))
        return self.act(main_out + cv2_out)

    def forward_fuse(self, x):
        main_out = super().forward_fuse(x)

        if x.is_complex():
            x_real, x_imag = x.real, x.imag
        else:
            x_real, x_imag = x, torch.zeros_like(x)

        real_cv2 = self.cv2.conv_real(x_real) - self.cv2.conv_imag(x_imag)
        imag_cv2 = self.cv2.conv_real(x_imag) + self.cv2.conv_imag(x_real)
        cv2_out = torch.sqrt(torch.clamp(real_cv2.pow(2) + imag_cv2.pow(2), min=1e-12))
        return main_out + cv2_out


class ComplexDWConv(ComplexConv):
    """Depth-wise Complex Convolution.

    Sets groups = gcd(c1, c2) for channel-wise complex convolution.
    """

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


class ComplexLightConv(nn.Module):
    """Lightweight Complex Convolution: 1×1 ComplexConv + ComplexDWConv.

    Used for efficient channel adjustment and spatial feature extraction.
    """

    def __init__(self, c1, c2, k=1, act=None):
        super().__init__()
        act = act if act is not None else nn.ReLU()
        self.conv1 = ComplexConv(c1, c2, 1, act=False)
        self.conv2 = ComplexDWConv(c2, c2, k, act=act)

    def forward(self, x):
        return self.conv2(self.conv1(x))


class ComplexGhostConv(nn.Module):
    """Complex Ghost Convolution.

    Generates primary features via standard complex conv,
    then produces 'cheap' features via depth-wise complex conv.
    Reduces parameter count while maintaining expressiveness.
    """

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = ComplexConv(c1, c_, k, s, None, g, act=act)
        self.cv2 = ComplexConv(c_, c_, 5, 1, None, c_, act=act)

    def forward(self, x):
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)
