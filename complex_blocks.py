# Colorful-yolo: Complex Block Modules
# =====================================
# Complex-valued building blocks for YOLO-style architectures.
# Internally use ComplexConv (aliased as Conv) for all convolutions.
#
# References:
#   - Ultralytics YOLO (AGPL-3.0)

import torch
import torch.nn as nn
import torch.nn.functional as F

from .complex_conv import ComplexConv

# Alias: all Conv() calls in this module actually create ComplexConv instances
Conv = ComplexConv


class ComplexDFL(nn.Module):
    """Complex Distribution Focal Loss integral module.

    For complex input: takes magnitude first, then applies standard DFL.
    DFL converts softmax-normalized distribution over reg_max+1 bins
    into a single scalar per anchor via weighted sum.

    Args:
        c1: number of input channels (= reg_max + 1, default 16)
    """

    def __init__(self, c1: int = 16):
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Magnitude for complex input
        if x.is_complex():
            x = torch.abs(x)

        if x.dim() == 4:
            b, c, h, w = x.shape
            return self.conv(
                x.view(b, 4, self.c1, h * w).transpose(2, 1).softmax(1)
            ).view(b, 4, h * w)

        b, _, a = x.shape
        return self.conv(
            x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)
        ).view(b, 4, a)


class ComplexBottleneck(nn.Module):
    """Complex-valued residual bottleneck block.

    Structure: ComplexConv(1×1, e·c2) → SiLU → ComplexConv(3×3, c2) → SiLU
    With optional residual shortcut when c1 == c2.

    Internal convolutions use ComplexConv (via the Conv alias).

    Args:
        c1: input channels
        c2: output channels
        shortcut: whether to add residual connection
        g: groups for the second conv
        k: kernel sizes (k1, k2) default (3, 3)
        e: expansion ratio (default 0.5)
    """

    def __init__(self, c1: int, c2: int, shortcut: bool = True,
                 g: int = 1, k: tuple = (3, 3), e: float = 0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)      # 1×1 or 3×3 complex conv
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)  # 3×3 complex conv
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional residual connection."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class ComplexSPPF(nn.Module):
    """Complex Spatial Pyramid Pooling - Fast.

    Multi-scale feature fusion using max-pooling with kernel_size=5.
    For complex inputs: max_pool operates on magnitude (|z|).

    Structure:
        cv1: ComplexConv(c1 → c1//2, 1×1, no act)
        pool: MaxPool2d(5×5) × n rounds
        cv2: ComplexConv((n+1)·c_ → c2, 1×1)

    Args:
        c1: input channels
        c2: output channels
        k: kernel size for max pooling
        n: number of pooling rounds
    """

    def __init__(self, c1: int, c2: int, k: int = 5, n: int = 3):
        super().__init__()
        c_ = c1 // 2  # intermediate channels
        self.cv1 = Conv(c1, c_, 1, 1, act=False)   # 1×1 complex conv
        self.cv2 = Conv(c_ * (n + 1), c2, 1, 1)    # fusion conv
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.n = n

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with complex-aware pooling."""
        y = [self.cv1(x)]
        for _ in range(self.n):
            last = y[-1]
            # For complex tensor, pool on magnitude
            pooled = self.m(torch.abs(last)) if last.is_complex() else self.m(last)
            y.append(pooled)
        return self.cv2(torch.cat(y, 1))


class ComplexC3k2(nn.Module):
    """Complex C3k2: CSP Bottleneck with 2 convolutions.

    Cross Stage Partial network with ComplexConv throughout.

    Structure:
        cv1: ComplexConv(c1 → 2·c, 1×1)  → split into two halves
        m:   n × ComplexBottleneck
        cv2: ComplexConv((2+n)·c → c2, 1×1)

    Args:
        c1: input channels
        c2: output channels
        n: number of bottleneck blocks
        e: hidden channel ratio (default 0.5)
        g: groups
        shortcut: whether bottleneck uses residual
    """

    def __init__(self, c1: int, c2: int, n: int = 1,
                 e: float = 0.5, g: int = 1, shortcut: bool = True):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)        # input → 2 halves
        self.cv2 = Conv((2 + n) * self.c, c2, 1)     # concat → output
        self.m = nn.ModuleList(
            ComplexBottleneck(self.c, self.c, shortcut, g) for _ in range(n)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward: cv1 → split → bottlenecks → concat → cv2."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
