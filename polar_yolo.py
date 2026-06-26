# Colorful-yolo: Polar Complex YOLO26 Classification Model
# ==========================================================
# YOLO-style backbone with complex-valued convolutions,
# fed with RGB (real) + HSV polar decomposition (imag).
#
# Architecture:
#   Input: RGB → split into Real[RGB] + Imag[S·cos(H), S·sin(H), S]
#   Stem:   2 × ComplexConv (downsampling 224→112→56)
#   Stage1: downConv + 2 × ComplexBottleneck  (128ch, 56→28)
#   Stage2: downConv + 2 × ComplexBottleneck  (256ch, 28→14)
#   Stage3: downConv + 4 × ComplexBottleneck  (512ch, 14→7)
#   SPPF:   1×1→pool×3→1×1  multi-scale fusion
#   Head:   4-layer MLP  (512→512→256→128→num_classes)

import torch
import torch.nn as nn
import numpy as np

from .complex_conv import ComplexConv
from .image_preprocess import rgb_to_hsv_polar_imag


class PolarComplexYOLO(nn.Module):
    """Colorful-yolo: Complex-valued YOLO26 for RGB image classification.

    The key innovation is feeding the network with a complex-valued input:
        Real = RGB (spatial structure)
        Imag = HSV polar decomposition  (color/hue information)

    This enables ComplexConv to learn cross-interactions between
    spatial features (real) and color-phase features (imag).

    Architecture details:
        - Backbone: YOLO26-style with 3 stages + SPPF
        - Stage3 deepened to 4 bottleneck blocks for stronger features
        - Total ~48.6M parameters (roughly 2× the real-valued counterpart
          due to dual weight sets in ComplexConv)
        - SiLU activation on real & imag separately preserves phase information

    Args:
        num_classes: number of output classes (default 10)
        init_imag: imag-weight initialization mode ('phase_shift' default)

    Usage:
        model = PolarComplexYOLO(num_classes=10)
        x = torch.randn(2, 3, 224, 224)  # ImageNet-normalized RGB
        logits = model(x)                 # (2, 10)
    """

    def __init__(self, num_classes: int = 10, init_imag: str = 'phase_shift'):
        super().__init__()
        self.num_classes = num_classes
        CC = lambda c1, c2, k, s, *a, **kw: ComplexConv(c1, c2, k, s, *a,
            init_imag=init_imag, **kw)

        # ═══════════════════════════════════════════════════════════════
        # Stem: 3×224² → 64×56²
        # ═══════════════════════════════════════════════════════════════
        self.stem_c1 = CC(3, 32, 3, 2, 1)
        self.stem_bn1 = nn.BatchNorm2d(32)
        self.stem_act1 = nn.SiLU()
        self.stem_c2 = CC(32, 64, 3, 2, 1)
        self.stem_bn2 = nn.BatchNorm2d(64)
        self.stem_act2 = nn.SiLU()

        # ═══════════════════════════════════════════════════════════════
        # Stage 1: 64×56² → 128×28²  (2 bottlenecks)
        # ═══════════════════════════════════════════════════════════════
        self.s1_down = CC(64, 128, 3, 2, 1)
        self.s1_down_bn = nn.BatchNorm2d(128)
        self.s1_down_act = nn.SiLU()
        self._build_bottleneck_group('s1', 128, 2)

        # ═══════════════════════════════════════════════════════════════
        # Stage 2: 128×28² → 256×14²  (2 bottlenecks)
        # ═══════════════════════════════════════════════════════════════
        self.s2_down = CC(128, 256, 3, 2, 1)
        self.s2_down_bn = nn.BatchNorm2d(256)
        self.s2_down_act = nn.SiLU()
        self._build_bottleneck_group('s2', 256, 2)

        # ═══════════════════════════════════════════════════════════════
        # Stage 3: 256×14² → 512×7²  (4 bottlenecks — deepened)
        # ═══════════════════════════════════════════════════════════════
        self.s3_down = CC(256, 512, 3, 2, 1)
        self.s3_down_bn = nn.BatchNorm2d(512)
        self.s3_down_act = nn.SiLU()
        self._build_bottleneck_group('s3', 512, 4)

        # ═══════════════════════════════════════════════════════════════
        # SPPF: 512×7² → 512×7²  (multi-scale pooling)
        # ═══════════════════════════════════════════════════════════════
        self.sp_cv1 = CC(512, 256, 1, 1)
        self.sp_bn1 = nn.BatchNorm2d(256)
        self.sp_act1 = nn.SiLU()
        self.sp_pool = nn.MaxPool2d(5, 1, 2)
        self.sp_cv2 = CC(256 * 4, 512, 1, 1)
        self.sp_bn2 = nn.BatchNorm2d(512)
        self.sp_act2 = nn.SiLU()

        # ═══════════════════════════════════════════════════════════════
        # Head: 512 → 512 → 256 → 128 → num_classes
        # ═══════════════════════════════════════════════════════════════
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.flat = nn.Flatten()

        self.fc1 = nn.Linear(512, 512)
        self.a_fc1 = nn.SiLU()
        self.dp1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(512, 256)
        self.a_fc2 = nn.SiLU()
        self.dp2 = nn.Dropout(0.5)

        self.fc3 = nn.Linear(256, 128)
        self.a_fc3 = nn.SiLU()
        self.dp3 = nn.Dropout(0.5)

        self.fc4 = nn.Linear(128, num_classes)

    def _build_bottleneck_group(self, prefix: str, channels: int, n: int):
        """Create n bottleneck blocks with name prefix."""
        for bi in range(n):
            setattr(self, f'{prefix}_b{bi}_cv1', ComplexConv(channels, channels, 3, 1, 1))
            setattr(self, f'{prefix}_b{bi}_bn1', nn.BatchNorm2d(channels))
            setattr(self, f'{prefix}_b{bi}_a1', nn.SiLU())
            setattr(self, f'{prefix}_b{bi}_cv2', ComplexConv(channels, channels, 3, 1, 1))
            setattr(self, f'{prefix}_b{bi}_bn2', nn.BatchNorm2d(channels))
            setattr(self, f'{prefix}_b{bi}_a2', nn.SiLU())

    def _bottleneck_forward(self, x, prefix: str, bi: int):
        """Run a single bottleneck block with residual connection."""
        cv1 = getattr(self, f'{prefix}_b{bi}_cv1')
        bn1 = getattr(self, f'{prefix}_b{bi}_bn1')
        a1 = getattr(self, f'{prefix}_b{bi}_a1')
        cv2 = getattr(self, f'{prefix}_b{bi}_cv2')
        bn2 = getattr(self, f'{prefix}_b{bi}_bn2')
        a2 = getattr(self, f'{prefix}_b{bi}_a2')
        return x + a2(bn2(cv2(a1(bn1(cv1(x))))))

    def _rgb2polar_imag(self, x: torch.Tensor) -> torch.Tensor:
        """RGB → HSV polar decomposition for the imaginary channel.

        Args:
            x: ImageNet-normalized RGB tensor (B, 3, H, W)

        Returns:
            imag tensor (B, 3, H, W): [S·cos(2πH), S·sin(2πH), S]
        """
        # De-normalize to [0, 1] for HSV computation
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        rgb = torch.clamp(x * std + mean, 0, 1)
        return rgb_to_hsv_polar_imag(rgb)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: RGB tensor (B, 3, H, W), ImageNet-normalized

        Returns:
            logits tensor (B, num_classes)
        """
        # ─── Split RGB into real + polar imaginary ───
        x_real = x
        x_imag = self._rgb2polar_imag(x)
        x = torch.complex(x_real, x_imag)

        # ─── Stem ───
        x = self.stem_act1(self.stem_bn1(self.stem_c1(x)))
        x = self.stem_act2(self.stem_bn2(self.stem_c2(x)))

        # ─── Stage 1 ───
        x = self.s1_down_act(self.s1_down_bn(self.s1_down(x)))
        x = self._bottleneck_forward(x, 's1', 0)
        x = self._bottleneck_forward(x, 's1', 1)

        # ─── Stage 2 ───
        x = self.s2_down_act(self.s2_down_bn(self.s2_down(x)))
        x = self._bottleneck_forward(x, 's2', 0)
        x = self._bottleneck_forward(x, 's2', 1)

        # ─── Stage 3 (deepened: 4 bottlenecks) ───
        x = self.s3_down_act(self.s3_down_bn(self.s3_down(x)))
        for bi in range(4):
            x = self._bottleneck_forward(x, 's3', bi)

        # ─── SPPF ───
        x = self.sp_act1(self.sp_bn1(self.sp_cv1(x)))
        p1 = self.sp_pool(x)
        p2 = self.sp_pool(p1)
        p3 = self.sp_pool(p2)
        x = self.sp_act2(self.sp_bn2(self.sp_cv2(torch.cat([x, p1, p2, p3], 1))))

        # ─── Head ───
        x = self.gap(x)
        x = self.flat(x)

        x = self.a_fc1(self.dp1(self.fc1(x)))
        x = self.a_fc2(self.dp2(self.fc2(x)))
        x = self.a_fc3(self.dp3(self.fc3(x)))
        x = self.fc4(x)

        return x
