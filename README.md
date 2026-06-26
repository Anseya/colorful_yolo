# Colorful-yolo &middot; 彩色YOLO

> Complex-valued YOLO26 with HSV Polar Decomposition &middot; 基于HSV极坐标分解的复数卷积YOLO26

[English](#english) | [中文](#chinese)

---

<a id="english"></a>
## English

### 1. Overview

**Colorful-yolo** extends the YOLO26 backbone into the **complex number domain**. Instead of discarding color information during RGB-to-grayscale conversion (as most vision models implicitly do via channel mixing), Colorful-yolo explicitly splits each input image into two orthogonal representations:

| Component | Content | Role |
|-----------|---------|------|
| **Real part** (`Xr`) | Raw RGB values | Spatial structure, texture, edges |
| **Imag part** (`Xi`) | HSV in polar coordinates | Color phase, hue patterns, saturation |

By feeding these as `torch.complex(Xr, Xi)` into ComplexConv, the network learns **cross-interactions** between spatial and color-phase features — something that real-valued convolutions cannot do because they treat all input channels as interchangeable scalars.

---

### 2. Design Philosophy

#### 2.1 Why Complex Numbers?

Standard real-valued convolution:
```
Y = Σ W_c * X_c          (c = 1...C_in)
```
Each input channel `X_c` is multiplied by a scalar weight `W_c`. All channels are treated equally — there is no concept of "phase" or "orthogonal channel pairs."

Complex-valued convolution:
```
Y = (Wr + j·Wi) * (Xr + j·Xi)
  = (Wr·Xr - Wi·Xi) + j(Wr·Xi + Wi·Xr)
```
The real and imaginary parts of the weight **interact** with both the real and imaginary parts of the input. The cross-term `Wi·Xi` acting on the real output means that the imaginary input can *modulate* the real feature maps — and vice versa.

This gives the network an extra degree of freedom: it can learn to **amplify or suppress** spatial features based on color-phase information.

#### 2.2 Why HSV Polar Decomposition?

The choice of imaginary component is critical. We decompose RGB into HSV and embed hue+saturation into polar coordinates:

```
H (hue)     →  θ = H · 2π          (angle in [0, 2π))
S (sat.)    →  r = S               (radius in [0, 1])
V (value)   →  stays in Real[RGB]  (not duplicated)
```

The imaginary input is:
```
Xi = [S·cos(θ),  S·sin(θ),  S]
```

This design has three mathematical advantages:

1. **Periodicity of hue is naturally handled**: The hue circle (0° = 360°) has no artificial boundary when mapped to cos/sin. A pixel with H=0.99 and H=0.01 maps to nearby points on the unit circle.

2. **Saturation as confidence**: Gray pixels (S≈0) map to Xi≈0 — they contribute nothing to the imaginary channel. The model automatically learns to ignore color information from achromatic regions.

3. **Orthogonality to RGB**: V (value/brightness) is already present in Real[RGB], so it is excluded from Xi to avoid information redundancy. Xi carries only the *pure chromatic* information.

#### 2.3 Channel Alignment

| Real channel | Imag channel | Pairing rationale |
|-------------|-------------|-------------------|
| R | S·cos(2πH) | Red-dominant hues get high cos |
| G | S·sin(2πH) | Green-dominant hues get high sin |
| B | S | Blue-dominant hues: S correlates with V-delta |

---

### 3. Network Architecture

```
Input (B,3,224,224) RGB
    │
    ├─ Real: [R, G, B]                    ─┐
    ├─ Imag: [S·cosθ, S·sinθ, S]          ─┤ → torch.complex(real, imag)
    │                                       │
    ▼                                       │
┌───────────────────────────────────────────┘
│  Stem
│   ComplexConv(3→32, k3s2)  + BN + SiLU    →  112×112, 32ch
│   ComplexConv(32→64, k3s2) + BN + SiLU    →  56×56,  64ch
│
│  Stage 1  (56×56 → 28×28)
│   ComplexConv(64→128, k3s2) + BN + SiLU
│   Bottleneck ×2:  [ComplexConv+BN+SiLU]×2 + residual
│
│  Stage 2  (28×28 → 14×14)
│   ComplexConv(128→256, k3s2) + BN + SiLU
│   Bottleneck ×2
│
│  Stage 3  (14×14 → 7×7)    ← deepened
│   ComplexConv(256→512, k3s2) + BN + SiLU
│   Bottleneck ×4
│
│  SPPF  (7×7, multi-scale)
│   ComplexConv(512→256, k1s1) + BN + SiLU
│   MaxPool2d(5×5) ×3  →  concat  → 256×4
│   ComplexConv(1024→512, k1s1) + BN + SiLU
│
│  Classification Head
│   GlobalAvgPool → Flatten
│   FC(512→512) + SiLU + Dropout(0.5)
│   FC(512→256) + SiLU + Dropout(0.5)
│   FC(256→128) + SiLU + Dropout(0.5)
│   FC(128→num_classes)
│
▼
Logits (B, num_classes)
```

**Layer count**: 13 complex convolution layers in backbone (each with dual real+imag weight sets).

---

### 4. ComplexConv Design Details

#### 4.1 Weight Initialization

The imaginary weights `Wi` are initialized via **per-channel random rotation**:

```python
for each output channel i:
    rotation = randint(0, 3) * 90°      # {0°, 90°, 180°, 270°}
    scale    = uniform(0.15, 0.5)        # random modulation
    sign     = ±1 with 50% probability   # random polarity
    Wi[i]    = sign * scale * rot90(Wr[i], rotation)
```

This avoids the **channel homogenization** problem where all channels share the same `Wi = rot90(Wr)*0.3` pattern. Per-channel diversity ensures different channels learn different phase relationships.

#### 4.2 Activation Strategy

Activation (SiLU) is applied **separately** to real and imaginary parts:
```
real' = SiLU(bn_real(conv_real(Xr) - conv_imag(Xi)))
imag' = SiLU(bn_imag(conv_real(Xi) + conv_imag(Xr)))
```

This **preserves phase information** `atan2(imag', real')` through the network. If we instead applied activation to the magnitude `|z|`, all phase information would be truncated at every layer.

#### 4.3 Output Mode

Default `output_real=True`: returns Euclidean magnitude `√(real² + imag²)` with safe clamp at `1e-12`. This produces a real-valued tensor compatible with downstream BatchNorm and MaxPool layers.

---

### 5. Available Complex Convolution Variants

| Class | Description | Use Case |
|-------|-------------|----------|
| `ComplexConv` | Standard complex conv, dual real/imag weight pairs | General-purpose |
| `ComplexConv2` | ComplexConv + parallel 1×1 branch | Multi-scale feature fusion |
| `ComplexDWConv` | Depth-wise complex conv (groups=gcd(c1,c2)) | Lightweight backbones |
| `ComplexLightConv` | 1×1 ComplexConv + ComplexDWConv | Efficient channel adjustment |
| `ComplexGhostConv` | Primary + cheap depth-wise complex convs | Parameter-efficient |
| `ComplexBottleneck` | Residual bottleneck (all ComplexConv internally) | Backbone building block |
| `ComplexSPPF` | Spatial Pyramid Pooling (magnitude-aware pooling) | Multi-scale fusion |
| `ComplexC3k2` | CSP Bottleneck with 2 convolutions | Higher-level block |
| `ComplexDFL` | Distribution Focal Loss integral (magnitude→DFL) | Detection head |

---

### 6. Training Principles

#### 6.1 Input Preprocessing
- Input images must be **3-channel RGB** (grayscale images lack hue information)
- ImageNet normalization: `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`
- HSV polar decomposition happens inside `forward()` — no external preprocessing needed
- Images are de-normalized to [0,1] before HSV computation, then re-normalized in the complex domain

#### 6.2 Loss Function
Standard `CrossEntropyLoss` is recommended. For imbalanced datasets, `FocalLoss(gamma=2.0)` with per-class weights computed from training set statistics can help the model focus on minority classes.

#### 6.3 Optimizer
`AdamW` with `weight_decay=1e-4` works well. Initial learning rates:
- `1e-4` for full model training
- `5e-4` to `1e-3` for fine-tuning the classification head only

#### 6.4 Training Considerations
- ComplexConv has approximately **2× the parameters** of standard Conv2d (dual weight sets)
- Forward pass is ~3-4× slower than real-valued convolution on CPU due to complex arithmetic
- BatchNorm requires ~2× memory for real+imag statistics
- Warm-up learning rate for 2-3 epochs helps stabilize early complex-valued training
- The imag-weight initialization should NOT be re-randomized between runs (no manual seed changes)

---

### 7. Current Limitations

#### 7.1 Grayscale Incompatibility
Colorful-yolo fundamentally **requires color information**. On grayscale datasets (e.g., MNIST), the HSV polar decomposition produces `S≈0` everywhere, making the imaginary channel degenerate. The model effectively falls back to real-valued convolution with extra overhead.

**Mitigation**: Use Colorful-yolo only on RGB datasets. For grayscale data, the standard real-valued YOLO26 is the correct choice.

#### 7.2 Dataset Size Sensitivity
Complex convolutions double the parameter count (~48.6M for the default backbone), making the model more prone to overfitting on small datasets (<1,000 images). Regularization (Dropout 0.5, weight decay, data augmentation) helps but cannot fully compensate for insufficient training data.

#### 7.3 Computational Cost
- **CPU training**: ~3-4× slower than real-valued YOLO26 per epoch
- **GPU memory**: ~2× for weights + ~2× for intermediate complex tensors
- **Inference latency**: Higher than real-valued counterparts; not suitable for real-time applications without optimization

#### 7.4 HSV Assumption
The model assumes HSV polar decomposition is the optimal imaginary representation for all color-sensitive tasks. This may not hold for:
- Medical imaging (where false-color mappings have arbitrary semantics)
- Thermal/infrared images (single-channel with color-mapped palettes)
- Synthetic data with non-photorealistic color distributions

#### 7.5 Training Stability
Complex-valued training can exhibit higher variance across runs due to the dual weight initialization and cross-interaction terms. Multiple random seeds are recommended for robust evaluation.

#### 7.6 Theoretical Gaps
- The optimal scale ratio between real and imaginary magnitudes is currently set by heuristics (`scale_imag=0.6`) rather than learned
- The interaction between SiLU activation and complex phase propagation is not fully characterized analytically
- Convergence guarantees for complex-valued CNNs under standard optimizers (AdamW, SGD) are less established than for real-valued networks

---

<a id="chinese"></a>
## 中文

### 1. 概述

**Colorful-yolo（彩色YOLO）** 将 YOLO26 骨干网络扩展到 **复数域**。大多数视觉模型通过通道混合隐式丢失颜色信息，而 Colorful-yolo 将每张输入图像显式地分解为两个正交表示：

| 组件 | 内容 | 作用 |
|------|------|------|
| **实部** (`Xr`) | 原始 RGB 值 | 空间结构、纹理、边缘 |
| **虚部** (`Xi`) | HSV 的极坐标表示 | 颜色相位、色相模式、饱和度 |

将两者组合为 `torch.complex(Xr, Xi)` 输入到 ComplexConv 中，使网络能够学习空间特征与颜色相位特征之间的 **交叉交互**——这是实数卷积无法做到的，因为实数卷积将所有输入通道视为可互换的标量。

---

### 2. 设计理念与思想原理

#### 2.1 为什么用复数？

标准实数卷积：
```
Y = Σ W_c * X_c          (c = 1...C_in)
```
每个输入通道 `X_c` 乘以一个标量权重 `W_c`。所有通道被平等对待——没有"相位"或"正交通道对"的概念。

复数卷积：
```
Y = (Wr + j·Wi) * (Xr + j·Xi)
  = (Wr·Xr - Wi·Xi) + j(Wr·Xi + Wi·Xr)
```
权重的实部与虚部分别与输入的实部和虚部 **交叉交互**。作用于实部输出的交叉项 `Wi·Xi` 意味着虚部输入可以 **调制** 实部特征图——反之亦然。

这赋予网络一个额外的自由度：它可以学习根据颜色相位信息来 **增强或抑制** 空间特征。

#### 2.2 为什么选 HSV 极坐标分解？

虚部成分的选择至关重要。我们将 RGB 分解为 HSV，并将色相和饱和度嵌入极坐标：

```
H（色相）  →  θ = H · 2π           (角度 ∈ [0, 2π))
S（饱和度）→  r = S                 (半径 ∈ [0, 1])
V（明度）  →  保留在实部 Real[RGB]  (不重复编码)
```

虚部输入为：
```
Xi = [S·cos(θ),  S·sin(θ),  S]
```

这个设计有三个数学优势：

1. **色相的周期性被自然处理**：色相环（0° = 360°）通过 cos/sin 映射没有人工边界。H=0.99 和 H=0.01 的像素映射到单位圆上的邻近点，消除了色相值的"跳变"问题。

2. **饱和度作为置信度**：灰色像素（S≈0）映射到 Xi≈0——它们对虚部通道没有任何贡献。模型自动学会忽略无色区域的"颜色信息"，避免噪声注入。

3. **与 RGB 正交**：V（明度/亮度）已经在实部 RGB 中编码，因此被排除在虚部之外以避免信息冗余。Xi 仅携带 **纯粹的色度信息**，实现了信息维度的真正解耦。

#### 2.3 通道配对原理

| 实部通道 | 虚部通道 | 配对逻辑 |
|---------|---------|---------|
| R | S·cos(2πH) | 红色主导的色相获得高 cos 值 |
| G | S·sin(2πH) | 绿色主导的色相获得高 sin 值 |
| B | S | 蓝色主导的色相与 V-delta 呈高饱和度关联 |

---

### 3. 网络架构

```
输入 (B,3,224,224) RGB
    │
    ├─ 实部: [R, G, B]                    ─┐
    ├─ 虚部: [S·cosθ, S·sinθ, S]          ─┤ → torch.complex(实部, 虚部)
    │                                       │
    ▼                                       │
┌───────────────────────────────────────────┘
│  Stem（茎干层）
│   ComplexConv(3→32, k3s2)  + BN + SiLU    →  112×112, 32ch
│   ComplexConv(32→64, k3s2) + BN + SiLU    →  56×56,  64ch
│
│  Stage 1（阶段一：56×56 → 28×28）
│   ComplexConv(64→128, k3s2) + BN + SiLU   ← 下采样卷积层
│   Bottleneck ×2:  [ComplexConv+BN+SiLU]×2 + 残差连接
│
│  Stage 2（阶段二：28×28 → 14×14）
│   ComplexConv(128→256, k3s2) + BN + SiLU
│   Bottleneck ×2
│
│  Stage 3（阶段三：14×14 → 7×7） ← 加深设计
│   ComplexConv(256→512, k3s2) + BN + SiLU
│   Bottleneck ×4                   ← 4个瓶颈块，比前两个阶段多一倍
│
│  SPPF（空间金字塔池化-快速版，7×7）
│   ComplexConv(512→256, k1s1) + BN + SiLU
│   MaxPool2d(5×5) ×3  →  拼接  → 256×4=1024ch
│   ComplexConv(1024→512, k1s1) + BN + SiLU
│
│  分类头（4层全连接）
│   GlobalAvgPool → Flatten
│   FC(512→512) + SiLU + Dropout(0.5)
│   FC(512→256) + SiLU + Dropout(0.5)
│   FC(256→128) + SiLU + Dropout(0.5)
│   FC(128→类别数)
│
▼
Logits (B, num_classes)
```

**层数统计**：骨干网络共 13 层复数卷积（每层含实部+虚部双权重组），分类头 4 层全连接。

---

### 4. ComplexConv 核心设计细节

#### 4.1 权重初始化策略

虚部权重 `Wi` 采用 **逐通道随机旋转** 初始化：

```python
对每个输出通道 i：
    旋转角度 = 随机选取 {0°, 90°, 180°, 270°}
    缩放因子 = 均匀分布 [0.15, 0.5]
    正负号   = ±1（各50%概率）
    Wi[i]    = sign * scale * rot90(Wr[i], 旋转角度)
```

这避免了早期版本中发现的 **通道同质化** 问题——所有通道共享同一个 `Wi = rot90(Wr)*0.3` 模式。逐通道的多样性确保不同通道学习不同的相位关系，提升特征表达的丰富度。

#### 4.2 激活策略

激活函数（SiLU）**分别** 作用于实部和虚部：
```
实部' = SiLU(BN_real(conv_real(Xr) - conv_imag(Xi)))
虚部' = SiLU(BN_imag(conv_real(Xi) + conv_imag(Xr)))
```

这 **保留了相位信息** `atan2(imag', real')` 在网络中逐层传播。如果改为对模长 `|z|` 施加激活，每层的相位信息都会被截断，退化为半实数网络。

#### 4.3 输出模式

默认 `output_real=True`：返回欧几里得模长 `√(real² + imag²)`，使用安全截断 `clamp(min=1e-12)` 避免除零。产生实数张量，兼容下游的 BatchNorm 和 MaxPool 层。

---

### 5. 可用的复数卷积变体

| 类名 | 说明 | 适用场景 |
|------|------|---------|
| `ComplexConv` | 标准复数卷积，双权重组 | 通用 |
| `ComplexConv2` | 复数Conv + 并行1×1分支 | 多尺度特征融合 |
| `ComplexDWConv` | 深度可分离复数卷积 | 轻量级骨干网络 |
| `ComplexLightConv` | 1×1 ComplexConv + DWConv | 高效通道调整 |
| `ComplexGhostConv` | 主卷积 + 廉价深度卷积 | 参数高效 |
| `ComplexBottleneck` | 残差瓶颈块（内部全部ComplexConv） | 骨干网络构建块 |
| `ComplexSPPF` | 空间金字塔池化（模长感知池化） | 多尺度特征融合 |
| `ComplexC3k2` | CSP瓶颈 + 双卷积 | 高层构建块 |
| `ComplexDFL` | 分布焦点损失积分模块 | 检测头 |

---

### 6. 训练原理

#### 6.1 输入预处理
- 输入必须是 **3 通道 RGB 图像**（灰度图缺乏色相信息，不适用）
- ImageNet 标准化：`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`
- HSV 极坐标分解在 `forward()` 内部自动完成——无需外部预处理
- 图像在 HSV 计算前反标准化到 [0,1]，然后在复数域重新归一化

#### 6.2 损失函数
推荐标准 `CrossEntropyLoss`。对于类别不平衡的数据集，可使用 `FocalLoss(gamma=2.0)` 并结合训练集统计量动态计算类别权重，帮助模型更关注少数类。

#### 6.3 优化器
`AdamW` + `weight_decay=1e-4` 效果良好。初始学习率：
- 完整训练：`1e-4`
- 仅微调分类头：`5e-4` 到 `1e-3`

#### 6.4 训练注意事项
- ComplexConv 的参数量约为标准 Conv2d 的 **2 倍**（双权重组）
- 前向传播在 CPU 上比实数卷积慢约 **3-4 倍**（复数运算开销）
- BatchNorm 需要约 2 倍显存（实部+虚部分别统计）
- 前 2-3 个 epoch 的学习率预热有助于稳定复数训练
- 虚部权重初始化不应在不同运行间重新随机化

---

### 7. 当前局限

#### 7.1 灰度图不兼容
Colorful-yolo **依赖颜色信息**。在灰度数据集（如 MNIST）上，HSV 极坐标分解产生 S≈0，虚部通道退化，模型实质上回退到带额外开销的实数卷积。

**缓解方案**：仅在 RGB 数据集上使用 Colorful-yolo。灰度数据应使用标准实数 YOLO26。

#### 7.2 小数据集敏感性
复数卷积使参数量翻倍（默认骨干约 48.6M），使模型在小数据集（<1,000 张图）上更易过拟合。Dropout 0.5、权重衰减和数据增强可以缓解，但不能完全补偿训练数据不足。

#### 7.3 计算成本
- **CPU 训练**：每 epoch 比实数 YOLO26 慢约 3-4 倍
- **GPU 显存**：权重约 2× + 中间复数张量约 2×
- **推理延迟**：高于实值对应模型；未经优化不适用于实时场景

#### 7.4 HSV 假设
模型假设 HSV 极坐标分解是所有颜色敏感任务的最优虚部表示。这在以下场景可能不成立：
- 医学影像（伪彩色映射的语义任意性）
- 热成像/红外图像（单通道 + 调色板映射）
- 非真实感色彩分布的合成数据

#### 7.5 训练稳定性
复数训练因双权重初始化和交叉交互项，在不同随机种子下可能表现出更高方差。建议使用多个随机种子进行鲁棒评估。

#### 7.6 理论待完善
- 实部与虚部的最佳幅值比目前基于启发式设定（`scale_imag=0.6`），而非学习得到
- SiLU 激活与复数相位传播之间的交互尚未完整的解析表征
- 复数 CNN 在标准优化器（AdamW、SGD）下的收敛性保证不如实数网络成熟

---

## Project Structure / 项目结构

```
Colorful-yolo/
├── colorful_yolo/
│   ├── __init__.py          # Package exports
│   ├── complex_conv.py      # ComplexConv and 4 variants
│   ├── complex_blocks.py    # ComplexBottleneck, ComplexSPPF, ComplexC3k2, ComplexDFL
│   ├── polar_yolo.py        # PolarComplexYOLO model
│   └── image_preprocess.py  # RGB→HSV polar decomposition
├── example.py               # Minimal usage example
└── README.md                # This file
```

## Quick Start / 快速开始

```python
from colorful_yolo import PolarComplexYOLO

model = PolarComplexYOLO(num_classes=10)
# x: (B, 3, 224, 224) ImageNet-normalized RGB tensor
logits = model(x)
```

See [example.py](example.py) for a complete training loop.

## References / 参考文献

- Trabelsi et al., "Deep Complex Networks" (ICLR 2018)
- Smith, "Color Gamut Transform Pairs" (SIGGRAPH 1978) — HSV color space
- Ultralytics YOLO (AGPL-3.0) — backbone architecture inspiration
- Li et al., "Generalized Focal Loss" (NeurIPS 2020) — DFL module design
