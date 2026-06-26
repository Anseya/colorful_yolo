#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Colorful-yolo Usage Example
============================
Minimal example: create model, run inference, export.

Requirements: torch, numpy, PIL
"""

import torch
from PIL import Image
from torchvision import transforms

from colorful_yolo import PolarComplexYOLO

# ─── 1. Create model ───
model = PolarComplexYOLO(num_classes=10)
model.eval()
print(f"PolarComplexYOLO parameters: {sum(p.numel() for p in model.parameters()):,}")

# ─── 2. Load and preprocess image ───
# Replace with your image path
img = Image.new('RGB', (224, 224), color=(128, 0, 0))  # dummy red image

tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

x = tf(img).unsqueeze(0)  # (1, 3, 224, 224)

# ─── 3. Run inference ───
with torch.no_grad():
    logits = model(x)
    probs = torch.softmax(logits, dim=1)
    pred = logits.argmax(dim=1)

print(f"Predicted class: {pred.item()}")
print(f"Probabilities:   {probs.squeeze().tolist()}")

# ─── 4. Training loop (minimal) ───
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
criterion = torch.nn.CrossEntropyLoss()

# Dummy batch
x_batch = torch.randn(4, 3, 224, 224)
y_batch = torch.randint(0, 10, (4,))

optimizer.zero_grad()
loss = criterion(model(x_batch), y_batch)
loss.backward()
optimizer.step()
print(f"Training loss: {loss.item():.4f}")
