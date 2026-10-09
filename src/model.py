"""Small CNN for Fashion-MNIST — 3 conv + 2 fc, ~68k parameters.

Deliberately small so the pre-trained .pt fits in a few hundred KB and
cold-start inference is sub-millisecond on CPU.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

CLASSES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


class FashionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(32 * 3 * 3, 64)
        self.fc2 = nn.Linear(64, 10)
        # Store last activations for Grad-CAM
        self._last_conv_activations: torch.Tensor | None = None

    def forward(self, x):
        x = F.relu(self.conv1(x)); x = self.pool(x)   # 14x14
        x = F.relu(self.conv2(x)); x = self.pool(x)   # 7x7
        x = F.relu(self.conv3(x))                     # 7x7
        self._last_conv_activations = x
        x = self.pool(x)                              # 3x3
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
