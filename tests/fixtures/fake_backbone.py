"""A small deterministic stand-in for a pretrained backbone. Tests never touch real weights."""

from __future__ import annotations

import torch
from torch import nn


class FakeBackbone(nn.Module):
    """Three strided conv stages named like a ResNet (layer1..layer3) plus an unused head."""

    def __init__(self, seed: int = 0, c1: int = 8, c2: int = 16, c3: int = 24) -> None:
        super().__init__()
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            self.layer1 = nn.Sequential(nn.Conv2d(3, c1, 3, stride=2, padding=1), nn.ReLU())
            self.layer2 = nn.Sequential(nn.Conv2d(c1, c2, 3, stride=2, padding=1), nn.ReLU())
            self.layer3 = nn.Sequential(nn.Conv2d(c2, c3, 3, stride=2, padding=1), nn.ReLU())
            self.head = nn.Linear(c3, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer3(self.layer2(self.layer1(x)))
        return self.head(x.mean(dim=(2, 3)))
