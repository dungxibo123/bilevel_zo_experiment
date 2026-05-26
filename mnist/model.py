"""Neural network models for the zeroth-order deep-learning experiments."""

from __future__ import annotations

import torch
import torch.nn as nn


class MLP(nn.Module):
    """Simple fully-connected classifier."""

    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SmallCNN(nn.Module):
    """Small CNN for MNIST/FashionMNIST/CIFAR10-scale experiments."""

    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def build_model(
    model_name: str,
    input_shape: tuple[int, int, int],
    num_classes: int,
    hidden_dim: int,
) -> nn.Module:
    """Factory for supported models."""
    c, h, w = input_shape
    if model_name == "mlp":
        return MLP(input_dim=c * h * w, num_classes=num_classes, hidden_dim=hidden_dim)
    if model_name == "cnn":
        return SmallCNN(in_channels=c, num_classes=num_classes)
    raise ValueError(f"Unknown model: {model_name}")
