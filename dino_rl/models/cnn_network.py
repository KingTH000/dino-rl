"""Atari-style CNN mapping normalized stacked frames to three action values."""

import torch
from torch import nn


def image_features(input_shape: tuple[int, int, int]) -> tuple[nn.Sequential, int]:
    """Build the shared convolutional trunk and calculate its flattened size."""
    channels, height, width = input_shape
    for kernel, stride in ((8, 4), (4, 2), (3, 1)):
        height, width = (height - kernel) // stride + 1, (width - kernel) // stride + 1
        if min(height, width) <= 0:
            raise ValueError("Image dimensions are too small for the three CNN convolutions")
    return nn.Sequential(
        nn.Conv2d(channels, 32, 8, stride=4),
        nn.ReLU(),
        nn.Conv2d(32, 64, 4, stride=2),
        nn.ReLU(),
        nn.Conv2d(64, 64, 3, stride=1),
        nn.ReLU(),
        nn.Flatten(),
    ), 64 * height * width


class CNNQNetwork(nn.Module):
    def __init__(
        self,
        input_shape: tuple[int, int, int] = (4, 84, 84),
        hidden_size: int = 256,
        action_count: int = 3,
    ):
        super().__init__()
        self.features, feature_size = image_features(input_shape)
        self.head = nn.Sequential(
            nn.Linear(feature_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_count),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(images))
