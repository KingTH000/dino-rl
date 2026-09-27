"""Combine visual features with numerical state before predicting action values."""

import torch
from torch import nn

from dino_rl.models.cnn_network import image_features


class HybridQNetwork(nn.Module):
    """Two trainable branches, concatenation, then a shared action-value head."""

    def __init__(
        self,
        input_shape: tuple[int, int, int] = (4, 84, 84),
        hidden_size: int = 256,
        state_hidden_size: int = 64,
        state_size: int = 10,
        action_count: int = 3,
    ):
        super().__init__()
        features, feature_size = image_features(input_shape)
        self.image_branch = nn.Sequential(features, nn.Linear(feature_size, hidden_size), nn.ReLU())
        self.state_branch = nn.Sequential(nn.Linear(state_size, state_hidden_size), nn.ReLU())
        self.head = nn.Sequential(
            nn.Linear(hidden_size + state_hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_count),
        )

    def forward(self, observation: dict[str, torch.Tensor]) -> torch.Tensor:
        image = self.image_branch(observation["image"])
        state = self.state_branch(observation["state"])
        return self.head(torch.cat((image, state), dim=1))
