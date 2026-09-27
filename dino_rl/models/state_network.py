"""MLP mapping normalized numerical features to one Q-value per action."""

import torch
from torch import nn


class StateQNetwork(nn.Module):
    def __init__(self, state_size: int = 10, hidden_size: int = 128, action_count: int = 3):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_count),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.layers(states)
