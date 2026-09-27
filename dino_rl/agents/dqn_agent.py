"""Vanilla DQN: replay batches, frozen target estimates, Huber loss and Adam."""

from copy import deepcopy

import numpy as np
import torch
from torch import nn

from dino_rl.config import DQNConfig
from dino_rl.memory.replay_buffer import Observation, ReplayBatch
from dino_rl.models.state_network import StateQNetwork


def epsilon_at_step(step: int, config: DQNConfig) -> float:
    """Linear decay on environment steps, clamped at both ends."""
    fraction = min(1.0, max(0.0, step / config.epsilon_decay_steps))
    return config.epsilon_start + fraction * (config.epsilon_end - config.epsilon_start)


class DQNAgent:
    def __init__(
        self,
        config: DQNConfig,
        device: torch.device,
        seed: int = 42,
        network: nn.Module | None = None,
    ):
        self.config, self.device = config, device
        self.rng = np.random.default_rng(seed)
        self.online = (
            network if network is not None else StateQNetwork(hidden_size=config.hidden_size)
        ).to(device)
        self.target = deepcopy(self.online).to(device)
        self.target.requires_grad_(False)
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=config.learning_rate)
        self.loss_function = nn.SmoothL1Loss()
        self.updates = 0

    def _input_tensor(self, values: Observation) -> torch.Tensor | dict[str, torch.Tensor]:
        if isinstance(values, dict):
            return {name: self._input_tensor(value) for name, value in values.items()}
        tensor = torch.as_tensor(values, device=self.device).to(torch.float32)
        return tensor / 255.0 if values.dtype == np.uint8 else tensor

    @torch.no_grad()
    def act(self, state: Observation, epsilon: float = 0.0) -> int:
        """epsilon=0 selects greedily, without learning or consuming exploration RNG."""
        if not 0 <= epsilon <= 1:
            raise ValueError("epsilon must be in [0, 1]")
        if epsilon > 0 and self.rng.random() < epsilon:
            return int(self.rng.integers(3))
        self.online.eval()
        tensor = self._input_tensor(state)
        tensor = (
            {name: value.unsqueeze(0) for name, value in tensor.items()}
            if isinstance(tensor, dict)
            else tensor.unsqueeze(0)
        )
        return int(self.online(tensor).argmax(dim=1).item())

    @torch.no_grad()
    def bellman_targets(
        self,
        rewards: torch.Tensor,
        next_states: torch.Tensor | dict[str, torch.Tensor],
        terminated: torch.Tensor,
    ) -> torch.Tensor:
        """Only real terminations remove the discounted future value."""
        next_values = self.target(next_states).max(dim=1).values
        return rewards + self.config.gamma * torch.where(terminated, 0.0, next_values)

    def update(self, batch: ReplayBatch) -> dict[str, float]:
        self.online.train()
        states = self._input_tensor(batch.states)
        actions = torch.as_tensor(batch.actions, device=self.device)
        rewards = torch.as_tensor(batch.rewards, device=self.device)
        next_states = self._input_tensor(batch.next_states)
        terminated = torch.as_tensor(batch.terminated, device=self.device)
        predicted = self.online(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        targets = self.bellman_targets(rewards, next_states, terminated)
        loss = self.loss_function(predicted, targets)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite DQN loss; inspect observations and configuration")
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        self.updates += 1
        return {
            "loss": loss.item(),
            "mean_q": predicted.detach().mean().item(),
            "mean_target": targets.mean().item(),
        }

    def sync_target(self) -> None:
        """Hard synchronization; the trainer schedules this by environment steps."""
        self.target.load_state_dict(self.online.state_dict())
