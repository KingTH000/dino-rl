"""A bounded NumPy ring buffer with uniform sampling without replacement."""

from dataclasses import dataclass
from typing import Union

import numpy as np
import torch

Observation = Union[np.ndarray, dict[str, np.ndarray]]


@dataclass(frozen=True)
class ReplayBatch:
    states: Observation
    actions: np.ndarray
    rewards: np.ndarray
    next_states: Observation
    terminated: np.ndarray


class ReplayBuffer:
    """Store owned copies. `terminated` means crash, not a time-limit reset.

    Sampling returns copies too, so later insertions cannot mutate a sampled batch.
    The RNG is local and independent of exploration and environment randomness.
    """

    def __init__(self, capacity: int, state_size: int = 10, seed: int = 42):
        if (
            type(capacity) is not int
            or capacity <= 0
            or type(state_size) is not int
            or state_size <= 0
        ):
            raise ValueError("capacity and state_size must be positive integers")
        self.capacity = capacity
        self.rng = np.random.default_rng(seed)
        self.states = np.empty((capacity, state_size), dtype=np.float32)
        self.next_states = np.empty_like(self.states)
        self.actions = np.empty(capacity, dtype=np.int64)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.terminated = np.empty(capacity, dtype=np.bool_)
        self._size = self._position = 0

    def __len__(self) -> int:
        return self._size

    def state_dict(self) -> dict:
        """Serialize only occupied slots, preserving physical ring order and RNG."""
        return {
            "capacity": self.capacity,
            "position": self._position,
            "size": self._size,
            "rng": self.rng.bit_generator.state,
            "arrays": {
                name: torch.from_numpy(getattr(self, name)[: self._size].copy())
                for name in ("states", "next_states", "actions", "rewards", "terminated")
            },
        }

    def load_state_dict(self, data: dict) -> None:
        if data["capacity"] != self.capacity:
            raise ValueError("Checkpoint replay capacity does not match configuration")
        size, position = data["size"], data["position"]
        if not 0 <= size <= self.capacity or not 0 <= position < self.capacity:
            raise ValueError("Invalid replay size or position")
        for name, tensor in data["arrays"].items():
            destination = getattr(self, name)
            if tuple(tensor.shape) != (size, *destination.shape[1:]):
                raise ValueError(f"Invalid replay shape: {name}")
            destination[:size] = tensor.cpu().numpy()
        self._size, self._position = size, position
        self.rng.bit_generator.state = data["rng"]

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        terminated: bool,
    ) -> None:
        state = np.asarray(state, dtype=np.float32)
        next_state = np.asarray(next_state, dtype=np.float32)
        expected = self.states.shape[1:]
        if state.shape != expected or next_state.shape != expected:
            raise ValueError(f"Both states must have shape {expected}")
        if (
            not np.isfinite(state).all()
            or not np.isfinite(next_state).all()
            or not np.isfinite(reward)
        ):
            raise ValueError("Replay transitions must be finite")
        if (
            not isinstance(action, (int, np.integer))
            or isinstance(action, bool)
            or not 0 <= action < 3
        ):
            raise ValueError("action must be 0, 1, or 2")
        if not isinstance(terminated, (bool, np.bool_)):
            raise ValueError("terminated must be a boolean")
        index = self._position
        self.states[index], self.next_states[index] = state, next_state
        self.actions[index], self.rewards[index], self.terminated[index] = (
            action,
            reward,
            terminated,
        )
        self._position = (index + 1) % self.capacity
        self._size = min(self.capacity, self._size + 1)

    def sample(self, batch_size: int) -> ReplayBatch:
        if type(batch_size) is not int or not 0 < batch_size <= self._size:
            raise ValueError(
                "batch_size must be positive and no larger than the stored transitions"
            )
        indices = self.rng.choice(self._size, size=batch_size, replace=False)
        return ReplayBatch(
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.terminated[indices],
        )
