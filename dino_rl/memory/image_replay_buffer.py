"""Bounded visual replay storing each state stack plus only its new next frame."""

import numpy as np
import torch

from dino_rl.memory.replay_buffer import ReplayBatch


class ImageReplayBuffer:
    """Five uint8 frames per four-frame transition, rather than eight float frames.

    Each transition owns its frames, so sampling cannot cross episode boundaries
    or refer to overwritten history. Next-state stacks are reconstructed by shifting.
    Truncated transitions retain their true final frame and continue to bootstrap.
    """

    kind = "image"
    array_names = ("frames", "actions", "rewards", "terminated")
    additional_bytes_per_transition = 0

    def __init__(
        self,
        capacity: int,
        shape: tuple[int, int, int] = (4, 84, 84),
        seed: int = 42,
        memory_limit_mb: int = 1024,
    ):
        if type(capacity) is not int or capacity <= 0:
            raise ValueError("capacity must be a positive integer")
        if len(shape) != 3 or any(type(v) is not int or v <= 0 for v in shape):
            raise ValueError("shape must be positive channels, height, width")
        self.capacity, self.shape = capacity, tuple(shape)
        required = capacity * (
            (shape[0] + 1) * shape[1] * shape[2] + 8 + 4 + 1 + self.additional_bytes_per_transition
        )
        if required > memory_limit_mb * 1024**2:
            raise ValueError(
                f"Image replay needs {required / 1024**2:.1f} MiB, exceeding "
                f"{memory_limit_mb} MiB limit; reduce replay_capacity (see image_dqn.yaml)"
            )
        self.frames = np.empty((capacity, shape[0] + 1, shape[1], shape[2]), dtype=np.uint8)
        self.actions = np.empty(capacity, dtype=np.int64)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.terminated = np.empty(capacity, dtype=np.bool_)
        self.rng = np.random.default_rng(seed)
        self._size = self._position = 0

    @property
    def storage_bytes(self) -> int:
        return sum(getattr(self, name).nbytes for name in self.array_names)

    def __len__(self) -> int:
        return self._size

    def _pixels(self, value: np.ndarray) -> np.ndarray:
        if not isinstance(value, np.ndarray) or value.shape != self.shape:
            raise ValueError(f"Expected image stack with shape {self.shape}")
        if value.dtype == np.uint8:
            return value
        if (
            value.dtype != np.float32
            or not np.isfinite(value).all()
            or value.min() < 0
            or value.max() > 1
        ):
            raise ValueError("Image replay expects uint8 pixels or normalized float32 pixels")
        return np.rint(value * 255).astype(np.uint8)

    def add(self, state, action, reward, next_state, terminated) -> None:
        state, next_state = self._pixels(state), self._pixels(next_state)
        if not np.array_equal(state[1:], next_state[:-1]):
            raise ValueError(
                "Next stack must shift the current stack; store final frame BEFORE reset"
            )
        if (
            not isinstance(action, (int, np.integer))
            or isinstance(action, bool)
            or not 0 <= action < 3
        ):
            raise ValueError("action must be 0, 1, or 2")
        if not np.isfinite(reward) or not isinstance(terminated, (bool, np.bool_)):
            raise ValueError("Reward must be finite and terminated must be boolean")
        index = self._position
        self.frames[index, :-1] = state
        self.frames[index, -1] = next_state[-1]
        self.actions[index], self.rewards[index], self.terminated[index] = (
            action,
            reward,
            terminated,
        )
        self._position = (index + 1) % self.capacity
        self._size = min(self.capacity, self._size + 1)

    def sample(self, batch_size: int) -> ReplayBatch:
        return self._batch(self._sample_indices(batch_size))

    def _sample_indices(self, batch_size: int) -> np.ndarray:
        if type(batch_size) is not int or not 0 < batch_size <= self._size:
            raise ValueError("batch_size must be positive and no larger than stored transitions")
        return self.rng.choice(self._size, batch_size, replace=False)

    def _batch(self, indices: np.ndarray) -> ReplayBatch:
        frames = self.frames[indices]
        return ReplayBatch(
            frames[:, :-1].copy(),
            self.actions[indices],
            self.rewards[indices],
            frames[:, 1:].copy(),
            self.terminated[indices],
        )

    def state_dict(self) -> dict:
        return {
            "kind": self.kind,
            "capacity": self.capacity,
            "shape": list(self.shape),
            "size": self._size,
            "position": self._position,
            "rng": self.rng.bit_generator.state,
            "arrays": {
                name: torch.from_numpy(getattr(self, name)[: self._size].copy())
                for name in self.array_names
            },
        }

    def load_state_dict(self, data: dict) -> None:
        if (
            data.get("kind") != self.kind
            or data["capacity"] != self.capacity
            or tuple(data["shape"]) != self.shape
        ):
            raise ValueError("Image replay checkpoint does not match configuration")
        size, position = data["size"], data["position"]
        if not 0 <= size <= self.capacity or not 0 <= position < self.capacity:
            raise ValueError("Invalid replay size/position")
        for name in self.array_names:
            destination, source = getattr(self, name), data["arrays"][name]
            if tuple(source.shape) != (size, *destination.shape[1:]):
                raise ValueError("Invalid image replay shape")
            destination[:size] = source.cpu().numpy()
        self._size, self._position = size, position
        self.rng.bit_generator.state = data["rng"]
