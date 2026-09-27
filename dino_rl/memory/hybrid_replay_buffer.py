"""Compact visual replay with synchronized numerical features in the same slots."""

import numpy as np

from dino_rl.memory.image_replay_buffer import ImageReplayBuffer
from dino_rl.memory.replay_buffer import ReplayBatch


class HybridReplayBuffer(ImageReplayBuffer):
    """One ring and one sample index set keep the two modalities aligned."""

    kind = "hybrid"
    array_names = (*ImageReplayBuffer.array_names, "states", "next_states")
    additional_bytes_per_transition = 2 * 10 * np.dtype(np.float32).itemsize

    def __init__(
        self,
        capacity: int,
        shape: tuple[int, int, int] = (4, 84, 84),
        seed: int = 42,
        memory_limit_mb: int = 1024,
    ):
        super().__init__(capacity, shape, seed, memory_limit_mb)
        self.states = np.empty((capacity, 10), dtype=np.float32)
        self.next_states = np.empty_like(self.states)

    @staticmethod
    def _state(observation: dict[str, np.ndarray]) -> np.ndarray:
        if not isinstance(observation, dict) or set(observation) != {"state", "image"}:
            raise ValueError("Hybrid observations require exactly state and image keys")
        state = observation["state"]
        if (
            not isinstance(state, np.ndarray)
            or state.shape != (10,)
            or state.dtype != np.float32
            or not np.isfinite(state).all()
        ):
            raise ValueError("Hybrid state must be a finite float32 vector of shape (10,)")
        return state

    def add(self, state, action, reward, next_state, terminated) -> None:
        current, following = self._state(state), self._state(next_state)
        index = self._position
        # Base validation completes before either modality is committed.
        super().add(state["image"], action, reward, next_state["image"], terminated)
        self.states[index], self.next_states[index] = current, following

    def sample(self, batch_size: int) -> ReplayBatch:
        indices = self._sample_indices(batch_size)
        images = self._batch(indices)
        return ReplayBatch(
            {"image": images.states, "state": self.states[indices]},
            images.actions,
            images.rewards,
            {"image": images.next_states, "state": self.next_states[indices]},
            images.terminated,
        )
