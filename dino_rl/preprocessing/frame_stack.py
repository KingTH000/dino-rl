"""Episode-local frame history with owned storage and channels-first outputs."""

from collections import deque

import numpy as np


class FrameStack:
    """Keep uint8 frames internally; expose fresh normalized float32 stacks.

    Ordering is oldest to newest. Reset padding repeats the initial observation,
    so there is no artificial black frame or history from a previous episode.
    This class stacks observations only; it does not skip frames or repeat actions.
    """

    def __init__(self, stack_size: int = 4, frame_shape: tuple[int, int] = (84, 84)):
        if type(stack_size) is not int or stack_size <= 0:
            raise ValueError("stack_size must be a positive integer")
        if len(frame_shape) != 2 or any(type(v) is not int or v <= 0 for v in frame_shape):
            raise ValueError("frame_shape must contain positive integer height and width")
        self.stack_size = stack_size
        self.frame_shape = tuple(frame_shape)
        self._frames: deque[np.ndarray] = deque(maxlen=stack_size)

    def _validate(self, frame: np.ndarray) -> None:
        if (
            not isinstance(frame, np.ndarray)
            or frame.dtype != np.uint8
            or frame.shape != self.frame_shape
        ):
            raise ValueError(f"Expected a uint8 grayscale frame of shape {self.frame_shape}")

    def clear(self) -> None:
        self._frames.clear()

    def reset(self, frame: np.ndarray) -> np.ndarray:
        self._validate(frame)
        self.clear()
        self._frames.extend(frame.copy() for _ in range(self.stack_size))
        return self.observation()

    def append(self, frame: np.ndarray) -> np.ndarray:
        self._validate(frame)
        if len(self._frames) != self.stack_size:
            raise RuntimeError("Call reset(frame) before appending frames")
        self._frames.append(frame.copy())
        return self.observation()

    def as_uint8(self) -> np.ndarray:
        """Return a fresh compact stack, useful for later memory-efficient replay."""
        if len(self._frames) != self.stack_size:
            raise RuntimeError("Call reset(frame) before reading the stack")
        return np.stack(self._frames, axis=0)

    def observation(self) -> np.ndarray:
        return self.as_uint8().astype(np.float32) / np.float32(255.0)
