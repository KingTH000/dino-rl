"""A visual observation wrapper that leaves the underlying game's dynamics intact."""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from dino_rl.config import ImageConfig
from dino_rl.envs.dino_env import DinoEnv
from dino_rl.preprocessing.frame_stack import FrameStack
from dino_rl.preprocessing.image_preprocessing import ImagePreprocessor


class ImageObservationWrapper(gym.Wrapper):
    """Expose stacked screenshots as observations; numerical info is debug-only.

    Supports headless rgb_array capture or a human viewer. Each call to step
    advances exactly one simulator tick and adds exactly one processed frame.
    Capturing an observation never adds an extra display refresh or sleep.
    """

    def __init__(self, env: DinoEnv, config: ImageConfig | None = None):
        if env.render_mode not in ("rgb_array", "human"):
            raise ValueError("Image observations require DinoEnv(render_mode='rgb_array')")
        super().__init__(env)
        self.image_config = config or ImageConfig()
        self.preprocessor = ImagePreprocessor(self.image_config)
        self.frame_stack = FrameStack(
            self.image_config.stack_size, (self.image_config.height, self.image_config.width)
        )
        self.observation_space = spaces.Box(
            0.0,
            1.0,
            shape=(self.image_config.stack_size, self.image_config.height, self.image_config.width),
            dtype=np.float32,
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        self.frame_stack.clear()
        _, info = self.env.reset(seed=seed, options=options)
        frame = self.preprocessor.process_uint8(self.env.capture_frame())
        return self.frame_stack.reset(frame), info

    def step(self, action: int):
        _, reward, terminated, truncated, info = self.env.step(action)
        frame = self.preprocessor.process_uint8(self.env.capture_frame())
        return self.frame_stack.append(frame), reward, terminated, truncated, info

    def close(self) -> None:
        self.frame_stack.clear()
        self.env.close()
