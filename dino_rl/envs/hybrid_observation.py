"""Synchronized pixels and normalized numerical features from one simulator tick."""

import numpy as np
from gymnasium import spaces

from dino_rl.config import ImageConfig
from dino_rl.envs.dino_env import DinoEnv
from dino_rl.envs.image_observation import ImageObservationWrapper
from dino_rl.preprocessing.state_normalization import StateNormalizer


class HybridObservationWrapper(ImageObservationWrapper):
    """Return a Dict observation; numerical features are normalized exactly once."""

    def __init__(self, env: DinoEnv, config: ImageConfig | None = None):
        super().__init__(env, config)
        self.normalizer = StateNormalizer(env.config)
        self.observation_space = spaces.Dict(
            image=self.observation_space,
            state=spaces.Box(-np.inf, np.inf, shape=(10,), dtype=np.float32),
        )

    def combine(self, image: np.ndarray) -> dict[str, np.ndarray]:
        """Also used when restoring a saved simulator and visual frame history."""
        return {"image": image, "state": self.normalizer(self.unwrapped.simulator.observation())}

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        image, info = super().reset(seed=seed, options=options)
        return self.combine(image), info

    def step(self, action: int):
        image, reward, terminated, truncated, info = super().step(action)
        return self.combine(image), reward, terminated, truncated, info
