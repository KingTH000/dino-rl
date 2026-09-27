"""Fixed, physics-derived scaling without running statistics or data leakage."""

import numpy as np

from dino_rl.config import SimulatorConfig
from dino_rl.envs.simulator import DinoSimulator


class StateNormalizer:
    """Scale the ten raw features to roughly unit size, preserving signs.

    There is no clipping: negative obstacle distances and heights above the
    standing Dino remain distinguishable. The raw Gymnasium environment is unchanged.
    """

    def __init__(self, config: SimulatorConfig):
        self.scales = np.array(
            [
                config.max_speed,
                config.jump_velocity**2 / (2 * config.gravity),
                config.jump_velocity + config.gravity / config.fps,
                DinoSimulator.WIDTH,
                DinoSimulator.STAND_HEIGHT,
                DinoSimulator.STAND_HEIGHT,
                DinoSimulator.STAND_HEIGHT,
                2.0,
                1.0,
                1.0,
            ],
            dtype=np.float32,
        )

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        raw = np.asarray(observation, dtype=np.float32)
        if raw.shape != (10,) or not np.isfinite(raw).all():
            raise ValueError("Expected a finite numerical observation of shape (10,)")
        return raw / self.scales
