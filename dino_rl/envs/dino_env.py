"""Gymnasium adapter: episode lifecycle, rewards, observations and optional display."""

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from dino_rl.config import RewardConfig, SimulatorConfig

from .simulator import Action, DinoSimulator


class DinoEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        config: SimulatorConfig | None = None,
        rewards: RewardConfig | None = None,
        render_mode: str | None = None,
    ):
        if render_mode not in (None, "human", "rgb_array"):
            raise ValueError("render_mode must be None, human, or rgb_array")
        self.config = config or SimulatorConfig()
        self.rewards = rewards or RewardConfig()
        self.render_mode = render_mode
        self.metadata = {**self.metadata, "render_fps": self.config.fps}
        self.simulator = DinoSimulator(self.config)
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(10,), dtype=np.float32)
        self._needs_reset = True
        self._screen = self._clock = None

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if options:
            raise ValueError("No reset options are currently supported")
        self.simulator.reset(self.np_random)
        self._needs_reset = False
        if self.render_mode == "human":
            self.render()
        return self.simulator.observation(), self._info()

    def step(self, action: int):
        if self._needs_reset:
            raise RuntimeError("Call reset() before stepping or after an episode ends")
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action!r}; expected 0, 1, or 2")
        passed = self.simulator.step(Action(action))
        terminated = self.simulator.crashed
        truncated = not terminated and self.simulator.steps >= self.config.max_episode_steps
        reward = (
            self.rewards.crash
            if terminated
            else self.rewards.survive + self.rewards.pass_obstacle * passed
        )
        self._needs_reset = terminated or truncated
        if self.render_mode == "human":
            self.render()
        return self.simulator.observation(), reward, terminated, truncated, self._info()

    def _info(self) -> dict[str, Any]:
        sim = self.simulator
        return {
            "score": sim.distance / 10.0,
            "distance": sim.distance,
            "speed": sim.speed,
            "obstacles_cleared": sim.obstacles_cleared,
            "episode_steps": sim.steps,
            "survival_time": sim.steps / self.config.fps,
            "crashed": sim.crashed,
        }

    def capture_frame(self) -> np.ndarray:
        """Render RGB pixels without opening a window or advancing time."""
        sim = self.simulator
        frame = np.full((sim.HEIGHT, sim.WIDTH, 3), 255, dtype=np.uint8)
        frame[sim.GROUND : sim.GROUND + 2] = 90

        def rectangle(x, bottom, width, height, color):
            x0, x1 = max(0, int(x)), min(sim.WIDTH, int(x + width))
            y0 = max(0, int(sim.GROUND - bottom - height))
            y1 = min(sim.HEIGHT, int(sim.GROUND - bottom))
            if x1 > x0 and y1 > y0:
                frame[y0:y1, x0:x1] = color

        rectangle(
            sim.DINO_X,
            sim.y,
            sim.DINO_WIDTH,
            sim.DUCK_HEIGHT if sim.ducking else sim.STAND_HEIGHT,
            45,
        )
        for obstacle in sim.obstacles:
            rectangle(obstacle.x, obstacle.bottom, obstacle.width, obstacle.height, 90)
        return frame

    def render(self) -> np.ndarray | None:
        if self.render_mode is None:
            return None
        sim = self.simulator
        frame = self.capture_frame()
        if self.render_mode == "rgb_array":
            return frame
        try:
            import pygame
        except ImportError as exc:
            raise RuntimeError('Human rendering requires: pip install -e ".[render]"') from exc
        if self._screen is None:
            pygame.display.init()
            self._screen = pygame.display.set_mode((sim.WIDTH, sim.HEIGHT))
            self._clock = pygame.time.Clock()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                raise KeyboardInterrupt("Simulator window closed")
        pygame.display.set_caption(f"Dino simulator | score {sim.distance / 10:.1f}")
        pygame.surfarray.blit_array(self._screen, frame.transpose(1, 0, 2))
        pygame.display.flip()
        self._clock.tick(self.config.fps)
        return None

    def close(self) -> None:
        if self._screen is not None:
            import pygame

            pygame.display.quit()
        self._screen = self._clock = None
        self._needs_reset = True
