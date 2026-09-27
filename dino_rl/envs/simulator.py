"""Fixed-step, headless Dino physics, independent of Gymnasium and rendering."""

from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from dino_rl.config import SimulatorConfig


class Action(IntEnum):
    NO_ACTION = 0
    JUMP = 1
    DUCK = 2


@dataclass
class Obstacle:
    x: float
    width: float
    height: float
    bottom: float
    kind: int  # 1 = cactus, 2 = bird
    passed: bool = False


class DinoSimulator:
    """Y is height above ground. Actions apply for exactly one physics tick.

    Holding JUMP causes another jump on landing. DUCK only changes the grounded
    hitbox, and NO_ACTION releases duck. Collision uses swept horizontal bounds.
    """

    WIDTH, HEIGHT, GROUND = 800, 240, 200
    DINO_X, DINO_WIDTH = 80.0, 36.0
    STAND_HEIGHT, DUCK_HEIGHT = 48.0, 24.0

    def __init__(self, config: SimulatorConfig):
        self.config = config
        self.reset(np.random.default_rng())

    def reset(self, rng: np.random.Generator) -> None:
        self.rng = rng
        self.y = self.velocity_y = self.distance = 0.0
        self.speed = self.config.initial_speed
        self.ducking = self.crashed = False
        self.steps = self.obstacles_cleared = 0
        self.obstacles: list[Obstacle] = []
        self._spawn(float(self.WIDTH))

    def _spawn(self, x: float) -> None:
        bird = self.rng.random() < self.config.bird_probability
        if bird:
            obstacle = Obstacle(x, 42.0, 24.0, 30.0, 2)
        else:
            width = float(self.rng.choice([22, 32, 44]))
            obstacle = Obstacle(x, width, float(self.rng.choice([38, 48, 58])), 0.0, 1)
        self.obstacles.append(obstacle)
        self.next_gap = float(self.rng.uniform(self.config.min_gap, self.config.max_gap))

    def step(self, action: Action) -> int:
        """Advance one tick and return newly cleared obstacles."""
        action = Action(action)
        if self.crashed:
            raise RuntimeError("Reset the simulator after a crash")
        dt = 1.0 / self.config.fps
        self.ducking = action == Action.DUCK and self.y == 0
        if action == Action.JUMP and self.y == 0:
            self.velocity_y = self.config.jump_velocity
        if self.y > 0 or self.velocity_y > 0:
            self.y = max(0.0, self.y + self.velocity_y * dt - 0.5 * self.config.gravity * dt**2)
            self.velocity_y -= self.config.gravity * dt
            if self.y == 0:
                self.velocity_y = 0.0
        self.speed = min(self.config.max_speed, self.speed + self.config.acceleration * dt)
        travel = self.speed * dt
        self.distance += travel
        self.steps += 1
        height = self.DUCK_HEIGHT if self.ducking else self.STAND_HEIGHT
        cleared = 0
        for obstacle in self.obstacles:
            old_right = obstacle.x + obstacle.width
            obstacle.x -= travel
            horizontal = obstacle.x < self.DINO_X + self.DINO_WIDTH and old_right > self.DINO_X
            vertical = (
                self.y < obstacle.bottom + obstacle.height and self.y + height > obstacle.bottom
            )
            if horizontal and vertical:
                self.crashed = True
            if not obstacle.passed and obstacle.x + obstacle.width <= self.DINO_X:
                obstacle.passed = True
                cleared += 1
        if self.crashed:
            cleared = 0
        self.obstacles_cleared += cleared
        last = self.obstacles[-1]
        if last.x + last.width + self.next_gap <= self.WIDTH:
            self._spawn(last.x + last.width + self.next_gap)
        self.obstacles = [o for o in self.obstacles if o.x + o.width > 0]
        return cleared

    def observation(self) -> np.ndarray:
        """Raw state; normalization belongs to Milestone 2 preprocessing."""
        nearest = next((o for o in self.obstacles if not o.passed), None)
        return np.array(
            [
                self.speed,
                self.y,
                self.velocity_y,
                nearest.x - (self.DINO_X + self.DINO_WIDTH) if nearest else self.WIDTH,
                nearest.width if nearest else 0,
                nearest.height if nearest else 0,
                nearest.bottom if nearest else 0,
                nearest.kind if nearest else 0,
                float(self.y > 0),
                float(self.ducking),
            ],
            dtype=np.float32,
        )
