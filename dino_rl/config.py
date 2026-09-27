"""Small, strict configuration objects; physics uses pixels and seconds."""

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True)
class SimulatorConfig:
    fps: int = 60
    initial_speed: float = 300.0
    max_speed: float = 600.0
    acceleration: float = 3.0
    gravity: float = 1800.0
    jump_velocity: float = 650.0
    min_gap: float = 360.0
    max_gap: float = 600.0
    bird_probability: float = 0.25
    max_episode_steps: int = 6000

    def __post_init__(self):
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"simulator.{name} must be numeric")
            if not math.isfinite(value):
                raise ValueError(f"simulator.{name} must be finite")
        for name in ("fps", "max_episode_steps"):
            value = getattr(self, name)
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("initial_speed", "max_speed", "gravity", "jump_velocity", "min_gap"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.acceleration < 0 or self.max_speed < self.initial_speed:
            raise ValueError("Require acceleration >= 0 and max_speed >= initial_speed")
        if self.max_gap < self.min_gap or not 0 <= self.bird_probability <= 1:
            raise ValueError("Require max_gap >= min_gap and bird_probability in [0, 1]")


@dataclass(frozen=True)
class RewardConfig:
    survive: float = 0.1
    pass_obstacle: float = 1.0
    crash: float = -10.0

    def __post_init__(self):
        for value in asdict(self).values():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("Rewards must be numeric")
            if not math.isfinite(value):
                raise ValueError("Rewards must be finite")


@dataclass(frozen=True)
class DQNConfig:
    gamma: float = 0.99
    learning_rate: float = 1e-4
    batch_size: int = 64
    replay_capacity: int = 100_000
    learning_starts: int = 5_000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 200_000
    target_update_frequency: int = 5_000
    train_frequency: int = 1
    hidden_size: int = 128
    cnn_hidden_size: int = 256
    hybrid_state_hidden_size: int = 64
    replay_memory_limit_mb: int = 1024

    def __post_init__(self):
        for name in (
            "batch_size",
            "replay_capacity",
            "epsilon_decay_steps",
            "target_update_frequency",
            "train_frequency",
            "hidden_size",
            "cnn_hidden_size",
            "hybrid_state_hidden_size",
            "replay_memory_limit_mb",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"dqn.{name} must be a positive integer")
        if type(self.learning_starts) is not int or self.learning_starts < 0:
            raise ValueError("dqn.learning_starts must be a nonnegative integer")
        if self.replay_capacity < self.batch_size:
            raise ValueError("replay_capacity must be at least batch_size")
        for name in ("gamma", "learning_rate", "epsilon_start", "epsilon_end"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"dqn.{name} must be a finite number")
        if not 0 <= self.gamma <= 1 or self.learning_rate <= 0:
            raise ValueError("Require gamma in [0, 1] and learning_rate > 0")
        if not 0 <= self.epsilon_end <= self.epsilon_start <= 1:
            raise ValueError("Require 0 <= epsilon_end <= epsilon_start <= 1")


@dataclass(frozen=True)
class TrainingConfig:
    steps: int = 500_000
    episodes: int | None = None
    log_frequency: int = 100
    progress_frequency: int = 5000
    torch_threads: int = 1
    checkpoint_frequency: int = 25_000
    evaluation_frequency: int = 25_000
    evaluation_episodes: int = 10
    evaluation_seed: int = 10_000
    tensorboard: bool = True
    keep_periodic_checkpoints: int = 0  # 0 keeps all; image profile retains two.

    def __post_init__(self):
        for name in (
            "steps",
            "log_frequency",
            "progress_frequency",
            "torch_threads",
            "checkpoint_frequency",
            "evaluation_frequency",
            "evaluation_episodes",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"training.{name} must be a positive integer")
        if self.episodes is not None and (type(self.episodes) is not int or self.episodes <= 0):
            raise ValueError("training.episodes must be null or a positive integer")
        if type(self.evaluation_seed) is not int or not 0 <= self.evaluation_seed < 2**32:
            raise ValueError("evaluation_seed must be an integer in [0, 2**32)")
        if type(self.tensorboard) is not bool:
            raise ValueError("tensorboard must be a boolean")
        if type(self.keep_periodic_checkpoints) is not int or self.keep_periodic_checkpoints < 0:
            raise ValueError("keep_periodic_checkpoints must be a nonnegative integer")


@dataclass(frozen=True)
class ImageConfig:
    """Crop is (left, top, right, bottom), with right/bottom exclusive; None uses all pixels."""

    crop: tuple[int, int, int, int] | None = (0, 24, 800, 210)
    width: int = 84
    height: int = 84
    stack_size: int = 4

    def __post_init__(self):
        for name in ("width", "height", "stack_size"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"image.{name} must be a positive integer")
        if self.crop is not None:
            if (
                not isinstance(self.crop, (list, tuple))
                or len(self.crop) != 4
                or any(type(value) is not int for value in self.crop)
            ):
                raise ValueError("image.crop must be null or four integer pixel coordinates")
            left, top, right, bottom = self.crop
            if not 0 <= left < right or not 0 <= top < bottom:
                raise ValueError("image.crop must have nonnegative origin and positive area")
            object.__setattr__(self, "crop", tuple(self.crop))


@dataclass(frozen=True)
class Config:
    seed: int = 42
    device: str = "auto"
    observation: str = "state"
    algorithm: str = "dqn"
    simulator: SimulatorConfig = field(default_factory=SimulatorConfig)
    rewards: RewardConfig = field(default_factory=RewardConfig)
    dqn: DQNConfig = field(default_factory=DQNConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    image: ImageConfig = field(default_factory=ImageConfig)

    def __post_init__(self):
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")
        if self.device not in ("auto", "cpu", "mps", "cuda"):
            raise ValueError("device must be auto, cpu, mps, or cuda")
        if self.observation not in ("state", "image", "hybrid"):
            raise ValueError("observation must be state, image, or hybrid")
        if self.algorithm not in ("dqn", "double_dqn"):
            raise ValueError("algorithm must be dqn or double_dqn")


def load_config(path: str | Path) -> Config:
    """Load YAML, rejecting unknown keys and invalid parameter values."""
    with Path(path).open() as handle:
        data = yaml.safe_load(handle)
    return config_from_dict(data)


def config_from_dict(data: dict) -> Config:
    """Reconstruct validated nested configuration, including from a checkpoint."""
    if not isinstance(data, dict):
        raise ValueError("Configuration must be a YAML mapping")
    try:
        return Config(
            **{
                **data,
                "simulator": SimulatorConfig(**data.get("simulator", {})),
                "rewards": RewardConfig(**data.get("rewards", {})),
                "dqn": DQNConfig(**data.get("dqn", {})),
                "training": TrainingConfig(**data.get("training", {})),
                "image": ImageConfig(**data.get("image", {})),
            }
        )
    except TypeError as exc:
        raise ValueError(f"Invalid configuration: {exc}") from exc
