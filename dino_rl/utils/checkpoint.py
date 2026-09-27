"""Versioned, atomic checkpoints using tensors and primitive data only."""

import os
import random
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from dino_rl.agents.dqn_agent import epsilon_at_step
from dino_rl.config import config_from_dict
from dino_rl.envs.simulator import Obstacle

FORMAT_VERSION = 2
SIM_FIELDS = (
    "y",
    "velocity_y",
    "distance",
    "speed",
    "ducking",
    "crashed",
    "steps",
    "obstacles_cleared",
    "next_gap",
)


def save_checkpoint(path: str | Path, trainer) -> None:
    """Save optimizer, replay, partial episode and RNGs, not just model weights."""
    sim = trainer.env.unwrapped.simulator
    numpy_rng = np.random.get_state()
    payload = {
        "format_version": FORMAT_VERSION,
        "algorithm": trainer.config.algorithm,
        "observation": trainer.config.observation,
        "config": asdict(trainer.config),
        "online": trainer.agent.online.state_dict(),
        "target": trainer.agent.target.state_dict(),
        "optimizer": trainer.agent.optimizer.state_dict(),
        "updates": trainer.agent.updates,
        "progress": trainer.progress.copy(),
        "epsilon": epsilon_at_step(trainer.progress["step"], trainer.config.dqn),
        "replay": trainer.replay.state_dict(),
        "image_frames": torch.from_numpy(trainer.env.frame_stack.as_uint8())
        if trainer.config.observation in ("image", "hybrid")
        else None,
        "agent_rng": trainer.agent.rng.bit_generator.state,
        "simulator": {
            **{name: getattr(sim, name) for name in SIM_FIELDS},
            "obstacles": [asdict(o) for o in sim.obstacles],
            "rng": sim.rng.bit_generator.state,
        },
        "python_rng": random.getstate(),
        "torch_rng": torch.get_rng_state(),
        "numpy_rng": [numpy_rng[0], numpy_rng[1].tolist(), *numpy_rng[2:]],
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "mps_rng": torch.mps.get_rng_state() if torch.backends.mps.is_available() else None,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Replace only once serialization succeeds, so a failed write preserves the last checkpoint.
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_checkpoint(path: str | Path) -> dict:
    data = torch.load(path, map_location="cpu", weights_only=True)
    if (
        data.get("format_version") not in (1, FORMAT_VERSION)
        or data.get("algorithm") not in ("dqn", "double_dqn")
        or data.get("observation") not in ("state", "image", "hybrid")
    ):
        raise ValueError("Unsupported checkpoint version, algorithm or observation mode")
    config = config_from_dict(data["config"])
    if config.algorithm != data["algorithm"]:
        raise ValueError("Checkpoint algorithm metadata does not match its configuration")
    if config.observation != data["observation"]:
        raise ValueError("Checkpoint observation metadata does not match its configuration")
    return data


def restore_training(trainer, data: dict) -> None:
    saved = config_from_dict(data["config"])
    for name in ("algorithm", "seed", "simulator", "rewards", "dqn", "observation", "image"):
        if getattr(trainer.config, name) != getattr(saved, name):
            raise ValueError(f"Cannot change {name} when resuming; start a new experiment")
    trainer.agent.online.load_state_dict(data["online"])
    trainer.agent.target.load_state_dict(data["target"])
    trainer.agent.optimizer.load_state_dict(data["optimizer"])
    trainer.agent.updates = data["updates"]
    trainer.agent.rng.bit_generator.state = data["agent_rng"]
    trainer.replay.load_state_dict(data["replay"])
    trainer.progress = data["progress"].copy()
    trainer.env.reset(seed=saved.seed)
    sim = trainer.env.unwrapped.simulator
    for name in SIM_FIELDS:
        setattr(sim, name, data["simulator"][name])
    sim.obstacles = [Obstacle(**o) for o in data["simulator"]["obstacles"]]
    sim.rng.bit_generator.state = data["simulator"]["rng"]
    if trainer.config.observation in ("image", "hybrid"):
        frames = data["image_frames"].numpy()
        trainer.env.frame_stack.reset(frames[0])
        for frame in frames[1:]:
            trainer.env.frame_stack.append(frame)
        trainer.current_observation = trainer.env.frame_stack.observation()
        if trainer.config.observation == "hybrid":
            trainer.current_observation = trainer.env.combine(trainer.current_observation)
    else:
        trainer.current_observation = trainer.normalizer(sim.observation())
    random.setstate(data["python_rng"])
    state = data["numpy_rng"]
    np.random.set_state((state[0], np.array(state[1], dtype=np.uint32), *state[2:]))
    torch.set_rng_state(data["torch_rng"])
    if data["cuda_rng"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(data["cuda_rng"])
    if data["mps_rng"] is not None and torch.backends.mps.is_available():
        torch.mps.set_rng_state(data["mps_rng"])
