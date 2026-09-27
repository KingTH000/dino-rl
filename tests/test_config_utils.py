import random

import numpy as np
import pytest
import torch

from dino_rl.config import SimulatorConfig, load_config
from dino_rl.utils.device import select_device
from dino_rl.utils.seeds import seed_everything


def test_config_defaults_and_override(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("seed: 7\nsimulator:\n  max_episode_steps: 100\n")
    config = load_config(path)
    assert config.seed == 7 and config.simulator.max_episode_steps == 100
    assert config.rewards.crash == -10
    path.write_text("unknown: 1")
    with pytest.raises(ValueError, match="Invalid configuration"):
        load_config(path)


@pytest.mark.parametrize(
    "kwargs",
    [{"fps": 0}, {"fps": 0.5}, {"min_gap": -1}, {"bird_probability": 2}, {"gravity": float("nan")}],
)
def test_bad_physics_config(kwargs):
    with pytest.raises(ValueError):
        SimulatorConfig(**kwargs)


def test_seed_repeats_all_rngs():
    seed_everything(42)
    first = random.random(), np.random.rand(), torch.rand(2)
    seed_everything(42)
    second = random.random(), np.random.rand(), torch.rand(2)
    assert first[:2] == second[:2]
    assert torch.equal(first[2], second[2])


@pytest.mark.parametrize(
    "cuda,mps,expected", [(True, True, "cuda"), (False, True, "mps"), (False, False, "cpu")]
)
def test_device_priority(monkeypatch, cuda, mps, expected):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)
    assert select_device().type == expected
    assert select_device("cpu").type == "cpu"
    if not cuda:
        with pytest.raises(RuntimeError, match="unavailable"):
            select_device("cuda")
    with pytest.raises(ValueError):
        select_device("invalid")
