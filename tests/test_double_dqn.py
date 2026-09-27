import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from dino_rl.agents.double_dqn_agent import DoubleDQNAgent
from dino_rl.agents.dqn_agent import DQNAgent
from dino_rl.config import Config, DQNConfig, ImageConfig, SimulatorConfig, TrainingConfig
from dino_rl.training.evaluator import evaluate
from dino_rl.training.factory import make_agent
from dino_rl.training.trainer import Trainer
from dino_rl.utils.checkpoint import load_checkpoint


class LookupQ(nn.Module):
    def __init__(self, rows):
        super().__init__()
        self.values = nn.Parameter(torch.tensor(rows, dtype=torch.float32))

    def forward(self, observation):
        state = observation["state"] if isinstance(observation, dict) else observation
        return self.values[state[:, 0].long()]


@pytest.mark.parametrize("hybrid", [False, True])
def test_double_targets_select_online_evaluate_target_and_mask(hybrid):
    settings = DQNConfig(gamma=0.5)
    online = [[100, 0, 0], [0, 100, 0], [0, 0, 100], [100, 0, 0]]
    target = [[1, 10, 0], [0, 2, 20], [30, 0, 3], [5, 4, 3]]
    double = DoubleDQNAgent(settings, torch.device("cpu"), network=LookupQ(online))
    vanilla = DQNAgent(settings, torch.device("cpu"), network=LookupQ(online))
    with torch.no_grad():
        for agent in (double, vanilla):
            agent.target.values.copy_(torch.tensor(target))
    states = torch.arange(4).float().reshape(4, 1)
    inputs = {"state": states, "image": torch.zeros(4, 4, 84, 84)} if hybrid else states
    rewards = torch.tensor([1.0, 2.0, 3.0, -10.0])
    done = torch.tensor([False, False, False, True])
    # Nonterminal includes a time-limit transition: it must still bootstrap.
    result = double.bellman_targets(rewards, inputs, done)
    torch.testing.assert_close(result, torch.tensor([1.5, 3.0, 4.5, -10.0]))
    torch.testing.assert_close(
        vanilla.bellman_targets(rewards, inputs, done), torch.tensor([6.0, 12.0, 18.0, -10.0])
    )
    assert not result.requires_grad
    assert all(p.grad is None for p in double.online.parameters())
    assert all(p.grad is None and not p.requires_grad for p in double.target.parameters())
    double.sync_target()
    vanilla.sync_target()
    torch.testing.assert_close(
        double.bellman_targets(rewards, inputs, done),
        vanilla.bellman_targets(rewards, inputs, done),
    )


def config(mode="hybrid", steps=24):
    return Config(
        algorithm="double_dqn",
        observation=mode,
        device="cpu",
        image=ImageConfig(width=40, height=40),
        simulator=SimulatorConfig(max_episode_steps=7),
        dqn=DQNConfig(
            batch_size=4,
            replay_capacity=16,
            learning_starts=4,
            train_frequency=2,
            target_update_frequency=4,
            cnn_hidden_size=16,
            hybrid_state_hidden_size=8,
        ),
        training=TrainingConfig(
            steps=steps,
            evaluation_frequency=12,
            evaluation_episodes=2,
            checkpoint_frequency=12,
            tensorboard=False,
            keep_periodic_checkpoints=2,
        ),
    )


def equal(a, b):
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            equal(x, y)
    elif isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    else:
        assert a == b


@pytest.mark.parametrize("mode", ["state", "image", "hybrid"])
def test_double_training_resume_evaluate_all_observations(mode, tmp_path):
    c = config(mode)
    full = Trainer(c)
    full.run(tmp_path / "full")
    part = Trainer(config(mode, 13))
    part.run(tmp_path / "part")
    path = tmp_path / "part/checkpoints/latest.pt"
    checkpoint = load_checkpoint(path)
    assert checkpoint["algorithm"] == "double_dqn"
    assert isinstance(full.agent, DoubleDQNAgent)
    resumed = Trainer(c)
    resumed.run(tmp_path / "resumed", resume=path)
    equal(full.progress, resumed.progress)
    equal(full.current_observation, resumed.current_observation)
    equal(full.agent.online.state_dict(), resumed.agent.online.state_dict())
    equal(full.agent.target.state_dict(), resumed.agent.target.state_dict())
    equal(full.agent.optimizer.state_dict(), resumed.agent.optimizer.state_dict())
    equal(full.replay.sample(10).__dict__, resumed.replay.sample(10).__dict__)
    before = {k: v.clone() for k, v in full.agent.online.state_dict().items()}
    updates = full.agent.updates
    report = evaluate(c, full.agent, 2, 20000)
    assert report["policy"] == "greedy_double_dqn" and full.agent.updates == updates
    equal(before, full.agent.online.state_dict())
    with pytest.raises(ValueError, match="Cannot change algorithm"):
        Trainer(replace(c, algorithm="dqn")).run(tmp_path / "bad", resume=path)
    checkpoint["algorithm"] = "dqn"
    torch.save(checkpoint, tmp_path / "mismatch.pt")
    with pytest.raises(ValueError, match="algorithm metadata"):
        load_checkpoint(tmp_path / "mismatch.pt")


def test_config_factory_and_legacy_defaults():
    assert Config().algorithm == "dqn"
    with pytest.raises(ValueError, match="algorithm"):
        Config(algorithm="unknown")
    assert type(make_agent(Config(), torch.device("cpu"))) is DQNAgent


def test_cli_algorithm_override_and_resume_without_flag(tmp_path):
    from dataclasses import asdict

    import yaml

    c = config("state", 8)
    # Explicit CLI selection overrides a DQN config.
    settings = tmp_path / "config.yaml"
    settings.write_text(yaml.safe_dump(asdict(replace(c, algorithm="dqn"))))
    script = Path(__file__).resolve().parents[1] / "scripts/train.py"
    first = tmp_path / "first"
    second = tmp_path / "second"
    for args in [
        ["--config", str(settings), "--algorithm", "double_dqn", "--output-dir", str(first)],
        [
            "--resume",
            str(first / "checkpoints/latest.pt"),
            "--steps",
            "10",
            "--output-dir",
            str(second),
        ],
    ]:
        result = subprocess.run(
            [sys.executable, str(script), *args], capture_output=True, text=True, timeout=60
        )
        assert result.returncode == 0, result.stderr
    assert json.loads((second / "config.json").read_text())["algorithm"] == "double_dqn"
    assert load_checkpoint(second / "checkpoints/latest.pt")["config"]["algorithm"] == "double_dqn"
