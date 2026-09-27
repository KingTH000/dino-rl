import csv
import json
from dataclasses import replace

import numpy as np
import pytest
import torch

from dino_rl.config import Config, DQNConfig, SimulatorConfig, TrainingConfig, load_config
from dino_rl.training.trainer import Trainer


def small_config():
    return Config(
        device="cpu",
        simulator=SimulatorConfig(max_episode_steps=5),
        dqn=DQNConfig(
            batch_size=4,
            replay_capacity=32,
            learning_starts=8,
            target_update_frequency=5,
            train_frequency=2,
        ),
        training=TrainingConfig(steps=20, log_frequency=1),
    )


def test_loop_warmup_updates_sync_and_truncation(tmp_path):
    trainer = Trainer(small_config())
    summary = trainer.run(tmp_path / "run")
    assert summary["first_update_step"] == 8
    assert summary["updates"] == 7  # 8, 10, 12, 14, 16, 18, 20
    assert summary["target_syncs"] == 4
    assert summary["completed_episodes"] == 4 and summary["partial_episode_steps"] == 0
    assert summary["online_weights_changed"]
    assert summary["last_update"]["loss"] >= 0
    for online, target in zip(trainer.agent.online.parameters(), trainer.agent.target.parameters()):
        torch.testing.assert_close(online, target)
    batch = trainer.replay.sample(20)
    assert not batch.terminated.any()  # Time limits must bootstrap.
    # At step 5, the next state is the final state, not a reset observation.
    assert trainer.replay.next_states[4, 0] > trainer.replay.states[0, 0]
    assert trainer.replay.states[5, 0] == trainer.replay.states[0, 0]
    with (tmp_path / "run/episodes.csv").open() as handle:
        episodes = list(csv.DictReader(handle))
    assert len(episodes) == 4 and all(row["truncated"] == "True" for row in episodes)
    metadata = json.loads((tmp_path / "run/config.json").read_text())
    assert metadata["config"]["seed"] == 42 and len(metadata["normalization_scales"]) == 10


def test_no_learning_before_warmup_and_partial_episode(tmp_path):
    config = small_config()
    config = replace(config, training=replace(config.training, steps=7))
    trainer = Trainer(config)
    summary = trainer.run(tmp_path / "warmup")
    assert summary["updates"] == 0 and summary["first_update_step"] is None
    assert not summary["online_weights_changed"]
    assert summary["completed_episodes"] == 1 and summary["partial_episode_steps"] == 2


def test_episode_limit_and_no_overwrite(tmp_path):
    config = small_config()
    config = replace(config, training=replace(config.training, episodes=2))
    trainer = Trainer(config)
    result = trainer.run(tmp_path / "limited")
    assert result["environment_steps"] == 10 and result["completed_episodes"] == 2
    with pytest.raises(RuntimeError, match="new Trainer"):
        trainer.run(tmp_path / "another")
    with pytest.raises(FileExistsError):
        Trainer(config).run(tmp_path / "limited")


def test_seeded_cpu_runs_repeat(tmp_path):
    first = Trainer(small_config())
    a = first.run(tmp_path / "a")
    second = Trainer(small_config())
    b = second.run(tmp_path / "b")
    a.pop("elapsed_seconds")
    b.pop("elapsed_seconds")
    assert a == b
    for left, right in zip(first.agent.online.parameters(), second.agent.online.parameters()):
        assert torch.equal(left, right)
    np.testing.assert_array_equal(first.replay.states[:20], second.replay.states[:20])


def test_training_yaml_and_invalid_config(tmp_path):
    path = tmp_path / "settings.yaml"
    path.write_text("dqn:\n  learning_rate: 0.0002\ntraining:\n  steps: 12\n")
    config = load_config(path)
    assert config.dqn.learning_rate == 0.0002 and config.training.steps == 12
    path.write_text("dqn:\n  misspelled: 12\n")
    with pytest.raises(ValueError):
        load_config(path)
    with pytest.raises(ValueError):
        TrainingConfig(steps=0)
    with pytest.raises(ValueError):
        TrainingConfig(episodes=0)
