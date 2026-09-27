import json
from dataclasses import replace

import pytest
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from dino_rl.agents.dqn_agent import DQNAgent
from dino_rl.config import Config, DQNConfig, SimulatorConfig, TrainingConfig
from dino_rl.training.evaluator import evaluate
from dino_rl.training.trainer import Trainer
from dino_rl.utils.checkpoint import load_checkpoint, save_checkpoint


def settings(steps=40):
    return Config(
        device="cpu",
        simulator=SimulatorConfig(max_episode_steps=7),
        dqn=DQNConfig(
            batch_size=4, replay_capacity=16, learning_starts=4, target_update_frequency=5
        ),
        training=TrainingConfig(
            steps=steps,
            log_frequency=1,
            checkpoint_frequency=10,
            evaluation_frequency=10,
            evaluation_episodes=2,
        ),
    )


def assert_nested_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_nested_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_nested_equal(a, b)
    else:
        assert left == right


@pytest.mark.parametrize("limit,split,total", [(7, 23, 40), (6000, 173, 300)])
def test_exact_cpu_resume_mid_episode_after_replay_wrap(tmp_path, limit, split, total):
    config = replace(settings(total), simulator=SimulatorConfig(max_episode_steps=limit))
    uninterrupted = Trainer(config)
    uninterrupted.run(tmp_path / "full")
    segment = Trainer(replace(config, training=replace(config.training, steps=split)))
    segment.run(tmp_path / "part")
    data = load_checkpoint(tmp_path / "part/checkpoints/latest.pt")
    assert data["progress"]["episode_steps"] > 0
    assert data["replay"]["size"] == 16
    resumed = Trainer(config)
    resumed.run(tmp_path / "resume", resume=tmp_path / "part/checkpoints/latest.pt")
    for name in ("online", "target", "optimizer"):
        assert_nested_equal(
            getattr(uninterrupted.agent, name).state_dict(),
            getattr(resumed.agent, name).state_dict(),
        )
    assert uninterrupted.progress == resumed.progress
    assert_nested_equal(uninterrupted.replay.state_dict(), resumed.replay.state_dict())
    assert uninterrupted.agent.rng.bit_generator.state == resumed.agent.rng.bit_generator.state


def test_checkpoint_files_and_tensorboard(tmp_path):
    trainer = Trainer(settings(12))
    trainer.run(tmp_path / "run")
    folder = tmp_path / "run/checkpoints"
    for name in ("latest.pt", "best.pt", "step_000000010.pt"):
        data = load_checkpoint(folder / name)
        assert data["config"]["seed"] == 42 and "optimizer" in data and "epsilon" in data
    assert load_checkpoint(folder / "latest.pt")["progress"]["step"] == 12
    accumulator = EventAccumulator(str(tmp_path / "run/tensorboard"))
    accumulator.Reload()
    assert {"updates/loss", "updates/epsilon", "episodes/score", "evaluations/mean_score"} <= set(
        accumulator.Tags()["scalars"]
    )
    assert accumulator.Scalars("evaluations/mean_score")[-1].step == 12


def test_evaluation_does_not_change_training_state(tmp_path):
    trainer = Trainer(settings(9))
    trainer.run(tmp_path / "run")
    before = [p.clone() for p in trainer.agent.online.parameters()]
    rng = trainer.agent.rng.bit_generator.state
    updates = trainer.agent.updates
    trainer.agent.online.train()
    first = evaluate(trainer.config, trainer.agent, episodes=3, seed=100)
    second = evaluate(trainer.config, trainer.agent, episodes=3, seed=100)
    assert first == second and trainer.agent.online.training
    assert trainer.agent.updates == updates and trainer.agent.rng.bit_generator.state == rng
    assert all(torch.equal(a, b) for a, b in zip(before, trainer.agent.online.parameters()))
    baseline = evaluate(trainer.config, None, episodes=3, seed=100)
    assert [r["seed"] for r in first["episodes"]] == [r["seed"] for r in baseline["episodes"]]
    assert baseline["summary"]["mean_survival_time"] == pytest.approx(7 / 60)


def test_atomic_write_failure_preserves_checkpoint(tmp_path, monkeypatch):
    trainer = Trainer(settings(8))
    trainer.run(tmp_path / "run")
    checkpoint = tmp_path / "run/checkpoints/latest.pt"
    original = checkpoint.read_bytes()

    def fail(*args, **kwargs):
        raise OSError("simulated disk failure")

    monkeypatch.setattr(torch, "save", fail)
    with pytest.raises(OSError):
        save_checkpoint(checkpoint, trainer)
    assert checkpoint.read_bytes() == original
    assert not list(checkpoint.parent.glob(".latest.pt.*"))


def test_bad_resume_config_and_version(tmp_path):
    trainer = Trainer(settings(8))
    trainer.run(tmp_path / "run")
    path = tmp_path / "run/checkpoints/latest.pt"
    with pytest.raises(ValueError, match="Cannot change dqn"):
        Trainer(replace(settings(), dqn=replace(settings().dqn, gamma=0.5))).run(
            tmp_path / "bad", resume=path
        )
    with pytest.raises(ValueError, match="must exceed"):
        Trainer(settings(8)).run(tmp_path / "exhausted", resume=path)
    data = load_checkpoint(path)
    data["format_version"] = -1
    torch.save(data, tmp_path / "invalid.pt")
    with pytest.raises(ValueError, match="Unsupported"):
        load_checkpoint(tmp_path / "invalid.pt")


def test_best_checkpoint_corresponds_to_highest_logged_evaluation(tmp_path):
    trainer = Trainer(settings(20))
    trainer.run(tmp_path / "run")
    best = load_checkpoint(tmp_path / "run/checkpoints/best.pt")
    agent = DQNAgent(trainer.config.dqn, torch.device("cpu"))
    agent.online.load_state_dict(best["online"])
    score = evaluate(trainer.config, agent, 2, trainer.config.training.evaluation_seed)["summary"][
        "mean_score"
    ]
    scores = [
        json.loads(p.read_text())["summary"]["mean_score"]
        for p in (tmp_path / "run").glob("evaluation_*.json")
    ]
    assert score == max(scores)


def test_evaluation_rejects_invalid_episode_count():
    with pytest.raises(ValueError):
        evaluate(settings(), None, 0)


def test_interrupt_retains_last_completed_checkpoint(tmp_path, monkeypatch):
    trainer = Trainer(settings(20))
    original_step = trainer._step

    def interrupted_step(logger):
        if trainer.progress["step"] == 12:
            raise KeyboardInterrupt
        return original_step(logger)

    monkeypatch.setattr(trainer, "_step", interrupted_step)
    result = trainer.run(tmp_path / "interrupted")
    assert result["interrupted"]
    checkpoint = load_checkpoint(tmp_path / "interrupted/checkpoints/latest.pt")
    assert checkpoint["progress"]["step"] == 10
