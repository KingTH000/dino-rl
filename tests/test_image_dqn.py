from dataclasses import replace

import numpy as np
import pytest
import torch

from dino_rl.config import Config, DQNConfig, ImageConfig, SimulatorConfig, TrainingConfig
from dino_rl.memory.image_replay_buffer import ImageReplayBuffer
from dino_rl.models.cnn_network import CNNQNetwork
from dino_rl.training.evaluator import evaluate
from dino_rl.training.factory import make_agent
from dino_rl.training.trainer import Trainer
from dino_rl.utils.checkpoint import load_checkpoint


def config(steps=20):
    return Config(
        observation="image",
        device="cpu",
        image=ImageConfig(width=40, height=40),
        simulator=SimulatorConfig(max_episode_steps=7),
        dqn=DQNConfig(
            batch_size=4,
            replay_capacity=16,
            learning_starts=4,
            cnn_hidden_size=16,
            target_update_frequency=4,
            train_frequency=2,
        ),
        training=TrainingConfig(
            steps=steps,
            log_frequency=1,
            checkpoint_frequency=4,
            evaluation_frequency=10,
            evaluation_episodes=2,
            tensorboard=False,
            keep_periodic_checkpoints=2,
        ),
    )


def frames(value, shape=(4, 3, 3)):
    return np.stack([np.full(shape[1:], value + i, dtype=np.uint8) for i in range(shape[0])])


def test_image_replay_shift_wrap_aliasing_and_memory():
    buffer = ImageReplayBuffer(3, (4, 3, 3))
    for i in range(6):
        buffer.add(frames(i), i % 3, float(i), frames(i + 1), i == 5)
    batch = buffer.sample(3)
    assert len(buffer) == 3 and set(batch.rewards) == {3, 4, 5}
    assert batch.states.dtype == np.uint8
    np.testing.assert_array_equal(batch.states[:, 1:], batch.next_states[:, :-1])
    assert batch.terminated.sum() == 1
    batch.states[:] = 0
    assert buffer.sample(3).states.min() >= 3
    assert buffer.storage_bytes == 3 * (5 * 3 * 3 + 13)
    assert buffer.storage_bytes < 3 * (8 * 3 * 3 * 4)


def test_replay_normalized_roundtrip_and_episode_boundary_guard():
    buffer = ImageReplayBuffer(4, (4, 3, 3))
    a, b = frames(10), frames(11)
    buffer.add(a.astype(np.float32) / 255, 1, 0.1, b.astype(np.float32) / 255, False)
    a[:] = 0
    np.testing.assert_array_equal(buffer.sample(1).states[0], frames(10))
    with pytest.raises(ValueError, match="BEFORE reset"):
        buffer.add(frames(10), 0, 1, frames(0), False)
    with pytest.raises(ValueError, match="exceeding"):
        ImageReplayBuffer(100000, memory_limit_mb=1024)
    with pytest.raises(ValueError):
        buffer.sample(0)


def test_replay_restore_preserves_samples_and_rng():
    a = ImageReplayBuffer(3, (4, 3, 3), seed=17)
    for i in range(5):
        a.add(frames(i), 1, 0, frames(i + 1), False)
    b = ImageReplayBuffer(3, (4, 3, 3), seed=1)
    b.load_state_dict(a.state_dict())
    for _ in range(3):
        np.testing.assert_array_equal(a.sample(2).states, b.sample(2).states)


def test_cnn_shapes_updates_and_pixel_scaling():
    torch.set_num_threads(1)
    network = CNNQNetwork()
    assert network(torch.zeros(2, 4, 84, 84)).shape == (2, 3)
    assert network.features(torch.zeros(2, 4, 84, 84)).shape == (2, 3136)
    with pytest.raises(ValueError, match="too small"):
        CNNQNetwork((4, 32, 32))
    c = config()
    agent = make_agent(c, torch.device("cpu"))
    buffer = ImageReplayBuffer(4, (4, 40, 40))
    for i in range(4):
        buffer.add(frames(i, (4, 40, 40)), i % 3, 1, frames(i + 1, (4, 40, 40)), False)
    batch = buffer.sample(4)
    torch.testing.assert_close(
        agent._input_tensor(batch.states), torch.tensor(batch.states, dtype=torch.float32) / 255
    )
    before = [p.clone() for p in agent.online.parameters()]
    metrics = agent.update(batch)
    assert all(np.isfinite(v) for v in metrics.values())
    assert any(not torch.equal(a, b) for a, b in zip(before, agent.online.parameters()))
    assert all(p.grad is None for p in agent.target.parameters())


def test_image_exact_resume_and_checkpoint_retention(tmp_path):
    full = Trainer(config())
    full.run(tmp_path / "full")
    part = Trainer(config(13))
    part.run(tmp_path / "part")
    checkpoint = tmp_path / "part/checkpoints/latest.pt"
    data = load_checkpoint(checkpoint)
    assert data["observation"] == "image" and data["image_frames"].shape == (4, 40, 40)
    resumed = Trainer(config())
    resumed.run(tmp_path / "resumed", resume=checkpoint)
    assert resumed.progress == full.progress
    for name in ("online", "target"):
        for a, b in zip(
            getattr(resumed.agent, name).parameters(), getattr(full.agent, name).parameters()
        ):
            assert torch.equal(a, b)
    np.testing.assert_array_equal(resumed.current_observation, full.current_observation)
    np.testing.assert_array_equal(resumed.replay.sample(10).states, full.replay.sample(10).states)
    assert len(list((tmp_path / "full/checkpoints").glob("step_*.pt"))) == 2
    with pytest.raises(ValueError, match="Cannot change observation"):
        Trainer(replace(config(), observation="state")).run(tmp_path / "bad", resume=checkpoint)


def test_pixel_only_evaluation_and_no_learning():
    c = config()
    agent = make_agent(c, torch.device("cpu"))
    seen = []
    original = agent.act

    def record(observation, epsilon=0):
        seen.append(observation.shape)
        assert epsilon == 0
        return original(observation, epsilon)

    agent.act = record
    result = evaluate(c, agent, 2, 20000)
    assert seen and all(shape == (4, 40, 40) for shape in seen)
    assert result["summary"]["truncated_episodes"] == 2 and agent.updates == 0
