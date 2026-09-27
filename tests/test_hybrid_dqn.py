import subprocess
import sys
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest
import torch
from gymnasium.utils.env_checker import check_env

from dino_rl.config import Config, DQNConfig, ImageConfig, SimulatorConfig, TrainingConfig
from dino_rl.memory.hybrid_replay_buffer import HybridReplayBuffer
from dino_rl.memory.image_replay_buffer import ImageReplayBuffer
from dino_rl.models.hybrid_network import HybridQNetwork
from dino_rl.preprocessing.state_normalization import StateNormalizer
from dino_rl.training.evaluator import evaluate
from dino_rl.training.factory import make_agent, make_env
from dino_rl.training.trainer import Trainer
from dino_rl.utils.checkpoint import load_checkpoint


def test_fresh_evaluation_import_order():
    """The standalone CLI imports normalization before environment constructors."""
    result = subprocess.run(
        [sys.executable, "-c", "from dino_rl.training.evaluator import evaluate"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def config(steps=24):
    return Config(
        observation="hybrid",
        device="cpu",
        image=ImageConfig(width=40, height=40),
        simulator=SimulatorConfig(max_episode_steps=7),
        dqn=DQNConfig(
            batch_size=4,
            replay_capacity=16,
            learning_starts=4,
            cnn_hidden_size=16,
            hybrid_state_hidden_size=8,
            target_update_frequency=4,
            train_frequency=2,
        ),
        training=TrainingConfig(
            steps=steps,
            checkpoint_frequency=4,
            evaluation_frequency=12,
            evaluation_episodes=2,
            tensorboard=False,
            keep_periodic_checkpoints=2,
        ),
    )


def observation(value, shape=(4, 3, 3)):
    return {
        "image": np.stack([np.full(shape[1:], value + i, dtype=np.uint8) for i in range(shape[0])]),
        "state": np.full(10, value / 10, dtype=np.float32),
    }


def assert_equal(a, b):
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for left, right in zip(a, b):
            assert_equal(left, right)
    elif isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    else:
        assert a == b


def test_hybrid_environment_matches_both_modalities_and_dynamics():
    c = replace(config(), simulator=SimulatorConfig())
    hybrid = make_env(c)
    pixels = make_env(replace(c, observation="image"))
    numeric = make_env(replace(c, observation="state"))
    normalize = StateNormalizer(c.simulator)
    try:
        obs, _ = hybrid.reset(seed=42)
        image, _ = pixels.reset(seed=42)
        state, _ = numeric.reset(seed=42)
        for index in range(500):
            assert hybrid.observation_space.contains(obs)
            assert_equal(obs["image"], image)
            assert_equal(obs["state"], normalize(state))
            action = index % 3
            obs, *h = hybrid.step(action)
            image, *p = pixels.step(action)
            state, *n = numeric.step(action)
            assert h == p == n
            if h[1] or h[2]:
                # Both terminal modalities still describe the final tick.
                assert_equal(obs["image"], image)
                assert_equal(obs["state"], normalize(state))
                obs, _ = hybrid.reset()
                image, _ = pixels.reset()
                state, _ = numeric.reset()
        check_env(hybrid, skip_render_check=True)
    finally:
        for env in (hybrid, pixels, numeric):
            env.close()


def test_hybrid_replay_aligns_modalities_on_wrap_and_restore():
    replay = HybridReplayBuffer(3, (4, 3, 3), seed=17)
    for i in range(7):
        current, following = observation(i), observation(i + 1)
        replay.add(current, i % 3, float(i), following, i == 6)
        current["state"][:] = -1
        current["image"][:] = 0
    batch = replay.sample(3)
    assert set(batch.rewards) == {4, 5, 6}
    np.testing.assert_array_equal(batch.states["image"][:, 0, 0, 0], batch.rewards)
    np.testing.assert_allclose(batch.states["state"][:, 0], batch.rewards / 10)
    np.testing.assert_allclose(batch.next_states["state"][:, 0], (batch.rewards + 1) / 10)
    np.testing.assert_array_equal(batch.states["image"][:, 1:], batch.next_states["image"][:, :-1])
    assert batch.terminated.sum() == 1
    assert replay.storage_bytes == 3 * (5 * 3 * 3 + 13 + 80)
    restored = HybridReplayBuffer(3, (4, 3, 3))
    restored.load_state_dict(replay.state_dict())
    for _ in range(3):
        assert_equal(replay.sample(2).__dict__, restored.sample(2).__dict__)
    batch.states["state"][:] = 100
    assert replay.states.max() < 1
    with pytest.raises(ValueError, match="exceeding"):
        # Image frames alone fit; adding numerical vectors exceeds 1 MiB.
        HybridReplayBuffer(10000, (4, 3, 3), memory_limit_mb=1)
    with pytest.raises(ValueError, match="does not match"):
        ImageReplayBuffer(3, (4, 3, 3)).load_state_dict(replay.state_dict())


def test_hybrid_replay_invalid_state_does_not_partially_insert():
    replay = HybridReplayBuffer(3, (4, 3, 3))
    bad = observation(0)
    bad["state"][0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        replay.add(observation(0), 0, 0, bad, False)
    assert len(replay) == 0 and replay._position == 0
    with pytest.raises(ValueError, match="exactly"):
        replay.add({"image": observation(0)["image"]}, 0, 0, observation(1), False)


def test_hybrid_forward_and_both_branches_learn():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    network = HybridQNetwork()
    inputs = {
        "image": torch.rand(2, 4, 84, 84, requires_grad=True),
        "state": torch.rand(2, 10, requires_grad=True),
    }
    result = network(inputs)
    assert result.shape == (2, 3)
    result.sum().backward()
    assert all(value.grad.abs().sum() > 0 for value in inputs.values())
    c = config()
    agent = make_agent(c, torch.device("cpu"))
    replay = HybridReplayBuffer(4, (4, 40, 40))
    for i in range(4):
        replay.add(observation(i, (4, 40, 40)), i % 3, 1, observation(i + 1, (4, 40, 40)), False)
    batch = replay.sample(4)
    tensors = agent._input_tensor(batch.states)
    torch.testing.assert_close(tensors["state"], torch.from_numpy(batch.states["state"]))
    torch.testing.assert_close(tensors["image"], torch.tensor(batch.states["image"]).float() / 255)
    before = deepcopy(agent.online.state_dict())
    target = deepcopy(agent.target.state_dict())
    metrics = agent.update(batch)
    assert all(np.isfinite(v) for v in metrics.values())
    for branch in ("image_branch", "state_branch", "head"):
        assert any(
            not torch.equal(before[key], value)
            for key, value in agent.online.state_dict().items()
            if key.startswith(branch)
        )
    assert_equal(agent.target.state_dict(), target)
    agent.sync_target()
    assert_equal(agent.online.state_dict(), agent.target.state_dict())
    assert agent.act(observation(0, (4, 40, 40))) in (0, 1, 2)


def test_hybrid_exact_resume_including_optimizer_and_modalities(tmp_path):
    full = Trainer(config())
    full.run(tmp_path / "full")
    part = Trainer(config(13))
    part.run(tmp_path / "part")
    path = tmp_path / "part/checkpoints/latest.pt"
    data = load_checkpoint(path)
    assert data["observation"] == "hybrid" and data["replay"]["kind"] == "hybrid"
    resumed = Trainer(config())
    resumed.run(tmp_path / "resumed", resume=path)
    assert_equal(full.progress, resumed.progress)
    assert_equal(full.current_observation, resumed.current_observation)
    assert_equal(full.agent.optimizer.state_dict(), resumed.agent.optimizer.state_dict())
    assert_equal(full.agent.online.state_dict(), resumed.agent.online.state_dict())
    assert_equal(full.agent.target.state_dict(), resumed.agent.target.state_dict())
    assert_equal(full.replay.sample(10).__dict__, resumed.replay.sample(10).__dict__)
    with pytest.raises(ValueError, match="Cannot change observation"):
        Trainer(replace(config(), observation="image")).run(tmp_path / "bad", resume=path)


def test_hybrid_evaluation_is_greedy_and_isolated():
    c = config()
    agent = make_agent(c, torch.device("cpu"))
    weights = deepcopy(agent.online.state_dict())
    rng = deepcopy(agent.rng.bit_generator.state)
    original = agent.act
    seen = []

    def record(obs, epsilon=0):
        assert epsilon == 0
        seen.append((obs["state"].shape, obs["image"].shape))
        return original(obs, epsilon)

    agent.act = record
    result = evaluate(c, agent, 2, 20000)
    assert seen and all(pair == ((10,), (4, 40, 40)) for pair in seen)
    assert result["summary"]["truncated_episodes"] == 2
    assert agent.updates == 0
    assert_equal(weights, agent.online.state_dict())
    assert_equal(rng, agent.rng.bit_generator.state)
