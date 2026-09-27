from dataclasses import replace

import numpy as np
import pytest
import torch

from dino_rl.agents.dqn_agent import DQNAgent, epsilon_at_step
from dino_rl.config import DQNConfig, SimulatorConfig
from dino_rl.envs import DinoEnv
from dino_rl.memory.replay_buffer import ReplayBatch, ReplayBuffer
from dino_rl.models.state_network import StateQNetwork
from dino_rl.preprocessing.state_normalization import StateNormalizer
from dino_rl.utils.seeds import seed_everything


def test_normalization_units_sign_and_copy():
    config = SimulatorConfig()
    normalize = StateNormalizer(config)
    state = np.array(
        [600, config.jump_velocity**2 / (2 * config.gravity), -680, -80, 48, 48, 24, 2, 1, 0],
        dtype=np.float32,
    )
    original = state.copy()
    normalized = normalize(state)
    np.testing.assert_allclose(normalized, [1, 1, -1, -0.1, 1, 1, 0.5, 1, 1, 0])
    np.testing.assert_array_equal(state, original)
    assert normalized.dtype == np.float32
    alternate = StateNormalizer(replace(config, max_speed=1200))
    assert alternate(state)[0] == 0.5
    with pytest.raises(ValueError):
        normalize(np.zeros(9))
    with pytest.raises(ValueError):
        normalize(np.full(10, np.nan))


def test_normalized_rollout_is_finite():
    env = DinoEnv()
    normalize = StateNormalizer(env.config)
    state, _ = env.reset(seed=42)
    env.action_space.seed(42)
    for _ in range(300):
        state, _, terminated, truncated, _ = env.step(env.action_space.sample())
        result = normalize(state)
        assert result.shape == (10,) and np.isfinite(result).all()
        if terminated or truncated:
            state, _ = env.reset()
    env.close()


def test_replay_wraparound_sampling_and_ownership():
    replay = ReplayBuffer(3, seed=1)
    state = np.zeros(10, dtype=np.float32)
    replay.add(state, 0, 0.0, state, False)
    state[:] = 100
    assert replay.sample(1).states[0, 0] == 0
    for value in range(1, 5):
        replay.add(np.full(10, value), value % 3, float(value), np.full(10, value + 1), value == 4)
    assert len(replay) == 3
    batch = replay.sample(3)
    assert set(batch.rewards) == {2, 3, 4}
    assert batch.states.shape == batch.next_states.shape == (3, 10)
    assert batch.actions.dtype == np.int64 and batch.terminated.dtype == np.bool_
    np.testing.assert_array_equal(batch.next_states, batch.states + 1)
    assert batch.terminated.sum() == 1
    batch.states[:] = -99
    assert (replay.sample(3).states >= 2).all()
    with pytest.raises(ValueError):
        replay.sample(4)
    with pytest.raises(ValueError):
        ReplayBuffer(2).sample(1)


def test_replay_sampling_reproducible():
    buffers = [ReplayBuffer(20, seed=17), ReplayBuffer(20, seed=17)]
    for replay in buffers:
        for value in range(20):
            replay.add(np.full(10, value), 1, value, np.zeros(10), False)
    for _ in range(3):
        np.testing.assert_array_equal(buffers[0].sample(5).states, buffers[1].sample(5).states)


@pytest.mark.parametrize(
    "state,action,reward,terminated",
    [
        (np.zeros(9), 0, 0.0, False),
        (np.zeros(10), 3, 0.0, False),
        (np.zeros(10), 0.5, 0.0, False),
        (np.zeros(10), 0, np.nan, False),
        (np.zeros(10), 0, 0.0, 1),
    ],
)
def test_invalid_transitions(state, action, reward, terminated):
    with pytest.raises(ValueError):
        ReplayBuffer(2).add(state, action, reward, np.zeros(10), terminated)


def test_network_shapes():
    model = StateQNetwork()
    assert model(torch.zeros(10)).shape == (3,)
    assert model(torch.zeros(64, 10)).shape == (64, 3)


@pytest.mark.parametrize(
    "step,expected", [(-10, 1.0), (0, 1.0), (100_000, 0.525), (200_000, 0.05), (300_000, 0.05)]
)
def test_epsilon_decay(step, expected):
    assert epsilon_at_step(step, DQNConfig()) == pytest.approx(expected)


def constant_output(network, values):
    with torch.no_grad():
        for parameter in network.parameters():
            parameter.zero_()
        network.layers[-1].bias.copy_(torch.tensor(values, dtype=torch.float32))


def test_bellman_targets_use_target_max_and_terminal_mask():
    agent = DQNAgent(DQNConfig(), torch.device("cpu"))
    # Online would select action 0; vanilla DQN must still use target action 1.
    constant_output(agent.online, [100, 0, 0])
    constant_output(agent.target, [1, 3, 2])
    targets = agent.bellman_targets(
        torch.tensor([-10.0, 0.1]), torch.zeros(2, 10), torch.tensor([True, False])
    )
    torch.testing.assert_close(targets, torch.tensor([-10.0, 0.1 + 0.99 * 3]))
    assert not targets.requires_grad


def test_huber_adam_update_and_explicit_target_sync():
    agent = DQNAgent(DQNConfig(), torch.device("cpu"))
    constant_output(agent.online, [0, 0, 0])
    agent.sync_target()
    target_before = [p.clone() for p in agent.target.parameters()]
    batch = ReplayBatch(
        np.zeros((2, 10), dtype=np.float32),
        np.array([0, 1]),
        np.array([2, -2], dtype=np.float32),
        np.zeros((2, 10), dtype=np.float32),
        np.array([True, True]),
    )
    metrics = agent.update(batch)
    assert metrics["loss"] == pytest.approx(1.5)  # Huber(|2|) = 2 - 0.5
    assert isinstance(agent.optimizer, torch.optim.Adam)
    assert agent.updates == 1
    assert agent.online.layers[-1].bias[0] > 0 and agent.online.layers[-1].bias[1] < 0
    assert all(torch.equal(a, b) for a, b in zip(target_before, agent.target.parameters()))
    assert all(p.grad is None and not p.requires_grad for p in agent.target.parameters())
    agent.sync_target()
    for online, target in zip(agent.online.parameters(), agent.target.parameters()):
        torch.testing.assert_close(online, target)
        assert online.data_ptr() != target.data_ptr()


def test_greedy_action_and_seeded_exploration():
    seed_everything(42)
    left = DQNAgent(DQNConfig(), torch.device("cpu"), seed=7)
    right = DQNAgent(DQNConfig(), torch.device("cpu"), seed=7)
    constant_output(left.online, [0, 1, 5])
    state = np.zeros(10, dtype=np.float32)
    before = [p.clone() for p in left.online.parameters()]
    assert left.act(state, 0) == 2
    actions = [left.act(state, 1) for _ in range(100)]
    assert actions == [right.act(state, 1) for _ in range(100)]
    assert set(actions) == {0, 1, 2}
    assert left.updates == 0
    assert all(torch.equal(a, b) for a, b in zip(before, left.online.parameters()))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"gamma": 1.1},
        {"learning_rate": -1},
        {"learning_starts": -1},
        {"batch_size": 0},
        {"replay_capacity": 5},
        {"epsilon_start": 0.01},
        {"target_update_frequency": 0},
        {"train_frequency": True},
        {"gamma": float("nan")},
        {"epsilon_decay_steps": 0},
    ],
)
def test_invalid_dqn_config(kwargs):
    with pytest.raises(ValueError):
        DQNConfig(**kwargs)
