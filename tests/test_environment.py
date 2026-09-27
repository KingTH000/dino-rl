from dataclasses import replace

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from dino_rl.config import RewardConfig, SimulatorConfig
from dino_rl.envs import Action, DinoEnv
from dino_rl.envs.simulator import Obstacle


def test_gym_contract():
    env = DinoEnv()
    check_env(env, skip_render_check=True)
    env.close()


def test_seeded_trajectory_and_reset():
    a, b = DinoEnv(), DinoEnv()
    for env in (a, b):
        obs, info = env.reset(seed=42)
        assert obs.shape == (10,) and obs.dtype == np.float32
        assert env.observation_space.contains(obs)
        assert info["score"] == 0
    rng = np.random.default_rng(10)
    for action in rng.integers(0, 3, size=300):
        left, right = a.step(action), b.step(action)
        np.testing.assert_array_equal(left[0], right[0])
        assert left[1:] == right[1:]
        if left[2] or left[3]:
            break
    a.reset(seed=42)
    assert a.simulator.steps == a.simulator.obstacles_cleared == 0
    assert a.simulator.y == 0 and not a.simulator.crashed


def test_action_mapping_jump_and_duck_release():
    assert [int(a) for a in Action] == [0, 1, 2]
    env = DinoEnv()
    env.reset(seed=2)
    assert env.step(Action.DUCK)[0][9] == 1
    assert env.step(Action.NO_ACTION)[0][9] == 0
    assert env.step(Action.JUMP)[0][1] > 0
    assert env.step(Action.DUCK)[0][9] == 0
    for _ in range(50):
        env.step(Action.NO_ACTION)
    assert env.simulator.y == 0


def test_collision_and_lifecycle():
    env = DinoEnv(rewards=RewardConfig(crash=-7))
    with pytest.raises(RuntimeError):
        env.step(0)
    env.reset(seed=1)
    env.simulator.obstacles = [Obstacle(115, 22, 48, 0, 1)]
    _, reward, terminated, truncated, _ = env.step(0)
    assert reward == -7 and terminated and not truncated
    with pytest.raises(RuntimeError):
        env.step(0)
    env.reset()
    with pytest.raises(ValueError):
        env.step(3)


def test_bird_can_be_ducked_and_pass_counted_once():
    env = DinoEnv()
    env.reset(seed=1)
    env.simulator.obstacles = [Obstacle(115, 42, 24, 30, 2)]
    rewards = []
    for _ in range(20):
        _, reward, terminated, _, info = env.step(2)
        assert not terminated
        rewards.append(reward)
    assert info["obstacles_cleared"] == 1
    assert rewards.count(1.1) == 1


def test_time_limit_is_truncation():
    env = DinoEnv(replace(SimulatorConfig(), max_episode_steps=1))
    env.reset(seed=0)
    _, reward, terminated, truncated, _ = env.step(0)
    assert reward == 0.1 and not terminated and truncated


def test_render_modes():
    env = DinoEnv(render_mode="rgb_array")
    env.reset(seed=0)
    frame = env.render()
    assert frame.shape == (240, 800, 3) and frame.dtype == np.uint8
    assert frame.min() < frame.max()
    env.close()
    env.close()
    assert DinoEnv().render() is None
    with pytest.raises(ValueError):
        DinoEnv(render_mode="invalid")


def test_headless_random_rollout():
    env = DinoEnv()
    env.action_space.seed(42)
    for episode in range(10):
        env.reset(seed=episode)
        for _ in range(env.config.max_episode_steps):
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
            assert np.isfinite(obs).all() and np.isfinite(reward)
            if terminated or truncated:
                break
        assert terminated or truncated
        assert info["survival_time"] > 0
