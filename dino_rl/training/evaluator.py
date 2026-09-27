"""Greedy or random evaluation in isolated environments with explicit episode seeds."""

import numpy as np

from dino_rl.agents.dqn_agent import DQNAgent
from dino_rl.config import Config
from dino_rl.preprocessing.state_normalization import StateNormalizer
from dino_rl.training.factory import make_env


def evaluate(
    config: Config,
    agent: DQNAgent | None,
    episodes: int = 10,
    seed: int = 20_000,
    render: bool = False,
) -> dict:
    """Use no optimizer or replay. `agent=None` is a uniform random baseline.

    Both policies receive exactly the same world seed for each episode. Each
    episode's action RNG is independent of environment and training RNGs.
    """
    if type(episodes) is not int or episodes <= 0 or type(seed) is not int or seed < 0:
        raise ValueError("episodes must be positive and seed must be nonnegative integers")
    env = make_env(config, "human" if render else None)
    normalize = StateNormalizer(config.simulator)
    was_training = agent.online.training if agent is not None else False
    records = []
    try:
        for index in range(episodes):
            episode_seed = seed + index
            observation, _ = env.reset(seed=episode_seed)
            rng = np.random.default_rng(episode_seed)
            total_reward = 0.0
            while True:
                action = (
                    int(rng.integers(3))
                    if agent is None
                    else agent.act(
                        normalize(observation) if config.observation == "state" else observation,
                        0.0,
                    )
                )
                observation, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                if terminated or truncated:
                    records.append(
                        {
                            "seed": episode_seed,
                            "reward": total_reward,
                            "terminated": terminated,
                            "truncated": truncated,
                            **info,
                        }
                    )
                    break
    finally:
        env.close()
        if agent is not None:
            agent.online.train(was_training)
    scores = [row["score"] for row in records]
    return {
        "policy": "random" if agent is None else f"greedy_{config.algorithm}",
        "episodes": records,
        "summary": {
            "episodes": episodes,
            "mean_score": float(np.mean(scores)),
            "max_score": float(np.max(scores)),
            "std_score": float(np.std(scores)),
            "mean_reward": float(np.mean([r["reward"] for r in records])),
            "mean_survival_time": float(np.mean([r["survival_time"] for r in records])),
            "mean_episode_steps": float(np.mean([r["episode_steps"] for r in records])),
            "mean_obstacles_cleared": float(np.mean([r["obstacles_cleared"] for r in records])),
            "truncated_episodes": sum(r["truncated"] for r in records),
        },
    }
