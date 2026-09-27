"""Small shared constructors keep training, evaluation and playback in agreement."""

import torch

from dino_rl.agents.double_dqn_agent import DoubleDQNAgent
from dino_rl.agents.dqn_agent import DQNAgent
from dino_rl.config import Config
from dino_rl.envs import DinoEnv, ImageObservationWrapper
from dino_rl.envs.hybrid_observation import HybridObservationWrapper
from dino_rl.models.cnn_network import CNNQNetwork
from dino_rl.models.hybrid_network import HybridQNetwork


def make_agent(config: Config, device: torch.device) -> DQNAgent:
    network = None
    if config.observation in ("image", "hybrid"):
        network_type = HybridQNetwork if config.observation == "hybrid" else CNNQNetwork
        kwargs = (
            {"state_hidden_size": config.dqn.hybrid_state_hidden_size}
            if config.observation == "hybrid"
            else {}
        )
        network = network_type(
            (config.image.stack_size, config.image.height, config.image.width),
            config.dqn.cnn_hidden_size,
            **kwargs,
        )
    agent_type = DoubleDQNAgent if config.algorithm == "double_dqn" else DQNAgent
    return agent_type(config.dqn, device, config.seed, network=network)


def make_env(config: Config, render_mode: str | None = None):
    if config.observation in ("image", "hybrid"):
        wrapper = (
            HybridObservationWrapper if config.observation == "hybrid" else ImageObservationWrapper
        )
        return wrapper(
            DinoEnv(config.simulator, config.rewards, render_mode or "rgb_array"), config.image
        )
    return DinoEnv(config.simulator, config.rewards, render_mode)
