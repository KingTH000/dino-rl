"""Double DQN separates next-action selection from next-action evaluation."""

import torch

from dino_rl.agents.dqn_agent import DQNAgent


class DoubleDQNAgent(DQNAgent):
    """Reuse DQN exploration, replay updates and checkpoints; change only targets."""

    @torch.no_grad()
    def bellman_targets(
        self,
        rewards: torch.Tensor,
        next_states: torch.Tensor | dict[str, torch.Tensor],
        terminated: torch.Tensor,
    ) -> torch.Tensor:
        best_actions = self.online(next_states).argmax(dim=1, keepdim=True)
        next_values = self.target(next_states).gather(1, best_actions).squeeze(1)
        return rewards + self.config.gamma * torch.where(terminated, 0.0, next_values)
