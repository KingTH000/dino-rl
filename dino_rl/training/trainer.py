"""State/image/hybrid DQN with checkpoints, isolated evaluation and live metrics."""

import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from dino_rl.agents.dqn_agent import epsilon_at_step
from dino_rl.config import Config
from dino_rl.memory.hybrid_replay_buffer import HybridReplayBuffer
from dino_rl.memory.image_replay_buffer import ImageReplayBuffer
from dino_rl.memory.replay_buffer import ReplayBuffer
from dino_rl.preprocessing.state_normalization import StateNormalizer
from dino_rl.training.evaluator import evaluate
from dino_rl.training.factory import make_agent, make_env
from dino_rl.utils.checkpoint import load_checkpoint, restore_training, save_checkpoint
from dino_rl.utils.device import select_device
from dino_rl.utils.logger import RunLogger
from dino_rl.utils.seeds import seed_everything


class Trainer:
    def __init__(self, config: Config, render_mode: str | None = None):
        self.config = config
        seed_everything(config.seed)
        torch.set_num_threads(config.training.torch_threads)
        self.device = select_device(config.device)
        self.env = make_env(config, render_mode)
        self.normalizer = StateNormalizer(config.simulator)
        self.agent = make_agent(config, self.device)
        visual_replay = HybridReplayBuffer if config.observation == "hybrid" else ImageReplayBuffer
        self.replay = (
            visual_replay(
                config.dqn.replay_capacity,
                (config.image.stack_size, config.image.height, config.image.width),
                config.seed,
                config.dqn.replay_memory_limit_mb,
            )
            if config.observation in ("image", "hybrid")
            else ReplayBuffer(config.dqn.replay_capacity, seed=config.seed)
        )
        self.current_observation = None
        self._started = False
        self.progress = {
            "step": 0,
            "completed": 0,
            "target_syncs": 0,
            "episode_steps": 0,
            "episode_reward": 0.0,
            "first_update_step": None,
            "last_metrics": None,
        }

    def process_observation(self, raw):
        return self.normalizer(raw) if self.config.observation == "state" else raw

    def _evaluate(self, output: Path, logger: RunLogger, best_score: float) -> float:
        settings = self.config.training
        result = evaluate(
            self.config, self.agent, settings.evaluation_episodes, settings.evaluation_seed
        )
        step = self.progress["step"]
        (output / f"evaluation_{step:09d}.json").write_text(json.dumps(result, indent=2) + "\n")
        logger.record("evaluations", {"environment_step": step, **result["summary"]})
        score = result["summary"]["mean_score"]
        print(
            f"evaluation step={step} score={score:.2f} "
            f"survival={result['summary']['mean_survival_time']:.2f}s "
            f"obstacles={result['summary']['mean_obstacles_cleared']:.1f}",
            flush=True,
        )
        if score > best_score:
            save_checkpoint(output / "checkpoints/best.pt", self)
            return score
        return best_score

    def _step(self, logger: RunLogger) -> bool:
        """Complete one transition/update before making it checkpoint-visible."""
        config, p = self.config, self.progress
        step = p["step"] + 1
        epsilon = epsilon_at_step(step - 1, config.dqn)
        state = self.current_observation
        action = self.agent.act(state, epsilon)
        raw_next, reward, terminated, truncated, info = self.env.step(action)
        next_state = self.process_observation(raw_next)
        self.replay.add(state, action, reward, next_state, terminated)
        self.current_observation = next_state
        p["episode_reward"] += reward
        p["episode_steps"] += 1
        if (
            step >= config.dqn.learning_starts
            and len(self.replay) >= config.dqn.batch_size
            and step % config.dqn.train_frequency == 0
        ):
            p["last_metrics"] = self.agent.update(self.replay.sample(config.dqn.batch_size))
            if p["first_update_step"] is None:
                p["first_update_step"] = step
            if self.agent.updates == 1 or self.agent.updates % config.training.log_frequency == 0:
                logger.record(
                    "updates",
                    {
                        "environment_step": step,
                        "update": self.agent.updates,
                        **p["last_metrics"],
                        "epsilon": epsilon,
                        "replay_size": len(self.replay),
                    },
                )
        if step % config.dqn.target_update_frequency == 0:
            self.agent.sync_target()
            p["target_syncs"] += 1
        if terminated or truncated:
            p["completed"] += 1
            logger.record(
                "episodes",
                {
                    "episode": p["completed"],
                    "environment_step": step,
                    "reward": p["episode_reward"],
                    "score": info["score"],
                    "episode_steps": p["episode_steps"],
                    "survival_time": info["survival_time"],
                    "obstacles_cleared": info["obstacles_cleared"],
                    "epsilon": epsilon,
                    "terminated": terminated,
                    "truncated": truncated,
                },
            )
            p["episode_steps"], p["episode_reward"] = 0, 0.0
            raw, _ = self.env.reset()
            self.current_observation = self.process_observation(raw)
        p["step"] = step
        if step % config.training.progress_frequency == 0:
            print(
                f"step={step} episodes={p['completed']} updates={self.agent.updates} epsilon={epsilon:.3f}",
                flush=True,
            )
        return config.training.episodes is not None and p["completed"] >= config.training.episodes

    def run(self, output_dir: str | Path, resume: str | Path | None = None) -> dict:
        """Resume into a new run directory. Step/episode budgets are cumulative.

        Best checkpoint selection is local to this run, including its initial
        evaluation. Resume preserves learning state, but starts a new log segment.
        """
        if self._started:
            raise RuntimeError("Create a new Trainer for a new run")
        config = self.config
        if resume is not None:
            restore_training(self, load_checkpoint(resume))
        else:
            raw, _ = self.env.reset(seed=config.seed)
            self.current_observation = self.process_observation(raw)
        if config.training.steps <= self.progress["step"]:
            raise ValueError("--steps must exceed the checkpoint's completed step count")
        if (
            config.training.episodes is not None
            and config.training.episodes <= self.progress["completed"]
        ):
            raise ValueError("Episode budget is already exhausted in the checkpoint")
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=False)
        self._started = True
        metadata = {
            "algorithm": config.algorithm,
            "observation": config.observation,
            "config": asdict(config),
            "device": str(self.device),
            "normalization_scales": self.normalizer.scales.tolist()
            if config.observation in ("state", "hybrid")
            else None,
            "replay_storage_bytes": getattr(self.replay, "storage_bytes", None),
            "torch_version": str(torch.__version__),
            "numpy_version": np.__version__,
            "resumed_from": str(Path(resume).resolve()) if resume else None,
        }
        (output / "config.json").write_text(json.dumps(metadata, indent=2) + "\n")
        initial = [p.detach().clone() for p in self.agent.online.parameters()]
        started = time.perf_counter()
        logger = RunLogger(output, config.training.tensorboard)
        best_score = -float("inf")
        interrupted = False
        try:
            save_checkpoint(output / "checkpoints/latest.pt", self)
            best_score = self._evaluate(output, logger, best_score)
            while self.progress["step"] < config.training.steps:
                episode_limit = self._step(logger)
                step = self.progress["step"]
                if (
                    step % config.training.evaluation_frequency == 0
                    or step == config.training.steps
                    or episode_limit
                ):
                    best_score = self._evaluate(output, logger, best_score)
                if step % config.training.checkpoint_frequency == 0:
                    save_checkpoint(output / f"checkpoints/step_{step:09d}.pt", self)
                    save_checkpoint(output / "checkpoints/latest.pt", self)
                    keep = config.training.keep_periodic_checkpoints
                    if keep:
                        for old in sorted((output / "checkpoints").glob("step_*.pt"))[:-keep]:
                            old.unlink()
                if episode_limit:
                    break
            save_checkpoint(output / "checkpoints/latest.pt", self)
        except KeyboardInterrupt:
            # An interrupt can land inside an optimizer step. Do not serialize a
            # potentially half-applied transition; keep the last completed checkpoint.
            interrupted = True
            print("Stopped; the last complete checkpoint is retained.", flush=True)
        finally:
            logger.close()
            self.env.close()
        p = self.progress
        summary = {
            "environment_steps": p["step"],
            "completed_episodes": p["completed"],
            "updates": self.agent.updates,
            "first_update_step": p["first_update_step"],
            "target_syncs": p["target_syncs"],
            "replay_size": len(self.replay),
            "epsilon_next": epsilon_at_step(p["step"], config.dqn),
            "last_update": p["last_metrics"],
            "online_weights_changed": any(
                not torch.equal(a, b) for a, b in zip(initial, self.agent.online.parameters())
            ),
            "partial_episode_steps": p["episode_steps"],
            "partial_episode_reward": p["episode_reward"],
            "best_validation_score": None if best_score == -float("inf") else best_score,
            "interrupted": interrupted,
            "elapsed_seconds": time.perf_counter() - started,
        }
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        return summary
