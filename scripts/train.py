"""Train or resume DQN/Double DQN with state, image or hybrid observations."""

import argparse
import json
import uuid
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from dino_rl.config import config_from_dict, load_config
from dino_rl.training.trainer import Trainer
from dino_rl.utils.checkpoint import load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        help="Defaults to configs/<observation>_dqn.yaml, or saved config when resuming",
    )
    parser.add_argument("--algorithm", choices=["dqn", "double_dqn"])
    parser.add_argument("--observation", choices=["state", "image", "hybrid"])
    parser.add_argument("--steps", type=int)
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--experiment")
    parser.add_argument(
        "--output-dir", type=Path, help="New directory; existing directories are never overwritten"
    )
    args = parser.parse_args()
    if args.experiment is not None and (
        not args.experiment
        or any(
            c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
            for c in args.experiment
        )
    ):
        parser.error("--experiment may contain only letters, digits, underscores and hyphens")
    try:
        if args.resume and args.config:
            parser.error("--resume uses its saved config; do not also specify --config")
        config = (
            config_from_dict(load_checkpoint(args.resume)["config"])
            if args.resume
            else load_config(
                args.config
                or Path(__file__).resolve().parents[1]
                / f"configs/{args.observation or 'state'}_dqn.yaml"
            )
        )
        if args.observation is not None:
            config = replace(config, observation=args.observation)
        if args.algorithm is not None:
            config = replace(config, algorithm=args.algorithm)
        training = replace(
            config.training,
            steps=config.training.steps if args.steps is None else args.steps,
            episodes=config.training.episodes if args.episodes is None else args.episodes,
        )
        config = replace(
            config,
            training=training,
            device=args.device or config.device,
            seed=config.seed if args.seed is None else args.seed,
        )
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output = (
            args.output_dir
            or Path("runs")
            / (args.experiment or f"{config.observation}_{config.algorithm}")
            / f"{stamp}-{uuid.uuid4().hex[:8]}"
        )
        trainer = Trainer(config, "human" if args.render else None)
        print(f"Device: {trainer.device}; diagnostics: {output}", flush=True)
        summary = trainer.run(output, resume=args.resume)
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, f"Training error: {exc}\n")
    except KeyboardInterrupt:
        parser.exit(130, "Stopped. Resume from the last completed checkpoint.\n")
    print(json.dumps(summary, indent=2))
    print(f"Checkpoints: {output / 'checkpoints'}; TensorBoard logs: {output / 'tensorboard'}")


if __name__ == "__main__":
    main()
