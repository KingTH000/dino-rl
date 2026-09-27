"""Evaluate a saved DQN greedily and optionally compare a matched random baseline."""

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import torch

from dino_rl.config import config_from_dict
from dino_rl.training.evaluator import evaluate
from dino_rl.training.factory import make_agent
from dino_rl.utils.checkpoint import load_checkpoint
from dino_rl.utils.device import select_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument(
        "--seed",
        type=int,
        default=20000,
        help="First evaluation seed; use seeds not used to select best.pt",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--compare-random", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.episodes <= 0 or args.seed < 0:
        parser.error("--episodes must be positive and --seed nonnegative")
    output = (
        args.output
        or Path("results")
        / f"{args.checkpoint.parent.parent.name}_{args.checkpoint.stem}_seed{args.seed}.json"
    )
    if output.exists():
        parser.error(f"Output already exists: {output}; choose another --output")
    data = load_checkpoint(args.checkpoint)
    # Evaluation needs no replay or optimizer; release these large training tensors.
    data.pop("replay", None)
    data.pop("optimizer", None)
    config = config_from_dict(data["config"])
    torch.set_num_threads(config.training.torch_threads)
    agent = make_agent(config, select_device(args.device))
    agent.online.load_state_dict(data["online"])
    digest = hashlib.sha256()
    with args.checkpoint.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": digest.hexdigest(),
        "training_step": data["progress"]["step"],
        "config": asdict(config),
        "agent": evaluate(config, agent, args.episodes, args.seed, args.render),
    }
    if args.compare_random:
        result["random"] = evaluate(config, None, args.episodes, args.seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                name: report["summary"]
                for name, report in result.items()
                if name in ("agent", "random")
            },
            indent=2,
        )
    )
    print(f"Results: {output}")


if __name__ == "__main__":
    main()
