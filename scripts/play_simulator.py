"""Run a seeded random policy; no learning or neural network is involved."""

import argparse
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

from dino_rl.config import load_config
from dino_rl.envs import DinoEnv
from dino_rl.utils.device import select_device
from dino_rl.utils.seeds import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs/environment.yaml",
    )
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--render", action="store_true")
    parser.add_argument(
        "--output", type=Path, help="Save episode records and resolved configuration"
    )
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    config = load_config(args.config)
    config = replace(
        config,
        seed=config.seed if args.seed is None else args.seed,
        device=args.device or config.device,
    )
    seed_everything(config.seed)
    device = select_device(config.device)
    print(f"Selected device: {device} (simulator and random policy run on CPU)")
    env = DinoEnv(config.simulator, config.rewards, "human" if args.render else None)
    env.action_space.seed(config.seed)
    records = []
    started = time.perf_counter()
    try:
        for episode in range(args.episodes):
            env.reset(seed=config.seed if episode == 0 else None)
            total_reward = 0.0
            while True:
                _, reward, terminated, truncated, info = env.step(env.action_space.sample())
                total_reward += reward
                if terminated or truncated:
                    break
            record = {
                "episode": episode + 1,
                "reward": total_reward,
                "terminated": terminated,
                "truncated": truncated,
                **info,
            }
            records.append(record)
            print(json.dumps(record))
    except KeyboardInterrupt:
        print("Stopped by user")
    finally:
        env.close()
    elapsed = time.perf_counter() - started
    print(f"Completed {len(records)} episodes in {elapsed:.3f}s")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {
                    "config": asdict(config),
                    "device": str(device),
                    "elapsed_seconds": elapsed,
                    "episodes": records,
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
