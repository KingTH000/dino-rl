"""Record a short greedy simulator rollout as a GIF for inspection or a portfolio."""

import argparse
from pathlib import Path

import torch
from PIL import Image

from dino_rl.config import config_from_dict
from dino_rl.preprocessing.state_normalization import StateNormalizer
from dino_rl.training.factory import make_agent, make_env
from dino_rl.utils.checkpoint import load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20000)
    parser.add_argument("--seconds", type=int, default=12)
    args = parser.parse_args()
    if not 1 <= args.seconds <= 30 or args.seed < 0:
        parser.error("--seconds must be 1–30 and --seed must be nonnegative")
    if args.output.exists() or args.output.suffix.lower() != ".gif":
        parser.error("--output must be a new .gif file")
    data = load_checkpoint(args.checkpoint)
    data.pop("replay", None)
    data.pop("optimizer", None)
    config = config_from_dict(data["config"])
    torch.set_num_threads(config.training.torch_threads)
    agent = make_agent(config, torch.device("cpu"))
    agent.online.load_state_dict(data["online"])
    normalize = StateNormalizer(config.simulator)
    env = make_env(config, "rgb_array")
    frames = []
    stride = max(1, round(config.simulator.fps / 20))
    try:
        obs, _ = env.reset(seed=args.seed)
        for step in range(config.simulator.fps * args.seconds):
            inputs = normalize(obs) if config.observation == "state" else obs
            obs, _, terminated, truncated, info = env.step(agent.act(inputs))
            if step % stride == 0 or terminated or truncated:
                frames.append(
                    Image.fromarray(env.render()).convert("P", palette=Image.Palette.ADAPTIVE)
                )
            if terminated or truncated:
                break
    finally:
        env.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        args.output,
        save_all=True,
        append_images=frames[1:],
        duration=round(1000 * stride / config.simulator.fps),
        loop=0,
    )
    print(f"Saved {len(frames)} frames from seed {args.seed}; final state: {info}")


if __name__ == "__main__":
    main()
