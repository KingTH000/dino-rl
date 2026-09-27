"""Inspect RGB frames and visual observations without training a vision model."""

import argparse
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle
from PIL import Image

from dino_rl.config import config_from_dict, load_config
from dino_rl.envs import DinoEnv, ImageObservationWrapper
from dino_rl.preprocessing.state_normalization import StateNormalizer
from dino_rl.training.factory import make_agent
from dino_rl.utils.checkpoint import load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs/image_observation.yaml",
    )
    parser.add_argument(
        "--checkpoint", type=Path, help="Optional state-DQN driver; it still acts from numbers"
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--steps", type=int, default=135)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New directory for PNGs, stack.npy and metadata",
    )
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    config = load_config(args.config)
    if args.seed is not None:
        config = replace(config, seed=args.seed)
    torch.set_num_threads(config.training.torch_threads)
    agent = None
    if args.checkpoint:
        data = load_checkpoint(args.checkpoint)
        saved = config_from_dict(data["config"])
        if saved.observation != "state":
            parser.error(
                "This preview's checkpoint driver must be a state policy; use record_gameplay.py for image agents"
            )
        if saved.simulator != config.simulator or saved.rewards != config.rewards:
            parser.error(
                "State-policy checkpoint and preview environment must have matching physics/rewards"
            )
        agent = make_agent(saved, torch.device("cpu"))
        agent.online.load_state_dict(data["online"])
    env = ImageObservationWrapper(
        DinoEnv(config.simulator, config.rewards, "rgb_array"), config.image
    )
    normalize_state = StateNormalizer(config.simulator)
    env.action_space.seed(config.seed)
    episode = 1
    try:
        stack, info = env.reset(seed=config.seed)
        for step in range(args.steps):
            action = (
                env.action_space.sample()
                if agent is None
                else agent.act(normalize_state(env.unwrapped.simulator.observation()), epsilon=0)
            )
            stack, _, terminated, truncated, info = env.step(action)
            if (terminated or truncated) and step < args.steps - 1:
                stack, info = env.reset()
                episode += 1
        rgb = env.render()
        grayscale = env.preprocessor.crop_grayscale(rgb)
        compact = env.frame_stack.as_uint8()
    finally:
        env.close()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    Image.fromarray(rgb).save(args.output_dir / "rgb.png")
    Image.fromarray(grayscale).save(args.output_dir / "crop_grayscale.png")
    Image.fromarray(compact[-1]).save(args.output_dir / "frame.png")
    np.save(args.output_dir / "stack.npy", stack, allow_pickle=False)
    driver = (
        "random actions" if agent is None else f"saved state {saved.algorithm} (numerical inputs)"
    )
    metadata = {
        "config": asdict(config),
        "driver": driver,
        "vision_model_trained": False,
        "steps": args.steps,
        "episode": episode,
        "info": info,
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
        if args.checkpoint
        else None,
        "rgb_shape": list(rgb.shape),
        "crop_shape": list(grayscale.shape),
        "stack_shape": list(stack.shape),
        "dtype": str(stack.dtype),
        "minimum": float(stack.min()),
        "maximum": float(stack.max()),
        "compact_stack_bytes": compact.nbytes,
        "normalized_stack_bytes": stack.nbytes,
        "adjacent_changed_pixels": [
            int(np.count_nonzero(a != b)) for a, b in zip(compact[:-1], compact[1:])
        ],
        "ordering": "oldest_to_newest",
        "frame_interval_seconds": 1 / config.simulator.fps,
        "pillow_version": Image.__version__,
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    figure = plt.figure(figsize=(12, 6), layout="constrained")
    grid = figure.add_gridspec(2, config.image.stack_size, height_ratios=[1, 1.3])
    raw_axis = figure.add_subplot(grid[0, :])
    raw_axis.imshow(rgb)
    if config.image.crop:
        left, top, right, bottom = config.image.crop
        raw_axis.add_patch(
            Rectangle(
                (left, top),
                right - left,
                bottom - top,
                fill=False,
                edgecolor="#e25635",
                linewidth=2,
            )
        )
    raw_axis.set_title(f"Simulator RGB screenshot — orange outline is the crop\nDriver: {driver}")
    raw_axis.axis("off")
    for index, frame in enumerate(stack):
        ax = figure.add_subplot(grid[1, index])
        ax.imshow(frame, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ago = len(stack) - 1 - index
        ax.set_title(f"t − {ago}" if ago else "t (newest)")
        ax.axis("off")
    figure.suptitle(
        f"Visual input: {tuple(stack.shape)} • float32 • range [0, 1]\n"
        f"Crop → grayscale → resize → normalize → {config.image.stack_size}-frame history",
        fontsize=14,
    )
    figure.savefig(args.output_dir / "preview.png", dpi=150)
    plt.close(figure)
    print(json.dumps(metadata, indent=2))
    print(f"Preview: {args.output_dir / 'preview.png'}")


if __name__ == "__main__":
    main()
