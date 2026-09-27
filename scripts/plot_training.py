"""Export a static training overview from CSV diagnostics and optional held-out results."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import EngFormatter, MaxNLocator


def read_rows(path: Path) -> list[dict]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--comparison", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    episodes = read_rows(args.run / "episodes.csv")
    updates = read_rows(args.run / "updates.csv")
    evaluations = read_rows(args.run / "evaluations.csv")
    figure, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)

    def plot(ax, rows, key, title, rolling=False):
        x = np.array([float(r["environment_step"]) for r in rows])
        y = np.array([float(r[key]) for r in rows])
        if len(y):
            ax.plot(x, y, alpha=0.3 if rolling else 1, linewidth=1)
            if rolling:
                window = min(50, len(y))
                ax.plot(
                    x[window - 1 :],
                    np.convolve(y, np.ones(window) / window, mode="valid"),
                    label=f"{window}-episode mean",
                )
                ax.legend()
        ax.set(title=title, xlabel="Environment steps")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        ax.xaxis.set_major_formatter(EngFormatter(sep=""))
        ax.grid(alpha=0.2)

    plot(axes[0, 0], episodes, "reward", "Training reward (exploration enabled)", True)
    plot(axes[0, 1], evaluations, "mean_score", "Greedy validation mean score")
    plot(axes[0, 2], updates, "loss", "Sampled training Huber loss")
    plot(axes[1, 0], evaluations, "mean_survival_time", "Greedy validation survival (seconds)")
    plot(axes[1, 1], evaluations, "mean_obstacles_cleared", "Greedy validation obstacles cleared")
    if args.comparison:
        results = json.loads(args.comparison.read_text())
        groups = [name for name in ("random", "agent") if name in results]
        axes[1, 2].boxplot(
            [[r["score"] for r in results[name]["episodes"]] for name in groups],
            showmeans=True,
        )
        axes[1, 2].set_xticks(range(1, len(groups) + 1), groups)
        axes[1, 2].set(title="Held-out evaluation score distribution", ylabel="Score")
    else:
        plot(axes[1, 2], updates, "epsilon", "Exploration probability")
    metadata = json.loads((args.run / "config.json").read_text())
    algorithm = metadata.get("algorithm", "dqn").replace("_", " ").upper()
    figure.suptitle(f"{metadata['observation'].title()} {algorithm} — {args.run.name}", fontsize=16)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=150)
    plt.close(figure)


if __name__ == "__main__":
    main()
