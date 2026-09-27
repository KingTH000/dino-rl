"""Compare saved evaluations on matched seeds; optionally plot their training logs."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import EngFormatter, MaxNLocator

from dino_rl.config import config_from_dict


def comparison_rows(
    reports: list[dict],
    labels: list[str],
    require_matched_training: bool = False,
    require_same_model: bool = False,
) -> list[dict]:
    """Reject comparisons whose game, rewards or evaluation episode seeds differ."""
    if not reports or len(reports) != len(labels) or len(set(labels)) != len(labels):
        raise ValueError("Provide one unique label per report")
    reference = reports[0]
    seeds = [episode["seed"] for episode in reference["agent"]["episodes"]]
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError("Evaluation requires at least two distinct episode seeds")
    rows = []
    for label, report in zip(labels, reports):
        config = report["config"]
        for field in ("simulator", "rewards"):
            if config[field] != reference["config"][field]:
                raise ValueError(f"Cannot compare different {field} settings")
        if [e["seed"] for e in report["agent"]["episodes"]] != seeds:
            raise ValueError("All reports must use the same ordered evaluation seeds")
        if require_same_model:
            current, baseline = config_from_dict(config), config_from_dict(reference["config"])
            if (
                current.observation != baseline.observation
                or current.image != baseline.image
                or any(
                    getattr(current.dqn, name) != getattr(baseline.dqn, name)
                    for name in ("hidden_size", "cnn_hidden_size", "hybrid_state_hidden_size")
                )
            ):
                raise ValueError("Algorithm comparison requires the same observation and model")
        if require_matched_training:
            current, baseline = config_from_dict(config), config_from_dict(reference["config"])
            for field in ("seed", "image", "training"):
                if getattr(current, field) != getattr(baseline, field):
                    raise ValueError(f"Training comparison requires matching {field}")
            architecture = {"hidden_size", "cnn_hidden_size", "hybrid_state_hidden_size"}
            for field, value in vars(current.dqn).items():
                if field not in architecture and value != getattr(baseline.dqn, field):
                    raise ValueError(f"Training comparison requires matching dqn.{field}")
        rows.append(
            {
                "label": label,
                "observation": config.get("observation", "state"),
                "algorithm": config.get("algorithm", "dqn"),
                "training_seed": config["seed"],
                "selected_step": report["training_step"],
                "planned_training_steps": config["training"]["steps"],
                "batch_size": config["dqn"]["batch_size"],
                "train_frequency": config["dqn"]["train_frequency"],
                **report["agent"]["summary"],
            }
        )
    return rows


def read_csv(path: Path) -> list[dict]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--runs", type=Path, nargs="+")
    parser.add_argument(
        "--require-same-model",
        action="store_true",
        help="Require the same observation and network widths for algorithm comparisons",
    )
    parser.add_argument(
        "--require-matched-training",
        action="store_true",
        help="Require equal training settings and completed step/update budgets",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    reports = [json.loads(path.read_text()) for path in args.reports]
    try:
        rows = comparison_rows(
            reports, args.labels, args.require_matched_training, args.require_same_model
        )
        if args.require_matched_training and not args.runs:
            raise ValueError("--require-matched-training needs --runs to verify completed budgets")
        if args.runs and len(args.runs) != len(reports):
            raise ValueError("Provide one run directory per report")
        if args.runs:
            for run, report in zip(args.runs, reports):
                # Older state reports omitted fields subsequently added to Config.
                run_config = json.loads((run / "config.json").read_text())["config"]
                if config_from_dict(run_config) != config_from_dict(report["config"]):
                    raise ValueError(f"Run configuration does not match report: {run}")
            if args.require_matched_training:
                summaries = [json.loads((run / "summary.json").read_text()) for run in args.runs]
                for summary, report in zip(summaries, reports):
                    if (
                        summary["interrupted"]
                        or summary["environment_steps"] != report["config"]["training"]["steps"]
                        or summary["updates"] != summaries[0]["updates"]
                    ):
                        raise ValueError(
                            "Runs must finish equal environment-step and update budgets"
                        )
        args.output_dir.mkdir(parents=True, exist_ok=False)
    except (ValueError, FileExistsError) as exc:
        parser.error(str(exc))
    with (args.output_dir / "comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "comparison.json").write_text(
        json.dumps(
            {
                "note": (
                    "Matched evaluation and training budgets/settings. "
                    if args.require_matched_training
                    else "Matched evaluation; training budgets/hyperparameters may differ. "
                )
                + "Episode spread is not uncertainty across independent training runs.",
                "sources": [str(p.resolve()) for p in args.reports],
                "same_model_required": args.require_same_model,
                "checkpoint_sha256": [r["checkpoint_sha256"] for r in reports],
                "rows": rows,
            },
            indent=2,
        )
        + "\n"
    )
    figure, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    specs = [
        ("episodes.csv", "reward", "Training reward (50-episode mean)"),
        ("evaluations.csv", "mean_score", "Greedy validation mean score"),
        ("updates.csv", "loss", "Sampled Huber loss"),
        ("evaluations.csv", "mean_survival_time", "Validation survival (seconds)"),
        ("evaluations.csv", "mean_obstacles_cleared", "Validation obstacles cleared"),
    ]
    for ax, (filename, key, title) in zip(axes.flat, specs):
        if args.runs:
            for run, label in zip(args.runs, args.labels):
                records = read_csv(run / filename)
                x = np.array([float(r["environment_step"]) for r in records])
                y = np.array([float(r[key]) for r in records])
                if filename == "episodes.csv" and len(y):
                    window = min(50, len(y))
                    x = x[window - 1 :]
                    y = np.convolve(y, np.ones(window) / window, mode="valid")
                ax.plot(x, y, label=label)
            ax.legend()
        else:
            ax.text(0.5, 0.5, "Pass --runs to plot training", ha="center", transform=ax.transAxes)
        ax.set(title=title, xlabel="Environment steps")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        ax.xaxis.set_major_formatter(EngFormatter(sep=""))
        ax.grid(alpha=0.2)
    axes[1, 2].boxplot(
        [[e["score"] for e in r["agent"]["episodes"]] for r in reports], showmeans=True
    )
    axes[1, 2].set_xticks(range(1, len(reports) + 1), args.labels)
    axes[1, 2].set(title="Matched held-out score distributions", ylabel="Score")
    figure.suptitle(
        "Agent comparison — matched training budgets"
        if args.require_matched_training
        else "Agent comparison — training budgets may differ",
        fontsize=16,
    )
    figure.savefig(args.output_dir / "comparison.png", dpi=150)
    plt.close(figure)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
