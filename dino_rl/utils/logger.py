"""CSV records and account-free live TensorBoard scalar logging."""

import csv
from pathlib import Path

FIELDS = {
    "episodes": [
        "episode",
        "environment_step",
        "reward",
        "score",
        "episode_steps",
        "survival_time",
        "obstacles_cleared",
        "epsilon",
        "terminated",
        "truncated",
    ],
    "updates": [
        "environment_step",
        "update",
        "loss",
        "mean_q",
        "mean_target",
        "epsilon",
        "replay_size",
    ],
    "evaluations": [
        "environment_step",
        "episodes",
        "mean_score",
        "max_score",
        "std_score",
        "mean_reward",
        "mean_survival_time",
        "mean_episode_steps",
        "mean_obstacles_cleared",
        "truncated_episodes",
    ],
}


class RunLogger:
    def __init__(self, output: Path, tensorboard: bool):
        self.handles = {}
        self.writers = {}
        self.tb = None
        try:
            for name, fields in FIELDS.items():
                handle = (output / f"{name}.csv").open("w", newline="")
                self.handles[name] = handle
                self.writers[name] = csv.DictWriter(handle, fieldnames=fields)
                self.writers[name].writeheader()
            if tensorboard:
                from torch.utils.tensorboard import SummaryWriter

                self.tb = SummaryWriter(str(output / "tensorboard"), flush_secs=5)
        except Exception:
            self.close()
            raise

    def record(self, category: str, values: dict) -> None:
        self.writers[category].writerow(values)
        self.handles[category].flush()
        if self.tb:
            for key, value in values.items():
                if key != "environment_step":
                    self.tb.add_scalar(f"{category}/{key}", value, values["environment_step"])

    def close(self) -> None:
        for handle in self.handles.values():
            handle.close()
        if self.tb:
            self.tb.close()
