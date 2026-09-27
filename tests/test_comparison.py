from copy import deepcopy
from dataclasses import asdict

import pytest

from dino_rl.config import Config
from scripts.compare_experiments import comparison_rows


def report():
    return {
        "config": asdict(Config()),
        "training_step": 25000,
        "agent": {
            "episodes": [{"seed": 20000}, {"seed": 20001}],
            "summary": {"mean_score": 70.0},
        },
    }


def test_algorithm_comparison_preserves_labels_and_requires_same_model():
    vanilla, double = report(), report()
    double["config"]["algorithm"] = "double_dqn"
    rows = comparison_rows([vanilla, double], ["DQN", "Double DQN"], True, True)
    assert [row["algorithm"] for row in rows] == ["dqn", "double_dqn"]
    double["config"]["dqn"]["hidden_size"] = 64
    with pytest.raises(ValueError, match="same observation and model"):
        comparison_rows([vanilla, double], ["DQN", "Double DQN"], True, True)
    double = report()
    double["config"]["observation"] = "hybrid"
    with pytest.raises(ValueError, match="same observation and model"):
        comparison_rows([vanilla, double], ["DQN", "Double DQN"], True, True)


def test_comparison_matches_protocol_but_exposes_different_budgets():
    state = report()
    image = deepcopy(state)
    image["config"]["observation"] = "image"
    image["config"]["dqn"]["batch_size"] = 16
    image["config"]["dqn"]["train_frequency"] = 8
    rows = comparison_rows([state, image], ["State", "Image"])
    assert rows[1]["observation"] == "image"
    assert rows[1]["batch_size"] == 16 and rows[1]["train_frequency"] == 8
    assert rows[0]["selected_step"] == 25000


@pytest.mark.parametrize("field", ["seed", "steps", "epsilon_decay_steps", "crop"])
def test_matched_training_rejects_changed_protocol(field):
    state, hybrid = report(), report()
    hybrid["config"]["observation"] = "hybrid"
    assert len(comparison_rows([state, hybrid], ["State", "Hybrid"], True)) == 2
    if field == "seed":
        hybrid["config"]["seed"] = 43
    elif field == "steps":
        hybrid["config"]["training"]["steps"] = 100
    elif field == "crop":
        hybrid["config"]["image"]["crop"] = None
    else:
        hybrid["config"]["dqn"]["epsilon_decay_steps"] = 100
    with pytest.raises(ValueError, match="requires matching"):
        comparison_rows([state, hybrid], ["State", "Hybrid"], True)


@pytest.mark.parametrize("difference", ["simulator", "rewards", "seeds"])
def test_comparison_rejects_unmatched_evaluation(difference):
    state, image = report(), report()
    if difference == "seeds":
        image["agent"]["episodes"][0]["seed"] = 10000
    elif difference == "simulator":
        image["config"]["simulator"]["initial_speed"] = 200
    else:
        image["config"]["rewards"]["crash"] = -5
    with pytest.raises(ValueError):
        comparison_rows([state, image], ["State", "Image"])
