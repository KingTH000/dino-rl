# Milestone 2: state-based DQN

Validated 2026-09-23 using the local Python 3.10 virtual environment and CPU.
Scope: a functioning learning loop, not a gameplay-performance claim.

## What changed

- Fixed state scaling based on simulator geometry and configured physics.
- A bounded NumPy replay buffer with owned copies, uniform sampling, and its own seeded RNG.
- A 10 → 128 → 128 → 3 MLP with ReLU hidden activations.
- Manually implemented vanilla DQN: online and frozen target networks, epsilon-greedy
  actions, discounted Bellman targets, Huber loss and Adam.
- Configurable warm-up, batch size, exploration schedule, update interval and hard
  target synchronization, with the requested initial hyperparameters.
- A training CLI that saves resolved settings, CSV diagnostics and a run summary.

Reward behavior and simulator physics remain those of Milestone 1. The replay flag
is explicitly called `terminated`: crashes disable bootstrapping; time limits do
not. A time-limit transition stores the final state before the environment reset.

## Verification

| Check | Result |
|---|---|
| `python -m pytest -q` | 51 passed |
| `ruff check dino_rl scripts tests` | Passed |
| `ruff format --check dino_rl scripts tests` | Passed |
| CPU seeded training reproducibility | Identical weights and metrics in two short test runs, excluding elapsed time |
| Full CLI smoke run | 12,000 environment steps, no runtime errors |
| Saved metrics | Finite logged loss/Q/target values; episode and partial-episode steps sum to total steps |

The two existing Gymnasium warnings about unbounded **raw** observation-space
limits remain. The neural network receives separately scaled observations.

Tests check replay wraparound and ownership, sampling reproducibility, invalid
inputs/configuration, normalization, network output shapes, exploration decay,
greedy and random action selection, exact Bellman arithmetic, exact Huber loss,
online weight updates, target freezing/synchronization, warm-up, training frequency,
time-limit transitions, partial episodes, episode budgets and output protection.
The Bellman test deliberately gives online and target networks different best
actions to verify vanilla DQN uses the maximum from the target network.

## Executed training check

```bash
python scripts/train.py --algorithm dqn --observation state \
  --steps 12000 --seed 42 --device cpu \
  --output-dir runs/milestone2_smoke_20260923
```

Choose a different output directory to repeat; existing directories are rejected.
Unchanged default DQN hyperparameters were used. Artifacts are preserved in
[milestone2-smoke](milestone2-smoke/): resolved configuration, summary, episode CSV
and update CSV. These diagnostic copies are tracked candidates, while `runs/`
remains ignored for ordinary experiments.

| Measurement | Actual result |
|---|---:|
| Environment steps | 12,000 |
| Completed episodes | 87 |
| Obstacles cleared in completed exploratory episodes | 0 |
| First learning update | Step 5,000 |
| Gradient updates | 7,001 |
| Target-network synchronizations, excluding initialization | 2 |
| Stored replay transitions | 12,000 |
| Epsilon for next action | 0.943 |
| Final sampled-batch Huber loss | 0.0001724444 |
| Online weights changed | Yes |
| Unfinished episode steps at budget exhaustion | 103 |

A small batch loss means the network fitted sampled value targets; it does **not**
prove good gameplay. This run was mostly exploratory, used only one seed, and did
not perform a greedy evaluation or random-policy comparison. Results in the README's
model-performance table remain TBD. Recorded elapsed time excludes imports/setup
and is not a cross-device benchmark. Actual MPS/CUDA training was not exercised.

## What remains for Milestone 3

Add checkpoint save/load/resume, TensorBoard, and a separate repeated-episode
evaluation procedure. Then train long enough and compare greedy performance with
random actions using consistent evaluation settings. Vision, hybrid, Double DQN
and real Chrome integration remain later milestones.

**Milestone 2 does not save model weights.** Only metrics/configuration survive a
CLI exit. Python callers can inspect the trained network via `Trainer.agent` during
the process. This intentionally avoids introducing the Milestone 3 persistence
system before its dedicated implementation and tests.
