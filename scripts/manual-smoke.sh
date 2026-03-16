#!/usr/bin/env bash
# Manual smoke pipeline -- runs the four-stage belief-feature smoke workflow.
# Intended for human verification, not CI (stages are heavyweight ML runs).
#
# Prereqs: pip install -e .  (see AGENTS.md for full setup)
# Outputs: artifacts/smoke/
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
if [ -f "$REPO_ROOT/.venv/bin/activate" ]; then
    source "$REPO_ROOT/.venv/bin/activate"
fi

PYTHON="${PYTHON:-python3}"

echo "=== Stage 1/4: Build MC dataset ==="
$PYTHON scripts/build_mc_dataset.py --smoke

echo "=== Stage 2/4: Run baselines ==="
$PYTHON scripts/run_baselines.py --smoke

echo "=== Stage 3/4: Train PPO ==="
$PYTHON scripts/train_ppo.py --smoke

echo "=== Stage 4/4: Evaluate all ==="
$PYTHON scripts/evaluate_all.py --smoke

echo "=== Smoke pipeline complete. Check artifacts/smoke/ ==="
