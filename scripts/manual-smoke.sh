#!/usr/bin/env bash
# Manual smoke pipeline -- runs the four-stage belief-feature smoke workflow.
# Intended for human verification, not CI (stages are heavyweight ML runs).
#
# Prereqs: repo-local .venv with `pip install -e .` (see AGENTS.md for setup)
# Outputs: artifacts/smoke/
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="$REPO_ROOT/.venv/bin/python"

if [ ! -x "$PYTHON" ]; then
    echo "ERROR: Expected repo-local Python at $PYTHON." >&2
    echo "Run: python3.11 -m venv .venv && .venv/bin/pip install -U pip && .venv/bin/pip install -e ." >&2
    exit 1
fi

echo "=== Stage 1/4: Build MC dataset ==="
"$PYTHON" scripts/build_mc_dataset.py --smoke

echo "=== Stage 2/4: Run baselines ==="
"$PYTHON" scripts/run_baselines.py --smoke

echo "=== Stage 3/4: Train PPO ==="
"$PYTHON" scripts/train_ppo.py --smoke

echo "=== Stage 4/4: Evaluate all ==="
"$PYTHON" scripts/evaluate_all.py --smoke

echo "=== Smoke pipeline complete. Check artifacts/smoke/ ==="
