#!/usr/bin/env bash
# Manual smoke pipeline -- runs the four-stage belief-feature smoke workflow.
# Intended for human verification, not CI (stages are heavyweight ML runs).
#
# Prereqs: pip install -e .  (see AGENTS.md for full setup)
# Outputs: artifacts/smoke/
set -euo pipefail

echo "=== Stage 1/4: Build MC dataset ==="
python scripts/build_mc_dataset.py --smoke

echo "=== Stage 2/4: Run baselines ==="
python scripts/run_baselines.py --smoke

echo "=== Stage 3/4: Train PPO ==="
python scripts/train_ppo.py --smoke

echo "=== Stage 4/4: Evaluate all ==="
python scripts/evaluate_all.py --smoke

echo "=== Smoke pipeline complete. Check artifacts/smoke/ ==="
