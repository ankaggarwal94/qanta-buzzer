#!/usr/bin/env bash
# CI entry point -- runs the full pytest suite from the project venv.
# Exit nonzero on any failure so CI gates catch regressions.
#
# Usage:
#   bash scripts/ci.sh              # full suite
#   bash scripts/ci.sh -k "not t5"  # skip T5-dependent tests
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

if [ -f "$REPO_ROOT/.venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$REPO_ROOT/.venv/bin/activate"
elif ! command -v pytest &>/dev/null; then
    echo "ERROR: No .venv found and pytest not on PATH." >&2
    echo "Run: python3 -m venv .venv && source .venv/bin/activate && pip install -e ." >&2
    exit 1
fi

pytest tests/ "$@"
