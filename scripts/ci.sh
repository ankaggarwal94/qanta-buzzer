#!/usr/bin/env bash
# CI entry point -- runs the full pytest suite from the repo-local venv.
# Exit nonzero on any failure so CI gates catch regressions.
#
# Usage:
#   bash scripts/ci.sh              # full suite
#   bash scripts/ci.sh -k "not t5"  # skip T5-dependent tests
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTEST="$REPO_ROOT/.venv/bin/pytest"

if [ ! -x "$PYTEST" ]; then
    echo "ERROR: Expected repo-local pytest at $PYTEST." >&2
    echo "Run: python3.11 -m venv .venv && .venv/bin/pip install -U pip && .venv/bin/pip install -e \".[dev]\"" >&2
    exit 1
fi

"$PYTEST" tests/ "$@"
