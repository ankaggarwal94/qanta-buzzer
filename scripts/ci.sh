#!/usr/bin/env bash
# CI entry point -- runs the full pytest suite.
# Exit nonzero on any failure so CI gates catch regressions.
set -euo pipefail
pytest "$@"
