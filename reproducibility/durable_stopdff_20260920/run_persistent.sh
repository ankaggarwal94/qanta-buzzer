#!/usr/bin/env bash
set -euo pipefail

# Invoke inside a durable scheduler/container/job. This wrapper does not daemonize
# itself or guarantee that an interactive shell survives disconnection.
: "${STOPDFF_PYTHON:?Set STOPDFF_PYTHON to the absolute Python 3.11.12 interpreter}"
control_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
exec "$STOPDFF_PYTHON" -u "$control_dir/run_durable_rerun.py" "$@"
