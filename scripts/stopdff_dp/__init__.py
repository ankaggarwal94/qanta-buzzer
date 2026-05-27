"""Finite-horizon dynamic-programming StopDFF helpers.

Imported by ``scripts/compute_stopdff_dp.py``. Kept in a sibling
package so unit tests can target individual modules without paying
the CLI/argparse import cost of the producer script.
"""

from __future__ import annotations

__all__ = [
    "adapter",
    "continuation",
    "diagnostics",
    "dp_solver",
    "rewards",
    "writers",
    "types",
]
