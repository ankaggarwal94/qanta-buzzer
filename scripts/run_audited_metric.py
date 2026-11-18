"""Run a frozen paper metric on new inputs after checking its saved splits."""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qb_data.data_audit import require_metric_splits

_METRICS = ("compute_csli", "compute_prefix_calibration", "compute_stopdff", "compute_stopdff_dp")


def main(argv: list[str] | None = None) -> int:
    """Parse the producer's options, audit the selected partitions, then forward the invocation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metric", choices=_METRICS)
    parser.add_argument("options", nargs=argparse.REMAINDER, help="arguments for the selected producer")
    command = parser.parse_args(argv)
    producer = importlib.import_module(f"scripts.{command.metric}")
    options = producer._parse_args(command.options)
    directory = Path(options.data_dir)
    selected = {"test": getattr(options, "split", "test")}
    if hasattr(options, "fit_split"):
        selected["val"] = options.fit_split
    if command.metric == "compute_csli":
        selected["train"] = "train"
    try:
        require_metric_splits(
            directory, **{role: directory / f"{split}_dataset.json" for role, split in selected.items()},
        )
    except (OSError, ValueError) as exc:
        print(f"Dataset preflight failed: {exc}", file=sys.stderr)
        return 1
    return producer.main(command.options)


if __name__ == "__main__":
    raise SystemExit(main())
