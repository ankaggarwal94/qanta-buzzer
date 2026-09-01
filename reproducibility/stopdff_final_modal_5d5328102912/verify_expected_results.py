#!/usr/bin/env python3
"""Independently reduce a completed StopDFF v5 package and compare paper claims."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


EXPECTED = {
    "n_cells": 96,
    "n_items": 3037,
    "family_M": 0.0,
    "family_ci": [0.0, 0.0],
    "family_histogram": {0.0: 984, 1.0: 16},
    "zero_fraction": {
        "min": 0.5205795192624301,
        "median": 0.7332894303589069,
        "max": 0.9996707276918011,
    },
    "signed_mean": {
        "min": -0.6766545933486993,
        "max": 0.1096476786302272,
    },
    "absolute_mean": {
        "min": 0.000329272308199,
        "max": 1.2169904511030622,
    },
    "interval_counts": {"negative": 50, "cross": 38, "positive": 8},
    "calibrator_interval_counts": {
        "isotonic": {"negative": 31, "cross": 1, "positive": 0},
        "platt-logistic": {"negative": 17, "cross": 7, "positive": 8},
        "similarity-temperature": {"negative": 2, "cross": 30, "positive": 0},
    },
    "representative_key": (
        "reward_schedule=acf_flat__continuation=empirical_bucket"
        "__calibrator=isotonic__prefix_bucketing=early_mid_late"
        "__category_pooling=per_category"
    ),
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return value


def close(actual: float, expected: float, tol: float = 1e-9) -> bool:
    return math.isclose(actual, expected, rel_tol=0.0, abs_tol=tol)


def classify_interval(interval: list[float]) -> str:
    lo, hi = map(float, interval)
    if hi < 0.0:
        return "negative"
    if lo > 0.0:
        return "positive"
    return "cross"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path, help="path containing aggregate.json and cells/")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    aggregate = load_json(run_dir / "aggregate.json")
    cell_paths = sorted((run_dir / "cells").glob("*.json"))
    errors: list[str] = []

    if len(cell_paths) != EXPECTED["n_cells"]:
        errors.append(f"expected 96 cell files, found {len(cell_paths)}")

    cells = [load_json(path) for path in cell_paths]
    completed = [cell for cell in cells if cell.get("status") == "completed"]
    if len(completed) != EXPECTED["n_cells"]:
        errors.append(f"expected 96 completed cells, found {len(completed)}")

    zero_fractions: list[float] = []
    signed_means: list[float] = []
    absolute_means: list[float] = []
    all_interval_counts: Counter[str] = Counter()
    by_calibrator_counts: dict[str, Counter[str]] = defaultdict(Counter)
    by_calibrator_signed: dict[str, list[float]] = defaultdict(list)
    by_calibrator_absolute: dict[str, list[float]] = defaultdict(list)
    replicate_vectors: list[list[float]] = []
    item_sets: list[set[str]] = []
    representative: dict[str, Any] | None = None

    for cell in completed:
        key = str(cell.get("cell_key"))
        shifts_obj = cell.get("index_shift_by_item")
        if not isinstance(shifts_obj, dict):
            errors.append(f"{key}: missing index_shift_by_item")
            continue

        shifts = [float(value) for value in shifts_obj.values()]
        item_sets.append(set(map(str, shifts_obj)))
        if len(shifts) != EXPECTED["n_items"]:
            errors.append(f"{key}: expected 3037 items, found {len(shifts)}")
            continue

        signed_mean = sum(shifts) / len(shifts)
        absolute_mean = sum(abs(value) for value in shifts) / len(shifts)
        zero_fraction = sum(value == 0.0 for value in shifts) / len(shifts)

        signed_means.append(signed_mean)
        absolute_means.append(absolute_mean)
        zero_fractions.append(zero_fraction)

        bootstrap = cell.get("bootstrap", {})
        point = bootstrap.get("point", {})
        intervals = bootstrap.get("ci", {})
        signed_interval = intervals.get("signed_index_mean")
        if not isinstance(signed_interval, list) or len(signed_interval) != 2:
            errors.append(f"{key}: missing signed mean interval")
            continue

        interval_class = classify_interval(signed_interval)
        all_interval_counts[interval_class] += 1
        calibrator = str(cell.get("cell", {}).get("calibrator"))
        by_calibrator_counts[calibrator][interval_class] += 1
        by_calibrator_signed[calibrator].append(signed_mean)
        by_calibrator_absolute[calibrator].append(absolute_mean)

        reps = bootstrap.get("abs_median_replicates")
        if not isinstance(reps, list) or len(reps) != 1000:
            errors.append(f"{key}: expected 1000 absolute-median replicates")
        else:
            replicate_vectors.append([float(value) for value in reps])

        if key == EXPECTED["representative_key"]:
            representative = cell

        for field, recomputed in (
            ("signed_index_mean", signed_mean),
            ("absolute_index_mean", absolute_mean),
            ("signed_index_median", float(statistics.median(shifts))),
            ("absolute_index_median", float(statistics.median(abs(x) for x in shifts))),
        ):
            reported = point.get(field)
            if not isinstance(reported, (int, float)) or not close(float(reported), recomputed, 1e-12):
                errors.append(f"{key}: {field} does not match item-level reduction")

    if item_sets and any(items != item_sets[0] for items in item_sets[1:]):
        errors.append("the 96 cells do not share one identical item set")

    def check_triplet(label: str, values: list[float], expected: dict[str, float]) -> None:
        if not values:
            errors.append(f"{label}: no values")
            return
        actual = {
            "min": min(values),
            "median": float(statistics.median(values)),
            "max": max(values),
        }
        for name, target in expected.items():
            if name in actual and not close(actual[name], target, 1e-9):
                errors.append(f"{label}.{name}: {actual[name]} != {target}")

    check_triplet("zero_fraction", zero_fractions, EXPECTED["zero_fraction"])
    check_triplet("signed_mean", signed_means, EXPECTED["signed_mean"])
    check_triplet("absolute_mean", absolute_means, EXPECTED["absolute_mean"])

    if dict(all_interval_counts) != EXPECTED["interval_counts"]:
        errors.append(
            f"pointwise signed-mean interval counts: "
            f"{dict(all_interval_counts)} != {EXPECTED['interval_counts']}"
        )
    for calibrator, target in EXPECTED["calibrator_interval_counts"].items():
        actual = dict(by_calibrator_counts[calibrator])
        if actual != target:
            errors.append(f"{calibrator} interval counts: {actual} != {target}")

    family_reps: list[float] = []
    if len(replicate_vectors) == EXPECTED["n_cells"]:
        family_reps = [
            max(vector[index] for vector in replicate_vectors)
            for index in range(1000)
        ]
        histogram = dict(Counter(family_reps))
        if histogram != EXPECTED["family_histogram"]:
            errors.append(f"family replicate histogram: {histogram} != {EXPECTED['family_histogram']}")

    family = aggregate.get("family", {})
    if not close(float(family.get("M", math.nan)), EXPECTED["family_M"]):
        errors.append(f"family M: {family.get('M')} != 0")
    if list(family.get("ci", [])) != EXPECTED["family_ci"]:
        errors.append(f"family CI: {family.get('ci')} != [0, 0]")
    if aggregate.get("requested") != 96 or aggregate.get("completed") != 96:
        errors.append("aggregate requested/completed counts are not 96/96")
    if aggregate.get("failed") != 0 or aggregate.get("skipped") != 0:
        errors.append("aggregate contains failed or skipped cells")
    if aggregate.get("release_status") != "VALID":
        errors.append(f"release_status is {aggregate.get('release_status')!r}, not VALID")

    if representative is None:
        errors.append("representative cell not found")
    else:
        desc = representative.get("descriptive", {})
        if desc.get("n_paired_items") != 3037:
            errors.append("representative cell n_paired_items != 3037")
        if desc.get("never_buzz_mc") != 178:
            errors.append("representative cell never_buzz_mc != 178")
        if desc.get("never_buzz_qa") != 723:
            errors.append("representative cell never_buzz_qa != 723")
        bootstrap = representative.get("bootstrap", {})
        point = bootstrap.get("point", {})
        intervals = bootstrap.get("ci", {})
        representative_expected = {
            "signed_index_mean": -0.2617714850,
            "absolute_index_mean": 0.3131379651,
            "signed_index_median": 0.0,
            "absolute_index_median": 0.0,
        }
        for field, target in representative_expected.items():
            actual = point.get(field)
            if not isinstance(actual, (int, float)) or not close(float(actual), target, 1e-9):
                errors.append(f"representative cell {field}: {actual} != {target}")
        representative_intervals = {
            "signed_index_mean": [-0.2831741851, -0.2383766875],
            "absolute_index_mean": [0.2904181758, 0.3358742180],
            "signed_index_median": [0.0, 0.0],
            "absolute_index_median": [0.0, 0.0],
        }
        for field, target in representative_intervals.items():
            actual = intervals.get(field)
            if (
                not isinstance(actual, list)
                or len(actual) != 2
                or any(
                    not close(float(observed), expected, 1e-9)
                    for observed, expected in zip(actual, target, strict=True)
                )
            ):
                errors.append(f"representative cell {field} interval: {actual} != {target}")

    summary = {
        "cells": len(completed),
        "items_per_cell": len(item_sets[0]) if item_sets else 0,
        "family": {
            "M": family.get("M"),
            "ci": family.get("ci"),
            "replicate_histogram": dict(Counter(family_reps)) if family_reps else {},
        },
        "zero_fraction": {
            "min": min(zero_fractions) if zero_fractions else None,
            "median": statistics.median(zero_fractions) if zero_fractions else None,
            "max": max(zero_fractions) if zero_fractions else None,
        },
        "signed_mean_range": [
            min(signed_means) if signed_means else None,
            max(signed_means) if signed_means else None,
        ],
        "absolute_mean_range": [
            min(absolute_means) if absolute_means else None,
            max(absolute_means) if absolute_means else None,
        ],
        "interval_counts": dict(all_interval_counts),
        "interval_counts_by_calibrator": {
            key: dict(value) for key, value in sorted(by_calibrator_counts.items())
        },
        "average_signed_mean_by_calibrator": {
            key: sum(value) / len(value)
            for key, value in sorted(by_calibrator_signed.items())
        },
        "average_absolute_mean_by_calibrator": {
            key: sum(value) / len(value)
            for key, value in sorted(by_calibrator_absolute.items())
        },
        "errors": errors,
        "status": "PASS" if not errors else "FAIL",
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
