"""Canonical profile: axes, 96-cell grid, smoke cells, FVI representative set.

Mirrors SCIENTIFIC_PROFILE.template.json and the SCIENTIFIC_CONTRACT.md index.
All scientific decimal constants are canonical strings (identity bytes).
"""
from __future__ import annotations

from itertools import product
from typing import Any

from . import PROFILE_NAME, SCHEMA_VERSION
from .rewards import REWARD_ORDER, REWARD_SCHEDULE_STRINGS

FORMAT_CONDITION = "paired_qa_prefix_vs_mc_fixed"

# Ordered axes (order defines FVI representative-cell index parity).
REWARD_SCHEDULES: tuple[str, ...] = REWARD_ORDER
CONTINUATIONS: tuple[str, ...] = ("empirical_bucket", "pooled_empirical")
CALIBRATORS: tuple[str, ...] = ("platt-logistic", "similarity-temperature", "isotonic")
PREFIX_BUCKETINGS: tuple[str, ...] = ("early_mid_late", "exact_prefix")
CATEGORY_POOLINGS: tuple[str, ...] = ("per_category", "pooled_category")

# Cell axis order (canonical key order).
CELL_AXES: tuple[str, ...] = (
    "reward_schedule",
    "continuation",
    "calibrator",
    "prefix_bucketing",
    "category_pooling",
)

# Legacy alias normalization (Task 0.2). Values only; artifacts MUST write canonical.
_CATEGORY_POOLING_ALIASES = {
    "per_subject": "per_category",
    "pooled_subject": "pooled_category",
}
_AXIS_NAME_ALIASES = {"subject_pooling": "category_pooling"}

# Calibration / continuation / bootstrap / gate canonical constant blocks.
CALIBRATION = {
    "fit_rows": "validation_mc_only",
    "application": "shared_map_applied_to_mc_and_qa",
    "phase_boundaries": ["0.33", "0.66"],
    "similarity_temperature_grid": ["0.25", "0.5", "0.75", "1.0", "1.5", "2.0", "3.0", "5.0"],
    "minimum_phase_rows": 10,
    "both_classes_required": True,
}
CONTINUATION = {
    "probability_bin_edges": ["0.0", "0.2", "0.4", "0.6", "0.8", "1.000000000001"],
    "entropy_bin_edges": ["0.0", "0.5", "0.9", "1.000000000001"],
    "minimum_bucket_count": 3,
    "missing_value": "0",
    "clean_fallback_fraction_max": "0.05",
    "clean_missing_fraction_max": "0.01",
}
BOOTSTRAP = {
    "bit_generator": "PCG64",
    "seed": 1,
    "interval": "percentile_95",
    "resampling_unit": "item",
    "common_resamples_across_cells": True,
}
FINAL_REPLICATES = 1000
SMOKE_REPLICATES = 100
GATE = {
    "primary_metric": "median_absolute_prefix_index_shift",
    "material_threshold": "1",
    "family_statistic": "maximum_cell_median_absolute_prefix_index_shift",
    "allow_low_mc_retention": False,
    "allow_incomplete_mc_coverage": False,
}

# FVI candidate grid (SCIENTIFIC_CONTRACT.md 9.7).
FVI_TOLERANCES: tuple[str, ...] = ("1e-6", "1e-8", "1e-10")
FVI_MAX_ITERATIONS: tuple[int, ...] = (50, 100, 200)
FVI_DAMPING = "0.5"
FVI_REQUIRED_CONSECUTIVE = 2
FVI_STRICT_REFERENCE = {"tolerance": "1e-10", "max_iterations": 200}

EXPECTED_CELLS = 96


def normalize_cell(cell: dict[str, Any]) -> dict[str, str]:
    """Return a canonical cell dictionary.

    Parameters
    ----------
    cell
        Cell axes using canonical names or a supported legacy alias.

    Returns
    -------
    dict[str, str]
        The validated cell in canonical axis order.

    Raises
    ------
    ValueError
        If an axis is missing, unknown, duplicated through an alias, or has an
        unsupported value.
    """
    out: dict[str, str] = {}
    remaining = dict(cell)
    # Axis-name aliases (subject_pooling -> category_pooling).
    for alias, canonical in _AXIS_NAME_ALIASES.items():
        if alias in remaining:
            if canonical in remaining:
                raise ValueError(
                    f"cell supplies both alias {alias!r} and canonical axis "
                    f"{canonical!r}"
                )
            remaining[canonical] = remaining.pop(alias)
    unknown = sorted(set(remaining) - set(CELL_AXES))
    if unknown:
        raise ValueError(f"cell has unknown axes {unknown}: {cell}")
    for axis in CELL_AXES:
        if axis not in remaining:
            raise ValueError(f"cell missing axis {axis!r}: {cell}")
        value = str(remaining[axis])
        if axis == "category_pooling":
            value = _CATEGORY_POOLING_ALIASES.get(value, value)
        out[axis] = value
    _validate_cell(out)
    return out


def _validate_cell(cell: dict[str, str]) -> None:
    checks = {
        "reward_schedule": REWARD_SCHEDULES,
        "continuation": CONTINUATIONS,
        "calibrator": CALIBRATORS,
        "prefix_bucketing": PREFIX_BUCKETINGS,
        "category_pooling": CATEGORY_POOLINGS,
    }
    for axis, allowed in checks.items():
        if cell[axis] not in allowed:
            raise ValueError(f"cell axis {axis}={cell[axis]!r} not in {allowed}")


def cell_key_str(cell: dict[str, str]) -> str:
    """Deterministic filesystem-safe cell key (canonical axis order)."""
    c = normalize_cell(cell)
    return "__".join(f"{axis}={c[axis]}" for axis in CELL_AXES)


def full_grid() -> list[dict[str, str]]:
    """The 96 unique cells: 1 x 4 x 2 x 3 x 2 x 2."""
    cells: list[dict[str, str]] = []
    for reward, cont, calib, prefix, catpool in product(
        REWARD_SCHEDULES, CONTINUATIONS, CALIBRATORS, PREFIX_BUCKETINGS, CATEGORY_POOLINGS
    ):
        cells.append(
            {
                "reward_schedule": reward,
                "continuation": cont,
                "calibrator": calib,
                "prefix_bucketing": prefix,
                "category_pooling": catpool,
            }
        )
    # Deterministic order by canonical key.
    cells.sort(key=cell_key_str)
    assert len(cells) == EXPECTED_CELLS, f"expected {EXPECTED_CELLS} cells, got {len(cells)}"
    assert len({cell_key_str(c) for c in cells}) == EXPECTED_CELLS, "duplicate cell keys"
    return cells


# Two explicitly listed smoke cells (NOT a truncation of the 96-cell grid).
SMOKE_CELLS: list[dict[str, str]] = [
    {
        "reward_schedule": "acf_flat",
        "continuation": "empirical_bucket",
        "calibrator": "platt-logistic",
        "prefix_bucketing": "early_mid_late",
        "category_pooling": "per_category",
    },
    {
        "reward_schedule": "strict_wrong",
        "continuation": "pooled_empirical",
        "calibrator": "isotonic",
        "prefix_bucketing": "exact_prefix",
        "category_pooling": "pooled_category",
    },
]


def smoke_cells() -> list[dict[str, str]]:
    """Return the two canonical smoke-test cells.

    Returns
    -------
    list[dict[str, str]]
        Fresh canonical copies of the explicitly registered smoke cells.
    """
    return [normalize_cell(c) for c in SMOKE_CELLS]


def representative_24() -> list[dict[str, str]]:
    """FVI study representative set (SCIENTIFIC_CONTRACT.md 9.7): 24 cells.

    all 4 rewards x both continuations x all 3 calibrators, with:
      prefix_bucketing = early_mid_late when (ri + ci + cali) even else exact_prefix
      category_pooling = per_category  when (ri + 2*ci + cali) even else pooled_category
    """
    cells: list[dict[str, str]] = []
    for ri, reward in enumerate(REWARD_SCHEDULES):
        for ci, cont in enumerate(CONTINUATIONS):
            for cali, calib in enumerate(CALIBRATORS):
                prefix = PREFIX_BUCKETINGS[0] if (ri + ci + cali) % 2 == 0 else PREFIX_BUCKETINGS[1]
                catpool = (
                    CATEGORY_POOLINGS[0] if (ri + 2 * ci + cali) % 2 == 0 else CATEGORY_POOLINGS[1]
                )
                cells.append(
                    normalize_cell(
                        {
                            "reward_schedule": reward,
                            "continuation": cont,
                            "calibrator": calib,
                            "prefix_bucketing": prefix,
                            "category_pooling": catpool,
                        }
                    )
                )
    assert len(cells) == 24
    return cells


def profile_static_identity() -> dict[str, Any]:
    """The backend-neutral scientific-profile identity block (no run-specific IDs)."""
    return {
        "schema_version": SCHEMA_VERSION,
        "profile_name": PROFILE_NAME,
        "splits": {"fit": "val", "eval": "test"},
        "format_condition": [FORMAT_CONDITION],
        "axes": {
            "reward_schedule": list(REWARD_SCHEDULES),
            "continuation": list(CONTINUATIONS),
            "calibrator": list(CALIBRATORS),
            "prefix_bucketing": list(PREFIX_BUCKETINGS),
            "category_pooling": list(CATEGORY_POOLINGS),
        },
        "reward_schedules": {name: dict(REWARD_SCHEDULE_STRINGS[name]) for name in REWARD_SCHEDULES},
        "bellman": {
            "actions": ["answer", "wait", "abstain"],
            "abstain_value": "0",
            "nonterminal_tie_priority": ["wait", "abstain", "answer"],
            "terminal_tie_priority": ["abstain", "answer"],
        },
        "calibration": CALIBRATION,
        "continuation": CONTINUATION,
        "bootstrap": {**BOOTSTRAP},
        "gate": GATE,
        "expected_cells": EXPECTED_CELLS,
    }
