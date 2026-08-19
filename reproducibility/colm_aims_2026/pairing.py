"""Pair/censoring decomposition: count identities, rates, keys, digests.

Spec rules owned here: R-005..R-011, R-015 (recompute helpers), R-032 (digest
membership of the declared tolerance).
Spec: .correctless/specs/camera-ready-aims-evidence.md
"""
from __future__ import annotations

import hashlib
import unicodedata
from typing import Any

import numpy as np

from .schema import (
    EXCLUSION_REASONS,
    MAX_ADMISSIBLE_TOLERANCE,
    ColmAimsError,
    EmptyEvaluationError,
    SchemaValidationError,
    canonical_estimand_digest,
)

JOINT_CLASSES = (
    "both_finite",
    "mc_finite_ref_timeout",
    "mc_timeout_ref_finite",
    "both_timeout",
)

RATE_KEYS = (
    "rate_both_finite",
    "rate_mc_finite_ref_timeout",
    "rate_mc_timeout_ref_finite",
    "rate_both_timeout",
)

TIMING_STATISTICS = (
    "signed_index_mean",
    "signed_index_median",
    "absolute_index_mean",
    "absolute_index_median",
)


class CountIdentityError(ColmAimsError):
    """A recorded count violates an exact count identity (R-005)."""


class RateError(ColmAimsError):
    """Joint-class rate denominator/sum-to-1/tolerance violation (R-006)."""


class KeySetError(ColmAimsError):
    """Key-set discipline violation: duplicates/overlap/hash mismatch (R-008)."""


class EstimandMismatchError(ColmAimsError):
    """Pooling or comparing cells with differing estimand digests (R-011)."""


def _is_valid_stop(value: Any) -> bool:
    """A well-formed stop is a non-negative zero-indexed integer (R-007)."""
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _is_valid_horizon(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def classify_stop(stop_step: Any, trajectory_horizon: int) -> str:
    """Classify one stop as ``finite`` or ``timeout`` (zero-indexed, R-007).

    ``0 <= stop_step < trajectory_horizon`` is finite;
    ``stop_step >= trajectory_horizon`` is timeout.
    """
    if not _is_valid_horizon(trajectory_horizon):
        raise ColmAimsError(
            f"trajectory_horizon must be a positive integer, got"
            f" {trajectory_horizon!r} (R-007)"
        )
    if not _is_valid_stop(stop_step):
        raise ColmAimsError(
            f"malformed stop_step {stop_step!r}; malformed stops become"
            " exclusions, never imputed stops (R-007)"
        )
    return "finite" if stop_step < trajectory_horizon else "timeout"


def classify_record(record: dict[str, Any]) -> dict[str, Any]:
    """Classify one per-item record into joint class or exclusion (R-007/R-008)."""
    key = record.get("item_key")
    result: dict[str, Any] = {"item_key": key}

    def _excluded(reason: str) -> dict[str, Any]:
        result["status"] = "excluded"
        result["exclusion_reason"] = reason
        result["joint_class"] = None
        return result

    if record.get("excluded") is True:
        declared = record.get("exclusion_reason")
        if isinstance(declared, str) and declared in EXCLUSION_REASONS:
            return _excluded(declared)
        # Missing/undeclared reasons are recorded, never guessed (R-008).
        return _excluded("UNKNOWN_NOT_INFERRED")

    shared = record.get("trajectory_horizon")
    mc_horizon = record.get("mc_trajectory_horizon", shared)
    ref_horizon = record.get("ref_trajectory_horizon", shared)
    if not _is_valid_horizon(mc_horizon) or not _is_valid_horizon(ref_horizon):
        return _excluded("MALFORMED_STOP")
    if mc_horizon != ref_horizon:
        # MC/reference grid mismatches are exclusions with reason (R-007).
        return _excluded("GRID_MISMATCH")

    mc_stop = record.get("mc_stop_step")
    ref_stop = record.get("ref_stop_step")
    if not _is_valid_stop(mc_stop) or not _is_valid_stop(ref_stop):
        return _excluded("MALFORMED_STOP")

    mc_kind = classify_stop(mc_stop, mc_horizon)
    ref_kind = classify_stop(ref_stop, ref_horizon)
    if mc_kind == "finite" and ref_kind == "finite":
        joint = "both_finite"
    elif mc_kind == "finite":
        joint = "mc_finite_ref_timeout"
    elif ref_kind == "finite":
        joint = "mc_timeout_ref_finite"
    else:
        joint = "both_timeout"
    result["status"] = "complete"
    result["joint_class"] = joint
    return result


def _check_duplicate_keys(records: list[dict[str, Any]]) -> None:
    """Duplicate pair keys fail closed; keys compare byte-exact (R-008)."""
    seen: set[str] = set()
    for record in records:
        key = record.get("item_key")
        if not isinstance(key, str) or not key:
            raise KeySetError("record without an opaque item_key (R-008)")
        if key in seen:
            raise KeySetError(
                f"duplicate pair key {key!r} — duplicate pair keys fail"
                " closed (R-008)"
            )
        seen.add(key)


def recompute_counts(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Recompute the per-cell count block from per-item records (R-005)."""
    _check_duplicate_keys(records)
    joint_counts = {joint: 0 for joint in JOINT_CLASSES}
    reason_counts: dict[str, int] = {}
    n_excluded = 0
    for record in records:
        outcome = classify_record(record)
        if outcome["status"] == "excluded":
            n_excluded += 1
            reason = outcome["exclusion_reason"]
            # Exactly one primary reason per excluded unit; secondary
            # diagnostics are not counted (R-008).
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        else:
            joint_counts[outcome["joint_class"]] += 1
    n_complete = sum(joint_counts.values())
    return {
        "n_both_finite": joint_counts["both_finite"],
        "n_mc_finite_ref_timeout": joint_counts["mc_finite_ref_timeout"],
        "n_mc_timeout_ref_finite": joint_counts["mc_timeout_ref_finite"],
        "n_both_timeout": joint_counts["both_timeout"],
        "n_complete": n_complete,
        "n_excluded_or_unpaired": n_excluded,
        "exclusion_reason_counts": reason_counts,
        "n_pairing_population": n_complete + n_excluded,
        "n_mc_timeout": (
            joint_counts["mc_timeout_ref_finite"] + joint_counts["both_timeout"]
        ),
        "n_ref_timeout": (
            joint_counts["mc_finite_ref_timeout"] + joint_counts["both_timeout"]
        ),
    }


def check_count_identities(
    counts: dict[str, Any], records: list[dict[str, Any]]
) -> None:
    """Enforce the five exact count identities against records (R-005)."""
    recomputed = recompute_counts(records)
    mismatches = []
    for field, value in recomputed.items():
        if field not in counts:
            mismatches.append(f"{field} missing from recorded counts")
        elif counts[field] != value:
            mismatches.append(
                f"{field} recorded {counts[field]!r} != recomputed {value!r}"
            )
    extras = sorted(set(counts) - set(recomputed))
    if extras:
        mismatches.append(f"unknown recorded count field(s) {extras}")
    if mismatches:
        raise CountIdentityError(
            "count identity violation against per-item records (R-005): "
            + "; ".join(mismatches)
        )


def compute_rates(counts: dict[str, Any]) -> dict[str, Any]:
    """Joint-class rates over ``n_complete``; ``None`` at zero (R-006)."""
    n_complete = counts["n_complete"]
    if n_complete == 0:
        return {key: None for key in RATE_KEYS}
    return {
        "rate_both_finite": counts["n_both_finite"] / n_complete,
        "rate_mc_finite_ref_timeout": (
            counts["n_mc_finite_ref_timeout"] / n_complete
        ),
        "rate_mc_timeout_ref_finite": (
            counts["n_mc_timeout_ref_finite"] / n_complete
        ),
        "rate_both_timeout": counts["n_both_timeout"] / n_complete,
    }


def _declared_tolerance(cell: dict[str, Any]) -> float:
    estimand = cell.get("estimand") or {}
    tolerance = estimand.get("numerical_tolerance")
    if (
        not isinstance(tolerance, (int, float))
        or isinstance(tolerance, bool)
        or not (0 < float(tolerance) <= MAX_ADMISSIBLE_TOLERANCE)
    ):
        raise SchemaValidationError(
            "cell declares no admissible numerical_tolerance (R-032)"
        )
    return float(tolerance)


def check_rates(cell: dict[str, Any]) -> None:
    """Validate a cell's recorded rates block (R-006)."""
    counts = cell["counts"]
    recorded = cell["rates"]
    tolerance = _declared_tolerance(cell)
    expected = compute_rates(counts)
    missing = sorted(set(RATE_KEYS) - set(recorded))
    if missing:
        raise RateError(f"rates block missing {missing} (R-006)")
    if counts["n_complete"] == 0:
        bad = [key for key in RATE_KEYS if recorded[key] is not None]
        if bad:
            raise RateError(
                f"rates must be null when n_complete is zero; found {bad}"
                " (R-006)"
            )
        return
    for key in RATE_KEYS:
        value = recorded[key]
        if value is None or isinstance(value, bool) or not isinstance(
            value, (int, float)
        ):
            raise RateError(f"rate {key!r} must be numeric (R-006)")
        if abs(float(value) - expected[key]) > tolerance:
            raise RateError(
                f"rate {key!r} recorded {value!r} != {expected[key]!r}"
                f" recomputed over denominator n_complete within declared"
                f" tolerance {tolerance!r} (R-006)"
            )
    total = sum(float(recorded[key]) for key in RATE_KEYS)
    if abs(total - 1.0) > tolerance:
        raise RateError(
            f"joint-class rates sum to {total!r}, not 1 within declared"
            f" tolerance {tolerance!r} (R-006)"
        )


def _both_finite_shifts(records: list[dict[str, Any]]) -> dict[str, int]:
    """Signed per-item shift D_i = mc_stop - ref_stop over both-finite pairs."""
    shifts: dict[str, int] = {}
    for record in records:
        outcome = classify_record(record)
        if outcome["status"] == "complete" and outcome["joint_class"] == (
            "both_finite"
        ):
            shifts[record["item_key"]] = (
                record["mc_stop_step"] - record["ref_stop_step"]
            )
    return shifts


def finite_only_timing_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Finite-only timing summary over exactly n_both_finite items.

    Reproduces the historical paired-summary estimator semantics
    (scripts/stopdff_v5/bootstrap.py::cell_bootstrap_stats point block) on
    the both-finite subset: signed D_i = mc - ref, absolute |D_i|
    (R-006/R-009/R-010).
    """
    _check_duplicate_keys(records)
    shifts = _both_finite_shifts(records)
    ordered = [float(shifts[key]) for key in sorted(shifts)]
    d = np.array(ordered, dtype=np.float64)
    if d.size == 0:
        raise EmptyEvaluationError(
            "finite-only timing summary over zero both-finite pairs refused"
            " (R-006)"
        )
    absd = np.abs(d)
    return {
        "conditional_on": "n_both_finite",
        "estimand": "signed_index_shift_mc_minus_ref",
        "n": int(d.size),
        "signed_index_mean": float(np.mean(d)),
        "signed_index_median": float(np.median(d)),
        "absolute_index_mean": float(np.mean(absd)),
        "absolute_index_median": float(np.median(absd)),
    }


def _sentinel_coded_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Retained sentinel-coded historical summary: timeouts coded as the
    horizon value; complete pairs only; separately named, never pooled
    (R-006)."""
    shifts: list[float] = []
    for record in records:
        outcome = classify_record(record)
        if outcome["status"] != "complete":
            continue
        shifts.append(float(record["mc_stop_step"] - record["ref_stop_step"]))
    d = np.array(sorted(shifts), dtype=np.float64)
    return {
        "convention": "timeout_coded_as_horizon",
        "n": int(d.size),
        "signed_index_mean": float(np.mean(d)) if d.size else None,
        "signed_index_median": float(np.median(d)) if d.size else None,
    }


def recompute_interval(
    records: list[dict[str, Any]], interval_spec: dict[str, Any]
) -> dict[str, Any]:
    """Deterministically re-run the recorded interval procedure (R-015)."""
    for key in ("procedure", "draw_count", "resampling_seeds", "statistic"):
        if key not in interval_spec:
            raise SchemaValidationError(
                f"interval spec missing recorded identity field {key!r} —"
                " missing interval identity leaves the interval"
                " non-certifying (R-015)"
            )
    procedure = interval_spec["procedure"]
    if procedure != "percentile_bootstrap":
        raise SchemaValidationError(
            f"unknown interval procedure {procedure!r} (R-015)"
        )
    statistic = interval_spec["statistic"]
    if statistic not in TIMING_STATISTICS:
        raise SchemaValidationError(
            f"unknown interval statistic {statistic!r} (R-015)"
        )
    draw_count = interval_spec["draw_count"]
    if not isinstance(draw_count, int) or isinstance(draw_count, bool) or (
        draw_count <= 0
    ):
        raise SchemaValidationError(
            f"interval draw_count must be a positive integer, got"
            f" {draw_count!r} (R-015)"
        )
    seeds = interval_spec["resampling_seeds"]
    if not isinstance(seeds, list) or not seeds:
        raise SchemaValidationError(
            "interval resampling_seeds must be a non-empty list (R-015)"
        )

    shifts = _both_finite_shifts(records)
    if not shifts:
        raise EmptyEvaluationError(
            "interval recomputation over zero both-finite pairs refused"
            " (R-015)"
        )
    # The pinned historical estimator (same plan, same resample indices).
    from scripts.stopdff_v5 import bootstrap as historical_bootstrap

    plan = historical_bootstrap.build_bootstrap_plan(
        sorted(shifts), replicates=draw_count, seed=seeds[0]
    )
    stats = historical_bootstrap.cell_bootstrap_stats(
        {key: shifts[key] for key in shifts}, plan
    )
    return {
        "procedure": procedure,
        "draw_count": draw_count,
        "resampling_seeds": list(seeds),
        "statistic": statistic,
        "ci": list(stats["ci"][statistic]),
    }


def check_key_sets(cell: dict[str, Any], records: list[dict[str, Any]]) -> None:
    """Disjointness/union/hash/duplicate discipline for pair key sets (R-008)."""
    complete = list(cell["complete_pair_keys"])
    excluded = list(cell["excluded_keys"])
    if len(set(complete)) != len(complete):
        raise KeySetError("complete_pair_keys contains duplicates (R-008)")
    if len(set(excluded)) != len(excluded):
        raise KeySetError("excluded_keys contains duplicates (R-008)")
    overlap = sorted(set(complete) & set(excluded))
    if overlap:
        raise KeySetError(
            f"complete-pair and excluded/unpaired key sets overlap on"
            f" {overlap} — the sets must be disjoint (R-008)"
        )
    union = sorted(set(complete) | set(excluded))
    declared_hash = cell["pairing_population_keyset_sha256"]
    union_hash = hashlib.sha256(
        "\n".join(union).encode("utf-8")
    ).hexdigest()
    if union_hash != declared_hash:
        raise KeySetError(
            "union of complete and excluded key sets does not match the"
            " declared pairing-population key-set hash (R-008)"
        )
    record_complete: set[str] = set()
    record_excluded: set[str] = set()
    _check_duplicate_keys(records)
    for record in records:
        outcome = classify_record(record)
        if outcome["status"] == "complete":
            record_complete.add(record["item_key"])
        else:
            record_excluded.add(record["item_key"])
    if set(complete) != record_complete or set(excluded) != record_excluded:
        raise KeySetError(
            "declared key sets do not match the key sets recomputed from"
            " per-item records (R-008)"
        )


def derive_item_key(source_text: str) -> str:
    """Stable opaque item-key derivation pinned by the profile (R-008).

    itm-<first 16 hex of sha256(NFC-normalized text, utf-8)>. Keys are
    compared byte-exact after derivation, so Unicode normalization-variant
    near-duplicates collide and fail closed as duplicates.
    """
    if not isinstance(source_text, str):
        raise ColmAimsError("item-key derivation requires text (R-008)")
    normalized = unicodedata.normalize("NFC", source_text)
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return "itm-" + digest[:16]


def estimand_digest(estimand: dict[str, Any]) -> str:
    """Digest over all estimand-defining fields (R-011, R-032)."""
    if not isinstance(estimand, dict):
        raise ColmAimsError("estimand must be an object (R-011)")
    return canonical_estimand_digest(estimand)


def _cell_digest_checked(cell: dict[str, Any]) -> str:
    recorded = cell.get("estimand_digest")
    recomputed = estimand_digest(cell.get("estimand") or {})
    if recorded != recomputed:
        raise EstimandMismatchError(
            f"cell {cell.get('cell_id')!r} recorded estimand_digest does not"
            " match the digest recomputed over its estimand-defining fields"
            " (R-011)"
        )
    return recomputed


def check_poolable(cell_a: dict[str, Any], cell_b: dict[str, Any]) -> None:
    """Refuse pooling/comparison across differing estimand digests (R-011)."""
    digest_a = _cell_digest_checked(cell_a)
    digest_b = _cell_digest_checked(cell_b)
    if digest_a != digest_b:
        raise EstimandMismatchError(
            f"cells {cell_a.get('cell_id')!r} and {cell_b.get('cell_id')!r}"
            " carry differing estimand digests; pooling or comparing them is"
            " refused (R-011)"
        )


def _check_timing_summaries(
    cell: dict[str, Any], records: list[dict[str, Any]], tolerance: float
) -> None:
    """Recompute both timing summaries from records (R-006/R-010/R-015)."""
    recorded = cell["timing_summary_finite_only"]
    if "convention" in recorded:
        raise SchemaValidationError(
            "finite-only timing summary carries a sentinel-coded convention"
            " marker — the sentinel-coded historical summary is separately"
            " named and never pooled (R-006)"
        )
    if recorded.get("conditional_on") != "n_both_finite":
        raise SchemaValidationError(
            "finite-only timing summary must declare its conditional"
            " estimand over exactly n_both_finite (R-006)"
        )
    expected = finite_only_timing_summary(records)
    if recorded.get("n") != expected["n"]:
        raise RateError(
            f"finite-only timing summary n recorded {recorded.get('n')!r}"
            f" != {expected['n']!r} both-finite pairs (R-006)"
        )
    for statistic in TIMING_STATISTICS:
        value = recorded.get(statistic)
        if value is None or isinstance(value, bool) or not isinstance(
            value, (int, float)
        ):
            raise RateError(
                f"finite-only timing summary missing statistic"
                f" {statistic!r} (R-015)"
            )
        if abs(float(value) - expected[statistic]) > tolerance:
            raise RateError(
                f"finite-only timing summary statistic {statistic!r}"
                f" recorded {value!r} does not recompute from retained"
                f" per-item records (expected {expected[statistic]!r},"
                f" declared tolerance {tolerance!r}) (R-015)"
            )

    sentinel = cell["timing_summary_sentinel_coded_historical"]
    if sentinel.get("convention") != "timeout_coded_as_horizon":
        raise SchemaValidationError(
            "sentinel-coded historical summary must be separately named via"
            " its convention marker (R-006)"
        )
    expected_sentinel = _sentinel_coded_summary(records)
    if sentinel.get("n") != expected_sentinel["n"]:
        raise RateError(
            "sentinel-coded historical summary n does not recompute from"
            " retained complete pairs (R-015)"
        )
    for statistic in ("signed_index_mean", "signed_index_median"):
        value = sentinel.get(statistic)
        expected_value = expected_sentinel[statistic]
        if expected_value is None:
            continue
        if value is None or isinstance(value, bool) or not isinstance(
            value, (int, float)
        ):
            raise RateError(
                f"sentinel-coded summary missing statistic {statistic!r}"
            )
        if abs(float(value) - expected_value) > tolerance:
            raise RateError(
                f"sentinel-coded summary statistic {statistic!r} recorded"
                f" {value!r} does not recompute (expected"
                f" {expected_value!r}) (R-015)"
            )


def _check_interval(
    cell: dict[str, Any], records: list[dict[str, Any]], tolerance: float
) -> None:
    interval = cell.get("interval")
    if interval is None:
        return
    for key in ("procedure", "draw_count", "resampling_seeds", "statistic", "ci"):
        if key not in interval:
            raise SchemaValidationError(
                f"interval-bearing cell missing recorded identity field"
                f" {key!r} — the interval is non-certifying (R-015)"
            )
    recomputed = recompute_interval(records, interval)
    recorded_ci = interval["ci"]
    if (
        not isinstance(recorded_ci, list)
        or len(recorded_ci) != 2
        or any(
            abs(float(a) - float(b)) > tolerance
            for a, b in zip(recorded_ci, recomputed["ci"])
        )
    ):
        raise RateError(
            f"interval ci recorded {recorded_ci!r} does not reproduce under"
            f" the recorded procedure (recomputed {recomputed['ci']!r},"
            f" declared tolerance {tolerance!r}) (R-015)"
        )


def validate_cell(cell: dict[str, Any], records: list[dict[str, Any]]) -> None:
    """Full cell check: counts, rates, keys, summaries, digest (R-005..R-011)."""
    counts = cell["counts"]
    if counts.get("n_pairing_population") == 0:
        raise EmptyEvaluationError(
            f"cell {cell.get('cell_id')!r} declares n_pairing_population == 0"
            " — an empty evaluation is a typed error, never a"
            " trivially-passing cell (R-006/R-012)"
        )
    recorded = cell.get("estimand_digest")
    if recorded != estimand_digest(cell.get("estimand") or {}):
        raise EstimandMismatchError(
            f"cell {cell.get('cell_id')!r} recorded estimand_digest does not"
            " match the recomputed digest (R-011)"
        )
    tolerance = _declared_tolerance(cell)
    check_count_identities(counts, records)
    check_key_sets(cell, records)
    check_rates(cell)
    _check_timing_summaries(cell, records, tolerance)
    _check_interval(cell, records, tolerance)
