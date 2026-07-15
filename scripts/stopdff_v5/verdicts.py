"""Cell / family verdicts and release validity (SCIENTIFIC_CONTRACT.md sections 11-13).

Primary material threshold: median absolute prefix-index shift = 1.

Cell (completed, converged):
    FAIL: L_c > 1
    PASS: U_c <= 1 and coverage clean and no ceiling flag and no MC gate overridden
    WARN: every other completed case

Family (across 96 cells), maximum-statistic control:
    FAIL: L_M > 1
    PASS: U_M <= 1 and every cell verdict PASS and no MC gate override active
    WARN: every other valid release
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .profile import CONTINUATION

MATERIAL_THRESHOLD = 1.0
CLEAN_FALLBACK_MAX = float(CONTINUATION["clean_fallback_fraction_max"])
CLEAN_MISSING_MAX = float(CONTINUATION["clean_missing_fraction_max"])


@dataclass
class TraceStop:
    stop_index: int
    never_buzz: bool
    T: int


def ceiling_flags(
    mc_stops: Sequence[TraceStop],
    qa_stops: Sequence[TraceStop],
    index_shifts: Sequence[int],
) -> dict[str, bool]:
    all_stops = list(mc_stops) + list(qa_stops)
    all_first = bool(all_stops) and all(s.stop_index == 0 for s in all_stops)
    all_final_or_never = bool(all_stops) and all(
        (s.never_buzz or s.stop_index == s.T - 1) for s in all_stops
    )
    all_zero_shift = bool(index_shifts) and all(int(d) == 0 for d in index_shifts)
    return {
        "all_answer_first_prefix": all_first,
        "all_answer_final_or_never": all_final_or_never,
        "all_paired_index_shifts_zero": all_zero_shift,
    }


def coverage_clean(fallback_fraction: float, missing_fraction: float) -> bool:
    return fallback_fraction <= CLEAN_FALLBACK_MAX and missing_fraction <= CLEAN_MISSING_MAX


def cell_verdict(
    *,
    abs_median_ci: Sequence[float],
    coverage_is_clean: bool,
    ceiling_any: bool,
    mc_gate_overridden: bool,
) -> str:
    """Return PASS | WARN | FAIL for a completed, converged cell."""
    lo, hi = float(abs_median_ci[0]), float(abs_median_ci[1])
    if lo > MATERIAL_THRESHOLD:
        return "FAIL"
    if hi <= MATERIAL_THRESHOLD and coverage_is_clean and not ceiling_any and not mc_gate_overridden:
        return "PASS"
    return "WARN"


def family_verdict(
    *,
    family_ci: Sequence[float],
    all_cells_pass: bool,
    mc_override_active: bool,
) -> str:
    lo, hi = float(family_ci[0]), float(family_ci[1])
    if lo > MATERIAL_THRESHOLD:
        return "FAIL"
    if hi <= MATERIAL_THRESHOLD and all_cells_pass and not mc_override_active:
        return "PASS"
    return "WARN"


@dataclass
class ReleaseCheck:
    valid: bool
    reasons: list[str]


def release_validity(
    *,
    expected_cell_keys: set[str],
    present_cell_keys: list[str],
    completed_keys: set[str],
    failed_keys: set[str],
    skipped_keys: set[str],
    all_calibrators_fitted: bool,
    all_fvi_converged: bool,
    manifests_valid: bool,
    cache_matches_aggregate: bool,
    bootstrap_valid: bool,
    family_valid: bool,
    backend_manifest_valid: bool,
    attempt_history_valid: bool,
) -> ReleaseCheck:
    """Recompute release validity (SCIENTIFIC_CONTRACT.md section 13)."""
    reasons: list[str] = []
    present_set = set(present_cell_keys)
    if len(present_cell_keys) != len(present_set):
        reasons.append("duplicate cell keys present")
    if present_set != expected_cell_keys:
        missing = sorted(expected_cell_keys - present_set)
        extra = sorted(present_set - expected_cell_keys)
        reasons.append(f"cell-set mismatch: missing={missing[:5]} extra={extra[:5]}")
    if completed_keys != expected_cell_keys:
        reasons.append("not all requested cells completed")
    if failed_keys:
        reasons.append(f"{len(failed_keys)} cell(s) failed")
    if skipped_keys:
        reasons.append(f"{len(skipped_keys)} cell(s) skipped")
    if not all_calibrators_fitted:
        reasons.append("a calibrator failed to fit")
    if not all_fvi_converged:
        reasons.append("an FVI fit did not converge")
    if not manifests_valid:
        reasons.append("a fingerprint/manifest failed to validate")
    if not cache_matches_aggregate:
        reasons.append("a cache entry does not match the aggregate")
    if not bootstrap_valid:
        reasons.append("bootstrap evidence invalid")
    if not family_valid:
        reasons.append("family-max evidence invalid")
    if not backend_manifest_valid:
        reasons.append("backend manifest invalid")
    if not attempt_history_valid:
        reasons.append("attempt history invalid")
    return ReleaseCheck(valid=(not reasons), reasons=reasons)
