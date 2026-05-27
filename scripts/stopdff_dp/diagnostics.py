"""Coverage + ceiling-effect diagnostics for DP StopDFF traces.

Operates only on DPTrace objects produced by the solver; no I/O.
"""

from __future__ import annotations

from collections import Counter
from typing import Iterable, Sequence

from .types import DPTrace

POOLED_WARN_THRESHOLD = 0.05
MISSING_WARN_THRESHOLD = 0.01


def summarize_coverage(traces: Iterable[DPTrace]) -> dict:
    """Tally per-step coverage tags across all traces and return a verdict."""
    tag_counter: Counter[str] = Counter()
    total_cells = 0
    for trace in traces:
        for tag in trace.coverage_tags:
            tag_counter[tag] += 1
            total_cells += 1
    if total_cells == 0:
        return {
            "n_cells": 0,
            "fraction_exact": None,
            "fraction_pooled": None,
            "fraction_missing": None,
            "verdict": "warn",
            "reason": "no_cells",
        }

    fraction_exact = tag_counter["exact"] / total_cells
    fraction_pooled = tag_counter["pooled"] / total_cells
    fraction_missing = tag_counter["missing"] / total_cells

    if fraction_pooled > POOLED_WARN_THRESHOLD:
        verdict = "warn"
        reason = (
            f"fraction_pooled={fraction_pooled:.3f} > "
            f"{POOLED_WARN_THRESHOLD}"
        )
    elif fraction_missing > MISSING_WARN_THRESHOLD:
        verdict = "warn"
        reason = (
            f"fraction_missing={fraction_missing:.3f} > "
            f"{MISSING_WARN_THRESHOLD}"
        )
    else:
        verdict = "pass"
        reason = "thresholds_clean"

    return {
        "n_cells": total_cells,
        "fraction_exact": float(fraction_exact),
        "fraction_pooled": float(fraction_pooled),
        "fraction_missing": float(fraction_missing),
        "verdict": verdict,
        "reason": reason,
    }


def detect_ceiling_effects(
    mc_traces: Sequence[DPTrace],
    qa_traces: Sequence[DPTrace],
) -> dict:
    """Return a dict of binary flags describing potential ceiling effects."""
    def _all_stop_at(traces: Sequence[DPTrace], target: str) -> bool:
        if not traces:
            return False
        for t in traces:
            T = len(t.values) if t.values else 0
            if target == "first":
                if t.stop_step != 0:
                    return False
            elif target == "last":
                if t.stop_step != T - 1:
                    return False
            else:
                raise ValueError(target)
        return True

    n_trajectories = max(len(mc_traces), len(qa_traces))
    stopped = sum(
        1 for t in (*mc_traces, *qa_traces)
        if 0 <= t.stop_step < len(t.values)
    )
    never_stopped = sum(
        1 for t in (*mc_traces, *qa_traces) if t.stop_step >= len(t.values)
    )

    no_variance = (
        bool(mc_traces) and bool(qa_traces) and
        all(
            mc.stop_step == qa.stop_step
            for mc, qa in zip(mc_traces, qa_traces)
        )
    )

    return {
        "all_stop_at_first_prefix": _all_stop_at(mc_traces, "first")
            and _all_stop_at(qa_traces, "first"),
        "all_stop_at_final_prefix": _all_stop_at(mc_traces, "last")
            and _all_stop_at(qa_traces, "last"),
        "no_cross_format_stopping_variance": no_variance,
        "n_trajectories": n_trajectories,
        "n_stopped_cells": stopped,
        "n_never_stopped_cells": never_stopped,
    }


def continuation_model_collapsed(coverage_summary: dict) -> bool:
    """Heuristic for the 'continuation model collapse' diagnostic.

    True when every cell uses the most-pooled rung (i.e. every lookup
    fell through to pooled and the per-bucket structure carried no
    information).
    """
    return (
        coverage_summary.get("fraction_exact") == 0.0
        and coverage_summary.get("fraction_pooled") == 1.0
    )
