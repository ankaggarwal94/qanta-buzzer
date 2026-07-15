"""Fitted value iteration (SCIENTIFIC_CONTRACT.md section 9).

Synchronous, float64, math.fsum reductions, zero initialization, fixed damping 0.5,
two-consecutive-iteration convergence, cycle detection, and value-bound checks.

    V^{k+1}_b = 0.5 * raw_mean_target_b + 0.5 * V^k_b
    Delta_k   = max_b |V^{k+1}_b - V^k_b|
    converged when Delta_k <= tolerance for two consecutive iterations (k >= 2)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

from .continuation import ContinuationEstimator
from .policy import solve_trajectory
from .rewards import RewardSchedule

DAMPING = 0.5
_BOUND_SLACK = 1e-12


@dataclass
class FitTrajectory:
    item_id: str
    fmt: str
    category: str
    p_trajectory: list[float]
    prefix_fractions: list[float]
    obs_at_t: list[dict]  # observation per nonterminal t (len == T-1), for lookup/assignment

    @property
    def T(self) -> int:
        return len(self.p_trajectory)


@dataclass
class FVIResult:
    status: str  # converged | max_iterations_reached | cycle_detected | value_out_of_bounds
    converged: bool
    iterations: int
    final_delta: float
    delta_history: list[float] = field(default_factory=list)
    tolerance: str = ""
    max_iterations: int = 0
    num_buckets: int = 0


def _cycle_signature(means: dict[tuple, float]) -> tuple:
    return tuple(sorted((repr(k), round(v, 12)) for k, v in means.items()))


def run_fvi(
    estimator: ContinuationEstimator,
    trajectories: Sequence[FitTrajectory],
    schedule: RewardSchedule,
    *,
    tolerance: float,
    max_iterations: int,
    tolerance_label: str = "",
    damping: float = DAMPING,
) -> FVIResult:
    """Run damped FVI to update ``estimator.bucket_means`` in place.

    Bucket counts must already be populated (build_counts). Bucket means start at 0.
    """
    max_reward = schedule.max_correct_reward
    means = {k: 0.0 for k in estimator.bucket_counts}
    estimator.bucket_means = means

    # Pre-sort trajectories by (item_id, fmt) for deterministic accumulation order.
    ordered = sorted(trajectories, key=lambda tr: (tr.item_id, tr.fmt))

    delta_history: list[float] = []
    seen_states: set[tuple] = set()
    consecutive_ok = 0

    for k in range(1, max_iterations + 1):
        # Accumulate targets per bucket key: list of (item_id, prefix_idx, value).
        target_lists: dict[tuple, list[tuple]] = {key: [] for key in estimator.bucket_counts}
        for tr in ordered:
            if tr.T < 2:
                continue

            def _cont(t: int, p: float, prefix_fraction: float, _tr=tr) -> float:
                return estimator.estimate(_tr.obs_at_t[t])

            trace = solve_trajectory(
                p_trajectory=tr.p_trajectory,
                prefix_fractions=tr.prefix_fractions,
                schedule=schedule,
                continuation_fn=_cont,
                item_id=tr.item_id,
                fmt=tr.fmt,
            )
            for t in range(tr.T - 1):
                target = float(trace.values[t + 1])
                for key in estimator.rung_keys(tr.obs_at_t[t]):
                    target_lists[key].append((tr.item_id, t, target))

        new_means: dict[tuple, float] = {}
        max_delta = 0.0
        for key, count in estimator.bucket_counts.items():
            entries = target_lists.get(key, [])
            entries.sort(key=lambda e: (e[0], e[1]))
            raw_mean = math.fsum(v for _, _, v in entries) / count if count else 0.0
            damped = damping * raw_mean + (1.0 - damping) * means[key]
            if damped < -_BOUND_SLACK or damped > max_reward + _BOUND_SLACK:
                return FVIResult(
                    status="value_out_of_bounds",
                    converged=False,
                    iterations=k,
                    final_delta=abs(damped - means[key]),
                    delta_history=delta_history,
                    tolerance=tolerance_label,
                    max_iterations=max_iterations,
                    num_buckets=len(estimator.bucket_counts),
                )
            new_means[key] = damped
            max_delta = max(max_delta, abs(damped - means[key]))

        means = new_means
        estimator.bucket_means = means
        delta_history.append(max_delta)

        if k >= 2 and max_delta <= tolerance:
            consecutive_ok += 1
            if consecutive_ok >= 2:
                return FVIResult(
                    status="converged",
                    converged=True,
                    iterations=k,
                    final_delta=max_delta,
                    delta_history=delta_history,
                    tolerance=tolerance_label,
                    max_iterations=max_iterations,
                    num_buckets=len(estimator.bucket_counts),
                )
        else:
            consecutive_ok = 0

        sig = _cycle_signature(means)
        if sig in seen_states:
            return FVIResult(
                status="cycle_detected",
                converged=False,
                iterations=k,
                final_delta=max_delta,
                delta_history=delta_history,
                tolerance=tolerance_label,
                max_iterations=max_iterations,
                num_buckets=len(estimator.bucket_counts),
            )
        seen_states.add(sig)

    return FVIResult(
        status="max_iterations_reached",
        converged=False,
        iterations=max_iterations,
        final_delta=delta_history[-1] if delta_history else math.inf,
        delta_history=delta_history,
        tolerance=tolerance_label,
        max_iterations=max_iterations,
        num_buckets=len(estimator.bucket_counts),
    )
