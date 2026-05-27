"""Bellman backward induction for finite-horizon DP stopping.

Implements:
    A_t(p)   = R(t) * p + R_wrong * (1 - p)
    V_T(h_T) = max(A_T(p_T), 0)
    V_t(h_t) = max(
        A_t(p_t),
        -c_wait + E[V_{t+1}(h_{t+1}) | h_t]
    )

The continuation expectation is passed in as ``continuation_fn`` so the
same solver works for the three estimators (oracle / empirical_bucket /
pooled_empirical). Callers supply the per-(item, format) trajectory and
the solver returns a DPTrace.
"""

from __future__ import annotations

from typing import Callable

from .rewards import answer_utility
from .types import DPTrace, RewardSchedule

ContinuationFn = Callable[..., float]


def solve_trajectory(
    *,
    p_trajectory: list[float],
    prefix_fractions: list[float],
    schedule: RewardSchedule,
    continuation_fn: ContinuationFn,
    item_id: str = "",
    fmt: str = "",
    coverage_tagger: Callable[[int], str] | None = None,
) -> DPTrace:
    """Run backward induction over a single (item, format) trajectory.

    Parameters
    ----------
    p_trajectory : list[float]
        Calibrated probabilities p_t for t=0..T.
    prefix_fractions : list[float]
        Prefix position (char_len / full_len) for each t. Same length.
    schedule : RewardSchedule
        Reward parameters.
    continuation_fn : callable
        Called as ``continuation_fn(t, p=p_t, prefix_fraction=...)`` and
        returns E[V_{t+1} | h_t]. The solver shields callers from the
        Bellman bookkeeping; estimators do not need to know V_t.
    item_id, fmt : str
        Passed through to the resulting DPTrace.
    coverage_tagger : callable, optional
        Called as ``coverage_tagger(t)`` to label per-step bucket
        coverage ("exact"/"pooled"/"missing"). Defaults to "exact".

    Returns
    -------
    DPTrace
        Per-step values, utilities, continuation estimates, stop step.

    Notes
    -----
    ``stop_step == len(p_trajectory)`` encodes "never stop" (when every
    A_t(p_t) <= -c_wait + continuation_t for all t and V_T <= 0).
    """
    T = len(p_trajectory)
    if T == 0:
        return DPTrace(item_id=item_id, fmt=fmt, stop_step=0)
    if len(prefix_fractions) != T:
        raise ValueError(
            "prefix_fractions must align with p_trajectory "
            f"(got {len(prefix_fractions)} vs {T})."
        )

    answer_utilities = [
        answer_utility(p_trajectory[t], prefix_fractions[t], schedule)
        for t in range(T)
    ]

    # Compute V_t and "stop now?" flags backward, then walk forward to
    # extract the first prefix at which the optimal action is to stop.
    values = [0.0] * T
    continuation_values = [0.0] * T
    stop_now: list[bool] = [False] * T

    # Terminal step: V_T = max(A_T, 0). Stop at T iff A_T > 0.
    terminal_value = max(answer_utilities[T - 1], 0.0)
    values[T - 1] = terminal_value
    continuation_values[T - 1] = 0.0
    stop_now[T - 1] = answer_utilities[T - 1] > 0.0

    # Backward recursion for t = T-2 .. 0.
    for t in range(T - 2, -1, -1):
        cont = float(
            continuation_fn(
                t,
                p=p_trajectory[t],
                prefix_fraction=prefix_fractions[t],
            )
        )
        continuation_values[t] = cont
        wait_value = -schedule.c_wait + cont
        if answer_utilities[t] > wait_value:
            values[t] = answer_utilities[t]
            stop_now[t] = True
        else:
            values[t] = wait_value
            stop_now[t] = False

    # Forward walk to find the first t where the optimal action is to stop.
    stop_step = T  # default: never stop
    for t in range(T):
        if stop_now[t]:
            stop_step = t
            break

    coverage_tags = [
        coverage_tagger(t) if coverage_tagger is not None else "exact"
        for t in range(T)
    ]

    return DPTrace(
        item_id=item_id,
        fmt=fmt,
        stop_step=stop_step,
        values=values,
        answer_utilities=answer_utilities,
        continuation_values=continuation_values,
        coverage_tags=coverage_tags,
    )


def stopdff_for_item(
    *,
    mc_trace: DPTrace,
    qa_trace: DPTrace,
) -> int:
    """Compute StopDFF_{sj} = stop_step_MC - stop_step_QA (signed)."""
    return mc_trace.stop_step - qa_trace.stop_step
