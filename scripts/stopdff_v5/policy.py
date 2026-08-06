"""Three-action Bellman policy (indexed by SCIENTIFIC_CONTRACT.md).

At every prefix t (including nonterminal) the action set is ANSWER / WAIT / ABSTAIN.
ABSTAIN means "never buzz" and has value 0.

    A_t(p_t) = R_correct(t) * p_t + R_wrong * (1 - p_t)
    V_{T-1}  = max(A_{T-1}, 0)
    V_t      = max(A_t, -c_wait + E[V_{t+1} | h_t], 0)

Tie policy (deterministic, conservative):
    nonterminal exact-equality priority: WAIT  > ABSTAIN > ANSWER
    terminal    exact-equality priority: ABSTAIN > ANSWER
ANSWER is chosen only when strictly greater than BOTH the wait value and the abstain
value (0). stop_index = first prefix where ANSWER is chosen; stop_index = T when ANSWER
is never chosen; never_buzz = (stop_index == T).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

from .rewards import RewardSchedule, answer_utility

# continuation_fn(t, p=p_t, prefix_fraction=frac_t) -> E[V_{t+1} | h_t]
ContinuationFn = Callable[..., float]
# coverage_tagger(t, p=p_t, prefix_fraction=frac_t) -> "primary"|"fallback"|"missing"
CoverageTagger = Callable[..., str]

ABSTAIN_VALUE = 0.0


@dataclass
class DPTrace:
    item_id: str
    fmt: str
    stop_index: int
    never_buzz: bool
    values: list[float] = field(default_factory=list)
    answer_utilities: list[float] = field(default_factory=list)
    continuation_values: list[float] = field(default_factory=list)
    coverage_tags: list[str] = field(default_factory=list)

    @property
    def T(self) -> int:
        return len(self.values)


def solve_trajectory(
    *,
    p_trajectory: Sequence[float],
    prefix_fractions: Sequence[float],
    schedule: RewardSchedule,
    continuation_fn: ContinuationFn,
    item_id: str = "",
    fmt: str = "",
    coverage_tagger: CoverageTagger | None = None,
) -> DPTrace:
    """Backward-induction solve of one (item, format) trajectory.

    ``continuation_fn`` reads a frozen continuation map (independent of the future of
    this trajectory), so evaluating it during the backward pass and reusing the stored
    values during the forward stop-decision pass is consistent.
    """
    T = len(p_trajectory)
    if T != len(prefix_fractions):
        raise ValueError(
            f"prefix_fractions must align with p_trajectory (got {len(prefix_fractions)} vs {T})."
        )
    if T == 0:
        return DPTrace(item_id=item_id, fmt=fmt, stop_index=0, never_buzz=True)

    answer_utils = [
        answer_utility(float(p_trajectory[t]), float(prefix_fractions[t]), schedule)
        for t in range(T)
    ]
    values = [0.0] * T
    continuation_values = [0.0] * T
    coverage_tags = ["primary"] * T

    # Terminal: V_{T-1} = max(A_{T-1}, 0). No continuation.
    values[T - 1] = max(answer_utils[T - 1], ABSTAIN_VALUE)
    continuation_values[T - 1] = 0.0
    if coverage_tagger is not None:
        coverage_tags[T - 1] = coverage_tagger(
            T - 1, p=float(p_trajectory[T - 1]), prefix_fraction=float(prefix_fractions[T - 1])
        )

    # Backward recursion for t = T-2 .. 0.
    for t in range(T - 2, -1, -1):
        cont = float(
            continuation_fn(t, p=float(p_trajectory[t]), prefix_fraction=float(prefix_fractions[t]))
        )
        continuation_values[t] = cont
        wait_value = -schedule.wait_cost + cont
        values[t] = max(answer_utils[t], wait_value, ABSTAIN_VALUE)
        if coverage_tagger is not None:
            coverage_tags[t] = coverage_tagger(
                t, p=float(p_trajectory[t]), prefix_fraction=float(prefix_fractions[t])
            )

    # Forward pass: first prefix where ANSWER is strictly optimal.
    stop_index = T
    for t in range(T):
        a = answer_utils[t]
        if t == T - 1:
            answered = a > ABSTAIN_VALUE  # terminal: ABSTAIN > ANSWER on tie
        else:
            wait_value = -schedule.wait_cost + continuation_values[t]
            answered = (a > wait_value) and (a > ABSTAIN_VALUE)
        if answered:
            stop_index = t
            break

    return DPTrace(
        item_id=item_id,
        fmt=fmt,
        stop_index=stop_index,
        never_buzz=(stop_index == T),
        values=values,
        answer_utilities=answer_utils,
        continuation_values=continuation_values,
        coverage_tags=coverage_tags,
    )


def signed_index_shift(mc_trace: DPTrace, qa_trace: DPTrace) -> int:
    """D_i^index = tau_{i,MC} - tau_{i,QA}. Negative => MC stops earlier."""
    return mc_trace.stop_index - qa_trace.stop_index


def stop_fraction(trace: DPTrace, prefix_fractions: Sequence[float]) -> float:
    """Descriptive stop fraction: prefix_fraction[stop] when it answers, else 1.0."""
    if trace.never_buzz:
        return 1.0
    return float(prefix_fractions[trace.stop_index])
