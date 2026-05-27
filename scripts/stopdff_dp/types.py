"""Shared dataclasses and column constants for the DP StopDFF pipeline.

Centralising these here keeps the adapter, DP solver, and continuation
estimators agreed on the same row schema without circular imports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

# Columns in the normalised adapter dataframe. Order is stable so
# writers / tests can rely on positional access where convenient.
ADAPTER_COLUMNS: tuple[str, ...] = (
    "subject",
    "item_id",
    "prefix_idx",
    "format",
    "split",
    "p_raw",
    "p_calibrated",
    "correct",
    "top_answer",
    "gold",
    "category",
    "option_set_id",
)

FORMATS: tuple[str, ...] = ("MC", "QA")
DEFAULT_FIT_SPLIT: str = "val"
DEFAULT_EVAL_SPLIT: str = "test"


@dataclass(frozen=True)
class RewardSchedule:
    """Parameters for the answer-utility function A_t(p) = R(t)*p + R_wrong*(1-p).

    Attributes
    ----------
    name : str
        Identifier the CLI surfaces (e.g. ``"power_mark"``).
    r_correct_early : float
        Reward for a correct stop in the early half of the question.
    r_correct_late : float
        Reward for a correct stop in the late half.
    r_wrong : float
        Penalty for a wrong stop (typically negative).
    power_split : float
        Prefix fraction (0–1) below which early reward applies.
    c_wait : float
        Per-step waiting cost.
    description : str
        Human-readable description echoed into the JSON output.
    """

    name: str
    r_correct_early: float
    r_correct_late: float
    r_wrong: float
    power_split: float
    c_wait: float
    description: str = ""

    def r_correct(self, prefix_fraction: float) -> float:
        return (
            self.r_correct_early
            if prefix_fraction < self.power_split
            else self.r_correct_late
        )


@dataclass
class DPTrace:
    """One DP trajectory result: per-step values + chosen stop step.

    Attributes
    ----------
    item_id : str
    fmt : str
    stop_step : int
        0-based index of the chosen stop prefix.
    values : list[float]
        V_t for each prefix t along the trajectory.
    answer_utilities : list[float]
        A_t(p_t) at each prefix.
    continuation_values : list[float]
        Estimated E[V_{t+1} | h_t] at each prefix; last entry is 0.0.
    coverage_tags : list[str]
        Per-step tag in {"exact","pooled","missing"} for diagnostics.
    """

    item_id: str
    fmt: str
    stop_step: int
    values: list[float] = field(default_factory=list)
    answer_utilities: list[float] = field(default_factory=list)
    continuation_values: list[float] = field(default_factory=list)
    coverage_tags: list[str] = field(default_factory=list)


def assert_columns(df_columns: Iterable[str]) -> None:
    """Validate that a dataframe has the canonical column set."""
    missing = set(ADAPTER_COLUMNS) - set(df_columns)
    if missing:
        raise ValueError(
            f"Adapter dataframe is missing canonical columns: "
            f"{sorted(missing)}. Expected: {ADAPTER_COLUMNS}."
        )
