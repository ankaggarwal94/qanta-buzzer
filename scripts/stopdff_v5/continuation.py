"""Continuation estimators (indexed by SCIENTIFIC_CONTRACT.md).

The estimator holds fixed per-bucket counts (from fit rows) and mutable per-bucket means
(updated by FVI). At a decision point it walks the ladder for the current
(continuation, category_pooling) mode and returns the mean of the first eligible rung
(count >= 3). A missing continuation returns 0 (abstain value) and is tagged "missing".

Coverage is relative to the estimator's intended first rung:
    primary  : first rung used
    fallback : a later eligible rung was used
    missing  : no eligible rung was found
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable

from .profile import CONTINUATION

P_BIN_EDGES = tuple(float(x) for x in CONTINUATION["probability_bin_edges"])
ENTROPY_BIN_EDGES = tuple(float(x) for x in CONTINUATION["entropy_bin_edges"])
MIN_COUNT = int(CONTINUATION["minimum_bucket_count"])

# Ladders keyed by (continuation_axis, category_pooling). Each rung is a tuple of
# conditioning-variable names. "prefix" is the prefix key (early_mid_late or exact idx).
LADDERS: dict[tuple[str, str], tuple[tuple[str, ...], ...]] = {
    ("empirical_bucket", "per_category"): (
        ("prefix", "format", "category", "p_bin", "entropy_bin"),
        ("prefix", "format", "category", "p_bin"),
        ("prefix", "format", "category"),
        ("prefix", "format"),
        ("format",),
    ),
    ("empirical_bucket", "pooled_category"): (
        ("prefix", "format", "p_bin", "entropy_bin"),
        ("prefix", "format", "p_bin"),
        ("prefix", "format"),
        ("format",),
    ),
    ("pooled_empirical", "per_category"): (
        ("prefix", "format", "category"),
        ("prefix", "format"),
        ("format",),
    ),
    ("pooled_empirical", "pooled_category"): (
        ("prefix", "format"),
        ("format",),
    ),
}


def binary_entropy_base2(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return float(-(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p)))


def assign_bin(value: float, edges: tuple[float, ...]) -> int:
    for i in range(len(edges) - 1):
        if edges[i] <= value < edges[i + 1]:
            return i
    if value < edges[0]:
        return 0
    return len(edges) - 2


def p_bin_of(p: float) -> int:
    return assign_bin(p, P_BIN_EDGES)


def entropy_bin_of(p: float) -> int:
    return assign_bin(binary_entropy_base2(p), ENTROPY_BIN_EDGES)


def prefix_key(prefix_bucketing: str, *, prefix_idx: int, prefix_fraction: float) -> Any:
    if prefix_bucketing == "exact_prefix":
        return int(prefix_idx)
    if prefix_bucketing == "early_mid_late":
        if prefix_fraction < 0.33:
            return "early"
        if prefix_fraction < 0.66:
            return "mid"
        return "late"
    raise ValueError(f"unknown prefix_bucketing {prefix_bucketing!r}")


def make_observation(
    *,
    prefix_bucketing: str,
    prefix_idx: int,
    prefix_fraction: float,
    fmt: str,
    category: str,
    p_calibrated: float,
) -> dict[str, Any]:
    return {
        "prefix": prefix_key(
            prefix_bucketing, prefix_idx=prefix_idx, prefix_fraction=prefix_fraction
        ),
        "format": fmt,
        "category": category,
        "p_bin": p_bin_of(p_calibrated),
        "entropy_bin": entropy_bin_of(p_calibrated),
    }


@dataclass
class ContinuationEstimator:
    continuation_axis: str  # empirical_bucket | pooled_empirical
    category_pooling: str  # per_category | pooled_category
    bucket_counts: dict[tuple, int] = field(default_factory=dict)
    bucket_means: dict[tuple, float] = field(default_factory=dict)
    min_count: int = MIN_COUNT
    _last_tag: str = field(default="missing", init=False)

    @property
    def ladder(self) -> tuple[tuple[str, ...], ...]:
        return LADDERS[(self.continuation_axis, self.category_pooling)]

    def rung_keys(self, obs: dict[str, Any]) -> list[tuple]:
        """Bucket keys for every rung applicable to this observation (FVI assignment)."""
        keys: list[tuple] = []
        for rung in self.ladder:
            keys.append((rung, *tuple(obs[name] for name in rung)))
        return keys

    def first_eligible(self, obs: dict[str, Any]) -> tuple[int, tuple] | None:
        for rung_idx, rung in enumerate(self.ladder):
            key = (rung, *tuple(obs[name] for name in rung))
            if self.bucket_counts.get(key, 0) >= self.min_count:
                return rung_idx, key
        return None

    def estimate(self, obs: dict[str, Any]) -> float:
        found = self.first_eligible(obs)
        if found is None:
            self._last_tag = "missing"
            return 0.0
        rung_idx, key = found
        self._last_tag = "primary" if rung_idx == 0 else "fallback"
        return float(self.bucket_means.get(key, 0.0))

    def coverage_tag(self, obs: dict[str, Any]) -> str:
        found = self.first_eligible(obs)
        if found is None:
            return "missing"
        return "primary" if found[0] == 0 else "fallback"


def build_counts(
    estimator: ContinuationEstimator, observations_per_trajectory: Iterable[list[dict[str, Any]]]
) -> None:
    """Populate fixed bucket_counts from fit-row nonterminal observations.

    observations_per_trajectory yields, per fit trajectory, the list of nonterminal
    observations (one per t in 0..T-2). Every rung key gets a count.
    """
    counts: dict[tuple, int] = {}
    for obs_list in observations_per_trajectory:
        for obs in obs_list:
            for key in estimator.rung_keys(obs):
                counts[key] = counts.get(key, 0) + 1
    estimator.bucket_counts = counts
    # Initialize eligible bucket means to 0 (SCIENTIFIC_CONTRACT.md 9.2).
    estimator.bucket_means = {k: 0.0 for k in counts}
