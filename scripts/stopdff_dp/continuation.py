"""Continuation-value estimators for finite-horizon DP StopDFF.

Three implementations are exposed:

* ``OracleTrajectoryEstimator`` -- upper-bound diagnostic using each
  realized held-out trajectory's next prefix probability. Marked
  non-confirmatory; the writer warns and records the flag.
* ``EmpiricalBucketEstimator`` -- primary confirmatory estimator. Fits
  E[V_{t+1} | prefix_bucket, format, subject_bucket, p_bin, entropy_bin]
  on the fit split (default ``val``) and looks up at decision time.
  Falls back along the FALLBACK_LADDER when a bucket is sparse.
* ``PooledEmpiricalEstimator`` -- convenience facade for callers who want
  to force the fallback ladder to start at the top rung. Internally
  delegates to ``EmpiricalBucketEstimator``.

All three guard against test-split leakage at fit time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence, TYPE_CHECKING

import numpy as np
import pandas as pd

from .types import assert_columns

if TYPE_CHECKING:
    from .types import RewardSchedule

# Pre-declared fallback ladder. Each rung is the set of conditioning
# variables that must still match for a bucket to count. The ladder
# walks specific -> general.
FALLBACK_LADDER: tuple[tuple[str, ...], ...] = (
    ("prefix_bucket", "format", "subject_bucket", "p_bin", "entropy_bin"),
    ("prefix_bucket", "format", "subject_bucket", "p_bin"),
    ("prefix_bucket", "format", "subject_bucket"),
    ("prefix_bucket", "format"),
    ("format",),
)

DEFAULT_P_BINS: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.01)
# Entropy is shannon over (p, 1-p); peaks at p=0.5. Three bins keep the
# joint bucket count tractable in the smoke pipeline.
DEFAULT_ENTROPY_BINS: tuple[float, ...] = (0.0, 0.5, 0.9, 1.01)


def _assign_p_bin(p: float, bin_edges: Sequence[float] = DEFAULT_P_BINS) -> int:
    """Return the index of the p-bin containing ``p``."""
    p = max(0.0, min(1.0, float(p)))
    for i in range(len(bin_edges) - 1):
        if bin_edges[i] <= p < bin_edges[i + 1]:
            return i
    return len(bin_edges) - 2


def _shannon_entropy(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return float(-(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p)))


def _assign_entropy_bin(
    p: float, bin_edges: Sequence[float] = DEFAULT_ENTROPY_BINS
) -> int:
    h = _shannon_entropy(p)
    for i in range(len(bin_edges) - 1):
        if bin_edges[i] <= h < bin_edges[i + 1]:
            return i
    return len(bin_edges) - 2


def _assign_prefix_bucket(prefix_fraction: float) -> str:
    # Matches scripts/compute_prefix_calibration.assign_bucket.
    if prefix_fraction < 0.33:
        return "early"
    if prefix_fraction < 0.66:
        return "mid"
    return "late"


def _compute_prefix_fraction(df: pd.DataFrame) -> pd.Series:
    """Compute prefix_fraction = (rank within (item_id, format)) / group size.

    Uses groupby().transform() to avoid the pandas >=2.2 DeprecationWarning
    that ``groupby().apply()`` emits when the function returns a Series.
    """
    grp = df.groupby(["item_id", "format"])
    size = grp["prefix_idx"].transform("size")
    # Rank within each group, 1-based, matching np.linspace(1/T, 1, T).
    rank = grp.cumcount() + 1
    return rank.astype(float) / size.astype(float)


@dataclass
class OracleTrajectoryEstimator:
    """Upper-bound diagnostic using realized next-step calibrated p.

    NON-CONFIRMATORY. Reports the exact V_{t+1} computed via backward
    induction over the realized sub-trajectory under the supplied
    ``schedule``. This leaks each item's realized future (p_{t+1}..p_T)
    to the present and is intended only as an upper bound on the
    realisable DP value.
    """

    confirmatory: bool = False

    @classmethod
    def fit(
        cls,
        *,
        fit_df: "pd.DataFrame | None" = None,
        schedule: "RewardSchedule | None" = None,
        fit_split_name: str = "val",
        min_bucket_size: int = 3,
    ) -> "OracleTrajectoryEstimator":
        """Return a fresh oracle estimator. All parameters unused — kept
        for signature symmetry with EmpiricalBucketEstimator.fit so the
        CLI dispatch can construct any of the three estimators with the
        same call."""
        return cls()

    def estimate(
        self,
        *,
        item_trajectory: Sequence[float],
        item_prefix_fractions: Sequence[float],
        t: int,
        schedule: "RewardSchedule",
        **_kwargs,
    ) -> float:
        """Return V_{t+1} via exact backward induction on the realized
        sub-trajectory under ``schedule``.

        NON-CONFIRMATORY: this leaks each item's realized future p_{t+1}..p_T
        into the present and is intended only as an upper bound on the
        realisable DP value. Fixed scale to match the DP solver's reward
        units (P1 PR review fix 2026-05-27).
        """
        if t + 1 >= len(item_trajectory):
            return 0.0
        from .rewards import answer_utility
        sub_p = list(item_trajectory[t + 1:])
        sub_fractions = list(item_prefix_fractions[t + 1:])
        T_sub = len(sub_p)
        # V_T = max(A_T, 0)
        v = max(answer_utility(sub_p[T_sub - 1], sub_fractions[T_sub - 1], schedule), 0.0)
        # Backward recursion: V_i = max(A_i, -c_wait + V_{i+1})
        for i in range(T_sub - 2, -1, -1):
            A_i = answer_utility(sub_p[i], sub_fractions[i], schedule)
            wait_value = -schedule.c_wait + v
            v = max(A_i, wait_value)
        return v

    def coverage_tag(self, *_args, **_kwargs) -> str:
        return "exact"


@dataclass
class EmpiricalBucketEstimator:
    """Validation-data continuation estimator with fallback ladder.

    Buckets are pre-computed once at ``fit`` time so per-trajectory
    lookups are O(1). The estimator records the rung used for the most
    recent lookup so the diagnostics layer can tally coverage.
    """

    bucket_means: dict[tuple, float] = field(default_factory=dict)
    bucket_counts: dict[tuple, int] = field(default_factory=dict)
    fit_split_name: str = "val"
    min_bucket_size: int = 3
    _last_rung: tuple[str, ...] | None = field(default=None, init=False)
    _last_tag: str = field(default="missing", init=False)
    confirmatory: bool = True

    @classmethod
    def fit(
        cls,
        *,
        fit_df: pd.DataFrame,
        schedule: "RewardSchedule",
        fit_split_name: str = "val",
        min_bucket_size: int = 3,
        num_value_iterations: int = 3,
    ) -> "EmpiricalBucketEstimator":
        """Fit per-bucket V_{t+1} means on the fit split via value iteration.

        Bucket values are in REWARD UNITS (matching the DP solver's
        answer_utility scale), not probabilities. We bootstrap the
        continuation fixed point by running ``num_value_iterations``
        passes of: (current bucket lookup -> DP -> aggregate V_{t+1}
        traces back into buckets). With monotone Bellman recursion the
        iteration converges quickly; the default 3 is sufficient on the
        smoke datasets.
        """
        assert_columns(fit_df.columns)
        other_splits = set(fit_df["split"]) - {fit_split_name}
        if other_splits:
            raise ValueError(
                "EmpiricalBucketEstimator.fit refusing to fit on test (or "
                f"non-fit) split data: leakage candidates: {other_splits}. "
                f"Pass a dataframe filtered to split == {fit_split_name!r}."
            )

        # Local import to avoid module-load circular with dp_solver (which is fine
        # at function-call time but not at module-import time).
        from .dp_solver import solve_trajectory

        # Sort and compute bucket keys once.
        df = fit_df.sort_values(["item_id", "format", "prefix_idx"]).copy()
        df["prefix_fraction"] = _compute_prefix_fraction(df)
        df["prefix_bucket"] = df["prefix_fraction"].map(_assign_prefix_bucket)
        df["subject_bucket"] = df["subject"]
        df["p_bin"] = df["p_calibrated"].map(_assign_p_bin)
        df["entropy_bin"] = df["p_calibrated"].map(_assign_entropy_bin)

        # Initialize estimator with empty buckets so the first iteration's
        # continuation_fn returns 0 (matches the "no info" baseline).
        estimator = cls(
            bucket_means={},
            bucket_counts={},
            fit_split_name=fit_split_name,
            min_bucket_size=min_bucket_size,
        )

        # Group val trajectories once.
        groups = list(df.groupby(["item_id", "format"], sort=False))

        for _iteration in range(num_value_iterations):
            # Per-rung accumulators for V_{t+1} samples observed in this pass.
            pairs_per_rung: dict[tuple[str, ...], list[tuple[tuple, float]]] = {
                rung: [] for rung in FALLBACK_LADDER
            }

            for (_item_id, fmt), group in groups:
                p_traj = group["p_calibrated"].tolist()
                prefix_fractions = group["prefix_fraction"].tolist()
                subject = group["subject"].iloc[0]
                T = len(p_traj)
                if T < 2:
                    # Single-prefix items contribute no V_{t+1} samples (the
                    # terminal has continuation = 0 by DP convention).
                    continue

                # Closure reads estimator state lazily; updated between iterations.
                def _cont(
                    t: int,
                    p: float,
                    prefix_fraction: float,
                    _fmt: str = fmt,
                    _subj: str = subject,
                ) -> float:
                    return estimator.estimate(
                        prefix_bucket=_assign_prefix_bucket(prefix_fraction),
                        fmt=_fmt,
                        subject_bucket=_subj,
                        p_bin=_assign_p_bin(p),
                        entropy_bin=_assign_entropy_bin(p),
                    )

                trace = solve_trajectory(
                    p_trajectory=p_traj,
                    prefix_fractions=prefix_fractions,
                    schedule=schedule,
                    continuation_fn=_cont,
                )

                # At time t we ask "what is E[V_{t+1} | h_t]?", so bucket the
                # V_{t+1} value by the conditioning variables OBSERVED AT t.
                for t in range(T - 1):
                    lookups = {
                        "prefix_bucket": group["prefix_bucket"].iloc[t],
                        "format": fmt,
                        "subject_bucket": subject,
                        "p_bin": int(group["p_bin"].iloc[t]),
                        "entropy_bin": int(group["entropy_bin"].iloc[t]),
                    }
                    v_next = float(trace.values[t + 1])
                    for rung in FALLBACK_LADDER:
                        key = (rung, *tuple(lookups[name] for name in rung))
                        pairs_per_rung[rung].append((key, v_next))

            # Aggregate into bucket_means / bucket_counts. Replace the entire
            # state each iteration; the value-iteration fixed point converges
            # quickly because the DP recursion is monotone.
            new_means: dict[tuple, float] = {}
            new_counts: dict[tuple, int] = {}
            for rung, pairs in pairs_per_rung.items():
                bucket_to_vs: dict[tuple, list[float]] = {}
                for key, v in pairs:
                    bucket_to_vs.setdefault(key, []).append(v)
                for key, vs in bucket_to_vs.items():
                    new_means[key] = float(sum(vs) / len(vs))
                    new_counts[key] = len(vs)
            estimator.bucket_means = new_means
            estimator.bucket_counts = new_counts

        return estimator

    def estimate(
        self,
        *,
        prefix_bucket: str,
        fmt: str,
        subject_bucket: str,
        p_bin: int,
        entropy_bin: int,
        **_kwargs,
    ) -> float:
        """Look up E[V_{t+1}] along the fallback ladder.

        Records the rung used in ``self._last_rung`` and a coverage tag
        in {"exact","pooled","missing"} in ``self._last_tag`` so the
        diagnostics layer can tally fallback usage.
        """
        lookups = {
            "prefix_bucket": prefix_bucket,
            "format": fmt,
            "subject_bucket": subject_bucket,
            "p_bin": p_bin,
            "entropy_bin": entropy_bin,
        }
        for rung_idx, rung in enumerate(FALLBACK_LADDER):
            key = (rung, *tuple(lookups[name] for name in rung))
            count = self.bucket_counts.get(key, 0)
            if count >= self.min_bucket_size:
                self._last_rung = rung
                self._last_tag = "exact" if rung_idx == 0 else "pooled"
                return self.bucket_means.get(key, 0.0)
        self._last_rung = None
        self._last_tag = "missing"
        return 0.0

    def last_coverage_tag_for(
        self,
        *,
        prefix_bucket: str,
        fmt: str,
        subject_bucket: str,
        p_bin: int,
        entropy_bin: int,
    ) -> str:
        """Run a lookup and return the resulting coverage tag.

        Convenience for unit tests; production callers use ``estimate``
        and then read ``_last_tag``.
        """
        self.estimate(
            prefix_bucket=prefix_bucket,
            fmt=fmt,
            subject_bucket=subject_bucket,
            p_bin=p_bin,
            entropy_bin=entropy_bin,
        )
        return self._last_tag


@dataclass
class PooledEmpiricalEstimator:
    """Force-pooled variant of EmpiricalBucketEstimator.

    Skips the two most-specific rungs of FALLBACK_LADDER. Useful when
    the operator already knows the bucket grid is too sparse and wants
    the diagnostics to record ``pooled`` everywhere.
    """

    inner: EmpiricalBucketEstimator
    confirmatory: bool = True

    @property
    def _last_tag(self) -> str:
        """Proxy to the inner estimator's last coverage tag.

        Bug fix: orchestrator reads ``getattr(estimator, "_last_tag",
        "exact")`` off the outer wrapper. Without this proxy, the default
        "exact" is silently returned and coverage diagnostics are wrong.
        """
        return self.inner._last_tag

    @property
    def _last_rung(self) -> tuple[str, ...] | None:
        return self.inner._last_rung

    @classmethod
    def fit(
        cls,
        *,
        fit_df: pd.DataFrame,
        schedule: "RewardSchedule",
        fit_split_name: str = "val",
        min_bucket_size: int = 3,
    ) -> "PooledEmpiricalEstimator":
        return cls(
            inner=EmpiricalBucketEstimator.fit(
                fit_df=fit_df,
                schedule=schedule,
                fit_split_name=fit_split_name,
                min_bucket_size=min_bucket_size,
            )
        )

    def estimate(self, **kwargs) -> float:
        # Skip the most specific rungs by pretending entropy_bin and
        # p_bin are wildcards: hand the inner estimator a value that
        # cannot match the per-row p_bin/entropy_bin distribution, so
        # the first two rungs fail their count check and the ladder
        # falls through to rung index 2.
        kwargs["entropy_bin"] = -1
        kwargs["p_bin"] = -1
        return self.inner.estimate(**kwargs)

    def last_coverage_tag_for(self, **kwargs) -> str:
        kwargs["entropy_bin"] = -1
        kwargs["p_bin"] = -1
        return self.inner.last_coverage_tag_for(**kwargs)
