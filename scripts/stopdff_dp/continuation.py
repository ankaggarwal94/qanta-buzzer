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
from typing import Sequence

import numpy as np
import pandas as pd

from .types import assert_columns

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

    NON-CONFIRMATORY. Reports each realized p_{t+1} on the held-out
    trajectory as the continuation expectation E[V_{t+1} | h_t]. This
    leaks the future to the present and is intended only as an upper
    bound on the realisable DP value.
    """

    confirmatory: bool = False

    @classmethod
    def fit(
        cls,
        *,
        fit_df: pd.DataFrame | None = None,
        fit_split_name: str = "val",
        min_bucket_size: int = 3,
    ) -> "OracleTrajectoryEstimator":
        """Return a fresh oracle estimator. fit_df is unused (oracle needs no data fit).

        The signature matches EmpiricalBucketEstimator.fit so Task 8's CLI dispatch
        can construct any of the three estimators with the same call.
        """
        return cls()

    def estimate(
        self,
        *,
        item_trajectory: Sequence[float],
        t: int,
        **_kwargs,
    ) -> float:
        """Return p_{t+1} (or 0.0 at the terminal step)."""
        if t + 1 >= len(item_trajectory):
            return 0.0
        return float(item_trajectory[t + 1])

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
        fit_split_name: str = "val",
        min_bucket_size: int = 3,
    ) -> "EmpiricalBucketEstimator":
        """Fit per-bucket V_{t+1} means on the fit split only."""
        assert_columns(fit_df.columns)
        other_splits = set(fit_df["split"]) - {fit_split_name}
        if other_splits:
            raise ValueError(
                "EmpiricalBucketEstimator.fit refusing to fit on test (or "
                f"non-fit) split data: leakage candidates: {other_splits}. "
                f"Pass a dataframe filtered to split == {fit_split_name!r}."
            )

        # Compute per-(item, format) "next step calibrated prob" as the
        # supervised target for V_{t+1}. The fit dataframe already
        # contains every prefix; we shift within (item_id, format).
        df = fit_df.sort_values(["item_id", "format", "prefix_idx"]).copy()
        df["v_next"] = (
            df.groupby(["item_id", "format"])["p_calibrated"].shift(-1)
        )
        # Prefix fraction must be computed over the FULL trajectory (pre-drop)
        # so prefix_idx=0 of a 4-prefix item lands at 0.25, not 0.33.
        df["prefix_fraction"] = _compute_prefix_fraction(df)
        df["prefix_bucket"] = df["prefix_fraction"].map(_assign_prefix_bucket)
        df["subject_bucket"] = df["subject"]
        df["p_bin"] = df["p_calibrated"].map(_assign_p_bin)
        df["entropy_bin"] = df["p_calibrated"].map(_assign_entropy_bin)

        # Drop terminal rows -- at t=T-1 the DP solver enforces continuation=0
        # directly (dp_solver.py), so empirical buckets only need non-terminal
        # observations of E[p_{t+1}] as a proxy for E[V_{t+1}].
        non_terminal = df.dropna(subset=["v_next"])

        bucket_means: dict[tuple, float] = {}
        bucket_counts: dict[tuple, int] = {}
        for rung in FALLBACK_LADDER:
            grouped = non_terminal.groupby(list(rung))["v_next"]
            means = grouped.mean()
            counts = grouped.count()
            for raw_key, mean_value in means.items():
                key = raw_key if isinstance(raw_key, tuple) else (raw_key,)
                bucket_means[(rung, *key)] = float(mean_value)
                bucket_counts[(rung, *key)] = int(counts.loc[raw_key])

        return cls(
            bucket_means=bucket_means,
            bucket_counts=bucket_counts,
            fit_split_name=fit_split_name,
            min_bucket_size=min_bucket_size,
        )

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

    @classmethod
    def fit(
        cls,
        *,
        fit_df: pd.DataFrame,
        fit_split_name: str = "val",
        min_bucket_size: int = 3,
    ) -> "PooledEmpiricalEstimator":
        return cls(
            inner=EmpiricalBucketEstimator.fit(
                fit_df=fit_df,
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
