"""Input-validation tests for scripts.compute_csli.bootstrap_ci (Iter1 IN-02).

The prior bootstrap_ci implementation silently produced degenerate
outputs on invalid inputs:

- ``values.ndim > 1`` made np.random.choice resample along axis 0
  with implicit broadcasting, producing per-row sub-statistics that
  np.mean averaged into a number bearing no relation to a CI.
- ``n_resamples <= 0`` produced an empty `means` array; np.percentile
  returned nan silently.
- ``confidence`` outside (0, 1) produced out-of-range percentiles
  that np.percentile clamps to extrema, silently returning min/max
  of `means` rather than a real CI.

All three are user-input errors; the audit JSON must not silently
carry nan or truncated CIs derived from them. Iter1 IN-02 adds
ValueError guards. These tests pin the contract.

The tests also include a happy-path regression test that confirms the
guards do not interfere with normal usage.
"""

from __future__ import annotations

import sys

# DATA-05 guard interaction (see WR-01):
# scripts.compute_csli's module-level _assert_no_controls_import()
# fires whenever ``evaluation.controls`` is already in
# ``sys.modules`` at our import time. Other test files transitively
# load evaluation.controls via ``scripts/evaluate_all.py`` (line 49)
# during pytest collection -- ``tests/test_pipeline_split_contracts.py``
# is the specific path in the focused-test suite. Pytest collects
# test files in CLI / alphabetical order, so the contamination
# depends on what was collected before us. Drop the offending
# module (it will be re-imported lazily on next use by anything
# that needs it) and any stale ``scripts.compute_csli`` cache so
# the next import sees a clean state. Local to this test file;
# does not modify shared conftest.py.
sys.modules.pop("evaluation.controls", None)
sys.modules.pop("scripts.compute_csli", None)

import numpy as np
import pytest

from scripts.compute_csli import bootstrap_ci


# ---------------------------------------------------------------------------
# Iter1 IN-02 guards
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_input,description",
    [
        (np.zeros((3, 4)), "2-D array"),
        (np.zeros((2, 2, 2)), "3-D array"),
    ],
)
def test_bootstrap_ci_rejects_non_1d_values(
    bad_input: np.ndarray,
    description: str,
) -> None:
    """``values.ndim != 1`` raises ValueError, never silently coerced.

    The prior implementation would have used multidim broadcasting
    inside np.random.choice and produced a number bearing no relation
    to the intended bootstrap mean. Fail closed instead.
    """
    with pytest.raises(ValueError, match="1-D input"):
        bootstrap_ci(bad_input, n_resamples=10, confidence=0.95, seed=789685)
    # description used in parametrize id only; reference suppresses unused-var lints
    assert description


@pytest.mark.parametrize("bad_n", [0, -1, -1000])
def test_bootstrap_ci_rejects_nonpositive_n_resamples(bad_n: int) -> None:
    """``n_resamples <= 0`` raises ValueError instead of returning nan.

    The prior implementation produced an empty `means` array and
    np.percentile silently returned nan. The audit JSON must not
    carry nan CIs derived from a typo in `n_resamples=0`.
    """
    with pytest.raises(ValueError, match="n_resamples must be positive"):
        bootstrap_ci(
            np.array([0.1, 0.2, 0.3]),
            n_resamples=bad_n,
            confidence=0.95,
            seed=789685,
        )


@pytest.mark.parametrize("bad_conf", [0.0, 1.0, -0.05, 1.05, 100.0])
def test_bootstrap_ci_rejects_confidence_out_of_range(bad_conf: float) -> None:
    """``confidence`` outside (0, 1) raises ValueError.

    The prior implementation passed out-of-range percentiles to
    np.percentile which clamps to [0, 100] -- silently returning the
    min/max of `means` rather than the intended CI. Fail closed.
    """
    with pytest.raises(ValueError, match=r"confidence must be in \(0, 1\)"):
        bootstrap_ci(
            np.array([0.1, 0.2, 0.3]),
            n_resamples=10,
            confidence=bad_conf,
            seed=789685,
        )


# ---------------------------------------------------------------------------
# Happy-path regression: guards must not break legitimate calls
# ---------------------------------------------------------------------------


def test_bootstrap_ci_happy_path_after_guards() -> None:
    """The guards do not interfere with normal usage.

    Pins the contract that valid inputs continue to produce
    well-shaped output: mean and CI bounds are finite floats with
    ci_lower <= mean <= ci_upper. The numeric assertions are
    seed-fixed for reproducibility.
    """
    values = np.array([0.0, 0.0, 1.0, 1.0, 0.5, 0.5, -0.5, -0.5], dtype=np.float64)
    mean, lo, hi = bootstrap_ci(
        values, n_resamples=200, confidence=0.95, seed=789685
    )
    assert np.isfinite(mean)
    assert np.isfinite(lo)
    assert np.isfinite(hi)
    assert lo <= mean <= hi
    # values.mean() is the seed-independent population mean
    assert mean == pytest.approx(float(values.mean()), abs=1e-12)


def test_bootstrap_ci_empty_input_returns_zeros() -> None:
    """Empty 1-D input continues to return (0, 0, 0).

    This is the existing behavior pre-IN-02. The empty case is
    distinct from `n_resamples <= 0`: an empty `values` array is a
    legitimate degenerate situation (e.g., zero questions match the
    filter), whereas `n_resamples=0` is always a user-input error.
    """
    out = bootstrap_ci(
        np.array([], dtype=np.float64),
        n_resamples=10,
        confidence=0.95,
        seed=789685,
    )
    assert out == (0.0, 0.0, 0.0)


def test_bootstrap_ci_accepts_list_input() -> None:
    """Plain Python list inputs are coerced via np.asarray.

    Confirms the asarray-based guard is permissive about input
    container type while still strict about dimensionality. This
    keeps the public API ergonomic.
    """
    mean, lo, hi = bootstrap_ci(
        [0.0, 1.0, 0.5, 0.5],
        n_resamples=50,
        confidence=0.95,
        seed=789685,
    )
    assert np.isfinite(mean)
    assert lo <= mean <= hi
