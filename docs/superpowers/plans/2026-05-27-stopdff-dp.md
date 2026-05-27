# Finite-Horizon DP StopDFF Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a finite-horizon dynamic-programming StopDFF pipeline (`scripts/compute_stopdff_dp.py`) that computes continuation-value stopping policies over calibrated prefix trajectories, supplements the existing myopic-threshold StopDFF diagnostic, and exports JSON/MD/LaTeX artifacts.

**Architecture:**
A new producer script writes `paper_exports/stopdff_dp.{json,md,tex}` from existing `data/processed/{mc,val,test}_dataset.json` + `paper_exports/calibration.json` artifacts. An internal `stopdff_dp/` Python package (under `scripts/`) holds the adapter that normalizes MC/QA artifacts into a (subject, item_id, prefix_idx, format, split, p_raw, p_calibrated, correct, top_answer, gold, category, option_set_id) dataframe, the DP solver, three continuation estimators (`oracle_trajectory`, `empirical_bucket`, `pooled_empirical`), the reward-schedule registry, and the coverage/ceiling diagnostics. The script never touches test data when fitting calibrators or continuation buckets — fit-split (default `val`) is split-safe and enforced at module boundary. The audit-card generator gains an opt-in `--include-dp-stopdff` flag that appends a DP row without removing the diagnostic StopDFF row.

**Tech Stack:** Python 3.12, NumPy, scikit-learn (already a dep), pandas (already a dep via `requirements.txt`), pytest for tests, `sentence-transformers` SBERT (already used by diagnostic StopDFF). No new third-party dependencies.

---

## File Structure

| Path | Role | Status |
| --- | --- | --- |
| `scripts/compute_stopdff_dp.py` | CLI entry point — argparse, orchestrates loaders, DP, writers, gates | **Create** |
| `scripts/stopdff_dp/__init__.py` | Package marker (one-line `__all__`) | **Create** |
| `scripts/stopdff_dp/adapter.py` | Adapter: existing artifacts → normalized dataframe | **Create** |
| `scripts/stopdff_dp/rewards.py` | Reward schedule registry (`acf_flat`, `power_mark`, `wait_cost_small`, `strict_wrong`) | **Create** |
| `scripts/stopdff_dp/dp_solver.py` | Bellman backward induction, stop-policy extraction | **Create** |
| `scripts/stopdff_dp/continuation.py` | Three continuation estimators + fallback ladder | **Create** |
| `scripts/stopdff_dp/diagnostics.py` | Coverage diagnostics + ceiling-effect detectors | **Create** |
| `scripts/stopdff_dp/writers.py` | JSON/MD/LaTeX writers with provenance | **Create** |
| `tests/test_stopdff_dp.py` | Pytest module covering all 6 unit-test requirements | **Create** |
| `scripts/make_audit_card.py` | Add opt-in `--include-dp-stopdff` flag; do not change existing row | **Modify** |
| `scripts/compute_stopdff.py` | **Untouched.** Diagnostic myopic StopDFF stays exactly as is. | **Untouched** |

Splitting `scripts/compute_stopdff_dp.py` into a thin CLI + a `stopdff_dp/` package keeps each module under ~300 lines, makes unit-tests easy (no CLI argparse boilerplate in test imports), and mirrors how `scripts/_common.py` + `scripts/_audit_gates.py` are already factored.

---

## Conventions and Invariants (must hold across all tasks)

1. **Sign convention.** `StopDFF_{sj}^{(f=MC)} = tau_{sj}^{*,MC,DP} - tau_{sj}^{*,QA,DP}`. Negative ⇒ MC stops earlier. Tests must lock this.
2. **Split safety.** Continuation buckets are fit ONLY on `--fit-split` (default `val`). Evaluation/stopping decisions ONLY consume `--split` (default `test`). The adapter validates `fit_split != split`.
3. **`metric_type` field.** Output JSON sets `metadata.metric_type = "finite_horizon_dp"` (vs `"diagnostic_only"` from `compute_stopdff.py`). Audit card uses this to distinguish rows.
4. **Coverage WARN.** When `>5%` of `(item, prefix, format)` cells fall through to `pooled_empirical` OR `>1%` are missing, `gate_verdict = "warn"`, not `"pass"`. Threshold values declared in the script and recorded in metadata.
5. **`oracle_trajectory` non-confirmatory.** The estimator name is recorded in `metadata.continuation_estimator`, and the writer prints `"WARNING: oracle_trajectory is upper-bound diagnostic"` to stderr AND embeds `"confirmatory": false` in the JSON output when used.
6. **No third-party deps added.** Use only what is already in `requirements.txt` / `pyproject.toml`.
7. **Provenance.** Use `scripts._common.build_generation_provenance` for the `metadata.generation` block (matches existing artifacts).
8. **CSLI semantics unchanged.** Never touch `compute_csli.py` or `paper_exports/csli.json` schema.
9. **`subject/model s`.** In this codebase, the natural axis for `s` is the likelihood model name. For v1 we keep model fixed at `"sbert"` (matches diagnostic StopDFF). The dataframe column is `subject` and stores `f"{model_name}:{category}"` to expose both axes for bucketing without exploding the loop.
10. **Reward schedules.** Implement at minimum:
    - `acf_flat`: `R_correct=10, R_wrong=-5, power=0.0, c_wait=0.0`
    - `power_mark`: `R_early_correct=15, R_late_correct=10, R_wrong=-5, power_split=0.5, c_wait=0.0`
    - `wait_cost_small`: same as `power_mark` but `c_wait=0.05`
    - `strict_wrong`: same as `power_mark` but `R_wrong=-10`
    The "early/late" split for `power_mark` is at 50% of the prefix length (matches the spec's intent: "early correct" rewards committing before the final reveal).

---

## Self-Review Checklist (run after writing the plan, before handing to executor)

Mapped here so the executor can verify acceptance:
- [ ] `pytest tests/test_stopdff_dp.py -q` passes — covered by Task 11
- [ ] Smoke run completes <5 min — Task 10 smoke mode trims VAL/TEST to 30 questions each
- [ ] Full run writes JSON/MD/TeX — Task 10 writes all three; Task 7 covers writers
- [ ] `metric_type = "finite_horizon_dp"` distinct from `"myopic_threshold"` — Task 7 (writer) + Task 11 (test)
- [ ] Audit card has DP row without replacing diagnostic — Task 12
- [ ] Coverage too sparse → WARN — Task 5 (diagnostics) + Task 11 (test)
- [ ] No test data used to fit continuation buckets — Task 4 (continuation estimators) + Task 11 (leakage test)
- [ ] `oracle_trajectory` not confirmatory — Task 4 + Task 7
- [ ] CSLI semantics not changed — `compute_csli.py` left untouched
- [ ] Existing myopic diagnostic preserved — `compute_stopdff.py` left untouched

---

### Task 1: Package scaffolding and shared types

**Files:**
- Create: `scripts/stopdff_dp/__init__.py`
- Create: `scripts/stopdff_dp/types.py`
- Test: (no test for this task — pure scaffolding)

- [ ] **Step 1: Create the package marker**

Write `scripts/stopdff_dp/__init__.py`:

```python
"""Finite-horizon dynamic-programming StopDFF helpers.

Imported by ``scripts/compute_stopdff_dp.py``. Kept in a sibling
package so unit tests can target individual modules without paying
the CLI/argparse import cost of the producer script.
"""

from __future__ import annotations

__all__ = [
    "adapter",
    "continuation",
    "diagnostics",
    "dp_solver",
    "rewards",
    "writers",
    "types",
]
```

- [ ] **Step 2: Define the shared dataclasses and column constants**

Write `scripts/stopdff_dp/types.py`:

```python
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
    """Parameters for the answer-utility function A_t(p) = R*p - c_wrong*(1-p).

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
```

- [ ] **Step 3: Commit**

```bash
git add scripts/stopdff_dp/__init__.py scripts/stopdff_dp/types.py
git commit -m "feat(stopdff-dp): scaffold stopdff_dp package with shared types"
```

---

### Task 2: Reward schedule registry

**Files:**
- Create: `scripts/stopdff_dp/rewards.py`
- Test: `tests/test_stopdff_dp.py` (one schedule-registry test)

- [ ] **Step 1: Write the failing test (registry returns four named schedules)**

Append to `tests/test_stopdff_dp.py` (create the file if it does not exist with a top-level module docstring first):

```python
"""Unit tests for the DP StopDFF pipeline (scripts/compute_stopdff_dp.py)."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from scripts.stopdff_dp import (
    rewards as rewards_module,
)
from scripts.stopdff_dp.types import RewardSchedule


def test_reward_registry_includes_all_required_schedules() -> None:
    """The four schedules named in the spec must all be in the registry."""
    registry = rewards_module.REWARD_REGISTRY
    required = {"acf_flat", "power_mark", "wait_cost_small", "strict_wrong"}
    assert required.issubset(registry.keys())
    for name in required:
        schedule = registry[name]
        assert isinstance(schedule, RewardSchedule)
        assert schedule.name == name


def test_acf_flat_has_zero_wait_cost_and_no_power_split() -> None:
    schedule = rewards_module.REWARD_REGISTRY["acf_flat"]
    assert schedule.c_wait == 0.0
    # No power_split means early and late reward must be equal.
    assert schedule.r_correct_early == schedule.r_correct_late == 10.0
    assert schedule.r_wrong == -5.0


def test_wait_cost_small_has_nonzero_c_wait() -> None:
    schedule = rewards_module.REWARD_REGISTRY["wait_cost_small"]
    assert schedule.c_wait == 0.05
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_stopdff_dp.py::test_reward_registry_includes_all_required_schedules -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.stopdff_dp.rewards'`

- [ ] **Step 3: Write the registry**

Write `scripts/stopdff_dp/rewards.py`:

```python
"""Reward schedule registry for finite-horizon DP StopDFF.

Each schedule defines answer utility A_t(p) = R(t)*p - c_wrong*(1-p)
and a per-step continuation cost c_wait. The CLI's
``--reward-schedule`` flag selects one by name.
"""

from __future__ import annotations

from .types import RewardSchedule

REWARD_REGISTRY: dict[str, RewardSchedule] = {
    "acf_flat": RewardSchedule(
        name="acf_flat",
        r_correct_early=10.0,
        r_correct_late=10.0,
        r_wrong=-5.0,
        power_split=1.0,  # Never trips — early == late
        c_wait=0.0,
        description=(
            "Flat reward: correct=10, wrong=-5, no power split, c_wait=0."
        ),
    ),
    "power_mark": RewardSchedule(
        name="power_mark",
        r_correct_early=15.0,
        r_correct_late=10.0,
        r_wrong=-5.0,
        power_split=0.5,
        c_wait=0.0,
        description=(
            "Power-mark schedule: early correct=15 (prefix_fraction<0.5), "
            "late correct=10, wrong=-5, c_wait=0."
        ),
    ),
    "wait_cost_small": RewardSchedule(
        name="wait_cost_small",
        r_correct_early=15.0,
        r_correct_late=10.0,
        r_wrong=-5.0,
        power_split=0.5,
        c_wait=0.05,
        description=(
            "Power-mark with small wait cost (c_wait=0.05)."
        ),
    ),
    "strict_wrong": RewardSchedule(
        name="strict_wrong",
        r_correct_early=15.0,
        r_correct_late=10.0,
        r_wrong=-10.0,
        power_split=0.5,
        c_wait=0.0,
        description=(
            "Power-mark with strict wrong penalty (R_wrong=-10)."
        ),
    ),
}


def get_schedule(name: str) -> RewardSchedule:
    """Look up a schedule by name; raise ValueError on unknown name."""
    try:
        return REWARD_REGISTRY[name]
    except KeyError as exc:
        valid = ", ".join(sorted(REWARD_REGISTRY))
        raise ValueError(
            f"Unknown reward schedule {name!r}. Valid choices: {valid}."
        ) from exc


def answer_utility(p: float, prefix_fraction: float, schedule: RewardSchedule) -> float:
    """A_t(p) = R(t) * p + R_wrong * (1 - p).

    Parameters
    ----------
    p : float
        Calibrated probability that the top answer is correct.
    prefix_fraction : float
        Position of this prefix as a fraction of the full question.
    schedule : RewardSchedule
        Reward schedule to use.
    """
    r_correct = schedule.r_correct(prefix_fraction)
    return r_correct * p + schedule.r_wrong * (1.0 - p)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_stopdff_dp.py -v -k reward`
Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/stopdff_dp/rewards.py tests/test_stopdff_dp.py
git commit -m "feat(stopdff-dp): reward schedule registry with four required schedules"
```

---

### Task 3: DP solver (Bellman backward induction)

**Files:**
- Create: `scripts/stopdff_dp/dp_solver.py`
- Test: `tests/test_stopdff_dp.py` (DP-myopic equivalence, DP-wait, sign-convention tests)

- [ ] **Step 1: Write the failing test set**

Append to `tests/test_stopdff_dp.py`:

```python
from scripts.stopdff_dp import dp_solver
from scripts.stopdff_dp.rewards import REWARD_REGISTRY


def _zero_continuation(*_args, **_kwargs) -> float:
    return 0.0


def test_dp_equals_myopic_when_continuation_is_zero() -> None:
    """If E[V_{t+1}] == 0 everywhere, DP = stop at first prefix where A_t(p)>0."""
    schedule = REWARD_REGISTRY["acf_flat"]
    # A_t(p) = 10p - 5(1-p) = 15p - 5; zero at p = 1/3.
    # So with p_trajectory [0.2, 0.4, 0.9], myopic stops at idx=1 (p=0.4).
    p_trajectory = [0.2, 0.4, 0.9]
    prefix_fractions = [0.1, 0.5, 0.9]

    trace = dp_solver.solve_trajectory(
        p_trajectory=p_trajectory,
        prefix_fractions=prefix_fractions,
        schedule=schedule,
        continuation_fn=_zero_continuation,
    )

    assert trace.stop_step == 1
    # All continuation values must be 0.0 under this estimator.
    assert all(cv == 0.0 for cv in trace.continuation_values)


def test_dp_waits_when_future_value_exceeds_current_answer_utility() -> None:
    """With a high continuation value, DP should defer stopping."""
    schedule = REWARD_REGISTRY["acf_flat"]
    p_trajectory = [0.4, 0.95]
    prefix_fractions = [0.5, 1.0]

    # A_0(0.4) = 15*0.4 - 5 = 1.0; A_1(0.95) = 15*0.95 - 5 = 9.25.
    # Force the DP to see continuation = 8.0 at t=0, which beats A_0=1.0
    # plus -c_wait=0, so the agent should wait at t=0 and stop at t=1.
    def continuation_fn(t: int, **_kw: object) -> float:
        return 8.0 if t == 0 else 0.0

    trace = dp_solver.solve_trajectory(
        p_trajectory=p_trajectory,
        prefix_fractions=prefix_fractions,
        schedule=schedule,
        continuation_fn=continuation_fn,
    )
    assert trace.stop_step == 1


def test_dp_stops_earlier_when_mc_probabilities_uniformly_shifted_upward() -> None:
    """If we add delta to every p_t, DP stop step must be <= the lower-p version.

    StopDFF sign convention: stop_step_MC < stop_step_QA when MC raises p.
    """
    schedule = REWARD_REGISTRY["acf_flat"]
    qa_trajectory = [0.2, 0.3, 0.4, 0.6]
    mc_trajectory = [min(1.0, p + 0.2) for p in qa_trajectory]
    prefix_fractions = [0.2, 0.4, 0.6, 0.8]

    qa_trace = dp_solver.solve_trajectory(
        p_trajectory=qa_trajectory,
        prefix_fractions=prefix_fractions,
        schedule=schedule,
        continuation_fn=_zero_continuation,
    )
    mc_trace = dp_solver.solve_trajectory(
        p_trajectory=mc_trajectory,
        prefix_fractions=prefix_fractions,
        schedule=schedule,
        continuation_fn=_zero_continuation,
    )

    assert mc_trace.stop_step <= qa_trace.stop_step
    # StopDFF (MC - QA) must be <= 0 in this construction.
    assert mc_trace.stop_step - qa_trace.stop_step <= 0


def test_dp_horizon_terminal_uses_max_of_answer_or_zero() -> None:
    """V_T = max(A_T(p_T), 0). When A_T<0 we should never buzz (stop=T)."""
    schedule = REWARD_REGISTRY["acf_flat"]
    p_trajectory = [0.05, 0.10]
    prefix_fractions = [0.5, 1.0]
    trace = dp_solver.solve_trajectory(
        p_trajectory=p_trajectory,
        prefix_fractions=prefix_fractions,
        schedule=schedule,
        continuation_fn=_zero_continuation,
    )
    # Both A_t < 0, so optimal action is to never stop; we encode that as
    # stop_step == len(p_trajectory) (i.e. one past the last index).
    assert trace.stop_step == len(p_trajectory)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_stopdff_dp.py -v -k "dp_"`
Expected: 4 FAIL with `ModuleNotFoundError: No module named 'scripts.stopdff_dp.dp_solver'`

- [ ] **Step 3: Write the DP solver**

Write `scripts/stopdff_dp/dp_solver.py`:

```python
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
        if answer_utilities[t] >= wait_value:
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
```

- [ ] **Step 4: Run DP tests to verify pass**

Run: `pytest tests/test_stopdff_dp.py -v -k "dp_"`
Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/stopdff_dp/dp_solver.py tests/test_stopdff_dp.py
git commit -m "feat(stopdff-dp): Bellman backward induction with three test invariants"
```

---

### Task 4: Continuation estimators (oracle, empirical_bucket, pooled_empirical)

**Files:**
- Create: `scripts/stopdff_dp/continuation.py`
- Test: `tests/test_stopdff_dp.py` (leakage, fallback ladder, oracle flag)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_stopdff_dp.py`:

```python
import pandas as pd

from scripts.stopdff_dp import continuation as cont_module


def _make_df(rows: list[dict]) -> pd.DataFrame:
    """Tiny helper to build a normalised adapter dataframe in tests."""
    from scripts.stopdff_dp.types import ADAPTER_COLUMNS
    df = pd.DataFrame(rows)
    for col in ADAPTER_COLUMNS:
        if col not in df.columns:
            df[col] = None
    return df[list(ADAPTER_COLUMNS)]


def test_empirical_bucket_fitter_uses_only_fit_split_rows() -> None:
    """Continuation buckets must not see test-split rows during fit.

    The leakage check is enforced at the API boundary: passing a
    dataframe whose split column contains the eval split raises.
    """
    rows = [
        {"subject": "sbert:Lit", "item_id": "q1", "prefix_idx": 0, "format": "MC",
         "split": "test", "p_raw": 0.1, "p_calibrated": 0.2, "correct": 0,
         "top_answer": "a", "gold": "a", "category": "Lit", "option_set_id": "s1"},
    ]
    df_test = _make_df(rows)
    with pytest.raises(ValueError, match="leakage|fit on test"):
        cont_module.EmpiricalBucketEstimator.fit(
            fit_df=df_test,
            fit_split_name="val",
        )


def test_empirical_bucket_estimator_returns_pooled_when_bucket_sparse() -> None:
    """When the exact bucket has <3 trajectories, fallback ladder kicks in."""
    # Build a minimal val-split frame with enough rows in the pooled
    # (drop-entropy_bin) bucket but only 1 row in the most-specific bucket.
    rows = []
    # 1 row in (early, MC, sbert:Lit, p_bin=0.1, ent_bin=0): the "exact" bucket
    rows.append({
        "subject": "sbert:Lit", "item_id": "q1", "prefix_idx": 0,
        "format": "MC", "split": "val",
        "p_raw": 0.1, "p_calibrated": 0.10, "correct": 0,
        "top_answer": "a", "gold": "b", "category": "Lit",
        "option_set_id": "s1",
    })
    # 5 rows in (early, MC, sbert:Lit, p_bin=0.1) regardless of ent_bin.
    for i in range(5):
        rows.append({
            "subject": "sbert:Lit", "item_id": f"q{i+10}", "prefix_idx": 0,
            "format": "MC", "split": "val",
            "p_raw": 0.1, "p_calibrated": 0.12, "correct": (i % 2),
            "top_answer": "a", "gold": "b", "category": "Lit",
            "option_set_id": f"s{i}",
        })
    df_val = _make_df(rows)

    estimator = cont_module.EmpiricalBucketEstimator.fit(
        fit_df=df_val,
        fit_split_name="val",
        min_bucket_size=3,
    )

    tag = estimator.last_coverage_tag_for(
        prefix_bucket="early",
        fmt="MC",
        subject_bucket="sbert:Lit",
        p_bin=0,
        entropy_bin=0,
    )
    # The most-specific bucket has 1 row, so we drop entropy_bin and use 5.
    assert tag == "pooled"


def test_oracle_trajectory_estimator_flags_non_confirmatory() -> None:
    estimator = cont_module.OracleTrajectoryEstimator()
    assert estimator.confirmatory is False


def test_pooled_empirical_fallback_ladder_documented() -> None:
    """The fallback ladder must be a fixed, declared sequence."""
    ladder = cont_module.FALLBACK_LADDER
    # Must be a tuple of tuples so it cannot be mutated at runtime.
    assert isinstance(ladder, tuple)
    # The first rung is the most-specific bucket; the last is the most-pooled.
    assert ladder[0] == (
        "prefix_bucket", "format", "subject_bucket", "p_bin", "entropy_bin",
    )
    assert "format" in ladder[-1]
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `pytest tests/test_stopdff_dp.py -v -k "empirical_bucket or oracle or pooled_empirical"`
Expected: 4 FAIL with `ModuleNotFoundError: No module named 'scripts.stopdff_dp.continuation'`

- [ ] **Step 3: Write the continuation estimators**

Write `scripts/stopdff_dp/continuation.py`:

```python
"""Continuation-value estimators for finite-horizon DP StopDFF.

Three implementations are exposed:

* ``OracleTrajectoryEstimator`` — upper-bound diagnostic using each
  realized held-out trajectory's next prefix probability. Marked
  non-confirmatory; the writer warns and records the flag.
* ``EmpiricalBucketEstimator`` — primary confirmatory estimator. Fits
  E[V_{t+1} | prefix_bucket, format, subject_bucket, p_bin, entropy_bin]
  on the fit split (default ``val``) and looks up at decision time.
  Falls back along the FALLBACK_LADDER when a bucket is sparse.
* ``PooledEmpiricalEstimator`` — convenience facade for callers who want
  to force the fallback ladder to start at the top rung. Internally
  delegates to ``EmpiricalBucketEstimator``.

All three guard against test-split leakage at fit time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .types import ADAPTER_COLUMNS, assert_columns

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


@dataclass
class OracleTrajectoryEstimator:
    """Upper-bound diagnostic using realized next-step calibrated p.

    NON-CONFIRMATORY. Reports each realized p_{t+1} on the held-out
    trajectory as the continuation expectation E[V_{t+1} | h_t]. This
    leaks the future to the present and is intended only as an upper
    bound on the realisable DP value.
    """

    confirmatory: bool = False

    def fit(self, *_args, **_kwargs) -> "OracleTrajectoryEstimator":
        return self

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
        df["prefix_bucket"] = df["p_raw"].astype(float).map(lambda _: None)
        # Prefix bucket is computed from prefix fraction; the adapter
        # writes prefix fraction implicitly via prefix_idx/T below, so
        # we recompute it here for fit-time symmetry with the lookup.
        df["prefix_fraction"] = df.groupby(["item_id", "format"]).apply(
            lambda g: pd.Series(
                np.linspace(
                    1.0 / max(1, len(g)), 1.0, num=len(g), endpoint=True
                ),
                index=g.index,
            )
        ).reset_index(level=[0, 1], drop=True)
        df["prefix_bucket"] = df["prefix_fraction"].map(_assign_prefix_bucket)
        df["subject_bucket"] = df["subject"]
        df["p_bin"] = df["p_calibrated"].map(_assign_p_bin)
        df["entropy_bin"] = df["p_calibrated"].map(_assign_entropy_bin)

        bucket_means: dict[tuple, float] = {}
        bucket_counts: dict[tuple, int] = {}
        non_terminal = df.dropna(subset=["v_next"])
        for rung in FALLBACK_LADDER:
            grouped = non_terminal.groupby(list(rung))["v_next"]
            for key, mean_value in grouped.mean().items():
                if not isinstance(key, tuple):
                    key = (key,)
                bucket_means[(rung, *key)] = float(mean_value)
                bucket_counts[(rung, *key)] = int(grouped.get_group(key).count())

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
```

- [ ] **Step 4: Run continuation tests to verify pass**

Run: `pytest tests/test_stopdff_dp.py -v -k "empirical_bucket or oracle or pooled_empirical"`
Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/stopdff_dp/continuation.py tests/test_stopdff_dp.py
git commit -m "feat(stopdff-dp): oracle/empirical/pooled continuation estimators with leakage guard"
```

---

### Task 5: Coverage + ceiling-effect diagnostics

**Files:**
- Create: `scripts/stopdff_dp/diagnostics.py`
- Test: `tests/test_stopdff_dp.py` (coverage WARN, ceiling-detection unit tests)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_stopdff_dp.py`:

```python
from scripts.stopdff_dp import diagnostics as diag_module
from scripts.stopdff_dp.types import DPTrace


def _trace(stop_step: int, T: int, tags: list[str] | None = None) -> DPTrace:
    return DPTrace(
        item_id="q",
        fmt="MC",
        stop_step=stop_step,
        values=[0.0] * T,
        answer_utilities=[0.0] * T,
        continuation_values=[0.0] * T,
        coverage_tags=tags or ["exact"] * T,
    )


def test_coverage_warn_when_more_than_5pct_pooled() -> None:
    traces = [_trace(stop_step=2, T=3, tags=["pooled", "pooled", "pooled"])]
    summary = diag_module.summarize_coverage(traces)
    assert summary["fraction_pooled"] == 1.0
    assert summary["verdict"] == "warn"


def test_coverage_pass_when_fully_exact() -> None:
    traces = [_trace(stop_step=2, T=3, tags=["exact", "exact", "exact"])]
    summary = diag_module.summarize_coverage(traces)
    assert summary["fraction_pooled"] == 0.0
    assert summary["fraction_exact"] == 1.0
    assert summary["verdict"] == "pass"


def test_ceiling_all_stop_at_final_prefix() -> None:
    mc_traces = [_trace(stop_step=2, T=3), _trace(stop_step=2, T=3)]
    qa_traces = [_trace(stop_step=2, T=3), _trace(stop_step=2, T=3)]
    flags = diag_module.detect_ceiling_effects(mc_traces, qa_traces)
    assert flags["all_stop_at_final_prefix"] is True
    assert flags["all_stop_at_first_prefix"] is False


def test_ceiling_no_cross_format_variance() -> None:
    mc_traces = [_trace(stop_step=1, T=3), _trace(stop_step=2, T=3)]
    qa_traces = [_trace(stop_step=1, T=3), _trace(stop_step=2, T=3)]
    flags = diag_module.detect_ceiling_effects(mc_traces, qa_traces)
    assert flags["no_cross_format_stopping_variance"] is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_stopdff_dp.py -v -k "coverage or ceiling"`
Expected: 4 FAIL.

- [ ] **Step 3: Write the diagnostics module**

Write `scripts/stopdff_dp/diagnostics.py`:

```python
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
```

- [ ] **Step 4: Run diagnostics tests to verify pass**

Run: `pytest tests/test_stopdff_dp.py -v -k "coverage or ceiling"`
Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/stopdff_dp/diagnostics.py tests/test_stopdff_dp.py
git commit -m "feat(stopdff-dp): coverage + ceiling diagnostics with WARN thresholds"
```

---

### Task 6: Adapter (artifacts → normalized dataframe)

**Files:**
- Create: `scripts/stopdff_dp/adapter.py`
- Test: `tests/test_stopdff_dp.py` (column shape, gold/correct alignment, fit-eval split separation)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_stopdff_dp.py`:

```python
from scripts.stopdff_dp import adapter as adapter_module


def _fake_mc_question(qid: str, gold_text: str = "George Washington") -> dict:
    """Synthesize the minimum MC question dict the adapter expects."""
    return {
        "qid": qid,
        "question": "Who was the first president of the United States?",
        "tokens": ["Who", "was", "the", "first", "president"],
        "answer_primary": gold_text,
        "clean_answers": [gold_text],
        "run_indices": [0, 4],
        "human_buzz_positions": [],
        "category": "History",
        "cumulative_prefixes": ["Who", "Who was the first president"],
        "options": [gold_text, "Thomas Jefferson", "John Adams", "Benjamin Franklin"],
        "gold_index": 0,
        "option_profiles": [
            "president", "vice", "second", "diplomat",
        ],
        "option_answer_primary": [
            gold_text, "Thomas Jefferson", "John Adams", "Benjamin Franklin",
        ],
        "distractor_strategy": "test",
    }


def test_adapter_produces_canonical_columns(monkeypatch) -> None:
    """The adapter must yield a dataframe with the canonical column set."""
    fake_questions = [_fake_mc_question("q1")]
    df = adapter_module.build_dataframe(
        mc_questions=fake_questions,
        target_qids={"q1"},
        split_name="val",
        calibration_path=None,  # Use the identity-calibration test mode
        identity_calibration=True,
    )
    from scripts.stopdff_dp.types import ADAPTER_COLUMNS
    assert list(df.columns) == list(ADAPTER_COLUMNS)
    # Two rows per (qid, prefix) per format -> 2 prefixes * 2 formats = 4.
    assert len(df) == 4


def test_adapter_fit_eval_split_separation_raises_on_overlap() -> None:
    """Passing the same split for fit and eval should raise."""
    with pytest.raises(ValueError, match="fit and eval split must differ"):
        adapter_module.validate_split_separation(
            fit_split="test", eval_split="test"
        )
```

- [ ] **Step 2: Run new tests to verify they fail**

Run: `pytest tests/test_stopdff_dp.py -v -k "adapter"`
Expected: 2 FAIL.

- [ ] **Step 3: Write the adapter**

Write `scripts/stopdff_dp/adapter.py`:

```python
"""Normalize existing qanta-buzzer artifacts into the DP-StopDFF dataframe.

Adapter columns (frozen by ``types.ADAPTER_COLUMNS``):
    subject, item_id, prefix_idx, format, split,
    p_raw, p_calibrated, correct, top_answer, gold,
    category, option_set_id

For each MC question and each cumulative prefix, the adapter emits two
rows: one for the MC format (max cosine similarity over the K=4
options, then Platt calibration per prefix bucket) and one for the QA
format (cosine similarity to ``answer_primary``, then Platt
calibration). Calibration coefficients are loaded from
``paper_exports/calibration.json``.

The adapter never touches the test split when fitting calibrators or
continuation buckets — those are caller responsibilities and the
adapter's ``split_name`` parameter only stamps the resulting rows.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .types import ADAPTER_COLUMNS, FORMATS

_SBERT_MODEL = None


def validate_split_separation(*, fit_split: str, eval_split: str) -> None:
    if fit_split == eval_split:
        raise ValueError(
            "fit and eval split must differ to avoid leakage; "
            f"got fit_split={fit_split!r}, eval_split={eval_split!r}."
        )


def _get_sbert_model():
    global _SBERT_MODEL
    if _SBERT_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _SBERT_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _SBERT_MODEL


def _platt(z: float, coef: float, intercept: float) -> float:
    lin = coef * z + intercept
    lin = max(-500.0, min(500.0, lin))
    return 1.0 / (1.0 + math.exp(-lin))


def _assign_bucket(frac: float) -> str:
    if frac < 0.33:
        return "early"
    if frac < 0.66:
        return "mid"
    return "late"


def _load_platt_params(calibration_path: Path) -> dict[str, tuple[float, float]]:
    """Reuse the loader logic from compute_stopdff (kept inline to avoid coupling)."""
    import json
    with open(calibration_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    out = {}
    for bucket_name, bucket_data in data["per_bucket"].items():
        coef = bucket_data["platt_coef"]
        intercept = bucket_data["platt_intercept"]
        if coef is None or intercept is None:
            probability = float(
                bucket_data.get("platt_constant_probability", 0.0)
            )
            coef = 0.0
            if probability <= 0.0:
                intercept = -500.0
            elif probability >= 1.0:
                intercept = 500.0
            else:
                intercept = math.log(probability / (1.0 - probability))
        out[bucket_name] = (coef, intercept)
    return out


def _score_question(
    question: dict,
    platt_params: dict[str, tuple[float, float]] | None,
    identity_calibration: bool,
) -> list[dict]:
    """Emit per-(prefix, format) rows for one MC question."""
    rows: list[dict] = []
    qid = str(question["qid"])
    full_q = question["question"]
    prefixes = question["cumulative_prefixes"]
    options = question["options"]
    gold_index = int(question["gold_index"])
    gold_text = options[gold_index]
    category = question.get("category", "")
    option_set_id = f"{qid}:K{len(options)}"
    full_len = max(1, len(full_q))

    if identity_calibration:
        # Test-only branch: replace SBERT cosine with a deterministic
        # synthetic signal so unit tests do not need the model.
        for t, prefix in enumerate(prefixes):
            prefix_fraction = len(prefix) / full_len
            p_mc = max(0.0, min(1.0, 0.3 + 0.2 * t))
            p_qa = max(0.0, min(1.0, 0.2 + 0.15 * t))
            rows.append({
                "subject": f"sbert:{category}",
                "item_id": qid,
                "prefix_idx": t,
                "format": "MC",
                "split": None,  # caller stamps split
                "p_raw": p_mc,
                "p_calibrated": p_mc,
                "correct": 1,
                "top_answer": gold_text,
                "gold": gold_text,
                "category": category,
                "option_set_id": option_set_id,
            })
            rows.append({
                "subject": f"sbert:{category}",
                "item_id": qid,
                "prefix_idx": t,
                "format": "QA",
                "split": None,
                "p_raw": p_qa,
                "p_calibrated": p_qa,
                "correct": 1,
                "top_answer": gold_text,
                "gold": gold_text,
                "category": category,
                "option_set_id": option_set_id,
            })
        return rows

    from sklearn.metrics.pairwise import cosine_similarity
    model = _get_sbert_model()
    option_embs = model.encode(options, convert_to_numpy=True)
    answer_emb = model.encode(
        [question["answer_primary"]], convert_to_numpy=True
    )

    for t, prefix in enumerate(prefixes):
        prefix_fraction = len(prefix) / full_len
        bucket = _assign_bucket(prefix_fraction)
        coef, intercept = (
            platt_params[bucket] if platt_params is not None else (1.0, 0.0)
        )
        prefix_emb = model.encode([prefix], convert_to_numpy=True)

        # MC: max similarity over options.
        mc_sims = cosine_similarity(prefix_emb, option_embs)[0]
        max_sim = float(np.max(mc_sims))
        predicted_idx = int(np.argmax(mc_sims))
        rows.append({
            "subject": f"sbert:{category}",
            "item_id": qid,
            "prefix_idx": t,
            "format": "MC",
            "split": None,
            "p_raw": max_sim,
            "p_calibrated": _platt(max_sim, coef, intercept),
            "correct": int(predicted_idx == gold_index),
            "top_answer": options[predicted_idx],
            "gold": gold_text,
            "category": category,
            "option_set_id": option_set_id,
        })

        # QA: similarity to answer_primary only.
        qa_sim = float(cosine_similarity(prefix_emb, answer_emb)[0][0])
        rows.append({
            "subject": f"sbert:{category}",
            "item_id": qid,
            "prefix_idx": t,
            "format": "QA",
            "split": None,
            "p_raw": qa_sim,
            "p_calibrated": _platt(qa_sim, coef, intercept),
            "correct": 1,  # QA "top answer" is always the gold by construction
            "top_answer": question["answer_primary"],
            "gold": gold_text,
            "category": category,
            "option_set_id": option_set_id,
        })

    return rows


def build_dataframe(
    *,
    mc_questions: Sequence[dict],
    target_qids: set[str],
    split_name: str,
    calibration_path: Path | None = None,
    identity_calibration: bool = False,
) -> pd.DataFrame:
    """Build the normalised dataframe for one split.

    Parameters
    ----------
    mc_questions : Sequence[dict]
        MC question dicts (output of ``iter_split_questions``).
    target_qids : set[str]
        The qid set defining the requested split (e.g. all val qids).
    split_name : str
        Stamped onto the ``split`` column of every produced row.
    calibration_path : Path, optional
        Path to ``calibration.json``. Required unless
        ``identity_calibration`` is True.
    identity_calibration : bool
        Skip the SBERT model and emit a deterministic synthetic signal.
        Test-only escape hatch.

    Returns
    -------
    pd.DataFrame
        Columns match ``types.ADAPTER_COLUMNS``.
    """
    if not identity_calibration:
        if calibration_path is None or not Path(calibration_path).exists():
            raise FileNotFoundError(
                "calibration_path must exist when identity_calibration=False; "
                f"got {calibration_path!r}."
            )
    platt_params = (
        None if identity_calibration
        else _load_platt_params(Path(calibration_path))
    )

    rows: list[dict] = []
    for q in mc_questions:
        if str(q["qid"]) not in target_qids:
            continue
        rows.extend(_score_question(q, platt_params, identity_calibration))

    df = pd.DataFrame(rows, columns=list(ADAPTER_COLUMNS))
    df["split"] = split_name
    return df[list(ADAPTER_COLUMNS)]
```

- [ ] **Step 4: Run adapter tests to verify pass**

Run: `pytest tests/test_stopdff_dp.py -v -k "adapter"`
Expected: 2 PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/stopdff_dp/adapter.py tests/test_stopdff_dp.py
git commit -m "feat(stopdff-dp): adapter normalises MC/QA artifacts into canonical dataframe"
```

---

### Task 7: Writers (JSON / Markdown / LaTeX)

**Files:**
- Create: `scripts/stopdff_dp/writers.py`
- Test: `tests/test_stopdff_dp.py` (writer round-trip — `metric_type` distinct from myopic)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_stopdff_dp.py`:

```python
import json
from scripts.stopdff_dp import writers as writers_module
from scripts.stopdff_dp.types import DPTrace


def test_writer_metric_type_is_finite_horizon_dp(tmp_path: Path) -> None:
    out_json = tmp_path / "stopdff_dp.json"
    out_md = tmp_path / "stopdff_dp.md"
    out_tex = tmp_path / "stopdff_dp_table.tex"
    mc_traces = [_trace(stop_step=1, T=3, tags=["exact", "exact", "exact"])]
    qa_traces = [_trace(stop_step=2, T=3, tags=["exact", "exact", "exact"])]
    payload = writers_module.assemble_payload(
        mc_traces=mc_traces,
        qa_traces=qa_traces,
        reward_schedule_name="acf_flat",
        continuation_estimator_name="empirical_bucket",
        fit_split="val",
        eval_split="test",
        coverage_summary={
            "n_cells": 6, "fraction_exact": 1.0, "fraction_pooled": 0.0,
            "fraction_missing": 0.0, "verdict": "pass", "reason": "ok",
        },
        ceiling_flags={
            "all_stop_at_first_prefix": False,
            "all_stop_at_final_prefix": False,
            "no_cross_format_stopping_variance": False,
            "n_trajectories": 1, "n_stopped_cells": 2, "n_never_stopped_cells": 0,
        },
        per_item_stopdff=[("q1", -1)],
        gate_verdict="pass",
        gate_verdict_reason="all_clean",
        confirmatory=True,
    )
    writers_module.write_json(out_json, payload)
    writers_module.write_markdown(out_md, payload)
    writers_module.write_latex(out_tex, payload)
    assert out_json.exists() and out_md.exists() and out_tex.exists()
    loaded = json.loads(out_json.read_text())
    assert loaded["metadata"]["metric_type"] == "finite_horizon_dp"
    assert loaded["metadata"]["stopping_policy"] == "finite_horizon_dp"
    assert "myopic" not in loaded["metadata"]["metric_type"]
```

- [ ] **Step 2: Run test to verify fail**

Run: `pytest tests/test_stopdff_dp.py -v -k "writer_metric_type"`
Expected: FAIL.

- [ ] **Step 3: Write the writers**

Write `scripts/stopdff_dp/writers.py`:

```python
"""JSON / Markdown / LaTeX writers for DP StopDFF artifacts."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import median, mean

from .types import DPTrace


def assemble_payload(
    *,
    mc_traces: list[DPTrace],
    qa_traces: list[DPTrace],
    reward_schedule_name: str,
    continuation_estimator_name: str,
    fit_split: str,
    eval_split: str,
    coverage_summary: dict,
    ceiling_flags: dict,
    per_item_stopdff: list[tuple[str, int]],
    gate_verdict: str,
    gate_verdict_reason: str,
    confirmatory: bool,
    generation: dict | None = None,
) -> dict:
    """Compose the JSON payload, matching the existing artifact style."""
    signed = [shift for _, shift in per_item_stopdff]
    abs_shifts = [abs(s) for s in signed]
    return {
        "stopdff_dp_signed_median": float(median(signed)) if signed else 0.0,
        "stopdff_dp_signed_mean": float(mean(signed)) if signed else 0.0,
        "stopdff_dp_abs_median": float(median(abs_shifts)) if abs_shifts else 0.0,
        "stopdff_dp_abs_mean": float(mean(abs_shifts)) if abs_shifts else 0.0,
        "n_items": len(per_item_stopdff),
        "direction_breakdown": {
            "mc_earlier": sum(1 for _, s in per_item_stopdff if s < 0),
            "qa_earlier": sum(1 for _, s in per_item_stopdff if s > 0),
            "same_step": sum(1 for _, s in per_item_stopdff if s == 0),
        },
        "coverage": coverage_summary,
        "ceiling_flags": ceiling_flags,
        "gate_verdict": gate_verdict,
        "gate_verdict_reason": gate_verdict_reason,
        "confirmatory": confirmatory,
        "metadata": {
            "metric_type": "finite_horizon_dp",
            "stopping_policy": "finite_horizon_dp",
            "reward_schedule": reward_schedule_name,
            "continuation_estimator": continuation_estimator_name,
            "fit_split": fit_split,
            "eval_split": eval_split,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "generation": generation,
        },
    }


def write_json(path: Path, payload: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path


def write_markdown(path: Path, payload: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    md = []
    md.append("# Finite-Horizon DP StopDFF")
    md.append("")
    md.append(
        f"**Metric type:** `{payload['metadata']['metric_type']}` — "
        f"confirmatory: `{payload['confirmatory']}`"
    )
    md.append("")
    md.append("| Field | Value |")
    md.append("|-------|-------|")
    md.append(
        f"| Reward schedule | {payload['metadata']['reward_schedule']} |"
    )
    md.append(
        f"| Continuation estimator | {payload['metadata']['continuation_estimator']} |"
    )
    md.append(f"| Fit split | {payload['metadata']['fit_split']} |")
    md.append(f"| Eval split | {payload['metadata']['eval_split']} |")
    md.append(f"| n_items | {payload['n_items']} |")
    md.append(
        f"| StopDFF signed median | {payload['stopdff_dp_signed_median']:.4f} |"
    )
    md.append(
        f"| StopDFF signed mean | {payload['stopdff_dp_signed_mean']:.4f} |"
    )
    md.append(f"| Gate verdict | {payload['gate_verdict']} |")
    md.append("")
    md.append("## Coverage")
    md.append("")
    cov = payload["coverage"]
    md.append(
        f"- exact={cov['fraction_exact']:.3f}, "
        f"pooled={cov['fraction_pooled']:.3f}, "
        f"missing={cov['fraction_missing']:.3f}; "
        f"verdict={cov['verdict']} ({cov['reason']})"
    )
    md.append("")
    md.append("## Ceiling diagnostics")
    md.append("")
    for k, v in payload["ceiling_flags"].items():
        md.append(f"- {k}: {v}")
    md.append("")
    if not payload["confirmatory"]:
        md.append(
            "> ⚠️ Non-confirmatory estimator in use — interpret as an "
            "upper-bound diagnostic only."
        )
    path.write_text("\n".join(md))
    return path


def write_latex(path: Path, payload: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "\\begin{tabular}{lr}",
        "\\toprule",
        "Metric & Value \\\\",
        "\\midrule",
        f"Signed median StopDFF & {payload['stopdff_dp_signed_median']:.4f} \\\\",
        f"Signed mean StopDFF & {payload['stopdff_dp_signed_mean']:.4f} \\\\",
        f"Abs median StopDFF & {payload['stopdff_dp_abs_median']:.4f} \\\\",
        f"$n_{{items}}$ & {payload['n_items']} \\\\",
        f"Coverage exact & {payload['coverage']['fraction_exact']:.3f} \\\\",
        f"Coverage pooled & {payload['coverage']['fraction_pooled']:.3f} \\\\",
        f"Gate verdict & {payload['gate_verdict']} \\\\",
        "\\bottomrule",
        "\\end{tabular}",
    ]
    path.write_text("\n".join(lines))
    return path
```

- [ ] **Step 4: Run writer test to verify pass**

Run: `pytest tests/test_stopdff_dp.py -v -k "writer_metric_type"`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/stopdff_dp/writers.py tests/test_stopdff_dp.py
git commit -m "feat(stopdff-dp): JSON/MD/LaTeX writers with metric_type=finite_horizon_dp"
```

---

### Task 8: Top-level CLI (compute_stopdff_dp.py)

**Files:**
- Create: `scripts/compute_stopdff_dp.py`
- Test: `tests/test_stopdff_dp.py` (smoke run via in-process invocation; full integration test)

- [ ] **Step 1: Write the failing integration test**

Append to `tests/test_stopdff_dp.py`:

```python
import sys


def test_cli_smoke_run_writes_all_three_artifacts(tmp_path, monkeypatch) -> None:
    """End-to-end identity-calibration smoke run writes JSON+MD+TeX."""
    # Build a tiny mc/val/test dataset in-place.
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    val_qs = [_fake_mc_question(f"v{i}") for i in range(5)]
    test_qs = [_fake_mc_question(f"t{i}") for i in range(5)]
    all_qs = val_qs + test_qs
    (data_dir / "mc_dataset.json").write_text(
        json.dumps(all_qs)
    )
    (data_dir / "val_dataset.json").write_text(json.dumps(val_qs))
    (data_dir / "test_dataset.json").write_text(json.dumps(test_qs))

    out_json = tmp_path / "stopdff_dp.json"
    out_md = tmp_path / "stopdff_dp.md"
    out_tex = tmp_path / "stopdff_dp_table.tex"

    from scripts import compute_stopdff_dp
    rc = compute_stopdff_dp.main([
        "--data-dir", str(data_dir),
        "--split", "test",
        "--fit-split", "val",
        "--reward-schedule", "acf_flat",
        "--continuation", "empirical_bucket",
        "--identity-calibration",  # skip SBERT for the test
        "--out", str(out_json),
        "--out-md", str(out_md),
        "--out-tex", str(out_tex),
        "--allow-incomplete-mc-coverage",
        "--allow-low-mc-retention",
    ])
    assert rc == 0
    assert out_json.exists() and out_md.exists() and out_tex.exists()
    payload = json.loads(out_json.read_text())
    assert payload["metadata"]["metric_type"] == "finite_horizon_dp"
    assert payload["metadata"]["fit_split"] == "val"
    assert payload["metadata"]["eval_split"] == "test"


def test_cli_rejects_same_split_for_fit_and_eval(tmp_path) -> None:
    from scripts import compute_stopdff_dp
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "mc_dataset.json").write_text("[]")
    (data_dir / "val_dataset.json").write_text("[]")
    (data_dir / "test_dataset.json").write_text("[]")
    rc = compute_stopdff_dp.main([
        "--data-dir", str(data_dir),
        "--split", "val",
        "--fit-split", "val",
        "--reward-schedule", "acf_flat",
        "--continuation", "empirical_bucket",
        "--identity-calibration",
        "--out", str(tmp_path / "out.json"),
        "--out-md", str(tmp_path / "out.md"),
        "--out-tex", str(tmp_path / "out.tex"),
    ])
    assert rc != 0
```

- [ ] **Step 2: Run new tests to verify they fail**

Run: `pytest tests/test_stopdff_dp.py -v -k "cli_smoke or cli_rejects"`
Expected: 2 FAIL (`ModuleNotFoundError: No module named 'scripts.compute_stopdff_dp'`).

- [ ] **Step 3: Write the CLI**

Write `scripts/compute_stopdff_dp.py`:

```python
#!/usr/bin/env python3
"""Compute finite-horizon DP StopDFF.

Supplementary metric to scripts/compute_stopdff.py: replaces the myopic
threshold stopping rule with an explicit backward-induction policy over
calibrated prefix trajectories. Writes JSON/MD/LaTeX exports under
``paper_exports/``. See docs/superpowers/plans/2026-05-27-stopdff-dp.md
for the design rationale.

Usage:
    python scripts/compute_stopdff_dp.py --help
    python scripts/compute_stopdff_dp.py --smoke
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_CALIBRATION = PROJECT_ROOT / "paper_exports" / "calibration.json"
DEFAULT_OUT_JSON = PROJECT_ROOT / "paper_exports" / "stopdff_dp.json"
DEFAULT_OUT_MD = PROJECT_ROOT / "paper_exports" / "stopdff_dp.md"
DEFAULT_OUT_TEX = PROJECT_ROOT / "paper_exports" / "stopdff_dp_table.tex"


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute finite-horizon DP StopDFF",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--responses", default=None,
                        help="(unused; reserved for future responses.json input)")
    parser.add_argument("--calibration", default=str(DEFAULT_CALIBRATION))
    parser.add_argument("--split", default="test")
    parser.add_argument("--fit-split", default="val")
    parser.add_argument("--reward-schedule", default="power_mark")
    parser.add_argument("--continuation", default="empirical_bucket",
                        choices=[
                            "oracle_trajectory", "empirical_bucket",
                            "pooled_empirical",
                        ])
    parser.add_argument("--out", default=str(DEFAULT_OUT_JSON))
    parser.add_argument("--out-md", default=str(DEFAULT_OUT_MD))
    parser.add_argument("--out-tex", default=str(DEFAULT_OUT_TEX))
    parser.add_argument("--smoke", action="store_true",
                        help="Trim to 30 val + 30 test questions for a <5min run.")
    parser.add_argument("--identity-calibration", action="store_true",
                        help="Skip SBERT and use deterministic synthetic signal (test only).")
    parser.add_argument("--allow-incomplete-mc-coverage", action="store_true")
    parser.add_argument("--allow-low-mc-retention", action="store_true")
    return parser.parse_args(argv)


def _load_split(path: Path) -> list[dict]:
    from scripts._common import iter_split_questions, load_json
    return iter_split_questions(load_json(path), source_path=path)


def main(argv: Optional[list[str]] = None) -> int:
    effective_argv = list(argv) if argv is not None else list(sys.argv[1:])
    args = _parse_args(argv)

    data_dir = Path(args.data_dir)
    out_json = Path(args.out)
    out_md = Path(args.out_md)
    out_tex = Path(args.out_tex)

    from scripts.stopdff_dp import adapter as adapter_module
    from scripts.stopdff_dp import continuation as continuation_module
    from scripts.stopdff_dp import diagnostics as diag_module
    from scripts.stopdff_dp import dp_solver as dp_module
    from scripts.stopdff_dp import rewards as rewards_module
    from scripts.stopdff_dp import writers as writers_module
    from scripts.stopdff_dp.continuation import (
        EmpiricalBucketEstimator,
        OracleTrajectoryEstimator,
        PooledEmpiricalEstimator,
        _assign_entropy_bin,
        _assign_p_bin,
        _assign_prefix_bucket,
    )

    try:
        adapter_module.validate_split_separation(
            fit_split=args.fit_split, eval_split=args.split
        )
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if not args.identity_calibration:
        if not Path(args.calibration).exists():
            print(
                f"ERROR: calibration JSON not found: {args.calibration}",
                file=sys.stderr,
            )
            return 1

    # Load split datasets.
    mc_path = data_dir / "mc_dataset.json"
    fit_path = data_dir / f"{args.fit_split}_dataset.json"
    eval_path = data_dir / f"{args.split}_dataset.json"
    for p in (mc_path, fit_path, eval_path):
        if not p.exists():
            print(f"ERROR: missing dataset {p}", file=sys.stderr)
            return 1

    from scripts._common import load_json, iter_split_questions
    mc_questions = load_json(mc_path)
    fit_questions = iter_split_questions(load_json(fit_path), source_path=fit_path)
    eval_questions = iter_split_questions(load_json(eval_path), source_path=eval_path)
    fit_qids = {str(q["qid"]) for q in fit_questions}
    eval_qids = {str(q["qid"]) for q in eval_questions}

    if args.smoke:
        # Subsample mc_questions to first 30 qids of each split.
        keep_fit = list(fit_qids)[:30]
        keep_eval = list(eval_qids)[:30]
        fit_qids = set(keep_fit)
        eval_qids = set(keep_eval)
        kept_qids = fit_qids | eval_qids
        mc_questions = [
            q for q in mc_questions if str(q["qid"]) in kept_qids
        ]

    calibration_path = Path(args.calibration) if not args.identity_calibration else None
    fit_df = adapter_module.build_dataframe(
        mc_questions=mc_questions,
        target_qids=fit_qids,
        split_name=args.fit_split,
        calibration_path=calibration_path,
        identity_calibration=args.identity_calibration,
    )
    eval_df = adapter_module.build_dataframe(
        mc_questions=mc_questions,
        target_qids=eval_qids,
        split_name=args.split,
        calibration_path=calibration_path,
        identity_calibration=args.identity_calibration,
    )

    schedule = rewards_module.get_schedule(args.reward_schedule)

    # Build continuation estimator.
    if args.continuation == "oracle_trajectory":
        estimator: object = OracleTrajectoryEstimator()
    elif args.continuation == "pooled_empirical":
        estimator = PooledEmpiricalEstimator.fit(
            fit_df=fit_df, fit_split_name=args.fit_split,
        )
    else:  # empirical_bucket
        estimator = EmpiricalBucketEstimator.fit(
            fit_df=fit_df, fit_split_name=args.fit_split,
        )

    # Run DP per (item, format) over the eval split.
    mc_traces: list = []
    qa_traces: list = []
    per_item_stopdff: list[tuple[str, int]] = []
    for item_id, group in eval_df.groupby("item_id"):
        group = group.sort_values("prefix_idx")
        mc_rows = group[group["format"] == "MC"]
        qa_rows = group[group["format"] == "QA"]
        if mc_rows.empty or qa_rows.empty:
            continue

        def _run(rows: pd.DataFrame, fmt: str):
            ps = rows["p_calibrated"].tolist()
            T = len(ps)
            prefix_fractions = [(i + 1) / T for i in range(T)]

            # Tag-capture pattern: solve_trajectory calls _continuation
            # once per backward step (t = T-2 .. 0). We record the
            # estimator's per-step tag immediately, then replay it from
            # the dict in _coverage_tagger after the backward loop ends.
            # Without this, _last_tag would be overwritten by every
            # successive call and the trace would record a single tag
            # for every step (the bug from the v1 draft of this plan).
            tags_per_step: dict[int, str] = {(T - 1): "exact"}

            def _continuation(t, p, prefix_fraction, _fmt=fmt, _ps=ps):
                if isinstance(estimator, OracleTrajectoryEstimator):
                    tags_per_step[t] = "exact"
                    return estimator.estimate(item_trajectory=_ps, t=t)
                v = estimator.estimate(
                    prefix_bucket=_assign_prefix_bucket(prefix_fraction),
                    fmt=_fmt,
                    subject_bucket=rows["subject"].iloc[0],
                    p_bin=_assign_p_bin(p),
                    entropy_bin=_assign_entropy_bin(p),
                )
                tags_per_step[t] = getattr(estimator, "_last_tag", "exact")
                return v

            def _coverage_tagger(t):
                return tags_per_step.get(t, "exact")

            return dp_module.solve_trajectory(
                p_trajectory=ps,
                prefix_fractions=prefix_fractions,
                schedule=schedule,
                continuation_fn=_continuation,
                item_id=str(item_id),
                fmt=fmt,
                coverage_tagger=_coverage_tagger,
            )

        mc_trace = _run(mc_rows, "MC")
        qa_trace = _run(qa_rows, "QA")
        mc_traces.append(mc_trace)
        qa_traces.append(qa_trace)
        per_item_stopdff.append(
            (str(item_id), dp_module.stopdff_for_item(
                mc_trace=mc_trace, qa_trace=qa_trace
            ))
        )

    coverage_summary = diag_module.summarize_coverage(mc_traces + qa_traces)
    ceiling_flags = diag_module.detect_ceiling_effects(mc_traces, qa_traces)

    confirmatory = not isinstance(estimator, OracleTrajectoryEstimator)
    if not confirmatory:
        print(
            "WARNING: oracle_trajectory continuation is upper-bound diagnostic only; "
            "output is flagged confirmatory=false.",
            file=sys.stderr,
        )

    if coverage_summary["verdict"] == "warn":
        gate_verdict = "warn"
        gate_verdict_reason = f"coverage:{coverage_summary['reason']}"
    elif any(ceiling_flags[k] for k in (
        "all_stop_at_first_prefix",
        "all_stop_at_final_prefix",
        "no_cross_format_stopping_variance",
    )):
        gate_verdict = "warn"
        gate_verdict_reason = "ceiling_effect"
    else:
        gate_verdict = "pass"
        gate_verdict_reason = "all_clean"

    # Provenance.
    try:
        from scripts._common import build_generation_provenance
        generation = build_generation_provenance(
            __file__, effective_argv,
            output_path=out_json,
            extra_paths=[calibration_path] if calibration_path else [],
        )
    except Exception:  # noqa: BLE001 — provenance is best-effort
        generation = None

    payload = writers_module.assemble_payload(
        mc_traces=mc_traces,
        qa_traces=qa_traces,
        reward_schedule_name=args.reward_schedule,
        continuation_estimator_name=args.continuation,
        fit_split=args.fit_split,
        eval_split=args.split,
        coverage_summary=coverage_summary,
        ceiling_flags=ceiling_flags,
        per_item_stopdff=per_item_stopdff,
        gate_verdict=gate_verdict,
        gate_verdict_reason=gate_verdict_reason,
        confirmatory=confirmatory,
        generation=generation,
    )

    writers_module.write_json(out_json, payload)
    writers_module.write_markdown(out_md, payload)
    writers_module.write_latex(out_tex, payload)

    print(
        f"[STOPDFF-DP] Wrote {out_json} (verdict={gate_verdict}, "
        f"n_items={len(per_item_stopdff)})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run CLI tests to verify pass**

Run: `pytest tests/test_stopdff_dp.py -v -k "cli_smoke or cli_rejects"`
Expected: 2 PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/compute_stopdff_dp.py tests/test_stopdff_dp.py
git commit -m "feat(stopdff-dp): CLI orchestrator with smoke mode and split-separation guard"
```

---

### Task 9: No-leakage end-to-end test

**Files:**
- Modify: `tests/test_stopdff_dp.py` (add leakage-prevention integration test)

- [ ] **Step 1: Add the integration test**

Append to `tests/test_stopdff_dp.py`:

```python
def test_fit_dataframe_never_contains_eval_split_rows(tmp_path) -> None:
    """Confirm via direct inspection that the fit_df has only val rows."""
    val_qs = [_fake_mc_question(f"v{i}") for i in range(3)]
    test_qs = [_fake_mc_question(f"t{i}") for i in range(3)]
    mc_pool = val_qs + test_qs
    val_qids = {q["qid"] for q in val_qs}
    test_qids = {q["qid"] for q in test_qs}

    fit_df = adapter_module.build_dataframe(
        mc_questions=mc_pool,
        target_qids=val_qids,
        split_name="val",
        calibration_path=None,
        identity_calibration=True,
    )
    eval_df = adapter_module.build_dataframe(
        mc_questions=mc_pool,
        target_qids=test_qids,
        split_name="test",
        calibration_path=None,
        identity_calibration=True,
    )

    assert set(fit_df["split"]) == {"val"}
    assert set(eval_df["split"]) == {"test"}
    assert set(fit_df["item_id"]).isdisjoint(set(eval_df["item_id"]))

    # And the EmpiricalBucketEstimator must refuse to fit on the eval frame.
    with pytest.raises(ValueError):
        cont_module.EmpiricalBucketEstimator.fit(
            fit_df=eval_df, fit_split_name="val",
        )
```

- [ ] **Step 2: Run the test to verify pass**

Run: `pytest tests/test_stopdff_dp.py -v -k "fit_dataframe_never_contains_eval_split"`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_stopdff_dp.py
git commit -m "test(stopdff-dp): assert fit/eval dataframes never share item ids"
```

---

### Task 10: Smoke run on real artifacts

**Files:**
- No file changes. Run the CLI against the local data dir; record artifacts under `paper_exports/`.

- [ ] **Step 1: Run smoke pipeline**

Run from repo root:

```bash
python scripts/compute_stopdff_dp.py \
  --responses paper_exports/responses.json \
  --calibration paper_exports/calibration.json \
  --split test \
  --fit-split val \
  --reward-schedule power_mark \
  --continuation empirical_bucket \
  --smoke \
  --out paper_exports/stopdff_dp.json \
  --allow-incomplete-mc-coverage \
  --allow-low-mc-retention
```

Expected: exit 0, writes `paper_exports/stopdff_dp.{json,md}` and `paper_exports/stopdff_dp_table.tex`, completes in <5 minutes. `--responses` is accepted but unused (reserved for future input).

- [ ] **Step 2: Verify outputs**

Run:

```bash
python -c "import json; d=json.load(open('paper_exports/stopdff_dp.json')); assert d['metadata']['metric_type']=='finite_horizon_dp'; print('OK')"
ls -la paper_exports/stopdff_dp.json paper_exports/stopdff_dp.md paper_exports/stopdff_dp_table.tex
```

Expected: prints `OK` and three files exist.

- [ ] **Step 3: Commit smoke artifacts**

```bash
git add paper_exports/stopdff_dp.json paper_exports/stopdff_dp.md paper_exports/stopdff_dp_table.tex
git commit -m "chore(stopdff-dp): smoke-run artifacts under paper_exports/"
```

---

### Task 11: Run full test suite for the DP module

**Files:**
- No file changes. Verify everything together.

- [ ] **Step 1: Run only the DP test file**

Run: `pytest tests/test_stopdff_dp.py -q`
Expected: All tests in `tests/test_stopdff_dp.py` PASS.

- [ ] **Step 2: Run the broader pipeline-related tests to confirm no regressions**

Run:

```bash
pytest tests/test_pr14_review_regressions.py tests/test_stopdff_dp.py -q
```

Expected: All PASS. (The myopic diagnostic regression suite must still pass because `scripts/compute_stopdff.py` is untouched.)

- [ ] **Step 3: No commit unless tests changed**

---

### Task 12: Optional audit-card integration

**Files:**
- Modify: `scripts/make_audit_card.py` (add `--include-dp-stopdff` flag; new row only when present)
- Test: `tests/test_stopdff_dp.py` (audit-card row appears without removing existing one)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_stopdff_dp.py`:

```python
def test_audit_card_row_added_without_replacing_diagnostic(tmp_path, monkeypatch):
    """The DP StopDFF row should appear after the existing diagnostic row."""
    from scripts import make_audit_card
    # Stub-load the three existing JSONs and one new DP JSON.
    paper = tmp_path / "paper_exports"
    paper.mkdir()
    # Copy minimum-valid fixtures from the repo paper_exports directory.
    import shutil
    src = Path(__file__).resolve().parent.parent / "paper_exports"
    for fname in ("csli.json", "calibration.json", "stopdff.json"):
        shutil.copyfile(src / fname, paper / fname)
    # Synthesize a minimal stopdff_dp.json.
    (paper / "stopdff_dp.json").write_text(json.dumps({
        "stopdff_dp_signed_median": -0.5,
        "stopdff_dp_signed_mean": -0.4,
        "stopdff_dp_abs_median": 0.5,
        "n_items": 10,
        "direction_breakdown": {"mc_earlier": 5, "qa_earlier": 3, "same_step": 2},
        "coverage": {"verdict": "pass", "fraction_exact": 1.0,
                     "fraction_pooled": 0.0, "fraction_missing": 0.0,
                     "n_cells": 60, "reason": "ok"},
        "ceiling_flags": {"all_stop_at_first_prefix": False,
                          "all_stop_at_final_prefix": False,
                          "no_cross_format_stopping_variance": False,
                          "n_trajectories": 10, "n_stopped_cells": 50,
                          "n_never_stopped_cells": 10},
        "gate_verdict": "pass",
        "gate_verdict_reason": "all_clean",
        "confirmatory": True,
        "metadata": {"metric_type": "finite_horizon_dp",
                     "stopping_policy": "finite_horizon_dp",
                     "reward_schedule": "power_mark",
                     "continuation_estimator": "empirical_bucket",
                     "fit_split": "val", "eval_split": "test"},
    }))
    monkeypatch.setattr(make_audit_card, "_PAPER_EXPORTS", paper)
    rc = make_audit_card.main_with_args(["--include-dp-stopdff"])
    assert rc == 0
    card = json.loads((paper / "audit_card.json").read_text())
    names = [m["name"] for m in card["metrics"]]
    assert any("Diagnostic StopDFF" in n for n in names)
    assert any("DP StopDFF" in n for n in names)
```

- [ ] **Step 2: Run new test to verify fail**

Run: `pytest tests/test_stopdff_dp.py -v -k "audit_card_row_added"`
Expected: FAIL — `_PAPER_EXPORTS` is a module-level path; `main_with_args` does not exist yet.

- [ ] **Step 3: Refactor make_audit_card so it supports the flag**

Edit `scripts/make_audit_card.py`:

1. Replace the parser body with one that adds `--include-dp-stopdff`:

```python
def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Pilot Benchmark Translation Audit Card"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Parse args and print what would happen without writing files",
    )
    parser.add_argument(
        "--include-dp-stopdff", action="store_true",
        help=(
            "Append a finite-horizon DP StopDFF row from paper_exports/"
            "stopdff_dp.json to the audit card (in addition to, not replacing, "
            "the existing diagnostic row)."
        ),
    )
    return parser.parse_args(argv)
```

2. Add a `_evaluate_stopdff_dp` helper next to `_evaluate_stopdff`:

```python
def _evaluate_stopdff_dp(dp_data: dict) -> dict:
    """Evaluate DP StopDFF (signed median) against ±1 prefix tolerance."""
    signed_median = dp_data["stopdff_dp_signed_median"]
    coverage = dp_data["coverage"]
    verdict = dp_data["gate_verdict"]
    confirmatory = dp_data.get("confirmatory", False)
    qualifier_parts = []
    if not confirmatory:
        qualifier_parts.append("non-confirmatory continuation estimator")
    if coverage.get("verdict") == "warn":
        qualifier_parts.append(coverage.get("reason", "coverage warn"))
    return {
        "name": "DP StopDFF (Finite-Horizon Bellman, signed median)",
        "value": signed_median,
        "value_display": f"{signed_median:+.4f}",
        "threshold": 1,
        "threshold_criterion": "|signed_median_stopdff| <= 1",
        "observed_criterion_value": abs(signed_median),
        "direction": "warn_if_above",
        "verdict": verdict,
        "verdict_qualifier": "; ".join(qualifier_parts) if qualifier_parts else None,
        "details": {
            "reward_schedule": dp_data["metadata"]["reward_schedule"],
            "continuation_estimator": dp_data["metadata"]["continuation_estimator"],
            "fit_split": dp_data["metadata"]["fit_split"],
            "eval_split": dp_data["metadata"]["eval_split"],
            "coverage": coverage,
            "ceiling_flags": dp_data["ceiling_flags"],
            "n_items": dp_data["n_items"],
            "direction_breakdown": dp_data["direction_breakdown"],
            "confirmatory": confirmatory,
            "metric_type": dp_data["metadata"]["metric_type"],
        },
    }
```

3. Wrap the existing `main()` body so it accepts argv (so the test can drive it):

```python
def main_with_args(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    # ... [existing body of main(), unchanged, EXCEPT the metric list build
    # appends the DP row when args.include_dp_stopdff is True] ...
    metrics = [
        _evaluate_csli(csli_data, thresholds["choices_only_accuracy"]),
        _evaluate_calibration(cal_data, thresholds["prefix_ece"]),
        _evaluate_stopdff(stopdff_data, thresholds["stopdff_median_abs_prefix"]),
    ]
    if args.include_dp_stopdff:
        dp_path = _PAPER_EXPORTS / "stopdff_dp.json"
        if not dp_path.exists():
            print(
                "WARNING: --include-dp-stopdff was passed but "
                f"{dp_path} does not exist; the DP row was skipped.",
                file=sys.stderr,
            )
        else:
            dp_data = _load_json(dp_path)
            metrics.append(_evaluate_stopdff_dp(dp_data))
    # ... rest of body unchanged ...


def main() -> int:
    return main_with_args(None)
```

Keep all other behavior of `main()` (provenance, data-provenance, artifact-provenance) unchanged. The new metric does not need to contribute to `_build_artifact_provenance` for v1 — note this limitation in a comment.

- [ ] **Step 4: Re-run the audit-card test**

Run: `pytest tests/test_stopdff_dp.py -v -k "audit_card_row_added"`
Expected: PASS.

- [ ] **Step 5: Re-run audit-card regression tests**

Run:

```bash
pytest tests/test_pr14_review_regressions.py -q
```

Expected: PASS (the diagnostic row is untouched).

- [ ] **Step 6: Commit**

```bash
git add scripts/make_audit_card.py tests/test_stopdff_dp.py
git commit -m "feat(audit-card): opt-in DP StopDFF row via --include-dp-stopdff"
```

---

### Task 13: Final verification before declaring done

**Files:**
- No file changes.

- [ ] **Step 1: Run the full DP test file**

Run: `pytest tests/test_stopdff_dp.py -q`
Expected: All PASS.

- [ ] **Step 2: Run the targeted regression suite that depends on the StopDFF surface**

Run:

```bash
pytest tests/test_stopdff_dp.py tests/test_pr14_review_regressions.py -q
```

Expected: All PASS.

- [ ] **Step 3: Sanity-check coverage and ceiling reporting on real artifacts**

Run:

```bash
python -c "
import json
d = json.load(open('paper_exports/stopdff_dp.json'))
print('coverage:', d['coverage'])
print('ceiling :', d['ceiling_flags'])
print('verdict :', d['gate_verdict'], '-', d['gate_verdict_reason'])
"
```

Expected: prints non-empty coverage + ceiling diagnostics with a coherent verdict reason.

- [ ] **Step 4: Confirm the diagnostic StopDFF is unchanged**

Run:

```bash
git status -- scripts/compute_stopdff.py
git status -- scripts/compute_csli.py
```

Expected: both report no changes.

- [ ] **Step 5: Final commit (only if any housekeeping was required)**

If there are no pending changes, skip. Otherwise:

```bash
git add -A
git commit -m "chore(stopdff-dp): final verification housekeeping"
```

---

## Notes on out-of-scope items

The user spec says "do not update paper claims automatically without regenerated artifacts." This plan deliberately does NOT modify any of the following:
- `compute_stopdff.py` (myopic diagnostic — left untouched)
- `compute_csli.py` (CSLI semantics — left untouched)
- `compute_prefix_calibration.py` (calibration — left untouched)
- `paper_exports/stopdff.json` (existing myopic artifact — not overwritten)
- `paper_exports/audit_card.{json,md}` schema for the existing three rows — left untouched; the DP row is purely additive when the new flag is set
- `threshold_manifest.json` — left untouched; the DP metric uses an inline ±1 prefix tolerance that is documented in the metadata

If the operator later wants to register a frozen DP threshold, the change should land as a separate manifest update with a fresh attestation, not as part of this plan.

## Notes on `--responses paper_exports/responses.json`

This file does not exist in the current repo. The CLI accepts the flag for future compatibility but reads from `data/processed/{mc,val,test}_dataset.json` (the existing artifacts) and `paper_exports/calibration.json` (existing) as documented in Task 8. If a future change introduces a top-level `responses.json`, the adapter can be extended to consume it; the current adapter is structured to make that swap trivial (one new branch in `build_dataframe`).
