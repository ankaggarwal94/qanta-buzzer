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
