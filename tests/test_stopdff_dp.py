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
