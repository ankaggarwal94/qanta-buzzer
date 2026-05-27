"""Reward schedule registry for finite-horizon DP StopDFF.

Each schedule defines answer utility A_t(p) = R(t)*p + R_wrong*(1-p)
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
