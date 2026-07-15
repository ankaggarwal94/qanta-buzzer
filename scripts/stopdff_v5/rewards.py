"""Reward schedules (SCIENTIFIC_CONTRACT.md section 6).

| Name            | Early correct | Late correct | Wrong | Early/late split | Wait cost |
|-----------------|--------------:|-------------:|------:|-----------------:|----------:|
| acf_flat        | 10            | 10           | -5    | 1.0              | 0         |
| power_mark      | 15            | 10           | -5    | 0.5              | 0         |
| wait_cost_small | 15            | 10           | -5    | 0.5              | 0.05      |
| strict_wrong    | 15            | 10           | -10   | 0.5              | 0         |

"Early" means prefix_fraction < early_late_split. All values are part of the scientific
identity and are stored as canonical decimal strings.
"""
from __future__ import annotations

from dataclasses import dataclass

# Canonical decimal-string reward parameters. These strings ARE the scientific
# identity; float() of each is the float64 value used in computation.
REWARD_SCHEDULE_STRINGS: dict[str, dict[str, str]] = {
    "acf_flat": {"correct_early": "10", "correct_late": "10", "wrong": "-5", "split": "1.0", "wait_cost": "0"},
    "power_mark": {"correct_early": "15", "correct_late": "10", "wrong": "-5", "split": "0.5", "wait_cost": "0"},
    "wait_cost_small": {"correct_early": "15", "correct_late": "10", "wrong": "-5", "split": "0.5", "wait_cost": "0.05"},
    "strict_wrong": {"correct_early": "15", "correct_late": "10", "wrong": "-10", "split": "0.5", "wait_cost": "0"},
}

# Canonical axis order (used for FVI representative-cell index parity).
REWARD_ORDER: tuple[str, ...] = ("acf_flat", "power_mark", "wait_cost_small", "strict_wrong")


@dataclass(frozen=True)
class RewardSchedule:
    name: str
    correct_early: float
    correct_late: float
    wrong: float
    split: float
    wait_cost: float

    def r_correct(self, prefix_fraction: float) -> float:
        """R_correct(t): early reward when prefix_fraction < split, else late."""
        return self.correct_early if prefix_fraction < self.split else self.correct_late

    @property
    def max_correct_reward(self) -> float:
        return max(self.correct_early, self.correct_late)

    def identity(self) -> dict[str, str]:
        return dict(REWARD_SCHEDULE_STRINGS[self.name])


def get_schedule(name: str) -> RewardSchedule:
    try:
        s = REWARD_SCHEDULE_STRINGS[name]
    except KeyError as exc:
        valid = ", ".join(REWARD_ORDER)
        raise ValueError(f"Unknown reward schedule {name!r}. Valid: {valid}.") from exc
    return RewardSchedule(
        name=name,
        correct_early=float(s["correct_early"]),
        correct_late=float(s["correct_late"]),
        wrong=float(s["wrong"]),
        split=float(s["split"]),
        wait_cost=float(s["wait_cost"]),
    )


def answer_utility(p: float, prefix_fraction: float, schedule: RewardSchedule) -> float:
    """A_t(p) = R_correct(t) * p + R_wrong * (1 - p)."""
    return schedule.r_correct(prefix_fraction) * p + schedule.wrong * (1.0 - p)
