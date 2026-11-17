"""Public reward instructions for both opt-in ACL horizon modes."""

from __future__ import annotations

from dataclasses import dataclass, fields
from decimal import Decimal
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qb_data.acl_protocol import ArmSpec
    from qb_env.acl_rewards import RewardSpec


def _number(value: float) -> str:
    """Use the shortest round-trip decimal, omitting redundant integer decimals."""
    return repr(float(value)).removesuffix(".0")


@dataclass(frozen=True)
class PublicPayoffs:
    reward: RewardSpec
    correct_reward: float
    wrong_reward: float
    wait_cost: float
    abstain_reward: float

    @classmethod
    def from_env(cls, env):
        return cls(env.acl_reward, env.buzz_correct, env.buzz_incorrect,
                   -env.wait_penalty, env.no_buzz_reward)

    def matches(self, spec):
        return all(getattr(self, field.name) == getattr(spec, field.name) for field in fields(self))


def reward_instructions(spec: ArmSpec | PublicPayoffs, *, end_mode: str = "no_buzz") -> str:
    """Describe public payoffs without taking a question, state, label or realized horizon."""
    reward = spec.reward
    arm = reward.arm
    if arm in ("baseline", "constant"):
        correct = spec.correct_reward if arm == "baseline" else reward.start
        payoff = f"Correct {_number(correct)}."
    else:
        k = _number(reward.shape)
        if arm == "linear" or (arm == "exponential" and reward.shape == 0):
            decay = "1-p"
        elif arm == "exponential":
            decay = f"(exp(-{k}*p)-exp(-{k}))/(1-exp(-{k}))"
        elif arm == "power":
            decay = f"(1-p)^{k}"
        elif arm == "cosine":
            decay = "(1+cos(pi*p))/2"
        elif arm == "step":
            decay = f"1 if p<{Decimal.from_float(reward.threshold)}, otherwise 0"
        elif arm == "piecewise":
            points = ", ".join(f"({_number(x)},{_number(y)})" for x, y in reward.knots)
            decay = f"linear interpolation through (p,D) points {points}"
        else:
            raise ValueError(f"Unknown ACL reward schedule {arm!r}")
        start, end = _number(reward.start), _number(reward.end)
        payoff = (f"Correct {end}+({start}-{end})*D. D={decay}. "
                  "p=(prefix number-1)/(total prefixes-1), p=1 for a single prefix.")
    gamma = _number(reward.gamma)
    if end_mode == "no_buzz":
        final_payment = f"final WAIT also adds {_number(spec.abstain_reward)} for passing."
    elif end_mode == "force_commit":
        final_payment = "final WAIT also earns the correct or wrong payoff for the forced choice."
    else:
        raise ValueError(f"unknown end mode {end_mode!r}")
    text = (f"{payoff} Wrong {_number(spec.wrong_reward)}, every WAIT {_number(spec.wait_cost)}, "
            f"{final_payment} "
            f"Maximize expected points with per-step discount {gamma}.")
    if reward.potential_scale:
        text += (f" Each step adds {gamma}*P(next)-P(now), where "
                 f"P={_number(reward.potential_scale)} times the environment's highest "
                 "current option probability and terminal P=0.")
    return text
