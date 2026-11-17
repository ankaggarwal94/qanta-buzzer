"""Exact dyadic settlement and explicit Gym/learner rounding receipts."""

from __future__ import annotations

from collections.abc import Callable
from fractions import Fraction
from math import isfinite
import struct

TransitionValidator = Callable[[Fraction, float, float, float, float], None]


def settle_reward(
    wait: float,
    outcome: float,
    gamma: float,
    current: float,
    successor: float,
    validator: TransitionValidator | None = None,
) -> dict:
    """Settle one exact transition and record the two observed ABI conversions.

    Input payoffs, discount and state potentials are the exact real values
    represented by their binary64 encodings. The Gym scalar is rounded once
    from the exact total. PPO's float32 reward buffer can round it again.

    Parameters
    ----------
    wait, outcome : float
        Wait charge and outcome payoff for this transition.
    gamma, current, successor : float
        Discount, current potential and successor potential.
    validator : callable, optional
        Checks the exact base, discount, potentials and rounded Gym reward.
        A rejection propagates to the caller.

    Returns
    -------
    dict
        Exact rational components, Gym and learner rewards, and rounding errors.
    """
    base = Fraction(wait) + Fraction(outcome)
    shaping = Fraction(gamma) * Fraction(successor) - Fraction(current)
    total = base + shaping
    try:
        reward = float(total)
        base_display = float(base)
        shaping_display = float(shaping)
        learner = struct.unpack(">f", struct.pack(">f", reward))[0]
    except OverflowError as exc:
        raise ValueError("ACL reward exceeds a finite ABI or diagnostic format") from exc
    if not isfinite(learner):
        raise ValueError("ACL learner reward must be finite in binary32")
    if validator is not None:
        validator(base, gamma, current, successor, reward)
    return {
        "base_reward": base_display,
        "shaping_reward": shaping_display,
        "reward": reward,
        "learner_reward_float32": learner,
        "exact": {
            "base": list(base.as_integer_ratio()),
            "shaping": list(shaping.as_integer_ratio()),
            "total": list(total.as_integer_ratio()),
            "gym_rounding_error": list((Fraction(reward) - total).as_integer_ratio()),
            "learner_rounding_error": list((Fraction(learner) - total).as_integer_ratio()),
        },
    }
