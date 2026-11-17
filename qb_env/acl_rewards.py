"""Reward schedules on the observed prefix and optional potential shaping."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from math import cos, exp, expm1, isfinite, pi, sin
from numbers import Real
from typing import Any, Mapping


def _finite(name: str, value: Real) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
        raise ValueError(f"{name} must be a finite real number")
    return float(value)


@dataclass(frozen=True)
class RewardSpec:
    """Parameters of the opt-in correct-buzz schedule and shaping potential."""

    arm: str = "baseline"
    start: float = 1.
    end: float = 0.
    shape: float = 1.
    knots: tuple[tuple[float, float], ...] = ((0., 1.), (1., 0.))
    threshold: float = .5
    gamma: float = .99
    potential_scale: float = 0.

    def __post_init__(self) -> None:
        for name in ("start", "end", "shape", "threshold", "gamma", "potential_scale"):
            object.__setattr__(self, name, _finite(name, getattr(self, name)))
        if self.arm not in ("baseline", "constant", "linear", "exponential",
                            "power", "cosine", "piecewise", "step"):
            raise ValueError(f"Unknown correct reward schedule {self.arm!r}")
        if not 0 <= self.end <= self.start:
            raise ValueError("Reward endpoints require 0 <= end <= start")
        if not 0 <= self.gamma <= 1 or not 0 < self.threshold <= 1:
            raise ValueError("gamma must be in [0,1] and threshold in (0,1]")
        if self.shape < 0 or (self.arm == "power" and self.shape == 0):
            raise ValueError("shape must be nonnegative and strictly positive for power")
        try:
            knots = tuple((_finite("position", x), _finite("decay", y)) for x, y in self.knots)
        except (ValueError, TypeError) as exc:
            raise ValueError("knots must be a sequence of finite position/decay pairs") from exc
        if (not knots or knots[0] != (0., 1.) or knots[-1] != (1., 0.)
                or any(x >= z or y < w for (x,y), (z,w) in zip(knots, knots[1:]))):
            raise ValueError("knots must descend from (0,1) to (1,0) with increasing positions")
        object.__setattr__(self, "knots", knots)

    @classmethod
    def from_dict(cls, mapping: Mapping[str, Any]) -> RewardSpec:
        return cls(**mapping)

    def correct_reward(self, progress: float, baseline: float) -> float:
        """Evaluate the correct-buzz payoff at normalized prefix progress."""
        p = _finite("progress", progress)
        if not 0 <= progress <= 1:
            raise ValueError("progress must lie in [0,1]")
        if self.arm == "step":
            # A rational prefix may lie strictly below a binary64 threshold
            # while rounding to exactly that threshold.
            p = progress
        if self.arm == "baseline":
            return _finite("baseline correct reward", baseline)
        if self.arm == "constant" or p == 0:
            return self.start
        if p == 1:
            return self.end
        remaining = 1. - p
        if self.arm == "linear":
            d = remaining
        elif self.arm == "exponential":
            k = self.shape
            if k == 0.:
                d = remaining
            elif k < 1.:
                # Factor out (1-p) before division. A direct ratio of two
                # subnormal expm1 values can lose all significant digits.
                kr = k * remaining
                relative = -expm1(-kr) / kr if kr else 1.
                d = remaining * exp(-k*p) * relative / (-expm1(-k)/k)
            else:
                d = exp(-k*p) * (-expm1(-k*remaining)) / (-expm1(-k))
        elif self.arm == "power":
            d = remaining ** self.shape
        elif self.arm == "cosine":
            d = sin(pi * remaining / 2.)**2 if p > .5 else cos(pi * p / 2.)**2
        elif self.arm == "step":
            d = float(p < self.threshold)
        else:
            right = bisect_right(self.knots, p, key=lambda knot: knot[0])
            x, y = self.knots[right-1]
            z, w = self.knots[right]
            d = w + (y-w) * ((z-p)/(z-x))
        return min(self.start, max(self.end, self.end + (self.start-self.end)*d))
