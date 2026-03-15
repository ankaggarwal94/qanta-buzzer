"""Stop-only action-space wrapper for the quiz bowl environment."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from qb_env.tossup_env import TossupMCEnv


class StopOnlyEnv(gym.Wrapper):
    """Wrap TossupMCEnv with a binary WAIT/BUZZ action space.

    Action mapping:
    - 0 -> WAIT
    - 1 -> BUZZ using the current answer-selection strategy

    The default answer-selection strategy commits to the current belief argmax.
    """

    def __init__(self, env: TossupMCEnv, answer_mode: str = "argmax_belief") -> None:
        super().__init__(env)
        self.answer_mode = answer_mode
        self.action_space = spaces.Discrete(2)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        return self.env.reset(seed=seed, options=options)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")
        if action == 0:
            return self.env.step(0)

        if self.answer_mode != "argmax_belief":
            raise ValueError(f"Unknown answer_mode: {self.answer_mode}")

        chosen_idx = int(np.argmax(self.env.belief))
        return self.env.step(1 + chosen_idx)
