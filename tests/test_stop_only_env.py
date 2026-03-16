"""Tests for StopOnlyEnv wrapper action_masks and step mapping."""

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces

from qb_env.stop_only_env import StopOnlyEnv


class FakeBaseEnv(gym.Env):
    """Minimal fake TossupMCEnv for testing StopOnlyEnv."""

    def __init__(self, belief=None):
        super().__init__()
        self.belief = belief
        self.observation_space = spaces.Box(low=-1, high=1, shape=(10,))
        self.action_space = spaces.Discrete(5)
        self._last_action = None

    def reset(self, seed=None, options=None):
        return np.zeros(10, dtype=np.float32), {}

    def step(self, action):
        self._last_action = action
        return np.zeros(10, dtype=np.float32), 0.0, True, False, {"step_idx": 0}


def test_action_masks_shape_and_dtype():
    env = StopOnlyEnv(FakeBaseEnv(belief=np.array([0.2, 0.8])))
    masks = env.action_masks()
    assert masks.shape == (2,)
    assert masks.dtype == bool


def test_action_masks_both_true_when_belief_present():
    env = StopOnlyEnv(FakeBaseEnv(belief=np.array([0.2, 0.8])))
    masks = env.action_masks()
    assert masks[0]
    assert masks[1]


def test_action_masks_buzz_false_when_no_belief():
    env = StopOnlyEnv(FakeBaseEnv(belief=None))
    masks = env.action_masks()
    assert masks[0]
    assert not masks[1]


def test_step_buzz_maps_to_argmax():
    base = FakeBaseEnv(belief=np.array([0.1, 0.3, 0.6]))
    env = StopOnlyEnv(base)
    env.step(1)
    assert base._last_action == 3  # 1 + argmax([0.1, 0.3, 0.6]) = 1 + 2


@pytest.mark.parametrize("belief", [None, np.array([])])
def test_step_buzz_raises_when_belief_unavailable(belief):
    env = StopOnlyEnv(FakeBaseEnv(belief=belief))

    with pytest.raises(ValueError, match="belief is unavailable"):
        env.step(1)
