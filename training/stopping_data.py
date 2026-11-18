"""Finite WAIT/BUZZ trajectories from the belief environment's argmax answerer."""

from __future__ import annotations

from copy import copy
from dataclasses import dataclass
from hashlib import sha256
from typing import Sequence

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from qb_data.dataset_splits import normalize_question_text
from qb_env.stop_only_env import StopOnlyEnv
from qb_env.tossup_env import TossupMCEnv


@dataclass(frozen=True)
class StoppingEpisode:
    """Actionable observations and rewards from a single question and menu."""

    observations: np.ndarray
    buzz_rewards: np.ndarray
    wait_rewards: np.ndarray
    qid: str
    group_id: str
    buzz_correct: np.ndarray
    visible_prefixes: np.ndarray
    total_prefixes: int
    end_mode: str
    final_correct: bool


def collect_episodes(env: TossupMCEnv, *, seed: int) -> list[StoppingEpisode]:
    """Follow WAIT once per question, taking a terminal BUZZ on a copy of each state."""
    episodes = []
    for i, question in enumerate(env.questions):
        obs, _ = env.reset(seed=seed + i, options={"question_idx": i})
        observations, buzz_rewards, wait_rewards, correct, visible = [], [], [], [], []
        for _ in range(env.total_steps):
            observations.append(obs)
            visible.append(env.step_idx)
            # BUZZ only reads shared beliefs, question and scorer. Its terminal
            # flags and scalar step state belong to this branch alone.
            fork = StopOnlyEnv(copy(env))
            _, buzz, ended, _, buzz_info = fork.step(1)
            if not ended:
                raise RuntimeError("A BUZZ must terminate the episode")
            buzz_rewards.append(buzz)
            correct.append(buzz_info["correct"])
            obs, wait, done, truncated, info = env.step(0)
            wait_rewards.append(wait)
            if done or truncated:
                break
        else:
            raise RuntimeError("WAIT exceeded the question horizon")
        if not all(np.isfinite(x).all() for x in (observations, buzz_rewards, wait_rewards)):
            raise ValueError(f"Nonfinite stopping trajectory for {question.qid}")
        text_key = normalize_question_text(question.question)
        episodes.append(StoppingEpisode(
            observations=np.stack(observations), buzz_rewards=np.asarray(buzz_rewards),
            wait_rewards=np.asarray(wait_rewards), qid=str(question.qid),
            group_id=sha256(text_key.encode()).hexdigest(), buzz_correct=np.asarray(correct),
            visible_prefixes=np.asarray(visible), total_prefixes=env.total_steps,
            end_mode=env.end_mode, final_correct=bool(info["forced_correct"]),
        ))
    return episodes


def evaluate_actions(episode: StoppingEpisode, actions: Sequence[int]) -> dict:
    """Score the first voluntary BUZZ, or the live environment's terminal WAIT."""
    actions = np.asarray(actions)
    if actions.shape != episode.wait_rewards.shape or not np.isin(actions, (0, 1)).all():
        raise ValueError("Expected one binary action per actionable observation")
    stops = np.flatnonzero(actions)
    if len(stops):
        t = stops[0]
        outcome = "voluntary"
        correct = bool(episode.buzz_correct[t])
        reward = episode.wait_rewards[:t].sum() + episode.buzz_rewards[t]
        visible = int(episode.visible_prefixes[t])
    else:
        outcome = episode.end_mode
        correct = episode.final_correct
        reward = episode.wait_rewards.sum()
        visible = episode.total_prefixes
    return {"qid": episode.qid, "outcome": outcome, "correct": correct,
            "visible_prefixes": visible, "return": float(reward)}


class ReplayEnv(gym.Env):
    """Sample training questions for SB3 and return the same finite payoff table."""

    def __init__(self, episodes: list[StoppingEpisode]):
        super().__init__()
        self.episodes = episodes
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-np.inf, np.inf, episodes[0].observations.shape[1:], np.float32)
        self._episode = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        index = (options["question_idx"] if options and "question_idx" in options
                 else self.np_random.integers(len(self.episodes)))
        self._episode = self.episodes[index]
        self._step = 0
        return self._episode.observations[0], {"qid": self._episode.qid}

    def step(self, action):
        episode = self._episode
        if episode is None:
            raise RuntimeError("Call reset before stepping")
        if not self.action_space.contains(action):
            raise ValueError(f"Unknown stop action {action}")
        t = self._step
        reward = episode.buzz_rewards[t] if action else episode.wait_rewards[t]
        done = bool(action or t + 1 == len(episode.observations))
        if done:
            self._episode = None
        else:
            self._step += 1
        # A final WAIT settles reward. There is no continuation value to bootstrap.
        return episode.observations[self._step], float(reward), done, False, {}
