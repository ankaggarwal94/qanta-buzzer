"""PPO Buzzer agent wrapping Stable-Baselines3's PPO.

Provides the PPOBuzzer class for training an MLP policy on belief-feature
observations from TossupMCEnv, and PPOEpisodeTrace for recording per-step
action probabilities needed to compute the S_q scoring metric.

The key design rationale: SB3's ``learn()`` does not expose per-step action
distributions, so ``run_episode()`` implements custom episode execution that
records c_trace (buzz probability) and p_correct_trace (correctness probability)
at each step for downstream S_q computation.

Ported from qb-rl reference implementation (agents/ppo_buzzer.py) with
import path adaptations for the unified qanta-buzzer codebase.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch as th
from stable_baselines3 import PPO

from qb_env.tossup_env import TossupMCEnv


@dataclass
class PPOEpisodeTrace:
    """Record of a single episode with per-step action probability traces.

    Used to compute the S_q scoring metric: S_q = sum(c_t * g_t) over steps.

    Attributes
    ----------
    qid : str
        Question identifier.
    buzz_step : int
        Step at which the agent buzzed (-1 if never buzzed voluntarily).
    buzz_index : int
        Index of the chosen answer option (0-based, -1 if forced).
    gold_index : int
        Index of the correct answer option (0-based).
    correct : bool
        Whether the agent selected the correct answer.
    episode_reward : float
        Total accumulated reward over the episode.
    c_trace : list[float]
        Per-step buzz probability: 1 - P(wait) at each timestep.
    p_correct_trace : list[float]
        Per-step correctness probability: P(gold_option) / P(buzz).
    entropy_trace : list[float]
        Per-step policy entropy over the full action distribution.
    """

    qid: str
    buzz_step: int
    buzz_index: int
    gold_index: int
    correct: bool
    episode_reward: float
    c_trace: list[float]
    p_correct_trace: list[float]
    entropy_trace: list[float]

    @property
    def g_trace(self) -> list[float]:
        return self.p_correct_trace


class PPOBuzzer:
    """PPO-trained buzzer agent wrapping Stable-Baselines3's PPO.

    Trains an MLP policy on belief-feature observations (Box(K+6,)) from
    TossupMCEnv. The policy maps observation vectors to a Discrete(K+1)
    action space: WAIT (0) or BUZZ with option i (1..K).

    Parameters
    ----------
    env : TossupMCEnv
        Gymnasium environment with belief-feature observations.
    learning_rate : float
        Learning rate for the Adam optimizer.
    n_steps : int
        Number of steps per rollout buffer collection.
    batch_size : int
        Minibatch size for PPO updates.
    n_epochs : int
        Number of optimization epochs per rollout.
    gamma : float
        Discount factor for return computation.
    policy_kwargs : dict or None
        Additional keyword arguments for the MLP policy. Defaults to
        ``{"net_arch": [64, 64]}`` (two hidden layers of 64 units).
    verbose : int
        SB3 verbosity level (0=silent, 1=info, 2=debug).
    """

    def __init__(
        self,
        env: TossupMCEnv,
        learning_rate: float = 3e-4,
        n_steps: int = 128,
        batch_size: int = 32,
        n_epochs: int = 10,
        gamma: float = 0.99,
        policy_kwargs: dict[str, Any] | None = None,
        verbose: int = 0,
    ):
        if policy_kwargs is None:
            policy_kwargs = {"net_arch": [64, 64]}

        self.env = env
        self.model = PPO(
            "MlpPolicy",
            env,
            verbose=verbose,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            policy_kwargs=policy_kwargs,
        )

    def train(self, total_timesteps: int = 100_000) -> None:
        """Train the PPO policy for the specified number of timesteps.

        Parameters
        ----------
        total_timesteps : int
            Total environment steps to collect during training.
        """
        self.model.learn(total_timesteps=total_timesteps)

    def save(self, path: str | Path) -> None:
        """Save the trained PPO model to disk.

        Parameters
        ----------
        path : str or Path
            File path for the saved model (SB3 appends .zip if needed).
        """
        self.model.save(str(path))

    @classmethod
    def load(cls, path: str | Path, env: TossupMCEnv) -> "PPOBuzzer":
        """Load a previously saved PPO model.

        Parameters
        ----------
        path : str or Path
            Path to the saved model file.
        env : TossupMCEnv
            Environment to attach to the loaded model.

        Returns
        -------
        PPOBuzzer
            A PPOBuzzer with the loaded model weights.
        """
        agent = cls(env=env)
        agent.model = PPO.load(str(path), env=env)
        return agent

    def action_probabilities(self, obs: np.ndarray) -> np.ndarray:
        """Extract action probabilities from the policy for a given observation.

        Parameters
        ----------
        obs : np.ndarray
            Observation vector of shape (K + 6,).

        Returns
        -------
        np.ndarray
            Action probability vector of shape (K + 1,), dtype float32.
            Index 0 = P(wait), indices 1..K = P(buzz with option i).
        """
        obs_tensor = th.as_tensor(
            obs, dtype=th.float32, device=self.model.device
        ).unsqueeze(0)
        dist = self.model.policy.get_distribution(obs_tensor)
        probs = dist.distribution.probs[0].detach().cpu().numpy()
        return probs.astype(np.float32)

    def c_t(self, obs: np.ndarray) -> float:
        """Compute buzz probability at the current step.

        Parameters
        ----------
        obs : np.ndarray
            Observation vector of shape (K + 6,).

        Returns
        -------
        float
            Probability of buzzing: 1 - P(wait). Range [0, 1].
        """
        probs = self.action_probabilities(obs)
        return float(1.0 - probs[0])

    def g_t(self, obs: np.ndarray, gold_index: int) -> float:
        """Compute correctness probability at the current step.

        Given that the agent buzzes, what is the probability it selects
        the correct answer? Formally: P(gold_action) / P(buzz).

        Parameters
        ----------
        obs : np.ndarray
            Observation vector of shape (K + 6,).
        gold_index : int
            Index of the correct answer option (0-based).

        Returns
        -------
        float
            Conditional correctness probability. Returns 0.0 if buzz
            probability is near zero (< 1e-12).
        """
        probs = self.action_probabilities(obs)
        c_t = float(1.0 - probs[0])
        if c_t <= 1e-12:
            return 0.0
        return float(probs[gold_index + 1] / c_t)

    def run_episode(
        self, deterministic: bool = False, seed: int | None = None
    ) -> PPOEpisodeTrace:
        """Run a full episode and record per-step action probability traces.

        Executes the policy in the environment, computing c_trace (buzz
        probability), p_correct_trace (correctness probability), and entropy_trace
        at each step. These traces are needed to compute the S_q metric.

        Parameters
        ----------
        deterministic : bool
            If True, select actions by argmax instead of sampling.
        seed : int or None
            If provided, seeds the environment reset for reproducibility.

        Returns
        -------
        PPOEpisodeTrace
            Complete episode record with action traces and outcome.
        """
        obs, info = self.env.reset(seed=seed)
        terminated = False
        truncated = False
        total_reward = 0.0
        c_trace: list[float] = []
        p_correct_trace: list[float] = []
        entropy_trace: list[float] = []

        buzz_step = -1
        buzz_index = -1
        base_env = self.env.unwrapped
        gold_index = base_env.question.gold_index if getattr(base_env, "question", None) is not None else -1

        while not (terminated or truncated):
            probs = self.action_probabilities(obs)
            if len(probs) == 2:
                c_val = float(probs[1])
                g_val = float(base_env.belief[gold_index]) if gold_index >= 0 else 0.0
            else:
                c_val = float(1.0 - probs[0])
                g_val = float(probs[gold_index + 1] / c_val) if c_val > 1e-12 else 0.0
            entropy = float(
                -(np.clip(probs, 1e-12, 1.0) * np.log(np.clip(probs, 1e-12, 1.0))).sum()
            )

            c_trace.append(c_val)
            p_correct_trace.append(g_val)
            entropy_trace.append(entropy)

            if deterministic:
                action = int(np.argmax(probs))
            else:
                action = int(np.random.choice(len(probs), p=probs))

            obs, reward, terminated, truncated, step_info = self.env.step(action)
            total_reward += reward

            if action != 0 and buzz_step < 0:
                buzz_step = int(step_info.get("step_idx", 0))
                buzz_index = int(step_info.get("chosen_idx", action - 1))
            if truncated and buzz_step < 0:
                buzz_step = int(step_info.get("step_idx", len(c_trace) - 1))
                if bool(step_info.get("no_buzz", False)):
                    buzz_index = -1
                else:
                    buzz_index = int(step_info.get("forced_choice", np.argmax(base_env.belief)))

        correct = buzz_index == gold_index
        return PPOEpisodeTrace(
            qid=info.get("qid", ""),
            buzz_step=buzz_step,
            buzz_index=buzz_index,
            gold_index=gold_index,
            correct=correct,
            episode_reward=total_reward,
            c_trace=c_trace,
            p_correct_trace=p_correct_trace,
            entropy_trace=entropy_trace,
        )
