"""PPO Buzzer agent wrapping Stable-Baselines3's PPO.

Provides the PPOBuzzer class for training an MLP policy on belief-feature
observations from TossupMCEnv, and PPOEpisodeTrace for recording per-step
action probabilities needed to compute the S_q scoring metric.

The key design rationale: SB3's ``learn()`` does not expose per-step action
distributions, so ``run_episode()`` implements custom episode execution that
records c_trace (buzz probability) and g_trace (correctness probability)
at each step for downstream S_q computation.

Ported from qb-rl reference implementation (agents/ppo_buzzer.py) with
import path adaptations for the unified qanta-buzzer codebase.
"""

from __future__ import annotations

from dataclasses import KW_ONLY, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch as th
from stable_baselines3 import PPO

from qb_env.tossup_env import TossupMCEnv


@dataclass
class PPOEpisodeTrace:
    """Record of a single episode with per-step action probability traces.

    Used to compute the S_q scoring metric: S_q = sum(c_t * g_t) over steps,
    and calibration metrics (ECE, Brier) via ``top_p_trace``.

    Attributes
    ----------
    qid : str
        Question identifier.
    buzz_step : int
        Step at which the agent buzzed (-1 if never buzzed voluntarily).
    buzz_trace_idx : int
        Index into ``c_trace``/``g_trace``/``top_p_trace`` for the
        decision-time trace entry when the policy buzzed (-1 otherwise).
    buzz_index : int
        Index of the chosen answer option (0-based, -1 if the policy
        never buzzed voluntarily).
    gold_index : int
        Index of the correct answer option (0-based).
    correct : bool
        Whether the policy buzzed and selected the correct answer.
    forced_commit : bool
        Whether the episode ended via environment force-commit at truncation.
    forced_step : int
        Clue index associated with the force-commit decision (-1 otherwise).
    forced_choice : int
        Answer index selected by the environment at truncation (-1 otherwise).
    forced_correct : bool
        Whether the environment's force-commit choice was correct.
    episode_reward : float
        Total accumulated reward over the episode.
    c_trace : list[float]
        Per-step buzz probability: 1 - P(wait) at each timestep.
    g_trace : list[float]
        Per-step correctness probability: P(gold_option) / P(buzz).
    top_p_trace : list[float]
        Per-step max belief probability: max(env.belief). Used as the
        confidence proxy for calibration metrics, consistent with
        baseline agents.
    entropy_trace : list[float]
        Per-step policy entropy over the full action distribution.
    """

    # Required core fields (legacy shape; positional construction
    # preserved for callers that match the pre-2026-05 signature).
    qid: str
    buzz_step: int
    buzz_index: int
    gold_index: int
    correct: bool
    episode_reward: float
    c_trace: list[float] = field(default_factory=list)
    g_trace: list[float] = field(default_factory=list)
    top_p_trace: list[float] = field(default_factory=list)
    entropy_trace: list[float] = field(default_factory=list)
    # Forced-commit / trace-index extensions (added 2026-05) are
    # keyword-only so positional callers that match the legacy field
    # order continue to work and callers that matched any intermediate
    # ordering get a ``TypeError`` instead of silently mis-binding
    # into ``buzz_index``. Legacy artifacts that lack these fields
    # are read with the same sentinels via ``dict.get(..., default)``
    # in metrics.py.
    _: KW_ONLY
    buzz_trace_idx: int = -1
    forced_commit: bool = False
    forced_step: int = -1
    forced_choice: int = -1
    forced_correct: bool = False


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
        seed: int | None = None,
        policy_kwargs: dict[str, Any] | None = None,
        verbose: int = 0,
        use_maskable_ppo: bool = False,
    ):
        if policy_kwargs is None:
            policy_kwargs = {"net_arch": [64, 64]}

        self.env = env
        self._use_maskable = use_maskable_ppo

        if use_maskable_ppo:
            try:
                from sb3_contrib import MaskablePPO
            except ImportError as exc:
                raise ImportError(
                    "MaskablePPO requires sb3-contrib. "
                    "Install with: pip install -e '.[maskable]'"
                ) from exc
            self.model = MaskablePPO(
                "MlpPolicy",
                env,
                verbose=verbose,
                seed=seed,
                learning_rate=learning_rate,
                n_steps=n_steps,
                batch_size=batch_size,
                n_epochs=n_epochs,
                gamma=gamma,
                policy_kwargs=policy_kwargs,
            )
        else:
            self.model = PPO(
                "MlpPolicy",
                env,
                verbose=verbose,
                seed=seed,
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

        The checkpoint does not record whether it was trained with PPO or
        MaskablePPO. Callers must pass ``use_maskable_ppo`` to ``load()``
        matching the training configuration.

        Parameters
        ----------
        path : str or Path
            File path for the saved model (SB3 appends .zip if needed).
        """
        self.model.save(str(path))

    @classmethod
    def load(
        cls,
        path: str | Path,
        env: TossupMCEnv,
        use_maskable_ppo: bool = False,
    ) -> "PPOBuzzer":
        """Load a previously saved PPO model.

        Parameters
        ----------
        path : str or Path
            Path to the saved model file.
        env : TossupMCEnv
            Environment to attach to the loaded model.
        use_maskable_ppo : bool
            If True, load with ``MaskablePPO`` from sb3-contrib instead
            of plain SB3 ``PPO``.

        Returns
        -------
        PPOBuzzer
            A PPOBuzzer with the loaded model weights.
        """
        agent = cls.__new__(cls)
        agent.env = env
        agent._use_maskable = use_maskable_ppo
        if use_maskable_ppo:
            try:
                from sb3_contrib import MaskablePPO
            except ImportError as exc:
                raise ImportError(
                    "MaskablePPO requires sb3-contrib. "
                    "Install with: pip install -e '.[maskable]'"
                ) from exc
            agent.model = MaskablePPO.load(str(path), env=env)
        else:
            agent.model = PPO.load(str(path), env=env)
        return agent

    def _current_action_masks(self) -> np.ndarray | None:
        """Return action masks from the env, or None if not maskable."""
        if not self._use_maskable:
            return None
        env_for_mask = self.env if hasattr(self.env, "action_masks") else self._base_env()
        if not hasattr(env_for_mask, "action_masks"):
            return None
        return np.asarray(env_for_mask.action_masks(), dtype=bool)

    def action_probabilities(self, obs: np.ndarray) -> np.ndarray:
        """Extract action probabilities from the policy for a given observation.

        When ``use_maskable_ppo=True``, passes ``action_masks`` to the
        policy distribution so that probabilities for invalid actions are
        zeroed out before action selection.

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

        masks = self._current_action_masks()
        dist = self.model.policy.get_distribution(obs_tensor)
        if masks is not None:
            masks_tensor = th.as_tensor(
                masks, dtype=th.bool, device=self.model.device
            ).unsqueeze(0)
            dist.apply_masking(masks_tensor)

        probs = dist.distribution.probs[0].detach().cpu().numpy()
        return probs.astype(np.float32)

    def _base_env(self) -> TossupMCEnv:
        """Return the underlying TossupMCEnv, unwrapping if needed."""
        return getattr(self.env, "unwrapped", self.env)

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
        base_env = self._base_env()
        c_t = float(1.0 - probs[0])
        if c_t <= 1e-12:
            return 0.0
        if len(probs) == 2:
            if gold_index < 0 or base_env.belief is None:
                return 0.0
            return float(base_env.belief[gold_index])
        return float(probs[gold_index + 1] / c_t)

    def run_episode(
        self,
        deterministic: bool = False,
        seed: int | None = None,
        question_idx: int | None = None,
    ) -> PPOEpisodeTrace:
        """Run a full episode and record per-step action probability traces.

        Executes the policy in the environment, computing c_trace (buzz
        probability), g_trace (correctness probability), and entropy_trace
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
        reset_options = None
        if question_idx is not None:
            reset_options = {"question_idx": int(question_idx)}

        obs, info = self.env.reset(seed=seed, options=reset_options)
        terminated = False
        truncated = False
        total_reward = 0.0
        c_trace: list[float] = []
        g_trace: list[float] = []
        top_p_trace: list[float] = []
        entropy_trace: list[float] = []

        base_env = self._base_env()
        buzz_step = -1
        buzz_index = -1
        gold_index = (
            base_env.question.gold_index
            if getattr(base_env, "question", None) is not None
            else -1
        )
        buzz_trace_idx = -1
        forced_commit = False
        forced_step = -1
        forced_choice = -1
        forced_correct = False
        # Always use a Generator. When ``seed is None``,
        # ``np.random.default_rng`` draws fresh OS entropy, so the
        # Generator is independent per call but NOT deterministic;
        # the win is that we avoid mutating the global ``np.random``
        # state (the documented mixed-RNG-API trap). Pass an explicit
        # ``seed`` for reproducibility.
        sample_rng = np.random.default_rng(seed)

        while not (terminated or truncated):
            probs = self.action_probabilities(obs)
            c_val = float(probs[1] if len(probs) == 2 else 1.0 - probs[0])
            if len(probs) == 2:
                g_val = (
                    float(base_env.belief[gold_index])
                    if gold_index >= 0 and base_env.belief is not None
                    else 0.0
                )
            else:
                g_val = (
                    float(probs[gold_index + 1] / c_val) if c_val > 1e-12 else 0.0
                )
            entropy = float(
                -(np.clip(probs, 1e-12, 1.0) * np.log(np.clip(probs, 1e-12, 1.0))).sum()
            )

            top_p_val = float(np.max(base_env.belief)) if base_env.belief is not None else c_val
            c_trace.append(c_val)
            g_trace.append(g_val)
            top_p_trace.append(top_p_val)
            entropy_trace.append(entropy)

            if deterministic:
                action = int(np.argmax(probs))
            else:
                action = int(sample_rng.choice(len(probs), p=probs))

            obs, reward, terminated, truncated, step_info = self.env.step(action)
            total_reward += reward

            if action != 0 and buzz_step < 0:
                buzz_step = int(step_info.get("step_idx", 0))
                buzz_trace_idx = len(c_trace) - 1
                if len(probs) == 2:
                    buzz_index = int(
                        step_info.get(
                            "chosen_idx",
                            step_info.get("forced_choice", np.argmax(base_env.belief)),
                        )
                    )
                else:
                    buzz_index = action - 1
            if truncated and buzz_step < 0 and not step_info.get("no_buzz", False):
                forced_commit = True
                forced_step = int(
                    step_info.get("step_idx", len(c_trace) - 1)
                )
                forced_choice = int(
                    step_info.get("forced_choice", np.argmax(base_env.belief))
                )
                forced_correct = bool(step_info.get("forced_correct", False))

        correct = buzz_index >= 0 and buzz_index == gold_index
        return PPOEpisodeTrace(
            qid=info.get("qid", ""),
            buzz_step=buzz_step,
            buzz_trace_idx=buzz_trace_idx,
            buzz_index=buzz_index,
            gold_index=gold_index,
            correct=correct,
            forced_commit=forced_commit,
            forced_step=forced_step,
            forced_choice=forced_choice,
            forced_correct=forced_correct,
            episode_reward=total_reward,
            c_trace=c_trace,
            g_trace=g_trace,
            top_p_trace=top_p_trace,
            entropy_trace=entropy_trace,
        )
