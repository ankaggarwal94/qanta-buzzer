"""
TextObservationWrapper for converting belief features to text observations.

Wraps TossupMCEnv to provide text-formatted observations (clues + choices)
instead of numeric belief feature vectors. This bridges the gap between
the environment's native observation space (Box(K+6,)) and T5PolicyModel's
text input requirement.

The underlying environment still operates on beliefs internally for reward
computation -- the wrapper only transforms what the agent SEES, not how the
environment computes rewards or transitions.

Text format matches T5PolicyModel's expected input:
    "CLUES: clue1 clue2 ... | CHOICES: (1) ans1 (2) ans2 (3) ans3 (4) ans4"

Ported from qanta-buzzer's environment.py get_text_representation() method,
adapted for the unified codebase's Gymnasium wrapper pattern.
"""

from __future__ import annotations

from typing import Any, Tuple

import gymnasium as gym
import numpy as np

from qb_data.mc_builder import MCQuestion


class TextObservationWrapper(gym.ObservationWrapper):
    """Wrap TossupMCEnv to provide text observations instead of belief features.

    The underlying env still operates on beliefs internally (for reward
    computation), but the agent sees text-formatted observations for T5 input.
    This is a Gymnasium ObservationWrapper that intercepts the observation
    returned by reset() and step() and converts it to a text string.

    The observation space is set to a placeholder Box(1,) since Gymnasium
    requires a defined space, but text observations are variable-length
    strings. Downstream code (T5PolicyModel) handles tokenization.

    Parameters
    ----------
    env : gym.Env
        The underlying TossupMCEnv instance. Must have ``question``
        (MCQuestion) and ``step_idx`` (int) attributes.

    Examples
    --------
    >>> from qb_env.tossup_env import TossupMCEnv
    >>> env = TossupMCEnv(questions=qs, likelihood_model=lm, K=4)
    >>> wrapped = TextObservationWrapper(env)
    >>> obs, info = wrapped.reset()
    >>> assert isinstance(obs, str)
    >>> assert "CLUES:" in obs and "CHOICES:" in obs
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        # Override observation space with a placeholder.
        # Text observations are variable-length strings; Gymnasium requires
        # a Space object, so we use a minimal Box as a sentinel.
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(1,), dtype=np.float32
        )

    def observation(self, obs: np.ndarray) -> str:
        """Convert numeric belief observation to formatted text string.

        Reconstructs visible clues from the underlying environment's current
        question and step index, then formats them with answer choices in the
        standard T5PolicyModel input format.

        Parameters
        ----------
        obs : np.ndarray
            Numeric belief features from the underlying environment.
            Shape ``(K+6,)``. Not used directly -- the text is reconstructed
            from ``env.question`` and ``env.step_idx``.

        Returns
        -------
        str
            Formatted text observation:
            ``"CLUES: <visible clue tokens> | CHOICES: (1) opt1 (2) opt2 ..."``
        """
        question: MCQuestion = self.env.question
        step_idx: int = self.env.step_idx

        # Build visible clue text from cumulative prefixes.
        #
        # TossupMCEnv step semantics:
        #   - reset() sets step_idx=0, belief is uniform (no clues processed).
        #   - step(WAIT) calls _compute_belief(step_idx), THEN increments step_idx.
        #   - The observation returned after step() has step_idx ALREADY incremented.
        #
        # So step_idx tells us how many WAIT actions have been taken:
        #   step_idx=0: No WAITs yet; no clues processed; show minimal context
        #   step_idx=N: N WAITs taken; beliefs from cumulative_prefixes[0..N-1]
        #
        # cumulative_prefixes[i] = text of tokens[0..run_indices[i]].
        # After N WAITs, the agent has seen information up to
        # cumulative_prefixes[N-1], so that is what the text obs shows.
        if step_idx == 0:
            # No clues processed yet; show question start as minimal context
            # (matches initial observation having some textual content for T5)
            clues_text = question.tokens[0] if question.tokens else ""
        elif step_idx <= len(question.cumulative_prefixes):
            clues_text = question.cumulative_prefixes[step_idx - 1]
        else:
            # Past all clues (truncated episode); show all text
            clues_text = question.cumulative_prefixes[-1]

        # Format answer choices
        choices_parts = [
            f"({i + 1}) {opt}" for i, opt in enumerate(question.options)
        ]
        choices_text = " ".join(choices_parts)

        return f"CLUES: {clues_text} | CHOICES: {choices_text}"

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and return a text observation.

        Parameters
        ----------
        seed : int or None
            Random seed passed to underlying environment.
        options : dict or None
            Options passed to underlying environment.

        Returns
        -------
        observation : str
            Text-formatted initial observation.
        info : dict[str, Any]
            Episode metadata from underlying environment.
        """
        obs, info = self.env.reset(seed=seed, options=options)
        return self.observation(obs), info

    def step(
        self, action: int
    ) -> Tuple[str, float, bool, bool, dict[str, Any]]:
        """Execute one step and return text observation.

        Parameters
        ----------
        action : int
            Action to take. 0 = WAIT, 1..K = BUZZ with answer (action-1).

        Returns
        -------
        observation : str
            Text-formatted observation after the step.
        reward : float
            Scalar reward for this step.
        terminated : bool
            True if the agent buzzed (natural episode end).
        truncated : bool
            True if all clues exhausted (forced termination).
        info : dict[str, Any]
            Step metadata from underlying environment.
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self.observation(obs), reward, terminated, truncated, info

    @property
    def unwrapped_env(self):
        """Access the underlying TossupMCEnv directly.

        Returns
        -------
        TossupMCEnv
            The unwrapped environment instance.
        """
        return self.env
