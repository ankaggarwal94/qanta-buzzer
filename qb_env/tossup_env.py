"""
Gymnasium-compliant POMDP Environment for Quiz Bowl

Implements a tossup question environment where clues are revealed incrementally.
At each step the agent observes a belief-based feature vector and chooses either
to WAIT (action 0, reveals next clue) or to BUZZ with a specific answer option
(actions 1..K, ends the episode).

The environment computes beliefs over K answer options using a pluggable
LikelihoodModel and converts them to observations via extract_belief_features.

Ported from qb-rl reference implementation (qb_env/tossup_env.py) and adapted
for the unified qanta-buzzer codebase.
"""

from __future__ import annotations

import random
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from models.features import extract_belief_features
from models.likelihoods import LikelihoodModel
from qb_data.mc_builder import MCQuestion


class TossupMCEnv(gym.Env[np.ndarray, int]):
    """Gymnasium environment for quiz bowl tossup questions with MC options.

    Models quiz bowl as a POMDP where clues are revealed incrementally.
    The agent maintains a belief distribution over K answer options, updated
    at each step by a likelihood model. The agent decides when to buzz and
    which answer to select.

    Action Space
    ------------
    Discrete(K + 1):
        - 0: WAIT -- reveal the next clue and update belief
        - 1..K: BUZZ with answer option (i-1), ending the episode

    Observation Space
    -----------------
    Box(K + 6,):
        Belief features: [belief[0..K-1], top_p, margin, entropy,
        stability, progress, clue_idx_norm].
        See ``models.features.extract_belief_features`` for details.

    Reward Modes
    ------------
    ``time_penalty`` (default):
        -wait_penalty per WAIT step; +buzz_correct for correct buzz,
        +buzz_incorrect (negative) for wrong buzz.
    ``simple``:
        +1.0 for correct buzz, -1.0 for incorrect buzz, no WAIT penalty.
    ``human_grounded``:
        0.0 if the agent buzzes after the sampled human buzz position;
        otherwise +buzz_correct/-buzz_incorrect for correct/incorrect.

    Belief Modes
    ------------
    ``from_scratch``:
        Recompute belief from all clues seen so far via cumulative_prefixes.
    ``sequential_bayes``:
        Bayesian update: multiply prior belief by likelihood of new clue
        fragment, then normalize.

    Parameters
    ----------
    questions : list[MCQuestion]
        Pool of questions to sample from. Must be non-empty.
    likelihood_model : LikelihoodModel
        Model that scores clue text against answer option profiles.
    K : int
        Number of answer options per question. Must be >= 2.
    reward_mode : str
        One of ``"time_penalty"``, ``"simple"``, ``"human_grounded"``.
    wait_penalty : float
        Per-step penalty when reward_mode is ``"time_penalty"``.
    buzz_correct : float
        Reward for buzzing with the correct answer.
    buzz_incorrect : float
        Reward (typically negative) for buzzing with an incorrect answer.
    belief_mode : str
        One of ``"from_scratch"``, ``"sequential_bayes"``.
    beta : float
        Softmax temperature for converting raw scores to probabilities.
        Higher values produce sharper distributions.
    seed : int
        Random seed for question sampling and human buzz simulation.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        questions: list[MCQuestion],
        likelihood_model: LikelihoodModel,
        K: int = 4,
        reward_mode: str = "time_penalty",
        wait_penalty: float = 0.01,
        buzz_correct: float = 1.0,
        buzz_incorrect: float = -0.5,
        belief_mode: str = "from_scratch",
        beta: float = 5.0,
        seed: int = 13,
    ) -> None:
        if not questions:
            raise ValueError("questions cannot be empty")
        if K < 2:
            raise ValueError("K must be >= 2")

        self.questions = questions
        self.likelihood_model = likelihood_model
        self.K = K
        self.reward_mode = reward_mode
        self.wait_penalty = wait_penalty
        self.buzz_correct = buzz_correct
        self.buzz_incorrect = buzz_incorrect
        self.belief_mode = belief_mode
        self.beta = beta
        self.rng = random.Random(seed)

        self.action_space = spaces.Discrete(self.K + 1)
        # belief[K] + (top_p, margin, entropy, stability, progress, clue_idx)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.K + 6,), dtype=np.float32
        )

        self.question: MCQuestion | None = None
        self.step_idx: int = 0
        self.prev_belief: np.ndarray | None = None
        self.belief: np.ndarray = np.ones(self.K, dtype=np.float32) / self.K
        self.terminated: bool = False
        self.truncated: bool = False
        self._sampled_human_buzz_pos: int | None = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def total_steps(self) -> int:
        """Total number of incremental clue steps for the current question.

        Returns
        -------
        int
            Length of ``question.run_indices`` if a question is loaded, else 1.
        """
        if self.question is None:
            return 1
        return len(self.question.run_indices)

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def _sample_question(self) -> MCQuestion:
        """Sample a random question from the question pool.

        Returns
        -------
        MCQuestion
            A randomly selected question.
        """
        return self.rng.choice(self.questions)

    def _sample_human_buzz(self, question: MCQuestion) -> int | None:
        """Sample a human buzz position from the question's distribution.

        Uses weighted random sampling based on the number of humans who
        buzzed at each position. Returns None if no human buzz data exists.

        Parameters
        ----------
        question : MCQuestion
            The question to sample a human buzz position for.

        Returns
        -------
        int or None
            Sampled token position, or None if no human buzz data.
        """
        if not question.human_buzz_positions:
            return None
        positions = []
        weights = []
        for pos, count in question.human_buzz_positions:
            positions.append(int(pos))
            weights.append(max(1, int(count)))
        if not positions:
            return None
        return self.rng.choices(positions, weights=weights, k=1)[0]

    def _softmax_scores(self, scores: np.ndarray) -> np.ndarray:
        """Convert raw likelihood scores to a probability distribution.

        Applies a temperature-scaled softmax with numerical stability
        (subtract max before exponentiation). Falls back to uniform
        distribution if the sum of exponentiated scores is non-positive.

        Parameters
        ----------
        scores : np.ndarray
            Raw similarity scores of shape (K,).

        Returns
        -------
        np.ndarray
            Probability distribution of shape (K,), dtype float32.
        """
        stable = scores - np.max(scores)
        probs = np.exp(self.beta * stable)
        probs_sum = np.sum(probs)
        if probs_sum <= 0:
            return np.ones_like(scores, dtype=np.float32) / len(scores)
        return (probs / probs_sum).astype(np.float32)

    def _compute_belief(self, question: MCQuestion, step_idx: int) -> np.ndarray:
        """Compute belief distribution over answer options at a given step.

        Two modes are supported:

        ``from_scratch``
            Score the cumulative clue prefix against all option profiles,
            then apply softmax. Each step is independent of the previous
            belief.

        ``sequential_bayes``
            Extract only the new clue fragment since the last step, score
            it, and perform a Bayesian update: posterior = prior * likelihood,
            then normalize. This is cheaper per step but may accumulate
            approximation errors.

        Parameters
        ----------
        question : MCQuestion
            Current question being played.
        step_idx : int
            Current step index (0-based, indexes into run_indices).

        Returns
        -------
        np.ndarray
            Updated belief distribution of shape (K,), dtype float32.

        Raises
        ------
        ValueError
            If ``self.belief_mode`` is not a recognized mode.
        """
        if self.belief_mode == "from_scratch":
            prefix = question.cumulative_prefixes[step_idx]
            scores = self.likelihood_model.score(prefix, question.option_profiles)
            return self._softmax_scores(scores)

        if self.belief_mode == "sequential_bayes":
            idx = question.run_indices[step_idx]
            prev_idx = question.run_indices[step_idx - 1] if step_idx > 0 else -1
            frag = " ".join(question.tokens[prev_idx + 1 : idx + 1])
            scores = self.likelihood_model.score(frag, question.option_profiles)
            likelihood = self._softmax_scores(scores)
            posterior = self.belief * likelihood
            denom = posterior.sum()
            if denom <= 0:
                posterior = np.ones(self.K, dtype=np.float32) / self.K
            else:
                posterior = posterior / denom
            return posterior.astype(np.float32)

        raise ValueError(f"Unknown belief_mode: {self.belief_mode}")

    def _obs(self) -> np.ndarray:
        """Build the observation vector from current belief state.

        Delegates to ``extract_belief_features`` which concatenates the raw
        belief vector with 6 derived scalar features.

        Returns
        -------
        np.ndarray
            Feature vector of shape (K + 6,), dtype float32.
        """
        return extract_belief_features(
            belief=self.belief,
            prev_belief=self.prev_belief,
            step_idx=self.step_idx,
            total_steps=self.total_steps,
        )

    def _step_to_token_pos(self, step_idx: int) -> int:
        """Convert a step index to the corresponding token position.

        Used by the ``human_grounded`` reward mode to compare the agent's
        buzz position against the sampled human buzz position.

        Parameters
        ----------
        step_idx : int
            Step index (0-based, indexes into run_indices).

        Returns
        -------
        int
            Token position in the original question text.
        """
        if self.question is None or not self.question.run_indices:
            return step_idx
        if step_idx >= len(self.question.run_indices):
            return self.question.run_indices[-1]
        if step_idx < 0:
            return self.question.run_indices[0]
        return self.question.run_indices[step_idx]

    def _buzz_reward(self, question: MCQuestion, chosen_idx: int, last_seen_step: int) -> float:
        """Compute the reward for buzzing with a given answer.

        Dispatches on ``self.reward_mode``:

        ``simple``
            +1.0 for correct, -1.0 for incorrect.
        ``human_grounded``
            0.0 if the agent buzzes after the sampled human would have;
            otherwise +buzz_correct / +buzz_incorrect.
        ``time_penalty`` (default)
            +buzz_correct / +buzz_incorrect. The per-step wait penalty
            is applied separately in ``step()``.

        Parameters
        ----------
        question : MCQuestion
            Current question.
        chosen_idx : int
            Index of the chosen answer option (0-based).
        last_seen_step : int
            Step index of the last clue seen before buzzing.

        Returns
        -------
        float
            Reward value.
        """
        correct = chosen_idx == question.gold_index
        if self.reward_mode == "simple":
            return 1.0 if correct else -1.0
        if self.reward_mode == "human_grounded":
            token_pos = self._step_to_token_pos(last_seen_step)
            if self._sampled_human_buzz_pos is not None and token_pos > self._sampled_human_buzz_pos:
                return 0.0
            return self.buzz_correct if correct else self.buzz_incorrect
        # default: time_penalty
        return self.buzz_correct if correct else self.buzz_incorrect

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the environment and start a new episode.

        Samples a random question from the pool, initializes belief to a
        uniform distribution, and returns the initial observation.

        Parameters
        ----------
        seed : int or None
            If provided, reseeds both the internal RNG and numpy's global
            RNG for reproducibility.
        options : dict or None
            Unused. Included for Gymnasium API compatibility.

        Returns
        -------
        observation : np.ndarray
            Initial observation of shape (K + 6,), dtype float32.
            Belief is uniform, so top_p = 1/K, margin = 0, entropy = max.
        info : dict[str, Any]
            Episode metadata. Contains ``"qid"`` (the sampled question ID).
        """
        if seed is not None:
            self.rng.seed(seed)
            np.random.seed(seed)

        self.question = self._sample_question()
        self.step_idx = 0
        self.prev_belief = None
        self.belief = np.ones(self.K, dtype=np.float32) / self.K
        self.terminated = False
        self.truncated = False
        self._sampled_human_buzz_pos = self._sample_human_buzz(self.question)
        return self._obs(), {"qid": self.question.qid}

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Execute one step in the environment.

        If ``action == 0`` (WAIT):
            - Saves previous belief, computes new belief from current clue.
            - Applies wait_penalty if reward_mode is ``"time_penalty"``.
            - Advances step counter.
            - If all clues exhausted: forced termination with best-guess
              answer (``truncated=True``).

        If ``action in 1..K`` (BUZZ):
            - Computes buzz reward for chosen answer option ``action - 1``.
            - Episode ends (``terminated=True``).

        Parameters
        ----------
        action : int
            Action to take. 0 = WAIT, 1..K = buzz with option (action-1).

        Returns
        -------
        observation : np.ndarray
            Updated observation of shape (K + 6,), dtype float32.
        reward : float
            Scalar reward for this step.
        terminated : bool
            True if the agent buzzed (natural episode end).
        truncated : bool
            True if all clues were exhausted (forced termination).
        info : dict[str, Any]
            Step metadata. Always contains ``"qid"`` and ``"step_idx"``.
            On BUZZ: also ``"chosen_idx"`` and ``"correct"``.
            On forced termination: also ``"forced_choice"`` and
            ``"forced_correct"``.

        Raises
        ------
        RuntimeError
            If called before ``reset()`` or after episode has ended.
        ValueError
            If ``action`` is not in the action space.
        """
        if self.question is None:
            raise RuntimeError("Environment must be reset() before step().")
        if self.terminated or self.truncated:
            raise RuntimeError("Cannot call step() on terminated/truncated episode.")
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")

        info: dict[str, Any] = {"qid": self.question.qid}
        reward = 0.0

        if action == 0:
            # WAIT: reveal next clue and update belief
            self.prev_belief = self.belief.copy()
            self.belief = self._compute_belief(self.question, self.step_idx)
            if self.reward_mode == "time_penalty":
                reward -= self.wait_penalty

            self.step_idx += 1
            if self.step_idx >= self.total_steps:
                # Forced termination: pick best answer from current belief
                last_seen = self.step_idx - 1
                forced_choice = int(np.argmax(self.belief))
                reward += self._buzz_reward(self.question, forced_choice, last_seen)
                self.truncated = True
                info["step_idx"] = last_seen
                info["forced_choice"] = forced_choice
                info["forced_correct"] = forced_choice == self.question.gold_index
            else:
                info["step_idx"] = self.step_idx

        else:
            # BUZZ: select an answer option
            last_seen = max(0, self.step_idx - 1)
            chosen_idx = action - 1
            reward += self._buzz_reward(self.question, chosen_idx, last_seen)
            self.terminated = True
            info["step_idx"] = last_seen
            info["chosen_idx"] = chosen_idx
            info["correct"] = chosen_idx == self.question.gold_index

        obs = self._obs()
        return obs, float(reward), self.terminated, self.truncated, info


def make_env_from_config(
    mc_questions: list[MCQuestion],
    likelihood_model: LikelihoodModel,
    config: dict[str, Any],
) -> TossupMCEnv:
    """Construct a TossupMCEnv from YAML configuration.

    Factory function that reads the ``environment``, ``data``, and
    ``likelihood`` sections of a config dict and instantiates a fully
    configured environment. The likelihood model must be pre-constructed
    (e.g., via ``build_likelihood_from_config``).

    Parameters
    ----------
    mc_questions : list[MCQuestion]
        List of MCQuestion instances with options and answer profiles.
        Must be non-empty.
    likelihood_model : LikelihoodModel
        Pre-constructed likelihood model for scoring clues against options.
        Use ``build_likelihood_from_config`` to create one from config.
    config : dict[str, Any]
        Full YAML config dict. Must contain the following sections:

        - ``environment``: reward mode, penalties, belief mode
        - ``data``: K (number of answer choices)
        - ``likelihood``: beta (softmax temperature)

    Returns
    -------
    TossupMCEnv
        A configured Gymnasium environment ready for ``reset()``.

    Examples
    --------
    >>> from qb_data.config import load_config
    >>> from models.likelihoods import build_likelihood_from_config
    >>> config = load_config("configs/default.yaml")
    >>> model = build_likelihood_from_config(config, corpus_texts=corpus)
    >>> env = make_env_from_config(mc_questions, model, config)
    >>> obs, info = env.reset()
    """
    env_cfg = config["environment"]
    data_cfg = config["data"]
    lik_cfg = config["likelihood"]
    return TossupMCEnv(
        questions=mc_questions,
        likelihood_model=likelihood_model,
        K=int(data_cfg.get("K", 4)),
        reward_mode=str(env_cfg.get("reward", env_cfg.get("reward_mode", "time_penalty"))),
        wait_penalty=float(env_cfg.get("wait_penalty", 0.01)),
        buzz_correct=float(env_cfg.get("buzz_correct", 1.0)),
        buzz_incorrect=float(env_cfg.get("buzz_incorrect", -0.5)),
        belief_mode=str(env_cfg.get("belief_mode", "from_scratch")),
        beta=float(lik_cfg.get("beta", 5.0)),
    )
