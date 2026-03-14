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


def _softmax(scores: np.ndarray, beta: float) -> np.ndarray:
    """Temperature-scaled softmax with numerical stability.

    Parameters
    ----------
    scores : np.ndarray
        Raw similarity scores of shape (K,).
    beta : float
        Temperature parameter. Higher values produce sharper distributions.

    Returns
    -------
    np.ndarray
        Probability distribution of shape (K,), dtype float32.
    """
    stable = scores - np.max(scores)
    probs = np.exp(beta * stable)
    probs_sum = np.sum(probs)
    if probs_sum <= 0:
        return np.ones_like(scores, dtype=np.float32) / len(scores)
    return (probs / probs_sum).astype(np.float32)


def precompute_beliefs(
    questions: list[MCQuestion],
    likelihood_model: LikelihoodModel,
    belief_mode: str = "from_scratch",
    beta: float = 5.0,
    K: int = 4,
) -> dict[tuple[int, int], np.ndarray]:
    """Precompute belief trajectories for all questions and steps.

    Iterates over each question and each step index, computing the belief
    using the same logic as ``TossupMCEnv._compute_belief``. The result is
    a dict keyed by ``(question_index, step_idx)`` for O(1) lookup during
    training rollouts.

    Parameters
    ----------
    questions : list[MCQuestion]
        Pool of questions to precompute beliefs for.
    likelihood_model : LikelihoodModel
        Model that scores clue text against answer option profiles.
    belief_mode : str
        One of ``"from_scratch"``, ``"sequential_bayes"``.
    beta : float
        Softmax temperature for converting raw scores to probabilities.
    K : int
        Number of answer options per question.

    Returns
    -------
    dict[tuple[int, int], np.ndarray]
        Maps ``(question_index, step_idx)`` to belief vectors of shape
        ``(K,)`` with dtype float32. Each belief sums to ~1.0.
    """
    cache: dict[tuple[int, int], np.ndarray] = {}

    for q_idx, question in enumerate(questions):
        num_steps = len(question.run_indices)
        belief = np.ones(K, dtype=np.float32) / K

        for step_idx in range(num_steps):
            if belief_mode == "from_scratch":
                prefix = question.cumulative_prefixes[step_idx]
                scores = likelihood_model.score(prefix, question.option_profiles)
                belief = _softmax(scores, beta)

            elif belief_mode == "sequential_bayes":
                idx = question.run_indices[step_idx]
                prev_idx = question.run_indices[step_idx - 1] if step_idx > 0 else -1
                frag = " ".join(question.tokens[prev_idx + 1 : idx + 1])
                scores = likelihood_model.score(frag, question.option_profiles)
                likelihood = _softmax(scores, beta)
                posterior = belief * likelihood
                denom = posterior.sum()
                if denom <= 0:
                    belief = np.ones(K, dtype=np.float32) / K
                else:
                    belief = (posterior / denom).astype(np.float32)

            else:
                raise ValueError(f"Unknown belief_mode: {belief_mode}")

            cache[(q_idx, step_idx)] = belief.copy()

    return cache


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
        early_buzz_penalty: float = 0.0,
        buzz_correct: float = 1.0,
        buzz_incorrect: float = -0.5,
        belief_mode: str = "from_scratch",
        beta: float = 5.0,
        seed: int = 13,
        precomputed_beliefs: dict[tuple[int, int], np.ndarray] | None = None,
        opponent_buzz_model: "OpponentBuzzModel | None" = None,
        ew_reward_correct: float = 10.0,
        ew_reward_incorrect: float = -5.0,
        ew_opponent_expected_value: float = 0.0,
        variable_K: bool = False,
        max_K: int | None = None,
        use_action_masking: bool = False,
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
        self.early_buzz_penalty = early_buzz_penalty
        self.buzz_correct = buzz_correct
        self.buzz_incorrect = buzz_incorrect
        self.belief_mode = belief_mode
        self.beta = beta
        self.rng = random.Random(seed)
        self.precomputed_beliefs = precomputed_beliefs

        self.opponent_buzz_model = opponent_buzz_model
        self.ew_reward_correct = ew_reward_correct
        self.ew_reward_incorrect = ew_reward_incorrect
        self.ew_opponent_expected_value = ew_opponent_expected_value

        self.variable_K = variable_K
        self.use_action_masking = use_action_masking
        if variable_K:
            self._max_K = max_K or max(len(q.options) for q in questions)
        else:
            self._max_K = K

        # Build qid -> list-index map for precomputed belief lookups
        self._question_index_map: dict[str, int] = {
            q.qid: i for i, q in enumerate(questions)
        }

        obs_K = self._max_K if self.variable_K else self.K
        self.action_space = spaces.Discrete(obs_K + 1)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_K + 6,), dtype=np.float32
        )

        self.question: MCQuestion | None = None
        self.step_idx: int = 0
        self.prev_belief: np.ndarray | None = None
        self.belief: np.ndarray = np.ones(self.K, dtype=np.float32) / self.K
        self.terminated: bool = False
        self.truncated: bool = False
        self._sampled_human_buzz_pos: int | None = None
        self._current_question_idx: int = 0

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

        Delegates to module-level ``_softmax`` with this environment's beta.

        Parameters
        ----------
        scores : np.ndarray
            Raw similarity scores of shape (K,).

        Returns
        -------
        np.ndarray
            Probability distribution of shape (K,), dtype float32.
        """
        return _softmax(scores, self.beta)

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
        if self.precomputed_beliefs is not None:
            key = (self._current_question_idx, step_idx)
            return self.precomputed_beliefs[key].copy()

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

        In variable-K mode, uses padded features sized to ``_max_K``.
        Otherwise delegates to ``extract_belief_features``.

        Returns
        -------
        np.ndarray
            Feature vector of shape (obs_K + 6,), dtype float32.
        """
        if self.variable_K:
            from models.features import extract_padded_belief_features

            return extract_padded_belief_features(
                belief=self.belief,
                prev_belief=self.prev_belief,
                step_idx=self.step_idx,
                total_steps=self.total_steps,
                max_K=self._max_K,
            )
        return extract_belief_features(
            belief=self.belief,
            prev_belief=self.prev_belief,
            step_idx=self.step_idx,
            total_steps=self.total_steps,
        )

    def action_masks(self) -> np.ndarray:
        """Return a boolean mask of valid actions.

        WAIT (action 0) is always valid.  Buzz actions ``1..K_actual``
        are valid; padded slots ``K_actual+1..max_K`` are invalid.

        Returns
        -------
        np.ndarray
            Boolean array of shape ``(max_K + 1,)`` or ``(K + 1,)``.
        """
        n_actions = self._max_K + 1 if self.variable_K else self.K + 1
        mask = np.zeros(n_actions, dtype=bool)
        mask[0] = True  # WAIT
        k_actual = len(self.question.options) if self.question is not None else self.K
        mask[1 : k_actual + 1] = True
        return mask

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

    def _expected_wins_reward(
        self, question: MCQuestion, chosen_idx: int, last_seen_step: int
    ) -> float:
        """Compute Expected Wins reward at buzz time.

        R_t = S_t * V_self + (1 - S_t) * V_opp

        where S_t = P(opponent has NOT buzzed by step t).
        """
        correct = chosen_idx == question.gold_index
        v_self = self.ew_reward_correct if correct else self.ew_reward_incorrect
        if self.opponent_buzz_model is None:
            return v_self
        s_t = self.opponent_buzz_model.prob_survive_to_step(question, last_seen_step)
        return s_t * v_self + (1.0 - s_t) * self.ew_opponent_expected_value

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
        ``expected_wins``
            S_t * V_self + (1 - S_t) * V_opp via opponent model.

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
        if self.reward_mode == "expected_wins":
            return self._expected_wins_reward(question, chosen_idx, last_seen_step)
        # default: time_penalty
        reward = self.buzz_correct if correct else self.buzz_incorrect

        if self.early_buzz_penalty > 0 and self.total_steps > 1:
            progress = np.clip((last_seen_step + 1) / self.total_steps, 0.0, 1.0)
            reward -= float(self.early_buzz_penalty) * (1.0 - progress)

        return reward

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

        if options and "question_idx" in options:
            q_idx = int(options["question_idx"])
            if q_idx < 0 or q_idx >= len(self.questions):
                raise ValueError(f"question_idx out of range: {q_idx}")
            self.question = self.questions[q_idx]
            self._current_question_idx = q_idx
        else:
            self.question = self._sample_question()
            self._current_question_idx = self._question_index_map.get(
                self.question.qid, self.questions.index(self.question)
            )
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
    precomputed_beliefs: dict[tuple[int, int], np.ndarray] | None = None,
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
    precomputed_beliefs : dict or None
        Optional precomputed belief cache from ``precompute_beliefs()``.
        When provided, ``_compute_belief`` uses O(1) lookups instead of
        calling ``likelihood_model.score()``.

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
        seed=int(env_cfg.get("seed", 13)),
        wait_penalty=float(env_cfg.get("wait_penalty", 0.01)),
        early_buzz_penalty=float(env_cfg.get("early_buzz_penalty", 0.0)),
        buzz_correct=float(env_cfg.get("buzz_correct", 1.0)),
        buzz_incorrect=float(env_cfg.get("buzz_incorrect", -0.5)),
        belief_mode=str(env_cfg.get("belief_mode", "from_scratch")),
        beta=float(lik_cfg.get("beta", 5.0)),
        precomputed_beliefs=precomputed_beliefs,
    )
