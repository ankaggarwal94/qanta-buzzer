"""Test suite for qb_env/tossup_env.py — TossupMCEnv Gymnasium environment.

Covers:
- ENV-01: Gymnasium interface compliance (reset, step, spaces)
- ENV-02: Action space Discrete(K+1) with WAIT and BUZZ actions
- ENV-04: Reward modes (time_penalty, simple, human_grounded)
- ENV-05: Likelihood model pluggability
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

from models.likelihoods import SBERTLikelihood, TfIdfLikelihood
from qb_data.mc_builder import MCQuestion
from qb_env.tossup_env import TossupMCEnv


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #


def _make_env(
    mc_question: MCQuestion,
    corpus: list[str] | None = None,
    reward_mode: str = "simple",
    wait_penalty: float = 0.0,
    buzz_correct: float = 1.0,
    buzz_incorrect: float = -1.0,
    belief_mode: str = "from_scratch",
    beta: float = 5.0,
    use_sbert: bool = False,
) -> TossupMCEnv:
    """Create a TossupMCEnv with TF-IDF or SBERT likelihood model.

    Helper for tests that need a configured environment without going
    through the factory function.
    """
    if use_sbert:
        model = SBERTLikelihood()
    else:
        if corpus is None:
            corpus = mc_question.option_profiles[:]
        model = TfIdfLikelihood(corpus_texts=corpus)
    return TossupMCEnv(
        questions=[mc_question],
        likelihood_model=model,
        K=4,
        reward_mode=reward_mode,
        wait_penalty=wait_penalty,
        buzz_correct=buzz_correct,
        buzz_incorrect=buzz_incorrect,
        belief_mode=belief_mode,
        beta=beta,
    )


# ------------------------------------------------------------------ #
# Tests: Gymnasium Interface (ENV-01)
# ------------------------------------------------------------------ #


class TestGymnasiumInterface:
    """Tests for Gymnasium API compliance."""

    def test_isinstance_gym_env(self, sample_mc_question: MCQuestion) -> None:
        """TossupMCEnv is a subclass of gym.Env."""
        env = _make_env(sample_mc_question)
        assert isinstance(env, gym.Env), "TossupMCEnv should be a gym.Env subclass"

    def test_has_reset_and_step(self, sample_mc_question: MCQuestion) -> None:
        """Environment has reset() and step() methods."""
        env = _make_env(sample_mc_question)
        assert hasattr(env, "reset"), "Missing reset() method"
        assert hasattr(env, "step"), "Missing step() method"
        assert callable(env.reset), "reset should be callable"
        assert callable(env.step), "step should be callable"

    def test_action_space_discrete(self, sample_mc_question: MCQuestion) -> None:
        """Action space is Discrete(K+1) = Discrete(5) for K=4."""
        env = _make_env(sample_mc_question)
        assert isinstance(env.action_space, gym.spaces.Discrete), (
            f"Expected Discrete, got {type(env.action_space)}"
        )
        assert env.action_space.n == 5, (
            f"Expected Discrete(5) for K=4, got Discrete({env.action_space.n})"
        )

    def test_observation_space_box(self, sample_mc_question: MCQuestion) -> None:
        """Observation space is Box(K+6,) = Box(10,) for K=4."""
        env = _make_env(sample_mc_question)
        assert isinstance(env.observation_space, gym.spaces.Box), (
            f"Expected Box, got {type(env.observation_space)}"
        )
        assert env.observation_space.shape == (10,), (
            f"Expected shape (10,), got {env.observation_space.shape}"
        )
        assert env.observation_space.dtype == np.float32, (
            f"Expected float32, got {env.observation_space.dtype}"
        )

    def test_action_space_contains_all_valid_actions(
        self, sample_mc_question: MCQuestion
    ) -> None:
        """All actions 0..K are valid in the action space."""
        env = _make_env(sample_mc_question)
        for action in range(5):
            assert env.action_space.contains(action), (
                f"Action {action} should be valid"
            )
        assert not env.action_space.contains(5), "Action 5 should be invalid for K=4"
        assert not env.action_space.contains(-1), "Action -1 should be invalid"


# ------------------------------------------------------------------ #
# Tests: Episode Flow
# ------------------------------------------------------------------ #


class TestEpisodeFlow:
    """Tests for reset/step/termination lifecycle."""

    def test_reset_returns_obs_and_info(self, sample_mc_question: MCQuestion) -> None:
        """reset() returns (observation, info) tuple."""
        env = _make_env(sample_mc_question)
        result = env.reset()
        assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
        assert len(result) == 2, f"Expected 2 elements, got {len(result)}"

    def test_reset_obs_shape_dtype(self, sample_mc_question: MCQuestion) -> None:
        """Observation from reset is (K+6,) float32."""
        env = _make_env(sample_mc_question)
        obs, info = env.reset()
        assert obs.shape == (10,), f"Expected (10,), got {obs.shape}"
        assert obs.dtype == np.float32, f"Expected float32, got {obs.dtype}"

    def test_reset_info_contains_qid(self, sample_mc_question: MCQuestion) -> None:
        """Info dict from reset contains qid."""
        env = _make_env(sample_mc_question)
        _obs, info = env.reset()
        assert "qid" in info, "Info should contain 'qid'"
        assert info["qid"] == "test_q1", f"Expected 'test_q1', got {info['qid']}"

    def test_reset_initializes_state(self, sample_mc_question: MCQuestion) -> None:
        """After reset, step_idx=0, not terminated, not truncated."""
        env = _make_env(sample_mc_question)
        env.reset()
        assert env.step_idx == 0, f"step_idx should be 0, got {env.step_idx}"
        assert env.terminated is False, "terminated should be False"
        assert env.truncated is False, "truncated should be False"

    def test_wait_action_advances_step(self, sample_mc_question: MCQuestion) -> None:
        """WAIT (action 0) increments step_idx and returns not terminated."""
        env = _make_env(sample_mc_question)
        env.reset()
        obs, reward, terminated, truncated, info = env.step(0)
        assert not terminated, "Should not terminate on WAIT"
        assert obs.shape == (10,), f"Expected (10,), got {obs.shape}"
        assert env.step_idx == 1, f"step_idx should be 1, got {env.step_idx}"

    def test_buzz_correct_terminates(self, sample_mc_question: MCQuestion) -> None:
        """Buzzing with correct answer (action 1 = option 0 = gold) terminates."""
        env = _make_env(sample_mc_question)
        env.reset()
        obs, reward, terminated, truncated, info = env.step(1)  # gold_index=0, action=1
        assert terminated is True, "Should terminate on buzz"
        assert truncated is False, "Should not be truncated"
        assert info["correct"] is True, "Buzzing with gold should be correct"
        assert info["chosen_idx"] == 0, f"chosen_idx should be 0, got {info['chosen_idx']}"

    def test_buzz_incorrect_terminates(self, sample_mc_question: MCQuestion) -> None:
        """Buzzing with incorrect answer terminates with correct=False."""
        env = _make_env(sample_mc_question)
        env.reset()
        obs, reward, terminated, truncated, info = env.step(2)  # option 1 = incorrect
        assert terminated is True, "Should terminate on buzz"
        assert info["correct"] is False, "Buzzing with wrong answer should be incorrect"

    def test_forced_termination(self, sample_mc_question: MCQuestion) -> None:
        """Exhausting all clues causes truncation with forced choice."""
        env = _make_env(sample_mc_question)
        env.reset()
        total = env.total_steps  # 6 steps for sample question

        # WAIT until all clues exhausted
        for i in range(total):
            obs, reward, terminated, truncated, info = env.step(0)
            if truncated:
                break

        assert truncated is True, "Should be truncated after exhausting clues"
        assert "forced_choice" in info, "Info should contain 'forced_choice'"
        assert "forced_correct" in info, "Info should contain 'forced_correct'"
        assert isinstance(info["forced_choice"], int), "forced_choice should be int"

    def test_step_before_reset_raises(self, sample_mc_question: MCQuestion) -> None:
        """Calling step() before reset() raises RuntimeError."""
        env = _make_env(sample_mc_question)
        with pytest.raises(RuntimeError, match="reset"):
            env.step(0)

    def test_step_after_terminated_raises(self, sample_mc_question: MCQuestion) -> None:
        """Calling step() after termination raises RuntimeError."""
        env = _make_env(sample_mc_question)
        env.reset()
        env.step(1)  # buzz to terminate
        with pytest.raises(RuntimeError, match="terminated"):
            env.step(0)

    def test_invalid_action_raises(self, sample_mc_question: MCQuestion) -> None:
        """Invalid action raises ValueError."""
        env = _make_env(sample_mc_question)
        env.reset()
        with pytest.raises(ValueError, match="Invalid action"):
            env.step(99)


# ------------------------------------------------------------------ #
# Tests: Reward Modes (ENV-04)
# ------------------------------------------------------------------ #


class TestRewardModes:
    """Tests for different reward computation modes."""

    def test_reward_simple_correct(self, sample_mc_question: MCQuestion) -> None:
        """Simple mode: correct buzz gives +1.0."""
        env = _make_env(sample_mc_question, reward_mode="simple")
        env.reset()
        _obs, reward, _term, _trunc, _info = env.step(1)  # correct buzz
        assert reward == 1.0, f"Simple correct reward should be 1.0, got {reward}"

    def test_reward_simple_incorrect(self, sample_mc_question: MCQuestion) -> None:
        """Simple mode: incorrect buzz gives -1.0."""
        env = _make_env(sample_mc_question, reward_mode="simple")
        env.reset()
        _obs, reward, _term, _trunc, _info = env.step(2)  # incorrect buzz
        assert reward == -1.0, f"Simple incorrect reward should be -1.0, got {reward}"

    def test_reward_simple_wait_no_penalty(self, sample_mc_question: MCQuestion) -> None:
        """Simple mode: WAIT has 0 reward regardless of wait_penalty setting."""
        env = _make_env(
            sample_mc_question, reward_mode="simple", wait_penalty=0.1
        )
        env.reset()
        _obs, reward, _term, _trunc, _info = env.step(0)
        assert reward == 0.0, f"Simple WAIT reward should be 0.0, got {reward}"

    def test_reward_time_penalty_wait(self, sample_mc_question: MCQuestion) -> None:
        """Time penalty mode: WAIT incurs -wait_penalty."""
        env = _make_env(
            sample_mc_question, reward_mode="time_penalty", wait_penalty=0.1
        )
        env.reset()
        _obs, reward, _term, _trunc, _info = env.step(0)
        assert abs(reward - (-0.1)) < 1e-6, (
            f"Time penalty WAIT reward should be -0.1, got {reward}"
        )

    def test_reward_time_penalty_buzz_correct(
        self, sample_mc_question: MCQuestion
    ) -> None:
        """Time penalty mode: correct buzz gives buzz_correct."""
        env = _make_env(
            sample_mc_question,
            reward_mode="time_penalty",
            buzz_correct=1.0,
            wait_penalty=0.1,
        )
        env.reset()
        _obs, reward, _term, _trunc, _info = env.step(1)
        assert reward == 1.0, f"Time penalty correct buzz should be 1.0, got {reward}"

    def test_reward_time_penalty_cumulative(
        self, sample_mc_question: MCQuestion
    ) -> None:
        """Time penalty mode: waiting then buzzing accumulates penalties."""
        env = _make_env(
            sample_mc_question,
            reward_mode="time_penalty",
            wait_penalty=0.1,
            buzz_correct=1.0,
        )
        env.reset()
        # Wait 2 steps (-0.2 cumulative), then buzz correct (+1.0)
        total_reward = 0.0
        _obs, r1, _t, _tr, _info = env.step(0)
        total_reward += r1
        _obs, r2, _t, _tr, _info = env.step(0)
        total_reward += r2
        _obs, r3, _t, _tr, _info = env.step(1)  # buzz correct
        total_reward += r3
        assert abs(total_reward - 0.8) < 1e-6, (
            f"Cumulative reward should be ~0.8, got {total_reward}"
        )

    def test_reward_human_grounded(self, sample_mc_question: MCQuestion) -> None:
        """Human grounded mode works without human buzz data (returns normal reward)."""
        env = _make_env(
            sample_mc_question,
            reward_mode="human_grounded",
            buzz_correct=1.0,
            buzz_incorrect=-0.5,
        )
        env.reset()
        # With no human buzz positions, reward should be buzz_correct/incorrect
        _obs, reward, _term, _trunc, _info = env.step(1)
        assert reward == 1.0, f"Human grounded correct buzz should be 1.0, got {reward}"

    def test_reward_human_grounded_with_positions(self) -> None:
        """Human grounded mode: buzzing after human position gives 0.0."""
        # Create question with human buzz at position 0 (very early)
        mc_q = MCQuestion(
            qid="hg_test",
            question="Who was the first president?",
            tokens=["Who", "was", "the", "first", "president", "?"],
            answer_primary="George Washington",
            clean_answers=["George Washington"],
            run_indices=[0, 2, 4, 5],
            human_buzz_positions=[(0, 10)],  # Most humans buzz at position 0
            category="History",
            cumulative_prefixes=[
                "Who",
                "Who was the",
                "Who was the first president",
                "Who was the first president ?",
            ],
            options=["George Washington", "Jefferson", "Adams", "Franklin"],
            gold_index=0,
            option_profiles=["Washington", "Jefferson", "Adams", "Franklin"],
            option_answer_primary=["George Washington", "Jefferson", "Adams", "Franklin"],
            distractor_strategy="test",
        )
        corpus = mc_q.option_profiles[:]
        model = TfIdfLikelihood(corpus_texts=corpus)
        env = TossupMCEnv(
            questions=[mc_q],
            likelihood_model=model,
            K=4,
            reward_mode="human_grounded",
            buzz_correct=1.0,
            buzz_incorrect=-0.5,
        )
        env.reset()
        # Wait a few steps so agent buzzes after human position (0)
        env.step(0)  # step 0 -> reveal clue at position 0
        env.step(0)  # step 1 -> reveal clue at position 2
        _obs, reward, _term, _trunc, _info = env.step(1)  # buzz at step 2
        # Agent buzzes at token pos > 0 (human), so reward should be 0.0
        assert reward == 0.0, f"Should get 0.0 for buzzing after human, got {reward}"


# ------------------------------------------------------------------ #
# Tests: Likelihood Model Pluggability (ENV-05)
# ------------------------------------------------------------------ #


class TestLikelihoodPluggability:
    """Tests for interchangeable likelihood models."""

    def test_tfidf_model_produces_valid_obs(
        self, sample_mc_question: MCQuestion
    ) -> None:
        """TF-IDF likelihood model produces valid observations."""
        env = _make_env(sample_mc_question, use_sbert=False)
        obs, info = env.reset()
        assert obs.shape == (10,), f"Expected (10,), got {obs.shape}"
        assert np.all(np.isfinite(obs)), "All observations should be finite"
        # Take a step
        obs2, _r, _t, _tr, _info = env.step(0)
        assert obs2.shape == (10,), f"Expected (10,), got {obs2.shape}"
        assert np.all(np.isfinite(obs2)), "Step observations should be finite"

    def test_sbert_model_produces_valid_obs(
        self, sample_mc_question: MCQuestion
    ) -> None:
        """SBERT likelihood model produces valid observations."""
        env = _make_env(sample_mc_question, use_sbert=True)
        obs, info = env.reset()
        assert obs.shape == (10,), f"Expected (10,), got {obs.shape}"
        assert np.all(np.isfinite(obs)), "All observations should be finite"
        # Take a step
        obs2, _r, _t, _tr, _info = env.step(0)
        assert obs2.shape == (10,), f"Expected (10,), got {obs2.shape}"
        assert np.all(np.isfinite(obs2)), "Step observations should be finite"

    def test_both_models_same_obs_shape(
        self, sample_mc_question: MCQuestion
    ) -> None:
        """Both TF-IDF and SBERT produce same observation shape."""
        env_tfidf = _make_env(sample_mc_question, use_sbert=False)
        env_sbert = _make_env(sample_mc_question, use_sbert=True)

        obs_tfidf, _ = env_tfidf.reset(seed=42)
        obs_sbert, _ = env_sbert.reset(seed=42)

        assert obs_tfidf.shape == obs_sbert.shape, (
            f"TF-IDF obs {obs_tfidf.shape} != SBERT obs {obs_sbert.shape}"
        )
        assert obs_tfidf.dtype == obs_sbert.dtype, (
            f"TF-IDF dtype {obs_tfidf.dtype} != SBERT dtype {obs_sbert.dtype}"
        )


# ------------------------------------------------------------------ #
# Tests: Belief Modes
# ------------------------------------------------------------------ #


class TestBeliefModes:
    """Tests for different belief computation modes."""

    def test_from_scratch_belief(self, sample_mc_question: MCQuestion) -> None:
        """from_scratch mode recomputes belief from cumulative prefix."""
        env = _make_env(sample_mc_question, belief_mode="from_scratch")
        env.reset()
        # Wait several steps to get a more discriminative clue prefix
        for _ in range(3):
            env.step(0)
        # After multiple steps with more context, belief should be valid
        # and at least one option should have higher probability
        assert abs(env.belief.sum() - 1.0) < 1e-5, (
            f"Belief should sum to 1.0, got {env.belief.sum()}"
        )
        assert all(env.belief >= 0), "All beliefs should be non-negative"
        assert env.belief.dtype == np.float32, "Belief should be float32"

    def test_sequential_bayes_belief(self, sample_mc_question: MCQuestion) -> None:
        """sequential_bayes mode updates belief incrementally."""
        env = _make_env(sample_mc_question, belief_mode="sequential_bayes")
        env.reset()
        env.step(0)  # first WAIT
        # Belief should sum to ~1.0
        assert abs(env.belief.sum() - 1.0) < 1e-5, (
            f"Belief should sum to 1.0, got {env.belief.sum()}"
        )

    def test_invalid_belief_mode_raises(self, sample_mc_question: MCQuestion) -> None:
        """Unknown belief mode raises ValueError on step."""
        env = _make_env(sample_mc_question, belief_mode="unknown_mode")
        env.reset()
        with pytest.raises(ValueError, match="Unknown belief_mode"):
            env.step(0)


# ------------------------------------------------------------------ #
# Tests: Constructor Validation
# ------------------------------------------------------------------ #


class TestConstructorValidation:
    """Tests for constructor input validation."""

    def test_empty_questions_raises(self) -> None:
        """Empty question list raises ValueError."""
        model = TfIdfLikelihood(corpus_texts=["test"])
        with pytest.raises(ValueError, match="cannot be empty"):
            TossupMCEnv(questions=[], likelihood_model=model)

    def test_k_less_than_2_raises(self, sample_mc_question: MCQuestion) -> None:
        """K < 2 raises ValueError."""
        model = TfIdfLikelihood(corpus_texts=["test"])
        with pytest.raises(ValueError, match="K must be >= 2"):
            TossupMCEnv(
                questions=[sample_mc_question], likelihood_model=model, K=1
            )
