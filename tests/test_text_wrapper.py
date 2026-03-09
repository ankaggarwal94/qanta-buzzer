"""Unit tests for TextObservationWrapper.

Tests verify that the wrapper correctly converts TossupMCEnv's numeric
belief observations into text-formatted strings for T5PolicyModel input.

Uses TF-IDF likelihood for fast test execution (<1 second total).
"""

from __future__ import annotations

import pytest

from qb_data.mc_builder import MCQuestion
from qb_env.text_wrapper import TextObservationWrapper
from qb_env.tossup_env import TossupMCEnv
from models.likelihoods import TfIdfLikelihood


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_mc_question() -> MCQuestion:
    """Return a minimal MCQuestion for wrapper testing."""
    tokens = [
        "Who", "was", "the", "first", "president",
        "of", "the", "United", "States", "?",
    ]
    run_indices = [0, 2, 4, 6, 8, 9]
    cumulative_prefixes = [
        "Who",
        "Who was the",
        "Who was the first president",
        "Who was the first president of the",
        "Who was the first president of the United States",
        "Who was the first president of the United States ?",
    ]
    return MCQuestion(
        qid="test_q1",
        question="Who was the first president of the United States?",
        tokens=tokens,
        answer_primary="George Washington",
        clean_answers=["George Washington", "Washington"],
        run_indices=run_indices,
        human_buzz_positions=[],
        category="History",
        cumulative_prefixes=cumulative_prefixes,
        options=[
            "George Washington",
            "Thomas Jefferson",
            "John Adams",
            "Benjamin Franklin",
        ],
        gold_index=0,
        option_profiles=[
            "George Washington first president commander revolutionary war",
            "Thomas Jefferson third president declaration independence",
            "John Adams second president Massachusetts diplomat",
            "Benjamin Franklin inventor diplomat Philadelphia printing",
        ],
        option_answer_primary=[
            "George Washington",
            "Thomas Jefferson",
            "John Adams",
            "Benjamin Franklin",
        ],
        distractor_strategy="test",
    )


@pytest.fixture
def wrapped_env(sample_mc_question: MCQuestion) -> TextObservationWrapper:
    """Return a TextObservationWrapper around a TossupMCEnv."""
    corpus = sample_mc_question.option_profiles[:]
    model = TfIdfLikelihood(corpus_texts=corpus)
    questions = [sample_mc_question] * 3
    env = TossupMCEnv(
        questions=questions,
        likelihood_model=model,
        K=4,
        reward_mode="simple",
        wait_penalty=0.0,
        buzz_correct=1.0,
        buzz_incorrect=-1.0,
        belief_mode="from_scratch",
        beta=5.0,
    )
    return TextObservationWrapper(env)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestTextObservationWrapper:
    """Tests for TextObservationWrapper class."""

    def test_wrapper_observation_format(self, wrapped_env: TextObservationWrapper):
        """Observation returns 'CLUES: ... | CHOICES: ...' format."""
        obs, info = wrapped_env.reset()

        assert isinstance(obs, str), f"Expected str, got {type(obs)}"
        assert "CLUES:" in obs, "Observation must contain 'CLUES:'"
        assert "CHOICES:" in obs, "Observation must contain 'CHOICES:'"
        assert "(1)" in obs, "Choices must be numbered starting at (1)"
        assert "(4)" in obs, "All 4 choices must be present"

    def test_wrapper_incremental_clues(self, wrapped_env: TextObservationWrapper):
        """Wrapper shows correct clues based on step_idx progression."""
        obs0, _ = wrapped_env.reset()

        # Initial: first token only
        clues_part = obs0.split(" | CHOICES:")[0].replace("CLUES: ", "")
        assert clues_part == "Who", f"Initial clues should be 'Who', got '{clues_part}'"

        # After first WAIT: cumulative_prefixes[0] = "Who"
        obs1, _, _, _, _ = wrapped_env.step(0)
        clues1 = obs1.split(" | CHOICES:")[0].replace("CLUES: ", "")
        assert clues1 == "Who", f"After 1st WAIT should be 'Who', got '{clues1}'"

        # After second WAIT: cumulative_prefixes[1] = "Who was the"
        obs2, _, _, _, _ = wrapped_env.step(0)
        clues2 = obs2.split(" | CHOICES:")[0].replace("CLUES: ", "")
        assert clues2 == "Who was the", f"After 2nd WAIT should be 'Who was the', got '{clues2}'"

    def test_wrapper_gymnasium_api(self, wrapped_env: TextObservationWrapper):
        """reset() and step() still work after wrapping."""
        # reset returns (obs, info) tuple
        result = wrapped_env.reset()
        assert isinstance(result, tuple)
        assert len(result) == 2
        obs, info = result
        assert isinstance(obs, str)
        assert isinstance(info, dict)
        assert "qid" in info

        # step returns (obs, reward, terminated, truncated, info)
        result = wrapped_env.step(0)  # WAIT
        assert isinstance(result, tuple)
        assert len(result) == 5
        obs, reward, terminated, truncated, info = result
        assert isinstance(obs, str)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_wrapper_preserves_reward(self, sample_mc_question: MCQuestion):
        """Reward from wrapped env matches underlying env behavior."""
        corpus = sample_mc_question.option_profiles[:]
        model = TfIdfLikelihood(corpus_texts=corpus)

        # Create unwrapped env
        env = TossupMCEnv(
            questions=[sample_mc_question] * 3,
            likelihood_model=model,
            K=4,
            reward_mode="simple",
            buzz_correct=1.0,
            buzz_incorrect=-1.0,
            seed=42,
        )

        # Create wrapped env with same seed
        env2 = TossupMCEnv(
            questions=[sample_mc_question] * 3,
            likelihood_model=model,
            K=4,
            reward_mode="simple",
            buzz_correct=1.0,
            buzz_incorrect=-1.0,
            seed=42,
        )
        wrapped = TextObservationWrapper(env2)

        # Reset both
        _, info1 = env.reset(seed=42)
        _, info2 = wrapped.reset(seed=42)

        # Take same actions
        _, r1, d1, t1, _ = env.step(0)
        _, r2, d2, t2, _ = wrapped.step(0)
        assert r1 == r2, f"Rewards differ: {r1} vs {r2}"
        assert d1 == d2, f"Terminated differs"
        assert t1 == t2, f"Truncated differs"

        # BUZZ with answer 1 (correct for gold_index=0)
        _, r1, d1, t1, _ = env.step(1)
        _, r2, d2, t2, _ = wrapped.step(1)
        assert r1 == r2, f"Buzz rewards differ: {r1} vs {r2}"
        assert d1 == d2

    def test_wrapper_multiple_steps(self, wrapped_env: TextObservationWrapper):
        """Multi-step episode produces increasing clue text."""
        obs, _ = wrapped_env.reset()
        prev_clues = obs.split(" | CHOICES:")[0]

        # Take multiple WAIT steps and verify clues grow
        grew_at_least_once = False
        for step in range(4):
            obs, _, terminated, truncated, _ = wrapped_env.step(0)
            if terminated or truncated:
                break
            current_clues = obs.split(" | CHOICES:")[0]
            if len(current_clues) > len(prev_clues):
                grew_at_least_once = True
            # Clues should never shrink
            assert len(current_clues) >= len(prev_clues), (
                f"Clues shrank at step {step}: '{prev_clues}' -> '{current_clues}'"
            )
            prev_clues = current_clues

        assert grew_at_least_once, "Clue text should grow with more WAITs"

    def test_wrapper_choices_include_all_options(
        self, wrapped_env: TextObservationWrapper
    ):
        """All 4 answer options appear in the choices section."""
        obs, _ = wrapped_env.reset()
        choices_part = obs.split("CHOICES: ")[1]

        assert "George Washington" in choices_part
        assert "Thomas Jefferson" in choices_part
        assert "John Adams" in choices_part
        assert "Benjamin Franklin" in choices_part

    def test_wrapper_buzz_ends_episode(self, wrapped_env: TextObservationWrapper):
        """Buzzing with an answer ends the episode."""
        wrapped_env.reset()
        _, _, terminated, truncated, info = wrapped_env.step(1)  # BUZZ answer 0
        assert terminated or truncated, "Episode should end after BUZZ"

    def test_wrapper_complete_episode(self, wrapped_env: TextObservationWrapper):
        """Full episode: WAIT until truncated or BUZZ."""
        wrapped_env.reset()

        for step in range(20):
            obs, reward, terminated, truncated, info = wrapped_env.step(0)
            if terminated or truncated:
                break
            assert isinstance(obs, str)

        # Episode must have ended (6 clue steps)
        assert terminated or truncated, "Episode should end within 20 steps"
