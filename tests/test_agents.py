"""Test suite for agents/ -- baseline agent execution and episode result schemas.

Covers:
- AGT-02: ThresholdBuzzer execution and buzzing logic
- AGT-03: AlwaysBuzzFinalBuzzer wait-then-buzz behavior
- AGT-04: SoftmaxProfileBuzzer from-scratch belief recomputation
- AGT-05: SequentialBayesBuzzer incremental Bayesian updates
- AGT-06: EpisodeResult and SoftmaxEpisodeResult schema validation
- Threshold sweep utility tests
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from agents import (
    AlwaysBuzzFinalBuzzer,
    EpisodeResult,
    SequentialBayesBuzzer,
    SoftmaxEpisodeResult,
    SoftmaxProfileBuzzer,
    ThresholdBuzzer,
    result_to_dict,
    sweep_thresholds,
)
from agents._math import sigmoid
from models.likelihoods import TfIdfLikelihood
from qb_data.mc_builder import MCQuestion


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #


def _make_likelihood(corpus: list[str]) -> TfIdfLikelihood:
    """Create a fitted TF-IDF likelihood model from a corpus.

    Uses TF-IDF (fast) for agent logic tests so tests run quickly.
    """
    return TfIdfLikelihood(corpus_texts=corpus)


class TestSigmoidMath:
    """Tests for stable scalar sigmoid helper."""

    def test_sigmoid_handles_extreme_inputs_without_warning(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            assert sigmoid(1000.0) == pytest.approx(1.0)
            assert sigmoid(-1000.0) == pytest.approx(0.0)


# ------------------------------------------------------------------ #
# ThresholdBuzzer tests (AGT-02)
# ------------------------------------------------------------------ #


class TestThresholdBuzzer:
    """Tests for ThresholdBuzzer execution and buzzing logic."""

    def test_threshold_buzzer_executes(
        self, sample_mc_question: MCQuestion, sample_corpus: list[str]
    ) -> None:
        """ThresholdBuzzer runs an episode without error and returns EpisodeResult."""
        likelihood = _make_likelihood(sample_corpus)
        agent = ThresholdBuzzer(likelihood_model=likelihood, threshold=0.7)
        result = agent.run_episode(sample_mc_question)

        assert isinstance(result, EpisodeResult)
        assert result.qid == sample_mc_question.qid
        assert len(result.c_trace) > 0

    def test_threshold_buzzer_buzzes_on_threshold(
        self, sample_mc_question: MCQuestion, sample_corpus: list[str]
    ) -> None:
        """ThresholdBuzzer buzzes when top_p >= threshold.

        With threshold=0.0, the agent should buzz immediately at step 0
        because any non-negative top_p will meet the threshold.
        """
        likelihood = _make_likelihood(sample_corpus)
        agent = ThresholdBuzzer(likelihood_model=likelihood, threshold=0.0)
        result = agent.run_episode(sample_mc_question)

        # With threshold 0.0, should buzz at step 0
        assert result.buzz_step == 0, (
            f"Expected buzz at step 0 with threshold=0.0, got step {result.buzz_step}"
        )

    def test_threshold_buzzer_waits_on_low_confidence(
        self, sample_mc_question: MCQuestion, sample_corpus: list[str]
    ) -> None:
        """ThresholdBuzzer waits when top_p < threshold.

        With threshold=1.0 (impossible for softmax to reach exactly 1.0 in
        practice), the agent should wait until the final step.
        """
        likelihood = _make_likelihood(sample_corpus)
        agent = ThresholdBuzzer(likelihood_model=likelihood, threshold=1.0)
        result = agent.run_episode(sample_mc_question)

        # With threshold 1.0, should wait until the last step
        expected_final = len(sample_mc_question.cumulative_prefixes) - 1
        assert result.buzz_step == expected_final, (
            f"Expected buzz at final step {expected_final} with threshold=1.0, "
            f"got step {result.buzz_step}"
        )

    def test_threshold_buzzer_buzzes_at_final(
        self, sample_mc_question: MCQuestion, sample_corpus: list[str]
    ) -> None:
        """ThresholdBuzzer always buzzes on final step regardless of threshold.

        Even with threshold=1.0 (unreachable), the agent must buzz at the
        final step as a forced fallback.
        """
        likelihood = _make_likelihood(sample_corpus)
        agent = ThresholdBuzzer(likelihood_model=likelihood, threshold=1.0)
        result = agent.run_episode(sample_mc_question)

        final_step = len(sample_mc_question.cumulative_prefixes) - 1
        assert result.buzz_step == final_step
        assert result.buzz_index in range(len(sample_mc_question.options))

    def test_threshold_buzzer_traces_valid(
        self, sample_mc_question: MCQuestion, sample_corpus: list[str]
    ) -> None:
        """c_trace and g_trace have correct and matching lengths.

        Traces should have length equal to buzz_step + 1 (one entry per
        step from 0 to buzz_step inclusive).
        """
        likelihood = _make_likelihood(sample_corpus)
        agent = ThresholdBuzzer(likelihood_model=likelihood, threshold=0.7)
        result = agent.run_episode(sample_mc_question)

        trace_len = result.buzz_step + 1
        assert len(result.c_trace) == trace_len, (
            f"c_trace length {len(result.c_trace)} != expected {trace_len}"
        )
        assert len(result.g_trace) == trace_len, (
            f"g_trace length {len(result.g_trace)} != expected {trace_len}"
        )
        assert len(result.top_p_trace) == trace_len
        assert len(result.entropy_trace) == trace_len

    def test_threshold_buzzer_confidence_proxy(
        self, sample_mc_question: MCQuestion, sample_corpus: list[str]
    ) -> None:
        """c_t values in [0, 1] via sigmoid transformation."""
        likelihood = _make_likelihood(sample_corpus)
        agent = ThresholdBuzzer(likelihood_model=likelihood, threshold=0.7)
        result = agent.run_episode(sample_mc_question)

        for c_t in result.c_trace:
            assert 0.0 <= c_t <= 1.0, (
                f"Confidence proxy {c_t} outside [0, 1]"
            )

    def test_threshold_buzzer_custom_params(
        self, sample_mc_question: MCQuestion, sample_corpus: list[str]
    ) -> None:
        """ThresholdBuzzer accepts custom beta and alpha parameters."""
        likelihood = _make_likelihood(sample_corpus)
        agent = ThresholdBuzzer(
            likelihood_model=likelihood,
            threshold=0.5,
            beta=10.0,
            alpha=20.0,
        )
        assert agent.beta == 10.0
        assert agent.alpha == 20.0

        result = agent.run_episode(sample_mc_question)
        assert isinstance(result, EpisodeResult)

    def test_threshold_buzzer_confidence_proxy_stable_extremes(
        self, sample_corpus: list[str]
    ) -> None:
        likelihood = _make_likelihood(sample_corpus)
        agent = ThresholdBuzzer(
            likelihood_model=likelihood,
            threshold=-100.0,
            alpha=100.0,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            assert agent._confidence_proxy(1.0) == pytest.approx(1.0)

        agent = ThresholdBuzzer(
            likelihood_model=likelihood,
            threshold=100.0,
            alpha=100.0,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            assert agent._confidence_proxy(0.0) == pytest.approx(0.0)

    def test_threshold_buzzer_top_p_in_range(
        self, sample_mc_question: MCQuestion, sample_corpus: list[str]
    ) -> None:
        """top_p_trace values are valid probabilities in [0, 1]."""
        likelihood = _make_likelihood(sample_corpus)
        agent = ThresholdBuzzer(likelihood_model=likelihood, threshold=0.7)
        result = agent.run_episode(sample_mc_question)

        for p in result.top_p_trace:
            assert 0.0 <= p <= 1.0, f"top_p {p} outside [0, 1]"

    def test_threshold_buzzer_entropy_nonnegative(
        self, sample_mc_question: MCQuestion, sample_corpus: list[str]
    ) -> None:
        """Entropy values are non-negative (Shannon entropy >= 0)."""
        likelihood = _make_likelihood(sample_corpus)
        agent = ThresholdBuzzer(likelihood_model=likelihood, threshold=0.7)
        result = agent.run_episode(sample_mc_question)

        for h in result.entropy_trace:
            assert h >= 0.0, f"Entropy {h} is negative"


# ------------------------------------------------------------------ #
# AlwaysBuzzFinalBuzzer tests (AGT-03)
# ------------------------------------------------------------------ #


class TestAlwaysBuzzFinalBuzzer:
    """Tests for AlwaysBuzzFinalBuzzer wait-then-buzz behavior."""

    def test_always_buzz_final_waits(
        self, sample_mc_question: MCQuestion, sample_corpus: list[str]
    ) -> None:
        """All c_trace entries except the last are 0.0 (agent waits)."""
        likelihood = _make_likelihood(sample_corpus)
        agent = AlwaysBuzzFinalBuzzer(likelihood_model=likelihood)
        result = agent.run_episode(sample_mc_question)

        # All entries except last should be 0.0
        for c_t in result.c_trace[:-1]:
            assert c_t == 0.0, f"Expected c_t=0.0 for waiting, got {c_t}"

    def test_always_buzz_final_buzzes_last(
        self, sample_mc_question: MCQuestion, sample_corpus: list[str]
    ) -> None:
        """The last c_trace entry is 1.0 (agent buzzes at final step)."""
        likelihood = _make_likelihood(sample_corpus)
        agent = AlwaysBuzzFinalBuzzer(likelihood_model=likelihood)
        result = agent.run_episode(sample_mc_question)

        assert result.c_trace[-1] == 1.0, (
            f"Expected c_trace[-1]=1.0, got {result.c_trace[-1]}"
        )

    def test_always_buzz_final_computes_beliefs(
        self, sample_mc_question: MCQuestion, sample_corpus: list[str]
    ) -> None:
        """Beliefs are computed at each step (not skipped).

        All top_p_trace entries should have valid probability values,
        demonstrating the model computed beliefs at every step.
        """
        likelihood = _make_likelihood(sample_corpus)
        agent = AlwaysBuzzFinalBuzzer(likelihood_model=likelihood)
        result = agent.run_episode(sample_mc_question)

        n_steps = len(sample_mc_question.cumulative_prefixes)
        assert len(result.top_p_trace) == n_steps, (
            f"Expected {n_steps} top_p entries, got {len(result.top_p_trace)}"
        )
        for p in result.top_p_trace:
            assert 0.0 <= p <= 1.0, f"top_p {p} outside [0, 1]"

    def test_always_buzz_final_buzz_step(
        self, sample_mc_question: MCQuestion, sample_corpus: list[str]
    ) -> None:
        """buzz_step equals len(cumulative_prefixes) - 1 (last step)."""
        likelihood = _make_likelihood(sample_corpus)
        agent = AlwaysBuzzFinalBuzzer(likelihood_model=likelihood)
        result = agent.run_episode(sample_mc_question)

        expected = len(sample_mc_question.cumulative_prefixes) - 1
        assert result.buzz_step == expected, (
            f"Expected buzz_step={expected}, got {result.buzz_step}"
        )

    def test_always_buzz_final_full_trace(
        self, sample_mc_question: MCQuestion, sample_corpus: list[str]
    ) -> None:
        """All traces have length equal to number of cumulative prefixes."""
        likelihood = _make_likelihood(sample_corpus)
        agent = AlwaysBuzzFinalBuzzer(likelihood_model=likelihood)
        result = agent.run_episode(sample_mc_question)

        n = len(sample_mc_question.cumulative_prefixes)
        assert len(result.c_trace) == n
        assert len(result.g_trace) == n
        assert len(result.top_p_trace) == n
        assert len(result.entropy_trace) == n


# ------------------------------------------------------------------ #
# SoftmaxProfileBuzzer tests (AGT-04)
# ------------------------------------------------------------------ #


class TestSoftmaxProfileBuzzer:
    """Tests for SoftmaxProfileBuzzer from-scratch belief computation."""

    def test_softmax_profile_executes(
        self, sample_mc_question: MCQuestion, sample_corpus: list[str]
    ) -> None:
        """SoftmaxProfileBuzzer runs an episode without error."""
        likelihood = _make_likelihood(sample_corpus)
        agent = SoftmaxProfileBuzzer(likelihood_model=likelihood, threshold=0.7)
        result = agent.run_episode(sample_mc_question)

        assert isinstance(result, SoftmaxEpisodeResult)
        assert result.qid == sample_mc_question.qid

    def test_softmax_profile_recomputes_belief(
        self, sample_mc_question: MCQuestion, sample_corpus: list[str]
    ) -> None:
        """SoftmaxProfileBuzzer calls _belief_from_scratch each step.

        Verifies the method exists and the agent stores beliefs, confirming
        from-scratch recomputation (not incremental Bayesian updates).
        """
        likelihood = _make_likelihood(sample_corpus)
        agent = SoftmaxProfileBuzzer(likelihood_model=likelihood, threshold=0.7)

        # Verify the from-scratch method exists
        assert hasattr(agent, "_belief_from_scratch")

        result = agent.run_episode(sample_mc_question)

        # After episode, agent should have a stored belief
        assert agent.belief is not None
        assert isinstance(agent.belief, np.ndarray)
        assert agent.belief.shape == (len(sample_mc_question.options),)

    def test_softmax_profile_result_schema(
        self, sample_mc_question: MCQuestion, sample_corpus: list[str]
    ) -> None:
        """SoftmaxProfileBuzzer returns SoftmaxEpisodeResult, not EpisodeResult."""
        likelihood = _make_likelihood(sample_corpus)
        agent = SoftmaxProfileBuzzer(likelihood_model=likelihood, threshold=0.7)
        result = agent.run_episode(sample_mc_question)

        assert isinstance(result, SoftmaxEpisodeResult)
        # SoftmaxEpisodeResult should NOT be an EpisodeResult (different dataclass)
        assert not isinstance(result, EpisodeResult)

    def test_softmax_profile_confidence_proxy(
        self, sample_mc_question: MCQuestion, sample_corpus: list[str]
    ) -> None:
        """SoftmaxProfileBuzzer c_t values in [0, 1] via sigmoid."""
        likelihood = _make_likelihood(sample_corpus)
        agent = SoftmaxProfileBuzzer(likelihood_model=likelihood, threshold=0.7)
        result = agent.run_episode(sample_mc_question)

        for c_t in result.c_trace:
            assert 0.0 <= c_t <= 1.0, f"c_t {c_t} outside [0, 1]"

    def test_softmax_profile_threshold_behavior(
        self, sample_mc_question: MCQuestion, sample_corpus: list[str]
    ) -> None:
        """SoftmaxProfileBuzzer respects threshold for buzzing."""
        likelihood = _make_likelihood(sample_corpus)

        # With threshold 0.0, should buzz immediately
        agent_low = SoftmaxProfileBuzzer(likelihood_model=likelihood, threshold=0.0)
        result_low = agent_low.run_episode(sample_mc_question)
        assert result_low.buzz_step == 0

        # With threshold 1.0, should wait until the end
        agent_high = SoftmaxProfileBuzzer(likelihood_model=likelihood, threshold=1.0)
        result_high = agent_high.run_episode(sample_mc_question)
        assert result_high.buzz_step == len(sample_mc_question.cumulative_prefixes) - 1

    def test_softmax_profile_confidence_proxy_stable_extremes(
        self, sample_corpus: list[str]
    ) -> None:
        likelihood = _make_likelihood(sample_corpus)
        agent = SoftmaxProfileBuzzer(
            likelihood_model=likelihood,
            threshold=-100.0,
            alpha=100.0,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            assert agent.confidence_proxy(1.0) == pytest.approx(1.0)

        agent = SoftmaxProfileBuzzer(
            likelihood_model=likelihood,
            threshold=100.0,
            alpha=100.0,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            assert agent.confidence_proxy(0.0) == pytest.approx(0.0)


# ------------------------------------------------------------------ #
# SequentialBayesBuzzer tests (AGT-05)
# ------------------------------------------------------------------ #


class TestSequentialBayesBuzzer:
    """Tests for SequentialBayesBuzzer incremental Bayesian update."""

    def test_sequential_bayes_executes(
        self, sample_mc_question: MCQuestion, sample_corpus: list[str]
    ) -> None:
        """SequentialBayesBuzzer runs an episode without error."""
        likelihood = _make_likelihood(sample_corpus)
        agent = SequentialBayesBuzzer(likelihood_model=likelihood, threshold=0.7)
        result = agent.run_episode(sample_mc_question)

        assert isinstance(result, SoftmaxEpisodeResult)
        assert result.qid == sample_mc_question.qid

    def test_sequential_bayes_uses_run_indices(
        self, sample_mc_question: MCQuestion, sample_corpus: list[str]
    ) -> None:
        """SequentialBayesBuzzer requires question.run_indices field.

        The agent iterates over run_indices to extract token fragments,
        not over cumulative_prefixes. The number of trace entries should
        match the number of run_indices steps processed.
        """
        likelihood = _make_likelihood(sample_corpus)
        agent = SequentialBayesBuzzer(likelihood_model=likelihood, threshold=0.7)
        result = agent.run_episode(sample_mc_question)

        # Trace length should be <= len(run_indices)
        assert len(result.c_trace) <= len(sample_mc_question.run_indices), (
            f"Trace length {len(result.c_trace)} > run_indices length "
            f"{len(sample_mc_question.run_indices)}"
        )

    def test_sequential_bayes_bayesian_update(
        self, sample_mc_question: MCQuestion, sample_corpus: list[str]
    ) -> None:
        """Belief is posterior proportional to prior * likelihood.

        Verify the _step_update method produces valid posterior:
        all entries >= 0 and sum to 1.
        """
        likelihood = _make_likelihood(sample_corpus)
        agent = SequentialBayesBuzzer(likelihood_model=likelihood, threshold=0.7)

        K = len(sample_mc_question.options)
        prior = np.ones(K, dtype=np.float32) / K
        fragment = "first president"
        profiles = sample_mc_question.option_profiles

        posterior = agent._step_update(prior, fragment, profiles)

        assert posterior.shape == (K,), f"Expected shape ({K},), got {posterior.shape}"
        assert all(posterior >= 0), "Posterior has negative entries"
        np.testing.assert_almost_equal(
            posterior.sum(), 1.0, decimal=5,
            err_msg="Posterior should sum to 1.0",
        )

    def test_sequential_bayes_result_schema(
        self, sample_mc_question: MCQuestion, sample_corpus: list[str]
    ) -> None:
        """SequentialBayesBuzzer returns SoftmaxEpisodeResult."""
        likelihood = _make_likelihood(sample_corpus)
        agent = SequentialBayesBuzzer(likelihood_model=likelihood, threshold=0.7)
        result = agent.run_episode(sample_mc_question)

        assert isinstance(result, SoftmaxEpisodeResult)
        assert not isinstance(result, EpisodeResult)

    def test_sequential_bayes_fragments(
        self, sample_mc_question: MCQuestion, sample_corpus: list[str]
    ) -> None:
        """SequentialBayesBuzzer processes token fragments, not full prefixes.

        With threshold 1.0 (never buzzes early), all run_indices should be
        processed, producing traces of length len(run_indices).
        """
        likelihood = _make_likelihood(sample_corpus)
        agent = SequentialBayesBuzzer(likelihood_model=likelihood, threshold=1.0)
        result = agent.run_episode(sample_mc_question)

        n_steps = len(sample_mc_question.run_indices)
        assert len(result.c_trace) == n_steps, (
            f"Expected {n_steps} trace entries, got {len(result.c_trace)}"
        )


# ------------------------------------------------------------------ #
# Episode result schema tests (AGT-06)
# ------------------------------------------------------------------ #


class TestEpisodeResultSchema:
    """Tests for EpisodeResult and SoftmaxEpisodeResult dataclass schemas."""

    def test_episode_result_fields(self) -> None:
        """EpisodeResult has all required fields."""
        result = EpisodeResult(
            qid="test_q",
            buzz_step=3,
            buzz_index=1,
            gold_index=0,
            correct=False,
            reward_like=-0.5,
            c_trace=[0.1, 0.2, 0.3, 0.4],
            g_trace=[0.0, 0.0, 0.0, 1.0],
            top_p_trace=[0.3, 0.4, 0.5, 0.6],
            entropy_trace=[1.4, 1.2, 1.0, 0.8],
        )
        assert result.qid == "test_q"
        assert result.buzz_step == 3
        assert result.buzz_index == 1
        assert result.gold_index == 0
        assert result.correct is False
        assert result.reward_like == -0.5

    def test_softmax_episode_result_fields(self) -> None:
        """SoftmaxEpisodeResult has all required fields."""
        result = SoftmaxEpisodeResult(
            qid="test_q",
            buzz_step=2,
            buzz_index=0,
            gold_index=0,
            correct=True,
            c_trace=[0.1, 0.5, 0.9],
            g_trace=[1.0, 1.0, 1.0],
            top_p_trace=[0.4, 0.6, 0.9],
            entropy_trace=[1.2, 0.8, 0.3],
        )
        assert result.qid == "test_q"
        assert result.buzz_step == 2
        assert result.correct is True

    def test_traces_same_length(
        self, sample_mc_question: MCQuestion, sample_corpus: list[str]
    ) -> None:
        """len(c_trace) == len(g_trace) for all agents."""
        likelihood = _make_likelihood(sample_corpus)

        agents = [
            ThresholdBuzzer(likelihood_model=likelihood, threshold=0.7),
            AlwaysBuzzFinalBuzzer(likelihood_model=likelihood),
            SoftmaxProfileBuzzer(likelihood_model=likelihood, threshold=0.7),
            SequentialBayesBuzzer(likelihood_model=likelihood, threshold=0.7),
        ]

        for agent in agents:
            result = agent.run_episode(sample_mc_question)
            agent_name = type(agent).__name__
            assert len(result.c_trace) == len(result.g_trace), (
                f"{agent_name}: c_trace ({len(result.c_trace)}) != "
                f"g_trace ({len(result.g_trace)})"
            )
            assert len(result.c_trace) == len(result.top_p_trace), (
                f"{agent_name}: c_trace ({len(result.c_trace)}) != "
                f"top_p_trace ({len(result.top_p_trace)})"
            )
            assert len(result.c_trace) == len(result.entropy_trace), (
                f"{agent_name}: c_trace ({len(result.c_trace)}) != "
                f"entropy_trace ({len(result.entropy_trace)})"
            )

    def test_g_trace_binary(
        self, sample_mc_question: MCQuestion, sample_corpus: list[str]
    ) -> None:
        """g_trace values are 0.0 or 1.0 (correctness is binary)."""
        likelihood = _make_likelihood(sample_corpus)

        agents = [
            ThresholdBuzzer(likelihood_model=likelihood, threshold=0.7),
            AlwaysBuzzFinalBuzzer(likelihood_model=likelihood),
            SoftmaxProfileBuzzer(likelihood_model=likelihood, threshold=0.7),
            SequentialBayesBuzzer(likelihood_model=likelihood, threshold=0.7),
        ]

        for agent in agents:
            result = agent.run_episode(sample_mc_question)
            agent_name = type(agent).__name__
            for g_t in result.g_trace:
                assert g_t in (0.0, 1.0), (
                    f"{agent_name}: g_t={g_t} not in {{0.0, 1.0}}"
                )

    def test_buzz_index_valid(
        self, sample_mc_question: MCQuestion, sample_corpus: list[str]
    ) -> None:
        """buzz_index in range(K) where K = len(options)."""
        likelihood = _make_likelihood(sample_corpus)
        K = len(sample_mc_question.options)

        agents = [
            ThresholdBuzzer(likelihood_model=likelihood, threshold=0.7),
            AlwaysBuzzFinalBuzzer(likelihood_model=likelihood),
            SoftmaxProfileBuzzer(likelihood_model=likelihood, threshold=0.7),
            SequentialBayesBuzzer(likelihood_model=likelihood, threshold=0.7),
        ]

        for agent in agents:
            result = agent.run_episode(sample_mc_question)
            agent_name = type(agent).__name__
            assert 0 <= result.buzz_index < K, (
                f"{agent_name}: buzz_index={result.buzz_index} not in [0, {K})"
            )

    def test_result_to_dict(
        self, sample_mc_question: MCQuestion, sample_corpus: list[str]
    ) -> None:
        """result_to_dict() converts EpisodeResult to dict."""
        likelihood = _make_likelihood(sample_corpus)
        agent = ThresholdBuzzer(likelihood_model=likelihood, threshold=0.7)
        result = agent.run_episode(sample_mc_question)

        d = result_to_dict(result)
        assert isinstance(d, dict)
        assert d["qid"] == sample_mc_question.qid
        assert "buzz_step" in d
        assert "buzz_index" in d
        assert "gold_index" in d
        assert "correct" in d
        assert "reward_like" in d
        assert "c_trace" in d
        assert "g_trace" in d
        assert isinstance(d["c_trace"], list)


# ------------------------------------------------------------------ #
# Threshold sweep utility tests
# ------------------------------------------------------------------ #


class TestSweepThresholds:
    """Tests for sweep_thresholds utility function."""

    def test_sweep_thresholds_runs(
        self, sample_mc_question: MCQuestion, sample_corpus: list[str]
    ) -> None:
        """sweep_thresholds() returns dict[float, list[EpisodeResult]]."""
        likelihood = _make_likelihood(sample_corpus)
        results = sweep_thresholds(
            questions=[sample_mc_question],
            likelihood_model=likelihood,
            thresholds=[0.7],
        )

        assert isinstance(results, dict)
        assert 0.7 in results
        assert len(results[0.7]) == 1
        assert isinstance(results[0.7][0], EpisodeResult)

    def test_sweep_thresholds_multiple_values(
        self, sample_mc_question: MCQuestion, sample_corpus: list[str]
    ) -> None:
        """Sweeps over [0.6, 0.7, 0.8, 0.9] and returns results for each."""
        likelihood = _make_likelihood(sample_corpus)
        thresholds = [0.6, 0.7, 0.8, 0.9]
        results = sweep_thresholds(
            questions=[sample_mc_question],
            likelihood_model=likelihood,
            thresholds=thresholds,
        )

        assert len(results) == len(thresholds)
        for thresh in thresholds:
            assert thresh in results, f"Missing results for threshold {thresh}"
            assert len(results[thresh]) == 1

    def test_sweep_thresholds_monotonic_buzz_step(
        self, sample_mc_question: MCQuestion, sample_corpus: list[str]
    ) -> None:
        """Higher thresholds should produce later or equal buzz steps.

        A higher threshold means the agent needs more confidence to buzz,
        so it should wait at least as long as with a lower threshold.
        """
        likelihood = _make_likelihood(sample_corpus)
        thresholds = [0.3, 0.5, 0.7, 0.9]
        results = sweep_thresholds(
            questions=[sample_mc_question],
            likelihood_model=likelihood,
            thresholds=thresholds,
        )

        buzz_steps = [results[t][0].buzz_step for t in thresholds]
        for i in range(len(buzz_steps) - 1):
            assert buzz_steps[i] <= buzz_steps[i + 1], (
                f"Buzz step not monotonic: threshold {thresholds[i]} "
                f"(step {buzz_steps[i]}) > threshold {thresholds[i+1]} "
                f"(step {buzz_steps[i+1]})"
            )
