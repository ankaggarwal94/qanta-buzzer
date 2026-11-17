from __future__ import annotations

from collections.abc import Iterator
from dataclasses import KW_ONLY, dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from agents._math import bayesian_update, softmax_belief
from agents._math import confidence_proxy as _confidence_proxy
from agents.threshold_buzzer import _ThresholdPolicy
from models.likelihoods import LikelihoodModel
from qb_data.mc_builder import MCQuestion

if TYPE_CHECKING:
    from agents.threshold_buzzer import _PrecomputedQuestion



@dataclass
class SoftmaxEpisodeResult:
    # Required core fields (legacy shape; positional construction
    # preserved for callers that match the pre-2026-05 signature).
    qid: str
    buzz_step: int
    buzz_index: int
    gold_index: int
    correct: bool
    c_trace: list[float] = field(default_factory=list)
    g_trace: list[float] = field(default_factory=list)
    top_p_trace: list[float] = field(default_factory=list)
    entropy_trace: list[float] = field(default_factory=list)
    # ``reward_like`` is keyword-only so positional callers that match
    # the legacy field order continue to work and callers that matched
    # any intermediate ordering get a ``TypeError`` instead of silently
    # mis-binding into ``c_trace``. Downstream metrics read this via
    # ``dict.get('reward_like', 0.0)`` on the asdict() form.
    _: KW_ONLY
    reward_like: float = 0.0


class SoftmaxProfileBuzzer(_ThresholdPolicy):
    def _belief_from_scratch(self, cumulative_prefix: str, option_profiles: list[str]) -> np.ndarray:
        scores = self.likelihood_model.score(cumulative_prefix, option_profiles)
        return softmax_belief(scores, self.beta)

    def confidence_proxy(self, top_p: float) -> float:
        return _confidence_proxy(top_p, self.threshold, self.alpha)

    def _beliefs(self, question: MCQuestion) -> Iterator[np.ndarray]:
        for prefix in question.cumulative_prefixes:
            self.belief = self._belief_from_scratch(prefix, question.option_profiles)
            yield self.belief

    def run_episode(self, question: MCQuestion) -> SoftmaxEpisodeResult:
        return self._replay(
            SoftmaxEpisodeResult, question, self._beliefs(question),
            self.confidence_proxy,
        )


class SequentialBayesBuzzer(_ThresholdPolicy):
    def _step_update(self, prior: np.ndarray, fragment: str, option_profiles: list[str]) -> np.ndarray:
        scores = self.likelihood_model.score(fragment, option_profiles)
        return bayesian_update(prior, scores, self.beta)

    def _beliefs(self, question: MCQuestion) -> Iterator[np.ndarray]:
        num_options = len(question.options)
        belief = np.ones(num_options, dtype=np.float32) / num_options
        for fragment in _clue_fragments(question):
            belief = self._step_update(belief, fragment, question.option_profiles)
            yield belief

    def run_episode(self, question: MCQuestion) -> SoftmaxEpisodeResult:
        return self._replay(
            SoftmaxEpisodeResult, question, self._beliefs(question),
            lambda p: _confidence_proxy(p, self.threshold, self.alpha),
        )


def _clue_fragments(question: MCQuestion) -> Iterator[str]:
    """Yield just the newly revealed tokens for sequential Bayesian updates."""
    start = 0
    for index in question.run_indices:
        yield " ".join(question.tokens[start:index + 1])
        start = index + 1


def precompute_sequential_beliefs(
    questions: list[MCQuestion],
    likelihood_model: LikelihoodModel,
    beta: float,
) -> list["_PrecomputedQuestion"]:
    """Compute Bayesian sequential beliefs at every step for every question.

    Starts with a uniform prior and applies Bayesian update
    ``posterior = prior * likelihood`` using token fragments derived from
    ``question.run_indices``.  Returns one ``_PrecomputedQuestion`` per
    question where ``beliefs`` are the Bayesian posteriors (NOT the
    from-scratch softmax beliefs).
    """
    from agents.threshold_buzzer import _PrecomputedQuestion

    out: list[_PrecomputedQuestion] = []
    for q in questions:
        K = len(q.options)
        belief = np.ones(K, dtype=np.float32) / K
        beliefs: list[np.ndarray] = []

        for fragment in _clue_fragments(q):
            scores = likelihood_model.score(fragment, q.option_profiles)
            belief = bayesian_update(belief, scores, beta)
            beliefs.append(belief.copy())

        out.append(_PrecomputedQuestion(
            qid=q.qid,
            gold_index=q.gold_index,
            num_options=K,
            beliefs=beliefs,
        ))
    return out


def _sequential_episode_from_precomputed(
    pq: "_PrecomputedQuestion",
    threshold: float,
    alpha: float,
    reward_mode: str = "time_penalty",
    wait_penalty: float = 0.0,
    buzz_correct: float = 1.0,
    buzz_incorrect: float = -0.5,
    early_buzz_penalty: float = 0.0,
) -> SoftmaxEpisodeResult:
    """Build a SoftmaxEpisodeResult from pre-computed sequential beliefs.

    Delegates to ``_softmax_episode_from_precomputed`` since the buzzing
    logic is identical regardless of how beliefs were computed.
    """
    from agents.threshold_buzzer import _softmax_episode_from_precomputed

    return _softmax_episode_from_precomputed(
        pq, threshold, alpha,
        reward_mode=reward_mode,
        wait_penalty=wait_penalty,
        buzz_correct=buzz_correct,
        buzz_incorrect=buzz_incorrect,
        early_buzz_penalty=early_buzz_penalty,
    )


def sweep_sequential_thresholds(
    questions: list[MCQuestion],
    likelihood_model: LikelihoodModel,
    thresholds: list[float],
    beta: float = 5.0,
    alpha: float = 10.0,
    reward_mode: str = "time_penalty",
    wait_penalty: float = 0.0,
    buzz_correct: float = 1.0,
    buzz_incorrect: float = -0.5,
    early_buzz_penalty: float = 0.0,
    precomputed: list["_PrecomputedQuestion"] | None = None,
) -> dict[float, list[SoftmaxEpisodeResult]]:
    """Sweep multiple thresholds with a single sequential belief pass.

    If *precomputed* is provided the expensive model calls are skipped
    entirely and the sweep is pure numpy.  Otherwise beliefs are computed
    once internally and reused across thresholds.
    """
    unique_thresholds = dict.fromkeys(thresholds)
    if not unique_thresholds:
        return {}
    if precomputed is None:
        precomputed = precompute_sequential_beliefs(questions, likelihood_model, beta)

    out: dict[float, list[SoftmaxEpisodeResult]] = {}
    for threshold in unique_thresholds:
        out[float(threshold)] = [
            _sequential_episode_from_precomputed(
                pq,
                threshold,
                alpha,
                reward_mode=reward_mode,
                wait_penalty=wait_penalty,
                buzz_correct=buzz_correct,
                buzz_incorrect=buzz_incorrect,
                early_buzz_penalty=early_buzz_penalty,
            )
            for pq in precomputed
        ]
    return out
