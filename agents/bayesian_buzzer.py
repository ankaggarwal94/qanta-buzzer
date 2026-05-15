from __future__ import annotations

from dataclasses import KW_ONLY, dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from agents._math import bayesian_update, belief_stats, sigmoid, softmax_belief
from agents._math import confidence_proxy as _confidence_proxy
from agents.threshold_buzzer import reward_from_buzz_step
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


class SoftmaxProfileBuzzer:
    def __init__(
        self,
        likelihood_model: LikelihoodModel,
        threshold: float = 0.8,
        beta: float = 5.0,
        alpha: float = 10.0,
        reward_mode: str = "time_penalty",
        wait_penalty: float = 0.0,
        buzz_correct: float = 1.0,
        buzz_incorrect: float = -0.5,
        early_buzz_penalty: float = 0.0,
    ):
        self.likelihood_model = likelihood_model
        self.threshold = threshold
        self.beta = beta
        self.alpha = alpha
        self.reward_mode = reward_mode
        self.wait_penalty = wait_penalty
        self.buzz_correct = buzz_correct
        self.buzz_incorrect = buzz_incorrect
        self.early_buzz_penalty = early_buzz_penalty
        self.belief: np.ndarray | None = None

    def _belief_from_scratch(self, cumulative_prefix: str, option_profiles: list[str]) -> np.ndarray:
        scores = self.likelihood_model.score(cumulative_prefix, option_profiles)
        return softmax_belief(scores, self.beta)

    def confidence_proxy(self, top_p: float) -> float:
        return _confidence_proxy(top_p, self.threshold, self.alpha)

    def run_episode(self, question: MCQuestion) -> SoftmaxEpisodeResult:
        c_trace: list[float] = []
        g_trace: list[float] = []
        top_p_trace: list[float] = []
        entropy_trace: list[float] = []

        chosen_idx = 0
        chosen_step = len(question.cumulative_prefixes) - 1

        for step_idx, prefix in enumerate(question.cumulative_prefixes):
            belief = self._belief_from_scratch(prefix, question.option_profiles)
            self.belief = belief
            top_idx, top_p, entropy = belief_stats(belief)
            c_t = self.confidence_proxy(top_p)
            g_t = 1.0 if top_idx == question.gold_index else 0.0

            c_trace.append(c_t)
            g_trace.append(g_t)
            top_p_trace.append(top_p)
            entropy_trace.append(entropy)

            is_last = step_idx == len(question.cumulative_prefixes) - 1
            if top_p >= self.threshold or is_last:
                chosen_step = step_idx
                chosen_idx = top_idx
                break

        return SoftmaxEpisodeResult(
            qid=question.qid,
            buzz_step=chosen_step,
            buzz_index=chosen_idx,
            gold_index=question.gold_index,
            correct=(chosen_idx == question.gold_index),
            reward_like=reward_from_buzz_step(
                correct=(chosen_idx == question.gold_index),
                buzz_step=chosen_step,
                total_steps=len(question.cumulative_prefixes),
                reward_mode=self.reward_mode,
                wait_penalty=self.wait_penalty,
                buzz_correct=self.buzz_correct,
                buzz_incorrect=self.buzz_incorrect,
                early_buzz_penalty=self.early_buzz_penalty,
            ),
            c_trace=c_trace,
            g_trace=g_trace,
            top_p_trace=top_p_trace,
            entropy_trace=entropy_trace,
        )


class SequentialBayesBuzzer:
    def __init__(
        self,
        likelihood_model: LikelihoodModel,
        threshold: float = 0.8,
        beta: float = 5.0,
        alpha: float = 10.0,
        reward_mode: str = "time_penalty",
        wait_penalty: float = 0.0,
        buzz_correct: float = 1.0,
        buzz_incorrect: float = -0.5,
        early_buzz_penalty: float = 0.0,
    ):
        self.likelihood_model = likelihood_model
        self.threshold = threshold
        self.beta = beta
        self.alpha = alpha
        self.reward_mode = reward_mode
        self.wait_penalty = wait_penalty
        self.buzz_correct = buzz_correct
        self.buzz_incorrect = buzz_incorrect
        self.early_buzz_penalty = early_buzz_penalty

    def _step_update(self, prior: np.ndarray, fragment: str, option_profiles: list[str]) -> np.ndarray:
        scores = self.likelihood_model.score(fragment, option_profiles)
        return bayesian_update(prior, scores, self.beta)

    def run_episode(self, question: MCQuestion) -> SoftmaxEpisodeResult:
        c_trace: list[float] = []
        g_trace: list[float] = []
        top_p_trace: list[float] = []
        entropy_trace: list[float] = []

        K = len(question.options)
        belief = np.ones(K, dtype=np.float32) / K
        chosen_idx = 0
        chosen_step = len(question.cumulative_prefixes) - 1

        for step_idx, token_idx in enumerate(question.run_indices):
            prev_token_idx = question.run_indices[step_idx - 1] if step_idx > 0 else -1
            fragment = " ".join(question.tokens[prev_token_idx + 1 : token_idx + 1])
            belief = self._step_update(belief, fragment, question.option_profiles)
            top_idx, top_p, entropy = belief_stats(belief)
            c_t = _confidence_proxy(top_p, self.threshold, self.alpha)
            g_t = 1.0 if top_idx == question.gold_index else 0.0

            c_trace.append(c_t)
            g_trace.append(g_t)
            top_p_trace.append(top_p)
            entropy_trace.append(entropy)

            is_last = step_idx == len(question.cumulative_prefixes) - 1
            if top_p >= self.threshold or is_last:
                chosen_step = step_idx
                chosen_idx = top_idx
                break

        return SoftmaxEpisodeResult(
            qid=question.qid,
            buzz_step=chosen_step,
            buzz_index=chosen_idx,
            gold_index=question.gold_index,
            correct=(chosen_idx == question.gold_index),
            reward_like=reward_from_buzz_step(
                correct=(chosen_idx == question.gold_index),
                buzz_step=chosen_step,
                total_steps=len(question.cumulative_prefixes),
                reward_mode=self.reward_mode,
                wait_penalty=self.wait_penalty,
                buzz_correct=self.buzz_correct,
                buzz_incorrect=self.buzz_incorrect,
                early_buzz_penalty=self.early_buzz_penalty,
            ),
            c_trace=c_trace,
            g_trace=g_trace,
            top_p_trace=top_p_trace,
            entropy_trace=entropy_trace,
        )


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

        for step_idx, token_idx in enumerate(q.run_indices):
            prev_token_idx = q.run_indices[step_idx - 1] if step_idx > 0 else -1
            fragment = " ".join(q.tokens[prev_token_idx + 1 : token_idx + 1])
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
    if precomputed is None:
        precomputed = precompute_sequential_beliefs(questions, likelihood_model, beta)

    out: dict[float, list[SoftmaxEpisodeResult]] = {}
    for threshold in thresholds:
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
