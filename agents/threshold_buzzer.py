from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from agents._math import sigmoid
from models.likelihoods import LikelihoodModel
from qb_data.mc_builder import MCQuestion


@dataclass
class EpisodeResult:
    qid: str
    buzz_step: int
    buzz_index: int
    gold_index: int
    correct: bool
    reward_like: float
    c_trace: list[float]
    g_trace: list[float]
    top_p_trace: list[float]
    entropy_trace: list[float]


def _scores_to_belief(scores: np.ndarray, beta: float) -> np.ndarray:
    """Convert raw similarity scores to a belief distribution via softmax."""
    shifted = scores - np.max(scores)
    probs = np.exp(beta * shifted)
    probs = probs / max(1e-12, probs.sum())
    return probs.astype(np.float32)


def _belief_stats(belief: np.ndarray) -> tuple[int, float, float]:
    """Return (top_idx, top_p, entropy) from a belief distribution."""
    top_idx = int(np.argmax(belief))
    top_p = float(belief[top_idx])
    clipped = np.clip(belief, 1e-12, 1.0)
    entropy = float(-(clipped * np.log(clipped)).sum())
    return top_idx, top_p, entropy


@dataclass
class _PrecomputedQuestion:
    """Pre-computed belief distributions for every clue step of one question."""
    qid: str
    gold_index: int
    num_options: int
    beliefs: list[np.ndarray]


def precompute_beliefs(
    questions: list[MCQuestion],
    likelihood_model: LikelihoodModel,
    beta: float,
) -> list[_PrecomputedQuestion]:
    """Compute beliefs at every step for every question (single model pass).

    After calling ``likelihood_model.precompute_embeddings()`` this is
    pure cache lookups + numpy math, so it runs in seconds rather than
    hours.
    """
    from tqdm import tqdm

    out: list[_PrecomputedQuestion] = []
    for q in tqdm(questions, desc="Computing beliefs"):
        beliefs = [
            _scores_to_belief(
                likelihood_model.score(prefix, q.option_profiles), beta
            )
            for prefix in q.cumulative_prefixes
        ]
        out.append(_PrecomputedQuestion(
            qid=q.qid,
            gold_index=q.gold_index,
            num_options=len(q.options),
            beliefs=beliefs,
        ))
    return out


class ThresholdBuzzer:
    def __init__(
        self,
        likelihood_model: LikelihoodModel,
        threshold: float = 0.8,
        beta: float = 5.0,
        alpha: float = 10.0,
    ):
        self.likelihood_model = likelihood_model
        self.threshold = threshold
        self.beta = beta
        self.alpha = alpha
        self.belief: np.ndarray | None = None

    def _belief_from_prefix(self, prefix: str, option_profiles: list[str]) -> np.ndarray:
        scores = self.likelihood_model.score(prefix, option_profiles)
        return _scores_to_belief(scores, self.beta)

    def _confidence_proxy(self, top_p: float) -> float:
        return sigmoid(self.alpha * (top_p - self.threshold))

    def run_episode(self, question: MCQuestion) -> EpisodeResult:
        c_trace: list[float] = []
        g_trace: list[float] = []
        top_p_trace: list[float] = []
        entropy_trace: list[float] = []

        chosen_step = len(question.cumulative_prefixes) - 1
        chosen_idx = 0

        for step_idx, prefix in enumerate(question.cumulative_prefixes):
            belief = self._belief_from_prefix(prefix, question.option_profiles)
            self.belief = belief
            top_idx, top_p, entropy = _belief_stats(belief)
            c_t = self._confidence_proxy(top_p)
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

        correct = chosen_idx == question.gold_index
        reward_like = 1.0 if correct else -0.5
        return EpisodeResult(
            qid=question.qid,
            buzz_step=chosen_step,
            buzz_index=chosen_idx,
            gold_index=question.gold_index,
            correct=correct,
            reward_like=reward_like,
            c_trace=c_trace,
            g_trace=g_trace,
            top_p_trace=top_p_trace,
            entropy_trace=entropy_trace,
        )


class AlwaysBuzzFinalBuzzer:
    def __init__(self, likelihood_model: LikelihoodModel, beta: float = 5.0):
        self.likelihood_model = likelihood_model
        self.beta = beta

    def run_episode(self, question: MCQuestion) -> EpisodeResult:
        c_trace: list[float] = []
        g_trace: list[float] = []
        top_p_trace: list[float] = []
        entropy_trace: list[float] = []

        final_step = len(question.cumulative_prefixes) - 1
        final_belief = np.ones(len(question.options), dtype=np.float32) / len(question.options)
        for prefix in question.cumulative_prefixes:
            scores = self.likelihood_model.score(prefix, question.option_profiles)
            probs = _scores_to_belief(scores, self.beta)
            final_belief = probs
            top_idx, top_p, entropy = _belief_stats(probs)
            c_trace.append(0.0)
            g_trace.append(1.0 if top_idx == question.gold_index else 0.0)
            top_p_trace.append(top_p)
            entropy_trace.append(entropy)

        c_trace[-1] = 1.0
        buzz_idx = int(np.argmax(final_belief))
        correct = buzz_idx == question.gold_index
        reward_like = 1.0 if correct else -0.5
        return EpisodeResult(
            qid=question.qid,
            buzz_step=final_step,
            buzz_index=buzz_idx,
            gold_index=question.gold_index,
            correct=correct,
            reward_like=reward_like,
            c_trace=c_trace,
            g_trace=g_trace,
            top_p_trace=top_p_trace,
            entropy_trace=entropy_trace,
        )


def _softmax_episode_from_precomputed(
    pq: _PrecomputedQuestion,
    threshold: float,
    alpha: float,
) -> "SoftmaxEpisodeResult":
    """Build a SoftmaxEpisodeResult from pre-computed beliefs (pure numpy).

    Identical buzzing logic to ``SoftmaxProfileBuzzer.run_episode`` but
    reads beliefs from a ``_PrecomputedQuestion`` instead of calling the
    likelihood model.
    """
    from agents.bayesian_buzzer import SoftmaxEpisodeResult

    c_trace: list[float] = []
    g_trace: list[float] = []
    top_p_trace: list[float] = []
    entropy_trace: list[float] = []

    chosen_step = len(pq.beliefs) - 1
    chosen_idx = 0

    for step_idx, belief in enumerate(pq.beliefs):
        top_idx, top_p, entropy = _belief_stats(belief)
        c_t = sigmoid(alpha * (top_p - threshold))
        g_t = 1.0 if top_idx == pq.gold_index else 0.0

        c_trace.append(c_t)
        g_trace.append(g_t)
        top_p_trace.append(top_p)
        entropy_trace.append(entropy)

        is_last = step_idx == len(pq.beliefs) - 1
        if top_p >= threshold or is_last:
            chosen_step = step_idx
            chosen_idx = top_idx
            break

    correct = chosen_idx == pq.gold_index
    return SoftmaxEpisodeResult(
        qid=pq.qid,
        buzz_step=chosen_step,
        buzz_index=chosen_idx,
        gold_index=pq.gold_index,
        correct=correct,
        c_trace=c_trace,
        g_trace=g_trace,
        top_p_trace=top_p_trace,
        entropy_trace=entropy_trace,
    )


def _always_final_from_precomputed(pq: _PrecomputedQuestion) -> EpisodeResult:
    """Build an EpisodeResult for AlwaysBuzzFinal from pre-computed beliefs.

    Iterates all beliefs (no early stopping), buzzes at the last step
    with argmax of the final belief.
    """
    c_trace: list[float] = []
    g_trace: list[float] = []
    top_p_trace: list[float] = []
    entropy_trace: list[float] = []

    for belief in pq.beliefs:
        top_idx, top_p, entropy = _belief_stats(belief)
        g_t = 1.0 if top_idx == pq.gold_index else 0.0
        c_trace.append(0.0)
        g_trace.append(g_t)
        top_p_trace.append(top_p)
        entropy_trace.append(entropy)

    c_trace[-1] = 1.0
    buzz_idx = int(np.argmax(pq.beliefs[-1]))
    correct = buzz_idx == pq.gold_index
    return EpisodeResult(
        qid=pq.qid,
        buzz_step=len(pq.beliefs) - 1,
        buzz_index=buzz_idx,
        gold_index=pq.gold_index,
        correct=correct,
        reward_like=1.0 if correct else -0.5,
        c_trace=c_trace,
        g_trace=g_trace,
        top_p_trace=top_p_trace,
        entropy_trace=entropy_trace,
    )


def _episode_from_precomputed(
    pq: _PrecomputedQuestion,
    threshold: float,
    alpha: float,
) -> EpisodeResult:
    """Build an EpisodeResult from pre-computed beliefs (pure numpy)."""
    c_trace: list[float] = []
    g_trace: list[float] = []
    top_p_trace: list[float] = []
    entropy_trace: list[float] = []

    chosen_step = len(pq.beliefs) - 1
    chosen_idx = 0

    for step_idx, belief in enumerate(pq.beliefs):
        top_idx, top_p, entropy = _belief_stats(belief)
        c_t = sigmoid(alpha * (top_p - threshold))
        g_t = 1.0 if top_idx == pq.gold_index else 0.0

        c_trace.append(c_t)
        g_trace.append(g_t)
        top_p_trace.append(top_p)
        entropy_trace.append(entropy)

        is_last = step_idx == len(pq.beliefs) - 1
        if top_p >= threshold or is_last:
            chosen_step = step_idx
            chosen_idx = top_idx
            break

    correct = chosen_idx == pq.gold_index
    return EpisodeResult(
        qid=pq.qid,
        buzz_step=chosen_step,
        buzz_index=chosen_idx,
        gold_index=pq.gold_index,
        correct=correct,
        reward_like=1.0 if correct else -0.5,
        c_trace=c_trace,
        g_trace=g_trace,
        top_p_trace=top_p_trace,
        entropy_trace=entropy_trace,
    )


def sweep_thresholds(
    questions: list[MCQuestion],
    likelihood_model: LikelihoodModel,
    thresholds: list[float],
    beta: float = 5.0,
    alpha: float = 10.0,
    precomputed: list[_PrecomputedQuestion] | None = None,
) -> dict[float, list[EpisodeResult]]:
    """Sweep multiple thresholds with a single belief-computation pass.

    If *precomputed* is provided the expensive model calls are skipped
    entirely and the sweep is pure numpy.  Otherwise beliefs are computed
    once internally and reused across thresholds.
    """
    if precomputed is None:
        precomputed = precompute_beliefs(questions, likelihood_model, beta)

    out: dict[float, list[EpisodeResult]] = {}
    for threshold in thresholds:
        out[float(threshold)] = [
            _episode_from_precomputed(pq, threshold, alpha)
            for pq in precomputed
        ]
    return out


def result_to_dict(result: EpisodeResult) -> dict[str, Any]:
    return {
        "qid": result.qid,
        "buzz_step": result.buzz_step,
        "buzz_index": result.buzz_index,
        "gold_index": result.gold_index,
        "correct": result.correct,
        "reward_like": result.reward_like,
        "c_trace": result.c_trace,
        "g_trace": result.g_trace,
        "top_p_trace": result.top_p_trace,
        "entropy_trace": result.entropy_trace,
    }
