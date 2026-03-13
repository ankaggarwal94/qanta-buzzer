from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from agents._math import sigmoid
from models.likelihoods import LikelihoodModel
from qb_data.mc_builder import MCQuestion



@dataclass
class SoftmaxEpisodeResult:
    qid: str
    buzz_step: int
    buzz_index: int
    gold_index: int
    correct: bool
    c_trace: list[float]
    g_trace: list[float]
    top_p_trace: list[float]
    entropy_trace: list[float]


class SoftmaxProfileBuzzer:
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

    def _belief_from_scratch(self, cumulative_prefix: str, option_profiles: list[str]) -> np.ndarray:
        scores = self.likelihood_model.score(cumulative_prefix, option_profiles)
        scores = scores - np.max(scores)
        probs = np.exp(self.beta * scores)
        probs = probs / max(1e-12, probs.sum())
        return probs.astype(np.float32)

    def confidence_proxy(self, top_p: float) -> float:
        return sigmoid(self.alpha * (top_p - self.threshold))

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
            top_idx = int(np.argmax(belief))
            top_p = float(np.max(belief))
            entropy = float(-(np.clip(belief, 1e-12, 1.0) * np.log(np.clip(belief, 1e-12, 1.0))).sum())
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
    ):
        self.likelihood_model = likelihood_model
        self.threshold = threshold
        self.beta = beta
        self.alpha = alpha

    def _step_update(self, prior: np.ndarray, fragment: str, option_profiles: list[str]) -> np.ndarray:
        scores = self.likelihood_model.score(fragment, option_profiles)
        scores = scores - np.max(scores)
        likelihood = np.exp(self.beta * scores)
        posterior = prior * likelihood
        denom = posterior.sum()
        if denom <= 0:
            return np.ones_like(prior) / len(prior)
        return (posterior / denom).astype(np.float32)

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
            top_idx = int(np.argmax(belief))
            top_p = float(np.max(belief))
            entropy = float(-(np.clip(belief, 1e-12, 1.0) * np.log(np.clip(belief, 1e-12, 1.0))).sum())
            c_t = sigmoid(self.alpha * (top_p - self.threshold))
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
            scores = scores - np.max(scores)
            likelihood = np.exp(beta * scores)
            posterior = belief * likelihood
            denom = posterior.sum()
            if denom <= 0:
                belief = np.ones_like(belief) / len(belief)
            else:
                belief = (posterior / denom).astype(np.float32)
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
) -> SoftmaxEpisodeResult:
    """Build a SoftmaxEpisodeResult from pre-computed sequential beliefs.

    Identical buzzing logic to ``SequentialBayesBuzzer.run_episode`` but
    reads beliefs from a ``_PrecomputedQuestion`` instead of calling the
    likelihood model.
    """
    from agents.threshold_buzzer import _belief_stats

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


def sweep_sequential_thresholds(
    questions: list[MCQuestion],
    likelihood_model: LikelihoodModel,
    thresholds: list[float],
    beta: float = 5.0,
    alpha: float = 10.0,
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
            _sequential_episode_from_precomputed(pq, threshold, alpha)
            for pq in precomputed
        ]
    return out
