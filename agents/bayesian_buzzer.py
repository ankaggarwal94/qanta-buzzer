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
