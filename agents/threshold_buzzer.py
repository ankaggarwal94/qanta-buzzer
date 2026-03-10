from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

<<<<<<< HEAD
=======
from agents._math import sigmoid
>>>>>>> cda02951d4f40d4e7f14fbb2626d3740699830af
from models.likelihoods import LikelihoodModel
from qb_data.mc_builder import MCQuestion


<<<<<<< HEAD
def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


=======
>>>>>>> cda02951d4f40d4e7f14fbb2626d3740699830af
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
        scores = scores - np.max(scores)
        probs = np.exp(self.beta * scores)
        probs = probs / max(1e-12, probs.sum())
        return probs.astype(np.float32)

    def _confidence_proxy(self, top_p: float) -> float:
<<<<<<< HEAD
        return _sigmoid(self.alpha * (top_p - self.threshold))
=======
        return sigmoid(self.alpha * (top_p - self.threshold))
>>>>>>> cda02951d4f40d4e7f14fbb2626d3740699830af

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
            top_p = float(np.max(belief))
            top_idx = int(np.argmax(belief))
            entropy = float(-(np.clip(belief, 1e-12, 1.0) * np.log(np.clip(belief, 1e-12, 1.0))).sum())
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
            scores = scores - np.max(scores)
            probs = np.exp(self.beta * scores)
            probs = probs / max(1e-12, probs.sum())
            final_belief = probs
            top_idx = int(np.argmax(probs))
            top_p = float(np.max(probs))
            entropy = float(-(np.clip(probs, 1e-12, 1.0) * np.log(np.clip(probs, 1e-12, 1.0))).sum())
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


def sweep_thresholds(
    questions: list[MCQuestion],
    likelihood_model: LikelihoodModel,
    thresholds: list[float],
    beta: float = 5.0,
    alpha: float = 10.0,
) -> dict[float, list[EpisodeResult]]:
    out: dict[float, list[EpisodeResult]] = {}
    for threshold in thresholds:
        agent = ThresholdBuzzer(
            likelihood_model=likelihood_model,
            threshold=float(threshold),
            beta=beta,
            alpha=alpha,
        )
        out[float(threshold)] = [agent.run_episode(q) for q in questions]
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
