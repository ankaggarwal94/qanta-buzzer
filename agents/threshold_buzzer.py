from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, Protocol, TypeVar

import numpy as np

from agents._math import belief_stats, confidence_proxy, softmax_belief
from models.likelihoods import LikelihoodModel
from qb_data.mc_builder import MCQuestion

if TYPE_CHECKING:
    from agents.bayesian_buzzer import SoftmaxEpisodeResult


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
    return softmax_belief(scores, beta)


def _belief_stats(belief: np.ndarray) -> tuple[int, float, float]:
    """Return (top_idx, top_p, entropy) from a belief distribution."""
    return belief_stats(belief)


def reward_from_buzz_step(
    *,
    correct: bool,
    buzz_step: int,
    total_steps: int,
    reward_mode: str,
    wait_penalty: float,
    buzz_correct: float,
    buzz_incorrect: float,
    early_buzz_penalty: float,
) -> float:
    """Replay canonical env reward from an offline buzz outcome.

    The offline baseline stores ``buzz_step`` as the last-seen clue
    index. Seeing clue ``buzz_step`` requires ``buzz_step + 1`` WAIT
    actions in the env (clues 0 through buzz_step), each charged
    ``wait_penalty``. Early-buzz shaping uses the same ``buzz_step``
    progress formula as ``TossupMCEnv._buzz_reward``.
    """
    if reward_mode == "simple":
        return 1.0 if correct else -1.0

    if reward_mode != "time_penalty":
        import warnings
        warnings.warn(
            f"reward_from_buzz_step: reward_mode={reward_mode!r} is not supported; "
            "returning 0.0. Only 'simple' and 'time_penalty' are implemented.",
            stacklevel=2,
        )
        return 0.0

    reward = buzz_correct if correct else buzz_incorrect
    if early_buzz_penalty > 0 and total_steps > 1:
        progress = np.clip((buzz_step + 1) / total_steps, 0.0, 1.0)
        reward -= float(early_buzz_penalty) * (1.0 - progress)
    reward -= float(wait_penalty) * (buzz_step + 1)
    return float(reward)


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


_Result = TypeVar("_Result", EpisodeResult, "SoftmaxEpisodeResult")


class _RewardAtBuzz(Protocol):
    def __call__(self, *, correct: bool, buzz_step: int, total_steps: int) -> float: ...


def _threshold_episode(
    result_type: type[_Result],
    beliefs: Iterable[np.ndarray],
    *,
    qid: str,
    gold_index: int,
    step_count: Callable[[], int],
    reached: Callable[[float], bool],
    confidence: Callable[[float], float],
    reward: _RewardAtBuzz,
) -> _Result:
    """Replay a threshold policy over live or cached beliefs, stopping lazily."""
    c_trace: list[float] = []
    g_trace: list[float] = []
    top_p_trace: list[float] = []
    entropy_trace: list[float] = []
    chosen_step = step_count() - 1
    chosen_idx = 0
    for step, belief in enumerate(beliefs):
        top_idx, top_p, entropy = belief_stats(belief)
        c_trace.append(confidence(top_p))
        g_trace.append(float(top_idx == gold_index))
        top_p_trace.append(top_p)
        entropy_trace.append(entropy)
        if reached(top_p) or step == step_count() - 1:
            chosen_step, chosen_idx = step, top_idx
            break

    correct = chosen_idx == gold_index
    return result_type(
        qid=qid,
        buzz_step=chosen_step,
        buzz_index=chosen_idx,
        gold_index=gold_index,
        correct=correct,
        reward_like=reward(correct=correct, buzz_step=chosen_step, total_steps=step_count()),
        c_trace=c_trace,
        g_trace=g_trace,
        top_p_trace=top_p_trace,
        entropy_trace=entropy_trace,
    )


class _ThresholdPolicy:
    """Settings and decision logic shared by the live threshold policies."""

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

    def _reward_at_buzz(self, *, correct: bool, buzz_step: int, total_steps: int) -> float:
        return reward_from_buzz_step(
            correct=correct, buzz_step=buzz_step, total_steps=total_steps,
            reward_mode=self.reward_mode, wait_penalty=self.wait_penalty,
            buzz_correct=self.buzz_correct, buzz_incorrect=self.buzz_incorrect,
            early_buzz_penalty=self.early_buzz_penalty,
        )

    def _replay(
        self,
        result_type: type[_Result],
        question: MCQuestion,
        beliefs: Iterable[np.ndarray],
        confidence: Callable[[float], float],
    ) -> _Result:
        return _threshold_episode(
            result_type, beliefs, qid=question.qid, gold_index=question.gold_index,
            step_count=lambda: len(question.cumulative_prefixes),
            reached=lambda p: p >= self.threshold, confidence=confidence,
            reward=self._reward_at_buzz,
        )


class ThresholdBuzzer(_ThresholdPolicy):
    def _belief_from_prefix(self, prefix: str, option_profiles: list[str]) -> np.ndarray:
        scores = self.likelihood_model.score(prefix, option_profiles)
        return _scores_to_belief(scores, self.beta)

    def _confidence_proxy(self, top_p: float) -> float:
        return confidence_proxy(top_p, self.threshold, self.alpha)

    def _beliefs(self, question: MCQuestion) -> Iterator[np.ndarray]:
        for prefix in question.cumulative_prefixes:
            self.belief = self._belief_from_prefix(prefix, question.option_profiles)
            yield self.belief

    def run_episode(self, question: MCQuestion) -> EpisodeResult:
        return self._replay(EpisodeResult, question, self._beliefs(question), self._confidence_proxy)


class AlwaysBuzzFinalBuzzer:
    def __init__(
        self,
        likelihood_model: LikelihoodModel,
        beta: float = 5.0,
        reward_mode: str = "time_penalty",
        wait_penalty: float = 0.0,
        buzz_correct: float = 1.0,
        buzz_incorrect: float = -0.5,
        early_buzz_penalty: float = 0.0,
    ):
        self.likelihood_model = likelihood_model
        self.beta = beta
        self.reward_mode = reward_mode
        self.wait_penalty = wait_penalty
        self.buzz_correct = buzz_correct
        self.buzz_incorrect = buzz_incorrect
        self.early_buzz_penalty = early_buzz_penalty

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
        reward_like = reward_from_buzz_step(
            correct=correct,
            buzz_step=final_step,
            total_steps=len(question.cumulative_prefixes),
            reward_mode=self.reward_mode,
            wait_penalty=self.wait_penalty,
            buzz_correct=self.buzz_correct,
            buzz_incorrect=self.buzz_incorrect,
            early_buzz_penalty=self.early_buzz_penalty,
        )
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


def _cached_threshold_episode(
    result_type: type[_Result],
    pq: _PrecomputedQuestion,
    threshold: float,
    alpha: float,
    reward_mode: str = "time_penalty",
    wait_penalty: float = 0.0,
    buzz_correct: float = 1.0,
    buzz_incorrect: float = -0.5,
    early_buzz_penalty: float = 0.0,
) -> _Result:
    return _threshold_episode(
        result_type, pq.beliefs,
        qid=pq.qid, gold_index=pq.gold_index, step_count=lambda: len(pq.beliefs),
        reached=lambda p: p >= threshold,
        confidence=partial(confidence_proxy, threshold=threshold, alpha=alpha),
        reward=partial(
            reward_from_buzz_step, reward_mode=reward_mode, wait_penalty=wait_penalty,
            buzz_correct=buzz_correct, buzz_incorrect=buzz_incorrect, early_buzz_penalty=early_buzz_penalty,
        ),
    )


def _softmax_episode_from_precomputed(
    pq: _PrecomputedQuestion,
    threshold: float,
    alpha: float,
    reward_mode: str = "time_penalty",
    wait_penalty: float = 0.0,
    buzz_correct: float = 1.0,
    buzz_incorrect: float = -0.5,
    early_buzz_penalty: float = 0.0,
) -> "SoftmaxEpisodeResult":
    """Build a SoftmaxEpisodeResult from pre-computed beliefs (pure numpy).

    Identical buzzing logic to ``SoftmaxProfileBuzzer.run_episode`` but
    reads beliefs from a ``_PrecomputedQuestion`` instead of calling the
    likelihood model.
    """
    from agents.bayesian_buzzer import SoftmaxEpisodeResult

    return _cached_threshold_episode(
        SoftmaxEpisodeResult, pq, threshold, alpha, reward_mode, wait_penalty,
        buzz_correct, buzz_incorrect, early_buzz_penalty,
    )


def _always_final_from_precomputed(
    pq: _PrecomputedQuestion,
    reward_mode: str = "time_penalty",
    wait_penalty: float = 0.0,
    buzz_correct: float = 1.0,
    buzz_incorrect: float = -0.5,
    early_buzz_penalty: float = 0.0,
) -> EpisodeResult:
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
        reward_like=reward_from_buzz_step(
            correct=correct,
            buzz_step=len(pq.beliefs) - 1,
            total_steps=len(pq.beliefs),
            reward_mode=reward_mode,
            wait_penalty=wait_penalty,
            buzz_correct=buzz_correct,
            buzz_incorrect=buzz_incorrect,
            early_buzz_penalty=early_buzz_penalty,
        ),
        c_trace=c_trace,
        g_trace=g_trace,
        top_p_trace=top_p_trace,
        entropy_trace=entropy_trace,
    )


def _episode_from_precomputed(
    pq: _PrecomputedQuestion,
    threshold: float,
    alpha: float,
    reward_mode: str = "time_penalty",
    wait_penalty: float = 0.0,
    buzz_correct: float = 1.0,
    buzz_incorrect: float = -0.5,
    early_buzz_penalty: float = 0.0,
) -> EpisodeResult:
    """Build an EpisodeResult from pre-computed beliefs (pure numpy)."""
    return _cached_threshold_episode(
        EpisodeResult, pq, threshold, alpha, reward_mode, wait_penalty,
        buzz_correct, buzz_incorrect, early_buzz_penalty,
    )


def sweep_thresholds(
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
    precomputed: list[_PrecomputedQuestion] | None = None,
) -> dict[float, list[EpisodeResult]]:
    """Sweep multiple thresholds with a single belief-computation pass.

    If *precomputed* is provided the expensive model calls are skipped
    entirely and the sweep is pure numpy.  Otherwise beliefs are computed
    once internally and reused across thresholds.
    """
    unique_thresholds = dict.fromkeys(thresholds)
    if not unique_thresholds:
        return {}
    if precomputed is None:
        precomputed = precompute_beliefs(questions, likelihood_model, beta)

    out: dict[float, list[EpisodeResult]] = {}
    for threshold in unique_thresholds:
        out[float(threshold)] = [
            _episode_from_precomputed(
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
