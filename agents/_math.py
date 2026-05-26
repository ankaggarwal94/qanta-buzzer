from __future__ import annotations

import math

import numpy as np


def sigmoid(x: float) -> float:
    """Numerically stable logistic sigmoid for scalar confidence proxies."""
    if x >= 0.0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)

    z = math.exp(x)
    return z / (1.0 + z)


def softmax_belief(scores: np.ndarray, beta: float) -> np.ndarray:
    """Convert raw similarity scores to a belief distribution via temperature-scaled softmax.

    Defensive against nonfinite inputs: PR #14 review (Blocker 6) flagged
    that ``-inf - (-inf) -> nan`` propagates silently because the
    NumPy comparison ``nan <= 0`` evaluates to ``False``, bypassing
    the ``total <= 0`` guard and returning a NaN belief that
    contaminates ``belief_stats``, feature extraction, calibration,
    and PPO reward. The default pipeline (TF-IDF / SBERT / T5
    L2-normalized cosine) cannot trigger the failure, but DSPy
    plug-in scorers and large-beta configurations can. Empty arrays
    raise rather than return a degenerate vector. All non-finite
    inputs degrade to a uniform belief, matching the existing
    underflow fallback.
    """
    scores = np.asarray(scores)
    if scores.ndim != 1 or scores.size == 0:
        raise ValueError(
            f"softmax_belief expects a non-empty 1D array; "
            f"got shape={scores.shape}, size={scores.size}."
        )

    if not np.all(np.isfinite(scores)):
        return np.ones(scores.shape, dtype=np.float32) / scores.size

    shifted = scores - np.max(scores)
    probs = np.exp(beta * shifted)
    total = probs.sum()
    if not np.isfinite(total) or total <= 0:
        return np.ones(scores.shape, dtype=np.float32) / scores.size
    return (probs / total).astype(np.float32)


def bayesian_update(prior: np.ndarray, scores: np.ndarray, beta: float) -> np.ndarray:
    """Bayesian belief update: posterior = prior * softmax_likelihood, normalized.

    Defensive against nonfinite inputs and mismatched shapes; see
    ``softmax_belief`` for rationale. A single NaN belief at step t
    persists across all later sequential updates (NaN * anything =
    NaN), so a single bad ``scores`` vector contaminates the entire
    trajectory if not guarded here.
    """
    prior = np.asarray(prior)
    scores = np.asarray(scores)
    if scores.ndim != 1 or scores.size == 0:
        raise ValueError(
            f"bayesian_update expects a non-empty 1D scores array; "
            f"got shape={scores.shape}, size={scores.size}."
        )
    if prior.shape != scores.shape:
        raise ValueError(
            f"bayesian_update prior and scores must have matching shapes; "
            f"got prior={prior.shape}, scores={scores.shape}."
        )

    if not np.all(np.isfinite(scores)) or not np.all(np.isfinite(prior)):
        return np.ones(prior.shape, dtype=np.float32) / prior.size

    shifted = scores - np.max(scores)
    likelihood = np.exp(beta * shifted)
    posterior = prior * likelihood
    denom = posterior.sum()
    if not np.isfinite(denom) or denom <= 0:
        return np.ones(prior.shape, dtype=np.float32) / prior.size
    return (posterior / denom).astype(np.float32)


def belief_stats(belief: np.ndarray) -> tuple[int, float, float]:
    """Return (top_idx, top_p, entropy) from a belief distribution."""
    top_idx = int(np.argmax(belief))
    top_p = float(belief[top_idx])
    clipped = np.clip(belief, 1e-12, 1.0)
    entropy = float(-(clipped * np.log(clipped)).sum())
    return top_idx, top_p, entropy


def confidence_proxy(top_p: float, threshold: float, alpha: float) -> float:
    """Sigmoid confidence proxy: sigmoid(alpha * (top_p - threshold))."""
    return sigmoid(alpha * (top_p - threshold))
