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
    """Convert raw similarity scores to a belief distribution via temperature-scaled softmax."""
    shifted = scores - np.max(scores)
    probs = np.exp(beta * shifted)
    total = probs.sum()
    if total <= 0:
        return np.ones_like(probs, dtype=np.float32) / len(probs)
    return (probs / total).astype(np.float32)


def bayesian_update(prior: np.ndarray, scores: np.ndarray, beta: float) -> np.ndarray:
    """Bayesian belief update: posterior = prior * softmax_likelihood, normalized."""
    shifted = scores - np.max(scores)
    likelihood = np.exp(beta * shifted)
    posterior = prior * likelihood
    denom = posterior.sum()
    if denom <= 0:
        return np.ones_like(prior, dtype=np.float32) / len(prior)
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
