"""
Belief Feature Extraction

Extracts derived features from belief probability distributions for use as
policy observations. Given a belief vector of K probabilities (one per answer
option), produces a (K + 6)-dimensional feature vector containing:

    belief[0..K-1]   raw belief probabilities
    top_p             max belief probability
    margin            gap between top two probabilities
    entropy           Shannon entropy of the distribution
    stability         L1 distance from previous belief (0 if first step)
    progress          fraction of total clue steps elapsed
    clue_idx_norm     normalized clue index (0 to 1 over steps)

Ported from qb-rl reference implementation (models/features.py).
"""

from __future__ import annotations

import numpy as np


def entropy_of_distribution(prob: np.ndarray) -> float:
    """Compute Shannon entropy of a probability distribution.

    Uses clipping for numerical stability to avoid log(0).

    Parameters
    ----------
    prob : np.ndarray
        1D probability vector. Values should sum to ~1.0.

    Returns
    -------
    float
        Shannon entropy H(p) = -sum(p * log(p)), non-negative.

    Examples
    --------
    >>> import numpy as np
    >>> uniform = np.array([0.25, 0.25, 0.25, 0.25])
    >>> abs(entropy_of_distribution(uniform) - 1.3863) < 0.001
    True
    """
    clipped = np.clip(prob, 1e-12, 1.0)
    return float(-(clipped * np.log(clipped)).sum())


def extract_belief_features(
    belief: np.ndarray,
    prev_belief: np.ndarray | None,
    step_idx: int,
    total_steps: int,
) -> np.ndarray:
    """Extract derived features from a belief probability vector.

    Concatenates the raw belief with 6 derived scalar features to produce
    a fixed-size observation vector for the RL policy.

    Parameters
    ----------
    belief : np.ndarray
        1D probability vector of shape (K,) over answer options.
    prev_belief : np.ndarray or None
        Previous step's belief vector, same shape as ``belief``.
        Pass None on the first step (stability will be 0.0).
    step_idx : int
        Current clue step index (0-based).
    total_steps : int
        Total number of clue steps in the episode.

    Returns
    -------
    np.ndarray
        Feature vector of shape (K + 6,) with dtype float32.
        Layout: [belief..., top_p, margin, entropy, stability, progress, clue_idx_norm].

    Raises
    ------
    ValueError
        If ``belief`` is not a 1D array.

    Examples
    --------
    >>> import numpy as np
    >>> belief = np.array([0.5, 0.3, 0.15, 0.05], dtype=np.float32)
    >>> feats = extract_belief_features(belief, None, 2, 6)
    >>> feats.shape
    (10,)
    >>> feats.dtype
    dtype('float32')
    """
    belief = np.asarray(belief, dtype=np.float32)
    if belief.ndim != 1:
        raise ValueError("belief must be a 1D probability vector")

    top_p = float(np.max(belief))
    sorted_probs = np.sort(belief)[::-1]
    second = float(sorted_probs[1]) if len(sorted_probs) > 1 else 0.0
    margin = top_p - second
    ent = entropy_of_distribution(belief)
    stability = float(np.abs(belief - prev_belief).sum()) if prev_belief is not None else 0.0
    progress = float(step_idx / max(1, total_steps))
    clue_idx_norm = float(step_idx / max(1, total_steps - 1))

    extras = np.array([top_p, margin, ent, stability, progress, clue_idx_norm], dtype=np.float32)
    return np.concatenate([belief, extras]).astype(np.float32)


def extract_padded_belief_features(
    belief: np.ndarray,
    prev_belief: np.ndarray | None,
    step_idx: int,
    total_steps: int,
    max_K: int,
) -> np.ndarray:
    """Extract belief features padded to a fixed ``max_K`` size.

    Identical to :func:`extract_belief_features` except the belief
    segment is zero-padded (or truncated) to exactly ``max_K`` elements,
    producing a ``(max_K + 6)``-dimensional vector regardless of the
    actual number of answer options.

    Parameters
    ----------
    belief : np.ndarray
        1D probability vector of shape (K_actual,).
    prev_belief : np.ndarray or None
        Previous belief vector (same shape as *belief*).
    step_idx : int
        Current clue step index (0-based).
    total_steps : int
        Total clue steps in the episode.
    max_K : int
        Target padded length for the belief segment.

    Returns
    -------
    np.ndarray
        Feature vector of shape (max_K + 6,), dtype float32.
    """
    belief = np.asarray(belief, dtype=np.float32)
    if belief.ndim != 1:
        raise ValueError("belief must be a 1D probability vector")

    K_actual = len(belief)

    top_p = float(np.max(belief))
    sorted_probs = np.sort(belief)[::-1]
    second = float(sorted_probs[1]) if len(sorted_probs) > 1 else 0.0
    margin = top_p - second
    ent = entropy_of_distribution(belief)
    stability = float(np.abs(belief - prev_belief).sum()) if prev_belief is not None else 0.0
    progress = float(step_idx / max(1, total_steps))
    clue_idx_norm = float(step_idx / max(1, total_steps - 1))

    padded = np.zeros(max_K, dtype=np.float32)
    padded[:K_actual] = belief[:max_K]
    extras = np.array([top_p, margin, ent, stability, progress, clue_idx_norm], dtype=np.float32)
    return np.concatenate([padded, extras]).astype(np.float32)
