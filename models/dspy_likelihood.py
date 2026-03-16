"""DSPy-based likelihood model with score caching.

Wraps a DSPy listwise scorer behind the ``LikelihoodModel.score()``
interface.  Unlike embedding-based models, the DSPy scorer calls an LM
to rank options — so caching is at the *score* level (keyed by clue +
options + program fingerprint), not at the embedding level.

This module is importable without the ``dspy`` extra installed.
The ``dspy`` package is only required at runtime when a DSPy-backed
scorer is actually invoked (e.g. via ``scripts/optimize_dspy.py``).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from models.likelihoods import LikelihoodModel


def _score_cache_key(
    clue_prefix: str,
    option_profiles: list[str],
    program_fingerprint: str,
) -> str:
    """Build a deterministic cache key for a score() call."""
    payload = json.dumps(
        {"clue": clue_prefix, "options": option_profiles, "fp": program_fingerprint},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class DSPyLikelihood(LikelihoodModel):
    """LikelihoodModel subclass backed by a DSPy program.

    Inherits from ``LikelihoodModel`` so it satisfies the factory
    return type and isinstance checks.  Overrides ``score()`` with
    LM-based scoring and a score-level cache.  ``_embed_batch()`` raises
    ``NotImplementedError`` because DSPy scoring is not embedding-based.

    Unlike TF-IDF/SBERT/T5, this model does NOT produce embeddings.
    ``_embed_batch`` is explicitly unsupported — calling it raises
    ``NotImplementedError``.  Instead, scores are cached directly,
    keyed by ``(clue, options, program_fingerprint)``.

    Parameters
    ----------
    scorer : callable
        A DSPy module or function that accepts ``(clue_prefix, options)``
        and returns a list/array of K scores.
    program_fingerprint : str
        Opaque identifier for the current compiled program state.
        Cache entries are invalidated when this changes.
    cache_dir : str or Path or None
        Directory for persistent score cache.  When None, caching is
        in-memory only.
    """

    def __init__(
        self,
        scorer: Any,
        program_fingerprint: str = "default",
        cache_dir: str | Path | None = None,
    ) -> None:
        super().__init__()
        self.scorer = scorer
        self.program_fingerprint = program_fingerprint
        self._score_cache: dict[str, np.ndarray] = {}
        self._cache_dir = Path(cache_dir) if cache_dir else None
        if self._cache_dir:
            self._load_persistent_cache()

    def _load_persistent_cache(self) -> None:
        if self._cache_dir is None:
            return
        cache_file = self._cache_dir / f"dspy_scores_{self.program_fingerprint}.npz"
        if cache_file.exists():
            with np.load(cache_file, allow_pickle=False) as data:
                for key in data.files:
                    self._score_cache[key] = data[key].astype(np.float32)

    def _save_persistent_cache(self) -> None:
        if self._cache_dir is None or not self._score_cache:
            return
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = self._cache_dir / f"dspy_scores_{self.program_fingerprint}.npz"
        np.savez_compressed(cache_file, **self._score_cache)

    def score(self, clue_prefix: str, option_profiles: list[str]) -> np.ndarray:
        """Score answer options using the DSPy scorer.

        Results are cached by ``(clue, options, program_fingerprint)``.
        Validates that the returned array has shape ``(K,)`` where
        ``K = len(option_profiles)``.
        """
        key = _score_cache_key(clue_prefix, option_profiles, self.program_fingerprint)
        if key in self._score_cache:
            return self._score_cache[key].copy()

        raw = self.scorer(clue_prefix, option_profiles)
        scores = np.array(raw, dtype=np.float32)
        expected_k = len(option_profiles)
        if scores.ndim != 1 or len(scores) != expected_k:
            raise ValueError(
                f"DSPy scorer returned shape {scores.shape}, "
                f"expected ({expected_k},)"
            )
        self._score_cache[key] = scores
        return scores.copy()

    def save_cache(self, path: str | Path | None = None) -> int:
        """Persist score cache to disk."""
        if path:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(p, **self._score_cache)
        else:
            self._save_persistent_cache()
        return len(self._score_cache)

    def load_cache(self, path: str | Path) -> int:
        """Load score cache from disk, merging without overwriting."""
        p = Path(path)
        if not p.exists():
            return 0
        loaded = 0
        with np.load(p, allow_pickle=False) as data:
            for key in data.files:
                if key not in self._score_cache:
                    self._score_cache[key] = data[key].astype(np.float32)
                    loaded += 1
        return loaded

    @property
    def cache_memory_bytes(self) -> int:
        return sum(v.nbytes for v in self._score_cache.values())

    def _embed_batch(self, texts: list[str]) -> np.ndarray:
        """Not supported — DSPy scoring is not embedding-based."""
        raise NotImplementedError(
            "DSPyLikelihood does not produce embeddings. "
            "Use score() directly."
        )

    def embed_and_cache(self, texts: list[str]) -> np.ndarray:
        """Not supported — DSPy scoring is not embedding-based."""
        raise NotImplementedError(
            "DSPyLikelihood does not produce embeddings. "
            "Use score() directly."
        )
