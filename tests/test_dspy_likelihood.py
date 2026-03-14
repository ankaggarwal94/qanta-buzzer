"""Tests for models/dspy_likelihood.py — DSPy-backed scorer with cache."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from models.dspy_likelihood import DSPyLikelihood, _score_cache_key


def _fake_scorer(clue: str, options: list[str]) -> list[float]:
    """Return uniform scores sized to the option list."""
    return [1.0 / len(options)] * len(options)


class TestDSPyLikelihood:
    def test_score_returns_ndarray_k(self) -> None:
        model = DSPyLikelihood(scorer=_fake_scorer)
        scores = model.score("clue text", ["A", "B", "C", "D"])
        assert scores.shape == (4,)
        assert scores.dtype == np.float32

    def test_repeated_call_hits_cache(self) -> None:
        call_count = 0

        def counting_scorer(clue, options):
            nonlocal call_count
            call_count += 1
            return [1.0] * len(options)

        model = DSPyLikelihood(scorer=counting_scorer)
        model.score("clue", ["A", "B"])
        model.score("clue", ["A", "B"])
        assert call_count == 1

    def test_changed_fingerprint_invalidates(self) -> None:
        model1 = DSPyLikelihood(scorer=_fake_scorer, program_fingerprint="v1")
        model2 = DSPyLikelihood(scorer=_fake_scorer, program_fingerprint="v2")
        model1.score("clue", ["A", "B"])
        assert len(model2._score_cache) == 0

    def test_persistence_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cache.npz"
            model = DSPyLikelihood(scorer=_fake_scorer)
            model.score("clue", ["A", "B", "C"])
            saved = model.save_cache(path)
            assert saved == 1

            model2 = DSPyLikelihood(scorer=_fake_scorer)
            loaded = model2.load_cache(path)
            assert loaded == 1
            np.testing.assert_array_equal(
                model2.score("clue", ["A", "B", "C"]),
                model.score("clue", ["A", "B", "C"]),
            )

    def test_embed_batch_raises(self) -> None:
        model = DSPyLikelihood(scorer=_fake_scorer)
        with pytest.raises(NotImplementedError):
            model._embed_batch(["text"])

    def test_cache_memory_bytes(self) -> None:
        model = DSPyLikelihood(scorer=_fake_scorer)
        assert model.cache_memory_bytes == 0
        model.score("c", ["A"])
        assert model.cache_memory_bytes > 0
