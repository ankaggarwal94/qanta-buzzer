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
        """Different fingerprints produce different cache keys for same input."""
        key_v1 = _score_cache_key("clue", ["A", "B"], "v1")
        key_v2 = _score_cache_key("clue", ["A", "B"], "v2")
        assert key_v1 != key_v2, "Fingerprint must affect cache key"

        model = DSPyLikelihood(scorer=_fake_scorer, program_fingerprint="v1")
        model.score("clue", ["A", "B"])
        assert key_v1 in model._score_cache
        assert key_v2 not in model._score_cache

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

    def test_score_shape_validation(self) -> None:
        """Scorer returning wrong length raises ValueError."""
        def bad_scorer(clue, options):
            return [1.0, 2.0]  # always 2, ignoring len(options)

        model = DSPyLikelihood(scorer=bad_scorer)
        with pytest.raises(ValueError, match="expected"):
            model.score("clue", ["A", "B", "C", "D"])

    def test_isinstance_likelihood_model(self) -> None:
        """DSPyLikelihood is a proper LikelihoodModel subclass."""
        from models.likelihoods import LikelihoodModel
        model = DSPyLikelihood(scorer=_fake_scorer)
        assert isinstance(model, LikelihoodModel)


class TestScoreVectorValidation:
    """PR #31 external review: fresh AND cached score vectors must fail loud.

    Cached entries previously bypassed all validation (returned before the
    shape check), and NaN/+inf were never rejected anywhere — downstream
    softmax_belief deliberately collapses those to a uniform belief, which
    would silently recreate the inert-uniform failure mode.
    """

    def test_cached_wrong_shape_fails_loud(self) -> None:
        model = DSPyLikelihood(scorer=_fake_scorer)
        key = _score_cache_key("clue", ["A", "B"], model.program_fingerprint)
        model._score_cache[key] = np.array([0.9], dtype=np.float32)
        with pytest.raises(ValueError, match="cached"):
            model.score("clue", ["A", "B"])

    def test_scorer_nan_fails_loud(self) -> None:
        model = DSPyLikelihood(scorer=lambda c, o: [float("nan")] * len(o))
        with pytest.raises(ValueError, match="NaN"):
            model.score("clue", ["A", "B"])

    def test_scorer_posinf_fails_loud(self) -> None:
        model = DSPyLikelihood(scorer=lambda c, o: [float("inf"), 0.0])
        with pytest.raises(ValueError, match=r"\+inf"):
            model.score("clue", ["A", "B"])

    def test_scorer_all_neginf_fails_loud(self) -> None:
        model = DSPyLikelihood(scorer=lambda c, o: [float("-inf")] * len(o))
        with pytest.raises(ValueError, match="finite"):
            model.score("clue", ["A", "B"])

    def test_mixed_neginf_allowed(self) -> None:
        # Documented impossible-option semantics (agents._math.softmax_belief):
        # finite + -inf mixes stay valid; -inf slots get zero mass downstream.
        model = DSPyLikelihood(scorer=lambda c, o: [0.7, float("-inf")])
        scores = model.score("clue", ["A", "B"])
        assert scores.shape == (2,)

    def test_load_cache_rejects_nan_entries(self, tmp_path) -> None:
        bad = tmp_path / "bad_cache.npz"
        np.savez_compressed(bad, somekey=np.array([np.nan, 0.5], dtype=np.float32))
        model = DSPyLikelihood(scorer=_fake_scorer)
        with pytest.raises(ValueError, match="NaN"):
            model.load_cache(bad)
