"""Test suite for models/likelihoods.py — likelihood model interface and implementations.

Covers:
- LIK-01: LikelihoodModel ABC contract
- LIK-02: TfIdfLikelihood with corpus fitting and cosine scoring
- LIK-03: SBERTLikelihood with semantic embeddings and caching
- LIK-04: T5Likelihood semantic scoring and embedding shape
- LIK-05: T5 embedding cache reuse and factory construction
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from models.likelihoods import (
    LikelihoodModel,
    SBERTLikelihood,
    T5Likelihood,
    TfIdfLikelihood,
)


# ------------------------------------------------------------------ #
# Tests for LikelihoodModel ABC
# ------------------------------------------------------------------ #


class TestLikelihoodModelABC:
    """Tests for the abstract base class contract."""

    def test_abstract_interface_cannot_instantiate(self) -> None:
        """LikelihoodModel ABC cannot be instantiated directly."""
        with pytest.raises(TypeError):
            LikelihoodModel()  # type: ignore[abstract]

    def test_embedding_cache_on_subclass(self, sample_corpus: list[str]) -> None:
        """Concrete subclass inherits embedding_cache dict."""
        model = TfIdfLikelihood(corpus_texts=sample_corpus)
        assert hasattr(model, "embedding_cache"), "Missing embedding_cache attribute"
        assert isinstance(model.embedding_cache, dict), "embedding_cache should be dict"


def test_embedding_misses_are_unique_and_return_in_request_order(monkeypatch):
    model = TfIdfLikelihood(["apple berry", "cherry date"])
    original = model._embed_batch
    calls = []

    def record(texts):
        calls.append(list(texts))
        return original(texts)

    monkeypatch.setattr(model, "_embed_batch", record)
    model.precompute_embeddings(["date", "date"])
    texts = ["berry", "date", "berry", "apple", "date"]
    actual = model.embed_and_cache(texts)
    np.testing.assert_array_equal(actual, original(texts))
    assert calls == [["date"], ["berry", "apple"]]


@pytest.mark.parametrize("method", ["embed_and_cache", "precompute_embeddings"])
@pytest.mark.parametrize("shape", [(1, 4), (3, 4), (2,), (2, 5)])
def test_malformed_embedding_batch_is_rejected_before_cache_update(monkeypatch, method, shape):
    model = TfIdfLikelihood(["apple berry", "cherry date"])
    old = model.embed_and_cache(["date"])
    old_keys = set(model.embedding_cache)
    original = model._embed_batch
    monkeypatch.setattr(model, "_embed_batch", lambda texts: np.zeros(shape))

    with pytest.raises(ValueError, match="embedding batch"):
        getattr(model, method)(["berry", "apple"])
    assert set(model.embedding_cache) == old_keys
    np.testing.assert_array_equal(model.embed_and_cache(["date"]), old)

    monkeypatch.setattr(model, "_embed_batch", original)
    np.testing.assert_array_equal(model.embed_and_cache(["berry", "apple"]), original(["berry", "apple"]))


def test_tfidf_refit_updates_the_vocabulary_and_cached_coordinates():
    model = TfIdfLikelihood(["apple banana", "cherry date"])
    model.embed_and_cache(["banana", "apple"])
    corpus = ["zebra banana", "cherry", "yarrow", "zebra", "lemon"]
    model.fit(corpus)
    reference = TfIdfLikelihood(corpus)
    texts = ["banana", "apple", "lemon"]
    np.testing.assert_array_equal(model.embed_and_cache(texts), reference.embed_and_cache(texts))
    np.testing.assert_array_equal(model.score("zebra", texts), reference.score("zebra", texts))


def test_failed_tfidf_refit_retains_last_working_model():
    corpus = ["apple banana", "cherry date"]
    model = TfIdfLikelihood(corpus)
    model.embed_and_cache(["banana"])
    with pytest.raises(ValueError, match="empty vocabulary"):
        model.fit(["the"])
    scores = model.score("apple banana", ["banana", "date"])
    np.testing.assert_array_equal(scores, TfIdfLikelihood(corpus).score("apple banana", ["banana", "date"]))


class _LocalEmbeddings(LikelihoodModel):
    def _embed_batch(self, texts):
        return np.array([[len(text), 1] for text in texts], dtype=np.float32)

    def score(self, clue_prefix, option_profiles):
        return self.embed_and_cache(option_profiles) @ self.embed_and_cache([clue_prefix])[0]


def test_bulk_precompute_releases_hashes_for_warm_entries():
    import tracemalloc
    from models.likelihoods import _text_key

    model = _LocalEmbeddings()
    texts = [f"warm-{i}" for i in range(8192)]
    vector = np.ones(2, dtype=np.float32)
    model.embedding_cache.update({_text_key(text): vector for text in texts})
    model.precompute_embeddings([])  # Exclude the initial tqdm import.
    tracemalloc.start()
    try:
        model.precompute_embeddings(texts)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    # Holding 8,192 new 64-character digests alone would exceed this bound.
    assert peak < 700 * 1024


@pytest.mark.parametrize("backend, encoder", [
    (SBERTLikelihood, SimpleNamespace(get_embedding_dimension=lambda: 2)),
    (SBERTLikelihood, SimpleNamespace(get_sentence_embedding_dimension=lambda: 2)),
    (T5Likelihood, SimpleNamespace(config=SimpleNamespace(d_model=2))),
])
def test_known_encoder_width_rejects_stale_archives(tmp_path, monkeypatch, backend, encoder):
    from models.likelihoods import _text_key

    model = backend.__new__(backend)
    LikelihoodModel.__init__(model)
    model.encoder = encoder
    monkeypatch.setattr(model, "_embed_batch", _LocalEmbeddings()._embed_batch)
    archive = tmp_path / "embeddings.npz"
    np.savez(archive, **{_text_key("old"): np.ones(3)})
    with pytest.warns(RuntimeWarning, match="width"):
        assert model.load_cache(archive) == 0
    assert not model.embedding_cache
    np.testing.assert_array_equal(model.embed_and_cache(["fresh"]), [[5, 1]])

    # An incompatible first row must not conceal a malformed later row.
    np.savez(archive, **{_text_key("old"): np.ones(3), _text_key("bad"): np.ones((3, 1))})
    with pytest.raises(ValueError, match="embedding"):
        model.load_cache(archive)
    assert list(model.embedding_cache) == [_text_key("fresh")]

    np.savez(archive, **{_text_key("old"): np.array([9, 8]), _text_key("fresh"): np.zeros(2)})
    assert model.load_cache(archive) == 1
    np.testing.assert_array_equal(model.embed_and_cache(["fresh", "old"]), [[5, 1], [9, 8]])


def test_unknown_width_mismatch_names_cache_recovery(tmp_path):
    from models.likelihoods import _text_key

    model = _LocalEmbeddings()
    archive = tmp_path / "legacy.npz"
    np.savez(archive, **{_text_key("old"): np.ones(3)})
    assert model.load_cache(archive) == 1
    with pytest.raises(ValueError, match="rebuild"):
        model.embed_and_cache(["fresh"])
    assert list(model.embedding_cache) == [_text_key("old")]


@pytest.mark.parametrize("warm", [False, True])
@pytest.mark.parametrize("bad_shape", [(3,), (1, 2), (0,)])
def test_legacy_npz_load_rejects_late_corruption_atomically(tmp_path, warm, bad_shape):
    from models.likelihoods import _text_key

    model = _LocalEmbeddings()
    if warm:
        model.embed_and_cache(["warm"])
    old = model.embedding_cache.copy()
    path = tmp_path / "cache.npz"
    rows = {_text_key("first"): np.array([7, 2], dtype=np.float64), _text_key("last"): np.zeros(bad_shape)}
    np.savez(path, **rows)
    with pytest.raises(ValueError, match="embedding"):
        model.load_cache(path)
    assert list(model.embedding_cache) == list(old)
    for key in old:
        assert old[key] is model.embedding_cache[key]

    # Valid legacy files merge into a warm cache, converting to float32. A
    # conflicting duplicate is ignored even when its on-disk shape is invalid.
    model.embed_and_cache(["warm"])
    rows.pop(_text_key("last"))
    rows[_text_key("warm")] = np.zeros((2, 3))
    np.savez_compressed(path, **rows)
    assert model.load_cache(path) == 1
    result = model.embed_and_cache(["warm", "first"])
    assert result.dtype == np.float32
    np.testing.assert_array_equal(result, [[4, 1], [7, 2]])


# ------------------------------------------------------------------ #
# Tests for TfIdfLikelihood
# ------------------------------------------------------------------ #


class TestTfIdfLikelihood:
    """Tests for TF-IDF based likelihood model."""

    def test_tfidf_requires_fit(self) -> None:
        """score() before fit() raises RuntimeError."""
        model = TfIdfLikelihood()
        with pytest.raises(RuntimeError, match="must be fit"):
            model.score("test clue", ["option1", "option2"])

    def test_tfidf_embed_requires_fit(self) -> None:
        """_embed_batch() before fit() raises RuntimeError."""
        model = TfIdfLikelihood()
        with pytest.raises(RuntimeError, match="must be fit"):
            model._embed_batch(["test text"])

    def test_tfidf_fit_and_score(self, sample_corpus: list[str]) -> None:
        """After fitting, score returns correct shape and dtype.

        Also verifies that more relevant text scores higher.
        """
        model = TfIdfLikelihood()
        model.fit(sample_corpus)

        scores = model.score(
            "Who was the first president?",
            ["George Washington first president", "Abraham Lincoln Civil War"],
        )
        assert scores.shape == (2,), f"Expected shape (2,), got {scores.shape}"
        assert scores.dtype == np.float32, f"Expected float32, got {scores.dtype}"
        # Washington should score higher for "first president" clue
        assert scores[0] >= scores[1], (
            f"Washington ({scores[0]:.3f}) should score >= Lincoln ({scores[1]:.3f})"
        )

    def test_tfidf_embed_batch(self, sample_corpus: list[str]) -> None:
        """_embed_batch produces dense vectors of correct shape."""
        model = TfIdfLikelihood(corpus_texts=sample_corpus)
        embeddings = model._embed_batch(["test text one", "test text two"])
        assert embeddings.shape[0] == 2, f"Expected 2 rows, got {embeddings.shape[0]}"
        assert embeddings.dtype == np.float32, f"Expected float32, got {embeddings.dtype}"
        vocab_size = len(model.vectorizer.vocabulary_)
        assert embeddings.shape[1] == vocab_size, (
            f"Expected {vocab_size} cols, got {embeddings.shape[1]}"
        )

    def test_tfidf_corpus_in_constructor(self, sample_corpus: list[str]) -> None:
        """Passing corpus_texts to __init__ auto-fits the model."""
        model = TfIdfLikelihood(corpus_texts=sample_corpus)
        assert model._is_fit is True, "Model should be fit after corpus in constructor"
        # Should work without explicit fit()
        scores = model.score("president", ["Washington", "Lincoln"])
        assert scores.shape == (2,)

    def test_tfidf_fit_returns_self(self, sample_corpus: list[str]) -> None:
        """fit() returns self for method chaining."""
        model = TfIdfLikelihood()
        result = model.fit(sample_corpus)
        assert result is model, "fit() should return self"

    def test_tfidf_score_all_options(self, sample_corpus: list[str]) -> None:
        """Score works with 4 options matching K=4 environment setup."""
        model = TfIdfLikelihood(corpus_texts=sample_corpus)
        scores = model.score(
            "first president United States",
            [
                "George Washington commander revolutionary",
                "Thomas Jefferson declaration independence",
                "John Adams Massachusetts diplomat",
                "Benjamin Franklin inventor Philadelphia",
            ],
        )
        assert scores.shape == (4,), f"Expected shape (4,), got {scores.shape}"
        assert all(np.isfinite(scores)), "All scores should be finite"

    def test_tfidf_embed_batch_normalized(self, sample_corpus: list[str]) -> None:
        """_embed_batch returns L2-normalized vectors (row norms ~1.0)."""
        model = TfIdfLikelihood(corpus_texts=sample_corpus)
        embeddings = model._embed_batch(["George Washington president", "Thomas Jefferson"])
        norms = np.linalg.norm(embeddings, axis=1)
        np.testing.assert_array_almost_equal(norms, np.ones(2), decimal=5)

    def test_tfidf_score_uses_cache(self, sample_corpus: list[str]) -> None:
        """score() populates embedding_cache via embed_and_cache()."""
        model = TfIdfLikelihood(corpus_texts=sample_corpus)
        assert len(model.embedding_cache) == 0
        model.score("first president", ["Washington profile", "Lincoln profile"])
        assert len(model.embedding_cache) == 3  # 1 clue + 2 options

    def test_tfidf_score_cache_hit(self, sample_corpus: list[str]) -> None:
        """Repeated score() with same options reuses cache."""
        model = TfIdfLikelihood(corpus_texts=sample_corpus)
        options = ["George Washington president", "Thomas Jefferson declaration"]
        model.score("first president", options)
        cache_after_first = len(model.embedding_cache)
        model.score("second president", options)
        # Only the new clue should be added; options are cached
        assert len(model.embedding_cache) == cache_after_first + 1

    def test_tfidf_score_matches_cosine_reference(self, sample_corpus: list[str]) -> None:
        """New cached score() matches sklearn cosine_similarity reference."""
        from sklearn.metrics.pairwise import cosine_similarity as sklearn_cos

        model = TfIdfLikelihood(corpus_texts=sample_corpus)
        clue = "Who was the first president?"
        options = [
            "George Washington first president commander revolutionary",
            "Abraham Lincoln Civil War emancipation",
            "Thomas Jefferson declaration independence Virginia",
            "Benjamin Franklin inventor Philadelphia diplomat",
        ]
        # Compute reference via sklearn cosine_similarity (old method)
        clue_vec = model.vectorizer.transform([clue])
        option_vecs = model.vectorizer.transform(options)
        ref_scores = sklearn_cos(clue_vec, option_vecs)[0].astype(np.float32)
        # Compute via new cached path
        actual_scores = model.score(clue, options)
        np.testing.assert_allclose(actual_scores, ref_scores, atol=1e-6)


# ------------------------------------------------------------------ #
# Tests for SBERTLikelihood
# ------------------------------------------------------------------ #


class TestSBERTLikelihood:
    """Tests for Sentence-BERT likelihood model."""

    def test_sbert_instantiation(self) -> None:
        """SBERTLikelihood can be instantiated with default model."""
        model = SBERTLikelihood()
        assert hasattr(model, "encoder"), "Missing encoder attribute"
        assert model.model_name == "all-MiniLM-L6-v2"

    def test_sbert_score_shape_and_dtype(self) -> None:
        """score() returns correct shape and dtype for 4 options."""
        model = SBERTLikelihood()
        scores = model.score(
            "first president United States",
            [
                "George Washington first president commander",
                "Thomas Jefferson third president declaration",
                "John Adams second president Massachusetts",
                "Benjamin Franklin inventor diplomat",
            ],
        )
        assert scores.shape == (4,), f"Expected shape (4,), got {scores.shape}"
        assert scores.dtype == np.float32, f"Expected float32, got {scores.dtype}"

    def test_sbert_semantic_ranking(self) -> None:
        """SBERT ranks semantically similar text higher."""
        model = SBERTLikelihood()
        scores = model.score(
            "George Washington was the first president of the United States and led the Continental Army",
            [
                "George Washington first president commander revolutionary war continental army",
                "The theory of relativity was developed by Albert Einstein in physics",
            ],
        )
        # Washington profile should score much higher than Einstein
        assert scores[0] > scores[1], (
            f"Washington ({scores[0]:.3f}) should score > Einstein ({scores[1]:.3f})"
        )

    def test_sbert_embedding_cache_populated(self) -> None:
        """Embedding cache grows after first scoring call."""
        model = SBERTLikelihood()
        assert len(model.embedding_cache) == 0, "Cache should start empty"

        model.score("test clue", ["option A", "option B"])
        cache_after_first = len(model.embedding_cache)
        assert cache_after_first > 0, "Cache should be populated after score()"

    def test_sbert_embedding_cache_hit(self) -> None:
        """Repeated calls with same text use cache (size unchanged)."""
        model = SBERTLikelihood()
        scores1 = model.score("test clue", ["option A", "option B"])
        cache_size_1 = len(model.embedding_cache)

        scores2 = model.score("test clue", ["option A", "option B"])
        cache_size_2 = len(model.embedding_cache)

        assert cache_size_2 == cache_size_1, (
            f"Cache grew from {cache_size_1} to {cache_size_2} on repeated call"
        )
        np.testing.assert_array_almost_equal(
            scores1, scores2, decimal=5,
            err_msg="Cached results should match original",
        )

    def test_sbert_normalized_embeddings(self) -> None:
        """SBERT embeddings are L2-normalized (norm ~ 1.0)."""
        model = SBERTLikelihood()
        embeddings = model._embed_batch(["test sentence one", "test sentence two"])
        norms = np.linalg.norm(embeddings, axis=1)
        np.testing.assert_array_almost_equal(
            norms, np.ones(2), decimal=4,
            err_msg="Embeddings should be L2-normalized",
        )


# ------------------------------------------------------------------ #
# Tests for T5Likelihood (LIK-04, LIK-05)
# ------------------------------------------------------------------ #


class TestT5Likelihood:
    """Tests for T5 encoder likelihood model.

    Uses the sample_t5_model fixture (t5-small, module-scoped) from conftest.py
    so the model is loaded once per test file, not per test function.
    """

    def test_t5_semantic_scoring(self, sample_t5_model) -> None:
        """T5 should score semantically relevant options higher (LIK-04).

        "First president" clue should rank Washington higher than Einstein,
        demonstrating that T5 captures semantic similarity between question
        content and answer profiles.
        """
        clue = "This person was the first president of the United States"
        options = [
            "George Washington first president commander revolutionary war",
            "Albert Einstein physicist theory relativity Nobel Prize",
        ]

        scores = sample_t5_model.score(clue, options)

        assert isinstance(scores, np.ndarray)
        assert scores.dtype == np.float32
        assert len(scores) == 2
        # Washington should score higher than Einstein for "first president" query
        assert scores[0] > scores[1], (
            f"Expected Washington > Einstein, got {scores}"
        )

    def test_t5_embedding_cache(self, sample_t5_model) -> None:
        """T5 should cache embeddings and reuse them (LIK-05).

        After embedding two texts, the cache should contain 2 entries.
        Re-embedding the same texts should not grow the cache, and the
        returned embeddings should be identical.
        """
        # Clear cache to get a clean test
        sample_t5_model.embedding_cache.clear()

        texts = ["George Washington", "Thomas Jefferson"]

        # First call embeds and caches
        emb1 = sample_t5_model.embed_and_cache(texts)
        cache_size_1 = len(sample_t5_model.embedding_cache)

        # Second call reuses cache
        emb2 = sample_t5_model.embed_and_cache(texts)
        cache_size_2 = len(sample_t5_model.embedding_cache)

        np.testing.assert_array_equal(
            emb1, emb2, err_msg="Cached embeddings should match"
        )
        assert cache_size_1 == cache_size_2 == 2, (
            f"Cache size should not grow on reuse, got {cache_size_1} -> {cache_size_2}"
        )

    def test_t5_score_returns_float32(self, sample_t5_model) -> None:
        """T5 score should return float32 array, not probabilities.

        Scores are raw cosine similarities (not softmax probabilities),
        so they do not necessarily sum to 1.
        """
        scores = sample_t5_model.score("test clue", ["option 1", "option 2"])
        assert scores.dtype == np.float32
        assert scores.shape == (2,)
        # Scores are raw similarities, not probabilities (don't sum to 1)
        assert all(np.isfinite(scores)), "All scores should be finite"

    def test_build_t5_from_config(self) -> None:
        """Factory should construct T5Likelihood from config (LIK-04).

        The build_likelihood_from_config factory should recognize
        model="t5" and instantiate a T5Likelihood with the specified
        t5_name parameter.
        """
        from models.likelihoods import T5Likelihood, build_likelihood_from_config

        config = {
            "likelihood": {
                "model": "t5",
                "t5_name": "t5-small",
            }
        }

        model = build_likelihood_from_config(config)
        assert isinstance(model, T5Likelihood)
        assert model.model_name == "t5-small"

    def test_t5_handles_variable_length(self, sample_t5_model) -> None:
        """T5 should handle variable-length texts via attention mask.

        Short and long texts should both embed without error, producing
        embeddings of the same hidden dimension regardless of input length.
        """
        short = "Washington"
        long = (
            "George Washington was the first president of the United States "
            "and commander of the Continental Army during the Revolutionary War"
        )

        # Both should embed without error
        embs = sample_t5_model.embed_and_cache([short, long])
        assert embs.shape == (2, sample_t5_model.encoder.config.d_model), (
            f"Expected shape (2, {sample_t5_model.encoder.config.d_model}), "
            f"got {embs.shape}"
        )


# ------------------------------------------------------------------ #
# Tests for Embedding Cache Persistence
# ------------------------------------------------------------------ #


class TestEmbeddingCachePersistence:
    """Tests for save_cache / load_cache disk persistence on LikelihoodModel."""

    def test_save_load_cache_round_trip(self, tmp_path: Path, sample_corpus: list[str]) -> None:
        """save_cache writes .npz; load_cache restores identical entries."""
        model = SBERTLikelihood()
        texts = ["George Washington", "Thomas Jefferson", "Abraham Lincoln"]
        model.embed_and_cache(texts)
        assert len(model.embedding_cache) == 3

        cache_path = tmp_path / "cache.npz"
        saved = model.save_cache(cache_path)
        assert saved == 3
        assert cache_path.exists()

        model2 = SBERTLikelihood()
        assert len(model2.embedding_cache) == 0
        loaded = model2.load_cache(cache_path)
        assert loaded == 3

        for key in model.embedding_cache:
            np.testing.assert_array_equal(
                model.embedding_cache[key],
                model2.embedding_cache[key],
                err_msg=f"Mismatch for key {key}",
            )

    def test_load_cache_missing_file(self, tmp_path: Path) -> None:
        """load_cache with nonexistent file returns 0 and leaves cache empty."""
        model = SBERTLikelihood()
        result = model.load_cache(tmp_path / "nonexistent.npz")
        assert result == 0
        assert len(model.embedding_cache) == 0

    def test_save_cache_empty(self, tmp_path: Path) -> None:
        """save_cache with empty cache creates a valid .npz with zero arrays."""
        model = SBERTLikelihood()
        cache_path = tmp_path / "empty.npz"
        saved = model.save_cache(cache_path)
        assert saved == 0
        assert cache_path.exists()

        # Should be loadable
        model2 = SBERTLikelihood()
        loaded = model2.load_cache(cache_path)
        assert loaded == 0

    def test_tfidf_save_cache_noop(self, sample_corpus: list[str]) -> None:
        """TfIdfLikelihood.save_cache is a no-op returning 0."""
        model = TfIdfLikelihood(corpus_texts=sample_corpus)
        # Populate the cache with some embeddings
        model.embed_and_cache(["test text one", "test text two"])
        assert len(model.embedding_cache) > 0

        import tempfile
        import os
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "should_not_exist.npz"
            result = model.save_cache(path)
            assert result == 0
            assert not path.exists(), "TfIdfLikelihood should NOT write a cache file"

    def test_load_cache_does_not_overwrite(self, tmp_path: Path) -> None:
        """load_cache merges without overwriting existing cache entries."""
        model = SBERTLikelihood()
        texts = ["Hello world"]
        model.embed_and_cache(texts)

        # Save this cache
        cache_path = tmp_path / "cache.npz"
        model.save_cache(cache_path)

        # Create a second model, pre-populate with the same key but different value
        model2 = SBERTLikelihood()
        from models.likelihoods import _text_key
        key = _text_key("Hello world")
        original_value = np.ones(384, dtype=np.float32)  # dummy
        model2.embedding_cache[key] = original_value

        loaded = model2.load_cache(cache_path)
        assert loaded == 0, "Key already present, so nothing should be loaded"

        # Original value should be preserved (not overwritten)
        np.testing.assert_array_equal(
            model2.embedding_cache[key],
            original_value,
            err_msg="Existing cache entry was overwritten by load_cache",
        )


class TestCacheMemory:
    """Verify cache_memory_bytes property for resource monitoring."""

    def test_tfidf_cache_memory_bytes(self, sample_corpus):
        model = TfIdfLikelihood(corpus_texts=sample_corpus)
        assert model.cache_memory_bytes == 0
        model.embed_and_cache(["George Washington"])
        assert model.cache_memory_bytes > 0

    def test_empty_cache_zero_bytes(self, sample_corpus):
        model = TfIdfLikelihood(corpus_texts=sample_corpus)
        assert model.cache_memory_bytes == 0
