"""Test suite for models/likelihoods.py — likelihood model interface and implementations.

Covers:
- LIK-01: LikelihoodModel ABC contract
- LIK-02: TfIdfLikelihood with corpus fitting and cosine scoring
- LIK-03: SBERTLikelihood with semantic embeddings and caching
"""

from __future__ import annotations

import numpy as np
import pytest

from models.likelihoods import (
    LikelihoodModel,
    SBERTLikelihood,
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
