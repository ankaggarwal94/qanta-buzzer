"""Exact, bounded-memory profile-ranking regressions for issue #25."""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import qb_data.mc_builder as mc_builder_module
from qb_data.mc_builder import MCBuilder


def _canonical_rankings(
    similarity: np.ndarray,
    answers: list[str],
    M: int,
) -> dict[str, list[str]]:
    """Full-matrix oracle for the inherited top-M ranking contract.

    The locked implementation sorts every column when ``M >= N - 1``;
    otherwise it selects ``M`` raw columns, then removes self.  Preserve that
    ordering and list-length behavior exactly; issue #25 changes allocation
    strategy, not the candidate-pool contract.  Equal scores are made
    deterministic by canonical column index.
    """
    rankings: dict[str, list[str]] = {}
    for row_index, answer in enumerate(answers):
        raw_limit = len(answers) if M >= len(answers) - 1 else M
        order = sorted(
            range(len(answers)),
            key=lambda column_index: (
                -float(similarity[row_index, column_index]),
                column_index,
            ),
        )[:raw_limit]
        rankings[answer] = [
            answers[column_index]
            for column_index in order
            if column_index != row_index
        ]
    return rankings


def _graded_profiles(n: int) -> tuple[list[str], dict[str, str]]:
    answers = [f"answer-{index:02d}" for index in range(n)]
    vocabulary = [f"gradedtoken{index:02d}" for index in range(n)]
    profiles = {
        answer: " ".join(vocabulary[: index + 1])
        for index, answer in enumerate(answers)
    }
    return answers, profiles


class _RecordedEmbeddings:
    """Small ndarray wrapper that records every similarity matmul shape."""

    def __init__(
        self,
        values: np.ndarray,
        calls: list[tuple[int, int]],
    ) -> None:
        self.values = values
        self.calls = calls

    @property
    def T(self) -> "_RecordedEmbeddings":
        return _RecordedEmbeddings(self.values.T, self.calls)

    def __getitem__(self, key: object) -> "_RecordedEmbeddings":
        return _RecordedEmbeddings(self.values[key], self.calls)

    def __matmul__(self, other: object) -> np.ndarray:
        assert isinstance(other, _RecordedEmbeddings)
        self.calls.append((self.values.shape[0], other.values.shape[1]))
        return self.values @ other.values


def _embedding_fixture(n: int) -> np.ndarray:
    rng = np.random.default_rng(20260728)
    values = rng.normal(size=(n, 7)).astype(np.float32)
    return values / np.linalg.norm(values, axis=1, keepdims=True)


def _install_embedding_backend(
    monkeypatch: pytest.MonkeyPatch,
    strategy: str,
    values: np.ndarray,
    calls: list[tuple[int, int]],
) -> None:
    if strategy == "sbert_profile":
        module = types.ModuleType("sentence_transformers")

        class FakeSentenceTransformer:
            def __init__(self, model: str) -> None:
                assert model

            def encode(
                self,
                docs: list[str],
                *,
                convert_to_numpy: bool,
                normalize_embeddings: bool,
            ) -> _RecordedEmbeddings:
                assert len(docs) == len(values)
                assert convert_to_numpy is True
                assert normalize_embeddings is True
                return _RecordedEmbeddings(values, calls)

        module.SentenceTransformer = FakeSentenceTransformer
        monkeypatch.setitem(sys.modules, "sentence_transformers", module)
        return

    module = types.ModuleType("models.likelihoods")

    class FakeOpenAILikelihood:
        def __init__(self, model: str) -> None:
            assert model

        def embed_and_cache(
            self, docs: list[str]
        ) -> _RecordedEmbeddings:
            assert len(docs) == len(values)
            return _RecordedEmbeddings(values, calls)

    module.OpenAILikelihood = FakeOpenAILikelihood
    monkeypatch.setitem(sys.modules, "models.likelihoods", module)


@pytest.mark.parametrize(
    "strategy",
    ["tfidf_profile", "sbert_profile", "openai_profile"],
)
@pytest.mark.parametrize("variable_k", [False, True])
def test_profile_rankings_are_exact_and_similarity_is_row_blocked(
    monkeypatch: pytest.MonkeyPatch,
    strategy: str,
    variable_k: bool,
) -> None:
    n = 37
    block_size = 7
    answers = [f"answer-{index:02d}" for index in range(n)]
    profiles = {
        answer: f"sharedtoken uniquetoken{index:02d}"
        for index, answer in enumerate(answers)
    }
    calls: list[tuple[int, int]] = []

    if strategy == "tfidf_profile":
        docs = [profiles[answer] for answer in answers]
        matrix = TfidfVectorizer(stop_words="english").fit_transform(docs)
        expected_similarity = cosine_similarity(matrix, matrix)
        real_cosine_similarity = mc_builder_module.cosine_similarity

        def recorded_cosine_similarity(
            left: object,
            right: object,
        ) -> np.ndarray:
            result = real_cosine_similarity(left, right)
            calls.append((result.shape[0], result.shape[1]))
            return result

        monkeypatch.setattr(
            mc_builder_module,
            "cosine_similarity",
            recorded_cosine_similarity,
        )
    else:
        values = _embedding_fixture(n)
        expected_similarity = values @ values.T
        _install_embedding_backend(monkeypatch, strategy, values, calls)

    builder = MCBuilder(
        K=4,
        strategy=strategy,
        variable_K=variable_k,
        min_K=2,
        max_K=4,
        similarity_block_size=block_size,
    )
    actual = builder._compute_rankings(answers, profiles, {})
    M = min(max(5 * builder.K, 30), n - 1)

    assert actual == _canonical_rankings(expected_similarity, answers, M)
    assert calls == [
        (min(block_size, n - start), n)
        for start in range(0, n, block_size)
    ]
    assert max(row_count for row_count, _ in calls) <= block_size
    assert all(column_count == n for _, column_count in calls)


@pytest.mark.parametrize("n", [9, 37])
def test_ties_use_canonical_answer_index_for_small_and_partition_paths(
    n: int,
) -> None:
    answers = [f"answer-{index:02d}" for index in range(n)]
    profiles = {answer: "identical shared profile" for answer in answers}
    builder = MCBuilder(
        K=4,
        strategy="tfidf_profile",
        similarity_block_size=5,
    )

    actual = builder._compute_rankings(answers, profiles, {})
    similarity = np.ones((n, n), dtype=np.float64)
    M = min(max(5 * builder.K, 30), n - 1)

    assert actual == _canonical_rankings(similarity, answers, M)


def test_small_n_non_tied_rankings_match_full_matrix_oracle() -> None:
    answers, profiles = _graded_profiles(9)
    docs = [profiles[answer] for answer in answers]
    matrix = TfidfVectorizer(stop_words="english").fit_transform(docs)
    similarity = cosine_similarity(matrix, matrix)
    assert any(
        len(set(similarity[row_index])) > 2
        for row_index in range(len(answers))
    )
    builder = MCBuilder(
        K=4,
        strategy="tfidf_profile",
        similarity_block_size=4,
    )
    M = min(max(5 * builder.K, 30), len(answers) - 1)

    assert builder._compute_rankings(answers, profiles, {}) == (
        _canonical_rankings(similarity, answers, M)
    )


def test_variable_k_preserves_k_based_ranking_pool_size() -> None:
    answers, profiles = _graded_profiles(47)
    docs = [profiles[answer] for answer in answers]
    matrix = TfidfVectorizer(stop_words="english").fit_transform(docs)
    similarity = cosine_similarity(matrix, matrix)
    builder = MCBuilder(
        K=8,
        strategy="tfidf_profile",
        variable_K=True,
        min_K=2,
        max_K=3,
        similarity_block_size=11,
    )
    # The inherited contract defines M from K, not from a sampled target K or
    # max_K.  A max_K-based implementation would retain only 30 raw columns.
    M = min(max(5 * builder.K, 30), len(answers) - 1)
    actual = builder._compute_rankings(answers, profiles, {})

    assert M == 40
    assert actual == _canonical_rankings(similarity, answers, M)
    assert any(len(ranking) > 30 for ranking in actual.values())


@pytest.mark.parametrize("bad_block_size", [True, 0, -1, 1.5, "7"])
def test_similarity_block_size_must_be_a_positive_integer(
    bad_block_size: object,
) -> None:
    with pytest.raises(ValueError, match="similarity_block_size"):
        MCBuilder(similarity_block_size=bad_block_size)
