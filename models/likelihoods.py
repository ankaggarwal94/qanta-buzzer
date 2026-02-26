"""
Likelihood Model Interface

Abstract base class for likelihood models that score answer options against
revealed clue text. Concrete implementations (TF-IDF, SBERT, T5) inherit
from ``LikelihoodModel`` and implement ``score()`` and ``_embed_batch()``.

The ``score()`` method returns **raw similarity scores**, not probabilities.
The environment applies softmax with a configurable temperature (beta) to
convert scores into a belief distribution.

Embedding caching is built into the base class: texts are hashed via SHA-256
and cached as float32 numpy arrays, so repeated calls with the same text
skip recomputation.

Ported from qb-rl reference implementation (models/likelihoods.py lines 1-38).
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod

import numpy as np


def _text_key(text: str) -> str:
    """Compute a SHA-256 hash key for embedding cache lookups.

    Parameters
    ----------
    text : str
        Input text to hash.

    Returns
    -------
    str
        64-character hexadecimal SHA-256 digest.

    Examples
    --------
    >>> key = _text_key("hello world")
    >>> len(key)
    64
    >>> _text_key("hello world") == _text_key("hello world")
    True
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class LikelihoodModel(ABC):
    """Abstract base class for likelihood models.

    Likelihood models score how well each answer option matches a given
    clue prefix. The environment uses these scores (via softmax) to compute
    belief distributions over answer options.

    Subclasses must implement:
        - ``score(clue_prefix, option_profiles) -> np.ndarray``
        - ``_embed_batch(texts) -> np.ndarray``

    The base class provides ``embed_and_cache()`` which handles caching of
    text embeddings via SHA-256 content hashing.

    Attributes
    ----------
    embedding_cache : dict[str, np.ndarray]
        Maps SHA-256 text hashes to float32 embedding vectors.
    """

    def __init__(self) -> None:
        self.embedding_cache: dict[str, np.ndarray] = {}

    @abstractmethod
    def score(self, clue_prefix: str, option_profiles: list[str]) -> np.ndarray:
        """Return raw similarity scores for each answer option.

        The caller (environment) converts these to probabilities via
        softmax with a beta temperature parameter. Higher scores indicate
        stronger match between clue and option.

        Parameters
        ----------
        clue_prefix : str
            Clue text revealed so far (concatenation of clues up to current step).
        option_profiles : list[str]
            Answer profile text for each of the K answer options.

        Returns
        -------
        np.ndarray
            Raw similarity scores of shape (K,) where K = len(option_profiles).
        """

    def embed_and_cache(self, texts: list[str]) -> np.ndarray:
        """Embed texts, using cache for previously seen inputs.

        Texts are identified by their SHA-256 hash. Only unseen texts
        are passed to ``_embed_batch()`` for actual computation; cached
        results are reused.

        Parameters
        ----------
        texts : list[str]
            Texts to embed.

        Returns
        -------
        np.ndarray
            Stacked embeddings of shape (len(texts), embed_dim), dtype float32.
        """
        missing = [text for text in texts if _text_key(text) not in self.embedding_cache]
        if missing:
            new_embeddings = self._embed_batch(missing)
            for text, emb in zip(missing, new_embeddings):
                self.embedding_cache[_text_key(text)] = emb.astype(np.float32)
        return np.stack([self.embedding_cache[_text_key(text)] for text in texts])

    @abstractmethod
    def _embed_batch(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of texts. Subclasses must implement.

        Parameters
        ----------
        texts : list[str]
            Texts to embed (guaranteed non-empty, all cache misses).

        Returns
        -------
        np.ndarray
            Embeddings of shape (len(texts), embed_dim), dtype float32.
        """
        raise NotImplementedError


class TfIdfLikelihood(LikelihoodModel):
    """TF-IDF based likelihood model using cosine similarity.

    Uses scikit-learn's ``TfidfVectorizer`` to learn vocabulary and IDF weights
    from a corpus, then scores clue-option similarity via cosine distance in the
    TF-IDF vector space.

    The model **must** be ``fit()`` on a corpus before calling ``score()`` or
    ``_embed_batch()``. Calling these methods on an unfitted model raises
    ``RuntimeError``.

    This is the fast, interpretable baseline: keyword overlap drives similarity.
    It works well when clues contain distinctive vocabulary but misses semantic
    relationships (e.g., "first president" vs "George Washington").

    Parameters
    ----------
    corpus_texts : list[str] or None
        If provided, ``fit()`` is called immediately on these texts.

    Attributes
    ----------
    vectorizer : TfidfVectorizer
        Scikit-learn vectorizer with English stop words removed.
    _is_fit : bool
        Whether the vectorizer has been fit on a corpus.

    Examples
    --------
    >>> corpus = ["George Washington was the first president",
    ...           "Abraham Lincoln freed the slaves"]
    >>> model = TfIdfLikelihood(corpus_texts=corpus)
    >>> scores = model.score("first president", ["Washington", "Lincoln"])
    >>> scores.shape
    (2,)
    """

    def __init__(self, corpus_texts: list[str] | None = None) -> None:
        super().__init__()
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vectorizer = TfidfVectorizer(stop_words="english")
        self._is_fit = False
        if corpus_texts:
            self.fit(corpus_texts)

    def fit(self, corpus_texts: list[str]) -> "TfIdfLikelihood":
        """Learn vocabulary and IDF weights from a text corpus.

        Parameters
        ----------
        corpus_texts : list[str]
            Corpus of documents to learn from. Should include answer profiles,
            clue texts, or both to capture domain vocabulary.

        Returns
        -------
        TfIdfLikelihood
            Self, for method chaining.
        """
        self.vectorizer.fit(corpus_texts)
        self._is_fit = True
        return self

    def score(self, clue_prefix: str, option_profiles: list[str]) -> np.ndarray:
        """Score each option against the clue using TF-IDF cosine similarity.

        Transforms both the clue and options into TF-IDF space, then computes
        cosine similarity between the clue vector and each option vector.

        Parameters
        ----------
        clue_prefix : str
            Clue text revealed so far.
        option_profiles : list[str]
            Answer profile text for each of the K answer options.

        Returns
        -------
        np.ndarray
            Cosine similarity scores of shape (K,), dtype float32.
            Values in [-1, 1] but typically [0, 1] for TF-IDF.

        Raises
        ------
        RuntimeError
            If called before ``fit()``.
        """
        if not self._is_fit:
            raise RuntimeError("TfIdfLikelihood must be fit() before score().")
        clue_vec = self.vectorizer.transform([clue_prefix])
        option_vecs = self.vectorizer.transform(option_profiles)
        from sklearn.metrics.pairwise import cosine_similarity

        sims = cosine_similarity(clue_vec, option_vecs)[0]
        return sims.astype(np.float32)

    def _embed_batch(self, texts: list[str]) -> np.ndarray:
        """Embed texts as dense TF-IDF vectors.

        Parameters
        ----------
        texts : list[str]
            Texts to embed (guaranteed non-empty, all cache misses).

        Returns
        -------
        np.ndarray
            Dense TF-IDF matrix of shape (len(texts), vocab_size), dtype float32.

        Raises
        ------
        RuntimeError
            If called before ``fit()``.
        """
        if not self._is_fit:
            raise RuntimeError("TfIdfLikelihood must be fit() before embedding.")
        mat = self.vectorizer.transform(texts).toarray()
        return mat.astype(np.float32)
