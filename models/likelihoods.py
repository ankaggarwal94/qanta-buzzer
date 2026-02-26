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
from typing import Any

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


class SBERTLikelihood(LikelihoodModel):
    """Sentence-BERT likelihood model using semantic embeddings.

    Uses a ``SentenceTransformer`` model to compute dense, L2-normalized
    embeddings. Cosine similarity is computed as a simple dot product since
    embeddings are pre-normalized (``normalize_embeddings=True``).

    Inherits ``embed_and_cache()`` from ``LikelihoodModel`` for transparent
    caching of embeddings via SHA-256 content hashing. The first call to
    ``score()`` computes and caches all embeddings; subsequent calls with the
    same texts are fast cache lookups.

    Compared to TF-IDF, SBERT captures semantic similarity (e.g., "first
    president" and "George Washington" score highly even without word overlap)
    but is slower due to the neural encoder.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier for ``SentenceTransformer``.
        Default is ``"all-MiniLM-L6-v2"`` (22M params, 384-dim embeddings).
        First run downloads the model (~80MB) from HuggingFace.

    Attributes
    ----------
    model_name : str
        The SentenceTransformer model name.
    encoder : SentenceTransformer
        The loaded sentence transformer model.

    Examples
    --------
    >>> model = SBERTLikelihood()  # downloads model on first run
    >>> scores = model.score("first president", ["Washington", "Lincoln"])
    >>> scores.shape
    (2,)
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        super().__init__()
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.encoder = SentenceTransformer(model_name)

    def _embed_batch(self, texts: list[str]) -> np.ndarray:
        """Embed texts using the SentenceTransformer encoder.

        Embeddings are L2-normalized so that cosine similarity can be computed
        as a simple dot product (avoiding the division by norms).

        Parameters
        ----------
        texts : list[str]
            Texts to embed (guaranteed non-empty, all cache misses).

        Returns
        -------
        np.ndarray
            Normalized embeddings of shape (len(texts), embed_dim), dtype float32.
        """
        return self.encoder.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True
        ).astype(np.float32)

    def score(self, clue_prefix: str, option_profiles: list[str]) -> np.ndarray:
        """Score each option using semantic cosine similarity.

        Computes dot product between the clue embedding and each option
        embedding. Since embeddings are L2-normalized, dot product equals
        cosine similarity.

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
            Values in [-1, 1].
        """
        clue_emb = self.embed_and_cache([clue_prefix])[0]
        option_embs = self.embed_and_cache(option_profiles)
        sims = option_embs @ clue_emb
        return sims.astype(np.float32)


class T5Likelihood(LikelihoodModel):
    """T5 encoder likelihood model using mean-pooled semantic embeddings.

    Uses ``T5EncoderModel`` (not full ``T5ForConditionalGeneration``) for 2x
    faster inference and half the memory. Embeddings are mean-pooled over
    sequence length with attention mask weighting to handle padding correctly.

    Inherits ``embed_and_cache()`` from ``LikelihoodModel`` for transparent
    caching of embeddings via SHA-256 content hashing. The first call to
    ``score()`` computes and caches all embeddings; subsequent calls with the
    same texts are fast cache lookups.

    Compared to SBERT, T5 captures deeper semantic relationships via its
    encoder-decoder pre-training on massive text corpora. This is the novel
    contribution: using T5 as a likelihood model rather than just as a policy
    encoder.

    Parameters
    ----------
    model_name : str
        HuggingFace T5 model identifier. Default is ``"t5-base"``
        (220M params). Options:

        - ``"t5-small"`` (60M params) -- fastest, lowest quality
        - ``"t5-base"`` (220M params) -- balanced (recommended)
        - ``"t5-large"`` (770M params) -- best quality, requires 8GB GPU VRAM

        First run downloads the model from HuggingFace (~850MB for t5-base).

    Attributes
    ----------
    model_name : str
        The T5 model identifier.
    encoder : T5EncoderModel
        Pre-trained T5 encoder loaded from HuggingFace.
    tokenizer : T5TokenizerFast
        Fast T5 tokenizer for text preprocessing.
    device : torch.device
        Computation device (cuda if available, else cpu).

    Examples
    --------
    >>> model = T5Likelihood(model_name="t5-small")
    >>> scores = model.score("first president", ["Washington", "Einstein"])
    >>> scores.shape
    (2,)
    """

    def __init__(self, model_name: str = "t5-base") -> None:
        super().__init__()
        import torch
        from transformers import T5EncoderModel, T5TokenizerFast

        self.model_name = model_name
        self.encoder = T5EncoderModel.from_pretrained(model_name)
        self.tokenizer = T5TokenizerFast.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder.to(self.device)
        self.encoder.eval()

    def _embed_batch(self, texts: list[str]) -> np.ndarray:
        """Embed texts using T5 encoder with attention-masked mean pooling.

        Mean pooling uses the attention mask to exclude padding tokens from the
        average, ensuring correct semantic embeddings when sequences have
        different lengths. Embeddings are L2-normalized so that cosine
        similarity can be computed as a simple dot product.

        Parameters
        ----------
        texts : list[str]
            Texts to embed (guaranteed non-empty, all cache misses).

        Returns
        -------
        np.ndarray
            L2-normalized embeddings of shape (len(texts), hidden_dim),
            dtype float32. Hidden dim is 512 (t5-small), 768 (t5-base),
            or 1024 (t5-large).

        Notes
        -----
        Tensors are detached and moved to CPU immediately after computation
        to prevent GPU memory leaks when called repeatedly during episodes.
        """
        import torch

        with torch.no_grad():
            encoded = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)

            outputs = self.encoder(**encoded)
            last_hidden = outputs.last_hidden_state  # (batch, seq_len, hidden_dim)

            # Mean pooling over sequence length with attention mask
            mask = encoded.attention_mask.unsqueeze(-1)  # (batch, seq_len, 1)
            masked_hidden = last_hidden * mask
            sum_hidden = masked_hidden.sum(dim=1)  # (batch, hidden_dim)
            mask_sum = mask.sum(dim=1).clamp(min=1e-9)  # (batch, 1)
            mean_pooled = sum_hidden / mask_sum  # (batch, hidden_dim)

            # L2 normalize for cosine similarity via dot product
            embeddings = torch.nn.functional.normalize(mean_pooled, p=2, dim=1)

            # Detach and move to CPU to prevent GPU memory leak
            embeddings = embeddings.detach().cpu().numpy().astype(np.float32)

        return embeddings

    def score(self, clue_prefix: str, option_profiles: list[str]) -> np.ndarray:
        """Score each option using T5 semantic cosine similarity.

        Computes dot product between the clue embedding and each option
        embedding. Since embeddings are L2-normalized, dot product equals
        cosine similarity.

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
            Values in [-1, 1].
        """
        clue_emb = self.embed_and_cache([clue_prefix])[0]
        option_embs = self.embed_and_cache(option_profiles)
        sims = option_embs @ clue_emb
        return sims.astype(np.float32)


def build_likelihood_from_config(
    config: dict[str, Any], corpus_texts: list[str] | None = None
) -> LikelihoodModel:
    """Construct a likelihood model from YAML configuration.

    Factory function that reads the ``likelihood`` section of the config dict
    and instantiates the appropriate ``LikelihoodModel`` subclass.

    Parameters
    ----------
    config : dict[str, Any]
        Full YAML config dict. Must contain a ``"likelihood"`` key with at
        least a ``"model"`` field specifying the model type.

        Supported model types:
        - ``"tfidf"``: TF-IDF cosine similarity (requires ``corpus_texts``)
        - ``"sbert"``: Sentence-BERT semantic similarity

        Optional config keys:
        - ``"sbert_name"`` or ``"embedding_model"``: SentenceTransformer model
          name (default: ``"all-MiniLM-L6-v2"``)

    corpus_texts : list[str] or None
        Text corpus for TF-IDF fitting. Required when ``model == "tfidf"``,
        ignored for other models.

    Returns
    -------
    LikelihoodModel
        An instantiated and ready-to-use likelihood model.

    Raises
    ------
    ValueError
        If ``model`` is ``"tfidf"`` and ``corpus_texts`` is None.
        If ``model`` is not a recognized model type.

    Examples
    --------
    >>> from qb_data.config import load_config
    >>> config = load_config("configs/default.yaml")
    >>> model = build_likelihood_from_config(config, corpus_texts=my_corpus)
    >>> scores = model.score("first president", ["Washington", "Lincoln"])
    """
    cfg = config["likelihood"]
    model_name = cfg.get("model", "sbert")

    if model_name == "tfidf":
        if not corpus_texts:
            raise ValueError("TF-IDF likelihood requires corpus_texts.")
        return TfIdfLikelihood(corpus_texts=corpus_texts)

    if model_name == "sbert":
        # Support both "sbert_name" (qb-rl convention) and
        # "embedding_model" (qanta-buzzer default.yaml convention)
        sbert_name = cfg.get("sbert_name", cfg.get("embedding_model", "all-MiniLM-L6-v2"))
        return SBERTLikelihood(model_name=sbert_name)

    raise ValueError(f"Unknown likelihood model: {model_name}")
