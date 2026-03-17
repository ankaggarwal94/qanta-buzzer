"""Tests for scripts._common helpers."""

from __future__ import annotations

from scripts._common import embedding_cache_path


def test_embedding_cache_path_keys_by_model_variant() -> None:
    """Cache filenames should distinguish supported model variants."""
    assert (
        embedding_cache_path(
            {"likelihood": {"model": "sbert", "embedding_model": "all-MiniLM-L6-v2"}}
        ).name
        == "embedding_cache_all-MiniLM-L6-v2.npz"
    )
    assert (
        embedding_cache_path({"likelihood": {"model": "openai", "openai_model": "text-embedding-3-large"}}).name
        == "embedding_cache_text-embedding-3-large.npz"
    )
    assert (
        embedding_cache_path({"likelihood": {"model": "t5", "t5_name": "t5-large"}}).name
        == "embedding_cache_t5-large.npz"
    )
    assert embedding_cache_path({"likelihood": {"model": "t5-base"}}).name == "embedding_cache_t5-base.npz"
    assert embedding_cache_path({"likelihood": {"model": "tfidf"}}).name == "embedding_cache_tfidf.npz"
