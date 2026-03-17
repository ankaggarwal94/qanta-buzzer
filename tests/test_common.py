"""Tests for scripts._common helpers."""

from __future__ import annotations

import scripts._common as common
from scripts._common import embedding_cache_path, resolve_default_dataset_path


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


def test_resolve_default_dataset_path_prefers_split_artifacts(tmp_path) -> None:
    """Split-specific artifacts should win over the combined legacy file."""
    (tmp_path / "train_dataset.json").write_text("[]", encoding="utf-8")
    (tmp_path / "val_dataset.json").write_text("[]", encoding="utf-8")
    (tmp_path / "test_dataset.json").write_text("[]", encoding="utf-8")
    (tmp_path / "mc_dataset.json").write_text("[]", encoding="utf-8")

    path, split, warning = resolve_default_dataset_path(tmp_path, preferred_split="val")

    assert path == tmp_path / "val_dataset.json"
    assert split == "val"
    assert warning is None


def test_resolve_default_dataset_path_warns_on_combined_fallback(
    tmp_path,
    monkeypatch,
) -> None:
    """When the preferred split is absent, the helper should flag in-sample fallback."""
    (tmp_path / "mc_dataset.json").write_text("[]", encoding="utf-8")
    monkeypatch.setattr(common, "PROCESSED_DIR", tmp_path / "missing_processed")

    path, split, warning = resolve_default_dataset_path(tmp_path, preferred_split="test")

    assert path == tmp_path / "mc_dataset.json"
    assert split == "combined"
    assert warning is not None
