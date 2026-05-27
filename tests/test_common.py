"""Tests for scripts._common helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

import scripts._common as common
from scripts._common import (
    PROJECT_ROOT,
    embedding_cache_path,
    project_relative,
    resolve_default_dataset_path,
)


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


def test_resolve_default_dataset_path_raises_when_no_datasets_exist(
    tmp_path,
    monkeypatch,
) -> None:
    """When no dataset files exist at all, a clear FileNotFoundError is raised."""
    monkeypatch.setattr(common, "PROCESSED_DIR", tmp_path / "missing_processed")

    with pytest.raises(FileNotFoundError, match="build_mc_dataset"):
        resolve_default_dataset_path(tmp_path, preferred_split="test")


def test_project_relative_anchors_relative_inputs_to_project_root(
    tmp_path,
    monkeypatch,
) -> None:
    """Relative paths must resolve against PROJECT_ROOT, not CWD.

    Regression for the Copilot PR #14 review finding: automation that
    invokes pipeline scripts from outside the repo would pass a
    repo-relative argument like ``"data/processed/mc_dataset.json"``.
    The old implementation called ``Path(path).resolve()`` which
    resolves against CWD, leaking a machine-specific absolute path into
    artifact provenance JSON. The fix anchors non-absolute inputs to
    ``PROJECT_ROOT`` before resolving.
    """
    # Simulate running from an unrelated CWD (the common automation case).
    monkeypatch.chdir(tmp_path)

    result = project_relative("data/processed/mc_dataset.json")

    assert result == "data/processed/mc_dataset.json"
    # Sanity check: the offending old behavior would have produced an
    # absolute path under tmp_path; ensure we did NOT regress to that.
    assert not Path(result).is_absolute()
    assert str(tmp_path) not in result


def test_project_relative_preserves_repo_relative_for_absolute_inputs(
    tmp_path,
    monkeypatch,
) -> None:
    """Absolute paths inside PROJECT_ROOT should still come back repo-relative."""
    monkeypatch.chdir(tmp_path)
    abs_inside = PROJECT_ROOT / "data" / "processed" / "mc_dataset.json"

    assert project_relative(abs_inside) == "data/processed/mc_dataset.json"


def test_project_relative_falls_back_to_absolute_for_outside_repo(
    tmp_path,
) -> None:
    """Absolute paths outside the repo should fall back to the resolved absolute string."""
    outside = (tmp_path / "elsewhere.json").resolve()

    assert project_relative(outside) == str(outside)
