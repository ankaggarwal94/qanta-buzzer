#!/usr/bin/env python3
"""Apply the centrally adjudicated PR #30 model-cache repair."""

from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"expected exactly one anchor in {path}, found {count}: {old[:100]!r}"
        )
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


def main() -> None:
    helper_anchor = '''\n\n@app.function(volumes={MNT: vol}, timeout=1800, max_containers=1)\ndef probe() -> dict:\n'''
    helper = '''\n\ndef _model_cache_state(root: Path) -> tuple[Path, bool]:\n    """Classify the immutable model cache without following unsafe paths."""\n    root = Path(root)\n    manifest_path = root / "model_snapshot_manifest.json"\n    if root.is_symlink():\n        raise ValueError("model cache root must be a non-symlink directory")\n    if root.exists() and not root.is_dir():\n        raise ValueError("model cache root must be a non-symlink directory")\n    if manifest_path.is_symlink():\n        raise ValueError(\n            "model cache manifest must be a non-symlink regular file"\n        )\n    if manifest_path.exists():\n        if not manifest_path.is_file():\n            raise ValueError(\n                "model cache manifest must be a non-symlink regular file"\n            )\n        return manifest_path, True\n    if root.exists():\n        try:\n            nonempty = next(root.iterdir(), None) is not None\n        except OSError as exc:\n            raise ValueError("model cache root cannot be inspected") from exc\n        if nonempty:\n            raise FileExistsError(\n                "model cache destination is incomplete or noncanonical"\n            )\n    return manifest_path, False\n''' + helper_anchor
    replace_once(
        "scripts/modal_stopdff_v5_runner.py",
        helper_anchor,
        helper,
    )

    freeze_anchor = '''    root = Path(_p("inputs", "model"))\n    mpath = root / "model_snapshot_manifest.json"\n    if mpath.exists():\n        cached = json.loads(mpath.read_text())\n        _verified_content_manifest(\n            root,\n            manifest_name="model_snapshot_manifest.json",\n            expected_id=cached["id"],\n            file_key="files",\n            name_key="path",\n            content_subdir="snapshot",\n            expected_kind="model_snapshot",\n        )\n        return {"model_id": cached["id"], "cached": True}\n    root.mkdir(parents=True, exist_ok=True)\n    man = adapter_build.freeze_model_snapshot(root)\n'''
    freeze_replacement = '''    root = Path(_p("inputs", "model"))\n    mpath, cached_entry = _model_cache_state(root)\n    if cached_entry:\n        cached = json.loads(mpath.read_text())\n        _verified_content_manifest(\n            root,\n            manifest_name="model_snapshot_manifest.json",\n            expected_id=cached["id"],\n            file_key="files",\n            name_key="path",\n            content_subdir="snapshot",\n            expected_kind="model_snapshot",\n        )\n        return {"model_id": cached["id"], "cached": True}\n    root.mkdir(parents=True, exist_ok=True)\n    _, appeared_after_mkdir = _model_cache_state(root)\n    if appeared_after_mkdir:\n        raise FileExistsError("model cache appeared during fresh creation")\n    man = adapter_build.freeze_model_snapshot(root)\n'''
    replace_once(
        "scripts/modal_stopdff_v5_runner.py",
        freeze_anchor,
        freeze_replacement,
    )

    test_anchor = '''\n\ndef test_freeze_model_postfreeze_recheck_requires_model_snapshot_kind(\n'''
    tests = '''\n\n@pytest.mark.parametrize(\n    "mutation",\n    [\n        "live_root_symlink",\n        "dangling_root_symlink",\n        "root_file",\n        "manifest_symlink",\n        "snapshot_symlink",\n    ],\n)\ndef test_freeze_model_rejects_noncanonical_fresh_cache_before_write(\n    tmp_path,\n    monkeypatch,\n    mutation,\n):\n    """A poisoned fresh model cache is rejected before snapshot writes."""\n    runner = _load_modal_runner(monkeypatch)\n    runner.MNT = str(tmp_path)\n    inputs = tmp_path / "inputs"\n    inputs.mkdir()\n    root = inputs / "model"\n    external = tmp_path / "external-model"\n    external.mkdir()\n    external_manifest = tmp_path / "external-manifest.json"\n    external_manifest.write_text("outside", encoding="utf-8")\n\n    if mutation == "live_root_symlink":\n        root.symlink_to(external, target_is_directory=True)\n    elif mutation == "dangling_root_symlink":\n        root.symlink_to(tmp_path / "missing-model", target_is_directory=True)\n    elif mutation == "root_file":\n        root.write_text("not a directory", encoding="utf-8")\n    elif mutation == "manifest_symlink":\n        root.mkdir()\n        (root / "model_snapshot_manifest.json").symlink_to(external_manifest)\n    else:\n        root.mkdir()\n        (root / "snapshot").symlink_to(external, target_is_directory=True)\n\n    from scripts.stopdff_v5 import adapter_build\n\n    monkeypatch.setattr(\n        adapter_build,\n        "freeze_model_snapshot",\n        lambda _root: pytest.fail("snapshot writer reached poisoned cache"),\n    )\n\n    with pytest.raises((ValueError, FileExistsError), match="model cache"):\n        runner.freeze_model()\n\n    assert list(external.iterdir()) == []\n    assert external_manifest.read_text(encoding="utf-8") == "outside"\n\n\ndef test_freeze_model_accepts_empty_canonical_fresh_cache(\n    tmp_path,\n    monkeypatch,\n):\n    """An empty real cache directory remains a valid fresh destination."""\n    runner = _load_modal_runner(monkeypatch)\n    runner.MNT = str(tmp_path)\n    root = tmp_path / "inputs" / "model"\n    root.mkdir(parents=True)\n    model_id = "3" * 64\n    from scripts.stopdff_v5 import adapter_build\n\n    def freeze_valid_model(destination):\n        manifest = _write_model_manifest(\n            Path(destination),\n            kind="model_snapshot",\n            model_id=model_id,\n        )\n        manifest["identity"]["model_revision"] = "a" * 40\n        (Path(destination) / "model_snapshot_manifest.json").write_text(\n            json.dumps(manifest),\n            encoding="utf-8",\n        )\n        return manifest\n\n    monkeypatch.setattr(\n        adapter_build,\n        "freeze_model_snapshot",\n        freeze_valid_model,\n    )\n    monkeypatch.setattr(\n        runner,\n        "_verified_content_manifest",\n        lambda *_args, **_kwargs: {"id": model_id},\n    )\n\n    assert runner.freeze_model() == {\n        "model_id": model_id,\n        "revision": "a" * 40,\n        "cached": False,\n    }\n''' + test_anchor
    replace_once(
        "tests/test_pr30_control_repairs.py",
        test_anchor,
        tests,
    )


if __name__ == "__main__":
    main()
