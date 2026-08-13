"""Direct branch coverage for ``promote_adapter``'s create-once canonical gate.

``promote_adapter`` (scripts/modal_stopdff_v5_runner.py) copies a validated
adapter build into its canonical create-once destination.  Control-plane tests
replace this stage with a fake that only produces the fresh-path shape, so the
real body is exercised here under the fake-modal seam from
``tests.test_stopdff_v5_control_plane``, with the real checker validating a real
selftest-built adapter bundle.

Branch matrix:

1. source fails ``checker.validate_adapter``      -> ``ValueError``
2. recomputed adapter id != requested id          -> ``ValueError``
3. destination exists, valid, same id             -> ``{"cached": True}`` reuse
4. destination exists but invalid                 -> ``FileExistsError``
5. fresh staged copy + pre-publish revalidation   -> ``{"cached": False}``
   + atomic rename into the canonical slot
   (including the staged revalidation failure     -> ``ValueError``,
   crash-persisted stale staging reclaim, and the interrupted-copy retry)
"""

from __future__ import annotations

import json
import os
import shutil
import time
import types
from pathlib import Path

import pytest

from scripts.stopdff_v5 import checker, selftest
from tests.harness_control_plane import _load_modal_runner


@pytest.fixture(scope="module")
def valid_adapter_bundle(tmp_path_factory) -> Path:
    """Build one checker-valid adapter bundle, shared read-only across tests."""
    base = tmp_path_factory.mktemp("promotion_pkg")
    return selftest.build_valid_package(base)["adapter_bundle"]


def _stage_build(
    runner,
    tmp_path: Path,
    bundle: Path,
    subdir: str = "build_a",
) -> str:
    """Point the runner volume at ``tmp_path`` and stage one adapter build."""
    runner.MNT = str(tmp_path)
    adapters = tmp_path / "adapters"
    adapters.mkdir(exist_ok=True)
    shutil.copytree(bundle, adapters / subdir)
    manifest = json.loads(
        (adapters / subdir / "manifest.json").read_text(encoding="utf-8")
    )
    return manifest["id"]


def test_promote_adapter_fresh_then_create_once_reuse(
    tmp_path,
    monkeypatch,
    valid_adapter_bundle,
):
    """Branches 5 then 3: fresh promotion copies once, reuse never rewrites."""
    runner = _load_modal_runner(monkeypatch)
    adapter_id = _stage_build(runner, tmp_path, valid_adapter_bundle)

    first = runner.promote_adapter("build_a", adapter_id)

    assert first == {
        "canonical_subdir": f"canonical_{adapter_id}",
        "cached": False,
    }
    dst = tmp_path / "adapters" / f"canonical_{adapter_id}"
    copied = checker.validate_adapter(dst)
    assert copied.passed
    assert copied.recomputed["adapter_bundle_id"] == adapter_id

    canonical_manifest = dst / "manifest.json"
    stat_before = canonical_manifest.stat()

    second = runner.promote_adapter("build_a", adapter_id)

    assert second == {
        "canonical_subdir": f"canonical_{adapter_id}",
        "cached": True,
    }
    # Create-once: reuse must not recreate or rewrite the canonical bytes.
    stat_after = canonical_manifest.stat()
    assert stat_after.st_ino == stat_before.st_ino
    assert stat_after.st_mtime_ns == stat_before.st_mtime_ns

    # Real results satisfy the control plane's promotion-shape validator,
    # so fake<->real shape agreement is a checked contract, not coincidence.
    runner._validate_promotion_result(first, adapter_id)
    runner._validate_promotion_result(second, adapter_id)


def test_promote_adapter_rejects_id_mismatch(
    tmp_path,
    monkeypatch,
    valid_adapter_bundle,
):
    """Branch 2: a valid source whose recomputed id differs fails closed."""
    runner = _load_modal_runner(monkeypatch)
    _stage_build(runner, tmp_path, valid_adapter_bundle)
    wrong_id = "0" * 64

    with pytest.raises(ValueError, match="does not match source manifest"):
        runner.promote_adapter("build_a", wrong_id)

    assert not (tmp_path / "adapters" / f"canonical_{wrong_id}").exists()


def test_promote_adapter_rejects_invalid_existing_destination(
    tmp_path,
    monkeypatch,
    valid_adapter_bundle,
):
    """Branch 4: an existing-but-invalid canonical destination hard-fails."""
    runner = _load_modal_runner(monkeypatch)
    adapter_id = _stage_build(runner, tmp_path, valid_adapter_bundle)
    runner.promote_adapter("build_a", adapter_id)

    dst = tmp_path / "adapters" / f"canonical_{adapter_id}"
    calibration = dst / "calibration.json"
    tampered = calibration.read_text(encoding="utf-8") + "\n"
    calibration.write_text(tampered, encoding="utf-8")

    with pytest.raises(FileExistsError, match="exists but is invalid"):
        runner.promote_adapter("build_a", adapter_id)

    # Fail-closed: the poisoned destination is reported, never repaired.
    assert calibration.read_text(encoding="utf-8") == tampered


def test_promote_adapter_rejects_invalid_source(
    tmp_path,
    monkeypatch,
    valid_adapter_bundle,
):
    """Branch 1: a source failing adapter validation never reaches the copy."""
    runner = _load_modal_runner(monkeypatch)
    adapter_id = _stage_build(runner, tmp_path, valid_adapter_bundle)
    src_manifest = tmp_path / "adapters" / "build_a" / "manifest.json"
    src_manifest.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="promotion source failed validation"):
        runner.promote_adapter("build_a", adapter_id)

    assert not (tmp_path / "adapters" / f"canonical_{adapter_id}").exists()


def test_promote_adapter_rejects_copy_that_fails_revalidation(
    tmp_path,
    monkeypatch,
):
    """Branch 5's guard: the staged copy is revalidated before publication."""
    runner = _load_modal_runner(monkeypatch)
    runner.MNT = str(tmp_path)
    adapter_id = "a" * 64
    adapters = tmp_path / "adapters"
    src = adapters / "build_a"
    src.mkdir(parents=True)
    (src / "manifest.json").write_text("{}", encoding="utf-8")

    results = iter(
        [
            types.SimpleNamespace(
                passed=True,
                errors=[],
                recomputed={"adapter_bundle_id": adapter_id},
            ),
            types.SimpleNamespace(
                passed=False,
                errors=["post-copy corruption"],
                recomputed={},
            ),
        ]
    )
    validated: list[Path] = []

    def fake_validate(path):
        validated.append(Path(path))
        return next(results)

    monkeypatch.setattr(checker, "validate_adapter", fake_validate)

    with pytest.raises(ValueError, match="copied adapter failed validation"):
        runner.promote_adapter("build_a", adapter_id)

    # The copy is validated inside the private staging area, never at the
    # live canonical name.
    assert validated[0] == src
    staged_copy = validated[1]
    assert staged_copy.name == "bundle"
    assert staged_copy.parent.parent == adapters
    assert staged_copy.parent.name.startswith(".staging_")
    # Fail-closed and clean: no canonical slot, no staging residue.
    assert not (adapters / f"canonical_{adapter_id}").exists()
    assert list(adapters.glob(".staging_*")) == []


def test_promote_adapter_crash_persisted_partial_staging_cannot_brick_promotion(
    tmp_path,
    monkeypatch,
    valid_adapter_bundle,
):
    """A crash-persisted partial staging leftover is reclaimed once stale, a
    live peer's young staging dir survives, and the canonical slot is only
    ever observed complete — the permanent partial-canonical brick is gone."""
    runner = _load_modal_runner(monkeypatch)
    adapter_id = _stage_build(runner, tmp_path, valid_adapter_bundle)
    adapters = tmp_path / "adapters"

    # Crash-persisted partial copy from a previous hard-killed promotion:
    # lives under .staging_*, never at the canonical name.
    stale = adapters / ".staging_deadbeef"
    (stale / "bundle").mkdir(parents=True)
    (stale / "bundle" / "manifest.json").write_text(
        "partial bytes", encoding="utf-8"
    )
    old = time.time() - runner._STAGING_REAP_AGE_S - 3600
    os.utime(stale, (old, old))

    # A young staging directory (a live peer mid-copy) must not be reclaimed.
    young = adapters / ".staging_young"
    young.mkdir()

    result = runner.promote_adapter("build_a", adapter_id)

    assert result == {
        "canonical_subdir": f"canonical_{adapter_id}",
        "cached": False,
    }
    dst = adapters / f"canonical_{adapter_id}"
    copied = checker.validate_adapter(dst)
    assert copied.passed
    assert copied.recomputed["adapter_bundle_id"] == adapter_id
    assert not stale.exists()
    assert young.is_dir()
    # No residue from the successful promotion itself.
    assert [entry.name for entry in adapters.glob(".staging_*")] == [
        ".staging_young"
    ]


def test_promote_adapter_interrupted_copy_leaves_no_canonical_and_retry_succeeds(
    tmp_path,
    monkeypatch,
    valid_adapter_bundle,
):
    """An interrupted copy never creates the canonical slot (partial bytes land
    only under .staging_*), and the immediate retry promotes cleanly."""
    runner = _load_modal_runner(monkeypatch)
    adapter_id = _stage_build(runner, tmp_path, valid_adapter_bundle)
    adapters = tmp_path / "adapters"
    dst = adapters / f"canonical_{adapter_id}"

    real_copytree = shutil.copytree
    calls = {"count": 0}

    def interrupted_copytree(source, destination, **kwargs):
        calls["count"] += 1
        real_copytree(source, destination, **kwargs)
        if calls["count"] == 1:
            raise RuntimeError("interrupted copy")

    monkeypatch.setattr(shutil, "copytree", interrupted_copytree)

    with pytest.raises(RuntimeError, match="interrupted copy"):
        runner.promote_adapter("build_a", adapter_id)

    # The live canonical name was never created; the controlled failure also
    # cleaned up its own staging directory.
    assert not dst.exists()
    assert list(adapters.glob(".staging_*")) == []

    retry = runner.promote_adapter("build_a", adapter_id)

    assert retry == {
        "canonical_subdir": f"canonical_{adapter_id}",
        "cached": False,
    }
    assert checker.validate_adapter(dst).passed
    assert list(adapters.glob(".staging_*")) == []
