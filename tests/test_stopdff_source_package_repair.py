"""Source diagnostics and conservative repair-copy regressions."""
from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest

from scripts.stopdff_v5.checker_package import _check_source_producer_map
from scripts.stopdff_v5.content_manifest import validate_bound_content_manifest
from scripts.stopdff_v5.identity import build_manifest, sha256_file
from scripts.stopdff_v5.manifests import source_manifest_identity


def package(tmp_path: Path) -> tuple[Path, dict]:
    root = tmp_path / "package"
    source = root / "evidence/source_snapshot/source"
    source.mkdir(parents=True)
    entries = []
    for name, payload, mode in (
        ("pyproject.toml", b"[project]\nname='fixture'\n", 0o644),
        ("uv.lock", b"version=1\n", 0o644),
        ("tool.py", b"print('hello')\n", 0o755),
    ):
        target = source / name
        target.write_bytes(payload)
        target.chmod(mode)
        entries.append({"path": name, "mode": "100755" if mode & 0o111 else "100644", "size": len(payload), "sha256": sha256_file(target)})
    hashes = {entry["path"]: entry["sha256"] for entry in entries}
    manifest = build_manifest(source_manifest_identity(git_sha="a" * 40, files=entries, pyproject_sha256=hashes["pyproject.toml"], uv_lock_sha256=hashes["uv.lock"]))
    (root / "evidence/source_manifest.json").write_text(json.dumps(manifest))
    (root / "other.json").write_text('{"preserve":"me"}\n')
    checksums(root)
    return root, manifest


def checksums(root: Path) -> None:
    (root / "SHA256SUMS").write_text("".join(f"{sha256_file(path)}  {path.relative_to(root).as_posix()}\n" for path in sorted(root.rglob("*")) if path.is_file() and path.name != "SHA256SUMS"))


def validate(root: Path, manifest: dict) -> dict:
    return validate_bound_content_manifest(root / "evidence", manifest_name="source_manifest.json", expected_id=manifest["id"], expected_kind="source_snapshot", file_key="files", name_key="path", content_subdir="source_snapshot/source")


@pytest.mark.parametrize("fault, marker", [("mode", "mode mismatch"), ("bytes", "sha256 mismatch"), ("size", "size mismatch"), ("missing", "missing file"), ("directory", "not a regular file"), ("symlink", "symlink")])
def test_precise_content_error(tmp_path: Path, fault: str, marker: str) -> None:
    root, manifest = package(tmp_path)
    path = root / "evidence/source_snapshot/source/tool.py"
    if fault == "mode":
        path.chmod(0o644)
    elif fault == "bytes":
        path.write_bytes(b"print('other')\n")
    elif fault == "size":
        path.write_bytes(b"changed")
    else:
        path.unlink()
        if fault == "directory":
            path.mkdir()
        elif fault == "symlink":
            path.symlink_to("uv.lock")
    with pytest.raises(ValueError, match=marker + ".*tool.py"):
        validate(root, manifest)


def test_unavailable_source_does_not_claim_producer_hash_mismatch() -> None:
    errors = []
    _check_source_producer_map(errors, source_hashes=None, claimed={"adapter_build.py": "a" * 64}, expected_basenames={"adapter_build.py"}, label="adapter producer_hashes")
    assert errors == []


def test_unavailable_source_still_rejects_malformed_producer_declarations() -> None:
    errors = []
    _check_source_producer_map(errors, source_hashes=None, claimed={"adapter_build.py": "not-a-hash"}, expected_basenames={"adapter_build.py"}, label="adapter producer_hashes")
    assert len(errors) == 1
    assert "canonical SHA256" in errors[0]


def test_available_source_still_rejects_wrong_producer_hash() -> None:
    errors = []
    _check_source_producer_map(errors, source_hashes={"scripts/stopdff_v5/adapter_build.py": "b" * 64}, claimed={"adapter_build.py": "a" * 64}, expected_basenames={"adapter_build.py"}, label="adapter producer_hashes")
    assert len(errors) == 1
    assert "does not match packaged source" in errors[0]


def test_mode_repair_only_changes_copy_and_writes_external_receipt(tmp_path: Path) -> None:
    from scripts.inspect_stopdff_source_package import inspect_package, repair_package
    root, manifest = package(tmp_path)
    tool = root / "evidence/source_snapshot/source/tool.py"
    project = root / "evidence/source_snapshot/source/pyproject.toml"
    tool.chmod(0o640)
    project.chmod(0o751)
    before = {str(p.relative_to(root)): (sha256_file(p), stat.S_IMODE(p.stat().st_mode)) for p in root.rglob("*") if p.is_file()}
    report = inspect_package(root, expected_source_id=manifest["id"])
    assert len(report["mode_mismatches"]) == 2
    assert report["source_files_verified"] == 3
    dest, receipt = tmp_path / "repaired", tmp_path / "receipt.json"
    repair_package(root, dest, receipt, expected_source_id=manifest["id"])
    validate(dest, manifest)
    assert before == {str(p.relative_to(root)): (sha256_file(p), stat.S_IMODE(p.stat().st_mode)) for p in root.rglob("*") if p.is_file()}
    assert {name: digest for name, (digest, _) in before.items()} == {str(p.relative_to(dest)): sha256_file(p) for p in dest.rglob("*") if p.is_file()}
    assert stat.S_IMODE((dest / tool.relative_to(root)).stat().st_mode) == 0o751
    assert stat.S_IMODE((dest / project.relative_to(root)).stat().st_mode) == 0o640
    saved = json.loads(receipt.read_text())
    assert saved["byte_changes"] == 0
    assert saved["source_files_changed"] == 2
    assert saved["source_manifest_id"] == manifest["id"]


@pytest.mark.parametrize("fault", ["bytes", "extra", "extra_empty_dir", "symlink", "wrong_id", "bad_checksums", "special", "manifest_symlink"])
def test_non_mode_failures_reject_before_copy(tmp_path: Path, fault: str) -> None:
    from scripts.inspect_stopdff_source_package import repair_package
    root, manifest = package(tmp_path)
    source = root / "evidence/source_snapshot/source"
    expected_id = manifest["id"]
    if fault == "bytes":
        (source / "tool.py").write_text("wrong")
        checksums(root)  # source manifest must independently reject changed bytes
    elif fault == "extra":
        (source / "extra").write_text("extra")
        checksums(root)
    elif fault == "extra_empty_dir":
        (source / "unlisted_dir").mkdir()
    elif fault == "symlink":
        (source / "tool.py").unlink()
        (source / "tool.py").symlink_to("uv.lock")
    elif fault == "manifest_symlink":
        path = root / "evidence/source_manifest.json"
        outside = tmp_path / "external.json"
        path.rename(outside)
        path.symlink_to(outside)
    elif fault == "wrong_id":
        expected_id = "0" * 64
    elif fault == "special":
        import os
        os.mkfifo(root / "fifo")
    else:
        (root / "other.json").write_text("tampered")
    dest = tmp_path / "repair"
    with pytest.raises((ValueError, OSError)):
        repair_package(root, dest, tmp_path / "receipt.json", expected_source_id=expected_id)
    assert not dest.exists()


@pytest.mark.parametrize("fault", ["existing_copy", "receipt_in_source", "receipt_in_copy", "existing_receipt", "copy_in_source", "copy_ancestor", "linked_parent"])
def test_unsafe_destinations_reject(tmp_path: Path, fault: str) -> None:
    from scripts.inspect_stopdff_source_package import repair_package
    root, manifest = package(tmp_path)
    dest, receipt = tmp_path / "copy", tmp_path / "receipt.json"
    if fault == "existing_copy":
        dest.mkdir()
    elif fault == "receipt_in_source":
        receipt = root / "receipt.json"
    elif fault == "receipt_in_copy":
        receipt = dest / "receipt.json"
    elif fault == "existing_receipt":
        receipt.write_text("existing")
    elif fault == "copy_in_source":
        dest = root / "copy"
    elif fault == "copy_ancestor":
        dest = tmp_path
    elif fault == "linked_parent":
        linked = tmp_path / "link"
        linked.symlink_to(tmp_path, target_is_directory=True)
        dest = linked / "copy"
    with pytest.raises((ValueError, OSError)):
        repair_package(root, dest, receipt, expected_source_id=manifest["id"])
    assert not (root / "receipt.json").exists()


def test_full_validator_mode_error_remains_fail_closed_without_hash_cascade(tmp_path: Path) -> None:
    from scripts.stopdff_v5 import checker, selftest
    built = selftest.build_valid_package(tmp_path / "fixture")
    root = built["run_root"]
    manifest = json.loads((root / "evidence/source_manifest.json").read_text())
    entry = manifest["identity"]["files"][0]
    target = root / "evidence/source_snapshot/source" / entry["path"]
    target.chmod(0o644 if entry["mode"] == "100755" else 0o755)
    result = checker.validate_run(root, backend="modal", adapter_bundle=built["adapter_bundle"], require_final_profile=False, require_package=True)
    assert not result.passed
    assert any("mode mismatch" in error and entry["path"] in error for error in result.errors)
    assert not any("does not match packaged source" in error for error in result.errors)


def test_cli_exit_codes_and_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from scripts.inspect_stopdff_source_package import main
    root, manifest = package(tmp_path)
    args = [str(root), "--expected-source-id", manifest["id"]]
    assert main(args) == 0
    assert json.loads(capsys.readouterr().out)["mode_mismatches"] == []
    (root / "evidence/source_snapshot/source/tool.py").chmod(0o644)
    assert main(args) == 2
    assert len(json.loads(capsys.readouterr().out)["mode_mismatches"]) == 1
    assert main(args + ["--repair-copy", str(tmp_path / "copy"), "--receipt", str(tmp_path / "receipt.json")]) == 0
    assert json.loads(capsys.readouterr().out)["byte_changes"] == 0
    (root / "other.json").write_text("wrong")
    assert main(args) == 1
    assert "checksum validation failed" in capsys.readouterr().err
