"""Production release finalizer regressions.

The test fixtures author the authority documents; the finalizer may only copy
and verify their exact bytes.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from reproducibility.colm_aims_2026 import phase4_finalize_release as finalizer
from reproducibility.colm_aims_2026 import schema, verifier
from reproducibility.colm_aims_2026.phase4_finalize_release import (
    finalize_release,
)
from scripts.stopdff_v5 import fileio
from tests._colm_aims_v2_helpers import build_runs_site


def _roots(tmp_path: Path):
    site = build_runs_site(tmp_path)
    output_root = tmp_path / "published-authority"
    receipts_dir = tmp_path / "release-receipts"
    output_root.mkdir()
    receipts_dir.mkdir()
    return site, output_root, receipts_dir


def _finalize(site, output_root: Path, receipts_dir: Path, **overrides):
    values = {
        "runs_root": site.runs_root,
        "run_id": "run-0001",
        "ledger_input": site.ledger_path,
        "rights_input": site.rights_path,
        "expectations_input": site.expectations_path,
        "output_root": output_root,
        "release_id": "release-0001",
        "receipts_dir": receipts_dir,
    }
    values.update(overrides)
    return finalize_release(**values)


def _commit_without_parent_sync(
    anchor, staged_name: str, destination_name: str, **kwargs
):
    """Commit through the platform's anchored branch, then skip its sync."""
    if anchor._fd is None:
        source_anchor = kwargs["source_anchor"]
        anchor._rename_windows_child_handle(source_anchor, destination_name)
    else:
        os.mkdir(destination_name, dir_fd=anchor._fd)
        os.rename(
            staged_name,
            destination_name,
            src_dir_fd=anchor._fd,
            dst_dir_fd=anchor._fd,
        )
    return anchor._path(destination_name)


def _publish_test_directory(
    staged: Path,
    destination: Path,
    *,
    parent_chain,
    exists_label: str,
    expected_names: tuple[str, ...] = ("payload.json",),
):
    """Publish caller-captured fixture bytes through the production boundary."""
    staged_chain = finalizer._capture_directory_chain(staged)
    expected_snapshot = {
        name: (staged / name).read_bytes() for name in expected_names
    }
    return finalizer._publish_verified_directory(
        staged,
        destination,
        exists_label=exists_label,
        parent_chain=parent_chain,
        staged_chain=staged_chain,
        expected_snapshot=expected_snapshot,
        expected_names=expected_names,
    )


def test_exact_external_bytes_are_published_and_release_passes(tmp_path):
    site, output_root, receipts_dir = _roots(tmp_path)
    expected = {
        "ledger.json": site.ledger_path.read_bytes(),
        "rights.json": site.rights_path.read_bytes(),
        "expectations.json": site.expectations_path.read_bytes(),
    }

    result = _finalize(site, output_root, receipts_dir)

    assert result.report.verdict == verifier.VERDICT_RELEASE_PASS
    assert result.published_dir == output_root / "release-0001"
    assert {
        path.name: path.read_bytes() for path in result.published_dir.iterdir()
    } == expected
    # The sole receipt binds the exact staged tree and exact supplied
    # expectations before terminal atomic publication.
    assert len(list(receipts_dir.glob("receipt-*.json"))) == 1
    receipt = json.loads(result.report.receipt_path.read_text("utf-8"))
    snapshot = verifier._read_tree_snapshot(site.run_tree)
    assert receipt["input_tree_sha256"] == verifier._tree_digest_from_shas(
        {
            rel: hashlib.sha256(data).hexdigest()
            for rel, data in snapshot.items()
        }
    )
    assert receipt["expectations_anchor_sha256"] == hashlib.sha256(
        expected["expectations.json"]
    ).hexdigest()
    assert not finalizer._pending_guard_path(result.published_dir).exists()
    assert finalizer._accepted_marker_path(result.published_dir).is_file()
    assert finalizer._require_accepted_directory(
        result.published_dir, "release bundle"
    ) == result.published_dir


def test_release_staging_never_uses_lexical_path_write(
    tmp_path, monkeypatch
):
    site, output_root, receipts_dir = _roots(tmp_path)
    original = Path.write_bytes

    def reject_lexical_stage_write(path, data):
        if (
            path.parent.parent == output_root
            and path.parent.name.startswith(".release-staged-")
        ):
            raise AssertionError("release staging write escaped its anchor")
        return original(path, data)

    monkeypatch.setattr(Path, "write_bytes", reject_lexical_stage_write)

    assert _finalize(site, output_root, receipts_dir).published_dir.is_dir()


def test_staging_materialization_rejects_replaced_generation_before_write(
    tmp_path,
):
    output_root = tmp_path / "output"
    output_root.mkdir()
    staged = output_root / ".staged"
    staged.mkdir()
    parent_snapshot = finalizer._capture_directory_chain(output_root)
    staged_snapshot = finalizer._capture_directory_chain(staged)
    displaced = output_root / ".captured-staged"
    staged.rename(displaced)
    staged.mkdir()
    replacement = {
        name: f"replacement {name}\n".encode("utf-8")
        for name in finalizer._SIDECAR_NAMES
    }
    for name, data in replacement.items():
        (staged / name).write_bytes(data)
    expected = {
        name: f"owned {name}\n".encode("utf-8")
        for name in finalizer._SIDECAR_NAMES
    }

    with pytest.raises(schema.TypedIngressError):
        finalizer._materialize_staged_directory(
            staged,
            parent_snapshot,
            staged_snapshot,
            expected,
            finalizer._SIDECAR_NAMES,
            label="test staging directory",
        )

    assert list(displaced.iterdir()) == []
    assert {item.name: item.read_bytes() for item in staged.iterdir()} == replacement


def test_staging_materialization_never_truncates_late_file_claim(tmp_path):
    output_root = tmp_path / "output"
    output_root.mkdir()
    staged = output_root / ".staged"
    staged.mkdir()
    parent_snapshot = finalizer._capture_directory_chain(output_root)
    staged_snapshot = finalizer._capture_directory_chain(staged)
    claimed = staged / finalizer._SIDECAR_NAMES[0]
    claimed.write_bytes(b"incumbent\n")
    expected = {
        name: f"owned {name}\n".encode("utf-8")
        for name in finalizer._SIDECAR_NAMES
    }

    with pytest.raises(FileExistsError):
        finalizer._materialize_staged_directory(
            staged,
            parent_snapshot,
            staged_snapshot,
            expected,
            finalizer._SIDECAR_NAMES,
            label="test staging directory",
        )

    assert claimed.read_bytes() == b"incumbent\n"
    assert {item.name for item in staged.iterdir()} == {claimed.name}


def test_explicit_run_id_mismatch_refuses_before_publication(tmp_path):
    site, output_root, receipts_dir = _roots(tmp_path)

    with pytest.raises(schema.ConfigSurfaceError, match="canonical_run_id"):
        _finalize(
            site,
            output_root,
            receipts_dir,
            run_id="different-run",
        )

    assert not (output_root / "release-0001").exists()
    assert list(receipts_dir.iterdir()) == []


def test_unbound_rights_bytes_refuse_without_consuming_release_id(tmp_path):
    site, output_root, receipts_dir = _roots(tmp_path)
    expectations = json.loads(site.expectations_path.read_text("utf-8"))
    expectations["rights_inventory"]["sha256"] = "f" * 64
    bad = site.root / "bad-expectations.json"
    bad.write_bytes(schema.encode_json(expectations))

    with pytest.raises(schema.ConfigSurfaceError, match="supplied rights"):
        _finalize(
            site,
            output_root,
            receipts_dir,
            expectations_input=bad,
        )

    assert not (output_root / "release-0001").exists()


def test_destination_inside_runs_root_is_refused(tmp_path):
    site = build_runs_site(tmp_path)
    receipts_dir = tmp_path / "release-receipts"
    receipts_dir.mkdir()

    with pytest.raises(schema.ConfigSurfaceError, match="outside the runs root"):
        _finalize(site, site.runs_root, receipts_dir)

    assert not (site.runs_root / "release-0001").exists()


def test_existing_release_slot_is_never_overwritten(tmp_path):
    site, output_root, receipts_dir = _roots(tmp_path)
    occupied = output_root / "release-0001"
    occupied.mkdir()
    sentinel = occupied / "sentinel.txt"
    sentinel.write_text("keep", encoding="utf-8")

    with pytest.raises(schema.ConfigSurfaceError, match="already exists"):
        _finalize(site, output_root, receipts_dir)

    assert sentinel.read_text("utf-8") == "keep"
    assert {path.name for path in occupied.iterdir()} == {"sentinel.txt"}


def test_external_input_symlink_is_rejected_without_descent(tmp_path):
    site, output_root, receipts_dir = _roots(tmp_path)
    link = site.root / "ledger-link.json"
    try:
        link.symlink_to(site.ledger_path)
    except OSError as exc:  # pragma: no cover - Windows privilege-dependent
        pytest.skip(f"symlink creation unavailable: {exc}")

    with pytest.raises(schema.TypedIngressError, match="symlink|reparse"):
        _finalize(
            site,
            output_root,
            receipts_dir,
            ledger_input=link,
        )

    assert not (output_root / "release-0001").exists()


def test_supplied_hashes_remain_external_not_reauthored(tmp_path):
    site, output_root, receipts_dir = _roots(tmp_path)
    result = _finalize(site, output_root, receipts_dir)
    ledger_bytes = result.ledger_path.read_bytes()
    expectations = json.loads(result.expectations_path.read_text("utf-8"))

    assert expectations["anchor"]["ledger_sha256"] == hashlib.sha256(
        ledger_bytes
    ).hexdigest()
    assert result.expectations_path.read_bytes() == site.expectations_path.read_bytes()


@pytest.mark.parametrize(
    "release_id",
    (
        "",
        ".",
        "..",
        "nested/name",
        r"nested\name",
        "stream:ads",
        "trailing.",
        "NUL",
        "con.txt",
        "COM1",
        "lpt9.log",
        "a" * 65,
    ),
)
def test_release_id_must_be_bounded_and_cross_platform_portable(
    tmp_path, release_id
):
    site, output_root, receipts_dir = _roots(tmp_path)

    with pytest.raises(schema.ConfigSurfaceError, match="portable"):
        _finalize(
            site,
            output_root,
            receipts_dir,
            release_id=release_id,
        )

    assert list(output_root.iterdir()) == []
    assert list(receipts_dir.iterdir()) == []


def test_run_id_uses_the_same_portable_policy(tmp_path):
    site, output_root, receipts_dir = _roots(tmp_path)

    with pytest.raises(schema.ConfigSurfaceError, match="portable"):
        _finalize(
            site,
            output_root,
            receipts_dir,
            run_id="run:alternate-stream",
        )

    assert list(output_root.iterdir()) == []
    assert list(receipts_dir.iterdir()) == []


def test_finalizer_will_not_mutate_authority_input_base(tmp_path):
    site, output_root, receipts_dir = _roots(tmp_path)

    with pytest.raises(schema.ConfigSurfaceError, match="authority input base"):
        _finalize(site, output_root, site.receipts_dir)
    with pytest.raises(schema.ConfigSurfaceError, match="authority input base"):
        _finalize(site, site.root, receipts_dir)

    assert not (output_root / "release-0001").exists()
    assert not (site.root / "release-0001").exists()


@pytest.mark.parametrize(
    "field",
    (
        "input_tree_sha256",
        "expectations_anchor_sha256",
        "verifier_code_sha256",
    ),
)
def test_verifier_receipt_must_bind_exact_finalization_inputs(
    tmp_path, monkeypatch, field
):
    site, output_root, receipts_dir = _roots(tmp_path)
    original = verifier.run_release_over_runs_root

    def corrupt_receipt(*args, **kwargs):
        report = original(*args, **kwargs)
        receipt = json.loads(report.receipt_path.read_text("utf-8"))
        receipt[field] = "0" * 64
        report.receipt_path.write_bytes(schema.encode_json(receipt))
        return report

    monkeypatch.setattr(
        verifier, "run_release_over_runs_root", corrupt_receipt
    )
    with pytest.raises(schema.TypedIngressError, match=field):
        _finalize(site, output_root, receipts_dir)

    assert not (output_root / "release-0001").exists()


def test_selected_tree_mutation_during_verification_refuses_publication(
    tmp_path, monkeypatch
):
    site, output_root, receipts_dir = _roots(tmp_path)
    original = verifier.run_release_over_runs_root

    def mutate_after_verification(*args, **kwargs):
        report = original(*args, **kwargs)
        profile = site.run_tree / "profile.json"
        profile.write_bytes(profile.read_bytes() + b" ")
        return report

    monkeypatch.setattr(
        verifier, "run_release_over_runs_root", mutate_after_verification
    )
    with pytest.raises(schema.TypedIngressError, match="tree changed"):
        _finalize(site, output_root, receipts_dir)

    assert not (output_root / "release-0001").exists()


def test_staged_sidecar_readback_failure_precedes_publication(
    tmp_path, monkeypatch
):
    site, output_root, receipts_dir = _roots(tmp_path)

    def refuse_readback(_directory):
        raise schema.TypedIngressError("injected staged readback failure")

    monkeypatch.setattr(finalizer, "_read_bundle", refuse_readback)
    with pytest.raises(schema.TypedIngressError, match="injected staged"):
        _finalize(site, output_root, receipts_dir)

    assert not (output_root / "release-0001").exists()
    assert list(output_root.iterdir()) == []


def test_failed_cleanup_never_follows_replaced_output_root(
    tmp_path, monkeypatch
):
    site, output_root, receipts_dir = _roots(tmp_path)
    displaced = tmp_path / "displaced-authority-output"
    replacement_bytes = {
        name: f"replacement {name}\n".encode("utf-8")
        for name in finalizer._SIDECAR_NAMES
    }

    def replace_parent_then_refuse(directory):
        stage_name = Path(directory).name
        output_root.rename(displaced)
        output_root.mkdir()
        replacement_stage = output_root / stage_name
        replacement_stage.mkdir()
        for name, data in replacement_bytes.items():
            (replacement_stage / name).write_bytes(data)
        raise schema.TypedIngressError("injected staged readback failure")

    monkeypatch.setattr(finalizer, "_read_bundle", replace_parent_then_refuse)
    with pytest.raises(schema.TypedIngressError, match="injected staged"):
        _finalize(site, output_root, receipts_dir)

    replacement_stage = next(output_root.iterdir())
    assert {
        item.name: item.read_bytes() for item in replacement_stage.iterdir()
    } == replacement_bytes
    assert (displaced / replacement_stage.name).is_dir()


def test_exact_cleanup_refuses_contaminated_staging_tree(tmp_path):
    output_root = tmp_path / "output"
    output_root.mkdir()
    staged = output_root / ".staged"
    staged.mkdir()
    for name in finalizer._SIDECAR_NAMES:
        (staged / name).write_bytes(b"owned\n")
    (staged / "unexpected.txt").write_bytes(b"do not remove\n")
    snapshot = finalizer._capture_directory_chain(output_root)

    removed = finalizer._remove_exact_staged_directory(
        parent=output_root,
        parent_snapshot=snapshot,
        staged_name=staged.name,
        staged_snapshot=finalizer._capture_directory_chain(staged),
        expected_names=finalizer._SIDECAR_NAMES,
    )

    assert removed is False
    assert {item.name for item in staged.iterdir()} == {
        *finalizer._SIDECAR_NAMES,
        "unexpected.txt",
    }


def test_failed_cleanup_never_removes_replacement_staging_generation(
    tmp_path, monkeypatch
):
    site, output_root, receipts_dir = _roots(tmp_path)
    original_stage = output_root / "original-stage"
    replacement_bytes = {
        name: f"replacement {name}\n".encode("utf-8")
        for name in finalizer._SIDECAR_NAMES
    }

    def replace_stage_then_refuse(directory):
        staged = Path(directory)
        staged.rename(original_stage)
        staged.mkdir()
        for name, data in replacement_bytes.items():
            (staged / name).write_bytes(data)
        raise schema.TypedIngressError("injected staged readback failure")

    monkeypatch.setattr(finalizer, "_read_bundle", replace_stage_then_refuse)
    with pytest.raises(schema.TypedIngressError, match="injected staged"):
        _finalize(site, output_root, receipts_dir)

    replacement_stage = next(
        path for path in output_root.iterdir() if path != original_stage
    )
    assert {
        item.name: item.read_bytes() for item in replacement_stage.iterdir()
    } == replacement_bytes
    assert original_stage.is_dir()


def test_finalizer_publication_rejects_byte_identical_replacement_stage(
    tmp_path, monkeypatch
):
    site, output_root, receipts_dir = _roots(tmp_path)
    original_publish = finalizer._publish_verified_directory
    verified_stage = output_root / "verified-stage"
    observed_stage = None

    def replace_before_publish(staged, destination, **kwargs):
        nonlocal observed_stage
        staged = Path(staged)
        observed_stage = staged
        verified_bytes = {
            item.name: item.read_bytes() for item in staged.iterdir()
        }
        staged.rename(verified_stage)
        staged.mkdir()
        for name, data in verified_bytes.items():
            (staged / name).write_bytes(data)
        return original_publish(staged, destination, **kwargs)

    monkeypatch.setattr(
        finalizer, "_publish_verified_directory", replace_before_publish
    )
    with pytest.raises(schema.TypedIngressError, match="identity|captured"):
        _finalize(site, output_root, receipts_dir)

    assert observed_stage is not None and observed_stage.is_dir()
    assert verified_stage.is_dir()
    destination = output_root / "release-0001"
    assert not destination.exists()
    assert not finalizer._pending_guard_path(destination).exists()
    assert not finalizer._accepted_marker_path(destination).exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows handle deletion contract")
def test_windows_cleanup_deletes_exact_handle_without_name_rmdir(
    tmp_path, monkeypatch
):
    output_root = tmp_path / "output"
    output_root.mkdir()
    staged = output_root / ".staged"
    staged.mkdir()
    for name in finalizer._SIDECAR_NAMES:
        (staged / name).write_bytes(b"owned\n")
    parent_snapshot = finalizer._capture_directory_chain(output_root)
    staged_snapshot = finalizer._capture_directory_chain(staged)

    def reject_name_rmdir(*_args, **_kwargs):
        raise AssertionError("cleanup must delete the verified Windows handle")

    monkeypatch.setattr(finalizer.os, "rmdir", reject_name_rmdir)
    assert finalizer._remove_exact_staged_directory(
        parent=output_root,
        parent_snapshot=parent_snapshot,
        staged_name=staged.name,
        staged_snapshot=staged_snapshot,
        expected_names=finalizer._SIDECAR_NAMES,
    )
    assert not staged.exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows handle-share contract")
def test_windows_published_generation_is_locked_through_guard_retirement(
    tmp_path, monkeypatch
):
    site, output_root, receipts_dir = _roots(tmp_path)
    destination = output_root / "release-0001"
    displaced = output_root / "displaced-release"
    original_marker = finalizer._create_accepted_marker
    original_retire = finalizer._retire_pending_guard
    blocked_at: list[str] = []

    def marker_while_locked(path, tree_sha256, **kwargs):
        with pytest.raises(OSError):
            destination.rename(displaced)
        blocked_at.append("marker")
        return original_marker(path, tree_sha256, **kwargs)

    def retire_while_locked(guard, encoded, **kwargs):
        with pytest.raises(OSError):
            destination.rename(displaced)
        blocked_at.append("retire")
        return original_retire(guard, encoded, **kwargs)

    monkeypatch.setattr(
        finalizer, "_create_accepted_marker", marker_while_locked
    )
    monkeypatch.setattr(
        finalizer, "_retire_pending_guard", retire_while_locked
    )
    result = _finalize(site, output_root, receipts_dir)

    assert result.published_dir == destination
    assert blocked_at == ["marker", "retire"]
    destination.rename(displaced)
    displaced.rename(destination)


def test_partial_cleanup_syncs_and_leaves_bounded_orphan(
    tmp_path, monkeypatch
):
    output_root = tmp_path / "output"
    output_root.mkdir()
    staged = output_root / ".staged"
    staged.mkdir()
    for name in finalizer._SIDECAR_NAMES:
        (staged / name).write_bytes(b"owned\n")
    parent_snapshot = finalizer._capture_directory_chain(output_root)
    staged_snapshot = finalizer._capture_directory_chain(staged)
    cleanup_unlinks = 0
    cleanup_syncs = 0

    if os.name == "posix":
        original_unlink = finalizer.os.unlink
        original_sync = finalizer.os.fsync

        def fail_second_unlink(name, *args, **kwargs):
            nonlocal cleanup_unlinks
            if kwargs.get("dir_fd") is not None:
                cleanup_unlinks += 1
                if cleanup_unlinks == 2:
                    raise OSError("injected cleanup unlink failure")
            return original_unlink(name, *args, **kwargs)

        def track_sync(fd):
            nonlocal cleanup_syncs
            cleanup_syncs += 1
            return original_sync(fd)

        monkeypatch.setattr(finalizer.os, "unlink", fail_second_unlink)
        monkeypatch.setattr(finalizer.os, "fsync", track_sync)
    else:
        original_unlink = finalizer._DirectoryAnchor.unlink
        original_sync = finalizer._DirectoryAnchor.sync

        def fail_second_unlink(anchor, name):
            nonlocal cleanup_unlinks
            if anchor.label == "staging cleanup directory":
                cleanup_unlinks += 1
                if cleanup_unlinks == 2:
                    raise OSError("injected cleanup unlink failure")
            return original_unlink(anchor, name)

        def track_sync(anchor):
            nonlocal cleanup_syncs
            if anchor.label == "staging cleanup directory":
                cleanup_syncs += 1
            return original_sync(anchor)

        monkeypatch.setattr(
            finalizer._DirectoryAnchor, "unlink", fail_second_unlink
        )
        monkeypatch.setattr(finalizer._DirectoryAnchor, "sync", track_sync)
    assert not finalizer._remove_exact_staged_directory(
        parent=output_root,
        parent_snapshot=parent_snapshot,
        staged_name=staged.name,
        staged_snapshot=staged_snapshot,
        expected_names=finalizer._SIDECAR_NAMES,
    )
    assert cleanup_syncs == 1
    assert staged.is_dir()
    assert len(list(staged.iterdir())) == 2


def test_staged_sidecar_fsync_failure_precedes_publication(
    tmp_path, monkeypatch
):
    site, output_root, receipts_dir = _roots(tmp_path)

    def refuse_fsync(_directory):
        raise OSError("injected staged fsync failure")

    monkeypatch.setattr(fileio, "fsync_tree", refuse_fsync)
    with pytest.raises(OSError, match="injected staged fsync"):
        _finalize(site, output_root, receipts_dir)

    assert not (output_root / "release-0001").exists()


def test_finalizer_committed_publish_requires_second_durability_barrier(
    tmp_path, monkeypatch
):
    site, output_root, receipts_dir = _roots(tmp_path)
    destination = output_root / "release-0001"
    original_sync_tree = finalizer._DirectoryAnchor.sync_directory
    original_sync_parent = finalizer._DirectoryAnchor.sync
    synced: list[Path] = []

    def commit_then_fail(anchor, staged_name, destination_name, **_kwargs):
        dest = _commit_without_parent_sync(
            anchor, staged_name, destination_name, **_kwargs
        )
        raise fileio.DirectoryPublicationCommittedError(
            dest, OSError("injected parent sync failure")
        )

    def record_tree(anchor, name, expected_names, **kwargs):
        synced.append(anchor._path(name))
        return original_sync_tree(anchor, name, expected_names, **kwargs)

    def record_directory(anchor):
        synced.append(anchor.path)
        return original_sync_parent(anchor)

    monkeypatch.setattr(
        finalizer._DirectoryAnchor, "publish_directory", commit_then_fail
    )
    monkeypatch.setattr(
        finalizer._DirectoryAnchor, "sync_directory", record_tree
    )
    monkeypatch.setattr(finalizer._DirectoryAnchor, "sync", record_directory)

    result = _finalize(site, output_root, receipts_dir)

    assert result.published_dir == destination
    assert any(path.name == destination.name for path in synced)
    assert output_root in synced
    assert not finalizer._pending_guard_path(destination).exists()
    assert finalizer._accepted_marker_path(destination).is_file()


def test_finalizer_has_no_fallible_postpublish_verification_or_readback(
    tmp_path, monkeypatch
):
    site, output_root, receipts_dir = _roots(tmp_path)
    destination = output_root / "release-0001"
    original_read = finalizer._read_bundle
    original_verify = verifier.run_release_over_runs_root
    verify_calls = 0

    def reject_public_readback(directory):
        if Path(directory) == destination:
            raise AssertionError("complete public directory read too early")
        return original_read(directory)

    def count_verification(*args, **kwargs):
        nonlocal verify_calls
        verify_calls += 1
        assert not destination.exists()
        return original_verify(*args, **kwargs)

    monkeypatch.setattr(finalizer, "_read_bundle", reject_public_readback)
    monkeypatch.setattr(
        verifier, "run_release_over_runs_root", count_verification
    )

    result = _finalize(site, output_root, receipts_dir)

    assert result.published_dir == destination
    assert verify_calls == 1


@pytest.mark.parametrize("swap_seam", ("guard", "rename", "marker"))
def test_publication_transaction_stays_bound_to_validated_parent_identity(
    tmp_path, monkeypatch, swap_seam
):
    output_root = tmp_path / "output"
    output_root.mkdir()
    staged = output_root / ".staged"
    staged.mkdir()
    (staged / "payload.json").write_bytes(b"{}\n")
    destination = output_root / "bundle"
    parent_chain = finalizer._capture_directory_chain(output_root)
    displaced = tmp_path / "displaced-output"

    def swap_parent():
        output_root.rename(displaced)
        output_root.mkdir()
        (output_root / "replacement-sentinel.txt").write_text(
            "preserve", encoding="utf-8"
        )

    if swap_seam == "guard":
        original_guard = finalizer._create_pending_guard

        def swap_before_guard(path, **kwargs):
            swap_parent()
            return original_guard(path, **kwargs)

        monkeypatch.setattr(
            finalizer, "_create_pending_guard", swap_before_guard
        )
    elif swap_seam == "rename":
        original_publish = finalizer._DirectoryAnchor.publish_directory

        def swap_after_rename(anchor, *args, **kwargs):
            original_publish(anchor, *args, **kwargs)
            swap_parent()

        monkeypatch.setattr(
            finalizer._DirectoryAnchor, "publish_directory", swap_after_rename
        )
    else:
        original_marker = finalizer._create_accepted_marker

        def swap_before_marker(path, tree_sha256, **kwargs):
            swap_parent()
            return original_marker(path, tree_sha256, **kwargs)

        monkeypatch.setattr(
            finalizer, "_create_accepted_marker", swap_before_marker
        )

    expected_error = OSError if os.name == "nt" else schema.TypedIngressError
    match = None if os.name == "nt" else "publication parent"
    with pytest.raises(expected_error, match=match):
        _publish_test_directory(
            staged,
            destination,
            exists_label="test bundle",
            parent_chain=parent_chain,
            expected_names=("payload.json",),
        )

    if os.name == "nt":
        assert output_root.is_dir()
        assert not displaced.exists()
    else:
        assert (output_root / "replacement-sentinel.txt").read_text("utf-8") == (
            "preserve"
        )
        assert not finalizer._accepted_marker_path(displaced / "bundle").exists()


def test_transient_parent_swap_and_restore_cannot_issue_acceptance(
    tmp_path, monkeypatch
):
    output_root = tmp_path / "output"
    output_root.mkdir()
    staged = output_root / ".staged"
    staged.mkdir()
    (staged / "payload.json").write_bytes(b"verified\n")
    destination = output_root / "bundle"
    chain = finalizer._capture_directory_chain(output_root)
    displaced = tmp_path / "displaced"
    transient = tmp_path / "transient"

    def swap_restore_and_fake_publish(
        anchor, _staged_name, destination_name, **_kwargs
    ):
        output_root.rename(displaced)
        output_root.mkdir()
        wrong = output_root / destination_name
        wrong.mkdir()
        (wrong / "payload.json").write_bytes(b"wrong\n")
        output_root.rename(transient)
        displaced.rename(output_root)

    monkeypatch.setattr(
        finalizer._DirectoryAnchor,
        "publish_directory",
        swap_restore_and_fake_publish,
    )

    expected_error = OSError if os.name == "nt" else schema.ColmAimsError
    with pytest.raises(expected_error):
        _publish_test_directory(
            staged,
            destination,
            exists_label="test bundle",
            parent_chain=chain,
            expected_names=("payload.json",),
        )

    assert finalizer._pending_guard_path(destination).is_file() is (
        os.name != "nt"
    )
    assert not finalizer._accepted_marker_path(destination).exists()
    assert not destination.exists()
    if os.name == "nt":
        assert not displaced.exists()
        assert not transient.exists()
    else:
        assert (transient / "bundle" / "payload.json").read_bytes() == b"wrong\n"


def test_fake_publish_of_wrong_bytes_at_correct_destination_stays_pending(
    tmp_path, monkeypatch
):
    output_root = tmp_path / "output"
    output_root.mkdir()
    staged = output_root / ".staged"
    staged.mkdir()
    (staged / "payload.json").write_bytes(b"verified\n")
    destination = output_root / "bundle"
    chain = finalizer._capture_directory_chain(output_root)

    def publish_wrong_bytes(anchor, _staged_name, destination_name, **_kwargs):
        dest = anchor._path(destination_name)
        dest.mkdir()
        (dest / "payload.json").write_bytes(b"wrong\n")

    monkeypatch.setattr(
        finalizer._DirectoryAnchor, "publish_directory", publish_wrong_bytes
    )

    with pytest.raises(
        schema.TypedIngressError, match="identity|verified staged tree"
    ):
        _publish_test_directory(
            staged,
            destination,
            exists_label="test bundle",
            parent_chain=chain,
            expected_names=("payload.json",),
        )

    assert finalizer._pending_guard_path(destination).is_file()
    assert not finalizer._accepted_marker_path(destination).exists()


def test_post_marker_tree_mutation_is_caught_before_guard_retirement(
    tmp_path, monkeypatch
):
    output_root = tmp_path / "output"
    output_root.mkdir()
    staged = output_root / ".staged"
    staged.mkdir()
    (staged / "payload.json").write_bytes(b"verified\n")
    destination = output_root / "bundle"
    chain = finalizer._capture_directory_chain(output_root)
    original_marker = finalizer._create_accepted_marker

    def mutate_after_marker(path, tree_sha256, **kwargs):
        original_marker(path, tree_sha256, **kwargs)
        (Path(path) / "payload.json").write_bytes(b"mutated\n")

    monkeypatch.setattr(
        finalizer, "_create_accepted_marker", mutate_after_marker
    )

    with pytest.raises(schema.TypedIngressError, match="verified staged tree"):
        _publish_test_directory(
            staged,
            destination,
            exists_label="test bundle",
            parent_chain=chain,
            expected_names=("payload.json",),
        )

    assert finalizer._pending_guard_path(destination).is_file()
    assert finalizer._accepted_marker_path(destination).is_file()


def test_non_sibling_staging_is_refused_before_protocol_sidecars(tmp_path):
    output_root = tmp_path / "output"
    output_root.mkdir()
    stage_root = tmp_path / "stage-root"
    stage_root.mkdir()
    staged = stage_root / "staged"
    staged.mkdir()
    (staged / "payload.json").write_bytes(b"verified\n")
    destination = output_root / "bundle"
    chain = finalizer._capture_directory_chain(output_root)

    with pytest.raises(schema.ConfigSurfaceError, match="must be siblings"):
        _publish_test_directory(
            staged,
            destination,
            exists_label="test bundle",
            parent_chain=chain,
            expected_names=("payload.json",),
        )

    assert list(output_root.iterdir()) == []


def test_alias_publication_parent_and_missing_parent_are_refused(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    alias = tmp_path / "alias"
    try:
        alias.symlink_to(target, target_is_directory=True)
    except OSError as exc:  # pragma: no cover - Windows privilege-dependent
        pytest.skip(f"directory symlink creation unavailable: {exc}")
    staged = alias / ".staged"
    staged.mkdir()
    (staged / "payload.json").write_bytes(b"verified\n")
    destination = alias / "bundle"
    target_chain = finalizer._capture_directory_chain(target)

    with pytest.raises(schema.TypedIngressError, match="publication parent|alias"):
        _publish_test_directory(
            staged,
            destination,
            exists_label="test bundle",
            parent_chain=target_chain,
        )
    assert not destination.exists()

    missing_destination = tmp_path / "missing-parent" / "bundle"
    with pytest.raises(schema.TypedIngressError, match="publication parent"):
        finalizer._create_pending_guard(
            missing_destination,
            parent_chain=target_chain,
        )
    assert not missing_destination.parent.exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows handle-share contract")
def test_windows_parent_anchor_denies_rename_or_replacement(tmp_path):
    output_root = tmp_path / "output"
    output_root.mkdir()
    chain = finalizer._capture_directory_chain(output_root)

    with finalizer._DirectoryAnchor(
        output_root, chain, "test publication parent"
    ):
        with pytest.raises(OSError):
            output_root.rename(tmp_path / "displaced")

    assert output_root.is_dir()
    assert not (tmp_path / "displaced").exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows Volume-GUID contract")
def test_windows_local_anchor_uses_pinned_volume_guid_path(tmp_path):
    output_root = tmp_path / "output"
    output_root.mkdir()
    chain = finalizer._capture_directory_chain(output_root)

    with finalizer._DirectoryAnchor(
        output_root, chain, "test publication parent"
    ) as anchor:
        operation_path = str(anchor._operation_path)
        assert operation_path.casefold().startswith(r"\\?\volume{")
        assert len(anchor._win_handles) == len(chain.lexical)
        assert not operation_path.casefold().startswith(
            str(output_root.drive).casefold()
        )
        anchor.create_once("pinned.txt", b"pinned\n", exists_label="probe")

    assert (output_root / "pinned.txt").read_bytes() == b"pinned\n"


@pytest.mark.skipif(os.name != "nt", reason="Windows Volume-GUID contract")
def test_windows_mutations_and_publication_receive_only_pinned_paths(
    tmp_path, monkeypatch
):
    output_root = tmp_path / "output"
    output_root.mkdir()
    staged = output_root / ".staged"
    staged.mkdir()
    (staged / "payload.json").write_bytes(b"verified\n")
    chain = finalizer._capture_directory_chain(output_root)
    original_open = finalizer.os.open
    original_publish = finalizer._DirectoryAnchor._rename_windows_child_handle
    original_sync = fileio.fsync_directory
    opened: list[Path] = []
    published: list[tuple[Path, Path]] = []
    synced: list[Path] = []

    with finalizer._DirectoryAnchor(
        output_root, chain, "test publication parent"
    ) as anchor:
        pinned = anchor._operation_path
        assert pinned is not None

        def track_open(path, flags, mode=0o777, *, dir_fd=None):
            candidate = Path(path)
            if candidate.name == "guard.pending":
                opened.append(candidate)
            return original_open(path, flags, mode, dir_fd=dir_fd)

        def track_publish(parent_anchor, source_anchor, destination_name):
            published.append(
                (
                    Path(source_anchor._operation_path),
                    parent_anchor._path(destination_name),
                )
            )
            return original_publish(
                parent_anchor, source_anchor, destination_name
            )

        def track_sync(path):
            synced.append(Path(path))
            return original_sync(path)

        monkeypatch.setattr(finalizer.os, "open", track_open)
        monkeypatch.setattr(
            finalizer._DirectoryAnchor,
            "_rename_windows_child_handle",
            track_publish,
        )
        monkeypatch.setattr(fileio, "fsync_directory", track_sync)
        anchor.create_once(
            "guard.pending", b"pending\n", exists_label="test guard"
        )
        staged_chain = finalizer._capture_directory_chain(staged)
        destination_chain = finalizer._relocated_child_snapshot(
            chain, staged_chain, output_root / "bundle"
        )
        with finalizer._DirectoryAnchor(
            staged,
            staged_chain,
            "test staging source",
            delete_access=True,
        ) as source_anchor:
            anchor.publish_directory(
                staged.name,
                "bundle",
                exists_label="test bundle",
                source_anchor=source_anchor,
                source_snapshot=staged_chain,
                destination_snapshot=destination_chain,
            )

        assert opened == [pinned / "guard.pending"]
        assert synced == [pinned, pinned]
        assert published == [(pinned / staged.name, pinned / "bundle")]
        assert all(path.parent == pinned for path in opened)
        assert all(path.parent == pinned for pair in published for path in pair)

    assert (output_root / "guard.pending").read_bytes() == b"pending\n"
    assert (output_root / "bundle" / "payload.json").read_bytes() == b"verified\n"


@pytest.mark.skipif(os.name != "nt", reason="Windows Volume-GUID contract")
@pytest.mark.parametrize(
    "untrusted_final_path",
    (
        r"\\?\UNC\server\share\output",
        r"C:\output",
        r"\Device\HarddiskVolume1\output",
        r"\\?\Volume{not-a-guid}\output",
    ),
)
def test_windows_anchor_rejects_non_volume_guid_final_paths(
    tmp_path, monkeypatch, untrusted_final_path
):
    output_root = tmp_path / "output"
    output_root.mkdir()
    chain = finalizer._capture_directory_chain(output_root)
    staged = output_root / ".staged"
    staged.mkdir()
    (staged / "payload.json").write_bytes(b"verified\n")
    destination = output_root / "bundle"
    monkeypatch.setattr(
        finalizer._DirectoryAnchor,
        "_windows_final_volume_path",
        staticmethod(lambda _handle: untrusted_final_path),
    )

    with pytest.raises(schema.TypedIngressError, match=r"Volume\{GUID\}"):
        _publish_test_directory(
            staged,
            destination,
            exists_label="test bundle",
            parent_chain=chain,
            expected_names=("payload.json",),
        )
    assert staged.is_dir()
    assert not destination.exists()
    assert not finalizer._pending_guard_path(destination).exists()
    assert not finalizer._accepted_marker_path(destination).exists()

    displaced = tmp_path.with_name(f"{tmp_path.name}-displaced")
    tmp_path.rename(displaced)
    displaced.rename(tmp_path)


@pytest.mark.skipif(os.name != "nt", reason="Windows Volume-GUID contract")
def test_windows_anchor_rejects_mixed_volume_guid_component_chain(
    tmp_path, monkeypatch
):
    output_root = tmp_path / "nested" / "output"
    output_root.mkdir(parents=True)
    chain = finalizer._capture_directory_chain(output_root)
    first_guid = "11111111-1111-1111-1111-111111111111"
    second_guid = "22222222-2222-2222-2222-222222222222"
    calls = 0

    def mixed_final_path(_handle):
        nonlocal calls
        calls += 1
        if calls == 1:
            return rf"\\?\Volume{{{first_guid}}}" + "\\"
        return rf"\\?\Volume{{{second_guid}}}\part-{calls}"

    monkeypatch.setattr(
        finalizer._DirectoryAnchor,
        "_windows_final_volume_path",
        staticmethod(mixed_final_path),
    )
    with pytest.raises(
        schema.TypedIngressError, match="mixed|non-descendant|path changed"
    ):
        with finalizer._DirectoryAnchor(
            output_root, chain, "test publication parent"
        ):
            pass
    assert calls == 1


@pytest.mark.skipif(os.name != "nt", reason="Windows Volume-GUID contract")
def test_windows_anchor_final_path_resolution_failure_closes_handles(
    tmp_path, monkeypatch
):
    output_root = tmp_path / "output"
    output_root.mkdir()
    chain = finalizer._capture_directory_chain(output_root)

    def fail_resolution(_handle):
        raise schema.TypedIngressError("injected Volume-GUID resolution failure")

    monkeypatch.setattr(
        finalizer._DirectoryAnchor,
        "_windows_final_volume_path",
        staticmethod(fail_resolution),
    )
    with pytest.raises(schema.TypedIngressError, match="injected"):
        with finalizer._DirectoryAnchor(
            output_root, chain, "test publication parent"
        ):
            pass

    displaced = tmp_path.with_name(f"{tmp_path.name}-displaced")
    tmp_path.rename(displaced)
    displaced.rename(tmp_path)


@pytest.mark.parametrize(
    ("profile", "lexical_pair", "expected"),
    (
        ("legacy-3.11", (0x12345678, 0x1122334455667788), True),
        (
            "full-3.12",
            (0x1122334455667788, (1 << 96) | 0x8877665544332211),
            True,
        ),
        ("hybrid", (0x1122334455667788, 0x1122334455667788), False),
        ("neither", (0xCAFEBABE, 0xDEADBEEF), False),
    ),
)
def test_windows_stat_pair_profiles_are_whole_and_filesystem_independent(
    profile, lexical_pair, expected
):
    full_pair = (0x1122334455667788, (1 << 96) | 0x8877665544332211)
    legacy_pair = (0x12345678, 0x1122334455667788)
    lexical_identity = (*lexical_pair, 0o040755)

    assert (
        finalizer._windows_stat_pair_matches(
            lexical_identity, full_pair, legacy_pair
        )
        is expected
    ), profile


@pytest.mark.skipif(os.name != "nt", reason="Windows native snapshot contract")
def test_windows_original_snapshot_rejects_lexical_recapture_drift_and_cleans_up(
    tmp_path, monkeypatch
):
    output_root = tmp_path / "output"
    output_root.mkdir()
    actual = schema.stable_directory_chain(output_root, output_root)
    changed = list(actual)
    component, (device, inode, mode) = changed[-1]
    changed[-1] = (component, (device, inode + 1, mode))
    captures = iter((actual, tuple(changed)))
    monkeypatch.setattr(
        schema, "stable_directory_chain", lambda _path, _root: next(captures)
    )

    with pytest.raises(schema.TypedIngressError, match="changed during original"):
        finalizer._capture_directory_chain(output_root)

    displaced = tmp_path.with_name(f"{tmp_path.name}-displaced")
    tmp_path.rename(displaced)
    displaced.rename(tmp_path)
    assert output_root.is_dir()


@pytest.mark.skipif(os.name != "nt", reason="Windows native snapshot contract")
def test_windows_directory_anchor_refuses_raw_lexical_chain(tmp_path):
    output_root = tmp_path / "output"
    output_root.mkdir()
    raw = schema.stable_directory_chain(output_root, output_root)

    with pytest.raises(schema.TypedIngressError, match="full-native"):
        finalizer._DirectoryAnchor(output_root, raw, "test publication parent")


@pytest.mark.skipif(os.name != "nt", reason="Windows handle-share contract")
def test_windows_anchor_denies_ancestor_rename_and_closes_every_handle(tmp_path):
    output_root = tmp_path / "nested" / "output"
    output_root.mkdir(parents=True)
    chain = finalizer._capture_directory_chain(output_root)
    displaced = tmp_path.with_name(f"{tmp_path.name}-displaced")

    with finalizer._DirectoryAnchor(
        output_root, chain, "test publication parent"
    ):
        with pytest.raises(OSError):
            tmp_path.rename(displaced)

    tmp_path.rename(displaced)
    displaced.rename(tmp_path)
    assert output_root.is_dir()


@pytest.mark.skipif(os.name != "nt", reason="Windows handle cleanup contract")
def test_windows_anchor_acquisition_failure_closes_component_handles(
    tmp_path, monkeypatch
):
    output_root = tmp_path / "nested" / "output"
    output_root.mkdir(parents=True)
    chain = finalizer._capture_directory_chain(output_root)
    original = finalizer._require_unchanged_directory
    calls = 0

    def fail_after_handles(path, captured, label):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise schema.TypedIngressError("injected post-open recapture failure")
        return original(path, captured, label)

    monkeypatch.setattr(
        finalizer, "_require_unchanged_directory", fail_after_handles
    )
    with pytest.raises(schema.TypedIngressError, match="post-open"):
        with finalizer._DirectoryAnchor(
            output_root, chain, "test publication parent"
        ):
            pass

    displaced = tmp_path.with_name(f"{tmp_path.name}-displaced")
    tmp_path.rename(displaced)
    displaced.rename(tmp_path)
    assert output_root.is_dir()


@pytest.mark.skipif(os.name != "nt", reason="Windows FileIdInfo contract")
@pytest.mark.parametrize("wrong_field", ("volume", "file-id-high-64"))
def test_windows_anchor_rejects_same_wrong_native_identity_for_both_handles(
    tmp_path, monkeypatch, wrong_field
):
    output_root = tmp_path / "nested" / "output"
    output_root.mkdir(parents=True)
    chain = finalizer._capture_directory_chain(output_root)
    original = finalizer._DirectoryAnchor._windows_native_file_id
    calls = 0

    def same_wrong_identity(information):
        nonlocal calls
        calls += 1
        volume, file_id = original(information)
        if wrong_field == "volume":
            volume ^= 1
        else:
            file_id = file_id[:8] + bytes([file_id[8] ^ 1]) + file_id[9:]
        return volume, file_id

    monkeypatch.setattr(
        finalizer._DirectoryAnchor,
        "_windows_native_file_id",
        staticmethod(same_wrong_identity),
    )
    with pytest.raises(schema.TypedIngressError, match="native volume/file"):
        with finalizer._DirectoryAnchor(
            output_root, chain, "test publication parent"
        ):
            pass

    # Both the temporary share-delete handle and the locked handle returned
    # the same forged identity. Direct binding to captured st_dev/st_ino—not
    # merely temp-vs-anchor equality—must reject it and close the locked handle.
    assert calls == 2
    displaced = tmp_path.with_name(f"{tmp_path.name}-displaced")
    tmp_path.rename(displaced)
    displaced.rename(tmp_path)
    assert output_root.is_dir()


@pytest.mark.skipif(os.name != "nt", reason="Windows FileIdInfo contract")
def test_windows_anchor_rejects_full_128_bit_file_id_mismatch(
    tmp_path, monkeypatch
):
    output_root = tmp_path / "nested" / "output"
    output_root.mkdir(parents=True)
    chain = finalizer._capture_directory_chain(output_root)
    original = finalizer._DirectoryAnchor._windows_native_file_id
    calls = 0

    def mismatch_high_64_bits(information):
        nonlocal calls
        calls += 1
        volume, file_id = original(information)
        if calls == 2:
            file_id = file_id[:8] + bytes([file_id[8] ^ 1]) + file_id[9:]
        return volume, file_id

    monkeypatch.setattr(
        finalizer._DirectoryAnchor,
        "_windows_native_file_id",
        staticmethod(mismatch_high_64_bits),
    )
    with pytest.raises(schema.TypedIngressError, match="native volume/file"):
        with finalizer._DirectoryAnchor(
            output_root, chain, "test publication parent"
        ):
            pass

    displaced = tmp_path.with_name(f"{tmp_path.name}-displaced")
    tmp_path.rename(displaced)
    displaced.rename(tmp_path)
    assert calls == 2


@pytest.mark.skipif(os.name != "posix", reason="POSIX dir_fd contract")
def test_posix_anchor_keeps_child_operation_on_captured_parent(tmp_path):
    output_root = tmp_path / "output"
    output_root.mkdir()
    chain = finalizer._capture_directory_chain(output_root)
    displaced = tmp_path / "displaced"

    with finalizer._DirectoryAnchor(
        output_root, chain, "test publication parent"
    ) as anchor:
        output_root.rename(displaced)
        output_root.mkdir()
        with pytest.raises(schema.TypedIngressError, match="identity changed"):
            anchor.create_once("anchored.txt", b"captured\n", exists_label="probe")

    assert (displaced / "anchored.txt").read_bytes() == b"captured\n"
    assert list(output_root.iterdir()) == []


@pytest.mark.skipif(os.name != "posix", reason="POSIX dir_fd contract")
def test_posix_publish_rename_stays_on_captured_parent_during_transient_swap(
    tmp_path, monkeypatch
):
    output_root = tmp_path / "output"
    output_root.mkdir()
    staged = output_root / ".staged"
    staged.mkdir()
    (staged / "payload.json").write_bytes(b"verified\n")
    destination = output_root / "bundle"
    chain = finalizer._capture_directory_chain(output_root)
    original_rename = finalizer.os.rename
    displaced = tmp_path / "displaced-output"
    decoy = tmp_path / "decoy-output"
    swapped = False

    def swap_restore_around_anchored_rename(
        src, dst, *, src_dir_fd=None, dst_dir_fd=None
    ):
        nonlocal swapped
        if (
            src == staged.name
            and dst == destination.name
            and src_dir_fd is not None
            and dst_dir_fd == src_dir_fd
            and not swapped
        ):
            swapped = True
            original_rename(output_root, displaced)
            output_root.mkdir()
            (output_root / "decoy.txt").write_bytes(b"decoy\n")
            try:
                return original_rename(
                    src,
                    dst,
                    src_dir_fd=src_dir_fd,
                    dst_dir_fd=dst_dir_fd,
                )
            finally:
                original_rename(output_root, decoy)
                original_rename(displaced, output_root)
        return original_rename(
            src, dst, src_dir_fd=src_dir_fd, dst_dir_fd=dst_dir_fd
        )

    monkeypatch.setattr(finalizer.os, "rename", swap_restore_around_anchored_rename)
    _publish_test_directory(
        staged,
        destination,
        exists_label="test bundle",
        parent_chain=chain,
        expected_names=("payload.json",),
    )

    assert swapped is True
    assert (destination / "payload.json").read_bytes() == b"verified\n"
    assert finalizer._accepted_marker_path(destination).is_file()
    assert (decoy / "decoy.txt").read_bytes() == b"decoy\n"
    assert not (decoy / destination.name).exists()


@pytest.mark.parametrize(
    "defect", ("empty", "extra", "directory-member", "empty-subdirectory")
)
def test_accepted_bundle_requires_exact_three_ordinary_files(tmp_path, defect):
    site, output_root, receipts_dir = _roots(tmp_path)
    result = _finalize(site, output_root, receipts_dir)
    marker = finalizer._accepted_marker_path(result.published_dir)

    if defect == "empty":
        for name in ("ledger.json", "rights.json", "expectations.json"):
            (result.published_dir / name).unlink()
    elif defect == "extra":
        (result.published_dir / "extra.json").write_bytes(b"{}\n")
    elif defect == "empty-subdirectory":
        (result.published_dir / "empty").mkdir()
    else:
        (result.published_dir / "ledger.json").unlink()
        (result.published_dir / "ledger.json").mkdir()
    marker.write_bytes(
        finalizer._accepted_marker_bytes(
            result.published_dir,
            finalizer._directory_tree_sha256(result.published_dir),
        )
    )

    with pytest.raises(
        schema.TypedIngressError, match="exactly three|ordinary regular"
    ):
        finalizer._read_accepted_directory_snapshot(
            result.published_dir, "release bundle"
        )


def test_exact_membership_scan_stops_at_expected_count_plus_one(
    tmp_path, monkeypatch
):
    site, output_root, receipts_dir = _roots(tmp_path)
    result = _finalize(site, output_root, receipts_dir)
    original_scandir = os.scandir
    requested = 0

    class OverflowEntries:
        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return None

        def __iter__(self):
            return self

        def __next__(self):
            nonlocal requested
            requested += 1
            if requested <= 4:
                names = (
                    "ledger.json",
                    "rights.json",
                    "expectations.json",
                    "overflow.json",
                )
                return SimpleNamespace(name=names[requested - 1])
            raise AssertionError("scanner consumed beyond expected_count + 1")

    def bounded_scandir(path):
        if isinstance(path, int) or Path(path).name == result.published_dir.name:
            return OverflowEntries()
        return original_scandir(path)

    monkeypatch.setattr(os, "scandir", bounded_scandir)
    with pytest.raises(schema.TypedIngressError, match="exactly three"):
        finalizer._read_accepted_directory_snapshot(
            result.published_dir, "release bundle"
        )
    assert requested == 4
