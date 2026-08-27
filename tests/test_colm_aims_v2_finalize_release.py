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


def _commit_without_parent_sync(anchor, staged_name: str, destination_name: str):
    """Commit through the platform's anchored branch, then skip its sync."""
    if anchor._fd is None:
        os.rename(anchor._path(staged_name), anchor._path(destination_name))
    else:
        os.mkdir(destination_name, dir_fd=anchor._fd)
        os.rename(
            staged_name,
            destination_name,
            src_dir_fd=anchor._fd,
            dst_dir_fd=anchor._fd,
        )
    return anchor._path(destination_name)


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
            anchor, staged_name, destination_name
        )
        raise fileio.DirectoryPublicationCommittedError(
            dest, OSError("injected parent sync failure")
        )

    def record_tree(anchor, name, expected_names):
        synced.append(anchor._path(name))
        return original_sync_tree(anchor, name, expected_names)

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
    assert destination in synced
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
    parent_chain = schema.stable_directory_chain(output_root, output_root)
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
        finalizer._publish_verified_directory(
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
    chain = schema.stable_directory_chain(output_root, output_root)
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
        finalizer._publish_verified_directory(
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
    chain = schema.stable_directory_chain(output_root, output_root)

    def publish_wrong_bytes(anchor, _staged_name, destination_name, **_kwargs):
        dest = anchor._path(destination_name)
        dest.mkdir()
        (dest / "payload.json").write_bytes(b"wrong\n")

    monkeypatch.setattr(
        finalizer._DirectoryAnchor, "publish_directory", publish_wrong_bytes
    )

    with pytest.raises(schema.TypedIngressError, match="verified staged tree"):
        finalizer._publish_verified_directory(
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
    chain = schema.stable_directory_chain(output_root, output_root)
    original_marker = finalizer._create_accepted_marker

    def mutate_after_marker(path, tree_sha256, **kwargs):
        original_marker(path, tree_sha256, **kwargs)
        (Path(path) / "payload.json").write_bytes(b"mutated\n")

    monkeypatch.setattr(
        finalizer, "_create_accepted_marker", mutate_after_marker
    )

    with pytest.raises(schema.TypedIngressError, match="verified staged tree"):
        finalizer._publish_verified_directory(
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
    chain = schema.stable_directory_chain(output_root, output_root)

    with pytest.raises(schema.ConfigSurfaceError, match="must be siblings"):
        finalizer._publish_verified_directory(
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
    target_chain = schema.stable_directory_chain(target, target)

    with pytest.raises(schema.TypedIngressError, match="publication parent"):
        finalizer._publish_verified_directory(
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
    chain = schema.stable_directory_chain(output_root, output_root)

    with finalizer._DirectoryAnchor(
        output_root, chain, "test publication parent"
    ):
        with pytest.raises(OSError):
            output_root.rename(tmp_path / "displaced")

    assert output_root.is_dir()
    assert not (tmp_path / "displaced").exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows handle-share contract")
def test_windows_anchor_denies_ancestor_rename_and_closes_every_handle(tmp_path):
    output_root = tmp_path / "nested" / "output"
    output_root.mkdir(parents=True)
    chain = schema.stable_directory_chain(output_root, output_root)
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
    chain = schema.stable_directory_chain(output_root, output_root)
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
def test_windows_anchor_rejects_full_128_bit_file_id_mismatch(
    tmp_path, monkeypatch
):
    output_root = tmp_path / "nested" / "output"
    output_root.mkdir(parents=True)
    chain = schema.stable_directory_chain(output_root, output_root)
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
    chain = schema.stable_directory_chain(output_root, output_root)
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
    chain = schema.stable_directory_chain(output_root, output_root)
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
    finalizer._publish_verified_directory(
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
        if isinstance(path, int) or Path(path) == result.published_dir:
            return OverflowEntries()
        return original_scandir(path)

    monkeypatch.setattr(os, "scandir", bounded_scandir)
    with pytest.raises(schema.TypedIngressError, match="exactly three"):
        finalizer._read_accepted_directory_snapshot(
            result.published_dir, "release bundle"
        )
    assert requested == 4
