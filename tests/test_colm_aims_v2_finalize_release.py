"""Production release finalizer regressions.

The test fixtures author the authority documents; the finalizer may only copy
and verify their exact bytes.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

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
    original_fsync_tree = fileio.fsync_tree
    original_fsync_directory = fileio.fsync_directory
    synced: list[Path] = []

    def commit_then_fail(staged, dest, **_kwargs):
        os.rename(staged, dest)
        raise fileio.DirectoryPublicationCommittedError(
            dest, OSError("injected parent sync failure")
        )

    def record_tree(path):
        synced.append(Path(path))
        return original_fsync_tree(path)

    def record_directory(path):
        synced.append(Path(path))
        return original_fsync_directory(path)

    monkeypatch.setattr(fileio, "publish_dir_create_once", commit_then_fail)
    monkeypatch.setattr(fileio, "fsync_tree", record_tree)
    monkeypatch.setattr(fileio, "fsync_directory", record_directory)

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
