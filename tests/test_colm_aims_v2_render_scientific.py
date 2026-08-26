"""Release-verified scientific JSON/CSV/TeX renderer regressions."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
from pathlib import Path

import pytest

from reproducibility.colm_aims_2026 import (
    phase4_finalize_release as finalizer,
    phase4_render_scientific as renderer,
)
from reproducibility.colm_aims_2026 import schema, verifier
from reproducibility.colm_aims_2026.phase4_finalize_release import (
    ReleaseVerificationFailed,
)
from reproducibility.colm_aims_2026.phase4_render_scientific import (
    CSV_NAME,
    DIRECTION_TEXT,
    JSON_NAME,
    KHARD_CAVEAT_ID,
    KHARD_CAVEAT_TEXT,
    KRANDOM_DISCLOSURE_TEXT,
    QUALIFIER_TEXT,
    TEX_NAME,
    render_scientific_release,
)
from scripts.stopdff_v5 import fileio
from tests._colm_aims_v2_helpers import (
    BANNED_PHRASES,
    BANNED_PHRASES_CASE_SENSITIVE,
    REQUIRED_QUALIFIER,
    build_runs_site,
)


def _roots(tmp_path: Path):
    site = build_runs_site(tmp_path)
    output_root = tmp_path / "scientific-output"
    receipts_dir = tmp_path / "scientific-receipts"
    output_root.mkdir()
    receipts_dir.mkdir()
    return site, output_root, receipts_dir


def _render(site, output_root: Path, receipts_dir: Path, **overrides):
    marker = finalizer._accepted_marker_path(site.root)
    if not marker.exists():
        marker.write_bytes(
            finalizer._accepted_marker_bytes(
                site.root,
                finalizer._directory_tree_sha256(site.root),
            )
        )
    values = {
        "runs_root": site.runs_root,
        "expectations": site.expectations_path,
        "output_root": output_root,
        "render_id": "render-0001",
        "receipts_dir": receipts_dir,
    }
    values.update(overrides)
    return render_scientific_release(**values)


def test_release_verified_render_has_exact_grid_and_disclosures(tmp_path):
    site, output_root, receipts_dir = _roots(tmp_path)

    result = _render(site, output_root, receipts_dir)

    assert result.report.verdict == verifier.VERDICT_RELEASE_PASS
    assert finalizer._accepted_marker_path(result.published_dir).is_file()
    assert not finalizer._pending_guard_path(result.published_dir).exists()
    assert {path.name for path in result.published_dir.iterdir()} == {
        JSON_NAME,
        CSV_NAME,
        TEX_NAME,
    }
    machine = json.loads(result.json_path.read_text("utf-8"))
    assert [row["cell_id"] for row in machine["cells"]] == list(schema.CELL_IDS)
    assert len(machine["cells"]) == 10
    random_rows = [
        row for row in machine["cells"] if row["reference_id"] == "krandom"
    ]
    assert len(random_rows) == 2
    assert all(row["headline_eligible"] is False for row in random_rows)
    assert all(
        row["reporting_eligibility"] == "non_headline_disclosure_only"
        for row in random_rows
    )
    hard_rows = [
        row for row in machine["cells"] if row["reference_id"] == "khard"
    ]
    assert all(row["caveat_id"] == KHARD_CAVEAT_ID for row in hard_rows)
    assert machine["semantic"] == schema.SEMANTIC_BLOCK
    assert machine["release_bindings"]["verdict"] == "PASS_RELEASE"
    assert machine["release_bindings"]["expectations_sha256"] == hashlib.sha256(
        site.expectations_path.read_bytes()
    ).hexdigest()
    receipt = json.loads(result.report.receipt_path.read_text("utf-8"))
    assert machine["release_bindings"]["input_tree_sha256"] == receipt[
        "input_tree_sha256"
    ]
    assert machine["release_bindings"]["expectations_anchor_sha256"] == (
        receipt["expectations_anchor_sha256"]
    )
    assert machine["release_bindings"]["verifier_code_sha256"] == receipt[
        "verifier_code_sha256"
    ]
    assert machine["release_bindings"]["verifier_revision"] == (
        schema.VERIFIER_REVISION
    )
    assert machine["qualifier"] == QUALIFIER_TEXT
    assert machine["direction"] == DIRECTION_TEXT

    csv_rows = list(
        csv.DictReader(io.StringIO(result.csv_path.read_text("utf-8")))
    )
    assert [row["cell_id"] for row in csv_rows] == list(schema.CELL_IDS)
    assert {row["qualifier"] for row in csv_rows} == {QUALIFIER_TEXT}
    assert {row["direction"] for row in csv_rows} == {DIRECTION_TEXT}
    assert {
        row["caveat_text"]
        for row in csv_rows
        if row["reference_id"] == "khard"
    } == {KHARD_CAVEAT_TEXT}
    assert {
        row["reference_disclosure_text"]
        for row in csv_rows
        if row["reference_id"] == "krandom"
    } == {KRANDOM_DISCLOSURE_TEXT}
    identity_fields = (
        "source_profile_sha256",
        "input_tree_sha256",
        "expectations_anchor_sha256",
        "verifier_revision",
        "verifier_code_sha256",
    )
    for row in csv_rows:
        for field in identity_fields:
            expected = (
                machine["source_profile_sha256"]
                if field == "source_profile_sha256"
                else machine["release_bindings"][field]
            )
            assert row[field] == expected
    tex = result.tex_path.read_text("utf-8")
    assert r"krandom$^{\dagger}$" in tex
    assert r"khard$^{\ddagger}$" in tex
    assert "same-space selection is circular" in tex
    assert "not observed open-ended response evidence" in tex
    for field in identity_fields:
        expected = (
            machine["source_profile_sha256"]
            if field == "source_profile_sha256"
            else machine["release_bindings"][field]
        )
        assert f"% {field}={expected}" in tex

    for path in (result.json_path, result.csv_path, result.tex_path):
        text = path.read_text("utf-8")
        lowered = text.lower()
        assert REQUIRED_QUALIFIER in lowered
        assert "difficulty-matched" not in lowered
        for phrase in BANNED_PHRASES:
            if phrase == "qa effect" and "constructed qa reference effect" in (
                lowered
            ):
                continue
            assert phrase not in lowered
        for term in BANNED_PHRASES_CASE_SENSITIVE:
            assert not re.search(rf"\b{term}\b", text)


def test_two_render_ids_produce_byte_identical_outputs(tmp_path):
    site, output_root, receipts_dir = _roots(tmp_path)
    first = _render(site, output_root, receipts_dir)
    second = _render(
        site,
        output_root,
        receipts_dir,
        render_id="render-0002",
    )

    for name in (JSON_NAME, CSV_NAME, TEX_NAME):
        assert (first.published_dir / name).read_bytes() == (
            second.published_dir / name
        ).read_bytes()


def test_failed_release_never_creates_scientific_output(tmp_path):
    site, output_root, receipts_dir = _roots(tmp_path)
    expectations = json.loads(site.expectations_path.read_text("utf-8"))
    expectations["bindings"]["inference"]["draw_count"] = 999
    site.expectations_path.write_bytes(schema.encode_json(expectations))

    with pytest.raises(ReleaseVerificationFailed, match="PASS_RELEASE"):
        _render(site, output_root, receipts_dir)

    assert not (output_root / "render-0001").exists()


def test_tree_mutation_during_release_verification_refuses_render(
    tmp_path, monkeypatch
):
    site, output_root, receipts_dir = _roots(tmp_path)
    original = verifier.run_release_over_runs_root

    def mutate_after_verification(*args, **kwargs):
        report = original(*args, **kwargs)
        profile_path = site.run_tree / "profile.json"
        profile_path.write_bytes(profile_path.read_bytes() + b" ")
        return report

    monkeypatch.setattr(
        verifier, "run_release_over_runs_root", mutate_after_verification
    )
    with pytest.raises(
        schema.TypedIngressError, match="changed during|exact tree"
    ):
        _render(site, output_root, receipts_dir)

    assert not (output_root / "render-0001").exists()


def test_scientific_output_inside_runs_root_is_refused(tmp_path):
    site = build_runs_site(tmp_path)
    receipts_dir = tmp_path / "scientific-receipts"
    receipts_dir.mkdir()

    with pytest.raises(schema.ConfigSurfaceError, match="outside the runs root"):
        _render(site, site.runs_root, receipts_dir)

    assert not (site.runs_root / "render-0001").exists()


def test_existing_render_slot_is_never_overwritten(tmp_path):
    site, output_root, receipts_dir = _roots(tmp_path)
    occupied = output_root / "render-0001"
    occupied.mkdir()
    sentinel = occupied / "sentinel.txt"
    sentinel.write_text("keep", encoding="utf-8")

    with pytest.raises(schema.ConfigSurfaceError, match="already exists"):
        _render(site, output_root, receipts_dir)

    assert sentinel.read_text("utf-8") == "keep"


@pytest.mark.parametrize("slot_kind", ("pending", "accepted"))
def test_existing_protocol_sibling_slot_is_never_overwritten(
    tmp_path, slot_kind
):
    site, output_root, receipts_dir = _roots(tmp_path)
    destination = output_root / "render-0001"
    sibling = (
        finalizer._pending_guard_path(destination)
        if slot_kind == "pending"
        else finalizer._accepted_marker_path(destination)
    )
    sibling.write_bytes(b"preexisting-sentinel")

    with pytest.raises(schema.ConfigSurfaceError, match="already exists"):
        _render(site, output_root, receipts_dir)

    assert not destination.exists()
    assert sibling.read_bytes() == b"preexisting-sentinel"


@pytest.mark.parametrize(
    "render_id",
    (
        "",
        ".",
        "..",
        "nested/name",
        r"nested\name",
        "stream:ads",
        "trailing.",
        "trailing ",
        "NUL",
        "con.txt",
        "COM1",
        "lpt9.log",
        "a" * 65,
        "naive-caf\N{LATIN SMALL LETTER E WITH ACUTE}",
    ),
)
def test_render_id_must_be_bounded_and_cross_platform_portable(
    tmp_path, render_id
):
    site, output_root, receipts_dir = _roots(tmp_path)

    with pytest.raises(schema.ConfigSurfaceError, match="portable"):
        _render(
            site,
            output_root,
            receipts_dir,
            render_id=render_id,
        )

    assert list(output_root.iterdir()) == []
    assert list(receipts_dir.iterdir()) == []


def test_renderer_will_not_mutate_expectations_authority_base(tmp_path):
    site, output_root, receipts_dir = _roots(tmp_path)

    with pytest.raises(schema.ConfigSurfaceError, match="authority base"):
        _render(
            site,
            output_root,
            site.receipts_dir,
        )
    with pytest.raises(schema.ConfigSurfaceError, match="authority base"):
        _render(
            site,
            site.root,
            receipts_dir,
        )

    assert not (output_root / "render-0001").exists()
    assert not (site.root / "render-0001").exists()


@pytest.mark.parametrize(
    "field",
    (
        "input_tree_sha256",
        "expectations_anchor_sha256",
        "verifier_code_sha256",
    ),
)
def test_verifier_receipt_must_bind_exact_captured_inputs_and_code(
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
        _render(site, output_root, receipts_dir)

    assert not (output_root / "render-0001").exists()


def test_staged_readback_failure_precedes_publication(tmp_path, monkeypatch):
    site, output_root, receipts_dir = _roots(tmp_path)

    def refuse_readback(_directory, **_kwargs):
        raise schema.TypedIngressError("injected staged readback failure")

    monkeypatch.setattr(renderer, "_read_outputs", refuse_readback)
    with pytest.raises(schema.TypedIngressError, match="injected staged"):
        _render(site, output_root, receipts_dir)

    assert not (output_root / "render-0001").exists()


def test_staged_fsync_failure_precedes_publication(tmp_path, monkeypatch):
    site, output_root, receipts_dir = _roots(tmp_path)

    def refuse_fsync(_directory):
        raise OSError("injected staged fsync failure")

    monkeypatch.setattr(fileio, "fsync_tree", refuse_fsync)
    with pytest.raises(OSError, match="injected staged fsync"):
        _render(site, output_root, receipts_dir)

    assert not (output_root / "render-0001").exists()


def test_committed_publish_requires_second_durability_barrier(
    tmp_path, monkeypatch
):
    site, output_root, receipts_dir = _roots(tmp_path)
    destination = output_root / "render-0001"
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

    result = _render(site, output_root, receipts_dir)

    assert result.published_dir == destination
    assert destination in synced
    assert output_root in synced
    assert not finalizer._pending_guard_path(destination).exists()


@pytest.mark.parametrize("failure_point", ("tree", "parent"))
def test_failed_retry_barrier_stays_guarded_and_is_not_success(
    tmp_path, monkeypatch, failure_point
):
    site, output_root, receipts_dir = _roots(tmp_path)
    destination = output_root / "render-0001"
    original_fsync_tree = fileio.fsync_tree
    committed = False

    def commit_then_fail(staged, dest, **_kwargs):
        nonlocal committed
        os.rename(staged, dest)
        committed = True
        raise fileio.DirectoryPublicationCommittedError(
            dest, OSError("injected parent sync failure")
        )

    def fail_destination_barrier(path):
        if failure_point == "tree" and Path(path) == destination:
            raise OSError("injected second durability failure")
        return original_fsync_tree(path)

    original_fsync_directory = fileio.fsync_directory

    def fail_parent_barrier(path):
        if (
            failure_point == "parent"
            and committed
            and Path(path) == output_root
        ):
            raise OSError("injected second durability failure")
        return original_fsync_directory(path)

    monkeypatch.setattr(fileio, "publish_dir_create_once", commit_then_fail)
    monkeypatch.setattr(fileio, "fsync_tree", fail_destination_barrier)
    monkeypatch.setattr(fileio, "fsync_directory", fail_parent_barrier)

    with pytest.raises(OSError, match="second durability"):
        _render(site, output_root, receipts_dir)

    # The rename already committed, but the slot remains mechanically pending.
    assert destination.is_dir()
    guard = finalizer._pending_guard_path(destination)
    assert guard.is_file()
    with pytest.raises(schema.TypedIngressError, match="pending"):
        renderer._require_accepted_directory(destination, "scientific output")


def test_terminal_publication_has_no_fallible_postpublish_readback(
    tmp_path, monkeypatch
):
    site, output_root, receipts_dir = _roots(tmp_path)
    destination = output_root / "render-0001"
    original = renderer._read_outputs

    def reject_public_readback(directory, **kwargs):
        if Path(directory) == destination:
            raise AssertionError("complete public directory read too early")
        return original(directory, **kwargs)

    monkeypatch.setattr(renderer, "_read_outputs", reject_public_readback)

    result = _render(site, output_root, receipts_dir)

    assert result.published_dir == destination


def test_guard_retirement_failure_leaves_complete_directory_pending(
    tmp_path, monkeypatch
):
    site, output_root, receipts_dir = _roots(tmp_path)
    destination = output_root / "render-0001"

    def refuse_retirement(_guard, _encoded, **_kwargs):
        raise OSError("injected guard retirement failure")

    monkeypatch.setattr(
        finalizer, "_retire_pending_guard", refuse_retirement
    )
    with pytest.raises(OSError, match="guard retirement"):
        _render(site, output_root, receipts_dir)

    assert destination.is_dir()
    assert finalizer._pending_guard_path(destination).is_file()
    with pytest.raises(schema.TypedIngressError, match="pending"):
        renderer._read_outputs(destination)


def test_acceptance_marker_failure_leaves_final_directory_guarded(
    tmp_path, monkeypatch
):
    site, output_root, receipts_dir = _roots(tmp_path)
    destination = output_root / "render-0001"

    def refuse_marker(_destination, _tree_sha256, **_kwargs):
        raise OSError("injected acceptance marker failure")

    monkeypatch.setattr(
        finalizer, "_create_accepted_marker", refuse_marker
    )
    with pytest.raises(OSError, match="acceptance marker"):
        _render(site, output_root, receipts_dir)

    assert destination.is_dir()
    assert finalizer._pending_guard_path(destination).is_file()
    assert not finalizer._accepted_marker_path(destination).exists()
    with pytest.raises(schema.TypedIngressError, match="pending"):
        renderer._read_outputs(destination)


def test_guard_unlink_sync_exhaustion_is_safe_live_acceptance(
    tmp_path, monkeypatch
):
    site, output_root, receipts_dir = _roots(tmp_path)
    destination = output_root / "render-0001"
    original = fileio.fsync_directory
    retirement_sync_calls = 0
    marker = finalizer._accepted_marker_path(destination)
    guard = finalizer._pending_guard_path(destination)

    def fail_guard_retirement_sync(path):
        nonlocal retirement_sync_calls
        if (
            Path(path) == output_root
            and marker.exists()
            and not guard.exists()
        ):
            retirement_sync_calls += 1
            raise OSError("injected guard retirement sync failure")
        return original(path)

    monkeypatch.setattr(
        fileio, "fsync_directory", fail_guard_retirement_sync
    )

    result = _render(site, output_root, receipts_dir)

    assert result.published_dir == destination
    assert retirement_sync_calls == 2
    assert finalizer._accepted_marker_path(destination).is_file()
    assert not finalizer._pending_guard_path(destination).exists()
    assert renderer._read_outputs(destination)


def test_positive_marker_binds_exact_detached_output_tree(tmp_path):
    site, output_root, receipts_dir = _roots(tmp_path)
    result = _render(site, output_root, receipts_dir)
    result.csv_path.write_bytes(result.csv_path.read_bytes() + b"\n")

    with pytest.raises(schema.TypedIngressError, match="exact tree"):
        renderer._read_outputs(result.published_dir)


def test_detached_reader_rejects_mutation_after_marker_bound_snapshot(
    tmp_path, monkeypatch
):
    site, output_root, receipts_dir = _roots(tmp_path)
    result = _render(site, output_root, receipts_dir)
    marker = finalizer._accepted_marker_path(result.published_dir)
    original_read = schema.read_regular_file_bytes
    mutated = False

    def mutate_after_marker_read(path, *args, **kwargs):
        nonlocal mutated
        data = original_read(path, *args, **kwargs)
        if Path(path) == marker and not mutated:
            mutated = True
            result.csv_path.write_bytes(result.csv_path.read_bytes() + b"\n")
        return data

    monkeypatch.setattr(
        schema, "read_regular_file_bytes", mutate_after_marker_read
    )

    with pytest.raises(schema.TypedIngressError, match="changed during"):
        renderer._read_outputs(result.published_dir)

    assert mutated is True


def test_guarded_release_authority_is_rejected_by_renderer(tmp_path):
    site, output_root, receipts_dir = _roots(tmp_path)
    guard = finalizer._pending_guard_path(site.root)
    guard.write_bytes(finalizer._pending_guard_bytes(site.root))

    with pytest.raises(schema.TypedIngressError, match="pending"):
        _render(site, output_root, receipts_dir)

    assert not (output_root / "render-0001").exists()
    assert list(receipts_dir.iterdir()) == []


def test_renderer_rechecks_directory_identities_before_publication(
    tmp_path, monkeypatch
):
    site, output_root, receipts_dir = _roots(tmp_path)
    original = renderer._require_unchanged_directory
    output_checks = 0

    def fail_second_output_check(path, captured, label):
        nonlocal output_checks
        if label == "scientific output root":
            output_checks += 1
            if output_checks == 2:
                raise schema.TypedIngressError("injected output-root swap")
        return original(path, captured, label)

    monkeypatch.setattr(
        renderer, "_require_unchanged_directory", fail_second_output_check
    )
    with pytest.raises(schema.TypedIngressError, match="output-root swap"):
        _render(site, output_root, receipts_dir)

    assert output_checks == 2
    assert not (output_root / "render-0001").exists()


def test_renderer_repeats_containment_immediately_before_publication(
    tmp_path, monkeypatch
):
    site, output_root, receipts_dir = _roots(tmp_path)
    original = renderer._require_render_boundaries
    checks = 0

    def fail_final_boundary_check(*args, **kwargs):
        nonlocal checks
        checks += 1
        if checks == 3:
            raise schema.ConfigSurfaceError("injected containment drift")
        return original(*args, **kwargs)

    monkeypatch.setattr(
        renderer, "_require_render_boundaries", fail_final_boundary_check
    )
    with pytest.raises(schema.ConfigSurfaceError, match="containment drift"):
        _render(site, output_root, receipts_dir)

    assert checks == 3
    assert not (output_root / "render-0001").exists()
