"""QA fix-round-1 regression suite — ingress hardening (QA-006, QA-013,
QA-014).

QA-006: every corrupted-input shape produces its pinned exit code with
typed, traceback-free, absolute-path-free stderr; a last-resort CLI handler
guarantees no ingest path can leak a traceback or collide with gate-FAIL.
QA-013: the vacuous-input error names the tree path exactly as supplied,
never the resolved form the caller did not supply.
QA-014: per-record validation errors name file + line; the rendered
"checks performed" list derives from legs actually present.
Spec: .correctless/specs/camera-ready-aims-evidence.md (R-020/R-026/R-033/R-037)
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from reproducibility.colm_aims_2026 import render, schema, verifier
from tests._colm_aims_helpers import (  # noqa: F401  (colm_no_network is an autouse fixture)
    EXIT_GATE_FAIL,
    EXIT_INGRESS_ERROR,
    EXIT_PASS,
    build_package,
    cli_args_for,
    colm_no_network,
    make_record,
    run_cli,
    standard_records,
)


def assert_typed_and_scrubbed(proc, pkg, fname: str) -> None:
    """QA-014 class helper: positive file identification (tree-relative name
    present) combined with the negative scrubs (no traceback, no absolute
    in-tree path, no home-rooted path on stderr)."""
    out = proc.stdout + proc.stderr
    assert fname in out, f"output does not name {fname!r}"
    assert "Traceback" not in proc.stderr, proc.stderr[-500:]
    assert str(pkg.tree / fname) not in out
    assert "/Users/" not in proc.stderr and "/home/" not in proc.stderr


CORRUPTIONS: dict[str, bytes] = {
    "truncated": b'{"schema_version": 1, "trunc',
    "bad_utf8": b'\xff\xfe{"schema_version": 1}',
    "wrong_type": b'"a bare string"',
    "empty": b"",
}

# QA-006 class fix: the corruption matrix — (file, mode, corruption,
# pinned exit code). Halting ingress defects exit 3; sidecar defects are
# collected legs (gate FAIL, exit 1). Exit-code assertions are EQUALITY.
CORRUPTION_MATRIX = [
    ("profile.json", "source", "truncated", EXIT_INGRESS_ERROR),
    ("profile.json", "source", "bad_utf8", EXIT_INGRESS_ERROR),
    ("profile.json", "source", "wrong_type", EXIT_INGRESS_ERROR),
    ("profile.json", "source", "empty", EXIT_INGRESS_ERROR),
    ("records.jsonl", "source", "truncated", EXIT_INGRESS_ERROR),
    ("records.jsonl", "source", "bad_utf8", EXIT_INGRESS_ERROR),
    ("records.jsonl", "source", "wrong_type", EXIT_INGRESS_ERROR),
    # Empty records parse to zero records: a collected (gate) defect, not an
    # ingress halt — collect-don't-halt keeps the verdict machinery running.
    ("records.jsonl", "source", "empty", EXIT_GATE_FAIL),
    ("presentation_manifest.json", "release", "truncated", EXIT_GATE_FAIL),
    ("presentation_manifest.json", "release", "bad_utf8", EXIT_GATE_FAIL),
    ("expectations.json", "release", "truncated", EXIT_INGRESS_ERROR),
    ("expectations.json", "release", "bad_utf8", EXIT_INGRESS_ERROR),
    ("expectations.json", "release", "wrong_type", EXIT_INGRESS_ERROR),
    ("expectations.json", "release", "empty", EXIT_INGRESS_ERROR),
    ("ledger.json", "release", "truncated", EXIT_GATE_FAIL),
    ("rights.json", "release", "truncated", EXIT_GATE_FAIL),
]


@pytest.mark.parametrize(
    "fname,mode,corruption,expected_code",
    CORRUPTION_MATRIX,
    ids=[f"{f}-{c}-{m}" for f, m, c, _ in CORRUPTION_MATRIX],
)
def test_corruption_matrix_pins_exit_codes_and_scrubbed_output(
    tmp_path: Path, fname, mode, corruption, expected_code
):
    # QA-006 [R-020/R-037/R-026]: every ingested file, four corruption
    # shapes — pinned exit code, typed message naming the file relatively,
    # traceback-free and absolute-path-free stderr.
    pkg = build_package(tmp_path)
    target = pkg.tree / fname if (pkg.tree / fname).exists() else pkg.root / fname
    target.write_bytes(CORRUPTIONS[corruption])
    proc = run_cli(*cli_args_for(pkg, mode))
    assert proc.returncode == expected_code, (
        fname,
        corruption,
        proc.returncode,
        proc.stderr[-300:],
    )
    if corruption != "empty":  # empty bytes carry no name to echo back
        assert_typed_and_scrubbed(proc, pkg, fname)
    else:
        assert "Traceback" not in proc.stderr
        assert str(pkg.tree / fname) not in (proc.stdout + proc.stderr)


def test_non_utf8_records_is_typed_at_the_api(tmp_path: Path):
    # QA-006 [R-020]: the API surface — non-UTF-8 records.jsonl raises the
    # TYPED ingress error naming the file, never a bare UnicodeDecodeError.
    tree = tmp_path / "tree"
    tree.mkdir()
    path = tree / "records.jsonl"
    path.write_bytes(b"\xff\xfe not utf8")
    with pytest.raises(schema.TypedIngressError) as exc:
        schema.load_artifact(path, tree_root=tree)
    msg = str(exc.value)
    assert "records.jsonl" in msg
    assert str(tmp_path) not in msg


def test_last_resort_handler_pins_exit_and_leaks_nothing(tmp_path: Path):
    # QA-006 class fix [R-037/R-026]: an unexpected exception on the ingest
    # path (receipts-dir pointing at a regular FILE trips an uncaught
    # OS-level error) hits the last-resort handler: pinned ingress exit code,
    # exception CLASS only, no traceback, no absolute paths.
    pkg = build_package(tmp_path)
    receipts_file = tmp_path / "receipts-as-file"
    receipts_file.write_text("not a directory", encoding="utf-8")
    proc = run_cli(
        "--mode",
        "source",
        "--tree",
        str(pkg.tree),
        "--receipts-dir",
        str(receipts_file),
    )
    assert proc.returncode == EXIT_INGRESS_ERROR, (
        proc.returncode,
        proc.stderr[-300:],
    )
    assert "Traceback" not in proc.stderr
    assert str(tmp_path) not in proc.stderr
    assert "no verdict was reached" in proc.stderr


# ---------------------------------------------------------------------------
# QA-013: vacuous-input error emits the tree path exactly as supplied
# ---------------------------------------------------------------------------


def test_vacuous_error_names_tree_path_as_supplied_never_resolved(
    tmp_path: Path,
):
    # QA-013 [R-033 vs R-026]: supplied-but-unresolved form appears; the
    # resolved target the caller never supplied does not.
    real = tmp_path / "real_tree_target"
    real.mkdir()
    link = tmp_path / "supplied_link"
    link.symlink_to(real)
    with pytest.raises(verifier.VacuousInputError) as exc:
        verifier.run_verifier(
            link, mode="source", receipts_dir=tmp_path / "receipts"
        )
    msg = str(exc.value)
    assert str(link) in msg
    assert "real_tree_target" not in msg
    assert "profile.json" in msg  # still names the expected layout (R-033)


def test_vacuous_cli_echoes_supplied_form(tmp_path: Path):
    # QA-013: the CLI surface — stderr carries the argument exactly as
    # supplied on the command line.
    real = tmp_path / "real_tree_target"
    real.mkdir()
    link = tmp_path / "supplied_link"
    link.symlink_to(real)
    proc = run_cli(
        "--mode",
        "source",
        "--tree",
        str(link),
        "--receipts-dir",
        str(tmp_path / "receipts"),
    )
    assert proc.returncode == EXIT_INGRESS_ERROR
    out = proc.stdout + proc.stderr
    assert str(link) in out
    assert "real_tree_target" not in out


# ---------------------------------------------------------------------------
# QA-014: file+line identification and legs-derived summary checks
# ---------------------------------------------------------------------------


def test_record_validation_error_names_file_and_line(tmp_path: Path):
    # QA-014 [R-020]: the seventh record carries a free-text field — the
    # records_validation leg names "records.jsonl: line 7: ...".
    records = standard_records() + [
        make_record("itm-0099", 1, 2, note="operator remark")
    ]
    pkg = build_package(tmp_path, records=records)
    report = verifier.run_verifier(
        pkg.tree, mode="source", receipts_dir=pkg.receipts_dir
    )
    assert report.verdict == "FAIL"
    legs = [
        leg
        for leg in report.legs
        if leg.get("leg_id") == "records_validation"
        and leg.get("outcome") == "FAIL"
    ]
    assert legs, "records_validation leg missing"
    observed = json.dumps(legs[0].get("observed"))
    assert "records.jsonl: line 7:" in observed
    assert str(pkg.tree) not in observed


def test_rendered_checks_derive_from_legs_actually_present():
    # QA-014: a report whose only leg is typed_ingress must not claim checks
    # that never ran — and without an emitted receipt, no receipt claim.
    report = verifier.VerificationReport(
        mode="source",
        verdict="FAIL",
        legs=[{"leg_id": "typed_ingress", "outcome": "PASS"}],
        validated_artifacts=[],
        receipt_path=None,
        classifications={},
    )
    summary = render.render_summary(report).lower()
    assert "typed ingress" in summary
    assert "profile validation" not in summary
    assert "rights inventory" not in summary
    assert "claim-ledger" not in summary
    assert "receipt emission" not in summary


def test_rendered_checks_include_release_families_when_present(tmp_path: Path):
    # QA-014: the full release run derives the full check list.
    pkg = build_package(tmp_path)
    report = verifier.run_verifier(
        pkg.tree,
        mode="release",
        receipts_dir=pkg.receipts_dir,
        expectations=pkg.expectations_path,
    )
    summary = render.render_summary(report).lower()
    for expected in (
        "typed ingress",
        "profile validation",
        "anchored expectation bindings",
        "rights inventory",
        "presentation manifest reconciliation",
        "claim-ledger status recomputation",
        "receipt emission",
    ):
        assert expected in summary, expected


def test_source_summary_still_lists_minimum_positive_checks(tmp_path: Path):
    # QA-014 guard: derivation preserves the R-017 minimum positive set on a
    # pristine source run (the existing R-017 assertions stay meaningful).
    pkg = build_package(tmp_path)
    report = verifier.run_verifier(
        pkg.tree, mode="source", receipts_dir=pkg.receipts_dir
    )
    summary = render.render_summary(report).lower()
    assert "profile validation" in summary
    assert "typed ingress" in summary
    assert "receipt" in summary


def test_cli_pass_still_exit_zero_after_ingress_hardening(tmp_path: Path):
    # Equality-pinned guard (QA-009 suite rule): the pristine source run
    # still exits EXIT_PASS after the QA-006 handler landed.
    pkg = build_package(tmp_path)
    proc = run_cli(*cli_args_for(pkg, "source"))
    assert proc.returncode == EXIT_PASS, proc.stderr[-300:]
