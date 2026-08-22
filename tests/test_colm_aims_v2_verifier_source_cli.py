"""Source mode, CLI contract, receipts, vocabulary: R-017 (source floor +
verdict enum), R-018 (namespace containment; legacy verifier untouched),
R-022 (fail-closed flags), R-026 (sentinel-leak), R-027/R-057 (vocabulary
over rendered outputs), R-036 (receipts), R-037 (exit codes + invocation
forms), R-016 (create-once receipts), R-028 (guard self-test).

GREENFIELD RED: fails at collection until the namespace exists.
"""
from __future__ import annotations

import ast
import json
import re

import pytest

from reproducibility.colm_aims_2026 import receipt as receipt_mod
from reproducibility.colm_aims_2026 import render, schema, verifier, verify

from tests._colm_aims_v2_helpers import (
    BANNED_PHRASES,
    BANNED_PHRASES_CASE_SENSITIVE,
    CLOSURE_GATE_TOKEN,
    EXIT_GATE_FAIL,
    EXIT_INGRESS_ERROR,
    EXIT_PASS,
    EXIT_USAGE_ERROR,
    LEG_COUNTS,
    LEG_ESTIMAND_LABELS,
    LEG_GRID_COMPLETENESS,
    LEG_INFERENCE_RECOMPUTE,
    LEG_ITEM_KEY_SET,
    LEG_PROFILE_VALIDATION,
    LEG_TYPED_INGRESS,
    REPO_ROOT,
    REQUIRED_QUALIFIER,
    SENTINEL,
    VERDICT_FAIL,
    VERDICT_RELEASE_PASS,
    VERDICT_SOURCE_PASS,
    VERIFY_AUDIT_RELEASE_SHA256,
    build_package_v2,
    cli_args_for,
    colm_no_network,  # noqa: F401 - autouse fixture
    expected_code_sha256,
    expected_tree_sha256,
    latest_receipt,
    make_record_v2,
    namespace_py_files,
    run_cli,
    sha256_file,
    source_report,
)


# ---------------------------------------------------------------------------
# R-028: the no-network guard actually guards (self-test)
# ---------------------------------------------------------------------------


def test_no_network_guard_fires():
    import socket

    with pytest.raises(RuntimeError, match="network disabled"):
        socket.create_connection(("192.0.2.1", 80), timeout=0.1)


# ---------------------------------------------------------------------------
# R-017: source-mode floor, closed verdict vocabulary
# ---------------------------------------------------------------------------

SOURCE_FLOOR_LEGS = {
    LEG_TYPED_INGRESS,
    LEG_PROFILE_VALIDATION,
    LEG_GRID_COMPLETENESS,
    LEG_ITEM_KEY_SET,
    LEG_COUNTS,
    LEG_ESTIMAND_LABELS,
    LEG_INFERENCE_RECOMPUTE,
}


class TestSourceFloor:
    def test_source_mode_verdict_enum_closed(self):
        assert verifier.SOURCE_MODE_VERDICTS == frozenset(
            {VERDICT_SOURCE_PASS, VERDICT_FAIL}
        )

    def test_source_pass_includes_minimum_positive_check_set(self, tmp_path):
        pkg = build_package_v2(tmp_path)
        report = source_report(pkg)
        assert report.verdict == VERDICT_SOURCE_PASS
        leg_ids = {leg.get("leg_id") for leg in report.legs}
        missing = SOURCE_FLOOR_LEGS - leg_ids
        assert not missing, f"source floor legs missing: {missing}"
        assert report.validated_artifacts

    def test_no_release_token_in_source_output(self, tmp_path):
        pkg = build_package_v2(tmp_path)
        proc = run_cli(*cli_args_for(pkg, "source"))
        assert proc.returncode == EXIT_PASS
        assert VERDICT_SOURCE_PASS in proc.stdout
        assert VERDICT_RELEASE_PASS not in proc.stdout
        assert CLOSURE_GATE_TOKEN not in proc.stdout

    def test_source_summary_states_what_is_not_certified(self, tmp_path):
        pkg = build_package_v2(tmp_path)
        proc = run_cli(*cli_args_for(pkg, "source"))
        lowered = proc.stdout.lower()
        assert "not" in lowered and "certif" in lowered
        # and it lists the checks performed
        assert "check" in lowered or "leg" in lowered


# ---------------------------------------------------------------------------
# R-027 / R-057: vocabulary gate over every rendered output
# ---------------------------------------------------------------------------


def _assert_vocabulary_clean(text: str) -> None:
    lowered = text.lower()
    for phrase in BANNED_PHRASES:
        # allowed exception: the constructed-qa-reference qualifier itself
        if phrase == "qa effect" and "constructed qa reference effect" in (
            lowered
        ):
            continue
        assert phrase not in lowered, f"banned phrase {phrase!r} rendered"
    for term in BANNED_PHRASES_CASE_SENSITIVE:
        assert not re.search(rf"\b{term}\b", text), (
            f"ACM third-party term {term!r} rendered (R-017/R-027)"
        )


class TestVocabulary:
    def test_source_summary_vocabulary(self, tmp_path):
        pkg = build_package_v2(tmp_path)
        proc = run_cli(*cli_args_for(pkg, "source"))
        _assert_vocabulary_clean(proc.stdout)
        assert REQUIRED_QUALIFIER in proc.stdout.lower()

    def test_release_summary_vocabulary(self, tmp_path):
        pkg = build_package_v2(tmp_path)
        proc = run_cli(*cli_args_for(pkg, "release"))
        _assert_vocabulary_clean(proc.stdout)
        assert REQUIRED_QUALIFIER in proc.stdout.lower()

    def test_failing_run_vocabulary(self, tmp_path):
        def mutate(profile):
            profile["cells"][0]["counts"]["n_both_finite"] += 1

        pkg = build_package_v2(tmp_path, profile_mutator=mutate)
        proc = run_cli(*cli_args_for(pkg, "source"))
        assert proc.returncode == EXIT_GATE_FAIL
        _assert_vocabulary_clean(proc.stdout + proc.stderr)

    def test_render_summary_never_upgrades_scope(self, tmp_path):
        pkg = build_package_v2(tmp_path)
        report = source_report(pkg)
        text = render.render_summary(report)
        _assert_vocabulary_clean(text)
        assert CLOSURE_GATE_TOKEN not in text


# ---------------------------------------------------------------------------
# R-026: sentinel-leak protection on error paths
# ---------------------------------------------------------------------------


class TestSentinelLeak:
    def test_restricted_record_value_never_echoed(self):
        # A defective record whose VALUE is restricted content: the typed
        # error must reference the item by opaque key/field, never echo it.
        rec = make_record_v2("itm-0001", 2, 3)
        rec["free_text"] = SENTINEL
        with pytest.raises(schema.RecordValidationError) as excinfo:
            schema.validate_record(rec)
        assert SENTINEL not in str(excinfo.value)

    def test_failing_cli_run_is_sentinel_free_including_receipt(
        self, tmp_path
    ):
        def mutate(profile):
            profile["cells"][0]["counts"]["n_both_finite"] += 1

        pkg = build_package_v2(tmp_path, profile_mutator=mutate)
        proc = run_cli(*cli_args_for(pkg, "source"))  # auto sentinel assert
        assert proc.returncode == EXIT_GATE_FAIL
        for receipt_path in pkg.receipts_dir.glob("receipt-*.json"):
            assert SENTINEL not in receipt_path.read_text("utf-8")


# ---------------------------------------------------------------------------
# R-037: CLI contract — pinned exit codes and invocation forms
# ---------------------------------------------------------------------------


class TestCliContract:
    def test_exit_code_constants_pinned(self):
        assert verify.EXIT_PASS == 0
        assert verify.EXIT_GATE_FAIL == 1
        assert verify.EXIT_USAGE_ERROR == 2
        assert verify.EXIT_INGRESS_ERROR == 3
        assert verify.EXIT_INTERNAL_ERROR == 4

    def test_exit_0_on_source_pass(self, tmp_path):
        pkg = build_package_v2(tmp_path)
        proc = run_cli(*cli_args_for(pkg, "source"))
        assert proc.returncode == EXIT_PASS

    def test_exit_1_on_gate_fail(self, tmp_path):
        def mutate(profile):
            profile["cells"][0]["rates"]["rate_both_finite"] += 0.01

        pkg = build_package_v2(tmp_path, profile_mutator=mutate)
        proc = run_cli(*cli_args_for(pkg, "source"))
        assert proc.returncode == EXIT_GATE_FAIL

    def test_exit_2_on_usage_error(self, tmp_path):
        proc = run_cli("--mode", "nonsense")
        assert proc.returncode == EXIT_USAGE_ERROR

    def test_exit_3_on_ingress_error(self, tmp_path):
        pkg = build_package_v2(tmp_path)
        pkg.profile_path.write_bytes(b"not json at all\n")
        proc = run_cli(*cli_args_for(pkg, "source"))
        assert proc.returncode == EXIT_INGRESS_ERROR

    def test_direct_path_invocation_bootstraps_repo_root(self, tmp_path):
        # R-037: direct-path invocation works from ANY cwd via the
        # dedupe-then-insert-at-front sys.path bootstrap (or errors naming
        # the module-run form — the bootstrap contract is carried from v1).
        pkg = build_package_v2(tmp_path)
        script = REPO_ROOT / "reproducibility" / "colm_aims_2026" / "verify.py"
        import sys as _sys

        proc = run_cli(
            *cli_args_for(pkg, "source"),
            cwd=tmp_path,
            argv0=[_sys.executable, str(script)],
        )
        assert proc.returncode == EXIT_PASS, proc.stderr


# ---------------------------------------------------------------------------
# R-022: fail-closed flags and config
# ---------------------------------------------------------------------------


class TestFailClosedFlags:
    def test_unknown_flag_is_usage_error_not_noop(self, tmp_path):
        pkg = build_package_v2(tmp_path)
        proc = run_cli(*cli_args_for(pkg, "source"), "--frobnicate")
        assert proc.returncode == EXIT_USAGE_ERROR

    def test_abbreviated_flags_rejected(self, tmp_path):
        # allow_abbrev=False: --mod cannot smuggle past the unknown-flag
        # check as an abbreviation of --mode.
        pkg = build_package_v2(tmp_path)
        proc = run_cli(
            "--mod",
            "source",
            "--tree",
            str(pkg.tree),
            "--receipts-dir",
            str(pkg.receipts_dir),
        )
        assert proc.returncode == EXIT_USAGE_ERROR

    def test_no_gate_disable_door(self, tmp_path):
        pkg = build_package_v2(tmp_path)
        for door in ("--skip-rights", "--no-verify", "--allow-dirty"):
            proc = run_cli(*cli_args_for(pkg, "release"), door)
            assert proc.returncode == EXIT_USAGE_ERROR, door

    def test_missing_required_args_usage_error(self):
        proc = run_cli("--mode", "source")
        assert proc.returncode == EXIT_USAGE_ERROR


# ---------------------------------------------------------------------------
# R-036: schema-versioned receipts, create-once, outside the tree
# ---------------------------------------------------------------------------


class TestReceipts:
    def test_every_run_emits_receipt_with_pinned_fields(self, tmp_path):
        pkg = build_package_v2(tmp_path)
        proc = run_cli(*cli_args_for(pkg, "release"))
        assert proc.returncode == EXIT_PASS
        receipt = latest_receipt(pkg.receipts_dir)
        assert receipt["schema_version"] == 2
        assert receipt["mode"] == "release"
        assert receipt["verdict"] == VERDICT_RELEASE_PASS
        assert isinstance(receipt["legs"], list) and receipt["legs"]
        assert receipt["input_tree_sha256"] == expected_tree_sha256(pkg.tree)
        assert receipt["expectations_anchor_sha256"] == sha256_file(
            pkg.expectations_path
        )
        assert receipt["verifier_code_sha256"] == expected_code_sha256()
        assert "timestamp" in receipt

    def test_failing_run_still_emits_receipt(self, tmp_path):
        def mutate(profile):
            profile["cells"][0]["counts"]["n_both_finite"] += 1

        pkg = build_package_v2(tmp_path, profile_mutator=mutate)
        proc = run_cli(*cli_args_for(pkg, "source"))
        assert proc.returncode == EXIT_GATE_FAIL
        receipt = latest_receipt(pkg.receipts_dir)
        assert receipt["verdict"] == VERDICT_FAIL

    def test_two_runs_emit_two_distinct_receipts(self, tmp_path):
        pkg = build_package_v2(tmp_path)
        run_cli(*cli_args_for(pkg, "source"))
        run_cli(*cli_args_for(pkg, "source"))
        receipts = list(pkg.receipts_dir.glob("receipt-*.json"))
        assert len(receipts) == 2

    def test_receipt_create_once_same_run_id_fails(self, tmp_path):
        receipts_dir = tmp_path / "receipts"
        receipts_dir.mkdir()
        tree = tmp_path / "tree"
        tree.mkdir()
        payload = {"mode": "source", "verdict": VERDICT_FAIL, "legs": []}
        receipt_mod.emit_receipt(
            payload,
            receipts_dir=receipts_dir,
            verified_tree=tree,
            run_id="fixed-run",
        )
        with pytest.raises(Exception):
            receipt_mod.emit_receipt(
                payload,
                receipts_dir=receipts_dir,
                verified_tree=tree,
                run_id="fixed-run",
            )

    def test_receipt_inside_verified_tree_refused(self, tmp_path):
        tree = tmp_path / "tree"
        (tree / "receipts").mkdir(parents=True)
        with pytest.raises(schema.ColmAimsError):
            receipt_mod.emit_receipt(
                {"mode": "source", "verdict": VERDICT_FAIL, "legs": []},
                receipts_dir=tree / "receipts",
                verified_tree=tree,
                run_id="in-tree",
            )

    def test_receipt_uses_single_version_checker_constants(self, tmp_path):
        # If versioned, receipts use the R-058 constants (schema_version 2).
        pkg = build_package_v2(tmp_path)
        run_cli(*cli_args_for(pkg, "source"))
        receipt = latest_receipt(pkg.receipts_dir)
        assert receipt["schema_version"] == schema.SCHEMA_VERSION == 2


# ---------------------------------------------------------------------------
# R-018: namespace containment; legacy verifier byte-identical; primitives
# consumed, not forked
# ---------------------------------------------------------------------------


class TestNamespaceContainment:
    def test_everything_new_lives_under_the_namespace(self):
        for mod in (schema, verifier, render, verify, receipt_mod):
            assert "reproducibility/colm_aims_2026" in mod.__file__.replace(
                "\\", "/"
            )

    def test_legacy_release_verifier_byte_identical(self):
        assert (
            sha256_file(REPO_ROOT / "scripts" / "verify_audit_release.py")
            == VERIFY_AUDIT_RELEASE_SHA256
        )

    def test_stopdff_fileio_consumed_not_forked(self):
        # The create-once primitive is IMPORTED from scripts/stopdff_v5,
        # never re-defined inside the namespace.
        imports_fileio = False
        for path in namespace_py_files():
            tree = ast.parse(path.read_text("utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module and (
                    "stopdff_v5" in node.module
                ):
                    imports_fileio = True
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if "stopdff_v5" in alias.name:
                            imports_fileio = True
                if isinstance(node, ast.FunctionDef) and node.name in (
                    "create_once_bytes",
                    "create_once_dir",
                ):
                    pytest.fail(
                        f"{path.name} re-defines fileio primitive"
                        f" {node.name} (R-018: consume, don't fork)"
                    )
        assert imports_fileio


# ---------------------------------------------------------------------------
# R-033 (source side): vacuous inputs
# ---------------------------------------------------------------------------


def test_pass_requires_at_least_one_validated_artifact(tmp_path):
    pkg = build_package_v2(tmp_path)
    report = source_report(pkg)
    assert report.verdict == VERDICT_SOURCE_PASS
    assert len(report.validated_artifacts) >= 1
    receipt_files = list(pkg.receipts_dir.glob("receipt-*.json"))
    assert receipt_files, "PASS-class verdict without a receipt (R-036)"
    receipt = json.loads(receipt_files[-1].read_text("utf-8"))
    assert receipt.get("validated_artifacts") or receipt.get("legs")


# ---------------------------------------------------------------------------
# QA round 1 fixes (QA2-001 / QA2-002): hostile-ingress error-path discipline
# ---------------------------------------------------------------------------


class TestQaRound1ErrorPathDiscipline:
    def test_qa2_001_deeply_nested_json_is_typed_ingress_not_internal(
        self, tmp_path
    ):
        # QA2-001 exploit [R-020/R-037/R-062]: a deeply-nested hostile
        # profile raised RecursionError through the hooked parser and leaked
        # to exit 4 (internal error). It must be a TYPED ingress refusal
        # (exit 3) whose message names the file, never the exception class.
        pkg = build_package_v2(tmp_path)
        pkg.profile_path.write_bytes(b"[" * 100000 + b"]" * 100000)
        proc = run_cli(*cli_args_for(pkg, "source"))
        assert proc.returncode == EXIT_INGRESS_ERROR, proc.stderr
        assert "RecursionError" not in proc.stderr
        assert "profile.json" in proc.stderr

    def test_qa2_002_exit3_messages_scrub_absolute_paths(self, tmp_path):
        # QA2-002 exploit [R-026]: the exit-3 branch printed the raw
        # absolute --tree path (VacuousInputError); every user-facing error
        # branch must be path-scrubbed like the exit-4 branch already was.
        marker = tmp_path / "qa2_scrub_marker_dir"
        empty = marker / "tree"
        empty.mkdir(parents=True)
        receipts = tmp_path / "receipts"
        proc = run_cli(
            "--mode", "source", "--tree", str(empty),
            "--receipts-dir", str(receipts),
        )
        assert proc.returncode == EXIT_INGRESS_ERROR, proc.stderr
        assert str(empty) not in proc.stderr
        assert "qa2_scrub_marker_dir" not in proc.stderr


# ---------------------------------------------------------------------------
# Mini-audit round 1 fixes (MA2-001 / MA2-002): scrub fidelity + leg-builder
# crash containment
# ---------------------------------------------------------------------------


class TestMiniAuditRound1Fixes:
    def test_ma2_001_scrubber_preserves_relative_paths_and_citations(
        self, tmp_path
    ):
        # MA2-001 exploit [R-020/R-026]: the QA2-002 scrubber's fallback
        # regex collapsed interior "/" tokens, mangling tree-relative file
        # names and R-x/R-y rule citations. Deep-nest ONE record line: the
        # typed exit-3 message must carry the record's tree-relative path
        # and the composite citation VERBATIM while the absolute tree path
        # stays scrubbed.
        bomb = b"[" * 100000 + b"]" * 100000 + b"\n"
        pkg = build_package_v2(
            tmp_path, raw_records_bytes={"idealized__shared": bomb}
        )
        proc = run_cli(*cli_args_for(pkg, "source"))
        assert proc.returncode == EXIT_INGRESS_ERROR, proc.stderr
        assert "records/idealized__shared.jsonl" in proc.stderr
        assert "R-062/R-020" in proc.stderr
        assert str(pkg.tree) not in proc.stderr

    @pytest.mark.parametrize(
        "mutate",
        [
            pytest.param(
                lambda p: p["grid"].__setitem__("reference_ids", [{}, {}]),
                id="unsortable-axis-entries",
            ),
            pytest.param(
                lambda p: p["grid"].__setitem__(
                    "record_files",
                    {k: {} for k in p["grid"]["record_files"]},
                ),
                id="unhashable-record-file-targets",
            ),
            pytest.param(
                lambda p: p["cells"][0].__setitem__("headline_summary", [1]),
                id="truthy-nondict-summary",
            ),
        ],
    )
    def test_ma2_002_profile_shapes_fail_closed_with_receipt(
        self, tmp_path, mutate
    ):
        # MA2-002 exploit [R-012/R-036/R-037]: artifact-controlled shapes
        # inside leg-read fields crashed leg builders to exit 4 with NO
        # receipt. They must be an owning-leg FAIL: verdict FAIL, exit 1,
        # receipt emitted, never "no verdict was reached".
        pkg = build_package_v2(tmp_path, profile_mutator=mutate)
        proc = run_cli(*cli_args_for(pkg, "source"))
        assert proc.returncode == EXIT_GATE_FAIL, proc.stderr
        assert "no verdict was reached" not in proc.stderr
        assert list(pkg.receipts_dir.glob("receipt-*.json")), (
            "failing verdict must still emit a receipt (R-036)"
        )

    def test_ma2_002_ledger_status_shape_fails_closed_with_receipt(
        self, tmp_path
    ):
        # MA2-002 release variant: an unhashable ledger row status escaped
        # the LedgerValidationError catch to exit 4.
        def bad_status(ledger):
            ledger["rows"][0]["status"] = ["FAIL"]

        pkg = build_package_v2(tmp_path, ledger_mutator=bad_status)
        proc = run_cli(*cli_args_for(pkg, "release"))
        assert proc.returncode == EXIT_GATE_FAIL, proc.stderr
        assert "no verdict was reached" not in proc.stderr
        assert list(pkg.receipts_dir.glob("receipt-*.json"))
