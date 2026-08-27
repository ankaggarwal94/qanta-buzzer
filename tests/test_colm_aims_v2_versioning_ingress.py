"""Versioning + ingress rules: R-059 (bool-safe version matrix, every
surface), R-060 (v1/v2 transition), R-061 (field-specific integer domains),
R-062 (overlong-token guard; +-2^53 ceiling removed), R-067 (native-finite CI
+ hardened parse hooks + AST parse-site enumeration), R-020 (typed ingress).

GREENFIELD RED: fails at collection until the namespace exists.
"""
from __future__ import annotations

import ast
import json

import pytest

from reproducibility.colm_aims_2026 import (
    phase4_finalize_release,
    schema,
    verifier,
)

from tests._colm_aims_v2_helpers import (
    EXIT_INGRESS_ERROR,
    SEMANTIC_BLOCK,
    V1_PROFILE_ID,
    VERDICT_SOURCE_PASS,
    VERIFIER_REVISION,
    build_package_v2,
    cli_args_for,
    colm_no_network,  # noqa: F401 - autouse fixture
    make_record_v2,
    namespace_py_files,
    rewrite_json,
    run_cli,
    run_verifier_on,
    source_report,
)

# The five injectable versioned surfaces (R-059). Records lines are NOT a
# versioned surface by decision (OQ-V2-003): enforced separately below.
SURFACE_FILES = {
    "profile": lambda pkg: pkg.profile_path,
    "ledger": lambda pkg: pkg.ledger_path,
    "manifest": lambda pkg: pkg.manifest_path,
    "expectations": lambda pkg: pkg.expectations_path,
    "rights": lambda pkg: pkg.rights_path,
}
BAD_VERSIONS = [True, 1.0, "2", 3]


class TestAggregateIngressBounds:
    def test_tree_file_count_limit(self, tmp_path, monkeypatch):
        tree = tmp_path / "tree"
        tree.mkdir()
        (tree / "a.json").write_text("{}", "utf-8")
        (tree / "b.json").write_text("{}", "utf-8")
        monkeypatch.setattr(verifier, "MAX_TREE_FILES", 1)
        with pytest.raises(schema.TypedIngressError, match="file-count"):
            verifier._read_tree_snapshot(tree)

    def test_tree_directory_count_limit(self, tmp_path, monkeypatch):
        tree = tmp_path / "tree"
        (tree / "nested").mkdir(parents=True)
        monkeypatch.setattr(verifier, "MAX_TREE_DIRECTORIES", 1)
        with pytest.raises(schema.TypedIngressError, match="directory-count"):
            verifier._read_tree_snapshot(tree)

    def test_tree_actual_byte_limit(self, tmp_path, monkeypatch):
        tree = tmp_path / "tree"
        tree.mkdir()
        (tree / "a.json").write_bytes(b"{}")
        monkeypatch.setattr(verifier, "MAX_TREE_BYTES", 1)
        with pytest.raises(schema.TypedIngressError, match="byte limit"):
            verifier._read_tree_snapshot(tree)

    def test_scandir_error_is_not_silently_ignored(self, tmp_path, monkeypatch):
        tree = tmp_path / "tree"
        tree.mkdir()

        def failing_scandir(*_args, **_kwargs):
            raise PermissionError("denied")

        monkeypatch.setattr(verifier.os, "scandir", failing_scandir)
        with pytest.raises(schema.TypedIngressError, match="traversal failed"):
            verifier._read_tree_snapshot(tree)

    def test_tree_depth_limit(self, tmp_path, monkeypatch):
        tree = tmp_path / "tree"
        (tree / "nested").mkdir(parents=True)
        monkeypatch.setattr(verifier, "MAX_TREE_DEPTH", 0)
        with pytest.raises(schema.TypedIngressError, match="depth limit"):
            verifier._read_tree_snapshot(tree)

    def test_tree_refuses_transient_child_swap_restore(
        self, tmp_path, monkeypatch
    ):
        tree = tmp_path / "tree"
        nested = tree / "nested"
        decoy = tree / "decoy"
        parked = tree / "parked"
        nested.mkdir(parents=True)
        decoy.mkdir()
        (nested / "original.json").write_bytes(b'{"source":"original"}')
        (decoy / "decoy.json").write_bytes(b'{"source":"decoy"}')
        original = (
            phase4_finalize_release._DirectoryAnchor
            ._claim_tree_child_directory
        )
        seam_fired = False

        def swap_before_claim(anchor, parent_fd, parent_path, name, *args):
            nonlocal seam_fired
            if name != "nested":
                return original(
                    anchor, parent_fd, parent_path, name, *args
                )
            seam_fired = True
            nested.rename(parked)
            decoy.rename(nested)
            try:
                return original(
                    anchor, parent_fd, parent_path, name, *args
                )
            finally:
                nested.rename(decoy)
                parked.rename(nested)

        monkeypatch.setattr(
            phase4_finalize_release._DirectoryAnchor,
            "_claim_tree_child_directory",
            swap_before_claim,
        )

        with pytest.raises(schema.TypedIngressError, match="changed identity"):
            verifier._read_tree_snapshot(tree)

        assert seam_fired is True
        assert (nested / "original.json").read_bytes() == (
            b'{"source":"original"}'
        )
        assert (decoy / "decoy.json").read_bytes() == b'{"source":"decoy"}'
        assert not parked.exists()

    def test_aggregate_record_row_limit(self, tmp_path, monkeypatch):
        pkg = build_package_v2(tmp_path)
        monkeypatch.setattr(verifier, "MAX_TOTAL_RECORDS", 1)
        with pytest.raises(schema.TypedIngressError, match="record-row"):
            run_verifier_on(pkg, "source")


def _set_version(path, value, *, drop: bool = False) -> None:
    def mutate(obj):
        if drop:
            obj.pop("schema_version", None)
        else:
            obj["schema_version"] = value

    rewrite_json(path, mutate)


# ---------------------------------------------------------------------------
# R-059: one bool-safe version checker across EVERY versioned surface
# ---------------------------------------------------------------------------


class TestVersionMatrix:
    @pytest.mark.parametrize("surface", sorted(SURFACE_FILES))
    @pytest.mark.parametrize(
        "bad", BAD_VERSIONS, ids=["true", "1.0", "str2", "3"]
    )
    def test_bad_version_raises_typed_version_error(
        self, tmp_path, surface, bad
    ):
        pkg = build_package_v2(tmp_path)
        _set_version(SURFACE_FILES[surface](pkg), bad)
        with pytest.raises(schema.SchemaVersionError) as excinfo:
            run_verifier_on(pkg, "release")
        message = str(excinfo.value)
        # Every version error names observed version, supported range, AND
        # the canonical revision token (closes A'-F1). String versions must
        # appear in repr form (quoting distinguishes "2" from 2).
        if isinstance(bad, str):
            assert repr(bad) in message
        else:
            assert repr(bad) in message or str(bad) in message
        assert "2" in message and (
            "2..2" in message or "supported" in message.lower()
        )
        assert VERIFIER_REVISION in message

    @pytest.mark.parametrize("surface", sorted(SURFACE_FILES))
    def test_absent_version_raises_typed_version_error(
        self, tmp_path, surface
    ):
        pkg = build_package_v2(tmp_path)
        _set_version(SURFACE_FILES[surface](pkg), None, drop=True)
        with pytest.raises(schema.SchemaVersionError) as excinfo:
            run_verifier_on(pkg, "release")
        assert VERIFIER_REVISION in str(excinfo.value)

    def test_json_true_is_not_version_one_or_two(self, tmp_path):
        # A'-F2 regression: bool must be rejected BEFORE any int comparison
        # (True == 1 under Python semantics). The expectations surface was
        # the v1 escape — pin it explicitly.
        pkg = build_package_v2(tmp_path)
        _set_version(pkg.expectations_path, True)
        with pytest.raises(schema.SchemaVersionError):
            run_verifier_on(pkg, "release")

    def test_version_error_is_typed_ingress_subclass(self):
        assert issubclass(schema.SchemaVersionError, schema.TypedIngressError)

    @pytest.mark.parametrize("surface", sorted(SURFACE_FILES))
    def test_combined_defect_names_version_first(self, tmp_path, surface):
        # Validation order: container shape -> schema_version -> everything
        # else. A co-present content defect must NOT preempt the version
        # error.
        pkg = build_package_v2(tmp_path)
        path = SURFACE_FILES[surface](pkg)

        def mutate(obj):
            obj["schema_version"] = 3
            obj["unknown_content_key"] = {"broken": True}
            if "rows" in obj:
                obj["rows"] = []
            if "artifacts" in obj:
                obj["artifacts"] = []

        rewrite_json(path, mutate)
        with pytest.raises(schema.SchemaVersionError) as excinfo:
            run_verifier_on(pkg, "release")
        assert VERIFIER_REVISION in str(excinfo.value)

    def test_container_shape_precedes_version_check(self, tmp_path):
        # A top-level ARRAY profile is a shape defect, not a version defect.
        pkg = build_package_v2(tmp_path)
        pkg.profile_path.write_bytes(b'[{"schema_version": 2}]\n')
        with pytest.raises(schema.TypedIngressError) as excinfo:
            run_verifier_on(pkg, "source")
        assert not isinstance(excinfo.value, schema.SchemaVersionError)

    def test_cli_exit_3_on_version_defect(self, tmp_path):
        pkg = build_package_v2(tmp_path)
        _set_version(pkg.profile_path, True)
        proc = run_cli(*cli_args_for(pkg, "source"))
        assert proc.returncode == EXIT_INGRESS_ERROR

    def test_records_lines_are_not_a_versioned_surface(self, tmp_path):
        # DECISION (OQ-V2-003): record JSONL lines carry NO envelope
        # version; a smuggled schema_version key is an unknown record field.
        rec = make_record_v2("itm-0001", 2, 3)
        rec["schema_version"] = 2
        with pytest.raises(schema.RecordValidationError):
            schema.validate_record(rec)


# ---------------------------------------------------------------------------
# R-060: strict v2 loaders never silently accept v1 documents
# ---------------------------------------------------------------------------


def _v1_flat_profile() -> dict:
    return {
        "schema_version": 1,
        "profile_id": V1_PROFILE_ID,
        "semantic": dict(SEMANTIC_BLOCK),
        "numerical_tolerance": 1e-9,
        "cells": [],
    }


class TestV1Transition:
    @pytest.mark.parametrize("surface", sorted(SURFACE_FILES))
    def test_v1_versioned_document_refused_on_every_strict_surface(
        self, tmp_path, surface
    ):
        pkg = build_package_v2(tmp_path)
        _set_version(SURFACE_FILES[surface](pkg), 1)
        with pytest.raises(schema.SchemaVersionError):
            run_verifier_on(pkg, "release")

    def test_legacy_loader_accepts_v1_but_never_certifies(self):
        from reproducibility.colm_aims_2026 import legacy

        blob = (json.dumps(_v1_flat_profile()) + "\n").encode("utf-8")
        out = legacy.load_legacy_v1_document(blob)
        assert out["certifying"] is False

    def test_strict_loader_refuses_what_legacy_loader_accepts(self):
        blob = (json.dumps(_v1_flat_profile()) + "\n").encode("utf-8")
        with pytest.raises(schema.TypedIngressError):
            schema.load_artifact_bytes(blob, "profile.json")

    def test_schema_module_has_no_strict_path_import_of_legacy(self):
        # OQ-V2-002: one legacy module, one entry point, no strict-path
        # imports of it from the schema loader.
        src = open(schema.__file__).read()
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert "legacy" not in node.module
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert "legacy" not in alias.name


# ---------------------------------------------------------------------------
# R-061: field-specific integer domains (both sides of each bound)
# ---------------------------------------------------------------------------


class TestIntegerDomains:
    def test_stop_step_lower_bound(self):
        rec = make_record_v2("itm-0001", -1, 2)
        with pytest.raises(schema.RecordValidationError):
            schema.validate_record(rec)

    def test_stop_step_zero_is_valid(self):
        schema.validate_record(make_record_v2("itm-0001", 0, 2))

    def test_stop_step_upper_bound_is_exclusive_horizon(self):
        # A FINITE stop at the horizon is the old sentinel coding — illegal
        # in the canonical representation.
        rec = make_record_v2("itm-0001", 6, 2)
        with pytest.raises(schema.RecordValidationError):
            schema.validate_record(rec)

    def test_stop_step_horizon_minus_one_is_valid(self):
        schema.validate_record(make_record_v2("itm-0001", 5, 2))

    def test_horizon_must_be_positive(self):
        rec = make_record_v2("itm-0001", 0, 0, trajectory_horizon=0)
        with pytest.raises(schema.RecordValidationError):
            schema.validate_record(rec)

    def test_horizon_bool_rejected(self):
        rec = make_record_v2("itm-0001", 0, 0)
        rec["trajectory_horizon"] = True
        with pytest.raises(schema.RecordValidationError):
            schema.validate_record(rec)

    def test_index_base_must_be_exactly_zero(self, tmp_path):
        from tests._colm_aims_v2_helpers import expected_estimand_digest

        def mutate(profile):
            for cell in profile["cells"]:
                cell["estimand"]["event_representation"]["index_base"] = 1
                cell["estimand_digest"] = expected_estimand_digest(
                    cell["estimand"]
                )

        pkg = build_package_v2(tmp_path, profile_mutator=mutate)
        report = source_report(pkg)
        assert report.verdict != VERDICT_SOURCE_PASS

    def test_counts_must_be_nonnegative(self, tmp_path):
        def mutate(profile):
            profile["cells"][0]["counts"]["n_both_timeout"] = -1
            profile["cells"][0]["counts"]["n_both_finite"] += 2

        pkg = build_package_v2(tmp_path, profile_mutator=mutate)
        report = source_report(pkg)
        assert report.verdict != VERDICT_SOURCE_PASS

    def test_count_bool_rejected(self, tmp_path):
        def mutate(profile):
            profile["cells"][0]["counts"]["n_excluded_or_unpaired"] = False

        pkg = build_package_v2(tmp_path, profile_mutator=mutate)
        report = source_report(pkg)
        assert report.verdict != VERDICT_SOURCE_PASS

    def test_count_exceeding_population_rejected(self, tmp_path):
        def mutate(profile):
            profile["cells"][0]["counts"]["n_both_finite"] = 5000

        pkg = build_package_v2(tmp_path, profile_mutator=mutate)
        report = source_report(pkg)
        assert report.verdict != VERDICT_SOURCE_PASS

    @pytest.mark.parametrize("bad_draws", [999, 1001])
    def test_draw_count_exactly_1000(self, tmp_path, bad_draws):
        def mutate(profile):
            profile["inference"]["draw_count"] = bad_draws

        pkg = build_package_v2(tmp_path, profile_mutator=mutate)
        report = source_report(pkg)
        assert report.verdict != VERDICT_SOURCE_PASS

    def test_schema_version_must_be_exactly_two(self, tmp_path):
        pkg = build_package_v2(tmp_path)
        _set_version(pkg.profile_path, 2)  # nearest-true: exact 2 stays green
        report = run_verifier_on(pkg, "source")
        assert report.verdict == VERDICT_SOURCE_PASS


# ---------------------------------------------------------------------------
# R-062: non-semantic token-length guard; global +-2^53 ceiling REMOVED
# ---------------------------------------------------------------------------

PARSE_SITE_FILES = {
    "profile": lambda pkg: pkg.profile_path,
    "ledger": lambda pkg: pkg.ledger_path,
    "manifest": lambda pkg: pkg.manifest_path,
    "expectations": lambda pkg: pkg.expectations_path,
    "rights": lambda pkg: pkg.rights_path,
    "records": lambda pkg: pkg.records_dir / "idealized__shared.jsonl",
}


def _inject_long_token(path, digits: int) -> None:
    raw = path.read_bytes().decode("utf-8")
    token = "9" * digits
    if path.suffix == ".jsonl":
        lines = raw.splitlines()
        lines[0] = lines[0].replace(
            '"trajectory_horizon": 6', f'"trajectory_horizon": {token}', 1
        )
        path.write_bytes(("\n".join(lines) + "\n").encode("utf-8"))
    else:
        assert '"schema_version": 2' in raw
        raw = raw.replace(
            '"schema_version": 2', f'"schema_version_pad": {token},'
            ' "schema_version": 2', 1
        )
        path.write_bytes(raw.encode("utf-8"))


class TestOverlongTokenGuard:
    @pytest.mark.parametrize("site", sorted(PARSE_SITE_FILES))
    def test_150_digit_token_typed_ingress_error_at_every_site(
        self, tmp_path, site
    ):
        pkg = build_package_v2(tmp_path)
        _inject_long_token(PARSE_SITE_FILES[site](pkg), 150)
        with pytest.raises(schema.TypedIngressError):
            run_verifier_on(pkg, "release")

    def test_4400_digit_token_never_escapes_as_bare_valueerror(
        self, tmp_path
    ):
        # Track A' R5: CPython's 4300-digit int-str limit raised a BARE
        # ValueError; the guard must fire lexically BEFORE int().
        pkg = build_package_v2(tmp_path)
        _inject_long_token(pkg.profile_path, 4400)
        with pytest.raises(schema.TypedIngressError):
            run_verifier_on(pkg, "source")

    def test_cli_exit_3_never_4_on_overlong_token(self, tmp_path):
        pkg = build_package_v2(tmp_path)
        _inject_long_token(pkg.profile_path, 4400)
        proc = run_cli(*cli_args_for(pkg, "source"))
        assert proc.returncode == EXIT_INGRESS_ERROR

    def test_legitimate_uint64_seed_parses(self, tmp_path):
        # D8/Track A' R4: the +-2^53 PARSE ceiling is removed — the 20-digit
        # uint64 canonical seed and a beyond-2^53 legacy metadata int both
        # parse. The canonical package itself carries a >2^53 seed.
        pkg = build_package_v2(tmp_path)
        assert pkg.profile["inference"]["seed"] > 2**53
        report = run_verifier_on(pkg, "source")
        assert report.verdict == VERDICT_SOURCE_PASS

    def test_beyond_float_exact_int_in_tolerated_sidecar_parses(
        self, tmp_path
    ):
        blob = json.dumps(
            {"legacy_ns_timestamp": 2**63 + 1, "note_family": "unknown"}
        ).encode("utf-8")
        pkg = build_package_v2(
            tmp_path,
            extra_tree_files={"sidecars/legacy_note.json": blob},
            extra_manifest_allowlist=("sidecars/legacy_note.json",),
            extra_rights_paths=("sidecars/legacy_note.json",),
        )
        report = run_verifier_on(pkg, "source")
        assert report.verdict == VERDICT_SOURCE_PASS

    def test_token_guard_constant_is_100_digits(self):
        assert schema.MAX_JSON_INT_TOKEN_DIGITS == 100


# ---------------------------------------------------------------------------
# R-067: native-finite ordered CI; hardened hooks at every parse site
# ---------------------------------------------------------------------------


class TestNativeFiniteInterval:
    @pytest.mark.parametrize(
        "bad_ci",
        [
            ["0.1", 0.4],
            [True, 1.0],
            [0.4, 0.1],
            [0.1],
            [0.1, 0.2, 0.3],
            [None, 0.4],
        ],
        ids=["str-lo", "bool-lo", "unordered", "one", "three", "null-lo"],
    )
    def test_ci_must_be_two_native_finite_ordered_numbers(
        self, tmp_path, bad_ci
    ):
        def mutate(profile):
            profile["cells"][0]["interval"]["ci"] = bad_ci

        pkg = build_package_v2(tmp_path, profile_mutator=mutate)
        report_or_error_fails(pkg)

    @pytest.mark.parametrize(
        "token", ["NaN", "Infinity", "-Infinity", "1e999"]
    )
    def test_nonfinite_tokens_rejected_at_ingress(self, tmp_path, token):
        pkg = build_package_v2(tmp_path)
        raw = pkg.profile_path.read_bytes().decode("utf-8")
        raw = raw.replace(
            '"schema_version": 2',
            f'"schema_version_pad": {token}, "schema_version": 2',
            1,
        )
        pkg.profile_path.write_bytes(raw.encode("utf-8"))
        with pytest.raises(schema.TypedIngressError):
            run_verifier_on(pkg, "source")

    def test_nonfinite_token_in_records_rejected(self, tmp_path):
        pkg = build_package_v2(tmp_path)
        path = pkg.records_dir / "idealized__shared.jsonl"
        lines = path.read_bytes().decode("utf-8").splitlines()
        lines[0] = lines[0].replace(
            '"trajectory_horizon": 6', '"trajectory_horizon": 1e999', 1
        )
        path.write_bytes(("\n".join(lines) + "\n").encode("utf-8"))
        with pytest.raises(schema.TypedIngressError):
            run_verifier_on(pkg, "source")

    def test_every_json_parse_site_routes_through_hardened_loader(self):
        # AST enumeration: json.loads/json.load may appear ONLY in schema.py
        # (the hardened loader home), and every schema.py json.loads carries
        # every protective hook, including duplicate-member rejection.
        for path in namespace_py_files():
            tree = ast.parse(path.read_text("utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                is_json_load = (
                    isinstance(func, ast.Attribute)
                    and func.attr in ("loads", "load")
                    and isinstance(func.value, ast.Name)
                    and func.value.id == "json"
                )
                if not is_json_load:
                    continue
                assert path.name == "schema.py", (
                    f"{path.name}:{node.lineno} raw json.{func.attr} outside"
                    " the hardened loader (R-067)"
                )
                kwarg_names = {kw.arg for kw in node.keywords}
                assert {
                    "object_pairs_hook",
                    "parse_constant",
                    "parse_float",
                    "parse_int",
                } <= kwarg_names, (
                    f"schema.py:{node.lineno} json parse without all"
                    " protective hooks (R-067/R-062)"
                )


def report_or_error_fails(pkg) -> None:
    """A defective package must FAIL the run — either a typed error or a
    FAIL verdict; a PASS is the only wrong answer."""
    try:
        report = run_verifier_on(pkg, "source")
    except schema.ColmAimsError:
        return
    assert report.verdict != VERDICT_SOURCE_PASS


# ---------------------------------------------------------------------------
# R-020: typed ingress names the file and field
# ---------------------------------------------------------------------------


class TestTypedIngress:
    def test_malformed_json_names_the_file(self, tmp_path):
        pkg = build_package_v2(tmp_path)
        pkg.profile_path.write_bytes(b'{"schema_version": 2,,,\n')
        with pytest.raises(schema.TypedIngressError) as excinfo:
            run_verifier_on(pkg, "source")
        assert "profile.json" in str(excinfo.value)

    def test_unknown_key_names_file_and_field(self, tmp_path):
        pkg = build_package_v2(tmp_path)
        rewrite_json(
            pkg.profile_path, lambda obj: obj.update(mystery_block={"x": 1})
        )
        with pytest.raises(
            (schema.TypedIngressError, schema.SchemaValidationError)
        ) as excinfo:
            run_verifier_on(pkg, "source")
        msg = str(excinfo.value)
        assert "mystery_block" in msg

    def test_invalid_utf8_names_the_file(self, tmp_path):
        pkg = build_package_v2(tmp_path)
        path = pkg.records_dir / "idealized__shared.jsonl"
        path.write_bytes(b'{"item_key": "itm-\xff\xfe"}\n')
        with pytest.raises(schema.TypedIngressError) as excinfo:
            run_verifier_on(pkg, "source")
        assert "idealized__shared.jsonl" in str(excinfo.value)

    def test_truncated_records_line_names_file_and_line(self, tmp_path):
        pkg = build_package_v2(tmp_path)
        path = pkg.records_dir / "khard__shared.jsonl"
        raw = path.read_bytes()
        path.write_bytes(raw[: len(raw) // 2])
        with pytest.raises(schema.TypedIngressError) as excinfo:
            run_verifier_on(pkg, "source")
        assert "khard__shared.jsonl" in str(excinfo.value)
