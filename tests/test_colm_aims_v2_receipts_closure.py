"""Suite-evidence receipts, CAMERA_READY_CLOSURE, QA-012, doc audit:
R-070 (A'-F4 machine-readable receipt), R-071 (closure gate distinct from
PASS_RELEASE, frozen D6 inventory, Holm-row blocking), R-072 (QA-012
format:"QA" detector + inventory manifest), R-038 (seven-target doc audit).

GREENFIELD RED: fails at collection until the namespace exists.
"""
from __future__ import annotations

import copy
import json

import pytest

from reproducibility.colm_aims_2026 import closure, qa012, receipt as receipt_mod
from reproducibility.colm_aims_2026 import schema
from reproducibility.colm_aims_2026 import phase4_assemble_d7b as assembler

from tests._colm_aims_v2_helpers import (
    ANALYSIS_PROVENANCE_D7B,
    CLOSURE_GATE_TOKEN,
    D6_MAIN_PDF_SHA256,
    D6_MAIN_TEX_SHA256,
    FAKE_SHA_A,
    REPO_ROOT,
    VERDICT_RELEASE_PASS,
    colm_no_network,  # noqa: F401 - autouse fixture
    make_closure_inventory,
    make_closure_profile_bytes,
    make_suite_receipt,
    sha256_bytes,
)


# ---------------------------------------------------------------------------
# R-070: machine-readable suite receipt (all bindings are hashes)
# ---------------------------------------------------------------------------

SUITE_RECEIPT_REQUIRED_FIELDS = (
    "environment_lock_sha256",
    "workflow_sha256",
    "interpreter_realpath",
    "commit",
    "tree_sha256",
    "dirty",
    "command",
    "exit_code",
    "junit_sha256",
    "transcript_sha256",
    "counts",
    "skip_identities",
)


class TestSuiteReceipt:
    def test_canonical_suite_receipt_validates(self):
        receipt_mod.validate_suite_receipt(make_suite_receipt())

    @pytest.mark.parametrize("field", SUITE_RECEIPT_REQUIRED_FIELDS)
    def test_missing_required_field_rejected(self, field):
        doc = make_suite_receipt()
        del doc[field]
        with pytest.raises(schema.ColmAimsError):
            receipt_mod.validate_suite_receipt(doc)

    def test_a_prime_f4_defect_environment_digest_object_rejected(self):
        # The A'-F4 defect shape: the environment-lock field is a metadata
        # OBJECT instead of a lockfile/environment-export HASH.
        doc = make_suite_receipt(
            environment_lock_sha256={
                "python": "3.11.15",
                "numpy": "2.4.6",
                "platform": "darwin",
            }
        )
        with pytest.raises(schema.ColmAimsError) as excinfo:
            receipt_mod.validate_suite_receipt(doc)
        assert "environment_lock_sha256" in str(excinfo.value)

    def test_environment_digest_must_be_64_hex(self):
        doc = make_suite_receipt(environment_lock_sha256="not-a-hash")
        with pytest.raises(schema.ColmAimsError):
            receipt_mod.validate_suite_receipt(doc)

    def test_command_must_be_exact_argv_list(self):
        doc = make_suite_receipt(command="python -m pytest tests/ -q")
        with pytest.raises(schema.ColmAimsError):
            receipt_mod.validate_suite_receipt(doc)

    def test_commit_and_tree_must_be_full_length(self):
        with pytest.raises(schema.ColmAimsError):
            receipt_mod.validate_suite_receipt(
                make_suite_receipt(commit="abc123")
            )
        with pytest.raises(schema.ColmAimsError):
            receipt_mod.validate_suite_receipt(
                make_suite_receipt(tree_sha256="HEAD")
            )

    def test_sha256_repository_commit_and_tree_ids_validate(self):
        receipt_mod.validate_suite_receipt(
            make_suite_receipt(commit="a" * 64, tree_sha256="b" * 64)
        )

    def test_skip_identities_must_be_a_list(self):
        doc = make_suite_receipt(skip_identities="none")
        with pytest.raises(schema.ColmAimsError):
            receipt_mod.validate_suite_receipt(doc)

    def test_dirty_must_be_bool(self):
        doc = make_suite_receipt(dirty="false")
        with pytest.raises(schema.ColmAimsError):
            receipt_mod.validate_suite_receipt(doc)

    def test_skip_identities_must_match_skipped_count(self):
        doc = make_suite_receipt(skip_identities=[], counts={
            "tests": 10, "failures": 0, "errors": 0, "skipped": 1
        })
        with pytest.raises(schema.ColmAimsError):
            receipt_mod.validate_suite_receipt(doc)

    @pytest.mark.parametrize(
        "counts",
        [
            {"tests": 0, "failures": 0, "errors": 0, "skipped": 0},
            {"tests": 2, "failures": 1, "errors": 0, "skipped": 0},
            {"tests": 2, "failures": 0, "errors": 0, "skipped": 2},
            {"tests": 2, "failures": 0, "errors": 0, "skipped": False},
        ],
    )
    def test_counts_must_describe_a_nonempty_green_suite(self, counts):
        with pytest.raises(schema.ColmAimsError):
            receipt_mod.validate_suite_receipt(make_suite_receipt(counts=counts))


# ---------------------------------------------------------------------------
# R-071: CAMERA_READY_CLOSURE gate
# ---------------------------------------------------------------------------


class TestClosureGate:
    @staticmethod
    def _inventory_and_roots(tmp_path):
        roots = {}
        for prong in qa012.REQUIRED_SCOPE_PRONGS:
            root = tmp_path / prong
            root.mkdir(parents=True)
            (root / "clean.json").write_text('{"kind":"clean"}', "utf-8")
            roots[prong] = root
        manifest = qa012.build_inventory_manifest(roots)
        inventory = make_closure_inventory(
            qa012=assembler.qa012_status_block(manifest)
        )
        return inventory, roots

    def test_gate_token_distinct_from_release_pass(self):
        assert closure.CAMERA_READY_CLOSURE == CLOSURE_GATE_TOKEN
        assert closure.CAMERA_READY_CLOSURE != VERDICT_RELEASE_PASS

    def test_d6_baseline_constants_pinned(self):
        assert closure.D6_MAIN_TEX_SHA256 == D6_MAIN_TEX_SHA256
        assert closure.D6_MAIN_PDF_SHA256 == D6_MAIN_PDF_SHA256

    @pytest.mark.parametrize("version", [None, True, 1, 3, "2"])
    def test_closure_schema_version_is_required_and_exact(self, version):
        inventory = make_closure_inventory()
        if version is None:
            del inventory["schema_version"]
        else:
            inventory["schema_version"] = version
        with pytest.raises(schema.ColmAimsError):
            closure.evaluate_closure(inventory)

    @pytest.mark.parametrize(
        "block",
        ["top", "d6_baseline", "row", "holm_row", "qa012"],
    )
    def test_closure_nested_key_sets_are_closed(self, block):
        inventory = make_closure_inventory()
        if block == "top":
            target = inventory
        elif block == "row":
            target = inventory["rows"][0]
        else:
            target = inventory[block]
        target["unexpected"] = "value"
        with pytest.raises(schema.SchemaValidationError):
            closure.evaluate_closure(inventory)

    def test_duplicate_closure_row_ids_are_rejected(self):
        inventory = make_closure_inventory()
        inventory["rows"].append(dict(inventory["rows"][0]))
        with pytest.raises(schema.SchemaValidationError, match="duplicate row"):
            closure.evaluate_closure(inventory)

    def test_fully_satisfied_inventory_passes(self, tmp_path):
        inventory, roots = self._inventory_and_roots(tmp_path)
        # The Holm row is satisfiable ONLY by the D7(b) discriminator token.
        assert inventory["holm_row"]["satisfied_by"] == ANALYSIS_PROVENANCE_D7B
        out = closure.evaluate_closure(
            inventory,
            profile_bytes=make_closure_profile_bytes(),
            qa012_roots=roots,
        )
        assert out["gate"] == CLOSURE_GATE_TOKEN
        assert out["satisfied"] is True

    def test_satisfied_except_holm_fails(self, tmp_path):
        # The Holm/inference row is satisfied ONLY by the D7(b) regenerated
        # outputs — until they exist the gate fails on that row.
        inventory, roots = self._inventory_and_roots(tmp_path)
        inventory["holm_row"]["satisfied_by"] = None
        out = closure.evaluate_closure(
            inventory,
            profile_bytes=make_closure_profile_bytes(),
            qa012_roots=roots,
        )
        assert out["satisfied"] is False
        assert any("holm" in str(row).lower() for row in out["failing_rows"])

    def test_holm_row_wrong_provenance_token_fails(self, tmp_path):
        inventory, roots = self._inventory_and_roots(tmp_path)
        inventory["holm_row"]["satisfied_by"] = "historical_2025_analysis"
        out = closure.evaluate_closure(
            inventory,
            profile_bytes=make_closure_profile_bytes(),
            qa012_roots=roots,
        )
        assert out["satisfied"] is False
        assert any("holm/inference" in row for row in out["failing_rows"])

    def test_main_tex_only_closure_binding_fails(self, tmp_path):
        # Closure binds the COMPLETE checksum manifest, never main.tex alone.
        inventory, roots = self._inventory_and_roots(tmp_path)
        inventory["d6_baseline"]["final_checksums_entries"] = {
            "main.tex": D6_MAIN_TEX_SHA256
        }
        out = closure.evaluate_closure(
            inventory,
            profile_bytes=make_closure_profile_bytes(),
            qa012_roots=roots,
        )
        assert out["satisfied"] is False
        assert any("entry map" in row for row in out["failing_rows"])

    def test_complete_checksum_map_matches_pinned_authority(self, tmp_path):
        inventory, roots = self._inventory_and_roots(tmp_path)
        out = closure.evaluate_closure(
            inventory,
            profile_bytes=make_closure_profile_bytes(),
            qa012_roots=roots,
        )
        assert out["satisfied"] is True
        assert out["failing_rows"] == []

    def test_raw_checksum_manifest_cannot_self_authorize(self, tmp_path):
        inventory, roots = self._inventory_and_roots(tmp_path)
        inventory["d6_baseline"]["final_checksums_sha256"] = "f" * 64
        out = closure.evaluate_closure(
            inventory,
            profile_bytes=make_closure_profile_bytes(),
            qa012_roots=roots,
        )
        assert out["satisfied"] is False
        assert any("pinned" in row for row in out["failing_rows"])

    def test_truncated_checksum_map_cannot_self_authorize(self, tmp_path):
        inventory, roots = self._inventory_and_roots(tmp_path)
        entries = inventory["d6_baseline"]["final_checksums_entries"]
        del entries["references.bib"]
        inventory["d6_baseline"]["final_checksums_entries_sha256"] = (
            closure.checksum_entries_sha256(entries)
        )
        out = closure.evaluate_closure(
            inventory,
            profile_bytes=make_closure_profile_bytes(),
            qa012_roots=roots,
        )
        assert out["satisfied"] is False
        assert any("pinned" in row for row in out["failing_rows"])

    @pytest.mark.parametrize("mutation", ["missing", "unexpected"])
    def test_expected_claim_row_set_is_exact(self, mutation):
        inventory = make_closure_inventory()
        if mutation == "missing":
            inventory["rows"] = inventory["rows"][:1]
        else:
            inventory["rows"].append(
                {"item": "arbitrary", "status": "EXTERNAL", "evidence": "x"}
            )
        with pytest.raises(schema.SchemaValidationError, match="row set"):
            closure.evaluate_closure(inventory)

    def test_qa012_unverified_blocks_closure(self):
        # R-072 is blocking for closure: the fail-closed default state.
        inventory = make_closure_inventory(
            qa012={"status": "UNVERIFIED", "inventory_sha256": None}
        )
        out = closure.evaluate_closure(
            inventory, profile_bytes=make_closure_profile_bytes()
        )
        assert out["satisfied"] is False
        assert any(
            "qa" in str(row).lower() or "012" in str(row)
            for row in out["failing_rows"]
        )

    def test_d6_hash_drift_fails(self, tmp_path):
        inventory, roots = self._inventory_and_roots(tmp_path)
        inventory["d6_baseline"]["main_tex_sha256"] = FAKE_SHA_A
        out = closure.evaluate_closure(
            inventory,
            profile_bytes=make_closure_profile_bytes(),
            qa012_roots=roots,
        )
        assert out["satisfied"] is False
        assert any("main.tex hash drifted" in row for row in out["failing_rows"])

    def test_unsatisfied_display_row_fails(self, tmp_path):
        # Handoff L526: every displayed number maps to clean bound evidence
        # or is removed/downgraded — an UNSATISFIED row blocks closure.
        inventory, roots = self._inventory_and_roots(tmp_path)
        inventory["rows"][0]["status"] = "UNSATISFIED"
        inventory["rows"][0]["evidence"] = None
        out = closure.evaluate_closure(
            inventory,
            profile_bytes=make_closure_profile_bytes(),
            qa012_roots=roots,
        )
        assert out["satisfied"] is False
        assert any("displayed number" in row for row in out["failing_rows"])

    def test_external_rows_stay_external_and_do_not_block(self, tmp_path):
        inventory, roots = self._inventory_and_roots(tmp_path)
        assert any(r["status"] == "EXTERNAL" for r in inventory["rows"])
        out = closure.evaluate_closure(
            inventory,
            profile_bytes=make_closure_profile_bytes(),
            qa012_roots=roots,
        )
        assert out["satisfied"] is True

    def test_missing_profile_bytes_blocks_closure(self):
        out = closure.evaluate_closure(make_closure_inventory())
        assert out["satisfied"] is False
        assert any("actual bound profile" in row for row in out["failing_rows"])

    def test_missing_qa012_roots_blocks_closure(self):
        out = closure.evaluate_closure(
            make_closure_inventory(),
            profile_bytes=make_closure_profile_bytes(),
        )
        assert out["satisfied"] is False
        assert any("actual five corpus roots" in row for row in out["failing_rows"])

    def test_qa012_root_manifest_drift_blocks_closure(self, tmp_path):
        inventory, roots = self._inventory_and_roots(tmp_path)
        (roots[qa012.REQUIRED_SCOPE_PRONGS[0]] / "added.json").write_text(
            '{}', "utf-8"
        )
        out = closure.evaluate_closure(
            inventory,
            profile_bytes=make_closure_profile_bytes(),
            qa012_roots=roots,
        )
        assert out["satisfied"] is False
        assert any("fresh scan" in row for row in out["failing_rows"])

    def test_profile_byte_drift_blocks_closure(self):
        out = closure.evaluate_closure(
            make_closure_inventory(),
            profile_bytes=make_closure_profile_bytes() + b" ",
        )
        assert out["satisfied"] is False
        assert any("does not bind" in row for row in out["failing_rows"])

    @pytest.mark.parametrize(
        "profile_bytes",
        [
            b'{"profile_id":"x","profile_id":"y"}',
            b'{"value":NaN}',
        ],
    )
    def test_profile_binding_uses_strict_json_ingress(self, profile_bytes):
        out = closure.evaluate_closure(
            make_closure_inventory(), profile_bytes=profile_bytes
        )
        assert out["satisfied"] is False
        assert any("invalid" in row for row in out["failing_rows"])


# ---------------------------------------------------------------------------
# R-072: QA-012 detector + inventory manifest
# ---------------------------------------------------------------------------


class TestQa012Detector:
    def test_nested_hit_exact_json_pointer(self):
        doc = {"a": {"b": {"format": "QA"}}}
        assert qa012.detect_format_qa(doc) == ["/a/b/format"]

    def test_array_embedded_hit_exact_json_pointer(self):
        doc = {"rows": [{"x": 1}, {"format": "QA"}]}
        assert qa012.detect_format_qa(doc) == ["/rows/1/format"]

    def test_top_level_hit(self):
        assert qa012.detect_format_qa({"format": "QA"}) == ["/format"]

    def test_value_must_be_exactly_qa(self):
        assert qa012.detect_format_qa({"format": "qa"}) == []
        assert qa012.detect_format_qa({"format": "QA "}) == []
        assert qa012.detect_format_qa({"format": "MC"}) == []

    def test_key_must_be_exactly_format(self):
        assert qa012.detect_format_qa({"Format": "QA"}) == []
        assert qa012.detect_format_qa({"response_format": "QA"}) == []

    def test_multiple_hits_all_reported(self):
        doc = {
            "format": "QA",
            "nested": {"format": "QA"},
            "rows": [{"format": "QA"}],
        }
        pointers = qa012.detect_format_qa(doc)
        assert sorted(pointers) == [
            "/format",
            "/nested/format",
            "/rows/0/format",
        ]

    def test_rfc6901_escaping_in_pointers(self):
        doc = {"x/y": {"format": "QA"}, "a~b": {"format": "QA"}}
        pointers = qa012.detect_format_qa(doc)
        assert sorted(pointers) == ["/a~0b/format", "/x~1y/format"]

    def test_non_string_qa_value_is_not_a_hit(self):
        assert qa012.detect_format_qa({"format": ["QA"]}) == []
        assert qa012.detect_format_qa({"format": {"value": "QA"}}) == []

    def test_detector_bounds_hits_during_traversal(self):
        with pytest.raises(schema.TypedIngressError, match="hit limit"):
            qa012.detect_format_qa(
                [{"format": "QA"}, {"format": "QA"}], max_hits=1
            )


class TestQa012Inventory:
    def _write_corpus(self, tmp_path, with_hit: bool):
        (tmp_path / "clean.json").write_text(
            json.dumps({"schema_version": 2, "kind": "clean"}),
            encoding="utf-8",
        )
        (tmp_path / "lines.jsonl").write_text(
            json.dumps({"item_key": "itm-0001"})
            + "\n"
            + json.dumps({"item_key": "itm-0002"})
            + "\n",
            encoding="utf-8",
        )
        if with_hit:
            (tmp_path / "historical.json").write_text(
                json.dumps({"meta": {"format": "QA"}}), encoding="utf-8"
            )
        return tmp_path

    def _scope_roots(self, tmp_path, *, with_hit: bool):
        roots = {}
        for index, prong in enumerate(qa012.REQUIRED_SCOPE_PRONGS):
            root = tmp_path / prong
            root.mkdir(parents=True)
            roots[prong] = self._write_corpus(
                root, with_hit=with_hit and index == 0
            )
        return roots

    def test_zero_hit_manifest_recorded_vacuous_with_inventory_hash(
        self, tmp_path
    ):
        roots = self._scope_roots(tmp_path, with_hit=False)
        manifest = qa012.build_inventory_manifest(roots)
        assert manifest["result"] == "zero_hit"
        assert manifest["inventory_sha256"]
        assert len(manifest["files"]) == 2 * len(qa012.REQUIRED_SCOPE_PRONGS)
        assert [entry["name"] for entry in manifest["scope_prongs"]] == list(
            qa012.REQUIRED_SCOPE_PRONGS
        )
        for entry in manifest["files"]:
            assert set(entry) >= {
                "scope_prong",
                "path",
                "size",
                "content_hash",
                "sha256",
                "hits",
            }
            assert entry["hits"] == []
            assert entry["sha256"] == sha256_bytes(
                (roots[entry["scope_prong"]] / entry["path"]).read_bytes()
            )

    def test_hit_manifest_carries_exact_pointers_and_bytes_hash(
        self, tmp_path
    ):
        roots = self._scope_roots(tmp_path, with_hit=True)
        manifest = qa012.build_inventory_manifest(roots)
        assert manifest["result"] == "hits"
        hits = [e for e in manifest["files"] if e["hits"]]
        assert len(hits) == 1
        assert hits[0]["path"] == "historical.json"
        assert hits[0]["hits"] == [{"line": None, "pointer": "/meta/format"}]

    def test_manifest_hash_is_deterministic(self, tmp_path):
        roots = self._scope_roots(tmp_path, with_hit=False)
        a = qa012.build_inventory_manifest(roots)
        b = qa012.build_inventory_manifest(roots)
        assert a["inventory_sha256"] == b["inventory_sha256"]

    def test_strict_parse_failure_is_typed_error(self, tmp_path):
        roots = self._scope_roots(tmp_path, with_hit=False)
        root = roots[qa012.REQUIRED_SCOPE_PRONGS[0]]
        (root / "broken.json").write_text("{not json", encoding="utf-8")
        with pytest.raises(schema.TypedIngressError):
            qa012.build_inventory_manifest(roots)

    @pytest.mark.parametrize("shape", ["missing", "empty", "non_json"])
    def test_vacuous_or_missing_root_escalates_incomplete_scope(
        self, tmp_path, shape
    ):
        roots = self._scope_roots(tmp_path / "base", with_hit=False)
        root = tmp_path / shape
        if shape != "missing":
            root.mkdir()
        if shape == "non_json":
            (root / "note.txt").write_text("not inventory scope")
        roots[qa012.REQUIRED_SCOPE_PRONGS[0]] = root
        manifest = qa012.build_inventory_manifest(roots)
        assert manifest["result"] == "incomplete_scope"
        assert manifest["scope_prongs"][0]["status"].endswith("ESCALATE")

    def test_jsonl_lines_are_scanned_too(self, tmp_path):
        roots = self._scope_roots(tmp_path, with_hit=False)
        root = roots[qa012.REQUIRED_SCOPE_PRONGS[0]]
        (root / "hist.jsonl").write_text(
            json.dumps({"format": "QA"}) + "\n", encoding="utf-8"
        )
        manifest = qa012.build_inventory_manifest(roots)
        assert manifest["result"] == "hits"
        hit = next(entry for entry in manifest["files"] if entry["hits"])
        assert hit["hits"] == [{"line": 1, "pointer": "/format"}]

    def test_duplicate_or_overlapping_roots_are_rejected(self, tmp_path):
        roots = self._scope_roots(tmp_path, with_hit=False)
        roots[qa012.REQUIRED_SCOPE_PRONGS[1]] = roots[
            qa012.REQUIRED_SCOPE_PRONGS[0]
        ]
        with pytest.raises(schema.ConfigSurfaceError, match="overlap"):
            qa012.build_inventory_manifest(roots)

    @pytest.mark.parametrize(
        "mutation",
        [
            "missing_prong",
            "reordered_prongs",
            "file_count",
            "duplicate_file",
            "unsafe_path",
            "bad_hash",
            "line_zero",
            "file_order",
            "result",
            "digest",
        ],
    )
    def test_manifest_mutations_fail_closed(self, tmp_path, mutation):
        manifest = qa012.build_inventory_manifest(
            self._scope_roots(tmp_path, with_hit=False)
        )
        doc = copy.deepcopy(manifest)
        if mutation == "missing_prong":
            doc["scope_prongs"].pop()
        elif mutation == "reordered_prongs":
            doc["scope_prongs"].reverse()
        elif mutation == "file_count":
            doc["scope_prongs"][0]["file_count"] += 1
        elif mutation == "duplicate_file":
            doc["files"].append(copy.deepcopy(doc["files"][0]))
        elif mutation == "unsafe_path":
            doc["files"][0]["path"] = "../escape.json"
        elif mutation == "bad_hash":
            doc["files"][0]["content_hash"] = "bad"
        elif mutation == "line_zero":
            entry = next(e for e in doc["files"] if e["path"].endswith(".jsonl"))
            entry["hits"] = [{"line": 0, "pointer": "/format"}]
        elif mutation == "file_order":
            doc["files"].reverse()
        elif mutation == "result":
            doc["result"] = "hits"
        else:
            doc["inventory_sha256"] = "0" * 64
        with pytest.raises(schema.SchemaValidationError):
            qa012.validate_inventory_manifest(doc)

    @pytest.mark.parametrize(
        "unsafe_path",
        ["C:relative.json", "dir\\file.json", "/absolute.json"],
    )
    def test_manifest_rejects_nonportable_relative_paths(
        self, tmp_path, unsafe_path
    ):
        manifest = qa012.build_inventory_manifest(
            self._scope_roots(tmp_path, with_hit=False)
        )
        manifest["files"][0]["path"] = unsafe_path
        with pytest.raises(schema.SchemaValidationError, match="safe JSON"):
            qa012.validate_inventory_manifest(manifest)

    def test_manifest_validation_enforces_resource_limits(
        self, tmp_path, monkeypatch
    ):
        manifest = qa012.build_inventory_manifest(
            self._scope_roots(tmp_path, with_hit=False)
        )
        monkeypatch.setattr(qa012, "MAX_QA_FILES", 1)
        with pytest.raises(schema.SchemaValidationError, match="file-count"):
            qa012.validate_inventory_manifest(manifest)

    def test_resource_limits_are_enforced(self, tmp_path, monkeypatch):
        roots = self._scope_roots(tmp_path, with_hit=False)
        monkeypatch.setattr(qa012, "MAX_QA_FILES", 1)
        with pytest.raises(schema.TypedIngressError, match="file-count"):
            qa012.build_inventory_manifest(roots)

    def test_jsonl_row_limit_is_enforced(self, tmp_path, monkeypatch):
        roots = self._scope_roots(tmp_path, with_hit=False)
        monkeypatch.setattr(qa012, "MAX_QA_JSONL_ROWS", 1)
        with pytest.raises(schema.TypedIngressError, match="row limit"):
            qa012.build_inventory_manifest(roots)

    def test_membership_change_during_scan_is_rejected(self, tmp_path, monkeypatch):
        roots = self._scope_roots(tmp_path, with_hit=False)
        original = qa012._scan_file
        mutated = False

        def mutate_after_scan(path, root, scope_prong):
            nonlocal mutated
            entry = original(path, root, scope_prong)
            if not mutated:
                (root / "late.json").write_text('{"format":"QA"}', "utf-8")
                mutated = True
            return entry

        monkeypatch.setattr(qa012, "_scan_file", mutate_after_scan)
        with pytest.raises(schema.TypedIngressError, match="membership changed"):
            qa012.build_inventory_manifest(roots)

    def test_hit_bytes_cannot_substitute_for_semantic_block(self):
        # Any hit document is a compatibility fixture that must NOT satisfy
        # the strict v2 semantic layer.
        hit_doc = {"schema_version": 2, "format": "QA"}
        with pytest.raises(
            (schema.SchemaValidationError, schema.TypedIngressError)
        ):
            schema.validate_profile(hit_doc)


# ---------------------------------------------------------------------------
# R-038: seven-target documentation audit
# ---------------------------------------------------------------------------

DOC_TARGETS = (
    "README.md",
    "DATA.md",
    "ARTIFACTS.md",
    "docs/CLAIM_SURFACE.md",
    "docs/stopdff-learned-value-fair-qa.md",
    "docs/stopdff_v5/REPRODUCTION.md",
    "reproducibility/source_to_claim.md",
)


class TestDocAudit:
    def test_namespace_readme_pins_the_cli_contract(self):
        readme = REPO_ROOT / "reproducibility" / "colm_aims_2026" / "README.md"
        assert readme.is_file(), "namespace README.md missing (R-038)"
        text = readme.read_text("utf-8")
        assert "python -m reproducibility.colm_aims_2026.verify" in text
        assert "--mode source" in text and "--mode release" in text
        assert "PASS_SOURCE_ONLY" in text and "PASS_RELEASE" in text
        for code in ("0", "1", "2", "3", "4"):
            assert code in text
        assert "expectations" in text.lower()
        assert "receipt" in text.lower()
        assert "verify_audit_release.py" in text  # one-line disambiguation

    @pytest.mark.parametrize("target", DOC_TARGETS)
    def test_target_carries_constructed_reference_qualification(self, target):
        path = REPO_ROOT / target
        assert path.is_file(), f"doc target missing: {target}"
        text = path.read_text("utf-8").lower()
        assert "constructed" in text and "reference" in text, (
            f"{target} lacks constructed-reference qualification (R-038)"
        )

    def test_source_to_claim_carries_historical_scope_header(self):
        text = (REPO_ROOT / "reproducibility" / "source_to_claim.md").read_text(
            "utf-8"
        )
        head = text[:2000].lower()
        assert "historical" in head, (
            "source_to_claim.md needs a historical-scope header naming the"
            " manuscript it maps (R-038)"
        )
        assert "ledger" in text.lower()

    def test_no_doc_redefines_legacy_verifier_as_camera_ready(self):
        readme = REPO_ROOT / "reproducibility" / "colm_aims_2026" / "README.md"
        assert readme.is_file()
        text = readme.read_text("utf-8").lower()
        assert "camera-ready certification" not in text
