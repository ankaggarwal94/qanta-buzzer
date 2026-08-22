"""Suite-evidence receipts, CAMERA_READY_CLOSURE, QA-012, doc audit:
R-070 (A'-F4 machine-readable receipt), R-071 (closure gate distinct from
PASS_RELEASE, frozen D6 inventory, Holm-row blocking), R-072 (QA-012
format:"QA" detector + inventory manifest), R-038 (seven-target doc audit).

GREENFIELD RED: fails at collection until the namespace exists.
"""
from __future__ import annotations

import json

import pytest

from reproducibility.colm_aims_2026 import closure, qa012, receipt as receipt_mod
from reproducibility.colm_aims_2026 import schema

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
    make_suite_receipt,
    sha256_bytes,
)


# ---------------------------------------------------------------------------
# R-070: machine-readable suite receipt (all bindings are hashes)
# ---------------------------------------------------------------------------

SUITE_RECEIPT_REQUIRED_FIELDS = (
    "environment_digest",
    "workflow_file_sha256",
    "interpreter_realpath",
    "commit",
    "tree",
    "dirty",
    "command",
    "exit_code",
    "junit_report_sha256",
    "skip_identities",
    "artifact_hashes",
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
        # The A'-F4 defect shape verbatim: environment_digest as a metadata
        # OBJECT instead of a lockfile/environment-export HASH.
        doc = make_suite_receipt(
            environment_digest={
                "python": "3.11.15",
                "numpy": "2.4.6",
                "platform": "darwin",
            }
        )
        with pytest.raises(schema.ColmAimsError) as excinfo:
            receipt_mod.validate_suite_receipt(doc)
        assert "environment_digest" in str(excinfo.value)

    def test_environment_digest_must_be_64_hex(self):
        doc = make_suite_receipt(environment_digest="not-a-hash")
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
            receipt_mod.validate_suite_receipt(make_suite_receipt(tree="HEAD"))

    def test_skip_identities_must_be_a_list(self):
        doc = make_suite_receipt(skip_identities="none")
        with pytest.raises(schema.ColmAimsError):
            receipt_mod.validate_suite_receipt(doc)

    def test_dirty_must_be_bool(self):
        doc = make_suite_receipt(dirty="false")
        with pytest.raises(schema.ColmAimsError):
            receipt_mod.validate_suite_receipt(doc)

    def test_artifact_hashes_values_must_be_hashes(self):
        doc = make_suite_receipt(
            artifact_hashes={"results/suite.xml": {"size": 12}}
        )
        with pytest.raises(schema.ColmAimsError):
            receipt_mod.validate_suite_receipt(doc)


# ---------------------------------------------------------------------------
# R-071: CAMERA_READY_CLOSURE gate
# ---------------------------------------------------------------------------


class TestClosureGate:
    def test_gate_token_distinct_from_release_pass(self):
        assert closure.CAMERA_READY_CLOSURE == CLOSURE_GATE_TOKEN
        assert closure.CAMERA_READY_CLOSURE != VERDICT_RELEASE_PASS

    def test_d6_baseline_constants_pinned(self):
        assert closure.D6_MAIN_TEX_SHA256 == D6_MAIN_TEX_SHA256
        assert closure.D6_MAIN_PDF_SHA256 == D6_MAIN_PDF_SHA256

    def test_fully_satisfied_inventory_passes(self):
        inventory = make_closure_inventory()
        # The Holm row is satisfiable ONLY by the D7(b) discriminator token.
        assert inventory["holm_row"]["satisfied_by"] == ANALYSIS_PROVENANCE_D7B
        out = closure.evaluate_closure(inventory)
        assert out["gate"] == CLOSURE_GATE_TOKEN
        assert out["satisfied"] is True

    def test_satisfied_except_holm_fails(self):
        # The Holm/inference row is satisfied ONLY by the D7(b) regenerated
        # outputs — until they exist the gate fails on that row.
        inventory = make_closure_inventory(holm_row={"satisfied_by": None})
        out = closure.evaluate_closure(inventory)
        assert out["satisfied"] is False
        assert any("holm" in str(row).lower() for row in out["failing_rows"])

    def test_holm_row_wrong_provenance_token_fails(self):
        inventory = make_closure_inventory(
            holm_row={"satisfied_by": "historical_2025_analysis"}
        )
        out = closure.evaluate_closure(inventory)
        assert out["satisfied"] is False

    def test_main_tex_only_closure_binding_fails(self):
        # Closure binds the COMPLETE checksum manifest, never main.tex alone.
        inventory = make_closure_inventory()
        inventory["d6_baseline"]["final_checksums_entries"] = {
            "main.tex": D6_MAIN_TEX_SHA256
        }
        out = closure.evaluate_closure(inventory)
        assert out["satisfied"] is False

    def test_qa012_unverified_blocks_closure(self):
        # R-072 is blocking for closure: the fail-closed default state.
        inventory = make_closure_inventory(
            qa012={"status": "UNVERIFIED", "inventory_sha256": None}
        )
        out = closure.evaluate_closure(inventory)
        assert out["satisfied"] is False
        assert any(
            "qa" in str(row).lower() or "012" in str(row)
            for row in out["failing_rows"]
        )

    def test_d6_hash_drift_fails(self):
        inventory = make_closure_inventory()
        inventory["d6_baseline"]["main_tex_sha256"] = FAKE_SHA_A
        out = closure.evaluate_closure(inventory)
        assert out["satisfied"] is False

    def test_unsatisfied_display_row_fails(self):
        # Handoff L526: every displayed number maps to clean bound evidence
        # or is removed/downgraded — an UNSATISFIED row blocks closure.
        inventory = make_closure_inventory()
        inventory["rows"].append(
            {
                "item": "figure-3-hazard-panel",
                "status": "UNSATISFIED",
                "evidence": None,
            }
        )
        out = closure.evaluate_closure(inventory)
        assert out["satisfied"] is False

    def test_external_rows_stay_external_and_do_not_block(self):
        inventory = make_closure_inventory()
        assert any(r["status"] == "EXTERNAL" for r in inventory["rows"])
        out = closure.evaluate_closure(inventory)
        assert out["satisfied"] is True


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

    def test_zero_hit_manifest_recorded_vacuous_with_inventory_hash(
        self, tmp_path
    ):
        root = self._write_corpus(tmp_path, with_hit=False)
        manifest = qa012.build_inventory_manifest([root])
        assert manifest["result"] == "zero_hit"
        assert manifest["inventory_sha256"]
        assert len(manifest["files"]) == 2
        for entry in manifest["files"]:
            assert set(entry) >= {"path", "size", "sha256", "hits"}
            assert entry["hits"] == []
            assert entry["sha256"] == sha256_bytes(
                (root / entry["path"]).read_bytes()
            )

    def test_hit_manifest_carries_exact_pointers_and_bytes_hash(
        self, tmp_path
    ):
        root = self._write_corpus(tmp_path, with_hit=True)
        manifest = qa012.build_inventory_manifest([root])
        assert manifest["result"] == "hits"
        hits = [e for e in manifest["files"] if e["hits"]]
        assert len(hits) == 1
        assert hits[0]["path"] == "historical.json"
        assert hits[0]["hits"] == ["/meta/format"]

    def test_manifest_hash_is_deterministic(self, tmp_path):
        root = self._write_corpus(tmp_path, with_hit=False)
        a = qa012.build_inventory_manifest([root])
        b = qa012.build_inventory_manifest([root])
        assert a["inventory_sha256"] == b["inventory_sha256"]

    def test_strict_parse_failure_is_typed_error(self, tmp_path):
        root = self._write_corpus(tmp_path, with_hit=False)
        (root / "broken.json").write_text("{not json", encoding="utf-8")
        with pytest.raises(schema.TypedIngressError):
            qa012.build_inventory_manifest([root])

    def test_jsonl_lines_are_scanned_too(self, tmp_path):
        root = self._write_corpus(tmp_path, with_hit=False)
        (root / "hist.jsonl").write_text(
            json.dumps({"format": "QA"}) + "\n", encoding="utf-8"
        )
        manifest = qa012.build_inventory_manifest([root])
        assert manifest["result"] == "hits"

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
