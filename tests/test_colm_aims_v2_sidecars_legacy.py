"""Legacy-sidecar boundary + legacy families: R-064 (ASK-2(a) pinned type
matrix, BOTH modes, end-to-end), R-014 (legacy families from captured bytes,
certifiability, no-mutation), R-060 legacy-loader naming (OQ-V2-002).

GREENFIELD RED: fails at collection until the namespace exists.
"""
from __future__ import annotations

import json

import pytest

from reproducibility.colm_aims_2026 import legacy, schema, verifier

from tests._colm_aims_v2_helpers import (
    REPO_ROOT,
    SIDECAR_LEG_PREFIX,
    VERDICT_SOURCE_PASS,
    assert_failing_leg_prefix,
    build_package_v2,
    colm_no_network,  # noqa: F401 - autouse fixture
    release_report,
    run_verifier_on,
    source_report,
    tree_hashes,
)

SIDECAR_REL = "sidecars/legacy_note.json"

# The sign-off SS5 matrix: top-level JSON array / string / number / Boolean /
# null are ingress-DEFECTIVE for this evidence-tree namespace.
NON_OBJECT_MATRIX = [
    ("array", b"[]\n"),
    ("string", b'"string"\n'),
    ("number", b"0\n"),
    ("boolean", b"true\n"),
    ("null", b"null\n"),
]


def _package_with_sidecar(tmp_path, blob: bytes):
    return build_package_v2(
        tmp_path,
        extra_tree_files={SIDECAR_REL: blob},
        extra_manifest_allowlist=(SIDECAR_REL,),
        extra_rights_paths=(SIDECAR_REL,),
    )


# ---------------------------------------------------------------------------
# R-064: parameterized non-object matrix, BOTH modes, end-to-end
# ---------------------------------------------------------------------------


class TestNonObjectSidecarMatrix:
    @pytest.mark.parametrize(("type_name", "blob"), NON_OBJECT_MATRIX)
    def test_source_mode_typed_error_names_file_and_type(
        self, tmp_path, type_name, blob
    ):
        pkg = _package_with_sidecar(tmp_path, blob)
        with pytest.raises(schema.ColmAimsError) as excinfo:
            source_report(pkg)
        message = str(excinfo.value)
        assert SIDECAR_REL in message
        assert type_name in message.lower()

    @pytest.mark.parametrize(("type_name", "blob"), NON_OBJECT_MATRIX)
    def test_release_mode_mandatory_failing_leg_names_file_and_type(
        self, tmp_path, type_name, blob
    ):
        pkg = _package_with_sidecar(tmp_path, blob)
        report = release_report(pkg)
        legs = assert_failing_leg_prefix(
            report, SIDECAR_LEG_PREFIX + SIDECAR_REL
        )
        observed = json.dumps(legs)
        assert type_name in observed.lower()

    def test_nearest_true_unknown_object_tolerated_source(self, tmp_path):
        # Accept-and-pin control: a well-formed unknown-family OBJECT stays
        # a tolerated historical sidecar.
        blob = json.dumps({"mystery_family": True, "n": 3}).encode("utf-8")
        pkg = _package_with_sidecar(tmp_path, blob)
        report = source_report(pkg)
        assert report.verdict == VERDICT_SOURCE_PASS

    def test_nearest_true_unknown_object_tolerated_release(self, tmp_path):
        blob = json.dumps({"mystery_family": True, "n": 3}).encode("utf-8")
        pkg = _package_with_sidecar(tmp_path, blob)
        report = release_report(pkg)
        assert report.verdict == "PASS_RELEASE"

    def test_malformed_sidecar_json_is_ingress_defect(self, tmp_path):
        pkg = _package_with_sidecar(tmp_path, b'{"unterminated": \n')
        with pytest.raises(schema.TypedIngressError):
            source_report(pkg)

    def test_invalid_utf8_sidecar_is_ingress_defect(self, tmp_path):
        pkg = _package_with_sidecar(tmp_path, b'{"k": "\xff\xfe"}\n')
        with pytest.raises(schema.TypedIngressError):
            source_report(pkg)

    def test_nonfinite_sidecar_is_ingress_defect(self, tmp_path):
        pkg = _package_with_sidecar(tmp_path, b'{"v": Infinity}\n')
        with pytest.raises(schema.TypedIngressError):
            source_report(pkg)

    def test_overlong_int_sidecar_is_ingress_defect(self, tmp_path):
        token = ("9" * 150).encode("utf-8")
        pkg = _package_with_sidecar(tmp_path, b'{"v": ' + token + b"}\n")
        with pytest.raises(schema.TypedIngressError):
            source_report(pkg)

    def test_release_run_with_defective_sidecar_still_evaluates_other_legs(
        self, tmp_path
    ):
        # Collect-don't-halt (R-012): the failing sidecar leg coexists with
        # the other evaluated legs in one report.
        pkg = _package_with_sidecar(tmp_path, b"[]\n")
        report = release_report(pkg)
        assert len(report.legs) > 1


# ---------------------------------------------------------------------------
# R-014: enumerated legacy families against CAPTURED bytes
# ---------------------------------------------------------------------------

CAPTURED = {
    "csli": REPO_ROOT / "paper_exports" / "csli.json",
    "calibration": REPO_ROOT / "paper_exports" / "calibration.json",
    "audit_card": REPO_ROOT / "paper_exports" / "audit_card.json",
}


class TestLegacyFamilies:
    @pytest.mark.parametrize("family", sorted(CAPTURED))
    def test_captured_family_bytes_parse_and_classify(self, family):
        # Captured bytes from the repo's actual historical artifacts —
        # never reconstructions (R-014).
        blob = CAPTURED[family].read_bytes()
        out = legacy.load_legacy_v1_document(blob)
        assert out["legacy_family"] == family
        assert out["aggregate_only"] is True
        assert out["certifying"] is False

    @pytest.mark.parametrize("family", sorted(CAPTURED))
    def test_aggregate_only_cannot_certify_per_item_claims(self, family):
        blob = CAPTURED[family].read_bytes()
        out = legacy.load_legacy_v1_document(blob)
        assert legacy.legacy_certifies(out, "per_item_paired") is False
        assert legacy.legacy_certifies(out, "aggregate") is True

    def test_unknown_family_object_refused_by_the_loader(self):
        # The LOADER refuses unknown families; the VERIFIER tolerates them
        # as unknown sidecars (R-064) — two different duties.
        blob = json.dumps({"mystery_family": True}).encode("utf-8")
        with pytest.raises(schema.SchemaValidationError):
            legacy.load_legacy_v1_document(blob)

    def test_nonfinite_bytes_refused_even_inside_known_family(self):
        blob = b'{"panel_csli": {"x": NaN}, "metadata": {}}\n'
        with pytest.raises(schema.TypedIngressError):
            legacy.load_legacy_v1_document(blob)


# ---------------------------------------------------------------------------
# R-014: certifiability classification + verifier never mutates inputs
# ---------------------------------------------------------------------------


class TestCertifiability:
    def test_dirty_closure_classifies_historical_noncertifying(self):
        profile = {"provenance": {"dirty_state": {"git_dirty": True}}}
        assert (
            verifier.classify_certifiability(profile)
            == "HISTORICAL_NONCERTIFYING"
        )

    def test_superseded_closure_classifies_historical_noncertifying(
        self, tmp_path
    ):
        def mutate(prov):
            prov["superseded_by_producer_sha256"] = "9" * 64

        pkg = build_package_v2(tmp_path, binding_mutator=mutate)
        report = release_report(pkg)
        assert report.verdict != "PASS_RELEASE"

    def test_unresolved_calibration_identity_noncertifying(self, tmp_path):
        def mutate(prov):
            prov["calibration_identity"] = {
                "shared": "UNRESOLVED",
                "format_specific": "cal-fmt-0001",
            }

        pkg = build_package_v2(tmp_path, binding_mutator=mutate)
        report = release_report(pkg)
        assert report.verdict != "PASS_RELEASE"

    def test_verifier_runs_never_mutate_inputs(self, tmp_path):
        pkg = build_package_v2(tmp_path)
        before = tree_hashes(pkg.tree)
        before_docs = {
            p.name: p.read_bytes()
            for p in (pkg.ledger_path, pkg.rights_path, pkg.expectations_path)
        }
        run_verifier_on(pkg, "source")
        run_verifier_on(pkg, "release")
        assert tree_hashes(pkg.tree) == before
        for p in (pkg.ledger_path, pkg.rights_path, pkg.expectations_path):
            assert p.read_bytes() == before_docs[p.name]


# ---------------------------------------------------------------------------
# Mini-audit round 1 fix (MA2-003): v1_profile must not inherit
# aggregate-certify privilege (R-014/R-060)
# ---------------------------------------------------------------------------


V1_PROFILE_BLOB = json.dumps(
    {"schema_version": 1, "profile_id": "colm_aims_constructed_reference_v1"}
).encode("utf-8")


class TestV1ProfileCertifyPrivilege:
    def test_v1_profile_family_recognized_but_never_certifies(self):
        # MA2-003 exploit: v1_profile joined the enumerated legacy set and
        # silently inherited the aggregate-certify privilege reserved for
        # the three captured paper_exports families.
        out = legacy.load_legacy_v1_document(V1_PROFILE_BLOB)
        assert out["legacy_family"] == "v1_profile"
        assert out["certifying"] is False
        assert legacy.legacy_certifies(out, "aggregate") is False
        assert legacy.legacy_certifies(out, "per_item_paired") is False

    def test_captured_families_keep_aggregate_privilege(self):
        # Nearest-true control: the three captured families keep exactly
        # the aggregate privilege they had.
        blob = CAPTURED["csli"].read_bytes()
        out = legacy.load_legacy_v1_document(blob)
        assert legacy.legacy_certifies(out, "aggregate") is True
