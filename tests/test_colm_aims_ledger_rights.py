"""RED suite — claim ledger, EXTERNAL immunity, Random-K gate, rights.

Covers: R-023, R-024, R-025, R-026, R-030.
Spec: .correctless/specs/camera-ready-aims-evidence.md
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from reproducibility.colm_aims_2026 import ledger as ledger_mod
from reproducibility.colm_aims_2026 import pairing, render, schema, verifier
from tests._colm_aims_helpers import (  # noqa: F401  (colm_no_network is an autouse fixture)
    MANUSCRIPT_PDF_SHA256,
    SENTINEL,
    EXIT_GATE_FAIL,
    build_package,
    cli_args_for,
    colm_no_network,
    make_cell,
    make_external_row,
    make_ledger,
    make_ledger_row,
    make_profile,
    make_rights,
    repo_head_commit,
    run_cli,
    sha256_file,
    standard_records,
)


def _ledger(**kwargs):
    return make_ledger(source_commit=repo_head_commit(), **kwargs)


# ---------------------------------------------------------------------------
# R-023: row completeness + closed status enum + provenance classes
# ---------------------------------------------------------------------------

REQUIRED_ROW_FIELDS = [
    "claim_id",
    "manuscript_location",
    "manuscript_wording",
    "estimand",
    "allowed_scope",
    "producer_entrypoint",
    "dependency_closure",
    "input_identity",
    "split_identity",
    "model_identity",
    "calibration_identity",
    "artifact_id",
    "renderer_id",
    "verifier_oracle",
    "rights_status",
    "status",
    "blocking_task",
    "provenance_class",
]


def test_complete_ledger_validates():
    # Tests R-023 [unit]: the complete fixture ledger validates.
    ledger_mod.validate_ledger(_ledger())


@pytest.mark.parametrize("field", REQUIRED_ROW_FIELDS)
def test_row_missing_required_field_fails(field):
    # Tests R-023 [unit]: each claim-ledger row carries the full pinned field
    # set; a missing field fails, naming the field.
    row = make_ledger_row()
    del row[field]
    doc = _ledger(rows=[row, make_external_row()])
    with pytest.raises(ledger_mod.LedgerValidationError) as exc:
        ledger_mod.validate_ledger(doc)
    assert field in str(exc.value)


@pytest.mark.parametrize("bad_status", ["MAYBE", "pass", "WARN", "OK", ""])
def test_status_outside_closed_enum_fails(bad_status):
    # Tests R-023 [unit]: status is in the closed enum
    # PASS | FAIL | UNVERIFIED | EXTERNAL.
    row = make_ledger_row(status=bad_status)
    with pytest.raises(ledger_mod.LedgerValidationError):
        ledger_mod.validate_ledger(_ledger(rows=[row, make_external_row()]))


@pytest.mark.parametrize(
    "good_class",
    [
        "manuscript_identity",
        "historical_submission_artifact",
        "historical_randomk_v5",
        "current_source",
        "future_evidence",
    ],
)
def test_provenance_classes_are_distinguished(good_class):
    # Tests R-023 [unit]: the ledger distinguishes manuscript identity,
    # historical submission artifacts, historical Random-K/v5 results,
    # current source, and future clean evidence.
    row = make_ledger_row(provenance_class=good_class)
    if good_class == "manuscript_identity":
        row["verifier_oracle"] = "human"
        row["status"] = "EXTERNAL"
    ledger_mod.validate_ledger(_ledger(rows=[row, make_external_row()]))


def test_unknown_provenance_class_fails():
    # Tests R-023 [unit]: provenance classes are a closed set.
    row = make_ledger_row(provenance_class="vibes")
    with pytest.raises(ledger_mod.LedgerValidationError):
        ledger_mod.validate_ledger(_ledger(rows=[row, make_external_row()]))


def test_ledger_pins_manuscript_pdf_sha256():
    # Tests R-023 [unit]: manuscript identity is the submission PDF SHA-256;
    # a ledger without it (or with a non-sha value) fails validation.
    doc = _ledger()
    assert doc["manuscript"]["submission_pdf_sha256"] == MANUSCRIPT_PDF_SHA256
    del doc["manuscript"]["submission_pdf_sha256"]
    with pytest.raises(ledger_mod.LedgerValidationError):
        ledger_mod.validate_ledger(doc)
    doc2 = _ledger()
    doc2["manuscript"]["submission_pdf_sha256"] = "not-a-sha"
    with pytest.raises(ledger_mod.LedgerValidationError):
        ledger_mod.validate_ledger(doc2)


def test_hazard_report_rows_require_manuscript_citation():
    # Tests R-023 [unit]: PR #41 hazard reports may appear only if the exact
    # manuscript cites them.
    row = make_ledger_row(
        claim_id="clm-hzrd-0001",
        artifact_family="pr41_hazard_report",
        artifact_id="hazard_efficacy_report.json",
        manuscript_cites=False,
    )
    with pytest.raises(ledger_mod.LedgerValidationError):
        ledger_mod.validate_ledger(_ledger(rows=[row, make_external_row()]))
    row["manuscript_cites"] = True
    ledger_mod.validate_ledger(_ledger(rows=[row, make_external_row()]))


# ---------------------------------------------------------------------------
# R-024: EXTERNAL rows are immune to repo tooling
# ---------------------------------------------------------------------------


def test_external_rows_byte_identical_across_every_enumerated_tool(tmp_path: Path):
    # Tests R-024 [integration]: every tool run leaves EXTERNAL rows
    # byte-identical. Enumerated tool list: the writer, the checker, the
    # verifier in both modes, and the ledger validator.
    pkg = build_package(tmp_path)
    ledger_before = pkg.ledger_path.read_bytes()

    # 1) the writer
    schema.write_profile(tmp_path / "elsewhere.json", make_profile())
    assert pkg.ledger_path.read_bytes() == ledger_before
    # 2) the checker
    pairing.validate_cell(make_cell(), standard_records())
    assert pkg.ledger_path.read_bytes() == ledger_before
    # 3) the verifier, source mode
    verifier.run_verifier(pkg.tree, mode="source", receipts_dir=pkg.receipts_dir)
    assert pkg.ledger_path.read_bytes() == ledger_before
    # 4) the verifier, release mode
    verifier.run_verifier(
        pkg.tree,
        mode="release",
        receipts_dir=pkg.receipts_dir,
        expectations=pkg.expectations_path,
    )
    assert pkg.ledger_path.read_bytes() == ledger_before
    # 5) the ledger validator
    ledger_mod.validate_ledger(json.loads(ledger_before.decode("utf-8")))
    assert pkg.ledger_path.read_bytes() == ledger_before


def test_external_to_pass_without_human_attribution_fails():
    # Tests R-024 [integration]: an EXTERNAL -> PASS edit without a
    # human-attribution field fails ledger validation.
    row = make_external_row(status="PASS")
    assert "human_attribution" not in row
    with pytest.raises(ledger_mod.LedgerValidationError):
        ledger_mod.validate_ledger(_ledger(rows=[make_ledger_row(), row]))


def test_external_to_pass_with_human_attribution_validates():
    # Tests R-024 [integration]: the human-attributed transition is the one
    # sanctioned path (repository green tests never substitute).
    row = make_external_row(
        status="PASS",
        human_attribution={"attributed_to": "author", "date": "2026-08-19"},
    )
    ledger_mod.validate_ledger(_ledger(rows=[make_ledger_row(), row]))


def test_venue_rule_rows_require_source_and_as_of_date():
    # Tests R-024 [integration]: venue-rule rows record only officially
    # published facts with source and as-of date.
    row = make_external_row(
        claim_id="clm-venue-0001",
        claim_kind="venue_rule",
        manuscript_wording="AIMS workshop page publishes no supplement rules.",
    )
    with pytest.raises(ledger_mod.LedgerValidationError):
        ledger_mod.validate_ledger(_ledger(rows=[make_ledger_row(), row]))
    row["source"] = "https://example.org/aims-2026-cfp"
    row["as_of_date"] = "2026-08-18"
    ledger_mod.validate_ledger(_ledger(rows=[make_ledger_row(), row]))


# ---------------------------------------------------------------------------
# R-025: Random-K disposition gate
# ---------------------------------------------------------------------------


def _random_k_row(**overrides):
    row = make_ledger_row(
        claim_id="clm-rk-0001",
        artifact_family="random_k",
        artifact_id="random_k_result.json",
        provenance_class="historical_randomk_v5",
        status="UNVERIFIED",
    )
    row.update(overrides)
    return row


def test_headline_random_k_without_author_decision_fails():
    # Tests R-025 [unit]: a headline-eligible Random-K row requires an
    # explicit author_decision.
    row = _random_k_row(headline_eligible=True)
    assert "author_decision" not in row
    with pytest.raises(ledger_mod.LedgerValidationError):
        ledger_mod.validate_ledger(_ledger(rows=[row, make_external_row()]))


def test_headline_random_k_rejects_undeclared_disposition():
    # Tests R-025 [unit]: only the two dispositions are allowed.
    row = _random_k_row(headline_eligible=True, author_decision="favorable_rerun")
    with pytest.raises(ledger_mod.LedgerValidationError):
        ledger_mod.validate_ledger(_ledger(rows=[row, make_external_row()]))


@pytest.mark.parametrize(
    "decision", ["historical_nonconfirmatory", "predeclared_multidraw_family"]
)
def test_headline_random_k_accepts_exactly_two_dispositions(decision):
    # Tests R-025 [unit]: historical/nonconfirmatory (excluded from
    # headlines) or predeclared multi-draw family.
    row = _random_k_row(headline_eligible=True, author_decision=decision)
    ledger_mod.validate_ledger(_ledger(rows=[row, make_external_row()]))


def test_non_headline_random_k_row_needs_no_disposition():
    # Tests R-025 [unit]: the gate is on headline eligibility.
    row = _random_k_row(headline_eligible=False)
    ledger_mod.validate_ledger(_ledger(rows=[row, make_external_row()]))


def test_substituted_random_k_draw_changes_digest_and_is_refused():
    # Tests R-025 [unit]: draw identity is estimand-defining (R-011) — a
    # substituted favorable draw changes the digest and pooling/comparison is
    # refused.
    from tests._colm_aims_helpers import expected_estimand_digest, make_estimand

    original = make_cell()
    substituted_est = make_estimand(random_k_draw_id="draw-favorable-7")
    substituted = make_cell(estimand=substituted_est)
    substituted["estimand_digest"] = expected_estimand_digest(substituted_est)
    with pytest.raises(pairing.EstimandMismatchError):
        pairing.check_poolable(original, substituted)


# ---------------------------------------------------------------------------
# R-026: rights inventory + sentinel-leak proof
# ---------------------------------------------------------------------------

RIGHTS_ENUM = {
    "VERIFIED_ALLOWED",
    "VERIFIED_RESTRICTED",
    "UNVERIFIED",
    "AUTHOR_DECISION_REQUIRED",
}


def test_rights_inventory_accepts_all_four_statuses():
    # Tests R-026 [unit]: the inventory enum is exactly the four values.
    rights = make_rights()
    rights["paths"][0]["status"] = "VERIFIED_RESTRICTED"
    rights["paths"][0]["upstream_terms_basis"] = "NAQT proprietary/excluded"
    rights["paths"][1]["status"] = "UNVERIFIED"
    rights["paths"][2]["status"] = "AUTHOR_DECISION_REQUIRED"
    rights["paths"][2]["upstream_terms_basis"] = (
        "PACE-archive author-retained packet copyright"
    )
    ledger_mod.validate_rights_inventory(rights)


def test_rights_status_outside_enum_fails():
    # Tests R-026 [unit]: unknown status values fail validation.
    rights = make_rights()
    rights["paths"][0]["status"] = "PROBABLY_FINE"
    with pytest.raises(ledger_mod.RightsError):
        ledger_mod.validate_rights_inventory(rights)


def test_rights_row_requires_upstream_terms_basis():
    # Tests R-026 [unit]: each row names its upstream terms basis rather than
    # a bare enum tick.
    rights = make_rights()
    del rights["paths"][0]["upstream_terms_basis"]
    with pytest.raises(ledger_mod.RightsError):
        ledger_mod.validate_rights_inventory(rights)
    rights2 = make_rights()
    rights2["paths"][0]["upstream_terms_basis"] = ""
    with pytest.raises(ledger_mod.RightsError):
        ledger_mod.validate_rights_inventory(rights2)


@pytest.mark.parametrize(
    "status", ["VERIFIED_RESTRICTED", "UNVERIFIED", "AUTHOR_DECISION_REQUIRED"]
)
def test_release_requires_every_included_path_verified_allowed(status):
    # Tests R-026 [unit]: release fails on any non-VERIFIED_ALLOWED value.
    rights = make_rights()
    rights["paths"][1]["status"] = status
    rights["paths"][1]["upstream_terms_basis"] = "NAQT proprietary/excluded"
    with pytest.raises(ledger_mod.RightsError):
        ledger_mod.check_rights_release(
            rights, ["profile.json", "records.jsonl", "presentation_manifest.json"]
        )


def test_release_fails_on_uninventoried_included_path():
    # Tests R-026 [unit]: any uninventoried included path fails.
    rights = make_rights()
    with pytest.raises(ledger_mod.RightsError):
        ledger_mod.check_rights_release(
            rights,
            [
                "profile.json",
                "records.jsonl",
                "presentation_manifest.json",
                "uninventoried.bin",
            ],
        )


def test_all_verified_allowed_release_check_passes():
    # Tests R-026 [unit]: the compliant inventory passes the release check.
    rights = make_rights()
    ledger_mod.check_rights_release(
        rights, ["profile.json", "records.jsonl", "presentation_manifest.json"]
    )


def _restricted_package(tmp_path: Path):
    """Adversarial rights fixture: a restricted file whose CONTENT and whose
    inventory note both carry the sentinel."""

    def add_restricted_row(rights):
        rights["paths"].append(
            {
                "path": "restricted-blob.bin",
                "status": "VERIFIED_RESTRICTED",
                "upstream_terms_basis": "NAQT proprietary/excluded",
                "content_note": SENTINEL,
            }
        )

    def declare(manifest):
        manifest["artifacts"].append(
            {"path": "restricted-blob.bin", "role": "restricted_payload"}
        )

    return build_package(
        tmp_path,
        extra_tree_files={
            "restricted-blob.bin": (SENTINEL + "\n").encode("utf-8")
        },
        rights_mutator=add_restricted_row,
        manifest_mutator=declare,
    )


def test_sentinel_never_leaks_from_rights_error_paths(tmp_path: Path):
    # Tests R-026 [unit]: error paths reference items by opaque stable keys
    # and never emit restricted content, credentials, or local absolute
    # paths — asserted over the API report, the rendered summary, and the
    # receipt bytes.
    pkg = _restricted_package(tmp_path)
    try:
        report = verifier.run_verifier(
            pkg.tree,
            mode="release",
            receipts_dir=pkg.receipts_dir,
            expectations=pkg.expectations_path,
        )
    except schema.ColmAimsError as exc:
        assert SENTINEL not in str(exc)
        assert str(pkg.tree) not in str(exc)
        return
    assert report.verdict == "FAIL"
    surface = json.dumps(report.legs)
    assert "restricted-blob.bin" in surface  # referenced by opaque key
    assert SENTINEL not in surface
    assert str(pkg.tree) not in surface  # no local absolute paths
    summary = render.render_summary(report)
    assert SENTINEL not in summary
    assert str(pkg.tree) not in summary
    for receipt in pkg.receipts_dir.glob("**/*"):
        if receipt.is_file():
            body = receipt.read_text("utf-8", errors="replace")
            assert SENTINEL not in body


def test_sentinel_never_leaks_from_cli_output(tmp_path: Path):
    # Tests R-026 [unit]: the CLI surface of the same adversarial fixture —
    # restricted content never appears in stdout, stderr, or receipts.
    pkg = _restricted_package(tmp_path)
    proc = run_cli(*cli_args_for(pkg, "release"))
    assert proc.returncode == EXIT_GATE_FAIL, (proc.returncode, proc.stderr)
    out = proc.stdout + proc.stderr
    assert SENTINEL not in out
    assert "restricted-blob.bin" in out
    for receipt in pkg.receipts_dir.glob("**/*"):
        if receipt.is_file():
            assert SENTINEL not in receipt.read_text("utf-8", errors="replace")


def test_rights_fixtures_are_synthetic():
    # Tests R-026 [unit]: fixtures are synthetic — no real qids, no raw
    # quizbowl text: every path key in the fixture inventory is an opaque
    # fixture name, and the standard records use opaque itm- keys.
    rights = make_rights()
    for row in rights["paths"]:
        assert row["path"].endswith((".json", ".jsonl", ".bin", ".txt"))
    for record in standard_records():
        assert record["item_key"].startswith("itm-")


# ---------------------------------------------------------------------------
# R-030: archival DOI gate
# ---------------------------------------------------------------------------


def test_available_grade_without_doi_fails():
    # Tests R-030 [unit]: "Artifacts Available"-grade assertion without a
    # DOI-class archival identifier fails validation.
    doc = _ledger()
    doc["availability_assertion"] = {"grade": "Available"}
    with pytest.raises(ledger_mod.LedgerValidationError):
        ledger_mod.validate_ledger(doc)


def test_github_url_does_not_qualify_as_archival_doi():
    # Tests R-030 [unit]: a GitHub URL does not qualify (ACM v1.1).
    doc = _ledger()
    doc["availability_assertion"] = {
        "grade": "Available",
        "archival_doi": "https://github.com/example/qanta-buzzer",
    }
    with pytest.raises(ledger_mod.LedgerValidationError):
        ledger_mod.validate_ledger(doc)


def test_doi_class_identifier_satisfies_available_grade():
    # Tests R-030 [unit]: a DOI-class identifier validates.
    doc = _ledger()
    doc["availability_assertion"] = {
        "grade": "Available",
        "archival_doi": "10.5281/zenodo.1234567",
    }
    ledger_mod.validate_ledger(doc)


def test_archival_doi_optional_absent_available_grade():
    # Tests R-030 [unit]: absent any Available-grade assertion the field is
    # optional — the default fixture ledger has neither and validates.
    doc = _ledger()
    assert "availability_assertion" not in doc
    ledger_mod.validate_ledger(doc)


def test_ledger_file_hash_matches_expectations_anchor(tmp_path: Path):
    # Tests R-023/R-013 [unit]: the frozen ledger the expectations anchor to
    # is the ledger on disk (hash equality is what release mode cross-checks).
    pkg = build_package(tmp_path)
    assert (
        pkg.expectations["anchor"]["ledger_sha256"] == sha256_file(pkg.ledger_path)
    )
