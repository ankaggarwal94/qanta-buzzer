"""Claim ledger, Random-K discipline, rights inventory: R-023, R-024, R-025,
R-026, R-030 (+ ledger-side R-056 rejected-ID storage, D3 blocking-task
golden).

GREENFIELD RED: fails at collection until the namespace exists.
"""
from __future__ import annotations

import pytest

from reproducibility.colm_aims_2026 import ledger as ledger_mod

from tests._colm_aims_v2_helpers import (
    RANDOM_K_BLOCKING_TASK,
    VERDICT_RELEASE_PASS,
    build_package_v2,
    canonical_data,
    colm_no_network,  # noqa: F401 - autouse fixture
    make_external_row,
    make_holm_row,
    make_ledger,
    make_ledger_row,
    make_random_k_row,
    make_rights,
    release_report,
    run_verifier_on,
    sha256_file,
)

COMMIT = "d" * 40


def _valid_ledger(**overrides):
    return make_ledger(source_commit=COMMIT, **overrides)


# ---------------------------------------------------------------------------
# R-023: row schema, closed status enum, anchored commit, blocking tasks
# ---------------------------------------------------------------------------

REQUIRED_ROW_FIELDS = (
    "claim_id",
    "claim_kind",
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
    "artifact_family",
    "headline_eligible",
)


class TestLedgerRows:
    def test_canonical_ledger_validates(self):
        ledger_mod.validate_ledger(_valid_ledger())

    @pytest.mark.parametrize("field", REQUIRED_ROW_FIELDS)
    def test_missing_row_field_rejected(self, field):
        row = make_ledger_row()
        del row[field]
        doc = _valid_ledger(
            rows=[row, make_holm_row(), make_random_k_row(), make_external_row()]
        )
        with pytest.raises(ledger_mod.LedgerValidationError):
            ledger_mod.validate_ledger(doc)

    def test_status_enum_closed(self):
        assert ledger_mod.LEDGER_STATUSES == frozenset(
            {"PASS", "FAIL", "UNVERIFIED", "EXTERNAL"}
        )
        doc = _valid_ledger(rows=[make_ledger_row(status="MAYBE")])
        with pytest.raises(ledger_mod.LedgerValidationError):
            ledger_mod.validate_ledger(doc)

    def test_missing_anchored_source_commit_rejected(self):
        doc = _valid_ledger()
        del doc["anchored_source_commit"]
        with pytest.raises(ledger_mod.LedgerValidationError):
            ledger_mod.validate_ledger(doc)

    def test_sha256_repository_commit_id_validates(self):
        doc = _valid_ledger()
        doc["anchored_source_commit"] = "a" * 64
        ledger_mod.validate_ledger(doc)

    def test_unverified_row_requires_genuine_blocking_task(self):
        # D3: a genuinely-open row must name a real remaining task.
        doc = _valid_ledger(
            rows=[
                make_ledger_row(),
                make_holm_row(),
                make_random_k_row(blocking_task=None),
                make_external_row(),
            ]
        )
        with pytest.raises(ledger_mod.LedgerValidationError):
            ledger_mod.validate_ledger(doc)

    def test_pass_row_must_not_carry_blocking_task(self):
        doc = _valid_ledger(
            rows=[make_ledger_row(blocking_task="tidy up later")]
        )
        with pytest.raises(ledger_mod.LedgerValidationError):
            ledger_mod.validate_ledger(doc)

    def test_random_k_blocking_task_golden_text(self):
        # Golden-ledger fixture pins the exact D3 task wording.
        assert make_random_k_row()["blocking_task"] == RANDOM_K_BLOCKING_TASK

    def test_holm_row_names_exact_rejected_cell_ids(self):
        data = canonical_data()
        assert make_holm_row()["rejected_cell_ids"] == (
            data["holm"]["rejected_cell_ids"]
        )

    def test_holm_row_missing_rejected_ids_rejected(self):
        row = make_holm_row()
        del row["rejected_cell_ids"]
        doc = _valid_ledger(rows=[make_ledger_row(), row])
        with pytest.raises(ledger_mod.LedgerValidationError):
            ledger_mod.validate_ledger(doc)

    def test_ledger_rejected_ids_must_match_package_inference(self, tmp_path):
        # Cross-document: the ledger's Holm row disagreeing with the package
        # rejected set FAILs release.
        def mutate(ledger):
            for row in ledger["rows"]:
                if row["claim_id"] == "clm-holm-inference":
                    row["rejected_cell_ids"] = row["rejected_cell_ids"][:-1]

        pkg = build_package_v2(tmp_path, ledger_mutator=mutate)
        report = release_report(pkg)
        assert report.verdict != VERDICT_RELEASE_PASS


# ---------------------------------------------------------------------------
# R-024: EXTERNAL immunity + human attribution + venue rows
# ---------------------------------------------------------------------------


class TestExternalRows:
    def test_external_to_pass_requires_human_attribution(self):
        doc = _valid_ledger(
            rows=[
                make_ledger_row(),
                make_external_row(status="PASS"),
            ]
        )
        with pytest.raises(ledger_mod.LedgerValidationError):
            ledger_mod.validate_ledger(doc)

    def test_external_to_pass_with_attribution_accepted(self):
        row = make_external_row(status="PASS")
        row["human_attribution"] = {
            "attributed_to": "author",
            "as_of": "2026-08-20",
        }
        doc = _valid_ledger(rows=[make_ledger_row(), row])
        ledger_mod.validate_ledger(doc)

    def test_venue_rule_row_requires_source_and_as_of(self):
        row = make_external_row(claim_kind="venue_rule")
        doc = _valid_ledger(rows=[make_ledger_row(), row])
        with pytest.raises(ledger_mod.LedgerValidationError):
            ledger_mod.validate_ledger(doc)
        row["source"] = "https://colmweb.org/aims-2026-cfp"
        row["as_of"] = "2026-08-20"
        ledger_mod.validate_ledger(
            _valid_ledger(rows=[make_ledger_row(), row])
        )

    def test_external_rows_byte_identical_across_tool_runs(self, tmp_path):
        # R-024: EXTERNAL rows are immune to repo tooling — the ledger file
        # is byte-identical across checker + both verifier modes.
        pkg = build_package_v2(tmp_path)
        before = sha256_file(pkg.ledger_path)
        ledger_mod.validate_ledger(pkg.ledger)
        run_verifier_on(pkg, "source")
        run_verifier_on(pkg, "release")
        assert sha256_file(pkg.ledger_path) == before

    def test_external_rows_exempt_from_release_recompute(self, tmp_path):
        # The canonical package carries an EXTERNAL manuscript row with no
        # repo-verifiable evidence; release still PASSes (R-012 exemption).
        pkg = build_package_v2(tmp_path)
        report = release_report(pkg)
        assert report.verdict == VERDICT_RELEASE_PASS


# ---------------------------------------------------------------------------
# R-025: Random-K discipline
# ---------------------------------------------------------------------------


class TestRandomK:
    def test_disposition_enum_is_closed_two_member(self):
        assert ledger_mod.RANDOM_K_DISPOSITIONS == frozenset(
            {"historical_nonconfirmatory", "predeclared_multidraw_family"}
        )

    def test_unknown_disposition_rejected(self):
        doc = _valid_ledger(
            rows=[
                make_ledger_row(),
                make_random_k_row(author_decision="confirmed_after_rerun"),
            ]
        )
        with pytest.raises(ledger_mod.LedgerValidationError):
            ledger_mod.validate_ledger(doc)

    def test_headline_eligible_with_historical_nonconfirmatory_rejected(self):
        # F1 joint validation: the PAIR is validated together.
        doc = _valid_ledger(
            rows=[
                make_ledger_row(),
                make_random_k_row(headline_eligible=True),
            ]
        )
        with pytest.raises(ledger_mod.LedgerValidationError):
            ledger_mod.validate_ledger(doc)

    def test_predeclared_family_row_requires_full_protocol(self):
        row = make_random_k_row(
            author_decision="predeclared_multidraw_family",
            headline_eligible=True,
        )
        doc = _valid_ledger(rows=[make_ledger_row(), row])
        with pytest.raises(ledger_mod.LedgerValidationError):
            ledger_mod.validate_ledger(doc)

    def test_predeclared_family_with_protocol_is_valid_but_inactive(self):
        # Discriminated INACTIVE contract: structurally valid with the full
        # protocol block; never run by this feature (SOURCE_CONTRACT_ONLY).
        row = make_random_k_row(
            author_decision="predeclared_multidraw_family",
            headline_eligible=True,
        )
        row["multidraw_protocol"] = {
            "frozen_protocol": "protocol-v2-0001",
            "seed_draw_registry": ["draw-0001", "draw-0002", "draw-0003"],
            "no_selective_omission": True,
            "aggregation_rule": "mean_over_registry",
            "sensitivity_analysis": "leave_one_draw_out",
            "multiplicity_treatment": "holm_within_registry",
        }
        ledger_mod.validate_ledger(
            _valid_ledger(rows=[make_ledger_row(), row])
        )

    def test_rng_pinned_false_must_be_recorded_explicitly(self):
        row = make_random_k_row()
        del row["rng_pinned"]
        doc = _valid_ledger(rows=[make_ledger_row(), row])
        with pytest.raises(ledger_mod.LedgerValidationError):
            ledger_mod.validate_ledger(doc)

    @pytest.mark.parametrize("field", ["archived_draw_id", "fresh_draw_id"])
    def test_both_draw_identities_required(self, field):
        row = make_random_k_row()
        del row[field]
        doc = _valid_ledger(rows=[make_ledger_row(), row])
        with pytest.raises(ledger_mod.LedgerValidationError):
            ledger_mod.validate_ledger(doc)

    def test_dagger_disclosure_retained(self):
        row = make_random_k_row()
        del row["disclosure_marker"]
        doc = _valid_ledger(rows=[make_ledger_row(), row])
        with pytest.raises(ledger_mod.LedgerValidationError):
            ledger_mod.validate_ledger(doc)

    def test_substituted_favorable_draw_changes_digest_and_is_refused(
        self, tmp_path
    ):
        # Draw identity is estimand-defining (R-011): substituting a
        # favorable fresh draw into a krandom cell changes the digest and
        # the package FAILs.
        def mutate(profile):
            for cell in profile["cells"]:
                if cell["reference_id"] == "krandom":
                    cell["estimand"]["random_k_draw_id"] = "draw-favorable-7"
                    # recorded digest left stale on purpose: substitution

        pkg = build_package_v2(tmp_path, profile_mutator=mutate)
        report = run_verifier_on(pkg, "source")
        assert report.verdict != "PASS_SOURCE_ONLY"


# ---------------------------------------------------------------------------
# R-030: archival DOI support
# ---------------------------------------------------------------------------


class TestArchivalDoi:
    def test_available_grade_without_doi_rejected(self):
        doc = _valid_ledger(
            availability_assertions=[{"grade": "available"}]
        )
        with pytest.raises(ledger_mod.LedgerValidationError):
            ledger_mod.validate_ledger(doc)

    def test_github_url_does_not_qualify(self):
        doc = _valid_ledger(
            availability_assertions=[
                {
                    "grade": "available",
                    "archival_identifier": (
                        "https://github.com/example/qanta-buzzer"
                    ),
                }
            ]
        )
        with pytest.raises(ledger_mod.LedgerValidationError):
            ledger_mod.validate_ledger(doc)

    def test_doi_identifier_accepted(self):
        doc = _valid_ledger(
            availability_assertions=[
                {
                    "grade": "available",
                    "archival_identifier": "10.5281/zenodo.1234567",
                }
            ]
        )
        ledger_mod.validate_ledger(doc)

    def test_absent_availability_assertion_is_optional(self):
        ledger_mod.validate_ledger(_valid_ledger())


# ---------------------------------------------------------------------------
# R-026: rights inventory
# ---------------------------------------------------------------------------


class TestRights:
    def test_status_enum_closed(self):
        rights = make_rights(["profile.json"])
        rights["paths"][0]["status"] = "PROBABLY_FINE"
        with pytest.raises(ledger_mod.RightsError):
            ledger_mod.validate_rights_inventory(rights)

    def test_upstream_terms_basis_required(self):
        rights = make_rights(["profile.json"])
        del rights["paths"][0]["upstream_terms_basis"]
        with pytest.raises(ledger_mod.RightsError):
            ledger_mod.validate_rights_inventory(rights)

    @pytest.mark.parametrize(
        "status",
        ["VERIFIED_RESTRICTED", "UNVERIFIED", "AUTHOR_DECISION_REQUIRED"],
    )
    def test_release_fails_on_any_non_allowed_status(self, tmp_path, status):
        def mutate(rights):
            rights["paths"][0]["status"] = status

        pkg = build_package_v2(tmp_path, rights_mutator=mutate)
        report = release_report(pkg)
        assert report.verdict != VERDICT_RELEASE_PASS

    def test_release_fails_on_uninventoried_included_path(self, tmp_path):
        # Present in the tree, declared in the manifest allowlist, but
        # MISSING from the rights inventory -> the inventory must cover
        # every file found.
        pkg = build_package_v2(
            tmp_path,
            extra_tree_files={"sidecars/extra.json": b'{"unknown_kind": 1}\n'},
            extra_manifest_allowlist=("sidecars/extra.json",),
            # no rights row on purpose
        )
        report = release_report(pkg)
        assert report.verdict != VERDICT_RELEASE_PASS

    def test_release_passes_with_full_rights_coverage(self, tmp_path):
        pkg = build_package_v2(tmp_path)
        report = release_report(pkg)
        assert report.verdict == VERDICT_RELEASE_PASS

    def test_stale_pass_ledger_row_fails_release_recompute(self, tmp_path):
        # R-012 recompute gate: a recorded PASS stronger than the recomputed
        # status fails (artifact evidence broken while the row says PASS).
        def mutate(ledger):
            for row in ledger["rows"]:
                if row["claim_id"] == "clm-0001":
                    row["input_identity"] = "9" * 64

        pkg = build_package_v2(tmp_path, ledger_mutator=mutate)
        report = release_report(pkg)
        assert report.verdict != VERDICT_RELEASE_PASS

    @pytest.mark.parametrize(
        "dependency_closure",
        [
            [],
            "scripts/fake_producer.py",
            ["scripts/fake_producer.py"],
            [
                "scripts/fake_producer.py",
                "scripts/fake_helper.py",
                "scripts/extra.py",
            ],
            ["scripts/fake_helper.py", "scripts/fake_producer.py"],
            [
                "scripts/fake_producer.py",
                "scripts/fake_helper.py",
                "scripts/fake_helper.py",
            ],
        ],
    )
    def test_pass_row_requires_exact_dependency_closure(
        self, tmp_path, dependency_closure
    ):
        def mutate(ledger):
            for row in ledger["rows"]:
                if row["claim_id"] == "clm-0001":
                    row["dependency_closure"] = dependency_closure

        pkg = build_package_v2(tmp_path, ledger_mutator=mutate)
        report = release_report(pkg)
        assert report.verdict != VERDICT_RELEASE_PASS
