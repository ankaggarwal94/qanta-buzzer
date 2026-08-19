"""QA fix-round-1 regression suite — ledger recompute interior (QA-003,
QA-004, QA-005, QA-010).

QA-003: recompute means re-derive from the verified provenance — every row
identity field is cross-checked; substitution with a different VALID value
still fires the gate.
QA-004: gate discriminants are validated closed enums (claim_kind,
artifact_family), never free-text string equality.
QA-005: the EXTERNAL predicate is anchored membership in the expectations
file — row-field evasions and reverse laundering both fail.
QA-010: the shipped repo ledger is exercised by its own validator.
Spec: .correctless/specs/camera-ready-aims-evidence.md (R-012/R-014/R-023/R-024)
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from reproducibility.colm_aims_2026 import ledger as ledger_mod
from reproducibility.colm_aims_2026 import verifier
from tests._colm_aims_helpers import (  # noqa: F401  (colm_no_network is an autouse fixture)
    REPO_ROOT,
    VERDICT_FAIL,
    VERDICT_RELEASE_PASS,
    build_package,
    colm_no_network,
    make_external_row,
    make_ledger,
    make_ledger_row,
    repo_head_commit,
)

REPO_LEDGER_PATH = (
    REPO_ROOT / "reproducibility" / "colm_aims_2026" / "ledger.json"
)


def _run_release(pkg):
    return verifier.run_verifier(
        pkg.tree,
        mode="release",
        receipts_dir=pkg.receipts_dir,
        expectations=pkg.expectations_path,
    )


def _failing(report):
    return [leg for leg in report.legs if leg.get("outcome") == "FAIL"]


def _ledger(**kwargs):
    return make_ledger(source_commit=repo_head_commit(), **kwargs)


# ---------------------------------------------------------------------------
# QA-003: identity cross-checks against the verified provenance
# ---------------------------------------------------------------------------


def test_full_identity_laundering_attack_fails_release(tmp_path: Path):
    # QA-003 [R-012]: the audit's exact exploit — a PASS row claiming an
    # observed-decision estimand with mismatched producer/input/split/model/
    # calibration identities and an arbitrary tree file as artifact_id must
    # produce a failing recompute leg, never PASS_RELEASE.
    def launder(ledger):
        ledger["rows"][0].update(
            estimand="observed_decision_preservation",
            producer_entrypoint="scripts/other_producer.py",
            dependency_closure=["scripts/other_producer.py"],
            input_identity="e" * 64,
            split_identity="mystery-split",
            model_identity="other-org/other-model@" + "9" * 40,
            calibration_identity="cal-9999",
            artifact_id="sealed-notes.bin",
        )

    pkg = build_package(tmp_path, ledger_mutator=launder)
    report = _run_release(pkg)
    assert report.verdict == VERDICT_FAIL
    legs = [
        leg
        for leg in _failing(report)
        if leg.get("leg_id") == "ledger_row_clm-0001_recompute"
    ]
    assert legs, "laundered row did not fail its recompute leg"
    assert legs[0]["observed"]["recomputed"] == "UNVERIFIED"


IDENTITY_SUBSTITUTIONS = [
    ("estimand", "some_other_estimand"),
    ("producer_entrypoint", "scripts/other_producer.py"),
    ("calibration_identity", "cal-0099"),
    ("split_identity", "other-split-v9"),
    ("model_identity", "other-org/scorer@" + "1" * 40),
    ("input_identity", "e" * 64),
    ("artifact_id", "sealed-notes.bin"),
]


@pytest.mark.parametrize(
    "field,value",
    IDENTITY_SUBSTITUTIONS,
    ids=[field for field, _ in IDENTITY_SUBSTITUTIONS],
)
def test_row_identity_substitution_negative(tmp_path: Path, field, value):
    # QA-003 class fix [R-012]: substitution-negative — one identity field
    # replaced with a DIFFERENT VALID value; the recompute gate must still
    # fire because it RE-DERIVES from the verified provenance rather than
    # proxy-checking the row against itself.
    def substitute(ledger):
        ledger["rows"][0][field] = value

    pkg = build_package(tmp_path, ledger_mutator=substitute)
    report = _run_release(pkg)
    assert report.verdict == VERDICT_FAIL, field
    legs = [
        leg
        for leg in _failing(report)
        if leg.get("leg_id") == "ledger_row_clm-0001_recompute"
    ]
    assert legs, f"substituted {field!r} did not fire the recompute gate"


def test_honest_row_still_recomputes_pass(tmp_path: Path):
    # QA-003 guard: the untouched default row keeps recomputing PASS —
    # identity cross-checks must not reject the consistent fixture.
    pkg = build_package(tmp_path)
    report = _run_release(pkg)
    assert report.verdict == VERDICT_RELEASE_PASS


# ---------------------------------------------------------------------------
# QA-004: closed claim_kind enum + discriminant sweep
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["aggregate", "per_item_paired"])
def test_claim_kind_enum_values_validate(kind):
    # QA-004 [R-014]: one fixture per enum value.
    row = make_ledger_row(claim_kind=kind)
    ledger_mod.validate_ledger(_ledger(rows=[row, make_external_row()]))


@pytest.mark.parametrize("bad", ["observed_paired", "aggregate_ish", "", None])
def test_claim_kind_outside_closed_enum_rejected(bad):
    # QA-004 [R-014]: renaming the discriminant is rejected, never routed to
    # a permissive branch.
    row = make_ledger_row(claim_kind=bad)
    with pytest.raises(ledger_mod.LedgerValidationError):
        ledger_mod.validate_ledger(_ledger(rows=[row, make_external_row()]))


def test_missing_claim_kind_rejected_naming_field():
    # QA-004 [R-014]: an omitted discriminant is rejected by name.
    row = make_ledger_row()
    del row["claim_kind"]
    with pytest.raises(ledger_mod.LedgerValidationError) as exc:
        ledger_mod.validate_ledger(_ledger(rows=[row, make_external_row()]))
    assert "claim_kind" in str(exc.value)


def test_renamed_estimand_cannot_reclassify_legacy_claim(tmp_path: Path):
    # QA-004 [R-014]: the audit's exploit — renaming the free-text estimand
    # used to flip a per-item paired claim into the permissive aggregate
    # branch, letting a legacy aggregate-only export certify it. The
    # validated claim_kind discriminant now decides; the renamed estimand
    # additionally fails the QA-003 estimand-membership re-derivation.
    from tests.test_colm_aims_verifier_gates import (
        _package_with_legacy_artifact,
    )

    def add_renamed_claim(ledger):
        ledger["rows"].append(
            make_ledger_row(
                claim_id="clm-legacy-renamed",
                claim_kind="per_item_paired",
                artifact_id="csli.json",
                artifact_family="legacy_aggregate",
                provenance_class="historical_submission_artifact",
                estimand="totally_not_per_item_paired",  # the rename
                status="PASS",
            )
        )

    pkg = _package_with_legacy_artifact(
        tmp_path, ledger_mutator=add_renamed_claim
    )
    report = _run_release(pkg)
    assert report.verdict == VERDICT_FAIL
    assert any(
        "clm-legacy-renamed" in json.dumps(leg) for leg in _failing(report)
    ), "renamed-estimand legacy claim escaped the recompute gate"


def test_artifact_family_missing_rejected():
    # QA-004 class fix: omitting the artifact_family discriminant is
    # rejected — never routed past the hazard/Random-K gates.
    row = make_ledger_row()
    del row["artifact_family"]
    with pytest.raises(ledger_mod.LedgerValidationError) as exc:
        ledger_mod.validate_ledger(_ledger(rows=[row, make_external_row()]))
    assert "artifact_family" in str(exc.value)


@pytest.mark.parametrize("renamed", ["pr41_hazard", "randomk", "hazardreport"])
def test_artifact_family_renamed_rejected(renamed):
    # QA-004 class fix: a renamed family value is outside the closed enum —
    # rejected instead of skipping the manuscript-citation / disposition
    # gates that key off it.
    row = make_ledger_row(artifact_family=renamed)
    with pytest.raises(ledger_mod.LedgerValidationError):
        ledger_mod.validate_ledger(_ledger(rows=[row, make_external_row()]))


def test_headline_eligible_must_be_boolean():
    # QA-004 class fix: the headline discriminant is typed — "yes" is not a
    # boolean and must be rejected, not treated as truthy.
    row = make_ledger_row(headline_eligible="yes")
    with pytest.raises(ledger_mod.LedgerValidationError):
        ledger_mod.validate_ledger(_ledger(rows=[row, make_external_row()]))


def test_recompute_rejects_row_with_missing_claim_kind(tmp_path: Path):
    # QA-004 [R-012/R-014]: a row that drops its claim_kind cannot recompute
    # to PASS (and the ledger validation leg names the field).
    def drop(ledger):
        del ledger["rows"][0]["claim_kind"]

    pkg = build_package(tmp_path, ledger_mutator=drop)
    report = _run_release(pkg)
    assert report.verdict == VERDICT_FAIL
    surface = json.dumps(_failing(report))
    assert "claim_kind" in surface
    assert "clm-0001" in surface


# ---------------------------------------------------------------------------
# QA-005: anchored EXTERNAL predicate + laundering evasions
# ---------------------------------------------------------------------------


def _pin_external(exp):
    exp["anchor"]["external_claim_ids"] = ["clm-ext-0001"]


def test_default_anchor_carries_external_claim_ids(tmp_path: Path):
    # QA-005: the builder anchors the EXTERNAL claim ids alongside the ledger
    # hash — the reviewed expectations file owns the predicate.
    pkg = build_package(tmp_path)
    assert pkg.expectations["anchor"]["external_claim_ids"] == ["clm-ext-0001"]


def test_two_field_evasion_fails_release(tmp_path: Path):
    # QA-005 [R-024]: flipping status+verifier_oracle launders the manuscript
    # row toward tool-verified PASS; anchored membership still catches it.
    def evade(ledger):
        ledger["rows"][1].update(
            status="PASS",
            verifier_oracle="reproducibility/colm_aims_2026/verifier.py",
        )

    pkg = build_package(
        tmp_path, ledger_mutator=evade, expectations_mutator=_pin_external
    )
    report = _run_release(pkg)
    assert report.verdict == VERDICT_FAIL
    legs = [
        leg
        for leg in _failing(report)
        if leg.get("leg_id") == "ledger_row_clm-ext-0001_external_immunity"
    ]
    assert legs, "two-field evasion escaped the anchored EXTERNAL predicate"


def test_three_field_evasion_fails_release(tmp_path: Path):
    # QA-005 [R-024]: the audit's full evasion — status+verifier_oracle+
    # producer_entrypoint flipped together dodges every row-field predicate;
    # only state the ledger editor cannot reach (the anchored id list) can
    # catch it.
    def evade(ledger):
        ledger["rows"][1].update(
            status="PASS",
            verifier_oracle="reproducibility/colm_aims_2026/verifier.py",
            producer_entrypoint="scripts/fake_producer.py",
        )

    pkg = build_package(
        tmp_path, ledger_mutator=evade, expectations_mutator=_pin_external
    )
    report = _run_release(pkg)
    assert report.verdict == VERDICT_FAIL
    legs = [
        leg
        for leg in _failing(report)
        if leg.get("leg_id") == "ledger_row_clm-ext-0001_external_immunity"
    ]
    assert legs, "three-field evasion escaped the anchored EXTERNAL predicate"
    assert legs[0]["observed"]["human_attribution"] is False


def test_attributed_external_transition_passes(tmp_path: Path):
    # QA-005 [R-024]: the one sanctioned path — a human-attributed transition
    # of an anchored-EXTERNAL row — passes release.
    def transition(ledger):
        ledger["rows"][1].update(
            status="PASS",
            human_attribution={"attributed_to": "author", "date": "2026-08-19"},
        )

    pkg = build_package(
        tmp_path, ledger_mutator=transition, expectations_mutator=_pin_external
    )
    report = _run_release(pkg)
    assert report.verdict == VERDICT_RELEASE_PASS


def test_reverse_laundering_to_external_fails(tmp_path: Path):
    # QA-005 class fix: a non-EXTERNAL row cannot grant itself recompute
    # immunity by relabeling its status EXTERNAL — unanchored EXTERNAL fails.
    def relabel(ledger):
        ledger["rows"][0].update(status="EXTERNAL")

    pkg = build_package(
        tmp_path, ledger_mutator=relabel, expectations_mutator=_pin_external
    )
    report = _run_release(pkg)
    assert report.verdict == VERDICT_FAIL
    legs = [
        leg
        for leg in _failing(report)
        if leg.get("leg_id") == "ledger_row_clm-0001_external_immunity"
    ]
    assert legs, "self-relabelled EXTERNAL row escaped the anchor check"
    assert legs[0]["observed"]["anchored_external"] is False


def test_missing_external_claim_ids_fails_closed(tmp_path: Path):
    # QA-005: an expectations anchor without the external_claim_ids list is a
    # MISSING_EXPECTATION failure — the predicate never silently degrades to
    # row-field trust.
    pkg = build_package(
        tmp_path,
        expectations_mutator=lambda exp: exp["anchor"].pop(
            "external_claim_ids"
        ),
    )
    report = _run_release(pkg)
    assert report.verdict == VERDICT_FAIL
    legs = [
        leg
        for leg in _failing(report)
        if leg.get("leg_id") == "anchor_external_claim_ids"
    ]
    assert legs
    assert legs[0]["remediation_class"] == "MISSING_EXPECTATION"


# ---------------------------------------------------------------------------
# QA-010: shipped repo ledger self-validation
# ---------------------------------------------------------------------------


def test_shipped_ledger_validates():
    # QA-010 [R-023]: the document the feature ships is exercised by its own
    # validator in the suite.
    doc = json.loads(REPO_LEDGER_PATH.read_text("utf-8"))
    ledger_mod.validate_ledger(doc)


def test_shipped_ledger_statuses_are_honest():
    # QA-010: while no evidence package run is published, every shipped row
    # stays EXTERNAL or UNVERIFIED — never a self-granted PASS.
    doc = json.loads(REPO_LEDGER_PATH.read_text("utf-8"))
    statuses = {row["status"] for row in doc["rows"]}
    assert statuses <= {"EXTERNAL", "UNVERIFIED"}, statuses
