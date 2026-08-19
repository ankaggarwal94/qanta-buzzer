"""QA fix-round-2 regression suite (QA-015..QA-019).

QA-015: validate_ledger — the surface R-024 names — rejects EXTERNAL
laundering standalone (anchored membership when supplied; widened kind
heuristic when not), plus the direct-API adversarial sweep over every
rule-named validator.
QA-016: ledger recompute decomposes namespace@revision and cross-checks
every component — no partial-string identity comparisons.
QA-017: honest helperless producers pass; invalid helper digests fail.
QA-018: coverage_rate is a unit interval; mc_build values must be
internally consistent.
QA-019: internal defects exit EXIT_INTERNAL_ERROR (4) with path-scrubbed
diagnostics; a render defect cannot convert a reached verdict.
Spec: .correctless/specs/camera-ready-aims-evidence.md
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from reproducibility.colm_aims_2026 import ledger as ledger_mod
from reproducibility.colm_aims_2026 import render, schema, verifier, verify
from tests._colm_aims_helpers import (  # noqa: F401  (colm_no_network is an autouse fixture)
    EXIT_GATE_FAIL,
    EXIT_INGRESS_ERROR,
    FAKE_SHA_A,
    VERDICT_FAIL,
    VERDICT_RELEASE_PASS,
    build_package,
    cli_args_for,
    colm_no_network,
    make_external_row,
    make_ledger,
    make_ledger_row,
    make_profile,
    make_provenance,
    make_record,
    make_rights,
    repo_head_commit,
    run_cli,
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


def _three_field_laundered_external_row():
    """The QA-005/QA-015 evasion: status + verifier_oracle +
    producer_entrypoint flipped together on the manuscript row."""
    return make_external_row(
        status="PASS",
        verifier_oracle="reproducibility/colm_aims_2026/verifier.py",
        producer_entrypoint="scripts/fake_producer.py",
    )


# ---------------------------------------------------------------------------
# QA-015: validate_ledger rejects laundering standalone
# ---------------------------------------------------------------------------


def test_validate_ledger_rejects_three_field_laundering_standalone():
    # QA-015 [R-024]: the rule-named surface itself — no verifier backstop —
    # rejects the three-field laundering via the widened kind heuristic
    # (provenance_class manuscript_identity survives the flip).
    doc = _ledger(
        rows=[make_ledger_row(), _three_field_laundered_external_row()]
    )
    with pytest.raises(ledger_mod.LedgerValidationError) as exc:
        ledger_mod.validate_ledger(doc)
    assert "clm-ext-0001" in str(exc.value)


@pytest.mark.parametrize(
    "kind_field,kind_value",
    [
        ("provenance_class", "manuscript_identity"),
        ("claim_kind", "venue_rule"),
        ("claim_kind", "external_fact"),
    ],
)
def test_widened_heuristic_covers_every_external_kind_marker(
    kind_field, kind_value
):
    # QA-015 [R-024]: each EXTERNAL-kind marker independently triggers the
    # attribution requirement on a non-EXTERNAL status.
    row = _three_field_laundered_external_row()
    row["provenance_class"] = "current_source"
    row["claim_kind"] = "per_item_paired"
    row[kind_field] = kind_value
    if kind_field == "claim_kind" and kind_value == "venue_rule":
        row["source"] = "recorded venue source"
        row["as_of_date"] = "2026-08-18"
    with pytest.raises(ledger_mod.LedgerValidationError):
        ledger_mod.validate_ledger(_ledger(rows=[make_ledger_row(), row]))


def test_membership_predicate_catches_full_five_field_laundering():
    # QA-015 [R-024]: a FULL laundering (also flipping provenance_class and
    # claim_kind) dodges every same-document heuristic — only the anchored
    # membership list, which the ledger editor cannot reach, catches it.
    row = _three_field_laundered_external_row()
    row["provenance_class"] = "current_source"
    row["claim_kind"] = "per_item_paired"
    doc = _ledger(rows=[make_ledger_row(), row])
    # Without the anchored list the heuristic is dodged (the documented
    # residual the membership predicate exists to close):
    ledger_mod.validate_ledger(doc)
    # With it, membership is the predicate:
    with pytest.raises(ledger_mod.LedgerValidationError) as exc:
        ledger_mod.validate_ledger(doc, external_claim_ids=["clm-ext-0001"])
    assert "clm-ext-0001" in str(exc.value)


def test_membership_predicate_allows_attributed_transition():
    # QA-015 [R-024]: the sanctioned human-attributed transition validates
    # under the anchored membership predicate.
    row = make_external_row(
        status="PASS",
        human_attribution={"attributed_to": "author", "date": "2026-08-19"},
    )
    ledger_mod.validate_ledger(
        _ledger(rows=[make_ledger_row(), row]),
        external_claim_ids=["clm-ext-0001"],
    )


def test_release_ledger_validation_leg_fires_on_full_laundering(
    tmp_path: Path,
):
    # QA-015 [R-024]: _ledger_legs passes the anchored list through, so the
    # ledger_validation leg fails on the full laundering in addition to the
    # external_immunity leg.
    def evade(ledger):
        ledger["rows"][1].update(
            status="PASS",
            verifier_oracle="reproducibility/colm_aims_2026/verifier.py",
            producer_entrypoint="scripts/fake_producer.py",
            provenance_class="current_source",
            claim_kind="per_item_paired",
        )

    pkg = build_package(
        tmp_path,
        ledger_mutator=evade,
        expectations_mutator=lambda exp: exp["anchor"].update(
            external_claim_ids=["clm-ext-0001"]
        ),
    )
    report = _run_release(pkg)
    assert report.verdict == VERDICT_FAIL
    failing_ids = {leg.get("leg_id") for leg in _failing(report)}
    assert "ledger_validation" in failing_ids
    assert "ledger_row_clm-ext-0001_external_immunity" in failing_ids


# QA-015 class fix: direct-API adversarial sweep over every rule-named
# validator API ("the validator rejects ..." / "fails ledger validation" /
# "fails validation" wording in the spec).
def _adv_validate_profile():
    profile = make_profile()
    profile["semantic"]["observed_open_ended"] = True  # R-001/R-002
    schema.validate_profile(profile)


def _adv_validate_record():
    schema.validate_record(
        make_record("itm-0001", 1, 3, question_text="free text")  # R-031
    )


def _adv_validate_ledger_laundering():
    ledger_mod.validate_ledger(  # R-024
        _ledger(rows=[make_ledger_row(), _three_field_laundered_external_row()])
    )


def _adv_validate_ledger_doi():
    doc = _ledger()
    doc["availability_assertion"] = {"grade": "Available"}  # R-030
    ledger_mod.validate_ledger(doc)


def _adv_validate_rights_inventory():
    rights = make_rights()
    rights["paths"][0]["status"] = "PROBABLY_FINE"  # R-026
    ledger_mod.validate_rights_inventory(rights)


def _adv_check_rights_release():
    rights = make_rights()
    rights["paths"][0]["status"] = "VERIFIED_RESTRICTED"
    rights["paths"][0]["upstream_terms_basis"] = "NAQT proprietary/excluded"
    ledger_mod.check_rights_release(rights, [rights["paths"][0]["path"]])


RULE_NAMED_VALIDATOR_ADVERSARIALS = [
    ("validate_profile_R001", _adv_validate_profile),
    ("validate_record_R031", _adv_validate_record),
    ("validate_ledger_R024_laundering", _adv_validate_ledger_laundering),
    ("validate_ledger_R030_doi", _adv_validate_ledger_doi),
    ("validate_rights_inventory_R026", _adv_validate_rights_inventory),
    ("check_rights_release_R026", _adv_check_rights_release),
]


@pytest.mark.parametrize(
    "name,adversarial",
    RULE_NAMED_VALIDATOR_ADVERSARIALS,
    ids=[name for name, _ in RULE_NAMED_VALIDATOR_ADVERSARIALS],
)
def test_rule_named_api_rejects_adversarial_input_directly(name, adversarial):
    # QA-015 class fix: every API the spec names as a rejecting surface gets
    # a DIRECT adversarial call (no end-to-end wrapper) that must raise.
    with pytest.raises(schema.ColmAimsError):
        adversarial()


# ---------------------------------------------------------------------------
# QA-016: model_identity decomposition in the recompute
# ---------------------------------------------------------------------------

MODEL_IDENTITY_MUTATIONS = [
    ("wrong_revision", "example-org/tiny-scorer@" + "9" * 40),
    ("bare_repo_id", "datasets/qanta"),
    ("branch_name", "example-org/tiny-scorer@main"),
    ("short_hash", "example-org/tiny-scorer@8ddb420"),
    ("wrong_namespace", "other-org/tiny-scorer@" + (
        "1234567890abcdef1234567890abcdef12345678"
    )),
]


@pytest.mark.parametrize(
    "name,identity",
    MODEL_IDENTITY_MUTATIONS,
    ids=[name for name, _ in MODEL_IDENTITY_MUTATIONS],
)
def test_row_model_identity_component_mismatch_fails(
    tmp_path: Path, name, identity
):
    # QA-016 [R-012]: every mutable revision form R-012 rejects (different
    # revision, bare repo id, branch, short hash) and a wrong namespace all
    # fail the recompute leg — no prefix-only acceptance.
    def substitute(ledger):
        ledger["rows"][0]["model_identity"] = identity

    pkg = build_package(tmp_path, ledger_mutator=substitute)
    report = _run_release(pkg)
    assert report.verdict == VERDICT_FAIL, name
    legs = [
        leg
        for leg in _failing(report)
        if leg.get("leg_id") == "ledger_row_clm-0001_recompute"
    ]
    assert legs, f"{name}: model-identity mutation escaped the recompute gate"
    assert legs[0]["observed"]["recomputed"] == "UNVERIFIED"


def _recompute(row, prov, files=None, estimands=None):
    files = files if files is not None else {
        "profile.json": Path("profile.json"),
        "records.jsonl": Path("records.jsonl"),
        "sealed-notes.bin": Path("sealed-notes.bin"),
    }
    estimands = estimands if estimands is not None else {
        "signed_index_shift_mc_minus_ref"
    }
    return verifier._recompute_row_status(
        row, files, {}, True, prov, estimands
    )


def test_recompute_baseline_reaches_pass():
    # Baseline for the meta-test below: the consistent default row PASSes.
    assert _recompute(make_ledger_row(), make_provenance()) == "PASS"


# QA-016 class fix: meta-test — every row identity field has a provenance
# cross-check; substituting any single field with a VALID-but-different
# value must drop the recomputed status below PASS.
ROW_IDENTITY_SUBSTITUTIONS = {
    "producer_entrypoint": "scripts/other_producer.py",
    "calibration_identity": "cal-9999",
    "split_identity": "other-split-v9",
    "model_identity": "example-org/tiny-scorer@" + "9" * 40,
    "input_identity": "e" * 64,
    "estimand": "another_estimand",
    "artifact_id": "sealed-notes.bin",
    "claim_kind": "venue_rule",
}


@pytest.mark.parametrize(
    "field", sorted(ROW_IDENTITY_SUBSTITUTIONS), ids=sorted(
        ROW_IDENTITY_SUBSTITUTIONS
    )
)
def test_every_row_identity_field_is_cross_checked(field):
    # QA-016 class fix [R-012]: direct unit meta-test over
    # _recompute_row_status — each identity field individually gates PASS.
    row = make_ledger_row()
    row[field] = ROW_IDENTITY_SUBSTITUTIONS[field]
    assert _recompute(row, make_provenance()) != "PASS", field


def test_byte_digest_row_revision_must_name_anchored_digest():
    # QA-016 [R-012]: under the byte-digest alternative the row's revision
    # part must name one of the anchored canonical digests — an arbitrary
    # (even commit-shaped) revision is refused.
    prov = make_provenance()
    del prov["model"]["revision"]
    prov["model"]["byte_digest_manifest"] = {"weights.bin": "a" * 64}
    good = make_ledger_row(
        model_identity="example-org/tiny-scorer@" + "a" * 64
    )
    assert _recompute(good, prov) == "PASS"
    bad = make_ledger_row(
        model_identity="example-org/tiny-scorer@" + "9" * 40
    )
    assert _recompute(bad, prov) != "PASS"


# ---------------------------------------------------------------------------
# QA-017: honest helperless producer
# ---------------------------------------------------------------------------


def test_helperless_producer_reaches_pass_release(tmp_path: Path):
    # QA-017 [R-012]: an honest producer with zero helpers ({}) is a valid
    # provenance shape — the release run reaches PASS_RELEASE.
    pkg = build_package(
        tmp_path,
        binding_mutator=lambda prov: prov.update(helper_sha256s={}),
    )
    report = _run_release(pkg)
    assert report.verdict == VERDICT_RELEASE_PASS


def test_helper_map_with_invalid_digest_fails(tmp_path: Path):
    # QA-017 [R-012]: entries, when present, must be sha256 digests.
    pkg = build_package(
        tmp_path,
        binding_mutator=lambda prov: prov.update(
            helper_sha256s={"x": "not-a-sha"}
        ),
    )
    report = _run_release(pkg)
    assert report.verdict == VERDICT_FAIL
    failing_ids = {leg.get("leg_id") for leg in _failing(report)}
    assert {"binding_producer", "producer_recorded"} & failing_ids


PROVENANCE_DEGENERATE_HONEST = [
    ("helperless_producer", lambda prov: prov.update(helper_sha256s={})),
    ("single_seed", lambda prov: prov.update(seeds=[1])),
]


@pytest.mark.parametrize(
    "name,mutate",
    PROVENANCE_DEGENERATE_HONEST,
    ids=[name for name, _ in PROVENANCE_DEGENERATE_HONEST],
)
def test_provenance_degenerate_honest_family_passes(tmp_path: Path, name, mutate):
    # QA-017 class extension: the degenerate-honest fixture family extends
    # from cell shapes to provenance shapes — minimal honest provenance
    # still verifies.
    pkg = build_package(tmp_path, binding_mutator=mutate)
    report = _run_release(pkg)
    assert report.verdict == VERDICT_RELEASE_PASS, name


# ---------------------------------------------------------------------------
# QA-018: coverage-rate calibration + mc_build internal consistency
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("coverage", [-3.0, 47.0])
def test_out_of_interval_coverage_rate_fails(tmp_path: Path, coverage):
    # QA-018 [R-012]: a coverage rate is a proportion — recorded-but-nonsense
    # values outside [0, 1] fail the table leg even when mirrored.
    pkg = build_package(
        tmp_path,
        binding_mutator=lambda prov: prov["mc_build"].update(
            coverage_rate=coverage
        ),
    )
    report = _run_release(pkg)
    assert report.verdict == VERDICT_FAIL
    legs = [
        leg
        for leg in _failing(report)
        if leg.get("leg_id") == "mc_build_coverage_retention"
    ]
    assert legs, f"coverage_rate {coverage!r} escaped the unit-interval gate"
    assert "unit_interval" in json.dumps(legs[0].get("observed"))


def test_zero_retained_with_full_coverage_fails_consistency(tmp_path: Path):
    # QA-018 [R-012]: retained_count 0 with coverage_rate 1.0 is internally
    # inconsistent (retained_count > 0 iff coverage_rate > 0).
    pkg = build_package(
        tmp_path,
        binding_mutator=lambda prov: prov["mc_build"].update(retained_count=0),
    )
    report = _run_release(pkg)
    assert report.verdict == VERDICT_FAIL
    legs = [
        leg
        for leg in _failing(report)
        if leg.get("leg_id") == "mc_build_internal_consistency"
    ]
    assert legs, "0-retained/full-coverage escaped the consistency gate"


def test_retained_count_exceeding_eval_split_fails_consistency(
    tmp_path: Path,
):
    # QA-018 [R-012]: retained_count must square with splits.eval.count under
    # the declared retention policy.
    pkg = build_package(
        tmp_path,
        binding_mutator=lambda prov: prov["mc_build"].update(
            retained_count=99
        ),
    )
    report = _run_release(pkg)
    assert report.verdict == VERDICT_FAIL
    legs = [
        leg
        for leg in _failing(report)
        if leg.get("leg_id") == "mc_build_internal_consistency"
    ]
    assert legs
    assert "exceeds" in json.dumps(legs[0].get("observed"))


# ---------------------------------------------------------------------------
# QA-019: internal-error exit code, scrubbed diagnostics, render isolation
# ---------------------------------------------------------------------------


def test_exit_internal_error_is_pinned_and_distinct():
    # QA-019 [R-037]: EXIT_INTERNAL_ERROR is 4 and distinct from the four
    # R-037 codes (equality assertions, per the suite rule).
    assert verify.EXIT_INTERNAL_ERROR == 4
    assert (
        len(
            {
                verify.EXIT_PASS,
                verify.EXIT_GATE_FAIL,
                verify.EXIT_USAGE_ERROR,
                verify.EXIT_INGRESS_ERROR,
                verify.EXIT_INTERNAL_ERROR,
            }
        )
        == 5
    )


def test_internal_defect_matrix_row_exits_four_not_ingress(tmp_path: Path):
    # QA-019 corruption-matrix row: a NON-ingress internal defect (receipt
    # emission hitting a file where its directory should be) exits 4 —
    # never colliding with typed-ingress (3) or gate-FAIL (1) — with a
    # path-scrubbed diagnostic on stderr.
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
    assert proc.returncode == verify.EXIT_INTERNAL_ERROR
    assert proc.returncode != EXIT_INGRESS_ERROR
    assert proc.returncode != EXIT_GATE_FAIL
    assert "Traceback" not in proc.stderr
    assert str(tmp_path) not in proc.stderr
    assert "receipts-as-file" in proc.stderr  # scrubbed basename diagnostics


def test_render_defect_cannot_convert_gate_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
):
    # QA-019: rendering happens outside the verification try — a render
    # defect on a reached FAIL verdict still exits EXIT_GATE_FAIL and still
    # prints the verdict token.
    pkg = build_package(
        tmp_path,
        expectations_mutator=lambda exp: exp["bindings"]["model"].update(
            revision="9" * 40
        ),
    )

    def boom(report):
        raise RuntimeError("render defect")

    monkeypatch.setattr(render, "render_summary", boom)
    code = verify.main(cli_args_for(pkg, "release"))
    captured = capsys.readouterr()
    assert code == EXIT_GATE_FAIL
    assert "FAIL" in captured.out
    assert "Traceback" not in captured.err


def test_render_defect_cannot_convert_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
):
    # QA-019: same isolation on the PASS side.
    pkg = build_package(tmp_path)

    def boom(report):
        raise RuntimeError("render defect")

    monkeypatch.setattr(render, "render_summary", boom)
    code = verify.main(cli_args_for(pkg, "source"))
    captured = capsys.readouterr()
    assert code == verify.EXIT_PASS
    assert "PASS_SOURCE_ONLY" in captured.out


def test_scrub_paths_strips_supplied_and_residual_absolutes(tmp_path: Path):
    # QA-019: the scrubber reduces supplied paths (and their resolved forms)
    # to basenames and collapses residual absolute tokens.
    supplied = str(tmp_path / "receipts-dir")
    message = (
        f"boom at {supplied}/receipt-1.json and also"
        " /somewhere/else/secret.txt"
    )
    scrubbed = verify._scrub_paths(message, [supplied])
    assert str(tmp_path) not in scrubbed
    assert "receipts-dir" in scrubbed
    assert "/somewhere/else" not in scrubbed
    assert "secret.txt" in scrubbed


def test_input_identity_closure_still_anchors_default_row():
    # Round-2 guard: the QA-016/QA-017 reworks keep the default fixture
    # row's input identity inside the provenance closure.
    prov = make_provenance()
    closure = verifier._provenance_identity_closure(prov)
    assert FAKE_SHA_A in closure
