"""RED suite — two-mode verifier gates.

Covers: R-012, R-013, R-014, R-015 (verifier surface), R-017, R-019,
R-033, R-035.
Spec: .correctless/specs/camera-ready-aims-evidence.md

No formal Entry/Through/Exit contracts exist for the [integration] rules
(spec OQ-003: ARCHITECTURE.md defines no entrypoints). Integration-level
coverage here uses the Python API; the documented CLI entrypoint is
exercised in tests/test_colm_aims_verifier_cli.py.
# No documented entrypoint — using inferred entry point (run_verifier API +
# the R-037 CLI), per OQ-003.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from reproducibility.colm_aims_2026 import render, schema, verifier
from tests._colm_aims_helpers import (  # noqa: F401  (colm_no_network is an autouse fixture)
    FAKE_COMMIT,
    LEGACY_DIR,
    REMEDIATION_CLASSES,
    VERDICT_FAIL,
    VERDICT_RELEASE_PASS,
    VERDICT_SOURCE_PASS,
    build_package,
    colm_no_network,
    expected_estimand_digest,
    keyset_sha256,
    make_ledger_row,
    make_record,
    repo_head_commit,
    rewrite_json,
    tree_hashes,
)


def _run_source(pkg):
    return verifier.run_verifier(
        pkg.tree, mode="source", receipts_dir=pkg.receipts_dir
    )


def _run_release(pkg, expectations=None):
    return verifier.run_verifier(
        pkg.tree,
        mode="release",
        receipts_dir=pkg.receipts_dir,
        expectations=expectations if expectations is not None else pkg.expectations_path,
    )


def _assert_rejected_release(pkg, expect_token: str | None = None) -> None:
    """A defective package must FAIL release mode (verdict) or raise a typed
    ColmAimsError. A NotImplementedError (RED stubs) propagates and fails the
    test, as intended. ``expect_token`` (audit ADV-5) additionally pins WHICH
    gate fired: the token must appear in the error text or in a failing leg."""
    try:
        report = _run_release(pkg)
    except schema.ColmAimsError as exc:
        if expect_token is not None:
            assert expect_token in str(exc).lower(), (
                f"gate token {expect_token!r} missing from error: {exc}"
            )
        return
    assert report.verdict == VERDICT_FAIL, (
        f"expected release rejection, got verdict {report.verdict!r}"
    )
    if expect_token is not None:
        surface = json.dumps(_failing_legs(report)).lower()
        assert expect_token in surface, (
            f"gate token {expect_token!r} missing from failing legs"
        )


def _assert_rejected_source(pkg, expect_token: str | None = None) -> None:
    try:
        report = _run_source(pkg)
    except schema.ColmAimsError as exc:
        if expect_token is not None:
            assert expect_token in str(exc).lower(), (
                f"gate token {expect_token!r} missing from error: {exc}"
            )
        return
    assert report.verdict == VERDICT_FAIL, (
        f"expected source rejection, got verdict {report.verdict!r}"
    )
    if expect_token is not None:
        surface = json.dumps(_failing_legs(report)).lower()
        assert expect_token in surface, (
            f"gate token {expect_token!r} missing from failing legs"
        )


def _failing_legs(report):
    return [leg for leg in report.legs if leg.get("outcome") == "FAIL"]


# ---------------------------------------------------------------------------
# R-012: release mode fails closed per leg
# ---------------------------------------------------------------------------

BINDING_KEYS = [
    "schema_profile",
    "producer",
    "semantic_command",
    "seeds",
    "dirty_state",
    "splits",
    "calibration_identity",
    "continuation_identity",
    "input_hashes",
    "split_metadata_sha256",
    "mc_build",
    "model",
    "runtime_packages",
]


@pytest.mark.parametrize("binding_key", BINDING_KEYS)
def test_release_fails_closed_per_missing_binding(tmp_path: Path, binding_key):
    # Tests R-012 [integration]: one fixture per missing binding — release
    # mode FAILs, names the leg, reports expected-vs-observed and a
    # remediation class of MISSING_EXPECTATION.
    def drop(exp):
        del exp["bindings"][binding_key]

    pkg = build_package(tmp_path, expectations_mutator=drop)
    report = _run_release(pkg)
    assert report.verdict == VERDICT_FAIL
    matching = [
        leg for leg in _failing_legs(report) if binding_key in leg.get("leg_id", "")
    ]
    assert matching, f"no failing leg names binding {binding_key!r}"
    leg = matching[0]
    assert "expected" in leg and "observed" in leg
    assert leg.get("remediation_class") == "MISSING_EXPECTATION"


def test_release_fails_closed_on_unresolved_binding(tmp_path: Path):
    # Tests R-012 [integration]: an UNRESOLVED binding fails the leg.
    def unresolve(profile):
        profile["provenance"]["calibration_identity"] = "UNRESOLVED"

    pkg = build_package(tmp_path, profile_mutator=unresolve)
    report = _run_release(pkg)
    assert report.verdict == VERDICT_FAIL
    assert any(
        "calibration" in leg.get("leg_id", "") for leg in _failing_legs(report)
    )


@pytest.mark.parametrize(
    "bad_revision",
    [
        "8ddb420",              # short hash
        "v1.0",                 # tag
        "main",                 # branch name
        "datasets/qanta",       # bare repo id (reassignable)
    ],
)
def test_release_rejects_mutable_model_revisions(tmp_path: Path, bad_revision):
    # Tests R-012 [integration]: the scoring/selector model binding requires
    # an immutable full-length commit SHA; short hashes, tags, branch names,
    # and bare repo ids are rejected.
    def mutate(profile):
        profile["provenance"]["model"]["revision"] = bad_revision

    pkg = build_package(tmp_path, profile_mutator=mutate)
    report = _run_release(pkg)
    assert report.verdict == VERDICT_FAIL
    model_legs = [
        leg for leg in _failing_legs(report) if "model" in leg.get("leg_id", "")
    ]
    assert model_legs
    # ADV-1: the leg's OBSERVED value round-trips the offending revision.
    assert bad_revision in json.dumps(model_legs[0].get("observed"))


def test_model_leg_reports_expected_and_observed_values(tmp_path: Path):
    # Tests R-012 [integration] (audit ADV-1): a binding-mismatch leg carries
    # the real expected/observed VALUES, not just the key name — the
    # expectations-side revision and the artifact-side revision both
    # round-trip through the report.
    expected_revision = "9" * 40
    observed_revision = "1234567890abcdef1234567890abcdef12345678"  # builder pin

    pkg = build_package(
        tmp_path,
        expectations_mutator=lambda exp: exp["bindings"]["model"].update(
            revision=expected_revision
        ),
    )
    assert pkg.profile["provenance"]["model"]["revision"] == observed_revision
    report = _run_release(pkg)
    assert report.verdict == VERDICT_FAIL
    model_legs = [
        leg for leg in _failing_legs(report) if "model" in leg.get("leg_id", "")
    ]
    assert model_legs
    leg = model_legs[0]
    assert expected_revision in json.dumps(leg.get("expected"))
    assert observed_revision in json.dumps(leg.get("observed"))


def test_release_accepts_byte_digest_manifest_alternative(tmp_path: Path):
    # Tests R-012 [integration]: a complete canonical byte-digest manifest is
    # the accepted alternative to an immutable revision.
    # QA-016 fix-round-2 builder update: the ledger row's model identity is
    # part of the consistency set — under a digest-pinned model the row's
    # revision part names one of the anchored canonical byte digests (the
    # recompute gate now decomposes namespace@revision and cross-checks every
    # component). Assertions untouched; edit logged for QA re-review.
    def mutate(profile):
        model = profile["provenance"]["model"]
        del model["revision"]
        model["byte_digest_manifest"] = {
            "weights.bin": "a" * 64,
            "tokenizer.json": "b" * 64,
            "config.json": "c" * 64,
        }

    def digest_row(ledger):
        ledger["rows"][0]["model_identity"] = (
            "example-org/tiny-scorer@" + "a" * 64
        )

    pkg = build_package(
        tmp_path, profile_mutator=mutate, ledger_mutator=digest_row
    )
    report = _run_release(pkg)
    assert report.verdict == VERDICT_RELEASE_PASS


def test_release_fails_on_missing_rights_status(tmp_path: Path):
    # Tests R-012 [integration]: missing rights status fails closed.
    def drop(exp):
        del exp["rights_inventory"]

    pkg = build_package(tmp_path, expectations_mutator=drop)
    report = _run_release(pkg)
    assert report.verdict == VERDICT_FAIL
    assert any("rights" in leg.get("leg_id", "") for leg in _failing_legs(report))


def test_release_fails_on_missing_presentation_manifest(tmp_path: Path):
    # Tests R-012 [integration]: missing presentation manifest fails closed.
    pkg = build_package(tmp_path)
    pkg.manifest_path.unlink()
    rewrite_json(
        pkg.expectations_path,
        lambda exp: exp["tree_files"].pop("presentation_manifest.json"),
    )
    report = _run_release(pkg)
    assert report.verdict == VERDICT_FAIL
    assert any("manifest" in leg.get("leg_id", "") for leg in _failing_legs(report))


def test_release_requires_anchored_expectations(tmp_path: Path):
    # Tests R-012/R-013 [integration]: release mode without independently
    # anchored expectations fails closed (error, not a PASS).
    pkg = build_package(tmp_path)
    with pytest.raises(schema.ColmAimsError):
        verifier.run_verifier(
            pkg.tree, mode="release", receipts_dir=pkg.receipts_dir
        )


def _build_empty_eval_package(tmp_path: Path):
    zero_counts = {
        "n_both_finite": 0,
        "n_mc_finite_ref_timeout": 0,
        "n_mc_timeout_ref_finite": 0,
        "n_both_timeout": 0,
        "n_complete": 0,
        "n_excluded_or_unpaired": 0,
        "exclusion_reason_counts": {},
        "n_pairing_population": 0,
        "n_mc_timeout": 0,
        "n_ref_timeout": 0,
    }

    def empty_cell(profile):
        cell = profile["cells"][0]
        cell["counts"] = zero_counts
        cell["rates"] = {k: None for k in cell["rates"]}
        cell["complete_pair_keys"] = []
        cell["excluded_keys"] = []
        import hashlib

        cell["pairing_population_keyset_sha256"] = hashlib.sha256(b"").hexdigest()

    return build_package(tmp_path, records=[], profile_mutator=empty_cell)


def test_empty_evaluation_errors_before_any_report(tmp_path: Path):
    # Tests R-012 [integration]: an explicitly empty evaluation dataset
    # errors before any report is emitted (typed error, no receipt).
    pkg = _build_empty_eval_package(tmp_path)
    with pytest.raises(schema.EmptyEvaluationError):
        _run_release(pkg)
    assert list(pkg.receipts_dir.iterdir()) == []


def test_tiny_nonempty_dataset_remains_valid(tmp_path: Path):
    # Tests R-012 [integration]: a genuine tiny nonempty dataset is valid —
    # the fail-closed empty gate must not reject small-but-real evidence.
    pkg = build_package(tmp_path)
    report = _run_release(pkg)
    assert report.verdict == VERDICT_RELEASE_PASS


def test_release_collects_all_leg_failures_not_just_first(tmp_path: Path):
    # Tests R-012 [integration]: collect-don't-halt — two independent broken
    # legs are BOTH reported in one run, each naming leg id, expected vs
    # observed, and a remediation class from the closed set.
    def brk(exp):
        del exp["bindings"]["calibration_identity"]
        exp["bindings"]["model"]["revision"] = "main"

    pkg = build_package(tmp_path, expectations_mutator=brk)
    report = _run_release(pkg)
    assert report.verdict == VERDICT_FAIL
    failing = _failing_legs(report)
    assert any("calibration" in leg.get("leg_id", "") for leg in failing)
    assert any("model" in leg.get("leg_id", "") for leg in failing)
    for leg in failing:
        assert leg.get("remediation_class") in REMEDIATION_CLASSES
        assert "leg_id" in leg and "expected" in leg and "observed" in leg


def test_release_recomputes_ledger_rows_and_fails_stale_stronger(tmp_path: Path):
    # Tests R-012 [integration]: every non-EXTERNAL ledger row's status is
    # recomputed; a recorded status stronger than the recomputed one fails.
    def stale(ledger):
        ledger["rows"][0]["rights_status"] = "UNVERIFIED"
        assert ledger["rows"][0]["status"] == "PASS"  # recorded stronger

    pkg = build_package(tmp_path, ledger_mutator=stale)
    report = _run_release(pkg)
    assert report.verdict == VERDICT_FAIL
    assert any(
        "clm-0001" in json.dumps(leg) for leg in _failing_legs(report)
    ), "no failing leg names the stale ledger row clm-0001"


# ---------------------------------------------------------------------------
# R-013: no self-attestation
# ---------------------------------------------------------------------------


def test_expectations_inside_tree_refused(tmp_path: Path):
    # Tests R-013 [integration]: certification requires an expectations file
    # located OUTSIDE the artifact tree it certifies.
    pkg = build_package(tmp_path)
    inside = pkg.tree / "expectations.json"
    inside.write_bytes(pkg.expectations_path.read_bytes())
    with pytest.raises(verifier.ContainmentError):
        _run_release(pkg, expectations=inside)


def test_expectations_symlink_resolving_inside_tree_refused(tmp_path: Path):
    # Tests R-013 [integration]: containment decisions use fully resolved,
    # symlink-free paths — a symlink outside the tree pointing at a file
    # inside it is refused.
    pkg = build_package(tmp_path)
    inside = pkg.tree / "expectations_real.json"
    inside.write_bytes(pkg.expectations_path.read_bytes())
    link = tmp_path / "looks_outside.json"
    link.symlink_to(inside)
    with pytest.raises(verifier.ContainmentError):
        _run_release(pkg, expectations=link)


def test_self_manifest_reaches_at_most_source_level(tmp_path: Path):
    # Tests R-013 [integration]: an artifact plus its own generated manifest
    # (timestamps and self-reported booleans included) reaches at most
    # source-level status — never a release verdict without external
    # expectations.
    pkg = build_package(tmp_path)
    report = _run_source(pkg)
    assert report.verdict == VERDICT_SOURCE_PASS
    assert report.verdict != VERDICT_RELEASE_PASS


def test_anchor_check_is_string_exact_and_works_without_git(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # Tests R-013 [integration]: the anchor check is a string-exact
    # comparison of recorded commit/ledger identities that works without a
    # git checkout (fake 40-hex commit, cwd outside any repo).
    # MA-CC-5 (fix round 3, additive setup): the object-existence check now
    # binds to an EXPLICIT source repo (never ambient cwd), so making git
    # genuinely unavailable — the scenario this test exercises — means
    # pointing that binding at a non-repo dir. The chdir is kept and a
    # non-git _SOURCE_REPO is set so the object leg is SKIPPED (not a False
    # FAIL) and the string-exact anchor alone governs. The RELEASE_PASS
    # assertion below is preserved verbatim.
    monkeypatch.chdir(tmp_path)
    from reproducibility.colm_aims_2026 import verifier as _verifier
    monkeypatch.setattr(_verifier, "_SOURCE_REPO", tmp_path)
    pkg = build_package(tmp_path, source_commit=FAKE_COMMIT)
    report = _run_release(pkg)
    assert report.verdict == VERDICT_RELEASE_PASS

    def bad_anchor(exp):
        exp["anchor"]["source_commit"] = "e" * 40

    pkg2 = build_package(tmp_path / "two", source_commit=FAKE_COMMIT,
                         expectations_mutator=bad_anchor)
    report2 = _run_release(pkg2)
    assert report2.verdict == VERDICT_FAIL


def test_verifier_cross_checks_ledger_anchor_before_consuming_expectations(
    tmp_path: Path,
):
    # Tests R-013 [integration]: expectations are anchored to the frozen
    # claim ledger; a ledger whose bytes no longer match the anchor hash is
    # refused.
    def bad_ledger_hash(exp):
        exp["anchor"]["ledger_sha256"] = "0" * 64

    pkg = build_package(tmp_path, expectations_mutator=bad_ledger_hash)
    _assert_rejected_release(pkg)


# ---------------------------------------------------------------------------
# R-014: immutability, HISTORICAL_NONCERTIFYING, legacy corpus
# ---------------------------------------------------------------------------


def test_verifier_runs_never_mutate_inputs_source_mode(tmp_path: Path):
    # Tests R-014 [integration]: artifact tree byte hashes identical before
    # and after a source-mode run.
    pkg = build_package(tmp_path)
    before = tree_hashes(pkg.tree)
    _run_source(pkg)
    assert tree_hashes(pkg.tree) == before


def test_verifier_runs_never_mutate_inputs_release_mode(tmp_path: Path):
    # Tests R-014 [integration]: same immutability for a passing release run,
    # including the expectations/ledger/rights sidecars.
    pkg = build_package(tmp_path)
    before_tree = tree_hashes(pkg.tree)
    before_sidecars = {
        p.name: p.read_bytes()
        for p in (pkg.expectations_path, pkg.ledger_path, pkg.rights_path)
    }
    _run_release(pkg)
    assert tree_hashes(pkg.tree) == before_tree
    for p in (pkg.expectations_path, pkg.ledger_path, pkg.rights_path):
        assert p.read_bytes() == before_sidecars[p.name]


def test_failing_release_run_never_mutates_inputs(tmp_path: Path):
    # Tests R-014 [integration]: a FAILING run also leaves every input byte
    # identical (current metadata is never backfilled into old bytes).
    def brk(exp):
        exp["bindings"]["model"]["revision"] = "main"

    pkg = build_package(tmp_path, expectations_mutator=brk)
    before = tree_hashes(pkg.tree)
    try:
        _run_release(pkg)
    except schema.ColmAimsError:
        pass
    assert tree_hashes(pkg.tree) == before


@pytest.mark.parametrize(
    "case,mutate",
    [
        ("dirty", lambda prov: prov["dirty_state"].update(git_dirty=True)),
        ("superseded", lambda prov: prov.update(superseded_by_producer_sha256="d" * 64)),
        ("unresolved", lambda prov: prov.update(calibration_identity="UNRESOLVED")),
    ],
)
def test_noncertifying_closure_classifications(case, mutate):
    # Tests R-014 [integration]: artifacts bound to a superseded, dirty, or
    # unresolved estimand-defining dependency closure classify
    # HISTORICAL_NONCERTIFYING.
    from tests._colm_aims_helpers import make_profile

    profile = make_profile()
    mutate(profile["provenance"])
    assert verifier.classify_certifiability(profile) == "HISTORICAL_NONCERTIFYING"


def test_clean_closure_classifies_certifiable():
    # Tests R-014 [integration]: only a producer/closure change invalidates —
    # a clean closure (and a non-closure metadata difference) stays
    # certifiable.
    from tests._colm_aims_helpers import make_profile

    profile = make_profile()
    assert verifier.classify_certifiability(profile) == "CERTIFIABLE"
    profile["cells"][0]["cell_id"] = "cell-0099"  # non-closure metadata
    assert verifier.classify_certifiability(profile) == "CERTIFIABLE"


@pytest.mark.parametrize(
    "fixture_name,family",
    [
        ("csli_captured.json", "csli"),
        ("calibration_captured.json", "calibration"),
        ("audit_card_captured.json", "audit_card"),
    ],
)
def test_legacy_profiles_parse_from_captured_bytes(fixture_name, family):
    # Tests R-014 [integration]: the known legacy profile set parses from
    # captured (sanitized) bytes of the repo's actual historical artifact
    # families — never refused merely for predating the strict schema.
    # Source: paper_exports/csli.json, paper_exports/calibration.json,
    # paper_exports/audit_card.json (captured verbatim 2026-08-19 into
    # tests/fixtures/colm_aims/legacy/; aggregates only — no quizbowl text).
    data = (LEGACY_DIR / fixture_name).read_bytes()
    parsed = verifier.parse_legacy_profile(data)
    assert parsed["legacy_family"] == family


def test_legacy_profile_refused_only_on_missing_named_invariant():
    # Tests R-014 [integration]: refusal requires a demonstrably missing
    # named invariant — the error must name it.
    # Source: paper_exports/csli.json (captured fixture).
    obj = json.loads((LEGACY_DIR / "csli_captured.json").read_text("utf-8"))
    del obj["metadata"]  # the named generation/provenance invariant
    with pytest.raises(schema.ColmAimsError) as exc:
        verifier.parse_legacy_profile(json.dumps(obj).encode("utf-8"))
    assert "metadata" in str(exc.value)


@pytest.mark.parametrize(
    "fixture_name",
    ["csli_captured.json", "calibration_captured.json", "audit_card_captured.json"],
)
def test_aggregate_only_legacy_cannot_certify_per_item_claims(fixture_name):
    # Tests R-014 [integration]: aggregate-only files cannot certify per-item
    # paired claims.
    # Source: paper_exports/ captured fixtures (see above).
    data = (LEGACY_DIR / fixture_name).read_bytes()
    parsed = verifier.parse_legacy_profile(data)
    assert verifier.legacy_certifies(parsed, "per_item_paired") is False


def test_legacy_aggregate_certifies_aggregate_claim_kind():
    # Tests R-014 [integration] (audit BL-2a): the positive case — a captured
    # legacy aggregate family DOES certify an aggregate-level claim kind, so
    # a constant `return False` cannot satisfy the suite.
    # Source: paper_exports/csli.json (captured fixture).
    data = (LEGACY_DIR / "csli_captured.json").read_bytes()
    parsed = verifier.parse_legacy_profile(data)
    assert verifier.legacy_certifies(parsed, "aggregate") is True
    assert verifier.legacy_certifies(parsed, "per_item_paired") is False


def _package_with_legacy_artifact(tmp_path: Path, ledger_mutator=None):
    """Pristine package plus the captured legacy csli.json inside the
    verified tree (declared + rights-covered so its presence is the only
    novelty)."""
    legacy_bytes = (LEGACY_DIR / "csli_captured.json").read_bytes()

    def declare(manifest):
        manifest["artifacts"].append(
            {"path": "csli.json", "role": "legacy_aggregate"}
        )

    def cover(rights):
        rights["paths"].append(
            {
                "path": "csli.json",
                "status": "VERIFIED_ALLOWED",
                "upstream_terms_basis": (
                    "synthetic capture of repo aggregate artifact"
                    " (no restricted content)"
                ),
            }
        )

    return build_package(
        tmp_path,
        extra_tree_files={"csli.json": legacy_bytes},
        manifest_mutator=declare,
        rights_mutator=cover,
        ledger_mutator=ledger_mutator,
    )


def test_verifier_parses_legacy_artifact_in_tree_without_refusal(tmp_path: Path):
    # Tests R-014 [integration] (audit BL-2b): run_verifier over a tree
    # containing the captured legacy csli.json — the run must NOT refuse it
    # merely for predating the strict schema (no typed-ingress refusal, the
    # strict profile still certifies source level) and must surface a
    # legacy/aggregate classification for it.
    # Source: paper_exports/csli.json (captured fixture).
    pkg = _package_with_legacy_artifact(tmp_path)
    report = _run_source(pkg)  # a ColmAimsError here would be a refusal
    assert report.verdict == VERDICT_SOURCE_PASS
    classification = report.classifications.get("csli.json")
    assert classification is not None, "legacy artifact must be classified"
    assert re.search(r"legacy|historical|aggregate", classification, re.IGNORECASE)


def test_release_refuses_per_item_claim_backed_by_legacy_aggregate(
    tmp_path: Path,
):
    # Tests R-014 [integration] (audit BL-2b): a ledger row claiming a
    # per-item paired estimand backed by the aggregate-only legacy artifact
    # is refused in release mode — via claim-support recomputation, not via
    # a predating-schema parse refusal.
    def add_legacy_claim(ledger):
        ledger["rows"].append(
            make_ledger_row(
                claim_id="clm-legacy-0001",
                artifact_id="csli.json",
                artifact_family="legacy_aggregate",
                provenance_class="historical_submission_artifact",
                estimand="signed_index_shift_mc_minus_ref",  # per-item paired
                status="PASS",
            )
        )

    pkg = _package_with_legacy_artifact(tmp_path, ledger_mutator=add_legacy_claim)
    report = _run_release(pkg)  # parse must not raise: refusal is a verdict
    assert report.verdict == VERDICT_FAIL
    assert any(
        "clm-legacy-0001" in json.dumps(leg) for leg in _failing_legs(report)
    ), "the legacy-backed per-item claim must be the named failure"


def test_release_mode_flags_historical_noncertifying_artifact(tmp_path: Path):
    # Tests R-014 [integration]: a dirty-closure artifact presented for
    # release cannot pass and its HISTORICAL_NONCERTIFYING classification
    # surfaces in the report.
    def dirty(profile):
        profile["provenance"]["dirty_state"]["git_dirty"] = True

    pkg = build_package(tmp_path, profile_mutator=dirty)
    report = _run_release(pkg)
    assert report.verdict == VERDICT_FAIL
    payload = json.dumps(report.legs) + json.dumps(report.classifications)
    assert "HISTORICAL_NONCERTIFYING" in payload


# ---------------------------------------------------------------------------
# R-015: aggregate + interval recomputation from retained records
# ---------------------------------------------------------------------------


def test_hand_edited_aggregate_fails_release_recomputation(tmp_path: Path):
    # Tests R-015 [unit]: every reported top-level aggregate recomputes from
    # retained per-question records; a hand-edited summary value FAILs even
    # though every byte-hash binding is consistent (mutation happens pre-hash).
    def edit(profile):
        cell = profile["cells"][0]
        cell["timing_summary_finite_only"]["signed_index_mean"] += 0.5

    pkg = build_package(tmp_path, profile_mutator=edit)
    report = _run_release(pkg)
    assert report.verdict == VERDICT_FAIL


def test_absent_records_are_non_certifying(tmp_path: Path):
    # Tests R-015 [unit]: absent per-item records leave paired claims
    # non-certifying — release cannot PASS.
    pkg = build_package(tmp_path)
    pkg.records_path.unlink()

    def strip(exp):
        exp["tree_files"].pop("records.jsonl", None)
        exp["bindings"]["input_hashes"].pop("records.jsonl", None)

    rewrite_json(pkg.expectations_path, strip)
    rewrite_json(
        pkg.manifest_path,
        lambda m: m.update(
            artifacts=[a for a in m["artifacts"] if a["path"] != "records.jsonl"]
        ),
    )
    _assert_rejected_release(pkg)


def test_interval_missing_identity_is_non_certifying(tmp_path: Path):
    # Tests R-015 [unit]: interval-bearing cells record procedure, draw
    # count, and resampling seed(s); missing interval identity leaves the
    # interval non-certifying.
    def strip_seed(profile):
        del profile["cells"][0]["interval"]["resampling_seeds"]

    pkg = build_package(tmp_path, profile_mutator=strip_seed)
    _assert_rejected_release(pkg)


def test_hand_edited_interval_fails_recomputation(tmp_path: Path):
    # Tests R-015 [unit]: interval recomputation re-runs the recorded
    # procedure deterministically; a hand-edited CI fails.
    def edit(profile):
        profile["cells"][0]["interval"]["ci"] = [-9.9, 9.9]

    pkg = build_package(tmp_path, profile_mutator=edit)
    _assert_rejected_release(pkg)


# ---------------------------------------------------------------------------
# R-017: source-contract mode ceiling and vocabulary
# ---------------------------------------------------------------------------


def test_source_mode_verdict_vocabulary_is_closed_enum():
    # Tests R-017 [unit]: closed enum whose strongest member is
    # PASS_SOURCE_ONLY.
    assert verifier.SOURCE_MODE_VERDICTS == {VERDICT_SOURCE_PASS, VERDICT_FAIL}


def test_source_mode_pristine_reaches_exactly_pass_source_only(tmp_path: Path):
    # Tests R-017 [unit]: source-contract ceiling.
    pkg = build_package(tmp_path)
    report = _run_source(pkg)
    assert report.verdict == VERDICT_SOURCE_PASS
    assert report.verdict in verifier.SOURCE_MODE_VERDICTS


def test_source_mode_summary_states_what_is_not_certified(tmp_path: Path):
    # Tests R-017 [unit]: the rendered summary states what source-only
    # verification does NOT certify, lists the checks performed, and emits no
    # release/camera-ready token and never the ACM third-party terms.
    pkg = build_package(tmp_path)
    report = _run_source(pkg)
    summary = render.render_summary(report)
    low = summary.lower()
    assert "pass_source_only" in low
    assert "not certif" in low  # "does NOT certify ..."
    # Minimum positive check set is enumerated in the summary.
    assert "profile validation" in low
    assert "typed ingress" in low
    assert "receipt" in low
    # No release/camera-ready token in source mode.
    assert "pass_release" not in low
    assert "camera-ready" not in low
    # Author-side verdicts never use ACM v1.1 third-party terms.
    assert re.search(r"\bReproduced\b", summary) is None
    assert re.search(r"\bReplicated\b", summary) is None


def test_source_mode_fails_on_schema_defect(tmp_path: Path):
    # Tests R-017 [unit]: source mode's minimum positive check set includes
    # profile validation (R-001..R-011) — a semantic defect fails source mode.
    def observed(profile):
        profile["semantic"]["observed_open_ended"] = True

    pkg = build_package(tmp_path, profile_mutator=observed)
    _assert_rejected_source(pkg)


# ---------------------------------------------------------------------------
# R-019: adversarial fixture corpus (8 handoff + 5 review-added), each with
# a nearest-true sibling, each mapped to the mode(s) where it must FAIL
# ---------------------------------------------------------------------------


def _mut_constructed_as_observed(tmp_path):
    return build_package(
        tmp_path,
        profile_mutator=lambda p: p["semantic"].update(observed_open_ended=True),
    )


def _mut_denominator_mismatch(tmp_path):
    def mutate(profile):
        profile["cells"][0]["rates"] = {
            "rate_both_finite": 3 / 8,
            "rate_mc_finite_ref_timeout": 1 / 8,
            "rate_mc_timeout_ref_finite": 1 / 8,
            "rate_both_timeout": 1 / 8,
        }

    return build_package(tmp_path, profile_mutator=mutate)


def _mut_timeout_mismatch(tmp_path):
    def mutate(profile):
        # Boundary miscount: the stop_step == horizon reference timeout of
        # itm-0004 counted as finite. Self-consistent identities, wrong
        # against the records.
        cell = profile["cells"][0]
        cell["counts"] = dict(cell["counts"])
        cell["counts"].update(
            n_both_finite=4,
            n_mc_finite_ref_timeout=0,
            n_ref_timeout=1,
        )
        cell["rates"] = {
            "rate_both_finite": 4 / 6,
            "rate_mc_finite_ref_timeout": 0.0,
            "rate_mc_timeout_ref_finite": 1 / 6,
            "rate_both_timeout": 1 / 6,
        }

    return build_package(tmp_path, profile_mutator=mutate)


def _mut_binding_mismatch(tmp_path):
    return build_package(
        tmp_path,
        expectations_mutator=lambda exp: exp["bindings"]["model"].update(
            revision="9" * 40
        ),
    )


def _mut_historical_substitution(tmp_path):
    # Artifact produced at an old commit substituted under the current
    # anchor: profile/bindings say old commit, anchor says current.
    return build_package(
        tmp_path,
        profile_mutator=lambda p: p["provenance"]["dirty_state"].update(
            source_commit="0" * 40
        ),
    )


def _mut_empty_evaluation(tmp_path):
    return _build_empty_eval_package(tmp_path)


def _mut_unbound_calibration(tmp_path):
    return build_package(
        tmp_path,
        profile_mutator=lambda p: p["provenance"].update(
            calibration_identity="UNRESOLVED"
        ),
    )


def _mut_unverified_rights(tmp_path):
    def mutate(rights):
        rights["paths"][1]["status"] = "UNVERIFIED"

    return build_package(tmp_path, rights_mutator=mutate)


def _mut_stale_pass_ledger_row(tmp_path):
    def mutate(ledger):
        ledger["rows"][0]["rights_status"] = "UNVERIFIED"

    return build_package(tmp_path, ledger_mutator=mutate)


def _mut_zero_artifact_tree(tmp_path):
    pkg = build_package(tmp_path)
    for p in list(pkg.tree.iterdir()):
        p.unlink()
    return pkg


def _mut_empty_ledger(tmp_path):
    return build_package(tmp_path, ledger_mutator=lambda l: l.update(rows=[]))


def _mut_empty_presentation_manifest(tmp_path):
    return build_package(
        tmp_path, manifest_mutator=lambda m: m.update(artifacts=[])
    )


def _mut_oversized_tolerance(tmp_path):
    def mutate(profile):
        profile["numerical_tolerance"] = 0.25
        cell = profile["cells"][0]
        cell["estimand"]["numerical_tolerance"] = 0.25
        cell["estimand_digest"] = expected_estimand_digest(cell["estimand"])

    return build_package(tmp_path, profile_mutator=mutate)


# --- nearest-true sibling builders (audit ADV-4): each exercises the SAME
# knob as its adversarial twin, at its minimal compliant setting -------------


def _sib_pristine(tmp_path):
    # Minimal compliant variant for the schema-level fixtures: the pristine
    # package already sits ON the exercised boundaries — the exact pinned
    # semantic block (constructed_as_observed), rates recomputed over
    # n_complete (denominator_mismatch), and records with stops on BOTH sides
    # of the timeout boundary: itm-0003 stop=5=horizon-1 finite,
    # itm-0006 stop=6=horizon timeout (timeout_mismatch).
    return build_package(tmp_path)


def _sib_consistent_alt_revision(tmp_path):
    # hash/model/split mismatch twin: the same binding knob moved to a
    # DIFFERENT valid immutable revision, consistently on both sides.
    # QA-016 fix-round-2 builder update: "consistently" includes the ledger
    # row's model identity — the recompute gate now decomposes
    # namespace@revision and requires the row's revision to equal the
    # verified provenance revision. Assertions untouched; edit logged.
    return build_package(
        tmp_path,
        profile_mutator=lambda p: p["provenance"]["model"].update(
            revision="9" * 40
        ),
        ledger_mutator=lambda l: l["rows"][0].update(
            model_identity="example-org/tiny-scorer@" + "9" * 40
        ),
    )


def _sib_current_commit_explicit(tmp_path):
    # historical/current substitution twin: the dirty-state commit knob set
    # EXPLICITLY to the anchored current commit.
    commit = repo_head_commit()
    return build_package(
        tmp_path,
        source_commit=commit,
        profile_mutator=lambda p: p["provenance"]["dirty_state"].update(
            source_commit=commit
        ),
    )


def _sib_single_record(tmp_path):
    # empty-evaluation twin: the SMALLEST nonempty evaluation (one complete
    # both-finite pair, hand-computed cell; CI of a constant statistic is
    # degenerate [-2.0, -2.0] under any resample plan).
    records = [make_record("itm-0001", 1, 3)]

    def mut(profile):
        cell = profile["cells"][0]
        cell["counts"] = {
            "n_both_finite": 1,
            "n_mc_finite_ref_timeout": 0,
            "n_mc_timeout_ref_finite": 0,
            "n_both_timeout": 0,
            "n_complete": 1,
            "n_excluded_or_unpaired": 0,
            "exclusion_reason_counts": {},
            "n_pairing_population": 1,
            "n_mc_timeout": 0,
            "n_ref_timeout": 0,
        }
        cell["rates"] = {
            "rate_both_finite": 1.0,
            "rate_mc_finite_ref_timeout": 0.0,
            "rate_mc_timeout_ref_finite": 0.0,
            "rate_both_timeout": 0.0,
        }
        cell["timing_summary_finite_only"] = {
            "conditional_on": "n_both_finite",
            "estimand": "signed_index_shift_mc_minus_ref",
            "n": 1,
            "signed_index_mean": -2.0,
            "signed_index_median": -2.0,
            "absolute_index_mean": 2.0,
            "absolute_index_median": 2.0,
        }
        cell["timing_summary_sentinel_coded_historical"] = {
            "convention": "timeout_coded_as_horizon",
            "n": 1,
            "signed_index_mean": -2.0,
            "signed_index_median": -2.0,
        }
        cell["interval"] = {
            "procedure": "percentile_bootstrap",
            "draw_count": 100,
            "resampling_seeds": [1],
            "statistic": "signed_index_mean",
            "ci": [-2.0, -2.0],
        }
        cell["complete_pair_keys"] = ["itm-0001"]
        cell["excluded_keys"] = []
        cell["pairing_population_keyset_sha256"] = keyset_sha256(["itm-0001"])

    return build_package(tmp_path, records=records, profile_mutator=mut)


def _sib_rebound_calibration(tmp_path):
    # unbound-calibration twin: the same knob explicitly bound to a different
    # RESOLVED identity (provenance + estimand + digest kept consistent).
    # QA-003 fix-round-1 builder update: the ledger row's calibration
    # identity is part of the same consistency set — the recompute gate now
    # RE-DERIVES row identities from the verified provenance, so the minimal
    # compliant variant rebinds all three sites of the one knob together.
    # (Assertions untouched; edit logged for QA round 2.)
    def mut(profile):
        profile["provenance"]["calibration_identity"] = "cal-0002"
        est = profile["cells"][0]["estimand"]
        est["calibration_identity"] = "cal-0002"
        profile["cells"][0]["estimand_digest"] = expected_estimand_digest(est)

    def rebind_row(ledger):
        ledger["rows"][0]["calibration_identity"] = "cal-0002"

    return build_package(tmp_path, profile_mutator=mut, ledger_mutator=rebind_row)


def _sib_verified_rights_explicit(tmp_path):
    # unverified-rights twin: the same row explicitly VERIFIED_ALLOWED with a
    # concrete upstream terms basis.
    def mut(rights):
        rights["paths"][1]["status"] = "VERIFIED_ALLOWED"
        rights["paths"][1]["upstream_terms_basis"] = (
            "QANTA permissioned-aggregation-without-redistribution"
        )

    return build_package(tmp_path, rights_mutator=mut)


def _sib_honest_unverified_row(tmp_path):
    # stale-PASS twin: an honestly-recorded row (recorded status NOT stronger
    # than recomputed). DECISION: an honest UNVERIFIED, non-headline row with
    # a named blocking task does not block release — only stale-stronger rows
    # fail R-012's recompute gate.
    def mut(ledger):
        ledger["rows"].append(
            make_ledger_row(
                claim_id="clm-0002",
                status="UNVERIFIED",
                headline_eligible=False,
                blocking_task="task-verify-clm-0002",
            )
        )

    return build_package(tmp_path, ledger_mutator=mut)


def _sib_minimal_tree(tmp_path):
    # zero-artifact-tree twin: the minimal nonempty tree (no optional sealed
    # payload — exactly the strict profile, records, and manifest).
    return build_package(tmp_path, include_sentinel_payload=False)


def _sib_single_row_ledger(tmp_path):
    # empty-ledger twin: exactly ONE retained claim row (the R-033 boundary).
    return build_package(
        tmp_path, ledger_mutator=lambda l: l.update(rows=[make_ledger_row()])
    )


def _sib_single_artifact_manifest(tmp_path):
    # empty-manifest twin: exactly ONE declared artifact (the R-033
    # boundary), remaining files admitted via the explicit allowlist.
    def mut(manifest):
        manifest["artifacts"] = [{"path": "profile.json", "role": "strict_profile"}]
        manifest["allowlist_undeclared"] = ["records.jsonl", "sealed-notes.bin"]

    return build_package(tmp_path, manifest_mutator=mut)


def _sib_tolerance_at_max(tmp_path):
    # oversized-tolerance twin: the declared tolerance EXACTLY at the pinned
    # maximum, digest recomputed (ADV-3 discipline).
    max_tol = schema.MAX_ADMISSIBLE_TOLERANCE

    def mut(profile):
        profile["numerical_tolerance"] = max_tol
        est = profile["cells"][0]["estimand"]
        est["numerical_tolerance"] = max_tol
        profile["cells"][0]["estimand_digest"] = expected_estimand_digest(est)

    return build_package(tmp_path, profile_mutator=mut)


ADVERSARIAL_CORPUS = [
    # (name, builder, sibling_builder, modes in which it MUST fail,
    #  gate token pinning WHICH gate fired — audit ADV-5 — or None)
    ("constructed_as_observed", _mut_constructed_as_observed, _sib_pristine,
     ("source", "release"), None),
    ("denominator_mismatch", _mut_denominator_mismatch, _sib_pristine,
     ("source", "release"), None),
    ("timeout_mismatch", _mut_timeout_mismatch, _sib_pristine,
     ("source", "release"), None),
    ("hash_model_split_mismatch", _mut_binding_mismatch,
     _sib_consistent_alt_revision, ("release",), None),
    ("historical_current_substitution", _mut_historical_substitution,
     _sib_current_commit_explicit, ("release",), "commit"),
    ("empty_evaluation", _mut_empty_evaluation, _sib_single_record,
     ("source", "release"), None),
    ("unbound_calibration", _mut_unbound_calibration, _sib_rebound_calibration,
     ("release",), None),
    ("unverified_rights_inclusion", _mut_unverified_rights,
     _sib_verified_rights_explicit, ("release",), None),
    ("stale_pass_ledger_row", _mut_stale_pass_ledger_row,
     _sib_honest_unverified_row, ("release",), None),
    ("zero_artifact_tree", _mut_zero_artifact_tree, _sib_minimal_tree,
     ("source", "release"), None),
    ("empty_ledger", _mut_empty_ledger, _sib_single_row_ledger,
     ("release",), "ledger"),
    ("empty_presentation_manifest", _mut_empty_presentation_manifest,
     _sib_single_artifact_manifest, ("release",), "manifest"),
    ("oversized_tolerance", _mut_oversized_tolerance, _sib_tolerance_at_max,
     ("source", "release"), "tolerance"),
]

_CORPUS_IDS = [entry[0] for entry in ADVERSARIAL_CORPUS]


@pytest.mark.parametrize(
    "name,builder,sibling_builder,fail_modes,gate_token",
    ADVERSARIAL_CORPUS,
    ids=_CORPUS_IDS,
)
def test_adversarial_fixture_fails_in_mapped_modes(
    tmp_path: Path, name, builder, sibling_builder, fail_modes, gate_token
):
    # Tests R-019 [unit]: each of the 13 adversarial fixtures FAILs in every
    # verifier mode it is mapped to; for the ADV-5 set the failing gate is
    # named, not just any rejection.
    pkg = builder(tmp_path)
    for mode in fail_modes:
        if mode == "source":
            _assert_rejected_source(pkg, expect_token=gate_token)
        else:
            _assert_rejected_release(pkg, expect_token=gate_token)


@pytest.mark.parametrize(
    "name,builder,sibling_builder,fail_modes,gate_token",
    ADVERSARIAL_CORPUS,
    ids=_CORPUS_IDS,
)
def test_nearest_true_sibling_passes(
    tmp_path: Path, name, builder, sibling_builder, fail_modes, gate_token
):
    # Tests R-019 [unit]: each adversarial fixture is paired with its OWN
    # nearest-true sibling — the minimal compliant variant of the same knob
    # (audit ADV-4) — which must PASS in the same mode(s).
    sibling = sibling_builder(tmp_path / "sibling")
    for mode in fail_modes:
        if mode == "source":
            report = _run_source(sibling)
            assert report.verdict == VERDICT_SOURCE_PASS, (name, mode)
        else:
            report = _run_release(sibling)
            assert report.verdict == VERDICT_RELEASE_PASS, (name, mode)


# ---------------------------------------------------------------------------
# R-033: no vacuous verdicts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["source", "release"])
def test_zero_candidate_artifacts_is_typed_error_naming_path_and_layout(
    tmp_path: Path, mode
):
    # Tests R-033 [unit]: both modes fail with a typed error naming the
    # resolved path and expected layout when zero candidate artifacts exist.
    pkg = _mut_zero_artifact_tree(tmp_path)
    with pytest.raises(verifier.VacuousInputError) as exc:
        verifier.run_verifier(
            pkg.tree,
            mode=mode,
            receipts_dir=pkg.receipts_dir,
            expectations=pkg.expectations_path if mode == "release" else None,
        )
    msg = str(exc.value)
    assert str(pkg.tree.resolve()) in msg or str(pkg.tree) in msg
    assert "profile.json" in msg  # names the expected layout


def test_pass_verdict_enumerates_validated_artifacts(tmp_path: Path):
    # Tests R-033 [unit]: any PASS-class verdict requires >= 1 validated
    # artifact enumerated in the report/rendered summary.
    pkg = build_package(tmp_path)
    report = _run_source(pkg)
    assert report.verdict == VERDICT_SOURCE_PASS
    assert len(report.validated_artifacts) >= 1
    assert any("profile.json" in a for a in report.validated_artifacts)
    summary = render.render_summary(report)
    assert "profile.json" in summary


# ---------------------------------------------------------------------------
# R-035: presentation-manifest reconciliation, both directions
# ---------------------------------------------------------------------------


def test_manifest_declared_but_absent_fails(tmp_path: Path):
    # Tests R-035 [unit]: manifest-declared-but-absent -> FAIL.
    def declare_ghost(manifest):
        manifest["artifacts"].append({"path": "missing.json", "role": "ghost"})

    pkg = build_package(tmp_path, manifest_mutator=declare_ghost)
    report = _run_release(pkg)
    assert report.verdict == VERDICT_FAIL
    assert any("missing.json" in json.dumps(leg) for leg in _failing_legs(report))


def test_present_but_undeclared_fails_without_allowlist(tmp_path: Path):
    # Tests R-035 [unit]: present-but-undeclared -> FAIL absent an explicit
    # per-file allowlist entry (rights row present, so the manifest leg is
    # the only defect).
    def cover_stray(rights):
        rights["paths"].append(
            {
                "path": "stray.txt",
                "status": "VERIFIED_ALLOWED",
                "upstream_terms_basis": "synthetic test fixture generated in-repo",
            }
        )

    pkg = build_package(
        tmp_path,
        extra_tree_files={"stray.txt": b"stray-but-rights-covered\n"},
        rights_mutator=cover_stray,
    )
    report = _run_release(pkg)
    assert report.verdict == VERDICT_FAIL
    assert any("stray.txt" in json.dumps(leg) for leg in _failing_legs(report))


def test_present_but_undeclared_passes_with_allowlist_entry(tmp_path: Path):
    # Tests R-035 [unit]: the explicit per-file allowlist entry admits the
    # undeclared file.
    def cover_stray(rights):
        rights["paths"].append(
            {
                "path": "stray.txt",
                "status": "VERIFIED_ALLOWED",
                "upstream_terms_basis": "synthetic test fixture generated in-repo",
            }
        )

    pkg = build_package(
        tmp_path,
        extra_tree_files={"stray.txt": b"stray-but-allowlisted\n"},
        rights_mutator=cover_stray,
        manifest_mutator=lambda m: m.update(allowlist_undeclared=["stray.txt"]),
    )
    report = _run_release(pkg)
    assert report.verdict == VERDICT_RELEASE_PASS


def test_rights_inventory_must_cover_every_file_found(tmp_path: Path):
    # Tests R-035 [unit]: the rights inventory covers every file FOUND, not
    # merely every file declared — an allowlisted stray file without a rights
    # row still fails.
    pkg = build_package(
        tmp_path,
        extra_tree_files={"stray.txt": b"stray-without-rights\n"},
        manifest_mutator=lambda m: m.update(allowlist_undeclared=["stray.txt"]),
    )
    report = _run_release(pkg)
    assert report.verdict == VERDICT_FAIL
    assert any("stray.txt" in json.dumps(leg) for leg in _failing_legs(report))
