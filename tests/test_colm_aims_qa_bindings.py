"""QA fix-round-1 regression suite — binding-leg interior (QA-001, QA-002).

QA-001: binding legs carry TWO obligations — the observed value must be
admissible in its own right AND match the anchored expectation. Artifact-side
mutations that the expectations mirror (author-controlled proxy) must still
FAIL via the validity predicate.
QA-002: every R-012 provenance noun is covered by the declarative
REQUIRED_PROVENANCE_FIELDS table; dropping any enumerated sub-field fails its
table leg even when the expectations mirror the drop.
Spec: .correctless/specs/camera-ready-aims-evidence.md (R-012)
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from reproducibility.colm_aims_2026 import verifier
from tests._colm_aims_helpers import (  # noqa: F401  (colm_no_network is an autouse fixture)
    REMEDIATION_CLASSES,
    VERDICT_FAIL,
    build_package,
    colm_no_network,
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


def _failing_with_id(report, leg_id):
    return [leg for leg in _failing(report) if leg.get("leg_id") == leg_id]


# ---------------------------------------------------------------------------
# QA-001 meta-test: validator registry covers every binding key
# ---------------------------------------------------------------------------


def test_every_binding_key_has_a_registered_validator():
    # QA-001 class fix: BINDING_VALIDATORS registry == BINDING_KEYS, all
    # callable — no binding leg can silently lack a validity predicate.
    assert set(verifier.BINDING_VALIDATORS) == set(verifier.BINDING_KEYS)
    for key, validator in verifier.BINDING_VALIDATORS.items():
        assert callable(validator), key


def test_validators_accept_admissible_and_reject_inadmissible_scalars():
    # QA-001: the shared admissibility core rejects None/empty/UNRESOLVED.
    validator = verifier.BINDING_VALIDATORS["calibration_identity"]
    assert validator("cal-0001") is None
    for bad in (None, "", "UNRESOLVED"):
        assert validator(bad) is not None, bad


# ---------------------------------------------------------------------------
# QA-001 red fixtures: artifact-side mutation for each of the 13 binding keys.
# The mutation runs PRE-hash, so the expectations MIRROR the inadmissible
# value — mirror-equality holds and only the validity predicate can fire.
# ---------------------------------------------------------------------------

QA1_PROV_MUTATIONS = [
    ("producer", lambda prov: prov.update(producer_sha256="UNRESOLVED")),
    ("semantic_command", lambda prov: prov.update(semantic_command=[])),
    ("seeds", lambda prov: prov.update(seeds=[])),
    (
        "dirty_state",
        lambda prov: prov["dirty_state"].update(source_commit=""),
    ),
    (
        "splits",
        lambda prov: prov["splits"]["eval"].update(keyset_sha256=""),
    ),
    (
        "calibration_identity",
        lambda prov: prov.update(calibration_identity=""),
    ),
    (
        "continuation_identity",
        lambda prov: prov.update(continuation_identity="UNRESOLVED"),
    ),
    ("input_hashes", lambda prov: prov.update(input_sha256={})),
    (
        "split_metadata_sha256",
        lambda prov: prov.update(split_metadata_sha256="UNRESOLVED"),
    ),
    ("mc_build", lambda prov: prov.update(mc_build={})),
    (
        "model",
        lambda prov: prov["model"].update(weights_sha256="UNRESOLVED"),
    ),
    ("runtime_packages", lambda prov: prov.update(runtime_packages={})),
]


@pytest.mark.parametrize(
    "binding_key,mutate",
    QA1_PROV_MUTATIONS,
    ids=[key for key, _ in QA1_PROV_MUTATIONS],
)
def test_artifact_side_inadmissible_binding_fails_despite_mirror(
    tmp_path: Path, binding_key, mutate
):
    # QA-001 [R-012]: the artifact-side value is inadmissible; the
    # expectations mirror it byte-for-byte — the leg must fail as a distinct
    # ARTIFACT_DEFECT (never PASS via mirror-equality, never
    # MISSING_EXPECTATION).
    pkg = build_package(tmp_path, binding_mutator=mutate)
    report = _run_release(pkg)
    assert report.verdict == VERDICT_FAIL
    legs = _failing_with_id(report, f"binding_{binding_key}")
    assert legs, f"no failing validity leg for binding_{binding_key}"
    assert legs[0]["remediation_class"] == "ARTIFACT_DEFECT"


def test_artifact_side_empty_profile_id_fails_schema_profile_binding(
    tmp_path: Path,
):
    # QA-001 [R-012]: the schema_profile binding key — profile-side mutation
    # (profile_id emptied) mirrored into expectations still fails.
    pkg = build_package(
        tmp_path, profile_mutator=lambda p: p.update(profile_id="")
    )
    report = _run_release(pkg)
    assert report.verdict == VERDICT_FAIL
    legs = _failing_with_id(report, "binding_schema_profile")
    assert legs, "no failing validity leg for binding_schema_profile"
    assert legs[0]["remediation_class"] == "ARTIFACT_DEFECT"


def test_validity_failure_reports_before_missing_expectation(tmp_path: Path):
    # QA-001: with BOTH defects present (inadmissible artifact value AND the
    # expectation deleted), the admissibility obligation wins — the leg is an
    # ARTIFACT_DEFECT, not a MISSING_EXPECTATION.
    pkg = build_package(
        tmp_path,
        binding_mutator=lambda prov: prov.update(seeds=[]),
        expectations_mutator=lambda exp: exp["bindings"].pop("seeds"),
    )
    report = _run_release(pkg)
    legs = _failing_with_id(report, "binding_seeds")
    assert legs
    assert legs[0]["remediation_class"] == "ARTIFACT_DEFECT"


# ---------------------------------------------------------------------------
# QA-003 class convention applied to binding legs: substitution-negative
# tests — the expectation replaced with a DIFFERENT VALID value must still
# flip the leg (mirror-equality is one obligation, not the only one).
# ---------------------------------------------------------------------------

BINDING_SUBSTITUTIONS = [
    ("seeds", lambda exp: exp["bindings"].update(seeds=[1, 2, 4])),
    (
        "semantic_command",
        lambda exp: exp["bindings"].update(
            semantic_command=["python", "scripts/fake_producer.py", "--seed", "2"]
        ),
    ),
    (
        "split_metadata_sha256",
        lambda exp: exp["bindings"].update(split_metadata_sha256="e" * 64),
    ),
    (
        "runtime_packages",
        lambda exp: exp["bindings"].update(
            runtime_packages={"python": "3.13", "numpy": "2.4.6"}
        ),
    ),
]


@pytest.mark.parametrize(
    "binding_key,mutate",
    BINDING_SUBSTITUTIONS,
    ids=[key for key, _ in BINDING_SUBSTITUTIONS],
)
def test_binding_substitution_negative(tmp_path: Path, binding_key, mutate):
    # QA-003 class fix (binding surface): a recorded binding replaced with a
    # different VALID value still fails the leg — the gate compares against
    # the anchored expectation, not against validity alone.
    pkg = build_package(tmp_path, expectations_mutator=mutate)
    report = _run_release(pkg)
    assert report.verdict == VERDICT_FAIL
    legs = _failing_with_id(report, f"binding_{binding_key}")
    assert legs, f"substitution did not flip binding_{binding_key}"
    assert legs[0]["expected"] != legs[0]["observed"]


# ---------------------------------------------------------------------------
# QA-002 meta-test: the declarative table covers every R-012 noun
# ---------------------------------------------------------------------------

R012_NOUN_PATHS = {
    "producer_entrypoint",
    "producer_sha256",
    "helper_sha256s",
    "semantic_command",
    "seeds",
    "dirty_state.git_dirty",
    "dirty_state.source_commit",
    "splits.fit.name",
    "splits.fit.count",
    "splits.fit.keyset_sha256",
    "splits.eval.name",
    "splits.eval.count",
    "splits.eval.keyset_sha256",
    "splits.zero_overlap",
    "calibration_identity",
    "continuation_identity",
    "input_sha256",
    "split_metadata_sha256",
    "mc_build.built_after_split",
    "mc_build.coverage_rate",
    "mc_build.retention_policy",
    "mc_build.retained_count",
    "model.repository_namespace",
    "model.weights_sha256",
    "model.tokenizer_config_sha256",
    "model.dtype",
    "model.device_class",
    "model.numerical_settings",
    "runtime_packages",
}


def test_required_provenance_table_covers_every_r012_noun():
    # QA-002 class fix: the table is transcribed one-to-one from R-012's
    # enumeration; every row names a registered predicate, a leg id, and a
    # remediation class from the closed set.
    table_paths = {row[0] for row in verifier.REQUIRED_PROVENANCE_FIELDS}
    assert table_paths == R012_NOUN_PATHS
    for dotted, predicate_name, leg_id, remediation in (
        verifier.REQUIRED_PROVENANCE_FIELDS
    ):
        assert predicate_name in verifier._FIELD_PREDICATES, dotted
        assert callable(verifier._FIELD_PREDICATES[predicate_name])
        assert leg_id, dotted
        assert remediation in REMEDIATION_CLASSES, dotted


# ---------------------------------------------------------------------------
# QA-002 red fixtures: one per enumerated sub-field the audit found droppable
# ---------------------------------------------------------------------------

QA2_FIELD_DROPS = [
    (
        "model.repository_namespace",
        "model_identity_completeness",
        lambda prov: prov["model"].pop("repository_namespace"),
    ),
    (
        "model.tokenizer_config_sha256",
        "model_identity_completeness",
        lambda prov: prov["model"].pop("tokenizer_config_sha256"),
    ),
    (
        "model.dtype",
        "model_identity_completeness",
        lambda prov: prov["model"].pop("dtype"),
    ),
    (
        "model.device_class",
        "model_identity_completeness",
        lambda prov: prov["model"].pop("device_class"),
    ),
    (
        "model.numerical_settings",
        "model_identity_completeness",
        lambda prov: prov["model"].pop("numerical_settings"),
    ),
    (
        "splits.fit",
        "splits_fit_recorded",
        lambda prov: prov["splits"].pop("fit"),
    ),
    (
        "splits.fit.name",
        "splits_fit_recorded",
        lambda prov: prov["splits"]["fit"].pop("name"),
    ),
    (
        "splits.fit.count",
        "splits_fit_recorded",
        lambda prov: prov["splits"]["fit"].pop("count"),
    ),
    (
        "splits.fit.keyset_sha256",
        "splits_fit_recorded",
        lambda prov: prov["splits"]["fit"].pop("keyset_sha256"),
    ),
    (
        "splits.eval.name",
        "splits_eval_recorded",
        lambda prov: prov["splits"]["eval"].pop("name"),
    ),
]


@pytest.mark.parametrize(
    "dropped,leg_id,mutate",
    QA2_FIELD_DROPS,
    ids=[dropped for dropped, _, _ in QA2_FIELD_DROPS],
)
def test_dropped_provenance_subfield_fails_its_table_leg(
    tmp_path: Path, dropped, leg_id, mutate
):
    # QA-002 [R-012]: dropping any enumerated sub-field from the provenance
    # (mirrored into expectations — mirror-equality holds) fails the
    # table-driven leg naming the dotted path.
    pkg = build_package(tmp_path, binding_mutator=mutate)
    report = _run_release(pkg)
    assert report.verdict == VERDICT_FAIL
    legs = _failing_with_id(report, leg_id)
    assert legs, f"no failing table leg {leg_id!r} after dropping {dropped!r}"
    assert dropped.split(".")[0] in json.dumps(legs[0].get("observed"))
