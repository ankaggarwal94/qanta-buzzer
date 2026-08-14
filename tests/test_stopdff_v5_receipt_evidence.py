"""Fail-closed regressions for content inventories and receipt evidence bytes."""
from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.stopdff_v5.content_manifest import (  # noqa: E402
    validate_bound_content_manifest,
)
from scripts.stopdff_v5.identity import (  # noqa: E402
    build_manifest,
    sha256_bytes,
    sha256_file,
)
from scripts.stopdff_v5.manifests import (  # noqa: E402
    ADAPTER_SCORING_SPEC,
    RAW_INPUT_ROLES,
    model_snapshot_identity,
    raw_input_identity,
    run_spec_identity,
    source_manifest_identity,
)
from scripts.stopdff_v5.profile import (  # noqa: E402
    SMOKE_REPLICATES,
    cell_key_str,
    smoke_cells,
)
from scripts.stopdff_v5.receipt_evidence import (  # noqa: E402
    DETERMINISM_FILES,
    MUTATION_ROSTER,
    build_prerequisite_evidence,
    prerequisite_evidence_bytes,
    validate_prerequisite_evidence,
    verify_prerequisite_evidence_bytes,
)
from scripts.stopdff_v5 import writers  # noqa: E402


def _bindings() -> dict[str, str]:
    return {
        "source_manifest_id": "1" * 64,
        "raw_input_bundle_id": "2" * 64,
        "model_snapshot_id": "3" * 64,
        "adapter_bundle_id": "4" * 64,
        "fvi_study_id": "5" * 64,
        "environment_contract_id": "6" * 64,
    }


def _determinism_bindings() -> dict[str, str]:
    return {
        key: _bindings()[key]
        for key in (
            "source_manifest_id",
            "raw_input_bundle_id",
            "model_snapshot_id",
            "adapter_bundle_id",
        )
    }


def _adapter_manifest() -> tuple[dict, dict[str, str]]:
    hashes = {
        name: hashlib.sha256(name.encode("utf-8")).hexdigest()
        for name in DETERMINISM_FILES
    }
    identity = {
        "kind": "adapter_bundle",
        "source_manifest_id": "1" * 64,
        "raw_input_bundle_id": "2" * 64,
        "model_snapshot_id": "3" * 64,
        "fit_rows_sha256": hashes["fit_rows.jsonl.gz"],
        "eval_rows_sha256": hashes["eval_rows.jsonl.gz"],
        "calibration_sha256": hashes["calibration.json"],
        "mc_retention_evidence": {
            "build_metadata_sha256": hashes["build_metadata.json"],
        },
    }
    manifest = build_manifest(identity)
    bindings = _determinism_bindings()
    bindings["adapter_bundle_id"] = manifest["id"]
    return manifest, bindings


def _mutation_results() -> list[dict]:
    return [
        {
            "mutation": name,
            "expected": "PASS" if index == 0 else "REJECT",
            "passed_check": index == 0,
            "ok": True,
            "errors": [],
        }
        for index, name in enumerate(MUTATION_ROSTER)
    ]


def _source_execution(source_id: str) -> dict[str, str]:
    return {
        "environment": "local_clean_worktree",
        "executing_source_manifest_id": source_id,
        "runtime_source_manifest_id": source_id,
    }


def _build_execution(
    *,
    bindings: dict[str, str],
    hashes: dict[str, str],
    execution_id: str,
    subdir: str,
) -> dict:
    return {
        "environment": "local_process",
        "execution_id": execution_id,
        "adapter_subdir": subdir,
        **bindings,
        "cached": False,
        "output_sha256": hashes,
    }


def _canonical_smoke_run_spec(
    bindings: dict[str, str],
    *,
    replicate_count: int = SMOKE_REPLICATES,
) -> dict:
    """Build a genuine canonical smoke run-spec manifest via the shared builder.

    This is the same ``run_spec_identity`` the pipeline mints and the checker
    validates, so a receipt embedding it satisfies the reused canonical
    run-spec/profile checks.
    """
    spec_identity = run_spec_identity(
        source_manifest_id=bindings["source_manifest_id"],
        raw_input_bundle_id=bindings["raw_input_bundle_id"],
        model_snapshot_id=bindings["model_snapshot_id"],
        adapter_bundle_id=bindings["adapter_bundle_id"],
        fvi_study_id=bindings["fvi_study_id"],
        bootstrap_plan_id="7" * 64,
        environment_contract_id=bindings["environment_contract_id"],
        resource_summary_id="8" * 64,
        fvi_selected={"tolerance": "1e-10", "max_iterations": 200},
        replicate_count=replicate_count,
        profile_variant="smoke",
        myopic_artifact_sha256="9" * 64,
        producer_hashes={"checker.py": "a" * 64, "sweep.py": "b" * 64},
        prerequisite_receipts={},
    )
    return build_manifest(spec_identity)


def _smoke_evidence() -> tuple[dict, dict[str, str]]:
    bindings = _bindings()
    run_spec = _canonical_smoke_run_spec(bindings)
    expected_cell_keys = sorted(cell_key_str(cell) for cell in smoke_cells())
    evidence = build_prerequisite_evidence(
        gate="smoke",
        bindings=bindings,
        details={
            "run_spec": run_spec,
            "aggregate": {
                "profile_variant": "smoke",
                "run_spec_id": run_spec["id"],
                "adapter_bundle_id": bindings["adapter_bundle_id"],
                "fvi_study_id": bindings["fvi_study_id"],
                "requested": len(expected_cell_keys),
                "completed": len(expected_cell_keys),
                "failed": 0,
                "skipped": 0,
                "release_status": "VALID",
                "release_reasons": [],
                "expected_cell_keys": expected_cell_keys,
            },
        },
    )
    return evidence, bindings


def _gate_fixture(gate: str) -> tuple[dict, dict[str, str]]:
    if gate == "smoke":
        return _smoke_evidence()
    if gate == "mutation":
        bindings = _bindings()
        return build_prerequisite_evidence(
            gate=gate,
            bindings=bindings,
            details={
                "source_execution": _source_execution(
                    bindings["source_manifest_id"]
                ),
                "results": _mutation_results(),
            },
        ), bindings
    manifest, bindings = _adapter_manifest()
    hashes = {
        name: manifest["identity"].get(
            {
                "fit_rows.jsonl.gz": "fit_rows_sha256",
                "eval_rows.jsonl.gz": "eval_rows_sha256",
                "calibration.json": "calibration_sha256",
            }.get(name, "")
        )
        for name in DETERMINISM_FILES
    }
    hashes["build_metadata.json"] = manifest["identity"][
        "mc_retention_evidence"
    ]["build_metadata_sha256"]
    return build_prerequisite_evidence(
        gate=gate,
        bindings=bindings,
        details={
            "source_execution": _source_execution(
                bindings["source_manifest_id"]
            ),
            "first_build_execution": _build_execution(
                bindings=bindings,
                hashes=hashes,
                execution_id="local-first",
                subdir="adapter_build_a",
            ),
            "second_build_execution": _build_execution(
                bindings=bindings,
                hashes=hashes,
                execution_id="local-second",
                subdir="adapter_build_b",
            ),
            "first_adapter_manifest": manifest,
            "second_adapter_manifest": manifest,
            "first_file_sha256": hashes,
            "second_file_sha256": hashes,
        },
    ), bindings


@pytest.mark.parametrize("gate", ["smoke", "mutation", "determinism"])
def test_receipt_evidence_round_trips_exact_packaged_bytes(gate):
    evidence, bindings = _gate_fixture(gate)
    receipt = writers.build_evidenced_prerequisite_receipt(
        gate=gate,
        bindings=bindings,
        evidence=evidence,
    )
    data = prerequisite_evidence_bytes(evidence)
    assert verify_prerequisite_evidence_bytes(
        gate=gate,
        bindings=bindings,
        receipt_evidence=receipt["identity"]["evidence"],
        data=data,
    ) == evidence


def test_genuine_canonical_smoke_receipt_is_accepted():
    """A smoke receipt proving the two registered smoke cells and the canonical
    100-replicate bootstrap must validate cleanly."""
    evidence, bindings = _smoke_evidence()
    validate_prerequisite_evidence(
        gate="smoke",
        bindings=bindings,
        evidence=evidence,
    )


def test_degenerate_one_cell_smoke_receipt_is_rejected():
    """Regression for the smoke prerequisite gate: a one-cell/one-replicate
    smoke aggregate with matching IDs, VALID status, and
    ``requested == completed == 1`` was previously accepted. It must now be
    rejected -- it does not prove the documented smoke gate (two registered
    smoke cells)."""
    evidence, bindings = _smoke_evidence()
    tampered = copy.deepcopy(evidence)
    one_cell = tampered["aggregate"]["expected_cell_keys"][:1]
    tampered["aggregate"].update(
        {"requested": 1, "completed": 1, "expected_cell_keys": one_cell}
    )
    with pytest.raises(ValueError, match="canonical smoke cell set"):
        validate_prerequisite_evidence(
            gate="smoke",
            bindings=bindings,
            evidence=tampered,
        )


def test_smoke_receipt_requires_the_canonical_100_replicate_bootstrap():
    """A smoke run spec whose bootstrap claims a non-canonical replicate count
    (a degenerate one-replicate smoke) must be rejected by the reused canonical
    run-spec validator -- even with content-addressed IDs recomputed to match,
    so the rejection is on the smoke shape, not an incidental ID mismatch."""
    evidence, bindings = _smoke_evidence()
    tampered = copy.deepcopy(evidence)
    spec_identity = tampered["run_spec"]["identity"]
    spec_identity["bootstrap"]["replicate_count"] = 1
    tampered["run_spec"] = build_manifest(spec_identity)
    tampered["aggregate"]["run_spec_id"] = tampered["run_spec"]["id"]
    with pytest.raises(ValueError, match="canonical contract"):
        validate_prerequisite_evidence(
            gate="smoke",
            bindings=bindings,
            evidence=tampered,
        )


@pytest.mark.parametrize(
    ("gate", "mutate", "message"),
    [
        (
            "smoke",
            lambda evidence: evidence["aggregate"].update(
                {"release_status": "INVALID"}
            ),
            "complete VALID run",
        ),
        (
            "mutation",
            lambda evidence: evidence["results"].pop(),
            "roster mismatch",
        ),
        (
            "determinism",
            lambda evidence: evidence["second_file_sha256"].update(
                {"fit_rows.jsonl.gz": "f" * 64}
            ),
            "hashes (differ|do not match)",
        ),
    ],
)
def test_semantically_false_gate_evidence_is_rejected(gate, mutate, message):
    evidence, bindings = _gate_fixture(gate)
    tampered = copy.deepcopy(evidence)
    mutate(tampered)
    with pytest.raises(ValueError, match=message):
        validate_prerequisite_evidence(
            gate=gate,
            bindings=bindings,
            evidence=tampered,
        )


def test_mutation_evidence_rejects_a_different_executing_source():
    evidence, bindings = _gate_fixture("mutation")
    tampered = copy.deepcopy(evidence)
    tampered["source_execution"]["executing_source_manifest_id"] = "f" * 64
    with pytest.raises(ValueError, match="executing source mismatch"):
        validate_prerequisite_evidence(
            gate="mutation",
            bindings=bindings,
            evidence=tampered,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("execution_id", "local-first", "not distinct"),
        ("adapter_subdir", "adapter_build_a", "not distinct"),
        ("adapter_subdir", "./adapter_build_b", "noncanonical"),
    ],
)
def test_determinism_evidence_requires_distinct_canonical_fresh_builds(
    field,
    value,
    message,
):
    evidence, bindings = _gate_fixture("determinism")
    tampered = copy.deepcopy(evidence)
    tampered["second_build_execution"][field] = value
    with pytest.raises(ValueError, match=message):
        validate_prerequisite_evidence(
            gate="determinism",
            bindings=bindings,
            evidence=tampered,
        )


@pytest.mark.parametrize("gate", ["mutation", "determinism"])
def test_legacy_provenance_evidence_is_rejected(gate):
    evidence, bindings = _gate_fixture(gate)
    legacy = copy.deepcopy(evidence)
    legacy["schema_version"] = 1
    legacy.pop("source_execution")
    if gate == "determinism":
        legacy.pop("first_build_execution")
        legacy.pop("second_build_execution")
    with pytest.raises(ValueError, match="fields mismatch"):
        validate_prerequisite_evidence(
            gate=gate,
            bindings=bindings,
            evidence=legacy,
        )


def test_arbitrary_hex_receipt_cannot_authorize_unrelated_bytes():
    evidence, bindings = _smoke_evidence()
    data = prerequisite_evidence_bytes(evidence)
    with pytest.raises(ValueError, match="digest mismatch"):
        verify_prerequisite_evidence_bytes(
            gate="smoke",
            bindings=bindings,
            receipt_evidence={"evidence_sha256": "a" * 64},
            data=data,
        )


def test_noncanonical_evidence_bytes_are_rejected_even_when_digest_matches():
    evidence, bindings = _smoke_evidence()
    data = prerequisite_evidence_bytes(evidence).replace(b"\n", b"\r\n")
    with pytest.raises(ValueError, match="noncanonical"):
        verify_prerequisite_evidence_bytes(
            gate="smoke",
            bindings=bindings,
            receipt_evidence={
                "evidence_sha256": hashlib.sha256(data).hexdigest(),
            },
            data=data,
        )


def test_source_manifest_rejects_extra_identity_fields_and_unlisted_bytes(tmp_path):
    content = tmp_path / "source"
    content.mkdir()
    declared = content / "declared.py"
    declared.write_text("declared\n", encoding="utf-8")
    identity = source_manifest_identity(
        git_sha="a" * 40,
        files=[
            {
                "path": "declared.py",
                "mode": "100644",
                "size": declared.stat().st_size,
                "sha256": sha256_file(declared),
            }
        ],
        pyproject_sha256="",
        uv_lock_sha256="",
    )
    manifest = build_manifest(identity)
    (tmp_path / "source_manifest.json").write_bytes(
        prerequisite_evidence_bytes(manifest)
    )
    validate_bound_content_manifest(
        tmp_path,
        manifest_name="source_manifest.json",
        expected_id=manifest["id"],
        expected_kind="source_snapshot",
        file_key="files",
        name_key="path",
        content_subdir="source",
    )

    (content / "unlisted.py").write_text("unlisted\n", encoding="utf-8")
    with pytest.raises(ValueError, match="inventory mismatch"):
        validate_bound_content_manifest(
            tmp_path,
            manifest_name="source_manifest.json",
            expected_id=manifest["id"],
            expected_kind="source_snapshot",
            file_key="files",
            name_key="path",
            content_subdir="source",
        )
    (content / "unlisted.py").unlink()
    identity["unbound_claim"] = True
    altered = build_manifest(identity)
    (tmp_path / "source_manifest.json").write_bytes(
        prerequisite_evidence_bytes(altered)
    )
    with pytest.raises(ValueError, match="identity fields mismatch"):
        validate_bound_content_manifest(
            tmp_path,
            manifest_name="source_manifest.json",
            expected_id=altered["id"],
            expected_kind="source_snapshot",
            file_key="files",
            name_key="path",
            content_subdir="source",
        )


def test_raw_manifest_requires_the_exact_producer_role_set(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    entries = []
    for role in RAW_INPUT_ROLES[:-1]:
        path = raw / role
        path.write_text("{}\n", encoding="utf-8")
        entries.append(
            {"role": role, "size": path.stat().st_size, "sha256": sha256_file(path)}
        )
    manifest = build_manifest(
        raw_input_identity(
            files=entries,
            semantic_checks={"all_semantic_checks_pass": True},
        )
    )
    (tmp_path / "raw_input_manifest.json").write_bytes(
        prerequisite_evidence_bytes(manifest)
    )
    with pytest.raises(ValueError, match="raw-input roles mismatch"):
        validate_bound_content_manifest(
            tmp_path,
            manifest_name="raw_input_manifest.json",
            expected_id=manifest["id"],
            expected_kind="raw_input_bundle",
            file_key="files",
            name_key="role",
            content_subdir="raw",
            require_semantic_pass=True,
        )


def _staged_model_snapshot(base: Path) -> dict:
    """Stage one canonical model-snapshot content bundle under ``base``."""
    content = base / "snapshot"
    content.mkdir(parents=True)
    payload = b"model-bytes"
    (content / "model.bin").write_bytes(payload)
    manifest = build_manifest(
        model_snapshot_identity(
            model_id=ADAPTER_SCORING_SPEC["model_id"],
            revision="b" * 40,
            files=[
                {
                    "path": "model.bin",
                    "size": len(payload),
                    "sha256": sha256_bytes(payload),
                }
            ],
            sentence_transformers_version="fixture",
            transformers_version="fixture",
        )
    )
    (base / "model_snapshot_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def test_modal_and_local_resume_use_the_same_closed_manifest_validator(
    tmp_path,
    monkeypatch,
):
    """Behavioral mirror check: both lanes' cached-manifest loaders accept the
    same staged content, and a tampered cached byte fails on BOTH lanes with
    the shared closed validator's exact error."""
    from scripts import run_stopdff_v5_local as local_runner
    from tests.harness_control_plane import _load_modal_runner

    modal_runner = _load_modal_runner(monkeypatch)
    base = tmp_path / "model"
    manifest = _staged_model_snapshot(base)
    kwargs = dict(
        manifest_name="model_snapshot_manifest.json",
        file_key="files",
        name_key="path",
        content_subdir="snapshot",
    )

    local_manifest = local_runner._load_bound_content_manifest(
        base, expected_kind="model_snapshot", **kwargs
    )
    modal_manifest = modal_runner._verified_content_manifest(
        base,
        expected_id=manifest["id"],
        expected_kind="model_snapshot",
        **kwargs,
    )
    assert local_manifest == modal_manifest == manifest

    (base / "snapshot" / "model.bin").write_bytes(b"tampered-bytes")
    with pytest.raises(ValueError) as local_error:
        local_runner._load_bound_content_manifest(
            base, expected_kind="model_snapshot", **kwargs
        )
    with pytest.raises(ValueError) as modal_error:
        modal_runner._verified_content_manifest(
            base,
            expected_id=manifest["id"],
            expected_kind="model_snapshot",
            **kwargs,
        )
    assert str(local_error.value) == str(modal_error.value)
    assert "file mismatch: model.bin" in str(local_error.value)
