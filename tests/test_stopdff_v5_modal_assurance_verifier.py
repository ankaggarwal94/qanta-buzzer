from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from scripts import verify_stopdff_v5_modal_assurance as verifier
from scripts.stopdff_v5.bootstrap import build_bootstrap_plan, plan_identity
from scripts.stopdff_v5.identity import compute_id


TAG = "45b7f81f-c79c-42fa-89f2-07707cc0d41c"
DEPLOYMENT = "qanta-stopdff-v5"
CRASH_CALL_ID = "fc-crash-123"
CRASH_INPUT_ID = "in-crash-123"
SOURCE_MANIFEST_ID = "1" * 64
OBSERVED_NUMPY_VERSION = "modal-runtime-numpy"


def _pretty_bytes(value: object) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _attempt_bytes(attempts: list[dict]) -> bytes:
    return b"".join(
        (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode(
            "utf-8"
        )
        for value in attempts
    )


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _oracle_id(label: str) -> str:
    payload = json.dumps(
        {"kind": "modal_assurance", "label": label, "tag": TAG},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return _digest(payload)


def _expected_assurance_evidence() -> tuple[dict, dict, str]:
    plan = build_bootstrap_plan(["oracle-item"], replicates=1, seed=1)
    bootstrap_identity = plan_identity(plan)
    bootstrap_identity["numpy_version_contract"] = OBSERVED_NUMPY_VERSION
    bootstrap_plan_id = compute_id(bootstrap_identity)
    bootstrap_plan = {
        "id": bootstrap_plan_id,
        "identity": bootstrap_identity,
        "item_ids": ["oracle-item"],
    }
    adapter_id = _oracle_id("adapter")
    run_spec_identity = {
        "profile_variant": "smoke",
        "identity": {
            "source_manifest_id": SOURCE_MANIFEST_ID,
            "raw_input_bundle_id": _oracle_id("raw"),
            "model_snapshot_id": _oracle_id("model"),
            "adapter_bundle_id": adapter_id,
            "fvi_study_id": _oracle_id("fvi"),
            "bootstrap_plan_id": bootstrap_plan_id,
            "environment_contract_id": _oracle_id("environment"),
            "resource_summary_id": _oracle_id("resources"),
        },
        "evidence_roots": {
            "myopic_artifact_sha256": _oracle_id("myopic"),
            "producer_hashes": {
                "checker.py": _oracle_id("checker"),
                "sweep.py": _oracle_id("sweep"),
            },
        },
        "fvi_selected": {"tolerance": "1e-6", "max_iterations": 1},
        "gate": {
            "allow_low_mc_retention": False,
            "allow_incomplete_mc_coverage": False,
        },
    }
    run_spec = {
        "id": compute_id(run_spec_identity),
        "identity": run_spec_identity,
    }
    return run_spec, bootstrap_plan, adapter_id


def _runtime(hostname: str, call_id: str, input_id: str) -> dict:
    return {
        "container_hostname": hostname,
        "function_call_id": call_id,
        "input_id": input_id,
    }


def _observation(
    attempts: list[dict],
    results: dict[str, dict],
    arm: dict,
    run_spec: dict,
    bootstrap_plan: dict,
) -> dict:
    return {
        "tag": TAG,
        "attempts": copy.deepcopy(attempts),
        "results": copy.deepcopy(results),
        "result_sha256": {
            name: _digest(_pretty_bytes(result))
            for name, result in sorted(results.items())
        },
        "crash_arm": copy.deepcopy(arm),
        "attempts_sha256": _digest(_attempt_bytes(attempts)),
        "crash_arm_sha256": _digest(_pretty_bytes(arm)),
        "run_spec": copy.deepcopy(run_spec),
        "run_spec_sha256": _digest(_pretty_bytes(run_spec)),
        "bootstrap_plan": copy.deepcopy(bootstrap_plan),
        "bootstrap_plan_sha256": _digest(_pretty_bytes(bootstrap_plan)),
    }


def _rehash_observation(observation: dict) -> None:
    observation["attempts_sha256"] = _digest(
        _attempt_bytes(observation["attempts"])
    )
    observation["crash_arm_sha256"] = _digest(
        _pretty_bytes(observation["crash_arm"])
    )
    observation["result_sha256"] = {
        name: _digest(_pretty_bytes(result))
        for name, result in sorted(observation["results"].items())
    }
    observation["run_spec_sha256"] = _digest(
        _pretty_bytes(observation["run_spec"])
    )
    observation["bootstrap_plan_sha256"] = _digest(
        _pretty_bytes(observation["bootstrap_plan"])
    )


def _receipt_observations(receipts: dict[str, dict]) -> list[dict]:
    return [
        receipts["recovered"]["result"]["observation"],
        receipts["classified"]["result"]["observation"],
        receipts["finished"]["result"]["observation"],
        receipts["verified"]["result"]["observation"],
    ]


def _substitute_run_spec(receipts: dict[str, dict]) -> None:
    replacement = ""
    for observation in _receipt_observations(receipts):
        observation["run_spec"]["identity"]["identity"][
            "raw_input_bundle_id"
        ] = "e" * 64
        replacement = compute_id(observation["run_spec"]["identity"])
        observation["run_spec"]["id"] = replacement
        for attempt in observation["attempts"]:
            attempt["run_spec_id"] = replacement
        for result in observation["results"].values():
            result["run_spec_id"] = replacement
        _rehash_observation(observation)
    receipts["finished"]["result"]["aggregate"]["run_spec_id"] = replacement


def _substitute_bootstrap_plan(receipts: dict[str, dict]) -> None:
    bootstrap_replacement = ""
    run_spec_replacement = ""
    for observation in _receipt_observations(receipts):
        observation["bootstrap_plan"]["identity"]["seed"] = 2
        bootstrap_replacement = compute_id(
            observation["bootstrap_plan"]["identity"]
        )
        observation["bootstrap_plan"]["id"] = bootstrap_replacement
        observation["run_spec"]["identity"]["identity"][
            "bootstrap_plan_id"
        ] = bootstrap_replacement
        run_spec_replacement = compute_id(observation["run_spec"]["identity"])
        observation["run_spec"]["id"] = run_spec_replacement
        for attempt in observation["attempts"]:
            attempt["bootstrap_plan_id"] = bootstrap_replacement
            attempt["run_spec_id"] = run_spec_replacement
        for result in observation["results"].values():
            result["run_spec_id"] = run_spec_replacement
        _rehash_observation(observation)
    aggregate = receipts["finished"]["result"]["aggregate"]
    aggregate["bootstrap_plan_id"] = bootstrap_replacement
    aggregate["run_spec_id"] = run_spec_replacement


def _valid_receipts() -> dict[str, dict]:
    run_spec, bootstrap_plan, adapter_id = _expected_assurance_evidence()
    run_spec_id = run_spec["id"]
    bootstrap_plan_id = bootstrap_plan["id"]
    first_runtime = _runtime("container-a", CRASH_CALL_ID, CRASH_INPUT_ID)
    resumed_runtime = _runtime("container-b", CRASH_CALL_ID, CRASH_INPUT_ID)
    arm = {
        "tag": TAG,
        "source_manifest_id": SOURCE_MANIFEST_ID,
        "runtime": first_runtime,
        "exit_code": 91,
        "armed_after_attempt_start_commit": True,
    }
    attempt_1 = {
        "attempt": 1,
        "mode": "fresh",
        "command": ["modal_assurance"],
        "run_spec_id": run_spec_id,
        "adapter_id": adapter_id,
        "bootstrap_plan_id": bootstrap_plan_id,
        "state": "started",
    }
    attempt_2 = {
        **attempt_1,
        "attempt": 2,
        "mode": "resume",
        "command": ["modal_assurance", "--resume"],
    }
    interrupted = {
        "attempt": 1,
        "state": "interrupted",
        "run_spec_id": run_spec_id,
        "reason": "terminal_result_missing_at_resume",
    }
    completed = {
        "attempt": 2,
        "state": "completed",
        "run_spec_id": run_spec_id,
        "completed": 0,
        "failed": 0,
    }
    initial = _observation(
        [attempt_1], {}, arm, run_spec, bootstrap_plan
    )
    classified = _observation(
        [attempt_1], {"1.json": interrupted}, arm, run_spec, bootstrap_plan
    )
    finished = _observation(
        [attempt_1, attempt_2],
        {"1.json": interrupted, "2.json": completed},
        arm,
        run_spec,
        bootstrap_plan,
    )
    aggregate = {
        "profile_name": "stopdff_bucketed_dp_paired_v2",
        "profile_variant": "smoke",
        "backend": "modal",
        "run_spec_id": run_spec_id,
        "adapter_bundle_id": adapter_id,
        "bootstrap_plan_id": bootstrap_plan_id,
        "fvi_study_id": _oracle_id("fvi"),
        "adapter_fit_rows_sha256": _oracle_id("fit-rows"),
        "adapter_eval_rows_sha256": _oracle_id("eval-rows"),
        "myopic_artifact_sha256": _oracle_id("myopic"),
        "requested": 0,
        "completed": 0,
        "skipped": 0,
        "failed": 0,
        "expected_cell_keys": [],
        "fvi_selected": {"tolerance": "1e-6", "max_iterations": 1},
        "cells": {},
        "family": None,
        "gate_overrides": {
            "allow_low_mc_retention": False,
            "allow_incomplete_mc_coverage": False,
        },
        "release_status": "INVALID",
        "release_reasons": [
            "bootstrap evidence invalid",
            "family-max evidence invalid",
        ],
    }
    return {
        "submitted": {
            "schema_version": 1,
            "deployment": DEPLOYMENT,
            "tag": TAG,
            "phase": "crash",
            "function_call_id": CRASH_CALL_ID,
        },
        "recovered": {
            "schema_version": 1,
            "function_call_id": CRASH_CALL_ID,
            "result": {
                "phase": "crash_rescheduled",
                "runtime": resumed_runtime,
                "observation": initial,
            },
        },
        "classified": {
            "schema_version": 1,
            "deployment": DEPLOYMENT,
            "tag": TAG,
            "phase": "classify",
            "function_call_id": "fc-classify",
            "result": {
                "phase": "classified",
                "runtime": _runtime("container-c", "fc-classify", "in-classify"),
                "observation": classified,
            },
        },
        "finished": {
            "schema_version": 1,
            "deployment": DEPLOYMENT,
            "tag": TAG,
            "phase": "finish",
            "function_call_id": "fc-finish",
            "result": {
                "phase": "finished",
                "runtime": _runtime("container-d", "fc-finish", "in-finish"),
                "aggregate": aggregate,
                "observation": finished,
            },
        },
        "verified": {
            "schema_version": 1,
            "deployment": DEPLOYMENT,
            "tag": TAG,
            "phase": "verify",
            "function_call_id": "fc-verify",
            "result": {
                "phase": "verify",
                "runtime": _runtime("container-e", "fc-verify", "in-verify"),
                "observation": copy.deepcopy(finished),
            },
        },
    }


def _write_receipts(tmp_path: Path, receipts: dict[str, dict]) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for name, receipt in receipts.items():
        path = tmp_path / f"{name}.json"
        path.write_bytes(_pretty_bytes(receipt))
        paths[name] = path
    return paths


def test_valid_assurance_receipts_pass_and_cli_emits_verdict(
    tmp_path, capsys
):
    receipts = _valid_receipts()
    observed_bootstrap = receipts["recovered"]["result"]["observation"][
        "bootstrap_plan"
    ]
    assert (
        observed_bootstrap["identity"]["numpy_version_contract"]
        == OBSERVED_NUMPY_VERSION
    )
    assert OBSERVED_NUMPY_VERSION != plan_identity(
        build_bootstrap_plan(["oracle-item"], replicates=1, seed=1)
    )["numpy_version_contract"]
    paths = _write_receipts(tmp_path, receipts)

    verdict = verifier.verify_assurance_files(
        paths,
        expected_source_manifest_id=SOURCE_MANIFEST_ID,
    )

    assert verdict["verdict"] == "PASS"
    assert verdict["deployment"] == DEPLOYMENT
    assert verdict["tag"] == TAG
    assert verdict["function_call_id"] == CRASH_CALL_ID
    expected_run_spec, expected_bootstrap_plan, _ = (
        _expected_assurance_evidence()
    )
    assert verdict["source_manifest_id"] == SOURCE_MANIFEST_ID
    assert verdict["run_spec_id"] == expected_run_spec["id"]
    assert verdict["bootstrap_plan_id"] == expected_bootstrap_plan["id"]
    assert verdict["receipt_sha256"] == {
        name: _digest(path.read_bytes()) for name, path in paths.items()
    }

    argv = [part for name, path in paths.items() for part in (f"--{name}", str(path))]
    argv.extend(("--expected-source-manifest-id", SOURCE_MANIFEST_ID))
    assert verifier.main(argv) == 0
    emitted = json.loads(capsys.readouterr().out)
    assert emitted == verdict


@pytest.mark.parametrize(
    "case",
    [
        "unknown_outer_field",
        "unknown_nested_field",
        "wrong_recovery_call_id",
        "changed_reschedule_input_id",
        "same_reschedule_container",
        "empty_crash_call_id",
        "initial_has_result",
        "classified_has_attempt_2",
        "noncanonical_interruption",
        "attempt_2_not_completed",
        "aggregate_unknown_field",
        "aggregate_wrong_oracle_id",
        "changed_deployment",
        "changed_tag",
        "attempt_hash_mismatch",
        "verified_readback_differs",
        "phase_outer_call_id_mismatch",
        "consistent_run_spec_substitution",
        "consistent_bootstrap_substitution",
        "run_spec_hash_mismatch",
        "bootstrap_hash_mismatch",
        "run_spec_unknown_field",
        "bootstrap_unknown_field",
        "bootstrap_empty_numpy_version",
        "bootstrap_wrong_item_ids",
        "all_phase_runtime_ids_equal",
        "duplicate_phase_input_ids",
    ],
)
def test_tampered_assurance_receipts_fail_closed(tmp_path, case):
    receipts = _valid_receipts()
    if case == "unknown_outer_field":
        receipts["submitted"]["surplus"] = True
    elif case == "unknown_nested_field":
        receipts["recovered"]["result"]["runtime"]["surplus"] = True
    elif case == "wrong_recovery_call_id":
        receipts["recovered"]["function_call_id"] = "fc-other"
    elif case == "changed_reschedule_input_id":
        receipts["recovered"]["result"]["runtime"]["input_id"] = "in-other"
    elif case == "same_reschedule_container":
        receipts["recovered"]["result"]["runtime"][
            "container_hostname"
        ] = "container-a"
    elif case == "empty_crash_call_id":
        receipts["recovered"]["result"]["observation"]["crash_arm"][
            "runtime"
        ]["function_call_id"] = ""
        _rehash_observation(receipts["recovered"]["result"]["observation"])
    elif case == "initial_has_result":
        receipts["recovered"]["result"]["observation"]["results"] = {
            "1.json": copy.deepcopy(
                receipts["classified"]["result"]["observation"]["results"][
                    "1.json"
                ]
            )
        }
    elif case == "classified_has_attempt_2":
        receipts["classified"]["result"]["observation"]["attempts"].append(
            copy.deepcopy(
                receipts["finished"]["result"]["observation"]["attempts"][1]
            )
        )
        _rehash_observation(receipts["classified"]["result"]["observation"])
    elif case == "noncanonical_interruption":
        receipts["finished"]["result"]["observation"]["results"]["1.json"][
            "reason"
        ] = "different"
    elif case == "attempt_2_not_completed":
        receipts["finished"]["result"]["observation"]["results"]["2.json"][
            "state"
        ] = "failed"
    elif case == "aggregate_unknown_field":
        receipts["finished"]["result"]["aggregate"]["surplus"] = True
    elif case == "aggregate_wrong_oracle_id":
        receipts["finished"]["result"]["aggregate"]["fvi_study_id"] = "f" * 64
    elif case == "changed_deployment":
        receipts["finished"]["deployment"] = "other-deployment"
    elif case == "changed_tag":
        receipts["classified"]["tag"] = "deadbeef"
    elif case == "attempt_hash_mismatch":
        receipts["recovered"]["result"]["observation"][
            "attempts_sha256"
        ] = "0" * 64
    elif case == "verified_readback_differs":
        observation = receipts["verified"]["result"]["observation"]
        observation["crash_arm"]["runtime"][
            "container_hostname"
        ] = "container-z"
        _rehash_observation(observation)
    elif case == "phase_outer_call_id_mismatch":
        receipts["finished"]["function_call_id"] = "fc-other"
    elif case == "consistent_run_spec_substitution":
        _substitute_run_spec(receipts)
    elif case == "consistent_bootstrap_substitution":
        _substitute_bootstrap_plan(receipts)
    elif case == "run_spec_hash_mismatch":
        receipts["recovered"]["result"]["observation"][
            "run_spec_sha256"
        ] = "0" * 64
    elif case == "bootstrap_hash_mismatch":
        receipts["classified"]["result"]["observation"][
            "bootstrap_plan_sha256"
        ] = "0" * 64
    elif case == "run_spec_unknown_field":
        observation = receipts["recovered"]["result"]["observation"]
        observation["run_spec"]["surplus"] = True
        _rehash_observation(observation)
    elif case == "bootstrap_unknown_field":
        observation = receipts["classified"]["result"]["observation"]
        observation["bootstrap_plan"]["surplus"] = True
        _rehash_observation(observation)
    elif case == "bootstrap_empty_numpy_version":
        observation = receipts["recovered"]["result"]["observation"]
        observation["bootstrap_plan"]["identity"][
            "numpy_version_contract"
        ] = ""
        _rehash_observation(observation)
    elif case == "bootstrap_wrong_item_ids":
        observation = receipts["classified"]["result"]["observation"]
        observation["bootstrap_plan"]["item_ids"] = ["other-item"]
        _rehash_observation(observation)
    elif case == "all_phase_runtime_ids_equal":
        for phase in ("classified", "finished", "verified"):
            receipts[phase]["function_call_id"] = CRASH_CALL_ID
            runtime = receipts[phase]["result"]["runtime"]
            runtime["function_call_id"] = CRASH_CALL_ID
            runtime["input_id"] = CRASH_INPUT_ID
    elif case == "duplicate_phase_input_ids":
        for phase in ("classified", "finished", "verified"):
            receipts[phase]["result"]["runtime"]["input_id"] = (
                CRASH_INPUT_ID
            )
    else:  # pragma: no cover - protects the mutation matrix itself
        raise AssertionError(case)

    paths = _write_receipts(tmp_path, receipts)
    with pytest.raises(verifier.AssuranceVerificationError):
        verifier.verify_assurance_files(
            paths,
            expected_source_manifest_id=SOURCE_MANIFEST_ID,
        )


def test_noncanonical_receipt_bytes_are_rejected(tmp_path):
    paths = _write_receipts(tmp_path, _valid_receipts())
    paths["submitted"].write_bytes(paths["submitted"].read_bytes() + b"\n")

    with pytest.raises(
        verifier.AssuranceVerificationError, match="canonical JSON encoding"
    ):
        verifier.verify_assurance_files(
            paths,
            expected_source_manifest_id=SOURCE_MANIFEST_ID,
        )


def test_duplicate_json_keys_are_rejected(tmp_path):
    paths = _write_receipts(tmp_path, _valid_receipts())
    text = paths["submitted"].read_text(encoding="utf-8")
    paths["submitted"].write_text(
        text.replace(
            '  "schema_version": 1,',
            '  "schema_version": 1,\n  "schema_version": 1,',
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        verifier.AssuranceVerificationError, match="duplicate JSON key"
    ):
        verifier.verify_assurance_files(
            paths,
            expected_source_manifest_id=SOURCE_MANIFEST_ID,
        )


def test_deeply_nested_receipt_fails_without_cli_traceback(tmp_path, capsys):
    paths = _write_receipts(tmp_path, _valid_receipts())
    nested = b"[" * 4000 + b"0" + b"]" * 4000
    paths["submitted"].write_bytes(b'{"nested":' + nested + b"}\n")

    with pytest.raises(
        verifier.AssuranceVerificationError,
        match="structural complexity",
    ):
        verifier.verify_assurance_files(
            paths,
            expected_source_manifest_id=SOURCE_MANIFEST_ID,
        )

    argv = [
        part
        for name, path in paths.items()
        for part in (f"--{name}", str(path))
    ]
    argv.extend(("--expected-source-manifest-id", SOURCE_MANIFEST_ID))
    assert verifier.main(argv) == 1
    stderr = capsys.readouterr().err
    assert "MODAL_ASSURANCE_VERIFICATION: FAIL" in stderr
    assert "Traceback" not in stderr


def test_huge_json_integer_fails_without_cli_traceback(tmp_path, capsys):
    paths = _write_receipts(tmp_path, _valid_receipts())
    huge_integer = b"9" * 5000
    paths["submitted"].write_bytes(
        b'{"schema_version":' + huge_integer + b"}\n"
    )

    with pytest.raises(
        verifier.AssuranceVerificationError,
        match="strict UTF-8 JSON",
    ):
        verifier.verify_assurance_files(
            paths,
            expected_source_manifest_id=SOURCE_MANIFEST_ID,
        )

    argv = [
        part
        for name, path in paths.items()
        for part in (f"--{name}", str(path))
    ]
    argv.extend(("--expected-source-manifest-id", SOURCE_MANIFEST_ID))
    assert verifier.main(argv) == 1
    stderr = capsys.readouterr().err
    assert "MODAL_ASSURANCE_VERIFICATION: FAIL" in stderr
    assert "Traceback" not in stderr


def test_expected_source_manifest_id_is_an_external_binding(tmp_path):
    paths = _write_receipts(tmp_path, _valid_receipts())
    with pytest.raises(
        verifier.AssuranceVerificationError,
        match="source_manifest_id",
    ):
        verifier.verify_assurance_files(
            paths,
            expected_source_manifest_id="2" * 64,
        )
