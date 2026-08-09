#!/usr/bin/env python3
"""Strictly verify the bounded Modal hard-exit assurance receipts offline."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping


_RECEIPT_NAMES = (
    "submitted",
    "recovered",
    "classified",
    "finished",
    "verified",
)
_ATTEMPT_FIELDS = frozenset(
    {
        "attempt",
        "mode",
        "command",
        "run_spec_id",
        "adapter_id",
        "bootstrap_plan_id",
        "state",
    }
)
_RUNTIME_FIELDS = frozenset(
    {"container_hostname", "function_call_id", "input_id"}
)
_OBSERVATION_FIELDS = frozenset(
    {
        "tag",
        "attempts",
        "results",
        "result_sha256",
        "crash_arm",
        "attempts_sha256",
        "crash_arm_sha256",
        "run_spec",
        "run_spec_sha256",
        "bootstrap_plan",
        "bootstrap_plan_sha256",
    }
)
_AGGREGATE_FIELDS = frozenset(
    {
        "profile_name",
        "profile_variant",
        "backend",
        "run_spec_id",
        "adapter_bundle_id",
        "bootstrap_plan_id",
        "fvi_study_id",
        "adapter_fit_rows_sha256",
        "adapter_eval_rows_sha256",
        "myopic_artifact_sha256",
        "requested",
        "completed",
        "skipped",
        "failed",
        "expected_cell_keys",
        "fvi_selected",
        "cells",
        "family",
        "gate_overrides",
        "release_status",
        "release_reasons",
    }
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_TAG_RE = re.compile(r"[0-9a-f][0-9a-f-]{7,63}\Z")


class AssuranceVerificationError(ValueError):
    """Raised when a receipt violates the assurance evidence contract."""


def _fail(message: str) -> None:
    raise AssuranceVerificationError(message)


def _exact_object(value: Any, fields: set[str] | frozenset[str], where: str) -> dict:
    if not isinstance(value, dict):
        _fail(f"{where} must be an object")
    actual = set(value)
    expected = set(fields)
    if actual != expected:
        _fail(
            f"{where} fields do not match the exact schema: "
            f"missing={sorted(expected - actual)} surplus={sorted(actual - expected)}"
        )
    return value


def _exact_value(value: Any, expected: Any, where: str) -> None:
    if type(value) is not type(expected) or value != expected:
        _fail(f"{where} must be exactly {expected!r}")


def _nonempty_string(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        _fail(f"{where} must be a nonempty, whitespace-trimmed string")
    return value


def _sha256(value: Any, where: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        _fail(f"{where} must be a lowercase SHA-256 digest")
    return value


def _pretty_json_bytes(value: Any) -> bytes:
    try:
        return (
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode("utf-8")
    except (RecursionError, OverflowError) as exc:
        raise AssuranceVerificationError(
            "receipt JSON exceeds the supported structural complexity"
        ) from exc
    except (TypeError, ValueError) as exc:
        raise AssuranceVerificationError(
            "receipt contains a non-finite or non-JSON value"
        ) from exc


def _attempt_bytes(attempts: list[dict]) -> bytes:
    try:
        return b"".join(
            (json.dumps(record, sort_keys=True, allow_nan=False) + "\n").encode(
                "utf-8"
            )
            for record in attempts
        )
    except (RecursionError, OverflowError) as exc:
        raise AssuranceVerificationError(
            "attempt history exceeds the supported structural complexity"
        ) from exc
    except (TypeError, ValueError) as exc:
        raise AssuranceVerificationError(
            "attempt history contains a non-finite or non-JSON value"
        ) from exc


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _reject_constant(value: str) -> Any:
    raise AssuranceVerificationError(f"non-finite JSON constant is forbidden: {value}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"duplicate JSON key is forbidden: {key!r}")
        result[key] = value
    return result


def _load_receipt(path: Path, name: str) -> tuple[dict, bytes]:
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        _fail(f"{name} receipt must be a regular file: {path}")
    try:
        data = path.read_bytes()
        text = data.decode("utf-8")
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except AssuranceVerificationError:
        raise
    except (RecursionError, OverflowError) as exc:
        raise AssuranceVerificationError(
            f"{name} receipt exceeds the supported structural complexity: {path}"
        ) from exc
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise AssuranceVerificationError(
            f"{name} receipt cannot be read as strict UTF-8 JSON: {path}"
        ) from exc
    if not isinstance(value, dict):
        _fail(f"{name} receipt must contain an object")
    if data != _pretty_json_bytes(value):
        _fail(f"{name} receipt bytes are not the producer's canonical JSON encoding")
    return value, data


def _validate_runtime(value: Any, where: str) -> dict:
    runtime = _exact_object(value, _RUNTIME_FIELDS, where)
    for field in sorted(_RUNTIME_FIELDS):
        _nonempty_string(runtime[field], f"{where}.{field}")
    return runtime


def _validate_attempt(
    value: Any,
    *,
    number: int,
    mode: str,
    command: list[str],
    where: str,
) -> dict:
    attempt = _exact_object(value, _ATTEMPT_FIELDS, where)
    _exact_value(attempt["attempt"], number, f"{where}.attempt")
    _exact_value(attempt["mode"], mode, f"{where}.mode")
    _exact_value(attempt["command"], command, f"{where}.command")
    _exact_value(attempt["state"], "started", f"{where}.state")
    for field in ("run_spec_id", "adapter_id", "bootstrap_plan_id"):
        _sha256(attempt[field], f"{where}.{field}")
    return attempt


def _validate_crash_arm(value: Any, *, tag: str, where: str) -> dict:
    arm = _exact_object(
        value,
        {
            "tag",
            "source_manifest_id",
            "runtime",
            "exit_code",
            "armed_after_attempt_start_commit",
        },
        where,
    )
    _exact_value(arm["tag"], tag, f"{where}.tag")
    _sha256(arm["source_manifest_id"], f"{where}.source_manifest_id")
    _exact_value(arm["exit_code"], 91, f"{where}.exit_code")
    _exact_value(
        arm["armed_after_attempt_start_commit"],
        True,
        f"{where}.armed_after_attempt_start_commit",
    )
    _validate_runtime(arm["runtime"], f"{where}.runtime")
    return arm


def _validate_interrupted_result(
    value: Any, *, run_spec_id: str, where: str
) -> dict:
    result = _exact_object(
        value,
        {"attempt", "state", "run_spec_id", "reason"},
        where,
    )
    _exact_value(result["attempt"], 1, f"{where}.attempt")
    _exact_value(result["state"], "interrupted", f"{where}.state")
    _exact_value(result["run_spec_id"], run_spec_id, f"{where}.run_spec_id")
    _exact_value(
        result["reason"],
        "terminal_result_missing_at_resume",
        f"{where}.reason",
    )
    return result


def _validate_completed_result(
    value: Any, *, run_spec_id: str, where: str
) -> dict:
    result = _exact_object(
        value,
        {"attempt", "state", "run_spec_id", "completed", "failed"},
        where,
    )
    _exact_value(result["attempt"], 2, f"{where}.attempt")
    _exact_value(result["state"], "completed", f"{where}.state")
    _exact_value(result["run_spec_id"], run_spec_id, f"{where}.run_spec_id")
    _exact_value(result["completed"], 0, f"{where}.completed")
    _exact_value(result["failed"], 0, f"{where}.failed")
    return result


def _validate_observation(
    value: Any,
    *,
    tag: str,
    source_manifest_id: str,
    state: str,
    where: str,
) -> dict:
    observation = _exact_object(value, _OBSERVATION_FIELDS, where)
    _exact_value(observation["tag"], tag, f"{where}.tag")
    _sha256(observation["attempts_sha256"], f"{where}.attempts_sha256")
    _sha256(observation["crash_arm_sha256"], f"{where}.crash_arm_sha256")
    arm = _validate_crash_arm(
        observation["crash_arm"], tag=tag, where=f"{where}.crash_arm"
    )
    _exact_value(
        arm["source_manifest_id"],
        source_manifest_id,
        f"{where}.crash_arm.source_manifest_id",
    )
    for object_field, hash_field in (
        ("run_spec", "run_spec_sha256"),
        ("bootstrap_plan", "bootstrap_plan_sha256"),
    ):
        digest = _sha256(observation[hash_field], f"{where}.{hash_field}")
        if digest != _sha256_bytes(_pretty_json_bytes(observation[object_field])):
            _fail(
                f"{where}.{hash_field} does not bind the canonical "
                f"{object_field} bytes"
            )
    expected_bindings = _expected_assurance_bindings(
        tag=tag,
        source_manifest_id=source_manifest_id,
        run_spec=observation["run_spec"],
        bootstrap_plan=observation["bootstrap_plan"],
        where=where,
    )
    attempts = observation["attempts"]
    if not isinstance(attempts, list):
        _fail(f"{where}.attempts must be an array")
    expected_count = 2 if state == "finished" else 1
    if len(attempts) != expected_count:
        _fail(f"{where}.attempts must contain exactly {expected_count} record(s)")
    first = _validate_attempt(
        attempts[0],
        number=1,
        mode="fresh",
        command=["modal_assurance"],
        where=f"{where}.attempts[0]",
    )
    for field, expected in (
        ("run_spec_id", expected_bindings["run_spec_id"]),
        ("adapter_id", expected_bindings["adapter_id"]),
        ("bootstrap_plan_id", expected_bindings["bootstrap_plan_id"]),
    ):
        _exact_value(first[field], expected, f"{where}.attempts[0].{field}")
    if expected_count == 2:
        second = _validate_attempt(
            attempts[1],
            number=2,
            mode="resume",
            command=["modal_assurance", "--resume"],
            where=f"{where}.attempts[1]",
        )
        for field in ("run_spec_id", "adapter_id", "bootstrap_plan_id"):
            if second[field] != first[field]:
                _fail(f"{where} attempt 2 changed {field}")

    results = observation["results"]
    if not isinstance(results, dict):
        _fail(f"{where}.results must be an object")
    expected_names = {
        "initial": set(),
        "classified": {"1.json"},
        "finished": {"1.json", "2.json"},
    }[state]
    if set(results) != expected_names:
        _fail(
            f"{where}.results must contain exactly {sorted(expected_names)}"
        )
    result_sha256 = _exact_object(
        observation["result_sha256"],
        expected_names,
        f"{where}.result_sha256",
    )
    for name in sorted(expected_names):
        _sha256(result_sha256[name], f"{where}.result_sha256[{name!r}]")
        if result_sha256[name] != _sha256_bytes(
            _pretty_json_bytes(results[name])
        ):
            _fail(
                f"{where}.result_sha256[{name!r}] does not bind the "
                "canonical result bytes"
            )
    if state in {"classified", "finished"}:
        _validate_interrupted_result(
            results["1.json"],
            run_spec_id=first["run_spec_id"],
            where=f"{where}.results['1.json']",
        )
    if state == "finished":
        _validate_completed_result(
            results["2.json"],
            run_spec_id=first["run_spec_id"],
            where=f"{where}.results['2.json']",
        )

    if _sha256_bytes(_attempt_bytes(attempts)) != observation["attempts_sha256"]:
        _fail(f"{where}.attempts_sha256 does not bind the canonical history bytes")
    if _sha256_bytes(_pretty_json_bytes(arm)) != observation["crash_arm_sha256"]:
        _fail(f"{where}.crash_arm_sha256 does not bind the canonical arm bytes")
    return observation


def _oracle_id(tag: str, label: str) -> str:
    payload = json.dumps(
        {"kind": "modal_assurance", "label": label, "tag": tag},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return _sha256_bytes(payload)


def _expected_assurance_bindings(
    *,
    tag: str,
    source_manifest_id: str,
    run_spec: Any,
    bootstrap_plan: Any,
    where: str,
) -> dict[str, Any]:
    """Validate persisted identities while isolating the producer NumPy version."""
    bootstrap_manifest = _exact_object(
        bootstrap_plan,
        {"id", "identity", "item_ids"},
        f"{where}.bootstrap_plan",
    )
    _sha256(bootstrap_manifest["id"], f"{where}.bootstrap_plan.id")
    bootstrap_identity = _exact_object(
        bootstrap_manifest["identity"],
        {
            "evaluation_item_id_list_sha256",
            "bit_generator",
            "seed",
            "replicate_count",
            "resample_index_sha256",
            "numpy_version_contract",
            "resample_dtype",
        },
        f"{where}.bootstrap_plan.identity",
    )
    numpy_version = _nonempty_string(
        bootstrap_identity["numpy_version_contract"],
        f"{where}.bootstrap_plan.identity.numpy_version_contract",
    )
    try:
        from scripts.stopdff_v5.bootstrap import (
            build_bootstrap_plan,
            plan_identity,
        )
        from scripts.stopdff_v5.identity import canonical_bytes

        plan = build_bootstrap_plan(
            ["oracle-item"],
            replicates=1,
            seed=1,
        )
        expected_bootstrap_identity = plan_identity(plan)
        expected_bootstrap_identity["numpy_version_contract"] = numpy_version
        bootstrap_identity_bytes = canonical_bytes(bootstrap_identity)
        bootstrap_plan_id = _sha256_bytes(bootstrap_identity_bytes)
    except Exception as exc:
        raise AssuranceVerificationError(
            "cannot recompute the canonical assurance bootstrap identity"
        ) from exc
    _exact_value(
        bootstrap_identity,
        expected_bootstrap_identity,
        f"{where}.bootstrap_plan.identity",
    )
    _exact_value(
        bootstrap_manifest["item_ids"],
        ["oracle-item"],
        f"{where}.bootstrap_plan.item_ids",
    )
    _exact_value(
        bootstrap_manifest["id"],
        bootstrap_plan_id,
        f"{where}.bootstrap_plan.id",
    )

    adapter_id = _oracle_id(tag, "adapter")
    myopic_id = _oracle_id(tag, "myopic")
    run_spec_identity = {
        "profile_variant": "smoke",
        "identity": {
            "source_manifest_id": source_manifest_id,
            "raw_input_bundle_id": _oracle_id(tag, "raw"),
            "model_snapshot_id": _oracle_id(tag, "model"),
            "adapter_bundle_id": adapter_id,
            "fvi_study_id": _oracle_id(tag, "fvi"),
            "bootstrap_plan_id": bootstrap_plan_id,
            "environment_contract_id": _oracle_id(tag, "environment"),
            "resource_summary_id": _oracle_id(tag, "resources"),
        },
        "evidence_roots": {
            "myopic_artifact_sha256": myopic_id,
            "producer_hashes": {
                "checker.py": _oracle_id(tag, "checker"),
                "sweep.py": _oracle_id(tag, "sweep"),
            },
        },
        "fvi_selected": {
            "tolerance": "1e-6",
            "max_iterations": 1,
        },
        "gate": {
            "allow_low_mc_retention": False,
            "allow_incomplete_mc_coverage": False,
        },
    }
    try:
        run_spec_identity_bytes = canonical_bytes(run_spec_identity)
    except Exception as exc:
        raise AssuranceVerificationError(
            "cannot recompute the canonical assurance run-spec identity"
        ) from exc
    run_spec_id = _sha256_bytes(run_spec_identity_bytes)
    expected_run_spec = {
        "id": run_spec_id,
        "identity": run_spec_identity,
    }
    run_spec_manifest = _exact_object(
        run_spec,
        {"id", "identity"},
        f"{where}.run_spec",
    )
    _sha256(run_spec_manifest["id"], f"{where}.run_spec.id")
    _exact_value(run_spec_manifest, expected_run_spec, f"{where}.run_spec")
    return {
        "adapter_id": adapter_id,
        "bootstrap_plan_id": bootstrap_plan_id,
        "bootstrap_identity": bootstrap_identity,
        "bootstrap_identity_bytes": bootstrap_identity_bytes,
        "run_spec": run_spec_identity,
        "run_spec_identity_bytes": run_spec_identity_bytes,
        "run_spec_id": run_spec_id,
    }


def _validate_aggregate(
    value: Any, *, tag: str, first_attempt: dict, where: str
) -> dict:
    aggregate = _exact_object(value, _AGGREGATE_FIELDS, where)
    static_values = {
        "profile_name": "stopdff_bucketed_dp_paired_v2",
        "profile_variant": "smoke",
        "backend": "modal",
        "requested": 0,
        "completed": 0,
        "skipped": 0,
        "failed": 0,
        "expected_cell_keys": [],
        "cells": {},
        "family": None,
        "release_status": "INVALID",
        "release_reasons": [
            "bootstrap evidence invalid",
            "family-max evidence invalid",
        ],
    }
    for field, expected in static_values.items():
        _exact_value(aggregate[field], expected, f"{where}.{field}")
    _exact_value(
        aggregate["run_spec_id"],
        first_attempt["run_spec_id"],
        f"{where}.run_spec_id",
    )
    _exact_value(
        aggregate["adapter_bundle_id"],
        first_attempt["adapter_id"],
        f"{where}.adapter_bundle_id",
    )
    _exact_value(
        aggregate["bootstrap_plan_id"],
        first_attempt["bootstrap_plan_id"],
        f"{where}.bootstrap_plan_id",
    )
    for field in (
        "fvi_study_id",
        "adapter_fit_rows_sha256",
        "adapter_eval_rows_sha256",
        "myopic_artifact_sha256",
    ):
        _sha256(aggregate[field], f"{where}.{field}")
    expected_oracles = {
        "adapter_bundle_id": "adapter",
        "fvi_study_id": "fvi",
        "adapter_fit_rows_sha256": "fit-rows",
        "adapter_eval_rows_sha256": "eval-rows",
        "myopic_artifact_sha256": "myopic",
    }
    for field, label in expected_oracles.items():
        _exact_value(aggregate[field], _oracle_id(tag, label), f"{where}.{field}")
    fvi_selected = _exact_object(
        aggregate["fvi_selected"], {"tolerance", "max_iterations"}, f"{where}.fvi_selected"
    )
    _exact_value(fvi_selected["tolerance"], "1e-6", f"{where}.fvi_selected.tolerance")
    _exact_value(
        fvi_selected["max_iterations"], 1, f"{where}.fvi_selected.max_iterations"
    )
    gate_overrides = _exact_object(
        aggregate["gate_overrides"],
        {"allow_low_mc_retention", "allow_incomplete_mc_coverage"},
        f"{where}.gate_overrides",
    )
    for field in sorted(gate_overrides):
        _exact_value(gate_overrides[field], False, f"{where}.gate_overrides.{field}")
    return aggregate


def _validate_schema_version(receipt: dict, where: str) -> None:
    _exact_value(receipt["schema_version"], 1, f"{where}.schema_version")


def _verify_assurance_receipts(
    *,
    submitted: Any,
    recovered: Any,
    classified: Any,
    finished: Any,
    verified: Any,
    expected_source_manifest_id: str,
) -> dict[str, Any]:
    """Verify five parsed assurance receipts and return a canonical PASS verdict."""
    expected_source_manifest_id = _sha256(
        expected_source_manifest_id,
        "expected_source_manifest_id",
    )
    submitted = _exact_object(
        submitted,
        {"schema_version", "deployment", "tag", "phase", "function_call_id"},
        "submitted",
    )
    _validate_schema_version(submitted, "submitted")
    deployment = _nonempty_string(submitted["deployment"], "submitted.deployment")
    tag = submitted["tag"]
    if not isinstance(tag, str) or _TAG_RE.fullmatch(tag) is None or ".." in tag:
        _fail("submitted.tag is not a canonical assurance tag")
    _exact_value(submitted["phase"], "crash", "submitted.phase")
    call_id = _nonempty_string(
        submitted["function_call_id"], "submitted.function_call_id"
    )

    recovered = _exact_object(
        recovered,
        {"schema_version", "function_call_id", "result"},
        "recovered",
    )
    _validate_schema_version(recovered, "recovered")
    _exact_value(recovered["function_call_id"], call_id, "recovered.function_call_id")
    recovered_result = _exact_object(
        recovered["result"], {"phase", "runtime", "observation"}, "recovered.result"
    )
    _exact_value(recovered_result["phase"], "crash_rescheduled", "recovered.result.phase")
    recovered_runtime = _validate_runtime(recovered_result["runtime"], "recovered.result.runtime")
    initial = _validate_observation(
        recovered_result["observation"],
        tag=tag,
        source_manifest_id=expected_source_manifest_id,
        state="initial",
        where="recovered.result.observation",
    )
    original_runtime = initial["crash_arm"]["runtime"]
    first_attempt = initial["attempts"][0]
    for field in ("function_call_id", "input_id"):
        if original_runtime[field] != recovered_runtime[field]:
            _fail(f"hard-exit reschedule changed runtime {field}")
    if original_runtime["function_call_id"] != call_id:
        _fail("submission call ID is not bound to the crash runtime")
    if original_runtime["container_hostname"] == recovered_runtime["container_hostname"]:
        _fail("hard-exit reschedule did not cross a container boundary")

    phase_receipts: dict[str, dict] = {}
    phase_call_ids: dict[str, str] = {}
    for outer_phase, receipt in (
        ("classify", classified),
        ("finish", finished),
        ("verify", verified),
    ):
        parsed = _exact_object(
            receipt,
            {
                "schema_version",
                "deployment",
                "tag",
                "phase",
                "function_call_id",
                "result",
            },
            outer_phase,
        )
        _validate_schema_version(parsed, outer_phase)
        _exact_value(parsed["deployment"], deployment, f"{outer_phase}.deployment")
        _exact_value(parsed["tag"], tag, f"{outer_phase}.tag")
        _exact_value(parsed["phase"], outer_phase, f"{outer_phase}.phase")
        phase_call_ids[outer_phase] = _nonempty_string(
            parsed["function_call_id"],
            f"{outer_phase}.function_call_id",
        )
        phase_receipts[outer_phase] = parsed

    classified_result = _exact_object(
        phase_receipts["classify"]["result"],
        {"phase", "runtime", "observation"},
        "classified.result",
    )
    _exact_value(classified_result["phase"], "classified", "classified.result.phase")
    classified_runtime = _validate_runtime(
        classified_result["runtime"], "classified.result.runtime"
    )
    _exact_value(
        classified_runtime["function_call_id"],
        phase_call_ids["classify"],
        "classified.result.runtime.function_call_id",
    )
    classified_observation = _validate_observation(
        classified_result["observation"],
        tag=tag,
        source_manifest_id=expected_source_manifest_id,
        state="classified",
        where="classified.result.observation",
    )

    finished_result = _exact_object(
        phase_receipts["finish"]["result"],
        {"phase", "runtime", "aggregate", "observation"},
        "finished.result",
    )
    _exact_value(finished_result["phase"], "finished", "finished.result.phase")
    finished_runtime = _validate_runtime(
        finished_result["runtime"], "finished.result.runtime"
    )
    _exact_value(
        finished_runtime["function_call_id"],
        phase_call_ids["finish"],
        "finished.result.runtime.function_call_id",
    )
    finished_observation = _validate_observation(
        finished_result["observation"],
        tag=tag,
        source_manifest_id=expected_source_manifest_id,
        state="finished",
        where="finished.result.observation",
    )
    _validate_aggregate(
        finished_result["aggregate"],
        tag=tag,
        first_attempt=finished_observation["attempts"][0],
        where="finished.result.aggregate",
    )

    verified_result = _exact_object(
        phase_receipts["verify"]["result"],
        {"phase", "runtime", "observation"},
        "verified.result",
    )
    _exact_value(verified_result["phase"], "verify", "verified.result.phase")
    verified_runtime = _validate_runtime(
        verified_result["runtime"], "verified.result.runtime"
    )
    _exact_value(
        verified_runtime["function_call_id"],
        phase_call_ids["verify"],
        "verified.result.runtime.function_call_id",
    )
    verified_observation = _validate_observation(
        verified_result["observation"],
        tag=tag,
        source_manifest_id=expected_source_manifest_id,
        state="finished",
        where="verified.result.observation",
    )

    function_call_ids = {
        "crash": call_id,
        **phase_call_ids,
    }
    if len(set(function_call_ids.values())) != len(function_call_ids):
        _fail("assurance phase function-call IDs must be pairwise distinct")
    input_ids = {
        "crash": original_runtime["input_id"],
        "classify": classified_runtime["input_id"],
        "finish": finished_runtime["input_id"],
        "verify": verified_runtime["input_id"],
    }
    if len(set(input_ids.values())) != len(input_ids):
        _fail("assurance phase input IDs must be pairwise distinct")

    if classified_observation["attempts"] != initial["attempts"]:
        _fail("classification changed the committed attempt-1 history bytes")
    if finished_observation["attempts"][0] != initial["attempts"][0]:
        _fail("finish changed the committed attempt-1 history bytes")
    for observation, where in (
        (classified_observation, "classified"),
        (finished_observation, "finished"),
    ):
        if observation["crash_arm"] != initial["crash_arm"]:
            _fail(f"{where} observation changed the committed crash-arm bytes")
    if (
        _pretty_json_bytes(classified_observation["results"]["1.json"])
        != _pretty_json_bytes(finished_observation["results"]["1.json"])
    ):
        _fail("finish did not byte-preserve the classified interruption result")
    if (
        classified_observation["result_sha256"]["1.json"]
        != finished_observation["result_sha256"]["1.json"]
    ):
        _fail("finish changed the classified interruption-result bytes")
    if verified_observation != finished_observation:
        _fail("verify readback observation does not exactly equal the finish observation")

    receipts = {
        "submitted": submitted,
        "recovered": recovered,
        "classified": classified,
        "finished": finished,
        "verified": verified,
    }
    return {
        "schema_version": 1,
        "verdict": "PASS",
        "deployment": deployment,
        "tag": tag,
        "function_call_id": call_id,
        "source_manifest_id": initial["crash_arm"]["source_manifest_id"],
        "run_spec_id": first_attempt["run_spec_id"],
        "bootstrap_plan_id": first_attempt["bootstrap_plan_id"],
        "receipt_sha256": {
            name: _sha256_bytes(_pretty_json_bytes(receipts[name]))
            for name in _RECEIPT_NAMES
        },
        "attempts_sha256": finished_observation["attempts_sha256"],
        "crash_arm_sha256": finished_observation["crash_arm_sha256"],
    }


def verify_assurance_receipts(
    *,
    submitted: Any,
    recovered: Any,
    classified: Any,
    finished: Any,
    verified: Any,
    expected_source_manifest_id: str,
) -> dict[str, Any]:
    """Normalize structural-complexity failures at the public verifier boundary."""
    try:
        return _verify_assurance_receipts(
            submitted=submitted,
            recovered=recovered,
            classified=classified,
            finished=finished,
            verified=verified,
            expected_source_manifest_id=expected_source_manifest_id,
        )
    except AssuranceVerificationError:
        raise
    except (RecursionError, OverflowError) as exc:
        raise AssuranceVerificationError(
            "assurance evidence exceeds the supported structural complexity"
        ) from exc


def verify_assurance_files(
    paths: Mapping[str, Path],
    *,
    expected_source_manifest_id: str,
) -> dict[str, Any]:
    """Load canonical receipt files, then verify their cross-phase contract."""
    if set(paths) != set(_RECEIPT_NAMES):
        _fail("receipt path mapping must contain the exact five phase names")
    receipts: dict[str, dict] = {}
    raw_hashes: dict[str, str] = {}
    for name in _RECEIPT_NAMES:
        receipt, data = _load_receipt(Path(paths[name]), name)
        receipts[name] = receipt
        raw_hashes[name] = _sha256_bytes(data)
    verdict = verify_assurance_receipts(
        **receipts,
        expected_source_manifest_id=expected_source_manifest_id,
    )
    if verdict["receipt_sha256"] != raw_hashes:
        _fail("canonical receipt hash accounting mismatch")
    return verdict


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in _RECEIPT_NAMES:
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--expected-source-manifest-id", required=True)
    args = parser.parse_args(argv)
    try:
        verdict = verify_assurance_files(
            {name: getattr(args, name) for name in _RECEIPT_NAMES},
            expected_source_manifest_id=args.expected_source_manifest_id,
        )
    except AssuranceVerificationError as exc:
        print(f"MODAL_ASSURANCE_VERIFICATION: FAIL: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(verdict, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
