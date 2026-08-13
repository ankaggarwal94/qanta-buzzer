#!/usr/bin/env python3
"""Recovery-assurance canary derivations for the StopDFF v5 Modal pipeline.

The bounded recovery assurance is a zero-cell canary for the production sweep
attempt protocol (hard-exit once, classify the interruption, finish attempt 2,
read the durable bytes back). This module owns the canary's pure derivations:
the tag grammar, the oracle sweep context, the durable-observation reader, the
expected-evidence/phase-state classifier, and the aggregate readback check.

Nothing here imports ``modal`` or touches Modal state. Deployment-specific
bindings (the validated image source-manifest ID and the volume-mounted canary
paths) arrive as explicit parameters; ``scripts/modal_stopdff_v5_runner.py``
keeps thin same-named wrappers that supply them, and its ``recovery_assurance``
stage function remains the single Modal-registered entry point. The
cross-process host driver stays in ``scripts/modal_stopdff_v5_assurance.py``.
"""
from __future__ import annotations

import json
import re
from pathlib import Path


def canonical_assurance_tag(tag: object) -> str:
    if (
        not isinstance(tag, str)
        or not re.fullmatch(r"[0-9a-f][0-9a-f-]{7,63}", tag)
        or ".." in tag
    ):
        raise ValueError("assurance tag must be 8-64 lowercase hex/hyphen characters")
    return tag


def assurance_sweep_context(
    tag: str,
    *,
    attempt: int,
    resume: bool,
    commit_fn,
    image_source_id: str,
    run_root: Path,
):
    """Build a zero-cell context that still uses the production attempt protocol."""
    from scripts.stopdff_v5.bootstrap import build_bootstrap_plan, plan_identity
    from scripts.stopdff_v5.identity import compute_id
    from scripts.stopdff_v5.sweep import SweepContext

    def oracle_id(label: str) -> str:
        return compute_id({"kind": "modal_assurance", "tag": tag, "label": label})

    plan = build_bootstrap_plan(["oracle-item"], replicates=1, seed=1)
    bootstrap_plan_id = compute_id(plan_identity(plan))
    adapter_id = oracle_id("adapter")
    myopic_id = oracle_id("myopic")
    producer_hashes = {
        "checker.py": oracle_id("checker"),
        "sweep.py": oracle_id("sweep"),
    }
    run_spec = {
        "profile_variant": "smoke",
        "identity": {
            "source_manifest_id": image_source_id,
            "raw_input_bundle_id": oracle_id("raw"),
            "model_snapshot_id": oracle_id("model"),
            "adapter_bundle_id": adapter_id,
            "fvi_study_id": oracle_id("fvi"),
            "bootstrap_plan_id": bootstrap_plan_id,
            "environment_contract_id": oracle_id("environment"),
            "resource_summary_id": oracle_id("resources"),
        },
        "evidence_roots": {
            "myopic_artifact_sha256": myopic_id,
            "producer_hashes": producer_hashes,
        },
        "fvi_selected": {"tolerance": "1e-6", "max_iterations": 1},
        "gate": {
            "allow_low_mc_retention": False,
            "allow_incomplete_mc_coverage": False,
        },
    }
    run_spec_id = compute_id(run_spec)
    return SweepContext(
        rows=[],
        calibration_json=None,
        run_spec=run_spec,
        run_spec_id=run_spec_id,
        bootstrap_plan=plan,
        output_dir=Path(run_root),
        fvi_tolerance="1e-6",
        fvi_max_iterations=1,
        backend="modal",
        profile_variant="smoke",
        adapter_bundle_id=adapter_id,
        adapter_fit_rows_sha256=oracle_id("fit-rows"),
        adapter_eval_rows_sha256=oracle_id("eval-rows"),
        myopic_artifact_sha256=myopic_id,
        producer_hashes=producer_hashes,
        cells=[],
        commit_fn=commit_fn,
        resource_summary={"backend": "modal", "assurance_tag": tag},
        attempt={
            "attempt": attempt,
            "mode": "resume" if resume else "fresh",
            "command": ["modal_assurance"] + (["--resume"] if resume else []),
        },
        resume=resume,
    )


def assurance_observation(tag: str, *, root: Path) -> dict:
    """Read and validate the durable attempt records for one assurance tag."""
    from scripts.stopdff_v5.attempt_history import load_attempt_history
    from scripts.stopdff_v5.identity import (
        loads_no_duplicate_keys,
        sha256_bytes,
        sha256_file,
    )

    root = Path(root)
    run_root = root / "run"
    _, attempts = load_attempt_history(run_root / "attempts.jsonl")
    results: dict[str, dict] = {}
    result_sha256: dict[str, str] = {}
    results_dir = run_root / "attempt_results"
    if results_dir.is_dir():
        for path in sorted(results_dir.iterdir()):
            if path.is_symlink() or not path.is_file():
                raise ValueError("assurance attempt result path is noncanonical")
            value = loads_no_duplicate_keys(path.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise ValueError("assurance attempt result is not an object")
            results[path.name] = value
            result_sha256[path.name] = sha256_file(path)
    arm_path = root / "crash_arm.json"
    if arm_path.is_symlink() or not arm_path.is_file():
        raise ValueError("assurance crash arm is missing")
    arm = loads_no_duplicate_keys(arm_path.read_text(encoding="utf-8"))
    if not isinstance(arm, dict):
        raise ValueError("assurance crash arm is invalid")

    def identity_file(name: str, fields: set[str]) -> tuple[dict, str]:
        path = run_root / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"assurance {name} is missing or noncanonical")
        data = path.read_bytes()
        try:
            value = loads_no_duplicate_keys(data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"assurance {name} is invalid JSON") from exc
        expected_bytes = (
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode("utf-8")
        if (
            not isinstance(value, dict)
            or set(value) != fields
            or data != expected_bytes
        ):
            raise ValueError(f"assurance {name} is noncanonical")
        return value, sha256_bytes(data)

    run_spec, run_spec_sha256 = identity_file(
        "run_spec.json",
        {"id", "identity"},
    )
    bootstrap_plan, bootstrap_plan_sha256 = identity_file(
        "bootstrap_plan.json",
        {"id", "identity", "item_ids"},
    )
    return {
        "tag": tag,
        "attempts": attempts,
        "results": results,
        "result_sha256": result_sha256,
        "crash_arm": arm,
        "attempts_sha256": sha256_file(run_root / "attempts.jsonl"),
        "crash_arm_sha256": sha256_file(arm_path),
        "run_spec": run_spec,
        "run_spec_sha256": run_spec_sha256,
        "bootstrap_plan": bootstrap_plan,
        "bootstrap_plan_sha256": bootstrap_plan_sha256,
    }


def assurance_expected_evidence(
    tag: str,
    *,
    image_source_id: str,
    run_root: Path,
) -> dict:
    """Derive the exact canary histories and immutable run identities."""
    from scripts.stopdff_v5.bootstrap import plan_identity
    from scripts.stopdff_v5.identity import compute_id

    first_context = assurance_sweep_context(
        tag,
        attempt=1,
        resume=False,
        commit_fn=lambda: None,
        image_source_id=image_source_id,
        run_root=run_root,
    )
    second_context = assurance_sweep_context(
        tag,
        attempt=2,
        resume=True,
        commit_fn=lambda: None,
        image_source_id=image_source_id,
        run_root=run_root,
    )

    def attempt_record(context) -> dict:
        return {
            **context.attempt,
            "run_spec_id": context.run_spec_id,
            "adapter_id": context.adapter_bundle_id,
            "bootstrap_plan_id": compute_id(
                plan_identity(context.bootstrap_plan)
            ),
            "state": "started",
        }

    first = attempt_record(first_context)
    second = attempt_record(second_context)
    return {
        "first_context": first_context,
        "second_context": second_context,
        "first_attempt": first,
        "second_attempt": second,
        "interrupted": {
            "attempt": 1,
            "state": "interrupted",
            "run_spec_id": first_context.run_spec_id,
            "reason": "terminal_result_missing_at_resume",
        },
        "completed": {
            "attempt": 2,
            "state": "completed",
            "run_spec_id": first_context.run_spec_id,
            "completed": 0,
            "failed": 0,
        },
        "run_spec": {
            "id": first_context.run_spec_id,
            "identity": first_context.run_spec,
        },
        "bootstrap_plan": {
            "id": first["bootstrap_plan_id"],
            "identity": plan_identity(first_context.bootstrap_plan),
            "item_ids": first_context.bootstrap_plan.item_ids,
        },
    }


def assurance_phase_state(
    tag: str,
    observation: dict,
    *,
    image_source_id: str,
    run_root: Path,
) -> tuple[str, dict]:
    """Classify only exact initial/classified/finished durable states."""
    expected = assurance_expected_evidence(
        tag,
        image_source_id=image_source_id,
        run_root=run_root,
    )
    arm = observation.get("crash_arm")
    if not isinstance(arm, dict) or set(arm) != {
        "tag",
        "source_manifest_id",
        "runtime",
        "exit_code",
        "armed_after_attempt_start_commit",
    }:
        raise ValueError("assurance crash arm schema is invalid")
    runtime = arm.get("runtime")
    if (
        arm.get("tag") != tag
        or arm.get("source_manifest_id") != image_source_id
        or arm.get("exit_code") != 91
        or arm.get("armed_after_attempt_start_commit") is not True
        or not isinstance(runtime, dict)
        or set(runtime)
        != {"container_hostname", "function_call_id", "input_id"}
        or not all(
            isinstance(runtime[field], str) and runtime[field]
            for field in runtime
        )
    ):
        raise ValueError("assurance crash arm bindings are invalid")
    if observation.get("run_spec") != expected["run_spec"]:
        raise ValueError("assurance run spec does not match the canary context")
    if observation.get("bootstrap_plan") != expected["bootstrap_plan"]:
        raise ValueError(
            "assurance bootstrap plan does not match the canary context"
        )

    attempts = observation.get("attempts")
    results = observation.get("results")
    if attempts == [expected["first_attempt"]] and results == {}:
        return "initial", expected
    if attempts == [expected["first_attempt"]] and results == {
        "1.json": expected["interrupted"]
    }:
        return "classified", expected
    if attempts == [
        expected["first_attempt"],
        expected["second_attempt"],
    ] and results == {
        "1.json": expected["interrupted"],
        "2.json": expected["completed"],
    }:
        return "finished", expected
    raise ValueError("assurance durable phase state is noncanonical")


def assurance_expected_aggregate(context, sweep_module) -> dict:
    identity = context.run_spec["identity"]
    return {
        "profile_name": sweep_module.PROFILE_NAME,
        "profile_variant": "smoke",
        "backend": "modal",
        "run_spec_id": context.run_spec_id,
        "adapter_bundle_id": context.adapter_bundle_id,
        "bootstrap_plan_id": identity["bootstrap_plan_id"],
        "fvi_study_id": identity["fvi_study_id"],
        "adapter_fit_rows_sha256": context.adapter_fit_rows_sha256,
        "adapter_eval_rows_sha256": context.adapter_eval_rows_sha256,
        "myopic_artifact_sha256": context.myopic_artifact_sha256,
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


def load_assurance_aggregate(context, sweep_module, *, path: Path) -> dict:
    from scripts.stopdff_v5.identity import loads_no_duplicate_keys

    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("assurance aggregate is missing or noncanonical")
    data = path.read_bytes()
    try:
        aggregate = loads_no_duplicate_keys(data.decode("utf-8"))
        expected_bytes = (
            json.dumps(
                aggregate,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ValueError("assurance aggregate is invalid JSON") from exc
    if (
        data != expected_bytes
        or aggregate != assurance_expected_aggregate(context, sweep_module)
    ):
        raise ValueError("assurance aggregate is noncanonical")
    return aggregate
