"""Synthetic valid-package fixture factory for the StopDFF v5 gates and tests.

Builds the self-contained synthetic package (adapter bundle + run package)
that the checker accepts: ``build_valid_package`` plus its receipt-minting and
row-synthesis helpers. Extracted verbatim from ``selftest``, which re-exports
the historical names so ``selftest.build_valid_package`` keeps working for the
mutation gates and every external test consumer.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .adapter_build import (
    ADAPTER_SCHEMA_COLUMNS,
    derive_bound_calibration,
    question_trajectory_binding_from_rows,
)
from .bootstrap import build_bootstrap_plan, plan_identity
from .identity import build_manifest, compute_id, sha256_bytes, sha256_file
from .fvi_study import run_fvi_study
from .manifests import (
    ADAPTER_SCORING_SPEC,
    ENVIRONMENT_PACKAGES,
    FVI_PRODUCER_FILES,
    RAW_INPUT_ROLES,
    adapter_identity,
    environment_contract_identity,
    fvi_study_identity,
    model_snapshot_identity,
    raw_input_identity,
    run_spec_identity,
    source_manifest_identity,
)
from .profile import FINAL_REPLICATES, full_grid, smoke_cells
from .receipt_evidence import (
    DETERMINISM_BINDINGS,
    MUTATION_ROSTER,
    build_prerequisite_evidence,
    prerequisite_evidence_bytes,
)
from .rowio import write_jsonl_gz
from .sweep import SweepContext, run_sweep
from .writers import (
    build_evidenced_prerequisite_receipt,
    package_run,
)

CATEGORIES = ["history", "science", "arts"]
PREFIX_FRACS = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
_SYNTH_FVI_STUDY: dict[str, Any] | None = None


def _synth_rows(n_items: int = 44, seed: int = 11) -> list[dict]:
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for i in range(n_items):
        qid = f"q{i:03d}"
        split = "val" if i < n_items // 2 else "test"
        cat = CATEGORIES[i % len(CATEGORIES)]
        off = rng.uniform(-0.15, 0.15)
        tokens = [f"{qid}w{index:02d}" for index in range(10)]
        full_question = " ".join(tokens)
        prefixes = [
            " ".join(tokens[:count]) for count in (1, 3, 5, 7, 9, 10)
        ]
        full_digest = sha256_bytes(full_question.encode("utf-8"))
        for t, prefix in enumerate(prefixes):
            frac = round(len(prefix) / len(full_question), 6)
            mc = round(
                float(
                    np.clip(
                        0.25
                        + 0.55 * frac
                        + off
                        + rng.uniform(-0.05, 0.05),
                        0.0,
                        1.0,
                    )
                ),
                6,
            )
            qa = round(
                float(
                    np.clip(
                        0.20
                        + 0.60 * frac
                        + off
                        + rng.uniform(-0.05, 0.05),
                        0.0,
                        1.0,
                    )
                ),
                6,
            )
            correct = int(mc + rng.uniform(-0.15, 0.15) > 0.55)
            base = {"item_id": qid, "prefix_idx": t, "prefix_fraction": frac, "split": split,
                    "prefix_text_sha256": sha256_bytes(prefix.encode("utf-8")),
                    "prefix_char_count": len(prefix),
                    "full_question_sha256": full_digest,
                    "full_question_char_count": len(full_question),
                    "category": cat, "K": 4, "option_set_id": f"{qid}:K4",
                    "distractor_strategy": "unknown"}
            second_best = round(max(-1.0, mc - 0.1), 6)
            rows.append({**base, "format": "MC", "raw_similarity": mc, "correct": correct,
                         "p_second_best": second_best,
                         "top2_margin": round(mc - second_best, 6)})
            rows.append({**base, "format": "QA", "raw_similarity": qa, "correct": 1,
                         "p_second_best": 0.0, "top2_margin": 0.0})
    return rows


def _hex(n: str) -> str:
    return (n * 64)[:64]


def build_valid_package(
    base_dir: Path,
    *,
    fixed_fvi: bool = False,
    final_variant: bool = False,
) -> dict[str, Any]:
    global _SYNTH_FVI_STUDY
    if final_variant and fixed_fvi:
        raise ValueError("a final-profile package requires a genuine FVI study")
    base_dir = Path(base_dir)
    bundle = base_dir / "adapter_bundle"
    bundle.mkdir(parents=True, exist_ok=True)
    rows = _synth_rows()

    def row_key(row: dict[str, Any]) -> tuple[str, str, int]:
        return (
            str(row["item_id"]),
            row["format"],
            int(row["prefix_idx"]),
        )

    fit_rows = sorted(
        (r for r in rows if r["split"] == "val"),
        key=row_key,
    )
    eval_rows = sorted(
        (r for r in rows if r["split"] == "test"),
        key=row_key,
    )
    write_jsonl_gz(bundle / "fit_rows.jsonl.gz", fit_rows)
    write_jsonl_gz(bundle / "eval_rows.jsonl.gz", eval_rows)
    fit_sha = sha256_file(bundle / "fit_rows.jsonl.gz")
    eval_sha = sha256_file(bundle / "eval_rows.jsonl.gz")
    fit_item_count = len({row["item_id"] for row in fit_rows})
    eval_item_count = len({row["item_id"] for row in eval_rows})
    build_metadata = {
        "retention_thresholds": {"smoke": 0.5, "full": 0.98},
        "splits": {
            "train": {
                "raw_count": 1,
                "retained_count": 1,
                "dropped_count": 0,
                "retention_rate": 1.0,
            },
            "val": {
                "raw_count": fit_item_count,
                "retained_count": fit_item_count,
                "dropped_count": 0,
                "retention_rate": 1.0,
            },
            "test": {
                "raw_count": eval_item_count,
                "retained_count": eval_item_count,
                "dropped_count": 0,
                "retention_rate": 1.0,
            },
        },
    }
    (bundle / "build_metadata.json").write_text(
        json.dumps(build_metadata, sort_keys=True),
        encoding="utf-8",
    )
    build_metadata_sha = sha256_file(bundle / "build_metadata.json")

    input_manifest_dir = base_dir / "input_manifests"
    input_manifest_dir.mkdir(parents=True, exist_ok=True)
    source_bundle = input_manifest_dir / "source_bundle"
    source_content = source_bundle / "source"
    raw_bundle = input_manifest_dir / "raw_bundle"
    model_bundle = input_manifest_dir / "model_bundle"
    model_content = model_bundle / "snapshot"
    source_content.mkdir(parents=True, exist_ok=True)
    raw_bundle.mkdir(parents=True, exist_ok=True)
    model_content.mkdir(parents=True, exist_ok=True)
    fvi_producer_hashes = {
        name: sha256_bytes(name.encode("utf-8"))
        for name in FVI_PRODUCER_FILES
    }
    source_producer_hashes = {
        **fvi_producer_hashes,
        "adapter_build.py": sha256_bytes(b"adapter_build.py"),
        "checker.py": sha256_bytes(b"checker.py"),
        "sweep.py": sha256_bytes(b"sweep.py"),
    }
    source_payloads = {
        **{
            f"scripts/stopdff_v5/{name}": name.encode("utf-8")
            for name in source_producer_hashes
        },
        "pyproject.toml": b"pyproject.toml",
        "uv.lock": b"uv.lock",
    }
    source_files = []
    for relative, payload in source_payloads.items():
        target = source_content / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        source_files.append(
            {
                "path": relative,
                "mode": "100644",
                "size": len(payload),
                "sha256": sha256_bytes(payload),
            }
        )
    trajectory_id = question_trajectory_binding_from_rows(fit_rows + eval_rows)
    def _raw_question(qid: str) -> str:
        return " ".join(f"{qid}w{index:02d}" for index in range(10))

    split_records = {
        "train": [
            {
                "qid": "train-only",
                "question": "train only question",
                "answer": "train answer",
            }
        ],
        "val": [
            {"qid": f"q{i:03d}", "question": _raw_question(f"q{i:03d}"), "answer": "a"}
            for i in range(fit_item_count)
        ],
        "test": [
            {"qid": f"q{i:03d}", "question": _raw_question(f"q{i:03d}"), "answer": "a"}
            for i in range(fit_item_count, fit_item_count + eval_item_count)
        ],
    }
    mc_records = []
    for record in split_records["val"] + split_records["test"]:
        tokens = record["question"].split()
        mc_records.append(
            {
                **record,
                "cumulative_prefixes": [
                    " ".join(tokens[:count])
                    for count in (1, 3, 5, 7, 9, 10)
                ],
            }
        )
    raw_payloads = {
        "mc_dataset.json": (
            json.dumps({"questions": mc_records}, sort_keys=True) + "\n"
        ).encode("utf-8"),
        **{
            f"{split}_dataset.json": (
                json.dumps({"questions": records}, sort_keys=True) + "\n"
            ).encode("utf-8")
            for split, records in split_records.items()
        },
        "build_metadata.json": (bundle / "build_metadata.json").read_bytes(),
    }
    for role in RAW_INPUT_ROLES:
        raw_payloads.setdefault(role, role.encode("utf-8"))
        (raw_bundle / role).write_bytes(raw_payloads[role])
    raw_files = [
        {
            "role": role,
            "size": len(raw_payloads[role]),
            "sha256": sha256_bytes(raw_payloads[role]),
        }
        for role in RAW_INPUT_ROLES
    ]
    model_payload = b"model.bin"
    (model_content / "model.bin").write_bytes(model_payload)
    input_manifests = {
        "source_manifest": build_manifest(
            source_manifest_identity(
                git_sha="a" * 40,
                files=source_files,
                pyproject_sha256=sha256_bytes(source_payloads["pyproject.toml"]),
                uv_lock_sha256=sha256_bytes(source_payloads["uv.lock"]),
            )
        ),
        "raw_input_manifest": build_manifest(
            raw_input_identity(
                files=raw_files,
                semantic_checks={
                    "all_semantic_checks_pass": True,
                    "question_trajectory_binding_id": trajectory_id,
                },
            )
        ),
        "model_snapshot_manifest": build_manifest(
            model_snapshot_identity(
                model_id=ADAPTER_SCORING_SPEC["model_id"],
                revision="b" * 40,
                files=[
                    {
                        "path": "model.bin",
                        "size": len(model_payload),
                        "sha256": sha256_bytes(model_payload),
                    }
                ],
                sentence_transformers_version="fixture",
                transformers_version="fixture",
            )
        ),
    }
    input_manifest_paths = {
        "source_manifest": source_bundle / "source_manifest.json",
        "raw_input_manifest": raw_bundle / "raw_input_manifest.json",
        "model_snapshot_manifest": model_bundle / "model_snapshot_manifest.json",
    }
    for role, input_manifest in input_manifests.items():
        input_manifest_path = input_manifest_paths[role]
        input_manifest_path.write_text(
            json.dumps(input_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    source_id = input_manifests["source_manifest"]["id"]
    raw_id = input_manifests["raw_input_manifest"]["id"]
    model_id = input_manifests["model_snapshot_manifest"]["id"]
    bound_calibration = derive_bound_calibration(
        fit_rows=fit_rows,
        eval_rows=eval_rows,
        model_snapshot_id=model_id,
        fit_rows_sha256=fit_sha,
    )
    (bundle / "calibration.json").write_text(
        json.dumps(bound_calibration, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    calibration_sha = sha256_file(bundle / "calibration.json")
    eval_mc_prefixes = {
        (r["item_id"], r["prefix_idx"])
        for r in eval_rows
        if r["format"] == "MC"
    }
    eval_qa_prefixes = {
        (r["item_id"], r["prefix_idx"])
        for r in eval_rows
        if r["format"] == "QA"
    }
    adapter_ident = adapter_identity(
        source_manifest_id=source_id, raw_input_bundle_id=raw_id, model_snapshot_id=model_id,
        scoring_spec=dict(ADAPTER_SCORING_SPEC), fit_split="val", eval_split="test",
        schema_columns=ADAPTER_SCHEMA_COLUMNS,
        fit_row_count=len(fit_rows), eval_row_count=len(eval_rows),
        fit_rows_sha256=fit_sha, eval_rows_sha256=eval_sha,
        calibration_sha256=calibration_sha,
        question_trajectory_binding_id=trajectory_id,
        mc_coverage={
            "eval_mc_items": len(
                {item_id for item_id, _ in eval_mc_prefixes}
            ),
            "eval_qa_items": len(
                {item_id for item_id, _ in eval_qa_prefixes}
            ),
            "paired": eval_mc_prefixes == eval_qa_prefixes,
        },
        mc_retention={
            "build_metadata_sha256": build_metadata_sha,
            "threshold_profile": "full",
            "splits": {
                "fit": {
                    "applies": True,
                    "split": "val",
                    "threshold": "0.98",
                    "retention_rate": "1.0",
                    "raw_count": len({row["item_id"] for row in fit_rows}),
                    "retained_count": len(
                        {row["item_id"] for row in fit_rows}
                    ),
                    "dropped_count": 0,
                    "passed": True,
                    "overridden": False,
                    "override_flag": "--allow-low-mc-retention",
                    "effective_pass": True,
                },
                "eval": {
                    "applies": True,
                    "split": "test",
                    "threshold": "0.98",
                    "retention_rate": "1.0",
                    "raw_count": len({row["item_id"] for row in eval_rows}),
                    "retained_count": len(
                        {row["item_id"] for row in eval_rows}
                    ),
                    "dropped_count": 0,
                    "passed": True,
                    "overridden": False,
                    "override_flag": "--allow-low-mc-retention",
                    "effective_pass": True,
                },
            },
            "fit_rows": len(fit_rows),
            "eval_rows": len(eval_rows),
        },
        producer_hashes={
            "adapter_build.py": source_producer_hashes["adapter_build.py"]
        },
    )
    adapter_man = build_manifest(adapter_ident)
    (bundle / "manifest.json").write_text(json.dumps(adapter_man, indent=2, sort_keys=True), encoding="utf-8")
    adapter_id = adapter_man["id"]

    test_items = sorted({r["item_id"] for r in eval_rows if r["format"] == "MC"} &
                        {r["item_id"] for r in eval_rows if r["format"] == "QA"})
    plan = build_bootstrap_plan(test_items, replicates=100, seed=1)
    bootstrap_id = compute_id(plan_identity(plan))

    if _SYNTH_FVI_STUDY is None:
        _SYNTH_FVI_STUDY = run_fvi_study(
            rows=rows,
            calibration_json=bound_calibration,
        )
    study = json.loads(json.dumps(_SYNTH_FVI_STUDY))
    selected_fvi = study["selected_parameters"]
    fvi_manifest = build_manifest(
        {
            "kind": "fvi_study_fixed",
            "adapter_bundle_id": adapter_id,
            "selected": selected_fvi,
        }
        if fixed_fvi
        else fvi_study_identity(
            adapter_bundle_id=adapter_id,
            candidate_grid=study["candidate_grid"],
            representative_generator=study["representative_cell_generator"],
            candidate_results=study["candidate_convergence_results"],
            strict_reference_results=study["strict_reference"],
            selector_rule=study["selector_rule"],
            selected_parameters=selected_fvi,
            all96_validation=study["all96_fit_only_validation"],
            producer_hashes=fvi_producer_hashes,
        )
    )
    fvi_study_id = fvi_manifest["id"]
    environment = {
        "python_version": "3.11.0",
        "package_versions": {
            name: np.__version__ if name == "numpy" else "synthetic-test"
            for name in ENVIRONMENT_PACKAGES
        },
    }
    environment_manifest = build_manifest(
        environment_contract_identity(**environment)
    )
    env_id = environment_manifest["id"]
    run_producers = {
        "checker.py": source_producer_hashes["checker.py"],
        "sweep.py": source_producer_hashes["sweep.py"],
    }
    myopic_sha256 = _hex("7")
    run_spec = run_spec_identity(
        source_manifest_id=source_id, raw_input_bundle_id=raw_id, model_snapshot_id=model_id,
        adapter_bundle_id=adapter_id, fvi_study_id=fvi_study_id, bootstrap_plan_id=bootstrap_id,
        environment_contract_id=env_id,
        resource_summary_id=compute_id({"backend": "modal", "usd": 0}),
        fvi_selected=selected_fvi,
        replicate_count=100, profile_variant="smoke",
        myopic_artifact_sha256=myopic_sha256,
        producer_hashes=run_producers,
        prerequisite_receipts={},
    )
    run_spec_id = compute_id(run_spec)

    run_root = base_dir / "run"
    ctx = SweepContext(
        rows=rows, calibration_json=bound_calibration, run_spec=run_spec, run_spec_id=run_spec_id,
        bootstrap_plan=plan, output_dir=run_root,
        fvi_tolerance=selected_fvi["tolerance"],
        fvi_max_iterations=selected_fvi["max_iterations"],
        backend="modal", profile_variant="smoke", adapter_bundle_id=adapter_id,
        adapter_fit_rows_sha256=fit_sha,
        adapter_eval_rows_sha256=eval_sha, myopic_artifact_sha256=myopic_sha256,
        producer_hashes=run_producers, cells=smoke_cells(),
        environment=environment, resource_summary={"backend": "modal", "usd": 0},
        attempt={"attempt": 1, "mode": "fresh", "command": ["dp_sweep"],
                 "run_spec_id": run_spec_id, "adapter_id": adapter_id},
    )
    aggregate = run_sweep(ctx)
    fvi_bytes = (
        json.dumps(fvi_manifest, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    environment_bytes = (
        json.dumps(environment_manifest, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    external_artifacts = [
        {
            "role": role,
            "content_id": input_manifests[role]["id"],
            "sha256": sha256_file(input_manifest_paths[role]),
            "byte_size": input_manifest_paths[role].stat().st_size,
            "retrieval_path": str(input_manifest_paths[role]),
        }
        for role in (
            "source_manifest",
            "raw_input_manifest",
            "model_snapshot_manifest",
        )
    ]
    external_artifacts.extend(
        [
            {
                "role": "fvi_study",
                "content_id": fvi_study_id,
                "sha256": sha256_bytes(fvi_bytes),
                "byte_size": len(fvi_bytes),
                "retrieval_path": "evidence/fvi_study.json",
            },
            {
                "role": "environment_contract",
                "content_id": env_id,
                "sha256": sha256_bytes(environment_bytes),
                "byte_size": len(environment_bytes),
                "retrieval_path": "evidence/environment_contract.json",
            },
        ]
    )
    if not final_variant:
        package_run(
            run_root, aggregate, resource_summary={"backend": "modal", "usd": 0},
            external_artifacts=external_artifacts,
            evidence_files={
                "evidence/fvi_study.json": fvi_bytes,
                "evidence/environment_contract.json": environment_bytes,
            },
        )
        return {"run_root": run_root, "adapter_bundle": bundle, "aggregate": aggregate,
                "run_spec_id": run_spec_id}

    # Final-profile variant: mint the three prerequisite receipts through the
    # production writers path, then run and package the 96-cell final profile.
    receipt_ids = _mint_final_prerequisite_receipts(
        base_dir,
        full_bindings={
            "source_manifest_id": source_id,
            "raw_input_bundle_id": raw_id,
            "model_snapshot_id": model_id,
            "adapter_bundle_id": adapter_id,
            "fvi_study_id": fvi_study_id,
            "environment_contract_id": env_id,
        },
        smoke_spec=run_spec,
        smoke_spec_id=run_spec_id,
        smoke_aggregate=json.loads(
            (run_root / "aggregate.json").read_text(encoding="utf-8")
        ),
        adapter_manifest=adapter_man,
        determinism_file_hashes={
            "fit_rows.jsonl.gz": fit_sha,
            "eval_rows.jsonl.gz": eval_sha,
            "calibration.json": calibration_sha,
            "build_metadata.json": build_metadata_sha,
        },
    )
    final_plan = build_bootstrap_plan(
        test_items, replicates=FINAL_REPLICATES, seed=1
    )
    final_bootstrap_id = compute_id(plan_identity(final_plan))
    final_spec = run_spec_identity(
        source_manifest_id=source_id, raw_input_bundle_id=raw_id,
        model_snapshot_id=model_id, adapter_bundle_id=adapter_id,
        fvi_study_id=fvi_study_id, bootstrap_plan_id=final_bootstrap_id,
        environment_contract_id=env_id,
        resource_summary_id=compute_id({"backend": "modal", "usd": 0}),
        fvi_selected=selected_fvi,
        replicate_count=FINAL_REPLICATES, profile_variant="final",
        myopic_artifact_sha256=myopic_sha256,
        producer_hashes=run_producers,
        prerequisite_receipts=receipt_ids,
    )
    final_spec_id = compute_id(final_spec)
    # package_run resolves receipts from run_root.parents[1] / "receipts", so
    # the final run root sits one level deeper than the smoke run root.
    final_root = base_dir / "runs" / "run"
    final_ctx = SweepContext(
        rows=rows, calibration_json=bound_calibration, run_spec=final_spec,
        run_spec_id=final_spec_id, bootstrap_plan=final_plan,
        output_dir=final_root,
        fvi_tolerance=selected_fvi["tolerance"],
        fvi_max_iterations=selected_fvi["max_iterations"],
        backend="modal", profile_variant="final", adapter_bundle_id=adapter_id,
        adapter_fit_rows_sha256=fit_sha,
        adapter_eval_rows_sha256=eval_sha,
        myopic_artifact_sha256=myopic_sha256,
        producer_hashes=run_producers, cells=full_grid(),
        environment=environment,
        resource_summary={"backend": "modal", "usd": 0},
        attempt={"attempt": 1, "mode": "fresh", "command": ["dp_sweep"],
                 "run_spec_id": final_spec_id, "adapter_id": adapter_id},
    )
    final_aggregate = run_sweep(final_ctx)
    package_run(
        final_root, final_aggregate,
        resource_summary={"backend": "modal", "usd": 0},
        external_artifacts=external_artifacts,
        evidence_files={
            "evidence/fvi_study.json": fvi_bytes,
            "evidence/environment_contract.json": environment_bytes,
        },
    )
    return {"run_root": final_root, "adapter_bundle": bundle,
            "aggregate": final_aggregate, "run_spec_id": final_spec_id,
            "prerequisite_receipt_ids": receipt_ids}


def _mint_final_prerequisite_receipts(
    base_dir: Path,
    *,
    full_bindings: dict[str, str],
    smoke_spec: dict[str, Any],
    smoke_spec_id: str,
    smoke_aggregate: dict[str, Any],
    adapter_manifest: dict[str, Any],
    determinism_file_hashes: dict[str, str],
) -> dict[str, str]:
    """Mint smoke/mutation/determinism receipts through the production path.

    Returns
    -------
    dict[str, str]
        Gate-to-receipt-ID mapping for the final run spec.
    """
    determinism_bindings = {
        key: full_bindings[key] for key in sorted(DETERMINISM_BINDINGS)
    }
    source_execution = {
        "environment": "local_clean_worktree",
        "executing_source_manifest_id": full_bindings["source_manifest_id"],
        "runtime_source_manifest_id": full_bindings["source_manifest_id"],
    }

    def _build_execution(execution_id: str, subdir: str) -> dict[str, Any]:
        return {
            "environment": "local_process",
            "execution_id": execution_id,
            "adapter_subdir": subdir,
            **determinism_bindings,
            "cached": False,
            "output_sha256": determinism_file_hashes,
        }

    gate_inputs = {
        "smoke": (
            full_bindings,
            {
                "run_spec": {"id": smoke_spec_id, "identity": smoke_spec},
                "aggregate": smoke_aggregate,
            },
        ),
        "mutation": (
            full_bindings,
            {
                "source_execution": source_execution,
                "results": [
                    {
                        "mutation": name,
                        "expected": "PASS" if index == 0 else "REJECT",
                        "passed_check": index == 0,
                        "ok": True,
                        "errors": [],
                    }
                    for index, name in enumerate(MUTATION_ROSTER)
                ],
            },
        ),
        "determinism": (
            determinism_bindings,
            {
                "source_execution": source_execution,
                "first_build_execution": _build_execution(
                    "selftest-determinism-first", "first"
                ),
                "second_build_execution": _build_execution(
                    "selftest-determinism-second", "second"
                ),
                "first_adapter_manifest": adapter_manifest,
                "second_adapter_manifest": adapter_manifest,
                "first_file_sha256": determinism_file_hashes,
                "second_file_sha256": determinism_file_hashes,
            },
        ),
    }
    receipt_ids: dict[str, str] = {}
    for gate, (bindings, details) in gate_inputs.items():
        evidence = build_prerequisite_evidence(
            gate=gate, bindings=bindings, details=details
        )
        receipt = build_evidenced_prerequisite_receipt(
            gate=gate, bindings=bindings, evidence=evidence
        )
        gate_dir = base_dir / "receipts" / gate
        gate_dir.mkdir(parents=True, exist_ok=True)
        (gate_dir / f"{receipt['id']}.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (gate_dir / f"{receipt['id']}.evidence.json").write_bytes(
            prerequisite_evidence_bytes(evidence)
        )
        receipt_ids[gate] = receipt["id"]
    return receipt_ids
