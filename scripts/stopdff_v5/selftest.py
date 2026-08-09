"""Negative mutation suite + valid-package builder (see ACCEPTANCE_CONTRACT.md).

build_valid_package() creates a self-contained synthetic package (adapter bundle + run
package) that the checker accepts. run_self_test() applies a battery of mutations and
asserts the checker REJECTS every one. Synthetic fixtures only; real-data mutation gate
runs the same logic on Modal against the real package.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Callable

import numpy as np

from . import checker
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
from .profile import smoke_cells
from .rowio import write_jsonl_gz
from .sweep import SweepContext, run_sweep
from .writers import package_run

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


def build_valid_package(base_dir: Path) -> dict[str, Any]:
    global _SYNTH_FVI_STUDY
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
        fvi_study_identity(
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


# --- mutations --------------------------------------------------------------------


def _load(p: Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))


def _save(p: Path, obj: Any) -> None:
    p.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def _first_cell(run_root: Path) -> Path:
    return sorted((run_root / "cells").glob("*.json"))[0]


def _mut_stale_cache(rr, b):
    p = _first_cell(rr); rec = _load(p)
    k = next(iter(rec["index_shift_by_item"]))
    rec["index_shift_by_item"][k] = int(rec["index_shift_by_item"][k]) + 3
    _save(p, rec)


def _mut_flip_verdict(rr, b):
    p = _first_cell(rr); rec = _load(p)
    flip = {"PASS": "FAIL", "WARN": "PASS", "FAIL": "PASS"}
    rec["verdict"] = flip.get(rec.get("verdict"), "PASS")
    _save(p, rec)


def _mut_hide_coverage(rr, b):
    p = _first_cell(rr); rec = _load(p)
    rec["coverage"]["clean"] = not rec["coverage"]["clean"]
    _save(p, rec)


def _mut_tamper_ceiling(rr, b):
    p = _first_cell(rr); rec = _load(p)
    key = next(iter(rec["ceiling_flags"]))
    rec["ceiling_flags"][key] = not rec["ceiling_flags"][key]
    _save(p, rec)


def _mut_wrong_family_ci(rr, b):
    agg = _load(rr / "aggregate.json"); agg["family"]["ci"] = [99.0, 100.0]; _save(rr / "aggregate.json", agg)


def _mut_wrong_family_verdict(rr, b):
    agg = _load(rr / "aggregate.json")
    agg["family"]["verdict"] = "FAIL" if agg["family"]["verdict"] != "FAIL" else "PASS"
    _save(rr / "aggregate.json", agg)


def _mut_wrong_release(rr, b):
    agg = _load(rr / "aggregate.json")
    agg["release_status"] = "INVALID" if agg["release_status"] == "VALID" else "VALID"
    _save(rr / "aggregate.json", agg)


def _mut_dual_backend(rr, b):
    shutil.copy(rr / "run_manifest.json", rr / "command_manifest.json")


def _mut_missing_backend_manifest(rr, b):
    (rr / "run_manifest.json").unlink()


def _mut_wrong_seed(rr, b):
    pm = _load(rr / "bootstrap_plan.json"); pm["identity"]["seed"] = 2
    pm["id"] = compute_id(pm["identity"]); _save(rr / "bootstrap_plan.json", pm)


def _mut_wrong_replicates(rr, b):
    pm = _load(rr / "bootstrap_plan.json"); pm["identity"]["replicate_count"] = 999
    pm["id"] = compute_id(pm["identity"]); _save(rr / "bootstrap_plan.json", pm)


def _mut_tamper_run_spec_id(rr, b):
    sm = _load(rr / "run_spec.json"); sm["id"] = _hex("0"); _save(rr / "run_spec.json", sm)


def _mut_tamper_plan_hash(rr, b):
    pm = _load(rr / "bootstrap_plan.json")
    pm["identity"]["resample_index_sha256"] = _hex("0")
    pm["id"] = compute_id(pm["identity"]); _save(rr / "bootstrap_plan.json", pm)


def _mut_fresh_with_resume(rr, b):
    (rr / "attempts.jsonl").write_text(
        json.dumps({"attempt": 1, "mode": "fresh", "command": ["dp_sweep", "--resume"]}) + "\n",
        encoding="utf-8")


def _mut_resume_without_bare(rr, b):
    with open(rr / "attempts.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"attempt": 2, "mode": "resume", "command": ["dp_sweep"]}) + "\n")


def _mut_overwrite(rr, b):
    with open(rr / "attempts.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"attempt": 2, "mode": "resume", "command": ["dp_sweep", "--resume", "--overwrite"]}) + "\n")


def _mut_unsafe_checksum(rr, b):
    with open(rr / "SHA256SUMS", "a", encoding="utf-8") as f:
        f.write(f"{_hex('a')}  ../evil.txt\n")


def _mut_duplicate_checksum(rr, b):
    lines = (rr / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    lines.append(lines[0])
    (rr / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _mut_symlink(rr, b):
    target = rr / "aggregate.json"
    link = rr / "evil_link.json"
    os.symlink(target, link)
    with open(rr / "SHA256SUMS", "a", encoding="utf-8") as f:
        f.write(f"{sha256_file(target)}  evil_link.json\n")


def _mut_checksum_value(rr, b):
    lines = (rr / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    parts = lines[0].split()
    lines[0] = f"{_hex('0')}  {parts[1]}"
    (rr / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _mut_invalid_png(rr, b):
    for png in (rr / "figures").glob("*.png"):
        png.write_bytes(b"not a png")
        break


def _mut_truncated_png_after_ihdr(rr, b):
    for png in (rr / "figures").glob("*.png"):
        png.write_bytes(png.read_bytes()[:24])
        break


def _mut_missing_external_artifacts(rr, b):
    (rr / "external_artifacts.json").unlink()


def _mut_unconverged_completed(rr, b):
    agg = _load(rr / "aggregate.json"); agg["fvi_selected"]["max_iterations"] = 1
    _save(rr / "aggregate.json", agg)


def _mut_fingerprint(rr, b):
    p = _first_cell(rr); rec = _load(p); rec["fingerprint_id"] = _hex("0"); _save(p, rec)


def _mut_adapter_row_hash(rr, b):
    # corrupt fit rows after manifest was written -> validate-adapter must reject
    with open(b / "fit_rows.jsonl.gz", "ab") as f:
        f.write(b"\x00corrupt")


def _mut_adapter_calibration_hash(rr, b):
    with open(b / "calibration.json", "ab") as handle:
        handle.write(b"\n")


def _mut_backend_adapter_binding(rr, b):
    path = rr / "run_manifest.json"
    record = _load(path)
    record["identity"]["adapter_bundle_id"] = _hex("9")
    record["id"] = compute_id(record["identity"])
    _save(path, record)


def _mut_attempt_adapter_binding(rr, b):
    path = rr / "attempts.jsonl"
    attempt = json.loads(path.read_text(encoding="utf-8").strip())
    attempt["adapter_id"] = _hex("9")
    path.write_text(json.dumps(attempt) + "\n", encoding="utf-8")


def _mut_cell_adapter_binding(rr, b):
    path = _first_cell(rr)
    record = _load(path)
    record["adapter_bundle_id"] = _hex("9")
    _save(path, record)


def _mut_fingerprint_adapter_hash(rr, b):
    path = _first_cell(rr)
    record = _load(path)
    record["fingerprint_identity"]["adapter_fit_rows_sha256"] = _hex("9")
    record["fingerprint_id"] = compute_id(record["fingerprint_identity"])
    _save(path, record)


def _mut_aggregate_adapter_binding(rr, b):
    path = rr / "aggregate.json"
    aggregate = _load(path)
    aggregate["adapter_bundle_id"] = _hex("9")
    _save(path, aggregate)


def _mut_aggregate_fvi_binding(rr, b):
    path = rr / "aggregate.json"
    aggregate = _load(path)
    aggregate["fvi_selected"]["max_iterations"] = 200
    _save(path, aggregate)


def _mut_unknown_attempt_mode(rr, b):
    path = rr / "attempts.jsonl"
    attempt = json.loads(path.read_text(encoding="utf-8").strip())
    attempt["mode"] = "replay"
    path.write_text(json.dumps(attempt) + "\n", encoding="utf-8")


def _mut_fingerprint_kind(rr, b):
    path = _first_cell(rr)
    record = _load(path)
    record["fingerprint_identity"]["kind"] = "forged"
    record["fingerprint_id"] = compute_id(record["fingerprint_identity"])
    _save(path, record)


def _mut_fingerprint_producers(rr, b):
    path = _first_cell(rr)
    record = _load(path)
    record["fingerprint_identity"]["producer_hashes"] = {
        "checker.py": _hex("f"),
        "sweep.py": _hex("f"),
    }
    record["fingerprint_id"] = compute_id(record["fingerprint_identity"])
    _save(path, record)


def _mut_cell_gate_override(rr, b):
    path = _first_cell(rr)
    record = _load(path)
    record["mc_gate_overridden"] = not record["mc_gate_overridden"]
    _save(path, record)


def _mut_backend_environment(rr, b):
    path = rr / "run_manifest.json"
    record = _load(path)
    record["environment"] = {
        "python_version": "0.0.0",
        "package_versions": {"numpy": "0"},
    }
    _save(path, record)


def _mut_missing_fvi_evidence(rr, b):
    path = rr / "external_artifacts.json"
    record = _load(path)
    record["artifacts"] = [
        artifact
        for artifact in record["artifacts"]
        if artifact.get("role") != "fvi_study"
    ]
    _save(path, record)


def _mut_missing_attempt_result(rr, b):
    next((rr / "attempt_results").glob("*.json")).unlink()


def _mut_attempt_result_counts(rr, b):
    path = next((rr / "attempt_results").glob("*.json"))
    record = _load(path)
    record["completed"] = 999
    _save(path, record)


_RUN_MUTATIONS: dict[str, Callable] = {
    "stale_cache": _mut_stale_cache,
    "cell_verdict_serialized_not_trusted": _mut_flip_verdict,
    "coverage_clean_serialized_not_trusted": _mut_hide_coverage,
    "ceiling_flags_tampered": _mut_tamper_ceiling,
    "wrong_family_maximum_statistic": _mut_wrong_family_ci,
    "family_verdict_hides_cell_warn": _mut_wrong_family_verdict,
    "wrong_release_status": _mut_wrong_release,
    "dual_backend_manifests": _mut_dual_backend,
    "missing_backend_manifest": _mut_missing_backend_manifest,
    "wrong_bootstrap_seed": _mut_wrong_seed,
    "wrong_bootstrap_count": _mut_wrong_replicates,
    "tampered_run_spec_id": _mut_tamper_run_spec_id,
    "wrong_bootstrap_plan_hash": _mut_tamper_plan_hash,
    "fresh_attempt_with_resume": _mut_fresh_with_resume,
    "resume_without_bare_resume": _mut_resume_without_bare,
    "overwrite_in_evidence_run": _mut_overwrite,
    "unsafe_checksum_traversal": _mut_unsafe_checksum,
    "duplicate_checksum_entry": _mut_duplicate_checksum,
    "symlink_in_package": _mut_symlink,
    "checksum_value_mismatch": _mut_checksum_value,
    "invalid_png": _mut_invalid_png,
    "truncated_png_after_ihdr": _mut_truncated_png_after_ihdr,
    "missing_external_artifacts": _mut_missing_external_artifacts,
    "unconverged_fvi_marked_completed": _mut_unconverged_completed,
    "cell_fingerprint_tampered": _mut_fingerprint,
    "adapter_calibration_bytes_tampered": _mut_adapter_calibration_hash,
    "backend_adapter_binding": _mut_backend_adapter_binding,
    "attempt_adapter_binding": _mut_attempt_adapter_binding,
    "cell_adapter_binding": _mut_cell_adapter_binding,
    "fingerprint_adapter_hash_binding": _mut_fingerprint_adapter_hash,
    "aggregate_adapter_binding": _mut_aggregate_adapter_binding,
    "aggregate_fvi_binding": _mut_aggregate_fvi_binding,
    "unknown_attempt_mode": _mut_unknown_attempt_mode,
    "fingerprint_kind": _mut_fingerprint_kind,
    "fingerprint_producer_binding": _mut_fingerprint_producers,
    "cell_gate_override": _mut_cell_gate_override,
    "backend_environment_binding": _mut_backend_environment,
    "missing_fvi_evidence": _mut_missing_fvi_evidence,
    "missing_attempt_result": _mut_missing_attempt_result,
    "attempt_result_counts": _mut_attempt_result_counts,
}


def run_self_test(base_dir: Path) -> tuple[bool, list[dict[str, Any]]]:
    base_dir = Path(base_dir)
    valid_dir = base_dir / "valid"
    built = build_valid_package(valid_dir)

    results: list[dict[str, Any]] = []
    baseline = checker.validate_run(
        built["run_root"], backend="modal", adapter_bundle=built["adapter_bundle"],
        require_final_profile=False, require_package=True,
    )
    results.append({"mutation": "<baseline valid>", "expected": "PASS",
                    "passed_check": baseline.passed, "ok": baseline.passed,
                    "errors": baseline.errors[:3]})

    all_ok = baseline.passed

    for name, fn in _RUN_MUTATIONS.items():
        mdir = base_dir / f"mut_{name}"
        if mdir.exists():
            shutil.rmtree(mdir)
        shutil.copytree(valid_dir, mdir, symlinks=True)
        rr, bundle = mdir / "run", mdir / "adapter_bundle"
        fn(rr, bundle)
        res = checker.validate_run(
            rr, backend="modal", adapter_bundle=bundle,
            require_final_profile=False, require_package=True,
        )
        rejected = not res.passed
        results.append({"mutation": name, "expected": "REJECT", "passed_check": res.passed,
                        "ok": rejected, "errors": res.errors[:2]})
        all_ok = all_ok and rejected

    # adapter-level mutation
    adir = base_dir / "mut_adapter_row_hash"
    if adir.exists():
        shutil.rmtree(adir)
    shutil.copytree(valid_dir, adir, symlinks=True)
    _mut_adapter_row_hash(adir / "run", adir / "adapter_bundle")
    ares = checker.validate_adapter(adir / "adapter_bundle")
    results.append({"mutation": "invalid_adapter_row_hash", "expected": "REJECT",
                    "passed_check": ares.passed, "ok": not ares.passed, "errors": ares.errors[:2]})
    all_ok = all_ok and not ares.passed

    return all_ok, results
