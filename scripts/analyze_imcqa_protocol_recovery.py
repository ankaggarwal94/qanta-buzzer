#!/usr/bin/env python3
"""Independent cached-batch-two recovery audit and combined two-model analysis.

The original validator and failed run remain immutable. Only the 3B recovery
has a separate execution contract; unchanged 7B is validated by the original
validator before the two normalized datasets are analyzed together.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any
from scripts import analyze_imcqa_protocol_pilot as core
from scripts import analyze_imcqa_protocol_partial as partial

old = core.old
PROTOCOL, ACTIONS, MENUS, SPLITS = core.PROTOCOL, core.ACTIONS, core.MENUS, core.SPLITS
OLD_PUBLIC_SHA = core.OLD_PUBLIC_SHA
score_view, compare_numeric, _only_attempt = core.score_view, core.compare_numeric, core._only_attempt
BASE_SOURCE_FILES = partial.SOURCE_FILES
RECOVERY_SOURCE_FILES = BASE_SOURCE_FILES + ("scripts/imcqa_3b_protocol_recovery.py", "configs/imcqa_3b_protocol_recovery.json")
TRIAL_SOURCE_FILES = BASE_SOURCE_FILES + ("scripts/imcqa_3b_numerical_diagnostic.py", "configs/imcqa_3b_numerical_diagnostic.json")
EXECUTION_PROTOCOL = "imcqa_3b_protocol_recovery_cached2_v1"
DIAGNOSTIC_RECEIPT_SHA = "33826498509e73e11f5f4f0e213f836b4aa8f2b2d29b0eb134923ed03bb2d705"
PREDECESSOR_RECEIPT_SHA = "27d0004b1e8a98821d32a1601632637b905c555ef3967664739743837e7fbf45"
PREDECESSOR_PLAN_SHA = "367755bc172c5abdad7bd6f4633c5d107643bda5b5f439de94889c4792e90e1e"
RECOVERY_LAUNCH_COMMIT = "1bc6ae00d33eb681c3f626513af471f2d24d7a55"


def validate_source_hashes(metadata, names, source_root):
    expected = {name: old.sha256(source_root / name) for name in names}
    if metadata.get("source_files_sha256") != expected:
        raise ValueError("source allowlist or file hashes differ")
    return expected


def production_pair_ids(ordered_ids, failing_sample_ids):
    """Include both actual batch-two neighbors for each frozen diagnostic ID."""
    if len(ordered_ids) % 2 or len(set(ordered_ids)) != len(ordered_ids) or len(set(failing_sample_ids)) != len(failing_sample_ids):
        raise ValueError("production or diagnostic identities are duplicated/incomplete")
    positions = {key: index for index, key in enumerate(ordered_ids)}
    if not set(failing_sample_ids) <= set(positions):
        raise ValueError("diagnostic state is missing from production plan")
    pair_starts = sorted({positions[key] // 2 * 2 for key in failing_sample_ids})
    return [ordered_ids[index + offset] for index in pair_starts for offset in (0, 1)]


def validate_production_raw(directory, summary, ids, production, cached, singles):
    expected = {"score_ids": ids, "production": production, "diagnostic": cached, "single_reference": singles}
    if any(summary.get(key) != value for key, value in expected.items()):
        raise ValueError("production diagnostic raw evidence differs")
    if _only_attempt(directory, "production_diagnostics_raw") != expected:
        raise ValueError("pre-gate production evidence differs")


def validate_active_raw(directory, rows, new_lookup, jobs, attempt=0):
    """Bind every visited state to exactly one pre-gate forward artifact."""
    expected_names = {f"{attempt:03d}_active_{index:03d}_raw.json" for index in range(len(rows))}
    actual_names = {path.name for path in (directory / "attempts").glob("*_active_*_raw.json")}
    if actual_names != expected_names:
        raise ValueError("active pre-gate raw evidence coverage differs")
    for index, row in enumerate(rows):
        filename = f"{attempt:03d}_active_{index:03d}_raw.json"
        path = directory / "attempts" / filename
        if row.get("raw_file") != filename or row.get("raw_file_sha256") != old.sha256(path):
            raise ValueError("active pre-gate filename or hash differs")
        job = jobs[row["score_id"]]
        source = new_lookup[row["score_id"]]
        expected = {"score_id": row["score_id"], "sequence_index": index,
            **{key: job[key] for key in ("qid", "condition", "block", "rotation", "round")},
            "live": row["live"], "production": {"logits": [source["raw_action_logits"][label] for label in ACTIONS]}}
        if old.load_json(path) != expected:
            raise ValueError("active pre-gate state, sequence, or forward evidence differs")


def selected_trial_mode(candidates, jobs, reference, numerical_config):
    """Recompute all comparisons and select the predeclared minimum runtime."""
    expected_names = {f"{mode}_{size}" for mode in ("uncached", "cached") for size in (2, 4, 8)}
    if {row["name"] for row in candidates} != expected_names or len(candidates) != 6:
        raise ValueError("six declared numerical candidate modes required")
    checks = {}
    for row in candidates:
        expected_name = ("cached" if row["cached"] else "uncached") + f'_{row["batch_size"]}'
        if expected_name != row["name"] or row["batch_size"] not in (2, 4, 8):
            raise ValueError("numerical candidate execution mode differs")
        elapsed = row["elapsed_forward_seconds"]
        if (type(elapsed) not in (float, int) or not math.isfinite(elapsed) or elapsed <= 0
            or any(type(value) not in (float, int) or not math.isfinite(value) or value <= 0 for value in row["batch_seconds"])
            or not math.isclose(math.fsum(row["batch_seconds"]), elapsed, rel_tol=1e-10, abs_tol=1e-10)):
            raise ValueError("numerical candidate timing evidence differs")
        checks[row["name"]] = compare_numeric(row["outputs"], reference, jobs, numerical_config)
        if row.get("diagnostic_compatible") is not True or row.get("production_approved") is not False:
            raise ValueError("diagnostic-only compatibility status differs")
    selected = min(candidates, key=lambda row: (row["elapsed_forward_seconds"], row["name"]))["name"]
    if selected != "cached_2":
        raise ValueError("recovery mode is not the predeclared fastest compatible observed candidate")
    return selected, checks


def validate_numerical_trial(directory, public_jobs, config, predecessor_directory, prior_directory, prior_rows, prior_public_path, public_path):
    """Audit the small diagnostic that selected the recovery execution mode."""
    receipt_path = directory / "receipt.json"
    if old.sha256(receipt_path) != DIAGNOSTIC_RECEIPT_SHA:
        raise ValueError("numerical trial receipt differs from frozen recovery predecessor")
    receipt = old.load_json(receipt_path)
    required = {"status": "complete", "model_tag": "qwen3b", "reference_valid": True,
        "production_rows": 0, "production_approved": False, "automatic_retries": 0, "generation": False, "sampling": False,
        "protocol": "imcqa_3b_numerical_shape_diagnostic_v1", "logical_evaluations": 200, "model_forward_calls": 117}
    if any(receipt.get(key) != value for key, value in required.items()):
        raise ValueError("numerical trial receipt contract differs")
    for name, digest in receipt["output_sha256"].items():
        if Path(name).name != name or old.sha256(directory / name) != digest:
            raise ValueError("numerical trial retained output differs")
    plan = old.load_json(directory / "plan.json")
    manifest = old.load_json(directory.parent.parent / "manifest.json")
    if plan["manifest"] != manifest or old.sha256(directory.parent.parent / "manifest.json") != receipt["manifest_sha256"]:
        raise ValueError("numerical trial manifest differs")
    for name, digest in manifest["source_evidence_sha256"].items():
        namespace, relative = name.split(":", 1)
        if relative == "public/pilot.json":
            path = prior_public_path if namespace == "prior" else public_path
        elif namespace == "prior" and relative.startswith("output/qwen3b/"):
            path = prior_directory / relative.removeprefix("output/qwen3b/")
        elif namespace == "protocol" and relative.startswith("output/qwen3b/"):
            path = predecessor_directory / relative.removeprefix("output/qwen3b/")
        else:
            raise ValueError("unexpected diagnostic source namespace")
        if old.sha256(path) != digest:
            raise ValueError("diagnostic prior/source identity differs")
    lookup = {job["score_id"]: job for job in public_jobs}
    jobs = plan["jobs"]
    if len(jobs) != 20 or len({job["score_id"] for job in jobs}) != 20:
        raise ValueError("numerical trial must retain twenty unique contexts")
    for job in jobs:
        expected = {key: value for key, value in lookup[job["score_id"]].items() if key not in {"canonical_gold_option_id", "expected_semantic_action"}}
        if job != expected:
            raise ValueError("numerical trial public prompt differs")
    contexts = plan["contexts"]
    records = manifest["original_contexts"] + manifest["overlap_contexts"]
    if len(contexts) != 20 or len(records) != 20:
        raise ValueError("diagnostic context evidence incomplete")
    for job, context, record in zip(jobs, contexts, records):
        if (job["score_id"] != record["score_id"] or context["scored_context_sha256"] != record["scored_context_sha256"]
            or len(context["scored_input_token_ids"]) != record["token_count"]):
            raise ValueError("numerical trial context identity differs")
    metadata = old.load_json(directory / "metadata.json")
    original_metadata = old.load_json(prior_directory / "metadata.json")
    for key in ("model", "revision", "versions", "model_files_sha256", "dtype", "loaded_dtype", "attention", "tf32", "seed",
                "generation", "sampling", "chat_template_sha256", "assistant_prefix", "action_token_ids"):
        if metadata.get(key) != original_metadata.get(key):
            raise ValueError("diagnostic runtime/model identity differs")
    if metadata.get("cublas_workspace_config") != ":4096:8" or metadata.get("float32_matmul_precision") != "highest":
        raise ValueError("diagnostic numerical runtime differs")
    validate_source_hashes(metadata, TRIAL_SOURCE_FILES, Path(__file__).resolve().parents[1])
    cache_path = directory.parent / "cache_prepare_receipt.json"
    if old.sha256(cache_path) != old.CACHE_PREPARE_SHA256:
        raise ValueError("numerical trial original cache differs")
    old.validate_provenance(metadata, old.load_json(directory / "dtype_promotion.json"), old.load_json(cache_path)["model_receipts"]["qwen3b"])
    reference = old.load_json(directory / "single_reference.json")["outputs"]
    reference_checks = old.load_json(directory / "reference_validation.json")
    if reference != reference_checks["replay"] or reference != reference_checks["reverse_order"]:
        raise ValueError("numerical single reference lacks exact replay/permutation agreement")
    failed_diagnostics = old.load_json(predecessor_directory / "attempts/000_diagnostics_raw.json")
    if [job["score_id"] for job in jobs[:10]] != failed_diagnostics["score_ids"]:
        raise ValueError("numerical trial changed original failing sample")
    single_gate = compare_numeric(reference[:10], failed_diagnostics["singles"], jobs[:10], config["numerical_checks"])
    previous = {row["score_id"]: row for row in prior_rows}
    old_values = [{"logits": [previous[job["source_score_id"]]["raw_action_logits"][label] for label in ACTIONS]} for job in jobs[10:]]
    if old_values != reference_checks["old_overlap"]:
        raise ValueError("numerical trial reused raw scores differ")
    overlap_gate = compare_numeric(reference[10:], old_values, jobs[10:], config["numerical_checks"])
    candidates = [old.load_json(directory / f"{mode}_{size}.json") for mode in ("uncached", "cached") for size in (2, 4, 8)]
    selected, checks = selected_trial_mode(candidates, jobs, reference, config["numerical_checks"])
    if receipt["fastest_observed_compatible_mode"] != selected or set(receipt["diagnostic_compatible_modes"]) != set(checks):
        raise ValueError("recorded numerical selection differs from reconstructed result")
    return {"passed": True, "receipt_sha256": DIAGNOSTIC_RECEIPT_SHA, "selected_mode": selected,
        "mode_comparisons": checks, "single_predecessor_agreement": single_gate, "prior_overlap_agreement": overlap_gate,
        "production_approved_by_diagnostic": False, "timing_scope": receipt["timing_scope"],
        "all_evidence_sha256": {str(path.relative_to(directory)): old.sha256(path) for path in sorted(directory.rglob("*")) if path.is_file()}}


def validate_recovery_contract(directory, public_hash, recovery_config, predecessor_directory, diagnostic_directory):
    """Require immutable batch-two execution with the unchanged scientific jobs."""
    receipt = old.load_json(directory / "receipt.json")
    metadata = old.load_json(directory / "metadata.json")
    if old.sha256(predecessor_directory / "receipt.json") != PREDECESSOR_RECEIPT_SHA or old.sha256(predecessor_directory / "plan.json") != PREDECESSOR_PLAN_SHA:
        raise ValueError("recovery predecessor differs")
    if old.sha256(diagnostic_directory / "receipt.json") != DIAGNOSTIC_RECEIPT_SHA:
        raise ValueError("recovery diagnostic selection receipt differs")
    expected = {"execution_protocol": EXECUTION_PROTOCOL, "selected_mode": "cached_2", "cached": True, "batch_size": 2,
                "diagnostic_receipt_sha256": DIAGNOSTIC_RECEIPT_SHA, "public_input_sha256": public_hash}
    if any(record.get(key) != value for record in (receipt, metadata) for key, value in expected.items()):
        raise ValueError("recovery execution mode or scientific input differs")
    if receipt.get("predecessor_run_id") != "imcqa-protocol-dev-20261004":
        raise ValueError("recovery predecessor run attribution differs")
    control = old.load_json(directory.parent.parent / "control.json")
    commit = control.get("source_commit", "")
    if commit != RECOVERY_LAUNCH_COMMIT or receipt.get("source_commit") != commit or metadata.get("source_commit") != commit:
        raise ValueError("recovery launch source attribution differs")
    predecessor_plan = old.load_json(predecessor_directory / "plan.json")
    plan = old.load_json(directory / "plan.json")
    if (plan.get("original_plan_sha256") != PREDECESSOR_PLAN_SHA or plan.get("execution_protocol") != EXECUTION_PROTOCOL
        or plan.get("batch_size") != 2 or plan.get("cached") is not True
        or any(plan.get(key) != predecessor_plan[key] for key in ("input_sha256", "ordered_score_ids", "context_sha256", "token_counts"))):
        raise ValueError("recovery changed production ordering or token contexts")
    return {"execution_protocol": EXECUTION_PROTOCOL, "selected_mode": "cached_2", "cached": True, "batch_size": 2,
        "source_commit": commit, "predecessor_receipt_sha256": PREDECESSOR_RECEIPT_SHA,
        "predecessor_plan_sha256": PREDECESSOR_PLAN_SHA, "diagnostic_receipt_sha256": DIAGNOSTIC_RECEIPT_SHA,
        "scientific_inputs_changed": False, "numerical_tolerances_changed": False}


def validate_recovery_config(recovery, original):
    expected = {"schema_version": "imcqa-3b-protocol-recovery-config-v1", "protocol": PROTOCOL,
        "execution_protocol": EXECUTION_PROTOCOL, "model_tag": "qwen3b", "selected_mode": "cached_2", "batch_size": 2,
        "cached": True, "new_contexts": 4032, "reused_contexts": 1200, "total_contexts": 5232,
        "diagnostic_receipt_sha256": DIAGNOSTIC_RECEIPT_SHA, "original_plan_sha256": PREDECESSOR_PLAN_SHA,
        "predecessor_receipt_sha256": PREDECESSOR_RECEIPT_SHA, "scientific_inputs_changed": False}
    if any(recovery.get(key) != value for key, value in expected.items()):
        raise ValueError("recovery config contract differs")
    for key in ("raw_logit_atol", "raw_logit_rtol", "probability_atol", "allow_action_argmax_flip", "allow_candidate_argmax_flip"):
        if recovery["numerical_checks"].get(key) != original["numerical_checks"][key]:
            raise ValueError("recovery numerical tolerance differs from original scientific contract")

def validate_recovery_model(directory: Path, tag: str, package: dict[str, Any], jobs: list[dict[str, Any]],
                       config: dict[str, Any], public_hash: str, prior_directory: Path,
                       prior_rows: list[dict[str, Any]], *, recovery_config: dict[str, Any],
                       predecessor_directory: Path, diagnostic_directory: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Validate new inference, untouched reuse, overlap, and active-only evidence."""
    recovery_contract = validate_recovery_contract(directory, public_hash, recovery_config, predecessor_directory, diagnostic_directory)
    receipt = old.load_json(directory / "receipt.json")
    wanted = {"status": "complete", "protocol": PROTOCOL, "model_tag": tag,
        "public_input_sha256": public_hash, "expected_rows": 4032, "completed_rows": 4032,
        "reused_rows": 1200, "total_contexts": 5232, "automatic_retries": 0, "sampling": False, "generation": False}
    if any(receipt.get(key) != value for key, value in wanted.items()):
        raise ValueError("new completion receipt differs")
    score_path = directory / "scores.jsonl"
    if old.sha256(score_path) != receipt["scores_sha256"]:
        raise ValueError("new scores hash differs from completed receipt")
    raw = score_path.read_bytes()
    if not raw.endswith(b"\n"):
        raise ValueError("new score file has truncated trailing record")
    raw_rows = [json.loads(line) for line in raw.splitlines()]
    metadata = old.load_json(directory / "metadata.json")
    prior_metadata = old.load_json(prior_directory / "metadata.json")
    shared = ("model", "revision", "versions", "model_files_sha256", "dtype", "loaded_dtype", "attention", "tf32", "seed",
              "generation", "sampling", "chat_template_sha256", "assistant_prefix", "action_token_ids")
    if any(metadata.get(key) != prior_metadata.get(key) for key in shared):
        raise ValueError("new model/tokenizer/runtime differs from prior evidence")
    if metadata.get("public_input_sha256") != public_hash or metadata.get("protocol") != PROTOCOL:
        raise ValueError("new metadata public/protocol differs")
    source_files = RECOVERY_SOURCE_FILES
    source_root = Path(__file__).resolve().parents[1]
    source_hashes = {name: old.sha256(source_root / name) for name in source_files}
    if metadata.get("source_files_sha256") != source_hashes:
        raise ValueError("new inference source differs from analysis checkout")
    cache_path = directory.parent / "cache_prepare_receipt.json"
    if old.sha256(cache_path) != old.CACHE_PREPARE_SHA256:
        raise ValueError("original cache preparation receipt differs")
    provenance = old.validate_provenance(metadata, old.load_json(directory / "dtype_promotion.json"),
        old.load_json(cache_path)["model_receipts"][tag])
    expected = {job["score_id"]: job for job in jobs}
    fresh = {key: job for key, job in expected.items() if job["execution"] == "new"}
    old_lookup = {row["score_id"]: row for row in prior_rows}
    new_lookup, views = {}, []
    for row in raw_rows:
        if row["score_id"] not in fresh or row["score_id"] in new_lookup:
            raise ValueError("unexpected or duplicate new score identity")
        job = fresh[row["score_id"]]
        if any(row.get(key) != value for key, value in job.items() if key not in {"prompt", "canonical_gold_option_id", "expected_semantic_action"}):
            raise ValueError("new score public identity differs")
        if row.get("schema_version") != "imcqa-protocol-recovery-scores-v1" or row.get("model_tag") != tag:
            raise ValueError("new score model/schema differs")
        old.validate_vocab_row(row)
        if row["option_token_ids"] != metadata["action_token_ids"]:
            raise ValueError("row action token IDs differ from model metadata")
        new_lookup[row["score_id"]] = row
        views.append(score_view(row, job))
    if set(new_lookup) != set(fresh) or len(raw_rows) != 4032:
        raise ValueError("new score coverage incomplete")
    plan = old.load_json(directory / "plan.json")
    if (plan.get("input_sha256") != public_hash or plan.get("ordered_score_ids") != [row["score_id"] for row in raw_rows]
            or plan.get("context_sha256") != [row["scored_context_sha256"] for row in raw_rows]
            or plan.get("token_counts") != [len(row["scored_input_token_ids"]) for row in raw_rows] or plan.get("batch_size") != 2 or plan.get("cached") is not True):
        raise ValueError("new production ordering/context plan differs")
    reuse = old.load_json(directory / "reuse_manifest.json")
    expected_reuse = {"prior_public_sha256": OLD_PUBLIC_SHA, "prior_scores_sha256": config["reuse"]["prior_scores_sha256"][tag],
        "prior_metadata_sha256": old.sha256(prior_directory / "metadata.json"),
        "prior_receipt_sha256": old.sha256(prior_directory / "receipt.json"), "count": 1200, "mutated_old_rows": False}
    if any(reuse.get(key) != value for key, value in expected_reuse.items()):
        raise ValueError("reuse provenance differs")
    reuse_mapping = []
    for job in jobs:
        if job["execution"] != "reuse":
            continue
        prior_row = old_lookup[job["source_score_id"]]
        for key in ("qid", "group_id", "split", "condition", "menu_id", "prefix_id", "fraction", "round", "reward",
                    "source_job_id", "source_prompt_sha256", "prompt_sha256", "option_source_ids", "allowed_actions"):
            if prior_row.get(key) != job[key]:
                raise ValueError("reused raw identity differs")
        reuse_mapping.append({"score_id": job["score_id"], "source_score_id": prior_row["score_id"],
            "scored_context_sha256": prior_row["scored_context_sha256"]})
        views.append(score_view(prior_row, job))
    if reuse.get("rows") != reuse_mapping:
        raise ValueError("reuse mapping/context evidence differs")
    diagnostic = _only_attempt(directory, "diagnostics")
    ids = diagnostic["score_ids"]
    if len(set(ids)) != len(ids) or not 1 <= len(ids) <= config["execution"]["numerical_contexts_max_per_model"] or not set(ids) <= set(fresh):
        raise ValueError("numeric diagnostic identities differ")
    if diagnostic.get("permutation_indices") != [i ^ 1 for i in range(len(ids))]:
        raise ValueError("recovery within-pair permutation differs")
    original_diag_ids = old.load_json(predecessor_directory / "attempts/000_diagnostics_raw.json")["score_ids"]
    expected_diagnostic_ids = production_pair_ids(plan["ordered_score_ids"], original_diag_ids)
    if (ids != expected_diagnostic_ids or diagnostic.get("batch_size") != 2
        or diagnostic.get("actual_production_pairs") is not True):
        raise ValueError("recovery diagnostic states differ from the actual production pairs")
    diag_jobs = [fresh[key] for key in ids]
    for key in ("arm", "condition", "round", "rotation", "block", "wait_label"):
        if {job[key] for job in diag_jobs} != {job[key] for job in fresh.values()}:
            raise ValueError("numeric diagnostics miss protocol factor: " + key)
    if diagnostic["cached"] != diagnostic["replay"]:
        raise ValueError("exact numerical replay failed")
    numeric_config = config["numerical_checks"]
    checks = {name: compare_numeric(diagnostic["cached"], diagnostic[right], diag_jobs, numeric_config)
        for name, right in (("cached_uncached", "uncached"), ("cached_single", "singles"), ("permutation", "permuted_aligned"))}
    raw_diagnostic = _only_attempt(directory, "diagnostics_raw")
    if any(raw_diagnostic.get(key) != diagnostic[key] for key in ("score_ids", "cached", "uncached", "singles", "replay", "permuted_aligned", "batch_size", "actual_production_pairs", "permutation_indices")):
        raise ValueError("recovery pre-gate numerical evidence differs")
    checks["permutation_single"] = compare_numeric(diagnostic["permuted_aligned"], diagnostic["singles"], diag_jobs, numeric_config)
    production = [{"logits": [new_lookup[key]["raw_action_logits"][label] for label in ACTIONS]} for key in ids]
    checks["production_diagnostic"] = compare_numeric(production, diagnostic["cached"], diag_jobs, numeric_config)
    checks["production_single"] = compare_numeric(production, diagnostic["singles"], diag_jobs, numeric_config)
    pd = _only_attempt(directory, "production_diagnostics")
    validate_production_raw(directory, pd, ids, production, diagnostic["cached"], diagnostic["singles"])
    overlap = _only_attempt(directory, "reuse_diagnostics")
    trial_plan = old.load_json(diagnostic_directory / "plan.json")
    expected_overlap = [item["score_id"] for item in trial_plan["manifest"]["overlap_contexts"]]
    recovery_evidence = old.load_json(directory / "recovery_evidence.json")
    expected_evidence = {"execution_protocol": EXECUTION_PROTOCOL, "predecessor_run_id": "imcqa-protocol-dev-20261004",
        "predecessor_receipt_sha256": PREDECESSOR_RECEIPT_SHA, "original_plan_sha256": PREDECESSOR_PLAN_SHA,
        "original_diagnostics_sha256": old.sha256(predecessor_directory / "attempts/000_diagnostics_raw.json"),
        "diagnostic_receipt_sha256": DIAGNOSTIC_RECEIPT_SHA, "selected_mode": "cached_2", "scientific_inputs_changed": False}
    if any(recovery_evidence.get(key) != value for key, value in expected_evidence.items()):
        raise ValueError("recovery predecessor or numerical evidence attribution differs")
    if overlap["score_ids"] != expected_overlap or recovery_evidence.get("overlap_score_ids") != expected_overlap:
        raise ValueError("recovery overlap differs from the frozen numerical study sample")
    overlap_jobs = [expected[key] for key in overlap["score_ids"]]
    if (not 10 <= len(overlap_jobs) <= config["execution"]["overlap_contexts_max_per_model"]
            or len({job["score_id"] for job in overlap_jobs}) != len(overlap_jobs)
            or any(job["execution"] != "reuse" for job in overlap_jobs)
            or {job["condition"] for job in overlap_jobs} != set(MENUS)
            or {job["round"] for job in overlap_jobs} != set(range(1, 6))
            or {(job["arm"], job["rotation"]) for job in overlap_jobs} != {("forced", 0), ("wait", 0), ("wait", 1)}):
        raise ValueError("prior-overlap diagnostics coverage differs")
    if overlap["source_score_ids"] != [job["source_score_id"] for job in overlap_jobs]:
        raise ValueError("prior-overlap source identities differ")
    old_values = [{"logits": [old_lookup[job["source_score_id"]]["raw_action_logits"][label] for label in ACTIONS]} for job in overlap_jobs]
    if overlap["old"] != old_values or overlap.get("selection_uses_gold_or_outputs") is not False:
        raise ValueError("prior-overlap raw evidence differs")
    if _only_attempt(directory, "reuse_diagnostics_raw") != {key: overlap[key] for key in ("score_ids", "source_score_ids", "old", "fresh")}:
        raise ValueError("prior-overlap pre-gate evidence differs")
    checks["prior_overlap"] = compare_numeric(old_values, overlap["fresh"], overlap_jobs, numeric_config)
    active = _only_attempt(directory, "active_only")
    active_qids = [qid for split in SPLITS for qid in sorted(package["selection"]["selected_qids"][split],
        key=lambda qid: (hashlib.sha256(f"protocol-live|1|{split}|{qid}".encode()).hexdigest(), qid))[:2]]
    if active["qids"] != active_qids or active["episodes"] != 16:
        raise ValueError("active-only question/episode identity differs")
    active_ids = []
    for qid in active_qids:
        for condition in MENUS:
            for block, rotation in (("factorial", 2), ("label_swap", 0)):
                trajectory = sorted((row for row in views if row["qid"] == qid and row["condition"] == condition
                    and row["block"] == block and row["rotation"] == rotation and row["arm"] == "wait"), key=lambda row: row["round"])
                if [row["round"] for row in trajectory] != list(range(1, 6)):
                    raise ValueError("active-only trajectory incomplete")
                for row in trajectory:
                    active_ids.append(row["score_id"])
                    if row["chosen_action"] != row["wait_label"]:
                        break
    if [row["score_id"] for row in active["rows"]] != active_ids or not 16 <= len(active_ids) <= 80:
        raise ValueError("active-only first-commit/PASS sequence differs")
    if receipt.get("attempt") != 0:
        raise ValueError("recovery permits exactly the original single allocation attempt")
    validate_active_raw(directory, active["rows"], new_lookup, fresh)
    active_checks = []
    for row in active["rows"]:
        source_row = new_lookup[row["score_id"]]
        if row["chosen_action"] != source_row["chosen_action"]:
            raise ValueError("active-only action differs")
        active_checks.append(compare_numeric([{"logits": [source_row["raw_action_logits"][label] for label in ACTIONS]}],
            [row["live"]], [fresh[row["score_id"]]], numeric_config))
    audit = {"passed": True, "recovery_contract": recovery_contract, "n_new_rows": len(raw_rows), "n_reused_rows": len(reuse_mapping), "n_total_rows": len(views),
        "scores_sha256": old.sha256(score_path), "metadata_sha256": old.sha256(directory / "metadata.json"),
        "elapsed_seconds": receipt["elapsed_seconds"], "numerical_checks": checks, "provenance": provenance,
        "verified_source_files_sha256": source_hashes, "prior_reuse": expected_reuse,
        "active_only_replay": {"passed": True, "n_questions": 4, "n_episodes": 16, "n_visited_states": len(active_ids),
            "max_action_probability_difference": max(check["max_action_probability_difference"] for check in active_checks),
            "max_candidate_probability_difference": max(check["max_candidate_probability_difference"] for check in active_checks)},
        "all_evidence_sha256": {str(path.relative_to(directory)): old.sha256(path) for path in sorted(directory.rglob("*.json"))}}
    return sorted(views, key=lambda row: row["score_index"]), audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("public", "prior-public", "frozen-source", "gold", "config", "recovery-config", "prior-outputs",
                 "original-outputs", "recovery-outputs", "diagnostic-run", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    source_root = Path(__file__).resolve().parents[1]
    config = old.load_json(args.config)
    recovery_config = old.load_json(args.recovery_config)
    validate_recovery_config(recovery_config, config)
    for path, source_name in ((args.config, "configs/imcqa_protocol_pilot.json"),
                              (args.recovery_config, "configs/imcqa_3b_protocol_recovery.json")):
        if old.sha256(path) != old.sha256(source_root / source_name):
            raise ValueError("analysis config differs from inference-source-bound config")
    if old.sha256(args.prior_public) != OLD_PUBLIC_SHA:
        raise ValueError("prior public hash differs")
    for path, key in ((args.frozen_source, "public/main_jobs.json"), (args.gold, "evaluator/main_dataset.json")):
        if old.sha256(path) != config["frozen_source_sha256"][key]:
            raise ValueError("frozen source/gold hash differs")
    public, prior = old.load_json(args.public), old.load_json(args.prior_public)
    source, dataset = old.load_json(args.frozen_source), old.load_json(args.gold)
    jobs = core.validate_public(public, dataset, source, prior, config)
    prior_config = old.load_json(source_root / "configs/imcqa_wait_pilot.json")
    prior_jobs = old.validate_jobs([old.normalize_public_job(job) for job in prior["jobs"]], dataset,
        n_per_split=prior_config["n_per_split"], n_diagnostic_per_split=prior_config["n_diagnostic_per_split"])
    old.validate_source_binding(prior_jobs, source)
    del source, dataset
    prior_rows, prior_audits = {}, {}
    for model in config["models"]:
        previous = args.prior_outputs / model
        if old.sha256(previous / "scores.jsonl") != config["reuse"]["prior_scores_sha256"][model]:
            raise ValueError("prior score reuse identity differs")
        prior_rows[model], prior_audits[model] = old.validate_model_output(previous, model, prior, prior_jobs, prior_config, OLD_PUBLIC_SHA)
    public_hash = old.sha256(args.public)
    predecessor_directory = args.original_outputs / "qwen3b"
    diagnostic_directory = args.diagnostic_run / "output/qwen3b"
    failure = partial.validate_failure(predecessor_directory, "qwen3b", public_hash, jobs, config, args.prior_outputs / "qwen3b")
    trial = validate_numerical_trial(diagnostic_directory, jobs, config, predecessor_directory, args.prior_outputs / "qwen3b",
        prior_rows["qwen3b"], args.prior_public, args.public)
    all_rows, audits = {}, {}
    all_rows["qwen7b"], audits["qwen7b"] = core.validate_new_model(args.original_outputs / "qwen7b", "qwen7b", public, jobs,
        config, public_hash, args.prior_outputs / "qwen7b", prior_rows["qwen7b"])
    all_rows["qwen3b"], audits["qwen3b"] = validate_recovery_model(args.recovery_outputs / "qwen3b", "qwen3b", public, jobs,
        config, public_hash, args.prior_outputs / "qwen3b", prior_rows["qwen3b"], recovery_config=recovery_config,
        predecessor_directory=predecessor_directory, diagnostic_directory=diagnostic_directory)
    if sum(map(len, all_rows.values())) != 10464 or any(len(rows) != 5232 for rows in all_rows.values()):
        raise ValueError("combined recovered scientific coverage is incomplete")
    report = core.analyze_rows(all_rows, samples=config["analysis"]["bootstrap_samples"], seed=config["analysis"]["bootstrap_seed"])
    args.out.mkdir(parents=True, exist_ok=False)
    for name, values in report.pop("records").items():
        old.write_csv(args.out / (name + ".csv"), values)
    report.update(schema_version="imcqa-protocol-recovered-analysis-v1", status="complete",
        completion_mode="recovered_3b_plus_unchanged_7b", protocol=PROTOCOL, evidence_scope="exploratory_already_inspected_development_questions",
        n_questions=40, n_synthetic_contexts_per_model=32, expected_models=["qwen3b", "qwen7b"], validated_models=["qwen3b", "qwen7b"],
        expected_n_score_rows=10464, n_score_rows=10464, n_new_score_rows=8064, n_reused_score_rows=2400,
        original_run_status="partial", recovered_model="qwen3b", unchanged_model="qwen7b", audits=audits, prior_audits=prior_audits,
        predecessor_failure=failure, numerical_mode_selection=trial, config=config, recovery_config=recovery_config,
        input_sha256={key: old.sha256(getattr(args, key)) for key in ("public", "prior_public", "frozen_source", "gold", "config", "recovery_config")},
        limitations=["This combined dataset contains a separate 3B recovery allocation and unchanged original 7B scores; it does not relabel the failed original allocation as complete.",
            "3B recovery changed cached production batch size from 32 to 2 after a separately validated numerical diagnostic; model files, FP32 arithmetic, prompts, jobs, and tolerances remained fixed.",
            "The fastest observed compatible mode was selected on twenty diagnostic contexts; this is not a general speed ranking or a claim of universal numerical equivalence.",
            "Failed predecessor diagnostics contribute no scientific score observations; all failure evidence remains retained and attributed.",
            "Question-level bootstrap intervals preserve all rotations; they are exploratory, conditional on this protocol, and not adjusted for multiplicity.",
            *config["interpretation_limits"]])
    old.write_json(args.out / "report.json", report)
    lines = ["RECOVERED TWO-MODEL PROTOCOL DEVELOPMENT ANALYSIS", "", "Original 3B allocation: failed before production. Original 7B: complete and unchanged.",
        "Recovered 3B: same scientific jobs and numerical tolerances, cached batch size two, all independent acceptance checks passed.",
        "Combined scientific scores: 10464 = 8064 new + 2400 exact earlier rows; forty real development questions.", ""]
    for summary in report["round_summaries"]:
        if summary["split"] == "selection" and summary["round"] == 5:
            lines.append(f"{summary['model']} / {summary['condition']} / {summary['arm']}: final candidate accuracy={summary['candidate_correct']['mean']:.3%}, averaged over four cyclic rotations.")
    (args.out / "FINDINGS.txt").write_text("\n".join(lines + ["", *report["limitations"]]) + "\n")
    old.write_json(args.out / "analysis_receipt.json", {"status": "complete", "completion_mode": report["completion_mode"],
        "analyzer_sha256": old.sha256(Path(__file__)), "unchanged_original_analyzer_sha256": old.sha256(Path(core.__file__)),
        "n_score_rows": 10464, "n_questions": 40, "public_sha256": public_hash,
        "recovery_source_commit": audits["qwen3b"]["recovery_contract"]["source_commit"],
        "predecessor_receipt_sha256": PREDECESSOR_RECEIPT_SHA, "numerical_diagnostic_receipt_sha256": DIAGNOSTIC_RECEIPT_SHA,
        "output_sha256": {path.name: old.sha256(path) for path in sorted(args.out.iterdir()) if path.is_file()}})
    print(json.dumps({"status": "complete", "completion_mode": report["completion_mode"], "out": str(args.out), "n_score_rows": 10464}))


if __name__ == "__main__":
    main()
