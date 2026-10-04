#!/usr/bin/env python3
"""A bounded, outcome-independent 3B numerical shape study; no production run."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import math
import os
from pathlib import Path
import time

from scripts import acl_option_scoring as base
from scripts import acl_paired_prompt_scoring as paired
from scripts import imcqa_protocol_design as design
from scripts import imcqa_protocol_scoring as scorer
from scripts import imcqa_wait_scoring as wait

PROTOCOL = "imcqa_3b_numerical_shape_diagnostic_v1"
SCHEMA = "imcqa-3b-numerical-manifest-v1"
PROTOCOL_PUBLIC_SHA256 = "3c84125a4891f276565f127436b1e2fa7aa3e30ffbab75c55549e742c145fdbe"
CONTEXT_KEYS = ("rendered_prompt_sha256", "scored_context_sha256", "scored_input_token_ids", "option_token_ids")
SOURCE_FILES = scorer.SOURCE_FILES + ("scripts/imcqa_3b_numerical_diagnostic.py", "configs/imcqa_3b_numerical_diagnostic.json")
EVIDENCE_PATHS = {
    "protocol": ("public/pilot.json", "output/qwen3b/attempts/000_diagnostics_raw.json",
                 "output/qwen3b/plan.json", "output/qwen3b/metadata.json", "output/qwen3b/receipt.json"),
    "prior": ("public/pilot.json", "output/qwen3b/scores.jsonl", "output/qwen3b/plan.json",
              "output/qwen3b/metadata.json", "output/qwen3b/receipt.json"),
}


def locate(root: Path, relative: str) -> Path:
    """Support both collected run artifacts and original volume directory layout."""
    path = root / relative
    if relative == "public/pilot.json" and not path.exists():
        path = root / "pilot.json"
    return path


def build_manifest(protocol_root: Path, prior_root: Path) -> dict:
    """Select the previously frozen diagnostics and overlap rows without answers."""
    roots = {"protocol": protocol_root, "prior": prior_root}
    hashes = {f"{name}:{relative}": base.file_hash(locate(roots[name], relative))
              for name, paths in EVIDENCE_PATHS.items() for relative in paths}
    if (hashes["protocol:public/pilot.json"] != PROTOCOL_PUBLIC_SHA256
            or hashes["prior:public/pilot.json"] != design.PRIOR_PUBLIC_SHA256
            or hashes["prior:output/qwen3b/scores.jsonl"] != design.PRIOR_SCORES_SHA256["qwen3b"]):
        raise ValueError("frozen public or prior score identity differs")
    package = base.load_json(locate(protocol_root, "public/pilot.json").read_bytes())
    old_package = base.load_json(locate(prior_root, "public/pilot.json").read_bytes())
    jobs = design.validate_public_package(package, prior_package=old_package)
    by_id = {job["score_id"]: job for job in jobs}
    old_rows = [base.load_json(line) for line in locate(prior_root, "output/qwen3b/scores.jsonl").read_bytes().splitlines()]
    old_by_id = {row["score_id"]: row for row in old_rows}
    old_public = {job["score_id"]: job for job in old_package["jobs"]}
    old_plan = base.load_json(locate(prior_root, "output/qwen3b/plan.json").read_bytes())
    wait.validate_rows([old_public[key] for key in old_plan["ordered_score_ids"]], old_rows, complete=True)
    reused = [(job, {key: old_by_id[job["source_score_id"]][key] for key in CONTEXT_KEYS}, old_by_id[job["source_score_id"]])
              for job in jobs if job["execution"] == "reuse"]
    overlap = scorer.reuse_sentinels(reused)
    raw = base.load_json(locate(protocol_root, "output/qwen3b/attempts/000_diagnostics_raw.json").read_bytes())
    plan = base.load_json(locate(protocol_root, "output/qwen3b/plan.json").read_bytes())
    identities = dict(zip(plan["ordered_score_ids"], zip(plan["context_sha256"], plan["token_counts"])))
    original_ids = raw["score_ids"]
    overlap_ids = [job["score_id"] for job, _, _ in overlap]
    if len(original_ids) != 10 or len(overlap_ids) != 10 or len(set(original_ids+overlap_ids)) != 20:
        raise ValueError("exactly ten original and ten distinct overlap contexts required")
    if any(by_id[key]["execution"] != "new" for key in original_ids):
        raise ValueError("original diagnostics must be new protocol contexts")
    original = [{"score_id": key, "scored_context_sha256": identities[key][0],
                 "token_count": identities[key][1]} for key in original_ids]
    overlap_records = [{"score_id": job["score_id"], "source_score_id": row["score_id"],
                        "scored_context_sha256": context["scored_context_sha256"],
                        "token_count": len(context["scored_input_token_ids"])} for job, context, row in overlap]
    return {"schema_version": SCHEMA, "protocol": PROTOCOL, "model_tag": "qwen3b",
            "source_evidence_sha256": hashes, "original_contexts": original, "overlap_contexts": overlap_records,
            "selection_rule": "Exact ten failed-run diagnostic IDs, then the unchanged reuse_sentinels rule selecting ten old-overlap contexts; no gold or correctness used",
            "expected_contexts": 20, "expected_logical_evaluations": 200, "maximum_model_forward_calls": 117}


def compare(left, right, jobs) -> dict:
    """Report every unchanged numerical criterion; candidate failures are findings."""
    if not left or len(left) != len(right) or len(left) != len(jobs):
        raise ValueError("numerical comparison cardinality differs")
    rows = []
    for a, b, job in zip(left, right, jobs):
        sa, sb = [scorer.action_statistics(value["logits"], job["allowed_actions"], job["option_source_ids"])
                  for value in (a, b)]
        labels = [label for label in "ABCDE" if label in job["option_source_ids"]]
        differences = [abs(x-y) for x, y in zip(a["logits"], b["logits"])]
        limits = [base.FP32_ATOL+base.FP32_RTOL*abs(y) for y in b["logits"]]
        action_delta = max(abs(sa["action_probabilities"][label]-sb["action_probabilities"][label]) for label in job["allowed_actions"])
        candidate_delta = max(abs(sa["conditional_answer_probabilities"][label]-sb["conditional_answer_probabilities"][label]) for label in labels)
        candidate_flip = max(labels, key=lambda label: sa["raw_action_logits"][label]) != max(labels, key=lambda label: sb["raw_action_logits"][label])
        violations = [label for label, delta, limit in zip("ABCDE", differences, limits) if delta > limit]
        rows.append({"score_id": job["score_id"], "max_logit_difference": max(differences),
                     "max_tolerance_ratio": max(delta/limit for delta, limit in zip(differences, limits)),
                     "logit_violation_labels": violations, "max_action_probability_difference": action_delta,
                     "max_candidate_probability_difference": candidate_delta,
                     "action_argmax_changed": sa["chosen_action"] != sb["chosen_action"],
                     "candidate_argmax_changed": candidate_flip,
                     "passed": not violations and action_delta <= .001 and candidate_delta <= .001
                         and sa["chosen_action"] == sb["chosen_action"] and not candidate_flip})
    return {"passed": all(row["passed"] for row in rows), "rows": rows,
            "atol": base.FP32_ATOL, "rtol": base.FP32_RTOL, "probability_atol": .001,
            "max_logit_difference": max(row["max_logit_difference"] for row in rows),
            "max_action_probability_difference": max(row["max_action_probability_difference"] for row in rows),
            "max_candidate_probability_difference": max(row["max_candidate_probability_difference"] for row in rows),
            "action_argmax_changes": sum(row["action_argmax_changed"] for row in rows),
            "candidate_argmax_changes": sum(row["candidate_argmax_changed"] for row in rows)}


def run_diagnostic(manifest_path: Path, manifest_sha256: str, protocol_root: Path, prior_root: Path,
                   cache_dir: Path, out_dir: Path, *, max_seconds: float = 300, progress=None) -> dict:
    """Run exactly one frozen shape study; never launch production or retry."""
    if max_seconds != 300 or base.file_hash(manifest_path) != manifest_sha256:
        raise ValueError("diagnostic deadline or manifest identity differs")
    manifest = base.load_json(manifest_path.read_bytes())
    if manifest != build_manifest(protocol_root, prior_root):
        raise ValueError("diagnostic manifest differs from exact prior evidence")
    out_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    receipt = {"protocol": PROTOCOL, "model_tag": "qwen3b", "status": "started", "production_rows": 0,
               "manifest_sha256": manifest_sha256, "started_utc": datetime.now(timezone.utc).isoformat(),
               "automatic_retries": 0, "sampling": False, "generation": False}
    forward_calls = logical_evaluations = 0
    def checkpoint(phase):
        if progress:
            progress({"phase": phase, "elapsed_seconds": time.monotonic()-started,
                      "model_forward_calls": forward_calls, "logical_evaluations": logical_evaluations})
    def check_deadline():
        if time.monotonic()-started > max_seconds-20:
            raise TimeoutError("internal diagnostic deadline reached")
    try:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        if os.environ["CUBLAS_WORKSPACE_CONFIG"] not in {":4096:8", ":16:8"}:
            raise ValueError("deterministic CUBLAS configuration differs")
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoModelForCausalLM, AutoTokenizer
        required = {"torch": "2.6.0", "transformers": "4.51.3", "tokenizers": "0.21.1", "safetensors": "0.5.3", "huggingface-hub": "0.30.2"}
        versions = {name: importlib.metadata.version(name) for name in required}
        if any(versions[key].split("+")[0] != value for key, value in required.items()):
            raise ValueError("model stack differs from pinned versions")
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
            raise ValueError("one BF16-capable CUDA GPU required")
        torch.set_num_threads(2); torch.manual_seed(1); torch.cuda.manual_seed_all(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        model_name = base.MODELS["qwen3b"]
        revision = base.PINNED_MODELS[model_name]
        snapshot = Path(snapshot_download(repo_id=model_name, revision=revision, cache_dir=cache_dir,
            local_files_only=True, allow_patterns=["*.json", "*.safetensors", "*.txt"]))
        if snapshot.name != revision:
            raise ValueError("cached revision differs")
        hashes = {}
        for path in sorted(snapshot.rglob("*")):
            if path.is_file() and ".cache" not in path.relative_to(snapshot).parts:
                check_deadline(); hashes[str(path.relative_to(snapshot))] = base.file_hash(path)
        expected_model = base.load_json((cache_dir / "qwen3b_expected_model_hashes.json").read_bytes())
        if expected_model != {"model": model_name, "revision": revision, "model_files_sha256": hashes}:
            raise ValueError("cached model bytes differ")
        tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=False)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.pad_token_id is None:
            raise ValueError("padding token unavailable")
        package = base.load_json(locate(protocol_root, "public/pilot.json").read_bytes())
        by_id = {job["score_id"]: job for job in package["jobs"]}
        selected = manifest["original_contexts"] + manifest["overlap_contexts"]
        jobs = [by_id[item["score_id"]] for item in selected]
        contexts = [scorer.prepare_context(tokenizer, job) for job in jobs]
        if any(context["scored_context_sha256"] != item["scored_context_sha256"]
               or len(context["scored_input_token_ids"]) != item["token_count"] for context, item in zip(contexts, selected)):
            raise ValueError("selected scored token contexts differ from prior evidence")
        metadata = {"protocol": PROTOCOL, "model": model_name, "revision": revision, "versions": versions,
            "model_files_sha256": hashes, "dtype": "float32", "loaded_dtype": "bfloat16", "attention": "eager",
            "tf32": False, "seed": 1, "generation": False, "sampling": False,
            "cuda_runtime_version": torch.version.cuda, "gpu_name": torch.cuda.get_device_name(0),
            "cuda_device_capability": list(torch.cuda.get_device_capability(0)),
            "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "chat_template_sha256": base.sha(tokenizer.chat_template.encode()), "assistant_prefix": scorer.ASSISTANT_PREFIX,
            "action_token_ids": contexts[0]["option_token_ids"],
            "source_files_sha256": {name: base.file_hash(Path(__file__).resolve().parents[1]/name) for name in SOURCE_FILES}}
        for root in (protocol_root, prior_root):
            old = base.load_json(locate(root, "output/qwen3b/metadata.json").read_bytes())
            for key in ("model", "revision", "versions", "model_files_sha256", "dtype", "loaded_dtype", "attention",
                        "tf32", "seed", "generation", "sampling", "chat_template_sha256", "assistant_prefix", "action_token_ids"):
                if old.get(key) != metadata[key]:
                    raise ValueError("prior runtime or model identity differs: " + key)
        base.write_once(out_dir / "metadata.json", metadata)
        order = scorer.paired_order(contexts)
        base.write_once(out_dir / "plan.json", {"manifest": manifest, "jobs": jobs, "contexts": contexts,
            "batch_order_indices": order, "batch_order_rule": "unchanged lexical token-pair ordering from protocol scorer",
            "candidate_modes": [{"cached": cached, "batch_size": size} for cached in (False, True) for size in (2, 4, 8)]})
        check_deadline()
        model = AutoModelForCausalLM.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=False,
            use_safetensors=True, torch_dtype=torch.bfloat16, attn_implementation="eager").to("cuda:0").eval()
        tensor_state = base.snapshot_tensor_state(model)
        model.float(); torch.cuda.synchronize()
        base.write_once(out_dir / "dtype_promotion.json", paired.validate_promotion(model, tensor_state))
        torch.cuda.empty_cache()
        def evaluate(indices, *, cached=False, batch_size=None):
            nonlocal forward_calls, logical_evaluations
            check_deadline()
            calls = 2 if cached else 1
            if forward_calls+calls > 117 or logical_evaluations+len(indices) > 200:
                raise ValueError("frozen diagnostic forward bound exceeded")
            tick = time.monotonic()
            result = scorer.forward(torch, model, tokenizer, [contexts[i] for i in indices], cached=cached, batch_size=batch_size)
            forward_calls += calls; logical_evaluations += len(indices)
            return result, time.monotonic()-tick
        all_indices = list(range(20))
        def single_pass(indices):
            outputs, timings = {}, []
            for index in indices:
                values, seconds = evaluate([index])
                outputs[index] = values[0]; timings.append(seconds)
            return [outputs[i] for i in all_indices], timings
        reference, single_times = single_pass(all_indices)
        base.write_once(out_dir / "single_reference.json", {"outputs": reference, "seconds": single_times})
        replay, replay_times = single_pass(all_indices)
        reverse, reverse_times = single_pass(list(reversed(all_indices)))
        old_rows = {row["score_id"]: row for row in [base.load_json(line) for line in locate(prior_root, "output/qwen3b/scores.jsonl").read_bytes().splitlines()]}
        old_overlap = [{"logits": [old_rows[item["source_score_id"]]["raw_action_logits"][label] for label in "ABCDE"]}
                       for item in manifest["overlap_contexts"]]
        old_diagnostics = base.load_json(locate(protocol_root, "output/qwen3b/attempts/000_diagnostics_raw.json").read_bytes())
        gates = {"single_replay": compare(replay, reference, jobs), "single_reverse_order": compare(reverse, reference, jobs),
                 "prior_overlap": compare(reference[10:], old_overlap, jobs[10:]),
                 "prior_single_reference": compare(reference[:10], old_diagnostics["singles"], jobs[:10])}
        reference_valid = all(gate["passed"] for gate in gates.values()) and reference == replay == reverse
        base.write_once(out_dir / "reference_validation.json", {"gates": gates, "reference_valid": reference_valid,
            "exact_replay": reference == replay, "exact_reverse_order": reference == reverse,
            "replay": replay, "reverse_order": reverse, "replay_seconds": replay_times,
            "reverse_order_seconds": reverse_times, "old_overlap": old_overlap})
        checkpoint("single_references_complete")
        candidates = []
        for cached in (False, True):
            for size in (2, 4, 8):
                name = ("cached" if cached else "uncached") + f"_{size}"
                outputs, timings = {}, []
                for offset in range(0, len(order), size):
                    indices = order[offset:offset+size]
                    values, seconds = evaluate(indices, cached=cached, batch_size=size)
                    outputs.update(zip(indices, values)); timings.append(seconds)
                aligned = [outputs[i] for i in all_indices]
                comparison = compare(aligned, reference, jobs)
                result = {"name": name, "cached": cached, "batch_size": size, "outputs": aligned,
                    "batch_seconds": timings, "elapsed_forward_seconds": math.fsum(timings), "comparison": comparison,
                    "diagnostic_compatible": reference_valid and comparison["passed"],
                    "candidate_batch_replay_tested": False, "production_approved": False}
                base.write_once(out_dir / f"{name}.json", result); candidates.append(result)
                checkpoint(name)
        reproduction = {}
        for cached in (True, False):
            values, seconds = evaluate(list(range(10)), cached=cached, batch_size=32)
            name = "cached" if cached else "uncached"
            reproduction[name] = {"outputs": values, "seconds": seconds,
                "versus_fresh_single": compare(values, reference[:10], jobs[:10]),
                "versus_prior_same_path": compare(values, old_diagnostics[name], jobs[:10])}
        base.write_once(out_dir / "original_batch32_reproduction.json", reproduction)
        if forward_calls != 117 or logical_evaluations != 200:
            raise ValueError("diagnostic execution did not meet the frozen count")
        eligible = [candidate for candidate in candidates if candidate["diagnostic_compatible"]]
        fastest = min(eligible, key=lambda result: (result["elapsed_forward_seconds"], result["name"])) if eligible else None
        receipt.update(status="complete" if reference_valid else "reference_invalid", reference_valid=reference_valid,
            diagnostic_compatible_modes=[candidate["name"] for candidate in eligible],
            fastest_observed_compatible_mode=fastest["name"] if fastest else None,
            production_approved=False, candidate_batch_replay_tested=False,
            timing_scope= "one pass over twenty deliberately selected diagnostic contexts; not a production throughput guarantee",
            max_cuda_memory_allocated_bytes=torch.cuda.max_memory_allocated())
    except TimeoutError as error:
        receipt.update(status="deadline_stop", error=str(error))
    except Exception as error:
        receipt.update(status="failed", error=f"{type(error).__name__}: {error}")
    receipt.update(elapsed_seconds=time.monotonic()-started, model_forward_calls=forward_calls,
                   logical_evaluations=logical_evaluations, finished_utc=datetime.now(timezone.utc).isoformat())
    receipt["output_sha256"] = {path.name: base.file_hash(path) for path in sorted(out_dir.glob("*.json"))}
    base.write_once(out_dir / "receipt.json", receipt)
    checkpoint(receipt["status"])
    return receipt


def main():
    parser = argparse.ArgumentParser(description="Freeze a 20-context manifest without model inference")
    parser.add_argument("--protocol-run-root", type=Path, required=True)
    parser.add_argument("--prior-run-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    base.write_once(args.out, build_manifest(args.protocol_run_root, args.prior_run_root))
    print(json.dumps({"path": str(args.out), "sha256": base.file_hash(args.out)}))


if __name__ == "__main__":
    main()
