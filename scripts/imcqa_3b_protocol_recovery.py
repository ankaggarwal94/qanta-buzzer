#!/usr/bin/env python3
"""Additive 3B cached-pair recovery of the unchanged development protocol jobs."""
from __future__ import annotations
from datetime import datetime, timezone
import importlib.metadata
import math
import os
from pathlib import Path
import time
from typing import Callable
from scripts import acl_option_scoring as base
from scripts import acl_paired_prompt_scoring as paired
from scripts import imcqa_protocol_design as design
from scripts import imcqa_protocol_scoring as original

PROTOCOL = design.PROTOCOL
EXECUTION_PROTOCOL = "imcqa_3b_protocol_recovery_cached2_v1"
PREDECESSOR_RUN = "imcqa-protocol-dev-20261004"
DIAGNOSTIC_RUN = "imcqa-3b-numerics-20261004"
PUBLIC_SHA256 = "3c84125a4891f276565f127436b1e2fa7aa3e30ffbab75c55549e742c145fdbe"
DIAGNOSTIC_RECEIPT_SHA256 = "33826498509e73e11f5f4f0e213f836b4aa8f2b2d29b0eb134923ed03bb2d705"
ORIGINAL_PLAN_SHA256 = "367755bc172c5abdad7bd6f4633c5d107643bda5b5f439de94889c4792e90e1e"
PREDECESSOR_RECEIPT_SHA256 = "27d0004b1e8a98821d32a1601632637b905c555ef3967664739743837e7fbf45"
ORIGINAL_DIAGNOSTICS_SHA256 = "baf10fd39c9a40cc3ea857065b497baa667e254a9f24ce98b7f9d9f6482b34d7"
BATCH_SIZE = 2
BENCHMARK_ROWS = 128
MAX_SECONDS = 1650
ASSISTANT_PREFIX = original.ASSISTANT_PREFIX
CONDITIONS = original.CONDITIONS
SOURCE_FILES = original.SOURCE_FILES + ("scripts/imcqa_3b_protocol_recovery.py", "configs/imcqa_3b_protocol_recovery.json")
prepare_context = original.prepare_context
forward = original.forward
numeric_agreement = original.numeric_agreement
action_statistics = original.action_statistics
validate_reuse = original.validate_reuse
validate_rows = original.validate_rows
reuse_sentinels = original.reuse_sentinels


def verify_recovery_evidence(protocol_root, numerical_root):
    public_path = protocol_root/"public/pilot.json"
    if not public_path.exists():
        public_path = protocol_root/"pilot.json"
    expected = {public_path: PUBLIC_SHA256,
                protocol_root/"output/qwen3b/plan.json": ORIGINAL_PLAN_SHA256,
                protocol_root/"output/qwen3b/receipt.json": PREDECESSOR_RECEIPT_SHA256,
                protocol_root/"output/qwen3b/attempts/000_diagnostics_raw.json": ORIGINAL_DIAGNOSTICS_SHA256,
                numerical_root/"output/qwen3b/receipt.json": DIAGNOSTIC_RECEIPT_SHA256}
    for path, digest in expected.items():
        if base.file_hash(path) != digest:
            raise ValueError("recovery predecessor/diagnostic evidence hash differs")
    receipt = base.load_json((numerical_root/"output/qwen3b/receipt.json").read_bytes())
    if (receipt["status"] != "complete" or receipt["reference_valid"] is not True
            or receipt["fastest_observed_compatible_mode"] != "cached_2"):
        raise ValueError("numerical study does not support declared recovery candidate")
    for relative, digest in receipt["output_sha256"].items():
        if "/" in relative or base.file_hash(numerical_root/"output/qwen3b"/relative) != digest:
            raise ValueError("numerical study output differs from frozen receipt")
    predecessor = base.load_json((protocol_root/"output/qwen3b/receipt.json").read_bytes())
    if predecessor["status"] != "failed" or predecessor["completed_rows"] != 0:
        raise ValueError("failed predecessor identity differs")
    candidates = [base.load_json((numerical_root/"output/qwen3b"/f"{mode}_{size}.json").read_bytes())
                  for mode in ("uncached", "cached") for size in (2, 4, 8)]
    eligible = [candidate for candidate in candidates if candidate["diagnostic_compatible"] and candidate["comparison"]["passed"]]
    if not eligible or min(eligible, key=lambda candidate: (candidate["elapsed_forward_seconds"], candidate["name"]))["name"] != "cached_2":
        raise ValueError("selected recovery mode does not follow frozen timing rule")
    numerical_plan = base.load_json((numerical_root/"output/qwen3b/plan.json").read_bytes())
    overlap_ids = [item["score_id"] for item in numerical_plan["manifest"]["overlap_contexts"]]
    if len(overlap_ids) != 10 or len(set(overlap_ids)) != 10:
        raise ValueError("ten frozen numerical-study overlap identities required")
    return {"execution_protocol": EXECUTION_PROTOCOL, "predecessor_run_id": PREDECESSOR_RUN,
            "predecessor_receipt_sha256": PREDECESSOR_RECEIPT_SHA256, "original_plan_sha256": ORIGINAL_PLAN_SHA256,
            "original_diagnostics_sha256": ORIGINAL_DIAGNOSTICS_SHA256,
            "diagnostic_run_id": DIAGNOSTIC_RUN, "diagnostic_receipt_sha256": DIAGNOSTIC_RECEIPT_SHA256,
            "overlap_score_ids": overlap_ids,
            "selected_mode": "cached_2", "scientific_inputs_changed": False,
            "selection_scope": "predeclared fastest compatible one-pass diagnostic timing; not a general speed ranking"}


def production_pair_diagnostic_indices(jobs, original_ids):
    if len(original_ids) != 10 or len(set(original_ids)) != 10:
        raise ValueError("ten unchanged original diagnostics required")
    lookup = {job["score_id"]: i for i, job in enumerate(jobs)}
    if not set(original_ids) <= set(lookup) or len(lookup) != len(jobs):
        raise ValueError("diagnostic production identities differ")
    pairs = sorted({lookup[key]//2*2 for key in original_ids})
    result = [i+j for i in pairs for j in (0, 1)]
    if len(result) > 20 or result[-1] >= len(jobs):
        raise ValueError("diagnostic pair companions exceed bound")
    return result


def forward_pairs(torch, model, tokenizer, contexts, *, cached, reverse_within_pairs=False):
    """Evaluate each exact physical batch of two, optionally swapping its rows."""
    if not contexts or len(contexts) % 2:
        raise ValueError("complete production pairs required")
    results = []
    for i in range(0, len(contexts), 2):
        pair = contexts[i:i+2]
        if reverse_within_pairs:
            pair = pair[::-1]
        values = forward(torch, model, tokenizer, pair, cached=cached, batch_size=2)
        results.extend(values[::-1] if reverse_within_pairs else values)
    return results


def record_active_comparison(attempts, attempt, sequence_index, job, live, production):
    """Persist the new live forward before any comparison can reject its values."""
    if type(sequence_index) is not int or not 0 <= sequence_index < 80:
        raise ValueError("active replay raw sequence exceeds the frozen bound")
    raw_name = f"{attempt:03d}_active_{sequence_index:03d}_raw.json"
    raw_path = attempts / raw_name
    base.write_once(raw_path, {"score_id": job["score_id"], "sequence_index": sequence_index,
        **{key: job[key] for key in ("qid", "condition", "block", "rotation", "round")},
        "live": live, "production": production})
    return {"agreement": numeric_agreement([production], [live], [job]),
            "raw_file": raw_name, "raw_file_sha256": base.file_hash(raw_path)}


def run_recovery(tag: str, public_path: Path, expected_input_sha256: str, cache_dir: Path,
                out_dir: Path, *, prior_root: Path, protocol_root: Path, numerical_root: Path, source_commit: str, max_seconds: float, progress: Callable | None = None) -> dict:
    """Run one bounded allocation; append-only checkpoints can be explicitly resumed.

    Resume never allocates a worker itself. The provider wrapper uses create-once
    allocation claims, so any new paid attempt requires a separate reviewed plan.
    """
    if tag != "qwen3b" or max_seconds != 1650 or not isinstance(source_commit, str) or len(source_commit) != 40 or any(c not in "0123456789abcdef" for c in source_commit):
        raise ValueError("invalid pinned model or bounded deadline")
    if expected_input_sha256 != PUBLIC_SHA256 or base.file_hash(public_path) != expected_input_sha256:
        raise ValueError("public input hash differs")
    package = base.load_json(public_path.read_bytes())
    jobs = design.validate_public_package(package)
    out_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    attempts = out_dir / "attempts"
    attempts.mkdir(exist_ok=True)
    attempt = len(list(attempts.glob("*.json")))
    receipt = {"protocol": PROTOCOL, "model_tag": tag, "public_input_sha256": expected_input_sha256,
        "status": "started", "expected_rows": 4032, "total_contexts": len(jobs), "attempt": attempt,
        "started_utc": datetime.now(timezone.utc).isoformat(), "max_seconds": max_seconds,
        "automatic_retries": 0, "sampling": False, "generation": False,
        "execution_protocol": EXECUTION_PROTOCOL, "source_commit": source_commit, "selected_mode": "cached_2",
        "batch_size": 2, "cached": True, "predecessor_run_id": PREDECESSOR_RUN,
        "diagnostic_receipt_sha256": DIAGNOSTIC_RECEIPT_SHA256}
    rows, expected, contexts = [], [], []
    def remaining():
        return max_seconds - (time.monotonic()-started)
    def check_deadline():
        if remaining() <= 30:
            raise TimeoutError("internal worker deadline reached")
    def checkpoint(phase):
        if progress:
            progress({"phase": phase, "completed_rows": len(rows), "expected_rows": len(expected) if expected else 4032, "elapsed_seconds": time.monotonic()-started})
    def evidence(name, value):
        path = out_dir / name
        if path.exists():
            if base.load_json(path.read_bytes()) != value:
                raise ValueError("immutable resume evidence differs: " + name)
        else:
            base.write_once(path, value)
    try:
        recovery_evidence = verify_recovery_evidence(protocol_root, numerical_root)
        evidence("recovery_evidence.json", recovery_evidence)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        if os.environ["CUBLAS_WORKSPACE_CONFIG"] not in {":4096:8", ":16:8"}:
            raise ValueError("deterministic CUBLAS configuration differs")
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoModelForCausalLM, AutoTokenizer
        required = {"torch": "2.6.0", "transformers": "4.51.3", "tokenizers": "0.21.1", "safetensors": "0.5.3", "huggingface-hub": "0.30.2"}
        versions = {name: importlib.metadata.version(name) for name in required}
        if any(versions[k].split("+")[0] != v for k,v in required.items()):
            raise ValueError("model stack differs from pinned versions")
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
            raise ValueError("one BF16-capable CUDA GPU required")
        torch.set_num_threads(2); torch.manual_seed(1); torch.cuda.manual_seed_all(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        name, revision = base.MODELS[tag], base.PINNED_MODELS[base.MODELS[tag]]
        snapshot = Path(snapshot_download(repo_id=name, revision=revision, cache_dir=cache_dir,
            local_files_only=True, allow_patterns=["*.json", "*.safetensors", "*.txt"]))
        if snapshot.name != revision:
            raise ValueError("cached revision differs")
        hashes = {}
        for path in sorted(snapshot.rglob("*")):
            if path.is_file() and ".cache" not in path.relative_to(snapshot).parts:
                check_deadline(); hashes[str(path.relative_to(snapshot))] = base.file_hash(path)
        original = base.load_json((cache_dir / f"{tag}_expected_model_hashes.json").read_bytes())
        if original != {"model": name, "revision": revision, "model_files_sha256": hashes} or not any(x.endswith(".safetensors") for x in hashes):
            raise ValueError("cached model files differ from original receipt")
        tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=False)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.pad_token_id is None:
            raise ValueError("padding token unavailable")
        original_contexts = []
        for index, job in enumerate(jobs):
            if index % 64 == 0: check_deadline()
            original_contexts.append(prepare_context(tokenizer, job))
        metadata = {"protocol": PROTOCOL, "model": name, "revision": revision,
            "versions": versions, "model_files_sha256": hashes, "dtype": "float32", "loaded_dtype": "bfloat16",
            "attention": "eager", "tf32": False, "seed": 1, "generation": False, "sampling": False,
            "chat_template_sha256": base.sha(tokenizer.chat_template.encode()), "assistant_prefix": ASSISTANT_PREFIX,
            "public_input_sha256": expected_input_sha256, "action_token_ids": original_contexts[0]["option_token_ids"],
            "execution_protocol": EXECUTION_PROTOCOL, "source_commit": source_commit, "selected_mode": "cached_2",
            "batch_size": 2, "cached": True, "diagnostic_receipt_sha256": DIAGNOSTIC_RECEIPT_SHA256,
            "source_files_sha256": {name: base.file_hash(Path(__file__).resolve().parents[1]/name) for name in SOURCE_FILES}}
        evidence("metadata.json", metadata)
        reused, reuse_manifest = validate_reuse(package, jobs, original_contexts, tag, prior_root, metadata)
        evidence("reuse_manifest.json", reuse_manifest)
        new_indices = [i for i, job in enumerate(jobs) if job["execution"] == "new"]
        if len(new_indices) != 4032:
            raise ValueError("exactly 4032 new production contexts required")
        new_jobs = [jobs[i] for i in new_indices]
        new_contexts = [original_contexts[i] for i in new_indices]
        original_plan = base.load_json((protocol_root/"output/qwen3b/plan.json").read_bytes())
        new_lookup = {job["score_id"]: (job, context) for job, context in zip(new_jobs, new_contexts)}
        original_ids = original_plan["ordered_score_ids"]
        if len(original_ids) != 4032 or len(set(original_ids)) != 4032 or set(original_ids) != set(new_lookup):
            raise ValueError("original production identities differ")
        expected = [new_lookup[key][0] for key in original_ids]
        contexts = [new_lookup[key][1] for key in original_ids]
        if (original_plan["context_sha256"] != [context["scored_context_sha256"] for context in contexts]
                or original_plan["token_counts"] != [len(context["scored_input_token_ids"]) for context in contexts]):
            raise ValueError("original production token contexts differ")
        receipt["expected_rows"] = len(expected)
        receipt["reused_rows"] = len(reused)
        evidence("plan.json", {"input_sha256": expected_input_sha256, "ordered_score_ids": original_ids,
            "batch_size": BATCH_SIZE, "cached": True, "execution_protocol": EXECUTION_PROTOCOL,
            "original_plan_sha256": ORIGINAL_PLAN_SHA256,
            "batch_order": "unchanged exact ordered_score_ids from failed predecessor plan; cached pairs at batch size 2",
            "context_sha256": [c["scored_context_sha256"] for c in contexts],
            "token_counts": [len(c["scored_input_token_ids"]) for c in contexts]})
        scores_path = out_dir / "scores.jsonl"
        if scores_path.exists():
            raw = scores_path.read_bytes()
            if raw and not raw.endswith(b"\n"):
                raise ValueError("incomplete trailing checkpoint row; manual evidence recovery required")
            rows = [base.load_json(line) for line in raw.splitlines()]
            validate_rows(expected, rows, complete=False)
            for row, context in zip(rows, contexts):
                if any(row.get(key) != value for key,value in context.items()):
                    raise ValueError("resumed token context differs from immutable input")
        else:
            scores_path.touch(exist_ok=False)
        check_deadline()
        model = AutoModelForCausalLM.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=False,
            use_safetensors=True, torch_dtype=torch.bfloat16, attn_implementation="eager").to("cuda:0").eval()
        state = base.snapshot_tensor_state(model)
        model.float(); torch.cuda.synchronize()
        evidence("dtype_promotion.json", paired.validate_promotion(model, state))
        torch.cuda.empty_cache()
        # Cover each public protocol factor and both context-length extremes.
        previous_raw = base.load_json((protocol_root/"output/qwen3b/attempts/000_diagnostics_raw.json").read_bytes())
        diag_indices = production_pair_diagnostic_indices(expected, previous_raw["score_ids"])
        diag_contexts, diag_jobs = [contexts[i] for i in diag_indices], [expected[i] for i in diag_indices]
        check_deadline()
        cached = forward_pairs(torch, model, tokenizer, diag_contexts, cached=True)
        native = forward_pairs(torch, model, tokenizer, diag_contexts, cached=False)
        singles, single_seconds = [], []
        for context in diag_contexts:
            check_deadline()
            single_started = time.monotonic()
            singles.extend(forward(torch, model, tokenizer, [context]))
            single_seconds.append(time.monotonic()-single_started)
        replay = forward_pairs(torch, model, tokenizer, diag_contexts, cached=True)
        aligned = forward_pairs(torch, model, tokenizer, diag_contexts, cached=True, reverse_within_pairs=True)
        base.write_once(attempts / f"{attempt:03d}_diagnostics_raw.json", {"score_ids": [j["score_id"] for j in diag_jobs],
            "cached": cached, "uncached": native, "singles": singles, "replay": replay, "permuted_aligned": aligned, "single_forward_seconds": single_seconds, "batch_size": 2, "actual_production_pairs": True,
            "permutation_rule": "reverse the two rows inside every cached production pair; restore output order",
            "permutation_indices": [i ^ 1 for i in range(len(diag_contexts))]})
        checkpoint("numeric_evidence_saved")
        gates = {"cached_uncached": numeric_agreement(cached, native, diag_jobs),
            "cached_single": numeric_agreement(cached, singles, diag_jobs),
            "permutation": numeric_agreement(cached, aligned, diag_jobs),
            "permutation_single": numeric_agreement(aligned, singles, diag_jobs)}
        if cached != replay:
            raise ValueError("exact FP32 cache replay failed")
        base.write_once(attempts / f"{attempt:03d}_diagnostics.json", {"score_ids": [j["score_id"] for j in diag_jobs],
            "cached": cached, "uncached": native, "singles": singles, "replay": replay, "permuted_aligned": aligned, "gates": gates, "batch_size": 2, "actual_production_pairs": True,
            "permutation_rule": "reverse the two rows inside every cached production pair; restore output order",
            "permutation_indices": [i ^ 1 for i in range(len(diag_contexts))]})
        sentinel_values = reuse_sentinels(reused)
        if [job["score_id"] for job, _, _ in sentinel_values] != recovery_evidence["overlap_score_ids"]:
            raise ValueError("recovery overlap differs from the exact ten frozen numerical-study contexts")
        sentinel_outputs = []
        for _, context, _ in sentinel_values:
            check_deadline()
            sentinel_outputs.extend(forward(torch, model, tokenizer, [context]))
        old_outputs = [{"logits": [row["raw_action_logits"][label] for label in "ABCDE"]}
                       for _, _, row in sentinel_values]
        sentinel_jobs = [job for job, _, _ in sentinel_values]
        # Save raw evidence before comparison so any failure remains inspectable.
        base.write_once(attempts / f"{attempt:03d}_reuse_diagnostics_raw.json", {
            "score_ids": [job["score_id"] for job in sentinel_jobs],
            "source_score_ids": [row["score_id"] for _, _, row in sentinel_values],
            "old": old_outputs, "fresh": sentinel_outputs})
        reuse_gate = numeric_agreement(old_outputs, sentinel_outputs, sentinel_jobs)
        base.write_once(attempts / f"{attempt:03d}_reuse_diagnostics.json", {
            "score_ids": [job["score_id"] for job in sentinel_jobs],
            "source_score_ids": [row["score_id"] for _, _, row in sentinel_values],
            "old": old_outputs, "fresh": sentinel_outputs, "gate": reuse_gate,
            "selection_uses_gold_or_outputs": False,
            "selection_rule": "greedy menu/round/reused-stratum coverage plus token-length extremes; at least 10, at most 16"})
        receipt["reuse_numerical_gate"] = reuse_gate
        checkpoint("diagnostics_passed")
        benchmark_started, benchmark_start_rows = time.monotonic(), len(rows)
        batch_times = []
        while len(rows) < len(expected):
            check_deadline()
            offset = len(rows); end = min(offset+BATCH_SIZE, len(expected))
            tick = time.monotonic()
            outputs = forward(torch, model, tokenizer, contexts[offset:end], cached=True, batch_size=BATCH_SIZE)
            batch = [{"schema_version": "imcqa-protocol-recovery-scores-v1", **{k:v for k,v in job.items() if k != "prompt"},
                **context, **action_statistics(output["logits"], job["allowed_actions"], job["option_source_ids"]),
                **{k:v for k,v in output.items() if k != "logits"},
                "legal_action_vocabulary_mass": math.fsum(math.exp(output["logits"]["ABCDE".index(label)]-output["vocabulary_logsumexp"]) for label in job["allowed_actions"]), "model_tag": tag}
                for job, context, output in zip(expected[offset:end], contexts[offset:end], outputs)]
            with scores_path.open("ab") as stream:
                stream.write(b"".join(base.canonical(row) for row in batch)); stream.flush(); os.fsync(stream.fileno())
            rows.extend(batch); batch_times.append(time.monotonic()-tick)
            if len(rows)-benchmark_start_rows == BENCHMARK_ROWS:
                active_only_reserve = 80 * max(single_seconds) * 1.2 + 30
                projection = paired.cached_budget_projection(len(rows)-benchmark_start_rows, len(expected)-benchmark_start_rows,
                    time.monotonic()-benchmark_started, remaining()-active_only_reserve, batch_times, BATCH_SIZE)
                projection["active_only_validation_reserve_seconds"] = active_only_reserve
                projection["active_only_reserve_rule"] = "80 times maximum measured diagnostic single-forward time times 1.2, plus 30 seconds evidence overhead"
                projection["diagnostic_single_seconds"] = single_seconds
                base.write_once(attempts / f"{attempt:03d}_benchmark.json", projection)
                receipt["benchmark"] = projection
                checkpoint("benchmark")
                if not projection["proceed"]:
                    receipt["status"] = "benchmark_budget_stop"; break
            if len(rows) % 256 == 0: checkpoint("scoring")
        validate_rows(expected, rows, complete=len(rows)==len(expected))
        if len(rows) == len(expected):
            lookup = {job["score_id"]: (job, context, row)
                      for job, context, row in zip(expected, contexts, rows)}
            production = [{"logits": [lookup[job["score_id"]][2]["raw_action_logits"][label] for label in "ABCDE"]}
                          for job in diag_jobs]
            base.write_once(attempts / f"{attempt:03d}_production_diagnostics_raw.json", {
                "score_ids": [job["score_id"] for job in diag_jobs],
                "production": production, "diagnostic": cached, "single_reference": singles})
            production_gate = numeric_agreement(production, cached, diag_jobs)
            production_single_gate = numeric_agreement(production, singles, diag_jobs)
            base.write_once(attempts / f"{attempt:03d}_production_diagnostics.json", {
                "score_ids": [job["score_id"] for job in diag_jobs],
                "production": production, "diagnostic": cached, "gate": production_gate,
                "single_reference": singles, "single_gate": production_single_gate})
            receipt["production_numerical_gate"] = production_gate
            receipt["production_single_gate"] = production_single_gate
            active_qids = []
            for split in ("calibration", "selection"):
                split_qids = {job["qid"] for job in jobs if job["block"] == "factorial" and job["split"] == split}
                active_qids.extend(sorted(split_qids, key=lambda qid: (base.sha(f"protocol-live|1|{split}|{qid}".encode()), qid))[:2])
            if len(active_qids) != 4 or len(set(active_qids)) != 4:
                raise ValueError("active replay requires four distinct preselected questions")
            active_evidence = []
            completed_episodes = 0
            for qid in active_qids:
                for condition in CONDITIONS:
                    for block, rotation in (("factorial", 2), ("label_swap", 0)):
                        for round_number in range(1, 6):
                            check_deadline()
                            matches = [value for value in lookup.values() if value[0]["qid"] == qid
                                       and value[0]["condition"] == condition and value[0]["round"] == round_number
                                       and value[0]["arm"] == "wait" and value[0]["block"] == block
                                       and value[0]["rotation"] == rotation]
                            if len(matches) != 1:
                                raise ValueError("active replay trajectory is not unique and complete")
                            job, context, row = matches[0]
                            live = forward(torch, model, tokenizer, [context])[0]
                            production_output = {"logits": [row["raw_action_logits"][label] for label in "ABCDE"]}
                            comparison = record_active_comparison(attempts, attempt, len(active_evidence),
                                                                  job, live, production_output)
                            action = action_statistics(live["logits"], job["allowed_actions"], job["option_source_ids"])["chosen_action"]
                            active_evidence.append({"score_id": job["score_id"], "live": live,
                                                    "chosen_action": action, **comparison})
                            if action != job["wait_label"]:
                                break
                        completed_episodes += 1
            if completed_episodes != 16 or not 16 <= len(active_evidence) <= 80:
                raise ValueError("active replay coverage differs from the declared 16 episodes")
            base.write_once(attempts / f"{attempt:03d}_active_only.json", {
                "qids": active_qids, "episodes": completed_episodes,
                "rows": active_evidence, "max_rows": 80,
                "variants": [{"block": "factorial", "rotation": 2}, {"block": "label_swap", "rotation": 0}],
                "selection_rule": "two qids per split by SHA256(protocol-live|1|split|qid), then qid",
                "history_mode": "canonical cumulative prompt with prior WAIT count; no generated conversation history"})
            receipt["status"] = "complete"
        receipt["max_cuda_memory_allocated_bytes"] = torch.cuda.max_memory_allocated()
    except TimeoutError as error:
        receipt.update(status="deadline_stop", error=str(error))
    except Exception as error:
        receipt.update(status="failed", error=f"{type(error).__name__}: {error}")
    receipt.update(completed_rows=len(rows), elapsed_seconds=time.monotonic()-started,
        finished_utc=datetime.now(timezone.utc).isoformat(),
        scores_sha256=base.file_hash(out_dir/"scores.jsonl") if (out_dir/"scores.jsonl").exists() else None)
    base.write_once(attempts / f"{attempt:03d}_receipt.json", receipt)
    from scripts.modal_acl_expansion import replace_progress
    replace_progress(out_dir / "receipt.json", receipt)
    checkpoint(receipt["status"])
    return receipt
