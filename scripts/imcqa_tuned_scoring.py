#!/usr/bin/env python3
"""Plain-only, bounded 7B evaluation with inherited FP32 evidence gates."""
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
from scripts import imcqa_protocol_scoring as original
from scripts import imcqa_tuned_design as design

PROTOCOL = design.PROTOCOL
BATCH_SIZE = 8
BENCHMARK_ROWS = 128
def worker_seconds(n_questions):
    """Freeze a conservative 9 seconds/question plus 600 seconds of overhead."""
    if type(n_questions) is not int or not 4 <= n_questions <= 5000:
        raise ValueError("question count must be an integer between 4 and 5000")
    return 9*n_questions + 600
ASSISTANT_PREFIX = original.ASSISTANT_PREFIX
CONDITIONS = original.CONDITIONS
SOURCE_FILES = original.SOURCE_FILES + (
    "scripts/imcqa_tuned_design.py", "scripts/imcqa_tuned_scoring.py")
prepare_context = original.prepare_context
forward = original.forward
numeric_agreement = original.numeric_agreement
action_statistics = original.action_statistics
paired_order = original.paired_order
validate_rows = original.validate_rows


def forward_batches(torch, model, tokenizer, contexts, *, cached, permute=False):
    """Bound each physical forward to eight rows, retaining complete cache pairs."""
    if not contexts or len(contexts) % 2:
        raise ValueError("nonempty complete context pairs required")
    result = []
    for offset in range(0, len(contexts), BATCH_SIZE):
        batch = contexts[offset:offset+BATCH_SIZE]
        permutation = paired.reverse_pair_permutation(len(batch)) if permute else list(range(len(batch)))
        values = forward(torch, model, tokenizer, [batch[i] for i in permutation],
                         cached=cached, batch_size=BATCH_SIZE)
        result.extend(values[permutation.index(i)] for i in range(len(batch)))
    return result


def diagnostic_indices(jobs, contexts):
    """Retain factor coverage and pairs containing both individual length extrema."""
    selected = set(original.diagnostic_indices(jobs, contexts))
    lengths = [len(context["scored_input_token_ids"]) for context in contexts]
    for index in (min(range(len(contexts)), key=lambda i: (lengths[i], i)),
                  max(range(len(contexts)), key=lambda i: (lengths[i], -i))):
        selected.update((index//2*2, index//2*2+1))
    if len(selected) > original.BATCH_SIZE:
        raise ValueError("factor and individual length coverage exceeds 32 diagnostic rows")
    return sorted(selected)


def production_offsets(total):
    """Choose actual first, middle, and last physical batches without outputs."""
    if type(total) is not int or total < BATCH_SIZE*3 or total % BATCH_SIZE:
        raise ValueError("complete production batches required")
    return sorted({0, (total//2)//BATCH_SIZE*BATCH_SIZE, total-BATCH_SIZE})


def replay_qids(jobs):
    """Choose four complete question trajectories without inspecting scores."""
    candidates = {job["qid"] for job in jobs}
    selected = sorted(candidates, key=lambda qid: (base.sha(f"tuned-live|1|{qid}".encode()), qid))[:4]
    if len(selected) != 4:
        raise ValueError("live replay requires four distinct questions")
    return selected


def validate_replay_evidence(evidence):
    qids, rows = evidence.get("qids", []), evidence.get("rows", [])
    if len(qids) != 4 or len(set(qids)) != 4 or len(rows) != 80:
        raise ValueError("live replay must contain 80 states from four questions")
    expected = {(qid, condition, rotation, r) for qid in qids
                for condition in CONDITIONS for rotation in (0, 2) for r in range(1, 6)}
    keys = [(r["qid"], r["condition"], r["rotation"], r["round"]) for r in rows]
    if (set(keys) != expected or len(set(r["score_id"] for r in rows)) != 80
            or any(r["agreement"].get("passed") is not True for r in rows)):
        raise ValueError("live replay states or agreement gates differ")


def record_numeric_gate(path, left, right, jobs, *, extra=None):
    """Persist both raw sides before a rejecting comparison is attempted."""
    base.write_once(path, {"score_ids": [j["score_id"] for j in jobs],
                           "production": left, "reference": right, **(extra or {})})
    return numeric_agreement(left, right, jobs)


def projection_with_validation(processed, total, seconds, remaining, batch_times, single_seconds):
    if not single_seconds or any(not math.isfinite(s) or s <= 0 for s in single_seconds):
        raise ValueError("finite measured single-forward times required")
    # Conservatively reserve all production checks even if first batch was checked.
    active_reserve = 80*max(single_seconds)*1.2
    production_reserve = (3*BATCH_SIZE+3*BATCH_SIZE*2)*max(single_seconds)*1.2
    reserve = active_reserve+production_reserve+45
    result = paired.cached_budget_projection(processed, total, seconds, remaining-reserve, batch_times, BATCH_SIZE)
    return {**result, "live_trajectory_validation_reserve_seconds": active_reserve,
            "production_validation_reserve_seconds": production_reserve,
            "evidence_overhead_reserve_seconds": 45, "total_validation_reserve_seconds": reserve,
            "diagnostic_single_seconds": single_seconds,
            "validation_reserve_rule": "80 full-trajectory singles + 24 production singles + replay/permutation allowance of 48 singles, all max measured single seconds times 1.2, plus 45 seconds"}


def run_scoring(tag: str, public_path: Path, expected_input_sha256: str, cache_dir: Path,
                out_dir: Path, *, source_commit: str, max_seconds: float, progress: Callable | None = None) -> dict:
    """Run one create-once paid allocation, preserving failure evidence."""
    if (tag != "qwen7b" or type(max_seconds) not in (int, float) or not math.isfinite(max_seconds) or max_seconds <= 0
            or len(source_commit) != 40 or any(c not in "0123456789abcdef" for c in source_commit)):
        raise ValueError("invalid pinned model, committed source, or bounded deadline")
    if base.file_hash(public_path) != expected_input_sha256:
        raise ValueError("public input hash differs")
    package = base.load_json(public_path.read_bytes())
    jobs = design.validate_public_package(package)
    expected_rows = len(jobs)
    n_questions = len({job["qid"] for job in jobs})
    if (expected_rows != n_questions*40 or max_seconds != worker_seconds(n_questions)-120
            or any(job["execution"] != "new" or job["arm"] != "plain" for job in jobs)):
        raise ValueError("plain-only question count or frozen runtime cap differs")
    out_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    attempts = out_dir / "attempts"
    attempts.mkdir()
    attempt = 0
    receipt = {"protocol": PROTOCOL, "model_tag": tag, "public_input_sha256": expected_input_sha256,
        "status": "started", "expected_rows": expected_rows, "total_contexts": len(jobs), "attempt": attempt,
        "started_utc": datetime.now(timezone.utc).isoformat(), "max_seconds": max_seconds,
        "automatic_retries": 0, "sampling": False, "generation": False, "reused_rows": 0,
        "source_commit": source_commit, "batch_size": BATCH_SIZE, "cached": True}
    rows, expected, contexts = [], [], []
    def remaining():
        return max_seconds-(time.monotonic()-started)
    def check_deadline():
        if remaining() <= 30:
            raise TimeoutError("internal worker deadline reached")
    def checkpoint(phase):
        if progress:
            progress({"phase": phase, "completed_rows": len(rows), "expected_rows": expected_rows,
                      "elapsed_seconds": time.monotonic()-started})
    def evidence(name, value):
        base.write_once(out_dir/name, value)
    try:
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
        cached_model_receipt = base.load_json((cache_dir / f"{tag}_expected_model_hashes.json").read_bytes())
        if cached_model_receipt != {"model": name, "revision": revision, "model_files_sha256": hashes} or not any(x.endswith(".safetensors") for x in hashes):
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
            "source_commit": source_commit, "batch_size": BATCH_SIZE, "cached": True,
            "source_files_sha256": {name: base.file_hash(Path(__file__).resolve().parents[1]/name) for name in SOURCE_FILES}}
        evidence("metadata.json", metadata)
        indices = paired_order(original_contexts)
        expected = [jobs[i] for i in indices]
        contexts = [original_contexts[i] for i in indices]
        selected_offsets = production_offsets(len(expected))
        evidence("plan.json", {"input_sha256": expected_input_sha256,
            "ordered_score_ids": [j["score_id"] for j in expected], "batch_size": BATCH_SIZE,
            "batch_order": "lexical token-sequence neighbors paired; descending maximum paired token length then pair indices",
            "context_sha256": [c["scored_context_sha256"] for c in contexts],
            "token_counts": [len(c["scored_input_token_ids"]) for c in contexts],
            "production_diagnostic_offsets": selected_offsets,
            "live_replay_qids": replay_qids(jobs), "live_replay_rotations": [0, 2], "new_rows": expected_rows, "reused_rows": 0})
        scores_path = out_dir/"scores.jsonl"
        scores_path.touch(exist_ok=False)
        check_deadline()
        model = AutoModelForCausalLM.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=False,
            use_safetensors=True, torch_dtype=torch.bfloat16, attn_implementation="eager").to("cuda:0").eval()
        state = base.snapshot_tensor_state(model)
        model.float(); torch.cuda.synchronize()
        evidence("dtype_promotion.json", paired.validate_promotion(model, state))
        torch.cuda.empty_cache()
        # Original feature selector has a 32-row coverage bound; physical chunks remain eight.
        diag_indices = diagnostic_indices(expected, contexts)
        diag_contexts, diag_jobs = [contexts[i] for i in diag_indices], [expected[i] for i in diag_indices]
        check_deadline()
        cached = forward_batches(torch, model, tokenizer, diag_contexts, cached=True)
        native = forward_batches(torch, model, tokenizer, diag_contexts, cached=False)
        singles, single_seconds = [], []
        for context in diag_contexts:
            check_deadline()
            tick = time.monotonic()
            singles.extend(forward(torch, model, tokenizer, [context]))
            single_seconds.append(time.monotonic()-tick)
        replay = forward_batches(torch, model, tokenizer, diag_contexts, cached=True)
        aligned = forward_batches(torch, model, tokenizer, diag_contexts, cached=True, permute=True)
        diagnostic_raw = {"score_ids": [j["score_id"] for j in diag_jobs],
            "cached": cached, "uncached": native, "singles": singles, "replay": replay,
            "permuted_aligned": aligned, "single_forward_seconds": single_seconds,
            "physical_batch_size": BATCH_SIZE, "feature_selection_bound": original.BATCH_SIZE}
        base.write_once(attempts/"000_diagnostics_raw.json", diagnostic_raw)
        checkpoint("numeric_evidence_saved")
        gates = {"cached_uncached": numeric_agreement(cached, native, diag_jobs),
                 "cached_single": numeric_agreement(cached, singles, diag_jobs),
                 "permutation": numeric_agreement(cached, aligned, diag_jobs),
                 "permutation_single": numeric_agreement(aligned, singles, diag_jobs)}
        if cached != replay:
            raise ValueError("exact FP32 cache replay failed")
        base.write_once(attempts/"000_diagnostics.json", {**diagnostic_raw, "gates": gates})
        checkpoint("diagnostics_passed")
        benchmark_started = time.monotonic()
        batch_times, production_records = [], []
        while len(rows) < len(expected):
            check_deadline()
            offset = len(rows)
            batch_jobs, batch_contexts = expected[offset:offset+BATCH_SIZE], contexts[offset:offset+BATCH_SIZE]
            tick = time.monotonic()
            outputs = forward(torch, model, tokenizer, batch_contexts, cached=True, batch_size=BATCH_SIZE)
            batch_seconds = time.monotonic()-tick
            batch = [{"schema_version": "imcqa-tuned-scores-v1", **{k:v for k,v in job.items() if k != "prompt"},
                **context, **action_statistics(output["logits"], job["allowed_actions"], job["option_source_ids"]),
                **{k:v for k,v in output.items() if k != "logits"},
                "legal_action_vocabulary_mass": math.fsum(math.exp(output["logits"]["ABCDE".index(label)]-output["vocabulary_logsumexp"]) for label in job["allowed_actions"]), "model_tag": tag}
                for job, context, output in zip(batch_jobs, batch_contexts, outputs)]
            with scores_path.open("ab") as stream:
                stream.write(b"".join(base.canonical(row) for row in batch)); stream.flush(); os.fsync(stream.fileno())
            rows.extend(batch); batch_times.append(batch_seconds)
            if offset in selected_offsets:
                live_singles = []
                for context in batch_contexts:
                    check_deadline()
                    live_singles.extend(forward(torch, model, tokenizer, [context]))
                replay_outputs = forward(torch, model, tokenizer, batch_contexts, cached=True, batch_size=BATCH_SIZE)
                raw = {"offset": offset, "score_ids": [j["score_id"] for j in batch_jobs],
                       "production": outputs, "single_reference": live_singles, "diagnostic": replay_outputs,
                       "actual_production_batch": True, "batch_size": BATCH_SIZE}
                base.write_once(attempts/f"000_production_batch_{offset:05d}_raw.json", raw)
                checkpoint("production_evidence_saved")
                gate = numeric_agreement(outputs, replay_outputs, batch_jobs)
                single_gate = numeric_agreement(outputs, live_singles, batch_jobs)
                if outputs != replay_outputs:
                    raise ValueError("exact production-batch replay failed")
                production_records.append({**raw, "gate": gate, "single_gate": single_gate})
            if len(rows) == BENCHMARK_ROWS:
                projection = projection_with_validation(len(rows), len(expected), time.monotonic()-benchmark_started,
                    remaining(), batch_times, single_seconds)
                base.write_once(attempts/"000_benchmark.json", projection)
                receipt["benchmark"] = projection
                checkpoint("benchmark")
                if not projection["proceed"]:
                    receipt["status"] = "benchmark_budget_stop"
                    break
            if len(rows) % 256 == 0:
                checkpoint("scoring")
        validate_rows(expected, rows, complete=len(rows)==len(expected))
        if len(rows) == len(expected):
            if [record["offset"] for record in production_records] != selected_offsets:
                raise ValueError("production diagnostic coverage differs")
            production_jobs = [expected[i] for offset in selected_offsets for i in range(offset, offset+BATCH_SIZE)]
            production = [output for record in production_records for output in record["production"]]
            production_singles = [output for record in production_records for output in record["single_reference"]]
            production_replay = [output for record in production_records for output in record["diagnostic"]]
            production_raw = {"score_ids": [j["score_id"] for j in production_jobs], "production": production,
                              "single_reference": production_singles, "diagnostic": production_replay,
                              "offsets": selected_offsets, "actual_production_batches": True, "batch_size": BATCH_SIZE}
            base.write_once(attempts/"000_production_diagnostics_raw.json", production_raw)
            production_gate = numeric_agreement(production, production_replay, production_jobs)
            production_single_gate = numeric_agreement(production, production_singles, production_jobs)
            base.write_once(attempts/"000_production_diagnostics.json", {**production_raw,
                "gate": production_gate, "single_gate": production_single_gate})
            receipt["production_numerical_gate"] = production_gate
            receipt["production_single_gate"] = production_single_gate
            lookup = {(job["qid"], job["condition"], job["rotation"], job["round"]): (job, context, row)
                      for job, context, row in zip(expected, contexts, rows)}
            qids = replay_qids(jobs)
            replay_rows = []
            for qid in qids:
                for condition in CONDITIONS:
                    for rotation in (0, 2):
                        for round_number in range(1, 6):
                            check_deadline()
                            job, context, row = lookup[(qid, condition, rotation, round_number)]
                            live = forward(torch, model, tokenizer, [context])[0]
                            output = {"logits": [row["raw_action_logits"][label] for label in "ABCDE"]}
                            comparison = record_numeric_gate(attempts/f"000_live_{len(replay_rows):03d}_raw.json",
                                [output], [live], [job], extra={"qid": qid, "condition": condition,
                                    "rotation": rotation, "round": round_number, "live": live})
                            replay_rows.append({"score_id": job["score_id"], "qid": qid, "condition": condition,
                                "rotation": rotation, "round": round_number, "agreement": comparison})
            live_evidence = {"qids": qids, "rows": replay_rows, "states": 80,
                "selection_rule": "four qids by SHA256(tuned-live|1|qid), then qid",
                "history_mode": "fresh canonical plain prompt at each round; no generated history"}
            validate_replay_evidence(live_evidence)
            base.write_once(attempts/"000_live_trajectories.json", live_evidence)
            receipt["status"] = "complete"
        receipt["max_cuda_memory_allocated_bytes"] = torch.cuda.max_memory_allocated()
    except TimeoutError as error:
        receipt.update(status="deadline_stop", error=str(error))
    except Exception as error:
        receipt.update(status="failed", error=f"{type(error).__name__}: {error}")
    receipt.update(completed_rows=len(rows), elapsed_seconds=time.monotonic()-started,
        finished_utc=datetime.now(timezone.utc).isoformat(),
        scores_sha256=base.file_hash(out_dir/"scores.jsonl") if (out_dir/"scores.jsonl").exists() else None)
    base.write_once(attempts/"000_receipt.json", receipt)
    base.write_once(out_dir/"receipt.json", receipt)
    checkpoint(receipt["status"])
    return receipt
