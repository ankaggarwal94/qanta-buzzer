#!/usr/bin/env python3
"""Matched development prompt and action-label diagnostics on pinned FP32 Qwen.

The numerical forward path is inherited from the validated WAIT experiment.
Candidate labels and the WAIT label are explicit because the action-label
control must not assume that the first four logits correspond to answers.
"""
from __future__ import annotations

from datetime import datetime, timezone
import importlib.metadata
import math
import os
from pathlib import Path
import time
from typing import Any, Callable

from scripts import acl_option_scoring as base
from scripts import acl_paired_prompt_scoring as paired
from scripts import imcqa_wait_scoring as wait
from scripts import imcqa_protocol_design as design

PROTOCOL = design.PROTOCOL
SCHEMA = design.SCHEMA
ASSISTANT_PREFIX = wait.ASSISTANT_PREFIX
CONDITIONS = wait.CONDITIONS
PREFIX_IDS = wait.PREFIX_IDS
REWARDS = wait.REWARDS
BATCH_SIZE = 32
BENCHMARK_ROWS = 128
MAX_SECONDS = 2900
SOURCE_FILES = ("scripts/imcqa_protocol_scoring.py", "scripts/imcqa_protocol_design.py",
    "scripts/imcqa_wait_scoring.py", "scripts/acl_paired_prompt_scoring.py",
    "scripts/acl_option_scoring.py", "scripts/jane_gpu_backend.py",
    "scripts/jane_qwen_backend.py", "scripts/jane_output_constraints.py",
    "configs/imcqa_protocol_pilot.json")
prepare_context = wait.prepare_context
forward = wait.forward


def action_statistics(logits: list[float], allowed_actions: str,
                      option_source_ids: dict[str, str], wait_label: str | None = None) -> dict[str, Any]:
    """Normalize legal actions and candidate answers with their explicit labels."""
    if (len(logits) != 5 or any(isinstance(x, bool) or not isinstance(x, (int, float))
                              or not math.isfinite(x) for x in logits)
            or not isinstance(allowed_actions, str) or not allowed_actions
            or allowed_actions != "".join(c for c in "ABCDE" if c in allowed_actions)
            or len(option_source_ids) != 4 or set(option_source_ids.values()) != set("ABCD")
            or not set(option_source_ids) <= set(allowed_actions)):
        raise ValueError("finite A-E logits, ordered legal labels, and four candidate identities required")
    raw = dict(zip("ABCDE", logits))
    def softmax(labels):
        values = [raw[label] for label in labels]
        weights = [math.exp(value-max(values)) for value in values]
        total = math.fsum(weights)
        return dict(zip(labels, (weight/total for weight in weights)))
    candidates = [label for label in "ABCDE" if label in option_source_ids]
    maximum = max(raw[label] for label in allowed_actions)
    ties = [label for label in allowed_actions if raw[label] == maximum]
    answer_probabilities = softmax(candidates)
    return {"raw_action_logits": raw, "action_probabilities": softmax(allowed_actions),
            "chosen_action": ties[0], "tied_top_actions": ties,
            "conditional_answer_probabilities": answer_probabilities,
            "canonical_answer_probabilities": {option_source_ids[label]: answer_probabilities[label]
                                                for label in candidates}}


def numeric_agreement(left, right, jobs):
    """Require FP32 agreement and identical legal action decisions for all labels."""
    if not left or len(left) != len(right) or len(left) != len(jobs):
        raise ValueError("numerical comparison cardinality differs")
    maximum_logit = maximum_probability = maximum_candidate_probability = 0.0
    for a, b, job in zip(left, right, jobs):
        sa, sb = [action_statistics(value["logits"], job["allowed_actions"], job["option_source_ids"])
                  for value in (a, b)]
        for x, y in zip(a["logits"], b["logits"]):
            maximum_logit = max(maximum_logit, abs(x-y))
            if abs(x-y) > base.FP32_ATOL + base.FP32_RTOL * abs(y):
                raise ValueError("FP32 logits exceed numerical tolerance")
        delta = max(abs(sa["action_probabilities"][label]-sb["action_probabilities"][label])
                    for label in job["allowed_actions"])
        maximum_probability = max(maximum_probability, delta)
        if delta > 1e-3 or sa["chosen_action"] != sb["chosen_action"]:
            raise ValueError("FP32 action probability or argmax changed")
        labels = [label for label in "ABCDE" if label in job["option_source_ids"]]
        candidate_delta = max(abs(sa["conditional_answer_probabilities"][label]-sb["conditional_answer_probabilities"][label])
                              for label in labels)
        maximum_candidate_probability = max(maximum_candidate_probability, candidate_delta)
        if candidate_delta > 1e-3:
            raise ValueError("FP32 candidate probability changed")
        candidates = [max(labels, key=lambda label: stats["raw_action_logits"][label]) for stats in (sa, sb)]
        if candidates[0] != candidates[1]:
            raise ValueError("FP32 candidate argmax changed")
    return {"passed": True, "rows": len(left), "max_logit_difference": maximum_logit,
            "max_action_probability_difference": maximum_probability, "argmax_changes": 0,
            "candidate_argmax_changes": 0,
            "max_candidate_probability_difference": maximum_candidate_probability,
            "atol": base.FP32_ATOL, "rtol": base.FP32_RTOL, "probability_atol": 1e-3}


def paired_order(contexts):
    """Pair neighboring lexical token sequences, then score long pairs first."""
    if not contexts or len(contexts) % 2:
        raise ValueError("an even nonzero number of contexts is required")
    lexical = sorted(range(len(contexts)), key=lambda i: (contexts[i]["scored_input_token_ids"], i))
    pairs = [lexical[i:i+2] for i in range(0, len(lexical), 2)]
    for a, b in pairs:
        paired.split_context_pair(contexts[a], contexts[b])
    pairs.sort(key=lambda pair: (-max(len(contexts[i]["scored_input_token_ids"]) for i in pair), pair))
    return [index for pair in pairs for index in pair]


def diagnostic_indices(jobs, contexts):
    """Cover protocol factors and context-length extremes using complete pairs."""
    if not jobs or len(jobs) != len(contexts) or len(jobs) % 2:
        raise ValueError("diagnostics require complete context pairs")
    pairs = range(0, len(jobs), 2)
    def features(i):
        return {(key, jobs[j].get(key)) for j in (i, i+1)
                for key in ("arm", "condition", "round", "rotation", "block", "wait_label")}
    required = set().union(*(features(i) for i in pairs))
    def length(i):
        return max(len(contexts[j]["scored_input_token_ids"]) for j in (i, i+1))
    selected = {min(pairs, key=lambda i: (length(i), i)), max(pairs, key=lambda i: (length(i), -i))}
    covered = set().union(*(features(i) for i in selected))
    while not required <= covered:
        chosen = max((i for i in pairs if i not in selected),
                     key=lambda i: (len(features(i)-covered), -i))
        selected.add(chosen)
        covered.update(features(chosen))
    indices = [i+j for i in sorted(selected) for j in (0, 1)]
    if len(indices) > BATCH_SIZE:
        raise ValueError("diagnostic coverage exceeds one bounded batch")
    return indices


def validate_rows(expected, rows, *, complete):
    """Check immutable public identities and independently recompute statistics."""
    if len(rows) > len(expected) or (complete and len(rows) != len(expected)):
        raise ValueError("score coverage mismatch")
    seen = set()
    for job, row in zip(expected, rows):
        if any(row.get(key) != job[key] for key in design.PUBLIC_JOB_KEYS - {"prompt"}):
            raise ValueError("score identity or order differs")
        if row["score_id"] in seen:
            raise ValueError("duplicate score row")
        seen.add(row["score_id"])
        stats = action_statistics([row["raw_action_logits"][label] for label in "ABCDE"],
                                  job["allowed_actions"], job["option_source_ids"])
        if any(row.get(key) != value for key, value in stats.items()):
            raise ValueError("checkpoint probability or action differs from logits")


def validate_reuse(package, jobs, contexts, tag, prior_root, metadata):
    """Bind untouched prior rows to exact new prompts, token contexts, and model."""
    prior_public_path = prior_root / "public/pilot.json"
    prior_public_hash = base.file_hash(prior_public_path)
    if prior_public_hash != package["reuse"]["prior_public_sha256"]:
        raise ValueError("prior public hash differs")
    prior_package = base.load_json(prior_public_path.read_bytes())
    design.validate_public_package(package, prior_package=prior_package)
    old_public = {job["score_id"]: job for job in wait.validate_public_package(prior_package)}
    directory = prior_root / "output" / tag
    scores_path = directory / "scores.jsonl"
    scores_hash = base.file_hash(scores_path)
    if scores_hash != package["reuse"]["prior_scores_sha256"][tag]:
        raise ValueError("prior score hash differs")
    receipt = base.load_json((directory / "receipt.json").read_bytes())
    if (receipt["status"] != "complete" or receipt["scores_sha256"] != scores_hash
            or receipt["public_input_sha256"] != prior_public_hash or receipt["completed_rows"] != len(old_public)):
        raise ValueError("prior run lacks complete hash-bound evidence")
    prior_metadata = base.load_json((directory / "metadata.json").read_bytes())
    for key in ("model", "revision", "versions", "model_files_sha256", "dtype", "loaded_dtype",
                "attention", "tf32", "seed", "generation", "sampling", "chat_template_sha256",
                "assistant_prefix", "action_token_ids"):
        if prior_metadata.get(key) != metadata.get(key):
            raise ValueError("prior model or runtime identity differs: " + key)
    raw = scores_path.read_bytes()
    if not raw.endswith(b"\n"):
        raise ValueError("prior score file has incomplete trailing row")
    old_rows = [base.load_json(line) for line in raw.splitlines()]
    plan = base.load_json((directory / "plan.json").read_bytes())
    wait.validate_rows([old_public[key] for key in plan["ordered_score_ids"]], old_rows, complete=True)
    old_lookup = {row["score_id"]: row for row in old_rows}
    reused = []
    for job, context in zip(jobs, contexts):
        if job["execution"] != "reuse":
            continue
        old_job = old_public[job["source_score_id"]]
        row = old_lookup[job["source_score_id"]]
        if (job["prompt"] != old_job["prompt"] or job["prompt_sha256"] != old_job["prompt_sha256"]
                or job["allowed_actions"] != old_job["allowed_actions"]
                or job["option_source_ids"] != old_job["option_source_ids"]
                or any(context[key] != row.get(key) for key in context)
                or row["model_tag"] != tag):
            raise ValueError("reused prompt, tokenization, or action identity differs")
        reused.append((job, context, row))
    if len(reused) != 1200:
        raise ValueError("exactly 1200 reused rows required")
    return reused, {"prior_public_sha256": prior_public_hash, "prior_scores_sha256": scores_hash,
        "prior_metadata_sha256": base.file_hash(directory / "metadata.json"),
        "prior_receipt_sha256": base.file_hash(directory / "receipt.json"),
        "count": len(reused), "mutated_old_rows": False,
        "rows": [{"score_id": job["score_id"], "source_score_id": row["score_id"],
                  "scored_context_sha256": context["scored_context_sha256"]}
                 for job, context, row in reused]}


def reuse_sentinels(reused):
    """Choose at least ten rows covering all reused strata and length extremes."""
    if len(reused) < 10:
        raise ValueError("insufficient reused sentinel rows")
    def features(index):
        job = reused[index][0]
        return {("condition", job["condition"]), ("round", job["round"]),
                ("stratum", (job["arm"], job["rotation"]))}
    def length(index):
        return len(reused[index][1]["scored_input_token_ids"])
    required = set().union(*(features(i) for i in range(len(reused))))
    chosen = {min(range(len(reused)), key=lambda i: (length(i), reused[i][0]["score_id"])),
              max(range(len(reused)), key=lambda i: (length(i), reused[i][0]["score_id"]))}
    covered = set().union(*(features(i) for i in chosen))
    while not required <= covered or len(chosen) < 10:
        pick = min((i for i in range(len(reused)) if i not in chosen),
                   key=lambda i: (-len(features(i)-covered), reused[i][0]["score_id"]))
        chosen.add(pick)
        covered.update(features(pick))
    if len(chosen) > 16:
        raise ValueError("reuse sentinel coverage exceeds declared bound")
    return [reused[i] for i in sorted(chosen, key=lambda i: reused[i][0]["score_id"])]


def run_scoring(tag: str, public_path: Path, expected_input_sha256: str, cache_dir: Path,
                out_dir: Path, *, prior_root: Path, max_seconds: float, progress: Callable | None = None) -> dict:
    """Run one bounded allocation; append-only checkpoints can be explicitly resumed.

    Resume never allocates a worker itself. The provider wrapper uses create-once
    allocation claims, so any new paid attempt requires a separate reviewed plan.
    """
    if tag not in base.MODELS or not isinstance(max_seconds, (int, float)) or isinstance(max_seconds, bool) or not 0 < max_seconds <= MAX_SECONDS:
        raise ValueError("invalid pinned model or bounded deadline")
    if base.file_hash(public_path) != expected_input_sha256:
        raise ValueError("public input hash differs")
    package = base.load_json(public_path.read_bytes())
    jobs = design.validate_public_package(package)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    attempts = out_dir / "attempts"
    attempts.mkdir(exist_ok=True)
    attempt = len(list(attempts.glob("*.json")))
    receipt = {"protocol": PROTOCOL, "model_tag": tag, "public_input_sha256": expected_input_sha256,
        "status": "started", "expected_rows": 4032, "total_contexts": len(jobs), "attempt": attempt,
        "started_utc": datetime.now(timezone.utc).isoformat(), "max_seconds": max_seconds,
        "automatic_retries": 0, "sampling": False, "generation": False}
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
            "source_files_sha256": {name: base.file_hash(Path(__file__).resolve().parents[1]/name) for name in SOURCE_FILES}}
        evidence("metadata.json", metadata)
        reused, reuse_manifest = validate_reuse(package, jobs, original_contexts, tag, prior_root, metadata)
        evidence("reuse_manifest.json", reuse_manifest)
        new_indices = [i for i, job in enumerate(jobs) if job["execution"] == "new"]
        if len(new_indices) != 4032:
            raise ValueError("exactly 4032 new production contexts required")
        new_jobs = [jobs[i] for i in new_indices]
        new_contexts = [original_contexts[i] for i in new_indices]
        indices = paired_order(new_contexts)
        expected = [new_jobs[i] for i in indices]
        contexts = [new_contexts[i] for i in indices]
        receipt["expected_rows"] = len(expected)
        receipt["reused_rows"] = len(reused)
        evidence("plan.json", {"input_sha256": expected_input_sha256, "ordered_score_ids": [j["score_id"] for j in expected],
            "batch_size": BATCH_SIZE,
            "batch_order": "lexical token-sequence neighbors paired; descending maximum paired token length then pair indices",
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
        diag_indices = diagnostic_indices(expected, contexts)
        diag_contexts, diag_jobs = [contexts[i] for i in diag_indices], [expected[i] for i in diag_indices]
        check_deadline()
        cached = forward(torch, model, tokenizer, diag_contexts, cached=True, batch_size=BATCH_SIZE)
        native = forward(torch, model, tokenizer, diag_contexts, batch_size=BATCH_SIZE)
        singles, single_seconds = [], []
        for context in diag_contexts:
            check_deadline()
            single_started = time.monotonic()
            singles.extend(forward(torch, model, tokenizer, [context]))
            single_seconds.append(time.monotonic()-single_started)
        replay = forward(torch, model, tokenizer, diag_contexts, cached=True, batch_size=BATCH_SIZE)
        permutation = paired.reverse_pair_permutation(len(diag_contexts))
        permuted = forward(torch, model, tokenizer, [diag_contexts[i] for i in permutation], cached=True, batch_size=BATCH_SIZE)
        aligned = [permuted[permutation.index(i)] for i in range(len(permutation))]
        base.write_once(attempts / f"{attempt:03d}_diagnostics_raw.json", {"score_ids": [j["score_id"] for j in diag_jobs],
            "cached": cached, "uncached": native, "singles": singles, "replay": replay, "permuted_aligned": aligned, "single_forward_seconds": single_seconds})
        checkpoint("numeric_evidence_saved")
        gates = {"cached_uncached": numeric_agreement(cached, native, diag_jobs),
            "cached_single": numeric_agreement(cached, singles, diag_jobs),
            "permutation": numeric_agreement(cached, aligned, diag_jobs)}
        if cached != replay:
            raise ValueError("exact FP32 cache replay failed")
        base.write_once(attempts / f"{attempt:03d}_diagnostics.json", {"score_ids": [j["score_id"] for j in diag_jobs],
            "cached": cached, "uncached": native, "singles": singles, "replay": replay, "permuted_aligned": aligned, "gates": gates})
        sentinel_values = reuse_sentinels(reused)
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
            batch = [{"schema_version": "imcqa-protocol-scores-v1", **{k:v for k,v in job.items() if k != "prompt"},
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
            production_gate = numeric_agreement(production, cached, diag_jobs)
            base.write_once(attempts / f"{attempt:03d}_production_diagnostics.json", {
                "score_ids": [job["score_id"] for job in diag_jobs],
                "production": production, "diagnostic": cached, "gate": production_gate})
            receipt["production_numerical_gate"] = production_gate
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
                            comparison = numeric_agreement([{"logits": [row["raw_action_logits"][label] for label in "ABCDE"]}], [live], [job])
                            action = action_statistics(live["logits"], job["allowed_actions"], job["option_source_ids"])["chosen_action"]
                            active_evidence.append({"score_id": job["score_id"], "live": live,
                                                    "chosen_action": action, "agreement": comparison})
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
