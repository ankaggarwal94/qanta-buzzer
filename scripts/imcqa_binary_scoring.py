#!/usr/bin/env python3
"""Pinned FP32 binary stopping scores with explicit X/Y semantic mappings.

This additive forward path preserves the old A-E implementations byte-for-byte.
The fixed proposal is already selected; this model call chooses SUBMIT or DEFER.
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
from scripts import imcqa_binary_design as design

PROTOCOL = design.PROTOCOL
SCHEMA = design.SCHEMA
ASSISTANT_PREFIX = '{"action":"'
CONDITIONS = ("independent_pool", "same_category_pool")
BATCH_SIZE = 32
BENCHMARK_ROWS = 128
MAX_SECONDS = 780
SOURCE_FILES = ("scripts/imcqa_binary_scoring.py", "scripts/imcqa_binary_design.py",
    "scripts/imcqa_protocol_design.py", "scripts/imcqa_wait_scoring.py",
    "scripts/acl_paired_prompt_scoring.py", "scripts/acl_option_scoring.py",
    "scripts/jane_gpu_backend.py", "scripts/jane_qwen_backend.py", "scripts/jane_output_constraints.py",
    "configs/imcqa_binary_pilot.json")


def action_statistics(logits, submit_label, defer_label):
    """Normalize only X/Y; break exact ties toward semantic DEFER."""
    if (len(logits) != 2 or {submit_label, defer_label} != set("XY")
            or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in logits)):
        raise ValueError("two finite X/Y logits and opposite semantic labels required")
    raw = dict(zip("XY", logits))
    weights = [math.exp(value-max(logits)) for value in logits]
    probabilities = dict(zip("XY", (value/math.fsum(weights) for value in weights)))
    ties = [label for label in "XY" if raw[label] == max(logits)]
    chosen = defer_label if len(ties) == 2 else ties[0]
    return {"raw_action_logits": raw, "action_probabilities": probabilities,
        "semantic_probabilities": {"SUBMIT": probabilities[submit_label], "DEFER": probabilities[defer_label]},
        "chosen_action": chosen, "chosen_semantic_action": "SUBMIT" if chosen == submit_label else "DEFER",
        "tied_top_actions": ties, "exact_tie": len(ties) == 2}


def numeric_agreement(left, right, jobs):
    """Apply unchanged FP32 tolerances and forbid semantic decision flips."""
    if not left or len(left) != len(right) or len(left) != len(jobs):
        raise ValueError("numerical comparison cardinality differs")
    max_logit = max_probability = 0.0
    for a, b, job in zip(left, right, jobs):
        sa, sb = [action_statistics(value["logits"], job["submit_label"], job["defer_label"]) for value in (a,b)]
        for x,y in zip(a["logits"],b["logits"]):
            max_logit = max(max_logit, abs(x-y))
            if abs(x-y) > base.FP32_ATOL + base.FP32_RTOL * abs(y):
                raise ValueError("FP32 logits exceed numerical tolerance")
        delta = max(abs(sa["action_probabilities"][label]-sb["action_probabilities"][label]) for label in "XY")
        max_probability = max(max_probability, delta)
        if delta > .001 or sa["chosen_action"] != sb["chosen_action"]:
            raise ValueError("FP32 action probability or argmax changed")
    return {"passed": True, "rows": len(left), "max_logit_difference": max_logit,
        "max_action_probability_difference": max_probability, "argmax_changes": 0,
        "atol": base.FP32_ATOL, "rtol": base.FP32_RTOL, "probability_atol": .001}


def prepare_context(tokenizer: Any, job: dict[str, Any]) -> dict[str, Any]:
    rendered = tokenizer.apply_chat_template([{"role": "user", "content": job["prompt"]}], tokenize=False, add_generation_prompt=True)
    scored = rendered + ASSISTANT_PREFIX
    ids = tokenizer(scored, add_special_tokens=False)["input_ids"]
    original = tokenizer(rendered, add_special_tokens=False)["input_ids"]
    if not original or ids[:len(original)] != original or not len(original) < len(ids) <= 2048:
        raise ValueError("invalid scored context boundary or token limit")
    option_ids = {}
    for label in "XY":
        extended = tokenizer(scored + label, add_special_tokens=False)["input_ids"]
        if (len(extended) != len(ids)+1 or extended[:-1] != ids or tokenizer.decode([extended[-1]], skip_special_tokens=False, clean_up_tokenization_spaces=False) != label):
            raise ValueError("action is not an exact one-token extension")
        option_ids[label] = extended[-1]
    if len(set(option_ids.values())) != 2:
        raise ValueError("action token IDs must differ")
    return {"rendered_prompt_sha256": base.sha(rendered.encode()), "scored_context_sha256": base.sha(scored.encode()),
        "scored_input_token_ids": ids, "option_token_ids": option_ids}


def batch_layout(contexts: list[dict[str, Any]], *, pad_token_id: int,
                 padded_width: int, batch_size: int) -> dict[str, Any]:
    """Build deterministic fixed shapes and discardable duplicate filler rows."""
    if (not contexts or type(batch_size) is not int or not 0 < len(contexts) <= batch_size
            or type(padded_width) is not int or padded_width < 1
            or type(pad_token_id) is not int or pad_token_id < 0):
        raise ValueError("invalid fixed batch layout")
    if any(not context["scored_input_token_ids"] or
           len(context["scored_input_token_ids"]) > padded_width for context in contexts):
        raise ValueError("context exceeds global padded width or is empty")
    real_rows = len(contexts)
    padded_contexts = contexts + [contexts[-1]] * (batch_size - real_rows)
    inputs, masks, positions, options = [], [], [], []
    for context in padded_contexts:
        tokens = context["scored_input_token_ids"]
        if any(type(token) is not int or token < 0 for token in tokens):
            raise ValueError("token IDs must be nonnegative integers")
        if set(context["option_token_ids"]) != set("XY"):
            raise ValueError("exactly two action IDs required")
        padding = padded_width - len(tokens)
        inputs.append([pad_token_id] * padding + tokens)
        masks.append([0] * padding + [1] * len(tokens))
        positions.append([0] * padding + list(range(len(tokens))))
        options.append([context["option_token_ids"][label] for label in "XY"])
    return {"input_ids": inputs, "attention_mask": masks, "position_ids": positions,
            "option_token_ids": options, "real_rows": real_rows,
            "filler_rows": batch_size - real_rows}


def cached_batch_layout(contexts: list[dict[str, Any]], *, pad_token_id: int,
                        prefix_width: int, suffix_width: int, batch_size: int) -> dict[str, Any]:
    """Construct physical cache positions and logical token positions separately."""
    if (not contexts or len(contexts) % 2 or type(batch_size) is not int
            or batch_size % 2 or not len(contexts) <= batch_size
            or type(prefix_width) is not int or prefix_width < 1
            or type(suffix_width) is not int or suffix_width < 1):
        raise ValueError("cache batches require complete pairs and positive fixed dimensions")
    physical_contexts = contexts + contexts[-2:] * ((batch_size - len(contexts)) // 2)
    prefixes, prefix_masks, prefix_positions = [], [], []
    suffixes, full_masks, suffix_positions, option_ids = [], [], [], []
    for index in range(0, len(physical_contexts), 2):
        prefix, tails = paired.split_context_pair(physical_contexts[index], physical_contexts[index+1])
        if len(prefix) > prefix_width or any(len(tail) > suffix_width for tail in tails):
            raise ValueError("context exceeds frozen prefix/suffix widths")
        left_padding = prefix_width - len(prefix)
        mask = [0] * left_padding + [1] * len(prefix)
        prefixes.append([pad_token_id] * left_padding + prefix)
        prefix_masks.append(mask)
        prefix_positions.append([0] * left_padding + list(range(len(prefix))))
        for offset, tail in enumerate(tails):
            suffix_padding = suffix_width - len(tail)
            suffixes.append([pad_token_id] * suffix_padding + tail)
            full_masks.append(mask + [0] * suffix_padding + [1] * len(tail))
            suffix_positions.append([0] * suffix_padding + list(range(len(prefix), len(prefix) + len(tail))))
            option_ids.append([physical_contexts[index + offset]["option_token_ids"][label] for label in "XY"])
    return {"prefix_input_ids": prefixes, "prefix_attention_mask": prefix_masks,
            "prefix_position_ids": prefix_positions, "prefix_cache_position": list(range(prefix_width)),
            "suffix_input_ids": suffixes, "full_attention_mask": full_masks,
            "suffix_position_ids": suffix_positions,
            "suffix_cache_position": list(range(prefix_width, prefix_width + suffix_width)),
            "option_token_ids": option_ids, "real_rows": len(contexts), "physical_rows": batch_size}


def _extract(torch, result, contexts, *, physical_rows):
    if tuple(result.logits.shape[:2]) != (physical_rows, 1):
        raise ValueError("exactly one output position per physical row required")
    logits = result.logits[:, 0, :].float()
    ids = torch.tensor([[c["option_token_ids"][k] for k in "XY"] for c in contexts], device="cuda:0", dtype=torch.long)
    selected = logits[:len(contexts)].gather(1, ids).cpu().tolist()
    normalization = torch.logsumexp(logits[:len(contexts)], dim=-1).cpu().tolist()
    top_values, top_ids = logits[:len(contexts)].max(dim=-1)
    top_values, top_ids = top_values.cpu().tolist(), top_ids.cpu().tolist()
    results = []
    for values, normalizer, top, top_id in zip(selected, normalization, top_values, top_ids):
        action_statistics(values, "X", "Y")
        if not math.isfinite(normalizer) or not math.isfinite(top):
            raise ValueError("nonfinite full-vocabulary summary")
        results.append({"logits": values, "vocabulary_logsumexp": normalizer,
            "unconstrained_top_token_id": top_id, "unconstrained_top_logit": top,
            "legal_action_vocabulary_mass": math.fsum(math.exp(v-normalizer) for v in values)})
    return results


def forward(torch, model, tokenizer, contexts, *, cached=False, batch_size=None):
    physical_rows = batch_size or len(contexts)
    def tensor(value):
        return torch.tensor(value, device="cuda:0", dtype=torch.long)
    with torch.inference_mode():
        if cached:
            plan = paired.cache_plan(contexts)
            layout = cached_batch_layout(contexts, pad_token_id=tokenizer.pad_token_id,
                prefix_width=plan["prefix_width"], suffix_width=plan["suffix_width"], batch_size=physical_rows)
            first = model(input_ids=tensor(layout["prefix_input_ids"]), attention_mask=tensor(layout["prefix_attention_mask"]),
                position_ids=tensor(layout["prefix_position_ids"]), cache_position=tensor(layout["prefix_cache_position"]),
                use_cache=True, logits_to_keep=1, return_dict=True)
            cache = first.past_key_values
            if cache is None or cache.get_seq_length() != plan["prefix_width"]:
                raise ValueError("invalid prefix cache length")
            cache.batch_repeat_interleave(2)
            del first
            result = model(input_ids=tensor(layout["suffix_input_ids"]), attention_mask=tensor(layout["full_attention_mask"]),
                position_ids=tensor(layout["suffix_position_ids"]), cache_position=tensor(layout["suffix_cache_position"]),
                past_key_values=cache, use_cache=True, logits_to_keep=1, return_dict=True)
            if cache.get_seq_length() != plan["prefix_width"] + plan["suffix_width"]:
                raise ValueError("invalid suffix cache length")
        else:
            layout = batch_layout(contexts, pad_token_id=tokenizer.pad_token_id,
                padded_width=max(len(c["scored_input_token_ids"]) for c in contexts), batch_size=physical_rows)
            result = model(**{key: tensor(layout[key]) for key in ("input_ids", "attention_mask", "position_ids")},
                use_cache=False, logits_to_keep=1, return_dict=True)
        extracted = _extract(torch, result, contexts, physical_rows=physical_rows)
    del result
    if cached:
        del cache
    torch.cuda.synchronize()
    return extracted


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
        values = {(key, jobs[j].get(key)) for j in (i, i+1)
                  for key in ("condition", "round", "block", "mapping", "proposal_id")}
        values.update(("synthetic_family", (jobs[j].get("synthetic_case") or "real").split("-")[0])
                      for j in (i, i+1))
        return values
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
        stats = action_statistics([row["raw_action_logits"][label] for label in "XY"],
                                  job["submit_label"], job["defer_label"])
        if any(row.get(key) != value for key, value in stats.items()):
            raise ValueError("checkpoint probability or action differs from logits")

def validate_proposal_source(package, prior_root, metadata):
    """Bind proposals to the prior validated 7B scores and unchanged model/runtime."""
    prior_public_path = prior_root / "public/pilot.json"
    score_root = prior_root / "output/qwen7b"
    scores_path = score_root / "scores.jsonl"
    if (base.file_hash(prior_public_path) != package["source"]["prior_public_sha256"]
            or base.file_hash(scores_path) != package["source"]["prior_qwen7b_scores_sha256"]):
        raise ValueError("prior proposal input hashes differ")
    raw = scores_path.read_bytes()
    if not raw.endswith(b"\n"):
        raise ValueError("prior score file has incomplete trailing row")
    prior_rows = [base.load_json(line) for line in raw.splitlines()]
    design.validate_public_package(package,
        protocol_public=base.load_json(prior_public_path.read_bytes()), protocol_scores=prior_rows)
    prior_metadata = base.load_json((score_root / "metadata.json").read_bytes())
    prior_receipt = base.load_json((score_root / "receipt.json").read_bytes())
    if (prior_receipt["status"] != "complete" or prior_receipt["completed_rows"] != 4032
            or prior_receipt["scores_sha256"] != base.file_hash(scores_path)
            or prior_receipt["public_input_sha256"] != base.file_hash(prior_public_path)):
        raise ValueError("prior proposal scores lack complete hash-bound evidence")
    for key in ("model", "revision", "versions", "model_files_sha256", "dtype", "loaded_dtype",
                "attention", "tf32", "seed", "generation", "sampling", "chat_template_sha256", "assistant_prefix"):
        if prior_metadata.get(key) != metadata.get(key):
            raise ValueError("proposal model or runtime identity differs: " + key)
    return {"prior_public_sha256": base.file_hash(prior_public_path),
        "prior_scores_sha256": base.file_hash(scores_path),
        "prior_metadata_sha256": base.file_hash(score_root / "metadata.json"),
        "prior_receipt_sha256": base.file_hash(score_root / "receipt.json"),
        "validated_real_proposals": 400, "old_score_rows_mutated": False,
        "model_runtime_identity_match": True,
        "source_score_ids": sorted({job["source_plain_score_id"] for job in package["jobs"] if job["block"] == "real"})}


def run_scoring(tag: str, public_path: Path, expected_input_sha256: str, cache_dir: Path,
                out_dir: Path, *, prior_root: Path, max_seconds: float, progress: Callable | None = None) -> dict:
    """Run one create-once allocation with immutable failure evidence and no retry."""
    if tag != "qwen7b" or not isinstance(max_seconds, (int, float)) or isinstance(max_seconds, bool) or not 0 < max_seconds <= MAX_SECONDS:
        raise ValueError("invalid pinned model or bounded deadline")
    if base.file_hash(public_path) != expected_input_sha256:
        raise ValueError("public input hash differs")
    package = base.load_json(public_path.read_bytes())
    jobs = design.validate_public_package(package)
    out_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    attempts = out_dir / "attempts"
    attempts.mkdir(exist_ok=True)
    attempt = len(list(attempts.glob("*.json")))
    receipt = {"protocol": PROTOCOL, "model_tag": tag, "public_input_sha256": expected_input_sha256,
        "status": "started", "expected_rows": 832, "total_contexts": len(jobs), "attempt": attempt,
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
            progress({"phase": phase, "completed_rows": len(rows), "expected_rows": len(expected) if expected else 832, "elapsed_seconds": time.monotonic()-started})
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
        evidence("proposal_source_evidence.json", validate_proposal_source(package, prior_root, metadata))
        if len(jobs) != 832:
            raise ValueError("exactly 832 new production contexts required")
        indices = paired_order(original_contexts)
        expected = [jobs[i] for i in indices]
        contexts = [original_contexts[i] for i in indices]
        receipt["expected_rows"] = len(expected)
        receipt["reused_rows"] = 0
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
        checkpoint("diagnostics_passed")
        benchmark_started, benchmark_start_rows = time.monotonic(), len(rows)
        batch_times = []
        while len(rows) < len(expected):
            check_deadline()
            offset = len(rows); end = min(offset+BATCH_SIZE, len(expected))
            tick = time.monotonic()
            outputs = forward(torch, model, tokenizer, contexts[offset:end], cached=True, batch_size=BATCH_SIZE)
            batch = [{"schema_version": "imcqa-binary-scores-v1", **{k:v for k,v in job.items() if k != "prompt"},
                **context, **action_statistics(output["logits"], job["submit_label"], job["defer_label"]),
                **{k:v for k,v in output.items() if k != "logits"},
                "model_tag": tag}
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
            production = [{"logits": [lookup[job["score_id"]][2]["raw_action_logits"][label] for label in "XY"]}
                          for job in diag_jobs]
            base.write_once(attempts / f"{attempt:03d}_production_diagnostics_raw.json", {
                "score_ids": [job["score_id"] for job in diag_jobs], "production": production, "diagnostic": cached})
            production_gate = numeric_agreement(production, cached, diag_jobs)
            base.write_once(attempts / f"{attempt:03d}_production_diagnostics.json", {
                "score_ids": [job["score_id"] for job in diag_jobs],
                "production": production, "diagnostic": cached, "gate": production_gate})
            receipt["production_numerical_gate"] = production_gate
            active_qids = []
            for split in ("calibration", "selection"):
                split_qids = {job["qid"] for job in jobs if job["block"] == "real" and job["split"] == split}
                active_qids.extend(sorted(split_qids, key=lambda qid: (base.sha(f"binary-live|1|{split}|{qid}".encode()), qid))[:2])
            if len(active_qids) != 4 or len(set(active_qids)) != 4:
                raise ValueError("active replay requires four distinct preselected questions")
            active_evidence = []
            completed_episodes = 0
            for qid in active_qids:
                for condition in CONDITIONS:
                    for mapping in ("submit_x", "submit_y"):
                        for round_number in range(1, 6):
                            check_deadline()
                            matches = [value for value in lookup.values() if value[0]["qid"] == qid
                                       and value[0]["condition"] == condition and value[0]["round"] == round_number
                                       and value[0]["block"] == "real" and value[0]["mapping"] == mapping]
                            if len(matches) != 1:
                                raise ValueError("active replay trajectory is not unique and complete")
                            job, context, row = matches[0]
                            live = forward(torch, model, tokenizer, [context])[0]
                            base.write_once(attempts / f"{attempt:03d}_active_{len(active_evidence):03d}_raw.json", {"score_id": job["score_id"], "live": live})
                            comparison = numeric_agreement([{"logits": [row["raw_action_logits"][label] for label in "XY"]}], [live], [job])
                            action = action_statistics(live["logits"], job["submit_label"], job["defer_label"])["chosen_action"]
                            active_evidence.append({"score_id": job["score_id"], "live": live,
                                                    "chosen_action": action, "agreement": comparison})
                            if action != job["defer_label"]:
                                break
                        completed_episodes += 1
            if completed_episodes != 16 or not 16 <= len(active_evidence) <= 80:
                raise ValueError("active replay coverage differs from the declared 16 episodes")
            base.write_once(attempts / f"{attempt:03d}_active_only.json", {
                "qids": active_qids, "episodes": completed_episodes,
                "rows": active_evidence, "max_rows": 80,
                "variants": ["submit_x", "submit_y"],
                "selection_rule": "two qids per split by SHA256(binary-live|1|split|qid), then qid",
                "history_mode": "canonical cumulative prefix and fixed proposal, no generated conversation history"})
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
