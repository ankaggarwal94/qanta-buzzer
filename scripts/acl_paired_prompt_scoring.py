#!/usr/bin/env python3
"""Paired menu-only prompt scoring in FP32 using the original pinned weights.

Original and minimally forced prompts are scored side by side. This diagnostic
uses no gold labels, generation, sampling, quantization, or precision fallback.
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
from scripts.jane_gpu_backend import build_choice_control_prompt

MODELS = base.MODELS
PINNED_MODELS = base.PINNED_MODELS
PUBLIC_SHA256 = base.PUBLIC_SHA256
ASSISTANT_PREFIX = base.ASSISTANT_PREFIX
PROMPT_CONDITIONS = ("original", "forced")
PROTOCOL = "paired_menu_prompt_fp32_option_softmax_v1"
VALIDATION_METHOD = "fp32_fixed_shape_single_replay_permutation_v1"
CACHE_VALIDATION_METHOD = "fp32_shared_prefix_cache_equivalence_v1"
CACHE_BATCH_SIZE = 128
BATCH_SIZE = 32
BENCHMARK_ROWS = 256
EXPECTED_JOBS = 10000
EXPECTED_ROWS = 20000
MAX_SECONDS = 1400
ORIGINAL_TAIL = ('If you decline to answer, use null for answer and confidence, and '
                 '"abstain" for status. Do not add markdown or explanation.')
FORCED_TAIL = 'You must choose one of A, B, C, or D. Do not add markdown or explanation.'
SOURCE_FILES = ("scripts/acl_paired_prompt_scoring.py", "scripts/acl_option_scoring.py",
                "scripts/jane_gpu_backend.py", "scripts/jane_qwen_backend.py",
                "scripts/jane_output_constraints.py")
NUMERIC_FILES = ("diagnostics_bf16.json", "diagnostics_fp32.json",
                 "diagnostics_validation.json", "dtype_promotion.json")


def forced_prompt(job: dict[str, Any]) -> str:
    """Transform only the canonical terminal abstention clause; fail closed."""
    original = build_choice_control_prompt(job["options"])
    if job["prompt"] != original or base.sha(original.encode()) != job["prompt_sha256"]:
        raise ValueError("prompt does not match frozen canonical menu template and hash")
    if original.count(ORIGINAL_TAIL) != 1 or not original.endswith(ORIGINAL_TAIL):
        raise ValueError("unexpected abstention clause; exact single terminal clause required")
    return original[:-len(ORIGINAL_TAIL)] + FORCED_TAIL


def prompt_manifest() -> dict[str, Any]:
    """Return exact original/forced templates and hashes, with one menu slot."""
    options = [{"id": label, "text": "template_option_" + label} for label in "ABCD"]
    original = build_choice_control_prompt(options)
    menu = "\n".join(f'{x["id"]}. {x["text"]}' for x in options)
    job = {"options": options, "prompt": original, "prompt_sha256": base.sha(original.encode())}
    templates = {"original": original.replace(menu, "{menu}"),
                 "forced": forced_prompt(job).replace(menu, "{menu}")}
    if any(template.count("{menu}") != 1 for template in templates.values()):
        raise ValueError("template must contain exactly one menu placeholder")
    return {"schema_version": "acl-paired-prompt-manifest-v1", "protocol": PROTOCOL,
            "templates": templates,
            "template_sha256": {key: base.sha(value.encode()) for key, value in templates.items()},
            "replaced_terminal_clause": ORIGINAL_TAIL, "replacement_terminal_clause": FORCED_TAIL,
            "assistant_prefix": ASSISTANT_PREFIX,
            "transform": "replace exact single terminal abstention clause; preserve every preceding byte"}


def derive_job_pair(job: dict[str, Any], job_index: int) -> list[dict[str, Any]]:
    """Derive original then forced records without modifying the frozen job."""
    if type(job_index) is not int or job_index < 0:
        raise ValueError("job_index must be a nonnegative integer")
    forced = forced_prompt(job)
    return [{**job, "job_index": job_index, "score_index": 2 * job_index + index,
             "score_id": job["job_id"] + ":" + condition, "prompt_condition": condition,
             "original_prompt_sha256": job["prompt_sha256"], "prompt": prompt,
             "prompt_sha256": base.sha(prompt.encode())}
            for index, (condition, prompt) in enumerate(zip(PROMPT_CONDITIONS, (job["prompt"], forced)))]


def derived_jobs(jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [record for index, job in enumerate(jobs) for record in derive_job_pair(job, index)]


def select_diagnostic_menus(jobs: list[dict[str, Any]],
                            contexts: list[dict[str, Any]]) -> dict[str, Any]:
    """Choose short and long public menus in each menu condition, without gold."""
    if len(contexts) != 2 * len(jobs):
        raise ValueError("diagnostic contexts must cover both prompts per menu")
    conditions = sorted({job["condition"] for job in jobs})
    if conditions != ["independent_pool", "same_category_pool"]:
        raise ValueError("unexpected menu conditions")
    selected = set()
    strata = {}
    for condition in conditions:
        indices = [index for index, job in enumerate(jobs) if job["condition"] == condition]
        if len(indices) < 4:
            raise ValueError("at least four menus per condition required for diagnostics")
        ordered = sorted(indices, key=lambda index: (
            max(len(contexts[2 * index + offset]["scored_input_token_ids"]) for offset in (0, 1)), index))
        picks = ordered[:2] + ordered[-2:]
        selected.update(picks)
        strata[condition] = {"shortest_menu_indices": ordered[:2], "longest_menu_indices": ordered[-2:]}
    menu_indices = sorted(selected)
    score_indices = [2 * index + offset for index in menu_indices for offset in (0, 1)]
    return {"menu_indices": menu_indices, "score_indices": score_indices, "strata": strata,
            "selection_rule": "two shortest and two longest max-paired-context lengths per menu condition; index breaks length ties",
            "selection_uses_gold": False}


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
        if set(context["option_token_ids"]) != set("ABCD"):
            raise ValueError("exactly four option IDs required")
        padding = padded_width - len(tokens)
        inputs.append([pad_token_id] * padding + tokens)
        masks.append([0] * padding + [1] * len(tokens))
        positions.append([0] * padding + list(range(len(tokens))))
        options.append([context["option_token_ids"][label] for label in "ABCD"])
    return {"input_ids": inputs, "attention_mask": masks, "position_ids": positions,
            "option_token_ids": options, "real_rows": real_rows,
            "filler_rows": batch_size - real_rows}


def forward(torch: Any, model: Any, tokenizer: Any, contexts: list[dict[str, Any]],
            *, padded_width: int | None = None, batch_size: int | None = None) -> list[list[float]]:
    """Read only the final-position A-D logits; no generation or full-logit save."""
    layout = batch_layout(contexts, pad_token_id=tokenizer.pad_token_id,
        padded_width=padded_width if padded_width is not None else max(len(c["scored_input_token_ids"]) for c in contexts),
        batch_size=batch_size if batch_size is not None else len(contexts))
    tensors = {key: torch.tensor(layout[key], device="cuda:0", dtype=torch.long)
               for key in ("input_ids", "attention_mask", "position_ids")}
    with torch.inference_mode():
        result = model(**tensors, use_cache=False, logits_to_keep=1, return_dict=True)
        physical_rows = len(layout["input_ids"])
        if tuple(result.logits.shape[:2]) != (physical_rows, 1):
            raise ValueError("forward output must have exactly one logit position per physical row")
        option_ids = torch.tensor(layout["option_token_ids"], device="cuda:0", dtype=torch.long)
        selected = result.logits[:, 0, :].gather(1, option_ids).float().cpu().tolist()
    del result
    torch.cuda.synchronize()
    if len(selected) != physical_rows:
        raise ValueError("forward output cardinality mismatch")
    for values in selected:
        base.option_statistics(values)
    return selected[:layout["real_rows"]]



def split_context_pair(left: dict[str, Any], right: dict[str, Any]) -> tuple[list[int], list[list[int]]]:
    """Split an original/forced pair at its exact token-level common prefix."""
    a, b = left["scored_input_token_ids"], right["scored_input_token_ids"]
    common = 0
    for x, y in zip(a, b):
        if x != y:
            break
        common += 1
    if common == 0 or common == min(len(a), len(b)):
        raise ValueError("paired contexts require a nonempty shared prefix and two nonempty different suffixes")
    if left["option_token_ids"] != right["option_token_ids"]:
        raise ValueError("paired option token IDs differ")
    return a[:common], [a[common:], b[common:]]


def cache_plan(contexts: list[dict[str, Any]]) -> dict[str, Any]:
    """Freeze global prefix/suffix shapes using only exact public token IDs."""
    if not contexts or len(contexts) % 2:
        raise ValueError("cache plan requires complete ordered prompt pairs")
    pairs = [split_context_pair(contexts[i], contexts[i+1]) for i in range(0, len(contexts), 2)]
    prefix_width = max(len(prefix) for prefix, _ in pairs)
    suffix_width = max(len(suffix) for _, suffixes in pairs for suffix in suffixes)
    full_width = max(len(context["scored_input_token_ids"]) for context in contexts)
    return {"schema_version": "acl-shared-prefix-cache-plan-v1", "n_pairs": len(pairs),
            "prefix_width": prefix_width, "suffix_width": suffix_width, "full_width": full_width,
            "total_full_input_tokens": sum(len(c["scored_input_token_ids"]) for c in contexts),
            "total_shared_prefix_tokens": sum(len(prefix) for prefix, _ in pairs),
            "total_suffix_tokens": sum(len(tail) for _, tails in pairs for tail in tails),
            "padded_cached_token_positions_per_pair": prefix_width + 2 * suffix_width,
            "padded_uncached_token_positions_per_pair": 2 * full_width,
            "padded_token_position_reduction_fraction": 1 - (prefix_width + 2 * suffix_width) / (2 * full_width)}


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
        prefix, tails = split_context_pair(physical_contexts[index], physical_contexts[index+1])
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
            option_ids.append([physical_contexts[index + offset]["option_token_ids"][label] for label in "ABCD"])
    return {"prefix_input_ids": prefixes, "prefix_attention_mask": prefix_masks,
            "prefix_position_ids": prefix_positions, "prefix_cache_position": list(range(prefix_width)),
            "suffix_input_ids": suffixes, "full_attention_mask": full_masks,
            "suffix_position_ids": suffix_positions,
            "suffix_cache_position": list(range(prefix_width, prefix_width + suffix_width)),
            "option_token_ids": option_ids, "real_rows": len(contexts), "physical_rows": batch_size}


def cached_forward(torch: Any, model: Any, tokenizer: Any, contexts: list[dict[str, Any]],
                   *, prefix_width: int, suffix_width: int, batch_size: int) -> list[list[float]]:
    """Prefill each exact shared prefix once, then score its two suffix branches.

    Transformers 4.51.3 DynamicCache.batch_repeat_interleave mutates every KV
    layer along batch dimension: [p0,p1] becomes [p0,p0,p1,p1]. A fresh cache is
    created per call; no question or diagnostic can inherit another's state.
    """
    layout = cached_batch_layout(contexts, pad_token_id=tokenizer.pad_token_id,
        prefix_width=prefix_width, suffix_width=suffix_width, batch_size=batch_size)
    def tensor(key: str):
        return torch.tensor(layout[key], device="cuda:0", dtype=torch.long)
    with torch.inference_mode():
        prefill = model(input_ids=tensor("prefix_input_ids"), attention_mask=tensor("prefix_attention_mask"),
            position_ids=tensor("prefix_position_ids"), cache_position=tensor("prefix_cache_position"),
            use_cache=True, logits_to_keep=1, return_dict=True)
        cache = prefill.past_key_values
        if cache is None or cache.get_seq_length() != prefix_width:
            raise ValueError("prefix prefill returned an invalid cache length")
        # Official pinned API mutates in place and returns None.
        cache.batch_repeat_interleave(2)
        del prefill
        result = model(input_ids=tensor("suffix_input_ids"), attention_mask=tensor("full_attention_mask"),
            position_ids=tensor("suffix_position_ids"), cache_position=tensor("suffix_cache_position"),
            past_key_values=cache, use_cache=True, logits_to_keep=1, return_dict=True)
        if tuple(result.logits.shape[:2]) != (batch_size, 1):
            raise ValueError("cached output must have exactly one logit position per physical row")
        if cache.get_seq_length() != prefix_width + suffix_width:
            raise ValueError("suffix cache length differs from physical positions")
        selected = result.logits[:, 0, :].gather(1, tensor("option_token_ids")).float().cpu().tolist()
    del result, cache
    torch.cuda.synchronize()
    for values in selected:
        base.option_statistics(values)
    return selected[:layout["real_rows"]]


def reverse_pair_permutation(count: int) -> list[int]:
    if type(count) is not int or count < 2 or count % 2:
        raise ValueError("pair permutation requires a positive even count")
    return [index + offset for index in reversed(range(0, count, 2)) for offset in (0, 1)]


def validate_cached_diagnostics(cached, native, replay, permuted, permutation, single) -> dict[str, Any]:
    if len(permuted) != len(cached) or permutation != reverse_pair_permutation(len(cached)):
        raise ValueError("cached permutation must reverse complete original/forced pairs")
    aligned = [permuted[permutation.index(index)] for index in range(len(permutation))]
    native_gate = base.validate_fp32_agreement(cached, native)
    single_gate = base.validate_fp32_agreement(cached, single)
    permutation_gate = base.validate_fp32_agreement(cached, aligned)
    if cached != replay:
        raise ValueError("exact cached FP32 replay failed")
    return {"cached_vs_uncached": native_gate, "cached_vs_unpadded_single": single_gate,
            "cached_permutation": permutation_gate,
            "exact_cached_replay_passed": True}


def cached_budget_projection(processed: int, total: int, seconds: float,
                             remaining_seconds: float, batch_seconds: list[float],
                             batch_size: int) -> dict[str, Any]:
    """Use the worse of mean times 1.2 and empirical nearest-rank p95 batches."""
    initial = base.budget_projection(processed, total, seconds, remaining_seconds)
    if (not batch_seconds or any(not math.isfinite(value) or value <= 0 for value in batch_seconds)
            or type(batch_size) is not int or batch_size < 1):
        raise ValueError("valid measured cached batches required")
    p95 = sorted(batch_seconds)[math.ceil(.95 * len(batch_seconds)) - 1]
    mean_projection = initial["projected_remaining_seconds"]
    p95_projection = p95 * math.ceil((total - processed) / batch_size)
    conservative = max(mean_projection * 1.2, p95_projection)
    return {**initial, "safety_factor": 1.2, "forecast_method": "max(mean_times_1_2,nearest_rank_p95_batch_projection)",
            "mean_projected_remaining_seconds": mean_projection,
            "p95_batch_seconds": p95, "p95_projected_remaining_seconds": p95_projection,
            "projected_remaining_seconds": conservative,
            "benchmark_batch_seconds": list(batch_seconds),
            "proceed": conservative + base.SHUTDOWN_SECONDS < remaining_seconds}

def validate_diagnostics(batched: list[list[float]], single: list[list[float]],
                         replay: list[list[float]], permuted: list[list[float]],
                         permutation: list[int]) -> dict[str, Any]:
    """Gate FP32 single/batch, exact replay, and reordering independently."""
    if sorted(permutation) != list(range(len(batched))) or len(permuted) != len(permutation):
        raise ValueError("diagnostic permutation is incomplete")
    aligned = [permuted[permutation.index(index)] for index in range(len(permutation))]
    single_gate = base.validate_fp32_agreement(batched, single)
    permutation_gate = base.validate_fp32_agreement(batched, aligned)
    if replay != batched:
        raise ValueError("exact FP32 fixed-shape replay failed")
    return {"validation_method": VALIDATION_METHOD, "all_gates_passed": True,
            "batch_vs_unpadded_single": single_gate, "permutation": permutation_gate,
            "exact_replay_passed": True, "fp32_atol": base.FP32_ATOL, "fp32_rtol": base.FP32_RTOL}


def validate_coverage(expected: list[dict[str, Any]], rows: list[dict[str, Any]],
                      *, complete: bool) -> None:
    """Require an exact paired ordered prefix and no duplicated score identity."""
    if len(rows) > len(expected) or (complete and len(rows) != len(expected)):
        raise ValueError("paired score coverage mismatch")
    seen = set()
    for job, row in zip(expected, rows):
        for key in ("score_id", "score_index", "job_index", "job_id", "prompt_condition",
                    "original_prompt_sha256", "prompt_sha256"):
            if row.get(key) != job[key]:
                raise ValueError("paired score identity, ordering, or prompt hash mismatch")
        if row["score_id"] in seen:
            raise ValueError("duplicate paired score identity")
        seen.add(row["score_id"])


def validate_promotion(model: Any, original: dict[str, Any]) -> dict[str, Any]:
    """Verify exact sampled values and FP32 dtype after lossless BF16 promotion."""
    current = base._named_tensors(model)
    if set(current) != set(original):
        raise ValueError("model tensor names changed during precision promotion")
    after = {}
    for name, tensor in current.items():
        state = original[name]
        if list(tensor.shape) != state["shape"] or base._tensor_sample(tensor) != state["sample"]:
            raise ValueError("model tensor shape or sampled values changed during promotion")
        if tensor.is_floating_point() and str(tensor.dtype) != "torch.float32":
            raise ValueError("all floating model tensors must remain FP32 for production")
        if not tensor.is_floating_point() and tensor.dtype != state["dtype"]:
            raise ValueError("nonfloating model tensor dtype changed during promotion")
        after[name] = str(tensor.dtype)
    return {"schema_version": "acl-paired-dtype-promotion-v1", "all_checks_passed": True,
            "original": {name: {**state, "dtype": str(state["dtype"])} for name, state in original.items()},
            "promoted_dtypes": after, "sampled_values_preserved_exactly": True,
            "all_floating_tensors_fp32": True,
            "sample_rule": "first two and last two flattened values of every named tensor"}


def run_scoring(tag: str, jobs_path: Path, cache_dir: Path, out_dir: Path,
                max_seconds: float = MAX_SECONDS, batch_size: int = BATCH_SIZE,
                progress: Callable[[dict[str, Any]], None] | None = None,
                shared_prefix_cache: bool = False) -> dict[str, Any]:
    """Execute paired prompt scoring with fixed FP32 production and a deadline.

    Parameters
    ----------
    tag : str
        Original pinned model tag, qwen3b or qwen7b.
    jobs_path, cache_dir, out_dir : Path
        Exact frozen public inputs, verified offline model cache, and new output.
    max_seconds : float
        Full worker deadline, including imports, hashing, preparation and load.
    batch_size : int
        Exactly 32 uncached, or 128 with shared_prefix_cache=True.
    shared_prefix_cache : bool
        Explicit reviewed recovery mode; cache only exact paired token prefixes.
    progress : callable, optional
        Called with all streams closed so the provider can commit checkpoints.

    Returns
    -------
    dict
        Durable receipt, including partial completion and failure evidence.
    """
    expected_batch_size = CACHE_BATCH_SIZE if shared_prefix_cache else BATCH_SIZE
    if (type(shared_prefix_cache) is not bool or tag not in MODELS
            or type(batch_size) is not int or batch_size != expected_batch_size):
        raise ValueError("a pinned model and mode-specific batch size (32 or cached 128) are required")
    if (isinstance(max_seconds, bool) or not isinstance(max_seconds, (int, float))
            or not math.isfinite(max_seconds) or not 0 < max_seconds <= MAX_SECONDS):
        raise ValueError("max_seconds must be finite, positive, and at most 1400")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    rows = []
    torch_runtime = None
    model_name, revision = MODELS[tag], PINNED_MODELS[MODELS[tag]]
    receipt = {"schema_version": "acl-paired-prompt-scoring-receipt-v1", "model_tag": tag,
        "model": model_name, "revision": revision, "protocol": PROTOCOL,
        "public_file_sha256": PUBLIC_SHA256, "expected_rows": EXPECTED_ROWS,
        "started_at": datetime.now(timezone.utc).isoformat(), "max_seconds": max_seconds,
        "automatic_retries": 0}
    base.write_once(out_dir / "started.json", receipt)

    def remaining() -> float:
        return max_seconds - (time.monotonic() - started)

    def checkpoint(phase: str, **extra: Any) -> None:
        if progress is not None:
            progress({"phase": phase, "completed_rows": len(rows),
                      "elapsed_seconds": time.monotonic() - started, **extra})

    def check_deadline() -> None:
        if remaining() <= base.SHUTDOWN_SECONDS:
            raise TimeoutError("paired scoring internal deadline reached")

    try:
        jobs = base.load_jobs(Path(jobs_path))
        expected = derived_jobs(jobs)
        base.write_once(out_dir / "prompts.json", prompt_manifest())
        check_deadline()
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        if os.environ["CUBLAS_WORKSPACE_CONFIG"] not in {":4096:8", ":16:8"}:
            raise ValueError("unsupported deterministic CUBLAS workspace")
        import torch
        torch_runtime = torch
        from huggingface_hub import snapshot_download
        from transformers import AutoModelForCausalLM, AutoTokenizer
        expected_versions = {"torch": "2.6.0", "transformers": "4.51.3", "tokenizers": "0.21.1",
                             "safetensors": "0.5.3", "huggingface-hub": "0.30.2"}
        versions = {name: importlib.metadata.version(name) for name in expected_versions}
        if any(versions[name].split("+")[0] != value for name, value in expected_versions.items()):
            raise ValueError("model stack differs from original pinned versions")
        if (not torch.cuda.is_available() or torch.cuda.device_count() != 1
                or not torch.cuda.is_bf16_supported()):
            raise RuntimeError("exactly one BF16-capable CUDA GPU required; no CPU fallback")
        torch.set_num_threads(2)
        torch.manual_seed(1)
        torch.cuda.manual_seed_all(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        snapshot = Path(snapshot_download(repo_id=model_name, revision=revision, cache_dir=cache_dir,
            local_files_only=True, allow_patterns=["*.json", "*.safetensors", "*.txt"]))
        if snapshot.name != revision:
            raise ValueError("snapshot differs from pinned revision")
        hashes = {}
        for path in sorted(snapshot.rglob("*")):
            if path.is_file() and ".cache" not in path.relative_to(snapshot).parts:
                check_deadline()
                hashes[str(path.relative_to(snapshot))] = base.file_hash(path)
        original_receipt = Path(cache_dir) / f"{tag}_expected_model_hashes.json"
        original = base.load_json(original_receipt.read_bytes())
        if (original.get("model") != model_name or original.get("revision") != revision
                or original.get("model_files_sha256") != hashes
                or not any(name.endswith(".safetensors") for name in hashes)):
            raise ValueError("model files differ from original frozen model receipt")
        tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=False)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise ValueError("tokenizer has no padding token")
            tokenizer.pad_token = tokenizer.eos_token
        prep_started = time.monotonic()
        contexts = []
        for index, job in enumerate(expected):
            if index % 64 == 0:
                check_deadline()
            contexts.append(base.prepare_context(tokenizer, job))
        if any(context["option_token_ids"] != contexts[0]["option_token_ids"] for context in contexts):
            raise ValueError("option token IDs vary across scored contexts")
        padded_width = max(len(context["scored_input_token_ids"]) for context in contexts)
        shared_plan = cache_plan(contexts) if shared_prefix_cache else None
        if shared_plan is not None:
            base.write_once(out_dir / "cache_plan.json", shared_plan)
        selection = select_diagnostic_menus(jobs, contexts)
        base.write_once(out_dir / "diagnostic_selection.json", selection)
        repo = Path(__file__).resolve().parents[1]
        metadata = {"schema_version": "acl-paired-prompt-scoring-metadata-v1", "model_tag": tag,
            "model": model_name, "revision": revision, "protocol": PROTOCOL,
            "public_file_sha256": PUBLIC_SHA256, "n_jobs": len(jobs), "n_score_rows": len(expected),
            "prompt_conditions": list(PROMPT_CONDITIONS), "assistant_prefix": ASSISTANT_PREFIX,
            "prompt_manifest_sha256": base.file_hash(out_dir / "prompts.json"),
            "source_files_sha256": {name: base.file_hash(repo / name) for name in SOURCE_FILES},
            "model_files_sha256": hashes, "model_hashes_bound_to_original_run": True,
            "expected_model_hashes_receipt_sha256": base.file_hash(original_receipt),
            "chat_template_sha256": base.sha(tokenizer.chat_template.encode()),
            "versions": versions, "dtype": "float32", "weight_load_dtype": "bfloat16",
            "promotion": "exact promotion of original loaded BF16 stored-weight values to FP32",
            "attention_implementation": "eager", "tf32": False, "float32_matmul_precision": "highest",
            "deterministic_algorithms": True, "seed": 1, "batch_size": batch_size,
            "global_padded_width": padded_width, "padding_side": "left",
            "filler_policy": ("duplicate final complete original/forced pair; discard filler outputs" if shared_prefix_cache else
                              "duplicate final real context to fixed batch size; discard filler outputs"),
            "option_token_ids": contexts[0]["option_token_ids"], "logits_to_keep": 1,
            "use_cache": shared_prefix_cache, "shared_prefix_cache": shared_prefix_cache,
            "reference_batch_size": BATCH_SIZE,
            "cache_mode": "paired_exact_token_lcp_dynamic_cache" if shared_prefix_cache else "disabled",
            "cache_prefix_width": shared_plan["prefix_width"] if shared_plan else None,
            "cache_suffix_width": shared_plan["suffix_width"] if shared_plan else None,
            "cache_plan_sha256": base.file_hash(out_dir / "cache_plan.json") if shared_plan else None,
            "cache_position_policy": "physical prefix/suffix offsets; logical position_ids exclude padding",
            "generation": False, "sampling": False, "quantization": False,
            "gpu": torch.cuda.get_device_name(0), "torch_cuda_version": torch.version.cuda,
            "validation_method": CACHE_VALIDATION_METHOD if shared_prefix_cache else VALIDATION_METHOD,
            "fp32_numeric_gate": {"atol": base.FP32_ATOL, "rtol": base.FP32_RTOL,
                                  "exact_top_equality_required": False, "exact_replay_required": True},
            "diagnostic_selection_sha256": base.file_hash(out_dir / "diagnostic_selection.json"),
            "context_preparation_seconds": time.monotonic() - prep_started,
            "scope": "conditional option preference under paired menu-only prompts; no correctness calibration"}
        base.write_once(out_dir / "metadata.json", metadata)
        checkpoint("prepared")
        check_deadline()
        load_started = time.monotonic()
        model = AutoModelForCausalLM.from_pretrained(str(snapshot), local_files_only=True,
            trust_remote_code=False, use_safetensors=True, torch_dtype=torch.bfloat16,
            attn_implementation="eager").to("cuda:0").eval()
        torch.cuda.synchronize()
        receipt["model_load_seconds"] = time.monotonic() - load_started
        original_state = base.snapshot_tensor_state(model)
        diag_started = time.monotonic()
        diag_contexts = [contexts[index] for index in selection["score_indices"]]
        diag_ids = [expected[index]["score_id"] for index in selection["score_indices"]]
        check_deadline()
        bf16_batched = forward(torch, model, tokenizer, diag_contexts, padded_width=padded_width, batch_size=BATCH_SIZE)
        bf16_single = []
        for context in diag_contexts:
            check_deadline()
            bf16_single.append(forward(torch, model, tokenizer, [context])[0])
        base.write_once(out_dir / "diagnostics_bf16.json", {
            "score_indices": selection["score_indices"], "score_ids": diag_ids, "contexts": diag_contexts,
            "batched_logits": bf16_batched, "single_logits": bf16_single,
            "diagnostics": base.comparison_diagnostics(bf16_batched, bf16_single),
            "gate": False, "batch_size": BATCH_SIZE, "global_padded_width": padded_width})
        checkpoint("bf16_diagnostics")
        promotion_started = time.monotonic()
        torch.cuda.empty_cache()
        model.float()
        torch.cuda.synchronize()
        base.write_once(out_dir / "dtype_promotion.json", validate_promotion(model, original_state))
        receipt["promotion_seconds"] = time.monotonic() - promotion_started
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        check_deadline()
        fp32_batched = forward(torch, model, tokenizer, diag_contexts, padded_width=padded_width, batch_size=BATCH_SIZE)
        fp32_single = []
        for context in diag_contexts:
            check_deadline()
            fp32_single.append(forward(torch, model, tokenizer, [context])[0])
        replay = forward(torch, model, tokenizer, diag_contexts, padded_width=padded_width, batch_size=BATCH_SIZE)
        permutation = list(reversed(range(len(diag_contexts))))
        permuted = forward(torch, model, tokenizer, [diag_contexts[index] for index in permutation],
                           padded_width=padded_width, batch_size=BATCH_SIZE)
        base.write_once(out_dir / "diagnostics_fp32.json", {
            "score_indices": selection["score_indices"], "score_ids": diag_ids, "contexts": diag_contexts,
            "batched_logits": fp32_batched, "single_logits": fp32_single,
            "replay_logits": replay, "permutation_indices": permutation, "permuted_logits": permuted,
            "batch_size": BATCH_SIZE, "global_padded_width": padded_width,
            "bf16_vs_fp32_batched": base.comparison_diagnostics(bf16_batched, fp32_batched),
            "bf16_vs_fp32_single": base.comparison_diagnostics(bf16_single, fp32_single)})
        checkpoint("fp32_diagnostic_vectors")
        cached_evidence = None
        if shared_plan is not None:
            kwargs = {"prefix_width": shared_plan["prefix_width"],
                      "suffix_width": shared_plan["suffix_width"], "batch_size": batch_size}
            check_deadline()
            cached = cached_forward(torch, model, tokenizer, diag_contexts, **kwargs)
            cached_replay = cached_forward(torch, model, tokenizer, diag_contexts, **kwargs)
            pair_permutation = reverse_pair_permutation(len(diag_contexts))
            cached_permuted = cached_forward(torch, model, tokenizer,
                [diag_contexts[index] for index in pair_permutation], **kwargs)
            cached_evidence = {"score_indices": selection["score_indices"], "score_ids": diag_ids,
                "contexts": diag_contexts, "cached_logits": cached, "uncached_logits": fp32_batched,
                "single_logits": fp32_single,
                "cached_replay_logits": cached_replay, "permutation_indices": pair_permutation,
                "permuted_cached_logits": cached_permuted, **kwargs}
            base.write_once(out_dir / "diagnostics_cached.json", cached_evidence)
            checkpoint("cached_diagnostic_vectors")
        gates = validate_diagnostics(fp32_batched, fp32_single, replay, permuted, permutation)
        if cached_evidence is not None:
            gates.update(validate_cached_diagnostics(cached, fp32_batched, cached_replay,
                                                       cached_permuted, pair_permutation, fp32_single))
            gates["validation_method"] = CACHE_VALIDATION_METHOD
            gates["diagnostic_cached_sha256"] = base.file_hash(out_dir / "diagnostics_cached.json")
        gates["diagnostic_bf16_sha256"] = base.file_hash(out_dir / "diagnostics_bf16.json")
        gates["diagnostic_fp32_sha256"] = base.file_hash(out_dir / "diagnostics_fp32.json")
        gates["dtype_promotion_sha256"] = base.file_hash(out_dir / "dtype_promotion.json")
        base.write_once(out_dir / "diagnostics_validation.json", gates)
        receipt["diagnostics_seconds"] = time.monotonic() - diag_started
        receipt["ready_seconds"] = time.monotonic() - started
        (out_dir / "scores.jsonl").touch(exist_ok=False)
        (out_dir / "batch_checks.jsonl").touch(exist_ok=False)
        checkpoint("diagnostics_passed")
        benchmark_started = time.monotonic()
        last_batch_seconds, next_checkpoint = 0.0, 512
        benchmark_batch_seconds = []
        while len(rows) < len(expected):
            check_deadline()
            if last_batch_seconds * base.SAFETY_FACTOR + base.SHUTDOWN_SECONDS >= remaining():
                receipt["status"] = "deadline_stop"
                break
            offset = len(rows)
            end = min(offset + batch_size, BENCHMARK_ROWS if offset < BENCHMARK_ROWS else len(expected))
            batch_started = time.monotonic()
            if shared_plan is not None:
                logits = cached_forward(torch, model, tokenizer, contexts[offset:end],
                    prefix_width=shared_plan["prefix_width"], suffix_width=shared_plan["suffix_width"],
                    batch_size=batch_size)
            else:
                logits = forward(torch, model, tokenizer, contexts[offset:end],
                                 padded_width=padded_width, batch_size=batch_size)
            batch_rows = []
            for job, context, values in zip(expected[offset:end], contexts[offset:end], logits):
                row = {"schema_version": "acl-paired-prompt-scores-v1",
                    **{key: job[key] for key in ("score_id", "score_index", "job_index", "job_id", "qid",
                        "group_id", "split", "condition", "menu_id", "prompt_condition",
                        "original_prompt_sha256", "prompt_sha256")},
                    **context, **base.option_statistics(values), "batch_index": offset // batch_size}
                batch_rows.append(row)
            with (out_dir / "scores.jsonl").open("ab") as stream:
                stream.write(b"".join(base.canonical(row) for row in batch_rows))
                stream.flush()
                os.fsync(stream.fileno())
            rows.extend(batch_rows)
            last_batch_seconds = time.monotonic() - batch_started
            if offset < BENCHMARK_ROWS:
                benchmark_batch_seconds.append(last_batch_seconds)
            check = {"batch_index": offset // batch_size, "start": offset, "stop": len(rows),
                "physical_rows": batch_size, "real_rows": len(batch_rows), "padded_width": padded_width,
                "rows_sha256": base.sha(b"".join(base.canonical(row) for row in batch_rows)),
                "seconds": last_batch_seconds, "elapsed_seconds": time.monotonic() - started}
            with (out_dir / "batch_checks.jsonl").open("ab") as stream:
                stream.write(base.canonical(check))
                stream.flush()
                os.fsync(stream.fileno())
            if len(rows) == BENCHMARK_ROWS:
                if shared_prefix_cache:
                    benchmark = cached_budget_projection(len(rows), len(expected),
                        time.monotonic() - benchmark_started, remaining(), benchmark_batch_seconds, batch_size)
                else:
                    benchmark = base.budget_projection(len(rows), len(expected),
                        time.monotonic() - benchmark_started, remaining())
                base.write_once(out_dir / "benchmark.json", benchmark)
                receipt["benchmark"] = benchmark
                checkpoint("benchmark", **benchmark)
                if not benchmark["proceed"]:
                    receipt["status"] = "benchmark_budget_stop"
                    break
            if len(rows) >= next_checkpoint:
                checkpoint("scoring")
                next_checkpoint = len(rows) + 512
        if len(rows) == len(expected):
            validate_coverage(expected, rows, complete=True)
            receipt["status"] = "complete"
        else:
            validate_coverage(expected, rows, complete=False)
    except TimeoutError as error:
        receipt["status"] = "deadline_stop"
        receipt["error"] = f"{type(error).__name__}: {error}"
    except Exception as error:
        receipt["status"] = "failed"
        receipt["error"] = f"{type(error).__name__}: {error}"
    if torch_runtime is not None:
        try:
            receipt["max_cuda_memory_allocated_bytes"] = torch_runtime.cuda.max_memory_allocated()
        except Exception:
            receipt["max_cuda_memory_allocated_bytes"] = None
    numeric_files = NUMERIC_FILES + (("diagnostics_cached.json",) if shared_prefix_cache else ())
    receipt.update({"completed_rows": len(rows), "finished_at": datetime.now(timezone.utc).isoformat(),
        "total_seconds": time.monotonic() - started,
        "scores_sha256": base.file_hash(out_dir / "scores.jsonl") if (out_dir / "scores.jsonl").exists() else None,
        "metadata_sha256": base.file_hash(out_dir / "metadata.json") if (out_dir / "metadata.json").exists() else None,
        "numeric_evidence_sha256": {name: base.file_hash(out_dir / name) for name in numeric_files if (out_dir / name).exists()}})
    base.write_once(out_dir / "receipt.json", receipt)
    return receipt
