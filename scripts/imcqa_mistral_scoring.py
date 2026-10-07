#!/usr/bin/env python3
"""Dedicated bounded Mistral replication with inherited FP32 evidence gates.

The orchestration is an additive adaptation of imcqa_tuned_scoring.run_scoring.
A wrapper cannot replace its hard-coded Qwen model and tokenizer boundary without
mutating globals. Shared, model-agnostic forward/cache/numeric helpers are called
unchanged. No Qwen modules or globals are modified. Live Mistral numerical
correctness must pass the retained gates; CPU tests do not establish it.
"""
from __future__ import annotations

from datetime import datetime, timezone
import importlib.metadata
import math
import os
from pathlib import Path
import time
from types import SimpleNamespace
from typing import Callable

from scripts import acl_option_scoring as base
from scripts import acl_paired_prompt_scoring as paired
from scripts import imcqa_protocol_scoring as original
from scripts import imcqa_tuned_scoring as inherited
from scripts import imcqa_mistral_design as design

MODEL_TAG = "mistral7b"
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
MODEL_REVISION = "c170c708c41dac9275d15a8fff4eca08d52bab71"
PROTOCOL = "imcqa-mistral-replication-v1"
SCORE_SCHEMA = "imcqa-mistral-scores-v1"
BATCH_SIZE = 8
BENCHMARK_ROWS = 128
ASSISTANT_PREFIX = '{"action":"'
CONDITIONS = ("independent_pool", "same_category_pool")
INTERNAL_SECONDS = {"development": 1740, "evaluation": 8130}
QUESTION_COUNTS = {"development": 140, "evaluation": 850}
REQUIRED_VERSIONS = {
    "torch": "2.6.0", "transformers": "4.51.3", "tokenizers": "0.21.1",
    "safetensors": "0.5.3", "huggingface-hub": "0.30.2", "jinja2": "3.1.6",
    "sentencepiece": "0.2.0", "protobuf": "5.29.5",
}
MODEL_FILES = (
    "config.json", "generation_config.json", "model.safetensors.index.json",
    "model-00001-of-00003.safetensors", "model-00002-of-00003.safetensors",
    "model-00003-of-00003.safetensors", "tokenizer.json", "tokenizer.model",
    "tokenizer_config.json", "special_tokens_map.json",
)
TOKENIZER_HASHES = {
    "special_tokens_map.json": "6fa06efa2785e450051989a6f8fb4416b10149ded485ddd3f127a40734f5cfd0",
    "tokenizer.json": "e553af6fff7d7ad76e830608b218c5c0b0822998d5a1a96099a74cd3c1cb1a49",
    "tokenizer.model": "37f00374dea48658ee8f5d0f21895b9bc55cb0103939607c8185bfd1c6ca1f89",
    "tokenizer_config.json": "0533dec9cfe319163801b6618d0f3ec9cfa126b6288e3df5deca6e32acb09cd2",
}
CHAT_TEMPLATE_SHA256 = "e16746b40344d6c5b5265988e0328a0bf7277be86f1c335156eae07e29c82826"
ACTION_TOKEN_IDS = {"A": 29509, "B": 29528, "C": 29511, "D": 29525, "E": 29517}
SOURCE_FILES = tuple(sorted(set(inherited.SOURCE_FILES + (
    "scripts/__init__.py", "scripts/imcqa_mistral_scoring.py",
    "scripts/imcqa_mistral_design.py", "configs/imcqa_mistral_replication.json",
))))
# No model-specific mutable bindings are replaced in the inherited modules.
forward = original.forward
numeric_agreement = original.numeric_agreement
action_statistics = original.action_statistics
paired_order = original.paired_order
diagnostic_indices = inherited.diagnostic_indices
production_offsets = inherited.production_offsets
replay_qids = inherited.replay_qids
validate_replay_evidence = inherited.validate_replay_evidence
projection_with_validation = inherited.projection_with_validation


def source_identity() -> dict[str, str]:
    """Hash the full local source closure used by this worker."""
    root = Path(__file__).resolve().parents[1]
    return {name: base.file_hash(root / name) for name in SOURCE_FILES}


def validate_run_package(package: dict, max_seconds: float) -> list[dict]:
    """Reject unsupported phases, runtime caps, models, or incomplete grids."""
    jobs = design.validate_public_package(package)
    stage = package.get("stage")
    if stage not in QUESTION_COUNTS or max_seconds != INTERNAL_SECONDS[stage]:
        raise ValueError("stage or frozen internal runtime cap differs")
    n = len({job["qid"] for job in jobs})
    if (n != QUESTION_COUNTS[stage] or len(jobs) != n * 40
            or any(job["execution"] != "new" or job["arm"] != "plain"
                   or job["block"] != "factorial" or job["allowed_actions"] != "ABCD"
                   for job in jobs)):
        raise ValueError("exact new plain-factorial phase grid required")
    if package["model"] != {"tag": MODEL_TAG, "name": MODEL_NAME, "revision": MODEL_REVISION}:
        raise ValueError("pinned Mistral identity differs")
    return jobs


def validate_model_file_hashes(hashes: dict[str, str]) -> None:
    """Require exact Transformers shards and previously checked tokenizer bytes."""
    if set(hashes) != set(MODEL_FILES):
        raise ValueError("model cache file allowlist differs")
    if any(len(v) != 64 or any(c not in "0123456789abcdef" for c in v) for v in hashes.values()):
        raise ValueError("model cache hashes malformed")
    if any(hashes[k] != v for k, v in TOKENIZER_HASHES.items()):
        raise ValueError("pinned tokenizer bytes differ")


def prepare_context(tokenizer, job: dict) -> dict:
    """Preserve native Mistral assistant spacing and exact A-E token boundaries."""
    user = [{"role": "user", "content": job["prompt"]}]
    rendered = tokenizer.apply_chat_template(user, tokenize=False, add_generation_prompt=True)
    scored = tokenizer.apply_chat_template(
        user + [{"role": "assistant", "content": ASSISTANT_PREFIX}],
        tokenize=False, continue_final_message=True)
    if scored != rendered + " " + ASSISTANT_PREFIX:
        raise ValueError("pinned native assistant boundary changed")
    encode = lambda text: tokenizer(text, add_special_tokens=False)["input_ids"]
    original_ids, ids = encode(rendered), encode(scored)
    if not original_ids or ids[:len(original_ids)] != original_ids or not len(original_ids) < len(ids) <= 2048:
        raise ValueError("invalid scored context boundary or token limit")
    option_ids = {}
    for label in "ABCDE":
        extended = encode(scored + label)
        if (len(extended) != len(ids) + 1 or extended[:-1] != ids
                or tokenizer.decode([extended[-1]], skip_special_tokens=False,
                                    clean_up_tokenization_spaces=False) != label):
            raise ValueError("action is not an exact one-token extension")
        option_ids[label] = extended[-1]
    if option_ids != ACTION_TOKEN_IDS:
        raise ValueError("pinned action token IDs differ")
    return {"rendered_prompt_sha256": base.sha(rendered.encode()),
            "scored_context_sha256": base.sha(scored.encode()),
            "scored_input_token_ids": ids, "option_token_ids": option_ids}


def validate_model_config(config, contexts: list[dict]) -> dict:
    """Bound physical cache widths so padding cannot trigger a sliding window."""
    if getattr(config, "model_type", None) != "mistral" or getattr(config, "_attn_implementation", None) != "eager":
        raise ValueError("Mistral eager-attention config required")
    if not contexts or len(contexts) % BATCH_SIZE:
        raise ValueError("complete physical batches required")
    # A global prefix/suffix upper bound also covers diagnostic regrouping and
    # pair permutations, not only the original physical production batches.
    plan = paired.cache_plan(contexts)
    max_width = max(plan["prefix_width"] + plan["suffix_width"],
                    max(len(c["scored_input_token_ids"]) for c in contexts))
    # Singleton diagnostics need no more than the longest unpadded context.
    window = getattr(config, "sliding_window", None)
    if window is not None and (type(window) is not int or window < max_width):
        raise ValueError("physical cache padding can reach the model sliding window")
    positions = getattr(config, "max_position_embeddings", None)
    if type(positions) is not int or positions < max_width:
        raise ValueError("physical cache width exceeds model position limit")
    return {"model_type": "mistral", "attention": "eager", "sliding_window": window,
            "max_position_embeddings": positions, "maximum_physical_cache_width": max_width,
            "cache_padding_window_gate": "passed"}


def forward_batches(torch, model, tokenizer, contexts, *, cached, permute=False):
    """Bound every physical forward to eight rows with complete cache pairs."""
    if not contexts or len(contexts) % 2:
        raise ValueError("nonempty complete context pairs required")
    result = []
    for offset in range(0, len(contexts), BATCH_SIZE):
        batch = contexts[offset:offset + BATCH_SIZE]
        permutation = paired.reverse_pair_permutation(len(batch)) if permute else list(range(len(batch)))
        values = forward(torch, model, tokenizer, [batch[i] for i in permutation],
                         cached=cached, batch_size=BATCH_SIZE)
        result.extend(values[permutation.index(i)] for i in range(len(batch)))
    return result


def record_numeric_gate(path, left, right, jobs, *, extra=None):
    """Save both raw comparison sides before a numerical rejection."""
    base.write_once(path, {"score_ids": [j["score_id"] for j in jobs],
                           "production": left, "reference": right, **(extra or {})})
    return numeric_agreement(left, right, jobs)


def validate_rows(expected, rows, *, complete):
    """Check exact new public identities plus independent legal-score statistics."""
    if len(rows) > len(expected) or (complete and len(rows) != len(expected)):
        raise ValueError("score coverage mismatch")
    seen = set()
    for job, row in zip(expected, rows):
        if any(row.get(k) != job[k] for k in design.PUBLIC_JOB_KEYS - {"prompt"}):
            raise ValueError("score identity or order differs")
        if row.get("model_tag") != MODEL_TAG or row.get("schema_version") != SCORE_SCHEMA:
            raise ValueError("score model or schema differs")
        if row["score_id"] in seen:
            raise ValueError("duplicate score row")
        seen.add(row["score_id"])
        stats = action_statistics([row["raw_action_logits"][k] for k in "ABCDE"],
                                  job["allowed_actions"], job["option_source_ids"])
        if any(row.get(k) != value for k, value in stats.items()):
            raise ValueError("checkpoint probability or action differs from logits")


def verify_numeric_record(record, reference, comparisons, jobs, production):
    """Bind raw comparison sides to the actual production logits, then compare."""
    ids = record.get("score_ids", [])
    if not ids or len(set(ids)) != len(ids) or not set(ids) <= set(jobs):
        raise ValueError("invalid numerical diagnostic identities")
    targets = [jobs[sid] for sid in ids]
    numeric_agreement([production[sid] for sid in ids], record[reference], targets)
    return {name: numeric_agreement(record[reference], record[name], targets) for name in comparisons}


def validate_completed_run(public_path: Path, run_dir: Path) -> dict:
    """CPU-replay complete Mistral evidence before fitting or interpreting scores.

    This validates saved arithmetic and bindings. It does not replace the actual
    live worker gates or prove model authenticity independently of cache/source
    provenance. An incomplete worker can never pass this entry point.
    """
    public_path, run_dir = Path(public_path), Path(run_dir)
    read = lambda path: base.load_json(Path(path).read_bytes())
    package = read(public_path)
    stage = package.get("stage")
    if stage not in INTERNAL_SECONDS:
        raise ValueError("unknown completed stage")
    jobs_list = validate_run_package(package, INTERNAL_SECONDS[stage])
    jobs = {j["score_id"]: j for j in jobs_list}
    receipt = read(run_dir / "receipt.json")
    expected_receipt = {"protocol": PROTOCOL, "stage": stage, "model_tag": MODEL_TAG,
        "model": MODEL_NAME, "revision": MODEL_REVISION, "status": "complete",
        "public_input_sha256": base.file_hash(public_path), "expected_rows": len(jobs),
        "completed_rows": len(jobs), "total_contexts": len(jobs), "attempt": 0,
        "automatic_retries": 0, "sampling": False, "generation": False, "reused_rows": 0,
        "batch_size": BATCH_SIZE, "cached": True, "max_seconds": INTERNAL_SECONDS[stage]}
    if any(receipt.get(k) != v for k, v in expected_receipt.items()):
        raise ValueError("complete frozen worker receipt required")
    if not isinstance(receipt.get("source_commit"), str) or len(receipt["source_commit"]) != 40 or any(c not in "0123456789abcdef" for c in receipt["source_commit"]):
        raise ValueError("committed worker source identity required")
    if read(run_dir / "attempts/000_receipt.json") != receipt:
        raise ValueError("attempt and final receipts disagree")
    scores_path = run_dir / "scores.jsonl"
    if receipt.get("scores_sha256") != base.file_hash(scores_path):
        raise ValueError("complete score hash differs")
    raw = scores_path.read_bytes()
    if not raw.endswith(b"\n"):
        raise ValueError("score file has incomplete trailing row")
    rows = [base.load_json(line) for line in raw.splitlines()]
    if len(rows) != len(jobs) or {r["score_id"] for r in rows} != set(jobs):
        raise ValueError("complete unique score coverage required")
    ordered_jobs = [jobs[r["score_id"]] for r in rows]
    validate_rows(ordered_jobs, rows, complete=True)
    for row in rows:
        ids = row.get("scored_input_token_ids")
        if (row.get("option_token_ids") != ACTION_TOKEN_IDS or not isinstance(ids, list)
                or not 1 < len(ids) <= 2048
                or any(type(i) is not int or not 0 <= i < 32768 for i in ids)):
            raise ValueError("saved Mistral token context differs")
    lookup = {r["score_id"]: r for r in rows}
    original_contexts = [lookup[j["score_id"]] for j in jobs_list]
    expected_order = [jobs_list[i]["score_id"] for i in paired_order(original_contexts)]
    plan = read(run_dir / "plan.json")
    if (plan.get("ordered_score_ids") != expected_order or [r["score_id"] for r in rows] != expected_order
            or plan.get("input_sha256") != base.file_hash(public_path)
            or plan.get("context_sha256") != [r["scored_context_sha256"] for r in rows]
            or plan.get("token_counts") != [len(r["scored_input_token_ids"]) for r in rows]
            or plan.get("new_rows") != len(jobs) or plan.get("reused_rows") != 0):
        raise ValueError("saved token/order plan differs")
    metadata = read(run_dir / "metadata.json")
    expected_metadata = {"protocol": PROTOCOL, "stage": stage, "model": MODEL_NAME,
        "revision": MODEL_REVISION, "dtype": "float32", "loaded_dtype": "bfloat16",
        "attention": "eager", "tf32": False, "seed": 1, "generation": False,
        "sampling": False, "chat_template_sha256": CHAT_TEMPLATE_SHA256,
        "assistant_prefix": ASSISTANT_PREFIX, "action_token_ids": ACTION_TOKEN_IDS,
        "source_commit": receipt["source_commit"], "source_files_sha256": source_identity(),
        "public_input_sha256": base.file_hash(public_path), "batch_size": BATCH_SIZE, "cached": True,
        "native_assistant_boundary": "one ASCII space before JSON action prefix"}
    if any(metadata.get(k) != v for k, v in expected_metadata.items()) or any(
        metadata.get("versions", {}).get(k, "").split("+")[0] != v for k, v in REQUIRED_VERSIONS.items()):
        raise ValueError("pinned model, source or numerical metadata differs")
    validate_model_file_hashes(metadata["model_files_sha256"])
    cache = read(run_dir.parent / "cache_prepare_receipt.json")
    if (cache.get("schema_version") != "imcqa-mistral-cache-v1" or cache.get("status") != "complete"
            or cache.get("chat_template_sha256") != CHAT_TEMPLATE_SHA256
            or cache.get("model_receipts", {}).get(MODEL_TAG) != {
            "model": MODEL_NAME, "revision": MODEL_REVISION,
            "model_files_sha256": metadata["model_files_sha256"]}):
        raise ValueError("model files differ from prepared cache receipt")
    config_record = read(run_dir / "model_config_validation.json")
    config = SimpleNamespace(model_type=config_record.get("model_type"),
        _attn_implementation=config_record.get("attention"), sliding_window=config_record.get("sliding_window"),
        max_position_embeddings=config_record.get("max_position_embeddings"))
    if validate_model_config(config, rows) != config_record:
        raise ValueError("model padding/window evidence differs")
    promotion = read(run_dir / "dtype_promotion.json")
    if any(promotion.get(k) is not True for k in (
            "all_checks_passed", "sampled_values_preserved_exactly", "all_floating_tensors_fp32")):
        raise ValueError("lossless FP32 promotion gate missing")
    if (not promotion.get("original") or set(promotion["original"]) != set(promotion.get("promoted_dtypes", {}))
            or any(promotion["promoted_dtypes"][k] != "torch.float32" for k, v in promotion["original"].items()
                   if v["dtype"] in {"torch.bfloat16", "torch.float16", "torch.float32", "torch.float64"})):
        raise ValueError("FP32 tensor promotion identity differs")
    attempts = run_dir / "attempts"
    benchmark = read(attempts / "000_benchmark.json")
    if benchmark != receipt.get("benchmark") or benchmark.get("proceed") is not True or benchmark.get("processed") != BENCHMARK_ROWS:
        raise ValueError("complete run requires matching accepted benchmark")
    projection = projection_with_validation(benchmark["processed"], len(jobs),
        benchmark["benchmark_wall_seconds"], benchmark["available_seconds_before_reserves"],
        benchmark["benchmark_batch_seconds"], benchmark["diagnostic_single_seconds"],
        production_diagnostic_seconds=benchmark["nonrecurring_production_diagnostic_seconds"],
        checkpoint_seconds=benchmark["checkpoint_callback_seconds"])
    if projection != {k: v for k, v in benchmark.items() if k != "checkpoint_timing_records"}:
        raise ValueError("benchmark projection evidence differs")
    production = {r["score_id"]: {"logits": [r["raw_action_logits"][k] for k in "ABCDE"]} for r in rows}
    diagnostic = read(attempts / "000_diagnostics.json")
    raw_diagnostic = read(attempts / "000_diagnostics_raw.json")
    if {k: v for k, v in diagnostic.items() if k != "gates"} != raw_diagnostic:
        raise ValueError("raw and checked initial diagnostics disagree")
    diagnostic_ids = [rows[i]["score_id"] for i in diagnostic_indices(ordered_jobs, rows)]
    if diagnostic.get("score_ids") != diagnostic_ids:
        raise ValueError("initial diagnostic coverage differs")
    checks = verify_numeric_record(diagnostic, "cached", ("uncached", "singles", "replay", "permuted_aligned"), jobs, production)
    permutation_single = numeric_agreement(diagnostic["permuted_aligned"], diagnostic["singles"], [jobs[i] for i in diagnostic_ids])
    expected_gates = {"cached_uncached": checks["uncached"], "cached_single": checks["singles"],
                      "permutation": checks["permuted_aligned"], "permutation_single": permutation_single}
    if diagnostic["cached"] != diagnostic["replay"] or diagnostic.get("gates") != expected_gates:
        raise ValueError("initial exact replay or gate record differs")
    offsets = production_offsets(len(jobs))
    record = read(attempts / "000_production_diagnostics.json")
    expected_ids = [rows[i]["score_id"] for offset in offsets for i in range(offset, offset + BATCH_SIZE)]
    if (record.get("offsets") != offsets or record.get("score_ids") != expected_ids
            or record.get("actual_production_batches") is not True or record.get("batch_size") != BATCH_SIZE
            or plan.get("production_diagnostic_offsets") != offsets):
        raise ValueError("production numerical coverage differs")
    raw_production = read(attempts / "000_production_diagnostics_raw.json")
    if {k: v for k, v in record.items() if k not in ("gate", "single_gate")} != raw_production:
        raise ValueError("raw and checked production diagnostics disagree")
    comparisons = verify_numeric_record(record, "production", ("single_reference", "diagnostic"), jobs, production)
    if (record["production"] != record["diagnostic"] or record.get("gate") != comparisons["diagnostic"]
            or record.get("single_gate") != comparisons["single_reference"]
            or receipt.get("production_numerical_gate") != comparisons["diagnostic"]
            or receipt.get("production_single_gate") != comparisons["single_reference"]):
        raise ValueError("production exact replay or gate record differs")
    for index, offset in enumerate(offsets):
        side = read(attempts / f"000_production_batch_{offset:05d}_raw.json")
        take = slice(index * BATCH_SIZE, (index + 1) * BATCH_SIZE)
        if any(side.get(k) != record[k][take] for k in ("score_ids", "production", "single_reference", "diagnostic")):
            raise ValueError("actual production-batch evidence differs from aggregate")
    live = read(attempts / "000_live_trajectories.json")
    validate_replay_evidence(live)
    if live["qids"] != replay_qids(jobs_list) or plan.get("live_replay_qids") != live["qids"]:
        raise ValueError("live question selection differs")
    for index, row in enumerate(live["rows"]):
        side = read(attempts / f"000_live_{index:03d}_raw.json")
        if side.get("score_ids") != [row["score_id"]] or any(
            row[k] != jobs[row["score_id"]][k] or side.get(k) != row[k] for k in ("qid", "condition", "rotation", "round")):
            raise ValueError("live diagnostic identity differs")
        agreement = verify_numeric_record(side, "production", ("reference",), jobs, production)["reference"]
        if agreement != row["agreement"] or side["reference"] != [side["live"]]:
            raise ValueError("live comparison record differs")
    return {"status": "complete", "passed": True, "validated_rows": len(rows),
        "validation": "saved_numerical_evidence_replay", "stage": stage,
        "receipt_sha256": base.file_hash(run_dir / "receipt.json"),
        "scores_sha256": receipt["scores_sha256"], "public_input_sha256": receipt["public_input_sha256"],
        "model": {"tag": MODEL_TAG, "name": MODEL_NAME, "revision": MODEL_REVISION},
        "source_commit": receipt["source_commit"], "rows": len(rows),
        "initial_diagnostic_rows": len(diagnostic_ids), "actual_production_diagnostic_rows": len(expected_ids),
        "live_trajectory_states": 80, "cache_prepare_receipt_sha256": base.file_hash(run_dir.parent / "cache_prepare_receipt.json")}

def run_scoring(tag: str, public_path: Path, expected_input_sha256: str, cache_dir: Path,
                out_dir: Path, *, source_commit: str, max_seconds: float, progress: Callable | None = None) -> dict:
    """Run one create-once paid allocation, preserving failure evidence."""
    if (tag != MODEL_TAG or type(max_seconds) not in (int, float) or not math.isfinite(max_seconds) or max_seconds <= 0
            or not isinstance(source_commit, str) or len(source_commit) != 40 or any(c not in "0123456789abcdef" for c in source_commit)):
        raise ValueError("invalid pinned model, committed source, or bounded deadline")
    if base.file_hash(public_path) != expected_input_sha256:
        raise ValueError("public input hash differs")
    package = base.load_json(public_path.read_bytes())
    jobs = validate_run_package(package, max_seconds)
    expected_rows = len(jobs)
    source_hashes = source_identity()
    out_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    attempts = out_dir / "attempts"
    attempts.mkdir()
    attempt = 0
    receipt = {"protocol": PROTOCOL, "model_tag": tag, "public_input_sha256": expected_input_sha256,
        "model": MODEL_NAME, "revision": MODEL_REVISION, "stage": package["stage"],
        "status": "started", "expected_rows": expected_rows, "total_contexts": len(jobs), "attempt": attempt,
        "started_utc": datetime.now(timezone.utc).isoformat(), "max_seconds": max_seconds,
        "automatic_retries": 0, "sampling": False, "generation": False, "reused_rows": 0,
        "source_commit": source_commit, "batch_size": BATCH_SIZE, "cached": True}
    rows, expected, contexts = [], [], []
    checkpoint_timings = []
    def remaining():
        return max_seconds-(time.monotonic()-started)
    def check_deadline():
        if remaining() <= 30:
            raise TimeoutError("internal worker deadline reached")
    def checkpoint(phase):
        if progress:
            checkpoint_started = time.monotonic()
            try:
                progress({"phase": phase, "completed_rows": len(rows), "expected_rows": expected_rows,
                          "elapsed_seconds": time.monotonic()-started})
            finally:
                checkpoint_timings.append({"phase": phase,
                    "seconds": time.monotonic()-checkpoint_started})
    def evidence(name, value):
        base.write_once(out_dir/name, value)
    try:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        if os.environ["CUBLAS_WORKSPACE_CONFIG"] not in {":4096:8", ":16:8"}:
            raise ValueError("deterministic CUBLAS configuration differs")
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoModelForCausalLM, AutoTokenizer
        required = REQUIRED_VERSIONS
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
        name, revision = MODEL_NAME, MODEL_REVISION
        snapshot = Path(snapshot_download(repo_id=name, revision=revision, cache_dir=cache_dir,
            local_files_only=True, allow_patterns=list(MODEL_FILES)))
        if snapshot.name != revision:
            raise ValueError("cached revision differs")
        hashes = {}
        for path in sorted(snapshot.rglob("*")):
            if path.is_file() and ".cache" not in path.relative_to(snapshot).parts:
                check_deadline(); hashes[str(path.relative_to(snapshot))] = base.file_hash(path)
        cached_model_receipt = base.load_json((cache_dir / f"{tag}_expected_model_hashes.json").read_bytes())
        if cached_model_receipt != {"model": name, "revision": revision, "model_files_sha256": hashes}:
            raise ValueError("cached model files differ from original receipt")
        validate_model_file_hashes(hashes)
        tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=False)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.pad_token_id is None:
            raise ValueError("padding token unavailable")
        if base.sha(tokenizer.chat_template.encode()) != CHAT_TEMPLATE_SHA256:
            raise ValueError("pinned native chat template differs")
        original_contexts = []
        for index, job in enumerate(jobs):
            if index % 64 == 0: check_deadline()
            original_contexts.append(prepare_context(tokenizer, job))
        metadata = {"protocol": PROTOCOL, "model": name, "revision": revision,
            "versions": versions, "model_files_sha256": hashes, "dtype": "float32", "loaded_dtype": "bfloat16",
            "attention": "eager", "tf32": False, "seed": 1, "generation": False, "sampling": False,
            "chat_template_sha256": base.sha(tokenizer.chat_template.encode()), "assistant_prefix": ASSISTANT_PREFIX,
            "stage": package["stage"], "native_assistant_boundary": "one ASCII space before JSON action prefix",
            "public_input_sha256": expected_input_sha256, "action_token_ids": original_contexts[0]["option_token_ids"],
            "source_commit": source_commit, "batch_size": BATCH_SIZE, "cached": True,
            "source_files_sha256": source_hashes}
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
        evidence("model_config_validation.json", validate_model_config(model.config, contexts))
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
        production_diagnostic_seconds = 0.
        batch_times, production_records = [], []
        while len(rows) < len(expected):
            check_deadline()
            offset = len(rows)
            batch_jobs, batch_contexts = expected[offset:offset+BATCH_SIZE], contexts[offset:offset+BATCH_SIZE]
            tick = time.monotonic()
            outputs = forward(torch, model, tokenizer, batch_contexts, cached=True, batch_size=BATCH_SIZE)
            batch_seconds = time.monotonic()-tick
            batch = [{"schema_version": SCORE_SCHEMA, **{k:v for k,v in job.items() if k != "prompt"},
                **context, **action_statistics(output["logits"], job["allowed_actions"], job["option_source_ids"]),
                **{k:v for k,v in output.items() if k != "logits"},
                "legal_action_vocabulary_mass": math.fsum(math.exp(output["logits"]["ABCDE".index(label)]-output["vocabulary_logsumexp"]) for label in job["allowed_actions"]), "model_tag": tag}
                for job, context, output in zip(batch_jobs, batch_contexts, outputs)]
            with scores_path.open("ab") as stream:
                stream.write(b"".join(base.canonical(row) for row in batch)); stream.flush(); os.fsync(stream.fileno())
            rows.extend(batch); batch_times.append(batch_seconds)
            if offset in selected_offsets:
                production_diagnostic_started = time.monotonic()
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
                production_diagnostic_seconds += time.monotonic()-production_diagnostic_started
            if len(rows) == BENCHMARK_ROWS:
                projection = projection_with_validation(len(rows), len(expected), time.monotonic()-benchmark_started,
                    remaining(), batch_times, single_seconds,
                    production_diagnostic_seconds=production_diagnostic_seconds,
                    checkpoint_seconds=[r["seconds"] for r in checkpoint_timings])
                projection["checkpoint_timing_records"] = list(checkpoint_timings)
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

