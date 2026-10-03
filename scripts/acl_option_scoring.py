#!/usr/bin/env python3
"""Score the frozen choices-only menus, conditional on beginning an answer.

This additive diagnostic performs no generation and never reads evaluator data.
The four scores are next-token logits after the original chat prompt and a fixed
assistant prefix. They are not probabilities that the original model would
answer rather than abstain. Model dependencies are imported only on execution.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Callable

from scripts.jane_gpu_backend import PINNED_MODELS, validate_package
from scripts.jane_qwen_backend import _reject_constant, _unique_object

MODELS = {"qwen3b": "Qwen/Qwen2.5-3B-Instruct", "qwen7b": "Qwen/Qwen2.5-7B-Instruct"}
PUBLIC_SHA256 = "9db13301d928cd31dc54c97f0c5cfd88b9bc25774ceb56027c032c52e4afb043"
EXPECTED_JOBS = 10000
ASSISTANT_PREFIX = '{"answer":"'
PROTOCOL = "conditional_next_token_option_softmax_v1"
BENCHMARK_ROWS = 256
SAFETY_FACTOR = 1.5
SHUTDOWN_SECONDS = 20.0
BATCH_SINGLE_ATOL = 0.125
MAX_INPUT_TOKENS = 2048


def canonical(value: Any) -> bytes:
    """Serialize portable finite JSON, including the trailing newline."""
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                       separators=(",", ":")) + "\n").encode("utf-8")


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(raw: bytes) -> Any:
    return json.loads(raw, object_pairs_hook=_unique_object, parse_constant=_reject_constant)


def write_once(path: Path, value: Any) -> None:
    with path.open("xb") as stream:
        stream.write(canonical(value))
        stream.flush()
        os.fsync(stream.fileno())


def load_jobs(path: Path) -> list[dict[str, Any]]:
    """Read only the exact frozen public control package, never gold labels."""
    if path.is_symlink() or not path.is_file():
        raise ValueError("public input must be a regular file")
    raw = path.read_bytes()
    if sha(raw) != PUBLIC_SHA256:
        raise ValueError("frozen choices-only input hash mismatch")
    package = load_json(raw)
    jobs = validate_package(package, max_jobs=EXPECTED_JOBS)
    if (package["schema_version"] != "jane-choice-controls-v1"
            or package["evidence_scope"] != "scientific" or len(jobs) != EXPECTED_JOBS):
        raise ValueError("public choices-only schema, scope, or count mismatch")
    return jobs


def option_statistics(logits: list[float]) -> dict[str, Any]:
    """Compute stable conditional A-D softmax and explicit exact-tie evidence."""
    if len(logits) != 4 or any(isinstance(x, bool) or not isinstance(x, (int, float))
                               or not math.isfinite(x) for x in logits):
        raise ValueError("four finite numeric option logits required")
    highest = max(logits)
    weights = [math.exp(value - highest) for value in logits]
    total = math.fsum(weights)
    probabilities = [weight / total for weight in weights]
    ranked_logits = sorted(logits, reverse=True)
    ranked_probs = sorted(probabilities, reverse=True)
    tied = [label for label, value in zip("ABCD", logits) if value == highest]
    return {"raw_option_logits": dict(zip("ABCD", map(float, logits))),
            "conditional_option_probabilities": dict(zip("ABCD", probabilities)),
            "top_option_id": tied[0], "tied_top_option_ids": tied,
            "logit_margin": ranked_logits[0] - ranked_logits[1],
            "probability_margin": ranked_probs[0] - ranked_probs[1]}


def prepare_context(tokenizer: Any, job: dict[str, Any]) -> dict[str, Any]:
    """Prove all four labels are single-token extensions of the scored context."""
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": job["prompt"]}], tokenize=False,
        add_generation_prompt=True)
    scored = rendered + ASSISTANT_PREFIX
    original_ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
    scored_ids = tokenizer(scored, add_special_tokens=False)["input_ids"]
    if not original_ids or scored_ids[:len(original_ids)] != original_ids:
        raise ValueError("assistant prefix changes existing prompt tokenization")
    if len(scored_ids) <= len(original_ids) or len(scored_ids) > MAX_INPUT_TOKENS:
        raise ValueError("scored input empty, missing prefix, or over token limit")
    option_ids = {}
    for label in "ABCD":
        extended = tokenizer(scored + label, add_special_tokens=False)["input_ids"]
        if len(extended) != len(scored_ids) + 1 or extended[:-1] != scored_ids:
            raise ValueError("option is not a single-token extension of scored context")
        token_id = extended[-1]
        if tokenizer.decode([token_id], skip_special_tokens=False,
                            clean_up_tokenization_spaces=False) != label:
            raise ValueError("option token does not decode to its exact label")
        option_ids[label] = token_id
    if len(set(option_ids.values())) != 4:
        raise ValueError("option token IDs must be distinct")
    return {"rendered_prompt_sha256": sha(rendered.encode()),
            "scored_context_sha256": sha(scored.encode()),
            "input_token_ids": original_ids, "scored_input_token_ids": scored_ids,
            "option_token_ids": option_ids}


def budget_projection(processed: int, total: int, seconds: float,
                      remaining_seconds: float) -> dict[str, Any]:
    """Conservatively gate continuation using measured, persisted throughput."""
    if (type(processed) is not int or type(total) is not int or not 0 < processed <= total
            or not math.isfinite(seconds) or seconds <= 0
            or not math.isfinite(remaining_seconds)):
        raise ValueError("invalid benchmark timing or coverage")
    projected = seconds / processed * (total - processed)
    return {"processed": processed, "seconds": seconds, "rows_per_second": processed / seconds,
            "projected_remaining_seconds": projected, "safety_factor": SAFETY_FACTOR,
            "remaining_seconds": remaining_seconds, "shutdown_reserve_seconds": SHUTDOWN_SECONDS,
            "proceed": projected * SAFETY_FACTOR + SHUTDOWN_SECONDS < remaining_seconds}


def validate_score_coverage(jobs: list[dict[str, Any]], rows: list[dict[str, Any]],
                            *, require_complete: bool = True) -> None:
    """Require a unique, ordered, hash-bound prefix, or the entire input corpus."""
    if len(rows) > len(jobs) or (require_complete and len(rows) != len(jobs)):
        raise ValueError("score coverage differs from public jobs")
    seen = set()
    for index, row in enumerate(rows):
        job = jobs[index]
        if (row.get("job_index") != index or row.get("job_id") != job["job_id"]
                or row.get("prompt_sha256") != job["prompt_sha256"]
                or row["job_id"] in seen):
            raise ValueError("score identity, order, uniqueness, or prompt hash mismatch")
        seen.add(row["job_id"])


def validate_batch_agreement(batched: list[list[float]], single: list[list[float]]) -> dict[str, Any]:
    """Reject padding/index errors and material numerical changes before benchmark."""
    if not batched or len(batched) != len(single):
        raise ValueError("batch/single validation cardinality mismatch")
    differences = []
    for left, right in zip(batched, single):
        left_stat, right_stat = option_statistics(left), option_statistics(right)
        if left_stat["tied_top_option_ids"] != right_stat["tied_top_option_ids"]:
            raise ValueError("batch/single top-option disagreement")
        differences.extend(abs(a - b) for a, b in zip(left, right))
    largest = max(differences)
    if largest > BATCH_SINGLE_ATOL:
        raise ValueError("batch/single logits exceed absolute tolerance")
    return {"rows": len(batched), "absolute_tolerance": BATCH_SINGLE_ATOL,
            "maximum_absolute_difference": largest, "exact_top_option_sets_match": True}


def _forward(torch: Any, model: Any, tokenizer: Any,
             contexts: list[dict[str, Any]]) -> list[list[float]]:
    # Use explicit position IDs: otherwise left-padding changes positions under
    # direct forward(), unlike generation's prepare_inputs_for_generation path.
    encoded = tokenizer.pad({"input_ids": [x["scored_input_token_ids"] for x in contexts]},
                            padding=True, return_tensors="pt")
    encoded = {key: value.to("cuda:0") for key, value in encoded.items()}
    position_ids = encoded["attention_mask"].long().cumsum(-1) - 1
    position_ids.masked_fill_(encoded["attention_mask"] == 0, 0)
    with torch.inference_mode():
        output = model(**encoded, position_ids=position_ids, use_cache=False,
                       logits_to_keep=1, return_dict=True)
        if tuple(output.logits.shape[:2]) != (len(contexts), 1):
            raise ValueError("model did not return exactly one last-position logit vector")
        ids = torch.tensor([list(x["option_token_ids"].values()) for x in contexts],
                           device="cuda:0", dtype=torch.long)
        selected = output.logits[:, 0, :].gather(1, ids).float().cpu().tolist()
    del output
    torch.cuda.synchronize()
    for values in selected:
        option_statistics(values)  # Fail before writing any non-finite row.
    return selected


def run_scoring(tag: str, jobs_path: Path, cache_dir: Path, out_dir: Path,
                max_seconds: float = 1650, batch_size: int = 32,
                progress: Callable[[dict[str, Any]], None] | None = None) -> dict[str, Any]:
    """Run a create-once, bounded benchmark and optionally complete all menus.

    Parameters
    ----------
    tag : str
        ``qwen3b`` or ``qwen7b`` with its original frozen Hugging Face revision.
    jobs_path : Path
        Exact public 10,000-control JSON file. No evaluator files are opened.
    cache_dir : Path
        Prepopulated Hugging Face cache; downloads are forbidden here. The required
        ``<tag>_expected_model_hashes.json`` binds its files to the original run.
        Its schema is {"model": ..., "revision": ..., "model_files_sha256": ...}.
    out_dir : Path
        New result directory. Existing results are never overwritten or replayed.
    max_seconds : float
        Internal elapsed deadline, including imports, hashes, load and warmup.
        Caller must also provide an external GPU timeout for blocking operations.
    batch_size : int
        Scoring batch size from one through 64; no precision fallback is used.
    progress : callable, optional
        Checkpoint hook after closed files at warmup, benchmark, and every 512
        completed rows. A provider wrapper can commit its persistent volume.

    Returns
    -------
    dict
        Durable receipt; complete, budget-stop, deadline-stop, or failed status.
    """
    if tag not in MODELS:
        raise ValueError("unknown pinned model tag")
    if (isinstance(max_seconds, bool) or not isinstance(max_seconds, (int, float))
            or not math.isfinite(max_seconds) or not 0 < max_seconds <= 1650):
        raise ValueError("max_seconds must be finite, positive and at most 1650")
    if type(batch_size) is not int or not 1 <= batch_size <= 64:
        raise ValueError("batch_size must be an integer from one through 64")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    started_at = datetime.now(timezone.utc).isoformat()
    model_name, revision = MODELS[tag], PINNED_MODELS[MODELS[tag]]
    rows: list[dict[str, Any]] = []
    metadata_path = out_dir / "metadata.json"
    scores_path = out_dir / "scores.jsonl"
    receipt: dict[str, Any] = {"schema_version": "acl-option-scoring-receipt-v1",
        "model_tag": tag, "model": model_name, "revision": revision,
        "started_at": started_at, "max_seconds": max_seconds,
        "expected_rows": EXPECTED_JOBS, "automatic_retries": 0,
        "protocol": PROTOCOL, "public_file_sha256": PUBLIC_SHA256}
    write_once(out_dir / "started.json", receipt)

    def remaining() -> float:
        return max_seconds - (time.monotonic() - started)

    def check_deadline() -> None:
        if remaining() <= SHUTDOWN_SECONDS:
            raise TimeoutError("internal scoring deadline reached")

    try:
        jobs = load_jobs(Path(jobs_path))
        check_deadline()
        workspace = os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        if workspace not in {":4096:8", ":16:8"}:
            raise ValueError("unsupported deterministic CUBLAS workspace")
        import torch
        import transformers
        from huggingface_hub import snapshot_download
        from transformers import AutoModelForCausalLM, AutoTokenizer

        expected_versions = {"torch": "2.6.0", "transformers": "4.51.3",
                             "tokenizers": "0.21.1", "safetensors": "0.5.3",
                             "huggingface-hub": "0.30.2"}
        versions = {name: importlib.metadata.version(name) for name in expected_versions}
        for name, expected in expected_versions.items():
            if versions[name].split("+")[0] != expected:
                raise ValueError(f"{name} differs from the original pinned stack")
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("exactly one CUDA GPU required")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 GPU required; no dtype fallback")
        torch.set_num_threads(2)
        torch.manual_seed(1)
        torch.cuda.manual_seed_all(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        snapshot = Path(snapshot_download(repo_id=model_name, revision=revision,
            cache_dir=cache_dir, local_files_only=True,
            allow_patterns=["*.json", "*.safetensors", "*.txt"]))
        if snapshot.name != revision:
            raise ValueError("cached model snapshot differs from pinned revision")
        model_hashes = {}
        hash_started = time.monotonic()
        for path in sorted(snapshot.rglob("*")):
            if path.is_file() and ".cache" not in path.relative_to(snapshot).parts:
                check_deadline()
                model_hashes[str(path.relative_to(snapshot))] = file_hash(path)
        if not model_hashes or not any(name.endswith(".safetensors") for name in model_hashes):
            raise ValueError("snapshot is missing model weights")
        expected_hash_path = Path(cache_dir) / f"{tag}_expected_model_hashes.json"
        if not expected_hash_path.is_file():
            raise ValueError("original run model hash receipt is required")
        expected = load_json(expected_hash_path.read_bytes())
        if (expected.get("model") != model_name or expected.get("revision") != revision
                or expected.get("model_files_sha256") != model_hashes):
            raise ValueError("cached model files differ from original run hash receipt")
        bound_original = True
        hash_seconds = time.monotonic() - hash_started
        check_deadline()
        tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True,
                                                  trust_remote_code=False)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise ValueError("tokenizer has no usable padding token")
            tokenizer.pad_token = tokenizer.eos_token
        warm_contexts = [prepare_context(tokenizer, job) for job in jobs[:3]]
        metadata = {"schema_version": "acl-option-scoring-metadata-v1", "model_tag": tag,
            "model": model_name, "revision": revision, "public_file_sha256": PUBLIC_SHA256,
            "n_jobs": len(jobs), "assistant_prefix": ASSISTANT_PREFIX, "protocol": PROTOCOL,
            "model_files_sha256": model_hashes, "model_hashes_bound_to_original_run": bound_original,
            "expected_model_hashes_receipt_sha256": file_hash(expected_hash_path) if bound_original else None,
            "chat_template_sha256": sha(tokenizer.chat_template.encode()),
            "source_sha256": file_hash(Path(__file__)), "versions": versions,
            "dtype": "bfloat16", "attention_implementation": "eager", "seed": 1,
            "deterministic_algorithms": True, "tf32": False, "batch_size": batch_size,
            "max_input_tokens": MAX_INPUT_TOKENS, "use_cache": False, "logits_to_keep": 1,
            "generation": False, "quantization": False, "gpu": torch.cuda.get_device_name(0),
            "torch_cuda_version": torch.version.cuda, "model_hash_seconds": hash_seconds,
            "option_token_ids": warm_contexts[0]["option_token_ids"],
            "tie_policy": "first A/B/C/D among exact equal maxima; ties retained explicitly",
            "position_ids": "attention_mask.cumsum(-1)-1; pad positions zero",
            "scope": "exploratory preference conditional on fixed answer prefix; no abstention score"}
        write_once(metadata_path, metadata)
        load_started = time.monotonic()
        model = AutoModelForCausalLM.from_pretrained(str(snapshot), local_files_only=True,
            trust_remote_code=False, use_safetensors=True, torch_dtype=torch.bfloat16,
            attn_implementation="eager").to("cuda:0").eval()
        torch.cuda.synchronize()
        receipt["model_load_seconds"] = time.monotonic() - load_started
        check_deadline()
        warm_started = time.monotonic()
        batched = _forward(torch, model, tokenizer, warm_contexts)
        single = [_forward(torch, model, tokenizer, [context])[0] for context in warm_contexts]
        agreement = validate_batch_agreement(batched, single)
        agreement["seconds"] = time.monotonic() - warm_started
        agreement["job_ids"] = [job["job_id"] for job in jobs[:3]]
        agreement["batched_logits"] = batched
        agreement["single_logits"] = single
        write_once(out_dir / "warmup.json", agreement)
        receipt["warmup_seconds"] = agreement["seconds"]
        receipt["ready_seconds"] = time.monotonic() - started
        benchmark_started = time.monotonic()
        benchmark = None
        last_batch_seconds = 0.0
        scores_path.touch(exist_ok=False)
        (out_dir / "batch_checks.jsonl").touch(exist_ok=False)
        if progress is not None:
            progress({"phase": "warmup", "completed_rows": 0,
                      "elapsed_seconds": time.monotonic() - started})
        next_checkpoint = 512
        while len(rows) < len(jobs):
            check_deadline()
            if last_batch_seconds * SAFETY_FACTOR + SHUTDOWN_SECONDS >= remaining():
                receipt["status"] = "deadline_stop"
                break
            offset = len(rows)
            phase_end = BENCHMARK_ROWS if offset < BENCHMARK_ROWS else len(jobs)
            batch_jobs = jobs[offset:min(offset + batch_size, phase_end)]
            batch_started = time.monotonic()
            contexts = [prepare_context(tokenizer, job) for job in batch_jobs]
            if any(x["option_token_ids"] != metadata["option_token_ids"] for x in contexts):
                raise ValueError("option token IDs vary across contexts")
            scores = _forward(torch, model, tokenizer, contexts)
            batch_rows = []
            batch_index = (len(rows) + batch_size - 1) // batch_size
            for j, (job, context, logits) in enumerate(zip(batch_jobs, contexts, scores)):
                row = {"schema_version": "acl-option-scores-v1", "job_index": offset + j,
                    **{k: job[k] for k in ("job_id", "qid", "group_id", "split", "condition",
                                           "menu_id", "prompt_sha256")},
                    **context, **option_statistics(logits), "batch_index": batch_index}
                batch_rows.append(row)
            with scores_path.open("ab") as output:
                output.write(b"".join(canonical(row) for row in batch_rows))
                output.flush()
                os.fsync(output.fileno())
            rows.extend(batch_rows)
            last_batch_seconds = time.monotonic() - batch_started
            check = {"batch_index": batch_index, "start": offset, "stop": len(rows),
                "rows_sha256": sha(b"".join(canonical(row) for row in batch_rows)),
                "seconds": last_batch_seconds, "elapsed_seconds": time.monotonic() - started}
            with (out_dir / "batch_checks.jsonl").open("ab") as checks:
                checks.write(canonical(check))
                checks.flush()
                os.fsync(checks.fileno())
            if len(rows) == BENCHMARK_ROWS:
                benchmark = budget_projection(len(rows), len(jobs),
                    time.monotonic() - benchmark_started, remaining())
                write_once(out_dir / "benchmark.json", benchmark)
                receipt["benchmark"] = benchmark
                if progress is not None:
                    progress({"phase": "benchmark", "completed_rows": len(rows), **benchmark})
                if not benchmark["proceed"]:
                    receipt["status"] = "benchmark_budget_stop"
                    break
            if progress is not None and len(rows) >= next_checkpoint:
                progress({"phase": "scoring", "completed_rows": len(rows),
                          "elapsed_seconds": time.monotonic() - started})
                next_checkpoint = len(rows) + 512
        if len(rows) == len(jobs):
            validate_score_coverage(jobs, rows)
            receipt["status"] = "complete"
        else:
            validate_score_coverage(jobs, rows, require_complete=False)
    except TimeoutError as error:
        receipt["status"] = "deadline_stop"
        receipt["error"] = f"{type(error).__name__}: {error}"
    except Exception as error:
        receipt["status"] = "failed"
        receipt["error"] = f"{type(error).__name__}: {error}"
    receipt.update({"completed_rows": len(rows), "finished_at": datetime.now(timezone.utc).isoformat(),
        "total_seconds": time.monotonic() - started,
        "scores_sha256": file_hash(scores_path) if scores_path.exists() else None,
        "metadata_sha256": file_hash(metadata_path) if metadata_path.exists() else None})
    write_once(out_dir / "receipt.json", receipt)
    return receipt
