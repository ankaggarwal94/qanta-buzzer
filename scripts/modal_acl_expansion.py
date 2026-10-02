#!/usr/bin/env python3
"""Detached, budget-bounded execution of the frozen ACL 5,000-question inputs.

Import and plan are CPU/read-only. Launch allocates at most two L40S workers,
with no automatic retries. Resume reuses verified complete shards and refuses
started incomplete shards: lost physical generations are never silently replayed.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import uuid
from typing import Any

from scripts.jane_gpu_backend import (PINNED_MODELS, GPUConfig, validate_package,
    validate_prediction_coverage, interface_diagnostics, CONFIDENCE_METHOD)
from scripts.jane_qwen_backend import _unique_object, _reject_constant

MODELS = (("qwen3b", "Qwen/Qwen2.5-3B-Instruct"), ("qwen7b", "Qwen/Qwen2.5-7B-Instruct"))
PUBLIC_FILES = {
    "main_jobs.json": (150000, "9bfeaf2d86116ced0e8c55c3c390d3be12050ad38820b4e02c6ed684cc8786bf"),
    "main_choices_only.json": (10000, "9db13301d928cd31dc54c97f0c5cfd88b9bc25774ceb56027c032c52e4afb043"),
}
SOURCE_FILES = ("scripts/__init__.py", "scripts/modal_acl_expansion.py", "scripts/jane_gpu_backend.py",
                "scripts/jane_qwen_backend.py", "scripts/jane_output_constraints.py")
ALLOCATION_RATE = Decimal("0.00063924")
MAX_CEILING = Decimal("80")
RESERVE_USD = Decimal("6")
MAX_MODEL_SECONDS = 57600
SHUTDOWN_SECONDS = 120
MAX_DOWNLOAD_BYTES = 3 * 1024**3
MAX_DOWNLOAD_FILE_BYTES = 64 * 1024**2
MODAL_VERSION = "1.6.0"
GENERATION = {"batch_size": 8, "max_input_tokens": 2048, "max_new_tokens": 160,
              "seed": 1, "threads": 4, "greedy": True, "dtype": "bfloat16",
              "constraint_failure_policy": "retain_invalid_at_token_cap_v1"}


def canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                       separators=(",", ":")) + "\n").encode()


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def load(raw: bytes | str) -> Any:
    return json.loads(raw, object_pairs_hook=_unique_object, parse_constant=_reject_constant)


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024**2), b""):
            h.update(block)
    return h.hexdigest()


def write_once(path: Path, value: Any) -> None:
    """Durably create an immutable JSON evidence file, never overwrite it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as f:
        f.write(canonical(value)); f.flush(); os.fsync(f.fileno())


def replace_progress(path: Path, value: Any) -> None:
    """Replace only the explicitly mutable, non-authoritative progress snapshot."""
    temp = path.with_suffix(".pending")
    with temp.open("wb") as f:
        f.write(canonical(value)); f.flush(); os.fsync(f.fileno())
    os.replace(temp, path)


def budget_plan(ceiling_usd: str, model_seconds: int, previous: list[dict] | None = None,
                additional_models: int = 2) -> dict:
    """Reserve worst-case durations, including failed/uncertain past allocations."""
    ceiling = Decimal(str(ceiling_usd))
    if not ceiling.is_finite() or not RESERVE_USD < ceiling <= MAX_CEILING:
        raise ValueError("fresh expansion ceiling must exceed $6 and be at most $80")
    if type(model_seconds) is not int or not 300 <= model_seconds <= MAX_MODEL_SECONDS:
        raise ValueError("model timeout must be an integer from 300 through 57600 seconds")
    if type(additional_models) is not int or additional_models not in {1, 2}:
        raise ValueError("reserve one or two pinned model allocations")
    previous = previous or []
    prior_seconds = 0
    for index, item in enumerate(previous):
        if item.get("reservation_index") != index or item.get("model_tag") not in dict(MODELS):
            raise ValueError("allocation reservation ledger is not a complete ordered prefix")
        seconds = item.get("max_model_seconds")
        if type(seconds) is not int or not 300 <= seconds <= MAX_MODEL_SECONDS:
            raise ValueError("invalid earlier allocation duration")
        prior_seconds += seconds
    maximum = RESERVE_USD + ALLOCATION_RATE * (prior_seconds + additional_models * model_seconds)
    if maximum > ceiling:
        raise ValueError("new allocation reservation exceeds the cumulative expansion ceiling")
    return {"schema_version": "acl-expansion-budget-v1", "ceiling_usd": str(ceiling),
            "reserve_usd": str(RESERVE_USD), "allocation_rate_usd_per_second": str(ALLOCATION_RATE),
            "max_model_seconds": model_seconds, "additional_models": additional_models,
            "previous_reserved_seconds": prior_seconds, "maximum_reserved_estimate_usd": str(maximum),
            "pricing_url": "https://modal.com/pricing", "pricing_checked": "2026-10-02",
            "invoice_verified": False, "prior_reservations": previous,
            "refund_policy": "none; each attempt retains its full worst-case reservation"}


def source_identity(repo: Path) -> dict:
    result = {}
    for name in SOURCE_FILES:
        path = repo / name
        if path.is_symlink() or not path.is_file():
            raise ValueError("inference source must be an allowlisted regular file")
        result[name] = file_hash(path)
    return result


def verify_source_commit(repo: Path, commit: str | None, hashes: dict) -> None:
    if commit is None:
        return
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("source commit must be an exact 40-character SHA")
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, check=True,
                          capture_output=True).stdout.decode().strip()
    if head != commit:
        raise ValueError("source checkout differs from the specified commit")
    for name, expected in hashes.items():
        raw = subprocess.run(["git", "show", f"{commit}:{name}"], cwd=repo,
                             check=True, capture_output=True).stdout
        if sha(raw) != expected:
            raise ValueError("inference source differs from its committed bytes")


def load_public_inputs(public_dir: Path) -> tuple[dict, dict]:
    """Only the two frozen public files can enter the cloud workload."""
    packages, identities = {}, {}
    for name, (count, expected_hash) in PUBLIC_FILES.items():
        path = public_dir / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 512 * 1024**2:
            raise ValueError("public input missing, nonregular, or oversized")
        if file_hash(path) != expected_hash:
            raise ValueError(f"frozen public input hash mismatch: {name}")
        package = load(path.read_bytes())
        jobs = validate_package(package, max_jobs=count)
        schema = "jane-choice-controls-v1" if "choices" in name else "jane-public-jobs-v1"
        if len(jobs) != count or package["schema_version"] != schema or package["evidence_scope"] != "scientific":
            raise ValueError("public input count/schema/scope mismatch")
        packages[name] = package
        identities[name] = {"sha256": expected_hash, "bytes": path.stat().st_size, "jobs": count}
    return packages, identities


def shards_for(package: dict, model_tag: str, role: str, shard_size: int) -> list[dict]:
    """Use contiguous batch-eight-aligned slices; request order never changes."""
    if type(shard_size) is not int or shard_size < 8 or shard_size > 16384 or shard_size % 8:
        raise ValueError("shard size must be a multiple of eight, between 8 and 16384")
    model = dict(MODELS)[model_tag]
    shards = []
    for start in range(0, len(package["jobs"]), shard_size):
        jobs = package["jobs"][start:start + shard_size]
        identity = {"model": model, "revision": PINNED_MODELS[model], "role": role,
                    "start": start, "stop": start + len(jobs), "generation": GENERATION,
                    "job_ids": [j["job_id"] for j in jobs]}
        shards.append({"shard_id": sha(canonical(identity)), "start": start,
                       "stop": start + len(jobs), "role": role, "model_tag": model_tag,
                       "package": {**package, "jobs": jobs}})
    return shards


def build_plan(public_dir: Path, repo: Path, run_id: str, budget_usd: str,
               max_model_seconds: int, shard_size: int, source_commit: str | None = None) -> dict:
    if not re.fullmatch(r"acl[a-z0-9-]{3,40}", run_id):
        raise ValueError("run ID must start acl and use 3-40 lowercase letters, digits or hyphens")
    packages, inputs = load_public_inputs(public_dir)
    sources = source_identity(repo)
    verify_source_commit(repo, source_commit, sources)
    inventory = []
    for tag, model in MODELS:
        for name, role in (("main_jobs.json", "main"), ("main_choices_only.json", "choices_only")):
            inventory.extend({k: v for k, v in s.items() if k != "package"}
                             for s in shards_for(packages[name], tag, role, shard_size))
    plan = {"schema_version": "acl-expansion-run-v1", "run_id": run_id,
            "public_files": inputs, "source_files_sha256": sources, "source_commit": source_commit,
            "models": [{"tag": t, "model": m, "revision": PINNED_MODELS[m]} for t, m in MODELS],
            "generation": GENERATION, "shard_size": shard_size, "shards": inventory,
            "budget": budget_plan(budget_usd, max_model_seconds), "max_concurrent_gpus": 2,
            "planned_responses": 320000, "automatic_retries": 0,
            "resume_policy": "reuse verified complete shards; block any started incomplete shard"}
    plan["run_identity"] = sha(canonical(plan))
    return plan


def validate_plan(plan: dict) -> None:
    value = {k: v for k, v in plan.items() if k != "run_identity"}
    if sha(canonical(value)) != plan.get("run_identity"):
        raise ValueError("run control identity mismatch")
    if plan.get("generation") != GENERATION or plan.get("models") != [
            {"tag": t, "model": m, "revision": PINNED_MODELS[m]} for t, m in MODELS]:
        raise ValueError("generation settings or model pins differ")
    for name, (count, digest) in PUBLIC_FILES.items():
        identity = plan["public_files"][name]
        if identity["sha256"] != digest or identity["jobs"] != count:
            raise ValueError("run uses a different public input snapshot")
    if plan["budget"] != budget_plan(plan["budget"]["ceiling_usd"], plan["budget"]["max_model_seconds"]):
        raise ValueError("initial budget is not the bounded two-model reservation")


def verify_trace(shard: dict, trace: dict, source_hashes: dict) -> None:
    """Require actual generation, exact coverage, raw evidence and batch membership."""
    jobs = shard["package"]["jobs"]
    validate_prediction_coverage(jobs, trace.get("predictions", []))
    diagnostics = interface_diagnostics(jobs, trace)
    if any(row["illegal_option_id"] for row in diagnostics.values()):
        raise ValueError("MC output contains an illegal option ID")
    metadata = trace.get("metadata", {})
    model = dict(MODELS)[shard["model_tag"]]
    expected_schema = "jane-choice-control-traces-v1" if shard["role"] == "choices_only" else "jane-traces-v1"
    if trace.get("schema_version") != expected_schema:
        raise ValueError("wrong shard trace schema")
    for key, expected in {"model": model, "revision": PINNED_MODELS[model],
                          "execution": "actual_cuda_model_generation", "context_policy": "fresh_per_prefix",
                          "evidence_scope": "scientific", "batch_size": 8, "seed": 1,
                          "max_input_tokens": 2048, "max_new_tokens": 160, "greedy": True,
                          "dtype": "bfloat16", "n_jobs": len(jobs),
                          "confidence_method": CONFIDENCE_METHOD,
                          "backend_sha256": source_hashes["scripts/jane_gpu_backend.py"],
                          "parser_sha256": source_hashes["scripts/jane_qwen_backend.py"],
                          "public_package_canonical_sha256": sha(canonical(shard["package"])[:-1])}.items():
        if metadata.get(key) != expected:
            raise ValueError(f"trace metadata mismatch: {key}")
    if metadata.get("output_constraints", {}).get("source_sha256") != source_hashes["scripts/jane_output_constraints.py"]:
        raise ValueError("trace constraint source mismatch")
    for index, (job, row) in enumerate(zip(jobs, trace["predictions"], strict=True)):
        expected_batch = [j["job_id"] for j in jobs[index // 8 * 8:index // 8 * 8 + 8]]
        if (row["job_id"] != job["job_id"] or row.get("batch_job_ids") != expected_batch
                or row.get("batch_index") != index // 8 or not isinstance(row.get("raw_response"), str)
                or not isinstance(row.get("input_token_ids"), list)
                or not isinstance(row.get("generated_token_ids"), list)):
            raise ValueError("raw trace order or batch membership differs from the frozen design")
        if row.get("constraint_failure_kind") not in {None, "max_new_tokens_incomplete_json"}:
            raise ValueError("unhandled constraint failure requires review")
        if (not 0 < len(row["input_token_ids"]) <= 2048 or not 0 < len(row["generated_token_ids"]) <= 160
                or row.get("input_tokens") != len(row["input_token_ids"])
                or row.get("output_tokens") != len(row["generated_token_ids"])):
            raise ValueError("raw token evidence exceeds or differs from the frozen token caps")
        if row.get("status") == "invalid" and row.get("constraint_failure_kind") != "max_new_tokens_incomplete_json":
            raise ValueError("unrecognized invalid completion")


def completed_shard(directory: Path, shard: dict, source_hashes: dict) -> dict | None:
    """Missing completion after a started claim is ambiguous and cannot be replayed."""
    start, completion, trace_path = directory / "started.json", directory / "completion.json", directory / "trace.json"
    if not start.exists():
        if directory.exists() and any(directory.iterdir()):
            raise ValueError("unclaimed shard contains unexpected evidence")
        return None
    claim = load(start.read_bytes())
    if claim.get("shard_id") != shard["shard_id"] or claim.get("source_files_sha256") != source_hashes:
        raise ValueError("shard claim identity/source mismatch")
    if not completion.is_file() or not trace_path.is_file():
        raise RuntimeError(f"AMBIGUOUS_INCOMPLETE_SHARD:{shard['shard_id']}")
    receipt = load(completion.read_bytes())
    if receipt.get("trace_sha256") != file_hash(trace_path) or receipt.get("shard_id") != shard["shard_id"]:
        raise ValueError("completed shard trace hash mismatch")
    trace = load(trace_path.read_bytes())
    verify_trace(shard, trace, source_hashes)
    return trace


class BatchCheckpoint:
    """Close checkpoint files before every Volume commit, including exceptions."""
    def __init__(self, path: Path):
        self.path, self.stream = path, None
        with path.open("x", encoding="utf-8"):
            pass

    def write(self, text):
        if self.stream is None:
            self.stream = self.path.open("a", encoding="utf-8")
        return self.stream.write(text)

    def flush(self):
        if self.stream is not None:
            self.stream.flush()

    def fileno(self):
        if self.stream is None:
            raise OSError("no open checkpoint batch")
        return self.stream.fileno()

    def close(self):
        if self.stream is not None:
            self.stream.flush(); os.fsync(self.stream.fileno()); self.stream.close(); self.stream = None


@contextmanager
def deadline(seconds: float):
    if seconds <= 0:
        raise TimeoutError("allocation deadline expired")
    prior = signal.getsignal(signal.SIGALRM)
    def expire(*_args):
        raise TimeoutError("allocation deadline expired")
    signal.signal(signal.SIGALRM, expire); signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0); signal.signal(signal.SIGALRM, prior)


def execute_model(plan: dict, allocation: dict, root: Path, repo: Path, commit,
                  generate=None, clock=time.monotonic) -> dict:
    """Run a single reserved model sequentially; injectable backend supports CPU tests."""
    from scripts.jane_gpu_backend import run
    generate = generate or run
    validate_plan(plan)
    if source_identity(repo) != plan["source_files_sha256"]:
        raise ValueError("executing source differs from the frozen run")
    packages, identities = load_public_inputs(root / "public")
    if identities != plan["public_files"]:
        raise ValueError("mounted inputs differ from launch control")
    allocations = load((root / "allocations" / "initial.json").read_bytes())
    for path in sorted((root / "allocations").glob("[0-9][0-9][0-9][0-9].json")):
        allocations.append(load(path.read_bytes()))
    validate_allocations(plan, allocations)
    index = allocation.get("reservation_index")
    if type(index) is not int or index < 0 or index >= len(allocations) or allocations[index] != allocation:
        raise ValueError("worker allocation differs from the committed reservation ledger")
    tag = allocation["model_tag"]
    model = dict(MODELS)[tag]
    out = root / "output" / tag
    attempt = out / "attempts" / f"{allocation['reservation_index']:04d}"
    attempt.mkdir(parents=True, exist_ok=False)
    start_clock = clock()
    write_once(attempt / "started.json", {"allocation": allocation, "run_identity": plan["run_identity"],
                                         "started_utc": datetime.now(timezone.utc).isoformat()})
    commit()
    complete, observed_jobs, observed_seconds, status, failure = 0, 0, 0.0, "FAILED", None
    def remaining():
        return allocation["max_model_seconds"] - SHUTDOWN_SECONDS - (clock() - start_clock)
    try:
        with deadline(remaining()):
            for name, role in (("main_jobs.json", "main"), ("main_choices_only.json", "choices_only")):
                for shard in shards_for(packages[name], tag, role, plan["shard_size"]):
                    directory = out / role / shard["shard_id"]
                    old = completed_shard(directory, shard, plan["source_files_sha256"])
                    if old is not None:
                        complete += len(shard["package"]["jobs"])
                        continue
                    estimate = (observed_seconds / observed_jobs * len(shard["package"]["jobs"]) * 1.10
                                if observed_jobs else 300)
                    if remaining() < estimate + SHUTDOWN_SECONDS:
                        status = "BUDGET_PAUSED_AT_SHARD_BOUNDARY"
                        break
                    directory.mkdir(parents=True, exist_ok=False)
                    write_once(directory / "started.json", {"shard_id": shard["shard_id"],
                               "start": shard["start"], "stop": shard["stop"],
                               "run_identity": plan["run_identity"], "allocation": allocation,
                               "source_files_sha256": plan["source_files_sha256"]})
                    commit()  # Durable claim precedes any model loading or physical generation.
                    checkpoint = BatchCheckpoint(directory / "predictions.checkpoint.jsonl")
                    def progress(update):
                        checkpoint.close()
                        replace_progress(out / "progress.json", {"model_tag": tag, "role": role,
                            "reservation_index": allocation["reservation_index"],
                            "shard_id": shard["shard_id"], "completed_shard_jobs": complete,
                            "current_shard_completed_jobs": update["completed_jobs"],
                            "observed_completed_jobs": complete + update["completed_jobs"],
                            "planned_model_jobs": 160000, "elapsed_seconds": clock() - start_clock,
                            "remaining_seconds": remaining(), "updated_utc": datetime.now(timezone.utc).isoformat()})
                        commit()
                    tick = clock()
                    try:
                        trace = generate(shard["package"], GPUConfig(model=model, revision=PINNED_MODELS[model],
                            batch_size=8, max_jobs=plan["shard_size"], max_input_tokens=2048, max_new_tokens=160,
                            max_elapsed_seconds=remaining(), seed=1, threads=4,
                            cache_dir=Path("/tmp/acl-expansion-models"), allow_download=True),
                            checkpoint=checkpoint, progress=progress)
                    finally:
                        checkpoint.close(); commit()  # Also handles pre-progress constraint errors.
                    verify_trace(shard, trace, plan["source_files_sha256"])
                    write_once(directory / "trace.json", trace)
                    write_once(directory / "completion.json", {"shard_id": shard["shard_id"],
                               "trace_sha256": file_hash(directory / "trace.json"),
                               "jobs": len(trace["predictions"]), "wall_seconds": clock() - tick})
                    commit()
                    complete += len(trace["predictions"])
                    observed_jobs += len(trace["predictions"]); observed_seconds += clock() - tick
                    prediction = (160000 - complete) * observed_seconds / observed_jobs * 1.10
                    replace_progress(out / "progress.json", {"model_tag": tag, "observed_completed_jobs": complete,
                        "planned_model_jobs": 160000, "elapsed_seconds": clock() - start_clock,
                        "remaining_seconds": remaining(), "predicted_remaining_seconds_with_10pct_margin": prediction,
                        "completion_predicted_within_remaining_budget": prediction <= remaining(),
                        "updated_utc": datetime.now(timezone.utc).isoformat()})
                    if observed_jobs == len(trace["predictions"]):
                        write_once(attempt / "first_shard_throughput.json", {
                            "jobs": observed_jobs, "wall_seconds": observed_seconds,
                            "predicted_remaining_seconds_with_10pct_margin": prediction,
                            "remaining_seconds": remaining(),
                            "completion_predicted_within_remaining_budget": prediction <= remaining(),
                            "policy": "continue bounded work; pause before a predicted non-fitting next shard"})
                    commit()
                if status == "BUDGET_PAUSED_AT_SHARD_BOUNDARY":
                    break
            if complete == 160000:
                status = "COMPLETED"
    except Exception as error:
        failure = type(error).__name__
        status = "FAILED_REQUIRES_REVIEW"
    receipt = {"status": status, "model_tag": tag, "completed_jobs": complete,
               "expected_jobs": 160000, "exception_type": failure, "allocation": allocation,
               "elapsed_seconds": clock() - start_clock, "run_identity": plan["run_identity"],
               "allocation_estimate_usd": str(ALLOCATION_RATE * Decimal(str(clock() - start_clock))),
               "actual_invoice_verified": False, "automatic_retries": 0}
    write_once(attempt / "receipt.json", receipt); commit()
    return receipt


def remote_model(plan: dict, allocation: dict) -> dict:
    import modal
    volume = modal.Volume.from_name(plan["run_id"], create_if_missing=False)
    volume.reload()
    return execute_model(plan, allocation, Path("/acl"), Path("/opt/acl"), volume.commit)


def connect():
    """Check the pinned SDK and expected account without exposing credentials."""
    try:
        import truststore
        truststore.inject_into_ssl()
    except ImportError:
        pass
    import modal
    if importlib.metadata.version("modal") != MODAL_VERSION:
        raise ValueError("Modal 1.6.0 is required")
    from modal._utils.async_utils import synchronize_api
    from modal.client import _Client
    from modal_proto import api_pb2
    async def check():
        client = await _Client.from_env()
        response = await client._stub.TokenInfoGet(api_pb2.TokenInfoGetRequest())
        if response.workspace_name != "ankaggarwal94":
            raise ValueError("unexpected Modal workspace")
        return {"workspace_name": response.workspace_name, "workspace_id": response.workspace_id}
    return modal, synchronize_api(check)()


def remote_read(volume, name: str, maximum: int = 1024**2) -> bytes:
    result = bytearray()
    for chunk in volume.read_file(name):
        result.extend(chunk)
        if len(result) > maximum:
            raise ValueError("remote metadata exceeds bound")
    return bytes(result)


def create_image(modal, repo: Path):
    image = modal.Image.debian_slim(python_version="3.11").pip_install(
        "torch==2.6.0", "transformers==4.51.3", "tokenizers==0.21.1", "safetensors==0.5.3",
        "huggingface-hub==0.30.2", "accelerate==1.6.0", "modal==1.6.0",
        "lm-format-enforcer==0.11.3", "interegular==0.3.3")
    for name in SOURCE_FILES:
        image = image.add_local_file(str(repo / name), remote_path="/opt/acl/" + name, copy=True)
    return image.env({"PYTHONPATH": "/opt/acl", "PYTHONUNBUFFERED": "1",
                      "HF_HUB_DISABLE_TELEMETRY": "1", "TOKENIZERS_PARALLELISM": "false"})


def initial_allocations(plan: dict) -> list[dict]:
    return [{"reservation_index": i, "model_tag": tag, "max_model_seconds": plan["budget"]["max_model_seconds"],
             "run_identity": plan["run_identity"]} for i, (tag, _model) in enumerate(MODELS)]


def validate_allocations(plan: dict, allocations: list[dict]) -> None:
    if allocations[:2] != initial_allocations(plan):
        raise ValueError("initial reservation ledger differs from the frozen two-model budget")
    total = 0
    for index, allocation in enumerate(allocations):
        if (set(allocation) != {"reservation_index", "model_tag", "max_model_seconds", "run_identity"}
                or allocation["reservation_index"] != index or allocation["model_tag"] not in dict(MODELS)
                or allocation["run_identity"] != plan["run_identity"]
                or type(allocation["max_model_seconds"]) is not int
                or not 300 <= allocation["max_model_seconds"] <= MAX_MODEL_SECONDS):
            raise ValueError("invalid cumulative allocation reservation")
        total += allocation["max_model_seconds"]
    if RESERVE_USD + ALLOCATION_RATE * total > Decimal(plan["budget"]["ceiling_usd"]):
        raise ValueError("cumulative allocations exceed the frozen ceiling")


def spawn_detached(modal, plan: dict, allocations: list[dict], repo: Path, volume, out: Path) -> dict:
    from modal._utils.function_utils import FunctionSourceInfo
    if FunctionSourceInfo(remote_model).module_name != "scripts.modal_acl_expansion":
        raise ValueError("launch requires the python -m scripts.modal_acl_expansion entrypoint")
    image = create_image(modal, repo)
    app = modal.App(plan["run_id"] + "-execution", image=image, include_source=False)
    function = app.function(gpu="L40S", cpu=(2, 2), memory=(32768, 32768), volumes={"/acl": volume},
        max_containers=2, min_containers=0, buffer_containers=0, scaledown_window=2, retries=0,
        timeout=max(a["max_model_seconds"] for a in allocations), startup_timeout=90,
        include_source=False, secrets=[])(remote_model)
    calls = []
    with modal.enable_output():
        with app.run(detach=True):
            for allocation in allocations:
                call = function.spawn(plan, allocation)
                record = {"reservation_index": allocation["reservation_index"], "model_tag": allocation["model_tag"],
                          "function_call_id": call.object_id, "app_id": app.app_id,
                          "volume_name": plan["run_id"], "status": "SUBMITTED", "detached": True,
                          "automatic_retries": 0, "allocation": allocation}
                write_once(out / f"provider_launch_{allocation['reservation_index']:04d}.json", record)
                calls.append(record)
    result = {"status": "SUBMITTED", "completed": False, "run_identity": plan["run_identity"],
              "calls": calls, "budget": plan["budget"]}
    write_once(out / "launch_receipt.json", result)
    return result


def launch(plan: dict, public_dir: Path, repo: Path, out: Path) -> dict:
    """Create one provider-side run claim, freeze reservations, then spawn detached."""
    validate_plan(plan)
    if not plan.get("source_commit"):
        raise ValueError("launch requires an exact --source-commit")
    if source_identity(repo) != plan["source_files_sha256"]:
        raise ValueError("source changed after planning")
    verify_source_commit(repo, plan["source_commit"], plan["source_files_sha256"])
    _packages, inputs = load_public_inputs(public_dir)
    if inputs != plan["public_files"]:
        raise ValueError("inputs changed after planning")
    out.mkdir(parents=True, exist_ok=False)
    write_once(out / "frozen_control.json", plan)
    modal, realm = connect()
    write_once(out / "workspace_receipt.json", realm)
    modal.Volume.objects.create(plan["run_id"], version=2, allow_existing=False)
    volume = modal.Volume.from_name(plan["run_id"], create_if_missing=False)
    allocations = initial_allocations(plan)
    allocation_path = out / "allocations.json"
    write_once(allocation_path, allocations)
    write_once(out / "submission.json", {"status": "ALLOCATIONS_RESERVED", "run_identity": plan["run_identity"]})
    with volume.batch_upload() as upload:
        upload.put_file(str(out / "frozen_control.json"), "/control.json")
        upload.put_file(str(allocation_path), "/allocations/initial.json")
        upload.put_file(str(out / "submission.json"), "/output/submission.json")
        for name in PUBLIC_FILES:
            upload.put_file(str(public_dir / name), "/public/" + name)
    return spawn_detached(modal, plan, allocations, repo, volume, out)


def status(run_id: str) -> dict:
    modal, _realm = connect()
    volume = modal.Volume.from_name(run_id, create_if_missing=False)
    plan = load(remote_read(volume, "control.json")); validate_plan(plan)
    evidence = {}
    for entry in volume.iterdir("/output", recursive=True):
        path = str(entry.path).lstrip("/")
        if getattr(entry, "type", None) == 1 and (path.endswith("/receipt.json") or path.endswith("/progress.json")):
            evidence[path] = load(remote_read(volume, path))
    completed = {row["model_tag"] for path, row in evidence.items()
                 if path.endswith("/receipt.json") and row.get("status") == "COMPLETED"}
    return {"run_id": run_id, "run_identity": plan["run_identity"],
            "status": "COMPLETED" if completed == set(dict(MODELS)) else "INCOMPLETE_OR_RUNNING",
            "evidence": evidence, "budget": plan["budget"]}


def download(run_id: str, out: Path) -> dict:
    """Stream immutable complete-shard files; repeated downloads verify existing bytes."""
    modal, _realm = connect()
    volume = modal.Volume.from_name(run_id, create_if_missing=False)
    entries = [e for prefix in ("/output", "/allocations") for e in volume.iterdir(prefix, recursive=True)
               if getattr(e, "type", None) == 1]
    names = {str(e.path).lstrip("/") for e in entries}
    complete_dirs = {str(Path(n).parent) for n in names if n.endswith("/completion.json")}
    total, files = 0, []
    out.mkdir(parents=True, exist_ok=True)
    control = remote_read(volume, "control.json"); plan = load(control); validate_plan(plan)
    if not (out / "control.json").exists():
        write_once(out / "control.json", plan)
    elif load((out / "control.json").read_bytes()) != plan:
        raise ValueError("download destination belongs to another run")
    for name in sorted(names):
        if ".." in Path(name).parts or not name.startswith(("output/", "allocations/")):
            raise ValueError("unsafe remote output path")
        # Mutable progress and incomplete checkpoints are intentionally left on provider storage.
        is_attempt_evidence = "/attempts/" in name and name.endswith(".json")
        if (str(Path(name).parent) not in complete_dirs and not is_attempt_evidence
                and not name.startswith("allocations/") and name != "output/submission.json"):
            continue
        target = out / name
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_name(target.name + "." + uuid.uuid4().hex + ".download")
        h, size = hashlib.sha256(), 0
        try:
            with temp.open("xb") as stream:
                for chunk in volume.read_file(name):
                    size += len(chunk); total += len(chunk)
                    if size > MAX_DOWNLOAD_FILE_BYTES or total > MAX_DOWNLOAD_BYTES:
                        raise ValueError("bounded evidence download exceeded size cap")
                    stream.write(chunk); h.update(chunk)
                stream.flush(); os.fsync(stream.fileno())
            if target.exists():
                if file_hash(target) != h.hexdigest():
                    raise ValueError("existing evidence differs from provider bytes")
            else:
                os.rename(temp, target)
        finally:
            temp.unlink(missing_ok=True)
        files.append({"path": name, "bytes": size, "sha256": h.hexdigest()})
    return {"files": files, "bytes": total, "complete_shards": len(complete_dirs)}


def assemble(download_dir: Path, public_dir: Path, out: Path) -> dict:
    """Build full exact-coverage traces only after every shard is validated."""
    plan = load((download_dir / "control.json").read_bytes()); validate_plan(plan)
    packages, identity = load_public_inputs(public_dir)
    if identity != plan["public_files"]:
        raise ValueError("assembly input mismatch")
    out.mkdir(parents=True, exist_ok=False)
    results = []
    for tag, model in MODELS:
        for name, role in (("main_jobs.json", "main"), ("main_choices_only.json", "choices_only")):
            rows, metadata, sources = [], None, []
            for shard in shards_for(packages[name], tag, role, plan["shard_size"]):
                directory = download_dir / "output" / tag / role / shard["shard_id"]
                trace = completed_shard(directory, shard, plan["source_files_sha256"])
                if trace is None:
                    raise ValueError("cannot assemble incomplete model/role coverage")
                if metadata is None:
                    metadata = dict(trace["metadata"])
                for row in trace["predictions"]:
                    rows.append({**row, "shard_batch_index": row["batch_index"],
                                 "batch_index": shard["start"] // 8 + row["batch_index"],
                                 "shard_id": shard["shard_id"]})
                sources.append({"shard_id": shard["shard_id"], "trace_sha256": file_hash(directory / "trace.json"),
                                "metadata": trace["metadata"]})
            validate_prediction_coverage(packages[name]["jobs"], rows)
            metadata.update({"n_jobs": len(rows), "max_jobs": len(rows), "assembled_from_shards": True,
                             "source_shard_traces": sources, "run_identity": plan["run_identity"],
                             "public_package_canonical_sha256": sha(canonical(packages[name])[:-1]),
                             "n_length_capped_invalid_predictions": sum(
                                 row.get("constraint_failure_kind") == "max_new_tokens_incomplete_json" for row in rows),
                             "max_elapsed_seconds": None,
                             "timing_scope": "summed backend shard durations; per-shard limits retained in source_shard_traces",
                             "total_seconds": sum(s["metadata"]["total_seconds"] for s in sources),
                             "model_load_seconds": sum(s["metadata"]["model_load_seconds"] for s in sources)})
            trace = {"schema_version": "jane-choice-control-traces-v1" if role == "choices_only" else "jane-traces-v1",
                     "metadata": metadata, "predictions": rows}
            target = out / tag / role / "trace.json"
            write_once(target, trace)
            results.append({"model_tag": tag, "role": role, "jobs": len(rows), "trace_sha256": file_hash(target)})
    return {"status": "COMPLETED", "traces": results}


def resume(run_id: str, model_tag: str, max_model_seconds: int, repo: Path, out: Path) -> dict:
    """Reserve a new bounded allocation; previous worst-case debits are never refunded."""
    modal, _realm = connect()
    volume = modal.Volume.from_name(run_id, create_if_missing=False)
    plan = load(remote_read(volume, "control.json")); validate_plan(plan)
    if source_identity(repo) != plan["source_files_sha256"]:
        raise ValueError("resume must use exactly the originally frozen inference source")
    allocations = load(remote_read(volume, "allocations/initial.json"))
    extra = sorted(str(e.path).lstrip("/") for e in volume.iterdir("/allocations")
                   if getattr(e, "type", None) == 1 and str(e.path).endswith(".json") and not str(e.path).endswith("initial.json"))
    for name in extra:
        allocations.append(load(remote_read(volume, name)))
    validate_allocations(plan, allocations)
    if model_tag not in dict(MODELS):
        raise ValueError("unknown model tag")
    previous = [a for a in allocations if a["model_tag"] == model_tag][-1]
    prior_receipt = load(remote_read(volume, f"output/{model_tag}/attempts/{previous['reservation_index']:04d}/receipt.json"))
    if prior_receipt.get("status") != "BUDGET_PAUSED_AT_SHARD_BOUNDARY":
        raise ValueError("only a clean shard-boundary pause can resume automatically")
    budget = budget_plan(plan["budget"]["ceiling_usd"], max_model_seconds, allocations, additional_models=1)
    index = len(allocations)
    # A shared ordinal claim serializes all further allocations, across model names.
    modal.Volume.objects.create(f"{run_id}-reservation-{index:04d}", version=2, allow_existing=False)
    allocation = {"reservation_index": index, "model_tag": model_tag,
                  "max_model_seconds": max_model_seconds, "run_identity": plan["run_identity"]}
    out.mkdir(parents=True, exist_ok=False)
    write_once(out / "allocation.json", allocation); write_once(out / "budget.json", budget)
    with volume.batch_upload() as upload:
        upload.put_file(str(out / "allocation.json"), f"/allocations/{index:04d}.json")
    return spawn_detached(modal, plan, [allocation], repo, volume, out)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("plan", "launch", "status", "download", "assemble", "resume"))
    parser.add_argument("--public-dir", type=Path)
    parser.add_argument("--run-id", default="acl5000-20261002")
    parser.add_argument("--source-commit")
    parser.add_argument("--budget-usd", default="80")
    parser.add_argument("--max-model-seconds", type=int, default=57600)
    parser.add_argument("--shard-size", type=int, default=4096)
    parser.add_argument("--model", choices=tuple(dict(MODELS)))
    parser.add_argument("--download-dir", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    repo = Path(__file__).resolve().parents[1]
    if args.mode in {"plan", "launch", "assemble"} and args.public_dir is None:
        parser.error("this mode requires --public-dir")
    if args.mode not in {"plan", "status"} and args.out is None:
        parser.error("this mode requires --out")
    if args.mode == "resume" and args.model is None:
        parser.error("resume requires --model")
    if args.mode == "assemble" and args.download_dir is None:
        parser.error("assemble requires --download-dir")
    if args.mode in {"plan", "launch"}:
        plan = build_plan(args.public_dir, repo, args.run_id, args.budget_usd,
                          args.max_model_seconds, args.shard_size, args.source_commit)
        result = plan if args.mode == "plan" else launch(plan, args.public_dir, repo, args.out)
    elif args.mode == "status":
        result = status(args.run_id)
    elif args.mode == "download":
        result = download(args.run_id, args.out)
    elif args.mode == "assemble":
        result = assemble(args.download_dir, args.public_dir, args.out)
    else:
        result = resume(args.run_id, args.model, args.max_model_seconds, repo, args.out)
    print(canonical(result).decode(), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
