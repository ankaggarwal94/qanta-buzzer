#!/usr/bin/env python3
"""Bounded, attached Modal execution of frozen public Jane MC/OE jobs.

Importing this module and ``plan`` perform no cloud operations. ``launch`` is
an explicit paid action. Gold datasets and answerlines never enter its image
or arguments. Receipts report allocation estimates, not an invoiced charge.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import Decimal, ROUND_FLOOR
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from typing import Any

PRICING_URL = "https://modal.com/pricing"
PRICING_CHECKED = "2026-10-01"
GPU_RATE = Decimal("0.000542")
CPU_CORE_RATE = Decimal("0.0000131")
MEMORY_GIB_RATE = Decimal("0.00000222")
CPU_CORES = 2
MEMORY_MIB = 32768
ALLOCATION_RATE = GPU_RATE + CPU_CORE_RATE * CPU_CORES + MEMORY_GIB_RATE * 32
RESERVE_USD = Decimal("2")
MAX_BUDGET_USD = Decimal("10")
MAX_SESSION_SECONDS = 12000
SHUTDOWN_RESERVE_SECONDS = 120
MODEL_LOAD_RESERVE_SECONDS = 900
THROUGHPUT_MULTIPLIER = 3
MODAL_VERSION = "1.6.0"
VOLUME_NAME = "jane-mcq-pilot-20261001-initial"
APP_NAME = "jane-mcq-pilot-20261001"
BRANCH = "ops/jane-modal-pilot-20261001"
LAUNCH_MESSAGE = "ops: launch frozen Jane Modal pilot 20261001"
MODELS = (
    ("qwen3b", "Qwen/Qwen2.5-3B-Instruct", "aa8e72537993ba99e69dfaafa59ed015b17504d1"),
    ("qwen7b", "Qwen/Qwen2.5-7B-Instruct", "a09a35458c702b33eeacc393d103063234e8bc28"),
)
INPUT_FILES = ("dev_jobs.json", "main_jobs.json", "dev_choices_only.json", "main_choices_only.json")
SOURCE_FILES = ("scripts/__init__.py", "scripts/modal_jane_pilot.py",
                "scripts/jane_gpu_backend.py", "scripts/jane_qwen_backend.py")
PUBLIC_JOB_KEYS = {"job_id", "qid", "group_id", "split", "format", "condition", "menu_id",
                   "prefix_id", "fraction", "prompt", "prompt_sha256"}
CHOICE_JOB_KEYS = PUBLIC_JOB_KEYS - {"prefix_id", "fraction"} | {"options"}
MAX_INPUT_BYTES = 64 * 1024 * 1024
MAX_OUTPUT_BYTES = 256 * 1024 * 1024
HEX64 = re.compile(r"[0-9a-f]{64}\Z")


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                       separators=(",", ":")) + "\n").encode("utf-8")


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _load(raw: bytes) -> Any:
    def reject_constant(_value):
        raise ValueError("nonfinite JSON value")
    return json.loads(raw, object_pairs_hook=_unique, parse_constant=reject_constant)


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _write_once(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(_canonical(value))
        stream.flush()
        os.fsync(stream.fileno())


def budget_plan(budget_usd: str | Decimal = "10") -> dict[str, Any]:
    """Reject unsupported ceilings and reserve costs beyond the GPU function."""
    try:
        budget = Decimal(str(budget_usd))
    except Exception as error:
        raise ValueError("invalid compute ceiling") from error
    if not budget.is_finite() or not RESERVE_USD < budget <= MAX_BUDGET_USD:
        raise ValueError("compute ceiling must be greater than $2 and at most $10")
    seconds = min(MAX_SESSION_SECONDS, int(((budget - RESERVE_USD) / ALLOCATION_RATE)
                                          .to_integral_value(rounding=ROUND_FLOOR)))
    if seconds <= SHUTDOWN_RESERVE_SECONDS + 2 * MODEL_LOAD_RESERVE_SECONDS:
        raise ValueError("ceiling cannot cover the fixed pilot safety reserves")
    estimate = ALLOCATION_RATE * seconds + RESERVE_USD
    if estimate > budget:
        raise ValueError("allocation exceeds compute ceiling")
    return {
        "schema_version": "jane-modal-budget-v1", "ceiling_usd": str(budget),
        "reserve_usd": str(RESERVE_USD), "allocation_rate_usd_per_second": str(ALLOCATION_RATE),
        "max_session_seconds": seconds, "max_allocation_plus_reserve_usd": str(estimate),
        "gpu": "L40S", "cpu_request_and_limit": [CPU_CORES, CPU_CORES],
        "memory_request_and_limit_mib": [MEMORY_MIB, MEMORY_MIB],
        "pricing_url": PRICING_URL, "pricing_checked": PRICING_CHECKED,
        "invoice_status": "estimated; not a provider billing receipt",
    }


def validate_public_package(package: Any, *, controls: bool = False) -> list[dict]:
    """Exact allowlist rejects evaluator/answerline fields before any upload."""
    schema = "jane-choice-controls-v1" if controls else "jane-public-jobs-v1"
    if (not isinstance(package, dict) or set(package) != {"schema_version", "evidence_scope", "jobs"}
            or package["schema_version"] != schema or package["evidence_scope"] != "engineering_smoke"):
        raise ValueError("invalid public jobs envelope")
    jobs = package["jobs"]
    if not isinstance(jobs, list) or not jobs or len(jobs) > 10000:
        raise ValueError("public job count is outside the permitted range")
    seen = set()
    for job in jobs:
        if not isinstance(job, dict) or set(job) != (CHOICE_JOB_KEYS if controls else PUBLIC_JOB_KEYS):
            raise ValueError("missing or unexpected public job fields")
        for key in PUBLIC_JOB_KEYS - {"fraction", "prefix_id", "menu_id"}:
            if not isinstance(job[key], str) or not job[key].strip():
                raise ValueError("invalid public job string")
        if job["job_id"] in seen:
            raise ValueError("duplicate public job identity")
        seen.add(job["job_id"])
        if job["split"] not in {"calibration", "selection", "test"}:
            raise ValueError("invalid split")
        if _sha(job["prompt"].encode()) != job["prompt_sha256"]:
            raise ValueError("public prompt hash mismatch")
        if job["format"] not in {"mc", "oe"}:
            raise ValueError("invalid job format")
        if job["format"] == "oe":
            if controls or job["condition"] != "oe" or job["menu_id"] is not None:
                raise ValueError("invalid OE job")
        elif not isinstance(job["menu_id"], str) or not job["menu_id"]:
            raise ValueError("MC job lacks menu identity")
        if controls:
            options = job["options"]
            if (not isinstance(options, list) or len(options) != 4
                    or any(not isinstance(x, dict) or set(x) != {"id", "text"} for x in options)
                    or [x["id"] for x in options] != list("ABCD")
                    or any(not isinstance(x["text"], str) or not x["text"].strip() for x in options)):
                raise ValueError("choices control must contain four public option texts")
        else:
            fraction = job["fraction"]
            if (isinstance(fraction, bool) or not isinstance(fraction, (float, int))
                    or not math.isfinite(fraction) or not 0 < fraction <= 1
                    or not isinstance(job["prefix_id"], str) or not job["prefix_id"]):
                raise ValueError("invalid prefix")
    return jobs


def load_public_inputs(public_dir: Path) -> tuple[dict[str, dict], dict]:
    """Read only four approved files; freeze their byte-level identities."""
    packages, files, total = {}, [], 0
    for name in INPUT_FILES:
        path = public_dir / name
        if path.is_symlink() or not path.is_file():
            raise ValueError("expected a regular frozen public input file")
        if total + path.stat().st_size > MAX_INPUT_BYTES:
            raise ValueError("public inputs exceed the upload size limit")
        raw = path.read_bytes()
        total += len(raw)
        if total > MAX_INPUT_BYTES:
            raise ValueError("public inputs exceed the upload size limit")
        package = _load(raw)
        jobs = validate_public_package(package, controls="choices_only" in name)
        packages[name] = package
        files.append({"path": name, "sha256": _sha(raw), "job_count": len(jobs), "bytes": len(raw)})
    declared = _load((public_dir / "manifest.json").read_bytes())
    if (not isinstance(declared, dict) or declared.get("schema_version") != "jane-public-input-manifest-v1"
            or set(declared.get("files", {})) != set(INPUT_FILES)):
        raise ValueError("invalid frozen public manifest")
    for entry in files:
        expected = {"sha256": entry["sha256"], "byte_count": entry["bytes"], "job_count": entry["job_count"]}
        if declared["files"][entry["path"]] != expected:
            raise ValueError("public file differs from its frozen manifest")
    dev = packages["dev_jobs.json"]["jobs"]
    main = packages["main_jobs.json"]["jobs"]
    # These 12 questions are development-only by their file/phase identity.
    # Compatibility labels span all three roles; none are the MAIN test set.
    for key in ("qid", "group_id"):
        if {j[key] for j in dev} & {j[key] for j in main}:
            raise ValueError("development and main questions overlap")
    if len({job["qid"] for job in main}) != 200:
        raise ValueError("main pilot must contain exactly 200 questions")
    if {job["split"] for job in main} != {"calibration", "selection", "test"}:
        raise ValueError("main must have separate calibration, selection, and test roles")
    for prefix in ("dev", "main"):
        normal = packages[f"{prefix}_jobs.json"]["jobs"]
        choices = packages[f"{prefix}_choices_only.json"]["jobs"]
        expected = {(j["qid"], j["condition"], j["menu_id"]) for j in normal if j["format"] == "mc"}
        observed = {(j["qid"], j["condition"], j["menu_id"]) for j in choices}
        if expected != observed or len(choices) != len(observed):
            raise ValueError("choices-only controls do not match the paired MC menus")
    identity = {"schema_version": "jane-modal-public-inputs-v1", "files": files}
    identity["public_input_id"] = _sha(_canonical(identity))
    return packages, identity


def interface_gate(jobs: list[dict], trace: dict) -> dict:
    """Gate formatting only; neither gold correctness nor confidence magnitude is selected."""
    from scripts.jane_qwen_backend import parse_response
    predictions = trace.get("predictions")
    if not isinstance(predictions, list) or len(predictions) != len(jobs):
        raise ValueError("development trace is incomplete")
    by_id = {p.get("job_id"): p for p in predictions if isinstance(p, dict)}
    if len(by_id) != len(jobs) or set(by_id) != {j["job_id"] for j in jobs}:
        raise ValueError("development trace identities differ from the frozen inputs")
    counts = {fmt: {"jobs": 0, "schema_valid": 0, "illegal_mc_ids": 0, "placeholders": 0}
              for fmt in ("mc", "oe")}
    for job in jobs:
        prediction = by_id[job["job_id"]]
        if prediction.get("prompt_sha256") != job["prompt_sha256"]:
            raise ValueError("development prediction prompt hash mismatch")
        raw = prediction.get("raw_response")
        if not isinstance(raw, str):
            raise ValueError("development prediction lacks raw response")
        parsed = parse_response(raw)
        for key in ("answer", "confidence", "status"):
            if prediction.get(key) != parsed[key]:
                raise ValueError("development parsed fields differ from exact raw parsing")
        c = counts[job["format"]]
        c["jobs"] += 1
        if parsed["status"] in {"answer", "abstain"}:
            c["schema_valid"] += 1
        if parsed["status"] == "answer":
            if job["format"] == "mc" and parsed["answer"] not in set("ABCD"):
                c["illegal_mc_ids"] += 1
            if job["format"] == "oe" and parsed["answer"].strip().lower() in {
                    "...", "…", "[answer]", "<answer>", "answer", "your answer", "answer here", "your answer here"}:
                c["placeholders"] += 1
    passed = all(c["jobs"] and c["schema_valid"] / c["jobs"] >= 0.95 for c in counts.values())
    passed = passed and not counts["mc"]["illegal_mc_ids"] and not counts["oe"]["placeholders"]
    return {"passed": bool(passed), "per_format": counts,
            "criterion": "each format >=95% exact JSON valid; zero illegal answered MC IDs; zero OE placeholders",
            "uses_gold_accuracy": False}


def throughput_gate(development: dict[str, dict], packages: dict, remaining_seconds: float) -> dict:
    predicted = 2 * MODEL_LOAD_RESERVE_SECONDS
    components = []
    for tag, _model, _revision in MODELS:
        seconds, count = development[tag]["inference_seconds"], development[tag]["jobs"]
        if not math.isfinite(seconds) or seconds <= 0 or count <= 0:
            raise ValueError("invalid development throughput measurement")
        main_count = len(packages["main_jobs.json"]["jobs"]) + len(packages["main_choices_only.json"]["jobs"])
        estimate = THROUGHPUT_MULTIPLIER * seconds / count * main_count
        components.append({"model": tag, "main_and_control_jobs": main_count,
                           "development_seconds_per_job": seconds / count,
                           "buffered_generation_seconds": estimate})
        predicted += estimate
    return {"passed": predicted <= remaining_seconds - SHUTDOWN_RESERVE_SECONDS,
            "predicted_seconds_with_load_reserve": predicted,
            "remaining_seconds": remaining_seconds, "components": components,
            "generation_multiplier": THROUGHPUT_MULTIPLIER,
            "per_model_load_reserve_seconds": MODEL_LOAD_RESERVE_SECONDS}


def phase_timing(trace: dict, wall_seconds: float) -> dict:
    metadata = trace.get("metadata", {})
    total, load = metadata.get("total_seconds"), metadata.get("model_load_seconds")
    if (isinstance(total, bool) or isinstance(load, bool)
            or not isinstance(total, (float, int)) or not isinstance(load, (float, int))
            or not math.isfinite(total) or not math.isfinite(load) or not 0 <= load < total):
        raise ValueError("backend does not supply a usable separate loading/inference timing")
    return {"phase_wall_seconds": wall_seconds, "backend_total_seconds": total,
            "loading_download_hash_seconds": load, "inference_seconds": total - load}


class BatchCheckpoint:
    """Create-once append stream closed before each Volume commit/reload.

    Modal 1.6.0 commit may internally reload. Its reload requires all mounted
    Volume files closed. The backend writes, flushes and fsyncs a batch, then
    invokes progress; that callback closes this handle before committing.
    The next write lazily opens the same create-once checkpoint for append.
    """
    def __init__(self, path: Path):
        self.path, self.stream = path, None
        with path.open("x", encoding="utf-8"):
            pass

    def write(self, value: str):
        if self.stream is None:
            self.stream = self.path.open("a", encoding="utf-8")
        return self.stream.write(value)

    def flush(self):
        if self.stream is not None:
            self.stream.flush()

    def fileno(self):
        if self.stream is None:
            raise OSError("checkpoint has no open batch")
        return self.stream.fileno()

    def close_batch(self):
        if self.stream is not None:
            self.stream.flush()
            os.fsync(self.stream.fileno())
            self.stream.close()
            self.stream = None

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close_batch()


def workspace_identity(response) -> dict:
    """Discard token/user fields and fail closed on the wrong account realm."""
    if getattr(response, "workspace_name", None) != "ankaggarwal94":
        raise ValueError("configured Modal credentials belong to an unexpected workspace")
    workspace_id = getattr(response, "workspace_id", None)
    if not isinstance(workspace_id, str) or not workspace_id:
        raise ValueError("Modal workspace identity is missing")
    return {"workspace_name": "ankaggarwal94", "workspace_id": workspace_id}


def verified_workspace() -> dict:
    # Mirrors the pinned SDK's `modal.cli.token.info` request, but never emits
    # token IDs, secrets, user identities, or the complete server response.
    # This low-level API is version-dependent and must fail closed if removed.
    from modal._utils.async_utils import synchronize_api
    from modal.client import _Client
    from modal_proto import api_pb2
    async def _read_workspace():
        client = await _Client.from_env()
        response = await client._stub.TokenInfoGet(api_pb2.TokenInfoGetRequest())
        return workspace_identity(response)
    return synchronize_api(_read_workspace)()


def source_identity(repo: Path) -> dict:
    files = {}
    for name in SOURCE_FILES:
        path = repo / name
        if path.is_symlink() or not path.is_file():
            raise ValueError("approved inference source is missing or not regular")
        files[name] = _sha(path.read_bytes())
    return files


def verify_committed_source(repo: Path, commit: str, expected: dict) -> None:
    """Bind the executing allowlist to the claimed commit without checking unrelated files."""
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, check=True).stdout.strip().decode()
    if head != commit:
        raise ValueError("source commit differs from the local checkout")
    for path, sha in expected.items():
        result = subprocess.run(["git", "show", f"{commit}:{path}"], cwd=repo,
                                capture_output=True, check=False)
        if result.returncode or _sha(result.stdout) != sha:
            raise ValueError("executing inference source is not exactly committed at the claimed SHA")


@contextmanager
def deadline(seconds: float):
    """POSIX process deadline cancels blocking download/generation and local waits."""
    if seconds <= 0:
        raise TimeoutError("pilot deadline expired")
    previous = signal.getsignal(signal.SIGALRM)
    def expire(_signum, _frame):
        raise TimeoutError("pilot allocation deadline expired")
    signal.signal(signal.SIGALRM, expire)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def remote_pilot(packages: dict, control: dict) -> dict:
    """Single input: both DEV gates first, then sequential frozen MAIN inference."""
    import modal
    from scripts.jane_gpu_backend import GPUConfig, run
    started = time.perf_counter()
    root = Path("/jane/output")
    volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
    volume.reload()
    root.mkdir(parents=True, exist_ok=True)
    # Persist before model loading; rescheduled input refuses replay.
    _write_once(root / "gpu_execution_started.json", {"started_utc": datetime.now(timezone.utc).isoformat(),
                "public_input_id": control["public_input_id"], "source_commit": control["source_commit"]})
    volume.commit()
    execution_limit = min(control["budget"]["max_session_seconds"] - SHUTDOWN_RESERVE_SECONDS,
                          control["absolute_deadline_unix"] - time.time())
    outcomes, development, status = {}, {}, "FAILED"
    def remaining():
        return min(execution_limit - (time.perf_counter() - started),
                   control["absolute_deadline_unix"] - time.time())
    try:
        if source_identity(Path("/opt/jane")) != control["source_files_sha256"]:
            raise ValueError("remote executing source differs from frozen source")
        for name, package in packages.items():
            validate_public_package(package, controls="choices_only" in name)
        with deadline(execution_limit):
            def generate(tag, model, revision, role, package):
                out = root / tag / role
                out.mkdir(parents=True, exist_ok=False)
                elapsed_limit = remaining() - SHUTDOWN_RESERVE_SECONDS
                if elapsed_limit <= 0:
                    raise TimeoutError("remaining allocation reserve exhausted")
                def commit_progress(_progress):
                    checkpoint.close_batch()
                    volume.commit()
                call_started = time.perf_counter()
                with BatchCheckpoint(out / "predictions.checkpoint.jsonl") as checkpoint:
                    trace = run(package, GPUConfig(model=model, revision=revision, batch_size=8,
                        max_jobs=10000, max_input_tokens=2048, max_new_tokens=96,
                        max_elapsed_seconds=elapsed_limit, seed=1, threads=4,
                        cache_dir=Path("/tmp/jane-models"), allow_download=True),
                        checkpoint=checkpoint, progress=commit_progress)
                _write_once(out / "trace.json", trace)
                result = {"jobs": len(package["jobs"]), **phase_timing(trace, time.perf_counter() - call_started),
                          "trace_sha256": _sha(_canonical(trace))}
                _write_once(out / "completion.json", result)
                volume.commit()
                return trace, result
            for tag, model, revision in MODELS:
                trace, measured = generate(tag, model, revision, "development", packages["dev_jobs.json"])
                gate = interface_gate(packages["dev_jobs.json"]["jobs"], trace)
                _write_once(root / tag / "development_gate.json", gate)
                outcomes[tag] = {"development": measured, "interface_gate": gate}
                development[tag] = measured
                volume.commit()
                if not gate["passed"]:
                    status = "DEVELOPMENT_INTERFACE_GATE_FAILED"
                    return {"status": status, "models": outcomes}
            gate = throughput_gate(development, packages, remaining())
            _write_once(root / "throughput_gate.json", gate)
            volume.commit()
            if not gate["passed"]:
                status = "DEVELOPMENT_THROUGHPUT_GATE_FAILED"
                return {"status": status, "models": outcomes, "throughput_gate": gate}
            for tag, model, revision in MODELS:
                for role, file_name in (("main", "main_jobs.json"), ("choices_only", "main_choices_only.json")):
                    _trace, measured = generate(tag, model, revision, role, packages[file_name])
                    outcomes[tag][role] = measured
            status = "COMPLETED"
            return {"status": status, "models": outcomes, "throughput_gate": gate}
    except Exception as error:
        _write_once(root / "failure.json", {"status": "FAILED", "exception_type": type(error).__name__,
                                           "raw_exception_logged": False})
        raise RuntimeError("Jane GPU execution failed; inspect persisted partial evidence") from None
    finally:
        elapsed = time.perf_counter() - started
        _write_once(root / "execution_receipt.json", {"status": status,
                    "elapsed_seconds": elapsed, "allocation_estimate_usd": str(ALLOCATION_RATE * Decimal(str(elapsed))),
                    "public_input_id": control["public_input_id"], "source_commit": control["source_commit"],
                    "models": outcomes, "model_pins": list(MODELS), "budget": control["budget"]})
        volume.commit()


def download_evidence(volume, out: Path) -> dict:
    """Download only output paths, including checkpoints; never model cache."""
    total, files = 0, []
    for entry in volume.iterdir("/output", recursive=True):
        path = str(entry.path).lstrip("/")
        if not path.startswith("output/") or ".." in Path(path).parts:
            raise ValueError("unexpected provider output path")
        if getattr(entry, "type", None) != 1:  # Modal FileEntryType.FILE == 1.
            continue
        target = out / path.removeprefix("output/")
        target.parent.mkdir(parents=True, exist_ok=True)
        hasher, size = hashlib.sha256(), 0
        with target.open("xb") as stream:
            for chunk in volume.read_file(path):
                total += len(chunk)
                size += len(chunk)
                if total > MAX_OUTPUT_BYTES:
                    raise ValueError("provider evidence exceeds bounded egress limit")
                stream.write(chunk)
                hasher.update(chunk)
        files.append({"path": str(target.relative_to(out)), "sha256": hasher.hexdigest(), "bytes": size})
    return {"schema_version": "jane-modal-download-v1", "files": files, "bytes": total}


def launch(packages: dict, control: dict, out: Path, repo: Path) -> dict:
    """Explicit paid operation. Existing named allocation claim forbids reruns."""
    verify_committed_source(repo, control["source_commit"], control["source_files_sha256"])
    # Some submission environments proxy TLS through a system-trusted CA.
    # This changes only the local submission process, never the GPU image.
    try:
        import truststore
    except ImportError:
        pass
    else:
        truststore.inject_into_ssl()
    import modal
    import importlib.metadata
    if importlib.metadata.version("modal") != MODAL_VERSION:
        raise ValueError("the reviewed Modal SDK version is required")
    from modal._utils.function_utils import FunctionSourceInfo
    if FunctionSourceInfo(remote_pilot).module_name != "scripts.modal_jane_pilot":
        raise ValueError("launch requires the python -m scripts.modal_jane_pilot entrypoint")
    out.mkdir(parents=True, exist_ok=False)
    _write_once(out / "frozen_control.json", control)
    client = modal.Client.from_env()
    client.hello()
    realm = verified_workspace()
    _write_once(out / "workspace_receipt.json", realm)
    # Atomic provider-name creation is the allocation claim, before GPU launch.
    modal.Volume.objects.create(VOLUME_NAME, version=2, allow_existing=False, client=client)
    volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False, client=client)
    volume.hydrate()
    _write_once(out / "allocation_claim.json", {"volume_name": VOLUME_NAME,
                "volume_id": volume.object_id, "source_commit": control["source_commit"],
                "public_input_id": control["public_input_id"]})
    image = modal.Image.debian_slim(python_version="3.11").pip_install(
        "torch==2.6.0", "transformers==4.51.3", "tokenizers==0.21.1",
        "safetensors==0.5.3", "huggingface-hub==0.30.2", "accelerate==1.6.0", "modal==1.6.0")
    for name in SOURCE_FILES:
        image = image.add_local_file(str(repo / name), remote_path="/opt/jane/" + name, copy=True)
    image = image.env({"PYTHONPATH": "/opt/jane", "PYTHONUNBUFFERED": "1",
                       "HF_HUB_DISABLE_TELEMETRY": "1", "TOKENIZERS_PARALLELISM": "false"})
    app = modal.App(APP_NAME, image=image, include_source=False)
    function = app.function(gpu="L40S", cpu=(2, 2), memory=(32768, 32768),
        volumes={"/jane": volume}, max_containers=1, min_containers=0, buffer_containers=0,
        scaledown_window=2, retries=0, timeout=control["budget"]["max_session_seconds"] - 120,
        startup_timeout=90, include_source=False, secrets=[])(remote_pilot)
    session_started, call, result, failure = time.monotonic(), None, None, None
    control = {**control, "absolute_deadline_unix": time.time() + control["budget"]["max_session_seconds"] - 120}
    _write_once(out / "submission_control.json", control)
    try:
        with deadline(control["budget"]["max_session_seconds"] - SHUTDOWN_RESERVE_SECONDS):
            with modal.enable_output():
                with app.run(client=client, detach=False):
                    try:
                        call = function.spawn(packages, control)
                        _write_once(out / "provider_launch.json", {"app_id": app.app_id,
                                    "function_call_id": call.object_id, "volume_name": VOLUME_NAME})
                        wait_seconds = control["absolute_deadline_unix"] - time.time()
                        if wait_seconds <= 0:
                            raise TimeoutError("image build exhausted initial allocation")
                        result = call.get(timeout=wait_seconds)
                    except BaseException:
                        if call is not None:
                            call.cancel(terminate_containers=True)
                        raise
    except Exception as error:
        failure = type(error).__name__
    finally:
        try:
            receipt = download_evidence(volume, out / "evidence")
            _write_once(out / "download_receipt.json", receipt)
        except Exception as error:
            _write_once(out / "download_failure.json", {"exception_type": type(error).__name__})
        _write_once(out / "host_receipt.json", {"status": (result or {}).get("status", "FAILED"),
                    "exception_type": failure, "attached": True, "automatic_function_retries": 0,
                    "session_elapsed_seconds": time.monotonic() - session_started,
                    "result": result, "budget": control["budget"], "actual_invoice_verified": False})
    if failure or not result or result.get("status") != "COMPLETED":
        raise RuntimeError("pilot did not complete; retained evidence describes the boundary")
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("plan", "launch"))
    parser.add_argument("--public-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--budget-usd", default="10")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    repo = Path(__file__).resolve().parents[1]
    try:
        if not re.fullmatch(r"[0-9a-f]{40}", args.source_commit):
            raise ValueError("source commit must be an exact Git SHA")
        packages, identity = load_public_inputs(args.public_dir)
        control = {**identity, "budget": budget_plan(args.budget_usd),
                   "source_commit": args.source_commit, "source_files_sha256": source_identity(repo),
                   "model_pins": list(MODELS)}
        if args.mode == "plan":
            print(_canonical(control).decode(), end="")
            return 0
        if args.out is None:
            raise ValueError("launch requires a create-once output directory")
        result = launch(packages, control, args.out, repo)
        print(json.dumps({"status": result["status"], "out": str(args.out)}))
        return 0
    except (ValueError, OSError, ImportError, RuntimeError) as error:
        # Credentialed SDK/server exception text is deliberately not emitted.
        print(f"Jane Modal pilot stopped: {type(error).__name__}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
