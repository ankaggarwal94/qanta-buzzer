#!/usr/bin/env python3
"""Dedicated create-once Mistral cache and two-stage scoring controller.

Dry runs never import Modal or download model files. Collection/status never
construct an App or scoring function. Every paid attempt retains its reservation.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import io
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import re
import signal
import subprocess
import sys
import time

from scripts.imcqa_mistral_budget import (APPROVAL_ID, STAGES, CACHE_TIMEOUT,
    STARTUP_TIMEOUT, SCALEDOWN, budget_plan, validate_budget)

MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
REVISION = "c170c708c41dac9275d15a8fff4eca08d52bab71"
MODEL_TAG = "mistral7b"
BASE_IMAGE_ID = "im-NbuDclsJozF3UpfDtW64cC"
INHERITED_VERSIONS = {"torch": "2.6.0", "transformers": "4.51.3", "tokenizers": "0.21.1",
    "safetensors": "0.5.3", "huggingface-hub": "0.30.2", "modal": "1.6.0"}
ADDED_VERSIONS = {"sentencepiece": "0.2.0", "protobuf": "5.29.5"}
CHAT_TEMPLATE_SHA256 = "e16746b40344d6c5b5265988e0328a0bf7277be86f1c335156eae07e29c82826"
MODEL_FILES = ("config.json", "generation_config.json", "model.safetensors.index.json",
    "model-00001-of-00003.safetensors", "model-00002-of-00003.safetensors",
    "model-00003-of-00003.safetensors", "special_tokens_map.json", "tokenizer.json",
    "tokenizer.model", "tokenizer_config.json")
SOURCES = ("scripts/__init__.py", "scripts/modal_imcqa_mistral.py", "scripts/imcqa_mistral_budget.py",
    "scripts/imcqa_mistral_scoring.py", "scripts/imcqa_mistral_design.py",
    "scripts/imcqa_tuned_scoring.py", "scripts/imcqa_tuned_design.py",
    "scripts/imcqa_protocol_scoring.py", "scripts/imcqa_protocol_design.py", "scripts/imcqa_wait_scoring.py",
    "scripts/acl_paired_prompt_scoring.py", "scripts/acl_option_scoring.py",
    "scripts/modal_acl_expansion.py", "scripts/jane_gpu_backend.py", "scripts/jane_qwen_backend.py",
    "scripts/jane_output_constraints.py", "configs/imcqa_protocol_pilot.json",
    "configs/imcqa_mistral_replication.json")
CONTROL_SOURCES = ("scripts/select_imcqa_mistral_policies.py", "scripts/select_imcqa_tuned_policies.py",
    "scripts/analyze_imcqa_protocol_pilot.py", "scripts/analyze_imcqa_wait_pilot.py",
    "scripts/analyze_imcqa_transfer.py", "scripts/analyze_imcqa_tuned.py",
    "scripts/analyze_imcqa_fixed_abstention.py", "scripts/analyze_imcqa_mistral.py",
    "scripts/modal_acl_paired_prompt_scores.py", "scripts/modal_acl_option_scores.py")
MODEL_HASHES = {
    "config.json": "affafc6478ec0fd07a32f0ca57aa2fc57743f4d17d6730f86a96ac24d1507f99",
    "generation_config.json": "b4669f1b8f4185324bd9b12a69e85a1ad2289bc48111ef739cfaaea3bba6b0a9",
    "model.safetensors.index.json": "e489ba553b87cde188d921b1a8283c2e0b9d33d635b88147d96ff0fcd6250016",
    "model-00001-of-00003.safetensors": "ce6fb6f6f4d0183f4813cbf4ece24109da629a08d4210da46f77e1d8b0bd5c19",
    "model-00002-of-00003.safetensors": "8c0e72f148366b6a3709e002a98706a33d31aec8515090c856c95b2044f92ae0",
    "model-00003-of-00003.safetensors": "905dd405363e43d95779c1c1155a2dbfd36155914ae95dbd934e12e490cfb4ca",
    "special_tokens_map.json": "6fa06efa2785e450051989a6f8fb4416b10149ded485ddd3f127a40734f5cfd0",
    "tokenizer.json": "e553af6fff7d7ad76e830608b218c5c0b0822998d5a1a96099a74cd3c1cb1a49",
    "tokenizer.model": "37f00374dea48658ee8f5d0f21895b9bc55cb0103939607c8185bfd1c6ca1f89",
    "tokenizer_config.json": "0533dec9cfe319163801b6618d0f3ec9cfa126b6288e3df5deca6e32acb09cd2",
}


def canonical(value) -> bytes:
    return (json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")) + "\n").encode()


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def load(raw: bytes) -> dict:
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("duplicate JSON key")
            value[key] = item
        return value
    def reject(value):
        raise ValueError("nonfinite JSON number: " + value)
    value = json.loads(raw, object_pairs_hook=unique, parse_constant=reject)
    if not isinstance(value, dict):
        raise ValueError("JSON object required")
    return value


def write_once(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(canonical(value))


def source_hashes(repo: Path) -> dict:
    for name in SOURCES:
        if (repo / name).is_symlink() or not (repo / name).is_file():
            raise ValueError("missing/nonregular source: " + name)
    return {name: digest(repo / name) for name in SOURCES}


def validate_control(control: dict, *, fresh=True) -> None:
    validate_budget(control["budget"], require_fresh=fresh)
    if (control.get("approval_id") != APPROVAL_ID or control.get("model") != MODEL
            or control.get("revision") != REVISION or control.get("base_image_id") != BASE_IMAGE_ID
            or not re.fullmatch(r"imcqa-mistral-[a-z0-9-]{1,24}", control["run_id"])
            or control["cache_volume"] != control["run_id"] + "-cache"
            or not re.fullmatch(r"[0-9a-f]{40}", control["source_commit"])
            or set(control["source_files_sha256"]) != set(SOURCES)
            or set(control["control_source_files_sha256"]) != set(CONTROL_SOURCES)):
        raise ValueError("unexpected approved experiment/model/source identity")
    if control["mode"] == "prepare-cache":
        if control["volume"] != control["cache_volume"]:
            raise ValueError("cache volume differs")
    elif control["mode"] == "score-stage":
        stage = control["stage"]
        if stage not in STAGES or control["volume"] != control["run_id"] + "-" + stage:
            raise ValueError("unknown scoring stage/volume")
        if control["expected_contexts"] != STAGES[stage]["contexts"]:
            raise ValueError("unexpected scoring count")
        for key in ("prepare_receipt_sha256", "public_input_sha256", "stage_manifest_sha256"):
            if not re.fullmatch(r"[0-9a-f]{64}", control[key]):
                raise ValueError("missing stage content identity")
    else:
        raise ValueError("unknown control mode")


def verify_sources(repo: Path, control: dict, *, committed=False) -> None:
    validate_control(control)
    if source_hashes(repo) != control["source_files_sha256"]:
        raise ValueError("sources changed after freezing cache control")
    if committed:
        from scripts.modal_acl_expansion import verify_source_commit
        if {name: digest(repo / name) for name in CONTROL_SOURCES} != control["control_source_files_sha256"]:
            raise ValueError("control-plane source changed after freezing cache control")
        verify_source_commit(repo, control["source_commit"],
            {**control["source_files_sha256"], **control["control_source_files_sha256"]})


def cache_control(repo: Path, run_id: str, source_commit: str, rate_verified_utc: str) -> dict:
    result = {"schema_version": "imcqa-mistral-control-v1", "mode": "prepare-cache",
        "approval_id": APPROVAL_ID, "run_id": run_id, "volume": run_id + "-cache",
        "cache_volume": run_id + "-cache", "model": MODEL, "revision": REVISION,
        "base_image_id": BASE_IMAGE_ID,
        "source_commit": source_commit, "source_files_sha256": source_hashes(repo),
        "control_source_files_sha256": {name: digest(repo / name) for name in CONTROL_SOURCES},
        "budget": budget_plan(rate_verified_utc), "created_utc": datetime.now(timezone.utc).isoformat()}
    validate_control(result)
    return result


def validate_prepare(receipt: dict, control: dict) -> None:
    verify_prepare(canonical(receipt))
    if (receipt.get("status") != "complete" or receipt.get("model") != MODEL
            or receipt.get("revision") != REVISION or receipt.get("cache_volume") != control["cache_volume"]
            or receipt.get("approval_id") != APPROVAL_ID
            or receipt.get("source_commit") != control["source_commit"]
            or receipt.get("source_files_sha256") != control["source_files_sha256"]
            or set(receipt.get("model_files_sha256", {})) != set(MODEL_FILES)
            or any(not re.fullmatch(r"[0-9a-f]{64}", h) for h in receipt["model_files_sha256"].values())):
        raise ValueError("preparation receipt differs from pinned source/model cache")


def verify_prepare(raw: bytes) -> dict:
    """Validate the dedicated model cache receipt without importing a scorer."""
    receipt = load(raw)
    sidecar = {"model": MODEL, "revision": REVISION,
               "model_files_sha256": receipt.get("model_files_sha256")}
    if (receipt.get("schema_version") != "imcqa-mistral-cache-v1" or receipt.get("status") != "complete"
            or receipt.get("model") != MODEL or receipt.get("revision") != REVISION
            or receipt.get("chat_template_sha256") != CHAT_TEMPLATE_SHA256
            or receipt.get("model_receipts") != {MODEL_TAG: sidecar}
            or receipt.get("base_image_id") != BASE_IMAGE_ID
            or receipt.get("dependency_versions") != ADDED_VERSIONS
            or not receipt.get("dependency_files_sha256")
            or receipt.get("model_files_sha256") != MODEL_HASHES):
        raise ValueError("incomplete/wrong dedicated Mistral preparation receipt")
    return receipt


def stage_control(cache: dict, stage: str, public_raw: bytes, manifest_raw: bytes,
                  preparation_raw: bytes, *, policy_lock=None, development_receipt=None) -> dict:
    from scripts.imcqa_mistral_design import validate_stage_manifest, validate_public_package
    validate_control(cache)
    if cache["mode"] != "prepare-cache" or stage not in STAGES:
        raise ValueError("require cache control and one known stage")
    validate_prepare(load(preparation_raw), cache)
    package, manifest = load(public_raw), load(manifest_raw)
    jobs = validate_public_package(package)
    validate_stage_manifest(manifest, package, policy_lock=policy_lock,
                            development_receipt=development_receipt)
    if stage == "evaluation":
        from scripts.select_imcqa_mistral_policies import validate_policy_lock
        validate_policy_lock(policy_lock, evaluation_package=package)
        if policy_lock["selection_script_sha256"] != cache["control_source_files_sha256"]["scripts/select_imcqa_mistral_policies.py"]:
            raise ValueError("policy selection source differs from frozen control plane")
    if (manifest["stage"] != stage or manifest["public_input_sha256"] != sha(public_raw)
            or len(jobs) != STAGES[stage]["contexts"]
            or len({row["qid"] for row in jobs}) != STAGES[stage]["questions"]):
        raise ValueError("stage count or serialized public identity differs")
    result = {**cache, "mode": "score-stage", "stage": stage,
        "volume": cache["run_id"] + "-" + stage, "expected_contexts": len(jobs),
        "prepare_receipt_sha256": sha(preparation_raw), "public_input_sha256": sha(public_raw),
        "stage_manifest_sha256": sha(manifest_raw), "stage_manifest": manifest,
        "policy_lock": policy_lock, "development_receipt": development_receipt}
    validate_control(result)
    return result


@contextmanager
def local_deadline(seconds: float):
    """Bound blocking operations in this Linux main process; no timeout resets."""
    if seconds <= 0:
        raise TimeoutError("allocation deadline already expired")
    def expired(signum, frame):
        raise TimeoutError("absolute allocation deadline exceeded")
    old_handler = signal.signal(signal.SIGALRM, expired)
    old_timer = signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, *old_timer)
        signal.signal(signal.SIGALRM, old_handler)


def worker_claim(control: dict):
    import modal
    validate_control(control)
    if time.time() >= control["absolute_deadline_unix"]:
        raise TimeoutError("allocation deadline expired before worker claim")
    suffix = "cache" if control["mode"] == "prepare-cache" else control["stage"]
    modal.Volume.objects.create(APPROVAL_ID + "-" + suffix + "-worker", version=2, allow_existing=False)
    verify_sources(Path("/root"), control)
    return modal.Volume.from_name(control["volume"], create_if_missing=False)


def remote_prepare(control: dict) -> dict:
    volume = worker_claim(control)
    output, cache = Path("/cached/output"), Path("/cached/models")
    started = time.monotonic()
    write_once(output / "cache_started.json", {"approval_id": APPROVAL_ID, "started_unix": time.time()})
    volume.commit()
    with local_deadline(min(CACHE_TIMEOUT - 60, control["absolute_deadline_unix"] - time.time() - 15)):
        for name, expected in INHERITED_VERSIONS.items():
            if importlib.metadata.version(name).split("+")[0] != expected:
                raise ValueError("inherited image dependency differs: " + name)
        # The only package installation happens inside this bounded CPU worker.
        # The preexisting base image is never rebuilt or modified.
        deps = Path("/cached/deps")
        if deps.exists():
            raise FileExistsError("new dependency cache unexpectedly exists")
        subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "--no-cache-dir",
            "--target", str(deps), *[k + "==" + v for k, v in ADDED_VERSIONS.items()]],
            check=True, timeout=max(1, min(120, control["absolute_deadline_unix"] - time.time() - 30)))
        sys.path.insert(0, str(deps))
        for name, expected in ADDED_VERSIONS.items():
            if importlib.metadata.version(name) != expected:
                raise ValueError("new cache dependency differs: " + name)
        dependency_hashes = {p.relative_to(deps).as_posix(): digest(p) for p in sorted(deps.rglob("*"))
                             if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"}
        for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
            os.environ.pop(key, None)
        os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
        from huggingface_hub import snapshot_download
        from transformers import AutoTokenizer
        snapshot = Path(snapshot_download(repo_id=MODEL, revision=REVISION, cache_dir=str(cache),
            allow_patterns=list(MODEL_FILES), max_workers=3, token=False))
        hashes = {p.relative_to(snapshot).as_posix(): digest(p) for p in sorted(snapshot.rglob("*")) if p.is_file()}
        if snapshot.name != REVISION or hashes != MODEL_HASHES:
            raise ValueError("cache lacks exact Transformers shard/tokenizer allowlist")
        index = load((snapshot / "model.safetensors.index.json").read_bytes())
        if set(index["weight_map"].values()) != {n for n in MODEL_FILES if n.endswith(".safetensors")}:
            raise ValueError("weight index references unexpected shards")
        tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=False)
        if not tokenizer.chat_template:
            raise ValueError("native tokenizer chat template unavailable")
        sidecar = {"model": MODEL, "revision": REVISION, "model_files_sha256": hashes}
        write_once(cache / (MODEL_TAG + "_expected_model_hashes.json"), sidecar)
        receipt = {"schema_version": "imcqa-mistral-cache-v1", "status": "complete", **sidecar,
            "model_receipts": {MODEL_TAG: sidecar}, "base_image_id": BASE_IMAGE_ID,
            "dependency_versions": ADDED_VERSIONS, "dependency_files_sha256": dependency_hashes,
            "approval_id": APPROVAL_ID, "cache_volume": control["cache_volume"],
            "source_commit": control["source_commit"], "source_files_sha256": control["source_files_sha256"],
            "total_snapshot_bytes": sum((snapshot / name).stat().st_size for name in hashes),
            "chat_template_sha256": sha(tokenizer.chat_template.encode()),
            "elapsed_seconds": time.monotonic() - started, "gpu_executed": False}
        validate_prepare(receipt, control)
        write_once(output / "prepare_receipt.json", receipt)
        volume.commit()
        return receipt


def remote_score(control: dict) -> dict:
    volume = worker_claim(control)
    os.environ.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1", "TOKENIZERS_PARALLELISM": "false",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8"})
    sys.path.insert(0, "/cached/deps")
    from scripts.imcqa_mistral_design import validate_stage_manifest
    from scripts.imcqa_mistral_scoring import run_scoring
    from scripts.modal_acl_expansion import replace_progress
    cache, root = Path("/cached"), Path("/pilot")
    preparation_raw = (cache / "output/prepare_receipt.json").read_bytes()
    if sha(preparation_raw) != control["prepare_receipt_sha256"]:
        raise ValueError("remote cache preparation differs from stage binding")
    validate_prepare(load(preparation_raw), control)
    dependency_hashes = {p.relative_to(Path("/cached/deps")).as_posix(): digest(p)
        for p in sorted(Path("/cached/deps").rglob("*"))
        if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"}
    if dependency_hashes != load(preparation_raw)["dependency_files_sha256"]:
        raise ValueError("cached dependency files changed after CPU preparation")
    public_path = root / "public/pilot.json"
    if sha(public_path.read_bytes()) != control["public_input_sha256"]:
        raise ValueError("remote public package differs")
    validate_stage_manifest(control["stage_manifest"], load(public_path.read_bytes()),
        policy_lock=control["policy_lock"], development_receipt=control["development_receipt"])
    def progress(value):
        replace_progress(root / "output" / MODEL_TAG / "progress.json", value)
        volume.commit()
        print(json.dumps(value, sort_keys=True), flush=True)
    timeout = STAGES[control["stage"]]["timeout"]
    internal_seconds = timeout - 120
    if control["absolute_deadline_unix"] - time.time() - 15 < internal_seconds:
        raise TimeoutError("full predeclared scoring window no longer fits absolute deadline")
    return run_scoring(MODEL_TAG, public_path, control["public_input_sha256"], cache / "models",
        root / "output" / MODEL_TAG, source_commit=control["source_commit"],
        max_seconds=internal_seconds, progress=progress)


def remote_read(volume, name: str, maximum=1024**2) -> bytes:
    result = bytearray()
    for chunk in volume.read_file(name):
        result.extend(chunk)
        if len(result) > maximum:
            raise ValueError("remote metadata exceeds byte limit")
    return bytes(result)


def upload(volume, records: dict[str, bytes]):
    with volume.batch_upload(force=False) as batch:
        for name, raw in records.items():
            batch.put_file(io.BytesIO(raw), "/" + name)


def collect(volume, out: Path, control: dict) -> dict:
    """Read only bounded JSON/text evidence, never cached weights or public data."""
    limit = 16 * 1024**2 if control["mode"] == "prepare-cache" else 16 * 1024**2 + control["expected_contexts"] * 32768
    files, seen, total = [], set(), 0
    for entry in volume.iterdir("/output", recursive=True):
        name = str(entry.path).lstrip("/")
        path = PurePosixPath(name)
        if (not name.startswith("output/") or ".." in path.parts or "\\" in name
                or path.as_posix() != name or name in seen):
            raise ValueError("unsafe or duplicate output path")
        if path.suffix not in (".json", ".jsonl", ".txt"):
            continue
        seen.add(name)
        target = out / name
        if not target.resolve().is_relative_to(out.resolve()):
            raise ValueError("collection destination escapes local directory")
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            for chunk in volume.read_file(name):
                total += len(chunk)
                if total > limit:
                    raise ValueError("output byte limit exceeded")
                stream.write(chunk)
        files.append({"path": name, "bytes": target.stat().st_size, "sha256": digest(target)})
    receipt = {"status": "collected_only", "gpu_executed": False, "files": files,
               "total_bytes": total, "byte_limit": limit}
    write_once(out / "download_manifest.json", receipt)
    return receipt


def connect():
    from scripts.modal_acl_expansion import connect as checked_connect
    return checked_connect()


def read_only(control: dict, out: Path | None = None) -> dict:
    validate_control(control, fresh=False)
    modal, workspace = connect()
    volume = modal.Volume.from_name(control["volume"], create_if_missing=False)
    if load(remote_read(volume, "control.json")) != control:
        raise ValueError("remote control differs; refusing mismatched collection")
    if out is None:
        return {"status": "read_only_status", "gpu_executed": False,
            "volume": control["volume"], "output_paths": [str(e.path) for e in volume.iterdir("/output", recursive=True)]}
    out.mkdir(parents=True, exist_ok=False)
    write_once(out / "control.json", control)
    write_once(out / "workspace.json", workspace)
    return collect(volume, out, control)


def cleanup_cache(control: dict, out: Path) -> dict:
    """Delete only the newly owned cache volume; keep all claims and score evidence."""
    validate_control(control, fresh=False)
    if control["mode"] != "prepare-cache":
        raise ValueError("cleanup requires original cache ownership control")
    modal, _ = connect()
    volume = modal.Volume.from_name(control["cache_volume"], create_if_missing=False)
    if load(remote_read(volume, "control.json")) != control:
        raise ValueError("cache ownership does not match")
    ledger = modal.Volume.from_name(APPROVAL_ID, create_if_missing=False)
    if load(remote_read(ledger, "control.json")) != control:
        raise ValueError("approval ledger does not own this cache")
    modal.Volume.objects.delete(control["cache_volume"])
    receipt = {"status": "cache_deleted", "volume": control["cache_volume"],
        "approval_id": APPROVAL_ID, "deleted_utc": datetime.now(timezone.utc).isoformat(),
        "score_evidence_and_claims_preserved": True}
    write_once(out, receipt)
    return receipt


def make_image(modal, repo: Path):
    image = modal.Image.from_id(BASE_IMAGE_ID)
    for name in SOURCES:
        image = image.add_local_file(str(repo / name), remote_path="/root/" + name, copy=False)
    return image


def await_one(scorer, control: dict, out: Path, seconds: int) -> dict:
    """One spawn only; total stage time includes startup, queueing and crash retries."""
    deadline = time.time() + seconds + STARTUP_TIMEOUT
    worker_control = {**control, "absolute_deadline_unix": deadline}
    write_once(out / "allocation_window.json", worker_control)
    call = None
    try:
        with local_deadline(max(0.001, deadline - time.time())):
            call = scorer.spawn(worker_control)
            write_once(out / "calls.json", {"call_id": call.object_id})
            return call.get(timeout=max(0.001, deadline - time.time()))
    finally:
        if call is not None:
            call.cancel(terminate_containers=True)


def execute(repo: Path, control: dict, out: Path, *, public_raw=None, preparation_raw=None,
            dry_run=False) -> dict:
    verify_sources(repo, control, committed=not dry_run)
    if dry_run:
        return {"status": "preflight_only", "gpu_executed": False,
                "provider_contacted": False, "committed_source_verified": False, "control": control}
    out.mkdir(parents=True, exist_ok=False)
    write_once(out / "control.json", control)
    modal, workspace = connect()
    write_once(out / "workspace.json", workspace)
    is_cache = control["mode"] == "prepare-cache"
    if is_cache:
        # Global fixed approval identity prevents a different run ID laundering a
        # repeated allocation into the same $8 authorization.
        modal.Volume.objects.create(APPROVAL_ID, version=2, allow_existing=False)
        ledger = modal.Volume.from_name(APPROVAL_ID, create_if_missing=False)
        upload(ledger, {"control.json": canonical(control), "budget.json": canonical(control["budget"])})
    else:
        ledger = modal.Volume.from_name(APPROVAL_ID, create_if_missing=False)
        original = load(remote_read(ledger, "control.json"))
        if any(original[k] != control[k] for k in ("run_id", "cache_volume", "source_commit", "source_files_sha256", "control_source_files_sha256", "budget")):
            raise ValueError("stage differs from create-once approval ledger")
        cache = modal.Volume.from_name(control["cache_volume"], create_if_missing=False)
        if remote_read(cache, "output/prepare_receipt.json") != preparation_raw:
            raise ValueError("local preparation receipt differs from remote cache")
    suffix = "cache" if is_cache else control["stage"]
    modal.Volume.objects.create(APPROVAL_ID + "-" + suffix + "-driver", version=2, allow_existing=False)
    modal.Volume.objects.create(control["volume"], version=2, allow_existing=False)
    volume = modal.Volume.from_name(control["volume"], create_if_missing=False)
    records = {"control.json": canonical(control), "output/ownership.json": canonical({"approval_id": APPROVAL_ID})}
    if not is_cache:
        records.update({"public/pilot.json": public_raw, "output/cache_prepare_receipt.json": preparation_raw,
                        "output/stage_binding.json": canonical(control)})
    upload(volume, records)
    image = make_image(modal, repo)
    app = modal.App(control["volume"], image=image, include_source=False)
    seconds = CACHE_TIMEOUT if is_cache else STAGES[control["stage"]]["timeout"]
    resources = dict(cpu=(2, 2), memory=(8192, 8192) if is_cache else (32768, 32768),
        timeout=seconds, startup_timeout=STARTUP_TIMEOUT, max_containers=1, min_containers=0,
        buffer_containers=0, scaledown_window=SCALEDOWN, retries=0, include_source=False,
        volumes={"/cached": volume} if is_cache else {"/cached": cache, "/pilot": volume})
    if not is_cache:
        resources["gpu"] = "L40S"
    function = app.function(**resources)(remote_prepare if is_cache else remote_score)
    result = {"status": "not_started", "approval_id": APPROVAL_ID, "stage": suffix}
    try:
        with modal.enable_output(), app.run():
            result = await_one(function, control, out, seconds)
    except Exception as error:
        result = {"status": "failed_no_retry", "error_type": type(error).__name__, "error": str(error)}
        raise
    finally:
        write_once(out / "launch_result.json", result)
        collect(volume, out, control)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare-cache")
    prep.add_argument("--run-id", required=True)
    prep.add_argument("--source-commit", required=True)
    prep.add_argument("--rate-verified-utc", required=True)
    prep.add_argument("--out", type=Path, required=True)
    prep.add_argument("--dry-run", action="store_true")
    score = commands.add_parser("score-stage")
    score.add_argument("--stage", choices=STAGES, required=True)
    for name in ("cache-control", "prepare-receipt", "public-package", "stage-manifest", "out"):
        score.add_argument("--" + name, type=Path, required=True)
    score.add_argument("--policy-lock", type=Path)
    score.add_argument("--development-receipt", type=Path)
    score.add_argument("--dry-run", action="store_true")
    for name in ("collect-only", "status", "cleanup-cache"):
        sub = commands.add_parser(name)
        sub.add_argument("--control", type=Path, required=True)
        if name != "status":
            sub.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    if args.command in ("collect-only", "status"):
        result = read_only(load(args.control.read_bytes()), getattr(args, "out", None))
    elif args.command == "cleanup-cache":
        result = cleanup_cache(load(args.control.read_bytes()), args.out)
    elif args.command == "prepare-cache":
        control = cache_control(repo, args.run_id, args.source_commit, args.rate_verified_utc)
        result = execute(repo, control, args.out, dry_run=args.dry_run)
    else:
        public, preparation = args.public_package.read_bytes(), args.prepare_receipt.read_bytes()
        control = stage_control(load(args.cache_control.read_bytes()), args.stage, public,
            args.stage_manifest.read_bytes(), preparation,
            policy_lock=load(args.policy_lock.read_bytes()) if args.policy_lock else None,
            development_receipt=load(args.development_receipt.read_bytes()) if args.development_receipt else None)
        result = execute(repo, control, args.out, public_raw=public, preparation_raw=preparation, dry_run=args.dry_run)
    print(json.dumps(result, sort_keys=True))
    if args.command in ("prepare-cache", "score-stage") and not args.dry_run and result.get("status") != "complete":
        raise SystemExit("incomplete evidence; no retry is authorized")


if __name__ == "__main__":
    main()
