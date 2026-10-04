#!/usr/bin/env python3
"""One frozen 7B transfer, bounded to a single create-once cached-model worker."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import io
import json
from pathlib import Path
import time

from scripts.modal_acl_option_scores import digest, write_json
from scripts.modal_acl_paired_prompt_scores import CACHE_RUN, PREPARE_SHA256, verify_prepare, collect
from scripts.imcqa_transfer_design import validate_public_package, PROTOCOL

RUN_ID = "imcqa-frozen-transfer-20261004"
MODEL_LIMITS = {"qwen7b": {"timeout": 3000, "deadline": 2880}}
SOURCES = ("scripts/__init__.py", "scripts/modal_imcqa_transfer.py", "scripts/imcqa_transfer_scoring.py",
    "scripts/imcqa_transfer_design.py", "scripts/imcqa_protocol_scoring.py", "scripts/imcqa_protocol_design.py",
    "scripts/imcqa_wait_scoring.py", "scripts/modal_acl_paired_prompt_scores.py", "scripts/acl_paired_prompt_scoring.py",
    "scripts/acl_option_scoring.py", "scripts/modal_acl_option_scores.py", "scripts/modal_acl_expansion.py",
    "scripts/jane_gpu_backend.py", "scripts/jane_qwen_backend.py", "scripts/jane_output_constraints.py",
    "configs/imcqa_protocol_pilot.json", "configs/imcqa_frozen_transfer.json")


def budget_plan() -> dict:
    rate = Decimal("0.00063924")
    reserve = rate * sum(v["timeout"] + 92 for v in MODEL_LIMITS.values()) + Decimal("0.10") + Decimal("0.01723232")
    return {"schema": "imcqa-transfer-budget-v1", "ceiling_usd": "4.00",
        "allocation_rate_usd_per_second": str(rate), "model_limits": {tag:dict(v) for tag,v in MODEL_LIMITS.items()},
        "startup_timeout_seconds": 90, "scaledown_seconds": 2, "gpu_calls": 1, "cpu_calls": 0,
        "automatic_retries": 0, "cpu_staging_allowance_usd": "0.01723232", "contingency_usd": "0.10",
        "reserved_estimate_usd": str(reserve), "invoice_verified": False}


def validate_plan(plan: dict) -> None:
    if plan != budget_plan() or Decimal(plan["reserved_estimate_usd"]) > Decimal("4"):
        raise ValueError("pilot must match the four-dollar allocation plan")


def verify_sources(root: Path, control: dict) -> None:
    validate_plan(control["budget"])
    if (control["run_id"] != RUN_ID or control["cache_run"] != CACHE_RUN
            or control["prepare_receipt_sha256"] != PREPARE_SHA256):
        raise ValueError("unexpected pilot/cache identity")
    if {name: digest(root/name) for name in SOURCES} != control["source_files_sha256"]:
        raise ValueError("pilot sources differ from frozen control")
    config = json.loads((root/"configs/imcqa_frozen_transfer.json").read_bytes())
    from scripts import acl_option_scoring as base
    from scripts import imcqa_transfer_scoring as scoring
    expected_execution = {"contexts_per_model": 8000, "new_contexts_per_model": 8000,
        "reused_contexts_per_model": 0, "model_tags": ["qwen7b"], "batch_size": scoring.BATCH_SIZE,
        "benchmark_rows": scoring.BENCHMARK_ROWS, "worker_timeout_seconds": MODEL_LIMITS["qwen7b"]["timeout"],
        "internal_deadline_seconds": scoring.MAX_SECONDS, "startup_timeout_seconds": 90,
        "scaledown_seconds": 2, "automatic_retries": 0}
    expected_numerical = {"raw_logit_atol": base.FP32_ATOL, "raw_logit_rtol": base.FP32_RTOL,
        "probability_atol": .001, "require_zero_action_and_candidate_argmax_changes": True,
        "active_only_episodes": 16, "production_offsets": scoring.production_offsets(8000)}
    if (config["protocol"] != PROTOCOL or config["execution"] != expected_execution
            or config["numerical"] != expected_numerical or config["budget"] != budget_plan()
            or config["contexts_per_model"] != 8000 or config["models"] != {
                "qwen7b": {"model": base.MODELS["qwen7b"], "revision": base.PINNED_MODELS[base.MODELS["qwen7b"]]}}):
        raise ValueError("semantic configuration differs from bounded implementation")


def verify_public(raw: bytes, expected_hash: str) -> dict:
    if hashlib.sha256(raw).hexdigest() != expected_hash:
        raise ValueError("public package hash differs")
    from scripts.acl_option_scoring import load_json
    package = load_json(raw)
    jobs = validate_public_package(package)
    if len(jobs) != 8000 or any(job["execution"] != "new" for job in jobs):
        raise ValueError("production transfer must contain exactly 8000 new contexts")
    return package


def remote_score(tag: str, control: dict) -> dict:
    import modal
    verify_sources(Path("/opt/scoring"), control)
    if tag not in MODEL_LIMITS:
        raise ValueError("unknown model tag")
    if time.time() >= control["absolute_deadlines"][tag]:
        raise TimeoutError("allocation window expired")
    # Separate atomic claim prevents provider infrastructure replay from renting
    # a second scoring worker even when application-level retries are disabled.
    modal.Volume.objects.create(RUN_ID+"-"+tag+"-claim", version=2, allow_existing=False)
    volume = modal.Volume.from_name(RUN_ID, create_if_missing=False)
    volume.reload()
    cache, output = Path("/cached"), Path("/pilot/output")
    started = time.monotonic()
    write_json(output/f"{tag}_claim.json", {"model_tag":tag, "source_commit":control["source_commit"], "started_unix":time.time()})
    volume.commit()
    try:
        preparation = verify_prepare((cache/"output/prepare_receipt.json").read_bytes())
        sidecar = json.loads((cache/"models"/f"{tag}_expected_model_hashes.json").read_bytes())
        if any(sidecar.get(k) != preparation["model_receipts"][tag][k] for k in ("model", "revision", "model_files_sha256")):
            raise ValueError("cached model sidecar differs from frozen preparation")
        public_path = Path("/pilot/public/pilot.json")
        verify_public(public_path.read_bytes(), control["input_sha256"])
        from scripts.imcqa_transfer_scoring import run_scoring
        def progress(update):
            from scripts.modal_acl_expansion import replace_progress
            replace_progress(output/tag/"progress.json", update)
            volume.commit()
            print(json.dumps({"model_tag":tag, **update}), flush=True)
        return run_scoring(tag, public_path, control["input_sha256"], cache/"models", output/tag,
            source_commit=control["source_commit"], max_seconds=control["budget"]["model_limits"][tag]["deadline"], progress=progress)
    finally:
        write_json(output/f"{tag}_allocation_receipt.json", {"model_tag":tag,
            "elapsed_seconds":time.monotonic()-started,
            "allocation_rate_usd_per_second":control["budget"]["allocation_rate_usd_per_second"], "invoice_verified":False})
        volume.commit()


def remote_score_7b(control):
    return remote_score("qwen7b", control)


def launch(repo: Path, public_path: Path, out: Path, source_commit: str) -> dict:
    from scripts.modal_acl_expansion import connect, verify_source_commit
    if len(source_commit) != 40 or any(c not in "0123456789abcdef" for c in source_commit):
        raise ValueError("exact committed source identity required")
    raw = public_path.read_bytes()
    input_hash = hashlib.sha256(raw).hexdigest()
    package = verify_public(raw, input_hash)
    control = {"run_id":RUN_ID, "cache_run":CACHE_RUN, "source_commit":source_commit,
        "input_sha256":input_hash, "prepare_receipt_sha256":PREPARE_SHA256,
        "source_files_sha256":{name:digest(repo/name) for name in SOURCES},
        "budget":budget_plan(), "created_utc":datetime.now(timezone.utc).isoformat(),
        "estimand":"frozen exploratory transfer of plain-MCQA external stopping and native WAIT on 100 protocol-unexposed development questions",
        "production_contexts_per_model":len(package["jobs"])}
    verify_sources(repo, control)
    verify_source_commit(repo, source_commit, control["source_files_sha256"])
    modal, workspace = connect()
    cache = modal.Volume.from_name(CACHE_RUN, create_if_missing=False)
    prepare_raw = b"".join(cache.read_file("output/prepare_receipt.json"))
    verify_prepare(prepare_raw)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out/"control.json", control); write_json(out/"workspace.json", workspace)
    (out/"pilot.json").write_bytes(raw)
    modal.Volume.objects.create(RUN_ID, version=2, allow_existing=False)
    volume = modal.Volume.from_name(RUN_ID, create_if_missing=False)
    with volume.batch_upload(force=False) as upload:
        upload.put_file(io.BytesIO(json.dumps(control).encode()), "/control.json")
        upload.put_file(io.BytesIO(prepare_raw), "/output/cache_prepare_receipt.json")
        upload.put_file(io.BytesIO(raw), "/public/pilot.json")
    image = modal.Image.debian_slim(python_version="3.11").pip_install(
        "torch==2.6.0", "transformers==4.51.3", "tokenizers==0.21.1", "safetensors==0.5.3",
        "huggingface-hub==0.30.2", "accelerate==1.6.0", "modal==1.6.0", "numpy==2.2.4",
        "jinja2==3.1.6", "lm-format-enforcer==0.11.3", "interegular==0.3.3")
    for name in SOURCES:
        image = image.add_local_file(str(repo/name), remote_path="/opt/scoring/"+name, copy=True)
    image = image.env({"PYTHONPATH":"/opt/scoring", "PYTHONUNBUFFERED":"1", "HF_HUB_DISABLE_TELEMETRY":"1",
        "TOKENIZERS_PARALLELISM":"false", "HF_HUB_OFFLINE":"1", "TRANSFORMERS_OFFLINE":"1"})
    app = modal.App(RUN_ID, image=image, include_source=False)
    scorers = {}
    for tag, fn in (("qwen7b",remote_score_7b),):
        scorers[tag] = app.function(gpu="L40S", cpu=(2,2), memory=(32768,32768),
            timeout=MODEL_LIMITS[tag]["timeout"], startup_timeout=90,
            max_containers=1, min_containers=0, buffer_containers=0, scaledown_window=2,
            retries=0, include_source=False, volumes={"/cached":cache,"/pilot":volume})(fn)
    result, calls = {"run_id":RUN_ID, "models":{}}, {}
    try:
        with modal.enable_output(), app.run():
            now = time.time()
            worker_control = {**control,"absolute_deadlines":{tag:now+limit["timeout"]+90 for tag,limit in MODEL_LIMITS.items()}}
            write_json(out/"allocation_window.json",worker_control)
            try:
                for tag, scorer in scorers.items(): calls[tag] = scorer.spawn(worker_control)
                write_json(out/"calls.json",{tag:call.object_id for tag,call in calls.items()})
                for tag, call in calls.items():
                    try:
                        result["models"][tag] = call.get(timeout=max(1,worker_control["absolute_deadlines"][tag]-time.time()))
                    except Exception as error:
                        call.cancel(terminate_containers=True)
                        result["models"][tag] = {"status":"provider_error", "error_type":type(error).__name__, "error":str(error)}
            finally:
                for tag, call in calls.items():
                    if tag not in result["models"]: call.cancel(terminate_containers=True)
    finally:
        collect(volume,out)
        write_json(out/"launch_result.json",result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-package", required=True, type=Path)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    result = launch(Path(__file__).resolve().parents[1],args.public_package,args.out,args.source_commit)
    print(json.dumps(result,sort_keys=True))
    if len(result["models"]) != 1 or any(r.get("status") != "complete" for r in result["models"].values()):
        raise SystemExit("transfer incomplete; preserved evidence requires review")


if __name__ == "__main__":
    main()
