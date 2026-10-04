#!/usr/bin/env python3
"""One create-once 3B recovery with unchanged scientific inputs and numerical limits."""
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
from scripts.imcqa_3b_protocol_recovery import (PROTOCOL, EXECUTION_PROTOCOL, PUBLIC_SHA256,
    DIAGNOSTIC_RECEIPT_SHA256, ORIGINAL_PLAN_SHA256, PREDECESSOR_RECEIPT_SHA256)

RUN_ID = "imcqa-3b-recovery-20261004"
NUMERICAL_RUN = "imcqa-3b-numerics-20261004"
PROTOCOL_RUN = "imcqa-protocol-dev-20261004"
PRIOR_RUN = "imcqa-wait-dev-20261004"
SOURCES = ("scripts/__init__.py", "scripts/modal_imcqa_3b_protocol_recovery.py",
    "scripts/imcqa_3b_protocol_recovery.py", "scripts/imcqa_protocol_scoring.py",
    "scripts/imcqa_protocol_design.py", "scripts/imcqa_wait_scoring.py",
    "scripts/modal_acl_paired_prompt_scores.py", "scripts/acl_paired_prompt_scoring.py",
    "scripts/acl_option_scoring.py", "scripts/modal_acl_option_scores.py", "scripts/modal_acl_expansion.py",
    "scripts/jane_gpu_backend.py", "scripts/jane_qwen_backend.py", "scripts/jane_output_constraints.py",
    "configs/imcqa_protocol_pilot.json", "configs/imcqa_3b_protocol_recovery.json")


def budget_plan():
    rate = Decimal("0.00063924")
    reserve = rate*Decimal(1800+90+2)+Decimal("0.01723232")+Decimal("0.15")
    return {"schema": "imcqa-3b-recovery-budget-v1", "ceiling_usd": "1.40",
            "allocation_rate_usd_per_second": str(rate), "worker_timeout_seconds": 1800,
            "internal_deadline_seconds": 1650, "startup_timeout_seconds": 90, "scaledown_seconds": 2,
            "gpu_calls": 1, "automatic_retries": 0, "cpu_staging_allowance_usd": "0.01723232",
            "contingency_usd": "0.15", "reserved_estimate_usd": str(reserve), "invoice_verified": False}


def verify_sources(root, control):
    if control["budget"] != budget_plan() or Decimal(control["budget"]["reserved_estimate_usd"]) >= Decimal("1.40"):
        raise ValueError("recovery budget differs")
    if (control["run_id"] != RUN_ID or control["cache_run"] != CACHE_RUN
            or control["protocol_run"] != PROTOCOL_RUN or control["prior_run"] != PRIOR_RUN
            or control["numerical_run"] != NUMERICAL_RUN or control["public_input_sha256"] != PUBLIC_SHA256
            or control["diagnostic_receipt_sha256"] != DIAGNOSTIC_RECEIPT_SHA256):
        raise ValueError("recovery run identities differ")
    if {name: digest(root/name) for name in SOURCES} != control["source_files_sha256"]:
        raise ValueError("recovery sources differ from frozen control")
    config = json.loads((root/"configs/imcqa_3b_protocol_recovery.json").read_bytes())
    if (config["protocol"] != PROTOCOL or config["public_input_sha256"] != PUBLIC_SHA256
            or config["execution_protocol"] != EXECUTION_PROTOCOL or config["batch_size"] != 2 or config["cached"] is not True
            or config["diagnostic_receipt_sha256"] != DIAGNOSTIC_RECEIPT_SHA256
            or config["original_plan_sha256"] != ORIGINAL_PLAN_SHA256
            or config["predecessor_receipt_sha256"] != PREDECESSOR_RECEIPT_SHA256
            or config["budget"] != budget_plan()):
        raise ValueError("recovery config differs from bounded implementation")


def remote_recovery(control):
    import modal
    verify_sources(Path("/opt/scoring"), control)
    if time.time() >= control["absolute_deadline"]:
        raise TimeoutError("recovery allocation window expired")
    modal.Volume.objects.create(RUN_ID+"-qwen3b-claim", version=2, allow_existing=False)
    volume = modal.Volume.from_name(RUN_ID, create_if_missing=False)
    volume.reload()
    output = Path("/recovery/output")
    started = time.monotonic()
    write_json(output/"qwen3b_claim.json", {"source_commit": control["source_commit"], "started_unix": time.time()})
    volume.commit()
    try:
        preparation = verify_prepare(Path("/cached/output/prepare_receipt.json").read_bytes())
        sidecar = json.loads(Path("/cached/models/qwen3b_expected_model_hashes.json").read_bytes())
        if any(sidecar.get(key) != preparation["model_receipts"]["qwen3b"][key]
               for key in ("model", "revision", "model_files_sha256")):
            raise ValueError("cached model identity differs from preparation")
        from scripts.imcqa_3b_protocol_recovery import run_recovery
        def progress(update):
            from scripts.modal_acl_expansion import replace_progress
            replace_progress(output/"progress.json", update)
            volume.commit()
            print(json.dumps(update), flush=True)
        return run_recovery("qwen3b", Path("/protocol/public/pilot.json"), control["public_input_sha256"],
            Path("/cached/models"), output/"qwen3b", prior_root=Path("/prior"), protocol_root=Path("/protocol"),
            numerical_root=Path("/numerical"), source_commit=control["source_commit"], max_seconds=1650, progress=progress)
    finally:
        write_json(output/"qwen3b_allocation_receipt.json", {"elapsed_seconds": time.monotonic()-started,
            "allocation_rate_usd_per_second": control["budget"]["allocation_rate_usd_per_second"], "invoice_verified": False})
        volume.commit()


def launch(repo: Path, out: Path, source_commit: str):
    from scripts.modal_acl_expansion import connect, verify_source_commit
    if len(source_commit) != 40 or any(c not in "0123456789abcdef" for c in source_commit):
        raise ValueError("exact committed source identity required")
    control = {"run_id": RUN_ID, "cache_run": CACHE_RUN, "protocol_run": PROTOCOL_RUN, "prior_run": PRIOR_RUN,
        "source_commit": source_commit, "public_input_sha256": PUBLIC_SHA256, "prepare_receipt_sha256": PREPARE_SHA256,
        "numerical_run": NUMERICAL_RUN, "execution_protocol": EXECUTION_PROTOCOL,
        "diagnostic_receipt_sha256": DIAGNOSTIC_RECEIPT_SHA256, "original_plan_sha256": ORIGINAL_PLAN_SHA256,
        "source_files_sha256": {name: digest(repo/name) for name in SOURCES}, "budget": budget_plan(),
        "created_utc": datetime.now(timezone.utc).isoformat(), "expected_new_rows": 4032, "reused_rows": 1200}
    verify_sources(repo, control)
    verify_source_commit(repo, source_commit, control["source_files_sha256"])
    modal, workspace = connect()
    cache = modal.Volume.from_name(CACHE_RUN, create_if_missing=False)
    prior = modal.Volume.from_name(PRIOR_RUN, create_if_missing=False)
    protocol = modal.Volume.from_name(PROTOCOL_RUN, create_if_missing=False)
    numerical = modal.Volume.from_name(NUMERICAL_RUN, create_if_missing=False)
    public_raw = b"".join(protocol.read_file("public/pilot.json"))
    if hashlib.sha256(public_raw).hexdigest() != PUBLIC_SHA256:
        raise ValueError("original public input changed before allocation")
    diagnostic_raw = b"".join(numerical.read_file("output/qwen3b/receipt.json"))
    if hashlib.sha256(diagnostic_raw).hexdigest() != DIAGNOSTIC_RECEIPT_SHA256:
        raise ValueError("numerical diagnostic receipt changed before allocation")
    prepare_raw = b"".join(cache.read_file("output/prepare_receipt.json"))
    verify_prepare(prepare_raw)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out/"control.json", control); write_json(out/"workspace.json", workspace)
    (out/"public.json").write_bytes(public_raw)
    (out/"selected_diagnostic_receipt.json").write_bytes(diagnostic_raw)
    modal.Volume.objects.create(RUN_ID, version=2, allow_existing=False)
    volume = modal.Volume.from_name(RUN_ID, create_if_missing=False)
    with volume.batch_upload(force=False) as upload:
        upload.put_file(io.BytesIO(json.dumps(control).encode()), "/control.json")
        upload.put_file(io.BytesIO(prepare_raw), "/output/cache_prepare_receipt.json")
    image = modal.Image.debian_slim(python_version="3.11").pip_install(
        "torch==2.6.0", "transformers==4.51.3", "tokenizers==0.21.1", "safetensors==0.5.3",
        "huggingface-hub==0.30.2", "accelerate==1.6.0", "modal==1.6.0", "numpy==2.2.4",
        "jinja2==3.1.6", "lm-format-enforcer==0.11.3", "interegular==0.3.3")
    for name in SOURCES:
        image = image.add_local_file(str(repo/name), remote_path="/opt/scoring/"+name, copy=True)
    image = image.env({"PYTHONPATH": "/opt/scoring", "PYTHONUNBUFFERED": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
                      "TOKENIZERS_PARALLELISM": "false", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    app = modal.App(RUN_ID, image=image, include_source=False)
    worker = app.function(gpu="L40S", cpu=(2, 2), memory=(32768, 32768), timeout=1800, startup_timeout=90,
        max_containers=1, min_containers=0, buffer_containers=0, scaledown_window=2, retries=0, include_source=False,
        volumes={"/cached": cache, "/protocol": protocol, "/prior": prior, "/numerical": numerical, "/recovery": volume})(remote_recovery)
    result, call = {"run_id": RUN_ID}, None
    try:
        with modal.enable_output(), app.run():
            allocation = {**control, "absolute_deadline": time.time()+1890}
            write_json(out/"allocation_window.json", allocation)
            try:
                call = worker.spawn(allocation)
                write_json(out/"calls.json", {"qwen3b": call.object_id})
                result["qwen3b"] = call.get(timeout=max(1, allocation["absolute_deadline"]-time.time()))
            except Exception as error:
                if call is not None:
                    call.cancel(terminate_containers=True)
                result["qwen3b"] = {"status": "provider_error", "error_type": type(error).__name__, "error": str(error)}
    finally:
        collect(volume, out)
        write_json(out/"launch_result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = launch(Path(__file__).resolve().parents[1], args.out, args.source_commit)
    print(json.dumps(result, sort_keys=True))
    if result.get("qwen3b", {}).get("status") != "complete":
        raise SystemExit("recovery incomplete; preserved evidence requires review")


if __name__ == "__main__":
    main()
