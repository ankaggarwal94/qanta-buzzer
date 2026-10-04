#!/usr/bin/env python3
"""One create-once L40S numerical diagnostic, reserved below fifty cents."""
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
from scripts.imcqa_3b_numerical_diagnostic import PROTOCOL, SCHEMA

RUN_ID = "imcqa-3b-numerics-20261004"
PROTOCOL_RUN = "imcqa-protocol-dev-20261004"
PRIOR_RUN = "imcqa-wait-dev-20261004"
MANIFEST_SHA256 = "a2cf2745e7291f43eb29ed7f7676e01f8edbf7b45326d59c337b32fa8aebc40a"
SOURCES = ("scripts/__init__.py", "scripts/modal_imcqa_3b_numerical_diagnostic.py",
    "scripts/imcqa_3b_numerical_diagnostic.py", "scripts/imcqa_protocol_scoring.py",
    "scripts/imcqa_protocol_design.py", "scripts/imcqa_wait_scoring.py",
    "scripts/modal_acl_paired_prompt_scores.py", "scripts/acl_paired_prompt_scoring.py",
    "scripts/acl_option_scoring.py", "scripts/modal_acl_option_scores.py", "scripts/modal_acl_expansion.py",
    "scripts/jane_gpu_backend.py", "scripts/jane_qwen_backend.py", "scripts/jane_output_constraints.py",
    "configs/imcqa_protocol_pilot.json", "configs/imcqa_3b_numerical_diagnostic.json")


def budget_plan():
    rate = Decimal("0.00063924")
    reserve = rate*Decimal(360+90+2)+Decimal("0.01723232")+Decimal("0.10")
    return {"schema": "imcqa-3b-numerics-budget-v1", "ceiling_usd": "0.50",
            "allocation_rate_usd_per_second": str(rate), "worker_timeout_seconds": 360,
            "internal_deadline_seconds": 300, "startup_timeout_seconds": 90, "scaledown_seconds": 2,
            "gpu_calls": 1, "automatic_retries": 0, "cpu_staging_allowance_usd": "0.01723232",
            "contingency_usd": "0.10", "reserved_estimate_usd": str(reserve), "invoice_verified": False}


def verify_manifest(raw):
    if hashlib.sha256(raw).hexdigest() != MANIFEST_SHA256:
        raise ValueError("diagnostic manifest hash differs")
    manifest = json.loads(raw)
    if (manifest["schema_version"] != SCHEMA or manifest["protocol"] != PROTOCOL
            or manifest["expected_contexts"] != 20 or manifest["expected_logical_evaluations"] != 200
            or manifest["maximum_model_forward_calls"] != 117):
        raise ValueError("diagnostic manifest semantics differ")
    return manifest


def verify_sources(root, control):
    if control["budget"] != budget_plan() or Decimal(control["budget"]["reserved_estimate_usd"]) >= Decimal(".50"):
        raise ValueError("diagnostic budget differs")
    if (control["run_id"] != RUN_ID or control["cache_run"] != CACHE_RUN
            or control["protocol_run"] != PROTOCOL_RUN or control["prior_run"] != PRIOR_RUN
            or control["manifest_sha256"] != MANIFEST_SHA256):
        raise ValueError("diagnostic run identities differ")
    if {name: digest(root/name) for name in SOURCES} != control["source_files_sha256"]:
        raise ValueError("diagnostic sources differ from frozen control")
    config = json.loads((root/"configs/imcqa_3b_numerical_diagnostic.json").read_bytes())
    if (config["protocol"] != PROTOCOL or config["manifest_sha256"] != MANIFEST_SHA256
            or config["logical_evaluations"] != 200 or config["model_forward_calls"] != 117
            or config["budget"] != budget_plan()):
        raise ValueError("diagnostic config differs from bounded implementation")


def remote_diagnostic(control):
    import modal
    verify_sources(Path("/opt/scoring"), control)
    if time.time() >= control["absolute_deadline"]:
        raise TimeoutError("diagnostic allocation window expired")
    modal.Volume.objects.create(RUN_ID+"-qwen3b-claim", version=2, allow_existing=False)
    volume = modal.Volume.from_name(RUN_ID, create_if_missing=False)
    volume.reload()
    output = Path("/diagnostic/output")
    started = time.monotonic()
    write_json(output/"qwen3b_claim.json", {"source_commit": control["source_commit"], "started_unix": time.time()})
    volume.commit()
    try:
        preparation = verify_prepare(Path("/cached/output/prepare_receipt.json").read_bytes())
        sidecar = json.loads(Path("/cached/models/qwen3b_expected_model_hashes.json").read_bytes())
        if any(sidecar.get(key) != preparation["model_receipts"]["qwen3b"][key]
               for key in ("model", "revision", "model_files_sha256")):
            raise ValueError("cached model identity differs from preparation")
        from scripts.imcqa_3b_numerical_diagnostic import run_diagnostic
        def progress(update):
            from scripts.modal_acl_expansion import replace_progress
            replace_progress(output/"progress.json", update)
            volume.commit()
            print(json.dumps(update), flush=True)
        return run_diagnostic(Path("/diagnostic/public/manifest.json"), control["manifest_sha256"],
            Path("/protocol"), Path("/prior"), Path("/cached/models"), output/"qwen3b",
            max_seconds=300, progress=progress)
    finally:
        write_json(output/"qwen3b_allocation_receipt.json", {"elapsed_seconds": time.monotonic()-started,
            "allocation_rate_usd_per_second": control["budget"]["allocation_rate_usd_per_second"], "invoice_verified": False})
        volume.commit()


def launch(repo: Path, manifest_path: Path, out: Path, source_commit: str):
    from scripts.modal_acl_expansion import connect, verify_source_commit
    if len(source_commit) != 40 or any(c not in "0123456789abcdef" for c in source_commit):
        raise ValueError("exact committed source identity required")
    raw = manifest_path.read_bytes()
    verify_manifest(raw)
    control = {"run_id": RUN_ID, "cache_run": CACHE_RUN, "protocol_run": PROTOCOL_RUN, "prior_run": PRIOR_RUN,
        "source_commit": source_commit, "manifest_sha256": MANIFEST_SHA256, "prepare_receipt_sha256": PREPARE_SHA256,
        "source_files_sha256": {name: digest(repo/name) for name in SOURCES}, "budget": budget_plan(),
        "created_utc": datetime.now(timezone.utc).isoformat(), "production_rows": 0}
    verify_sources(repo, control)
    verify_source_commit(repo, source_commit, control["source_files_sha256"])
    modal, workspace = connect()
    cache = modal.Volume.from_name(CACHE_RUN, create_if_missing=False)
    prior = modal.Volume.from_name(PRIOR_RUN, create_if_missing=False)
    protocol = modal.Volume.from_name(PROTOCOL_RUN, create_if_missing=False)
    prepare_raw = b"".join(cache.read_file("output/prepare_receipt.json"))
    verify_prepare(prepare_raw)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out/"control.json", control); write_json(out/"workspace.json", workspace)
    (out/"manifest.json").write_bytes(raw)
    modal.Volume.objects.create(RUN_ID, version=2, allow_existing=False)
    volume = modal.Volume.from_name(RUN_ID, create_if_missing=False)
    with volume.batch_upload(force=False) as upload:
        upload.put_file(io.BytesIO(json.dumps(control).encode()), "/control.json")
        upload.put_file(io.BytesIO(prepare_raw), "/output/cache_prepare_receipt.json")
        upload.put_file(io.BytesIO(raw), "/public/manifest.json")
    image = modal.Image.debian_slim(python_version="3.11").pip_install(
        "torch==2.6.0", "transformers==4.51.3", "tokenizers==0.21.1", "safetensors==0.5.3",
        "huggingface-hub==0.30.2", "accelerate==1.6.0", "modal==1.6.0", "numpy==2.2.4",
        "jinja2==3.1.6", "lm-format-enforcer==0.11.3", "interegular==0.3.3")
    for name in SOURCES:
        image = image.add_local_file(str(repo/name), remote_path="/opt/scoring/"+name, copy=True)
    image = image.env({"PYTHONPATH": "/opt/scoring", "PYTHONUNBUFFERED": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
                      "TOKENIZERS_PARALLELISM": "false", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    app = modal.App(RUN_ID, image=image, include_source=False)
    worker = app.function(gpu="L40S", cpu=(2, 2), memory=(32768, 32768), timeout=360, startup_timeout=90,
        max_containers=1, min_containers=0, buffer_containers=0, scaledown_window=2, retries=0, include_source=False,
        volumes={"/cached": cache, "/protocol": protocol, "/prior": prior, "/diagnostic": volume})(remote_diagnostic)
    result, call = {"run_id": RUN_ID}, None
    try:
        with modal.enable_output(), app.run():
            allocation = {**control, "absolute_deadline": time.time()+450}
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
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = launch(Path(__file__).resolve().parents[1], args.manifest, args.out, args.source_commit)
    print(json.dumps(result, sort_keys=True))
    if result.get("qwen3b", {}).get("status") != "complete":
        raise SystemExit("diagnostic is not usable; preserved evidence requires review")


if __name__ == "__main__":
    main()
