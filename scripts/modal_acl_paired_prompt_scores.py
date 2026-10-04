"""One bounded paired-prompt scoring run using the verified existing model cache.

Only public menus and pinned model bytes reach inference. No model downloads,
generation, automatic retries, or precision fallback are performed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import io
import json
from pathlib import Path
import time

from scripts.modal_acl_option_scores import digest, write_json, safe_output_path

RUN_ID = "acl5000-paired-prompts-20261003"
RECOVERY_RUN = "acl5000-paired-prompts-cache1-20261003"
CACHE_RUN = "acl5000-option-scores-20261003"
PREPARE_SHA256 = "95cc6dcd2e99e74597c95c1bf4580457dd843a01053267cf3888b21461a7a1e0"
INPUT_SHA256 = "9db13301d928cd31dc54c97f0c5cfd88b9bc25774ceb56027c032c52e4afb043"
MODEL_LIMITS = {"qwen3b": {"timeout": 850, "deadline": 780},
                "qwen7b": {"timeout": 1450, "deadline": 1350}}
RECOVERY_LIMITS = {"qwen3b": {"timeout": 650, "deadline": 600},
                   "qwen7b": {"timeout": 1350, "deadline": 1300}}
PRIOR_RECEIPTS = {
    "qwen3b_allocation_receipt.json": "176720795c861337b8899befc9eb50b028f4686255949019be86e774f1ad2c80",
    "qwen7b_allocation_receipt.json": "f7957fa8355c30ac78d106539810063d793912d219d131e1731db47839556bb4",
    "qwen3b/receipt.json": "4fde3481bca75e921f8438a6a5451588afdc5897e187ab56b80fb00eb7ee2238",
    "qwen7b/receipt.json": "ba3956ee0a90cc57ffbfa1b294775e23436e597a53e7b36e17da9135d00c057b",
}
SOURCES = ("scripts/__init__.py", "scripts/modal_acl_paired_prompt_scores.py",
           "scripts/acl_paired_prompt_scoring.py", "scripts/acl_option_scoring.py",
           "scripts/modal_acl_option_scores.py", "scripts/modal_acl_expansion.py",
           "scripts/jane_gpu_backend.py", "scripts/jane_qwen_backend.py",
           "scripts/jane_output_constraints.py")


def budget_plan() -> dict:
    rate = Decimal("0.00063924")
    seconds = sum(row["timeout"] + 90 + 2 for row in MODEL_LIMITS.values())
    reserve = rate * seconds + Decimal("0.40")
    return {"schema": "acl-paired-prompt-budget-v1", "ceiling_usd": "2",
            "allocation_rate_usd_per_second": str(rate),
            "model_limits": {tag: dict(row) for tag, row in MODEL_LIMITS.items()},
            "startup_timeout_seconds": 90, "scaledown_seconds": 2,
            "gpu_calls": 2, "cpu_calls": 0, "automatic_retries": 0,
            "contingency_usd": "0.40", "reserved_estimate_usd": str(reserve),
            "invoice_verified": False}


def validate_plan(plan: dict) -> None:
    if plan not in (budget_plan(), recovery_budget_plan()) or Decimal(plan["reserved_estimate_usd"]) > Decimal("2"):
        raise ValueError("paired scoring must match the two-dollar allocation plan")


def recovery_budget_plan() -> dict:
    """Bind one efficiency recovery to completed receipts and the original cap."""
    rate = Decimal("0.00063924")
    prior_seconds = Decimal("75.41704748299999") + Decimal("94.419165222")
    prior_reserve = rate * (prior_seconds + 2 * 92)
    new_reserve = rate * sum(v["timeout"] + 92 for v in RECOVERY_LIMITS.values())
    return {"schema": "acl-paired-prefix-cache-recovery-budget-v1", "ceiling_usd": "2",
            "allocation_rate_usd_per_second": str(rate),
            "model_limits": {tag: dict(row) for tag, row in RECOVERY_LIMITS.items()},
            "startup_timeout_seconds": 90, "scaledown_seconds": 2,
            "gpu_calls": 2, "cpu_calls": 0, "automatic_retries": 0,
            "prior_gpu_observed_seconds": str(prior_seconds),
            "prior_reserved_estimate_usd": str(prior_reserve),
            "new_reserved_estimate_usd": str(new_reserve), "contingency_usd": "0.35",
            "reserved_estimate_usd": str(prior_reserve + new_reserve + Decimal("0.35")),
            "invoice_verified": False}


def verify_prior_receipts(read_bytes) -> None:
    for name, expected in PRIOR_RECEIPTS.items():
        if hashlib.sha256(read_bytes("output/" + name)).hexdigest() != expected:
            raise ValueError("initial paired-run receipt mismatch: " + name)


def verify_sources(root: Path, control: dict) -> None:
    validate_plan(control["budget"])
    recovery = control.get("shared_prefix_cache", False)
    if type(recovery) is not bool or control["run_id"] != (RECOVERY_RUN if recovery else RUN_ID):
        raise ValueError("unexpected scoring run identity or cache mode")
    if control["budget"] != (recovery_budget_plan() if recovery else budget_plan()):
        raise ValueError("budget does not match the reviewed execution mode")
    if {name: digest(root / name) for name in SOURCES} != control["source_files_sha256"]:
        raise ValueError("paired scoring source differs from the committed control")


def verify_prepare(raw: bytes) -> dict:
    if hashlib.sha256(raw).hexdigest() != PREPARE_SHA256:
        raise ValueError("verified cache preparation receipt changed")
    return json.loads(raw)


def remote_score(tag: str, control: dict) -> dict:
    import modal
    verify_sources(Path("/opt/scoring"), control)
    if tag not in MODEL_LIMITS:
        raise ValueError("unknown model")
    if time.time() >= control["absolute_deadlines"][tag]:
        raise TimeoutError("allocation window expired")
    # Atomic provider-side create prevents a second physical scoring invocation,
    # including an infrastructure replay that is independent of retries=0.
    modal.Volume.objects.create(control["run_id"] + "-" + tag + "-claim", version=2,
                                allow_existing=False)
    volume = modal.Volume.from_name(control["run_id"], create_if_missing=False)
    volume.reload()
    cache_root, output = Path("/cached"), Path("/paired/output")
    started = time.monotonic()
    write_json(output / f"{tag}_claim.json", {"model_tag": tag,
        "source_commit": control["source_commit"], "started_unix": time.time()})
    volume.commit()
    try:
        if control.get("shared_prefix_cache"):
            verify_prior_receipts(lambda name: (Path("/prior") / name).read_bytes())
        preparation = verify_prepare((cache_root / "output/prepare_receipt.json").read_bytes())
        frozen_model = preparation["model_receipts"][tag]
        sidecar = json.loads((cache_root / "models" / f"{tag}_expected_model_hashes.json").read_bytes())
        if any(sidecar.get(key) != frozen_model[key]
               for key in ("model", "revision", "model_files_sha256")):
            raise ValueError("cached model sidecar differs from frozen preparation receipt")
        if digest(cache_root / "public/main_choices_only.json") != INPUT_SHA256:
            raise ValueError("cached public menus failed frozen hash")
        from scripts.acl_paired_prompt_scoring import run_scoring
        def progress(update):
            from scripts.modal_acl_expansion import replace_progress
            replace_progress(output / tag / "progress.json", update)
            volume.commit()
            print(json.dumps({"model_tag": tag, **update}), flush=True)
        return run_scoring(tag, cache_root / "public/main_choices_only.json",
            cache_root / "models", output / tag,
            max_seconds=control["budget"]["model_limits"][tag]["deadline"],
            batch_size=128 if control.get("shared_prefix_cache") else 32,
            shared_prefix_cache=control.get("shared_prefix_cache", False),
            progress=progress)
    finally:
        write_json(output / f"{tag}_allocation_receipt.json", {
            "model_tag": tag, "elapsed_seconds": time.monotonic() - started,
            "allocation_rate_usd_per_second": control["budget"]["allocation_rate_usd_per_second"],
            "invoice_verified": False})
        volume.commit()


def remote_score_3b(control: dict) -> dict:
    return remote_score("qwen3b", control)


def remote_score_7b(control: dict) -> dict:
    return remote_score("qwen7b", control)


def collect(volume, out: Path) -> dict:
    files, total = [], 0
    for entry in volume.iterdir("/output", recursive=True):
        name = str(entry.path).lstrip("/")
        if not name.endswith((".json", ".jsonl", ".txt")):
            continue
        name = safe_output_path(name)
        target = out / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            for chunk in volume.read_file(name):
                total += len(chunk)
                if total > 160 * 1024**2:
                    raise ValueError("bounded collection exceeds 160 MiB")
                stream.write(chunk)
        files.append({"path": name, "bytes": target.stat().st_size,
                      "sha256": digest(target)})
    report = {"files": files, "total_bytes": total}
    write_json(out / "download_manifest.json", report)
    return report


def launch(repo: Path, out: Path, source_commit: str, recovery: bool = False) -> dict:
    from scripts.modal_acl_expansion import connect, verify_source_commit
    if len(source_commit) != 40 or any(c not in "0123456789abcdef" for c in source_commit):
        raise ValueError("exact source commit required")
    run_id = RECOVERY_RUN if recovery else RUN_ID
    control = {"run_id": run_id, "cache_run": CACHE_RUN, "source_commit": source_commit,
        "input_sha256": INPUT_SHA256, "prepare_receipt_sha256": PREPARE_SHA256,
        "source_files_sha256": {name: digest(repo / name) for name in SOURCES},
        "budget": recovery_budget_plan() if recovery else budget_plan(),
        "shared_prefix_cache": recovery,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "estimand": "paired prompt effect on A-D softmax conditional on fixed answer prefix"}
    verify_sources(repo, control)
    verify_source_commit(repo, source_commit, control["source_files_sha256"])
    modal, workspace = connect()
    cache = modal.Volume.from_name(CACHE_RUN, create_if_missing=False)
    prepare_raw = b"".join(cache.read_file("output/prepare_receipt.json"))
    verify_prepare(prepare_raw)
    prior = None
    if recovery:
        prior = modal.Volume.from_name(RUN_ID, create_if_missing=False)
        verify_prior_receipts(lambda name: b"".join(prior.read_file(name)))
        control["prior_receipts_sha256"] = PRIOR_RECEIPTS
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "control.json", control)
    write_json(out / "workspace.json", workspace)
    # Create-once across all invocations; a retry cannot rent new model workers.
    modal.Volume.objects.create(run_id, version=2, allow_existing=False)
    volume = modal.Volume.from_name(run_id, create_if_missing=False)
    with volume.batch_upload(force=False) as upload:
        upload.put_file(io.BytesIO(json.dumps(control).encode()), "/control.json")
        upload.put_file(io.BytesIO(prepare_raw), "/output/cache_prepare_receipt.json")
        if recovery:
            for name in PRIOR_RECEIPTS:
                raw = b"".join(prior.read_file("output/" + name))
                upload.put_file(io.BytesIO(raw), "/output/prior_receipts/" + name)
    public = b"".join(cache.read_file("public/main_choices_only.json"))
    if hashlib.sha256(public).hexdigest() != INPUT_SHA256:
        raise ValueError("cached public menus failed frozen hash")
    (out / "main_choices_only.json").write_bytes(public)
    image = modal.Image.debian_slim(python_version="3.11").pip_install(
        "torch==2.6.0", "transformers==4.51.3", "tokenizers==0.21.1", "safetensors==0.5.3",
        "huggingface-hub==0.30.2", "accelerate==1.6.0", "modal==1.6.0", "numpy==2.2.4",
        "jinja2==3.1.6", "lm-format-enforcer==0.11.3", "interegular==0.3.3")
    for name in SOURCES:
        image = image.add_local_file(str(repo / name), remote_path="/opt/scoring/" + name, copy=True)
    image = image.env({"PYTHONPATH": "/opt/scoring", "PYTHONUNBUFFERED": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1", "TOKENIZERS_PARALLELISM": "false",
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    app = modal.App(run_id, image=image, include_source=False)
    limits = control["budget"]["model_limits"]
    volumes = {"/cached": cache, "/paired": volume}
    if recovery:
        volumes["/prior"] = prior
    scorers = {}
    for tag, fn in (("qwen3b", remote_score_3b), ("qwen7b", remote_score_7b)):
        scorers[tag] = app.function(gpu="L40S", cpu=(2, 2), memory=(32768, 32768),
            timeout=limits[tag]["timeout"], startup_timeout=90,
            max_containers=1, min_containers=0, buffer_containers=0,
            scaledown_window=2, retries=0, include_source=False,
            volumes=volumes)(fn)
    result = {"run_id": run_id, "models": {}}
    calls = {}
    try:
        with modal.enable_output(), app.run():
            now = time.time()
            worker_control = {**control, "absolute_deadlines": {
                tag: now + limit["timeout"] + 90 for tag, limit in limits.items()}}
            write_json(out / "allocation_window.json", worker_control)
            try:
                for tag, scorer in scorers.items():
                    calls[tag] = scorer.spawn(worker_control)
                write_json(out / "calls.json", {tag: call.object_id for tag, call in calls.items()})
                for tag, call in calls.items():
                    try:
                        result["models"][tag] = call.get(timeout=max(
                            1, worker_control["absolute_deadlines"][tag] - time.time()))
                    except Exception as error:
                        call.cancel(terminate_containers=True)
                        result["models"][tag] = {"status": "provider_error",
                            "error_type": type(error).__name__, "error": str(error)}
            finally:
                for tag, call in calls.items():
                    if tag not in result["models"]:
                        call.cancel(terminate_containers=True)
    finally:
        collect(volume, out)
        write_json(out / "launch_result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prefix-cache-recovery-once", action="store_true")
    args = parser.parse_args()
    result = launch(Path(__file__).resolve().parents[1], args.out, args.source_commit,
                    args.prefix_cache_recovery_once)
    print(json.dumps(result, sort_keys=True))
    if len(result["models"]) != 2 or any(r.get("status") != "complete" for r in result["models"].values()):
        raise SystemExit("paired scoring incomplete; preserved evidence requires review")


if __name__ == "__main__":
    main()
