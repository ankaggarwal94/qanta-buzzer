#!/usr/bin/env python3
"""Bounded create-once launch of the independently tuned plain-score experiment.

Pricing is supplied as an explicit operator attestation at launch; no historical
rate is silently substituted. --dry-run validates inputs without contacting Modal.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import re
from decimal import Decimal
import hashlib
import io
import json
from pathlib import Path
import time

from scripts.modal_acl_option_scores import digest, write_json, safe_output_path
from scripts.modal_acl_paired_prompt_scores import CACHE_RUN, PREPARE_SHA256, verify_prepare
from scripts.imcqa_tuned_design import validate_public_package, PROTOCOL
from scripts.imcqa_tuned_scoring import worker_seconds

RUN_PREFIX = "imcqa-tuned-"
SOURCES = ("scripts/__init__.py", "scripts/modal_imcqa_tuned.py", "scripts/imcqa_tuned_scoring.py",
    "scripts/imcqa_tuned_design.py", "scripts/imcqa_protocol_scoring.py", "scripts/imcqa_protocol_design.py",
    "scripts/imcqa_wait_scoring.py", "scripts/modal_acl_paired_prompt_scores.py", "scripts/acl_paired_prompt_scoring.py",
    "scripts/acl_option_scoring.py", "scripts/modal_acl_option_scores.py", "scripts/modal_acl_expansion.py",
    "scripts/jane_gpu_backend.py", "scripts/jane_qwen_backend.py", "scripts/jane_output_constraints.py",
    "configs/imcqa_protocol_pilot.json")


def budget_plan(n_questions: int, *, max_cost_usd: str, rate_usd_per_second: str,
                rate_verified_utc: str, rate_source_url: str, now=None) -> dict:
    """Bound one allocation using a fresh, explicit all-in pricing attestation."""
    now = now or datetime.now(timezone.utc)
    verified = datetime.fromisoformat(rate_verified_utc.replace("Z", "+00:00"))
    if (verified.tzinfo is None or not 0 <= (now-verified).total_seconds() <= 86400
            or rate_source_url != "https://modal.com/pricing"):
        raise ValueError("current official all-in allocation rate attestation required")
    ceiling, rate = Decimal(max_cost_usd), Decimal(rate_usd_per_second)
    if not ceiling.is_finite() or not rate.is_finite() or ceiling <= 0 or rate <= 0:
        raise ValueError("finite positive ceiling and all-in rate required")
    timeout = worker_seconds(n_questions)
    reserve = rate*(timeout+92)+Decimal("0.20")
    if reserve > ceiling:
        raise ValueError("bounded allocation reservation exceeds explicit cost ceiling")
    return {"schema": "imcqa-tuned-budget-v1", "n_questions": n_questions,
        "ceiling_usd": str(ceiling), "allocation_rate_usd_per_second": str(rate),
        "rate_verified_utc": rate_verified_utc, "rate_source_url": rate_source_url,
        "pricing_validation": "operator-attested current all-in rate for L40S + 2 CPU + 32768 MiB; not independently invoice verified",
        "worker_timeout_seconds": timeout, "internal_deadline_seconds": timeout-120,
        "startup_timeout_seconds": 90, "scaledown_seconds": 2,
        "gpu_calls": 1, "cpu_calls": 0, "automatic_retries": 0,
        "staging_and_contingency_usd": "0.20", "reserved_estimate_usd": str(reserve),
        "invoice_verified": False}


def validate_plan(plan: dict, *, now=None) -> None:
    expected = budget_plan(plan["n_questions"], max_cost_usd=plan["ceiling_usd"],
        rate_usd_per_second=plan["allocation_rate_usd_per_second"],
        rate_verified_utc=plan["rate_verified_utc"], rate_source_url=plan["rate_source_url"], now=now)
    if plan != expected:
        raise ValueError("allocation plan differs from bounded implementation")


def verify_sources(root: Path, control: dict) -> None:
    validate_plan(control["budget"])
    if (not re.fullmatch(r"imcqa-tuned-[a-z0-9-]{1,40}", control["run_id"])
            or control["cache_run"] != CACHE_RUN
            or control["prepare_receipt_sha256"] != PREPARE_SHA256):
        raise ValueError("unexpected run/cache identity")
    if {name: digest(root/name) for name in SOURCES} != control["source_files_sha256"]:
        raise ValueError("scoring sources differ from frozen control")
    if control["production_contexts_per_model"] != 40*control["budget"]["n_questions"]:
        raise ValueError("runtime question count differs from scoring count")


def verify_public(raw: bytes, expected_hash: str) -> dict:
    if hashlib.sha256(raw).hexdigest() != expected_hash:
        raise ValueError("public package hash differs")
    from scripts.acl_option_scoring import load_json
    package = load_json(raw)
    jobs = validate_public_package(package)
    if (len(jobs) != 40*len({job["qid"] for job in jobs})
            or any(job["execution"] != "new" or job["arm"] != "plain" for job in jobs)):
        raise ValueError("exact new plain-only factorial coverage required")
    return package


def collection_limit(expected_contexts: int) -> int:
    """Allow bounded token evidence proportional to the predeclared score grid."""
    if type(expected_contexts) is not int or expected_contexts % 40:
        raise ValueError("complete per-question context grid required")
    worker_seconds(expected_contexts // 40)
    return max(160*1024**2, expected_contexts*32768 + 16*1024**2)


def collect(volume, out: Path, *, expected_contexts: int) -> dict:
    """Download create-once evidence files with traversal and cumulative-size gates."""
    limit = collection_limit(expected_contexts)
    files, total, seen = [], 0, set()
    manifest = out / "download_manifest.json"
    if manifest.exists():
        raise FileExistsError("collection manifest already exists")
    for entry in volume.iterdir("/output", recursive=True):
        name = str(entry.path).lstrip("/")
        if not name.endswith((".json", ".jsonl", ".txt")):
            continue
        name = safe_output_path(name)
        if name in seen:
            raise ValueError("duplicate remote output path")
        seen.add(name)
        target = out / name
        if not target.resolve().is_relative_to(out.resolve()):
            raise ValueError("output symlink escapes collection directory")
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            for chunk in volume.read_file(name):
                total += len(chunk)
                if total > limit:
                    raise ValueError("collection exceeds frozen context-scaled byte limit")
                stream.write(chunk)
        files.append({"path":name, "bytes":target.stat().st_size, "sha256":digest(target)})
    report = {"files":files, "total_bytes":total, "byte_limit":limit,
              "expected_contexts":expected_contexts, "scope":"read-only evidence collection; no GPU launch"}
    with manifest.open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return report


def collect_only(run_id: str, out: Path, control_path: Path) -> dict:
    """Recover an existing allocation without importing or calling a scorer."""
    from scripts.modal_acl_expansion import connect, remote_read
    control = json.loads(control_path.read_bytes())
    if (not re.fullmatch(r"imcqa-tuned-[a-z0-9-]{1,40}", run_id)
            or control.get("run_id") != run_id or control.get("cache_run") != CACHE_RUN
            or control.get("prepare_receipt_sha256") != PREPARE_SHA256):
        raise ValueError("collection run or cache identity differs")
    expected = control["production_contexts_per_model"]
    collection_limit(expected)
    modal, workspace = connect()
    volume = modal.Volume.from_name(run_id, create_if_missing=False)
    remote_control = json.loads(remote_read(volume, "control.json", maximum=65536))
    if remote_control != control:
        raise ValueError("remote allocation control differs from local frozen control")
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "control.json", control)
    write_json(out / "workspace.json", workspace)
    return {"status":"collected_only", "run_id":run_id, "gpu_executed":False,
            **collect(volume, out, expected_contexts=expected)}


def remote_score(tag: str, control: dict) -> dict:
    import modal
    verify_sources(Path("/opt/scoring"), control)
    if tag != "qwen7b":
        raise ValueError("unknown model tag")
    if time.time() >= control["absolute_deadlines"][tag]:
        raise TimeoutError("allocation window expired")
    # Separate atomic claim prevents provider infrastructure replay from renting
    # a second scoring worker even when application-level retries are disabled.
    modal.Volume.objects.create(control["run_id"]+"-"+tag+"-claim", version=2, allow_existing=False)
    volume = modal.Volume.from_name(control["run_id"], create_if_missing=False)
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
        from scripts.imcqa_tuned_scoring import run_scoring
        def progress(update):
            from scripts.modal_acl_expansion import replace_progress
            replace_progress(output/tag/"progress.json", update)
            volume.commit()
            print(json.dumps({"model_tag":tag, **update}), flush=True)
        return run_scoring(tag, public_path, control["input_sha256"], cache/"models", output/tag,
            source_commit=control["source_commit"], max_seconds=control["budget"]["internal_deadline_seconds"], progress=progress)
    finally:
        write_json(output/f"{tag}_allocation_receipt.json", {"model_tag":tag,
            "elapsed_seconds":time.monotonic()-started,
            "allocation_rate_usd_per_second":control["budget"]["allocation_rate_usd_per_second"], "invoice_verified":False})
        volume.commit()


def remote_score_7b(control):
    return remote_score("qwen7b", control)


def launch(repo: Path, public_path: Path, out: Path, source_commit: str, *, run_id: str, budget: dict, dry_run=False) -> dict:
    from scripts.modal_acl_expansion import connect, verify_source_commit
    if len(source_commit) != 40 or any(c not in "0123456789abcdef" for c in source_commit):
        raise ValueError("exact committed source identity required")
    raw = public_path.read_bytes()
    input_hash = hashlib.sha256(raw).hexdigest()
    package = verify_public(raw, input_hash)
    control = {"run_id":run_id, "cache_run":CACHE_RUN, "source_commit":source_commit,
        "input_sha256":input_hash, "prepare_receipt_sha256":PREPARE_SHA256,
        "source_files_sha256":{name:digest(repo/name) for name in SOURCES},
        "budget":budget, "created_utc":datetime.now(timezone.utc).isoformat(),
        "estimand":"fresh evaluation of independently selected plain adaptive and fixed selective policies",
        "production_contexts_per_model":len(package["jobs"])}
    verify_sources(repo, control)
    verify_source_commit(repo, source_commit, control["source_files_sha256"])
    if dry_run:
        return {"status":"preflight_only", "gpu_executed":False, "control":control}
    modal, workspace = connect()
    cache = modal.Volume.from_name(CACHE_RUN, create_if_missing=False)
    prepare_raw = b"".join(cache.read_file("output/prepare_receipt.json"))
    verify_prepare(prepare_raw)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out/"control.json", control); write_json(out/"workspace.json", workspace)
    (out/"pilot.json").write_bytes(raw)
    modal.Volume.objects.create(run_id, version=2, allow_existing=False)
    volume = modal.Volume.from_name(control["run_id"], create_if_missing=False)
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
    app = modal.App(run_id, image=image, include_source=False)
    scorers = {}
    for tag, fn in (("qwen7b",remote_score_7b),):
        scorers[tag] = app.function(gpu="L40S", cpu=(2,2), memory=(32768,32768),
            timeout=budget["worker_timeout_seconds"], startup_timeout=90,
            max_containers=1, min_containers=0, buffer_containers=0, scaledown_window=2,
            retries=0, include_source=False, volumes={"/cached":cache,"/pilot":volume})(fn)
    result, calls = {"run_id":run_id, "models":{}}, {}
    try:
        with modal.enable_output(), app.run():
            now = time.time()
            worker_control = {**control,"absolute_deadlines":{"qwen7b":now+budget["worker_timeout_seconds"]+90}}
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
        collect(volume, out, expected_contexts=control["production_contexts_per_model"])
        write_json(out/"launch_result.json",result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-package", type=Path)
    parser.add_argument("--source-commit")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--max-cost-usd")
    parser.add_argument("--allocation-rate-usd-per-second",
        help="Current combined L40S, 2 CPU, and 32768 MiB RAM rate; do not use an old estimate")
    parser.add_argument("--rate-verified-utc")
    parser.add_argument("--rate-source-url")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--control", type=Path, help="Original launch control.json, required for collect-only")
    args = parser.parse_args()
    if args.collect_only:
        if args.control is None or args.dry_run:
            parser.error("--collect-only requires --control and cannot combine with --dry-run")
        print(json.dumps(collect_only(args.run_id, args.out, args.control), sort_keys=True))
        return
    required = ("public_package", "source_commit", "max_cost_usd", "allocation_rate_usd_per_second",
                "rate_verified_utc", "rate_source_url")
    missing = ["--"+name.replace("_", "-") for name in required if getattr(args,name) is None]
    if missing:
        parser.error("launch/preflight requires " + ", ".join(missing))
    package = verify_public(args.public_package.read_bytes(), digest(args.public_package))
    n = len({job["qid"] for job in package["jobs"]})
    budget = budget_plan(n, max_cost_usd=args.max_cost_usd,
        rate_usd_per_second=args.allocation_rate_usd_per_second,
        rate_verified_utc=args.rate_verified_utc, rate_source_url=args.rate_source_url)
    result = launch(Path(__file__).resolve().parents[1], args.public_package, args.out, args.source_commit,
                    run_id=args.run_id, budget=budget, dry_run=args.dry_run)
    print(json.dumps(result, sort_keys=True))
    if not args.dry_run and (len(result["models"]) != 1 or any(r.get("status") != "complete" for r in result["models"].values())):
        raise SystemExit("scoring incomplete; preserved evidence requires review")


if __name__ == "__main__":
    main()
