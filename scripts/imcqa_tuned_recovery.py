"""CPU-only, hash-bound preparation of one manual operational recovery."""
from __future__ import annotations

import argparse
from datetime import datetime
from decimal import Decimal, ROUND_CEILING
import gzip
import hashlib
import io
import json
from pathlib import Path

from scripts.imcqa_tuned_design import MODEL, PROTOCOL, validate_public_package
from scripts.modal_imcqa_tuned import SOURCES, budget_plan, validate_plan

PRIOR_RUN = "imcqa-tuned-fresh-20261006"
RECOVERY_RUN = "imcqa-tuned-fresh-recovery1-20261006"
PRIOR_COMMIT = "7646fe20c668dc0fac979cb577b40887b23416ef"
PRIOR_GITHUB_RUN = 37401012406
TRANSPORT_SHA256 = "247dfba50e09ed7fa186462a716cc257b27c8e9ac0e4436c764a2c83436b22de"
PUBLIC_SHA256 = "250d3fd3aff11cd9566e470eb79da057f2444ce2429654e8b27034760dddc430"
PRIOR_FILES = ("control.json", "worker_receipt.json", "allocation_receipt.json", "github_job.json")
PRICING = {"max_cost_usd": "5.60", "rate_usd_per_second": "0.00063924",
           "rate_verified_utc": "2026-10-06T01:41:54Z", "rate_source_url": "https://modal.com/pricing"}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def timestamp(value):
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("timezone-aware receipt timestamps required")
    return result


def cumulative_budget(manifest, control, worker, allocation, job, *, now=None):
    """Reserve the whole completed CI interval for the stopped GPU attempt."""
    if (job["id"] != 112067958647 or job["run_id"] != PRIOR_GITHUB_RUN
            or job["run_attempt"] != 1 or job["head_sha"] != PRIOR_COMMIT
            or job["status"] != "completed" or job["conclusion"] != "failure"):
        raise ValueError("prior GitHub allocation is not the completed frozen attempt")
    started, completed = timestamp(job["started_at"]), timestamp(job["completed_at"])
    seconds = Decimal(str((completed-started).total_seconds()))
    if seconds != Decimal("312"):
        raise ValueError("prior completed job interval differs")
    if (control["run_id"] != PRIOR_RUN or control["source_commit"] != PRIOR_COMMIT
            or control["input_sha256"] != PUBLIC_SHA256 or control["production_contexts_per_model"] != 34000):
        raise ValueError("prior allocation control identity differs")
    validate_plan(control["budget"], now=started)
    if (worker["status"] != "benchmark_budget_stop" or worker["completed_rows"] != 128
            or worker["expected_rows"] != 34000 or worker["total_contexts"] != 34000
            or worker["source_commit"] != PRIOR_COMMIT or worker["public_input_sha256"] != PUBLIC_SHA256
            or worker["protocol"] != PROTOCOL or worker["model_tag"] != "qwen7b"
            or worker["attempt"] != 0 or worker["automatic_retries"] != 0 or worker["reused_rows"] != 0
            or worker["batch_size"] != 8 or worker["max_seconds"] != 8130
            or worker["benchmark"]["proceed"] is not False or worker["benchmark"]["processed"] != 128
            or not started <= timestamp(worker["started_utc"]) <= timestamp(worker["finished_utc"]) <= completed):
        raise ValueError("prior scoring attempt differs from the frozen operational stop")
    elapsed = Decimal(str(allocation["elapsed_seconds"]))
    worker_elapsed = Decimal(str(worker["elapsed_seconds"]))
    rate = Decimal(control["budget"]["allocation_rate_usd_per_second"])
    if (not elapsed.is_finite() or not worker_elapsed.is_finite() or not 0 < worker_elapsed <= elapsed <= seconds
            or allocation["model_tag"] != "qwen7b"
            or allocation["allocation_rate_usd_per_second"] != str(rate)
            or allocation["invoice_verified"] is not False):
        raise ValueError("prior allocation timing or rate differs")
    if (manifest["pricing"] != PRICING or manifest["total_user_ceiling_usd"] != "6.00"
            or manifest["prior_attempt_reserved_usd"] != "0.40"
            or manifest["recovery_allocation_ceiling_usd"] != "5.60"):
        raise ValueError("reviewed cumulative budget ceiling differs")
    prior_bound = seconds*rate + Decimal("0.20")
    prior_debit = prior_bound.quantize(Decimal("0.01"), rounding=ROUND_CEILING)
    if prior_debit != Decimal(manifest["prior_attempt_reserved_usd"]):
        raise ValueError("prior reservation is understated")
    budget = budget_plan(850, **manifest["pricing"], now=now)
    ceiling = Decimal(manifest["total_user_ceiling_usd"])
    if (prior_debit + Decimal(budget["ceiling_usd"]) > ceiling
            or prior_debit + Decimal(budget["reserved_estimate_usd"]) > ceiling
            or budget["worker_timeout_seconds"] != 8250 or budget["internal_deadline_seconds"] != 8130):
        raise ValueError("cumulative reservation exceeds total user ceiling")
    ledger = {"schema_version": "imcqa-tuned-cumulative-budget-v1", "total_user_ceiling_usd": str(ceiling),
        "prior_github_run_id": PRIOR_GITHUB_RUN, "prior_job_elapsed_seconds": str(seconds),
        "prior_measured_allocation_seconds": str(elapsed), "prior_interval_cost_bound_usd": str(prior_bound),
        "prior_attempt_reserved_usd": str(prior_debit), "recovery_allocation_ceiling_usd": budget["ceiling_usd"],
        "recovery_reserved_estimate_usd": budget["reserved_estimate_usd"],
        "combined_reserved_estimate_usd": str(prior_debit+Decimal(budget["reserved_estimate_usd"])),
        "method": "whole completed GitHub job at all-in GPU rate plus 0.20 USD, rounded up to cents, then new bounded allocation",
        "invoice_verified": False, "automatic_retries": 0, "reused_prior_scores": 0}
    return budget, ledger


def prepare(root: Path, out: Path, *, now=None):
    """Validate every byte and cumulative cap before creating a public-only handoff."""
    recovery = root/"imcqa_tuned_recovery"
    manifest_raw = (recovery/"recovery_manifest.json").read_bytes()
    manifest = json.loads(manifest_raw)
    fields = {"schema_version", "protocol", "run_id", "original_transport_sha256", "prior_files_sha256",
              "source_files_sha256", "pricing", "total_user_ceiling_usd", "prior_attempt_reserved_usd",
              "recovery_allocation_ceiling_usd", "changed_source_files", "scientific_changes", "reused_prior_scores"}
    if (set(manifest) != fields or manifest["schema_version"] != "imcqa-tuned-recovery-v1"
            or manifest["protocol"] != PROTOCOL or manifest["run_id"] != RECOVERY_RUN
            or manifest["scientific_changes"] is not False or manifest["reused_prior_scores"] != 0
            or manifest["changed_source_files"] != ["scripts/imcqa_tuned_scoring.py"]):
        raise ValueError("single operational recovery contract differs")
    transport_raw = (root/"imcqa_tuned_public/transport_manifest.json").read_bytes()
    if sha(transport_raw) != TRANSPORT_SHA256 or manifest["original_transport_sha256"] != TRANSPORT_SHA256:
        raise ValueError("original frozen transport manifest differs")
    transport = json.loads(transport_raw)
    if set(manifest["prior_files_sha256"]) != set(PRIOR_FILES):
        raise ValueError("prior evidence file allowlist differs")
    prior_raw = {name: (recovery/"prior"/name).read_bytes() for name in PRIOR_FILES}
    if {name:sha(raw) for name, raw in prior_raw.items()} != manifest["prior_files_sha256"]:
        raise ValueError("prior allocation evidence bytes differ")
    control, worker, allocation, job = (json.loads(prior_raw[name]) for name in PRIOR_FILES)
    if control["source_files_sha256"] != transport["source_files_sha256"]:
        raise ValueError("prior scoring sources differ from original transport")
    budget, ledger = cumulative_budget(manifest, control, worker, allocation, job, now=now)
    actual_sources = {name:sha((root/name).read_bytes()) for name in SOURCES}
    if actual_sources != manifest["source_files_sha256"]:
        raise ValueError("recovery sources differ from reviewed manifest")
    changed = sorted(name for name in SOURCES if actual_sources[name] != transport["source_files_sha256"][name])
    if changed != manifest["changed_source_files"]:
        raise ValueError("unexpected changes to frozen scientific source closure")
    packed = (root/"imcqa_tuned_public/public.json.gz").read_bytes()
    if (len(packed) != transport["compressed_bytes"] or sha(packed) != transport["compressed_sha256"]
            or transport["public_sha256"] != PUBLIC_SHA256 or not 0 < transport["public_bytes"] <= 128*1024**2):
        raise ValueError("original public compressed bytes differ")
    with gzip.GzipFile(fileobj=io.BytesIO(packed), mode="rb") as stream:
        raw = stream.read(transport["public_bytes"]+1)
    if len(raw) != transport["public_bytes"] or sha(raw) != PUBLIC_SHA256:
        raise ValueError("original public expanded bytes differ")
    package = json.loads(raw)
    jobs = validate_public_package(package)
    if len(jobs) != 34000 or package["n_questions"] != 850 or package["model"] != MODEL:
        raise ValueError("same complete 850-question scoring grid required")
    for key in ("policy_lock_sha256", "sample_size_planning_sha256", "selection_id", "main_dataset_sha256", "source_input_sha256"):
        if package[key] != transport[key]:
            raise ValueError("frozen public protocol binding differs")
    cfg_raw = (root/"configs/imcqa_tuned_fresh.json").read_bytes()
    if sha(cfg_raw) != transport["config_sha256"]:
        raise ValueError("frozen policy and analysis configuration differs")
    cfg = json.loads(cfg_raw)
    if (cfg["policy_lock_sha256"] != package["policy_lock_sha256"]
            or cfg["sample_size_planning_sha256"] != package["sample_size_planning_sha256"]
            or cfg["post_result_changes_allowed"] is not False):
        raise ValueError("policy locking contract differs")
    out.mkdir(parents=True, exist_ok=False)
    (out/"prior").mkdir()
    for name, value in prior_raw.items():
        (out/"prior"/name).write_bytes(value)
    (out/"public.json").write_bytes(raw)
    (out/"transport_manifest.json").write_bytes(transport_raw)
    (out/"recovery_manifest.json").write_bytes(manifest_raw)
    for name, value in (("budget_preflight.json", budget), ("cumulative_budget_ledger.json", ledger)):
        (out/name).write_text(json.dumps(value, indent=2, sort_keys=True)+"\n")
    return ledger


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(Path("."), args.out), sort_keys=True))


if __name__ == "__main__":
    main()
