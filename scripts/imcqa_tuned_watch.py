"""Bounded metadata-only observation of the existing recovery allocation."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

from scripts.modal_acl_expansion import connect, remote_read
from scripts.modal_imcqa_tuned import verify_sources

RUN_ID = "imcqa-tuned-fresh-recovery1-20261006"
SOURCE_COMMIT = "8cd0b71150170ec545716b28fa63a07a163bad61"
INPUT_SHA256 = "250d3fd3aff11cd9566e470eb79da057f2444ce2429654e8b27034760dddc430"
FILES = ("output/qwen7b_claim.json", "output/qwen7b/progress.json",
         "output/qwen7b/attempts/000_benchmark.json", "output/qwen7b/receipt.json",
         "output/qwen7b_allocation_receipt.json")
TERMINAL = {"complete", "benchmark_budget_stop", "deadline_stop", "failed"}


def inspect(volume, *, root=Path(".")):
    """Read only five allowlisted status files after verifying frozen control."""
    raw = remote_read(volume, "control.json", maximum=65536)
    control = json.loads(raw)
    if (control["run_id"] != RUN_ID or control["input_sha256"] != INPUT_SHA256
            or control["source_commit"] != SOURCE_COMMIT or control["production_contexts_per_model"] != 34000
            or control["budget"]["n_questions"] != 850 or control["budget"]["ceiling_usd"] != "5.60"):
        raise ValueError("existing allocation identity differs")
    verify_sources(root, control)
    files = {}
    for name in FILES:
        try:
            content = remote_read(volume, name, maximum=1024**2)
        except FileNotFoundError:
            files[name] = {"present": False}
            continue
        files[name] = {"present": True, "sha256": hashlib.sha256(content).hexdigest(), "data": json.loads(content)}
    claim = files[FILES[0]].get("data", {})
    progress = files[FILES[1]].get("data", {})
    benchmark = files[FILES[2]].get("data", {})
    receipt = files[FILES[3]].get("data", {})
    allocation = files[FILES[4]].get("data", {})
    if claim and (claim["source_commit"] != SOURCE_COMMIT or claim["model_tag"] != "qwen7b"):
        raise ValueError("worker claim identity differs")
    if receipt and (receipt["source_commit"] != SOURCE_COMMIT or receipt["public_input_sha256"] != INPUT_SHA256
                    or receipt["expected_rows"] != 34000 or receipt["status"] not in TERMINAL):
        raise ValueError("terminal worker receipt identity differs")
    if allocation and (allocation["model_tag"] != "qwen7b"
                       or allocation["allocation_rate_usd_per_second"] != control["budget"]["allocation_rate_usd_per_second"]):
        raise ValueError("allocation completion identity differs")
    return {"control": {"sha256": hashlib.sha256(raw).hexdigest(), "input_sha256": INPUT_SHA256,
                         "source_commit": SOURCE_COMMIT, "identity_verified": True},
            "files": files, "terminal": bool(receipt or allocation),
            "summary": {"worker_entered": bool(claim), "worker_started_unix": claim.get("started_unix"),
                "phase": progress.get("phase"), "completed_rows": progress.get("completed_rows"),
                "expected_rows": progress.get("expected_rows"), "elapsed_seconds": progress.get("elapsed_seconds"),
                "benchmark_proceed": benchmark.get("proceed"), "benchmark_rows_per_second": benchmark.get("rows_per_second"),
                "projected_remaining_seconds": benchmark.get("projected_remaining_seconds"),
                "checkpoint_reserve_seconds": benchmark.get("checkpoint_reserve_seconds"),
                "worker_status": receipt.get("status"), "allocation_finished": bool(allocation)}}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--index", type=int, choices=range(15), required=True)
    args = parser.parse_args()
    out = Path("imcqa_tuned_recovery_watch")
    out.mkdir(exist_ok=True)
    path = out/f"snapshot-{args.index:02d}.json"
    report = {"schema_version": "imcqa-tuned-watch-v1", "mode": "read_only", "snapshot_index": args.index,
        "checked_utc": datetime.now(timezone.utc).isoformat(), "run_id": RUN_ID, "status": "started",
        "gpu_calls": 0, "cpu_function_calls": 0, "terminal": False}
    with path.open("x") as stream:
        json.dump(report, stream)
    try:
        modal, workspace = connect()
        report["workspace"] = workspace
        volume = modal.Volume.from_name(RUN_ID, create_if_missing=False)
        report.update(inspect(volume))
        report["status"] = "passed"
    except BaseException as error:
        report.update(status="failed", error_type=type(error).__name__)
    path.write_text(json.dumps(report, indent=2, sort_keys=True)+"\n")
    print(json.dumps({k: v for k, v in report.items() if k not in {"files", "workspace"}}, sort_keys=True))
    if report["status"] != "passed":
        raise SystemExit(1)
    if report["terminal"] and os.environ.get("GITHUB_ENV"):
        with Path(os.environ["GITHUB_ENV"]).open("a") as stream:
            stream.write("IMCQA_WATCH_COMPLETE=true\n")


if __name__ == "__main__":
    main()
