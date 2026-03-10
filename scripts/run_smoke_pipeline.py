#!/usr/bin/env python3
"""Run the full canonical smoke pipeline end-to-end.

Stages:
1) build_mc_dataset
2) run_baselines
3) train_ppo
4) evaluate_all

Writes a summary JSON to artifacts/smoke/smoke_pipeline_summary.json.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "smoke"


STAGES = [
    ["scripts/build_mc_dataset.py", "--smoke"],
    ["scripts/run_baselines.py", "--smoke"],
    ["scripts/train_ppo.py", "--smoke"],
    ["scripts/evaluate_all.py", "--smoke"],
]


def run_stage(python_exe: str, args: list[str]) -> tuple[int, float]:
    """Run one stage command and return (exit_code, seconds)."""
    cmd = [python_exe, *args]
    start = time.time()
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT)
    elapsed = time.time() - start
    return proc.returncode, elapsed


def main() -> int:
    parser = argparse.ArgumentParser(description="Run full smoke pipeline")
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter to use (default: current interpreter)",
    )
    ns = parser.parse_args()

    print("=" * 60)
    print("Smoke Pipeline Runner")
    print("=" * 60)
    print(f"Python: {ns.python}")
    print()

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    summary: dict[str, object] = {
        "python": ns.python,
        "started_at_unix": time.time(),
        "stages": [],
    }

    pipeline_start = time.time()
    for stage_args in STAGES:
        stage_name = stage_args[0]
        print(f"Running: {stage_name} {' '.join(stage_args[1:])}")
        code, seconds = run_stage(ns.python, stage_args)
        summary["stages"].append(
            {
                "stage": stage_name,
                "args": stage_args[1:],
                "exit_code": code,
                "seconds": round(seconds, 3),
            }
        )
        if code != 0:
            summary["status"] = "failed"
            summary["failed_stage"] = stage_name
            summary["total_seconds"] = round(time.time() - pipeline_start, 3)
            out_path = ARTIFACT_DIR / "smoke_pipeline_summary.json"
            out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            print(f"\nFAILED at {stage_name} (exit={code})")
            print(f"Summary written: {out_path}")
            return code
        print(f"✓ {stage_name} completed in {seconds:.1f}s\n")

    summary["status"] = "ok"
    summary["total_seconds"] = round(time.time() - pipeline_start, 3)
    out_path = ARTIFACT_DIR / "smoke_pipeline_summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("=" * 60)
    print("Smoke pipeline completed successfully")
    print(f"Summary written: {out_path}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
