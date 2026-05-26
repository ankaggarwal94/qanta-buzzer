"""CS321M Final Project -- Modal A100 compute orchestrator.

Thin wrapper dispatching existing pipeline stages on managed GPU. Budget: $300 max.

Usage:
    modal run modal_cs321m.py -- --dry-run
    modal run modal_cs321m.py -- --config configs/cs321m_smoke.yaml
    modal run modal_cs321m.py -- --config configs/cs321m_final.yaml
    modal run modal_cs321m.py -- --stages build_mc_dataset run_baselines
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import modal

app = modal.App("cs321m-qanta-buzzer")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_pyproject("pyproject.toml")
    .pip_install("modal")
    .copy_local_dir(".", "/app")
)

PIPELINE_STAGES = ["build_mc_dataset", "run_baselines", "train_ppo", "evaluate_all"]
A100_RATE_PER_HOUR = 3.00  # Approximate Modal A100-80GB rate


def spend_tracker(
    stage_name: str, duration_seconds: float, gpu_type: str = "A100-80GB"
) -> None:
    """Append a spend log entry to PROJECT_WIKI/TRANSCRIPTS/modal_spend.log."""
    log_path = Path("PROJECT_WIKI/TRANSCRIPTS/modal_spend.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    estimate = duration_seconds / 3600 * A100_RATE_PER_HOUR
    entry = (
        f"{timestamp} | {stage_name} | {duration_seconds:.1f}s "
        f"| {gpu_type} | estimated_cost: ${estimate:.4f}\n"
    )
    with log_path.open("a", encoding="utf-8") as f:
        f.write(entry)
    print(f"[spend] {entry.strip()}")


@app.function(gpu="A100", timeout=3600, image=image)
def run_pipeline_stage(
    stage: str, config_path: str, extra_args: list[str] | None = None
) -> dict:
    """Run a single pipeline stage on Modal A100 GPU.

    Returns dict with stage, exit_code, duration_seconds.
    """
    cmd = [sys.executable, f"scripts/{stage}.py", "--config", config_path]
    if extra_args:
        cmd.extend(extra_args)

    start = time.time()
    result = subprocess.run(cmd, cwd="/app", capture_output=True, text=True)
    duration = time.time() - start

    if result.stdout:
        print(result.stdout[-2000:])
    if result.returncode != 0 and result.stderr:
        print(f"STDERR: {result.stderr[-2000:]}", file=sys.stderr)

    return {
        "stage": stage,
        "exit_code": result.returncode,
        "duration_seconds": round(duration, 2),
    }


@app.local_entrypoint()
def main():
    """CLI entrypoint for Modal dispatch of CS321M pipeline stages."""
    parser = argparse.ArgumentParser(
        description="CS321M Modal A100 pipeline orchestrator"
    )
    parser.add_argument(
        "--config", default="configs/cs321m_smoke.yaml",
        help="Path to YAML config (default: cs321m_smoke.yaml)",
    )
    parser.add_argument(
        "--stages", nargs="+", default=PIPELINE_STAGES, choices=PIPELINE_STAGES,
        help="Pipeline stages to run (default: all 4)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print configuration and exit without launching Modal",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Override output directory for all stages",
    )
    args = parser.parse_args()

    if args.dry_run:
        print("=" * 60)
        print("CS321M Modal Pipeline -- DRY RUN")
        print("=" * 60)
        print(f"Config:     {args.config}")
        print(f"Stages:     {', '.join(args.stages)}")
        print(f"GPU:        A100-80GB")
        print(f"Output dir: {args.output_dir or '(default)'}")
        print(f"Budget:     $300 max")
        print("=" * 60)
        print("No Modal functions launched. Pass without --dry-run to execute.")
        return

    print("=" * 60)
    print("CS321M Modal Pipeline -- EXECUTING")
    print(f"Config: {args.config} | Stages: {', '.join(args.stages)}")
    print("=" * 60)

    results = []
    total_start = time.time()

    for stage in args.stages:
        print(f"\n>>> Stage: {stage}")
        extra_args = []
        if args.output_dir:
            extra_args.extend(["--output-dir", args.output_dir])

        result = run_pipeline_stage.remote(stage, args.config, extra_args)
        results.append(result)
        spend_tracker(stage, result["duration_seconds"])

        if result["exit_code"] != 0:
            print(f"\nFAILED: {stage} (exit={result['exit_code']})")
            break
        print(f"Done: {stage} in {result['duration_seconds']:.1f}s")

    total_duration = time.time() - total_start
    print(f"\n{'=' * 60}")
    print(f"Pipeline complete in {total_duration:.1f}s")
    print(f"Results: {json.dumps(results, indent=2)}")
