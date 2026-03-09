#!/usr/bin/env python3
"""Sweep PPO smoke reward-shaping settings and record results.

Runs `scripts/train_ppo.py` in smoke mode across a small grid of:
- environment.wait_penalty
- environment.early_buzz_penalty

Collects metrics from artifacts/smoke/ppo_summary.json after each run and writes:
- artifacts/smoke/reward_sweep_results.json
- artifacts/smoke/reward_sweep_results.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SMOKE_CONFIG = PROJECT_ROOT / "configs" / "smoke.yaml"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "smoke"
TMP_CONFIG = ARTIFACT_DIR / "_tmp_sweep_smoke.yaml"
PPO_SUMMARY = ARTIFACT_DIR / "ppo_summary.json"

WAIT_PENALTIES = [0.0, 0.02, 0.05]
EARLY_BUZZ_PENALTIES = [0.2, 0.5, 0.8]
SEEDS = [13, 42, 123]


def run_cmd(cmd: list[str]) -> int:
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT)
    return proc.returncode


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep PPO reward shaping")
    parser.add_argument(
        "--seeds",
        type=str,
        default=",".join(str(s) for s in SEEDS),
        help="Comma-separated seeds, e.g. 13,42,123",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Optional timesteps override for train_ppo during sweep",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    base_cfg = load_yaml(SMOKE_CONFIG)

    python_exe = sys.executable
    results = []

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    grid = [(w, e) for w in WAIT_PENALTIES for e in EARLY_BUZZ_PENALTIES]

    print("=" * 72)
    print(f"Reward sweep: {len(grid)} configs x {len(seeds)} seeds")
    print("=" * 72)

    for idx, (wait_penalty, early_buzz_penalty) in enumerate(grid, start=1):
        per_seed = []
        print(f"[{idx}/{len(grid)}] wait_penalty={wait_penalty}, early_buzz_penalty={early_buzz_penalty}")

        for seed in seeds:
            cfg = dict(base_cfg)
            cfg.setdefault("environment", {})
            cfg["environment"] = dict(cfg["environment"])
            cfg["environment"]["wait_penalty"] = float(wait_penalty)
            cfg["environment"]["early_buzz_penalty"] = float(early_buzz_penalty)
            cfg["environment"]["seed"] = int(seed)

            cfg.setdefault("ppo", {})
            cfg["ppo"] = dict(cfg["ppo"])
            cfg["ppo"]["seed"] = int(seed)
            save_yaml(TMP_CONFIG, cfg)

            cmd = [python_exe, "scripts/train_ppo.py", "--config", str(TMP_CONFIG), "--smoke", "--seed", str(seed)]
            if args.timesteps is not None:
                cmd.extend(["--timesteps", str(args.timesteps)])

            start = time.time()
            code = run_cmd(cmd)
            elapsed = time.time() - start

            if code != 0 or not PPO_SUMMARY.exists():
                per_seed.append({"seed": seed, "status": "failed", "seconds": round(elapsed, 3)})
                continue

            summary = load_json(PPO_SUMMARY)
            per_seed.append(
                {
                    "seed": seed,
                    "status": "ok",
                    "seconds": round(elapsed, 3),
                    "buzz_accuracy": float(summary.get("buzz_accuracy", 0.0)),
                    "mean_sq": float(summary.get("mean_sq", 0.0)),
                    "mean_buzz_step": float(summary.get("mean_buzz_step", 0.0)),
                    "ece": float(summary.get("ece", 0.0)),
                    "brier": float(summary.get("brier", 0.0)),
                }
            )

        ok = [r for r in per_seed if r.get("status") == "ok"]
        if not ok:
            results.append(
                {
                    "wait_penalty": wait_penalty,
                    "early_buzz_penalty": early_buzz_penalty,
                    "status": "failed",
                    "num_ok": 0,
                    "num_total": len(per_seed),
                    "per_seed": per_seed,
                }
            )
            continue

        mean_acc = sum(r["buzz_accuracy"] for r in ok) / len(ok)
        mean_sq = sum(r["mean_sq"] for r in ok) / len(ok)
        mean_step = sum(r["mean_buzz_step"] for r in ok) / len(ok)
        mean_ece = sum(r["ece"] for r in ok) / len(ok)
        mean_brier = sum(r["brier"] for r in ok) / len(ok)
        mean_seconds = sum(r["seconds"] for r in ok) / len(ok)

        # Balanced objective: maximize accuracy + S_q while penalizing calibration error.
        objective = mean_acc + mean_sq - 0.5 * mean_ece

        results.append(
            {
                "wait_penalty": wait_penalty,
                "early_buzz_penalty": early_buzz_penalty,
                "status": "ok",
                "num_ok": len(ok),
                "num_total": len(per_seed),
                "seconds": round(mean_seconds, 3),
                "buzz_accuracy": mean_acc,
                "mean_sq": mean_sq,
                "mean_buzz_step": mean_step,
                "ece": mean_ece,
                "brier": mean_brier,
                "objective": objective,
                "per_seed": per_seed,
            }
        )

    # cleanup temp config
    if TMP_CONFIG.exists():
        TMP_CONFIG.unlink()

    out_json = ARTIFACT_DIR / "reward_sweep_results.json"
    out_csv = ARTIFACT_DIR / "reward_sweep_results.csv"

    with out_json.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    fields = [
        "wait_penalty",
        "early_buzz_penalty",
        "status",
        "num_ok",
        "num_total",
        "seconds",
        "buzz_accuracy",
        "mean_sq",
        "mean_buzz_step",
        "ece",
        "brier",
        "objective",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in results:
            flat = {k: row.get(k, "") for k in fields}
            writer.writerow(flat)

    ok_runs = [r for r in results if r.get("status") == "ok"]
    if not ok_runs:
        print("No successful runs.")
        return 1

    best = max(ok_runs, key=lambda r: float(r.get("objective", 0.0)))

    print("\nBest run:")
    print(best)
    print(f"Wrote: {out_json}")
    print(f"Wrote: {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
