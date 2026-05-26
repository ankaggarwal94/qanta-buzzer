#!/usr/bin/env python3
"""
Generate the Pilot Benchmark Translation Audit Card.

Aggregates all three metric verdicts (CSLI, calibration, StopDFF) from cached
paper_exports/ JSON files and compares against pre-registered thresholds from
threshold_manifest.json. Produces a machine-readable audit_card.json and a
human-readable audit_card.md summarizing the overall audit outcome.

Usage:
    python scripts/make_audit_card.py --help
    python scripts/make_audit_card.py --dry-run    # Parse args, print plan, exit 0
    python scripts/make_audit_card.py              # Full run

Inputs:
    paper_exports/csli.json             (panel CSLI results with per-model verdicts)
    paper_exports/calibration.json      (per-bucket ECE and gate verdict)
    paper_exports/stopdff.json          (median abs prefix shift and gate verdict)
    threshold_manifest.json             (pre-registered thresholds)

Outputs:
    paper_exports/audit_card.json       (machine-readable card with per-metric verdicts)
    paper_exports/audit_card.md         (human-readable markdown summary)

Exit codes:
    0 = success
    1 = runtime error
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_PAPER_EXPORTS = _REPO_ROOT / "paper_exports"
_SCRIPT_VERSION = "1.0.0"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Pilot Benchmark Translation Audit Card"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse args and print what would happen without writing files",
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict:
    """Load a JSON file, raising a clear error on failure."""
    if not path.exists():
        print(f"ERROR: Required file not found: {path}", file=sys.stderr)
        sys.exit(1)
    with open(path) as f:
        return json.load(f)


def _evaluate_csli(csli_data: dict, threshold: float) -> dict:
    """Evaluate CSLI metric against threshold.

    The threshold applies to choices-only accuracy per model: if any model's
    acc_choices_only exceeds the threshold, it indicates leakage.
    """
    per_model = csli_data["per_model"]
    max_acc_choices_only = max(
        m["acc_choices_only"] for m in per_model.values()
    )
    # Per threshold_manifest: direction is warn_if_above
    # If no model exceeds threshold, pass; otherwise warn
    if max_acc_choices_only > threshold:
        verdict = "warn"
    else:
        verdict = "pass"

    return {
        "name": "CSLI (Choice-Set Leakage Index)",
        "value": csli_data["panel_csli"]["mean"],
        "value_display": f"{csli_data['panel_csli']['mean']:.4f}",
        "ci_lower": csli_data["panel_csli"]["ci_lower"],
        "ci_upper": csli_data["panel_csli"]["ci_upper"],
        "threshold": threshold,
        "threshold_criterion": "max(acc_choices_only) <= threshold",
        "observed_criterion_value": max_acc_choices_only,
        "direction": "warn_if_above",
        "verdict": verdict,
        "details": {
            "n_models": len(per_model),
            "per_model_acc_choices_only": {
                k: v["acc_choices_only"] for k, v in per_model.items()
            },
            "leakage_flags": {
                k: v["leakage_flag"] for k, v in per_model.items()
            },
        },
    }


def _evaluate_calibration(cal_data: dict, threshold: float) -> dict:
    """Evaluate prefix-wise calibration ECE against threshold."""
    max_ece = cal_data["max_ece"]
    # Cross-verify with stored gate_verdict
    computed_verdict = "pass" if max_ece <= threshold else "warn"
    stored_verdict = cal_data["gate_verdict"]

    if computed_verdict != stored_verdict:
        print(
            f"WARNING: Calibration verdict mismatch: computed={computed_verdict}, "
            f"stored={stored_verdict}",
            file=sys.stderr,
        )

    return {
        "name": "Prefix-wise Calibration (ECE)",
        "value": max_ece,
        "value_display": f"{max_ece:.4f}",
        "threshold": threshold,
        "threshold_criterion": "max(bucket_ECE) <= threshold",
        "observed_criterion_value": max_ece,
        "direction": "warn_if_above",
        "verdict": computed_verdict,
        "details": {
            "per_bucket_ece": {
                k: v["ece"] for k, v in cal_data["per_bucket"].items()
            },
            "stored_gate_verdict": stored_verdict,
        },
    }


def _evaluate_stopdff(stopdff_data: dict, threshold: float) -> dict:
    """Evaluate diagnostic StopDFF against threshold."""
    median_shift = stopdff_data["median_abs_prefix_shift"]
    computed_verdict = "pass" if median_shift <= threshold else "warn"
    stored_verdict = stopdff_data["gate_verdict"]

    if computed_verdict != stored_verdict:
        print(
            f"WARNING: StopDFF verdict mismatch: computed={computed_verdict}, "
            f"stored={stored_verdict}",
            file=sys.stderr,
        )

    return {
        "name": "Diagnostic StopDFF (Median Abs Prefix Shift)",
        "value": median_shift,
        "value_display": f"{median_shift:.1f}",
        "threshold": threshold,
        "threshold_criterion": "median_abs_prefix_shift <= threshold",
        "observed_criterion_value": median_shift,
        "direction": "warn_if_above",
        "verdict": computed_verdict,
        "details": {
            "direction_breakdown": stopdff_data["direction_breakdown"],
            "stored_gate_verdict": stored_verdict,
            "metric_type": stopdff_data["metadata"]["metric_type"],
        },
    }


def _compute_overall_verdict(metrics: list[dict]) -> str:
    """Compute overall verdict from per-metric verdicts.

    PASS if all pass, WARN if any warn, FAIL if any fail.
    """
    verdicts = [m["verdict"] for m in metrics]
    if "fail" in verdicts:
        return "FAIL"
    if "warn" in verdicts:
        return "WARN"
    return "PASS"


def _write_audit_card_json(metrics: list[dict], overall_verdict: str) -> Path:
    """Write the machine-readable audit card JSON."""
    card = {
        "metrics": metrics,
        "overall_verdict": overall_verdict,
        "metadata": {
            "generated_by": "scripts/make_audit_card.py",
            "script_version": _SCRIPT_VERSION,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "threshold_source": "threshold_manifest.json",
        },
    }
    out_path = _PAPER_EXPORTS / "audit_card.json"
    with open(out_path, "w") as f:
        json.dump(card, f, indent=2)
    return out_path


def _write_audit_card_md(metrics: list[dict], overall_verdict: str) -> Path:
    """Write the human-readable audit card Markdown."""
    lines = [
        "# Pilot Benchmark Translation Audit Card",
        "",
        "| Metric | Value | Threshold | Verdict |",
        "|--------|-------|-----------|---------|",
    ]
    for m in metrics:
        ci_str = ""
        if "ci_lower" in m and m["ci_lower"] is not None:
            ci_str = f" [{m['ci_lower']:.4f}, {m['ci_upper']:.4f}]"
        lines.append(
            f"| {m['name']} | {m['value_display']}{ci_str} | {m['threshold']} | {m['verdict'].upper()} |"
        )
    lines.extend([
        "",
        f"**Overall Verdict: {overall_verdict}**",
        "",
        "---",
        "",
        f"*Generated by `make_audit_card.py` v{_SCRIPT_VERSION} at "
        f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}*",
        "",
        "All thresholds pre-registered in `threshold_manifest.json` and frozen before "
        "test-set inspection.",
        "",
    ])
    out_path = _PAPER_EXPORTS / "audit_card.md"
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    return out_path


def main() -> int:
    args = _parse_args()

    print("=== Pilot Benchmark Translation Audit Card ===")
    print()

    if args.dry_run:
        print("[DRY-RUN] Would load:")
        print(f"  - {_PAPER_EXPORTS / 'csli.json'}")
        print(f"  - {_PAPER_EXPORTS / 'calibration.json'}")
        print(f"  - {_PAPER_EXPORTS / 'stopdff.json'}")
        print(f"  - {_REPO_ROOT / 'threshold_manifest.json'}")
        print()
        print("[DRY-RUN] Would write:")
        print(f"  - {_PAPER_EXPORTS / 'audit_card.json'}")
        print(f"  - {_PAPER_EXPORTS / 'audit_card.md'}")
        return 0

    # Load inputs
    csli_data = _load_json(_PAPER_EXPORTS / "csli.json")
    cal_data = _load_json(_PAPER_EXPORTS / "calibration.json")
    stopdff_data = _load_json(_PAPER_EXPORTS / "stopdff.json")
    manifest = _load_json(_REPO_ROOT / "threshold_manifest.json")

    # Extract thresholds from manifest
    thresholds = {}
    for t in manifest["thresholds"]:
        metric = t["metric"]
        value = t.get("numeric_value_K4", t["threshold"])
        thresholds[metric] = float(value)

    # Evaluate each metric
    metrics = [
        _evaluate_csli(csli_data, thresholds["choices_only_accuracy"]),
        _evaluate_calibration(cal_data, thresholds["prefix_ece"]),
        _evaluate_stopdff(stopdff_data, thresholds["stopdff_median_abs_prefix"]),
    ]

    # Compute overall verdict
    overall_verdict = _compute_overall_verdict(metrics)

    # Write outputs
    json_path = _write_audit_card_json(metrics, overall_verdict)
    md_path = _write_audit_card_md(metrics, overall_verdict)

    # Print summary
    print("Per-metric results:")
    for m in metrics:
        print(f"  {m['name']}: {m['value_display']} (threshold: {m['threshold']}) -> {m['verdict'].upper()}")
    print()
    print(f"Overall Verdict: {overall_verdict}")
    print()
    print(f"Written: {json_path}")
    print(f"Written: {md_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
