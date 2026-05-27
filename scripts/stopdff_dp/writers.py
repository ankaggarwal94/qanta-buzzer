"""JSON / Markdown / LaTeX writers for DP StopDFF artifacts."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import median, mean

from .types import DPTrace


def assemble_payload(
    *,
    mc_traces: list[DPTrace],
    qa_traces: list[DPTrace],
    reward_schedule_name: str,
    continuation_estimator_name: str,
    fit_split: str,
    eval_split: str,
    coverage_summary: dict,
    ceiling_flags: dict,
    per_item_stopdff: list[tuple[str, int]],
    gate_verdict: str,
    gate_verdict_reason: str,
    confirmatory: bool,
    generation: dict | None = None,
) -> dict:
    """Compose the JSON payload, matching the existing artifact style."""
    signed = [shift for _, shift in per_item_stopdff]
    abs_shifts = [abs(s) for s in signed]
    return {
        "stopdff_dp_signed_median": float(median(signed)) if signed else 0.0,
        "stopdff_dp_signed_mean": float(mean(signed)) if signed else 0.0,
        "stopdff_dp_abs_median": float(median(abs_shifts)) if abs_shifts else 0.0,
        "stopdff_dp_abs_mean": float(mean(abs_shifts)) if abs_shifts else 0.0,
        "n_items": len(per_item_stopdff),
        "direction_breakdown": {
            "mc_earlier": sum(1 for _, s in per_item_stopdff if s < 0),
            "qa_earlier": sum(1 for _, s in per_item_stopdff if s > 0),
            "same_step": sum(1 for _, s in per_item_stopdff if s == 0),
        },
        "coverage": coverage_summary,
        "ceiling_flags": ceiling_flags,
        "gate_verdict": gate_verdict,
        "gate_verdict_reason": gate_verdict_reason,
        "confirmatory": confirmatory,
        "metadata": {
            "metric_type": "finite_horizon_dp",
            "stopping_policy": "finite_horizon_dp",
            "reward_schedule": reward_schedule_name,
            "continuation_estimator": continuation_estimator_name,
            "fit_split": fit_split,
            "eval_split": eval_split,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "generation": generation,
        },
    }


def write_json(path: Path, payload: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path


def write_markdown(path: Path, payload: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    md = []
    md.append("# Finite-Horizon DP StopDFF")
    md.append("")
    md.append(
        f"**Metric type:** `{payload['metadata']['metric_type']}` — "
        f"confirmatory: `{payload['confirmatory']}`"
    )
    md.append("")
    md.append("| Field | Value |")
    md.append("|-------|-------|")
    md.append(
        f"| Reward schedule | {payload['metadata']['reward_schedule']} |"
    )
    md.append(
        f"| Continuation estimator | {payload['metadata']['continuation_estimator']} |"
    )
    md.append(f"| Fit split | {payload['metadata']['fit_split']} |")
    md.append(f"| Eval split | {payload['metadata']['eval_split']} |")
    md.append(f"| n_items | {payload['n_items']} |")
    md.append(
        f"| StopDFF signed median | {payload['stopdff_dp_signed_median']:.4f} |"
    )
    md.append(
        f"| StopDFF signed mean | {payload['stopdff_dp_signed_mean']:.4f} |"
    )
    md.append(f"| Gate verdict | {payload['gate_verdict']} |")
    md.append("")
    md.append("## Coverage")
    md.append("")
    cov = payload["coverage"]
    md.append(
        f"- exact={cov['fraction_exact']:.3f}, "
        f"pooled={cov['fraction_pooled']:.3f}, "
        f"missing={cov['fraction_missing']:.3f}; "
        f"verdict={cov['verdict']} ({cov['reason']})"
    )
    md.append("")
    md.append("## Ceiling diagnostics")
    md.append("")
    for k, v in payload["ceiling_flags"].items():
        md.append(f"- {k}: {v}")
    md.append("")
    if not payload["confirmatory"]:
        md.append(
            "> ⚠️ Non-confirmatory estimator in use — interpret as an "
            "upper-bound diagnostic only."
        )
    path.write_text("\n".join(md))
    return path


def write_latex(path: Path, payload: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "\\begin{tabular}{lr}",
        "\\toprule",
        "Metric & Value \\\\",
        "\\midrule",
        f"Signed median StopDFF & {payload['stopdff_dp_signed_median']:.4f} \\\\",
        f"Signed mean StopDFF & {payload['stopdff_dp_signed_mean']:.4f} \\\\",
        f"Abs median StopDFF & {payload['stopdff_dp_abs_median']:.4f} \\\\",
        f"$n_{{items}}$ & {payload['n_items']} \\\\",
        f"Coverage exact & {payload['coverage']['fraction_exact']:.3f} \\\\",
        f"Coverage pooled & {payload['coverage']['fraction_pooled']:.3f} \\\\",
        f"Gate verdict & {payload['gate_verdict']} \\\\",
        "\\bottomrule",
        "\\end{tabular}",
    ]
    path.write_text("\n".join(lines))
    return path
