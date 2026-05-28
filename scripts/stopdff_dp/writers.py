"""JSON / Markdown / LaTeX writers for DP StopDFF artifacts."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from statistics import median, mean

from .types import DPTrace


def _fmt_float(value: object, spec: str = ".3f") -> str:
    """Format a float with spec, or return 'n/a' for None.

    diagnostics.summarize_coverage returns None for fraction_* keys when
    no trace cells exist. The writers preserve that diagnostic signal
    in the MD/TeX output rather than rendering it as a misleading 0.
    """
    if value is None:
        return "n/a"
    return format(value, spec)


def _latex_escape(value: object) -> str:
    """Escape LaTeX special characters in a string field.

    Defensive even when current values are well-known constants like
    ``pass`` / ``warn``: future schedule names or producer-augmented
    string fields could include ``_``, ``&``, ``%``, ``$``, ``#``,
    ``{``, ``}``, ``~``, ``^``, or ``\\``. Mirrors the sweep writer's
    helper at scripts/sweep_stopdff_dp.py:_latex_escape.

    Implemented as a single-pass ``re.sub`` so replacement strings (e.g.
    ``\\textbackslash{}`` containing ``{}``) are not re-processed by
    later rules.
    """
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    pattern = re.compile("|".join(re.escape(c) for c in replacements))
    return pattern.sub(lambda m: replacements[m.group(0)], text)


def assemble_payload(
    *,
    mc_traces: list[DPTrace],
    qa_traces: list[DPTrace],
    reward_schedule_name: str,
    reward_schedule_description: str = "",
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
    mc_coverage: dict | None = None,
    mc_retention_gate: dict | None = None,
    mc_build_metadata: dict | None = None,
) -> dict:
    """Compose the JSON payload, matching the existing artifact style.

    The optional ``mc_coverage``, ``mc_retention_gate``, and
    ``mc_build_metadata`` arguments carry audit-card-ready blocks produced
    by ``scripts/_audit_gates.py``. When provided, they are surfaced as
    top-level payload keys so the DP StopDFF artifact matches the
    convention used by ``csli.json`` and ``stopdff.json``.
    """
    signed = [shift for _, shift in per_item_stopdff]
    abs_shifts = [abs(s) for s in signed]
    payload = {
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
            "reward_schedule_description": reward_schedule_description,
            "continuation_estimator": continuation_estimator_name,
            "fit_split": fit_split,
            "eval_split": eval_split,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "generation": generation,
        },
    }
    if mc_coverage is not None:
        payload["mc_coverage"] = mc_coverage
    if mc_retention_gate is not None:
        payload["mc_retention_gate"] = mc_retention_gate
    if mc_build_metadata is not None:
        payload["mc_build_metadata"] = mc_build_metadata
    return payload


def write_json(path: Path, payload: dict) -> Path:
    """Write payload to JSON, normalising numpy types via to_serializable.

    Project invariant (see scripts/_common.to_serializable): every artifact
    must run through the canonical serializer to avoid silent TypeError on
    numpy scalars from pandas/diagnostics counters.
    """
    from scripts._common import to_serializable  # local import: avoid circular at module load
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_serializable(payload), f, indent=2)
    return path


def write_markdown(path: Path, payload: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    md = []
    md.append("# Finite-Horizon DP StopDFF")
    md.append("")
    md.append(
        f"**Metric type:** `{payload['metadata']['metric_type']}` - "
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
        f"- exact={_fmt_float(cov['fraction_exact'])}, "
        f"pooled={_fmt_float(cov['fraction_pooled'])}, "
        f"missing={_fmt_float(cov['fraction_missing'])}; "
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
            "> WARNING: Non-confirmatory estimator in use - interpret as an "
            "upper-bound diagnostic only."
        )
    path.write_text("\n".join(md), encoding="utf-8")
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
        f"Coverage exact & {_fmt_float(payload['coverage']['fraction_exact'])} \\\\",
        f"Coverage pooled & {_fmt_float(payload['coverage']['fraction_pooled'])} \\\\",
        f"Gate verdict & {_latex_escape(payload['gate_verdict'])} \\\\",
        "\\bottomrule",
        "\\end{tabular}",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
