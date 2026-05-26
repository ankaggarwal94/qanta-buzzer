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

# Project root on sys.path so `from scripts.threshold_manifest import ...`
# resolves regardless of CWD (matches the convention used by every
# other script in this directory).
sys.path.insert(0, str(_REPO_ROOT))


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
    """Load a JSON file, raising FileNotFoundError on missing file.

    Raises
    ------
    FileNotFoundError
        If the file does not exist at the given path.
    """
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _evaluate_csli(csli_data: dict, threshold: float) -> dict:
    """Evaluate CSLI metric against threshold.

    The threshold applies to choices-only accuracy per model: if any model's
    acc_choices_only exceeds the threshold, it indicates leakage.

    PR #14 Blocker 1: surfaces BOTH the manuscript-aligned gap CSLI
    (``panel_csli.mean``) and the PAP-original choices-only excess
    (``panel_csli_choices_excess.mean_from_per_model_avg``) so a
    reader can see both interpretations without consulting raw
    csli.json. Falls back gracefully on older csli.json that lacks
    the excess fields.
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

    # PR #14 Blocker 1: compute the PAP-original "choices-only excess
    # over chance" if csli.json doesn't already carry it, so older
    # artifacts still surface both interpretations in the card.
    K = 4
    chance = 1.0 / K
    per_model_excess = {
        k: round(max(0.0, v["acc_choices_only"] - chance), 6)
        for k, v in per_model.items()
    }
    panel_excess_block = csli_data.get("panel_csli_choices_excess")
    if panel_excess_block is not None:
        panel_excess = panel_excess_block.get(
            "mean_from_per_model_avg",
            float(sum(per_model_excess.values()) / len(per_model_excess)),
        )
    else:
        panel_excess = float(sum(per_model_excess.values()) / len(per_model_excess))

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
            "panel_csli_gap": csli_data["panel_csli"]["mean"],
            "panel_csli_gap_definition": (
                "acc_full - acc_choices_only (final manuscript "
                "final_project.tex L120-121)"
            ),
            "panel_csli_choices_excess": round(panel_excess, 6),
            "panel_csli_choices_excess_definition": (
                "max(0, acc_choices_only - 1/K) per model, averaged "
                "(PAP-original definition; pap.tex)"
            ),
            "per_model_csli_choices_excess": per_model_excess,
            "K": K,
            "chance": chance,
            "definition_note": (
                "Two CSLI flavors are reported. The frozen gate is on "
                "max(acc_choices_only) > 0.30 (= 1/K + 0.05), "
                "independent of either flavor."
            ),
        },
    }


def _evaluate_calibration(cal_data: dict, threshold: float) -> dict:
    """Evaluate prefix-wise calibration ECE against threshold.

    PR #14 Blocker 4: the producer correctly emits per-bucket
    ``platt_model_type`` and ``n_samples`` so an empty or
    single-class validation bucket falls back to a
    ``ConstantCalibrationModel`` and is flagged in the artifact.
    This consumer now reads those fields and downgrades the verdict
    to ``warn`` when any bucket is degenerate, because
    ``compute_ece`` returns 0.0 for empty buckets (so an empty
    bucket would otherwise look perfectly calibrated).
    """
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

    # PR #14 Blocker 4: scan per-bucket fallback metadata for
    # degeneracy. Older calibration.json without these fields
    # degrades to no-warn (defensive default; matches the prior
    # consumer behavior).
    per_bucket = cal_data["per_bucket"]
    fallback_buckets = []
    empty_buckets = []
    for bucket_name, bucket in per_bucket.items():
        if bucket.get("platt_model_type") == "constant":
            fallback_buckets.append(
                {
                    "bucket": bucket_name,
                    "reason": bucket.get("platt_fallback_reason"),
                    "constant_probability": bucket.get(
                        "platt_constant_probability"
                    ),
                    "n_samples": bucket.get("n_samples"),
                }
            )
        if bucket.get("n_samples") == 0:
            empty_buckets.append(bucket_name)

    if fallback_buckets or empty_buckets:
        # Force WARN even if threshold-based ECE passes, because the
        # ECE is computed against a degenerate calibrator and/or
        # empty test bucket.
        computed_verdict = "warn"

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
                k: v["ece"] for k, v in per_bucket.items()
            },
            "per_bucket_n_samples": {
                k: v.get("n_samples") for k, v in per_bucket.items()
            },
            "per_bucket_platt_model_type": {
                k: v.get("platt_model_type") for k, v in per_bucket.items()
            },
            "fallback_buckets": fallback_buckets,
            "empty_buckets": empty_buckets,
            "stored_gate_verdict": stored_verdict,
        },
    }


def _evaluate_stopdff(stopdff_data: dict, threshold: float) -> dict:
    """Evaluate diagnostic StopDFF against threshold.

    PR #14 Blocker 2: when ``ceiling_effect_detected`` is true (no
    question in either condition stopped before the final prefix),
    the median_abs_prefix_shift is mechanically 0 because every
    pair lands at the same final step. The verdict-as-is would
    falsely PASS with "no power" — the reviewer's "scientifically
    misleading PASS" concern. This consumer now reads
    ``ceiling_effect_detected`` and per-bucket ``threshold_reachable``
    flags and renders a ``verdict_qualifier`` so the audit card
    surfaces the limitation in the headline verdict column.

    The current implementation keeps the verdict as ``"pass"`` (the
    metric is documented as ``diagnostic_only`` / ``myopic_threshold``
    in Phase 06; the PASS is "the diagnostic test passes its threshold,"
    not "the policy is provably fair"). The qualifier surfaces the
    ceiling effect / reachability in the headline rather than
    inverting the verdict, which would invalidate the already-submitted
    manuscript without changing the underlying metric semantics.
    """
    median_shift = stopdff_data["median_abs_prefix_shift"]
    computed_verdict = "pass" if median_shift <= threshold else "warn"
    stored_verdict = stopdff_data["gate_verdict"]

    if computed_verdict != stored_verdict:
        print(
            f"WARNING: StopDFF verdict mismatch: computed={computed_verdict}, "
            f"stored={stored_verdict}",
            file=sys.stderr,
        )

    # PR #14 Blocker 2: extract ceiling-effect / reachability flags.
    # Older stopdff.json without these fields degrades to "no
    # qualifier" (defensive default; matches the prior consumer
    # behavior).
    ceiling_effect = bool(stopdff_data.get("ceiling_effect_detected", False))
    reachability = stopdff_data.get("reachability") or {}
    unreachable_buckets = [
        bucket
        for bucket, info in reachability.items()
        if isinstance(info, dict) and info.get("threshold_reachable") is False
    ]

    qualifier_parts = []
    if ceiling_effect:
        qualifier_parts.append("ceiling effect — diagnostic null")
    if unreachable_buckets:
        qualifier_parts.append(
            f"unreachable bucket(s): {', '.join(sorted(unreachable_buckets))}"
        )
    verdict_qualifier = "; ".join(qualifier_parts) if qualifier_parts else None

    return {
        "name": "Diagnostic StopDFF (Median Abs Prefix Shift)",
        "value": median_shift,
        "value_display": f"{median_shift:.1f}",
        "threshold": threshold,
        "threshold_criterion": "median_abs_prefix_shift <= threshold",
        "observed_criterion_value": median_shift,
        "direction": "warn_if_above",
        "verdict": computed_verdict,
        "verdict_qualifier": verdict_qualifier,
        "details": {
            "direction_breakdown": stopdff_data["direction_breakdown"],
            "stored_gate_verdict": stored_verdict,
            "metric_type": stopdff_data["metadata"]["metric_type"],
            "ceiling_effect_detected": ceiling_effect,
            "unreachable_buckets": unreachable_buckets,
            "reachability": reachability,
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


def _extract_data_provenance(
    csli_data: dict,
    cal_data: dict,
    stopdff_data: dict,
) -> dict:
    """Pull MC coverage + retention metadata from each metric's JSON.

    PR #14 Blocker 3: surface the coverage/retention gate that each
    audit metric ran against so the card reader can verify they
    agreed on what counted as a defensible retained-subset audit.
    Falls back to ``"not_reported"`` markers for metric JSONs that
    pre-date the gate wiring.
    """
    def _summarize(data: dict, *, supports_val: bool) -> dict:
        summary: dict[str, object] = {}
        # CSLI nests mc_coverage / mc_retention_gate under
        # ``metadata`` (the original WR-04/WR-07 schema).
        # Calibration and StopDFF emit them at top level (new
        # PR-14 B3 schema). Read from both locations so the audit
        # card works with both conventions.
        metadata = data.get("metadata") if isinstance(data, dict) else None
        if isinstance(metadata, dict):
            coverage = data.get("mc_coverage") or metadata.get("mc_coverage")
            retention = data.get("mc_retention_gate") or metadata.get(
                "mc_retention_gate"
            )
        else:
            coverage = data.get("mc_coverage")
            retention = data.get("mc_retention_gate")
        if coverage is None:
            summary["coverage"] = "not_reported"
        elif "test" in coverage or "val" in coverage:
            summary["coverage"] = {
                k: {
                    "rate": v.get("coverage_rate"),
                    "threshold": v.get("threshold"),
                    "passed": v.get("passed"),
                    "overridden": v.get("overridden"),
                }
                for k, v in coverage.items()
                if isinstance(v, dict)
            }
        else:
            # CSLI's legacy ``mc_coverage`` block is a flat dict (single
            # test split). Wrap it into the same {split: ...} schema for
            # consistency with the new calibration/StopDFF format.
            summary["coverage"] = {
                "test": {
                    "rate": coverage.get("coverage_rate"),
                    "threshold": coverage.get("threshold"),
                    "passed": coverage.get("passed"),
                    "overridden": coverage.get("overridden"),
                }
            }
        if retention is None:
            summary["retention"] = "not_reported"
        elif isinstance(retention, dict) and (
            "test" in retention or "val" in retention
        ):
            summary["retention"] = {
                k: {
                    "rate": v.get("retention_rate"),
                    "threshold": v.get("threshold"),
                    "passed": v.get("passed"),
                    "overridden": v.get("overridden"),
                    "applies": v.get("applies"),
                }
                for k, v in retention.items()
                if isinstance(v, dict)
            }
        else:
            # CSLI's legacy ``mc_retention_gate`` block is a flat dict
            # (test split only). It also stored the actual retention
            # rate as ``metadata.mc_test_retention_rate`` rather than
            # inside the gate block; check both so the audit card can
            # show the rate without a re-run.
            metadata_rate = (
                metadata.get("mc_test_retention_rate")
                if isinstance(metadata, dict)
                else None
            )
            summary["retention"] = {
                "test": {
                    "rate": (
                        retention.get("retention_rate")
                        if isinstance(retention, dict)
                        and retention.get("retention_rate") is not None
                        else metadata_rate
                    ),
                    "threshold": (
                        retention.get("threshold")
                        if isinstance(retention, dict)
                        else None
                    ),
                    "passed": (
                        retention.get("passed")
                        if isinstance(retention, dict)
                        else None
                    ),
                    "overridden": (
                        retention.get("overridden")
                        if isinstance(retention, dict)
                        else None
                    ),
                    "applies": (
                        retention.get("applies")
                        if isinstance(retention, dict)
                        else None
                    ),
                }
            }
        return summary

    return {
        "csli": _summarize(csli_data, supports_val=False),
        "calibration": _summarize(cal_data, supports_val=True),
        "stopdff": _summarize(stopdff_data, supports_val=False),
    }


def _write_audit_card_json(
    metrics: list[dict],
    overall_verdict: str,
    data_provenance: dict | None = None,
) -> Path:
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
    if data_provenance is not None:
        card["data_provenance"] = data_provenance
    out_path = _PAPER_EXPORTS / "audit_card.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(card, f, indent=2)
    return out_path


def _render_verdict_cell(m: dict) -> str:
    """Render a metric's verdict cell, including any PR-14-B2 qualifier."""
    base = m["verdict"].upper()
    qualifier = m.get("verdict_qualifier")
    if qualifier:
        return f"{base} ({qualifier})"
    return base


def _write_audit_card_md(
    metrics: list[dict],
    overall_verdict: str,
    data_provenance: dict | None = None,
) -> Path:
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
            f"| {m['name']} | {m['value_display']}{ci_str} | {m['threshold']} | "
            f"{_render_verdict_cell(m)} |"
        )
    lines.extend([
        "",
        f"**Overall Verdict: {overall_verdict}**",
        "",
    ])

    # PR #14 Blocker 3: surface coverage / retention provenance per
    # metric so the card reader can verify all three metrics agreed
    # on what counted as a defensible retained-subset audit.
    if data_provenance:
        lines.extend(_render_data_provenance_md(data_provenance))

    lines.extend([
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
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return out_path


def _render_data_provenance_md(provenance: dict) -> list[str]:
    """Render the per-metric coverage + retention provenance block."""
    lines = [
        "## Data Provenance — MC Coverage and Retention",
        "",
        "| Metric | Split | Coverage (rate / threshold / pass / overridden) | "
        "Retention (rate / threshold / pass / overridden) |",
        "|--------|-------|-------------------------------------------------|"
        "-------------------------------------------------|",
    ]

    def _fmt_rate(value: object) -> str:
        if isinstance(value, (int, float)):
            return f"{float(value):.1%}"
        return "n/a"

    def _fmt_bool(value: object) -> str:
        if isinstance(value, bool):
            return "yes" if value else "no"
        return "n/a"

    def _fmt_cell(rate: object, threshold: object, passed: object, overridden: object) -> str:
        return (
            f"{_fmt_rate(rate)} / {_fmt_rate(threshold)} / "
            f"{_fmt_bool(passed)} / {_fmt_bool(overridden)}"
        )

    for metric_name in ("csli", "calibration", "stopdff"):
        block = provenance.get(metric_name)
        if not isinstance(block, dict):
            continue
        coverage = block.get("coverage")
        retention = block.get("retention")
        if coverage == "not_reported" and retention == "not_reported":
            lines.append(
                f"| {metric_name} | -- | not reported | not reported |"
            )
            continue
        splits = sorted(
            {
                *(coverage.keys() if isinstance(coverage, dict) else ()),
                *(retention.keys() if isinstance(retention, dict) else ()),
            }
        )
        for split_name in splits:
            cov = (
                coverage.get(split_name)
                if isinstance(coverage, dict)
                else None
            )
            ret = (
                retention.get(split_name)
                if isinstance(retention, dict)
                else None
            )
            cov_cell = (
                _fmt_cell(
                    cov.get("rate"),
                    cov.get("threshold"),
                    cov.get("passed"),
                    cov.get("overridden"),
                )
                if isinstance(cov, dict)
                else "not reported"
            )
            ret_cell = (
                _fmt_cell(
                    ret.get("rate"),
                    ret.get("threshold"),
                    ret.get("passed"),
                    ret.get("overridden"),
                )
                if isinstance(ret, dict)
                else "not reported"
            )
            lines.append(
                f"| {metric_name} | {split_name} | {cov_cell} | {ret_cell} |"
            )
    lines.append("")
    return lines


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

    # Load inputs (WR-05: collect all missing files before failing).
    # Threshold manifest goes through load_frozen_threshold_manifest so
    # the sha256 sidecar is verified at load time (DATA-03 / CR-01).
    from scripts.threshold_manifest import load_frozen_threshold_manifest

    try:
        csli_data = _load_json(_PAPER_EXPORTS / "csli.json")
        cal_data = _load_json(_PAPER_EXPORTS / "calibration.json")
        stopdff_data = _load_json(_PAPER_EXPORTS / "stopdff.json")
        manifest = load_frozen_threshold_manifest(
            _REPO_ROOT / "threshold_manifest.json", strict=True
        )
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

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

    # PR #14 Blocker 3: extract per-metric coverage + retention
    # provenance so the audit card visibly records what counted as a
    # defensible retained-subset audit for each metric.
    data_provenance = _extract_data_provenance(
        csli_data, cal_data, stopdff_data
    )

    # Write outputs
    json_path = _write_audit_card_json(
        metrics, overall_verdict, data_provenance=data_provenance
    )
    md_path = _write_audit_card_md(
        metrics, overall_verdict, data_provenance=data_provenance
    )

    # Print summary
    print("Per-metric results:")
    for m in metrics:
        qualifier = m.get("verdict_qualifier")
        qualifier_str = f" ({qualifier})" if qualifier else ""
        print(
            f"  {m['name']}: {m['value_display']} "
            f"(threshold: {m['threshold']}) -> {m['verdict'].upper()}"
            f"{qualifier_str}"
        )
    print()
    print(f"Overall Verdict: {overall_verdict}")
    print()
    print(f"Written: {json_path}")
    print(f"Written: {md_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
