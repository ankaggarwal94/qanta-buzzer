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


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Pilot Benchmark Translation Audit Card"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse args and print what would happen without writing files",
    )
    parser.add_argument(
        "--include-dp-stopdff",
        action="store_true",
        help=(
            "Append a finite-horizon DP StopDFF row from paper_exports/"
            "stopdff_dp.json to the audit card (in addition to, not replacing, "
            "the existing diagnostic row)."
        ),
    )
    return parser.parse_args(argv)


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

    The threshold applies to choices-only accuracy per model: if any
    model's acc_choices_only exceeds the threshold, it indicates leakage.

    PR #14 follow-up review (Blocker 3): the canonical CSLI is the
    PAP-original choices-only excess over chance, published at
    ``panel_csli`` with a bootstrap CI. The full-minus-choices gap is
    published at ``panel_question_use_gap``. Both are surfaced here so a
    reader can see either. Falls back on older csli.json (where
    ``panel_csli`` was the gap) by reading the per-model
    ``acc_choices_only`` values and recomputing the excess; in that case
    the displayed CSLI value carries an ``artifact_format`` flag so the
    audit card reader knows the bootstrap CI was unavailable.
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

    K = 4
    chance = 1.0 / K
    per_model_excess = {
        k: round(max(0.0, v["acc_choices_only"] - chance), 6)
        for k, v in per_model.items()
    }
    per_model_gap = {
        k: v.get("question_use_gap", v.get("csli"))
        for k, v in per_model.items()
    }

    panel_csli_block = csli_data.get("panel_csli")
    panel_gap_block = csli_data.get("panel_question_use_gap")

    # New artifact format: panel_csli is the choices-excess CSLI with a
    # bootstrap CI; panel_question_use_gap is the gap (also with a CI).
    new_format = (
        isinstance(panel_csli_block, dict)
        and "definition" in panel_csli_block
        and panel_csli_block["definition"].startswith("max(0,")
    )

    if new_format and isinstance(panel_gap_block, dict):
        csli_value = panel_csli_block["mean"]
        csli_ci_lower = panel_csli_block.get("ci_lower")
        csli_ci_upper = panel_csli_block.get("ci_upper")
        gap_value = panel_gap_block["mean"]
        gap_ci_lower = panel_gap_block.get("ci_lower")
        gap_ci_upper = panel_gap_block.get("ci_upper")
        artifact_format = "v2_choices_excess_canonical"
    else:
        # Legacy artifact: panel_csli was the gap. Reconstruct excess
        # from per-model acc_choices_only so the audit card still
        # publishes the canonical CSLI, and surface that the displayed
        # value lacks a bootstrap CI from this artifact.
        legacy_excess_block = csli_data.get("panel_csli_choices_excess")
        if isinstance(legacy_excess_block, dict):
            csli_value = legacy_excess_block.get(
                "mean_from_per_model_avg",
                float(sum(per_model_excess.values()) / len(per_model_excess)),
            )
        else:
            csli_value = float(
                sum(per_model_excess.values()) / len(per_model_excess)
            )
        csli_ci_lower = None
        csli_ci_upper = None
        gap_value = (panel_csli_block or {}).get("mean")
        gap_ci_lower = (panel_csli_block or {}).get("ci_lower")
        gap_ci_upper = (panel_csli_block or {}).get("ci_upper")
        artifact_format = "v1_legacy_gap_under_panel_csli"

    return {
        "name": "CSLI (Choice-Set Leakage Index, choices-only excess)",
        "value": csli_value,
        "value_display": (
            f"{csli_value:.4f}" if csli_value is not None else "n/a"
        ),
        "ci_lower": csli_ci_lower,
        "ci_upper": csli_ci_upper,
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
                k: v.get("leakage_flag") for k, v in per_model.items()
            },
            "panel_csli_choices_excess": (
                round(csli_value, 6) if csli_value is not None else None
            ),
            "panel_csli_choices_excess_definition": (
                "max(0, acc_choices_only - 1/K) per model, averaged "
                "(canonical CSLI; PAP-original definition)"
            ),
            "panel_question_use_gap": (
                round(gap_value, 6) if gap_value is not None else None
            ),
            "panel_question_use_gap_ci": (
                [gap_ci_lower, gap_ci_upper]
                if gap_ci_lower is not None and gap_ci_upper is not None
                else None
            ),
            "panel_question_use_gap_definition": (
                "acc_full - acc_choices_only (formerly published as CSLI "
                "in the in-flight manuscript; kept for transparency)"
            ),
            "per_model_csli_choices_excess": per_model_excess,
            "per_model_question_use_gap": per_model_gap,
            "K": K,
            "chance": chance,
            "artifact_format": artifact_format,
            "definition_note": (
                "Canonical CSLI = max(0, acc_choices_only - 1/K). The "
                "former gap interpretation is published as "
                "question_use_gap. The frozen gate is on "
                "max(acc_choices_only) > 0.30 (= 1/K + 0.05), "
                "independent of either summary."
            ),
        },
    }


def _evaluate_calibration(cal_data: dict, threshold: float) -> dict:
    """Evaluate prefix-wise calibration ECE against threshold.

    PR #14 follow-up review (Issue C): the producer now emits the
    final scientific verdict (downgrades to ``"warn"`` when any
    per-bucket calibrator fell back to ``ConstantCalibrationModel``
    or any test bucket was empty). This consumer prefers the
    producer's ``gate_verdict`` and only recomputes when the
    producer did not record the new ``gate_verdict_reason`` field
    (i.e., the artifact pre-dates the producer-side downgrade fix).
    """
    max_ece = cal_data["max_ece"]
    threshold_verdict = "pass" if max_ece <= threshold else "warn"
    stored_verdict = cal_data["gate_verdict"]
    stored_reason = cal_data.get("gate_verdict_reason")

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

    if stored_reason is not None:
        # Producer already emitted the final scientific verdict; trust
        # it. ``stored_reason`` lets the audit card explain the call.
        final_verdict = stored_verdict
        verdict_reason = stored_reason
    else:
        # Legacy artifact: producer recorded only the threshold-only
        # verdict. Reproduce the downgrade-on-degeneracy decision here
        # so older committed artifacts surface the same WARN.
        if fallback_buckets or empty_buckets:
            final_verdict = "warn"
            verdict_reason = (
                "degenerate_calibrator_or_empty_bucket: "
                f"fallback={[b['bucket'] for b in fallback_buckets]}, "
                f"empty={empty_buckets}"
            )
        else:
            final_verdict = threshold_verdict
            verdict_reason = "threshold_only"

    if final_verdict != stored_verdict:
        print(
            f"WARNING: Calibration verdict mismatch: final={final_verdict}, "
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
        "verdict": final_verdict,
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
            "verdict_reason": verdict_reason,
            "threshold_only_verdict": threshold_verdict,
        },
    }


def _evaluate_stopdff(stopdff_data: dict, threshold: float) -> dict:
    """Evaluate diagnostic StopDFF against threshold.

    PR #14 follow-up review (Blocker 1): when ``ceiling_effect_detected``
    is true (no question in either condition stopped before the final
    prefix) or any per-bucket calibrated threshold is unreachable, the
    metric has no power to detect prefix shifts and a threshold-only PASS
    is scientifically misleading. The producer now emits the final
    scientific verdict (``"warn"`` under either flag) and records
    ``gate_verdict_reason``. This consumer prefers the producer's verdict
    and only re-decides when the artifact pre-dates the producer-side
    downgrade fix (legacy artifacts whose ``gate_verdict`` was
    threshold-only).
    """
    median_shift = stopdff_data["median_abs_prefix_shift"]
    threshold_verdict = "pass" if median_shift <= threshold else "warn"
    stored_verdict = stopdff_data["gate_verdict"]
    stored_reason = stopdff_data.get("gate_verdict_reason")

    ceiling_effect = bool(stopdff_data.get("ceiling_effect_detected", False))
    reachability = stopdff_data.get("reachability") or {}
    unreachable_buckets = [
        bucket
        for bucket, info in reachability.items()
        if isinstance(info, dict) and info.get("threshold_reachable") is False
    ]

    if stored_reason is not None:
        # Producer already emitted the final scientific verdict; trust
        # it. The card still surfaces the qualifier text for the reader.
        final_verdict = stored_verdict
        verdict_reason = stored_reason
    else:
        # Legacy artifact: producer recorded only the threshold-only
        # verdict. Reproduce the downgrade-on-ceiling/unreachable
        # decision so older committed artifacts surface the same WARN.
        if ceiling_effect or unreachable_buckets:
            final_verdict = "warn"
            verdict_reason = "diagnostic_null: " + ", ".join(
                filter(
                    None,
                    [
                        "ceiling_effect" if ceiling_effect else "",
                        (
                            f"unreachable_buckets={sorted(unreachable_buckets)}"
                            if unreachable_buckets
                            else ""
                        ),
                    ],
                )
            )
        else:
            final_verdict = threshold_verdict
            verdict_reason = "threshold_only"

    if final_verdict != stored_verdict:
        print(
            f"WARNING: StopDFF verdict mismatch: final={final_verdict}, "
            f"stored={stored_verdict}",
            file=sys.stderr,
        )

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
        "verdict": final_verdict,
        "verdict_qualifier": verdict_qualifier,
        "details": {
            "direction_breakdown": stopdff_data["direction_breakdown"],
            "stored_gate_verdict": stored_verdict,
            "verdict_reason": verdict_reason,
            "threshold_only_verdict": threshold_verdict,
            "metric_type": stopdff_data["metadata"]["metric_type"],
            "ceiling_effect_detected": ceiling_effect,
            "unreachable_buckets": unreachable_buckets,
            "reachability": reachability,
        },
    }


def _evaluate_stopdff_dp(dp_data: dict) -> dict:
    """Evaluate DP StopDFF (signed median) against +/-1 prefix tolerance.

    Mirrors _evaluate_stopdff for the new finite-horizon DP artifact. Uses
    the same hard threshold (|signed_median| <= 1 prefix) as the diagnostic.
    Surfaces continuation estimator name + confirmatory flag in details.

    PR #15 review (chatgpt-codex-connector top-level on 1057-1059): the DP
    producer's script_sha256 is now cross-checked via
    ``_build_artifact_provenance`` when ``dp_data`` is loaded (i.e. the
    ``--include-dp-stopdff`` flag is on), so stale ``stopdff_dp.json``
    triggers the same WARN downgrade as the other source artifacts.
    """
    signed_median = dp_data["stopdff_dp_signed_median"]
    coverage = dp_data["coverage"]
    verdict = dp_data["gate_verdict"]
    confirmatory = dp_data.get("confirmatory", False)
    qualifier_parts = []
    # PR #15 review (Copilot 3313506967): the producer's gate_verdict only
    # reflects coverage/ceiling checks. Combine with the threshold check on
    # |signed_median| so the displayed criterion and verdict are consistent.
    threshold_verdict = "pass" if abs(signed_median) <= 1 else "warn"
    # Take the stricter outcome (warn dominates pass).
    if threshold_verdict == "warn" or verdict == "warn":
        verdict = "warn"
        if threshold_verdict == "warn":
            qualifier_parts.append(
                f"|signed_median|={abs(signed_median):.4f} > 1"
            )
    if not confirmatory:
        qualifier_parts.append("non-confirmatory continuation estimator")
        # PR #15 review (chatgpt-codex-connector 3313779391): non-confirmatory
        # DP artifacts (e.g. --continuation oracle_trajectory) leak future
        # data and are upper-bound diagnostics only. They must not let the
        # audit card report an overall PASS even when coverage + threshold
        # both pass. Force WARN whenever confirmatory=False.
        if verdict == "pass":
            verdict = "warn"
    if coverage.get("verdict") == "warn":
        qualifier_parts.append(coverage.get("reason", "coverage warn"))
    return {
        "name": "DP StopDFF (Finite-Horizon Bellman, signed median)",
        "value": signed_median,
        "value_display": f"{signed_median:+.4f}",
        "threshold": 1,
        "threshold_criterion": "|signed_median_stopdff| <= 1",
        "observed_criterion_value": abs(signed_median),
        "direction": "warn_if_above",
        "verdict": verdict,
        "verdict_qualifier": "; ".join(qualifier_parts) if qualifier_parts else None,
        "details": {
            "reward_schedule": dp_data["metadata"]["reward_schedule"],
            "continuation_estimator": dp_data["metadata"]["continuation_estimator"],
            "fit_split": dp_data["metadata"]["fit_split"],
            "eval_split": dp_data["metadata"]["eval_split"],
            "coverage": coverage,
            "ceiling_flags": dp_data["ceiling_flags"],
            "n_items": dp_data["n_items"],
            "direction_breakdown": dp_data["direction_breakdown"],
            "confirmatory": confirmatory,
            "metric_type": dp_data["metadata"]["metric_type"],
        },
    }


def _retention_or_coverage_override_qualifiers(
    data_provenance: dict | None,
) -> list[str]:
    """Return formatted strings for any retention/coverage gates that were overridden.

    PR #14 follow-up review (Blocker 2): an audit card whose retention
    or coverage gate failed and was overridden is a "retained-subset"
    result, not a clean PASS. Returns a list like
    ``["calibration/val retention", "csli/test retention"]`` describing
    every gate that triggered an override.
    """
    if not isinstance(data_provenance, dict):
        return []
    qualifiers: list[str] = []
    for metric_name, block in data_provenance.items():
        if not isinstance(block, dict):
            continue
        for gate_name in ("coverage", "retention"):
            gate = block.get(gate_name)
            if not isinstance(gate, dict):
                continue
            for split_name, split_block in gate.items():
                if not isinstance(split_block, dict):
                    continue
                if split_block.get("overridden") is True:
                    qualifiers.append(f"{metric_name}/{split_name} {gate_name}")
    return qualifiers


def _compute_overall_verdict(
    metrics: list[dict],
    data_provenance: dict | None = None,
) -> tuple[str, str | None]:
    """Compute overall verdict from per-metric verdicts.

    PR #14 follow-up review (Blocker 2): any retention or coverage gate
    that failed and was overridden downgrades a clean PASS to WARN with
    a ``retained-subset`` qualifier. Per-metric ``"warn"`` or ``"fail"``
    still dominate the ladder (FAIL > WARN > PASS).

    Returns
    -------
    tuple[str, str | None]
        (overall_verdict, optional_qualifier). The qualifier describes
        why a PASS was downgraded (e.g., listing each overridden gate).
    """
    verdicts = [m["verdict"] for m in metrics]
    if "fail" in verdicts:
        return "FAIL", None
    if "warn" in verdicts:
        return "WARN", None
    overrides = _retention_or_coverage_override_qualifiers(data_provenance)
    if overrides:
        qualifier = "retained-subset (override on " + ", ".join(overrides) + ")"
        return "WARN", qualifier
    return "PASS", None


def _apply_artifact_provenance_to_overall(
    overall_verdict: str,
    overall_verdict_qualifier: str | None,
    artifact_provenance: dict | None,
) -> tuple[str, str | None]:
    """Downgrade or qualify the headline when producer hashes are stale."""
    stale_artifacts = [
        artifact_name
        for artifact_name, block in (artifact_provenance or {}).items()
        if isinstance(block, dict) and block.get("sha_matches") is False
    ]
    if not stale_artifacts:
        return overall_verdict, overall_verdict_qualifier

    provenance_qualifier = "stale producer hash for " + ", ".join(stale_artifacts)
    if overall_verdict == "PASS":
        return "WARN", provenance_qualifier
    if overall_verdict_qualifier:
        return overall_verdict, f"{overall_verdict_qualifier}; {provenance_qualifier}"
    return overall_verdict, provenance_qualifier


def _extract_data_provenance(
    csli_data: dict,
    cal_data: dict,
    stopdff_data: dict,
    dp_data: dict | None = None,
) -> dict:
    """Pull MC coverage + retention metadata from each metric's JSON.

    PR #14 Blocker 3: surface the coverage/retention gate that each
    audit metric ran against so the card reader can verify they
    agreed on what counted as a defensible retained-subset audit.
    Falls back to ``"not_reported"`` markers for metric JSONs that
    pre-date the gate wiring.

    PR #15 review (chatgpt-codex-connector 3313709124): when the DP row is
    included via --include-dp-stopdff and was produced with retention/
    coverage overrides, include the dp's mc_coverage and mc_retention_gate
    blocks so _retention_or_coverage_override_qualifiers downgrades the
    overall verdict accordingly. ``dp_data`` is ``None`` when the flag is
    off; the DP block is only added to the returned dict when ``dp_data``
    was successfully loaded.
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

    result: dict[str, dict] = {
        "csli": _summarize(csli_data, supports_val=False),
        "calibration": _summarize(cal_data, supports_val=True),
        "stopdff": _summarize(stopdff_data, supports_val=False),
    }
    if dp_data is not None:
        result["stopdff_dp"] = _summarize(dp_data, supports_val=True)
    return result


def _write_audit_card_json(
    metrics: list[dict],
    overall_verdict: str,
    data_provenance: dict | None = None,
    overall_verdict_qualifier: str | None = None,
    artifact_provenance: dict | None = None,
    generation: dict | None = None,
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
    if overall_verdict_qualifier is not None:
        card["overall_verdict_qualifier"] = overall_verdict_qualifier
    if data_provenance is not None:
        card["data_provenance"] = data_provenance
    if artifact_provenance is not None:
        card["artifact_provenance"] = artifact_provenance
    if generation is not None:
        card["metadata"]["generation"] = generation
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
    overall_verdict_qualifier: str | None = None,
    artifact_provenance: dict | None = None,
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
        # PR #14 follow-up review (NB-B): the bare ``value | threshold``
        # cells can look unit-mismatched when the gate is on a different
        # quantity than the headline value (e.g., CSLI headline = 0.0035
        # choices-only excess; gate = max(acc_choices_only) ≤ 0.30).
        # Append the ``threshold_criterion`` and ``observed_criterion_value``
        # so the gate's measurement target is unambiguous in the rendered MD.
        threshold_cell = str(m["threshold"])
        criterion = m.get("threshold_criterion")
        observed = m.get("observed_criterion_value")
        if criterion:
            threshold_cell = f"{m['threshold']} ({criterion})"
            if observed is not None and observed != m.get("value"):
                # Surface the actual gate-measurement value when it differs
                # from the headline value (CSLI case; for ECE / StopDFF
                # observed == value so we skip to avoid duplication).
                threshold_cell += f"; observed {observed:.4f}"
        lines.append(
            f"| {m['name']} | {m['value_display']}{ci_str} | {threshold_cell} | "
            f"{_render_verdict_cell(m)} |"
        )
    headline = f"**Overall Verdict: {overall_verdict}**"
    if overall_verdict_qualifier:
        headline += f" — {overall_verdict_qualifier}"
    lines.extend([
        "",
        headline,
    ])
    # PR #14 follow-up review (R5 / Lane D): when any coverage/retention
    # gate was overridden, the audit is a retained-MC-subset audit even
    # if a per-metric ``warn`` collapsed the override qualifier in
    # ``_compute_overall_verdict``. Surface that fact in the headline area
    # so a reader of the MD doesn't need to scan the Data Provenance table
    # to discover the retained-subset nature of the result.
    retained_overrides = _retention_or_coverage_override_qualifiers(data_provenance)
    if retained_overrides and (
        overall_verdict_qualifier is None
        or "retained-subset" not in overall_verdict_qualifier
    ):
        lines.append(
            "*Note: this audit ran on a retained MC subset — coverage or "
            "retention gate was overridden for "
            f"{', '.join(retained_overrides)}. See Data Provenance table below.*"
        )
    lines.append("")

    # PR #14 Blocker 3: surface coverage / retention provenance per
    # metric so the card reader can verify all three metrics agreed
    # on what counted as a defensible retained-subset audit.
    if data_provenance:
        lines.extend(_render_data_provenance_md(data_provenance))

    if artifact_provenance:
        lines.extend(_render_artifact_provenance_md(artifact_provenance))

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


def _render_artifact_provenance_md(provenance: dict) -> list[str]:
    """Render the per-source-artifact script_sha256 provenance block."""
    lines = [
        "## Artifact Provenance — Source Script SHA-256 Match",
        "",
        "| Source Artifact | Recorded commit | Recorded sha256 | Current sha256 | Match |",
        "|-----------------|-----------------|------------------|------------------|-------|",
    ]
    for artifact_name, block in provenance.items():
        if not isinstance(block, dict):
            continue
        commit = block.get("recorded_commit") or "n/a"
        recorded_sha = (block.get("recorded_sha256") or "n/a")[:12]
        current_sha = (block.get("current_sha256") or "n/a")[:12]
        match = block.get("sha_matches")
        match_cell = "yes" if match is True else ("no" if match is False else "n/a")
        lines.append(
            f"| {artifact_name} | {commit[:12]} | {recorded_sha} | {current_sha} | "
            f"{match_cell} |"
        )
    lines.append("")
    return lines


def _build_artifact_provenance(
    csli_data: dict,
    cal_data: dict,
    stopdff_data: dict,
    dp_data: dict | None = None,
) -> dict:
    """Cross-check each source artifact's recorded script sha256 against the live script.

    PR #14 follow-up review (Blocker 4): each producer (compute_csli,
    compute_prefix_calibration, compute_stopdff) now embeds a
    ``generation`` block recording the script's sha256 and the git
    commit at generation time. Here we recompute the live script
    sha256 and surface whether the committed source artifact was
    produced by the current script. Mismatches mean the JSON is stale.

    PR #15 review (chatgpt-codex-connector top-level on lines 1057-1059):
    when ``dp_data`` is supplied (i.e., the ``--include-dp-stopdff`` flag
    loaded ``stopdff_dp.json``), include the DP producer
    (``scripts/compute_stopdff_dp.py``) in the cross-check so a stale
    DP artifact also triggers the WARN downgrade via
    ``_apply_artifact_provenance_to_overall``.
    """
    from scripts._common import sha256_file

    sources = {
        "csli.json": (csli_data, _REPO_ROOT / "scripts" / "compute_csli.py"),
        "calibration.json": (
            cal_data,
            _REPO_ROOT / "scripts" / "compute_prefix_calibration.py",
        ),
        "stopdff.json": (
            stopdff_data,
            _REPO_ROOT / "scripts" / "compute_stopdff.py",
        ),
    }
    if dp_data is not None:
        sources["stopdff_dp.json"] = (
            dp_data,
            _REPO_ROOT / "scripts" / "compute_stopdff_dp.py",
        )
    out: dict[str, dict] = {}
    for name, (data, script_path) in sources.items():
        metadata = data.get("metadata") if isinstance(data, dict) else None
        gen_block = None
        if isinstance(metadata, dict):
            gen_block = metadata.get("generation")
        if not isinstance(gen_block, dict):
            gen_block = data.get("generation") if isinstance(data, dict) else None

        recorded_sha = (
            gen_block.get("script_sha256") if isinstance(gen_block, dict) else None
        )
        recorded_commit = (
            gen_block.get("git_commit") if isinstance(gen_block, dict) else None
        )
        try:
            current_sha = sha256_file(script_path) if script_path.exists() else None
        except OSError:
            current_sha = None
        if recorded_sha is None or current_sha is None:
            match = None
        else:
            match = recorded_sha == current_sha
        out[name] = {
            "recorded_commit": recorded_commit,
            "recorded_sha256": recorded_sha,
            "current_sha256": current_sha,
            "script_path": str(script_path.relative_to(_REPO_ROOT))
            if script_path.exists() and script_path.is_absolute()
            else None,
            "sha_matches": match,
        }
    return out


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

    # PR #15 review (chatgpt-codex-connector 3313709124): include the
    # stopdff_dp slot so an overridden DP row surfaces in the rendered MD
    # provenance table. ``provenance.get`` returns ``None`` when the DP
    # block wasn't emitted (flag off / DP not loaded), and the existing
    # ``if not isinstance(block, dict): continue`` skips it gracefully.
    for metric_name in ("csli", "calibration", "stopdff", "stopdff_dp"):
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


def main_with_args(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

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
    # Opt-in: append a finite-horizon DP StopDFF row alongside (not in place
    # of) the existing diagnostic StopDFF row. The DP row contributes to the
    # overall verdict ladder via _compute_overall_verdict below, so it must
    # be appended BEFORE that call.
    # PR #15 review (chatgpt-codex-connector 3313709124): define dp_data
    # unconditionally (None when the flag is off or the file is absent) so
    # the data_provenance call site can pass it symmetrically and the DP's
    # coverage/retention overrides flow into the overall verdict ladder.
    dp_data: dict | None = None
    if args.include_dp_stopdff:
        dp_path = _PAPER_EXPORTS / "stopdff_dp.json"
        if not dp_path.exists():
            print(
                "WARNING: --include-dp-stopdff was passed but "
                f"{dp_path} does not exist; the DP row was skipped.",
                file=sys.stderr,
            )
        else:
            dp_data = _load_json(dp_path)
            metrics.append(_evaluate_stopdff_dp(dp_data))

    # PR #14 Blocker 3: extract per-metric coverage + retention
    # provenance so the audit card visibly records what counted as a
    # defensible retained-subset audit for each metric.
    data_provenance = _extract_data_provenance(
        csli_data, cal_data, stopdff_data, dp_data=dp_data,
    )

    # PR #14 follow-up review (Blocker 2): retention/coverage overrides
    # downgrade a clean PASS to WARN with a 'retained-subset' qualifier.
    overall_verdict, overall_verdict_qualifier = _compute_overall_verdict(
        metrics, data_provenance
    )

    # PR #14 follow-up review (Blocker 4): cross-check each source
    # artifact's recorded script sha256 against the live script and
    # surface mismatches in the audit card.
    artifact_provenance = _build_artifact_provenance(
        csli_data, cal_data, stopdff_data, dp_data=dp_data,
    )
    sha_mismatches = [
        name
        for name, block in artifact_provenance.items()
        if block.get("sha_matches") is False
    ]
    if sha_mismatches:
        print(
            f"WARNING: source artifact(s) {sha_mismatches} report a "
            f"script_sha256 that no longer matches the live producer "
            f"script. The committed JSON is stale relative to the "
            f"current code. Regenerate before treating it as final "
            f"evidence.",
            file=sys.stderr,
        )
    overall_verdict, overall_verdict_qualifier = (
        _apply_artifact_provenance_to_overall(
            overall_verdict,
            overall_verdict_qualifier,
            artifact_provenance,
        )
    )

    # PR #14 follow-up review (Blocker 4): emit own generation block.
    from scripts._common import build_generation_provenance

    generation = build_generation_provenance(
        __file__,
        list(sys.argv[1:]),
        output_path=_PAPER_EXPORTS / "audit_card.json",
        extra_paths=[
            # Only paths whose dirtiness would invalidate the audit go
            # here. The three upstream pipeline JSONs (csli, calibration,
            # stopdff) intentionally are NOT in this list: they get
            # rewritten by their own producer scripts immediately before
            # make_audit_card runs, so including them guarantees a
            # self-inflicted ``git_dirty: true`` from any clean tree
            # (the previous behavior produced misleading provenance, see
            # bb41819 / 2654ad2 + the audit-card "git_dirty noise" thread
            # in the stale ChatGPT 5.5 ultrareview). Freshness of those
            # three source JSONs is already enforced separately by
            # ``_build_artifact_provenance`` via per-file SHA cross-check
            # against the live producer script.
            _REPO_ROOT / "threshold_manifest.json",
        ],
    )

    # Write outputs
    json_path = _write_audit_card_json(
        metrics,
        overall_verdict,
        data_provenance=data_provenance,
        overall_verdict_qualifier=overall_verdict_qualifier,
        artifact_provenance=artifact_provenance,
        generation=generation,
    )
    md_path = _write_audit_card_md(
        metrics,
        overall_verdict,
        data_provenance=data_provenance,
        overall_verdict_qualifier=overall_verdict_qualifier,
        artifact_provenance=artifact_provenance,
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
    overall_str = f"Overall Verdict: {overall_verdict}"
    if overall_verdict_qualifier:
        overall_str += f" — {overall_verdict_qualifier}"
    print(overall_str)
    print()
    print(f"Written: {json_path}")
    print(f"Written: {md_path}")

    return 0


def main() -> int:
    """CLI entry point; argv comes from sys.argv via _parse_args."""
    return main_with_args(None)


if __name__ == "__main__":
    sys.exit(main())
