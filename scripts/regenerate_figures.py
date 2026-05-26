#!/usr/bin/env python3
"""
Regenerate figures and LaTeX table from cached paper_exports/ data.

Produces publication-ready figures (CSLI panel bar chart) and a LaTeX audit
table (booktabs style) from the cached JSON metric files. Does NOT require
any model inference -- purely reads aggregated results.

Reliability diagrams (reliability_{early,mid,late}.png) are produced by
compute_prefix_calibration.py during Phase 5; this script verifies their
presence but does not regenerate them (the cached PNGs ARE the artifacts).

Usage:
    python scripts/regenerate_figures.py --help
    python scripts/regenerate_figures.py --dry-run    # Parse args, print plan, exit 0
    python scripts/regenerate_figures.py              # Full run

Inputs:
    paper_exports/csli.json             (panel CSLI results with per-model values)
    paper_exports/calibration.json      (per-bucket ECE results)
    paper_exports/stopdff.json          (median abs prefix shift)
    paper_exports/audit_card.json       (aggregated verdicts)

Outputs:
    paper_exports/audit_table.tex       (LaTeX booktabs table for manuscript)
    paper_exports/csli_panel.png        (bar chart of per-model CSLI values)

Exit codes:
    0 = success
    1 = runtime error
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_PAPER_EXPORTS = _REPO_ROOT / "paper_exports"
_SCRIPT_VERSION = "1.0.0"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate figures and LaTeX table from cached paper_exports/"
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
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _generate_audit_table(
    csli_data: dict, cal_data: dict, stopdff_data: dict, audit_card: dict
) -> Path:
    """Generate a LaTeX booktabs table with the three audit metrics.

    Reads verdicts dynamically from audit_card.json rather than hardcoding.

    CSLI is rendered as TWO rows to avoid mixing different quantities:

    * "Max choices-only accuracy" -- the gate criterion (``observed_criterion_value``
      in audit_card.json) compared against ``csli_threshold``. This is the
      quantity the CSLI gate actually evaluates: ``max(acc_choices_only) <= 0.30``
      (= 1/K + 0.05), per ``compute_csli.py`` metadata and
      ``threshold_manifest.json``. The verdict in this row is the CSLI gate verdict.
    * "Panel CSLI (mean gap)" -- the descriptive panel summary
      (``panel_csli.mean`` = mean of per-model ``acc_full - acc_choices_only``)
      with bootstrap CI. The threshold column is rendered as ``--`` because the
      frozen gate is NOT on this gap value (see ``threshold_deprecated_note``
      in ``csli.json``). Splitting the row prevents the LaTeX output from
      presenting a threshold comparison between unrelated quantities, which
      could misstate audit conclusions (e.g., a WARN verdict with a small
      panel mean appearing to sit comfortably "below threshold").
    """
    # Extract values
    csli_mean = csli_data["panel_csli"]["mean"]
    csli_ci_lo = csli_data["panel_csli"]["ci_lower"]
    csli_ci_hi = csli_data["panel_csli"]["ci_upper"]
    max_ece = cal_data["max_ece"]
    median_shift = stopdff_data["median_abs_prefix_shift"]

    # Extract verdicts from audit card (CR-02 fix: dynamic, not hardcoded)
    metrics = {m["name"]: m for m in audit_card["metrics"]}
    csli_metric = metrics.get("CSLI (Choice-Set Leakage Index)", {})
    cal_metric = metrics.get("Prefix-wise Calibration (ECE)", {})
    stop_metric = metrics.get("Diagnostic StopDFF (Median Abs Prefix Shift)", {})
    csli_verdict = csli_metric.get("verdict", "unknown").lower()
    cal_verdict = cal_metric.get("verdict", "unknown").lower()
    stop_verdict = stop_metric.get("verdict", "unknown").lower()
    csli_threshold = float(csli_metric.get("threshold", csli_data["metadata"]["threshold"]))
    cal_threshold = float(cal_metric.get("threshold", cal_data["threshold"]))
    stop_threshold = float(stop_metric.get("threshold", stopdff_data["threshold"]))

    # The CSLI gate's observed criterion value is max(acc_choices_only),
    # which is what ``csli_threshold`` actually compares against. The
    # audit card exposes this as ``observed_criterion_value``; fall back
    # to recomputing from per-model data on older audit cards.
    observed_csli_criterion = csli_metric.get("observed_criterion_value")
    if observed_csli_criterion is None:
        per_model = csli_data.get("per_model", {})
        if per_model:
            observed_csli_criterion = max(
                m["acc_choices_only"] for m in per_model.values()
            )
        else:
            observed_csli_criterion = float("nan")
    observed_csli_criterion = float(observed_csli_criterion)

    # Build LaTeX
    lines = [
        r"% Requires booktabs package: \usepackage{booktabs}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Metric & Value (95\% CI) & Threshold & Verdict \\",
        r"\midrule",
        # CSLI gate row: the quantity the threshold actually applies to.
        f"Max choices-only accuracy & {observed_csli_criterion:.4f} & {csli_threshold:.2f} & \\textsc{{{csli_verdict}}} \\\\",
        # CSLI descriptive row: panel gap with CI, no threshold comparison
        # (``--`` in Threshold and Verdict columns) because the frozen gate
        # is NOT on this quantity.
        f"Panel CSLI (mean gap) & {csli_mean:.4f} [{csli_ci_lo:.4f}, {csli_ci_hi:.4f}] & -- & -- \\\\",
        f"Calibration ECE (max bucket) & {max_ece:.4f} & {cal_threshold:.2f} & \\textsc{{{cal_verdict}}} \\\\",
        f"StopDFF (median abs shift) & {median_shift:.1f} & {stop_threshold:.1f} & \\textsc{{{stop_verdict}}} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
    ]

    out_path = _PAPER_EXPORTS / "audit_table.tex"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        f.write("\n")
    return out_path


def _generate_csli_panel(csli_data: dict) -> Path:
    """Generate a bar chart showing per-model CSLI gap values with panel mean.

    Does NOT draw the choices-only leakage gate (``csli.json`` metadata
    ``threshold``) as a horizontal line on this chart, because that gate is
    a threshold on ``max(acc_choices_only)`` (per ``compute_csli.py``
    metadata: ``threshold_metric == "choices_only_accuracy"``,
    ``threshold_criterion == "acc_choices_only > choices_only_accuracy_threshold"``),
    NOT a threshold on per-model CSLI gap values
    (``acc_full - acc_choices_only``) or ``panel_csli.mean``. Reusing it
    as a reference line in a CSLI gap bar chart would visually imply
    pass/fail against the wrong metric. A zero reference line is shown
    instead -- gap > 0 means a model uses question content beyond the
    choices alone; gap < 0 means choices-only beats the full-question
    condition.

    Uses explicit numeric x positions for both bars and the panel-mean
    marker so the two share a single coordinate system (mixing
    categorical ``ax.bar(strings, ...)`` with numeric tick positions
    can misalign bars/labels across Matplotlib versions).

    Y-axis limits are derived from both min and max plotted values
    (including CI bounds) with margin, NOT clamped to zero. Per-model
    CSLI gap values can legitimately be negative (e.g., TF-IDF in
    ``paper_exports/csli.json`` where ``acc_choices_only > acc_full``);
    a hard ``ylim=(0, ...)`` would silently clip those bars and flip
    the visual sign of the metric.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    per_model = csli_data["per_model"]
    models = list(per_model.keys())
    csli_values = [per_model[m]["csli"] for m in models]
    panel_mean = csli_data["panel_csli"]["mean"]
    ci_lower = csli_data["panel_csli"]["ci_lower"]
    ci_upper = csli_data["panel_csli"]["ci_upper"]

    # Clean academic style
    fig, ax = plt.subplots(figsize=(5, 3.5))

    # Explicit numeric x positions so bars and the panel-mean marker
    # share one coordinate system (avoids Matplotlib's categorical vs.
    # numeric axis quirks).
    x = np.arange(len(models))
    mean_x = float(len(models))  # Position to the right of bars

    # Bar chart with dynamic-length color palette (handles 1+ models)
    palette = ["#4878A8", "#6AAE6A", "#D97B3F"]
    bar_colors = [palette[i % len(palette)] for i in range(len(models))]
    ax.bar(x, csli_values, color=bar_colors, width=0.6, edgecolor="black", linewidth=0.5)

    # Zero reference line: clarifies the sign of the CSLI gap.
    # (Deliberately NOT plotting csli.json metadata.threshold -- that is
    # the choices-only leakage gate, not a threshold on the gap.)
    ax.axhline(y=0.0, color="gray", linestyle=":", linewidth=1.0, label="Zero (gap = 0)")

    # Panel mean with CI error bar
    ax.errorbar(
        mean_x, panel_mean,
        yerr=[[panel_mean - ci_lower], [ci_upper - panel_mean]],
        fmt="D", color="black", markersize=7, capsize=4, linewidth=1.5,
        label=f"Panel mean [{ci_lower:.3f}, {ci_upper:.3f}]",
    )

    # Formatting
    ax.set_xticks(list(x) + [mean_x])
    ax.set_xticklabels(models + ["Panel\nmean"], fontsize=9)
    ax.set_ylabel("CSLI gap (acc_full - acc_choices_only)", fontsize=10)
    ax.set_title("Choice-Set Leakage Index by Model", fontsize=11)
    # Y-limits cover both negative and positive values plus CI bounds,
    # with a small margin. Never hard-clamp at zero (would clip negative
    # per-model CSLI gaps and misreport model behavior).
    plotted_values = list(csli_values) + [ci_lower, ci_upper, 0.0]
    y_min = min(plotted_values)
    y_max = max(plotted_values)
    y_span = y_max - y_min if y_max > y_min else 1.0
    margin = 0.05 * y_span if y_span > 0 else 0.05
    ax.set_ylim(y_min - margin, y_max + margin)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)

    plt.tight_layout()
    out_path = _PAPER_EXPORTS / "csli_panel.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _check_reliability_diagrams() -> tuple[list[str], list[str]]:
    """Check whether reliability diagrams exist (produced by compute_prefix_calibration.py).

    Returns
    -------
    tuple[list[str], list[str]]
        (present, missing) where each is a list of filenames.
    """
    expected = ["reliability_early.png", "reliability_mid.png", "reliability_late.png"]
    present = []
    missing = []
    for fname in expected:
        path = _PAPER_EXPORTS / fname
        if path.exists():
            present.append(fname)
        else:
            missing.append(fname)
    return present, missing


def main() -> int:
    args = _parse_args()

    print("=== Figure and Table Regeneration ===")
    print()

    if args.dry_run:
        # Must enumerate every input the non-dry-run path loads (see
        # ``_load_json`` calls in ``main()`` below). Omitting any of these
        # makes the dry-run plan misleading -- e.g., earlier versions left
        # off audit_card.json and operators could not tell from --dry-run
        # that it was a hard dependency.
        print("[DRY-RUN] Would load:")
        print(f"  - {_PAPER_EXPORTS / 'csli.json'}")
        print(f"  - {_PAPER_EXPORTS / 'calibration.json'}")
        print(f"  - {_PAPER_EXPORTS / 'stopdff.json'}")
        print(f"  - {_PAPER_EXPORTS / 'audit_card.json'}")
        print()
        print("[DRY-RUN] Would generate:")
        print(f"  - {_PAPER_EXPORTS / 'audit_table.tex'}")
        print(f"  - {_PAPER_EXPORTS / 'csli_panel.png'}")
        print()
        print("[DRY-RUN] Would verify:")
        print(f"  - {_PAPER_EXPORTS / 'reliability_early.png'}")
        print(f"  - {_PAPER_EXPORTS / 'reliability_mid.png'}")
        print(f"  - {_PAPER_EXPORTS / 'reliability_late.png'}")
        return 0

    # Load inputs
    csli_data = _load_json(_PAPER_EXPORTS / "csli.json")
    cal_data = _load_json(_PAPER_EXPORTS / "calibration.json")
    stopdff_data = _load_json(_PAPER_EXPORTS / "stopdff.json")
    audit_card = _load_json(_PAPER_EXPORTS / "audit_card.json")

    # Generate audit table (LaTeX) with dynamic verdicts from audit_card.json
    tex_path = _generate_audit_table(csli_data, cal_data, stopdff_data, audit_card)
    print(f"Generated: {tex_path}")

    # Generate CSLI panel bar chart
    png_path = _generate_csli_panel(csli_data)
    print(f"Generated: {png_path}")

    # Check reliability diagrams
    present, missing = _check_reliability_diagrams()
    if present:
        print(f"Reliability diagrams already present (from compute_prefix_calibration.py): {', '.join(present)}")
    if missing:
        print(
            f"WARNING: Missing reliability diagrams: {', '.join(missing)}. "
            "Full recompute via compute_prefix_calibration.py is needed "
            "(cached calibration.json lacks per-bin counts)."
        )

    # Print manifest
    print()
    print("=== Output Manifest ===")
    all_outputs = [
        ("audit_table.tex", tex_path.exists()),
        ("csli_panel.png", png_path.exists()),
    ] + [(f, f in [p for p in present]) for f in ["reliability_early.png", "reliability_mid.png", "reliability_late.png"]]

    for name, exists in all_outputs:
        status = "OK" if exists else "MISSING"
        print(f"  [{status}] paper_exports/{name}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
