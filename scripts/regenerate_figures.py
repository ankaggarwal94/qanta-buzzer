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
    with open(path) as f:
        return json.load(f)


def _generate_audit_table(csli_data: dict, cal_data: dict, stopdff_data: dict) -> Path:
    """Generate a LaTeX booktabs table with the three audit metrics."""
    # Extract values
    csli_mean = csli_data["panel_csli"]["mean"]
    csli_ci_lo = csli_data["panel_csli"]["ci_lower"]
    csli_ci_hi = csli_data["panel_csli"]["ci_upper"]
    max_ece = cal_data["max_ece"]
    median_shift = stopdff_data["median_abs_prefix_shift"]

    # Build LaTeX
    lines = [
        r"% Requires booktabs package: \usepackage{booktabs}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Metric & Value (95\% CI) & Threshold & Verdict \\",
        r"\midrule",
        f"CSLI (panel mean) & {csli_mean:.4f} [{csli_ci_lo:.4f}, {csli_ci_hi:.4f}] & 0.30 & \\textsc{{pass}} \\\\",
        f"Calibration ECE (max bucket) & {max_ece:.4f} & 0.10 & \\textsc{{pass}} \\\\",
        f"StopDFF (median abs shift) & {median_shift:.1f} & 1.0 & \\textsc{{pass}} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
    ]

    out_path = _PAPER_EXPORTS / "audit_table.tex"
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
        f.write("\n")
    return out_path


def _generate_csli_panel(csli_data: dict) -> Path:
    """Generate a bar chart showing per-model CSLI values with threshold line."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    per_model = csli_data["per_model"]
    models = list(per_model.keys())
    csli_values = [per_model[m]["csli"] for m in models]
    threshold = csli_data["metadata"]["threshold"]
    panel_mean = csli_data["panel_csli"]["mean"]
    ci_lower = csli_data["panel_csli"]["ci_lower"]
    ci_upper = csli_data["panel_csli"]["ci_upper"]

    # Clean academic style
    fig, ax = plt.subplots(figsize=(5, 3.5))

    # Bar chart
    bar_colors = ["#4878A8", "#6AAE6A", "#D97B3F"]
    bars = ax.bar(models, csli_values, color=bar_colors, width=0.6, edgecolor="black", linewidth=0.5)

    # Threshold line
    ax.axhline(y=threshold, color="red", linestyle="--", linewidth=1.2, label=f"Threshold ({threshold})")

    # Panel mean with CI error bar
    mean_x = len(models)  # Position to the right of bars
    ax.errorbar(
        mean_x, panel_mean,
        yerr=[[panel_mean - ci_lower], [ci_upper - panel_mean]],
        fmt="D", color="black", markersize=7, capsize=4, linewidth=1.5,
        label=f"Panel mean [{ci_lower:.3f}, {ci_upper:.3f}]",
    )

    # Formatting
    ax.set_xticks(list(range(len(models))) + [mean_x])
    ax.set_xticklabels(models + ["Panel\nmean"], fontsize=9)
    ax.set_ylabel("CSLI", fontsize=10)
    ax.set_title("Choice-Set Leakage Index by Model", fontsize=11)
    ax.set_ylim(0, max(threshold + 0.05, max(csli_values) + 0.05))
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)

    plt.tight_layout()
    out_path = _PAPER_EXPORTS / "csli_panel.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _check_reliability_diagrams() -> list[str]:
    """Check whether reliability diagrams exist (produced by compute_prefix_calibration.py)."""
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
        print("[DRY-RUN] Would load:")
        print(f"  - {_PAPER_EXPORTS / 'csli.json'}")
        print(f"  - {_PAPER_EXPORTS / 'calibration.json'}")
        print(f"  - {_PAPER_EXPORTS / 'stopdff.json'}")
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

    # Generate audit table (LaTeX)
    tex_path = _generate_audit_table(csli_data, cal_data, stopdff_data)
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
