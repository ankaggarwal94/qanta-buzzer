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
    0 = full success (or reliability missing with --allow-missing-reliability)
    1 = runtime error or input validation failure
    2 = success but reliability diagrams missing (CI signal)
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple, Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
_PAPER_EXPORTS = _REPO_ROOT / "paper_exports"
_SCRIPT_VERSION = "1.0.0"


def _get_git_sha() -> str:
    """Return short git SHA of HEAD, or 'unknown' on any error."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


class MetricView(NamedTuple):
    """Canonical view of one audit metric, fused from audit_card + JSON.

    Attributes
    ----------
    value : float
        Canonical observed value (prefers audit_card.observed_criterion_value;
        falls back to the metric's JSON field).
    threshold : float
        Canonical threshold (prefers audit_card.threshold; falls back to the
        metric's JSON field, preferring a canonical key over a deprecated alias).
    verdict : str
        ``pass`` / ``warn`` / ``fail`` / ``unknown``.
    verdict_qualifier : Optional[str]
        Free-text qualifier (e.g., "ceiling effect — diagnostic null; ...").
    direction : str
        ``warn_if_above`` (default) or ``warn_if_below``.
    drift_warning : Optional[str]
        Populated when audit_card and JSON carry value/threshold pairs that
        disagree beyond a 1e-9 tolerance. Caller is expected to surface to stderr.
    """

    value: float
    threshold: float
    verdict: str
    verdict_qualifier: Optional[str]
    direction: str
    drift_warning: Optional[str]


def _extract_metric_view(
    metric_name: str | tuple[str, ...],
    audit_card: dict,
    json_data: dict,
    json_value_key: str,
    json_threshold_key: str,
    json_threshold_canonical_key: Optional[str] = None,
) -> MetricView:
    """Build a canonical :class:`MetricView` for one of the three audit metrics.

    Prefers ``audit_card['metrics'][metric_name]`` for value / threshold /
    verdict; falls back to the metric's own JSON file when the audit-card
    entry is missing or has an explicit ``None`` for a field.

    ``metric_name`` may be a tuple of acceptable names (PR #14 follow-up
    review: the audit_card CSLI metric was renamed to include the
    choices-only-excess clarifier; passing a tuple lets the helper match
    either the new or legacy name without burdening every caller).

    If both audit_card and JSON carry a value and they disagree beyond a
    ``1e-9`` tolerance, the resulting :class:`MetricView` ``drift_warning`` is
    populated and the caller is expected to print to stderr.
    """
    metrics = {m["name"]: m for m in audit_card.get("metrics", [])}
    name_candidates = (
        (metric_name,) if isinstance(metric_name, str) else tuple(metric_name)
    )
    entry: dict = {}
    matched_name: Optional[str] = None
    for candidate in name_candidates:
        candidate_entry = metrics.get(candidate)
        if candidate_entry:
            entry = candidate_entry
            matched_name = candidate
            break
    if not entry:
        print(
            f"WARNING: audit_card.json has no metric named any of "
            f"{name_candidates!r}; falling back to "
            f"{json_value_key}/{json_threshold_key} from JSON. "
            f"Verdict will render as 'unknown'.",
            file=sys.stderr,
        )
    metric_name = matched_name or name_candidates[0]

    # Value: audit_card preferred; fall back to JSON.
    ac_value = entry.get("observed_criterion_value")
    json_value = json_data.get(json_value_key)
    value = ac_value if ac_value is not None else json_value
    drift_warning: Optional[str] = None
    if ac_value is not None and json_value is not None:
        if abs(float(ac_value) - float(json_value)) > 1e-9:
            drift_warning = (
                f"{metric_name}: audit_card observed_criterion_value={ac_value} "
                f"disagrees with JSON {json_value_key}={json_value}"
            )

    # Threshold: audit_card first, then canonical JSON name, then deprecated
    # alias. The canonical-vs-alias split lets CSLI prefer
    # ``choices_only_accuracy_threshold`` over the deprecated ``threshold`` key
    # without leaking that policy into every caller.
    threshold = entry.get("threshold")
    if threshold is None:
        if json_threshold_canonical_key is not None:
            threshold = json_data.get(json_threshold_canonical_key)
        if threshold is None:
            threshold = json_data.get(json_threshold_key)

    if value is None or threshold is None:
        raise ValueError(
            f"Could not extract value or threshold for {metric_name}: "
            f"value={value}, threshold={threshold}. "
            f"Check audit_card.json and {json_value_key}/{json_threshold_key} "
            f"in metric JSON."
        )

    return MetricView(
        value=float(value),
        threshold=float(threshold),
        verdict=entry.get("verdict", "unknown"),
        verdict_qualifier=entry.get("verdict_qualifier"),
        direction=entry.get("direction", "warn_if_above"),
        drift_warning=drift_warning,
    )


def _synthesize_calibration_qualifier(audit_card: dict) -> Optional[str]:
    """Synthesize a ``verdict_qualifier`` for calibration force-WARN paths.

    ``make_audit_card.py`` (lines 200-207) forces ``verdict='warn'`` when
    ``fallback_buckets`` or ``empty_buckets`` is non-empty, regardless of
    the threshold comparison. The producer does not emit a
    ``verdict_qualifier`` for calibration, so this function reconstructs
    one from the buckets exposed in the audit-card ``details`` so the
    LaTeX row can surface the force-WARN reason analogously to the StopDFF
    diagnostic-null qualifier.
    """
    metrics = {m["name"]: m for m in audit_card.get("metrics", [])}
    entry = metrics.get("Prefix-wise Calibration (ECE)", {})
    details = entry.get("details") or {}
    fallback = details.get("fallback_buckets") or []
    empty = details.get("empty_buckets") or []
    if not fallback and not empty:
        return None
    parts = []
    if fallback:
        # ``fallback_buckets`` is a list of dict records (see
        # ``make_audit_card.py`` lines 186-198: each entry carries
        # ``bucket``/``reason``/``constant_probability``/``n_samples``).
        # Extract bucket names before joining; tolerate legacy string
        # entries for forward compatibility.
        fallback_names = sorted(
            b.get("bucket", "?") if isinstance(b, dict) else str(b)
            for b in fallback
        )
        parts.append(f"fallback bucket(s): {', '.join(fallback_names)}")
    if empty:
        # ``empty_buckets`` is a list of bucket-name strings.
        parts.append(f"empty bucket(s): {', '.join(sorted(empty))}")
    return "force WARN: " + "; ".join(parts)


_LATEX_ESCAPES = {
    "%": r"\%",
    "#": r"\#",
    "&": r"\&",
    "_": r"\_",
    "$": r"\$",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\^{}",
    "\\": r"\textbackslash{}",
}


def _escape_latex(s: str) -> str:
    """Escape LaTeX-special characters in a free-text string.

    em-dash (``U+2014``) and en-dash (``U+2013``) pass through unchanged --
    the manuscript preamble handles them via ``inputenc`` / ``utf8``.
    """
    if not s:
        return s
    out = []
    for ch in s:
        out.append(_LATEX_ESCAPES.get(ch, ch))
    return "".join(out)


_FOOTNOTE_SYMBOLS = ["\\dagger", "\\ddagger", "\\S", "\\P"]


def _render_verdict_cell(verdict: str, footnote_index: Optional[int]) -> str:
    """Render the LaTeX verdict cell.

    If ``footnote_index`` is provided, append the corresponding footnote
    symbol from :data:`_FOOTNOTE_SYMBOLS` as a superscript. Raises if the
    index exceeds the number of available footnote symbols (currently 4:
    dagger, double-dagger, section, pilcrow).
    """
    base = f"\\textsc{{{verdict}}}"
    if footnote_index is None:
        return base
    if footnote_index >= len(_FOOTNOTE_SYMBOLS):
        raise ValueError(
            f"Too many qualifiers for footnote symbols (max {len(_FOOTNOTE_SYMBOLS)})"
        )
    sym = _FOOTNOTE_SYMBOLS[footnote_index]
    return f"{base}\\textsuperscript{{${sym}$}}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate figures and LaTeX table from cached paper_exports/"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse args and print what would happen without writing files",
    )
    parser.add_argument(
        "--allow-missing-reliability",
        action="store_true",
        default=False,
        help="Suppress non-zero exit code when reliability_*.png are MISSING",
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

    CSLI is rendered as THREE rows to avoid mixing different quantities:

    * "Max choices-only accuracy" -- the gate criterion (``observed_criterion_value``
      in audit_card.json) compared against ``csli_threshold``. This is the
      quantity the CSLI gate actually evaluates: ``max(acc_choices_only) <= 0.30``
      (= 1/K + 0.05), per ``compute_csli.py`` metadata and
      ``threshold_manifest.json``. The verdict in this row is the CSLI gate verdict.
    * "Panel CSLI (choices-only excess)" -- the canonical CSLI summary
      (``panel_csli.mean`` = mean of per-model
      ``max(0, acc_choices_only - 1/K)``) with bootstrap CI. The threshold
      column is rendered as ``--`` because the frozen gate is on
      ``max(acc_choices_only)``, not on the panel mean of either CSLI
      flavor. PR #14 follow-up review (Blocker 3) renamed this from the
      former "Panel CSLI (mean gap)".
    * "Panel question-use gap" -- the legacy in-flight-manuscript CSLI
      summary (``panel_question_use_gap.mean`` = mean of per-model
      ``acc_full - acc_choices_only``) with bootstrap CI. Kept for
      transparency; not the headline CSLI.
    """
    # Descriptive panel summaries stay sourced directly from csli.json --
    # they are not gate quantities and the audit_card does not carry them.
    csli_excess_mean = csli_data["panel_csli"]["mean"]
    csli_excess_ci_lo = csli_data["panel_csli"]["ci_lower"]
    csli_excess_ci_hi = csli_data["panel_csli"]["ci_upper"]
    gap_block = csli_data.get("panel_question_use_gap") or {}
    gap_mean = gap_block.get("mean")
    gap_ci_lo = gap_block.get("ci_lower")
    gap_ci_hi = gap_block.get("ci_upper")

    # Calibration view: synthesize a force-WARN qualifier from the
    # audit-card details before extraction so the canonical view carries
    # it the same way StopDFF carries its diagnostic-null qualifier.
    cal_qualifier_synth = _synthesize_calibration_qualifier(audit_card)

    # CSLI threshold lookup is special: the metric JSON nests metadata one
    # level deep (``csli_data["metadata"]["threshold"]`` or the canonical
    # ``choices_only_accuracy_threshold``), but ``_extract_metric_view``
    # expects a flat ``json_data[key]`` lookup. Build a flat view that
    # exposes both the canonical and deprecated threshold keys AND a
    # pre-computed value fallback (``max(acc_choices_only)``) for older
    # audit cards that lack ``observed_criterion_value``.
    csli_metadata = csli_data.get("metadata", {})
    csli_per_model = csli_data.get("per_model", {})
    csli_json_view = {
        "choices_only_accuracy_threshold": csli_metadata.get(
            "choices_only_accuracy_threshold"
        ),
        "threshold": csli_metadata.get("threshold"),
    }
    # Pre-compute the recompute-from-per_model fallback so the helper can
    # find a value when the audit_card lacks ``observed_criterion_value``.
    # The faux key is descriptive so any failure message reads correctly.
    if csli_per_model:
        csli_json_view["_recomputed_max_acc_choices_only"] = max(
            m["acc_choices_only"] for m in csli_per_model.values()
        )

    csli_view = _extract_metric_view(
        # PR #14 follow-up review (Blocker 3): audit_card metric name now
        # includes the choices-only-excess clarifier. Pass a tuple of
        # acceptable names so this helper finds the row in either the
        # new audit card (choices-excess canonical) or a legacy one.
        (
            "CSLI (Choice-Set Leakage Index, choices-only excess)",
            "CSLI (Choice-Set Leakage Index)",
        ),
        audit_card,
        csli_json_view,
        json_value_key="_recomputed_max_acc_choices_only",
        json_threshold_key="threshold",
        json_threshold_canonical_key="choices_only_accuracy_threshold",
    )

    cal_view = _extract_metric_view(
        "Prefix-wise Calibration (ECE)",
        audit_card,
        cal_data,
        json_value_key="max_ece",
        json_threshold_key="threshold",
    )
    # Inject the synthesized force-WARN qualifier if the producer did not
    # emit one (it never does for calibration). A producer-supplied
    # qualifier, if it ever appears, takes precedence.
    if cal_view.verdict_qualifier is None and cal_qualifier_synth is not None:
        cal_view = cal_view._replace(verdict_qualifier=cal_qualifier_synth)

    stop_view = _extract_metric_view(
        "Diagnostic StopDFF (Median Abs Prefix Shift)",
        audit_card,
        stopdff_data,
        json_value_key="median_abs_prefix_shift",
        json_threshold_key="threshold",
    )

    # Surface any drift between audit_card and JSON-side values to stderr;
    # the LaTeX still renders, but the operator is alerted that one of the
    # cached artifacts is stale relative to the audit card.
    for view in (csli_view, cal_view, stop_view):
        if view.drift_warning:
            print(f"WARNING: {view.drift_warning}", file=sys.stderr)

    # Collect qualifiers into footnote assignments. The map key is the
    # short metric label used below for the verdict cell; only metrics
    # with a qualifier get a footnote symbol.
    footnote_assignments: list[tuple[int, str]] = []  # (symbol_index, escaped_text)
    metric_to_footnote: dict[str, int] = {}
    for name, view in [("CSLI", csli_view), ("CAL", cal_view), ("STOP", stop_view)]:
        if view.verdict_qualifier:
            idx = len(footnote_assignments)
            metric_to_footnote[name] = idx
            footnote_assignments.append((idx, _escape_latex(view.verdict_qualifier)))

    csli_verdict = csli_view.verdict.lower()
    cal_verdict = cal_view.verdict.lower()
    stop_verdict = stop_view.verdict.lower()

    csli_cell = _render_verdict_cell(csli_verdict, metric_to_footnote.get("CSLI"))
    cal_cell = _render_verdict_cell(cal_verdict, metric_to_footnote.get("CAL"))
    stop_cell = _render_verdict_cell(stop_verdict, metric_to_footnote.get("STOP"))

    # PFN-2: Inspect direction across the three views; if homogeneous,
    # surface it in the column header. If mixed, keep the generic
    # ``Threshold`` header and emit a footnote line listing direction
    # per metric (using the next available footnote symbol, or plain
    # text when symbols are exhausted).
    directions = [csli_view.direction, cal_view.direction, stop_view.direction]
    if all(d == "warn_if_above" for d in directions):
        threshold_header = r"Threshold (warn if above)"
        direction_footnote: Optional[tuple[Optional[int], str]] = None
    elif all(d == "warn_if_below" for d in directions):
        threshold_header = r"Threshold (warn if below)"
        direction_footnote = None
    else:
        threshold_header = r"Threshold"
        # Build per-metric direction footnote text.
        _DIRECTION_LABEL = {
            "warn_if_above": "warn if above",
            "warn_if_below": "warn if below",
        }
        direction_parts = [
            f"CSLI: {_DIRECTION_LABEL.get(csli_view.direction, csli_view.direction)}",
            f"Calibration: {_DIRECTION_LABEL.get(cal_view.direction, cal_view.direction)}",
            f"StopDFF: {_DIRECTION_LABEL.get(stop_view.direction, stop_view.direction)}",
        ]
        direction_text = "; ".join(direction_parts)
        # Use the next available footnote symbol after verdict-qualifier
        # footnotes; fall back to plain text without a symbol if we'd
        # overflow the 4-symbol palette.
        next_idx = len(footnote_assignments)
        if next_idx < len(_FOOTNOTE_SYMBOLS):
            direction_footnote = (next_idx, direction_text)
        else:
            direction_footnote = (None, direction_text)

    # Build LaTeX
    lines = [
        # PFN-1: stamp the script version + git SHA at the top of the
        # generated artifact so the source of any rendered table is
        # traceable from the file alone.
        r"% Generated by regenerate_figures.py v" + _SCRIPT_VERSION + " from commit " + _get_git_sha(),
        r"% Requires booktabs package: \usepackage{booktabs}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        f"Metric & Value (95\\% CI) & {threshold_header} & Verdict \\\\",
        r"\midrule",
        # CSLI gate row: the quantity the threshold actually applies to.
        f"Max choices-only accuracy & {csli_view.value:.4f} & {csli_view.threshold:.2f} & {csli_cell} \\\\",
        # Canonical CSLI row: panel choices-only excess (PAP-original)
        # with bootstrap CI. The frozen gate is on max(acc_choices_only),
        # not on this panel mean, so threshold/verdict cells are ``--``.
        f"Panel CSLI (choices-only excess) & {csli_excess_mean:.4f} [{csli_excess_ci_lo:.4f}, {csli_excess_ci_hi:.4f}] & -- & -- \\\\",
        # Transparency row: legacy gap (former in-flight-manuscript CSLI).
        f"Panel question-use gap & {gap_mean:.4f} [{gap_ci_lo:.4f}, {gap_ci_hi:.4f}] & -- & -- \\\\"
        if gap_mean is not None
        else r"% Panel question-use gap row skipped (csli.json predates rename)",
        f"Calibration ECE (max bucket) & {cal_view.value:.4f} & {cal_view.threshold:.2f} & {cal_cell} \\\\",
        f"StopDFF (median abs shift) & {stop_view.value:.1f} & {stop_view.threshold:.1f} & {stop_cell} \\\\",
        r"\bottomrule",
    ]
    # Footnotes after \bottomrule (still inside tabular -- multicolumn rows
    # render as separate lines below the rule). One row per qualifier; the
    # footnote symbol matches the superscript on the verdict cell above.
    for idx, text in footnote_assignments:
        sym = _FOOTNOTE_SYMBOLS[idx]
        lines.append(
            f"\\multicolumn{{4}}{{l}}{{\\footnotesize{{${sym}$\\ {text}}}}} \\\\"
        )
    # PFN-2: append the direction footnote (mixed-direction case) right
    # after \bottomrule and after any verdict-qualifier footnotes.
    if direction_footnote is not None:
        sym_idx, dir_text = direction_footnote
        if sym_idx is not None:
            sym = _FOOTNOTE_SYMBOLS[sym_idx]
            lines.append(
                f"\\multicolumn{{4}}{{l}}{{\\footnotesize{{${sym}$\\ {dir_text}}}}} \\\\"
            )
        else:
            # Footnote-symbol palette exhausted; emit plain text only.
            lines.append(
                f"\\multicolumn{{4}}{{l}}{{\\footnotesize{{{dir_text}}}}} \\\\"
            )
    # PR #14 follow-up review (Lane E FN-2 + FN-3): propagate the audit
    # card's overall_verdict_qualifier and retained-subset status into the
    # paper-facing TeX. Without this propagation, a future regeneration
    # that produces all-PASS metrics + retention overrides would emit a
    # TeX exhibit reading "WARN" with no indication that the audit ran on
    # a retained MC subset under override (the .md surfaces this; the TeX
    # previously did not). Two emission sources:
    #   1. audit_card.overall_verdict_qualifier (set when overrides
    #      promote a clean PASS to WARN; suppressed when a per-metric
    #      WARN already dominates).
    #   2. audit_card.data_provenance: any retention/coverage gate with
    #      overridden=True is a retained-subset signal even when the
    #      qualifier collapsed (mirrors the MD's reader-facing note).
    overall_qualifier = audit_card.get("overall_verdict_qualifier")
    if overall_qualifier:
        lines.append(
            r"\multicolumn{4}{l}{\footnotesize{"
            f"Overall verdict qualifier: {_escape_latex(overall_qualifier)}"
            r"}} \\"
        )
    data_provenance = audit_card.get("data_provenance") or {}
    retained_overrides: list[str] = []
    for metric_name, block in data_provenance.items():
        if not isinstance(block, dict):
            continue
        for gate_name in ("coverage", "retention"):
            gate = block.get(gate_name)
            if not isinstance(gate, dict):
                continue
            for split_name, split_block in gate.items():
                if (
                    isinstance(split_block, dict)
                    and split_block.get("overridden") is True
                ):
                    retained_overrides.append(
                        f"{metric_name}/{split_name} {gate_name}"
                    )
    if retained_overrides and (
        not overall_qualifier or "retained-subset" not in overall_qualifier
    ):
        lines.append(
            r"\multicolumn{4}{l}{\footnotesize{"
            "Retained MC subset (gate overridden for "
            f"{_escape_latex(', '.join(retained_overrides))}"
            ")"
            r"}} \\"
        )
    lines.append(r"\end{tabular}")

    out_path = _PAPER_EXPORTS / "audit_table.tex"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        f.write("\n")
    return out_path


def _generate_csli_panel(csli_data: dict) -> Path:
    """Generate a bar chart showing per-model question-use gap values with panel mean.

    PR #14 follow-up review (Blocker 3): the chart visualizes the
    full-minus-choices gap per model (previously called CSLI in the
    in-flight manuscript; now renamed to ``question_use_gap`` in
    csli.json). The canonical CSLI is the choices-only excess and is
    surfaced separately in the audit table.

    Does NOT draw the choices-only leakage gate (``csli.json`` metadata
    ``threshold``) as a horizontal line on this chart, because that gate
    is a threshold on ``max(acc_choices_only)``, not on the gap. A zero
    reference line is shown instead -- gap > 0 means a model uses
    question content beyond the choices alone; gap < 0 means
    choices-only beats the full-question condition.

    Y-axis limits are derived from both min and max plotted values
    (including CI bounds) with margin, NOT clamped to zero. Per-model
    gap values can legitimately be negative; a hard ``ylim=(0, ...)``
    would silently clip those bars and flip the visual sign.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    per_model = csli_data["per_model"]
    models = list(per_model.keys())
    gap_block = csli_data.get("panel_question_use_gap")
    if isinstance(gap_block, dict) and gap_block.get("mean") is not None:
        # New csli.json format: question_use_gap fields are canonical.
        csli_values = [per_model[m]["question_use_gap"] for m in models]
        panel_mean = gap_block["mean"]
        ci_lower = gap_block["ci_lower"]
        ci_upper = gap_block["ci_upper"]
    else:
        # Legacy csli.json: panel_csli held the gap; per-model ``csli``
        # held the gap. Fall back so older committed artifacts still render.
        csli_values = [per_model[m]["csli"] for m in models]
        panel_mean = csli_data["panel_csli"]["mean"]
        ci_lower = csli_data["panel_csli"]["ci_lower"]
        ci_upper = csli_data["panel_csli"]["ci_upper"]

    # Clean academic style. Width scales with model count so >4 models
    # don't crowd. Floor at 5.0 keeps the baseline 3-model layout intact.
    width = max(5.0, 1.5 + 1.0 * len(models))
    fig, ax = plt.subplots(figsize=(width, 3.5))

    # Explicit numeric x positions so bars and the panel-mean marker
    # share one coordinate system (avoids Matplotlib's categorical vs.
    # numeric axis quirks).
    x = np.arange(len(models))
    mean_x = float(len(models))  # Position to the right of bars

    # PFN-3: bar chart with matplotlib's tab10 colormap (10 distinct
    # colors; doesn't wrap unless models > 10). The prior 3-color list
    # forced cyclic reuse as soon as the panel grew past 3 models, which
    # made the chart visually misleading. Using ``plt.get_cmap`` (not
    # ``cm.get_cmap``) for matplotlib >=3.9 forward-compatibility.
    n = len(models)
    tab10 = plt.get_cmap("tab10")
    if n > 10:
        print(
            f"WARNING: panel has {n} models but tab10 colormap only has 10 "
            "distinct colors; colors will repeat cyclically.",
            file=sys.stderr,
        )
    bar_colors = [tab10(i % 10) for i in range(n)]
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
    # Rotate labels when >4 models to avoid overlap.
    if len(models) > 4:
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    ax.set_ylabel("Question-use gap (acc_full - acc_choices_only)", fontsize=10)
    ax.set_title("Per-model question-use gap (CSLI = choices-excess; not shown)", fontsize=11)
    # Y-limits cover both negative and positive values plus CI bounds,
    # with a small margin. Never hard-clamp at zero (would clip negative
    # per-model CSLI gaps and misreport model behavior).
    # Include panel_mean so the diamond marker can't fall outside the
    # derived y-range (matters when CI is tight but the mean drifts).
    plotted_values = list(csli_values) + [ci_lower, ci_upper, panel_mean, 0.0]
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


def _validate_inputs(
    csli_data: dict,
    cal_data: dict,
    stopdff_data: dict,
) -> int:
    """Validate the three metric JSONs before any output is written.

    Closes the half-build hazard where an audit_table.tex was written
    successfully and then the CSLI panel render failed on a NaN, leaving
    paper_exports/ in a half-coherent state (table reflects a verdict
    the panel cannot illustrate).

    Returns
    -------
    int
        ``0`` if all inputs pass validation, ``1`` on a fatal error.
        A panel-mean-outside-CI condition emits a stderr WARNING but
        does NOT block (return ``0``); it is a data-bug signal, not a
        contract violation.
    """
    per_model = csli_data.get("per_model", {})
    if not per_model:
        print(
            "ERROR: csli.json per_model is empty; cannot generate panel.",
            file=sys.stderr,
        )
        return 1

    # Per-model gap + acc_choices_only must be finite. PR #14 follow-up
    # review (Blocker 3): csli.json schema rename -- per-model 'csli' is
    # now the choices-only excess and 'question_use_gap' carries the gap.
    # Accept either schema so legacy artifacts still validate.
    for model_name, entry in per_model.items():
        gap_val = entry.get("question_use_gap", entry.get("csli"))
        if gap_val is not None and not math.isfinite(float(gap_val)):
            print(
                f"ERROR: csli.json per_model[{model_name!r}].question_use_gap "
                f"(or legacy 'csli') is non-finite ({gap_val}); cannot generate "
                f"panel.",
                file=sys.stderr,
            )
            return 1
        acc_co = entry.get("acc_choices_only")
        if acc_co is not None and not math.isfinite(float(acc_co)):
            print(
                f"ERROR: csli.json per_model[{model_name!r}].acc_choices_only is "
                f"non-finite ({acc_co}); cannot generate audit table.",
                file=sys.stderr,
            )
            return 1

    # Validate whichever panel-mean block carries the gap (new schema
    # publishes it as panel_question_use_gap; legacy under panel_csli).
    gap_panel = csli_data.get("panel_question_use_gap") or csli_data.get(
        "panel_csli", {}
    )
    panel_mean = gap_panel.get("mean")
    ci_lower = gap_panel.get("ci_lower")
    ci_upper = gap_panel.get("ci_upper")
    for field, val in (
        ("panel_question_use_gap.mean (or legacy panel_csli.mean)", panel_mean),
        ("panel_question_use_gap.ci_lower (or legacy)", ci_lower),
        ("panel_question_use_gap.ci_upper (or legacy)", ci_upper),
    ):
        if val is None or not math.isfinite(float(val)):
            print(
                f"ERROR: csli.json {field} is non-finite ({val}); cannot "
                f"generate panel.",
                file=sys.stderr,
            )
            return 1

    max_ece = cal_data.get("max_ece")
    if max_ece is None or not math.isfinite(float(max_ece)):
        print(
            f"ERROR: calibration.json max_ece is non-finite ({max_ece}); "
            f"cannot generate audit table.",
            file=sys.stderr,
        )
        return 1

    median_abs = stopdff_data.get("median_abs_prefix_shift")
    if median_abs is None or not math.isfinite(float(median_abs)):
        print(
            f"ERROR: stopdff.json median_abs_prefix_shift is non-finite "
            f"({median_abs}); cannot generate audit table.",
            file=sys.stderr,
        )
        return 1

    # Soft check: panel_mean outside [ci_lower, ci_upper] is a data-bug
    # signal but not a contract violation -- still warn loudly.
    if not (float(ci_lower) <= float(panel_mean) <= float(ci_upper)):
        print(
            f"WARNING: csli.json question-use-gap panel mean={panel_mean} is "
            f"outside [ci_lower={ci_lower}, ci_upper={ci_upper}]; bootstrap CI "
            f"computation may be stale relative to the panel mean.",
            file=sys.stderr,
        )

    return 0


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

    # Cluster C: validate inputs before writing any files. Closes the
    # half-build hazard where the LaTeX table was emitted before a NaN
    # in panel_csli took down _generate_csli_panel.
    validation_rc = _validate_inputs(csli_data, cal_data, stopdff_data)
    if validation_rc != 0:
        return validation_rc

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
    ] + [(f, f in present) for f in ["reliability_early.png", "reliability_mid.png", "reliability_late.png"]]

    for name, exists in all_outputs:
        status = "OK" if exists else "MISSING"
        print(f"  [{status}] paper_exports/{name}")

    # Cluster D: exit code 2 signals "everything succeeded except the
    # reliability diagrams are MISSING" -- a CI gate can act on this
    # distinctly from a hard failure (rc=1) or full success (rc=0).
    # ``--allow-missing-reliability`` collapses rc=2 to rc=0 for callers
    # that intentionally exclude reliability artifacts.
    if missing and not args.allow_missing_reliability:
        print(
            f"ERROR: {len(missing)} reliability diagram(s) missing; "
            f"exit code 2. Use --allow-missing-reliability to bypass.",
            file=sys.stderr,
        )
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
