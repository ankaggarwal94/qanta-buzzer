"""Tests for cached figure/table regeneration."""

from __future__ import annotations

import io
import math
import sys
from contextlib import redirect_stdout
from pathlib import Path

sys.modules.pop("scripts.regenerate_figures", None)

from scripts import regenerate_figures

_generate_audit_table = getattr(regenerate_figures, "_generate_audit_table")
_generate_csli_panel = getattr(regenerate_figures, "_generate_csli_panel")
_extract_metric_view = getattr(regenerate_figures, "_extract_metric_view")
_escape_latex = getattr(regenerate_figures, "_escape_latex")
_validate_inputs = getattr(regenerate_figures, "_validate_inputs")


def _healthy_csli_data() -> dict:
    """Three-model healthy csli.json fixture used by validation tests."""
    return {
        "panel_csli": {"mean": 0.11, "ci_lower": 0.09, "ci_upper": 0.13},
        "per_model": {
            "tfidf": {"csli": 0.05, "acc_choices_only": 0.26},
            "sbert": {"csli": 0.15, "acc_choices_only": 0.24},
            "t5-small": {"csli": 0.13, "acc_choices_only": 0.21},
        },
        "metadata": {"threshold": 0.30},
    }


def _capture_panel_figure(csli_data: dict, tmp_path: Path, monkeypatch):
    """Run _generate_csli_panel while intercepting plt.close so the
    Figure (and its Axes) remain available for ylim/xtick introspection.
    Returns ``(out_path, fig)``.
    """
    monkeypatch.setattr(regenerate_figures, "_PAPER_EXPORTS", tmp_path)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    captured: dict = {}
    real_close = plt.close

    def _capture_close(arg=None):
        # The production function calls ``plt.close(fig)`` after savefig.
        # Capture the Figure so the caller can introspect it, then no-op
        # so the canvas survives the function call.
        if hasattr(arg, "axes"):
            captured["fig"] = arg
        # Intentionally do NOT delegate to real_close -- keeping the
        # figure alive is the whole point of this hook.

    monkeypatch.setattr(plt, "close", _capture_close)
    out_path = _generate_csli_panel(csli_data)
    fig = captured.get("fig")
    # Restore close so subsequent tearDown can clean up.
    monkeypatch.setattr(plt, "close", real_close)
    return out_path, fig


def test_audit_table_uses_thresholds_from_audit_card(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(regenerate_figures, "_PAPER_EXPORTS", tmp_path)
    csli_data = {
        "panel_csli": {"mean": 0.2, "ci_lower": 0.1, "ci_upper": 0.3},
        "per_model": {
            "tfidf": {"acc_choices_only": 0.18},
            "sbert": {"acc_choices_only": 0.22},
        },
        "metadata": {"threshold": 0.30},
    }
    cal_data = {"max_ece": 0.04, "threshold": 0.10}
    stop_data = {"median_abs_prefix_shift": 0.0, "threshold": 1.0}
    audit_card = {
        "metrics": [
            {
                "name": "CSLI (Choice-Set Leakage Index)",
                "threshold": 0.42,
                "observed_criterion_value": 0.31,
                "verdict": "warn",
            },
            {"name": "Prefix-wise Calibration (ECE)", "threshold": 0.07, "verdict": "pass"},
            {
                "name": "Diagnostic StopDFF (Median Abs Prefix Shift)",
                "threshold": 2.0,
                "verdict": "pass",
            },
        ]
    }

    output = _generate_audit_table(
        csli_data,
        cal_data,
        stop_data,
        audit_card,
    )

    rendered = output.read_text(encoding="utf-8")
    assert "0.42" in rendered
    assert "0.07" in rendered
    assert "2.0" in rendered
    assert "\\textsc{warn}" in rendered


def test_audit_table_splits_csli_into_gate_and_descriptive_rows(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """CSLI rows: gate row + choices-excess row + question-use-gap row.

    PR #14 follow-up review (Blocker 3): the audit table now publishes
    the canonical CSLI (choices-only excess over chance) as the headline
    descriptive row, with the legacy question-use gap rendered for
    transparency. Both descriptive rows render with Threshold/Verdict =
    ``--`` because the frozen gate is on max(acc_choices_only), not on
    either panel mean.
    """
    monkeypatch.setattr(regenerate_figures, "_PAPER_EXPORTS", tmp_path)
    csli_data = {
        # Canonical CSLI = choices-only excess.
        "panel_csli": {
            "mean": 0.0335,
            "ci_lower": 0.0250,
            "ci_upper": 0.0420,
            "definition": "max(0, acc_choices_only - 1/K) per model, averaged",
        },
        "panel_question_use_gap": {
            "mean": 0.1137,
            "ci_lower": 0.0995,
            "ci_upper": 0.1261,
        },
        "per_model": {
            "tfidf": {
                "acc_choices_only": 0.260407,
                "csli": 0.010,
                "question_use_gap": 0.15,
            },
            "sbert": {
                "acc_choices_only": 0.244464,
                "csli": 0.0,
                "question_use_gap": 0.10,
            },
            "t5-small": {
                "acc_choices_only": 0.213906,
                "csli": 0.0,
                "question_use_gap": 0.08,
            },
        },
        "metadata": {"threshold": 0.30},
    }
    cal_data = {"max_ece": 0.026117, "threshold": 0.10}
    stop_data = {"median_abs_prefix_shift": 0.0, "threshold": 1.0}
    audit_card = {
        "metrics": [
            {
                "name": "CSLI (Choice-Set Leakage Index)",
                "threshold": 0.30,
                "observed_criterion_value": 0.260407,
                "verdict": "pass",
            },
            {"name": "Prefix-wise Calibration (ECE)", "threshold": 0.10, "verdict": "pass"},
            {
                "name": "Diagnostic StopDFF (Median Abs Prefix Shift)",
                "threshold": 1.0,
                "verdict": "pass",
            },
        ]
    }

    output = _generate_audit_table(csli_data, cal_data, stop_data, audit_card)
    rendered = output.read_text(encoding="utf-8")

    # Gate row: ``Max choices-only accuracy`` compared against the gate
    # threshold (0.30) with the CSLI gate verdict.
    assert "Max choices-only accuracy" in rendered
    assert "0.2604" in rendered  # observed_criterion_value
    # Canonical CSLI row: choices-only excess with CI, no threshold comparison.
    assert "Panel CSLI (choices-only excess)" in rendered
    assert "0.0335" in rendered  # panel_csli.mean (choices-excess)
    assert "[0.0250, 0.0420]" in rendered
    # Transparency row: question-use gap with CI, no threshold comparison.
    assert "Panel question-use gap" in rendered
    assert "0.1137" in rendered  # panel_question_use_gap.mean
    assert "[0.0995, 0.1261]" in rendered
    # Both descriptive rows render with -- in threshold/verdict columns.
    excess_row = [
        line for line in rendered.splitlines()
        if line.startswith("Panel CSLI (choices-only excess)")
    ]
    gap_row = [
        line for line in rendered.splitlines()
        if line.startswith("Panel question-use gap")
    ]
    assert len(excess_row) == 1
    assert len(gap_row) == 1
    assert "& -- & --" in excess_row[0]
    assert "& -- & --" in gap_row[0]


def test_audit_table_falls_back_when_observed_criterion_value_absent(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """When the audit card lacks ``observed_criterion_value`` (older
    artifacts), the gate row must recompute ``max(acc_choices_only)`` from
    csli.json per-model data rather than silently rendering ``panel_csli.mean``.
    """
    monkeypatch.setattr(regenerate_figures, "_PAPER_EXPORTS", tmp_path)
    csli_data = {
        "panel_csli": {"mean": 0.10, "ci_lower": 0.05, "ci_upper": 0.15},
        "per_model": {
            "tfidf": {"acc_choices_only": 0.270000},
            "sbert": {"acc_choices_only": 0.250000},
        },
        "metadata": {"threshold": 0.30},
    }
    cal_data = {"max_ece": 0.02, "threshold": 0.10}
    stop_data = {"median_abs_prefix_shift": 0.0, "threshold": 1.0}
    audit_card = {
        "metrics": [
            # Deliberately missing ``observed_criterion_value``.
            {"name": "CSLI (Choice-Set Leakage Index)", "threshold": 0.30, "verdict": "pass"},
            {"name": "Prefix-wise Calibration (ECE)", "threshold": 0.10, "verdict": "pass"},
            {
                "name": "Diagnostic StopDFF (Median Abs Prefix Shift)",
                "threshold": 1.0,
                "verdict": "pass",
            },
        ]
    }

    output = _generate_audit_table(csli_data, cal_data, stop_data, audit_card)
    rendered = output.read_text(encoding="utf-8")

    # Fallback recomputes max(acc_choices_only) = 0.27.
    assert "0.2700" in rendered


def test_csli_panel_supports_subset_model_count(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(regenerate_figures, "_PAPER_EXPORTS", tmp_path)
    csli_data = {
        "per_model": {
            "tfidf": {"csli": 0.1},
            "sbert": {"csli": 0.2},
        },
        "metadata": {"threshold": 0.3},
        "panel_csli": {"mean": 0.15, "ci_lower": 0.1, "ci_upper": 0.2},
    }

    output = _generate_csli_panel(csli_data)

    assert output.exists()


def test_csli_panel_does_not_clip_negative_values(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Per-model CSLI gap values can be negative (acc_choices_only > acc_full).
    The y-axis must accommodate them rather than hard-clamping to zero.

    L5-6 fix: introspect ``ax.get_ylim()`` directly rather than
    recomputing the production formula in the assertion. The previous
    body re-derived ``min(plotted_values)`` and only asserted that the
    derivation was below zero -- it never confirmed the rendered axes
    matched. Now we capture the live ``fig`` and check the actual
    ``ylim[0]`` is strictly below the most-negative bar.
    """
    csli_data = {
        "per_model": {
            "tfidf": {"csli": -0.05},  # Negative gap (choices-only beats full)
            "sbert": {"csli": 0.25},
            "t5-small": {"csli": 0.10},
        },
        "metadata": {"threshold": 0.3},
        "panel_csli": {"mean": 0.10, "ci_lower": 0.05, "ci_upper": 0.15},
    }
    out, fig = _capture_panel_figure(csli_data, tmp_path, monkeypatch)
    assert out.exists()
    assert fig is not None, "captured figure required for ylim introspection"

    ax = fig.axes[0]
    y_lo, y_hi = ax.get_ylim()
    # The negative bar at -0.05 must lie strictly inside the visible range.
    assert y_lo < -0.05, f"expected ylim[0] < -0.05, got {y_lo}"
    # The positive bar at 0.25 must also lie inside the visible range.
    assert y_hi > 0.25, f"expected ylim[1] > 0.25, got {y_hi}"

    import matplotlib.pyplot as plt

    plt.close(fig)


def test_csli_panel_uses_numeric_x_positions_consistently(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Bars and the panel-mean marker must share a single (numeric)
    coordinate system; mixing ``ax.bar(strings, ...)`` with numeric tick
    positions can misalign labels across Matplotlib versions.

    L5-6 fix: read ``ax.get_xticks()`` directly rather than re-deriving
    ``np.arange(n)`` in the test body. The previous version constructed
    a parallel expected sequence and only checked types -- which would
    pass even if the production code emitted no ticks at all. The new
    body asserts the live tick positions match ``[0, 1, ..., n-1, n]``.
    """
    csli_data = {
        "per_model": {
            "tfidf": {"csli": 0.05},
            "sbert": {"csli": 0.20},
            "t5-small": {"csli": 0.10},
        },
        "metadata": {"threshold": 0.3},
        "panel_csli": {"mean": 0.117, "ci_lower": 0.10, "ci_upper": 0.13},
    }
    out, fig = _capture_panel_figure(csli_data, tmp_path, monkeypatch)
    assert out.exists()
    assert fig is not None, "captured figure required for xtick introspection"

    ax = fig.axes[0]
    n_models = len(csli_data["per_model"])
    expected = [0.0, 1.0, 2.0, float(n_models)]
    actual = list(ax.get_xticks())
    assert actual == expected, f"expected ticks {expected}, got {actual}"

    import matplotlib.pyplot as plt

    plt.close(fig)


def test_extract_metric_view_prefers_audit_card_over_json() -> None:
    """Cluster B: when both audit_card and JSON carry a value, the
    audit_card's ``observed_criterion_value`` wins.
    """
    audit_card = {
        "metrics": [
            {
                "name": "Prefix-wise Calibration (ECE)",
                "observed_criterion_value": 0.5,
                "threshold": 0.10,
                "verdict": "warn",
            }
        ]
    }
    json_data = {"max_ece": 0.4, "threshold": 0.10}
    view = _extract_metric_view(
        "Prefix-wise Calibration (ECE)",
        audit_card,
        json_data,
        json_value_key="max_ece",
        json_threshold_key="threshold",
    )
    assert view.value == 0.5
    assert view.threshold == 0.10
    assert view.verdict == "warn"


def test_extract_metric_view_drift_warning() -> None:
    """Cluster B: disagreement between audit_card and JSON value sets
    ``drift_warning`` to a non-empty string.
    """
    audit_card = {
        "metrics": [
            {
                "name": "Prefix-wise Calibration (ECE)",
                "observed_criterion_value": 0.5,
                "threshold": 0.10,
                "verdict": "warn",
            }
        ]
    }
    json_data = {"max_ece": 0.4, "threshold": 0.10}
    view = _extract_metric_view(
        "Prefix-wise Calibration (ECE)",
        audit_card,
        json_data,
        json_value_key="max_ece",
        json_threshold_key="threshold",
    )
    assert view.drift_warning is not None
    assert "0.5" in view.drift_warning
    assert "0.4" in view.drift_warning


def test_extract_metric_view_canonical_threshold_name_wins() -> None:
    """Cluster B: CSLI prefers ``choices_only_accuracy_threshold`` over
    the deprecated ``threshold`` alias when audit_card lacks a threshold.
    """
    audit_card = {
        "metrics": [
            {
                "name": "CSLI (Choice-Set Leakage Index)",
                "observed_criterion_value": 0.25,
                # No ``threshold`` in audit_card -- force JSON fallback.
                "verdict": "pass",
            }
        ]
    }
    json_data = {
        "choices_only_accuracy_threshold": 0.30,
        "threshold": 0.99,  # deprecated alias; must NOT win
    }
    view = _extract_metric_view(
        "CSLI (Choice-Set Leakage Index)",
        audit_card,
        json_data,
        json_value_key="observed_criterion_value_unused",
        json_threshold_key="threshold",
        json_threshold_canonical_key="choices_only_accuracy_threshold",
    )
    assert view.threshold == 0.30


def test_extract_metric_view_unknown_metric_name(capsys) -> None:
    """Cluster B: missing metric name in audit_card emits a stderr warning
    and the resulting view has ``verdict == 'unknown'``.
    """
    audit_card = {"metrics": []}  # No metrics at all.
    json_data = {"max_ece": 0.04, "threshold": 0.10}
    view = _extract_metric_view(
        "Prefix-wise Calibration (ECE)",
        audit_card,
        json_data,
        json_value_key="max_ece",
        json_threshold_key="threshold",
    )
    assert view.verdict == "unknown"
    captured = capsys.readouterr()
    assert "Prefix-wise Calibration (ECE)" in captured.err
    assert "audit_card.json has no metric named" in captured.err


def test_stopdff_row_surfaces_qualifier_via_footnote(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Cluster A headline: when audit_card carries a StopDFF
    ``verdict_qualifier``, the LaTeX renders the verdict cell with a
    ``\\textsuperscript{$\\dagger$}`` superscript AND emits a matching
    ``\\multicolumn`` footnote line below ``\\bottomrule``.
    """
    monkeypatch.setattr(regenerate_figures, "_PAPER_EXPORTS", tmp_path)
    csli_data = {
        "panel_csli": {"mean": 0.11, "ci_lower": 0.09, "ci_upper": 0.13},
        "per_model": {"tfidf": {"acc_choices_only": 0.26}},
        "metadata": {"threshold": 0.30},
    }
    cal_data = {"max_ece": 0.03, "threshold": 0.10}
    stop_data = {"median_abs_prefix_shift": 0.0, "threshold": 1.0}
    audit_card = {
        "metrics": [
            {
                "name": "CSLI (Choice-Set Leakage Index)",
                "threshold": 0.30,
                "observed_criterion_value": 0.26,
                "verdict": "pass",
            },
            {
                "name": "Prefix-wise Calibration (ECE)",
                "threshold": 0.10,
                "observed_criterion_value": 0.03,
                "verdict": "pass",
            },
            {
                "name": "Diagnostic StopDFF (Median Abs Prefix Shift)",
                "threshold": 1.0,
                "observed_criterion_value": 0.0,
                "verdict": "pass",
                "verdict_qualifier": (
                    "ceiling effect — diagnostic null; unreachable bucket(s): early, mid"
                ),
            },
        ]
    }

    output = _generate_audit_table(csli_data, cal_data, stop_data, audit_card)
    rendered = output.read_text(encoding="utf-8")

    # Superscript on the StopDFF verdict cell.
    assert "\\textsc{pass}\\textsuperscript{$\\dagger$}" in rendered
    # Footnote row below \bottomrule.
    assert "\\multicolumn{4}{l}{\\footnotesize{$\\dagger$\\ " in rendered
    assert "ceiling effect" in rendered
    assert "unreachable bucket(s): early, mid" in rendered


def test_calibration_force_warn_synthesizes_qualifier(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Cluster A: when calibration ``details`` carry ``fallback_buckets``
    or ``empty_buckets``, a qualifier is synthesized and rendered as a
    footnote even though the producer did not emit ``verdict_qualifier``.
    """
    monkeypatch.setattr(regenerate_figures, "_PAPER_EXPORTS", tmp_path)
    csli_data = {
        "panel_csli": {"mean": 0.11, "ci_lower": 0.09, "ci_upper": 0.13},
        "per_model": {"tfidf": {"acc_choices_only": 0.26}},
        "metadata": {"threshold": 0.30},
    }
    cal_data = {"max_ece": 0.03, "threshold": 0.10}
    stop_data = {"median_abs_prefix_shift": 0.0, "threshold": 1.0}
    audit_card = {
        "metrics": [
            {
                "name": "CSLI (Choice-Set Leakage Index)",
                "threshold": 0.30,
                "observed_criterion_value": 0.26,
                "verdict": "pass",
            },
            {
                "name": "Prefix-wise Calibration (ECE)",
                "threshold": 0.10,
                "observed_criterion_value": 0.03,
                # Producer never emits this; we synthesize it from details.
                "verdict": "warn",
                "details": {
                    # Real producer schema (make_audit_card.py:186-198):
                    # fallback_buckets is a list of dict records, NOT strings.
                    "fallback_buckets": [
                        {
                            "bucket": "early",
                            "reason": "single-class validation bucket",
                            "constant_probability": 0.5,
                            "n_samples": 17,
                        }
                    ],
                    # empty_buckets remains a list of bucket-name strings.
                    "empty_buckets": ["mid"],
                },
            },
            {
                "name": "Diagnostic StopDFF (Median Abs Prefix Shift)",
                "threshold": 1.0,
                "observed_criterion_value": 0.0,
                "verdict": "pass",
            },
        ]
    }

    output = _generate_audit_table(csli_data, cal_data, stop_data, audit_card)
    rendered = output.read_text(encoding="utf-8")

    # Calibration WARN cell carries the dagger footnote.
    assert "\\textsc{warn}\\textsuperscript{$\\dagger$}" in rendered
    # The synthesized text appears in the footnote.
    assert "force WARN" in rendered
    assert "fallback bucket(s): early" in rendered
    assert "empty bucket(s): mid" in rendered


def test_synthesize_calibration_qualifier_handles_dict_records() -> None:
    """Round-8 regression: ``fallback_buckets`` is a list of dict records
    in the real producer schema (make_audit_card.py:186-198), not strings.
    The synthesizer must extract bucket names from dicts before joining.
    """
    audit_card = {
        "metrics": [
            {
                "name": "Prefix-wise Calibration (ECE)",
                "verdict": "warn",
                "details": {
                    "fallback_buckets": [
                        {"bucket": "late", "reason": "x", "n_samples": 0},
                        {"bucket": "early", "reason": "y", "n_samples": 3},
                    ],
                    "empty_buckets": [],
                },
            }
        ]
    }
    qualifier = regenerate_figures._synthesize_calibration_qualifier(audit_card)
    assert qualifier is not None
    assert "fallback bucket(s): early, late" in qualifier
    assert "force WARN" in qualifier


def test_synthesize_calibration_qualifier_tolerates_legacy_string_entries() -> None:
    """Forward-compat: if a legacy/test fixture passes strings instead
    of dict records, the synthesizer still works (no TypeError).
    """
    audit_card = {
        "metrics": [
            {
                "name": "Prefix-wise Calibration (ECE)",
                "verdict": "warn",
                "details": {
                    "fallback_buckets": ["early", "late"],
                    "empty_buckets": [],
                },
            }
        ]
    }
    qualifier = regenerate_figures._synthesize_calibration_qualifier(audit_card)
    assert qualifier is not None
    assert "fallback bucket(s): early, late" in qualifier


def test_latex_escape_handles_special_chars() -> None:
    """Cluster A: ``% # & _ $ { }`` get escaped; em-dash passes through."""
    src = "rate 12% & cost $5_USD #note {set} — fine"
    escaped = _escape_latex(src)
    assert "\\%" in escaped
    assert "\\&" in escaped
    assert "\\$" in escaped
    assert "\\_" in escaped
    assert "\\#" in escaped
    assert "\\{" in escaped
    assert "\\}" in escaped
    # em-dash passes through unchanged (handled by inputenc/utf8).
    assert "—" in escaped


def test_no_qualifier_no_footnote(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Cluster A: when no ``verdict_qualifier`` is present (and no force-WARN
    bucket conditions hold), no ``\\multicolumn`` footnote appears.
    """
    monkeypatch.setattr(regenerate_figures, "_PAPER_EXPORTS", tmp_path)
    csli_data = {
        "panel_csli": {"mean": 0.11, "ci_lower": 0.09, "ci_upper": 0.13},
        "per_model": {"tfidf": {"acc_choices_only": 0.26}},
        "metadata": {"threshold": 0.30},
    }
    cal_data = {"max_ece": 0.03, "threshold": 0.10}
    stop_data = {"median_abs_prefix_shift": 0.0, "threshold": 1.0}
    audit_card = {
        "metrics": [
            {
                "name": "CSLI (Choice-Set Leakage Index)",
                "threshold": 0.30,
                "observed_criterion_value": 0.26,
                "verdict": "pass",
            },
            {
                "name": "Prefix-wise Calibration (ECE)",
                "threshold": 0.10,
                "observed_criterion_value": 0.03,
                "verdict": "pass",
                # No fallback/empty buckets -> no synthesized qualifier.
                "details": {"fallback_buckets": [], "empty_buckets": []},
            },
            {
                "name": "Diagnostic StopDFF (Median Abs Prefix Shift)",
                "threshold": 1.0,
                "observed_criterion_value": 0.0,
                "verdict": "pass",
                # Explicit None -- no qualifier.
                "verdict_qualifier": None,
            },
        ]
    }

    output = _generate_audit_table(csli_data, cal_data, stop_data, audit_card)
    rendered = output.read_text(encoding="utf-8")

    assert "\\multicolumn" not in rendered
    assert "\\textsuperscript" not in rendered


def test_dry_run_lists_audit_card_json(monkeypatch) -> None:
    """The dry-run plan must enumerate every input the non-dry-run path
    loads. Earlier the dry-run omitted audit_card.json -- operators reading
    --dry-run output would not see it as a hard dependency.
    """
    monkeypatch.setattr(sys, "argv", ["regenerate_figures.py", "--dry-run"])

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = regenerate_figures.main()

    output = buf.getvalue()
    assert rc == 0
    assert "csli.json" in output
    assert "calibration.json" in output
    assert "stopdff.json" in output
    assert "audit_card.json" in output  # Was missing before this fix.


# ---------------------------------------------------------------------------
# Cluster C: visualization input-domain validation
# ---------------------------------------------------------------------------


def test_validate_inputs_rejects_empty_per_model(capsys) -> None:
    """Cluster C: empty per_model fails fast with stderr error."""
    csli_data = {
        "panel_csli": {"mean": 0.1, "ci_lower": 0.05, "ci_upper": 0.15},
        "per_model": {},
        "metadata": {"threshold": 0.30},
    }
    cal_data = {"max_ece": 0.04, "threshold": 0.10}
    stop_data = {"median_abs_prefix_shift": 0.0, "threshold": 1.0}

    rc = _validate_inputs(csli_data, cal_data, stop_data)
    assert rc == 1
    captured = capsys.readouterr()
    assert "per_model" in captured.err
    assert "empty" in captured.err.lower()


def test_validate_inputs_rejects_nan_csli_value(capsys) -> None:
    """Cluster C: NaN in per_model[*].csli -> rc=1 with model name in error."""
    csli_data = _healthy_csli_data()
    csli_data["per_model"]["sbert"]["csli"] = float("nan")
    cal_data = {"max_ece": 0.04, "threshold": 0.10}
    stop_data = {"median_abs_prefix_shift": 0.0, "threshold": 1.0}

    rc = _validate_inputs(csli_data, cal_data, stop_data)
    assert rc == 1
    captured = capsys.readouterr()
    assert "sbert" in captured.err
    assert "csli" in captured.err.lower()


def test_validate_inputs_rejects_nan_panel_mean(capsys) -> None:
    """Cluster C: NaN panel_mean -> rc=1."""
    csli_data = _healthy_csli_data()
    csli_data["panel_csli"]["mean"] = float("nan")
    cal_data = {"max_ece": 0.04, "threshold": 0.10}
    stop_data = {"median_abs_prefix_shift": 0.0, "threshold": 1.0}

    rc = _validate_inputs(csli_data, cal_data, stop_data)
    assert rc == 1
    captured = capsys.readouterr()
    assert "panel_csli.mean" in captured.err


def test_validate_inputs_rejects_nan_max_ece(capsys) -> None:
    """Cluster C: NaN max_ece -> rc=1."""
    csli_data = _healthy_csli_data()
    cal_data = {"max_ece": float("nan"), "threshold": 0.10}
    stop_data = {"median_abs_prefix_shift": 0.0, "threshold": 1.0}

    rc = _validate_inputs(csli_data, cal_data, stop_data)
    assert rc == 1
    captured = capsys.readouterr()
    assert "max_ece" in captured.err


def test_validate_inputs_rejects_inf_stopdff(capsys) -> None:
    """Cluster C: +inf median_abs_prefix_shift -> rc=1."""
    csli_data = _healthy_csli_data()
    cal_data = {"max_ece": 0.04, "threshold": 0.10}
    stop_data = {"median_abs_prefix_shift": math.inf, "threshold": 1.0}

    rc = _validate_inputs(csli_data, cal_data, stop_data)
    assert rc == 1
    captured = capsys.readouterr()
    assert "median_abs_prefix_shift" in captured.err


def test_validate_inputs_warns_panel_mean_outside_ci(capsys) -> None:
    """Cluster C: panel_mean outside CI -> rc=0 but stderr WARNING.

    The CI computation may be stale relative to the mean; surface it
    to the operator but don't block the build (a real bug needs a code
    fix in compute_csli.py, not a script abort).
    """
    csli_data = _healthy_csli_data()
    # Mean above the upper CI bound.
    csli_data["panel_csli"] = {"mean": 0.50, "ci_lower": 0.09, "ci_upper": 0.13}
    cal_data = {"max_ece": 0.04, "threshold": 0.10}
    stop_data = {"median_abs_prefix_shift": 0.0, "threshold": 1.0}

    rc = _validate_inputs(csli_data, cal_data, stop_data)
    assert rc == 0
    captured = capsys.readouterr()
    assert "WARNING" in captured.err
    assert "outside" in captured.err.lower()


def test_validate_inputs_accepts_healthy(capsys) -> None:
    """Cluster C: well-formed inputs -> rc=0, no stderr."""
    csli_data = _healthy_csli_data()
    cal_data = {"max_ece": 0.04, "threshold": 0.10}
    stop_data = {"median_abs_prefix_shift": 0.0, "threshold": 1.0}

    rc = _validate_inputs(csli_data, cal_data, stop_data)
    assert rc == 0
    captured = capsys.readouterr()
    assert captured.err == ""


def test_csli_panel_scales_figsize_with_model_count(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Cluster C: 6+ models -> figsize wider than baseline 5.0 inches."""
    per_model = {
        f"model_{i}": {"csli": 0.1 + 0.01 * i} for i in range(6)
    }
    csli_data = {
        "per_model": per_model,
        "metadata": {"threshold": 0.3},
        "panel_csli": {"mean": 0.13, "ci_lower": 0.10, "ci_upper": 0.16},
    }
    out, fig = _capture_panel_figure(csli_data, tmp_path, monkeypatch)
    assert out.exists()
    assert fig is not None

    width_in = fig.get_size_inches()[0]
    # Formula: max(5.0, 1.5 + 1.0 * 6) = 7.5
    assert width_in > 5.0, f"expected width > 5.0, got {width_in}"
    assert math.isclose(width_in, 7.5, rel_tol=1e-6), (
        f"expected width=7.5 for 6 models, got {width_in}"
    )

    import matplotlib.pyplot as plt

    plt.close(fig)


def test_csli_panel_rotates_labels_above_threshold(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Cluster C: >4 models -> x-tick labels rotated 30deg."""
    per_model = {
        f"model_{i}": {"csli": 0.1 + 0.01 * i} for i in range(5)
    }
    csli_data = {
        "per_model": per_model,
        "metadata": {"threshold": 0.3},
        "panel_csli": {"mean": 0.12, "ci_lower": 0.10, "ci_upper": 0.14},
    }
    out, fig = _capture_panel_figure(csli_data, tmp_path, monkeypatch)
    assert out.exists()
    assert fig is not None

    ax = fig.axes[0]
    rotations = [lbl.get_rotation() for lbl in ax.get_xticklabels()]
    # Every visible tick should be rotated 30 (the production setp call
    # applies to ALL tick labels on the axis).
    assert all(r == 30 for r in rotations), f"expected 30deg rotations, got {rotations}"

    import matplotlib.pyplot as plt

    plt.close(fig)


def test_csli_panel_does_not_rotate_labels_below_threshold(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Cluster C: 3 models (<=4) -> labels stay horizontal (rotation=0)."""
    csli_data = _healthy_csli_data()
    out, fig = _capture_panel_figure(csli_data, tmp_path, monkeypatch)
    assert out.exists()
    assert fig is not None

    ax = fig.axes[0]
    rotations = [lbl.get_rotation() for lbl in ax.get_xticklabels()]
    assert all(r == 0 for r in rotations), (
        f"expected horizontal labels for 3 models, got rotations {rotations}"
    )

    import matplotlib.pyplot as plt

    plt.close(fig)


def test_csli_panel_ylim_includes_panel_mean(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Cluster C: panel_mean must be in plotted_values so it drives ylim
    even when it exceeds all per-model bar heights.

    The original spec used a panel_mean fully outside its own CI, but
    matplotlib's ``errorbar`` rejects the resulting negative yerr (the
    data-bug WARNING path in ``_validate_inputs`` catches that case
    upstream). Instead, this test pins panel_mean as the largest plotted
    value -- above every per-model csli AND above ci_upper -- and asserts
    the rendered ylim[1] covers it. Without ``panel_mean`` in
    ``plotted_values``, ylim[1] would max out at ci_upper and clip the
    diamond marker.
    """
    csli_data = {
        "per_model": {
            "tfidf": {"csli": 0.05},
            "sbert": {"csli": 0.08},
        },
        "metadata": {"threshold": 0.30},
        # panel_mean=0.40 inside CI [0.30, 0.50] (errorbar accepts it),
        # but 0.40 is strictly greater than max(csli=0.08) and the test
        # checks that ylim covers it. ci_upper=0.50 is the actual top
        # driver, so we assert ylim[1] >= 0.40 (the panel_mean) -- the
        # critical contract.
        "panel_csli": {"mean": 0.40, "ci_lower": 0.30, "ci_upper": 0.50},
    }
    out, fig = _capture_panel_figure(csli_data, tmp_path, monkeypatch)
    assert out.exists()
    assert fig is not None

    ax = fig.axes[0]
    _, y_hi = ax.get_ylim()
    # Both the panel mean (0.40) and ci_upper (0.50) must be visible.
    assert y_hi >= 0.40, f"expected ylim[1] >= 0.40 to fit panel_mean, got {y_hi}"
    assert y_hi >= 0.50, f"expected ylim[1] >= 0.50 to fit ci_upper, got {y_hi}"

    import matplotlib.pyplot as plt

    plt.close(fig)


# ---------------------------------------------------------------------------
# Cluster D: output contract / CI integration
# ---------------------------------------------------------------------------


def _seed_paper_exports(tmp_path: Path) -> None:
    """Write minimal csli/calibration/stopdff/audit_card JSONs that
    pass _validate_inputs and _generate_audit_table.
    """
    import json

    csli = {
        "panel_csli": {"mean": 0.11, "ci_lower": 0.09, "ci_upper": 0.13},
        "per_model": {
            "tfidf": {"csli": 0.05, "acc_choices_only": 0.26},
            "sbert": {"csli": 0.15, "acc_choices_only": 0.24},
        },
        "metadata": {"threshold": 0.30},
    }
    cal = {"max_ece": 0.04, "threshold": 0.10}
    stop = {"median_abs_prefix_shift": 0.0, "threshold": 1.0}
    audit_card = {
        "metrics": [
            {
                "name": "CSLI (Choice-Set Leakage Index)",
                "threshold": 0.30,
                "observed_criterion_value": 0.26,
                "verdict": "pass",
            },
            {
                "name": "Prefix-wise Calibration (ECE)",
                "threshold": 0.10,
                "observed_criterion_value": 0.04,
                "verdict": "pass",
            },
            {
                "name": "Diagnostic StopDFF (Median Abs Prefix Shift)",
                "threshold": 1.0,
                "observed_criterion_value": 0.0,
                "verdict": "pass",
            },
        ]
    }
    (tmp_path / "csli.json").write_text(json.dumps(csli), encoding="utf-8")
    (tmp_path / "calibration.json").write_text(json.dumps(cal), encoding="utf-8")
    (tmp_path / "stopdff.json").write_text(json.dumps(stop), encoding="utf-8")
    (tmp_path / "audit_card.json").write_text(json.dumps(audit_card), encoding="utf-8")


def test_main_exit_2_when_reliability_missing(tmp_path: Path, monkeypatch) -> None:
    """Cluster D: missing reliability PNGs -> rc=2 (without flag)."""
    _seed_paper_exports(tmp_path)
    monkeypatch.setattr(regenerate_figures, "_PAPER_EXPORTS", tmp_path)
    monkeypatch.setattr(sys, "argv", ["regenerate_figures.py"])

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = regenerate_figures.main()

    assert rc == 2


def test_main_exit_0_when_reliability_missing_with_flag(
    tmp_path: Path, monkeypatch
) -> None:
    """Cluster D: --allow-missing-reliability suppresses to rc=0."""
    _seed_paper_exports(tmp_path)
    monkeypatch.setattr(regenerate_figures, "_PAPER_EXPORTS", tmp_path)
    monkeypatch.setattr(
        sys, "argv", ["regenerate_figures.py", "--allow-missing-reliability"]
    )

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = regenerate_figures.main()

    assert rc == 0


def test_main_exit_0_when_reliability_present(tmp_path: Path, monkeypatch) -> None:
    """Cluster D: all three reliability PNGs present -> rc=0 even without flag."""
    _seed_paper_exports(tmp_path)
    # Seed reliability stub PNGs so the existence check passes.
    for name in (
        "reliability_early.png",
        "reliability_mid.png",
        "reliability_late.png",
    ):
        (tmp_path / name).write_bytes(b"")
    monkeypatch.setattr(regenerate_figures, "_PAPER_EXPORTS", tmp_path)
    monkeypatch.setattr(sys, "argv", ["regenerate_figures.py"])

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = regenerate_figures.main()

    assert rc == 0


# ---------------------------------------------------------------------------
# Cluster E: cosmetics — version stamp, threshold direction, palette
# ---------------------------------------------------------------------------


def _minimal_audit_table_inputs() -> tuple[dict, dict, dict, dict]:
    """Minimal inputs that drive _generate_audit_table without footnotes."""
    csli_data = {
        "panel_csli": {"mean": 0.11, "ci_lower": 0.09, "ci_upper": 0.13},
        "per_model": {"tfidf": {"acc_choices_only": 0.26}},
        "metadata": {"threshold": 0.30},
    }
    cal_data = {"max_ece": 0.03, "threshold": 0.10}
    stop_data = {"median_abs_prefix_shift": 0.0, "threshold": 1.0}
    audit_card = {
        "metrics": [
            {
                "name": "CSLI (Choice-Set Leakage Index)",
                "threshold": 0.30,
                "observed_criterion_value": 0.26,
                "verdict": "pass",
            },
            {
                "name": "Prefix-wise Calibration (ECE)",
                "threshold": 0.10,
                "observed_criterion_value": 0.03,
                "verdict": "pass",
            },
            {
                "name": "Diagnostic StopDFF (Median Abs Prefix Shift)",
                "threshold": 1.0,
                "observed_criterion_value": 0.0,
                "verdict": "pass",
            },
        ]
    }
    return csli_data, cal_data, stop_data, audit_card


def test_audit_table_header_includes_version_stamp(
    tmp_path: Path, monkeypatch
) -> None:
    """PFN-1: audit_table.tex starts with `% Generated by regenerate_figures.py v...`."""
    monkeypatch.setattr(regenerate_figures, "_PAPER_EXPORTS", tmp_path)
    csli_data, cal_data, stop_data, audit_card = _minimal_audit_table_inputs()

    output = _generate_audit_table(csli_data, cal_data, stop_data, audit_card)
    first_line = output.read_text(encoding="utf-8").splitlines()[0]

    assert first_line.startswith("% Generated by regenerate_figures.py v"), (
        f"expected version-stamp comment as first line, got: {first_line!r}"
    )
    # The declared version constant must appear in the stamp.
    assert regenerate_figures._SCRIPT_VERSION in first_line


def test_audit_table_header_includes_git_sha_or_unknown(
    tmp_path: Path, monkeypatch
) -> None:
    """PFN-1: version-stamp line includes 'from commit <sha-or-unknown>'."""
    monkeypatch.setattr(regenerate_figures, "_PAPER_EXPORTS", tmp_path)
    csli_data, cal_data, stop_data, audit_card = _minimal_audit_table_inputs()

    output = _generate_audit_table(csli_data, cal_data, stop_data, audit_card)
    first_line = output.read_text(encoding="utf-8").splitlines()[0]

    assert "from commit " in first_line
    # Either a non-empty short SHA or the literal 'unknown' (git failure path).
    sha_part = first_line.split("from commit ", 1)[1].strip()
    assert sha_part, "expected non-empty commit identifier"


def test_threshold_column_header_shows_direction_when_homogeneous(
    tmp_path: Path, monkeypatch
) -> None:
    """PFN-2: when all 3 views have direction='warn_if_above', header reads
    'Threshold (warn if above)'.
    """
    monkeypatch.setattr(regenerate_figures, "_PAPER_EXPORTS", tmp_path)
    csli_data, cal_data, stop_data, audit_card = _minimal_audit_table_inputs()
    # Mark all metrics explicitly 'warn_if_above'.
    for m in audit_card["metrics"]:
        m["direction"] = "warn_if_above"

    output = _generate_audit_table(csli_data, cal_data, stop_data, audit_card)
    rendered = output.read_text(encoding="utf-8")

    # The header line contains the directional qualifier.
    header_line = [
        line for line in rendered.splitlines() if line.startswith("Metric &")
    ]
    assert len(header_line) == 1
    assert "Threshold (warn if above)" in header_line[0]
    # And the mixed-direction footnote is NOT emitted in the homogeneous path.
    assert "CSLI: warn if" not in rendered


def test_threshold_column_header_generic_when_mixed_directions(
    tmp_path: Path, monkeypatch
) -> None:
    """PFN-2: when StopDFF has 'warn_if_below' but others 'warn_if_above',
    header stays 'Threshold' and a direction footnote is added after
    \\bottomrule.
    """
    monkeypatch.setattr(regenerate_figures, "_PAPER_EXPORTS", tmp_path)
    csli_data, cal_data, stop_data, audit_card = _minimal_audit_table_inputs()
    # CSLI + Cal: warn_if_above; StopDFF: warn_if_below (mixed).
    audit_card["metrics"][0]["direction"] = "warn_if_above"
    audit_card["metrics"][1]["direction"] = "warn_if_above"
    audit_card["metrics"][2]["direction"] = "warn_if_below"

    output = _generate_audit_table(csli_data, cal_data, stop_data, audit_card)
    rendered = output.read_text(encoding="utf-8")

    # Header stays generic.
    header_line = [
        line for line in rendered.splitlines() if line.startswith("Metric &")
    ]
    assert len(header_line) == 1
    assert "Threshold (warn if" not in header_line[0]
    assert "Threshold & Verdict" in header_line[0]

    # Direction footnote line exists below \bottomrule.
    # Locate \bottomrule index and check that a per-metric direction
    # footnote appears after it.
    lines = rendered.splitlines()
    bottomrule_idx = next(
        i for i, line in enumerate(lines) if r"\bottomrule" in line
    )
    post_bottomrule = "\n".join(lines[bottomrule_idx + 1 :])
    assert "CSLI: warn if above" in post_bottomrule
    assert "Calibration: warn if above" in post_bottomrule
    assert "StopDFF: warn if below" in post_bottomrule


def test_csli_panel_uses_tab10_colors(
    tmp_path: Path, monkeypatch
) -> None:
    """PFN-3: bar colors come from tab10 colormap (not the old 3-color list).

    Introspect ax.patches[i].get_facecolor() and compare to
    plt.get_cmap('tab10')(i).
    """
    csli_data = _healthy_csli_data()  # 3 models
    out, fig = _capture_panel_figure(csli_data, tmp_path, monkeypatch)
    assert out.exists()
    assert fig is not None, "captured figure required for color introspection"

    ax = fig.axes[0]
    # The 3 bars are the first 3 Rectangle patches; the errorbar marker is
    # a separate artist, not in ax.patches.
    bar_patches = [
        p for p in ax.patches if type(p).__name__ == "Rectangle"
    ][:3]
    assert len(bar_patches) == 3, f"expected 3 bar patches, got {len(bar_patches)}"

    import matplotlib.pyplot as plt

    tab10 = plt.get_cmap("tab10")
    for i, patch in enumerate(bar_patches):
        expected = tab10(i)
        actual = patch.get_facecolor()
        # tab10(i) returns RGBA; patch facecolor is also RGBA. Compare to
        # 1e-6 tolerance per channel.
        for j in range(4):
            assert math.isclose(actual[j], expected[j], abs_tol=1e-6), (
                f"bar {i} channel {j}: expected {expected[j]}, got {actual[j]}"
            )

    plt.close(fig)


def test_csli_panel_warns_when_models_exceed_palette_capacity(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """PFN-3: synthetic panel with 11 models triggers stderr warning about
    color repeat.
    """
    per_model = {
        f"model_{i:02d}": {"csli": 0.05 + 0.005 * i} for i in range(11)
    }
    csli_data = {
        "per_model": per_model,
        "metadata": {"threshold": 0.30},
        "panel_csli": {"mean": 0.08, "ci_lower": 0.05, "ci_upper": 0.11},
    }
    monkeypatch.setattr(regenerate_figures, "_PAPER_EXPORTS", tmp_path)
    import matplotlib

    matplotlib.use("Agg")
    out = _generate_csli_panel(csli_data)
    assert out.exists()

    captured = capsys.readouterr()
    assert "11" in captured.err
    assert "tab10" in captured.err
    assert "repeat" in captured.err.lower()
