"""Tests for the exploratory metric-row opt-out in make_audit_card.

Pins:
- Every ``_evaluate_*`` helper now sets ``exploratory=False`` by
  default so existing CSLI / calibration / StopDFF / DP rows keep
  feeding the overall-verdict ladder.
- ``_compute_overall_verdict`` filters rows whose ``exploratory`` flag
  is True before applying the ``fail > warn > pass`` ladder, with a
  documented fallback to the full list when every row is exploratory.
- The MD renderer surfaces an ``[exploratory; not in overall verdict]``
  suffix on exploratory verdict cells (and only those).
- A dedicated ``## Prior-WARN Resolution`` MD subsection renders when
  any metric carries ``prior_warn_resolution`` and is omitted entirely
  otherwise.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts import make_audit_card


def _confirmatory(verdict: str, name: str = "metric") -> dict:
    """Minimal confirmatory metric dict (exploratory defaults to absent)."""
    return {
        "name": name,
        "value": 0.0,
        "value_display": "0.0000",
        "threshold": 1.0,
        "verdict": verdict,
    }


def _exploratory(verdict: str, name: str = "exploratory metric") -> dict:
    """Minimal exploratory metric dict."""
    return {
        "name": name,
        "value": 0.0,
        "value_display": "0.0000",
        "threshold": 1.0,
        "verdict": verdict,
        "exploratory": True,
    }


def test_overall_verdict_ignores_exploratory_warn() -> None:
    """3 confirmatory PASS + 1 exploratory WARN must yield PASS."""
    metrics = [
        _confirmatory("pass", "csli"),
        _confirmatory("pass", "calibration"),
        _confirmatory("pass", "stopdff"),
        _exploratory("warn", "learned_value_stopdff"),
    ]
    overall, qualifier = make_audit_card._compute_overall_verdict(metrics, None)
    assert overall == "PASS"
    assert qualifier is None


def test_overall_verdict_ignores_exploratory_fail() -> None:
    """3 confirmatory PASS + 1 exploratory FAIL must still yield PASS."""
    metrics = [
        _confirmatory("pass", "csli"),
        _confirmatory("pass", "calibration"),
        _confirmatory("pass", "stopdff"),
        _exploratory("fail", "learned_value_stopdff"),
    ]
    overall, qualifier = make_audit_card._compute_overall_verdict(metrics, None)
    assert overall == "PASS"
    assert qualifier is None


def test_overall_verdict_falls_back_when_all_exploratory() -> None:
    """0 confirmatory + 2 exploratory must compute against the exploratory list.

    The fallback exists so a pure-exploratory card still surfaces a
    verdict instead of silently returning PASS-by-vacuous-truth.
    """
    metrics = [
        _exploratory("warn", "exp1"),
        _exploratory("pass", "exp2"),
    ]
    overall, qualifier = make_audit_card._compute_overall_verdict(metrics, None)
    assert overall == "WARN"
    assert qualifier is None


def test_md_renders_exploratory_qualifier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exploratory rows get a ``[exploratory; not in overall verdict]`` suffix;
    confirmatory rows do not."""
    monkeypatch.setattr(make_audit_card, "_PAPER_EXPORTS", tmp_path)
    metrics = [
        {
            "name": "Confirmatory CSLI",
            "value": 0.0035,
            "value_display": "0.0035",
            "threshold": 0.3,
            "verdict": "pass",
        },
        {
            "name": "Learned-Value StopDFF",
            "value": 0.1,
            "value_display": "+0.1000",
            "threshold": 1,
            "verdict": "warn",
            "exploratory": True,
        },
    ]
    out_path = make_audit_card._write_audit_card_md(metrics, "PASS", None)
    md = out_path.read_text(encoding="utf-8")

    suffix = "[exploratory; not in overall verdict]"
    confirmatory_line = next(
        line for line in md.splitlines() if "Confirmatory CSLI" in line
    )
    exploratory_line = next(
        line for line in md.splitlines() if "Learned-Value StopDFF" in line
    )
    assert suffix not in confirmatory_line
    assert suffix in exploratory_line
    # Exactly one space before the bracketed annotation.
    assert "WARN " + suffix in exploratory_line


def test_prior_warn_resolution_subsection_renders(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When a metric carries ``prior_warn_resolution`` + notes, the MD
    must emit a ``## Prior-WARN Resolution`` section listing it."""
    monkeypatch.setattr(make_audit_card, "_PAPER_EXPORTS", tmp_path)
    metrics = [
        {
            "name": "DP StopDFF",
            "value": -0.5,
            "value_display": "-0.5000",
            "threshold": 1,
            "verdict": "pass",
            "prior_warn_resolution": "reduces",
            "prior_warn_resolution_notes": "DP shows partial fix",
        },
        {
            "name": "Confirmatory CSLI",
            "value": 0.003,
            "value_display": "0.0030",
            "threshold": 0.3,
            "verdict": "pass",
        },
    ]
    out_path = make_audit_card._write_audit_card_md(metrics, "PASS", None)
    md = out_path.read_text(encoding="utf-8")
    assert "## Prior-WARN Resolution" in md
    assert "DP StopDFF" in md
    assert "reduces" in md
    assert "DP shows partial fix" in md
    # CSLI row has no resolution claim and must not appear in the list.
    resolution_section = md.split("## Prior-WARN Resolution", 1)[1]
    next_heading_idx = resolution_section.find("\n## ")
    if next_heading_idx >= 0:
        resolution_section = resolution_section[:next_heading_idx]
    assert "Confirmatory CSLI" not in resolution_section


def test_prior_warn_resolution_subsection_absent_when_no_metric_has_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without any ``prior_warn_resolution`` set, the heading must not appear."""
    monkeypatch.setattr(make_audit_card, "_PAPER_EXPORTS", tmp_path)
    metrics = [
        {
            "name": "CSLI",
            "value": 0.003,
            "value_display": "0.0030",
            "threshold": 0.3,
            "verdict": "pass",
        },
        {
            "name": "Calibration",
            "value": 0.026,
            "value_display": "0.0260",
            "threshold": 0.1,
            "verdict": "pass",
        },
    ]
    out_path = make_audit_card._write_audit_card_md(metrics, "PASS", None)
    md = out_path.read_text(encoding="utf-8")
    assert "## Prior-WARN Resolution" not in md
