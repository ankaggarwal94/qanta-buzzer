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
- ``--include-learned-value-stopdff`` is a no-op-on-absent-artifact
  flag (mirrors ``--include-dp-stopdff``); when the artifact exists,
  the appended row carries ``exploratory=True`` and never affects the
  overall verdict (integration test against Commit 1's filter).
"""

from __future__ import annotations

import json
import shutil
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


# ---------------------------------------------------------------------------
# Commit 2: --include-learned-value-stopdff flag integration tests
# ---------------------------------------------------------------------------


def _seed_paper_exports_minimum(paper_dir: Path) -> None:
    """Copy the three existing artifacts so make_audit_card can load them."""
    src = Path(__file__).resolve().parent.parent / "paper_exports"
    for fname in ("csli.json", "calibration.json", "stopdff.json"):
        shutil.copyfile(src / fname, paper_dir / fname)


def _minimal_learned_value_payload(*, gate_verdict: str = "warn") -> dict:
    """Synthesize a stopdff_learned_value.json payload for tests."""
    return {
        "stopdff_signed_median": 0.5,
        "n_items": 50,
        "gate_verdict": gate_verdict,
        "gate_verdict_reason": (
            "synthetic_test_payload"
            if gate_verdict != "pass"
            else None
        ),
        "metadata": {
            "metric_type": "learned_value_dp",
            "checkpoint_path": "artifacts/value_model/seed1/best.ckpt",
            "seeds": [1, 2, 3],
        },
    }


def test_learned_value_flag_skips_gracefully_when_artifact_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """Missing stopdff_learned_value.json must exit 0 with a stderr warning."""
    paper = tmp_path / "paper_exports"
    paper.mkdir()
    _seed_paper_exports_minimum(paper)
    monkeypatch.setattr(make_audit_card, "_PAPER_EXPORTS", paper)

    rc = make_audit_card.main_with_args(["--include-learned-value-stopdff"])

    assert rc == 0
    captured = capsys.readouterr()
    assert "--include-learned-value-stopdff" in captured.err
    assert "stopdff_learned_value.json" in captured.err
    assert "skipped" in captured.err.lower()


def test_learned_value_flag_appends_exploratory_row_when_artifact_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real stopdff_learned_value.json must surface an exploratory row."""
    paper = tmp_path / "paper_exports"
    paper.mkdir()
    _seed_paper_exports_minimum(paper)
    (paper / "stopdff_learned_value.json").write_text(
        json.dumps(_minimal_learned_value_payload(gate_verdict="pass")),
        encoding="utf-8",
    )
    monkeypatch.setattr(make_audit_card, "_PAPER_EXPORTS", paper)

    rc = make_audit_card.main_with_args(["--include-learned-value-stopdff"])
    assert rc == 0

    card = json.loads((paper / "audit_card.json").read_text(encoding="utf-8"))
    lv_rows = [
        m for m in card["metrics"]
        if m["name"] == "Learned-Value StopDFF (Exploratory)"
    ]
    assert len(lv_rows) == 1
    assert lv_rows[0]["exploratory"] is True


def test_learned_value_row_does_not_affect_overall_verdict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Integration check: a learned-value FAIL must not flip the headline.

    Depends on Commit 1's filter in _compute_overall_verdict — without
    it, an exploratory FAIL would cascade into a headline WARN/FAIL.
    """
    paper = tmp_path / "paper_exports"
    paper.mkdir()
    _seed_paper_exports_minimum(paper)
    (paper / "stopdff_learned_value.json").write_text(
        json.dumps(_minimal_learned_value_payload(gate_verdict="fail")),
        encoding="utf-8",
    )
    monkeypatch.setattr(make_audit_card, "_PAPER_EXPORTS", paper)

    # Baseline (no flag): record what the headline verdict is on the
    # repo's committed artifacts so we can compare against the flagged
    # run on the same artifacts.
    rc_base = make_audit_card.main_with_args([])
    assert rc_base == 0
    baseline = json.loads((paper / "audit_card.json").read_text(encoding="utf-8"))

    # Flagged run on the same paper_exports dir.
    rc_flag = make_audit_card.main_with_args(["--include-learned-value-stopdff"])
    assert rc_flag == 0
    flagged = json.loads((paper / "audit_card.json").read_text(encoding="utf-8"))

    # Headline verdict must be identical despite the appended exploratory FAIL.
    assert flagged["overall_verdict"] == baseline["overall_verdict"]
    # And the exploratory row must still be present + carry verdict=fail
    # (sanity-check that the assertion above isn't trivially satisfied by
    # the row being missing).
    lv_rows = [
        m for m in flagged["metrics"]
        if m["name"] == "Learned-Value StopDFF (Exploratory)"
    ]
    assert len(lv_rows) == 1
    assert lv_rows[0]["verdict"] == "fail"
    assert lv_rows[0]["exploratory"] is True
