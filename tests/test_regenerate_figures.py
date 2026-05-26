"""Tests for cached figure/table regeneration."""

from __future__ import annotations

import sys
from pathlib import Path

sys.modules.pop("scripts.regenerate_figures", None)

from scripts import regenerate_figures

_generate_audit_table = getattr(regenerate_figures, "_generate_audit_table")
_generate_csli_panel = getattr(regenerate_figures, "_generate_csli_panel")


def test_audit_table_uses_thresholds_from_audit_card(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(regenerate_figures, "_PAPER_EXPORTS", tmp_path)
    csli_data = {
        "panel_csli": {"mean": 0.2, "ci_lower": 0.1, "ci_upper": 0.3},
        "metadata": {"threshold": 0.30},
    }
    cal_data = {"max_ece": 0.04, "threshold": 0.10}
    stop_data = {"median_abs_prefix_shift": 0.0, "threshold": 1.0}
    audit_card = {
        "metrics": [
            {"name": "CSLI (Choice-Set Leakage Index)", "threshold": 0.42, "verdict": "warn"},
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
