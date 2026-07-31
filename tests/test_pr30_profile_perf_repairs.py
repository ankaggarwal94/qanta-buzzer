"""Focused regressions for the PR #30 profile and performance repairs."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.stopdff_v5 import fvi_study, profile, sweep


REPO = Path(__file__).resolve().parents[1]


def _canonical_cell() -> dict[str, str]:
    return {
        "reward_schedule": "acf_flat",
        "continuation": "empirical_bucket",
        "calibrator": "platt-logistic",
        "prefix_bucketing": "early_mid_late",
        "category_pooling": "per_category",
    }


def test_normalize_cell_rejects_unknown_and_alias_collision() -> None:
    unknown = {**_canonical_cell(), "category_poolng": "per_category"}
    with pytest.raises(ValueError, match="unknown axes"):
        profile.normalize_cell(unknown)

    collision = {
        **_canonical_cell(),
        "subject_pooling": "per_subject",
    }
    with pytest.raises(ValueError, match="both alias"):
        profile.normalize_cell(collision)


def test_normalize_cell_still_accepts_one_legacy_alias() -> None:
    legacy = _canonical_cell()
    legacy.pop("category_pooling")
    legacy["subject_pooling"] = "per_subject"
    assert profile.normalize_cell(legacy)["category_pooling"] == "per_category"


def test_scientific_profile_template_is_the_canonical_static_identity() -> None:
    template = json.loads(
        (REPO / "SCIENTIFIC_PROFILE.template.json").read_text(encoding="utf-8")
    )
    schema = json.loads(
        (
            REPO / "schemas" / "stopdff_scientific_profile.schema.json"
        ).read_text(encoding="utf-8")
    )
    assert template == profile.profile_static_identity()
    assert set(template) == set(schema["required"])


def test_fvi_study_reuses_strict_representative_result(monkeypatch) -> None:
    representative = [{"scope": "representative"}]
    full = [{"scope": "full"}]
    calls: list[tuple[str, int, str]] = []

    monkeypatch.setattr(fvi_study, "representative_24", lambda: representative)
    monkeypatch.setattr(fvi_study, "full_grid", lambda: full)
    monkeypatch.setattr(fvi_study, "FVI_TOLERANCES", ("1e-6", "1e-10"))
    monkeypatch.setattr(fvi_study, "FVI_MAX_ITERATIONS", (50, 200))

    def fake_run(
        *,
        rows,
        cells,
        calibration_json,
        tolerance_label,
        max_iterations,
    ):
        del rows, calibration_json
        scope = "representative" if cells is representative else "full"
        calls.append((tolerance_label, max_iterations, scope))
        return {
            "tolerance": tolerance_label,
            "max_iterations": max_iterations,
            "total_iterations": (
                1
                if (tolerance_label, max_iterations) == ("1e-6", 50)
                else 100
            ),
            "all_converged": True,
            "cells": {},
        }

    monkeypatch.setattr(fvi_study, "run_candidate_on_cells", fake_run)
    result = fvi_study.run_fvi_study(rows=[], calibration_json=None)

    assert result["selected_parameters"] == {
        "tolerance": "1e-6",
        "max_iterations": 50,
    }
    assert calls.count(("1e-10", 200, "representative")) == 1


def test_fvi_selector_falls_through_failed_full_validation(monkeypatch) -> None:
    representative = [{"scope": "representative"}]
    full = [{"scope": "full"}]
    full_calls = 0

    monkeypatch.setattr(fvi_study, "representative_24", lambda: representative)
    monkeypatch.setattr(fvi_study, "full_grid", lambda: full)
    monkeypatch.setattr(fvi_study, "FVI_TOLERANCES", ("1e-6", "1e-8"))
    monkeypatch.setattr(fvi_study, "FVI_MAX_ITERATIONS", (50,))
    monkeypatch.setattr(
        fvi_study,
        "FVI_STRICT_REFERENCE",
        {"tolerance": "1e-10", "max_iterations": 200},
    )

    def fake_run(
        *,
        rows,
        cells,
        calibration_json,
        tolerance_label,
        max_iterations,
    ):
        nonlocal full_calls
        del rows, calibration_json
        if cells is full:
            full_calls += 1
            converged = full_calls > 1
        else:
            converged = True
        return {
            "tolerance": tolerance_label,
            "max_iterations": max_iterations,
            "total_iterations": 1 if tolerance_label == "1e-6" else 2,
            "all_converged": converged,
            "cells": {},
        }

    monkeypatch.setattr(fvi_study, "run_candidate_on_cells", fake_run)
    result = fvi_study.run_fvi_study(rows=[], calibration_json=None)
    assert full_calls == 2
    assert result["selected_parameters"] == {
        "tolerance": "1e-8",
        "max_iterations": 50,
    }


def test_resume_preflight_recomputes_only_cached_cells(
    monkeypatch, tmp_path
) -> None:
    cells = [{"key": "cached"}, {"key": "missing"}]
    cells_dir = tmp_path / "cells"
    cells_dir.mkdir()
    (tmp_path / "attempts.jsonl").write_text(
        json.dumps({"attempt": 1}) + "\n",
        encoding="utf-8",
    )
    expected = {"cell_key": "cached"}
    (cells_dir / "cached.json").write_text(
        json.dumps(expected, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    calls: list[str] = []

    monkeypatch.setattr(sweep, "cell_key_str", lambda cell: cell["key"])

    def fake_cell_record(ctx, cell):
        del ctx
        calls.append(cell["key"])
        return {"cell_key": cell["key"]}

    monkeypatch.setattr(sweep, "_cell_record", fake_cell_record)
    ctx = SimpleNamespace(
        output_dir=tmp_path,
        run_spec_id="a" * 64,
    )
    records, interrupted = sweep._resume_preflight(
        ctx,
        cells=cells,
        spec_ids={},
        bootstrap_plan_id="b" * 64,
    )
    assert calls == ["cached"]
    assert records == {"cached": expected}
    assert interrupted == {
        "attempt": 1,
        "state": "interrupted",
        "run_spec_id": "a" * 64,
        "reason": "terminal_result_missing_at_resume",
    }
