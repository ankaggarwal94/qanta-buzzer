"""Invocation-local reuse regressions for StopDFF cell computation."""
from __future__ import annotations

from scripts.stopdff_v5 import cellcompute
from scripts.stopdff_v5.profile import full_grid
from tests.harness_control_plane import _calibration_json, _synth_rows


def test_cell_family_reuses_calibrator_and_row_preparation(monkeypatch):
    rows = _synth_rows(n_items=20)
    calibration = _calibration_json()
    prepared = cellcompute.prepare_cell_inputs(rows, calibration)
    original = cellcompute.fit_calibrator
    fitted: list[str] = []

    def counted_fit(name, **kwargs):
        fitted.append(name)
        return original(name, **kwargs)

    monkeypatch.setattr(cellcompute, "fit_calibrator", counted_fit)
    cells = [
        cell
        for cell in full_grid()
        if cell["calibrator"] == "platt-logistic"
    ][:2]
    for cell in cells:
        cellcompute.compute_cell(
            rows=rows,
            cell=cell,
            calibration_json=calibration,
            tolerance=1e-6,
            max_iterations=50,
            tolerance_label="1e-6",
            metric_split="test",
            prepared=prepared,
        )

    assert fitted == ["platt-logistic"]
    assert set(prepared.trajectories_by_split) == {"val", "test"}
    assert set(prepared.calibrated_trajectories) == {
        ("platt-logistic", "val"),
        ("platt-logistic", "test"),
    }
