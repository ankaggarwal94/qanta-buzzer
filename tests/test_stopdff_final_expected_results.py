"""Regression controls for the final-run numerical reducer (no model inference)."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys

import pytest


VERIFIER_PATH = (
    Path(__file__).resolve().parents[1]
    / "reproducibility/stopdff_final_modal_5d5328102912/verify_expected_results.py"
)


@pytest.fixture
def synthetic_package(tmp_path, monkeypatch):
    """Exercise the reducer with 96 cells, including absent zero categories.

    These are reducer inputs, not a scientifically valid inference package.
    The representative vectors reproduce its fixed descriptive expectations;
    other vectors and family replicates are deliberately simple synthetic data.
    """
    spec = importlib.util.spec_from_file_location("final_expected_results", VERIFIER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    expected = copy.deepcopy(module.EXPECTED)
    shifts = [-1] * 873 + [1] * 78 + [0] * 2086
    signed_mean = -795 / 3037
    absolute_mean = 951 / 3037
    expected["zero_fraction"] = {"min": 2086 / 3037, "median": 1.0, "max": 1.0}
    expected["signed_mean"] = {"min": signed_mean, "max": 0.0}
    expected["absolute_mean"] = {"min": 0.0, "max": absolute_mean}
    expected["family_histogram"] = {0.0: 1000}
    monkeypatch.setattr(module, "EXPECTED", expected)
    monkeypatch.setattr(sys, "argv", [str(VERIFIER_PATH), str(tmp_path)])

    (tmp_path / "cells").mkdir()
    intervals = {"negative": [-2.0, -1.0], "cross": [-1.0, 1.0], "positive": [1.0, 2.0]}
    paths = []
    for calibrator, counts in expected["calibrator_interval_counts"].items():
        for category, count in counts.items():
            for _ in range(count):
                index = len(paths)
                representative = index == 0
                values = shifts if representative else [0] * 3037
                cell = {
                    "status": "completed",
                    "cell_key": expected["representative_key"] if representative else f"cell-{index}",
                    "cell": {"calibrator": calibrator},
                    "index_shift_by_item": {str(i): value for i, value in enumerate(values)},
                    "bootstrap": {
                        "point": {
                            "signed_index_mean": signed_mean if representative else 0.0,
                            "absolute_index_mean": absolute_mean if representative else 0.0,
                            "signed_index_median": 0.0,
                            "absolute_index_median": 0.0,
                        },
                        "ci": {
                            "signed_index_mean": (
                                [-0.2831741851, -0.2383766875]
                                if representative else intervals[category]
                            ),
                            "absolute_index_mean": [0.2904181758, 0.3358742180],
                            "signed_index_median": [0.0, 0.0],
                            "absolute_index_median": [0.0, 0.0],
                        },
                        "abs_median_replicates": [0.0] * 1000,
                    },
                    "descriptive": {
                        "n_paired_items": 3037,
                        "never_buzz_mc": 178,
                        "never_buzz_qa": 723,
                    },
                }
                path = tmp_path / "cells" / f"{index:02d}.json"
                path.write_text(json.dumps(cell), encoding="utf-8")
                paths.append(path)
    aggregate = {
        "requested": 96,
        "completed": 96,
        "failed": 0,
        "skipped": 0,
        "release_status": "VALID",
        "family": {"M": 0.0, "ci": [0.0, 0.0]},
    }
    (tmp_path / "aggregate.json").write_text(json.dumps(aggregate), encoding="utf-8")
    return module, tmp_path, paths


def test_absent_zero_count_categories_pass(synthetic_package, capsys):
    module, _, _ = synthetic_package
    assert module.main() == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == "PASS"
    assert summary["errors"] == []


def test_changed_nonzero_interval_count_fails(synthetic_package, capsys):
    module, _, paths = synthetic_package
    cell = json.loads(paths[1].read_text())
    cell["bootstrap"]["ci"]["signed_index_mean"] = [1.0, 2.0]
    paths[1].write_text(json.dumps(cell))
    assert module.main() == 1
    errors = json.loads(capsys.readouterr().out)["errors"]
    assert any("isotonic interval counts" in error for error in errors)


def test_unexpected_nonzero_interval_category_fails(synthetic_package, capsys, monkeypatch):
    module, _, _ = synthetic_package
    original_classify = module.classify_interval
    monkeypatch.setattr(
        module,
        "classify_interval",
        lambda interval: "unexpected" if interval == [-2.0, -1.0] else original_classify(interval),
    )
    assert module.main() == 1
    errors = json.loads(capsys.readouterr().out)["errors"]
    assert any("isotonic interval counts" in error and "unexpected" in error for error in errors)


def test_wrong_family_statistic_fails(synthetic_package, capsys):
    module, root, _ = synthetic_package
    path = root / "aggregate.json"
    aggregate = json.loads(path.read_text())
    aggregate["family"]["M"] = 1.0
    path.write_text(json.dumps(aggregate))
    assert module.main() == 1
    assert "family M: 1.0 != 0" in json.loads(capsys.readouterr().out)["errors"]
