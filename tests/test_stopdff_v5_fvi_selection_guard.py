"""Regressions for fail-closed FVI study publication on PR #30."""
from __future__ import annotations

import json
import types

import pytest

from scripts.stopdff_v5 import checker, fvi_study as fvi_study_module, identity
from scripts.stopdff_v5.manifests import fvi_study_identity
from tests.test_stopdff_v5_control_plane import _load_modal_runner


def _identity_kwargs() -> dict:
    return {
        "adapter_bundle_id": "a" * 64,
        "candidate_grid": {
            "tolerance": ["1e-6"],
            "max_iterations": [50],
        },
        "representative_generator": "representative_24_parity",
        "candidate_results": [],
        "strict_reference_results": {
            "tolerance": "1e-10",
            "max_iterations": 200,
            "all_converged": True,
            "total_iterations": 1,
        },
        "selector_rule": (
            "min_total_iterations__then_larger_tolerance__then_smaller_max_iter"
        ),
        "producer_hashes": {"fvi_study.py": "b" * 64},
    }


def _study_without_selection() -> dict:
    return {
        "candidate_grid": {
            "tolerance": ["1e-6"],
            "max_iterations": [50],
        },
        "representative_cell_generator": "representative_24_parity",
        "candidate_convergence_results": [],
        "strict_reference": {
            "tolerance": "1e-10",
            "max_iterations": 200,
            "all_converged": True,
            "total_iterations": 1,
        },
        "selector_rule": (
            "min_total_iterations__then_larger_tolerance__then_smaller_max_iter"
        ),
        "selected_parameters": None,
        "all96_fit_only_validation": None,
    }


def test_fvi_identity_rejects_unpublishable_selection_state() -> None:
    kwargs = _identity_kwargs()
    with pytest.raises(ValueError, match="no eligible candidate"):
        fvi_study_identity(
            **kwargs,
            selected_parameters=None,
            all96_validation=None,
        )

    with pytest.raises(ValueError, match="lacks all-96 validation"):
        fvi_study_identity(
            **kwargs,
            selected_parameters={
                "tolerance": "1e-6",
                "max_iterations": 50,
            },
            all96_validation=None,
        )


def test_modal_fvi_study_does_not_cache_failed_selection(
    tmp_path,
    monkeypatch,
) -> None:
    runner = _load_modal_runner(monkeypatch)
    runner.MNT = str(tmp_path)
    adapter_id = "a" * 64
    adapter_dir = tmp_path / "adapters" / f"canonical_{adapter_id}"
    adapter_dir.mkdir(parents=True)
    (adapter_dir / "calibration.json").write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(
        checker,
        "validate_adapter",
        lambda _path: types.SimpleNamespace(
            passed=True,
            errors=[],
            recomputed={"adapter_bundle_id": adapter_id},
        ),
    )
    monkeypatch.setattr(checker, "load_adapter_rows", lambda _path: [])
    monkeypatch.setattr(
        fvi_study_module,
        "run_fvi_study",
        lambda **_kwargs: _study_without_selection(),
    )
    monkeypatch.setattr(identity, "sha256_file", lambda _path: "c" * 64)

    with pytest.raises(ValueError, match="no eligible candidate"):
        runner.fvi_study(adapter_id)

    assert not (tmp_path / "fvi").exists()
    assert list(tmp_path.rglob("fvi_study.json")) == []
    assert list(tmp_path.rglob("fvi_study_execution.json")) == []
