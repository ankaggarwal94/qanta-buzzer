"""Focused regressions for Round 2 package/provenance closure."""
from __future__ import annotations

import builtins
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.stopdff_v5 import writers  # noqa: E402


def _bindings() -> dict[str, str]:
    return {
        "source_manifest_id": "1" * 64,
        "raw_input_bundle_id": "2" * 64,
        "model_snapshot_id": "3" * 64,
        "adapter_bundle_id": "4" * 64,
        "fvi_study_id": "5" * 64,
        "environment_contract_id": "6" * 64,
    }


@pytest.mark.parametrize(
    ("gate", "evidence"),
    [
        ("smoke", {"evidence_sha256": "a" * 64}),
        ("mutation", {"evidence_sha256": "b" * 64}),
        ("determinism", {"evidence_sha256": "c" * 64}),
    ],
)
def test_receipts_accept_only_emitted_evidence_shapes(gate, evidence):
    bindings = _bindings()
    if gate == "determinism":
        bindings = {
            key: value
            for key, value in bindings.items()
            if key not in {"fvi_study_id", "environment_contract_id"}
        }
    receipt = writers.build_prerequisite_receipt(
        gate=gate,
        bindings=bindings,
        evidence=evidence,
    )
    assert receipt["identity"]["evidence"] == evidence


@pytest.mark.parametrize(
    ("gate", "evidence"),
    [
        ("smoke", {"evidence_sha256": "a" * 64, "passed": False}),
        ("mutation", {"fixture": "mutation"}),
        ("determinism", {"evidence_sha256": "NOT-A-HASH"}),
    ],
)
def test_receipts_reject_arbitrary_or_contradictory_evidence(gate, evidence):
    bindings = _bindings()
    if gate == "determinism":
        bindings = {
            key: value
            for key, value in bindings.items()
            if key not in {"fvi_study_id", "environment_contract_id"}
        }
    with pytest.raises(ValueError, match="evidence|SHA-256"):
        writers.build_prerequisite_receipt(
            gate=gate,
            bindings=bindings,
            evidence=evidence,
        )


def test_figure_renderer_is_deterministic_without_matplotlib(
    tmp_path,
    monkeypatch,
):
    real_import = builtins.__import__

    def fail_matplotlib(name, *args, **kwargs):
        if name == "matplotlib" or name.startswith("matplotlib."):
            raise ImportError("matplotlib unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_matplotlib)
    final_written = writers.write_figures(
        tmp_path / "final",
        {"cells": {}},
        profile_variant="final",
    )
    smoke_written = writers.write_figures(
        tmp_path / "smoke",
        {"cells": {}},
        profile_variant="smoke",
    )
    assert final_written == smoke_written == [
        "figures/cell_median_index_shift.png"
    ]
    assert (
        tmp_path / "final" / final_written[0]
    ).read_bytes() == (tmp_path / "smoke" / smoke_written[0]).read_bytes()
