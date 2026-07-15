"""Integration test: full v5 scientific pipeline on synthetic adapter rows."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.stopdff_v5 import bootstrap, profile, sweep  # noqa: E402
from scripts.stopdff_v5.fvi_study import order_eligible  # noqa: E402

CATEGORIES = ["history", "science", "arts"]
PREFIX_FRACS = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]


def _synth_rows(n_items: int = 40, seed: int = 7) -> list[dict]:
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for i in range(n_items):
        qid = f"q{i:03d}"
        split = "val" if i < n_items // 2 else "test"
        cat = CATEGORIES[i % len(CATEGORIES)]
        item_off = rng.uniform(-0.15, 0.15)
        for t, frac in enumerate(PREFIX_FRACS):
            mc_sim = float(np.clip(0.25 + 0.55 * frac + item_off + rng.uniform(-0.05, 0.05), 0.0, 1.0))
            qa_sim = float(np.clip(0.20 + 0.60 * frac + item_off + rng.uniform(-0.05, 0.05), 0.0, 1.0))
            mc_correct = int(mc_sim + rng.uniform(-0.15, 0.15) > 0.55)
            rows.append({
                "item_id": qid, "prefix_idx": t, "prefix_fraction": frac, "format": "MC",
                "split": split, "raw_similarity": mc_sim, "correct": mc_correct, "category": cat,
            })
            rows.append({
                "item_id": qid, "prefix_idx": t, "prefix_fraction": frac, "format": "QA",
                "split": split, "raw_similarity": qa_sim, "correct": 1, "category": cat,
            })
    return rows


def _calibration_json() -> dict:
    block = {"platt_coef": 5.0, "platt_intercept": -2.5}
    return {"per_bucket": {"early": dict(block), "mid": dict(block), "late": dict(block)}}


def _test_item_ids(rows: list[dict]) -> list[str]:
    mc = {r["item_id"] for r in rows if r["split"] == "test" and r["format"] == "MC"}
    qa = {r["item_id"] for r in rows if r["split"] == "test" and r["format"] == "QA"}
    return sorted(mc & qa)


def _make_ctx(tmp_path: Path, rows, cells, replicates=100) -> sweep.SweepContext:
    plan = bootstrap.build_bootstrap_plan(_test_item_ids(rows), replicates=replicates, seed=1)
    return sweep.SweepContext(
        rows=rows, calibration_json=_calibration_json(),
        run_spec={"kind": "run_spec"}, run_spec_id="deadbeef" * 8,
        bootstrap_plan=plan, output_dir=tmp_path / "run",
        fvi_tolerance="1e-8", fvi_max_iterations=100, backend="modal",
        profile_variant="smoke", adapter_fit_rows_sha256="a" * 64,
        adapter_eval_rows_sha256="b" * 64, myopic_artifact_sha256="c" * 64,
        producer_hashes={"sweep.py": "d" * 64}, cells=cells,
        environment={"python": "3.11"}, resource_summary={"backend": "modal"},
        attempt={"attempt": 1, "mode": "fresh"},
    )


def test_full_pipeline_smoke_cells(tmp_path):
    rows = _synth_rows()
    cells = profile.smoke_cells()
    ctx = _make_ctx(tmp_path, rows, cells)
    agg = sweep.run_sweep(ctx)

    assert agg["requested"] == 2
    assert agg["completed"] == 2
    assert agg["skipped"] == 0
    assert agg["failed"] == 0
    assert agg["release_status"] == "VALID"
    assert agg["family"] is not None
    assert agg["family"]["verdict"] in {"PASS", "WARN", "FAIL"}
    for key, summary in agg["cells"].items():
        assert summary["verdict"] in {"PASS", "WARN", "FAIL"}
    # backend manifest exclusivity
    assert (ctx.output_dir / "run_manifest.json").exists()
    assert not (ctx.output_dir / "command_manifest.json").exists()
    # per-cell json present and re-derivable
    for cell in cells:
        p = ctx.output_dir / "cells" / f"{profile.cell_key_str(cell)}.json"
        assert p.exists()
        rec = json.loads(p.read_text())
        assert rec["status"] == "completed"
        assert "index_shift_by_item" in rec


def test_pipeline_determinism(tmp_path):
    rows = _synth_rows()
    cells = profile.smoke_cells()
    a1 = sweep.run_sweep(_make_ctx(tmp_path / "a", rows, cells))
    a2 = sweep.run_sweep(_make_ctx(tmp_path / "b", rows, cells))
    # Compare scientific content (family + per-cell verdicts/points).
    assert a1["family"] == a2["family"]
    assert a1["cells"] == a2["cells"]


def test_selector_ordering_pure():
    cands = [
        {"tolerance": "1e-10", "max_iterations": 200, "total_iterations": 50},
        {"tolerance": "1e-6", "max_iterations": 50, "total_iterations": 20},
        {"tolerance": "1e-8", "max_iterations": 100, "total_iterations": 20},
    ]
    ordered = order_eligible(cands)
    # smallest total_iterations first; tie -> larger tolerance (1e-6 > 1e-8)
    assert ordered[0]["tolerance"] == "1e-6"
    assert ordered[1]["tolerance"] == "1e-8"
    assert ordered[2]["total_iterations"] == 50
