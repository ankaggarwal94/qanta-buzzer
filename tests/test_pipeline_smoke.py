"""Subprocess-based smoke tests for pipeline entry points.

Each test runs a pipeline script as a subprocess with ``--output-dir``
pointing at a pytest ``tmp_path``, so no artifacts leak to ``artifacts/``.
These tests verify that each script's CLI wiring and end-to-end path
work without errors; they do not validate result quality.

Marked with ``@pytest.mark.slow`` and ``@pytest.mark.pipeline``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _run(args: list[str], timeout: int = 300) -> subprocess.CompletedProcess:
    """Run a Python command as a subprocess from the project root."""
    cmd = [sys.executable, *args]
    return subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


@pytest.fixture(scope="module")
def smoke_mc_dataset(tmp_path_factory) -> Path:
    """Build a smoke MC dataset once per module for reuse by downstream tests."""
    out = tmp_path_factory.mktemp("mc_data")
    result = _run([
        "scripts/build_mc_dataset.py",
        "--smoke",
        "--output-dir", str(out),
    ])
    assert result.returncode == 0, f"build_mc_dataset failed:\n{result.stderr}"
    mc_path = out / "mc_dataset.json"
    assert mc_path.exists(), f"mc_dataset.json not created in {out}"
    return mc_path


@pytest.mark.slow
@pytest.mark.pipeline
def test_build_mc_dataset_smoke(tmp_path):
    """build_mc_dataset.py --smoke --output-dir writes expected outputs."""
    result = _run([
        "scripts/build_mc_dataset.py",
        "--smoke",
        "--output-dir", str(tmp_path),
    ])
    assert result.returncode == 0, f"build_mc_dataset failed:\n{result.stderr}"
    assert (tmp_path / "mc_dataset.json").exists()
    assert (tmp_path / "train_dataset.json").exists()
    assert (tmp_path / "val_dataset.json").exists()
    assert (tmp_path / "test_dataset.json").exists()


@pytest.mark.slow
@pytest.mark.pipeline
def test_run_baselines_smoke(tmp_path, smoke_mc_dataset):
    """run_baselines.py --smoke --output-dir writes baseline_summary.json."""
    result = _run([
        "scripts/run_baselines.py",
        "--smoke",
        "--output-dir", str(tmp_path),
        "--mc-path", str(smoke_mc_dataset),
        "likelihood.model=tfidf",
    ])
    assert result.returncode == 0, f"run_baselines failed:\n{result.stderr}"
    summary = tmp_path / "baseline_summary.json"
    assert summary.exists(), f"baseline_summary.json not created in {tmp_path}"
    data = json.loads(summary.read_text())
    assert "softmax_profile" in data or "threshold" in data


@pytest.mark.slow
@pytest.mark.pipeline
def test_train_ppo_smoke(tmp_path, smoke_mc_dataset):
    """train_ppo.py --smoke --output-dir --timesteps 100 produces a model."""
    result = _run([
        "scripts/train_ppo.py",
        "--smoke",
        "--output-dir", str(tmp_path),
        "--mc-path", str(smoke_mc_dataset),
        "--timesteps", "100",
        "likelihood.model=tfidf",
    ])
    assert result.returncode == 0, f"train_ppo failed:\n{result.stderr}"
    assert (tmp_path / "ppo_model.zip").exists()
    assert (tmp_path / "ppo_summary.json").exists()
    assert (tmp_path / "config_used.json").exists()


@pytest.mark.slow
@pytest.mark.pipeline
def test_evaluate_all_smoke(tmp_path, smoke_mc_dataset):
    """evaluate_all.py --smoke --output-dir writes evaluation_report.json."""
    result = _run([
        "scripts/evaluate_all.py",
        "--smoke",
        "--output-dir", str(tmp_path),
        "--mc-path", str(smoke_mc_dataset),
        "likelihood.model=tfidf",
    ])
    assert result.returncode == 0, f"evaluate_all failed:\n{result.stderr}"
    report = tmp_path / "evaluation_report.json"
    assert report.exists(), f"evaluation_report.json not created in {tmp_path}"


@pytest.mark.slow
@pytest.mark.pipeline
@pytest.mark.skipif(
    not os.environ.get("RUN_PIPELINE_E2E"),
    reason="set RUN_PIPELINE_E2E=1 to run full 4-stage pipeline test",
)
def test_run_smoke_pipeline(tmp_path):
    """run_smoke_pipeline.py --output-dir runs all 4 stages in a temp dir.

    Skipped by default because it re-runs the full 4-stage pipeline (~18s),
    which the individual stage tests already cover. Run explicitly with:
        RUN_PIPELINE_E2E=1 pytest tests/test_pipeline_smoke.py -k run_smoke_pipeline
    """
    result = _run([
        "scripts/run_smoke_pipeline.py",
        "--output-dir", str(tmp_path),
    ], timeout=600)
    assert result.returncode == 0, (
        f"run_smoke_pipeline failed:\n{result.stdout}\n{result.stderr}"
    )
    summary_path = tmp_path / "smoke_pipeline_summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text())
    assert summary["status"] == "ok"
    assert len(summary["stages"]) == 4
    assert all(s["exit_code"] == 0 for s in summary["stages"])
