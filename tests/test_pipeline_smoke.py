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
def smoke_artifact_dir(tmp_path_factory) -> Path:
    """Build the smoke dataset once and return its artifact directory."""
    out = tmp_path_factory.mktemp("mc_data")
    result = _run([
        "scripts/build_mc_dataset.py",
        "--smoke",
        "--output-dir", str(out),
    ])
    assert result.returncode == 0, f"build_mc_dataset failed:\n{result.stderr}"
    assert (out / "mc_dataset.json").exists(), f"mc_dataset.json not created in {out}"
    return out


@pytest.fixture(scope="module")
def smoke_pipeline_dir(smoke_artifact_dir: Path) -> Path:
    """Run baseline selection and PPO training once on split-aware defaults."""
    baseline = _run([
        "scripts/run_baselines.py",
        "--smoke",
        "--output-dir", str(smoke_artifact_dir),
        "likelihood.model=tfidf",
    ])
    assert baseline.returncode == 0, f"run_baselines failed:\n{baseline.stderr}"

    ppo = _run([
        "scripts/train_ppo.py",
        "--smoke",
        "--output-dir", str(smoke_artifact_dir),
        "--seed", "123",
        "--timesteps", "100",
        "likelihood.model=tfidf",
    ])
    assert ppo.returncode == 0, f"train_ppo failed:\n{ppo.stderr}"
    return smoke_artifact_dir


def _smoke_build_metadata(artifact_dir: Path) -> dict:
    """Load build metadata emitted by the smoke dataset builder."""
    return json.loads((artifact_dir / "build_metadata.json").read_text())


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
    combined = json.loads((tmp_path / "mc_dataset.json").read_text())
    train = json.loads((tmp_path / "train_dataset.json").read_text())
    val = json.loads((tmp_path / "val_dataset.json").read_text())
    test = json.loads((tmp_path / "test_dataset.json").read_text())
    metadata = json.loads((tmp_path / "build_metadata.json").read_text())

    assert combined == train + val + test
    assert metadata["combined_mc_dataset_count"] == len(combined)
    assert metadata["splits"]["train"]["retained_count"] == len(train)
    assert metadata["splits"]["val"]["retained_count"] == len(val)
    assert metadata["splits"]["test"]["retained_count"] == len(test)
    assert "drop_reasons" in metadata["splits"]["val"]
    assert "drop_reasons" in metadata["splits"]["test"]


@pytest.mark.slow
@pytest.mark.pipeline
def test_run_baselines_smoke(smoke_pipeline_dir: Path):
    """run_baselines.py defaults to the validation split when present."""
    summary = smoke_pipeline_dir / "baseline_summary.json"
    assert summary.exists(), f"baseline_summary.json not created in {smoke_pipeline_dir}"
    data = json.loads(summary.read_text())
    metadata = _smoke_build_metadata(smoke_pipeline_dir)
    expected_split = (
        "val" if metadata["splits"]["val"]["retained_count"] > 0 else "combined"
    )
    assert "softmax_profile" in data or "threshold" in data
    assert data["dataset_split"] == expected_split
    assert data["selection_metric"] == "mean_sq"


@pytest.mark.slow
@pytest.mark.pipeline
def test_train_ppo_smoke(smoke_pipeline_dir: Path):
    """train_ppo.py defaults to train/val split semantics and writes metadata."""
    assert (smoke_pipeline_dir / "ppo_model.zip").exists()
    assert (smoke_pipeline_dir / "ppo_summary.json").exists()
    assert (smoke_pipeline_dir / "config_used.json").exists()
    assert (smoke_pipeline_dir / "run_metadata.json").exists()

    summary = json.loads((smoke_pipeline_dir / "ppo_summary.json").read_text())
    config_used = json.loads((smoke_pipeline_dir / "config_used.json").read_text())
    run_metadata = json.loads((smoke_pipeline_dir / "run_metadata.json").read_text())
    metadata = _smoke_build_metadata(smoke_pipeline_dir)
    expected_eval_split = (
        "val" if metadata["splits"]["val"]["retained_count"] > 0 else "train"
    )
    assert summary["train_split"] == "train"
    assert summary["eval_split"] == expected_eval_split
    assert config_used["ppo"]["seed"] == 123
    assert config_used["environment"]["seed"] == 123
    assert config_used["ppo"]["total_timesteps"] == 100
    assert run_metadata["policy_mode"] == "flat_kplus1"


@pytest.mark.slow
@pytest.mark.pipeline
def test_evaluate_all_smoke(smoke_pipeline_dir: Path):
    """evaluate_all.py reports test-set evaluation with explicit split metadata."""
    result = _run([
        "scripts/evaluate_all.py",
        "--smoke",
        "--output-dir", str(smoke_pipeline_dir),
        "likelihood.model=tfidf",
    ])
    assert result.returncode == 0, f"evaluate_all failed:\n{result.stderr}"
    report = smoke_pipeline_dir / "evaluation_report.json"
    assert report.exists(), f"evaluation_report.json not created in {smoke_pipeline_dir}"
    data = json.loads(report.read_text())
    metadata = _smoke_build_metadata(smoke_pipeline_dir)
    expected_baseline_split = (
        "val" if metadata["splits"]["val"]["retained_count"] > 0 else "combined"
    )
    expected_ppo_validation_split = (
        "val" if metadata["splits"]["val"]["retained_count"] > 0 else "train"
    )
    expected_test_split = (
        "test" if metadata["splits"]["test"]["retained_count"] > 0 else "combined"
    )
    assert data["split_contract"]["baseline_selection_split"] == expected_baseline_split
    assert data["split_contract"]["softmax_eval_split"] == expected_test_split
    assert data["split_contract"]["ppo_validation_split"] == expected_ppo_validation_split
    assert data["split_contract"]["ppo_test_split"] == expected_test_split
    assert data["ppo_validation_summary"]["eval_split"] == expected_ppo_validation_split
    assert data["ppo_test_summary"]["eval_split"] == expected_test_split
    assert data["ppo_summary"]["eval_split"] == expected_test_split


@pytest.mark.slow
@pytest.mark.pipeline
def test_evaluate_all_discards_stale_validation_summary_when_training_incomplete(
    smoke_pipeline_dir: Path, tmp_path
):
    """Regression for the Codex P1 finding on PR #12.

    When ``run_metadata.json`` reports ``training_completed=False``, the
    co-written ``ppo_summary.json`` is also from the failed prior run
    and must not seep into ``evaluation_report.json`` via the
    "validation" fallback. Expected behavior: ``ppo_summary_source``
    falls through to ``"missing"`` and ``ppo_validation_summary`` is
    empty in the saved report.
    """
    import shutil

    work = tmp_path / "stale_metadata_run"
    shutil.copytree(smoke_pipeline_dir, work)
    run_metadata_path = work / "run_metadata.json"
    run_metadata = json.loads(run_metadata_path.read_text())
    run_metadata["training_completed"] = False
    run_metadata_path.write_text(json.dumps(run_metadata, indent=2))

    result = _run([
        "scripts/evaluate_all.py",
        "--smoke",
        "--output-dir", str(work),
        "likelihood.model=tfidf",
    ])
    assert result.returncode == 0, f"evaluate_all failed:\n{result.stderr}"
    assert "training_completed=False" in result.stdout, (
        "evaluate_all should print a loud warning about the stale "
        "checkpoint refusal."
    )
    report = json.loads((work / "evaluation_report.json").read_text())
    assert report["ppo_summary_source"] == "missing", (
        f"Expected ppo_summary_source='missing' but got "
        f"{report['ppo_summary_source']!r}; the stale ppo_summary.json "
        "from a prior run leaked into the report."
    )
    assert report["ppo_validation_summary"] == {}, (
        "ppo_validation_summary should be empty when training_completed=False; "
        "any non-empty value means stale prior-run metrics were preserved "
        "verbatim in the report."
    )
    assert report["ppo_test_summary"] == {}, (
        "ppo_test_summary should also be empty since replay is skipped."
    )


@pytest.mark.slow
@pytest.mark.pipeline
def test_evaluate_all_discards_stale_validation_summary_when_save_failed(
    smoke_pipeline_dir: Path, tmp_path
):
    """Regression for the second Codex P1 finding on PR #12.

    Symmetric to the ``training_completed=False`` case but on a freshly
    failed save: when ``ppo_model.zip`` is missing AND the
    ``run_metadata.json`` does NOT report ``training_completed=True``
    (failed save scenario), the stale ``ppo_summary.json`` from the
    failed run must not seep into the report.
    """
    import shutil

    work = tmp_path / "failed_save_run"
    shutil.copytree(smoke_pipeline_dir, work)
    # Simulate a failed agent.save(): metadata was overwritten with the
    # current run's values BEFORE the save failed, so training_completed
    # is False; ppo_model.zip never got rewritten so it gets removed
    # here to model the "fresh out_dir, save failed" subset.
    (work / "ppo_model.zip").unlink()
    rm_path = work / "run_metadata.json"
    rm = json.loads(rm_path.read_text())
    rm["training_completed"] = False
    rm_path.write_text(json.dumps(rm, indent=2))
    assert (work / "ppo_summary.json").exists()

    result = _run([
        "scripts/evaluate_all.py",
        "--smoke",
        "--output-dir", str(work),
        "likelihood.model=tfidf",
    ])
    assert result.returncode == 0, f"evaluate_all failed:\n{result.stderr}"
    assert "ppo_model.zip is missing" in result.stdout, (
        "evaluate_all should print a loud warning about the missing "
        "checkpoint with stale sidecars."
    )
    report = json.loads((work / "evaluation_report.json").read_text())
    assert report["ppo_summary_source"] == "missing", (
        f"Expected ppo_summary_source='missing' but got "
        f"{report['ppo_summary_source']!r}; the stale ppo_summary.json "
        "from a failed run leaked into the report."
    )
    assert report["ppo_validation_summary"] == {}, (
        "ppo_validation_summary should be empty when training_completed "
        "is False; any non-empty value means stale failed-run metrics "
        "were preserved verbatim."
    )


@pytest.mark.slow
@pytest.mark.pipeline
def test_evaluate_all_keeps_validation_summary_when_checkpoint_pruned(
    smoke_pipeline_dir: Path, tmp_path
):
    """Regression for the Codex P2 finding on PR #12.

    Distinguish "failed save" from "successful training, checkpoint
    later pruned for storage cleanup". When ``ppo_model.zip`` is
    missing but ``run_metadata.json`` reports
    ``training_completed=True``, ``ppo_summary.json`` holds valid
    validation metrics from the prior successful run -- evaluate_all.py
    must preserve them (replay is impossible, but the validation
    metrics are real). Pre-fix this branch was conflated with the
    failed-save branch and silently discarded valid metrics.
    """
    import shutil

    work = tmp_path / "pruned_checkpoint_run"
    shutil.copytree(smoke_pipeline_dir, work)
    # Smoke run was successful, so run_metadata already has
    # training_completed=True. Just remove the checkpoint to model
    # storage cleanup.
    (work / "ppo_model.zip").unlink()
    rm = json.loads((work / "run_metadata.json").read_text())
    assert rm.get("training_completed") is True, (
        "smoke fixture must report a completed training run for this "
        "test to exercise the pruned-checkpoint branch."
    )
    assert (work / "ppo_summary.json").exists()

    result = _run([
        "scripts/evaluate_all.py",
        "--smoke",
        "--output-dir", str(work),
        "likelihood.model=tfidf",
    ])
    assert result.returncode == 0, f"evaluate_all failed:\n{result.stderr}"
    assert "checkpoint pruned" in result.stdout, (
        "evaluate_all should print a NOTE explaining the pruned-checkpoint "
        "interpretation."
    )
    report = json.loads((work / "evaluation_report.json").read_text())
    assert report["ppo_summary_source"] == "validation", (
        f"Expected ppo_summary_source='validation' (valid pre-pruned "
        f"metrics) but got {report['ppo_summary_source']!r}; the "
        "pruned-checkpoint branch should preserve the validation summary."
    )
    assert report["ppo_validation_summary"] != {}, (
        "ppo_validation_summary must be preserved when training "
        "previously completed successfully and the checkpoint was "
        "pruned for storage cleanup."
    )
    assert report["ppo_test_summary"] == {}, (
        "ppo_test_summary should still be empty since replay is "
        "impossible without a checkpoint on disk."
    )


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
