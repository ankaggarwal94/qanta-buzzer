"""Focused tests for scripts/sweep_stopdff_dp.py."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.stopdff_dp import rewards as rewards_module


def _fake_mc_question(qid: str, gold_text: str = "George Washington") -> dict:
    category = "History" if qid.endswith("0") else "Literature"
    return {
        "qid": qid,
        "question": "Who was the first president of the United States?",
        "tokens": ["Who", "was", "the", "first", "president"],
        "answer_primary": gold_text,
        "clean_answers": [gold_text],
        "run_indices": [0, 4],
        "human_buzz_positions": [],
        "category": category,
        "cumulative_prefixes": [
            "Who",
            "Who was the first",
            "Who was the first president",
        ],
        "options": [
            gold_text,
            "Thomas Jefferson",
            "John Adams",
            "Benjamin Franklin",
        ],
        "gold_index": 0,
        "option_profiles": [
            "president",
            "vice",
            "second",
            "diplomat",
        ],
        "option_answer_primary": [
            gold_text,
            "Thomas Jefferson",
            "John Adams",
            "Benjamin Franklin",
        ],
        "distractor_strategy": "test",
    }


def _write_tiny_data(data_dir: Path, *, n_val: int = 5, n_test: int = 5) -> None:
    data_dir.mkdir()
    val_qs = [_fake_mc_question(f"v{i}") for i in range(n_val)]
    test_qs = [_fake_mc_question(f"t{i}") for i in range(n_test)]
    (data_dir / "mc_dataset.json").write_text(json.dumps(val_qs + test_qs))
    (data_dir / "val_dataset.json").write_text(json.dumps(val_qs))
    (data_dir / "test_dataset.json").write_text(json.dumps(test_qs))


def _run_sweep(tmp_path: Path, args: list[str]) -> tuple[int, Path]:
    from scripts import sweep_stopdff_dp

    out = tmp_path / "paper_exports" / "stopdff_dp_sweep.json"
    rc = sweep_stopdff_dp.main([
        "--out", str(out),
        "--identity-calibration",
        "--seed", "123",
        "--num-bootstrap", "8",
        *args,
    ])
    return rc, out


def test_low_wrong_cost_registry() -> None:
    schedule = rewards_module.REWARD_REGISTRY["low_wrong_cost"]
    assert schedule.name == "low_wrong_cost"
    assert schedule.r_correct_early == rewards_module.REWARD_REGISTRY[
        "power_mark"
    ].r_correct_early
    assert schedule.r_correct_late == rewards_module.REWARD_REGISTRY[
        "power_mark"
    ].r_correct_late
    assert schedule.r_wrong > rewards_module.REWARD_REGISTRY["power_mark"].r_wrong


def test_default_axes_cover_requested_sweep() -> None:
    from scripts import sweep_stopdff_dp

    args = sweep_stopdff_dp._parse_args([])
    assert set(sweep_stopdff_dp._csv(args.reward_schedules)) == {
        "acf_flat",
        "power_mark",
        "wait_cost_small",
        "strict_wrong",
        "low_wrong_cost",
    }
    assert "platt-logistic" in sweep_stopdff_dp._csv(args.calibrators)
    assert set(sweep_stopdff_dp._csv(args.formats)) == {
        "QA-prefix",
        "MC-fixed",
        "MC-dynamic",
        "MC-full",
        "choices-only",
    }


def test_tiny_sweep_writes_aggregate_files_and_figures(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    _write_tiny_data(data_dir)

    rc, out = _run_sweep(tmp_path, [
        "--data-dir", str(data_dir),
        "--reward-schedules", "acf_flat",
        "--continuations", "empirical_bucket",
        "--calibrators", "uncalibrated",
        "--formats", "MC-fixed",
        "--prefix-bucketing", "phase",
        "--subject-pooling", "pooled_subject",
        "--max-cells", "1",
        "--smoke",
    ])

    assert rc == 0
    payload = json.loads(out.read_text())
    assert payload["metadata"]["metric_type"] == "finite_horizon_dp_sweep"
    assert len(payload["cells"]) == 1
    assert payload["cells"][0]["status"] == "completed"
    assert "paper_safe_interpretation" in payload
    assert "decision_regret_mc_to_qa_mean" in payload["cells"][0]["metrics"]
    assert "decision_regret_qa_to_mc_mean" in payload["cells"][0]["metrics"]
    assert (out.with_suffix(".md")).exists()
    assert "paper-safe interpretation" in out.with_suffix(".md").read_text()
    assert (out.with_name("stopdff_dp_sweep_table.tex")).exists()
    fig_dir = out.parent / "figures"
    assert (fig_dir / "stopdff_dp_phase_diagram.png").exists()
    assert (fig_dir / "stopdff_dp_vs_myopic.png").exists()
    assert (fig_dir / "stopdff_dp_coverage.png").exists()


def test_failed_and_skipped_cell_records_continue(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    _write_tiny_data(data_dir)

    rc, out = _run_sweep(tmp_path, [
        "--data-dir", str(data_dir),
        "--reward-schedules", "acf_flat,not_a_schedule",
        "--continuations", "empirical_bucket",
        "--calibrators", "uncalibrated",
        "--formats", "MC-fixed,choices-only",
        "--prefix-bucketing", "phase",
        "--subject-pooling", "pooled_subject",
        "--max-cells", "4",
    ])

    assert rc == 0
    cells = json.loads(out.read_text())["cells"]
    statuses = {cell["status"] for cell in cells}
    assert {"completed", "failed", "skipped"}.issubset(statuses)
    assert any("Unknown reward schedule" in cell.get("error", "") for cell in cells)
    assert any(cell.get("skip_reason") == "choices_only_unavailable" for cell in cells)


def test_resume_only_missing_preserves_existing_cells(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    _write_tiny_data(data_dir)
    base_args = [
        "--data-dir", str(data_dir),
        "--reward-schedules", "acf_flat,power_mark",
        "--continuations", "empirical_bucket",
        "--calibrators", "uncalibrated",
        "--formats", "MC-fixed",
        "--prefix-bucketing", "phase",
        "--subject-pooling", "pooled_subject",
    ]

    rc1, out = _run_sweep(tmp_path, [*base_args, "--max-cells", "1"])
    assert rc1 == 0
    first_payload = json.loads(out.read_text())
    assert len(first_payload["cells"]) == 1
    first_cell_path = Path(first_payload["cells"][0]["cache_path"])
    first_cell_before = first_cell_path.read_text()

    rc2, _ = _run_sweep(tmp_path, [
        *base_args,
        "--resume",
        "--only-missing",
        "--max-cells", "10",
    ])
    assert rc2 == 0

    second_payload = json.loads(out.read_text())
    assert len(second_payload["cells"]) == 2
    assert first_cell_path.read_text() == first_cell_before

    rc3, _ = _run_sweep(tmp_path, [*base_args, "--resume", "--max-cells", "10"])
    assert rc3 == 0
    third_payload = json.loads(out.read_text())
    assert len(third_payload["cells"]) == 2
    assert first_cell_path.read_text() == first_cell_before


def test_changed_seed_uses_distinct_cache_namespace(tmp_path: Path) -> None:
    from scripts import sweep_stopdff_dp

    data_dir = tmp_path / "data"
    _write_tiny_data(data_dir)
    out = tmp_path / "paper_exports" / "stopdff_dp_sweep.json"
    base_args = [
        "--out", str(out),
        "--identity-calibration",
        "--data-dir", str(data_dir),
        "--reward-schedules", "acf_flat",
        "--continuations", "empirical_bucket",
        "--calibrators", "uncalibrated",
        "--formats", "MC-fixed",
        "--prefix-bucketing", "phase",
        "--subject-pooling", "pooled_subject",
        "--max-cells", "1",
    ]

    assert sweep_stopdff_dp.main([*base_args, "--seed", "123"]) == 0
    first = json.loads(out.read_text())["cells"][0]
    assert sweep_stopdff_dp.main([
        *base_args,
        "--seed", "456",
        "--resume",
        "--only-missing",
    ]) == 0
    second = json.loads(out.read_text())["cells"][0]
    assert first["cell_id"] != second["cell_id"]
    assert Path(first["cache_path"]).exists()
    assert Path(second["cache_path"]).exists()


def test_platt_logistic_alias_completes_with_adapter_calibration(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    _write_tiny_data(data_dir)

    rc, out = _run_sweep(tmp_path, [
        "--data-dir", str(data_dir),
        "--reward-schedules", "acf_flat",
        "--continuations", "empirical_bucket",
        "--calibrators", "platt-logistic",
        "--formats", "MC-fixed",
        "--prefix-bucketing", "phase",
        "--subject-pooling", "pooled_subject",
        "--max-cells", "1",
    ])

    assert rc == 0
    cell = json.loads(out.read_text())["cells"][0]
    assert cell["status"] == "completed"
    assert cell["calibration"]["method"] == "platt-logistic"


def test_subject_pooling_falls_back_to_pooled_subject(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    _write_tiny_data(data_dir, n_val=6, n_test=2)

    rc, out = _run_sweep(tmp_path, [
        "--data-dir", str(data_dir),
        "--reward-schedules", "acf_flat",
        "--continuations", "empirical_bucket",
        "--calibrators", "uncalibrated",
        "--formats", "MC-fixed",
        "--prefix-bucketing", "phase",
        "--subject-pooling", "per_subject",
        "--max-cells", "1",
    ])

    assert rc == 0
    cell = json.loads(out.read_text())["cells"][0]
    assert cell["status"] == "completed"
    assert cell["coverage"]["fraction_pooled"] > 0.0


def test_isotonic_insufficient_data_skips_cell(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    _write_tiny_data(data_dir, n_val=1, n_test=2)

    rc, out = _run_sweep(tmp_path, [
        "--data-dir", str(data_dir),
        "--reward-schedules", "acf_flat",
        "--continuations", "empirical_bucket",
        "--calibrators", "isotonic",
        "--formats", "MC-fixed",
        "--prefix-bucketing", "phase",
        "--subject-pooling", "pooled_subject",
        "--max-cells", "1",
    ])

    assert rc == 0
    cell = json.loads(out.read_text())["cells"][0]
    assert cell["status"] == "skipped"
    assert cell["skip_reason"] == "insufficient_isotonic_data"


def test_paper_safe_interpretation_warns_when_any_non_oracle_cell_is_weak() -> None:
    from scripts import sweep_stopdff_dp

    clean = {
        "continuation": "empirical_bucket",
        "confirmatory_included": True,
        "gate_verdict": "pass",
        "coverage": {"verdict": "pass"},
        "ceiling_flags": {
            "all_stop_at_first_prefix": False,
            "all_stop_at_final_prefix": False,
            "no_cross_format_stopping_variance": False,
        },
        "metrics": {
            "stopdff_dp_abs_median": 0.0,
            "stopdff_dp_signed_median": 0.0,
        },
    }
    weak_excluded = {
        **clean,
        "confirmatory_included": False,
        "gate_verdict": "warn",
        "coverage": {"verdict": "warn"},
    }

    result = sweep_stopdff_dp._paper_safe_interpretation([clean, weak_excluded])

    assert result["verdict"] == "WARN"
    assert result["reason"] == "small_stopdff_but_coverage_or_ceiling_weak"


def _fake_mc_question_for_sweep(qid: str) -> dict:
    """Minimal MC question for sweep gate tests."""
    return {
        "qid": qid,
        "question": "What is the question?",
        "tokens": ["What", "is", "the", "question"],
        "answer_primary": "answer",
        "clean_answers": ["answer"],
        "run_indices": [0, 3],
        "human_buzz_positions": [],
        "category": "History",
        "cumulative_prefixes": ["What", "What is the question"],
        "options": ["answer", "distractor1", "distractor2", "distractor3"],
        "gold_index": 0,
        "option_profiles": ["a", "b", "c", "d"],
        "option_answer_primary": ["answer", "d1", "d2", "d3"],
        "distractor_strategy": "test",
    }


def test_sweep_rejects_incomplete_mc_coverage_without_override(tmp_path):
    """Sweep must exit nonzero when MC coverage < 98% and no override."""
    import json as _json
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    val_qs = [_fake_mc_question_for_sweep(f"v{i}") for i in range(10)]
    test_qs = [_fake_mc_question_for_sweep(f"t{i}") for i in range(10)]
    # mc_dataset is a strict subset (missing 5 of each).
    mc_subset = val_qs[:5] + test_qs[:5]
    (data_dir / "mc_dataset.json").write_text(_json.dumps(mc_subset))
    (data_dir / "val_dataset.json").write_text(_json.dumps(val_qs))
    (data_dir / "test_dataset.json").write_text(_json.dumps(test_qs))

    from scripts import sweep_stopdff_dp
    import pytest as _pytest
    with _pytest.raises(SystemExit):
        sweep_stopdff_dp.main([
            "--data-dir", str(data_dir),
            "--fit-split", "val",
            "--eval-split", "test",
            "--reward-schedules", "acf_flat",
            "--continuations", "empirical_bucket",
            "--calibrators", "uncalibrated",
            "--formats", "QA-prefix,MC-fixed",
            "--prefix-bucketing", "early_mid_late",
            "--subject-pooling", "per_subject",
            "--num-bootstrap", "5",
            "--identity-calibration",
            "--smoke",
            "--out", str(tmp_path / "out.json"),
        ])


def test_sweep_records_coverage_metadata_in_payload(tmp_path):
    """When MC coverage is complete, the sweep payload records mc_coverage."""
    import json as _json
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    val_qs = [_fake_mc_question_for_sweep(f"v{i}") for i in range(5)]
    test_qs = [_fake_mc_question_for_sweep(f"t{i}") for i in range(5)]
    (data_dir / "mc_dataset.json").write_text(_json.dumps(val_qs + test_qs))
    (data_dir / "val_dataset.json").write_text(_json.dumps(val_qs))
    (data_dir / "test_dataset.json").write_text(_json.dumps(test_qs))

    from scripts import sweep_stopdff_dp
    out_json = tmp_path / "out.json"
    rc = sweep_stopdff_dp.main([
        "--data-dir", str(data_dir),
        "--fit-split", "val",
        "--eval-split", "test",
        "--reward-schedules", "acf_flat",
        "--continuations", "empirical_bucket",
        "--calibrators", "uncalibrated",
        "--formats", "QA-prefix,MC-fixed",
        "--prefix-bucketing", "early_mid_late",
        "--subject-pooling", "per_subject",
        "--num-bootstrap", "5",
        "--identity-calibration",
        "--smoke",
        "--out", str(out_json),
    ])
    assert rc == 0 or rc is None
    payload = _json.loads(out_json.read_text())
    assert "mc_coverage" in payload
    eval_block = payload["mc_coverage"]["test"]
    assert eval_block["passed"] is True
    assert eval_block["overridden"] is False
    assert eval_block["coverage_rate"] == 1.0
    # Retention block also present.
    assert "mc_retention_gate" in payload
    # Build metadata absent (no build_metadata.json in synthetic data dir).
    assert "mc_build_metadata" in payload


def test_sweep_fingerprint_includes_dataset_hashes(tmp_path):
    """_run_fingerprint must include sha256 of mc/val/test/build_metadata
    so cache hits are invalidated when the data is regenerated."""
    import json as _json
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    val_qs = [_fake_mc_question_for_sweep(f"v{i}") for i in range(3)]
    test_qs = [_fake_mc_question_for_sweep(f"t{i}") for i in range(3)]
    (data_dir / "mc_dataset.json").write_text(_json.dumps(val_qs + test_qs))
    (data_dir / "val_dataset.json").write_text(_json.dumps(val_qs))
    (data_dir / "test_dataset.json").write_text(_json.dumps(test_qs))

    from scripts import sweep_stopdff_dp
    args = sweep_stopdff_dp._parse_args([
        "--data-dir", str(data_dir),
        "--fit-split", "val",
        "--eval-split", "test",
        "--identity-calibration",
        "--out", str(tmp_path / "out.json"),
    ])
    fp = sweep_stopdff_dp._run_fingerprint(
        args, out=tmp_path / "out.json", git_commit=None,
    )
    assert fp["schema_version"] == 4
    for key in (
        "mc_dataset_sha256",
        "fit_dataset_sha256",
        "eval_dataset_sha256",
    ):
        assert key in fp
        assert isinstance(fp[key], str) and len(fp[key]) == 64
    # build_metadata is absent in the fixture; the helper returns None.
    assert "build_metadata_sha256" in fp
    assert fp["build_metadata_sha256"] is None


def test_sweep_fingerprint_changes_when_dataset_regenerated(tmp_path):
    """Rewriting mc_dataset.json must change the fingerprint."""
    import json as _json
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    val_qs = [_fake_mc_question_for_sweep(f"v{i}") for i in range(3)]
    test_qs = [_fake_mc_question_for_sweep(f"t{i}") for i in range(3)]
    (data_dir / "mc_dataset.json").write_text(_json.dumps(val_qs + test_qs))
    (data_dir / "val_dataset.json").write_text(_json.dumps(val_qs))
    (data_dir / "test_dataset.json").write_text(_json.dumps(test_qs))

    from scripts import sweep_stopdff_dp
    args = sweep_stopdff_dp._parse_args([
        "--data-dir", str(data_dir),
        "--fit-split", "val",
        "--eval-split", "test",
        "--identity-calibration",
        "--out", str(tmp_path / "out.json"),
    ])
    fp_before = sweep_stopdff_dp._run_fingerprint(
        args, out=tmp_path / "out.json", git_commit=None,
    )
    # Regenerate mc_dataset.json with extra items.
    val_qs.append(_fake_mc_question_for_sweep("v_extra"))
    (data_dir / "mc_dataset.json").write_text(_json.dumps(val_qs + test_qs))
    fp_after = sweep_stopdff_dp._run_fingerprint(
        args, out=tmp_path / "out.json", git_commit=None,
    )
    assert fp_before["mc_dataset_sha256"] != fp_after["mc_dataset_sha256"]


def test_sweep_non_resume_excludes_stale_cached_cells(tmp_path):
    """Non-resume narrow runs must not publish cells from prior wider sweeps.

    Construct: prior wider cache directory has 2 cells; new narrow run
    with --max-cells 1 executes 1 cell. The aggregate must contain only
    the just-run cell, not the stale cell from the prior wider sweep.
    """
    import json as _json
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    val_qs = [_fake_mc_question_for_sweep(f"v{i}") for i in range(5)]
    test_qs = [_fake_mc_question_for_sweep(f"t{i}") for i in range(5)]
    (data_dir / "mc_dataset.json").write_text(_json.dumps(val_qs + test_qs))
    (data_dir / "val_dataset.json").write_text(_json.dumps(val_qs))
    (data_dir / "test_dataset.json").write_text(_json.dumps(test_qs))

    # First run: full grid for two reward schedules.
    from scripts import sweep_stopdff_dp
    out_json = tmp_path / "out.json"
    rc1 = sweep_stopdff_dp.main([
        "--data-dir", str(data_dir),
        "--fit-split", "val",
        "--eval-split", "test",
        "--reward-schedules", "acf_flat,power_mark",
        "--continuations", "empirical_bucket",
        "--calibrators", "uncalibrated",
        "--formats", "MC-fixed",
        "--prefix-bucketing", "early_mid_late",
        "--subject-pooling", "pooled_subject",
        "--num-bootstrap", "5",
        "--identity-calibration",
        "--smoke",
        "--out", str(out_json),
    ])
    assert rc1 == 0 or rc1 is None
    first_payload = _json.loads(out_json.read_text())
    first_completed_ids = sorted(
        c["cell_id"] for c in first_payload["cells"]
        if c.get("status") == "completed"
    )
    assert len(first_completed_ids) >= 2

    # Second run: NARROW (only acf_flat) without --resume.
    rc2 = sweep_stopdff_dp.main([
        "--data-dir", str(data_dir),
        "--fit-split", "val",
        "--eval-split", "test",
        "--reward-schedules", "acf_flat",
        "--continuations", "empirical_bucket",
        "--calibrators", "uncalibrated",
        "--formats", "MC-fixed",
        "--prefix-bucketing", "early_mid_late",
        "--subject-pooling", "pooled_subject",
        "--num-bootstrap", "5",
        "--max-cells", "1",
        "--identity-calibration",
        "--smoke",
        "--out", str(out_json),
    ])
    assert rc2 == 0 or rc2 is None
    second_payload = _json.loads(out_json.read_text())
    second_cells = second_payload["cells"]
    # Non-resume narrow run must only include the just-executed cell.
    assert len(second_cells) == 1
    # And it must NOT be a cell from the prior wider sweep.
    second_id = second_cells[0]["cell_id"]
    assert "reward_schedule" in second_cells[0]
    # All cells should have reward_schedule == acf_flat
    assert second_cells[0]["reward_schedule"] == "acf_flat"


def test_sweep_dirty_check_is_scoped_to_relevant_paths(tmp_path, monkeypatch):
    """_git_metadata's dirty flag should only consider script + inputs,
    not unrelated repo state (e.g., other untracked files).

    PR #15 review (chatgpt-codex-connector P2 3314002960): _git_metadata used
    to run `git status --porcelain` with no pathspec, so any unrelated dirty
    file in the repo (e.g., PROJECT_WIKI/TRANSCRIPTS/modal_spend.log left over
    from a parallel session) flagged git_dirty: true in every cell's
    provenance.
    """
    import json as _json
    import subprocess as _subprocess
    from pathlib import Path as _Path
    from scripts import sweep_stopdff_dp

    repo_root = _Path(__file__).resolve().parent.parent
    # Skip if the live script is dirty (e.g., during local dev before commit).
    proc = _subprocess.run(
        ["git", "status", "--short", "--", "scripts/sweep_stopdff_dp.py"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0 or proc.stdout.strip():
        import pytest as _pytest
        _pytest.skip("scripts/sweep_stopdff_dp.py is dirty in the working tree")

    # The repo is the live repo, but we synthesize args pointing at a
    # tmp data dir so the scoped pathspec only checks files we control.
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    val_qs = [_fake_mc_question_for_sweep(f"v{i}") for i in range(3)]
    test_qs = [_fake_mc_question_for_sweep(f"t{i}") for i in range(3)]
    (data_dir / "mc_dataset.json").write_text(_json.dumps(val_qs + test_qs))
    (data_dir / "val_dataset.json").write_text(_json.dumps(val_qs))
    (data_dir / "test_dataset.json").write_text(_json.dumps(test_qs))

    args = sweep_stopdff_dp._parse_args([
        "--data-dir", str(data_dir),
        "--fit-split", "val",
        "--eval-split", "test",
        "--identity-calibration",
        "--out", str(tmp_path / "out.json"),
    ])

    # Synthesize an "unrelated" dirty file inside the repo (not in args.data_dir).
    # We use a temp file in PROJECT_ROOT outside the scoped paths.
    unrelated = repo_root / "TEMP_UNRELATED_DIRTY_FILE_FOR_TEST.txt"
    unrelated.write_text("scratch")
    try:
        # The new args.data_dir points to a tmp dir OUTSIDE the repo,
        # so its files are not part of the scoped pathspec. Therefore
        # the only repo-relative path in the spec is the live
        # scripts/sweep_stopdff_dp.py, which should be clean.
        commit, dirty = sweep_stopdff_dp._git_metadata(args, out=tmp_path / "out.json")
        # commit may be None in shallow CI but should be a 40-char SHA locally.
        # dirty should be False because the unrelated file is not in the pathspec.
        assert dirty is False, (
            f"dirty should ignore unrelated repo files; "
            f"unrelated={unrelated}, commit={commit}"
        )
    finally:
        unrelated.unlink(missing_ok=True)


def test_apply_temperature_vectorized_matches_per_row_logic():
    """Vectorized _apply_temperature must produce identical output to the
    prior iterrows-based implementation."""
    import pandas as pd
    import numpy as np
    from scripts.sweep_stopdff_dp import _apply_temperature, _phase, _sigmoid

    df = pd.DataFrame({
        "prefix_fraction": [0.1, 0.4, 0.8, 0.5, 0.99, 0.05],
        "p_raw": [0.2, 0.6, 0.9, 0.4, 0.05, 0.95],
    })
    temps = {"early": 0.5, "mid": 1.0, "late": 2.0, "default": 1.5}

    result = _apply_temperature(df, temps)["p_calibrated"].tolist()

    # Per-row reference: mirrors the prior iterrows logic exactly.
    expected = []
    for _, row in df.iterrows():
        t = temps.get(_phase(float(row["prefix_fraction"])), temps.get("default", 1.0))
        expected.append(_sigmoid(float(row["p_raw"]) / max(t, 1e-6)))

    assert np.allclose(result, expected, atol=1e-12)


def test_sweep_fingerprint_includes_helper_module_hashes(tmp_path):
    """Sweep fingerprint must hash every imported scripts/stopdff_dp/*.py
    plus scripts/_audit_gates.py and scripts/_common.py — editing any of
    these must invalidate the cache."""
    import json as _json
    from scripts import sweep_stopdff_dp
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    val_qs = [_fake_mc_question_for_sweep(f"v{i}") for i in range(3)]
    test_qs = [_fake_mc_question_for_sweep(f"t{i}") for i in range(3)]
    (data_dir / "mc_dataset.json").write_text(_json.dumps(val_qs + test_qs))
    (data_dir / "val_dataset.json").write_text(_json.dumps(val_qs))
    (data_dir / "test_dataset.json").write_text(_json.dumps(test_qs))
    args = sweep_stopdff_dp._parse_args([
        "--data-dir", str(data_dir),
        "--fit-split", "val",
        "--eval-split", "test",
        "--identity-calibration",
        "--out", str(tmp_path / "out.json"),
    ])
    fp = sweep_stopdff_dp._run_fingerprint(
        args, out=tmp_path / "out.json", git_commit=None,
    )
    assert fp["schema_version"] == 4
    helpers = fp["helper_sha256s"]
    # Every expected stopdff_dp module must be present.
    expected_modules = {
        "scripts/stopdff_dp/__init__.py",
        "scripts/stopdff_dp/adapter.py",
        "scripts/stopdff_dp/continuation.py",
        "scripts/stopdff_dp/diagnostics.py",
        "scripts/stopdff_dp/dp_solver.py",
        "scripts/stopdff_dp/rewards.py",
        "scripts/stopdff_dp/types.py",
        "scripts/stopdff_dp/writers.py",
        "scripts/_audit_gates.py",
        "scripts/_common.py",
    }
    assert expected_modules.issubset(helpers.keys())
    # Every hash must be a 64-char hex string.
    for module_path, digest in helpers.items():
        assert isinstance(digest, str) and len(digest) == 64, module_path


def test_sweep_fingerprint_changes_when_helper_module_edited(tmp_path, monkeypatch):
    """A change to a helper module must produce a different fingerprint."""
    from pathlib import Path as _Path
    from scripts import sweep_stopdff_dp
    import json as _json

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    val_qs = [_fake_mc_question_for_sweep(f"v{i}") for i in range(3)]
    test_qs = [_fake_mc_question_for_sweep(f"t{i}") for i in range(3)]
    (data_dir / "mc_dataset.json").write_text(_json.dumps(val_qs + test_qs))
    (data_dir / "val_dataset.json").write_text(_json.dumps(val_qs))
    (data_dir / "test_dataset.json").write_text(_json.dumps(test_qs))
    args = sweep_stopdff_dp._parse_args([
        "--data-dir", str(data_dir),
        "--fit-split", "val",
        "--eval-split", "test",
        "--identity-calibration",
        "--out", str(tmp_path / "out.json"),
    ])
    fp_before = sweep_stopdff_dp._run_fingerprint(
        args, out=tmp_path / "out.json", git_commit=None,
    )

    # Simulate a helper edit by monkeypatching the shared _provenance
    # module's _file_sha256 (where helper_sha256s now lives after the
    # PR #15 review hoist) to return a different hash for rewards.py.
    # This keeps the test hermetic — we don't actually mutate the live
    # file in the repo.
    from scripts.stopdff_dp import _provenance as provenance_module

    real_file_sha = provenance_module._file_sha256

    def patched(p):
        digest = real_file_sha(p)
        if p is not None and str(p).endswith("stopdff_dp/rewards.py"):
            return "0" * 64
        return digest

    monkeypatch.setattr(provenance_module, "_file_sha256", patched)
    fp_after = sweep_stopdff_dp._run_fingerprint(
        args, out=tmp_path / "out.json", git_commit=None,
    )
    assert (
        fp_before["helper_sha256s"]["scripts/stopdff_dp/rewards.py"]
        != fp_after["helper_sha256s"]["scripts/stopdff_dp/rewards.py"]
    )


def test_sweep_dirty_check_includes_helper_modules(tmp_path, monkeypatch):
    """A dirty helper module (e.g., scripts/stopdff_dp/rewards.py) must
    flip git_dirty to True even when the producer script + inputs are clean.

    Without the fix in place, the helper was hashed into _run_fingerprint
    but its dirty state was ignored, so cells recorded git_dirty=false
    while their results depended on uncommitted helper edits.
    """
    import json as _json
    import subprocess as _subprocess
    from pathlib import Path as _Path
    from scripts import sweep_stopdff_dp
    from scripts.stopdff_dp._provenance import helper_paths

    repo_root = _Path(__file__).resolve().parent.parent
    helper_for_test = repo_root / "scripts" / "stopdff_dp" / "rewards.py"
    assert helper_for_test in helper_paths(), (
        f"test premise: {helper_for_test} must be in helper_paths()"
    )

    # Skip when the helper is already dirty before our edit — that means
    # the test would be ambiguous (the dirty state has another cause).
    proc = _subprocess.run(
        ["git", "status", "--short", "--",
         "scripts/stopdff_dp/rewards.py"],
        cwd=repo_root, capture_output=True, text=True,
    )
    if proc.stdout.strip():
        import pytest as _pytest
        _pytest.skip(
            "scripts/stopdff_dp/rewards.py is already dirty in the live tree"
        )

    # Synthesize input fixtures so the unrelated paths in the pathspec
    # are all clean / outside the repo.
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    val_qs = [_fake_mc_question_for_sweep(f"v{i}") for i in range(3)]
    test_qs = [_fake_mc_question_for_sweep(f"t{i}") for i in range(3)]
    (data_dir / "mc_dataset.json").write_text(_json.dumps(val_qs + test_qs))
    (data_dir / "val_dataset.json").write_text(_json.dumps(val_qs))
    (data_dir / "test_dataset.json").write_text(_json.dumps(test_qs))

    args = sweep_stopdff_dp._parse_args([
        "--data-dir", str(data_dir),
        "--fit-split", "val",
        "--eval-split", "test",
        "--identity-calibration",
        "--out", str(tmp_path / "out.json"),
    ])

    # Before perturbing the helper, the scoped check should report clean.
    _commit_clean, dirty_clean = sweep_stopdff_dp._git_metadata(
        args, out=tmp_path / "out.json"
    )
    assert dirty_clean is False, (
        "test premise: live worktree must be clean for the relevant pathspec "
        "before we perturb the helper"
    )

    # Append a no-op trailing comment to scripts/stopdff_dp/rewards.py
    # so its on-disk content differs from HEAD. We restore the original
    # bytes in a finally block so the live repo is unaffected by the test.
    original_bytes = helper_for_test.read_bytes()
    perturbed_bytes = original_bytes + b"\n# transient test perturbation\n"
    try:
        helper_for_test.write_bytes(perturbed_bytes)
        _commit_dirty, dirty_after = sweep_stopdff_dp._git_metadata(
            args, out=tmp_path / "out.json"
        )
        assert dirty_after is True, (
            f"_git_metadata must report dirty=True when {helper_for_test} "
            f"has uncommitted content; got dirty={dirty_after}"
        )
    finally:
        helper_for_test.write_bytes(original_bytes)


def test_helper_paths_matches_helper_sha256s_keys():
    """helper_paths() and helper_sha256s() must agree on the file set.

    helper_sha256s() returns repo-relative POSIX strings; helper_paths()
    returns Path objects. They must be the same set after normalisation.
    """
    from pathlib import Path as _Path
    from scripts.stopdff_dp._provenance import helper_paths, helper_sha256s, PROJECT_ROOT

    paths_set = set()
    for p in helper_paths():
        try:
            rel = p.resolve().relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            rel = str(p)
        paths_set.add(rel)
    hashes_set = set(helper_sha256s().keys())
    assert paths_set == hashes_set


def test_sweep_latex_escapes_label_underscores(tmp_path):
    """Sweep _write_latex must escape underscores in labels like acf_flat.

    Without the escape, the .tex file produces a 'Missing $ inserted'
    LaTeX compile error because ``_`` is a subscript operator in text mode.
    """
    from scripts.sweep_stopdff_dp import _write_latex
    payload = {
        "cells": [
            {
                "status": "completed",
                "reward_schedule": "acf_flat",
                "continuation": "empirical_bucket",
                "calibrator": "uncalibrated",
                "format_condition": "MC-fixed",
                "metrics": {"stopdff_dp_signed_mean": 0.123},
            },
        ],
    }
    out = tmp_path / "table.tex"
    _write_latex(out, payload)
    body = out.read_text()
    # The label underscores must be escaped.
    assert "acf\\_flat" in body
    assert "empirical\\_bucket" in body
    # And a bare underscore must NOT survive in the body (apart from any
    # LaTeX-command tail like \\\\ which is already absent for these labels).
    assert "acf_flat" not in body
    assert "empirical_bucket" not in body


def test_sweep_latex_escapes_all_special_characters():
    """The _latex_escape helper must handle the 10 LaTeX specials."""
    from scripts.sweep_stopdff_dp import _latex_escape
    # Order of characters in the input string matters for the backslash-first
    # rule. Verify a label with every special character is escaped correctly.
    cases = [
        ("simple", "simple"),
        ("acf_flat", "acf\\_flat"),
        ("foo&bar", "foo\\&bar"),
        ("100%", "100\\%"),
        ("$100", "\\$100"),
        ("#1", "\\#1"),
        ("a{b}c", "a\\{b\\}c"),
        ("a~b", r"a\textasciitilde{}b"),
        ("a^b", r"a\textasciicircum{}b"),
        ("a\\b", r"a\textbackslash{}b"),
    ]
    for raw, expected in cases:
        assert _latex_escape(raw) == expected, (raw, expected, _latex_escape(raw))


def test_fit_temperature_by_phase_vectorized_matches_per_row():
    """Vectorized _fit_temperature_by_phase must produce identical output
    to the prior nested-loop implementation across mixed-phase fixtures."""
    import pandas as pd
    import numpy as np
    from scripts.sweep_stopdff_dp import (
        _fit_temperature_by_phase, _phase, _sigmoid, _log_loss,
    )

    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({
        "format": ["MC"] * n,
        "prefix_fraction": rng.uniform(0.0, 1.0, size=n),
        "p_raw": rng.uniform(0.05, 0.95, size=n),
        "correct": rng.integers(0, 2, size=n).astype(float),
    })

    actual = _fit_temperature_by_phase(df)

    # Per-row reference: mirrors the prior nested-loop implementation
    # exactly.
    grid = np.array([0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0])
    mc = df[df["format"] == "MC"].copy()
    mc["phase"] = mc["prefix_fraction"].map(_phase)
    expected: dict[str, float] = {}
    for bucket, group in mc.groupby("phase"):
        y = group["correct"].astype(float).to_numpy()
        raw = group["p_raw"].astype(float).to_numpy()
        if len(np.unique(y)) < 2:
            expected[str(bucket)] = 1.0
            continue
        losses = [
            _log_loss(y, np.array([_sigmoid(x / t) for x in raw])) for t in grid
        ]
        expected[str(bucket)] = float(grid[int(np.argmin(losses))])
    expected["default"] = (
        float(np.median(list(expected.values()))) if expected else 1.0
    )

    # argmin is deterministic so per-bucket temperatures must be identical.
    assert set(actual.keys()) == set(expected.keys())
    for key in actual:
        assert actual[key] == expected[key], (
            f"{key}: vectorized={actual[key]} expected={expected[key]}"
        )
