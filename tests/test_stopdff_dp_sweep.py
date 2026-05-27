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
