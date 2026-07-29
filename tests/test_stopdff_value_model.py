"""Focused tests for the learned StopDFF value-model trainer."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from models.stopdff_value_model import StopDFFValueModel
from scripts.stopdff_dp import dp_solver
from scripts.stopdff_dp.rewards import REWARD_REGISTRY, answer_utility
from scripts.stopdff_dp.types import ADAPTER_COLUMNS
from scripts.train_stopdff_value_model import (
    StopDFFTrajectory,
    build_fvi_targets,
    dataframe_to_trajectories,
)


def _adapter_row(
    *,
    item_id: str = "q1",
    prefix_idx: int = 0,
    split: str = "train",
    fmt: str = "MC",
    p_calibrated: float = 0.5,
) -> dict:
    return {
        "subject": "sbert:History",
        "item_id": item_id,
        "prefix_idx": prefix_idx,
        "prefix_fraction": float(prefix_idx + 1) / 3.0,
        "format": fmt,
        "split": split,
        "p_raw": p_calibrated,
        "p_calibrated": p_calibrated,
        "p_second_best": max(0.0, p_calibrated - 0.1) if fmt == "MC" else 0.0,
        "top2_margin": 0.1 if fmt == "MC" else 0.0,
        "correct": 1,
        "top_answer": "George Washington",
        "gold": "George Washington",
        "category": "History",
        "K": 4,
        "option_set_id": f"{item_id}:K4",
        "distractor_strategy": "test",
    }


def _adapter_df(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    for col in ADAPTER_COLUMNS:
        if col not in df.columns:
            df[col] = None
    return df[list(ADAPTER_COLUMNS)]


def _trajectory(
    *,
    p: list[float] | None = None,
    fractions: list[float] | None = None,
    features: np.ndarray | None = None,
) -> StopDFFTrajectory:
    p_values = p or [0.2, 0.4, 0.8]
    frac_values = fractions or [0.2, 0.5, 0.8]
    feature_values = (
        features
        if features is not None
        else np.arange(len(p_values), dtype=np.float32).reshape(-1, 1)
    )
    return StopDFFTrajectory(
        item_id="q1",
        fmt="MC",
        features=feature_values.astype(np.float32),
        p_calibrated=p_values,
        prefix_fractions=frac_values,
        prefix_indices=list(range(len(p_values))),
    )


def _fake_mc_question(qid: str, gold_text: str = "George Washington") -> dict:
    return {
        "qid": qid,
        "question": "Who was the first president of the United States?",
        "tokens": ["Who", "was", "the", "first", "president"],
        "answer_primary": gold_text,
        "clean_answers": [gold_text],
        "run_indices": [0, 4],
        "human_buzz_positions": [],
        "category": "History",
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
        "option_profiles": ["president", "vice", "second", "diplomat"],
        "option_answer_primary": [
            gold_text,
            "Thomas Jefferson",
            "John Adams",
            "Benjamin Franklin",
        ],
        "distractor_strategy": "test",
    }


def _write_tiny_training_data(data_dir: Path) -> None:
    data_dir.mkdir(parents=True)
    train_qs = [_fake_mc_question(f"train_{i}") for i in range(3)]
    val_qs = [_fake_mc_question(f"val_{i}") for i in range(2)]
    test_qs = [_fake_mc_question(f"test_{i}") for i in range(2)]
    (data_dir / "train_dataset.json").write_text(json.dumps(train_qs), encoding="utf-8")
    (data_dir / "val_dataset.json").write_text(json.dumps(val_qs), encoding="utf-8")
    (data_dir / "test_dataset.json").write_text(json.dumps(test_qs), encoding="utf-8")


def _run_smoke_training(tmp_path: Path, out_name: str) -> Path:
    from scripts import train_stopdff_value_model

    data_dir = tmp_path / "data"
    _write_tiny_training_data(data_dir)
    out_dir = tmp_path / out_name
    rc = train_stopdff_value_model.main(
        [
            "--smoke",
            "--device",
            "cpu",
            "--epochs",
            "2",
            "--seeds",
            "1",
            "--hidden",
            "8",
            "--batch-size",
            "4",
            "--data-dir",
            str(data_dir),
            "--artifact-dir",
            str(tmp_path / "paper_exports"),
            "--out",
            str(out_dir),
        ]
    )
    assert rc == 0
    return out_dir


def _load_checkpoint(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def test_no_test_split_leakage() -> None:
    df = _adapter_df([_adapter_row(split="test")])

    with pytest.raises(ValueError, match="split == 'test'"):
        dataframe_to_trajectories(df, fit=True)


def test_seed_determinism_within_one_seed(tmp_path: Path) -> None:
    out_a = _run_smoke_training(tmp_path / "run_a", "value_model")
    out_b = _run_smoke_training(tmp_path / "run_b", "value_model")

    ckpt_a = _load_checkpoint(out_a / "seed_1" / "best_model" / "best.ckpt")
    ckpt_b = _load_checkpoint(out_b / "seed_1" / "best_model" / "best.ckpt")

    assert ckpt_a["state_dict"].keys() == ckpt_b["state_dict"].keys()
    for key, tensor_a in ckpt_a["state_dict"].items():
        tensor_b = ckpt_b["state_dict"][key]
        assert torch.allclose(tensor_a, tensor_b, atol=1e-5), key


def test_save_load_round_trip(tmp_path: Path) -> None:
    torch.manual_seed(1)
    model = StopDFFValueModel(input_dim=3, hidden_sizes=(4,), dropout=0.0)
    model.eval()
    fixed_input = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    expected = model(fixed_input).detach()

    ckpt_path = tmp_path / "best.ckpt"
    torch.save(
        {"state_dict": model.state_dict(), "config": model.to_config()},
        ckpt_path,
    )
    payload = _load_checkpoint(ckpt_path)
    loaded = StopDFFValueModel.load_from_state_dict(
        payload["state_dict"],
        payload["config"],
    )
    loaded.eval()

    actual = loaded(fixed_input).detach()
    assert torch.allclose(actual, expected, atol=1e-5)


def test_fvi_targets_terminal_step_is_max_of_answer_or_zero() -> None:
    schedule = REWARD_REGISTRY["acf_flat"]
    traj = _trajectory(p=[0.2, 0.8], fractions=[0.25, 1.0])

    result = build_fvi_targets([traj], schedule, predict_fn=lambda _x: torch.tensor([99.0]))

    terminal_answer = answer_utility(
        traj.p_calibrated[-1],
        traj.prefix_fractions[-1],
        schedule,
    )
    assert result.targets[-1].item() == pytest.approx(max(terminal_answer, 0.0))
    assert result.traces[0]["continuation_values"][-1] == 0.0


def test_dp_target_construction_matches_solve_trajectory() -> None:
    schedule = REWARD_REGISTRY["wait_cost_small"]
    traj = _trajectory(
        p=[0.15, 0.55, 0.85],
        fractions=[0.2, 0.6, 1.0],
        features=np.array([[0.0], [1.5], [3.0]], dtype=np.float32),
    )

    def predict_fn(features: torch.Tensor) -> torch.Tensor:
        return features[:, 0]

    result = build_fvi_targets([traj], schedule, predict_fn=predict_fn)
    expected = dp_solver.solve_trajectory(
        p_trajectory=traj.p_calibrated,
        prefix_fractions=traj.prefix_fractions,
        schedule=schedule,
        continuation_fn=lambda t, **_kw: float(traj.features[t + 1, 0])
        if t + 1 < len(traj.features)
        else 0.0,
        item_id=traj.item_id,
        fmt=traj.fmt,
    )

    assert result.traces[0]["values"] == pytest.approx(expected.values)
    assert result.targets.tolist() == pytest.approx(expected.values)


def test_smoke_training_runs_on_cpu_in_under_60s(tmp_path: Path) -> None:
    started = time.monotonic()
    out_dir = _run_smoke_training(tmp_path, "value_model")
    elapsed = time.monotonic() - started

    assert (out_dir / "history.json").exists()
    assert (out_dir / "seed_1" / "history.json").exists()
    assert (out_dir / "seed_1" / "best_model" / "best.ckpt").exists()
    assert not (tmp_path / "paper_exports" / "stopdff_learned_value.json").exists()
    assert elapsed < 60.0


def test_audit_card_consumes_learned_value_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import make_audit_card
    from scripts import threshold_manifest

    paper = tmp_path / "paper_exports"
    paper.mkdir()
    (paper / "csli.json").write_text(
        json.dumps(
            {
                "per_model": {
                    "synthetic": {
                        "acc_choices_only": 0.25,
                        "question_use_gap": 0.0,
                        "leakage_flag": False,
                    }
                },
                "panel_csli": {
                    "definition": "max(0, acc_choices_only - 1/K)",
                    "mean": 0.0,
                },
                "panel_question_use_gap": {"mean": 0.0},
            }
        ),
        encoding="utf-8",
    )
    (paper / "calibration.json").write_text(
        json.dumps(
            {
                "max_ece": 0.01,
                "gate_verdict": "pass",
                "gate_verdict_reason": "threshold_only",
                "per_bucket": {
                    "early": {
                        "ece": 0.01,
                        "n_samples": 4,
                        "platt_model_type": "logistic",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    (paper / "stopdff.json").write_text(
        json.dumps(
            {
                "median_abs_prefix_shift": 0.0,
                "gate_verdict": "pass",
                "gate_verdict_reason": "threshold_only",
                "direction_breakdown": {
                    "mc_earlier": 0,
                    "qa_earlier": 0,
                    "same_step": 2,
                },
                "metadata": {"metric_type": "diagnostic_stopdff"},
            }
        ),
        encoding="utf-8",
    )
    (paper / "stopdff_learned_value.json").write_text(
        json.dumps(
            {
                "stopdff_signed_median": -0.5,
                "n_items": 2,
                "gate_verdict": "pass",
                "metadata": {
                    "checkpoint_path": "artifacts/value_model/seed_1/best_model/best.ckpt",
                    "seeds": [1],
                    "metric_type": "learned_value_dp",
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(make_audit_card, "_PAPER_EXPORTS", paper)
    monkeypatch.setattr(
        threshold_manifest,
        "load_frozen_threshold_manifest",
        lambda *_args, **_kwargs: {
            "thresholds": [
                {
                    "metric": "choices_only_accuracy",
                    "threshold": "1/K + 0.05",
                    "numeric_value_K4": 0.30,
                },
                {"metric": "prefix_ece", "threshold": 0.10},
                {"metric": "stopdff_median_abs_prefix", "threshold": 1},
            ]
        },
    )

    rc = make_audit_card.main_with_args(["--include-learned-value-stopdff"])

    assert rc == 0
    card = json.loads((paper / "audit_card.json").read_text(encoding="utf-8"))
    learned_rows = [
        metric
        for metric in card["metrics"]
        if metric["name"] == "Learned-Value StopDFF (Exploratory)"
    ]
    assert len(learned_rows) == 1
    row = learned_rows[0]
    assert row["exploratory"] is True
    assert row["value"] == -0.5
    assert row["details"]["value_model_seeds"] == [1]
