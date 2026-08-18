"""Shared fixture builders for the hazard-efficacy harness RED tests.

Fabricates run directories whose sidecar files replicate the REAL
producers byte-for-schema (AP-031 format pinning):

- ``config_used.json``: resolved nested config PLUS the R-003 ``hazard``
  block and top-level ``seed``; producer
  ``scripts/train_t5_policy.py::main`` (written under ``<run>/ppo_t5/``).
- ``split_manifest.json``: exact field set of
  ``scripts/train_t5_policy.py::_build_split_manifest``.
- ``hazard_history.json``: pinned R-010 schema
  ``{"steps": [{"epoch", "question_index", "loss"}], "config": {...},
  "wall_clock_seconds": float}`` (top-level hazard-phase wall clock added
  in QA fix round 1, QA-006; producer:
  ``training/hazard_pretrain.py::run_hazard_pretrain``).
- ``eval_result.json``: pass-through keys use the REAL
  ``scripts/compare_policies.py::evaluate_t5_policy`` output names
  (``accuracy``, ``mean_sq``, ``ece``, ``brier``, ``avg_buzz_pos``,
  ``n_questions``, ``test_set_source``) plus the harness enrichment.
- ``RUN_COMPLETE.json``: harness marker incl. ``wall_clock_seconds``
  (the child's elapsed seconds, recorded by ``execute_plan``) and
  ``smoke`` (the invocation's --smoke flag, additive QA-R2-2 field; the
  fixture default ``True`` matches the smoke-resolved config the other
  sidecars replicate).
- Per-question eval ``runs`` records: R-002 field set.
- MCQuestion JSON split artifacts: complete records deserializable by the
  real persisted-split loader (schema: ``qb_data/mc_builder.py``; writer
  idiom from ``tests/test_train_seed_e1.py``).

Underscore-prefixed module: never collected by pytest.
"""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

# Default split qids shared by all fabricated runs (identical across arms
# unless a test doctors them).
DEFAULT_SPLIT_QIDS = {
    "train": ["t1", "t2", "t3", "t4"],
    "val": ["v1"],
    "test": ["q1", "q2", "q3"],
}


def current_git_sha() -> str:
    """Return the repo's current HEAD sha (real git, R-008/R-013 Through)."""
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def write_json(path: Path, obj: Any) -> Path:
    """Write ``obj`` as JSON, creating parents."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))
    return path


def hazard_block_for_arm(
    arm: str,
    *,
    beta_terminal: float = 1.0,
    freeze_answer_head: bool = False,
) -> dict:
    """R-003 hazard block: A = control, C = shuffled_nll ablation, else treatment."""
    pretrain = arm != "A"
    ablation = "shuffled_nll" if arm == "C" else None
    return {
        "pretrain": pretrain,
        "beta_terminal": beta_terminal,
        "freeze_answer_head": freeze_answer_head,
        "ablation": ablation,
    }


def make_config_used(
    run_dir: Path,
    *,
    arm: str = "A",
    seed: int = 1,
    model_name: str = "t5-small",
    device: str = "cpu",
    ppo_iterations: int = 5,
    ppo_checkpoint_dir: str = "checkpoints",
) -> dict:
    """Resolved nested config mirroring configs/t5_policy.yaml (smoke-resolved)."""
    return {
        "model": {
            "model_name": model_name,
            "device": device,
            "max_input_length": 128,
            "num_choices": 4,
        },
        "supervised": {
            "lr": 3.0e-4,
            "epochs": 2,
            "batch_size": 4,
            "grad_accum_steps": 1,
            "max_grad_norm": 1.0,
            "weight_decay": 0.01,
            "checkpoint_dir": str(run_dir),
        },
        "ppo": {
            "lr": 1.0e-5,
            "iterations": ppo_iterations,
            "batch_size": 4,
            "epochs_per_iter": 2,
            "clip_ratio": 0.2,
            "value_coef": 0.5,
            "entropy_coef": 0.01,
            "max_grad_norm": 0.5,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "target_kl": 0.03,
            "eval_interval": 1,
            "checkpoint_dir": ppo_checkpoint_dir,
        },
        "data": {
            "csv_path": "questions.csv",
            "K": 4,
            "train_size": 0.7,
            "val_size": 0.15,
            "test_size": 0.15,
            "seed": 42,
            "max_questions": 50,
            "max_questions_scope": "global",
        },
        "seed": seed,
        "hazard": hazard_block_for_arm(arm),
    }


def make_split_manifest(
    *,
    source: str = "persisted_artifacts",
    split_qids: dict | None = None,
) -> dict:
    """Replicate scripts/train_t5_policy.py::_build_split_manifest fields."""
    qids = split_qids or DEFAULT_SPLIT_QIDS
    train, val, test = qids["train"], qids["val"], qids["test"]
    total = len(train) + len(val) + len(test)
    return {
        "source": source,
        "mc_path": "artifacts/smoke/mc_dataset.json",
        "train_path": "artifacts/smoke/train_dataset.json",
        "val_path": "artifacts/smoke/val_dataset.json",
        "test_path": "artifacts/smoke/test_dataset.json",
        "train_qids": list(train),
        "val_qids": list(val),
        "test_qids": list(test),
        "train_count": len(train),
        "val_count": len(val),
        "test_count": len(test),
        "effective_train_ratio": len(train) / max(1, total),
        "effective_val_ratio": len(val) / max(1, total),
        "effective_test_ratio": len(test) / max(1, total),
    }


def make_eval_runs(qids: list[str] | None = None) -> list[dict]:
    """Per-question R-002 ``runs`` records: 2 policy buzzes (1 correct at
    position 4), 1 forced commit."""
    qids = qids or list(DEFAULT_SPLIT_QIDS["test"])
    template = [
        {"sq": 0.20, "buzz_position": 4, "buzzed": True, "correct": True,
         "forced_correct": False, "confidence": 0.8, "episode_reward": 1.0,
         "n_steps": 4},
        {"sq": 0.10, "buzz_position": 2, "buzzed": True, "correct": False,
         "forced_correct": False, "confidence": 0.6, "episode_reward": -1.0,
         "n_steps": 2},
        {"sq": 0.05, "buzz_position": None, "buzzed": False, "correct": False,
         "forced_correct": True, "confidence": None, "episode_reward": 0.5,
         "n_steps": 6},
    ]
    runs = []
    for i, qid in enumerate(qids):
        rec = dict(template[i % len(template)])
        rec["qid"] = qid
        runs.append(rec)
    return runs


def make_eval_result(
    arm: str,
    seed: int,
    *,
    mean_sq: float = 0.1,
    accuracy: float = 1.0 / 3.0,
    mean_correct_buzz_position: float | None = 4.0,
    n_correct_policy_buzzes: int = 1,
    qids: list[str] | None = None,
) -> dict:
    """Fabricated ``<run_dir>/eval_result.json`` payload (harness schema).

    Pass-through metric keys are the REAL ``evaluate_t5_policy`` output
    names (``scripts/compare_policies.py``): ``accuracy``, ``mean_sq``,
    ``ece``, ``brier``, ``avg_buzz_pos``, ``n_questions``,
    ``test_set_source`` — never fixture-only aliases, so GREEN cannot key
    the report to names the real producer never emits (AP-031).
    """
    runs = make_eval_runs(qids)
    n = len(runs)
    buzzed = sum(1 for r in runs if r["buzzed"])
    return {
        "arm": arm,
        "seed": seed,
        "accuracy": accuracy,
        "mean_sq": mean_sq,
        "ece": 0.10,
        "brier": 0.20,
        "avg_buzz_pos": 3.0,
        "n_questions": n,
        "test_set_source": "persisted_artifacts",
        "policy_buzz_rate": buzzed / n,
        "forced_commit_rate": (n - buzzed) / n,
        "n_correct_policy_buzzes": n_correct_policy_buzzes,
        "mean_correct_buzz_position": mean_correct_buzz_position,
        "runs": runs,
    }


def make_hazard_history(
    losses: list[float] | None = None,
    *,
    wall_clock_seconds: float = 3.75,
) -> dict:
    """Pinned R-010 hazard_history.json schema (QA-006: top-level
    ``wall_clock_seconds`` = the HAZARD-PHASE wall clock; the default 3.75
    is deliberately distinct from every RUN_COMPLETE.json marker value used
    by the tests so the report's data source is discriminable)."""
    losses = losses if losses is not None else [4.0, 4.0, 2.0, 2.0]
    steps = [
        {"epoch": i // 2, "question_index": i % 2, "loss": float(loss)}
        for i, loss in enumerate(losses)
    ]
    return {
        "steps": steps,
        "config": {
            "beta_terminal": 1.0,
            "freeze_answer_head": False,
            "ablation": None,
            "lr": 1e-3,
            "epochs": 2,
        },
        "wall_clock_seconds": float(wall_clock_seconds),
    }


def make_hazard_dynamics() -> dict:
    """Fabricated ``<run_dir>/hazard_dynamics.json`` (R-010b block)."""
    return {
        "per_position_mean_before": [0.1, 0.1, 0.1, 0.1],
        "per_position_mean_after": [0.3, 0.4, 0.5, 0.6],
        "expected_buzz_time_before": 3.5,
        "expected_buzz_time_after": 2.1,
        "expected_buzz_time_delta": -1.4,
        "first_half_mean_loss": 4.0,
        "second_half_mean_loss": 2.0,
    }


def make_run_dir(
    base: Path,
    arm: str,
    seed: int,
    *,
    hazard: bool | None = None,
    marker: bool = True,
    marker_git_sha: str | None = None,
    marker_wall_clock_seconds: float = 12.5,
    marker_smoke: bool = True,
    marker_extra: dict[str, Any] | None = None,
    model_name: str = "t5-small",
    ppo_checkpoint_dir: str = "checkpoints",
    split_source: str = "persisted_artifacts",
    split_qids: dict | None = None,
    write_eval_result: bool = True,
    eval_overrides: dict | None = None,
    write_checkpoints: bool = True,
    include_prunables: bool = False,
    include_hazard_dynamics: bool = False,
    hazard_history_losses: list[float] | None = None,
    config_mutations: dict[str, Any] | None = None,
) -> Path:
    """Fabricate a full ``<base>/<arm>_seed<seed>`` run directory.

    Layout mirrors the real training pipeline: ``ppo_t5/`` holds the
    sidecars + ``best_model/``; hazard arms additionally hold
    ``hazard/best_model/`` + ``hazard/hazard_history.json``; the harness
    adds ``eval_result.json`` and ``RUN_COMPLETE.json`` at the top.
    """
    if hazard is None:
        hazard = arm != "A"
    run_dir = base / f"{arm}_seed{seed}"
    ppo_dir = run_dir / "ppo_t5"
    ppo_dir.mkdir(parents=True, exist_ok=True)

    config_used = make_config_used(
        run_dir,
        arm=arm,
        seed=seed,
        model_name=model_name,
        ppo_checkpoint_dir=ppo_checkpoint_dir,
    )
    for dotted, value in (config_mutations or {}).items():
        node = config_used
        parts = dotted.split(".")
        for part in parts[:-1]:
            node = node[part]
        node[parts[-1]] = value
    write_json(ppo_dir / "config_used.json", config_used)
    write_json(
        ppo_dir / "split_manifest.json",
        make_split_manifest(source=split_source, split_qids=split_qids),
    )
    write_json(ppo_dir / "history.json", [])

    if write_checkpoints:
        best = ppo_dir / "best_model"
        best.mkdir(parents=True, exist_ok=True)
        (best / "policy_head.pt").write_bytes(b"stub-weights")
        if hazard:
            hz_best = run_dir / "hazard" / "best_model"
            hz_best.mkdir(parents=True, exist_ok=True)
            (hz_best / "policy_head.pt").write_bytes(b"stub-weights")

    if hazard:
        write_json(
            run_dir / "hazard" / "hazard_history.json",
            make_hazard_history(hazard_history_losses),
        )
        if include_hazard_dynamics:
            write_json(run_dir / "hazard_dynamics.json", make_hazard_dynamics())

    if include_prunables:
        for sub in ("iter_1", "iter_2", "epoch_1"):
            d = ppo_dir / sub
            d.mkdir(parents=True, exist_ok=True)
            (d / "policy_head.pt").write_bytes(b"stub-weights")
        (ppo_dir / "iter_1" / "training_state.pt").write_bytes(b"optim-state")
        if write_checkpoints:
            (ppo_dir / "best_model" / "training_state.pt").write_bytes(b"optim-state")

    if write_eval_result:
        payload = make_eval_result(arm, seed, qids=(split_qids or DEFAULT_SPLIT_QIDS)["test"])
        payload.update(eval_overrides or {})
        write_json(run_dir / "eval_result.json", payload)

    if marker:
        marker_payload: dict[str, Any] = {
            "git_sha": marker_git_sha or current_git_sha(),
            "arm": arm,
            "seed": seed,
            "completed_at": "2026-08-18T00:00:00Z",
            "wall_clock_seconds": marker_wall_clock_seconds,
            "smoke": marker_smoke,
        }
        # marker_extra (additive, mini-audit round): extra/overriding marker
        # fields, e.g. MA-003's shared_supervised_weights_sha256 or MA-015's
        # doctored-type fields.
        marker_payload.update(marker_extra or {})
        write_json(run_dir / "RUN_COMPLETE.json", marker_payload)
    return run_dir


def make_mc_question_record(qid: str, gold_index: int = 0) -> dict:
    """Complete MCQuestion JSON record (schema: ``qb_data/mc_builder.py``).

    Every field required by the real persisted-split loader is present
    (writer idiom adapted from ``tests/test_train_seed_e1.py``).
    """
    return {
        "qid": qid,
        "question": "Who was the first president of the United States",
        "tokens": ["Who", "was", "the", "first", "president"],
        "answer_primary": "George Washington",
        "clean_answers": ["George Washington"],
        "run_indices": [0, 2, 4],
        "human_buzz_positions": [],
        "category": "History",
        "cumulative_prefixes": [
            "Who",
            "Who was the",
            "Who was the first president",
        ],
        "options": [
            "George Washington",
            "Thomas Jefferson",
            "John Adams",
            "Benjamin Franklin",
        ],
        "gold_index": gold_index,
        "option_profiles": [
            "George Washington first president commander revolutionary war",
            "Thomas Jefferson third president declaration independence",
            "John Adams second president Massachusetts diplomat",
            "Benjamin Franklin inventor diplomat Philadelphia",
        ],
        "option_answer_primary": [
            "George Washington",
            "Thomas Jefferson",
            "John Adams",
            "Benjamin Franklin",
        ],
        "distractor_strategy": "test",
    }


def write_split_artifacts(
    artifacts_dir: Path, split_qids: dict | None = None
) -> dict[str, Path]:
    """Write REAL persisted split artifacts (train/val/test + mc) to disk.

    Returns ``{"train": Path, "val": Path, "test": Path}`` for the three
    ``<split>_dataset.json`` files; a combined ``mc_dataset.json`` sits
    beside them. Records are complete MCQuestion JSON so the real loader
    deserializes them.
    """
    qids = split_qids or DEFAULT_SPLIT_QIDS
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    combined: list[dict] = []
    for split in ("train", "val", "test"):
        records = [
            make_mc_question_record(qid, gold_index=i % 4)
            for i, qid in enumerate(qids[split])
        ]
        paths[split] = write_json(
            artifacts_dir / f"{split}_dataset.json", records
        )
        combined.extend(records)
    write_json(artifacts_dir / "mc_dataset.json", combined)
    return paths


def make_plan_record(
    out_dir: Path,
    arm: str,
    seed: int,
    *,
    hazard: bool | None = None,
    argv: list[str] | None = None,
) -> dict:
    """Hand-built plan record matching the pinned plan_runs record shape."""
    run_dir = out_dir / f"{arm}_seed{seed}"
    return {
        "arm": arm,
        "seed": seed,
        "run_dir": run_dir,
        "hazard": (arm != "A") if hazard is None else hazard,
        "argv": argv
        or [sys.executable, str(REPO_ROOT / "scripts" / "train_t5_policy.py"), "--fake"],
        "log_path": run_dir / "train.log",
        "variant": None,
    }


def fabricate_child_outputs(
    record: dict,
    *,
    split_source: str = "persisted_artifacts",
    model_name: str = "t5-small",
) -> None:
    """What a successful child run leaves behind (no marker: harness-owned)."""
    make_run_dir(
        Path(record["run_dir"]).parent,
        record["arm"],
        record["seed"],
        hazard=record.get("hazard", record["arm"] != "A"),
        marker=False,
        write_eval_result=False,
        split_source=split_source,
        model_name=model_name,
    )


def fabricating_runner(
    records: list[dict],
    *,
    split_source: str = "persisted_artifacts",
    exit_code: int = 0,
    log_text: str = "child ok\n",
):
    """Fake ``_run_child(argv, log_path)`` that fabricates child outputs.

    Returns ``(runner, calls)`` where ``calls`` collects
    ``(argv, log_path)`` tuples.
    """
    by_log = {str(Path(r["log_path"])): r for r in records}
    calls: list[tuple[list[str], Path]] = []

    def runner(argv, log_path):
        calls.append((list(argv), Path(log_path)))
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        Path(log_path).write_text(log_text)
        if exit_code == 0:
            fabricate_child_outputs(
                by_log[str(Path(log_path))], split_source=split_source
            )
        return exit_code

    return runner, calls


def deep_key_scan(obj: Any, needle: str) -> list[str]:
    """Recursively collect dict keys containing ``needle`` (case-insensitive)."""
    found: list[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if needle.lower() in str(k).lower():
                found.append(str(k))
            found.extend(deep_key_scan(v, needle))
    elif isinstance(obj, list):
        for item in obj:
            found.extend(deep_key_scan(item, needle))
    return found


def copy_with(obj: dict, **kwargs: Any) -> dict:
    """Deep-copy a dict and update top-level keys."""
    out = copy.deepcopy(obj)
    out.update(kwargs)
    return out
