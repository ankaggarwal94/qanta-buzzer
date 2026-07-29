#!/usr/bin/env python3
"""Train a learned StopDFF continuation-value model.

The model learns fixed-point value-iteration targets over adapter rows from
``scripts.stopdff_dp``. Training data is built only from the configured train
split and validation only from the configured val split. The test split is
read only for optional qid-overlap rejection when ``test_dataset.json`` is
present; test rows are never converted into training trajectories.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.stopdff_value_model import StopDFFValueModel
from scripts._audit_gates import (
    build_retention_metadata,
    filter_mc_questions_to_split,
    load_mc_build_metadata,
)
from scripts._common import iter_split_questions, load_json, sha256_file
from scripts.fresh_split import set_all_seeds
from scripts.stopdff_dp.adapter import (
    build_dataframe,
    validate_qid_separation,
    validate_split_separation,
)
from scripts.stopdff_dp.dp_solver import solve_trajectory
from scripts.stopdff_dp.rewards import get_schedule
from scripts.stopdff_dp.types import RewardSchedule

DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "paper_exports"
DEFAULT_OUT = PROJECT_ROOT / "artifacts" / "value_model"

NUMERIC_COLUMNS = [
    "prefix_idx",
    "prefix_fraction",
    "p_raw",
    "p_calibrated",
    "p_second_best",
    "top2_margin",
    "K",
]
CATEGORICAL_COLUMNS = ["format", "category", "distractor_strategy"]
TARGET_STRATEGY = "fvi_next_prefix_bootstrap"
MIN_MC_COVERAGE = 0.98
PRODUCER_SCRIPT_PATH = "scripts/train_stopdff_value_model.py"


@dataclass
class StopDFFTrajectory:
    """One sorted (item_id, format) trajectory for FVI target building."""

    item_id: str
    fmt: str
    features: np.ndarray
    p_calibrated: list[float]
    prefix_fractions: list[float]
    prefix_indices: list[int]


@dataclass
class TrajectoryBundle:
    """Feature-preprocessed trajectories plus reusable preprocessing state."""

    trajectories: list[StopDFFTrajectory]
    feature_schema: dict[str, Any]
    scaler: dict[str, Any]


@dataclass
class FVIResult:
    """Flattened fitted-value-iteration targets and solver trace metadata."""

    features: torch.Tensor
    targets: torch.Tensor
    dataset: "ValueTargetDataset"
    traces: list[dict[str, Any]]
    target_stats: dict[str, float]


class ValueTargetDataset(Dataset):
    """Torch dataset of encoded features and scalar value targets."""

    def __init__(self, features: torch.Tensor, targets: torch.Tensor) -> None:
        if features.ndim != 2:
            raise ValueError("features must have shape [n, d]")
        if targets.ndim != 1:
            raise ValueError("targets must have shape [n]")
        if features.shape[0] != targets.shape[0]:
            raise ValueError("features and targets must have matching first dimension")
        self.features = features.float()
        self.targets = targets.float()

    def __len__(self) -> int:
        return int(self.targets.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.targets[idx]


def _target_bounds(schedule: RewardSchedule) -> tuple[float, float]:
    """Known finite-horizon value range under the configured reward schedule."""
    lower = min(0.0, float(schedule.r_wrong))
    upper = max(
        0.0,
        float(schedule.r_correct_early),
        float(schedule.r_correct_late),
    )
    return lower, upper


def _resolve_path(path: str | Path) -> Path:
    raw = Path(path)
    return raw if raw.is_absolute() else PROJECT_ROOT / raw


def _resolve_calibration_path(args: argparse.Namespace) -> Path:
    if args.calibration:
        return _resolve_path(args.calibration)
    return _resolve_path(args.artifact_dir) / "calibration_train.json"


def _calibration_fit_split(path: Path) -> str:
    with path.open("r", encoding="utf-8") as f:
        metadata = json.load(f).get("metadata", {})
    return str(metadata.get("fit_split", "")).lower()


def _resolve_train_fit_calibration(args: argparse.Namespace) -> Path | None:
    if args.effective_identity_calibration:
        return None

    calibration_path = _resolve_calibration_path(args)
    if not calibration_path.exists():
        raise FileNotFoundError(f"calibration JSON not found: {calibration_path}")

    train_split = str(args.train_split).lower()
    fit_split = _calibration_fit_split(calibration_path)
    if fit_split == train_split:
        return calibration_path

    # The pre-existing Device 2 wrapper has a global --calibration default
    # pointing at the val-fit audit artifact. For this new trainer, recover
    # by using the train-fit sibling generated by the Prompt 5 pre-step.
    fallback_path = calibration_path.with_name("calibration_train.json")
    if calibration_path.name == "calibration.json" and fallback_path.exists():
        fallback_fit_split = _calibration_fit_split(fallback_path)
        if fallback_fit_split == train_split:
            print(
                "INFO: using train-fit calibration artifact "
                f"{fallback_path} instead of runner default {calibration_path}",
                file=sys.stderr,
            )
            return fallback_path

    raise ValueError(
        "learned-value training requires a calibration artifact fit on "
        f"{args.train_split!r}; got fit_split={fit_split!r} from "
        f"{calibration_path}. Pass --calibration "
        "paper_exports/calibration_train.json or use --smoke for the "
        "identity-calibration test path."
    )


def _parse_csv_ints(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return values


def _parse_hidden(raw: str) -> tuple[int, ...]:
    values = tuple(_parse_csv_ints(raw))
    if any(v <= 0 for v in values):
        raise argparse.ArgumentTypeError("--hidden values must all be > 0")
    return values


def _git_sha() -> str:
    """Return ``git rev-parse HEAD`` or ``unknown`` when unavailable."""
    injected = os.environ.get("MODAL_HOST_GIT_COMMIT")
    if injected is not None:
        return injected or "unknown"
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"
    if result.returncode != 0:
        return "unknown"
    return result.stdout.strip() or "unknown"


def _git_dirty() -> bool:
    """Return whether the working tree has any tracked or untracked changes."""
    injected = os.environ.get("MODAL_HOST_GIT_STATUS")
    if injected is not None:
        return bool(injected.strip())
    try:
        result = subprocess.run(
            ["git", "status", "--short"],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0 and bool(result.stdout.strip())


def _committed_script_sha256(commit: str, script_path: str) -> str | None:
    injected = os.environ.get("MODAL_HOST_PRODUCER_SCRIPT_SHA256")
    if injected:
        return injected
    try:
        result = subprocess.run(
            ["git", "show", f"{commit}:{script_path}"],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return hashlib.sha256(result.stdout).hexdigest()


def _training_provenance() -> dict[str, Any]:
    script_sha = sha256_file(PROJECT_ROOT / PRODUCER_SCRIPT_PATH)
    commit = _git_sha()
    dirty = _git_dirty()
    if dirty:
        raise RuntimeError("dirty training producer or dependency state")
    if (
        len(commit) != 40
        or any(char not in "0123456789abcdef" for char in commit.lower())
    ):
        raise RuntimeError("missing exact training producer commit")
    committed_sha = _committed_script_sha256(commit, PRODUCER_SCRIPT_PATH)
    if committed_sha != script_sha:
        raise RuntimeError(
            "training source commit does not contain the exact producer bytes"
        )
    return {
        "script_path": PRODUCER_SCRIPT_PATH,
        "script_sha256": script_sha,
        "git_commit": commit,
        "git_dirty": False,
        "commit_script_sha256": committed_sha,
        "commit_contains_exact_script": True,
    }


def _checkpoint_save_identity(producer_provenance: dict[str, Any]) -> str:
    """Revalidate source identity immediately before publishing a checkpoint."""
    commit = _git_sha()
    if _git_dirty():
        raise RuntimeError("dirty checkpoint save state")
    if commit != producer_provenance.get("git_commit"):
        raise RuntimeError(
            "checkpoint save commit differs from the validated training producer"
        )
    return commit


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if hasattr(obj, "__dataclass_fields__"):
        return _to_jsonable(asdict(obj))
    return obj


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_to_jsonable(payload), f, indent=2)


def _assert_no_test_rows(df: pd.DataFrame, *, context: str) -> None:
    if "split" not in df.columns:
        return
    if (df["split"].astype(str).str.lower() == "test").any():
        raise ValueError(f"{context} dataframe contains split == 'test' rows")


def _fit_feature_schema(df: pd.DataFrame) -> dict[str, Any]:
    categorical_levels: dict[str, list[str]] = {}
    for col in CATEGORICAL_COLUMNS:
        values = df[col].fillna("unknown").astype(str)
        categorical_levels[col] = sorted(values.unique().tolist())

    feature_names = list(NUMERIC_COLUMNS)
    for col in CATEGORICAL_COLUMNS:
        feature_names.extend([f"{col}={level}" for level in categorical_levels[col]])

    return {
        "numeric_columns": list(NUMERIC_COLUMNS),
        "categorical_columns": list(CATEGORICAL_COLUMNS),
        "categorical_levels": categorical_levels,
        "feature_names": feature_names,
        "skipped_columns": [
            "subject",
            "choices_only_excess",
            "reward_schedule",
            "correct",
        ],
        "skip_reasons": {
            "subject": "degenerate alias for category in current adapter",
            "choices_only_excess": "panel-level artifact field, not adapter-local",
            "reward_schedule": "per-run constant stored in metadata",
            "correct": "gold-derived label, excluded from deployable features",
        },
    }


def _encode_dataframe(df: pd.DataFrame, feature_schema: dict[str, Any]) -> np.ndarray:
    missing = set(NUMERIC_COLUMNS + CATEGORICAL_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"adapter dataframe missing feature columns: {sorted(missing)}")

    parts: list[np.ndarray] = []
    numeric = (
        df[feature_schema["numeric_columns"]]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
    )
    parts.append(numeric)

    levels_by_col = feature_schema["categorical_levels"]
    for col in feature_schema["categorical_columns"]:
        values = df[col].fillna("unknown").astype(str).to_numpy()
        levels = list(levels_by_col[col])
        one_hot = np.zeros((len(df), len(levels)), dtype=np.float32)
        index = {level: i for i, level in enumerate(levels)}
        for row_idx, value in enumerate(values):
            level_idx = index.get(value)
            if level_idx is not None:
                one_hot[row_idx, level_idx] = 1.0
        parts.append(one_hot)

    if not parts:
        raise ValueError("no feature columns configured")
    return np.concatenate(parts, axis=1).astype(np.float32)


def _fit_scaler(matrix: np.ndarray, feature_names: list[str]) -> dict[str, Any]:
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return {
        "type": "standard",
        "feature_names": list(feature_names),
        "mean": mean.astype(float).tolist(),
        "std": std.astype(float).tolist(),
    }


def _scale_matrix(matrix: np.ndarray, scaler: dict[str, Any]) -> np.ndarray:
    mean = np.asarray(scaler["mean"], dtype=np.float32)
    std = np.asarray(scaler["std"], dtype=np.float32)
    if matrix.shape[1] != mean.shape[0] or mean.shape != std.shape:
        raise ValueError("scaler dimensions do not match encoded features")
    return ((matrix - mean) / std).astype(np.float32)


def dataframe_to_trajectories(
    df: pd.DataFrame,
    feature_schema: dict[str, Any] | None = None,
    scaler: dict[str, Any] | None = None,
    fit: bool = False,
) -> TrajectoryBundle:
    """Convert adapter rows to scaled trajectory objects.

    Parameters
    ----------
    df:
        Adapter dataframe from ``build_dataframe``.
    feature_schema:
        Existing schema for eval/validation transforms. Required unless
        ``fit=True``.
    scaler:
        Existing scaler for eval/validation transforms. Required unless
        ``fit=True``.
    fit:
        Fit categorical levels, feature order, train bounding box, and scaler.

    Returns
    -------
    TrajectoryBundle
        Sorted trajectories and reusable feature preprocessing metadata.
    """
    if df.empty:
        raise ValueError("cannot build trajectories from an empty dataframe")
    if fit:
        _assert_no_test_rows(df, context="fit")
    if fit:
        feature_schema = _fit_feature_schema(df)
    if feature_schema is None:
        raise ValueError("feature_schema is required when fit=False")

    encoded = _encode_dataframe(df, feature_schema)
    if fit:
        mins = encoded.min(axis=0)
        maxs = encoded.max(axis=0)
        feature_schema = dict(feature_schema)
        feature_schema["train_bounds"] = {
            name: {"min": float(lo), "max": float(hi)}
            for name, lo, hi in zip(feature_schema["feature_names"], mins, maxs)
        }
        scaler = _fit_scaler(encoded, feature_schema["feature_names"])
    if scaler is None:
        raise ValueError("scaler is required when fit=False")

    scaled = _scale_matrix(encoded, scaler)
    working = df.reset_index(drop=True).copy()
    working["_feature_row"] = list(scaled)

    trajectories: list[StopDFFTrajectory] = []
    for (item_id, fmt), group in working.groupby(["item_id", "format"], sort=True):
        group = group.sort_values("prefix_idx")
        features = np.stack(group["_feature_row"].to_list()).astype(np.float32)
        trajectories.append(
            StopDFFTrajectory(
                item_id=str(item_id),
                fmt=str(fmt),
                features=features,
                p_calibrated=[
                    float(x) for x in group["p_calibrated"].to_list()
                ],
                prefix_fractions=[
                    float(x) for x in group["prefix_fraction"].to_list()
                ],
                prefix_indices=[int(x) for x in group["prefix_idx"].to_list()],
            )
        )

    return TrajectoryBundle(
        trajectories=trajectories,
        feature_schema=feature_schema,
        scaler=scaler,
    )


def build_fvi_targets(
    trajectories: Iterable[StopDFFTrajectory],
    schedule: RewardSchedule,
    predict_fn: Callable[[torch.Tensor], torch.Tensor | float],
) -> FVIResult:
    """Build FVI targets by solving each trajectory with model bootstraps.

    The continuation function for prefix ``t`` is the current model's value
    prediction for the next prefix ``t + 1`` of the same trajectory. The DP
    solver supplies the terminal target ``max(A_T(p_T), 0)`` itself.
    """
    feature_rows: list[np.ndarray] = []
    target_values: list[float] = []
    traces: list[dict[str, Any]] = []
    target_min, target_max = _target_bounds(schedule)

    for traj in trajectories:
        if len(traj.p_calibrated) != len(traj.prefix_fractions):
            raise ValueError(f"trajectory {traj.item_id}/{traj.fmt} has misaligned p/fractions")
        if len(traj.features) != len(traj.p_calibrated):
            raise ValueError(f"trajectory {traj.item_id}/{traj.fmt} has misaligned features")

        def _continuation(t: int, **_: Any) -> float:
            next_idx = t + 1
            if next_idx >= len(traj.features):
                return 0.0
            next_features = torch.as_tensor(
                traj.features[next_idx], dtype=torch.float32
            ).unsqueeze(0)
            pred = predict_fn(next_features)
            if isinstance(pred, torch.Tensor):
                value = float(pred.detach().reshape(-1)[0].cpu().item())
            else:
                value = float(pred)
            if not np.isfinite(value):
                raise ValueError(
                    f"non-finite continuation prediction for "
                    f"{traj.item_id}/{traj.fmt} at prefix {next_idx}: {value}"
                )
            return float(np.clip(value, target_min, target_max))

        trace = solve_trajectory(
            p_trajectory=list(traj.p_calibrated),
            prefix_fractions=list(traj.prefix_fractions),
            schedule=schedule,
            continuation_fn=_continuation,
            item_id=traj.item_id,
            fmt=traj.fmt,
        )

        feature_rows.extend([np.asarray(row, dtype=np.float32) for row in traj.features])
        target_values.extend(float(v) for v in trace.values)
        traces.append(
            {
                "item_id": traj.item_id,
                "format": traj.fmt,
                "prefix_indices": list(traj.prefix_indices),
                "stop_step": int(trace.stop_step),
                "values": [float(v) for v in trace.values],
                "answer_utilities": [float(v) for v in trace.answer_utilities],
                "continuation_values": [float(v) for v in trace.continuation_values],
            }
        )

    if not feature_rows:
        raise ValueError("no FVI target rows were produced")
    features = torch.as_tensor(np.stack(feature_rows), dtype=torch.float32)
    targets = torch.as_tensor(target_values, dtype=torch.float32)
    if not torch.isfinite(targets).all():
        raise ValueError("FVI target construction produced non-finite targets")
    dataset = ValueTargetDataset(features, targets)
    stats = {
        "min": float(targets.min().item()),
        "max": float(targets.max().item()),
        "mean": float(targets.mean().item()),
    }
    if stats["min"] < target_min - 1e-6 or stats["max"] > target_max + 1e-6:
        raise ValueError(
            "FVI target construction produced values outside reward bounds "
            f"[{target_min}, {target_max}]: {stats}"
        )
    return FVIResult(
        features=features,
        targets=targets,
        dataset=dataset,
        traces=traces,
        target_stats=stats,
    )


def _model_predict_fn(model: StopDFFValueModel, device: torch.device) -> Callable[[torch.Tensor], torch.Tensor]:
    def _predict(features: torch.Tensor) -> torch.Tensor:
        features = features.to(device)
        return model(features).detach().cpu()

    return _predict


def _train_one_epoch(
    *,
    model: StopDFFValueModel,
    dataset: ValueTargetDataset,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    batch_size: int,
    grad_accum_steps: int,
    max_grad_norm: float,
    amp_enabled: bool,
    epoch_seed: int,
) -> float:
    model.train()
    generator = torch.Generator()
    generator.manual_seed(epoch_seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    criterion = nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler() if amp_enabled else None
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    total_examples = 0
    num_batches = len(loader)

    for batch_idx, (features, targets) in enumerate(loader):
        features = features.to(device)
        targets = targets.to(device)
        ctx = torch.cuda.amp.autocast() if amp_enabled else nullcontext()
        with ctx:
            preds = model(features)
            loss = criterion(preds, targets)
            scaled_loss = loss / max(1, grad_accum_steps)

        if amp_enabled:
            assert scaler is not None
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        total_loss += float(loss.detach().cpu().item()) * int(targets.shape[0])
        total_examples += int(targets.shape[0])

        should_step = (batch_idx + 1) % max(1, grad_accum_steps) == 0
        is_last = batch_idx + 1 == num_batches
        if should_step or is_last:
            if amp_enabled:
                assert scaler is not None
                scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), max_grad_norm)
            if amp_enabled:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

    return total_loss / max(1, total_examples)


def _evaluate_loss(
    model: StopDFFValueModel,
    dataset: ValueTargetDataset,
    device: torch.device,
    batch_size: int,
) -> float:
    model.eval()
    criterion = nn.MSELoss(reduction="sum")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    total_loss = 0.0
    total_examples = 0
    with torch.no_grad():
        for features, targets in loader:
            features = features.to(device)
            targets = targets.to(device)
            preds = model(features)
            total_loss += float(criterion(preds, targets).detach().cpu().item())
            total_examples += int(targets.shape[0])
    return total_loss / max(1, total_examples)


def _save_checkpoint(
    *,
    path: Path,
    model: StopDFFValueModel,
    config: dict[str, Any],
    scaler: dict[str, Any],
    feature_schema: dict[str, Any],
    train_losses: list[float],
    val_losses: list[float],
    calibration_path: str | None,
    seeds: list[int],
    reward_schedule: str,
    target_strategy: str,
    producer_provenance: dict[str, Any],
) -> None:
    save_commit = _checkpoint_save_identity(producer_provenance)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "config": dict(config),
        "scaler": scaler,
        "feature_schema": feature_schema,
        "train_losses": list(train_losses),
        "val_losses": list(val_losses),
        "calibration_path": calibration_path,
        "seeds": list(seeds),
        "git_sha": save_commit,
        "git_dirty": False,
        "reward_schedule": reward_schedule,
        "target_strategy": target_strategy,
        "producer_provenance": dict(producer_provenance),
    }
    torch.save(payload, path)


def _load_questions_for_training(
    *,
    data_dir: Path,
    train_split: str,
    val_split: str,
    smoke: bool,
    allow_incomplete_mc_coverage: bool,
    allow_low_mc_retention: bool,
) -> tuple[list[dict[str, Any]], set[str], set[str], dict[str, Any]]:
    train_path = data_dir / f"{train_split}_dataset.json"
    val_path = data_dir / f"{val_split}_dataset.json"
    for path in (train_path, val_path):
        if not path.exists():
            raise FileNotFoundError(f"missing dataset {path}")

    train_questions = iter_split_questions(load_json(train_path), source_path=train_path)
    val_questions = iter_split_questions(load_json(val_path), source_path=val_path)
    train_qids = {str(q["qid"]) for q in train_questions}
    val_qids = {str(q["qid"]) for q in val_questions}
    validate_qid_separation(
        fit_qids=train_qids,
        eval_qids=val_qids,
        fit_split=train_split,
        eval_split=val_split,
    )

    test_path = data_dir / "test_dataset.json"
    if test_path.exists():
        test_questions = iter_split_questions(load_json(test_path), source_path=test_path)
        test_qids = {str(q["qid"]) for q in test_questions}
        validate_qid_separation(
            fit_qids=train_qids,
            eval_qids=test_qids,
            fit_split=train_split,
            eval_split="test",
        )
        validate_qid_separation(
            fit_qids=val_qids,
            eval_qids=test_qids,
            fit_split=val_split,
            eval_split="test",
        )

    mc_questions = list(train_questions) + list(val_questions)
    if smoke:
        keep_train = set(sorted(train_qids)[:30])
        keep_val = set(sorted(val_qids)[:30])
        train_qids = keep_train
        val_qids = keep_val
        keep = train_qids | val_qids
        mc_questions = [q for q in mc_questions if str(q["qid"]) in keep]

    coverage_meta: dict[str, Any] = {}
    for split_name, qids in ((train_split, train_qids), (val_split, val_qids)):
        _, coverage = filter_mc_questions_to_split(mc_questions, qids)
        coverage_meta[split_name] = {
            "target_qids": coverage["target_qids"],
            "matched_qids": coverage["matched_qids"],
            "coverage_rate": coverage["coverage_rate"],
        }
        if (
            coverage["coverage_rate"] < MIN_MC_COVERAGE
            and not allow_incomplete_mc_coverage
        ):
            raise RuntimeError(
                f"MC {split_name} coverage is {coverage['coverage_rate']:.1%} "
                f"(threshold: {MIN_MC_COVERAGE:.1%}); pass "
                "--allow-incomplete-mc-coverage to override."
            )

    try:
        build_metadata = load_mc_build_metadata(data_dir)
    except RuntimeError:
        raise
    for split_name in (train_split, val_split):
        retention = build_retention_metadata(
            build_metadata,
            split=split_name,
            smoke=smoke,
            explicit_threshold=None,
            override=allow_low_mc_retention,
        )
        coverage_meta[split_name]["retention"] = retention

    return mc_questions, train_qids, val_qids, coverage_meta


def build_train_val_data(args: argparse.Namespace) -> tuple[TrajectoryBundle, TrajectoryBundle, dict[str, Any]]:
    """Load train/val datasets and convert them to trajectory bundles."""
    validate_split_separation(fit_split=args.train_split, eval_split=args.val_split)
    if str(args.val_split).strip().lower() == "test":
        raise ValueError(
            "val_split must not be test; test data is reserved for final "
            "learned-value StopDFF evaluation."
        )
    data_dir = _resolve_path(args.data_dir)
    calibration_path = _resolve_train_fit_calibration(args)

    mc_questions, train_qids, val_qids, coverage_meta = _load_questions_for_training(
        data_dir=data_dir,
        train_split=args.train_split,
        val_split=args.val_split,
        smoke=args.smoke,
        allow_incomplete_mc_coverage=args.allow_incomplete_mc_coverage,
        allow_low_mc_retention=args.allow_low_mc_retention,
    )

    train_df = build_dataframe(
        mc_questions=mc_questions,
        target_qids=train_qids,
        split_name=args.train_split,
        calibration_path=calibration_path,
        identity_calibration=args.effective_identity_calibration,
    )
    val_df = build_dataframe(
        mc_questions=mc_questions,
        target_qids=val_qids,
        split_name=args.val_split,
        calibration_path=calibration_path,
        identity_calibration=args.effective_identity_calibration,
    )
    _assert_no_test_rows(train_df, context="training")
    _assert_no_test_rows(val_df, context="validation")

    train_bundle = dataframe_to_trajectories(train_df, fit=True)
    val_bundle = dataframe_to_trajectories(
        val_df,
        feature_schema=train_bundle.feature_schema,
        scaler=train_bundle.scaler,
        fit=False,
    )
    metadata = {
        "data_dir": str(data_dir),
        "calibration_path": None if calibration_path is None else str(calibration_path),
        "train_qids": len(train_qids),
        "val_qids": len(val_qids),
        "coverage": coverage_meta,
        "identity_calibration": bool(args.effective_identity_calibration),
    }
    return train_bundle, val_bundle, metadata


class StopDFFValueTrainer:
    """Raw PyTorch trainer for StopDFFValueModel."""

    def __init__(
        self,
        *,
        train_bundle: TrajectoryBundle,
        val_bundle: TrajectoryBundle,
        schedule: RewardSchedule,
        config: dict[str, Any],
        seed: int,
        out_dir: Path,
        all_seeds: list[int],
    ) -> None:
        set_all_seeds(seed)
        self.train_bundle = train_bundle
        self.val_bundle = val_bundle
        self.schedule = schedule
        self.config = dict(config)
        self.seed = int(seed)
        self.out_dir = out_dir
        self.all_seeds = list(all_seeds)
        requested_device = str(config["device"])
        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            print(
                f"WARNING: requested device {requested_device!r} but CUDA is unavailable; using CPU.",
                file=sys.stderr,
            )
            requested_device = "cpu"
        self.device = torch.device(requested_device)
        self.amp_enabled = bool(config["amp"] and self.device.type == "cuda")

        input_dim = len(train_bundle.feature_schema["feature_names"])
        model_config = {
            "input_dim": input_dim,
            "hidden_sizes": list(config["hidden_sizes"]),
            "dropout": float(config["dropout"]),
            "feature_schema": train_bundle.feature_schema,
        }
        self.model = StopDFFValueModel.from_config(model_config).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(config["lr"]),
            weight_decay=float(config["weight_decay"]),
        )
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []
        self.history: dict[str, Any] = {
            "seed": self.seed,
            "train": [],
            "val": [],
            "epochs": [],
            "config": self.config,
            "feature_schema": train_bundle.feature_schema,
            "scaler": train_bundle.scaler,
        }

    def _targets(self, trajectories: list[StopDFFTrajectory]) -> FVIResult:
        self.model.eval()
        with torch.no_grad():
            return build_fvi_targets(
                trajectories,
                self.schedule,
                _model_predict_fn(self.model, self.device),
            )

    def train(self) -> dict[str, Any]:
        epochs = int(self.config["epochs"])
        patience = int(self.config["patience"])
        batch_size = int(self.config["batch_size"])
        grad_accum_steps = int(self.config["grad_accum_steps"])
        max_grad_norm = float(self.config["max_grad_norm"])
        best_val = float("inf")
        stale_epochs = 0
        best_epoch = 0
        best_ckpt = self.out_dir / "best_model" / "best.ckpt"

        print(f"[seed {self.seed}] training {epochs} epochs on {self.device}")
        for epoch in range(epochs):
            set_all_seeds(self.seed + epoch)
            train_targets = self._targets(self.train_bundle.trajectories)
            # Build validation targets before the optimizer update. This makes
            # the selection metric a one-step lagged Bellman loss instead of
            # evaluating the model against targets regenerated from itself.
            val_targets = self._targets(self.val_bundle.trajectories)
            train_loss = _train_one_epoch(
                model=self.model,
                dataset=train_targets.dataset,
                optimizer=self.optimizer,
                device=self.device,
                batch_size=batch_size,
                grad_accum_steps=grad_accum_steps,
                max_grad_norm=max_grad_norm,
                amp_enabled=self.amp_enabled,
                epoch_seed=self.seed + epoch,
            )

            val_loss = _evaluate_loss(
                self.model,
                val_targets.dataset,
                self.device,
                batch_size,
            )
            self.train_losses.append(float(train_loss))
            self.val_losses.append(float(val_loss))
            row = {
                "epoch": epoch + 1,
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "train_targets": len(train_targets.dataset),
                "val_targets": len(val_targets.dataset),
                "train_target_stats": train_targets.target_stats,
                "val_target_stats": val_targets.target_stats,
            }
            self.history["epochs"].append(row)
            self.history["train"].append(
                {
                    "epoch": epoch + 1,
                    "loss": float(train_loss),
                    "target_stats": train_targets.target_stats,
                }
            )
            self.history["val"].append(
                {
                    "epoch": epoch + 1,
                    "loss": float(val_loss),
                    "target_stats": val_targets.target_stats,
                    "target_policy": "one_step_lagged_fvi",
                }
            )
            print(
                f"[seed {self.seed}] epoch {epoch + 1}/{epochs} "
                f"train_loss={train_loss:.6f} val_loss={val_loss:.6f}"
            )

            if val_loss < best_val - 1e-12:
                best_val = float(val_loss)
                best_epoch = epoch + 1
                stale_epochs = 0
                _save_checkpoint(
                    path=best_ckpt,
                    model=self.model,
                    config=self.model.to_config() | {
                        "training": self.config,
                        "seed": self.seed,
                    },
                    scaler=self.train_bundle.scaler,
                    feature_schema=self.train_bundle.feature_schema,
                    train_losses=self.train_losses,
                    val_losses=self.val_losses,
                    calibration_path=self.config.get("calibration_path"),
                    seeds=self.all_seeds,
                    reward_schedule=self.schedule.name,
                    target_strategy=TARGET_STRATEGY,
                    producer_provenance=self.config["producer_provenance"],
                )
            else:
                stale_epochs += 1
                if patience >= 0 and stale_epochs >= patience:
                    print(
                        f"[seed {self.seed}] early stopping after {stale_epochs} "
                        f"epoch(s) without val improvement"
                    )
                    break

        self.history["summary"] = {
            "best_val_loss": best_val,
            "best_epoch": best_epoch,
            "epochs_ran": len(self.history["epochs"]),
            "best_checkpoint": str(best_ckpt),
        }
        _write_json(self.out_dir / "history.json", self.history)
        return self.history


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", default=str(DEFAULT_ARTIFACT_DIR))
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument(
        "--calibration",
        default=None,
        help=(
            "Train-fit calibration JSON. Defaults to "
            "<artifact-dir>/calibration_train.json."
        ),
    )
    parser.add_argument("--reward-schedule", default="power_mark")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seeds", type=_parse_csv_ints, default=[1, 2, 3])
    parser.add_argument("--hidden", type=_parse_hidden, default=(128, 128))
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true", default=False)
    parser.add_argument("--identity-calibration", action="store_true")
    parser.add_argument("--allow-incomplete-mc-coverage", action="store_true")
    parser.add_argument("--allow-low-mc-retention", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    if args.epochs <= 0:
        raise ValueError("--epochs must be > 0")
    if args.patience < 0:
        raise ValueError("--patience must be >= 0")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if args.grad_accum_steps <= 0:
        raise ValueError("--grad-accum-steps must be > 0")
    if args.max_grad_norm <= 0:
        raise ValueError("--max-grad-norm must be > 0")
    if args.lr <= 0:
        raise ValueError("--lr must be > 0")
    if args.weight_decay < 0:
        raise ValueError("--weight-decay must be >= 0")
    if args.dropout < 0.0 or args.dropout >= 1.0:
        raise ValueError("--dropout must be in [0.0, 1.0)")


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    """Run the full multi-seed training job."""
    args.effective_identity_calibration = bool(args.identity_calibration or args.smoke)
    _validate_args(args)
    producer_provenance = _training_provenance()
    schedule = get_schedule(args.reward_schedule)
    out_dir = _resolve_path(args.out)
    artifact_dir = _resolve_path(args.artifact_dir)
    train_bundle, val_bundle, data_metadata = build_train_val_data(args)

    common_config = {
        "artifact_dir": str(artifact_dir),
        "data_dir": str(_resolve_path(args.data_dir)),
        "train_split": args.train_split,
        "val_split": args.val_split,
        "device": args.device,
        "epochs": int(args.epochs),
        "patience": int(args.patience),
        "hidden_sizes": list(args.hidden),
        "dropout": float(args.dropout),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "grad_accum_steps": int(args.grad_accum_steps),
        "max_grad_norm": float(args.max_grad_norm),
        "amp": bool(args.amp),
        "smoke": bool(args.smoke),
        "identity_calibration": bool(args.effective_identity_calibration),
        "calibration_path": data_metadata["calibration_path"],
        "reward_schedule": args.reward_schedule,
        "target_strategy": TARGET_STRATEGY,
        "producer_provenance": producer_provenance,
    }

    seed_histories: list[dict[str, Any]] = []
    for seed in args.seeds:
        seed_out = out_dir / f"seed_{seed}"
        trainer = StopDFFValueTrainer(
            train_bundle=train_bundle,
            val_bundle=val_bundle,
            schedule=schedule,
            config=common_config,
            seed=int(seed),
            out_dir=seed_out,
            all_seeds=list(args.seeds),
        )
        seed_histories.append(trainer.train())

    best_vals = [
        float(history["summary"]["best_val_loss"])
        for history in seed_histories
        if np.isfinite(float(history["summary"]["best_val_loss"]))
    ]
    aggregate = {
        "status": "DONE",
        "seeds": list(args.seeds),
        "config": common_config,
        "data": data_metadata,
        "feature_schema": train_bundle.feature_schema,
        "scaler": train_bundle.scaler,
        "seed_histories": seed_histories,
        "summary": {
            "n_seeds": len(seed_histories),
            "best_val_loss_mean": float(np.mean(best_vals)) if best_vals else None,
            "best_val_loss_std": float(np.std(best_vals)) if best_vals else None,
            "best_val_loss_min": float(np.min(best_vals)) if best_vals else None,
        },
        "git_sha": _git_sha(),
        "git_dirty": _git_dirty(),
        "reward_schedule": args.reward_schedule,
        "target_strategy": TARGET_STRATEGY,
        "producer_provenance": producer_provenance,
    }
    _write_json(out_dir / "history.json", aggregate)
    return aggregate


def main(argv: list[str] | None = None) -> int:
    try:
        args = _parse_args(argv)
        result = run_training(args)
    except Exception as exc:  # noqa: BLE001 - CLI should report the blocking reason.
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(
        f"[STOPDFF-VALUE] status={result['status']} "
        f"n_seeds={result['summary']['n_seeds']} out={_resolve_path(args.out)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
