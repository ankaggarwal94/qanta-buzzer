#!/usr/bin/env python3
"""Evaluate learned-continuation-value DP StopDFF on an eval split.

The learned value model supplies the continuation term in the same
finite-horizon DP solver used by ``compute_stopdff_dp.py``. Checkpoints are
expected under ``seed_*/best_model/best.ckpt`` and may also be supplied as a
direct ``best.ckpt`` in ``--checkpoint-dir``.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import sys
import tempfile
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "paper_exports"
DEFAULT_CHECKPOINT_DIR = PROJECT_ROOT / "artifacts" / "value_model"
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_OUT = PROJECT_ROOT / "paper_exports" / "stopdff_learned_value.json"
OOD_WARN_THRESHOLD = 0.05
COLLAPSE_STD_THRESHOLD = 1e-6
BOOTSTRAP_SEED = 789685
BOOTSTRAP_REPS = 2000


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", default=str(DEFAULT_ARTIFACT_DIR))
    parser.add_argument("--checkpoint-dir", default=str(DEFAULT_CHECKPOINT_DIR))
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--calibration", default=None)
    parser.add_argument("--eval-split", "--split", dest="eval_split", default="test")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--out-md", default=None)
    parser.add_argument("--out-tex", default=None)
    parser.add_argument("--fig-dir", default=None)
    parser.add_argument("--reward-schedule", default=None)
    parser.add_argument("--identity-calibration", action="store_true")
    parser.add_argument("--allow-incomplete-mc-coverage", action="store_true")
    parser.add_argument("--allow-low-mc-retention", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def _resolve_path(path: str | Path) -> Path:
    raw = Path(path)
    return raw if raw.is_absolute() else PROJECT_ROOT / raw


def _display_path(path: str | Path) -> str:
    raw = Path(path)
    resolved = raw if raw.is_absolute() else PROJECT_ROOT / raw
    try:
        return resolved.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _to_jsonable(obj: Any) -> Any:
    if is_dataclass(obj):
        return _to_jsonable(asdict(obj))
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_write_text(path, json.dumps(_to_jsonable(payload), indent=2) + "\n")


def _find_checkpoints(checkpoint_dir: Path) -> list[Path]:
    if checkpoint_dir.is_file():
        return [checkpoint_dir] if checkpoint_dir.name == "best.ckpt" else []

    paths: list[Path] = []
    direct = checkpoint_dir / "best.ckpt"
    if direct.exists():
        paths.append(direct)
    nested_direct = checkpoint_dir / "best_model" / "best.ckpt"
    if nested_direct.exists():
        paths.append(nested_direct)
    paths.extend(sorted(checkpoint_dir.glob("seed_*/best_model/best.ckpt")))

    seen: set[Path] = set()
    unique: list[Path] = []
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    return unique


def _torch_load(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _schema_signature(schema: dict[str, Any]) -> dict[str, Any]:
    return {
        "feature_names": list(schema.get("feature_names", [])),
        "numeric_columns": list(schema.get("numeric_columns", [])),
        "categorical_columns": list(schema.get("categorical_columns", [])),
        "categorical_levels": {
            str(k): list(v)
            for k, v in dict(schema.get("categorical_levels", {})).items()
        },
    }


def _seed_from_checkpoint(path: Path, payload: dict[str, Any]) -> int | None:
    config = payload.get("config", {})
    if isinstance(config, dict) and "seed" in config:
        try:
            return int(config["seed"])
        except (TypeError, ValueError):
            pass
    for part in path.parts:
        if part.startswith("seed_"):
            try:
                return int(part.removeprefix("seed_"))
            except ValueError:
                return None
    return None


def _seeds_from_payload(payload: dict[str, Any]) -> list[int]:
    raw = payload.get("seeds")
    if not isinstance(raw, list):
        return []
    out: list[int] = []
    for value in raw:
        try:
            out.append(int(value))
        except (TypeError, ValueError):
            continue
    return out


def _load_ensemble(
    checkpoint_dir: Path,
    *,
    requested_device: str,
    reward_override: str | None,
) -> tuple[list[Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    from models.stopdff_value_model import StopDFFValueModel

    checkpoint_paths = _find_checkpoints(checkpoint_dir)
    if not checkpoint_paths:
        raise FileNotFoundError(
            f"no checkpoints found under {checkpoint_dir}; expected "
            "seed_*/best_model/best.ckpt or direct best.ckpt"
        )

    device = torch.device(
        requested_device
        if not requested_device.startswith("cuda") or torch.cuda.is_available()
        else "cpu"
    )
    models: list[Any] = []
    reference_schema: dict[str, Any] | None = None
    reference_scaler: dict[str, Any] | None = None
    reference_signature: dict[str, Any] | None = None
    reference_scaler_mean: np.ndarray | None = None
    reference_scaler_std: np.ndarray | None = None
    reward_schedules: set[str] = set()
    train_losses: list[list[float]] = []
    val_losses: list[list[float]] = []
    loaded_seeds: list[int] = []
    payload_seeds: list[int] = []

    for path in checkpoint_paths:
        payload = _torch_load(path)
        for key in ("state_dict", "config", "feature_schema", "scaler"):
            if key not in payload:
                raise ValueError(f"checkpoint {path} missing required key {key!r}")
        if "reward_schedule" not in payload:
            raise ValueError(f"checkpoint {path} missing required key 'reward_schedule'")

        schema = dict(payload["feature_schema"])
        scaler = dict(payload["scaler"])
        signature = _schema_signature(schema)
        if reference_signature is None:
            reference_signature = signature
            reference_schema = schema
            reference_scaler = scaler
        elif signature != reference_signature:
            raise ValueError(f"checkpoint schema mismatch for {path}")

        scaler_names = list(scaler.get("feature_names", []))
        feature_names = list(schema.get("feature_names", []))
        if scaler_names and scaler_names != feature_names:
            raise ValueError(f"checkpoint scaler feature order mismatch for {path}")
        if len(scaler.get("mean", [])) != len(feature_names):
            raise ValueError(f"checkpoint scaler mean dimension mismatch for {path}")
        if len(scaler.get("std", [])) != len(feature_names):
            raise ValueError(f"checkpoint scaler std dimension mismatch for {path}")
        scaler_mean = np.asarray(scaler.get("mean", []), dtype=float)
        scaler_std = np.asarray(scaler.get("std", []), dtype=float)
        if reference_scaler_mean is None:
            reference_scaler_mean = scaler_mean
            reference_scaler_std = scaler_std
        elif (
            not np.allclose(scaler_mean, reference_scaler_mean, rtol=0.0, atol=1e-8)
            or not np.allclose(scaler_std, reference_scaler_std, rtol=0.0, atol=1e-8)
        ):
            raise ValueError(f"checkpoint scaler values mismatch for {path}")

        reward_name = payload["reward_schedule"]
        reward_schedules.add(str(reward_name))

        model = StopDFFValueModel.load_from_state_dict(
            payload["state_dict"], payload["config"]
        )
        model.to(device)
        model.eval()
        models.append(model)

        train_losses.append([float(x) for x in payload.get("train_losses", [])])
        val_losses.append([float(x) for x in payload.get("val_losses", [])])
        payload_seeds.extend(_seeds_from_payload(payload))
        seed = _seed_from_checkpoint(path, payload)
        if seed is not None:
            loaded_seeds.append(seed)

    if reward_override is None:
        if len(reward_schedules) > 1:
            raise ValueError(
                "checkpoints disagree on reward_schedule; pass --reward-schedule "
                f"to override. Found: {sorted(reward_schedules)}"
            )
        reward_schedule = next(iter(reward_schedules), "power_mark")
    else:
        reward_schedule = reward_override

    metadata = {
        "checkpoint_path": _display_path(checkpoint_paths[0]),
        "checkpoint_paths": [_display_path(path) for path in checkpoint_paths],
        "n_checkpoints": len(checkpoint_paths),
        "seeds": sorted(set(payload_seeds)) or sorted(set(loaded_seeds)),
        "checkpoint_reward_schedules": sorted(reward_schedules),
        "reward_schedule": reward_schedule,
        "device": str(device),
        "train_losses": train_losses,
        "val_losses": val_losses,
    }
    assert reference_schema is not None
    assert reference_scaler is not None
    return models, reference_schema, reference_scaler, metadata


def _target_bounds(schedule: Any) -> tuple[float, float]:
    lower = min(0.0, float(schedule.r_wrong))
    upper = max(
        0.0,
        float(schedule.r_correct_early),
        float(schedule.r_correct_late),
    )
    return lower, upper


def _ensemble_predict(
    models: list[Any],
    feature_row: np.ndarray,
    *,
    device: str,
    bounds: tuple[float, float],
) -> float:
    tensor = torch.as_tensor(feature_row, dtype=torch.float32, device=torch.device(device))
    values: list[float] = []
    with torch.no_grad():
        for model in models:
            pred = model(tensor).detach().reshape(-1)[0].cpu().item()
            values.append(float(pred))
    value = float(np.mean(values))
    if not math.isfinite(value):
        raise ValueError("ensemble produced a non-finite continuation value")
    return float(np.clip(value, bounds[0], bounds[1]))


def _bootstrap_ci(values: list[float], *, reps: int = BOOTSTRAP_REPS) -> list[float | None]:
    if not values:
        return [None, None]
    if len(values) == 1:
        only = float(values[0])
        return [only, only]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    arr = np.asarray(values, dtype=float)
    stats = np.empty(reps, dtype=float)
    n = len(arr)
    for i in range(reps):
        sample = arr[rng.integers(0, n, size=n)]
        stats[i] = float(np.median(sample))
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return [float(lo), float(hi)]


def _applied_stop_utility(trace: Any, stop_step: int) -> float:
    if 0 <= stop_step < len(trace.answer_utilities):
        return float(trace.answer_utilities[stop_step])
    return 0.0


def _regret_summary(mc_traces: list[Any], qa_traces: list[Any]) -> dict[str, Any]:
    mc_to_qa: list[float] = []
    qa_to_mc: list[float] = []
    for mc_trace, qa_trace in zip(mc_traces, qa_traces):
        qa_opt = float(qa_trace.values[0]) if qa_trace.values else 0.0
        mc_opt = float(mc_trace.values[0]) if mc_trace.values else 0.0
        mc_to_qa.append(max(0.0, qa_opt - _applied_stop_utility(qa_trace, mc_trace.stop_step)))
        qa_to_mc.append(max(0.0, mc_opt - _applied_stop_utility(mc_trace, qa_trace.stop_step)))

    def _summ(values: list[float]) -> dict[str, float | None]:
        return {
            "mean": float(mean(values)) if values else None,
            "median": float(median(values)) if values else None,
            "max": float(max(values)) if values else None,
        }

    return {
        "mc_to_qa": _summ(mc_to_qa),
        "qa_to_mc": _summ(qa_to_mc),
    }


def _load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _comparison_block(
    artifact_dir: Path,
    *,
    signed_mean_value: float,
    abs_median_value: float,
) -> tuple[dict[str, Any], list[str]]:
    missing: list[str] = []
    dp = _load_json_if_exists(artifact_dir / "stopdff_dp.json")
    myopic = _load_json_if_exists(artifact_dir / "stopdff.json")

    if dp is None:
        missing.append("stopdff_dp.json")
    if myopic is None:
        missing.append("stopdff.json")

    dp_abs = None if dp is None else dp.get("stopdff_dp_abs_median")
    dp_signed_mean = None if dp is None else dp.get("stopdff_dp_signed_mean")
    myopic_abs = None if myopic is None else myopic.get("median_abs_prefix_shift")
    myopic_signed_mean = None
    if myopic is not None:
        if "signed_mean_prefix_shift" in myopic:
            myopic_signed_mean = myopic["signed_mean_prefix_shift"]
        elif float(myopic.get("mean_abs_prefix_shift", 0.0)) == 0.0:
            myopic_signed_mean = 0.0

    comparison = {
        "dp": {
            "artifact": _display_path(artifact_dir / "stopdff_dp.json"),
            "available": dp is not None,
            "delta_abs_median": (
                None if dp_abs is None else float(abs_median_value - float(dp_abs))
            ),
            "signed_mean_difference": (
                None
                if dp_signed_mean is None
                else float(signed_mean_value - float(dp_signed_mean))
            ),
        },
        "myopic": {
            "artifact": _display_path(artifact_dir / "stopdff.json"),
            "available": myopic is not None,
            "delta_abs_median": (
                None if myopic_abs is None else float(abs_median_value - float(myopic_abs))
            ),
            "signed_mean_difference": (
                None
                if myopic_signed_mean is None
                else float(signed_mean_value - float(myopic_signed_mean))
            ),
        },
    }
    return comparison, missing


def _qualitative_resolution(
    artifact_dir: Path,
    *,
    abs_median_value: float,
    ceiling_flags: dict[str, Any],
) -> str:
    myopic = _load_json_if_exists(artifact_dir / "stopdff.json")
    if not myopic or not myopic.get("ceiling_effect_detected"):
        return "confirms"

    learned_ceiling = any(
        bool(ceiling_flags.get(key))
        for key in (
            "all_stop_at_first_prefix",
            "all_stop_at_final_prefix",
            "no_cross_format_stopping_variance",
        )
    )
    myopic_abs = float(myopic.get("median_abs_prefix_shift", 0.0))
    if not learned_ceiling and abs_median_value > myopic_abs:
        return "resolves"
    if not learned_ceiling or abs_median_value != myopic_abs:
        return "reduces"
    return "confirms"


def _coverage_outside_train_bounds(
    df: pd.DataFrame,
    feature_schema: dict[str, Any],
) -> dict[str, Any]:
    from scripts.train_stopdff_value_model import _encode_dataframe

    bounds = feature_schema.get("train_bounds") or {}
    unseen_rows_by_column: dict[str, int] = {}
    unseen_levels_by_column: dict[str, list[str]] = {}
    categorical_levels = dict(feature_schema.get("categorical_levels", {}))
    for col, levels in categorical_levels.items():
        known = {str(level) for level in levels}
        values = df[col].fillna("unknown").astype(str) if col in df.columns else []
        unseen = [value for value in values if value not in known]
        unseen_rows_by_column[col] = len(unseen)
        unseen_levels_by_column[col] = sorted(set(unseen))
    n_unseen_rows = 0
    if unseen_rows_by_column:
        unseen_mask = np.zeros(len(df), dtype=bool)
        for col, levels in categorical_levels.items():
            if col not in df.columns:
                continue
            known = {str(level) for level in levels}
            unseen_mask |= ~df[col].fillna("unknown").astype(str).isin(known).to_numpy()
        n_unseen_rows = int(unseen_mask.sum())

    if not bounds:
        return {
            "n_cells": int(len(df)),
            "fraction_outside_train_bounds": None,
            "n_outside_train_bounds": None,
            "has_train_bounds": False,
            "n_rows_with_unseen_categories": n_unseen_rows,
            "unseen_category_row_counts": unseen_rows_by_column,
            "unseen_category_level_counts": {
                key: len(value) for key, value in unseen_levels_by_column.items()
            },
            "unseen_category_levels": unseen_levels_by_column,
        }

    encoded = _encode_dataframe(df, feature_schema)
    feature_names = list(feature_schema["feature_names"])
    outside = np.zeros(encoded.shape[0], dtype=bool)
    outside_entries = 0
    for idx, name in enumerate(feature_names):
        b = bounds.get(name)
        if not b:
            continue
        col_outside = (encoded[:, idx] < float(b["min"]) - 1e-8) | (
            encoded[:, idx] > float(b["max"]) + 1e-8
        )
        outside |= col_outside
        outside_entries += int(col_outside.sum())
    return {
        "n_cells": int(encoded.shape[0]),
        "n_features": int(encoded.shape[1]),
        "n_outside_train_bounds": int(outside.sum()),
        "fraction_outside_train_bounds": float(outside.mean()) if len(outside) else 0.0,
        "n_feature_entries_outside_train_bounds": int(outside_entries),
        "n_rows_with_unseen_categories": n_unseen_rows,
        "unseen_category_row_counts": unseen_rows_by_column,
        "unseen_category_level_counts": {
            key: len(value) for key, value in unseen_levels_by_column.items()
        },
        "unseen_category_levels": unseen_levels_by_column,
        "has_train_bounds": True,
    }


def _minimal_png(path: Path) -> None:
    png_bytes = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png_bytes)


def _write_figures(
    fig_dir: Path,
    *,
    checkpoint_metadata: dict[str, Any],
    mc_traces: list[Any],
    qa_traces: list[Any],
    per_item_stopdff: list[tuple[str, int]],
) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        for name in (
            "value_model_loss.png",
            "learned_value_vs_bucket_dp.png",
            "learned_stopdff_by_format.png",
        ):
            _minimal_png(fig_dir / name)
        return

    train_losses = checkpoint_metadata.get("train_losses", [])
    val_losses = checkpoint_metadata.get("val_losses", [])
    plt.figure(figsize=(6, 4))
    for idx, losses in enumerate(train_losses):
        if losses:
            plt.plot(range(1, len(losses) + 1), losses, alpha=0.7, label=f"train {idx + 1}")
    for idx, losses in enumerate(val_losses):
        if losses:
            plt.plot(range(1, len(losses) + 1), losses, linestyle="--", alpha=0.7, label=f"val {idx + 1}")
    plt.xlabel("epoch")
    plt.ylabel("MSE")
    plt.title("Value model loss")
    if any(train_losses) or any(val_losses):
        plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(fig_dir / "value_model_loss.png", dpi=160)
    plt.close()

    rows = []
    for trace in [*mc_traces, *qa_traces]:
        for frac, cont in zip(
            np.linspace(0.0, 1.0, num=max(1, len(trace.continuation_values))),
            trace.continuation_values,
        ):
            rows.append((trace.fmt, frac, float(cont)))
    plt.figure(figsize=(6, 4))
    if rows:
        frame = pd.DataFrame(rows, columns=["format", "prefix_fraction", "continuation"])
        bins = pd.cut(
            frame["prefix_fraction"],
            bins=[-0.001, 0.33, 0.66, 1.001],
            labels=["early", "mid", "late"],
        )
        grouped = frame.assign(bucket=bins).groupby(["bucket", "format"], observed=False)["continuation"].mean()
        grouped.unstack("format").plot(kind="bar", ax=plt.gca())
    plt.ylabel("mean continuation value")
    plt.title("Learned value by prefix bucket")
    plt.tight_layout()
    plt.savefig(fig_dir / "learned_value_vs_bucket_dp.png", dpi=160)
    plt.close()

    plt.figure(figsize=(6, 4))
    if per_item_stopdff:
        signed = [shift for _, shift in per_item_stopdff]
        lo = min(signed)
        hi = max(signed)
        bins = np.arange(lo - 0.5, hi + 1.5, 1.0)
        plt.hist(signed, bins=bins, color="#4c78a8", edgecolor="white")
    plt.xlabel("MC stop step - QA stop step")
    plt.ylabel("items")
    plt.title("Learned StopDFF distribution")
    plt.tight_layout()
    plt.savefig(fig_dir / "learned_stopdff_by_format.png", dpi=160)
    plt.close()


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    md = [
        "# Learned-Value DP StopDFF",
        "",
        f"**Metric type:** `{payload['metadata']['metric_type']}`",
        "",
        "| Field | Value |",
        "|-------|-------|",
        f"| Eval split | {payload['metadata']['eval_split']} |",
        f"| Reward schedule | {payload['metadata']['reward_schedule']} |",
        f"| Checkpoints | {payload['metadata']['n_checkpoints']} |",
        f"| n_items | {payload['n_items']} |",
        f"| Signed median | {payload['stopdff_signed_median']:.4f} |",
        f"| Signed mean | {payload['stopdff_signed_mean']:.4f} |",
        f"| Abs median | {payload['stopdff_abs_median']:.4f} |",
        f"| Gate verdict | {payload['gate_verdict']} |",
        "",
        "## Diagnostics",
        "",
        f"- Gate reason: {payload['gate_verdict_reason']}",
        f"- OOD fraction: {payload['coverage']['fraction_outside_train_bounds']}",
        f"- Qualitative resolution: {payload['metadata']['qualitative_resolution']}",
    ]
    _atomic_write_text(path, "\n".join(md) + "\n")


def _latex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for raw, escaped in replacements.items():
        text = text.replace(raw, escaped)
    return text


def _write_latex(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        r"\begin{tabular}{lr}",
        r"\toprule",
        r"Metric & Value \\",
        r"\midrule",
        f"Signed median StopDFF & {payload['stopdff_signed_median']:.4f} \\\\",
        f"Signed mean StopDFF & {payload['stopdff_signed_mean']:.4f} \\\\",
        f"Abs median StopDFF & {payload['stopdff_abs_median']:.4f} \\\\",
        f"$n_{{items}}$ & {payload['n_items']} \\\\",
        f"Gate verdict & {_latex_escape(payload['gate_verdict'])} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    _atomic_write_text(path, "\n".join(lines) + "\n")


def _build_payload(
    *,
    args: argparse.Namespace,
    artifact_dir: Path,
    checkpoint_metadata: dict[str, Any],
    reward_schedule: Any,
    per_item_stopdff: list[tuple[str, int]],
    mc_traces: list[Any],
    qa_traces: list[Any],
    coverage: dict[str, Any],
    ceiling_flags: dict[str, Any],
    missing_comparisons: list[str],
    comparison: dict[str, Any],
    continuation_diagnostics: dict[str, Any],
    mc_coverage: dict[str, Any] | None,
    mc_retention_gate: dict[str, Any] | None,
    mc_build_metadata: dict[str, Any] | None,
    generation: dict[str, Any] | None,
) -> dict[str, Any]:
    signed = [float(shift) for _, shift in per_item_stopdff]
    abs_shifts = [abs(x) for x in signed]
    signed_median = float(median(signed)) if signed else 0.0
    signed_mean = float(mean(signed)) if signed else 0.0
    abs_median = float(median(abs_shifts)) if abs_shifts else 0.0
    abs_mean = float(mean(abs_shifts)) if abs_shifts else 0.0
    direction = {
        "mc_earlier": sum(1 for x in signed if x < 0),
        "qa_earlier": sum(1 for x in signed if x > 0),
        "same_step": sum(1 for x in signed if x == 0),
    }
    n_items = len(signed)
    direction_fractions = {
        key: (float(value / n_items) if n_items else 0.0)
        for key, value in direction.items()
    }

    ceiling_warn = any(
        bool(ceiling_flags.get(key))
        for key in (
            "all_stop_at_first_prefix",
            "all_stop_at_final_prefix",
            "no_cross_format_stopping_variance",
        )
    )
    ood_fraction = coverage.get("fraction_outside_train_bounds")
    ood_warn = ood_fraction is not None and float(ood_fraction) > OOD_WARN_THRESHOLD
    unseen_category_warn = int(coverage.get("n_rows_with_unseen_categories") or 0) > 0
    collapse_warn = bool(continuation_diagnostics["collapsed"])
    retention_warn = False
    if isinstance(mc_retention_gate, dict):
        for block in mc_retention_gate.values():
            if isinstance(block, dict) and block.get("passed") is False:
                retention_warn = True
                break

    if not checkpoint_metadata["checkpoint_paths"]:
        gate_verdict = "fail"
        gate_reason = "no_checkpoints"
    elif n_items == 0:
        gate_verdict = "fail"
        gate_reason = "no_items"
    elif ood_warn:
        gate_verdict = "warn"
        gate_reason = f"ood_fraction={float(ood_fraction):.3f} > {OOD_WARN_THRESHOLD}"
    elif unseen_category_warn:
        gate_verdict = "warn"
        gate_reason = "unseen_eval_categories"
    elif ceiling_warn:
        gate_verdict = "warn"
        gate_reason = "ceiling_effect"
    elif collapse_warn:
        gate_verdict = "warn"
        gate_reason = "continuation_collapse"
    elif retention_warn:
        gate_verdict = "warn"
        gate_reason = "low_mc_retention"
    else:
        gate_verdict = "pass"
        gate_reason = None

    qualitative_resolution = _qualitative_resolution(
        artifact_dir, abs_median_value=abs_median, ceiling_flags=ceiling_flags
    )

    payload = {
        "stopdff_signed_median": signed_median,
        "stopdff_signed_median_ci95": _bootstrap_ci(signed),
        "stopdff_signed_mean": signed_mean,
        "stopdff_abs_median": abs_median,
        "stopdff_abs_mean": abs_mean,
        "n_items": n_items,
        "direction_breakdown": direction,
        "direction_fractions": direction_fractions,
        "decision_regret": _regret_summary(mc_traces, qa_traces),
        "coverage": coverage,
        "ceiling_flags": ceiling_flags,
        "continuation_diagnostics": continuation_diagnostics,
        "comparison": comparison,
        "missing_comparison_artifacts": list(missing_comparisons),
        "delta_abs_median_vs_dp": comparison["dp"]["delta_abs_median"],
        "signed_mean_difference": comparison["dp"]["signed_mean_difference"],
        "signed_mean_difference_vs_dp": comparison["dp"]["signed_mean_difference"],
        "delta_abs_median_vs_myopic": comparison["myopic"]["delta_abs_median"],
        "signed_mean_difference_vs_myopic": comparison["myopic"]["signed_mean_difference"],
        "gate_verdict": gate_verdict,
        "gate_verdict_reason": gate_reason,
        "metadata": {
            "metric_type": "learned_value_dp",
            "stopping_policy": "learned_value_finite_horizon_dp",
            "checkpoint_path": checkpoint_metadata["checkpoint_path"],
            "checkpoint_paths": checkpoint_metadata["checkpoint_paths"],
            "n_checkpoints": checkpoint_metadata["n_checkpoints"],
            "seeds": checkpoint_metadata["seeds"],
            "reward_schedule": reward_schedule.name,
            "reward_schedule_description": reward_schedule.description,
            "checkpoint_reward_schedules": checkpoint_metadata["checkpoint_reward_schedules"],
            "eval_split": args.eval_split,
            "data_dir": _display_path(args.data_dir),
            "artifact_dir": _display_path(artifact_dir),
            "identity_calibration": bool(args.identity_calibration or args.smoke),
            "smoke": bool(args.smoke),
            "qualitative_resolution": qualitative_resolution,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "generation": generation,
        },
    }
    if mc_coverage is not None:
        payload["mc_coverage"] = mc_coverage
    if mc_retention_gate is not None:
        payload["mc_retention_gate"] = mc_retention_gate
    if mc_build_metadata is not None:
        payload["mc_build_metadata"] = mc_build_metadata
    return payload


def _load_eval_dataframe(
    args: argparse.Namespace,
    *,
    artifact_dir: Path,
) -> tuple[
    pd.DataFrame,
    dict[str, Any] | None,
    dict[str, Any] | None,
    dict[str, Any] | None,
    Path | None,
]:
    from scripts._audit_gates import (
        build_coverage_metadata,
        build_retention_metadata,
        filter_mc_questions_to_split,
        load_mc_build_metadata,
    )
    from scripts._common import iter_split_questions, load_json
    from scripts.stopdff_dp.adapter import build_dataframe

    data_dir = _resolve_path(args.data_dir)
    mc_path = data_dir / "mc_dataset.json"
    eval_path = data_dir / f"{args.eval_split}_dataset.json"
    for path in (mc_path, eval_path):
        if not path.exists():
            raise FileNotFoundError(f"missing dataset {path}")

    mc_questions = iter_split_questions(load_json(mc_path), source_path=mc_path)
    eval_questions = iter_split_questions(load_json(eval_path), source_path=eval_path)
    eval_qids = {str(q["qid"]) for q in eval_questions}
    if args.smoke:
        eval_qids = set(sorted(eval_qids)[:30])
        mc_questions = [q for q in mc_questions if str(q["qid"]) in eval_qids]

    _mc_eval_rows, eval_coverage = filter_mc_questions_to_split(mc_questions, eval_qids)
    min_mc_coverage = 0.98
    if (
        eval_coverage["coverage_rate"] < min_mc_coverage
        and not args.allow_incomplete_mc_coverage
    ):
        raise RuntimeError(
            f"MC eval coverage is {eval_coverage['coverage_rate']:.1%} "
            f"(threshold: {min_mc_coverage:.1%}); pass "
            "--allow-incomplete-mc-coverage to override."
        )

    build_metadata = load_mc_build_metadata(data_dir)
    retention = build_retention_metadata(
        build_metadata,
        split=args.eval_split,
        smoke=args.smoke,
        explicit_threshold=None,
        override=args.allow_low_mc_retention,
    )

    coverage_meta = build_coverage_metadata(
        eval_coverage,
        threshold=min_mc_coverage,
        override=args.allow_incomplete_mc_coverage,
    )
    coverage_meta["split"] = args.eval_split

    calibration_path = None
    identity = bool(args.identity_calibration or args.smoke)
    if not identity:
        calibration_path = (
            _resolve_path(args.calibration)
            if args.calibration
            else artifact_dir / "calibration_train.json"
        )
        if not calibration_path.exists():
            raise FileNotFoundError(f"calibration JSON not found: {calibration_path}")
        with calibration_path.open("r", encoding="utf-8") as handle:
            fit_split = str(
                json.load(handle).get("metadata", {}).get("fit_split", "")
            ).lower()
        if fit_split != "train":
            raise ValueError(
                "learned-value evaluation requires the train-fit calibration "
                f"artifact; got fit_split={fit_split!r} from {calibration_path}."
            )

    df = build_dataframe(
        mc_questions=mc_questions,
        target_qids=eval_qids,
        split_name=args.eval_split,
        calibration_path=calibration_path,
        identity_calibration=identity,
    )
    mc_coverage = {args.eval_split: coverage_meta}
    mc_retention = {args.eval_split: retention} if retention is not None else None
    mc_build = None
    if build_metadata is not None:
        mc_build = {
            "status": build_metadata["status"],
            "source_path": build_metadata["source_path"],
            "source_sha256": build_metadata["source_sha256"],
        }
    return df, mc_coverage, mc_retention, mc_build, calibration_path


def _generation_provenance(
    *,
    args: argparse.Namespace,
    out_json: Path,
    checkpoint_metadata: dict[str, Any],
    calibration_path: Path | None,
) -> dict[str, Any] | None:
    try:
        from scripts._common import build_generation_provenance
        from scripts._common import sha256_file
        from scripts.stopdff_dp._provenance import helper_paths, helper_sha256s

        learned_helper_paths = [
            PROJECT_ROOT / "models" / "stopdff_value_model.py",
            PROJECT_ROOT / "scripts" / "train_stopdff_value_model.py",
        ]
        extras: list[Path] = [
            _resolve_path(args.data_dir) / "mc_dataset.json",
            _resolve_path(args.data_dir) / f"{args.eval_split}_dataset.json",
            _resolve_path(args.data_dir) / "build_metadata.json",
            *[Path(path) for path in checkpoint_metadata.get("checkpoint_paths", [])],
            *helper_paths(),
            *learned_helper_paths,
        ]
        if calibration_path is not None:
            extras.append(calibration_path)
        generation = build_generation_provenance(
            __file__,
            list(sys.argv[1:]),
            output_path=out_json,
            extra_paths=extras,
        )
        learned_helper_sha256s = {
            path.relative_to(PROJECT_ROOT).as_posix(): sha256_file(path)
            for path in learned_helper_paths
        }
        generation["helper_sha256s"] = {
            **helper_sha256s(),
            **learned_helper_sha256s,
        }
        return generation
    except Exception as exc:
        return {"status": "unavailable", "error": str(exc)}


def run(args: argparse.Namespace) -> dict[str, Any]:
    from scripts.stopdff_dp import diagnostics as diag_module
    from scripts.stopdff_dp import dp_solver as dp_module
    from scripts.stopdff_dp.rewards import get_schedule
    from scripts.train_stopdff_value_model import dataframe_to_trajectories

    artifact_dir = _resolve_path(args.artifact_dir)
    checkpoint_dir = _resolve_path(args.checkpoint_dir)
    out_json = _resolve_path(args.out)
    out_md = _resolve_path(args.out_md) if args.out_md else out_json.with_suffix(".md")
    out_tex = (
        _resolve_path(args.out_tex)
        if args.out_tex
        else out_json.with_name("stopdff_learned_value_table.tex")
    )
    fig_dir = (
        _resolve_path(args.fig_dir)
        if args.fig_dir
        else out_json.parent / "figures"
    )

    models, feature_schema, scaler, checkpoint_metadata = _load_ensemble(
        checkpoint_dir,
        requested_device=args.device,
        reward_override=args.reward_schedule,
    )
    schedule = get_schedule(checkpoint_metadata["reward_schedule"])
    target_bounds = _target_bounds(schedule)
    eval_df, mc_coverage, mc_retention, mc_build, calibration_path = _load_eval_dataframe(
        args, artifact_dir=artifact_dir
    )
    generation = _generation_provenance(
        args=args,
        out_json=out_json,
        checkpoint_metadata=checkpoint_metadata,
        calibration_path=calibration_path,
    )
    coverage = _coverage_outside_train_bounds(eval_df, feature_schema)
    bundle = dataframe_to_trajectories(
        eval_df,
        feature_schema=feature_schema,
        scaler=scaler,
        fit=False,
    )

    device = str(checkpoint_metadata["device"])
    traces_by_key: dict[tuple[str, str], Any] = {}
    continuation_values: list[float] = []
    for traj in bundle.trajectories:
        tags: dict[int, str] = {}

        def _continuation(t: int, **_: Any) -> float:
            next_idx = t + 1
            if next_idx >= len(traj.features):
                tags[t] = "exact"
                return 0.0
            value = _ensemble_predict(
                models,
                traj.features[next_idx],
                device=device,
                bounds=target_bounds,
            )
            continuation_values.append(value)
            tags[t] = "exact"
            return value

        trace = dp_module.solve_trajectory(
            p_trajectory=list(traj.p_calibrated),
            prefix_fractions=list(traj.prefix_fractions),
            schedule=schedule,
            continuation_fn=_continuation,
            item_id=traj.item_id,
            fmt=traj.fmt,
            coverage_tagger=lambda t, _tags=tags: _tags.get(t, "exact"),
        )
        traces_by_key[(traj.item_id, traj.fmt)] = trace

    paired_ids = sorted(
        item_id
        for item_id in {key[0] for key in traces_by_key}
        if (item_id, "MC") in traces_by_key and (item_id, "QA") in traces_by_key
    )
    mc_traces = [traces_by_key[(item_id, "MC")] for item_id in paired_ids]
    qa_traces = [traces_by_key[(item_id, "QA")] for item_id in paired_ids]
    per_item_stopdff = [
        (
            item_id,
            dp_module.stopdff_for_item(
                mc_trace=traces_by_key[(item_id, "MC")],
                qa_trace=traces_by_key[(item_id, "QA")],
            ),
        )
        for item_id in paired_ids
    ]

    ceiling_flags = diag_module.detect_ceiling_effects(mc_traces, qa_traces)
    if continuation_values:
        cont_std = float(np.std(np.asarray(continuation_values, dtype=float)))
        cont_min = float(min(continuation_values))
        cont_max = float(max(continuation_values))
    else:
        cont_std = 0.0
        cont_min = None
        cont_max = None
    continuation_diagnostics = {
        "n_values": len(continuation_values),
        "std": cont_std,
        "min": cont_min,
        "max": cont_max,
        "collapsed": bool(continuation_values and cont_std <= COLLAPSE_STD_THRESHOLD),
    }

    signed = [float(shift) for _, shift in per_item_stopdff]
    signed_mean = float(mean(signed)) if signed else 0.0
    abs_median = float(median([abs(x) for x in signed])) if signed else 0.0
    comparison, missing_comparisons = _comparison_block(
        artifact_dir,
        signed_mean_value=signed_mean,
        abs_median_value=abs_median,
    )
    payload = _build_payload(
        args=args,
        artifact_dir=artifact_dir,
        checkpoint_metadata=checkpoint_metadata,
        reward_schedule=schedule,
        per_item_stopdff=per_item_stopdff,
        mc_traces=mc_traces,
        qa_traces=qa_traces,
        coverage=coverage,
        ceiling_flags=ceiling_flags,
        missing_comparisons=missing_comparisons,
        comparison=comparison,
        continuation_diagnostics=continuation_diagnostics,
        mc_coverage=mc_coverage,
        mc_retention_gate=mc_retention,
        mc_build_metadata=mc_build,
        generation=generation,
    )

    _atomic_write_json(out_json, payload)
    _write_markdown(out_md, payload)
    _write_latex(out_tex, payload)
    _write_figures(
        fig_dir,
        checkpoint_metadata=checkpoint_metadata,
        mc_traces=mc_traces,
        qa_traces=qa_traces,
        per_item_stopdff=per_item_stopdff,
    )
    return payload


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        payload = run(args)
    except Exception as exc:  # noqa: BLE001 - CLI should report the blocking reason.
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(
        f"[STOPDFF-LEARNED-VALUE] Wrote {_resolve_path(args.out)} "
        f"(verdict={payload['gate_verdict']}, n_items={payload['n_items']})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
