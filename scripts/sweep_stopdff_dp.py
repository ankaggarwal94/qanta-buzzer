#!/usr/bin/env python3
"""Run a resumable finite-horizon DP StopDFF sensitivity sweep."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import struct
import sys
import time
import traceback
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_CALIBRATION = PROJECT_ROOT / "paper_exports" / "calibration.json"
DEFAULT_OUT = PROJECT_ROOT / "paper_exports" / "stopdff_dp_sweep.json"
DEFAULT_REWARD_SCHEDULES = (
    "acf_flat",
    "power_mark",
    "wait_cost_small",
    "strict_wrong",
    "low_wrong_cost",
)
DEFAULT_CALIBRATORS = (
    "uncalibrated",
    "platt-logistic",
    "temperature",
    "isotonic",
)

FORMAT_CONDITIONS = (
    "QA-prefix",
    "MC-fixed",
    "MC-dynamic",
    "MC-full",
    "choices-only",
)


def _csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a resumable DP StopDFF sweep.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--artifact-dir", default=None)
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--calibration", default=str(DEFAULT_CALIBRATION))
    parser.add_argument("--fit-split", default="val")
    parser.add_argument("--eval-split", default="test")
    parser.add_argument(
        "--reward-schedules",
        default=",".join(DEFAULT_REWARD_SCHEDULES),
    )
    parser.add_argument(
        "--continuations",
        default="empirical_bucket,pooled_empirical,oracle_trajectory",
    )
    parser.add_argument(
        "--calibrators",
        default=",".join(DEFAULT_CALIBRATORS),
    )
    parser.add_argument(
        "--formats",
        default="QA-prefix,MC-fixed,MC-dynamic,MC-full,choices-only",
    )
    parser.add_argument("--prefix-bucketing", default="early_mid_late,exact_prefix")
    parser.add_argument("--subject-pooling", default="per_subject,pooled_subject")
    parser.add_argument("--max-wall-hours", type=float, default=None)
    parser.add_argument("--max-cells", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--only-missing", action="store_true")
    parser.add_argument("--num-bootstrap", type=int, default=500)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--allow-incomplete-mc-coverage",
        action="store_true",
        help=(
            "Override the MC coverage gate. Use only when downstream "
            "interpretation accounts for the non-random subset."
        ),
    )
    parser.add_argument(
        "--allow-low-mc-retention",
        action="store_true",
        help=(
            "Override the MC retention gate. Use only when reporting the "
            "sweep artifact explicitly as a retained-MC-subset metric."
        ),
    )
    parser.add_argument("--identity-calibration", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args(argv)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _to_jsonable(value: object) -> object:
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(_to_jsonable(payload), indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _git_metadata(args: argparse.Namespace, *, out: Path) -> tuple[str | None, bool | None]:
    """Return (git_commit, git_dirty) scoped to the sweep's relevant paths.

    The dirty check explicitly EXCLUDES the output paths (sweep JSON / MD / TeX,
    figures, cell cache directory) because the run is supposed to modify those.
    A dirty flag should reflect changes to the producer script or its INPUTS
    (mc_dataset, val/test splits, build_metadata, calibration) relative to HEAD.

    Mirrors scripts/_common.build_generation_provenance's scoped approach (see
    its comment about `git status -- <abs_path>` aborting when paths are outside
    the repo).
    """
    import subprocess

    from scripts.stopdff_dp._provenance import helper_paths

    data_dir = Path(args.data_dir).resolve()
    script_path = Path(__file__).resolve()
    calibration_path = (
        None if args.identity_calibration else Path(args.calibration).resolve()
    )

    # Pathspec must include EVERY file whose contents could change the
    # sweep's results — that's the producer script, the input datasets,
    # the calibration JSON, AND every helper module hashed by
    # helper_sha256s() (since they directly affect _cell_id via the
    # fingerprint). Keeping the dirty check and the hash set in sync via
    # helper_paths() prevents the bug where a dirty helper would be hashed
    # into the fingerprint but not flagged in git_dirty.
    candidate_paths = [
        script_path,
        data_dir / "mc_dataset.json",
        data_dir / f"{args.fit_split}_dataset.json",
        data_dir / f"{args.eval_split}_dataset.json",
        data_dir / "build_metadata.json",
    ]
    if calibration_path is not None:
        candidate_paths.append(calibration_path)
    candidate_paths.extend(helper_paths())

    repo_relative_paths: list[str] = []
    for p in candidate_paths:
        try:
            rel = p.resolve().relative_to(PROJECT_ROOT)
        except ValueError:
            # Path is outside the repo; skip rather than abort.
            continue
        repo_relative_paths.append(rel.as_posix())

    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:  # noqa: BLE001 - provenance is best-effort
        commit = None

    try:
        status = subprocess.check_output(
            ["git", "status", "--short", "--", *repo_relative_paths],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = bool(status)
    except Exception:  # noqa: BLE001
        dirty = None
    return commit, dirty


def _file_sha256(path: Path | None) -> str | None:
    if path is None or not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_fingerprint(
    args: argparse.Namespace,
    *,
    out: Path,
    git_commit: str | None,
    myopic_artifact_path: Path | None = None,
) -> dict:
    from scripts._common import project_relative
    from scripts.stopdff_dp._provenance import helper_sha256s

    data_dir = Path(args.data_dir).resolve()
    calibration_path = (
        None if args.identity_calibration else Path(args.calibration).resolve()
    )
    mc_path = data_dir / "mc_dataset.json"
    fit_split_path = data_dir / f"{args.fit_split}_dataset.json"
    eval_split_path = data_dir / f"{args.eval_split}_dataset.json"
    build_metadata_path = data_dir / "build_metadata.json"
    return {
        "schema_version": 5,
        "data_dir": project_relative(data_dir),
        "fit_split": args.fit_split,
        "eval_split": args.eval_split,
        "calibration_path": project_relative(calibration_path) if calibration_path else None,
        "calibration_sha256": _file_sha256(calibration_path),
        "mc_dataset_sha256": _file_sha256(mc_path),
        "fit_dataset_sha256": _file_sha256(fit_split_path),
        "eval_dataset_sha256": _file_sha256(eval_split_path),
        "build_metadata_sha256": _file_sha256(build_metadata_path),
        # PR #15 review (chatgpt-codex-connector 3314472776): each cell
        # serializes myopic_artifact_comparison from stopdff.json. Hash
        # that artifact so regenerating it in place invalidates the
        # cache and prevents --resume/--only-missing from republishing
        # stale comparison fields.
        "myopic_artifact_path": (
            project_relative(myopic_artifact_path)
            if myopic_artifact_path is not None
            else None
        ),
        "myopic_artifact_sha256": _file_sha256(myopic_artifact_path),
        "identity_calibration": bool(args.identity_calibration),
        "smoke": bool(args.smoke),
        "seed": int(args.seed),
        "num_bootstrap": int(args.num_bootstrap),
        "script_sha256": _file_sha256(Path(__file__).resolve()),
        "helper_sha256s": helper_sha256s(),
        "git_commit": git_commit,
        "out_parent": project_relative(out.parent),
    }


def _cell_id(cell: dict, run_fingerprint: dict) -> str:
    encoded = json.dumps(
        {"cell": cell, "run": run_fingerprint},
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha1(encoded.encode("utf-8")).hexdigest()[:16]
    return f"{digest}_{cell['reward_schedule']}_{cell['continuation']}_{cell['calibrator']}"


def _phase(prefix_fraction: float) -> str:
    if prefix_fraction < 0.33:
        return "early"
    if prefix_fraction < 0.66:
        return "mid"
    return "late"


def _normalise_prefix_mode(prefix_mode: str) -> str:
    if prefix_mode in {"phase", "early_mid_late", "early/mid/late"}:
        return "early_mid_late"
    if prefix_mode == "exact prefix":
        return "exact_prefix"
    return prefix_mode


def _normalise_subject_pooling(subject_pooling: str) -> str:
    return subject_pooling.replace("-", "_")


def _p_bin(p: float) -> int:
    p = max(0.0, min(1.0, float(p)))
    return min(4, int(p * 5.0))


def _entropy_bin(p: float) -> int:
    p = max(0.0, min(1.0, float(p)))
    if p <= 0.0 or p >= 1.0:
        h = 0.0
    else:
        h = float(-(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p)))
    if h < 0.5:
        return 0
    if h < 0.9:
        return 1
    return 2


def _sigmoid(x: float) -> float:
    x = max(-500.0, min(500.0, float(x)))
    return 1.0 / (1.0 + math.exp(-x))


def _log_loss(y: np.ndarray, p: np.ndarray) -> float:
    clipped = np.clip(p, 1e-6, 1.0 - 1e-6)
    return float(-np.mean(y * np.log(clipped) + (1.0 - y) * np.log(1.0 - clipped)))


def _apply_uncalibrated(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["p_calibrated"] = out["p_raw"].astype(float).clip(0.0, 1.0)
    return out


def _fit_temperature_by_phase(fit_df: pd.DataFrame) -> dict[str, float]:
    """Pick the temperature in ``grid`` that minimises per-bucket log-loss.

    Vectorized: for each phase bucket the loss surface
    ``L(t) = log_loss(y, sigmoid(raw / t))`` is evaluated across the
    entire temperature grid in one broadcast — ``z`` has shape
    ``(n_rows, n_temps)``, sigmoid is one numpy call, log-loss reduces
    along axis 0. Avoids the previous O(n_rows * n_temps) Python loop
    that dominated runtime on large sweeps when
    ``--calibrators temperature`` was in play.
    """
    temps: dict[str, float] = {}
    grid = np.array([0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0], dtype=float)
    mc = fit_df[fit_df["format"] == "MC"].copy()
    if mc.empty:
        return {"default": 1.0}
    mc["phase"] = mc["prefix_fraction"].map(_phase)
    for bucket, group in mc.groupby("phase"):
        y = group["correct"].astype(float).to_numpy()
        raw = group["p_raw"].astype(float).to_numpy()
        if len(np.unique(y)) < 2:
            temps[str(bucket)] = 1.0
            continue
        # Broadcast: z shape (n_rows, n_temps), clamped to match the
        # per-element guard the prior _sigmoid had.
        z = raw[:, None] / grid[None, :]
        z = np.clip(z, -500.0, 500.0)
        probs = 1.0 / (1.0 + np.exp(-z))
        # Log-loss per temperature column. Clip mirrors the existing
        # _log_loss helper exactly so values are bit-comparable.
        clipped = np.clip(probs, 1e-6, 1.0 - 1e-6)
        losses = -np.mean(
            y[:, None] * np.log(clipped)
            + (1.0 - y[:, None]) * np.log(1.0 - clipped),
            axis=0,
        )
        temps[str(bucket)] = float(grid[int(np.argmin(losses))])
    temps["default"] = float(np.median(list(temps.values()))) if temps else 1.0
    return temps


def _apply_temperature(df: pd.DataFrame, temps: dict[str, float]) -> pd.DataFrame:
    """Apply per-phase temperature scaling to p_raw vector.

    Vectorized: compute phase mapping once via Series.map, gather
    per-row temperatures via Series.map on phase labels, then apply
    sigmoid(p/T) to the entire p_raw array in one pass. Avoids the
    iterrows() bottleneck that would dominate large sweeps.
    """
    out = df.copy()
    default_t = float(temps.get("default", 1.0))
    phases = out["prefix_fraction"].astype(float).map(_phase)
    t_values = phases.map(temps).fillna(default_t).astype(float)
    # Avoid divide-by-near-zero; mirrors the per-row max(t, 1e-6).
    t_values = t_values.clip(lower=1e-6)
    z = out["p_raw"].astype(float).to_numpy() / t_values.to_numpy()
    # Clamp before sigmoid (matches _sigmoid's per-element behavior).
    z = np.clip(z, -500.0, 500.0)
    out["p_calibrated"] = 1.0 / (1.0 + np.exp(-z))
    return out


def _fit_isotonic(fit_df: pd.DataFrame) -> tuple[object | None, str | None]:
    mc = fit_df[fit_df["format"] == "MC"].copy()
    if len(mc) < 10 or mc["correct"].nunique() < 2:
        return None, "insufficient_isotonic_data"
    try:
        from sklearn.isotonic import IsotonicRegression
    except Exception as exc:  # noqa: BLE001
        return None, f"isotonic_unavailable:{exc}"
    model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    model.fit(mc["p_raw"].astype(float), mc["correct"].astype(float))
    return model, None


def _apply_isotonic(df: pd.DataFrame, model: object) -> pd.DataFrame:
    out = df.copy()
    out["p_calibrated"] = model.predict(out["p_raw"].astype(float)).clip(0.0, 1.0)
    return out


def _calibrate(
    *,
    calibrator: str,
    fit_df: pd.DataFrame,
    eval_df: pd.DataFrame,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None, dict, str | None]:
    meta = {"method": calibrator}
    canonical = calibrator.replace("_", "-")
    if calibrator == "uncalibrated":
        meta["warning"] = "raw_scores_clipped_to_unit_interval"
        return _apply_uncalibrated(fit_df), _apply_uncalibrated(eval_df), meta, None
    if canonical in {"platt", "logistic", "platt-logistic"}:
        meta["canonical_method"] = "platt-logistic"
        meta["source"] = "adapter_p_calibrated"
        return fit_df.copy(), eval_df.copy(), meta, None
    if calibrator == "temperature":
        temps = _fit_temperature_by_phase(fit_df)
        meta["temperatures"] = temps
        return _apply_temperature(fit_df, temps), _apply_temperature(eval_df, temps), meta, None
    if calibrator == "isotonic":
        model, reason = _fit_isotonic(fit_df)
        if model is None:
            return None, None, meta, reason or "isotonic_fit_failed"
        return _apply_isotonic(fit_df, model), _apply_isotonic(eval_df, model), meta, None
    return None, None, meta, f"unknown_calibrator:{calibrator}"


@dataclass
class SweepEstimator:
    bucket_means: dict[tuple, float]
    bucket_counts: dict[tuple, int]
    prefix_mode: str
    subject_pooling: str
    force_pooled: bool = False
    min_bucket_size: int = 3
    confirmatory: bool = True
    _last_tag: str = "missing"

    @classmethod
    def fit(
        cls,
        *,
        fit_df: pd.DataFrame,
        schedule: object,
        prefix_mode: str,
        subject_pooling: str,
        force_pooled: bool,
    ) -> "SweepEstimator":
        from scripts.stopdff_dp.dp_solver import solve_trajectory

        prefix_mode = _normalise_prefix_mode(prefix_mode)
        subject_pooling = _normalise_subject_pooling(subject_pooling)
        df = fit_df.sort_values(["item_id", "format", "prefix_idx"]).copy()
        estimator = cls(
            bucket_means={},
            bucket_counts={},
            prefix_mode=prefix_mode,
            subject_pooling=subject_pooling,
            force_pooled=force_pooled,
        )
        groups = list(df.groupby(["item_id", "format"], sort=False))
        for _iteration in range(3):
            buckets: dict[tuple, list[float]] = {}
            for (_item_id, fmt), group in groups:
                group = group.reset_index(drop=True)
                ps = group["p_calibrated"].astype(float).clip(0.0, 1.0).tolist()
                fracs = group["prefix_fraction"].astype(float).tolist()
                if len(ps) < 2:
                    continue

                def _cont(t: int, p: float, prefix_fraction: float) -> float:
                    return estimator.estimate(row=group.iloc[t], fmt=str(fmt))

                trace = solve_trajectory(
                    p_trajectory=ps,
                    prefix_fractions=fracs,
                    schedule=schedule,
                    continuation_fn=_cont,
                )
                for t in range(len(ps) - 1):
                    v_next = float(trace.values[t + 1])
                    for key in estimator._aggregate_keys(group.iloc[t], str(fmt)):
                        buckets.setdefault(key, []).append(v_next)

            estimator.bucket_means = {
                key: float(mean(values)) for key, values in buckets.items()
            }
            estimator.bucket_counts = {
                key: len(values) for key, values in buckets.items()
            }
        return estimator

    @staticmethod
    def _prefix_value(row: pd.Series, prefix_mode: str) -> object:
        prefix_mode = _normalise_prefix_mode(prefix_mode)
        if prefix_mode == "exact_prefix":
            return int(row["prefix_idx"])
        if prefix_mode == "early_mid_late":
            return _phase(float(row["prefix_fraction"]))
        raise ValueError(f"unknown prefix bucketing mode {prefix_mode!r}")

    @staticmethod
    def _subject_value(row: pd.Series, subject_pooling: str) -> str:
        subject_pooling = _normalise_subject_pooling(subject_pooling)
        if subject_pooling == "pooled_subject":
            return "__pooled_subject__"
        if subject_pooling == "per_subject":
            return str(row["subject"])
        raise ValueError(f"unknown subject pooling mode {subject_pooling!r}")

    @classmethod
    def _key(
        cls,
        *,
        row: pd.Series,
        fmt: str,
        prefix_mode: str,
        subject_value: str,
        include_prob: bool,
        rung: str,
    ) -> tuple:
        parts: list[object] = [
            rung,
            cls._prefix_value(row, prefix_mode),
            fmt,
            subject_value,
        ]
        if include_prob:
            p = float(row["p_calibrated"])
            parts.extend([_p_bin(p), _entropy_bin(p)])
        return tuple(parts)

    def _aggregate_keys(self, row: pd.Series, fmt: str) -> list[tuple]:
        requested_subject = self._subject_value(row, self.subject_pooling)
        pooled_subject = "__pooled_subject__"
        keys: list[tuple] = []
        if not self.force_pooled:
            keys.append(self._key(
                row=row,
                fmt=fmt,
                prefix_mode=self.prefix_mode,
                subject_value=requested_subject,
                include_prob=True,
                rung="specific_prob",
            ))
        keys.append(self._key(
            row=row,
            fmt=fmt,
            prefix_mode=self.prefix_mode,
            subject_value=requested_subject,
            include_prob=False,
            rung="specific",
        ))
        if requested_subject != pooled_subject:
            if not self.force_pooled:
                keys.append(self._key(
                    row=row,
                    fmt=fmt,
                    prefix_mode=self.prefix_mode,
                    subject_value=pooled_subject,
                    include_prob=True,
                    rung="pooled_subject_prob",
                ))
            keys.append(self._key(
                row=row,
                fmt=fmt,
                prefix_mode=self.prefix_mode,
                subject_value=pooled_subject,
                include_prob=False,
                rung="pooled_subject",
            ))
        keys.append(("format", fmt))
        return keys

    def _lookup_keys(self, row: pd.Series, fmt: str) -> list[tuple[str, tuple]]:
        requested_subject = self._subject_value(row, self.subject_pooling)
        pooled_subject = "__pooled_subject__"
        keys: list[tuple[str, tuple]] = []
        if not self.force_pooled:
            keys.append(("exact", self._key(
                row=row,
                fmt=fmt,
                prefix_mode=self.prefix_mode,
                subject_value=requested_subject,
                include_prob=True,
                rung="specific_prob",
            )))
        keys.append(("pooled", self._key(
            row=row,
            fmt=fmt,
            prefix_mode=self.prefix_mode,
            subject_value=requested_subject,
            include_prob=False,
            rung="specific",
        )))
        if requested_subject != pooled_subject:
            if not self.force_pooled:
                keys.append(("pooled", self._key(
                    row=row,
                    fmt=fmt,
                    prefix_mode=self.prefix_mode,
                    subject_value=pooled_subject,
                    include_prob=True,
                    rung="pooled_subject_prob",
                )))
            keys.append(("pooled", self._key(
                row=row,
                fmt=fmt,
                prefix_mode=self.prefix_mode,
                subject_value=pooled_subject,
                include_prob=False,
                rung="pooled_subject",
            )))
        keys.append(("pooled", ("format", fmt)))
        return keys

    def estimate(self, *, row: pd.Series, fmt: str) -> float:
        for tag, key in self._lookup_keys(row, fmt):
            count = self.bucket_counts.get(key, 0)
            if count >= self.min_bucket_size:
                self._last_tag = tag
                return self.bucket_means[key]
        self._last_tag = "missing"
        return 0.0


def _load_dataframes(
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict, dict]:
    """Load split dataframes after applying MC coverage + retention gates.

    Returns ``(fit_df, eval_df, mc_coverage_block, mc_retention_block,
    mc_build_metadata_block)`` so the gate metadata flows into the sweep
    payload alongside the dataframes. Mirrors the gate enforcement in
    ``scripts/compute_stopdff_dp.py`` (PR #15 074df51) so the sweep
    artifact is treated as audit-card eligible.
    """
    from scripts.stopdff_dp import adapter as adapter_module
    from scripts._audit_gates import (
        build_coverage_metadata,
        build_retention_metadata,
        filter_mc_questions_to_split,
        load_mc_build_metadata,
    )

    MIN_MC_COVERAGE = 0.98

    adapter_module.validate_split_separation(
        fit_split=args.fit_split,
        eval_split=args.eval_split,
    )
    data_dir = Path(args.data_dir)
    mc_path = data_dir / "mc_dataset.json"
    fit_path = data_dir / f"{args.fit_split}_dataset.json"
    eval_path = data_dir / f"{args.eval_split}_dataset.json"
    for path in (mc_path, fit_path, eval_path):
        if not path.exists():
            raise FileNotFoundError(f"missing dataset {path}")

    def _load_questions(path: Path) -> list[dict]:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, list):
            return raw
        if isinstance(raw, dict) and isinstance(raw.get("questions"), list):
            return raw["questions"]
        raise ValueError(f"unsupported dataset shape in {path}")

    mc_questions = _load_questions(mc_path)
    fit_questions = _load_questions(fit_path)
    eval_questions = _load_questions(eval_path)
    fit_qids = {str(q["qid"]) for q in fit_questions}
    eval_qids = {str(q["qid"]) for q in eval_questions}

    # PR #15 review (chatgpt-codex-connector 3314450592): mirror the
    # producer's qid-overlap rejection so the sweep won't silently
    # leak val/test qid sharing into its empirical_bucket cells.
    adapter_module.validate_qid_separation(
        fit_qids=fit_qids,
        eval_qids=eval_qids,
        fit_split=args.fit_split,
        eval_split=args.eval_split,
    )

    if args.smoke:
        fit_qids = set(sorted(fit_qids)[:30])
        eval_qids = set(sorted(eval_qids)[:30])
        keep = fit_qids | eval_qids
        mc_questions = [q for q in mc_questions if str(q["qid"]) in keep]

    # PR #15 review (chatgpt-codex-connector P2 3313920358): the sweep
    # artifact is audit-card eligible, so it must enforce the same MC
    # coverage (>=98%) and retention gates that
    # scripts/compute_stopdff_dp.py enforces. Missing MC rows are not
    # random (items where good distractors could not be built), so a
    # partial subset would silently bias every cell's metrics.
    _mc_eval_rows, eval_coverage = filter_mc_questions_to_split(
        mc_questions, eval_qids
    )
    _mc_fit_rows, fit_coverage = filter_mc_questions_to_split(
        mc_questions, fit_qids
    )

    for split_name, coverage in (
        (args.eval_split, eval_coverage),
        (args.fit_split, fit_coverage),
    ):
        if (
            coverage["coverage_rate"] < MIN_MC_COVERAGE
            and not args.allow_incomplete_mc_coverage
        ):
            raise SystemExit(
                f"ERROR: MC {split_name} coverage is "
                f"{coverage['coverage_rate']:.1%} "
                f"(threshold: {MIN_MC_COVERAGE:.1%}). The sweep artifact "
                f"is audit-card eligible; missing MC qids are not random "
                f"(selected against 'hard to find distractors'). Pass "
                f"--allow-incomplete-mc-coverage to override."
            )

    # Retention gate from build_metadata.json. Mirrors compute_stopdff_dp.py.
    try:
        build_metadata = load_mc_build_metadata(data_dir)
    except RuntimeError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc

    eval_retention = build_retention_metadata(
        build_metadata,
        split=args.eval_split,
        smoke=args.smoke,
        explicit_threshold=None,
        override=args.allow_low_mc_retention,
    )
    fit_retention = build_retention_metadata(
        build_metadata,
        split=args.fit_split,
        smoke=args.smoke,
        explicit_threshold=None,
        override=args.allow_low_mc_retention,
    )

    for split_name, ret in (
        (args.eval_split, eval_retention),
        (args.fit_split, fit_retention),
    ):
        if (
            ret["applies"]
            and ret["passed"] is False
            and not args.allow_low_mc_retention
        ):
            raise SystemExit(
                f"ERROR: raw-{split_name} MC retention is "
                f"{ret['retention_rate']:.1%} (threshold: "
                f"{ret['threshold']:.1%}). Pass --allow-low-mc-retention "
                f"only if you intend the sweep artifact to qualify as a "
                f"retained-MC-subset metric."
            )

    # Audit-card-ready metadata blocks.
    eval_coverage_meta = build_coverage_metadata(
        eval_coverage,
        threshold=MIN_MC_COVERAGE,
        override=args.allow_incomplete_mc_coverage,
    )
    eval_coverage_meta["split"] = args.eval_split

    fit_coverage_meta = build_coverage_metadata(
        fit_coverage,
        threshold=MIN_MC_COVERAGE,
        override=args.allow_incomplete_mc_coverage,
    )
    fit_coverage_meta["split"] = args.fit_split

    mc_coverage_block = {
        args.eval_split: eval_coverage_meta,
        args.fit_split: fit_coverage_meta,
    }
    mc_retention_block = {
        args.eval_split: eval_retention,
        args.fit_split: fit_retention,
    }
    mc_build_metadata_block = {
        "status": build_metadata["status"],
        "source_path": build_metadata["source_path"],
        "source_sha256": build_metadata["source_sha256"],
    }

    calibration_path = None if args.identity_calibration else Path(args.calibration)
    fit_df = adapter_module.build_dataframe(
        mc_questions=mc_questions,
        target_qids=fit_qids,
        split_name=args.fit_split,
        calibration_path=calibration_path,
        identity_calibration=args.identity_calibration,
    )
    eval_df = adapter_module.build_dataframe(
        mc_questions=mc_questions,
        target_qids=eval_qids,
        split_name=args.eval_split,
        calibration_path=calibration_path,
        identity_calibration=args.identity_calibration,
    )
    return (
        fit_df,
        eval_df,
        mc_coverage_block,
        mc_retention_block,
        mc_build_metadata_block,
    )


def _fit_estimator(
    *,
    continuation: str,
    fit_df: pd.DataFrame,
    schedule: object,
    prefix_bucketing: str,
    subject_pooling: str,
) -> object:
    if continuation == "oracle_trajectory":
        from scripts.stopdff_dp.continuation import OracleTrajectoryEstimator

        return OracleTrajectoryEstimator.fit(
            fit_df=fit_df,
            schedule=schedule,
            fit_split_name=str(fit_df["split"].iloc[0]) if not fit_df.empty else "val",
        )
    if continuation in {"empirical_bucket", "pooled_empirical"}:
        return SweepEstimator.fit(
            fit_df=fit_df,
            schedule=schedule,
            prefix_mode=prefix_bucketing,
            subject_pooling=subject_pooling,
            force_pooled=(continuation == "pooled_empirical"),
        )
    raise ValueError(f"unknown continuation estimator {continuation!r}")


def _solve_rows(
    *,
    rows: pd.DataFrame,
    fmt: str,
    estimator: object,
    schedule: object,
) -> object:
    from scripts.stopdff_dp import dp_solver as dp_module
    from scripts.stopdff_dp.continuation import OracleTrajectoryEstimator

    rows = rows.sort_values("prefix_idx").reset_index(drop=True)
    ps = rows["p_calibrated"].astype(float).clip(0.0, 1.0).tolist()
    prefix_fractions = rows["prefix_fraction"].astype(float).tolist()
    T = len(ps)
    tags_per_step: dict[int, str] = {(T - 1): "exact"} if T else {}

    def _continuation(t: int, p: float, prefix_fraction: float) -> float:
        if isinstance(estimator, OracleTrajectoryEstimator):
            tags_per_step[t] = "exact"
            return estimator.estimate(
                item_trajectory=ps,
                item_prefix_fractions=prefix_fractions,
                t=t,
                schedule=schedule,
            )
        value = estimator.estimate(row=rows.iloc[t], fmt=fmt)
        tags_per_step[t] = getattr(estimator, "_last_tag", "missing")
        return value

    return dp_module.solve_trajectory(
        p_trajectory=ps,
        prefix_fractions=prefix_fractions,
        schedule=schedule,
        continuation_fn=_continuation,
        item_id=str(rows["item_id"].iloc[0]) if not rows.empty else "",
        fmt=fmt,
        coverage_tagger=lambda t: tags_per_step.get(t, "missing"),
    )


def _myopic_trace(rows: pd.DataFrame, fmt: str, schedule: object) -> object:
    from scripts.stopdff_dp import dp_solver as dp_module

    rows = rows.sort_values("prefix_idx")
    return dp_module.solve_trajectory(
        p_trajectory=rows["p_calibrated"].astype(float).clip(0.0, 1.0).tolist(),
        prefix_fractions=rows["prefix_fraction"].astype(float).tolist(),
        schedule=schedule,
        continuation_fn=lambda *_args, **_kwargs: 0.0,
        item_id=str(rows["item_id"].iloc[0]),
        fmt=fmt,
    )


def _bootstrap_ci(values_by_item: list[tuple[str, float]], *, n: int, seed: int) -> dict:
    values = np.array([v for _, v in values_by_item], dtype=float)
    if len(values) == 0 or n <= 0:
        return {"mean": [None, None], "median": [None, None]}
    rng = np.random.default_rng(seed)
    mean_samples = []
    median_samples = []
    for _ in range(n):
        sample = rng.choice(values, size=len(values), replace=True)
        mean_samples.append(float(np.mean(sample)))
        median_samples.append(float(np.median(sample)))
    return {
        "mean": [
            float(np.percentile(mean_samples, 2.5)),
            float(np.percentile(mean_samples, 97.5)),
        ],
        "median": [
            float(np.percentile(median_samples, 2.5)),
            float(np.percentile(median_samples, 97.5)),
        ],
    }


def _summarize_stopdff(per_item: list[tuple[str, int]]) -> dict:
    signed = [s for _, s in per_item]
    abs_values = [abs(s) for s in signed]
    return {
        "stopdff_dp_signed_median": float(median(signed)) if signed else 0.0,
        "stopdff_dp_signed_mean": float(mean(signed)) if signed else 0.0,
        "stopdff_dp_abs_median": float(median(abs_values)) if abs_values else 0.0,
        "stopdff_dp_abs_mean": float(mean(abs_values)) if abs_values else 0.0,
        "n_items": len(signed),
        "direction_breakdown": {
            "mc_earlier": sum(1 for s in signed if s < 0),
            "qa_earlier": sum(1 for s in signed if s > 0),
            "same_step": sum(1 for s in signed if s == 0),
        },
        "direction_fractions": {
            "mc_earlier": (
                sum(1 for s in signed if s < 0) / len(signed) if signed else 0.0
            ),
            "qa_earlier": (
                sum(1 for s in signed if s > 0) / len(signed) if signed else 0.0
            ),
            "same_step": (
                sum(1 for s in signed if s == 0) / len(signed) if signed else 0.0
            ),
        },
    }


def _run_cell(
    *,
    cell: dict,
    fit_df_base: pd.DataFrame,
    eval_df_base: pd.DataFrame,
    args: argparse.Namespace,
    effective_argv: list[str],
    git_commit: str | None,
    git_dirty: bool | None,
    run_fingerprint: dict,
    myopic_artifact: dict | None,
) -> dict:
    from scripts.stopdff_dp import diagnostics as diag_module
    from scripts.stopdff_dp import dp_solver as dp_module
    from scripts.stopdff_dp import rewards as rewards_module

    started = time.time()
    base = {
        **cell,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "argv": effective_argv,
        "seed": args.seed,
        "fit_split": args.fit_split,
        "eval_split": args.eval_split,
        "run_fingerprint": run_fingerprint,
        "started_at": _now(),
    }
    try:
        if cell["format_condition"] not in {"QA-prefix", "MC-fixed"}:
            reason = (
                "choices_only_unavailable"
                if cell["format_condition"] == "choices-only"
                else "format_condition_unavailable"
            )
            return {
                **base,
                "status": "skipped",
                "skip_reason": reason,
                "format_fallback": {
                    "requested": cell["format_condition"],
                    "distinct_condition_available": False,
                    "reason": (
                        "current artifact exposes paired QA-prefix and MC-fixed rows only"
                    ),
                },
                "confirmatory_included": False,
                "wall_clock_seconds": time.time() - started,
                "completed_at": _now(),
            }

        schedule = rewards_module.get_schedule(cell["reward_schedule"])
        fit_df, eval_df, calibration_meta, skip_reason = _calibrate(
            calibrator=cell["calibrator"],
            fit_df=fit_df_base,
            eval_df=eval_df_base,
        )
        if skip_reason:
            return {
                **base,
                "status": "skipped",
                "skip_reason": skip_reason,
                "calibration": calibration_meta,
                "confirmatory_included": False,
                "wall_clock_seconds": time.time() - started,
                "completed_at": _now(),
            }

        assert fit_df is not None and eval_df is not None
        estimator = _fit_estimator(
            continuation=cell["continuation"],
            fit_df=fit_df,
            schedule=schedule,
            prefix_bucketing=cell["prefix_bucketing"],
            subject_pooling=cell["subject_pooling"],
        )

        mc_traces = []
        qa_traces = []
        myopic_mc = []
        myopic_qa = []
        per_item = []
        myopic_per_item = []
        regret_mc_to_qa = []
        regret_qa_to_mc = []
        regret_vs_myopic = []

        for item_id, group in eval_df.groupby("item_id", sort=False):
            mc_rows = group[group["format"] == "MC"]
            qa_rows = group[group["format"] == "QA"]
            if mc_rows.empty or qa_rows.empty:
                continue
            mc_trace = _solve_rows(rows=mc_rows, fmt="MC", estimator=estimator, schedule=schedule)
            qa_trace = _solve_rows(rows=qa_rows, fmt="QA", estimator=estimator, schedule=schedule)
            mc_myopic = _myopic_trace(mc_rows, "MC", schedule)
            qa_myopic = _myopic_trace(qa_rows, "QA", schedule)
            mc_traces.append(mc_trace)
            qa_traces.append(qa_trace)
            myopic_mc.append(mc_myopic)
            myopic_qa.append(qa_myopic)
            per_item.append((str(item_id), dp_module.stopdff_for_item(
                mc_trace=mc_trace,
                qa_trace=qa_trace,
            )))
            myopic_per_item.append((str(item_id), dp_module.stopdff_for_item(
                mc_trace=mc_myopic,
                qa_trace=qa_myopic,
            )))
            if mc_trace.values and qa_trace.values:
                qa_at_mc = (
                    qa_trace.values[mc_trace.stop_step]
                    if 0 <= mc_trace.stop_step < len(qa_trace.values)
                    else 0.0
                )
                mc_at_qa = (
                    mc_trace.values[qa_trace.stop_step]
                    if 0 <= qa_trace.stop_step < len(mc_trace.values)
                    else 0.0
                )
                regret_mc_to_qa.append(max(0.0, float(qa_trace.values[0] - qa_at_mc)))
                regret_qa_to_mc.append(max(0.0, float(mc_trace.values[0] - mc_at_qa)))
            if mc_trace.values and qa_trace.values and mc_myopic.values and qa_myopic.values:
                dp_start = float(mc_trace.values[0] + qa_trace.values[0])
                myopic_start = float(mc_myopic.values[0] + qa_myopic.values[0])
                regret_vs_myopic.append(max(0.0, dp_start - myopic_start))

        coverage = diag_module.summarize_coverage(mc_traces + qa_traces)
        ceiling = diag_module.detect_ceiling_effects(mc_traces, qa_traces)
        confirmatory = cell["continuation"] != "oracle_trajectory"
        if coverage.get("verdict") == "warn":
            gate_verdict = "warn"
            gate_reason = f"coverage:{coverage.get('reason')}"
            confirmatory_included = False
        elif any(ceiling.get(k) for k in (
            "all_stop_at_first_prefix",
            "all_stop_at_final_prefix",
            "no_cross_format_stopping_variance",
        )):
            gate_verdict = "warn"
            gate_reason = "ceiling_effect"
            confirmatory_included = bool(confirmatory)
        else:
            gate_verdict = "pass"
            gate_reason = "all_clean"
            confirmatory_included = bool(confirmatory)
        if not confirmatory:
            confirmatory_included = False

        summary = _summarize_stopdff(per_item)
        myopic_summary = _summarize_stopdff(myopic_per_item)
        signed_values = [(item_id, float(value)) for item_id, value in per_item]
        abs_values = [(item_id, float(abs(value))) for item_id, value in per_item]
        trace_count = len(mc_traces) + len(qa_traces)
        stopped = sum(
            1 for tr in [*mc_traces, *qa_traces]
            if 0 <= tr.stop_step < len(tr.values)
        )

        return {
            **base,
            "status": "completed",
            "calibration": calibration_meta,
            "format_fallback": {
                "requested": cell["format_condition"],
                "used_mc_format": "MC",
                "used_qa_format": "QA",
                "distinct_condition_available": True,
                "reason": (
                    "current artifact supports paired QA-prefix and MC-fixed comparison"
                ),
            },
            "metrics": {
                **summary,
                "bootstrap_ci": {
                    "signed_stopdff": _bootstrap_ci(
                        signed_values,
                        n=args.num_bootstrap,
                        seed=args.seed,
                    ),
                    "absolute_stopdff": _bootstrap_ci(
                        abs_values,
                        n=args.num_bootstrap,
                        seed=args.seed + 1,
                    ),
                },
                "myopic": myopic_summary,
                "dp_minus_myopic_signed_mean": (
                    summary["stopdff_dp_signed_mean"]
                    - myopic_summary["stopdff_dp_signed_mean"]
                ),
                "myopic_artifact_comparison": (
                    {
                        "available": True,
                        "median_abs_prefix_shift": myopic_artifact.get(
                            "median_abs_prefix_shift"
                        ),
                        "mean_abs_prefix_shift": myopic_artifact.get(
                            "mean_abs_prefix_shift"
                        ),
                        "gate_verdict": myopic_artifact.get("gate_verdict"),
                        "ceiling_effect_detected": myopic_artifact.get(
                            "ceiling_effect_detected"
                        ),
                        "delta_abs_median_vs_myopic_artifact": (
                            summary["stopdff_dp_abs_median"]
                            - float(myopic_artifact.get("median_abs_prefix_shift", 0.0))
                        ),
                    }
                    if myopic_artifact is not None
                    else {"available": False}
                ),
                "decision_regret_proxy_mean": (
                    float(mean(regret_vs_myopic)) if regret_vs_myopic else None
                ),
                "decision_regret_mc_to_qa_mean": (
                    float(mean(regret_mc_to_qa)) if regret_mc_to_qa else None
                ),
                "decision_regret_qa_to_mc_mean": (
                    float(mean(regret_qa_to_mc)) if regret_qa_to_mc else None
                ),
                "no_stop_rate": (
                    ceiling["n_never_stopped_cells"] / trace_count if trace_count else 0.0
                ),
                "final_stop_rate": (
                    sum(
                        1 for tr in [*mc_traces, *qa_traces]
                        if len(tr.values) and tr.stop_step == len(tr.values) - 1
                    ) / trace_count if trace_count else 0.0
                ),
                "stopped_rate": stopped / trace_count if trace_count else 0.0,
                "fallback_rate": (
                    (coverage.get("fraction_pooled") or 0.0)
                    + (coverage.get("fraction_missing") or 0.0)
                ),
            },
            "coverage": coverage,
            "ceiling_flags": ceiling,
            "gate_verdict": gate_verdict,
            "gate_verdict_reason": gate_reason,
            "confirmatory_included": confirmatory_included,
            "confirmatory": confirmatory,
            "wall_clock_seconds": time.time() - started,
            "completed_at": _now(),
        }
    except Exception as exc:  # noqa: BLE001 - failed cells are data
        return {
            **base,
            "status": "failed",
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "confirmatory_included": False,
            "wall_clock_seconds": time.time() - started,
            "completed_at": _now(),
        }


def _iter_cells(args: argparse.Namespace) -> list[dict]:
    cells = []
    for reward in _csv(args.reward_schedules):
        for continuation in _csv(args.continuations):
            for calibrator in _csv(args.calibrators):
                for fmt in _csv(args.formats):
                    for prefix in _csv(args.prefix_bucketing):
                        for subject in _csv(args.subject_pooling):
                            cells.append({
                                "reward_schedule": reward,
                                "continuation": continuation,
                                "calibrator": calibrator,
                                "format_condition": fmt,
                                "prefix_bucketing": prefix,
                                "subject_pooling": subject,
                            })
    return cells


def _cell_cache_dir(args: argparse.Namespace, out: Path) -> Path:
    if args.artifact_dir:
        return Path(args.artifact_dir) / "stopdff_dp_sweep_cells"
    return out.parent / "stopdff_dp_sweep_cells"


def _load_cached_cells(cell_dir: Path) -> list[dict]:
    cells = []
    if not cell_dir.exists():
        return cells
    for path in sorted(cell_dir.glob("*.json")):
        try:
            cell = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        cell["cache_path"] = str(path)
        cells.append(cell)
    return cells


def _load_myopic_artifact(
    args: argparse.Namespace, out: Path
) -> tuple[dict | None, Path | None]:
    """Load the diagnostic myopic StopDFF artifact, returning the payload + path.

    Returns (None, None) when no candidate file can be parsed. The
    resolved path is needed so the sweep can hash the file into the
    cache fingerprint (PR #15 review 3314472776 — without the hash,
    regenerating stopdff.json in place would leave fingerprints
    matching and cached cells republishing stale comparison fields).
    """
    candidates = []
    if args.artifact_dir:
        candidates.append(Path(args.artifact_dir) / "stopdff.json")
    candidates.extend([
        out.parent / "stopdff.json",
        Path(args.calibration).parent / "stopdff.json",
        PROJECT_ROOT / "paper_exports" / "stopdff.json",
    ])
    seen: set[Path] = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        return payload, resolved
    return None, None


def _paper_safe_interpretation(completed: list[dict]) -> dict:
    non_oracle = [
        c for c in completed
        if c.get("continuation") != "oracle_trajectory"
    ]
    included = [c for c in completed if c.get("confirmatory_included")]
    if not included:
        return {
            "verdict": "WARN",
            "reason": "no_confirmatory_cells",
        }
    shifted = [
        c for c in non_oracle
        if abs(c.get("metrics", {}).get("stopdff_dp_abs_median", 0.0)) > 1.0
        or abs(c.get("metrics", {}).get("stopdff_dp_signed_median", 0.0)) > 1.0
    ]
    if shifted:
        return {
            "verdict": "FAIL",
            "reason": "material_mc_shift",
            "n_shifted_cells": len(shifted),
        }
    weak = [
        c for c in non_oracle
        if c.get("gate_verdict") != "pass"
        or c.get("coverage", {}).get("verdict") == "warn"
        or any(c.get("ceiling_flags", {}).get(k) for k in (
            "all_stop_at_first_prefix",
            "all_stop_at_final_prefix",
            "no_cross_format_stopping_variance",
        ))
    ]
    if weak:
        return {
            "verdict": "WARN",
            "reason": "small_stopdff_but_coverage_or_ceiling_weak",
            "n_weak_cells": len(weak),
        }
    return {
        "verdict": "PASS",
        "reason": "small_stopdff_and_coverage_calibration_gates_pass",
    }


def _aggregate(cells: list[dict], args: argparse.Namespace, effective_argv: list[str]) -> dict:
    completed = [c for c in cells if c.get("status") == "completed"]
    confirmatory = [c for c in completed if c.get("confirmatory_included")]
    return {
        "metadata": {
            "metric_type": "finite_horizon_dp_sweep",
            "timestamp": _now(),
            "argv": effective_argv,
            "seed": args.seed,
            "fit_split": args.fit_split,
            "eval_split": args.eval_split,
            "num_bootstrap": args.num_bootstrap,
            "n_jobs": args.n_jobs,
            "confirmatory_cell_count": len(confirmatory),
            "completed_cell_count": len(completed),
            "failed_cell_count": sum(1 for c in cells if c.get("status") == "failed"),
            "skipped_cell_count": sum(1 for c in cells if c.get("status") == "skipped"),
        },
        "paper_safe_interpretation": _paper_safe_interpretation(completed),
        "cells": cells,
        "confirmatory_cells": [
            c.get("cell_id") for c in confirmatory
        ],
    }


def _write_markdown(path: Path, payload: dict) -> None:
    lines = [
        "# DP StopDFF Sweep",
        "",
        f"Generated: `{payload['metadata']['timestamp']}`",
        "",
        "## paper-safe interpretation",
        "",
        f"**Verdict:** {payload.get('paper_safe_interpretation', {}).get('verdict', 'WARN')} "
        f"({payload.get('paper_safe_interpretation', {}).get('reason', 'not_evaluated')})",
        "",
        "- PASS only if DP StopDFF is small and coverage/calibration gates pass.",
        "- WARN if DP StopDFF is small but coverage is weak or a ceiling effect persists.",
        "- FAIL if DP StopDFF is materially shifted under MC.",
        "- Oracle continuation cells are diagnostic upper bounds and are excluded from confirmatory interpretation.",
        "",
        "## Cell Status",
        "",
        "| Status | Count |",
        "|---|---:|",
    ]
    for status in ("completed", "skipped", "failed"):
        lines.append(
            f"| {status} | {sum(1 for c in payload['cells'] if c.get('status') == status)} |"
        )
    lines.extend(["", "## Completed Cells", ""])
    lines.append("| Cell | Reward | Continuation | Calibrator | Format | Signed mean | Gate |")
    lines.append("|---|---|---|---|---|---:|---|")
    for cell in payload["cells"]:
        if cell.get("status") != "completed":
            continue
        metrics = cell.get("metrics", {})
        lines.append(
            "| {cell_id} | {reward} | {cont} | {cal} | {fmt} | {mean:.3f} | {gate} |".format(
                cell_id=cell.get("cell_id", ""),
                reward=cell.get("reward_schedule", ""),
                cont=cell.get("continuation", ""),
                cal=cell.get("calibrator", ""),
                fmt=cell.get("format_condition", ""),
                mean=float(metrics.get("stopdff_dp_signed_mean", 0.0)),
                gate=cell.get("gate_verdict", ""),
            )
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def _latex_escape(value: object) -> str:
    """Escape LaTeX special characters in a string field.

    Identifier-style labels like ``acf_flat``, ``empirical_bucket``,
    ``QA-prefix`` will not compile as plain LaTeX text because ``_`` is
    a subscript character; ``&``, ``%``, ``$``, ``#``, ``{``, ``}``,
    ``~``, ``^`` and ``\\`` have their own catastrophic-or-silent issues.
    Replaces each with its standard ``\\<char>`` equivalent (``\\textasciitilde{}``
    / ``\\textasciicircum{}`` for the two diacritic-mark cases that don't
    have a simple backslash form).

    Implemented as a single-pass ``re.sub`` so that replacement strings
    (e.g. ``\\textbackslash{}`` containing ``{}``) are not re-processed
    by later rules.
    """
    text = str(value)
    # Map of source char to its LaTeX-safe replacement. Single-pass via
    # re.sub avoids double-escaping: each input character is matched at
    # most once and its replacement is written verbatim into the output.
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
    pattern = re.compile("|".join(re.escape(c) for c in replacements))
    return pattern.sub(lambda m: replacements[m.group(0)], text)


def _write_latex(path: Path, payload: dict) -> None:
    rows = [
        "\\begin{tabular}{llllr}",
        "\\toprule",
        "Reward & Continuation & Calibrator & Format & Signed mean \\\\",
        "\\midrule",
    ]
    for cell in payload["cells"]:
        if cell.get("status") != "completed":
            continue
        rows.append(
            f"{_latex_escape(cell['reward_schedule'])} & "
            f"{_latex_escape(cell['continuation'])} & "
            f"{_latex_escape(cell['calibrator'])} & "
            f"{_latex_escape(cell['format_condition'])} & "
            f"{cell['metrics']['stopdff_dp_signed_mean']:.3f} \\\\"
        )
    rows.extend(["\\bottomrule", "\\end{tabular}"])
    path.write_text("\n".join(rows), encoding="utf-8")


def _write_rgb_png(path: Path, rows: list[bytearray]) -> None:
    height = len(rows)
    width = len(rows[0]) // 3 if rows else 1

    def _chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    raw = b"".join(b"\x00" + bytes(row) for row in rows)
    png = (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + _chunk(b"IDAT", zlib.compress(raw))
        + _chunk(b"IEND", b"")
    )
    path.write_bytes(png)


def _simple_canvas(width: int = 640, height: int = 360) -> list[bytearray]:
    return [bytearray([255, 255, 255] * width) for _ in range(height)]


def _draw_rect(
    rows: list[bytearray],
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    color: tuple[int, int, int],
) -> None:
    height = len(rows)
    width = len(rows[0]) // 3
    x0 = max(0, min(width, x0))
    x1 = max(0, min(width, x1))
    y0 = max(0, min(height, y0))
    y1 = max(0, min(height, y1))
    for y in range(y0, y1):
        row = rows[y]
        for x in range(x0, x1):
            idx = x * 3
            row[idx:idx + 3] = bytes(color)


def _draw_simple_bar_png(
    path: Path,
    values: list[float],
    *,
    color: tuple[int, int, int],
) -> None:
    rows = _simple_canvas()
    width = len(rows[0]) // 3
    height = len(rows)
    _draw_rect(rows, 48, 24, 50, height - 36, (80, 80, 80))
    _draw_rect(rows, 48, height - 38, width - 24, height - 36, (80, 80, 80))
    values = values or [0.0]
    max_abs = max(1.0, max(abs(v) for v in values))
    bar_w = max(4, (width - 90) // max(1, len(values)))
    zero_y = height // 2
    _draw_rect(rows, 50, zero_y, width - 24, zero_y + 1, (190, 190, 190))
    for i, value in enumerate(values):
        x0 = 56 + i * bar_w
        x1 = x0 + max(3, bar_w - 2)
        y = int(zero_y - (value / max_abs) * (height // 2 - 44))
        _draw_rect(rows, x0, min(zero_y, y), x1, max(zero_y, y) + 1, color)
    _write_rgb_png(path, rows)


def _draw_simple_scatter_png(
    path: Path,
    xs: list[float],
    ys: list[float],
) -> None:
    rows = _simple_canvas(420, 420)
    width = len(rows[0]) // 3
    height = len(rows)
    _draw_rect(rows, 42, 24, 44, height - 42, (80, 80, 80))
    _draw_rect(rows, 42, height - 44, width - 24, height - 42, (80, 80, 80))
    xs = xs or [0.0]
    ys = ys or [0.0]
    lo = min(0.0, min(xs), min(ys))
    hi = max(1.0, max(xs), max(ys))
    span = hi - lo if hi > lo else 1.0
    for x, y in zip(xs, ys):
        px = int(44 + ((x - lo) / span) * (width - 76))
        py = int(height - 45 - ((y - lo) / span) * (height - 76))
        _draw_rect(rows, px - 3, py - 3, px + 4, py + 4, (47, 143, 91))
    _write_rgb_png(path, rows)


def _draw_simple_stacked_png(
    path: Path,
    exact: list[float],
    pooled: list[float],
) -> None:
    rows = _simple_canvas()
    width = len(rows[0]) // 3
    height = len(rows)
    _draw_rect(rows, 48, 24, 50, height - 36, (80, 80, 80))
    _draw_rect(rows, 48, height - 38, width - 24, height - 36, (80, 80, 80))
    exact = exact or [0.0]
    pooled = pooled or [0.0]
    bar_w = max(4, (width - 90) // max(1, len(exact)))
    plot_h = height - 72
    for i, (e, p) in enumerate(zip(exact, pooled)):
        x0 = 56 + i * bar_w
        x1 = x0 + max(3, bar_w - 2)
        e_h = int(max(0.0, min(1.0, e)) * plot_h)
        p_h = int(max(0.0, min(1.0, p)) * plot_h)
        y1 = height - 38
        _draw_rect(rows, x0, y1 - e_h, x1, y1, (80, 125, 188))
        _draw_rect(rows, x0, y1 - e_h - p_h, x1, y1 - e_h, (240, 162, 2))
    _write_rgb_png(path, rows)


def _write_figures(fig_dir: Path, payload: dict) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        fig_dir / "stopdff_dp_phase_diagram.png",
        fig_dir / "stopdff_dp_vs_myopic.png",
        fig_dir / "stopdff_dp_coverage.png",
    ]
    completed = [c for c in payload["cells"] if c.get("status") == "completed"]
    labels = [c.get("cell_id", "")[:6] for c in completed] or ["none"]
    signed = [
        c.get("metrics", {}).get("stopdff_dp_signed_mean", 0.0)
        for c in completed
    ] or [0.0]
    myopic = [
        c.get("metrics", {}).get("myopic", {}).get("stopdff_dp_signed_mean", 0.0)
        for c in completed
    ] or [0.0]
    exact = [
        c.get("coverage", {}).get("fraction_exact") or 0.0
        for c in completed
    ] or [0.0]
    pooled = [
        c.get("coverage", {}).get("fraction_pooled") or 0.0
        for c in completed
    ] or [0.0]
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001
        _draw_simple_bar_png(paths[0], signed, color=(59, 110, 168))
        _draw_simple_scatter_png(paths[1], myopic, signed)
        _draw_simple_stacked_png(paths[2], exact, pooled)
        return

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(labels, signed, color="#3b6ea8")
    ax.set_ylabel("signed mean")
    ax.set_title("DP StopDFF by cell")
    fig.tight_layout()
    fig.savefig(paths[0], dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.scatter(myopic, signed, color="#2f8f5b")
    ax.axline((0, 0), slope=1, color="#555555", linewidth=1)
    ax.set_xlabel("myopic")
    ax.set_ylabel("DP")
    fig.tight_layout()
    fig.savefig(paths[1], dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(labels, exact, label="exact", color="#507dbc")
    ax.bar(labels, pooled, bottom=exact, label="pooled", color="#f0a202")
    ax.set_ylim(0, 1)
    ax.set_ylabel("coverage fraction")
    ax.legend()
    fig.tight_layout()
    fig.savefig(paths[2], dpi=120)
    plt.close(fig)


def _write_aggregate_outputs(out: Path, payload: dict) -> None:
    _write_json_atomic(out, payload)
    _write_markdown(out.with_suffix(".md"), payload)
    _write_latex(out.with_name("stopdff_dp_sweep_table.tex"), payload)
    _write_figures(out.parent / "figures", payload)


def main(argv: Optional[list[str]] = None) -> int:
    effective_argv = list(argv) if argv is not None else list(sys.argv[1:])
    args = _parse_args(argv)
    artifact_dir = Path(args.artifact_dir) if args.artifact_dir else DEFAULT_OUT.parent
    out = Path(args.out) if args.out else artifact_dir / "stopdff_dp_sweep.json"
    cell_dir = _cell_cache_dir(args, out)
    git_commit, git_dirty = _git_metadata(args, out=out)
    myopic_artifact, myopic_artifact_path = _load_myopic_artifact(args, out)
    run_fingerprint = _run_fingerprint(
        args, out=out, git_commit=git_commit,
        myopic_artifact_path=myopic_artifact_path,
    )
    start = time.time()
    max_seconds = None if args.max_wall_hours is None else args.max_wall_hours * 3600.0

    try:
        (
            fit_df_base,
            eval_df_base,
            mc_coverage_block,
            mc_retention_block,
            mc_build_metadata_block,
        ) = _load_dataframes(args)
    except SystemExit:
        # Gate violations (coverage/retention) deliberately raise SystemExit
        # with a stderr message; let the operator see the original message
        # rather than silently degrading to a payload write.
        raise
    except Exception as exc:  # noqa: BLE001
        payload = {
            "metadata": {
                "metric_type": "finite_horizon_dp_sweep",
                "timestamp": _now(),
                "argv": effective_argv,
                "seed": args.seed,
                "status": "failed",
                "error": str(exc),
            },
            "cells": [],
        }
        _write_aggregate_outputs(out, payload)
        return 1

    existing_by_id = {}
    if args.resume or args.only_missing:
        for cached in _load_cached_cells(cell_dir):
            if (
                cached.get("cell_id")
                and cached.get("run_fingerprint") == run_fingerprint
            ):
                existing_by_id[cached["cell_id"]] = cached

    cells_to_run: list[tuple[dict, str, Path]] = []
    for cell in _iter_cells(args):
        cid = _cell_id(cell, run_fingerprint)
        path = cell_dir / f"{cid}.json"
        if (args.resume or args.only_missing) and cid in existing_by_id:
            continue
        if args.max_cells is not None and len(cells_to_run) >= args.max_cells:
            break
        if max_seconds is not None and time.time() - start >= max_seconds:
            break
        cells_to_run.append((cell, cid, path))

    def _execute_one(item: tuple[dict, str, Path]) -> dict:
        cell, cid, path = item
        payload = _run_cell(
            cell=cell,
            fit_df_base=fit_df_base,
            eval_df_base=eval_df_base,
            args=args,
            effective_argv=effective_argv,
            git_commit=git_commit,
            git_dirty=git_dirty,
            run_fingerprint=run_fingerprint,
            myopic_artifact=myopic_artifact,
        )
        payload["cell_id"] = cid
        payload["cache_path"] = str(path)
        _write_json_atomic(path, payload)
        return payload

    if args.n_jobs and args.n_jobs > 1 and len(cells_to_run) > 1:
        with ThreadPoolExecutor(max_workers=args.n_jobs) as executor:
            for future in as_completed(executor.submit(_execute_one, item) for item in cells_to_run):
                payload = future.result()
                existing_by_id[payload["cell_id"]] = payload
    else:
        for item in cells_to_run:
            payload = _execute_one(item)
            existing_by_id[payload["cell_id"]] = payload

    # PR #15 review (chatgpt-codex-connector 3313958597): non-resume runs
    # must not publish cells from prior wider sweeps cached on disk. Build
    # the aggregate from existing_by_id which contains exactly the cells
    # either executed this invocation OR pre-loaded under --resume/--only-
    # missing. Cached cells from older runs whose fingerprint coincidentally
    # matches but whose cell_id is outside this invocation's executed set
    # are NOT silently included.
    requested_ids = {_cell_id(c, run_fingerprint) for c in _iter_cells(args)}
    visible_cells = [
        c for c in existing_by_id.values()
        if c.get("cell_id") in requested_ids
        and c.get("run_fingerprint") == run_fingerprint
    ]
    aggregate = _aggregate(visible_cells, args, effective_argv)
    aggregate["mc_coverage"] = mc_coverage_block
    aggregate["mc_retention_gate"] = mc_retention_block
    aggregate["mc_build_metadata"] = mc_build_metadata_block
    _write_aggregate_outputs(out, aggregate)
    print(
        f"[STOPDFF-DP-SWEEP] Wrote {out} "
        f"(cells={len(visible_cells)}, completed={aggregate['metadata']['completed_cell_count']})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
