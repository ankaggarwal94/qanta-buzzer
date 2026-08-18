#!/usr/bin/env python3
"""Hazard-pretrain efficacy eval harness.

Orchestrates the paired WITH/WITHOUT/compute-control comparison for the
``--hazard-pretrain`` warm-start bridge (spec: ``.correctless/specs/
hazard-efficacy-eval.md``, Track B rules R-003, R-005..R-009, R-010b,
R-011..R-014).

Pipeline:

1. Preflight (R-013): resolve persisted split artifacts, plan every
   arm x seed run up front (distinct dirs ``<out>/<arm>_seed<k>``, full
   child argv lists), print the plan; ``--dry-run`` stops here.
2. Shared supervised warm-start runs ONCE; its checkpoint is branched
   into every arm via ``--skip-supervised --model-path <shared>`` and a
   split manifest is persisted next to it (R-003 / R-008).
3. Child training runs via ``scripts/train_t5_policy.py`` argv LISTS,
   shell=False, tee'd to ``<run_dir>/train.log`` (R-011); complete
   run dirs (RUN_COMPLETE.json marker + checkpoints + sidecars) are
   skipped/resumed, partial dirs fail loud unless ``--force`` (R-013).
4. Arm-control sidecar assertions diff ``config_used.json`` /
   ``split_manifest.json`` across all runs (R-003).
5. Evaluation through ``scripts.compare_policies.evaluate_t5_policy``
   only (R-005 / R-012), persisted per run as
   ``<run_dir>/eval_result.json`` (R-014).
6. Stop-probability probe of supervised vs hazard checkpoints plus
   ``hazard_history.json`` loss halves -> ``hazard_dynamics`` (R-010b).
7. Report assembly reads EXCLUSIVELY per-run files and writes
   ``hazard_efficacy_report.json`` + ``hazard_efficacy_plot.png``
   (headless Agg backend, ``savefig`` only) (R-009 / R-014).
"""

from __future__ import annotations

import argparse
import contextlib
import fnmatch
import hashlib
import io
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import yaml

# R-012: the harness reuses the shared eval entry point and bootstrap
# resampler as module attributes (identity-pinned by the tests) — it never
# reimplements episode rollouts or metric math.
from evaluation.controls import bootstrap_ci
from scripts._common import (
    ARTIFACT_DIR,
    PROCESSED_DIR,
    load_mc_questions,
    save_json,
)
from scripts.compare_policies import evaluate_t5_policy

# ---------------------------------------------------------------------------
# Pinned constants (interface — values are spec-mandated)
# ---------------------------------------------------------------------------

SCHEMA_VERSION = 1
DEVICE2_CAVEAT = "Full-scale t5-large efficacy remains a Device-2 (RTX 5090) run."
RUN_COMPLETE_MARKER = "RUN_COMPLETE.json"
# QA-002: identity marker written beside the shared supervised checkpoint
# ({"config_hash", "git_sha", "model_name", "supervised_seed"}); validated on
# every reuse so a stale checkpoint (different model/config) can never
# silently seed all arms. QA-R2-3: "supervised_seed" is RECORDED-AND-WARN
# only (never enforced) — any fixed shared prefix preserves the paired
# contrast, so a seed drift warns and records
# "shared_supervised_seed_mismatch": true instead of raising.
SHARED_SUPERVISED_MARKER = "SHARED_SUPERVISED.json"
REPORT_FILENAME = "hazard_efficacy_report.json"
PLOT_FILENAME = "hazard_efficacy_plot.png"
EVAL_RESULT_FILENAME = "eval_result.json"
HAZARD_DYNAMICS_FILENAME = "hazard_dynamics.json"
TRAIN_LOG_FILENAME = "train.log"
DEFAULT_ARMS = ("A", "B", "C")
DEFAULT_SEEDS = (1, 2, 3)
DEFAULT_OUT_DIR = "results/hazard_efficacy"
PRIMARY_MIN_POSITION_GAIN = 1.0
PRIMARY_MAX_ACCURACY_DROP = 0.01
SIGNIFICANCE_MIN_QUESTIONS = 50
PROBE_MAX_QUESTIONS = 32

_TRAIN_SCRIPT = PROJECT_ROOT / "scripts" / "train_t5_policy.py"
_SHARED_SUPERVISED_DIRNAME = "shared_supervised"
_PERSISTED_SOURCE = "persisted_artifacts"
_CORE_ARMS = frozenset(DEFAULT_ARMS)
_LOG_TAIL_LINES = 20

# R-011: the ONLY flags a smoke child receives, shared by the arm children and
# the shared supervised child. The eval-interval override suffices (the first
# validation writes best_model because best_val_reward starts at -inf); a
# per-iteration ``ppo.save_interval`` would waste full model + optimizer-state
# writes and is deliberately NOT injected.
_SMOKE_CHILD_FLAGS = ("--smoke", "ppo.eval_interval=1")

# Inclusive-threshold comparisons on IEEE-754 doubles: an intended exact tie
# (e.g. accuracy 0.60 -> 0.59 against tolerance 0.01) can differ from the
# written arithmetic by one ulp depending on the operand representations.
# The guard keeps R-006's inclusive boundaries inclusive without admitting
# any materially sub-threshold value.
_FLOAT_TOLERANCE = 1e-9

_HAZARD_STEP_MATCHING_NOTE = (
    "Hazard optimizer steps are 1-per-question (B=1) and are NOT "
    "commensurable with supervised B=N optimizer steps; compare arm B "
    "against the step-matched arm C ablation, not against supervised "
    "step counts."
)
_SMOKE_CAVEAT = (
    "Smoke-scale run: the test split is far below the significance gate, "
    "so metric deltas are plumbing/training-dynamics evidence only."
)
_PAIRED_DESIGN_CAVEAT = (
    "The shared supervised checkpoint fixes the supervised-phase RNG; "
    "seeds sample hazard/PPO variance only (paired design)."
)

ENDPOINT_DEFINITION = (
    "Primary endpoint (R-006): treatment mean correct-answer buzz position "
    "<= control mean correct-answer buzz position - 1.0 prefixes AND "
    "treatment accuracy >= control accuracy - 0.01 (absolute), replicated "
    "in >= 2 of 3 seeds (inclusive thresholds). A seed where either arm has "
    "zero correct policy buzzes is a non-success with undefined_position."
)


# ---------------------------------------------------------------------------
# Pinned exception hierarchy
# ---------------------------------------------------------------------------


class HarnessError(RuntimeError):
    """Base class for all fail-loud harness errors."""


class PreflightError(HarnessError):
    """Preflight failed before any child run (missing artifacts, bad plan)."""


class ArmControlError(HarnessError):
    """Cross-arm sidecar assertion failed (R-003); names the offending arm."""


class PartialRunError(HarnessError):
    """A run dir has outputs but no completion marker (R-013)."""


class ChildRunError(HarnessError):
    """A child training run failed or left no checkpoint (R-011)."""


class ProvenanceError(HarnessError):
    """A required provenance field is missing or invalid (R-008)."""


# ---------------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------------


def _git_output(*args: str) -> str:
    """Run a read-only git command in ``PROJECT_ROOT``; return stripped stdout.

    Parameters
    ----------
    *args : str
        Arguments appended to ``git`` (e.g. ``"rev-parse", "HEAD"``).

    Returns
    -------
    str
        The command's stdout with surrounding whitespace stripped.

    Raises
    ------
    ProvenanceError
        If the git binary is unavailable or the command exits nonzero.
    """
    label = " ".join(("git", *args))
    try:
        result = subprocess.run(
            ["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True
        )
    except OSError as exc:  # pragma: no cover - git binary missing
        raise ProvenanceError(f"{label} failed: {exc}") from exc
    if result.returncode != 0:
        raise ProvenanceError(
            f"{label} failed (exit {result.returncode}): {result.stderr.strip()!r}"
        )
    return result.stdout.strip()


def _git_sha() -> str:
    """Return the repo HEAD sha via real ``git rev-parse HEAD`` (R-008)."""
    sha = _git_output("rev-parse", "HEAD")
    if not sha:
        raise ProvenanceError("git rev-parse HEAD succeeded but printed no sha")
    return sha


def _git_dirty() -> bool:
    """Return True when ``git status --porcelain`` is non-empty (R-008)."""
    return bool(_git_output("status", "--porcelain"))


def _load_json_file(path: Path, *, error_cls: type = HarnessError) -> Any:
    """Load a JSON sidecar, raising ``error_cls`` on absence/corruption."""
    path = Path(path)
    if not path.exists():
        raise error_cls(f"required file is missing: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise error_cls(f"could not read {path}: {exc}") from exc


def _log_tail(log_path: Path, n_lines: int = _LOG_TAIL_LINES) -> str:
    """Return the last ``n_lines`` of a child log (empty-safe)."""
    log_path = Path(log_path)
    if not log_path.exists():
        return "<no log written>"
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        return f"<log unreadable: {exc}>"
    return "\n".join(lines[-n_lines:]) if lines else "<log empty>"


def _run_name(record: dict[str, Any]) -> str:
    """Canonical ``<arm>_seed<k>`` display name for a plan record."""
    return f"{record['arm']}_seed{record['seed']}"


def shared_supervised_root(out_dir: Path) -> Path:
    """Root dir the shared supervised child writes into (its override)."""
    return Path(out_dir) / _SHARED_SUPERVISED_DIRNAME


def shared_supervised_checkpoint(out_dir: Path) -> Path:
    """The branched checkpoint every arm child receives as ``--model-path``.

    The real trainer (``training/train_supervised_t5.py``) saves the best
    supervised model under ``<checkpoint_dir>/supervised/best_model``, so
    with ``supervised.checkpoint_dir=<root>`` the checkpoint lands at
    ``<root>/supervised/best_model``.
    """
    return shared_supervised_root(out_dir) / "supervised" / "best_model"


# ---------------------------------------------------------------------------
# CLI / planning
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse harness CLI arguments.

    Pinned flags (dest names): ``--config`` (str, default
    ``configs/t5_policy.yaml``), ``--smoke`` (flag), ``--seeds`` (int
    nargs=+, default [1, 2, 3]), ``--arms`` (str nargs=+, default
    ["A", "B", "C"]), ``--beta-terminal`` (float, default 1.0),
    ``--freeze-answer-head`` (flag), ``--out-dir`` (str, default
    ``results/hazard_efficacy``), ``--force`` (flag), ``--dry-run``
    (flag), ``--report-only`` (flag), ``--prune-checkpoints`` (flag),
    ``--variant`` (repeatable ``NAME:FLAGS``; ``args.variant`` is a
    list[str], default []).
    """
    parser = argparse.ArgumentParser(
        description=(
            "Paired WITH/WITHOUT/compute-control efficacy eval for the "
            "--hazard-pretrain warm-start bridge."
        ),
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/t5_policy.yaml",
        help="Path to the trainer YAML config (default: configs/t5_policy.yaml).",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke scale (t5-small); injects ppo.eval_interval=1 per child.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEEDS),
        help="Training seeds replicated per arm (default: 1 2 3).",
    )
    parser.add_argument(
        "--arms",
        type=str,
        nargs="+",
        default=list(DEFAULT_ARMS),
        help="Core arms to run: A control, B treatment, C shuffled_nll ablation.",
    )
    parser.add_argument(
        "--beta-terminal",
        type=float,
        default=1.0,
        help="Hazard-bridge terminal survival penalty threaded to arms B/C.",
    )
    parser.add_argument(
        "--freeze-answer-head",
        action="store_true",
        help="Freeze the answer head during the hazard phase (arms B/C).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=DEFAULT_OUT_DIR,
        help=f"Output tree for run dirs, report, and plot (default: {DEFAULT_OUT_DIR}).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run every planned run, complete and partial dirs included.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preflight + plan print only; zero children launched.",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Reassemble report + plot from existing run dirs; zero train/eval.",
    )
    parser.add_argument(
        "--prune-checkpoints",
        action="store_true",
        help="After eval, delete iter_*/epoch_*/training_state.pt per run.",
    )
    parser.add_argument(
        "--variant",
        action="append",
        default=None,
        help=(
            "Extra hazard B-variant as NAME:FLAGS (repeatable); FLAGS are "
            "whitespace-split and appended to a hazard-arm child argv."
        ),
    )
    args = parser.parse_args(argv)
    # ``action="append"`` with a shared mutable default would accumulate
    # across parses; normalize the None sentinel to a fresh list instead.
    args.variant = list(args.variant) if args.variant else []
    return args


def validate_flag_compatibility(args: argparse.Namespace) -> None:
    """Central flag-compatibility matrix, run right after parse (QA-005).

    Rules:

    - ``--report-only --force`` is contradictory (force demands re-running
      every child; report-only forbids launching any) and raises
      ``PreflightError``.
    - ``--report-only --dry-run`` is ALLOWED and means plan-print only:
      :func:`main` performs zero writes and zero deletions for the
      combination (``--prune-checkpoints`` included), never a destructive
      action under a zero-actions flag.
    """
    if args.report_only and args.force:
        raise PreflightError(
            "--report-only and --force are incompatible: --force re-runs "
            "every child while --report-only forbids launching any. Drop "
            "one of the two flags."
        )


def _resolve_config_path(config_path: str) -> Path:
    """Absolutize ``--config`` once at the CLI boundary (QA-012).

    A relative path is anchored to the CURRENT working directory when it
    exists there, then to ``PROJECT_ROOT``; the result is threaded to
    every child argv so children (which inherit an arbitrary CWD) and the
    harness resolve the same file. A missing config is returned
    best-effort absolute — :func:`_load_yaml_config` fails loud on it.
    """
    path = Path(config_path)
    if path.is_absolute():
        return path
    if path.exists():
        return path.resolve()
    anchored = PROJECT_ROOT / path
    if anchored.exists():
        return anchored
    return path.resolve()


def _resolved_child_base_config(config_path: str, smoke: bool) -> dict[str, Any]:
    """The child trainer's base resolved config: YAML + its smoke section.

    Reuses ``scripts.train_t5_policy.load_config_with_overrides`` (never a
    reimplementation) with a minimal namespace, i.e. the config every child
    of THIS invocation starts from before positional overrides (QA-002).
    """
    import scripts.train_t5_policy as train_t5_policy

    namespace = argparse.Namespace(
        config=str(config_path), smoke=bool(smoke), ppo_iterations=None
    )
    return train_t5_policy.load_config_with_overrides(namespace)


def _config_identity_hash(payload: Any) -> str:
    """Deterministic sha256 over a JSON-serializable payload (QA-002)."""
    canonical = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _shared_supervised_identity(args: argparse.Namespace) -> dict[str, Any]:
    """Identity the shared supervised checkpoint must match (QA-002).

    Returns ``{"config_hash", "git_sha", "model_name", "supervised_seed"}``
    derived from THIS invocation's resolved base config (YAML + smoke
    section), checkout, and build seed (``seeds[0]`` — the seed the shared
    supervised child trains with). QA-R2-3: ``supervised_seed`` is
    RECORDED-AND-WARN only, never enforced on reuse (see
    :func:`_run_shared_supervised`).
    """
    base_config = _resolved_child_base_config(str(args.config), bool(args.smoke))
    return {
        "config_hash": _config_identity_hash(
            {"config": base_config, "smoke": bool(args.smoke)}
        ),
        "git_sha": _git_sha(),
        "model_name": base_config.get("model", {}).get("model_name"),
        "supervised_seed": int(args.seeds[0]),
    }


def resolve_split_artifacts(
    *,
    smoke: bool,
    mc_path: str | None = None,
    search_dirs: list[Path] | None = None,
) -> dict[str, Path]:
    """Resolve persisted train/val/test split artifacts (R-013 preflight).

    Returns ``{"train": Path, "val": Path, "test": Path}``. Raises
    ``PreflightError`` naming the missing location and advising
    ``scripts/build_mc_dataset.py`` when the artifacts are absent.
    ``search_dirs`` overrides the standard artifact directories (tests).

    The default search order mirrors the child trainer's own resolution
    (``scripts/train_t5_policy.py::load_question_splits_with_metadata``)
    so the harness preflights the same artifacts every child will use.
    """
    if search_dirs is not None:
        candidate_dirs = [Path(d) for d in search_dirs]
    elif mc_path:
        candidate_dirs = [Path(mc_path).parent]
    elif smoke:
        candidate_dirs = [ARTIFACT_DIR / "smoke", ARTIFACT_DIR / "main", PROCESSED_DIR]
    else:
        candidate_dirs = [ARTIFACT_DIR / "main", ARTIFACT_DIR / "smoke", PROCESSED_DIR]

    if not candidate_dirs:
        raise PreflightError("resolve_split_artifacts: no search directories given")

    for base in candidate_dirs:
        paths = {
            split: Path(base) / f"{split}_dataset.json"
            for split in ("train", "val", "test")
        }
        if all(p.exists() for p in paths.values()):
            return paths

    searched = ", ".join(str(d) for d in candidate_dirs)
    build_cmd = "python scripts/build_mc_dataset.py" + (" --smoke" if smoke else "")
    raise PreflightError(
        "Persisted split artifacts (train/val/test *_dataset.json) not "
        f"found. Searched: {searched}. Run `{build_cmd}` first to build "
        "the persisted splits (scripts/build_mc_dataset.py)."
    )


def plan_runs(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Build the full arm x seed run plan up front (R-008 / R-013).

    Returns one record per run: ``{"arm": str, "seed": int, "run_dir":
    Path == <out>/<arm>_seed<k>, "hazard": bool, "argv": list[str],
    "log_path": Path == run_dir / "train.log", "variant": str | None}``.
    ``--variant NAME:FLAGS`` (split on the FIRST colon; FLAGS
    whitespace-split and appended to a hazard-arm argv) adds one extra
    B-variant arm named NAME per seed. Variant names containing path
    separators or ``..`` raise ``PreflightError``.
    """
    out_dir = Path(args.out_dir)
    shared_ckpt = shared_supervised_checkpoint(out_dir)

    arms = list(args.arms)
    seeds = [int(s) for s in args.seeds]
    if not arms or not seeds:
        raise PreflightError("plan_runs: --arms and --seeds must be non-empty")

    unknown = [arm for arm in arms if arm not in _CORE_ARMS]
    if unknown:
        raise PreflightError(
            f"Unknown core arm(s) {unknown}; core arms are A (control), "
            "B (hazard treatment), C (shuffled_nll ablation). Use "
            "--variant NAME:FLAGS for extra hazard variants."
        )

    variants: list[tuple[str, list[str]]] = []
    for spec in getattr(args, "variant", None) or []:
        name, sep, flags_str = str(spec).partition(":")
        name = name.strip()
        if not sep or not name:
            raise PreflightError(
                f"--variant must be NAME:FLAGS with a non-empty name; got {spec!r}"
            )
        if "/" in name or "\\" in name or ".." in name:
            raise PreflightError(
                f"variant name {name!r} must not contain path separators or "
                "'..' (it becomes the run-dir name <out>/<NAME>_seed<k>)"
            )
        variants.append((name, flags_str.split()))

    records: list[dict[str, Any]] = []

    def _add(
        arm: str,
        seed: int,
        *,
        hazard: bool,
        variant: str | None = None,
        **argv_kwargs: Any,
    ) -> None:
        run_dir = out_dir / f"{arm}_seed{seed}"
        argv = build_child_argv(
            arm=arm,
            seed=seed,
            run_dir=run_dir,
            shared_supervised_path=shared_ckpt,
            config_path=str(args.config),
            smoke=bool(args.smoke),
            hazard=hazard,
            **argv_kwargs,
        )
        records.append(
            {
                "arm": arm,
                "seed": seed,
                "run_dir": run_dir,
                "hazard": hazard,
                "argv": argv,
                "log_path": run_dir / TRAIN_LOG_FILENAME,
                "variant": variant,
            }
        )

    # Arms B and C share every hazard knob; C additionally carries the
    # step-matched null-signal ablation.
    hazard_knobs: dict[str, Any] = {
        "beta_terminal": args.beta_terminal,
        "freeze_answer_head": bool(args.freeze_answer_head),
    }
    for arm in arms:
        for seed in seeds:
            if arm == "A":
                _add(arm, seed, hazard=False)
            elif arm == "B":
                _add(arm, seed, hazard=True, **hazard_knobs)
            else:  # arm == "C": step-matched null-signal compute control
                _add(arm, seed, hazard=True, ablation="shuffled_nll", **hazard_knobs)

    for name, flags in variants:
        for seed in seeds:
            _add(name, seed, hazard=True, variant=name, extra_flags=flags)

    run_dirs = [rec["run_dir"] for rec in records]
    if len(set(run_dirs)) != len(run_dirs):
        raise PreflightError(
            "plan_runs: duplicate run dirs in the plan (repeated arm/seed "
            "or a variant name colliding with a core arm)"
        )
    # QA-003 (R-013): every planned argv must round-trip through the real
    # child parser BEFORE any child is launched.
    for record in records:
        _roundtrip_child_argv(record)
    return records


def build_child_argv(
    *,
    arm: str,
    seed: int,
    run_dir: Path,
    shared_supervised_path: Path,
    config_path: str,
    smoke: bool = False,
    hazard: bool = False,
    beta_terminal: float | None = None,
    freeze_answer_head: bool = False,
    ablation: str | None = None,
    extra_flags: list[str] | None = None,
) -> list[str]:
    """Assemble one child training argv LIST (R-011; never a shell string).

    Pinned shape: ``[sys.executable, <scripts/train_t5_policy.py>,
    "--config", config_path, "--skip-supervised", "--model-path",
    str(shared_supervised_path), "--seed", str(seed), ...hazard flags...,
    "supervised.checkpoint_dir=<run_dir>"]``; smoke adds ``--smoke`` and
    the positional override ``ppo.eval_interval=1`` and NEVER any
    ``ppo.save_interval`` override.

    QA-003: bare ``key=value`` tokens in ``extra_flags`` are positional
    config overrides for the child parser's trailing ``overrides`` group
    and are kept CONTIGUOUS at the argv tail (immediately before the
    checkpoint-dir override); interleaving them with later optional flags
    splits argparse's positional groups and the real child exits 2. Flag
    tokens (and their values) keep their given order in the flag zone.
    """
    if not hazard and (
        beta_terminal is not None or freeze_answer_head or ablation is not None
    ):
        raise ValueError(
            "build_child_argv: hazard knobs (beta_terminal / "
            "freeze_answer_head / ablation) require hazard=True "
            f"(arm={arm!r})"
        )

    argv: list[str] = [
        sys.executable,
        str(_TRAIN_SCRIPT),
        "--config",
        str(config_path),
        "--skip-supervised",
        "--model-path",
        str(shared_supervised_path),
        "--seed",
        str(int(seed)),
    ]
    if hazard:
        argv.append("--hazard-pretrain")
        if beta_terminal is not None:
            argv += ["--beta-terminal", str(beta_terminal)]
        if freeze_answer_head:
            argv.append("--freeze-answer-head")
        if ablation is not None:
            argv += ["--hazard-ablation", str(ablation)]
    flag_tokens: list[str] = []
    positional_overrides: list[str] = []
    for token in extra_flags or []:
        token = str(token)
        if "=" in token and not token.startswith("-"):
            positional_overrides.append(token)
        else:
            flag_tokens.append(token)
    argv += flag_tokens
    if smoke:
        argv.append("--smoke")
    # Positional key=value overrides: contiguous tail, nothing after them
    # but more positionals (QA-003).
    argv += positional_overrides
    if smoke:
        argv.append("ppo.eval_interval=1")
    argv.append(f"supervised.checkpoint_dir={run_dir}")
    return argv


def _roundtrip_child_argv(record: dict[str, Any]) -> None:
    """Preflight-validate one planned argv against the REAL child parser.

    QA-003 (R-013): orchestrators validate composed argvs against the
    target's actual parser at plan time — a variant flag typo (or a
    positional-grouping bug) must cost zero children, never the shared
    supervised phase plus nine runs. Parses ``record["argv"]`` (minus the
    ``[sys.executable, script]`` prefix) through
    ``scripts.train_t5_policy.parse_args`` and raises ``PreflightError``
    naming the run when argparse rejects it.
    """
    import scripts.train_t5_policy as train_t5_policy

    child_argv = [str(token) for token in record["argv"][2:]]
    stderr_capture = io.StringIO()
    try:
        with contextlib.redirect_stderr(stderr_capture):
            train_t5_policy.parse_args(argv=child_argv)
    except SystemExit as exc:
        raise PreflightError(
            f"Planned child argv for run {_run_name(record)} is rejected by "
            f"the real scripts/train_t5_policy.py parser (exit {exc.code}): "
            f"{stderr_capture.getvalue().strip()}\n  argv: {child_argv}"
        ) from exc


# ---------------------------------------------------------------------------
# Child execution / resume
# ---------------------------------------------------------------------------


def _run_child(argv: list[str], log_path: Path) -> int:
    """Run one child via subprocess (shell=False), tee output to log_path.

    Single injectable seam: tests monkeypatch
    ``scripts.run_hazard_efficacy._run_child``. Returns the exit code.
    The child's stderr is merged into stdout so BOTH streams land in the
    log. QA-009: the Popen lifetime is bound to a context manager and a
    tee-loop exception kills the child — a leaked child would keep
    writing into the run dir after the harness moved on.
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    argv = [str(token) for token in argv]
    with log_path.open("w", encoding="utf-8") as log_file:
        with subprocess.Popen(  # noqa: S603 - fixed argv list, shell=False
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
        ) as proc:
            assert proc.stdout is not None  # PIPE above guarantees a stream
            try:
                for line in proc.stdout:
                    log_file.write(line)
                    sys.stdout.write(line)
            except BaseException:
                proc.kill()
                raise
            return proc.wait()


def check_child_outputs(record: dict[str, Any]) -> None:
    """Fail loud when ``<run_dir>/ppo_t5/best_model`` is missing (R-011).

    Raises ``ChildRunError`` naming the run (arm, seed) and the log path.
    """
    run_dir = Path(record["run_dir"])
    best_model = run_dir / "ppo_t5" / "best_model"
    if not best_model.exists():
        raise ChildRunError(
            f"Run {_run_name(record)} (arm={record['arm']}, "
            f"seed={record['seed']}) left no PPO checkpoint at {best_model}; "
            f"inspect the child log: {record['log_path']}"
        )


def classify_run_dir(run_dir: Path, *, hazard: bool = False) -> str:
    """Classify a run dir for resume (R-013).

    Returns ``"complete"`` (RUN_COMPLETE.json + ppo_t5/best_model +
    config_used.json + split_manifest.json, plus hazard/best_model for
    hazard arms), ``"partial"`` (some outputs, incomplete), or
    ``"fresh"`` (absent/empty).
    """
    run_dir = Path(run_dir)
    if not run_dir.exists():
        return "fresh"
    if not run_dir.is_dir():
        return "partial"
    if not any(run_dir.iterdir()):
        return "fresh"

    required = [
        run_dir / RUN_COMPLETE_MARKER,
        run_dir / "ppo_t5" / "best_model",
        run_dir / "ppo_t5" / "config_used.json",
        run_dir / "ppo_t5" / "split_manifest.json",
    ]
    if hazard:
        required.append(run_dir / "hazard" / "best_model")
    return "complete" if all(p.exists() for p in required) else "partial"


def _assert_split_source(record: dict[str, Any]) -> None:
    """R-013: ``split_manifest.source`` must be persisted_artifacts."""
    manifest_path = Path(record["run_dir"]) / "ppo_t5" / "split_manifest.json"
    manifest = _load_json_file(manifest_path, error_cls=ProvenanceError)
    source = manifest.get("source") if isinstance(manifest, dict) else None
    if source != _PERSISTED_SOURCE:
        raise ProvenanceError(
            f"Run {_run_name(record)}: split_manifest.source is {source!r} "
            f"but must be '{_PERSISTED_SOURCE}'. The silent random-split "
            "fallback invalidates split provenance; rebuild the persisted "
            "artifacts and delete this run dir."
        )


def validate_resumed_run(
    record: dict[str, Any], expected: dict[str, Any]
) -> None:
    """QA-002: a resumed dir must match what THIS invocation would produce.

    Resume must never settle for internal self-consistency alone. Two
    checks against the CURRENT invocation, both raising
    ``ProvenanceError`` naming the run with delete/--force remediation:

    - ``expected["model_name"]`` (when given): the resumed
      ``config_used.json`` model name must equal the model THIS
      invocation's resolved config would train (catches a smoke t5-small
      dir silently relabeled into a t5-base — or non-smoke — report).
    - ``expected["artifact_qids"]`` (when given; ``{split: set(qids)}``):
      every qid in the resumed ``split_manifest.json`` must exist in the
      CURRENT persisted artifacts (catches dirs predating an artifact
      rebuild).
    """
    run_dir = Path(record["run_dir"])
    name = _run_name(record)

    expected_model = expected.get("model_name")
    if expected_model is not None:
        config_used = _load_json_file(
            run_dir / "ppo_t5" / "config_used.json", error_cls=ProvenanceError
        )
        actual_model = None
        if isinstance(config_used, dict):
            actual_model = config_used.get("model", {}).get("model_name")
        if actual_model != expected_model:
            raise ProvenanceError(
                f"Resumed run {name} was trained with model_name="
                f"{actual_model!r} but this invocation would train "
                f"{expected_model!r}; a stale dir cannot join the "
                "comparison. Delete the directory or pass --force to "
                "re-run everything."
            )

    artifact_qids = expected.get("artifact_qids")
    if artifact_qids:
        manifest = _load_json_file(
            run_dir / "ppo_t5" / "split_manifest.json", error_cls=ProvenanceError
        )
        for split in ("train", "val", "test"):
            available = artifact_qids.get(split)
            if available is None:
                continue
            available_set = set(available)
            manifest_qids = manifest.get(f"{split}_qids") or []
            missing = [q for q in manifest_qids if q not in available_set]
            if missing:
                raise ProvenanceError(
                    f"Resumed run {name}: {len(missing)} {split} qid(s) in "
                    f"its split_manifest.json are absent from the CURRENT "
                    f"persisted artifacts (e.g. {missing[:5]}); the dir "
                    "predates an artifact rebuild. Delete the directory or "
                    "pass --force to re-run everything."
                )


def execute_plan(
    records: list[dict[str, Any]],
    *,
    force: bool = False,
    expected_run_context: dict[str, Any] | None = None,
    smoke: bool = False,
) -> list[dict[str, Any]]:
    """Execute (or resume) every planned run sequentially (R-011/R-013).

    Complete dirs are skipped (``resumed: True``; marker git-SHA mismatch
    sets ``git_sha_mismatch: True`` and warns, never raises). Partial
    dirs raise ``PartialRunError`` with delete/--force instructions
    unless ``force``. Fresh runs call ``_run_child(argv, log_path)``;
    nonzero exit raises ``ChildRunError`` (run name, exit code, log path,
    log tail). After each run: checkpoint check, ``split_manifest.source
    == "persisted_artifacts"`` assertion, and the harness writes
    ``RUN_COMPLETE.json`` (``{"git_sha", "arm", "seed", "completed_at",
    "wall_clock_seconds", "smoke"}``; ``wall_clock_seconds`` is the
    child's elapsed run time in seconds measured around
    :func:`_run_child` — a float >= 0; ``smoke`` is the invocation's
    ``--smoke`` flag, an additive marker field — QA-R2-2: ``--report-only``
    rereads it so the report's smoke labeling comes from what the runs
    were actually trained as, never from a later invocation's flag).
    Ends by running :func:`assert_arm_control` over all records (resumed
    dirs included). Returns updated records.

    QA-001: before any child is (re-)launched into an EXISTING dir, the
    stale ``RUN_COMPLETE.json`` and ``eval_result.json`` are unlinked
    (invalidate-before-mutate) — a crash mid-child then leaves an honest
    partial dir instead of a half-trained checkpoint masquerading as
    complete under old provenance.

    Parameters (additive)
    ---------------------
    expected_run_context : dict or None, keyword-only
        When given, every RESUMED dir is additionally validated against
        this invocation via :func:`validate_resumed_run` (QA-002; keys
        ``model_name`` and/or ``artifact_qids``). Default ``None`` keeps
        the pre-existing behavior for direct callers.
    smoke : bool, keyword-only
        The invocation's ``--smoke`` flag, persisted verbatim into each
        fresh run's ``RUN_COMPLETE.json`` marker (QA-R2-2). Default
        ``False`` keeps the pre-existing behavior for direct callers.
    """
    current_sha = _git_sha()
    total = len(records)

    for index, record in enumerate(records, start=1):
        run_dir = Path(record["run_dir"])
        log_path = Path(record["log_path"])
        name = _run_name(record)
        state = classify_run_dir(run_dir, hazard=bool(record.get("hazard")))

        if state == "complete" and not force:
            print(f"[{index}/{total}] arm={record['arm']} seed={record['seed']} "
                  "resumed (complete run dir found, skipping child)")
            record["resumed"] = True
            marker = _load_json_file(
                run_dir / RUN_COMPLETE_MARKER, error_cls=PartialRunError
            )
            mismatch = marker.get("git_sha") != current_sha
            record["git_sha_mismatch"] = bool(mismatch)
            if mismatch:
                print(
                    f"WARNING: resumed run {name} was completed at git sha "
                    f"{marker.get('git_sha')!r} but the current checkout is "
                    f"{current_sha!r}; recording the drift in provenance "
                    "(not fatal)."
                )
            _assert_split_source(record)
            if expected_run_context:
                validate_resumed_run(record, expected_run_context)
            continue

        if state == "partial" and not force:
            raise PartialRunError(
                f"Run dir {run_dir} has outputs but no valid completion "
                f"state ({RUN_COMPLETE_MARKER} + checkpoints + sidecars). "
                f"Delete the directory to re-run {name} fresh, or pass "
                "--force to re-run everything."
            )

        # QA-001: invalidate-before-mutate — stale completion/eval markers
        # must never survive into (or beyond) a failed re-run of an
        # existing dir.
        if run_dir.exists():
            for stale_name in (RUN_COMPLETE_MARKER, EVAL_RESULT_FILENAME):
                stale = run_dir / stale_name
                if stale.exists():
                    stale.unlink()

        print(f"[{index}/{total}] arm={record['arm']} seed={record['seed']} started")
        start = time.monotonic()
        exit_code = _run_child(record["argv"], log_path)
        elapsed = time.monotonic() - start
        if exit_code != 0:
            raise ChildRunError(
                f"Child run {name} (arm={record['arm']}, "
                f"seed={record['seed']}) failed with exit code {exit_code}; "
                f"log: {log_path}\n--- log tail ---\n{_log_tail(log_path)}"
            )

        check_child_outputs(record)
        if record.get("hazard") and not (run_dir / "hazard" / "best_model").exists():
            raise ChildRunError(
                f"Hazard run {name} left no hazard checkpoint at "
                f"{run_dir / 'hazard' / 'best_model'}; inspect the child "
                f"log: {log_path}"
            )
        _assert_split_source(record)

        marker = {
            "git_sha": current_sha,
            "arm": record["arm"],
            "seed": record["seed"],
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "wall_clock_seconds": float(elapsed),
            # QA-R2-2: persist the invocation's smoke flag so --report-only
            # can label the report from the runs' real provenance.
            "smoke": bool(smoke),
        }
        save_json(run_dir / RUN_COMPLETE_MARKER, marker)
        record["resumed"] = False
        record["git_sha_mismatch"] = False
        print(
            f"[{index}/{total}] arm={record['arm']} seed={record['seed']} "
            f"finished, elapsed {elapsed:.1f}s"
        )

    assert_arm_control(records)
    return records


# ---------------------------------------------------------------------------
# Arm control / provenance (R-003 / R-008)
# ---------------------------------------------------------------------------


def _flatten_dotted(obj: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten nested dicts into dotted-path leaves for key-wise diffing."""
    flat: dict[str, Any] = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(value, dict):
                flat.update(_flatten_dotted(value, path))
            else:
                flat[path] = value
    return flat


def _is_exempt_config_key(path: str) -> bool:
    """R-003 exemptions: hazard block, top-level seed, *.checkpoint_dir."""
    parts = path.split(".")
    return path == "seed" or parts[0] == "hazard" or parts[-1] == "checkpoint_dir"


def assert_arm_control(records: list[dict[str, Any]]) -> None:
    """Diff every run's config_used.json / split_manifest.json (R-003).

    All config keys must be equal across runs EXCEPT the top-level
    ``hazard`` block, top-level ``seed``, and any key path whose last
    component is ``checkpoint_dir``. All split manifests must carry
    identical train/val/test qids. Any mismatch raises
    ``ArmControlError`` naming the offending arm (and key).
    """
    if not records:
        return

    loaded: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = []
    for record in records:
        run_dir = Path(record["run_dir"])
        config = _load_json_file(
            run_dir / "ppo_t5" / "config_used.json", error_cls=ArmControlError
        )
        manifest = _load_json_file(
            run_dir / "ppo_t5" / "split_manifest.json", error_cls=ArmControlError
        )
        flat = {
            key: value
            for key, value in _flatten_dotted(config).items()
            if not _is_exempt_config_key(key)
        }
        loaded.append((record, flat, manifest))

    ref_record, ref_flat, ref_manifest = loaded[0]
    ref_name = _run_name(ref_record)

    for record, flat, manifest in loaded[1:]:
        name = _run_name(record)
        for key in sorted(set(ref_flat) | set(flat)):
            if key not in flat:
                raise ArmControlError(
                    f"Arm-control violation (R-003): run {name} (arm="
                    f"{record['arm']}) config_used.json is missing key "
                    f"'{key}' present in {ref_name}"
                )
            if key not in ref_flat:
                raise ArmControlError(
                    f"Arm-control violation (R-003): run {name} (arm="
                    f"{record['arm']}) config_used.json has unexpected extra "
                    f"key '{key}' absent from {ref_name}"
                )
            if flat[key] != ref_flat[key]:
                raise ArmControlError(
                    f"Arm-control violation (R-003): run {name} (arm="
                    f"{record['arm']}) differs from {ref_name} at config key "
                    f"'{key}': {flat[key]!r} != {ref_flat[key]!r}"
                )
        for split in ("train", "val", "test"):
            key = f"{split}_qids"
            if manifest.get(key) != ref_manifest.get(key):
                raise ArmControlError(
                    f"Arm-control violation (R-003): run {name} (arm="
                    f"{record['arm']}) split_manifest.json {key} differ "
                    f"from {ref_name}; all arms must train/evaluate on "
                    "identical splits"
                )


def write_supervised_split_manifest(
    supervised_ckpt_dir: Path, manifest: dict[str, Any]
) -> Path:
    """Persist the split manifest next to the shared supervised ckpt (R-008).

    Asserts test qids are disjoint from supervised-train qids
    (``ProvenanceError`` on overlap) and writes ``split_manifest.json``
    into ``supervised_ckpt_dir``. Returns the written path.
    """
    if not isinstance(manifest, dict):
        raise ProvenanceError(
            "write_supervised_split_manifest: manifest must be a dict, got "
            f"{type(manifest).__name__}"
        )
    train_qids = manifest.get("train_qids")
    test_qids = manifest.get("test_qids")
    if not isinstance(train_qids, list) or not isinstance(test_qids, list):
        raise ProvenanceError(
            "write_supervised_split_manifest: manifest must carry "
            "train_qids and test_qids lists"
        )
    overlap = sorted(set(train_qids) & set(test_qids))
    if overlap:
        raise ProvenanceError(
            "Supervised split manifest violates train/test disjointness "
            f"(R-008): {len(overlap)} qid(s) appear in both, e.g. "
            f"{overlap[:5]}; the supervised phase must never see test qids."
        )
    out_dir = Path(supervised_ckpt_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "split_manifest.json"
    save_json(out_path, manifest)
    return out_path


def collect_provenance(config_used: dict[str, Any]) -> dict[str, Any]:
    """Assemble one run's provenance block (R-008), complete or fail-loud.

    Returns ``{"model_name", "seed", "device", "git_sha", "git_dirty",
    "torch_version", "platform"}`` using real ``git rev-parse HEAD`` /
    ``git status --porcelain``. A missing source field raises
    ``ProvenanceError`` (no report may be written).
    """
    try:
        model_cfg = config_used["model"]
        model_name = model_cfg["model_name"]
        device = model_cfg["device"]
        seed = config_used["seed"]
    except (KeyError, TypeError) as exc:
        raise ProvenanceError(
            f"config_used.json is missing a required provenance field: {exc}"
        ) from exc
    if not model_name or not device or seed is None:
        raise ProvenanceError(
            "config_used.json carries an empty provenance field "
            f"(model_name={model_name!r}, device={device!r}, seed={seed!r})"
        )

    try:
        import torch

        torch_version = str(torch.__version__)
    except ImportError as exc:  # pragma: no cover - torch is a core dep
        raise ProvenanceError(f"torch is unavailable: {exc}") from exc
    platform_str = platform.platform()
    if not torch_version or not platform_str:
        raise ProvenanceError(
            "provenance is incomplete: torch_version/platform resolved empty"
        )

    return {
        "model_name": model_name,
        "seed": seed,
        "device": device,
        "git_sha": _git_sha(),
        "git_dirty": _git_dirty(),
        "torch_version": torch_version,
        "platform": platform_str,
    }


# ---------------------------------------------------------------------------
# Evaluation (R-005 / R-012 / R-014)
# ---------------------------------------------------------------------------


def evaluate_run(
    record: dict[str, Any], test_questions: list, config: dict[str, Any]
) -> dict[str, Any]:
    """Evaluate one run through ``evaluate_t5_policy`` ONLY (R-005/R-012).

    Calls the module-level ``evaluate_t5_policy`` (imported from
    ``scripts.compare_policies``) with KEYWORD arguments only, including
    ``checkpoint_path=<run_dir>/ppo_t5/best_model``, the shared
    ``test_questions``, and ``return_runs=True``. Writes
    ``<run_dir>/eval_result.json`` = the eval payload passed through
    PLUS ``{"arm", "seed", "policy_buzz_rate", "forced_commit_rate",
    "n_correct_policy_buzzes", "mean_correct_buzz_position"}`` derived
    from the per-question ``runs`` records (R-014). Returns the enriched
    payload.

    Parameters
    ----------
    record : dict
        One plan record (``arm``, ``seed``, ``run_dir`` are read).
    test_questions : list
        The shared held-out test split (identical object across runs).
    config : dict
        Eval context. Optional keys threaded through to the shared eval
        entry point: ``reference_questions`` (train-split corpus for the
        env reward helper), ``test_set_source`` (provenance label,
        default ``"persisted_artifacts"``), ``config`` (the loaded YAML
        config dict). All other keys are ignored, so tests may pass a
        bare ``{}``.
    """
    run_dir = Path(record["run_dir"])
    checkpoint_path = run_dir / "ppo_t5" / "best_model"
    context = config if isinstance(config, dict) else {}

    payload = evaluate_t5_policy(
        checkpoint_path=str(checkpoint_path),
        test_questions=test_questions,
        reference_questions=context.get("reference_questions", []),
        test_set_source=context.get("test_set_source", _PERSISTED_SOURCE),
        config=context.get("config", {}),
        return_runs=True,
    )

    runs = payload.get("runs") if isinstance(payload, dict) else None
    if not isinstance(runs, list):
        raise HarnessError(
            f"evaluate_t5_policy returned no per-question 'runs' records for "
            f"{_run_name(record)}; return_runs=True is required (R-002/R-005)"
        )

    n = len(runs)
    n_buzzed = sum(1 for r in runs if r.get("buzzed"))
    correct_buzzes = [r for r in runs if r.get("buzzed") and r.get("correct")]
    correct_positions = [
        r["buzz_position"]
        for r in correct_buzzes
        if r.get("buzz_position") is not None
    ]
    n_correct = len(correct_buzzes)

    enriched = dict(payload)
    enriched.update(
        {
            "arm": record["arm"],
            "seed": record["seed"],
            "policy_buzz_rate": (n_buzzed / n) if n else 0.0,
            "forced_commit_rate": ((n - n_buzzed) / n) if n else 0.0,
            "n_correct_policy_buzzes": int(n_correct),
            "mean_correct_buzz_position": (
                float(np.mean(correct_positions)) if correct_positions else None
            ),
        }
    )
    # R-014: persisted IMMEDIATELY, before any later run can fail.
    save_json(run_dir / EVAL_RESULT_FILENAME, enriched)
    return enriched


def evaluate_all_runs(
    records: list[dict[str, Any]], test_questions: list, config: dict[str, Any]
) -> list[dict[str, Any]]:
    """Evaluate every run: exactly one ``evaluate_t5_policy`` call each,
    identical test split and kwargs (except checkpoint path) across calls.
    Per-run ``eval_result.json`` files already written are never deleted
    by a later failure (R-014)."""
    results: list[dict[str, Any]] = []
    total = len(records)
    for index, record in enumerate(records, start=1):
        print(f"[eval {index}/{total}] arm={record['arm']} seed={record['seed']}")
        results.append(evaluate_run(record, test_questions, config))
    return results


# ---------------------------------------------------------------------------
# Endpoint / significance (R-006 / R-007)
# ---------------------------------------------------------------------------


def compute_primary_endpoint(per_seed: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute the pre-committed primary endpoint (R-006).

    Parameters
    ----------
    per_seed : list of dict
        ``[{"seed": int,
            "control": {"mean_correct_buzz_position": float | None,
                         "accuracy": float,
                         "n_correct_policy_buzzes": int},
            "treatment": {same keys}}]``

    Returns
    -------
    dict
        ``{"success": bool, "n_seeds": int, "n_seeds_replicated": int,
        "per_seed": [{"seed", "seed_success", "undefined_position",
        "control_position", "treatment_position", "control_accuracy",
        "treatment_accuracy"}]}``. Seed success iff treatment position
        <= control - 1.0 AND treatment accuracy >= control - 0.01
        (inclusive thresholds); a zero-correct-buzz arm makes the seed a
        non-success with ``undefined_position: True``. Overall success
        iff >= 2 seeds replicate. Empty input raises ``ValueError``.
    """
    if not per_seed:
        raise ValueError("compute_primary_endpoint: per_seed is empty")

    seed_records: list[dict[str, Any]] = []
    for entry in per_seed:
        try:
            seed = entry["seed"]
            control = entry["control"]
            treatment = entry["treatment"]
            control_pos = control["mean_correct_buzz_position"]
            treatment_pos = treatment["mean_correct_buzz_position"]
            control_acc = control["accuracy"]
            treatment_acc = treatment["accuracy"]
            control_n = control["n_correct_policy_buzzes"]
            treatment_n = treatment["n_correct_policy_buzzes"]
        except (KeyError, TypeError) as exc:
            raise ValueError(
                f"compute_primary_endpoint: malformed per-seed record: {exc}"
            ) from exc

        undefined = (
            control_n == 0
            or treatment_n == 0
            or control_pos is None
            or treatment_pos is None
        )
        if undefined:
            seed_success = False
        else:
            # Inclusive thresholds; _FLOAT_TOLERANCE guards exact-tie
            # boundaries against one-ulp representation error.
            position_ok = (
                treatment_pos
                <= control_pos - PRIMARY_MIN_POSITION_GAIN + _FLOAT_TOLERANCE
            )
            accuracy_ok = (
                treatment_acc
                >= control_acc - PRIMARY_MAX_ACCURACY_DROP - _FLOAT_TOLERANCE
            )
            seed_success = bool(position_ok and accuracy_ok)

        seed_records.append(
            {
                "seed": seed,
                "seed_success": bool(seed_success),
                "undefined_position": bool(undefined),
                "control_position": control_pos,
                "treatment_position": treatment_pos,
                "control_accuracy": control_acc,
                "treatment_accuracy": treatment_acc,
            }
        )

    n_replicated = sum(1 for rec in seed_records if rec["seed_success"])
    return {
        "success": n_replicated >= 2,
        "n_seeds": len(seed_records),
        "n_seeds_replicated": n_replicated,
        "per_seed": seed_records,
    }


def _validate_arm_qids(
    sq_by_seed: dict[int, dict[str, float]], arm_label: str
) -> set[str]:
    """Return the arm's qid set; raise ValueError on per-seed drift."""
    if not sq_by_seed:
        raise ValueError(f"compute_significance: {arm_label} arm map is empty")
    reference: set[str] | None = None
    reference_seed: int | None = None
    for seed, qid_map in sq_by_seed.items():
        if not isinstance(qid_map, dict) or not qid_map:
            raise ValueError(
                f"compute_significance: {arm_label} arm seed {seed} carries "
                "no per-qid S_q values"
            )
        qids = set(qid_map)
        if reference is None:
            reference, reference_seed = qids, seed
        elif qids != reference:
            missing = sorted(reference - qids)[:5]
            extra = sorted(qids - reference)[:5]
            raise ValueError(
                f"compute_significance: {arm_label} arm qids differ across "
                f"seeds ({seed} vs {reference_seed}); missing={missing}, "
                f"extra={extra}"
            )
    assert reference is not None
    return reference


def compute_significance(
    control_sq_by_seed: dict[int, dict[str, float]],
    treatment_sq_by_seed: dict[int, dict[str, float]],
    *,
    n_bootstrap: int = 1000,
    seed: int = 13,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Paired-by-qid bootstrap CI on per-question S_q deltas (R-007).

    Pooling: average S_q over seeds within each arm per qid -> per-qid
    delta (treatment - control) -> deterministic bootstrap over qids
    via ``bootstrap_ci`` IMPORTED from ``evaluation.controls`` at module
    level (tests identity-pin ``run_hazard_efficacy.bootstrap_ci is
    evaluation.controls.bootstrap_ci`` — never a reimplementation), with
    the fixed ``seed``.
    Returns ``{"mean_delta", "ci_low", "ci_high", "n_questions",
    "significance_evaluable", "significance"}`` where ``significance``
    is ``"paired_bootstrap_ci"`` when n >= 50 and
    ``"not_evaluable_at_this_scale"`` otherwise. Mismatched or missing
    qids (across arms, or across seeds within an arm) raise
    ``ValueError``; empty input raises ``ValueError``.
    """
    if not control_sq_by_seed or not treatment_sq_by_seed:
        raise ValueError("compute_significance: empty per-seed S_q input")

    control_qids = _validate_arm_qids(control_sq_by_seed, "control")
    treatment_qids = _validate_arm_qids(treatment_sq_by_seed, "treatment")
    if control_qids != treatment_qids:
        missing = sorted(control_qids - treatment_qids)[:5]
        extra = sorted(treatment_qids - control_qids)[:5]
        raise ValueError(
            "compute_significance: control/treatment qid sets differ; "
            f"control-only={missing}, treatment-only={extra}"
        )

    qids = sorted(control_qids)
    deltas: list[float] = []
    for qid in qids:
        control_mean = float(
            np.mean([qid_map[qid] for qid_map in control_sq_by_seed.values()])
        )
        treatment_mean = float(
            np.mean([qid_map[qid] for qid_map in treatment_sq_by_seed.values()])
        )
        deltas.append(treatment_mean - control_mean)

    ci_low, ci_high = bootstrap_ci(
        deltas, n_samples=n_bootstrap, alpha=alpha, seed=seed
    )
    n_questions = len(qids)
    evaluable = n_questions >= SIGNIFICANCE_MIN_QUESTIONS
    return {
        "mean_delta": float(np.mean(deltas)),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "n_questions": n_questions,
        "significance_evaluable": bool(evaluable),
        "significance": (
            "paired_bootstrap_ci" if evaluable else "not_evaluable_at_this_scale"
        ),
    }


# ---------------------------------------------------------------------------
# Stop-probability probe / hazard dynamics (R-010b)
# ---------------------------------------------------------------------------


def select_probe_questions(train_questions: list) -> list:
    """Deterministic probe sample: first ``min(32, len(train))`` train
    questions in split order (R-010)."""
    return list(train_questions[:PROBE_MAX_QUESTIONS])


def stop_prob_probe(model: Any, questions: list) -> list[list[float]]:
    """Probe P(BUZZ) per prefix position for each question (R-010b).

    Runs the model in ``eval()`` mode under ``torch.no_grad()`` (dropout
    off => deterministic). Returns one list per question of length
    ``T_q`` (its prefix count), each value a probability in [0, 1].

    Each prefix is rendered in the canonical policy input format
    ``"CLUES: <prefix> | CHOICES: (1) opt1 (2) opt2 ..."`` (the same
    format used by the hazard phase — ``training/hazard_pretrain.py`` —
    and the PPO text observations), and P(BUZZ) is the wait head's
    softmax mass on the BUZZ column (index 1). Zero-prefix questions are
    skipped, mirroring the hazard training loop.
    """
    import torch

    was_training = bool(getattr(model, "training", False))
    model.eval()
    per_question: list[list[float]] = []
    try:
        with torch.no_grad():
            for question in questions:
                prefixes = list(question.cumulative_prefixes)
                if not prefixes:
                    continue
                choices_text = " ".join(
                    f"({i + 1}) {opt}" for i, opt in enumerate(question.options)
                )
                texts = [
                    f"CLUES: {prefix} | CHOICES: {choices_text}"
                    for prefix in prefixes
                ]
                wait_logits, _, _ = model(texts)  # [T, 2]
                buzz_probs = torch.softmax(wait_logits, dim=-1)[:, 1]
                per_question.append(
                    [float(p) for p in buzz_probs.detach().cpu().tolist()]
                )
    finally:
        if was_training:
            model.train()
    return per_question


def _expected_buzz_time(rows: list[list[float]]) -> float:
    """Mean expected buzz position over questions (1-indexed positions).

    Per question with stop probs ``p_1..p_T``: E[T] = sum_t t * p_t *
    prod_{s<t}(1 - p_s) + T * prod_s(1 - p_s) — the never-buzz survival
    mass commits at the final position (forced commit).
    """
    times: list[float] = []
    for row in rows:
        probs = np.asarray(row, dtype=np.float64)
        stay = 1.0 - probs
        # Probability of REACHING position t (before stopping there).
        reach = np.concatenate(([1.0], np.cumprod(stay)[:-1]))
        stop_mass = reach * probs
        survive_all = float(np.prod(stay))
        positions = np.arange(1, len(probs) + 1, dtype=np.float64)
        times.append(float((positions * stop_mass).sum() + len(probs) * survive_all))
    return float(np.mean(times))


def build_hazard_dynamics(
    before: list[list[float]],
    after: list[list[float]],
    hazard_history: dict[str, Any],
) -> dict[str, Any]:
    """Summarize the stop-prob shift + hazard-loss halves (R-010b).

    Returns ``{"per_position_mean_before": [float], "per_position_mean_after":
    [float] (length == max T among probed questions; per-position mean over
    the questions that reach that position), "expected_buzz_time_before":
    float, "expected_buzz_time_after": float, "expected_buzz_time_delta":
    float == after - before, "first_half_mean_loss": float,
    "second_half_mean_loss": float}`` where the loss halves are the means
    of the first and second halves of ``hazard_history["steps"][*]["loss"]``
    in step order. Empty probes or empty ``steps`` raise ``ValueError``.
    """
    if not before or not after:
        raise ValueError("build_hazard_dynamics: empty probe results")
    if len(before) != len(after):
        raise ValueError(
            "build_hazard_dynamics: before/after probed different question "
            f"counts ({len(before)} vs {len(after)})"
        )
    lengths = [len(row) for row in before]
    if lengths != [len(row) for row in after]:
        raise ValueError(
            "build_hazard_dynamics: before/after per-question prefix counts "
            "differ; both probes must cover the same questions"
        )
    if any(length == 0 for length in lengths):
        raise ValueError("build_hazard_dynamics: a probed question has 0 prefixes")

    steps = hazard_history.get("steps") if isinstance(hazard_history, dict) else None
    if not steps:
        raise ValueError(
            "build_hazard_dynamics: hazard_history has no optimizer steps"
        )
    try:
        losses = [float(step["loss"]) for step in steps]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"build_hazard_dynamics: malformed hazard_history step: {exc}"
        ) from exc

    mid = len(losses) // 2
    if mid == 0:  # single step: both halves are that step
        first_half, second_half = losses, losses
    else:
        first_half, second_half = losses[:mid], losses[mid:]

    max_t = max(lengths)

    def _per_position_means(rows: list[list[float]]) -> list[float]:
        return [
            float(np.mean([row[t] for row in rows if len(row) > t]))
            for t in range(max_t)
        ]

    time_before = _expected_buzz_time(before)
    time_after = _expected_buzz_time(after)
    return {
        "per_position_mean_before": _per_position_means(before),
        "per_position_mean_after": _per_position_means(after),
        "expected_buzz_time_before": float(time_before),
        "expected_buzz_time_after": float(time_after),
        "expected_buzz_time_delta": float(time_after - time_before),
        "first_half_mean_loss": float(np.mean(first_half)),
        "second_half_mean_loss": float(np.mean(second_half)),
    }


def probe_and_write_hazard_dynamics(
    supervised_ckpt: str,
    hazard_ckpt: str,
    questions: list,
    hazard_history_path: Path,
    out_path: Path,
    *,
    before_cache: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load both checkpoints, probe them, and persist the R-010b block.

    Writes :func:`build_hazard_dynamics` output as JSON to ``out_path``
    (``<run_dir>/hazard_dynamics.json`` in the pipeline) and returns it.

    Parameters (additive)
    ---------------------
    before_cache : dict or None, keyword-only
        Optional cross-call memo (QA-007: shared-prefix work stops at the
        branch point). When given, the supervised "before" probe is
        computed once and stored under key ``"before"``; later calls with
        the SAME dict reuse it instead of re-loading and re-probing the
        supervised checkpoint (6x redundant across hazard runs). Callers
        own cache validity: share one dict only across calls with the
        same ``supervised_ckpt`` and ``questions``. Default ``None``
        keeps the uncached behavior.
    """
    from models.t5_policy import T5PolicyModel  # heavy transformers import

    before = None if before_cache is None else before_cache.get("before")
    if before is None:
        supervised_model = T5PolicyModel.load_pretrained(str(supervised_ckpt))
        before = stop_prob_probe(supervised_model, questions)
        del supervised_model
        if before_cache is not None:
            before_cache["before"] = before

    hazard_model = T5PolicyModel.load_pretrained(str(hazard_ckpt))
    after = stop_prob_probe(hazard_model, questions)
    del hazard_model

    history = _load_json_file(Path(hazard_history_path), error_cls=ProvenanceError)
    block = build_hazard_dynamics(before, after, history)
    save_json(Path(out_path), block)
    return block


# ---------------------------------------------------------------------------
# Report / plot / prune (R-009 / R-014)
# ---------------------------------------------------------------------------


def prune_run_checkpoints(run_dir: Path) -> None:
    """Reclaim disk for one run (R-014, ``--prune-checkpoints``).

    Only when ``<run_dir>/eval_result.json`` exists: delete every
    ``iter_*``/``epoch_*`` directory and every ``training_state.pt``
    file under ``run_dir``, keeping ``best_model/`` model files,
    sidecars, ``history.json`` and ``eval_result.json``. When
    ``eval_result.json`` is absent it refuses (raises ``HarnessError``)
    without deleting anything.

    QA-010: recursive deletes resolve-and-contain against the intended
    root. The walk never follows directory symlinks (``pathlib.rglob``
    on <=3.12 does, so a symlinked parent could alias content OUTSIDE
    the run dir and cycles could loop the scan); symlink candidates are
    skipped outright; and every candidate must resolve INSIDE
    ``run_dir.resolve()`` before deletion.
    """
    run_dir = Path(run_dir)
    if not (run_dir / EVAL_RESULT_FILENAME).exists():
        raise HarnessError(
            f"Refusing to prune {run_dir}: {EVAL_RESULT_FILENAME} does not "
            "exist yet, so the report would not be regenerable. Evaluate "
            "the run first (nothing was deleted)."
        )

    root = run_dir.resolve()

    def _contained(path: Path) -> bool:
        resolved = path.resolve()
        return resolved == root or resolved.is_relative_to(root)

    prunable_dirs: list[Path] = []
    state_files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        for name in list(dirnames):
            full = Path(dirpath) / name
            if full.is_symlink():
                # Never descend into — or delete through — a dir symlink.
                dirnames.remove(name)
                continue
            if fnmatch.fnmatch(name, "iter_*") or fnmatch.fnmatch(name, "epoch_*"):
                prunable_dirs.append(full)
                # Its contents die with the rmtree below; don't walk in.
                dirnames.remove(name)
        for name in filenames:
            if name == "training_state.pt":
                state_files.append(Path(dirpath) / name)

    for path in prunable_dirs:
        if not path.is_symlink() and path.is_dir() and _contained(path):
            shutil.rmtree(path)
    for state_file in state_files:
        if (
            not state_file.is_symlink()
            and state_file.is_file()
            and _contained(state_file)
        ):
            state_file.unlink()


def write_plot(report: dict[str, Any], out_dir: Path) -> Path:
    """Write ``<out>/hazard_efficacy_plot.png`` headlessly; returns the path.

    Matplotlib is imported lazily inside this function with the Agg
    backend selected before pyplot, and the figure is persisted via
    ``savefig`` only.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise HarnessError(
            f"matplotlib is required to write the efficacy plot: {exc}"
        ) from exc

    runs = report.get("runs", []) or []
    arms = sorted({run["arm"] for run in runs})

    def _arm_mean(arm: str, key: str) -> float:
        """Mean of one numeric report key over an arm's runs (0.0 if none)."""
        values = []
        for run in runs:
            if run.get("arm") != arm:
                continue
            value = run.get(key)
            # bool is an int subclass; a flag must never plot as 0/1.
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            values.append(float(value))
        return float(np.mean(values)) if values else 0.0

    fig, (ax_pos, ax_acc) = plt.subplots(1, 2, figsize=(10.0, 4.0))

    positions = [_arm_mean(arm, "mean_correct_buzz_position") for arm in arms]
    ax_pos.bar(arms, positions, color="steelblue")
    ax_pos.set_title("Mean correct-buzz position (lower = earlier)")
    ax_pos.set_xlabel("arm")
    ax_pos.set_ylabel("prefix position")

    accuracies = [_arm_mean(arm, "accuracy") for arm in arms]
    ax_acc.bar(arms, accuracies, color="darkorange")
    ax_acc.set_title("Policy accuracy")
    ax_acc.set_xlabel("arm")
    ax_acc.set_ylabel("accuracy")
    ax_acc.set_ylim(0.0, 1.0)

    scale = report.get("scale", {})
    fig.suptitle(
        "Hazard-pretrain efficacy — "
        f"{scale.get('model_name', '?')} (n_test={scale.get('n_test', '?')})"
    )
    fig.tight_layout()

    plot_path = Path(out_dir) / PLOT_FILENAME
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    return plot_path


def _endpoint_arm_view(eval_result: dict[str, Any]) -> dict[str, Any]:
    """The three eval fields :func:`compute_primary_endpoint` reads per arm."""
    return {
        "mean_correct_buzz_position": eval_result.get("mean_correct_buzz_position"),
        "accuracy": eval_result.get("accuracy"),
        "n_correct_policy_buzzes": eval_result.get("n_correct_policy_buzzes", 0),
    }


def _endpoint_pairs_from_evals(
    eval_by_run: dict[tuple[str, int], dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build R-006 per-seed control(A)/treatment(B) pairs from eval files."""
    seeds = sorted(
        {
            seed
            for (arm, seed) in eval_by_run
            if arm == "A" and ("B", seed) in eval_by_run
        }
    )
    return [
        {
            "seed": seed,
            "control": _endpoint_arm_view(eval_by_run[("A", seed)]),
            "treatment": _endpoint_arm_view(eval_by_run[("B", seed)]),
        }
        for seed in seeds
    ]


def _sq_by_seed_for_arm(
    eval_by_run: dict[tuple[str, int], dict[str, Any]], arm: str
) -> dict[int, dict[str, float]]:
    """Per-seed {qid: S_q} maps for one arm from eval_result runs records."""
    by_seed: dict[int, dict[str, float]] = {}
    for (run_arm, seed), payload in eval_by_run.items():
        if run_arm != arm:
            continue
        runs = payload.get("runs")
        if not isinstance(runs, list) or not runs:
            continue
        qid_map: dict[str, float] = {}
        for run in runs:
            qid = run.get("qid")
            sq = run.get("sq")
            if qid is not None and sq is not None:
                qid_map[str(qid)] = float(sq)
        if qid_map:
            by_seed[seed] = qid_map
    return by_seed


_UNEVALUABLE_SIGNIFICANCE: dict[str, Any] = {
    "mean_delta": None,
    "ci_low": None,
    "ci_high": None,
    "n_questions": 0,
    "significance_evaluable": False,
    "significance": "not_evaluable_at_this_scale",
}


def _read_optional_json(path: Path) -> Any:
    """Load a sidecar that may legitimately be absent; ``None`` when missing."""
    path = Path(path)
    if not path.exists():
        return None
    return _load_json_file(path, error_cls=ProvenanceError)


def _read_run_sidecars(
    run_records: list[dict[str, Any]],
) -> tuple[
    dict[tuple[str, int], dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
]:
    """Read every run's required sidecars into the report's raw inputs.

    Parameters
    ----------
    run_records : list of dict
        Plan records; ``arm``, ``seed``, ``run_dir``, ``resumed`` and
        ``git_sha_mismatch`` are read. Must be non-empty.

    Returns
    -------
    tuple
        ``(eval_by_run, report_runs, first_config, first_manifest)`` — the
        ``(arm, seed) -> eval payload`` index, the per-run report rows
        (provenance included), and the FIRST run's ``config_used.json`` /
        ``split_manifest.json`` (the scale block's source; arm control has
        already proven every run agrees).
    """
    eval_by_run: dict[tuple[str, int], dict[str, Any]] = {}
    report_runs: list[dict[str, Any]] = []
    first_config: dict[str, Any] | None = None
    first_manifest: dict[str, Any] | None = None

    for record in run_records:
        run_dir = Path(record["run_dir"])
        config_used = _load_json_file(
            run_dir / "ppo_t5" / "config_used.json", error_cls=ProvenanceError
        )
        manifest = _load_json_file(
            run_dir / "ppo_t5" / "split_manifest.json", error_cls=ProvenanceError
        )
        eval_result = _load_json_file(
            run_dir / EVAL_RESULT_FILENAME, error_cls=ProvenanceError
        )
        if first_config is None:
            first_config, first_manifest = config_used, manifest

        eval_by_run[(record["arm"], record["seed"])] = eval_result
        report_runs.append(
            {
                "arm": record["arm"],
                "seed": record["seed"],
                "resumed": bool(record.get("resumed", False)),
                "git_sha_mismatch": bool(record.get("git_sha_mismatch", False)),
                "policy_buzz_rate": eval_result.get("policy_buzz_rate"),
                "forced_commit_rate": eval_result.get("forced_commit_rate"),
                "ece": eval_result.get("ece"),
                "brier": eval_result.get("brier"),
                "accuracy": eval_result.get("accuracy"),
                "mean_sq": eval_result.get("mean_sq"),
                "avg_buzz_pos": eval_result.get("avg_buzz_pos"),
                "n_correct_policy_buzzes": eval_result.get("n_correct_policy_buzzes"),
                "mean_correct_buzz_position": eval_result.get(
                    "mean_correct_buzz_position"
                ),
                "n_questions": eval_result.get("n_questions"),
                "provenance": collect_provenance(config_used),
            }
        )

    assert first_config is not None and first_manifest is not None
    return eval_by_run, report_runs, first_config, first_manifest


def _build_scale(
    config_used: dict[str, Any], manifest: dict[str, Any], out_dir: Path
) -> dict[str, Any]:
    """Scale block: model, split sizes, PPO iterations, on-disk footprint.

    ``disk_usage_bytes`` is measured BEFORE the report and plot are written,
    so it reflects the run tree the report describes.
    """
    try:
        return {
            "model_name": config_used["model"]["model_name"],
            "n_train": manifest["train_count"],
            "n_val": manifest["val_count"],
            "n_test": manifest["test_count"],
            "ppo_iterations": config_used["ppo"]["iterations"],
            "device": config_used["model"]["device"],
            "disk_usage_bytes": int(
                sum(f.stat().st_size for f in Path(out_dir).rglob("*") if f.is_file())
            ),
        }
    except (KeyError, TypeError) as exc:
        raise ProvenanceError(
            f"assemble_report: sidecars are missing a scale field: {exc}"
        ) from exc


def _build_endpoint(
    eval_by_run: dict[tuple[str, int], dict[str, Any]],
) -> dict[str, Any]:
    """Primary endpoint (R-006): treatment arm B against control arm A."""
    endpoint_pairs = _endpoint_pairs_from_evals(eval_by_run)
    if not endpoint_pairs:
        return {
            "success": False,
            "n_seeds": 0,
            "n_seeds_replicated": 0,
            "per_seed": [],
            "note": "no paired A/B seeds available; endpoint not evaluable",
        }
    return compute_primary_endpoint(endpoint_pairs)


def _build_significance(
    eval_by_run: dict[tuple[str, int], dict[str, Any]],
) -> dict[str, Any]:
    """Significance (R-007): paired-by-qid bootstrap on B-vs-A S_q deltas."""
    control_sq = _sq_by_seed_for_arm(eval_by_run, "A")
    treatment_sq = _sq_by_seed_for_arm(eval_by_run, "B")
    shared_seeds = sorted(set(control_sq) & set(treatment_sq))
    if not shared_seeds:
        return dict(_UNEVALUABLE_SIGNIFICANCE)
    return compute_significance(
        {seed: control_sq[seed] for seed in shared_seeds},
        {seed: treatment_sq[seed] for seed in shared_seeds},
    )


def _reconcile_seed_coverage(
    eval_by_run: dict[tuple[str, int], dict[str, Any]],
    run_records: list[dict[str, Any]],
) -> list[str]:
    """QA-011: aggregations reconcile against the PLAN, not surviving files.

    ``_sq_by_seed_for_arm`` drops seeds whose eval payload carries no
    per-question ``runs`` records; a paired CI could then silently pool
    fewer seeds than planned. Returns one warning string per planned arm
    contributing fewer seeds to the per-qid S_q pool than the plan says
    (arms A/B feed the R-007 CI; the reconciliation covers every arm).
    """
    planned: dict[str, set[int]] = {}
    for record in run_records:
        planned.setdefault(record["arm"], set()).add(record["seed"])

    warnings: list[str] = []
    for arm in sorted(planned):
        contributed = set(_sq_by_seed_for_arm(eval_by_run, arm))
        missing = sorted(planned[arm] - contributed)
        if missing:
            warnings.append(
                f"arm {arm} contributes {len(contributed)} of "
                f"{len(planned[arm])} planned seeds to the per-question "
                f"S_q pool; seeds {missing} have no per-question runs "
                "records (QA-011: any A/B gap shrinks the paired "
                "bootstrap CI's seed coverage)"
            )
    return warnings


def _reconcile_planned_dirs(
    plan: list[dict[str, Any]],
    existing: list[dict[str, Any]],
) -> list[str]:
    """QA-R2-1 (QA-011 family): ``--report-only`` reconciles the FULL plan.

    The report-only branch assembles from the run dirs that exist on disk;
    a planned run whose dir is wholly missing would otherwise be silently
    dropped BEFORE :func:`assemble_report`, bypassing
    :func:`_reconcile_seed_coverage` (which only ever sees the surviving
    records). Returns one warning string per planned arm with absent run
    dir(s), in the QA-011 wording family; empty when every planned dir
    exists.
    """
    planned: dict[str, set[int]] = {}
    for record in plan:
        planned.setdefault(record["arm"], set()).add(record["seed"])
    present: dict[str, set[int]] = {}
    for record in existing:
        present.setdefault(record["arm"], set()).add(record["seed"])

    warnings: list[str] = []
    for arm in sorted(planned):
        contributed = present.get(arm, set())
        missing = sorted(planned[arm] - contributed)
        if missing:
            warnings.append(
                f"arm {arm} contributes {len(contributed)} of "
                f"{len(planned[arm])} planned run dirs to the report; "
                f"seeds {missing} have no run dir on disk (QA-R2-1: "
                "--report-only reconciles against the FULL plan — a "
                "wholly missing dir must never be silently dropped from "
                "the report)"
            )
    return warnings


def _arm_metric_mean(
    eval_by_run: dict[tuple[str, int], dict[str, Any]], arm: str, key: str
) -> float | None:
    """Mean of one eval metric across an arm's seeds; ``None`` when absent."""
    values = [
        payload.get(key)
        for (run_arm, _), payload in eval_by_run.items()
        if run_arm == arm and payload.get(key) is not None
    ]
    return float(np.mean(values)) if values else None


def _signed_delta(value: float | None, baseline: float | None) -> float | None:
    """``value - baseline``, or ``None`` when either side is unavailable."""
    if value is None or baseline is None:
        return None
    return value - baseline


def _build_arm_deltas(
    eval_by_run: dict[tuple[str, int], dict[str, Any]], arm_order: list[str]
) -> dict[str, dict[str, float | None]]:
    """Every non-control arm's mean_sq/accuracy delta vs arm A (R-004).

    Empty when the plan carries no control arm A to difference against.
    """
    if "A" not in arm_order:
        return {}
    control_mean_sq = _arm_metric_mean(eval_by_run, "A", "mean_sq")
    control_accuracy = _arm_metric_mean(eval_by_run, "A", "accuracy")
    return {
        f"{arm}_vs_A": {
            "mean_sq_delta": _signed_delta(
                _arm_metric_mean(eval_by_run, arm, "mean_sq"), control_mean_sq
            ),
            "accuracy_delta": _signed_delta(
                _arm_metric_mean(eval_by_run, arm, "accuracy"), control_accuracy
            ),
        }
        for arm in arm_order
        if arm != "A"
    }


def _read_hazard_artifacts(
    run_records: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Arm B's hazard compute block + dynamics block (R-004 / R-010b).

    Returns ``(hazard_compute, hazard_dynamics)``. Both are best-effort: a
    plan without an arm B, or an arm B whose optional sidecars were never
    written, yields ``None`` fields rather than an error.

    QA-006: report fields are sourced from the component they describe —
    ``wall_clock_seconds`` is the HAZARD-PHASE wall clock read from arm
    B's ``hazard_history.json`` (its producer times the hazard phase
    itself), while the whole-child elapsed time (PPO-dominated) is
    carried under the renamed ``child_total_wall_clock_seconds`` field,
    read from arm B's ``RUN_COMPLETE.json`` marker. Both are single-point
    values from the plan's first arm-B run (the paired design fixes the
    hazard workload across seeds; noted, not averaged).
    """
    optimizer_steps: int | None = None
    hazard_wall_clock_seconds: float | None = None
    child_total_wall_clock_seconds: float | None = None
    hazard_dynamics: dict[str, Any] | None = None

    b_records = [rec for rec in run_records if rec["arm"] == "B"]
    if b_records:
        b_dir = Path(b_records[0]["run_dir"])
        history = _read_optional_json(b_dir / "hazard" / "hazard_history.json")
        if history is not None:
            steps = history.get("steps")
            if isinstance(steps, list):
                optimizer_steps = len(steps)
            raw = history.get("wall_clock_seconds")
            if raw is not None:
                hazard_wall_clock_seconds = float(raw)
        marker = _read_optional_json(b_dir / RUN_COMPLETE_MARKER)
        if marker is not None:
            raw = marker.get("wall_clock_seconds")
            if raw is not None:
                child_total_wall_clock_seconds = float(raw)
        hazard_dynamics = _read_optional_json(b_dir / HAZARD_DYNAMICS_FILENAME)

    hazard_compute = {
        "optimizer_steps": optimizer_steps,
        "wall_clock_seconds": hazard_wall_clock_seconds,
        "child_total_wall_clock_seconds": child_total_wall_clock_seconds,
        "step_matching_note": _HAZARD_STEP_MATCHING_NOTE,
    }
    return hazard_compute, hazard_dynamics


def _build_caveats(*, smoke: bool) -> list[str]:
    """Scope caveats carried by every report (``DEVICE2_CAVEAT`` first)."""
    caveats = [DEVICE2_CAVEAT, _HAZARD_STEP_MATCHING_NOTE]
    if smoke:
        caveats.append(_SMOKE_CAVEAT)
    caveats.append(_PAIRED_DESIGN_CAVEAT)
    return caveats


def _build_verdict(
    endpoint: dict[str, Any],
    significance: dict[str, Any],
    arm_deltas: dict[str, dict[str, float | None]],
    scale: dict[str, Any],
    *,
    smoke: bool,
) -> dict[str, Any]:
    """Headline verdict plus the scope-limited evidence behind it (R-009)."""
    if endpoint.get("n_seeds", 0) == 0:
        verdict_label = "not_evaluable"
    elif endpoint["success"]:
        verdict_label = "endpoint_met_at_this_scale"
    else:
        verdict_label = "endpoint_not_met_at_this_scale"

    scale_note = (
        "smoke: plumbing/training-dynamics evidence only"
        if smoke
        else "preliminary Device-1 scale"
    )
    return {
        "verdict": verdict_label,
        "scope": f"{scale['model_name']} on n_test={scale['n_test']} ({scale_note})",
        "evidence": {
            "endpoint_success": endpoint.get("success"),
            "n_seeds_replicated": endpoint.get("n_seeds_replicated"),
            "n_seeds": endpoint.get("n_seeds"),
            "significance": significance.get("significance"),
            "mean_sq_delta_B_vs_A": arm_deltas.get("B_vs_A", {}).get("mean_sq_delta"),
        },
    }


def assemble_report(
    out_dir: Path,
    run_records: list[dict[str, Any]],
    *,
    smoke: bool = False,
    extra_warnings: list[str] | None = None,
) -> dict[str, Any]:
    """Assemble + write the report EXCLUSIVELY from per-run files (R-009/R-014).

    Reads each run's ``ppo_t5/config_used.json``, ``ppo_t5/
    split_manifest.json``, ``eval_result.json`` and (hazard arms)
    ``hazard/hazard_history.json`` + ``hazard_dynamics.json``. Writes
    ``<out>/hazard_efficacy_report.json`` and the plot; returns the
    report dict with pinned top-level keys: ``schema_version`` (== 1),
    ``git_sha``, ``endpoint_definition``, ``scale`` (``model_name``,
    ``n_train``, ``n_val``, ``n_test``, ``ppo_iterations``, ``device``,
    ``disk_usage_bytes``), ``caveats`` (always containing
    ``DEVICE2_CAVEAT`` verbatim), ``verdict`` (``{"verdict", "scope",
    "evidence"}``), ``endpoint``, ``significance``, ``arm_deltas``
    (``B_vs_A`` / ``C_vs_A`` side by side, each with ``mean_sq_delta`` +
    ``accuracy_delta``), ``hazard_compute`` (``optimizer_steps`` counted
    from arm B's ``hazard_history.json`` steps; ``wall_clock_seconds``
    is the HAZARD-PHASE wall clock from that same history file —
    QA-006 — while ``child_total_wall_clock_seconds`` carries the
    PPO-dominated child elapsed from arm B's ``RUN_COMPLETE.json``
    marker), ``hazard_dynamics``, ``warnings`` (QA-011: plan-vs-pool seed
    reconciliation; QA-R2-1: caller-supplied ``extra_warnings`` — e.g. the
    report-only branch's plan-vs-disk reconciliation — are PREPENDED and
    printed with them; empty list when clean), ``runs`` (per-run records
    incl. ``arm``, ``seed``, ``resumed``, ``policy_buzz_rate``,
    ``forced_commit_rate``, ``ece``, ``brier``, ``provenance``), and
    ``plot_path`` (relative ``hazard_efficacy_plot.png``). Never contains
    any Expected Wins key.

    Parameters (additive)
    ---------------------
    extra_warnings : list of str or None, keyword-only
        Caller-supplied warning strings merged (prepended) into the
        report's ``warnings`` and printed alongside the QA-011 ones
        (QA-R2-1). Default ``None`` keeps the pre-existing behavior.
    """
    out_dir = Path(out_dir)
    if not run_records:
        raise HarnessError("assemble_report: no run records to assemble")

    eval_by_run, report_runs, first_config, first_manifest = _read_run_sidecars(
        run_records
    )
    scale = _build_scale(first_config, first_manifest, out_dir)
    endpoint = _build_endpoint(eval_by_run)
    significance = _build_significance(eval_by_run)
    # QA-011: reconcile per-arm seed coverage against the plan and surface
    # any silent drop as a report warning (printed too, never swallowed).
    # QA-R2-1: caller-supplied warnings (report-only plan-vs-disk
    # reconciliation) are merged in ahead of them.
    warnings = list(extra_warnings or [])
    warnings += _reconcile_seed_coverage(eval_by_run, run_records)
    for warning in warnings:
        print(f"WARNING: {warning}")
    arm_order = list(dict.fromkeys(rec["arm"] for rec in run_records))
    arm_deltas = _build_arm_deltas(eval_by_run, arm_order)
    hazard_compute, hazard_dynamics = _read_hazard_artifacts(run_records)
    caveats = _build_caveats(smoke=smoke)
    verdict = _build_verdict(endpoint, significance, arm_deltas, scale, smoke=smoke)

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "git_sha": _git_sha(),
        "endpoint_definition": ENDPOINT_DEFINITION,
        "scale": scale,
        "caveats": caveats,
        "verdict": verdict,
        "endpoint": endpoint,
        "significance": significance,
        "arm_deltas": arm_deltas,
        "hazard_compute": hazard_compute,
        "hazard_dynamics": hazard_dynamics,
        "warnings": warnings,
        "runs": report_runs,
        "plot_path": PLOT_FILENAME,
    }

    write_plot(report, out_dir)
    save_json(out_dir / REPORT_FILENAME, report)
    return report


# ---------------------------------------------------------------------------
# main() composition
# ---------------------------------------------------------------------------


def _prune_all_runs(records: list[dict[str, Any]]) -> None:
    """Reclaim disk across every evaluated run dir (``--prune-checkpoints``)."""
    for record in records:
        prune_run_checkpoints(Path(record["run_dir"]))


def _print_plan(records: list[dict[str, Any]], splits: dict[str, Path] | None) -> None:
    """Print the full preflighted plan (R-013)."""
    print("=" * 60)
    print(f"HAZARD EFFICACY PLAN — {len(records)} run(s)")
    if splits:
        for split in ("train", "val", "test"):
            if split in splits:
                print(f"  split[{split}]: {splits[split]} (source: persisted artifacts)")
    for index, record in enumerate(records, start=1):
        print(
            f"  [{index}/{len(records)}] {Path(record['run_dir']).name}  "
            f"arm={record['arm']} seed={record['seed']} "
            f"hazard={record['hazard']}"
            + (f" variant={record['variant']}" if record.get("variant") else "")
        )
        print(f"      argv: {' '.join(record['argv'])}")
    print("=" * 60)


def _manifest_from_artifacts(
    splits: dict[str, Path],
    train_questions: list,
    val_questions: list,
    test_questions: list,
) -> dict[str, Any]:
    """Split manifest for the shared supervised checkpoint (R-008).

    Field set pinned to ``scripts/train_t5_policy.py::_build_split_manifest``
    (the spec's Format-pinning section) so downstream consumers see one
    manifest schema everywhere.
    """
    mc_path = Path(splits["train"]).parent / "mc_dataset.json"
    total = len(train_questions) + len(val_questions) + len(test_questions)
    return {
        "source": _PERSISTED_SOURCE,
        "mc_path": str(mc_path) if mc_path.exists() else None,
        "train_path": str(Path(splits["train"]).resolve()),
        "val_path": str(Path(splits["val"]).resolve()),
        "test_path": str(Path(splits["test"]).resolve()),
        "train_qids": [q.qid for q in train_questions],
        "val_qids": [q.qid for q in val_questions],
        "test_qids": [q.qid for q in test_questions],
        "train_count": len(train_questions),
        "val_count": len(val_questions),
        "test_count": len(test_questions),
        "effective_train_ratio": len(train_questions) / max(1, total),
        "effective_val_ratio": len(val_questions) / max(1, total),
        "effective_test_ratio": len(test_questions) / max(1, total),
    }


def _run_shared_supervised(args: argparse.Namespace, out_dir: Path) -> Path:
    """Run the ONE shared supervised warm-start child (R-003/R-008).

    The supervised child runs through :func:`_run_child` FIRST, with an
    argv carrying NO ``--skip-supervised`` and a
    ``supervised.checkpoint_dir=<root>`` positional override, so the real
    trainer leaves the branched checkpoint at ``<root>/supervised/
    best_model`` — the exact ``--model-path`` every arm child receives.
    QA-007: the child additionally receives ``--ppo-iterations 1`` — its
    PPO phase is discarded (every arm re-runs PPO from the branch point),
    so the shared-prefix job stops at the branch point instead of burning
    a full PPO budget (hours at t5-base).

    QA-002: a fresh build writes an identity marker
    (``<root>/SHARED_SUPERVISED.json`` = ``{"config_hash", "git_sha",
    "model_name", "supervised_seed"}``). An existing non-empty shared
    checkpoint is reused ONLY when the marker exists and its
    ``config_hash``/``model_name`` match THIS invocation
    (``ProvenanceError`` otherwise — a t5-small smoke checkpoint must
    never silently seed a t5-base comparison); a ``git_sha`` drift warns
    without raising (R-013 policy). ``--force`` rebuilds the shared
    checkpoint unconditionally.

    QA-R2-3: ``supervised_seed`` (the build's ``seeds[0]``) is
    RECORDED-AND-WARN, never enforced: on reuse with a different
    ``seeds[0]``, a warning is printed and
    ``"shared_supervised_seed_mismatch": true`` is recorded into the
    marker (durable provenance) — never a raise, because ANY fixed
    shared prefix preserves the paired contrast (deliberate exemption,
    now recorded).
    """
    sup_root = shared_supervised_root(out_dir)
    shared_ckpt = shared_supervised_checkpoint(out_dir)
    sup_log = sup_root / TRAIN_LOG_FILENAME
    marker_path = sup_root / SHARED_SUPERVISED_MARKER
    identity = _shared_supervised_identity(args)

    force = bool(getattr(args, "force", False))
    if shared_ckpt.is_dir() and any(shared_ckpt.iterdir()):
        if not force:
            if not marker_path.exists():
                raise ProvenanceError(
                    f"Shared supervised checkpoint at {shared_ckpt} has no "
                    f"identity marker ({SHARED_SUPERVISED_MARKER}); it cannot "
                    "be validated against this invocation (QA-002). Delete "
                    f"{sup_root} or pass --force to rebuild it."
                )
            marker = _load_json_file(marker_path, error_cls=ProvenanceError)
            mismatched = [
                key
                for key in ("config_hash", "model_name")
                if marker.get(key) != identity[key]
            ]
            if mismatched:
                details = ", ".join(
                    f"{key}: marker={marker.get(key)!r} != current="
                    f"{identity[key]!r}"
                    for key in mismatched
                )
                raise ProvenanceError(
                    f"Shared supervised checkpoint at {shared_ckpt} was built "
                    f"for a DIFFERENT invocation ({details}); reusing it would "
                    "poison every arm (QA-002). Delete "
                    f"{sup_root} or pass --force to rebuild it."
                )
            if marker.get("git_sha") != identity["git_sha"]:
                print(
                    "WARNING: shared supervised checkpoint was built at git "
                    f"sha {marker.get('git_sha')!r} but the current checkout "
                    f"is {identity['git_sha']!r}; reusing it (drift recorded, "
                    "not fatal)."
                )
            # QA-R2-3: build-seed drift is RECORDED-AND-WARN, never enforced —
            # any fixed shared prefix preserves the paired contrast.
            marker_seed = marker.get("supervised_seed")
            if (
                marker_seed is not None
                and marker_seed != identity["supervised_seed"]
            ):
                print(
                    "WARNING: shared supervised checkpoint was built with "
                    f"seed {marker_seed!r} but this invocation's build seed "
                    f"(seeds[0]) is {identity['supervised_seed']!r}; reusing "
                    "it (any fixed shared prefix preserves the paired "
                    "contrast — recording shared_supervised_seed_mismatch, "
                    "not fatal)."
                )
                if not marker.get("shared_supervised_seed_mismatch"):
                    save_json(
                        marker_path,
                        {**marker, "shared_supervised_seed_mismatch": True},
                    )
            print(f"[supervised] resumed: shared checkpoint exists at {shared_ckpt}")
            return shared_ckpt
        print("[supervised] --force: rebuilding the shared supervised checkpoint")

    # QA-001 discipline: invalidate the identity marker BEFORE mutating the
    # checkpoint so a crashed rebuild cannot pass validation next time.
    if marker_path.exists():
        marker_path.unlink()

    argv: list[str] = [
        sys.executable,
        str(_TRAIN_SCRIPT),
        "--config",
        str(args.config),
        "--seed",
        str(int(args.seeds[0])),
        # QA-007: the branched checkpoint is the SUPERVISED one; the child's
        # own PPO phase is discarded, so stop it at the branch point.
        "--ppo-iterations",
        "1",
    ]
    if args.smoke:
        argv += _SMOKE_CHILD_FLAGS
    argv.append(f"supervised.checkpoint_dir={sup_root}")

    print("[supervised] shared warm-start started")
    start = time.monotonic()
    exit_code = _run_child(argv, sup_log)
    elapsed = time.monotonic() - start
    if exit_code != 0:
        raise ChildRunError(
            f"Shared supervised run failed with exit code {exit_code}; "
            f"log: {sup_log}\n--- log tail ---\n{_log_tail(sup_log)}"
        )
    if not shared_ckpt.is_dir():
        raise ChildRunError(
            f"Shared supervised run left no checkpoint at {shared_ckpt}; "
            f"inspect the child log: {sup_log}"
        )
    save_json(marker_path, identity)
    print(f"[supervised] finished, elapsed {elapsed:.1f}s")
    return shared_ckpt


def _load_yaml_config(config_path: str) -> dict[str, Any]:
    """Load the trainer YAML config for the eval context (fail-loud)."""
    path = _resolve_config_path(str(config_path))
    if not path.exists():
        raise PreflightError(
            f"Config file not found: {config_path} (also tried anchoring to "
            f"{PROJECT_ROOT})"
        )
    with path.open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise PreflightError(f"Config file {path} did not parse to a mapping")
    return loaded


def subset_questions_by_manifest(
    questions: list, qids: list, *, split: str, manifest_path: Path
) -> list:
    """Select loaded artifact questions BY a child manifest's qids (QA-004).

    R-005 literally: the harness's eval/probe/manifest question sets are
    resolved FROM the split manifest a child actually trained against —
    children honor ``data.max_questions``/scope while the raw artifacts do
    not, so at capped scales the artifact lists are supersets. Returns the
    subset in MANIFEST order and asserts harness-loaded/manifest agreement:
    a manifest qid missing from the loaded artifacts raises
    ``ProvenanceError`` (the run predates an artifact rebuild).
    """
    by_qid: dict[str, Any] = {}
    for question in questions:
        by_qid.setdefault(question.qid, question)
    missing = [qid for qid in qids if qid not in by_qid]
    if missing:
        raise ProvenanceError(
            f"{len(missing)} {split} qid(s) named by {manifest_path} are "
            f"absent from the loaded persisted artifacts (e.g. "
            f"{missing[:5]}); the harness cannot evaluate on the split the "
            "children trained against. Rebuild the artifacts or delete the "
            "stale run dirs."
        )
    return [by_qid[qid] for qid in qids]


def main(argv: list[str] | None = None) -> None:
    """Harness entry point: preflight -> plan -> execute -> eval -> report.

    ``--dry-run`` stops after preflight+plan with zero children;
    ``--report-only`` performs zero training/eval calls (and no split
    preflight) and reassembles report + plot from existing run dirs,
    reconciling the surviving dirs against the FULL plan (QA-R2-1: a
    wholly missing planned dir becomes a report warning, never a silent
    drop) and deriving the report's smoke label from the runs'
    ``RUN_COMPLETE.json`` markers (QA-R2-2: mixed smoke provenance raises
    ``ProvenanceError``; markers predating the field fall back to
    ``--smoke``); ``--report-only --dry-run`` is plan-print only — zero
    writes and zero deletions, ``--prune-checkpoints`` included (QA-005).
    """
    args = parse_args(argv)
    validate_flag_compatibility(args)
    # QA-012: absolutize path-shaped CLI inputs ONCE at the boundary so the
    # harness and its children (which inherit an arbitrary CWD) resolve the
    # same config file and output tree regardless of invocation directory.
    args.config = str(_resolve_config_path(str(args.config)))
    args.out_dir = str(Path(args.out_dir).resolve())
    out_dir = Path(args.out_dir)

    if args.report_only:
        plan = plan_runs(args)
        records = [rec for rec in plan if Path(rec["run_dir"]).exists()]
        if args.dry_run:
            # QA-005: plan-print only under the zero-actions combination.
            _print_plan(plan, None)
            print(
                "--report-only --dry-run: would reassemble the report from "
                f"{len(records)} existing run dir(s) under {out_dir}; zero "
                "children, zero writes, zero deletions."
            )
            return
        if not records:
            raise PreflightError(
                f"--report-only found no existing run dirs under {out_dir} "
                f"for the planned arms/seeds "
                f"({[Path(r['run_dir']).name for r in plan]})"
            )
        # QA-R2-1: reconcile the surviving dirs against the FULL plan — a
        # planned arm/seed whose dir is wholly missing becomes a printed
        # report warning instead of a silent drop.
        plan_warnings = _reconcile_planned_dirs(plan, records)
        current_sha = _git_sha()
        marker_smoke_by_run: dict[str, bool] = {}
        for record in records:
            record["resumed"] = True
            # QA-008: git-sha drift is read from each run's real marker —
            # never unconditionally False (R-013 drift recording).
            marker = _read_optional_json(
                Path(record["run_dir"]) / RUN_COMPLETE_MARKER
            )
            record["git_sha_mismatch"] = (
                marker is None or marker.get("git_sha") != current_sha
            )
            # QA-R2-2: collect each run's persisted smoke flag (additive
            # marker field; markers predating it contribute nothing).
            if marker is not None and marker.get("smoke") is not None:
                marker_smoke_by_run[_run_name(record)] = bool(marker["smoke"])
        smoke_values = set(marker_smoke_by_run.values())
        if len(smoke_values) > 1:
            smoke_runs = sorted(
                name for name, value in marker_smoke_by_run.items() if value
            )
            full_runs = sorted(
                name for name, value in marker_smoke_by_run.items() if not value
            )
            raise ProvenanceError(
                f"--report-only found MIXED smoke provenance under {out_dir}: "
                f"runs {smoke_runs} completed with --smoke while runs "
                f"{full_runs} did not (their RUN_COMPLETE.json markers "
                "disagree); one report cannot honestly label both cohorts "
                "(QA-R2-2). Delete the stale run dirs or report on a single "
                "cohort."
            )
        # QA-R2-2: the report's smoke labeling (verdict scale note + smoke
        # caveat) comes from the runs' own provenance, not this
        # invocation's flag; legacy markers without the field fall back to
        # --smoke (the pre-existing behavior).
        report_smoke = (
            next(iter(smoke_values)) if smoke_values else bool(args.smoke)
        )
        assemble_report(
            out_dir, records, smoke=report_smoke, extra_warnings=plan_warnings
        )
        if args.prune_checkpoints:
            _prune_all_runs(records)
        print(f"Report written to {out_dir / REPORT_FILENAME}")
        return

    # R-013 preflight: split artifacts + full plan validation BEFORE any child.
    splits = resolve_split_artifacts(smoke=bool(args.smoke))
    plan = plan_runs(args)
    _print_plan(plan, splits)
    if args.dry_run:
        print("--dry-run: stopping after preflight; zero children launched.")
        return

    yaml_config = _load_yaml_config(str(args.config))
    train_questions = load_mc_questions(splits["train"])
    val_questions = load_mc_questions(splits["val"])
    test_questions = load_mc_questions(splits["test"])
    if not train_questions or not test_questions:
        raise PreflightError(
            "Persisted split artifacts resolved but are empty "
            f"(train={len(train_questions)}, test={len(test_questions)}); "
            "rebuild them with scripts/build_mc_dataset.py"
        )

    # Shared supervised warm-start: the FIRST child (R-003/R-008), identity-
    # validated on reuse and rebuilt under --force (QA-002).
    shared_ckpt = _run_shared_supervised(args, out_dir)

    # QA-002: resumed arm dirs must match what THIS invocation would produce
    # (model identity + current-artifact split membership).
    base_config = _resolved_child_base_config(str(args.config), bool(args.smoke))
    expected_run_context = {
        "model_name": base_config.get("model", {}).get("model_name"),
        "artifact_qids": {
            "train": {q.qid for q in train_questions},
            "val": {q.qid for q in val_questions},
            "test": {q.qid for q in test_questions},
        },
    }

    # Arm children with resume/partial/force semantics + arm control.
    # QA-R2-2: the invocation's smoke flag is persisted into each fresh
    # run's completion marker so --report-only can relabel honestly.
    records = execute_plan(
        plan,
        force=bool(args.force),
        expected_run_context=expected_run_context,
        smoke=bool(args.smoke),
    )

    # QA-004 / R-005: single-source split resolution — the eval/probe/
    # supervised-manifest question sets are subset from the loaded artifacts
    # BY the first completed run's split manifest (children honor
    # data.max_questions; the raw artifacts do not). assert_arm_control has
    # already proven every run's manifest agrees with run 1's.
    first_manifest_path = (
        Path(records[0]["run_dir"]) / "ppo_t5" / "split_manifest.json"
    )
    first_manifest = _load_json_file(
        first_manifest_path, error_cls=ProvenanceError
    )
    train_questions = subset_questions_by_manifest(
        train_questions,
        first_manifest.get("train_qids") or [],
        split="train",
        manifest_path=first_manifest_path,
    )
    val_questions = subset_questions_by_manifest(
        val_questions,
        first_manifest.get("val_qids") or [],
        split="val",
        manifest_path=first_manifest_path,
    )
    test_questions = subset_questions_by_manifest(
        test_questions,
        first_manifest.get("test_qids") or [],
        split="test",
        manifest_path=first_manifest_path,
    )

    # R-008: persist the supervised split manifest from the SAME resolved
    # split every run trained/evaluated against (QA-004), asserting
    # supervised-train/test disjointness.
    manifest = _manifest_from_artifacts(
        splits, train_questions, val_questions, test_questions
    )
    write_supervised_split_manifest(shared_ckpt, manifest)

    # R-005/R-014: identical eval path per run on the TEST split, persisted
    # immediately per run.
    eval_context = {
        "config": yaml_config,
        "reference_questions": train_questions,
        "test_set_source": _PERSISTED_SOURCE,
    }
    evaluate_all_runs(records, test_questions, eval_context)

    # R-010b: stop-probability probe per hazard run (supervised vs hazard).
    # QA-007: the supervised "before" probe is identical across hazard runs
    # (same shared checkpoint, same probe questions), so it is computed once
    # via the shared cache and reused.
    probe_questions = select_probe_questions(train_questions)
    probe_cache: dict[str, Any] = {}
    for record in records:
        if not record.get("hazard"):
            continue
        run_dir = Path(record["run_dir"])
        probe_and_write_hazard_dynamics(
            str(shared_ckpt),
            str(run_dir / "hazard" / "best_model"),
            probe_questions,
            run_dir / "hazard" / "hazard_history.json",
            run_dir / HAZARD_DYNAMICS_FILENAME,
            before_cache=probe_cache,
        )

    # R-009/R-014: report + plot, read exclusively from per-run files.
    assemble_report(out_dir, records, smoke=bool(args.smoke))

    if args.prune_checkpoints:
        _prune_all_runs(records)

    print(f"Report written to {out_dir / REPORT_FILENAME}")
    print(f"Plot written to {out_dir / PLOT_FILENAME}")


if __name__ == "__main__":
    main()
