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
import math
import os
import platform
import queue
import shutil
import subprocess
import sys
import threading
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
# MA-006: default output-staleness watchdog for child runs (minutes without a
# single output line before the child is killed); 0/negative disables.
# Mini-audit-verify F2: raised 60 -> 120 — full-scale supervised epochs and
# PPO iterations can legitimately stay quiet for over an hour on Device-1.
DEFAULT_STALL_TIMEOUT_MINUTES = 120.0

_TRAIN_SCRIPT = PROJECT_ROOT / "scripts" / "train_t5_policy.py"
_SHARED_SUPERVISED_DIRNAME = "shared_supervised"
_PERSISTED_SOURCE = "persisted_artifacts"
_CORE_ARMS = frozenset(DEFAULT_ARMS)
_LOG_TAIL_LINES = 20

# MA-006: active watchdog limits (seconds), armed by main() from the CLI
# flags. Module-level (not extra _run_child positional args) so the pinned
# single injectable seam signature `_run_child(argv, log_path)` survives.
_ACTIVE_STALL_TIMEOUT_SECONDS: float | None = DEFAULT_STALL_TIMEOUT_MINUTES * 60.0
_ACTIVE_CHILD_TIMEOUT_SECONDS: float | None = None

# MA-005 / QA-001: ONE atomic invalidation set for all per-run derived
# artifacts — every file here is unlinked before a child is (re-)launched
# into an existing dir, so no stale derived artifact can outlive its run.
_STALE_RUN_ARTIFACTS = (
    RUN_COMPLETE_MARKER,
    EVAL_RESULT_FILENAME,
    HAZARD_DYNAMICS_FILENAME,
)

# MA-008: the variant namespace is DISTINCT from the role-bearing core-arm
# literals: run dirs are ``variant_<NAME>_seed<k>`` and the arm label is
# ``variant:<NAME>``, so a variant can never capture a control/treatment
# role in the endpoint/significance/deltas keying (which keys on "A"/"B").
_VARIANT_DIR_PREFIX = "variant_"
_VARIANT_ARM_PREFIX = "variant:"

# MA-008: FLAGS tokens a variant may NEVER smuggle past arm control. The
# flag set covers identity-bearing child flags (last-wins would silently
# rebind the run identity); the override prefixes cover the two positional
# overrides the harness itself injects.
_RESERVED_VARIANT_FLAGS = frozenset(
    {"--seed", "--model-path", "--config", "--skip-supervised", "--mc-path",
     "--smoke"}
)
_RESERVED_VARIANT_OVERRIDE_PREFIXES = (
    "supervised.checkpoint_dir=",
    "ppo.eval_interval=",
)

# MA-008: value-taking flags of the child parser (scripts/train_t5_policy.py)
# — the token FOLLOWING one of these in variant FLAGS is its value, not a
# bare positional (the roundtrip through the real parser remains the
# authority; this mirror only scopes the bare-token preflight check).
_CHILD_VALUE_FLAGS = frozenset(
    {"--config", "--model-path", "--mc-path", "--ppo-iterations",
     "--beta-terminal", "--hazard-ablation", "--seed"}
)

# MA-007: per-checkpoint disk estimates (bytes) used by the preflight disk
# budget when no shared checkpoint exists yet to measure.
_MODEL_SIZE_ESTIMATE_BYTES: dict[str, float] = {
    "t5-small": 0.25e9,
    "t5-base": 0.95e9,
    "t5-large": 3.1e9,
}
_DEFAULT_MODEL_SIZE_ESTIMATE_BYTES = 3.1e9  # unknown model: assume t5-large scale

# MA-014: resume affordance appended to abort paths (child failures, SIGINT).
_RESUME_GUIDANCE = (
    "Completed runs keep their RUN_COMPLETE.json markers — re-running the "
    "same command resumes from them (partial dirs need deletion or --force)."
)

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

# MA-011 (amended in mini-audit fix round): the endpoint string names the
# EXACT payload keys and index conventions so it is executable by hand from
# the report alone.
ENDPOINT_DEFINITION = (
    "Primary endpoint (R-006): treatment mean correct-answer buzz position "
    "<= control mean correct-answer buzz position - 1.0 prefixes AND "
    "treatment accuracy >= control accuracy - 0.01 (absolute), replicated "
    "in >= 2 of 3 seeds (inclusive thresholds). A seed where either arm has "
    "zero correct policy buzzes is a non-success with undefined_position. "
    "Exact payload keys (MA-011): per arm x seed, position = eval_result.json"
    "['mean_correct_buzz_position'] — the mean of runs[].buzz_position over "
    "records with buzzed=true AND correct=true, where buzz_position is the "
    "0-indexed prefix index at buzz time (step_count - 1) as emitted by "
    "scripts.compare_policies.evaluate_t5_policy; accuracy = eval_result.json"
    "['accuracy'] (policy-buzz answer accuracy over all evaluated questions); "
    "the zero-buzz guard reads eval_result.json['n_correct_policy_buzzes']. "
    "NOTE the report's hazard_dynamics block uses 1-indexed expected buzz "
    "positions (see its index_convention field) — the two blocks are NOT on "
    "the same index base."
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
    """Canonical run-dir display name for a plan record.

    ``<arm>_seed<k>`` for core arms; ``variant_<NAME>_seed<k>`` for variant
    records (MA-008: the variant namespace is distinct from the core-arm
    literals), matching the run-dir name in both cases.
    """
    variant = record.get("variant")
    if variant:
        return f"{_VARIANT_DIR_PREFIX}{variant}_seed{record['seed']}"
    return f"{record['arm']}_seed{record['seed']}"


def _atomic_save_json(path: Path, payload: Any) -> Path:
    """Write a JSON marker atomically: temp file + ``os.replace`` (MA-015).

    An in-place rewrite (e.g. the QA-R2-3 identity-marker update) can leave
    a truncated file behind on crash, which then classifies as complete and
    dies unreadable. ``allow_nan=False`` (MA-017): a marker must never carry
    strict-invalid JSON.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
    )
    os.replace(tmp, path)
    return path


def _read_run_marker(path: Path) -> dict[str, Any]:
    """Typed ``RUN_COMPLETE.json`` reader (MA-015).

    A corrupt/truncated/mistyped marker raises ``PartialRunError`` WITH
    remediation (delete the dir or ``--force``) instead of an unexplained
    decode error mid-resume; field types are validated before any consumer
    coerces them (a string ``smoke`` or boolean ``wall_clock_seconds`` must
    never silently skew the cohort/compute blocks).
    """
    path = Path(path)
    remediation = (
        "Delete the run directory to re-run it fresh, or pass --force to "
        "re-run everything (MA-015)."
    )
    if not path.exists():
        raise PartialRunError(f"run marker is missing: {path}. {remediation}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise PartialRunError(
            f"Run marker {path} is corrupt or unreadable ({exc}); the run "
            f"cannot be trusted as complete. {remediation}"
        ) from exc
    if not isinstance(payload, dict):
        raise PartialRunError(
            f"Run marker {path} must be a JSON object, got "
            f"{type(payload).__name__}. {remediation}"
        )
    type_specs: list[tuple[str, tuple[type, ...], bool]] = [
        # (field, allowed types, required)
        ("git_sha", (str,), True),
        ("arm", (str,), True),
        ("completed_at", (str,), False),
        ("smoke", (bool,), False),
        ("shared_supervised_weights_sha256", (str,), False),
    ]
    for field, allowed, required in type_specs:
        value = payload.get(field)
        if value is None:
            if required:
                raise PartialRunError(
                    f"Run marker {path} is missing required field "
                    f"{field!r}. {remediation}"
                )
            continue
        if not isinstance(value, allowed):
            raise PartialRunError(
                f"Run marker {path} field {field!r} has invalid type "
                f"{type(value).__name__} (value {value!r}). {remediation}"
            )
    seed = payload.get("seed")
    if seed is None or isinstance(seed, bool) or not isinstance(seed, int):
        raise PartialRunError(
            f"Run marker {path} field 'seed' must be an integer, got "
            f"{seed!r}. {remediation}"
        )
    wall = payload.get("wall_clock_seconds")
    if wall is not None and (
        isinstance(wall, bool) or not isinstance(wall, (int, float))
        or not math.isfinite(float(wall))
    ):
        raise PartialRunError(
            f"Run marker {path} field 'wall_clock_seconds' must be a finite "
            f"number, got {wall!r}. {remediation}"
        )
    return payload


# MA-003: files that live inside a checkpoint dir but are NOT weight
# content, so the fingerprint must ignore them:
# - split_manifest.json: the HARNESS itself writes it into the shared
#   checkpoint dir after the build (R-008) — otherwise the harness's own
#   manifest write would read as a checkpoint mutation on the very next
#   reuse.
# - training_state.pt: optimizer state saved by the trainer beside the
#   weights and legitimately deleted by --prune-checkpoints (mini-audit-
#   verify F1) — hashing it would make a pruned-but-unchanged shared
#   checkpoint read as mutated and break shared-checkpoint reuse.
_FINGERPRINT_EXCLUDED_FILES = frozenset(
    {"split_manifest.json", "training_state.pt"}
)


def _weights_fingerprint(model_dir: Path) -> str:
    """Content sha256 of a saved checkpoint dir (MA-003).

    Hashes every regular MODEL file under ``model_dir`` in sorted
    relative-path order (path + NUL + bytes per file), skipping symlinks and
    the harness-written provenance sidecars
    (``_FINGERPRINT_EXCLUDED_FILES``), so two saves of different weights can
    never share a fingerprint and a rebuilt branch point is detectable from
    descendants.
    """
    model_dir = Path(model_dir)
    if not model_dir.is_dir():
        raise ProvenanceError(
            f"_weights_fingerprint: {model_dir} is not a directory"
        )
    files = sorted(
        p
        for p in model_dir.rglob("*")
        if p.is_file()
        and not p.is_symlink()
        and p.name not in _FINGERPRINT_EXCLUDED_FILES
    )
    if not files:
        raise ProvenanceError(
            f"_weights_fingerprint: {model_dir} contains no model files"
        )
    digest = hashlib.sha256()
    for file_path in files:
        digest.update(file_path.relative_to(model_dir).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_path.read_bytes())
    return digest.hexdigest()


def assert_unique_qids(
    qids: list, *, where: str, error_cls: type = ProvenanceError
) -> None:
    """MA-016: uniqueness asserted at every qid ingestion boundary.

    Duplicate qids double-weight accuracy/means while the paired bootstrap
    dedups per qid — an inconsistent weighting that must fail loud instead
    of silently skewing the report.
    """
    seen: set = set()
    dupes: set = set()
    for qid in qids:
        if qid in seen:
            dupes.add(qid)
        seen.add(qid)
    if dupes:
        raise error_cls(
            f"{where} carries {len(dupes)} duplicate qid(s) (e.g. "
            f"{sorted(dupes)[:5]}); duplicates double-weight accuracy while "
            "the paired bootstrap dedups per qid (MA-016). Rebuild the "
            "artifacts / delete the offending run dir."
        )


def _validated_metric(value: Any, *, key: str, where: str) -> float | None:
    """MA-018: metric-float validator for eval-payload consumers.

    ``None`` passes through (legitimately-absent metric); anything else must
    be a finite real number (bool excluded) or the consumer fails loud
    naming the payload instead of propagating NaN/inf into the report.
    """
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProvenanceError(
            f"{where}: metric {key!r} must be numeric or null, got "
            f"{value!r} (MA-018)"
        )
    number = float(value)
    if not math.isfinite(number):
        raise ProvenanceError(
            f"{where}: metric {key!r} is non-finite ({value!r}) (MA-018)"
        )
    return number


def _measure_disk_usage(root: Path) -> int:
    """Total bytes of regular files under ``root`` (MA-018 hardened walk).

    Never follows directory symlinks (no cycles, no counting content
    outside the tree), skips symlinked files, and tolerates files vanishing
    mid-walk (children may still be flushing/cleaning up).
    """
    total = 0
    for dirpath, _dirnames, filenames in os.walk(root, followlinks=False):
        for name in filenames:
            file_path = Path(dirpath) / name
            try:
                if file_path.is_symlink():
                    continue
                total += file_path.stat().st_size
            except OSError:
                continue  # vanished mid-walk — tolerated, not fatal
    return int(total)


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

    Additive flags (mini-audit fix round): ``--re-eval`` (flag, MA-012),
    ``--stall-timeout-minutes`` (float, default 120 — raised from 60 in the
    mini-audit-verify round, F2; MA-006), ``--child-timeout-minutes``
    (float, default None, MA-006).
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
            "whitespace-split and appended to a hazard-arm child argv. The "
            "variant inherits the invocation's hazard knobs (FLAGS override); "
            "its run dirs are variant_<NAME>_seed<k> with arm label "
            "variant:<NAME> (MA-008)."
        ),
    )
    parser.add_argument(
        "--re-eval",
        action="store_true",
        help=(
            "Re-run eval + probe for resumed runs even when eval_result.json"
            " / hazard_dynamics.json already exist (MA-012)."
        ),
    )
    parser.add_argument(
        "--stall-timeout-minutes",
        type=float,
        default=DEFAULT_STALL_TIMEOUT_MINUTES,
        help=(
            "Kill a child that prints no output line for this many minutes "
            f"(0 disables; default {DEFAULT_STALL_TIMEOUT_MINUTES:g} — MA-006)."
        ),
    )
    parser.add_argument(
        "--child-timeout-minutes",
        type=float,
        default=None,
        help=(
            "Optional hard cap on any single child's total runtime in "
            "minutes (default: none — MA-006)."
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
    "log_path": Path == run_dir / "train.log", "variant": str | None,
    "hazard_knobs": dict}`` — ``hazard_knobs`` (additive, MA-001) is the
    hazard identity the CHILD will record into ``config_used.json``
    (``{"pretrain", "beta_terminal", "freeze_answer_head", "ablation"}``),
    derived from round-tripping the composed argv through the REAL child
    parser, and is positively asserted at every consumption site.
    ``expected_identity`` (additive, mini-audit-verify F3) is the
    ``{"model_path", "config", "smoke"}`` identity surface the roundtrip
    asserts against the parsed namespace (plus ``skip_supervised is True``
    and ``mc_path is None``).

    ``--variant NAME:FLAGS`` (split on the FIRST colon; FLAGS
    whitespace-split) adds one extra hazard variant per seed with run dir
    ``<out>/variant_<NAME>_seed<k>`` and arm label ``variant:<NAME>``
    (MA-008: distinct namespace from the role-bearing core arms). Variants
    inherit the invocation's hazard knobs; FLAGS override them. Rejected at
    preflight (``PreflightError``): names with path separators / ``..``,
    names shadowing core arms A/B/C, reserved FLAGS tokens (``--seed``,
    ``--model-path``, ``--config``, ``--skip-supervised``, ``--mc-path``,
    ``--smoke``), reserved overrides (``supervised.checkpoint_dir=`` /
    ``ppo.eval_interval=``), and bare ``=``-less non-flag tokens (silent
    positional no-ops in the child).

    MA-018 preflight domain checks: every seed must satisfy
    ``0 <= seed < 2**32`` (NumPy seeding constraint) and
    ``--beta-terminal`` must be finite.
    """
    out_dir = Path(args.out_dir)
    shared_ckpt = shared_supervised_checkpoint(out_dir)

    arms = list(args.arms)
    seeds = [int(s) for s in args.seeds]
    if not arms or not seeds:
        raise PreflightError("plan_runs: --arms and --seeds must be non-empty")

    # MA-018: seed domain check at preflight (np.random.seed rejects values
    # outside [0, 2**32) only after hours of child work otherwise).
    for seed in seeds:
        if not (0 <= seed < 2**32):
            raise PreflightError(
                f"--seeds value {seed} is outside the valid seed domain "
                "[0, 2**32) (NumPy seeding constraint; MA-018)."
            )

    # MA-018: a non-finite terminal penalty poisons the hazard loss.
    if not math.isfinite(float(args.beta_terminal)):
        raise PreflightError(
            f"--beta-terminal must be finite, got {args.beta_terminal!r} "
            "(MA-018)."
        )

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
                "'..' (it becomes the run-dir name "
                f"<out>/{_VARIANT_DIR_PREFIX}<NAME>_seed<k>)"
            )
        if name in _CORE_ARMS:
            raise PreflightError(
                f"variant name {name!r} shadows a core arm (A/B/C); the core "
                "arms carry the control/treatment roles and cannot be "
                "re-bound by a variant (MA-008). Pick a distinct name."
            )
        flags = flags_str.split()
        index = 0
        while index < len(flags):
            token = flags[index]
            base = token.split("=", 1)[0]
            if token.startswith("-"):
                if base in _RESERVED_VARIANT_FLAGS:
                    raise PreflightError(
                        f"variant {name!r}: FLAGS token {token!r} is reserved "
                        "— --seed/--model-path/--config/--skip-supervised/"
                        "--mc-path/--smoke are owned by arm control and a "
                        "last-wins duplicate would smuggle a different run "
                        "identity past it (MA-008)."
                    )
                if base in _CHILD_VALUE_FLAGS and "=" not in token:
                    # The next token is this flag's VALUE, not a bare
                    # positional; the roundtrip still validates it.
                    index += 2
                    continue
            elif "=" in token:
                if token.startswith(_RESERVED_VARIANT_OVERRIDE_PREFIXES):
                    raise PreflightError(
                        f"variant {name!r}: override {token!r} is reserved — "
                        "supervised.checkpoint_dir= and ppo.eval_interval= "
                        "are injected by the harness itself (MA-008)."
                    )
            else:
                raise PreflightError(
                    f"variant {name!r}: bare token {token!r} is neither a "
                    "--flag nor a key=value override; the child parser would "
                    "swallow it as a silent positional no-op at non-smoke "
                    "scale (MA-008)."
                )
            index += 1
        variants.append((name, flags))

    records: list[dict[str, Any]] = []

    def _add(
        arm: str,
        seed: int,
        *,
        hazard: bool,
        variant: str | None = None,
        **argv_kwargs: Any,
    ) -> None:
        # MA-008: variants live in their own dir/arm namespace so they can
        # never capture a core arm's control/treatment role.
        if variant:
            run_dir = out_dir / f"{_VARIANT_DIR_PREFIX}{variant}_seed{seed}"
            arm_label = f"{_VARIANT_ARM_PREFIX}{variant}"
        else:
            run_dir = out_dir / f"{arm}_seed{seed}"
            arm_label = arm
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
                "arm": arm_label,
                "seed": seed,
                "run_dir": run_dir,
                "hazard": hazard,
                "argv": argv,
                "log_path": run_dir / TRAIN_LOG_FILENAME,
                "variant": variant,
                # Mini-audit-verify F3 (additive): the identity surface the
                # roundtrip asserts the PARSED argv still matches — a
                # doctored/smuggled token rebinding any of these dies at
                # preflight (--skip-supervised True and --mc-path None are
                # implied invariants of every arm child).
                "expected_identity": {
                    "model_path": str(shared_ckpt),
                    "config": str(args.config),
                    "smoke": bool(args.smoke),
                },
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
        # MA-008: variants INHERIT the invocation's hazard knobs; FLAGS
        # override them. An overridden knob is not re-injected, so the argv
        # carries exactly ONE occurrence of each knob flag. Mini-audit-verify
        # F5: suppression matches on the flag BASE (token.split("=", 1)[0])
        # so the =-joined form (--beta-terminal=0.5) suppresses the
        # injection exactly like the two-token form (--beta-terminal 0.5).
        inherited: dict[str, Any] = dict(hazard_knobs)
        flag_bases = {token.split("=", 1)[0] for token in flags}
        if "--beta-terminal" in flag_bases:
            inherited["beta_terminal"] = None
        if "--freeze-answer-head" in flag_bases:
            inherited["freeze_answer_head"] = False
        for seed in seeds:
            _add(
                name, seed, hazard=True, variant=name,
                extra_flags=flags, **inherited,
            )

    run_dirs = [rec["run_dir"] for rec in records]
    if len(set(run_dirs)) != len(run_dirs):
        raise PreflightError(
            "plan_runs: duplicate run dirs in the plan (repeated arm/seed "
            "or a repeated variant name)"
        )
    # QA-003 (R-013): every planned argv must round-trip through the real
    # child parser BEFORE any child is launched. MA-001/MA-008: the parsed
    # namespace IS the run's expected identity — the exact values the child
    # will record into config_used.json — stored on the record and asserted
    # at every later consumption site.
    for record in records:
        parsed = _roundtrip_child_argv(record)
        record["hazard_knobs"] = {
            "pretrain": bool(parsed.hazard_pretrain),
            "beta_terminal": float(parsed.beta_terminal),
            "freeze_answer_head": bool(parsed.freeze_answer_head),
            "ablation": parsed.hazard_ablation,
        }
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


def _roundtrip_child_argv(record: dict[str, Any]) -> argparse.Namespace:
    """Preflight-validate one planned argv against the REAL child parser.

    QA-003 (R-013): orchestrators validate composed argvs against the
    target's actual parser at plan time — a variant flag typo (or a
    positional-grouping bug) must cost zero children, never the shared
    supervised phase plus nine runs. Parses ``record["argv"]`` (minus the
    ``[sys.executable, script]`` prefix) through
    ``scripts.train_t5_policy.parse_args`` and raises ``PreflightError``
    naming the run when argparse rejects it.

    MA-008 (class fix): the roundtrip additionally compares the PARSED
    namespace to the planned values — the parsed ``--seed`` must equal the
    record's seed (a smuggled last-wins ``--seed`` can never survive to a
    child), the parsed hazard flag must match the record's arm role, and a
    non-finite parsed ``--beta-terminal`` is rejected (MA-018).
    Mini-audit-verify F3: when the record carries ``expected_identity``
    (plan_runs records do), the parsed ``model_path``/``config``/``smoke``
    must equal the planned values, ``skip_supervised`` must be True and
    ``mc_path`` must be None. Returns the parsed namespace (MA-001: it is
    the run's expected recorded identity).
    """
    import scripts.train_t5_policy as train_t5_policy

    child_argv = [str(token) for token in record["argv"][2:]]
    stderr_capture = io.StringIO()
    try:
        with contextlib.redirect_stderr(stderr_capture):
            parsed = train_t5_policy.parse_args(argv=child_argv)
    except SystemExit as exc:
        raise PreflightError(
            f"Planned child argv for run {_run_name(record)} is rejected by "
            f"the real scripts/train_t5_policy.py parser (exit {exc.code}): "
            f"{stderr_capture.getvalue().strip()}\n  argv: {child_argv}"
        ) from exc

    if parsed.seed != record.get("seed"):
        raise PreflightError(
            f"Planned child argv for run {_run_name(record)} parses to "
            f"--seed {parsed.seed!r} but the plan slot is seed "
            f"{record.get('seed')!r}; a variant FLAGS token has rebound the "
            "run identity (MA-008).\n  argv: " + str(child_argv)
        )
    if bool(parsed.hazard_pretrain) != bool(record.get("hazard")):
        raise PreflightError(
            f"Planned child argv for run {_run_name(record)} parses to "
            f"hazard_pretrain={bool(parsed.hazard_pretrain)} but the plan "
            f"slot expects hazard={bool(record.get('hazard'))} (MA-008).\n"
            "  argv: " + str(child_argv)
        )
    if not math.isfinite(float(parsed.beta_terminal)):
        raise PreflightError(
            f"Planned child argv for run {_run_name(record)} parses to a "
            f"non-finite --beta-terminal ({parsed.beta_terminal!r}); a "
            "non-finite terminal penalty poisons the hazard loss (MA-018).\n"
            "  argv: " + str(child_argv)
        )
    # Mini-audit-verify F3: the parsed IDENTITY SURFACE must match the plan
    # — model path (the shared branch point), config path, smoke, the
    # skip-supervised invariant, and the mc-path invariant. Records without
    # the key (the shared supervised record, hand-built test records) skip
    # this block.
    expected_identity = record.get("expected_identity")
    if expected_identity is not None:
        identity_checks: list[tuple[str, Any, Any]] = [
            ("--model-path", parsed.model_path, expected_identity["model_path"]),
            ("--config", parsed.config, expected_identity["config"]),
            ("--smoke", bool(parsed.smoke), bool(expected_identity["smoke"])),
            ("--skip-supervised", bool(parsed.skip_supervised), True),
            ("--mc-path", parsed.mc_path, None),
        ]
        for flag, actual, planned in identity_checks:
            if actual != planned:
                raise PreflightError(
                    f"Planned child argv for run {_run_name(record)} parses "
                    f"to {flag} = {actual!r} but the plan mandates "
                    f"{planned!r}; the run identity surface has been "
                    "rebound (MA-008 / mini-audit-verify F3).\n  argv: "
                    + str(child_argv)
                )
    return parsed


# ---------------------------------------------------------------------------
# Child execution / resume
# ---------------------------------------------------------------------------


def _run_child(
    argv: list[str],
    log_path: Path,
    *,
    stall_timeout_seconds: float | None = None,
    child_timeout_seconds: float | None = None,
) -> int:
    """Run one child via subprocess (shell=False), tee output to log_path.

    Single injectable seam: tests monkeypatch
    ``scripts.run_hazard_efficacy._run_child``. Returns the exit code.
    The child's stderr is merged into stdout so BOTH streams land in the
    log. QA-009: the Popen lifetime is bound to a context manager and a
    tee-loop exception kills the child — a leaked child would keep
    writing into the run dir after the harness moved on.

    MA-006 (watchdog): the tee loop runs an output-staleness watchdog — a
    child that prints NO output line for ``stall_timeout_seconds`` (default:
    the module-level limit armed by ``main()`` from
    ``--stall-timeout-minutes``; 120 min out of the box) is killed and
    ``ChildRunError`` is raised naming the last-output age, so a wedged
    multi-hour child can never hang the harness silently forever.
    ``child_timeout_seconds`` (``--child-timeout-minutes``; default off)
    additionally caps the child's TOTAL runtime. Both keyword-only
    parameters are additive; ``None`` defers to the module-level limits and
    ``<= 0`` disables. Lines are pumped by a daemon reader thread into a
    queue so the watchdog's clock never blocks on the pipe.

    Mini-audit-verify F2/F6: the child runs with ``PYTHONUNBUFFERED=1`` so
    the watchdog sees output lines as they are produced (never in
    block-buffered bursts that read as a stall); after any watchdog kill
    the already-pumped lines are drained into the log before the error
    snapshots its tail; EOF on the pipe is followed by a ``proc.wait``
    bounded by the stall budget (a child that closed its streams but never
    exits is killed + ``ChildRunError``); and the pump thread tolerates
    the kill-time close-race on the pipe.
    """
    stall_limit = (
        _ACTIVE_STALL_TIMEOUT_SECONDS
        if stall_timeout_seconds is None
        else stall_timeout_seconds
    )
    if stall_limit is not None and stall_limit <= 0:
        stall_limit = None
    total_limit = (
        _ACTIVE_CHILD_TIMEOUT_SECONDS
        if child_timeout_seconds is None
        else child_timeout_seconds
    )
    if total_limit is not None and total_limit <= 0:
        total_limit = None

    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    argv = [str(token) for token in argv]

    limits = [limit for limit in (stall_limit, total_limit) if limit]
    poll_seconds = max(0.01, min(0.5, (min(limits) / 5.0) if limits else 0.5))

    with log_path.open("w", encoding="utf-8") as log_file:
        with subprocess.Popen(  # noqa: S603 - fixed argv list, shell=False
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            # Mini-audit-verify F2: children run UNBUFFERED so the MA-006
            # output-staleness watchdog sees lines as they are produced —
            # a healthy child behind an 8KiB block buffer used to read as
            # stalled.
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        ) as proc:
            assert proc.stdout is not None  # PIPE above guarantees a stream
            lines: queue.SimpleQueue = queue.SimpleQueue()

            def _pump(stream: Any) -> None:
                try:
                    for line in stream:
                        lines.put(line)
                except ValueError:
                    # Mini-audit-verify F6: close-race — the parent killed
                    # the child and the Popen context closed the pipe while
                    # the reader was blocked mid-readline. EOF semantics,
                    # not an error.
                    pass
                finally:
                    lines.put(None)  # EOF sentinel

            pump = threading.Thread(
                target=_pump, args=(proc.stdout,), daemon=True
            )
            pump.start()

            def _kill_and_drain() -> None:
                # QA-009: killing an already-dead child is a safe no-op.
                proc.kill()
                proc.wait()
                # Mini-audit-verify F6: whatever the pump already read must
                # land in the log BEFORE the error snapshots its tail (the
                # last lines are usually the ones naming the wedge). The
                # child is dead, so the pump ends at pipe EOF promptly.
                pump.join(timeout=1.0)
                while True:
                    try:
                        pending = lines.get_nowait()
                    except queue.Empty:
                        break
                    if pending is None:
                        break
                    log_file.write(pending)
                    sys.stdout.write(pending)
                log_file.flush()

            started = time.monotonic()
            last_output = started
            try:
                while True:
                    try:
                        line = lines.get(timeout=poll_seconds)
                    except queue.Empty:
                        now = time.monotonic()
                        age = now - last_output
                        if stall_limit is not None and age > stall_limit:
                            _kill_and_drain()
                            raise ChildRunError(
                                f"Child stalled: no output line for "
                                f"{age:.1f}s (stall timeout "
                                f"{stall_limit:.1f}s — --stall-timeout-"
                                "minutes); the child was killed (MA-006). "
                                f"Log: {log_path}\n--- log tail ---\n"
                                f"{_log_tail(log_path)}"
                            )
                        if (
                            total_limit is not None
                            and now - started > total_limit
                        ):
                            _kill_and_drain()
                            raise ChildRunError(
                                f"Child exceeded its total runtime cap of "
                                f"{total_limit:.1f}s (--child-timeout-"
                                "minutes) and was killed (MA-006). Log: "
                                f"{log_path}\n--- log tail ---\n"
                                f"{_log_tail(log_path)}"
                            )
                        continue
                    if line is None:
                        break
                    last_output = time.monotonic()
                    log_file.write(line)
                    sys.stdout.write(line)
                    if (
                        total_limit is not None
                        and last_output - started > total_limit
                    ):
                        _kill_and_drain()
                        raise ChildRunError(
                            f"Child exceeded its total runtime cap of "
                            f"{total_limit:.1f}s (--child-timeout-minutes) "
                            f"and was killed (MA-006). Log: {log_path}\n"
                            f"--- log tail ---\n{_log_tail(log_path)}"
                        )
            except BaseException:
                # QA-009: never leak a child; killing an already-dead child
                # is a safe no-op.
                proc.kill()
                raise
            # Mini-audit-verify F6: EOF on the merged pipe does not imply
            # exit — a child that closed its streams but wedged before
            # exiting must not hang the harness. The stall budget bounds
            # the post-EOF wait (None keeps the unbounded wait when the
            # watchdog is disabled).
            try:
                return proc.wait(timeout=stall_limit)
            except subprocess.TimeoutExpired as exc:
                _kill_and_drain()
                raise ChildRunError(
                    f"Child closed its output stream but did not exit "
                    f"within the stall budget ({stall_limit:.1f}s — "
                    "--stall-timeout-minutes); the child was killed "
                    f"(MA-006 / mini-audit-verify F6). Log: {log_path}\n"
                    f"--- log tail ---\n{_log_tail(log_path)}"
                ) from exc


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


def _assert_run_identity(
    record: dict[str, Any], *, marker: dict[str, Any] | None = None
) -> None:
    """MA-001: positively identify a run dir against its plan slot.

    ``hazard.*`` and ``seed`` are exactly the arm-control-EXEMPT config keys
    (R-003), so equality-diffing can never catch a copied dir fabricating a
    replication or a C checkpoint swapped in as B. This assertion replaces
    exemption-from-equality with per-run identity:

    - ``config_used.json``'s top-level ``seed`` must equal the plan seed;
    - its ``hazard`` block must equal the plan record's ``hazard_knobs``
      (derived from round-tripping the composed argv through the real child
      parser — skipped for hand-built records without the key);
    - the ``RUN_COMPLETE.json`` marker's ``arm``/``seed`` (when given) must
      match the plan slot (sidecars are self-identifying).

    Raises ``ProvenanceError`` naming the run with delete/--force
    remediation.
    """
    run_dir = Path(record["run_dir"])
    name = _run_name(record)
    remediation = (
        "Delete the directory or pass --force to re-run everything (MA-001)."
    )

    config_used = _load_json_file(
        run_dir / "ppo_t5" / "config_used.json", error_cls=ProvenanceError
    )
    if not isinstance(config_used, dict):
        raise ProvenanceError(
            f"Run {name}: config_used.json is not an object; the run cannot "
            f"be identified. {remediation}"
        )
    actual_seed = config_used.get("seed")
    if actual_seed != record["seed"]:
        raise ProvenanceError(
            f"Run {name}: config_used.json records seed={actual_seed!r} but "
            f"the plan slot expects seed={record['seed']!r}; the dir does "
            f"not belong to this plan slot (copied/renamed run dir?). "
            f"{remediation}"
        )
    expected_hazard = record.get("hazard_knobs")
    if expected_hazard is not None:
        actual_hazard = config_used.get("hazard")
        if actual_hazard != expected_hazard:
            raise ProvenanceError(
                f"Run {name}: config_used.json hazard block "
                f"{actual_hazard!r} does not match the planned hazard "
                f"identity {expected_hazard!r} (arm role swap or knob "
                f"drift). {remediation}"
            )
    if marker is not None:
        marker_arm = marker.get("arm")
        marker_seed = marker.get("seed")
        if marker_arm != record["arm"] or marker_seed != record["seed"]:
            raise ProvenanceError(
                f"Run {name}: {RUN_COMPLETE_MARKER} identifies itself as "
                f"arm={marker_arm!r} seed={marker_seed!r} but the plan slot "
                f"is arm={record['arm']!r} seed={record['seed']!r}. "
                f"{remediation}"
            )


def _assert_shared_fingerprint(
    record: dict[str, Any],
    marker: dict[str, Any],
    expected_run_context: dict[str, Any] | None,
) -> None:
    """MA-003: a resumed run must have branched from the CURRENT shared
    checkpoint content.

    Compares the marker's recorded ``shared_supervised_weights_sha256``
    against the invocation's freshly computed fingerprint
    (``expected_run_context["shared_weights_sha256"]``). Mismatch raises
    ``ProvenanceError`` (a rebuilt branch point would make the probe's
    "before" leg and the paired contrast span two different checkpoints);
    a legacy marker without the field warns (unverifiable, not fatal).
    """
    expected = (expected_run_context or {}).get("shared_weights_sha256")
    if not expected:
        return
    recorded = marker.get("shared_supervised_weights_sha256")
    if recorded is None:
        print(
            f"WARNING: resumed run {_run_name(record)} predates the "
            "shared-checkpoint weight fingerprint (legacy marker); its "
            "branch-point identity cannot be verified (MA-003)."
        )
        return
    if recorded != expected:
        raise ProvenanceError(
            f"Resumed run {_run_name(record)} branched from a shared "
            f"supervised checkpoint with weights sha256 {recorded[:12]}… "
            f"but the CURRENT shared checkpoint hashes to {expected[:12]}…; "
            "the branch point was rebuilt since this run trained (MA-003). "
            "Delete the run dir or pass --force to re-run everything."
        )


def verify_run_records(
    run_records: list[dict[str, Any]],
    *,
    expected_run_context: dict[str, Any] | None = None,
) -> None:
    """MA-002: the single validated-records gate for report-producing paths.

    BOTH the execute path and ``--report-only`` route through this gate
    before ``assemble_report``, so a heterogeneous or partial run tree can
    never assemble into a normal-looking report. Per record: complete
    classification (R-013), typed marker read (MA-015), positive run
    identity (MA-001), split-source assertion (R-013); then the full
    cross-run arm-control diff (R-003). ``expected_run_context`` (optional)
    additionally validates every record against the CURRENT invocation via
    :func:`validate_resumed_run` (QA-002).

    Mini-audit-verify F4: every marker carrying the MA-003
    ``shared_supervised_weights_sha256`` field must agree on ONE value —
    descendants of two DIFFERENT shared supervised checkpoints must never
    co-assemble into one report (``ProvenanceError`` otherwise); legacy
    field-less markers warn (their branch point is unverifiable).
    """
    shared_fps: dict[str, list[str]] = {}
    legacy_fp_runs: list[str] = []
    for record in run_records:
        run_dir = Path(record["run_dir"])
        state = classify_run_dir(run_dir, hazard=bool(record.get("hazard")))
        if state != "complete":
            raise PartialRunError(
                f"Run {_run_name(record)} dir {run_dir} is {state}, not "
                f"complete ({RUN_COMPLETE_MARKER} + checkpoints + sidecars "
                "required); it cannot enter a report (MA-002). Delete the "
                "directory, or re-run the harness (with --force if needed) "
                "to rebuild it."
            )
        marker = _read_run_marker(run_dir / RUN_COMPLETE_MARKER)
        _assert_run_identity(record, marker=marker)
        _assert_split_source(record)
        fingerprint = marker.get("shared_supervised_weights_sha256")
        if fingerprint is None:
            legacy_fp_runs.append(_run_name(record))
        else:
            shared_fps.setdefault(str(fingerprint), []).append(
                _run_name(record)
            )
        if expected_run_context:
            validate_resumed_run(record, expected_run_context)
    if len(shared_fps) > 1:
        detail = "; ".join(
            f"{fp[:12]}…: {sorted(names)}"
            for fp, names in sorted(shared_fps.items())
        )
        raise ProvenanceError(
            "Run markers disagree on shared_supervised_weights_sha256 — "
            "the runs branched from DIFFERENT shared supervised "
            f"checkpoints and cannot enter one report ({detail}) "
            "(MA-003 / mini-audit-verify F4). Delete the stale run dirs "
            "or pass --force to re-run everything."
        )
    if legacy_fp_runs:
        print(
            "WARNING: run marker(s) without the MA-003 "
            "shared_supervised_weights_sha256 field (legacy markers): "
            f"{sorted(legacy_fp_runs)}; their branch-point identity cannot "
            "be cross-checked (mini-audit-verify F4)."
        )
    assert_arm_control(run_records)


def preflight_resume_states(
    plan: list[dict[str, Any]],
    *,
    force: bool = False,
    expected_run_context: dict[str, Any] | None = None,
) -> None:
    """MA-013: resume state is decided at PREFLIGHT, before any child runs.

    A partial dir or a stale resumed dir used to be discovered fail-late,
    mid-execution, after the shared supervised phase (hours at scale). This
    sweep classifies every planned dir up front: partial dirs raise
    ``PartialRunError`` (unless ``force``); complete dirs are validated
    (identity, split source, current-invocation match) NOW. ``force``
    short-circuits: everything re-runs anyway.
    """
    if force:
        return
    for record in plan:
        run_dir = Path(record["run_dir"])
        state = classify_run_dir(run_dir, hazard=bool(record.get("hazard")))
        if state == "partial":
            raise PartialRunError(
                f"Run dir {run_dir} has outputs but no valid completion "
                f"state ({RUN_COMPLETE_MARKER} + checkpoints + sidecars). "
                f"Delete the directory to re-run {_run_name(record)} fresh, "
                "or pass --force to re-run everything. (MA-013: decided at "
                "preflight, before any child ran.)"
            )
        if state == "complete":
            marker = _read_run_marker(run_dir / RUN_COMPLETE_MARKER)
            _assert_run_identity(record, marker=marker)
            _assert_split_source(record)
            if expected_run_context:
                validate_resumed_run(record, expected_run_context)


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
    were actually trained as, never from a later invocation's flag;
    ``shared_supervised_weights_sha256`` — additive, MA-003 — records the
    branch point's content fingerprint when the caller supplies
    ``expected_run_context["shared_weights_sha256"]``, and is asserted on
    resume; markers are written atomically via temp + ``os.replace`` —
    MA-015).
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
            # MA-015: typed, remediation-carrying marker read.
            marker = _read_run_marker(run_dir / RUN_COMPLETE_MARKER)
            # MA-001: the resumed dir must positively identify as THIS plan
            # slot (config seed + hazard block + marker arm/seed).
            _assert_run_identity(record, marker=marker)
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
                # MA-003: the run must have branched from the CURRENT
                # shared checkpoint content.
                _assert_shared_fingerprint(record, marker, expected_run_context)
            continue

        if state == "partial" and not force:
            raise PartialRunError(
                f"Run dir {run_dir} has outputs but no valid completion "
                f"state ({RUN_COMPLETE_MARKER} + checkpoints + sidecars). "
                f"Delete the directory to re-run {name} fresh, or pass "
                "--force to re-run everything."
            )

        # QA-001 + MA-005: invalidate-before-mutate — the FULL per-run
        # derived-artifact set (completion marker, eval result, hazard
        # dynamics) is unlinked before any child is (re-)launched into an
        # existing dir, so no stale derived artifact can outlive its run.
        if run_dir.exists():
            for stale_name in _STALE_RUN_ARTIFACTS:
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
                f"log: {log_path}\n--- log tail ---\n{_log_tail(log_path)}\n"
                f"{_RESUME_GUIDANCE}"
            )

        check_child_outputs(record)
        if record.get("hazard") and not (run_dir / "hazard" / "best_model").exists():
            raise ChildRunError(
                f"Hazard run {name} left no hazard checkpoint at "
                f"{run_dir / 'hazard' / 'best_model'}; inspect the child "
                f"log: {log_path}\n{_RESUME_GUIDANCE}"
            )
        _assert_split_source(record)
        # MA-001: the fresh child's recorded identity (config seed + hazard
        # block) must match the plan slot before the completion marker is
        # written.
        _assert_run_identity(record)

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
        # MA-003: stamp the shared branch point's content identity into the
        # descendant's marker (when the caller supplied it).
        shared_fp = (expected_run_context or {}).get("shared_weights_sha256")
        if shared_fp:
            marker["shared_supervised_weights_sha256"] = str(shared_fp)
        # MA-015: markers are written atomically (temp + os.replace).
        _atomic_save_json(run_dir / RUN_COMPLETE_MARKER, marker)
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
    save_json(out_path, manifest, allow_nan=False)  # MA-017: strict JSON
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

    # MA-016: uniqueness asserted at the eval ingestion boundary — duplicate
    # qids would double-weight accuracy while the bootstrap dedups.
    assert_unique_qids(
        [r.get("qid") for r in runs if isinstance(r, dict)],
        where=f"run {_run_name(record)} evaluate_t5_policy runs records",
        error_cls=HarnessError,
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
    # MA-017: strict JSON — non-finite floats must never reach an artifact.
    save_json(run_dir / EVAL_RESULT_FILENAME, enriched, allow_nan=False)
    return enriched


def evaluate_all_runs(
    records: list[dict[str, Any]],
    test_questions: list,
    config: dict[str, Any],
    *,
    prune: bool = False,
    re_eval: bool = False,
) -> list[dict[str, Any]]:
    """Evaluate every run: exactly one ``evaluate_t5_policy`` call each,
    identical test split and kwargs (except checkpoint path) across calls.
    Per-run ``eval_result.json`` files already written are never deleted
    by a later failure (R-014).

    Parameters (additive)
    ---------------------
    prune : bool, keyword-only
        MA-007 (reclaim at unit completion): when True, each run's
        ``iter_*``/``epoch_*``/``training_state.pt`` are pruned right after
        its ``eval_result.json`` exists — peak disk never accumulates
        across the whole pipeline. Default ``False`` keeps the pre-existing
        behavior for direct callers.
    re_eval : bool, keyword-only
        MA-012 (read-before-recompute): a RESUMED run whose
        ``eval_result.json`` already exists is loaded instead of
        re-evaluated (with an "eval resumed" print and an MA-001 identity
        check on the loaded payload) unless ``re_eval`` forces a recompute.
        Default ``False``.
    """
    results: list[dict[str, Any]] = []
    total = len(records)
    for index, record in enumerate(records, start=1):
        run_dir = Path(record["run_dir"])
        eval_path = run_dir / EVAL_RESULT_FILENAME
        if record.get("resumed") and not re_eval and eval_path.exists():
            print(
                f"[eval {index}/{total}] arm={record['arm']} "
                f"seed={record['seed']} eval resumed ({EVAL_RESULT_FILENAME} "
                "exists, skipping recompute; --re-eval overrides)"
            )
            payload = _load_json_file(eval_path, error_cls=ProvenanceError)
            if not isinstance(payload, dict) or (
                payload.get("arm") != record["arm"]
                or payload.get("seed") != record["seed"]
            ):
                raise ProvenanceError(
                    f"Run {_run_name(record)}: resumed {EVAL_RESULT_FILENAME} "
                    "identifies itself as arm="
                    f"{payload.get('arm') if isinstance(payload, dict) else None!r} "
                    f"seed={payload.get('seed') if isinstance(payload, dict) else None!r} "
                    f"but the plan slot is arm={record['arm']!r} "
                    f"seed={record['seed']!r} (MA-001). Delete the stale run "
                    "dir or pass --re-eval."
                )
            results.append(payload)
        else:
            print(
                f"[eval {index}/{total}] arm={record['arm']} "
                f"seed={record['seed']}"
            )
            results.append(evaluate_run(record, test_questions, config))
        if prune:
            # MA-007: reclaim per run, immediately after its eval artifact
            # exists — never only after the entire pipeline.
            prune_run_checkpoints(run_dir)
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
        # MA-011 (additive): name the block's index conventions so it is
        # executable by hand — and explicitly NOT on the same index base as
        # the endpoint's 0-indexed buzz_position.
        "index_convention": (
            "per_position_mean_before/after: array index i is prefix "
            "position i+1 (positions are 1-indexed prefixes); "
            "expected_buzz_time_* are expected 1-indexed stop positions "
            "(forced commit at position T). NOTE eval_result runs[]."
            "buzz_position (the endpoint's source) is 0-indexed."
        ),
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
    save_json(Path(out_path), block, allow_nan=False)  # MA-017: strict JSON
    return block


def _probe_all_hazard_runs(
    records: list[dict[str, Any]],
    shared_ckpt: Path,
    probe_questions: list,
    *,
    re_eval: bool = False,
) -> None:
    """R-010b probe stage over every hazard run (extracted for MA-012/018).

    Prints a ``[probe j/M]`` banner per hazard run (MA-018) and, for
    RESUMED runs whose ``hazard_dynamics.json`` already exists, skips the
    recompute with a "probe resumed" print unless ``re_eval`` (MA-012:
    read-before-recompute at every stage granularity). The supervised
    "before" probe is shared across runs via the QA-007 cache.
    """
    hazard_records = [rec for rec in records if rec.get("hazard")]
    total = len(hazard_records)
    probe_cache: dict[str, Any] = {}
    for index, record in enumerate(hazard_records, start=1):
        run_dir = Path(record["run_dir"])
        out_path = run_dir / HAZARD_DYNAMICS_FILENAME
        if record.get("resumed") and not re_eval and out_path.exists():
            print(
                f"[probe {index}/{total}] arm={record['arm']} "
                f"seed={record['seed']} probe resumed "
                f"({HAZARD_DYNAMICS_FILENAME} exists, skipping recompute; "
                "--re-eval overrides)"
            )
            continue
        print(
            f"[probe {index}/{total}] arm={record['arm']} "
            f"seed={record['seed']}"
        )
        probe_and_write_hazard_dynamics(
            str(shared_ckpt),
            str(run_dir / "hazard" / "best_model"),
            probe_questions,
            run_dir / "hazard" / "hazard_history.json",
            out_path,
            before_cache=probe_cache,
        )


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

    removed = _delete_prunables(run_dir.resolve())
    # MA-014: prune narrates what it reclaimed and what it kept.
    print(
        f"[prune] {run_dir.name}: removed {removed['dirs']} checkpoint "
        f"dir(s) + {removed['files']} optimizer-state file(s), reclaimed "
        f"{removed['bytes']} bytes; kept best_model/, sidecars, and "
        f"{EVAL_RESULT_FILENAME}"
    )


def _delete_prunables(root: Path) -> dict[str, int]:
    """Delete ``iter_*``/``epoch_*`` dirs + ``training_state.pt`` under
    ``root`` with the QA-010 symlink/containment discipline.

    Shared by :func:`prune_run_checkpoints` and
    :func:`prune_shared_supervised_checkpoints` (MA-007). Returns
    ``{"dirs", "files", "bytes"}`` reclaim stats (MA-014 narration).
    """
    root = Path(root).resolve()

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

    reclaimed = 0
    n_dirs = 0
    n_files = 0
    for path in prunable_dirs:
        if not path.is_symlink() and path.is_dir() and _contained(path):
            reclaimed += _measure_disk_usage(path)
            shutil.rmtree(path)
            n_dirs += 1
    for state_file in state_files:
        if (
            not state_file.is_symlink()
            and state_file.is_file()
            and _contained(state_file)
        ):
            try:
                reclaimed += state_file.stat().st_size
            except OSError:
                pass
            state_file.unlink()
            n_files += 1
    return {"dirs": n_dirs, "files": n_files, "bytes": int(reclaimed)}


def prune_shared_supervised_checkpoints(out_dir: Path) -> None:
    """MA-007: reclaim the SHARED supervised tree's transient checkpoints.

    Deletes ``epoch_*``/``iter_*`` dirs and ``training_state.pt`` files
    under ``<out>/shared_supervised`` (the supervised trainer writes
    per-epoch checkpoints there; its discarded 1-iteration PPO phase may
    add ``iter_*``), keeping every ``best_model/`` dir and sidecar so the
    branch point stays loadable. Called only AFTER all arms complete
    (``--prune-checkpoints``); a missing shared root is a silent no-op
    (e.g. ``--report-only`` over a tree without one).
    """
    root = shared_supervised_root(Path(out_dir))
    if not root.is_dir():
        return
    removed = _delete_prunables(root.resolve())
    print(
        f"[prune] {root.name}: removed {removed['dirs']} checkpoint dir(s) "
        f"+ {removed['files']} optimizer-state file(s), reclaimed "
        f"{removed['bytes']} bytes; kept best_model/ dirs and sidecars"
    )


def _plot_arm_mean(
    runs: list[dict[str, Any]], arm: str, key: str
) -> float | None:
    """Mean of one numeric report key over an arm's runs; ``None`` if none.

    MA-009: a no-data arm must return ``None`` (rendered as a gap +
    annotation), NEVER 0.0 — on the lower-is-earlier buzz-position axis a
    coerced 0.0 reads as the OPTIMAL value.
    """
    values = []
    for run in runs:
        if run.get("arm") != arm:
            continue
        value = run.get(key)
        # bool is an int subclass; a flag must never plot as 0/1.
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        values.append(float(value))
    return float(np.mean(values)) if values else None


def write_plot(report: dict[str, Any], out_dir: Path) -> Path:
    """Write ``<out>/hazard_efficacy_plot.png`` headlessly; returns the path.

    Matplotlib is imported lazily inside this function with the Agg
    backend selected before pyplot, and the figure is persisted via
    ``savefig`` only. MA-009: an arm with no data for a metric renders as a
    visible NaN gap annotated "no data" — never as a 0.0 bar.
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

    def _bars(ax: Any, values: list[float | None], color: str) -> None:
        heights = [v if v is not None else float("nan") for v in values]
        ax.bar(arms, heights, color=color)
        for index, value in enumerate(values):
            if value is None:
                ax.annotate(
                    "no data",
                    xy=(index, 0.0),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                    color="dimgray",
                )

    fig, (ax_pos, ax_acc) = plt.subplots(1, 2, figsize=(10.0, 4.0))

    positions = [
        _plot_arm_mean(runs, arm, "mean_correct_buzz_position") for arm in arms
    ]
    _bars(ax_pos, positions, "steelblue")
    ax_pos.set_title("Mean correct-buzz position (lower = earlier)")
    ax_pos.set_xlabel("arm")
    ax_pos.set_ylabel("prefix position")

    accuracies = [_plot_arm_mean(runs, arm, "accuracy") for arm in arms]
    _bars(ax_acc, accuracies, "darkorange")
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
    """The three eval fields :func:`compute_primary_endpoint` reads per arm.

    MA-018: the metric floats are validated (finite real number or null) at
    this ingestion boundary so a doctored/NaN payload fails loud instead of
    propagating into the endpoint booleans.
    """
    where = (
        f"eval_result for arm={eval_result.get('arm')!r} "
        f"seed={eval_result.get('seed')!r}"
    )
    return {
        "mean_correct_buzz_position": _validated_metric(
            eval_result.get("mean_correct_buzz_position"),
            key="mean_correct_buzz_position",
            where=where,
        ),
        "accuracy": _validated_metric(
            eval_result.get("accuracy"), key="accuracy", where=where
        ),
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
                # MA-018: validated at the ingestion boundary — a NaN/inf
                # S_q must fail loud, never skew the paired bootstrap.
                validated = _validated_metric(
                    sq,
                    key="sq",
                    where=(
                        f"eval runs record arm={run_arm!r} seed={seed!r} "
                        f"qid={qid!r}"
                    ),
                )
                assert validated is not None  # sq is non-None here
                qid_map[str(qid)] = validated
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
        name = _run_name(record)
        config_used = _load_json_file(
            run_dir / "ppo_t5" / "config_used.json", error_cls=ProvenanceError
        )
        manifest = _load_json_file(
            run_dir / "ppo_t5" / "split_manifest.json", error_cls=ProvenanceError
        )
        eval_result = _load_json_file(
            run_dir / EVAL_RESULT_FILENAME, error_cls=ProvenanceError
        )
        if not isinstance(eval_result, dict):
            raise ProvenanceError(
                f"Run {name}: {EVAL_RESULT_FILENAME} is not an object"
            )
        # MA-001: the eval sidecar is self-identifying — it must name THIS
        # plan slot before feeding the report.
        if (
            eval_result.get("arm") != record["arm"]
            or eval_result.get("seed") != record["seed"]
        ):
            raise ProvenanceError(
                f"Run {name}: {EVAL_RESULT_FILENAME} identifies itself as "
                f"arm={eval_result.get('arm')!r} "
                f"seed={eval_result.get('seed')!r} but the plan slot is "
                f"arm={record['arm']!r} seed={record['seed']!r} (MA-001). "
                "Delete the stale run dir or pass --force to re-run "
                "everything."
            )
        # MA-016: uniqueness asserted at the report's ingestion boundary.
        runs_records = eval_result.get("runs")
        if isinstance(runs_records, list):
            assert_unique_qids(
                [r.get("qid") for r in runs_records if isinstance(r, dict)],
                where=f"run {name} {EVAL_RESULT_FILENAME} runs records",
            )
        if first_config is None:
            first_config, first_manifest = config_used, manifest

        # MA-018: a ProvenanceError from the provenance assembly names the
        # run whose sidecar broke it.
        try:
            provenance = collect_provenance(config_used)
        except ProvenanceError as exc:
            raise ProvenanceError(f"Run {name}: {exc}") from exc

        eval_by_run[(record["arm"], record["seed"])] = eval_result
        report_runs.append(
            {
                "arm": record["arm"],
                "seed": record["seed"],
                # MA-001: the run's hazard identity is echoed into its
                # report row so the report is auditable per run.
                "hazard": config_used.get("hazard")
                if isinstance(config_used, dict)
                else None,
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
                "provenance": provenance,
            }
        )

    assert first_config is not None and first_manifest is not None
    return eval_by_run, report_runs, first_config, first_manifest


def _build_scale(
    config_used: dict[str, Any], manifest: dict[str, Any], out_dir: Path
) -> dict[str, Any]:
    """Scale block: model, split sizes, PPO iterations, on-disk footprint.

    ``disk_usage_bytes`` is measured BEFORE the report and plot are written,
    so it reflects the run tree the report describes. MA-018: the disk walk
    never follows symlinks (no cycles, no out-of-tree bytes) and tolerates
    files vanishing mid-walk.
    """
    try:
        return {
            "model_name": config_used["model"]["model_name"],
            "n_train": manifest["train_count"],
            "n_val": manifest["val_count"],
            "n_test": manifest["test_count"],
            "ppo_iterations": config_used["ppo"]["iterations"],
            "device": config_used["model"]["device"],
            "disk_usage_bytes": _measure_disk_usage(Path(out_dir)),
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
    source_seed: int | None = None

    b_records = [rec for rec in run_records if rec["arm"] == "B"]
    if b_records:
        b_dir = Path(b_records[0]["run_dir"])
        source_seed = b_records[0]["seed"]
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
        if isinstance(hazard_dynamics, dict):
            # MA-010: single-replicate blocks carry explicit source labels.
            hazard_dynamics = {
                **hazard_dynamics,
                "source_arm": "B",
                "source_seed": source_seed,
            }

    hazard_compute = {
        "optimizer_steps": optimizer_steps,
        "wall_clock_seconds": hazard_wall_clock_seconds,
        "child_total_wall_clock_seconds": child_total_wall_clock_seconds,
        # MA-010: name the single replicate these numbers came from.
        "source_arm": "B" if b_records else None,
        "source_seed": source_seed,
        "step_matching_note": _HAZARD_STEP_MATCHING_NOTE,
    }
    return hazard_compute, hazard_dynamics


def _assert_hazard_step_parity(run_records: list[dict[str, Any]]) -> None:
    """MA-010: C's step-matching invariant is asserted at report time.

    Arm C is the STEP-MATCHED compute control — its whole point is running
    exactly as many hazard optimizer steps as arm B. For every seed where
    BOTH arms' ``hazard_history.json`` files exist, their step counts must
    be equal; a mismatch raises ``ProvenanceError`` naming the seed and
    counts. Missing histories are skipped (the blocks they feed are
    best-effort), never silently compared.
    """
    steps_by: dict[tuple[str, int], int | None] = {}
    for record in run_records:
        if record["arm"] not in ("B", "C"):
            continue
        history = _read_optional_json(
            Path(record["run_dir"]) / "hazard" / "hazard_history.json"
        )
        if history is None:
            continue
        steps = history.get("steps") if isinstance(history, dict) else None
        steps_by[(record["arm"], record["seed"])] = (
            len(steps) if isinstance(steps, list) else None
        )

    shared_seeds = {seed for (arm, seed) in steps_by if arm == "B"} & {
        seed for (arm, seed) in steps_by if arm == "C"
    }
    for seed in sorted(shared_seeds):
        b_steps = steps_by[("B", seed)]
        c_steps = steps_by[("C", seed)]
        if b_steps is not None and c_steps is not None and b_steps != c_steps:
            raise ProvenanceError(
                f"Arm C is the step-matched compute control, but at seed "
                f"{seed} it ran {c_steps} hazard optimizer step(s) vs arm "
                f"B's {b_steps}; the compute-confound control is void "
                "(MA-010). Inspect the hazard histories and re-run the "
                "mismatched arm."
            )


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
    # MA-010: the step-matched control's invariant is asserted from run
    # artifacts at report time, before any block is built from them.
    _assert_hazard_step_parity(run_records)
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
    save_json(out_dir / REPORT_FILENAME, report, allow_nan=False)  # MA-017
    return report


# ---------------------------------------------------------------------------
# main() composition
# ---------------------------------------------------------------------------


def _prune_all_runs(records: list[dict[str, Any]]) -> None:
    """Reclaim disk across every evaluated run dir (``--prune-checkpoints``)."""
    for record in records:
        prune_run_checkpoints(Path(record["run_dir"]))


def _print_plan(
    records: list[dict[str, Any]],
    splits: dict[str, Path] | None,
    *,
    supervised_argv: list[str] | None = None,
) -> None:
    """Print the full preflighted plan (R-013).

    ``supervised_argv`` (additive, MA-013): the shared supervised child's
    argv is part of the plan and is printed (and round-tripped by the
    caller) like every arm argv — it must never be the one child the
    dry-run cannot see.
    """
    print("=" * 60)
    print(f"HAZARD EFFICACY PLAN — {len(records)} run(s)")
    if splits:
        for split in ("train", "val", "test"):
            if split in splits:
                print(f"  split[{split}]: {splits[split]} (source: persisted artifacts)")
    if supervised_argv:
        print("  [shared supervised] runs FIRST; every arm branches from it")
        print(f"      argv: {' '.join(str(t) for t in supervised_argv)}")
    for index, record in enumerate(records, start=1):
        print(
            f"  [{index}/{len(records)}] {Path(record['run_dir']).name}  "
            f"arm={record['arm']} seed={record['seed']} "
            f"hazard={record['hazard']}"
            + (f" variant={record['variant']}" if record.get("variant") else "")
        )
        print(f"      argv: {' '.join(record['argv'])}")
    print("=" * 60)


def _free_disk_bytes(path: Path) -> int:
    """Free bytes on the filesystem holding ``path`` (nearest existing
    ancestor when the path itself does not exist yet)."""
    probe = Path(path)
    while not probe.exists():
        parent = probe.parent
        if parent == probe:
            break
        probe = parent
    return int(shutil.disk_usage(probe).free)


def _print_disk_preflight(
    args: argparse.Namespace, plan: list[dict[str, Any]], out_dir: Path
) -> None:
    """MA-007: byte budget at preflight — estimate vs free space.

    Estimate = number of planned checkpoint dirs (2 for the shared
    supervised tree + 1 per run + 1 extra per hazard run) x a per-checkpoint
    size that is MEASURED from the existing shared checkpoint when present,
    else ESTIMATED from the resolved model name. Printed with the plan;
    warns (never fails) when the estimate exceeds 80% of free space.
    """
    shared_ckpt = shared_supervised_checkpoint(Path(out_dir))
    if shared_ckpt.is_dir() and any(shared_ckpt.iterdir()):
        per_ckpt = float(_measure_disk_usage(shared_ckpt))
        source = "measured from the existing shared checkpoint"
    else:
        base_config = _resolved_child_base_config(
            str(args.config), bool(args.smoke)
        )
        model_name = base_config.get("model", {}).get("model_name")
        per_ckpt = _MODEL_SIZE_ESTIMATE_BYTES.get(
            str(model_name), _DEFAULT_MODEL_SIZE_ESTIMATE_BYTES
        )
        source = f"estimated for model {model_name!r}"
    n_ckpt_dirs = 2 + sum(
        1 + (1 if rec.get("hazard") else 0) for rec in plan
    )
    estimate = n_ckpt_dirs * per_ckpt
    free = _free_disk_bytes(Path(out_dir))
    print(
        f"  disk preflight: ~{n_ckpt_dirs} checkpoint dir(s) x "
        f"{per_ckpt / 1e9:.2f} GB ({source}) ≈ {estimate / 1e9:.2f} GB; "
        f"free: {free / 1e9:.2f} GB"
    )
    if estimate > free * 0.8:
        print(
            "WARNING: the planned checkpoint footprint exceeds 80% of free "
            "disk space (MA-007); consider --prune-checkpoints, fewer "
            "arms/seeds, or a smaller model."
        )


def _scan_unplanned_run_dirs(
    out_dir: Path, plan: list[dict[str, Any]]
) -> list[str]:
    """MA-015: disk-minus-plan reconciliation for ``--report-only``.

    Returns one warning per run-dir-shaped directory under ``out_dir``
    (carries a ``RUN_COMPLETE.json`` or a ``ppo_t5/`` tree) that the
    current plan does NOT name — it is silently excluded from the report,
    which the reader deserves to know.
    """
    out_dir = Path(out_dir)
    if not out_dir.is_dir():
        return []
    planned_names = {Path(rec["run_dir"]).name for rec in plan}
    warnings: list[str] = []
    for child in sorted(out_dir.iterdir()):
        if not child.is_dir() or child.is_symlink():
            continue
        if child.name in planned_names or child.name == _SHARED_SUPERVISED_DIRNAME:
            continue
        if (child / RUN_COMPLETE_MARKER).exists() or (child / "ppo_t5").is_dir():
            warnings.append(
                f"run dir {child.name} exists on disk but is NOT in the "
                "current plan; it is EXCLUDED from the report (MA-015: "
                "disk-minus-plan reconciliation)"
            )
    return warnings


def _print_closing_summary(
    report: dict[str, Any], out_dir: Path, *, pruned: bool
) -> None:
    """MA-014: closing summary = outcome + footprint + cleanup + paths."""
    verdict = report.get("verdict", {}) if isinstance(report, dict) else {}
    scale = report.get("scale", {}) if isinstance(report, dict) else {}
    print("=" * 60)
    print(f"VERDICT: {verdict.get('verdict')} — {verdict.get('scope')}")
    disk = scale.get("disk_usage_bytes")
    if isinstance(disk, int):
        print(f"Output tree disk usage: {disk / 1e9:.2f} GB ({disk} bytes)")
    if pruned:
        print(
            "Checkpoints pruned (--prune-checkpoints): iter_*/epoch_*/"
            "training_state.pt reclaimed; best_model dirs + sidecars kept."
        )
    else:
        print(
            "Cleanup hint: pass --prune-checkpoints to reclaim "
            "iter_*/epoch_*/training_state.pt once eval results exist."
        )
    print(f"Report written to {Path(out_dir) / REPORT_FILENAME}")
    print(f"Plot written to {Path(out_dir) / PLOT_FILENAME}")


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


def build_shared_supervised_argv(
    args: argparse.Namespace, out_dir: Path
) -> list[str]:
    """The ONE shared supervised child's argv (MA-013: planned + printed +
    round-tripped like every arm argv).

    Carries NO ``--skip-supervised``; ``--ppo-iterations 1`` (QA-007: the
    child's PPO phase is discarded — every arm re-runs PPO from the branch
    point); ``--skip-test-eval`` (MA-017: the unconditional full test-eval
    tail is pure waste for a discarded PPO phase); the smoke flags when
    smoke; and the ``supervised.checkpoint_dir=<root>`` positional override
    last.
    """
    sup_root = shared_supervised_root(Path(out_dir))
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
        # MA-017: the discarded PPO phase must not pay the full test-eval
        # tail either.
        "--skip-test-eval",
    ]
    if args.smoke:
        argv += list(_SMOKE_CHILD_FLAGS)
    argv.append(f"supervised.checkpoint_dir={sup_root}")
    return argv


def shared_manifest_for_reuse(
    out_dir: Path, test_qids: list
) -> dict[str, Any] | None:
    """MA-004: on reuse, the disjointness proof comes from the PRODUCER.

    Reads the shared supervised child's OWN recorded split
    (``<shared_root>/ppo_t5/split_manifest.json``, written by the real
    trainer's PPO phase) and asserts ITS ``train_qids`` are disjoint from
    the CURRENT invocation's test qids — a reused checkpoint after an
    artifact rebuild must never ship a disjointness proof computed from
    artifacts it never saw. Returns the producer manifest (the persisted
    supervised manifest is then derived from it), or ``None`` when the
    producer manifest does not exist (fresh build / legacy tree — the
    caller falls back to the current-invocation manifest, which IS the
    producer's input in the fresh case).
    """
    producer_path = (
        shared_supervised_root(Path(out_dir)) / "ppo_t5" / "split_manifest.json"
    )
    if not producer_path.exists():
        return None
    producer = _load_json_file(producer_path, error_cls=ProvenanceError)
    if not isinstance(producer, dict):
        raise ProvenanceError(
            f"{producer_path} is not an object; the shared checkpoint's "
            "recorded split cannot be validated (MA-004)."
        )
    overlap = sorted(
        set(producer.get("train_qids") or []) & set(test_qids)
    )
    if overlap:
        raise ProvenanceError(
            f"The reused shared supervised checkpoint's OWN recorded split "
            f"({producer_path}) trained on {len(overlap)} qid(s) that are "
            f"in the CURRENT test split (e.g. {overlap[:5]}); the "
            "supervised phase has seen test questions (MA-004 / R-008). "
            "Rebuild the shared checkpoint with --force or rebuild the "
            "artifacts."
        )
    return producer


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
    "model_name", "supervised_seed", "weights_sha256"}`` — the last is the
    MA-003 content fingerprint of the saved checkpoint). An existing
    non-empty shared checkpoint is reused ONLY when the marker exists, its
    ``config_hash``/``model_name`` match THIS invocation, AND (MA-003) the
    checkpoint bytes still hash to the marker's ``weights_sha256``
    (``ProvenanceError`` otherwise — a t5-small smoke checkpoint or a
    silently mutated/rebuilt checkpoint must never seed the comparison); a
    ``git_sha`` drift warns without raising (R-013 policy; the drift is not
    persisted — MA-018 wording). ``--force`` rebuilds the shared checkpoint
    unconditionally.

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
                # MA-018 wording: nothing persists this drift for the shared
                # checkpoint — say so ("recorded" only where written).
                print(
                    "WARNING: shared supervised checkpoint was built at git "
                    f"sha {marker.get('git_sha')!r} but the current checkout "
                    f"is {identity['git_sha']!r}; reusing it (warning only — "
                    "this drift is NOT persisted anywhere, not fatal)."
                )
            # MA-003: rebuilds/mutations of the branch point must be
            # detectable — the marker's content fingerprint must match the
            # bytes on disk right now.
            current_fp = _weights_fingerprint(shared_ckpt)
            recorded_fp = marker.get("weights_sha256")
            if recorded_fp is None:
                print(
                    "WARNING: shared supervised identity marker predates the "
                    "weight fingerprint (legacy marker); the checkpoint "
                    "content cannot be verified against its build (MA-003)."
                )
            elif recorded_fp != current_fp:
                raise ProvenanceError(
                    f"Shared supervised checkpoint at {shared_ckpt} hashes "
                    f"to weights sha256 {current_fp[:12]}… but its identity "
                    f"marker recorded {str(recorded_fp)[:12]}…; the "
                    "checkpoint content changed since it was built (MA-003). "
                    f"Delete {sup_root} or pass --force to rebuild it."
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
                    # MA-015: identity-marker rewrites are atomic.
                    _atomic_save_json(
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

    argv = build_shared_supervised_argv(args, out_dir)

    print("[supervised] shared warm-start started")
    start = time.monotonic()
    exit_code = _run_child(argv, sup_log)
    elapsed = time.monotonic() - start
    if exit_code != 0:
        raise ChildRunError(
            f"Shared supervised run failed with exit code {exit_code}; "
            f"log: {sup_log}\n--- log tail ---\n{_log_tail(sup_log)}\n"
            f"{_RESUME_GUIDANCE}"
        )
    if not shared_ckpt.is_dir():
        raise ChildRunError(
            f"Shared supervised run left no checkpoint at {shared_ckpt}; "
            f"inspect the child log: {sup_log}"
        )
    # MA-003: the branch point carries its own content identity; MA-015:
    # written atomically.
    _atomic_save_json(
        marker_path,
        {**identity, "weights_sha256": _weights_fingerprint(shared_ckpt)},
    )
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
    # MA-016: uniqueness asserted at the manifest-read boundary.
    assert_unique_qids(
        list(qids), where=f"{split} qids named by {manifest_path}"
    )
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
    routing them through the shared :func:`verify_run_records` gate
    (MA-002), reconciling the surviving dirs against the FULL plan
    (QA-R2-1: a wholly missing planned dir becomes a report warning, never
    a silent drop; MA-015: unplanned on-disk run dirs warn too) and
    deriving the report's smoke label from the runs' ``RUN_COMPLETE.json``
    markers (QA-R2-2: mixed smoke provenance raises ``ProvenanceError``;
    markers predating the field fall back to ``--smoke`` with a legacy
    warning — MA-015); ``--report-only --dry-run`` is plan-print only —
    zero writes and zero deletions, ``--prune-checkpoints`` included
    (QA-005). MA-006: ``--stall-timeout-minutes`` /
    ``--child-timeout-minutes`` arm the child watchdog. MA-014: a SIGINT
    abort prints the resume affordance before re-raising.
    """
    args = parse_args(argv)
    validate_flag_compatibility(args)
    # QA-012: absolutize path-shaped CLI inputs ONCE at the boundary so the
    # harness and its children (which inherit an arbitrary CWD) resolve the
    # same config file and output tree regardless of invocation directory.
    args.config = str(_resolve_config_path(str(args.config)))
    args.out_dir = str(Path(args.out_dir).resolve())
    out_dir = Path(args.out_dir)

    # MA-006: arm the child watchdog for this invocation.
    global _ACTIVE_STALL_TIMEOUT_SECONDS, _ACTIVE_CHILD_TIMEOUT_SECONDS
    stall_minutes = getattr(
        args, "stall_timeout_minutes", DEFAULT_STALL_TIMEOUT_MINUTES
    )
    _ACTIVE_STALL_TIMEOUT_SECONDS = (
        float(stall_minutes) * 60.0
        if stall_minutes is not None and stall_minutes > 0
        else None
    )
    child_minutes = getattr(args, "child_timeout_minutes", None)
    _ACTIVE_CHILD_TIMEOUT_SECONDS = (
        float(child_minutes) * 60.0
        if child_minutes is not None and child_minutes > 0
        else None
    )

    if args.report_only:
        _main_report_only(args, out_dir)
        return

    try:
        _main_execute(args, out_dir)
    except KeyboardInterrupt:
        # MA-014: abort messages state the resume invariant.
        print(f"\nInterrupted (SIGINT). {_RESUME_GUIDANCE}")
        raise


def _main_report_only(args: argparse.Namespace, out_dir: Path) -> None:
    """The ``--report-only`` entry path (see :func:`main` docstring)."""
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
    # report warning instead of a silent drop. MA-015: the opposite
    # direction too — unplanned on-disk run dirs are named.
    plan_warnings = _reconcile_planned_dirs(plan, records)
    plan_warnings += _scan_unplanned_run_dirs(out_dir, plan)
    # MA-002: --report-only routes through the SAME validated-records gate
    # as the execute path (complete classification + typed markers +
    # MA-001 identity + split source + arm control) before any assembly.
    verify_run_records(records)
    current_sha = _git_sha()
    marker_smoke_by_run: dict[str, bool] = {}
    legacy_marker_runs: list[str] = []
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
        elif marker is not None:
            legacy_marker_runs.append(_run_name(record))
    if legacy_marker_runs:
        # MA-015: legacy field-less markers must not SILENTLY skew the
        # smoke-cohort derivation.
        print(
            "WARNING: run marker(s) without the 'smoke' field (legacy "
            f"markers predating QA-R2-2): {sorted(legacy_marker_runs)}; "
            "they contribute nothing to the smoke-cohort check and the "
            "report label may fall back to this invocation's --smoke flag "
            "(MA-015)."
        )
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
    report = assemble_report(
        out_dir, records, smoke=report_smoke, extra_warnings=plan_warnings
    )
    pruned = False
    if args.prune_checkpoints:
        _prune_all_runs(records)
        prune_shared_supervised_checkpoints(out_dir)
        pruned = True
    # MA-014: closing summary (verdict + disk + cleanup + report AND plot
    # paths — report-only used to drop the plot line).
    _print_closing_summary(report, out_dir, pruned=pruned)


def _main_execute(args: argparse.Namespace, out_dir: Path) -> None:
    """The execute entry path: preflight -> children -> eval -> report."""
    # R-013 preflight: split artifacts + config + full plan validation
    # BEFORE any child. MA-013: a typo'd --config dies HERE, not after the
    # supervised phase; the shared supervised argv is planned, printed, and
    # round-tripped like every arm argv.
    splits = resolve_split_artifacts(smoke=bool(args.smoke))
    yaml_config = _load_yaml_config(str(args.config))
    plan = plan_runs(args)
    supervised_argv = build_shared_supervised_argv(args, out_dir)
    _roundtrip_child_argv(
        {
            "arm": "shared_supervised",
            "seed": int(args.seeds[0]),
            "run_dir": shared_supervised_root(out_dir),
            "hazard": False,
            "argv": supervised_argv,
            "variant": None,
        }
    )
    _print_plan(plan, splits, supervised_argv=supervised_argv)
    # MA-007: byte budget at preflight (plan print + warn, never fail).
    _print_disk_preflight(args, plan, out_dir)
    if args.dry_run:
        print("--dry-run: stopping after preflight; zero children launched.")
        return

    train_questions = load_mc_questions(splits["train"])
    val_questions = load_mc_questions(splits["val"])
    test_questions = load_mc_questions(splits["test"])
    if not train_questions or not test_questions:
        raise PreflightError(
            "Persisted split artifacts resolved but are empty "
            f"(train={len(train_questions)}, test={len(test_questions)}); "
            "rebuild them with scripts/build_mc_dataset.py"
        )
    # MA-016: uniqueness asserted at the artifact ingestion boundary.
    for split_name, questions in (
        ("train", train_questions),
        ("val", val_questions),
        ("test", test_questions),
    ):
        assert_unique_qids(
            [q.qid for q in questions],
            where=f"persisted {split_name} artifact {splits[split_name]}",
            error_cls=PreflightError,
        )

    # QA-002: resumed arm dirs must match what THIS invocation would produce
    # (model identity + current-artifact split membership).
    base_config = _resolved_child_base_config(str(args.config), bool(args.smoke))
    expected_run_context: dict[str, Any] = {
        "model_name": base_config.get("model", {}).get("model_name"),
        "artifact_qids": {
            "train": {q.qid for q in train_questions},
            "val": {q.qid for q in val_questions},
            "test": {q.qid for q in test_questions},
        },
    }

    # MA-013: resume state (partial dirs, resumed-dir validation) is decided
    # NOW — before the shared supervised child burns hours.
    preflight_resume_states(
        plan, force=bool(args.force), expected_run_context=expected_run_context
    )

    # MA-004: remember whether the shared checkpoint pre-existed (reuse) —
    # on reuse the disjointness proof must come from ITS recorded split.
    shared_ckpt_path = shared_supervised_checkpoint(out_dir)
    shared_reused = (
        shared_ckpt_path.is_dir()
        and any(shared_ckpt_path.iterdir())
        and not bool(args.force)
    )

    # Shared supervised warm-start: the FIRST child (R-003/R-008), identity-
    # validated on reuse and rebuilt under --force (QA-002).
    shared_ckpt = _run_shared_supervised(args, out_dir)
    # MA-003: the branch point's content identity for this invocation —
    # stamped into every fresh run's marker and asserted on resumed ones.
    shared_weights_sha256 = _weights_fingerprint(shared_ckpt)
    expected_run_context["shared_weights_sha256"] = shared_weights_sha256

    # Arm children with resume/partial/force semantics + arm control.
    # QA-R2-2: the invocation's smoke flag is persisted into each fresh
    # run's completion marker so --report-only can relabel honestly.
    records = execute_plan(
        plan,
        force=bool(args.force),
        expected_run_context=expected_run_context,
        smoke=bool(args.smoke),
    )

    # MA-002: the execute path routes through the SAME validated-records
    # gate as --report-only before anything report-shaped is derived.
    verify_run_records(records)

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

    # R-008: persist the supervised split manifest, asserting supervised-
    # train/test disjointness. MA-004: when the shared checkpoint was
    # REUSED, the proof (and the persisted manifest) comes from the
    # producer's own recorded split, never from artifacts it never saw.
    manifest: dict[str, Any] | None = None
    if shared_reused:
        manifest = shared_manifest_for_reuse(
            out_dir, [q.qid for q in test_questions]
        )
    if manifest is None:
        manifest = _manifest_from_artifacts(
            splits, train_questions, val_questions, test_questions
        )
    write_supervised_split_manifest(shared_ckpt, manifest)

    # R-005/R-014: identical eval path per run on the TEST split, persisted
    # immediately per run. MA-007: per-run prune right after each
    # eval_result.json when requested; MA-012: resumed runs with eval
    # artifacts are read, not recomputed.
    eval_context = {
        "config": yaml_config,
        "reference_questions": train_questions,
        "test_set_source": _PERSISTED_SOURCE,
    }
    evaluate_all_runs(
        records,
        test_questions,
        eval_context,
        prune=bool(args.prune_checkpoints),
        re_eval=bool(getattr(args, "re_eval", False)),
    )

    # R-010b: stop-probability probe per hazard run (supervised vs hazard).
    # QA-007: the supervised "before" probe is computed once via the shared
    # cache. MA-003: the probe's "before" leg must span exactly ONE
    # checkpoint content — re-verified right before probing.
    probe_questions = select_probe_questions(train_questions)
    current_fp = _weights_fingerprint(shared_ckpt)
    if current_fp != shared_weights_sha256:
        raise ProvenanceError(
            "Shared supervised checkpoint content changed mid-pipeline "
            f"(weights sha256 {shared_weights_sha256[:12]}… -> "
            f"{current_fp[:12]}…); the probe's before-leg would span two "
            "different checkpoints (MA-003)."
        )
    _probe_all_hazard_runs(
        records,
        shared_ckpt,
        probe_questions,
        re_eval=bool(getattr(args, "re_eval", False)),
    )

    # R-009/R-014: report + plot, read exclusively from per-run files.
    report = assemble_report(out_dir, records, smoke=bool(args.smoke))

    pruned = False
    if args.prune_checkpoints:
        # Per-run prune already ran inside evaluate_all_runs (MA-007);
        # the SHARED tree is reclaimed only after all arms complete.
        prune_shared_supervised_checkpoints(out_dir)
        pruned = True

    # MA-014: closing summary — outcome + footprint + cleanup + paths.
    _print_closing_summary(report, out_dir, pruned=pruned)


if __name__ == "__main__":
    main()
