# STUB:TDD
"""Hazard-pretrain efficacy eval harness (RED-phase structural stub).

Orchestrates the paired WITH/WITHOUT/compute-control comparison for the
``--hazard-pretrain`` warm-start bridge (spec: ``.correctless/specs/
hazard-efficacy-eval.md``, Track B rules R-003, R-005..R-009, R-010b,
R-011..R-014).

Pipeline (GREEN implements; tests in ``tests/test_hazard_efficacy_*.py``
pin the behavior):

1. Preflight (R-013): resolve persisted split artifacts, plan every
   arm x seed run up front (distinct dirs ``<out>/<arm>_seed<k>``, full
   child argv lists), print the plan; ``--dry-run`` stops here.
2. Shared supervised warm-start runs ONCE; its checkpoint is branched
   into every arm via ``--skip-supervised --model-path <shared>`` and a
   split manifest is persisted next to it (R-003 / R-008).
3. Child training runs via ``scripts/train_t5_policy.py`` argv LISTS,
   ``shell=False``, tee'd to ``<run_dir>/train.log`` (R-011); complete
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
   (Agg backend, ``savefig`` never ``show``) (R-009 / R-014).

Every public function below is a structural stub raising
``NotImplementedError``; signatures, constants, exception types, and the
documented record/report schemas are the pinned interface the GREEN
phase must implement against.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Pinned constants (interface — values are spec-mandated)
# ---------------------------------------------------------------------------

SCHEMA_VERSION = 1
DEVICE2_CAVEAT = "Full-scale t5-large efficacy remains a Device-2 (RTX 5090) run."
RUN_COMPLETE_MARKER = "RUN_COMPLETE.json"
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
    # STUB:TDD
    raise NotImplementedError


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
    """
    # STUB:TDD
    raise NotImplementedError


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
    # STUB:TDD
    raise NotImplementedError


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
    """
    # STUB:TDD
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Child execution / resume
# ---------------------------------------------------------------------------


def _run_child(argv: list[str], log_path: Path) -> int:
    """Run one child via subprocess (shell=False), tee output to log_path.

    Single injectable seam: tests monkeypatch
    ``scripts.run_hazard_efficacy._run_child``. Returns the exit code.
    """
    # STUB:TDD
    raise NotImplementedError


def check_child_outputs(record: dict[str, Any]) -> None:
    """Fail loud when ``<run_dir>/ppo_t5/best_model`` is missing (R-011).

    Raises ``ChildRunError`` naming the run (arm, seed) and the log path.
    """
    # STUB:TDD
    raise NotImplementedError


def classify_run_dir(run_dir: Path, *, hazard: bool = False) -> str:
    """Classify a run dir for resume (R-013).

    Returns ``"complete"`` (RUN_COMPLETE.json + ppo_t5/best_model +
    config_used.json + split_manifest.json, plus hazard/best_model for
    hazard arms), ``"partial"`` (some outputs, incomplete), or
    ``"fresh"`` (absent/empty).
    """
    # STUB:TDD
    raise NotImplementedError


def execute_plan(
    records: list[dict[str, Any]], *, force: bool = False
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
    "wall_clock_seconds"}``; ``wall_clock_seconds`` is the child's
    elapsed run time in seconds measured around :func:`_run_child` —
    a float >= 0). Ends by running :func:`assert_arm_control` over
    all records (resumed dirs included). Returns updated records.
    """
    # STUB:TDD
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Arm control / provenance (R-003 / R-008)
# ---------------------------------------------------------------------------


def assert_arm_control(records: list[dict[str, Any]]) -> None:
    """Diff every run's config_used.json / split_manifest.json (R-003).

    All config keys must be equal across runs EXCEPT the top-level
    ``hazard`` block, top-level ``seed``, and any key path whose last
    component is ``checkpoint_dir``. All split manifests must carry
    identical train/val/test qids. Any mismatch raises
    ``ArmControlError`` naming the offending arm (and key).
    """
    # STUB:TDD
    raise NotImplementedError


def write_supervised_split_manifest(
    supervised_ckpt_dir: Path, manifest: dict[str, Any]
) -> Path:
    """Persist the split manifest next to the shared supervised ckpt (R-008).

    Asserts test qids are disjoint from supervised-train qids
    (``ProvenanceError`` on overlap) and writes ``split_manifest.json``
    into ``supervised_ckpt_dir``. Returns the written path.
    """
    # STUB:TDD
    raise NotImplementedError


def collect_provenance(config_used: dict[str, Any]) -> dict[str, Any]:
    """Assemble one run's provenance block (R-008), complete or fail-loud.

    Returns ``{"model_name", "seed", "device", "git_sha", "git_dirty",
    "torch_version", "platform"}`` using real ``git rev-parse HEAD`` /
    ``git status --porcelain``. A missing source field raises
    ``ProvenanceError`` (no report may be written).
    """
    # STUB:TDD
    raise NotImplementedError


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
    """
    # STUB:TDD
    raise NotImplementedError


def evaluate_all_runs(
    records: list[dict[str, Any]], test_questions: list, config: dict[str, Any]
) -> list[dict[str, Any]]:
    """Evaluate every run: exactly one ``evaluate_t5_policy`` call each,
    identical test split and kwargs (except checkpoint path) across calls.
    Per-run ``eval_result.json`` files already written are never deleted
    by a later failure (R-014)."""
    # STUB:TDD
    raise NotImplementedError


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
    # STUB:TDD
    raise NotImplementedError


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
    # STUB:TDD
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Stop-probability probe / hazard dynamics (R-010b)
# ---------------------------------------------------------------------------


def select_probe_questions(train_questions: list) -> list:
    """Deterministic probe sample: first ``min(32, len(train))`` train
    questions in split order (R-010)."""
    # STUB:TDD
    raise NotImplementedError


def stop_prob_probe(model: Any, questions: list) -> list[list[float]]:
    """Probe P(BUZZ) per prefix position for each question (R-010b).

    Runs the model in ``eval()`` mode under ``torch.no_grad()`` (dropout
    off => deterministic). Returns one list per question of length
    ``T_q`` (its prefix count), each value a probability in [0, 1].
    """
    # STUB:TDD
    raise NotImplementedError


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
    # STUB:TDD
    raise NotImplementedError


def probe_and_write_hazard_dynamics(
    supervised_ckpt: str,
    hazard_ckpt: str,
    questions: list,
    hazard_history_path: Path,
    out_path: Path,
) -> dict[str, Any]:
    """Load both checkpoints, probe them, and persist the R-010b block.

    Writes :func:`build_hazard_dynamics` output as JSON to ``out_path``
    (``<run_dir>/hazard_dynamics.json`` in the pipeline) and returns it.
    """
    # STUB:TDD
    raise NotImplementedError


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
    """
    # STUB:TDD
    raise NotImplementedError


def write_plot(report: dict[str, Any], out_dir: Path) -> Path:
    """Write ``<out>/hazard_efficacy_plot.png`` via Agg + ``savefig``
    (never ``show``); returns the plot path."""
    # STUB:TDD
    raise NotImplementedError


def assemble_report(
    out_dir: Path,
    run_records: list[dict[str, Any]],
    *,
    smoke: bool = False,
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
    read from arm B's ``RUN_COMPLETE.json`` marker), ``hazard_dynamics``,
    ``runs`` (per-run records incl. ``arm``, ``seed``, ``resumed``,
    ``policy_buzz_rate``, ``forced_commit_rate``, ``ece``, ``brier``,
    ``provenance``), and ``plot_path`` (relative
    ``hazard_efficacy_plot.png``). Never contains any Expected Wins key.
    """
    # STUB:TDD
    raise NotImplementedError


def main(argv: list[str] | None = None) -> None:
    """Harness entry point: preflight -> plan -> execute -> eval -> report.

    ``--dry-run`` stops after preflight+plan with zero children;
    ``--report-only`` performs zero training/eval calls (and no split
    preflight) and reassembles report + plot from existing run dirs.

    Composition contract (pinned by the main() happy-path test): the
    shared supervised warm-start phase itself runs through
    :func:`_run_child` as the FIRST child; its argv carries NO
    ``--skip-supervised`` and directs outputs via a
    ``supervised.checkpoint_dir=<root>`` positional override, so the real
    trainer leaves the branched checkpoint at
    ``<root>/supervised/best_model`` (see
    ``training/train_supervised_t5.py``) — the exact ``--model-path``
    every arm child receives. Evaluation uses the TEST split questions
    loaded from the persisted artifacts (never train's), via the
    module-level ``evaluate_t5_policy``.
    """
    # STUB:TDD
    raise NotImplementedError


if __name__ == "__main__":
    main()
