#!/usr/bin/env python3
"""
Compute CSLI (Choice-Set Leakage Index) via local model panel.

This script emits two related quantities so downstream consumers (audit
card, manuscript, figures) can show either interpretation. The frozen
gate is on per-model choices-only accuracy and is independent of which
quantity a consumer chooses to display.

* ``csli`` (per-model) and ``panel_csli`` -- CANONICAL CSLI:
  ``max(0, acc_choices_only - 1/K)`` (PAP-original choices-only excess
  over chance). Larger excess = more leakage from choices alone.
  Always >= 0. The panel-level summary carries a percentile bootstrap
  CI computed over the per-question choices-only-correct panel mean.

* ``question_use_gap`` (per-model) and ``panel_question_use_gap`` --
  FORMER IN-FLIGHT-MANUSCRIPT CSLI: ``acc_full - acc_choices_only``.
  Large gap = model uses the question (low leakage in the manuscript's
  inverted-semantic reading). Can be negative when choices-only
  accuracy exceeds full accuracy. Kept for transparency under the
  ``panel_question_use_gap`` block; not the headline value.

* The frozen leakage gate (threshold_manifest.json) is on
  per-model choices-only accuracy:
  ``max(acc_choices_only) > 1/K + 0.05 = 0.30`` for K=4 raises a
  leakage flag. The gate verdict depends only on the per-model
  ``acc_choices_only`` values, NOT on either summary's panel value.

PR #14 follow-up review (Blocker 3) named CSLI=gap as a load-bearing
naming bug: the PAP defined CSLI as the choices-only excess, while
the in-flight manuscript adopted the gap without renaming. The audit
card was publishing the gap under the ``CSLI`` headline. This script
now publishes the canonical PAP definition under ``csli`` /
``panel_csli`` and keeps the gap under ``question_use_gap`` /
``panel_question_use_gap`` for transparency.

IMPORTANT -- DATA-05 SYMBOL COLLISION GUARD:
    DO NOT import evaluation.controls.run_choices_only_control -- that function
    uses surface-feature logistic regression (char n-gram TF-IDF sklearn), which
    is a DIFFERENT experiment measuring surface-feature artifacts. This script
    implements a local-model panel approach: TF-IDF similarity, SBERT embeddings,
    and T5-small log-likelihood scoring. The two approaches measure fundamentally
    different constructs.

LIMITATION -- WR-07 PRE-FRESH-SPLIT DISTRACTOR PROVENANCE:
    mc_dataset.json carries MC options whose distractors were chosen
    by scripts/build_mc_dataset.py using the OLD train_dataset.json as
    the reference_questions pool. If scripts/fresh_split.py was then
    run with a new seed, the test_dataset.json on disk reflects the
    NEW partition but the distractors in mc_dataset.json were assembled
    against the OLD partition. CSLI here is therefore computed on
    options whose construction predates the freeze, which means the
    "fresh split" integrity claim does NOT extend to MC construction.

    Remediations, in preference order:
      (a) Re-run scripts/build_mc_dataset.py AFTER scripts/fresh_split.py
          so distractor selection respects the new train pool. This is
          the methodologically correct fix. WR-05 already made this
          script accept both on-disk shapes for test_dataset.json so
          the rebuild is unblocked.
      (b) Quantify the realistic effect size (audit how many NEW test
          items had their gold option used as a distractor for another
          NEW test item) and document the residual bias in the
          manuscript Limitations section.

    At runtime, this script prints whether a pre-freshsplit data
    archive exists; that is the signal an operator should see before
    interpreting any CSLI value here as a clean fresh-split metric.

Usage:
    python scripts/compute_csli.py --help
    python scripts/compute_csli.py --smoke           # 1 model, 10 questions
    python scripts/compute_csli.py --dry-run         # Parse args, no compute
    python scripts/compute_csli.py --models tfidf,sbert,t5-small

Inputs:
    data/processed/mc_dataset.json   (MC questions with options, gold_index)
    data/processed/test_dataset.json (test split for qid filtering)
    data/processed/build_metadata.json (optional MC raw/retained/drop counts)
    PROJECT_WIKI/SPLIT_PROVENANCE.md (optional fresh-split provenance)
    data/processed.pre_v10_freshsplit_*/  (optional, used only for the
                                          WR-07 distractor-provenance warning)

Outputs:
    paper_exports/csli.json  (panel CSLI results with bootstrap CIs)

Exit codes:
    0 = success
    1 = runtime error
    2 = argument error
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, cast

import numpy as np

# ============================================================================
# SYMBOL_COLLISION_GUARD
# ============================================================================
# DO NOT import evaluation.controls.run_choices_only_control
# DO NOT import from evaluation.controls or from evaluation import controls
#
# That function uses surface-feature logistic regression (char-trigram TF-IDF),
# which is a different experiment from the local-model panel approach here.
# See DATA-05 in MASTER_PLAN_v10 for the symbol collision prohibition.
#
# The runtime guard below makes DATA-05 enforced (not comment-only). The
# pytest in tests/test_data05_symbol_collision.py adds a static AST check
# so a violating edit fails CI before it can ship.
# ============================================================================

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

_DATA05_DISALLOWED = ("evaluation.controls",)


def _assert_no_controls_import() -> None:
    """Fail loud if ``evaluation.controls`` (or its members) leaked into
    ``sys.modules`` via a future edit.

    DATA-05 (MASTER_PLAN_v10) symbol-collision guard. This script's
    panel must NOT call ``evaluation.controls.run_choices_only_control``
    -- that is a different experiment (surface-feature logistic
    regression on char n-grams) measuring a different construct.
    """
    for name in _DATA05_DISALLOWED:
        if name in sys.modules:
            raise ImportError(
                f"DATA-05 violation: {name} is imported in compute_csli.py. "
                "Use the local-model panel only -- not evaluation.controls. "
                "See MASTER_PLAN_v10 DATA-05 and Phase 02 review WR-01."
            )


_assert_no_controls_import()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_OUTPUT = PROJECT_ROOT / "paper_exports" / "csli.json"
DEFAULT_MODELS = "tfidf,sbert,t5-small"
THRESHOLD_MANIFEST = PROJECT_ROOT / "threshold_manifest.json"
SPLIT_PROVENANCE = PROJECT_ROOT / "PROJECT_WIKI" / "SPLIT_PROVENANCE.md"
SUPPORTED_MODELS = frozenset({"tfidf", "sbert", "t5-small"})
BOOTSTRAP_SEED = 789685

# Lazy-loaded model caches.
#
# Iter1 IN-03: each cache is protected by a dedicated threading.Lock
# so concurrent callers (downstream Phase 4/5/6 orchestrators that
# import these helpers from multiple worker threads) do not race the
# non-atomic test-and-set on the global. The double-checked locking
# pattern (fast-path None check without lock + re-check inside the
# lock) keeps the hot path lock-free once the cache is warm.
#
# Each model/cache gets its own lock so that loading T5 does not block
# a concurrent thread loading SBERT (the locks would serialize all
# model loads pointlessly if there were a single lock). The prior
# loaders intentionally acquire only their own lock and then call
# _get_sbert_model() -- the nested order is always prior_lock then
# model_lock and never reversed, so there is no deadlock potential.
#
# Prefers threading.Lock over functools.lru_cache because (a) the
# cached objects are heavy (SBERT model, T5 model, ~14k-row TF-IDF
# matrix) and the explicit lock makes the protected region obvious
# to readers, and (b) lru_cache does not provide an in-process
# "invalidate" hook -- which the prior loaders need when
# _set_answer_prior_train_path is re-called with a new path.
_SBERT_MODEL = None
_T5_MODEL = None
_T5_TOKENIZER = None
_SBERT_LOCK = threading.Lock()
_T5_LOCK = threading.Lock()  # protects BOTH _T5_MODEL and _T5_TOKENIZER together

# Iter1 IN-01 ANSWER-PRIOR CACHES.
#
# The choices-only scoring for TF-IDF and SBERT now scores each option
# against a "generic answer prior" assembled from training-split
# answer_primary texts (replacement for the prior folklore heuristic
# "most dissimilar = most specific" -- see docstrings on
# _score_tfidf_choices_only and _score_sbert_choices_only). The prior
# is stable across the test split, so we lazy-load and cache it. The
# train split path is injected from main() via
# _set_answer_prior_train_path so the --data-dir override flows
# through (the prior must follow the same data partition the rest of
# the script is using).
_ANSWER_PRIOR_TRAIN_PATH: Optional[Path] = None
_TFIDF_ANSWER_PRIOR = None  # (vectorizer, centroid_vector)
_SBERT_ANSWER_PRIOR = None  # centroid embedding (np.ndarray)
_TFIDF_PRIOR_LOCK = threading.Lock()
_SBERT_PRIOR_LOCK = threading.Lock()


def _sha256_file(path: Path) -> str:
    """Return the SHA-256 digest for a local artifact."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: Path) -> str:
    """Return a stable project-relative path when possible."""
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path)


def _git_output(args: list[str]) -> str | None:
    """Return stdout for a read-only git command, or None when unavailable."""
    try:
        # SECURITY-REVIEW: subprocess is fixed-argv, shell=False, and scoped to
        # local read-only git metadata; no user-controlled shell interpolation.
        result = subprocess.run(
            ["git", *args],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _build_generation_provenance(
    argv: list[str],
    *,
    output_path: Path,
    data_dir: Path,
) -> dict[str, Any]:
    """Build audit metadata tying a generated artifact to code and command.

    PR #14 follow-up review (Codex 3308444266): the build_metadata path is
    derived from the resolved ``data_dir`` so non-default invocations
    (e.g., ``--data-dir artifacts/smoke`` via the Modal smoke pipeline)
    capture the git status of the build metadata that actually governed
    the run, not the production ``data/processed/build_metadata.json``.
    Without this, a smoke artifact could record ``git_dirty: false`` even
    when the smoke build metadata was dirty or stale.
    """
    script_path = Path(__file__).resolve()
    # PR #14 follow-up review (Codex #3308590294): filter out-of-repo paths
    # from the git pathspec or `git status -- <abs_path>` aborts with
    # `fatal: ... is outside repository`, silently flipping git_dirty to
    # False and erasing git_status_relevant_paths exactly when an absolute
    # --output is in use. The display path list retains everything so the
    # output_path field is still accurate.
    display_paths = [
        _display_path(script_path),
        _display_path(output_path),
        _display_path(THRESHOLD_MANIFEST),
        _display_path(data_dir / "build_metadata.json"),
        _display_path(SPLIT_PROVENANCE),
    ]
    repo_relative_pathspec = [p for p in display_paths if not Path(p).is_absolute()]
    status_args = ["status", "--short", "--", *repo_relative_pathspec]
    git_status = _git_output(status_args)

    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": ["python", "scripts/compute_csli.py", *argv],
        "argv": list(argv),
        "cwd": _display_path(Path.cwd()),
        "output_path": _display_path(output_path),
        "script_path": _display_path(script_path),
        "script_sha256": _sha256_file(script_path),
        "git_commit": _git_output(["rev-parse", "HEAD"]),
        "git_dirty": bool(git_status),
        "git_status_relevant_paths": git_status or "",
    }


def _parse_split_provenance(path: Path) -> dict[str, str]:
    """Parse ``KEY=VALUE`` provenance fields from SPLIT_PROVENANCE.md."""
    fields: dict[str, str] = {}
    pattern = re.compile(r"^([A-Z0-9_]+)=(.+)$")
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        match = pattern.match(raw_line.strip())
        if match:
            fields[match.group(1)] = match.group(2).strip()
    return fields


def _load_split_provenance(data_dir: Path) -> dict[str, Any]:
    """Load durable fresh-split provenance when the default data dir is used."""
    provenance: dict[str, Any] = {
        "test_split_provenance_status": "unverified",
        "test_split_provenance_path": _display_path(SPLIT_PROVENANCE),
        "test_split_provenance_sha256": None,
        "fresh_split_seed": None,
        "fresh_split_created_at": None,
        "fresh_split_commit_sha": None,
        "threshold_manifest_sha256": None,
        "test_split_seed": None,
        "test_split_provenance_note": None,
    }

    try:
        is_default_data_dir = data_dir.resolve() == DEFAULT_DATA_DIR.resolve()
    except OSError:
        is_default_data_dir = False

    if not is_default_data_dir:
        provenance["test_split_provenance_note"] = (
            "unverified because --data-dir does not match the frozen default "
            f"data directory ({DEFAULT_DATA_DIR})"
        )
        return provenance

    if not SPLIT_PROVENANCE.exists():
        provenance["test_split_provenance_note"] = (
            "unverified because PROJECT_WIKI/SPLIT_PROVENANCE.md is missing"
        )
        return provenance

    fields = _parse_split_provenance(SPLIT_PROVENANCE)
    provenance["test_split_provenance_sha256"] = _sha256_file(SPLIT_PROVENANCE)
    provenance["fresh_split_seed"] = _maybe_int(fields.get("FRESH_SPLIT_SEED"))
    provenance["fresh_split_created_at"] = fields.get("FRESH_SPLIT_CREATED_AT")
    provenance["fresh_split_commit_sha"] = fields.get("FRESH_SPLIT_COMMIT_SHA")
    provenance["threshold_manifest_sha256"] = fields.get("THRESHOLD_MANIFEST_SHA256")

    manifest_hash_ok = False
    if THRESHOLD_MANIFEST.exists() and provenance["threshold_manifest_sha256"]:
        manifest_hash_ok = (
            _sha256_file(THRESHOLD_MANIFEST)
            == provenance["threshold_manifest_sha256"]
        )

    manifest_seed_ok = False
    if THRESHOLD_MANIFEST.exists() and provenance["fresh_split_seed"] is not None:
        with open(THRESHOLD_MANIFEST, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        manifest_seed_ok = str(manifest.get("fresh_split_seed_at_freeze")) == str(
            provenance["fresh_split_seed"]
        )

    if manifest_hash_ok and manifest_seed_ok:
        provenance["test_split_provenance_status"] = "verified"
        provenance["test_split_seed"] = provenance["fresh_split_seed"]
        provenance["test_split_provenance_note"] = (
            "verified against PROJECT_WIKI/SPLIT_PROVENANCE.md and "
            "threshold_manifest.json"
        )
    else:
        missing_or_mismatch = []
        if not manifest_hash_ok:
            missing_or_mismatch.append("threshold manifest hash mismatch")
        if not manifest_seed_ok:
            missing_or_mismatch.append("fresh split seed mismatch")
        provenance["test_split_provenance_note"] = (
            "unverified: " + "; ".join(missing_or_mismatch)
        )

    return provenance


def _maybe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _load_mc_build_metadata(data_dir: Path) -> dict[str, Any]:
    """Load MC-construction retention metadata when present."""
    path = data_dir / "build_metadata.json"
    summary: dict[str, Any] = {
        "status": "missing",
        "source_path": _display_path(path),
        "source_sha256": None,
        "test": None,
        "retention_thresholds": None,
    }
    if not path.exists():
        return summary

    try:
        with open(path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        test = metadata["splits"]["test"]
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Invalid MC build metadata at {path}: {exc}") from exc

    summary["status"] = "loaded"
    summary["source_sha256"] = _sha256_file(path)
    summary["retention_thresholds"] = metadata.get("retention_thresholds")
    summary["test"] = {
        "raw_count": int(test["raw_count"]),
        "retained_count": int(test["retained_count"]),
        "dropped_count": int(test["dropped_count"]),
        "retention_rate": float(test["retention_rate"]),
        "drop_reasons": test.get("drop_reasons", {}),
    }
    return summary


def _metadata_retention_threshold(
    build_metadata: dict[str, Any],
    *,
    smoke: bool,
    explicit_threshold: float | None,
) -> float:
    if explicit_threshold is not None:
        return explicit_threshold

    thresholds = build_metadata.get("retention_thresholds")
    if isinstance(thresholds, dict):
        key = "smoke" if smoke else "full"
        try:
            return float(thresholds[key])
        except (KeyError, TypeError, ValueError):
            pass

    return 0.98


def _filter_test_mc_questions(
    mc_questions: list[dict],
    test_qids: set[str],
) -> tuple[list[dict], dict[str, Any]]:
    """Filter MC rows to test qids and compute coverage by unique qid.

    PR #14 follow-up review (Issue B): coverage block uses split-neutral
    field names (``target_qids``, ``matched_questions``, ``matched_qids``,
    ``missing_qids``) so the same dict can describe a val or test split
    without leaking the test-only naming convention.
    """
    questions = [q for q in mc_questions if str(q["qid"]) in test_qids]
    matched_qids = {str(q["qid"]) for q in questions}
    missing_qids = test_qids - matched_qids
    coverage = len(matched_qids) / max(1, len(test_qids))
    return questions, {
        "target_qids": len(test_qids),
        "mc_questions_total": len(mc_questions),
        "matched_questions": len(questions),
        "matched_qids": len(matched_qids),
        "missing_qids": len(missing_qids),
        "missing_qids_set": missing_qids,
        "coverage_rate": coverage,
    }


def bootstrap_ci(
    values: np.ndarray,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float, float]:
    """Compute bootstrap confidence interval for the mean.

    Uses the **percentile method** (not BCa or studentized bootstrap).
    The percentile method is simpler but may undercover for skewed
    distributions. For the CSLI per-question indicators (discrete values
    in {-1, -2/3, ..., 2/3, 1}), this provides a CI that captures
    sampling variability (which questions are in the test set) but does
    NOT account for model selection uncertainty or between-model variance.

    Parameters
    ----------
    values : np.ndarray
        1-D array of observed values (e.g., per-item correctness indicators).
    n_resamples : int
        Number of bootstrap resamples.
    confidence : float
        Confidence level (default 0.95 for 95% CI).
    seed : int
        Random seed for reproducibility. Defaults to 789685 (the
        FRESH_SPLIT_SEED in PROJECT_WIKI/SPLIT_PROVENANCE.md). MUST NOT
        be in fresh_split.FORBIDDEN_SEEDS ({42, 13}); see WR-03 in the
        Phase 02 review for the rationale -- the prior default (42)
        contradicted the project seed-hygiene policy at the API surface
        even though the production call site (compute_panel_csli) always
        passed seed=789685 explicitly.

    Returns
    -------
    tuple[float, float, float]
        (mean, ci_lower, ci_upper) where ci_lower and ci_upper are the
        percentile-based confidence interval bounds.

    Raises
    ------
    ValueError
        If ``values`` is not 1-D, ``n_resamples`` is not positive, or
        ``confidence`` is not strictly between 0 and 1. Iter1 IN-02 added
        these guards because the prior implementation silently produced
        nan (via empty-resample arrays and 0/1 percentiles) on invalid
        inputs, contaminating the audit JSON.
    """
    # Iter1 IN-02: input validation. The prior implementation silently
    # produced degenerate outputs on invalid inputs:
    #   - values.ndim > 1 made `rng.choice(values, size=n)` resample
    #     along axis 0 with implicit broadcasting, producing
    #     unexpected per-row sub-statistics that np.mean averaged into
    #     a number bearing no relation to the intended CI.
    #   - n_resamples <= 0 produced an empty `means` array and
    #     np.percentile returned nan with a "Mean of empty slice"
    #     RuntimeWarning that consumers ignored.
    #   - confidence outside (0, 1) produced out-of-range percentiles
    #     (e.g., -2.5 or 102.5) which np.percentile clamps to extrema,
    #     silently returning the min/max of `means` and not a CI at all.
    # All three failure modes are user errors, so raising ValueError is
    # the right contract (the audit JSON should not carry nan or
    # silently-truncated CIs).
    values = np.asarray(values)
    if values.ndim != 1:
        raise ValueError(
            f"bootstrap_ci expects 1-D input, got shape {values.shape}"
        )
    if n_resamples <= 0:
        raise ValueError(
            f"bootstrap_ci n_resamples must be positive, got {n_resamples}"
        )
    if not 0 < confidence < 1:
        raise ValueError(
            f"bootstrap_ci confidence must be in (0, 1), got {confidence}"
        )

    rng = np.random.default_rng(seed)
    n = len(values)
    if n == 0:
        return (0.0, 0.0, 0.0)

    means = np.empty(n_resamples)
    for i in range(n_resamples):
        sample = rng.choice(values, size=n, replace=True)
        means[i] = np.mean(sample)

    alpha = 1.0 - confidence
    ci_lower = float(np.percentile(means, 100 * alpha / 2))
    ci_upper = float(np.percentile(means, 100 * (1 - alpha / 2)))
    mean = float(np.mean(values))

    return (mean, ci_lower, ci_upper)


# ============================================================================
# MODEL IMPLEMENTATIONS
# ============================================================================


def _get_sbert_model():
    """Lazy-load SBERT model (all-MiniLM-L6-v2). Thread-safe (Iter1 IN-03).

    Uses double-checked locking: the fast-path None check stays
    lock-free once the cache is warm; the slow path acquires
    _SBERT_LOCK and re-checks before constructing.
    """
    global _SBERT_MODEL
    if _SBERT_MODEL is None:
        with _SBERT_LOCK:
            if _SBERT_MODEL is None:
                from sentence_transformers import SentenceTransformer
                print(
                    "[CSLI] Loading SBERT model (all-MiniLM-L6-v2)...",
                    flush=True,
                )
                _SBERT_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _SBERT_MODEL


def _get_t5_device():
    """Pick the torch device for T5 scoring.

    Honors ``CSLI_T5_DEVICE`` (e.g. ``cuda:0``, ``cpu``) when set,
    otherwise auto-selects CUDA when available. The Modal stage for
    full CSLI requests an A100 — without this, the T5 model and its
    tokenized inputs would stay on CPU even when CUDA is present, and
    the full split's T5 forwards would time out the GPU lane (Codex
    PR-14 follow-up review 3308098595).
    """
    import os
    import torch

    explicit = os.environ.get("CSLI_T5_DEVICE")
    if explicit:
        return torch.device(explicit)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _get_t5_model():
    """Lazy-load T5-small model and tokenizer. Thread-safe (Iter1 IN-03).

    Loads _T5_MODEL and _T5_TOKENIZER atomically together under a
    single _T5_LOCK -- a partial state with model set but tokenizer
    not yet assigned would crash a concurrent reader.

    PR #14 follow-up review (Codex 3308098595): the model is moved
    onto ``_get_t5_device()`` (CUDA when available; CPU otherwise) so
    the Modal A100 lane is actually used for T5 scoring on the full
    split. Per-call input/target tensors are moved to the same device
    by the scorer functions.
    """
    global _T5_MODEL, _T5_TOKENIZER
    if _T5_MODEL is None:
        with _T5_LOCK:
            if _T5_MODEL is None:
                from transformers import T5ForConditionalGeneration, T5Tokenizer
                device = _get_t5_device()
                print(
                    f"[CSLI] Loading T5-small model on device={device}...",
                    flush=True,
                )
                tokenizer = T5Tokenizer.from_pretrained("t5-small")
                model = T5ForConditionalGeneration.from_pretrained("t5-small")
                model.to(device)
                model.eval()
                # Assign tokenizer first, then model. The
                # ``_T5_MODEL is None`` check above is the gate, so
                # model must be the last write -- otherwise a reader
                # could observe model set + tokenizer still None.
                _T5_TOKENIZER = tokenizer
                _T5_MODEL = model
    return _T5_MODEL, _T5_TOKENIZER


# ============================================================================
# ANSWER-PRIOR LOADING (Iter1 IN-01 -- folklore-heuristic replacement)
# ============================================================================


def _set_answer_prior_train_path(path: Path) -> None:
    """Inject the train split path for the answer-prior loaders.

    Called from ``main()`` so the prior is loaded from the same data
    partition that the rest of the script is operating against
    (respects the ``--data-dir`` override). Invalidates any cached
    prior so a subsequent in-process run with a different path
    rebuilds the prior cleanly.

    Thread-safe (Iter1 IN-03): acquires both prior locks so the
    invalidation is atomic with respect to concurrent readers. The
    locks are acquired in a consistent order (TFIDF then SBERT) to
    match the no-cross-lock convention enforced elsewhere in this
    module (the only locks that may be held simultaneously are
    prior_lock followed by the corresponding model_lock).

    Parameters
    ----------
    path : Path
        Path to ``train_dataset.json`` (wrapped or plain-list form;
        ``_load_answer_prior_texts`` routes through the shared
        ``iter_split_questions`` helper).
    """
    global _ANSWER_PRIOR_TRAIN_PATH, _TFIDF_ANSWER_PRIOR, _SBERT_ANSWER_PRIOR
    with _TFIDF_PRIOR_LOCK, _SBERT_PRIOR_LOCK:
        _ANSWER_PRIOR_TRAIN_PATH = path
        _TFIDF_ANSWER_PRIOR = None
        _SBERT_ANSWER_PRIOR = None


def _load_answer_prior_texts() -> list[str]:
    """Load the answer-prior corpus from the train split.

    Reads ``train_dataset.json`` (whichever shape is on disk -- both
    are accepted via the shared ``iter_split_questions`` helper) and
    returns the list of ``answer_primary`` strings, filtered to
    non-empty values. This corpus serves as the empirical "what does
    an answer look like in this domain" prior the IN-01 replacement
    heuristic scores options against.

    Returns
    -------
    list[str]
        Non-empty ``answer_primary`` strings from the train split.

    Raises
    ------
    RuntimeError
        If ``_set_answer_prior_train_path`` has not been called
        (i.e., the scorer was invoked from a context that did not set
        up the prior path); if the train split file does not exist;
        if no usable answer_primary strings survive filtering.
    """
    if _ANSWER_PRIOR_TRAIN_PATH is None:
        raise RuntimeError(
            "Iter1 IN-01: answer-prior path not initialized. Call "
            "_set_answer_prior_train_path(<train_dataset.json>) from "
            "main() before scoring."
        )
    if not _ANSWER_PRIOR_TRAIN_PATH.exists():
        raise RuntimeError(
            f"Iter1 IN-01: answer-prior train split not found: "
            f"{_ANSWER_PRIOR_TRAIN_PATH}. Run scripts/fresh_split.py "
            "to produce train_dataset.json (the choices-only TF-IDF "
            "and SBERT scorers now score options against this "
            "training-split answer prior instead of relying on the "
            "prior folklore heuristic; see docstrings on "
            "_score_tfidf_choices_only and _score_sbert_choices_only)."
        )

    from scripts._common import iter_split_questions

    with open(_ANSWER_PRIOR_TRAIN_PATH, "r", encoding="utf-8") as f:
        payload = json.load(f)
    questions = iter_split_questions(
        payload, source_path=_ANSWER_PRIOR_TRAIN_PATH
    )
    texts = [
        str(q["answer_primary"]).strip()
        for q in questions
        if isinstance(q.get("answer_primary"), str) and q["answer_primary"].strip()
    ]
    if not texts:
        raise RuntimeError(
            f"Iter1 IN-01: train split at {_ANSWER_PRIOR_TRAIN_PATH} "
            "contains no usable answer_primary strings. The answer "
            "prior cannot be built; CSLI choices-only scoring would "
            "be ill-defined."
        )
    return texts


def _get_tfidf_answer_prior():
    """Lazy-load the TF-IDF answer-prior (vectorizer + centroid vector).

    Thread-safe (Iter1 IN-03): double-checked locking on
    _TFIDF_PRIOR_LOCK. The fitted vectorizer and centroid are
    assigned together as a single tuple, so a reader either sees
    None or sees a fully-formed (vectorizer, centroid) pair.

    Builds a TF-IDF vectorizer over the training-split answer corpus
    and returns ``(vectorizer, centroid_vector)``. The centroid is the
    mean of the TF-IDF representations of the prior corpus and serves
    as the per-corpus "typical answer" representation that options are
    scored against in the IN-01 replacement heuristic.

    Returns
    -------
    tuple[TfidfVectorizer, np.ndarray]
        Fitted vectorizer (so options can be transformed in the same
        TF-IDF space) and a dense centroid vector of shape (1, n_features).
    """
    global _TFIDF_ANSWER_PRIOR
    if _TFIDF_ANSWER_PRIOR is None:
        with _TFIDF_PRIOR_LOCK:
            if _TFIDF_ANSWER_PRIOR is None:
                from sklearn.feature_extraction.text import TfidfVectorizer

                texts = _load_answer_prior_texts()
                print(
                    f"[CSLI] Building TF-IDF answer prior from "
                    f"{len(texts)} train-split answers (Iter1 IN-01)...",
                    flush=True,
                )
                vectorizer = TfidfVectorizer()
                prior_matrix = vectorizer.fit_transform(texts)
                # Centroid is the mean TF-IDF vector. np.asarray is needed
                # because csr_matrix.mean(axis=0) returns a np.matrix subclass
                # which is deprecated and produces warnings; np.asarray casts
                # to a plain ndarray of shape (1, n_features) -- exactly what
                # cosine_similarity expects as a row vector.
                centroid = np.asarray(prior_matrix.mean(axis=0))
                _TFIDF_ANSWER_PRIOR = (vectorizer, centroid)
    return _TFIDF_ANSWER_PRIOR


def _get_sbert_answer_prior() -> np.ndarray:
    """Lazy-load the SBERT answer-prior (centroid embedding).

    Thread-safe (Iter1 IN-03): double-checked locking on
    _SBERT_PRIOR_LOCK. Internally calls _get_sbert_model() (which
    acquires _SBERT_LOCK); the lock-nesting order is always
    prior_lock -> model_lock and never reversed, so no deadlock.

    Encodes the training-split answer corpus with SBERT and returns
    the mean embedding as a 2-D ndarray of shape (1, embed_dim). This
    is the per-corpus "typical answer" embedding that options are
    scored against in the IN-01 replacement heuristic.

    Returns
    -------
    np.ndarray
        Centroid embedding of shape (1, embed_dim), suitable for
        passing as the first argument of
        ``sklearn.metrics.pairwise.cosine_similarity``.
    """
    global _SBERT_ANSWER_PRIOR
    if _SBERT_ANSWER_PRIOR is None:
        with _SBERT_PRIOR_LOCK:
            if _SBERT_ANSWER_PRIOR is None:
                texts = _load_answer_prior_texts()
                model = _get_sbert_model()
                print(
                    f"[CSLI] Building SBERT answer prior from "
                    f"{len(texts)} train-split answers (Iter1 IN-01)...",
                    flush=True,
                )
                embeddings = model.encode(
                    texts, convert_to_numpy=True, show_progress_bar=False
                )
                centroid = embeddings.mean(axis=0, keepdims=True)
                _SBERT_ANSWER_PRIOR = centroid
    return _SBERT_ANSWER_PRIOR


def _score_tfidf_choices_only(options: list[str]) -> int:
    """TF-IDF choices-only: select option most similar to the answer prior.

    Iter1 IN-01 REPLACEMENT (NOT additive). The old heuristic picked the
    option with the LOWEST mean similarity to the OTHER options, on the
    folklore assumption "most dissimilar = most specific = correct".
    That assumption inverse-picks on well-constructed MC datasets,
    because the WHOLE POINT of a good distractor is to be similar to
    the correct answer (similar surface form, similar specificity,
    similar domain). On distractors crafted that way, "most dissimilar"
    systematically picks the WORST distractor (the one that least
    resembles a real answer for this question's domain) rather than the
    gold answer.

    The replacement scores each option's TF-IDF cosine similarity to a
    "generic answer prior" assembled from all ``answer_primary`` strings
    in the training split. The prior is a centroid TF-IDF vector
    representing "what answers look like in this corpus" (proper nouns,
    named works, year/place strings, etc.). The option with the
    HIGHEST similarity to that prior is predicted as the answer.

    Defensible-choice caveat: this is ONE defensible choice among many
    for the choices-only condition. Alternatives that the manuscript
    Limitations section should mention: (a) cosine to nearest prior
    neighbor instead of centroid, (b) length-only baseline, (c) GPT-2
    perplexity prior, (d) keep the old heuristic as a contrast.
    The centroid TF-IDF choice is principled (prior over an explicit
    training corpus that is observed before any test inspection) and
    avoids the inverse-pick failure mode of the old heuristic.

    The prior is cached lazily and rebuilt only if
    ``_set_answer_prior_train_path`` is re-called with a new path.

    Parameters
    ----------
    options : list[str]
        The K=4 answer options.

    Returns
    -------
    int
        Predicted answer index (0..K-1).
    """
    from sklearn.metrics.pairwise import cosine_similarity

    vectorizer, prior_centroid = _get_tfidf_answer_prior()
    try:
        option_matrix = vectorizer.transform(options)
    except ValueError:
        # All options empty -- fall back to 0 (matches the old behavior
        # for this degenerate edge case, which is rare in practice).
        return 0

    # cosine_similarity returns shape (n_options, 1); flatten to 1-D
    sims = cosine_similarity(option_matrix, prior_centroid).reshape(-1)
    return int(np.argmax(sims))


def _score_tfidf_full(question: str, options: list[str]) -> int:
    """TF-IDF full: select option most similar to question text.

    Parameters
    ----------
    question : str
        The full question text.
    options : list[str]
        The K=4 answer options.

    Returns
    -------
    int
        Predicted answer index (0-3).
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    documents = [question] + options
    vectorizer = TfidfVectorizer()
    try:
        tfidf_matrix = vectorizer.fit_transform(documents)
    except ValueError:
        return 0

    # Similarity between question (index 0) and each option (indices 1..K)
    question_vec = tfidf_matrix[0:1]
    option_vecs = tfidf_matrix[1:]
    sims = cosine_similarity(question_vec, option_vecs)[0]

    return int(np.argmax(sims))


def _score_sbert_choices_only(options: list[str]) -> int:
    """SBERT choices-only: select option most similar to the answer prior.

    Iter1 IN-01 REPLACEMENT (NOT additive). See the matching docstring
    on ``_score_tfidf_choices_only`` for the full rationale: the old
    heuristic ("most dissimilar from other options = most specific =
    correct") inverse-picks on well-constructed distractors because
    distractors are intentionally crafted to be similar to the correct
    answer.

    The replacement scores each option's SBERT cosine similarity to a
    "generic answer prior" -- the mean SBERT embedding of all
    ``answer_primary`` strings in the training split. The option with
    the HIGHEST cosine similarity to that centroid is predicted as the
    answer.

    Same defensible-choice caveat as the TF-IDF version: this is one
    defensible choice among many. The centroid-based prior is
    principled (training-split only, observed before any test
    inspection) and replaces an indefensible folklore heuristic.

    Parameters
    ----------
    options : list[str]
        The K=4 answer options.

    Returns
    -------
    int
        Predicted answer index (0..K-1).
    """
    from sklearn.metrics.pairwise import cosine_similarity

    prior_centroid = _get_sbert_answer_prior()
    model = _get_sbert_model()
    embeddings = model.encode(options, convert_to_numpy=True, show_progress_bar=False)

    # cosine_similarity returns shape (n_options, 1); flatten to 1-D
    sims = cosine_similarity(embeddings, prior_centroid).reshape(-1)
    return int(np.argmax(sims))


def _score_sbert_full(question: str, options: list[str]) -> int:
    """SBERT full: select option most similar to question embedding.

    Parameters
    ----------
    question : str
        The full question text.
    options : list[str]
        The K=4 answer options.

    Returns
    -------
    int
        Predicted answer index (0-3).
    """
    from sklearn.metrics.pairwise import cosine_similarity

    model = _get_sbert_model()
    all_texts = [question] + options
    embeddings = model.encode(all_texts, convert_to_numpy=True)

    question_emb = embeddings[0:1]
    option_embs = embeddings[1:]
    sims = cosine_similarity(question_emb, option_embs)[0]

    return int(np.argmax(sims))


def _score_t5_choices_only(options: list[str]) -> int:
    """T5-small choices-only: pick the option with highest generation likelihood.

    Standard MC scoring with T5: input is a neutral prompt (no question
    text and no other options visible in the input), target is just the
    option text. Lower mean per-token cross-entropy = higher likelihood
    = predicted answer.

    WR-06 rewrite: the prior implementation put ALL options into the
    input AND repeated the option with a fixed "A: "/"B: " prefix in
    the target. That collapsed the discriminative signal -- the model
    was being asked to score how well it could reproduce a substring
    already present in its input, and the noise-free label prefix
    diluted the option-specific loss proportional to (prefix tokens /
    total target tokens). Net effect: T5's choices-only accuracy sat
    near chance not because the options were leakage-free, but because
    the scorer was too noisy to detect leakage, dampening T5's panel
    contribution.

    Parameters
    ----------
    options : list[str]
        The K=4 answer options.

    Returns
    -------
    int
        Predicted answer index (0-3).
    """
    import torch

    model, tokenizer = _get_t5_model()
    device = next(model.parameters()).device
    input_ids = tokenizer(
        "Which of the following is most likely correct?",
        return_tensors="pt",
        max_length=64,
        truncation=True,
    ).input_ids.to(device)

    losses = []
    with torch.no_grad():
        for opt in options:
            target_ids = tokenizer(
                opt, return_tensors="pt", max_length=128, truncation=True
            ).input_ids.to(device)
            outputs = model(input_ids=input_ids, labels=target_ids)
            losses.append(outputs.loss.item())

    return int(np.argmin(losses))


def _score_t5_full(question: str, options: list[str]) -> int:
    """T5-small full: pick the option with highest generation likelihood given question.

    Standard MC scoring with T5: input is "question: <question>" (no
    option labels in the input), target is just the option text. Lower
    mean per-token cross-entropy = higher likelihood = predicted answer.

    WR-06 rewrite: see _score_t5_choices_only for the same rationale.

    Parameters
    ----------
    question : str
        The full question text.
    options : list[str]
        The K=4 answer options.

    Returns
    -------
    int
        Predicted answer index (0-3).
    """
    import torch

    model, tokenizer = _get_t5_model()
    device = next(model.parameters()).device
    input_ids = tokenizer(
        f"question: {question}",
        return_tensors="pt",
        max_length=512,
        truncation=True,
    ).input_ids.to(device)

    losses = []
    with torch.no_grad():
        for opt in options:
            target_ids = tokenizer(
                opt, return_tensors="pt", max_length=128, truncation=True
            ).input_ids.to(device)
            outputs = model(input_ids=input_ids, labels=target_ids)
            losses.append(outputs.loss.item())

    return int(np.argmin(losses))


# ============================================================================
# EXPERIMENT RUNNERS
# ============================================================================


def run_choices_only_prompt(
    questions: list[dict],
    model_name: str,
) -> tuple[float, np.ndarray]:
    """Run choices-only experiment for a given model.

    Presents ONLY the answer choices (no question text) to the model and
    determines which option it selects. Returns accuracy and per-question
    correctness array.

    Parameters
    ----------
    questions : list[dict]
        List of MC question dicts with 'options' and 'gold_index' fields.
    model_name : str
        Model identifier: 'tfidf', 'sbert', or 't5-small'.

    Returns
    -------
    tuple[float, np.ndarray]
        (accuracy, per_question_correct) where per_question_correct is a
        binary array of shape (n_questions,).
    """
    n = len(questions)
    correct = np.zeros(n, dtype=np.int32)

    for i, q in enumerate(questions):
        options = q["options"]
        gold_idx = q["gold_index"]

        if model_name == "tfidf":
            pred = _score_tfidf_choices_only(options)
        elif model_name == "sbert":
            pred = _score_sbert_choices_only(options)
        elif model_name == "t5-small":
            pred = _score_t5_choices_only(options)
        else:
            raise ValueError(f"Unknown model: {model_name}")

        correct[i] = int(pred == gold_idx)

        if (i + 1) % 500 == 0:
            print(f"[CSLI] {model_name} choices-only: {i+1}/{n}", flush=True)

    accuracy = float(np.mean(correct))
    return (accuracy, correct)


def run_full_prompt(
    questions: list[dict],
    model_name: str,
) -> tuple[float, np.ndarray]:
    """Run full-question experiment (question text + choices) for a given model.

    Presents the full question text along with answer choices. Returns accuracy
    and per-question correctness array.

    Parameters
    ----------
    questions : list[dict]
        List of MC question dicts with 'question', 'options', 'gold_index'.
    model_name : str
        Model identifier: 'tfidf', 'sbert', or 't5-small'.

    Returns
    -------
    tuple[float, np.ndarray]
        (accuracy, per_question_correct) where per_question_correct is a
        binary array of shape (n_questions,).
    """
    n = len(questions)
    correct = np.zeros(n, dtype=np.int32)

    for i, q in enumerate(questions):
        question_text = q["question"]
        options = q["options"]
        gold_idx = q["gold_index"]

        if model_name == "tfidf":
            pred = _score_tfidf_full(question_text, options)
        elif model_name == "sbert":
            pred = _score_sbert_full(question_text, options)
        elif model_name == "t5-small":
            pred = _score_t5_full(question_text, options)
        else:
            raise ValueError(f"Unknown model: {model_name}")

        correct[i] = int(pred == gold_idx)

        if (i + 1) % 500 == 0:
            print(f"[CSLI] {model_name} full: {i+1}/{n}", flush=True)

    accuracy = float(np.mean(correct))
    return (accuracy, correct)


def compute_panel_csli(
    results: dict[str, dict],
    per_question_gap: np.ndarray,
    per_model_choices_correct: dict[str, np.ndarray],
    n_questions: int,
    model_list: list[str],
    metadata_context: dict[str, Any] | None = None,
) -> dict:
    """Compute panel-level CSLI from per-model results with bootstrap CI.

    PR #14 follow-up review (Blocker 3): per the frozen reviewer
    request, the canonical CSLI is the PAP-original choices-only excess
    over chance and is published under ``panel_csli`` with a percentile
    bootstrap CI. The full-minus-choices gap (formerly named CSLI in the
    in-flight manuscript) is published under ``panel_question_use_gap``
    so a reader can still see it. The frozen gate (max acc_choices_only
    > 0.30 = 1/K + 0.05) is independent of either flavor.

    The bootstrap resamples questions (with replacement) and on each
    resample recomputes per-model ``acc_choices_only`` and the PAP
    per-model excess ``max(0, acc - 1/K)``, then averages those
    per-model excesses. This is the PAP estimator's own sampling
    distribution under question-resampling.

    Parameters
    ----------
    results : dict[str, dict]
        Mapping from model_name -> per-model record with ``acc_full``,
        ``acc_choices_only``, ``csli`` (choices-excess), and
        ``question_use_gap``.
    per_question_gap : np.ndarray
        Per-question full-minus-choices-correctness indicator,
        averaged across models. Used for bootstrap CI of
        ``panel_question_use_gap``.
    per_model_choices_correct : dict[str, np.ndarray]
        Per-model per-question 0/1 choices-only-correctness arrays.
        Used to recompute the PAP estimator on each bootstrap resample.
    n_questions : int
        Number of questions evaluated.
    model_list : list[str]
        Ordered list of model names.

    Returns
    -------
    dict
        Panel CSLI summary with per-model and aggregate statistics.
    """
    K = 4
    chance = 1.0 / K

    # Canonical CSLI = PAP per-model excess average. Point estimate is
    # exact; bootstrap CI is computed by resampling questions and
    # recomputing per-model accuracies on each sample.
    csli_excess_per_model = [
        max(0.0, results[m]["acc_choices_only"] - chance) for m in model_list
    ]
    panel_csli_excess_from_models = float(np.mean(csli_excess_per_model))

    per_model_choices_arrays = [
        per_model_choices_correct[m].astype(float) for m in model_list
    ]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n_boot = 1000
    boot_panel_excess = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n_questions, n_questions)
        excesses = []
        for arr in per_model_choices_arrays:
            acc_m = float(arr[idx].mean())
            excesses.append(max(0.0, acc_m - chance))
        boot_panel_excess[i] = float(np.mean(excesses))
    ci_lower = float(np.percentile(boot_panel_excess, 2.5))
    ci_upper = float(np.percentile(boot_panel_excess, 97.5))
    # Headline CSLI value is the PAP point estimate (not the bootstrap
    # mean, which would be biased by half-rectification asymmetry).
    mean_ci = panel_csli_excess_from_models

    # question_use_gap = acc_full - acc_choices_only (in-flight manuscript
    # CSLI). Kept for transparency. Per-question gap is the population
    # the CI samples from; per-model average is the diagnostic cross-check.
    gap_values = [results[m]["question_use_gap"] for m in model_list]
    gap_mean_from_models = float(np.mean(gap_values))
    gap_mean_ci, gap_ci_lower, gap_ci_upper = bootstrap_ci(
        per_question_gap, n_resamples=1000, seed=BOOTSTRAP_SEED
    )

    from scripts.threshold_manifest import (
        load_frozen_threshold_manifest,
        threshold_value,
    )

    manifest = load_frozen_threshold_manifest(THRESHOLD_MANIFEST, strict=True)
    # threshold_value raises RuntimeError if the metric or field is
    # missing -- WR-02 fix: no silent fallback to a hardcoded default,
    # which would have decoupled the leakage flag from the frozen
    # manifest provenance.
    threshold = float(
        threshold_value(manifest, "choices_only_accuracy", key="numeric_value_K4")
    )
    for m in model_list:
        results[m]["leakage_flag"] = results[m]["acc_choices_only"] > threshold

    metadata = {
        "n_questions": n_questions,
        "n_models": len(model_list),
        "K": K,
        "chance": chance,
        "choices_only_accuracy_threshold": threshold,
        "threshold_metric": "choices_only_accuracy",
        "threshold_criterion": "acc_choices_only > choices_only_accuracy_threshold",
        "threshold_direction": "warn_if_above",
        "threshold_source": _display_path(THRESHOLD_MANIFEST),
        "threshold": threshold,
        "threshold_deprecated_note": (
            "Deprecated alias for choices_only_accuracy_threshold; applies to "
            "per-model acc_choices_only, not to panel_csli.mean or per-model csli."
        ),
        "bootstrap_resamples": 1000,
        "bootstrap_method": "percentile",
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_note": (
            "Percentile-method bootstrap (not BCa). CI captures "
            "sampling variability over questions but not model "
            "selection uncertainty."
        ),
        "csli_definition": (
            "Canonical CSLI = max(0, acc_choices_only - 1/K) (PAP-original "
            "choices-only excess over chance). question_use_gap = acc_full - "
            "acc_choices_only (in-flight manuscript's pre-rename CSLI). The "
            "frozen leakage gate is on per-model acc_choices_only > "
            "1/K + 0.05 = 0.30 for K=4, independent of either flavor."
        ),
        "models": model_list,
    }
    if metadata_context is not None:
        metadata.update(metadata_context)

    return {
        # Canonical CSLI: choices-only excess over chance, with bootstrap CI.
        "panel_csli": {
            "mean": mean_ci,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "mean_from_per_model_avg": panel_csli_excess_from_models,
            "definition": "max(0, acc_choices_only - 1/K) per model, averaged",
            "K": K,
            "chance": chance,
        },
        # Renamed from panel_csli: full-minus-choices gap (former
        # in-flight-manuscript CSLI). Kept for transparency.
        "panel_question_use_gap": {
            "mean": gap_mean_ci,
            "ci_lower": gap_ci_lower,
            "ci_upper": gap_ci_upper,
            "mean_from_per_model_avg": gap_mean_from_models,
            "definition": "acc_full - acc_choices_only per model, averaged",
        },
        "per_model": results,
        "metadata": metadata,
    }


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments for CSLI computation.

    Parameters
    ----------
    argv : Optional[list[str]]
        Argument list (defaults to sys.argv[1:]).

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Compute CSLI via local model panel (TF-IDF, SBERT, T5-small)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(DEFAULT_DATA_DIR),
        help="Directory containing processed MC datasets (default: data/processed)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help="Output path for CSLI results JSON (default: paper_exports/csli.json)",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=DEFAULT_MODELS,
        help=(
            "Comma-separated list of model names for the panel. "
            "Available: tfidf, sbert, t5-small (default: tfidf,sbert,t5-small)"
        ),
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Quick mode: 1 model, 10 questions (for pipeline testing)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse arguments and validate data path, but do not run compute",
    )
    parser.add_argument(
        "--min-mc-coverage",
        type=float,
        default=0.98,
        help=(
            "Minimum fraction of test-split qids that must have a matching "
            "MCQuestion in mc_dataset.json (default: 0.98). When coverage is "
            "below this fraction the script refuses to run, because CSLI would "
            "be computed on a non-random subset of the test split (WR-04). "
            "Re-run scripts/build_mc_dataset.py against the new train split, "
            "or pass --allow-incomplete-mc-coverage to override."
        ),
    )
    parser.add_argument(
        "--min-mc-retention",
        type=float,
        default=None,
        help=(
            "Minimum raw-test retention rate required by build_metadata.json. "
            "Defaults to build_metadata.retention_thresholds.full (or .smoke "
            "in --smoke mode) when present, otherwise 0.98. This gate is "
            "separate from --min-mc-coverage, which only checks qid coverage "
            "against the on-disk test_dataset.json."
        ),
    )
    parser.add_argument(
        "--allow-incomplete-mc-coverage",
        action="store_true",
        help=(
            "Override the --min-mc-coverage gate. Use only when downstream "
            "interpretation accounts for the non-random subset (e.g., when "
            "MC reconstruction against the new train pool is intentionally "
            "deferred)."
        ),
    )
    parser.add_argument(
        "--allow-low-mc-retention",
        action="store_true",
        help=(
            "Override the build_metadata.json raw-test retention gate. Use "
            "only when reporting CSLI explicitly as a retained-MC-test-subset "
            "metric rather than a full raw fresh-test metric."
        ),
    )

    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    """Main entry point for CSLI computation.

    Parameters
    ----------
    argv : Optional[list[str]]
        Command-line arguments (defaults to sys.argv[1:]).

    Returns
    -------
    int
        Exit code (0 = success, 1 = error).
    """
    effective_argv = list(sys.argv[1:] if argv is None else argv)
    args = _parse_args(effective_argv)

    data_dir = Path(args.data_dir)
    output_path = Path(args.output)

    # Iter1 IN-06: validate --models is non-empty BEFORE smoke
    # truncation. `args.models.split(",")` produces `[""]` for an
    # empty string and `[]` for a comma-only string; both filter to
    # the empty list once whitespace-stripped. Previously, an empty
    # model_list survived the `if args.smoke: model_list[:1]` slice
    # as `[]`, the loop over `model_list` ran zero times, and then
    # `per_question_csli /= len(model_list)` raised
    # `ZeroDivisionError` on a numpy array -- failing far enough
    # downstream that the operator could not tell `--models ""` was
    # the typo. Per the script docstring exit codes, argument errors
    # use exit 2.
    model_list = [m.strip() for m in args.models.split(",") if m.strip()]
    if not model_list:
        print(
            "ERROR: --models must specify at least one model "
            "(comma-separated). Available: tfidf, sbert, t5-small.",
            file=sys.stderr,
        )
        return 2
    unknown_models = sorted(set(model_list) - SUPPORTED_MODELS)
    if unknown_models:
        print(
            f"ERROR: unsupported --models entries: {', '.join(unknown_models)}. "
            f"Available: {', '.join(sorted(SUPPORTED_MODELS))}.",
            file=sys.stderr,
        )
        return 2

    if args.smoke:
        model_list = model_list[:1]
        print(f"[CSLI] Smoke mode: 1 model ({model_list[0]}), 10 questions")

    # Validate data directory exists
    if not data_dir.exists():
        print(f"ERROR: Data directory not found: {data_dir}", file=sys.stderr)
        return 1

    if args.dry_run:
        print(f"[CSLI] Dry run -- data_dir={data_dir}, models={model_list}")
        print(f"[CSLI] Output would be written to: {output_path}")
        return 0

    # Iter1 IN-01: register the train split path for the answer-prior
    # loaders. The TF-IDF and SBERT choices-only scorers now score
    # options against a centroid of training-split answer_primary
    # texts (replacing the prior folklore "most dissimilar = most
    # specific" heuristic). The path follows --data-dir so the prior
    # is loaded from the same partition as the test split being scored.
    _set_answer_prior_train_path(data_dir / "train_dataset.json")

    # Load MC dataset (has options and gold_index)
    mc_path = data_dir / "mc_dataset.json"
    if not mc_path.exists():
        print(f"ERROR: MC dataset not found: {mc_path}", file=sys.stderr)
        return 1

    with open(mc_path, "r") as f:
        mc_questions = json.load(f)

    # WR-07: warn loudly if a pre-freshsplit archive of data/processed/
    # exists. Its presence means fresh_split.py ran AFTER
    # build_mc_dataset.py and the MC distractors here were assembled
    # against the pre-freshsplit train pool. The warning is informational
    # -- it does not gate (the WR-04 coverage check is the gate). Tells
    # the operator the same thing the docstring's LIMITATION section
    # documents, but loud, at runtime, where it cannot be missed.
    archive_dir = data_dir.parent  # typically PROJECT_ROOT/data
    pre_freshsplit_archives = sorted(
        archive_dir.glob("processed.pre_v10_freshsplit_*")
    ) if archive_dir.exists() else []
    if pre_freshsplit_archives:
        print(
            f"[CSLI] WR-07 PROVENANCE WARNING: pre-freshsplit data "
            f"archive(s) detected: "
            f"{', '.join(p.name for p in pre_freshsplit_archives)}.",
            flush=True,
        )
        print(
            "[CSLI]   mc_dataset.json distractors were assembled against "
            "the OLD train pool. The fresh-split integrity claim does "
            "NOT extend to MC construction.",
            flush=True,
        )
        print(
            "[CSLI]   Preferred remediation: re-run "
            "scripts/build_mc_dataset.py AFTER scripts/fresh_split.py "
            "(this is now safe -- WR-05 made compute_csli.py accept "
            "both test_dataset.json shapes).",
            flush=True,
        )

    # Load test split (for qid filtering)
    test_path = data_dir / "test_dataset.json"
    if not test_path.exists():
        print(f"ERROR: Test dataset not found: {test_path}", file=sys.stderr)
        return 1

    with open(test_path, "r") as f:
        test_data = json.load(f)

    # WR-05 / Iter2 IN-01: accept both on-disk shapes for
    # test_dataset.json via the shared helper. See
    # scripts/_common.iter_split_questions for the rationale; this
    # call site was the original WR-05 fix and now uses the same
    # helper that compute_stopdff.py and compute_prefix_calibration.py
    # use, closing the cross-consumer gap.
    from scripts._common import iter_split_questions

    try:
        test_questions_iter = iter_split_questions(
            test_data, source_path=test_path
        )
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    # Extract test split qids
    test_qids = set(str(q["qid"]) for q in test_questions_iter)

    # Filter MC questions to only those in test split. The coverage gate uses
    # unique qids, not row count, so duplicate MC rows cannot mask missing qids.
    questions, mc_coverage = _filter_test_mc_questions(mc_questions, test_qids)
    matched_test_mc_questions = mc_coverage["matched_questions"]
    matched_test_mc_qids = mc_coverage["matched_qids"]
    missing_qids = mc_coverage["missing_qids_set"]
    coverage = mc_coverage["coverage_rate"]
    print(f"[CSLI] Loaded {len(questions)} test-split MC questions "
          f"(from {len(mc_questions)} MC total, {len(test_qids)} test qids)")

    # WR-04: refuse to compute CSLI on a non-random subset of the test
    # split. fresh_split.py partitions raw TossupQuestions, but
    # mc_dataset.json only contains questions that survived MC
    # construction in build_mc_dataset.py. The intersection can be
    # strictly smaller than len(test_qids), and the missing items are
    # NOT missing at random (they're "hard to find distractors for"),
    # which is itself a confounder of choices-only accuracy. Fail closed
    # unless the operator explicitly overrides.
    print(
        f"[CSLI] MC coverage of test split: {coverage:.1%} "
        f"({matched_test_mc_qids}/{len(test_qids)} unique qids, "
        f"{len(questions)} rows, {len(missing_qids)} missing)"
    )
    if coverage < args.min_mc_coverage and not args.allow_incomplete_mc_coverage:
        print(
            f"ERROR: WR-04 / CSLI-01 violation: only {coverage:.1%} of "
            f"new test qids have matching MCQuestions in mc_dataset.json "
            f"(threshold: {args.min_mc_coverage:.1%}). CSLI would be "
            f"computed on a non-random subset selected against 'hard to "
            f"find distractors for'. Re-run scripts/build_mc_dataset.py "
            f"against the new train split, or pass "
            f"--allow-incomplete-mc-coverage to override.",
            file=sys.stderr,
        )
        return 1

    try:
        build_metadata = _load_mc_build_metadata(data_dir)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    retention_gate: dict[str, Any] = {
        "applies": build_metadata["status"] == "loaded",
        "threshold": None,
        "passed": None,
        "overridden": False,
        "override_flag": "--allow-low-mc-retention",
    }
    if build_metadata["status"] == "loaded":
        test_retention_raw = build_metadata["test"]
        if not isinstance(test_retention_raw, dict):
            print(
                "ERROR: Invalid MC build metadata: missing splits.test object.",
                file=sys.stderr,
            )
            return 1
        test_retention = cast(dict[str, Any], test_retention_raw)
        min_retention = _metadata_retention_threshold(
            build_metadata,
            smoke=args.smoke,
            explicit_threshold=args.min_mc_retention,
        )
        retention_rate = float(test_retention["retention_rate"])
        retention_passed = retention_rate >= min_retention
        retention_overridden = (
            not retention_passed and args.allow_low_mc_retention
        )
        retention_gate.update(
            {
                "threshold": min_retention,
                "passed": retention_passed,
                "overridden": retention_overridden,
            }
        )
        print(
            "[CSLI] MC raw-test retention from build_metadata.json: "
            f"{retention_rate:.1%} "
            f"({test_retention['retained_count']}/{test_retention['raw_count']}, "
            f"{test_retention['dropped_count']} dropped; threshold "
            f"{min_retention:.1%})"
        )
        if int(test_retention["retained_count"]) != len(test_qids):
            print(
                "[CSLI] WARNING: build_metadata.json retained_count does not "
                f"match test_dataset.json qid count "
                f"({test_retention['retained_count']} vs {len(test_qids)}).",
                flush=True,
            )
        if not retention_passed and not args.allow_low_mc_retention:
            print(
                f"ERROR: CSLI raw-test MC retention is only "
                f"{retention_rate:.1%} (threshold: {min_retention:.1%}). "
                f"CSLI would be computed on the retained MC-test subset, not "
                f"the full raw fresh-test split. Pass "
                f"--allow-low-mc-retention only if the artifact/report "
                f"explicitly qualifies the result as retained-subset CSLI.",
                file=sys.stderr,
            )
            return 1
        if retention_overridden:
            print(
                "[CSLI] LOW-RETENTION OVERRIDE: proceeding with retained-subset "
                "CSLI because --allow-low-mc-retention was supplied.",
                flush=True,
            )
    else:
        print(
            "[CSLI] build_metadata.json not found; raw-test MC retention "
            "counts are unavailable. Reporting coverage against "
            "test_dataset.json only.",
            flush=True,
        )

    if args.smoke:
        questions = questions[:10]
        print(f"[CSLI] Smoke mode: trimmed to {len(questions)} questions")

    # PR #14 follow-up review (Codex 3307859491): fail closed when the
    # coverage override left zero matching MC questions. Without this
    # guard, ``run_choices_only_prompt``/``run_full_prompt`` take
    # ``np.mean`` of empty correctness arrays, producing NaN accuracies
    # and a degenerate csli.json that downstream consumers cannot
    # interpret. The override is for low coverage, not zero coverage.
    if not questions:
        print(
            "ERROR: zero MC questions remain after filtering to the test "
            "split. The --allow-incomplete-mc-coverage override is for "
            "low-but-nonzero coverage; with no matching qids, every "
            "downstream accuracy is NaN. Re-run "
            "scripts/build_mc_dataset.py so mc_dataset.json overlaps "
            "with the current test_dataset.json.",
            file=sys.stderr,
        )
        return 1

    print(f"[CSLI] Panel models: {model_list}")

    # PR #14 follow-up review (Issue A): enforce K=4 at runtime. The
    # frozen leakage gate (threshold_manifest.json) and the panel
    # choices-only-excess CSLI both assume K=4 (chance = 1/K = 0.25).
    # If any question has a different option count, the chance baseline
    # and the gate threshold are silently wrong. Fail closed so a
    # variable-K artifact cannot reach the audit card unflagged.
    K = 4
    chance = 1.0 / K
    bad_k = [
        (q.get("qid"), len(q.get("options") or []))
        for q in questions
        if len(q.get("options") or []) != K
    ]
    if bad_k:
        first_qid, first_count = bad_k[0]
        print(
            f"ERROR: CSLI assumes K={K} options per MC question, but "
            f"{len(bad_k)} questions have a different K (first: qid={first_qid}, "
            f"K={first_count}). The choices-only-excess flavor and the frozen "
            f"chance baseline (1/K = {chance}) would be inconsistent. Rebuild "
            f"the MC dataset so every retained question has exactly K options.",
            file=sys.stderr,
        )
        return 1

    # Run experiments per model
    results: dict[str, dict] = {}
    per_model_choices_correct: dict[str, np.ndarray] = {}
    per_model_full_correct: dict[str, np.ndarray] = {}

    # PR #14 follow-up review (Blocker 3): per the frozen reviewer
    # request, ``csli`` (the canonical CSLI) is now the PAP-original
    # choices-only-excess flavor and ``question_use_gap`` carries the
    # full-minus-choices gap that the in-flight manuscript used to call
    # CSLI. Both are emitted so consumers can render either. The frozen
    # gate (manifest: max acc_choices_only > 0.30 = 1/K + 0.05) is
    # independent of which name is used for display.

    for model_name in model_list:
        print(f"\n[CSLI] === Running model: {model_name} ===", flush=True)

        print(f"[CSLI] {model_name}: choices-only condition...", flush=True)
        acc_choices, choices_correct = run_choices_only_prompt(questions, model_name)

        print(f"[CSLI] {model_name}: full condition...", flush=True)
        acc_full, full_correct = run_full_prompt(questions, model_name)

        question_use_gap = acc_full - acc_choices
        csli = max(0.0, acc_choices - chance)
        results[model_name] = {
            "acc_full": round(acc_full, 6),
            "acc_choices_only": round(acc_choices, 6),
            # Canonical CSLI: PAP-original choices-only excess over chance.
            "csli": round(csli, 6),
            # Renamed from "csli" -- the in-flight manuscript's full-minus-
            # choices gap, kept for transparency but no longer the headline.
            "question_use_gap": round(question_use_gap, 6),
        }
        per_model_choices_correct[model_name] = choices_correct
        per_model_full_correct[model_name] = full_correct

        print(f"[CSLI] {model_name}: full={acc_full:.4f}, "
              f"choices_only={acc_choices:.4f}, "
              f"CSLI(choices_excess)={csli:.4f}, "
              f"question_use_gap={question_use_gap:.4f}", flush=True)

    # Compute per-question gap (acc_full - acc_choices) averaged across
    # models. Feeds panel_question_use_gap's bootstrap CI. Per-model
    # per-question choices-correct arrays are passed through to
    # compute_panel_csli for the PAP-estimator question-bootstrap.
    n_questions = len(questions)
    per_question_gap = np.zeros(n_questions)
    for m in model_list:
        per_question_gap += (
            per_model_full_correct[m].astype(float)
            - per_model_choices_correct[m].astype(float)
        )
    per_question_gap /= len(model_list)

    # Compute panel-level CSLI with bootstrap CI
    split_provenance = _load_split_provenance(data_dir)
    metadata_context: dict[str, Any] = {
        "generation": _build_generation_provenance(
            effective_argv,
            output_path=output_path,
            data_dir=data_dir,
        ),
        "mc_coverage": {
            "target_qids": len(test_qids),
            "mc_questions_total": len(mc_questions),
            "matched_questions": matched_test_mc_questions,
            "matched_qids": matched_test_mc_qids,
            "missing_qids": len(missing_qids),
            "coverage_rate": coverage,
            "threshold": args.min_mc_coverage,
            "passed": coverage >= args.min_mc_coverage,
            "overridden": coverage < args.min_mc_coverage
            and args.allow_incomplete_mc_coverage,
            "override_flag": "--allow-incomplete-mc-coverage",
            "split": "test",
        },
        "mc_build_metadata": {
            "status": build_metadata["status"],
            "source_path": build_metadata["source_path"],
            "source_sha256": build_metadata["source_sha256"],
            "test": build_metadata["test"],
        },
        "mc_retention_gate": retention_gate,
        **split_provenance,
    }

    if build_metadata["test"] is not None:
        test_retention_raw = build_metadata["test"]
        if not isinstance(test_retention_raw, dict):
            print(
                "ERROR: Invalid MC build metadata: missing splits.test object.",
                file=sys.stderr,
            )
            return 1
        test_retention = cast(dict[str, Any], test_retention_raw)
        metadata_context.update(
            {
                "raw_test_questions": test_retention["raw_count"],
                "retained_mc_test_questions": test_retention["retained_count"],
                "dropped_test_questions": test_retention["dropped_count"],
                "mc_test_retention_rate": test_retention["retention_rate"],
                "mc_test_drop_reasons": test_retention["drop_reasons"],
            }
        )

    panel = compute_panel_csli(
        results,
        per_question_gap,
        per_model_choices_correct,
        n_questions,
        model_list,
        metadata_context=metadata_context,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("[CSLI] RESULTS SUMMARY")
    print("=" * 60)
    p = panel["panel_csli"]
    g = panel["panel_question_use_gap"]
    print(
        f"Panel CSLI (choices-excess): {p['mean']:.4f} "
        f"[{p['ci_lower']:.4f}, {p['ci_upper']:.4f}]"
    )
    print(
        f"Panel question_use_gap:      {g['mean']:.4f} "
        f"[{g['ci_lower']:.4f}, {g['ci_upper']:.4f}]"
    )
    for m, v in panel["per_model"].items():
        flag = " ** LEAKAGE WARNING **" if v["leakage_flag"] else ""
        print(
            f"  {m}: full={v['acc_full']:.4f}, "
            f"choices_only={v['acc_choices_only']:.4f}, "
            f"csli={v['csli']:.4f}, "
            f"question_use_gap={v['question_use_gap']:.4f}{flag}"
        )
    print(
        "Threshold: acc_choices_only > "
        f"{panel['metadata']['choices_only_accuracy_threshold']:.2f} "
        "triggers flag"
    )
    print("=" * 60)

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(panel, f, indent=2)

    print(f"\n[CSLI] Results written to: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
