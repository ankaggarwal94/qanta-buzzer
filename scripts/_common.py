"""Shared utilities for pipeline scripts.

Provides config loading, JSON serialization, MC question deserialization,
and path constants used across all pipeline scripts (build, baseline, train,
evaluate).

Ported from qb-rl reference implementation with import path adaptations
for the unified qanta-buzzer codebase.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from models.likelihoods import LikelihoodModel, build_likelihood_from_config
from qb_data.config import load_config as load_yaml_config
from qb_data.mc_builder import MCQuestion

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_FILENAMES = {
    "combined": "mc_dataset.json",
    "train": "train_dataset.json",
    "val": "val_dataset.json",
    "test": "test_dataset.json",
}
_FILENAME_TO_SPLIT = {v: k for k, v in DATASET_FILENAMES.items()}


def split_name_from_path(path: str | Path) -> str:
    """Infer the canonical split name from a dataset filename.

    Returns the split name (``train``, ``val``, ``test``, ``combined``)
    when the filename matches a known artifact, otherwise ``"explicit"``.
    """
    return _FILENAME_TO_SPLIT.get(Path(path).name, "explicit")


def redirect_combined_to_split(
    mc_path: Path,
    preferred_split: str,
) -> tuple[Path, str, str | None]:
    """Redirect a combined dataset path to the preferred sibling split.

    When ``mc_path`` points to ``mc_dataset.json`` and a sibling split
    artifact exists, returns the split path instead to maintain the
    split-safe contract. Returns the original path unchanged for
    non-combined or non-redirectable inputs.

    Returns
    -------
    tuple[Path, str, str or None]
        (resolved_path, split_name, warning_or_None)
    """
    if split_name_from_path(mc_path) != "combined":
        return mc_path, split_name_from_path(mc_path), None
    sibling = dataset_path_for_split(mc_path.parent, preferred_split)
    if sibling.exists():
        warning = (
            f"Warning: --mc-path points to combined artifact {mc_path.name}; "
            f"redirecting to sibling {sibling.name} for split-safe {preferred_split} usage."
        )
        return sibling, preferred_split, warning
    # Sibling is missing — fall through to the combined artifact, but warn
    # loudly. Callers print this string; silent fallback was the upstream
    # of multiple split-leakage findings in the 2026-05 review.
    warning = (
        f"Warning: --mc-path points to combined artifact {mc_path.name} and "
        f"no sibling {sibling.name} was found; using combined corpus. "
        f"Split-safety is NOT guaranteed for {preferred_split} usage; "
        "build the sibling split next to mc_dataset.json for a clean run."
    )
    return mc_path, "combined", warning


def _parse_value(value: str) -> Any:
    """Parse a CLI override value string into a typed Python value.

    Tries JSON first, then bool/int/float, and falls back to str.
    """
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        pass
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    if value.lstrip("-").isdigit():
        return int(value)
    try:
        return float(value)
    except ValueError:
        return value


def parse_overrides(args: argparse.Namespace) -> dict[str, Any]:
    """Parse CLI override arguments into flat dotted-key overrides.

    Returns a dict with dotted keys (e.g. ``{"data.K": 5}``) that
    ``merge_overrides`` can apply leaf-by-leaf without clobbering
    sibling config entries.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments.  Positional ``overrides`` are
        ``key=value`` strings where *key* uses dot-notation
        (e.g. ``data.K=5``).

    Returns
    -------
    dict[str, Any]
        Flat dotted-key overrides ready for ``merge_overrides()``.
    """
    overrides: dict[str, Any] = {}
    if hasattr(args, "overrides") and args.overrides:
        for token in args.overrides:
            if "=" not in token:
                continue
            key, value = token.split("=", 1)
            overrides[key] = _parse_value(value)
    return overrides
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def project_relative(path: str | Path) -> str:
    """Return a path string relative to ``PROJECT_ROOT`` when possible.

    Used by artifact-provenance fields (e.g., ``resolved_mc_path``) so
    that committed JSONs stay machine-portable instead of leaking the
    author's absolute home directory. Falls back to the absolute path
    string when the file lives outside the repository.

    Non-absolute inputs are anchored to ``PROJECT_ROOT`` BEFORE
    resolution so that repo-relative arguments like
    ``"data/processed/mc_dataset.json"`` stay repo-relative regardless
    of the caller's CWD (common in automation that invokes scripts from
    outside the repo). Without this anchoring, ``Path(path).resolve()``
    would resolve the relative path against CWD, producing a
    machine-specific absolute path that would fail the
    ``relative_to(PROJECT_ROOT)`` check and leak through the
    provenance fallback.

    Parameters
    ----------
    path : str or Path
        Path to convert. Absolute paths are resolved as-is; relative
        paths are anchored to ``PROJECT_ROOT`` first.

    Returns
    -------
    str
        Repo-relative path (forward-slash) when the resolved path is
        inside ``PROJECT_ROOT``; otherwise the resolved absolute path.
    """
    raw = Path(path)
    if not raw.is_absolute():
        raw = PROJECT_ROOT / raw
    p = raw.resolve()
    try:
        return p.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(p)


_TEXT_HASH_SUFFIXES = frozenset(
    {
        ".cfg",
        ".csv",
        ".gitattributes",
        ".gitignore",
        ".json",
        ".md",
        ".py",
        ".rst",
        ".sh",
        ".sha256",
        ".tex",
        ".toml",
        ".tsv",
        ".txt",
        ".yaml",
        ".yml",
    }
)


def _canonical_hash_bytes(path: Path, data: bytes) -> bytes:
    """Return hash input bytes with portable text line endings.

    Audit provenance is committed on Windows and verified on Linux CI.
    Text files must therefore hash the Git blob's logical LF content, not
    whichever CRLF/LF form happens to be in the working tree. Binary
    artifacts remain byte-exact.
    """
    if path.suffix.lower() not in _TEXT_HASH_SUFFIXES:
        return data
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return data
    return text.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")


def sha256_file(path: str | Path) -> str:
    """Return a portable SHA-256 digest of a local file.

    UTF-8 text files are normalized to LF before hashing so provenance is
    stable across Windows and Linux checkouts. Non-text and non-UTF-8 files
    are hashed as exact bytes.
    """
    resolved = Path(path)
    payload = _canonical_hash_bytes(resolved, resolved.read_bytes())
    return hashlib.sha256(payload).hexdigest()


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


def build_generation_provenance(
    script_path: str | Path,
    argv: list[str],
    *,
    output_path: str | Path,
    extra_paths: list[str | Path] | None = None,
) -> dict[str, Any]:
    """Build audit metadata tying a generated artifact to its source script.

    PR #14 follow-up review (Blocker 4): every paper_exports artifact must
    carry a ``generation`` block recording the script path, sha256, argv,
    git commit, and dirty-status so the audit card can verify that the
    committed JSONs were produced by the current code at the recorded
    commit.

    Parameters
    ----------
    script_path : str or Path
        Path to the script producing the artifact (typically ``__file__``).
    argv : list[str]
        The effective ``sys.argv[1:]`` of the invocation.
    output_path : str or Path
        The artifact path being generated. Recorded as repo-relative.
    extra_paths : list of str or Path, optional
        Additional repo-relative paths whose git status to capture (e.g.,
        the threshold manifest, split provenance file). The script and
        output paths are always included automatically.

    Returns
    -------
    dict[str, Any]
        A schema-versioned generation provenance dict.
    """
    script_resolved = Path(script_path).resolve()
    # PR #14 follow-up review (Codex #3308590294): `git status -- <abs_path>`
    # aborts with `fatal: ... is outside repository` when any pathspec arg is
    # outside the repo. That happens whenever a caller passes an absolute
    # ``--output``/``--output-dir`` outside REPO_ROOT (a case the absolute-
    # output-dir round-trip from commit 41e19c4 explicitly supports). The
    # abort would silently flip `git_dirty` to False and erase
    # `git_status_relevant_paths`, defeating the provenance check exactly for
    # the case it most matters. Render every path for the `output_path`
    # display fields, but FILTER non-repo paths from the git pathspec so the
    # dirty-status check still runs against the script + threshold-manifest
    # + extras that ARE inside the repo.
    display_paths = [
        project_relative(script_resolved),
        project_relative(output_path),
    ]
    for p in extra_paths or []:
        display_paths.append(project_relative(p))
    repo_relative_pathspec = [p for p in display_paths if not Path(p).is_absolute()]

    host_status_env = os.environ.get("MODAL_HOST_GIT_STATUS")
    if host_status_env is not None:
        git_status = host_status_env
    else:
        status_args = ["status", "--short", "--", *repo_relative_pathspec]
        git_status = _git_output(status_args)

    # PR #14 follow-up validation (2026-05-27): Modal's debian_slim base
    # image lacks the `git` binary, so `git rev-parse HEAD` inside the
    # container raises FileNotFoundError and `git_commit` records as None.
    # The orchestrator injects the host's commit SHA via the
    # ``MODAL_HOST_GIT_COMMIT`` env var; prefer that when set, fall back to
    # a live `git rev-parse HEAD` query otherwise. Deterministic-build
    # best practice: provenance reflects the host commit, not whatever the
    # container's incidental git binary state was.
    host_commit_env = os.environ.get("MODAL_HOST_GIT_COMMIT")
    git_commit = host_commit_env if host_commit_env else _git_output(["rev-parse", "HEAD"])

    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": ["python", project_relative(script_resolved), *argv],
        "argv": list(argv),
        "cwd": project_relative(Path.cwd()),
        "output_path": project_relative(output_path),
        "script_path": project_relative(script_resolved),
        "script_sha256": sha256_file(script_resolved),
        "git_commit": git_commit,
        "git_dirty": bool(git_status),
        "git_status_relevant_paths": git_status or "",
    }


def load_config(config_path: str | None = None, smoke: bool = False) -> dict[str, Any]:
    """Load YAML configuration from a file path.

    Parameters
    ----------
    config_path : str or None
        Path to YAML config file. If None, loads ``configs/default.yaml``.

    Returns
    -------
    dict[str, Any]
        Parsed config dict with nested structure (data, likelihood,
        environment, ppo, etc.).
    """
    return load_yaml_config(config_path, smoke=smoke)


def build_likelihood_model(config: dict[str, Any], mc_questions: list[MCQuestion]):
    """Build a likelihood model with shared TF-IDF corpus handling."""
    corpus = None
    if config["likelihood"].get("model") == "tfidf":
        corpus = [q.question for q in mc_questions] + [
            profile
            for question in mc_questions
            for profile in question.option_profiles
        ]
    return build_likelihood_from_config(config, corpus_texts=corpus)


def collect_env_texts(questions: list[MCQuestion]) -> list[str]:
    """Collect all texts the env will ever score for ``questions``.

    Used to pre-warm a likelihood model's embedding cache via a single
    batched encoder pass before ``precompute_beliefs`` would otherwise
    issue thousands of single-text or per-K-options encoder calls. Lifted
    from a triplicate inline loop in ``train_ppo.py`` /
    ``evaluate_all.py`` / ``run_baselines.py`` to keep the per-step
    token-slice arithmetic in one place.

    Parameters
    ----------
    questions : list[MCQuestion]
        MC questions whose cumulative_prefixes, option_profiles, and
        per-step token slices will be reachable through env reward
        shaping or sequential-Bayes belief updates.

    Returns
    -------
    list[str]
        All texts the env may pass to ``likelihood_model.score(...)``
        across episodes for ``questions`` (duplicates included; the
        likelihood ``precompute_embeddings`` call dedups internally).
    """
    texts: list[str] = []
    for q in questions:
        texts.extend(q.cumulative_prefixes)
        texts.extend(q.option_profiles)
        for step_idx in range(len(q.run_indices)):
            prev_idx = q.run_indices[step_idx - 1] if step_idx > 0 else -1
            texts.append(
                " ".join(q.tokens[prev_idx + 1 : q.run_indices[step_idx] + 1])
            )
    return texts


def ensure_dir(path: str | Path) -> Path:
    """Create a directory (and parents) if it does not exist.

    Parameters
    ----------
    path : str or Path
        Directory path to create.

    Returns
    -------
    Path
        The created (or existing) directory path.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def dataset_path_for_split(base_dir: str | Path, split: str) -> Path:
    """Return the canonical dataset path for a split name.

    Parameters
    ----------
    base_dir : str or Path
        Directory containing dataset artifacts.
    split : str
        One of ``combined``, ``train``, ``val``, or ``test``.

    Returns
    -------
    Path
        Path to the split dataset JSON file.
    """
    if split not in DATASET_FILENAMES:
        raise ValueError(f"Unknown split '{split}'")
    return Path(base_dir) / DATASET_FILENAMES[split]


def resolve_persisted_split_paths(base_dir: str | Path) -> dict[str, Path] | None:
    """Return persisted train/val/test paths when all three exist."""
    base = Path(base_dir)
    paths = {
        split: dataset_path_for_split(base, split)
        for split in ("train", "val", "test")
    }
    if all(path.exists() for path in paths.values()):
        return paths
    return None


def iter_split_questions(
    split_data: Any,
    *,
    source_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Yield MC question dicts from a train/val/test split payload.

    The on-disk shape of ``{train,val,test}_dataset.json`` depends on
    which upstream producer ran last:

    - ``qb_data.dataset_splits.save_splits`` (called by
      ``scripts/fresh_split.py``) writes the wrapped form
      ``{"metadata": {...}, "questions": [...]}``.
    - ``scripts/build_mc_dataset.py`` writes the plain-list form via
      ``save_json``.

    Both shapes are valid producer outputs in this repo, so every
    downstream consumer (``compute_csli.py``, ``compute_stopdff.py``,
    ``compute_prefix_calibration.py``) must accept both. Before this
    helper existed, ``compute_csli.py`` had an inline dual-shape check
    (WR-05 fix) but the other two siblings still indexed
    ``data["questions"]`` and would crash with
    ``TypeError: list indices must be integers or slices, not str``
    on the plain-list shape -- the cross-consumer gap Phase 02 IN-01
    flagged.

    Parameters
    ----------
    split_data : Any
        Parsed JSON payload from a split dataset file. Either a
        ``dict`` with a ``"questions"`` key (wrapped form) or a
        ``list`` of MC question dicts (plain-list form).
    source_path : str or Path or None
        Path the payload was loaded from. Used only to make the
        error message more actionable when the shape is unrecognized;
        ``None`` is accepted for in-memory test payloads.

    Returns
    -------
    list[dict]
        The list of MC question dicts (with at minimum a ``"qid"``
        field). Returned as a list (not a generator) so callers can
        consume it multiple times -- ``set(...)`` immediately
        followed by ``len(...)``, for example.

    Raises
    ------
    RuntimeError
        If ``split_data`` is neither a dict-with-questions nor a
        plain list. Failing closed is preferred over silently
        coercing an unknown shape (e.g., a dict that happens to
        iterate over its keys would yield strings, not question
        dicts).
    """
    if isinstance(split_data, dict) and "questions" in split_data:
        return list(split_data["questions"])
    if isinstance(split_data, list):
        return list(split_data)
    path_str = str(source_path) if source_path is not None else "<in-memory payload>"
    raise RuntimeError(
        f"Unrecognized shape for {path_str}: expected list "
        f"or {{'questions': [...]}}; got {type(split_data).__name__}. "
        "Producer mismatch: qb_data.dataset_splits.save_splits writes "
        "the wrapped form; scripts/build_mc_dataset.py writes the "
        "plain-list form. Both are accepted, but this payload is neither."
    )


def resolve_default_dataset_path(
    out_dir: str | Path,
    preferred_split: str,
    fallback_split: str = "combined",
) -> tuple[Path, str, str | None]:
    """Resolve the default dataset path for a pipeline stage.

    Searches the stage output directory first, then ``data/processed``.
    Returns the preferred split when available, otherwise a fallback split
    plus a warning string.
    """
    candidate_dirs = [Path(out_dir), PROCESSED_DIR]
    preferred_name = DATASET_FILENAMES[preferred_split]

    for base_dir in candidate_dirs:
        preferred_path = dataset_path_for_split(base_dir, preferred_split)
        if preferred_path.exists():
            return preferred_path, preferred_split, None

    for base_dir in candidate_dirs:
        fallback_path = dataset_path_for_split(base_dir, fallback_split)
        if fallback_path.exists():
            warning = None
            if fallback_split != preferred_split:
                warning = (
                    f"Warning: {preferred_name} not found; using {fallback_path.name} "
                    f"at {fallback_path}. Results use legacy/in-sample data."
                )
            return fallback_path, fallback_split, warning

    fallback_name = DATASET_FILENAMES[fallback_split]
    search_locations = ", ".join(str(d) for d in candidate_dirs)
    raise FileNotFoundError(
        f"Could not find dataset files '{preferred_name}' or '{fallback_name}' "
        f"in any of: {search_locations}. "
        "Have you run scripts/build_mc_dataset.py to generate the MC dataset?"
    )


def to_serializable(item: Any) -> Any:
    """Recursively convert dataclasses and numpy types to JSON-serializable forms.

    Numpy scalar/array handling is deliberate: ``json.dump`` rejects
    ``np.int64``/``np.float32``/``np.ndarray`` even though ``np.float64``
    happens to inherit from ``float``. Without explicit conversion, any
    metric path that produces a non-float64 numpy value (for example
    ``np.argmax``, ``np.sum`` over int arrays, or anything pre-cast to
    ``float32``) would silently raise mid-evaluation.

    Parameters
    ----------
    item : Any
        Object to convert. Dataclasses are converted via ``asdict()``,
        numpy arrays via ``tolist()``, numpy scalars via ``.item()``,
        and dicts/lists/tuples are processed recursively.

    Returns
    -------
    Any
        JSON-serializable version of the input.
    """
    if is_dataclass(item):
        return to_serializable(asdict(item))
    if isinstance(item, np.ndarray):
        return item.tolist()
    if isinstance(item, np.generic):
        return item.item()
    if isinstance(item, dict):
        return {k: to_serializable(v) for k, v in item.items()}
    if isinstance(item, (list, tuple)):
        return [to_serializable(v) for v in item]
    return item


def save_json(path: str | Path, data: Any) -> Path:
    """Save data to a JSON file, creating parent directories as needed.

    Applies ``to_serializable`` to convert dataclasses before writing.

    Parameters
    ----------
    path : str or Path
        Output file path.
    data : Any
        Data to serialize. Dataclasses are converted to dicts automatically.

    Returns
    -------
    Path
        The path where the JSON was written.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(to_serializable(data), f, indent=2)
    return p


def load_json(path: str | Path) -> Any:
    """Load data from a JSON file.

    Parameters
    ----------
    path : str or Path
        Path to JSON file.

    Returns
    -------
    Any
        Parsed JSON data.
    """
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def load_checkpoint_sidecar(
    checkpoint_path: str | Path,
    filename: str,
) -> tuple[Any | None, Path | None, str | None]:
    """Load a JSON sidecar from a checkpoint path or its parent directory.

    Parameters
    ----------
    checkpoint_path : str or Path
        Checkpoint file or directory path.
    filename : str
        Sidecar filename to load.

    Returns
    -------
    tuple[Any or None, Path or None, str or None]
        Parsed JSON payload, matched path, and an error string when the file
        exists but could not be decoded/read.
    """
    cp = Path(checkpoint_path).resolve()
    candidates = [cp / filename] if cp.is_dir() else []
    candidates.append(cp.parent / filename)

    for sidecar in candidates:
        if not sidecar.exists():
            continue
        try:
            return load_json(sidecar), sidecar, None
        except (json.JSONDecodeError, OSError) as exc:
            return None, sidecar, str(exc)
    return None, None, None


def _restore_human_buzz_positions(
    value: Any,
) -> list[tuple[int, int]] | None:
    """Restore and strictly validate JSON-encoded ``(position, count)`` pairs."""
    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError(
            "human_buzz_positions must be a JSON list or null; "
            f"got {type(value).__name__}"
        )

    restored: list[tuple[int, int]] = []
    for index, item in enumerate(value):
        field = f"human_buzz_positions[{index}]"
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError(
                f"{field} must be a two-element [position, count] sequence"
            )
        position, count = item
        for component_index, component in enumerate((position, count)):
            if isinstance(component, bool) or not isinstance(component, int):
                raise ValueError(
                    f"{field}[{component_index}] must be an integer; "
                    f"got {type(component).__name__}"
                )
        restored.append((position, count))
    return restored


def mc_question_from_dict(row: dict[str, Any]) -> MCQuestion:
    """Reconstruct an MCQuestion dataclass from a JSON-deserialized dict.

    Parameters
    ----------
    row : dict[str, Any]
        Dictionary with all MCQuestion fields.

    Returns
    -------
    MCQuestion
        Reconstructed MCQuestion instance.
    """
    return MCQuestion(
        qid=row["qid"],
        question=row["question"],
        tokens=list(row["tokens"]),
        answer_primary=row["answer_primary"],
        clean_answers=list(row["clean_answers"]),
        run_indices=list(row["run_indices"]),
        human_buzz_positions=_restore_human_buzz_positions(
            row.get("human_buzz_positions")
        ),
        category=row.get("category", ""),
        cumulative_prefixes=list(row["cumulative_prefixes"]),
        options=list(row["options"]),
        gold_index=int(row["gold_index"]),
        option_profiles=list(row["option_profiles"]),
        option_answer_primary=list(row["option_answer_primary"]),
        distractor_strategy=row.get("distractor_strategy", "unknown"),
    )


def load_mc_questions(path: str | Path) -> list[MCQuestion]:
    """Load and deserialize a list of MCQuestions from a JSON file.

    Accepts both producer shapes via ``iter_split_questions`` (PR #14
    review Blocker 5): the plain-list shape written by
    ``scripts/build_mc_dataset.py`` AND the wrapped
    ``{"metadata": ..., "questions": [...]}`` shape written by
    ``qb_data.dataset_splits.save_splits`` (called from
    ``scripts/fresh_split.py``). The earlier Iter2 IN-01 fix wired
    the helper into the three CSLI/calibration/StopDFF consumers but
    left this convenience loader iterating ``for item in raw``,
    which silently turned wrapped payloads into iteration over the
    dict keys (``"metadata"``, ``"questions"``) and raised
    ``TypeError: string indices must be integers`` on the first
    ``mc_question_from_dict("metadata")`` call.

    NOTE: ``save_splits`` writes TOSSUP-only rows (no ``options`` or
    ``gold_index`` fields). When ``raw`` is the wrapped tossup-only
    shape, this loader will pass the rows to
    ``mc_question_from_dict`` and surface a ``KeyError: 'options'``
    with a producer-mismatch hint rather than the original
    ``TypeError``. Callers reading ``{train,val,test}_dataset.json``
    must ensure ``scripts/build_mc_dataset.py`` has run after
    ``scripts/fresh_split.py`` so the persisted rows carry the MC
    schema this loader expects.

    Parameters
    ----------
    path : str or Path
        Path to JSON file containing serialized MCQuestion dicts in
        either the plain-list or wrapped form.

    Returns
    -------
    list[MCQuestion]
        List of reconstructed MCQuestion instances.

    Raises
    ------
    RuntimeError
        If the payload shape is neither the plain-list nor the
        wrapped form (delegated to ``iter_split_questions``).
    KeyError
        If the payload is a TOSSUP-only wrapped split that lacks
        the MC schema; re-raised with a producer-mismatch hint so
        the operator knows to run ``build_mc_dataset.py``.
    """
    raw = load_json(path)
    rows = iter_split_questions(raw, source_path=path)
    try:
        return [mc_question_from_dict(item) for item in rows]
    except KeyError as exc:
        missing = str(exc)
        if missing in {"'options'", "'gold_index'", "'option_profiles'"}:
            raise KeyError(
                f"load_mc_questions({path!r}) succeeded on the outer "
                f"shape but the inner rows are missing MC field "
                f"{missing}. Likely cause: the file was last written "
                f"by qb_data.dataset_splits.save_splits (TOSSUP-only "
                f"wrapped form) without a subsequent "
                f"scripts/build_mc_dataset.py run to materialize MC "
                f"fields. Run scripts/build_mc_dataset.py to fix."
            ) from exc
        raise


# ------------------------------------------------------------------ #
# Embedding cache persistence helpers
# ------------------------------------------------------------------ #


def embedding_cache_path(config: dict[str, Any]) -> Path:
    """Return the resolved embedding cache file path from config.

    Uses ``config['likelihood']['cache_dir']`` (default ``'cache/embeddings'``)
    and appends ``'embedding_cache_{model}.npz'`` where ``{model}`` is the
    likelihood model name from config (e.g., ``tfidf``, ``t5-base``).

    Parameters
    ----------
    config : dict
        Full YAML config dict.

    Returns
    -------
    Path
        Absolute path to the embedding cache ``.npz`` file.
    """
    lik_cfg = config.get("likelihood", {})
    cache_dir = lik_cfg.get("cache_dir", "cache/embeddings")
    model_family = str(lik_cfg.get("model", "unknown"))
    if model_family == "sbert":
        variant = lik_cfg.get("sbert_name", lik_cfg.get("embedding_model", "all-MiniLM-L6-v2"))
    elif model_family == "openai":
        variant = lik_cfg.get("openai_model", "text-embedding-3-small")
    elif model_family == "t5":
        variant = lik_cfg.get("t5_name", "t5-base")
    elif model_family.startswith("t5"):
        variant = model_family
    else:
        variant = model_family
    safe_name = str(variant).replace("/", "_")
    return PROJECT_ROOT / cache_dir / f"embedding_cache_{safe_name}.npz"


def load_embedding_cache(model: LikelihoodModel, config: dict[str, Any]) -> None:
    """Load persisted embedding cache into model if file exists.

    Parameters
    ----------
    model : LikelihoodModel
        Likelihood model whose embedding_cache will be populated.
    config : dict
        Full YAML config dict (used to resolve cache path).
    """
    path = embedding_cache_path(config)
    n = model.load_cache(path)
    if n > 0:
        print(f"Loaded {n} cached embeddings from {path}")


def save_embedding_cache(model: LikelihoodModel, config: dict[str, Any]) -> None:
    """Persist model's embedding cache to disk.

    Parameters
    ----------
    model : LikelihoodModel
        Likelihood model whose embedding_cache will be saved.
    config : dict
        Full YAML config dict (used to resolve cache path).
    """
    path = embedding_cache_path(config)
    n = model.save_cache(path)
    if n > 0:
        print(f"Saved {n} embeddings to {path}")
