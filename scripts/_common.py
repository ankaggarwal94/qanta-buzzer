"""Shared utilities for pipeline scripts.

Provides config loading, JSON serialization, MC question deserialization,
and path constants used across all pipeline scripts (build, baseline, train,
evaluate).

Ported from qb-rl reference implementation with import path adaptations
for the unified qanta-buzzer codebase.
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from models.likelihoods import LikelihoodModel, build_likelihood_from_config
from qb_data.config import load_config as load_yaml_config
from qb_data.mc_builder import MCQuestion

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"


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


def to_serializable(item: Any) -> Any:
    """Recursively convert dataclasses to dicts for JSON serialization.

    Parameters
    ----------
    item : Any
        Object to convert. Dataclasses are converted via ``asdict()``,
        dicts and lists are processed recursively.

    Returns
    -------
    Any
        JSON-serializable version of the input.
    """
    if is_dataclass(item):
        return asdict(item)
    if isinstance(item, dict):
        return {k: to_serializable(v) for k, v in item.items()}
    if isinstance(item, list):
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
        human_buzz_positions=row.get("human_buzz_positions"),
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

    Parameters
    ----------
    path : str or Path
        Path to JSON file containing a list of serialized MCQuestion dicts.

    Returns
    -------
    list[MCQuestion]
        List of reconstructed MCQuestion instances.
    """
    raw = load_json(path)
    return [mc_question_from_dict(item) for item in raw]


# ------------------------------------------------------------------ #
# Embedding cache persistence helpers
# ------------------------------------------------------------------ #


def embedding_cache_path(config: dict[str, Any]) -> Path:
    """Return the resolved embedding cache file path from config.

    Uses ``config['likelihood']['cache_dir']`` (default ``'cache/embeddings'``)
    and appends ``'embedding_cache.npz'``.

    Parameters
    ----------
    config : dict
        Full YAML config dict.

    Returns
    -------
    Path
        Absolute path to the embedding cache ``.npz`` file.
    """
    cache_dir = config.get("likelihood", {}).get("cache_dir", "cache/embeddings")
    return PROJECT_ROOT / cache_dir / "embedding_cache.npz"


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
