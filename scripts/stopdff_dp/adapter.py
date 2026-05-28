"""Normalize existing qanta-buzzer artifacts into the DP-StopDFF dataframe.

Adapter columns (frozen by ``types.ADAPTER_COLUMNS``):
    subject, item_id, prefix_idx, prefix_fraction, format, split,
    p_raw, p_calibrated, correct, top_answer, gold,
    category, option_set_id

``prefix_fraction`` = ``len(prefix) / len(full_question)`` — matches the
existing calibration / myopic-StopDFF convention and is used directly by
the empirical bucket fit and the CLI to assign early/mid/late buckets.

For each MC question and each cumulative prefix, the adapter emits two
rows: one for the MC format (max cosine similarity over the K=4
options, then Platt calibration per prefix bucket) and one for the QA
format (cosine similarity to ``answer_primary``, then Platt
calibration). Calibration coefficients are loaded from
``paper_exports/calibration.json``. SBERT variant: ``all-MiniLM-L6-v2``
(matches scripts/compute_stopdff.py for cross-metric consistency).

The ``correct`` column for QA rows is ALWAYS 1 by construction — the
QA condition simulates "the model knows the gold answer text", mirroring
``compute_stopdff.compute_stop_step_nonmc``. Downstream consumers must
not treat ``correct`` as a per-format accuracy signal for QA rows.

The adapter never touches the test split when fitting calibrators or
continuation buckets — those are caller responsibilities and the
adapter's ``split_name`` parameter only stamps the resulting rows.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .types import ADAPTER_COLUMNS

_SBERT_MODEL = None


def validate_split_separation(*, fit_split: str, eval_split: str) -> None:
    fit_name = fit_split.strip().lower()
    eval_name = eval_split.strip().lower()
    if fit_name == eval_name:
        raise ValueError(
            "fit and eval split must differ to avoid leakage; "
            f"got fit_split={fit_split!r}, eval_split={eval_split!r}."
        )
    if fit_name == "test":
        raise ValueError(
            "fit_split must not be test; test data is reserved for final "
            f"evaluation only (got eval_split={eval_split!r})."
        )


def validate_qid_separation(
    *,
    fit_qids: set[str] | frozenset[str],
    eval_qids: set[str] | frozenset[str],
    fit_split: str = "fit",
    eval_split: str = "eval",
) -> None:
    """Reject overlapping qids between fit and eval splits.

    ``validate_split_separation`` only checks the split NAMES. A
    consumer that hand-edits or regenerates the split JSON files
    could end up with two distinct split names whose qid sets
    overlap, in which case the empirical_bucket DP path would fit
    continuation buckets on items it later evaluates -- exactly
    the leakage the contract aims to prevent.

    Pass the actual qid sets here (after loading the two split
    files) so the leakage check is data-grounded rather than
    name-only.

    Raises
    ------
    ValueError
        If ``fit_qids & eval_qids`` is non-empty. The message
        lists up to 10 example overlapping qids so the operator
        can diagnose the bad split file.
    """
    overlap = set(fit_qids) & set(eval_qids)
    if not overlap:
        return

    def _natural_key(qid: str) -> tuple:
        # Natural-order key: split into runs of digits / non-digits so
        # 'shared_9' sorts before 'shared_10', and operators see the
        # numerically-lowest qids as the 10 examples rather than the
        # lexicographically-first ones (which is misleading when qids
        # follow a "<prefix>_<int>" convention).
        import re
        parts = re.split(r"(\d+)", qid)
        return tuple(
            (1, int(p)) if p.isdigit() else (0, p) for p in parts
        )

    sample = sorted(overlap, key=_natural_key)[:10]
    raise ValueError(
        f"fit split {fit_split!r} and eval split {eval_split!r} share "
        f"{len(overlap)} qid(s) -- DP empirical_bucket leakage. Examples: "
        f"{sample}. Rebuild the splits via scripts/fresh_split.py so the "
        f"qid sets are disjoint, or check the split JSON files for a bad "
        f"hand-edit."
    )


def _get_sbert_model():
    global _SBERT_MODEL
    if _SBERT_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _SBERT_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _SBERT_MODEL


def _platt(z: float, coef: float, intercept: float) -> float:
    lin = coef * z + intercept
    lin = max(-500.0, min(500.0, lin))
    return 1.0 / (1.0 + math.exp(-lin))


def _assign_bucket(frac: float) -> str:
    if frac < 0.33:
        return "early"
    if frac < 0.66:
        return "mid"
    return "late"


def _load_platt_params(calibration_path: Path) -> dict[str, tuple[float, float]]:
    """Reuse the loader logic from compute_stopdff (kept inline to avoid coupling)."""
    import json
    with open(calibration_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    out = {}
    for bucket_name, bucket_data in data["per_bucket"].items():
        coef = bucket_data["platt_coef"]
        intercept = bucket_data["platt_intercept"]
        if coef is None or intercept is None:
            if bucket_data.get("platt_model_type") != "constant":
                raise ValueError(
                    f"Bucket {bucket_name!r} has null Platt parameters "
                    "without platt_model_type='constant'."
                )
            probability = float(
                bucket_data.get("platt_constant_probability", 0.0)
            )
            coef = 0.0
            if probability <= 0.0:
                intercept = -500.0
            elif probability >= 1.0:
                intercept = 500.0
            else:
                intercept = math.log(probability / (1.0 - probability))
        out[bucket_name] = (coef, intercept)
    return out


def _score_question(
    question: dict,
    platt_params: dict[str, tuple[float, float]] | None,
    identity_calibration: bool,
) -> list[dict]:
    """Emit per-(prefix, format) rows for one MC question."""
    rows: list[dict] = []
    qid = str(question["qid"])
    full_q = question["question"]
    prefixes = question["cumulative_prefixes"]
    options = question["options"]
    gold_index = int(question["gold_index"])
    gold_text = options[gold_index]
    category = question.get("category", "")
    option_set_id = f"{qid}:K{len(options)}"
    full_len = max(1, len(full_q))

    if identity_calibration:
        # Test-only branch: replace SBERT cosine with a deterministic
        # synthetic signal so unit tests do not need the model.
        for t, prefix in enumerate(prefixes):
            prefix_fraction = len(prefix) / full_len
            p_mc = max(0.0, min(1.0, 0.3 + 0.5 * prefix_fraction))
            p_qa = max(0.0, min(1.0, 0.2 + 0.5 * prefix_fraction))
            rows.append({
                "subject": f"sbert:{category}",
                "item_id": qid,
                "prefix_idx": t,
                "prefix_fraction": prefix_fraction,
                "format": "MC",
                "split": None,  # caller stamps split
                "p_raw": p_mc,
                "p_calibrated": p_mc,
                "correct": int(t % 2 == 0),  # deterministic synthetic pattern
                "top_answer": gold_text if t % 2 == 0 else "synthetic_distractor",
                "gold": gold_text,
                "category": category,
                "option_set_id": option_set_id,
            })
            rows.append({
                "subject": f"sbert:{category}",
                "item_id": qid,
                "prefix_idx": t,
                "prefix_fraction": prefix_fraction,
                "format": "QA",
                "split": None,
                "p_raw": p_qa,
                "p_calibrated": p_qa,
                "correct": 1,
                "top_answer": gold_text,
                "gold": gold_text,
                "category": category,
                "option_set_id": option_set_id,
            })
        return rows

    from sklearn.metrics.pairwise import cosine_similarity
    model = _get_sbert_model()
    option_embs = model.encode(options, convert_to_numpy=True)
    answer_emb = model.encode(
        [question["answer_primary"]], convert_to_numpy=True
    )

    for t, prefix in enumerate(prefixes):
        prefix_fraction = len(prefix) / full_len
        bucket = _assign_bucket(prefix_fraction)
        coef, intercept = (
            platt_params[bucket] if platt_params is not None else (1.0, 0.0)
        )
        prefix_emb = model.encode([prefix], convert_to_numpy=True)

        # MC: max similarity over options.
        mc_sims = cosine_similarity(prefix_emb, option_embs)[0]
        max_sim = float(np.max(mc_sims))
        predicted_idx = int(np.argmax(mc_sims))
        rows.append({
            "subject": f"sbert:{category}",
            "item_id": qid,
            "prefix_idx": t,
            "prefix_fraction": prefix_fraction,
            "format": "MC",
            "split": None,
            "p_raw": max_sim,
            "p_calibrated": _platt(max_sim, coef, intercept),
            "correct": int(predicted_idx == gold_index),
            "top_answer": options[predicted_idx],
            "gold": gold_text,
            "category": category,
            "option_set_id": option_set_id,
        })

        # QA: similarity to answer_primary only.
        qa_sim = float(cosine_similarity(prefix_emb, answer_emb)[0][0])
        rows.append({
            "subject": f"sbert:{category}",
            "item_id": qid,
            "prefix_idx": t,
            "prefix_fraction": prefix_fraction,
            "format": "QA",
            "split": None,
            "p_raw": qa_sim,
            "p_calibrated": _platt(qa_sim, coef, intercept),
            "correct": 1,  # QA "top answer" is always the gold by construction
            "top_answer": question["answer_primary"],
            "gold": gold_text,
            "category": category,
            "option_set_id": option_set_id,
        })

    return rows


def build_dataframe(
    *,
    mc_questions: Sequence[dict],
    target_qids: set[str],
    split_name: str,
    calibration_path: Path | None = None,
    identity_calibration: bool = False,
) -> pd.DataFrame:
    """Build the normalised dataframe for one split.

    Parameters
    ----------
    mc_questions : Sequence[dict]
        MC question dicts (output of ``iter_split_questions``).
    target_qids : set[str]
        The qid set defining the requested split (e.g. all val qids).
    split_name : str
        Stamped onto the ``split`` column of every produced row.
    calibration_path : Path, optional
        Path to ``calibration.json``. Required unless
        ``identity_calibration`` is True.
    identity_calibration : bool
        Skip the SBERT model and emit a deterministic synthetic signal.
        Test-only escape hatch.

    Returns
    -------
    pd.DataFrame
        Columns match ``types.ADAPTER_COLUMNS``.
    """
    if not identity_calibration:
        if calibration_path is None or not Path(calibration_path).exists():
            raise FileNotFoundError(
                "calibration_path must exist when identity_calibration=False; "
                f"got {calibration_path!r}."
            )
    platt_params = (
        None if identity_calibration
        else _load_platt_params(Path(calibration_path))
    )

    rows: list[dict] = []
    for q in mc_questions:
        if str(q["qid"]) not in target_qids:
            continue
        rows.extend(_score_question(q, platt_params, identity_calibration))

    df = pd.DataFrame(rows, columns=list(ADAPTER_COLUMNS))
    df["split"] = split_name
    return df[list(ADAPTER_COLUMNS)]
