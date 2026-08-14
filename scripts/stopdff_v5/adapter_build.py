"""Model-snapshot freeze + deterministic adapter builder (Modal-side, real data).

freeze_model_snapshot: resolve a concrete 40-hex revision of
sentence-transformers/all-MiniLM-L6-v2, download the snapshot (trust_remote_code=false),
hash every file, and write a model-snapshot manifest.

build_adapter_bundle: score paired MC/QA rows per prefix with the frozen model and write
byte-deterministic fit_rows.jsonl.gz / eval_rows.jsonl.gz plus the adapter manifest.
"""
from __future__ import annotations

import json
import math
import re
import shutil
from pathlib import Path
from typing import Any

from qb_data.dataset_splits import normalize_question_text, normalize_split_answer

from .identity import build_manifest, load_json_strict, sha256_bytes, sha256_file
from .manifests import (
    adapter_identity,
    model_snapshot_identity,
    question_trajectory_binding_id,
)
from .producers import _record_value

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
ADAPTER_SCHEMA_COLUMNS = [
    "item_id", "prefix_idx", "prefix_fraction", "format", "split",
    "prefix_text_sha256", "prefix_char_count",
    "full_question_sha256", "full_question_char_count",
    "raw_similarity", "correct", "category", "K", "option_set_id",
    "distractor_strategy", "p_second_best", "top2_margin",
]
_ROUND = 6  # decimals for similarity fields; coarse enough to absorb GPU float jitter
_ENCODE_BATCH_SIZE = 64
# (byte-identical rows across builds) while far finer than calibration/binning needs.


def freeze_model_snapshot(out_dir: Path, *, model_id: str = MODEL_ID, revision: str | None = None) -> dict[str, Any]:
    from huggingface_hub import HfApi, snapshot_download
    import sentence_transformers
    import transformers

    out_dir = Path(out_dir)
    snap = out_dir / "snapshot"
    snap.mkdir(parents=True, exist_ok=True)

    if revision is None:
        revision = HfApi().model_info(model_id).sha
    if not re.fullmatch(r"[0-9a-f]{40}", revision or ""):
        raise ValueError(f"model revision must be a concrete 40-hex commit, got {revision!r}")

    snapshot_download(repo_id=model_id, revision=revision, local_dir=str(snap))

    # huggingface_hub's local_dir layout records transport metadata (etags,
    # wall-clock timestamps) under .cache/huggingface/. Those bytes are not
    # part of the pinned revision, and hashing them would make identical
    # revisions freeze to different content-addressed identities, so prune
    # the directory before the inventory walk (downstream bound-content
    # validation reproduces this walk and requires exact correspondence).
    hub_cache = snap / ".cache"
    if hub_cache.is_dir():
        shutil.rmtree(hub_cache)

    files: list[dict[str, Any]] = []
    for p in sorted(snap.rglob("*")):
        if p.is_file():
            files.append({"path": p.relative_to(snap).as_posix(), "size": p.stat().st_size,
                          "sha256": sha256_file(p)})

    identity = model_snapshot_identity(
        model_id=model_id, revision=revision, files=files,
        sentence_transformers_version=sentence_transformers.__version__,
        transformers_version=transformers.__version__,
    )
    manifest = build_manifest(identity, snapshot_dir=str(snap))
    (out_dir / "model_snapshot_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def _load_mc_questions(mc_dataset_path: Path) -> list[dict]:
    data = load_json_strict(mc_dataset_path)
    records = data["questions"] if isinstance(data, dict) and "questions" in data else data
    if not isinstance(records, list) or not all(isinstance(record, dict) for record in records):
        raise ValueError(f"MC dataset {mc_dataset_path} must contain question objects")
    return records


def _record_qid(record: dict[str, Any]) -> str:
    raw_qid = _record_value(record, ("qid", "question_id", "id"))
    if not isinstance(raw_qid, (str, int)) or isinstance(raw_qid, bool):
        return ""
    return str(raw_qid)


def _dataset_index(path: Path, *, split: str) -> dict[str, dict[str, str]]:
    data = load_json_strict(path)
    records = data["questions"] if isinstance(data, dict) and "questions" in data else data
    if not isinstance(records, list) or not all(isinstance(record, dict) for record in records):
        raise ValueError(f"{split} dataset {path} must contain question objects")
    out: dict[str, dict[str, str]] = {}
    for rec in records:
        qid = _record_qid(rec)
        try:
            text = normalize_question_text(_record_value(rec, ("question", "text")))
            answer = normalize_split_answer(
                _record_value(rec, ("answer_primary", "answer", "answer_text"))
            )
        except TypeError as exc:
            raise ValueError(
                f"{split} dataset record has non-string question text or answer"
            ) from exc
        category = rec.get("category")
        if not isinstance(category, str) or not category.strip():
            raise ValueError(f"{split} dataset record has invalid category")
        if not qid or not text or not answer:
            raise ValueError(f"{split} dataset record lacks qid, question text, or answer")
        if qid in out:
            raise ValueError(f"{split} dataset contains duplicate qid {qid!r}")
        out[qid] = {"text": text, "answer": answer, "category": category}
    return out


def _load_calibration(path: Path) -> dict[str, Any]:
    """Load the calibration artifact and bind it to the adapter fit split."""
    data = load_json_strict(path)
    if not isinstance(data, dict):
        raise ValueError(f"calibration {path} must contain an object")
    metadata = data.get("metadata")
    if metadata is not None and not isinstance(metadata, dict):
        raise ValueError(f"calibration {path} metadata must contain an object")
    fit_split = data.get("fit_split")
    if fit_split is None and isinstance(metadata, dict):
        fit_split = metadata.get("fit_split")
    if fit_split != "val":
        raise ValueError(
            f"calibration {path} must declare fit_split='val', got {fit_split!r}"
        )
    return data


def derive_bound_calibration(
    *,
    fit_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    model_snapshot_id: str,
    fit_rows_sha256: str,
) -> dict[str, Any]:
    """Fit the staged Platt contract from the exact adapter row bytes.

    The historical pipeline copied an independently generated calibration
    artifact into a newly scored adapter.  Re-fitting here removes that
    co-presence gap: the coefficients are deterministically derived from the
    bound validation MC rows and the resulting artifact records both the row
    hash and model-snapshot identity.
    """
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from .calibrators import apply_platt_logistic, require_phase_fit_prerequisites

    phases = ("early", "mid", "late")

    def phase_of(value: float) -> str:
        if value < 0.33:
            return "early"
        if value < 0.66:
            return "mid"
        return "late"

    fit_by_phase: dict[str, list[dict[str, Any]]] = {phase: [] for phase in phases}
    eval_by_phase: dict[str, list[dict[str, Any]]] = {phase: [] for phase in phases}
    for rows, expected_split, target in (
        (fit_rows, "val", fit_by_phase),
        (eval_rows, "test", eval_by_phase),
    ):
        for row in rows:
            if row.get("format") == "MC" and row.get("split") == expected_split:
                target[phase_of(float(row["prefix_fraction"]))].append(row)

    def ece(probabilities: np.ndarray, labels: np.ndarray) -> float:
        if len(probabilities) == 0:
            return 0.0
        total = 0.0
        edges = np.linspace(0.0, 1.0, 11)
        for index in range(10):
            lower, upper = edges[index], edges[index + 1]
            mask = (
                (probabilities >= lower)
                & (
                    probabilities <= upper
                    if index == 9
                    else probabilities < upper
                )
            )
            count = int(mask.sum())
            if count:
                total += (count / len(probabilities)) * abs(
                    float(labels[mask].mean())
                    - float(probabilities[mask].mean())
                )
        return total

    per_bucket: dict[str, dict[str, Any]] = {}
    for phase in phases:
        require_phase_fit_prerequisites(
            fit_by_phase[phase], phase, "platt-logistic"
        )
        fit_scores = np.asarray(
            [float(row["raw_similarity"]) for row in fit_by_phase[phase]],
            dtype=np.float64,
        )
        fit_labels = np.asarray(
            [int(row["correct"]) for row in fit_by_phase[phase]],
            dtype=np.int64,
        )
        eval_scores = np.asarray(
            [float(row["raw_similarity"]) for row in eval_by_phase[phase]],
            dtype=np.float64,
        )
        eval_labels = np.asarray(
            [int(row["correct"]) for row in eval_by_phase[phase]],
            dtype=np.int64,
        )
        model = LogisticRegression(
            C=1.0,
            solver="lbfgs",
            max_iter=1000,
            random_state=789685,
        )
        model.fit(fit_scores.reshape(-1, 1), fit_labels)
        coefficient = round(float(model.coef_[0][0]), 6)
        intercept = round(float(model.intercept_[0]), 6)
        model_type = "logistic"
        fallback_reason = None
        constant_probability = None
        probabilities = np.asarray(
            [
                apply_platt_logistic(score, coefficient, intercept)
                for score in eval_scores
            ],
            dtype=np.float64,
        )
        per_bucket[phase] = {
            "ece": round(ece(probabilities, eval_labels), 6),
            "n_samples": int(len(eval_labels)),
            "platt_coef": coefficient,
            "platt_intercept": intercept,
            "platt_model_type": model_type,
            "platt_fallback_reason": fallback_reason,
            "platt_constant_probability": constant_probability,
        }
    return {
        "fit_split": "val",
        "per_bucket": per_bucket,
        "metadata": {
            "fit_split": "val",
            "model": MODEL_ID,
            "model_snapshot_id": model_snapshot_id,
            "fit_rows_sha256": fit_rows_sha256,
            "algorithm": "platt_lbfgs_c1_seed789685_v1",
        },
    }


def _validate_scoring_question(question: dict[str, Any]) -> None:
    """Reject malformed MC rows before any model scoring occurs."""
    qid = _record_qid(question)
    if not qid:
        raise ValueError("MC scoring question lacks a scalar qid")
    raw_question = _record_value(question, ("question", "text"))
    if not isinstance(raw_question, str) or not normalize_question_text(raw_question):
        raise ValueError(f"MC scoring question {qid!r} has invalid question text")
    raw_answer = _record_value(question, ("answer_primary", "answer", "answer_text"))
    if not isinstance(raw_answer, str) or not normalize_split_answer(raw_answer):
        raise ValueError(f"MC scoring question {qid!r} has invalid answer_primary")
    prefixes = question.get("cumulative_prefixes")
    if (
        not isinstance(prefixes, list)
        or not prefixes
        or any(not isinstance(prefix, str) or not prefix.strip() for prefix in prefixes)
    ):
        raise ValueError(
            f"MC scoring question {qid!r} has invalid cumulative_prefixes"
        )
    full_tokens = normalize_question_text(raw_question).split()
    prefix_tokens = [
        normalize_question_text(prefix).split()
        for prefix in prefixes
    ]
    if any(
        len(current) >= len(following)
        for current, following in zip(prefix_tokens, prefix_tokens[1:])
    ):
        raise ValueError(
            f"MC scoring question {qid!r} does not have nondecreasing "
            "cumulative_prefixes or strictly extending cumulative_prefixes"
        )
    if any(
        tokens != full_tokens[: len(tokens)]
        for tokens in prefix_tokens
    ):
        raise ValueError(
            f"MC scoring question {qid!r} cumulative_prefixes contain a value "
            "that is not a canonical question-token prefix"
        )
    if prefix_tokens[-1] != full_tokens:
        raise ValueError(
            f"MC scoring question {qid!r} final cumulative prefix does not "
            "equal the canonical full question"
        )
    options = question.get("options")
    if (
        not isinstance(options, list)
        or len(options) < 2
        or any(not isinstance(option, str) or not option.strip() for option in options)
    ):
        raise ValueError(f"MC scoring question {qid!r} has invalid options")
    gold_index = question.get("gold_index")
    if (
        isinstance(gold_index, bool)
        or not isinstance(gold_index, int)
        or not 0 <= gold_index < len(options)
    ):
        raise ValueError(f"MC scoring question {qid!r} has invalid gold_index")
    normalized_options = [normalize_split_answer(option) for option in options]
    if any(not option for option in normalized_options):
        raise ValueError(
            f"MC scoring question {qid!r} has an empty normalized option"
        )
    if len(set(normalized_options)) != len(normalized_options):
        raise ValueError(
            f"MC scoring question {qid!r} has duplicate normalized options"
        )
    if normalized_options[gold_index] != normalize_split_answer(raw_answer):
        raise ValueError(
            f"MC scoring question {qid!r} gold_index does not identify "
            "answer_primary"
        )
    category = question.get("category")
    if not isinstance(category, str) or not category.strip():
        raise ValueError(f"MC scoring question {qid!r} has invalid category")


def _validate_split_bindings(
    val_index: dict[str, dict[str, str]],
    test_index: dict[str, dict[str, str]],
    questions: list[dict],
) -> dict[str, dict[str, Any]]:
    """Validate adapter split membership against dataset and MC question bytes."""
    qid_overlap = set(val_index) & set(test_index)
    if qid_overlap:
        raise ValueError(f"adapter fit/eval qid overlap: {sorted(qid_overlap)[:10]}")

    val_texts = {entry["text"] for entry in val_index.values()}
    test_texts = {entry["text"] for entry in test_index.values()}
    text_overlap = val_texts & test_texts
    if text_overlap:
        raise ValueError(
            "adapter fit/eval normalized question-text overlap: "
            f"{sorted(text_overlap)[:10]}"
        )

    answers_by_text: dict[str, set[str]] = {}
    for entry in list(val_index.values()) + list(test_index.values()):
        answers_by_text.setdefault(entry["text"], set()).add(entry["answer"])
    conflicts = {
        text: answers
        for text, answers in answers_by_text.items()
        if len(answers) > 1
    }
    if conflicts:
        raise ValueError(
            "adapter source contains conflicting normalized answers: "
            f"{sorted(conflicts)[:10]}"
        )

    mc_index: dict[str, dict[str, Any]] = {}
    for question in questions:
        qid = _record_qid(question)
        if not qid:
            raise ValueError("MC dataset record lacks qid")
        if qid in mc_index:
            raise ValueError(f"MC dataset contains duplicate qid {qid!r}")
        mc_index[qid] = question

    for split, split_index in (("val", val_index), ("test", test_index)):
        for qid, question in mc_index.items():
            if qid not in split_index:
                continue
            source = split_index[qid]
            try:
                mc_text = normalize_question_text(
                    _record_value(question, ("question", "text"))
                )
                mc_answer = normalize_split_answer(
                    _record_value(
                        question,
                        ("answer_primary", "answer", "answer_text"),
                    )
                )
            except TypeError as exc:
                raise ValueError(
                    f"MC question {qid!r} has non-string question text or answer"
                ) from exc
            if mc_text != source["text"]:
                raise ValueError(
                    f"MC question {qid!r} text does not match its {split} split record"
                )
            if mc_answer != source["answer"]:
                raise ValueError(
                    f"MC question {qid!r} answer does not match its {split} split record"
                )
            mc_category = question.get("category")
            if mc_category != source["category"]:
                raise ValueError(
                    f"MC question {qid!r} category does not match its "
                    f"{split} split record"
                )

        missing = set(split_index) - set(mc_index)
        if missing:
            raise ValueError(
                f"MC dataset is missing {split} split qids: {sorted(missing)[:10]}"
            )

    return mc_index


def _score_question_rows(
    question: dict,
    model,
    split: str,
    *,
    embeddings=None,
) -> list[dict]:
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity

    _validate_scoring_question(question)
    qid = _record_qid(question)
    full_q = _record_value(question, ("question", "text"))
    prefixes = question["cumulative_prefixes"]
    options = question["options"]
    gold_index = int(question["gold_index"])
    category = question["category"]
    K = int(len(options))
    option_set_id = f"{qid}:K{K}"
    distractor_strategy = str(question.get("distractor_strategy", "unknown"))
    canonical_full_q = normalize_question_text(full_q)
    canonical_prefixes = [
        normalize_question_text(prefix)
        for prefix in prefixes
    ]
    full_len = len(canonical_full_q)
    full_question_sha256 = sha256_bytes(canonical_full_q.encode("utf-8"))

    texts = [
        *options,
        _record_value(question, ("answer_primary", "answer", "answer_text")),
        *prefixes,
    ]
    if embeddings is None:
        embeddings = model.encode(
            texts,
            batch_size=_ENCODE_BATCH_SIZE,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
    embeddings = np.asarray(embeddings)
    if len(embeddings) != len(texts):
        raise ValueError("encoder output does not align with requested texts")
    option_end = len(options)
    option_embs = embeddings[:option_end]
    answer_emb = embeddings[option_end : option_end + 1]
    prefix_embs = embeddings[option_end + 1 :]

    rows: list[dict] = []
    for t, prefix in enumerate(prefixes):
        canonical_prefix = canonical_prefixes[t]
        prefix_len = len(canonical_prefix)
        prefix_fraction = (
            1.0
            if t == len(prefixes) - 1
            else round(prefix_len / full_len, _ROUND)
        )
        content_binding = {
            "prefix_text_sha256": sha256_bytes(
                canonical_prefix.encode("utf-8")
            ),
            "prefix_char_count": prefix_len,
            "full_question_sha256": full_question_sha256,
            "full_question_char_count": full_len,
        }
        pe = prefix_embs[t : t + 1]
        mc_sims = cosine_similarity(pe, option_embs)[0]
        max_sim = float(np.max(mc_sims))
        predicted_idx = int(np.argmax(mc_sims))
        second_best = float(np.partition(mc_sims, -2)[-2]) if len(mc_sims) >= 2 else 0.0
        rows.append({
            "item_id": qid, "prefix_idx": t, "prefix_fraction": prefix_fraction, "format": "MC",
            **content_binding,
            "split": split, "raw_similarity": round(max_sim, _ROUND),
            "correct": int(predicted_idx == gold_index), "category": category, "K": K,
            "option_set_id": option_set_id, "distractor_strategy": distractor_strategy,
            "p_second_best": round(second_best, _ROUND), "top2_margin": round(max_sim - second_best, _ROUND),
        })
        qa_sim = float(cosine_similarity(pe, answer_emb)[0][0])
        rows.append({
            "item_id": qid, "prefix_idx": t, "prefix_fraction": prefix_fraction, "format": "QA",
            **content_binding,
            "split": split, "raw_similarity": round(qa_sim, _ROUND), "correct": 1,
            "category": category, "K": K, "option_set_id": option_set_id,
            "distractor_strategy": distractor_strategy, "p_second_best": 0.0, "top2_margin": 0.0,
        })
    return rows


def _select_retained_questions(
    questions: list[dict[str, Any]],
    val_qids: set[str],
    test_qids: set[str],
) -> list[tuple[dict[str, Any], str]]:
    """Select validated MC rows using the repository's accepted qid aliases."""
    retained: list[tuple[dict[str, Any], str]] = []
    for question in sorted(questions, key=_record_qid):
        qid = _record_qid(question)
        if not qid:
            raise ValueError("MC dataset record lacks qid")
        if qid in val_qids:
            retained.append((question, "val"))
        if qid in test_qids:
            retained.append((question, "test"))
    return retained


def _score_questions_rows(
    questions: list[tuple[dict[str, Any], str]],
    model,
) -> list[dict[str, Any]]:
    """Score all retained questions through one bounded encoder dispatch."""
    flattened_texts: list[str] = []
    slices: list[tuple[dict[str, Any], str, int, int]] = []
    for question, split in questions:
        _validate_scoring_question(question)
        texts = [
            *question["options"],
            _record_value(question, ("answer_primary", "answer", "answer_text")),
            *question["cumulative_prefixes"],
        ]
        start = len(flattened_texts)
        flattened_texts.extend(texts)
        slices.append((question, split, start, len(flattened_texts)))
    if not flattened_texts:
        return []
    embeddings = model.encode(
        flattened_texts,
        batch_size=_ENCODE_BATCH_SIZE,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    if len(embeddings) != len(flattened_texts):
        raise ValueError("encoder output does not align with requested texts")
    rows: list[dict[str, Any]] = []
    for question, split, start, end in slices:
        rows.extend(
            _score_question_rows(
                question,
                None,
                split,
                embeddings=embeddings[start:end],
            )
        )
    return rows


def _sorted_rows(rows: list[dict]) -> list[dict]:
    return sorted(rows, key=lambda r: (str(r["item_id"]), r["format"], int(r["prefix_idx"])))


def question_trajectory_binding_from_rows(rows: list[dict[str, Any]]) -> str:
    """Bind each accepted trajectory to normalized prefix/full-question bytes."""
    fields = (
        "split",
        "item_id",
        "prefix_idx",
        "prefix_text_sha256",
        "prefix_char_count",
        "full_question_sha256",
        "full_question_char_count",
    )
    records = [
        {field: row[field] for field in fields}
        for row in rows
        if row.get("format") == "MC"
    ]
    return question_trajectory_binding_id(records)


def _mc_coverage_evidence(rows: list[dict]) -> dict[str, Any]:
    """Summarize eval item counts and exact MC/QA prefix pairing."""
    mc_prefixes = {
        (str(row["item_id"]), int(row["prefix_idx"]))
        for row in rows
        if row["format"] == "MC"
    }
    qa_prefixes = {
        (str(row["item_id"]), int(row["prefix_idx"]))
        for row in rows
        if row["format"] == "QA"
    }
    return {
        "eval_mc_items": len({item_id for item_id, _ in mc_prefixes}),
        "eval_qa_items": len({item_id for item_id, _ in qa_prefixes}),
        "paired": mc_prefixes == qa_prefixes,
    }


def _mc_retention_evidence(
    data_dir: Path,
    *,
    fit_item_count: int,
    eval_item_count: int,
    allow_low_mc_retention: bool,
) -> dict[str, Any]:
    """Bind and enforce the repository's full-profile MC retention policy."""
    from scripts._audit_gates import (
        build_retention_metadata,
        load_mc_build_metadata,
    )

    if not isinstance(allow_low_mc_retention, bool):
        raise TypeError("allow_low_mc_retention must be boolean")
    try:
        build_metadata = load_mc_build_metadata(Path(data_dir))
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid MC build retention metadata: {exc}") from exc
    if build_metadata.get("status") != "loaded":
        raise ValueError("MC build retention metadata is required")

    split_evidence: dict[str, dict[str, Any]] = {}
    for role, split, retained_items in (
        ("fit", "val", fit_item_count),
        ("eval", "test", eval_item_count),
    ):
        try:
            decision = build_retention_metadata(
                build_metadata,
                split=split,
                smoke=False,
                explicit_threshold=None,
                override=allow_low_mc_retention,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"invalid MC {split} retention metadata: {exc}"
            ) from exc
        if not decision["applies"] or decision["passed"] is None:
            raise ValueError(f"MC {split} retention metadata is required")

        raw_count = decision["raw_count"]
        retained_count = decision["retained_count"]
        dropped_count = decision["dropped_count"]
        if (
            any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                for value in (raw_count, retained_count, dropped_count)
            )
            or retained_count + dropped_count != raw_count
        ):
            raise ValueError(f"MC {split} retention counts are inconsistent")
        if retained_count != retained_items:
            raise ValueError(
                f"MC {split} retained_count does not match retained dataset"
            )
        recomputed_rate = retained_count / raw_count if raw_count else 0.0
        if not math.isclose(
            decision["retention_rate"],
            recomputed_rate,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"MC {split} retention_rate is inconsistent")

        decision = dict(decision)
        decision["effective_pass"] = bool(
            decision["passed"] or decision["overridden"]
        )
        if not decision["effective_pass"]:
            raise ValueError(
                f"MC {split} retention {decision['retention_rate']:.1%} "
                f"is below the full-profile threshold "
                f"{decision['threshold']:.1%}"
            )
        split_evidence[role] = {
            **decision,
            "threshold": repr(float(decision["threshold"])),
            "retention_rate": repr(float(decision["retention_rate"])),
        }

    return {
        "build_metadata_sha256": build_metadata["source_sha256"],
        "threshold_profile": "full",
        "splits": split_evidence,
    }


def build_adapter_bundle(
    *,
    mc_dataset_path: Path,
    val_dataset_path: Path,
    test_dataset_path: Path,
    calibration_path: Path,
    model_snapshot_dir: Path,
    out_dir: Path,
    source_manifest_id: str,
    raw_input_bundle_id: str,
    model_snapshot_id: str,
    producer_hashes: dict[str, str],
    allow_low_mc_retention: bool = False,
) -> dict[str, Any]:
    from .rowio import write_jsonl_gz

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    val_index = _dataset_index(val_dataset_path, split="val")
    test_index = _dataset_index(test_dataset_path, split="test")
    questions = _load_mc_questions(mc_dataset_path)
    _validate_split_bindings(val_index, test_index, questions)
    val_qids = set(val_index)
    test_qids = set(test_index)
    mc_retention = _mc_retention_evidence(
        Path(mc_dataset_path).parent,
        fit_item_count=len(val_index),
        eval_item_count=len(test_index),
        allow_low_mc_retention=allow_low_mc_retention,
    )
    _load_calibration(calibration_path)
    for question in questions:
        if _record_qid(question) in val_qids | test_qids:
            _validate_scoring_question(question)

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(str(model_snapshot_dir), trust_remote_code=False)

    retained_questions = _select_retained_questions(
        questions,
        val_qids,
        test_qids,
    )

    scored_rows = _score_questions_rows(retained_questions, model)
    fit_rows = [row for row in scored_rows if row["split"] == "val"]
    eval_rows = [row for row in scored_rows if row["split"] == "test"]

    fit_rows = _sorted_rows(fit_rows)
    eval_rows = _sorted_rows(eval_rows)
    write_jsonl_gz(out_dir / "fit_rows.jsonl.gz", fit_rows)
    write_jsonl_gz(out_dir / "eval_rows.jsonl.gz", eval_rows)
    fit_sha = sha256_file(out_dir / "fit_rows.jsonl.gz")
    eval_sha = sha256_file(out_dir / "eval_rows.jsonl.gz")

    # Derive calibration from the exact scored fit rows; do not copy unrelated
    # staged coefficients into this adapter.
    bound_calibration = derive_bound_calibration(
        fit_rows=fit_rows,
        eval_rows=eval_rows,
        model_snapshot_id=model_snapshot_id,
        fit_rows_sha256=fit_sha,
    )
    (out_dir / "calibration.json").write_text(
        json.dumps(bound_calibration, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    build_metadata_path = Path(mc_dataset_path).parent / "build_metadata.json"
    (out_dir / "build_metadata.json").write_bytes(build_metadata_path.read_bytes())
    calibration_sha = sha256_file(out_dir / "calibration.json")

    mc_coverage = _mc_coverage_evidence(eval_rows)
    mc_retention.update(
        {"fit_rows": len(fit_rows), "eval_rows": len(eval_rows)}
    )

    identity = adapter_identity(
        source_manifest_id=source_manifest_id, raw_input_bundle_id=raw_input_bundle_id,
        model_snapshot_id=model_snapshot_id,
        scoring_spec={"model_id": MODEL_ID, "mc": "max_cosine_over_options",
                      "qa": "cosine_to_answer_primary", "round_decimals": _ROUND},
        fit_split="val", eval_split="test", schema_columns=ADAPTER_SCHEMA_COLUMNS,
        fit_row_count=len(fit_rows), eval_row_count=len(eval_rows),
        fit_rows_sha256=fit_sha, eval_rows_sha256=eval_sha,
        calibration_sha256=calibration_sha,
        question_trajectory_binding_id=question_trajectory_binding_from_rows(
            fit_rows + eval_rows
        ),
        mc_coverage=mc_coverage, mc_retention=mc_retention, producer_hashes=producer_hashes,
    )
    manifest = build_manifest(identity)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    (out_dir / "adapter_build_record.json").write_text(
        json.dumps({"fit_rows": len(fit_rows), "eval_rows": len(eval_rows),
                    "val_items": len(val_qids), "test_items": len(test_qids)},
                   indent=2, sort_keys=True), encoding="utf-8")
    return manifest
