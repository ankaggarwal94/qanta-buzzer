"""Model-snapshot freeze + deterministic adapter builder (Modal-side, real data).

freeze_model_snapshot: resolve a concrete 40-hex revision of
sentence-transformers/all-MiniLM-L6-v2, download the snapshot (trust_remote_code=false),
hash every file, and write a model-snapshot manifest.

build_adapter_bundle: score paired MC/QA rows per prefix with the frozen model and write
byte-deterministic fit_rows.jsonl.gz / eval_rows.jsonl.gz plus the adapter manifest.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Sequence

from .identity import build_manifest, compute_id, sha256_file
from .manifests import adapter_identity, model_snapshot_identity

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
ADAPTER_SCHEMA_COLUMNS = [
    "item_id", "prefix_idx", "prefix_fraction", "format", "split",
    "raw_similarity", "correct", "category", "K", "option_set_id",
    "distractor_strategy", "p_second_best", "top2_margin",
]
_ROUND = 10  # decimals for similarity fields (stable bytes across identical builds)


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
    data = json.loads(Path(mc_dataset_path).read_text(encoding="utf-8"))
    return data["questions"] if isinstance(data, dict) and "questions" in data else data


def _dataset_qids(path: Path) -> set[str]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    records = data["questions"] if isinstance(data, dict) and "questions" in data else data
    out: set[str] = set()
    for rec in records:
        for key in ("qid", "question_id", "id"):
            if key in rec:
                out.add(str(rec[key]))
                break
    return out


def _score_question_rows(question: dict, model, split: str) -> list[dict]:
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity

    qid = str(question["qid"])
    full_q = question["question"]
    prefixes = question["cumulative_prefixes"]
    options = question["options"]
    gold_index = int(question["gold_index"])
    category = str(question.get("category", ""))
    K = int(len(options))
    option_set_id = f"{qid}:K{K}"
    distractor_strategy = str(question.get("distractor_strategy", "unknown"))
    full_len = max(1, len(full_q))

    option_embs = model.encode(options, convert_to_numpy=True)
    answer_emb = model.encode([question["answer_primary"]], convert_to_numpy=True)
    prefix_embs = model.encode(list(prefixes), convert_to_numpy=True)

    rows: list[dict] = []
    for t, prefix in enumerate(prefixes):
        prefix_fraction = round(len(prefix) / full_len, _ROUND)
        pe = prefix_embs[t : t + 1]
        mc_sims = cosine_similarity(pe, option_embs)[0]
        max_sim = float(np.max(mc_sims))
        predicted_idx = int(np.argmax(mc_sims))
        second_best = float(np.partition(mc_sims, -2)[-2]) if len(mc_sims) >= 2 else 0.0
        rows.append({
            "item_id": qid, "prefix_idx": t, "prefix_fraction": prefix_fraction, "format": "MC",
            "split": split, "raw_similarity": round(max_sim, _ROUND),
            "correct": int(predicted_idx == gold_index), "category": category, "K": K,
            "option_set_id": option_set_id, "distractor_strategy": distractor_strategy,
            "p_second_best": round(second_best, _ROUND), "top2_margin": round(max_sim - second_best, _ROUND),
        })
        qa_sim = float(cosine_similarity(pe, answer_emb)[0][0])
        rows.append({
            "item_id": qid, "prefix_idx": t, "prefix_fraction": prefix_fraction, "format": "QA",
            "split": split, "raw_similarity": round(qa_sim, _ROUND), "correct": 1,
            "category": category, "K": K, "option_set_id": option_set_id,
            "distractor_strategy": distractor_strategy, "p_second_best": 0.0, "top2_margin": 0.0,
        })
    return rows


def _sorted_rows(rows: list[dict]) -> list[dict]:
    return sorted(rows, key=lambda r: (str(r["item_id"]), r["format"], int(r["prefix_idx"])))


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
) -> dict[str, Any]:
    from sentence_transformers import SentenceTransformer

    from .rowio import write_jsonl_gz

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = SentenceTransformer(str(model_snapshot_dir), trust_remote_code=False)

    val_qids = _dataset_qids(val_dataset_path)
    test_qids = _dataset_qids(test_dataset_path)
    questions = _load_mc_questions(mc_dataset_path)

    fit_rows: list[dict] = []
    eval_rows: list[dict] = []
    for q in sorted(questions, key=lambda x: str(x["qid"])):
        qid = str(q["qid"])
        if qid in val_qids:
            fit_rows.extend(_score_question_rows(q, model, "val"))
        if qid in test_qids:
            eval_rows.extend(_score_question_rows(q, model, "test"))

    fit_rows = _sorted_rows(fit_rows)
    eval_rows = _sorted_rows(eval_rows)
    write_jsonl_gz(out_dir / "fit_rows.jsonl.gz", fit_rows)
    write_jsonl_gz(out_dir / "eval_rows.jsonl.gz", eval_rows)
    fit_sha = sha256_file(out_dir / "fit_rows.jsonl.gz")
    eval_sha = sha256_file(out_dir / "eval_rows.jsonl.gz")

    # copy calibration.json into the bundle so the standalone checker is self-contained
    (out_dir / "calibration.json").write_bytes(Path(calibration_path).read_bytes())

    mc_items = sorted({r["item_id"] for r in eval_rows if r["format"] == "MC"})
    qa_items = sorted({r["item_id"] for r in eval_rows if r["format"] == "QA"})
    mc_coverage = {"eval_mc_items": len(mc_items), "eval_qa_items": len(qa_items),
                   "paired": mc_items == qa_items}
    mc_retention = {"fit_rows": len(fit_rows), "eval_rows": len(eval_rows)}

    identity = adapter_identity(
        source_manifest_id=source_manifest_id, raw_input_bundle_id=raw_input_bundle_id,
        model_snapshot_id=model_snapshot_id,
        scoring_spec={"model_id": MODEL_ID, "mc": "max_cosine_over_options",
                      "qa": "cosine_to_answer_primary", "round_decimals": _ROUND},
        fit_split="val", eval_split="test", schema_columns=ADAPTER_SCHEMA_COLUMNS,
        fit_row_count=len(fit_rows), eval_row_count=len(eval_rows),
        fit_rows_sha256=fit_sha, eval_rows_sha256=eval_sha,
        mc_coverage=mc_coverage, mc_retention=mc_retention, producer_hashes=producer_hashes,
    )
    manifest = build_manifest(identity)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    (out_dir / "adapter_build_record.json").write_text(
        json.dumps({"fit_rows": len(fit_rows), "eval_rows": len(eval_rows),
                    "val_items": len(val_qids), "test_items": len(test_qids)},
                   indent=2, sort_keys=True), encoding="utf-8")
    return manifest
