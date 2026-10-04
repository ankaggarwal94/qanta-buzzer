"""Join verified questionless option scores to the frozen evaluator labels."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import beta

from scripts.analyze_acl_paired_prompt_scores import load_public, validate_rows

GOLD_SHA = "93a4eec6e9792e432f96b55e88a527765b812635f3c11a70f94e8361095a8162"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def accuracy_interval(correct: int, n: int) -> list[float]:
    """Return the two-sided 95% Clopper-Pearson binomial interval."""
    if not 0 <= correct <= n or n <= 0:
        raise ValueError("invalid binomial counts")
    return [0.0 if correct == 0 else float(beta.ppf(.025, correct, n-correct+1)),
            1.0 if correct == n else float(beta.ppf(.975, correct+1, n-correct))]


def summarize(rows: list[dict]) -> dict:
    """Summarize one model/menu/prompt/split cell, one row per question."""
    if not rows or len({r["qid"] for r in rows}) != len(rows):
        raise ValueError("expected one nonempty observation per question")
    n, correct = len(rows), sum(r["correct"] for r in rows)
    return {"n_questions": n, "n_correct": correct, "accuracy": correct/n,
            "accuracy_ci95_binomial": accuracy_interval(correct, n),
            "uniform_random_choice_accuracy": .25,
            "mean_gold_probability": float(np.mean([r["gold_probability"] for r in rows])),
            "mean_max_probability": float(np.mean([r["max_probability"] for r in rows])),
            "brier_top_correctness": float(np.mean([(r["max_probability"]-r["correct"])**2 for r in rows])),
            "predicted_position_counts": dict(Counter(r["top"] for r in rows)),
            "gold_position_counts": dict(Counter(r["gold"] for r in rows))}


def run(public: Path, gold_path: Path, root: Path, out: Path) -> dict:
    if digest(gold_path) != GOLD_SHA:
        raise ValueError("frozen gold hash mismatch")
    gold = json.loads(gold_path.read_text())
    jobs = load_public(public)
    if set(gold) != {j["job_id"] for j in jobs} or any(v not in {"A", "B", "C", "D"} for v in gold.values()):
        raise ValueError("gold/public coverage mismatch")
    derived, cells, files = [], {}, {}
    for model in ("qwen3b", "qwen7b"):
        path = root/model/"scores.jsonl"
        scores = [json.loads(line) for line in path.read_text().splitlines()]
        validate_rows(scores, jobs)
        files[model] = digest(path)
        for r in scores:
            g = gold[r["job_id"]]
            derived.append({"model": model, **{k:r[k] for k in ("qid", "job_id", "split", "condition", "prompt_condition")},
                            "gold":g, "top":r["top_option_id"], "correct":r["top_option_id"]==g,
                            "gold_probability":r["conditional_option_probabilities"][g],
                            "max_probability":max(r["conditional_option_probabilities"].values())})
    for model in ("qwen3b", "qwen7b"):
        for menu in ("independent_pool", "same_category_pool"):
            for prompt in ("original", "forced"):
                for split in ("calibration", "selection", "test", "all"):
                    rows = [r for r in derived if r["model"]==model and r["condition"]==menu and
                            r["prompt_condition"]==prompt and (split=="all" or r["split"]==split)]
                    cells[f"{model}/{menu}/{prompt}/{split}"] = summarize(rows)
    report = {"schema":"imcqa-questionless-gold-analysis-v1", "status":"complete", "rows":len(derived),
              "scope":"Descriptive next-token A-D conditional scores; no question text, generated response frequencies, or abstention probabilities.",
              "interval_scope":"Per-cell question-independent binomial intervals; no multiplicity correction, no generalization across menus or model families.",
              "inputs_sha256":{"public":digest(public), "gold":digest(gold_path), **files}, "cells":cells}
    out.mkdir(parents=True, exist_ok=True)
    (out/"menu_control_report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    with (out/"menu_control_rows.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(derived[0])); w.writeheader(); w.writerows(derived)
    return report


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("public", "gold", "scores-root", "out"):
        parser.add_argument("--"+name,required=True,type=Path)
    a=parser.parse_args(); r=run(a.public,a.gold,a.scores_root,a.out)
    print(json.dumps({"status":r["status"],"rows":r["rows"]}))


if __name__ == "__main__":
    main()
