#!/usr/bin/env python3
"""Fail-closed evaluation replay of a frozen model-specific Mistral policy lock.

This module never fits a calibrator or selects a policy. Production entrypoints
require complete numerical evidence and offline native-tokenizer reconstruction.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np

from scripts import analyze_imcqa_protocol_pilot as protocol
from scripts import analyze_imcqa_tuned as tuned
from scripts import imcqa_mistral_design as design
from scripts import imcqa_mistral_scoring as scoring
from scripts import select_imcqa_mistral_policies as selector
from scripts import select_imcqa_tuned_policies as io

QWEN_EPISODES_SHA256 = "37fd99f07749890c510586beaa9286b78b59a46478acaade2187fefa2672ef13"
QWEN_SOURCE_COMMIT = "8cd0b71150170ec545716b28fa63a07a163bad61"
INTERACTION_SEED = 20261007


def read(path):
    """Read one immutable JSON input."""
    return json.loads(Path(path).read_text())


def validate_token_contexts(jobs, rows, tokenizer):
    """Re-tokenize every original prompt and compare exact saved contexts."""
    if scoring.base.sha(tokenizer.chat_template.encode()) != scoring.CHAT_TEMPLATE_SHA256:
        raise ValueError("native tokenizer template differs")
    lookup = {j["score_id"]: j for j in jobs}
    if len(rows) != len(jobs) or len({r["score_id"] for r in rows}) != len(rows):
        raise ValueError("tokenizer coverage differs")
    for row in rows:
        if row["score_id"] not in lookup:
            raise ValueError("unknown tokenizer score identity")
        context = scoring.prepare_context(tokenizer, lookup[row["score_id"]])
        if any(row.get(key) != value for key, value in context.items()):
            raise ValueError("saved score tokens/context differ from native prompt")
    return {"passed": True, "reconstructed_contexts": len(rows),
            "chat_template_sha256": scoring.CHAT_TEMPLATE_SHA256,
            "action_token_ids": scoring.ACTION_TOKEN_IDS}


def load_tokenizer(directory):
    """Load only locally provided, content-pinned tokenizer artifacts."""
    from transformers import AutoTokenizer
    directory = Path(directory)
    if any(design.file_hash(directory / name) != expected
           for name, expected in scoring.TOKENIZER_HASHES.items()):
        raise ValueError("local tokenizer artifact differs from pinned revision")
    tokenizer = AutoTokenizer.from_pretrained(str(directory), local_files_only=True,
                                            trust_remote_code=False)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def reconstruct_views(package, evaluator, rows):
    """Join CPU-only labels to exactly the scored full 850-question grid."""
    jobs = design.validate_public_package(package)
    if (package["stage"] != "evaluation" or evaluator.get("stage") != "evaluation"
            or design.value_hash(evaluator) != package["main_dataset_sha256"]):
        raise ValueError("evaluation dataset identity differs")
    questions = {q["qid"]: q for q in evaluator["questions"]}
    if (len(questions) != 850 or len(questions) != len(evaluator["questions"])
            or set(questions) != set(package["selection"]["selected_qids"])):
        raise ValueError("850 distinct evaluator questions required")
    lookup = {j["score_id"]: j for j in jobs}
    if len(rows) != len(jobs) or {r["score_id"] for r in rows} != set(lookup):
        raise ValueError("complete distinct evaluation scores required")
    scoring.validate_rows([lookup[r["score_id"]] for r in rows], rows, complete=True)
    result = []
    for row in rows:
        job, question = lookup[row["score_id"]], questions[row["qid"]]
        if question["split"] != "test" or question["group_id"] != job["group_id"]:
            raise ValueError("evaluator role/group differs")
        menus = {m["condition"]: m for m in question["menus"]}
        prefixes = {p["prefix_id"]: p for p in question["prefixes"]}
        if (len(menus) != len(question["menus"]) or set(menus) != set(tuned.MENUS)
                or len(prefixes) != len(question["prefixes"])):
            raise ValueError("duplicate or missing evaluator menu/prefix")
        menu, prefix = menus[job["condition"]], prefixes[job["prefix_id"]]
        options = {o["id"]: o["text"] for o in menu["options"]}
        if (len(menu["options"]) != 4 or set(options) != set("ABCD")
                or menu["gold_option_id"] not in options or menu["menu_id"] != job["menu_id"]
                or prefix["fraction"] != job["fraction"]):
            raise ValueError("evaluator gold/menu/prefix differs")
        displayed = [{"id": label, "text": options[identity]}
                     for label, identity in job["option_source_ids"].items()]
        expected = protocol.expected_prompt(prefix["text"], displayed, job["round"], "plain", "E")
        if expected != job["prompt"]:
            raise ValueError("evaluator gold join differs from scored prompt")
        result.append(protocol.score_view(row, {**job, "canonical_gold_option_id": menu["gold_option_id"]}))
    return result


def load_views(public, evaluator, run, lock, manifest, development_receipt,
               tokenizer_dir, source_commit):
    """Validate every execution/lock binding before interpreting any gold labels."""
    package, frozen, dev_receipt = read(public), read(lock), read(development_receipt)
    design.validate_stage_manifest(read(manifest), package, policy_lock=frozen,
                                   development_receipt=dev_receipt)
    projection = selector.validate_policy_lock(frozen, evaluation_package=package)
    if (design.file_hash(lock) != package["policy_lock_sha256"]
            or design.file_hash(evaluator) != package["main_dataset_sha256"]
            or design.file_hash(development_receipt) != package["development_receipt_sha256"]):
        raise ValueError("on-disk evaluation input binding differs")
    receipt = read(Path(run) / "receipt.json")
    if receipt.get("source_commit") != source_commit:
        raise ValueError("evaluation worker differs from expected committed source")
    numeric = scoring.validate_completed_run(Path(public), Path(run))
    rows = [json.loads(line) for line in (Path(run) / "scores.jsonl").read_text().splitlines()]
    tokens = validate_token_contexts(package["jobs"], rows, load_tokenizer(tokenizer_dir))
    views = reconstruct_views(package, read(evaluator), rows)
    return views, projection, {"numerics": numeric, "native_tokenization": tokens,
                               "policy_lock_projection": "explicit validated model-agnostic schema projection"}


def validate_episode_grid(episodes):
    """Check complete paired rotation coverage and reward/PASS arithmetic."""
    qids = sorted({r["qid"] for r in episodes})
    expected = {(q, m, r, p) for q in qids for m in tuned.MENUS
                for r in range(4) for p in tuned.POLICIES}
    lookup = {}
    groups = defaultdict(set)
    for row in episodes:
        key = row["qid"], row["condition"], row["rotation"], row["policy"]
        if key in lookup:
            raise ValueError("duplicate policy episode")
        lookup[key] = row
        groups[row["qid"]].add(row["group_id"])
        if any(type(row[k]) is not bool for k in ("committed", "correct", "wrong", "terminal_pass")):
            raise ValueError("episode booleans required")
        committed, rnd = row["committed"], row["round"]
        if ((committed and (type(rnd) is not int or rnd not in range(1, 6)))
                or (not committed and rnd is not None)
                or row["terminal_pass"] == committed
                or row["wrong"] != (committed and not row["correct"])
                or (row["correct"] and not committed)
                or row["observed_round"] != (rnd if committed else 5)):
            raise ValueError("episode action/terminal semantics differ")
        reward = (io.REWARDS[rnd - 1] if row["correct"] else -1.) if committed else 0.
        if not np.isfinite(row["reward"]) or abs(row["reward"] - reward) > 1e-12:
            raise ValueError("episode reward arithmetic differs")
    if not qids or set(lookup) != expected:
        raise ValueError("incomplete paired policy/rotation/menu grid")
    if any(len(g) != 1 for g in groups.values()) or len({next(iter(g)) for g in groups.values()}) != len(qids):
        raise ValueError("question/group independence differs")
    return qids, lookup


def supplemental_summaries(episodes, views):
    """Report commitment timing conditional on an answer, with PASS separate."""
    qids, _ = validate_episode_grid(episodes)
    indices = np.random.default_rng(1).integers(0, len(qids), (20000, len(qids)))
    result = []
    for menu in tuned.MENUS:
        for policy in tuned.POLICIES:
            rows = [r for r in episodes if r["condition"] == menu and r["policy"] == policy]
            byq = {q: [r for r in rows if r["qid"] == q] for q in qids}
            totals = np.asarray([[sum(r["round"] or 0 for r in byq[q]),
                                  sum(r["committed"] for r in byq[q])] for q in qids], float)
            answered = int(totals[:, 1].sum())
            draws = totals[indices].sum(axis=1)
            valid = draws[:, 1] > 0
            round_draws = draws[valid, 0] / draws[valid, 1]
            result.append({"condition": menu, "policy": policy,
                "answer_only_mean_round": float(totals[:, 0].sum() / answered) if answered else None,
                "answer_only_mean_round_ci95": np.quantile(round_draws, [.025, .975]).tolist() if len(round_draws) else None,
                "bootstrap_draws_with_answers": int(valid.sum()), "answered_episodes": answered,
                "terminal_pass_episodes": len(rows) - answered,
                "answer_round_counts": {str(t): sum(r["round"] == t for r in rows) for t in range(1, 6)},
                "scope": "Secondary descriptive timing conditional on commitment; PASS excluded from mean."})
    prefix = []
    for menu in tuned.MENUS:
        for rnd in range(1, 6):
            rows = [r for r in views if r["condition"] == menu and r["round"] == rnd]
            if len(rows) != 4 * len(qids):
                raise ValueError("prefix accuracy grid differs")
            prefix.append({"condition": menu, "round": rnd, "n_questions": len(qids),
                "n_states": len(rows), "accuracy": sum(r["candidate_correct"] for r in rows) / len(rows),
                "scope": "Four rotations equally averaged; full-question MC accuracy is round 5."})
    return {"answer_timing": result, "prefix_accuracy": prefix}


def load_qwen_episodes(path):
    """Accept only the previously validated immutable Qwen episode artifact."""
    if design.file_hash(path) != QWEN_EPISODES_SHA256:
        raise ValueError("Qwen comparison differs from frozen validated episodes")
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in ("committed", "correct", "wrong", "terminal_pass"):
            if row[key] not in ("True", "False"):
                raise ValueError("reference boolean differs")
            row[key] = row[key] == "True"
        row["rotation"], row["observed_round"] = int(row["rotation"]), int(row["observed_round"])
        row["round"] = int(row["round"]) if row["round"] else None
        row["reward"] = float(row["reward"])
    validate_episode_grid(rows)
    return rows


def cross_model_interactions(mistral, qwen, *, samples=20000, seed=INTERACTION_SEED):
    """Use shared question draws for exploratory model-by-policy contrasts."""
    qids, new = validate_episode_grid(mistral)
    oldq, previous = validate_episode_grid(qwen)
    if oldq != qids or set(new) != set(previous):
        raise ValueError("cross-model question/rotation pairing differs")
    if any(new[k]["group_id"] != previous[k]["group_id"] for k in new):
        raise ValueError("cross-model question group differs")
    values = np.asarray([[np.mean([new[q, m, r, "adaptive_selective"]["reward"]
                                  - new[q, m, r, "fixed_selective"]["reward"]
                                  - previous[q, m, r, "adaptive_selective"]["reward"]
                                  + previous[q, m, r, "fixed_selective"]["reward"]
                                  for r in range(4)]) for m in tuned.MENUS] for q in qids])
    rng = np.random.default_rng(seed)
    draws = []
    for start in range(0, samples, 256):
        idx = rng.integers(0, len(qids), (min(256, samples-start), len(qids)))
        draws.append(values[idx].mean(axis=1))
    draws = np.concatenate(draws)
    contrasts = [{"condition": m, "contrast": "(Mistral adaptive-fixed) - (Qwen adaptive-fixed)",
                  "mean_delta": float(values[:, i].mean()), "ci95": np.quantile(draws[:, i], [.025, .975]).tolist()}
                 for i, m in enumerate(tuned.MENUS)]
    contrasts.append({"condition": "same_category_minus_independent", "contrast": "menu-by-policy-by-model interaction",
        "mean_delta": float((values[:, 1]-values[:, 0]).mean()),
        "ci95": np.quantile(draws[:, 1]-draws[:, 0], [.025, .975]).tolist()})
    return {"n_questions": len(qids), "bootstrap_samples": samples, "bootstrap_seed": seed,
        "scope": "Exploratory paired 95% intervals without multiplicity adjustment; all rotations, menus, and models paired by question. Policies and calibrators differ by model, so this is a comparison of fitted pipelines, not a pure model causal effect.",
        "qwen_source_commit": QWEN_SOURCE_COMMIT, "qwen_episodes_sha256": QWEN_EPISODES_SHA256,
        "contrasts": contrasts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("public", "evaluator", "run", "policy-lock", "stage-manifest",
                 "development-receipt", "tokenizer-dir", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--qwen-episodes", type=Path)
    args = parser.parse_args()
    views, projection, validation = load_views(args.public, args.evaluator, args.run,
        args.policy_lock, args.stage_manifest, args.development_receipt,
        args.tokenizer_dir, args.source_commit)
    episodes = tuned.build_episodes(views, projection)
    qids, _ = validate_episode_grid(episodes)
    if len(qids) != 850:
        raise ValueError("full 850-question evaluation required")
    result, per_question = tuned.summarize(episodes)
    result.update(schema_version="imcqa-mistral-evaluation-v1", model=design.MODEL,
        scope="Cross-model replication on the existing 850-question cohort; frozen Mistral calibration/policies, no evaluation refit, retuning or semantic input correction. No optimality or pure timing causal claim.",
        **supplemental_summaries(episodes, views))
    if args.qwen_episodes:
        result["cross_model_interactions"] = cross_model_interactions(episodes, load_qwen_episodes(args.qwen_episodes))
    args.out.mkdir(parents=True, exist_ok=False)
    io.write_json(args.out / "results.json", result)
    io.write_csv(args.out / "episodes.csv", episodes)
    io.write_csv(args.out / "per_question.csv", per_question)
    inputs = {name: design.file_hash(getattr(args, name)) for name in
              ("public", "evaluator", "policy_lock", "stage_manifest", "development_receipt")}
    inputs.update(scores=design.file_hash(args.run / "scores.jsonl"),
                  worker_receipt=design.file_hash(args.run / "receipt.json"))
    if args.qwen_episodes:
        inputs["qwen_episodes"] = design.file_hash(args.qwen_episodes)
    io.write_json(args.out / "analysis_receipt.json", {"status": "complete", "model": design.MODEL,
        "analysis_source_sha256": {str(Path(m.__file__).name): design.file_hash(m.__file__)
            for m in (tuned, selector, scoring, design)},
        "analyzer_sha256": design.file_hash(__file__), "inputs_sha256": inputs,
        "source_commit": args.source_commit, "fitting_performed": False, "policy_selection_performed": False,
        "validation": validation, "outputs_sha256": {p.name: design.file_hash(p) for p in sorted(args.out.iterdir())}})
    print(json.dumps(result["primary_contrasts"]))


if __name__ == "__main__":
    main()
