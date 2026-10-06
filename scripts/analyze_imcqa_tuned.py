#!/usr/bin/env python3
"""Replay independently selected policies on a complete fresh plain-score run.

There is no policy selection in this module. Its public-input, policy-lock and
worker receipts must agree before any gold joins or result calculation.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

from scripts import analyze_imcqa_fixed_abstention as old
from scripts import analyze_imcqa_protocol_pilot as protocol
from scripts import select_imcqa_tuned_policies as selection

MENUS = selection.MENUS
POLICIES = old.POLICIES


def policy_map(lock):
    """Require exactly one frozen choice per family and menu."""
    if lock.get("schema_version") != "imcqa-independently-tuned-policies-v1" or lock.get("status") != "frozen_for_fresh_evaluation":
        raise ValueError("complete frozen policy lock required")
    result = {}
    for row in lock["policies"]:
        key = row["condition"], row["family"]
        if key in result:
            raise ValueError("duplicate selected policy")
        if row["always_pass"]:
            if row["threshold"] is not None or row["fixed_round"] is not None:
                raise ValueError("PASS must have no threshold or round")
        elif row["threshold"] not in selection.GRID:
            raise ValueError("threshold outside frozen candidate grid")
        elif row["family"] == "fixed_selective" and row["fixed_round"] not in range(1, 6):
            raise ValueError("invalid selected fixed round")
        elif row["family"] == "adaptive_selective" and row["fixed_round"] is not None:
            raise ValueError("adaptive policy cannot have a fixed round")
        result[key] = row
    if set(result) != {(m, f) for m in MENUS for f in ("fixed_selective", "adaptive_selective")}:
        raise ValueError("missing or extra selected policy")
    return result


def build_episodes(views, lock):
    """Apply separate selected thresholds to the same candidate trajectories."""
    policies = policy_map(lock)
    trajectories = defaultdict(list)
    for row in views:
        if row["arm"] != "plain":
            raise ValueError("only plain answer states permitted")
        trajectories[row["qid"], row["condition"], row["rotation"]].append(row)
    qids = {r["qid"] for r in views}
    if not qids or set(trajectories) != {(q, m, r) for q in qids for m in MENUS for r in range(4)}:
        raise ValueError("incomplete question/menu/rotation coverage")
    if qids & (set(lock["selection_qids"]) | set(lock["calibration_qids"])):
        raise ValueError("development question in fresh evaluation")
    question_groups = defaultdict(set)
    episodes = []
    for (qid, menu, rotation), states in sorted(trajectories.items()):
        states.sort(key=lambda row: row["round"])
        if [s["round"] for s in states] != list(range(1, 6)):
            raise ValueError("five distinct sequential rounds required")
        question_groups[qid].update(s["group_id"] for s in states)
        fs, adaptive = policies[menu, "fixed_selective"], policies[menu, "adaptive_selective"]
        probs = [old.transfer.calibrated_probability(max(s["canonical_answer_probabilities"].values()), lock["calibrators"][menu]) for s in states]
        crossing = None if adaptive["always_pass"] else next((i + 1 for i, p in enumerate(probs) if p >= adaptive["threshold"]), None)
        # If the selected fixed family is PASS, its forced control is final-round
        # answering by prespecification. This convention never changes FS or AS.
        fixed_round = 5 if fs["always_pass"] else fs["fixed_round"]
        rounds = {"fixed_forced": fixed_round,
                  "fixed_selective": fixed_round if not fs["always_pass"] and probs[fixed_round - 1] >= fs["threshold"] else None,
                  "adaptive_forced": crossing or 5, "adaptive_selective": crossing}
        for name, rnd in rounds.items():
            state = states[(rnd or 5) - 1]
            committed = rnd is not None
            correct = committed and bool(state["candidate_correct"])
            episodes.append({"qid": qid, "group_id": state["group_id"], "condition": menu,
                "rotation": rotation, "policy": name, "round": rnd,
                "observed_round": rnd or 5, "committed": committed, "correct": correct,
                "wrong": committed and not correct, "terminal_pass": not committed,
                "canonical_choice": state["candidate_choice"] if committed else None,
                "answer_score_id": state["score_id"] if committed else None,
                "reward": (float(selection.REWARDS[rnd - 1]) if correct else -1.) if committed else 0.})
    if any(len(groups) != 1 for groups in question_groups.values()) or len({next(iter(g)) for g in question_groups.values()}) != len(qids):
        raise ValueError("require one unique independent group per question")
    prior_groups = set(lock["calibration_group_ids"]) | set(lock["selection_group_ids"])
    if prior_groups & {next(iter(g)) for g in question_groups.values()}:
        raise ValueError("development or calibration group in fresh evaluation")
    return episodes


def summarize(episodes):
    """Use the frozen two-menu paired question bootstrap; controls secondary."""
    result, per_question = old.summarize(episodes, samples=20000, seed=1)
    primary = []
    for row in result["primary_contrasts"]:
        primary.append({**row, "contrast": "independently_tuned_adaptive_minus_fixed",
            "interval_scope": "Two prespecified menu contrasts; 97.5% percentile intervals, Bonferroni familywise alpha .05.",
            "minimum_worthwhile_gain": .05,
            "positive_difference_supported": row["ci97_5"][0] > 0,
            "worthwhile_difference_supported": row["ci97_5"][0] > .05})
    return {"schema_version": "imcqa-tuned-fresh-analysis-v1",
        "n_questions": result["n_questions"], "n_policy_episodes": len(episodes),
        "bootstrap_samples": 20000, "bootstrap_seed": 1,
        "unit": result["unit"], "policy_summaries": result["policy_summaries"],
        "primary_contrasts": primary,
        "scope": "Frozen policy comparison on prepared fresh questions; no optimality, risk guarantee, or pure timing causal claim. Secondary metric intervals are descriptive."}, per_question


def load_views(public, evaluator, run, lock):
    """Reject incomplete, mismatched, or numerically unvalidated inference."""
    from scripts.imcqa_tuned_design import validate_public_package
    from scripts.imcqa_protocol_scoring import validate_rows
    package = selection.read_json(public)
    jobs = validate_public_package(package)
    if package["policy_lock_sha256"] != selection.sha256(lock):
        raise ValueError("policy lock differs from pre-inference public package")
    if package["main_dataset_sha256"] != selection.sha256(evaluator):
        raise ValueError("evaluator differs from pre-inference public package")
    receipt = selection.read_json(run / "receipt.json")
    score_path = run / "scores.jsonl"
    if receipt.get("status") != "complete" or receipt.get("public_input_sha256") != selection.sha256(public) or receipt.get("scores_sha256") != selection.sha256(score_path):
        raise ValueError("missing complete hash-bound worker receipt")
    raw = [json.loads(line) for line in score_path.read_text().splitlines()]
    lookup = {job["score_id"]: job for job in jobs}
    if len(raw) != len(jobs) or {row["score_id"] for row in raw} != set(lookup):
        raise ValueError("score identity coverage differs")
    validate_rows([lookup[r["score_id"]] for r in raw], raw, complete=True)
    validate_numerics(run, package, raw, receipt)
    dataset = selection.read_json(evaluator)
    questions = {q["qid"]: q for q in dataset["questions"]}
    if len(questions) != len(dataset["questions"]) or set(questions) != {j["qid"] for j in jobs}:
        raise ValueError("evaluator coverage differs")
    views = []
    for row in raw:
        job, question = lookup[row["score_id"]], questions[row["qid"]]
        if question["group_id"] != job["group_id"] or question["split"] != "test":
            raise ValueError("evaluator group differs")
        menus = {m["condition"]: m for m in question["menus"]}
        prefixes = {p["prefix_id"]: p for p in question["prefixes"]}
        if len(menus) != 2 or len(question["menus"]) != 2 or set(menus) != set(MENUS) or len(prefixes) != len(question["prefixes"]):
            raise ValueError("duplicate or missing evaluator menu/prefix")
        menu, prefix = menus[job["condition"]], prefixes[job["prefix_id"]]
        if menu["gold_option_id"] not in "ABCD" or len(menu["gold_option_id"]) != 1 or menu["menu_id"] != job["menu_id"] or prefix["fraction"] != job["fraction"]:
            raise ValueError("gold/menu/prefix identity differs")
        options = {o["id"]: o["text"] for o in menu["options"]}
        if len(menu["options"]) != 4 or set(options) != set("ABCD"):
            raise ValueError("evaluator option identities differ")
        displayed = [{"id": label, "text": options[identity]} for label, identity in job["option_source_ids"].items()]
        expected = protocol.expected_prompt(prefix["text"], displayed, job["round"], "plain", "E")
        if expected != job["prompt"]:
            raise ValueError("public prompt differs from evaluator prefix/options")
        views.append(protocol.score_view(row, {**job, "canonical_gold_option_id": menu["gold_option_id"]}))
    return views


def validate_numerics(run, package, rows, receipt):
    """Recompute saved numerical comparisons and bind them to production rows."""
    from scripts import imcqa_tuned_scoring as scoring
    from scripts import acl_option_scoring as base
    n = len(package["jobs"])
    if any(receipt.get(k) != v for k, v in {"protocol": scoring.PROTOCOL, "model_tag": "qwen7b",
            "expected_rows": n, "completed_rows": n, "total_contexts": n, "batch_size": 8,
            "cached": True, "generation": False, "sampling": False, "reused_rows": 0}.items()):
        raise ValueError("worker execution contract differs")
    metadata = selection.read_json(run / "metadata.json")
    required = {"torch": "2.6.0", "transformers": "4.51.3", "tokenizers": "0.21.1",
                "safetensors": "0.5.3", "huggingface-hub": "0.30.2"}
    if (metadata.get("model") != base.MODELS["qwen7b"] or
            metadata.get("revision") != base.PINNED_MODELS[base.MODELS["qwen7b"]] or
            metadata.get("dtype") != "float32" or metadata.get("loaded_dtype") != "bfloat16" or
            metadata.get("attention") != "eager" or metadata.get("tf32") is not False or
            metadata.get("public_input_sha256") != receipt["public_input_sha256"] or
            metadata.get("source_commit") != receipt.get("source_commit") or
            any(metadata.get("versions", {}).get(k, "").split("+")[0] != v for k, v in required.items())):
        raise ValueError("pinned model or numerical stack differs")
    from scripts.modal_acl_paired_prompt_scores import verify_prepare
    cache = verify_prepare((run.parent / "cache_prepare_receipt.json").read_bytes())
    if metadata.get("model_files_sha256") != cache["model_receipts"]["qwen7b"]["model_files_sha256"]:
        raise ValueError("model files differ from original cached-model receipt")
    source_root = Path(__file__).resolve().parents[1]
    if metadata.get("source_files_sha256") != {name: base.file_hash(source_root / name) for name in scoring.SOURCE_FILES}:
        raise ValueError("analysis checkout differs from scored numerical sources")
    jobs = {j["score_id"]: j for j in package["jobs"]}
    production = {r["score_id"]: {"logits": [r["raw_action_logits"][k] for k in "ABCDE"]} for r in rows}
    def check(record, reference, comparisons):
        ids = record["score_ids"]
        if not ids or len(set(ids)) != len(ids) or not set(ids) <= set(jobs):
            raise ValueError("invalid numerical diagnostic identities")
        targets = [jobs[i] for i in ids]
        scoring.numeric_agreement([production[i] for i in ids], record[reference], targets)
        for name in comparisons:
            scoring.numeric_agreement(record[reference], record[name], targets)
    attempts = run / "attempts"
    diagnostic = selection.read_json(attempts / "000_diagnostics.json")
    ordered_jobs = [jobs[r["score_id"]] for r in rows]
    expected_diagnostic_ids = [rows[i]["score_id"] for i in scoring.diagnostic_indices(ordered_jobs, rows)]
    if diagnostic["score_ids"] != expected_diagnostic_ids:
        raise ValueError("initial numerical diagnostic coverage differs")
    check(diagnostic, "cached", ("uncached", "singles", "permuted_aligned", "replay"))
    if diagnostic["cached"] != diagnostic["replay"]:
        raise ValueError("diagnostic exact replay failed")
    record = selection.read_json(attempts / "000_production_diagnostics.json")
    offsets = scoring.production_offsets(n)
    expected_ids = [rows[i]["score_id"] for offset in offsets for i in range(offset, offset + 8)]
    if record["offsets"] != offsets or record["score_ids"] != expected_ids or record.get("actual_production_batches") is not True:
        raise ValueError("production numerical coverage differs")
    check(record, "production", ("single_reference", "diagnostic"))
    if record["production"] != record["diagnostic"]:
        raise ValueError("production exact replay failed")
    live = selection.read_json(attempts / "000_live_trajectories.json")
    scoring.validate_replay_evidence(live)
    if live["qids"] != scoring.replay_qids(package["jobs"]):
        raise ValueError("live diagnostic question selection differs")
    for i, row in enumerate(live["rows"]):
        side = selection.read_json(attempts / f"000_live_{i:03d}_raw.json")
        if side["score_ids"] != [row["score_id"]] or any(row[k] != jobs[row["score_id"]][k] for k in ("qid", "condition", "rotation", "round")):
            raise ValueError("live diagnostic identity differs")
        check(side, "production", ("reference",))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("public", "evaluator", "policy-lock", "run", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    views = load_views(args.public, args.evaluator, args.run, args.policy_lock)
    episodes = build_episodes(views, selection.read_json(args.policy_lock))
    result, per_question = summarize(episodes)
    args.out.mkdir(parents=True, exist_ok=False)
    selection.write_json(args.out / "results.json", result)
    selection.write_csv(args.out / "episodes.csv", episodes)
    selection.write_csv(args.out / "per_question.csv", per_question)
    selection.write_json(args.out / "analysis_receipt.json", {"status": "complete",
        "analyzer_sha256": selection.sha256(Path(__file__)),
        "inputs_sha256": {name: selection.sha256(path) for name, path in
            (("public", args.public), ("evaluator", args.evaluator), ("policy_lock", args.policy_lock),
             ("worker_receipt", args.run / "receipt.json"), ("scores", args.run / "scores.jsonl"))},
        "outputs_sha256": {p.name: selection.sha256(p) for p in sorted(args.out.iterdir())}})
    print(json.dumps(result["primary_contrasts"]))


if __name__ == "__main__":
    main()
