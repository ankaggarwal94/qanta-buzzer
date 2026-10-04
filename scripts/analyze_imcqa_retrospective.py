"""Replay frozen MC prefix generations under explicitly retrospective policies.

No model inference runs here. The original prompts did not describe a sequential
game or its payoff. This analysis cannot establish native WAIT-policy behavior,
an IRT score, broad model-rank validity, or semantic distractor quality.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import tarfile
from typing import Any

import numpy as np
import scipy
import sklearn
from scipy.stats import beta

from evaluation.jane_paired import fit_calibrator


PUBLIC_SHA = "9bfeaf2d86116ced0e8c55c3c390d3be12050ad38820b4e02c6ed684cc8786bf"
DATASET_SHA = "d16d8e611965fba3829f01cda936145743b7d46187030e2138caf52068aa9b62"
MODELS = {
    "qwen3b": ("Qwen/Qwen2.5-3B-Instruct", "aa8e72537993ba99e69dfaafa59ed015b17504d1"),
    "qwen7b": ("Qwen/Qwen2.5-7B-Instruct", "a09a35458c702b33eeacc393d103063234e8bc28"),
}
CONDITIONS = ("independent_pool", "same_category_pool")
SPLITS = ("calibration", "selection", "test")
REWARDS = {
    "early_wrong1": {"correct": "(11-round)/10", "wrong": -1.0, "no_answer": 0.0},
    "early_wrong025": {"correct": "(11-round)/10", "wrong": -.25, "no_answer": 0.0},
    "flat_wrong1": {"correct": "1", "wrong": -1.0, "no_answer": 0.0},
    "flat_wrong025": {"correct": "1", "wrong": -.25, "no_answer": 0.0},
}


def sha(path: Path) -> str:
    """Return the SHA-256 of a file without loading it all into memory."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def unique_json(data: str | bytes) -> Any:
    """Reject duplicate object keys and non-finite JSON constants."""
    def pairs(items):
        d = {}
        for k, v in items:
            if k in d:
                raise ValueError(f"duplicate JSON key: {k}")
            d[k] = v
        return d
    def bad(value):
        raise ValueError(f"non-finite JSON value: {value}")
    return json.loads(data, object_pairs_hook=pairs, parse_constant=bad)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def validate_trajectories(trajectories: dict[str, list[dict]]) -> None:
    """Require complete ten-round trajectories with independent group IDs.

    Parameters
    ----------
    trajectories : dict
        One model and menu condition, indexed by question ID.
    """
    groups = set()
    for qid, rows in trajectories.items():
        if len(rows) != 10 or [r["round"] for r in rows] != list(range(1, 11)):
            raise ValueError(f"missing/duplicate/out-of-order round: {qid}")
        if any(r["qid"] != qid or r["split"] != rows[0]["split"] or
               r["group_id"] != rows[0]["group_id"] for r in rows):
            raise ValueError(f"inconsistent trajectory identity: {qid}")
        if rows[0]["group_id"] in groups:
            raise ValueError("question bootstrap requires unique group IDs")
        groups.add(rows[0]["group_id"])
        fractions = [r["fraction"] for r in rows]
        if fractions[-1] != 1.0 or any(a >= b for a, b in zip(fractions, fractions[1:])):
            raise ValueError(f"prefix fractions not strictly increasing: {qid}")


def load_evidence(public: Path, dataset_path: Path, archive: Path) -> tuple[dict, dict]:
    """Verify frozen inputs and join complete MC generations to gold options.

    Parameters
    ----------
    public, dataset_path, archive : Path
        Exact historical public jobs, evaluator dataset, and raw trace archive.

    Returns
    -------
    trajectories, audit : tuple of dict
        Model/menu/question rows and independently checked coverage/provenance.
    """
    if sha(public) != PUBLIC_SHA or sha(dataset_path) != DATASET_SHA:
        raise ValueError("frozen public jobs or evaluator dataset SHA-256 changed")
    jobs = unique_json(public.read_bytes())["jobs"]
    jm = {j["job_id"]: j for j in jobs}
    dataset = unique_json(dataset_path.read_bytes())
    questions = {q["qid"]: q for q in dataset["questions"]}
    if len(jobs) != 150000 or len(jm) != len(jobs) or len(questions) != 5000:
        raise ValueError("frozen input coverage mismatch")
    gold = {}
    for qid, q in questions.items():
        if len(q["menus"]) != 2 or len(q["prefixes"]) != 10:
            raise ValueError("frozen menu/prefix count mismatch")
        for menu in q["menus"]:
            if menu["condition"] not in CONDITIONS or menu["gold_option_id"] not in "ABCD":
                raise ValueError("unknown MC condition or gold option")
            gold[qid, menu["condition"]] = menu
    for job in jobs:
        q = questions[job["qid"]]
        if any(job[key] != q[key] for key in ("qid", "group_id", "split")):
            raise ValueError("public/evaluator identity mismatch")
        if hashlib.sha256(job["prompt"].encode()).hexdigest() != job["prompt_sha256"]:
            raise ValueError("public prompt hash mismatch")
        if job["format"] == "mc":
            payload = unique_json(job["prompt"].split("\n\n", 1)[1])
            if payload["options"] != gold[job["qid"], job["condition"]]["options"]:
                raise ValueError("public/gold menu text or order mismatch")
    rows = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    seen = {tag: set() for tag in MODELS}
    trace_hashes, completions, metadata = {}, {}, {}
    outcomes = Counter()
    # Streaming read avoids extracting untrusted archive paths or large traces.
    with tarfile.open(archive, "r|gz") as tar:
        member_names = set()
        for member in tar:
            if not member.isfile():
                continue
            name = member.name
            if name in member_names:
                raise ValueError(f"duplicate archive member: {name}")
            member_names.add(name)
            if "/main/" not in name:
                continue
            parts = name.split("/")
            tag = parts[3]
            if tag not in MODELS:
                raise ValueError("unexpected model in raw archive")
            if name.endswith("/completion.json"):
                completions[name.rsplit("/", 1)[0]] = unique_json(tar.extractfile(member).read())
                continue
            if not name.endswith("/trace.json"):
                continue
            blob = tar.extractfile(member).read()
            prefix = name.rsplit("/", 1)[0]
            trace_hashes[prefix] = hashlib.sha256(blob).hexdigest()
            trace = unique_json(blob)
            meta = trace["metadata"]
            if (meta["model"], meta["revision"]) != MODELS[tag]:
                raise ValueError("pinned model identity mismatch")
            if (meta["context_policy"] != "fresh_per_prefix" or not meta["greedy"] or
                    meta["dtype"] != "bfloat16" or meta["execution"] != "actual_cuda_model_generation"):
                raise ValueError("original generation protocol mismatch")
            identity = {k: meta[k] for k in ("model", "revision", "dtype", "greedy", "context_policy",
                        "confidence_method", "model_files_sha256", "max_input_tokens", "max_new_tokens")}
            if tag in metadata and metadata[tag] != identity:
                raise ValueError("model metadata changed between shards")
            metadata[tag] = identity
            for pred in trace["predictions"]:
                job_id = pred["job_id"]
                if job_id in seen[tag] or job_id not in jm:
                    raise ValueError("duplicate/unknown main generation job")
                seen[tag].add(job_id)
                job = jm[job_id]
                if pred["prompt_sha256"] != job["prompt_sha256"]:
                    raise ValueError("generation prompt hash mismatch")
                outcomes[tag, job["condition"], pred["status"]] += 1
                if job["format"] != "mc":
                    continue
                status, answer, confidence = pred["status"], pred["answer"], pred["confidence"]
                if status == "answer":
                    if answer not in {"A", "B", "C", "D"} or not isinstance(confidence, (int, float)) or isinstance(confidence, bool) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
                        raise ValueError("invalid MC answer/confidence")
                elif status in {"abstain", "invalid"}:
                    if answer is not None or confidence is not None:
                        raise ValueError("non-answer has answer/confidence")
                else:
                    raise ValueError("unknown response status")
                if status != "invalid":
                    parsed = unique_json(pred["raw_response"])
                    if parsed != {"answer": answer, "confidence": confidence, "status": status}:
                        raise ValueError("raw/parsed response mismatch")
                expected = gold[job["qid"], job["condition"]]["gold_option_id"]
                row = {k: job[k] for k in ("job_id", "qid", "group_id", "split", "condition", "prefix_id", "fraction")}
                row.update(round=int(job["prefix_id"][1:]), status=status, answer=answer,
                           confidence=confidence, gold_option=expected,
                           correct=status == "answer" and answer == expected)
                rows[tag][job["condition"]][job["qid"]].append(row)
    if set(completions) != set(trace_hashes):
        raise ValueError("main trace/completion coverage mismatch")
    for name, digest in trace_hashes.items():
        if digest != completions[name]["trace_sha256"]:
            raise ValueError("main trace hash differs from completion receipt")
    for tag in MODELS:
        if seen[tag] != set(jm):
            raise ValueError(f"incomplete main job coverage: {tag}")
        for condition in CONDITIONS:
            if set(rows[tag][condition]) != set(questions):
                raise ValueError("incomplete MC question coverage")
            for trajectory in rows[tag][condition].values():
                trajectory.sort(key=lambda r: r["round"])
            validate_trajectories(rows[tag][condition])
    audit = {"main_rows_per_model": {tag: len(s) for tag, s in seen.items()},
             "mc_rows_total": 200000, "questions": 5000, "main_shards_verified": len(trace_hashes),
             "question_splits": dict(Counter(q["split"] for q in questions.values())),
             "outcome_counts": {"/".join(k): v for k, v in sorted(outcomes.items())},
             "models": metadata, "trace_hashes": trace_hashes,
             "inputs_sha256": {str(p.resolve()): sha(p) for p in (public, dataset_path, archive)}}
    return rows, audit


def policy_records(trajectories: dict[str, list[dict]], policy: dict, calibrator: dict | None) -> list[dict]:
    """Select one original answer per question without using correctness.

    Parameters
    ----------
    trajectories : dict
        Complete question trajectories from one model/menu cell.
    policy : dict
        First answer, fixed round, calibrated threshold, or never.
    calibrator : dict or None
        Frozen interpolation knots needed only by a threshold policy.

    Returns
    -------
    list of dict
        Question-sorted decisions; no-commit records retain null positions.
    """
    calibrator = policy.get("frozen_historical_calibrator", calibrator)
    records = []
    for qid in sorted(trajectories):
        rows, chosen = trajectories[qid], None
        if policy["kind"] == "first_answer":
            chosen = next((r for r in rows if r["status"] == "answer"), None)
        elif policy["kind"] == "fixed_round":
            r = rows[policy["round"] - 1]
            chosen = r if r["status"] == "answer" else None
        elif policy["kind"] == "threshold":
            chosen = next((r for r in rows if r["status"] == "answer" and
                           float(np.interp(r["confidence"], calibrator["x"], calibrator["y"])) >= policy["value"]), None)
        elif policy["kind"] != "never":
            raise ValueError("unknown replay policy")
        records.append({"qid": qid, "split": rows[0]["split"], "committed": chosen is not None,
                        "correct": bool(chosen and chosen["correct"]),
                        **{key: chosen[key] if chosen else None for key in
                           ("job_id", "round", "fraction", "answer", "confidence") if key in rows[0]}})
    return records


def reward_values(records: list[dict], name: str) -> np.ndarray:
    """Return per-question reward, including zero for no commitment."""
    spec = REWARDS[name]
    return np.asarray([0.0 if not r["committed"] else
                       ((11 - r["round"]) / 10 if spec["correct"] != "1" else 1.0)
                       if r["correct"] else spec["wrong"] for r in records])


def select_policies(trajectories: dict[str, list[dict]], calibrator: dict) -> dict:
    """Choose threshold and fixed-round policies on selection questions only.

    The primary objective is mean early_wrong1 reward. Ties prefer lower error
    incidence, then greater coverage, then the higher confidence threshold.
    Sensitivity rewards never retune this selection.
    """
    if any(r["split"] != "selection" for rows in trajectories.values() for r in rows):
        raise ValueError("policy fitting accepts selection trajectories only")
    values = sorted({float(np.interp(r["confidence"], calibrator["x"], calibrator["y"]))
                     for rows in trajectories.values() for r in rows if r["status"] == "answer"})
    candidates = [{"kind": "never"}] + [{"kind": "threshold", "value": v} for v in values]
    audit = []
    for p in candidates:
        recs = policy_records(trajectories, p, calibrator)
        reward = float(reward_values(recs, "early_wrong1").mean())
        errors = sum(r["committed"] and not r["correct"] for r in recs)
        commits = sum(r["committed"] for r in recs)
        audit.append({"policy": p, "reward": reward, "errors": errors, "commits": commits})
    def key(row):
        return (row["reward"], -row["errors"], row["commits"], row["policy"].get("value", 2.0))
    best = max(audit, key=key)
    feasible = [r for r in audit if not r["commits"] or r["errors"] / r["commits"] <= .1]
    risk_best = max(feasible, key=key)
    fixed = []
    for n in range(1, 11):
        p = {"kind": "fixed_round", "round": n}
        records = policy_records(trajectories, p, calibrator)
        fixed.append((float(reward_values(records, "early_wrong1").mean()), -n, p))
    return {"confidence_reward_selected": best["policy"],
            "confidence_risk10_selected": risk_best["policy"],
            "fixed_round_selected": max(fixed, key=lambda x: x[:2])[2],
            "selection_audit": {"fit_split": "selection", "objective": "early_wrong1",
                                "threshold_candidates": audit, "risk10_rule": "empirical selection error/commit <=0.1; no population guarantee",
                                "fixed_round_rewards": {str(-r[1]): r[0] for r in fixed}}}


def interval(values: np.ndarray) -> dict:
    """Summarize finite bootstrap statistics without imputing undefined ratios."""
    valid = values[np.isfinite(values)]
    return {"ci95": np.quantile(valid, [.025, .975]).tolist() if len(valid) else None,
            "defined_resamples": len(valid), "total_resamples": len(values)}


def summarize_policy(records: list[dict], indices: np.ndarray | None = None) -> dict:
    """Compute unconditional reward/accuracy and answer-conditional risk."""
    committed = np.asarray([r["committed"] for r in records], dtype=float)
    correct = np.asarray([r["correct"] for r in records], dtype=float)
    errors = committed - correct
    rounds = np.asarray([r["round"] or 0 for r in records], dtype=float)
    fraction = np.asarray([r["fraction"] or 0 for r in records], dtype=float)
    n, nc = len(records), int(committed.sum())
    result = {"n_questions": n, "n_committed": nc, "n_correct": int(correct.sum()),
              "n_wrong": int(errors.sum()), "coverage": float(committed.mean()),
              "correct_per_question": float(correct.mean()),
              "risk": float(errors.sum() / nc) if nc else None,
              "mean_round_when_committed": float(rounds.sum() / nc) if nc else None,
              "mean_fraction_when_committed": float(fraction.sum() / nc) if nc else None,
              "rewards": {name: float(reward_values(records, name).mean()) for name in REWARDS}}
    result["risk_clopper_pearson95"] = (
        [float(beta.ppf(.025, int(errors.sum()), nc-int(errors.sum())+1)) if errors.sum() else 0.0,
         float(beta.ppf(.975, int(errors.sum())+1, nc-int(errors.sum()))) if errors.sum() < nc else 1.0]
        if nc else None)
    if indices is not None:
        counts = committed[indices].sum(axis=1)
        def ratio(a):
            return np.divide(a[indices].sum(axis=1), counts, out=np.full(len(indices), np.nan), where=counts > 0)
        result["bootstrap"] = {"coverage": interval(committed[indices].mean(axis=1)),
                               "correct_per_question": interval(correct[indices].mean(axis=1)),
                               "risk": interval(ratio(errors)),
                               "mean_round_when_committed": interval(ratio(rounds)),
                               "mean_fraction_when_committed": interval(ratio(fraction)),
                               **{f"reward_{name}": interval(reward_values(records, name)[indices].mean(axis=1)) for name in REWARDS}}
    return result


def paired_difference(left: list[dict], right: list[dict], indices: np.ndarray) -> dict:
    """Bootstrap matched question differences, explicitly left minus right."""
    if [r["qid"] for r in left] != [r["qid"] for r in right]:
        raise ValueError("paired difference requires identical question order")
    vectors = {f"reward_{name}_difference": reward_values(left, name) - reward_values(right, name) for name in REWARDS}
    vectors["correct_per_question_difference"] = np.asarray([float(l["correct"]) - float(r["correct"]) for l, r in zip(left, right)])
    vectors["coverage_difference"] = np.asarray([float(l["committed"]) - float(r["committed"]) for l, r in zip(left, right)])
    return {name: {"mean": float(v.mean()), **interval(v[indices].mean(axis=1))} for name, v in vectors.items()}


def trajectory_diagnostics(trajectories: dict[str, list[dict]]) -> dict:
    """Distinguish loss of correctness from transitions to a wrong answer."""
    qlaterwrong = qlost = adjacentwrong = adjacentlost = 0
    answerchanges = 0
    for rows in trajectories.values():
        seen_correct = laterwrong = lost = False
        for r in rows:
            laterwrong |= seen_correct and r["status"] == "answer" and not r["correct"]
            lost |= seen_correct and not r["correct"]
            seen_correct |= r["correct"]
        qlaterwrong += laterwrong
        qlost += lost
        for l, r in zip(rows, rows[1:]):
            adjacentwrong += l["correct"] and r["status"] == "answer" and not r["correct"]
            adjacentlost += l["correct"] and not r["correct"]
            answerchanges += l["status"] == r["status"] == "answer" and l.get("answer") != r.get("answer")
    n = len(trajectories)
    return {"n_questions": n, "n_adjacent_pairs": 9*n,
            "questions_correct_then_later_wrong_answer": qlaterwrong,
            "questions_correct_then_later_noncorrect": qlost,
            "adjacent_correct_to_wrong_answer": adjacentwrong,
            "adjacent_correct_to_noncorrect": adjacentlost,
            "adjacent_answer_changes_among_two_answered_rounds": answerchanges}


def round_summary(rows: list[dict], calibrator: dict) -> dict:
    answered = [r for r in rows if r["status"] == "answer"]
    raw = np.asarray([r["confidence"] for r in answered])
    y = np.asarray([r["correct"] for r in answered], dtype=float)
    calibrated = np.interp(raw, calibrator["x"], calibrator["y"])
    return {"n_questions": len(rows), "n_answered": len(answered),
            "n_abstain": sum(r["status"] == "abstain" for r in rows),
            "n_invalid": sum(r["status"] == "invalid" for r in rows),
            "accuracy_all_questions": sum(r["correct"] for r in rows) / len(rows),
            "accuracy_when_answered": float(y.mean()) if len(y) else None,
            "mean_self_report_confidence_when_answered": float(raw.mean()) if len(y) else None,
            "brier_self_report_when_answered": float(np.mean((raw-y)**2)) if len(y) else None,
            "brier_calibrated_when_answered": float(np.mean((calibrated-y)**2)) if len(y) else None}


def analyze(rows: dict, *, samples: int = 2000, seed: int = 1, historical_reports: dict | None = None) -> tuple[dict, list[dict], list[dict]]:
    """Fit on development splits, evaluate all fixed policies, and compare test.

    Parameters
    ----------
    rows : dict
        Model/menu/question trajectories verified by ``load_evidence``.
    samples, seed : int
        Question bootstrap resamples and explicit random seed.

    Returns
    -------
    report, round_rows, decision_rows : tuple
        JSON report and exportable per-round/per-question tables.
    """
    report = {"cells": {}, "test_comparisons": {}}
    round_rows, decision_rows, test_records = [], [], {}
    reference_test = sorted(q for q, tr in rows["qwen3b"][CONDITIONS[0]].items() if tr[0]["split"] == "test")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(reference_test), size=(samples, len(reference_test)), dtype=np.int32)
    for tag in MODELS:
        for condition in CONDITIONS:
            cellname = f"{tag}/{condition}"
            trajectories = rows[tag][condition]
            split_rows = {s: {q: tr for q, tr in trajectories.items() if tr[0]["split"] == s} for s in SPLITS}
            if sorted(split_rows["test"]) != reference_test:
                raise ValueError("test pairing differs by model/menu")
            calibrator = fit_calibrator(split_rows["calibration"].values())
            selected = select_policies(split_rows["selection"], calibrator)
            policies = {"first_answer": {"kind": "first_answer"},
                        **{f"fixed_round_{n}": {"kind": "fixed_round", "round": n} for n in range(1, 11)},
                        "final_round": {"kind": "fixed_round", "round": 10},
                        **{k: selected[k] for k in ("confidence_reward_selected", "confidence_risk10_selected", "fixed_round_selected")}}
            if historical_reports:
                historical = historical_reports[tag]["native_mc_policies"][f"mc:{condition}:fixed_1"]
                old_threshold = historical["selection"]["threshold"]
                policies["historical_risk10_selected"] = {
                    "kind": old_threshold["mode"], "value": old_threshold["value"],
                    "frozen_historical_calibrator": historical["calibrator"],
                    "selection_source": "original automatic_report.json; preserved without refitting or reward reselection"}
            cell = {"calibrator": calibrator, "selected_policies": selected, "policies": policies, "splits": {}}
            for split, split_tr in {**split_rows, "all": trajectories}.items():
                summaries = {}
                for name, policy in policies.items():
                    recs = policy_records(split_tr, policy, calibrator)
                    summaries[name] = summarize_policy(recs, indices if split == "test" else None)
                    if split != "all":
                        for r in recs:
                            decision_rows.append({"model": tag, "condition": condition, "policy": name, **r,
                                                  **{f"reward_{k}": float(reward_values([r], k)[0]) for k in REWARDS}})
                    if split == "test":
                        test_records[cellname, name] = recs
                cell["splits"][split] = {"policies": summaries, "trajectory_diagnostics": trajectory_diagnostics(split_tr)}
                for n in range(1, 11):
                    round_rows.append({"model": tag, "condition": condition, "split": split, "round": n,
                                       **round_summary([tr[n-1] for tr in split_tr.values()], calibrator)})
            if historical_reports:
                # The old policy is replayed with its original calibrator, not
                # silently reinterpreted through a newly fitted map.
                old_commitments = {r["qid"]: r for r in historical["test"]["commitments"]}
                for record in test_records[cellname, "historical_risk10_selected"]:
                    old = old_commitments[record["qid"]]
                    expected_round = int(old["prefix_id"][1:]) if old["prefix_id"] else None
                    if record["round"] != expected_round or record["correct"] != bool(old["correct"]):
                        raise ValueError("historical policy replay differs from frozen original commitments")
                cell["historical_policy_replay_matches_original_commitments"] = True
            report["cells"][cellname] = cell
            for name in ("first_answer", "confidence_reward_selected", "confidence_risk10_selected", "fixed_round_selected"):
                report["test_comparisons"][f"{cellname}/{name}-minus-final_round"] = paired_difference(test_records[cellname, name], test_records[cellname, "final_round"], indices)
    for name in ("first_answer", "final_round", "confidence_reward_selected", "confidence_risk10_selected", "fixed_round_selected"):
        for condition in CONDITIONS:
            report["test_comparisons"][f"{condition}/{name}/qwen7b-minus-qwen3b"] = paired_difference(test_records[f"qwen7b/{condition}", name], test_records[f"qwen3b/{condition}", name], indices)
        for tag in MODELS:
            report["test_comparisons"][f"{tag}/{name}/same_category-minus-independent"] = paired_difference(test_records[f"{tag}/same_category_pool", name], test_records[f"{tag}/independent_pool", name], indices)
    return report, round_rows, decision_rows


def csv_write(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--raw-archive", type=Path, required=True)
    parser.add_argument("--historical-analysis-root", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("output directory already exists; preserve earlier selections/results")
    if args.bootstrap_samples < 100 or args.seed < 0:
        raise ValueError("invalid bootstrap configuration")
    # Reserve the output atomically before expensive archive verification.
    # Other reports should use sibling directories, never this reserved path.
    args.out.mkdir(parents=True)
    rows, audit = load_evidence(args.public, args.dataset, args.raw_archive)
    print(json.dumps({"phase": "evidence_verified", "questions": audit["questions"], "mc_rows": audit["mc_rows_total"]}), flush=True)
    historical_reports = {}
    if args.historical_analysis_root:
        for tag in MODELS:
            report_path = args.historical_analysis_root / tag / "main" / "automatic_report.json"
            receipt_path = report_path.with_name("completion_receipt.json")
            receipt = unique_json(receipt_path.read_bytes())
            if sha(report_path) != receipt["outputs_sha256"]["automatic_report.json"]:
                raise ValueError("historical report differs from original completion receipt")
            historical_reports[tag] = unique_json(report_path.read_bytes())
            audit["inputs_sha256"].update({str(p.resolve()): sha(p) for p in (report_path, receipt_path)})
    report, round_rows, decision_rows = analyze(rows, samples=args.bootstrap_samples, seed=args.seed,
                                              historical_reports=historical_reports)
    report.update(schema_version="imcqa-retrospective-v1", scope="post-hoc offline replay of original independent-prefix greedy MC generations", rewards=REWARDS,
                  bootstrap={"samples": args.bootstrap_samples, "seed": args.seed, "unit": "question (verified unique group)", "splits_with_intervals": ["test"],
                             "conditioning": "fixed fitted calibration and selected policies; does not include fitted-policy uncertainty", "multiplicity_correction": False,
                             "undefined_ratios": "omit undefined ratio resamples and report defined count; all undefined yields null interval"},
                  risk_binomial_interval={"method": "two-sided exact Clopper-Pearson 95%", "denominator": "committed questions", "zero_commitments": None,
                                          "scope": "descriptive independent-question conditional-binomial interval for a fixed policy; not a selection guarantee; complements the empirical bootstrap which degenerates with zero observed errors"},
                  limitations=["Original prompts did not describe sequential WAIT actions or these rewards.",
                               "Retrospective policies do not measure reward-aware behavior or actual online latency savings.",
                               "Only two related model sizes; no broad model ranking or IRT inference.",
                               "Word-decile prefixes are not certified clue boundaries or decreasing difficulty.",
                               "Correctness uses frozen MC gold; distractor semantic validity remains unreviewed.",
                               "Calibration is answer-conditional self-report calibration, not four-option softmax calibration.",
                               "All-answer accuracy counts abstention/invalid as not correct; conditional risk uses committed questions only.",
                               "Current analyses were chosen after earlier results were observed; test is a held-out split for fitting, not an untouched confirmatory preregistration.",
                               "Sensitivity rewards use the same primary-objective-selected decisions, without retuning."])
    write_json(args.out / "report.json", report)
    write_json(args.out / "evidence_audit.json", audit)
    csv_write(args.out / "round_metrics.csv", round_rows)
    csv_write(args.out / "policy_decisions.csv", decision_rows)
    policy_rows = []
    for cell, result in report["cells"].items():
        tag, condition = cell.split("/")
        for split, data in result["splits"].items():
            for name, metrics in data["policies"].items():
                policy_rows.append({"model": tag, "condition": condition, "split": split, "policy": name,
                                    **{k: v for k, v in metrics.items() if k not in {"bootstrap", "rewards"}},
                                    **{f"reward_{k}": v for k, v in metrics["rewards"].items()}})
    csv_write(args.out / "policy_summary.csv", policy_rows)
    for p, digest in audit["inputs_sha256"].items():
        if sha(Path(p)) != digest:
            raise ValueError("input mutated during analysis")
    receipt = {"schema_version": "imcqa-retrospective-receipt-v1", "status": "completed",
               "script_sha256": sha(Path(__file__)), "inputs_sha256": audit["inputs_sha256"],
               "calibration_source_sha256": sha(Path(__file__).resolve().parents[1] / "evaluation" / "jane_paired.py"),
               "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                           "scipy": scipy.__version__, "sklearn": sklearn.__version__},
               "outputs_sha256": {p.name: sha(p) for p in args.out.iterdir() if p.is_file()}}
    write_json(args.out / "analysis_receipt.json", receipt)
    print(json.dumps({"phase": "completed", "out": str(args.out.resolve()), "decision_rows": len(decision_rows)}), flush=True)


if __name__ == "__main__":
    main()
