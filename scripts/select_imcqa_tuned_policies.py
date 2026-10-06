#!/usr/bin/env python3
"""Select matched IMCQA policy families using saved development scores only.

The existing logistic calibrators remain frozen. This is policy selection on
previously inspected development data, not a confirmatory evaluation. No model
forward pass, remote service, or calibration fitting is performed.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist

import numpy as np

from scripts import analyze_imcqa_protocol_pilot as protocol
from scripts import analyze_imcqa_transfer as transfer

MENUS = tuple(transfer.MENUS)
REWARDS = np.asarray(transfer.REWARDS, dtype=float)
GRID = tuple(i / 20 for i in range(21))
TOLERANCE = 1e-12
PLAN = {
    "schema_version": "imcqa-independent-policy-selection-plan-v1",
    "evidence_scope": "previously inspected development data only",
    "calibration": "unchanged original plain-MCQA per-menu logistic maps fitted on 20 questions",
    "selection_questions": {"protocol_selection": 20, "transfer_selection": 100},
    "answer_source": "plain", "rotations": [0, 1, 2, 3], "rounds": [1, 2, 3, 4, 5],
    "threshold_grid": list(GRID), "threshold_comparison": ">=",
    "adaptive_selective": "first qualifying round; terminal PASS if none; explicit always-PASS candidate",
    "fixed_selective": "joint round and threshold selection; answer only at that round if qualifying, otherwise terminal PASS; explicit always-PASS candidate",
    "objective": "mean reward after averaging four rotations within each question; all 120 questions equally weighted",
    "tie_break": ["within 1e-12 of maximum mean reward", "explicit PASS", "lowest coverage within 1e-12", "highest threshold", "earliest fixed round"],
    "wrong_reward": -1, "pass_reward": 0, "correct_rewards": REWARDS.tolist(),
    "planning": {"target_reward_difference": .05, "marginal_power_per_menu": .90,
        "two_sided_family_alpha": .05, "family_size": 2, "per_menu_confidence": .975,
        "bootstrap_samples": 20000, "bootstrap_seed": 1,
        "planning_sd": "maximum across menus of the 95th percentile question-bootstrap SD for the independently selected AS-FS pair",
        "rounding": "round upward to next multiple of 50; minimum 200 fresh questions",
        "formula": "ceil((z_(1-alpha/(2*family_size))+z_power)^2 * planning_sd^2 / delta^2)",
        "scope": "normal-approximation planning heuristic; neither a population variance bound nor guaranteed power; 90% is marginal per menu, not joint success probability"},
}


def sha256(path):
    """Hash a file without retaining its complete contents in memory."""
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def verify_manifest(root):
    """Verify every retained archive member and reject ambiguous paths."""
    manifest = read_json(root / "artifact_manifest.json")
    names = set()
    for row in manifest["files"]:
        name = Path(row["path"])
        if name.is_absolute() or ".." in name.parts or str(name) in names:
            raise ValueError("unsafe or duplicate archive member")
        names.add(str(name))
        path = root / name
        if path.stat().st_size != row["bytes"] or sha256(path) != row["sha256"]:
            raise ValueError("archive member differs: " + str(name))
    return {"manifest_sha256": sha256(root / "artifact_manifest.json"), "verified_members": len(names)}


def verify_analysis(root):
    receipt = read_json(root / "analysis_receipt.json")
    if receipt["status"] not in ("complete", "partial"):
        raise ValueError("upstream analysis did not complete its declared scope")
    hashes = receipt.get("outputs_sha256", receipt.get("output_sha256", {}))
    if not hashes:
        raise ValueError("upstream analysis lacks output bindings")
    for name, expected in hashes.items():
        if Path(name).is_absolute() or ".." in Path(name).parts or sha256(root / name) != expected:
            raise ValueError("upstream analysis output differs: " + name)
    return sha256(root / "analysis_receipt.json")


def plain_views(package, score_path, receipt_path, qids, questions, cohort):
    """Reconstruct plain candidates from receipt-bound logits and gold joins."""
    receipt = read_json(receipt_path)
    if receipt["status"] != "complete" or receipt["scores_sha256"] != sha256(score_path):
        raise ValueError("incomplete or changed worker scores")
    jobs = {j["score_id"]: j for j in package["jobs"]
            if j["qid"] in qids and j["arm"] == "plain" and j["block"] == "factorial"}
    if len(jobs) != len(qids) * 40:
        raise ValueError("incomplete plain public design")
    result, seen, n_rows = [], set(), 0
    with score_path.open() as handle:
        for line in handle:
            row = json.loads(line); n_rows += 1
            sid = row["score_id"]
            if sid in seen:
                raise ValueError("duplicate raw score id")
            seen.add(sid)
            if sid not in jobs:
                continue
            job = jobs[sid]
            for key in ("qid", "group_id", "condition", "arm", "rotation", "round", "prompt_sha256"):
                if row[key] != job[key]:
                    raise ValueError("score/public identity mismatch: " + key)
            if hashlib.sha256(job["prompt"].encode()).hexdigest() != job["prompt_sha256"]:
                raise ValueError("public prompt bytes differ")
            question = questions[job["qid"]]
            if question["group_id"] != job["group_id"] or question["split"] != "selection":
                raise ValueError("question identity or development split differs")
            menu = next(m for m in question["menus"] if m["condition"] == job["condition"])
            view = protocol.score_view(row, {**job, "canonical_gold_option_id": menu["gold_option_id"]})
            result.append({**{k: view[k] for k in ("qid", "group_id", "condition", "rotation", "round", "score_id", "candidate_choice", "candidate_correct", "canonical_gold_option_id")},
                "candidate_confidence": max(view["canonical_answer_probabilities"].values()), "cohort": cohort})
    if n_rows != receipt["completed_rows"] or not set(jobs) <= seen:
        raise ValueError("raw score coverage differs")
    return result


def validate_roles(states, fits, questions, expected_counts=None):
    """Reject calibration leakage, repeated groups, and incomplete trajectories."""
    if not states:
        raise ValueError("no development states")
    fit_sets = [set(fits[m]["fit_qids"]) for m in MENUS]
    if fit_sets[0] != fit_sets[1] or len(fit_sets[0]) != 20:
        raise ValueError("the two calibrators must share the original 20 fit questions")
    calibration = fit_sets[0]
    qids = {s["qid"] for s in states}
    if calibration & qids:
        raise ValueError("calibration question leaked into policy selection")
    cal_groups = {questions[q]["group_id"] for q in calibration}
    dev_groups = {questions[q]["group_id"] for q in qids}
    if cal_groups & dev_groups:
        raise ValueError("calibration group leaked into policy selection")
    if len(dev_groups) != len(qids) or len(cal_groups) != len(calibration):
        raise ValueError("question bootstrap requires one distinct group per question")
    cohorts = defaultdict(set)
    seen = set()
    for s in states:
        key = (s["qid"], s["condition"], s["rotation"], s["round"])
        if key in seen:
            raise ValueError("duplicate development state")
        seen.add(key); cohorts[s["cohort"]].add(s["qid"])
        if s["group_id"] != questions[s["qid"]]["group_id"]:
            raise ValueError("state group differs from gold source")
    if sum(map(len, cohorts.values())) != len(qids):
        raise ValueError("development cohorts overlap")
    expected = {(q, m, rotation, rnd) for q in qids for m in MENUS for rotation in range(4) for rnd in range(1, 6)}
    if seen != expected:
        raise ValueError("incomplete question/menu/rotation/round grid")
    if expected_counts is not None and {k: len(v) for k, v in cohorts.items()} != expected_counts:
        raise ValueError("development cohort counts differ")
    return {"calibration_qids": sorted(calibration), "selection_qids": sorted(qids),
        "cohort_qids": {k: sorted(v) for k, v in sorted(cohorts.items())},
        "calibration_group_ids": sorted(cal_groups), "selection_group_ids": sorted(dev_groups),
        "calibration_selection_qid_overlap": [], "calibration_selection_group_overlap": [],
        "unique_development_groups": len(dev_groups), "n_development_states": len(states)}


def load_inputs(factorized, evidence):
    """Bind the two archived studies and leave all fitted maps unchanged."""
    validation = {"factorized_archive": verify_manifest(factorized), "transfer_archive": verify_manifest(evidence)}
    validation["factorized_analysis_receipt_sha256"] = verify_analysis(factorized / "cpu_analysis")
    validation["transfer_analysis_receipt_sha256"] = verify_analysis(evidence / "analysis")
    if read_json(evidence / "analysis/validation.json")["passed"] is not True:
        raise ValueError("upstream transfer validation failed")
    factor_fit = factorized / "cpu_analysis/fitted_parameters.json"
    if sha256(factor_fit) != sha256(evidence / "inputs/fitted_parameters.json"):
        raise ValueError("the two studies do not share identical frozen calibrators")
    dataset_path = evidence / "inputs/main_dataset.json"
    if sha256(dataset_path) != sha256(factorized / "inputs/frozen/main_dataset.json"):
        raise ValueError("source gold datasets differ")
    questions = {q["qid"]: q for q in read_json(dataset_path)["questions"]}
    old_public = read_json(factorized / "inputs/protocol/public.json")
    new_public = read_json(evidence / "run/pilot.json")
    if old_public["source_input_sha256"] != new_public["source_input_sha256"]:
        raise ValueError("source job datasets differ")
    old_qids = old_public["selection"]["selected_qids"]["selection"]
    new_qids = new_public["selection"]["selected_qids"]["selection"]
    fits = transfer.frozen_parameters(read_json(factor_fit), old_qids + new_qids)
    states = plain_views(old_public, factorized / "prior_protocol_run/output/qwen7b/scores.jsonl",
        factorized / "prior_protocol_run/output/qwen7b/receipt.json", old_qids, questions, "protocol_selection")
    states += plain_views(new_public, evidence / "run/output/qwen7b/scores.jsonl",
        evidence / "run/output/qwen7b/receipt.json", new_qids, questions, "transfer_selection")
    validation["roles"] = validate_roles(states, fits, questions, PLAN["selection_questions"])
    validation["input_sha256"] = {"fitted_parameters": sha256(factor_fit), "main_dataset": sha256(dataset_path),
        "protocol_public": sha256(factorized / "inputs/protocol/public.json"),
        "transfer_public": sha256(evidence / "run/pilot.json"),
        "protocol_scores": sha256(factorized / "prior_protocol_run/output/qwen7b/scores.jsonl"),
        "transfer_scores": sha256(evidence / "run/output/qwen7b/scores.jsonl")}
    validation.update(passed=True, model_inference_performed=False, calibration_fitting_performed=False,
        scope="retained manifests and accepted receipts plus independent raw-logit candidate reconstruction; original prompt/numerical audits remain upstream evidence")
    return states, fits, validation


def candidates(family):
    """Enumerate explicit PASS and the complete prespecified numeric grid."""
    if family not in ("adaptive_selective", "fixed_selective"):
        raise ValueError("unknown policy family")
    result = [{"family": family, "threshold": None, "fixed_round": None, "always_pass": True}]
    for rnd in ([None] if family == "adaptive_selective" else range(1, 6)):
        result += [{"family": family, "threshold": tau, "fixed_round": rnd, "always_pass": False} for tau in GRID]
    for index, row in enumerate(result):
        row["candidate_id"] = f"{family}:{index:03d}"
    return result


def evaluate_candidate(probabilities, correct, candidate):
    """Return episode rewards/commitments/rounds for n-question x 4 x 5 arrays."""
    if probabilities.shape != correct.shape or probabilities.ndim != 3 or probabilities.shape[1:] != (4, 5):
        raise ValueError("balanced question-by-four-orders-by-five-rounds arrays required")
    shape = probabilities.shape[:2]
    rounds = np.full(shape, -1, dtype=int)
    if not candidate["always_pass"]:
        met = probabilities >= candidate["threshold"]
        if candidate["family"] == "adaptive_selective":
            rounds = np.where(met.any(axis=2), met.argmax(axis=2), -1)
        else:
            fixed = candidate["fixed_round"] - 1
            rounds = np.where(met[:, :, fixed], fixed, -1)
    committed = rounds >= 0
    chosen_correct = np.take_along_axis(correct, np.maximum(rounds, 0)[:, :, None], axis=2)[:, :, 0] & committed
    rewards = np.where(committed, np.where(chosen_correct, REWARDS[np.maximum(rounds, 0)], -1.), 0.)
    return {"reward": rewards, "committed": committed, "correct": chosen_correct,
        "wrong": committed & ~chosen_correct, "round": rounds + 1}


def choose_candidate(rows):
    """Apply the frozen reward/coverage/PASS/threshold/round tie hierarchy."""
    best = max(r["mean_reward"] for r in rows)
    tied = [r for r in rows if best - r["mean_reward"] <= TOLERANCE]
    passes = [r for r in tied if r["always_pass"]]
    if passes:
        return passes[0]
    minimum = min(r["coverage"] for r in tied)
    tied = [r for r in tied if r["coverage"] - minimum <= TOLERANCE]
    return min(tied, key=lambda r: (-r["threshold"], r["fixed_round"] or 0, r["candidate_id"]))


def select_policies(states, fits):
    """Select each family independently, retaining every candidate's outcome."""
    qids = sorted({r["qid"] for r in states})
    lookup = {(r["qid"], r["condition"], r["rotation"], r["round"]): r for r in states}
    grids, selected, values, selected_records = [], {}, {}, []
    for menu in MENUS:
        probabilities = np.asarray([[[transfer.calibrated_probability(lookup[q, menu, rot, rnd]["candidate_confidence"], fits[menu])
            for rnd in range(1, 6)] for rot in range(4)] for q in qids])
        correct = np.asarray([[[lookup[q, menu, rot, rnd]["candidate_correct"] for rnd in range(1, 6)] for rot in range(4)] for q in qids], dtype=bool)
        for family in ("adaptive_selective", "fixed_selective"):
            rows, outcomes = [], {}
            for candidate in candidates(family):
                result = evaluate_candidate(probabilities, correct, candidate)
                outcomes[candidate["candidate_id"]] = result
                row = {**candidate, "condition": menu,
                    "mean_reward": float(result["reward"].mean(axis=1).mean()),
                    "coverage": float(result["committed"].mean(axis=1).mean()),
                    "conditional_error": float(result["wrong"].sum() / result["committed"].sum()) if result["committed"].any() else None,
                    "n_questions": len(qids), "n_episodes": len(qids) * 4}
                rows.append(row)
            chosen = choose_candidate(rows)
            selected[menu, family] = chosen
            grids.extend(rows)
            values[menu, family] = np.stack([outcomes[r["candidate_id"]]["reward"].mean(axis=1) for r in rows], axis=1)
            result = outcomes[chosen["candidate_id"]]
            for i, qid in enumerate(qids):
                for rotation in range(4):
                    rnd = int(result["round"][i, rotation])
                    state = lookup[qid, menu, rotation, rnd or 5]
                    selected_records.append({"qid": qid, "group_id": state["group_id"], "cohort": state["cohort"],
                        "condition": menu, "policy": family, "rotation": rotation, "candidate_id": chosen["candidate_id"],
                        "threshold": chosen["threshold"], "fixed_round": chosen["fixed_round"], "round": rnd or None,
                        **{key: (float(result[key][i, rotation]) if key == "reward" else bool(result[key][i, rotation])) for key in ("reward", "committed", "correct", "wrong")},
                        "answer_score_id": state["score_id"] if rnd else None,
                        "canonical_choice": state["candidate_choice"] if rnd else None})
    return qids, grids, selected, values, selected_records


def sample_size(sd, settings):
    z = NormalDist().inv_cdf(1 - settings["two_sided_family_alpha"] / (2 * settings["family_size"]))
    power = NormalDist().inv_cdf(settings["marginal_power_per_menu"])
    return math.ceil((z + power) ** 2 * sd ** 2 / settings["target_reward_difference"] ** 2)


def plan_sample_size(values, selected, *, samples=20000, seed=1):
    """Use selected-pair variability; retain maximum-grid SD as sensitivity."""
    n = next(iter(values.values())).shape[0]
    if n < 2:
        raise ValueError("variance planning requires at least two questions")
    indices = np.random.default_rng(seed).integers(0, n, (samples, n))
    largest_point_sd, largest_pair = 0., None
    per_menu = []
    for menu in MENUS:
        left, right = values[menu, "adaptive_selective"], values[menu, "fixed_selective"]
        differences = (left[:, :, None] - right[:, None, :]).reshape(n, -1)
        point = differences.std(axis=0, ddof=1)
        index = int(point.argmax()); pair = np.unravel_index(index, (left.shape[1], right.shape[1]))
        if point[index] > largest_point_sd:
            largest_point_sd = float(point[index]); largest_pair = {"condition": menu, "as_index": int(pair[0]), "fs_index": int(pair[1])}
        li = int(selected[menu, "adaptive_selective"]["candidate_id"].split(":")[1])
        ri = int(selected[menu, "fixed_selective"]["candidate_id"].split(":")[1])
        delta = left[:, li] - right[:, ri]
        per_menu.append({"condition": menu, "development_mean_difference": float(delta.mean()),
            "selected_pair_sd": float(delta.std(ddof=1)),
            "selected_pair_sd_bootstrap95_upper": float(np.quantile(delta[indices].std(axis=1, ddof=1), .95)),
            "selected_pair_point_sd_planned_n_unrounded": sample_size(float(delta.std(ddof=1)), PLAN["planning"]),
            "full_grid_largest_sd": float(point.max()), "full_grid_pairs": int(len(point))})
    planning_sd = max(r["selected_pair_sd_bootstrap95_upper"] for r in per_menu)
    unrounded = sample_size(planning_sd, PLAN["planning"])
    rounded = max(200, 50 * math.ceil(unrounded / 50))
    return {**PLAN["planning"], "bootstrap_samples": samples, "bootstrap_seed": seed,
        "development_questions": n, "per_menu": per_menu,
        "used_planning_sd": planning_sd, "planned_questions_unrounded": unrounded,
        "planned_questions": rounded,
        "full_grid_largest_point_sd": largest_point_sd, "largest_point_sd_pair": largest_pair,
        "sensitivity_all_candidate_pairs_n_unrounded": sample_size(largest_point_sd, PLAN["planning"]),
        "sensitivity_scope": "Maximum grid variance includes deliberately poor, unselected policies; it does not set the primary sample size.",
        "universal_bounded_difference_sd_ceiling": 2.,
        "normal_approximation_n_at_universal_sd_ceiling": sample_size(2., PLAN["planning"]),
        "universal_ceiling_scope": "AS-FS per-question rewards lie in [-2,2]; SD<=2. The normal power formula itself remains approximate."}


def write_csv(path, rows):
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("factorized", "transfer-evidence", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--previous-lock", type=Path,
        help="Bind a prior lock when adding role metadata without changing the selected policies.")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    # The complete contract is persisted before any development outcome is read.
    write_json(args.out / "selection_plan.json", PLAN)
    states, fits, validation = load_inputs(args.factorized, args.transfer_evidence)
    qids, grids, chosen, values, episodes = select_policies(states, fits)
    planning = plan_sample_size(values, chosen)
    frozen = {"schema_version": "imcqa-independently-tuned-policies-v1",
        "status": "frozen_for_fresh_evaluation", "evidence_scope": "development policy selection, not a confirmatory result",
        "selection_plan_sha256": sha256(args.out / "selection_plan.json"),
        "input_sha256": validation["input_sha256"], "selection_qids": qids,
        "calibration_qids": validation["roles"]["calibration_qids"],
        "selection_group_ids": validation["roles"]["selection_group_ids"],
        "calibration_group_ids": validation["roles"]["calibration_group_ids"],
        "calibrators": fits, "policies": [chosen[key] for key in sorted(chosen)],
        "new_model_forward_passes": 0, "new_calibration_fits": 0}
    if args.previous_lock:
        previous = read_json(args.previous_lock)
        previous_planning = read_json(args.previous_lock.parent / "sample_size_planning.json")
        unchanged = ("calibrators", "policies", "selection_qids", "calibration_qids", "input_sha256", "selection_plan_sha256")
        if any(previous[key] != frozen[key] for key in unchanged) or previous_planning != planning:
            raise ValueError("metadata-only lock revision changed policies, source identities, or sample planning")
        frozen["freeze_history"] = {"previous_policy_lock_sha256": sha256(args.previous_lock),
            "revision_reason": "Added independent calibration/development group-overlap guard before fresh inference; no policy retuning or sample-size change.",
            "unchanged_policies_and_planning_verified": True}
    write_json(args.out / "frozen_policies.json", frozen)
    write_json(args.out / "validation.json", validation)
    write_json(args.out / "sample_size_planning.json", planning)
    write_csv(args.out / "candidate_grid.csv", grids)
    write_csv(args.out / "selected_episodes.csv", episodes)
    per_question = []
    for menu in MENUS:
        for family in ("adaptive_selective", "fixed_selective"):
            rows = candidates(family)
            for j, candidate in enumerate(rows):
                for i, qid in enumerate(qids):
                    per_question.append({"qid": qid, "condition": menu, "candidate_id": candidate["candidate_id"], "reward": float(values[menu, family][i, j])})
    write_csv(args.out / "candidate_question_rewards.csv", per_question)
    write_json(args.out / "selection_receipt.json", {"status": "complete", "validation_passed": True,
        "selector_sha256": sha256(Path(__file__)), "new_model_forward_passes": 0,
        "outputs_sha256": {p.name: sha256(p) for p in sorted(args.out.iterdir()) if p.is_file()}})
    print(json.dumps({"status": "complete", "n_questions": len(qids), "policies": frozen["policies"],
        "planned_questions": planning["planned_questions"], "output": str(args.out)}))


if __name__ == "__main__":
    main()
