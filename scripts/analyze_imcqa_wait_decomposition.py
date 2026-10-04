#!/usr/bin/env python3
"""CPU-only, same-candidate diagnosis of the completed development WAIT pilot.

Counterfactual candidate availability is not evidence that a deployable stopping
policy could identify that candidate as correct. All fitted quantities use the
calibration split only; selection results remain exploratory.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import time
from typing import Any

import numpy as np
import scipy
from scipy.optimize import minimize
from scipy.special import expit, logit

from scripts import analyze_imcqa_wait_pilot as base


PUBLIC_SHA256 = "b5f5f6b5437942904a90f8e49d153ead9e70f215ec855e95db3323951a55078e"
CALIBRATION_L2 = 0.01
CALIBRATION_BINS = (0.0, 0.4, 0.6, 0.8, 1.0)


def candidate_states(trajectory: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract top A--D under unchanged WAIT logits, preserving each state."""
    if [row["round"] for row in trajectory] != [1, 2, 3, 4, 5]:
        raise ValueError("exactly five ordered rounds are required")
    if any(row["variant"] != "wait" for row in trajectory):
        raise ValueError("same-WAIT diagnosis requires WAIT-prompt rows")
    result = []
    for row in trajectory:
        conditional = base.softmax(row["raw_action_logits"], base.OPTIONS)
        candidate = max(base.OPTIONS, key=conditional.get)
        result.append({**row, "candidate": candidate,
                       "candidate_confidence": conditional[candidate],
                       "candidate_correct": candidate == row["gold_option_id"]})
    return result


def candidate_record(states: list[dict[str, Any]], selected_round: int | None,
                     policy: str) -> dict[str, Any]:
    """Answer using the unchanged WAIT candidate, or explicitly PASS."""
    if selected_round is not None and selected_round not in range(1, 6):
        raise ValueError("invalid candidate commitment round")
    row = states[(selected_round or 5) - 1]
    committed = selected_round is not None
    correct = committed and row["candidate_correct"]
    return {**{key: row[key] for key in ("qid", "group_id", "split", "condition", "variant")},
            "policy": policy, "committed": committed, "correct": correct,
            "wrong": committed and not correct, "terminal_pass": not committed,
            "round": selected_round, "observed_round": selected_round or 5,
            "actual_fraction": row["actual_fraction"],
            "canonical_choice": row["candidate"] if committed else None,
            "reward": (base.REWARDS[selected_round - 1] if correct else -1.0) if committed else 0.0,
            "counterfactual_uncached_input_tokens": None}


def decompose_trajectory(trajectory: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Partition native outcomes and verify an exact same-candidate regret sum."""
    states = candidate_states(trajectory)
    native = {**base.policy_record(trajectory), "policy": "native_wait"}
    stop = native["observed_round"]
    correct_rounds = [row["round"] for row in states if row["candidate_correct"]]
    earliest = min(correct_rounds) if correct_rounds else None
    oracle = candidate_record(states, earliest, "hindsight_same_wait_or_pass")
    if native["reward"] > oracle["reward"] + 1e-12:
        raise ValueError("same-candidate hindsight failed to bound native reward")
    if native["committed"]:
        chosen = states[stop - 1]
        if chosen["candidate"] != trajectory[stop - 1]["top_option_id"]:
            raise ValueError("native answer and same-WAIT conditional candidate disagree")
    outcome = ("correct_" if native["correct"] else "wrong_") + ("early" if stop < 5 else "terminal") if native["committed"] else "terminal_pass"
    penalty = float(native["wrong"])
    forgone = oracle["reward"] if not native["correct"] else 0.0
    delay = oracle["reward"] - native["reward"] if native["correct"] else 0.0
    regret = oracle["reward"] - native["reward"]
    if min(penalty, forgone, delay) < -1e-12 or not math.isclose(regret, penalty + forgone + delay, abs_tol=1e-12):
        raise ValueError("additive regret identity failed")
    skipped = [row["round"] for row in states if row["round"] <= stop and row["top_option_id"] == "E" and row["candidate_correct"]]
    later = [r for r in correct_rounds if r > stop]
    record = {**{key: native[key] for key in ("qid", "group_id", "split", "condition")},
              "native_outcome": outcome, "native_reward": native["reward"],
              "native_observed_round": stop, "native_committed": native["committed"],
              "earliest_correct_candidate_round": earliest,
              "correct_candidate_any_round": bool(correct_rounds),
              "visited_correct_candidate_skipped": bool(skipped),
              "n_visited_correct_candidate_skipped": len(skipped),
              "skipped_correct_rounds": json.dumps(skipped),
              "post_termination_correct_candidate": bool(later),
              "post_termination_correct_rounds": json.dumps(later),
              "wrong_early_with_later_correct": outcome == "wrong_early" and bool(later),
              "pass_with_correct_candidate": outcome == "terminal_pass" and bool(correct_rounds),
              "wrong_with_no_correct_candidate": native["wrong"] and not correct_rounds,
              "hindsight_reward": oracle["reward"], "same_wait_regret": regret,
              "wrong_answer_penalty_component": penalty,
              "forgone_correct_reward_component": forgone,
              "correct_answer_delay_component": delay}
    rounds = []
    for row in states:
        visited = row["round"] <= stop
        rounds.append({**{key: row[key] for key in ("qid", "group_id", "split", "condition", "round")},
                       "candidate": row["candidate"], "candidate_confidence": row["candidate_confidence"],
                       "candidate_correct": row["candidate_correct"], "native_action": row["top_option_id"],
                       "visited_by_native_policy": visited,
                       "skipped_correct_visited": visited and row["top_option_id"] == "E" and row["candidate_correct"],
                       "counterfactual_after_termination": not visited})
    policies = [native, oracle, candidate_record(states, None, "always_pass")]
    policies.extend(candidate_record(states, r, f"same_wait_fixed_round_{r}") for r in range(1, 6))
    return record, rounds, policies


def fit_calibrator(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Fit two parameters on calibration data; nonnegative slope, fixed L2."""
    if not rows or any(row["split"] != "calibration" for row in rows):
        raise ValueError("calibrator accepts only nonempty calibration rows")
    qids = sorted({row["qid"] for row in rows})
    counts = Counter(row["qid"] for row in rows)
    if len(set(counts.values())) != 1 or next(iter(counts.values())) != 5:
        raise ValueError("calibrator requires all five rounds equally per question")
    x = logit(np.clip([row["candidate_confidence"] for row in rows], 1e-6, 1 - 1e-6))
    y = np.asarray([row["candidate_correct"] for row in rows], dtype=float)
    start = np.array([float(logit((y.sum() + 0.5) / (len(y) + 1))), 0.0])
    def objective(theta):
        z = theta[0] + theta[1] * x
        residual = expit(z) - y
        loss = float(np.mean(np.logaddexp(0, z) - y * z) + CALIBRATION_L2 * theta[1] ** 2 / 2)
        gradient = np.array([np.mean(residual), np.mean(residual * x) + CALIBRATION_L2 * theta[1]])
        return loss, gradient
    if len(set(y)) == 1:
        theta = start
        method = "smoothed_intercept_only_degenerate_labels"
    else:
        fit = minimize(objective, start, jac=True, method="L-BFGS-B", bounds=[(None, None), (0, None)],
                       options={"maxiter": 1000, "ftol": 1e-12, "gtol": 1e-8})
        if not fit.success or not np.all(np.isfinite(fit.x)):
            raise ValueError(f"calibrator failed: {fit.message}")
        theta = fit.x
        method = "monotone_logistic_max_conditional_probability"
    return {"method": method, "intercept": float(theta[0]), "slope": float(theta[1]),
            "l2_mean_logloss_penalty": CALIBRATION_L2, "fit_split": "calibration",
            "fit_qids": qids, "n_fit_questions": len(qids), "n_fit_states": len(rows),
            "fit_target": "same-WAIT top A-D candidate correctness across all five counterfactual rounds"}


def calibrated_probability(fit: dict[str, Any], confidence: float) -> float:
    return float(expit(fit["intercept"] + fit["slope"] * logit(np.clip(confidence, 1e-6, 1 - 1e-6))))


def myopic_record(trajectory: list[dict[str, Any]], fit: dict[str, Any]) -> dict[str, Any]:
    """Take first positive answer EV against PASS; no continuation value model."""
    states = candidate_states(trajectory)
    chosen = next((row["round"] for row in states
                   if (1 + base.REWARDS[row["round"] - 1]) * calibrated_probability(fit, row["candidate_confidence"]) - 1 > 0), None)
    return candidate_record(states, chosen, "calibrated_myopic_positive_ev_vs_pass")


def calibration_metrics(rows: list[dict[str, Any]], probability_key: str) -> dict[str, Any]:
    """Fixed-bin descriptive reliability; states are not independent samples."""
    if not rows:
        return {"n_states": 0, "n_questions": 0, "brier": None, "log_loss": None, "bins": []}
    p = np.asarray([row[probability_key] for row in rows], dtype=float)
    y = np.asarray([row["candidate_correct"] for row in rows], dtype=float)
    clipped = np.clip(p, 1e-12, 1 - 1e-12)
    bins = []
    for i, (low, high) in enumerate(zip(CALIBRATION_BINS[:-1], CALIBRATION_BINS[1:])):
        mask = (p >= low) & ((p <= high) if i == len(CALIBRATION_BINS) - 2 else (p < high))
        bins.append({"lower_inclusive": low, "upper": high, "upper_inclusive": i == len(CALIBRATION_BINS) - 2,
                     "n_states": int(mask.sum()), "mean_probability": float(p[mask].mean()) if mask.any() else None,
                     "accuracy": float(y[mask].mean()) if mask.any() else None})
    return {"n_states": len(rows), "n_questions": len({row["qid"] for row in rows}),
            "brier": float(np.mean((p - y) ** 2)),
            "log_loss": float(np.mean(-(y * np.log(clipped) + (1 - y) * np.log1p(-clipped)))),
            "mean_probability": float(p.mean()), "accuracy": float(y.mean()), "bins": bins}


def match_historical(jobs: list[dict[str, Any]], rows: list[dict[str, Any]], all_scores: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Join old full responses by original job identity and prompt hash."""
    wanted = {job["source_job_id"]: job for job in jobs if job["variant"] == "wait" and job["round"] == 5}
    lookup = {}
    for row in rows:
        if row["job_id"] in wanted:
            if row["job_id"] in lookup:
                raise ValueError("duplicate historical source job")
            lookup[row["job_id"]] = row
    if set(lookup) != set(wanted):
        raise ValueError("historical source job coverage mismatch")
    scores = {(r["qid"], r["condition"], r["variant"]): r for r in all_scores if r["round"] == 5}
    output = []
    for job_id, job in sorted(wanted.items()):
        row = lookup[job_id]
        for key in ("qid", "group_id", "split", "condition", "menu_id", "prefix_id", "fraction"):
            if row[key] != job[key]:
                raise ValueError(f"historical {key} identity mismatch")
        if row["prompt_sha256"] != job["source_prompt_sha256"] or row["format"] != "mc":
            raise ValueError("historical original prompt hash/format mismatch")
        parsed = json.loads(row["raw_response"])
        if parsed["status"] not in ("answer", "abstain") or parsed["status"] != row["status"] or parsed["answer"] != row["answer"]:
            raise ValueError("historical raw response fields disagree with graded row")
        correct = parsed["status"] == "answer" and parsed["answer"] == job["gold_option_id"]
        if bool(row["correct"]) != correct:
            raise ValueError("historical grade differs from independent frozen-gold join")
        wait = scores[job["qid"], job["condition"], "wait"]
        forced = scores[job["qid"], job["condition"], "forced"]
        candidate = max(base.OPTIONS, key=lambda label: wait["raw_action_logits"][label])
        output.append({**{key: job[key] for key in ("qid", "group_id", "split", "condition", "menu_id", "prefix_id")},
                       "source_job_id": job_id, "source_prompt_sha256": row["prompt_sha256"],
                       "historical_correct": correct, "historical_answer": row["answer"],
                       "historical_status": row["status"], "forced_game_correct": forced["correct"],
                       "same_wait_candidate_correct": candidate == job["gold_option_id"]})
    return output


def run_analysis(all_rows: dict[str, list[dict[str, Any]]], samples: int = 2000, seed: int = 1) -> dict[str, Any]:
    """Fit calibration-only baselines and summarize paired question outcomes."""
    decomposition, round_rows, policies, fits = [], [], [], []
    trajectories = {}
    for model, rows in all_rows.items():
        for key, trajectory in base.make_trajectories(rows).items():
            split, variant, condition, qid = key
            if variant != "wait":
                continue
            trajectories[model, split, condition, qid] = trajectory
            record, rounds, records = decompose_trajectory(trajectory)
            decomposition.append({"model": model, **record})
            round_rows.extend({"model": model, **row} for row in rounds)
            policies.extend({"model": model, **row} for row in records)
    for model in all_rows:
        for condition in base.MENUS:
            calibration = [row for row in round_rows if row["model"] == model and row["condition"] == condition and row["split"] == "calibration"]
            fit = fit_calibrator(calibration)
            fits.append({"model": model, "condition": condition, **fit})
            choices = ["always_pass", *[f"same_wait_fixed_round_{r}" for r in range(1, 6)]]
            reward_means = {name: float(np.mean([r["reward"] for r in policies if r["model"] == model and r["condition"] == condition and r["split"] == "calibration" and r["policy"] == name])) for name in choices}
            selected = max(choices, key=reward_means.get)
            fits[-1]["calibration_selected_fixed_policy"] = selected
            fits[-1]["calibration_fixed_policy_mean_rewards"] = reward_means
            for row in round_rows:
                if row["model"] == model and row["condition"] == condition:
                    row["calibrated_probability"] = calibrated_probability(fit, row["candidate_confidence"])
            for (tag, split, menu, qid), trajectory in trajectories.items():
                if tag != model or menu != condition:
                    continue
                policies.append({"model": model, **myopic_record(trajectory, fit)})
                selected_round = None if selected == "always_pass" else int(selected.rsplit("_", 1)[1])
                policies.append({"model": model, **candidate_record(candidate_states(trajectory), selected_round, "calibration_selected_fixed_or_pass")})
    cells = defaultdict(list)
    for record in policies:
        cells[record["model"], record["split"], record["condition"], record["policy"]].append(record)
    summaries = []
    comparisons = []
    for (model, split, condition, policy), records in sorted(cells.items()):
        indices = base.bootstrap_indices(len(records), samples, seed)
        summaries.append({"model": model, "split": split, "condition": condition, "policy": policy, **base.summarize_policy(records, indices)})
        if policy != "native_wait":
            comparisons.append({"model": model, "split": split, "condition": condition, "comparison": f"{policy}_minus_native_wait",
                                **base.paired_difference(records, cells[model, split, condition, "native_wait"], indices)})
    decomposition_summaries, calibration_summaries = [], []
    for model in all_rows:
        for split in base.SPLITS:
            for condition in base.MENUS:
                records = [r for r in decomposition if (r["model"], r["split"], r["condition"]) == (model, split, condition)]
                summary = {"model": model, "split": split, "condition": condition, "n_questions": len(records),
                           "native_outcome_counts": dict(Counter(r["native_outcome"] for r in records))}
                for field in ("correct_candidate_any_round", "visited_correct_candidate_skipped", "post_termination_correct_candidate",
                              "wrong_early_with_later_correct", "pass_with_correct_candidate", "wrong_with_no_correct_candidate"):
                    summary[f"n_{field}"] = sum(r[field] for r in records)
                for field in ("native_reward", "hindsight_reward", "same_wait_regret", "wrong_answer_penalty_component", "forgone_correct_reward_component", "correct_answer_delay_component"):
                    summary[f"mean_{field}"] = float(np.mean([r[field] for r in records]))
                decomposition_summaries.append(summary)
                for scope in ("all_counterfactual_states", "visited_native_states", "post_termination_counterfactual_states"):
                    rows = [r for r in round_rows if (r["model"], r["split"], r["condition"]) == (model, split, condition)
                            and (scope == "all_counterfactual_states" or (r["visited_by_native_policy"] if scope == "visited_native_states" else r["counterfactual_after_termination"]))]
                    calibration_summaries.append({"model": model, "split": split, "condition": condition, "scope": scope,
                                                 "raw_conditional_confidence": calibration_metrics(rows, "candidate_confidence"),
                                                 "calibrated_correctness": calibration_metrics(rows, "calibrated_probability")})
    return {"decomposition_records": decomposition, "round_records": round_rows, "policy_records": policies,
            "decomposition_summaries": decomposition_summaries, "policy_summaries": summaries,
            "paired_comparisons": comparisons, "calibration_fits": fits, "calibration_summaries": calibration_summaries}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("public", "frozen-source", "gold", "config", "outputs", "pilot-analysis", "historical-analysis", "out"):
        parser.add_argument(f"--{arg}", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    if args.out.exists():
        raise FileExistsError(f"create-once output already exists: {args.out}")
    config = base.load_json(args.config)
    input_hashes = {name: base.sha256(getattr(args, name)) for name in ("public", "frozen_source", "gold", "config")}
    if input_hashes["public"] != PUBLIC_SHA256:
        raise ValueError("unexpected pilot public input")
    for name, key in (("frozen_source", "public/main_jobs.json"), ("gold", "evaluator/main_dataset.json")):
        if input_hashes[name] != config["frozen_source_sha256"][key]:
            raise ValueError("frozen source hash mismatch")
    previous_receipt = base.load_json(args.pilot_analysis / "analysis_receipt.json")
    if previous_receipt["status"] != "complete" or previous_receipt["public_sha256"] != PUBLIC_SHA256:
        raise ValueError("prior analysis incomplete or wrong public input")
    if base.sha256(args.pilot_analysis / "report.json") != previous_receipt["output_sha256"]["report.json"]:
        raise ValueError("prior verified report hash mismatch")
    package = base.load_json(args.public)
    jobs = base.validate_jobs([base.normalize_public_job(job) for job in package["jobs"]], base.load_json(args.gold),
                              n_per_split=config["n_per_split"], n_diagnostic_per_split=config["n_diagnostic_per_split"])
    base.validate_source_binding(jobs, base.load_json(args.frozen_source))
    all_rows, audits, historical = {}, {}, []
    for model in config["models"]:
        all_rows[model], audits[model] = base.validate_model_output(args.outputs / model, model, package, jobs, config, PUBLIC_SHA256)
        oldpath = args.historical_analysis / model / "main" / "automatic_graded_rows.json"
        input_hashes[f"historical_{model}"] = base.sha256(oldpath)
        historical.extend({"model": model, **row} for row in match_historical(jobs, base.load_json(oldpath), all_rows[model]))
    report = run_analysis(all_rows, config["analysis"]["bootstrap_samples"], config["analysis"]["bootstrap_seed"])
    historical_summaries = []
    for model in all_rows:
        for split in base.SPLITS:
            for condition in base.MENUS:
                rows = [r for r in historical if (r["model"], r["split"], r["condition"]) == (model, split, condition)]
                historical_summaries.append({"model": model, "split": split, "condition": condition, "n_questions": len(rows),
                    **{key + "_fraction": float(np.mean([r[key] for r in rows])) for key in ("historical_correct", "forced_game_correct", "same_wait_candidate_correct")}})
    args.out.mkdir(parents=True, exist_ok=False)
    for key in ("decomposition_records", "round_records", "policy_records"):
        base.write_csv(args.out / f"{key}.csv", report.pop(key))
    base.write_csv(args.out / "historical_matches.csv", historical)
    report.update(schema_version="imcqa-wait-decomposition-v1", evidence_scope="exploratory_development_only",
                  input_sha256=input_hashes, verified_pilot_report_sha256=base.sha256(args.pilot_analysis / "report.json"),
                  audits=audits, historical_summaries=historical_summaries,
                  definitions={"regret_identity": "same-WAIT hindsight minus native reward = wrong-answer penalty + forgone correct reward after wrong/PASS + delay cost on correct commitment",
                               "oracle": "earliest round with a correct same-WAIT A-D argmax, otherwise PASS; hindsight only, not deployable",
                               "fixed_round": "answer top A-D under unchanged WAIT prompt at selected round, ignoring E",
                               "visited": "round <= native first A-D round; all five rounds if terminal PASS",
                               "myopic": "first round with calibrated p*(correct_reward+1)-1 > 0; comparison is against PASS, ignores continuation value",
                               "calibration": "per model/menu two-parameter monotone logistic map; fixed L2=0.01; all 500 calibration states from 100 distinct questions equally weighted",
                               "fixed_policy_selection": "choose highest mean calibration reward among PASS and rounds1..5; ties prefer PASS, then earlier round",
                               "bootstrap": "2000 whole-question resamples, seed1; conditional on fitted calibration parameters, no fit uncertainty"},
                  limitations=["Post-hoc exploratory analysis; the selection split was inspected previously and is not a new confirmatory test.",
                    "All five prefixes are independent counterfactual contexts; only states through the native stopping round are visited.",
                    "Oracle availability is gold-assisted hindsight, not a feasible policy performance forecast.",
                    "One-dimensional calibration targets correctness pooled across rounds; calibration at individual rounds or under policy-induced selection is not guaranteed.",
                    "Positive EV against PASS is not an optimal-stopping calculation and does not estimate continuation value.",
                    "Candidate softmax values are not full-response sampling frequencies.",
                    "Historical comparisons bind exact questions/menus/prefixes but change prompt, output protocol, and numerical precision jointly.",
                    "Descriptive bootstrap intervals hold the calibration fit fixed and omit model-selection uncertainty and multiplicity adjustments."])
    base.write_json(args.out / "report.json", report)
    lines = ["IMCQA SAME-WAIT CANDIDATE DECOMPOSITION", "CPU-only; no new model inference.", ""]
    for row in report["decomposition_summaries"]:
        if row["split"] == "selection":
            lines.append(f"{row['model']} {row['condition']}: native reward {row['mean_native_reward']:.3f}; same-WAIT hindsight {row['mean_hindsight_reward']:.3f}; outcomes {row['native_outcome_counts']}; skipped correct visited {row['n_visited_correct_candidate_skipped']}; wrong early with later correct {row['n_wrong_early_with_later_correct']}.")
    lines.extend(["", *report["limitations"]])
    (args.out / "FINDINGS.txt").write_text("\n".join(lines) + "\n")
    base.write_json(args.out / "analysis_receipt.json", {"status": "complete", "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": time.monotonic() - started, "runtime": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
        "analyzer_sha256": base.sha256(Path(__file__)), "upstream_analyzer_sha256": base.sha256(Path(base.__file__)),
        "input_sha256": input_hashes, "output_sha256": {path.name: base.sha256(path) for path in sorted(args.out.iterdir()) if path.is_file()},
        "n_questions": 200, "n_model_question_menu_trajectories": 800, "n_model_inference_calls": 0,
        "same_wait_hindsight_upper_bound_all_trajectories": True, "additive_regret_identity_all_trajectories": True})
    print(json.dumps({"status": "complete", "out": str(args.out), "runtime_seconds": time.monotonic() - started}))


if __name__ == "__main__":
    main()
