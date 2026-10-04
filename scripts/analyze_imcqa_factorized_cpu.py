#!/usr/bin/env python3
"""CPU-only, frozen-method decomposition of answer elicitation and stopping.

Fits and policy choices use calibration questions only. The selection questions
were inspected previously; the resulting estimates are exploratory, not a new
confirmatory test. No model forward pass occurs here.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import time

import numpy as np
import scipy
from scipy.optimize import minimize
from scipy.special import expit, logit

from scripts import analyze_imcqa_protocol_pilot as protocol


PLAN_SHA256 = "2994d9f4db151e0c32dfe698e2ff454267486382d0b3d6c34ff9c239db30a6b8"
ARMS = ("plain", "forced", "wait")
ENSEMBLE = "plain_ensemble"
FIT_SOURCES = (*ARMS, ENSEMBLE)
EXTERNAL_METHODS = ("myopic", "calibration_selected_threshold", "calibration_selected_fixed")
FIXED_METHODS = ("pass", "fixed_1", "fixed_2", "fixed_3", "fixed_4", "fixed_5")


def states_from_rows(rows):
    """Retain the real factorial only; controls never enter calibration."""
    result = []
    seen = set()
    for row in rows:
        if row["block"] != "factorial":
            continue
        key = tuple(row[name] for name in ("qid", "condition", "round", "rotation", "arm"))
        if key in seen:
            raise ValueError("duplicate factorial state")
        seen.add(key)
        probabilities = row["canonical_answer_probabilities"]
        result.append({name: row[name] for name in ("qid", "group_id", "split", "condition", "round", "rotation", "fraction",
            "candidate_choice", "candidate_correct", "canonical_gold_option_id")} | {
                "answer_source": row["arm"], "candidate_confidence": max(probabilities.values()),
                "candidate_probabilities": probabilities,
                "native_commit": row["chosen_action"] in row["option_source_ids"]})
    return result


def cyclic_ensemble(states):
    """Average canonical probability vectors, not labels or correctness outcomes."""
    groups = defaultdict(list)
    for row in states:
        if row["answer_source"] == "plain":
            groups[row["qid"], row["condition"], row["round"]].append(row)
    result = []
    for _, rows in sorted(groups.items()):
        if sorted(row["rotation"] for row in rows) != list(range(4)):
            raise ValueError("cyclic ensemble requires all four distinct rotations")
        first = rows[0]
        for name in ("group_id", "split", "fraction", "canonical_gold_option_id"):
            if len({row[name] for row in rows}) != 1:
                raise ValueError("ensemble question identity differs")
        probabilities = {label: float(np.mean([row["candidate_probabilities"][label] for row in rows])) for label in protocol.OPTIONS}
        choice = max(protocol.OPTIONS, key=probabilities.get)
        result.append({**first, "answer_source": ENSEMBLE, "rotation": -1, "candidate_probabilities": probabilities,
            "candidate_choice": choice, "candidate_correct": choice == first["canonical_gold_option_id"],
            "candidate_confidence": probabilities[choice], "native_commit": None})
    return result


def trajectories(states):
    result = defaultdict(list)
    for row in states:
        result[row["answer_source"], row["split"], row["condition"], row["qid"], row["rotation"]].append(row)
    for values in result.values():
        values.sort(key=lambda row: row["round"])
        if [row["round"] for row in values] != list(range(1, 6)):
            raise ValueError("incomplete five-round candidate trajectory")
    return result


def fit_calibrator(rows, settings):
    """Fit one fixed monotone logistic map with question-balanced mean loss."""
    if not rows or any(row["split"] != "calibration" for row in rows):
        raise ValueError("calibrator accepts calibration data only")
    if len({row["answer_source"] for row in rows}) != 1 or len({row["condition"] for row in rows}) != 1:
        raise ValueError("calibrator cannot mix answer sources or menus")
    qids = sorted({row["qid"] for row in rows})
    per_question = Counter(row["qid"] for row in rows)
    expected = 5 if rows[0]["answer_source"] == ENSEMBLE else 20
    if set(per_question.values()) != {expected}:
        raise ValueError("calibration states must balance all rounds/orders within question")
    lower, upper = settings["feature_clip"]
    x = logit(np.clip([row["candidate_confidence"] for row in rows], lower, upper))
    y = np.asarray([row["candidate_correct"] for row in rows], dtype=float)
    initial = np.array([logit((float(y.sum()) + .5) / (len(y) + 1)), 0.])
    penalty = settings["l2"]
    def objective(theta):
        z = theta[0] + theta[1] * x
        residual = expit(z) - y
        return (float(np.mean(np.logaddexp(0, z) - y*z) + penalty*theta[1]**2/2),
                np.array([np.mean(residual), np.mean(residual*x) + penalty*theta[1]]))
    if len(set(y)) == 1:
        theta, method = initial, "smoothed_intercept_only_degenerate_labels"
    else:
        fitted = minimize(objective, initial, jac=True, method="L-BFGS-B", bounds=[(None, None), (0, None)],
            options={key: settings[key] for key in ("maxiter", "ftol", "gtol")})
        if not fitted.success or not np.all(np.isfinite(fitted.x)):
            raise ValueError("fixed calibration optimizer failed: " + fitted.message)
        theta, method = fitted.x, "monotone_regularized_logistic_correctness"
    return {"answer_source": rows[0]["answer_source"], "condition": rows[0]["condition"], "fit_split": "calibration",
        "fit_qids": qids, "n_fit_questions": len(qids), "n_fit_states": len(rows),
        "method": method, "intercept": float(theta[0]), "slope": float(theta[1]), "l2": penalty,
        "feature_clip": settings["feature_clip"], "objective_value": objective(theta)[0]}


def calibrated_probability(row, fit):
    value = float(logit(np.clip(row["candidate_confidence"], *fit["feature_clip"])))
    return float(expit(fit["intercept"] + fit["slope"] * value))


def stopping_round(trajectory, method, fit=None, *, threshold=None):
    """Return an observed commitment round or None for terminal PASS."""
    if method == "pass":
        return None
    if method.startswith("fixed_"):
        value = int(method.split("_")[1])
        if value not in range(1, 6):
            raise ValueError("invalid fixed round")
        return value
    if method == "native":
        if trajectory[0]["answer_source"] != "wait":
            raise ValueError("native stopping schedule must come from WAIT")
        return next((row["round"] for row in trajectory if row["native_commit"]), None)
    if fit is None:
        raise ValueError("external stopping requires a frozen calibrator")
    if method == "calibration_selected_fixed":
        return stopping_round(trajectory, fit["selected_fixed_policy"])
    if method == "calibration_selected_threshold":
        threshold = fit["selected_threshold"]
        if threshold is None:
            return None
        method = "threshold"
    if method == "myopic":
        return next((row["round"] for row in trajectory if calibrated_probability(row, fit) > 1 / (1 + protocol.REWARDS[row["round"] - 1])), None)
    if method == "threshold":
        if threshold is None or not 0 <= threshold <= 1:
            raise ValueError("numeric threshold required")
        return next((row["round"] for row in trajectory if calibrated_probability(row, fit) >= threshold), None)
    raise ValueError("unknown stop method")


def outcome(answer_trajectory, round_number, stop_source, stop_method):
    chosen = answer_trajectory[-1] if round_number is None else answer_trajectory[round_number - 1]
    committed = round_number is not None
    correct = bool(committed and chosen["candidate_correct"])
    return {name: chosen[name] for name in ("qid", "group_id", "split", "condition", "answer_source", "rotation")} | {
        "stop_source": stop_source, "stop_method": stop_method,
        "policy": f'{chosen["answer_source"]}|{stop_source}|{stop_method}',
        "committed": committed, "correct": correct, "wrong": committed and not correct, "terminal_pass": not committed,
        "round": round_number, "observed_round": chosen["round"], "actual_fraction": chosen["fraction"],
        "canonical_choice": chosen["candidate_choice"] if committed else None,
        "reward": (protocol.REWARDS[round_number - 1] if correct else -1.) if committed else 0.,
        "counterfactual_uncached_input_tokens": None}


def select_calibration_policies(trajectories_for_fit, fit, threshold_grid):
    """Select from the frozen grid using calibration episodic reward alone."""
    if not trajectories_for_fit or any(row["split"] != "calibration" for trajectory in trajectories_for_fit for row in trajectory):
        raise ValueError("policy selection accepts calibration trajectories only")
    def value(method, threshold=None):
        return float(np.mean([outcome(trajectory, stopping_round(trajectory, method, fit, threshold=threshold),
            fit["answer_source"], method)["reward"] for trajectory in trajectories_for_fit]))
    fixed = [{"policy": name, "mean_reward": value(name)} for name in FIXED_METHODS]
    best_fixed_value = max(row["mean_reward"] for row in fixed)
    selected_fixed = next(row["policy"] for row in fixed if best_fixed_value - row["mean_reward"] <= 1e-12)
    thresholds = [{"threshold": None, "mean_reward": 0.}] + [{"threshold": tau, "mean_reward": value("threshold", tau)} for tau in sorted(threshold_grid, reverse=True)]
    best_threshold_value = max(row["mean_reward"] for row in thresholds)
    selected_threshold = next(row["threshold"] for row in thresholds if best_threshold_value - row["mean_reward"] <= 1e-12)
    return {**fit, "selected_fixed_policy": selected_fixed, "selected_threshold": selected_threshold,
        "fixed_calibration_rewards": fixed, "threshold_calibration_rewards": thresholds,
        "selection_uses_only_calibration": True}


def freeze_fits(states, plan):
    """This function never accepts selection rows into a fitting operation."""
    calibration = [row for row in states if row["split"] == "calibration"]
    lookup = trajectories(calibration)
    fits = {}
    for source in FIT_SOURCES:
        for condition in protocol.MENUS:
            rows = [row for row in calibration if row["answer_source"] == source and row["condition"] == condition]
            fit = fit_calibrator(rows, plan["calibration"])
            if fit["n_fit_questions"] != plan["fit_questions"]:
                raise ValueError("calibration question count differs from frozen plan")
            selected = [values for key, values in lookup.items() if key[0] == source and key[2] == condition]
            fits[source, condition] = select_calibration_policies(selected, fit, plan["stop_policies"]["threshold_grid"])
    return fits


def evaluate_policies(states, fits):
    """Cross answer candidates with exactly reused observed stopping schedules."""
    lookup = trajectories(states)
    records, schedules = [], []
    for key, answers in lookup.items():
        source, split, condition, qid, rotation = key
        for method in FIXED_METHODS:
            records.append(outcome(answers, stopping_round(answers, method), "fixed", method))
        signal_sources = (ENSEMBLE,) if source == ENSEMBLE else ARMS
        for signal_source in signal_sources:
            signal = lookup[signal_source, split, condition, qid, rotation]
            fit = fits[signal_source, condition]
            for method in EXTERNAL_METHODS:
                stop = stopping_round(signal, method, fit)
                records.append(outcome(answers, stop, signal_source, method))
                if signal_source == source:
                    schedules.append({"split": split, "condition": condition, "qid": qid, "rotation": rotation,
                        "stop_source": signal_source, "stop_method": method, "round": stop})
        if source != ENSEMBLE:
            native = lookup["wait", split, condition, qid, rotation]
            stop = stopping_round(native, "native")
            records.append(outcome(answers, stop, "wait", "native"))
            if source == "wait":
                schedules.append({"split": split, "condition": condition, "qid": qid, "rotation": rotation,
                    "stop_source": "wait", "stop_method": "native", "round": stop})
    return records, schedules


def paired_summary(left, right, samples, seed):
    """Pair questions even when one pipeline has one episode and another four."""
    left_qids, left_totals = protocol.old.question_totals(left)
    right_qids, right_totals = protocol.old.question_totals(right)
    if left_qids != right_qids or len({row["condition"] for row in left + right}) != 1:
        raise ValueError("pipeline comparison must pair identical questions within one menu")
    if any(len(set(Counter(row["qid"] for row in values).values())) != 1 for values in (left, right)):
        raise ValueError("pipeline episodes must be balanced within each question cohort")
    indices = protocol.old.bootstrap_indices(len(left_qids), samples, seed)
    a = protocol.old.metrics_from_totals(left_totals[indices].sum(axis=1))
    b = protocol.old.metrics_from_totals(right_totals[indices].sum(axis=1))
    pa, pb = protocol.old.metric_vector(left), protocol.old.metric_vector(right)
    names = ("mean_reward", "coverage", "correct_fraction", "risk", "mean_observed_round")
    return {"n_questions": len(left_qids), "n_left_episodes": len(left), "n_right_episodes": len(right),
        "direction": "left minus right", "unit": "question; fixed fitted policies; all orders retained within question",
        "differences": {name: {"mean": pa[name] - pb[name] if pa[name] is not None and pb[name] is not None else None,
            **protocol.old.interval(a[name] - b[name])} for name in names}}


def calibration_metrics(rows, fit):
    raw = np.asarray([row["candidate_confidence"] for row in rows])
    calibrated = np.asarray([calibrated_probability(row, fit) for row in rows])
    y = np.asarray([row["candidate_correct"] for row in rows], dtype=float)
    result = {"n_questions": len({row["qid"] for row in rows}), "n_states": len(rows)}
    for name, p in (("raw", raw), ("calibrated", calibrated)):
        clipped = np.clip(p, 1e-12, 1-1e-12)
        result[name] = {"brier": float(np.mean((p-y)**2)),
                       "log_loss": float(-np.mean(y*np.log(clipped) + (1-y)*np.log1p(-clipped)))}
    return result


def summarize(records, states, fits, plan):
    samples, seed = plan["evaluation"]["bootstrap_samples"], plan["evaluation"]["bootstrap_seed"]
    cells = defaultdict(list)
    for row in records:
        cells[row["split"], row["condition"], row["answer_source"], row["stop_source"], row["stop_method"]].append(row)
    summaries = []
    for key, values in sorted(cells.items()):
        summaries.append({**dict(zip(("split", "condition", "answer_source", "stop_source", "stop_method"), key)),
            **protocol.old.summarize_policy(values, protocol.old.bootstrap_indices(len({row["qid"] for row in values}), samples, seed))})
    contrasts = []
    def add(split, condition, name, tier, left_key, right_key, identical_schedule=False):
        left, right = cells[split, condition, *left_key], cells[split, condition, *right_key]
        if identical_schedule:
            key = lambda row: (row["qid"], row["rotation"])
            if [(key(row), row["round"]) for row in sorted(left, key=key)] != [(key(row), row["round"]) for row in sorted(right, key=key)]:
                raise ValueError("primary answer-source contrast changed the stopping schedule")
        contrasts.append({"split": split, "condition": condition, "contrast": name, "tier": tier,
            "left": dict(zip(("answer_source", "stop_source", "stop_method"), left_key)),
            "right": dict(zip(("answer_source", "stop_source", "stop_method"), right_key)),
            "identical_stopping_schedule": identical_schedule, **paired_summary(left, right, samples, seed)})
    for split in protocol.SPLITS:
        for condition in protocol.MENUS:
            for method in ("native", "myopic", "calibration_selected_threshold"):
                add(split, condition, f"plain_minus_wait_answers_same_wait_{method}_schedule", "primary",
                    ("plain", "wait", method), ("wait", "wait", method), True)
            for method in EXTERNAL_METHODS:
                add(split, condition, f"plain_minus_wait_own_{method}_pipeline", "secondary",
                    ("plain", "plain", method), ("wait", "wait", method))
                for source in ("plain", "wait"):
                    add(split, condition, f"{source}_own_{method}_minus_native_wait", "secondary",
                        (source, source, method), ("wait", "wait", "native"))
            for method in (*FIXED_METHODS, *EXTERNAL_METHODS):
                left_stop = "fixed" if method in FIXED_METHODS else ENSEMBLE
                right_stop = "fixed" if method in FIXED_METHODS else "plain"
                add(split, condition, f"plain_ensemble_minus_order_average_plain_{method}_pipeline", "ensemble_complete_pipeline",
                    (ENSEMBLE, left_stop, method), ("plain", right_stop, method))
    metrics = []
    for source in FIT_SOURCES:
        for split in protocol.SPLITS:
            for condition in protocol.MENUS:
                rows = [row for row in states if (row["answer_source"], row["split"], row["condition"]) == (source, split, condition)]
                metrics.append({"answer_source": source, "split": split, "condition": condition,
                    **calibration_metrics(rows, fits[source, condition])})
    return {"policy_summaries": summaries, "paired_contrasts": contrasts, "calibration_metrics": metrics}


def load_verified_rows(args, plan):
    """Revalidate immutable source, old reuse, and complete 7B numerical evidence."""
    report = protocol.old.load_json(args.protocol_analysis / "report.json")
    receipt = protocol.old.load_json(args.protocol_analysis / "analysis_receipt.json")
    if (report["schema_version"] != "imcqa-protocol-partial-analysis-v1" or report["status"] != "partial"
        or report["validated_models"] != ["qwen7b"] or receipt["status"] != "partial"
        or receipt["output_sha256"]["report.json"] != protocol.old.sha256(args.protocol_analysis / "report.json")):
        raise ValueError("upstream verified partial analysis identity differs")
    if protocol.old.sha256(args.public) != plan["source_public_sha256"] or protocol.old.sha256(args.prior_public) != plan["prior_public_sha256"]:
        raise ValueError("frozen public package differs")
    config = protocol.old.load_json(args.protocol_config)
    if protocol.old.sha256(args.protocol_config) != protocol.old.sha256(Path(__file__).resolve().parents[1] / "configs/imcqa_protocol_pilot.json"):
        raise ValueError("protocol config differs from inference-source-bound config")
    for path, key in ((args.frozen_source, "public/main_jobs.json"), (args.gold, "evaluator/main_dataset.json")):
        if protocol.old.sha256(path) != config["frozen_source_sha256"][key]:
            raise ValueError("frozen source/gold differs")
    public, prior = protocol.old.load_json(args.public), protocol.old.load_json(args.prior_public)
    dataset, source = protocol.old.load_json(args.gold), protocol.old.load_json(args.frozen_source)
    jobs = protocol.validate_public(public, dataset, source, prior, config)
    prior_config = protocol.old.load_json(Path(__file__).resolve().parents[1] / "configs/imcqa_wait_pilot.json")
    prior_jobs = protocol.old.validate_jobs([protocol.old.normalize_public_job(job) for job in prior["jobs"]], dataset,
        n_per_split=prior_config["n_per_split"], n_diagnostic_per_split=prior_config["n_diagnostic_per_split"])
    protocol.old.validate_source_binding(prior_jobs, source)
    del dataset, source
    previous, prior_audit = protocol.old.validate_model_output(args.prior_outputs / "qwen7b", "qwen7b", prior, prior_jobs, prior_config, plan["prior_public_sha256"])
    rows, audit = protocol.validate_new_model(args.outputs / "qwen7b", "qwen7b", public, jobs, config,
        plan["source_public_sha256"], args.prior_outputs / "qwen7b", previous)
    if audit["scores_sha256"] != report["audits"]["qwen7b"]["scores_sha256"]:
        raise ValueError("new raw scores differ from verified partial report")
    return rows, {"new": audit, "prior": prior_audit, "verified_report_sha256": protocol.old.sha256(args.protocol_analysis / "report.json")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "public", "prior-public", "frozen-source", "gold", "protocol-config", "protocol-analysis", "prior-outputs", "outputs", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    if protocol.old.sha256(args.plan) != PLAN_SHA256:
        raise ValueError("CPU analysis plan differs from the pre-fit frozen hash")
    plan = protocol.old.load_json(args.plan)
    rows, audits = load_verified_rows(args, plan)
    states = states_from_rows(rows)
    if len(states) != plan["n_states"] or len({row["qid"] for row in states}) != plan["n_questions"]:
        raise ValueError("factorial state/question coverage differs")
    for split, expected_count in (("calibration", plan["fit_questions"]), ("selection", plan["evaluation_questions"])):
        if len({row["qid"] for row in states if row["split"] == split}) != expected_count:
            raise ValueError("split coverage differs")
    states += cyclic_ensemble(states)
    fits = freeze_fits(states, plan)
    args.out.mkdir(parents=True, exist_ok=False)
    fit_path = args.out / "fitted_parameters.json"
    protocol.old.write_json(fit_path, {"status": "frozen_before_selection_policy_evaluation", "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "plan_sha256": PLAN_SHA256, "parameters": list(fits.values()), "fit_split": "calibration",
        "selection_outcomes_used_for_fitting": False, "prior_selection_results_were_already_inspected": True})
    # No selection policy outcomes are evaluated before the parameter file above.
    records, schedules = evaluate_policies(states, fits)
    report = summarize(records, states, fits, plan)
    protocol.old.write_csv(args.out / "policy_records.csv", records)
    protocol.old.write_csv(args.out / "stopping_schedules.csv", schedules)
    state_records = [{key: value for key, value in row.items() if key != "candidate_probabilities"} | {
        "calibrated_probability": calibrated_probability(row, fits[row["answer_source"], row["condition"]])} for row in states]
    protocol.old.write_csv(args.out / "candidate_states.csv", state_records)
    report.update(schema_version="imcqa-factorized-cpu-analysis-v1", status="complete", model="qwen7b",
        evidence_scope="exploratory_posthoc_development_analysis", n_questions=40, n_calibration_questions=20, n_selection_questions=20,
        n_verified_factorial_states=4800, n_derived_ensemble_states=400, n_model_forward_passes=0,
        plan=plan, plan_sha256=PLAN_SHA256, fitted_parameters_sha256=protocol.old.sha256(fit_path),
        fits=list(fits.values()), audits=audits, limitations=plan["limitations"])
    protocol.old.write_json(args.out / "report.json", report)
    lines = ["FACTORIZED ANSWER/STOPPING DEVELOPMENT ANALYSIS", "CPU only: zero new model forward passes.",
        "Methods fixed before this policy fitting, after prior selection outcomes had been inspected.", ""]
    for contrast in report["paired_contrasts"]:
        if contrast["split"] == "selection" and contrast["tier"] == "primary":
            value = contrast["differences"]["mean_reward"]
            lines.append(f"{contrast['condition']} / {contrast['contrast']}: reward difference={value['mean']:.4f}, conditional question-bootstrap CI={value['ci95']}.")
    lines += ["", *plan["limitations"]]
    (args.out / "FINDINGS.txt").write_text("\n".join(lines) + "\n")
    protocol.old.write_json(args.out / "analysis_receipt.json", {"status": "complete", "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": time.monotonic() - started, "n_model_forward_passes": 0, "plan_sha256": PLAN_SHA256,
        "analyzer_sha256": protocol.old.sha256(Path(__file__)), "upstream_analyzer_sha256": protocol.old.sha256(Path(protocol.__file__)),
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
        "output_sha256": {path.name: protocol.old.sha256(path) for path in sorted(args.out.iterdir()) if path.is_file()}})
    print(json.dumps({"status": "complete", "out": str(args.out), "n_model_forward_passes": 0, "runtime_seconds": time.monotonic()-started}))


if __name__ == "__main__":
    main()
