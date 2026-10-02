"""Paired, question-weighted analysis for Jane's incremental evaluation protocol.

This module analyzes *graded* traces. It does not generate or grade answers and
never treats synthetic or retrieval controls as evidence about language models.
Calibration and threshold selection use distinct development splits. Bootstrap
intervals resample complete groups and condition on those fitted choices.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations
import math
from numbers import Real
from typing import Any, Iterable, Mapping

import numpy as np
from sklearn.isotonic import IsotonicRegression

_SCHEMA = "jane-analysis-v1"
_SPLITS = ("calibration", "selection", "test")
_UNRESOLVED = {"clarification_required", "needs_review"}


def _finite(value: Any, label: str, *, lower: float = 0, upper: float = 1) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a finite number")
    number = float(value)
    if not math.isfinite(number) or not lower <= number <= upper:
        raise ValueError(f"{label} must be finite in [{lower}, {upper}]")
    return number


def arm_id(row: Mapping[str, Any]) -> str:
    """Return the stable analysis arm identifier for a graded row."""
    return "oe" if row["format"] == "oe" else f"mc:{row['condition']}:{row['menu_id']}"


def _validate(rows: Iterable[Mapping[str, Any]]) -> tuple[list[dict], dict, dict]:
    if isinstance(rows, (str, bytes, dict)):
        raise ValueError("rows must be a nonempty sequence of graded records")
    records = [dict(row) for row in rows]
    if not records:
        raise ValueError("rows must be nonempty")
    required = {
        "job_id", "qid", "group_id", "split", "format", "condition", "menu_id",
        "prefix_id", "fraction", "status", "confidence", "grade", "correct",
    }
    jobs: set[str] = set()
    qmeta: dict[str, dict] = {}
    group_splits: dict[str, str] = {}
    arms: dict[str, dict] = {}
    trajectories: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in records:
        missing = required - row.keys()
        if missing:
            raise ValueError(f"graded row missing fields: {sorted(missing)}")
        for key in ("job_id", "qid", "group_id", "prefix_id", "condition"):
            if not isinstance(row[key], str) or not row[key].strip():
                raise ValueError(f"{key} must be a nonempty string")
        if row["job_id"] in jobs:
            raise ValueError("duplicate job_id")
        jobs.add(row["job_id"])
        if row["split"] not in _SPLITS:
            raise ValueError("split must be calibration, selection, or test")
        if row["format"] not in ("mc", "oe"):
            raise ValueError("format must be mc or oe")
        if row["format"] == "oe":
            if row["menu_id"] is not None or row["condition"] != "oe":
                raise ValueError("OE requires condition=oe and menu_id=null")
        elif not isinstance(row["menu_id"], str) or not row["menu_id"].strip():
            raise ValueError("MC menu_id must be a nonempty string")
        fraction = _finite(row["fraction"], "fraction")
        if fraction <= 0:
            raise ValueError("fraction must be positive")
        row["fraction"] = fraction
        status, grade, correct = row["status"], row["grade"], row["correct"]
        if grade in _UNRESOLVED:
            raise ValueError(f"unresolved answer requires adjudication: {row['job_id']} ({grade})")
        if status == "answer":
            if grade not in ("accepted", "rejected"):
                raise ValueError("answer status requires accepted/rejected grade")
            if type(correct) is not bool or correct != (grade == "accepted"):
                raise ValueError("grade and correct disagree")
            row["confidence"] = _finite(row["confidence"], "confidence")
        elif status in ("abstain", "invalid"):
            if grade != status or (correct is not None and correct is not False):
                raise ValueError("non-answer status/grade/correct disagree")
            if row["confidence"] is not None:
                raise ValueError("abstain/invalid confidence must be null")
            row["correct"] = False
        else:
            raise ValueError("status must be answer, abstain, or invalid")
        qid, group, split = row["qid"], row["group_id"], row["split"]
        metadata = {"group_id": group, "split": split}
        if qid in qmeta and qmeta[qid] != metadata:
            raise ValueError("a question cannot cross groups or splits")
        qmeta[qid] = metadata
        if group in group_splits and group_splits[group] != split:
            raise ValueError("a group cannot cross calibration/selection/test splits")
        group_splits[group] = split
        arm = arm_id(row)
        identity = {"format": row["format"], "condition": row["condition"], "menu_id": row["menu_id"]}
        if arm in arms and arms[arm] != identity:
            raise ValueError("ambiguous arm identifier; condition/menu IDs contain colliding delimiters")
        arms[arm] = identity
        trajectories[qid][arm].append(row)
    if set(group_splits.values()) != set(_SPLITS):
        raise ValueError("analysis requires calibration, selection, and test splits")
    if "oe" not in arms or len(arms) < 2:
        raise ValueError("analysis requires OE and at least one MC arm")
    for qid, by_arm in trajectories.items():
        if set(by_arm) != set(arms):
            raise ValueError(f"incomplete arm coverage for question {qid}")
        reference = None
        for arm, trajectory in by_arm.items():
            trajectory.sort(key=lambda row: (row["fraction"], row["prefix_id"]))
            signature = [(row["prefix_id"], row["fraction"]) for row in trajectory]
            if len({key for key, _ in signature}) != len(signature):
                raise ValueError(f"duplicate prefix in trajectory {qid}/{arm}")
            fractions = [fraction for _, fraction in signature]
            if any(left >= right for left, right in zip(fractions, fractions[1:])):
                raise ValueError("trajectory fractions must strictly increase")
            if fractions[-1] != 1.0:
                raise ValueError("complete trajectory must include the full question (fraction=1)")
            if reference is not None and signature != reference:
                raise ValueError(f"incomplete matching prefix trajectory for {qid}/{arm}")
            reference = signature
    records.sort(key=lambda row: (row["split"], row["qid"], arm_id(row), row["fraction"]))
    return records, dict(arms), dict(trajectories)


def fit_calibrator(trajectories: Iterable[list[dict]]) -> dict:
    """Fit answer-conditional isotonic calibration with unit question weights.

    Parameters
    ----------
    trajectories : iterable of lists of dict
        Calibration-only trajectories. Each question's confidence-bearing answer
        prefixes have total fitting weight one; unanswered questions are counted
        but cannot contribute invented confidence values.

    Returns
    -------
    dict
        Serializable interpolation knots and explicit fitting denominators.
    """
    x, y, weights = [], [], []
    n_questions = n_contributing = 0
    for trajectory in trajectories:
        if not trajectory or any(row["split"] != "calibration" for row in trajectory):
            raise ValueError("calibrator fitting accepts calibration trajectories only")
        n_questions += 1
        answers = [row for row in trajectory if row["status"] == "answer"]
        if not answers:
            continue
        n_contributing += 1
        for row in answers:
            x.append(row["confidence"])
            y.append(int(row["correct"]))
            weights.append(1 / len(answers))
    if not x:
        raise ValueError("calibration arm has no resolved answer confidences")
    fitted = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(x, y, sample_weight=weights)
    return {
        "method": "isotonic", "out_of_bounds": "clip",
        "x": fitted.X_thresholds_.tolist(), "y": fitted.y_thresholds_.tolist(),
        "n_questions": n_questions, "n_contributing_questions": n_contributing,
        "n_questions_without_answer_confidence": n_questions - n_contributing,
        "n_answer_prefixes": len(x), "total_fitting_weight": float(sum(weights)),
        "weighting": "unit weight per question over its resolved answer prefixes",
        "fit_split": "calibration",
    }


def calibrated_probability(calibrator: Mapping[str, Any], confidence: float) -> float:
    """Apply saved monotone knots without fitting or consulting any labels."""
    return float(np.interp(confidence, calibrator["x"], calibrator["y"]))


def first_crossing(trajectory: list[dict], calibrator: Mapping[str, Any], threshold: Mapping[str, Any]) -> dict | None:
    """Return the earliest eligible answer crossing a frozen threshold, or None."""
    if threshold["mode"] == "never":
        return None
    if threshold["mode"] != "threshold":
        raise ValueError("threshold mode must be threshold or never")
    value = _finite(threshold["value"], "threshold")
    for row in sorted(trajectory, key=lambda row: row["fraction"]):
        if row["status"] == "answer":
            probability = calibrated_probability(calibrator, row["confidence"])
            if probability >= value:
                return {**row, "calibrated_confidence": probability}
    return None


def _policy_questions(trajectories: Mapping[str, list[dict]], calibrator: dict, threshold: dict) -> dict:
    results = {}
    for qid, trajectory in sorted(trajectories.items()):
        selected = first_crossing(trajectory, calibrator, threshold)
        results[qid] = {
            "group_id": trajectory[0]["group_id"], "selected": selected,
            "n_invalid_prefixes": sum(row["status"] == "invalid" for row in trajectory),
            "n_abstain_prefixes": sum(row["status"] == "abstain" for row in trajectory),
            "n_prefixes": len(trajectory),
        }
    return results


def _policy_metrics(question_records: list[dict], *, calibration_bins: bool = True) -> dict:
    n_questions = len(question_records)
    selected = [item["selected"] for item in question_records if item["selected"] is not None]
    n_commit = len(selected)
    correct = sum(bool(row["correct"]) for row in selected)
    errors = n_commit - correct
    probabilities = np.asarray([row["calibrated_confidence"] for row in selected], dtype=float)
    outcomes = np.asarray([row["correct"] for row in selected], dtype=float)
    result = {
        "n_questions": n_questions, "n_committed": n_commit, "n_correct_commits": correct,
        "n_incorrect_commits": errors, "n_no_commit": n_questions - n_commit,
        "risk": errors / n_commit if n_commit else None,
        "coverage": n_commit / n_questions if n_questions else None,
        "mean_commitment_fraction": float(np.mean([row["fraction"] for row in selected])) if n_commit else None,
        "correct_commit_fraction": correct / n_questions if n_questions else None,
        "mean_utility": (correct - errors) / n_questions if n_questions else None,
        "n_invalid_prefixes": sum(item["n_invalid_prefixes"] for item in question_records),
        "n_abstain_prefixes": sum(item["n_abstain_prefixes"] for item in question_records),
        "n_prefixes": sum(item["n_prefixes"] for item in question_records),
        "selected_point_brier": float(np.mean((probabilities - outcomes) ** 2)) if n_commit else None,
        "selected_point_mean_confidence": float(np.mean(probabilities)) if n_commit else None,
        "selected_point_accuracy": correct / n_commit if n_commit else None,
        "selected_point_calibration_gap": float(np.mean(probabilities - outcomes)) if n_commit else None,
    }
    if calibration_bins:
        bins = []
        ece = 0.0
        for index in range(10):
            mask = (probabilities >= index / 10) & ((probabilities < (index + 1) / 10) if index < 9 else (probabilities <= 1))
            count = int(np.sum(mask))
            mean_confidence = float(np.mean(probabilities[mask])) if count else None
            accuracy = float(np.mean(outcomes[mask])) if count else None
            if count:
                ece += count / n_commit * abs(mean_confidence - accuracy)
            bins.append({"lower": index / 10, "upper": (index + 1) / 10,
                         "n_committed": count, "mean_confidence": mean_confidence, "accuracy": accuracy})
        result["selected_point_calibration_bins"] = bins
        result["selected_point_ece_10_equal_width_bins"] = ece if n_commit else None
    return result


def select_threshold(trajectories: Mapping[str, list[dict]], calibrator: dict, risk_budget: float) -> dict:
    """Select on distinct selection questions using the specified deterministic ties.

    Maximize observed coverage subject to empirical selective error <= budget,
    then minimize mean commitment fraction, then maximize the numeric threshold.
    If no nonempty feasible policy exists, return the explicit never candidate.
    This empirical selection is not a statistical risk guarantee.
    """
    risk_budget = _finite(risk_budget, "risk_budget")
    if not trajectories or any(not trajectory or any(row["split"] != "selection" for row in trajectory)
                               for trajectory in trajectories.values()):
        raise ValueError("threshold fitting accepts selection trajectories only")
    candidates = {0.0, 1.0}
    for trajectory in trajectories.values():
        for row in trajectory:
            if row["status"] == "answer":
                candidates.add(calibrated_probability(calibrator, row["confidence"]))
    never = {"mode": "never", "value": None}
    best_threshold = never
    best_metrics = _policy_metrics(list(_policy_questions(trajectories, calibrator, never).values()))
    best_key = (-1, -math.inf, -math.inf)
    for value in sorted(candidates):
        threshold = {"mode": "threshold", "value": value}
        metrics = _policy_metrics(list(_policy_questions(trajectories, calibrator, threshold).values()))
        if metrics["n_committed"] and metrics["risk"] <= risk_budget:
            key = (metrics["n_committed"], -metrics["mean_commitment_fraction"], value)
            if key > best_key:
                best_key, best_threshold, best_metrics = key, threshold, metrics
    return {"threshold": best_threshold, "selection_metrics": best_metrics,
            "n_numeric_candidates": len(candidates), "explicit_never_candidate": True,
            "risk_budget": risk_budget, "selection_split": "selection",
            "guarantee": "none; empirical exploratory threshold selection"}


class _GroupBootstrap:
    """Shared group draws preserve pairing across arms and policies."""

    def __init__(self, qgroups: Mapping[str, str], samples: int, seed: int):
        self.qgroups = dict(qgroups)
        self.groups = sorted(set(qgroups.values()))
        self.samples = samples
        self.draws = (np.random.default_rng(seed).integers(0, len(self.groups), size=(samples, len(self.groups)))
                      if samples and len(self.groups) >= 2 else [])
        self.by_group = {group: sorted(qid for qid, assigned in qgroups.items() if assigned == group) for group in self.groups}

    def qid_draws(self):
        for draw in self.draws:
            yield [qid for index in draw for qid in self.by_group[self.groups[int(index)]]]

    @staticmethod
    def interval(values: list[float]) -> list[float] | None:
        return [float(value) for value in np.quantile(values, [.025, .975])] if values else None

    def mean_interval(self, values: Mapping[str, float]) -> tuple[list[float] | None, int]:
        estimates = []
        # A subset present in only one independent group has no estimable CI.
        if len({self.qgroups[qid] for qid in values}) < 2:
            return None, 0
        for draw in self.qid_draws():
            observed = [values[qid] for qid in draw if qid in values]
            if observed:
                estimates.append(float(np.mean(observed)))
        return self.interval(estimates), len(estimates)


def _summary(values: Mapping[str, float], prefix_counts: Mapping[str, int], bootstrap: _GroupBootstrap,
             *, name: str = "accuracy") -> dict:
    interval, n_valid = bootstrap.mean_interval(values)
    return {name: float(np.mean(list(values.values()))) if values else None,
            "n_questions": len(values), "n_groups": len({bootstrap.qgroups[qid] for qid in values}),
            "n_prefixes": sum(prefix_counts.values()), "ci95": interval,
            "bootstrap_valid_replicates": n_valid}


def _question_accuracy(trajectories: Mapping[str, list[dict]], *, bin_index: int | None = None,
                       early: bool = False, full: bool = False) -> tuple[dict, dict]:
    values, counts = {}, {}
    for qid, trajectory in trajectories.items():
        selected = [row for row in trajectory if (not early or row["fraction"] <= .2)
                    and (not full or row["fraction"] == 1.0)
                    and (bin_index is None or min(9, math.ceil(row["fraction"] * 10) - 1) == bin_index)]
        if selected:
            values[qid] = sum(bool(row["correct"]) for row in selected) / len(selected)
            counts[qid] = len(selected)
    return values, counts


def _high_confidence(trajectories: Mapping[str, list[dict]], bootstrap: _GroupBootstrap,
                     *, fraction_upper: float = 1.0) -> dict:
    counts = {}
    for qid, trajectory in trajectories.items():
        selected = [row for row in trajectory if row["status"] == "answer" and row["confidence"] > .99
                    and row["fraction"] <= fraction_upper]
        if selected:
            counts[qid] = (len(selected), sum(not row["correct"] for row in selected))
    denominator = sum(count[0] for count in counts.values())
    numerator = sum(count[1] for count in counts.values())
    estimates = []
    groups = {bootstrap.qgroups[qid] for qid in counts}
    if len(groups) >= 2:
        for draw in bootstrap.qid_draws():
            sampled = [counts[qid] for qid in draw if qid in counts]
            total = sum(count[0] for count in sampled)
            if total:
                estimates.append(sum(count[1] for count in sampled) / total)
    return {"raw_confidence_threshold": .99, "comparison": ">",
            "scope": "test resolved answer prefixes within stated fraction bound; prefix-weighted diagnostic",
            "fraction_upper_inclusive": fraction_upper,
            "n_predictions": denominator, "n_incorrect": numerator,
            "error_rate": numerator / denominator if denominator else None,
            "n_questions": len(counts), "n_groups": len(groups),
            "ci95": bootstrap.interval(estimates), "bootstrap_valid_replicates": len(estimates)}


def _policy_report(trajectories: Mapping[str, list[dict]], calibrator: dict, threshold: dict,
                   bootstrap: _GroupBootstrap, risk_budget: float) -> dict:
    question_records = _policy_questions(trajectories, calibrator, threshold)
    metrics = _policy_metrics(list(question_records.values()))
    n_committing_groups = len({item["group_id"] for item in question_records.values() if item["selected"] is not None})
    conditional_metrics = {"risk", "mean_commitment_fraction", "selected_point_brier", "selected_point_calibration_gap"}
    scalar_keys = ("risk", "coverage", "mean_commitment_fraction", "correct_commit_fraction", "mean_utility",
                   "selected_point_brier", "selected_point_calibration_gap")
    intervals = {key: [] for key in scalar_keys}
    for draw in bootstrap.qid_draws():
        sample = _policy_metrics([question_records[qid] for qid in draw], calibration_bins=False)
        for key in scalar_keys:
            if sample[key] is not None and (key not in conditional_metrics or n_committing_groups >= 2):
                intervals[key].append(sample[key])
    metrics["ci95"] = {key: bootstrap.interval(values) for key, values in intervals.items()}
    metrics["bootstrap_valid_replicates"] = {key: len(values) for key, values in intervals.items()}
    metrics["observed_test_budget_exceeded"] = (metrics["risk"] > risk_budget if metrics["risk"] is not None else None)
    metrics["n_groups"] = len({row["group_id"] for row in question_records.values()})
    metrics["n_committing_groups"] = n_committing_groups
    metrics["conditional_interval_status"] = ("available when resampling enabled" if n_committing_groups >= 2
                                               else "unavailable: fewer than two committing groups")
    # Export selections for inspection without copying answer text into the report.
    metrics["commitments"] = [{"qid": qid, "group_id": item["group_id"],
                               "prefix_id": item["selected"]["prefix_id"] if item["selected"] else None,
                               "fraction": item["selected"]["fraction"] if item["selected"] else None,
                               "correct": item["selected"]["correct"] if item["selected"] else None,
                               "calibrated_confidence": item["selected"]["calibrated_confidence"] if item["selected"] else None}
                              for qid, item in question_records.items()]
    return metrics


def _policy_comparison(name_a: str, name_b: str, questions_a: dict, questions_b: dict,
                       bootstrap: _GroupBootstrap) -> dict:
    """Paired differences with joint resampling and explicit conditioning sets."""
    if set(questions_a) != set(questions_b):
        raise ValueError("policy comparison requires identical question coverage")
    keys = ("risk", "coverage", "mean_commitment_fraction", "correct_commit_fraction", "mean_utility")
    point_a = _policy_metrics(list(questions_a.values()), calibration_bins=False)
    point_b = _policy_metrics(list(questions_b.values()), calibration_bins=False)
    samples = {key: [] for key in keys}
    n_committing_groups_a = len({item["group_id"] for item in questions_a.values() if item["selected"] is not None})
    n_committing_groups_b = len({item["group_id"] for item in questions_b.values() if item["selected"] is not None})
    conditional_metrics = {"risk", "mean_commitment_fraction"}
    for draw in bootstrap.qid_draws():
        metric_a = _policy_metrics([questions_a[qid] for qid in draw], calibration_bins=False)
        metric_b = _policy_metrics([questions_b[qid] for qid in draw], calibration_bins=False)
        for key in keys:
            if (metric_a[key] is not None and metric_b[key] is not None
                    and (key not in conditional_metrics or min(n_committing_groups_a, n_committing_groups_b) >= 2)):
                samples[key].append(metric_a[key] - metric_b[key])
    joint = Counter()
    for qid in questions_a:
        selected_a = questions_a[qid]["selected"] is not None
        selected_b = questions_b[qid]["selected"] is not None
        joint["both_commit" if selected_a and selected_b else "a_only" if selected_a else "b_only" if selected_b else "neither"] += 1
    return {
        "policy_a": name_a, "policy_b": name_b, "difference_direction": "policy_a minus policy_b",
        "n_questions": len(questions_a),
        "n_groups": len({record["group_id"] for record in questions_a.values()}),
        "n_committing_groups_a": n_committing_groups_a, "n_committing_groups_b": n_committing_groups_b,
        "joint_commitment_counts": {key: joint[key] for key in ("both_commit", "a_only", "b_only", "neither")},
        "differences": {key: {"estimate": point_a[key] - point_b[key] if point_a[key] is not None and point_b[key] is not None else None,
                               "ci95": bootstrap.interval(samples[key]),
                               "bootstrap_valid_replicates": len(samples[key])} for key in keys},
        "conditioning": "risk and mean commitment fraction compare each policy's own committing population; they are not restricted to questions both policies answer",
        "resampling": "identical whole-group draws for both policies; calibrators and thresholds frozen",
    }


def analyze(rows: Iterable[Mapping[str, Any]], *, risk_budget: float = .1,
            bootstrap_samples: int = 500, seed: int = 1) -> dict:
    """Analyze complete paired graded traces without using held-out labels to fit.

    Parameters
    ----------
    rows : iterable of dict
        Fully graded rows from ``qb_data.jane_paired.grade_predictions``.
    risk_budget : float
        Empirical development risk constraint, not a certified error guarantee.
    bootstrap_samples : int
        Group-clustered percentile replicates, conditional on fitted choices.
        Zero disables intervals. Fewer than two groups cannot provide an interval.
    seed : int
        Explicit nonnegative resampling seed.

    Returns
    -------
    dict
        JSON-serializable calibration evidence, held-out RQ1/RQ2 curves and RQ3
        first-crossing transfer metrics. All zero-denominator estimates are null.
    """
    risk_budget = _finite(risk_budget, "risk_budget")
    if type(bootstrap_samples) is not int or bootstrap_samples < 0:
        raise ValueError("bootstrap_samples must be a nonnegative integer")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    records, arms, trajectories = _validate(rows)
    arm_names = ["oe"] + sorted(arm for arm in arms if arm != "oe")
    split_data = {
        split: {arm: {qid: by_arm[arm] for qid, by_arm in sorted(trajectories.items())
                      if by_arm[arm][0]["split"] == split} for arm in arm_names}
        for split in _SPLITS
    }
    test = split_data["test"]
    bootstrap = _GroupBootstrap({qid: trajectory[0]["group_id"] for qid, trajectory in test["oe"].items()}, bootstrap_samples, seed)
    calibrators = {arm: fit_calibrator(split_data["calibration"][arm].values()) for arm in arm_names}
    selections = {arm: select_threshold(split_data["selection"][arm], calibrators[arm], risk_budget) for arm in arm_names}
    counts = {}
    for split in _SPLITS:
        questions = split_data[split]["oe"]
        counts[split] = {"n_questions": len(questions),
                         "n_groups": len({trajectory[0]["group_id"] for trajectory in questions.values()}),
                         "n_prefixes_per_arm": sum(len(trajectory) for trajectory in questions.values()),
                         "n_arms": len(arms),
                         "n_rows": sum(len(trajectory) for arm in arm_names for trajectory in split_data[split][arm].values())}
    rq1_arms, accuracy_values = {}, {}
    for arm in arm_names:
        early_values, early_counts = _question_accuracy(test[arm], early=True)
        bins, bin_values = [], []
        for index in range(10):
            values, prefix_counts = _question_accuracy(test[arm], bin_index=index)
            bins.append({"bin": index + 1, "lower": index / 10, "upper": (index + 1) / 10,
                         **_summary(values, prefix_counts, bootstrap)})
            bin_values.append((values, prefix_counts))
        all_values, all_counts = _question_accuracy(test[arm])
        full_values, full_counts = _question_accuracy(test[arm], full=True)
        rq1_arms[arm] = {**arms[arm], "curve": bins,
                         "early_accuracy": {"fraction_upper_inclusive": .2, **_summary(early_values, early_counts, bootstrap)},
                         "all_prefix_accuracy": _summary(all_values, all_counts, bootstrap),
                         "full_question_accuracy": _summary(full_values, full_counts, bootstrap),
                         "early_high_confidence_errors": _high_confidence(test[arm], bootstrap, fraction_upper=.2),
                         "high_confidence_errors": _high_confidence(test[arm], bootstrap)}
        accuracy_values[arm] = {"early": (early_values, early_counts), "bins": bin_values}
    contrasts = []
    pairs = [(arm, "oe") for arm in arm_names if arm != "oe"] + list(combinations(arm_names[1:], 2))
    for arm_a, arm_b in pairs:
        def contrast(values_a, values_b):
            av, ac = values_a
            bv, bc = values_b
            if set(av) != set(bv) or ac != bc:
                raise ValueError("paired comparison lost matched prefixes")
            return _summary({qid: av[qid] - bv[qid] for qid in av}, ac, bootstrap, name="accuracy_difference")
        contrasts.append({"arm_a": arm_a, "arm_b": arm_b, "difference_direction": "arm_a minus arm_b",
                          "early_accuracy": contrast(accuracy_values[arm_a]["early"], accuracy_values[arm_b]["early"]),
                          "curve": [{"bin": index + 1, "lower": index / 10, "upper": (index + 1) / 10,
                                     **contrast(accuracy_values[arm_a]["bins"][index], accuracy_values[arm_b]["bins"][index])}
                                    for index in range(10)]})
    rq3_arms = {}
    for mc_arm in arm_names[1:]:
        mc_threshold, oe_threshold = selections[mc_arm]["threshold"], selections["oe"]["threshold"]
        configurations = {
            "mc_selected_on_mc": (mc_arm, mc_arm, mc_threshold, 0),
            "zero_oe_label_transfer": ("oe", mc_arm, mc_threshold, 0),
            "oe_calibrated_mc_threshold": ("oe", "oe", mc_threshold, counts["calibration"]["n_questions"]),
            "oe_selected_on_oe": ("oe", "oe", oe_threshold, counts["calibration"]["n_questions"] + counts["selection"]["n_questions"]),
        }
        policies = {}
        policy_questions = {}
        for name, (target_arm, calibration_arm, threshold, oe_labeled_questions) in configurations.items():
            policy_questions[name] = _policy_questions(test[target_arm], calibrators[calibration_arm], threshold)
            policies[name] = {
                "target_arm": target_arm, "calibration_arm": calibration_arm,
                "threshold_selection_arm": "oe" if name == "oe_selected_on_oe" else mc_arm,
                "threshold": dict(threshold), "n_oe_development_labeled_questions": oe_labeled_questions,
                "metrics": _policy_report(test[target_arm], calibrators[calibration_arm], threshold, bootstrap, risk_budget),
            }
        rq3_arms[mc_arm] = {
            "mc_selection": selections[mc_arm], "oe_selection": selections["oe"], "policies": policies,
            "policy_comparisons": [_policy_comparison(name_a, name_b, policy_questions[name_a], policy_questions[name_b], bootstrap)
                                   for name_a, name_b in (("mc_selected_on_mc", "zero_oe_label_transfer"),
                                                          ("zero_oe_label_transfer", "oe_selected_on_oe"),
                                                          ("oe_calibrated_mc_threshold", "oe_selected_on_oe"))],
            "development_budget": {arm: {"calibration_questions": len(split_data["calibration"][arm]),
                                           "selection_questions": len(split_data["selection"][arm]),
                                           "calibration_answer_prefix_labels": calibrators[arm]["n_answer_prefixes"],
                                           "selection_prefix_labels": sum(len(rows) for rows in split_data["selection"][arm].values())}
                                   for arm in (mc_arm, "oe")},
        }
    warnings = [
        "Empirical selection risk budgets are exploratory constraints, not certified risk guarantees.",
        "Intervals are paired whole-group percentile bootstrap, conditional on fitted calibrators and thresholds; no refitting.",
        "Intervals do not correct for multiple bins, arms, or policies.",
        "Calibration fits answered prefixes only; abstain/invalid are unconditional errors and policy-ineligible.",
        "Zero-OE-label transfer assumes comparable confidence semantics and has no cross-format calibration guarantee.",
        "Equal labeled-question budgets do not imply equal annotation effort or equal numbers of answered prefixes.",
        "High-confidence error diagnostic uses raw reported confidence; it does not establish normalized option probability semantics.",
    ]
    if any(value["n_groups"] < 30 for value in counts.values()):
        warnings.append("Tiny development/test samples: selection and bootstrap intervals may be highly unstable; this is not powered scientific evidence.")
    if counts["test"]["n_groups"] < 2:
        warnings.append("Fewer than two independent test groups: confidence intervals are unavailable.")
    return {
        "schema_version": _SCHEMA,
        "config": {"risk_budget": risk_budget, "bootstrap_samples": bootstrap_samples, "seed": seed},
        "definitions": {
            "accuracy": "accepted=1; rejected, abstain, invalid=0; prefixes averaged within question, then questions",
            "fraction_bins": "ten equal-width intervals (lower, upper]; final bin includes fraction 1",
            "risk": "incorrect commitments / all commitments; null when no commitments",
            "coverage": "committed questions / all eligible questions",
            "utility": "+1 correct commitment, -1 incorrect commitment, 0 no commitment; mean over all questions",
            "commitment_fraction": "mean fraction among committed questions; null if none",
            "intervals": "95% percentile whole-group bootstrap, paired across arms, conditional on frozen fits; undefined replicates omitted and counted; conditional metrics need at least two contributing groups",
            "invalid_abstain_counts": "all prefixes of evaluated complete trajectories, including those after any commitment",
            "calibration_diagnostics": "computed only at selected commitment points on held-out test",
            "units": "accuracy, risk, coverage, differences and confidence are proportions, not percentages",
        },
        "counts": counts, "calibrators": calibrators,
        "rq1": {"split": "test", "arms": rq1_arms},
        "rq2": {"split": "test", "contrasts": contrasts},
        "rq3": {"split": "test", "arms": rq3_arms},
        "warnings": warnings,
    }
