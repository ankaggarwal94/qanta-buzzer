#!/usr/bin/env python3
"""Fail-closed, development-only analysis of the constrained IMCQA WAIT pilot.

Every prefix is an independently scored state. First-commit policy replay is
therefore an evaluation of a stateless policy, not a conversation or a sampled
distribution of complete generated responses. Gold is joined only on the CPU.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


OPTIONS = tuple("ABCD")
ACTIONS = tuple("ABCDE")
SPLITS = ("calibration", "selection")
MENUS = ("independent_pool", "same_category_pool")
VARIANTS = ("wait", "forced", "questionless", "rotation")
PREFIX_IDS = ("p2", "p4", "p6", "p8", "p10")
REWARDS = (1.0, .8, .6, .4, .2)
WRONG_REWARD = -1.0
PASS_REWARD = 0.0
OMISSION_MARKER = "[Question text withheld in this control.]"


def normalize_public_job(job: dict[str, Any]) -> dict[str, Any]:
    """Read the runner's public schema without importing its validator."""
    try:
        payload = json.loads(job["prompt"].split("\n\n")[1])
    except (IndexError, json.JSONDecodeError) as error:
        raise ValueError("public prompt payload is not a single JSON object") from error
    if set(payload) != {"question_prefix", "options"}:
        raise ValueError("unexpected public prompt payload fields")
    return {**job, "job_id": job["score_id"], "variant": job["arm"],
            "canonical_option_ids": job["option_source_ids"], **payload}


def normalize_score_row(row: dict[str, Any]) -> dict[str, Any]:
    """Preserve A--E raw evidence while exposing legal scores for validation."""
    allowed = tuple(row["allowed_actions"])
    logits = row["raw_action_logits"]
    if set(logits) != set(ACTIONS) or any(type(value) not in (int, float) or not math.isfinite(value) for value in logits.values()):
        raise ValueError("five finite raw action logits are required")
    expected = softmax(logits, OPTIONS)
    actual = row.get("conditional_answer_probabilities", {})
    if set(actual) != set(OPTIONS) or any(type(actual[label]) not in (int, float) or not math.isfinite(actual[label]) or abs(actual[label] - expected[label]) > 2e-6 for label in OPTIONS):
        raise ValueError("conditional answer softmax mismatch")
    return {**row, "job_id": row["score_id"], "variant": row["arm"],
            "raw_option_logits": {label: logits[label] for label in allowed},
            "conditional_option_probabilities": row["action_probabilities"],
            "top_option_id": row["chosen_action"], "tied_top_option_ids": row["tied_top_actions"]}


def sha256(path: Path) -> str:
    """Hash a file without reading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def softmax(logits: dict[str, float], labels: tuple[str, ...]) -> dict[str, float]:
    """Calculate a stable softmax restricted to the specified candidate set."""
    top = max(logits[label] for label in labels)
    weights = {label: math.exp(logits[label] - top) for label in labels}
    total = math.fsum(weights.values())
    return {label: weights[label] / total for label in labels}


def validate_score(row: dict[str, Any], labels: tuple[str, ...]) -> None:
    """Independently reconstruct probabilities and the lexicographic argmax."""
    logits = row.get("raw_option_logits", {})
    probabilities = row.get("conditional_option_probabilities", {})
    if set(logits) != set(labels) or set(probabilities) != set(labels):
        raise ValueError("wrong action set in scores")
    if any(type(value) not in (int, float) or not math.isfinite(value)
           for value in (*logits.values(), *probabilities.values())):
        raise ValueError("scores must be finite numeric values")
    if any(value < 0 or value > 1 for value in probabilities.values()):
        raise ValueError("probabilities outside [0, 1]")
    expected = softmax(logits, labels)
    if abs(math.fsum(probabilities.values()) - 1) > 2e-6 or any(
            abs(probabilities[label] - expected[label]) > 2e-6 for label in labels):
        raise ValueError("reported probabilities differ from reconstructed softmax")
    ties = [label for label in labels if logits[label] == max(logits.values())]
    if row.get("top_option_id") != ties[0] or row.get("tied_top_option_ids") != ties:
        raise ValueError("argmax/tie evidence mismatch")


def canonical_mapping(rotation: int) -> dict[str, str]:
    """Map displayed IDs to original candidates for a cyclic forward shift."""
    if type(rotation) is not int or rotation not in (0, 1):
        raise ValueError("only the frozen identity or one-slot cyclic rotation is allowed")
    return {label: OPTIONS[(index - rotation) % 4] for index, label in enumerate(OPTIONS)}


def frozen_index(dataset: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Validate immutable question/group/split identities before any gold join."""
    result = {}
    groups = {}
    for question in dataset["questions"]:
        qid = question["qid"]
        if qid in result:
            raise ValueError("duplicate frozen qid, possibly crossing splits")
        if question["split"] not in (*SPLITS, "test"):
            raise ValueError("unknown frozen split")
        group = question["group_id"]
        if group in groups:
            raise ValueError("duplicate frozen group: question bootstrap would be invalid")
        groups[group] = qid
        menus = {menu["condition"]: menu for menu in question["menus"]}
        if len(menus) != len(question["menus"]) or set(menus) != set(MENUS):
            raise ValueError("duplicate or missing frozen menu")
        for menu in menus.values():
            if [option["id"] for option in menu["options"]] != list(OPTIONS):
                raise ValueError("frozen menu labels/order invalid")
            if menu["gold_option_id"] not in OPTIONS:
                raise ValueError("invalid frozen gold option")
        prefixes = {prefix["prefix_id"]: prefix for prefix in question["prefixes"]}
        if len(prefixes) != len(question["prefixes"]):
            raise ValueError("duplicate frozen prefix")
        result[qid] = {**question, "menu_index": menus, "prefix_index": prefixes}
    return result


def join_job_gold(job: dict[str, Any], questions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Verify a public job against frozen text/menu metadata, then attach gold.

    The generator's alleged gold, if present, never defines correctness. The
    rotated label must represent exactly the original frozen candidate text.
    """
    qid = job["qid"]
    if qid not in questions:
        raise ValueError("unknown public qid")
    question = questions[qid]
    if job["split"] not in SPLITS or job["split"] != question["split"]:
        raise ValueError("public qid split differs from frozen split or includes test")
    if job["group_id"] != question["group_id"]:
        raise ValueError("public group differs from frozen group")
    variant = job["variant"]
    if variant not in VARIANTS:
        raise ValueError("unknown pilot variant")
    round_number = job["round"]
    if type(round_number) is not int or not 1 <= round_number <= 5:
        raise ValueError("invalid round")
    if job["prefix_id"] != PREFIX_IDS[round_number - 1]:
        raise ValueError("round/prefix mismatch")
    prefix = question["prefix_index"][job["prefix_id"]]
    condition = job["condition"]
    if condition not in MENUS:
        raise ValueError("unknown menu condition")
    menu = question["menu_index"][condition]
    if job["menu_id"] != menu["menu_id"]:
        raise ValueError("menu ID differs from frozen menu")
    rotation = 1 if variant == "rotation" else 0
    mapping = canonical_mapping(rotation)
    if job.get("rotation", rotation) != rotation:
        raise ValueError("rotation disagrees with variant")
    if "canonical_option_ids" in job and job["canonical_option_ids"] != mapping:
        raise ValueError("canonical option mapping disagrees with rotation")
    original_texts = {option["id"]: option["text"] for option in menu["options"]}
    wanted_options = [{"id": label, "text": original_texts[mapping[label]]} for label in OPTIONS]
    if job["options"] != wanted_options:
        raise ValueError("displayed option text/rotation differs from frozen menu")
    expected_gold = next(label for label, canonical in mapping.items() if canonical == menu["gold_option_id"])
    if "gold_option_id" in job and job["gold_option_id"] != expected_gold:
        raise ValueError("public alleged gold label disagrees with frozen gold after rotation")
    if "fraction" in job and not math.isclose(job["fraction"], prefix["fraction"], abs_tol=1e-12):
        raise ValueError("actual prefix fraction differs from frozen prefix")
    if "question_prefix" in job:
        expected_prefix = OMISSION_MARKER if variant == "questionless" else prefix["text"]
        if job["question_prefix"] != expected_prefix:
            raise ValueError("question prefix differs from frozen source")
    if hashlib.sha256(job["prompt"].encode()).hexdigest() != job["prompt_sha256"]:
        raise ValueError("public prompt hash mismatch")
    return {**job, "gold_option_id": expected_gold,
            "canonical_gold_option_id": menu["gold_option_id"], "canonical_option_ids": mapping,
            "actual_fraction": prefix["fraction"]}


def validate_source_binding(jobs: list[dict[str, Any]], source: dict[str, Any]) -> None:
    """Bind each transformed job to its exact original public MC job."""
    needed = {job["source_job_id"] for job in jobs}
    lookup = {}
    for row in source["jobs"]:
        if row["job_id"] in needed:
            if row["job_id"] in lookup:
                raise ValueError("duplicate frozen source job")
            lookup[row["job_id"]] = row
    if set(lookup) != needed:
        raise ValueError("missing original frozen source jobs")
    for job in jobs:
        original = lookup[job["source_job_id"]]
        if original.get("format") != "mc":
            raise ValueError("pilot source is not MC")
        if hashlib.sha256(original["prompt"].encode()).hexdigest() != original["prompt_sha256"]:
            raise ValueError("original source prompt hash mismatch")
        if job["source_prompt_sha256"] != original["prompt_sha256"]:
            raise ValueError("pilot source prompt hash binding mismatch")
        for key in ("qid", "group_id", "split", "condition", "menu_id", "prefix_id", "fraction"):
            if job[key] != original[key]:
                raise ValueError(f"pilot/source {key} binding mismatch")


def validate_jobs(jobs: list[dict[str, Any]], dataset: dict[str, Any], *,
                  n_per_split: int = 100, n_diagnostic_per_split: int = 20) -> list[dict[str, Any]]:
    """Require the exact factorial design and a shared diagnostic subset."""
    questions = frozen_index(dataset)
    enriched = [join_job_gold(job, questions) for job in jobs]
    ids = [job["job_id"] for job in jobs]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate public job ID")
    cells = defaultdict(set)
    qids = {split: set() for split in SPLITS}
    diagnostics = {split: set() for split in SPLITS}
    for job in enriched:
        key = (job["qid"], job["variant"])
        cell = (job["condition"], job["round"])
        if cell in cells[key]:
            raise ValueError("duplicate public qid/variant/menu/round")
        cells[key].add(cell)
        qids[job["split"]].add(job["qid"])
        if job["variant"] in ("questionless", "rotation"):
            diagnostics[job["split"]].add(job["qid"])
    expected_cells = {(condition, round_number) for condition in MENUS for round_number in range(1, 6)}
    for split in SPLITS:
        if len(qids[split]) != n_per_split or len(diagnostics[split]) != n_diagnostic_per_split:
            raise ValueError("public question or diagnostic split coverage mismatch")
        for qid in qids[split]:
            for variant in VARIANTS:
                expected = expected_cells if variant in ("wait", "forced") or qid in diagnostics[split] else set()
                if cells.get((qid, variant), set()) != expected:
                    raise ValueError("incomplete factorial or unmatched diagnostic subset")
    if qids["calibration"] & qids["selection"]:
        raise ValueError("public qid crosses development splits")
    return enriched


def validate_rows(rows: list[dict[str, Any]], jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Require every frozen public job exactly once and independently grade it."""
    expected = {job["job_id"]: job for job in jobs}
    if len(expected) != len(jobs) or len(rows) != len(jobs):
        raise ValueError("duplicate jobs or score coverage mismatch")
    seen = set()
    result = []
    for row in rows:
        job_id = row["job_id"]
        if job_id not in expected or job_id in seen:
            raise ValueError("unknown or duplicate score job ID")
        seen.add(job_id)
        job = expected[job_id]
        for key in ("qid", "group_id", "split", "condition", "menu_id", "prefix_id", "round", "variant", "prompt_sha256"):
            if row.get(key) != job[key]:
                raise ValueError(f"score/public {key} mismatch")
        for key in ("score_id", "score_index", "source_job_id", "source_prompt_sha256", "fraction", "reward", "arm", "rotation", "option_source_ids", "allowed_actions"):
            if key in job and row.get(key) != job[key]:
                raise ValueError(f"score/public {key} mismatch")
        labels = OPTIONS if job["variant"] == "forced" else ACTIONS
        validate_score(row, labels)
        action = row["top_option_id"]
        canonical = job["canonical_option_ids"].get(action)
        result.append({**row, "gold_option_id": job["gold_option_id"],
                       "canonical_gold_option_id": job["canonical_gold_option_id"],
                       "canonical_option_ids": job["canonical_option_ids"],
                       "actual_fraction": job["actual_fraction"],
                       "canonical_choice": canonical,
                       "correct": action in OPTIONS and action == job["gold_option_id"],
                       "action_semantics": action if action in OPTIONS else ("PASS" if job["round"] == 5 else "WAIT")})
    if seen != set(expected):
        raise ValueError("missing scores")
    return result


def make_trajectories(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str, str], list[dict[str, Any]]]:
    """Preserve all paired states; reject duplicate/missing rounds."""
    trajectories = defaultdict(list)
    for row in rows:
        trajectories[(row["split"], row["variant"], row["condition"], row["qid"])].append(row)
    for key, trajectory in trajectories.items():
        trajectory.sort(key=lambda row: row["round"])
        if [row["round"] for row in trajectory] != [1, 2, 3, 4, 5]:
            raise ValueError(f"trajectory has missing or duplicate rounds: {key}")
        if len({row["group_id"] for row in trajectory}) != 1:
            raise ValueError("trajectory changes question group")
    return dict(trajectories)


def policy_record(trajectory: list[dict[str, Any]], *, fixed_round: int | None = None) -> dict[str, Any]:
    """Replay first A--D, treating final E as PASS and any wrong answer as terminal."""
    if [row["round"] for row in trajectory] != [1, 2, 3, 4, 5]:
        raise ValueError("policy requires five ordered rounds")
    if fixed_round is not None and fixed_round not in range(1, 6):
        raise ValueError("invalid fixed round")
    if fixed_round is None:
        chosen = next((row for row in trajectory if row["top_option_id"] in OPTIONS), trajectory[-1])
        observed = trajectory[:chosen["round"]]
    else:
        chosen = trajectory[fixed_round - 1]
        observed = [chosen]
    committed = chosen["top_option_id"] in OPTIONS
    correct = committed and chosen["correct"]
    reward = (REWARDS[chosen["round"] - 1] if correct else WRONG_REWARD) if committed else PASS_REWARD
    token_lengths = [len(row["scored_input_token_ids"]) for row in observed if "scored_input_token_ids" in row]
    return {**{key: chosen[key] for key in ("qid", "group_id", "split", "condition", "variant")},
            "policy": "first_commit" if fixed_round is None else f"fixed_round_{fixed_round}",
            "committed": committed, "correct": correct, "wrong": committed and not correct,
            "terminal_pass": not committed, "round": chosen["round"] if committed else None,
            "observed_round": chosen["round"], "actual_fraction": chosen["actual_fraction"],
            "canonical_choice": chosen.get("canonical_choice") if committed else None,
            "reward": reward,
            "counterfactual_uncached_input_tokens": sum(token_lengths) if len(token_lengths) == len(observed) else None}


def bootstrap_indices(n: int, samples: int = 2000, seed: int = 1) -> np.ndarray:
    if n < 1 or samples < 1:
        raise ValueError("positive bootstrap question/sample count required")
    return np.random.default_rng(seed).integers(0, n, (samples, n))


def interval(values: np.ndarray) -> dict[str, Any]:
    finite = values[np.isfinite(values)]
    return {"ci95": np.quantile(finite, [.025, .975]).tolist() if len(finite) else None,
            "defined_resamples": int(len(finite)), "total_resamples": int(len(values))}


def metric_vector(records: list[dict[str, Any]]) -> dict[str, float | None]:
    n = len(records)
    if not n:
        raise ValueError("cannot summarize an empty policy")
    commits = [record for record in records if record["committed"]]
    return {"coverage": len(commits) / n,
            "correct_fraction": sum(record["correct"] for record in records) / n,
            "risk": sum(record["wrong"] for record in commits) / len(commits) if commits else None,
            "mean_reward": float(np.mean([record["reward"] for record in records])),
            "mean_round_when_committed": float(np.mean([record["round"] for record in commits])) if commits else None,
            "mean_observed_round": float(np.mean([record["observed_round"] for record in records])),
            "mean_revealed_fraction": float(np.mean([record["actual_fraction"] for record in records]))}


def question_totals(records: list[dict[str, Any]]) -> tuple[list[str], np.ndarray]:
    """Sufficient statistics keep all menu observations within each question."""
    groups = defaultdict(list)
    for record in records:
        groups[record["qid"]].append(record)
    qids = sorted(groups)
    totals = []
    for qid in qids:
        rows = groups[qid]
        totals.append([len(rows), sum(row["committed"] for row in rows), sum(row["correct"] for row in rows),
                       sum(row["wrong"] for row in rows), sum(row["reward"] for row in rows),
                       sum(row["round"] or 0 for row in rows), sum(row["observed_round"] for row in rows),
                       sum(row["actual_fraction"] for row in rows)])
    return qids, np.asarray(totals, dtype=float)


def metrics_from_totals(totals: np.ndarray) -> dict[str, np.ndarray]:
    n, committed, correct, wrong, reward, rounds, observed, fraction = totals.T
    with np.errstate(divide="ignore", invalid="ignore"):
        return {"coverage": committed / n, "correct_fraction": correct / n, "risk": wrong / committed,
                "mean_reward": reward / n, "mean_round_when_committed": rounds / committed,
                "mean_observed_round": observed / n, "mean_revealed_fraction": fraction / n}


def summarize_policy(records: list[dict[str, Any]], indices: np.ndarray) -> dict[str, Any]:
    """Bootstrap question units, retaining paired menus inside a question."""
    questions = defaultdict(list)
    for record in records:
        questions[record["qid"]].append(record)
    qids = sorted(questions)
    if indices.ndim != 2 or indices.shape[1] != len(qids) or np.min(indices) < 0 or np.max(indices) >= len(qids):
        raise ValueError("bootstrap indices do not match question units")
    if len({len(group) for group in questions.values()}) != 1:
        raise ValueError("unequal menu records per question")
    point = metric_vector(records)
    _, totals = question_totals(records)
    sampled = metrics_from_totals(totals[indices].sum(axis=1))
    tokens = [record["counterfactual_uncached_input_tokens"] for record in records]
    return {"n_questions": len(qids), "n_question_menu_records": len(records),
            "n_committed": sum(record["committed"] for record in records),
            "n_correct": sum(record["correct"] for record in records),
            "n_wrong": sum(record["wrong"] for record in records),
            "n_terminal_pass": sum(record["terminal_pass"] for record in records),
            **point, "stopping_round_counts": dict(sorted(Counter(str(record["round"]) if record["committed"] else "PASS" for record in records).items())),
            "mean_counterfactual_uncached_input_tokens": float(np.mean(tokens)) if all(value is not None for value in tokens) else None,
            "bootstrap": {key: interval(np.asarray(values)) for key, values in sampled.items()},
            "interval_scope": "Descriptive development-set percentile intervals; question resampling retains paired menus. Conditional risk intervals with few commitments are not risk guarantees."}


def paired_difference(left: list[dict[str, Any]], right: list[dict[str, Any]], indices: np.ndarray) -> dict[str, Any]:
    """Report left minus right with exactly paired questions and menus."""
    def key(record):
        return record["qid"], record["condition"]
    left = sorted(left, key=key)
    right = sorted(right, key=key)
    if [key(record) for record in left] != [key(record) for record in right]:
        raise ValueError("paired comparison question/menu mismatch")
    questions = sorted({record["qid"] for record in left})
    if indices.shape[1] != len(questions):
        raise ValueError("paired bootstrap question count mismatch")
    names = ("mean_reward", "coverage", "correct_fraction", "risk", "mean_observed_round")
    left_metrics = metric_vector(left)
    right_metrics = metric_vector(right)
    points = {name: left_metrics[name] - right_metrics[name] if left_metrics[name] is not None and right_metrics[name] is not None else None for name in names}
    _, left_totals = question_totals(left)
    _, right_totals = question_totals(right)
    a = metrics_from_totals(left_totals[indices].sum(axis=1))
    b = metrics_from_totals(right_totals[indices].sum(axis=1))
    samples = {name: a[name] - b[name] for name in names}
    return {"n_questions": len(questions), "n_question_menu_pairs": len(left),
            "direction": "left minus right",
            "differences": {name: {"mean": points[name], **interval(np.asarray(samples[name]))} for name in names}}


def prompt_effect(wait_row: dict[str, Any], forced_row: dict[str, Any]) -> dict[str, Any]:
    """Compare forced A--D with primary A--D conditional on not taking E."""
    for key in ("qid", "condition", "round"):
        if wait_row[key] != forced_row[key]:
            raise ValueError("prompt comparison must pair the same state")
    p = softmax(wait_row["raw_option_logits"], OPTIONS)
    q = softmax(forced_row["raw_option_logits"], OPTIONS)
    return {"qid": wait_row["qid"], "split": wait_row["split"], "condition": wait_row["condition"], "round": wait_row["round"],
            "conditional_total_variation": .5 * math.fsum(abs(p[label] - q[label]) for label in OPTIONS),
            "conditional_top_changed": max(OPTIONS, key=p.get) != max(OPTIONS, key=q.get),
            "primary_wait_probability": wait_row["conditional_option_probabilities"]["E"],
            "primary_conditional_gold_probability": p[wait_row["gold_option_id"]],
            "forced_gold_probability": q[forced_row["gold_option_id"]]}


def rotation_effect(primary: dict[str, Any], rotated: dict[str, Any]) -> dict[str, Any]:
    """Compare distributions after mapping displayed IDs to frozen candidates."""
    for key in ("qid", "condition", "round"):
        if primary[key] != rotated[key]:
            raise ValueError("rotation comparison must pair the same state")
    p = {**primary["conditional_option_probabilities"]}
    q = {rotated["canonical_option_ids"].get(label, label): value
         for label, value in rotated["conditional_option_probabilities"].items()}
    first = primary.get("canonical_choice") or "E"
    second = rotated.get("canonical_choice") or "E"
    return {"qid": primary["qid"], "split": primary["split"], "condition": primary["condition"], "round": primary["round"],
            "semantic_action_changed": first != second,
            "wait_decision_changed": (first == "E") != (second == "E"),
            "semantic_total_variation": .5 * math.fsum(abs(p[label] - q[label]) for label in ACTIONS)}


def summarize_diagnostics(rows: list[dict[str, Any]], metrics: tuple[str, ...], *, samples: int, seed: int) -> dict[str, Any]:
    """Bootstrap all diagnostic rounds/menus jointly within their question."""
    groups = defaultdict(list)
    for row in rows:
        groups[row["qid"]].append(row)
    values = np.asarray([[np.mean([row[metric] for row in groups[qid]]) for metric in metrics] for qid in sorted(groups)])
    indices = bootstrap_indices(len(values), samples, seed)
    boot = values[indices].mean(axis=1)
    return {"n_questions": len(groups), "n_paired_states": len(rows),
            **{metric: {"mean": float(values[:, index].mean()), **interval(boot[:, index])} for index, metric in enumerate(metrics)}}


def compare_numeric(left: list[dict[str, Any]], right: list[dict[str, Any]], jobs: list[dict[str, Any]],
                    config: dict[str, Any]) -> dict[str, Any]:
    """Recheck retained raw diagnostics without trusting a passed gate flag."""
    if not left or len(left) != len(right) or len(left) != len(jobs):
        raise ValueError("numeric diagnostic cardinality mismatch")
    atol = config["raw_logit_atol"]
    rtol = config["raw_logit_rtol"]
    probability_atol = config["max_action_probability_absolute_difference"]
    maximum_logit = maximum_probability = 0.0
    for a, b, job in zip(left, right, jobs):
        if len(a["logits"]) != 5 or len(b["logits"]) != 5:
            raise ValueError("numeric diagnostic requires five logits")
        x, y = np.asarray(a["logits"], dtype=float), np.asarray(b["logits"], dtype=float)
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
            raise ValueError("nonfinite numeric diagnostic")
        delta = np.abs(x - y)
        if np.any(delta > atol + rtol * np.abs(y)):
            raise ValueError("numerical logit tolerance exceeded")
        labels = tuple(job["allowed_actions"])
        p, q = softmax(dict(zip(ACTIONS, x)), labels), softmax(dict(zip(ACTIONS, y)), labels)
        probability_delta = max(abs(p[label] - q[label]) for label in labels)
        if probability_delta > probability_atol or max(labels, key=p.get) != max(labels, key=q.get):
            raise ValueError("numerical probability tolerance or action agreement failed")
        maximum_logit = max(maximum_logit, float(delta.max()))
        maximum_probability = max(maximum_probability, probability_delta)
    return {"passed": True, "rows": len(jobs), "max_logit_difference": maximum_logit,
            "max_action_probability_difference": maximum_probability, "argmax_changes": 0}


def validate_vocab_row(row: dict[str, Any]) -> None:
    """Check retained full-vocabulary summaries and legal-token mass."""
    logits = row["raw_action_logits"]
    normalizer = row["vocabulary_logsumexp"]
    top_logit = row["unconstrained_top_logit"]
    if any(type(value) not in (int, float) or not math.isfinite(value) for value in (normalizer, top_logit)):
        raise ValueError("nonfinite vocabulary summary")
    if max(logits.values()) > top_logit + 2e-5 or top_logit > normalizer + 2e-5:
        raise ValueError("inconsistent vocabulary max/normalizer")
    for field, labels in (("all_five_action_token_vocabulary_mass", ACTIONS),
                          ("legal_action_vocabulary_mass", tuple(row["allowed_actions"]))):
        actual = row.get(field)
        expected = math.fsum(math.exp(logits[label] - normalizer) for label in labels)
        if (type(actual) not in (int, float) or not math.isfinite(actual) or
                not 0 <= actual <= 1 + 2e-5 or not math.isclose(actual, expected, abs_tol=2e-6, rel_tol=1e-6)):
            raise ValueError(f"{field} mismatch")
    tokens = row["scored_input_token_ids"]
    action_ids = row["option_token_ids"]
    if not tokens or len(tokens) > 2048 or any(type(token) is not int or token < 0 for token in tokens):
        raise ValueError("invalid scored input token IDs")
    if set(action_ids) != set(ACTIONS) or len(set(action_ids.values())) != 5 or any(type(value) is not int or value < 0 for value in action_ids.values()):
        raise ValueError("invalid action token IDs")
    top_id = row["unconstrained_top_token_id"]
    if type(top_id) is not int or top_id < 0:
        raise ValueError("invalid vocabulary argmax token ID")
    if top_id in action_ids.values():
        label = next(label for label, token in action_ids.items() if token == top_id)
        if abs(logits[label] - top_logit) > 2e-5:
            raise ValueError("vocabulary argmax logit differs from selected token logit")


def validate_model_output(directory: Path, tag: str, package: dict[str, Any], jobs: list[dict[str, Any]],
                          config: dict[str, Any], public_hash: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Validate completed coverage, model identity, numerical gates and live replay."""
    receipt = load_json(directory / "receipt.json")
    if (receipt.get("status") != "complete" or receipt.get("completed_rows") != len(jobs)
            or receipt.get("expected_rows") != len(jobs) or receipt.get("model_tag") != tag
            or receipt.get("public_input_sha256") != public_hash):
        raise ValueError("incomplete or mismatched model completion receipt")
    score_path = directory / "scores.jsonl"
    if sha256(score_path) != receipt["scores_sha256"]:
        raise ValueError("score file hash differs from completion receipt")
    raw = score_path.read_bytes()
    if not raw.endswith(b"\n"):
        raise ValueError("truncated score file")
    raw_rows = [json.loads(line) for line in raw.splitlines()]
    metadata = load_json(directory / "metadata.json")
    expected_metadata = {**config["models"][tag], "public_input_sha256": public_hash,
                         "dtype": "float32", "loaded_dtype": "bfloat16", "attention": "eager", "tf32": False,
                         "generation": False, "sampling": False, "seed": 1, "assistant_prefix": '{"action":"'}
    for key, value in expected_metadata.items():
        if metadata.get(key) != value:
            raise ValueError(f"model metadata {key} mismatch")
    if receipt.get("protocol") != package["protocol"] or metadata.get("protocol") != package["protocol"]:
        raise ValueError("receipt/metadata protocol mismatch")
    if any(row.get("model_tag") != tag or row.get("schema_version") != "imcqa-wait-scores-v1" for row in raw_rows):
        raise ValueError("row model/schema mismatch")
    for row in raw_rows:
        validate_vocab_row(row)
    rows = validate_rows([normalize_score_row(row) for row in raw_rows], jobs)
    lookup = {row["score_id"]: row for row in rows}
    job_lookup = {job["score_id"]: job for job in jobs}
    plan = load_json(directory / "plan.json")
    if (plan.get("input_sha256") != public_hash or
            plan.get("ordered_score_ids") != [row["score_id"] for row in rows] or
            plan.get("context_sha256") != [row["scored_context_sha256"] for row in rows] or
            plan.get("token_counts") != [len(row["scored_input_token_ids"]) for row in rows]):
        raise ValueError("production ordering/context plan mismatch")
    diagnostics_paths = sorted((directory / "attempts").glob("*_diagnostics.json"))
    live_paths = sorted((directory / "attempts").glob("*_active_only.json"))
    if len(diagnostics_paths) != 1 or len(live_paths) != 1:
        raise ValueError("exactly one attempt with numerical and active-only evidence required")
    diagnostic = load_json(diagnostics_paths[0])
    ids = diagnostic["score_ids"]
    if len(set(ids)) != len(ids) or not 1 <= len(ids) <= config["execution"]["numerical_contexts_max_per_model"]:
        raise ValueError("invalid numerical diagnostic sample")
    diagnostic_jobs = [job_lookup[score_id] for score_id in ids]
    if {job["arm"] for job in diagnostic_jobs} != set(VARIANTS) or {job["condition"] for job in diagnostic_jobs} != set(MENUS) or {job["round"] for job in diagnostic_jobs} != set(range(1, 6)):
        raise ValueError("numerical diagnostics do not span arms, menus and rounds")
    if diagnostic["cached"] != diagnostic["replay"]:
        raise ValueError("exact numerical replay failed")
    checks = {name: compare_numeric(diagnostic["cached"], diagnostic[right], diagnostic_jobs, config["numerical_checks"])
              for name, right in (("cached_uncached", "uncached"), ("cached_single", "singles"), ("permutation", "permuted_aligned"))}
    production = [{"logits": [lookup[score_id]["raw_action_logits"][label] for label in ACTIONS]} for score_id in ids]
    checks["production_diagnostic"] = compare_numeric(production, diagnostic["cached"], diagnostic_jobs, config["numerical_checks"])
    live = load_json(live_paths[0])
    selected = package["selection"]["selected_qids"]
    expected_qids = [qid for split in SPLITS for qid in sorted(selected[split], key=lambda qid: (hashlib.sha256(f"live|1|{split}|{qid}".encode()).hexdigest(), qid))[:4]]
    if live["qids"] != expected_qids or live["episodes"] != len(expected_qids) * 2:
        raise ValueError("active-only replay question selection differs")
    expected_live = []
    for qid in expected_qids:
        for condition in MENUS:
            candidates = sorted((row for row in rows if row["qid"] == qid and row["condition"] == condition and row["arm"] == "wait"), key=lambda row: row["round"])
            for row in candidates:
                expected_live.append(row["score_id"])
                if row["chosen_action"] != "E":
                    break
    if [row["score_id"] for row in live["rows"]] != expected_live:
        raise ValueError("active-only replay failed first-commit/terminal-PASS semantics")
    live_checks = []
    for live_row in live["rows"]:
        row = lookup[live_row["score_id"]]
        if live_row["chosen_action"] != row["chosen_action"]:
            raise ValueError("active-only action differs from counterfactual scoring")
        live_checks.append(compare_numeric([{"logits": [row["raw_action_logits"][label] for label in ACTIONS]}],
            [live_row["live"]], [job_lookup[row["score_id"]]], config["numerical_checks"]))
    audit = {"passed": True, "n_rows": len(rows), "n_questions": len({row["qid"] for row in rows}),
             "scores_sha256": sha256(score_path), "metadata_sha256": sha256(directory / "metadata.json"),
             "elapsed_seconds": receipt["elapsed_seconds"], "numerical_checks": checks,
             "active_only_replay": {"n_questions": len(expected_qids), "n_episodes": len(expected_qids) * 2,
                                    "n_visited_states": len(expected_live), "passed": True,
                                    "max_action_probability_difference": max(check["max_action_probability_difference"] for check in live_checks)},
             "mean_legal_action_vocabulary_mass": float(np.mean([row["legal_action_vocabulary_mass"] for row in rows])),
             "all_file_sha256": {str(path.relative_to(directory)): sha256(path) for path in sorted(directory.rglob("*.json"))}}
    return rows, audit


def reference_record(trajectory: list[dict[str, Any]], policy: str) -> dict[str, Any]:
    """Construct clearly labelled zero-reward and forced-answer oracle references."""
    record = policy_record(trajectory, fixed_round=5)
    if policy == "always_terminal_pass":
        return {**record, "policy": policy, "committed": False, "correct": False, "wrong": False,
                "terminal_pass": True, "round": None, "canonical_choice": None, "reward": 0.,
                "counterfactual_uncached_input_tokens": 0}
    if policy == "hindsight_forced_or_pass":
        candidates = [policy_record(trajectory, fixed_round=index) for index in range(1, 6)]
        best = max(candidates, key=lambda row: row["reward"])
        if best["reward"] < 0:
            best = reference_record(trajectory, "always_terminal_pass")
        return {**best, "policy": policy, "counterfactual_uncached_input_tokens": None}
    raise ValueError("unknown reference policy")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write empty CSV")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze_outputs(all_rows: dict[str, list[dict[str, Any]]], *, samples: int = 2000, seed: int = 1) -> dict[str, Any]:
    """Build descriptive development summaries without fitting or selecting policies."""
    policy_records = []
    round_records = []
    effects = []
    rotation_effects = []
    for tag, rows in all_rows.items():
        trajectories = make_trajectories(rows)
        for (split, variant, condition, qid), trajectory in sorted(trajectories.items()):
            records = ([policy_record(trajectory, fixed_round=index) for index in range(1, 6)]
                       + [reference_record(trajectory, name) for name in ("always_terminal_pass", "hindsight_forced_or_pass")]
                       if variant == "forced" else [policy_record(trajectory)])
            policy_records.extend({"model": tag, **record} for record in records)
        lookup = {(row["qid"], row["condition"], row["round"], row["variant"]): row for row in rows}
        for row in rows:
            labels = tuple(row["allowed_actions"])
            conditional = softmax(row["raw_action_logits"], OPTIONS)
            candidate = max(OPTIONS, key=conditional.get)
            round_records.append({"model": tag, "split": row["split"], "qid": row["qid"], "condition": row["condition"],
                "variant": row["variant"], "round": row["round"], "native_action_correct": row["correct"],
                "native_answered": row["chosen_action"] != "E", "E_probability": row["action_probabilities"].get("E"),
                "conditional_candidate_correct": candidate == row["gold_option_id"],
                "conditional_gold_probability": conditional[row["gold_option_id"]],
                "legal_action_vocabulary_mass": row["legal_action_vocabulary_mass"],
                "unconstrained_top_is_legal": row["unconstrained_top_token_id"] in {row["option_token_ids"][label] for label in labels}})
            if row["variant"] == "wait":
                forced = lookup[row["qid"], row["condition"], row["round"], "forced"]
                effects.append({"model": tag, **prompt_effect(row, forced)})
                rotated = lookup.get((row["qid"], row["condition"], row["round"], "rotation"))
                if rotated is not None:
                    rotation_effects.append({"model": tag, **rotation_effect(row, rotated)})
    cells = defaultdict(list)
    for record in policy_records:
        cells[(record["model"], record["split"], record["condition"], record["variant"], record["policy"])].append(record)
    summaries = []
    for (model, split, condition, variant, policy), records in sorted(cells.items()):
        indices = bootstrap_indices(len({record["qid"] for record in records}), samples, seed)
        summaries.append({"model": model, "split": split, "condition": condition, "variant": variant, "policy": policy,
                          **summarize_policy(records, indices)})
    comparisons = []
    for model in sorted(all_rows):
        for split in SPLITS:
            for condition in MENUS:
                primary = cells[model, split, condition, "wait", "first_commit"]
                comparisons.append({"model": model, "split": split, "condition": condition, "comparison": "primary_minus_forced_final",
                    **paired_difference(primary, cells[model, split, condition, "forced", "fixed_round_5"], bootstrap_indices(len(primary), samples, seed))})
                for control in ("questionless", "rotation"):
                    right = cells[model, split, condition, control, "first_commit"]
                    qids = {record["qid"] for record in right}
                    left = [record for record in primary if record["qid"] in qids]
                    comparisons.append({"model": model, "split": split, "condition": condition, "comparison": f"primary_minus_{control}_matched_subset",
                        **paired_difference(left, right, bootstrap_indices(len(left), samples, seed))})
    models = sorted(all_rows)
    if len(models) == 2:
        for split in SPLITS:
            for condition in MENUS:
                for variant, policy in (("wait", "first_commit"), ("forced", "fixed_round_5")):
                    left = cells[models[1], split, condition, variant, policy]
                    right = cells[models[0], split, condition, variant, policy]
                    comparisons.append({"model": f"{models[1]} minus {models[0]}", "split": split, "condition": condition, "comparison": f"paired_models_{variant}_{policy}",
                        **paired_difference(left, right, bootstrap_indices(len(left), samples, seed))})
    diagnostic_summaries = []
    for model in sorted(all_rows):
        for split in SPLITS:
            for condition in MENUS:
                for name, records, metrics in (("forced_prompt_effect", effects, ("conditional_total_variation", "conditional_top_changed", "primary_wait_probability")),
                    ("semantic_rotation_effect", rotation_effects, ("semantic_total_variation", "semantic_action_changed", "wait_decision_changed"))):
                    selected = [row for row in records if row["model"] == model and row["split"] == split and row["condition"] == condition]
                    diagnostic_summaries.append({"model": model, "split": split, "condition": condition, "diagnostic": name,
                        **summarize_diagnostics(selected, metrics, samples=samples, seed=seed)})
    round_summaries = []
    round_cells = defaultdict(list)
    for record in round_records:
        round_cells[record["model"], record["split"], record["condition"], record["variant"], record["round"]].append(record)
    for (model, split, condition, variant, round_number), records in sorted(round_cells.items()):
        round_summaries.append({"model": model, "split": split, "condition": condition, "variant": variant, "round": round_number,
            "n_questions": len(records), "native_correct_fraction": float(np.mean([row["native_action_correct"] for row in records])),
            "native_answer_rate": float(np.mean([row["native_answered"] for row in records])),
            "mean_E_probability": float(np.mean([row["E_probability"] for row in records])) if variant != "forced" else None,
            "conditional_candidate_accuracy": float(np.mean([row["conditional_candidate_correct"] for row in records])),
            "mean_conditional_gold_probability": float(np.mean([row["conditional_gold_probability"] for row in records])),
            "mean_legal_action_vocabulary_mass": float(np.mean([row["legal_action_vocabulary_mass"] for row in records])),
            "unconstrained_top_legal_fraction": float(np.mean([row["unconstrained_top_is_legal"] for row in records]))})
    return {"policy_summaries": summaries, "paired_comparisons": comparisons,
            "diagnostic_summaries": diagnostic_summaries, "round_summaries": round_summaries,
            "policy_records": policy_records, "round_records": round_records,
            "forced_prompt_effects": effects, "rotation_effects": rotation_effects}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public", type=Path, required=True)
    parser.add_argument("--frozen-source", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--outputs", type=Path, required=True, help="Directory containing qwen3b and qwen7b subdirectories")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    config = load_json(args.config)
    for path, name in ((args.frozen_source, "public/main_jobs.json"), (args.gold, "evaluator/main_dataset.json")):
        if sha256(path) != config["frozen_source_sha256"][name]:
            raise ValueError(f"immutable source hash mismatch: {name}")
    package = load_json(args.public)
    if (package["rewards"] != list(REWARDS) or package["wrong_reward"] != WRONG_REWARD or package["pass_reward"] != PASS_REWARD
            or package["prefix_ids"] != list(PREFIX_IDS) or package["source_input_sha256"] != sha256(args.frozen_source)):
        raise ValueError("public package reward/prefix/source contract mismatch")
    jobs = validate_jobs([normalize_public_job(job) for job in package["jobs"]], load_json(args.gold),
                         n_per_split=config["n_per_split"], n_diagnostic_per_split=config["n_diagnostic_per_split"])
    validate_source_binding(jobs, load_json(args.frozen_source))
    if len(jobs) != config["execution"]["production_contexts_per_model"]:
        raise ValueError("pilot production count differs from frozen config")
    public_hash = sha256(args.public)
    all_rows, audits = {}, {}
    for model in config["models"]:
        all_rows[model], audits[model] = validate_model_output(args.outputs / model, model, package, jobs, config, public_hash)
    report = analyze_outputs(all_rows, samples=config["analysis"]["bootstrap_samples"], seed=config["analysis"]["bootstrap_seed"])
    record_names = ("policy_records", "round_records", "forced_prompt_effects", "rotation_effects")
    args.out.mkdir(parents=True, exist_ok=False)
    for name in record_names:
        write_csv(args.out / f"{name}.csv", report.pop(name))
    report.update(schema_version="imcqa-wait-analysis-v1", evidence_scope="exploratory_development_only",
                  protocol=package["protocol"], n_questions=len({job["qid"] for job in jobs}), n_score_rows=sum(map(len, all_rows.values())),
                  frozen_inputs_sha256={"public": public_hash, "config": sha256(args.config), "gold": sha256(args.gold), "frozen_source": sha256(args.frozen_source)},
                  audits=audits, config=config,
                  limitations=["Development data only; no confirmatory test set was used or new policies fitted.",
                    "Constrained next-token action probabilities are not calibrated correctness or complete-response sampling frequencies.",
                    "Each scored state assumes prior WAITs; active-only replay checks equivalence for the declared stateless deterministic policy.",
                    "Questionless controls retain the round, rewards and prior-WAIT count while withholding current question text at every round.",
                    "Hindsight forced-answer/PASS oracle is not deployable and is not an upper bound for candidates changed by the primary prompt.",
                    "Bootstrap units are questions; small-commitment risk intervals are descriptive and not population risk guarantees.",
                    "Counterfactual input-token sums exclude KV reuse, generation and orchestration and do not establish actual deployment latency or cost.",
                    "Any comparison with the older BF16 generated-response run changes precision, prompts and output protocol simultaneously.",
                    "No IRT model or broadly representative model ranking is estimated."])
    write_json(args.out / "report.json", report)
    lines = ["IMCQA EXPLICIT WAIT DEVELOPMENT PILOT", "", f"Validated {report['n_score_rows']} scores over {report['n_questions']} development questions.",
             "Primary = first legal A-D argmax; E waits in rounds 1-4 and passes at round 5.",
             "Correct rewards: 1, .8, .6, .4, .2; wrong: -1; terminal PASS: 0.", ""]
    for summary in report["policy_summaries"]:
        if summary["variant"] == "wait":
            risk = "undefined" if summary["risk"] is None else f"{summary['risk']:.3%}"
            lines.append(f"{summary['model']} / {summary['split']} / {summary['condition']}: reward={summary['mean_reward']:.4f}; commitments={summary['n_committed']}/{summary['n_questions']}; errors={summary['n_wrong']}; conditional risk={risk}; terminal PASS={summary['n_terminal_pass']}.")
    lines.extend(["", "INTERPRETATION LIMITS", *report["limitations"]])
    (args.out / "FINDINGS.txt").write_text("\n".join(lines) + "\n")
    write_json(args.out / "analysis_receipt.json", {"status": "complete", "analyzer_sha256": sha256(Path(__file__)),
        "n_rows": report["n_score_rows"], "n_questions": report["n_questions"], "public_sha256": public_hash,
        "output_sha256": {path.name: sha256(path) for path in sorted(args.out.iterdir()) if path.is_file()}})
    print(json.dumps({"status": "complete", "out": str(args.out), "n_rows": report["n_score_rows"], "n_questions": report["n_questions"]}))


if __name__ == "__main__":
    main()
