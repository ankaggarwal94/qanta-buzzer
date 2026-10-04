#!/usr/bin/env python3
"""Independently evaluate frozen IMCQA policies on protocol-unexposed dev items.

No fitting occurs. Uncertainty resamples questions after averaging their four
rotations; every paired arm and menu remains inside the same resampling unit.
The previously studied source corpus makes this a development transfer screen,
not a new benchmark test or a confirmatory optimal-stopping experiment.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any
import unicodedata

import numpy as np

from scripts import analyze_imcqa_protocol_pilot as prior

old = prior.old
PROTOCOL = "imcqa_frozen_transfer_fp32_v1"
ARMS = ("plain", "wait")
MENUS, REWARDS, OPTIONS = old.MENUS, old.REWARDS, old.OPTIONS
POLICIES = ("frozen_plain_threshold", "native_wait", "frozen_plain_fixed", "always_pass", "plain_final_round")
FROZEN_FIELDS = ("answer_source", "condition", "intercept", "slope", "feature_clip", "selected_threshold",
                 "selected_fixed_policy", "fit_qids", "n_fit_questions", "fit_split", "method")
PUBLIC_KEYS = {"schema_version", "protocol", "source_input_sha256", "main_dataset_sha256", "prior_public_sha256",
               "selection", "frozen_policy", "rewards", "wrong_reward", "pass_reward", "prefix_ids", "jobs"}
JOB_KEYS = {"score_id", "score_index", "source_job_id", "source_prompt_sha256", "qid", "group_id", "split",
            "condition", "menu_id", "prefix_id", "fraction", "round", "reward", "arm", "rotation",
            "option_source_ids", "allowed_actions", "prompt", "prompt_sha256", "block", "wait_label",
            "execution", "source_score_id", "synthetic_case"}


def digest(value: Any) -> str:
    return hashlib.sha256(prior.canonical(value)).hexdigest()


def frozen_parameters(fitted: dict[str, Any], selected_qids: list[str]) -> dict[str, dict[str, Any]]:
    """Read and validate prior fits; never estimate parameters or thresholds."""
    if (fitted.get("fit_split") != "calibration"
            or fitted.get("selection_outcomes_used_for_fitting") is not False):
        raise ValueError("frozen policy lacks calibration-only provenance")
    parameters = [p for p in fitted["parameters"] if p["answer_source"] == "plain"]
    if len(parameters) != 2 or {p["condition"] for p in parameters} != set(MENUS):
        raise ValueError("exactly one frozen plain policy per menu required")
    result = {}
    for p in parameters:
        if (p.get("fit_split") != "calibration" or p.get("selection_uses_only_calibration") is not True
                or p["n_fit_questions"] != len(p["fit_qids"]) or len(set(p["fit_qids"])) != len(p["fit_qids"])
                or set(p["fit_qids"]) & set(selected_qids)):
            raise ValueError("fitted/evaluation question overlap or provenance mismatch")
        if any(type(p[k]) not in (int, float) or not math.isfinite(p[k]) for k in ("intercept", "slope")) or p["slope"] < 0:
            raise ValueError("invalid frozen logistic parameters")
        clip = p["feature_clip"]
        if len(clip) != 2 or not 0 < clip[0] < clip[1] < 1:
            raise ValueError("invalid frozen clipping bounds")
        threshold = p["selected_threshold"]
        if threshold is not None and (type(threshold) not in (int, float) or not 0 <= threshold <= 1):
            raise ValueError("invalid frozen threshold")
        wanted = "fixed_1" if p["condition"] == MENUS[0] else "fixed_2"
        if p["selected_fixed_policy"] != wanted:
            raise ValueError("frozen fixed benchmark differs")
        result[p["condition"]] = {key: p[key] for key in FROZEN_FIELDS}
    return result


def validate_public(package, dataset, source, previous, fitted, config):
    """Reconstruct exact prompts and join gold only after source validation."""
    if (set(package) != PUBLIC_KEYS or package.get("schema_version") != "imcqa-transfer-public-v1"
            or package.get("protocol") != PROTOCOL):
        raise ValueError("transfer public schema/protocol differs")
    if (package.get("rewards") != list(REWARDS) or package.get("wrong_reward") != -1
            or package.get("pass_reward") != 0 or package.get("prefix_ids") != list(old.PREFIX_IDS)):
        raise ValueError("public reward or round contract differs")
    selected = package["selection"]["selected_qids"]
    if set(selected) != {"selection"}:
        raise ValueError("only selection-split development questions permitted")
    qids = selected["selection"]
    if len(qids) != 100 or len(set(qids)) != 100:
        raise ValueError("100 unique selection questions required")
    excluded = {qid for values in previous["selection"]["selected_qids"].values() for qid in values}
    if len(excluded) != 200 or excluded.intersection(qids):
        raise ValueError("prior 200-question protocol cohort overlaps transfer")
    questions = old.frozen_index(dataset)
    if any(qid not in questions or questions[qid]["split"] != "selection" for qid in qids):
        raise ValueError("evaluation set includes unknown or non-selection questions")
    fit = frozen_parameters(fitted, qids)
    if package["frozen_policy"]["parameters"] != [fit[condition] for condition in MENUS]:
        raise ValueError("public frozen policy differs from retained fitted parameters")
    if package["frozen_policy"]["source_fitted_parameters_sha256"] != config["frozen_source_sha256"]["fitted_parameters"]:
        raise ValueError("public frozen fit source hash differs")
    manifest = package["selection"]["manifest"]
    if package["selection"]["manifest_sha256"] != digest(manifest):
        raise ValueError("selection manifest hash differs")
    # Ranking is outcome-blind. The exact frozen salt and complete manifest are
    # independently checked against the eligible original development corpus.
    salt = package["selection"]["salt"]
    if salt != config["selection"]["salt"] or salt != "imcqa-frozen-transfer-20261004":
        raise ValueError("frozen question selection salt differs")
    if package["selection"]["qid_category"] != {qid: questions[qid]["source"]["category"] for qid in qids}:
        raise ValueError("public categories differ from original source")
    excluded_groups = {questions[qid]["group_id"] for qid in excluded}
    eligible = [qid for qid, q in questions.items() if q["split"] == "selection" and qid not in excluded
                and q["group_id"] not in excluded_groups]
    ranked = sorted(eligible, key=lambda qid: (hashlib.sha256(f"{salt}|{qid}".encode()).hexdigest(), qid))
    normalized = {qid: tuple(re.findall(r"[^\W_]+", unicodedata.normalize("NFKC", q["question"]).casefold()))
                  for qid, q in questions.items() if qid in set(ranked) | excluded}
    shingles = {qid: {tokens[i:i+5] for i in range(max(1, len(tokens)-4))} for qid, tokens in normalized.items()}
    references, accepted, skipped = sorted(excluded), [], []
    for qid in ranked:
        collision = next((other for other in references if
                          5*len(shingles[qid] & shingles[other]) >= 4*len(shingles[qid] | shingles[other])), None)
        if collision is not None:
            skipped.append({"qid": qid, "reason": "full_question_5gram_jaccard_ge_0.8", "reference_qid": collision})
            continue
        accepted.append(qid); references.append(qid)
        if len(accepted) == 100:
            break
    if qids != accepted:
        raise ValueError("selection is not the frozen outcome-blind hash ranking")
    manifest_checks = {"excluded_prior_qids": sorted(excluded), "excluded_prior_groups": sorted(excluded_groups),
        "eligible_ranked_qids": ranked, "selected_qids": qids, "skipped": skipped,
        "selected_group_ids": {qid: questions[qid]["group_id"] for qid in qids},
        "selected_normalized_text_sha256": {qid: digest(normalized[qid]) for qid in qids},
        "selected_rank_sha256": {qid: hashlib.sha256(f"{salt}|{qid}".encode()).hexdigest() for qid in qids},
        "jaccard_threshold": .8, "ngram_size": 5, "outcomes_used_for_selection": False}
    if any(manifest.get(key) != value for key, value in manifest_checks.items()):
        raise ValueError("independently reconstructed selection manifest differs")
    expected = {(qid, condition, r, rotation, arm) for qid in qids for condition in MENUS
                for r in range(1, 6) for rotation in range(4) for arm in ARMS}
    source_ids = {job["source_job_id"] for job in package["jobs"]}
    originals = {}
    for row in source["jobs"]:
        if row["job_id"] in source_ids:
            if row["job_id"] in originals:
                raise ValueError("duplicate original source job")
            originals[row["job_id"]] = row
    if set(originals) != source_ids:
        raise ValueError("missing original source job")
    seen, ids, enriched = set(), set(), []
    for index, job in enumerate(package["jobs"]):
        if set(job) != JOB_KEYS:
            raise ValueError("unexpected public job fields")
        key = tuple(job[name] for name in ("qid", "condition", "round", "rotation", "arm"))
        if key not in expected or key in seen or job["score_id"] in ids or job["score_index"] != index:
            raise ValueError("duplicate, unexpected, or unordered public state")
        seen.add(key); ids.add(job["score_id"])
        q = questions[job["qid"]]
        menu = q["menu_index"][job["condition"]]
        prefix = q["prefix_index"][job["prefix_id"]]
        wanted = {"split": "selection", "group_id": q["group_id"], "menu_id": menu["menu_id"],
                  "prefix_id": old.PREFIX_IDS[job["round"]-1], "reward": REWARDS[job["round"]-1],
                  "block": "factorial", "wait_label": "E", "execution": "new", "source_score_id": None,
                  "synthetic_case": None, "allowed_actions": "ABCDE" if job["arm"] == "wait" else "ABCD"}
        if any(job.get(k) != v for k, v in wanted.items()) or not math.isclose(job["fraction"], prefix["fraction"], abs_tol=1e-12):
            raise ValueError("public identity, round, reward, or legal action differs")
        original = originals[job["source_job_id"]]
        if (original["format"] != "mc" or any(original[k] != job[k] for k in
                ("qid", "group_id", "split", "condition", "menu_id", "prefix_id", "fraction"))
                or original["prompt_sha256"] != hashlib.sha256(original["prompt"].encode()).hexdigest()
                or original["prompt_sha256"] != job["source_prompt_sha256"]):
            raise ValueError("original source identity or prompt differs")
        expected_id = job["source_job_id"] + f':transfer:factorial:{job["arm"]}:r{job["rotation"]}:wE'
        if job["score_id"] != expected_id:
            raise ValueError("transfer score identity differs")
        mapping = prior.canonical_mapping(job["rotation"], "E")
        texts = {option["id"]: option["text"] for option in menu["options"]}
        displayed = [{"id": label, "text": texts[identity]} for label, identity in mapping.items()]
        prompt = prior.expected_prompt(prefix["text"], displayed, job["round"], job["arm"], "E")
        if (job["option_source_ids"] != mapping or job["prompt"] != prompt
                or job["prompt_sha256"] != hashlib.sha256(prompt.encode()).hexdigest()):
            raise ValueError("public prompt, option identity, or prefix differs")
        if any("gold" in name or "correct" in name for name in job):
            raise ValueError("public job contains evaluator fields")
        enriched.append({**job, "canonical_gold_option_id": menu["gold_option_id"]})
    if seen != expected or len(enriched) != 8000:
        raise ValueError("incomplete public factorial")
    return enriched, fit


def calibrated_probability(confidence, parameters):
    """Apply the exact frozen logistic map, including its clipping operation."""
    value = min(max(confidence, parameters["feature_clip"][0]), parameters["feature_clip"][1])
    z = parameters["intercept"] + parameters["slope"] * math.log(value/(1-value))
    return 1/(1+math.exp(-z)) if z >= 0 else math.exp(z)/(1+math.exp(z))


def stopping_round(trajectory, policy, parameters):
    """Preserve prior first-threshold semantics, without adding a myopic gate."""
    if [row["round"] for row in trajectory] != list(range(1, 6)):
        raise ValueError("five ordered rounds required")
    if policy == "always_pass":
        return None
    if policy == "plain_final_round":
        return 5
    if policy == "frozen_plain_fixed":
        return int(parameters["selected_fixed_policy"].split("_")[1])
    if policy == "native_wait":
        if any(row["arm"] != "wait" for row in trajectory):
            raise ValueError("native policy needs WAIT candidates")
        return next((row["round"] for row in trajectory if row["chosen_action"] != "E"), None)
    if policy == "frozen_plain_threshold":
        if parameters["selected_threshold"] is None:
            return None
        return next((row["round"] for row in trajectory if
                     calibrated_probability(max(row["canonical_answer_probabilities"].values()), parameters)
                     >= parameters["selected_threshold"]), None)
    raise ValueError("unknown frozen policy")


def evaluate(views, fits, *, samples=20000, seed=1):
    """Compute trajectories and paired question-level summaries without fitting."""
    trajectories = defaultdict(list)
    states = []
    for row in views:
        trajectories[row["qid"], row["condition"], row["rotation"], row["arm"]].append(row)
        confidence = max(row["canonical_answer_probabilities"].values())
        p = calibrated_probability(confidence, fits[row["condition"]]) if row["arm"] == "plain" else None
        states.append({k: row[k] for k in ("qid", "group_id", "condition", "round", "rotation", "arm", "candidate_choice", "candidate_correct")} |
                      {"candidate_confidence": confidence, "calibrated_probability": p,
                       "chosen_action": row["chosen_action"], "native_semantic_action": row["native_semantic_action"]})
    for trajectory in trajectories.values():
        trajectory.sort(key=lambda row: row["round"])
        if [row["round"] for row in trajectory] != list(range(1, 6)):
            raise ValueError("missing or duplicate trajectory round")
    qids = sorted({row["qid"] for row in views})
    episodes = []
    for qid in qids:
        for condition in MENUS:
            for rotation in range(4):
                for policy in POLICIES:
                    arm = "wait" if policy == "native_wait" else "plain"
                    trajectory = trajectories[qid, condition, rotation, arm]
                    stop = stopping_round(trajectory, policy, fits[condition])
                    chosen = trajectory[-1] if stop is None else trajectory[stop-1]
                    committed, correct = stop is not None, stop is not None and bool(chosen["candidate_correct"])
                    episodes.append({"qid": qid, "group_id": chosen["group_id"], "condition": condition,
                        "rotation": rotation, "policy": policy, "committed": committed, "correct": correct,
                        "wrong": committed and not correct, "terminal_pass": not committed,
                        "round": stop, "observed_round": 5 if stop is None else stop,
                        "canonical_choice": chosen["candidate_choice"] if committed else None,
                        "reward": (REWARDS[stop-1] if correct else -1.) if committed else 0.})
    per_question = []
    grouped = defaultdict(list)
    for row in episodes:
        grouped[row["qid"], row["condition"], row["policy"]].append(row)
    for (qid, condition, policy), rows in sorted(grouped.items()):
        if sorted(row["rotation"] for row in rows) != [0, 1, 2, 3]:
            raise ValueError("policy question does not contain all four rotations")
        per_question.append({"qid": qid, "condition": condition, "policy": policy,
            **{key: float(np.mean([row[key] for row in rows])) for key in
               ("reward", "committed", "correct", "wrong", "terminal_pass", "observed_round")}})
    indices = old.bootstrap_indices(len(qids), samples, seed)
    summaries, contrast_rows, screening, cells = [], [], [], {}
    for condition in MENUS:
        for policy in POLICIES:
            rows = sorted((r for r in per_question if r["condition"] == condition and r["policy"] == policy), key=lambda r:r["qid"])
            if [r["qid"] for r in rows] != qids:
                raise ValueError("paired question set differs across policies")
            cells[condition, policy] = rows
            summary = {"condition": condition, "policy": policy, "n_questions": len(qids), "n_episodes": 4*len(qids)}
            for label, key in (("mean_reward", "reward"), ("coverage", "committed"),
                               ("correct_fraction", "correct"), ("mean_observed_round", "observed_round")):
                values = np.asarray([r[key] for r in rows])
                summary[label] = estimate(values.mean(), values[indices].mean(axis=1))
            wrong = np.asarray([r["wrong"] for r in rows]); commits = np.asarray([r["committed"] for r in rows])
            boot_commits = commits[indices].sum(axis=1)
            boot_risk = np.divide(wrong[indices].sum(axis=1), boot_commits,
                                  out=np.full(samples, np.nan), where=boot_commits>0)
            summary["risk"] = estimate(float(wrong.sum()/commits.sum()) if commits.sum() else None, boot_risk)
            summaries.append(summary)
        threshold = np.asarray([r["reward"] for r in cells[condition, "frozen_plain_threshold"]])
        current = {}
        for right in ("native_wait", "frozen_plain_fixed"):
            delta = threshold - np.asarray([r["reward"] for r in cells[condition, right]])
            boot = delta[indices].mean(axis=1)
            result = {"condition": condition, "left": "frozen_plain_threshold", "right": right,
                "n_questions": len(qids), "mean_delta": float(delta.mean()),
                "ci95": np.quantile(boot, [.025, .975]).tolist(),
                "ci98_75": np.quantile(boot, [.00625, .99375]).tolist(), "bonferroni_family_size": 4,
                "interval_method": "question percentile bootstrap; four prespecified reward contrasts"}
            contrast_rows.append(result); current[right] = result
        reward_ci = np.quantile(threshold[indices].mean(axis=1), [.025, .975]).tolist()
        vs_pass = reward_ci[0] > 0
        vs_wait = current["native_wait"]["ci98_75"][0] > 0
        vs_fixed = current["frozen_plain_fixed"]["ci98_75"][0] > 0
        screening.append({"condition": condition, "threshold_positive_vs_pass95": vs_pass,
            "beats_wait_family_interval": vs_wait, "beats_fixed_family_interval": vs_fixed,
            "continue_development_screen": vs_pass and vs_wait,
            "evidence_beyond_frozen_fixed_baseline": vs_pass and vs_wait and vs_fixed,
            "scope": "Exploratory development screen; PASS interval is descriptive and outside the four-contrast family."})
    calibration = []
    for condition in MENUS:
        for arm in ARMS:
            for round_number in (None, 1, 2, 3, 4, 5):
                rows = [r for r in states if r["condition"] == condition and r["arm"] == arm
                        and (round_number is None or r["round"] == round_number)]
                y = np.asarray([r["candidate_correct"] for r in rows], dtype=float)
                result = {"condition": condition, "arm": arm, "round": round_number,
                    "n_questions": len({r["qid"] for r in rows}), "n_states": len(rows), "accuracy": float(y.mean())}
                for field in ("candidate_confidence", "calibrated_probability"):
                    if field == "calibrated_probability" and arm != "plain":
                        continue
                    p = np.asarray([r[field] for r in rows]); clipped = np.clip(p, 1e-12, 1-1e-12)
                    result[field] = {"mean_probability": float(p.mean()), "brier": float(np.mean((p-y)**2)),
                        "log_loss": float(np.mean(-y*np.log(clipped)-(1-y)*np.log1p(-clipped)))}
                calibration.append(result)
    summary = {"schema_version": "imcqa-transfer-analysis-v1", "protocol": PROTOCOL,
        "n_questions": len(qids), "n_score_rows": len(views), "model": "qwen7b",
        "bootstrap_samples": samples, "bootstrap_seed": seed,
        "unit": "Question, averaging four cyclic rotations before paired resampling",
        "scope": "Protocol-unexposed development questions from a previously studied corpus; no refitting or new test-set claim",
        "threshold_semantics": "First calibrated correctness probability >= frozen threshold; otherwise terminal PASS. No additional positive-EV gate.",
        "policy_summaries": summaries, "primary_contrasts": contrast_rows, "screening": screening,
        "state_summaries": calibration,
        "limitations": ["Intervals condition on the previously fitted policies and omit fit-selection uncertainty.",
            "Four cyclic rotations do not enumerate all 24 option orders.",
            "Conditional one-token scores and stateless canonical WAIT-history replay do not measure generated-response frequencies.",
            "Word-fraction reveals are not validated clue boundaries or difficulty levels.",
            "Reward contrasts against native WAIT change both answer elicitation and stopping; the frozen fixed baseline tests the extra value of adaptive timing."]}
    return summary, {"episodes": episodes, "per_question": per_question, "states": states}


def estimate(point, bootstrap):
    finite = bootstrap[np.isfinite(bootstrap)]
    return {"mean": None if point is None else float(point),
            "ci95": np.quantile(finite, [.025, .975]).tolist() if len(finite) else None,
            "defined_resamples": int(len(finite)), "total_resamples": len(bootstrap)}


def write_csv(path, rows):
    if not rows:
        raise ValueError("refuse empty analysis table")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v for k,v in row.items()} for row in rows)


def validate_model(directory, package, jobs, config, public_hash, prior_directory, cache_path):
    """Check complete source-bound scores plus all retained raw numerical gates."""
    receipt = old.load_json(directory / "receipt.json")
    wanted = {"status": "complete", "protocol": PROTOCOL, "model_tag": "qwen7b",
        "public_input_sha256": public_hash, "expected_rows": 8000, "completed_rows": 8000,
        "total_contexts": 8000, "reused_rows": 0, "automatic_retries": 0,
        "sampling": False, "generation": False, "batch_size": 8, "cached": True}
    if any(receipt.get(key) != value for key, value in wanted.items()):
        raise ValueError("transfer completion receipt differs")
    if old.sha256(directory / "scores.jsonl") != receipt["scores_sha256"]:
        raise ValueError("score hash differs from completion receipt")
    metadata = old.load_json(directory / "metadata.json")
    prior_metadata = old.load_json(prior_directory / "metadata.json")
    shared = ("model", "revision", "versions", "model_files_sha256", "dtype", "loaded_dtype", "attention", "tf32",
              "seed", "generation", "sampling", "chat_template_sha256", "assistant_prefix", "action_token_ids")
    if any(metadata.get(key) != prior_metadata.get(key) for key in shared):
        raise ValueError("transfer model/tokenizer/runtime differs from frozen fitted-policy model")
    expected_metadata = {"protocol": PROTOCOL, "public_input_sha256": public_hash, "dtype": "float32",
        "loaded_dtype": "bfloat16", "attention": "eager", "tf32": False, "seed": 1,
        "generation": False, "sampling": False, "assistant_prefix": '{"action":"', "batch_size": 8, "cached": True}
    if any(metadata.get(key) != value for key, value in expected_metadata.items()):
        raise ValueError("transfer numerical metadata differs")
    if metadata.get("source_commit") != receipt.get("source_commit") or not re.fullmatch(r"[0-9a-f]{40}", metadata.get("source_commit", "")):
        raise ValueError("source commit receipt differs")
    root = Path(__file__).resolve().parents[1]
    source_hashes = metadata.get("source_files_sha256", {})
    required = {"scripts/imcqa_transfer_scoring.py", "scripts/imcqa_transfer_design.py", "configs/imcqa_frozen_transfer.json",
        "scripts/imcqa_protocol_scoring.py", "scripts/imcqa_protocol_design.py", "scripts/imcqa_wait_scoring.py",
        "scripts/acl_paired_prompt_scoring.py", "scripts/acl_option_scoring.py", "scripts/jane_gpu_backend.py",
        "scripts/jane_qwen_backend.py", "scripts/jane_output_constraints.py", "configs/imcqa_protocol_pilot.json"}
    if set(source_hashes) != required or any(old.sha256(root/name) != value for name,value in source_hashes.items()):
        raise ValueError("retained inference source hashes differ from analysis checkout")
    if old.sha256(cache_path) != old.CACHE_PREPARE_SHA256:
        raise ValueError("original cache preparation hash differs")
    provenance = old.validate_provenance(metadata, old.load_json(directory / "dtype_promotion.json"),
                                         old.load_json(cache_path)["model_receipts"]["qwen7b"])
    raw = (directory / "scores.jsonl").read_bytes()
    if not raw or not raw.endswith(b"\n"):
        raise ValueError("missing/truncated score file")
    rows = [json.loads(line) for line in raw.splitlines()]
    expected = {job["score_id"]: job for job in jobs}
    lookup, views = {}, []
    for row in rows:
        score_id = row["score_id"]
        if score_id not in expected or score_id in lookup:
            raise ValueError("unknown or duplicate score identity")
        job = expected[score_id]
        if any(row.get(key) != value for key,value in job.items() if key not in {"prompt", "canonical_gold_option_id"}):
            raise ValueError("score/public identity differs")
        if row.get("schema_version") != "imcqa-transfer-scores-v1" or row.get("model_tag") != "qwen7b":
            raise ValueError("score schema/model differs")
        old.validate_vocab_row(row)
        if row["option_token_ids"] != metadata["action_token_ids"]:
            raise ValueError("score action token identity differs")
        lookup[score_id] = row
        views.append(prior.score_view(row, job))
    if len(rows) != 8000 or set(lookup) != set(expected):
        raise ValueError("incomplete score coverage")
    worker_plan = old.load_json(directory / "plan.json")
    plan_checks = {"input_sha256": public_hash, "ordered_score_ids": [row["score_id"] for row in rows],
        "context_sha256": [row["scored_context_sha256"] for row in rows],
        "token_counts": [len(row["scored_input_token_ids"]) for row in rows], "batch_size": 8,
        "production_diagnostic_offsets": [0,4000,7992], "active_rotations": [0,2], "new_rows": 8000, "reused_rows": 0}
    if any(worker_plan.get(key) != value for key,value in plan_checks.items()):
        raise ValueError("production context/order plan differs")
    diagnostic = prior._only_attempt(directory, "diagnostics")
    raw_diagnostic = prior._only_attempt(directory, "diagnostics_raw")
    if raw_diagnostic != {k:v for k,v in diagnostic.items() if k != "gates"}:
        raise ValueError("diagnostic raw evidence differs")
    ids = diagnostic["score_ids"]
    if len(set(ids)) != len(ids) or not 1 <= len(ids) <= 32 or not set(ids) <= set(expected):
        raise ValueError("numerical diagnostic identities differ")
    diag_jobs = [expected[score_id] for score_id in ids]
    for key in ("arm", "condition", "round", "rotation"):
        if {row[key] for row in diag_jobs} != {row[key] for row in jobs}:
            raise ValueError("numerical diagnostic factor missing: "+key)
    lengths = [len(row["scored_input_token_ids"]) for row in rows]
    diag_lengths = [len(lookup[score_id]["scored_input_token_ids"]) for score_id in ids]
    if min(diag_lengths) != min(lengths) or max(diag_lengths) != max(lengths):
        raise ValueError("numerical diagnostics miss context length extrema")
    if diagnostic["cached"] != diagnostic["replay"] or diagnostic.get("physical_batch_size") != 8:
        raise ValueError("diagnostic exact replay or physical batching differs")
    numeric = config["numerical"]
    if any(numeric.get(k) != v for k,v in {"raw_logit_atol": .001, "raw_logit_rtol": 1e-5, "probability_atol": .001}.items()):
        raise ValueError("predeclared numerical tolerances changed")
    checks = {name: prior.compare_numeric(diagnostic["cached"], diagnostic[right], diag_jobs, numeric)
              for name,right in (("cached_uncached", "uncached"), ("cached_single", "singles"), ("permutation", "permuted_aligned"))}
    checks["permutation_single"] = prior.compare_numeric(diagnostic["permuted_aligned"], diagnostic["singles"], diag_jobs, numeric)
    production_from_scores = [{"logits": [lookup[score_id]["raw_action_logits"][label] for label in old.ACTIONS]} for score_id in ids]
    checks["production_diagnostic"] = prior.compare_numeric(production_from_scores, diagnostic["cached"], diag_jobs, numeric)
    pd = prior._only_attempt(directory, "production_diagnostics")
    pd_raw = prior._only_attempt(directory, "production_diagnostics_raw")
    if pd_raw != {k:v for k,v in pd.items() if k not in {"gate", "single_gate"}}:
        raise ValueError("production raw evidence differs")
    production_ids = [rows[i]["score_id"] for offset in (0,4000,7992) for i in range(offset,offset+8)]
    if (pd["score_ids"] != production_ids or pd["offsets"] != [0,4000,7992]
            or pd.get("batch_size") != 8 or pd.get("actual_production_batches") is not True
            or pd["production"] != pd["diagnostic"]):
        raise ValueError("actual production batch exact replay/coverage differs")
    for score_id, output in zip(production_ids, pd["production"]):
        row = lookup[score_id]
        if output["logits"] != [row["raw_action_logits"][label] for label in old.ACTIONS]:
            raise ValueError("production raw logits differ from score rows")
        if any(row.get(key) != value for key,value in output.items() if key != "logits"):
            raise ValueError("production vocabulary evidence differs from score rows")
    prod_jobs = [expected[score_id] for score_id in production_ids]
    checks["production_single"] = prior.compare_numeric(pd["production"], pd["single_reference"], prod_jobs, numeric)
    checks["production_exact_replay"] = prior.compare_numeric(pd["production"], pd["diagnostic"], prod_jobs, numeric)
    for index, offset in enumerate((0,4000,7992)):
        batch = old.load_json(directory / "attempts" / f"000_production_batch_{offset:05d}_raw.json")
        start,end = index*8,(index+1)*8
        if (batch.get("offset") != offset or batch.get("actual_production_batch") is not True or batch.get("batch_size") != 8
                or any(batch.get(key) != pd[key][start:end] for key in ("score_ids", "production", "diagnostic", "single_reference"))):
            raise ValueError("actual batch raw evidence differs from aggregate")
    active = prior._only_attempt(directory, "active_only")
    selected = package["selection"]["selected_qids"]["selection"]
    active_qids = sorted(selected, key=lambda qid:(hashlib.sha256(f"transfer-live|1|{qid}".encode()).hexdigest(),qid))[:4]
    if active["qids"] != active_qids or active["episodes"] != 16 or worker_plan["active_qids"] != active_qids:
        raise ValueError("active-only episode selection differs")
    active_ids = []
    for qid in active_qids:
        for condition in MENUS:
            for rotation in (0,2):
                trajectory = sorted((row for row in views if row["qid"] == qid and row["condition"] == condition
                    and row["rotation"] == rotation and row["arm"] == "wait"),key=lambda row:row["round"])
                if [row["round"] for row in trajectory] != [1,2,3,4,5]:
                    raise ValueError("active trajectory incomplete")
                for row in trajectory:
                    active_ids.append(row["score_id"])
                    if row["chosen_action"] != "E":
                        break
    if [row["score_id"] for row in active["rows"]] != active_ids or not 16 <= len(active_ids) <= 80:
        raise ValueError("active first-commit/PASS sequence differs")
    active_checks = []
    for index, row in enumerate(active["rows"]):
        source = lookup[row["score_id"]]
        if any(row.get(k) != source[k] for k in ("qid","condition","rotation","round","chosen_action")):
            raise ValueError("live replay identity/action differs")
        production = [{"logits":[source["raw_action_logits"][label] for label in old.ACTIONS]}]
        evidence = old.load_json(directory / "attempts" / f"000_active_{index:03d}_raw.json")
        if (evidence["score_ids"] != [row["score_id"]] or evidence["production"] != production
                or evidence["reference"] != [row["live"]] or evidence["live"] != row["live"]):
            raise ValueError("live raw evidence differs")
        active_checks.append(prior.compare_numeric(production, [row["live"]], [expected[row["score_id"]]], numeric))
    return views, {"passed": True, "n_score_rows": len(rows), "scores_sha256": receipt["scores_sha256"],
        "metadata_sha256": old.sha256(directory/"metadata.json"), "elapsed_seconds": receipt["elapsed_seconds"],
        "source_commit": receipt["source_commit"], "verified_source_files_sha256": source_hashes,
        "provenance": provenance, "numerical_checks": checks,
        "active_only": {"episodes": 16, "states": len(active_ids), "checks": active_checks},
        "independent_source_gold_join": True, "all_score_softmax_argmax_checks_passed": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("public", "source", "evaluator", "fitted", "prior-public", "prior-model-dir", "cache-receipt", "plan", "run-dir", "out"):
        parser.add_argument("--"+name, required=True, type=Path)
    args = parser.parse_args()
    config = old.load_json(args.plan)
    paths = {"main_jobs": args.source, "main_dataset": args.evaluator,
             "prior_public": args.prior_public, "fitted_parameters": args.fitted}
    for key,path in paths.items():
        if old.sha256(path) != config["frozen_source_sha256"][key]:
            raise ValueError("frozen input file hash differs: "+key)
    package = old.load_json(args.public)
    source = old.load_json(args.source)
    dataset = old.load_json(args.evaluator)
    previous = old.load_json(args.prior_public)
    fitted = old.load_json(args.fitted)
    for key,field in (("main_jobs","source_input_sha256"),("main_dataset","main_dataset_sha256"),("prior_public","prior_public_sha256")):
        if package[field] != config["frozen_source_sha256"][key]:
            raise ValueError("public source input hash differs")
    jobs, fits = validate_public(package,dataset,source,previous,fitted,config)
    views, audit = validate_model(args.run_dir,package,jobs,config,old.sha256(args.public),args.prior_model_dir,args.cache_receipt)
    settings = config["analysis"]
    if settings["bootstrap_samples"] != 20000 or settings["bootstrap_seed"] != 1:
        raise ValueError("frozen bootstrap design differs")
    contrasts = {(row["condition"], row["left"], row["right"]) for row in settings["primary_contrasts"]}
    if (len(settings["primary_contrasts"]) != 4 or contrasts != {(menu,"frozen_plain_threshold",baseline)
            for menu in MENUS for baseline in ("native_wait","frozen_plain_fixed")}
            or settings["primary_family_size"] != 4 or settings["primary_confidence_level"] != .9875):
        raise ValueError("frozen primary comparison family differs")
    summary, records = evaluate(views,fits,samples=settings["bootstrap_samples"],seed=settings["bootstrap_seed"])
    args.out.mkdir(parents=True,exist_ok=False)
    old.write_json(args.out/"summary.json", summary)
    old.write_json(args.out/"validation.json", audit)
    for name,rows in records.items():
        write_csv(args.out/(name+".csv"), rows)
    for name in ("policy_summaries", "primary_contrasts", "state_summaries", "screening"):
        write_csv(args.out/(name+".csv"), summary[name])
    old.write_json(args.out/"analysis_receipt.json", {"status":"complete", "fitting_performed":False,
        "analyzer_sha256":old.sha256(Path(__file__)), "plan_sha256":old.sha256(args.plan),
        "public_sha256":old.sha256(args.public), "input_sha256":{k:old.sha256(p) for k,p in paths.items()},
        "outputs_sha256":{path.name:old.sha256(path) for path in sorted(args.out.iterdir())}})
    print(json.dumps({"status":"complete", "n_questions":summary["n_questions"], "n_score_rows":len(views), "output":str(args.out)}))


if __name__ == "__main__":
    main()
