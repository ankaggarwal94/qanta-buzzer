#!/usr/bin/env python3
"""Explicit partial-model analysis; never relax the complete-pilot acceptance gate.

Complete models pass the unchanged full validator. Failed models are retained as
hash-bound execution evidence and contribute no scientific score observations.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from scripts import analyze_imcqa_protocol_pilot as a


SOURCE_FILES = ("scripts/imcqa_protocol_scoring.py", "scripts/imcqa_protocol_design.py", "scripts/imcqa_wait_scoring.py",
    "scripts/acl_paired_prompt_scoring.py", "scripts/acl_option_scoring.py", "scripts/jane_gpu_backend.py",
    "scripts/jane_qwen_backend.py", "scripts/jane_output_constraints.py", "configs/imcqa_protocol_pilot.json")


def numerical_failure_summary(diagnostic, jobs, config):
    """Describe every retained comparison without silently replacing its gate."""
    lookup = {job["score_id"]: job for job in jobs}
    ids = diagnostic["score_ids"]
    if not ids or len(set(ids)) != len(ids) or not set(ids) <= set(lookup):
        raise ValueError("failed diagnostic identity is invalid")
    results = []
    for comparison in ("uncached", "singles", "permuted_aligned", "replay"):
        left, right = diagnostic["cached"], diagnostic[comparison]
        if len(left) != len(right) or len(left) != len(ids):
            raise ValueError("failed diagnostic rows are truncated")
        maximum_raw = maximum_ratio = maximum_action = maximum_candidate = 0.
        raw_failed, probability_failed, action_flips, candidate_flips, violations = 0, 0, 0, 0, []
        for first, second, score_id in zip(left, right, ids):
            x, y = first["logits"], second["logits"]
            if len(x) != 5 or len(y) != 5 or any(type(value) not in (int, float) or not math.isfinite(value) for value in x + y):
                raise ValueError("failed diagnostic logits are malformed")
            job = lookup[score_id]
            deltas = [abs(v - w) for v, w in zip(x, y)]
            ratio = max(delta / (config["raw_logit_atol"] + config["raw_logit_rtol"] * abs(v)) for delta, v in zip(deltas, y))
            p, q = (a.old.softmax(dict(zip(a.ACTIONS, values)), tuple(job["allowed_actions"])) for values in (x, y))
            candidate_labels = tuple(label for label in a.ACTIONS if label in job["option_source_ids"])
            cp, cq = (a.old.softmax(dict(zip(a.ACTIONS, values)), candidate_labels) for values in (x, y))
            action_delta = max(abs(p[label] - q[label]) for label in p)
            candidate_delta = max(abs(cp[label] - cq[label]) for label in cp)
            af, cf = max(p, key=p.get) != max(q, key=q.get), max(cp, key=cp.get) != max(cq, key=cq.get)
            raw_bad, prob_bad = ratio > 1., max(action_delta, candidate_delta) > config["probability_atol"]
            maximum_raw, maximum_ratio = max(maximum_raw, max(deltas)), max(maximum_ratio, ratio)
            maximum_action, maximum_candidate = max(maximum_action, action_delta), max(maximum_candidate, candidate_delta)
            raw_failed += raw_bad
            probability_failed += prob_bad
            action_flips += af
            candidate_flips += cf
            if raw_bad or prob_bad or af or cf:
                violations.append({"score_id": score_id, **{key: job[key] for key in ("arm", "block", "round", "rotation", "wait_label")},
                    "max_raw_logit_difference": max(deltas), "maximum_tolerance_ratio": ratio,
                    "max_action_probability_difference": action_delta, "max_candidate_probability_difference": candidate_delta,
                    "action_argmax_changed": af, "candidate_argmax_changed": cf})
        results.append({"comparison": "cached_vs_" + comparison, "rows": len(ids), "raw_logit_failed_rows": raw_failed,
            "probability_failed_rows": probability_failed, "action_argmax_changes": action_flips, "candidate_argmax_changes": candidate_flips,
            "max_raw_logit_difference": maximum_raw, "maximum_tolerance_ratio": maximum_ratio,
            "max_action_probability_difference": maximum_action, "max_candidate_probability_difference": maximum_candidate,
            "passed": not (raw_failed or probability_failed or action_flips or candidate_flips), "violations": violations})
    return results


def validate_failure(directory, tag, public_hash, jobs, config, prior_directory):
    """Bind the observed pre-production numerical failure; discard no evidence."""
    receipt = a.old.load_json(directory / "receipt.json")
    expected = {"status": "failed", "protocol": a.PROTOCOL, "model_tag": tag, "public_input_sha256": public_hash,
        "expected_rows": 4032, "completed_rows": 0, "total_contexts": 5232, "reused_rows": 1200,
        "automatic_retries": 0, "sampling": False, "generation": False}
    if any(receipt.get(key) != value for key, value in expected.items()) or not receipt.get("error"):
        raise ValueError("partial path requires the declared pre-production failed receipt")
    score_path = directory / "scores.jsonl"
    if score_path.read_bytes() != b"" or a.old.sha256(score_path) != receipt["scores_sha256"]:
        raise ValueError("failed model unexpectedly retains production scores or has a score hash mismatch")
    metadata = a.old.load_json(directory / "metadata.json")
    prior = a.old.load_json(prior_directory / "metadata.json")
    keys = ("model", "revision", "versions", "model_files_sha256", "dtype", "loaded_dtype", "attention", "tf32", "seed",
        "generation", "sampling", "chat_template_sha256", "assistant_prefix", "action_token_ids")
    if any(metadata.get(key) != prior.get(key) for key in keys) or metadata.get("protocol") != a.PROTOCOL or metadata.get("public_input_sha256") != public_hash:
        raise ValueError("failed model identity/runtime differs")
    root = Path(__file__).resolve().parents[1]
    if metadata.get("source_files_sha256") != {name: a.old.sha256(root / name) for name in SOURCE_FILES}:
        raise ValueError("failed worker inference sources differ")
    cache = directory.parent / "cache_prepare_receipt.json"
    if a.old.sha256(cache) != a.old.CACHE_PREPARE_SHA256:
        raise ValueError("failed worker cache provenance differs")
    a.old.validate_provenance(metadata, a.old.load_json(directory / "dtype_promotion.json"), a.old.load_json(cache)["model_receipts"][tag])
    diagnostic = a._only_attempt(directory, "diagnostics_raw")
    if len(diagnostic["score_ids"]) > config["execution"]["numerical_contexts_max_per_model"]:
        raise ValueError("failed numerical diagnostic exceeds frozen bound")
    numerical = numerical_failure_summary(diagnostic, [job for job in jobs if job["execution"] == "new"], config["numerical_checks"])
    if all(row["passed"] for row in numerical):
        raise ValueError("failed model has no independently reproduced numeric violation")
    return {"status": "failed", "receipt": receipt, "error": receipt["error"], "n_production_rows_retained": 0,
        "excluded_from_scientific_analysis": True, "numerical_diagnostics": numerical,
        "interpretation": "The frozen numerical contract failed before production; this is not an estimate of answer quality. No failed-model observations enter the scientific summaries.",
        "evidence_sha256": {str(path.relative_to(directory)): a.old.sha256(path) for path in sorted(directory.rglob("*")) if path.is_file()}}


def partial_scope(expected_models, validated_models):
    """Require an explicit proper subset; never relabel it a complete experiment."""
    if (not validated_models or len(set(validated_models)) != len(validated_models)
        or not set(validated_models) < set(expected_models) or set(expected_models) != {"qwen3b", "qwen7b"}):
        raise ValueError("partial analysis requires an explicit nonempty proper subset of the two expected models")
    return {"schema_version": "imcqa-protocol-partial-analysis-v1", "status": "partial",
        "evidence_scope": "exploratory_partial_already_inspected_development_questions", "expected_models": sorted(expected_models),
        "validated_models": sorted(validated_models), "failed_models": sorted(set(expected_models) - set(validated_models)),
        "expected_n_score_rows": len(expected_models) * 5232, "n_score_rows": len(validated_models) * 5232,
        "expected_n_new_score_rows": len(expected_models) * 4032, "expected_n_reused_score_rows": len(expected_models) * 1200,
        "n_new_score_rows": len(validated_models) * 4032, "n_reused_score_rows": len(validated_models) * 1200,
        "n_validated_models": len(validated_models)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("public", "prior-public", "frozen-source", "gold", "config", "prior-outputs", "outputs", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--validated-models", nargs="+", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = a.old.load_json(args.config)
    scope = partial_scope(list(config["models"]), args.validated_models)
    if a.old.sha256(args.config) != a.old.sha256(root / "configs/imcqa_protocol_pilot.json"):
        raise ValueError("partial analysis config differs from frozen source")
    if a.old.sha256(args.prior_public) != a.OLD_PUBLIC_SHA:
        raise ValueError("prior public hash differs")
    for path, key in ((args.frozen_source, "public/main_jobs.json"), (args.gold, "evaluator/main_dataset.json")):
        if a.old.sha256(path) != config["frozen_source_sha256"][key]:
            raise ValueError("frozen source differs: " + key)
    package, prior = a.old.load_json(args.public), a.old.load_json(args.prior_public)
    dataset, source = a.old.load_json(args.gold), a.old.load_json(args.frozen_source)
    jobs = a.validate_public(package, dataset, source, prior, config)
    prior_config = a.old.load_json(root / "configs/imcqa_wait_pilot.json")
    prior_jobs = a.old.validate_jobs([a.old.normalize_public_job(job) for job in prior["jobs"]], dataset,
        n_per_split=prior_config["n_per_split"], n_diagnostic_per_split=prior_config["n_diagnostic_per_split"])
    a.old.validate_source_binding(prior_jobs, source)
    del dataset, source
    all_rows, audits, prior_audits, failures = {}, {}, {}, {}
    public_hash = a.old.sha256(args.public)
    for model in config["models"]:
        previous = args.prior_outputs / model
        if a.old.sha256(previous / "scores.jsonl") != config["reuse"]["prior_scores_sha256"][model]:
            raise ValueError("prior score reuse identity differs")
        previous_rows, prior_audits[model] = a.old.validate_model_output(previous, model, prior, prior_jobs, prior_config, a.OLD_PUBLIC_SHA)
        if model in scope["validated_models"]:
            all_rows[model], audits[model] = a.validate_new_model(args.outputs / model, model, package, jobs,
                config, public_hash, previous, previous_rows)
        else:
            failures[model] = validate_failure(args.outputs / model, model, public_hash, jobs, config, previous)
    if sum(map(len, all_rows.values())) != scope["n_score_rows"]:
        raise ValueError("validated row count differs from explicit partial scope")
    report = a.analyze_rows(all_rows, samples=config["analysis"]["bootstrap_samples"], seed=config["analysis"]["bootstrap_seed"])
    args.out.mkdir(parents=True, exist_ok=False)
    for name, values in report.pop("records").items():
        a.old.write_csv(args.out / (name + ".csv"), values)
    report.update(**scope, protocol=a.PROTOCOL, n_questions=40, n_synthetic_contexts_per_model=32,
        audits=audits, prior_audits=prior_audits, failures=failures, config=config,
        input_sha256={key: a.old.sha256(getattr(args, key)) for key in ("public", "prior_public", "frozen_source", "gold", "config")},
        limitations=["The originally planned two-model experiment is incomplete. New scientific summaries include only the explicitly validated model; failed-model diagnostics are excluded.",
            "No cross-model conclusion can be drawn from this partial matched-prompt run.",
            "Same-question four-rotation observations are retained inside the question-level bootstrap; 40 questions remain the sample size.",
            "Candidate softmax is not calibrated correctness, and constrained actions are not complete-response sampling frequencies.",
            *config["interpretation_limits"]])
    a.old.write_json(args.out / "report.json", report)
    lines = ["PARTIAL MATCHED PROTOCOL ANALYSIS", f"Expected models: {', '.join(scope['expected_models'])}.",
        f"Validated models: {', '.join(scope['validated_models'])}.",
        f"Analyzed {scope['n_score_rows']} of {scope['expected_n_score_rows']} planned score contexts; failed-model scores contribute zero observations.", ""]
    for model, failure in failures.items():
        lines.extend([f"{model}: {failure['error']}", failure["interpretation"]])
        for check in failure["numerical_diagnostics"]:
            lines.append(f"{check['comparison']}: passed={check['passed']}; raw-logit failed rows={check['raw_logit_failed_rows']}; "
                f"maximum raw difference={check['max_raw_logit_difference']:.10g}; maximum tolerance ratio={check['maximum_tolerance_ratio']:.6g}; "
                f"action/candidate argmax changes={check['action_argmax_changes']}/{check['candidate_argmax_changes']}.")
    (args.out / "FAILED_MODELS.txt").write_text("\n".join(lines) + "\n")
    (args.out / "FINDINGS.txt").write_text("\n".join(lines + ["", *report["limitations"]]) + "\n")
    a.old.write_json(args.out / "analysis_receipt.json", {"status": "partial", "analyzer_sha256": a.old.sha256(Path(__file__)),
        "complete_analyzer_sha256": a.old.sha256(Path(a.__file__)), "expected_models": scope["expected_models"],
        "validated_models": scope["validated_models"], "failed_models": scope["failed_models"],
        "expected_n_score_rows": scope["expected_n_score_rows"], "n_score_rows": scope["n_score_rows"],
        "public_sha256": public_hash, "output_sha256": {path.name: a.old.sha256(path) for path in sorted(args.out.iterdir()) if path.is_file()}})
    print(json.dumps({"status": "partial", "out": str(args.out), "validated_models": scope["validated_models"], "n_score_rows": scope["n_score_rows"]}))


if __name__ == "__main__":
    main()
