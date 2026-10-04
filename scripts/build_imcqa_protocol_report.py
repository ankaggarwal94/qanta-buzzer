#!/usr/bin/env python3
"""Render verified CPU decomposition and matched protocol analyses as one HTML.

This renderer does not fit policies or recompute scientific estimates. Report
hashes must match complete analyzer receipts. Missing inputs fail closed.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path
import re
from typing import Any

from scripts.build_imcqa_results_report import read_json, sha256


MODELS = {"qwen3b": "Qwen 3B", "qwen7b": "Qwen 7B"}
MENUS = {"independent_pool": "Independent", "same_category_pool": "Same category"}
ARMS = {"plain": "Plain MCQA", "forced": "Game, answer now", "wait": "Game, WAIT allowed"}
EVIDENCE_SCOPES = {
    "imcqa-wait-decomposition-v1": "exploratory_development_only",
    "imcqa-protocol-analysis-v1": "exploratory_already_inspected_development_questions",
    "imcqa-protocol-partial-analysis-v1": "exploratory_partial_already_inspected_development_questions",
}


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def num(value: float | None, digits: int = 3) -> str:
    return "undefined" if value is None else f"{value:.{digits}f}"


def pct(value: float | None, digits: int = 2) -> str:
    return "undefined" if value is None else f"{100 * value:.{digits}f}%"


def estimate(value: dict[str, Any], percent: bool = False) -> str:
    formatter = pct if percent else num
    point = formatter(value["mean"])
    interval = value.get("ci95")
    return point + (f" [{formatter(interval[0])}, {formatter(interval[1])}]" if interval is not None else " [interval undefined]")


def percentage_points(value: dict[str, Any]) -> str:
    point = f"{100 * value['mean']:.2f}"
    interval = value.get("ci95")
    return point + (f" [{100 * interval[0]:.2f}, {100 * interval[1]:.2f}]" if interval is not None else " [interval undefined]")


def table(headers: list[str], rows: list[list[Any]], caption: str) -> str:
    if any(len(row) != len(headers) for row in rows):
        raise ValueError("HTML table width differs from header width")
    return (f"<div class='table-wrap'><table><caption>{esc(caption)}</caption><thead><tr>"
            + "".join(f"<th scope='col'>{esc(label)}</th>" for label in headers)
            + "</tr></thead><tbody>"
            + "".join("<tr>" + "".join(f"<td>{esc(cell)}</td>" for cell in row) + "</tr>" for row in rows)
            + "</tbody></table></div>")


def unique(rows: list[dict[str, Any]], **filters: Any) -> dict[str, Any]:
    matches = [row for row in rows if all(row.get(key) == value for key, value in filters.items())]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one row for {filters}, found {len(matches)}")
    return matches[0]


def verified_report(directory: Path, schema: str) -> tuple[dict[str, Any], dict[str, Any]]:
    report_path = directory / "report.json"
    receipt = read_json(directory / "analysis_receipt.json")
    required_status = "partial" if schema == "imcqa-protocol-partial-analysis-v1" else "complete"
    if receipt.get("status") != required_status or receipt.get("output_sha256", {}).get("report.json") != sha256(report_path):
        raise ValueError(f"Unverified or incomplete report: {directory}")
    report = read_json(report_path)
    if schema not in EVIDENCE_SCOPES or report.get("schema_version") != schema or report.get("evidence_scope") != EVIDENCE_SCOPES[schema]:
        raise ValueError(f"Unexpected report scope/schema: {directory}")
    return report, receipt


def decomposition_section(report: dict[str, Any]) -> str:
    rows, outcome_rows, calibration_rows = [], [], []
    for model in MODELS:
        for menu in MENUS:
            common = {"model": model, "condition": menu, "split": "selection"}
            summary = unique(report["decomposition_summaries"], **common)
            native = unique(report["policy_summaries"], policy="native_wait", **common)
            oracle = unique(report["policy_summaries"], policy="hindsight_same_wait_or_pass", **common)
            calibrated = unique(report["policy_summaries"], policy="calibrated_myopic_positive_ev_vs_pass", **common)
            paired = unique(report["paired_comparisons"], comparison="calibrated_myopic_positive_ev_vs_pass_minus_native_wait", **common)
            rows.append([MODELS[model], MENUS[menu], summary["n_questions"], num(native["mean_reward"]), num(oracle["mean_reward"]),
                         summary["n_correct_candidate_any_round"], summary["n_visited_correct_candidate_skipped"],
                         summary["n_wrong_early_with_later_correct"]])
            counts = summary["native_outcome_counts"]
            outcome_rows.append([MODELS[model], MENUS[menu], *[counts.get(key, 0) for key in
                ("correct_early", "correct_terminal", "wrong_early", "wrong_terminal", "terminal_pass")],
                num(summary["mean_wrong_answer_penalty_component"]), num(summary["mean_forgone_correct_reward_component"]),
                num(summary["mean_correct_answer_delay_component"])])
            reliability = unique(report["calibration_summaries"], scope="all_counterfactual_states", **common)
            calibration_rows.append([MODELS[model], MENUS[menu], num(reliability["raw_conditional_confidence"]["brier"]),
                num(reliability["calibrated_correctness"]["brier"]), num(calibrated["mean_reward"]),
                calibrated["n_committed"], estimate(paired["differences"]["mean_reward"])])
    return ("<section id='decomposition'><h2>What the saved WAIT outputs already establish</h2>"
            "<p>The first stage used no new model inference. It keeps the original WAIT prompt and its top A–D candidate at each round. "
            "The hindsight reference chooses the earliest correct candidate, or PASS if none is correct. This is a gold-assisted upper bound, "
            "not a deployable policy or a forecast of achievable reward.</p>"
            + table(["Model", "Menu", "Questions", "Native reward", "Hindsight reward", "Correct candidate ever", "Visited correct candidate skipped", "Wrong early, later correct"], rows,
                    "Original pilot selection split: 100 questions per model/menu; availability and skipped-opportunity columns are question counts")
            + "<p>“Visited” includes only states through the native first answer, or all five states on a terminal PASS. "
              "“Later correct” after an early wrong answer is counterfactual; that question already ended. "
              "For every trajectory, hindsight reward − native reward equals wrong-answer penalty + forgone correct reward after wrong/PASS + delay cost on correct commitments.</p>"
            + "<details><summary>Native outcomes and additive regret components</summary>"
            + table(["Model", "Menu", "Correct early", "Correct terminal", "Wrong early", "Wrong terminal", "PASS", "Wrong penalty", "Forgone reward", "Correct delay"], outcome_rows,
                    "Outcome counts sum to 100. The three reward components sum to mean hindsight-minus-native regret.")
            + "</details><h3>A small calibration-only baseline</h3>"
            + table(["Model", "Menu", "Raw Brier", "Calibrated Brier", "Myopic reward", "Myopic commitments / 100", "Reward minus native [95% CI]"], calibration_rows,
                    "Calibration fits use only the separate 100-question calibration split. Brier diagnostics use 500 counterfactual selection states per cell.")
            + "<p>The fixed two-parameter monotone logistic fit maps conditional candidate probability to correctness. "
              "The decision rule answers at the first round where calibrated p × (correct reward + 1) − 1 is positive; otherwise it waits or passes. "
              "This compares answering with PASS and omits continuation value. Better Brier score therefore need not produce a better stopping policy. "
              "Intervals hold the calibration fit fixed and omit fitting uncertainty.</p></section>")


def terminal_table(report: dict[str, Any], split: str, models: tuple[str, ...] = tuple(MODELS)) -> str:
    rows = []
    for model in models:
        for menu in MENUS:
            values = [unique(report["round_summaries"], model=model, condition=menu, split=split, arm=arm, round=5) for arm in ARMS]
            contrast = unique(report["round_prompt_contrasts"], model=model, condition=menu, split=split, contrast="plain_minus_forced", round=5)
            rows.append([MODELS[model], MENUS[menu], values[0]["n_questions"], *[estimate(row["candidate_correct"], True) for row in values],
                         percentage_points(contrast["candidate_accuracy_difference"])])
    return table(["Model", "Menu", "Questions", "Plain accuracy [95% CI]", "Forced-game accuracy [95% CI]", "WAIT-candidate accuracy [95% CI]", "Plain − forced, pp [95% CI]"], rows,
                 f"{split.capitalize()} split, full-question round: average over four cyclic candidate rotations within each question")


def protocol_section(report: dict[str, Any], models: tuple[str, ...] = tuple(MODELS)) -> str:
    reward_rows = []
    for model in models:
        for menu in MENUS:
            for split in ("selection", "calibration"):
                common = {"model": model, "condition": menu, "split": split, "arm": "wait"}
                native = unique(report["policy_summaries"], policy="native_wait", **common)
                oracle = unique(report["policy_summaries"], policy="candidate_hindsight_or_pass", **common)
                final = unique(report["policy_summaries"], policy="candidate_fixed_5", **common)
                reward_rows.append([MODELS[model], MENUS[menu], split, native["n_questions"],
                    estimate({"mean": native["mean_reward"], **native["bootstrap"]["mean_reward"]}),
                    pct(native["coverage"]), pct(native["risk"]), num(final["mean_reward"]), num(oracle["mean_reward"])])
    return ("<section id='matched'><h2>The matched prompt experiment</h2>"
            "<p>All three conditions use identical pinned model weights, FP32 arithmetic, question prefixes, answer content, "
            "chat formatting, and the assistant boundary <code>{&quot;action&quot;:&quot;</code>. Plain MCQA requires an answer; "
            "the game’s forced condition includes the game instructions but requires an answer now; the WAIT condition permits continuation. "
            "The matched intervention identifies the effect of each complete instruction condition, not the contribution of an individual sentence.</p>"
            + terminal_table(report, "selection", models)
            + "<p>WAIT-candidate accuracy is the top answer after conditioning A–D on answering; it does not count WAIT/PASS as an answer. "
              "All displayed intervals resample questions, retaining their rotations together. Four rotations do not create four independent questions.</p>"
            + "<details><summary>Calibration split terminal accuracy</summary>" + terminal_table(report, "calibration", models) + "</details>"
            + "<h3>Stopping reward under the WAIT prompt</h3>"
            + table(["Model", "Menu", "Split", "Questions", "Native reward [95% CI]", "Commitment rate", "Error / commitment", "Same-candidate final reward", "Same-candidate hindsight"], reward_rows,
                    "Average across the four cyclic rotations per question. PASS earns zero. Each candidate reference uses its own WAIT prompt’s predictions.")
            + "<p>Correct-answer rewards are 1.0, 0.8, 0.6, 0.4, and 0.2 across five rounds; any wrong answer earns −1. "
              "At the last round, answering beats PASS only when calibrated correctness probability exceeds 1/1.2 = 83.33%. "
              "At earlier rounds, optimal stopping would additionally require the value of future information.</p></section>")


def sensitivity_section(report: dict[str, Any], models: tuple[str, ...] = tuple(MODELS)) -> str:
    rotation_rows, label_rows = [], []
    for split in ("selection", "calibration"):
        for model in models:
            for menu in MENUS:
                common = {"model": model, "condition": menu, "split": split}
                for arm in ARMS:
                    row = unique(report["rotation_contrasts"], arm=arm, **common)
                    orbit = unique(report["rotation_consistency"], arm=arm, **common)
                    rotation_rows.append([MODELS[model], MENUS[menu], split, ARMS[arm], row["n_questions"],
                        estimate(row["candidate_changed"], True), estimate(row["semantic_action_changed"], True),
                        estimate(orbit["semantic_action_inconsistent"], True)])
                row = unique(report["action_label_contrasts"], **common)
                label_policy = unique(report["action_label_policy_summaries"], wait_label="A", **common)
                label_contrast = unique(report["action_label_policy_contrasts"], contrast="A_wait_minus_E_wait_rotation0", **common)
                label_rows.append([MODELS[model], MENUS[menu], split, row["n_questions"], row["n_states"],
                    estimate(row["semantic_action_changed"], True), estimate(row["wait_decision_changed"], True),
                    estimate(row["candidate_changed"], True), num(label_policy["mean_reward"]),
                    estimate(label_contrast["differences"]["mean_reward"])])
    return ("<section id='sensitivity'><h2>Candidate positions and the WAIT label</h2>"
            + table(["Model", "Menu", "Split", "Prompt", "Questions", "Candidate changed [95% CI]", "Semantic action changed [95% CI]", "Any action disagreement across four orders [95% CI]"], rotation_rows,
                    "Changed compares each nonzero rotation with rotation 0, averaging over three comparisons and five rounds within a question. The last column checks the full four-order orbit.")
            + "<p>Predictions are mapped back to candidate identities before comparison. Cyclic rotations balance the position of each candidate, "
              "but cover only four of the 24 possible candidate orders. A changed label alone is not counted as a changed answer.</p>"
            + table(["Model", "Menu", "Split", "Questions", "Matched states", "Semantic action changed [95% CI]", "WAIT/PASS decision changed [95% CI]", "Answer candidate changed [95% CI]", "Native reward with A = WAIT", "A-WAIT − E-WAIT reward [95% CI]"], label_rows,
                    "Targeted A/E relabeling: WAIT/PASS moves from E to A and the displaced candidate from A to E, with candidate content order held fixed")
            + f"<p>The label intervention includes 400 newly scored states per verified model, {400 * len(models):,} reported here, covering all 40 questions, both menus, and all five rounds. "
              "Both the state comparisons and episodic reward contrasts use rotation 0 in each labeling; the E-WAIT baseline here is not averaged over rotations. "
              "The intervention changes both the continuation label and a candidate label. It diagnoses label sensitivity without isolating a pure preference for the E token.</p></section>")


def comprehension_section(report: dict[str, Any], directory: Path, receipt: dict[str, Any],
                          models: tuple[str, ...] = tuple(MODELS)) -> str:
    path = directory / "comprehension_records.csv"
    if receipt["output_sha256"].get(path.name) != sha256(path):
        raise ValueError("synthetic outcomes CSV differs from analyzer receipt")
    with path.open(newline="") as stream:
        cases = list(csv.DictReader(stream))
    if len(cases) != 32 * len(models) or {row["model"] for row in cases} != set(models):
        raise ValueError("expected all 32 synthetic cases from each verified model")
    rows = [[MODELS[row["model"]], row["passed"], row["total"]] for row in report["comprehension_summary"]]
    label_rows, findings = [], []
    for model in models:
        model_cases = [row for row in cases if row["model"] == model]
        failed = [row for row in model_cases if row["passed"].lower() != "true"]
        for wait_label in ("E", "A"):
            selected = [row for row in model_cases if row["wait_label"] == wait_label]
            label_rows.append([MODELS[model], wait_label, sum(row["passed"].lower() == "true" for row in selected), len(selected)])
        uniform = [row for row in model_cases if row["expected_semantic_action"] in ("WAIT", "PASS")]
        if uniform and all(row["passed"].lower() == "true" for row in uniform):
            findings.append(f"{MODELS[model]} passed all {len(uniform)} uniform-probability WAIT/PASS checks.")
        if (failed and all(row["expected_semantic_action"] in "ABCD" and row["wait_label"] == "A"
                           and row["chosen_action"] == "E" and row["native_semantic_action"] == "A" for row in failed)):
            findings.append(f"All {len(failed)} failures for {MODELS[model]} occurred on known-correct-answer cases under A-WAIT. "
                            "Each chose displayed E, which represented canonical candidate A after relabeling. "
                            "This localizes the observed errors to these relabeled cases without establishing a token-bias mechanism.")
    details = [[MODELS[row["model"]], row["synthetic_case"], row["round"], row["wait_label"], row["expected_semantic_action"],
                row["native_semantic_action"], row["passed"]] for row in cases]
    return ("<section id='comprehension'><h2>Explicit protocol comprehension checks</h2>"
            "<p>The synthetic prompts supply exact knowledge probabilities. A known correct answer should be taken now; "
            "a uniform choice with a guaranteed answer next round should WAIT; a uniform terminal choice should PASS. "
            "Both continuation labels are tested. These checks diagnose the declared protocol and do not estimate quizbowl ability.</p>"
            + table(["Model", "Correct semantic actions", "Cases"], rows, "All cases are reported, including failures; no failed case was discarded or used to retune the prompt")
            + table(["Model", "WAIT/PASS label", "Correct semantic actions", "Cases"], label_rows,
                    "Same synthetic premises, grouped by continuation label")
            + "<p>" + esc(" ".join(findings)) + "</p>"
            + f"<details><summary>All {32 * len(models)} scored synthetic cases from verified models</summary>"
            + table(["Model", "Case", "Round", "WAIT/PASS label", "Expected semantic action", "Chosen semantic action", "Passed"], details,
                    "A–D in semantic columns refer to canonical candidate identities, after undoing displayed labels")
            + "</details></section>")


def historical_section(report: dict[str, Any]) -> str:
    rows = [[MODELS[row["model"]], MENUS[row["condition"]], row["n_questions"], pct(row["historical_correct_fraction"]),
             pct(row["forced_game_correct_fraction"]), pct(row["same_wait_candidate_correct_fraction"])]
            for row in report["historical_summaries"] if row["split"] == "selection"]
    return ("<section><h2>Historical comparison: identity matched, protocol confounded</h2>"
            + table(["Model", "Menu", "Questions", "Old full response", "Forced game", "WAIT candidate"], rows,
                    "Original 100-question selection subset; identical question, menu, full-prefix ID, and source prompt hashes")
            + "<p>These historical differences jointly change instruction wording, generated-response versus next-token output protocol, "
              "and BF16 versus FP32 execution. Matching question identities rules out a different question sample as the explanation. "
              "It does not identify which protocol change caused the accuracy difference. The new matched experiment holds these other components fixed.</p></section>")


def execution_section(execution: dict[str, Any]) -> str:
    rows = []
    for key, label, unit in (("workflow_elapsed_seconds", "Workflow wall time", "seconds"),
                             ("estimated_compute_usd", "Estimated resource cost, stated allowances included", "USD"),
                             ("reserved_estimate_usd", "Reserved estimate", "USD")):
        value = execution.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"Missing or invalid execution quantity: {key}")
        rows.append([label, num(value, 3 if unit == "USD" else 1), unit])
    for model, value in execution.get("allocation_seconds_by_model", {}).items():
        if model not in MODELS or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError("invalid allocation time")
        rows.append([MODELS[model] + " observed worker function time", num(value, 1), "seconds"])
    return ("<section><h2>Execution, cost, and numerical validation</h2>"
            + table(["Quantity", "Value", "Unit"], rows, "New protocol-diagnosis run only; earlier completed runs are excluded")
            + "<p>Worker times add across parallel workers; workflow wall time does not. Observed function timing excludes some provider startup, "
              "orchestration, and idle allocation time. Cost is a resource estimate, not an invoice. "
            + ("The supplied summary marks invoice verification complete." if execution.get("invoice_verified") is True else "Invoice verification is UNVERIFIED.")
            + "</p>" + (f"<p>{esc(execution['notes'])}</p>" if execution.get("notes") else "") + "</section>")


def validate_execution_identity(execution: dict[str, Any], launch_commit: str,
                                workflow_url: str) -> str:
    """Bind timing/cost attribution to this run and require its analysis source."""
    if execution.get("launch_commit") != launch_commit or execution.get("workflow_url") != workflow_url:
        raise ValueError("execution summary launch/workflow identity differs from this report")
    analysis_commit = execution.get("analysis_source_commit")
    if not isinstance(analysis_commit, str) or not re.fullmatch(r"[0-9a-f]{40}", analysis_commit):
        raise ValueError("execution summary requires a complete analysis source Git SHA")
    return analysis_commit


def verified_result_interpretation(report: dict[str, Any]) -> str:
    """Present the declared selection contrasts without inferring a mechanism."""
    contrasts, label_changes, native_intervals, accuracy_differences = [], [], [], []
    for menu in MENUS:
        common = {"model": "qwen7b", "condition": menu, "split": "selection"}
        contrast = unique(report["round_prompt_contrasts"], contrast="plain_minus_forced", round=5, **common)
        effect = unique(report["action_label_contrasts"], **common)
        native = unique(report["policy_summaries"], arm="wait", policy="native_wait", **common)
        contrasts.append(f"{MENUS[menu].lower()} menu: {percentage_points(contrast['candidate_accuracy_difference'])} percentage points")
        accuracy_differences.append(contrast["candidate_accuracy_difference"]["mean"])
        label_changes.append(f"{MENUS[menu].lower()} menu: WAIT/PASS versus answer changed in {pct(effect['wait_decision_changed']['mean'])} "
                             f"of states and the semantic action changed in {pct(effect['semantic_action_changed']['mean'])}")
        native_intervals.append(native["bootstrap"]["mean_reward"]["ci95"])
    interval_sentence = ("The native WAIT reward interval includes zero with both menus, so this small selection subset does not establish a positive reward advantage over always passing."
                         if all(interval is not None and interval[0] <= 0 <= interval[1] for interval in native_intervals)
                         else "Native WAIT reward intervals are reported below; conclusions are limited to this already-inspected development subset.")
    result_sentence = ("In this matched 7B experiment, plain MCQA had higher full-question accuracy than the forced-answer game prompt with both menus."
                       if all(value > 0 for value in accuracy_differences)
                       else "The matched 7B experiment quantifies the full-question accuracy difference between plain MCQA and the forced-answer game prompt.")
    return ("<section id='findings'><h2>What changes the next decision</h2><p><strong>" + esc(result_sentence)
            + "</strong> Paired differences and descriptive 95% intervals were "
            + esc("; ".join(contrasts)) + ". The comparison holds weights, precision, answer content, and output boundary fixed, "
            "but changes the full instruction block; it does not isolate a particular reward sentence or mechanism.</p>"
            "<p>The A/E relabeling also changed decisions: " + esc("; ".join(label_changes))
            + ". These are state-level proportions over five paired rounds within 20 questions per menu. "
              "They include counterfactual states after an episode would have ended. " + esc(interval_sentence) + "</p></section>")


def build_report(decomposition_directory: Path, protocol_directory: Path, execution_path: Path,
                 launch_commit: str, workflow_url: str, *, partial: bool = False) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", launch_commit):
        raise ValueError("launch commit must be a complete Git SHA")
    if not re.fullmatch(r"https://github\.com/ankaggarwal94/qanta-buzzer/actions/runs/[0-9]+", workflow_url):
        raise ValueError("workflow URL is not the expected repository run")
    decomposition, decomposition_receipt = verified_report(decomposition_directory, "imcqa-wait-decomposition-v1")
    protocol_schema = "imcqa-protocol-partial-analysis-v1" if partial else "imcqa-protocol-analysis-v1"
    protocol, receipt = verified_report(protocol_directory, protocol_schema)
    execution = read_json(execution_path)
    analysis_commit = validate_execution_identity(execution, launch_commit, workflow_url)
    # The exact declaration is checked against input/output validation in the analyzer.
    if decomposition_receipt.get("n_questions") != 200 or not decomposition_receipt.get("additive_regret_identity_all_trajectories"):
        raise ValueError("incomplete CPU decomposition scope")
    models = ("qwen7b",) if partial else tuple(MODELS)
    expected_protocol_counts = {"n_questions": 40, "n_score_rows": 5232 * len(models), "n_new_score_rows": 4032 * len(models),
                                "n_reused_score_rows": 1200 * len(models), "n_synthetic_contexts_per_model": 32}
    if any(protocol.get(key) != value for key, value in expected_protocol_counts.items()):
        raise ValueError("unexpected protocol analysis scope/counts")
    if set(protocol.get("audits", {})) != set(models) or any(not row.get("passed") for row in protocol["audits"].values()):
        raise ValueError("incomplete or failed protocol numerical/provenance audit")
    failure_header = ""
    if partial:
        if (protocol.get("status") != "partial" or protocol.get("expected_models") != ["qwen3b", "qwen7b"]
                or protocol.get("validated_models") != ["qwen7b"] or protocol.get("failed_models") != ["qwen3b"]
                or protocol.get("expected_n_score_rows") != 10464):
            raise ValueError("partial model coverage is not explicitly declared")
        failure = protocol.get("failures", {}).get("qwen3b", {})
        if failure.get("n_production_rows_retained") != 0 or failure.get("excluded_from_scientific_analysis") is not True:
            raise ValueError("partial renderer only supports the documented zero-production 3B failure")
        failed_path = protocol_directory / "FAILED_MODELS.txt"
        if receipt["output_sha256"].get(failed_path.name) != sha256(failed_path):
            raise ValueError("failure evidence summary differs from analyzer receipt")
        failure_header = ("<div class='failure'><strong>Partial experiment: the 3B worker failed its initial numerical gate before producing any new scientific rows.</strong>"
                          "<p>The matched protocol results below cover verified Qwen 7B only. The 3B failure is excluded from new scientific estimates; "
                          "the earlier, independently validated two-model CPU decomposition is retained. The planned two-model pilot is incomplete.</p>"
                          + table(["Worker", "Planned new rows", "Retained new production rows", "Analyzed reused rows", "Analyzed contexts", "Status"],
                                  [["Qwen 3B", 4032, 0, 0, 0, "Failed; excluded"], ["Qwen 7B", 4032, 4032, 1200, 5232, "Validated"]],
                                  "The plan required 8,064 new plus 2,400 reused contexts; this report analyzes 4,032 new plus 1,200 reused contexts.")
                          + f"<p>Recorded failure: <code>{esc(failure.get('error', failure.get('receipt', {}).get('error', 'See retained failure evidence')))}</code></p>"
                          + "<details><summary>Retained 3B failure and diagnostic evidence</summary><pre>"
                          + esc(json.dumps(failure, indent=2, sort_keys=True)) + "</pre></details></div>")
    required = ("round_summaries", "round_prompt_contrasts", "policy_summaries", "rotation_contrasts", "rotation_consistency",
                "action_label_contrasts", "action_label_policy_summaries", "action_label_policy_contrasts", "comprehension_summary")
    if any(not protocol.get(key) for key in required):
        raise ValueError("missing protocol analysis tables")
    for key in required:
        if {row.get("model") for row in protocol[key]} != set(models):
            raise ValueError(f"scientific table includes unvalidated or missing models: {key}")
    headline = []
    for menu in MENUS:
        common = {"model": "qwen7b", "condition": menu, "split": "selection", "round": 5}
        plain = unique(protocol["round_summaries"], arm="plain", **common)["candidate_correct"]["mean"]
        forced = unique(protocol["round_summaries"], arm="forced", **common)["candidate_correct"]["mean"]
        wait = unique(protocol["round_summaries"], arm="wait", **common)["candidate_correct"]["mean"]
        headline.append(f"{MENUS[menu]} menu: {pct(plain)} plain, {pct(forced)} forced game, {pct(wait)} WAIT candidate.")
    sources = {"CPU decomposition report": decomposition_directory / "report.json", "CPU decomposition receipt": decomposition_directory / "analysis_receipt.json",
               "Matched protocol report": protocol_directory / "report.json", "Matched protocol receipt": protocol_directory / "analysis_receipt.json", "Execution summary": execution_path}
    sources_rows = [[label, path.name, sha256(path)] for label, path in sources.items()]
    limitations = list(dict.fromkeys(decomposition["limitations"] + protocol.get("limitations", protocol.get("interpretation_limits", []))))
    limitations += ["The 40 diagnostic questions are 20 per development split and were already inspected. There is no new confirmatory test set.",
                    f"The {5232 * len(models):,} analyzed contexts comprise {4800 * len(models):,} factorial contexts, {400 * len(models):,} action-label contexts, and {32 * len(models)} synthetic checks; they do not represent independent questions.",
                    "Deterministic constrained next-token scoring does not measure repeated full-response sampling frequencies."]
    if partial:
        limitations.append("The planned two-model matched pilot is incomplete: 3B failed an initial numerical gate. Its missing results are not treated as zero accuracy, zero reward, or evidence about model ability.")
    css = """
    :root{color-scheme:light}*{box-sizing:border-box}body{margin:0;background:#f4f6f9;color:#162333;font:16px/1.6 system-ui,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif}
    main{max-width:1280px;margin:auto;background:white;padding:40px 44px 60px}h1{font-size:32px;line-height:1.2;margin:0 0 10px}h2{font-size:24px;line-height:1.3;margin:0 0 16px}h3{font-size:19px}
    .kicker{font-size:12px;text-transform:uppercase;letter-spacing:.09em;color:#42617f;font-weight:700}.lead{font-size:19px;line-height:1.55;max-width:1050px}.callout{background:#edf5fb;border-left:4px solid #3776a9;padding:16px 20px}
    .failure{background:#fff2df;border-left:5px solid #b86900;padding:18px 20px;margin:20px 0}.failure strong{font-size:18px}
    section{margin-top:38px;padding-top:26px;border-top:1px solid #dce3eb}p{max-width:1100px}a{color:#155e93}code{background:#eef1f5;padding:2px 5px;border-radius:3px}
    .table-wrap{overflow-x:auto;margin:18px 0}table{border-collapse:collapse;width:100%;font-size:13px;line-height:1.4}caption{text-align:left;color:#4b6075;padding:0 0 10px;font-size:13px}th,td{padding:10px 12px;border-bottom:1px solid #dce3eb;text-align:left;vertical-align:top}th{background:#eaf0f7;font-weight:650}tbody tr:nth-child(even){background:#f8fafc}
    details{margin:20px 0;padding:12px 16px;background:#f7f9fc;border:1px solid #dae3ed;border-radius:4px}summary{cursor:pointer;font-weight:650}.meta{font-size:13px;color:#52677b}.hashes td:last-child{font:11px/1.5 ui-monospace,monospace;word-break:break-all}pre{white-space:pre-wrap;word-break:break-word;font-size:12px}li{margin:6px 0}
    @media(max-width:700px){main{padding:24px 16px}h1{font-size:27px}.lead{font-size:17px}}@media print{body{background:white}main{max-width:none;padding:0}section{break-inside:auto}details{display:block}details>*{display:block!important}.table-wrap{overflow:visible}th,td{padding:5px;font-size:10px}}
    """
    return ("<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
            "<title>IMCQA: " + ("PARTIAL " if partial else "") + "WAIT decomposition and matched protocol diagnosis</title><style>" + css + "</style></head><body><main>"
            "<p class='kicker'>Exploratory development analysis · 4 October 2026 UTC</p><h1>" + ("Partial protocol diagnosis and completed WAIT decomposition" if partial else "WAIT decomposition and matched protocol diagnosis") + "</h1>"
            + failure_header + "<p class='lead'>The saved outputs separate candidate availability from stopping decisions. The new matched experiment tests prompt instructions, "
            "candidate positions, and continuation labels while preserving the questions, weights, arithmetic, and output boundary.</p>"
            "<div class='callout'><strong>Selection-set full-question accuracy, Qwen 7B, averaged over four answer orders:</strong><ul>"
            + "".join(f"<li>{esc(line)}</li>" for line in headline) + "</ul><p>Each menu result contains 20 independent questions. "
              "The decomposition below covers the earlier, larger 100-question selection split. These denominators should not be combined.</p></div>"
            f"<p class='meta'>The CPU decomposition was completed first. The validated model evidence adds {4032 * len(models):,} scored contexts and reuses {1200 * len(models):,} exact prior contexts, "
            "with explicit identity and numerical checks. Both splits remain exploratory.</p>"
            + verified_result_interpretation(protocol)
            + decomposition_section(decomposition) + protocol_section(protocol, models) + sensitivity_section(protocol, models)
            + comprehension_section(protocol, protocol_directory, receipt, models) + historical_section(decomposition) + execution_section(execution)
            + "<details><summary>Numerical and provenance audit details</summary><pre>" + esc(json.dumps(protocol.get("audits", {}), indent=2, sort_keys=True)) + "</pre></details>"
            + "<section><h2>Recommended next experiment and limits</h2><p><strong>Stabilize answer elicitation and label handling before scaling to fresh questions.</strong> "
              "Use the plain MCQA condition as the answer-quality reference, preserve rotation-averaged reporting, and keep the A/E intervention as a required diagnostic. "
              "Then evaluate a separately specified stopping rule against calibrated and fixed-round baselines using its own candidate predictions. "
              "The current myopic baseline is not an optimal stopping policy and the observed label effects do not identify a unique mechanism.</p>"
            + ("<p>Repair the 3B numerical execution issue in a separate pre-production diagnostic, using the existing frozen contexts and unchanged tolerances. "
               "Determine whether the discrepancy comes from cached, batched, or single-context execution before attempting a replacement production run. "
               "Its missing scientific results cannot be inferred from the completed 7B experiment.</p>" if partial else "")
            + "<p>Only after these development decisions are frozen should a fresh question set support a confirmatory evaluation. "
              "The synthetic checks test explicit payoff premises; passing them does not establish clue recognition or an optimal policy.</p>"
            + "<ul>" + "".join(f"<li>{esc(item)}</li>" for item in limitations) + "</ul></section>"
            + "<section class='hashes'><h2>Evidence and reproducibility</h2><p>Launch source: <a href='https://github.com/ankaggarwal94/qanta-buzzer/commit/"
            + launch_commit + "'>" + launch_commit + "</a>. Analysis and report source: <a href='https://github.com/ankaggarwal94/qanta-buzzer/tree/"
            + analysis_commit + "'>" + analysis_commit + "</a>. <a href='" + esc(workflow_url) + "'>Workflow and retained raw evidence</a>. "
            "The report builder checks that both analysis reports and the synthetic-outcome CSV match their declared analysis receipts. "
            "The reproducibility package preserves these reports, all derived CSVs, configuration, and execution receipts.</p>"
            + table(["Evidence", "File", "SHA256"], sources_rows, "Input hashes for this report; code is stored in the linked repository")
            + "</section></main></body></html>\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decomposition", type=Path, required=True)
    parser.add_argument("--protocol-analysis", type=Path, required=True)
    parser.add_argument("--execution", type=Path, required=True)
    parser.add_argument("--launch-commit", required=True)
    parser.add_argument("--workflow-url", required=True)
    parser.add_argument("--out-html", type=Path, required=True)
    parser.add_argument("--partial", action="store_true", help="Require the distinct partial schema; report verified 7B science and explicit 3B failure")
    args = parser.parse_args()
    text = build_report(args.decomposition, args.protocol_analysis, args.execution, args.launch_commit, args.workflow_url, partial=args.partial)
    with args.out_html.open("x", encoding="utf-8") as stream:
        stream.write(text)
    print(json.dumps({"status": "partial" if args.partial else "complete", "path": str(args.out_html), "sha256": sha256(args.out_html), "bytes": args.out_html.stat().st_size}))


if __name__ == "__main__":
    main()
