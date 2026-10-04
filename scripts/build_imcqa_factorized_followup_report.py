#!/usr/bin/env python3
"""Render source-bound CPU factorization, binary stopping, and numerical evidence.

This is a new follow-up report. It does not replace prior failed-run evidence,
fit policies, pool four-order trajectories with rotation-zero binary episodes,
or treat successful numerical diagnostics as approved production recovery.
"""
from __future__ import annotations

import argparse
import csv
from decimal import Decimal
import hashlib
import html
import json
import math
from pathlib import Path
import re
import subprocess
from typing import Any

MENUS = {"independent_pool": "Independent", "same_category_pool": "Same category"}
MAPPINGS = {"submit_x": "X submits; Y defers", "submit_y": "Y submits; X defers", "mapping_average": "Equal interface average"}
REPO = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def number(value: float | None, places: int = 3) -> str:
    return "undefined" if value is None else f"{value:.{places}f}"


def percent(value: float | None) -> str:
    return "undefined" if value is None else f"{100*value:.1f}%"


def interval(point: float | None, bounds: list[float] | None, percentage: bool = False) -> str:
    fmt = percent if percentage else number
    return fmt(point) + (f" [{fmt(bounds[0])}, {fmt(bounds[1])}]" if bounds is not None else " [undefined interval]")


def estimate(value: dict[str, Any], percentage: bool = False) -> str:
    return interval(value["mean"], value.get("ci95"), percentage)


def percentage_points(value: dict[str, Any]) -> str:
    bounds = value.get("ci95")
    return interval(100*value["mean"], [100*x for x in bounds] if bounds is not None else None)


def policy_estimate(row: dict[str, Any], key: str = "mean_reward") -> str:
    return interval(row[key], row["bootstrap"][key].get("ci95"))


def unique(rows: list[dict[str, Any]], **filters: Any) -> dict[str, Any]:
    selected = [row for row in rows if all(row.get(key) == value for key, value in filters.items())]
    if len(selected) != 1:
        raise ValueError(f"Expected one row for {filters}; found {len(selected)}")
    return selected[0]


def table(headers: list[str], rows: list[list[Any]], caption: str) -> str:
    if any(len(row) != len(headers) for row in rows):
        raise ValueError("table row/header widths differ")
    return ("<div class='table-wrap'><table><caption>" + esc(caption) + "</caption><thead><tr>"
            + "".join("<th scope='col'>" + esc(label) + "</th>" for label in headers)
            + "</tr></thead><tbody>" + "".join("<tr>" + "".join("<td>" + esc(cell) + "</td>" for cell in row) + "</tr>" for row in rows)
            + "</tbody></table></div>")


def verified_files(directory: Path, receipt_name: str) -> dict[str, Any]:
    receipt = read_json(directory / receipt_name)
    outputs = receipt.get("output_sha256")
    if not isinstance(outputs, dict) or not outputs:
        raise ValueError(f"No output hash manifest: {directory}")
    for name, checksum in outputs.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or not re.fullmatch(r"[0-9a-f]{64}", checksum):
            raise ValueError("invalid output manifest entry")
        path = directory / relative
        if path.is_symlink() or not path.is_file() or digest(path) != checksum:
            raise ValueError(f"Receipt-bound artifact differs: {path}")
    return receipt


def source_hash(commit: str, relative: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", commit) or Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError("invalid source commit/path")
    try:
        raw = subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=REPO, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as error:
        raise ValueError(f"Source file unavailable at declared commit: {relative}") from error
    return hashlib.sha256(raw).hexdigest()


def verified_analysis(directory: Path, *, schema: str, scope: str, analyzer: str,
                      analysis_commit: str) -> tuple[dict[str, Any], dict[str, Any]]:
    receipt = verified_files(directory, "analysis_receipt.json")
    report = read_json(directory / "report.json")
    if (receipt.get("status") != "complete" or report.get("status") != "complete"
            or report.get("schema_version") != schema or report.get("evidence_scope") != scope
            or receipt["output_sha256"].get("report.json") != digest(directory / "report.json")
            or receipt.get("analyzer_sha256") != source_hash(analysis_commit, analyzer)):
        raise ValueError(f"Analysis schema, completion, or source identity differs: {directory}")
    return report, receipt


def read_csv(directory: Path, name: str) -> list[dict[str, str]]:
    with (directory / name).open(newline="") as stream:
        return list(csv.DictReader(stream))


def cpu_section(report: dict[str, Any], directory: Path) -> str:
    specifications = [
        ("Native WAIT candidates and stopping", "wait", "wait", "native"),
        ("Plain candidates; unchanged native WAIT stopping", "plain", "wait", "native"),
        ("WAIT candidates; selected external threshold", "wait", "wait", "calibration_selected_threshold"),
        ("Plain candidates; selected external threshold", "plain", "plain", "calibration_selected_threshold"),
        ("Plain cyclic ensemble; selected external threshold", "plain_ensemble", "plain_ensemble", "calibration_selected_threshold"),
    ]
    rows = []
    for label, answer, stop, method in specifications:
        cells = [unique(report["policy_summaries"], split="selection", condition=menu,
                        answer_source=answer, stop_source=stop, stop_method=method) for menu in MENUS]
        expected_episodes = 20 if answer == "plain_ensemble" else 80
        if any(cell["n_questions"] != 20 or cell["n_question_menu_records"] != expected_episodes for cell in cells):
            raise ValueError("CPU trajectory denominator differs")
        rows.append([label, *[policy_estimate(cell) for cell in cells], expected_episodes])
    fixed_rows, pipeline_rows, fit_rows = [], [], []
    for split in ("selection", "calibration"):
        for menu in MENUS:
            common = {"split": split, "condition": menu}
            fixed = unique(report["paired_contrasts"], contrast="plain_minus_wait_answers_same_wait_native_schedule", **common)
            pipeline = unique(report["paired_contrasts"], contrast="plain_own_calibration_selected_threshold_minus_native_wait", **common)
            if not fixed["identical_stopping_schedule"] or fixed["n_questions"] != 20:
                raise ValueError("answer-component contrast does not hold schedule fixed")
            fixed_rows.append([split, MENUS[menu], estimate(fixed["differences"]["mean_reward"]),
                               estimate(fixed["differences"]["coverage"], True), estimate(fixed["differences"]["mean_observed_round"])])
            pipeline_rows.append([split, MENUS[menu], estimate(pipeline["differences"]["mean_reward"])])
    for answer in ("plain", "wait", "plain_ensemble"):
        for menu in MENUS:
            fit = unique(report["fits"], answer_source=answer, condition=menu)
            if fit["fit_split"] != "calibration" or fit["n_fit_questions"] != 20 or not fit["selection_uses_only_calibration"]:
                raise ValueError("external policy fit split differs")
            fit_rows.append([answer, MENUS[menu], fit["n_fit_questions"], fit["n_fit_states"],
                             number(fit["selected_threshold"]), fit["selected_fixed_policy"]])
    states = read_csv(directory, "candidate_states.csv")
    calibration_rows = []
    for answer in ("plain", "plain_ensemble"):
        for menu in MENUS:
            for split in ("calibration", "selection"):
                metric = unique(report["calibration_metrics"], answer_source=answer, condition=menu, split=split)
                selected = [row for row in states if row["answer_source"] == answer and row["condition"] == menu and row["split"] == split]
                if len(selected) != metric["n_states"]:
                    raise ValueError("calibration state denominator differs from report")
                correct = sum(row["candidate_correct"] == "True" for row in selected)
                calibration_rows.append([answer, MENUS[menu], split, f"{correct}/{len(selected)}", metric["n_questions"],
                                         number(metric["raw"]["brier"]), number(metric["calibrated"]["brier"]),
                                         number(metric["raw"]["log_loss"]), number(metric["calibrated"]["log_loss"])])
    ensemble_comparison = []
    for menu in MENUS:
        ensemble = unique(report["policy_summaries"], split="selection", condition=menu, answer_source="plain_ensemble", stop_source="plain_ensemble", stop_method="calibration_selected_threshold")
        plain = unique(report["policy_summaries"], split="selection", condition=menu, answer_source="plain", stop_source="plain", stop_method="calibration_selected_threshold")
        ensemble_comparison.append(f"{MENUS[menu].lower()}: ensemble {number(ensemble['mean_reward'])}, plain {number(plain['mean_reward'])}")
    return ("<section id='cpu'><h2>Separate the answer from the stopping schedule</h2>"
            "<p>The CPU analysis reused 4,800 verified Qwen 7B prompt states and derived 400 cyclic-ensemble states. "
            "It ran zero new model forward passes. Each menu has 20 calibration and 20 selection questions. "
            "The non-ensemble results below average four complete candidate-order trajectories within each question.</p>"
            + table(["Pipeline", "Independent reward [95% CI]", "Same-category reward [95% CI]", "Selection trajectories / menu"], rows,
                    "Selection split; every cell has 20 independent questions. Ensemble cells have one derived trajectory per question.")
            + "<h3>The answer-elicitation component</h3><p>The next comparison holds every native WAIT stopping decision fixed. "
              "Only the candidate submitted at that stopping round changes, from the WAIT prompt's candidate to the plain-MCQA candidate. "
              "A change in reward here identifies an answer-component contribution within this replay, not a change in stopping time.</p>"
            + table(["Split", "Menu", "Plain − WAIT reward [95% CI]", "Commitment-rate difference [95% CI]", "Observed-round difference [95% CI]"], fixed_rows,
                    "Paired question bootstrap, retaining all four candidate orders. The last two columns must remain zero.")
            + "<h3>A separately fitted external stopping rule</h3><p>A monotone regularized logistic calibration maps the current candidate's score "
              "to estimated correctness. The threshold policy and fixed-round reference are chosen using calibration questions only. "
              "This is a complete answer-and-stopping pipeline comparison; its gain is not attributable solely to calibration.</p>"
            + table(["Split", "Menu", "Plain + external threshold − native WAIT reward [95% CI]"], pipeline_rows,
                    "Policies are held fixed in the question bootstrap; intervals exclude fitting uncertainty")
            + "<details><summary>Frozen fitted policy choices</summary>"
            + table(["Answer source", "Menu", "Fit questions", "Fit states", "Selected threshold", "Selected fixed policy"], fit_rows,
                    "All fits and policy selection use the calibration split") + "</details>"
            + "<h3>The ensemble and calibration gap</h3><p>Averaging four canonicalized candidate distributions changes the answerer and its fitted calibration. "
              "It must be assessed as its own pipeline. The table retains adverse results as well as gains. "
              "The selection split was previously inspected; the calibration-to-selection difference is descriptive and does not identify its cause.</p>"
            + "<p>Selection reward point estimates for the chosen threshold pipelines were " + esc("; ".join(ensemble_comparison))
            + ". The ensemble therefore did not improve either point estimate. The Brier table shows whether its fitted probabilities also transferred poorly.</p>"
            + table(["Answer source", "Menu", "Split", "Correct states", "Questions", "Raw Brier", "Calibrated Brier", "Raw log loss", "Calibrated log loss"], calibration_rows,
                    "Lower Brier and log loss are better. Plain uses four orders × five rounds; ensemble uses one derived distribution × five rounds.")
            + "<p>Calibration can worsen on another subset, and pooled-round calibration need not hold at every round or among policy-selected commitments. "
              "The myopic positive-expected-value rule compares an answer with PASS; it does not estimate the value of another clue or solve optimal stopping.</p></section>")


def binary_section(report: dict[str, Any], directory: Path) -> str:
    reward_rows, mapping_rows, contrast_rows, baseline_rows = [], [], [], []
    selection_intervals = []
    for split in ("selection", "calibration"):
        for menu in MENUS:
            common = {"split": split, "condition": menu}
            references = [unique(report["policy_summaries"], mapping="mapping_average", policy=policy, **common)
                          for policy in ("binary_native", "old_wait_timing_plain_proposal", "plain_fixed_round_1", "plain_hindsight_or_pass")]
            baseline_rows.append([split, MENUS[menu], *[policy_estimate(row) for row in references]])
            effect = unique(report["mapping_summaries"], **common)
            mapping_rows.append([split, MENUS[menu], effect["n_questions"], effect["n_states"],
                                 estimate(effect["semantic_action_changed"], True), estimate(effect["semantic_total_variation"])])
            for mapping in MAPPINGS:
                cell = unique(report["policy_summaries"], policy="binary_native", mapping=mapping, **common)
                reference = unique(report["policy_summaries"], policy="old_wait_timing_plain_proposal", mapping=mapping, **common)
                if cell["n_questions"] != 20 or cell["n_question_menu_records"] != (40 if mapping == "mapping_average" else 20):
                    raise ValueError("binary trajectory denominator differs")
                reward_rows.append([split, MENUS[menu], MAPPINGS[mapping], policy_estimate(cell),
                                    f"{cell['n_committed']}/{cell['n_question_menu_records']}", percent(cell["risk"]),
                                    f"{cell['stopping_round_counts'].get('1', 0)}/{cell['n_question_menu_records']}",
                                    number(reference["mean_reward"])])
                contrast = unique(report["paired_comparisons"], contrast="binary_native_minus_old_wait_timing_plain_proposal", mapping=mapping, **common)
                contrast_rows.append([split, MENUS[menu], MAPPINGS[mapping], estimate(contrast["differences"]["mean_reward"])])
                if split == "selection" and mapping == "mapping_average":
                    selection_intervals.append(contrast["differences"]["mean_reward"]["ci95"])
    synthetic = read_csv(directory, "comprehension_records.csv")
    if len(synthetic) != 32 or len({row["score_id"] for row in synthetic}) != 32:
        raise ValueError("binary synthetic case coverage differs")
    case_rows = [[row["case_family"], row["proposal_id"], row["round"], MAPPINGS[row["mapping"]],
                  row["expected_semantic_action"], row["game_action"], row["passed"]] for row in synthetic]
    case_summary = [[MAPPINGS[row["mapping"]], row["passed"], row["n_contexts"]] for row in report["comprehension_summary"]]
    for summary in report["comprehension_summary"]:
        selected = [row for row in synthetic if row["mapping"] == summary["mapping"]]
        if summary["n_contexts"] != len(selected) or summary["passed"] != sum(row["passed"] == "True" for row in selected):
            raise ValueError("synthetic summary differs from raw analyzed cases")
    uncertain_future = [row for row in synthetic if row["case_family"] == "uniform_future"]
    future_passed = sum(row["passed"] == "True" for row in uncertain_future)
    relative_result = ("The binary controller does not establish a reward improvement over the same-proposal hybrid: both selection comparison intervals include zero."
                       if all(bounds is not None and bounds[0] <= 0 <= bounds[1] for bounds in selection_intervals)
                       else "The selection comparison intervals quantify the reward difference from the same-proposal hybrid; these are exploratory development results.")
    tie_rows = [[row["block"], MAPPINGS[row["mapping"]], row["exact_ties"], row["n_contexts"]] for row in report["tie_summaries"]]
    return ("<section id='binary'><h2>Can a binary interface stop reliably?</h2>"
            "<p>The new controller sees the same current question, all four options, and a fixed answer proposal from the saved plain-MCQA rotation-zero prediction. "
            "It scores only X and Y: one means SUBMIT the proposal, the other means WAIT (rounds 1–4) or PASS (round 5). "
            "The two interfaces reverse those labels while retaining semantic row order. The controller cannot change the candidate; a later round may supply a different proposal. "
            "No numerical confidence or real correctness label enters the prompt.</p>"
            "<p><strong>These results use one candidate order.</strong> Each mapping has 20 question trajectories per menu and split. "
            "The equal mapping average is a summary over interfaces, not a single deployed policy. "
            "Do not compare its trajectory count with the four-order CPU table as though they were independent samples.</p>"
            + "<p><strong>" + esc(relative_result) + "</strong> First-round commitments are shown explicitly because high reward can come from submitting immediately rather than selectively waiting.</p>"
            + table(["Split", "Menu", "Interface", "Binary reward [95% CI]", "Commitments / trajectories", "Errors / commitments", "Stopped in round 1 / trajectories", "Old WAIT timing + same plain proposal reward"], reward_rows,
                    "Every row has 20 independent questions; mapping-average rows contain two paired interfaces per question")
            + table(["Split", "Menu", "Binary interface-average reward", "Old WAIT timing + same proposal", "Always submit in round 1", "Same-proposal hindsight"], baseline_rows,
                    "All entries include descriptive 95% intervals and use the same rotation-zero plain proposals. PASS earns zero; hindsight is unavailable to a deployable policy.")
            + table(["Split", "Menu", "Interface", "Binary − old WAIT timing, same-proposal reward [95% CI]"], contrast_rows,
                    "Only stopping schedules differ in this replay; the model prompts and action spaces also differ between the protocols")
            + "<p>This comparison tests a complete stopping-interface change, not a pure token-label mechanism. "
              "Candidate invariance is guaranteed by fixing the proposal; it is not evidence that the model learned label-invariant answer selection.</p>"
            + "<h3>Label invariance of the stopping decision</h3>"
            + table(["Split", "Menu", "Questions", "Paired states", "SUBMIT/DEFER changed [95% CI]", "Semantic probability TV [95% CI]"], mapping_rows,
                    "All five counterfactual rounds are included, even after a native episode would have ended; 100 matched states per split/menu")
            + "<p>Probabilities are normalized over the two legal next tokens. They are not calibrated correctness probabilities or frequencies of complete generated responses. "
              "Full-vocabulary legal-action mass and unconstrained top-token evidence are retained in the raw scores and validation audit.</p>"
            + "<details><summary>Exact ties</summary>" + table(["Block", "Interface", "Exact ties", "Contexts"], tie_rows,
                    "Exact X/Y logit ties always choose semantic DEFER, under either label mapping") + "</details>"
            + "<h3>Explicitly solvable protocol checks</h3>"
            + table(["Interface", "Expected actions chosen", "Cases"], case_summary,
                    "All 32 synthetic checks are retained; none serves as an outcome-dependent collection gate")
            + "<p>The checks state exact correctness probabilities and, when needed, guarantee a correct next-round proposal. "
              "A known correct proposal should be submitted. A known incorrect terminal proposal should be passed. "
              "At round 1, a wrong or uniform proposal with a guaranteed correct next proposal should WAIT; a uniform terminal proposal should PASS. "
              "Passing these small checks establishes only the tested instructions and payoffs.</p>"
            + f"<p><strong>Uniform-now / certain-next checks passed: {future_passed}/{len(uncertain_future)}.</strong> "
              "For these round-1 cases, SUBMIT has expected reward 0.25 × 1 + 0.75 × (−1) = −0.5, whereas WAIT guarantees +0.8 from a correct round-2 proposal. "
              "This tests the value of the explicitly promised next proposal. The synthetic battery differs from the earlier five-action battery; their overall pass rates are not a matched comparison.</p>"
            + "<details><summary>All synthetic outcomes</summary>"
            + table(["Case family", "Proposal", "Round", "Interface", "Expected", "Chosen", "Passed"], case_rows,
                    "Proposal letters identify the fixed candidate, not the X/Y action tokens") + "</details></section>")


def numerical_section(directory: Path, launch_commit: str) -> tuple[str, dict[str, Any]]:
    receipt = verified_files(directory, "receipt.json")
    if receipt.get("model_tag") != "qwen3b" or receipt.get("production_rows") != 0 or receipt.get("automatic_retries") != 0:
        raise ValueError("numerical study scope differs")
    metadata = read_json(directory / "metadata.json")
    for name, checksum in metadata["source_files_sha256"].items():
        if source_hash(launch_commit, name) != checksum:
            raise ValueError("numerical source differs from launch commit: " + name)
    rows = []
    for cached in (False, True):
        for size in (2, 4, 8):
            name = ("cached" if cached else "uncached") + f"_{size}"
            path = directory / (name + ".json")
            if not path.exists():
                rows.append([name, "not completed", "—", "—", "—", "—", "—"])
                continue
            if path.name not in receipt["output_sha256"]:
                raise ValueError("unbound numerical mode file")
            item = read_json(path)
            comparison = item["comparison"]
            ratio = max(row["max_tolerance_ratio"] for row in comparison["rows"])
            rows.append([name, item["diagnostic_compatible"], number(comparison["max_logit_difference"], 7), number(ratio),
                         number(comparison["max_action_probability_difference"], 7),
                         f"{comparison['action_argmax_changes']} / {comparison['candidate_argmax_changes']}", number(item["elapsed_forward_seconds"], 2)])
    reproduction_rows = []
    path = directory / "original_batch32_reproduction.json"
    if path.exists():
        for mode, item in read_json(path).items():
            for reference in ("versus_fresh_single", "versus_prior_same_path"):
                comp = item[reference]
                reproduction_rows.append([mode, reference, comp["passed"], number(comp["max_logit_difference"], 7),
                                          f"{comp['action_argmax_changes']} / {comp['candidate_argmax_changes']}"])
    reference_status = receipt.get("reference_valid")
    result = ("<section id='numerics'><h2>3B numerical execution remains a separate evidence track</h2>"
              "<p>The earlier 3B protocol worker failed its unchanged numerical gate before producing new scientific rows. "
              "That run remains failed and excluded. This follow-up studies execution shapes on 20 frozen diagnostic contexts, "
              "including original diagnostics and prior-run overlap contexts; it produces no quizbowl production results.</p>"
              + f"<p>Study status: <strong>{esc(receipt['status'])}</strong>. Single-reference validation: <strong>{esc(reference_status)}</strong>. "
                f"Recorded model forward calls: {esc(receipt.get('model_forward_calls'))}; logical context evaluations: {esc(receipt.get('logical_evaluations'))}. "
                f"Compatible modes: {esc(', '.join(receipt.get('diagnostic_compatible_modes', [])) or 'none recorded')}.</p>"
              + table(["Execution mode", "Diagnostic compatible", "Max raw-logit delta", "Max tolerance ratio", "Max action-probability delta", "Action / candidate flips", "Forward seconds"], rows,
                      "Unchanged gate: absolute tolerance 0.001 + relative tolerance 0.00001; action/candidate probabilities within 0.001; no argmax changes")
              + "<p>A tolerance ratio above 1 fails the raw-logit gate even if the selected action stays unchanged. "
                "The timing covers one diagnostic pass; it is not a production-throughput guarantee. "
                "Passing this study does not approve production: actual production-batch context, exact replay, nontrivial permutation, prior overlap, "
                "and throughput checks remain necessary for a separately frozen recovery run.</p>"
              + (table(["Batch-32 mode", "Reference", "Passed", "Max raw-logit delta", "Action / candidate flips"], reproduction_rows,
                       "Reproduction of the original failed execution shape; earlier artifacts are preserved") if reproduction_rows else "")
              + "<details><summary>Numerical receipt and provenance</summary><pre>" + esc(json.dumps(receipt, indent=2, sort_keys=True)) + "</pre></details></section>")
    return result, receipt


def execution_section(execution: dict[str, Any]) -> str:
    def quantity(key: str) -> float:
        raw = execution.get(key)
        if isinstance(raw, bool):
            raise ValueError("invalid boolean execution quantity: " + key)
        try:
            value = float(Decimal(str(raw)))
        except Exception as error:
            raise ValueError("missing/invalid execution quantity: " + key) from error
        if not math.isfinite(value) or value < 0:
            raise ValueError("missing/invalid execution quantity: " + key)
        return value
    rows = []
    fields = [("workflow_elapsed_seconds", "Initial binary + numerical workflow wall time", "seconds")]
    if "recovery_workflow_elapsed_seconds" in execution:
        fields.extend([( "recovery_workflow_elapsed_seconds", "Separate recovery workflow wall time", "seconds"),
                       ("combined_workflow_elapsed_seconds", "Sum of the two sequential workflow durations", "seconds")])
        if not math.isclose(quantity("combined_workflow_elapsed_seconds"), quantity("workflow_elapsed_seconds") + quantity("recovery_workflow_elapsed_seconds"), abs_tol=1e-6):
            raise ValueError("combined workflow duration does not equal the two sequential workflows")
    if "followup_launch_to_final_completion_seconds" in execution:
        fields.append(("followup_launch_to_final_completion_seconds", "Initial launch to final completion, including intervening review/analysis", "seconds"))
        minimum = quantity("combined_workflow_elapsed_seconds") if "combined_workflow_elapsed_seconds" in execution else quantity("workflow_elapsed_seconds")
        if quantity("followup_launch_to_final_completion_seconds") + 1e-6 < minimum:
            raise ValueError("follow-up span is shorter than its contained workflow durations")
    fields.extend([( "estimated_compute_usd", "Estimated resource cost including stated allowances", "USD"),
                   ("reserved_estimate_usd", "Reserved resource estimate", "USD")])
    for key, label, unit in fields:
        rows.append([label, number(quantity(key), 3 if unit == "USD" else 1), unit])
    for label, seconds in execution.get("allocation_seconds_by_model", {}).items():
        if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0:
            raise ValueError("invalid allocation time")
        rows.append([label + " worker window", number(seconds, 1), "seconds"])
    return ("<section id='execution'><h2>Runtime and cost</h2>"
            + table(["Quantity", "Value", "Unit"], rows, "Scope: this follow-up, including separately itemized recovery if present in the ledger; predecessor runs excluded")
            + "<p>Parallel worker seconds add; their workflow wall time does not. The initial workflow's duration is not the full follow-up duration when a separate recovery is included. "
              "Resource cost is an estimate, not an invoice. "
            + ("The execution ledger marks invoice verification complete." if execution.get("invoice_verified") is True else "Invoice amount remains UNVERIFIED.")
            + "</p>" + ("<ul>" + "".join("<li>" + esc(note) + "</li>" for note in (execution["notes"] if isinstance(execution["notes"], list) else [execution["notes"]])) + "</ul>" if execution.get("notes") else "")
            + "<details><summary>Complete execution ledger, including per-study costs</summary><pre>" + esc(json.dumps(execution, indent=2, sort_keys=True)) + "</pre></details></section>")


def optional_recovery_section(directory: Path | None, *, execution: dict[str, Any], analysis_commit: str,
                              numerical_receipt_sha256: str, cpu: dict[str, Any]) -> str:
    """Bind a separately validated 3B recovery without changing predecessor status."""
    if directory is None:
        return ""
    recovery_commit = execution.get("recovery_launch_commit", "")
    recovery_url = execution.get("recovery_workflow_url", "")
    recovery_id = execution.get("recovery_workflow_id")
    if (not re.fullmatch(r"[0-9a-f]{40}", recovery_commit)
            or not re.fullmatch(r"https://github\.com/ankaggarwal94/qanta-buzzer/actions/runs/[0-9]+", recovery_url)
            or str(recovery_id) != recovery_url.rsplit("/", 1)[-1]):
        raise ValueError("recovery launch and workflow identities must be explicit in the ledger")
    report, receipt = verified_analysis(directory, schema="imcqa-protocol-recovered-analysis-v1",
        scope="exploratory_already_inspected_development_questions", analyzer="scripts/analyze_imcqa_protocol_recovery.py",
        analysis_commit=analysis_commit)
    expected = {"completion_mode": "recovered_3b_plus_unchanged_7b", "n_questions": 40,
        "n_synthetic_contexts_per_model": 32, "expected_models": ["qwen3b", "qwen7b"],
        "validated_models": ["qwen3b", "qwen7b"], "expected_n_score_rows": 10464, "n_score_rows": 10464,
        "n_new_score_rows": 8064, "n_reused_score_rows": 2400, "original_run_status": "partial",
        "recovered_model": "qwen3b", "unchanged_model": "qwen7b"}
    if any(report.get(key) != value for key, value in expected.items()):
        raise ValueError("recovered analysis scope or model coverage differs")
    if (receipt.get("completion_mode") != expected["completion_mode"] or receipt.get("n_questions") != 40
            or receipt.get("n_score_rows") != 10464 or receipt.get("recovery_source_commit") != recovery_commit
            or receipt.get("numerical_diagnostic_receipt_sha256") != numerical_receipt_sha256
            or receipt.get("unchanged_original_analyzer_sha256") != source_hash(analysis_commit, "scripts/analyze_imcqa_protocol_pilot.py")):
        raise ValueError("recovery analysis receipt, predecessor analyzer, or numerical source differs")
    audits = report.get("audits", {})
    if set(audits) != {"qwen3b", "qwen7b"} or any(not audit.get("passed") for audit in audits.values()):
        raise ValueError("both recovered analysis model audits must pass")
    recovered = audits["qwen3b"]
    contract = recovered.get("recovery_contract", {})
    required_contract = {"execution_protocol": "imcqa_3b_protocol_recovery_cached2_v1", "cached": True,
        "batch_size": 2, "selected_mode": "cached_2", "source_commit": recovery_commit,
        "scientific_inputs_changed": False, "numerical_tolerances_changed": False,
        "diagnostic_receipt_sha256": numerical_receipt_sha256}
    if (any(contract.get(key) != value for key, value in required_contract.items())
            or recovered.get("n_new_rows") != 4032 or recovered.get("n_reused_rows") != 1200 or recovered.get("n_total_rows") != 5232):
        raise ValueError("recovered worker contract or production counts differ")
    for name, checksum in recovered["verified_source_files_sha256"].items():
        if source_hash(recovery_commit, name) != checksum:
            raise ValueError("recovery worker source differs from separate launch commit: " + name)
    if (audits["qwen7b"]["scores_sha256"] != cpu["audits"]["new"]["scores_sha256"]
            or audits["qwen7b"]["metadata_sha256"] != cpu["audits"]["new"]["metadata_sha256"]):
        raise ValueError("recovered analysis changed the original validated 7B artifacts")
    failure = report.get("predecessor_failure", {})
    predecessor_sha = receipt.get("predecessor_receipt_sha256")
    if (failure.get("status") != "failed" or failure.get("n_production_rows_retained") != 0
            or failure.get("excluded_from_scientific_analysis") is not True
            or failure.get("evidence_sha256", {}).get("receipt.json") != predecessor_sha
            or contract.get("predecessor_receipt_sha256") != predecessor_sha
            or not report.get("numerical_mode_selection", {}).get("passed")):
        raise ValueError("failed predecessor exclusion or mode-selection evidence differs")
    accuracy_rows, stopping_rows = [], []
    for model in ("qwen3b", "qwen7b"):
        for menu in MENUS:
            common = {"model": model, "condition": menu, "split": "selection"}
            final = [unique(report["round_summaries"], arm=arm, round=5, **common) for arm in ("plain", "forced", "wait")]
            contrast = unique(report["round_prompt_contrasts"], contrast="plain_minus_forced", round=5, **common)
            if any(row["n_questions"] != 20 for row in final):
                raise ValueError("recovered accuracy denominator differs")
            accuracy_rows.append([model, MENUS[menu], *[estimate(row["candidate_correct"], True) for row in final],
                                  percentage_points(contrast["candidate_accuracy_difference"])])
            native = unique(report["policy_summaries"], arm="wait", policy="native_wait", **common)
            label_effect = unique(report["action_label_contrasts"], **common)
            stopping_rows.append([model, MENUS[menu], policy_estimate(native), percent(native["coverage"]), percent(native["risk"]),
                                  estimate(label_effect["semantic_action_changed"], True)])
    cases = [[row["model"], row["passed"], row["total"]] for row in report["comprehension_summary"]]
    return ("<section id='recovery'><h2>The separate 3B recovery completes the earlier matched-prompt dataset</h2>"
            "<p><strong>The original 3B run remains failed.</strong> A new allocation used the same frozen scientific jobs, model files, prompts, FP32 arithmetic, "
            "and tolerances, changing the cached production batch size from 32 to 2. The independent recovery analyzer rechecked actual production pairs, "
            "single-context agreement, replay, row permutation, prior-run overlap, completed production, and active-only trajectories. "
            "Only after those checks passed were the new 3B scores combined with the unchanged earlier 7B evidence.</p>"
            + table(["Model evidence", "New rows in this recovery", "Total analyzed matched-protocol contexts", "Status"],
                    [["Separate 3B recovery", 4032, 5232, "4032 recovered + 1200 exact original WAIT rows"],
                     ["Earlier 7B protocol pilot", 0, 5232, "Unchanged; all 5232 contexts already existed"]],
                    "The combined earlier-protocol dataset has 10,464 contexts on forty questions. It is separate from the 832-context binary-controller experiment.")
            + table(["Model", "Menu", "Plain final accuracy [95% CI]", "Forced-game final accuracy [95% CI]", "WAIT-candidate final accuracy [95% CI]", "Plain − forced, percentage points [95% CI]"],
                    accuracy_rows, "Selection split: twenty questions per menu/model; each question averages four cyclic candidate orders")
            + table(["Model", "Menu", "Native WAIT reward [95% CI]", "Commitment rate", "Errors / commitments", "A/E semantic-action change [95% CI]"],
                    stopping_rows, "Native reward averages four orders; the separate A/E action-label comparison uses rotation zero and 100 paired states per cell")
            + table(["Model", "Expected actions chosen", "Synthetic cases"], cases,
                    "The earlier five-action comprehension battery; its overall pass rate is not directly comparable with the different binary battery")
            + "<p>This restores the missing model's measurements for the original development design. It does not turn these two related model sizes "
              "and previously inspected questions into a confirmatory ranking study. The 3B model was not run through the new binary controller.</p>"
            + f"<p><a href='https://github.com/ankaggarwal94/qanta-buzzer/commit/{recovery_commit}'>Separate recovery source {recovery_commit}</a>. "
              f"<a href='{esc(recovery_url)}'>Separate recovery workflow</a>.</p>"
            + "<details><summary>Recovery contract and retained failed predecessor</summary><pre>"
            + esc(json.dumps({"recovery_contract": contract, "predecessor_failure": failure,
                              "receipt": receipt, "limitations": report["limitations"]}, indent=2, sort_keys=True))
            + "</pre></details></section>")


def build_report(cpu_dir: Path, binary_dir: Path, numerics_dir: Path, execution_path: Path,
                 launch_commit: str, analysis_commit: str, workflow_url: str, recovery_dir: Path | None = None) -> str:
    for value in (launch_commit, analysis_commit):
        if not re.fullmatch(r"[0-9a-f]{40}", value):
            raise ValueError("complete launch and analysis commit SHAs required")
    if not re.fullmatch(r"https://github\.com/ankaggarwal94/qanta-buzzer/actions/runs/[0-9]+", workflow_url):
        raise ValueError("unexpected workflow URL")
    execution = read_json(execution_path)
    if (execution.get("launch_commit") != launch_commit or execution.get("analysis_source_commit") != analysis_commit
            or execution.get("workflow_url") != workflow_url):
        raise ValueError("execution ledger source or run identity differs")
    if digest(Path(__file__)) != source_hash(analysis_commit, "scripts/build_imcqa_factorized_followup_report.py"):
        raise ValueError("report builder differs from declared analysis source")
    cpu, cpu_receipt = verified_analysis(cpu_dir, schema="imcqa-factorized-cpu-analysis-v1", scope="exploratory_posthoc_development_analysis",
                                         analyzer="scripts/analyze_imcqa_factorized_cpu.py", analysis_commit=analysis_commit)
    binary, binary_receipt = verified_analysis(binary_dir, schema="imcqa-binary-analysis-v1", scope="exploratory_already_inspected_development_questions",
                                               analyzer="scripts/analyze_imcqa_binary_pilot.py", analysis_commit=analysis_commit)
    if (cpu.get("n_questions") != 40 or cpu.get("n_calibration_questions") != 20 or cpu.get("n_selection_questions") != 20
            or cpu.get("n_model_forward_passes") != 0 or cpu.get("n_verified_factorial_states") != 4800
            or cpu.get("n_derived_ensemble_states") != 400 or cpu.get("model") != "qwen7b"
            or not cpu.get("audits", {}).get("new", {}).get("passed")):
        raise ValueError("CPU scope or verified upstream evidence differs")
    if (binary.get("n_questions") != 40 or binary.get("n_score_rows") != 832 or binary.get("n_real_contexts") != 800
            or binary.get("n_synthetic_contexts") != 32 or binary.get("model") != "qwen7b" or not binary.get("audit", {}).get("passed")):
        raise ValueError("binary scope or numerical audit differs")
    for name, checksum in binary["audit"]["verified_source_files_sha256"].items():
        if source_hash(launch_commit, name) != checksum:
            raise ValueError("binary worker source differs from launch commit: " + name)
    numerical_html, numerical_receipt = numerical_section(numerics_dir, launch_commit)
    passed = sum(row["passed"] for row in binary["comprehension_summary"])
    cpu_values = [unique(cpu["policy_summaries"], split="selection", condition=menu, answer_source="plain", stop_source="plain", stop_method="calibration_selected_threshold")["mean_reward"] for menu in MENUS]
    binary_values = [unique(binary["mapping_summaries"], split="selection", condition=menu)["semantic_action_changed"]["mean"] for menu in MENUS]
    sources = [["CPU report", "report.json", digest(cpu_dir / "report.json")], ["CPU analysis receipt", "analysis_receipt.json", digest(cpu_dir / "analysis_receipt.json")],
               ["Binary report", "report.json", digest(binary_dir / "report.json")], ["Binary analysis receipt", "analysis_receipt.json", digest(binary_dir / "analysis_receipt.json")],
               ["3B numerical receipt", "receipt.json", digest(numerics_dir / "receipt.json")], ["Execution ledger", execution_path.name, digest(execution_path)],
               ["Report builder", "scripts/build_imcqa_factorized_followup_report.py", digest(Path(__file__))]]
    if recovery_dir is not None:
        sources.extend([["Recovery report", "report.json", digest(recovery_dir / "report.json")], ["Recovery receipt", "analysis_receipt.json", digest(recovery_dir / "analysis_receipt.json")]])
    limitations = list(dict.fromkeys(cpu["limitations"] + binary["limitations"]))
    style = """body{margin:0;background:#f3f5f7;color:#182332;font:16px/1.6 system-ui,sans-serif}main{max-width:1180px;margin:auto;padding:40px 28px 70px}header,section{background:white;border:1px solid #dce3ea;border-radius:14px;padding:26px 30px;margin-bottom:24px}h1{font-size:34px;line-height:1.2;max-width:880px}h2{font-size:24px;line-height:1.3;margin-top:0}h3{font-size:19px;margin-top:30px}p{max-width:1000px}a{color:#1557a0}nav{display:flex;flex-wrap:wrap;gap:18px;margin:20px 0}.eyebrow{color:#536575;font-size:13px;letter-spacing:.06em;text-transform:uppercase}.notice{background:#edf4ff;border-left:4px solid #3572ad;padding:15px 20px;border-radius:4px}.table-wrap{overflow:auto;margin:22px 0}table{border-collapse:collapse;width:100%;font-size:14px}caption{text-align:left;color:#4f6070;margin-bottom:10px;font-size:13px}th,td{padding:11px 12px;border-bottom:1px solid #dfe5eb;text-align:left;vertical-align:top}th{background:#edf2f6;font-weight:650}tbody tr:nth-child(even){background:#f8fafc}details{border-top:1px solid #dfe5eb;padding:15px 0;margin-top:15px}summary{cursor:pointer;font-weight:600}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f6f8;padding:15px;font:12px/1.5 ui-monospace,monospace}.hashes td:last-child{font:11px/1.6 ui-monospace,monospace;overflow-wrap:anywhere}.muted{color:#536575;font-size:14px}@media(max-width:700px){main{padding:15px}header,section{padding:20px 16px}h1{font-size:27px}}@media print{body{background:white}main{padding:0}section,header{break-inside:avoid;border:0}details{display:block}nav{display:none}}"""
    return ("<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
            "<title>IMCQA follow-up: answer elicitation, binary stopping, and numerical repair</title><style>" + style + "</style></head><body><main>"
            "<header><div class='eyebrow'>Exploratory development follow-up · 4 October 2026 UTC</div>"
            "<h1>Separate answer quality from the stopping interface</h1>"
            "<p class='notice'><strong>Forty questions, not thousands of independent trials.</strong> The analysis uses 20 calibration and 20 selection questions. "
            "Both subsets were already inspected. Candidate orders, prompts, rounds, and action labels are paired measurements of these questions.</p>"
            + f"<p>The plain-answerer plus external-threshold pipeline earned selection rewards of <strong>{number(cpu_values[0])}</strong> and "
              f"<strong>{number(cpu_values[1])}</strong> with independent and same-category menus, averaging four candidate orders. "
              f"The new rotation-zero binary controller changed SUBMIT/DEFER decisions across its label mappings in <strong>{percent(binary_values[0])}</strong> and "
              f"<strong>{percent(binary_values[1])}</strong> of matched selection states. It passed <strong>{passed}/32</strong> explicit synthetic checks; "
              "the uniform-now / certain-next cases are examined individually below.</p>"
            "<p>These are separate comparisons with different interfaces and trajectory counts. The report retains uncertainty, adverse calibration results, "
            "and the earlier 3B failure instead of treating a new numerical study as replacement production evidence.</p>"
            "<nav><a href='#cpu'>CPU factorization</a><a href='#binary'>Binary controller</a><a href='#numerics'>3B numerical study</a>"
            "<a href='#execution'>Time and cost</a><a href='#evidence'>Evidence</a></nav></header>"
            + cpu_section(cpu, cpu_dir) + binary_section(binary, binary_dir) + numerical_html + optional_recovery_section(recovery_dir, execution=execution, analysis_commit=analysis_commit,
                numerical_receipt_sha256=digest(numerics_dir / "receipt.json"), cpu=cpu)
            + execution_section(execution)
            + "<section><h2>Interpretation and next decision</h2><p>Preserve the plain answerer as a candidate-quality reference, "
              "judge stopping rules against references using exactly the same candidates, and retain explicit action-label controls. "
              "Freeze any selected interface before testing fresh, deduplicated questions with reviewed clue boundaries. "
              "Additional inference on these forty questions is protocol development, not confirmatory evidence or a broad model ranking.</p>"
            "<p>At the final round, correct submission earns +0.2, an error earns −1, and PASS earns 0. "
            "Answering therefore has positive expected value only for calibrated correctness above 1/1.2 = 83.33%. "
            "At earlier rounds, an optimal decision additionally needs the value of waiting.</p><ul>"
            + "".join("<li>" + esc(item) + "</li>" for item in limitations) + "</ul></section>"
            + "<section id='evidence' class='hashes'><h2>Evidence and reproducibility</h2><p>"
            + f"<a href='https://github.com/ankaggarwal94/qanta-buzzer/commit/{launch_commit}'>Launch source {launch_commit}</a>. "
            + f"<a href='https://github.com/ankaggarwal94/qanta-buzzer/tree/{analysis_commit}'>Analysis/report source {analysis_commit}</a>. "
            + f"<a href='{esc(workflow_url)}'>Workflow and retained run evidence</a>.</p>"
            "<p>This renderer verifies every file named in the CPU and binary analysis receipts, the numerical study's file manifest, "
            "the analyzers and report builder against the declared analysis commit, and numerical/binary worker source hashes against the launch commit. "
            "Those checks bind the presentation to its artifacts; they do not independently attest remote hardware execution.</p>"
            + table(["Evidence", "File", "SHA256"], sources, "Full hashes of this report's primary inputs")
            + "<details><summary>Binary numerical and provenance audit</summary><pre>" + esc(json.dumps(binary["audit"], indent=2, sort_keys=True))
            + "</pre></details></section></main></body></html>\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cpu", "binary", "numerics", "execution", "out-html"):
        parser.add_argument("--"+name, type=Path, required=True)
    for name in ("launch-commit", "analysis-commit", "workflow-url"):
        parser.add_argument("--"+name, required=True)
    parser.add_argument("--recovery", type=Path)
    args = parser.parse_args()
    rendered = build_report(args.cpu, args.binary, args.numerics, args.execution,
                            args.launch_commit, args.analysis_commit, args.workflow_url, args.recovery)
    with args.out_html.open("x", encoding="utf-8") as stream:
        stream.write(rendered)
    print(json.dumps({"status": "complete", "path": str(args.out_html), "sha256": digest(args.out_html), "bytes": args.out_html.stat().st_size}))


if __name__ == "__main__":
    main()
