#!/usr/bin/env python3
"""Prepare, execute and analyze paired prefix jobs for Jane's three RQs.

The fixture command generates artificial engineering checks, never model results.
An output directory is create-once; only a final receipt indicates completion.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import html
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import shlex
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CODE_FILES = (
    "scripts/run_jane_paired.py", "qb_data/jane_paired.py",
    "evaluation/jane_paired.py", "scripts/jane_qwen_backend.py",
)
SCOPES = ("synthetic_fixture", "legacy_engineering_control", "engineering_smoke", "scientific")
PUBLIC_JOB_FIELDS = {
    "job_id", "qid", "group_id", "split", "format", "condition", "menu_id",
    "prefix_id", "fraction", "prompt", "prompt_sha256",
}


def _hash(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _strict_json(raw: bytes) -> Any:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def bad_constant(value):
        raise ValueError(f"nonfinite JSON constant: {value}")

    return json.loads(raw.decode("utf-8"), object_pairs_hook=pairs,
                      parse_constant=bad_constant)


def _load(path: str | Path) -> Any:
    return _strict_json(Path(path).read_bytes())


def _write(path: Path, value: Any) -> None:
    with path.open("xb") as stream:
        stream.write(_json_bytes(value))


def _snapshot(paths: list[Path]) -> dict[str, str]:
    return {str(path.resolve()): _hash(path.read_bytes()) for path in paths}


def _code_snapshot() -> dict[str, str]:
    return {name: _hash((ROOT / name).read_bytes()) for name in CODE_FILES
            if (ROOT / name).is_file()}


def _runtime() -> dict[str, Any]:
    versions = {}
    for package in ("numpy", "scikit-learn", "matplotlib"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {"python": sys.version, "executable": sys.executable,
            "platform": platform.platform(), "packages": versions}


def _finish(out: Path, *, started: str, code: dict, inputs: dict,
            command: list[str], config: dict) -> None:
    """Verify immutable inputs and write the completion marker last."""
    if code != _code_snapshot():
        raise ValueError("source files changed during execution; no completion receipt")
    if inputs != _snapshot([Path(name) for name in inputs]):
        raise ValueError("input files changed during execution; no completion receipt")
    outputs = {str(path.relative_to(out)): {"sha256": _hash(path.read_bytes()),
               "bytes": path.stat().st_size}
               for path in sorted(out.rglob("*")) if path.is_file()}
    _write(out / "completion_receipt.json", {
        "schema_version": "jane-execution-receipt-v1", "status": "complete",
        "started_utc": started, "completed_utc": datetime.now(timezone.utc).isoformat(),
        "command": command, "config": config, "runtime": _runtime(),
        "code_sha256_start": code, "code_sha256_end": _code_snapshot(),
        "input_sha256_start": inputs, "input_sha256_end": inputs,
        "outputs": outputs,
        "interpretation": "Hashes bind this local execution; declared model identity is not independently authenticated.",
    })


def _jobs_package(dataset: dict, jobs: list[dict]) -> dict:
    return {"schema_version": "jane-public-jobs-v1",
            "evidence_scope": dataset["evidence_scope"], "jobs": jobs}


def _validate_public_jobs(package: Any) -> list[dict]:
    if not isinstance(package, dict) or set(package) != {
        "schema_version", "evidence_scope", "jobs"
    } or package["schema_version"] != "jane-public-jobs-v1":
        raise ValueError("expected a jane-public-jobs-v1 package")
    if package["evidence_scope"] not in SCOPES:
        raise ValueError("invalid public-job evidence scope")
    jobs = package["jobs"]
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("public jobs must be a nonempty list")
    seen = set()
    for job in jobs:
        if not isinstance(job, dict) or set(job) != PUBLIC_JOB_FIELDS:
            raise ValueError("public job has missing or unexpected fields; evaluator metadata is forbidden")
        if not isinstance(job["job_id"], str) or not job["job_id"] or job["job_id"] in seen:
            raise ValueError("job IDs must be nonempty and unique")
        seen.add(job["job_id"])
        if not isinstance(job["prompt"], str) or not job["prompt"]:
            raise ValueError("public job prompt must be nonempty")
        if job["prompt_sha256"] != _hash(job["prompt"].encode()):
            raise ValueError("public job prompt hash mismatch")
    return jobs


def _validate_backend_predictions(jobs: list[dict], predictions: Any) -> None:
    if not isinstance(predictions, list):
        raise ValueError("backend stdout must be a JSON array of predictions")
    by_id = {job["job_id"]: job for job in jobs}
    seen = set()
    required = {"job_id", "prompt_sha256", "answer", "status", "confidence", "raw_response"}
    for prediction in predictions:
        if not isinstance(prediction, dict) or not required <= prediction.keys():
            raise ValueError("backend prediction is missing required fields")
        job_id = prediction["job_id"]
        if not isinstance(job_id, str) or job_id not in by_id or job_id in seen:
            raise ValueError("backend returned duplicate or unknown job ID")
        seen.add(job_id)
        if prediction["prompt_sha256"] != by_id[job_id]["prompt_sha256"]:
            raise ValueError("backend prediction prompt hash mismatch")
        status = prediction["status"]
        confidence = prediction["confidence"]
        if status not in ("answer", "abstain", "invalid"):
            raise ValueError("unsupported backend prediction status")
        if not isinstance(prediction["raw_response"], str):
            raise ValueError("backend raw_response must be a string")
        if status == "answer":
            if not isinstance(prediction["answer"], str) or not prediction["answer"].strip():
                raise ValueError("answered prediction requires nonempty answer")
            if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
                raise ValueError("answered prediction requires finite confidence in [0,1]")
        elif confidence is not None or prediction["answer"] is not None:
            raise ValueError("abstain/invalid predictions require null answer and confidence")
    if seen != by_id.keys():
        raise ValueError("backend prediction IDs must exactly cover public jobs")


def synthetic_fixture() -> tuple[dict, dict]:
    """Return intentionally artificial questions and deterministic responses.

    The correct-answer labels deliberately generate the responses. This is a
    controlled test of bookkeeping, grading and analysis, not model inference.
    """
    from qb_data.jane_paired import build_jobs, validate_dataset

    questions = []
    for split_index, split in enumerate(("calibration", "selection", "test")):
        for index in range(12):
            number = split_index * 12 + index
            tokens = [f"artificial{number}token{position}" for position in range(50)]
            answer = f"Synthetic answer {number}"
            wrong = f"Synthetic wrong answer {number}"
            menus = []
            for menu_index, (condition, menu_id) in enumerate((
                ("current", "menu1"), ("current", "menu2"), ("hard_negative", "menu1")
            )):
                gold = (number + menu_index) % 4
                menus.append({"condition": condition, "menu_id": menu_id,
                              "options": [{"id": letter, "text": answer if option == gold else f"Artificial distractor {number}-{option}"}
                                          for option, letter in enumerate("ABCD")],
                              "gold_option_id": "ABCD"[gold],
                              "provenance": {"construction": "synthetic_fixture", "review_status": "engineering_only"}})
            questions.append({"qid": f"synthetic-{number}", "group_id": f"synthetic-group-{number}",
                              "split": split, "question": " ".join(tokens),
                              "prefixes": [{"prefix_id": f"p{step}", "text": " ".join(tokens[:10 * step]), "fraction": step / 5}
                                           for step in range(1, 6)],
                              "answer": {"raw": answer, "accepted": [answer], "rejected": [wrong], "prompt": []},
                              "menus": menus})
    dataset = {"schema_version": "jane-paired-v1", "evidence_scope": "synthetic_fixture",
               "source": {"origin": "Built-in artificial engineering fixture",
                          "provenance": {"generator": "scripts/run_jane_paired.py", "scope": "No real questions or model outputs"}},
               "questions": questions}
    validate_dataset(dataset)
    lookup = {question["qid"]: question for question in questions}
    predictions = []
    for job in build_jobs(dataset):
        question = lookup[job["qid"]]
        number = int(job["qid"].split("-")[-1])
        within_split = number % 12
        step = int(job["prefix_id"][1:])
        offset = 0 if job["format"] == "oe" else (2 if job["condition"] == "current" else 1)
        # Include early, wrong >.99 confidence and later improvement, with menu variation.
        correct = within_split % 6 < min(6, step + offset)
        if job["menu_id"] == "menu2" and step == 1 and within_split % 4 == 0:
            correct = not correct
        confidence = 0.995 if step == 1 and within_split % 3 == 0 else min(0.98, 0.40 + 0.105 * step + 0.015 * (within_split % 3))
        if job["format"] == "oe":
            answer = question["answer"]["accepted" if correct else "rejected"][0]
        else:
            menu = next(menu for menu in question["menus"] if (menu["condition"], menu["menu_id"]) == (job["condition"], job["menu_id"]))
            answer = menu["gold_option_id"] if correct else next(option["id"] for option in menu["options"] if option["id"] != menu["gold_option_id"])
        response = {"answer": answer, "confidence": confidence, "status": "answer"}
        predictions.append({"job_id": job["job_id"], "prompt_sha256": job["prompt_sha256"],
                            **response, "raw_response": json.dumps(response, sort_keys=True)})
    trace = {"schema_version": "jane-traces-v1", "metadata": {
        "model": "ARTIFICIAL_ENGINEERING_FIXTURE_NO_MODEL", "revision": "jane-fixture-v1",
        "confidence_method": "deterministic artificial confidence, not a probability estimate",
        "context_policy": "fresh_per_prefix", "evidence_scope": "synthetic_fixture"},
        "predictions": predictions}
    return dataset, trace


def _plot_data(report: dict) -> dict[str, Any]:
    """Expose exact evaluator estimates and denominators used by the figures."""
    output = {}
    for arm, data in sorted(report["rq1"]["arms"].items()):
        high = data["high_confidence_errors"]
        output[arm] = {"bins": [
            {"upper_fraction": record["upper"], "accuracy": record["accuracy"],
             "ci95": record["ci95"], "n_questions": record["n_questions"], "n_prefixes": record["n_prefixes"]}
            for record in data["curve"]], "raw_high_confidence": {
                "condition": "raw confidence strictly > 0.99; test answer rows",
                "n": high["n_predictions"], "errors": high["n_incorrect"],
                "error_rate": high["error_rate"]}}
    return output


def _render(out: Path, report: dict, rows: list[dict], scope: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_data = _plot_data(report)
    report["plot_data"] = plot_data
    label = "SYNTHETIC ENGINEERING CHECK — NO MODEL RESULTS" if scope == "synthetic_fixture" else (
        "LEGACY ENGINEERING CONTROL" if scope == "legacy_engineering_control" else (
            "MODEL ENGINEERING SMOKE — NOT SCIENTIFIC RESULTS" if scope == "engineering_smoke" else "SUPPLIED MODEL TRACE ANALYSIS"))
    scope_note = {
        "synthetic_fixture": "Responses are generated using evaluator labels and deliberately constructed patterns. They cannot establish an LLM finding or reproduce the earlier Qwen pilot.",
        "engineering_smoke": "This small run exercises model generation and analysis on authentic question text. It is not the earlier Qwen pilot and does not support general empirical conclusions.",
        "legacy_engineering_control": "This control reuses legacy data for engineering validation. It does not establish open-ended model generation or reproduce the earlier Qwen pilot.",
        "scientific": "Interpret findings against the declared corpus, grading, split, model, and generation provenance. This analysis does not reproduce the earlier Qwen pilot by itself.",
    }[scope]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for arm, data in plot_data.items():
        bins = [record for record in data["bins"] if record["accuracy"] is not None]
        line, = ax.plot([record["upper_fraction"] for record in bins], [record["accuracy"] for record in bins], marker="o", label=arm)
        interval_bins = [record for record in bins if record["ci95"] is not None]
        if interval_bins:
            ax.fill_between([record["upper_fraction"] for record in interval_bins],
                            [record["ci95"][0] for record in interval_bins],
                            [record["ci95"][1] for record in interval_bins],
                            color=line.get_color(), alpha=.10)
    ax.set(xlabel="Revealed token fraction (bin upper boundary)", ylabel="Unconditional accuracy", ylim=(-.03, 1.03), xlim=(0, 1.03), title=label)
    ax.grid(alpha=.2)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "test_accuracy.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 5.2))
    arms = list(plot_data)
    values = [plot_data[arm]["raw_high_confidence"]["error_rate"] for arm in arms]
    observed = [index for index, value in enumerate(values) if value is not None]
    ax.bar(observed, [values[index] for index in observed], color="#b95146")
    for index, arm in enumerate(arms):
        data = plot_data[arm]["raw_high_confidence"]
        ax.text(index, (values[index] or 0) + .025, f"{data['errors']}/{data['n']}" if data["n"] else "no observations", ha="center", fontsize=9)
    ax.set_xticks(range(len(arms)), arms, rotation=12, ha="right", fontsize=8)
    ax.set(ylabel="Error rate among answers with raw confidence > 0.99", ylim=(0, 1.13), title=label)
    fig.tight_layout()
    fig.savefig(out / "high_confidence_errors.png", dpi=160)
    plt.close(fig)
    def scalar(value):
        return "undefined" if value is None else f"{value:.3f}"

    def estimate(value, interval=None):
        interval_text = "undefined" if interval is None else f"[{interval[0]:.3f}, {interval[1]:.3f}]"
        return f"{scalar(value)}; 95% CI {interval_text}"

    def table(headers, table_rows):
        heading = "".join(f"<th>{html.escape(str(value))}</th>" for value in headers)
        body = "".join("<tr>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in row) + "</tr>" for row in table_rows)
        return f'<div class="table-wrap"><table><thead><tr>{heading}</tr></thead><tbody>{body}</tbody></table></div>'

    metadata = report["evidence"]["trace_metadata"]
    identity_html = table(["Model", "Revision", "Confidence definition", "Context"], [[
        metadata["model"], metadata["revision"], metadata["confidence_method"], metadata["context_policy"]]])
    split_html = table(["Split", "Questions", "Groups", "Prefixes per arm", "Arms"], [
        [split, data["n_questions"], data["n_groups"], data["n_prefixes_per_arm"], data["n_arms"]]
        for split, data in report["counts"].items()])
    accuracy_rows, high_rows, quality_rows = [], [], []
    report["format_quality_test"] = {}
    for arm, data in report["rq1"]["arms"].items():
        early, full = data["early_accuracy"], data["full_question_accuracy"]
        accuracy_rows.append([arm, estimate(early["accuracy"], early["ci95"]),
                              f"{early['n_questions']} / {early['n_prefixes']}",
                              estimate(full["accuracy"], full["ci95"]),
                              f"{full['n_questions']} / {full['n_prefixes']}"])
        for scope_name, key in (("first 20%", "early_high_confidence_errors"), ("all prefixes", "high_confidence_errors")):
            high = data[key]
            high_rows.append([arm, scope_name, f"{high['n_incorrect']} / {high['n_predictions']}",
                              high["n_questions"], estimate(high["error_rate"], high["ci95"])])
        arm_rows = [row for row in rows if row["split"] == "test" and (
            "oe" if row["format"] == "oe" else f"mc:{row['condition']}:{row['menu_id']}"
        ) == arm]
        grades = Counter(row["grade"] for row in arm_rows)
        quality = {grade: grades[grade] for grade in ("accepted", "rejected", "invalid", "abstain")}
        quality["n_predictions"] = len(arm_rows)
        report["format_quality_test"][arm] = quality
        quality_rows.append([arm, len(arm_rows), quality["accepted"], quality["rejected"], quality["invalid"], quality["abstain"]])
    accuracy_html = table(["Arm", "Early accuracy (fraction ≤ 0.2)", "Early questions / prefixes",
                           "Full-question accuracy", "Full questions / prefixes"], accuracy_rows)
    high_html = table(["Arm", "Prefix scope", "Errors / predictions", "Questions", "Error rate"], high_rows)
    quality_html = table(["Arm", "Test predictions", "Accepted", "Rejected", "Invalid", "Abstained"], quality_rows)
    contrast_rows, contrast_bin_rows = [], []
    for contrast in report["rq2"]["contrasts"]:
        name = f"{contrast['arm_a']} minus {contrast['arm_b']}"
        early = contrast["early_accuracy"]
        contrast_rows.append([name, estimate(early["accuracy_difference"], early["ci95"]),
                              early["n_questions"], early["n_prefixes"]])
        for record in contrast["curve"]:
            if record["n_questions"]:
                contrast_bin_rows.append([name, f"({record['lower']:.1f}, {record['upper']:.1f}]",
                                          estimate(record["accuracy_difference"], record["ci95"]),
                                          record["n_questions"], record["n_prefixes"]])
    contrast_html = table(["Paired contrast", "Early accuracy difference", "Questions", "Prefixes"], contrast_rows)
    contrast_bins_html = table(["Paired contrast", "Fraction bin", "Accuracy difference", "Questions", "Prefixes"], contrast_bin_rows)
    policy_labels = {
        "mc_selected_on_mc": "MC-selected → MC", "zero_oe_label_transfer": "MC-selected → OE (zero OE labels)",
        "oe_calibrated_mc_threshold": "OE-calibrated + MC threshold → OE",
        "oe_selected_on_oe": "OE-calibrated + OE threshold → OE",
    }
    policy_sections = []
    for arm, data in report["rq3"]["arms"].items():
        policy_rows = []
        for name, policy in data["policies"].items():
            metrics = policy["metrics"]
            threshold = "never commit" if policy["threshold"]["mode"] == "never" else scalar(policy["threshold"]["value"])
            policy_rows.append([
                policy_labels[name], policy["target_arm"], policy["n_oe_development_labeled_questions"], threshold,
                f"{metrics['n_committed']} / {metrics['n_questions']}",
                f"{metrics['n_correct_commits']} / {metrics['n_incorrect_commits']} / {metrics['n_no_commit']}",
                estimate(metrics["risk"], metrics["ci95"]["risk"]),
                estimate(metrics["coverage"], metrics["ci95"]["coverage"]),
                estimate(metrics["mean_commitment_fraction"], metrics["ci95"]["mean_commitment_fraction"]),
            ])
        policy_html = table(["Policy", "Test arm", "OE dev. labeled questions", "Calibrated threshold",
                             "Committed / test questions", "Correct / wrong / no commit",
                             "Selective risk", "Coverage", "Mean fraction at commitment"], policy_rows)
        comparison_rows = []
        for comparison in data["policy_comparisons"]:
            differences = comparison["differences"]
            joint = comparison["joint_commitment_counts"]
            comparison_rows.append([
                f"{policy_labels[comparison['policy_a']]} minus {policy_labels[comparison['policy_b']]}",
                estimate(differences["risk"]["estimate"], differences["risk"]["ci95"]),
                estimate(differences["coverage"]["estimate"], differences["coverage"]["ci95"]),
                f"{joint['both_commit']} / {joint['a_only']} / {joint['b_only']} / {joint['neither']}",
            ])
        comparison_html = table(["Paired policy contrast (A minus B)", "Risk difference", "Coverage difference",
                                 "Both / A only / B only / neither commit"], comparison_rows)
        budget_rows = [[budget_arm, budget["calibration_questions"], budget["selection_questions"],
                        budget["calibration_answer_prefix_labels"], budget["selection_prefix_labels"]]
                       for budget_arm, budget in data["development_budget"].items()]
        budget_html = table(["Development arm", "Calibration questions", "Selection questions",
                             "Calibration answered-prefix labels", "Selection prefix labels"], budget_rows)
        policy_sections.append(f"<h3>{html.escape(arm)}</h3>{policy_html}<details><summary>Development label budget</summary>{budget_html}</details>{comparison_html}")
    policy_html = "".join(policy_sections)
    warnings_html = "".join(f"<li>{html.escape(warning)}</li>" for warning in report["warnings"])
    document = f"""<!doctype html><html lang="en"><meta charset="utf-8"><title>Jane paired evaluation</title>
<style>body{{font:16px/1.55 system-ui,sans-serif;max-width:1200px;margin:40px auto;padding:0 24px;color:#1c2933}}h1{{font-size:28px}}h2{{margin-top:36px}}.scope{{padding:18px;background:#fff0d8;border-left:6px solid #ba7200}}img{{width:100%;height:auto}}.table-wrap{{overflow-x:auto;margin:16px 0}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{border:1px solid #ccd4da;padding:8px 12px;text-align:left;vertical-align:top}}th{{background:#eff3f6}}a{{color:#1559a3}}summary{{cursor:pointer}}</style>
<h1>Jane's paired prefix evaluation</h1><p class="scope"><strong>{html.escape(label)}</strong><br>
Evidence scope: {html.escape(scope)}. {html.escape(scope_note)}</p>
{identity_html}{split_html}
<p>Calibration and threshold selection use separate development splits; figures use test rows. Threshold selection is empirical and exploratory, not a risk guarantee. Reported bootstrap intervals are conditional on fitted calibrators and thresholds, with whole-group resampling and no refitting.</p>
<p><a href="report.json">Full analysis and policy-transfer report (JSON)</a> · <a href="graded_rows.json">Graded rows</a> · <a href="completion_receipt.json">Execution receipt</a></p>
<h2>RQ1: accuracy across prefixes</h2><p>Within each fraction bin, prefixes are averaged within question, then across questions. Empty bins are omitted. Shaded bands are 95% whole-group bootstrap intervals for each arm; paired differences are reported separately below. Values are proportions. Undefined estimates and intervals are displayed explicitly.</p>{accuracy_html}<img src="test_accuracy.png" alt="Test accuracy by revealed token fraction for each format and menu with available 95 percent bootstrap bands">
<h2>Test response and grading outcomes</h2><p>These are prefix-prediction counts over full test trajectories, including prefixes after any selected commitment. Invalid responses and abstentions are unconditional accuracy errors and cannot trigger commitment. Unresolved grading prevents this report from completing.</p>{quality_html}
<h2>Errors at raw confidence above 0.99</h2><p>The subset uses a strict greater-than comparison. Ratios count test prefix predictions; these are not independent question counts.</p><img src="high_confidence_errors.png" alt="Wrong predictions divided by all predictions with raw confidence above 0.99">
{high_html}
<h2>RQ2: paired option and menu contrasts</h2><p>Differences subtract the second named arm from the first and use identical whole-group bootstrap draws. Early accuracy includes fraction ≤ 0.2. These contrasts characterize the supplied menus; a condition name does not establish distractor quality or pyramid alignment.</p>{contrast_html}<details><summary>Paired contrasts by fraction bin</summary>{contrast_bins_html}</details>
<h2>RQ3: stopping-policy transfer</h2><p>The observed development error budget is {scalar(report['config']['risk_budget'])}. Each row evaluates a frozen first-crossing policy on {report['counts']['test']['n_questions']} held-out questions. OE label counts refer only to additional OE development labels, not the MC labels used to select the source policy. Never-commit policies have zero coverage and undefined selective risk and commitment fraction. Intervals remain conditional on the selected calibrator and threshold.</p>{policy_html}
<p>Paired risk and commitment-fraction differences compare each policy's own committing population; they are not restricted to jointly answered questions. Joint commitment counts make that population difference visible. Equal labeled-question budgets do not imply equal annotation effort.</p>
<h2>Interpretation and limitations</h2><ul>{warnings_html}</ul><p>No uncertainty interval, model declaration, or completion receipt authenticates how externally supplied predictions were generated. Scientific conclusions additionally require verified corpus provenance, reviewed full answerlines, model execution evidence, and a justified analysis plan.</p></html>"""
    with (out / "report.html").open("x", encoding="utf-8") as stream:
        stream.write(document)


def _analyze_to(out: Path, dataset: dict, trace: dict, adjudications: dict | None, args) -> None:
    from qb_data.jane_paired import build_jobs, grade_predictions
    from evaluation.jane_paired import analyze

    jobs = build_jobs(dataset)
    rows = grade_predictions(dataset, jobs, trace, adjudications=adjudications)
    report = analyze(rows, risk_budget=args.risk_budget,
                     bootstrap_samples=args.bootstrap_samples, seed=args.seed)
    report["evidence"] = {"scope": dataset["evidence_scope"], "trace_metadata": trace["metadata"],
                          "model_identity_verification": "declared_only",
                          "pilot_reproduction": False}
    _render(out, report, rows, dataset["evidence_scope"])
    _write(out / "graded_rows.json", rows)
    _write(out / "report.json", report)


def main(argv: list[str] | None = None) -> int:
    """Run one create-once operation; validation failures produce no receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "analyze", "fixture", "run"):
        sub = subparsers.add_parser(name)
        sub.add_argument("--out", required=True, type=Path, help="New output directory; overwrite is refused")
        if name in ("prepare", "analyze"):
            sub.add_argument("--dataset", required=True, type=Path)
        if name == "analyze":
            sub.add_argument("--trace", required=True, type=Path)
            sub.add_argument("--adjudications", type=Path)
        if name in ("analyze", "fixture"):
            sub.add_argument("--risk-budget", type=float, default=.1)
            sub.add_argument("--bootstrap-samples", type=int, default=500)
            sub.add_argument("--seed", type=int, default=1)
        if name == "run":
            sub.add_argument("--jobs", required=True, type=Path)
            sub.add_argument("--backend-command", required=True, help="Executable and arguments; invoked without a shell")
            sub.add_argument("--model", required=True)
            sub.add_argument("--revision", required=True)
            sub.add_argument("--confidence-method", required=True)
            sub.add_argument("--timeout", type=float, default=3600)
    command = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(command)
    input_paths = [getattr(args, name) for name in ("dataset", "trace", "adjudications", "jobs") if getattr(args, name, None) is not None]
    code = _code_snapshot()
    inputs = _snapshot(input_paths)
    started = datetime.now(timezone.utc).isoformat()
    args.out.mkdir(parents=True, exist_ok=False)
    if args.command == "prepare":
        from qb_data.jane_paired import build_jobs
        dataset = _load(args.dataset)
        _write(args.out / "public_jobs.json", _jobs_package(dataset, build_jobs(dataset)))
    elif args.command == "analyze":
        _analyze_to(args.out, _load(args.dataset), _load(args.trace),
                    _load(args.adjudications) if args.adjudications else None, args)
    elif args.command == "fixture":
        from qb_data.jane_paired import build_jobs
        dataset, trace = synthetic_fixture()
        _write(args.out / "dataset.json", dataset)
        _write(args.out / "trace.json", trace)
        _write(args.out / "public_jobs.json", _jobs_package(dataset, build_jobs(dataset)))
        _analyze_to(args.out, dataset, trace, None, args)
    else:
        package = _load(args.jobs)
        jobs = _validate_public_jobs(package)
        if any(not value.strip() for value in (args.model, args.revision, args.confidence_method)):
            raise ValueError("model, revision, and confidence method must be nonempty")
        if not math.isfinite(args.timeout) or args.timeout <= 0:
            raise ValueError("timeout must be positive and finite")
        backend = shlex.split(args.backend_command)
        if not backend:
            raise ValueError("backend command must not be empty")
        result = subprocess.run(backend, input=_json_bytes(package), stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, timeout=args.timeout, check=False)
        with (args.out / "backend_stderr.txt").open("xb") as stream:
            stream.write(result.stderr)
        if result.returncode:
            raise ValueError(f"backend exited with status {result.returncode}; inspect backend_stderr.txt")
        backend_output = _strict_json(result.stdout)
        expected_metadata = {
            "model": args.model, "revision": args.revision,
            "confidence_method": args.confidence_method, "context_policy": "fresh_per_prefix",
            "evidence_scope": package["evidence_scope"]}
        if isinstance(backend_output, dict):
            if backend_output.get("schema_version") != "jane-traces-v1" or not isinstance(backend_output.get("metadata"), dict):
                raise ValueError("backend trace must use jane-traces-v1 with metadata")
            if any(backend_output["metadata"].get(key) != value for key, value in expected_metadata.items()):
                raise ValueError("backend trace metadata does not match declared run configuration")
            trace = backend_output
            predictions = trace.get("predictions")
        else:
            predictions = backend_output
            trace = {"schema_version": "jane-traces-v1", "metadata": expected_metadata,
                     "predictions": predictions}
        _validate_backend_predictions(jobs, predictions)
        _write(args.out / "trace.json", trace)
    _finish(args.out, started=started, code=code, inputs=inputs, command=command,
            config={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()})
    print(str(args.out / "completion_receipt.json"))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2) from error
