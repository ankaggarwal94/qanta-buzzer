#!/usr/bin/env python3
"""Build a self-contained scientific HTML report from reviewed IMCQA analyses.

This renderer does not run inference, select policies, or recompute scientific
estimates. It checks analyzer receipts and displays the supplied results. A
missing pilot is visibly pending; a supplied incomplete pilot is rejected.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import io
import json
import math
from pathlib import Path
import re
from typing import Any


MODELS = {"qwen3b": "Qwen 3B", "qwen7b": "Qwen 7B"}
MENUS = {"independent_pool": "Independent pool", "same_category_pool": "Same-category pool"}
REPOSITORY = "https://github.com/ankaggarwal94/qanta-buzzer"


def sha256(path: Path) -> str:
    """Return the SHA256 digest of a local evidence file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    """Read finite JSON and reject duplicate keys."""
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    def reject(value: str) -> None:
        raise ValueError(f"Nonfinite JSON {value} in {path}")

    def finite_float(value: str) -> float:
        parsed = float(value)
        if not math.isfinite(parsed):
            reject(value)
        return parsed

    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique,
                      parse_constant=reject, parse_float=finite_float)


def report_path(path: Path) -> Path:
    return path / "report.json" if path.is_dir() else path


def verified_report(path: Path, *, pilot: bool = False) -> dict[str, Any]:
    """Require the matching complete analyzer receipt before presentation."""
    report = read_json(path)
    receipt = read_json(path.with_name("analysis_receipt.json"))
    statuses = {"complete"} if pilot else {"completed"}
    if receipt.get("status") not in statuses:
        raise ValueError(f"Incomplete analyzer receipt: {path.parent}")
    outputs = receipt.get("output_sha256" if pilot else "outputs_sha256", {})
    if outputs.get(path.name) != sha256(path):
        raise ValueError(f"Report does not match its analyzer receipt: {path}")
    if not pilot:
        audit = path.with_name("evidence_audit.json")
        if outputs.get(audit.name) != sha256(audit):
            raise ValueError("Retrospective evidence audit hash mismatch")
        evidence = read_json(audit)
        if (report.get("schema_version") != "imcqa-retrospective-v1"
                or evidence.get("mc_rows_total") != 200000
                or evidence.get("questions") != 5000
                or evidence.get("question_splits") != {"calibration": 1000, "selection": 1000, "test": 3000}):
            raise ValueError("Unexpected retrospective scientific scope")
    elif (report.get("schema_version") != "imcqa-wait-analysis-v1"
          or report.get("evidence_scope") != "exploratory_development_only"
          or report.get("n_questions") != 200 or report.get("n_score_rows") != 9600
          or set(report.get("audits", {})) != set(MODELS)
          or not all(item.get("passed") for item in report["audits"].values())):
        raise ValueError("Supplied pilot is incomplete or fails its declared scope/audits")
    return report


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def number(value: float | None, decimals: int = 3) -> str:
    return "undefined" if value is None else f"{value:.{decimals}f}"


def percent(value: float | None, decimals: int = 2) -> str:
    return "undefined" if value is None else f"{100 * value:.{decimals}f}%"


def ci(values: list[float] | None, *, percentage: bool = False) -> str:
    if values is None:
        return "undefined"
    formatter = percent if percentage else number
    return f"[{formatter(values[0])}, {formatter(values[1])}]"


def table(headers: list[str], rows: list[list[Any]], caption: str) -> str:
    """Render an accessible plain-value table without unescaped source HTML."""
    head = "".join(f"<th scope='col'>{esc(value)}</th>" for value in headers)
    body = "".join("<tr>" + "".join(f"<td>{esc(value)}</td>" for value in row) + "</tr>"
                   for row in rows)
    return (f"<div class='table-wrap'><table><caption>{esc(caption)}</caption>"
            f"<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>")


def retrospective_policy(report: dict[str, Any], model: str, menu: str, policy: str) -> dict[str, Any]:
    return report["cells"][f"{model}/{menu}"]["splits"]["test"]["policies"][policy]


def prefix_plot(report: dict[str, Any]) -> str:
    """Return an inline SVG of reviewed test accuracy across ten prefixes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 11,
                         "svg.fonttype": "none", "axes.spines.top": False,
                         "axes.spines.right": False}):
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.8), sharey=True)
        for axis, (menu, title) in zip(axes, MENUS.items()):
            for model, color in (("qwen3b", "#176b91"), ("qwen7b", "#9a4d1f")):
                values = [retrospective_policy(report, model, menu, f"fixed_round_{round_number}")["correct_per_question"]
                          for round_number in range(1, 11)]
                axis.plot(range(10, 101, 10), values, label=MODELS[model], color=color,
                          marker="o", markersize=4, linewidth=2)
            axis.set(title=title, xlabel="Nominal question text revealed (%)", xlim=(7, 103), ylim=(0, 1))
            axis.set_xticks([10, 20, 40, 60, 80, 100])
            axis.yaxis.set_major_formatter(PercentFormatter(1))
            axis.grid(axis="y", color="#dfe6eb", linewidth=.7)
            axis.legend(loc="lower right", frameon=False)
        axes[0].set_ylabel("Correct / all questions")
        fig.tight_layout(pad=1.0)
        output = io.StringIO()
        fig.savefig(output, format="svg", metadata={"Date": None})
        plt.close(fig)
    svg = output.getvalue()
    svg = svg[svg.index("<svg"):]
    return svg.replace("<svg ", "<svg role='img' aria-label='Held-out question accuracy by prefix for Qwen 3B and 7B in each menu regime' ", 1)


def retrospective_section(report: dict[str, Any]) -> str:
    rows = []
    for model in MODELS:
        for menu in MENUS:
            final = retrospective_policy(report, model, menu, "final_round")
            first = retrospective_policy(report, model, menu, "first_answer")
            selected = retrospective_policy(report, model, menu, "confidence_reward_selected")
            rows.append([MODELS[model], MENUS[menu],
                         f"{percent(final['correct_per_question'])} ({final['n_correct']:,}/3,000)",
                         number(first["rewards"]["early_wrong1"]),
                         f"{number(selected['rewards']['early_wrong1'])} {ci(selected['bootstrap']['reward_early_wrong1']['ci95'])}",
                         f"{percent(selected['risk'])} ({selected['n_wrong']:,}/{selected['n_committed']:,})"])
    diff = report["test_comparisons"]["qwen7b/confidence_reward_selected/same_category-minus-independent"]["reward_early_wrong1_difference"]
    content = "<section id='retrospective'><h2>Full-question accuracy misses a difference in early-answer reward</h2>"
    content += ("<p>The same 3,000 held-out questions are compared in each cell. "
                "Full-question accuracy counts abstentions and invalid outputs as not correct. "
                "The first-answer policy commits at the first prefix that produced an answer. "
                "The reward-selected policy uses an isotonic-calibrated self-reported confidence "
                "threshold, including a never-answer candidate. Calibration used 1,000 questions; "
                "policy selection used a separate 1,000.</p>")
    content += table(["Model", "Menu", "Full-question accuracy", "First-answer reward", "Selected reward [95% CI]", "Selected error / commitments"],
                     rows, "Held-out test results; rewards are unitless mean payoff per question")
    content += ("<p class='note'>Retrospective payoff: a correct answer at round r earns (11−r)/10, "
                "an incorrect answer earns −1, and no answer earns 0, across ten rounds. "
                "These rewards were applied after generation; the original prompts described neither "
                "this game nor a WAIT action.</p>")
    content += (f"<p>For 7B, the selected-policy reward difference, same-category minus independent, "
                f"is <strong>{number(diff['mean'])}</strong> with paired 95% interval "
                f"<strong>{ci(diff['ci95'])}</strong>. Nearly equal endpoint accuracy therefore "
                "does not imply equal usefulness under this early-answer objective. This is a "
                "comparison of separately selected policies, not a pure stopping-time treatment effect.</p>")
    content += "<figure>" + prefix_plot(report) + ("<figcaption>Each point uses all 3,000 test questions; "
                "abstentions and invalid responses count as not correct. Curves show independent-prefix "
                "generations, not sequential visits. The x-axis uses the original nominal word deciles.</figcaption></figure>")
    content += ("<p class='note'>Intervals use 2,000 paired question-bootstrap resamples, conditional on the "
                "fitted calibration maps and selected policies, without multiplicity correction. "
                "The test partition was held out from fitting; this retrospective analysis was formulated "
                "after earlier results were observed and is not a new confirmatory preregistration.</p></section>")
    return content


def risk_section(report: dict[str, Any]) -> str:
    rows = []
    for model in MODELS:
        for menu in MENUS:
            row = retrospective_policy(report, model, menu, "historical_risk10_selected")
            rows.append([MODELS[model], MENUS[menu], f"{row['n_committed']:,}/3,000",
                         row["n_wrong"], percent(row["risk"]), ci(row["risk_clopper_pearson95"], percentage=True)])
    return ("<section><h2>The historical 10% risk rule gives too few answers to establish safety</h2>"
            + table(["Model", "Menu", "Commitments", "Errors", "Observed risk", "Exact 95% risk interval"], rows,
                    "Frozen historical policy, replayed unchanged on the test partition")
            + "<p>Zero observed errors in seven or four commitments does not establish a population "
              "error rate below 10%. Exact two-sided Clopper–Pearson intervals are shown because an "
              "ordinary empirical bootstrap of zero errors degenerates at zero. With no commitments, "
              "conditional risk is undefined. The selection rule was an empirical filter, not a risk guarantee.</p></section>")


def menu_section(report: dict[str, Any]) -> str:
    rows = []
    for model in MODELS:
        for menu in MENUS:
            original = report["cells"][f"{model}/{menu}/original/test"]
            forced = report["cells"][f"{model}/{menu}/forced/test"]
            if original["n_questions"] != 3000 or forced["n_questions"] != 3000:
                raise ValueError("Menu control test denominator is not 3000")
            rows.append([MODELS[model], MENUS[menu], percent(original["accuracy"]), percent(forced["accuracy"]),
                         percent(original["mean_max_probability"]), percent(forced["mean_max_probability"])])
    return ("<section><h2>Peaked questionless scores do not establish answer knowledge</h2>"
            + table(["Model", "Menu", "Original accuracy", "Forced accuracy", "Original mean top p", "Forced mean top p"], rows,
                    "Questionless A-D scoring; each cell contains 3,000 test menus")
            + "<p>The gold positions are balanced, so uniform selection has 25% expected accuracy. "
              "The top probabilities are softmax values restricted to A-D after a supplied answer "
              "prefix, not calibrated probabilities of correctness. This control provides no "
              "abstention rate or generated-response frequency. The prompt comparison measures "
              "instruction sensitivity and does not isolate a classical IIA test.</p></section>")


def pilot_section(report: dict[str, Any] | None) -> str:
    intro = ("<section id='pilot'><h2>An explicit WAIT pilot tests the missing decision protocol</h2>"
             "<p>The pilot uses 200 development questions, 100 from calibration and 100 from selection, "
             "with no test questions. Five original prefixes reveal nominally 20%, 40%, 60%, 80%, "
             "and 100% of the question. A-D commit; E requests the next prefix in rounds 1–4 "
             "and means terminal PASS in round 5. Correct rewards are 1, .8, .6, .4, .2; wrong "
             "answers earn −1 and PASS earns 0. This five-round schedule differs from the "
             "ten-round retrospective schedule above.</p>"
             "<p>The policy is FP32 argmax over legal next-token action labels after the supplied "
             "<code>{&quot;action&quot;:&quot;</code> prefix. States use a fresh context and assume earlier WAITs. "
             "All-round scoring supports counterfactual replay; active-only execution checks the "
             "same policy on eight preselected questions. This is an explicit constrained decision "
             "policy, not free-response generation or repeated sampling.</p>")
    if report is None:
        return intro + "<p class='pending'><strong>Pending:</strong> no complete, validated pilot report was supplied. No pilot outcomes are claimed.</p></section>"
    rows = []
    primary = [row for row in report["policy_summaries"] if row["variant"] == "wait" and row["policy"] == "first_commit"]
    if len(primary) != 8:
        raise ValueError("Expected eight primary pilot model/split/menu cells")
    for row in primary:
        rows.append([MODELS[row["model"]], row["split"].capitalize(), MENUS[row["condition"]],
                     f"{number(row['mean_reward'])} {ci(row['bootstrap']['mean_reward']['ci95'])}",
                     f"{row['n_committed']}/{row['n_questions']}",
                     f"{row['n_wrong']}/{row['n_committed']} ({percent(row['risk'])})",
                     row["n_terminal_pass"]])
    content = intro + table(["Model", "Split", "Menu", "Mean reward [95% CI]", "Commitments", "Errors / commitments", "PASS"],
                            rows, "Primary explicit-action policy; 100 questions per cell, development data only")
    policy_lookup = {(row["model"], row["split"], row["condition"], row["variant"], row["policy"]): row
                     for row in report["policy_summaries"]}
    baseline_rows = []
    below_pass = 0
    for model in MODELS:
        for menu in MENUS:
            primary_reward = policy_lookup[model, "selection", menu, "wait", "first_commit"]["mean_reward"]
            forced_first = policy_lookup[model, "selection", menu, "forced", "fixed_round_1"]["mean_reward"]
            forced_final = policy_lookup[model, "selection", menu, "forced", "fixed_round_5"]["mean_reward"]
            pass_reward = policy_lookup[model, "selection", menu, "forced", "always_terminal_pass"]["mean_reward"]
            below_pass += primary_reward < pass_reward
            baseline_rows.append([MODELS[model], MENUS[menu], number(primary_reward),
                                  number(forced_first), number(forced_final), number(pass_reward)])
    content += table(["Model", "Menu", "Primary WAIT policy", "Answer now: round 1", "Answer now: round 5", "Always PASS"],
                     baseline_rows, "Selection split: descriptive mean reward, 100 paired questions per cell")
    content += (f"<p>The primary mean reward is below the always-PASS value in {below_pass} of "
                "the four selection cells. Beating the final-round answer-now baseline alone therefore "
                "does not establish a useful stopping policy. The first-round and PASS columns are "
                "descriptive reference values; no paired first-round uncertainty estimate is claimed.</p>")
    comparisons = []
    for row in report["paired_comparisons"]:
        if row["comparison"] != "primary_minus_forced_final":
            continue
        estimate = row["differences"]["mean_reward"]
        comparisons.append([MODELS[row["model"]], row["split"].capitalize(), MENUS[row["condition"]],
                            f"{number(estimate['mean'])} {ci(estimate['ci95'])}"])
    content += table(["Model", "Split", "Menu", "Reward difference [95% CI]"], comparisons,
                     "Primary minus answer-now-at-final-round under the same five-round game prompt")
    diagnostics = {(row["model"], row["condition"], row["diagnostic"]): row
                   for row in report["diagnostic_summaries"] if row["split"] == "selection"}
    diagnostic_rows = []
    for model in MODELS:
        for menu in MENUS:
            forced = diagnostics[model, menu, "forced_prompt_effect"]
            rotation = diagnostics[model, menu, "semantic_rotation_effect"]
            if (forced["n_questions"], forced["n_paired_states"], rotation["n_questions"], rotation["n_paired_states"]) != (100, 500, 20, 100):
                raise ValueError("Unexpected selection diagnostic denominators")
            formatted = []
            for metric in (forced["conditional_top_changed"], rotation["semantic_action_changed"], rotation["wait_decision_changed"]):
                formatted.append(f"{percent(metric['mean'], 1)} {ci(metric['ci95'], percentage=True)}")
            diagnostic_rows.append([MODELS[model], MENUS[menu], *formatted])
    content += table(["Model", "Menu", "Answer-now prompt: top A-D changes", "Rotation: semantic action changes", "Rotation: E decision changes"],
                     diagnostic_rows, "Selection diagnostics, rates [95% question-bootstrap CI]: prompt comparison 500 states / 100 questions; rotation 100 states / 20 questions")
    content += ("<p class='note'>The prompt comparison conditions the primary scores on A-D and "
                "compares their top candidate with the separately scored answer-now prompt. Rotation "
                "changes are computed after mapping labels back to candidate identities. E means WAIT "
                "before the last round and PASS at the last round; its changes include both. These "
                "rates use all five scored states per question, including states not visited after an "
                "early commitment. They are not independent-state sample sizes or episode-level flip rates.</p>")
    controls = []
    for row in report["paired_comparisons"]:
        if row["comparison"] not in {"primary_minus_questionless_matched_subset", "primary_minus_rotation_matched_subset"}:
            continue
        estimate = row["differences"]["mean_reward"]
        control = "Questionless" if "questionless" in row["comparison"] else "Cyclic rotation"
        controls.append([MODELS[row["model"]], row["split"].capitalize(), MENUS[row["condition"]], control,
                         row["n_questions"], f"{number(estimate['mean'])} {ci(estimate['ci95'])}"])
    content += ("<details><summary>Matched questionless and position controls</summary>"
                + table(["Model", "Split", "Menu", "Control", "Paired questions", "Primary − control reward [95% CI]"], controls,
                        "Diagnostic subset: 20 questions per split, shared by the questionless and rotation conditions")
                + "</details><p class='note'>Pilot intervals are descriptive question-bootstrap intervals. "
                  "A small or empty set of commitments cannot establish a population risk guarantee. "
                  "The answer-now baseline retains the same payoff-aware game description; it is "
                  "not the original full-question MCQA protocol.</p>")
    audit_rows = []
    for model, audit in report["audits"].items():
        audit_rows.append([MODELS[model], audit.get("n_rows"), "passed" if audit.get("passed") else "failed",
                           number(audit.get("elapsed_seconds"), 1)])
    content += table(["Model", "Validated rows", "Numerical/provenance/replay audit", "Scorer seconds"], audit_rows,
                     "Scorer time excludes worker setup and provider startup; it is not complete billed allocation time")
    return content + "</section>"


def execution_section(execution: dict[str, Any] | None) -> str:
    if execution is None:
        return ("<p class='note'>The pilot allocation ceiling is $4.00. No worker-timing and cost "
                "summary was supplied to this renderer; the ceiling is not actual spend.</p>")
    rows = []
    for key, label, unit in (("workflow_elapsed_seconds", "Workflow wall time", "seconds"),
                             ("estimated_compute_usd", "Estimated compute, stated allowance included", "USD"),
                             ("reserved_estimate_usd", "Reserved estimate", "USD")):
        value = execution.get(key)
        if value is not None:
            value = float(value)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid execution metric {key}")
            rows.append([label, number(value, 3 if unit == "USD" else 1), unit])
    for model, value in execution.get("allocation_seconds_by_model", {}).items():
        value = float(value)
        if model not in MODELS or not math.isfinite(value) or value < 0:
            raise ValueError("Invalid worker function elapsed seconds")
        rows.append([f"{MODELS[model]} worker function elapsed", number(value, 1), "seconds"])
    verified = execution.get("invoice_verified") is True
    return ("<section><h2>Runtime and cost</h2>" + table(["Quantity", "Value", "Unit"], rows,
             "Supplied execution summary; worker function seconds are additive, parallel workflow wall time is not")
            + f"<p>Invoice verification: {'verified in supplied summary' if verified else 'not verified'}. "
              "Compute estimates are not billing statements. The pilot ceiling is $4.00. "
              "The planning rate for one L40S worker with two physical CPU cores and 32 GiB memory "
              "is $0.00063924 per second; see <a href='https://modal.com/pricing'>Modal pricing</a>. "
              "Worker function timing covers only part of the allocation: it excludes provider startup, "
              "pre-timer source and claim checks, initial volume reload, final receipt commit, and idle "
              "scaledown. Startup or contingency allowances in the estimate are separate from these "
              "observed function timings.</p>"
            + (f"<p>{esc(execution['notes'])}</p>" if execution.get("notes") else "") + "</section>")


def build_report(retrospective_path: Path, menus_path: Path, pilot_path: Path | None, *,
                 source_commit: str | None = None, workflow_url: str | None = None,
                 execution_path: Path | None = None) -> str:
    """Assemble the checked reports into a portable document.

    Parameters
    ----------
    retrospective_path, menus_path, pilot_path
        Reviewed JSON evidence paths. The retrospective and pilot directories
        must contain their matching analyzer receipts.
    source_commit, workflow_url
        Optional exact GPU-launch provenance, supplied by the orchestrator.
    execution_path
        Optional JSON with observed allocation times and cost estimates.

    Returns
    -------
    str
        Complete HTML with an embedded SVG chart and source-hash metadata.
    """
    retro = verified_report(retrospective_path)
    menus = read_json(menus_path)
    if menus.get("schema") != "imcqa-questionless-gold-analysis-v1" or menus.get("status") != "complete" or menus.get("rows") != 40000:
        raise ValueError("Incomplete or unexpected questionless score analysis")
    pilot = verified_report(pilot_path, pilot=True) if pilot_path else None
    execution = read_json(execution_path) if execution_path else None
    if source_commit and not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        raise ValueError("source-commit must be an exact lowercase commit SHA")
    if workflow_url and not re.fullmatch(re.escape(REPOSITORY) + r"/actions/runs/\d+", workflow_url):
        raise ValueError("workflow-url must identify this repository's exact GitHub Actions run")
    sources = {"retrospective report": (retrospective_path.name, sha256(retrospective_path)),
               "questionless report": (menus_path.name, sha256(menus_path))}
    if pilot_path:
        sources["pilot report"] = (pilot_path.name, sha256(pilot_path))
    if execution_path:
        sources["execution summary"] = (execution_path.name, sha256(execution_path))
    scope = table(["Evidence", "Unique questions", "Measurements", "Scope"], [
        ["Original generation archive", "5,000", "320,000 complete responses", "300,000 question-conditioned responses: 200,000 MC + 100,000 open-ended; 20,000 questionless responses"],
        ["Retrospective MC analysis", "5,000; test = 3,000", "200,000 MC responses", "2 models × 2 menus × 10 prefixes × 5,000 questions; reused responses"],
        ["Paired questionless score analysis", "5,000; test = 3,000", "40,000 FP32 score rows", "2 models × 2 menus × 2 instructions; no question text"],
        ["Explicit WAIT pilot", "200 development; test = 0", "9,600 planned score rows" if pilot is None else "9,600 validated score rows", "5 prefixes; both models and menus; questionless/rotation controls on 40 of the 200 questions"]
    ], "Rows are repeated measurements; they are not independent questions. These evidence sets overlap.")
    body = ("<header><p class='eyebrow'>IMCQA evidence report · 4 October 2026 UTC</p>"
            "<h1>Full-question scores conceal differences in early-answer value</h1>"
            "<p class='lede'>The existing experiment supports a substantial retrospective IMCQA analysis. "
            "On the same test questions, Qwen 7B reaches about 90% full-question accuracy with either menu, "
            "yet its selected early-answer reward differs materially between the two. An explicit "
            "WAIT pilot separately tests payoff-aware decisions.</p></header>"
            "<nav aria-label='Report sections'><a href='#scope'>Evidence</a><a href='#retrospective'>Retrospective</a>"
            "<a href='#pilot'>WAIT pilot</a><a href='#limits'>Interpretation</a><a href='#provenance'>Reproduce</a></nav>"
            "<section id='scope'><h2>Three analyses answer different questions</h2>" + scope + "</section>"
            + retrospective_section(retro) + risk_section(retro) + menu_section(menus)
            + pilot_section(pilot) + execution_section(execution)
            + "<section id='limits'><h2>What this supports, and what remains open</h2>"
              "<p>The data support cumulative-prefix decision analysis and sensitivity to menu construction. "
              "They do not establish Nishant's full proposal of reformulated same-answer chains across "
              "benchmarks. Word position is not an independently measured difficulty scale. A future "
              "changing-menu experiment needs a menu-history-only control because the persistent answer "
              "can be exposed by intersecting candidate sets.</p>"
              "<p>No IRT model was fitted, no broad model-ranking claim is supported by two related "
              "Qwen sizes, and the retrospective endpoint/reward comparison does not demonstrate a "
              "model-rank reversal. Semantic distractor validity and independently validated chain "
              "difficulty remain separate audit tasks. BenchMarker is an audit framework in the "
              "proposal, not a generator of question chains.</p>"
              "<p>The archived open-ended responses remain available, but unresolved semantic grading "
              "prevents using them here as a definitive open-ended-versus-MC comparison. Comparing the "
              "new action pilot with old generations also changes arithmetic precision, prompt, and "
              "output protocol. Reward, accuracy, and conditional error risk therefore retain their "
              "own definitions rather than being combined into one score.</p></section>"
              "<section id='provenance'><h2>Reproduction and evidence</h2>")
    links = []
    if source_commit:
        links.append(f"<a href='{REPOSITORY}/commit/{source_commit}'>Exact launch commit {source_commit[:12]}</a>")
        links.append(f"<a href='{REPOSITORY}/blob/{source_commit}/configs/imcqa_wait_pilot.json'>Frozen WAIT protocol</a>")
    if workflow_url:
        links.append(f"<a href='{esc(workflow_url)}'>Exact pilot workflow</a>")
    body += "<p>" + " · ".join(links) + "</p>" if links else "<p>Exact launch links were not supplied to this build.</p>"
    body += ("<details><summary>Input file hashes and rendering scope</summary>"
             + table(["Input", "File", "SHA256"], [[name, *values] for name, values in sources.items()], "Files used to build this report")
             + "<p>The renderer requires matching complete analyzer receipts for retrospective and "
               "pilot reports. It does not rerun inference, refit policies, or substitute missing "
               "outcomes. Complete audit vectors and source manifests belong to the accompanying "
               "results package.</p></details></section>")
    css = """
    :root{color-scheme:light;--ink:#182630;--muted:#53636e;--line:#dbe3e8;--accent:#176b91}
    *{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:#f6f8fa;color:var(--ink);font:16px/1.58 system-ui,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif}
    main{max-width:1160px;margin:36px auto;background:white;padding:48px 56px;box-shadow:0 2px 20px #1627370b}
    header{max-width:920px}.eyebrow{font-size:13px;letter-spacing:.09em;text-transform:uppercase;color:var(--accent);font-weight:650}
    h1{font-size:42px;line-height:1.16;letter-spacing:-.035em;margin:12px 0 20px}h2{font-size:25px;line-height:1.28;letter-spacing:-.018em;margin:0 0 16px}
    .lede{font-size:20px;line-height:1.5;color:#344955}p{margin:14px 0;max-width:980px}nav{display:flex;gap:24px;flex-wrap:wrap;border-top:1px solid var(--line);border-bottom:1px solid var(--line);padding:14px 0;margin:32px 0}
    a{color:var(--accent);text-underline-offset:3px}section{margin:36px 0 44px}.note,figcaption{color:var(--muted);font-size:14px;line-height:1.5}
    .table-wrap{overflow-x:auto;margin:20px 0}table{border-collapse:collapse;width:100%;font-size:14px;line-height:1.4}caption{text-align:left;padding:0 0 10px;font-size:13px;color:var(--muted)}
    th,td{text-align:left;vertical-align:top;border-bottom:1px solid var(--line);padding:11px 10px}th{background:#edf3f6;font-weight:650}td{font-variant-numeric:tabular-nums}tbody tr:nth-child(even){background:#fafcfd}
    #provenance td:last-child{font:11px/1.5 ui-monospace,SFMono-Regular,monospace;overflow-wrap:anywhere}figure{margin:24px 0}figure svg{display:block;width:100%;height:auto}figcaption{margin:8px 12px}
    .pending{background:#fff4d9;border-left:4px solid #b17a0e;padding:14px 18px}code{background:#edf2f5;padding:2px 4px;border-radius:3px}details{margin:20px 0}summary{cursor:pointer;font-weight:600;color:var(--accent)}
    footer{border-top:1px solid var(--line);padding-top:18px;color:var(--muted);font-size:13px}
    @media(max-width:750px){main{margin:0;padding:26px 18px}h1{font-size:32px}h2{font-size:23px}.lede{font-size:18px}nav{gap:14px}table{font-size:13px}th,td{padding:9px 7px}}
    @media print{body{background:white}main{max-width:none;margin:0;padding:0;box-shadow:none}nav{display:none}h1{font-size:30px}h2{font-size:20px}section{break-inside:auto}table,figure{break-inside:avoid}a{color:inherit}details{display:block}footer{font-size:11px}}
    """
    metadata = json.dumps({"input_sha256": {key: values[1] for key, values in sources.items()},
                           "source_commit": source_commit, "workflow_url": workflow_url,
                           "pilot_complete": pilot is not None}, sort_keys=True).replace("<", "\\u003c")
    return ("<!doctype html><html lang='en'><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width,initial-scale=1'>"
            "<title>IMCQA retrospective and explicit WAIT pilot · 2026-10-04</title>"
            f"<style>{css}</style></head><body><main>{body}"
            "<footer>Report date uses UTC. All scientific estimates are drawn from the supplied analysis files; "
            "no new scientific estimation occurs during rendering.</footer></main>"
            f"<script type='application/json' id='report-provenance'>{metadata}</script></body></html>")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrospective", type=Path, required=True, help="Report JSON or its directory")
    parser.add_argument("--menu-controls", type=Path, required=True)
    parser.add_argument("--pilot", type=Path, help="Complete report JSON or directory; omit for a clearly pending report")
    parser.add_argument("--out-html", "--outhtml", dest="out_html", type=Path, required=True)
    parser.add_argument("--source-commit", help="Exact pilot launch commit SHA, not the report-builder commit")
    parser.add_argument("--workflow-url", help="Exact pilot GitHub Actions run URL")
    parser.add_argument("--execution", type=Path, help="Optional actual allocation/cost summary JSON")
    args = parser.parse_args()
    rendered = build_report(report_path(args.retrospective), args.menu_controls,
                            report_path(args.pilot) if args.pilot else None,
                            source_commit=args.source_commit, workflow_url=args.workflow_url,
                            execution_path=args.execution)
    args.out_html.parent.mkdir(parents=True, exist_ok=True)
    args.out_html.write_text(rendered, encoding="utf-8")
    print(json.dumps({"status": "rendered", "path": str(args.out_html.resolve()),
                      "bytes": args.out_html.stat().st_size, "sha256": sha256(args.out_html),
                      "pilot_complete": args.pilot is not None}))


if __name__ == "__main__":
    main()
