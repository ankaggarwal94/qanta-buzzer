"""Render the frozen-policy transfer results without fitting or inference."""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


def esc(value):
    return html.escape(str(value))


def number(value):
    return "undefined" if value is None else f"{value:.4f}"


def render(summary: dict, provenance: dict, covariate_audit: dict | None = None) -> str:
    """Render only values supplied by the validated analyzer."""
    policies = summary["policy_summaries"]
    contrasts = summary["primary_contrasts"]
    screens = summary.get("screening", [])
    labels = {"independent_pool": "independent distractors", "same_category_pool": "same-category distractors"}
    policy_labels = {"frozen_plain_threshold": "Frozen threshold", "native_wait": "Native WAIT",
                     "frozen_plain_fixed": "Fixed round", "always_pass": "Always pass",
                     "plain_final_round": "Final round"}
    passed = [labels[s["condition"]] for s in screens if s["continue_development_screen"]]
    fixed = [labels[s["condition"]] for s in screens if s["beats_fixed_family_interval"]]
    if len(passed) == 2:
        finding = "The frozen policy passed the development screen with both menus: positive reward and improvement over native WAIT."
    elif passed:
        finding = "The frozen policy passed the development screen only with " + passed[0] + "."
    else:
        finding = "The frozen policy did not pass the prespecified development screen with either menu."
    finding += (" Its paired reward also exceeded the frozen fixed-round baseline with " + " and ".join(fixed) + "."
                if fixed else " Added value over the frozen fixed-round baselines remains unestablished.")
    rows = []
    for item in policies:
        reward = item["mean_reward"]
        mean = reward["mean"] if isinstance(reward, dict) else reward
        ci = reward.get("ci95") if isinstance(reward, dict) else None
        rows.append("<tr>" + "".join(f"<td>{esc(v)}</td>" for v in (
            labels[item["condition"]], policy_labels[item["policy"]], number(mean),
            "" if ci is None else f"[{number(ci[0])}, {number(ci[1])}]",
            number(item["coverage"]["mean"]), number(item["risk"]["mean"]),
            item["n_questions"])) + "</tr>")
    pairs = []
    for item in contrasts:
        ci = item["ci98_75"]
        pairs.append("<tr>" + "".join(f"<td>{esc(v)}</td>" for v in (
            labels[item["condition"]], policy_labels[item["right"]], number(item["mean_delta"]),
            f"[{number(ci[0])}, {number(ci[1])}]")) + "</tr>")
    boundary_note = ""
    independent_fixed = next((item for item in contrasts if item["condition"] == "independent_pool"
                              and item["right"] == "frozen_plain_fixed"), None)
    if independent_fixed is not None and 0 < independent_fixed["ci98_75"][0] < .01:
        lo, hi = independent_fixed["ci98_75"]
        boundary_note = ("<p>The independent-menu gain over fixed-round answering is "
                         f"{number(independent_fixed['mean_delta'])}, with a 98.75% interval of "
                         f"[{number(lo)}, {number(hi)}]. Its lower bound is close to zero; "
                         "the size and robustness of that gain remain unsettled beyond this transfer sample.</p>")
    screen = esc(json.dumps(summary.get("screening", {}), indent=2))
    source = esc(provenance.get("source_commit", ""))
    analysis_source = esc(provenance.get("analysis_commit", ""))
    run = esc(provenance.get("workflow_url", ""))
    cost = esc(json.dumps(provenance.get("compute", {}), indent=2))
    risk_by_menu = {item["condition"]: item["risk"]["mean"] for item in policies
                    if item["policy"] == "frozen_plain_threshold"}
    risk_note = ""
    if all(risk_by_menu.get(menu) is not None for menu in labels):
        risk_note = ("<p>Among the frozen policy's committed answers, "
                     f"{risk_by_menu['independent_pool']:.2%} were wrong with independent distractors and "
                     f"{risk_by_menu['same_category_pool']:.2%} with same-category distractors. "
                     "The reward gains therefore do not establish low-risk answering. "
                     "Comparison with native WAIT changes both answer elicitation and stopping; "
                     "it does not isolate the contribution of stopping alone.</p>")
    compute = provenance.get("compute", {})
    cost_sentence = ""
    if all(k in compute for k in ("actual_elapsed_seconds", "estimated_resource_usd_including_allowances", "authorized_bound_usd")):
        seconds = round(float(compute["actual_elapsed_seconds"]))
        cost_sentence = (f"<p>Timed worker execution took {seconds//60} minutes {seconds%60} seconds. "
                         f"Estimated resource cost including allowances was ${float(compute['estimated_resource_usd_including_allowances']):.2f}, "
                         f"within the ${float(compute['authorized_bound_usd']):.2f} bound. The invoice remains unverified.</p>")
    cohort_note, input_checks = "", ""
    if covariate_audit is not None:
        audit = covariate_audit
        options = audit["option_audit"]
        transfer = audit["cohorts"]["transfer_100"]
        if (audit.get("schema_version") != "imcqa-transfer-covariate-audit-v1"
                or audit.get("status") != "complete" or audit.get("cohorts_disjoint") is not True
                or audit.get("new_scores_read") is not False or audit.get("model_inference_calls") != 0
                or audit["source_sha256"]["transfer_public"] != "4172b83e16f753f51442ff314d60ac2c1d75b12b7faa6574d5918319936ad2b9"
                or transfer["n_questions"] != 100 or sum(transfer["category_counts"].values()) != 100
                or options["n_score_contexts"] != 8000 or options["n_unique_question_menus"] != 200
                or options["empty_or_duplicate_or_mapping_issues"] or options["unbalanced_strata"]
                or options["displayed_gold_counts"] != dict.fromkeys("ABCD", 2000)):
            raise ValueError("input-only covariate audit failed or differs from this study")
        categories = transfer["category_counts"]
        if categories.get("Religion", 0) == 0:
            cohort_note = ("<li>The sample follows a fixed hash ranking, without category quotas. "
                           "No selected question has the source category Religion, so these results "
                           "provide no separate estimate for that category.</li>")
        cohort_rows = []
        for name, label in (("old_calibration_20", "Earlier calibration"),
                            ("old_selection_20", "Earlier selection"), ("transfer_100", "This transfer")):
            cohort = audit["cohorts"][name]
            cohort_rows.append("<tr>" + "".join(f"<td>{esc(v)}</td>" for v in (
                label, cohort["n_questions"], f'{cohort["word_count"]["mean"]:.2f}',
                f'{cohort["word_count"]["median"]:.1f}')) + "</tr>")
        input_checks = f"""<details><summary>Input-only checks and sample composition</summary>
<p>The recorded option audit found no empty options, normalized duplicate options, or
mapping errors. The four rotations put the gold answer equally often at A, B, C, and D
within each matched condition. These are structural checks; they do not establish
semantic answer uniqueness or distractor plausibility. This audit read no new model scores.</p>
<table><thead><tr><th>Cohort</th><th>Questions</th><th>Mean words</th><th>Median words</th>
</tr></thead><tbody>{''.join(cohort_rows)}</tbody></table>
<p>Transfer categories: {esc('; '.join(f'{k}: {v}' for k,v in sorted(categories.items())))}.
Source categories and question length describe the sample; they do not validate difficulty ordering.</p></details>"""
    return f"""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>IMCQA: frozen-policy transfer</title><style>
body{{font:17px/1.6 system-ui,sans-serif;color:#173047;background:#f4f7fa;margin:0}}
main{{max-width:1080px;margin:36px auto;padding:32px;background:white;border-radius:14px}}
h1{{font-size:32px;line-height:1.2}} h2{{margin-top:32px;color:#145573}}
.tag{{font-size:13px;font-weight:700;letter-spacing:.06em;color:#176b68}}
.note{{background:#edf5f8;padding:18px;border-left:4px solid #26758b}}
table{{border-collapse:collapse;width:100%;font-size:14px;display:block;overflow-x:auto}}
th,td{{text-align:left;padding:9px 12px;border-bottom:1px solid #dbe4eb}}
th{{background:#e9f0f6}} code,pre{{font-size:12px;overflow-wrap:anywhere;white-space:pre-wrap}}
small{{color:#526575}} a{{color:#145573}}
@media print{{body{{background:white}}main{{margin:0;padding:10px}}}}
</style><main><div class="tag">DEVELOPMENT VALIDATION · 2026-10-04 UTC</div>
<h1>Frozen-policy transfer test</h1><p><strong>{esc(finding)}</strong></p>
<p class="note">A paired test of the existing Qwen 7B plain-MCQA answer and external
stopping pipeline on 100 questions excluded from all previous WAIT and protocol pilots.
The prompts, fitted coefficients, thresholds, rewards, and comparisons were frozen
before this run. No parameters were refitted on these outcomes.</p>
<h2>What was tested</h2>
<p>100 selection questions × two distractor menus × five cumulative word-fraction
reveals × four cyclic answer orders × two prompts = <strong>8,000 scored contexts</strong>.
The independent unit is the question, not a prefix, order, menu, or score row.
The model uses pinned weights with FP32 arithmetic and the same fixed answer prefix.
Native WAIT uses the existing game prompt and E for WAIT/PASS.</p>
<p>Correct answers receive +1.0, +0.8, +0.6, +0.4, or +0.2 across the five rounds;
wrong answers receive −1 and terminal PASS receives zero. The frozen plain thresholds
are 0.60 for independent distractors and 0.85 for same-category distractors. The
thresholds apply to the frozen calibrator's estimate of candidate correctness, computed
from the largest conditional A–D probability. The policy submits the first candidate
whose estimate reaches its threshold, or passes at the final round if none does. The
fixed-round benchmarks answer at rounds 1 and 2, respectively. The threshold policy
has no added positive-expected-value gate; its historical behavior is preserved.</p>
{input_checks}
<h2>Reward results</h2><table><thead><tr><th>Menu</th><th>Policy</th><th>Mean reward</th>
<th>95% interval</th><th>Coverage</th><th>Error among answers</th><th>Questions</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
{risk_note}
<h2>Prespecified paired comparisons</h2><p>Each contrast subtracts the benchmark
from the frozen plain threshold policy. The intervals below are 98.75% question-bootstrap
intervals for a four-comparison Bonferroni family. They condition on the previously fitted
policy and do not incorporate uncertainty from its original 20-question calibration fit.</p>
<table><thead><tr><th>Menu</th><th>Benchmark</th><th>Reward difference</th>
<th>98.75% interval</th></tr></thead><tbody>{''.join(pairs)}</tbody></table>
{boundary_note}
<h2>Frozen development screening rule</h2>
<p>Continue development only when the frozen policy has positive reward relative
to PASS by its descriptive 95% interval and its family-adjusted paired interval against
native WAIT excludes zero in the favorable direction. Superiority to a fixed answering
round requires the separate paired contrast. These are development screening rules,
not a declaration of benchmark or optimal-stopping validity.</p>
<details><summary>Machine-readable screening results</summary><pre>{screen}</pre></details>
<h2>Scope and limitations</h2><ul>
<li>These questions were unused by the protocol pilots. The original corpus had already
been studied using other prompts and retrospective analysis. This is not a pristine
confirmatory test or an independent external dataset.</li>
<li>The run measures constrained next-token scores conditional on a fixed answer prefix,
not free generated responses or repeated-response frequencies.</li>
<li>Results concern one model and two frozen menu constructions. Four cyclic rotations
do not cover every permutation. Reveals use word fractions, not validated clue boundaries.</li>
{cohort_note}
<li>Deterministic identity and text-overlap exclusions reduce obvious duplication;
they do not establish semantic independence or rule out training-data contamination.</li>
<li>Answer keys and distractor menus are inherited from the frozen source dataset.
This run checks their identity and use; it does not add a manual semantic audit of them.</li>
<li>No new controller prompt, calibration fit, threshold search, or reward redesign was
selected using these results. The external threshold policy does not estimate continuation value.</li>
<li>The fixed-round comparison changes both when the policy answers and whether it
answers at all. A gain does not isolate the contribution of timing from selective abstention.</li>
</ul><h2>Execution and evidence</h2><p>Inference source commit: <code>{source}</code><br>
Analysis source commit: <code>{analysis_source}</code><br>
Workflow: <a href="{run}">{run}</a>. The companion package contains scores, frozen inputs,
policy parameters, numerical checks, per-question outcomes, independent audit, and CPU
reproduction commands.</p>
{cost_sentence}
<details><summary>Timing and estimated compute ledger</summary><pre>{cost}</pre></details>
<p><small>Cost figures are estimates, not verified invoices. Resource rates were checked
on 2026-10-04 at <a href="https://modal.com/pricing">Modal pricing</a>.
Question-bootstrap results are conditional on the frozen policy and this sample.</small></p>
</main></html>"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--provenance", required=True, type=Path)
    parser.add_argument("--covariate-audit", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    covariates = json.loads(args.covariate_audit.read_text()) if args.covariate_audit else None
    text = render(json.loads(args.summary.read_text()), json.loads(args.provenance.read_text()), covariates)
    args.out.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
