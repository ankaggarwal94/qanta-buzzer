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


def render(summary: dict, provenance: dict) -> str:
    """Render only values supplied by the validated analyzer."""
    policies = summary["policy_summaries"]
    contrasts = summary["primary_contrasts"]
    rows = []
    for item in policies:
        reward = item["mean_reward"]
        mean = reward["mean"] if isinstance(reward, dict) else reward
        ci = reward.get("ci95") if isinstance(reward, dict) else None
        rows.append("<tr>" + "".join(f"<td>{esc(v)}</td>" for v in (
            item["condition"], item["policy"], number(mean),
            "" if ci is None else f"[{number(ci[0])}, {number(ci[1])}]",
            item["n_questions"])) + "</tr>")
    pairs = []
    for item in contrasts:
        ci = item["ci98_75"]
        pairs.append("<tr>" + "".join(f"<td>{esc(v)}</td>" for v in (
            item["condition"], item["right"], number(item["mean_delta"]),
            f"[{number(ci[0])}, {number(ci[1])}]")) + "</tr>")
    screen = esc(json.dumps(summary.get("screening", {}), indent=2))
    source = esc(provenance.get("source_commit", ""))
    run = esc(provenance.get("workflow_url", ""))
    cost = esc(json.dumps(provenance.get("compute", {}), indent=2))
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
<h1>Does the frozen stopping policy transfer?</h1>
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
fixed-round benchmarks answer at rounds 1 and 2, respectively. The threshold policy
has no added positive-expected-value gate; its historical behavior is preserved.</p>
<h2>Reward results</h2><table><thead><tr><th>Menu</th><th>Policy</th><th>Mean reward</th>
<th>95% interval</th><th>Questions</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<h2>Prespecified paired comparisons</h2><p>Each contrast subtracts the benchmark
from the frozen plain threshold policy. The intervals below are 98.75% question-bootstrap
intervals for a four-comparison Bonferroni family. They condition on the previously fitted
policy and do not incorporate uncertainty from its original 20-question calibration fit.</p>
<table><thead><tr><th>Menu</th><th>Benchmark</th><th>Reward difference</th>
<th>98.75% interval</th></tr></thead><tbody>{''.join(pairs)}</tbody></table>
<h2>Frozen development screening rule</h2>
<p>Continue development only when the frozen policy has positive reward relative
to PASS by its descriptive 95% interval and its family-adjusted paired interval against
native WAIT excludes zero in the favorable direction. Superiority to a fixed answering
round requires the separate paired contrast. These are development screening rules,
not a declaration of benchmark or optimal-stopping validity.</p><pre>{screen}</pre>
<h2>Scope and limitations</h2><ul>
<li>These questions were unused by the protocol pilots. The original corpus had already
been studied using other prompts and retrospective analysis. This is not a pristine
confirmatory test or an independent external dataset.</li>
<li>The run measures constrained next-token scores conditional on a fixed answer prefix,
not free generated responses or repeated-response frequencies.</li>
<li>Results concern one model and two frozen menu constructions. Four cyclic rotations
do not cover every permutation. Reveals use word fractions, not validated clue boundaries.</li>
<li>Deterministic identity and text-overlap exclusions reduce obvious duplication;
they do not establish semantic independence or rule out training-data contamination.</li>
<li>No new controller prompt, calibration fit, threshold search, or reward redesign was
selected using these results. The external threshold policy does not estimate continuation value.</li>
</ul><h2>Execution and evidence</h2><p>Source commit: <code>{source}</code><br>
Workflow: <a href="{run}">{run}</a>. The companion package contains scores, frozen inputs,
policy parameters, numerical checks, per-question outcomes, independent audit, and CPU
reproduction commands.</p><pre>{cost}</pre>
<p><small>Cost figures are estimates, not verified invoices. Resource rates were checked
on 2026-10-04 at <a href="https://modal.com/pricing">Modal pricing</a>.
Question-bootstrap results are conditional on the frozen policy and this sample.</small></p>
</main></html>"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--provenance", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    text = render(json.loads(args.summary.read_text()), json.loads(args.provenance.read_text()))
    args.out.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
