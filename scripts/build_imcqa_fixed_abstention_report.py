"""Build an inspectable PDF and scientific figure from the saved-score analysis."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from xml.sax.saxutils import escape

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image, KeepTogether


MENUS = {"independent_pool": "Independent distractors", "same_category_pool": "Same-category distractors"}
POLICIES = {"fixed_forced": "Fixed, forced", "fixed_selective": "Fixed, abstention", "adaptive_forced": "Adaptive, forced", "adaptive_selective": "Adaptive, abstention"}
BLUE = "#235D9F"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads((args.analysis / "summary.json").read_text())
    audit = json.loads(args.audit.read_text())
    examples = json.loads(args.examples.read_text())["examples"]
    assert audit["status"] == "passed" and audit["main_comparison"]["status"] == "passed"
    args.out.mkdir(parents=True, exist_ok=True)
    rows = {(r["condition"], r["policy"]): r for r in summary["policy_summaries"]}
    primary = summary["primary_contrasts"]
    assert len(primary) == 2 and len(rows) == 8
    flat_rows = []
    for r in summary["policy_summaries"]:
        flat = {key:r[key] for key in ("condition","policy","n_questions","n_episodes","committed_count","correct_count","wrong_count","terminal_pass_count")}
        for metric in ("mean_reward","coverage","conditional_error","mean_observed_round"):
            flat[metric] = r[metric]["mean"]
            flat[metric+"_ci95_low"],flat[metric+"_ci95_high"] = r[metric]["ci95"]
        flat_rows.append(flat)
    table_path = args.out / "IMCQA_Fixed_Abstention_Table_2026-10-05.csv"
    with table_path.open("w",newline="") as stream:
        writer = csv.DictWriter(stream,fieldnames=list(flat_rows[0]))
        writer.writeheader(); writer.writerows(flat_rows)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                        "axes.spines.right": False, "axes.spines.left": False})
    fig, ax = plt.subplots(figsize=(7.1, 2.4), layout="constrained")
    for i, item in enumerate(primary):
        lo, hi = item["ci97_5"]
        point = item["mean_delta"]
        ax.errorbar(point, 1-i, xerr=[[point-lo], [hi-point]], fmt="o", color=BLUE,
                    capsize=5, markersize=7, linewidth=2)
        ax.annotate(f"{point:+.4f} [{lo:+.4f}, {hi:+.4f}]", (point, 1-i),
                    xytext=(0, 13), textcoords="offset points", ha="center", fontsize=9)
    ax.axvline(0, color="#444444", lw=1)
    ax.set_yticks([1, 0], [MENUS[x["condition"]].replace(" distractors", "") for x in primary])
    ax.set_ylim(-.45, 1.65)
    ax.grid(axis="x", color="#E7E7E7", linewidth=.7)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Reward difference: adaptive abstention minus fixed abstention")
    ax.set_title("Paired reward contrast | 97.5% intervals | 100 questions", loc="left", fontsize=10, pad=10)
    chart = args.out / "IMCQA_Fixed_Abstention_Comparison_2026-10-05.png"
    fig.savefig(chart, dpi=220, facecolor="white")
    plt.close(fig)

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="TitleCustom", fontName="Helvetica-Bold", fontSize=21, leading=25,
                              textColor=colors.HexColor("#153A5A"), spaceAfter=10))
    styles.add(ParagraphStyle(name="BodyCustom", fontName="Helvetica", fontSize=10, leading=14,
                              textColor=colors.HexColor("#263342"), spaceAfter=8))
    styles.add(ParagraphStyle(name="SmallCustom", fontName="Helvetica", fontSize=8.3, leading=11,
                              textColor=colors.HexColor("#4B5563"), spaceAfter=6))
    styles.add(ParagraphStyle(name="HeadCustom", fontName="Helvetica-Bold", fontSize=12, leading=16,
                              textColor=colors.HexColor("#153A5A"), spaceBefore=8, spaceAfter=7))
    def para(s, kind="BodyCustom"):
        return Paragraph(s, styles[kind])
    def table(data, widths, header=True):
        out = Table([[para(str(x), "SmallCustom") for x in row] for row in data], colWidths=widths, repeatRows=int(header))
        out.setStyle(TableStyle([("VALIGN", (0,0), (-1,-1), "TOP"), ("TOPPADDING",(0,0),(-1,-1),5),
                                ("BOTTOMPADDING",(0,0),(-1,-1),4), ("LINEBELOW",(0,0),(-1,0),.7,colors.HexColor("#9EAEBB")),
                                ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#EEF3F7")),
                                ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#F8FAFC")])]))
        return out
    story = [para("Fixed-round abstention changes<br/>the baseline comparison", "TitleCustom"),
             para("SAVED-SCORE ANALYSIS | 5 OCTOBER 2026 | EXPLORATORY DEVELOPMENT RESULTS", "SmallCustom")]
    positive = [x for x in primary if x["ci97_5"][0] > 0]
    if len(positive) == 2:
        finding = "Adaptive abstention retains a positive reward advantage over the added fixed-round abstaining baseline for both distractor menus."
    elif len(positive) == 1:
        finding = ("A positive reward advantage over fixed-round abstention is supported only for "
                   + MENUS[positive[0]["condition"]].lower() + " under the two-comparison interval adjustment.")
    else:
        finding = "The added comparison does not establish a positive adaptive reward advantage for either menu under the two-comparison interval adjustment."
    story += [para("<b>" + finding + "</b>"),
              para("Qwen2.5-7B-Instruct; 100 questions, two distractor menus, four cyclic option rotations and five cumulative prefixes. This addition uses the October 4 saved scores. There was no new model inference or parameter selection.")]
    data = [["Menu / policy", "Reward<br/>[95% interval]", "Coverage", "Error among<br/>answers"]]
    for menu, label in MENUS.items():
        for policy, plabel in POLICIES.items():
            r = rows[menu, policy]
            reward, ci = r["mean_reward"]["mean"], r["mean_reward"]["ci95"]
            data.append([("<b>" + label + "</b><br/>" if policy == "fixed_forced" else "") + plabel,
                         f"{reward:.4f}<br/>[{ci[0]:.4f}, {ci[1]:.4f}]",
                         f'{r["coverage"]["mean"]:.2%}<br/>{r["committed_count"]}/400',
                         f'{r["conditional_error"]["mean"]:.2%}<br/>{r["wrong_count"]}/{r["committed_count"]}'])
    story += [table(data, [185,133,86,112]), Spacer(1, 9), Image(str(chart), width=516, height=174),
              para("Figure: paired question-bootstrap 97.5% intervals, Bonferroni-adjusted for the two added reward contrasts. The adjustment covers this pair only. The 400 episodes per menu comprise 100 question clusters; rotations are not independent observations.", "SmallCustom"), PageBreak()]
    story += [para("What this comparison establishes", "TitleCustom"),
              para("The new baseline gives fixed-round answering the same calibrated confidence threshold and terminal abstention option as the existing adaptive policy. Both use exactly the same answer scores. It directly addresses the missing comparator in the October 4 analysis."),
              para("Frozen policy definitions", "HeadCustom")]
    data = [["Policy", "Answer rule", "Fallback"]]
    data += [["Fixed, forced", "Answer at the previously selected fixed round.", "Always answers there."],
             ["Fixed, abstention", "At that fixed round, answer iff calibrated confidence reaches the original threshold.", "No later answer opportunity; terminal PASS at round 5."],
             ["Adaptive, forced", "Answer at the first threshold crossing.", "If none, answer at round 5."],
             ["Adaptive, abstention", "Answer at the first threshold crossing.", "If none, terminal PASS at round 5."]]
    story += [table(data, [103,254,159]), Spacer(1,8),
              para("Independent distractors: fixed round 1, threshold 0.60. Same-category distractors: fixed round 2, threshold 0.85. Logistic coefficients, clipping, answer mapping and tie handling remain frozen. Passing yields 0; a wrong answer yields -1; a correct answer yields 1.0, 0.8, 0.6, 0.4 or 0.2 at rounds 1 through 5."),
              para("Uncertainty and verification", "HeadCustom"),
              para("We average the four rotations within each question, then draw 20,000 paired question bootstrap samples with seed 1. Conditional error is total wrong answers divided by total answers, recomputed in every sample. The full results include descriptive 95% intervals for coverage, conditional error, other simple effects and the interaction. Intervals condition on the previously fitted policies; they omit calibration and policy-selection uncertainty."),
              para("An independent replay reconstructs outcomes from raw scores and evaluator gold labels. The existing fixed-forced and adaptive-abstaining episode outcomes are checked against the prior outputs. Input hashes bind the analysis to the recovered evidence package. The audit JSON and per-episode CSVs are supplied for inspection."),
              para("Limits on the ARR takeaway", "HeadCustom"),
              para("The added comparison is exploratory because these development questions and the earlier results were already inspected. The old fixed round and threshold were not selected to optimize the new fixed abstaining policy. Adaptive answering also has more opportunities to cross the threshold, so its coverage can differ. These results do not isolate a causal timing effect at matched coverage, establish optimal stopping, or establish a reliable low error rate."),
              para("This is a component sensitivity result for one model and reward schedule. Outcomes come from constrained one-token scores. Cumulative word-fraction prefixes do not establish a validated difficulty ordering. A paper claim should match the menu-specific result. The interval adjustment covers only the new pair."),
              para("Reproduction", "HeadCustom"),
              para("Use the original IMCQA_Frozen_Transfer_2026-10-04.zip as the input. The companion evidence package contains the new analysis, independent audit, frozen plan identity, numerical tables and provenance. Code is maintained in ankaggarwal94/qanta-buzzer on branch feat/imcqa-fixed-abstention-20261005. See the package README for exact commands and commit.", "SmallCustom")]
    if examples:
        story += [PageBreak(), para("Checked trajectories", "TitleCustom"),
                  para("Illustrative examples selected after observing outcomes. These are mechanical checks against stored gold labels, not an independent semantic audit or evidence that a human has reviewed the question content.", "SmallCustom")]
        for e in examples:
            label = {"adaptive_benefit":"A later answer helps", "adaptive_harm":"A later answer hurts", "no_crossing":"No threshold crossing"}[e["category"]]
            block = [para(label, "HeadCustom"),
                     para(escape(MENUS[e["condition"]] + " | " + e["qid"] + " | rotation " + str(e["rotation"])), "SmallCustom")]
            if "options" in e:
                options = e["options"]
                if isinstance(options,dict):
                    options = [{"id":k,"text":v} for k,v in options.items()]
                block += [para("Canonical option IDs: " + escape("; ".join(str(x["id"])+": "+str(x["text"]) for x in options)) + ". Gold: " + escape(str(e["gold_canonical_id"])), "SmallCustom")]
            data = [["Round", "Canonical candidate", "Calibrated p", "Correct?"]]
            for r in e["rounds"]:
                data.append([r["round"],r["choice"],f'{r["calibrated_probability"]:.6f}',"Yes" if r["correct"] else "No"])
            block += [table(data,[58,170,145,143]),para(escape(e["why_selected"]),"SmallCustom")]
            story += [KeepTogether(block)]
    def footer(canvas, doc):
        canvas.setStrokeColor(colors.HexColor("#D7E0E8")); canvas.line(48,40,564,40)
        canvas.setFont("Helvetica",8); canvas.setFillColor(colors.HexColor("#667085"))
        canvas.drawString(48,27,"IMCQA | Saved-score sensitivity analysis | 2026-10-05")
        canvas.drawRightString(564,27,str(doc.page))
    pdf = args.out / "IMCQA_Fixed_Abstention_2026-10-05.pdf"
    SimpleDocTemplate(str(pdf), pagesize=(612,792), leftMargin=48,rightMargin=48,
                      topMargin=42,bottomMargin=52, title="IMCQA fixed-round abstention comparison",
                      author="IMCQA research analysis").build(story,onFirstPage=footer,onLaterPages=footer)
    print(json.dumps({"pdf":str(pdf),"figure":str(chart),"finding":finding}))


if __name__ == "__main__":
    main()
