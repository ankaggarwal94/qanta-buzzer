"""Render the crossed-setting sensitivity table and its complete policy results."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Spacer, PageBreak

MENUS={"independent_pool":"Independent","same_category_pool":"Same category"}
POLICIES=("fixed_forced","fixed_selective","adaptive_forced","adaptive_selective")

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--analysis",required=True,type=Path)
    p.add_argument("--audit",required=True,type=Path)
    p.add_argument("--out",required=True,type=Path)
    a=p.parse_args()
    summary=json.loads((a.analysis/'summary.json').read_text())
    audit=json.loads(a.audit.read_text())
    if audit['status']!='passed':raise ValueError('independent audit has not passed')
    a.out.mkdir(parents=True,exist_ok=True)
    rows=summary['policy_summaries']
    lookup={(r['setting_id'],r['condition'],r['policy']):r for r in rows}
    settings=sorted({r['setting_id'] for r in rows})
    assert len(rows)==32 and len(settings)==4
    flat=[]
    for r in rows:
        v={k:r[k] for k in ('setting_id','threshold','fixed_round','condition','policy','n_questions','n_episodes','committed_count','correct_count','wrong_count','terminal_pass_count')}
        for metric in ('mean_reward','coverage','conditional_error','mean_observed_round'):
            v[metric]=r[metric]['mean']
            v[metric+'_ci95_low'],v[metric+'_ci95_high']=r[metric]['ci95']
        flat.append(v)
    with (a.out/'IMCQA_Crossed_Settings_Table_2026-10-05.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name='T',fontName='Helvetica-Bold',fontSize=20,leading=24,textColor=colors.HexColor('#163B58'),spaceAfter=10))
    styles.add(ParagraphStyle(name='B',fontName='Helvetica',fontSize=10,leading=13,spaceAfter=8))
    styles.add(ParagraphStyle(name='S',fontName='Helvetica',fontSize=8.5,leading=11,spaceAfter=4))
    styles.add(ParagraphStyle(name='H',fontName='Helvetica-Bold',fontSize=12,leading=15,spaceBefore=9,spaceAfter=6,textColor=colors.HexColor('#163B58')))
    def P(s,k='B'):return Paragraph(str(s),styles[k])
    def T(data,widths):
        table=Table([[P(v,'S') for v in row] for row in data],colWidths=widths,repeatRows=1)
        table.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5),('BACKGROUND',(0,0),(-1,0),colors.HexColor('#EAF1F6')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#F5F8FA')]),('LINEBELOW',(0,0),(-1,0),.5,colors.HexColor('#8CA3B4'))]))
        return table
    def label(s):
        r=lookup[s,'independent_pool','fixed_forced']
        return f"{r['threshold']:.2f} / {r['fixed_round']}"
    primary={(r['setting_id'],r['condition']):r for r in summary['primary_contrasts']}
    story=[P('Crossed thresholds and fixed rounds','T'),P('5 OCTOBER 2026 | SAVED-SCORE SENSITIVITY ANALYSIS','S'),
        P('Both menus were evaluated at thresholds 0.60 and 0.85 and fixed rounds 1 and 2. This includes both requested swaps and the two mixed settings. The original per-menu confidence calibrators, answer scores, rewards and 100-question cohort remain unchanged. No fitting or new inference occurred.'),
        P('<b>The original menu-specific interpretation is sensitive to policy settings.</b> At threshold 0.85 / fixed round 2, both menus show a positive adaptive-selective reward gain under the eight-comparison adjustment. At threshold 0.60 / round 1, neither does. The paired between-menu differences in gains at those two settings remain uncertain.'),
        P('Adaptive versus fixed answering, both with abstention','H')]
    data=[['Menu','Threshold /<br/>fixed round','Fixed + PASS<br/>reward','Adaptive + PASS<br/>reward','Difference<br/>[99.375% interval]']]
    for s in settings:
        for menu in MENUS:
            fs=lookup[s,menu,'fixed_selective'];ad=lookup[s,menu,'adaptive_selective'];c=primary[s,menu]
            data.append([MENUS[menu],label(s),f"{fs['mean_reward']['mean']:.4f}",f"{ad['mean_reward']['mean']:.4f}",f"{c['mean_delta']:+.4f}<br/>[{c['ci99_375'][0]:+.4f}, {c['ci99_375'][1]:+.4f}]"])
    story += [T(data,[95,78,91,100,152]),Spacer(1,7),P('Differences are adaptive selective minus fixed selective. The 99.375% question-bootstrap intervals apply a Bonferroni adjustment to the eight specified contrasts in this added analysis only. They do not adjust the complete history of analyses on these development questions. Descriptive 95% intervals are retained in the numerical results.','S'),
        P('Does the adaptive gain differ between menus at shared settings?','H')]
    data=[['Threshold / fixed round','Difference in gains','Descriptive 95% interval']]
    for r in summary['menu_gain_differences']:
        data.append([label(r['setting_id']),f"{r['mean_delta']:+.4f}",f"[{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}]"])
    story += [T(data,[180,155,181]),P('Difference in gains = (adaptive - fixed) with same-category distractors minus (adaptive - fixed) with independent distractors, with both policies permitting abstention. Each comparison keeps numerical threshold and fixed round equal across menus. These four intervals are descriptive and unadjusted.','S'),
        PageBreak(),P('Complete policy comparison','T')]
    data=[['Menu','Threshold /<br/>fixed round','Fixed<br/>forced','Fixed<br/>+ PASS','Adaptive<br/>forced','Adaptive<br/>+ PASS']]
    for s in settings:
        for menu in MENUS:
            data.append([MENUS[menu],label(s)]+[f"{lookup[s,menu,pol]['mean_reward']['mean']:.4f}" for pol in POLICIES])
    story += [P('Entries below are mean rewards. Fixed forced answers at the specified round. Fixed + PASS has one threshold-gated opportunity at that round. Adaptive policies answer at the first threshold crossing; if none occurs, adaptive forced answers at round 5 while adaptive + PASS ends with PASS.'),T(data,[104,84,82,82,82,82]),P('Coverage and error for the two abstaining policies','H')]
    data=[['Menu','Threshold /<br/>fixed round','Fixed<br/>coverage','Fixed error<br/>among answers','Adaptive<br/>coverage','Adaptive error<br/>among answers']]
    for s in settings:
        for menu in MENUS:
            fs=lookup[s,menu,'fixed_selective'];ad=lookup[s,menu,'adaptive_selective']
            data.append([MENUS[menu],label(s),f"{fs['coverage']['mean']:.2%}",f"{fs['conditional_error']['mean']:.2%}",f"{ad['coverage']['mean']:.2%}",f"{ad['conditional_error']['mean']:.2%}"])
    story += [T(data,[104,84,82,82,82,82]),P('Method and validation','H'),P('Each cell contains 100 questions and four cyclic option rotations per question. Rotations are averaged within question before 20,000 paired bootstrap resamples (seed 1). All settings, menus and policies use the same resampling indices. Conditional error is wrong answers divided by committed answers, recomputed in each resample.'),
        P('The independent raw-score replay verifies the original results and all comparisons. Repeated adaptive cells across fixed rounds, and fixed-forced cells across thresholds, do not add independent evidence. Intervals condition on the original fitted calibrators and omit their fitting uncertainty.'),
        P('Shared numerical settings retain different calibration maps and do not match coverage or isolate a pure menu effect. These development results select no validated winner. Fixed selective has not been optimized for its own objective.')]
    def footer(c,d):
        c.setFont('Helvetica',8);c.setFillColor(colors.HexColor('#617381'))
        c.drawString(48,27,'IMCQA | Crossed settings | Exploratory development analysis')
        c.drawRightString(564,27,str(d.page))
    out=a.out/'IMCQA_Crossed_Settings_2026-10-05.pdf'
    SimpleDocTemplate(str(out),pagesize=(612,792),leftMargin=48,rightMargin=48,topMargin=42,bottomMargin=50,title='IMCQA crossed policy settings').build(story,onFirstPage=footer,onLaterPages=footer)
    print(str(out))

if __name__=='__main__':main()
