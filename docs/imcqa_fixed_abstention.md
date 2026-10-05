# Fixed-round abstention sensitivity analysis

This CPU-only addition replays the saved October 4 transfer scores. The question
cohort and original results had already been inspected. Its new policy contract
was frozen on October 5, 2026 at 22:52:18 UTC before computing the added outcomes,
in `configs/imcqa_fixed_abstention.json` (SHA256
`4a455bb6bc3d46eb7b2a547130b69008d67ed41f3391e97f50dfd3e87ae7d195`).
This chronology does not make the addition preregistered or confirmatory.

All four policies use the same saved plain-MCQA scores, original logistic
calibration, clipping, threshold, and answer tie handling:

| Policy | Answer opportunity | If the threshold is never met |
| --- | --- | --- |
| Fixed forced | Original fixed round, regardless of confidence | Answer at that round |
| Fixed selective | Original fixed round, only if confidence meets threshold | No later answer; terminal PASS |
| Adaptive forced | First threshold crossing | Answer at round 5 |
| Adaptive selective | First threshold crossing | Terminal PASS |

Fixed selective has exactly one answer opportunity. A failure at that round is
represented as continuing without an answer to terminal PASS, since early PASS
is not an action in the original protocol. Forced fallback is an offline policy
change, not a new prompt or inference run. The original fixed round is 1 with
independent distractors and 2 with same-category distractors; the original
thresholds are 0.60 and 0.85 respectively.

The principal new comparisons are adaptive selective minus fixed selective,
separately for each menu. There are 100 question clusters, each containing four
cyclic option rotations. Average rotations within question; bootstrap all
policies and menus with the same 20,000 question resamples (seed 1). Report 95%
descriptive percentile intervals and 97.5% intervals for each of the two main
reward contrasts (Bonferroni family size 2). The adjustment covers this added
pair only, not every analysis performed on the development data.

Coverage is the fraction of episodes that answer. Conditional error is wrong
answers divided by committed answers, with the ratio recomputed within each
bootstrap sample. The 400 rotations per menu are not 400 independent questions.
Other simple effects and the interaction are descriptive.

Interpretation remains conditional on one model, these fixed menus, the chosen
linear reward schedule, and fitted policies. The frozen threshold and fixed
round were not optimized for the new fixed-selective policy. Adaptive policies
have multiple opportunities and may cover different questions; this comparison
does not identify a causal effect of timing at matched coverage or establish
optimal stopping. Structural and mechanical checks do not establish semantic
answer uniqueness, empirical difficulty ordering, or personal review by a human.

## Observed result

| Menu | Fixed forced | Fixed selective | Adaptive forced | Adaptive selective |
| --- | ---: | ---: | ---: | ---: |
| Independent | 0.3600 | 0.4375 | 0.4240 | 0.4335 |
| Same category | 0.1025 | 0.1825 | 0.2020 | 0.2635 |

Entries are mean rewards. Adaptive selective minus fixed selective is -0.0040
with a 97.5% paired interval [-0.0580, 0.0470] for independent distractors, and
+0.0810 [0.0225, 0.1360] for same-category distractors. This fails to establish
an added adaptive reward gain for independent distractors; it does not establish
equivalence. The same-category gain remains positive under the stated adjustment.

Fixed-selective coverage is 69.75% and 40.25%, versus adaptive-selective 98.75%
and 82.75%, respectively. Conditional error is 18.64% and 19.25% for fixed
selective, versus 25.06% and 21.75% for adaptive selective. These different
answering sets must remain visible when interpreting reward.

## CPU reproduction

Extract the original October 4 evidence archive into a new directory. From the
repository checkout used by the delivered provenance, run the following with
absolute paths substituted. The output directory must not already exist.

```bash
python3 -m scripts.analyze_imcqa_fixed_abstention \
  --evidence /path/to/original/extracted/evidence \
  --archive /path/to/IMCQA_Frozen_Transfer_2026-10-04.zip \
  --plan configs/imcqa_fixed_abstention.json \
  --out /path/to/new/analysis

python3 scripts/audit_imcqa_fixed_abstention.py \
  --root /path/to/original/extracted/evidence \
  --plan configs/imcqa_fixed_abstention.json \
  --analysis /path/to/new/analysis \
  --out /path/to/new/independent_audit.json \
  --examples-out /path/to/new/checked_trajectories.json

python3 -m scripts.build_imcqa_fixed_abstention_report \
  --analysis /path/to/new/analysis \
  --audit /path/to/new/independent_audit.json \
  --examples /path/to/new/checked_trajectories.json \
  --out /path/to/new/report
```

The numerical analysis requires Python and NumPy (executed with Python 3.12.14
and NumPy 2.3.5; consult delivered provenance for the exact runtime). The report
additionally uses Matplotlib and ReportLab. No GPU or model download is needed.
