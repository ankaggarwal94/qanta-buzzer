# Crossed threshold and fixed-round sensitivity

The user's requested swaps apply threshold 0.85 / fixed round 2 to independent
distractors and threshold 0.60 / fixed round 1 to same-category distractors.
This analysis includes those pairs and completes the 2 x 2 threshold/round grid
within both menus: thresholds {0.60, 0.85}, fixed rounds {1, 2}.

The analysis plan was fixed before inspecting these added outcomes at
2026-10-05T23:40:48Z. Its SHA256 is
`d678b4bc4a28bb9570d77e1ad79f388d5ad7c7b9080a2b3d1b801a8b5a23ebd1`.
The cohort and prior policies had already been inspected. This is an
exploratory sensitivity analysis, not a preregistered or fresh-data evaluation.

Each menu retains its own original calibration function and clipping bounds.
Only the numerical threshold and fixed round are crossed. Answer scores, gold
labels, option mapping, reward and question membership do not change. Four
policies are replayed as in the preceding fixed-abstention analysis:

- Fixed forced: answer at the specified fixed round.
- Fixed selective: answer at that round only if calibrated confidence reaches
  the specified threshold, otherwise no later answer and terminal PASS.
- Adaptive forced: answer at first threshold crossing, otherwise at round 5.
- Adaptive selective: answer at first threshold crossing, otherwise PASS.

There are 32 menu-setting-policy cells and 12,800 episode-cell records. Each
cell still contains only 100 questions and four rotations per question.
Adaptive policies repeat across the irrelevant fixed-round dimension;
fixed-forced repeats across the irrelevant threshold dimension. Neither those
repeated cells nor rotations add independent question observations.

Mean reward, coverage, conditional error and observed round use 20,000 paired
question bootstrap samples, seed 1, after averaging rotations within question.
All menus/settings/policies share resampling indices. Conditional error is a
pooled wrong/committed ratio, recomputed for each resample.

The eight adaptive-selective minus fixed-selective reward contrasts receive
95% descriptive intervals and 99.375% intervals adjusted for this family of
eight (Bonferroni). The four matched-setting cross-menu differences in those
gains and direct cross-menu policy differences have descriptive 95% intervals.
The adjustment covers this analysis only, not the history of analyses on this
development set. Calibration fitting uncertainty is omitted.

Shared numerical settings do not imply a shared calibration map, matched
coverage, or matched error rate. These results cannot identify a pure causal
effect of distractor type or select a validated optimum. They also do not
replace independent tuning of the fixed-selective policy on calibration data.

## Observed reward differences

Adaptive selective minus fixed selective, using the same numerical setting
within each row:

| Threshold / fixed round | Independent | Same category |
| --- | ---: | ---: |
| 0.60 / 1 | -0.0040 | +0.0070 |
| 0.60 / 2 | +0.1190 | +0.0445 |
| 0.85 / 1 | +0.0735 | +0.0635 |
| 0.85 / 2 | +0.1140 | +0.0810 |

Under the family-eight adjustment, both menus have positive intervals at
0.85 / round 2; neither does at 0.60 / round 1. Thus the earlier pattern of a
clear adaptive gain only in the same-category condition depends on the
original menu-specific settings. The paired same-category-minus-independent
differences in gains at these two bundles are +0.0110 (descriptive 95% interval
[-0.0455, +0.0685]) and -0.0330 ([-0.0940, +0.0285]), respectively. Those
intervals do not establish either a difference or equivalence of menu effects.
Full uncertainty, absolute rewards, coverage and conditional error are retained
in the report and numerical outputs.

## Reproduction

Use the original `IMCQA_Frozen_Transfer_2026-10-04.zip` and its complete extracted
contents, plus the preceding `fixed_abstention/analysis` outputs. Exact commands
and the code commit are recorded in the delivered evidence package README.
The analyzer validates source identities, original episode reproduction, and
invariance across irrelevant setting dimensions. A separately implemented
raw-score replay audits the full comparison. No model inference is required.
