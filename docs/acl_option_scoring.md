# ACL choices-only conditional option scoring

Authorized on 2026-10-03 as an additive diagnostic of the 5,000-question run.
There are 10,000 original menus per model, across independent-pool and
same-category-pool conditions. The original abstention responses remain intact.

The scoring context is the original rendered chat prompt followed by the fixed
assistant prefix `{"answer":"`. A single forward pass reads the next-token
logits for A, B, C and D; their softmax is conditional on this answer prefix and
on restricting the next token to those four labels. This is neither an
abstention probability nor calibrated correctness nor a replay of ordinary
forced-choice generation. Exact ties are retained and resolved A/B/C/D for the
primary top-1 statistic, with tie sensitivity in the analysis.

The original model revisions, file hashes, BF16 precision and eager attention
are preserved. Direct forward passes use attention-mask-derived position IDs
with left padding. CPU staging verifies every original model file and every
menu's token boundary before GPU work. GPU warmup compares three single and
batched score vectors with absolute tolerance 0.125 and identical top-option
sets; any failure stops without automatic fallback or rerun.

The first 256 original-order menus are the timing benchmark and stay in the
result. Continue only when 1.5 times the projected remaining runtime plus the
shutdown reserve fits the internal 1,650-second deadline. Each GPU function is
limited to 1,800 seconds, with a shared wall-clock limit and durable allocation
claims to prevent infrastructure replay from repeating scoring. CPU staging is
limited to 1,200 seconds. Two GPU allocations, startup/shutdown allowances,
CPU staging, and $0.40 contingency reserve total $2.87568048 at the verified
base allocation rates. This is a reserved compute estimate, not a verified
invoice or an account-level billing limit. No automatic retries or further
allocations are authorized by the runner.

Run from an exact committed checkout with the existing Modal workspace:

```bash
python -m scripts.modal_acl_option_scores --source-commit "$(git rev-parse HEAD)" --out results/acl_option_scores
```

The guarded GitHub workflow uses the existing Modal credentials. CPU-only
analysis runs after score collection, with the frozen evaluator data kept
outside all inference containers:

```bash
python scripts/analyze_acl_option_scores.py --public PUBLIC/main_choices_only.json --gold EVALUATOR/main_choices_only_gold.json --scores-root results/acl_option_scores/output --out results/acl_option_analysis
```

Analysis validates exact coverage and provenance. Report all-question and
held-out test results separately, including Wilson intervals, four-comparison
Holm-adjusted binomial references against uniform 25% guessing, answer-position
frequencies, a majority-position comparator and question-paired bootstrap
differences. Above-chance conditional choice accuracy motivates investigation
of menu construction; it does not establish exploitation during clue-bearing
answering. The selected 174 unresolved open-ended answer cases are untouched.
