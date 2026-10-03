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
menu's token boundary before GPU work. The initial BF16 warmup gate failed
before any production rows. Its receipts are preserved. The reviewed recovery
records BF16 batch/single sensitivity, then temporarily converts the same
weights to FP32 to check three batched versus unpadded single score vectors
at `atol=1e-3, rtol=1e-5`. Raw vectors are saved before validation. This checks
padding/indexing without demanding exact argmax equality near numerical ties.
Each parameter and buffer is restored to its individual original dtype,
including originally FP32 RoPE buffers, and sampled values are checked exactly
before BF16 production. The diagnostic is recorded as validation method v2;
the scoring estimand and production batch shape are unchanged.

The first 256 original-order menus are the timing benchmark and stay in the
result. Continue only when 1.5 times the projected remaining runtime plus the
shutdown reserve fits the internal 1,650-second deadline. Each GPU function is
limited to 1,800 seconds, with a shared wall-clock limit and durable allocation
claims to prevent infrastructure replay from repeating scoring. CPU staging is
limited to 1,200 seconds. Two GPU allocations, startup/shutdown allowances,
CPU staging, and $0.40 contingency reserve total $2.87568048 at the verified
base allocation rates. This is a reserved compute estimate, not a verified
invoice or an account-level billing limit. No automatic retries are permitted.

The one reviewed recovery uses cached inputs and weights without CPU staging.
It binds the five completed initial receipts by SHA256 and refuses any existing
recovery claim. Two GPU functions have 1,500-second timeouts, 1,300-second
internal deadlines, and 90-second startup plus 2-second scaledown allowances.
The initial measured allocations plus their startup/scaledown allowances,
the new maximum allocations, and $0.40 contingency total $2.6006284568.
Initial failed results remain separate from `output/recovery1`.

Run from an exact committed checkout with the existing Modal workspace:

```bash
python -m scripts.modal_acl_option_scores --source-commit "$(git rev-parse HEAD)" --out results/acl_option_scores
```

The explicitly reviewed recovery adds `--recover-once`; it is not a generic
retry facility. Its analyzer uses `--scores-root results/acl_option_scores/output/recovery1`.

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
differences. Recovered results additionally require independent validation of
retained FP32 and BF16 warmup vectors and the original dtype restoration
evidence. `per_menu_scores.csv` joins every score to the exact public option
texts, displaying the preferred answer, gold answer, correctness, ties, all four
probabilities and logits, and margins. Above-chance conditional choice accuracy motivates investigation
of menu construction; it does not establish exploitation during clue-bearing
answering. The selected 174 unresolved open-ended answer cases are untouched.
