# Independently tuned IMCQA comparison

## Completed CPU stage and remaining execution

The 2026-10-05 request completed policy selection on 120 previously inspected
development questions, with 20 distinct calibration questions excluded. The
four rotations are averaged within question. Per-menu calibration maps remain
unchanged. The new policy parameters are in the lock's `policies` array;
the calibrators' older `selected_threshold` fields are provenance only.

| Distractors | Adaptive selective | Fixed selective | Development AS-FS reward |
|---|---|---|---:|
| Independent | threshold 0.65 | round 1, threshold 0.60 | -0.00125 |
| Same category | threshold 0.85 | round 1, threshold 0.65 | +0.05583333 |

These estimates were used to select policies. They are not fresh evaluation
results. All 21 thresholds plus PASS were considered for AS; all 5 rounds × 21
thresholds plus PASS for FS. No new inference or calibrator fit occurred.

The frozen target is 850 fresh questions, both menus, four rotations and five
prefixes: 34,000 new plain score contexts. `configs/imcqa_tuned_fresh.json` records
the analysis protocol. N targets approximately 90% power per menu to detect a
true reward difference 0.05 against zero, using the upper 95th percentile of the
development paired-SD bootstrap, the larger menu variance, and a two-menu
Bonferroni correction. This is neither guaranteed nor joint power. Establishing
a gain greater than 0.05 is a stricter claim and is not the sizing target.

The original source files were restored on 2026-10-06 UTC and all five input
hashes matched. The frozen cohort now contains 850 historical questions and
34,000 plain contexts. Its public input SHA256 is
`250d3fd3aff11cd9566e470eb79da057f2444ce2429654e8b27034760dddc430`.
See `docs/imcqa_tuned_preparation.md` for preparation provenance and commands.
GitHub Actions run `37396365453` authenticated to the existing Modal workspace
using repository secrets and verified the cache receipt, sidecar and expected
file existence. Weight bytes are rehashed by the scoring worker. Fresh model
inference and outcome analysis have not yet occurred at this launch checkpoint.
Synthetic integration tests validate software behavior, not model performance
or GPU numerics. The historical population and frozen policies remain unchanged.

The narrow `.github/workflows/imcqa-tuned-fresh.yml` workflow transports only the
compressed public package, verifies its manifest and the frozen analysis plan,
and runs the command below using the exact pushed source commit. Real gold
labels remain in the separate local evaluator. The one-attempt job has a $6
allocation ceiling and preserves complete or partial evidence as an Actions
artifact. A failed or interrupted attempt must not be silently rerun.

## Launch after preparation

The existing model cache must still be present in Modal workspace
`ankaggarwal94`; the launcher verifies its original preparation receipt and
all pinned weight hashes. Install the pinned Modal 1.6.0 SDK in the launch
environment if needed. The container uses the original pinned Python 3.11,
Torch 2.6.0 and Transformers 4.51.3 stack, and promotes the original loaded BF16
weights to FP32 before production scoring. No precision fallback is allowed.

Check https://modal.com/pricing immediately before launch. On 2026-10-06 UTC,
L40S + 2 physical CPU cores + 32 GiB memory totals $0.00063924/second. The bounded
allocation reservation is $5.53254008, including startup/scaledown and $0.20 for
staging/contingency. The explicit $6 ceiling covers this reservation, not an
independently verified invoice or indefinitely retained storage. No regions
or non-preemptible multipliers are requested. The rate timestamp must reflect
an actual check within 24 hours, not be refreshed automatically.

Set `IMCQA_RATE_VERIFIED_UTC` to that actual check time and run from the exact
committed source checkout. The first invocation validates without contacting
Modal:

```bash
cd /workspace/scratch/cb54ff173a76/qanta-buzzer
.venv/bin/python -m scripts.modal_imcqa_tuned \
  --public-package /workspace/scratch/cb54ff173a76/tuned_experiment/fresh/public/pilot.json \
  --source-commit "$(git rev-parse HEAD)" \
  --out /workspace/scratch/cb54ff173a76/tuned_experiment/run \
  --run-id imcqa-tuned-fresh-20261006 \
  --max-cost-usd 6.00 \
  --allocation-rate-usd-per-second 0.00063924 \
  --rate-source-url https://modal.com/pricing \
  --rate-verified-utc "${IMCQA_RATE_VERIFIED_UTC:?actual verified UTC timestamp required}" \
  --dry-run
```

After that succeeds, repeat the same command without `--dry-run`. The original
request authorizes the bounded experiment; no new scientific decisions are
needed. The worker runs once, with no automatic retries. Its hard timeout is
8,250 seconds and internal deadline 8,130 seconds. A measured first 128-row
benchmark can stop the run early if throughput cannot meet the budget. In that
case preserve all evidence and do not claim a primary result from partial data.
Collect-only recovery does not require another GPU allocation.

If scoring finished remotely but local collection failed, recover to a new
directory using the original launch control:

```bash
.venv/bin/python -m scripts.modal_imcqa_tuned --collect-only \
  --control /workspace/scratch/cb54ff173a76/tuned_experiment/run/control.json \
  --run-id imcqa-tuned-fresh-20261006 \
  --out /workspace/scratch/cb54ff173a76/tuned_experiment/recovered_run
```

Use that directory's `output/qwen7b` in the analysis command below.

## Analyze complete outputs

```bash
.venv/bin/python -m scripts.analyze_imcqa_tuned \
  --public /workspace/scratch/cb54ff173a76/tuned_experiment/fresh/public/pilot.json \
  --evaluator /workspace/scratch/cb54ff173a76/tuned_experiment/fresh/evaluator/main_dataset.json \
  --policy-lock /workspace/scratch/cb54ff173a76/tuned_policies/selection_v2/frozen_policies.json \
  --run /workspace/scratch/cb54ff173a76/tuned_experiment/run/output/qwen7b \
  --out /workspace/scratch/cb54ff173a76/tuned_experiment/analysis
```

The analyzer rejects incomplete receipts, mismatched public/evaluator/policy
hashes, changed model identities, altered numerical evidence, incorrect gold
joins, development questions/groups, and incomplete trajectories. It replays
the four policies with the separately selected thresholds. Primary results are
AS-FS mean rewards with 97.5% paired question-bootstrap intervals for each menu.
Coverage, error among answered and observed round have descriptive 95% intervals.
An interval including zero is inconclusive, not equivalence. A positive result
supports this adaptive policy against this independently tuned fixed family;
it does not isolate timing, certify optimal stopping, or guarantee low risk.

After actual scoring, independently reconstruct the reported contrasts and
inspect retained numerical evidence before making a scientific conclusion.
