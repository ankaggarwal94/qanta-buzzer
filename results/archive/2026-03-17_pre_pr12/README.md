# Pre-PR-#12 Result Archive (2026-03-17 Mode-A run)

Historical snapshot of result artifacts that were generated on the
`codex/split-leakage-remediation` branch from a 2026-03-17 full-pipeline
run, before PR #12 landed in `main`.

These files were rescued from `codex/split-leakage-remediation` commit
[`752874e`](https://github.com/ankaggarwal94/qanta-buzzer/commit/752874e)
during the 2026-05-14 multi-agent CoVe audit that retired that branch
(see the audit notes referenced from `.cleanup-assessment/`). The
branch's code was fully superseded by what landed in PR #10 + PR #12
on `main`; only these output artifacts had unique informational value
worth preserving.

## What's here

| File | Description |
|------|-------------|
| `GENERATED_FILES_MANIFEST.md` | SHA-256 byte-equivalence audit table identifying three duplicate clusters from the 2026-05-13 Dropbox-conflict cleanup (e.g. `catrandom == k4`, `sbert == tfidf == seqbayes`, `ew_logistic == ew_empirical`). No equivalent exists in `main`. |
| `baselines_distractor_{catrandom,sbert,tfidf}.json` | Per-distractor-strategy baseline sweeps. Main has only the `tfidf` variant in canonical results. |
| `baselines_k4.json` | K=4 baseline from a clean post-fix run. Main's `results/baselines_k*.json` set lacks this specific output. |
| `baselines_variable_k.json` | Variable-K baseline sweep. Main lacks this. |
| `eval_expected_wins_empirical.json` | Empirical Expected Wins eval. Main only has the logistic variant. |
| `ppo_expected_wins.json` | PPO Expected Wins headline metrics. |
| `ppo_seed{1,2,3}.json` | Multi-seed PPO results (seeds 1/2/3). Main lacks per-seed evidence of variance. |

## Important caveat: not directly comparable to current `main`

The numbers in these files reflect the reward-shaping semantics of
`codex/split-leakage-remediation`, NOT `main` after PR #12 merged:

- Branch's `reward_from_buzz_step` used `wait_penalty * max(0, buzz_step)` —
  step-0 buzz incurs NO wait penalty.
- Main uses `wait_penalty * (buzz_step + 1)` — step-0 buzz incurs ONE wait
  penalty.

Both versions assert this expectation in tests
(`test_step_zero_buzz_has_no_wait_penalty` on branch,
`test_step_zero_buzz_includes_one_wait_penalty` on main), confirming it
is a deliberate semantic fork, not a bug fix in either direction.

The branch also reaches higher PPO accuracy (0.975 in `ppo_default.json`)
than the corresponding stale entry currently committed under main's
`results/ppo_default.json` (0.252, degenerate). The branch's number is
real — it came from a run on that branch's code state — but
should NOT be treated as a canonical post-PR-#12 result. Main's
`results/ppo_default.json` is itself technical debt: it was last
regenerated before PR #12's hardening landed and main has not re-run
the full pipeline since. Future re-runs on main will produce a new
canonical number that may differ from both of these.

## Recommended use

- Retrospective analysis of the 2026-03-17 run.
- Provenance for the SHA-256 duplicate-cluster audit in
  `GENERATED_FILES_MANIFEST.md`.
- Reference when retiring stale entries in `results/`.

Do NOT use these files as canonical results for `main`. Regenerate
`results/` by re-running the full pipeline on current `main` once the
underlying reward semantics are documented as stable.
