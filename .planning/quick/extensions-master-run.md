# Extensions Master Run — Execution Ledger

**Started:** 2026-03-14
**Branch:** main
**Scope:** Expected Wins + Variable-K + DSPy extensions

---

## Summary

| Patch | Status | Commit | Tests |
|-------|--------|--------|-------|
| PATCH 00 — Config override fix | completed | 239a2a81 | 5 |
| EW-01 — Opponent model scaffold | completed | a080c997 | 11 |
| EW-02 — Env expected_wins reward | completed | 7b844f00 | 4 |
| EW-03 — Offline EW metric + traces | completed | 2bf22856 | 4 |
| EW-04 — Evaluation reporting | completed | 7cb1d870 | 0 |
| VK-01 — Variable-K dataset | completed | db9afcb5 | 5 |
| VK-02 — Padded belief features | completed | 85574886 | 5 |
| VK-03 — Variable-K env + masks | completed | d6c34835 | 3 |
| VK-04 — Baselines mixed K | completed | b7156c57 | 1 |
| VK-05 — Optional MaskablePPO | completed | 0fbeb85f | 2 (1 skip) |
| VK-06 — Text wrapper + T5 | completed | 06152803 | 1 |
| VK-07 — Mixed-K integration | completed | fc358ecc | 2 |
| DSPY-01 — Config shell | completed | 78e5ddc1 | 0 |
| DSPY-02 — DSPyLikelihood | completed | a93d4c47 | 6 |
| DSPY-03 — Factory integration | completed | e1bc7c12 | 2 |
| DSPY-04 — Offline compile | completed | 34e98645 | 4 (1 skip) |
| DSPY-05 — Answer profiles | completed | 943019c6 | 2 (1 skip) |
| FINAL-01 — Docs + verification | completed | a70fb00e | 0 |
| REVIEW — ChatGPT anti-hallucination fixes | completed | fd34e25a | 3 |

**Total new tests: 60**
**Total tests in repo: 318 passed, 3 skipped**

---

## Final Verification

| Check | Result |
|-------|--------|
| Full CI (`pytest tests/`) | 318 passed, 3 skipped (58s) |
| Manual smoke (`bash scripts/manual-smoke.sh`) | 4/4 stages complete (10s) |
| T5 smoke | Supervised + PPO, test accuracy reported (22s) |
| Existing invariants | calibration uses top_p_trace, splits use hashlib.md5, alias re-scores live |
| Default smoke unchanged | No new config defaults change existing behavior |

### Skipped optional verifications
- MaskablePPO integration: sb3-contrib not installed locally
- DSPy live compile: dspy not installed locally
- Full 100k PPO: intentionally out of scope

---

## Remaining Risks

1. Full 100k PPO training run not verified end-to-end
2. SBERT/T5-large likelihood paths not exercised locally
3. MaskablePPO path untested at integration level (requires sb3-contrib)
4. DSPy compile/optimize requires live LM backend not tested locally
5. compare_policies S_q/reward comparisons remain qualitative across architectures
6. TF-IDF cache memory grows with corpus size (bounded by cache_memory_bytes monitoring)

---

## Rollback

Each patch is one atomic commit. To roll back any patch:
```bash
git revert <commit-hash>
```

To roll back the entire extension campaign:
```bash
git revert --no-commit 239a2a81..HEAD
git commit -m "revert: roll back extension campaign"
```
