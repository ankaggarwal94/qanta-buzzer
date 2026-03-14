# Extensions Master Run — Execution Ledger

**Started:** 2026-03-14
**Branch:** main
**Scope:** Expected Wins + Variable-K + DSPy extensions

---

## PATCH 00 — CONFIG / CLI OVERRIDE CONTRACT

- **Objective:** Fix parse_overrides to return flat dotted-key overrides instead of nested dicts that clobber sibling config sections
- **Status:** in_progress
- **Repo evidence:** `scripts/build_mc_dataset.py:41-97` — `parse_overrides()` builds nested dicts (`{"data": {"K": 5}}`). When passed to `merge_overrides()` (qb_data/config.py:165-207), which splits keys on `.`, a pre-nested dict key `"data"` has no dots, so `keys = ["data"]` and `current["data"] = {"K": 5}` replaces the entire `data` section, losing `csv_path`, `distractor_strategy`, etc.
- **Files changed:** (pending)
- **Tests added/updated:** (pending)
- **Verification:** (pending)
- **Commit:** (pending)
- **Rollback:** (pending)

---

## EW-01 — OPPONENT MODEL SCAFFOLD
- **Status:** pending

## EW-02 — ENV EXPECTED_WINS REWARD MODE
- **Status:** pending

## EW-03 — OFFLINE EXPECTED WINS METRIC + TRACE PLUMBING
- **Status:** pending

## EW-04 — EVALUATION / REPORTING INTEGRATION
- **Status:** pending

## VK-01 — VARIABLE-K DATASET CONTRACT
- **Status:** pending

## VK-02 — PADDED BELIEF FEATURES
- **Status:** pending

## VK-03 — VARIABLE-K ENV MODE + ACTION MASKS
- **Status:** pending

## VK-04 — GENERALIZE BASELINES AND EVALUATION TO MIXED K
- **Status:** pending

## VK-05 — OPTIONAL MASKED PPO FOR MLP PATH
- **Status:** pending

## VK-06 — VARIABLE-K TEXT WRAPPER + T5 POLICY
- **Status:** pending

## VK-07 — MIXED-K INTEGRATION COVERAGE
- **Status:** pending

## DSPY-01 — OPTIONAL DEPENDENCY + CONFIG SHELL
- **Status:** pending

## DSPY-02 — DSPYLIKELIHOOD WITH REAL SCORE CACHE
- **Status:** pending

## DSPY-03 — FACTORY INTEGRATION
- **Status:** pending

## DSPY-04 — OFFLINE DSPY COMPILE / OPTIMIZE SCRIPT
- **Status:** pending

## DSPY-05 — OPTIONAL DSPY ANSWER-PROFILE AUGMENTATION
- **Status:** pending

## FINAL-01 — DOCS SYNC + FINAL VERIFICATION BUNDLE
- **Status:** pending
