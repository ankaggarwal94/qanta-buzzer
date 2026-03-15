---
phase: quick-6
verified: 2026-03-12T08:00:00Z
status: passed
score: 4/4 must-haves verified
---

# Quick Task 6: Replace Full All-Pairs Distractor Ranking Verification Report

**Phase Goal:** Replace full all-pairs distractor ranking with top-M retrieval — optimization item #5. Replace np.argsort with np.argpartition top-M in MCBuilder._compute_rankings() for O(N + M log M) per answer instead of O(N log N). Behavior-preserving for common case.

**Verified:** 2026-03-12T08:00:00Z
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| #   | Truth                                                                                   | Status     | Evidence                                                                                                     |
| --- | --------------------------------------------------------------------------------------- | ---------- | ------------------------------------------------------------------------------------------------------------ |
| 1   | Distractor rankings contain only top-M candidates per answer, not all N                | ✓ VERIFIED | _rank_by_similarity uses argpartition and returns lists of length <= M; test_top_m_truncation passes        |
| 2   | Top distractors are identical to the old full-sort for answers with many candidates    | ✓ VERIFIED | test_order_preservation compares top-3 against full-sort baseline and passes for all 20 answers              |
| 3   | Random fallback still triggers when top-M list is exhausted by guard rejections        | ✓ VERIFIED | build() lines 368-379 implement unchanged random fallback when len(selected) < K-1                           |
| 4   | Existing tests pass unchanged (behavior-preserving for small N)                        | ✓ VERIFIED | All 4 top-M tests pass; 7/7 bridge tests pass; test_small_n_graceful confirms N=5 degrades gracefully       |

**Score:** 4/4 truths verified

### Required Artifacts

| Artifact                         | Expected                                    | Status     | Details                                                                                                                     |
| -------------------------------- | ------------------------------------------- | ---------- | --------------------------------------------------------------------------------------------------------------------------- |
| `qb_data/mc_builder.py`          | Top-M argpartition in _compute_rankings     | ✓ VERIFIED | argpartition appears at line 178 in _rank_by_similarity helper; M computed at line 224; full docstring update at line 193  |
| `tests/test_mc_builder_topk.py`  | Regression test for top-M truncation        | ✓ VERIFIED | 136 lines (min 30 required); 4 tests cover truncation, order preservation, small-N graceful, category_random unaffected    |

### Key Link Verification

| From                      | To                        | Via                                                        | Status     | Details                                                                                                      |
| ------------------------- | ------------------------- | ---------------------------------------------------------- | ---------- | ------------------------------------------------------------------------------------------------------------ |
| `qb_data/mc_builder.py`   | `qb_data/mc_builder.py`   | _compute_rankings rankings consumed by build() guard loop  | ✓ WIRED    | Line 344 computes rankings, line 351 consumes via `rankings.get(gold, ...)`, lines 355-366 iterate over ranked list |

### Requirements Coverage

| Requirement | Source Plan         | Description                                              | Status         | Evidence                                                                       |
| ----------- | ------------------- | -------------------------------------------------------- | -------------- | ------------------------------------------------------------------------------ |
| OPT-5       | 6-PLAN.md           | Replace full all-pairs ranking with top-M retrieval      | ✓ SATISFIED    | _rank_by_similarity uses np.argpartition; M = max(5*K, 30) caps memory/time    |

**Note:** OPT-5 not found in .planning/REQUIREMENTS.md but is documented in PLAN and SUMMARY as the optimization item driving this task.

### Anti-Patterns Found

No blocking anti-patterns found.

| File                      | Line | Pattern         | Severity | Impact                                                                |
| ------------------------- | ---- | --------------- | -------- | --------------------------------------------------------------------- |
| `qb_data/mc_builder.py`   | 334  | `return []`     | ℹ️ Info  | Valid guard for empty input list; early return prevents downstream errors |

### Human Verification Required

None — all behavioral assertions are covered by automated tests.

### Gaps Summary

No gaps found. All must-haves verified:

1. **Top-M truncation implemented:** `_rank_by_similarity` uses `np.argpartition(-row, M)[:M]` for O(N + M log M) complexity when M < N-1, and falls back to full `np.argsort` when M >= N-1 (small N scenario).

2. **Order preservation confirmed:** `test_order_preservation` verifies top-3 distractors match full-sort baseline across 20 synthetic answers with distinct TF-IDF profiles.

3. **Random fallback intact:** build() lines 368-379 preserve the existing random fallback logic that activates when the ranked list (now truncated to top-M) is exhausted by guard rejections.

4. **Behavior-preserving for small N:** `test_small_n_graceful` confirms N=5 case uses full sort (M >= N-1 branch) and produces complete N-1=4 length rankings without error.

5. **Code duplication eliminated:** `_rank_by_similarity` helper extracted to avoid maintaining identical ranking loops across tfidf_profile, sbert_profile, and openai_profile strategies.

6. **All tests passing:** 4/4 new regression tests pass, 7/7 existing bridge tests pass, including `test_openai_profile_uses_openai_embeddings` which exercises the new code path.

7. **Commits verified:** Both task commits exist in git history:
   - `b0d5d21b` — test(quick-6): add regression tests for top-M distractor ranking
   - `bc8b3b46` — feat(quick-6): replace full argsort with top-M argpartition in _compute_rankings

---

_Verified: 2026-03-12T08:00:00Z_
_Verifier: Claude (gsd-verifier)_
