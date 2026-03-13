---
phase: quick-8
verified: 2026-03-13T19:45:00Z
status: passed
score: 4/4 must-haves verified
re_verification: false
---

# Quick Task 8: Stop Re-scoring Control Experiments Verification Report

**Task Goal:** Stop rescoring control experiments from scratch, starting with shuffle control — optimization item #7. Add run_shuffle_control_precomputed() that permutes precomputed belief vectors instead of re-scoring. Wire precomputed beliefs into evaluate_all.py for both full evaluation and shuffle control. Alias control stays unchanged. Behavior-preserving.

**Verified:** 2026-03-13T19:45:00Z
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Shuffle control produces numerically identical results to re-scoring from scratch | ✓ VERIFIED | test_shuffle_precomputed_matches_rescore passes, comparing buzz_step, buzz_index, correct, c_trace, g_trace, top_p_trace, entropy_trace across all questions |
| 2 | Full evaluation uses precomputed beliefs instead of re-running SoftmaxProfileBuzzer | ✓ VERIFIED | evaluate_all.py:177 calls precompute_beliefs() once, then evaluate_all.py:181 uses _softmax_episode_from_precomputed() in evaluate_questions_precomputed closure for full eval |
| 3 | Zero likelihood_model.score() calls during shuffle control | ✓ VERIFIED | run_shuffle_control_precomputed (controls.py:316-372) only permutes belief arrays via numpy indexing (beliefs[perm]), makes zero likelihood model calls |
| 4 | Alias control is unchanged and still re-scores from scratch via callback evaluator | ✓ VERIFIED | evaluate_all.py:187-197 defines evaluate_questions_live() using SoftmaxProfileBuzzer.run_episode(), passed to run_alias_substitution_control at line 223 |

**Score:** 4/4 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `evaluation/controls.py` | run_shuffle_control_precomputed() that permutes precomputed beliefs | ✓ VERIFIED | Function exists at lines 316-372, imports _PrecomputedQuestion and _softmax_episode_from_precomputed, creates permutation via rng.shuffle(perm), permutes beliefs with belief[perm], computes new_gold via perm.index(), returns summary dict |
| `scripts/evaluate_all.py` | Full eval and shuffle control wired to precomputed belief path | ✓ VERIFIED | precompute_beliefs imported (line 45), called (line 177), used by evaluate_questions_precomputed (line 180-184), shuffle control uses run_shuffle_control_precomputed (line 220) |
| `tests/test_agents.py` | Equivalence test: shuffle-precomputed vs shuffle-rescore | ✓ VERIFIED | TestShufflePrecomputedEquivalence class added (lines 850-940), test_shuffle_precomputed_matches_rescore compares summary metrics and per-run results, test_permutation_consistency verifies perm matches gold_index transformation, both tests pass |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|----|--------|---------|
| scripts/evaluate_all.py | agents/threshold_buzzer.py | precompute_beliefs() and _softmax_episode_from_precomputed() | ✓ WIRED | Line 45 imports both functions, line 177 calls precompute_beliefs(), line 181 calls _softmax_episode_from_precomputed() in list comprehension |
| evaluation/controls.py | agents/threshold_buzzer.py | _PrecomputedQuestion import and belief permutation | ✓ WIRED | Line 350 imports _PrecomputedQuestion, line 362 instantiates it with permuted beliefs, line 317 type-hints parameter as list["_PrecomputedQuestion"] |
| scripts/evaluate_all.py | evaluation/controls.py | run_shuffle_control_precomputed(precomputed_beliefs, ...) | ✓ WIRED | Line 50 imports run_shuffle_control_precomputed, line 220 calls it with precomputed beliefs, threshold, and alpha |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| OPT-7 | 8-PLAN.md | Stop rescoring control experiments from scratch | ✓ SATISFIED | Shuffle control now permutes precomputed beliefs (zero score() calls), full eval uses precomputed beliefs (single pass), alias control unchanged (still live scoring) |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| N/A | N/A | None detected | N/A | No blockers, warnings, or notable patterns |

### Human Verification Required

None — all verification items are automatable and have been verified programmatically.

### Verification Evidence Summary

**Commits verified:**
- `a4603919` — test(quick-8): add failing tests for shuffle precomputed equivalence (RED)
- `01902552` — feat(quick-8): add run_shuffle_control_precomputed to evaluation/controls.py (GREEN)
- `af199b8b` — feat(quick-8): wire precomputed beliefs into evaluate_all.py

**Tests executed:**
- `test_shuffle_precomputed_matches_rescore` — PASSED (compares live rescore shuffle vs precomputed shuffle for summary metrics and per-run results)
- `test_permutation_consistency` — PASSED (verifies permutation applied to beliefs matches gold_index transformation)
- Full test suite: 140/140 non-transformer tests PASSED (transformer tests fail due to pre-existing huggingface_hub incompatibility, unrelated to changes)

**File integrity checks:**
- `scripts/evaluate_all.py` — syntax valid, imports successfully
- `evaluation/controls.py` — syntax valid
- `tests/test_agents.py` — syntax valid, new tests pass

**Implementation verification:**
1. `run_shuffle_control_precomputed` exists in controls.py with correct signature (lines 316-372)
2. Belief permutation logic uses numpy indexing `belief[perm]` matching shuffled_option_copy semantics
3. `precompute_beliefs` called once in evaluate_all.py (line 177) before any evaluation
4. Full evaluation uses `evaluate_questions_precomputed` closure (lines 180-184) with _softmax_episode_from_precomputed
5. Shuffle control uses `run_shuffle_control_precomputed` (line 220) with precomputed beliefs
6. Alias control uses `evaluate_questions_live` closure (lines 187-197) with SoftmaxProfileBuzzer.run_episode
7. All imports are wired correctly (lines 44-50 in evaluate_all.py)

**Numerical equivalence proven:**
- test_shuffle_precomputed_matches_rescore compares:
  - Summary metrics: mean_sq, buzz_accuracy (floating point equality)
  - Per-run: buzz_step, buzz_index, correct (exact equality)
  - Traces: c_trace, g_trace, top_p_trace, entropy_trace (almost_equal tolerance)
- test_permutation_consistency verifies:
  - Same random seed produces same permutation
  - Permuted gold_index matches shuffled_option_copy result

---

_Verified: 2026-03-13T19:45:00Z_
_Verifier: Claude (gsd-verifier)_
