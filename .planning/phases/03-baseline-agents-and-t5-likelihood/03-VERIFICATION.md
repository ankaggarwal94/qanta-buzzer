---
phase: 03-baseline-agents-and-t5-likelihood
verified: 2026-02-26T04:30:00Z
status: passed
score: 11/11 must-haves verified
re_verification: false
---

# Phase 3: Baseline Agents and T5 Likelihood Verification Report

**Phase Goal:** Users can run baseline agents and leverage T5 for semantic similarity scoring
**Verified:** 2026-02-26T04:30:00Z
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| #   | Truth                                                                       | Status     | Evidence                                                                 |
| --- | --------------------------------------------------------------------------- | ---------- | ------------------------------------------------------------------------ |
| 1   | ThresholdBuzzer produces valid episodes with c_trace and g_trace            | ✓ VERIFIED | Manual test: len(c_trace)=4, len(g_trace)=4, traces same length         |
| 2   | AlwaysBuzzFinal waits until last clue, then buzzes (c_trace[-1]=1.0)       | ✓ VERIFIED | Manual test: c_trace=[0.0, 0.0, 0.0, 1.0]                                |
| 3   | SoftmaxProfile recomputes belief from cumulative prefix each step           | ✓ VERIFIED | test_softmax_profile_recomputes_belief passes                            |
| 4   | SequentialBayes applies incremental Bayesian updates on clue fragments      | ✓ VERIFIED | test_sequential_bayes_bayesian_update passes                             |
| 5   | All agents return EpisodeResult/SoftmaxEpisodeResult with trace fields      | ✓ VERIFIED | test_episode_result_fields, test_softmax_episode_result_fields pass      |
| 6   | All four baseline agents execute without errors on test questions           | ✓ VERIFIED | 33 agent tests pass in 0.24s                                             |
| 7   | T5Likelihood computes semantic similarity scores using T5 encoder           | ✓ VERIFIED | Manual test: Washington=0.575 > Einstein=0.440 for "first president"    |
| 8   | T5 embeddings are cached automatically via inherited embed_and_cache()      | ✓ VERIFIED | Manual test: cache_size=2, embeddings match exactly                      |
| 9   | T5 scores 'first president' higher for 'Washington' than 'Einstein'         | ✓ VERIFIED | test_t5_semantic_scoring passes                                          |
| 10  | GPU tensors are detached and moved to CPU to prevent memory leaks           | ✓ VERIFIED | Code inspection: embeddings.detach().cpu().numpy() at line 459          |
| 11  | Test suite runs in under 60 seconds                                         | ✓ VERIFIED | Agents: 0.24s, T5 tests: 4.99s, total < 10s                              |

**Score:** 11/11 truths verified

### Required Artifacts

| Artifact                        | Expected                                                      | Status     | Details                                                  |
| ------------------------------- | ------------------------------------------------------------- | ---------- | -------------------------------------------------------- |
| `agents/threshold_buzzer.py`    | ThresholdBuzzer, AlwaysBuzzFinalBuzzer, EpisodeResult         | ✓ VERIFIED | 175 lines, exports all classes, imports work            |
| `agents/bayesian_buzzer.py`     | SoftmaxProfileBuzzer, SequentialBayesBuzzer                   | ✓ VERIFIED | 159 lines, exports both classes, imports work           |
| `agents/__init__.py`            | Package exports for 4 agents, 2 result types, 2 utilities     | ✓ VERIFIED | 23 lines, exports 8 items total                         |
| `models/likelihoods.py`         | T5Likelihood class added                                      | ✓ VERIFIED | 553 lines total (+140 for T5), class exists at line 349 |
| `models/__init__.py`            | T5Likelihood export                                           | ✓ VERIFIED | Updated to include T5Likelihood                         |
| `tests/conftest.py`             | sample_t5_model fixture                                       | ✓ VERIFIED | Module-scoped fixture, t5-small                         |
| `tests/test_agents.py`          | 33 agent tests                                                | ✓ VERIFIED | 643 lines, 33 tests pass in 0.24s                       |
| `tests/test_likelihoods.py`     | 5 T5 tests                                                    | ✓ VERIFIED | 315 lines, 5 T5 tests pass in 4.99s                     |

### Key Link Verification

| From                                    | To                            | Via                                     | Status     | Details                                                     |
| --------------------------------------- | ----------------------------- | --------------------------------------- | ---------- | ----------------------------------------------------------- |
| agents/threshold_buzzer.py              | models.likelihoods            | import LikelihoodModel                  | ✓ WIRED    | Line: from models.likelihoods import LikelihoodModel        |
| agents/threshold_buzzer.py              | qb_data.mc_builder            | import MCQuestion                       | ✓ WIRED    | Line: from qb_data.mc_builder import MCQuestion             |
| agents/bayesian_buzzer.py               | models.likelihoods            | import LikelihoodModel                  | ✓ WIRED    | Line: from models.likelihoods import LikelihoodModel        |
| agents/bayesian_buzzer.py               | qb_data.mc_builder            | import MCQuestion                       | ✓ WIRED    | Line: from qb_data.mc_builder import MCQuestion             |
| ThresholdBuzzer.__init__                | LikelihoodModel               | accepts likelihood_model parameter      | ✓ WIRED    | Signature: def __init__(self, likelihood_model: LikelihoodModel) |
| ThresholdBuzzer.run_episode             | MCQuestion                    | accepts question parameter              | ✓ WIRED    | Signature: def run_episode(self, question: MCQuestion)      |
| models.likelihoods.T5Likelihood         | transformers.T5EncoderModel   | T5EncoderModel.from_pretrained()        | ✓ WIRED    | Line 403: T5EncoderModel.from_pretrained(model_name)        |
| models.likelihoods.T5Likelihood.score   | embed_and_cache               | inherited cache lookup                  | ✓ WIRED    | Lines 483-484: clue_emb = self.embed_and_cache([clue_prefix]) |
| models.likelihoods.T5Likelihood._embed_batch | attention mask           | mean pooling with mask                  | ✓ WIRED    | Lines 449-453: masked_hidden = last_hidden * mask           |
| tests/test_agents.py                    | agents.ThresholdBuzzer        | instantiate and test                    | ✓ WIRED    | 33 tests instantiate and run agents                         |
| tests/test_likelihoods.py               | models.T5Likelihood           | semantic scoring and cache tests        | ✓ WIRED    | 5 tests verify T5 functionality                             |

### Requirements Coverage

| Requirement | Source Plan | Description                                              | Status      | Evidence                                                    |
| ----------- | ----------- | -------------------------------------------------------- | ----------- | ----------------------------------------------------------- |
| AGT-02      | 03-01       | ThresholdBuzzer baseline (sweeps configurable thresholds)| ✓ SATISFIED | agents/threshold_buzzer.py, test_threshold_buzzer_* pass    |
| AGT-03      | 03-01       | AlwaysBuzzFinalBuzzer baseline (buzzes on last clue)     | ✓ SATISFIED | agents/threshold_buzzer.py, test_always_buzz_final_* pass   |
| AGT-04      | 03-01       | SoftmaxProfileBuzzer baseline with explicit scoring      | ✓ SATISFIED | agents/bayesian_buzzer.py, test_softmax_profile_* pass      |
| AGT-05      | 03-01       | SequentialBayesBuzzer baseline with Bayesian updates     | ✓ SATISFIED | agents/bayesian_buzzer.py, test_sequential_bayes_* pass     |
| AGT-06      | 03-01, 03-03| All agents produce episode traces (c_trace, g_trace)     | ✓ SATISFIED | EpisodeResult/SoftmaxEpisodeResult dataclasses, tests pass  |
| LIK-04      | 03-02, 03-03| T5Likelihood implementation using T5 encoder             | ✓ SATISFIED | models/likelihoods.py lines 349-486, semantic test passes   |
| LIK-05      | 03-02, 03-03| Embedding cache with text hashing for T5                 | ✓ SATISFIED | Inherited from LikelihoodModel, cache test passes           |

### Anti-Patterns Found

No anti-patterns detected.

**Scanned files:**
- agents/threshold_buzzer.py (175 lines)
- agents/bayesian_buzzer.py (159 lines)
- agents/__init__.py (23 lines)
- models/likelihoods.py (T5 section: lines 349-486)

**Checks performed:**
- ✓ No TODO/FIXME/PLACEHOLDER comments
- ✓ No empty implementations (return null/{}[])
- ✓ No console.log placeholders
- ✓ No stub functions (all methods substantive)

### Human Verification Required

None. All functionality is programmatically verifiable and has been tested.

**Why no human verification needed:**
- Agent execution is deterministic and testable via pytest
- T5 semantic scoring is quantitatively measurable (Washington > Einstein scores)
- Episode trace format is programmatically inspectable
- All wiring is statically verifiable via imports and type signatures

### Phase Quality Summary

**Strengths:**
- Clean direct port from qb-rl reference implementation with minimal changes
- All agents produce episode traces compatible with S_q evaluation metric
- T5 semantic scoring verified with meaningful discrimination (Washington 0.575 vs Einstein 0.440)
- Comprehensive test coverage: 38 new tests (33 agents + 5 T5) all passing
- Fast test execution: agents 0.24s, T5 4.99s (uses t5-small for speed)
- Zero anti-patterns detected in all modified files

**Phase-level decisions:**
- Consolidated softmax_profile_buzzer.py and bayesian_buzzer.py into single file (both are Bayesian-family agents)
- Used T5EncoderModel (not T5ForConditionalGeneration) for 2x faster inference and half memory
- Used T5TokenizerFast for faster tokenization
- Module-scoped T5 fixture reduces test runtime from ~25s to ~5s

**Next phase readiness:**
- All baseline agents ready for Phase 4 (PPO Training Pipeline)
- Episode trace format (c_trace, g_trace) compatible with S_q evaluation
- T5Likelihood ready for use in TossupMCEnv via factory configuration
- All requirements (AGT-02 through AGT-06, LIK-04, LIK-05) satisfied

---

_Verified: 2026-02-26T04:30:00Z_
_Verifier: Claude (gsd-verifier)_
