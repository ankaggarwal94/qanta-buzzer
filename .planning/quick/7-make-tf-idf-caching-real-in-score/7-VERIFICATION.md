---
phase: quick-7
verified: 2026-03-13T08:30:00Z
status: passed
score: 4/4 must-haves verified
---

# Quick Task 7: Make TF-IDF Caching Real in score() Verification Report

**Phase Goal:** Make TfIdfLikelihood.score() use the base class embed_and_cache() infrastructure instead of calling vectorizer.transform() directly on every invocation, eliminating redundant TF-IDF vectorization when the same option_profiles or clue_prefixes are scored multiple times.

**Verified:** 2026-03-13T08:30:00Z
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | TfIdfLikelihood.score() uses embed_and_cache() for both clue and option texts | ✓ VERIFIED | Lines 338-339 in models/likelihoods.py: `clue_emb = self.embed_and_cache([clue_prefix])[0]` and `option_embs = self.embed_and_cache(option_profiles)` |
| 2 | Repeated score() calls with the same option_profiles skip vectorizer.transform() | ✓ VERIFIED | Test `test_tfidf_score_cache_hit` passes: cache size grows by 1 (only new clue) when same options are reused |
| 3 | Scores are numerically identical (to float32 tolerance) to the old cosine_similarity implementation | ✓ VERIFIED | Test `test_tfidf_score_matches_cosine_reference` passes with atol=1e-6 comparing new vs sklearn cosine_similarity |
| 4 | TfIdfLikelihood._embed_batch() returns L2-normalized dense vectors | ✓ VERIFIED | Lines 369-371 in models/likelihoods.py: L2 normalization via `np.linalg.norm` with zero-norm guard; test `test_tfidf_embed_batch_normalized` confirms row norms ~1.0 |

**Score:** 4/4 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `models/likelihoods.py` | TfIdfLikelihood with cached score() and L2-normalized _embed_batch() | ✓ VERIFIED | Lines 310-341 (score method) and 343-371 (_embed_batch method) exist and contain `embed_and_cache` calls; L2 normalization present at lines 369-371 |
| `tests/test_likelihoods.py` | Tests for TF-IDF caching behavior and score equivalence | ✓ VERIFIED | 4 new tests exist at lines 124-166: `test_tfidf_embed_batch_normalized`, `test_tfidf_score_uses_cache`, `test_tfidf_score_cache_hit`, `test_tfidf_score_matches_cosine_reference` — all pass |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|----|--------|---------|
| `models/likelihoods.py:TfIdfLikelihood.score` | `models/likelihoods.py:LikelihoodModel.embed_and_cache` | `self.embed_and_cache()` call replaces direct vectorizer.transform() | ✓ WIRED | Lines 338-339: `self.embed_and_cache([clue_prefix])[0]` and `self.embed_and_cache(option_profiles)` found; pattern `self\.embed_and_cache\(` matches |
| `models/likelihoods.py:TfIdfLikelihood._embed_batch` | sklearn L2 normalization | Row-wise L2 normalization so dot product = cosine similarity | ✓ WIRED | Lines 369-371: `norms = np.linalg.norm(mat, axis=1, keepdims=True)` followed by `return mat / norms`; pattern `np\.linalg\.norm` matches |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| OPT-6 | 7-PLAN.md | Make TF-IDF caching real in score() | ✓ SATISFIED | Implementation verified: score() uses embed_and_cache(), _embed_batch() L2-normalizes, tests confirm caching behavior and numerical equivalence |

**Note:** OPT-6 is an internal optimization requirement referenced in the PLAN but not tracked in formal REQUIREMENTS.md. Verification confirms the optimization was successfully implemented.

### Anti-Patterns Found

None — no TODO/FIXME/HACK/PLACEHOLDER comments, no empty implementations, no console.log-only patterns in modified sections (lines 310-341, 343-371 of models/likelihoods.py; lines 124-166 of tests/test_likelihoods.py).

### Test Results

**All TfIdfLikelihood tests pass:**
```
tests/test_likelihoods.py::TestTfIdfLikelihood::test_tfidf_requires_fit PASSED
tests/test_likelihoods.py::TestTfIdfLikelihood::test_tfidf_embed_requires_fit PASSED
tests/test_likelihoods.py::TestTfIdfLikelihood::test_tfidf_fit_and_score PASSED
tests/test_likelihoods.py::TestTfIdfLikelihood::test_tfidf_embed_batch PASSED
tests/test_likelihoods.py::TestTfIdfLikelihood::test_tfidf_corpus_in_constructor PASSED
tests/test_likelihoods.py::TestTfIdfLikelihood::test_tfidf_fit_returns_self PASSED
tests/test_likelihoods.py::TestTfIdfLikelihood::test_tfidf_score_all_options PASSED
tests/test_likelihoods.py::TestTfIdfLikelihood::test_tfidf_embed_batch_normalized PASSED [NEW]
tests/test_likelihoods.py::TestTfIdfLikelihood::test_tfidf_score_uses_cache PASSED [NEW]
tests/test_likelihoods.py::TestTfIdfLikelihood::test_tfidf_score_cache_hit PASSED [NEW]
tests/test_likelihoods.py::TestTfIdfLikelihood::test_tfidf_score_matches_cosine_reference PASSED [NEW]

11 passed in 0.02s
```

**Commits verified:**
- `e65b5cab` - test(quick-7): add failing tests for TF-IDF caching and L2 normalization (RED)
- `ba773e72` - feat(quick-7): make TF-IDF score() use embed_and_cache with L2-normalized embeddings (GREEN)

### Implementation Verification

**Score() pattern matches SBERT/T5:**
```python
# TfIdfLikelihood.score() (lines 338-341)
clue_emb = self.embed_and_cache([clue_prefix])[0]
option_embs = self.embed_and_cache(option_profiles)
sims = option_embs @ clue_emb
return sims.astype(np.float32)
```

**L2 normalization in _embed_batch():**
```python
# TfIdfLikelihood._embed_batch() (lines 368-371)
mat = self.vectorizer.transform(texts).toarray().astype(np.float32)
norms = np.linalg.norm(mat, axis=1, keepdims=True)
norms[norms == 0] = 1.0  # avoid division by zero for empty docs
return mat / norms
```

**No cosine_similarity imports remaining in score():**
- `grep -n "cosine_similarity" models/likelihoods.py` returns no results in the score() method
- sklearn import removed from score(); only TfidfVectorizer import remains in `__init__`

## Summary

All must-haves verified. Phase goal achieved.

**Key accomplishments:**
1. TfIdfLikelihood.score() now routes through embed_and_cache(), matching the SBERT/T5/OpenAI pattern
2. _embed_batch() returns L2-normalized vectors, making dot product equivalent to cosine similarity
3. Embedding cache populated on first call and reused on subsequent calls (verified by cache hit test)
4. Numerical equivalence confirmed: new implementation produces identical scores to sklearn cosine_similarity reference (atol=1e-6)
5. All 11 TfIdfLikelihood tests pass, including 4 new regression tests
6. Zero-norm guard prevents NaN for empty documents
7. Behavior-preserving optimization: no API changes, no new dependencies

**Performance impact:** Eliminates redundant vectorizer.transform() calls when the same option_profiles repeat across questions/steps (common in baseline sweeps where K option profiles are constant).

**Pattern consistency:** All 4 LikelihoodModel subclasses (TF-IDF, SBERT, T5, OpenAI) now use identical score() patterns through embed_and_cache().

---

_Verified: 2026-03-13T08:30:00Z_
_Verifier: Claude (gsd-verifier)_
