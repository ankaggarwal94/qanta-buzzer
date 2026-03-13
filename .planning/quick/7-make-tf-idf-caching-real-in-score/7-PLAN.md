---
phase: quick-7
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - models/likelihoods.py
  - tests/test_likelihoods.py
autonomous: true
requirements: [OPT-6]

must_haves:
  truths:
    - "TfIdfLikelihood.score() uses embed_and_cache() for both clue and option texts"
    - "Repeated score() calls with the same option_profiles skip vectorizer.transform()"
    - "Scores are numerically identical (to float32 tolerance) to the old cosine_similarity implementation"
    - "TfIdfLikelihood._embed_batch() returns L2-normalized dense vectors"
  artifacts:
    - path: "models/likelihoods.py"
      provides: "TfIdfLikelihood with cached score() and L2-normalized _embed_batch()"
      contains: "embed_and_cache"
    - path: "tests/test_likelihoods.py"
      provides: "Tests for TF-IDF caching behavior and score equivalence"
      contains: "test_tfidf_score_uses_cache"
  key_links:
    - from: "models/likelihoods.py:TfIdfLikelihood.score"
      to: "models/likelihoods.py:LikelihoodModel.embed_and_cache"
      via: "self.embed_and_cache() call replaces direct vectorizer.transform()"
      pattern: "self\\.embed_and_cache\\("
    - from: "models/likelihoods.py:TfIdfLikelihood._embed_batch"
      to: "sklearn L2 normalization"
      via: "Row-wise L2 normalization so dot product = cosine similarity"
      pattern: "np\\.linalg\\.norm\\|normalize"
---

<objective>
Make TfIdfLikelihood.score() use the base class embed_and_cache() infrastructure instead of calling vectorizer.transform() directly on every invocation.

Purpose: Eliminate redundant TF-IDF vectorization when the same option_profiles or clue_prefixes are scored multiple times (common in baseline sweeps where K option profiles repeat across all questions/steps).

Output: Updated models/likelihoods.py with ~10 lines changed, new regression tests confirming behavioral equivalence.
</objective>

<execution_context>
@./.claude/get-shit-done/workflows/execute-plan.md
@./.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@models/likelihoods.py
@tests/test_likelihoods.py
@tests/conftest.py

<interfaces>
<!-- Key contracts the executor needs -->

From models/likelihoods.py — LikelihoodModel base class:
```python
def embed_and_cache(self, texts: list[str]) -> np.ndarray:
    """Embed texts, using cache for previously seen inputs.
    Returns np.ndarray of shape (len(texts), embed_dim), dtype float32."""

def _embed_batch(self, texts: list[str]) -> np.ndarray:
    """Embed a batch of texts. Subclasses must implement.
    Returns np.ndarray of shape (len(texts), embed_dim), dtype float32."""
```

From models/likelihoods.py — Current TfIdfLikelihood.score() (lines 310-341):
```python
def score(self, clue_prefix: str, option_profiles: list[str]) -> np.ndarray:
    if not self._is_fit:
        raise RuntimeError("TfIdfLikelihood must be fit() before score().")
    clue_vec = self.vectorizer.transform([clue_prefix])
    option_vecs = self.vectorizer.transform(option_profiles)
    from sklearn.metrics.pairwise import cosine_similarity
    sims = cosine_similarity(clue_vec, option_vecs)[0]
    return sims.astype(np.float32)
```

From models/likelihoods.py — Current TfIdfLikelihood._embed_batch() (lines 343-364):
```python
def _embed_batch(self, texts: list[str]) -> np.ndarray:
    if not self._is_fit:
        raise RuntimeError("TfIdfLikelihood must be fit() before embedding.")
    mat = self.vectorizer.transform(texts).toarray()
    return mat.astype(np.float32)
```

From models/likelihoods.py — SBERTLikelihood.score() pattern to match (lines 432-455):
```python
def score(self, clue_prefix: str, option_profiles: list[str]) -> np.ndarray:
    clue_emb = self.embed_and_cache([clue_prefix])[0]
    option_embs = self.embed_and_cache(option_profiles)
    sims = option_embs @ clue_emb
    return sims.astype(np.float32)
```
</interfaces>
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Add L2 normalization to _embed_batch and rewrite score() to use embed_and_cache</name>
  <files>models/likelihoods.py, tests/test_likelihoods.py</files>
  <behavior>
    - Test: TF-IDF _embed_batch returns L2-normalized vectors (row norms ~1.0)
    - Test: TF-IDF score() populates embedding_cache (cache size grows from 0 after first call)
    - Test: Repeated score() calls with same option_profiles do not grow cache (cache hit)
    - Test: Scores from new implementation match cosine_similarity reference to atol=1e-6
    - Test: score() before fit() still raises RuntimeError
    - Test: _embed_batch before fit() still raises RuntimeError
  </behavior>
  <action>
Two changes in models/likelihoods.py, both small and behavior-preserving:

**1. L2-normalize in _embed_batch() (line ~363):**
Replace:
```python
mat = self.vectorizer.transform(texts).toarray()
return mat.astype(np.float32)
```
With:
```python
mat = self.vectorizer.transform(texts).toarray().astype(np.float32)
norms = np.linalg.norm(mat, axis=1, keepdims=True)
norms[norms == 0] = 1.0  # avoid division by zero for empty docs
return mat / norms
```
This makes TF-IDF embeddings L2-normalized, matching SBERT/T5 convention. With normalized vectors, dot product = cosine similarity.

**2. Rewrite score() (lines 310-341):**
Replace the body (after the RuntimeError guard) with:
```python
clue_emb = self.embed_and_cache([clue_prefix])[0]
option_embs = self.embed_and_cache(option_profiles)
sims = option_embs @ clue_emb
return sims.astype(np.float32)
```
Remove the `from sklearn.metrics.pairwise import cosine_similarity` import inside score(). The sklearn import in `__init__` for TfidfVectorizer stays.

This exactly mirrors SBERT's score() pattern. The embed_and_cache() call routes through _embed_batch() which now returns L2-normalized vectors, so the dot product gives cosine similarity.

**3. Update score() docstring** to note it now uses the embedding cache.

**Tests to add in tests/test_likelihoods.py (TestTfIdfLikelihood class):**

```python
def test_tfidf_embed_batch_normalized(self, sample_corpus):
    """_embed_batch returns L2-normalized vectors (norms ~1.0)."""
    model = TfIdfLikelihood(corpus_texts=sample_corpus)
    embeddings = model._embed_batch(["George Washington president", "Thomas Jefferson"])
    norms = np.linalg.norm(embeddings, axis=1)
    np.testing.assert_array_almost_equal(norms, np.ones(2), decimal=5)

def test_tfidf_score_uses_cache(self, sample_corpus):
    """score() populates embedding_cache via embed_and_cache()."""
    model = TfIdfLikelihood(corpus_texts=sample_corpus)
    assert len(model.embedding_cache) == 0
    model.score("first president", ["Washington profile", "Lincoln profile"])
    assert len(model.embedding_cache) == 3  # 1 clue + 2 options

def test_tfidf_score_cache_hit(self, sample_corpus):
    """Repeated score() with same options reuses cache."""
    model = TfIdfLikelihood(corpus_texts=sample_corpus)
    options = ["George Washington president", "Thomas Jefferson declaration"]
    model.score("first president", options)
    cache_after_first = len(model.embedding_cache)
    model.score("second president", options)
    # Only the new clue should be added; options are cached
    assert len(model.embedding_cache) == cache_after_first + 1

def test_tfidf_score_matches_cosine_reference(self, sample_corpus):
    """New cached score() matches sklearn cosine_similarity reference."""
    from sklearn.metrics.pairwise import cosine_similarity as sklearn_cos
    model = TfIdfLikelihood(corpus_texts=sample_corpus)
    clue = "Who was the first president?"
    options = [
        "George Washington first president commander revolutionary",
        "Abraham Lincoln Civil War emancipation",
        "Thomas Jefferson declaration independence Virginia",
        "Benjamin Franklin inventor Philadelphia diplomat",
    ]
    # Compute reference via sklearn cosine_similarity (old method)
    clue_vec = model.vectorizer.transform([clue])
    option_vecs = model.vectorizer.transform(options)
    ref_scores = sklearn_cos(clue_vec, option_vecs)[0].astype(np.float32)
    # Compute via new cached path
    actual_scores = model.score(clue, options)
    np.testing.assert_allclose(actual_scores, ref_scores, atol=1e-6)
```

Write tests FIRST (RED), then apply the production code changes (GREEN). Existing tests must continue passing.
  </action>
  <verify>
    <automated>cd /Users/ankit.aggarwal/Dropbox/Stanford/CS234/final_project/qanta-buzzer && python -m pytest tests/test_likelihoods.py -x -v 2>&1 | tail -40</automated>
  </verify>
  <done>
    - TfIdfLikelihood.score() calls self.embed_and_cache() instead of vectorizer.transform()
    - TfIdfLikelihood._embed_batch() returns L2-normalized float32 arrays
    - 4 new tests pass: normalization, cache population, cache hit, cosine equivalence
    - All existing TF-IDF tests pass unchanged
    - No new dependencies added
  </done>
</task>

<task type="auto">
  <name>Task 2: Run full test suite to confirm no regressions</name>
  <files></files>
  <action>
Run the full pytest suite via scripts/ci.sh to confirm no regressions across the entire codebase. The TF-IDF likelihood is used pervasively in test fixtures (sample_tfidf_env), agent tests, PPO tests, and environment tests. All must still pass.

If any test fails, diagnose whether the failure is due to:
1. Floating point drift from L2 normalization (fix by adjusting tolerance)
2. Changed embedding dimensionality (should not happen - _embed_batch shape unchanged)
3. Unrelated pre-existing failure (document and move on)

Pay special attention to:
- tests/test_ppo_buzzer.py (uses sample_tfidf_env fixture)
- tests/test_factories.py (TF-IDF factory construction)
- tests/test_env.py or similar environment tests using TF-IDF

Note: The score values may differ slightly from before since we now go through L2-normalize + dot-product instead of sklearn cosine_similarity. But the mathematical result is identical (cosine similarity), so any test comparing relative ordering or approximate values should pass.
  </action>
  <verify>
    <automated>cd /Users/ankit.aggarwal/Dropbox/Stanford/CS234/final_project/qanta-buzzer && scripts/ci.sh 2>&1 | tail -30</automated>
  </verify>
  <done>
    - Full pytest suite passes with 0 failures
    - No regressions in agent, environment, factory, or PPO tests
    - Commit is safe to land
  </done>
</task>

</tasks>

<verification>
1. `pytest tests/test_likelihoods.py -x -v` -- all TF-IDF tests pass including 4 new ones
2. `scripts/ci.sh` -- full suite passes with 0 failures
3. `grep -n "embed_and_cache" models/likelihoods.py` -- appears in TfIdfLikelihood.score()
4. `grep -n "cosine_similarity" models/likelihoods.py` -- no longer imported in score()
</verification>

<success_criteria>
- TfIdfLikelihood.score() delegates to embed_and_cache() (matching SBERT/T5 pattern)
- _embed_batch() returns L2-normalized vectors (dot product = cosine similarity)
- New scores match old cosine_similarity reference to atol=1e-6
- Embedding cache is populated and reused on repeated calls
- Full test suite green with no regressions
- No new dependencies
</success_criteria>

<output>
After completion, create `.planning/quick/7-make-tf-idf-caching-real-in-score/7-SUMMARY.md`
</output>
