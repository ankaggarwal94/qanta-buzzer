---
phase: quick-6
plan: 1
type: execute
wave: 1
depends_on: []
files_modified:
  - qb_data/mc_builder.py
  - tests/test_mc_builder_topk.py
autonomous: true
requirements: ["OPT-5"]

must_haves:
  truths:
    - "Distractor rankings contain only top-M candidates per answer, not all N"
    - "Top distractors are identical to the old full-sort for answers with many candidates"
    - "Random fallback still triggers when top-M list is exhausted by guard rejections"
    - "Existing tests pass unchanged (behavior-preserving for small N)"
  artifacts:
    - path: "qb_data/mc_builder.py"
      provides: "Top-M argpartition in _compute_rankings"
      contains: "argpartition"
    - path: "tests/test_mc_builder_topk.py"
      provides: "Regression test for top-M truncation"
      min_lines: 30
  key_links:
    - from: "qb_data/mc_builder.py"
      to: "qb_data/mc_builder.py"
      via: "_compute_rankings rankings consumed by build() guard loop"
      pattern: "rankings\\.get\\(gold"
---

<objective>
Replace full all-pairs distractor ranking with top-M retrieval in MCBuilder._compute_rankings().

Purpose: For N unique answers, the current code stores N full-length ranked lists (O(N^2) memory) and does O(N^2 log N) sorting. Only K-1=3 distractors are ever needed. Replacing np.argsort with np.argpartition + partial sort reduces per-answer work from O(N log N) to O(N + M log M) and memory from O(N^2) to O(N*M), where M = max(5*K, 30).

Output: Modified _compute_rankings method, new regression test.
</objective>

<execution_context>
@./.claude/get-shit-done/workflows/execute-plan.md
@./.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@qb_data/mc_builder.py
@tests/test_qb_rl_bridge.py
@configs/default.yaml
</context>

<interfaces>
<!-- Key contracts the executor needs -->

From qb_data/mc_builder.py:
```python
def _compute_rankings(
    self,
    answers: List[str],
    answer_profiles: Dict[str, str],
    answer_to_category: Dict[str, str],
) -> Dict[str, List[str]]:
    """Returns dict mapping each answer to a ranked list of distractors."""
```

Consumer in build() (lines 310-325):
```python
ranked = rankings.get(gold, [a for a in answers if a != gold])
for candidate in ranked:
    # guard checks...
    selected.append(candidate)
    if len(selected) >= self.K - 1:
        break
```

Random fallback (lines 327-338) activates when ranked list is exhausted.
</interfaces>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Add top-M regression test</name>
  <files>tests/test_mc_builder_topk.py</files>
  <behavior>
    - Test: With N=20 synthetic answers and TF-IDF strategy, _compute_rankings returns lists of length <= M (not N-1)
    - Test: Top-3 distractors from top-M list match top-3 from a full-sort baseline (order preserved)
    - Test: With N=5 and K=4 (so M would exceed N), rankings degrade gracefully to N-1 length lists
    - Test: category_random strategy is unaffected (no argpartition path)
  </behavior>
  <action>
Create tests/test_mc_builder_topk.py with a TestTopMRanking class.

Construct 20 synthetic answers with known TF-IDF profiles (short distinct sentences). Instantiate MCBuilder(K=4, strategy="tfidf_profile"). Call _compute_rankings with the 20 answers and profiles.

Test 1 (top_m_truncation): Assert all returned lists have length <= max(5*4, 30) = 30. Since N=20, this means N-1=19 (all fit within M=30), so also test with a tighter M scenario: mock the internal M computation or just verify the list is <= min(M, N-1).

Test 2 (order_preservation): Compute full rankings by temporarily using the old np.argsort approach (inline in test). Compare top-3 of each answer's ranking. They must match.

Test 3 (small_n_graceful): With N=5 answers, M=max(20, 30)=30 > N. Rankings should have length 4 (N-1) without error.

Test 4 (category_random_unaffected): With strategy="category_random", verify rankings still contain all same-category candidates shuffled.

Run these tests against the CURRENT code first (they should pass for Tests 2-4 since full-sort produces a superset; Test 1 will initially pass trivially since current code returns full lists). After Task 2 modifies the code, re-run to verify truncation.
  </action>
  <verify>
    <automated>cd /Users/ankit.aggarwal/Dropbox/Stanford/CS234/final_project/qanta-buzzer && python -m pytest tests/test_mc_builder_topk.py -x -v</automated>
  </verify>
  <done>All 4 tests pass against current (unmodified) code, establishing the baseline for Task 2.</done>
</task>

<task type="auto">
  <name>Task 2: Replace full argsort with top-M argpartition in _compute_rankings</name>
  <files>qb_data/mc_builder.py</files>
  <action>
Modify MCBuilder._compute_rankings() to use top-M retrieval instead of full sort. The change affects the three profile-based strategy blocks (tfidf_profile, sbert_profile, openai_profile).

1. At the top of _compute_rankings, after the category_random early return (line 170), compute M:
   ```python
   M = min(max(5 * self.K, 30), len(answers) - 1)
   ```
   This caps M at N-1 so argpartition never requests more indices than exist.

2. Extract a helper method or inline the top-M logic. Replace the ranking loop pattern:

   OLD (appears 2x: once for tfidf, once for sbert/openai):
   ```python
   for answer in answers:
       idx = answer_idx[answer]
       order = np.argsort(-sim[idx]).tolist()
       rankings[answer] = [answers[i] for i in order if answers[i] != answer]
   ```

   NEW:
   ```python
   for answer in answers:
       idx = answer_idx[answer]
       row = sim[idx]
       if M >= len(answers) - 1:
           # Small N: full sort (no benefit from partition)
           order = np.argsort(-row).tolist()
       else:
           # Top-M retrieval: O(N) partition + O(M log M) sort
           top_m_idx = np.argpartition(-row, M)[:M]
           top_m_idx = top_m_idx[np.argsort(-row[top_m_idx])]
           order = top_m_idx.tolist()
       rankings[answer] = [answers[i] for i in order if answers[i] != answer]
   ```

3. To avoid code duplication across tfidf and sbert/openai blocks, extract a private method `_rank_by_similarity(self, sim, answers, answer_idx, M)` that takes the similarity matrix and returns the rankings dict. Call it from both blocks.

4. Do NOT change category_random strategy -- it has no similarity matrix.

5. Do NOT change the similarity computation itself (cosine_similarity or embeddings @ embeddings.T).

6. Add a brief docstring note to _compute_rankings mentioning the top-M optimization.

Key correctness invariant: The `if answers[i] != answer` filter may remove the self-entry from the top-M, yielding M-1 usable candidates. With M = max(5*K, 30) and K=4, that gives at least 19 candidates -- far more than the 3 needed. The existing random fallback in build() handles the extremely rare case of all 19 being rejected by guards.
  </action>
  <verify>
    <automated>cd /Users/ankit.aggarwal/Dropbox/Stanford/CS234/final_project/qanta-buzzer && python -m pytest tests/test_mc_builder_topk.py tests/test_qb_rl_bridge.py -x -v && scripts/ci.sh</automated>
  </verify>
  <done>
    - np.argpartition used in _compute_rankings for all profile-based strategies
    - All new top-M tests pass (truncation, order preservation, small-N graceful, category_random unaffected)
    - Existing test_qb_rl_bridge.py::test_openai_profile_strategy_uses_openai_embeddings passes unchanged
    - Full test suite (scripts/ci.sh) passes with zero failures
  </done>
</task>

</tasks>

<verification>
1. `python -m pytest tests/test_mc_builder_topk.py -x -v` -- all 4 new tests pass
2. `python -m pytest tests/test_qb_rl_bridge.py -x -v` -- existing _compute_rankings test passes unchanged
3. `scripts/ci.sh` -- full 220-test suite passes
4. `grep -n "argpartition" qb_data/mc_builder.py` -- confirms optimization is present
5. `grep -n "argsort" qb_data/mc_builder.py` -- only appears in the small-N fallback branch
</verification>

<success_criteria>
- _compute_rankings uses np.argpartition for profile-based strategies when N > M
- Rankings are truncated to top-M length (not full N-1)
- Top distractors are identical to old full-sort (order preserved within top-M)
- No new dependencies added
- All existing tests pass unchanged
- New regression test covers truncation, order preservation, small-N, and category_random
</success_criteria>

<output>
After completion, create `.planning/quick/6-replace-full-all-pairs-distractor-rankin/6-SUMMARY.md`
</output>
