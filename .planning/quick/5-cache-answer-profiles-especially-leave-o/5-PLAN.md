---
phase: quick-5
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - qb_data/answer_profiles.py
  - tests/test_answer_profile_cache.py
autonomous: true
requirements: [OPT-4]

must_haves:
  truths:
    - "Distractor profiles (exclude_qid=None) are computed once and returned from cache on subsequent calls"
    - "Leave-one-out gold profiles (answer, qid) are computed once per unique pair and returned from cache on subsequent calls"
    - "Cache is cleared when fit() is called, preventing stale data"
    - "Cached output is byte-identical to uncached output for all call patterns"
  artifacts:
    - path: "qb_data/answer_profiles.py"
      provides: "_cache dict and cache lookup in _profile_text, cache invalidation in fit()"
      contains: "_cache"
    - path: "tests/test_answer_profile_cache.py"
      provides: "Tests proving cache correctness, equivalence, and invalidation"
  key_links:
    - from: "qb_data/answer_profiles.py::_profile_text"
      to: "qb_data/answer_profiles.py::_cache"
      via: "dict lookup before computation, dict store after computation"
      pattern: "self\\._cache"
    - from: "qb_data/answer_profiles.py::fit"
      to: "qb_data/answer_profiles.py::_cache"
      via: "cache invalidation on new data"
      pattern: "self\\._cache\\s*="
---

<objective>
Add memoization to AnswerProfileBuilder._profile_text() so that repeated calls with the same (answer_primary, exclude_qid) pair return a cached result instead of re-iterating, re-joining, and re-truncating the grouped question texts.

Purpose: In MCBuilder.build(), every distractor answer gets profile_for_answer called once per question it appears as a distractor for (e.g., "Thomas Jefferson" as distractor in 200 questions = 200 identical calls with exclude_qid=None). Caching eliminates this O(N*K) redundant string processing.

Output: Modified answer_profiles.py with transparent _cache dict; new test file proving equivalence and invalidation.
</objective>

<execution_context>
@./.claude/get-shit-done/workflows/execute-plan.md
@./.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@qb_data/answer_profiles.py
@qb_data/mc_builder.py (consumer -- no changes needed, read-only reference)
@tests/conftest.py
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Add _cache dict to AnswerProfileBuilder and memoize _profile_text</name>
  <files>qb_data/answer_profiles.py, tests/test_answer_profile_cache.py</files>
  <behavior>
    - Test 1: profile_for_answer returns identical string on repeated calls with same (answer, None) args
    - Test 2: profile_for_answer returns identical string on repeated calls with same (answer, qid) args
    - Test 3: After fit() with new data, cache is empty and profiles reflect new data
    - Test 4: Cached profile for (answer, None) is byte-identical to a freshly computed profile (correctness equivalence)
    - Test 5: Cached leave-one-out profile for (answer, qid) is byte-identical to a freshly computed profile
    - Test 6: Cache reduces actual computation -- calling _profile_text N times with same args results in only 1 real computation (verify via call count or cache size)
  </behavior>
  <action>
1. Create tests/test_answer_profile_cache.py with the 6 tests above. Use TossupQuestion instances directly (import from qb_data.data_loader). Build a small fixture of ~5 questions with 2-3 shared answers so cache hits are exercisable. For test 6, monkeypatch the internal join/split logic or check len(builder._cache) after repeated calls.

2. In qb_data/answer_profiles.py, make the following minimal changes:

   a. In __init__, add: `self._cache: Dict[Tuple[str, Optional[str]], str] = {}`

   b. In fit(), add after `self._grouped = dict(grouped)`: `self._cache = {}` (invalidate on re-fit)

   c. In _profile_text(), wrap the existing body in a cache check:
      ```python
      key = (answer_primary, exclude_qid)
      if key in self._cache:
          return self._cache[key]
      # ... existing body unchanged ...
      result = " ".join(merged) if merged else answer_primary
      self._cache[key] = result
      return result
      ```
      Handle both early-return paths (the fallback `return answer_primary` when len(texts) < min_questions) by caching those too.

   d. Update the Tuple import in the typing line (already imported).

3. Do NOT modify mc_builder.py, profile_for_answer(), or build_profiles(). The cache is fully transparent -- callers see no API change.
  </action>
  <verify>
    <automated>cd /Users/ankit.aggarwal/Dropbox/Stanford/CS234/final_project/qanta-buzzer && python -m pytest tests/test_answer_profile_cache.py -x -v</automated>
  </verify>
  <done>All 6 tests pass. _profile_text returns cached results on repeated calls. Cache invalidated on fit(). Cached output identical to uncached.</done>
</task>

<task type="auto">
  <name>Task 2: Run full test suite to confirm no regressions</name>
  <files>qb_data/answer_profiles.py</files>
  <action>
Run scripts/ci.sh (which runs the full pytest suite). The cache is behavior-preserving, so all 220+ existing tests must pass unchanged. If any test fails, diagnose whether the failure is pre-existing or caused by the cache change:

- If caused by cache: the cache is returning stale or incorrect data. Fix the invalidation logic.
- If pre-existing: note in summary but do not fix (out of scope).

Also run the smoke pipeline stage 1 as an integration sanity check:
```bash
python scripts/build_mc_dataset.py --smoke
```
This exercises MCBuilder.build() -> profile_builder.profile_for_answer() in a realistic loop. Verify it completes without error and produces artifacts/smoke/ output.
  </action>
  <verify>
    <automated>cd /Users/ankit.aggarwal/Dropbox/Stanford/CS234/final_project/qanta-buzzer && scripts/ci.sh</automated>
  </verify>
  <done>Full test suite passes (220+ tests). Smoke build_mc_dataset completes without error. No behavioral regressions from the cache addition.</done>
</task>

</tasks>

<verification>
1. `python -m pytest tests/test_answer_profile_cache.py -x -v` -- all 6 new cache tests pass
2. `scripts/ci.sh` -- full suite passes, no regressions
3. `python scripts/build_mc_dataset.py --smoke` -- integration sanity: MC dataset builds correctly with cached profiles
</verification>

<success_criteria>
- AnswerProfileBuilder._profile_text() returns cached result on repeated (answer, exclude_qid) calls
- Cache is invalidated on fit() so stale data is impossible
- All existing 220+ tests pass unchanged (behavior-preserving)
- Smoke build_mc_dataset.py completes without error
</success_criteria>

<output>
After completion, create `.planning/quick/5-cache-answer-profiles-especially-leave-o/5-SUMMARY.md`
</output>
