---
phase: quick
plan: 4
type: execute
wave: 1
depends_on: []
files_modified:
  - agents/threshold_buzzer.py
  - agents/bayesian_buzzer.py
  - scripts/run_baselines.py
  - tests/test_agents.py
autonomous: true
requirements: [OPT-03]

must_haves:
  truths:
    - "SoftmaxProfileBuzzer sweep uses precomputed beliefs instead of calling likelihood_model.score() per threshold"
    - "SequentialBayesBuzzer beliefs are computed once and swept across thresholds with pure numpy"
    - "AlwaysBuzzFinalBuzzer uses precomputed beliefs instead of calling likelihood_model.score()"
    - "All baseline outputs are numerically identical to the original (behavior-preserving)"
  artifacts:
    - path: "agents/threshold_buzzer.py"
      provides: "_softmax_episode_from_precomputed, _always_final_from_precomputed functions"
      contains: "def _softmax_episode_from_precomputed"
    - path: "agents/bayesian_buzzer.py"
      provides: "precompute_sequential_beliefs, _sequential_episode_from_precomputed, sweep_sequential_thresholds functions"
      contains: "def precompute_sequential_beliefs"
    - path: "scripts/run_baselines.py"
      provides: "One-pass precomputed evaluation for all baseline agents"
      contains: "sweep_sequential_thresholds"
    - path: "tests/test_agents.py"
      provides: "Equivalence tests proving precomputed paths match live paths"
      contains: "test_softmax_precomputed_matches_live"
  key_links:
    - from: "agents/threshold_buzzer.py"
      to: "scripts/run_baselines.py"
      via: "_softmax_episode_from_precomputed, _always_final_from_precomputed imports"
      pattern: "from agents\\.threshold_buzzer import.*_softmax_episode_from_precomputed"
    - from: "agents/bayesian_buzzer.py"
      to: "scripts/run_baselines.py"
      via: "sweep_sequential_thresholds import"
      pattern: "from agents\\.bayesian_buzzer import.*sweep_sequential_thresholds"
---

<objective>
Eliminate redundant likelihood_model.score() calls in the baseline sweep by making SoftmaxProfileBuzzer, SequentialBayesBuzzer, and AlwaysBuzzFinalBuzzer reuse precomputed beliefs.

Purpose: The SoftmaxProfile sweep recomputes beliefs that are mathematically identical to the already-precomputed from_scratch beliefs (N_thresholds x N_questions x avg_steps redundant calls). The SequentialBayes sweep recomputes beliefs that are threshold-independent (N_thresholds-1 redundant passes). AlwaysBuzzFinal recomputes beliefs identical to precomputed. This collapses all into one-pass precomputed evaluation.

Output: Modified agents and run_baselines.py with zero redundant model calls. Equivalence tests proving numerical identity.
</objective>

<execution_context>
@./.claude/get-shit-done/workflows/execute-plan.md
@./.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@agents/threshold_buzzer.py
@agents/bayesian_buzzer.py
@scripts/run_baselines.py
@tests/test_agents.py
@tests/conftest.py
@agents/__init__.py

<interfaces>
<!-- Existing contracts the executor needs -->

From agents/threshold_buzzer.py:
```python
@dataclass
class EpisodeResult:
    qid: str; buzz_step: int; buzz_index: int; gold_index: int
    correct: bool; reward_like: float
    c_trace: list[float]; g_trace: list[float]
    top_p_trace: list[float]; entropy_trace: list[float]

@dataclass
class _PrecomputedQuestion:
    qid: str; gold_index: int; num_options: int; beliefs: list[np.ndarray]

def _scores_to_belief(scores: np.ndarray, beta: float) -> np.ndarray
def _belief_stats(belief: np.ndarray) -> tuple[int, float, float]
def _episode_from_precomputed(pq, threshold, alpha) -> EpisodeResult
def precompute_beliefs(questions, likelihood_model, beta) -> list[_PrecomputedQuestion]
def sweep_thresholds(questions, likelihood_model, thresholds, beta, alpha, precomputed) -> dict[float, list[EpisodeResult]]
```

From agents/bayesian_buzzer.py:
```python
@dataclass
class SoftmaxEpisodeResult:
    qid: str; buzz_step: int; buzz_index: int; gold_index: int
    correct: bool
    c_trace: list[float]; g_trace: list[float]
    top_p_trace: list[float]; entropy_trace: list[float]

class SequentialBayesBuzzer:
    def _step_update(self, prior, fragment, option_profiles) -> np.ndarray
    def run_episode(self, question) -> SoftmaxEpisodeResult
    # Uses question.run_indices and question.tokens for fragment extraction
    # Uses prior * likelihood Bayesian update (NOT cumulative prefix)
```

Key math identity: SoftmaxProfileBuzzer._belief_from_scratch() == _scores_to_belief() from threshold_buzzer.py. Both compute: shifted = scores - max(scores); probs = exp(beta * shifted); probs / sum. Therefore precomputed beliefs from precompute_beliefs() ARE the softmax beliefs.
</interfaces>
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Add precomputed-path functions for SoftmaxProfile, SequentialBayes, and AlwaysBuzzFinal</name>
  <files>agents/threshold_buzzer.py, agents/bayesian_buzzer.py, agents/__init__.py, tests/test_agents.py</files>
  <behavior>
    - Test: _softmax_episode_from_precomputed(pq, threshold, alpha) returns SoftmaxEpisodeResult with identical buzz_step, buzz_index, correct, c_trace, g_trace, top_p_trace, entropy_trace as SoftmaxProfileBuzzer.run_episode() for the same question
    - Test: _always_final_from_precomputed(pq) returns EpisodeResult with buzz_step == len(beliefs)-1, buzz_index == argmax(final_belief), c_trace all zeros except last which is 1.0, same top_p_trace and entropy_trace as AlwaysBuzzFinalBuzzer.run_episode()
    - Test: precompute_sequential_beliefs() produces beliefs via Bayesian update (prior * likelihood) that match SequentialBayesBuzzer._step_update() for the same question
    - Test: _sequential_episode_from_precomputed(pq, threshold, alpha) returns SoftmaxEpisodeResult matching SequentialBayesBuzzer.run_episode() for the same question and threshold
    - Test: sweep_sequential_thresholds() returns dict[float, list[SoftmaxEpisodeResult]] matching per-threshold SequentialBayesBuzzer results
  </behavior>
  <action>
1. In `agents/threshold_buzzer.py`, add two functions after `_episode_from_precomputed`:

   a. `_softmax_episode_from_precomputed(pq: _PrecomputedQuestion, threshold: float, alpha: float) -> SoftmaxEpisodeResult` — identical logic to `_episode_from_precomputed` but returns `SoftmaxEpisodeResult` (no `reward_like` field). Import `SoftmaxEpisodeResult` from `agents.bayesian_buzzer`. The buzzing logic is the same threshold comparison: iterate beliefs, compute (top_idx, top_p, entropy) via `_belief_stats`, compute `c_t = sigmoid(alpha * (top_p - threshold))`, buzz when `top_p >= threshold` or last step.

   b. `_always_final_from_precomputed(pq: _PrecomputedQuestion) -> EpisodeResult` — iterates ALL beliefs (no early stopping), sets `c_trace` to all 0.0 except last which is 1.0, `g_trace[i] = 1.0 if top_idx == gold_index else 0.0` at each step, `buzz_step = len(beliefs)-1`, `buzz_index = argmax(beliefs[-1])`. Returns `EpisodeResult` with `reward_like = 1.0 if correct else -0.5`.

2. In `agents/bayesian_buzzer.py`, add three functions after the `SequentialBayesBuzzer` class:

   a. `precompute_sequential_beliefs(questions: list[MCQuestion], likelihood_model: LikelihoodModel, beta: float) -> list[_PrecomputedQuestion]` — for each question, starts with uniform prior, iterates `question.run_indices` extracting token fragments (same logic as `SequentialBayesBuzzer._step_update`), applies Bayesian update `posterior = prior * likelihood; posterior /= posterior.sum()`, collects beliefs at each step. Import `_PrecomputedQuestion` from `agents.threshold_buzzer`. Returns list of `_PrecomputedQuestion` where beliefs are the Bayesian posteriors (NOT the from-scratch softmax).

   b. `_sequential_episode_from_precomputed(pq: _PrecomputedQuestion, threshold: float, alpha: float) -> SoftmaxEpisodeResult` — same as `_softmax_episode_from_precomputed` but operates on sequential beliefs. Identical logic: iterate beliefs, compute stats, buzz when top_p >= threshold or last step. (Can literally be the same function since _PrecomputedQuestion is the same type, but define separately for clarity.)

   c. `sweep_sequential_thresholds(questions: list[MCQuestion], likelihood_model: LikelihoodModel, thresholds: list[float], beta: float = 5.0, alpha: float = 10.0, precomputed: list[_PrecomputedQuestion] | None = None) -> dict[float, list[SoftmaxEpisodeResult]]` — if precomputed is None, calls `precompute_sequential_beliefs()`. Then for each threshold, builds results via `_sequential_episode_from_precomputed()`. Signature mirrors `sweep_thresholds()` from threshold_buzzer.

3. In `agents/__init__.py`, add `sweep_sequential_thresholds` to the imports from `agents.bayesian_buzzer` and to `__all__`.

4. In `tests/test_agents.py`, add a new test class `TestPrecomputedEquivalence` with these tests:

   a. `test_softmax_precomputed_matches_live` — for sample_mc_question with TF-IDF, run SoftmaxProfileBuzzer.run_episode() at threshold=0.7 and _softmax_episode_from_precomputed() with the same precomputed beliefs. Assert buzz_step, buzz_index, correct, c_trace, g_trace, top_p_trace, entropy_trace are identical (use `np.testing.assert_array_almost_equal` for float lists, exact match for ints/bools).

   b. `test_always_final_precomputed_matches_live` — run AlwaysBuzzFinalBuzzer.run_episode() and _always_final_from_precomputed(). Assert all fields match.

   c. `test_sequential_precomputed_matches_live` — run SequentialBayesBuzzer.run_episode() at threshold=0.7 and _sequential_episode_from_precomputed() with precomputed sequential beliefs. Assert all fields match.

   d. `test_sweep_sequential_matches_per_threshold` — sweep 3 thresholds [0.5, 0.7, 0.9] via sweep_sequential_thresholds() and verify against individual SequentialBayesBuzzer.run_episode() calls at each threshold. Assert all results match.

   Import the new functions: `from agents.threshold_buzzer import _softmax_episode_from_precomputed, _always_final_from_precomputed, precompute_beliefs` and `from agents.bayesian_buzzer import precompute_sequential_beliefs, _sequential_episode_from_precomputed, sweep_sequential_thresholds`.
  </action>
  <verify>
    <automated>cd /Users/ankit.aggarwal/Dropbox/Stanford/CS234/final_project/qanta-buzzer && python -m pytest tests/test_agents.py -x -v 2>&1 | tail -40</automated>
  </verify>
  <done>All 4 new equivalence tests pass. All existing 40+ agent tests still pass. New functions are exported from agents/__init__.py.</done>
</task>

<task type="auto">
  <name>Task 2: Wire run_baselines.py to use precomputed paths, eliminating all redundant model calls</name>
  <files>scripts/run_baselines.py</files>
  <action>
Modify `scripts/run_baselines.py` main() to replace the per-threshold live agent loops with precomputed sweeps:

1. **Add imports** at top: Add `_softmax_episode_from_precomputed`, `_always_final_from_precomputed` to the existing `from agents.threshold_buzzer import ...` line. Add `sweep_sequential_thresholds` to the existing `from agents.bayesian_buzzer import ...` line. Remove `SoftmaxProfileBuzzer` from the bayesian_buzzer import since it is no longer instantiated.

2. **Replace Softmax sweep** (lines 172-189). The current code creates a SoftmaxProfileBuzzer per threshold and calls run_episode() per question. Replace with:

```python
# --- Softmax profile sweep (reuse from_scratch precomputed beliefs) ---
print("\nRunning SoftmaxProfile sweep (precomputed)...")
softmax_payload: dict[str, list[dict]] = {}
softmax_summary: dict[str, dict] = {}
for threshold in thresholds:
    results = [
        asdict(_softmax_episode_from_precomputed(pq, threshold, alpha))
        for pq in precomputed
    ]
    softmax_payload[str(threshold)] = results
    softmax_summary[str(threshold)] = summarize(results)
```

3. **Replace Sequential Bayes sweep** (lines 191-199). Currently creates SequentialBayesBuzzer per threshold and calls run_episode() per question. Replace with:

```python
# --- Sequential Bayes sweep (one belief pass, pure numpy threshold sweep) ---
print("Pre-computing sequential Bayes beliefs...")
seq_precomputed = precompute_sequential_beliefs(mc_questions, likelihood_model, beta)
print("Running SequentialBayes sweep (precomputed)...")
seq_results = sweep_sequential_thresholds(
    questions=mc_questions,
    likelihood_model=likelihood_model,
    thresholds=thresholds,
    beta=beta,
    alpha=alpha,
    precomputed=seq_precomputed,
)
sequential_payload: dict[str, list[dict]] = {}
sequential_summary: dict[str, dict] = {}
for threshold, runs in seq_results.items():
    rows = [asdict(r) for r in runs]
    sequential_payload[str(threshold)] = rows
    sequential_summary[str(threshold)] = summarize(rows)
```

4. **Replace AlwaysBuzzFinal** (lines 201-205). Currently creates AlwaysBuzzFinalBuzzer and calls run_episode(). Replace with:

```python
# --- AlwaysBuzzFinal (reuse from_scratch precomputed beliefs) ---
print("Running AlwaysBuzzFinal baseline (precomputed)...")
floor_runs = [asdict(_always_final_from_precomputed(pq)) for pq in precomputed]
floor_summary = summarize(floor_runs)
```

5. **Remove unused imports**: Remove `SequentialBayesBuzzer`, `SoftmaxProfileBuzzer` from bayesian_buzzer imports. Remove `AlwaysBuzzFinalBuzzer` from threshold_buzzer imports (no longer instantiated). Keep the class imports if they appear elsewhere in the codebase, but in this file they are only used in the now-removed loops.

6. **Keep the precompute_sequential_beliefs fragment extraction** in the embedding precomputation block (lines 140-148) so that the sequential fragments are already embedded when precompute_sequential_beliefs() runs. This block already exists and computes fragments from run_indices — no change needed.

The rest of main() (artifact saving, summary printing) stays unchanged since the payload/summary variable names and dict structures are identical.
  </action>
  <verify>
    <automated>cd /Users/ankit.aggarwal/Dropbox/Stanford/CS234/final_project/qanta-buzzer && scripts/ci.sh 2>&1 | tail -20</automated>
  </verify>
  <done>run_baselines.py makes zero redundant likelihood_model.score() calls. The Softmax sweep, Sequential Bayes sweep, and AlwaysBuzzFinal all use precomputed beliefs. Full test suite passes (scripts/ci.sh exit 0). Smoke pipeline still works: `python scripts/run_baselines.py --smoke` produces identical artifacts.</done>
</task>

</tasks>

<verification>
1. `scripts/ci.sh` passes (all 220+ tests green)
2. Equivalence tests in TestPrecomputedEquivalence prove numerical identity
3. `python scripts/run_baselines.py --smoke` completes and produces same artifact files in artifacts/smoke/
4. No new dependencies added
</verification>

<success_criteria>
- Zero redundant likelihood_model.score() calls in run_baselines.py (SoftmaxProfile, SequentialBayes, AlwaysBuzzFinal all use precomputed beliefs)
- All baseline outputs numerically identical to before (verified by equivalence tests)
- Full test suite passes unchanged
- Smoke pipeline produces same artifacts
</success_criteria>

<output>
After completion, create `.planning/quick/4-collapse-duplicate-baseline-sweeps-into-/4-SUMMARY.md`
</output>
