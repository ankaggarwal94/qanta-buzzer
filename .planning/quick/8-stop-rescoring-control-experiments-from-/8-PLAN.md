---
phase: quick-8
plan: 1
type: execute
wave: 1
depends_on: []
files_modified:
  - evaluation/controls.py
  - scripts/evaluate_all.py
  - tests/test_agents.py
autonomous: true
requirements: [OPT-7]

must_haves:
  truths:
    - "Shuffle control produces numerically identical results to re-scoring from scratch"
    - "Full evaluation uses precomputed beliefs instead of re-running SoftmaxProfileBuzzer"
    - "Zero likelihood_model.score() calls during shuffle control"
    - "Alias control is unchanged and still re-scores from scratch via callback evaluator"
  artifacts:
    - path: "evaluation/controls.py"
      provides: "run_shuffle_control_precomputed() that permutes precomputed beliefs"
      contains: "def run_shuffle_control_precomputed"
    - path: "scripts/evaluate_all.py"
      provides: "Full eval and shuffle control wired to precomputed belief path"
      contains: "precompute_beliefs"
    - path: "tests/test_agents.py"
      provides: "Equivalence test: shuffle-precomputed vs shuffle-rescore"
      contains: "test_shuffle_precomputed_matches_rescore"
  key_links:
    - from: "scripts/evaluate_all.py"
      to: "agents/threshold_buzzer.py"
      via: "precompute_beliefs() and _softmax_episode_from_precomputed()"
      pattern: "precompute_beliefs.*_softmax_episode_from_precomputed"
    - from: "evaluation/controls.py"
      to: "agents/threshold_buzzer.py"
      via: "_PrecomputedQuestion import and belief permutation"
      pattern: "_PrecomputedQuestion"
    - from: "scripts/evaluate_all.py"
      to: "evaluation/controls.py"
      via: "run_shuffle_control_precomputed(precomputed_beliefs, ...)"
      pattern: "run_shuffle_control_precomputed"
---

<objective>
Eliminate redundant likelihood_model.score() calls in evaluate_all.py by (1) using precomputed beliefs for the full SoftmaxProfileBuzzer evaluation, and (2) adding a precomputed-belief shuffle control that permutes belief vectors instead of re-scoring from scratch.

Purpose: The shuffle control currently re-runs the full SoftmaxProfileBuzzer pipeline on shuffled questions, making O(N*steps) redundant score() calls. Since shuffling only permutes option ordering, precomputed beliefs can be permuted with zero model calls.

Output: Modified evaluate_all.py, new run_shuffle_control_precomputed() in controls.py, equivalence test.
</objective>

<execution_context>
@./.claude/get-shit-done/workflows/execute-plan.md
@./.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@./CLAUDE.md
@./AGENTS.md
@./evaluation/controls.py
@./scripts/evaluate_all.py
@./agents/threshold_buzzer.py
@./agents/bayesian_buzzer.py
@./tests/test_agents.py
@./tests/conftest.py

<interfaces>
<!-- Key types and contracts the executor needs. -->

From agents/threshold_buzzer.py:
```python
@dataclass
class _PrecomputedQuestion:
    qid: str
    gold_index: int
    num_options: int
    beliefs: list[np.ndarray]   # one belief vector per clue step

def precompute_beliefs(
    questions: list[MCQuestion],
    likelihood_model: LikelihoodModel,
    beta: float,
) -> list[_PrecomputedQuestion]: ...

def _softmax_episode_from_precomputed(
    pq: _PrecomputedQuestion,
    threshold: float,
    alpha: float,
) -> SoftmaxEpisodeResult: ...
```

From evaluation/controls.py:
```python
def shuffled_option_copy(question: MCQuestion, rng: random.Random) -> MCQuestion:
    # Creates perm = list(range(K)); rng.shuffle(perm); applies perm to options/profiles/gold_index

def run_shuffle_control(
    questions: list[MCQuestion],
    evaluator: Callable[[list[MCQuestion]], dict[str, Any]],
    random_seed: int = 13,
) -> dict[str, Any]: ...
```

From agents/bayesian_buzzer.py:
```python
@dataclass
class SoftmaxEpisodeResult:
    qid: str
    buzz_step: int
    buzz_index: int
    gold_index: int
    correct: bool
    c_trace: list[float]
    g_trace: list[float]
    top_p_trace: list[float]
    entropy_trace: list[float]
```
</interfaces>
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Add run_shuffle_control_precomputed to evaluation/controls.py</name>
  <files>evaluation/controls.py, tests/test_agents.py</files>
  <behavior>
    - Test: run_shuffle_control_precomputed with precomputed beliefs produces numerically identical results to run_shuffle_control with live SoftmaxProfileBuzzer evaluator (same buzz_step, buzz_index, correct, c_trace, g_trace, top_p_trace, entropy_trace for every question)
    - Test: The permutation applied to beliefs matches the permutation applied to gold_index (same random seed produces same permutation)
  </behavior>
  <action>
1. In `evaluation/controls.py`, add a new function `run_shuffle_control_precomputed()`:

```python
def run_shuffle_control_precomputed(
    precomputed: list["_PrecomputedQuestion"],
    threshold: float,
    alpha: float,
    random_seed: int = 13,
) -> dict[str, Any]:
```

The function:
- Imports `_PrecomputedQuestion` and `_softmax_episode_from_precomputed` from `agents.threshold_buzzer`.
- For each `_PrecomputedQuestion`, generates the same permutation as `shuffled_option_copy` would (using `random.Random(random_seed)` — but note: `run_shuffle_control` creates ONE rng and calls `shuffled_option_copy(q, rng)` in sequence, so the new function must also use ONE rng across all questions to get the same permutation sequence).
- For each question: creates `perm = list(range(pq.num_options)); rng.shuffle(perm)` — exactly mirroring `shuffled_option_copy`.
- Permutes each belief vector: `shuffled_belief = belief[perm]` for each step's belief.
- Computes `new_gold = perm.index(pq.gold_index)` — same as `shuffled_option_copy`.
- Creates a new `_PrecomputedQuestion(qid=pq.qid, gold_index=new_gold, num_options=pq.num_options, beliefs=[b[perm] for b in pq.beliefs])`.
- Passes to `_softmax_episode_from_precomputed(shuffled_pq, threshold, alpha)`.
- Collects results as dicts via `dataclasses.asdict()`, passes to `summarize_buzz_metrics` and `calibration_at_buzz` (same pattern as `evaluate_questions` in evaluate_all.py).

CRITICAL correctness note: `belief[perm]` where perm is the SAME permutation used on options does the correct thing — if option i moved to position perm.index(i), then belief[perm] maps belief[perm[j]] to position j, which is the "gather by new ordering" semantics matching how `shuffled_option_copy` reorders options.

2. In `tests/test_agents.py`, add a new test class `TestShufflePrecomputedEquivalence`:

```python
class TestShufflePrecomputedEquivalence:
    def test_shuffle_precomputed_matches_rescore(self, sample_mc_question, sample_corpus):
        """Precomputed shuffle control matches live rescore shuffle control."""
```

The test:
- Creates a TF-IDF likelihood model.
- Sets threshold=0.7, beta=5.0, alpha=10.0.
- Runs `run_shuffle_control(questions, evaluator, random_seed=13)` where evaluator uses `SoftmaxProfileBuzzer`.
- Runs `precompute_beliefs(questions, likelihood, beta)` then `run_shuffle_control_precomputed(precomputed, threshold, alpha, random_seed=13)`.
- Asserts both produce identical summary metrics (mean_sq, buzz_accuracy) to floating point.
- Also compares per-run results: same buzz_step, buzz_index, correct for each question.

Note: The function returns the same dict structure as `evaluate_questions` in evaluate_all.py: `{**summarize_buzz_metrics(runs), **calibration_at_buzz(runs), "runs": runs}`. To make this self-contained, `run_shuffle_control_precomputed` should accept an optional `summarizer` callback, OR just return raw results and let the caller summarize. Prefer the simpler approach: have the function return the summary dict directly (importing metrics internally), matching the evaluator pattern used in evaluate_all.py.
  </action>
  <verify>
    <automated>cd /Users/ankit.aggarwal/Dropbox/Stanford/CS234/final_project/qanta-buzzer && python -m pytest tests/test_agents.py::TestShufflePrecomputedEquivalence -xvs 2>&1 | tail -20</automated>
  </verify>
  <done>run_shuffle_control_precomputed exists, equivalence test passes proving numerical identity with the live rescore path</done>
</task>

<task type="auto">
  <name>Task 2: Wire precomputed paths into evaluate_all.py</name>
  <files>scripts/evaluate_all.py</files>
  <action>
Modify `scripts/evaluate_all.py` to use precomputed beliefs for both the full evaluation and shuffle control. The alias control remains unchanged (it genuinely needs re-scoring).

1. Add imports at the top of the file:
   - `from agents.threshold_buzzer import precompute_beliefs, _softmax_episode_from_precomputed`
   - `from evaluation.controls import run_shuffle_control_precomputed` (new function from Task 1)

2. After building the likelihood model and loading config params (after line 168 "Using best softmax threshold"), add belief precomputation:
   ```python
   print("Precomputing beliefs...")
   precomputed = precompute_beliefs(mc_questions, likelihood_model, beta)
   ```

3. Replace the `evaluate_questions` closure (lines 172-182) with a precomputed version:
   ```python
   def evaluate_questions_precomputed(pqs):
       runs = [asdict(_softmax_episode_from_precomputed(pq, threshold, alpha)) for pq in pqs]
       summary = {**summarize_buzz_metrics(runs), **calibration_at_buzz(runs)}
       summary["runs"] = runs
       return summary
   ```

4. Replace line 186 `full_eval = evaluate_questions(mc_questions)` with:
   ```python
   full_eval = evaluate_questions_precomputed(precomputed)
   ```

5. Replace lines 204-207 (shuffle control) with the precomputed version:
   ```python
   print("Running shuffle control...")
   shuffle_eval = run_shuffle_control_precomputed(precomputed, threshold, alpha)
   ```

6. Keep the `evaluate_questions` closure (or a variant) ONLY for the alias control (lines 209-214), since alias substitution genuinely changes option text and profiles. Rename it to `evaluate_questions_live`:
   ```python
   def evaluate_questions_live(qset):
       agent = SoftmaxProfileBuzzer(
           likelihood_model=likelihood_model,
           threshold=threshold,
           beta=beta,
           alpha=alpha,
       )
       runs = [asdict(agent.run_episode(q)) for q in qset]
       summary = {**summarize_buzz_metrics(runs), **calibration_at_buzz(runs)}
       summary["runs"] = runs
       return summary
   ```
   And update the alias control call:
   ```python
   alias_eval = run_alias_substitution_control(
       mc_questions,
       alias_lookup=alias_lookup,
       evaluator=lambda qset: evaluate_questions_live(qset),
   )
   ```

7. Remove the now-unused import of `run_shuffle_control` from `evaluation.controls` (it was used on old line 205). Keep `run_alias_substitution_control`, `run_choices_only_control`.

Key constraint: The `SoftmaxProfileBuzzer` import must stay (used by alias control's live evaluator). The `asdict` import must stay.
  </action>
  <verify>
    <automated>cd /Users/ankit.aggarwal/Dropbox/Stanford/CS234/final_project/qanta-buzzer && python -c "from scripts.evaluate_all import main; print('import OK')" && scripts/ci.sh 2>&1 | tail -5</automated>
  </verify>
  <done>evaluate_all.py uses precomputed beliefs for full eval and shuffle control; alias control still uses live scoring; all existing tests pass</done>
</task>

</tasks>

<verification>
1. `scripts/ci.sh` passes (full pytest suite, 220+ tests)
2. New equivalence test `test_shuffle_precomputed_matches_rescore` proves numerical identity
3. `python -c "import ast; ast.parse(open('scripts/evaluate_all.py').read()); print('syntax OK')"` confirms no syntax errors
4. `python -c "import ast; ast.parse(open('evaluation/controls.py').read()); print('syntax OK')"` confirms no syntax errors
</verification>

<success_criteria>
- Shuffle control in evaluate_all.py makes zero likelihood_model.score() calls
- Full evaluation in evaluate_all.py uses precomputed beliefs (single pass)
- Alias control is unchanged (still uses live SoftmaxProfileBuzzer)
- Equivalence test proves shuffle-precomputed == shuffle-rescore to floating point
- All 220+ existing tests continue to pass
</success_criteria>

<output>
After completion, create `.planning/quick/8-stop-rescoring-control-experiments-from-/8-SUMMARY.md`
</output>
