---
phase: quick
plan: 2
type: execute
wave: 1
depends_on: []
files_modified:
  - qb_env/tossup_env.py
  - scripts/train_ppo.py
  - tests/test_environment.py
autonomous: true
requirements: [OPT-1]

must_haves:
  truths:
    - "PPO training produces identical observations whether beliefs are precomputed or computed live"
    - "TossupMCEnv with precomputed_beliefs never calls likelihood_model.score() during step()"
    - "train_ppo.py precomputes beliefs before PPO training when the feature is enabled"
    - "Existing tests continue to pass unchanged (backward compatible)"
  artifacts:
    - path: "qb_env/tossup_env.py"
      provides: "precomputed_beliefs parameter on TossupMCEnv, bypass in _compute_belief"
      contains: "precomputed_beliefs"
    - path: "scripts/train_ppo.py"
      provides: "precompute_beliefs helper and integration before PPOBuzzer construction"
      contains: "precompute_beliefs"
    - path: "tests/test_environment.py"
      provides: "Tests proving precomputed path matches live path exactly"
      contains: "precomputed_beliefs"
  key_links:
    - from: "scripts/train_ppo.py"
      to: "qb_env/tossup_env.py"
      via: "precompute_beliefs() builds dict, passed to TossupMCEnv constructor"
      pattern: "precomputed_beliefs="
    - from: "qb_env/tossup_env.py _compute_belief"
      to: "precomputed_beliefs dict"
      via: "lookup bypasses likelihood_model.score() when key exists"
      pattern: "self\\.precomputed_beliefs"
---

<objective>
Precompute belief trajectories for all questions ONCE before PPO training to eliminate redundant likelihood_model.score() calls during rollout collection.

Purpose: `_compute_belief()` calls `likelihood_model.score()` on every `step()` during SB3 PPO training. For n_steps=128 across 100k+ timesteps, the same (question, step_idx) pairs are scored repeatedly. The likelihood model (SBERT/T5) dominates wall time. By precomputing the full belief trajectory per question upfront (one pass), training steps become pure numpy lookups.

Output: Modified TossupMCEnv with optional precomputed_beliefs bypass, updated train_ppo.py with precomputation step, and tests proving equivalence.
</objective>

<execution_context>
@./.claude/get-shit-done/workflows/execute-plan.md
@./.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@qb_env/tossup_env.py
@models/features.py
@scripts/train_ppo.py
@scripts/_common.py
@tests/test_environment.py
@tests/conftest.py
@models/likelihoods.py

<interfaces>
<!-- Key types and contracts the executor needs -->

From qb_env/tossup_env.py:
```python
class TossupMCEnv(gym.Env[np.ndarray, int]):
    def __init__(self, questions, likelihood_model, K=4, reward_mode="time_penalty",
                 wait_penalty=0.01, early_buzz_penalty=0.0, buzz_correct=1.0,
                 buzz_incorrect=-0.5, belief_mode="from_scratch", beta=5.0, seed=13)
    def _compute_belief(self, question: MCQuestion, step_idx: int) -> np.ndarray
    def _softmax_scores(self, scores: np.ndarray) -> np.ndarray
    def reset(self, *, seed=None, options=None) -> tuple[np.ndarray, dict]
    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]

def make_env_from_config(mc_questions, likelihood_model, config) -> TossupMCEnv
```

From qb_data/mc_builder.py:
```python
@dataclass
class MCQuestion(TossupQuestion):
    options: List[str]
    gold_index: int
    option_profiles: List[str]
    option_answer_primary: List[str]
    distractor_strategy: str
    # Inherited: qid, question, tokens, cumulative_prefixes, run_indices, ...
```

From models/likelihoods.py:
```python
class LikelihoodModel(ABC):
    def score(self, clue_prefix: str, option_profiles: list[str]) -> np.ndarray
```

From models/features.py:
```python
def extract_belief_features(belief, prev_belief, step_idx, total_steps) -> np.ndarray
```
</interfaces>
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Add precomputed_beliefs bypass to TossupMCEnv and precompute_beliefs helper</name>
  <files>qb_env/tossup_env.py, tests/test_environment.py</files>
  <behavior>
    - Test: precomputed env produces identical beliefs as live env at every step for from_scratch mode
    - Test: precomputed env produces identical beliefs as live env at every step for sequential_bayes mode
    - Test: precomputed env never calls likelihood_model.score() (mock assert_not_called)
    - Test: env with precomputed_beliefs=None behaves identically to current env (backward compat)
    - Test: precompute_beliefs helper returns dict with correct keys and belief shapes
  </behavior>
  <action>
1. In `qb_env/tossup_env.py`, add a module-level helper function `precompute_beliefs`:

```python
def precompute_beliefs(
    questions: list[MCQuestion],
    likelihood_model: LikelihoodModel,
    belief_mode: str = "from_scratch",
    beta: float = 5.0,
    K: int = 4,
) -> dict[tuple[int, int], np.ndarray]:
```

This function iterates over each question (by index) and each step_idx in range(len(question.run_indices)), computes the belief using the same logic as `_compute_belief` (call likelihood_model.score, apply _softmax_scores equivalent), and stores the result in a dict keyed by `(question_index_in_list, step_idx)`. For `sequential_bayes` mode, it must simulate the sequential update chain (prior starts uniform, each step multiplies by likelihood and normalizes). Use a standalone softmax helper (extract `_softmax_scores` logic into a module-level `_softmax` function that takes scores and beta, so both the env method and precompute can share it).

2. In `TossupMCEnv.__init__`, add an optional `precomputed_beliefs: dict[tuple[int, int], np.ndarray] | None = None` parameter. Store as `self.precomputed_beliefs`. Also store `self._question_index_map: dict[str, int]` mapping `question.qid -> index` from the questions list (needed to look up the right key at runtime).

3. In `TossupMCEnv._compute_belief`, at the top, check if `self.precomputed_beliefs is not None`. If so, look up the current question's index via `self._current_question_idx` (set during reset, see below) and return `self.precomputed_beliefs[(self._current_question_idx, step_idx)].copy()`. The `.copy()` is important to prevent mutation of the cached array.

4. In `TossupMCEnv.reset`, after selecting `self.question`, set `self._current_question_idx`:
   - If `options` has `question_idx`, use that directly.
   - Otherwise, find the index via `self.questions.index(self.question)`.

5. Update `make_env_from_config` to accept and pass through an optional `precomputed_beliefs` parameter.

6. In `tests/test_environment.py`, add a new test class `TestPrecomputedBeliefs` with these tests:
   - `test_precomputed_matches_live_from_scratch`: Create an env without precomputation, run a full episode recording beliefs at each step. Then create a second env WITH precomputed beliefs (computed via `precompute_beliefs`), run the same episode (same seed, same question_idx=0), assert beliefs match at every step within 1e-6 tolerance.
   - `test_precomputed_matches_live_sequential_bayes`: Same as above but with belief_mode="sequential_bayes".
   - `test_precomputed_skips_scoring`: Create env with precomputed_beliefs, mock the likelihood_model.score method, run an episode with WAIT actions, assert score was never called.
   - `test_no_precomputed_backward_compat`: Create env with precomputed_beliefs=None (default), run episode, assert it works exactly as before (no regression).
   - `test_precompute_beliefs_helper_shape`: Call `precompute_beliefs` on sample questions, verify the dict has `(0, s)` keys for each step and each belief is shape (K,) float32 summing to ~1.0.

IMPORTANT: Do NOT change the behavior of the env when precomputed_beliefs is None. The existing `_compute_belief` path must remain untouched for that case. The precomputed path is a pure bypass.
  </action>
  <verify>
    <automated>cd /Users/ankit.aggarwal/Dropbox/Stanford/CS234/final_project/qanta-buzzer && python -m pytest tests/test_environment.py -x -v 2>&1 | tail -40</automated>
  </verify>
  <done>
    - TossupMCEnv accepts optional precomputed_beliefs parameter
    - precompute_beliefs() helper function exists at module level
    - _compute_belief() bypasses likelihood_model.score() when precomputed_beliefs is available
    - All existing tests pass unchanged
    - 5 new tests pass proving equivalence and bypass behavior
  </done>
</task>

<task type="auto">
  <name>Task 2: Integrate precomputation into train_ppo.py</name>
  <files>scripts/train_ppo.py</files>
  <action>
In `scripts/train_ppo.py`, modify `main()` to precompute beliefs before constructing the environment:

1. After `likelihood_model = build_likelihood_model(config, mc_questions)` (line 105), add a precomputation step:

```python
from qb_env.tossup_env import precompute_beliefs

env_cfg = config["environment"]
lik_cfg = config["likelihood"]

print(f"Precomputing belief trajectories for {len(mc_questions)} questions...")
belief_cache = precompute_beliefs(
    questions=mc_questions,
    likelihood_model=likelihood_model,
    belief_mode=str(env_cfg.get("belief_mode", "from_scratch")),
    beta=float(lik_cfg.get("beta", 5.0)),
    K=int(config["data"].get("K", 4)),
)
print(f"Cached {len(belief_cache)} belief vectors")
```

2. Pass `precomputed_beliefs=belief_cache` to `make_env_from_config`. Since Task 1 updated `make_env_from_config` to accept this parameter, the call becomes:

```python
env = make_env_from_config(
    mc_questions=mc_questions,
    likelihood_model=likelihood_model,
    config=config,
    precomputed_beliefs=belief_cache,
)
```

3. The likelihood_model is still passed to the env (needed if any code path falls back to live scoring, and for the evaluation phase which uses `run_episode` on the same env). The precomputed_beliefs bypass only affects the `step()` hot path during PPO `model.learn()`.

Keep the print statements so the user can see precomputation timing in the training log. This is the only file change needed for integration.
  </action>
  <verify>
    <automated>cd /Users/ankit.aggarwal/Dropbox/Stanford/CS234/final_project/qanta-buzzer && python -c "from scripts.train_ppo import main; print('import OK')" 2>&1</automated>
  </verify>
  <done>
    - train_ppo.py precomputes beliefs before PPO training
    - Precomputed beliefs are passed to the environment
    - Script imports cleanly without errors
    - Existing smoke pipeline contract preserved (--smoke flag still works)
  </done>
</task>

</tasks>

<verification>
Full test suite must pass:

```bash
cd /Users/ankit.aggarwal/Dropbox/Stanford/CS234/final_project/qanta-buzzer && scripts/ci.sh
```

Targeted environment tests:

```bash
cd /Users/ankit.aggarwal/Dropbox/Stanford/CS234/final_project/qanta-buzzer && python -m pytest tests/test_environment.py -x -v
```

Smoke pipeline still works (manual verification if time permits):

```bash
cd /Users/ankit.aggarwal/Dropbox/Stanford/CS234/final_project/qanta-buzzer && python scripts/train_ppo.py --smoke
```
</verification>

<success_criteria>
- All 220+ existing tests pass (zero regressions)
- 5 new precomputed belief tests pass
- TossupMCEnv with precomputed_beliefs=None behaves identically to before (backward compat)
- TossupMCEnv with precomputed_beliefs skips likelihood_model.score() entirely during step()
- train_ppo.py precomputes beliefs in a single pass before PPO training
- Precomputed beliefs are bitwise identical to live-computed beliefs
</success_criteria>

<output>
After completion, create `.planning/quick/2-precompute-belief-observation-trajectori/2-SUMMARY.md`
</output>
