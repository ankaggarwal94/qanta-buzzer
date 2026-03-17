# Cursor patch brief: finish the PPO/maskable + compare-policy fixes on `main`

Work on the **current local `main` branch** of `qanta-buzzer`.

Important:
- Use the **local repo state as source of truth**.
- If some pieces from PR #13 are already present locally, **keep them** and only add the missing behavior below.
- Make the patch **small, surgical, and test-backed**.
- Do **not** change unrelated behavior.

## Goal

Close the remaining correctness gaps around:
1. comparing MLP-policy checkpoints with the **wrong likelihood model/config**,
2. mask-aware PPO inference/evaluation when using a **binary `StopOnlyEnv` wrapper**, and
3. calibration consistency for **no-buzz** episodes.

## Why this patch is needed

The local `main` branch currently has these issues:
- `scripts/compare_policies.py` hardcodes `TfIdfLikelihood` for MLP evaluation and runs `agent.run_episode(deterministic=True)` without `question_idx`, so it can both use the wrong likelihood family and sample questions with replacement.
- `agents/ppo_buzzer.py` still loads plain `PPO` only and computes action probabilities from an **unmasked** policy distribution.
- `qb_env/stop_only_env.py` is a binary WAIT/BUZZ wrapper but does **not** expose a binary `action_masks()` method.
- `evaluation/metrics.py` lacks a single canonical helper for calibration-at-buzz pairs that skips `buzz_step < 0`.

## Required code changes

### 1) `qb_env/stop_only_env.py`

Add a binary action mask method so maskable PPO can use the wrapper’s own action space instead of falling back to the base env’s K+1 mask.

Implement:

```python
def action_masks(self) -> np.ndarray:
    """Return a binary action mask for the wrapper action space.

    Action 0 (WAIT) is always valid.
    Action 1 (BUZZ) is valid when the wrapper can deterministically map BUZZ
    to a concrete underlying answer action.
    """
```

Behavior:
- Return a NumPy bool array of shape `(2,)`.
- `mask[0]` must always be `True`.
- `mask[1]` should be `True` when:
  - `self.answer_mode == "argmax_belief"`, and
  - the wrapped env has a non-empty `belief` vector.
- Otherwise `mask[1] = False`.

Keep `step()` behavior unchanged except for any tiny cleanup needed.

### 2) `agents/ppo_buzzer.py`

Make loading and inference mask-aware.

#### A. Extend `load()`

Change the signature to:

```python
@classmethod
def load(
    cls,
    path: str | Path,
    env: TossupMCEnv,
    use_maskable_ppo: bool = False,
) -> "PPOBuzzer":
```

Behavior:
- Instantiate `cls(env=env, use_maskable_ppo=use_maskable_ppo)`.
- If `use_maskable_ppo` is true, load with `MaskablePPO.load(...)`.
- Otherwise load with `PPO.load(...)`.

#### B. Add `_current_action_masks()`

Add a helper:

```python
def _current_action_masks(self) -> np.ndarray | None:
```

Behavior:
- If `self._use_maskable` is false, return `None`.
- Prefer `self.env.action_masks()` if the wrapper/env exposes it.
- Otherwise fall back to the unwrapped base env via `_base_env()`.
- If neither exposes `action_masks`, return `None`.
- Return `np.asarray(masks, dtype=bool)`.

#### C. Make `action_probabilities()` mask-aware

When masks are available, call the policy distribution with `action_masks=...`.
Use the env’s current action-space shape. For the mask tensor, pass a batch dimension of 1.

Pseudo-shape:

```python
obs_tensor = th.as_tensor(obs, dtype=th.float32, device=self.model.device).unsqueeze(0)
masks = self._current_action_masks()
if masks is not None:
    dist = self.model.policy.get_distribution(
        obs_tensor,
        action_masks=th.as_tensor(masks, dtype=th.bool, device=self.model.device).unsqueeze(0),
    )
else:
    dist = self.model.policy.get_distribution(obs_tensor)
```

Then extract `dist.distribution.probs[0]` as before.

Do **not** change `run_episode()` action selection logic beyond using the now-masked probabilities.

### 3) `evaluation/metrics.py`

Add a canonical helper for buzz-time calibration and use it everywhere.

Implement:

```python
def calibration_pairs_at_buzz(results: list[Any]) -> tuple[list[float], list[int]]:
```

Behavior:
- Accept the same result objects currently handled by `_to_dict()`.
- Use `top_p_trace` if present, otherwise `c_trace`.
- If neither trace exists or the chosen trace is empty, skip the row.
- If `buzz_step < 0`, **skip** the row entirely.
- Otherwise clamp with `idx = min(buzz_step, len(conf_trace) - 1)`.
- Return parallel `(confidences, outcomes)` lists.

Then refactor `calibration_at_buzz()` to call this helper.

### 4) `scripts/train_ppo.py`

#### A. Plumb maskable PPO from config

Read:

```python
use_maskable = bool(ppo_cfg.get("use_maskable_ppo", False))
```

Pass it into `PPOBuzzer(..., use_maskable_ppo=use_maskable)`.

If helpful, print a short message when maskable PPO is enabled.

#### B. Save the effective config next to the checkpoint

After the final merged config is known (after overrides, smoke handling, etc.), save it to:

```python
out_dir / "config_used.json"
```

Use the existing JSON helper.

This file will let evaluation scripts use the same likelihood family/settings that were used during training.

### 5) `scripts/compare_policies.py`

This file needs the biggest cleanup.

#### A. Stop hardcoding TF-IDF for MLP evaluation

Do **not** instantiate `TfIdfLikelihood` directly in `evaluate_mlp_policy()`.

Instead:
- import and use `build_likelihood_model` from `scripts._common`,
- resolve the config to use for the checkpoint,
- then build the likelihood model from that config and `test_questions`.

#### B. Add a helper to resolve the checkpoint’s config

Add something like:

```python
def resolve_mlp_eval_config(checkpoint_path: str | Path, fallback_config: dict[str, Any]) -> dict[str, Any]:
```

Behavior:
- Let `checkpoint_dir = Path(checkpoint_path).resolve().parent`.
- If `checkpoint_dir / "config_used.json"` exists, load and return it.
- Otherwise return `fallback_config` unchanged.

Keep this intentionally simple; no need for a broad metadata search.

#### C. Load the PPO model consistently with the resolved config

Use:

```python
resolved_config = resolve_mlp_eval_config(checkpoint_path, config)
use_maskable = bool(resolved_config.get("ppo", {}).get("use_maskable_ppo", False))
agent = PPOBuzzer.load(checkpoint_path, env=env, use_maskable_ppo=use_maskable)
```

#### D. Evaluate deterministically over the test split exactly once

Replace the current loop with:

```python
results = [
    agent.run_episode(deterministic=True, question_idx=i)
    for i in range(len(test_questions))
]
```

#### E. Use the canonical calibration helper

Import `calibration_pairs_at_buzz` and use it instead of custom indexing logic.

#### F. Update module docstring / caveat text

The current docstring text implies the MLP path always uses TF-IDF. Update it to say:
- the MLP path uses the likelihood model specified by the checkpoint-sidecar config when available,
- otherwise it falls back to the provided config,
- if that resolved config selects TF-IDF, the TF-IDF corpus is fit on the evaluation set’s question/options text.

Keep the wording short and accurate.

### 6) Tests

Add focused regression tests.

#### A. `tests/test_metrics.py`

If not already present locally, add:

```python
def test_calibration_pairs_skip_no_buzz():
    results = [
        {
            "buzz_step": -1,
            "correct": True,
            "top_p_trace": [0.2, 0.3, 0.4],
        },
        {
            "buzz_step": 1,
            "correct": False,
            "top_p_trace": [0.6, 0.7],
        },
    ]
    confidences, outcomes = calibration_pairs_at_buzz(results)
    assert confidences == [0.7]
    assert outcomes == [0]
```

and a consistency test that `calibration_at_buzz()` equals the metrics computed from `calibration_pairs_at_buzz()`.

#### B. `tests/test_stop_only_env.py`

Create a small dummy wrapped env with:
- `belief = np.array([0.2, 0.8])`
- `reset()` returning a simple observation
- `step()` recording the received action

Tests:
1. `action_masks()` returns a bool array of shape `(2,)` and equals `[True, True]` when belief exists.
2. `step(1)` maps to underlying action `2` (`1 + argmax([0.2, 0.8])`).

#### C. `tests/test_compare_policies.py`

Add two tests.

1. `test_resolve_mlp_eval_config_prefers_checkpoint_sidecar`
   - Create a temp dir with a fake checkpoint path and `config_used.json` next to it.
   - Assert the helper loads the sidecar config instead of returning the fallback.

2. `test_evaluate_mlp_policy_uses_shared_likelihood_builder_and_question_idx`
   - Monkeypatch:
     - `scripts.compare_policies.build_likelihood_model`
     - `scripts.compare_policies.make_env_from_config`
     - `agents.ppo_buzzer.PPOBuzzer.load`
   - Have the fake agent record every `question_idx` passed to `run_episode()`.
   - Assert:
     - `build_likelihood_model` is called once,
     - direct TF-IDF construction is not used,
     - `question_idx` values are exactly `[0, 1, ..., n-1]`.

#### D. Optional but encouraged: `tests/test_ppo_buzzer_masking.py`

Add a lightweight unit test that `_current_action_masks()` prefers a wrapper-provided 2-action mask over the unwrapped base env mask.

This can be very small and use dummy objects; no real PPO model needed.

## Implementation notes

- Keep imports tidy and localize any new helper imports if that matches surrounding style.
- Preserve public behavior except where explicitly fixed above.
- Do not add heavy new dependencies.
- Use existing repo helpers (`save_json`, `build_likelihood_model`, etc.) instead of duplicating logic.

## Acceptance criteria

The patch is done when all of the following are true:
1. `compare_policies.py` no longer hardcodes `TfIdfLikelihood` for MLP evaluation.
2. `compare_policies.py` evaluates each test question exactly once with `question_idx=i`.
3. `StopOnlyEnv` exposes a correct binary `action_masks()` method.
4. `PPOBuzzer.load()` and `action_probabilities()` support maskable PPO correctly.
5. `calibration_at_buzz()` is implemented via `calibration_pairs_at_buzz()` and skips no-buzz episodes.
6. `train_ppo.py` saves `config_used.json` and plumbs `ppo.use_maskable_ppo` into the agent.
7. The new tests pass.

## Suggested validation commands

Run the smallest focused test slice first:

```bash
pytest -q \
  tests/test_metrics.py \
  tests/test_stop_only_env.py \
  tests/test_compare_policies.py
```

If you add the optional PPO masking test, include it too.

Then run the broader relevant subset if available:

```bash
pytest -q tests/test_metrics.py tests/test_variable_k_integration.py
```

## Deliverable

Commit the implementation plus tests as one cohesive patch.
In the final summary, list:
- files changed,
- the exact config sidecar filename used,
- whether `StopOnlyEnv.action_masks()` always returns `[True, True]` or conditionally disables buzz when belief is unavailable,
- and the focused pytest commands you ran.
