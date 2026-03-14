# Testing

## Framework

- **pytest** with shared fixtures in `tests/conftest.py`
- No pytest plugins or custom markers in use
- Tests run from project root: `pytest` or `pytest tests/`

## Test Structure

```
tests/
├── conftest.py                # Shared fixtures (module-scoped for heavy models)
├── test_agents.py             # ThresholdBuzzer, SoftmaxProfileBuzzer, precomputed equivalence
├── test_answer_profile_cache.py # Answer profile memoization cache correctness
├── test_build_mc_dataset.py   # MC dataset construction, anti-artifact guards
├── test_dataset_splits.py     # Stratified split reproducibility (cross-process determinism)
├── test_environment.py        # TossupMCEnv reset/step/reward/done, precomputed beliefs
├── test_factories.py          # Factory functions (make_env_from_config, build_likelihood_from_config)
├── test_features.py           # Belief feature extraction (shape, range, edge cases)
├── test_likelihoods.py        # TfIdf, SBERT, T5 scoring (shape, dtype, cache persistence, memory)
├── test_mc_builder_topk.py    # Top-M argpartition distractor ranking correctness
├── test_metrics.py            # S_q, ECE, Brier, calibration_at_buzz (top_p_trace, not g_trace)
├── test_ppo_buzzer.py         # PPOBuzzer training, run_episode traces, PPO calibration
├── test_ppo_t5.py             # T5 PPO training integration
├── test_qb_rl_bridge.py       # qb-rl backward compatibility (import paths work)
├── test_supervised_t5.py      # T5 supervised warm-start training
├── test_t5_policy.py          # T5PolicyModel forward/backward pass
└── test_text_wrapper.py       # TextObservationWrapper observation format
```

## Key Fixtures (`tests/conftest.py`)

| Fixture | Scope | Purpose |
|---------|-------|---------|
| `sample_mc_question` | function | Single MCQuestion with 4 options, 6 clue steps |
| `sample_config` | function | Minimal config dict (simple reward, sbert likelihood) |
| `sample_corpus` | function | 10 short texts for TF-IDF fitting |
| `sample_t5_model` | module | T5Likelihood with t5-small (loaded once per file) |
| `sample_tfidf_env` | function | TossupMCEnv with TF-IDF likelihood, 3 questions |

The `sample_t5_model` fixture uses `scope="module"` to avoid reloading the T5 model per test function.

## Running Tests

```bash
# Full suite
pytest

# Focused bridge/runtime checks
pytest tests/test_qb_rl_bridge.py tests/test_factories.py tests/test_ppo_buzzer.py

# Single file
pytest tests/test_environment.py -v

# Single test
pytest tests/test_metrics.py::test_system_score_basic -v
```

## Smoke Testing

Pipeline scripts support `--smoke` flag for fast end-to-end validation:

```bash
python scripts/build_mc_dataset.py --smoke
python scripts/run_baselines.py --smoke
python scripts/train_ppo.py --smoke
python scripts/evaluate_all.py --smoke
```

Smoke mode uses `configs/smoke.yaml` with reduced dataset size and training steps. Output goes to `artifacts/smoke/`.

## Test Patterns

- **Dataclass fixtures:** Tests construct minimal `MCQuestion` instances with known values
- **Environment tests:** Verify reset/step/done cycle, reward computation, observation shape
- **Likelihood tests:** Check output shape, dtype (float32), score ordering for known inputs
- **Agent tests:** Run single episodes and verify trace lengths, buzz decisions
- **Bridge tests:** Import from `qb_env.*` paths and verify they resolve to `qb_data.*` implementations

## No Mocking

Tests use real (lightweight) model instances:
- `TfIdfLikelihood` with small corpus (fast, no downloads)
- `t5-small` for T5 tests (60M params, downloads on first run)
- `SBERTLikelihood` with default model (downloads on first run)

No mock objects or monkeypatching is used. This keeps tests high-fidelity but means some tests require network access on first run.
