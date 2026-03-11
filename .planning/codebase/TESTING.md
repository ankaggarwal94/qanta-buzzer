# Testing Patterns

**Analysis Date:** 2026-03-08

## Test Framework

**Runner:** `pytest`

**Primary suites:**
- `tests/test_features.py`: belief-feature extraction
- `tests/test_likelihoods.py`: TF-IDF, SBERT, T5, and optional OpenAI likelihood behavior
- `tests/test_environment.py`: `TossupMCEnv` reset/step semantics and reward modes
- `tests/test_agents.py`: threshold, Bayesian, and floor baselines
- `tests/test_ppo_buzzer.py`: PPO agent utilities and trace handling
- `tests/test_supervised_t5.py`, `tests/test_ppo_t5.py`, `tests/test_t5_policy.py`, `tests/test_text_wrapper.py`: T5 policy track
- `tests/test_qb_rl_bridge.py`: qb-rl compatibility shims and loader/config bridge

**Smoke validation:** the canonical smoke workflow exercises the full modular pipeline:

```bash
python scripts/build_mc_dataset.py --smoke
python scripts/run_baselines.py --smoke
python scripts/train_ppo.py --smoke
python scripts/evaluate_all.py --smoke
```

**Walkthrough validation:** the generated walkthrough is verified with Showboat:

```bash
uvx showboat verify walkthrough.md
```

## Test File Organization

**Location:** canonical tests live under `tests/` with shared fixtures in `tests/conftest.py`.

```text
qanta-buzzer/
├── tests/
│   ├── conftest.py
│   ├── test_agents.py
│   ├── test_build_mc_dataset.py
│   ├── test_environment.py
│   ├── test_factories.py
│   ├── test_likelihoods.py
│   ├── test_qb_rl_bridge.py
│   └── ...
├── scripts/
├── qb_data/
├── qb_env/
└── models/
```

**Naming:**
- Test modules use the `test_*.py` pattern.
- Suites are organized by subsystem rather than by historical prototype file.
- Shared fixtures live in `tests/conftest.py`; heavier model fixtures are module-scoped to keep runtime reasonable.

## Execution Hierarchy

**Fastest path first:**
1. Run the narrowest relevant pytest module or test case.
2. Run the smoke pipeline if a change affects cross-module orchestration.
3. Re-verify `walkthrough.md` if Showboat content or recorded smoke evidence changes.

**Common commands:**

```bash
venv/bin/python -m pytest
venv/bin/python -m pytest tests/test_agents.py -q
venv/bin/python -m pytest tests/test_build_mc_dataset.py tests/test_qb_rl_bridge.py -q
```

## Test Patterns

**Data and environment tests:**
- Prefer TF-IDF-based fixtures for speed and determinism.
- Build small synthetic or smoke-sized question sets rather than loading the full corpus.
- Assert trace lengths, probability ranges, reward behavior, and anti-artifact guard outcomes directly.

**Agent and math tests:**
- Treat warnings as errors for numerical-stability checks.
- Verify `c_trace`, `g_trace`, entropy, and buzz timing invariants explicitly.
- Keep agent tests focused on behavioral contracts, not heavyweight training.

**T5-track tests:**
- Keep fixtures scoped so expensive model setup happens once per module.
- Test architecture wiring, text observation conversion, and trainer bookkeeping without requiring a full end-to-end training run.

## Legacy Notes

The older root-level prototype path still contains ad-hoc scripts and verification helpers, but the current branch’s canonical validation strategy is `pytest` plus smoke scripts. Historical one-off scripts should be treated as supplementary inspection tools, not the primary regression suite.

## Coverage and Gaps

- There is no enforced coverage threshold in CI.
- OpenAI-backed paths are tested with mocks rather than live API calls.
- Full T5 training is documented and partially unit-tested, but not treated as a routine verification step.

---

*Testing analysis: 2026-03-08*
