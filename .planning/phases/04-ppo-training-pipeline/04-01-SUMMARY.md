---
phase: 04-ppo-training-pipeline
plan: 01
subsystem: agents
tags: [ppo, stable-baselines3, sb3, mlp-policy, belief-features, s_q-metric, yaml-config]

# Dependency graph
requires:
  - phase: 03-baseline-agents-and-t5-likelihood
    provides: "TossupMCEnv, LikelihoodModel, TfIdfLikelihood, baseline agents"
provides:
  - "scripts/_common.py — shared utilities for config, JSON, artifact paths"
  - "agents/ppo_buzzer.py — PPOBuzzer wrapper with episode trace generation"
  - "tests/test_ppo_buzzer.py — 19 unit tests for utilities and PPOBuzzer"
  - "sample_tfidf_env conftest fixture for fast agent testing"
affects: [04-02, 04-03, 05-evaluation-framework]

# Tech tracking
tech-stack:
  added: [stable-baselines3, torch]
  patterns: [sb3-wrapper-pattern, episode-trace-generation, lazy-imports-for-optional-deps]

key-files:
  created:
    - scripts/_common.py
    - agents/ppo_buzzer.py
    - tests/test_ppo_buzzer.py
  modified:
    - agents/__init__.py
    - tests/conftest.py

key-decisions:
  - "Lazy import for PPOBuzzer in agents/__init__.py to avoid requiring SB3 for baseline-only runs"
  - "Direct port from qb-rl with only import path changes (qb_env -> qb_data)"
  - "TF-IDF likelihood in sample_tfidf_env fixture for fast test execution (2.4s total)"

patterns-established:
  - "SB3 wrapper pattern: PPOBuzzer wraps PPO with custom episode execution for trace generation"
  - "Episode trace pattern: c_trace, g_trace, entropy_trace for S_q computation"
  - "Shared utilities pattern: _common.py centralizes config/JSON/path functions for pipeline scripts"

requirements-completed: [AGT-01, AGT-07]

# Metrics
duration: 5min
completed: 2026-02-26
---

# Phase 04 Plan 01: PPO Infrastructure Summary

**PPOBuzzer wrapping SB3 PPO with episode trace generation (c_trace, g_trace) for S_q metric, plus shared _common.py utilities for pipeline scripts**

## Performance

- **Duration:** 5 min
- **Started:** 2026-02-26T04:17:50Z
- **Completed:** 2026-02-26T04:23:04Z
- **Tasks:** 3
- **Files modified:** 5

## Accomplishments
- Created scripts/_common.py with config loading, JSON serialization, MCQuestion deserialization, and path utilities
- Created agents/ppo_buzzer.py with PPOBuzzer class wrapping SB3 PPO and PPOEpisodeTrace dataclass
- Episode trace generation supports S_q computation: c_trace (buzz probability), g_trace (correctness), entropy_trace
- 19 unit tests covering all utilities and PPOBuzzer methods, passing in 2.4 seconds
- Made PPOBuzzer import lazy in agents/__init__.py for environments without SB3

## Task Commits

Each task was committed atomically:

1. **Task 1: Port _common.py utilities from qb-rl** - `068a5d0` (feat)
2. **Task 2: Port PPOBuzzer class from qb-rl** - `80f6796` (feat)
3. **Task 3: Create unit tests for _common and PPOBuzzer** - `a91d9ac` (test)

## Files Created/Modified
- `scripts/_common.py` - Shared utilities: load_config, save_json, load_json, mc_question_from_dict, load_mc_questions, ensure_dir, to_serializable
- `agents/ppo_buzzer.py` - PPOBuzzer wrapper with train/save/load/run_episode and PPOEpisodeTrace dataclass
- `tests/test_ppo_buzzer.py` - 19 tests for utilities and PPOBuzzer
- `agents/__init__.py` - Added lazy PPOBuzzer/PPOEpisodeTrace exports
- `tests/conftest.py` - Added sample_tfidf_env fixture

## Decisions Made
- Lazy import for PPOBuzzer in agents/__init__.py: Avoids requiring stable_baselines3 just to import baseline agents. Uses module-level __getattr__ for on-demand loading.
- Direct port from qb-rl: Only changed import paths (qb_env.mc_builder -> qb_data.mc_builder), preserving exact logic for compatibility.
- TF-IDF for test fixture: 2.4s execution vs 5+ seconds with neural models. Tests focus on agent logic, not model quality.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Installed stable-baselines3 dependency**
- **Found during:** Task 2 (PPOBuzzer import verification)
- **Issue:** stable_baselines3 package not installed in the environment
- **Fix:** Ran `pip install stable-baselines3` which also installed gymnasium, matplotlib, pandas
- **Files modified:** None (package manager change only)
- **Verification:** Import succeeds, PPOBuzzer instantiates

**2. [Rule 3 - Blocking] Installed pytest in .venv**
- **Found during:** Task 3 (test execution)
- **Issue:** pytest only available in homebrew Python 3.11, not in project .venv (Python 3.12)
- **Fix:** Ran `pip install pytest` in the active venv
- **Files modified:** None (package manager change only)
- **Verification:** All 19 tests pass via `python -m pytest`

**3. [Rule 2 - Missing Critical] Added sample_tfidf_env fixture to conftest.py**
- **Found during:** Task 3 (test writing)
- **Issue:** Plan references sample_tfidf_env fixture but it didn't exist in conftest.py
- **Fix:** Created fixture providing TossupMCEnv with TF-IDF likelihood and 3 sample questions
- **Files modified:** tests/conftest.py
- **Verification:** All tests using the fixture pass
- **Committed in:** a91d9ac (Task 3 commit)

---

**Total deviations:** 3 auto-fixed (2 blocking dependencies, 1 missing fixture)
**Impact on plan:** All auto-fixes necessary for test execution. No scope creep.

## Issues Encountered
None beyond the auto-fixed deviations above.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- PPOBuzzer ready for integration into training script (04-02)
- _common.py utilities ready for baseline runner and evaluation scripts (04-02, 04-03)
- Episode trace generation supports S_q metric computation in evaluation framework (Phase 05)
- Full test suite: 134 tests passing (115 existing + 19 new)

## Self-Check: PASSED

All files exist: scripts/_common.py, agents/ppo_buzzer.py, tests/test_ppo_buzzer.py
All commits exist: 068a5d0, 80f6796, a91d9ac

---
*Phase: 04-ppo-training-pipeline*
*Completed: 2026-02-26*
