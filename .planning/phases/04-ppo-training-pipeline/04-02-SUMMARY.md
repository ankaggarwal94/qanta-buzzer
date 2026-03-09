---
phase: 04-ppo-training-pipeline
plan: 02
subsystem: evaluation
tags: [metrics, baselines, s_q, ece, brier, tfidf, threshold-sweep]

# Dependency graph
requires:
  - phase: 03-baseline-agents
    provides: ThresholdBuzzer, SoftmaxProfileBuzzer, SequentialBayesBuzzer, AlwaysBuzzFinalBuzzer agents
provides:
  - evaluation/metrics.py with system_score, summarize_buzz_metrics, calibration_at_buzz
  - scripts/run_baselines.py baseline orchestration with 4 agent types
  - bayesian config section with threshold_sweep and alpha parameters
  - baseline_summary.json artifact with accuracy, S_q, ECE, Brier metrics
affects: [04-03-PLAN, 05-evaluation-framework]

# Tech tracking
tech-stack:
  added: [numpy (metrics computation)]
  patterns: [qb-rl port pattern (adapt imports only), artifact directory convention (artifacts/{smoke,main}/)]

key-files:
  created:
    - evaluation/__init__.py
    - evaluation/metrics.py
    - scripts/run_baselines.py
    - scripts/__init__.py
  modified:
    - configs/default.yaml
    - configs/smoke.yaml
    - agents/__init__.py

key-decisions:
  - "TF-IDF for smoke mode baselines: 0.9s execution vs estimated 30s+ with T5-small"
  - "Lazy import PPOBuzzer in agents/__init__.py to avoid hard stable_baselines3 dependency"
  - "Fallback MC dataset path: checks data/processed/ when artifacts/ not found"
  - "3 thresholds in smoke (vs 5 in default): reduces sweep time for quick validation"

patterns-established:
  - "Artifact output convention: artifacts/{smoke,main}/ subdirectories for all pipeline scripts"
  - "Config-driven baseline sweep: bayesian.threshold_sweep and likelihood.beta parameters"
  - "Evaluation metrics port pattern: exact qb-rl logic with _to_dict adapter for dataclass flexibility"

requirements-completed: [CFG-03, AGT-07]

# Metrics
duration: 5min
completed: 2026-02-26
---

# Phase 4 Plan 2: Baseline Agent Orchestration Summary

**Evaluation metrics (S_q, ECE, Brier) and baseline orchestration script running 4 agents across threshold sweep with smoke test in <1 second**

## Performance

- **Duration:** 5 min
- **Started:** 2026-02-26T04:17:43Z
- **Completed:** 2026-02-26T04:23:22Z
- **Tasks:** 3
- **Files modified:** 7

## Accomplishments
- Ported evaluation metrics module from qb-rl: system_score (S_q), ECE, Brier score, buzz metrics aggregation
- Created run_baselines.py orchestrating ThresholdBuzzer, SoftmaxProfile, SequentialBayes, AlwaysBuzzFinal across configurable threshold sweep
- Smoke test completed in 0.9 seconds (44 questions, 3 thresholds, TF-IDF model) producing all 5 expected artifact files
- All 4 agent types generate valid episode traces with accuracy=0.386, S_q scores ranging 0.053-0.386 depending on agent/threshold

## Task Commits

Each task was committed atomically:

1. **Task 1: Port evaluation metrics from qb-rl** - `81b0312` (feat)
2. **Task 2: Create run_baselines.py script** - `27b8220` (feat)
3. **Task 3: Smoke test run_baselines.py** - (verification only, no code changes)

## Files Created/Modified
- `evaluation/__init__.py` - Package init with public API exports
- `evaluation/metrics.py` - system_score, summarize_buzz_metrics, calibration_at_buzz, ECE, Brier
- `scripts/run_baselines.py` - Baseline orchestration: 4 agents, threshold sweep, artifact persistence
- `scripts/__init__.py` - Package init for script imports
- `configs/default.yaml` - Added bayesian section (threshold_sweep, alpha) and likelihood.beta
- `configs/smoke.yaml` - Added bayesian section (3 thresholds) and switched to tfidf for speed
- `agents/__init__.py` - Lazy import for PPOBuzzer to avoid hard stable_baselines3 dependency

## Decisions Made
- Used TF-IDF for smoke mode baselines instead of T5-small: 0.9s vs estimated 30s+ execution time
- Made PPOBuzzer import lazy in agents/__init__.py to allow baseline-only runs without stable_baselines3
- Added fallback path logic: run_baselines.py checks data/processed/ when artifacts/smoke/ not found
- Reduced smoke threshold sweep to 3 values (0.5, 0.7, 0.9) vs 5 in default config

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] PPOBuzzer import causing ModuleNotFoundError**
- **Found during:** Task 2 (run_baselines.py --help verification)
- **Issue:** agents/__init__.py eagerly imports PPOBuzzer which requires stable_baselines3 (not installed in current venv)
- **Fix:** Changed to lazy __getattr__ import pattern in agents/__init__.py
- **Files modified:** agents/__init__.py
- **Verification:** `python scripts/run_baselines.py --help` succeeds
- **Committed in:** 27b8220 (Task 2 commit)

**2. [Rule 3 - Blocking] Missing bayesian config section in YAML files**
- **Found during:** Task 2 (config analysis)
- **Issue:** Both default.yaml and smoke.yaml lacked the bayesian section needed by run_baselines.py (threshold_sweep, alpha)
- **Fix:** Added bayesian section to both config files matching qb-rl structure
- **Files modified:** configs/default.yaml, configs/smoke.yaml
- **Verification:** `python scripts/run_baselines.py --smoke` reads config successfully
- **Committed in:** 27b8220 (Task 2 commit)

**3. [Rule 3 - Blocking] Missing likelihood.beta config parameter**
- **Found during:** Task 2 (config analysis)
- **Issue:** likelihood section in both configs lacked beta (softmax temperature) used by all baseline agents
- **Fix:** Added beta: 5.0 to likelihood section in both YAML files
- **Files modified:** configs/default.yaml, configs/smoke.yaml
- **Verification:** beta correctly read as 5.0 during smoke test
- **Committed in:** 27b8220 (Task 2 commit)

---

**Total deviations:** 3 auto-fixed (3 blocking)
**Impact on plan:** All fixes necessary for run_baselines.py to function. No scope creep.

## Issues Encountered
None beyond the blocking dependency fixes documented above.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Evaluation metrics module ready for integration with PPO training (evaluate_all.py)
- Baseline summary artifacts provide performance floor for PPO comparison
- Config structure established for all pipeline scripts (bayesian, likelihood.beta)
- Missing: scripts/_common.py already exists from prior work; PPOBuzzer requires stable_baselines3 installation for Plan 04-01 and 04-03

## Self-Check: PASSED

All files verified present:
- evaluation/__init__.py, evaluation/metrics.py
- scripts/run_baselines.py, scripts/__init__.py
- artifacts/smoke/baseline_summary.json
- .planning/phases/04-ppo-training-pipeline/04-02-SUMMARY.md

All commits verified:
- 81b0312 (Task 1: evaluation metrics)
- 27b8220 (Task 2: run_baselines.py + config)

---
*Phase: 04-ppo-training-pipeline*
*Completed: 2026-02-26*
