# Project Retrospective

## Milestone: v1.0 — Quiz Bowl RL Buzzer

**Shipped:** 2026-02-26
**Phases:** 6 | **Plans:** 20 | **Tests:** 204

### What Was Built
- Complete data pipeline with anti-artifact guards and YAML configuration
- Gymnasium POMDP environment with belief features and 3 reward modes
- 3 likelihood models (TF-IDF, SBERT, T5) with pluggable factory
- 4 baseline agents with episode trace generation for S_q
- MLP policy trained with SB3 PPO on belief features
- T5 policy with custom heads (wait/answer/value), supervised warm-start, custom PPO with GAE
- Full evaluation framework: S_q, ECE, Brier, 3 controls, per-category breakdown, comparison plots
- Four-stage pipeline with --smoke mode (<15s)

### What Worked
- **Direct port from qb-rl**: Porting proven code rather than reimplementing saved significant time and avoided bugs
- **Wave-based parallel execution**: Plans within waves ran simultaneously, cutting phase execution time roughly in half
- **Smoke testing early**: The --smoke flag validated integration at every stage before committing to long runs
- **Comprehensive pytest suite**: 204 tests caught regressions immediately; test-driven tasks prevented debugging spirals
- **YAML configuration**: Single config change enables experiment sweeps without code modifications

### What Was Inefficient
- **Phase 1 summary gap**: One agent (01-03) completed all code but didn't write its SUMMARY.md — required manual tracking
- **Sequential waves in Phase 6**: Three sequential waves (model → wrapper → training) couldn't parallelize due to strict dependencies
- **API token expiration**: One agent hit Bedrock token expiration mid-execution, requiring re-run
- **Bedrock model alias confusion**: `opus` alias resolved to wrong model for subagents; had to use `sonnet` explicitly

### Patterns Established
- Port-first strategy: Read reference implementation, adapt imports, verify with tests
- Lean gap-filling phases: Phase 5 was 2 plans filling 3 gaps (not rebuilding what Phase 4 built)
- Factory pattern everywhere: `build_likelihood_from_config()`, `make_env_from_config()` enable YAML-driven experiments
- Module-scoped T5 fixtures: Load model once per test file, not per test — 10x faster test suite

### Key Lessons
1. Build the pipeline end-to-end first (even with simple models), then upgrade components
2. Anti-artifact guards are essential — without them, agents learn surface patterns instead of semantics
3. Belief features (margin, entropy, stability) are more informative than raw likelihood scores
4. Supervised warm-start is important for T5 policy — PPO from scratch is unstable with 770M params
5. The S_q metric better captures buzzer quality than raw accuracy

### Cost Observations
- Model mix: 100% Opus 4.6 for main conversation, Sonnet for subagents (researcher, planner, checker, executor)
- Haiku used only for codebase mapping
- Notable: Parallel Wave 1 execution (2 plans simultaneously) was the most cost-efficient pattern

---

## Cross-Milestone Trends

| Metric | v1.0 |
|--------|------|
| Phases | 6 |
| Plans | 20 |
| Tests | 204 |
| LOC | 16,675 |
| Python files | 61 |
| Commits | 108 |
| Timeline | 2 days |
