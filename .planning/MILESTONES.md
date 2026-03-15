# Milestones

## v1.0 Quiz Bowl RL Buzzer (Shipped: 2026-03-13)

**Phases completed:** 6 phases, 20 plans
**Requirements:** 40/40 (37 core + 3 stretch)
**Lines of code:** 22,464 Python
**Commits:** 196
**Timeline:** 17 days (2026-02-23 to 2026-03-13)

**Key accomplishments:**
1. Modular data pipeline with QANTA CSV loader, K=4 MC question builder with 4 anti-artifact guards, stratified splits, and HuggingFace fallback
2. Gymnasium POMDP environment with belief-based observations, 3 reward modes, and pluggable likelihood models (TF-IDF, SBERT, T5, OpenAI)
3. Four baseline agents (ThresholdBuzzer, SoftmaxProfileBuzzer, SequentialBayesBuzzer, AlwaysBuzzFinalBuzzer) with full S_q/ECE/Brier metrics
4. PPO training pipeline with SB3 on belief features, configurable reward shaping, and smoke mode for fast validation
5. Evaluation framework with S_q scoring, calibration metrics, 3 control experiments (choices-only, shuffle, alias substitution), and comparison plots
6. T5 policy integration with custom policy heads (wait/answer/value), supervised warm-start, custom PPO, and comparison experiment

**Post-milestone optimization campaign (10 quick tasks):**
- Repo-contract scaffolding (AGENTS.md, ci.sh, manual-smoke.sh)
- 7 ranked performance optimizations (precomputed beliefs, embedding cache persistence, collapsed baseline sweeps, profile memoization, top-M distractor ranking, TF-IDF caching, precomputed shuffle control)
- Final verification handoff and ci.sh fix

**Archives:**
- `.planning/milestones/v1.0-ROADMAP.md`
- `.planning/milestones/v1.0-REQUIREMENTS.md`

---
