# Project Roadmap: Quiz Bowl RL Buzzer (Unified)

**Project:** Quiz Bowl RL Buzzer (Unified System)
**Mode:** yolo
**Depth:** comprehensive
**Created:** 2026-02-25

## Phases

- [ ] **Phase 1: Data Pipeline Foundation** - Build MC dataset construction with anti-artifact guards and YAML configuration
- [ ] **Phase 2: Environment and Core Likelihood Models** - Implement Gymnasium environment with belief features and TF-IDF/SBERT likelihood models
- [ ] **Phase 3: Baseline Agents and T5 Likelihood** - Add baseline agents, T5 likelihood model, and episode trace generation
- [ ] **Phase 4: PPO Training Pipeline** - Train MLP policy with SB3 PPO and pipeline scripts
- [ ] **Phase 5: Evaluation Framework** - Complete S_q metric, control experiments, and visualization
- [ ] **Phase 6: T5 Policy Integration** - Optional T5 policy model with supervised warm-start

## Phase Details

### Phase 1: Data Pipeline Foundation
**Goal**: Users can load quiz bowl questions and construct valid multiple-choice questions with anti-artifact protection
**Depends on**: Nothing (first phase)
**Requirements**: DATA-01, DATA-02, DATA-03, DATA-04, DATA-05, DATA-06, CFG-01, CFG-04
**Success Criteria** (what must be TRUE):
  1. User can load quiz bowl questions from local CSV file with clues separated by `|||`
  2. System constructs K=4 multiple-choice questions with distractor generation that passes anti-artifact guards
  3. Answer profiles are built with leave-one-out exclusion per question
  4. Dataset splits are stratified by category (train 70% / val 15% / test 15%)
  5. YAML configuration system loads and can be overridden via CLI
**Plans**: 5 plans

Plans:
- [ ] 01-01-PLAN.md — Create core data structures and CSV loading
- [x] 01-02-PLAN.md — Set up YAML configuration system ✓
- [ ] 01-03-PLAN.md — Port MCBuilder and answer profiles with guards
- [ ] 01-04-PLAN.md — Implement stratified splits and HuggingFace loader
- [ ] 01-05-PLAN.md — Create main dataset construction script

### Phase 2: Environment and Core Likelihood Models
**Goal**: Users can run quiz bowl episodes in a Gymnasium environment with belief-based observations
**Depends on**: Phase 1
**Requirements**: ENV-01, ENV-02, ENV-03, ENV-04, ENV-05, LIK-01, LIK-02, LIK-03, LIK-06, CFG-02
**Success Criteria** (what must be TRUE):
  1. TossupMCEnv implements full Gymnasium interface and can be instantiated via factory
  2. Action space properly implements Discrete(K+1) with WAIT and buzz actions
  3. Environment computes all belief features (belief[K], top_p, margin, entropy, stability, progress)
  4. User can configure different reward modes (time_penalty, simple, human_grounded)
  5. TF-IDF and SBERT likelihood models produce valid belief distributions
**Plans**: TBD

### Phase 3: Baseline Agents and T5 Likelihood
**Goal**: Users can run baseline agents and leverage T5 for semantic similarity scoring
**Depends on**: Phase 2
**Requirements**: AGT-02, AGT-03, AGT-04, AGT-05, AGT-06, LIK-04, LIK-05
**Success Criteria** (what must be TRUE):
  1. All four baseline agents (Threshold, AlwaysBuzzFinal, SoftmaxProfile, SequentialBayes) produce valid episodes
  2. T5 likelihood model computes semantic similarity scores for belief updates
  3. Embedding cache reduces redundant T5 computations
  4. All agents generate episode traces with c_trace (buzz probability) and g_trace (correctness)
**Plans**: TBD

### Phase 4: PPO Training Pipeline
**Goal**: Users can train an MLP policy with SB3 PPO and run smoke tests for validation
**Depends on**: Phase 3
**Requirements**: AGT-01, AGT-07, CFG-03
**Success Criteria** (what must be TRUE):
  1. MLP policy trains successfully with SB3 PPO on belief feature observations
  2. Smoke test mode runs complete pipeline in under 2 minutes with small dataset
  3. Four-stage pipeline scripts (build_mc, run_baselines, train_ppo, evaluate_all) execute without errors
  4. Training produces checkpoints that can be loaded for evaluation
**Plans**: TBD

### Phase 5: Evaluation Framework
**Goal**: Users can evaluate agents with S_q metric, control experiments, and comprehensive visualizations
**Depends on**: Phase 4
**Requirements**: EVAL-01, EVAL-02, EVAL-03, EVAL-04, EVAL-05, EVAL-06, EVAL-07
**Success Criteria** (what must be TRUE):
  1. S_q metric correctly computes system score = Σ(b_t × g_t) per episode
  2. Calibration metrics (ECE and Brier score) quantify uncertainty quality
  3. Control experiments (choices-only, shuffle, alias) verify agent uses clues properly
  4. Comparison plots and tables show relative performance of all agents
  5. Per-category accuracy breakdown reveals performance patterns
**Plans**: TBD

### Phase 6: T5 Policy Integration
**Goal**: Users can train and compare T5-based policy with custom heads as alternative to MLP
**Depends on**: Phase 2
**Requirements**: STR-01, STR-02, STR-03
**Success Criteria** (what must be TRUE):
  1. T5PolicyModel with custom policy heads (wait/answer/value) trains successfully
  2. Supervised warm-start on complete questions improves convergence
  3. Comparison experiment shows performance difference between T5-as-likelihood vs T5-as-policy
**Plans**: TBD

## Progress

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Data Pipeline Foundation | 1/5 | In progress | - |
| 2. Environment and Core Likelihood Models | 0/0 | Not started | - |
| 3. Baseline Agents and T5 Likelihood | 0/0 | Not started | - |
| 4. PPO Training Pipeline | 0/0 | Not started | - |
| 5. Evaluation Framework | 0/0 | Not started | - |
| 6. T5 Policy Integration | 0/0 | Not started | - |

## Success Metrics

- **Phase Success**: Phase is complete when all success criteria are met
- **Project Success**: Working RL system with S_q evaluation and CS234 writeup
- **Quality Indicators**:
  - S_q score improvement over baselines
  - Control experiments pass (choices-only ~25%, no position bias)
  - Calibration error < 0.1
  - Smoke tests complete in < 2 minutes

## Dependencies

### Phase Dependencies
```
Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 5
                ↘                        ↗
                  Phase 6 ---------------
```

### Key Integration Points
- Phase 2 defines LikelihoodModel interface that Phase 3 implements for T5
- Phase 3 agents must produce traces that Phase 5 uses for S_q computation
- Phase 6 is independent path after Phase 2 (alternative to Phase 3-4 pipeline)

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Scope explosion with tight deadline | HIGH | Focus on Phase 1-5 critical path, defer Phase 6 |
| T5 memory requirements | MEDIUM | Support T5-base (220M) as fallback option |
| Belief state collapse in early training | MEDIUM | Pre-compute answer profiles, add margin threshold |
| Observation space incompatibility | HIGH | Clear interfaces (BeliefObservation vs TextObservation) |

---
*Roadmap created: 2026-02-25*
*Phase 1 planned: 2026-02-25*
*Next: `/gsd:execute-phase 1`*