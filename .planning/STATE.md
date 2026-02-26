---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
current_plan: 2 of 4
status: executing
last_updated: "2026-02-26T02:28:30.456Z"
progress:
  total_phases: 2
  completed_phases: 1
  total_plans: 9
  completed_plans: 7
  percent: 78
---

# Project State: Quiz Bowl RL Buzzer (Unified)

**Project:** Quiz Bowl RL Buzzer (Unified System)
**Last Updated:** 2026-02-25
**Current Sprint:** Milestone 1

## Project Reference

### Core Value
A principled, modular RL system that produces rigorous experimental results — S_q scoring, baseline comparisons, calibration metrics, and ablation controls — for the CS234 writeup, while supporting both lightweight belief-feature policies and T5-based semantic policies.

### Current Focus
Building unified system by merging qb-rl's modular architecture with qanta-buzzer's T5 integration. Critical path: data pipeline → environment → T5 likelihood → MLP policy → evaluation.

## Current Position

**Phase:** 2 - Environment and Core Likelihood Models
**Current Plan:** 2 of 4
**Status:** In Progress
**Progress:** [████████░░] 78%

### Active Work
- Completed: Plan 02-01 (Belief features and LikelihoodModel ABC)
- Completed: Plan 02-02 (TF-IDF and SBERT likelihood models with factory)
- Next: Plan 02-03 (TossupMCEnv Gymnasium environment)

### Completed Phases
1. Phase 01 - Data Pipeline Foundation (5/5 plans complete)

### Upcoming Phases
1. Phase 2: Environment and Core Likelihood Models
2. Phase 3: Baseline Agents and T5 Likelihood
3. Phase 4: PPO Training Pipeline
4. Phase 5: Evaluation Framework
5. Phase 6: T5 Policy Integration (optional)

## Performance Metrics

### Velocity
- **Plans Completed (24h):** 1
- **Plans Completed (7d):** 1
- **Average Plan Duration:** 6 minutes

### Quality
- **First-Try Success Rate:** N/A
- **Revision Count:** 0
- **Test Coverage:** N/A

### Time Distribution
- **Planning:** 100% (roadmap creation)
- **Implementation:** 0%
- **Debugging:** 0%
- **Documentation:** 0%

## Accumulated Context

### Key Decisions
| Decision | Rationale | Date |
|----------|-----------|------|
| Derive 6 phases from requirements | Natural groupings: data, environment, baselines+T5, training, evaluation, optional T5 policy | 2026-02-25 |
| Phase 1-5 critical path | Delivers MLP policy with T5 likelihood - core contribution | 2026-02-25 |
| Phase 6 optional | T5 policy is stretch goal given tight deadline | 2026-02-25 |
| Support T5-base fallback | Memory constraints may require smaller model | 2026-02-25 |
| Unified column name support | Accept both QANTA (Text/Answer) and generic formats | 2026-02-25 |
| Hash-based ID generation | Use MD5 for deterministic unique IDs when not provided | 2026-02-25 |
| Cumulative prefix pre-computation | Build all prefixes during loading to avoid repeated operations | 2026-02-25 |
| Use YAML for configuration | Human-readable, standard in ML projects, supports comments | 2026-02-25 |
| Dot notation for CLI overrides | Easy experimentation without editing files (e.g., data.K=5) | 2026-02-25 |
| Category-based stratification | Preserve category distribution across train/val/test splits | 2026-02-25 |
| HuggingFace as optional fallback | Provide alternative data source when CSV unavailable | 2026-02-25 |
| Match existing dataclass structure | Fixed field name inconsistencies between plans and implementation | 2026-02-25 |
| Fix CSV paths in configs | Updated configs to point to root directory where questions.csv exists | 2026-02-25 |
| Use _grouped attribute | AnswerProfileBuilder stores data in _grouped, not profiles | 2026-02-25 |
| Add .gitignore for generated data | Prevent large JSON files from being committed | 2026-02-25 |
| Port qb-rl features.py exactly | Maintain compatibility with downstream environment and agent plans | 2026-02-25 |
| LikelihoodModel returns raw scores | Environment applies softmax with temperature (separation of concerns) | 2026-02-25 |
| Factory supports dual config keys | Both sbert_name and embedding_model keys for cross-project compat | 2026-02-25 |
| Lazy imports for optional deps | sklearn and sentence_transformers imported inside class constructors | 2026-02-25 |

### Architecture Decisions
- Four-layer modular architecture: Pipeline → Agent → Environment → Model
- Dual policy support: MLP on belief features vs T5 end-to-end
- T5 serves dual purpose: likelihood model and optional policy
- YAML configuration with factory methods
- Gymnasium-compliant environment
- SB3 PPO for MLP policy, custom for T5 policy

### Known Issues
None yet

### Technical Debt
- Two existing codebases to merge (qb-rl and qanta-buzzer)
- Different observation spaces (numeric beliefs vs text)
- Memory requirements for T5-large (may need T5-base)

### Performance Bottlenecks
None identified yet

## Session Continuity

### Last Session Summary
- Executed Plan 02-02: TF-IDF and SBERT Likelihood Models
- Added TfIdfLikelihood with corpus fitting and cosine similarity scoring
- Added SBERTLikelihood with SentenceTransformer and L2-normalized embeddings
- Created build_likelihood_from_config() factory for YAML-driven construction
- Updated models/__init__.py with all likelihood exports
- 1 deviation: factory handles both sbert_name and embedding_model config keys

### Next Session Priority
1. Execute Plan 02-03: TossupMCEnv Gymnasium environment
2. Execute Plan 02-04: Factory functions and pytest test scaffolding
3. Begin Phase 3: Baseline Agents and T5 Likelihood

### Context for Next Claude
This is a CS234 final project due this week. We're merging two existing codebases:
- qb-rl: Has the modular architecture we want (Gymnasium env, belief features, S_q metric, baselines)
- qanta-buzzer: Has T5 integration we need (encoder, policy heads, supervised warm-start)

The novel contribution is using T5 as a likelihood model to compute beliefs for an MLP policy, then comparing with T5 as an end-to-end policy. Phase 1-5 is the critical path for the deadline. Phase 6 (T5 policy) is optional if time permits.

Key risks to watch:
- Scope explosion (stick to critical path)
- Memory issues with T5-large (have T5-base ready)
- Observation space incompatibility (keep interfaces clean)
- Belief state collapse (pre-compute answer profiles)

### Open Questions
1. Should we start with existing qanta-buzzer data loading or rebuild from qb-rl?
2. Is supervised warm-start necessary for T5 policy or just helpful?
3. What's the optimal time penalty coefficient for reward shaping?
4. Should we implement all 4 baselines or just threshold for MVP?

### Environment State
- Working directory: `/Users/ankit.aggarwal/Dropbox/Stanford/CS234/final_project/qanta-buzzer`
- Python environment: Not yet configured (needs 3.11+)
- Git status: Roadmap files created, not yet committed
- Dependencies: Not yet installed

---
*State file initialized: 2026-02-25*
*Last update: 2026-02-25 (Plan 02-01 completed)*