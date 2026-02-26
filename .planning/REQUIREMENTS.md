# Requirements: Quiz Bowl RL Buzzer (Unified)

**Defined:** 2026-02-25
**Core Value:** A principled, modular RL system that produces rigorous experimental results for the CS234 writeup

## v1 Requirements

Requirements for initial release. Each maps to roadmap phases.

### Data Pipeline

- [x] **DATA-01**: System loads quiz bowl questions from local CSV (QANTA format, clues separated by `|||`)
- [ ] **DATA-02**: System constructs K=4 multiple-choice questions with distractor generation
- [ ] **DATA-03**: Anti-artifact guards reject MC options with alias collision, token overlap >50%, or length ratio >3x
- [ ] **DATA-04**: Answer profiles built with leave-one-out exclusion per question
- [x] **DATA-05**: Dataset splits stratified by category (train 70% / val 15% / test 15%)
- [x] **DATA-06**: System can optionally load questions from HuggingFace datasets as fallback

### Environment

- [x] **ENV-01**: TossupMCEnv implements Gymnasium Env interface (reset/step/observation_space/action_space)
- [x] **ENV-02**: Action space is Discrete(K+1): action 0 = WAIT, actions 1..K = buzz with option i
- [x] **ENV-03**: Environment computes belief features per step: belief[K], top_p, margin, entropy, stability, progress
- [x] **ENV-04**: Configurable reward modes: time_penalty (R = ±1 - penalty*t/T), simple (±1), human_grounded
- [x] **ENV-05**: Environment accepts any LikelihoodModel for belief computation via factory

### Likelihood Models

- [x] **LIK-01**: Abstract LikelihoodModel ABC with `score(clue_prefix, option_profiles) -> ndarray[K]`
- [x] **LIK-02**: TfIdfLikelihood implementation using sklearn TfidfVectorizer
- [x] **LIK-03**: SBERTLikelihood implementation using sentence-transformers (all-MiniLM-L6-v2)
- [ ] **LIK-04**: T5Likelihood implementation using T5 encoder for semantic similarity scoring
- [ ] **LIK-05**: Embedding cache with text hashing for SBERT and T5 models
- [x] **LIK-06**: Factory function `build_likelihood_from_config()` constructs model from YAML

### Agents & Training

- [ ] **AGT-01**: MLP policy trained with SB3 PPO on belief feature observations
- [x] **AGT-02**: ThresholdBuzzer baseline (sweeps configurable thresholds on top_p)
- [x] **AGT-03**: AlwaysBuzzFinalBuzzer baseline (buzzes on last clue)
- [x] **AGT-04**: SoftmaxProfileBuzzer baseline with explicit scoring
- [x] **AGT-05**: SequentialBayesBuzzer baseline with Bayesian updates
- [x] **AGT-06**: All agents produce episode traces with c_trace (buzz probability) and g_trace (correctness)
- [ ] **AGT-07**: Smoke test mode (`--smoke`) for fast pipeline validation with small dataset

### Evaluation

- [ ] **EVAL-01**: S_q metric computation: system score = Σ(b_t × g_t) per episode
- [ ] **EVAL-02**: Calibration metrics: ECE (expected calibration error) and Brier score
- [ ] **EVAL-03**: Control experiment: choices-only (remove clues, verify ~25% random baseline)
- [ ] **EVAL-04**: Control experiment: shuffle (permute option order, verify no position bias)
- [ ] **EVAL-05**: Control experiment: alias substitution (swap answer text, verify robustness)
- [ ] **EVAL-06**: Comparison plots: calibration curves, entropy vs clue index, agent comparison tables
- [ ] **EVAL-07**: Per-category accuracy breakdown with summary statistics

### Configuration

- [x] **CFG-01**: YAML configuration system with sections: data, likelihood, environment, ppo, evaluation
- [x] **CFG-02**: Factory methods for all components: `make_env_from_config()`, `build_likelihood_from_config()`
- [ ] **CFG-03**: Four-stage pipeline scripts: build_mc_dataset, run_baselines, train_ppo, evaluate_all
- [x] **CFG-04**: CLI override support: `--config`, `--smoke`, key overrides

## v1 Stretch Goals

Include if time permits after core pipeline works.

- [ ] **STR-01**: T5PolicyModel with custom policy heads (wait/answer/value) as alternative to MLP policy
- [ ] **STR-02**: Supervised warm-start training for T5 policy on complete questions
- [ ] **STR-03**: Comparison experiment: T5-as-likelihood (MLP policy) vs T5-as-policy (end-to-end)

## v2 Requirements

Deferred to future work section of writeup.

### Advanced Features

- **ADV-01**: Human buzz position comparison and KL divergence
- **ADV-02**: Bootstrap confidence intervals on all metrics
- **ADV-03**: Cross-dataset generalization testing
- **ADV-04**: Category-specific likelihood models

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| Web UI or interactive demo | Not needed for CS234 writeup |
| OpenAI embedding likelihood | API cost, SBERT sufficient |
| Multi-GPU distributed training | Dataset fits on single GPU |
| Real-time game integration | Academic project only |
| Question generation | Different problem entirely |
| Custom PPO implementation | SB3 is battle-tested |
| Ensemble models | Time constraint |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| DATA-01 | Phase 1 | Complete |
| DATA-02 | Phase 1 | Pending |
| DATA-03 | Phase 1 | Pending |
| DATA-04 | Phase 1 | Pending |
| DATA-05 | Phase 1 | Complete |
| DATA-06 | Phase 1 | Complete |
| ENV-01 | Phase 2 | Complete |
| ENV-02 | Phase 2 | Complete |
| ENV-03 | Phase 2 | Complete |
| ENV-04 | Phase 2 | Complete |
| ENV-05 | Phase 2 | Complete |
| LIK-01 | Phase 2 | Complete |
| LIK-02 | Phase 2 | Complete |
| LIK-03 | Phase 2 | Complete |
| LIK-04 | Phase 3 | Pending |
| LIK-05 | Phase 3 | Pending |
| LIK-06 | Phase 2 | Complete |
| AGT-01 | Phase 4 | Pending |
| AGT-02 | Phase 3 | Complete |
| AGT-03 | Phase 3 | Complete |
| AGT-04 | Phase 3 | Complete |
| AGT-05 | Phase 3 | Complete |
| AGT-06 | Phase 3 | Complete |
| AGT-07 | Phase 4 | Pending |
| EVAL-01 | Phase 5 | Pending |
| EVAL-02 | Phase 5 | Pending |
| EVAL-03 | Phase 5 | Pending |
| EVAL-04 | Phase 5 | Pending |
| EVAL-05 | Phase 5 | Pending |
| EVAL-06 | Phase 5 | Pending |
| EVAL-07 | Phase 5 | Pending |
| CFG-01 | Phase 1 | Complete |
| CFG-02 | Phase 2 | Complete |
| CFG-03 | Phase 4 | Pending |
| CFG-04 | Phase 1 | Complete |
| STR-01 | Phase 6 | Pending |
| STR-02 | Phase 6 | Pending |
| STR-03 | Phase 6 | Pending |

**Coverage:**
- v1 requirements: 35 total
- Stretch goals: 3 total
- Mapped to phases: 38
- Unmapped: 0 ✓

---
*Requirements defined: 2026-02-25*
*Last updated: 2026-02-25 after roadmap creation*