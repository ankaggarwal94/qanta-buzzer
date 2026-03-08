# Project Research Summary

**Project:** Quiz Bowl RL Buzzer (Unified System)
**Domain:** Reinforcement Learning for Calibrated Question Answering
**Researched:** 2026-02-24
**Confidence:** HIGH

## Executive Summary

This is an academic CS234 final project merging two existing codebases (qb-rl and qanta-buzzer) to create a unified quiz bowl RL system with T5 integration. The domain has established patterns: quiz bowl agents learn when to "buzz" on incrementally revealed clues in a POMDP environment, evaluated using the S_q metric (system score = Σ(buzz_probability × correctness)). Experts build these systems with modular architectures separating likelihood models (TF-IDF, SBERT, T5), RL policies (PPO), and environments (Gymnasium).

The recommended approach is a **four-layer modular architecture** with dual policy support: lightweight MLP on belief features (fast baseline) and T5-large encoder with policy heads (semantic understanding). Use Stable-Baselines3 PPO for the MLP policy and adapt qanta-buzzer's custom T5 policy implementation. The critical integration point is implementing T5 as a LikelihoodModel to compute beliefs for the MLP policy, creating a clean comparison between T5-as-likelihood vs T5-as-policy. Start with qb-rl's proven infrastructure (Gymnasium environment, anti-artifact guards, S_q computation) and add qanta-buzzer's T5 integration as the novel contribution.

Key risks are scope explosion (too many features to integrate) and observation space incompatibility (qb-rl uses numeric belief vectors, qanta-buzzer uses text strings). Mitigate by focusing on a minimal viable integration first: qb-rl environment + T5 likelihood model + basic PPO training + S_q evaluation. Defer supervised warm-start, multiple baselines, and advanced likelihood models until core pipeline works. The tight deadline (one week) demands ruthless prioritization and smoke testing at every step.

## Key Findings

### Recommended Stack

The stack is Python 3.11+ with PyTorch 2.3.0+, Gymnasium 1.1.0+, Stable-Baselines3 2.6.0+, and Transformers 4.45.0+ for T5 integration. This combination provides production-ready PPO implementation, standardized RL environments, and excellent T5 support. Critical version constraint: NumPy <2.0.0 (NumPy 2.0 breaks many dependencies). Use sentence-transformers for SBERT embeddings (no API costs), PyYAML for configuration, and standard matplotlib/seaborn for academic plots.

**Core technologies:**
- **PyTorch 2.3.0+**: Neural networks — industry standard for research, better debugging than TF, MPS support for Mac
- **Gymnasium 1.1.0+**: RL environment API — successor to OpenAI Gym, actively maintained, SB3 integration
- **Stable-Baselines3 2.6.0+**: PPO implementation — battle-tested, vectorized envs, automatic advantage normalization
- **Transformers 4.45.0+**: T5 model loading — Hugging Face standard, automatic downloads, excellent T5 support
- **T5-large (770M params)**: Semantic encoder — optimal for GPU constraints, can downscale to T5-base (220M)
- **sentence-transformers 3.3.0+**: SBERT embeddings — fast semantic similarity, lightweight baseline

**Architecture implications:**
- Dual policy support: MLP on belief features (SB3) vs T5 end-to-end (custom)
- T5 serves dual purpose: likelihood model (encoder similarity) and optional policy
- YAML-driven configuration for experiment management
- Memory requirements: 16GB RAM, 8GB GPU VRAM for T5-large (reduce to T5-base if constrained)

### Expected Features

The domain has well-established evaluation standards. The S_q metric (system score) is table stakes for academic credibility. Anti-artifact guards in multiple-choice construction are critical — without them, agents exploit spurious patterns like token overlap or length ratios rather than learning from clues. Control experiments (choices-only, shuffle, alias substitution) verify the agent actually uses clues. Baseline comparisons are expected in academic papers; qb-rl implements four (Threshold, SoftmaxProfile, SequentialBayes, AlwaysBuzzFinal).

**Must have (table stakes):**
- S_q metric and episode traces — standard evaluation in quiz bowl literature
- Anti-artifact guards (alias collision, token overlap, length ratio) — ensures valid experiments
- Control experiments (choices-only, shuffle, alias) — academic rigor, verifies agent uses clues
- Baseline agent comparisons — establishes performance floor
- Calibration metrics (ECE, Brier score) — uncertainty quantification standard
- Belief feature extraction (margin, entropy, stability, progress) — standard approach for POMDP
- Multiple choice K=4 format — standard quiz bowl setup

**Should have (differentiators):**
- T5 encoder as likelihood model — novel contribution, pre-trained semantic understanding
- T5 as optional policy encoder — end-to-end learning from text, unique to qanta-buzzer
- Dual architecture support (MLP vs T5) — key differentiator for writeup comparison
- Supervised warm-start for T5 — speeds convergence for large models
- Bootstrap confidence intervals — statistical rigor
- YAML configuration system — better than Python config classes for experiments
- Smoke test mode — fast iteration during development

**Defer (v2+):**
- Web UI or interactive demo — not needed for CS234 writeup
- Multi-GPU distributed training — dataset fits on single GPU
- Ensemble models — time constraint, single model comparison sufficient
- Cross-dataset generalization — QANTA dataset sufficient
- Real-time latency optimization — batch evaluation only

### Architecture Approach

The unified system adopts a four-layer modular architecture with clear separation: (1) Pipeline scripts orchestrate data preparation, training, evaluation; (2) Agent layer implements policies (PPO, baselines), action selection; (3) Environment layer provides Gymnasium interface, POMDP dynamics, reward computation; (4) Model layer handles likelihood scoring, belief features, neural networks. Communication flows downward: pipeline configures agents, agents interact with environment, environment queries models. Configuration is YAML-driven with factory methods for component instantiation.

**Major components:**
1. **Pipeline Layer** (scripts/) — Orchestrates MC dataset construction, training loops, evaluation runs; consumes YAML config
2. **Agent Layer** (agents/) — Policies (SB3 PPO for MLP, T5PolicyModel for end-to-end), baseline agents, episode trace generation
3. **Environment Layer** (qb_env/) — TossupMCEnv implements Gymnasium interface, computes beliefs via LikelihoodModel, manages POMDP state
4. **Model Layer** (models/) — Abstract LikelihoodModel with implementations (TF-IDF, SBERT, T5), belief feature extraction, T5 policy heads

**Key patterns:**
- Factory-based construction: components built from YAML config via factory functions
- Pluggable likelihood models: abstract base class with concrete implementations (TF-IDF, SBERT, T5)
- Dual policy architecture: support both lightweight (MLP on features) and heavyweight (T5 encoder)
- Episode traces for S_q: agents return per-step buzz probability and correctness traces

### Critical Pitfalls

**Top 5 pitfalls to avoid:**

1. **Belief State Collapse in Early Training** — Likelihood models output uniform distributions early, causing belief features (margin=0, entropy=max) to be uninformative. PPO can't learn from constant features. Prevention: pre-compute answer profiles on full dataset, add minimum margin threshold (0.05), monitor entropy in first 10 episodes.

2. **Reward Shaping Overfitting** — Time penalty coefficient dominates reward signal, agent learns fixed buzz position regardless of confidence. Prevention: use multiple reward modes (time_penalty, human_grounded, simple), validate on held-out categories, add reward noise during training.

3. **Incompatible Architecture Merge** — qanta-buzzer uses text observations, qb-rl uses numeric belief vectors. Naively combining creates observation space mismatch. Prevention: define clear observation interfaces (BeliefObservation, TextObservation classes), never mix types in same training loop, add shape assertions in model forward.

4. **Gradient Accumulation Memory Leak** — PPO stores full trajectory (6-12 steps × batch_size × 512 tokens × 1024 hidden) in memory. OOM after ~50 iterations with T5-large. Prevention: detach and move to CPU immediately, use gradient checkpointing, implement trajectory buffer with max size, monitor GPU memory.

5. **Scope Explosion During Merge (Tight Deadline)** — Trying to merge all features from both codebases creates 2-week integration task. Nothing works by deadline. Prevention: Week 1 critical path is qb-rl env + T5 likelihood + basic PPO only. Defer all baselines except threshold, supervised pretraining, SBERT/OpenAI likelihoods. MVP first, enhancements only if time remains.

## Implications for Roadmap

Based on research, the critical path for a one-week deadline requires ruthless prioritization. The core value is demonstrating T5 as a likelihood model — this is the novel contribution. Everything else (supervised warm-start, multiple baselines, advanced evaluation) is secondary. Build vertically through the stack: environment → T5 likelihood → MLP policy → training → evaluation. Once this works end-to-end, add T5 policy as a comparison point if time permits.

### Phase 1: Core Infrastructure and Data Pipeline
**Rationale:** Need working data structures and MC dataset before any training. This establishes the foundation and validates dataset quality early.
**Delivers:** MCQuestion dataclass, answer profile building, distractor generation with anti-artifact guards, train/val/test splits
**Addresses:** Anti-artifact guards (critical pitfall prevention), MC K=4 format (table stakes)
**Avoids:** Distractor quality degradation (Pitfall 6), answer distribution shift (Pitfall 5)
**Research flag:** Standard patterns — skip research-phase. MC construction is well-documented in qb-rl.

### Phase 2: Environment and Belief Models
**Rationale:** Must have working environment before training agents. Likelihood models compute beliefs that feed MLP policy.
**Delivers:** TossupMCEnv (Gymnasium), belief feature extraction, TF-IDF/SBERT likelihood models, observation interface
**Uses:** Gymnasium 1.1.0+, sentence-transformers, scikit-learn
**Implements:** Environment layer, Model layer (partial)
**Avoids:** Belief state collapse (Pitfall 1 — pre-compute profiles), incompatible architecture merge (Pitfall 3 — clear interfaces)
**Research flag:** Standard patterns — Gymnasium environment well-documented.

### Phase 3: T5 Likelihood Model Integration
**Rationale:** This is the novel contribution. T5 encoder computes semantic similarity for belief updates. Must work before comparing policies.
**Delivers:** T5Likelihood class implementing abstract interface, embedding cache for efficiency
**Uses:** Transformers 4.45.0+, T5-large encoder
**Implements:** Model layer completion
**Avoids:** Tokenization overhead (Pitfall 9 — cache embeddings), memory leak (Pitfall 4 — detach tensors)
**Research flag:** May need phase research — T5 encoder similarity scoring less documented than standard usage.

### Phase 4: MLP Policy Training (SB3 PPO)
**Rationale:** Lightweight baseline using belief features. SB3 PPO is battle-tested, gets results quickly.
**Delivers:** Working PPO agent on belief features, training loop, checkpointing
**Uses:** Stable-Baselines3 2.6.0+, PyTorch 2.3.0+
**Implements:** Agent layer (MLP policy)
**Avoids:** Reward shaping overfitting (Pitfall 2 — multiple reward modes), scope explosion (Pitfall 12 — defer baselines)
**Research flag:** Standard patterns — SB3 PPO well-documented.

### Phase 5: Evaluation Pipeline
**Rationale:** Must validate results with S_q metric and control experiments for academic credibility.
**Delivers:** S_q computation, episode traces, control experiments, calibration metrics, comparison plots
**Addresses:** S_q metric (table stakes), control experiments (table stakes), calibration metrics (table stakes)
**Avoids:** Evaluation metric gaming (Pitfall 8 — multiple metrics), determinism loss (Pitfall 10 — set seeds)
**Research flag:** Standard patterns — S_q computation documented in qb-rl.

### Phase 6: T5 Policy (Optional, If Time Permits)
**Rationale:** Comparison point between T5-as-likelihood (Phase 3) vs T5-as-policy. Only if Phase 1-5 complete and stable.
**Delivers:** T5PolicyModel with custom policy heads, text observation interface, optional supervised warm-start
**Uses:** Existing qanta-buzzer implementation adapted
**Implements:** Agent layer (T5 policy)
**Avoids:** Checkpoint compatibility break (Pitfall 7 — version architecture), gradient accumulation memory leak (Pitfall 4)
**Research flag:** May need phase research — custom policy heads on T5 encoder less standard.

### Phase Ordering Rationale

- **Vertical slice first**: Phase 1-5 builds complete pipeline from data → training → evaluation with MLP policy. Phase 6 is horizontal expansion (alternative policy).
- **Novel contribution early**: T5 likelihood (Phase 3) is the key contribution, comes before policy training so we can validate it works.
- **Battle-tested before custom**: Use SB3 PPO (Phase 4) before attempting custom T5 policy (Phase 6). If SB3 works, architecture is sound.
- **Validation at every step**: Each phase delivers testable output. Smoke tests catch integration bugs before expensive training runs.
- **Deferred complexity**: Supervised warm-start, multiple baselines, advanced evaluation deferred to Phase 6+. Critical path is lean.

**Dependency chain:**
- Phase 2 depends on Phase 1 (needs MCQuestion dataclass)
- Phase 3 depends on Phase 2 (implements LikelihoodModel interface)
- Phase 4 depends on Phase 3 (trains on beliefs computed by T5 likelihood)
- Phase 5 depends on Phase 4 (evaluates trained policy)
- Phase 6 depends on Phase 2 (environment) but independent of Phase 3-4 (different policy)

### Research Flags

**Phases likely needing deeper research during planning:**
- **Phase 3 (T5 Likelihood)**: T5 encoder for similarity scoring less documented than standard seq2seq usage. May need to research embedding extraction best practices.
- **Phase 6 (T5 Policy)**: Custom policy heads on T5 encoder is novel architecture. If pursued, will need research on architecture design and training stability.

**Phases with standard patterns (skip research-phase):**
- **Phase 1 (Data Pipeline)**: MC construction, distractor generation well-documented in qb-rl codebase.
- **Phase 2 (Environment)**: Gymnasium environment creation has established patterns, qb-rl reference implementation.
- **Phase 4 (MLP PPO)**: SB3 PPO integration is standard, extensive documentation and examples.
- **Phase 5 (Evaluation)**: S_q metric computation, control experiments implemented in qb-rl, clear reference.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | Based on working qb-rl pyproject.toml and standard 2024-2025 RL research practices. Version compatibility matrix verified. |
| Features | HIGH | S_q metric, anti-artifact guards, control experiments are documented standards in quiz bowl literature. Feature priorities clear from CS234 project scope. |
| Architecture | HIGH | Four-layer modular architecture observed in qb-rl codebase. Gymnasium/SB3 integration patterns well-established. T5 integration adapted from working qanta-buzzer. |
| Pitfalls | MEDIUM-HIGH | Critical pitfalls (belief collapse, reward overfitting, memory leak) identified from CONCERNS.md and common RL failure modes. Integration pitfalls inferred from architecture differences. |

**Overall confidence:** HIGH

The research benefits from two working reference implementations (qb-rl and qanta-buzzer). Stack recommendations are based on verified dependencies from qb-rl. Architecture patterns are observed from qb-rl's proven structure. Pitfalls are identified from explicit CONCERNS.md warnings plus standard RL failure modes. The main uncertainty is in integration details (Phase 3, Phase 6) where novel combinations occur.

### Gaps to Address

**Integration testing strategy**: Research identifies memory leaks and observation space mismatches as critical risks, but optimal testing strategy for rapid iteration needs refinement. During Phase 3 planning, design smoke tests that catch integration bugs in <1 minute runtime.

**T5 encoder similarity scoring**: While T5 as seq2seq is well-documented, using T5 encoder for semantic similarity scoring (Phase 3) is less standard. During Phase 3 planning, research whether to use mean pooling, CLS token, or last hidden state for text embeddings.

**Supervised warm-start necessity**: qanta-buzzer uses supervised pre-training before PPO for T5 policy, but unclear if this is required or just helpful. If Phase 6 is attempted, validate whether PPO can train T5 policy from scratch or if warm-start is essential.

**Hyperparameter sensitivity**: Research identifies reward shaping overfitting risk but doesn't specify optimal time penalty coefficient or other hyperparameters. During Phase 4, may need to sweep time penalty values (0.05, 0.1, 0.2) to find stable setting.

**Category stratification importance**: Pitfall 5 warns about answer distribution shift across categories. If validation accuracy varies >30% across categories, may need category-specific models or multi-task learning (out of scope for week 1 but note for limitations section).

## Sources

### Primary (HIGH confidence)
- qb-rl codebase analysis (/Users/ankit.aggarwal/Dropbox/Stanford/CS234/final_project/qb-rl/) — verified working stack, architecture patterns, evaluation metrics
- qanta-buzzer codebase analysis (this repository) — T5 integration, supervised warm-start, policy head design
- CS234 project CLAUDE.md files — project constraints, testing commands, conventions
- qb-rl CONCERNS.md — explicit warnings about memory leaks, gradient accumulation issues

### Secondary (MEDIUM confidence)
- Gymnasium 1.1.0 documentation — environment interface, observation space design
- Stable-Baselines3 2.6.0 documentation — PPO implementation, vectorized environments
- Transformers library documentation — T5 model loading, tokenization
- NumPy <2.0 compatibility — known ecosystem issue documented in multiple sources

### Tertiary (MEDIUM confidence, inferred patterns)
- Quiz bowl RL literature — S_q metric standard, belief feature extraction patterns (inferred from qb-rl implementation)
- Common RL pitfalls — reward hacking, distribution shift, exploration collapse (general RL knowledge)
- Integration patterns — BERT+RL, T5+classical features (analogous hybrid architectures)

---
*Research completed: 2026-02-24*
*Ready for roadmap: yes*
