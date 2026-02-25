# Domain Pitfalls

**Domain:** RL-based quiz bowl buzzer (merging two codebases)
**Researched:** 2026-02-24

## Critical Pitfalls

Mistakes that cause rewrites or major issues.

### Pitfall 1: Belief State Collapse in Early Training
**What goes wrong:** Likelihood models output uniform distributions early in training, causing belief features (margin=0, entropy=max) to be uninformative. PPO agent can't learn meaningful patterns from constant features.
**Why it happens:** TF-IDF/SBERT models need sufficient answer profile data. With small datasets or poor initialization, all options score similarly.
**Consequences:** PPO converges to always-wait or always-buzz-immediately policies. Training appears to work but agent never learns to discriminate.
**Prevention:**
- Pre-compute answer profiles on full dataset before training
- Add minimum margin threshold (0.05) to synthetic data generation
- Monitor belief entropy during first 10 episodes — if always max, stop and debug
**Detection:** Track `margin` feature statistics in tensorboard. If mean < 0.01 for >100 episodes, belief state has collapsed.

### Pitfall 2: Reward Shaping Overfitting
**What goes wrong:** Time penalty coefficient (0.1 default) dominates reward signal. Agent learns to buzz at fixed position regardless of confidence.
**Why it happens:** Linear time penalty `R = correct - 0.1 * (position/total)` creates predictable gradient. Agent exploits this instead of learning from question content.
**Consequences:** High training reward but poor test accuracy. Agent ignores actual clues, just memorizes optimal buzz position.
**Prevention:**
- Use multiple reward modes: `time_penalty`, `human_grounded` (match human buzz distribution), `simple` (no time component)
- Validate on held-out categories with different optimal buzz positions
- Add reward noise (±0.05) during training to prevent memorization
**Detection:** Plot buzz position histogram. If >80% buzzes cluster at single position across different questions, reward is overfit.

### Pitfall 3: Incompatible Architecture Merge
**What goes wrong:** qanta-buzzer uses text observations `"CLUES: ... | CHOICES: ..."` while qb-rl uses numeric belief vectors `[belief[K], margin, entropy, ...]`. Naively combining creates observation space mismatch.
**Why it happens:** Two codebases evolved independently with different observation abstractions. T5 expects text, MLP expects features.
**Consequences:** Model forward pass fails with dimension mismatch. Or worse, silently processes wrong data shape producing garbage outputs.
**Prevention:**
- Define clear observation interface: `BeliefObservation` and `TextObservation` classes
- T5 as likelihood model converts text → beliefs → MLP policy
- T5 as policy takes text + belief features concatenated
- Never mix observation types in same training loop
**Detection:** Add shape assertions in model forward: `assert obs.shape[-1] == expected_dim`

### Pitfall 4: Gradient Accumulation Memory Leak
**What goes wrong:** PPO stores full trajectory (6-12 steps × batch_size × 512 tokens × hidden_size) in memory. With T5-large (1024 hidden), OOM after ~50 iterations.
**Why it happens:** RolloutStep dataclass stores input_ids, attention_mask, and hidden states for all timesteps. Never explicitly freed.
**Consequences:** Training crashes unpredictably after seeming to work initially. Loses hours of compute.
**Prevention:**
- Detach and move to CPU immediately: `hidden.detach().cpu()`
- Use gradient checkpointing for T5 encoder
- Implement trajectory buffer with max size, flush every 10 episodes
- Monitor GPU memory in training loop, warn if >90% utilized
**Detection:** Profile memory with `torch.cuda.memory_summary()`. If "reserved" grows linearly with iterations, leak exists.

## Moderate Pitfalls

### Pitfall 5: Answer Distribution Shift
**What goes wrong:** Training on history/literature questions, testing on science. Vocabulary and answer patterns completely different.
**Why it happens:** Quiz bowl categories have distinct language patterns. "Napoleon" frequent in history, never in biology.
**Prevention:**
- Stratified train/val/test splits by category
- Category-specific answer profiles
- Multi-task training head with category embedding
**Detection:** Compare per-category accuracy. If variance >30%, distribution shift is problematic.

### Pitfall 6: Distractor Quality Degradation
**What goes wrong:** Generated distractors become too easy (random names) or too hard (near-synonyms), breaking MC task difficulty.
**Why it happens:** Embedding similarity doesn't capture quiz bowl difficulty. "Franklin Roosevelt" and "Theodore Roosevelt" are close embeddings but different answers.
**Prevention:**
- Use multiple distractor strategies: category-based (40%), embedding-based (40%), common-confusion (20%)
- Anti-artifact guards: no token overlap >50%, no aliases of correct answer
- Manual review of 100 random MC questions before training
**Detection:** Choices-only baseline should achieve 25-35%. If >50%, distractors too easy. If <20%, too hard.

### Pitfall 7: Checkpoint Compatibility Break
**What goes wrong:** Saved supervised model can't load into PPO training due to architecture changes between phases.
**Why it happens:** qanta-buzzer saves full T5 + policy head. If policy head architecture changes (e.g., hidden_size), checkpoint invalid.
**Prevention:**
- Version policy head architecture in checkpoint metadata
- Save base T5 and policy head separately
- Implement `strict=False` loading with clear warnings for missing keys
**Detection:** Try loading checkpoint immediately after saving. If fails, compatibility broken.

### Pitfall 8: Evaluation Metric Gaming
**What goes wrong:** Agent achieves high S_q score by always buzzing on final clue (100% accuracy, moderate speed).
**Why it happens:** S_q = Σ(buzz_prob × correctness) can be maximized by conservative strategies that look good on paper but aren't competitive.
**Prevention:**
- Report multiple metrics: S_q, average buzz position, accuracy@position
- Compare to human buzz distribution via KL divergence
- Require minimum buzz variance (not all at same position)
**Detection:** If buzz_position.std() < 0.5, agent is position-locked.

## Minor Pitfalls

### Pitfall 9: Tokenization Overhead
**What goes wrong:** Re-tokenizing same text every forward pass adds 30% latency.
**Why it happens:** T5 tokenizer called repeatedly on same clue prefixes.
**Prevention:**
- Cache tokenized representations in Question dataclass
- Pre-tokenize during dataset loading, not during training
**Detection:** Profile with `cProfile`: if tokenization >10% of runtime, needs caching.

### Pitfall 10: Determinism Loss
**What goes wrong:** Same model produces different results on same test set.
**Why it happens:** Missing seeds, non-deterministic CUDA ops, or sampling in evaluation.
**Prevention:**
- Set all seeds: `torch`, `numpy`, `random`, `transformers.set_seed()`
- Use `torch.use_deterministic_algorithms(True)` in eval
- Evaluation must use `deterministic=True` mode
**Detection:** Run evaluation twice. Results should match exactly.

### Pitfall 11: Progress Feature Misleading
**What goes wrong:** `progress = step_idx / total_steps` assumes all questions have same length, but they vary 3-12 clues.
**Why it happens:** Normalization by total_steps makes progress=0.5 mean different things for different questions.
**Prevention:**
- Include absolute `clue_idx` as separate feature
- Normalize by average question length (6) not actual length
- Learn question-length embedding
**Detection:** Plot progress feature vs actual clue number. Should be uniform, not clustered.

## Phase-Specific Warnings

| Phase Topic | Likely Pitfall | Mitigation |
|-------------|---------------|------------|
| Environment Setup | Observation space mismatch between T5/MLP | Define TypedDict observations with validation |
| Data Pipeline | Distractor quality, answer distribution | Stratified splits, multiple distractor strategies |
| Belief Models | Likelihood collapse to uniform | Pre-compute profiles, add margin thresholds |
| PPO Training | Memory leak from trajectory storage | Gradient checkpointing, explicit tensor cleanup |
| Evaluation | Metric gaming, determinism | Multiple metrics, human baseline comparison |
| Architecture Merge | Checkpoint incompatibility | Separate base/head saves, version metadata |
| Reward Design | Time penalty overfitting | Multiple reward modes, validation on held-out categories |
| Final Integration | Different codebases use different conventions | Create adapter layer, don't modify both codebases |

## Tight Deadline Specific (This Week)

### Pitfall 12: Scope Explosion During Merge
**What goes wrong:** Trying to merge ALL features from both codebases creates 2-week integration task.
**Why it happens:** qb-rl has 4 baselines, 3 likelihood models, complex evaluation. qanta-buzzer has supervised pretraining, T5 integration. Combining everything is massive.
**Consequences:** Nothing works by deadline. Half-integrated system worse than either original.
**Prevention:**
- Week 1 critical path: qb-rl env + qanta-buzzer T5 as likelihood model + basic PPO
- Defer: All baselines except threshold, SBERT/OpenAI likelihoods, supervised pretraining
- MVP first, enhancements only if time remains
**Detection:** If integration not working after 2 days, immediately reduce scope.

### Pitfall 13: Testing on Integration
**What goes wrong:** Discovering integration bugs only after full training runs wastes GPU hours.
**Why it happens:** No integration tests between components. First full run is the test.
**Consequences:** 6-hour training run fails at hour 5. Multiple iterations burn entire week.
**Prevention:**
- Smoke test immediately: 10 questions, 1 epoch, 10 PPO steps
- Add integration test that runs full pipeline on synthetic data in <1 minute
- Test checkpoint save/load before starting long training
**Detection:** If first full run hasn't completed successfully within 24 hours, stop and add tests.

### Pitfall 14: Git Merge Conflicts
**What goes wrong:** Both codebases modify same files (config.py, train_ppo.py), creating complex merge conflicts.
**Why it happens:** Similar filenames, overlapping functionality.
**Consequences:** Hours spent resolving conflicts, introducing subtle bugs.
**Prevention:**
- Keep codebases in separate directories initially
- Create new unified module that imports from both
- Only merge after interfaces stabilized
**Detection:** If merge has >10 conflicts, abort and use adapter pattern instead.

## Sources

- Analysis of existing codebases (CONCERNS.md highlights memory leaks, gradient accumulation issues)
- Common RL pitfalls from literature (reward hacking, distribution shift, exploration collapse)
- Integration-specific issues from similar hybrid architectures (BERT + RL, T5 + classical features)
- Time-pressure patterns from rapid prototyping scenarios

---

*Pitfalls audit: 2026-02-24*
*Confidence: HIGH for codebase-specific issues (found in CONCERNS.md), MEDIUM for general RL pitfalls, MEDIUM for integration patterns*