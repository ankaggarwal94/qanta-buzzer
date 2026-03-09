# Feature Landscape

**Domain:** RL-based Quiz Bowl Buzzer System
**Researched:** 2026-02-24

## Table Stakes

Features users expect. Missing = product feels incomplete.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| POMDP environment with incremental clue revelation | Core quiz bowl mechanic — agents must decide with partial information | Low | Already implemented in both codebases |
| Multiple choice answer selection (K=4) | Standard format for quiz bowl experiments | Low | Both codebases have this |
| S_q metric (system score) | Standard evaluation metric in quiz bowl literature: sum of buzz probability × correctness | Medium | qb-rl has it, critical for rigorous evaluation |
| Baseline agent comparisons | Academic standard — must compare against reasonable baselines | Medium | qb-rl has 4: Threshold, SoftmaxProfile, SequentialBayes, AlwaysBuzzFinal |
| PPO or other standard RL algorithm | Established algorithm for credibility | Medium | Both have PPO (custom in qanta, SB3 in qb-rl) |
| Belief feature extraction | Standard approach: margin, entropy, stability, progress features | Low | qb-rl has complete implementation |
| Anti-artifact guards in MC construction | Prevents trivial solutions (alias collision, token overlap, length ratio) | Medium | Critical for valid experiments — qb-rl has all three |
| Control experiments | Academic rigor: choices-only, shuffle, alias substitution | Medium | Verifies agent uses clues, not artifacts |
| Calibration metrics (ECE, Brier) | Standard for uncertainty quantification | Low | Both codebases compute these |
| Per-category performance analysis | Standard breakdown for understanding strengths/weaknesses | Low | Both track category accuracy |
| Episode trace tracking (c_trace, g_trace) | Required for S_q computation and analysis | Low | qb-rl has clean implementation |
| Train/val/test splits | Basic ML requirement | Low | Both have proper splits |
| Checkpoint save/load | Standard for experiment management | Low | Both implement this |
| Configurable reward modes | Different training objectives (time_penalty, human_grounded, simple) | Low | qb-rl has all three modes |

## Differentiators

Features that set product apart. Not expected, but valued.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| T5 encoder as likelihood model | Pre-trained semantic understanding beats TF-IDF/SBERT baselines | High | Novel contribution — qanta has T5 policy, needs adaptation |
| T5 as optional policy encoder | End-to-end learning from text, not just belief features | High | Unique to qanta-buzzer, strong semantic policy |
| Supervised warm-start for T5 | Speeds convergence for large models | Medium | qanta-buzzer has this, valuable for T5 |
| Dual architecture support (MLP vs T5) | Direct comparison of lightweight vs semantic policies | High | Key differentiator for writeup |
| Human buzz position comparison | "Expected wins vs humans" metric | Medium | qb-rl tracks human_buzz_positions |
| Entropy vs clue index visualization | Shows information gain dynamics | Low | qb-rl has plotting infrastructure |
| Bootstrap confidence intervals | Statistical rigor for metrics | Low | qb-rl implements for all metrics |
| YAML configuration system | Better than Python config classes for experiments | Low | qb-rl has clean YAML setup |
| Smoke test mode | Fast iteration during development | Low | qb-rl has --smoke throughout |
| Sequential Bayes belief update | More efficient than from-scratch recomputation | Medium | qb-rl has both modes |
| Multiple likelihood models (TF-IDF, SBERT, OpenAI, T5) | Comprehensive comparison of text → belief approaches | High | qb-rl has 3, adding T5 is novel |
| Answer profile building with leave-one-out | Better distractor quality via aggregated question text | Medium | qb-rl has sophisticated implementation |
| Learning curve plots | Shows training dynamics | Low | qb-rl generates automatically |
| Calibration curve plots | Visual assessment of uncertainty quality | Low | qb-rl has implementation |

## Anti-Features

Features to explicitly NOT build.

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| Web UI or interactive demo | Not needed for CS234 writeup, time sink | Focus on batch evaluation and plots |
| Real-time game integration | Academic project scope only | Offline evaluation is sufficient |
| Multi-GPU distributed training | Dataset fits on single GPU | Single GPU/MPS is fine |
| Custom tokenization | Pre-trained models handle this | Use T5 tokenizer as-is |
| Dynamic question generation | Out of scope, need fixed test set | Use existing QANTA dataset |
| Adversarial robustness testing | Beyond project scope | Standard evaluation sufficient |
| Beam search for answer selection | Unnecessary complexity | Greedy/sampling is standard |
| Ensemble models | Time constraint, single model sufficient | Compare architectures instead |
| Hyperparameter auto-tuning | Time constraint | Manual config based on literature |
| Cross-dataset generalization | Single dataset sufficient for writeup | QANTA dataset only |
| Real-time latency optimization | Not a deployment system | Focus on accuracy metrics |
| Custom reward shaping beyond standard modes | Three modes sufficient | Use time_penalty, human_grounded, simple |
| Audio/video question formats | Text-only is standard | Text questions only |
| Team play coordination | Single agent scope | Individual buzzer only |
| Question difficulty estimation | Interesting but out of scope | Fixed question set |

## Feature Dependencies

```
S_q metric → Episode traces (c_trace, g_trace)
Episode traces → Belief features
Belief features → Likelihood model
T5 as likelihood → Answer profiles + T5 encoder
T5 as policy → Supervised warm-start (recommended)
Supervised warm-start → T5 encoder + training pipeline
Bootstrap CI → Multiple evaluation runs
Human comparison → Human buzz position data
Sequential Bayes → Prior + likelihood model
Anti-artifact guards → MC construction pipeline
Control experiments → Base evaluation pipeline
Calibration metrics → Probability outputs from policy
YAML config → Config loading infrastructure
Smoke mode → Subset data loading
```

## MVP Recommendation

Prioritize:
1. S_q metric and episode traces — **critical for rigorous evaluation**
2. Four baseline agents — **establishes performance floor**
3. Anti-artifact guards — **ensures valid experiments**
4. Control experiments — **academic rigor**
5. T5 as likelihood model — **novel contribution**
6. Belief feature MLP policy — **lightweight baseline**
7. T5 policy with supervised warm-start — **strong semantic agent**
8. YAML configuration — **experiment management**
9. Calibration metrics — **uncertainty quantification**
10. Comparison plots and tables — **writeup figures**

Defer:
- Web UI: Not needed for writeup
- OpenAI embeddings: SBERT sufficient, avoid API costs
- Cross-dataset evaluation: Time constraint
- Ensemble approaches: Single model comparison sufficient
- Real-time optimizations: Batch evaluation only

## Sources

- Analysis of existing qanta-buzzer codebase (this repository)
- Analysis of qb-rl codebase architecture and features
- CS234 project requirements (academic writeup focus)
- Quiz bowl RL literature (S_q metric is standard)