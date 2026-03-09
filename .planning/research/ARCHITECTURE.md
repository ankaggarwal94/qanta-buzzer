# Architecture Patterns

**Domain:** RL-based quiz bowl buzzer system
**Researched:** 2026-02-24

## Recommended Architecture

The unified system adopts a **four-layer modular architecture** with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────┐
│                    Pipeline Layer                        │
│         (scripts/, orchestration, configuration)         │
└─────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────┐
│                     Agent Layer                          │
│    (agents/, policies, training algorithms, baselines)   │
└─────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────┐
│                 Environment Layer                        │
│     (qb_env/, Gymnasium interface, POMDP dynamics)       │
└─────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────┐
│                     Model Layer                          │
│  (models/, likelihood scoring, features, neural nets)    │
└─────────────────────────────────────────────────────────┘
```

### Component Boundaries

| Component | Responsibility | Communicates With |
|-----------|---------------|-------------------|
| **Pipeline Scripts** | Orchestration, data preparation, training loops, evaluation | Agents (train/eval), Environment (dataset building), Configuration |
| **Agent Module** | Policy implementations (PPO, baselines), action selection, episode execution | Environment (step/reset), Models (for T5 policy), Evaluation |
| **Environment Module** | POMDP simulation, observation generation, reward computation | Models (likelihood scoring), Data structures (MCQuestion) |
| **Model Module** | Likelihood scoring, belief features, neural networks (T5/MLP) | None (leaf layer, pure computation) |
| **Configuration** | YAML-based hyperparameters, factory methods | All layers (dependency injection) |
| **Evaluation Module** | Metrics (S_q, calibration), controls, visualization | Agents (episode traces), Environment (for controls) |

### Data Flow

**Training Pipeline:**

1. **MC Dataset Construction** (`build_mc_dataset.py`)
   - Load raw tossups from CSV/HuggingFace → Parse into TossupQuestion objects
   - Build answer profiles via aggregation → Store in AnswerProfileDB
   - Generate distractors via LikelihoodModel ranking → Apply anti-artifact guards
   - Output: `mc_dataset.json` with MCQuestion objects

2. **Environment Initialization**
   - Load MCQuestion dataset → Initialize LikelihoodModel (TF-IDF/SBERT/T5)
   - Create TossupMCEnv (Gymnasium) → Configure reward mode and belief computation
   - Observation: `[belief[K], top_p, margin, entropy, stability, progress, clue_idx_norm]`

3. **Agent Training**
   - **Baselines**: ThresholdBuzzer, SoftmaxProfileBuzzer, SequentialBayesBuzzer
   - **MLP Policy**: Belief features → SB3 PPO → Wait/buzz actions
   - **T5 Policy**: Text + features → T5 encoder → Policy heads (wait/answer/value)
   - **Supervised Warm-start** (T5 only): Complete questions → Cross-entropy on answers

4. **Episode Execution**
   - Agent receives observation → Selects action (0=wait, 1..K=buzz)
   - Environment steps → Updates belief, reveals clue, computes reward
   - Episode trace: `c_trace` (buzz probability), `g_trace` (correctness)

5. **Evaluation**
   - Compute S_q = Σ(c_t × g_t) across episodes
   - Run controls: choices_only, shuffle, alias_substitution
   - Generate calibration plots, comparison tables

## Patterns to Follow

### Pattern 1: Factory-based Construction
**What:** Components built via factory functions from YAML config
**When:** Creating environments, models, agents from configuration
**Example:**
```python
def make_env_from_config(config: dict, questions: list[MCQuestion]) -> TossupMCEnv:
    likelihood_model = build_likelihood_from_config(config["likelihood"])
    return TossupMCEnv(
        questions=questions,
        likelihood_model=likelihood_model,
        K=config["data"]["K"],
        reward_mode=config["environment"]["reward"],
        belief_mode=config["environment"]["belief_mode"],
        **config["environment"]
    )
```

### Pattern 2: Pluggable Likelihood Models
**What:** Abstract base class with concrete implementations for different scoring methods
**When:** Supporting multiple text similarity approaches (TF-IDF, SBERT, T5)
**Example:**
```python
class LikelihoodModel(ABC):
    @abstractmethod
    def score(self, clue_prefix: str, option_profiles: list[str]) -> np.ndarray:
        """Return raw similarity scores for softmax conversion"""
```

### Pattern 3: Dual Policy Architecture
**What:** Support both lightweight (MLP on features) and heavyweight (T5 encoder) policies
**When:** Comparing computational efficiency vs. semantic understanding
**Example:**
```python
# Lightweight: belief features → MLP
policy = PPO("MlpPolicy", env, policy_kwargs={"net_arch": [64, 64]})

# Heavyweight: text → T5 → policy heads
policy = T5PolicyModel(model_name="t5-large", num_options=4)
```

### Pattern 4: Episode Traces for S_q
**What:** Agents return traces with per-step buzz probability and correctness
**When:** Computing system score S_q = Σ(buzz_prob × correctness)
**Example:**
```python
@dataclass
class EpisodeTrace:
    c_trace: list[float]  # buzz probability per step
    g_trace: list[float]  # correctness indicator per step

def compute_sq(traces: list[EpisodeTrace]) -> float:
    return sum(sum(c * g for c, g in zip(t.c_trace, t.g_trace)) for t in traces)
```

## Anti-Patterns to Avoid

### Anti-Pattern 1: Monolithic Training Scripts
**What:** Single file handling data loading, model creation, training, evaluation
**Why bad:** Difficult to test, modify, or run partial pipelines
**Instead:** Separate scripts per pipeline stage with shared configuration

### Anti-Pattern 2: Tight Model-Environment Coupling
**What:** Environment directly contains neural network models
**Why bad:** Can't swap models, difficult to test environment logic
**Instead:** Environment accepts abstract LikelihoodModel, agents own policy networks

### Anti-Pattern 3: Hard-coded Hyperparameters
**What:** Hyperparameters scattered across Python files
**Why bad:** Requires code changes for experiments, no version control of configs
**Instead:** YAML configuration with override capability via CLI

### Anti-Pattern 4: Observation as Raw Text
**What:** Passing raw question text as environment observation
**Why bad:** Incompatible with standard RL libraries, inefficient
**Instead:** Extract numeric belief features, optionally augment with text in agent

## Scalability Considerations

| Concern | At 100 questions | At 10K questions | At 1M questions |
|---------|------------------|------------------|-----------------|
| **Likelihood Caching** | Not needed | Embedding cache in memory | Redis/disk cache for embeddings |
| **Dataset Loading** | Load all in memory | Load all in memory | Streaming from disk/database |
| **Training** | Single GPU/CPU | Single GPU recommended | Multi-GPU data parallel |
| **Answer Profiles** | Build on-the-fly | Pre-compute and cache | Distributed profile building |

## Build Order Dependencies

**Phase 1: Core Infrastructure**
1. Configuration system (YAML loading, factory methods)
2. Data structures (MCQuestion, TossupQuestion, AnswerProfile)
3. LikelihoodModel abstraction and TF-IDF/SBERT implementations

**Phase 2: Environment**
1. TossupMCEnv with Gymnasium interface
2. Belief feature extraction
3. Reward modes (time_penalty, human_grounded, simple)

**Phase 3: Agents**
1. Baseline agents (Threshold, Softmax, Bayes)
2. MLP policy with SB3 PPO
3. Episode trace generation

**Phase 4: T5 Integration**
1. T5 as LikelihoodModel (encoder similarity scoring)
2. T5PolicyModel with custom heads (wait/answer/value)
3. Supervised warm-start training

**Phase 5: Evaluation**
1. S_q metric computation
2. Control experiments (shuffle, choices_only, alias)
3. Calibration metrics and visualization

## Integration Points

### T5 as Likelihood Model
```python
class T5Likelihood(LikelihoodModel):
    def __init__(self, model_name: str = "t5-large"):
        self.encoder = T5EncoderModel.from_pretrained(model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)

    def score(self, clue_prefix: str, option_profiles: list[str]) -> np.ndarray:
        # Encode clue and options, compute cosine similarity
        clue_emb = self._encode(clue_prefix)
        option_embs = np.stack([self._encode(p) for p in option_profiles])
        return cosine_similarity(clue_emb, option_embs)[0]
```

### Unified Agent Interface
```python
class BuzzerAgent(ABC):
    @abstractmethod
    def run_episode(self, env: TossupMCEnv) -> EpisodeTrace:
        """Execute one episode, return trace for S_q computation"""

    @abstractmethod
    def action_probabilities(self, obs: np.ndarray) -> np.ndarray:
        """Return probability distribution over actions"""
```

### Configuration-Driven Pipeline
```yaml
# configs/experiment.yaml
model:
  type: "t5_policy"  # or "mlp_policy"
  t5_name: "t5-large"
  supervised_warmstart: true

likelihood:
  model: "sbert"  # or "tfidf", "t5"

training:
  algorithm: "ppo"
  total_timesteps: 100000
```

## Sources

- Gymnasium environment design patterns (inferred from qb-rl implementation)
- Stable-Baselines3 PPO integration patterns (observed in qb-rl/agents/)
- PyTorch model checkpointing patterns (from qanta-buzzer implementation)
- YAML configuration best practices (common in ML pipelines)