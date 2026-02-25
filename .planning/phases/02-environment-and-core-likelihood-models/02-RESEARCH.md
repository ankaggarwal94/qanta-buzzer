# Phase 2: Environment and Core Likelihood Models - Research

**Researched:** 2026-02-25
**Domain:** Gymnasium RL Environment with Belief-Based Observations
**Confidence:** HIGH

## Summary

Phase 2 implements the POMDP environment and likelihood models that convert incremental question clues into belief distributions over answer choices. This is the foundation for all RL agent training. The environment follows Gymnasium's standard interface (reset/step/observation_space/action_space) and computes rich belief features (belief[K], top_p, margin, entropy, stability, progress) at each step. Likelihood models use text similarity (TF-IDF or SBERT) to score how well each answer option matches the clues revealed so far.

The qb-rl codebase provides a complete, battle-tested reference implementation. All architectural patterns, configuration structures, and mathematical formulations are verified and working. The main implementation task is adapting this proven design to work with Phase 1's data structures (MCQuestion dataclass) and integrating with the existing YAML configuration system.

**Primary recommendation:** Port qb-rl's TossupMCEnv, LikelihoodModel abstract class, TfIdfLikelihood, and SBERTLikelihood directly. These components are production-ready and already handle all edge cases (belief collapse, forced termination, reward shaping). Use factory pattern for environment construction and maintain strict separation between environment logic (POMDP dynamics) and model logic (likelihood scoring).

## <phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| ENV-01 | TossupMCEnv implements Gymnasium Env interface (reset/step/observation_space/action_space) | Gymnasium interface is standard pattern with clear documentation. qb-rl reference shows exact implementation. |
| ENV-02 | Action space is Discrete(K+1): action 0 = WAIT, actions 1..K = buzz with option i | Standard discrete action encoding. WAIT=0 convention simplifies forced-choice logic at episode end. |
| ENV-03 | Environment computes belief features per step: belief[K], top_p, margin, entropy, stability, progress | Belief feature extraction is pure math with no ambiguity. qb-rl's extract_belief_features() handles all edge cases. |
| ENV-04 | Configurable reward modes: time_penalty (R = ±1 - penalty*t/T), simple (±1), human_grounded | Three reward modes support different training objectives. time_penalty is default, human_grounded uses human buzz position data. |
| ENV-05 | Environment accepts any LikelihoodModel for belief computation via factory | Abstract base class pattern enables swapping TF-IDF, SBERT, or future T5 without changing environment code. |
| LIK-01 | Abstract LikelihoodModel ABC with `score(clue_prefix, option_profiles) -> ndarray[K]` | Abstract interface enforces consistent contract: text → similarity scores (not probabilities, environment does softmax). |
| LIK-02 | TfIdfLikelihood implementation using sklearn TfidfVectorizer | Standard sklearn API. Requires fit() on corpus before use. Fast baseline for smoke tests. |
| LIK-03 | SBERTLikelihood implementation using sentence-transformers (all-MiniLM-L6-v2) | sentence-transformers library provides encode() method. Caching essential to avoid re-encoding same texts. |
| LIK-06 | Factory function `build_likelihood_from_config()` constructs model from YAML | Factory pattern reads config["likelihood"]["model"] and instantiates appropriate class. Handles corpus_texts for TF-IDF. |
| CFG-02 | Factory methods for all components: `make_env_from_config()`, `build_likelihood_from_config()` | Factory functions provide single point of configuration → object construction. Environment factory calls likelihood factory. |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| gymnasium | 1.1.0+ | RL environment interface | Successor to OpenAI Gym, required for SB3 integration, actively maintained |
| numpy | 1.26.4 (NOT 2.0+) | Numerical arrays | Universal standard, but must avoid NumPy 2.0 (breaks scikit-learn) |
| scikit-learn | 1.5.0+ | TF-IDF vectorization | Industry standard for text features, TfidfVectorizer well-optimized |
| sentence-transformers | 3.3.0+ | SBERT embeddings | Best lightweight semantic similarity, no API costs unlike OpenAI |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| PyYAML | 6.0+ | Configuration loading | Already used in Phase 1, continue for consistency |
| scipy | 1.13.0+ | Statistical functions (entropy) | Optional optimization for entropy computation, NumPy fallback works |

### Installation
```bash
# Already in environment from Phase 1
pip install gymnasium>=1.1.0
pip install scikit-learn>=1.5.0
pip install sentence-transformers>=3.3.0
```

## Architecture Patterns

### Recommended Project Structure
```
qb_env/
├── __init__.py
├── tossup_env.py        # TossupMCEnv class, make_env_from_config factory
models/
├── __init__.py
├── likelihoods.py       # LikelihoodModel ABC, TfIdf, SBERT, factory
├── features.py          # extract_belief_features, entropy_of_distribution
```

### Pattern 1: Gymnasium Environment Interface
**What:** Standard RL environment implementing reset(), step(), action_space, observation_space
**When to use:** All RL environments to ensure compatibility with training libraries (SB3, RLlib)
**Example:**
```python
# Source: qb-rl/qb_env/tossup_env.py (verified working)
class TossupMCEnv(gym.Env[np.ndarray, int]):
    metadata = {"render_modes": []}

    def __init__(self, questions: list[MCQuestion], likelihood_model: LikelihoodModel, K: int = 4, ...):
        self.action_space = spaces.Discrete(K + 1)  # 0=WAIT, 1..K=buzz
        # belief[K] + (top_p, margin, entropy, stability, progress, clue_idx_norm)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(K + 6,), dtype=np.float32
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        """Reset to random question, return initial observation."""
        self.question = self.rng.choice(self.questions)
        self.step_idx = 0
        self.belief = np.ones(self.K) / self.K  # uniform prior
        return self._obs(), {"qid": self.question.qid}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Take action, return (obs, reward, terminated, truncated, info)."""
        if action == 0:  # WAIT
            self.belief = self._compute_belief(self.question, self.step_idx)
            self.step_idx += 1
            if self.step_idx >= len(self.question.run_indices):
                # Force answer at end
                forced_choice = int(np.argmax(self.belief))
                reward = self._buzz_reward(forced_choice)
                return self._obs(), reward, False, True, {"forced_choice": forced_choice}
            return self._obs(), -self.wait_penalty, False, False, {}
        else:  # BUZZ with option (action - 1)
            chosen_idx = action - 1
            reward = self._buzz_reward(chosen_idx)
            return self._obs(), reward, True, False, {"chosen_idx": chosen_idx}
```

### Pattern 2: Abstract Likelihood Model with Caching
**What:** ABC defines score() interface, concrete classes implement scoring strategies
**When to use:** Supporting multiple text similarity approaches (TF-IDF, SBERT, T5)
**Example:**
```python
# Source: qb-rl/models/likelihoods.py (verified working)
class LikelihoodModel(ABC):
    def __init__(self):
        self.embedding_cache: dict[str, np.ndarray] = {}

    @abstractmethod
    def score(self, clue_prefix: str, option_profiles: list[str]) -> np.ndarray:
        """Return raw similarity scores (NOT probabilities).
        Environment converts to probabilities via softmax with beta temperature."""

    def embed_and_cache(self, texts: list[str]) -> np.ndarray:
        """Cache embeddings using SHA256 hash of text as key."""
        missing = [t for t in texts if _text_key(t) not in self.embedding_cache]
        if missing:
            new_embeddings = self._embed_batch(missing)
            for text, emb in zip(missing, new_embeddings):
                self.embedding_cache[_text_key(text)] = emb.astype(np.float32)
        return np.stack([self.embedding_cache[_text_key(t)] for t in texts])

class SBERTLikelihood(LikelihoodModel):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        super().__init__()
        from sentence_transformers import SentenceTransformer
        self.encoder = SentenceTransformer(model_name)

    def _embed_batch(self, texts: list[str]) -> np.ndarray:
        return self.encoder.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True
        ).astype(np.float32)

    def score(self, clue_prefix: str, option_profiles: list[str]) -> np.ndarray:
        clue_emb = self.embed_and_cache([clue_prefix])[0]
        option_embs = self.embed_and_cache(option_profiles)
        return (option_embs @ clue_emb).astype(np.float32)  # cosine similarity
```

### Pattern 3: Belief Feature Extraction
**What:** Convert belief distribution into rich feature vector for policy network
**When to use:** Every environment observation step
**Example:**
```python
# Source: qb-rl/models/features.py (verified working)
def extract_belief_features(
    belief: np.ndarray,
    prev_belief: np.ndarray | None,
    step_idx: int,
    total_steps: int,
) -> np.ndarray:
    """Extract belief features for policy network.

    Returns:
        Array of shape (K + 6,) containing:
        - belief[K]: probability distribution over options
        - top_p: max probability
        - margin: difference between top and second
        - entropy: information-theoretic uncertainty
        - stability: L1 distance from previous belief
        - progress: step_idx / total_steps
        - clue_idx_norm: normalized clue position
    """
    top_p = float(np.max(belief))
    sorted_probs = np.sort(belief)[::-1]
    margin = top_p - (sorted_probs[1] if len(sorted_probs) > 1 else 0.0)

    # Entropy with numerical stability
    clipped = np.clip(belief, 1e-12, 1.0)
    entropy = float(-(clipped * np.log(clipped)).sum())

    # Stability: how much belief changed from previous step
    stability = float(np.abs(belief - prev_belief).sum()) if prev_belief is not None else 0.0

    progress = float(step_idx / max(1, total_steps))
    clue_idx_norm = float(step_idx / max(1, total_steps - 1))

    extras = np.array([top_p, margin, entropy, stability, progress, clue_idx_norm], dtype=np.float32)
    return np.concatenate([belief, extras]).astype(np.float32)
```

### Pattern 4: Factory-Based Configuration
**What:** Factory functions construct components from YAML config dictionaries
**When to use:** All component instantiation to enable experiment configuration
**Example:**
```python
# Source: qb-rl/models/likelihoods.py and qb_env/tossup_env.py (verified working)
def build_likelihood_from_config(
    config: dict, corpus_texts: list[str] | None = None
) -> LikelihoodModel:
    """Construct likelihood model from config."""
    cfg = config["likelihood"]
    model_name = cfg.get("model", "sbert")

    if model_name == "tfidf":
        if not corpus_texts:
            raise ValueError("TF-IDF requires corpus_texts")
        return TfIdfLikelihood(corpus_texts=corpus_texts)
    elif model_name == "sbert":
        return SBERTLikelihood(model_name=cfg.get("sbert_name", "all-MiniLM-L6-v2"))
    else:
        raise ValueError(f"Unknown likelihood model: {model_name}")

def make_env_from_config(
    mc_questions: list[MCQuestion],
    likelihood_model: LikelihoodModel,
    config: dict,
) -> TossupMCEnv:
    """Construct environment from config."""
    env_cfg = config["environment"]
    data_cfg = config["data"]
    lik_cfg = config["likelihood"]

    return TossupMCEnv(
        questions=mc_questions,
        likelihood_model=likelihood_model,
        K=int(data_cfg.get("K", 4)),
        reward_mode=str(env_cfg.get("reward", "time_penalty")),
        wait_penalty=float(env_cfg.get("wait_penalty", 0.01)),
        buzz_correct=float(env_cfg.get("buzz_correct", 1.0)),
        buzz_incorrect=float(env_cfg.get("buzz_incorrect", -0.5)),
        belief_mode=str(env_cfg.get("belief_mode", "from_scratch")),
        beta=float(lik_cfg.get("beta", 5.0)),
    )
```

### Anti-Patterns to Avoid

- **Environment owns neural networks**: Environment should only compute beliefs via abstract LikelihoodModel interface. Policy networks live in agents, not environment.
- **Raw text observations**: Passing question text as observation breaks SB3 compatibility. Extract numeric belief features, optionally augment in agent layer.
- **Softmax in likelihood model**: Likelihood models return raw similarity scores. Environment applies softmax with temperature parameter beta.
- **Hard-coded hyperparameters**: All reward coefficients, belief modes, and model settings must come from config for experiment flexibility.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| RL environment interface | Custom environment base class | gymnasium.Env | SB3 and all modern RL libraries expect Gymnasium interface. Custom interfaces break compatibility. |
| Text vectorization | Manual term frequency counting | sklearn.TfidfVectorizer | Handles edge cases: empty documents, IDF smoothing, L2 normalization. 1000+ LOC to replicate. |
| Semantic embeddings | Fine-tune BERT from scratch | sentence-transformers library | Pre-trained models (all-MiniLM-L6-v2) already optimized for sentence similarity. Training requires massive GPU budget. |
| Entropy computation | Manual probability × log | scipy.stats.entropy or NumPy | Numerical stability is tricky. Clipping, zero-handling, NaN edge cases already solved. |
| Observation space validation | Manual shape assertions | gymnasium.spaces.Box | Gymnasium spaces provide automatic validation, error messages, dtype conversion. |

**Key insight:** RL environment design has many edge cases (forced termination, seed propagation, info dict structure). Gymnasium's interface has been battle-tested on thousands of environments. Following the standard saves debugging time.

## Common Pitfalls

### Pitfall 1: Belief State Collapse in Early Training
**What goes wrong:** Likelihood models output uniform distributions early, causing belief features (margin=0, entropy=max) to be uninformative. PPO can't learn from constant features.
**Why it happens:** TF-IDF/SBERT models need sufficient answer profile data. With small profiles or poor initialization, all options score similarly.
**How to avoid:**
- Pre-compute answer profiles on full dataset (Phase 1 already does this)
- Use softmax temperature beta=5.0 (default) to amplify score differences
- Monitor belief entropy in first 10 episodes — if always max, stop and debug
- Minimum 50 tokens per answer profile (Phase 1 default is 2000)
**Warning signs:** If `margin` feature has mean < 0.01 for >100 episodes, belief has collapsed. Check likelihood model fit() was called with corpus.

### Pitfall 2: NumPy 2.0 Compatibility Break
**What goes wrong:** scikit-learn 1.5.x fails to import with NumPy 2.0+, breaking TF-IDF likelihood model.
**Why it happens:** NumPy 2.0 changed internal APIs. scikit-learn needs patches to support it. Many transitive dependencies also break.
**How to avoid:**
- Pin numpy<2.0.0 in requirements
- Verify with `python -c "import sklearn; print(sklearn.__version__)"` after install
- Use numpy==1.26.4 (last stable 1.x release)
**Warning signs:** ImportError or AttributeError when importing sklearn after NumPy 2.0 install.

### Pitfall 3: TF-IDF Not Fit Before Use
**What goes wrong:** TfIdfLikelihood.score() raises RuntimeError because fit() wasn't called with corpus.
**Why it happens:** TF-IDF needs to learn vocabulary and IDF weights from corpus before scoring. Forgetting to fit is common when refactoring.
**How to avoid:**
- Factory function checks if corpus_texts provided for TF-IDF
- Add `_is_fit` flag that score() checks before running
- Include smoke test that creates TF-IDF model and immediately scores
**Warning signs:** RuntimeError "TfIdfLikelihood must be fit() before score()" on first environment reset.

### Pitfall 4: Embedding Cache Memory Leak with SBERT
**What goes wrong:** embedding_cache grows unbounded as new clue prefixes are seen, eventually OOM after thousands of episodes.
**Why it happens:** Each question has ~6 clue prefixes (cumulative: "clue1", "clue1 clue2", ...). With 10K questions, cache stores 60K embeddings × 384 dims = 90MB+.
**How to avoid:**
- Implement LRU cache with max_size=10000 entries
- Pre-compute and cache embeddings for all cumulative_prefixes during dataset loading
- Monitor cache size: `len(model.embedding_cache)` should plateau
**Warning signs:** Memory usage grows linearly with episodes. Cache size > 100K entries.

### Pitfall 5: Observation Space Dimension Mismatch
**What goes wrong:** Policy network forward pass fails with "expected input size X, got Y" because observation shape doesn't match declared space.
**Why it happens:** observation_space declared as (K+6,) but extract_belief_features returns different shape if logic changes.
**How to avoid:**
- Assert observation shape in _obs(): `assert obs.shape == self.observation_space.shape`
- Add unit test that resets environment and validates observation shape
- Document observation space layout in docstring with exact feature order
**Warning signs:** Policy training crashes with dimension mismatch on first batch.

## Code Examples

Verified patterns from qb-rl reference implementation:

### Reward Mode Implementation
```python
# Source: qb-rl/qb_env/tossup_env.py lines 127-137
def _buzz_reward(self, question: MCQuestion, chosen_idx: int, last_seen_step: int) -> float:
    """Compute reward for buzzing with chosen answer index."""
    correct = chosen_idx == question.gold_index

    if self.reward_mode == "simple":
        return 1.0 if correct else -1.0

    if self.reward_mode == "human_grounded":
        token_pos = self._step_to_token_pos(last_seen_step)
        if self._sampled_human_buzz_pos is not None and token_pos > self._sampled_human_buzz_pos:
            return 0.0  # Penalize buzzing after human would have
        return self.buzz_correct if correct else self.buzz_incorrect

    # Default: time_penalty mode
    # Note: wait_penalty is deducted per WAIT step in step() method
    return self.buzz_correct if correct else self.buzz_incorrect
```

### Belief Computation with Sequential Bayes
```python
# Source: qb-rl/qb_env/tossup_env.py lines 88-108
def _compute_belief(self, question: MCQuestion, step_idx: int) -> np.ndarray:
    """Compute belief distribution over answer options."""
    if self.belief_mode == "from_scratch":
        # Recompute from all clues seen so far
        prefix = question.cumulative_prefixes[step_idx]
        scores = self.likelihood_model.score(prefix, question.option_profiles)
        return self._softmax_scores(scores)

    if self.belief_mode == "sequential_bayes":
        # Bayesian update using only new clue fragment
        idx = question.run_indices[step_idx]
        prev_idx = question.run_indices[step_idx - 1] if step_idx > 0 else -1
        frag = " ".join(question.tokens[prev_idx + 1 : idx + 1])

        scores = self.likelihood_model.score(frag, question.option_profiles)
        likelihood = self._softmax_scores(scores)

        posterior = self.belief * likelihood  # Bayesian update
        denom = posterior.sum()
        if denom <= 0:
            posterior = np.ones(self.K) / self.K  # Fallback to uniform
        else:
            posterior = posterior / denom
        return posterior.astype(np.float32)

    raise ValueError(f"Unknown belief_mode: {self.belief_mode}")

def _softmax_scores(self, scores: np.ndarray) -> np.ndarray:
    """Convert scores to probabilities with temperature scaling."""
    stable = scores - np.max(scores)  # Numerical stability
    probs = np.exp(self.beta * stable)
    probs_sum = np.sum(probs)
    if probs_sum <= 0:
        return np.ones_like(scores) / len(scores)  # Uniform fallback
    return (probs / probs_sum).astype(np.float32)
```

### TF-IDF Likelihood with Corpus Fitting
```python
# Source: qb-rl/models/likelihoods.py lines 40-59
class TfIdfLikelihood(LikelihoodModel):
    def __init__(self, corpus_texts: list[str] | None = None):
        super().__init__()
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self._is_fit = False
        if corpus_texts:
            self.fit(corpus_texts)

    def fit(self, corpus_texts: list[str]) -> "TfIdfLikelihood":
        """Fit vectorizer on corpus to learn vocabulary and IDF weights."""
        self.vectorizer.fit(corpus_texts)
        self._is_fit = True
        return self

    def score(self, clue_prefix: str, option_profiles: list[str]) -> np.ndarray:
        if not self._is_fit:
            raise RuntimeError("TfIdfLikelihood must be fit() before score().")

        clue_vec = self.vectorizer.transform([clue_prefix])
        option_vecs = self.vectorizer.transform(option_profiles)
        sims = cosine_similarity(clue_vec, option_vecs)[0]
        return sims.astype(np.float32)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| OpenAI Gym | Gymnasium | 2022 (Gym 0.26+) | Gymnasium is the maintained fork. Use `gym.Env` → `gymnasium.Env`, `gym.spaces` → `gymnasium.spaces` |
| Custom episode termination flags | Gymnasium terminated/truncated | 2022 | step() returns 5-tuple: (obs, reward, terminated, truncated, info). Old Gym returned 4-tuple with done. |
| Manual softmax implementation | scipy.special.softmax | Always available | Use scipy if installed, but manual is fine for small arrays with proper stabilization (subtract max). |
| BERT sentence embeddings | sentence-transformers | 2019+ | sentence-transformers wraps BERT/RoBERTa with sentence pooling. Easier API than raw Transformers library. |

**Deprecated/outdated:**
- **OpenAI Gym (gym package)**: Unmaintained since 2022. Use gymnasium instead.
- **TF-IDF without stopwords**: Modern best practice always uses stop_words="english" to remove noise.
- **Unnormalized cosine similarity**: Always normalize embeddings before computing cosine to avoid magnitude bias.

## Validation Architecture

> Workflow setting: nyquist_validation not found in config, assuming enabled

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest 8.0+ |
| Config file | None — see Wave 0 |
| Quick run command | `pytest tests/test_environment.py -v -x` |
| Full suite command | `pytest tests/ -v` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| ENV-01 | TossupMCEnv implements Gymnasium interface | unit | `pytest tests/test_environment.py::test_gymnasium_interface -x` | ❌ Wave 0 |
| ENV-02 | Action space is Discrete(K+1) with correct semantics | unit | `pytest tests/test_environment.py::test_action_space -x` | ❌ Wave 0 |
| ENV-03 | Belief features computed correctly | unit | `pytest tests/test_features.py::test_belief_features -x` | ❌ Wave 0 |
| ENV-04 | Three reward modes produce expected values | unit | `pytest tests/test_environment.py::test_reward_modes -x` | ❌ Wave 0 |
| ENV-05 | Environment accepts pluggable likelihood models | integration | `pytest tests/test_environment.py::test_likelihood_models -x` | ❌ Wave 0 |
| LIK-01 | Abstract interface enforces score() signature | unit | `pytest tests/test_likelihoods.py::test_abstract_interface -x` | ❌ Wave 0 |
| LIK-02 | TfIdfLikelihood fits and scores correctly | unit | `pytest tests/test_likelihoods.py::test_tfidf -x` | ❌ Wave 0 |
| LIK-03 | SBERTLikelihood produces valid scores | unit | `pytest tests/test_likelihoods.py::test_sbert -x` | ❌ Wave 0 |
| LIK-06 | Factory builds models from config | integration | `pytest tests/test_factories.py::test_likelihood_factory -x` | ❌ Wave 0 |
| CFG-02 | make_env_from_config constructs environment | integration | `pytest tests/test_factories.py::test_env_factory -x` | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `pytest tests/test_environment.py tests/test_likelihoods.py -v -x`
- **Per wave merge:** `pytest tests/ -v`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/test_environment.py` — covers ENV-01, ENV-02, ENV-04, ENV-05
- [ ] `tests/test_likelihoods.py` — covers LIK-01, LIK-02, LIK-03
- [ ] `tests/test_features.py` — covers ENV-03
- [ ] `tests/test_factories.py` — covers LIK-06, CFG-02
- [ ] `tests/conftest.py` — shared fixtures (sample MCQuestions, mock config)
- [ ] Framework install: `pip install pytest>=8.0` — if not already present

## Sources

### Primary (HIGH confidence)
- qb-rl codebase (/Users/ankit.aggarwal/Dropbox/Stanford/CS234/final_project/qb-rl/) — verified working implementation of all components
- Gymnasium 1.1.0 documentation — official API reference for Env interface
- scikit-learn 1.5.0 documentation — TfidfVectorizer API and parameters
- sentence-transformers 3.3.0 documentation — SentenceTransformer.encode() API

### Secondary (MEDIUM confidence)
- Phase 1 implementation (qb_data package) — MCQuestion dataclass structure, config loading patterns
- PROJECT research files (SUMMARY.md, ARCHITECTURE.md, PITFALLS.md) — identified pitfalls and architectural patterns

### Tertiary (contextual)
- qanta-buzzer environment.py — alternative implementation showing POMDP patterns (simpler but less battle-tested than qb-rl)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - All libraries verified in qb-rl pyproject.toml, versions confirmed compatible
- Architecture: HIGH - Complete reference implementation in qb-rl, all patterns proven in production
- Code examples: HIGH - Direct copy from qb-rl with verified line numbers and working code
- Pitfalls: HIGH - Belief collapse, NumPy 2.0 break, TF-IDF fit() documented in qb-rl CONCERNS.md
- Integration: HIGH - Phase 1 provides MCQuestion dataclass, YAML config system already working

**Research date:** 2026-02-25
**Valid until:** 2026-03-27 (30 days, stable domain)
