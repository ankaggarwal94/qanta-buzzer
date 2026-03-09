# Phase 3: Baseline Agents and T5 Likelihood - Research

**Researched:** 2026-02-25
**Domain:** Baseline agent implementations and T5 encoder for semantic similarity
**Confidence:** HIGH

## Summary

Phase 3 builds four baseline agents (ThresholdBuzzer, AlwaysBuzzFinal, SoftmaxProfile, SequentialBayes) and integrates T5-large as a likelihood model for semantic similarity scoring. The baseline agents provide performance floors for comparison with the MLP PPO policy (Phase 4). The T5 likelihood model is the novel contribution — it uses pre-trained semantic understanding to compute beliefs, which the MLP policy then learns from.

All agents implement a common pattern: iterate through cumulative clue prefixes, compute beliefs via a likelihood model, track buzz probability (c_trace) and correctness (g_trace) at each step, and return an EpisodeResult dataclass. The qb-rl reference implementation provides fully working versions that can be ported directly with only import path adjustments. The agents differ in their decision strategies: ThresholdBuzzer uses a configurable confidence threshold, AlwaysBuzzFinal always waits until the last clue, SoftmaxProfile recomputes belief from scratch each step, and SequentialBayes applies incremental Bayesian updates.

T5 as a likelihood model extracts encoder embeddings (mean pooling over sequence length) and computes cosine similarity between clue prefix embeddings and option profile embeddings. The critical insight is that T5Likelihood must inherit from LikelihoodModel (Phase 2 ABC) and return raw similarity scores, not probabilities — the environment applies softmax with beta temperature. Embedding caching (LIK-05) is already built into the LikelihoodModel base class via SHA-256 text hashing, so T5Likelihood only needs to implement _embed_batch() using the T5EncoderModel. Memory management is critical: detach tensors and move to CPU immediately after embedding to prevent GPU memory leaks.

**Primary recommendation:** Port all four baseline agents directly from qb-rl (agents/threshold_buzzer.py, agents/softmax_profile_buzzer.py), adjusting only import paths (models.likelihoods → models.likelihoods, qb_env.mc_builder → qb_data.mc_builder). Implement T5Likelihood following the exact pattern of SBERTLikelihood with mean pooling and cosine similarity. Add comprehensive tests for agent execution (valid episode results, correct traces) and T5 semantic scoring (verify it scores "first president" higher for "Washington" than "Lincoln"). Use T5-base (220M params) not T5-large (770M) if GPU memory is constrained.

## Phase Requirements

<phase_requirements>
| ID | Description | Research Support |
|----|-------------|-----------------|
| AGT-02 | ThresholdBuzzer baseline (sweeps configurable thresholds on top_p) | Direct port from qb-rl agents/threshold_buzzer.py lines 30-96; uses sigmoid confidence proxy and threshold comparison |
| AGT-03 | AlwaysBuzzFinalBuzzer baseline (buzzes on last clue) | Direct port from qb-rl agents/threshold_buzzer.py lines 99-141; sets c_trace[:-1]=0.0, c_trace[-1]=1.0 |
| AGT-04 | SoftmaxProfileBuzzer baseline with explicit scoring | Direct port from qb-rl agents/softmax_profile_buzzer.py lines 28-91; recomputes belief from cumulative prefix each step |
| AGT-05 | SequentialBayesBuzzer baseline with Bayesian updates | Direct port from qb-rl agents/bayesian_buzzer.py (softmax_profile_buzzer.py lines 94-159); multiplies prior by fragment likelihood |
| AGT-06 | All agents produce episode traces with c_trace (buzz probability) and g_trace (correctness) | Common pattern: c_trace and g_trace lists built per-step, returned in EpisodeResult/SoftmaxEpisodeResult dataclass |
| LIK-04 | T5Likelihood implementation using T5 encoder for semantic similarity scoring | Use transformers T5EncoderModel + T5Tokenizer; extract last hidden state, mean pool over sequence, compute cosine similarity; inherit from LikelihoodModel ABC |
| LIK-05 | Embedding cache with text hashing for SBERT and T5 models | Already implemented in LikelihoodModel.embed_and_cache() (models/likelihoods.py lines 96-118); T5Likelihood inherits this automatically |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| transformers | 4.45.0+ | T5 model loading and tokenization | Official HuggingFace library, automatic downloads, excellent T5 support, widely used in NLP |
| torch | 2.3.0+ | T5 encoder inference and tensor operations | PyTorch powers transformers models, MPS support for Mac GPU, better debugging than TF |
| numpy | <2.0.0 | Numeric operations for beliefs and features | Universal ML array library; NumPy 2.0 breaks many dependencies so pin to 1.x |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| sentence-transformers | 3.3.0+ | Existing SBERT likelihood for comparison | Already used in Phase 2; provides reference for T5 implementation pattern |
| sklearn | 1.3.0+ | TF-IDF baseline for fast testing | Already used in Phase 2 TfIdfLikelihood; agents tested with this first |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| T5-large (770M) | T5-base (220M) or T5-small (60M) | Base: 3x faster, 1/3 memory, slightly lower semantic quality; use if GPU VRAM < 8GB |
| Mean pooling | CLS token or last token | T5 has no CLS token by design; mean pooling is standard for sentence-level embeddings |
| T5EncoderModel | Full T5ForConditionalGeneration | Encoder-only is 2x faster and uses half the memory; we don't need the decoder for similarity scoring |

**Installation:**
```bash
# transformers and torch already in requirements.txt from Phase 2
# No new dependencies needed for Phase 3
pip install -r requirements.txt
```

## Architecture Patterns

### Recommended Agent Structure
```
agents/
├── __init__.py              # Export all agent classes and EpisodeResult
├── threshold_buzzer.py      # ThresholdBuzzer, AlwaysBuzzFinalBuzzer, sweep_thresholds, EpisodeResult
└── bayesian_buzzer.py       # SoftmaxProfileBuzzer, SequentialBayesBuzzer, SoftmaxEpisodeResult
```

Note: qb-rl splits agents across two files (threshold_buzzer.py and softmax_profile_buzzer.py) but softmax_profile_buzzer.py also contains SequentialBayesBuzzer and exports are in bayesian_buzzer.py. Follow qb-rl structure exactly for consistency.

### Pattern 1: Baseline Agent Execution
**What:** All baseline agents follow a common execution pattern: iterate through clue steps, compute beliefs via likelihood model, track traces, decide when to buzz.

**When to use:** Every baseline agent (Threshold, AlwaysBuzzFinal, SoftmaxProfile, SequentialBayes).

**Example:**
```python
# Source: qb-rl agents/threshold_buzzer.py lines 54-96
from dataclasses import dataclass
import numpy as np
from models.likelihoods import LikelihoodModel
from qb_data.mc_builder import MCQuestion

@dataclass
class EpisodeResult:
    qid: str
    buzz_step: int
    buzz_index: int
    gold_index: int
    correct: bool
    reward_like: float
    c_trace: list[float]
    g_trace: list[float]
    top_p_trace: list[float]
    entropy_trace: list[float]

class ThresholdBuzzer:
    def __init__(
        self,
        likelihood_model: LikelihoodModel,
        threshold: float = 0.8,
        beta: float = 5.0,
        alpha: float = 10.0,
    ):
        self.likelihood_model = likelihood_model
        self.threshold = threshold
        self.beta = beta
        self.alpha = alpha

    def run_episode(self, question: MCQuestion) -> EpisodeResult:
        c_trace: list[float] = []
        g_trace: list[float] = []
        top_p_trace: list[float] = []
        entropy_trace: list[float] = []

        chosen_step = len(question.cumulative_prefixes) - 1
        chosen_idx = 0

        for step_idx, prefix in enumerate(question.cumulative_prefixes):
            belief = self._belief_from_prefix(prefix, question.option_profiles)
            top_p = float(np.max(belief))
            top_idx = int(np.argmax(belief))
            entropy = float(-(np.clip(belief, 1e-12, 1.0) * np.log(np.clip(belief, 1e-12, 1.0))).sum())

            c_t = self._confidence_proxy(top_p)
            g_t = 1.0 if top_idx == question.gold_index else 0.0

            c_trace.append(c_t)
            g_trace.append(g_t)
            top_p_trace.append(top_p)
            entropy_trace.append(entropy)

            is_last = step_idx == len(question.cumulative_prefixes) - 1
            if top_p >= self.threshold or is_last:
                chosen_step = step_idx
                chosen_idx = top_idx
                break

        correct = chosen_idx == question.gold_index
        reward_like = 1.0 if correct else -0.5
        return EpisodeResult(
            qid=question.qid,
            buzz_step=chosen_step,
            buzz_index=chosen_idx,
            gold_index=question.gold_index,
            correct=correct,
            reward_like=reward_like,
            c_trace=c_trace,
            g_trace=g_trace,
            top_p_trace=top_p_trace,
            entropy_trace=entropy_trace,
        )
```

### Pattern 2: T5 Encoder for Similarity Scoring
**What:** Use T5EncoderModel to extract sequence embeddings, mean pool over sequence length, compute cosine similarity between clue and option embeddings.

**When to use:** Implementing T5Likelihood for semantic similarity scoring (LIK-04).

**Example:**
```python
# Pattern adapted from SBERTLikelihood (models/likelihoods.py lines 258-346)
# and T5 encoder best practices
from models.likelihoods import LikelihoodModel
import numpy as np
import torch
from transformers import T5EncoderModel, T5Tokenizer

class T5Likelihood(LikelihoodModel):
    def __init__(self, model_name: str = "t5-base") -> None:
        super().__init__()
        self.model_name = model_name
        self.encoder = T5EncoderModel.from_pretrained(model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder.to(self.device)
        self.encoder.eval()

    def _embed_batch(self, texts: list[str]) -> np.ndarray:
        """Embed texts using T5 encoder with mean pooling."""
        with torch.no_grad():
            encoded = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)

            outputs = self.encoder(**encoded)
            last_hidden = outputs.last_hidden_state  # (batch, seq_len, hidden_dim)

            # Mean pooling over sequence length
            # Mask padded tokens using attention_mask
            mask = encoded.attention_mask.unsqueeze(-1)  # (batch, seq_len, 1)
            masked_hidden = last_hidden * mask
            sum_hidden = masked_hidden.sum(dim=1)  # (batch, hidden_dim)
            mask_sum = mask.sum(dim=1).clamp(min=1e-9)  # (batch, 1)
            mean_pooled = sum_hidden / mask_sum  # (batch, hidden_dim)

            # L2 normalize for cosine similarity via dot product
            embeddings = torch.nn.functional.normalize(mean_pooled, p=2, dim=1)

            # CRITICAL: Detach and move to CPU to prevent memory leak
            embeddings = embeddings.detach().cpu().numpy().astype(np.float32)

        return embeddings

    def score(self, clue_prefix: str, option_profiles: list[str]) -> np.ndarray:
        """Score each option using T5 semantic cosine similarity."""
        clue_emb = self.embed_and_cache([clue_prefix])[0]
        option_embs = self.embed_and_cache(option_profiles)
        sims = option_embs @ clue_emb
        return sims.astype(np.float32)
```

### Pattern 3: Sequential Bayesian Update
**What:** Maintain prior belief, multiply by likelihood of new clue fragment, normalize to get posterior. Differs from SoftmaxProfile which recomputes from scratch.

**When to use:** Implementing SequentialBayesBuzzer (AGT-05).

**Example:**
```python
# Source: qb-rl agents/softmax_profile_buzzer.py lines 107-115
def _step_update(self, prior: np.ndarray, fragment: str, option_profiles: list[str]) -> np.ndarray:
    """Bayesian update: posterior ∝ prior × likelihood."""
    scores = self.likelihood_model.score(fragment, option_profiles)
    scores = scores - np.max(scores)  # Numerical stability
    likelihood = np.exp(self.beta * scores)
    posterior = prior * likelihood
    denom = posterior.sum()
    if denom <= 0:
        return np.ones_like(prior) / len(prior)  # Fallback to uniform
    return (posterior / denom).astype(np.float32)
```

### Pattern 4: Embedding Cache Integration
**What:** All likelihood models inherit embed_and_cache() from LikelihoodModel base class. Texts are hashed via SHA-256, cached embeddings are reused automatically.

**When to use:** T5Likelihood and any future likelihood models. No explicit caching code needed in subclass.

**Example:**
```python
# Source: models/likelihoods.py lines 96-118 (already implemented in Phase 2)
# T5Likelihood automatically inherits this by extending LikelihoodModel

def embed_and_cache(self, texts: list[str]) -> np.ndarray:
    """Embed texts, using cache for previously seen inputs."""
    missing = [text for text in texts if _text_key(text) not in self.embedding_cache]
    if missing:
        new_embeddings = self._embed_batch(missing)
        for text, emb in zip(missing, new_embeddings):
            self.embedding_cache[_text_key(text)] = emb.astype(np.float32)
    return np.stack([self.embedding_cache[_text_key(text)] for text in texts])
```

### Anti-Patterns to Avoid

- **Hard-coding agent hyperparameters**: Baseline agents should accept threshold, beta, alpha as constructor args, not hard-code values. This enables threshold sweeps.
- **Returning probabilities instead of raw scores**: LikelihoodModel.score() must return raw similarity scores, not probabilities. The environment applies softmax with configurable beta temperature.
- **Forgetting to detach tensors**: T5 embeddings must be detached and moved to CPU immediately after computation. Keeping them on GPU causes memory leaks in trajectory rollouts.
- **Using T5ForConditionalGeneration**: We only need the encoder for similarity scoring. Using the full seq2seq model wastes 2x memory and compute.
- **Different EpisodeResult schemas**: ThresholdBuzzer uses EpisodeResult, SoftmaxProfile uses SoftmaxEpisodeResult (identical fields). Keep both for qb-rl compatibility.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Baseline agent logic | Custom implementations from scratch | Direct port from qb-rl agents/ | qb-rl agents are tested, debugged, and used in published results; writing from scratch introduces bugs and wastes time |
| T5 tokenization | Manual BPE tokenizer | transformers.T5Tokenizer | HuggingFace tokenizers handle special tokens, padding, truncation correctly; manual implementation will have edge cases |
| Text embedding caching | Custom dict with manual cache invalidation | LikelihoodModel.embed_and_cache() | Base class already implements SHA-256 hashing and cache lookup; reinventing wastes effort |
| Episode trace computation | Custom per-step tracking | Follow EpisodeResult dataclass pattern | qb-rl's EpisodeResult schema is already consumed by evaluation metrics; incompatible schema breaks downstream |

**Key insight:** Phase 3 is integration, not invention. The qb-rl codebase has battle-tested baseline agents and qanta-buzzer has T5 integration patterns. Success is porting cleanly with minimal changes, not rewriting algorithms. The only novel code is T5Likelihood itself, which follows SBERTLikelihood's pattern exactly.

## Common Pitfalls

### Pitfall 1: Belief State Collapse with T5 (Early Training)
**What goes wrong:** T5 embeddings for short clue prefixes (1-2 sentences) produce uniform similarity scores across all options. Beliefs collapse to 25% each, margin=0, entropy=max. Agents can't learn.

**Why it happens:** T5 is pre-trained on full sentences and paragraphs. Very short text lacks context for semantic discrimination. All options score similarly until sufficient clue content accumulates.

**How to avoid:**
- Pre-compute answer profiles with full question text, not just answer names
- Monitor belief entropy in first 10 episodes — if always >log(K)-0.1, T5 may not be discriminating
- Consider minimum clue length threshold (2-3 clues) before T5 becomes effective
- Have TF-IDF fallback for comparison (faster and works on short text)

**Warning signs:** Test T5Likelihood on question "This author wrote..." with options ["Shakespeare", "Hemingway", "Tolstoy", "Dickens"]. If all scores within 0.05 of each other, belief will be too uniform.

### Pitfall 2: GPU Memory Leak in Trajectory Rollouts
**What goes wrong:** Training runs fine for 50 episodes, then OOM crashes. GPU memory usage grows linearly with episodes despite no batch size increase.

**Why it happens:** T5 encoder outputs keep PyTorch computation graph attached. If embeddings aren't explicitly detached, gradients accumulate across episodes even though we're in eval mode.

**How to avoid:**
- Wrap T5 forward pass in `torch.no_grad()` context
- Immediately after embeddings = encoder(...), call `embeddings.detach().cpu().numpy()`
- Never store raw torch tensors in EpisodeResult or agent state
- Monitor GPU memory: `torch.cuda.memory_allocated()` should be constant across episodes

**Warning signs:** Run 100 episodes with T5Likelihood. GPU memory usage increases monotonically. This is a leak — memory should plateau after first few episodes once cache is populated.

### Pitfall 3: Tokenizer Special Token Handling
**What goes wrong:** T5 adds `</s>` end-of-sequence token automatically. Not accounting for this in mean pooling distorts embeddings (all sequences end with same token's embedding).

**Why it happens:** T5Tokenizer appends `</s>` by default. If mean pooling includes this token equally with content tokens, semantic signal dilutes.

**How to avoid:**
- Use attention_mask for mean pooling (already shown in Pattern 2) — mask handles special tokens correctly
- Padding tokens have attention_mask=0, content tokens have attention_mask=1
- Sum over masked hidden states, divide by sum of mask (not sequence length)

**Warning signs:** Test embeddings for "president" and "president </s> </s> </s>". If embeddings differ significantly (cosine similarity <0.95), mean pooling isn't masking correctly.

### Pitfall 4: Sequential Bayes Run Index Dependency
**What goes wrong:** SequentialBayesBuzzer relies on `question.run_indices` which must be pre-computed during MC dataset construction. If run_indices is empty, agent crashes.

**Why it happens:** Sequential Bayes updates on clue fragments (differences between consecutive run indices), not full cumulative prefixes. If MCQuestion doesn't have run_indices, extraction fails.

**How to avoid:**
- Verify MCQuestion dataclass includes `run_indices: list[int]` field
- During dataset loading, ensure run_indices is populated (Phase 1 build_mc_dataset must compute this)
- Add assertion in SequentialBayesBuzzer.__init__: check first question has non-empty run_indices

**Warning signs:** SequentialBayesBuzzer works with qb-rl dataset but crashes with local dataset. Check if run_indices field exists and is populated.

### Pitfall 5: Import Path Inconsistency
**What goes wrong:** Direct port from qb-rl uses `from qb_env.mc_builder import MCQuestion` but this codebase has `qb_data.mc_builder`. Code fails to import.

**Why it happens:** qb-rl project structure differs from unified qanta-buzzer structure. Environment code lives in qb_env/, data code lives in qb_data/.

**How to avoid:**
- Replace `qb_env.mc_builder` → `qb_data.mc_builder` in all ported agent files
- Replace `models.likelihoods` → `models.likelihoods` (same path, but verify LikelihoodModel is exported)
- Run import test immediately after porting: `python -c "from agents.threshold_buzzer import ThresholdBuzzer"`

**Warning signs:** ModuleNotFoundError or ImportError when trying to run ported agents. Check import paths first before debugging logic.

## Code Examples

### Threshold Sweep Utility
```python
# Source: qb-rl agents/threshold_buzzer.py lines 144-160
from models.likelihoods import LikelihoodModel
from qb_data.mc_builder import MCQuestion

def sweep_thresholds(
    questions: list[MCQuestion],
    likelihood_model: LikelihoodModel,
    thresholds: list[float],
    beta: float = 5.0,
    alpha: float = 10.0,
) -> dict[float, list[EpisodeResult]]:
    """Run ThresholdBuzzer over multiple threshold values.

    Returns dict mapping threshold → list of episode results, one per question.
    Used for finding optimal threshold on validation set.
    """
    out: dict[float, list[EpisodeResult]] = {}
    for threshold in thresholds:
        agent = ThresholdBuzzer(
            likelihood_model=likelihood_model,
            threshold=float(threshold),
            beta=beta,
            alpha=alpha,
        )
        out[float(threshold)] = [agent.run_episode(q) for q in questions]
    return out
```

### Sigmoid Confidence Proxy
```python
# Source: qb-rl agents/threshold_buzzer.py lines 12-13, 51-52
import numpy as np

def _sigmoid(x: float) -> float:
    """Sigmoid activation for confidence proxy."""
    return float(1.0 / (1.0 + np.exp(-x)))

def _confidence_proxy(self, top_p: float) -> float:
    """Convert top probability to buzz confidence via sigmoid.

    alpha controls steepness; threshold is the inflection point.
    c_t = sigmoid(alpha * (top_p - threshold))

    This gives smooth buzz probabilities rather than hard threshold.
    """
    return _sigmoid(self.alpha * (top_p - self.threshold))
```

### T5 Factory Function
```python
# Pattern: extend build_likelihood_from_config() in models/likelihoods.py
from models.likelihoods import LikelihoodModel, TfIdfLikelihood, SBERTLikelihood
from typing import Any

def build_likelihood_from_config(
    config: dict[str, Any], corpus_texts: list[str] | None = None
) -> LikelihoodModel:
    """Construct a likelihood model from YAML configuration.

    Supports: tfidf, sbert, t5 (new).
    """
    cfg = config["likelihood"]
    model_name = cfg.get("model", "sbert")

    if model_name == "tfidf":
        if not corpus_texts:
            raise ValueError("TF-IDF likelihood requires corpus_texts.")
        return TfIdfLikelihood(corpus_texts=corpus_texts)

    if model_name == "sbert":
        sbert_name = cfg.get("sbert_name", cfg.get("embedding_model", "all-MiniLM-L6-v2"))
        return SBERTLikelihood(model_name=sbert_name)

    if model_name == "t5":
        # NEW: T5 likelihood model
        from models.likelihoods import T5Likelihood
        t5_name = cfg.get("t5_name", "t5-base")
        return T5Likelihood(model_name=t5_name)

    raise ValueError(f"Unknown likelihood model: {model_name}")
```

### Agent Module Exports
```python
# agents/__init__.py
from agents.threshold_buzzer import (
    ThresholdBuzzer,
    AlwaysBuzzFinalBuzzer,
    EpisodeResult,
    sweep_thresholds,
    result_to_dict,
)
from agents.bayesian_buzzer import (
    SoftmaxProfileBuzzer,
    SequentialBayesBuzzer,
    SoftmaxEpisodeResult,
)

__all__ = [
    "ThresholdBuzzer",
    "AlwaysBuzzFinalBuzzer",
    "SoftmaxProfileBuzzer",
    "SequentialBayesBuzzer",
    "EpisodeResult",
    "SoftmaxEpisodeResult",
    "sweep_thresholds",
    "result_to_dict",
]
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| T5ForConditionalGeneration | T5EncoderModel only | transformers 4.0+ | 2x faster inference, 50% memory reduction; decoder unnecessary for similarity scoring |
| Manual mean pooling without mask | Attention-mask-weighted pooling | Established best practice 2023+ | Correctly handles padding and special tokens; improves embedding quality |
| OpenAI embeddings (API) | Local T5 or SBERT | 2024+ privacy/cost concerns | No API cost, no data leakage, faster inference, works offline |
| Separate cache per model | Unified LikelihoodModel.embed_and_cache() | qb-rl design pattern | Code reuse, consistent behavior across likelihood models |

**Deprecated/outdated:**
- Using T5 without attention mask in mean pooling: Produces incorrect embeddings when sequences have different lengths
- Storing raw torch tensors in agent state: Causes memory leaks in trajectory rollouts
- Hard-coded thresholds in agent constructors: Prevents threshold sweep for optimal value finding

## Open Questions

1. **T5-base vs T5-large tradeoff for this dataset**
   - What we know: T5-large has 770M params (better semantic understanding), T5-base has 220M (3x faster)
   - What's unclear: Whether T5-large's quality improvement justifies 3x slower inference for quiz bowl
   - Recommendation: Start with T5-base, compare accuracy to SBERT baseline. Only upgrade to T5-large if semantic scoring significantly improves (>5% accuracy gain)

2. **Minimum clue length for T5 effectiveness**
   - What we know: T5 pre-trained on full sentences, may not discriminate on 1-2 word clues
   - What's unclear: At what clue index does T5 start outperforming TF-IDF/SBERT
   - Recommendation: Track per-step accuracy by clue index for all three likelihood models. If T5 underperforms TF-IDF before step 3, document this as a limitation

3. **Threshold sweep range and granularity**
   - What we know: qb-rl sweeps [0.5, 0.6, 0.7, 0.8, 0.9] (5 values)
   - What's unclear: Whether finer granularity (0.05 steps) or wider range (0.3-0.95) finds better optima
   - Recommendation: Start with qb-rl's range for consistency, expand only if validation accuracy varies >10% between adjacent thresholds

4. **Beta temperature optimal value**
   - What we know: qb-rl uses beta=5.0 by default; higher beta sharpens softmax distribution
   - What's unclear: Whether T5's raw similarity scores need different beta than SBERT's
   - Recommendation: Keep beta=5.0 for consistency across models. Only tune if beliefs consistently collapse (entropy always >log(K)-0.1) or over-sharpen (top_p always >0.95)

## Validation Architecture

> Phase validation testing included below (workflow.nyquist_validation is not explicitly set but pytest infrastructure exists from Phase 2)

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest 7.4.0+ |
| Config file | pytest.ini |
| Quick run command | `pytest tests/test_agents.py tests/test_likelihoods.py -x` |
| Full suite command | `pytest tests/ -v` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| AGT-02 | ThresholdBuzzer produces valid episodes, buzzes when top_p >= threshold | unit | `pytest tests/test_agents.py::test_threshold_buzzer -x` | ❌ Wave 0 |
| AGT-03 | AlwaysBuzzFinal waits until last step, c_trace[-1]=1.0 | unit | `pytest tests/test_agents.py::test_always_buzz_final -x` | ❌ Wave 0 |
| AGT-04 | SoftmaxProfile recomputes belief from cumulative prefix | unit | `pytest tests/test_agents.py::test_softmax_profile -x` | ❌ Wave 0 |
| AGT-05 | SequentialBayes applies incremental Bayesian updates | unit | `pytest tests/test_agents.py::test_sequential_bayes -x` | ❌ Wave 0 |
| AGT-06 | All agents return EpisodeResult with c_trace and g_trace | unit | `pytest tests/test_agents.py::test_episode_result_schema -x` | ❌ Wave 0 |
| LIK-04 | T5Likelihood computes semantic similarity, scores "first president" higher for "Washington" | unit | `pytest tests/test_likelihoods.py::test_t5_semantic_scoring -x` | ❌ Wave 0 |
| LIK-05 | T5Likelihood inherits embed_and_cache, reuses cached embeddings | unit | `pytest tests/test_likelihoods.py::test_t5_embedding_cache -x` | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `pytest tests/test_agents.py tests/test_likelihoods.py -x` (~10 seconds with TF-IDF, ~60 seconds with T5-base on CPU)
- **Per wave merge:** `pytest tests/ -v` (full suite from Phase 2 + new Phase 3 tests)
- **Phase gate:** Full suite green, plus manual smoke test of baseline scripts (run_baselines.py --smoke)

### Wave 0 Gaps
- [ ] `tests/test_agents.py` — covers AGT-02 through AGT-06 (baseline agent execution and traces)
- [ ] `tests/test_likelihoods.py::test_t5_semantic_scoring` — verifies T5 semantic similarity (LIK-04)
- [ ] `tests/test_likelihoods.py::test_t5_embedding_cache` — verifies cache reuse (LIK-05)
- [ ] `tests/conftest.py` — add fixtures for sample T5 model (t5-small for fast tests), sample agent configs
- [ ] `agents/__init__.py` — export all agent classes
- [ ] Framework already installed from Phase 2

## Sources

### Primary (HIGH confidence)
- qb-rl agents/threshold_buzzer.py — ThresholdBuzzer and AlwaysBuzzFinal reference implementations (lines 30-141)
- qb-rl agents/softmax_profile_buzzer.py — SoftmaxProfile and SequentialBayes reference implementations (lines 28-159)
- qb-rl models/likelihoods.py — LikelihoodModel ABC and embedding cache pattern (lines 1-38)
- qanta-buzzer models/likelihoods.py — Phase 2 implementation with SBERTLikelihood pattern to follow (lines 258-346)
- HuggingFace Transformers documentation — T5EncoderModel API, tokenization, mean pooling best practices

### Secondary (MEDIUM confidence)
- Sentence-Transformers documentation — Semantic similarity patterns (already used for SBERT in Phase 2)
- qb-rl scripts/run_baselines.py — Orchestration of baseline sweeps and evaluation (lines 44-113)

### Tertiary (LOW confidence, architectural decisions)
- Quiz bowl baseline agent patterns — Threshold-based and Bayesian strategies (inferred from qb-rl implementations)
- T5 mean pooling — Standard practice but not officially documented by HuggingFace (community best practice)

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - T5 via transformers is well-documented, versions verified from qb-rl
- Architecture: HIGH - Direct port from qb-rl with only import path changes
- Pitfalls: HIGH - Memory leaks and belief collapse documented in qb-rl CONCERNS.md
- T5 implementation details: MEDIUM - Mean pooling pattern standard but requires attention mask handling

**Research date:** 2026-02-25
**Valid until:** 2026-03-25 (30 days — transformers and PyTorch stable, baseline patterns established)
