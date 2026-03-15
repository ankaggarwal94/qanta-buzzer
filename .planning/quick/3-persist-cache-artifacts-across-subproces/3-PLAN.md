---
phase: quick-3
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - models/likelihoods.py
  - scripts/_common.py
  - scripts/run_baselines.py
  - scripts/train_ppo.py
  - scripts/evaluate_all.py
  - tests/test_likelihoods.py
autonomous: true
requirements: [OPT-2]

must_haves:
  truths:
    - "Embedding cache persists to disk as .npz after stage 2 completes"
    - "Stages 3 and 4 load the persisted cache and skip redundant transformer passes"
    - "Scores produced with cached embeddings are bitwise identical to freshly computed scores"
    - "TF-IDF models (smoke mode) skip disk caching entirely since vocab-sized vectors vary per fit"
    - "Pipeline works identically when no cache file exists (cold start)"
  artifacts:
    - path: "models/likelihoods.py"
      provides: "save_cache() and load_cache() methods on LikelihoodModel"
      contains: "def save_cache"
    - path: "scripts/_common.py"
      provides: "save_embedding_cache() and load_embedding_cache() helpers"
      contains: "def save_embedding_cache"
    - path: "tests/test_likelihoods.py"
      provides: "Round-trip save/load cache tests"
      contains: "test_save_load_cache"
  key_links:
    - from: "models/likelihoods.py"
      to: "numpy .npz format"
      via: "np.savez_compressed / np.load"
      pattern: "np\\.savez_compressed|np\\.load"
    - from: "scripts/_common.py"
      to: "models/likelihoods.py"
      via: "model.save_cache() / model.load_cache()"
      pattern: "save_cache|load_cache"
    - from: "scripts/run_baselines.py"
      to: "scripts/_common.py"
      via: "save_embedding_cache(model, config) after precompute"
      pattern: "save_embedding_cache"
    - from: "scripts/train_ppo.py"
      to: "scripts/_common.py"
      via: "load_embedding_cache(model, config) before belief computation"
      pattern: "load_embedding_cache"
---

<objective>
Add disk persistence for the LikelihoodModel embedding cache so that expensive
transformer forward passes (SBERT, T5, OpenAI) computed in stage 2
(run_baselines.py) are reused by stages 3 (train_ppo.py) and 4
(evaluate_all.py) without recomputation.

Purpose: Stages 2-4 of the pipeline each construct a fresh LikelihoodModel
with an empty in-memory cache, then re-embed the same texts from scratch.
For neural models, this wastes 2x the embedding time. The config already
declares `likelihood.cache_dir` but nothing reads/writes to it.

Output: Two new methods on LikelihoodModel (save_cache, load_cache), two
helper functions in _common.py, and integration calls in stages 2-4 scripts.
</objective>

<execution_context>
@./.claude/get-shit-done/workflows/execute-plan.md
@./.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@models/likelihoods.py
@scripts/_common.py
@scripts/run_baselines.py
@scripts/train_ppo.py
@scripts/evaluate_all.py
@tests/test_likelihoods.py
@configs/default.yaml
@configs/smoke.yaml

<interfaces>
<!-- Key types and contracts the executor needs -->

From models/likelihoods.py:
```python
class LikelihoodModel(ABC):
    def __init__(self) -> None:
        self.embedding_cache: dict[str, np.ndarray] = {}
    def embed_and_cache(self, texts: list[str]) -> np.ndarray: ...
    def precompute_embeddings(self, texts: list[str], batch_size: int = 64, desc: str = ...) -> None: ...

def _text_key(text: str) -> str:
    """SHA-256 hex digest of text."""

class TfIdfLikelihood(LikelihoodModel): ...  # _embed_batch returns vocab-sized dense vectors -- NOT cacheable across runs
class SBERTLikelihood(LikelihoodModel): ...  # Fixed-dim embeddings -- cacheable
class T5Likelihood(LikelihoodModel): ...     # Fixed-dim embeddings -- cacheable
class OpenAILikelihood(LikelihoodModel): ... # Fixed-dim embeddings -- cacheable
```

From scripts/_common.py:
```python
def build_likelihood_model(config: dict, mc_questions: list[MCQuestion]) -> LikelihoodModel: ...
def load_config(config_path: str | None = None, smoke: bool = False) -> dict: ...
```

From configs/default.yaml:
```yaml
likelihood:
  cache_dir: "cache/embeddings"   # Already declared, currently unused
```
</interfaces>
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Add save_cache / load_cache to LikelihoodModel + tests</name>
  <files>models/likelihoods.py, tests/test_likelihoods.py</files>
  <behavior>
    - Test: save_cache writes a .npz file to the given path; load_cache restores the exact same dict (round-trip fidelity)
    - Test: load_cache with a nonexistent file does nothing (no error, cache unchanged)
    - Test: save_cache with an empty cache creates a valid .npz with zero arrays
    - Test: TfIdfLikelihood.save_cache is a no-op (or skipped) because TF-IDF embeddings are vocab-dependent and not portable across fits
    - Test: Loaded cache entries produce identical scores to freshly computed entries
  </behavior>
  <action>
Add two methods to the `LikelihoodModel` base class in `models/likelihoods.py`:

```python
def save_cache(self, path: str | Path) -> int:
    """Persist embedding_cache to disk as .npz. Returns count of entries saved.
    Creates parent directories. Skips if cache is empty."""

def load_cache(self, path: str | Path) -> int:
    """Load embedding_cache from .npz on disk. Returns count of entries loaded.
    If file does not exist, silently returns 0 (cold start). Merges into
    existing cache without overwriting (existing keys win)."""
```

Implementation details:
- Use `np.savez_compressed(path, **cache)` where keys are the SHA-256 hex strings and values are float32 arrays. The SHA-256 keys are valid Python identifiers for npz (64 hex chars).
- On load: `data = np.load(path)` then iterate `data.files` and populate `self.embedding_cache` for keys not already present.
- Override `save_cache` in `TfIdfLikelihood` to be a no-op that returns 0 and prints a debug message -- TF-IDF vectors are vocabulary-specific and not reusable across separate `fit()` calls.
- Path type: accept `str | Path`, convert to `Path` internally.

Add tests to `tests/test_likelihoods.py` in a new `TestEmbeddingCachePersistence` class:
- `test_save_load_cache_round_trip`: Create SBERTLikelihood, embed 3 texts, save to tmp_path, create fresh model, load cache, verify all 3 entries restored with `np.testing.assert_array_equal`.
- `test_load_cache_missing_file`: Call load_cache with nonexistent path, verify returns 0 and cache stays empty.
- `test_save_cache_empty`: Save empty cache, verify file created and loadable.
- `test_tfidf_save_cache_noop`: Create fitted TfIdfLikelihood with cached embeddings, call save_cache, verify returns 0.
- `test_load_cache_does_not_overwrite`: Pre-populate cache with a key, load a file that has the same key with different values, verify original value preserved.
  </action>
  <verify>
    <automated>cd /Users/ankit.aggarwal/Dropbox/Stanford/CS234/final_project/qanta-buzzer && python -m pytest tests/test_likelihoods.py::TestEmbeddingCachePersistence -xvs 2>&1 | tail -30</automated>
  </verify>
  <done>
    - LikelihoodModel has save_cache() and load_cache() methods
    - TfIdfLikelihood overrides save_cache() as a no-op
    - 5 new tests pass covering round-trip, missing file, empty cache, TF-IDF no-op, and no-overwrite
    - Existing tests still pass (no regressions)
  </done>
</task>

<task type="auto">
  <name>Task 2: Integrate cache persistence into pipeline scripts</name>
  <files>scripts/_common.py, scripts/run_baselines.py, scripts/train_ppo.py, scripts/evaluate_all.py</files>
  <action>
**In `scripts/_common.py`**, add two helper functions:

```python
def embedding_cache_path(config: dict) -> Path:
    """Return the resolved embedding cache file path from config.
    Uses config['likelihood']['cache_dir'] (default 'cache/embeddings')
    and appends 'embedding_cache.npz'."""
    cache_dir = config.get("likelihood", {}).get("cache_dir", "cache/embeddings")
    return PROJECT_ROOT / cache_dir / "embedding_cache.npz"

def load_embedding_cache(model: LikelihoodModel, config: dict) -> None:
    """Load persisted embedding cache into model if file exists.
    Prints count of entries loaded."""
    path = embedding_cache_path(config)
    n = model.load_cache(path)
    if n > 0:
        print(f"Loaded {n} cached embeddings from {path}")

def save_embedding_cache(model: LikelihoodModel, config: dict) -> None:
    """Persist model's embedding cache to disk.
    Prints count of entries saved."""
    path = embedding_cache_path(config)
    n = model.save_cache(path)
    if n > 0:
        print(f"Saved {n} embeddings to {path}")
```

Import `LikelihoodModel` is already available via `build_likelihood_from_config` import; add a direct import of the class: `from models.likelihoods import LikelihoodModel, build_likelihood_from_config`.

**In `scripts/run_baselines.py`** (stage 2 -- the first stage that uses LikelihoodModel):
- After `likelihood_model = build_likelihood_model(config, mc_questions)` (line 126), add:
  `load_embedding_cache(likelihood_model, config)`
- After `likelihood_model.precompute_embeddings(all_texts, batch_size=64)` (line 145), add:
  `save_embedding_cache(likelihood_model, config)`
- Add `load_embedding_cache, save_embedding_cache` to the imports from `scripts._common`.

**In `scripts/train_ppo.py`** (stage 3):
- After `likelihood_model = build_likelihood_model(config, mc_questions)` (line 105), add:
  `load_embedding_cache(likelihood_model, config)`
- After the `precompute_beliefs` call (line ~117), add:
  `save_embedding_cache(likelihood_model, config)`
- Add `load_embedding_cache, save_embedding_cache` to the imports from `scripts._common`.

**In `scripts/evaluate_all.py`** (stage 4):
- After `likelihood_model = build_likelihood_model(config, mc_questions)` (line 162), add:
  `load_embedding_cache(likelihood_model, config)`
- No save needed in the final stage (no new embeddings computed beyond what stages 2-3 cached).
- Add `load_embedding_cache` to the imports from `scripts._common`.

**Do NOT modify `scripts/build_mc_dataset.py`** -- stage 1 does not use a LikelihoodModel. It uses MCBuilder which has its own SBERT encoder, and that is a separate concern.
  </action>
  <verify>
    <automated>cd /Users/ankit.aggarwal/Dropbox/Stanford/CS234/final_project/qanta-buzzer && python -m pytest tests/ -x --timeout=120 2>&1 | tail -20</automated>
  </verify>
  <done>
    - scripts/_common.py has embedding_cache_path(), load_embedding_cache(), save_embedding_cache()
    - run_baselines.py loads cache before precompute, saves after
    - train_ppo.py loads cache before belief computation, saves after
    - evaluate_all.py loads cache on startup
    - Full test suite passes with no regressions
    - Pipeline works identically on cold start (no cache file exists)
  </done>
</task>

</tasks>

<verification>
1. Full test suite: `scripts/ci.sh` passes
2. Regression check: `python -m pytest tests/test_likelihoods.py -x` -- all existing + new tests pass
3. Behavioral equivalence: The embedding cache is keyed by SHA-256 of input text and stores float32 arrays -- loading from disk produces the exact same numpy arrays as fresh computation
4. Cold start: When `cache/embeddings/embedding_cache.npz` does not exist, all scripts run identically to before (load_cache returns 0, pipeline proceeds normally)
5. TF-IDF safety: In smoke mode (tfidf model), save_cache is a no-op -- no spurious files created
</verification>

<success_criteria>
- LikelihoodModel.save_cache() and load_cache() exist and are tested (5 new tests)
- Pipeline scripts call load/save at the right points
- `scripts/ci.sh` passes (full test suite, no regressions)
- No new dependencies added (numpy .npz is built-in)
</success_criteria>

<output>
After completion, create `.planning/quick/3-persist-cache-artifacts-across-subproces/3-SUMMARY.md`
</output>
