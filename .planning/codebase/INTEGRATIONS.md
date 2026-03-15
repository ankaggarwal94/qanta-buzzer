# Integrations

## External APIs

### HuggingFace Hub
- **Module:** `qb_data/huggingface_loader.py`
- **Purpose:** Fallback data source when local CSV is unavailable
- **Dataset:** QANTA quiz bowl questions via `datasets` library
- **Auth:** None required (public datasets)
- **Usage:** `load_from_huggingface()` called by `scripts/build_mc_dataset.py` when CSV path missing

### HuggingFace Model Hub
- **Module:** `models/likelihoods.py`, `models/t5_policy.py`
- **Models downloaded:**
  - T5 variants: `t5-small`, `t5-base`, `t5-large` (likelihood scoring)
  - SentenceTransformers: `all-MiniLM-L6-v2` (SBERT embeddings)
- **Cache:** Default HuggingFace cache (`~/.cache/huggingface/`)
- **Auth:** None required (public models)

### OpenAI API (Optional)
- **Module:** `models/likelihoods.py` → `OpenAILikelihood`
- **Purpose:** Alternative embedding model for answer likelihood scoring
- **Model:** `text-embedding-3-small` (configurable)
- **Auth:** `OPENAI_API_KEY` environment variable
- **Install:** `pip install -e '.[openai]'`
- **Guard:** Import-time check with helpful error message if `openai` package not installed

## Data Sources

### QANTA CSV
- **Primary data format:** CSV with `|||`-separated clues in `question`/`Text` column
- **Path:** Configured via `data.csv_path` in YAML config (default: `questions.csv`)
- **Loader:** `qb_data/data_loader.py` → `QANTADatasetLoader`
- **Fields:** question text, answer, category, optional human buzz positions

### Artifacts Directory
- **Path:** `artifacts/` (main runs), `artifacts/smoke/` (smoke tests)
- **Contents:** `mc_dataset.json`, `alias_lookup.json`, `baseline_summary.json`, `ppo_summary.json`, `evaluation_report.json`
- **Format:** JSON with custom serialization via `scripts/_common.py:to_serializable()`

## Embedding Cache
- **Module:** `models/likelihoods.py` (base class `LikelihoodModel`)
- **Strategy:** SHA-256 content hashing of input text → float32 numpy arrays
- **Storage:** In-memory dict (`embedding_cache`), no persistent disk cache for embeddings themselves
- **Config:** `likelihood.cache_embeddings` and `likelihood.cache_dir` in YAML (cache_dir used for optional on-disk persistence)

## No External Databases / Auth Providers / Webhooks

This is a research project with no:
- Database connections
- Authentication/authorization systems
- Webhook endpoints
- Message queues
- External monitoring services
