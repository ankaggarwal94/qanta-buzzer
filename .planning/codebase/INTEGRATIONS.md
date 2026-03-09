# External Integrations

**Analysis Date:** 2026-02-24

## APIs & External Services

**Hugging Face Model Hub:**
- T5 pre-trained models (t5-large, t5-base, t5-small)
  - SDK/Client: `transformers` library (T5ForConditionalGeneration, T5Tokenizer)
  - Authentication: Public access (no credentials required)
  - Usage: `model.py` loads T5 model automatically on first run (~3GB download)
  - Location: Lines 96-97 in `model.py`: `T5ForConditionalGeneration.from_pretrained(config.MODEL_NAME)`

## Data Storage

**File Storage:**
- **Local filesystem only**
  - Training data: `questions.csv` (QANTA dataset, 14.9MB)
  - Processed datasets: `data/processed_dataset.json`, `data/{train,val,test}_dataset.json`
  - Checkpoints: `checkpoints/supervised/best_model/` and `checkpoints/ppo/best_model/`
  - Results: `results/` directory (JSON files with evaluation metrics)
  - Logs: `logs/` directory

**No Database:**
- All data is file-based (JSON, CSV)
- No external database connections
- No ORM usage

**No Caching Service:**
- In-memory caching during training (Python objects)
- No Redis, Memcached, or similar

## Authentication & Identity

**Auth Provider:**
- Not applicable (no user authentication)
- Model downloads use public Hugging Face Hub access

## Monitoring & Observability

**Error Tracking:**
- None (no Sentry or similar)

**Logs:**
- Standard Python logging via console print statements
- Training progress via tqdm progress bars
- Metrics saved to `history.json` in checkpoints
- Evaluation results dumped to JSON files in `results/`

**Supported Monitoring:**
- Manual inspection of training curves (saved in `history.json`)
- Command-line output during training
- Optional W&B integration (currently commented out in `requirements.txt`)

## CI/CD & Deployment

**Hosting:**
- Not deployed (academic project)
- Local development and evaluation only
- Can run on local machine, HPC cluster, or personal GPU machine

**CI Pipeline:**
- None (no automated CI/CD)

## Environment Configuration

**Required Environment Variables:**
- None (all hardcoded in `config.py` or overridable via CLI)

**Optional Environment Variables:**
- `CUDA_VISIBLE_DEVICES` - Control GPU visibility (if using CUDA)
- `PYTORCH_MPS_HIGH_WATERMARK_RATIO` - MPS memory optimization (if using Apple Silicon)

**Secrets Location:**
- No secrets required (public model, local data)
- `.env` file: Not used

## Webhooks & Callbacks

**Incoming:**
- None

**Outgoing:**
- None

## Optional Integrations (Disabled)

**W&B (Weights & Biases):**
- Commented out in `requirements.txt` (line 22-23)
- If enabled, would require: `wandb>=0.15.0`
- Usage: `import wandb` in training scripts
- Authentication: Would require W&B API key (not in current codebase)

**Jupyter Notebooks:**
- Commented out in `requirements.txt` (lines 24-26)
- Not used in active training pipeline
- Optional for interactive analysis

## Data Processing Pipeline

**QANTA Dataset Loading:**
- Source: `questions.csv` (local CSV file)
- Format: Comma-separated with clues separated by `|||`
- Loader: `QANTADatasetLoader` in `dataset.py`
- Process:
  1. Load from CSV (lines 90-120 in `dataset.py`)
  2. Generate multiple-choice with distractors (3 strategies: category-based 40%, embedding-based 40%, common-confusion 20%)
  3. Save to JSON format for caching
  4. Split into train/val/test (70/15/15)

**No External Data APIs:**
- All data is static, pre-downloaded
- No streaming from remote sources
- No real-time data ingestion

## Model Loading & Persistence

**First-Run Behavior:**
- T5 model downloads automatically from Hugging Face Hub on first instantiation
- Approximately 3GB for t5-large (770M params)
- Cached in torch cache directory (~/.cache/huggingface/hub/)

**Checkpoint Management:**
- Manual save/load via `T5PolicyModel.save()` and `load_pretrained()`
- Locations: `model.py` lines 370-440
- Saved artifacts:
  - T5 model: Saved via `save_pretrained()`
  - Tokenizer: Saved via `save_pretrained()`
  - Policy head: `policy_head.pt` (custom state dict)
  - Training metadata: `history.json` (loss curves, metrics)

---

*Integration audit: 2026-02-24*
