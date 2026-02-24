# External Integrations

**Analysis Date:** 2026-02-23

## APIs & External Services

**Hugging Face Hub:**
- Service: Pre-trained model repository
  - Model: T5-large (encoder-decoder transformer)
  - What it's used for: Language understanding and answer selection
  - SDK/Client: `transformers` library (Hugging Face)
  - Authentication: Public models (no API key required)
  - How accessed: `T5ForConditionalGeneration.from_pretrained()`, `T5Tokenizer.from_pretrained()`
  - Files: `model.py` (lines ~80-90)

**QANTA Dataset:**
- Service: Public quiz bowl question corpus
  - Format: CSV file (`questions.csv`)
  - What it's used for: Quiz bowl question and answer training data
  - Access: Local file loading (pre-downloaded)
  - Files: `dataset.py` (QANTADatasetLoader class)

## Data Storage

**Databases:**
- Not applicable - No database connections
- All data stored as local JSON files
- No ORM or database client libraries

**File Storage:**
- Local filesystem only (no cloud storage)
- Storage locations:
  - `data/processed_dataset.json` - Full processed questions
  - `data/train_dataset.json` - Training split (70%, ~50 questions)
  - `data/val_dataset.json` - Validation split (15%, ~75 questions)
  - `data/test_dataset.json` - Test split (15%, ~75 questions)
  - `questions.csv` - Original QANTA CSV (~15MB)
  - `checkpoints/supervised/best_model/` - Supervised model checkpoint
  - `checkpoints/ppo/best_model/` - PPO model checkpoint
  - `results/` - Evaluation results (JSON)
  - `logs/` - Training logs (JSON)

**Caching:**
- Hugging Face transformers cache: Auto-managed in `~/.cache/huggingface/`
  - T5-large model (~3GB) cached on first download
  - No custom caching layer
- PyTorch model caching: Embedded in checkpoint system

## Authentication & Identity

**Auth Provider:**
- Not applicable - No authentication required
- Hugging Face Hub access is public/anonymous
- All model downloads unrestricted

## Monitoring & Observability

**Error Tracking:**
- Not integrated - No external error tracking service
- Errors logged to console output

**Logs:**
- Local file system logging approach
  - Files saved to `logs/` directory as JSON
  - Training history saved: `training_history.json`
  - Evaluation metrics saved: `eval_results.json`
  - No centralized logging service (Splunk, CloudWatch, etc.)
  - Optional: W&B integration commented out in `requirements.txt` (wandb>=0.15.0)

**Optional Monitoring (Disabled by Default):**
- Weights & Biases (W&B) logging - Dependency listed as optional/commented
  - To enable: Uncomment `wandb>=0.15.0` in `requirements.txt`
  - Not currently integrated in training code
  - Would require API key setup if enabled

## CI/CD & Deployment

**Hosting:**
- Not deployed - Research/experimental project
- Runs locally on development machines
- No cloud platform deployment

**CI Pipeline:**
- None - No automated CI/CD pipeline
- Manual testing via command-line scripts
- Test suite: `test_csv_loader.py`, `test_imports.py`, `test_dataset.json`

## Environment Configuration

**Required env vars:**
- None - No environment variables required
- All configuration in `config.py`
- Device selection automatic (CUDA > MPS > CPU)

**Secrets location:**
- Not applicable - No external secrets
- No `.env` file or secrets management system

## Webhooks & Callbacks

**Incoming:**
- None - Not applicable

**Outgoing:**
- None - Not applicable

## Model & Data Sources

**Model Source:**
- Hugging Face Hub: T5-large transformer
- Downloaded via `transformers.AutoModel.from_pretrained()`
- License: Apache 2.0 (T5 model)

**Training Data Source:**
- QANTA Quiz Bowl Dataset (public)
- Local CSV file: `questions.csv`
- Original source: https://www.qanta.org/

## Import Dependencies (External Packages Only)

**Direct Imports from External Packages:**
- `torch.*` - PyTorch (torch.nn, torch.optim, torch.utils.data)
- `transformers.*` - Hugging Face Transformers
- `numpy.*` - NumPy arrays and operations
- `sklearn.metrics` - scikit-learn metrics
- `datasets.*` - Hugging Face Datasets (optional)
- `tqdm` - Progress bars
- `jsonlines` - JSON utilities
- `matplotlib.*` - Visualization (optional)
- `seaborn.*` - Statistical visualization (optional)
- `pandas.*` - Data manipulation (optional)

---

*Integration audit: 2026-02-23*
