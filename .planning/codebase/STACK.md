# Technology Stack

**Analysis Date:** 2026-02-23

## Languages

**Primary:**
- Python 3.9+ - All core implementation, training, evaluation
  - Developed/tested on Python 3.13.5
  - Type hints used throughout (from `typing`)

**Secondary:**
- Bash - Build/run scripts (`run.sh`)

## Runtime

**Environment:**
- Python 3.9 minimum (specified for compatibility)
- No external runtime required (native Python execution)

**Package Manager:**
- pip - Primary package manager
- Lockfile: `requirements.txt` (present, version pinned with `>=` constraints)

## Frameworks

**Deep Learning:**
- PyTorch - Core neural network framework
  - `torch>=2.0.0` - Tensors, autograd, optimizers
  - `torch.nn` - Neural network modules
  - `torch.optim` - Optimization (AdamW, SGD)
  - Used in `model.py`, `train_supervised.py`, `train_ppo.py`

**NLP/Language Models:**
- Transformers (Hugging Face) - `transformers>=4.30.0`
  - Pre-trained T5 model loading (`T5ForConditionalGeneration`, `T5Tokenizer`)
  - Used in `model.py` for encoder-decoder architecture
  - Model: T5-large (770M parameters, ~3GB download)

**ML/Data Processing:**
- scikit-learn - `scikit-learn>=1.3.0`
  - Metrics computation (accuracy, calibration error)
  - Used in `metrics.py` for `accuracy_score`
- Hugging Face Datasets - `datasets>=2.14.0`
  - Dataset loading and processing utilities
  - Optional for extended data handling

**Progress/CLI:**
- tqdm - `tqdm>=4.65.0` - Progress bars for training loops

**Data Format:**
- jsonlines - `jsonlines>=3.1.0` - Line-delimited JSON handling

## Key Dependencies

**Critical (Core Functionality):**
- torch (>=2.0.0) - Neural network computation, gradients, device management
- transformers (>=4.30.0) - Pre-trained T5 model and tokenization
- numpy (>=1.24.0) - Numerical operations, array handling
- scikit-learn (>=1.3.0) - Evaluation metrics (accuracy, ECE)

**Essential (Training/Inference):**
- tqdm (>=4.65.0) - Progress bars, interactive feedback
- datasets (>=2.14.0) - Dataset utilities

**Data I/O:**
- jsonlines (>=3.1.0) - JSON serialization (legacy dependency, minimal use)

**Optional (Visualization/Analysis):**
- matplotlib (>=3.7.0) - Plotting, visualization
- seaborn (>=0.12.0) - Statistical visualization
- pandas (>=2.0.0) - Data analysis and manipulation

## Configuration

**Environment:**
- Configuration via `config.py` class-based approach (static attributes on `Config` class)
- No `.env` file required - configuration hardcoded in Python
- Device detection: Automatic selection (CUDA > MPS > CPU)

**Key Configuration Parameters:**
- Model: T5-large (configurable as `MODEL_NAME`)
- Batch sizes: Supervised (8), PPO (32)
- Learning rates: Supervised (5e-5), PPO (3e-5)
- Training iterations: Supervised (50 epochs), PPO (250 iterations)
- Paths: Data, checkpoints, results, logs (relative to project root)

**Build:**
- No build configuration file (pure Python, no compilation)
- Shell script wrapper: `run.sh` - Interactive training launcher

## Platform Requirements

**Development:**
- macOS/Linux/Windows with Python 3.9+
- 8GB+ RAM recommended for T5-large model
- GPU optional but recommended (CUDA, Metal Performance Shaders)
- Virtual environment (venv) for isolation

**Production:**
- Deployment: Local filesystem or cloud platforms supporting Python/PyTorch
- Model weights stored in `checkpoints/` directory (HuggingFace format)
- Data stored as JSON files in `data/` directory

## Model & Data

**Pre-trained Models:**
- T5-large encoder-decoder from Hugging Face Hub
- Downloaded on first run to default HuggingFace cache location (~3GB)

**Datasets:**
- Quiz Bowl questions with pyramidal clues (500 questions)
- Multiple-choice format (4 choices per question)
- Splits: 70% train, 15% val, 15% test
- Storage: JSON format in `data/` directory
  - `processed_dataset.json` - Full dataset
  - `train_dataset.json` - Training split
  - `val_dataset.json` - Validation split
  - `test_dataset.json` - Test split
  - `questions.csv` - Original QANTA CSV (~15MB)

**Checkpoints:**
- Model checkpoints saved as HuggingFace-compatible format in `checkpoints/` directory
  - `checkpoints/supervised/` - Supervised training results
  - `checkpoints/ppo/` - PPO training results
  - Contains: `config.json`, `pytorch_model.bin`, `tokenizer files`, `policy_head.pt`

---

*Stack analysis: 2026-02-23*
