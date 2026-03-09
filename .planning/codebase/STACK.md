# Technology Stack

**Analysis Date:** 2026-02-24

## Languages

**Primary:**
- Python 3.11+ - All source code, training, evaluation

## Runtime

**Environment:**
- Python 3.11 or higher (specified in CLAUDE.md)

**Package Manager:**
- pip + setuptools
- Installation: `pip install -e .` in project root

## Frameworks

**Core ML/RL:**
- PyTorch 2.0.0+ - Neural network framework, distributed training primitives
- Transformers 4.30.0+ - Hugging Face transformers library for T5 model loading

**Data Processing:**
- Datasets 2.14.0+ - Data loading and preprocessing
- scikit-learn 1.3.0+ - Metrics (accuracy_score, classification metrics)
- NumPy 1.24.0+ - Numerical operations, vectorized computations

**Visualization & Analysis:**
- matplotlib 3.7.0+ - Plotting, metric visualization (`visualize.py`)
- seaborn 0.12.0+ - Statistical data visualization
- pandas 2.0.0+ - DataFrame operations for analysis

**Utilities:**
- tqdm 4.65.0+ - Progress bars for training loops
- jsonlines 3.1.0+ - JSON line format reading/writing

## Key Dependencies

**Critical (Used Daily):**
- `torch` - Core RL training, PPO updates, gradient computation
  - Location: `train_ppo.py`, `train_supervised.py`, `model.py`
  - Why: Essential for all neural network operations
- `transformers` - T5-large model loading and tokenization
  - Location: `model.py` (T5ForConditionalGeneration, T5Tokenizer)
  - Why: Pre-trained encoder backbone (770M parameters)
- `numpy` - Array operations, reward computation, rollout handling
  - Location: `train_ppo.py`, `environment.py`, `metrics.py`
  - Why: Efficient vectorized operations across batches

**Important (Used in Training/Eval):**
- `tqdm` - Progress bars during supervised and PPO training
  - Location: `train_supervised.py`, `train_ppo.py`
- `scikit-learn` - Metric computation (accuracy, calibration error)
  - Location: `metrics.py` (accuracy_score)

**Optional (Commented in requirements.txt):**
- `wandb` 0.15.0+ - Weights & Biases experiment tracking (currently disabled)
- `jupyter` 1.0.0+ - Interactive notebooks
- `ipywidgets` 8.0.0+ - Interactive widgets for Jupyter

## Configuration

**Environment:**
- Configured via `config.py` using Config class attributes
- CLI arguments in `main.py` override Config defaults
- Key env-dependent settings:
  - `DEVICE`: Auto-detected (cuda > mps > cpu) via `torch.cuda.is_available()` and `torch.backends.mps.is_available()`
  - `SEED`: Default 42, override with `--seed` CLI arg

**Build:**
- No build configuration file (native Python project)
- Training entry point: `python main.py --mode {supervised|ppo|full|eval}`
- Checkpoints saved to `checkpoints/` directory
- Results logged to `results/` and `logs/` directories

## Platform Requirements

**Development:**
- Python 3.11+
- 16GB RAM minimum (for T5-large)
- 8GB GPU VRAM recommended (for t5-large on CUDA)
- Can fall back to CPU or Apple Metal Performance Shaders (MPS)
- Supports: Linux, macOS (with MPS), Windows (with CUDA)

**Production (Inference):**
- Python 3.11+
- 8GB+ RAM
- Model size: t5-large (~3GB first download), t5-base (~1GB), t5-small (~500MB)
- Optional: GPU for faster inference

## Model Architecture

**T5 Encoder:**
- Base model: `t5-large` (770M parameters, configurable to t5-base or t5-small)
- Type: Sequence-to-sequence encoder-decoder, used encoder-only
- Download: Automatic via Hugging Face transformers on first run
- Tokenizer: T5Tokenizer (loads from same checkpoint)

**Custom Policy Head:**
- Input: T5 encoder pooled hidden state
- Outputs:
  - Wait head: 2 logits (WAIT vs SELECT NOW)
  - Answer head: 4 logits (choice selection 0-3)
  - Value head: 1 scalar (state value for PPO)
- Location: `model.py` - PolicyHead class

---

*Stack analysis: 2026-02-24*
