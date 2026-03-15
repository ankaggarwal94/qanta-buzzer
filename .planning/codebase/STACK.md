# Stack

## Language & Runtime

- **Python >= 3.11** (specified in `pyproject.toml`)
- Virtual environment: `.venv/` with Python 3.13 (local dev)
- Package manager: pip with setuptools build backend
- Install: `pip install -e .` (editable) or `pip install -r requirements.txt`

## Core Frameworks

| Framework | Version | Purpose |
|-----------|---------|---------|
| PyTorch | >= 2.0.0 | Neural network inference (T5, SBERT), PPO policy networks |
| Gymnasium | >= 1.1.0 | POMDP environment interface (`TossupMCEnv`) |
| Stable-Baselines3 | >= 2.6.0 | PPO training loop, MLP policy networks |
| Transformers | >= 4.30.0 | T5 model loading, tokenization, likelihood scoring |
| Sentence-Transformers | >= 2.2.0 | SBERT embeddings for distractor selection and scoring |

## ML / Data Libraries

| Library | Version | Purpose |
|---------|---------|---------|
| NumPy | >= 1.24.0 | Array operations, belief distributions, feature extraction |
| scikit-learn | >= 1.3.0 | TF-IDF vectorizer, cosine similarity, distractor ranking |
| pandas | >= 2.0.0 | Data loading, evaluation tables |
| datasets | >= 2.14.0 | HuggingFace dataset loading (QANTA fallback) |

## Visualization & IO

| Library | Version | Purpose |
|---------|---------|---------|
| matplotlib | >= 3.7.0 | Calibration curves, entropy plots |
| seaborn | >= 0.12.0 | Statistical plot styling |
| PyYAML | >= 6.0.0 | Config file parsing (`configs/*.yaml`) |
| jsonlines | >= 3.1.0 | Streaming JSON I/O for datasets |
| tqdm | >= 4.65.0 | Progress bars in pipeline scripts |

## Optional Dependencies

| Library | Version | Purpose |
|---------|---------|---------|
| openai | >= 1.0.0 | OpenAI embedding API for `OpenAILikelihood` (opt-in via `pip install -e '.[openai]'`) |

## Configuration

- YAML-based config system: `configs/default.yaml`, `configs/smoke.yaml`
- Config loaded via `qb_data.config.load_config()` with CLI override support
- Sections: `data`, `answer_profiles`, `likelihood`, `environment`, `mc_guards`, `bayesian`, `ppo`, `evaluation`, `supervised`

## Device Selection

- Auto-selects best accelerator: CUDA > MPS > CPU via `_best_torch_device()` in `models/likelihoods.py`
- Seeds set explicitly for reproducibility (numpy, torch, random) — convention uses seeds 1, 2, 3 or 13, 42

## Build & Packaging

- `pyproject.toml` defines the package with setuptools backend
- Installable packages: `agents`, `evaluation`, `models`, `qb_data`, `qb_env`, `training`
- Legacy root-level files (`config.py`, `dataset.py`, `environment.py`, `model.py`, etc.) coexist with the modular package structure
