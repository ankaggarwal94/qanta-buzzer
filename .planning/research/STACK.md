# Technology Stack

**Project:** Quiz Bowl RL Buzzer (Unified)
**Researched:** 2026-02-24

## Recommended Stack

### Core Framework
| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| Python | 3.11+ | Runtime | Stable ecosystem, type hints mature, uvloop support for async if needed |
| PyTorch | 2.3.0+ | Neural networks | Industry standard for research, better debugging than TF2, MPS support for Mac |
| Gymnasium | 1.1.0+ | RL environment API | Successor to OpenAI Gym, actively maintained, standardized API |
| Stable-Baselines3 | 2.6.0+ | PPO implementation | Production-ready, well-tested, integrates with Gymnasium |

**Rationale:** PyTorch over TensorFlow for research flexibility. Gymnasium over raw custom envs for standardization. SB3 over custom PPO for robustness — your custom PPO in qanta-buzzer has educational value but SB3 is battle-tested with proper vectorized envs, automatic advantage normalization, and extensive hyperparameter tuning. Keep custom PPO as a comparison point.

### Language Models
| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| Transformers | 4.45.0+ | T5 model loading | Hugging Face standard, excellent T5 support, automatic downloads |
| sentence-transformers | 3.3.0+ | SBERT embeddings | Fast semantic similarity, no API costs, good baseline |
| T5-large | - | Semantic encoder | 770M params optimal for your GPU constraints, can downscale to T5-base |

**Rationale:** T5 serves dual purpose — likelihood scorer (via encoder similarity) and optional end-to-end policy. SBERT for lightweight belief features. Avoid OpenAI embeddings (API costs, latency). T5-large fits in 8GB VRAM, downgrade to T5-base (220M) if memory constrained.

### Data Processing
| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| Datasets | 4.0.0+ | HF dataset loading | Standardized loading, caching, streaming support |
| pandas | 2.2.0+ | CSV processing | Mature, fast for tabular ops |
| scikit-learn | 1.5.0+ | TF-IDF, metrics | Vectorized implementations, calibration metrics |
| NumPy | <2.0.0 | Array ops | Pin to 1.x for compatibility with older packages |

**Rationale:** Datasets library for HF integration but CSV as primary (already local). NumPy <2.0 critical — NumPy 2.0 breaks many packages. Pandas for CSV ops over raw Python csv module.

### Configuration & Logging
| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| PyYAML | 6.0.2+ | Config files | Human-readable, standard for ML configs |
| Hydra | - | NOT recommended | Overly complex for this scope |
| loguru | 0.7.0+ | Logging | Better than stdlib logging, structured output |
| tqdm | 4.66.0+ | Progress bars | Essential for long training runs |
| rich | 13.9.0+ | Terminal output | Tables, progress, better UX |

**Rationale:** YAML over Python config classes for external editability. Loguru over logging module for structured logs. Avoid Hydra — overkill for single experiments. Rich for professional output formatting.

### Evaluation & Visualization
| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| matplotlib | 3.9.0+ | Plots | Standard for papers, LaTeX rendering |
| seaborn | 0.13.0+ | Statistical plots | Better defaults than raw matplotlib |
| scipy | 1.13.0+ | Statistics | Bootstrap CIs, statistical tests |

**Rationale:** Matplotlib + Seaborn standard for academic papers. Avoid Plotly/Dash (web frameworks unnecessary).

### Infrastructure
| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| pip + pyproject.toml | - | Package management | Standard, editable installs |
| uv | - | NOT for this project | Great tool but adds complexity |
| pytest | 8.3.0+ | Testing | Better than unittest, fixtures, parametrization |
| pre-commit | 3.8.0+ | Code quality | Auto-format, lint before commits |

**Rationale:** Standard pip over uv to minimize toolchain complexity given time constraints. Pytest over unittest for better test organization. Pre-commit optional but recommended for code quality.

## Alternatives Considered

| Category | Recommended | Alternative | Why Not |
|----------|-------------|-------------|---------|
| RL Framework | Stable-Baselines3 | Custom PPO | SB3 battle-tested, but keep custom for comparison |
| RL Framework | SB3 | RLlib | Too heavy, distributed training overkill |
| RL Framework | SB3 | CleanRL | Less mature, fewer features |
| LLM Library | Transformers | TensorFlow | PyTorch ecosystem stronger for research |
| Embeddings | SBERT | OpenAI API | Costs, latency, external dependency |
| Embeddings | SBERT | Word2Vec/GloVe | Outdated, poor semantic understanding |
| Config | PyYAML | Hydra | Overly complex for single experiments |
| Config | PyYAML | Python Config class | Less flexible, requires code changes |
| Environment | Gymnasium | Raw Python | Lose standardization, vec env support |

## Installation

```bash
# Core dependencies
pip install torch>=2.3.0 --index-url https://download.pytorch.org/whl/cpu  # or cu118 for CUDA
pip install transformers>=4.45.0
pip install gymnasium>=1.1.0
pip install "stable-baselines3[extra]>=2.6.0"
pip install sentence-transformers>=3.3.0

# Data and utils
pip install "numpy<2.0.0"  # Critical: NumPy 2.0 breaks compatibility
pip install pandas>=2.2.0
pip install scikit-learn>=1.5.0
pip install datasets>=4.0.0
pip install pyyaml>=6.0.2

# Logging and UX
pip install loguru>=0.7.0
pip install tqdm>=4.66.0
pip install rich>=13.9.0

# Evaluation
pip install matplotlib>=3.9.0
pip install seaborn>=0.13.0
pip install scipy>=1.13.0

# Development (optional)
pip install pytest>=8.3.0
pip install pre-commit>=3.8.0
pip install ipython>=8.29.0
```

Or via pyproject.toml:
```toml
[project]
requires-python = ">=3.11"
dependencies = [
    "torch>=2.3.0",
    "transformers>=4.45.0",
    "gymnasium>=1.1.0",
    "stable-baselines3[extra]>=2.6.0",
    "sentence-transformers>=3.3.0",
    "numpy<2.0.0",
    "pandas>=2.2.0",
    "scikit-learn>=1.5.0",
    "datasets>=4.0.0",
    "pyyaml>=6.0.2",
    "loguru>=0.7.0",
    "tqdm>=4.66.0",
    "rich>=13.9.0",
    "matplotlib>=3.9.0",
    "seaborn>=0.13.0",
    "scipy>=1.13.0",
]
```

## Architecture Implications

### Dual Model Support
The stack supports two policy approaches:
1. **Lightweight MLP on belief features** — Use SBERT/TF-IDF to compute beliefs → extract features → SB3 PPO with small MLP
2. **T5 end-to-end** — Text input → T5 encoder → custom policy heads (keep your existing implementation)

### Modular Pipeline
```
Data (CSV/HF) → MC Builder → Gymnasium Env → Agent (SB3 or custom) → Evaluation
                     ↓
              Likelihood Models (TF-IDF, SBERT, T5)
```

### Key Integration Points
- **T5 as LikelihoodModel**: Implement `T5Likelihood` class following qb-rl's ABC pattern
- **Dual environment observations**: Support both text (for T5 policy) and belief features (for MLP)
- **Config-driven model selection**: YAML switches between T5 policy vs MLP policy
- **Unified evaluation**: Same metrics (S_q, ECE, Brier) regardless of policy type

## Version Compatibility Matrix

| Package | Min Version | Max Version | Notes |
|---------|-------------|-------------|-------|
| Python | 3.11 | 3.12 | 3.13 not yet fully supported by all packages |
| PyTorch | 2.3.0 | 2.5.x | 2.6 may have breaking changes |
| NumPy | 1.24.0 | 1.26.x | MUST be <2.0.0 |
| Transformers | 4.45.0 | 4.47.x | Check T5 compatibility |
| Gymnasium | 1.1.0 | 1.1.x | 1.2 may change API |
| SB3 | 2.6.0 | 2.6.x | 3.0 will have breaking changes |

## Memory Requirements

| Configuration | RAM | GPU VRAM | Notes |
|---------------|-----|----------|-------|
| T5-large + SB3 | 16GB | 8GB | Full system |
| T5-base + SB3 | 12GB | 4GB | Reduced model |
| T5-small + SB3 | 8GB | 2GB | Minimal model |
| MLP only (no T5) | 8GB | - | CPU-only feasible |

## Platform Support

| Platform | Status | Notes |
|----------|--------|-------|
| Linux + CUDA | ✅ Optimal | Best performance |
| macOS + MPS | ✅ Supported | Good performance on M1/M2/M3 |
| Windows + CUDA | ✅ Supported | Requires CUDA toolkit |
| CPU only | ✅ Fallback | Slow but functional |

## Sources

- [HIGH confidence] qb-rl pyproject.toml — verified working stack from existing codebase
- [HIGH confidence] Gymnasium 1.1.0 documentation — current stable version with good SB3 integration
- [HIGH confidence] Stable-Baselines3 2.6.0 release notes — latest stable with Gymnasium support
- [MEDIUM confidence] Transformers library versioning — 4.45+ has stable T5 support based on qb-rl usage
- [HIGH confidence] NumPy <2.0 requirement — known compatibility issue across ecosystem

## Critical Decisions

1. **Use SB3 PPO as primary, keep custom PPO for comparison** — Gets you working results fast, custom implementation becomes an interesting comparison point for the paper
2. **SBERT for belief features, T5 for optional end-to-end** — Best of both worlds, can compare in experiments
3. **YAML config over Python Config class** — External editability crucial for experiments
4. **NumPy <2.0.0 is non-negotiable** — NumPy 2.0 breaks too many dependencies
5. **Avoid external APIs (OpenAI)** — Latency, costs, reproducibility issues

---

*Stack recommendation confidence: HIGH — Based on working qb-rl codebase and standard 2024-2025 RL research practices*