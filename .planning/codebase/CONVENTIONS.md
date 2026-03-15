# Conventions

## Code Style

- **Python version features:** `from __future__ import annotations` used consistently across all modules
- **Type hints:** Full type annotations on function signatures using modern syntax (`list[str]`, `dict[str, Any]`, `str | None`)
- **Docstrings:** NumPy-style with `Parameters`, `Returns`, `Notes`, `Examples` sections
- **Imports:** Standard library → third-party → local, grouped with blank lines
- **Line length:** No enforced limit; lines typically under 100 characters
- **String quotes:** Double quotes for docstrings and strings, single quotes in `__all__` lists

## Naming Patterns

- **RL notation:** `V` (value), `R` (reward), `T` (transition), `gamma` (discount), `s`/`a` (state/action)
- **Traces:** `c_trace` (buzz confidence per step), `g_trace` (correctness per step), `top_p_trace`, `entropy_trace`
- **Config keys:** Match YAML section names exactly (`data.K`, `likelihood.beta`, `environment.reward_mode`)
- **File naming:** Module files named after their primary class in snake_case
- **Private helpers:** Prefixed with underscore (`_text_key`, `_belief_from_scratch`, `_to_dict`)

## Dataclass Patterns

All core data structures are `@dataclass`:

```python
@dataclass
class TossupQuestion:
    qid: str
    question: str
    tokens: list[str]
    answer_primary: str
    # ...

@dataclass
class MCQuestion(TossupQuestion):  # Inheritance
    options: list[str]
    gold_index: int
    # ...

@dataclass
class SoftmaxEpisodeResult:  # Standalone result type
    qid: str
    buzz_step: int
    # ...
```

Each agent type has its own result dataclass (`EpisodeResult`, `SoftmaxEpisodeResult`, `PPOEpisodeTrace`).

## Lazy Imports

Heavy optional dependencies use `__getattr__` for lazy loading in `__init__.py`:

```python
# agents/__init__.py
def __getattr__(name: str):
    if name in ("PPOBuzzer", "PPOEpisodeTrace"):
        from agents.ppo_buzzer import PPOBuzzer, PPOEpisodeTrace
        return {"PPOBuzzer": PPOBuzzer, "PPOEpisodeTrace": PPOEpisodeTrace}[name]
    raise AttributeError(...)
```

Same pattern in `models/__init__.py` for `T5PolicyModel` and `PolicyHead`.

## Error Handling

- **Validation at boundaries:** `ValueError` for invalid config values, missing data fields, out-of-range indices
- **Import guards:** `ImportError` with helpful messages for optional dependencies (OpenAI, HuggingFace datasets)
- **File guards:** `FileNotFoundError` for missing CSV files, config files
- **Runtime guards:** `RuntimeError` for models used before fitting (e.g., TF-IDF before `fit()`)
- **No blanket try/except:** Errors propagate with descriptive messages

## Configuration Pattern

- YAML config loaded once at script entry → passed as `dict[str, Any]` through the call stack
- Factory functions accept config dict: `make_env_from_config(config, ...)`, `build_likelihood_from_config(config, ...)`
- `scripts/_common.py` provides `load_config()` with `--smoke` flag support that auto-selects `configs/smoke.yaml`
- CLI overrides via argparse with `--data.K=5` style nested key overrides → `merge_overrides()`

## Reproducibility

- Seeds set explicitly: `random.seed()`, `np.random.seed()`, `torch.manual_seed()`
- Convention: seeds 13 (default), 42 (shuffle), or 1/2/3 for multi-seed runs
- `shuffle_seed` in config controls data shuffling separately from environment seed

## qb-rl Compatibility Convention

Backward-compatible re-exports use thin shim modules:

```python
# qb_env/data_loader.py
"""qb-rl compatibility re-exports for tossup data loading."""
from qb_data.data_loader import (
    QANTADatasetLoader, TossupQuestion, load_tossup_questions, ...
)
```

This pattern is used in `qb_env/data_loader.py`, `qb_env/mc_builder.py`, `qb_env/text_utils.py`, and `models/answer_profiles.py`.
