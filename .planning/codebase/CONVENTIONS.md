# Coding Conventions

**Analysis Date:** 2026-02-23

## Naming Patterns

**Files:**
- Lowercase with underscores: `model.py`, `train_supervised.py`, `environment.py`
- Test files: `test_*.py` (e.g., `test_imports.py`, `test_csv_loader.py`)
- Main entry point: `main.py`
- Configuration: `config.py`

**Classes:**
- PascalCase: `T5PolicyModel`, `PolicyHead`, `QuizBowlEnvironment`, `QuizBowlDataset`, `MetricsTracker`
- Descriptive names reflecting purpose: `SupervisedTrainer`, `PPOTrainer`, `RolloutBuffer`, `BatchedEnvironment`

**Functions:**
- snake_case: `encode_input()`, `get_encoder_output()`, `prepare_batch()`, `collect_rollouts()`, `compute_returns_and_advantages()`
- Private methods prefixed with underscore: `_print_model_info()`, `_get_observation()`, `_question_to_dict()`, `_dict_to_question()`

**Variables:**
- snake_case for local variables: `wait_logits`, `answer_logits`, `pooled_output`, `text_inputs`, `input_ids`
- UPPERCASE for class constants: `WAIT_ACTION = 0`, `NUM_ANSWER_CHOICES`, `MAX_INPUT_LENGTH`
- Torch tensors and arrays use descriptive names with shape hints in comments: `[batch_size, hidden_size]`, `[batch_size, seq_len]`

**Types:**
- PascalCase for custom types: `Question`, `RolloutStep`, `Config`
- Standard imports: `List`, `Dict`, `Tuple`, `Optional` from `typing`

## Code Style

**Formatting:**
- No explicit formatting tool configured (eslint/prettier not present)
- Default Python style followed with 4-space indentation
- Line continuations use parentheses and alignment

**Example from model.py:**
```python
def select_action(self,
                 input_ids: torch.Tensor,
                 attention_mask: torch.Tensor,
                 deterministic: bool = False,
                 temperature: float = 1.0) -> Tuple[torch.Tensor, Dict]:
```

**Linting:**
- No linting configuration found (no .flake8, .eslintrc, or pyproject.toml)
- Code follows PEP 8 conventions implicitly

## Import Organization

**Order:**
1. Standard library imports (`json`, `csv`, `random`, `argparse`, `os`)
2. Third-party library imports (`torch`, `numpy`, `transformers`, `sklearn`)
3. Local module imports (relative imports from project: `from config import Config`)

**Observed Pattern:**
```python
# Standard library
import json
import csv
import random
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path

# Third-party
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Local
from environment import Question
from config import Config
```

**Path Aliases:**
- None detected. Project uses relative imports from project root.

## Error Handling

**Patterns:**
- Explicit validation with `ValueError` for invalid inputs:
  - `environment.py:78`: `raise ValueError("Episode is already done. Call reset().")`
  - `environment.py:104`: `raise ValueError(f"Invalid action: {action}. Must be 0-{self.num_actions-1}")`
- Minimal try-catch blocks observed; errors allowed to propagate
- Path existence checks before loading/saving: `Path.exists()` used in main.py

## Logging

**Framework:** `print()` statements (no structured logging framework)

**Patterns:**
- Config summary: `config.print_config()` prints all settings with borders
- Training progress: Via tqdm progress bars (`from tqdm import tqdm`)
- Checkpoint messages: `print(f"Model saved to {save_dir}")`, `print(f"Model loaded from {load_dir}")`
- Section headers with ASCII borders:
```python
print("\n" + "=" * 60)
print("Running supervised training only")
print("=" * 60)
```

## Comments

**When to Comment:**
- Docstrings on all public methods and classes
- Inline comments for non-obvious logic (shape hints, mathematical notation)
- No comments for simple, self-documenting code

**JSDoc/TSDoc:**
- Python docstrings follow NumPy-style format with `Args:`, `Returns:`, and typed parameter descriptions
- Example from `model.py:59-68`:
```python
def forward(self, encoder_hidden_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Forward pass through policy head.

    Args:
        encoder_hidden_state: [batch_size, hidden_size] - pooled encoder output

    Returns:
        wait_logits: [batch_size, 2] - logits for wait/answer
        answer_logits: [batch_size, num_choices] - logits for answer selection
        value: [batch_size, 1] - value estimate
    """
```

## Function Design

**Size:** Functions generally 10-50 lines; complex operations broken into steps with meaningful names.

**Parameters:**
- Use type hints for all parameters: `text_inputs: List[str]`, `hidden_size: int = 1024`
- Default values provided for optional parameters
- No `*args` or `**kwargs` observed; explicit parameter lists

**Return Values:**
- Type hints always provided: `-> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]`
- Returns dictionaries with descriptive keys for multiple related values
- Example from `model.py:305-316`:
```python
info = {
    'wait_logits': wait_logits,
    'answer_logits': answer_logits,
    'wait_probs': wait_probs,
    'answer_probs': answer_probs,
    'wait_actions': wait_actions,
    'answer_actions': answer_actions,
    'values': values,
    'log_probs': log_probs
}
```

## Module Design

**Exports:**
- Modules export classes and functions used by other modules
- Example: `model.py` exports `T5PolicyModel` and `PolicyHead`
- Example: `dataset.py` exports `QuizBowlDataset` and loader functions

**Barrel Files:**
- No barrel files (index.py) used; imports are direct from modules

## Dataclasses

**Usage:** Dataclasses used for simple data containers:
- `Question` in `environment.py:10-18`: Holds question metadata
- `RolloutStep` in `train_ppo.py:22-34`: Holds single trajectory step
- Rich type hints on all fields

**Pattern from environment.py:**
```python
@dataclass
class Question:
    """Represents a quiz bowl question with pyramidal clues"""
    question_id: str
    clues: List[str]
    answer_choices: List[str]
    correct_answer_idx: int
    category: str
    metadata: Optional[Dict] = None
```

## Configuration Management

**Pattern:** Centralized `Config` class in `config.py` with all uppercase class attributes.

**Usage:**
- Imported into modules: `from config import Config`
- Instantiated once in main: `config = Config()`
- Overridden via command-line arguments in `main.py:49-72`
- Passed to trainer and model classes: `SupervisedTrainer(model, train_dataset, val_dataset, config)`

## Type Hints

**Coverage:** Nearly 100% on function signatures and class attributes.

**Style:**
- Always used for parameters and return types
- Optional types explicit: `Optional[Dict] = None`
- Numpy arrays typically: `np.ndarray`
- Torch tensors: `torch.Tensor`
- Collections explicit: `List[str]`, `Dict[str, float]`, `Tuple[...]`

---

*Convention analysis: 2026-02-23*
