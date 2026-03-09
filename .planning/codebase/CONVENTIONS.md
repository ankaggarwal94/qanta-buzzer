# Coding Conventions

**Analysis Date:** 2026-03-08

## Naming Patterns

**Files:**
- Lowercase with underscores: `config.py`, `train_supervised.py`, `train_ppo.py`, `metrics.py`
- Main entry point: `main.py`
- Test files: `test_*.py` (e.g., `test_imports.py`, `test_csv_loader.py`)

**Functions:**
- Snake case for all function names: `prepare_batch()`, `train_epoch()`, `compute_accuracy()`, `get_text_representation()`
- Public methods without leading underscore: `forward()`, `save()`, `load()`
- Private methods with leading underscore: `_print_model_info()`, `_get_observation()`
- Descriptive names for class methods: `load_pretrained()`, `create_train_val_test_splits()`

**Variables:**
- Snake case for local/instance variables: `total_loss`, `batch_size`, `current_epoch`, `best_val_acc`
- Uppercase for class/module constants: `WAIT_ACTION`, `DEVICE`, `MODEL_NAME`, `SUPERVISED_BATCH_SIZE`
- Abbreviated but clear RL terms: `gamma`, `gae_lambda`, `clip_ratio`, `entropy_coef`, `log_probs`, `obs`, `env`

**Classes:**
- PascalCase: `Config`, `PolicyHead`, `T5PolicyModel`, `QuizBowlEnvironment`, `MetricsTracker`, `SupervisedTrainer`
- Dataclasses: `Question`, `RolloutStep` (with `@dataclass` decorator)

## Code Style

**Formatting:**
- No explicit linter configured; follows PEP 8 conventions
- Line length generally 100+ characters (observed in practice)
- Consistent spacing: 4-space indentation

**Imports:**
- Standard library imports first: `import torch`, `import numpy as np`, `from pathlib import Path`
- Third-party library imports: `from transformers import T5ForConditionalGeneration`
- Local imports: `from config import Config`, `from model import T5PolicyModel`
- Imports grouped by category with blank lines between groups

**Type Hints:**
- Used consistently throughout: `def forward(self, text_inputs: List[str], return_value: bool = True)`
- Return types specified: `-> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]`
- Generic types from `typing` module: `List`, `Dict`, `Tuple`, `Optional`

## Error Handling

**Patterns:**
- Validation errors raised explicitly: `raise ValueError(f"Invalid action: {action}. Must be 0-{self.num_actions-1}")`
- File/path operations use `pathlib.Path` and check existence: `if potential_path.exists()`
- Device fallback strategy in config: `DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"`

**Check locations:**
- `config.py` line 62: Device detection with fallback chain
- `environment.py` lines 77-78, 103-104: Input validation with error messages
- `dataset.py` lines 153-155: Skip handling for edge cases (questions without clues)

## Logging

**Framework:** Print statements and `tqdm` progress bars (no formal logging library)

**Patterns:**
- Status messages to stdout: `print("Loading T5 model: {config.MODEL_NAME}")`
- Progress tracking with `tqdm`: `progress_bar = tqdm(range(num_batches), desc=f"Epoch {self.current_epoch + 1}")`
- Postfix metrics in progress bar: `progress_bar.set_postfix({'loss': f'{avg_loss:.4f}', 'acc': f'{avg_acc:.4f}'})`
- Separator lines for section breaks: `print("=" * 60)`

**Key locations:**
- `model.py` line 95-124: Model initialization and info logging
- `train_supervised.py` lines 119-155: Progress bar with metrics
- `train_supervised.py` lines 194-214: Epoch results and checkpoint logging

## Comments

**When to Comment:**
- Docstrings on all classes and public methods (see below)
- Inline comments for non-obvious logic: `# Terminal state has value 0` in `train_ppo.py`
- Algorithm steps: `# GAE computation`, `# Decompose combined actions into wait and answer actions`
- TODOs only for incomplete sections (none currently observed in codebase)

**Docstring Style:** NumPy-style with sections

```python
def step(self, action: int) -> Tuple[Dict, float, bool, Dict]:
    """
    Take an action in the environment.

    Args:
        action: 0 for WAIT, 1-4 for SELECT answer choice

    Returns:
        observation: Current observation
        reward: Reward for this step
        done: Whether episode is finished
        info: Additional information
    """
```

Examples:
- `config.py` lines 10-14: Class docstring
- `model.py` lines 23-28: Method with Args/Returns sections
- `dataset.py` lines 89-110: Function with detailed Args/Returns
- `environment.py` lines 33-41: Constructor with Args description

**TSDoc/JSDoc:** Not applicable (Python project)

## Function Design

**Size Guidelines:**
- Short focused methods: `__len__()` (1 line), `shuffle()` (1 line)
- Medium methods: `forward()` (10-15 lines), `train_epoch()` (50 lines)
- Large methods: `load_from_csv()` (140+ lines) only for complex initialization

**Parameters:**
- Pass configuration via `Config` class (centralized hyperparameters)
- Dataset as object: `dataset: QuizBowlDataset`
- Model as object: `model: T5PolicyModel`
- Deterministic flag for train/eval: `deterministic: bool = False`
- Optional parameters with sensible defaults: `max_length: int = None` defaults to config value

**Return Values:**
- Tuples for multiple outputs: `Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]`
- Dictionaries for composite results: `Dict[str, float]` for metrics, `Dict` for info/metadata
- Objects for complex returns: Return `QuizBowlDataset` instead of `List[Question]`

**Example (concise):**
```python
def predict_answer(self,
                  input_ids: torch.Tensor,
                  attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Predict answer choice (for supervised training)."""
    pooled_output = self.get_encoder_output(input_ids, attention_mask)
    _, answer_logits, _ = self.policy_head(pooled_output)
    predictions = torch.argmax(answer_logits, dim=-1)
    return answer_logits, predictions
```

**Example (complex with internal steps):**
```python
def select_action(self, input_ids, attention_mask, deterministic=False, temperature=1.0):
    """Select actions based on current policy. (57 lines)"""
    with torch.no_grad():
        # Get encoder output
        pooled_output = self.get_encoder_output(input_ids, attention_mask)
        # Get logits from policy head
        wait_logits, answer_logits, values = self.policy_head(pooled_output)
        # Apply temperature
        # ... (full implementation)
        return combined_actions, info
```

## Module Design

**Exports:**
- Classes exported implicitly: `from model import T5PolicyModel`
- Functions exported implicitly: `from dataset import setup_datasets`
- No `__all__` variable observed; all public names available

**Barrel Files:**
- Not used; imports reference specific modules directly
- Example: `from config import Config` rather than `from qanta_buzzer import Config`

## PyTorch Conventions

**Model Structure:**
- Inherit from `nn.Module`: `class PolicyHead(nn.Module):`
- Implement `forward()` method
- All layers defined in `__init__()`: `self.wait_head = nn.Sequential(...)`
- No layers created dynamically in forward pass

**Tensor Operations:**
- Type hints on tensors: `input_ids: torch.Tensor`
- Shape documentation in comments: `[batch_size, seq_len, hidden_size]`
- Device handling: `to(self.device)` called on model and tensors

**Example:**
```python
class PolicyHead(nn.Module):
    def __init__(self, hidden_size: int = 1024, num_choices: int = 4):
        super().__init__()
        self.wait_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 2)
        )

    def forward(self, encoder_hidden_state: torch.Tensor) -> Tuple[...]:
        wait_logits = self.wait_head(encoder_hidden_state)
        return wait_logits, answer_logits, value
```

## Configuration Management

**All hyperparameters centralized in `config.py` as class attributes:**
- Model settings: `MODEL_NAME`, `MAX_INPUT_LENGTH`, `POLICY_HIDDEN_DIM`
- Training hyperparameters: `SUPERVISED_LR`, `PPO_CLIP_RATIO`, `PPO_GAMMA`
- Dataset settings: `NUM_QUESTIONS`, `TRAIN_SPLIT`, `CATEGORY_DISTRIBUTION`

**CLI Override Pattern (in `main.py`):**
```python
def setup_config(args):
    config = Config()
    if args.supervised_epochs is not None:
        config.SUPERVISED_EPOCHS = args.supervised_epochs
    return config
```

**Access Pattern:**
- Trainers receive config object: `def __init__(self, config: Config)`
- Access via `self.config.DEVICE`, `self.config.SUPERVISED_BATCH_SIZE`
- Config printed for debugging: `config.print_config()`

---

*Convention analysis: 2026-03-08*
