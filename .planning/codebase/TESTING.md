# Testing Patterns

**Analysis Date:** 2026-02-23

## Test Framework

**Runner:**
- No formal test framework configured (pytest/unittest not in use)
- Test files are standalone Python scripts executed directly
- No testing library in requirements.txt

**Assertion Library:**
- Print-based validation (not a formal assertion library)

**Run Commands:**
```bash
python test_imports.py              # Check module imports
python test_csv_loader.py           # Test dataset loading from CSV
```

## Test File Organization

**Location:**
- Co-located with source code in project root
- Files: `test_imports.py`, `test_csv_loader.py`

**Naming:**
- `test_*.py` convention for all test files

**Structure:**
```
qanta-buzzer/
├── test_imports.py        # Import verification
├── test_csv_loader.py     # Dataset loading tests
├── model.py               # Core source files
├── environment.py
└── dataset.py
```

## Test Structure

**Suite Organization:**
Tests are organized as standalone scripts rather than test classes. Each script has a specific purpose.

**test_imports.py Pattern:**
```python
"""Quick import test"""

from config import Config
from environment import Question, QuizBowlEnvironment
from dataset import QuizBowlDataset, SyntheticDatasetGenerator
from model import T5PolicyModel, PolicyHead

print('✓ All core modules imported successfully!')
print('✓ Config:', Config.MODEL_NAME)
print('✓ Question class available')
print('✓ QuizBowlEnvironment class available')
print('✓ QuizBowlDataset class available')
print('✓ T5PolicyModel class available')
```

**test_csv_loader.py Pattern:**
```python
"""Test loading questions from CSV"""

from config import Config
from dataset import setup_datasets

# Create a config with fewer questions for testing
config = Config()
config.NUM_QUESTIONS = 100  # Load only 100 questions for testing

print("=" * 60)
print("Testing QANTA CSV Dataset Loader")
print("=" * 60)

# Load datasets
train_dataset, val_dataset, test_dataset = setup_datasets(config)

# Validation: Show sample questions
for i in range(min(3, len(train_dataset))):
    question = train_dataset.questions[i]
    print(f"\n--- Question {i+1} ---")
    # Print and visually verify structure
```

**Patterns:**
- Setup phase: Create config and modify settings for test conditions
- Execution phase: Call functions/load datasets
- Validation phase: Print output for manual inspection
- No automated assertions; success is indicated by successful execution

## Mocking

**Framework:** No mocking framework used

**Patterns:**
- Configuration objects modified directly for testing: `config.NUM_QUESTIONS = 100`
- No stub/mock objects created
- Real datasets loaded and processed

**What to Mock:**
- Not applicable; no mocking infrastructure present

**What NOT to Mock:**
- All components run with real implementations

## Fixtures and Factories

**Test Data:**
- Synthetic data generated via `SyntheticDatasetGenerator` (referenced in test_imports.py)
- CSV data loaded from `questions.csv` in dataset loading tests
- Config values act as test parameters: `config.NUM_QUESTIONS`, `config.MIN_CLUES_PER_QUESTION`

**Location:**
- CSV source: `questions.csv` (14MB, in project root)
- Processed datasets: `data/` directory contains JSON versions:
  - `data/train_dataset.json`
  - `data/val_dataset.json`
  - `data/test_dataset.json`
  - `data/processed_dataset.json`

## Coverage

**Requirements:** None enforced

**View Coverage:**
- No coverage measurement tooling present
- Manual verification via print statements and visual inspection

## Test Types

**Unit Tests:**
- Minimal unit testing; focus is on integration testing
- Import verification in `test_imports.py`: Verifies all modules load correctly
- No isolated function tests

**Integration Tests:**
- `test_csv_loader.py`: Tests full pipeline of loading CSV, creating questions, splitting datasets
- Validates data structure and correctness of question objects

**E2E Tests:**
- Not formal E2E tests, but training scripts (`train_supervised.py`, `train_ppo.py`) serve as integration tests
- `main.py --mode eval`: Evaluation mode tests complete training and inference pipeline
- Full pipeline tested via `main.py --mode full`: Supervised training + PPO training on actual data

## Evaluation Infrastructure

**MetricsTracker (metrics.py):**
Central evaluation class used across training and inference:

```python
class MetricsTracker:
    """Track and compute various metrics for QA evaluation"""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all tracked values"""
        self.predictions = []
        self.targets = []
        self.confidences = []
        self.rewards = []
        self.buzz_positions = []
        self.is_correct = []
        self.categories = []

    def update(self, pred: int, target: int, confidence: float,
               reward: float = None, buzz_position: int = None, category: str = None):
        """Update metrics with new sample"""

    def compute_accuracy(self) -> float:
        """Compute overall accuracy"""

    def compute_average_reward(self) -> float:
        """Compute average reward"""

    def compute_ece(self, num_bins: int = 10) -> float:
        """Compute Expected Calibration Error"""
```

**Usage in Training:**
```python
# In train_supervised.py:
metrics = MetricsTracker()
for batch in data_loader:
    predictions = model(batch)
    metrics.update(pred=pred_idx, target=label, confidence=prob)
accuracy = metrics.compute_accuracy()
```

**Evaluation Functions:**
- `evaluate_model(model, dataset, device)` in `metrics.py`: Full evaluation on dataset
- `evaluate_choices_only(model, dataset, device)`: Baseline - only answer choices (no clues)
- Called from `main.py --mode eval`: Final evaluation

## Common Patterns

**Async Testing:**
- Not applicable; no async code in codebase

**Error Testing:**
- Explicit error validation in environment:
  - `environment.py:78`: "Episode is already done" error
  - `environment.py:104`: "Invalid action" error
- Tests in `test_csv_loader.py` validate Question object structure and print for visual inspection

**Dataset Loading Pattern:**
```python
# From dataset.py (loader function pattern)
train_dataset, val_dataset, test_dataset = setup_datasets(config)

# Test validates:
# 1. Datasets load successfully
# 2. Split ratios correct (0.7/0.15/0.15 for train/val/test)
# 3. Question objects have correct fields
# 4. Answer choices and metadata are properly structured
```

## Manual Validation Checklist

When running tests, verify:
1. All imports succeed without errors
2. Config values print correctly
3. Questions load from CSV
4. Question objects contain:
   - Valid question_id
   - Non-empty clues list (min 4, max 6 per config)
   - 4 answer choices
   - Valid correct_answer_idx (0-3)
   - Category from configured distribution
5. Train/val/test split respects configured ratios
6. No corrupted JSON in saved datasets

---

*Testing analysis: 2026-02-23*
