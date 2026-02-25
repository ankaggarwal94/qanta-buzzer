# Testing Patterns

**Analysis Date:** 2025-02-24

## Test Framework

**Runner:** No formal test framework (pytest/unittest not configured)

**Test Files:**
- `test_imports.py`: Import verification script
- `test_csv_loader.py`: Dataset loading script

**Run Commands:**
```bash
python test_imports.py              # Verify all modules import successfully
python test_csv_loader.py           # Verify dataset loading and inspect samples
```

**No Coverage Tools:** No pytest, coverage, or similar tools configured. Tests are ad-hoc verification scripts.

## Test File Organization

**Location:** Tests in project root next to source modules

```
qanta-buzzer/
├── config.py
├── model.py
├── environment.py
├── dataset.py
├── metrics.py
├── test_imports.py          # Test file 1
├── test_csv_loader.py       # Test file 2
└── .planning/
```

**Naming:**
- Test files prefixed with `test_`: `test_imports.py`, `test_csv_loader.py`
- No test classes or functions; scripts run procedurally

## Test Structure

**Import Verification (`test_imports.py`):**

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

**Patterns:**
- Import all critical classes
- Print status messages on success
- No assertions; passes if no exceptions raised

**Dataset Loading Verification (`test_csv_loader.py`):**

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

print("\n" + "=" * 60)
print("Sample Questions from Training Set")
print("=" * 60)

# Show a few sample questions
for i in range(min(3, len(train_dataset))):
    question = train_dataset.questions[i]
    print(f"\n--- Question {i+1} ---")
    print(f"ID: {question.question_id}")
    print(f"Category: {question.category}")
    # ... detailed output
```

**Patterns:**
- Modify config for test run (smaller dataset)
- Call main setup function
- Inspect results manually with print statements
- Visual verification only; no automated checks

## Data Fixtures

**Synthetic Data Generator:** Built-in for testing without CSV

Located in `dataset.py` (lines 234-407):

```python
class SyntheticDatasetGenerator:
    """Generate synthetic quiz bowl questions for development and testing."""

    SAMPLE_QUESTIONS = {
        'history': [
            {
                'entity': 'Napoleon Bonaparte',
                'clues': [
                    'This military leader established the Continental System...',
                    'He crowned himself emperor in 1804...',
                    # ... 3-5 clues per question
                ],
                'distractors': ['Julius Caesar', 'Alexander the Great', 'Charlemagne']
            },
            # ... more historical figures
        ],
        'literature': [...],
        'science': [...],
        'arts': [...]
    }

    @classmethod
    def generate_dataset(cls,
                        num_questions: int = 500,
                        category_distribution: Dict[str, float] = None,
                        seed: int = 42) -> QuizBowlDataset:
        """Generate synthetic dataset."""
        # ... implementation
```

**Usage in Tests:**
- `setup_datasets(config)` falls back to synthetic generator if no CSV found
- Deterministic: seeded with `seed=42` by default
- Configurable: `num_questions`, `category_distribution`, `min_clues`, `max_clues`

**Fixture locations:**
- Synthetic templates: `dataset.py` lines 240-326 (SAMPLE_QUESTIONS dict)
- CSV parser: `dataset.py` lines 89-231 (QANTADatasetLoader.load_from_csv)

## Evaluation Infrastructure

**Not traditional "tests" but evaluation utilities in `metrics.py`:**

### MetricsTracker Class

```python
class MetricsTracker:
    """Track and compute various metrics for QA evaluation"""

    def reset(self):
        """Reset all tracked values"""
        self.predictions = []
        self.targets = []
        self.confidences = []
        self.rewards = []
        # ...

    def update(self, pred: int, target: int, confidence: float,
               reward: float = None, buzz_position: int = None, category: str = None):
        """Update metrics with new sample"""
        self.predictions.append(pred)
        # ...

    def compute_accuracy(self) -> float:
        """Compute overall accuracy"""
        if len(self.predictions) == 0:
            return 0.0
        return accuracy_score(self.targets, self.predictions)

    def compute_ece(self, num_bins: int = 10) -> float:
        """Compute Expected Calibration Error"""
        # ... implementation

    def get_summary(self) -> Dict:
        """Get summary of all metrics"""
        summary = {
            'num_samples': len(self.predictions),
            'accuracy': self.compute_accuracy(),
            # ... additional metrics
        }
        return convert_to_json_serializable(summary)
```

**Patterns:**
- Accumulate predictions and ground truth
- Compute standard metrics: accuracy, ECE, Brier score
- Support hierarchical analysis: per-category accuracy, per-position accuracy
- JSON serialization helper: `convert_to_json_serializable()` (handles numpy types)

**Locations:**
- `metrics.py` lines 38-280: MetricsTracker implementation
- `metrics.py` lines 313-384: evaluate_model() function

### Evaluation Functions

**`evaluate_model()` - Full question evaluation**

```python
def evaluate_model(model, dataset, device: str = 'cpu',
                  max_samples: int = None,
                  deterministic: bool = True) -> MetricsTracker:
    """Evaluate model on a dataset using the RL environment."""
    model.eval()
    metrics = MetricsTracker()

    questions = dataset.questions[:max_samples] if max_samples else dataset.questions

    with torch.no_grad():
        for question in questions:
            env = QuizBowlEnvironment(question)
            obs = env.reset()
            done = False

            while not done:
                text = env.get_text_representation(obs)
                inputs = model.tokenizer(text, return_tensors='pt', ...).to(device)
                actions, info = model.select_action(inputs['input_ids'],
                                                   inputs['attention_mask'],
                                                   deterministic=deterministic)
                obs, reward, done, step_info = env.step(action)

            # Extract final metrics
            if 'is_correct' in step_info:
                metrics.update(pred=step_info['answer_idx'],
                             target=step_info['correct_idx'],
                             confidence=confidence,
                             reward=reward,
                             buzz_position=step_info['clue_position'],
                             category=question.category)

    return metrics
```

**`evaluate_choices_only()` - Control experiment**

```python
def evaluate_choices_only(model, dataset, device: str = 'cpu',
                         max_samples: int = None) -> MetricsTracker:
    """Evaluate model on answer choices only (control experiment)."""
    model.eval()
    metrics = MetricsTracker()

    # Get choices-only text (no clues)
    text = env.get_choices_only_text()

    # Verify model uses clues (expected baseline: ~25% random)
```

**Locations:**
- `metrics.py` lines 313-384: evaluate_model()
- `metrics.py` lines 387-443: evaluate_choices_only()
- Calls in `train_supervised.py` lines 166-172: Used in validation loop

## Test Patterns Observed

**1. Import Safety:**
- `test_imports.py` catches import errors early
- All critical classes verified accessible

**2. Data Pipeline Verification:**
- `test_csv_loader.py` confirms dataset loading end-to-end
- Checks CSV parsing, distractor generation, train/val/test splits
- Inspects sample questions for manual review

**3. Metric Computation:**
- Standalone MetricsTracker tested implicitly during training
- `evaluate_model()` used in `train_supervised.py` validation loop
- Results printed and saved to JSON for inspection

**4. Environment Simulation:**
- `QuizBowlEnvironment` tested indirectly through `evaluate_model()`
- Step-by-step episode execution verified by checking rewards, done flags
- No unit tests; verification is integration-level

## What is NOT Tested

**No explicit testing for:**
- Token counting/overflow scenarios (`tokenizer` behavior assumed correct)
- Gradient flow in backward pass (PyTorch assumed correct)
- Edge cases in rare states (e.g., all clues exhausted, forced answer)
- Memory/performance under heavy load
- Distributed training across devices
- Model save/load checkpoint compatibility
- Adversarial inputs or malformed data

**Risk Areas:**
- `model.save()`/`model.load()` checkpoint format compatibility untested
- T5 model architecture changes in future transformers versions
- CSV format changes or missing columns in questions.csv

## Test Execution in Training

**Implicit Testing During `train_supervised.py`:**
- Each epoch validates on validation set
- `evaluate_model()` called with `deterministic=True`
- Results logged and best model saved
- No assertions; failures are manual inspection

**Example (`train_supervised.py` lines 175-225):**
```python
def train(self):
    """Run full supervised training"""
    for epoch in range(self.config.SUPERVISED_EPOCHS):
        train_loss, train_acc = self.train_epoch()
        val_summary = self.validate()  # Calls evaluate_model()
        val_acc = val_summary['accuracy']

        print(f"Epoch {epoch + 1}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Acc: {val_acc:.4f}")

        # Save best model based on val_acc
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.save_checkpoint(is_best=True)
```

## Testing Philosophy

**Approach:** Empirical rather than unit-tested

- **No pytest/unittest:** Tests are standalone scripts run manually
- **No mocking:** All components tested in realistic settings
- **No fixtures in code:** Synthetic data generator used, but not fixtures
- **Manual verification:** Print output inspected; no automated assertions
- **Integration-focused:** Full pipeline tested end-to-end
- **Relies on training loops:** Model correctness inferred from training behavior (accuracy improving, loss decreasing)

**When tests fail:**
- Import errors: Dependency issue (transformers, torch)
- Dataset loading errors: CSV parsing or path issues
- Training errors: Usually model/device issues, not logic

---

*Testing analysis: 2025-02-24*
