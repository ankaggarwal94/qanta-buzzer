# Codebase Structure

**Analysis Date:** 2026-02-24

## Directory Layout

```
qanta-buzzer/
├── main.py                    # CLI entry point, mode routing, phase orchestration
├── config.py                  # Centralized configuration (Config class)
├── model.py                   # T5PolicyModel + PolicyHead architecture
├── environment.py             # QuizBowlEnvironment, BatchedEnvironment, Question
├── dataset.py                 # QuizBowlDataset, QANTADatasetLoader, SyntheticDatasetGenerator
├── train_supervised.py        # SupervisedTrainer, run_supervised_training()
├── train_ppo.py               # PPOTrainer, RolloutBuffer, RolloutStep, run_ppo_training()
├── metrics.py                 # MetricsTracker, evaluate_model(), evaluate_choices_only()
├── demo.py                    # Interactive question answering demo
├── visualize.py               # Visualization utilities for checkpoints
├── test_imports.py            # Module import verification
├── test_csv_loader.py         # Dataset loading verification
├── run.sh                      # Interactive shell script menu
├── README.md                  # Project documentation
├── IMPLEMENTATION_README.md   # Implementation details
├── PROJECT_OVERVIEW.md        # High-level overview
├── CLAUDE.md                  # Development guidance for Claude
└── .planning/
    └── codebase/
        ├── ARCHITECTURE.md    # (This analysis)
        ├── STRUCTURE.md       # (This file)
```

Data and checkpoints (generated at runtime):
```
qanta-buzzer/
├── data/                      # Dataset storage
│   ├── questions.csv          # Input: QANTA quiz bowl data (14.9MB)
│   ├── processed_dataset.json # Parsed dataset with distractors
│   ├── train_dataset.json     # 70% of questions
│   ├── val_dataset.json       # 15% of questions
│   └── test_dataset.json      # 15% of questions
├── checkpoints/               # Model checkpoints
│   ├── supervised/
│   │   ├── best_model/        # Best supervised model (T5 + policy_head.pt)
│   │   ├── epoch_1/           # Intermediate checkpoints
│   │   └── history.json       # Training history
│   └── ppo/
│       ├── best_model/        # Best PPO model
│       ├── iter_50/           # Intermediate checkpoints
│       └── history.json       # PPO training history
└── results/                   # Evaluation outputs
    └── evaluation_results.json
```

## Directory Purposes

**Root directory (qanta-buzzer/):**
- Purpose: Python source code for training pipeline
- Contains: Model definition, training loops, CLI orchestration
- Key files: `main.py` (entry point), `config.py` (hyperparameters), `model.py` (neural net)

**data/ directory:**
- Purpose: Dataset storage and preprocessing
- Contains: Raw QANTA CSV, processed question objects (JSON), train/val/test splits
- Generated at runtime if not present; loads from `questions.csv` if available
- Falls back to synthetic data generation if CSV missing

**checkpoints/ directory:**
- Purpose: Model weights and training state persistence
- Contains: Supervised and PPO model directories with T5 weights, policy head weights, optimizer states
- Structure: Two phases (supervised/, ppo/) each with best_model/ and periodic snapshots
- Each checkpoint includes: pytorch_model.bin (T5 weights), config.json (T5 config), sentencepiece.model (tokenizer), policy_head.pt, training_state.pt

**results/ directory:**
- Purpose: Final evaluation outputs
- Contains: JSON files with metrics (accuracy, ECE, rewards, per-category breakdown)

## Key File Locations

**Entry Points:**
- `main.py`: Primary CLI entry point - routes to supervised/PPO/eval modes
- `run.sh`: Interactive shell script menu (wrapper around main.py)
- `demo.py`: Interactive demo for manual testing

**Configuration:**
- `config.py`: Single Config class with all hyperparameters (model, learning rates, batch sizes, paths, device selection)

**Core Logic:**
- `model.py`: T5PolicyModel class (encoder + policy heads), PolicyHead neural network
- `environment.py`: QuizBowlEnvironment (POMDP simulation), Question dataclass
- `dataset.py`: QuizBowlDataset (data wrapper), QANTADatasetLoader (CSV parsing), SyntheticDatasetGenerator

**Training:**
- `train_supervised.py`: SupervisedTrainer class, run_supervised_training() function
- `train_ppo.py`: PPOTrainer class, RolloutBuffer, RolloutStep, run_ppo_training() function

**Evaluation & Metrics:**
- `metrics.py`: MetricsTracker class, evaluate_model(), evaluate_choices_only() functions

**Testing & Utilities:**
- `test_imports.py`: Verifies all modules can be imported
- `test_csv_loader.py`: Verifies dataset loading from CSV
- `visualize.py`: Checkpoint visualization utilities

## Naming Conventions

**Files:**
- Training phases use underscore separators: `train_supervised.py`, `train_ppo.py`
- Utility/test files use underscore separators: `test_imports.py`, `test_csv_loader.py`
- Config file lowercase: `config.py`
- Single-word modules lowercase: `model.py`, `environment.py`, `dataset.py`, `metrics.py`

**Classes:**
- PascalCase: `T5PolicyModel`, `PolicyHead`, `QuizBowlEnvironment`, `BatchedEnvironment`, `Question`, `SupervisedTrainer`, `PPOTrainer`, `RolloutBuffer`, `RolloutStep`, `MetricsTracker`, `QANTADatasetLoader`, `SyntheticDatasetGenerator`, `QuizBowlDataset`, `Config`
- Exceptions: ValueError (built-in), always raised with descriptive messages

**Functions:**
- snake_case: `run_supervised_training()`, `run_ppo_training()`, `setup_datasets()`, `create_train_val_test_splits()`, `evaluate_model()`, `evaluate_choices_only()`, `compute_system_score()`, `parse_args()`, `setup_config()`, `get_text_representation()`, `get_choices_only_text()`, `get_encoder_output()`, `select_action()`, `get_action_log_probs()`, `predict_answer()`
- Internal/private functions use leading underscore: `_get_observation()`, `_print_model_info()`, `_question_to_dict()`, `_dict_to_question()`, `convert_to_json_serializable()` (helper function, not private)

**Variables:**
- snake_case: `model`, `train_dataset`, `val_dataset`, `test_dataset`, `batch_size`, `learning_rate`, `epoch`, `iteration`, `loss`, `reward`, `best_val_acc`, `best_val_reward`
- RL notation: `gamma` (discount), `gae_lambda`, `action`, `observation`, `reward`, `done`, `value`, `log_prob`, `advantage`, `return_`, `entropy`
- Abbreviations: `ppo` (PPO trainer), `env` (environment), `obs` (observation), `pred` (prediction), `acc` (accuracy), `ece` (expected calibration error)

**Paths (in code):**
- Relative to config.DATA_DIR: "questions.csv" → `data/questions.csv`
- Checkpoint subdirs: config.CHECKPOINT_DIR / "supervised" / "best_model"
- Results: config.RESULTS_DIR / "evaluation_results.json"

## Where to Add New Code

**New Feature (e.g., different reward shaping):**
- Primary code: `environment.py` → Modify `QuizBowlEnvironment.step()` reward computation
- Config: `config.py` → Add hyperparameter (e.g., `REWARD_PENALTY_SCHEME`)
- Tests: `test_csv_loader.py` or new test file for validation

**New Model Component (e.g., different encoder):**
- Implementation: `model.py` → Create new class inheriting or replacing T5PolicyModel
- Config: `config.py` → Add MODEL_NAME or similar override
- Entry point: `main.py` → Update initialization to use new model class
- Tests: Verify tokenization and forward pass in test file

**New Training Algorithm (e.g., A2C instead of PPO):**
- Implementation: Create new file `train_a2c.py` mirroring `train_ppo.py` structure
- Trainer class: `A2CTrainer` with collect_rollouts(), update_policy() methods
- Entry point: `main.py` → Add new mode (e.g., `--mode a2c`)
- Orchestration: New function `run_a2c_training()` called from main.py

**New Utility/Metric:**
- Metrics: Add method to `MetricsTracker` class in `metrics.py`
- Helper: Create new file `utils.py` if utility is general purpose
- Tests: Add to `test_csv_loader.py` or create dedicated test file

**New Dataset Source:**
- Loader: Add new class in `dataset.py` (e.g., `EBQADatasetLoader`)
- Configuration: `config.py` → Add path constants
- Integration: Modify `setup_datasets()` to check for new source and load appropriately

## Special Directories

**data/ directory:**
- Purpose: Input data and processed datasets
- Generated: Yes (processed_dataset.json, splits created at runtime if not present)
- Committed: No (data is generated, .gitignore excludes *.json in data/)
- Notes: questions.csv should be placed here for CSV-based loading

**checkpoints/ directory:**
- Purpose: Model weights, optimizer states, training history
- Generated: Yes (created during training)
- Committed: No (large files, excluded via .gitignore)
- Contents: Each checkpoint is a directory with:
  - `pytorch_model.bin` - T5 encoder weights
  - `config.json` - T5 model config
  - `sentencepiece.model` - T5 tokenizer
  - `policy_head.pt` - PolicyHead weights (custom)
  - `training_state.pt` - Optimizer state dict and training metadata
  - `history.json` - Training curves

**results/ directory:**
- Purpose: Final evaluation metrics and predictions
- Generated: Yes (created during eval mode)
- Committed: No (excluded via .gitignore)
- Contains: JSON files with accuracy, ECE, per-category breakdown, etc.

**logs/ directory:**
- Purpose: Optional detailed logging (not currently used in codebase)
- Referenced in: `config.py` as LOG_DIR = "logs"
- Usage: Can be expanded for TensorBoard logs or detailed metric logging

**.planning/codebase/ directory:**
- Purpose: GSD (Code Mapper) analysis documents
- Generated: By codebase mapper tool
- Committed: Yes (reference documentation for future Claude instances)
- Contains: ARCHITECTURE.md, STRUCTURE.md, CONVENTIONS.md, TESTING.md, etc.

---

*Structure analysis: 2026-02-24*
