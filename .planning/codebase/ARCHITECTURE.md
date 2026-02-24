# Architecture

**Analysis Date:** 2026-02-23

## Pattern Overview

**Overall:** Staged learning pipeline with reinforcement learning optimization

**Key Characteristics:**
- Two-phase training: supervised warm-start → PPO fine-tuning
- POMDP-based environment with partial observability
- T5 encoder + learnable policy head architecture
- Modular layer-based separation (environment, model, training, evaluation)

## Layers

**Environment Layer:**
- Purpose: Defines the quiz bowl POMDP problem with incremental clue revelation
- Location: `environment.py`
- Contains: `Question` dataclass, `QuizBowlEnvironment`, `BatchedEnvironment`
- Depends on: numpy, Python standard library
- Used by: Training scripts, evaluation metrics

**Model Layer:**
- Purpose: T5-based neural network with custom policy head for action selection
- Location: `model.py`
- Contains: `PolicyHead` (wait/answer/value heads), `T5PolicyModel` (encoder + policy)
- Depends on: torch, transformers (T5), config
- Used by: Training loops, inference/evaluation

**Data Layer:**
- Purpose: Dataset management, question loading, and train/val/test splitting
- Location: `dataset.py`
- Contains: `QuizBowlDataset`, `QANTADatasetLoader`, `SyntheticDatasetGenerator`
- Depends on: json, csv, numpy, environment (Question)
- Used by: main.py, training scripts

**Training Layer:**
- Purpose: Implements supervised and PPO training algorithms
- Location: `train_supervised.py`, `train_ppo.py`
- Contains: `SupervisedTrainer`, `PPOTrainer`, `RolloutBuffer`, `RolloutStep`
- Depends on: torch, model, dataset, environment, metrics, config
- Used by: main.py

**Evaluation Layer:**
- Purpose: Computes metrics (accuracy, ECE, reward, calibration, buzzing behavior)
- Location: `metrics.py`
- Contains: `MetricsTracker`, `evaluate_model()`, `evaluate_choices_only()`
- Depends on: numpy, torch, sklearn
- Used by: Training loops, main.py

**Configuration Layer:**
- Purpose: Centralized hyperparameters and settings
- Location: `config.py`
- Contains: `Config` class with model, training, reward, dataset, device settings
- Depends on: torch (for device detection)
- Used by: All other layers via dependency injection

**Orchestration Layer:**
- Purpose: Main entry point coordinating pipeline execution
- Location: `main.py`
- Contains: Argument parsing, mode selection (supervised/ppo/full/eval)
- Depends on: All other layers
- Used by: CLI user

## Data Flow

**Phase 1: Supervised Training**

1. `main.py` parses CLI args and loads config
2. `dataset.py` loads/generates questions from CSV, splits into train/val/test
3. `SupervisedTrainer` in `train_supervised.py`:
   - Iterates over epochs
   - For each epoch, batches questions from train dataset
   - Prepares batch: full question text (all clues) → tokenized input_ids/attention_mask
   - Forward pass through `T5PolicyModel` → answer_logits
   - Compute cross-entropy loss against correct_answer_idx
   - Backprop, optimizer step with gradient accumulation
   - Validate on val dataset, save best checkpoint
4. `metrics.py` computes accuracy on validation set each epoch
5. Best supervised model saved to `checkpoints/supervised/best_model/`

**Phase 2: PPO Training**

1. `PPOTrainer` in `train_ppo.py` loads pretrained supervised model
2. For each PPO iteration (250 total):
   - Collect 32 episodes by rolling out in `BatchedEnvironment`
   - For each step in episode:
     - Get partial question observation (clues so far + choices)
     - Tokenize observation text
     - Forward through model → wait_logits, answer_logits, value
     - Sample action from policy (wait=0 or select_choice=1-4)
     - Step environment, get reward and next observation
     - Store `RolloutStep`: observation_text, action, reward, done, value, log_prob
   - Compute GAE advantages and returns across all episodes
   - Train on batches from rollout buffer (4 epochs per iteration):
     - Recompute log probs with current policy
     - Compute PPO loss: -advantage * log_prob_ratio (with clipping)
     - Add value loss: (return - value_pred)^2
     - Add entropy bonus: -entropy_coef * entropy(policy)
     - Backprop, optimizer step
3. Evaluate on validation set every 50 iterations
4. Save best model to `checkpoints/ppo/best_model/`
5. Final evaluation on test set

**Evaluation Phase**

1. Load trained model from checkpoint
2. Iterate over test dataset questions
3. For each question:
   - Initialize `QuizBowlEnvironment`
   - Loop until done:
     - Get observation (partial clues + choices)
     - Forward model to get action probabilities
     - Deterministically select action (argmax)
     - Step environment
   - Record prediction, confidence, reward, buzz position, category
4. `MetricsTracker` computes: accuracy, avg_reward, ECE, Brier score, category breakdown, buzz stats
5. Control experiment: repeat with choices-only (no clues)

**State Management:**

- `Question` objects are immutable
- `QuizBowlEnvironment` maintains episode state: current_clue_idx, done flag, selected_answer
- `RolloutBuffer` stores complete episode trajectories for PPO minibatches
- Model weights are updated via optimizer, checkpointed to disk
- Training history (losses, accuracies) stored in JSON

## Key Abstractions

**Question:**
- Purpose: Represents a quiz bowl question with pyramidal clues and multiple-choice answers
- Examples: `environment.py` lines 10-18
- Pattern: Immutable dataclass with question_id, clues, answer_choices, correct_answer_idx, category

**QuizBowlEnvironment (POMDP):**
- Purpose: Encapsulates the decision-making loop: reveal clue → decide to wait or answer
- Examples: `environment.py` lines 21-193
- Pattern: Gym-like interface with reset()/step() returning observation/reward/done/info
- Actions: 0=WAIT, 1-4=SELECT answer (maps to answer_choices[0-3])
- Rewards: R_t = 1.0 - 0.1×(t/T) if correct, -0.1×(t/T) if incorrect

**T5PolicyModel:**
- Purpose: Combines pretrained T5 encoder with learned policy head for action selection
- Examples: `model.py` lines 77-346
- Pattern: Wraps T5ForConditionalGeneration, adds PolicyHead for wait/answer/value outputs
- Encoding: Text input → T5 tokenizer → encoder → mean-pooled hidden state
- Policy: Pooled state → wait_head (binary), answer_head (num_choices), value_head (scalar)

**SupervisedTrainer:**
- Purpose: Warm-start training on complete questions with cross-entropy loss
- Examples: `train_supervised.py` lines 21-267
- Pattern: Takes full question (all clues visible) → predict answer choice → CE loss vs label
- Gradient accumulation: accumulate 4 steps → effective batch size 32 vs 8 actual

**PPOTrainer:**
- Purpose: Policy optimization with clipped surrogate loss and GAE advantages
- Examples: `train_ppo.py` lines 107-434
- Pattern: Rollout collection → advantage computation → PPO minibatch updates with clipping
- Clipping: PPO_clip_ratio=0.2 prevents large policy updates

**RolloutBuffer:**
- Purpose: Stores episode trajectories and computes discounted returns/advantages
- Examples: `train_ppo.py` lines 37-105
- Pattern: Accumulates complete episodes, computes GAE backward pass for advantages

**MetricsTracker:**
- Purpose: Accumulates predictions and computes multiple evaluation metrics
- Examples: `metrics.py` lines 12-185
- Pattern: Update() for each sample, then compute_*() methods for metrics
- Metrics: accuracy, avg_reward, ECE (calibration), Brier score, category breakdown, buzzing statistics

## Entry Points

**main.py:**
- Location: `main.py` lines 75-206
- Triggers: `python main.py --mode {supervised|ppo|full|eval}`
- Responsibilities:
  - Parse arguments (mode, model_path, epochs, iterations, device, seed)
  - Setup config with CLI overrides
  - Initialize datasets
  - Route to phase-specific trainer
  - Coordinate full pipeline (supervised → PPO → eval)

**train_supervised.py:**
- Location: `train_supervised.py` function `run_supervised_training()` (lines 190-267)
- Triggers: Called from main.py with config, datasets
- Responsibilities:
  - Create SupervisedTrainer
  - Loop over epochs: train on batches, validate, save checkpoints
  - Log training history

**train_ppo.py:**
- Location: `train_ppo.py` function `run_ppo_training()` (lines 390-434)
- Triggers: Called from main.py with config, datasets, pretrained_model_path
- Responsibilities:
  - Load pretrained model or start from scratch
  - Create PPOTrainer
  - Loop over iterations: collect rollouts, compute advantages, PPO updates
  - Log metrics, save checkpoints

**demo.py:**
- Location: `demo.py` function `main()` (lines 200-264)
- Triggers: `python demo.py --model_path <path> --mode {sample|interactive}`
- Responsibilities:
  - Load trained model
  - Run inference on demo questions
  - Display step-by-step clue revelation and model predictions

**visualize.py:**
- Location: `visualize.py` functions plot_*()/main() (lines 150-280)
- Triggers: `python visualize.py --checkpoint_dir <path>`
- Responsibilities:
  - Load training history from checkpoints
  - Generate plots: training curves, reliability diagrams, category performance

## Error Handling

**Strategy:** Exception propagation with descriptive messages

**Patterns:**

- **Invalid actions:** `QuizBowlEnvironment.step()` raises `ValueError` if action outside [0, 4]
- **Missing models:** `main.py` line 117-120 checks for pretrained model, warns if missing
- **Device selection:** `config.py` line 62 auto-selects cuda/mps/cpu with fallback
- **Tokenization overflow:** `model.py` uses truncation=True to handle long inputs
- **File I/O:** `dataset.py` uses try-except for JSON/CSV loading with pathlib for path safety

## Cross-Cutting Concerns

**Logging:**
- Approach: Print statements to stdout via tqdm progress bars, JSON history files
- Key files: `train_supervised.py` lines 128-180, `train_ppo.py` lines 260-350

**Validation:**
- Supervised phase: Cross-entropy loss automatically validates batch shapes
- PPO phase: Assertions check rollout consistency in `compute_returns_and_advantages()`
- No explicit schema validation; relies on type hints and runtime exceptions

**Authentication:**
- Not applicable (no external APIs or auth)

**Device Management:**
- Centralized in `Config.DEVICE` with fallback logic
- All tensors explicitly moved to device via `.to(self.device)`
- Models initialized with `.to(device)` after construction

**Random Seed Control:**
- `main.py` lines 82-86 sets numpy, torch, random seeds
- `dataset.py` seeds random/np.random in loader functions
- Ensures reproducible train/val/test splits and sampling

---

*Architecture analysis: 2026-02-23*
