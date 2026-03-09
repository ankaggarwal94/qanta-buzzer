# Architecture

**Analysis Date:** 2026-02-24

## Pattern Overview

**Overall:** Two-phase transfer learning with RL fine-tuning for incremental question answering.

**Key Characteristics:**
- Pre-trained T5 encoder backbone with custom multi-head policy output
- Supervised warm-start phase learns answer selection from complete questions
- PPO fine-tuning phase learns when to "buzz in" as clues are revealed incrementally
- POMDP formulation: agent observes partial clues + answer choices, decides to WAIT or SELECT
- Gradient accumulation in supervised phase; GAE (Generalized Advantage Estimation) in PPO phase

## Layers

**Presentation Layer:**
- Purpose: CLI orchestration and mode selection
- Location: `main.py`
- Contains: Argument parsing, configuration setup, phase routing
- Depends on: All training modules
- Used by: Entry point for all training modes

**Model Layer:**
- Purpose: Neural network architecture combining T5 encoder with policy heads
- Location: `model.py` (T5PolicyModel, PolicyHead)
- Contains: T5 tokenization, encoder forward pass, three independent policy heads (wait/answer/value)
- Depends on: transformers library (T5), torch
- Used by: Training and inference pipelines

**Environment Layer:**
- Purpose: POMDP simulation for incremental question answering
- Location: `environment.py` (QuizBowlEnvironment, BatchedEnvironment, Question)
- Contains: Observation generation, action execution, reward computation, episodic state management
- Depends on: None (pure Python/dataclass)
- Used by: PPO data collection, evaluation

**Dataset Layer:**
- Purpose: Data loading, preprocessing, and augmentation
- Location: `dataset.py` (QuizBowlDataset, QANTADatasetLoader, SyntheticDatasetGenerator)
- Contains: CSV parsing, distractor generation, train/val/test splitting, JSON serialization
- Depends on: csv, json, random, numpy
- Used by: Training initialization and supervised batch preparation

**Configuration Layer:**
- Purpose: Centralized hyperparameter management
- Location: `config.py` (Config class)
- Contains: Model hyperparameters, learning rates, batch sizes, device selection, data paths
- Depends on: torch (for device detection)
- Used by: All components via dependency injection

**Training Layer:**
- Purpose: Two independent training pipelines
- Location: `train_supervised.py` (SupervisedTrainer), `train_ppo.py` (PPOTrainer)
- Contains: Loss computation, optimization loops, checkpoint management, validation
- Depends on: model, dataset, metrics, environment
- Used by: main.py orchestration

**Metrics & Evaluation:**
- Purpose: Performance measurement and logging
- Location: `metrics.py` (MetricsTracker, evaluate_model, evaluate_choices_only)
- Contains: Accuracy, ECE (Expected Calibration Error), Brier score, reward tracking, category breakdown
- Depends on: numpy, torch, sklearn
- Used by: Training validation loops and final evaluation

## Data Flow

**Supervised Training Phase:**

1. `main.py` → `run_supervised_training()`
2. Load datasets via `setup_datasets()` → CSV parsing or synthetic generation → train/val/test splits
3. Initialize `T5PolicyModel` → Load T5-large encoder + create 3 policy heads
4. `SupervisedTrainer.train_epoch()`:
   - Get batch from dataset
   - Prepare batch: Create text with ALL clues visible, tokenize
   - Forward: `model.predict_answer()` → encoder output → answer head → logits
   - Loss: CrossEntropyLoss between answer logits and correct answer index
   - Backward: Gradient accumulation (effective batch = 8 * 4 = 32)
   - Update: AdamW optimizer step every 4 accumulated gradients
5. Validate: Run deterministic inference on validation set
6. Save best model by validation accuracy to `checkpoints/supervised/best_model/`

**PPO Training Phase:**

1. `main.py` → `run_ppo_training()` with pretrained supervised checkpoint
2. Load pretrained model: T5 + policy head weights
3. `PPOTrainer.train()` loop (250 iterations by default):
   - `collect_rollouts()`:
     - Sample N questions from training set (batch_size episodes in parallel via BatchedEnvironment)
     - For each episode: reset environment, loop until done:
       - Get text representation: "CLUES: clue1 clue2 ... | CHOICES: (1) ans1 (2) ans2 (3) ans3 (4) ans4"
       - `model.select_action()` → sample wait/answer actions from policy heads
       - Combine into single action: 0=WAIT, 1-4=SELECT answer 0-3
       - `env.step(action)` → reveal next clue if WAIT, or end episode with reward if SELECT
       - Store RolloutStep: observation, action, reward, value, log_prob
   - `update_policy()`:
     - Compute GAE returns and advantages across all rollouts
     - Normalize advantages
     - 4 epochs of PPO updates (PPO_EPOCHS_PER_ITER = 4):
       - Shuffle all steps, iterate in mini-batches
       - `model.get_action_log_probs()` → current log probs and entropy for stored actions
       - PPO objective: min(ratio * advantage, clipped_ratio * advantage) - entropy bonus + value loss
       - Gradient clipping (PPO_MAX_GRAD_NORM = 0.5)
       - AdamW optimizer step
   - Validate: Deterministic inference, track metrics
   - Save best model by validation reward to `checkpoints/ppo/best_model/`

**Evaluation Flow:**

1. Run deterministic episodes via `evaluate_model()`:
   - For each test question, loop environment until done (no stochasticity)
   - Collect metrics: accuracy, reward, buzz position, answer probabilities
2. Control experiment via `evaluate_choices_only()`:
   - Remove clues from input, use only answer choices
   - Compare accuracy to 0.25 baseline (1/4 random)
3. Save results to JSON: accuracy, ECE, Brier score, per-category breakdown

## Key Abstractions

**Question:**
- Purpose: Representation of a quiz bowl question with pyramidal clues
- Examples: `environment.py:Question` dataclass
- Pattern: Immutable dataclass with question_id, clues list, answer_choices list, correct_answer_idx, category, metadata

**QuizBowlEnvironment:**
- Purpose: Single-episode POMDP simulator
- Examples: `environment.py:QuizBowlEnvironment`
- Pattern: Stateful environment with reset/step interface. Maintains current_clue_idx, done, selected_answer. Returns observation dict, reward, done flag, info dict.

**T5PolicyModel:**
- Purpose: Combined encoder + policy head neural network
- Examples: `model.py:T5PolicyModel`
- Pattern: Wraps T5 encoder with three independent linear heads (wait, answer, value). Provides: forward(), select_action(), get_action_log_probs(), predict_answer() methods. Loads/saves via Hugging Face transformer utilities.

**PolicyHead:**
- Purpose: Multi-head decoder for policy and value outputs
- Examples: `model.py:PolicyHead`
- Pattern: Three Sequential modules (wait_head, answer_head, value_head). Takes mean-pooled encoder hidden state [batch_size, 1024] and outputs [batch_size, 2], [batch_size, 4], [batch_size, 1] respectively.

**RolloutStep:**
- Purpose: Single experience tuple for PPO
- Examples: `train_ppo.py:RolloutStep` dataclass
- Pattern: Stores observation_text, action, reward, done, value, log_prob, plus tokenized input_ids/attention_mask for efficient batching.

**RolloutBuffer:**
- Purpose: Efficient storage and batch computation over PPO rollouts
- Examples: `train_ppo.py:RolloutBuffer`
- Pattern: Accumulates list of episode rollouts. Provides get_all_steps() and compute_returns_and_advantages() using GAE algorithm. Modifies RolloutStep objects in-place to attach returns/advantages.

**SupervisedTrainer:**
- Purpose: Phase 1 training loop orchestration
- Examples: `train_supervised.py:SupervisedTrainer`
- Pattern: Manages optimizer, loss criterion, checkpoint directory. Methods: train_epoch(), validate(), train(), save_checkpoint(), save_history(). Tracks best_val_acc.

**PPOTrainer:**
- Purpose: Phase 2 training loop orchestration
- Examples: `train_ppo.py:PPOTrainer`
- Pattern: Manages optimizer, checkpoint directory, training state. Methods: collect_rollouts(), update_policy(), validate(), train(), save_checkpoint(), save_history(). Tracks best_val_reward.

**MetricsTracker:**
- Purpose: Accumulate and compute evaluation metrics
- Examples: `metrics.py:MetricsTracker`
- Pattern: Append-only tracker for predictions, targets, confidences, rewards, buzz_positions. Lazy computation of accuracy, ECE, Brier score, category accuracy, buzz position stats via compute_*() methods. Returns JSON-serializable summary.

## Entry Points

**main.py:**
- Location: `/Users/ankit.aggarwal/Dropbox/Stanford/CS234/final_project/qanta-buzzer/main.py`
- Triggers: `python main.py --mode {supervised|ppo|full|eval}`
- Responsibilities:
  - Parse CLI arguments (mode, model_path, hyperparameter overrides, seed)
  - Load config and apply overrides
  - Initialize random seeds (torch, numpy, random)
  - Call `setup_datasets()` to prepare train/val/test splits
  - Route to appropriate training function or evaluation
  - Handle checkpointing paths (e.g., supervised → PPO requires pretrained checkpoint)

**train_supervised.py (if run standalone):**
- Location: `train_supervised.py` main block
- Triggers: `python train_supervised.py`
- Responsibilities: Initialize Config, setup_datasets, create SupervisedTrainer, call trainer.train()

**train_ppo.py (if run standalone):**
- Location: `train_ppo.py` main block
- Triggers: `python train_ppo.py`
- Responsibilities: Initialize Config, setup_datasets, check for pretrained model at `checkpoints/supervised/best_model/`, call run_ppo_training()

## Error Handling

**Strategy:** Defensive early checks with informative error messages.

**Patterns:**
- `QuizBowlEnvironment.step()`: Raises ValueError if episode already done, or if invalid action index
- `main.py`: Warns if PPO mode requested but no supervised checkpoint found (allows starting PPO without pretraining but logs warning)
- `dataset.py`: Gracefully skips questions with insufficient clues (< 1 clue)
- `model.py`: Checks file existence before loading policy head checkpoint in `load_pretrained()`
- Gradient clipping in both phases (1.0 for supervised, PPO_MAX_GRAD_NORM=0.5 for PPO)
- Tokenizer padding/truncation applied uniformly (max_length=512)

## Cross-Cutting Concerns

**Logging:**
- TQDM progress bars in training loops (`train_epoch()`, `update_policy()`)
- Print statements for phase/iteration transitions and best metric updates
- JSON history files saved to checkpoint directories: `history.json` (supervised), `history.json` (PPO)
- Results saved as JSON: `test_results.json`

**Validation:**
- During supervised training: validate every epoch, select best model by validation accuracy
- During PPO training: validate every EVAL_INTERVAL (50 iterations), select best model by validation reward
- Deterministic inference (no sampling) used for validation/evaluation

**Device Management:**
- Config auto-detects device: cuda > mps > cpu
- Override via CLI: `--device {cuda|mps|cpu}`
- All tensors moved to device explicitly in trainer classes
- Model moved to device in `__init__` and after loading

**Reproducibility:**
- Seeds set explicitly: torch.manual_seed, np.random.seed, random.seed (all same value from Config.SEED or CLI)
- Deterministic action selection in evaluation via `deterministic=True` flag
- Data shuffling uses random.shuffle (respects seed)
- Batch sampling uses random.sample (respects seed)

---

*Architecture analysis: 2026-02-24*
