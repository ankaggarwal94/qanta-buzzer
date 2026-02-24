# Codebase Concerns

**Analysis Date:** 2026-02-23

## Tech Debt

**No exception handling in training loops:**
- Issue: Training scripts (`train_supervised.py`, `train_ppo.py`, `metrics.py`) lack try-except blocks for handling data loading errors, tokenization failures, or OOM conditions
- Files: `train_supervised.py:80-200`, `train_ppo.py:135-300`, `metrics.py:295-330`
- Impact: Training will crash without graceful failure messages. Memory errors during large batch processing are unrecoverable.
- Fix approach: Add try-except blocks around tokenization, model forward passes, and disk I/O operations. Add memory monitoring and checkpoint recovery.

**Loose coupling between dataset splits and training:**
- Issue: `setup_datasets()` in `dataset.py:480-550` shuffles raw questions before splitting, but if dataset loading fails partway through, train/val/test splits may become contaminated or desynchronized
- Files: `dataset.py:480-550`, `main.py:90-95`
- Impact: Accidental train/test leakage possible if splits are regenerated mid-experiment. Reproducibility issues.
- Fix approach: Implement deterministic split based on question IDs (not shuffle index). Validate split integrity on load.

**No input validation for CSV data:**
- Issue: `QANTADatasetLoader.load_from_csv()` (`dataset.py:90-170`) assumes CSV columns ('Text', 'Answer', 'Category') exist and are non-empty. Malformed CSV crashes silently.
- Files: `dataset.py:95-115`
- Impact: Invalid datasets will cause obscure tokenization or shape errors downstream during training.
- Fix approach: Add schema validation at load time. Log row-level errors and skip malformed rows with warnings.

**Hardcoded paths in multiple locations:**
- Issue: Checkpoint directories, data paths, model names are scattered across config and individual files
- Files: `config.py:50-55`, `main.py:110`, `train_ppo.py:520`, `metrics.py:300`
- Impact: Difficult to test or run on different systems. Path conflicts if multiple jobs run simultaneously.
- Fix approach: Consolidate all path logic to `config.py`. Use timestamped subdirectories for concurrent runs.

**Missing gradient clipping in supervised training:**
- Issue: `train_supervised.py:125-135` clips gradients but only after accumulation steps, not per mini-batch
- Files: `train_supervised.py:125-140`
- Impact: Gradient spikes can accumulate during early accumulation steps, degrading training stability.
- Fix approach: Move gradient clipping before accumulation or clip each gradient step individually.

## Known Bugs

**Off-by-one error in clue indexing:**
- Symptoms: When showing all clues in supervised training, `env.current_clue_idx = len(question.clues) - 1` is set but `get_text_representation()` shows clues `[:self.current_clue_idx + 1]`, which correctly includes all clues. However, if environment is used for RL after supervised training, the clue representation may be inconsistent.
- Files: `train_supervised.py:71-72`, `environment.py:138-142`
- Trigger: Running PPO after supervised training without proper environment reset verification
- Workaround: Always verify `env.reset()` is called before stepping

**Reward computation doesn't match specification:**
- Symptoms: Time penalty in `environment.py:120` is `penalty = reward_time_penalty * (current_clue_idx / num_clues)` but should be `(current_clue_idx + 1) / num_clues` to match problem statement
- Files: `environment.py:119-122`
- Trigger: Every episode with time penalty > 0
- Workaround: None; penalty values are slightly optimistic

**Deterministic action selection during training:**
- Symptoms: PPO uses `deterministic=False` for collection but supervised training uses `deterministic=True` during validation. During PPO training, the model may explore suboptimal policies that supervised training never saw.
- Files: `train_ppo.py:197-198`, `train_supervised.py:142-146`
- Trigger: PPO performance plateau or divergence from supervised baseline
- Workaround: Use same determinism setting for fair comparison

## Security Considerations

**No input sanitization for model paths:**
- Risk: `T5PolicyModel.load_pretrained()` (`model.py:380-420`) uses `from_pretrained()` which could load arbitrary models from HuggingFace Hub if path is user-controlled
- Files: `model.py:380-420`, `main.py:123-127`
- Current mitigation: Only local filesystem paths in default flow, but no validation
- Recommendations: Validate model paths are within expected directories. Whitelist allowed model names.

**Unvalidated tokenizer inputs:**
- Risk: Dataset CSV answers are passed directly to tokenizer without sanitization (`dataset.py:150-155`)
- Files: `dataset.py:150-155`
- Current mitigation: T5 tokenizer is robust, but arbitrary text could cause issues
- Recommendations: Add length limits and character validation on answer strings

## Performance Bottlenecks

**Sequential rollout collection is O(n*episode_length):**
- Problem: `collect_rollouts()` in `train_ppo.py:150-215` processes episodes sequentially in Python loop. With T5-large model, each forward pass is slow.
- Files: `train_ppo.py:150-215`
- Cause: Episodes vary in length (1-6 clues), making batching difficult. No parallelization.
- Improvement path: Implement vectorized environment steps or multi-threaded collection. Use batch processing for all forward passes.

**Full validation on every epoch during supervised training:**
- Problem: `validate()` in `train_supervised.py:142-148` runs on entire val set after every epoch. With 75+ validation questions * 6 clues * tokenization, this is ~450+ forward passes per epoch.
- Files: `train_supervised.py:117-150`
- Cause: No sampling or early stopping. Validation doesn't scale.
- Improvement path: Sample 10-20% of validation set for frequent checks. Only full validation every N epochs.

**Tokenization happens per-step in RL evaluation:**
- Problem: `evaluate_model()` in `metrics.py:295-330` tokenizes every observation step-by-step. For 75 test questions * 6 steps, this is 450 tokenizations.
- Files: `metrics.py:295-330`
- Cause: Observation changes per step (more clues revealed), but question/choice part is constant. No caching.
- Improvement path: Cache tokenized question/choices. Only retokenize when clues change.

**T5-large model loading on every checkpoint load:**
- Problem: `load_pretrained()` in `model.py:380-420` reloads entire T5 model from disk even for multi-step training
- Files: `model.py:380-420`
- Cause: Each training phase reloads model (supervised → PPO → eval)
- Improvement path: Keep model in memory between phases or use memory-mapped weights

## Fragile Areas

**Environment state machine is brittle:**
- Files: `environment.py:60-130`
- Why fragile: Multiple mutable fields (`current_clue_idx`, `done`, `selected_answer`). Calling `step()` without `reset()` raises ValueError, but no state validation. If `_get_observation()` is called during episode, it returns mutable dict that could be modified.
- Safe modification: Use immutable observation tuples or frozen dataclasses. Add state assertions in every method.
- Test coverage: Environment has 2 raise statements but no tests that verify exception paths. `test_imports.py` doesn't test environment.

**Policy head architecture is fixed:**
- Files: `model.py:15-55`
- Why fragile: Wait/answer heads use hard-coded layer sizes (256, 512). If num_choices changes from 4, only `answer_head` updates; wait_head doesn't. No validation of output shapes.
- Safe modification: Parameterize all layer sizes. Add shape assertions in forward pass.
- Test coverage: Only loaded via T5PolicyModel. No unit tests for PolicyHead forward pass.

**Dataset splits are cached but not versioned:**
- Files: `dataset.py:480-550`
- Why fragile: Once splits are saved to disk (`train_dataset.json`, `val_dataset.json`, `test_dataset.json`), `setup_datasets()` always reuses them. If config changes, splits don't refresh.
- Safe modification: Add version number to split files. Invalidate cache when config hash changes.
- Test coverage: Only tested via `test_csv_loader.py` which tests CSV parsing, not split integrity. No test verifies splits are consistent.

**Reward function is not differentiable:**
- Files: `environment.py:119-130`
- Why fragile: Rewards are computed in environment (NumPy), not PyTorch. This means reward shaping or gradient-based exploration would fail. If reward formulation changes, it must be changed in two places: environment and any downstream loss computations.
- Safe modification: Move reward computation to PyTorch in `train_ppo.py`. Make it a differentiable function.
- Test coverage: Reward computation is not tested. No unit tests for step() return values.

## Scaling Limits

**Dataset size is hard-limited to config.NUM_QUESTIONS:**
- Current capacity: 500 questions (config.py:48)
- Limit: Memory usage scales linearly with dataset size. T5 tokenization on 500 questions takes ~30s. At 5000+ questions, data loading becomes bottleneck.
- Scaling path: Implement streaming dataset loader (load batches from disk on-demand). Use HuggingFace datasets library for efficient caching.

**Batch size is fixed at training time:**
- Current capacity: PPO_BATCH_SIZE=32, SUPERVISED_BATCH_SIZE=8
- Limit: Effective batch size = 8 * 4 (grad accum) = 32 for supervised. If GPU memory allows larger, cannot scale without config change + restart.
- Scaling path: Implement dynamic batch sizing based on available GPU memory. Use gradient checkpointing to reduce memory per batch.

**Model checkpoints are uncompressed:**
- Current capacity: T5-large checkpoints are ~2.4GB each. Saving best_model + PPO checkpoints fills disk quickly.
- Limit: Storage grows linearly with number of saved checkpoints. At 50+ checkpoints, can exceed 100GB.
- Scaling path: Only save recent N checkpoints. Compress old checkpoints to tar.gz. Use distributed checkpoint format (sharding).

**Single GPU / single machine:**
- Current capacity: T5-large is 770M params. Fits on 1 GPU with gradient accumulation on most modern cards (24GB+ VRAM).
- Limit: Training one policy takes 48+ hours on V100. Cannot parallelize training across multiple experiments.
- Scaling path: Implement distributed training with torch.distributed. Use ray or Ludwig for experiment parallelization.

## Dependencies at Risk

**transformers library version lock:**
- Risk: `requirements.txt` specifies `transformers>=4.30.0` but HuggingFace makes breaking API changes frequently (e.g., `from_pretrained` signature changes)
- Files: `requirements.txt`, `model.py:100-110`
- Impact: Major version bumps (4.30 → 4.40 → 5.0) may break model loading
- Migration plan: Pin to exact version `transformers==4.30.2` during development. Test major version upgrades in isolated environment.

**torch version compatibility:**
- Risk: Code uses `torch.distributions.Categorical` and `torch.nn.utils.clip_grad_norm_()` which have stability variations across versions
- Files: `requirements.txt:1`, `model.py:320-330`, `train_ppo.py:304`
- Impact: torch 2.0 has different semantics for some operations than 1.13
- Migration plan: Test on both torch 2.0.x and 2.1.x. Document minimum version.

**numpy random state management:**
- Risk: Code uses both `random.seed()` and `np.random.seed()` but torch also has its own seed. No coordinated seeding.
- Files: `dataset.py:68-70`, `main.py:75-80`, `train_ppo.py:200`
- Impact: Reproducibility breaks if pytorch version changes RNG algorithm
- Migration plan: Use a unified seeding utility. Verify reproducibility across versions.

## Missing Critical Features

**No checkpoint recovery on failure:**
- Problem: If training crashes at iteration 200/250, no way to resume. Must restart from epoch 0.
- Blocks: Multi-GPU training, long training runs, hyperparameter sweeps
- Fix: Save optimizer state, LR scheduler state, and training step counter. Implement `resume_from_checkpoint()`.

**No learning rate scheduling:**
- Problem: Config.py hardcodes learning rates but no scheduler. LR stays constant across epochs.
- Files: `config.py:22-24`, `train_supervised.py:50-55`
- Blocks: Optimal learning requires decay. Current setup may undershoot or overshoot in later epochs.
- Fix: Add `lr_scheduler` (e.g., `CosineAnnealingLR`). Make it configurable.

**No hyperparameter sweep infrastructure:**
- Problem: Cannot easily run grid search over PPO_CLIP_RATIO, PPO_LR, PPO_ENTROPY_COEF, etc.
- Blocks: Finding optimal hyperparameters requires manual config edits and reruns
- Fix: Add argparse for key hyperparams. Use wandb or ray tune for sweep orchestration.

**No data augmentation:**
- Problem: Dataset is fixed-size after loading. No augmentation, no curriculum learning.
- Blocks: Limited data diversity. Overfitting risk on small datasets.
- Fix: Implement on-the-fly clue resampling, paraphrase augmentation using T5, answer distractors resampling.

## Test Coverage Gaps

**No unit tests for environment state machine:**
- What's not tested: Environment.step() with all action types, reward computation, terminal state handling, observation format
- Files: `environment.py:60-130`
- Risk: State mutations or off-by-one errors in environment could propagate through entire training pipeline undetected
- Priority: High (environment is core POMDP)

**No tests for dataset integrity:**
- What's not tested: Dataset splits don't leak (train ∩ test = ∅), correct answer indices are valid, category distribution matches config
- Files: `dataset.py:480-550`
- Risk: Accidental data leakage between train/test during grid search or multi-run experiments
- Priority: High (data integrity is prerequisite for valid results)

**No tests for model shapes:**
- What's not tested: PolicyHead output shapes for batch processing, token embedding dimensions, value head outputs
- Files: `model.py:15-55`, `model.py:180-210`
- Risk: Shape mismatches only caught at runtime during training, wasting hours
- Priority: Medium

**No tests for metrics computation:**
- What's not tested: ECE calculation correctness, category accuracy binning, confidence calibration
- Files: `metrics.py:50-130`
- Risk: Incorrect metric reporting. Published results based on wrong ECE/accuracy could be invalid.
- Priority: Medium

**No integration tests for full training pipeline:**
- What's not tested: Full supervised training → PPO training → evaluation pipeline with small dataset
- Files: `main.py:73-168`
- Risk: Regressions in main.py CLI argument handling, dataset loading pipeline order, checkpoint saving/loading flow undetected
- Priority: Medium

**No tests for distributed training (if implemented):**
- What's not tested: Multi-GPU forward pass synchronization, gradient averaging, checkpoint saving in distributed mode
- Files: N/A (not yet implemented)
- Risk: Future multi-GPU training will have subtle bugs
- Priority: Low (for now)

---

*Concerns audit: 2026-02-23*
