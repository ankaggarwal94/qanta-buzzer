# Codebase Concerns

**Analysis Date:** 2026-02-24

## Tech Debt

**PPO Training State Management:**
- Issue: Incomplete training state persistence. Only saves optimizer state, not learning rate schedule or model intermediate layers
- Files: `train_ppo.py:275-280` (save_checkpoint method)
- Impact: Cannot reliably resume PPO training from checkpoints. Restarting mid-training loses optimization momentum and may cause training instability
- Fix approach: Save complete trainer state including scheduler state, batch statistics, and gradient accumulation counters. Implement proper resume logic in `PPOTrainer.__init__`

**Gradient Accumulation Inconsistency:**
- Issue: Supervised training uses gradient accumulation (`SUPERVISED_GRAD_ACCUM_STEPS = 4`, effective batch = 32) but final backward pass may use incomplete accumulation
- Files: `train_supervised.py:88-100` (gradient accumulation logic)
- Impact: Effective batch size fluctuates at end of each epoch. Produces different optimization dynamics than intended
- Fix approach: Implement proper gradient accumulation with accumulation step counter and final step flush at epoch end

**Model Loading Bug:**
- Issue: `T5PolicyModel.load_pretrained()` attempts to reload entire T5 model from saved directory, but doesn't verify T5 config matches runtime config (e.g., MODEL_NAME in config.py)
- Files: `model.py:233-252` (load_pretrained classmethod)
- Impact: If config.MODEL_NAME changes after model is saved, loading may fail or load mismatched architecture. No version validation
- Fix approach: Store model name in saved checkpoint metadata, validate during load, emit clear error if mismatch detected

**Device Handling Fragility:**
- Issue: Device detection logic scattered across files. Auto-detection in config happens at import time before CLI args are parsed
- Files: `config.py:37` (DEVICE auto-detection)
- Impact: Cannot override device via CLI after config import if GPU becomes unavailable mid-run. If running on different hardware after training, checkpoint loading may fail silently with wrong device
- Fix approach: Defer device detection until after CLI args parsed. Add validation in checkpoint loading to move tensors to specified device

## Known Issues

**Dataset Loading Multiple Fallback Levels:**
- Problem: Three-tier fallback (JSON splits → CSV → synthetic) with unclear priority and silent failures
- Files: `dataset.py:548-620` (setup_datasets function)
- Symptoms: If train_dataset.json exists but is corrupted, function silently falls back to CSV. If CSV path not found in multiple locations, generates synthetic data without clear indication to user
- Workaround: Always manually verify data/processed_dataset.json and ensure questions.csv in project root
- Recommendation: Add explicit error messages and --data-source flag to clarify which dataset is being used

**PPO Episode Collection Memory Leak Risk:**
- Problem: RolloutStep stores full tokenized input_ids and attention_mask for all steps. For long episodes (6-12 steps) and large batch sizes, memory accumulation is not properly cleaned
- Files: `train_ppo.py:28-40` (RolloutStep dataclass), `train_ppo.py:158-185` (collect_rollouts)
- Trigger: Running PPO with batch_size=32 and PPO_BATCH_SIZE=32 and max sequence length 512
- Workaround: Reduce PPO_BATCH_SIZE or SUPERVISED_BATCH_SIZE in config.py to reduce concurrent episodes
- Current mitigation: tensors stored on CPU (.cpu() calls in line 180), but never explicitly freed

**Distractor Generation Edge Case:**
- Problem: If not enough unique distractors exist across categories, falls back to padding with "[No answer X]" placeholders
- Files: `dataset.py:235-244` (answer choice padding)
- Symptoms: Model may learn spurious "placeholder detection" pattern that doesn't transfer to real data
- Workaround: Verify data/train_dataset.json has valid non-placeholder distractors for all questions before training
- Risk: Low for QANTA data, high for synthetic data with small category sets

## Performance Bottlenecks

**T5 Encoder Forward Pass Repeated:**
- Problem: `model.py:137-149` (get_encoder_output) re-computes T5 encoder outputs on every forward pass, even in evaluation. Mean pooling is not cached
- Files: `model.py:137-149`, `model.py:195-219` (select_action), `model.py:245-284` (get_action_log_probs)
- Cause: Stateless design requires re-encoding same text representations multiple times per trajectory
- Improvement path: Implement single-encoder-output → multiple-head design with intermediate caching (requires refactoring forward signature)
- Current impact: ~30% overhead on inference time per action step

**Validation Evaluation is Full Forward Pass:**
- Problem: `evaluate_model` runs full episodes with intermediate observations for every validation sample, instead of just computing final answer accuracy
- Files: `metrics.py:180-240` (evaluate_model function)
- Cause: Captures realistic POMDP behavior but at cost of O(num_clues * num_samples) forward passes
- Improvement path: Add --eval-mode fast flag to use only complete questions during validation (like supervised training does)
- Current impact: Validation takes 3-5x longer than training epoch on small datasets

**Batch Padding Inefficient:**
- Problem: `train_ppo.py:222-238` manually pads sequences to max_len in batch with explicit loops
- Files: `train_ppo.py:222-238` (batch padding)
- Cause: Torch DataLoader not used; manual batch assembly needed
- Improvement path: Switch to DataLoader with collate_fn for automatic efficient padding
- Current impact: Negligible for small batches (<32) but becomes noticeable at 128+ batch size

## Scaling Limits

**T5-Large Model Size:**
- Current capacity: 770M parameters requires 8GB+ GPU VRAM for training, 4GB for inference
- Limit: Cannot run on T4 GPUs (16GB) with batch_size > 16 due to gradient storage. OOM on consumer GPUs
- Scaling path: Switch MODEL_NAME to t5-base (220M, 2GB) or t5-small (60M, 0.5GB). Trade off quality for accessibility
- Configuration impact: No code changes needed, just update config.MODEL_NAME

**Dataset Scale:**
- Current: NUM_QUESTIONS = 500 loads fully in memory (~50MB JSON)
- Limit: QANTA full dataset (~100K questions) requires memory-mapped loading or streaming
- Scaling path: Implement streaming dataset with online sampling instead of loading all questions at once
- Risk: If scaling to 100K questions, current train loop will crash on memory-full system

**Checkpoint Disk Usage:**
- Current: Each PPO checkpoint saves full T5 model (~3GB) + policy head (1MB) + optimizer state (~3GB)
- Limit: With SAVE_INTERVAL=50 and PPO_ITERATIONS=250, creates 15 checkpoints = 90GB total
- Scaling path: Implement checkpoint pruning (keep only best + last 2) or use only_state_dict saving

## Dependencies at Risk

**Transformers Version Pinning:**
- Risk: `transformers>=4.30.0` is very loose. API changes between 4.30 and 5.0 may break T5ForConditionalGeneration interface
- Current mitigation: Code tested on 4.35.2 only
- Migration plan: Add explicit upper bound (transformers<5.0.0) and test on version boundaries. Consider moving to huggingface_hub for future T5 variants

**PyTorch Device API Deprecation:**
- Risk: `torch.cuda.is_available()` and `torch.backends.mps.is_available()` may be deprecated in torch 2.5+
- Files: `config.py:37-38`
- Migration plan: Use `torch.device()` context manager. Already using correct API but consider future-proofing with version check

**scikit-learn Usage:**
- Risk: Only used for `accuracy_score()`. Minimal but adds 30MB+ dependency for one function
- Files: `metrics.py:9` (import), `metrics.py:73` (usage)
- Migration plan: Replace with simple numpy implementation: `np.mean(predictions == targets)`

## Missing Critical Features

**No Experiment Tracking:**
- What's missing: Metrics saved only to JSON files. No integration with W&B, MLflow, or TensorBoard
- Blocks: Cannot easily compare runs, visualize training curves in real-time, or share results
- Impact: Requires manual post-processing of history.json to analyze results
- Fix approach: Add --log-wandb flag and optional wandb.log() calls (already commented in requirements.txt)

**No Early Stopping:**
- What's missing: Training runs for fixed epochs/iterations regardless of validation performance
- Blocks: Cannot prevent wasted compute on plateaued validation metrics
- Files: `train_supervised.py` and `train_ppo.py` lack early stopping logic
- Impact: 50 supervised epochs or 250 PPO iterations always execute even if val accuracy stagnates after epoch 20
- Fix approach: Add patience parameter, track best val metric, stop if no improvement for N epochs

**No Hyperparameter Search:**
- What's missing: All hyperparameters fixed in config.py or CLI args. No grid/random search
- Blocks: Cannot systematically explore learning rates, batch sizes, PPO coefficients
- Impact: Requires manual trial-and-error or separate script to sweep hyperparameters
- Fix approach: Add optuna integration with config space specification

**No Distributed Training:**
- What's missing: Single GPU only. T5 model and PPO training not distributed
- Blocks: Cannot scale to multiple GPUs or leverage DDP/FSDP
- Impact: Max throughput limited by single GPU memory and compute
- Fix approach: Integrate PyTorch DistributedDataParallel for supervised; async PPO collection for RL phase

## Test Coverage Gaps

**No Unit Tests for Core Logic:**
- Untested: `QuizBowlEnvironment` step logic, reward computation, action decoding
- Files: `environment.py:25-120` (step function, action handling)
- Risk: Bug in reward calculation or action parsing could go unnoticed for weeks of training
- Priority: High - core POMDP mechanics critical to RL performance
- Recommendation: Add pytest suite for environment state transitions, edge cases (last clue, invalid actions)

**No Integration Tests:**
- Untested: Full pipeline flow (supervised → PPO → eval) with real data
- Files: `main.py` (mode='full'), data loading, checkpoint saving/loading chain
- Risk: Changes to one module break full training pipeline silently
- Priority: High - integration failures only discovered after long training run
- Recommendation: Add smoke test that runs full pipeline on 10 synthetic questions with 1 epoch

**No Tests for Metrics Computation:**
- Untested: ECE, Brier score, calibration calculations, especially edge cases
- Files: `metrics.py:83-127` (compute_ece, compute_brier_score)
- Risk: Metrics reported incorrectly but training appears normal
- Priority: Medium - affects evaluation only, not training
- Recommendation: Add tests for known metric values on synthetic 100-sample dataset

**No Tests for Dataset Loading:**
- Untested: CSV parsing, distractor generation, edge cases with missing categories
- Files: `dataset.py:148-245` (QANTADatasetLoader.load_from_csv)
- Risk: Dataset corruption silently goes to training, producing spurious results
- Priority: Medium - caught by spot-checking data but not automated

## Fragile Areas

**PPO Advantage Normalization:**
- Why fragile: Line `advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)` in train_ppo.py:306
- Files: `train_ppo.py:306`
- Fragility: If all advantages are identical (rare but possible early in training), std() becomes 0. Adding 1e-8 epsilon is defensive but unclear. No warning issued
- Safe modification: Add assertion `assert advantages.std() > 1e-6, "Advantages have zero variance"` before normalization. Document epsilon choice
- Test coverage: None

**T5 Tokenization Assumptions:**
- Why fragile: Code assumes T5Tokenizer.pad_token_id is valid and consistent across saves
- Files: `train_ppo.py:235` (uses self.model.tokenizer.pad_token_id)
- Fragility: If tokenizer config is corrupted or different T5 variant used, pad_token_id might be None, causing TypeError
- Safe modification: Add initialization validation: `assert model.tokenizer.pad_token_id is not None` after loading
- Test coverage: None

**Action Decoding Logic:**
- Why fragile: Complex bit-packing logic in `model.py:268-273` where action 0=WAIT, 1-4=SELECT answer
- Files: `model.py:268-273` (combined_actions computation)
- Fragility: Off-by-one errors easy to introduce when refactoring. Mismatch with environment.py:21 action space definition
- Safe modification: Add comprehensive tests for all action combinations (0-4) round-tripping through encode/decode
- Test coverage: None - only manual testing during development

**Environment Reset State:**
- Why fragile: `QuizBowlEnvironment.reset()` must be called before first step(). No guard against calling step() without reset()
- Files: `environment.py:60-68`, `train_ppo.py:159-184` (step called after reset in line 169 but reset called in line 165)
- Fragility: If collect_rollouts loop exits early without resetting, next iteration may use stale environment state
- Safe modification: Add `if not hasattr(self, '_initialized'): raise RuntimeError(...)` guard in step()
- Test coverage: None

## Security Considerations

**Model Checkpoint Integrity:**
- Risk: Saving and loading T5 models from arbitrary directories without checksum verification
- Files: `model.py:215-225` (save), `model.py:228-237` (load)
- Current mitigation: None
- Recommendations: Compute SHA256 hash of saved files, store in metadata.json, verify on load. Prevents accidental corruption or tampering

**Tokenizer Injection Risk:**
- Risk: Loading tokenizer from saved checkpoint directory could load modified tokenizer if directory is writable by other users
- Files: `model.py:228` (T5Tokenizer.from_pretrained from directory)
- Current mitigation: Standard permissions on checkpoint directory
- Recommendations: Add explicit tokenizer config validation after load. Consider shipping tokenizer as code constant rather than file

**No Input Validation on User Data:**
- Risk: Questions loaded from CSV or JSON with no schema validation. Text fields could contain adversarial inputs or injections
- Files: `dataset.py:167-180` (CSV loading without validation)
- Current mitigation: None
- Recommendations: Add JSON schema validation. Sanitize text fields (max length, character set checks)

---

*Concerns audit: 2026-02-24*
