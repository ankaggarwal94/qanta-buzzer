# CS234 RL Question Answering - Python Implementation

Complete implementation of the CS234 project: **Reinforcement Learning for Calibrated Question Answering with Pyramidal Clues and Multiple-Choice Constraints**.

## Project Structure

```
.
├── config.py                 # Configuration settings
├── environment.py            # POMDP environment for quiz bowl
├── dataset.py                # Dataset handling and generation
├── model.py                  # T5 with policy head architecture
├── metrics.py                # Evaluation metrics (accuracy, ECE, etc.)
├── train_supervised.py       # Supervised warm-start training
├── train_ppo.py             # PPO reinforcement learning training
├── main.py                   # Main training script
├── visualize.py             # Visualization utilities
├── requirements.txt          # Python dependencies
├── data/                     # Dataset directory
│   ├── processed_dataset.json  # Full dataset (QANTA CSV)
│   ├── train_dataset.json     # Training split (70%)
│   ├── val_dataset.json       # Validation split (15%)
│   └── test_dataset.json      # Test split (15%)
├── checkpoints/              # Model checkpoints
│   ├── supervised/          # Supervised model checkpoints
│   └── ppo/                 # PPO model checkpoints
├── results/                  # Evaluation results
└── logs/                     # Training logs
```

## Installation

### 1. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note:** The first run will download the T5-large model (~3GB), which may take some time.

## Quick Start

### Full Pipeline (Supervised + PPO)

Run the complete training pipeline (this replicates the milestone results):

```bash
python main.py --mode full
```

This will:
1. Generate/load 500 quiz bowl questions with multiple-choice answers
2. Train supervised baseline for 50 epochs
3. Train PPO for 250 iterations
4. Evaluate on test set and report all metrics

### Individual Training Modes

**Supervised training only:**
```bash
python main.py --mode supervised
```

**PPO training only** (requires pretrained supervised model):
```bash
python main.py --mode ppo --model_path checkpoints/supervised/best_model
```

**Evaluation only:**
```bash
python main.py --mode eval --model_path checkpoints/ppo/best_model
```

## Command-Line Arguments

```bash
python main.py --help
```

Key arguments:
- `--mode`: Training mode (`supervised`, `ppo`, `full`, or `eval`)
- `--model_path`: Path to pretrained model
- `--supervised_epochs`: Number of supervised epochs (default: 50)
- `--ppo_iterations`: Number of PPO iterations (default: 250)
- `--batch_size`: Batch size (default: 32)
- `--device`: Device (`cuda`, `mps`, or `cpu`)
- `--seed`: Random seed (default: 42)
- `--num_questions`: Dataset size (default: 500)

## Configuration

Edit `config.py` to customize:

### Model Settings
- `MODEL_NAME`: Base T5 model (`t5-small`, `t5-base`, `t5-large`)
- `POLICY_HIDDEN_DIM`: Policy head hidden dimension
- `NUM_ANSWER_CHOICES`: Number of answer choices (default: 4)

### Training Settings
- Supervised: learning rate, batch size, epochs
- PPO: learning rate, clip ratio, entropy coefficient, GAE lambda

### Reward Settings
- `REWARD_CORRECT`: Reward for correct answer (default: 1.0)
- `REWARD_TIME_PENALTY`: Time penalty coefficient (default: 0.1)

## Expected Results (Milestone Report)

After 50 PPO iterations (20% of full training):

| Metric | Supervised Baseline | PPO (50 iter) |
|--------|-------------------|---------------|
| Accuracy | 68.0% | 64.0% |
| Average Reward | +0.31 | +0.42 |
| ECE | 0.18 | 0.15 |
| Avg Buzz Position | 2.3 | 3.1 |

**Choices-Only Control:** 28% accuracy (vs 25% random)

## Project Components

### 1. POMDP Environment (`environment.py`)

Implements the quiz bowl as a Partially Observable Markov Decision Process:
- **States:** Complete questions with all clues
- **Observations:** Partial questions (clues revealed so far) + answer choices
- **Actions:** WAIT (0) or SELECT answer i (1-4)
- **Rewards:** `R_t = 1_{correct} - 0.1 × (t/T)`

```python
from environment import QuizBowlEnvironment, Question

question = Question(
    question_id="history_0001",
    clues=["Clue 1...", "Clue 2...", "Clue 3..."],
    answer_choices=["Napoleon", "Caesar", "Alexander", "Charlemagne"],
    correct_answer_idx=0,
    category="history"
)

env = QuizBowlEnvironment(question)
obs = env.reset()
obs, reward, done, info = env.step(action=0)  # WAIT
obs, reward, done, info = env.step(action=1)  # SELECT answer 0
```

### 2. Model Architecture (`model.py`)

T5-large (770M parameters) with custom policy head:

```python
from model import T5PolicyModel

model = T5PolicyModel(
    model_name="t5-large",
    num_answer_choices=4,
    policy_hidden_dim=256
)

# Get action probabilities
outputs = model(input_ids, attention_mask)
# outputs['wait_prob']: P(wait)
# outputs['action_probs']: P(wait), P(select 1), P(select 2), P(select 3), P(select 4)
# outputs['value']: State value estimate

# Sample action
actions, info = model.select_action(input_ids, attention_mask, deterministic=False)
```

### 3. Dataset (`dataset.py`)

Synthetic quiz bowl questions with carefully curated distractors:

```python
from dataset import setup_datasets

train_dataset, val_dataset, test_dataset = setup_datasets(config)
# 350 train, 75 val, 75 test questions
# Categories: History 35%, Literature 25%, Science 25%, Arts 15%
```

### 4. Metrics (`metrics.py`)

Comprehensive evaluation metrics:
- **Accuracy:** Standard classification accuracy
- **Average Reward:** Mean reward per episode
- **ECE (Expected Calibration Error):** Confidence calibration
- **Brier Score:** Probabilistic accuracy
- **Category Accuracy:** Per-category breakdown
- **Buzzing Statistics:** Position analysis

```python
from metrics import evaluate_model, evaluate_choices_only

# Full evaluation
metrics = evaluate_model(model, test_dataset, device='cuda')
metrics.print_summary()

# Choices-only control
choices_metrics = evaluate_choices_only(model, test_dataset, device='cuda')
print(f"Choices-only accuracy: {choices_metrics.compute_accuracy()}")
```

### 5. Training

**Supervised Training (`train_supervised.py`):**
- Warm-start training on complete questions
- Cross-entropy loss on answer choices
- 50 epochs, learning rate 5e-5

**PPO Training (`train_ppo.py`):**
- Policy gradient optimization with clipping
- GAE for advantage estimation
- 250 iterations, 32 episodes per iteration

## Key Features

### ✅ Novel Contributions
1. **First combination** of pyramidal questions with multiple-choice constraints
2. **Rigorous testing** via choices-only control experiment
3. **RL-based calibration** using reward shaping

### ✅ Complete Implementation
- Full POMDP environment with incremental clue revelation
- T5-large with learned policy head (772M parameters)
- Supervised warm-start + PPO fine-tuning
- Comprehensive metrics: accuracy, reward, ECE, buzzing behavior
- Choices-only control experiment

### ✅ Reproducibility
- Fixed random seeds
- Deterministic evaluation
- Saved checkpoints and training history
- Detailed logging

## Computational Requirements

### Hardware
- **Minimum:** 16GB RAM, 8GB GPU VRAM
- **Recommended:** 32GB RAM, 16GB GPU VRAM (for T5-large)
- **Alternative:** Use `t5-base` (220M params) or `t5-small` (60M params) for lower memory

### Training Time (on V100 GPU)
- **Supervised (50 epochs):** ~2-3 hours
- **PPO (250 iterations):** ~8-10 hours
- **Full pipeline:** ~10-13 hours

For CPU-only training, expect 5-10x longer training times.

## Troubleshooting

### Out of Memory
```python
# In config.py, reduce batch sizes:
SUPERVISED_BATCH_SIZE = 4  # Instead of 8
PPO_BATCH_SIZE = 16  # Instead of 32

# Or use smaller model:
MODEL_NAME = "t5-base"  # Instead of "t5-large"
```

### Slow Training
```bash
# Reduce dataset size for quick testing:
python main.py --mode full --num_questions 100 --supervised_epochs 10 --ppo_iterations 50
```

### Model Download Issues
```bash
# Pre-download model:
python -c "from transformers import T5ForConditionalGeneration; T5ForConditionalGeneration.from_pretrained('t5-large')"
```

## Visualization and Analysis

Generate plots and analysis:

```bash
python visualize.py --checkpoint_dir checkpoints/ppo
```

This creates:
- Training curves (reward, accuracy, loss)
- Calibration plot (reliability diagram)
- Buzzing behavior histogram
- Category-specific performance

## Citation

If you use this code, please cite:

```bibtex
@misc{cs234-rl-qa-2026,
  title={Reinforcement Learning for Calibrated Question Answering},
  author={CS234 Project Team},
  year={2026},
  institution={Stanford University}
}
```

## Acknowledgments

- **Primary Advisor:** Prof. Jordan Boyd-Graber (University of Maryland)
- **Secondary Mentor:** Rohan Garg
- **QANTA Project:** Quiz bowl questions and evaluation framework
- **Stanford CS234:** Reinforcement Learning course

## License

MIT License - See LICENSE file for details

## Contact

For questions or issues, please contact the CS234 project team at Stanford Center for Global and Online Education.

---

**Last Updated:** February 2026
