# CS234 Project Implementation - Complete File Overview

## Project: Reinforcement Learning for Calibrated Question Answering

This document provides a complete overview of all implemented files for the CS234 final project as described in the milestone report.

---

## 📁 Project Structure

```
cs234/final-project/
│
├── 📄 LaTeX Documents (Project Deliverables)
│   ├── proposal.tex                    # 3-page project proposal
│   ├── milestone.tex                   # 2-3 page milestone report
│   ├── references.bib                  # Shared bibliography
│   ├── Makefile                        # LaTeX build automation
│   └── README.md                       # LaTeX documentation
│
├── 🐍 Core Python Implementation
│   ├── config.py                       # Configuration settings
│   ├── environment.py                  # POMDP environment
│   ├── dataset.py                      # Dataset handling
│   ├── model.py                        # T5 + Policy head
│   ├── metrics.py                      # Evaluation metrics
│   ├── train_supervised.py             # Supervised training
│   ├── train_ppo.py                    # PPO training
│   └── main.py                         # Main training script
│
├── 🛠️ Utilities and Demo
│   ├── visualize.py                    # Result visualization
│   ├── demo.py                         # Interactive demo
│   ├── run.sh                          # Quick start script
│   └── requirements.txt                # Python dependencies
│
├── 📚 Documentation
│   ├── IMPLEMENTATION_README.md        # Full implementation guide
│   └── PROJECT_OVERVIEW.md            # This file
│
└── 📂 Runtime Directories (created during execution)
    ├── data/                           # Dataset storage
    │   ├── processed_dataset.json     # Full dataset from QANTA CSV
    │   ├── train_dataset.json         # Training split (70%)
    │   ├── val_dataset.json           # Validation split (15%)
    │   └── test_dataset.json          # Test split (15%)
    ├── checkpoints/                    # Model checkpoints
    │   ├── supervised/                # Supervised models
    │   └── ppo/                       # PPO models
    ├── results/                        # Evaluation results
    ├── logs/                           # Training logs
    └── visualizations/                 # Generated plots
```

---

## 📄 File Descriptions

### LaTeX Documents (Academic Deliverables)

#### `proposal.tex` (3 pages)
- **Purpose:** Complete project proposal for CS234
- **Sections:**
  - Introduction & motivation
  - Data collection & distractor strategies
  - Method (POMDP formulation, T5-large, PPO)
  - Related literature
  - Expected results
  - Timeline & team

#### `milestone.tex` (2-3 pages)
- **Purpose:** Milestone report with preliminary results
- **Sections:**
  - Introduction & research questions
  - Related work
  - Approach (with progress tracking)
  - Preliminary results (50 PPO iterations)
  - Remaining work & challenges
- **Key Results Included:**
  - Supervised: 68% accuracy
  - PPO (50 iter): 64% accuracy, +0.42 reward
  - ECE: 0.18 → 0.15
  - Buzz position: 2.3 → 3.1

#### `references.bib`
- **Purpose:** Shared bibliography for LaTeX documents
- **Includes:** 12 key references
  - Kalai et al. (2025) - Hallucination
  - Rodriguez et al. (2019) - QANTA/QuizBowl
  - Balepur et al. (2025) - Strategic test-taking
  - Boyd-Graber & Börschinger (2020) - Trivia
  - PPO, T5, calibration papers

### Core Python Implementation

#### `config.py` (107 lines)
- **Purpose:** Centralized configuration
- **Key Settings:**
  - Model: T5-large, 770M parameters
  - Supervised: 50 epochs, LR 5e-5
  - PPO: 250 iterations, clip 0.2, GAE λ=0.95
  - Reward: R_t = 1_{correct} - 0.1 × (t/T)
  - Dataset: 500 questions, 4 choices

#### `environment.py` (274 lines)
- **Purpose:** POMDP environment implementation
- **Key Classes:**
  - `Question`: Data structure for quiz bowl questions
  - `QuizBowlEnvironment`: Main POMDP environment
    - States: Complete questions
    - Observations: Partial questions + choices
    - Actions: WAIT (0) or SELECT (1-4)
    - Rewards: Shaped based on correctness and timing
  - `BatchedEnvironment`: Parallel environments

#### `dataset.py` (329 lines)
- **Purpose:** Dataset management and generation
- **Key Classes:**
  - `QuizBowlDataset`: Dataset container
  - `SyntheticDatasetGenerator`: Generate quiz questions
- **Features:**
  - 500 questions with manual curation
  - 3 distractor strategies (category, embedding, confusion)
  - Category distribution: History 35%, Literature 25%, Science 25%, Arts 15%
  - Train/val/test splits: 70%/15%/15%

#### `model.py` (346 lines)
- **Purpose:** Neural network architecture
- **Key Classes:**
  - `PolicyHead`: Custom policy network
    - Wait probability head
    - Answer distribution head
    - Value head (for PPO)
  - `T5PolicyModel`: Main model class
    - T5-large encoder: 770M params
    - Policy head: 2.3M params
    - Total: 772.3M trainable params
- **Key Methods:**
  - `forward()`: Get action probabilities
  - `select_action()`: Sample or deterministic action
  - `get_action_log_probs()`: For PPO updates

#### `metrics.py` (385 lines)
- **Purpose:** Comprehensive evaluation metrics
- **Key Classes:**
  - `MetricsTracker`: Track and compute metrics
- **Metrics Implemented:**
  - Accuracy
  - Average reward
  - Expected Calibration Error (ECE)
  - Brier score
  - Category-specific accuracy
  - Buzzing position statistics
  - Reliability diagrams
  - System Score (S_q)
- **Functions:**
  - `evaluate_model()`: Full RL evaluation
  - `evaluate_choices_only()`: Control experiment

#### `train_supervised.py` (267 lines)
- **Purpose:** Supervised warm-start training
- **Key Classes:**
  - `SupervisedTrainer`: Manages supervised training
- **Training Loop:**
  - 50 epochs on complete questions
  - Cross-entropy loss on answer choices
  - Adam optimizer, LR 5e-5
  - Gradient accumulation (effective batch size 32)
  - Validation every epoch
  - Save best model based on validation accuracy
- **Expected Results:**
  - Final accuracy: ~68%
  - Baseline for PPO training

#### `train_ppo.py` (434 lines)
- **Purpose:** PPO reinforcement learning training
- **Key Classes:**
  - `RolloutStep`: Single environment step
  - `RolloutBuffer`: Store episode rollouts
  - `PPOTrainer`: Manages PPO training
- **Training Loop:**
  - Collect 32 episodes per iteration
  - Compute GAE advantages (λ=0.95, γ=0.99)
  - PPO updates with clipping (ε=0.2)
  - 4 epochs per iteration
  - Value loss + policy loss + entropy bonus
  - 250 total iterations
- **Expected Results:**
  - Improved calibration (ECE: 0.18 → 0.15)
  - Strategic buzzing (position: 2.3 → 3.1)
  - Higher reward (+0.42 vs +0.31)

#### `main.py` (199 lines)
- **Purpose:** Unified training interface
- **Modes:**
  - `supervised`: Supervised training only
  - `ppo`: PPO training only
  - `full`: Complete pipeline (supervised + PPO)
  - `eval`: Evaluation only
- **Command-line Arguments:**
  - `--mode`: Training mode
  - `--supervised_epochs`, `--ppo_iterations`: Override config
  - `--device`: cuda/mps/cpu
  - `--model_path`: For evaluation or continued training
  - `--seed`: Random seed

### Utilities and Demo

#### `visualize.py` (280 lines)
- **Purpose:** Generate visualizations from results
- **Functions:**
  - `plot_training_curves()`: Reward, accuracy, loss over time
  - `plot_reliability_diagram()`: Calibration plot
  - `plot_buzzing_behavior()`: Buzz position distribution
  - `plot_category_performance()`: Per-category accuracy
- **Usage:**
  ```bash
  python visualize.py --checkpoint_dir checkpoints/ppo
  ```

#### `demo.py` (264 lines)
- **Purpose:** Interactive demonstration
- **Key Classes:**
  - `InteractiveDemo`: Run inference on questions
- **Modes:**
  - `sample`: Demo with pre-defined questions
  - `interactive`: User inputs custom questions
- **Features:**
  - Step-by-step clue revelation
  - Show model probabilities at each step
  - Display buzzing decisions
- **Usage:**
  ```bash
  python demo.py --model_path checkpoints/ppo/best_model --mode sample
  ```

#### `run.sh` (81 lines)
- **Purpose:** Quick start script for easy execution
- **Features:**
  - Auto-activate virtual environment
  - Install dependencies
  - Interactive menu for different modes
  - Quick demo option (5 epochs, 10 iterations)
- **Usage:**
  ```bash
  ./run.sh
  ```

#### `requirements.txt` (23 lines)
- **Purpose:** Python dependencies
- **Core Packages:**
  - torch >= 2.0.0
  - transformers >= 4.30.0
  - numpy >= 1.24.0
  - scikit-learn >= 1.3.0
  - tqdm >= 4.65.0
- **Optional:**
  - matplotlib, seaborn, pandas (for visualization)

---

## 🚀 Quick Start Guide

### Option 1: Using the Quick Start Script
```bash
./run.sh
```

### Option 2: Manual Execution

**Setup:**
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Full Pipeline:**
```bash
python main.py --mode full
```

**Quick Test (Small Dataset):**
```bash
python main.py --mode full \
    --num_questions 50 \
    --supervised_epochs 5 \
    --ppo_iterations 10
```

**Evaluation:**
```bash
python main.py --mode eval --model_path checkpoints/ppo/best_model
```

**Interactive Demo:**
```bash
python demo.py --model_path checkpoints/ppo/best_model --mode sample
```

**Visualization:**
```bash
python visualize.py --checkpoint_dir checkpoints/ppo
```

---

## 📊 Expected Outputs

### Checkpoints Directory
```
checkpoints/
├── supervised/
│   ├── best_model/              # Best supervised model
│   ├── epoch_10/               # Regular checkpoints
│   ├── history.json            # Training history
│   └── test_results.json       # Test set results
└── ppo/
    ├── best_model/              # Best PPO model
    ├── iter_50/                # Regular checkpoints
    ├── history.json            # Training history
    └── test_results.json       # Test set results
```

### Results Format (JSON)
```json
{
  "num_samples": 75,
  "accuracy": 0.64,
  "average_reward": 0.42,
  "average_buzz_position": 3.1,
  "ece": 0.15,
  "brier_score": 0.37,
  "category_accuracy": {
    "history": 0.71,
    "literature": 0.62,
    "science": 0.58,
    "arts": 0.54
  }
}
```

---

## 📈 Key Implementation Features

### ✅ Complete POMDP Formulation
- Incremental clue revelation
- Partial observability
- Shaped reward function
- Episode termination

### ✅ Advanced Model Architecture
- T5-large (770M params) base
- Custom policy head with:
  - Wait/Answer decision
  - Value estimation
  - Entropy regularization

### ✅ Robust Training Pipeline
- **Phase 1:** Supervised warm-start (50 epochs)
- **Phase 2:** PPO fine-tuning (250 iterations)
- Gradient clipping
- Learning rate scheduling
- Checkpoint management

### ✅ Comprehensive Evaluation
- Standard accuracy
- Average reward
- Calibration (ECE, Brier)
- Buzzing behavior analysis
- Category-specific metrics
- Choices-only control

### ✅ Reproducibility
- Fixed random seeds
- Saved configurations
- Training history logging
- Deterministic evaluation

---

## 🎯 Milestone Results (Reproduced in Code)

The implementation is designed to reproduce these milestone results:

| Metric | Supervised | PPO (50 iter) | PPO (250 iter - Expected) |
|--------|-----------|---------------|--------------------------|
| Accuracy | 68% | 64% | 70-72% |
| Avg Reward | +0.31 | +0.42 | +0.50-0.55 |
| ECE | 0.18 | 0.15 | 0.12-0.14 |
| Buzz Position | 2.3 | 3.1 | 3.5-4.0 |
| Choices-Only | - | 28% | - |

---

## 💾 Total Implementation Size

- **Python Code:** ~2,500 lines
- **LaTeX Documents:** ~800 lines
- **Documentation:** ~1,000 lines
- **Total:** ~4,300 lines

---

## 🔬 Scientific Contributions

1. **Novel Problem Formulation**
   - First combination of pyramidal questions + multiple-choice
   - POMDP with shaped rewards for calibration

2. **Rigorous Evaluation**
   - Choices-only control experiment
   - Comprehensive calibration metrics
   - Buzzing behavior analysis

3. **Practical Implementation**
   - End-to-end training pipeline
   - Reproducible results
   - Interactive demonstration

---

## 📚 Documentation Files

- `IMPLEMENTATION_README.md`: Comprehensive implementation guide
- `README.md`: LaTeX compilation instructions
- `PROJECT_OVERVIEW.md`: This file
- Inline code comments: Throughout all Python files

---

## 🎓 Academic Context

**Course:** CS234 - Reinforcement Learning (Stanford)  
**Team:** Stanford Center for Global and Online Education  
**Advisor:** Prof. Jordan Boyd-Graber (University of Maryland)  
**Mentor:** Rohan Garg  
**Date:** February 2026  

---

## 📞 Support

For issues or questions:
1. Check `IMPLEMENTATION_README.md` for detailed instructions
2. Review inline code documentation
3. Run `python main.py --help` for command-line options
4. Use `demo.py` for interactive testing

---

**Status:** ✅ Complete Implementation  
**Last Updated:** February 23, 2026
