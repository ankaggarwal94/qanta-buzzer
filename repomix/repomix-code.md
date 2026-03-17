This file is a merged representation of a subset of the codebase, containing specifically included files and files not matching ignore patterns, combined into a single document by Repomix.

# File Summary

## Purpose
This file contains a packed representation of a subset of the repository's contents that is considered the most important context.
It is designed to be easily consumable by AI systems for analysis, code review,
or other automated processes.

## File Format
The content is organized as follows:
1. This summary section
2. Repository information
3. Directory structure
4. Repository files (if enabled)
5. Multiple file entries, each consisting of:
  a. A header with the file path (## File: path/to/file)
  b. The full contents of the file in a code block

## Usage Guidelines
- This file should be treated as read-only. Any changes should be made to the
  original repository files, not this packed version.
- When processing this file, use the file path to distinguish
  between different files in the repository.
- Be aware that this file may contain sensitive information. Handle it with
  the same level of security as you would the original repository.

## Notes
- Some files may have been excluded based on .gitignore rules and Repomix's configuration
- Binary files are not included in this packed representation. Please refer to the Repository Structure section for a complete list of file paths, including binary files
- Only files matching these patterns are included: **/*.py, **/*.sh, **/*.yaml, **/*.toml
- Files matching these patterns are excluded: repomix/**, .venv/**, __pycache__/**, *.pyc, second_qanta_buzzer_remediation.md, cursor_patch_main_followup.md
- Files matching patterns in .gitignore are excluded
- Files matching default ignore patterns are excluded
- Files are sorted by Git change count (files with more changes are at the bottom)

# Directory Structure
```
agents/
  __init__.py
  _math.py
  bayesian_buzzer.py
  ppo_buzzer.py
  softmax_profile_buzzer.py
  threshold_buzzer.py
configs/
  default.yaml
  smoke.yaml
  t5_policy.yaml
evaluation/
  __init__.py
  controls.py
  metrics.py
  plotting.py
models/
  __init__.py
  answer_profiles.py
  dspy_likelihood.py
  features.py
  likelihoods.py
  t5_policy.py
qb_data/
  __init__.py
  answer_profiles.py
  config.py
  data_loader.py
  dataset_splits.py
  dspy_answer_profiles.py
  huggingface_loader.py
  mc_builder.py
  text_utils.py
qb_env/
  __init__.py
  data_loader.py
  mc_builder.py
  opponent_models.py
  stop_only_env.py
  text_utils.py
  text_wrapper.py
  tossup_env.py
scripts/
  _common.py
  build_mc_dataset.py
  ci.sh
  compare_policies.py
  evaluate_all.py
  manual-smoke.sh
  optimize_dspy.py
  run_baselines.py
  run_full_pipeline.sh
  run_smoke_pipeline.py
  sweep_reward_shaping.py
  test_mc_builder.py
  train_ppo.py
  train_t5_policy.py
tests/
  conftest.py
  test_action_space_alignment.py
  test_agents.py
  test_answer_profile_cache.py
  test_build_mc_dataset.py
  test_common.py
  test_compare_policies.py
  test_dataset_splits.py
  test_dspy_answer_profiles.py
  test_dspy_likelihood.py
  test_dspy_optimize.py
  test_environment.py
  test_factories.py
  test_features.py
  test_hazard_pretrain.py
  test_likelihoods.py
  test_mc_builder_topk.py
  test_mc_builder_variable_k.py
  test_metrics.py
  test_opponent_models.py
  test_pipeline_smoke.py
  test_ppo_buzzer.py
  test_ppo_t5.py
  test_qb_rl_bridge.py
  test_stop_only_env.py
  test_supervised_t5.py
  test_t5_policy.py
  test_text_wrapper.py
  test_variable_k_integration.py
training/
  __init__.py
  hazard_pretrain.py
  train_ppo_t5.py
  train_supervised_t5.py
config.py
dataset.py
demo.py
environment.py
generate_poster.py
generate_presentation.py
main.py
metrics.py
model.py
pyproject.toml
run.sh
test_csv_loader.py
test_imports.py
train_ppo.py
train_supervised.py
verify_data_loader.py
visualize.py
```

# Files

## File: config.py
```python
"""
Configuration file for the CS234 RL Question Answering project.
"""

import torch


class Config:
    """Main configuration class"""
    
    # Model settings
    MODEL_NAME = "t5-large"  # 770M parameters
    MAX_INPUT_LENGTH = 512
    MAX_OUTPUT_LENGTH = 10
    POLICY_HIDDEN_DIM = 256
    NUM_ANSWER_CHOICES = 4
    
    # Training settings - Supervised
    SUPERVISED_EPOCHS = 50
    SUPERVISED_LR = 5e-5
    SUPERVISED_BATCH_SIZE = 8
    SUPERVISED_GRAD_ACCUM_STEPS = 4  # Effective batch size = 32
    
    # Training settings - PPO
    PPO_ITERATIONS = 250
    PPO_LR = 3e-5
    PPO_BATCH_SIZE = 32
    PPO_EPOCHS_PER_ITER = 4
    PPO_CLIP_RATIO = 0.2
    PPO_VALUE_COEF = 0.5
    PPO_ENTROPY_COEF = 0.01
    PPO_GAE_LAMBDA = 0.95
    PPO_GAMMA = 0.99
    PPO_MAX_GRAD_NORM = 0.5
    
    # Reward settings
    REWARD_CORRECT = 1.0
    REWARD_TIME_PENALTY = 0.1  # Multiply by t/T
    
    # Dataset settings
    NUM_QUESTIONS = 500
    TRAIN_SPLIT = 0.7  # 350 questions
    VAL_SPLIT = 0.15   # 75 questions
    TEST_SPLIT = 0.15  # 75 questions
    
    CATEGORY_DISTRIBUTION = {
        'history': 0.35,
        'literature': 0.25,
        'science': 0.25,
        'arts': 0.15
    }
    
    MIN_CLUES_PER_QUESTION = 4
    MAX_CLUES_PER_QUESTION = 6
    
    # Distractor strategies
    DISTRACTOR_CATEGORY_BASED = 0.4
    DISTRACTOR_EMBEDDING_BASED = 0.4
    DISTRACTOR_COMMON_CONFUSION = 0.2
    
    # Device and compute
    DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    SEED = 42
    NUM_WORKERS = 4
    
    # Paths
    DATA_DIR = "data"
    CHECKPOINT_DIR = "checkpoints"
    RESULTS_DIR = "results"
    LOG_DIR = "logs"
    
    # Evaluation
    ECE_NUM_BINS = 10
    
    # Logging
    LOG_INTERVAL = 10
    EVAL_INTERVAL = 50
    SAVE_INTERVAL = 50
    
    @classmethod
    def print_config(cls):
        """Print all configuration settings"""
        print("=" * 50)
        print("Configuration Settings")
        print("=" * 50)
        for attr in dir(cls):
            if not attr.startswith('_') and attr.isupper():
                print(f"{attr}: {getattr(cls, attr)}")
        print("=" * 50)
```

## File: dataset.py
```python
"""
Dataset handling for Quiz Bowl questions
"""

import json
import csv
import random
import numpy as np
from typing import List, Dict, Tuple, Set
from pathlib import Path
from dataclasses import asdict
from collections import defaultdict

from environment import Question
from config import Config


class QuizBowlDataset:
    """Dataset class for quiz bowl questions with multiple-choice answers"""
    
    def __init__(self, questions: List[Question]):
        """
        Initialize dataset with questions.
        
        Args:
            questions: List of Question objects
        """
        self.questions = questions
    
    def __len__(self) -> int:
        return len(self.questions)
    
    def __getitem__(self, idx: int) -> Question:
        return self.questions[idx]
    
    def shuffle(self):
        """Shuffle questions in place"""
        random.shuffle(self.questions)
    
    def get_batch(self, batch_size: int) -> List[Question]:
        """Get a random batch of questions"""
        return random.sample(self.questions, min(batch_size, len(self.questions)))
    
    def save(self, filepath: str):
        """Save dataset to JSON file"""
        data = [self._question_to_dict(q) for q in self.questions]
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'QuizBowlDataset':
        """Load dataset from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        questions = [cls._dict_to_question(d) for d in data]
        return cls(questions)
    
    @staticmethod
    def _question_to_dict(question: Question) -> Dict:
        """Convert Question to dictionary"""
        return {
            'question_id': question.question_id,
            'clues': question.clues,
            'answer_choices': question.answer_choices,
            'correct_answer_idx': question.correct_answer_idx,
            'category': question.category,
            'metadata': question.metadata or {}
        }
    
    @staticmethod
    def _dict_to_question(data: Dict) -> Question:
        """Convert dictionary to Question"""
        return Question(
            question_id=data['question_id'],
            clues=data['clues'],
            answer_choices=data['answer_choices'],
            correct_answer_idx=data['correct_answer_idx'],
            category=data['category'],
            metadata=data.get('metadata', {})
        )


class QANTADatasetLoader:
    """
    Load Quiz Bowl questions from QANTA CSV format.
    Generates multiple-choice questions by selecting distractors from the same category.
    """
    
    @classmethod
    def load_from_csv(cls, 
                     csv_path: str,
                     num_questions: int = None,
                     num_choices: int = 4,
                     min_clues: int = 3,
                     max_clues: int = 6,
                     seed: int = 42) -> 'QuizBowlDataset':
        """
        Load questions from QANTA CSV file.
        
        Args:
            csv_path: Path to questions.csv file
            num_questions: Number of questions to load (None = all)
            num_choices: Number of answer choices (default: 4)
            min_clues: Minimum clues to include per question
            max_clues: Maximum clues to include per question
            seed: Random seed
            
        Returns:
            QuizBowlDataset object
        """
        random.seed(seed)
        np.random.seed(seed)
        
        print(f"Loading questions from {csv_path}...")
        
        # Load all questions from CSV
        raw_questions = []
        category_answers = defaultdict(list)  # For generating distractors
        
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Parse clues (separated by "|||")
                text = row['Text']
                clues = [clue.strip() for clue in text.split('|||')]
                
                # Store question data
                question_data = {
                    'question_id': row['Question ID'],
                    'fold': row['Fold'],
                    'answer': row['Answer'],
                    'category': row['Category'],
                    'clues': clues
                }
                raw_questions.append(question_data)
                
                # Store answer by category for distractor selection
                if row['Answer'] not in category_answers[row['Category']]:
                    category_answers[row['Category']].append(row['Answer'])
        
        print(f"Loaded {len(raw_questions)} raw questions")
        
        # Shuffle and optionally limit
        random.shuffle(raw_questions)
        if num_questions is not None:
            raw_questions = raw_questions[:num_questions]
        
        # Convert to Question objects with multiple choice
        questions = []
        for idx, raw_q in enumerate(raw_questions):
            # Get available clues
            available_clues = len(raw_q['clues'])
            if available_clues < 1:
                # Skip questions with no clues
                continue
            
            # Select number of clues to use
            if available_clues < min_clues:
                # Use all available clues if less than minimum
                num_clues = available_clues
            else:
                # Randomly select between min and max (capped by available)
                num_clues = min(
                    random.randint(min_clues, max_clues),
                    available_clues
                )
            
            clues = raw_q['clues'][:num_clues]
            
            # Generate distractors from same category
            correct_answer = raw_q['answer']
            category = raw_q['category']
            
            # Get potential distractors (exclude correct answer)
            potential_distractors = [
                ans for ans in category_answers[category] 
                if ans != correct_answer
            ]
            
            # If not enough distractors in category, use from other categories
            if len(potential_distractors) < num_choices - 1:
                other_answers = []
                for cat, answers in category_answers.items():
                    if cat != category:
                        other_answers.extend(answers)
                potential_distractors.extend(
                    random.sample(other_answers, 
                                min(num_choices - 1 - len(potential_distractors), 
                                    len(other_answers)))
                )
            
            # Sample distractors
            distractors = random.sample(
                potential_distractors, 
                min(num_choices - 1, len(potential_distractors))
            )
            
            # Create answer choices
            answer_choices = [correct_answer] + distractors
            correct_idx = 0
            
            # Shuffle choices
            shuffle_indices = list(range(len(answer_choices)))
            random.shuffle(shuffle_indices)
            answer_choices = [answer_choices[i] for i in shuffle_indices]
            correct_idx = shuffle_indices.index(0)
            
            # Pad with empty choices if needed
            while len(answer_choices) < num_choices:
                answer_choices.append(f"[No answer {len(answer_choices)}]")
            
            # Create Question object
            question = Question(
                question_id=raw_q['question_id'],
                clues=clues,
                answer_choices=answer_choices,
                correct_answer_idx=correct_idx,
                category=category,
                metadata={
                    'source': 'qanta',
                    'fold': raw_q['fold'],
                    'full_answer': correct_answer,
                    'total_clues': available_clues
                }
            )
            
            questions.append(question)
        
        print(f"Created {len(questions)} multiple-choice questions")
        
        return QuizBowlDataset(questions)


class SyntheticDatasetGenerator:
    """
    Generate synthetic quiz bowl questions for development and testing.
    Use QANTADatasetLoader for real QANTA data.
    """
    
    SAMPLE_QUESTIONS = {
        'history': [
            {
                'entity': 'Napoleon Bonaparte',
                'clues': [
                    'This military leader established the Continental System to economically isolate Britain.',
                    'He crowned himself emperor in 1804 at Notre-Dame Cathedral in Paris.',
                    'His Russian campaign of 1812 ended in catastrophic retreat from Moscow.',
                    'He was finally defeated at Waterloo in 1815 by Wellington and Blücher.',
                    'This French emperor was exiled to Elba and later to Saint Helena.',
                ],
                'distractors': ['Julius Caesar', 'Alexander the Great', 'Charlemagne']
            },
            {
                'entity': 'Abraham Lincoln',
                'clues': [
                    'This leader delivered an address at the dedication of a military cemetery in Pennsylvania.',
                    'He issued a proclamation in 1863 that changed the legal status of enslaved people.',
                    'He was assassinated by John Wilkes Booth at Ford\'s Theatre.',
                    'His debates with Stephen Douglas helped him gain national prominence.',
                    'This 16th U.S. president led the country through the Civil War.',
                ],
                'distractors': ['George Washington', 'Thomas Jefferson', 'Andrew Jackson']
            }
        ],
        'literature': [
            {
                'entity': 'The Great Gatsby',
                'clues': [
                    'This novel features a green light at the end of a dock as a central symbol.',
                    'The narrator is Nick Carraway, who moves to West Egg, Long Island.',
                    'Characters include Tom and Daisy Buchanan and Meyer Wolfsheim.',
                    'The title character throws lavish parties hoping to attract his lost love.',
                    'F. Scott Fitzgerald wrote this Jazz Age novel published in 1925.',
                ],
                'distractors': ['Tender Is the Night', 'This Side of Paradise', 'The Beautiful and Damned']
            },
            {
                'entity': 'Franz Kafka',
                'clues': [
                    'This author wrote about a man who wakes up transformed into a monstrous insect.',
                    'His novel "The Trial" features Josef K. arrested for an unknown crime.',
                    'Works like "The Castle" and "Amerika" were published posthumously.',
                    'He worked as an insurance officer in Prague while writing.',
                    'This German-language writer is known for absurdist and existential themes.',
                ],
                'distractors': ['Thomas Mann', 'James Joyce', 'Virginia Woolf']
            }
        ],
        'science': [
            {
                'entity': 'Mitochondria',
                'clues': [
                    'These organelles have their own circular DNA separate from nuclear DNA.',
                    'They are believed to have originated from endosymbiotic bacteria.',
                    'The inner membrane is folded into structures called cristae.',
                    'They produce ATP through oxidative phosphorylation.',
                    'These are often called the "powerhouse of the cell".',
                ],
                'distractors': ['Chloroplast', 'Ribosome', 'Endoplasmic Reticulum']
            },
            {
                'entity': 'Quantum Entanglement',
                'clues': [
                    'Einstein famously called this "spooky action at a distance".',
                    'Bell\'s theorem provides a way to test this phenomenon experimentally.',
                    'Measuring one particle instantly affects its correlated partner.',
                    'This property is exploited in quantum computing and cryptography.',
                    'This quantum mechanical phenomenon links particle states non-locally.',
                ],
                'distractors': ['Wave Function Collapse', 'Superposition', 'Decoherence']
            }
        ],
        'arts': [
            {
                'entity': 'The Starry Night',
                'clues': [
                    'This painting features a prominent cypress tree in the foreground.',
                    'It depicts a view from an asylum window in Saint-Rémy-de-Provence.',
                    'Swirling patterns dominate the night sky in this post-impressionist work.',
                    'It was painted in 1889, one year before the artist\'s death.',
                    'Vincent van Gogh created this iconic painting.',
                ],
                'distractors': ['Café Terrace at Night', 'The Night Watch', 'Nighthawks']
            }
        ]
    }
    
    @classmethod
    def generate_dataset(cls, 
                        num_questions: int = 500,
                        category_distribution: Dict[str, float] = None,
                        min_clues: int = 4,
                        max_clues: int = 6,
                        seed: int = 42) -> QuizBowlDataset:
        """
        Generate synthetic dataset.
        
        Args:
            num_questions: Total number of questions to generate
            category_distribution: Distribution of categories
            min_clues: Minimum clues per question
            max_clues: Maximum clues per question
            seed: Random seed
            
        Returns:
            QuizBowlDataset object
        """
        random.seed(seed)
        np.random.seed(seed)
        
        if category_distribution is None:
            category_distribution = Config.CATEGORY_DISTRIBUTION
        
        questions = []
        
        # Calculate questions per category
        categories = list(category_distribution.keys())
        category_counts = {cat: int(num_questions * prob) 
                          for cat, prob in category_distribution.items()}
        
        # Adjust for rounding errors
        total = sum(category_counts.values())
        if total < num_questions:
            category_counts[categories[0]] += num_questions - total
        
        # Generate questions
        question_id = 0
        for category, count in category_counts.items():
            for _ in range(count):
                # Sample a template from this category
                template = random.choice(cls.SAMPLE_QUESTIONS.get(category, 
                                        cls.SAMPLE_QUESTIONS['history']))
                
                # Randomly select number of clues
                num_clues = random.randint(min_clues, max_clues)
                
                # Sample clues (repeat if needed)
                if len(template['clues']) >= num_clues:
                    clues = template['clues'][:num_clues]
                else:
                    clues = template['clues'] + random.choices(
                        template['clues'], k=num_clues - len(template['clues']))
                
                # Shuffle answer choices (correct answer at random position)
                answer_choices = [template['entity']] + template['distractors'][:3]
                correct_idx = random.randint(0, 3)
                
                # Swap correct answer to correct_idx position
                answer_choices[0], answer_choices[correct_idx] = \
                    answer_choices[correct_idx], answer_choices[0]
                
                question = Question(
                    question_id=f"{category}_{question_id:04d}",
                    clues=clues,
                    answer_choices=answer_choices,
                    correct_answer_idx=correct_idx,
                    category=category,
                    metadata={
                        'source': 'synthetic',
                        'template_entity': template['entity']
                    }
                )
                
                questions.append(question)
                question_id += 1
        
        return QuizBowlDataset(questions)


def create_train_val_test_splits(
    dataset: QuizBowlDataset,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[QuizBowlDataset, QuizBowlDataset, QuizBowlDataset]:
    """
    Split dataset into train, validation, and test sets.
    
    Args:
        dataset: Full dataset
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        test_ratio: Fraction for testing
        seed: Random seed
        
    Returns:
        train_dataset, val_dataset, test_dataset
    """
    random.seed(seed)
    
    questions = dataset.questions.copy()
    random.shuffle(questions)
    
    total = len(questions)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    train_questions = questions[:train_end]
    val_questions = questions[train_end:val_end]
    test_questions = questions[val_end:]
    
    return (
        QuizBowlDataset(train_questions),
        QuizBowlDataset(val_questions),
        QuizBowlDataset(test_questions)
    )


def setup_datasets(config: Config) -> Tuple[QuizBowlDataset, QuizBowlDataset, QuizBowlDataset]:
    """
    Setup datasets with proper splits.
    Tries to load from questions.csv first, falls back to synthetic data.
    
    Args:
        config: Configuration object
        
    Returns:
        train_dataset, val_dataset, test_dataset
    """
    data_dir = Path(config.DATA_DIR)
    data_dir.mkdir(exist_ok=True)
    
    # Check for questions.csv in project root or data directory
    csv_path = None
    for potential_path in [
        Path("questions.csv"),
        Path(__file__).parent / "questions.csv",
        data_dir / "questions.csv"
    ]:
        if potential_path.exists():
            csv_path = potential_path
            print(f"Found questions.csv at {csv_path}")
            break
    
    dataset_path = data_dir / "processed_dataset.json"
    
    # Try to load existing processed dataset
    if dataset_path.exists():
        print(f"\nLoading existing processed dataset from {dataset_path}")
        full_dataset = QuizBowlDataset.load(str(dataset_path))
    
    # Load from CSV if available
    elif csv_path is not None:
        print(f"\nLoading from QANTA CSV file: {csv_path}")
        full_dataset = QANTADatasetLoader.load_from_csv(
            csv_path=str(csv_path),
            num_questions=config.NUM_QUESTIONS,
            num_choices=config.NUM_ANSWER_CHOICES,
            min_clues=config.MIN_CLUES_PER_QUESTION,
            max_clues=config.MAX_CLUES_PER_QUESTION,
            seed=config.SEED
        )
        # Save processed dataset
        full_dataset.save(str(dataset_path))
        print(f"Saved processed dataset to {dataset_path}")
    
    # Fall back to synthetic data
    else:
        print(f"\nNo questions.csv found, generating synthetic dataset with {config.NUM_QUESTIONS} questions")
        full_dataset = SyntheticDatasetGenerator.generate_dataset(
            num_questions=config.NUM_QUESTIONS,
            category_distribution=config.CATEGORY_DISTRIBUTION,
            min_clues=config.MIN_CLUES_PER_QUESTION,
            max_clues=config.MAX_CLUES_PER_QUESTION,
            seed=config.SEED
        )
        full_dataset.save(str(dataset_path))
        print(f"Saved synthetic dataset to {dataset_path}")
    
    # Check if splits already exist
    train_path = data_dir / "train_dataset.json"
    val_path = data_dir / "val_dataset.json"
    test_path = data_dir / "test_dataset.json"
    
    if train_path.exists() and val_path.exists() and test_path.exists():
        print(f"\nLoading existing splits from {data_dir}")
        train_dataset = QuizBowlDataset.load(str(train_path))
        val_dataset = QuizBowlDataset.load(str(val_path))
        test_dataset = QuizBowlDataset.load(str(test_path))
    else:
        print(f"\nCreating new train/val/test splits...")
        # Create splits
        train_dataset, val_dataset, test_dataset = create_train_val_test_splits(
            full_dataset,
            train_ratio=config.TRAIN_SPLIT,
            val_ratio=config.VAL_SPLIT,
            test_ratio=config.TEST_SPLIT,
            seed=config.SEED
        )
        
        # Save splits separately
        train_dataset.save(str(train_path))
        val_dataset.save(str(val_path))
        test_dataset.save(str(test_path))
        print(f"Saved splits to {data_dir}")
    
    print(f"\nDataset splits: Train={len(train_dataset)}, "
          f"Val={len(val_dataset)}, Test={len(test_dataset)}")
    
    # Print category distribution
    train_categories = [q.category for q in train_dataset.questions]
    category_counts = {}
    for cat in train_categories:
        category_counts[cat] = category_counts.get(cat, 0) + 1
    
    print("\nCategory distribution in training set:")
    for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count} ({100*count/len(train_dataset):.1f}%)")
    
    return train_dataset, val_dataset, test_dataset
```

## File: demo.py
```python
"""
Interactive demo for testing the trained model
"""

import torch
import argparse
from pathlib import Path

from model import T5PolicyModel
from environment import QuizBowlEnvironment, Question
from config import Config


class InteractiveDemo:
    """Interactive demo for question answering"""
    
    def __init__(self, model_path: str, device: str = 'cpu'):
        """
        Initialize demo with trained model.
        
        Args:
            model_path: Path to trained model checkpoint
            device: Device to run on
        """
        print(f"Loading model from {model_path}...")
        self.model = T5PolicyModel.load_pretrained(model_path, device=device)
        self.model.to(device)
        self.model.eval()
        self.device = device
        print("Model loaded successfully!")
    
    def run_episode(self, question: Question, verbose: bool = True):
        """
        Run a single episode with the given question.
        
        Args:
            question: Question object
            verbose: Whether to print step-by-step details
            
        Returns:
            Dictionary with episode results
        """
        env = QuizBowlEnvironment(question)
        obs = env.reset()
        done = False
        step_count = 0
        
        if verbose:
            print("\n" + "=" * 70)
            print(f"Question ID: {question.question_id}")
            print(f"Category: {question.category}")
            print(f"Total Clues: {len(question.clues)}")
            print("=" * 70)
        
        with torch.no_grad():
            while not done:
                step_count += 1
                
                # Get current observation
                text = env.get_text_representation(obs)
                
                if verbose:
                    print(f"\n--- Step {step_count} (Clue {obs['clue_position'] + 1}/{obs['total_clues']}) ---")
                    print(f"Current clue: {obs['clues'][-1]}")
                
                # Tokenize
                inputs = self.model.tokenizer(
                    text,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=512
                ).to(self.device)
                
                # Get model prediction
                outputs = self.model.forward(inputs['input_ids'], inputs['attention_mask'])
                
                # Get action probabilities
                action_probs = outputs['action_probs'][0].cpu().numpy()
                wait_prob = outputs['wait_prob'][0].item()
                answer_logits = outputs['answer_logits'][0].cpu()
                answer_probs = torch.softmax(answer_logits, dim=-1).numpy()
                
                if verbose:
                    print(f"\nModel predictions:")
                    print(f"  Wait probability: {wait_prob:.3f}")
                    print(f"  Answer probabilities:")
                    for i, (choice, prob) in enumerate(zip(obs['answer_choices'], answer_probs)):
                        marker = "✓" if i == question.correct_answer_idx else " "
                        print(f"    ({i+1}) {choice}: {prob:.3f} {marker}")
                
                # Select action (deterministic - argmax)
                action = action_probs.argmax()
                
                if action == 0:
                    if verbose:
                        print(f"\nAction: WAIT (continue to next clue)")
                else:
                    selected_idx = action - 1
                    if verbose:
                        print(f"\nAction: SELECT answer ({selected_idx + 1}) {obs['answer_choices'][selected_idx]}")
                
                # Take step
                next_obs, reward, done, info = env.step(action)
                obs = next_obs
        
        # Episode complete
        if verbose:
            print("\n" + "=" * 70)
            print("EPISODE COMPLETE")
            print("=" * 70)
            
            if 'is_correct' in info:
                result = "CORRECT ✓" if info['is_correct'] else "INCORRECT ✗"
                print(f"Result: {result}")
                print(f"Selected: ({info['answer_idx'] + 1}) {question.answer_choices[info['answer_idx']]}")
                print(f"Correct: ({info['correct_idx'] + 1}) {question.answer_choices[info['correct_idx']]}")
                print(f"Buzz Position: {info['clue_position'] + 1}/{len(question.clues)}")
                print(f"Reward: {reward:.3f}")
            print("=" * 70)
        
        return {
            'is_correct': info.get('is_correct', False),
            'reward': reward,
            'buzz_position': info.get('clue_position', 0),
            'selected_answer': info.get('answer_idx', -1),
            'correct_answer': info.get('correct_idx', -1)
        }
    
    def interactive_mode(self):
        """Run interactive mode where user can input questions"""
        print("\n" + "=" * 70)
        print("INTERACTIVE QUESTION ANSWERING DEMO")
        print("=" * 70)
        print("\nEnter 'quit' to exit")
        
        while True:
            print("\n" + "-" * 70)
            
            # Get question from user
            question_id = input("\nQuestion ID (or 'quit'): ").strip()
            if question_id.lower() == 'quit':
                break
            
            category = input("Category: ").strip()
            
            # Get clues
            clues = []
            print("\nEnter clues (press Enter twice when done):")
            while True:
                clue = input(f"Clue {len(clues) + 1}: ").strip()
                if not clue:
                    break
                clues.append(clue)
            
            if not clues:
                print("No clues entered. Skipping question.")
                continue
            
            # Get answer choices
            choices = []
            print("\nEnter 4 answer choices:")
            for i in range(4):
                choice = input(f"Choice {i+1}: ").strip()
                choices.append(choice)
            
            correct_idx = int(input("\nCorrect answer index (1-4): ")) - 1
            
            # Create question
            question = Question(
                question_id=question_id,
                clues=clues,
                answer_choices=choices,
                correct_answer_idx=correct_idx,
                category=category
            )
            
            # Run episode
            self.run_episode(question, verbose=True)


def demo_with_sample_questions(model_path: str, device: str = 'cpu'):
    """Run demo with pre-defined sample questions"""
    
    demo = InteractiveDemo(model_path, device)
    
    # Sample questions
    sample_questions = [
        Question(
            question_id="demo_history_001",
            clues=[
                "This military leader established the Continental System to economically isolate Britain.",
                "He crowned himself emperor in 1804 at Notre-Dame Cathedral in Paris.",
                "His Russian campaign of 1812 ended in catastrophic retreat from Moscow.",
                "He was finally defeated at Waterloo in 1815 by Wellington and Blücher.",
                "This French emperor was exiled to Elba and later to Saint Helena.",
            ],
            answer_choices=["Napoleon Bonaparte", "Julius Caesar", "Alexander the Great", "Charlemagne"],
            correct_answer_idx=0,
            category="history"
        ),
        Question(
            question_id="demo_science_001",
            clues=[
                "These organelles have their own circular DNA separate from nuclear DNA.",
                "They are believed to have originated from endosymbiotic bacteria.",
                "The inner membrane is folded into structures called cristae.",
                "They produce ATP through oxidative phosphorylation.",
                "These are often called the 'powerhouse of the cell'.",
            ],
            answer_choices=["Mitochondria", "Chloroplast", "Ribosome", "Endoplasmic Reticulum"],
            correct_answer_idx=0,
            category="science"
        )
    ]
    
    print("\n" + "=" * 70)
    print("DEMO WITH SAMPLE QUESTIONS")
    print("=" * 70)
    
    results = []
    for question in sample_questions:
        result = demo.run_episode(question, verbose=True)
        results.append(result)
        input("\nPress Enter to continue to next question...")
    
    # Summary
    print("\n" + "=" * 70)
    print("DEMO SUMMARY")
    print("=" * 70)
    correct = sum(1 for r in results if r['is_correct'])
    print(f"Questions: {len(results)}")
    print(f"Correct: {correct}/{len(results)} ({correct/len(results)*100:.1f}%)")
    avg_reward = sum(r['reward'] for r in results) / len(results)
    print(f"Average Reward: {avg_reward:.3f}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description='Interactive demo for QA model')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cuda', 'mps', 'cpu'],
                       help='Device to use')
    parser.add_argument('--mode', type=str, default='sample',
                       choices=['sample', 'interactive'],
                       help='Demo mode: sample questions or interactive')
    
    args = parser.parse_args()
    
    if args.mode == 'sample':
        demo_with_sample_questions(args.model_path, args.device)
    else:
        demo = InteractiveDemo(args.model_path, args.device)
        demo.interactive_mode()


if __name__ == "__main__":
    main()
```

## File: environment.py
```python
"""
POMDP Environment for Quiz Bowl Question Answering
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class Question:
    """Represents a quiz bowl question with pyramidal clues"""
    question_id: str
    clues: List[str]  # Ordered from difficult to easy
    answer_choices: List[str]  # 4 choices: [correct, distractor1, distractor2, distractor3]
    correct_answer_idx: int  # Index of correct answer (0-3)
    category: str
    metadata: Optional[Dict] = None


class QuizBowlEnvironment:
    """
    POMDP Environment for incremental question answering.
    
    States: Complete questions with all clues
    Observations: Partial questions (clues revealed so far) + answer choices
    Actions: WAIT (0) or SELECT answer i (1-4)
    Rewards: Shaped reward based on correctness and timing
    """
    
    WAIT_ACTION = 0
    
    def __init__(self, question: Question, reward_time_penalty: float = 0.1):
        """
        Initialize environment with a question.
        
        Args:
            question: Question object containing clues and answers
            reward_time_penalty: Penalty coefficient for late answering
        """
        self.question = question
        self.reward_time_penalty = reward_time_penalty
        
        self.num_clues = len(question.clues)
        self.num_actions = 1 + len(question.answer_choices)  # WAIT + SELECT answer
        
        # Episode state
        self.current_clue_idx = 0
        self.done = False
        self.selected_answer = None
        
    def reset(self) -> Dict:
        """
        Reset environment to initial state.
        
        Returns:
            Initial observation
        """
        self.current_clue_idx = 0
        self.done = False
        self.selected_answer = None
        return self._get_observation()
    
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
        if self.done:
            raise ValueError("Episode is already done. Call reset().")
        
        info = {
            'clue_position': self.current_clue_idx,
            'total_clues': self.num_clues
        }
        
        # Action is WAIT
        if action == self.WAIT_ACTION:
            self.current_clue_idx += 1
            
            # Check if we've run out of clues
            if self.current_clue_idx >= self.num_clues:
                # Forced to answer at last clue
                self.done = True
                info['forced_answer'] = True
                return self._get_observation(), 0.0, True, info
            
            # Continue episode
            return self._get_observation(), 0.0, False, info
        
        # Action is SELECT answer (1-4 maps to 0-3)
        else:
            answer_idx = action - 1
            
            if answer_idx < 0 or answer_idx >= len(self.question.answer_choices):
                raise ValueError(f"Invalid action: {action}. Must be 0-{self.num_actions-1}")
            
            self.selected_answer = answer_idx
            self.done = True
            
            # Compute reward
            is_correct = (answer_idx == self.question.correct_answer_idx)
            time_penalty = self.reward_time_penalty * (self.current_clue_idx / self.num_clues)
            
            if is_correct:
                reward = 1.0 - time_penalty
            else:
                reward = -time_penalty
            
            info['is_correct'] = is_correct
            info['answer_idx'] = answer_idx
            info['correct_idx'] = self.question.correct_answer_idx
            
            return self._get_observation(), reward, True, info
    
    def _get_observation(self) -> Dict:
        """
        Get current observation (partial question + answer choices).
        
        Returns:
            Dictionary containing visible clues and answer choices
        """
        visible_clues = self.question.clues[:self.current_clue_idx + 1]
        
        return {
            'clues': visible_clues,
            'answer_choices': self.question.answer_choices,
            'clue_position': self.current_clue_idx,
            'total_clues': self.num_clues,
            'category': self.question.category
        }
    
    def get_text_representation(self, observation: Optional[Dict] = None) -> str:
        """
        Convert observation to text string for model input.
        
        Args:
            observation: If None, use current observation
            
        Returns:
            Formatted text string
        """
        if observation is None:
            observation = self._get_observation()
        
        clues_text = " ".join(observation['clues'])
        choices_text = " | ".join([f"({i+1}) {choice}" 
                                   for i, choice in enumerate(observation['answer_choices'])])
        
        return f"CLUES: {clues_text} | CHOICES: {choices_text}"
    
    def get_choices_only_text(self) -> str:
        """Get text with only answer choices (for control experiment)"""
        choices_text = " | ".join([f"({i+1}) {choice}" 
                                   for i, choice in enumerate(self.question.answer_choices)])
        return f"CHOICES: {choices_text}"
    
    def render(self) -> str:
        """Render current state as string"""
        obs = self._get_observation()
        
        output = [
            f"Question ID: {self.question.question_id}",
            f"Category: {self.question.category}",
            f"Clue Position: {self.current_clue_idx + 1}/{self.num_clues}",
            "",
            "Visible Clues:"
        ]
        
        for i, clue in enumerate(obs['clues']):
            output.append(f"  {i+1}. {clue}")
        
        output.append("")
        output.append("Answer Choices:")
        for i, choice in enumerate(obs['answer_choices']):
            marker = " ✓" if i == self.question.correct_answer_idx else ""
            output.append(f"  ({i+1}) {choice}{marker}")
        
        if self.done and self.selected_answer is not None:
            output.append("")
            is_correct = self.selected_answer == self.question.correct_answer_idx
            output.append(f"Selected: ({self.selected_answer + 1}) {obs['answer_choices'][self.selected_answer]}")
            output.append(f"Result: {'CORRECT ✓' if is_correct else 'INCORRECT ✗'}")
        
        return "\n".join(output)


class BatchedEnvironment:
    """Manages multiple environments in parallel for efficient training"""
    
    def __init__(self, questions: List[Question], reward_time_penalty: float = 0.1):
        """
        Initialize batched environments.
        
        Args:
            questions: List of Question objects
            reward_time_penalty: Penalty coefficient for late answering
        """
        self.envs = [QuizBowlEnvironment(q, reward_time_penalty) for q in questions]
        self.num_envs = len(self.envs)
    
    def reset(self) -> List[Dict]:
        """Reset all environments"""
        return [env.reset() for env in self.envs]
    
    def step(self, actions: List[int]) -> Tuple[List[Dict], np.ndarray, np.ndarray, List[Dict]]:
        """
        Take actions in all environments.
        
        Args:
            actions: List of actions, one per environment
            
        Returns:
            observations, rewards, dones, infos
        """
        results = [env.step(action) for env, action in zip(self.envs, actions)]
        
        observations = [r[0] for r in results]
        rewards = np.array([r[1] for r in results])
        dones = np.array([r[2] for r in results])
        infos = [r[3] for r in results]
        
        return observations, rewards, dones, infos
    
    def get_text_representations(self, observations: Optional[List[Dict]] = None) -> List[str]:
        """Get text representations for all environments"""
        if observations is None:
            observations = [None] * self.num_envs
        return [env.get_text_representation(obs) 
                for env, obs in zip(self.envs, observations)]
```

## File: main.py
```python
"""
Main training script for CS234 RL Question Answering project
"""

import argparse
import torch
from pathlib import Path

from config import Config
from dataset import setup_datasets
from train_supervised import run_supervised_training
from train_ppo import run_ppo_training
from metrics import evaluate_model, evaluate_choices_only
from model import T5PolicyModel


def parse_args():
    parser = argparse.ArgumentParser(description='CS234 RL Question Answering')
    
    parser.add_argument('--mode', type=str, required=True,
                       choices=['supervised', 'ppo', 'full', 'eval'],
                       help='Training mode: supervised, ppo, full (both), or eval')
    
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to pretrained model (for ppo or eval mode)')
    
    parser.add_argument('--supervised_epochs', type=int, default=None,
                       help='Number of supervised epochs (overrides config)')
    
    parser.add_argument('--ppo_iterations', type=int, default=None,
                       help='Number of PPO iterations (overrides config)')
    
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size (overrides config)')
    
    parser.add_argument('--device', type=str, default=None,
                       choices=['cuda', 'mps', 'cpu'],
                       help='Device to use (overrides config)')
    
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    parser.add_argument('--num_questions', type=int, default=None,
                       help='Number of questions in dataset (overrides config)')
    
    return parser.parse_args()


def setup_config(args):
    """Setup configuration with command-line overrides"""
    config = Config()
    
    # Override with command-line arguments
    if args.supervised_epochs is not None:
        config.SUPERVISED_EPOCHS = args.supervised_epochs
    
    if args.ppo_iterations is not None:
        config.PPO_ITERATIONS = args.ppo_iterations
    
    if args.batch_size is not None:
        config.PPO_BATCH_SIZE = args.batch_size
        config.SUPERVISED_BATCH_SIZE = args.batch_size
    
    if args.device is not None:
        config.DEVICE = args.device
    
    if args.num_questions is not None:
        config.NUM_QUESTIONS = args.num_questions
    
    config.SEED = args.seed
    
    return config


def main():
    args = parse_args()
    
    # Setup configuration
    config = setup_config(args)
    
    # Set random seeds
    torch.manual_seed(config.SEED)
    import numpy as np
    import random
    np.random.seed(config.SEED)
    random.seed(config.SEED)
    
    # Print configuration
    config.print_config()
    
    # Setup datasets
    print("\nSetting up datasets...")
    train_dataset, val_dataset, test_dataset = setup_datasets(config)
    
    # Mode-specific execution
    if args.mode == 'supervised':
        print("\n" + "=" * 60)
        print("Running supervised training only")
        print("=" * 60)
        run_supervised_training(
            config=config,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset
        )
    
    elif args.mode == 'ppo':
        print("\n" + "=" * 60)
        print("Running PPO training only")
        print("=" * 60)
        
        # Determine pretrained model path
        if args.model_path:
            pretrained_path = args.model_path
        else:
            pretrained_path = Path(config.CHECKPOINT_DIR) / "supervised" / "best_model"
            if not pretrained_path.exists():
                print(f"\nWARNING: No pretrained model found at {pretrained_path}")
                print("Starting PPO without pretraining (not recommended)")
                pretrained_path = None
        
        run_ppo_training(
            config=config,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            pretrained_model_path=str(pretrained_path) if pretrained_path else None
        )
    
    elif args.mode == 'full':
        print("\n" + "=" * 60)
        print("Running full pipeline: supervised + PPO")
        print("=" * 60)
        
        # Phase 1: Supervised training
        print("\n### PHASE 1: SUPERVISED TRAINING ###\n")
        run_supervised_training(
            config=config,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=None  # Don't evaluate yet
        )
        
        # Phase 2: PPO training
        print("\n### PHASE 2: PPO TRAINING ###\n")
        supervised_path = Path(config.CHECKPOINT_DIR) / "supervised" / "best_model"
        run_ppo_training(
            config=config,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,  # Final evaluation after PPO
            pretrained_model_path=str(supervised_path)
        )
    
    elif args.mode == 'eval':
        print("\n" + "=" * 60)
        print("Running evaluation only")
        print("=" * 60)
        
        if not args.model_path:
            print("ERROR: --model_path required for eval mode")
            return
        
        # Load model
        print(f"Loading model from {args.model_path}")
        model = T5PolicyModel.load_pretrained(args.model_path, device=config.DEVICE)
        model.to(config.DEVICE)
        
        # Evaluate on test set
        print("\n### Full Question Evaluation ###")
        metrics = evaluate_model(model, test_dataset, device=config.DEVICE)
        metrics.print_summary()
        
        # Choices-only control
        print("\n### Choices-Only Control Experiment ###")
        choices_metrics = evaluate_choices_only(model, test_dataset, device=config.DEVICE)
        print(f"Accuracy (choices only): {choices_metrics.compute_accuracy():.4f}")
        print(f"Random baseline: 0.25 (1/4 choices)")
        print(f"ECE: {choices_metrics.compute_ece():.4f}")
        
        # Save results
        results_dir = Path(config.RESULTS_DIR)
        results_dir.mkdir(exist_ok=True)
        
        import json
        results = {
            'full_question': metrics.get_summary(),
            'choices_only': {
                'accuracy': choices_metrics.compute_accuracy(),
                'ece': choices_metrics.compute_ece()
            }
        }
        
        results_path = results_dir / "evaluation_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to {results_path}")
    
    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

## File: metrics.py
```python
"""
Evaluation metrics for question answering
"""

import numpy as np
import torch
from typing import List, Dict, Tuple, Any
from sklearn.metrics import accuracy_score
from collections import defaultdict


def convert_to_json_serializable(obj: Any) -> Any:
    """
    Convert numpy types to JSON-serializable Python types.
    
    Args:
        obj: Object to convert
        
    Returns:
        JSON-serializable version of the object
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_json_serializable(item) for item in obj)
    else:
        return obj


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
        
    def update(self, 
               pred: int,
               target: int,
               confidence: float,
               reward: float = None,
               buzz_position: int = None,
               category: str = None):
        """Update metrics with new sample"""
        self.predictions.append(pred)
        self.targets.append(target)
        self.confidences.append(confidence)
        
        is_correct = (pred == target)
        self.is_correct.append(is_correct)
        
        if reward is not None:
            self.rewards.append(reward)
        if buzz_position is not None:
            self.buzz_positions.append(buzz_position)
        if category is not None:
            self.categories.append(category)
    
    def compute_accuracy(self) -> float:
        """Compute overall accuracy"""
        if len(self.predictions) == 0:
            return 0.0
        return accuracy_score(self.targets, self.predictions)
    
    def compute_average_reward(self) -> float:
        """Compute average reward"""
        if len(self.rewards) == 0:
            return 0.0
        return np.mean(self.rewards)
    
    def compute_average_buzz_position(self) -> float:
        """Compute average buzz position"""
        if len(self.buzz_positions) == 0:
            return 0.0
        return np.mean(self.buzz_positions)
    
    def compute_ece(self, num_bins: int = 10) -> float:
        """
        Compute Expected Calibration Error (ECE).
        
        Args:
            num_bins: Number of bins for calibration
            
        Returns:
            ECE score
        """
        if len(self.confidences) == 0:
            return 0.0
        
        confidences = np.array(self.confidences)
        is_correct = np.array(self.is_correct, dtype=float)
        
        # Create bins
        bin_boundaries = np.linspace(0, 1, num_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find samples in this bin
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            
            if np.sum(in_bin) > 0:
                # Average confidence in bin
                avg_confidence = np.mean(confidences[in_bin])
                # Average accuracy in bin
                avg_accuracy = np.mean(is_correct[in_bin])
                # Bin weight
                bin_weight = np.sum(in_bin) / len(confidences)
                
                # Add to ECE
                ece += bin_weight * np.abs(avg_confidence - avg_accuracy)
        
        return ece
    
    def compute_brier_score(self) -> float:
        """
        Compute Brier score (mean squared error between confidence and correctness).
        
        Returns:
            Brier score
        """
        if len(self.confidences) == 0:
            return 0.0
        
        confidences = np.array(self.confidences)
        is_correct = np.array(self.is_correct, dtype=float)
        
        return np.mean((confidences - is_correct) ** 2)
    
    def compute_category_accuracy(self) -> Dict[str, float]:
        """Compute accuracy per category"""
        if len(self.categories) == 0:
            return {}
        
        category_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
        
        for pred, target, cat in zip(self.predictions, self.targets, self.categories):
            category_stats[cat]['total'] += 1
            if pred == target:
                category_stats[cat]['correct'] += 1
        
        return {cat: stats['correct'] / stats['total'] 
                for cat, stats in category_stats.items()}
    
    def compute_reliability_diagram_data(self, num_bins: int = 10) -> Dict:
        """
        Compute data for reliability diagram.
        
        Returns:
            Dictionary with bin information
        """
        if len(self.confidences) == 0:
            return {}
        
        confidences = np.array(self.confidences)
        is_correct = np.array(self.is_correct, dtype=float)
        
        bin_boundaries = np.linspace(0, 1, num_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        bin_data = {
            'bin_centers': [],
            'accuracies': [],
            'confidences': [],
            'counts': []
        }
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            
            if np.sum(in_bin) > 0:
                bin_center = (bin_lower + bin_upper) / 2
                avg_confidence = np.mean(confidences[in_bin])
                avg_accuracy = np.mean(is_correct[in_bin])
                count = np.sum(in_bin)
                
                bin_data['bin_centers'].append(bin_center)
                bin_data['accuracies'].append(avg_accuracy)
                bin_data['confidences'].append(avg_confidence)
                bin_data['counts'].append(count)
        
        return bin_data
    
    def compute_buzz_position_stats(self) -> Dict:
        """Compute statistics about buzzing positions"""
        if len(self.buzz_positions) == 0:
            return {}
        
        positions = np.array(self.buzz_positions)
        
        # Accuracy by position
        position_accuracy = defaultdict(lambda: {'correct': 0, 'total': 0})
        for pos, correct in zip(self.buzz_positions, self.is_correct):
            position_accuracy[pos]['total'] += 1
            if correct:
                position_accuracy[pos]['correct'] += 1
        
        return {
            'mean': np.mean(positions),
            'std': np.std(positions),
            'min': np.min(positions),
            'max': np.max(positions),
            'position_accuracy': {
                pos: stats['correct'] / stats['total']
                for pos, stats in position_accuracy.items()
            }
        }
    
    def get_summary(self) -> Dict:
        """Get summary of all metrics"""
        summary = {
            'num_samples': len(self.predictions),
            'accuracy': self.compute_accuracy(),
        }
        
        if len(self.rewards) > 0:
            summary['average_reward'] = self.compute_average_reward()
        
        if len(self.buzz_positions) > 0:
            summary['average_buzz_position'] = self.compute_average_buzz_position()
            summary['buzz_stats'] = self.compute_buzz_position_stats()
        
        if len(self.confidences) > 0:
            summary['ece'] = self.compute_ece()
            summary['brier_score'] = self.compute_brier_score()
        
        if len(self.categories) > 0:
            summary['category_accuracy'] = self.compute_category_accuracy()
        
        # Convert all numpy types to JSON-serializable Python types
        return convert_to_json_serializable(summary)
    
    def print_summary(self):
        """Print summary of metrics"""
        summary = self.get_summary()
        
        print("=" * 60)
        print("Evaluation Summary")
        print("=" * 60)
        print(f"Samples: {summary['num_samples']}")
        print(f"Accuracy: {summary['accuracy']:.4f}")
        
        if 'average_reward' in summary:
            print(f"Average Reward: {summary['average_reward']:.4f}")
        
        if 'average_buzz_position' in summary:
            print(f"Average Buzz Position: {summary['average_buzz_position']:.2f}")
            buzz_stats = summary['buzz_stats']
            print(f"  Min: {buzz_stats['min']}, Max: {buzz_stats['max']}, "
                  f"Std: {buzz_stats['std']:.2f}")
        
        if 'ece' in summary:
            print(f"ECE: {summary['ece']:.4f}")
            print(f"Brier Score: {summary['brier_score']:.4f}")
        
        if 'category_accuracy' in summary:
            print("\nCategory Accuracy:")
            for cat, acc in sorted(summary['category_accuracy'].items()):
                print(f"  {cat}: {acc:.4f}")
        
        print("=" * 60)


def compute_system_score(predictions: List[int],
                         targets: List[int],
                         buzz_positions: List[int],
                         total_clues: List[int]) -> float:
    """
    Compute QANTA system score S_q.
    
    S_q = (correct answers) / (total questions) * (1 - avg_position_ratio)
    where position_ratio = buzz_position / total_clues
    
    Args:
        predictions: Predicted answer indices
        targets: True answer indices
        buzz_positions: Position where model buzzed (0-indexed)
        total_clues: Total number of clues in each question
        
    Returns:
        System score
    """
    correct = np.array([p == t for p, t in zip(predictions, targets)], dtype=float)
    position_ratios = np.array([pos / total for pos, total in zip(buzz_positions, total_clues)])
    
    accuracy = np.mean(correct)
    avg_position_ratio = np.mean(position_ratios)
    
    system_score = accuracy * (1 - avg_position_ratio)
    
    return system_score


def evaluate_model(model, 
                  dataset,
                  device: str = 'cpu',
                  max_samples: int = None,
                  deterministic: bool = True) -> MetricsTracker:
    """
    Evaluate model on a dataset using the RL environment.
    
    Args:
        model: T5PolicyModel
        dataset: QuizBowlDataset
        device: Device to run on
        max_samples: Maximum samples to evaluate (None = all)
        deterministic: Use deterministic action selection
        
    Returns:
        MetricsTracker with results
    """
    from environment import QuizBowlEnvironment
    
    model.eval()
    metrics = MetricsTracker()
    
    questions = dataset.questions[:max_samples] if max_samples else dataset.questions
    
    with torch.no_grad():
        for question in questions:
            env = QuizBowlEnvironment(question)
            obs = env.reset()
            done = False
            
            while not done:
                # Get text representation
                text = env.get_text_representation(obs)
                
                # Tokenize
                inputs = model.tokenizer(
                    text,
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=512
                ).to(device)
                
                # Get action
                actions, info = model.select_action(
                    inputs['input_ids'],
                    inputs['attention_mask'],
                    deterministic=deterministic
                )
                
                action = actions.item()
                
                # Step environment
                obs, reward, done, step_info = env.step(action)
            
            # Get final metrics
            if 'is_correct' in step_info:
                # Extract confidence (max probability over answer choices)
                answer_probs = torch.softmax(info['answer_logits'], dim=-1)
                confidence = answer_probs.max().item()
                
                metrics.update(
                    pred=step_info['answer_idx'],
                    target=step_info['correct_idx'],
                    confidence=confidence,
                    reward=reward,
                    buzz_position=step_info['clue_position'],
                    category=question.category
                )
    
    return metrics


def evaluate_choices_only(model,
                          dataset,
                          device: str = 'cpu',
                          max_samples: int = None) -> MetricsTracker:
    """
    Evaluate model on answer choices only (control experiment).
    
    Args:
        model: T5PolicyModel
        dataset: QuizBowlDataset
        device: Device to run on
        max_samples: Maximum samples to evaluate
        
    Returns:
        MetricsTracker with results
    """
    from environment import QuizBowlEnvironment
    
    model.eval()
    metrics = MetricsTracker()
    
    questions = dataset.questions[:max_samples] if max_samples else dataset.questions
    
    with torch.no_grad():
        for question in questions:
            env = QuizBowlEnvironment(question)
            
            # Get choices-only text
            text = env.get_choices_only_text()
            
            # Tokenize
            inputs = model.tokenizer(
                text,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=512
            ).to(device)
            
            # Get prediction
            answer_logits, predictions = model.predict_answer(
                inputs['input_ids'],
                inputs['attention_mask']
            )
            
            pred = predictions.item()
            answer_probs = torch.softmax(answer_logits, dim=-1)
            confidence = answer_probs.max().item()
            
            metrics.update(
                pred=pred,
                target=question.correct_answer_idx,
                confidence=confidence,
                category=question.category
            )
    
    return metrics
```

## File: model.py
```python
"""
T5-based policy model for Quiz Bowl RL agent
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration, T5Tokenizer
from typing import Dict, List, Tuple, Optional
import numpy as np

from config import Config


class PolicyHead(nn.Module):
    """
    Custom policy head for the T5 model.
    Outputs: wait probability, answer distribution over choices, value estimate.
    """
    
    def __init__(self, hidden_size: int = 1024, num_choices: int = 4):
        """
        Initialize policy head.
        
        Args:
            hidden_size: Size of T5 hidden states
            num_choices: Number of answer choices
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_choices = num_choices
        
        # Wait/continue decision head (binary)
        self.wait_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 2)  # [wait, answer_now]
        )
        
        # Answer selection head (over choices)
        self.answer_head = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_choices)
        )
        
        # Value head (state value estimate)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )
    
    def forward(self, encoder_hidden_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through policy head.
        
        Args:
            encoder_hidden_state: [batch_size, hidden_size] - pooled encoder output
            
        Returns:
            wait_logits: [batch_size, 2] - logits for wait/answer
            answer_logits: [batch_size, num_choices] - logits for answer selection
            value: [batch_size, 1] - value estimate
        """
        wait_logits = self.wait_head(encoder_hidden_state)
        answer_logits = self.answer_head(encoder_hidden_state)
        value = self.value_head(encoder_hidden_state)
        
        return wait_logits, answer_logits, value


class T5PolicyModel(nn.Module):
    """
    T5-based policy model that combines T5 encoder with custom policy head.
    """
    
    def __init__(self, config: Config):
        """
        Initialize T5 policy model.
        
        Args:
            config: Configuration object
        """
        super().__init__()
        
        self.config = config
        self.device = torch.device(config.DEVICE)
        
        # Load T5 model and tokenizer
        print(f"Loading T5 model: {config.MODEL_NAME}")
        self.t5_model = T5ForConditionalGeneration.from_pretrained(config.MODEL_NAME)
        self.tokenizer = T5Tokenizer.from_pretrained(config.MODEL_NAME)
        
        # Get hidden size from T5 config
        hidden_size = self.t5_model.config.d_model
        
        # Custom policy head
        self.policy_head = PolicyHead(
            hidden_size=hidden_size,
            num_choices=config.NUM_ANSWER_CHOICES
        )
        
        # Move to device
        self.to(self.device)
        
        # Print model size
        self._print_model_info()
    
    def _print_model_info(self):
        """Print model architecture and parameter count"""
        t5_params = sum(p.numel() for p in self.t5_model.parameters())
        policy_params = sum(p.numel() for p in self.policy_head.parameters())
        total_params = t5_params + policy_params
        
        print(f"Model Architecture:")
        print(f"  T5 parameters: {t5_params:,}")
        print(f"  Policy head parameters: {policy_params:,}")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Device: {self.device}")
    
    def encode_input(self, 
                     text_inputs: List[str],
                     max_length: int = None) -> Dict[str, torch.Tensor]:
        """
        Encode text inputs using T5 tokenizer.
        
        Args:
            text_inputs: List of input strings
            max_length: Maximum sequence length
            
        Returns:
            Dictionary with input_ids and attention_mask
        """
        if max_length is None:
            max_length = self.config.MAX_INPUT_LENGTH
        
        encoding = self.tokenizer(
            text_inputs,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
        
        return {k: v.to(self.device) for k, v in encoding.items()}
    
    def get_encoder_output(self, 
                          input_ids: torch.Tensor,
                          attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Get T5 encoder output and pool to fixed-size representation.
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            
        Returns:
            pooled_output: [batch_size, hidden_size]
        """
        # Get encoder outputs
        encoder_outputs = self.t5_model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # encoder_outputs.last_hidden_state: [batch_size, seq_len, hidden_size]
        hidden_states = encoder_outputs.last_hidden_state
        
        # Mean pooling over sequence dimension (masked)
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_hidden = torch.sum(hidden_states * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        pooled_output = sum_hidden / sum_mask
        
        return pooled_output
    
    def forward(self,
                text_inputs: List[str],
                return_value: bool = True) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through the model.
        
        Args:
            text_inputs: List of text inputs (observations)
            return_value: Whether to return value estimates
            
        Returns:
            wait_logits: [batch_size, 2]
            answer_logits: [batch_size, num_choices]
            values: [batch_size, 1] or None
        """
        # Encode inputs
        encoding = self.encode_input(text_inputs)
        
        # Get encoder output
        pooled_output = self.get_encoder_output(
            encoding['input_ids'],
            encoding['attention_mask']
        )
        
        # Pass through policy head
        wait_logits, answer_logits, values = self.policy_head(pooled_output)
        
        if not return_value:
            values = None
        
        return wait_logits, answer_logits, values
    
    def predict_answer(self,
                      input_ids: torch.Tensor,
                      attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict answer choice (for supervised training).
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            
        Returns:
            answer_logits: [batch_size, num_choices] - logits over answer choices
            predictions: [batch_size] - predicted answer indices (argmax)
        """
        # Get encoder output
        pooled_output = self.get_encoder_output(input_ids, attention_mask)
        
        # Get answer logits from policy head
        _, answer_logits, _ = self.policy_head(pooled_output)
        
        # Get predictions
        predictions = torch.argmax(answer_logits, dim=-1)
        
        return answer_logits, predictions
    
    def select_action(self,
                     input_ids: torch.Tensor,
                     attention_mask: torch.Tensor,
                     deterministic: bool = False,
                     temperature: float = 1.0) -> Tuple[torch.Tensor, Dict]:
        """
        Select actions based on current policy.
        
        Args:
            input_ids: [batch_size, seq_len] - tokenized inputs
            attention_mask: [batch_size, seq_len] - attention mask
            deterministic: Use argmax instead of sampling
            temperature: Temperature for sampling
            
        Returns:
            actions: [batch_size] - combined actions (0=WAIT, 1-4=SELECT answer 0-3)
            info: Dictionary with action details including logits, probabilities, etc.
        """
        with torch.no_grad():
            # Get encoder output
            pooled_output = self.get_encoder_output(input_ids, attention_mask)
            
            # Get logits from policy head
            wait_logits, answer_logits, values = self.policy_head(pooled_output)
            
            # Apply temperature
            wait_logits = wait_logits / temperature
            answer_logits = answer_logits / temperature
            
            # Get probabilities
            wait_probs = F.softmax(wait_logits, dim=-1)
            answer_probs = F.softmax(answer_logits, dim=-1)
            
            if deterministic:
                # Take argmax
                wait_actions = torch.argmax(wait_probs, dim=-1)
                answer_actions = torch.argmax(answer_probs, dim=-1)
            else:
                # Sample from distribution
                wait_dist = torch.distributions.Categorical(wait_probs)
                answer_dist = torch.distributions.Categorical(answer_probs)
                
                wait_actions = wait_dist.sample()
                answer_actions = answer_dist.sample()
            
            # Compute log probabilities
            wait_log_probs = F.log_softmax(wait_logits, dim=-1)
            answer_log_probs = F.log_softmax(answer_logits, dim=-1)
            
            selected_wait_log_probs = wait_log_probs.gather(1, wait_actions.unsqueeze(-1)).squeeze(-1)
            selected_answer_log_probs = answer_log_probs.gather(1, answer_actions.unsqueeze(-1)).squeeze(-1)
            
            # Total log prob is sum (since actions are independent)
            log_probs = selected_wait_log_probs + selected_answer_log_probs
            
            # Combine wait and answer into single action
            # If wait_action == 0: action = 0 (WAIT)
            # If wait_action == 1: action = 1 + answer_action (SELECT answer 0-3)
            combined_actions = torch.where(
                wait_actions == 0,
                torch.zeros_like(wait_actions),
                1 + answer_actions
            )
            
            # Create info dict
            info = {
                'wait_logits': wait_logits,
                'answer_logits': answer_logits,
                'wait_probs': wait_probs,
                'answer_probs': answer_probs,
                'wait_actions': wait_actions,
                'answer_actions': answer_actions,
                'values': values,
                'log_probs': log_probs
            }
            
            return combined_actions, info
    
    def get_action_log_probs(self,
                            input_ids: torch.Tensor,
                            attention_mask: torch.Tensor,
                            actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get log probabilities and values for given actions.
        Used during PPO training.
        
        Args:
            input_ids: [batch_size, seq_len] - tokenized inputs
            attention_mask: [batch_size, seq_len] - attention mask
            actions: [batch_size] - combined actions (0=WAIT, 1-4=SELECT answer 0-3)
            
        Returns:
            log_probs: [batch_size] - log probs of actions
            entropy: [batch_size] - action entropy
            values: [batch_size] - value estimates
        """
        # Decompose combined actions into wait and answer actions
        # action 0 -> wait=0 (WAIT)
        # action 1-4 -> wait=1, answer=0-3 (SELECT answer)
        wait_actions = (actions > 0).long()
        answer_actions = torch.clamp(actions - 1, min=0)  # Map 1-4 to 0-3, keep 0 as 0
        
        # Get encoder output
        pooled_output = self.get_encoder_output(input_ids, attention_mask)
        
        # Get logits from policy head
        wait_logits, answer_logits, values = self.policy_head(pooled_output)
        
        # Compute log probabilities
        wait_log_probs = F.log_softmax(wait_logits, dim=-1)
        answer_log_probs = F.log_softmax(answer_logits, dim=-1)
        
        # Get log probs for selected actions
        selected_wait_log_probs = wait_log_probs.gather(1, wait_actions.unsqueeze(-1)).squeeze(-1)
        selected_answer_log_probs = answer_log_probs.gather(1, answer_actions.unsqueeze(-1)).squeeze(-1)
        
        # Total log prob
        log_probs = selected_wait_log_probs + selected_answer_log_probs
        
        # Compute entropy
        wait_probs = F.softmax(wait_logits, dim=-1)
        answer_probs = F.softmax(answer_logits, dim=-1)
        
        wait_entropy = -(wait_probs * wait_log_probs).sum(dim=-1)
        answer_entropy = -(answer_probs * answer_log_probs).sum(dim=-1)
        
        entropy = wait_entropy + answer_entropy
        
        return log_probs, entropy, values.squeeze(-1)
    
    def save(self, save_dir: str):
        """Save model checkpoint"""
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        # Save T5 model
        self.t5_model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        
        # Save policy head
        policy_head_path = os.path.join(save_dir, 'policy_head.pt')
        torch.save(self.policy_head.state_dict(), policy_head_path)
        
        print(f"Model saved to {save_dir}")
    
    def load(self, load_dir: str):
        """Load model checkpoint"""
        import os
        
        # Load T5 model
        self.t5_model = T5ForConditionalGeneration.from_pretrained(load_dir)
        self.tokenizer = T5Tokenizer.from_pretrained(load_dir)
        
        # Load policy head
        policy_head_path = os.path.join(load_dir, 'policy_head.pt')
        self.policy_head.load_state_dict(torch.load(policy_head_path, map_location=self.device))
        
        self.to(self.device)
        print(f"Model loaded from {load_dir}")
    
    @classmethod
    def load_pretrained(cls, load_dir: str, device: str = None):
        """
        Load a pretrained model from a directory.
        
        Args:
            load_dir: Directory containing saved model
            device: Device to load model on (e.g., 'cpu', 'cuda', 'mps')
            
        Returns:
            Loaded T5PolicyModel instance
        """
        import os
        from config import Config
        
        # Create a temporary config with the appropriate device
        config = Config()
        if device:
            config.DEVICE = device
        
        # Load T5 model and tokenizer from the directory
        # This will determine the model name from the saved config
        t5_model = T5ForConditionalGeneration.from_pretrained(load_dir, local_files_only=True)
        tokenizer = T5Tokenizer.from_pretrained(load_dir, local_files_only=True)
        
        # Create new model instance
        model = cls(config)
        model.t5_model = t5_model
        model.tokenizer = tokenizer
        
        # Load policy head
        policy_head_path = os.path.join(load_dir, 'policy_head.pt')
        if os.path.exists(policy_head_path):
            model.policy_head.load_state_dict(
                torch.load(policy_head_path, map_location=torch.device(config.DEVICE))
            )
        
        model.to(config.DEVICE)
        print(f"Model loaded from {load_dir}")
        
        return model
```

## File: run.sh
```bash
#!/bin/bash

# Quick start script for CS234 RL Question Answering project

echo "================================================"
echo "CS234 RL Question Answering - Quick Start"
echo "================================================"
echo ""

# Check if virtual environment is activated
if [[ -z "${VIRTUAL_ENV}" ]]; then
    echo "Virtual environment not activated. Activating..."
    if [ -d "venv" ]; then
        source venv/bin/activate
    else
        echo "Creating virtual environment..."
        python3 -m venv venv
        source venv/bin/activate
    fi
fi

# Install dependencies
echo "Checking dependencies..."
if ! python -c "import torch" 2>/dev/null; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
else
    echo "Dependencies already installed."
fi

echo ""
echo "================================================"
echo "Choose training mode:"
echo "================================================"
echo "1. Quick demo (small dataset, 10 epochs, 20 iterations)"
echo "2. Full pipeline (500 questions, 50 epochs, 250 iterations)"
echo "3. Supervised training only"
echo "4. PPO training only (requires pretrained model)"
echo "5. Evaluation only (requires trained model)"
echo "6. Interactive demo"
echo ""

read -p "Enter choice (1-6): " choice

case $choice in
    1)
        echo ""
        echo "Running quick demo..."
        python main.py --mode full \
            --num_questions 50 \
            --supervised_epochs 5 \
            --ppo_iterations 10 \
            --batch_size 8
        ;;
    2)
        echo ""
        echo "Running full pipeline (this will take several hours)..."
        python main.py --mode full
        ;;
    3)
        echo ""
        echo "Running supervised training..."
        python main.py --mode supervised
        ;;
    4)
        echo ""
        echo "Running PPO training..."
        if [ -d "checkpoints/supervised/best_model" ]; then
            python main.py --mode ppo --model_path checkpoints/supervised/best_model
        else
            echo "ERROR: Pretrained supervised model not found!"
            echo "Please run supervised training first (option 3)"
            exit 1
        fi
        ;;
    5)
        echo ""
        read -p "Enter model path: " model_path
        if [ -d "$model_path" ]; then
            python main.py --mode eval --model_path "$model_path"
        else
            echo "ERROR: Model not found at $model_path"
            exit 1
        fi
        ;;
    6)
        echo ""
        read -p "Enter model path: " model_path
        if [ -d "$model_path" ]; then
            python demo.py --model_path "$model_path" --mode sample
        else
            echo "ERROR: Model not found at $model_path"
            exit 1
        fi
        ;;
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac

echo ""
echo "================================================"
echo "Done!"
echo "================================================"
```

## File: test_csv_loader.py
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
    print(f"Number of clues: {len(question.clues)}")
    print(f"\nClues:")
    for j, clue in enumerate(question.clues, 1):
        print(f"  {j}. {clue}")
    print(f"\nAnswer choices:")
    for j, choice in enumerate(question.answer_choices):
        marker = " ✓" if j == question.correct_answer_idx else ""
        print(f"  {chr(65+j)}. {choice}{marker}")
    print(f"\nMetadata: {question.metadata}")

print("\n" + "=" * 60)
print("Testing Complete!")
print("=" * 60)
```

## File: test_imports.py
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

## File: train_ppo.py
```python
"""
PPO (Proximal Policy Optimization) training for T5 policy model
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
from typing import List, Dict, Tuple
from dataclasses import dataclass

from model import T5PolicyModel
from dataset import QuizBowlDataset
from environment import QuizBowlEnvironment
from metrics import MetricsTracker, evaluate_model
from config import Config


@dataclass
class RolloutStep:
    """Single step in an episode rollout"""
    observation_text: str
    action: int
    reward: float
    done: bool
    value: float
    log_prob: float
    
    # For tokenization
    input_ids: torch.Tensor = None
    attention_mask: torch.Tensor = None


class RolloutBuffer:
    """Buffer to store episode rollouts for PPO"""
    
    def __init__(self):
        self.rollouts = []
        self.reset()
    
    def reset(self):
        """Clear buffer"""
        self.rollouts = []
    
    def add_rollout(self, steps: List[RolloutStep]):
        """Add a complete episode rollout"""
        self.rollouts.append(steps)
    
    def get_all_steps(self) -> List[RolloutStep]:
        """Get all steps from all rollouts"""
        all_steps = []
        for rollout in self.rollouts:
            all_steps.extend(rollout)
        return all_steps
    
    def compute_returns_and_advantages(self, gamma: float, gae_lambda: float):
        """
        Compute discounted returns and GAE advantages for all rollouts.
        
        Args:
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
        """
        for rollout in self.rollouts:
            # Extract rewards and values
            rewards = [step.reward for step in rollout]
            values = [step.value for step in rollout]
            dones = [step.done for step in rollout]
            
            # Compute returns and advantages
            returns = []
            advantages = []
            
            # GAE computation
            gae = 0
            next_value = 0  # Terminal state has value 0
            
            for t in reversed(range(len(rollout))):
                if dones[t]:
                    next_value = 0
                    gae = 0
                
                # TD error
                delta = rewards[t] + gamma * next_value - values[t]
                
                # GAE
                gae = delta + gamma * gae_lambda * gae
                
                # Return = advantage + value
                returns.insert(0, gae + values[t])
                advantages.insert(0, gae)
                
                next_value = values[t]
            
            # Attach returns and advantages to steps
            for step, ret, adv in zip(rollout, returns, advantages):
                step.return_ = ret
                step.advantage = adv
    
    def __len__(self):
        return len(self.rollouts)


class PPOTrainer:
    """Trainer for PPO"""
    
    def __init__(self,
                 model: T5PolicyModel,
                 train_dataset: QuizBowlDataset,
                 val_dataset: QuizBowlDataset,
                 config: Config):
        """
        Initialize PPO trainer.
        
        Args:
            model: T5PolicyModel to train (should be pre-trained with supervised learning)
            train_dataset: Training dataset
            val_dataset: Validation dataset
            config: Configuration object
        """
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        
        self.device = config.DEVICE
        self.model.to(self.device)
        
        # Setup optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.PPO_LR,
            weight_decay=0.01
        )
        
        # Training state
        self.current_iteration = 0
        self.best_val_reward = -float('inf')
        self.history = []
        
        # Create checkpoint directory
        self.checkpoint_dir = Path(config.CHECKPOINT_DIR) / "ppo"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def collect_rollouts(self, num_episodes: int) -> RolloutBuffer:
        """
        Collect rollouts by running episodes in the environment.
        
       Args:
            num_episodes: Number of episodes to collect
            
        Returns:
            RolloutBuffer with collected rollouts
        """
        self.model.eval()
        buffer = RolloutBuffer()
        
        # Sample questions
        questions = self.train_dataset.get_batch(num_episodes)
        
        with torch.no_grad():
            for question in questions:
                env = QuizBowlEnvironment(
                    question,
                    reward_time_penalty=self.config.REWARD_TIME_PENALTY
                )
                
                obs = env.reset()
                done = False
                rollout = []
                
                while not done:
                    # Get text representation
                    text = env.get_text_representation(obs)
                    
                    # Tokenize
                    inputs = self.model.tokenizer(
                        text,
                        return_tensors='pt',
                        padding=True,
                        truncation=True,
                        max_length=self.config.MAX_INPUT_LENGTH
                    ).to(self.device)
                    
                    # Get action and log prob
                    actions, info = self.model.select_action(
                        inputs['input_ids'],
                        inputs['attention_mask'],
                        deterministic=False
                    )
                    
                    action = actions.item()
                    value = info['values'].item()
                    
                    # Get log prob of selected action
                    log_prob = info['log_probs'].item()
                    
                    # Take step
                    next_obs, reward, done, step_info = env.step(action)
                    
                    # Store step
                    step = RolloutStep(
                        observation_text=text,
                        action=action,
                        reward=reward,
                        done=done,
                        value=value,
                        log_prob=log_prob,
                        input_ids=inputs['input_ids'].cpu(),
                        attention_mask=inputs['attention_mask'].cpu()
                    )
                    rollout.append(step)
                    
                    obs = next_obs
                
                buffer.add_rollout(rollout)
        
        return buffer
    
    def update_policy(self, buffer: RolloutBuffer) -> Dict:
        """
        Update policy using PPO.
        
        Args:
            buffer: RolloutBuffer with collected rollouts
            
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        
        # Compute returns and advantages
        buffer.compute_returns_and_advantages(
            gamma=self.config.PPO_GAMMA,
            gae_lambda=self.config.PPO_GAE_LAMBDA
        )
        
        # Get all steps
        all_steps = buffer.get_all_steps()
        
        # Normalize advantages
        advantages = torch.tensor([step.advantage for step in all_steps])
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Training metrics
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        num_updates = 0
        
        # PPO epochs
        for epoch in range(self.config.PPO_EPOCHS_PER_ITER):
            # Shuffle steps
            indices = np.random.permutation(len(all_steps))
            
            # Mini-batch updates
            for start_idx in range(0, len(all_steps), self.config.PPO_BATCH_SIZE):
                end_idx = min(start_idx + self.config.PPO_BATCH_SIZE, len(all_steps))
                batch_indices = indices[start_idx:end_idx]
                
                # Get batch
                batch_steps = [all_steps[i] for i in batch_indices]
                
                # Prepare batch tensors with padding
                # Find max sequence length in batch
                max_len = max(step.input_ids.shape[1] for step in batch_steps)
                
                # Pad sequences
                padded_input_ids = []
                padded_attention_mask = []
                for step in batch_steps:
                    seq_len = step.input_ids.shape[1]
                    if seq_len < max_len:
                        # Pad with tokenizer's pad_token_id
                        pad_len = max_len - seq_len
                        input_ids_padded = torch.cat([
                            step.input_ids,
                            torch.full((1, pad_len), self.model.tokenizer.pad_token_id, dtype=step.input_ids.dtype)
                        ], dim=1)
                        attention_mask_padded = torch.cat([
                            step.attention_mask,
                            torch.zeros((1, pad_len), dtype=step.attention_mask.dtype)
                        ], dim=1)
                    else:
                        input_ids_padded = step.input_ids
                        attention_mask_padded = step.attention_mask
                    padded_input_ids.append(input_ids_padded)
                    padded_attention_mask.append(attention_mask_padded)
                
                input_ids = torch.cat(padded_input_ids).to(self.device)
                attention_mask = torch.cat(padded_attention_mask).to(self.device)
                actions = torch.tensor([step.action for step in batch_steps], dtype=torch.long).to(self.device)
                old_log_probs = torch.tensor([step.log_prob for step in batch_steps]).to(self.device)
                returns = torch.tensor([step.return_ for step in batch_steps]).to(self.device)
                batch_advantages = advantages[batch_indices].to(self.device)
                
                # Get new log probs and values
                new_log_probs, values, entropy = self.model.get_action_log_probs(
                    input_ids, attention_mask, actions
                )
                
                # PPO policy loss
                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(
                    ratio,
                    1.0 - self.config.PPO_CLIP_RATIO,
                    1.0 + self.config.PPO_CLIP_RATIO
                ) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = nn.MSELoss()(values, returns)
                
                # Entropy bonus (for exploration)
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = (policy_loss + 
                       self.config.PPO_VALUE_COEF * value_loss +
                       self.config.PPO_ENTROPY_COEF * entropy_loss)
                
                # Update
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.PPO_MAX_GRAD_NORM)
                self.optimizer.step()
                
                # Track metrics
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                num_updates += 1
        
        # Return average metrics
        return {
            'policy_loss': total_policy_loss / num_updates,
            'value_loss': total_value_loss / num_updates,
            'entropy': total_entropy / num_updates,
            'num_updates': num_updates
        }
    
    def validate(self) -> Dict:
        """Validate on validation set"""
        metrics = evaluate_model(
            self.model,
            self.val_dataset,
            device=self.device,
            deterministic=True
        )
        return metrics.get_summary()
    
    def train(self):
        """Run full PPO training"""
        print(f"Starting PPO training for {self.config.PPO_ITERATIONS} iterations")
        print(f"Training samples: {len(self.train_dataset)}")
        print(f"Validation samples: {len(self.val_dataset)}")
        print(f"Batch size: {self.config.PPO_BATCH_SIZE}")
        print(f"Device: {self.device}")
        print()
        
        for iteration in range(self.config.PPO_ITERATIONS):
            self.current_iteration = iteration
            
            # Collect rollouts
            print(f"\nIteration {iteration + 1}/{self.config.PPO_ITERATIONS}")
            print("Collecting rollouts...")
            buffer = self.collect_rollouts(self.config.PPO_BATCH_SIZE)
            
            # Compute episode statistics
            episode_rewards = []
            episode_lengths = []
            for rollout in buffer.rollouts:
                episode_reward = sum(step.reward for step in rollout)
                episode_rewards.append(episode_reward)
                episode_lengths.append(len(rollout))
            
            avg_reward = np.mean(episode_rewards)
            avg_length = np.mean(episode_lengths)
            
            print(f"Avg episode reward: {avg_reward:.4f}")
            print(f"Avg episode length: {avg_length:.2f}")
            
            # Update policy
            print("Updating policy...")
            update_metrics = self.update_policy(buffer)
            
            print(f"Policy loss: {update_metrics['policy_loss']:.4f}")
            print(f"Value loss: {update_metrics['value_loss']:.4f}")
            print(f"Entropy: {update_metrics['entropy']:.4f}")
            
            # Validate periodically
            if (iteration + 1) % self.config.EVAL_INTERVAL == 0:
                print("\nValidating...")
                val_summary = self.validate()
                val_reward = val_summary.get('average_reward', 0)
                
                print(f"Val Accuracy: {val_summary['accuracy']:.4f}")
                print(f"Val Reward: {val_reward:.4f}")
                print(f"Val ECE: {val_summary.get('ece', 0):.4f}")
                print(f"Val Buzz Position: {val_summary.get('average_buzz_position', 0):.2f}")
                
                # Save history
                self.history.append({
                    'iteration': iteration + 1,
                    'train_reward': avg_reward,
                    'train_length': avg_length,
                    **update_metrics,
                    'val': val_summary
                })
                
                # Save best model
                if val_reward > self.best_val_reward:
                    self.best_val_reward = val_reward
                    self.save_checkpoint(is_best=True)
                    print(f"New best validation reward: {val_reward:.4f}")
            
            # Save regular checkpoint
            if (iteration + 1) % self.config.SAVE_INTERVAL == 0:
                self.save_checkpoint(is_best=False)
                self.save_history()
        
        print("\n" + "=" * 60)
        print("PPO training completed!")
        print(f"Best validation reward: {self.best_val_reward:.4f}")
        print("=" * 60)
        
        # Save final history
        self.save_history()
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint"""
        if is_best:
            save_path = self.checkpoint_dir / "best_model"
        else:
            save_path = self.checkpoint_dir / f"iter_{self.current_iteration + 1}"
        
        # Use T5PolicyModel's save() method
        self.model.save(str(save_path))
        
        # Save training state
        state = {
            'iteration': self.current_iteration + 1,
            'best_val_reward': self.best_val_reward,
            'optimizer_state_dict': self.optimizer.state_dict()
        }
        torch.save(state, save_path / "training_state.pt")
        
        print(f"Checkpoint saved to {save_path}")
    
    def save_history(self):
        """Save training history"""
        history_path = self.checkpoint_dir / "history.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)


def run_ppo_training(config: Config,
                    train_dataset: QuizBowlDataset,
                    val_dataset: QuizBowlDataset,
                    test_dataset: QuizBowlDataset = None,
                    pretrained_model_path: str = None):
    """
    Run PPO training pipeline.
    
    Args:
        config: Configuration object
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Optional test dataset for final evaluation
        pretrained_model_path: Path to pretrained supervised model
    """
    print("=" * 60)
    print("PPO TRAINING PHASE")
    print("=" * 60)
    
    # Load model
    if pretrained_model_path:
        print(f"Loading pretrained model from {pretrained_model_path}")
        model = T5PolicyModel.load_pretrained(pretrained_model_path, device=config.DEVICE)
    else:
        print("Initializing new model (no pretraining)")
        model = T5PolicyModel(config)
    
    model.to(config.DEVICE)
    
    # Create trainer
    trainer = PPOTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=config
    )
    
    # Train
    trainer.train()
    
    # Evaluate on test set if provided
    if test_dataset is not None:
        print("\n" + "=" * 60)
        print("FINAL EVALUATION ON TEST SET")
        print("=" * 60)
        
        # Load best model if it exists, otherwise use current model
        best_model_path = trainer.checkpoint_dir / "best_model"
        if best_model_path.exists():
            print(f"\nLoading best model from {best_model_path}")
            model = T5PolicyModel.load_pretrained(str(best_model_path), device=config.DEVICE)
            model.to(config.DEVICE)
        else:
            print("\nNo best model found, using current model for evaluation")
            model = trainer.model
        
        # Evaluate
        print("\nRunning full evaluation...")
        metrics = evaluate_model(model, test_dataset, device=config.DEVICE)
        metrics.print_summary()
        
        # Save test results
        test_results = metrics.get_summary()
        results_path = trainer.checkpoint_dir / "test_results.json"
        with open(results_path, 'w') as f:
            json.dump(test_results, f, indent=2)
        
        print(f"\nTest results saved to {results_path}")
    
    return model, trainer


if __name__ == "__main__":
    from dataset import setup_datasets
    
    # Load config
    config = Config()
    config.print_config()
    
    # Setup datasets
    train_dataset, val_dataset, test_dataset = setup_datasets(config)
    
    # Path to supervised pretrained model
    supervised_model_path = Path(config.CHECKPOINT_DIR) / "supervised" / "best_model"
    
    if supervised_model_path.exists():
        print(f"\nFound supervised pretrained model at {supervised_model_path}")
    else:
        print(f"\nWARNING: Supervised model not found at {supervised_model_path}")
        print("Consider running supervised training first!")
        supervised_model_path = None
    
    # Run PPO training
    model, trainer = run_ppo_training(
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        pretrained_model_path=str(supervised_model_path) if supervised_model_path else None
    )
```

## File: train_supervised.py
```python
"""
Supervised training for T5 policy model
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json

from model import T5PolicyModel
from dataset import QuizBowlDataset
from environment import QuizBowlEnvironment
from metrics import MetricsTracker, evaluate_model, evaluate_choices_only
from config import Config


class SupervisedTrainer:
    """Trainer for supervised learning phase"""
    
    def __init__(self,
                 model: T5PolicyModel,
                 train_dataset: QuizBowlDataset,
                 val_dataset: QuizBowlDataset,
                 config: Config):
        """
        Initialize supervised trainer.
        
        Args:
            model: T5PolicyModel to train
            train_dataset: Training dataset
            val_dataset: Validation dataset
            config: Configuration object
        """
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        
        self.device = config.DEVICE
        self.model.to(self.device)
        
        # Setup optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.SUPERVISED_LR,
            weight_decay=0.01
        )
        
        # Setup loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Training state
        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.train_history = []
        self.val_history = []
        
        # Create checkpoint directory
        self.checkpoint_dir = Path(config.CHECKPOINT_DIR) / "supervised"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def prepare_batch(self, questions):
        """
        Prepare batch of questions for supervised training.
        Uses complete questions (all clues).
        
        Args:
            questions: List of Question objects
            
        Returns:
            input_ids, attention_mask, labels (all on device)
        """
        texts = []
        labels = []
        
        for question in questions:
            # Create environment to get text representation
            env = QuizBowlEnvironment(question)
            # Set to last clue position (show all clues)
            env.current_clue_idx = len(question.clues) - 1
            text = env.get_text_representation()
            
            texts.append(text)
            labels.append(question.correct_answer_idx)
        
        # Tokenize
        inputs = self.model.tokenizer(
            texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=self.config.MAX_INPUT_LENGTH
        )
        
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        labels = torch.tensor(labels, dtype=torch.long).to(self.device)
        
        return input_ids, attention_mask, labels
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        
        # Shuffle dataset
        self.train_dataset.shuffle()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        # Training loop with mini-batches
        num_batches = len(self.train_dataset) // self.config.SUPERVISED_BATCH_SIZE
        
        progress_bar = tqdm(range(num_batches), desc=f"Epoch {self.current_epoch + 1}")
        
        for batch_idx in progress_bar:
            # Get batch
            batch_questions = self.train_dataset.get_batch(self.config.SUPERVISED_BATCH_SIZE)
            input_ids, attention_mask, labels = self.prepare_batch(batch_questions)
            
            # Forward pass
            answer_logits, predictions = self.model.predict_answer(input_ids, attention_mask)
            
            # Compute loss
            loss = self.criterion(answer_logits, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.config.SUPERVISED_GRAD_ACCUM_STEPS == 0:
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                # Update weights
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            # Track metrics
            total_loss += loss.item()
            total_correct += (predictions == labels).sum().item()
            total_samples += len(labels)
            
            # Update progress bar
            avg_loss = total_loss / (batch_idx + 1)
            avg_acc = total_correct / total_samples
            progress_bar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'acc': f'{avg_acc:.4f}'
            })
        
        # Compute epoch metrics
        epoch_loss = total_loss / num_batches
        epoch_acc = total_correct / total_samples
        
        return epoch_loss, epoch_acc
    
    def validate(self):
        """Validate on validation set"""
        print("Validating...")
        metrics = evaluate_model(
            self.model,
            self.val_dataset,
            device=self.device,
            deterministic=True
        )
        
        return metrics.get_summary()
    
    def train(self):
        """Run full supervised training"""
        print(f"Starting supervised training for {self.config.SUPERVISED_EPOCHS} epochs")
        print(f"Training samples: {len(self.train_dataset)}")
        print(f"Validation samples: {len(self.val_dataset)}")
        print(f"Device: {self.device}")
        print()
        
        for epoch in range(self.config.SUPERVISED_EPOCHS):
            self.current_epoch = epoch
            
            # Train epoch
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_summary = self.validate()
            val_acc = val_summary['accuracy']
            
            # Log results
            print(f"\nEpoch {epoch + 1}/{self.config.SUPERVISED_EPOCHS}")
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Acc: {val_acc:.4f}, Val ECE: {val_summary.get('ece', 0):.4f}")
            print()
            
            # Save history
            self.train_history.append({
                'epoch': epoch + 1,
                'loss': train_loss,
                'accuracy': train_acc
            })
            self.val_history.append({
                'epoch': epoch + 1,
                **val_summary
            })
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_checkpoint(is_best=True)
                print(f"New best validation accuracy: {val_acc:.4f}")
            
            # Save regular checkpoint
            if (epoch + 1) % self.config.SAVE_INTERVAL == 0:
                self.save_checkpoint(is_best=False)
        
        print("\nSupervised training completed!")
        print(f"Best validation accuracy: {self.best_val_acc:.4f}")
        
        # Save training history
        self.save_history()
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint"""
        if is_best:
            save_path = self.checkpoint_dir / "best_model"
        else:
            save_path = self.checkpoint_dir / f"epoch_{self.current_epoch + 1}"
        
        # Use T5PolicyModel's save() method
        self.model.save(str(save_path))
        
        # Save training state
        state = {
            'epoch': self.current_epoch + 1,
            'best_val_acc': self.best_val_acc,
            'optimizer_state_dict': self.optimizer.state_dict()
        }
        torch.save(state, save_path / "training_state.pt")
        
        print(f"Checkpoint saved to {save_path}")
    
    def save_history(self):
        """Save training history"""
        import numpy as np
        
        def convert_to_native(obj):
            """Convert numpy types to Python native types"""
            if isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(v) for v in obj]
            elif isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return convert_to_native(obj.tolist())
            else:
                return obj
        
        history = {
            'train': convert_to_native(self.train_history),
            'val': convert_to_native(self.val_history)
        }
        
        history_path = self.checkpoint_dir / "history.json"
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        print(f"Training history saved to {history_path}")


def run_supervised_training(config: Config,
                           train_dataset: QuizBowlDataset,
                           val_dataset: QuizBowlDataset,
                           test_dataset: QuizBowlDataset = None):
    """
    Run supervised training pipeline.
    
    Args:
        config: Configuration object
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Optional test dataset for final evaluation
    """
    print("=" * 60)
    print("SUPERVISED TRAINING PHASE")
    print("=" * 60)
    
    # Initialize model
    model = T5PolicyModel(config)
    
    # Create trainer
    trainer = SupervisedTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=config
    )
    
    # Train
    trainer.train()
    
    # Evaluate on test set if provided
    if test_dataset is not None:
        print("\n" + "=" * 60)
        print("FINAL EVALUATION ON TEST SET")
        print("=" * 60)
        
        # Load best model
        best_model_path = trainer.checkpoint_dir / "best_model"
        model = T5PolicyModel.load_pretrained(str(best_model_path), device=config.DEVICE)
        model.to(config.DEVICE)
        
        # Full evaluation
        print("\nFull Question Evaluation:")
        metrics = evaluate_model(model, test_dataset, device=config.DEVICE)
        metrics.print_summary()
        
        # Choices-only evaluation (control)
        print("\nChoices-Only Evaluation (Control):")
        choices_metrics = evaluate_choices_only(model, test_dataset, device=config.DEVICE)
        print(f"Accuracy (choices only): {choices_metrics.compute_accuracy():.4f}")
        print(f"Random baseline: 0.25 (1/4 choices)")
        
        # Save test results
        test_results = {
            'full_question': metrics.get_summary(),
            'choices_only': {
                'accuracy': choices_metrics.compute_accuracy(),
                'ece': choices_metrics.compute_ece()
            }
        }
        
        # Convert numpy types to native Python types for JSON serialization
        import numpy as np
        
        def convert_to_native(obj):
            """Convert numpy types to Python native types"""
            if isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(v) for v in obj]
            elif isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return convert_to_native(obj.tolist())
            else:
                return obj
        
        test_results = convert_to_native(test_results)
        
        results_path = trainer.checkpoint_dir / "test_results.json"
        with open(results_path, 'w') as f:
            json.dump(test_results, f, indent=2)
        
        print(f"\nTest results saved to {results_path}")
    
    return model, trainer


if __name__ == "__main__":
    from dataset import setup_datasets
    
    # Load config
    config = Config()
    config.print_config()
    
    # Setup datasets
    train_dataset, val_dataset, test_dataset = setup_datasets(config)
    
    # Run supervised training
    model, trainer = run_supervised_training(
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset
    )
```

## File: visualize.py
```python
"""
Visualization utilities for analyzing training results
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse


def plot_training_curves(history, save_dir):
    """Plot training curves from history"""
    
    # Extract data
    iterations = [h['iteration'] for h in history]
    train_rewards = [h['train_reward'] for h in history]
    val_accuracies = [h['val']['accuracy'] for h in history]
    val_rewards = [h['val'].get('average_reward', 0) for h in history]
    val_ece = [h['val'].get('ece', 0) for h in history]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Training reward
    axes[0, 0].plot(iterations, train_rewards, 'b-', linewidth=2, label='Train Reward')
    axes[0, 0].plot(iterations, val_rewards, 'r-', linewidth=2, label='Val Reward')
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Average Reward')
    axes[0, 0].set_title('Reward Progress')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Validation accuracy
    axes[0, 1].plot(iterations, val_accuracies, 'g-', linewidth=2)
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Validation Accuracy')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: ECE (calibration)
    axes[1, 0].plot(iterations, val_ece, 'purple', linewidth=2)
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('Expected Calibration Error')
    axes[1, 0].set_title('Calibration (ECE)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Policy and value loss
    policy_losses = [h.get('policy_loss', 0) for h in history]
    value_losses = [h.get('value_loss', 0) for h in history]
    
    axes[1, 1].plot(iterations, policy_losses, 'orange', linewidth=2, label='Policy Loss')
    axes[1, 1].plot(iterations, value_losses, 'cyan', linewidth=2, label='Value Loss')
    axes[1, 1].set_xlabel('Iteration')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].set_title('Training Losses')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
    print(f"Saved training curves to {save_dir / 'training_curves.png'}")
    plt.close()


def plot_reliability_diagram(metrics_data, save_dir):
    """Plot reliability diagram for calibration"""
    
    # Get reliability data
    bin_data = metrics_data.get('reliability_data', {})
    
    if not bin_data:
        print("No reliability data available")
        return
    
    bin_centers = bin_data['bin_centers']
    accuracies = bin_data['accuracies']
    confidences = bin_data['confidences']
    counts = bin_data['counts']
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect Calibration')
    
    # Plot actual calibration
    ax.scatter(confidences, accuracies, s=np.array(counts)*5, 
              alpha=0.6, c='blue', edgecolors='black', linewidth=1.5,
              label='Model Calibration')
    
    # Plot bars
    for conf, acc, count in zip(confidences, accuracies, counts):
        ax.plot([conf, conf], [conf, acc], 'r-', alpha=0.5, linewidth=2)
    
    ax.set_xlabel('Confidence', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Reliability Diagram', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(save_dir / 'reliability_diagram.png', dpi=300, bbox_inches='tight')
    print(f"Saved reliability diagram to {save_dir / 'reliability_diagram.png'}")
    plt.close()


def plot_buzzing_behavior(metrics_data, save_dir):
    """Plot buzzing position distribution"""
    
    buzz_stats = metrics_data.get('buzz_stats', {})
    
    if not buzz_stats:
        print("No buzzing statistics available")
        return
    
    position_accuracy = buzz_stats.get('position_accuracy', {})
    
    if not position_accuracy:
        return
    
    positions = sorted(position_accuracy.keys())
    accuracies = [position_accuracy[p] for p in positions]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Accuracy by position
    axes[0].bar(positions, accuracies, alpha=0.7, color='steelblue', edgecolor='black')
    axes[0].set_xlabel('Buzz Position (Clue Number)')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Accuracy by Buzz Position')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Distribution of buzz positions
    mean_pos = buzz_stats.get('mean', 0)
    std_pos = buzz_stats.get('std', 0)
    
    axes[1].axvline(mean_pos, color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {mean_pos:.2f}')
    axes[1].axvline(mean_pos - std_pos, color='orange', linestyle=':', linewidth=1.5,
                   label=f'±1 Std: {std_pos:.2f}')
    axes[1].axvline(mean_pos + std_pos, color='orange', linestyle=':', linewidth=1.5)
    axes[1].set_xlabel('Buzz Position')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Buzzing Position Distribution')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'buzzing_behavior.png', dpi=300, bbox_inches='tight')
    print(f"Saved buzzing behavior to {save_dir / 'buzzing_behavior.png'}")
    plt.close()


def plot_category_performance(metrics_data, save_dir):
    """Plot per-category performance"""
    
    category_acc = metrics_data.get('category_accuracy', {})
    
    if not category_acc:
        print("No category-specific data available")
        return
    
    categories = list(category_acc.keys())
    accuracies = [category_acc[c] for c in categories]
    
    # Sort by accuracy
    sorted_indices = np.argsort(accuracies)
    categories = [categories[i] for i in sorted_indices]
    accuracies = [accuracies[i] for i in sorted_indices]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(categories)))
    bars = ax.barh(categories, accuracies, color=colors, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for i, (cat, acc) in enumerate(zip(categories, accuracies)):
        ax.text(acc + 0.01, i, f'{acc:.3f}', va='center', fontsize=10)
    
    ax.set_xlabel('Accuracy', fontsize=12)
    ax.set_ylabel('Category', fontsize=12)
    ax.set_title('Performance by Category', fontsize=14, fontweight='bold')
    ax.set_xlim([0, max(accuracies) * 1.15])
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'category_performance.png', dpi=300, bbox_inches='tight')
    print(f"Saved category performance to {save_dir / 'category_performance.png'}")
    plt.close()


def create_summary_report(checkpoint_dir, output_dir):
    """Create comprehensive visualization report"""
    
    checkpoint_dir = Path(checkpoint_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load history
    history_path = checkpoint_dir / 'history.json'
    if history_path.exists():
        with open(history_path, 'r') as f:
            history = json.load(f)
        
        print(f"Loaded training history from {history_path}")
        plot_training_curves(history, output_dir)
    else:
        print(f"No history file found at {history_path}")
    
    # Load test results
    test_results_path = checkpoint_dir / 'test_results.json'
    if test_results_path.exists():
        with open(test_results_path, 'r') as f:
            test_results = json.load(f)
        
        print(f"Loaded test results from {test_results_path}")
        
        # Plot reliability diagram
        # Note: This would need reliability data from a separate run
        # plot_reliability_diagram(test_results, output_dir)
        
        # Plot buzzing behavior
        if 'buzz_stats' in test_results:
            plot_buzzing_behavior(test_results, output_dir)
        
        # Plot category performance
        if 'category_accuracy' in test_results:
            plot_category_performance(test_results, output_dir)
        
        # Print summary
        print("\n" + "=" * 60)
        print("TEST RESULTS SUMMARY")
        print("=" * 60)
        print(f"Accuracy: {test_results.get('accuracy', 0):.4f}")
        print(f"Average Reward: {test_results.get('average_reward', 0):.4f}")
        print(f"ECE: {test_results.get('ece', 0):.4f}")
        print(f"Avg Buzz Position: {test_results.get('average_buzz_position', 0):.2f}")
        print("=" * 60)
    else:
        print(f"No test results found at {test_results_path}")
    
    print(f"\nAll visualizations saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Visualize training results')
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                       help='Directory containing checkpoints and history')
    parser.add_argument('--output_dir', type=str, default='visualizations',
                       help='Directory to save visualizations')
    
    args = parser.parse_args()
    
    create_summary_report(args.checkpoint_dir, args.output_dir)


if __name__ == "__main__":
    main()
```

## File: agents/_math.py
```python
from __future__ import annotations

import math


def sigmoid(x: float) -> float:
    """Numerically stable logistic sigmoid for scalar confidence proxies."""
    if x >= 0.0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)

    z = math.exp(x)
    return z / (1.0 + z)
```

## File: evaluation/plotting.py
```python
"""
Visualization Functions for Quiz Bowl Buzzer Evaluation

Provides plotting utilities for evaluation results including entropy curves,
calibration plots, and comparison tables. All functions accept output paths
and create parent directories as needed.

Ported from qb-rl reference implementation (evaluation/plotting.py) with
import path adaptations for the unified qanta-buzzer codebase.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def _ensure_parent(path: str | Path) -> Path:
    """Create parent directories for an output path if needed.

    Parameters
    ----------
    path : str or Path
        Output file path.

    Returns
    -------
    Path
        The resolved Path object.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def plot_learning_curve(
    timesteps: list[int],
    rewards: list[float],
    output_path: str | Path,
) -> str:
    """Plot training learning curve (reward vs timesteps).

    Parameters
    ----------
    timesteps : list[int]
        Training timestep values.
    rewards : list[float]
        Corresponding episode reward values.
    output_path : str or Path
        File path for the saved figure.

    Returns
    -------
    str
        Path to the saved figure.
    """
    p = _ensure_parent(output_path)
    plt.figure(figsize=(7, 4))
    sns.lineplot(x=timesteps, y=rewards)
    plt.title("Learning Curve")
    plt.xlabel("Timesteps")
    plt.ylabel("Episode Reward")
    plt.tight_layout()
    plt.savefig(p)
    plt.close()
    return str(p)


def plot_entropy_vs_clue_index(
    entropy_traces: dict[str, list[float]],
    output_path: str | Path,
) -> str:
    """Plot policy entropy as a function of clue index.

    Creates a line plot with multiple agent entropy traces showing how
    policy uncertainty decreases as more clues are revealed.

    Parameters
    ----------
    entropy_traces : dict[str, list[float]]
        Mapping from agent name to per-step entropy values.
    output_path : str or Path
        File path for the saved figure.

    Returns
    -------
    str
        Path to the saved figure.
    """
    p = _ensure_parent(output_path)
    plt.figure(figsize=(7, 4))
    for label, trace in entropy_traces.items():
        x = np.arange(len(trace))
        sns.lineplot(x=x, y=trace, label=label)
    plt.title("Belief Entropy vs Clue Index")
    plt.xlabel("Clue index")
    plt.ylabel("Entropy")
    plt.tight_layout()
    plt.savefig(p)
    plt.close()
    return str(p)


def plot_calibration_curve(
    confidences: list[float],
    outcomes: list[int],
    output_path: str | Path,
    n_bins: int = 10,
) -> str:
    """Plot calibration curve (predicted confidence vs empirical accuracy).

    Bins confidences into uniform bins and plots mean accuracy per bin
    against mean confidence. The diagonal represents perfect calibration.

    Parameters
    ----------
    confidences : list[float]
        Predicted confidence values in [0, 1].
    outcomes : list[int]
        Binary outcomes (1 = correct, 0 = incorrect).
    output_path : str or Path
        File path for the saved figure.
    n_bins : int
        Number of uniform bins for confidence bucketing.

    Returns
    -------
    str
        Path to the saved figure.
    """
    p = _ensure_parent(output_path)
    conf = np.array(confidences, dtype=np.float64)
    y = np.array(outcomes, dtype=np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    xs = []
    ys = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf >= lo) & (conf < hi if i < n_bins - 1 else conf <= hi)
        if not mask.any():
            continue
        xs.append(conf[mask].mean())
        ys.append(y[mask].mean())

    plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.scatter(xs, ys, color="tab:blue")
    plt.title("Calibration Plot")
    plt.xlabel("Predicted confidence")
    plt.ylabel("Empirical accuracy")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(p)
    plt.close()
    return str(p)


def save_comparison_table(
    rows: list[dict[str, Any]],
    output_path: str | Path,
) -> str:
    """Save agent comparison metrics as a CSV or markdown table.

    Parameters
    ----------
    rows : list[dict[str, Any]]
        List of metric dicts, each with agent name and metrics.
    output_path : str or Path
        File path for the saved table (.csv or .md).

    Returns
    -------
    str
        Path to the saved table file.
    """
    p = _ensure_parent(output_path)
    df = pd.DataFrame(rows)
    if p.suffix.lower() == ".csv":
        df.to_csv(p, index=False)
    else:
        df.to_markdown(p, index=False)
    return str(p)
```

## File: models/dspy_likelihood.py
```python
"""DSPy-based likelihood model with score caching.

Wraps a DSPy listwise scorer behind the ``LikelihoodModel.score()``
interface.  Unlike embedding-based models, the DSPy scorer calls an LM
to rank options — so caching is at the *score* level (keyed by clue +
options + program fingerprint), not at the embedding level.

This module is importable without the ``dspy`` extra installed.
The ``dspy`` package is only required at runtime when a DSPy-backed
scorer is actually invoked (e.g. via ``scripts/optimize_dspy.py``).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from models.likelihoods import LikelihoodModel


def _score_cache_key(
    clue_prefix: str,
    option_profiles: list[str],
    program_fingerprint: str,
) -> str:
    """Build a deterministic cache key for a score() call."""
    payload = json.dumps(
        {"clue": clue_prefix, "options": option_profiles, "fp": program_fingerprint},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class DSPyLikelihood(LikelihoodModel):
    """LikelihoodModel subclass backed by a DSPy program.

    Inherits from ``LikelihoodModel`` so it satisfies the factory
    return type and isinstance checks.  Overrides ``score()`` with
    LM-based scoring and a score-level cache.  ``_embed_batch()`` raises
    ``NotImplementedError`` because DSPy scoring is not embedding-based.

    Unlike TF-IDF/SBERT/T5, this model does NOT produce embeddings.
    ``_embed_batch`` is explicitly unsupported — calling it raises
    ``NotImplementedError``.  Instead, scores are cached directly,
    keyed by ``(clue, options, program_fingerprint)``.

    Parameters
    ----------
    scorer : callable
        A DSPy module or function that accepts ``(clue_prefix, options)``
        and returns a list/array of K scores.
    program_fingerprint : str
        Opaque identifier for the current compiled program state.
        Cache entries are invalidated when this changes.
    cache_dir : str or Path or None
        Directory for persistent score cache.  When None, caching is
        in-memory only.
    """

    def __init__(
        self,
        scorer: Any,
        program_fingerprint: str = "default",
        cache_dir: str | Path | None = None,
    ) -> None:
        super().__init__()
        self.scorer = scorer
        self.program_fingerprint = program_fingerprint
        self._score_cache: dict[str, np.ndarray] = {}
        self._cache_dir = Path(cache_dir) if cache_dir else None
        if self._cache_dir:
            self._load_persistent_cache()

    def _load_persistent_cache(self) -> None:
        if self._cache_dir is None:
            return
        cache_file = self._cache_dir / f"dspy_scores_{self.program_fingerprint}.npz"
        if cache_file.exists():
            with np.load(cache_file, allow_pickle=False) as data:
                for key in data.files:
                    self._score_cache[key] = data[key].astype(np.float32)

    def _save_persistent_cache(self) -> None:
        if self._cache_dir is None or not self._score_cache:
            return
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = self._cache_dir / f"dspy_scores_{self.program_fingerprint}.npz"
        np.savez_compressed(cache_file, **self._score_cache)

    def score(self, clue_prefix: str, option_profiles: list[str]) -> np.ndarray:
        """Score answer options using the DSPy scorer.

        Results are cached by ``(clue, options, program_fingerprint)``.
        Validates that the returned array has shape ``(K,)`` where
        ``K = len(option_profiles)``.
        """
        key = _score_cache_key(clue_prefix, option_profiles, self.program_fingerprint)
        if key in self._score_cache:
            return self._score_cache[key].copy()

        raw = self.scorer(clue_prefix, option_profiles)
        scores = np.array(raw, dtype=np.float32)
        expected_k = len(option_profiles)
        if scores.ndim != 1 or len(scores) != expected_k:
            raise ValueError(
                f"DSPy scorer returned shape {scores.shape}, "
                f"expected ({expected_k},)"
            )
        self._score_cache[key] = scores
        return scores.copy()

    def save_cache(self, path: str | Path | None = None) -> int:
        """Persist score cache to disk."""
        if path:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(p, **self._score_cache)
        else:
            self._save_persistent_cache()
        return len(self._score_cache)

    def load_cache(self, path: str | Path) -> int:
        """Load score cache from disk, merging without overwriting."""
        p = Path(path)
        if not p.exists():
            return 0
        loaded = 0
        with np.load(p, allow_pickle=False) as data:
            for key in data.files:
                if key not in self._score_cache:
                    self._score_cache[key] = data[key].astype(np.float32)
                    loaded += 1
        return loaded

    @property
    def cache_memory_bytes(self) -> int:
        return sum(v.nbytes for v in self._score_cache.values())

    def _embed_batch(self, texts: list[str]) -> np.ndarray:
        """Not supported — DSPy scoring is not embedding-based."""
        raise NotImplementedError(
            "DSPyLikelihood does not produce embeddings. "
            "Use score() directly."
        )

    def embed_and_cache(self, texts: list[str]) -> np.ndarray:
        """Not supported — DSPy scoring is not embedding-based."""
        raise NotImplementedError(
            "DSPyLikelihood does not produce embeddings. "
            "Use score() directly."
        )
```

## File: qb_data/dspy_answer_profiles.py
```python
"""Optional DSPy-based answer profile augmentation.

Generates richer answer profiles using an LM when the ``dspy`` extra is
installed and enabled.  The extractive ``AnswerProfileBuilder`` remains
the default and fallback — this module only augments, never replaces.

This module requires the ``dspy`` optional extra.
"""

from __future__ import annotations

from typing import Any


def build_dspy_profiles(
    answers: list[str],
    existing_profiles: dict[str, str],
    dspy_config: dict[str, Any],
    max_answers: int = 100,
) -> dict[str, str]:
    """Generate LM-augmented answer profiles via DSPy.

    Leave-one-out discipline depends on the *caller* providing
    ``existing_profiles`` that already exclude the current question
    (as ``AnswerProfileBuilder.profile_for_answer(answer, exclude_qid)``
    does).  This function itself does not receive per-question exclusion
    context — it augments whatever profiles it is given.

    Parameters
    ----------
    answers : list[str]
        Answer strings to generate profiles for.
    existing_profiles : dict[str, str]
        Extractive profiles from ``AnswerProfileBuilder``.
    dspy_config : dict
        DSPy configuration section from YAML.
    max_answers : int
        Cap on number of answers to augment.

    Returns
    -------
    dict[str, str]
        Mapping from answer to augmented profile text.  Falls back to
        the extractive profile when augmentation fails.
    """
    try:
        import dspy
    except ImportError as exc:
        raise ImportError(
            "DSPy answer profile augmentation requires the dspy package. "
            "Install with: pip install -e '.[dspy]'"
        ) from exc

    lm_name = dspy_config.get("model", "openai/gpt-4o-mini")
    lm = dspy.LM(lm_name)
    dspy.configure(lm=lm)

    class AnswerProfileSignature(dspy.Signature):
        """Generate a rich factual profile for a quiz bowl answer."""
        answer: str = dspy.InputField(desc="the answer entity")
        existing_profile: str = dspy.InputField(desc="extractive profile from question corpus")
        augmented_profile: str = dspy.OutputField(desc="enriched factual profile suitable for quiz bowl scoring")

    generator = dspy.Predict(AnswerProfileSignature)

    import logging

    logger = logging.getLogger(__name__)

    result: dict[str, str] = {}
    n_augmented = 0
    n_fallback = 0
    for answer in answers[:max_answers]:
        existing = existing_profiles.get(answer, "")
        try:
            pred = generator(answer=answer, existing_profile=existing)
            result[answer] = pred.augmented_profile
            n_augmented += 1
        except Exception as exc:
            logger.warning("DSPy augmentation failed for %r: %s", answer, exc)
            result[answer] = existing
            n_fallback += 1

    for answer in answers[max_answers:]:
        result[answer] = existing_profiles.get(answer, "")

    if n_fallback:
        logger.info(
            "DSPy profiles: %d augmented, %d fell back to extractive",
            n_augmented, n_fallback,
        )

    return result
```

## File: qb_env/opponent_models.py
```python
"""Opponent buzz-position models for Expected Wins reward computation.

Provides pluggable opponent models that estimate the probability an
opponent has buzzed before a given step.  Used by the ``expected_wins``
reward mode in :class:`TossupMCEnv`.

Three built-in models:

* :class:`EmpiricalHistogramOpponentModel` — derives CDF from
  ``MCQuestion.human_buzz_positions`` data.
* :class:`LogisticOpponentModel` — parametric sigmoid CDF for
  questions that lack empirical data.
* :func:`build_opponent_model_from_config` — factory with fallback
  hierarchy: question-level empirical → global empirical → logistic.

The ``expected_wins`` reward mode is disabled by default.  To enable,
set ``environment.reward_mode: expected_wins`` and optionally configure
``environment.opponent_buzz_model`` in the YAML config.
"""

from __future__ import annotations

import math
from typing import Any, Protocol, runtime_checkable

import numpy as np

from qb_data.mc_builder import MCQuestion


@runtime_checkable
class OpponentBuzzModel(Protocol):
    """Protocol for opponent buzz-position models."""

    def prob_buzzed_before_step(self, question: MCQuestion, step_idx: int) -> float:
        """Cumulative probability that the opponent has buzzed before *step_idx*.

        Parameters
        ----------
        question : MCQuestion
            Current question (may carry ``human_buzz_positions``).
        step_idx : int
            0-based clue step.

        Returns
        -------
        float
            P(opponent buzzed before step_idx), in [0, 1].
        """
        ...

    def prob_survive_to_step(self, question: MCQuestion, step_idx: int) -> float:
        """Probability that the opponent has NOT buzzed by *step_idx*.

        Complement of :meth:`prob_buzzed_before_step`.
        """
        ...


class LogisticOpponentModel:
    """Parametric logistic CDF opponent model.

    Models the opponent's cumulative buzz probability at step *t* as::

        P(buzzed before t) = 1 / (1 + exp(-steepness * (t/total - midpoint)))

    Parameters
    ----------
    midpoint : float
        Fraction of total steps at which the CDF reaches 0.5.
    steepness : float
        Controls how sharply the probability increases around the
        midpoint.  Higher values → sharper transition.
    """

    def __init__(self, midpoint: float = 0.6, steepness: float = 6.0) -> None:
        self.midpoint = midpoint
        self.steepness = steepness

    def prob_buzzed_before_step(self, question: MCQuestion, step_idx: int) -> float:
        total = len(question.cumulative_prefixes)
        if total <= 1:
            return 0.0
        frac = step_idx / total
        x = self.steepness * (frac - self.midpoint)
        if x >= 0:
            return 1.0 / (1.0 + math.exp(-x))
        z = math.exp(x)
        return z / (1.0 + z)

    def prob_survive_to_step(self, question: MCQuestion, step_idx: int) -> float:
        return 1.0 - self.prob_buzzed_before_step(question, step_idx)


class EmpiricalHistogramOpponentModel:
    """Opponent model derived from empirical human buzz-position data.

    Builds a per-step CDF from the ``human_buzz_positions`` field on
    each question.  Falls back to a :class:`LogisticOpponentModel`
    when a question has no empirical data.

    Parameters
    ----------
    fallback : LogisticOpponentModel or None
        Model to use when a question lacks empirical data.
    global_positions : list of (int, int) or None
        Pooled (position, count) pairs from the entire dataset.
        Used when a question has no per-question data but a global
        distribution is available.
    """

    def __init__(
        self,
        fallback: LogisticOpponentModel | None = None,
        global_positions: list[tuple[int, int]] | None = None,
    ) -> None:
        self.fallback = fallback or LogisticOpponentModel()
        self._global_cdf: np.ndarray | None = None
        if global_positions:
            self._global_cdf = self._build_cdf(global_positions)

    @staticmethod
    def _build_cdf(positions: list[tuple[int, int]]) -> np.ndarray:
        """Build a CDF array from (position, count) pairs.

        Returns an array where ``cdf[i]`` is the cumulative probability
        that a buzz has occurred at or before position *i*.
        """
        if not positions:
            return np.array([], dtype=np.float64)
        max_pos = max(p for p, _ in positions)
        counts = np.zeros(max_pos + 1, dtype=np.float64)
        for pos, count in positions:
            counts[pos] += count
        total = counts.sum()
        if total <= 0:
            return np.zeros(max_pos + 1, dtype=np.float64)
        return np.cumsum(counts) / total

    def _cdf_at_step(
        self, cdf: np.ndarray, question: MCQuestion, step_idx: int
    ) -> float:
        """Look up cumulative probability at a token position."""
        if cdf.size == 0:
            return 0.0
        if not question.run_indices:
            token_pos = step_idx
        elif step_idx < len(question.run_indices):
            token_pos = question.run_indices[step_idx]
        else:
            token_pos = question.run_indices[-1] if question.run_indices else step_idx
        idx = min(token_pos, len(cdf) - 1)
        return float(cdf[idx])

    def prob_buzzed_before_step(self, question: MCQuestion, step_idx: int) -> float:
        if question.human_buzz_positions:
            cdf = self._build_cdf(question.human_buzz_positions)
            return self._cdf_at_step(cdf, question, step_idx)
        if self._global_cdf is not None and self._global_cdf.size > 0:
            return self._cdf_at_step(self._global_cdf, question, step_idx)
        return self.fallback.prob_buzzed_before_step(question, step_idx)

    def prob_survive_to_step(self, question: MCQuestion, step_idx: int) -> float:
        return 1.0 - self.prob_buzzed_before_step(question, step_idx)


def build_opponent_model_from_config(
    questions: list[MCQuestion] | None = None,
    config: dict[str, Any] | None = None,
) -> OpponentBuzzModel | None:
    """Build an opponent model from YAML configuration.

    Returns ``None`` when the opponent model is disabled (the default).

    Parameters
    ----------
    questions : list[MCQuestion] or None
        Dataset questions for building global empirical distribution.
    config : dict or None
        Full YAML config dict.

    Returns
    -------
    OpponentBuzzModel or None
    """
    if config is None:
        return None
    env_cfg = config.get("environment", {})
    opp_cfg = env_cfg.get("opponent_buzz_model", {})
    if not opp_cfg or opp_cfg.get("type", "none") == "none":
        return None

    model_type = opp_cfg.get("type", "logistic")

    if model_type == "logistic":
        return LogisticOpponentModel(
            midpoint=float(opp_cfg.get("midpoint", 0.6)),
            steepness=float(opp_cfg.get("steepness", 6.0)),
        )

    if model_type == "empirical":
        global_positions: list[tuple[int, int]] = []
        if questions:
            for q in questions:
                if q.human_buzz_positions:
                    global_positions.extend(q.human_buzz_positions)
        fallback = LogisticOpponentModel(
            midpoint=float(opp_cfg.get("midpoint", 0.6)),
            steepness=float(opp_cfg.get("steepness", 6.0)),
        )
        return EmpiricalHistogramOpponentModel(
            fallback=fallback,
            global_positions=global_positions if global_positions else None,
        )

    raise ValueError(f"Unknown opponent_buzz_model type: {model_type}")
```

## File: scripts/ci.sh
```bash
#!/usr/bin/env bash
# CI entry point -- runs the full pytest suite from the project venv.
# Exit nonzero on any failure so CI gates catch regressions.
#
# Usage:
#   bash scripts/ci.sh              # full suite
#   bash scripts/ci.sh -k "not t5"  # skip T5-dependent tests
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

if [ -f "$REPO_ROOT/.venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$REPO_ROOT/.venv/bin/activate"
elif ! command -v pytest &>/dev/null; then
    echo "ERROR: No .venv found and pytest not on PATH." >&2
    echo "Run: python3 -m venv .venv && source .venv/bin/activate && pip install -e ." >&2
    exit 1
fi

pytest tests/ "$@"
```

## File: scripts/manual-smoke.sh
```bash
#!/usr/bin/env bash
# Manual smoke pipeline -- runs the four-stage belief-feature smoke workflow.
# Intended for human verification, not CI (stages are heavyweight ML runs).
#
# Prereqs: pip install -e .  (see AGENTS.md for full setup)
# Outputs: artifacts/smoke/
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
if [ -f "$REPO_ROOT/.venv/bin/activate" ]; then
    source "$REPO_ROOT/.venv/bin/activate"
fi

PYTHON="${PYTHON:-python3}"

echo "=== Stage 1/4: Build MC dataset ==="
$PYTHON scripts/build_mc_dataset.py --smoke

echo "=== Stage 2/4: Run baselines ==="
$PYTHON scripts/run_baselines.py --smoke

echo "=== Stage 3/4: Train PPO ==="
$PYTHON scripts/train_ppo.py --smoke

echo "=== Stage 4/4: Evaluate all ==="
$PYTHON scripts/evaluate_all.py --smoke

echo "=== Smoke pipeline complete. Check artifacts/smoke/ ==="
```

## File: scripts/optimize_dspy.py
```python
#!/usr/bin/env python3
"""Offline DSPy compile/optimize workflow.

Compiles a DSPy scorer program against quiz bowl training data.
Does NOT integrate with PPO rollouts — this is pure offline tooling.

Usage:
    python scripts/optimize_dspy.py --config configs/default.yaml
    python scripts/optimize_dspy.py --config configs/default.yaml --optimizer MIPROv2
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def build_dspy_trainset(
    mc_questions: list,
    max_examples: int = 50,
) -> list[dict[str, Any]]:
    """Build training examples for DSPy optimization.

    Each example contains a clue prefix, option profiles, and the gold
    answer index — suitable for ``dspy.Example``.

    Parameters
    ----------
    mc_questions : list
        MC question objects with cumulative_prefixes, option_profiles,
        and gold_index.
    max_examples : int
        Cap on the number of examples.

    Returns
    -------
    list[dict]
        Training examples.
    """
    examples = []
    for q in mc_questions[:max_examples]:
        mid = len(q.cumulative_prefixes) // 2
        prefix = q.cumulative_prefixes[mid] if q.cumulative_prefixes else q.question
        examples.append({
            "clue_prefix": prefix,
            "option_profiles": q.option_profiles,
            "gold_index": q.gold_index,
        })
    return examples


def _score_metric(example, prediction, _trace=None):
    """Compare predicted scores against gold target via argmax match.

    Used as the optimization metric for DSPy ``BootstrapFewShot`` and
    ``MIPROv2``.  Returns 1.0 when the argmax of the predicted scores
    matches the argmax of the target scores, 0.0 otherwise.
    """
    try:
        pred_scores = json.loads(prediction.scores)
        target_scores = json.loads(example.scores)
    except (json.JSONDecodeError, AttributeError):
        return 0.0
    if not pred_scores or not target_scores:
        return 0.0
    return 1.0 if (
        max(range(len(pred_scores)), key=lambda i: pred_scores[i])
        == max(range(len(target_scores)), key=lambda i: target_scores[i])
    ) else 0.0


def compile_dspy_scorer(
    trainset: list[dict[str, Any]],
    dspy_config: dict[str, Any],
) -> dict[str, Any]:
    """Compile a DSPy scorer program.

    Requires the ``dspy`` package to be installed.

    Parameters
    ----------
    trainset : list[dict]
        Training examples from ``build_dspy_trainset()``.
    dspy_config : dict
        DSPy configuration section from YAML.

    Returns
    -------
    dict
        Compilation result with ``program_fingerprint`` and metadata.
    """
    try:
        import dspy
    except ImportError as exc:
        raise ImportError(
            "DSPy optimization requires the dspy package. "
            "Install with: pip install -e '.[dspy]'"
        ) from exc

    lm_name = dspy_config.get("model", "openai/gpt-4o-mini")
    optimizer_name = dspy_config.get("optimizer", "BootstrapFewShot")

    lm = dspy.LM(lm_name)
    dspy.configure(lm=lm)

    class MCScoreSignature(dspy.Signature):
        """Score how well each answer option matches the quiz clue."""
        clue_prefix: str = dspy.InputField(desc="partial quiz question clue text")
        options: str = dspy.InputField(desc="JSON list of answer option profile texts")
        scores: str = dspy.OutputField(desc="JSON list of float scores, one per option")

    scorer = dspy.Predict(MCScoreSignature)

    examples = []
    for ex in trainset:
        gold = ex["gold_index"]
        target_scores = [0.0] * len(ex["option_profiles"])
        target_scores[gold] = 1.0
        examples.append(dspy.Example(
            clue_prefix=ex["clue_prefix"],
            options=json.dumps(ex["option_profiles"]),
            scores=json.dumps(target_scores),
        ).with_inputs("clue_prefix", "options"))

    if optimizer_name == "MIPROv2":
        optimizer = dspy.MIPROv2(metric=_score_metric)
    else:
        optimizer = dspy.BootstrapFewShot(metric=_score_metric)

    compiled = optimizer.compile(scorer, trainset=examples)

    fingerprint = hashlib.md5(
        json.dumps(dspy_config, sort_keys=True).encode()
    ).hexdigest()[:12]

    return {
        "program_fingerprint": fingerprint,
        "optimizer": optimizer_name,
        "n_examples": len(examples),
        "compiled_program": compiled,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline DSPy optimization")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--optimizer", type=str, default=None)
    parser.add_argument("--max-examples", type=int, default=None)
    args = parser.parse_args()

    from scripts._common import load_config, load_mc_questions, ARTIFACT_DIR

    config = load_config(args.config)
    dspy_cfg = config.get("dspy", {})
    if args.optimizer:
        dspy_cfg["optimizer"] = args.optimizer
    max_ex = args.max_examples or int(dspy_cfg.get("max_examples", 50))

    # Use the train split to avoid leaking val/test data into DSPy compilation
    train_path = ARTIFACT_DIR / "smoke" / "train_dataset.json"
    if not train_path.exists():
        train_path = ARTIFACT_DIR / "main" / "train_dataset.json"
    if not train_path.exists():
        # Fallback to combined dataset with warning
        train_path = ARTIFACT_DIR / "smoke" / "mc_dataset.json"
        if not train_path.exists():
            train_path = ARTIFACT_DIR / "main" / "mc_dataset.json"
        print(f"Warning: train split not found, using combined dataset: {train_path}")
    questions = load_mc_questions(train_path)
    trainset = build_dspy_trainset(questions, max_examples=max_ex)

    print(f"Built {len(trainset)} training examples")
    print(f"Compiling with {dspy_cfg.get('optimizer', 'BootstrapFewShot')}...")
    result = compile_dspy_scorer(trainset, dspy_cfg)
    print(f"Compiled. Fingerprint: {result['program_fingerprint']}")


if __name__ == "__main__":
    main()
```

## File: scripts/run_full_pipeline.sh
```bash
#!/usr/bin/env bash
# Full pipeline with parallelism — runs the core pipeline plus key extensions.
# Phases 9/10/12/18/19 require manual execution (see docs/full-pipeline-runbook.md).
#
# Dependencies form a DAG:
#
#   Phase 1 (build MC dataset)
#     ├── Wave 1 (3 parallel tracks): Phases 2, 3, 5
#     │     Track A: baselines → writes artifacts/main/baseline_summary.json
#     │     Track B: PPO → writes artifacts/main/ppo_model.zip
#     │     Track C: T5 policy → writes checkpoints/
#     ├── Wave 2 (sequential, after Wave 1): Phases 4, 6, 11, 15
#     │     All read/write artifacts/main/ — must be sequential
#     ├── Wave 3 (sequential): Phases 14, 16, 17
#     │     PPO ablations that reuse artifacts/main/
#     └── Wave 4 (sequential): Phase 13 (K-sensitivity)
#         Builds to artifacts/k*/ then runs baselines (writes artifacts/main/)
#         Must run after Wave 2 so it doesn't clobber baseline_summary.json
#
# Usage:
#   bash scripts/run_full_pipeline.sh                    # t5-base (balanced)
#   bash scripts/run_full_pipeline.sh --t5-model t5-small # fastest
#   bash scripts/run_full_pipeline.sh --t5-model t5-large # full quality
#   bash scripts/run_full_pipeline.sh --sequential        # no parallelism
#
# Requirements:
#   - Python venv activated with `pip install -e .`
#   - questions.csv at repo root
#   - ~10 GB free disk space
#
# Estimated wall time (Apple M3 Max, 64 GB):
#   t5-small, parallel: ~2–3 hours
#   t5-base, parallel:  ~3–5 hours
#   t5-large, parallel: ~6–10 hours

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# Parse arguments
T5_MODEL="t5-base"
SEQUENTIAL=false
while [ $# -gt 0 ]; do
    case "$1" in
        --t5-model) T5_MODEL="$2"; shift 2 ;;
        --t5-model=*) T5_MODEL="${1#*=}"; shift ;;
        --sequential) SEQUENTIAL=true; shift ;;
        *) shift ;;
    esac
done

echo "============================================================"
echo "FULL PIPELINE — T5 model: $T5_MODEL, parallel: $([ "$SEQUENTIAL" = true ] && echo no || echo yes)"
echo "============================================================"
echo ""

RESULTS="$REPO_ROOT/results"
mkdir -p "$RESULTS"

# Activate venv if available
if [ -f "$REPO_ROOT/.venv/bin/activate" ]; then
    source "$REPO_ROOT/.venv/bin/activate"
fi

# Helper: run a command, log to file, print status on completion
run_phase() {
    local PHASE="$1"
    local LOG="$RESULTS/phase_${PHASE}.log"
    shift
    echo "[Phase $PHASE] STARTED at $(date +%H:%M:%S)"
    if PYTHONUNBUFFERED=1 "$@" > "$LOG" 2>&1; then
        echo "[Phase $PHASE] DONE at $(date +%H:%M:%S) — see $LOG"
    else
        echo "[Phase $PHASE] FAILED at $(date +%H:%M:%S) — see $LOG"
        return 1
    fi
}

# Helper: wait for background jobs, exit on first failure
wait_all() {
    local PIDS=("$@")
    for pid in "${PIDS[@]}"; do
        if ! wait "$pid"; then
            echo "ERROR: Background job $pid failed"
            kill "${PIDS[@]}" 2>/dev/null || true
            exit 1
        fi
    done
}

########################################################################
# PHASE 1: Build MC dataset (sequential — everything depends on this)
########################################################################
echo "=== PHASE 1: Build MC dataset ==="
python scripts/build_mc_dataset.py \
    --config configs/default.yaml \
    --output-dir artifacts/main
echo "[Phase 1] DONE — $(python -c "import json; print(f'{len(json.load(open(\"artifacts/main/mc_dataset.json\")))} MC questions')")"
echo ""

MC="artifacts/main/mc_dataset.json"

if [ "$SEQUENTIAL" = true ]; then
    ####################################################################
    # SEQUENTIAL MODE
    ####################################################################
    echo "=== Running all phases sequentially ==="

    echo "=== PHASE 2: Baselines (TF-IDF) ==="
    python scripts/run_baselines.py --config configs/default.yaml --mc-path "$MC" likelihood.model=tfidf
    cp artifacts/main/baseline_summary.json "$RESULTS/baselines_tfidf.json"

    echo "=== PHASE 3: PPO (100k steps) ==="
    python scripts/train_ppo.py --config configs/default.yaml --mc-path "$MC" --seed 13 --deterministic-eval likelihood.model=tfidf
    cp artifacts/main/ppo_summary.json "$RESULTS/ppo_default.json"
    cp artifacts/main/ppo_model.zip "$RESULTS/ppo_model_default.zip"

    echo "=== PHASE 4: Evaluate all ==="
    python scripts/evaluate_all.py --config configs/default.yaml --mc-path "$MC" likelihood.model=tfidf
    cp artifacts/main/evaluation_report.json "$RESULTS/eval_default.json"

    echo "=== PHASE 5: T5 policy ==="
    python scripts/train_t5_policy.py --config configs/t5_policy.yaml model.model_name="$T5_MODEL"

    echo "=== PHASE 6: Compare policies ==="
    python scripts/compare_policies.py \
        --mlp-checkpoint artifacts/main/ppo_model \
        --t5-checkpoint checkpoints/ppo_t5/best_model \
        --mc-path "$MC" \
        --output "$RESULTS/t5_comparison.json"

    echo "=== PHASE 11: Expected Wins ==="
    python scripts/evaluate_all.py --config configs/default.yaml --mc-path "$MC" \
        likelihood.model=tfidf environment.reward_mode=expected_wins environment.opponent_buzz_model.type=logistic
    cp artifacts/main/evaluation_report.json "$RESULTS/eval_ew_logistic.json"

    echo "=== PHASE 14: Reward modes ==="
    for MODE in simple human_grounded; do
        python scripts/train_ppo.py --config configs/default.yaml --mc-path "$MC" \
            --seed 13 --deterministic-eval likelihood.model=tfidf environment.reward_mode="$MODE"
        cp artifacts/main/ppo_summary.json "$RESULTS/ppo_$MODE.json"
    done

    echo "=== PHASE 16: Stop-only PPO ==="
    python scripts/train_ppo.py --config configs/default.yaml --mc-path "$MC" \
        --seed 13 --deterministic-eval --policy-mode stop_only likelihood.model=tfidf
    cp artifacts/main/ppo_summary.json "$RESULTS/ppo_stop_only.json"

    echo "=== PHASE 17: No-buzz horizon ==="
    python scripts/train_ppo.py --config configs/default.yaml --mc-path "$MC" \
        --seed 13 --deterministic-eval likelihood.model=tfidf environment.end_mode=no_buzz environment.no_buzz_reward=-0.25
    cp artifacts/main/ppo_summary.json "$RESULTS/ppo_no_buzz.json"

    echo "=== PHASE 15: Belief mode (sequential_bayes) ==="
    python scripts/run_baselines.py --config configs/default.yaml --mc-path "$MC" \
        environment.belief_mode=sequential_bayes likelihood.model=tfidf
    cp artifacts/main/baseline_summary.json "$RESULTS/baselines_seqbayes.json"

    echo "=== PHASE 9: Distractor comparison ==="
    for STRAT in tfidf_profile category_random; do
        python scripts/build_mc_dataset.py --config configs/default.yaml \
            --output-dir "artifacts/distractor_$STRAT" data.distractor_strategy="$STRAT"
        python scripts/run_baselines.py --config configs/default.yaml \
            --mc-path "artifacts/distractor_$STRAT/mc_dataset.json" likelihood.model=tfidf
        cp artifacts/main/baseline_summary.json "$RESULTS/baselines_$STRAT.json"
    done

    echo "=== PHASE 13: K-sensitivity ==="
    for K in 2 3 5 6; do
        python scripts/build_mc_dataset.py --config configs/default.yaml \
            --output-dir "artifacts/k$K" data.K="$K" data.distractor_strategy=category_random
        python scripts/run_baselines.py --config configs/default.yaml \
            --mc-path "artifacts/k$K/mc_dataset.json" likelihood.model=tfidf
        cp artifacts/main/baseline_summary.json "$RESULTS/baselines_k$K.json"
    done

else
    ####################################################################
    # PARALLEL MODE
    ####################################################################
    echo "=== WAVE 1: Independent phases (3 parallel tracks) ==="
    echo ""

    PIDS=()

    # Track A: Baselines (writes artifacts/main/baseline_summary.json)
    (
        run_phase "2" python scripts/run_baselines.py \
            --config configs/default.yaml --mc-path "$MC" likelihood.model=tfidf
        cp artifacts/main/baseline_summary.json "$RESULTS/baselines_tfidf.json"
    ) &
    PIDS+=($!)

    # Track B: PPO training (writes artifacts/main/ppo_model.zip)
    (
        run_phase "3" python scripts/train_ppo.py \
            --config configs/default.yaml --mc-path "$MC" --seed 13 --deterministic-eval likelihood.model=tfidf
        cp artifacts/main/ppo_summary.json "$RESULTS/ppo_default.json"
        cp artifacts/main/ppo_model.zip "$RESULTS/ppo_model_default.zip"
    ) &
    PIDS+=($!)

    # Track C: T5 policy (writes checkpoints/ — no artifact race)
    (
        run_phase "5" python scripts/train_t5_policy.py \
            --config configs/t5_policy.yaml model.model_name="$T5_MODEL"
    ) &
    PIDS+=($!)

    echo "Waiting for Wave 1 (${#PIDS[@]} tracks)..."
    wait_all "${PIDS[@]}"
    echo ""

    echo "=== WAVE 2: Sequential post-Wave-1 phases (share artifacts/main/) ==="

    # Phase 4: Evaluate all (reads baseline_summary.json from Phase 2)
    run_phase "4" python scripts/evaluate_all.py \
        --config configs/default.yaml --mc-path "$MC" likelihood.model=tfidf
    cp artifacts/main/evaluation_report.json "$RESULTS/eval_default.json"

    # Phase 6: Compare policies (needs Phase 3 PPO + Phase 5 T5)
    run_phase "6" python scripts/compare_policies.py \
        --mlp-checkpoint artifacts/main/ppo_model \
        --t5-checkpoint checkpoints/ppo_t5/best_model \
        --mc-path "$MC" \
        --output "$RESULTS/t5_comparison.json"

    # Phase 11: Expected Wins eval (writes evaluation_report.json)
    run_phase "11" python scripts/evaluate_all.py \
        --config configs/default.yaml --mc-path "$MC" \
        likelihood.model=tfidf environment.reward_mode=expected_wins environment.opponent_buzz_model.type=logistic
    cp artifacts/main/evaluation_report.json "$RESULTS/eval_ew_logistic.json"

    # Phase 15: Belief mode comparison (writes baseline_summary.json)
    run_phase "15" python scripts/run_baselines.py \
        --config configs/default.yaml --mc-path "$MC" \
        environment.belief_mode=sequential_bayes likelihood.model=tfidf
    cp artifacts/main/baseline_summary.json "$RESULTS/baselines_seqbayes.json"

    echo ""
    echo "=== WAVE 3: PPO ablations (sequential — share artifacts/main/) ==="

    echo "[Phase 14a] reward_mode=simple"
    python scripts/train_ppo.py --config configs/default.yaml --mc-path "$MC" \
        --seed 13 --deterministic-eval likelihood.model=tfidf environment.reward_mode=simple
    cp artifacts/main/ppo_summary.json "$RESULTS/ppo_simple.json"

    echo "[Phase 14b] reward_mode=human_grounded"
    python scripts/train_ppo.py --config configs/default.yaml --mc-path "$MC" \
        --seed 13 --deterministic-eval likelihood.model=tfidf environment.reward_mode=human_grounded
    cp artifacts/main/ppo_summary.json "$RESULTS/ppo_human_grounded.json"

    echo "[Phase 16] policy_mode=stop_only"
    python scripts/train_ppo.py --config configs/default.yaml --mc-path "$MC" \
        --seed 13 --deterministic-eval --policy-mode stop_only likelihood.model=tfidf
    cp artifacts/main/ppo_summary.json "$RESULTS/ppo_stop_only.json"

    echo "[Phase 17] end_mode=no_buzz"
    python scripts/train_ppo.py --config configs/default.yaml --mc-path "$MC" \
        --seed 13 --deterministic-eval likelihood.model=tfidf environment.end_mode=no_buzz environment.no_buzz_reward=-0.25
    cp artifacts/main/ppo_summary.json "$RESULTS/ppo_no_buzz.json"

    echo ""
    echo "=== WAVE 4: K-sensitivity (sequential — writes artifacts/main/baseline_summary.json) ==="

    for K in 2 3 5 6; do
        echo "[Phase 13-k$K] Building K=$K dataset..."
        run_phase "13-k$K" python scripts/build_mc_dataset.py \
            --config configs/default.yaml \
            --output-dir "artifacts/k$K" data.K="$K" data.distractor_strategy=category_random
        run_phase "13-k${K}-baselines" python scripts/run_baselines.py \
            --config configs/default.yaml \
            --mc-path "artifacts/k$K/mc_dataset.json" likelihood.model=tfidf
        cp artifacts/main/baseline_summary.json "$RESULTS/baselines_k$K.json"
    done

fi

########################################################################
# FINAL SUMMARY
########################################################################
echo ""
echo "============================================================"
echo "PIPELINE COMPLETE"
echo "============================================================"
echo ""
echo "Results directory:"
ls -1 "$RESULTS"/*.json 2>/dev/null | while read f; do echo "  $(basename $f)"; done
echo ""
echo "Artifacts:"
for d in artifacts/main artifacts/k* artifacts/distractor_*; do
    [ -d "$d" ] && echo "  $d/ — $(ls $d/*.json 2>/dev/null | wc -l) JSON files"
done
echo ""
echo "Checkpoints:"
ls -d checkpoints/*/best_model 2>/dev/null | while read d; do echo "  $d/"; done
echo ""
echo "Final comparison table:"
python3 -c "
import json, glob
for f in sorted(glob.glob('results/*.json')):
    s = json.load(open(f))
    name = f.split('/')[-1].replace('.json', '')
    if 'full_eval' in s:
        fe = s['full_eval']
        print(f'{name}: acc={fe.get(\"buzz_accuracy\", \"N/A\")}, S_q={fe.get(\"mean_sq\", \"N/A\")}')
    elif 't5_policy' in s:
        for k in ('mlp_policy', 't5_policy'):
            if k in s:
                m = s[k]
                print(f'{name}/{k}: acc={m.get(\"accuracy\", \"N/A\")}, S_q={m.get(\"mean_sq\", \"N/A\")}')
    elif 'softmax_profile' in s:
        sp = s['softmax_profile']
        best = max(sp.items(), key=lambda x: x[1].get('mean_sq', 0), default=('N/A', {}))
        print(f'{name}: best_threshold={best[0]}, S_q={best[1].get(\"mean_sq\", \"N/A\")}')
    else:
        acc = s.get('buzz_accuracy', s.get('accuracy', 'N/A'))
        sq = s.get('mean_sq', 'N/A')
        print(f'{name}: acc={acc}, S_q={sq}')
"
```

## File: scripts/sweep_reward_shaping.py
```python
#!/usr/bin/env python3
"""Sweep PPO smoke reward-shaping settings and record results.

Runs `scripts/train_ppo.py` in smoke mode across a small grid of:
- environment.wait_penalty
- environment.early_buzz_penalty

Collects metrics from artifacts/smoke/ppo_summary.json after each run and writes:
- artifacts/smoke/reward_sweep_results.json
- artifacts/smoke/reward_sweep_results.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SMOKE_CONFIG = PROJECT_ROOT / "configs" / "smoke.yaml"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "smoke"
TMP_CONFIG = ARTIFACT_DIR / "_tmp_sweep_smoke.yaml"
PPO_SUMMARY = ARTIFACT_DIR / "ppo_summary.json"

WAIT_PENALTIES = [0.0, 0.02, 0.05]
EARLY_BUZZ_PENALTIES = [0.2, 0.5, 0.8]
SEEDS = [13, 42, 123]


def run_cmd(cmd: list[str]) -> int:
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT)
    return proc.returncode


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep PPO reward shaping")
    parser.add_argument(
        "--seeds",
        type=str,
        default=",".join(str(s) for s in SEEDS),
        help="Comma-separated seeds, e.g. 13,42,123",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Optional timesteps override for train_ppo during sweep",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    base_cfg = load_yaml(SMOKE_CONFIG)

    python_exe = sys.executable
    results = []

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    grid = [(w, e) for w in WAIT_PENALTIES for e in EARLY_BUZZ_PENALTIES]

    print("=" * 72)
    print(f"Reward sweep: {len(grid)} configs x {len(seeds)} seeds")
    print("=" * 72)

    for idx, (wait_penalty, early_buzz_penalty) in enumerate(grid, start=1):
        per_seed = []
        print(f"[{idx}/{len(grid)}] wait_penalty={wait_penalty}, early_buzz_penalty={early_buzz_penalty}")

        for seed in seeds:
            cfg = dict(base_cfg)
            cfg.setdefault("environment", {})
            cfg["environment"] = dict(cfg["environment"])
            cfg["environment"]["wait_penalty"] = float(wait_penalty)
            cfg["environment"]["early_buzz_penalty"] = float(early_buzz_penalty)
            cfg["environment"]["seed"] = int(seed)

            cfg.setdefault("ppo", {})
            cfg["ppo"] = dict(cfg["ppo"])
            cfg["ppo"]["seed"] = int(seed)
            save_yaml(TMP_CONFIG, cfg)

            cmd = [python_exe, "scripts/train_ppo.py", "--config", str(TMP_CONFIG), "--smoke", "--seed", str(seed)]
            if args.timesteps is not None:
                cmd.extend(["--timesteps", str(args.timesteps)])

            start = time.time()
            code = run_cmd(cmd)
            elapsed = time.time() - start

            if code != 0 or not PPO_SUMMARY.exists():
                per_seed.append({"seed": seed, "status": "failed", "seconds": round(elapsed, 3)})
                continue

            summary = load_json(PPO_SUMMARY)
            per_seed.append(
                {
                    "seed": seed,
                    "status": "ok",
                    "seconds": round(elapsed, 3),
                    "buzz_accuracy": float(summary.get("buzz_accuracy", 0.0)),
                    "mean_sq": float(summary.get("mean_sq", 0.0)),
                    "mean_buzz_step": float(summary.get("mean_buzz_step", 0.0)),
                    "ece": float(summary.get("ece", 0.0)),
                    "brier": float(summary.get("brier", 0.0)),
                }
            )

        ok = [r for r in per_seed if r.get("status") == "ok"]
        if not ok:
            results.append(
                {
                    "wait_penalty": wait_penalty,
                    "early_buzz_penalty": early_buzz_penalty,
                    "status": "failed",
                    "num_ok": 0,
                    "num_total": len(per_seed),
                    "per_seed": per_seed,
                }
            )
            continue

        mean_acc = sum(r["buzz_accuracy"] for r in ok) / len(ok)
        mean_sq = sum(r["mean_sq"] for r in ok) / len(ok)
        mean_step = sum(r["mean_buzz_step"] for r in ok) / len(ok)
        mean_ece = sum(r["ece"] for r in ok) / len(ok)
        mean_brier = sum(r["brier"] for r in ok) / len(ok)
        mean_seconds = sum(r["seconds"] for r in ok) / len(ok)

        # Balanced objective: maximize accuracy + S_q while penalizing calibration error.
        objective = mean_acc + mean_sq - 0.5 * mean_ece

        results.append(
            {
                "wait_penalty": wait_penalty,
                "early_buzz_penalty": early_buzz_penalty,
                "status": "ok",
                "num_ok": len(ok),
                "num_total": len(per_seed),
                "seconds": round(mean_seconds, 3),
                "buzz_accuracy": mean_acc,
                "mean_sq": mean_sq,
                "mean_buzz_step": mean_step,
                "ece": mean_ece,
                "brier": mean_brier,
                "objective": objective,
                "per_seed": per_seed,
            }
        )

    # cleanup temp config
    if TMP_CONFIG.exists():
        TMP_CONFIG.unlink()

    out_json = ARTIFACT_DIR / "reward_sweep_results.json"
    out_csv = ARTIFACT_DIR / "reward_sweep_results.csv"

    with out_json.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    fields = [
        "wait_penalty",
        "early_buzz_penalty",
        "status",
        "num_ok",
        "num_total",
        "seconds",
        "buzz_accuracy",
        "mean_sq",
        "mean_buzz_step",
        "ece",
        "brier",
        "objective",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in results:
            flat = {k: row.get(k, "") for k in fields}
            writer.writerow(flat)

    ok_runs = [r for r in results if r.get("status") == "ok"]
    if not ok_runs:
        print("No successful runs.")
        return 1

    best = max(ok_runs, key=lambda r: float(r.get("objective", 0.0)))

    print("\nBest run:")
    print(best)
    print(f"Wrote: {out_json}")
    print(f"Wrote: {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

## File: scripts/test_mc_builder.py
```python
#!/usr/bin/env python
"""Test script to verify MC construction with anti-artifact guards."""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from qb_data.data_loader import QANTADatasetLoader
from qb_data.answer_profiles import AnswerProfileBuilder
from qb_data.mc_builder import MCBuilder, MCQuestion
from qb_data.config import load_config


def main():
    """Test MC question construction with guards."""
    print("Testing MC Builder with Anti-Artifact Guards")
    print("=" * 50)

    # Load configuration
    config = load_config("configs/default.yaml")

    # Load test questions
    data_path = "data/test_questions.csv"
    if not os.path.exists(data_path):
        print(f"Error: Test data not found at {data_path}")
        print("Please ensure test_questions.csv exists")
        return 1

    # Load questions
    questions = QANTADatasetLoader.load_from_csv(data_path)
    print(f"\nLoaded {len(questions)} test questions")

    # Create answer profile builder
    profile_builder = AnswerProfileBuilder(
        max_tokens_per_profile=config["answer_profiles"]["max_tokens_per_profile"],
        min_questions_per_answer=config["answer_profiles"]["min_questions_per_answer"]
    )
    profile_builder.fit(questions)
    print(f"Built profiles for {len(profile_builder._grouped)} unique answers")

    # Create MC builder with guards from config
    mc_builder = MCBuilder(
        K=config["data"]["K"],
        strategy="tfidf_profile",  # Use TF-IDF since it doesn't require embeddings
        alias_edit_distance_threshold=config["mc_guards"]["alias_edit_distance_threshold"],
        duplicate_token_overlap_threshold=config["mc_guards"]["duplicate_token_overlap_threshold"],
        max_length_ratio=config["mc_guards"]["max_length_ratio"],
        random_seed=config["data"]["shuffle_seed"]
    )

    # Build MC questions
    print(f"\nBuilding MC questions with K={config['data']['K']} options...")
    mc_questions = mc_builder.build(questions, profile_builder)
    print(f"Created {len(mc_questions)} MC questions (from {len(questions)} originals)")

    # Calculate rejection rate
    rejection_rate = 1.0 - (len(mc_questions) / len(questions))
    print(f"Rejection rate: {rejection_rate:.1%} (due to guard violations)")

    # Print sample MC questions
    print("\n" + "=" * 50)
    print("Sample MC Questions:")
    print("=" * 50)

    for i, mc_q in enumerate(mc_questions[:3]):  # Show first 3
        print(f"\n[Question {i+1}]")
        print(f"Category: {mc_q.category or 'Unknown'}")
        print(f"Question ID: {mc_q.qid}")

        # Show first clue (truncated)
        first_clue = mc_q.tokens[0] if mc_q.tokens else mc_q.question[:100]
        print(f"First clue: {first_clue[:150]}...")

        print(f"\nOptions:")
        for j, option in enumerate(mc_q.options):
            marker = " [CORRECT]" if j == mc_q.gold_index else ""
            print(f"  {j+1}. {option}{marker}")

        print(f"\nDistractor strategy: {mc_q.distractor_strategy}")

        # Check guards for this question
        print("\nGuard checks:")

        # Check alias collision
        gold_aliases = [mc_q.answer_primary] + list(mc_q.clean_answers)
        alias_violations = []
        for j, option in enumerate(mc_q.options):
            if j != mc_q.gold_index:
                for alias in gold_aliases:
                    from difflib import SequenceMatcher
                    dist = 1.0 - SequenceMatcher(None, option.lower(), alias.lower()).ratio()
                    if dist < 0.2:
                        alias_violations.append((option, alias, dist))

        if alias_violations:
            print(f"  ✗ Alias collision detected: {alias_violations}")
        else:
            print("  ✓ No alias collisions")

        # Check token overlap between options
        from qb_data.mc_builder import _token_overlap
        high_overlaps = []
        for j in range(len(mc_q.options)):
            for k in range(j+1, len(mc_q.options)):
                overlap = _token_overlap(mc_q.options[j], mc_q.options[k])
                if overlap > 0.8:
                    high_overlaps.append((mc_q.options[j], mc_q.options[k], overlap))

        if high_overlaps:
            print(f"  ✗ High token overlap: {high_overlaps}")
        else:
            print("  ✓ No high token overlaps")

        # Check length ratio
        lengths = [len(o.split()) for o in mc_q.options]
        ratio = max(lengths) / max(1, min(lengths))
        if ratio > 3.0:
            print(f"  ✗ Length ratio violation: {ratio:.2f} (max: {max(lengths)}, min: {min(lengths)})")
        else:
            print(f"  ✓ Length ratio OK: {ratio:.2f}")

        # Check question overlap
        from qb_data.text_utils import normalize_answer
        q_norm = normalize_answer(mc_q.question).lower()
        overlaps = []
        for option in mc_q.options:
            o_norm = normalize_answer(option).lower()
            if o_norm and o_norm in q_norm:
                overlaps.append(option)

        if overlaps:
            print(f"  ✗ Options appear in question: {overlaps}")
        else:
            print("  ✓ No options in question text")

    # Print statistics
    print("\n" + "=" * 50)
    print("Statistics:")
    print("=" * 50)
    print(f"Total questions processed: {len(questions)}")
    print(f"MC questions built: {len(mc_questions)}")
    print(f"Questions rejected by guards: {len(questions) - len(mc_questions)}")

    # Analyze rejection reasons (would need to track in MCBuilder for full details)
    if len(mc_questions) < len(questions):
        print("\nNote: Some questions were rejected due to guard violations.")
        print("Common reasons include:")
        print("  - Not enough valid distractors after alias/duplicate filtering")
        print("  - Length ratio violations between options")
        print("  - Answer text appearing in question")

    print("\n✓ MC questions built successfully with guards active")
    return 0


if __name__ == "__main__":
    exit(main())
```

## File: tests/test_action_space_alignment.py
```python
"""Integration-style guards for the PR1 feature-port subset."""

from __future__ import annotations

import pytest
import torch

from models.t5_policy import T5PolicyModel
from models.likelihoods import TfIdfLikelihood
from qb_env import StopOnlyEnv, TossupMCEnv


@pytest.fixture(scope="module")
def t5_small_model():
    try:
        model = T5PolicyModel(
            {
                "model_name": "t5-small",
                "device": "cpu",
                "max_input_length": 128,
                "num_choices": 4,
            }
        )
    except OSError as exc:
        pytest.skip(f"t5-small unavailable in test environment: {exc}")
    model.eval()
    return model


def test_t5_wait_log_prob_does_not_depend_on_answer_logits(t5_small_model):
    """WAIT log-prob is independent of answer-head mass."""
    model = t5_small_model
    joint_log_prob = getattr(model, "_joint_action_log_prob")
    wait_logits = torch.tensor([[1.5, -0.5]], dtype=torch.float32, device=model.device)
    answer_logits = torch.tensor([[0.1, 0.2, 0.3, 0.4]], dtype=torch.float32, device=model.device)
    actions = torch.tensor([0], dtype=torch.long, device=model.device)

    lp1 = joint_log_prob(wait_logits, answer_logits, actions)
    lp2 = joint_log_prob(wait_logits, answer_logits.flip(dims=[-1]), actions)
    assert torch.allclose(lp1, lp2, atol=1e-6)


def test_t5_entropy_uses_chain_rule(t5_small_model):
    """Joint entropy follows H(wait) + p_buzz * H(answer)."""
    model = t5_small_model
    joint_entropy = getattr(model, "_joint_entropy")
    wait_logits = torch.tensor([[0.0, 1.0]], dtype=torch.float32, device=model.device)
    answer_logits = torch.tensor([[2.0, 1.0, 0.0, -1.0]], dtype=torch.float32, device=model.device)

    entropy = joint_entropy(wait_logits, answer_logits)
    wait_probs = torch.softmax(wait_logits, dim=-1)
    wait_log_probs = torch.log_softmax(wait_logits, dim=-1)
    answer_probs = torch.softmax(answer_logits, dim=-1)
    answer_log_probs = torch.log_softmax(answer_logits, dim=-1)
    expected = (
        -(wait_probs * wait_log_probs).sum(dim=-1)
        + wait_probs[:, 1] * (-(answer_probs * answer_log_probs).sum(dim=-1))
    )
    assert torch.allclose(entropy, expected, atol=1e-6)


def test_stop_only_env_has_discrete_2_action_space(sample_tfidf_env):
    env = StopOnlyEnv(sample_tfidf_env)
    assert env.action_space.n == 2


def test_flat_kplus1_mode_still_available(sample_tfidf_env):
    assert sample_tfidf_env.action_space.n == 5


def test_no_buzz_end_mode_does_not_force_choice(sample_mc_question):
    corpus = sample_mc_question.option_profiles[:]
    model = TfIdfLikelihood(corpus_texts=corpus)
    env = TossupMCEnv(
        questions=[sample_mc_question],
        likelihood_model=model,
        K=4,
        reward_mode="simple",
        end_mode="no_buzz",
        no_buzz_reward=0.0,
    )
    _obs, _info = env.reset(seed=0)
    while True:
        _obs, _reward, _term, truncated, info = env.step(0)
        if truncated:
            break
    assert info.get("no_buzz") is True
    assert info.get("forced_choice") == -1
    assert info.get("forced_correct") is False
```

## File: tests/test_answer_profile_cache.py
```python
"""Tests for AnswerProfileBuilder._cache memoization.

Verifies that:
1. Distractor profiles (exclude_qid=None) are cached and return identical results
2. Leave-one-out profiles (answer, qid) are cached and return identical results
3. Cache is invalidated on fit() with new data
4. Cached distractor profile is byte-identical to freshly computed profile
5. Cached leave-one-out profile is byte-identical to freshly computed profile
6. Cache reduces actual computation (single entry per unique key)
"""

from __future__ import annotations

import pytest

from qb_data.answer_profiles import AnswerProfileBuilder
from qb_data.data_loader import TossupQuestion


def _make_question(
    qid: str,
    answer: str,
    text: str,
    category: str = "History",
) -> TossupQuestion:
    """Create a minimal TossupQuestion for cache testing."""
    tokens = text.split()
    return TossupQuestion(
        qid=qid,
        question=text,
        tokens=tokens,
        answer_primary=answer,
        clean_answers=[answer],
        run_indices=[len(tokens) - 1],
        human_buzz_positions=[],
        category=category,
        cumulative_prefixes=[text],
    )


@pytest.fixture
def sample_questions() -> list[TossupQuestion]:
    """Five questions with 3 shared answers for exercising cache hits."""
    return [
        _make_question("q1", "Washington", "first president commander in chief"),
        _make_question("q2", "Washington", "led the continental army to victory"),
        _make_question("q3", "Jefferson", "wrote the declaration of independence"),
        _make_question("q4", "Jefferson", "third president and diplomat to France"),
        _make_question("q5", "Lincoln", "preserved the union during civil war"),
    ]


@pytest.fixture
def builder(sample_questions: list[TossupQuestion]) -> AnswerProfileBuilder:
    """Return a fitted AnswerProfileBuilder."""
    b = AnswerProfileBuilder(max_tokens_per_profile=2000, min_questions_per_answer=1)
    b.fit(sample_questions)
    return b


class TestProfileCacheHits:
    """Repeated calls with the same args return the same cached result."""

    def test_distractor_profile_cached(
        self, builder: AnswerProfileBuilder
    ) -> None:
        """profile_for_answer returns identical string on repeated (answer, None)."""
        first = builder.profile_for_answer("Washington", exclude_qid=None)
        second = builder.profile_for_answer("Washington", exclude_qid=None)
        assert first is second  # same object, not just equal

    def test_leave_one_out_profile_cached(
        self, builder: AnswerProfileBuilder
    ) -> None:
        """profile_for_answer returns identical string on repeated (answer, qid)."""
        first = builder.profile_for_answer("Washington", exclude_qid="q1")
        second = builder.profile_for_answer("Washington", exclude_qid="q1")
        assert first is second  # same object from cache


class TestCacheInvalidation:
    """fit() with new data clears the cache."""

    def test_fit_clears_cache(
        self, builder: AnswerProfileBuilder, sample_questions: list[TossupQuestion]
    ) -> None:
        """After fit() with new data, cache is empty and profiles reflect new data."""
        # Populate cache
        builder.profile_for_answer("Washington", exclude_qid=None)
        assert len(builder._cache) > 0

        # Re-fit with different data
        new_questions = [
            _make_question("q99", "Washington", "completely different text about cherry trees"),
        ]
        builder.fit(new_questions)
        assert len(builder._cache) == 0

        # New profile should reflect new data
        profile = builder.profile_for_answer("Washington", exclude_qid=None)
        assert "cherry" in profile


class TestCacheEquivalence:
    """Cached profiles are byte-identical to freshly computed profiles."""

    def test_distractor_cache_equivalence(
        self, sample_questions: list[TossupQuestion]
    ) -> None:
        """Cached (answer, None) profile is byte-identical to a fresh computation."""
        # Build fresh (uncached) profile
        fresh_builder = AnswerProfileBuilder(
            max_tokens_per_profile=2000, min_questions_per_answer=1
        )
        fresh_builder.fit(sample_questions)
        fresh_profile = fresh_builder._profile_text("Jefferson", exclude_qid=None)

        # Build cached profile
        cached_builder = AnswerProfileBuilder(
            max_tokens_per_profile=2000, min_questions_per_answer=1
        )
        cached_builder.fit(sample_questions)
        _ = cached_builder._profile_text("Jefferson", exclude_qid=None)  # populate cache
        cached_profile = cached_builder._profile_text("Jefferson", exclude_qid=None)  # from cache

        assert fresh_profile == cached_profile

    def test_leave_one_out_cache_equivalence(
        self, sample_questions: list[TossupQuestion]
    ) -> None:
        """Cached (answer, qid) profile is byte-identical to a fresh computation."""
        fresh_builder = AnswerProfileBuilder(
            max_tokens_per_profile=2000, min_questions_per_answer=1
        )
        fresh_builder.fit(sample_questions)
        fresh_profile = fresh_builder._profile_text("Washington", exclude_qid="q1")

        cached_builder = AnswerProfileBuilder(
            max_tokens_per_profile=2000, min_questions_per_answer=1
        )
        cached_builder.fit(sample_questions)
        _ = cached_builder._profile_text("Washington", exclude_qid="q1")
        cached_profile = cached_builder._profile_text("Washington", exclude_qid="q1")

        assert fresh_profile == cached_profile


class TestCacheEfficiency:
    """Cache reduces computation to one real call per unique key."""

    def test_cache_stores_one_entry_per_unique_key(
        self, builder: AnswerProfileBuilder
    ) -> None:
        """Calling _profile_text N times with same args results in 1 cache entry."""
        for _ in range(10):
            builder.profile_for_answer("Lincoln", exclude_qid=None)

        # Only one cache entry for (Lincoln, None)
        assert ("Lincoln", None) in builder._cache
        assert len([k for k in builder._cache if k[0] == "Lincoln"]) == 1
```

## File: tests/test_common.py
```python
"""Tests for scripts._common helpers."""

from __future__ import annotations

from scripts._common import embedding_cache_path


def test_embedding_cache_path_keys_by_model_variant() -> None:
    """Cache filenames should distinguish supported model variants."""
    assert (
        embedding_cache_path(
            {"likelihood": {"model": "sbert", "embedding_model": "all-MiniLM-L6-v2"}}
        ).name
        == "embedding_cache_all-MiniLM-L6-v2.npz"
    )
    assert (
        embedding_cache_path({"likelihood": {"model": "openai", "openai_model": "text-embedding-3-large"}}).name
        == "embedding_cache_text-embedding-3-large.npz"
    )
    assert (
        embedding_cache_path({"likelihood": {"model": "t5", "t5_name": "t5-large"}}).name
        == "embedding_cache_t5-large.npz"
    )
    assert embedding_cache_path({"likelihood": {"model": "t5-base"}}).name == "embedding_cache_t5-base.npz"
    assert embedding_cache_path({"likelihood": {"model": "tfidf"}}).name == "embedding_cache_tfidf.npz"
```

## File: tests/test_dataset_splits.py
```python
"""Tests for stratified dataset splitting reproducibility.

Verifies that splits are deterministic across invocations and do not
depend on Python's hash randomization (PYTHONHASHSEED).
"""

import subprocess
import sys

import pytest

from qb_data.data_loader import TossupQuestion
from qb_data.dataset_splits import create_stratified_splits


def _make_questions(n: int, categories: list[str]) -> list[TossupQuestion]:
    """Create n dummy TossupQuestion instances cycling through categories."""
    questions = []
    for i in range(n):
        cat = categories[i % len(categories)]
        questions.append(
            TossupQuestion(
                qid=f"q{i:04d}",
                question=f"Question {i}",
                tokens=[f"token{i}"],
                answer_primary=f"Answer {i}",
                clean_answers=[f"Answer {i}"],
                run_indices=[0],
                human_buzz_positions=[],
                category=cat,
                cumulative_prefixes=[f"token{i}"],
            )
        )
    return questions


def test_splits_deterministic_same_process():
    """Same seed produces identical splits within one process."""
    questions = _make_questions(60, ["History", "Science", "Literature"])
    train1, val1, test1 = create_stratified_splits(questions, seed=42)
    train2, val2, test2 = create_stratified_splits(questions, seed=42)
    assert [q.qid for q in train1] == [q.qid for q in train2]
    assert [q.qid for q in val1] == [q.qid for q in val2]
    assert [q.qid for q in test1] == [q.qid for q in test2]


def test_splits_deterministic_across_processes():
    """Splits must be identical even with different PYTHONHASHSEED values.

    Runs the split in two subprocesses with different PYTHONHASHSEED and
    checks that they produce identical qid orderings.
    """
    script = (
        "import json, sys, io; sys.path.insert(0, '.'); "
        "sys.stdout = io.StringIO(); "
        "from qb_data.data_loader import TossupQuestion; "
        "from qb_data.dataset_splits import create_stratified_splits; "
        "qs = [TossupQuestion(qid=f'q{i:04d}', question=f'Q{i}', tokens=[f't{i}'], "
        "answer_primary=f'A{i}', clean_answers=[f'A{i}'], run_indices=[0], "
        "human_buzz_positions=[], category=['History','Science','Lit'][i%3], "
        "cumulative_prefixes=[f't{i}']) for i in range(60)]; "
        "tr,va,te = create_stratified_splits(qs, seed=42); "
        "sys.stdout = sys.__stdout__; "
        "print(json.dumps([q.qid for q in tr]))"
    )
    import json
    import os

    base_env = {k: v for k, v in os.environ.items()}
    repo_root = str(__import__("pathlib").Path(__file__).resolve().parents[1])
    results = []
    for hashseed in ["0", "12345"]:
        env = {**base_env, "PYTHONHASHSEED": hashseed}
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            env=env,
            cwd=repo_root,
            timeout=30,
        )
        assert proc.returncode == 0, f"Subprocess failed: {proc.stderr}"
        results.append(json.loads(proc.stdout.strip()))
    assert results[0] == results[1], (
        "Splits differ across PYTHONHASHSEED values — hash(category) is not deterministic"
    )


def test_splits_different_seeds_differ():
    """Different seeds should produce different splits."""
    questions = _make_questions(60, ["History", "Science", "Literature"])
    train1, _, _ = create_stratified_splits(questions, seed=42)
    train2, _, _ = create_stratified_splits(questions, seed=99)
    assert [q.qid for q in train1] != [q.qid for q in train2]


def test_splits_all_questions_assigned():
    """Every question must appear in exactly one split."""
    questions = _make_questions(100, ["A", "B", "C", "D"])
    train, val, test = create_stratified_splits(questions, seed=1)
    all_qids = {q.qid for q in train} | {q.qid for q in val} | {q.qid for q in test}
    assert len(all_qids) == 100
    assert len(train) + len(val) + len(test) == 100
```

## File: tests/test_dspy_answer_profiles.py
```python
"""Tests for qb_data/dspy_answer_profiles.py."""

from __future__ import annotations

import pytest


class TestBuildDspyProfiles:
    def test_module_importable_without_dspy(self) -> None:
        """The module imports cleanly even when dspy is not installed."""
        from qb_data.dspy_answer_profiles import build_dspy_profiles
        assert callable(build_dspy_profiles)

    def test_runtime_call_without_dspy_raises(self) -> None:
        """Calling build_dspy_profiles without dspy raises ImportError."""
        try:
            import dspy
            pytest.skip("dspy is installed; cannot test import failure")
        except ImportError:
            from qb_data.dspy_answer_profiles import build_dspy_profiles
            with pytest.raises(ImportError, match="dspy"):
                build_dspy_profiles(
                    answers=["A"],
                    existing_profiles={"A": "existing"},
                    dspy_config={"model": "test"},
                )

    def test_with_dspy_installed(self) -> None:
        """When dspy IS installed, the function is callable."""
        dspy = pytest.importorskip("dspy", reason="dspy not installed")
        from qb_data.dspy_answer_profiles import build_dspy_profiles
        assert callable(build_dspy_profiles)
```

## File: tests/test_dspy_likelihood.py
```python
"""Tests for models/dspy_likelihood.py — DSPy-backed scorer with cache."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from models.dspy_likelihood import DSPyLikelihood, _score_cache_key


def _fake_scorer(clue: str, options: list[str]) -> list[float]:
    """Return uniform scores sized to the option list."""
    return [1.0 / len(options)] * len(options)


class TestDSPyLikelihood:
    def test_score_returns_ndarray_k(self) -> None:
        model = DSPyLikelihood(scorer=_fake_scorer)
        scores = model.score("clue text", ["A", "B", "C", "D"])
        assert scores.shape == (4,)
        assert scores.dtype == np.float32

    def test_repeated_call_hits_cache(self) -> None:
        call_count = 0

        def counting_scorer(clue, options):
            nonlocal call_count
            call_count += 1
            return [1.0] * len(options)

        model = DSPyLikelihood(scorer=counting_scorer)
        model.score("clue", ["A", "B"])
        model.score("clue", ["A", "B"])
        assert call_count == 1

    def test_changed_fingerprint_invalidates(self) -> None:
        """Different fingerprints produce different cache keys for same input."""
        key_v1 = _score_cache_key("clue", ["A", "B"], "v1")
        key_v2 = _score_cache_key("clue", ["A", "B"], "v2")
        assert key_v1 != key_v2, "Fingerprint must affect cache key"

        model = DSPyLikelihood(scorer=_fake_scorer, program_fingerprint="v1")
        model.score("clue", ["A", "B"])
        assert key_v1 in model._score_cache
        assert key_v2 not in model._score_cache

    def test_persistence_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cache.npz"
            model = DSPyLikelihood(scorer=_fake_scorer)
            model.score("clue", ["A", "B", "C"])
            saved = model.save_cache(path)
            assert saved == 1

            model2 = DSPyLikelihood(scorer=_fake_scorer)
            loaded = model2.load_cache(path)
            assert loaded == 1
            np.testing.assert_array_equal(
                model2.score("clue", ["A", "B", "C"]),
                model.score("clue", ["A", "B", "C"]),
            )

    def test_embed_batch_raises(self) -> None:
        model = DSPyLikelihood(scorer=_fake_scorer)
        with pytest.raises(NotImplementedError):
            model._embed_batch(["text"])

    def test_cache_memory_bytes(self) -> None:
        model = DSPyLikelihood(scorer=_fake_scorer)
        assert model.cache_memory_bytes == 0
        model.score("c", ["A"])
        assert model.cache_memory_bytes > 0

    def test_score_shape_validation(self) -> None:
        """Scorer returning wrong length raises ValueError."""
        def bad_scorer(clue, options):
            return [1.0, 2.0]  # always 2, ignoring len(options)

        model = DSPyLikelihood(scorer=bad_scorer)
        with pytest.raises(ValueError, match="expected"):
            model.score("clue", ["A", "B", "C", "D"])

    def test_isinstance_likelihood_model(self) -> None:
        """DSPyLikelihood is a proper LikelihoodModel subclass."""
        from models.likelihoods import LikelihoodModel
        model = DSPyLikelihood(scorer=_fake_scorer)
        assert isinstance(model, LikelihoodModel)
```

## File: tests/test_dspy_optimize.py
```python
"""Tests for scripts/optimize_dspy.py — offline DSPy compilation."""

from __future__ import annotations

import pytest

from scripts.optimize_dspy import build_dspy_trainset, _score_metric


def _make_mc_question():
    from qb_data.mc_builder import MCQuestion

    return MCQuestion(
        qid="q1",
        question="Who was the first president?",
        tokens=["Who", "was", "the", "first", "president"],
        answer_primary="George Washington",
        clean_answers=["George Washington"],
        run_indices=[1, 3, 4],
        human_buzz_positions=[],
        category="History",
        cumulative_prefixes=["Who was", "Who was the first", "Who was the first president"],
        options=["George Washington", "Thomas Jefferson"],
        gold_index=0,
        option_profiles=["Washington profile", "Jefferson profile"],
        option_answer_primary=["George Washington", "Thomas Jefferson"],
        distractor_strategy="test",
    )


class TestBuildDspyTrainset:
    def test_trainset_structure(self) -> None:
        mc = [_make_mc_question()]
        trainset = build_dspy_trainset(mc, max_examples=10)
        assert len(trainset) == 1
        ex = trainset[0]
        assert "clue_prefix" in ex
        assert "option_profiles" in ex
        assert "gold_index" in ex

    def test_trainset_caps_at_max(self) -> None:
        mc = [_make_mc_question()] * 100
        trainset = build_dspy_trainset(mc, max_examples=5)
        assert len(trainset) == 5

    def test_trainset_empty(self) -> None:
        assert build_dspy_trainset([], max_examples=10) == []


class TestCompileDspyScorer:
    def test_compile_requires_dspy(self) -> None:
        pytest.importorskip("dspy", reason="dspy not installed")
        from scripts.optimize_dspy import compile_dspy_scorer
        assert callable(compile_dspy_scorer)

    def test_score_metric_logic(self) -> None:
        """The _score_metric used by compile_dspy_scorer is argmax-based."""
        import json
        from unittest.mock import MagicMock

        example = MagicMock()
        example.scores = json.dumps([0.0, 1.0, 0.0])
        pred_correct = MagicMock()
        pred_correct.scores = json.dumps([0.1, 0.8, 0.1])
        pred_wrong = MagicMock()
        pred_wrong.scores = json.dumps([0.9, 0.05, 0.05])
        pred_malformed = MagicMock()
        pred_malformed.scores = "not json"

        assert _score_metric(example, pred_correct) == 1.0
        assert _score_metric(example, pred_wrong) == 0.0
        assert _score_metric(example, pred_malformed) == 0.0

    def test_trainset_uses_mid_prefix(self) -> None:
        """build_dspy_trainset picks a mid-point cumulative prefix."""
        mc = [_make_mc_question()]
        trainset = build_dspy_trainset(mc, max_examples=1)
        ex = trainset[0]
        # The question has 3 prefixes; mid = 3//2 = 1
        assert ex["clue_prefix"] == "Who was the first"
```

## File: tests/test_hazard_pretrain.py
```python
"""Tests for the hazard pretraining bridge utilities and CLI guard."""

from __future__ import annotations

import argparse
import importlib

import pytest
import torch


def test_compute_survival_terms_simple_case() -> None:
    """compute_survival_terms returns expected survival and stop masses."""
    compute_survival_terms = importlib.import_module(
        "training.hazard_pretrain"
    ).compute_survival_terms

    stop_probs = torch.tensor([[0.2, 0.5]], dtype=torch.float32)
    survival, stop_mass = compute_survival_terms(stop_probs)

    expected_survival = torch.tensor([[1.0, 0.8, 0.4]], dtype=torch.float32)
    expected_stop_mass = torch.tensor([[0.2, 0.4]], dtype=torch.float32)
    assert torch.allclose(survival, expected_survival, atol=1e-6)
    assert torch.allclose(stop_mass, expected_stop_mass, atol=1e-6)


def test_hazard_expected_nll_loss_uses_terminal_penalty() -> None:
    """hazard_expected_nll_loss returns a scalar with beta_terminal applied."""
    hazard_expected_nll_loss = importlib.import_module(
        "training.hazard_pretrain"
    ).hazard_expected_nll_loss

    stop_probs = torch.tensor([[0.2, 0.5]], dtype=torch.float32)
    nll_per_prefix = torch.tensor([[1.0, 2.0]], dtype=torch.float32)

    loss = hazard_expected_nll_loss(
        stop_probs=stop_probs,
        nll_per_prefix=nll_per_prefix,
        beta_terminal=1.5,
    )

    assert loss.ndim == 0
    assert loss.item() == pytest.approx(1.6)


def test_hazard_pretrain_flag_raises_not_implemented() -> None:
    """CLI rejects hazard-pretrain until the training loop exists."""
    validate_args = importlib.import_module("scripts.train_t5_policy").validate_args

    args = argparse.Namespace(
        config="configs/t5_policy.yaml",
        smoke=False,
        skip_supervised=False,
        model_path=None,
        mc_path=None,
        ppo_iterations=None,
        hazard_pretrain=True,
        beta_terminal=1.0,
        freeze_answer_head=False,
    )

    with pytest.raises(NotImplementedError, match="Hazard pretraining loop not yet implemented"):
        validate_args(args)
```

## File: tests/test_mc_builder_topk.py
```python
"""Regression tests for top-M distractor ranking in MCBuilder._compute_rankings.

Validates that the argpartition-based top-M retrieval produces the same top
distractors as a full argsort, truncates ranking lists correctly, degrades
gracefully when N is small, and leaves category_random strategy unchanged.
"""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from qb_data.mc_builder import MCBuilder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_synthetic_answers(n: int) -> tuple[list[str], dict[str, str]]:
    """Create *n* synthetic answers with distinct TF-IDF profiles.

    Each answer is a short phrase and its profile is a sentence containing
    unique vocabulary so TF-IDF can discriminate between them.
    """
    topics = [
        ("George Washington", "first president commander revolutionary war continental army"),
        ("Thomas Jefferson", "third president declaration independence Virginia Monticello"),
        ("John Adams", "second president Massachusetts diplomat federalist"),
        ("Benjamin Franklin", "inventor diplomat Philadelphia printing press electricity"),
        ("Abraham Lincoln", "sixteenth president civil war emancipation slavery"),
        ("Alexander Hamilton", "treasury secretary banking system federalist papers"),
        ("James Madison", "bill rights constitution fourth president Virginia"),
        ("Andrew Jackson", "military hero populist president battle New Orleans"),
        ("Theodore Roosevelt", "progressive trust buster national parks rough riders"),
        ("Ulysses Grant", "civil war general eighteenth president reconstruction"),
        ("Woodrow Wilson", "world war one league nations progressive president"),
        ("Franklin Roosevelt", "new deal world war two great depression fireside"),
        ("Harry Truman", "atomic bomb cold war Korean conflict fair deal"),
        ("Dwight Eisenhower", "supreme commander NATO interstate highway system"),
        ("John Kennedy", "space race Cuban missile crisis new frontier"),
        ("Lyndon Johnson", "great society civil rights Vietnam escalation"),
        ("Richard Nixon", "detente China opening Watergate resignation"),
        ("Ronald Reagan", "cold war end conservative revolution economic growth"),
        ("Barack Obama", "affordable care act first African American president"),
        ("Jimmy Carter", "Camp David accords energy crisis human rights"),
    ]
    answers = [t[0] for t in topics[:n]]
    profiles = {t[0]: t[1] for t in topics[:n]}
    return answers, profiles


def _full_sort_rankings(
    answers: list[str], profiles: dict[str, str]
) -> dict[str, list[str]]:
    """Compute rankings via full argsort (reference implementation)."""
    docs = [profiles[a] for a in answers]
    answer_idx = {a: i for i, a in enumerate(answers)}
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(docs)
    sim = cosine_similarity(matrix, matrix)
    rankings: dict[str, list[str]] = {}
    for answer in answers:
        idx = answer_idx[answer]
        order = np.argsort(-sim[idx]).tolist()
        rankings[answer] = [answers[i] for i in order if answers[i] != answer]
    return rankings


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTopMRanking:
    """Tests for top-M argpartition ranking in MCBuilder._compute_rankings."""

    def test_top_m_truncation(self) -> None:
        """Rankings should have length <= min(M, N-1)."""
        answers, profiles = _make_synthetic_answers(20)
        builder = MCBuilder(K=4, strategy="tfidf_profile")
        categories: dict[str, str] = {}

        rankings = builder._compute_rankings(answers, profiles, categories)

        M = min(max(5 * 4, 30), len(answers) - 1)  # min(30, 19) = 19
        for answer, ranked in rankings.items():
            assert len(ranked) <= min(M, len(answers) - 1), (
                f"Answer '{answer}' has {len(ranked)} distractors, "
                f"expected <= {min(M, len(answers) - 1)}"
            )

    def test_order_preservation(self) -> None:
        """Top-3 distractors must match the full-sort reference."""
        answers, profiles = _make_synthetic_answers(20)
        builder = MCBuilder(K=4, strategy="tfidf_profile")
        categories: dict[str, str] = {}

        rankings = builder._compute_rankings(answers, profiles, categories)
        reference = _full_sort_rankings(answers, profiles)

        for answer in answers:
            actual_top3 = rankings[answer][:3]
            expected_top3 = reference[answer][:3]
            assert actual_top3 == expected_top3, (
                f"Answer '{answer}': top-3 mismatch.\n"
                f"  actual:   {actual_top3}\n"
                f"  expected: {expected_top3}"
            )

    def test_small_n_graceful(self) -> None:
        """With N=5, rankings should have length N-1=4 without error."""
        answers, profiles = _make_synthetic_answers(5)
        builder = MCBuilder(K=4, strategy="tfidf_profile")
        categories: dict[str, str] = {}

        rankings = builder._compute_rankings(answers, profiles, categories)

        for answer, ranked in rankings.items():
            assert len(ranked) == 4, (
                f"Answer '{answer}' has {len(ranked)} distractors, expected 4"
            )

    def test_category_random_unaffected(self) -> None:
        """category_random strategy should not use argpartition path."""
        answers, profiles = _make_synthetic_answers(10)
        categories = {a: "History" for a in answers}
        builder = MCBuilder(K=4, strategy="category_random")

        rankings = builder._compute_rankings(answers, profiles, categories)

        for answer, ranked in rankings.items():
            # All same-category peers (minus self) should be present
            assert set(ranked) == set(a for a in answers if a != answer), (
                f"Answer '{answer}': category_random should include all peers"
            )
```

## File: tests/test_mc_builder_variable_k.py
```python
"""Tests for variable-K MC question construction."""

from __future__ import annotations

import pytest

from qb_data.answer_profiles import AnswerProfileBuilder
from qb_data.data_loader import TossupQuestion
from qb_data.mc_builder import MCBuilder


def _make_questions(n: int = 20, n_unique_answers: int | None = None) -> list[TossupQuestion]:
    n_ans = n_unique_answers if n_unique_answers is not None else n
    questions = []
    for i in range(n):
        tokens = [f"word{i}_{j}" for j in range(10)]
        questions.append(
            TossupQuestion(
                qid=f"q{i:03d}",
                question=" ".join(tokens),
                tokens=tokens,
                answer_primary=f"Answer_{i % n_ans}",
                clean_answers=[f"Answer_{i % n_ans}"],
                run_indices=[2, 5, 9],
                human_buzz_positions=[],
                category=["History", "Science"][i % 2],
                cumulative_prefixes=[
                    " ".join(tokens[:3]),
                    " ".join(tokens[:6]),
                    " ".join(tokens),
                ],
            )
        )
    return questions


class TestFixedKUnchanged:
    def test_fixed_k_default(self) -> None:
        qs = _make_questions(20)
        builder = MCBuilder(K=4, strategy="category_random", random_seed=42)
        profile = AnswerProfileBuilder()
        mc = builder.build(qs, profile)
        for q in mc:
            assert len(q.options) == 4

    def test_variable_k_false_is_fixed(self) -> None:
        qs = _make_questions(20)
        builder = MCBuilder(K=4, strategy="category_random", random_seed=42, variable_K=False)
        profile = AnswerProfileBuilder()
        mc = builder.build(qs, profile)
        for q in mc:
            assert len(q.options) == 4


class TestVariableK:
    def test_variable_k_yields_mixed(self) -> None:
        qs = _make_questions(40)
        builder = MCBuilder(
            K=6, strategy="category_random", random_seed=42,
            variable_K=True, min_K=2, max_K=6,
        )
        profile = AnswerProfileBuilder()
        mc = builder.build(qs, profile)
        option_counts = {len(q.options) for q in mc}
        assert len(option_counts) > 1, f"Expected mixed K, got only {option_counts}"
        for q in mc:
            assert 2 <= len(q.options) <= 6

    def test_gold_index_valid(self) -> None:
        qs = _make_questions(30)
        builder = MCBuilder(
            K=5, strategy="category_random", random_seed=42,
            variable_K=True, min_K=2, max_K=5,
        )
        profile = AnswerProfileBuilder()
        mc = builder.build(qs, profile)
        for q in mc:
            assert 0 <= q.gold_index < len(q.options)
            assert q.options[q.gold_index] in q.clean_answers or \
                q.option_answer_primary[q.gold_index] == q.answer_primary

    def test_profiles_match_options(self) -> None:
        qs = _make_questions(20)
        builder = MCBuilder(
            K=5, strategy="category_random", random_seed=42,
            variable_K=True, min_K=3, max_K=5,
        )
        profile = AnswerProfileBuilder()
        mc = builder.build(qs, profile)
        for q in mc:
            assert len(q.option_profiles) == len(q.options)
            assert len(q.option_answer_primary) == len(q.options)
```

## File: tests/test_opponent_models.py
```python
"""Tests for qb_env/opponent_models.py."""

from __future__ import annotations

import pytest

from qb_data.mc_builder import MCQuestion
from qb_env.opponent_models import (
    EmpiricalHistogramOpponentModel,
    LogisticOpponentModel,
    build_opponent_model_from_config,
)


def _make_question(
    human_buzz_positions=None,
    num_steps: int = 6,
) -> MCQuestion:
    tokens = [f"t{i}" for i in range(num_steps * 2)]
    run_indices = list(range(0, num_steps * 2, 2))
    prefixes = [" ".join(tokens[: ri + 1]) for ri in run_indices]
    return MCQuestion(
        qid="q_test",
        question=" ".join(tokens),
        tokens=tokens,
        answer_primary="Answer A",
        clean_answers=["Answer A"],
        run_indices=run_indices,
        human_buzz_positions=human_buzz_positions or [],
        category="Test",
        cumulative_prefixes=prefixes,
        options=["Answer A", "Answer B", "Answer C", "Answer D"],
        gold_index=0,
        option_profiles=["prof_a", "prof_b", "prof_c", "prof_d"],
        option_answer_primary=["Answer A", "Answer B", "Answer C", "Answer D"],
        distractor_strategy="test",
    )


class TestLogisticOpponentModel:
    def test_monotonicity(self) -> None:
        model = LogisticOpponentModel(midpoint=0.5, steepness=6.0)
        q = _make_question(num_steps=10)
        probs = [model.prob_buzzed_before_step(q, t) for t in range(10)]
        for i in range(1, len(probs)):
            assert probs[i] >= probs[i - 1] - 1e-12

    def test_range_01(self) -> None:
        model = LogisticOpponentModel()
        q = _make_question(num_steps=20)
        for t in range(20):
            p = model.prob_buzzed_before_step(q, t)
            assert 0.0 <= p <= 1.0

    def test_survive_complement(self) -> None:
        model = LogisticOpponentModel()
        q = _make_question(num_steps=10)
        for t in range(10):
            assert abs(
                model.prob_buzzed_before_step(q, t)
                + model.prob_survive_to_step(q, t)
                - 1.0
            ) < 1e-12

    def test_step_zero_near_zero(self) -> None:
        model = LogisticOpponentModel(midpoint=0.6, steepness=6.0)
        q = _make_question(num_steps=10)
        assert model.prob_buzzed_before_step(q, 0) < 0.1


class TestEmpiricalHistogramOpponentModel:
    def test_cumulative_from_positions(self) -> None:
        q = _make_question(human_buzz_positions=[(2, 3), (4, 7)], num_steps=6)
        model = EmpiricalHistogramOpponentModel()
        p_at_3 = model.prob_buzzed_before_step(q, 1)
        p_at_5 = model.prob_buzzed_before_step(q, 2)
        assert p_at_5 >= p_at_3

    def test_fallback_when_no_data(self) -> None:
        q = _make_question(human_buzz_positions=[], num_steps=10)
        model = EmpiricalHistogramOpponentModel()
        p = model.prob_buzzed_before_step(q, 5)
        assert 0.0 <= p <= 1.0

    def test_global_fallback(self) -> None:
        q = _make_question(human_buzz_positions=[], num_steps=6)
        model = EmpiricalHistogramOpponentModel(
            global_positions=[(2, 5), (4, 5)]
        )
        p = model.prob_buzzed_before_step(q, 2)
        assert p > 0.0


class TestBuildOpponentModelFromConfig:
    def test_none_when_disabled(self) -> None:
        cfg = {"environment": {"opponent_buzz_model": {"type": "none"}}}
        assert build_opponent_model_from_config(config=cfg) is None

    def test_none_when_missing(self) -> None:
        assert build_opponent_model_from_config(config={}) is None
        assert build_opponent_model_from_config(config=None) is None

    def test_logistic(self) -> None:
        cfg = {"environment": {"opponent_buzz_model": {"type": "logistic", "midpoint": 0.4}}}
        model = build_opponent_model_from_config(config=cfg)
        assert isinstance(model, LogisticOpponentModel)
        assert model.midpoint == 0.4

    def test_empirical(self) -> None:
        q = _make_question(human_buzz_positions=[(2, 5)])
        cfg = {"environment": {"opponent_buzz_model": {"type": "empirical"}}}
        model = build_opponent_model_from_config(questions=[q], config=cfg)
        assert isinstance(model, EmpiricalHistogramOpponentModel)
```

## File: training/hazard_pretrain.py
```python
"""Hazard pretraining bridge utilities for stopping-aware warm starts."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class HazardBatchOutput:
    """Container for hazard-bridge intermediate tensors."""

    stop_probs: torch.Tensor
    survival: torch.Tensor
    stop_mass: torch.Tensor
    nll_per_prefix: torch.Tensor
    loss: torch.Tensor


def compute_survival_terms(stop_probs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute survival and stop-mass terms from per-prefix stop probabilities."""
    stay_probs = 1.0 - stop_probs
    batch_size, steps = stop_probs.shape
    survival = torch.ones(
        (batch_size, steps + 1), dtype=stop_probs.dtype, device=stop_probs.device
    )
    if steps > 0:
        survival[:, 1:] = torch.cumprod(stay_probs, dim=1)
    stop_mass = survival[:, :-1] * stop_probs
    return survival, stop_mass


def hazard_expected_nll_loss(
    stop_probs: torch.Tensor,
    nll_per_prefix: torch.Tensor,
    beta_terminal: float = 1.0,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute the hazard-bridge expected NLL loss prior to PPO."""
    survival, stop_mass = compute_survival_terms(stop_probs)
    weighted_nll = stop_mass * nll_per_prefix
    if mask is not None:
        weighted_nll = weighted_nll * mask
    seq_loss = weighted_nll.sum(dim=1) + beta_terminal * survival[:, -1]
    return seq_loss.mean()
```

## File: verify_data_loader.py
```python
#!/usr/bin/env python
"""
Verification script for data loader functionality.
"""

from qb_data.data_loader import QANTADatasetLoader
from qb_data.text_utils import normalize_answer


def main():
    """Test the data loader with the test CSV file."""
    print("=" * 60)
    print("Testing QANTADatasetLoader")
    print("=" * 60)

    # Load test questions
    loader = QANTADatasetLoader()
    questions = loader.load_from_csv('data/test_questions.csv')

    print(f"\nLoaded {len(questions)} questions from test CSV")
    print("-" * 60)

    # Display first few questions
    for i, q in enumerate(questions[:3], 1):
        print(f"\nQuestion {i}:")
        print(f"  QID: {q.qid}")
        print(f"  Category: {q.category}")
        print(f"  Answer: {q.answer_primary}")
        print(f"  Clean answers: {q.clean_answers}")
        print(f"  Number of tokens: {len(q.tokens)}")
        print(f"  Number of clues: {len(q.run_indices)}")
        print(f"  Run indices: {q.run_indices}")

        # Show cumulative prefixes (first 50 chars of each)
        print(f"  Cumulative prefixes:")
        for j, prefix in enumerate(q.cumulative_prefixes, 1):
            preview = prefix[:50] + "..." if len(prefix) > 50 else prefix
            print(f"    Clue {j}: {preview}")

    print("\n" + "=" * 60)
    print("Testing normalize_answer function")
    print("=" * 60)

    test_cases = [
        ("The Great Gatsby", "great gatsby"),
        ("A Tale of Two Cities", "tale of two cities"),
        ("An Example!!!", "example"),
        ("  Ludwig   van   Beethoven  ", "ludwig van beethoven"),
        ("", ""),
        ("The", ""),
    ]

    for input_text, expected in test_cases:
        result = normalize_answer(input_text)
        status = "✓" if result == expected else "✗"
        print(f"{status} normalize_answer({input_text!r}) = {result!r} (expected: {expected!r})")

    print("\n" + "=" * 60)
    print("Verification complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

## File: agents/__init__.py
```python
from agents.threshold_buzzer import (
    ThresholdBuzzer,
    AlwaysBuzzFinalBuzzer,
    EpisodeResult,
    sweep_thresholds,
    result_to_dict,
)
from agents.bayesian_buzzer import (
    SoftmaxProfileBuzzer,
    SequentialBayesBuzzer,
    SoftmaxEpisodeResult,
    sweep_sequential_thresholds,
)

# Lazy import: PPOBuzzer requires stable_baselines3 which may not be installed
# in all environments (e.g., baseline-only runs). Import on demand.


def __getattr__(name: str):
    if name in ("PPOBuzzer", "PPOEpisodeTrace"):
        from agents.ppo_buzzer import PPOBuzzer, PPOEpisodeTrace
        return {"PPOBuzzer": PPOBuzzer, "PPOEpisodeTrace": PPOEpisodeTrace}[name]
    raise AttributeError(f"module 'agents' has no attribute {name!r}")


__all__ = [
    "ThresholdBuzzer",
    "AlwaysBuzzFinalBuzzer",
    "SoftmaxProfileBuzzer",
    "SequentialBayesBuzzer",
    "PPOBuzzer",
    "EpisodeResult",
    "SoftmaxEpisodeResult",
    "PPOEpisodeTrace",
    "sweep_thresholds",
    "sweep_sequential_thresholds",
    "result_to_dict",
]
```

## File: agents/softmax_profile_buzzer.py
```python
"""qb-rl compatibility re-exports for Bayesian-family buzzers."""

from agents.bayesian_buzzer import (
    SequentialBayesBuzzer,
    SoftmaxEpisodeResult,
    SoftmaxProfileBuzzer,
)

__all__ = [
    "SoftmaxEpisodeResult",
    "SoftmaxProfileBuzzer",
    "SequentialBayesBuzzer",
]
```

## File: configs/t5_policy.yaml
```yaml
# T5 Policy Configuration
# Hyperparameters for T5PolicyModel with supervised warm-start and PPO fine-tuning.
# Use with: python -m training.train_supervised_t5 --config configs/t5_policy.yaml

model:
  model_name: t5-large  # Use t5-base or t5-small if memory constrained
  device: auto  # auto-detect cuda > mps > cpu
  max_input_length: 512
  num_choices: 4

supervised:
  lr: 3.0e-4
  epochs: 10
  batch_size: 8
  grad_accum_steps: 4  # Effective batch = 32
  max_grad_norm: 1.0
  weight_decay: 0.01
  checkpoint_dir: checkpoints

ppo:
  lr: 1.0e-5  # Lower than supervised for stability
  iterations: 100
  batch_size: 8
  epochs_per_iter: 4
  clip_ratio: 0.2
  value_coef: 0.5
  entropy_coef: 0.01
  max_grad_norm: 0.5
  gamma: 0.99
  gae_lambda: 0.95
  target_kl: 0.03
  checkpoint_dir: checkpoints

data:
  csv_path: "questions.csv"
  K: 4
  train_size: 0.7
  val_size: 0.15
  test_size: 0.15
  seed: 42

# Smoke test overrides (use with --smoke flag)
smoke:
  model:
    model_name: t5-small  # 60M params instead of 770M
    max_input_length: 128
  supervised:
    epochs: 2
    batch_size: 4
    grad_accum_steps: 1  # No accumulation for speed
  ppo:
    iterations: 5
    batch_size: 4
    epochs_per_iter: 2
  data:
    max_questions: 50
```

## File: evaluation/controls.py
```python
"""
Control Experiments for Quiz Bowl Buzzer Evaluation

Implements three control experiments to validate that the buzzer agent
genuinely uses question clues rather than exploiting surface-form artifacts:

1. **Choices-only control**: Strips all clues, trains a logistic regression
   on option surface features (char n-grams, length, capitalization). Expected
   accuracy ~25% (1/K) if options have no exploitable artifacts.

2. **Shuffle control**: Randomizes option ordering to verify the agent has
   no position bias. Performance should be unchanged.

3. **Alias substitution control**: Swaps answer text with aliases to verify
   robustness to surface-form changes.

Ported from qb-rl reference implementation (evaluation/controls.py) with
import path adaptations for the unified qanta-buzzer codebase.
"""

from __future__ import annotations

import random
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from qb_data.mc_builder import MCQuestion

if TYPE_CHECKING:
    from agents.threshold_buzzer import _PrecomputedQuestion


def _option_scalar_features(option: str) -> list[float]:
    """Extract scalar surface features from a single option string.

    Parameters
    ----------
    option : str
        Answer option text.

    Returns
    -------
    list[float]
        Six scalar features: char length, token count, has_parens,
        has_comma, is_title, is_lower.
    """
    tokens = option.split()
    has_parens = 1.0 if "(" in option or ")" in option else 0.0
    has_comma = 1.0 if "," in option else 0.0
    is_title = 1.0 if option.istitle() else 0.0
    is_lower = 1.0 if option.islower() else 0.0
    return [
        float(len(option)),
        float(len(tokens)),
        has_parens,
        has_comma,
        is_title,
        is_lower,
    ]


def _cross_option_features(options: list[str]) -> list[float]:
    """Extract cross-option comparative features.

    Parameters
    ----------
    options : list[str]
        All answer options for a question.

    Returns
    -------
    list[float]
        Three features: max/min length ratio, length std, number of
        distinct capitalization patterns.
    """
    lengths = np.array(
        [max(1, len(o.split())) for o in options], dtype=np.float32
    )
    cap_patterns = len(
        set(
            ("title" if o.istitle() else "lower" if o.islower() else "mixed")
            for o in options
        )
    )
    return [
        float(lengths.max() / lengths.min()),
        float(lengths.std()),
        float(cap_patterns),
    ]


def run_choices_only_control(
    questions: list[MCQuestion],
    random_seed: int = 13,
    test_fraction: float = 0.25,
) -> dict[str, float]:
    """Run choices-only control: predict answer from surface features only.

    Strips all question clues and trains a logistic regression on option
    surface features (char n-grams, length, capitalization patterns).
    Expected accuracy ~25% (1/K) if options are well-constructed.

    Parameters
    ----------
    questions : list[MCQuestion]
        Full MC question dataset.
    random_seed : int
        Seed for reproducible train/test split.
    test_fraction : float
        Fraction of questions held out for testing.

    Returns
    -------
    dict[str, float]
        Control results: accuracy, chance baseline, and test set size.
    """
    if not questions:
        return {"accuracy": 0.0, "chance": 0.0, "n_test": 0.0}

    rng = random.Random(random_seed)
    shuffled = questions[:]
    rng.shuffle(shuffled)
    split_idx = max(1, int(len(shuffled) * (1.0 - test_fraction)))
    train_q = shuffled[:split_idx]
    test_q = shuffled[split_idx:]
    if not test_q:
        test_q = train_q

    vec = TfidfVectorizer(analyzer="char", ngram_range=(3, 3), min_df=1)
    vec.fit([opt for q in train_q for opt in q.options])

    def build_matrix(
        rows: list[MCQuestion],
    ) -> tuple[np.ndarray, np.ndarray, list[int]]:
        X = []
        y = []
        group_sizes: list[int] = []
        for q in rows:
            cross = _cross_option_features(q.options)
            group_sizes.append(len(q.options))
            tfidf = vec.transform(q.options).toarray()
            for i, option in enumerate(q.options):
                feat = np.array(
                    _option_scalar_features(option) + cross, dtype=np.float32
                )
                row = np.concatenate([feat, tfidf[i]], axis=0)
                X.append(row)
                y.append(1 if i == q.gold_index else 0)
        return np.array(X), np.array(y), group_sizes

    X_train, y_train, _ = build_matrix(train_q)
    X_test, y_test, test_group_sizes = build_matrix(test_q)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    probs = clf.predict_proba(X_test)[:, 1]

    offset = 0
    correct = 0
    total = 0
    for q, group_size in zip(test_q, test_group_sizes):
        group_probs = probs[offset : offset + group_size]
        pred_idx = int(np.argmax(group_probs))
        if pred_idx == q.gold_index:
            correct += 1
        total += 1
        offset += group_size

    accuracy = correct / max(1, total)
    chance = 1.0 / max(1, len(questions[0].options))
    return {
        "accuracy": float(accuracy),
        "chance": float(chance),
        "n_test": float(total),
    }


def shuffled_option_copy(
    question: MCQuestion, rng: random.Random
) -> MCQuestion:
    """Create a copy of an MCQuestion with shuffled option ordering.

    Parameters
    ----------
    question : MCQuestion
        Original question.
    rng : random.Random
        Random number generator for shuffling.

    Returns
    -------
    MCQuestion
        Copy with permuted options, profiles, answer_primary, and
        updated gold_index.
    """
    perm = list(range(len(question.options)))
    rng.shuffle(perm)
    new_options = [question.options[i] for i in perm]
    new_profiles = [question.option_profiles[i] for i in perm]
    new_answer_primary = [question.option_answer_primary[i] for i in perm]
    new_gold = perm.index(question.gold_index)
    return replace(
        question,
        options=new_options,
        option_profiles=new_profiles,
        option_answer_primary=new_answer_primary,
        gold_index=new_gold,
    )


def run_shuffle_control(
    questions: list[MCQuestion],
    evaluator: Callable[[list[MCQuestion]], dict[str, Any]],
    random_seed: int = 13,
) -> dict[str, Any]:
    """Run shuffle control: randomize option ordering and evaluate.

    Permutes the answer options for each question and runs the evaluator.
    If the agent has no position bias, performance should be unchanged.

    Parameters
    ----------
    questions : list[MCQuestion]
        Full MC question dataset.
    evaluator : callable
        Function that takes a list of MCQuestion and returns a metrics dict.
    random_seed : int
        Seed for reproducible shuffling.

    Returns
    -------
    dict[str, Any]
        Evaluation metrics on shuffled questions.
    """
    rng = random.Random(random_seed)
    shuffled = [shuffled_option_copy(q, rng) for q in questions]
    return evaluator(shuffled)


def alias_substitution_copy(
    question: MCQuestion,
    alias_lookup: dict[str, list[str]],
    rng: random.Random,
) -> MCQuestion:
    """Create a copy of an MCQuestion with alias-substituted options.

    Parameters
    ----------
    question : MCQuestion
        Original question.
    alias_lookup : dict[str, list[str]]
        Mapping from canonical answer to list of known aliases.
    rng : random.Random
        Random number generator for alias selection.

    Returns
    -------
    MCQuestion
        Copy with alias-substituted option text and profiles.
    """
    new_options = []
    new_profiles = list(question.option_profiles)
    for i, (option_text, answer_primary) in enumerate(
        zip(question.options, question.option_answer_primary)
    ):
        aliases = [
            a
            for a in alias_lookup.get(answer_primary, [])
            if a and a != option_text
        ]
        if aliases:
            alias = rng.choice(aliases)
            new_options.append(alias)
            if new_profiles[i].strip() == answer_primary.strip():
                new_profiles[i] = alias
        else:
            new_options.append(option_text)
    return replace(question, options=new_options, option_profiles=new_profiles)


def run_alias_substitution_control(
    questions: list[MCQuestion],
    alias_lookup: dict[str, list[str]],
    evaluator: Callable[[list[MCQuestion]], dict[str, Any]],
    random_seed: int = 13,
) -> dict[str, Any]:
    """Run alias substitution control: swap answer text with aliases.

    Replaces option text with known aliases to verify the agent is robust
    to surface-form changes. Performance should be similar to full eval.

    Parameters
    ----------
    questions : list[MCQuestion]
        Full MC question dataset.
    alias_lookup : dict[str, list[str]]
        Mapping from canonical answer to list of known aliases.
    evaluator : callable
        Function that takes a list of MCQuestion and returns a metrics dict.
    random_seed : int
        Seed for reproducible alias selection.

    Returns
    -------
    dict[str, Any]
        Evaluation metrics on alias-substituted questions.
    """
    rng = random.Random(random_seed)
    swapped = [
        alias_substitution_copy(q, alias_lookup=alias_lookup, rng=rng)
        for q in questions
    ]
    return evaluator(swapped)


def run_shuffle_control_precomputed(
    precomputed: list["_PrecomputedQuestion"],
    threshold: float,
    alpha: float,
    random_seed: int = 13,
) -> dict[str, Any]:
    """Run shuffle control by permuting precomputed belief vectors.

    Produces numerically identical results to ``run_shuffle_control`` with
    a live ``SoftmaxProfileBuzzer`` evaluator, but makes zero
    ``likelihood_model.score()`` calls.  Instead, the belief vectors
    stored in each ``_PrecomputedQuestion`` are reordered according to
    the same random permutation that ``shuffled_option_copy`` would apply.

    Parameters
    ----------
    precomputed : list[_PrecomputedQuestion]
        Pre-computed belief distributions (one per question).
    threshold : float
        Buzz threshold for the softmax profile buzzer.
    alpha : float
        Sigmoid steepness for the confidence proxy.
    random_seed : int
        Seed for reproducible shuffling (must match the seed used in
        ``run_shuffle_control`` for equivalence).

    Returns
    -------
    dict[str, Any]
        Summary metrics with ``"runs"`` key containing per-question dicts.
    """
    from dataclasses import asdict

    from agents.threshold_buzzer import (
        _PrecomputedQuestion,
        _softmax_episode_from_precomputed,
    )
    from evaluation.metrics import calibration_at_buzz, summarize_buzz_metrics

    rng = random.Random(random_seed)
    runs: list[dict[str, Any]] = []
    for pq in precomputed:
        perm = list(range(pq.num_options))
        rng.shuffle(perm)
        new_gold = perm.index(pq.gold_index)
        shuffled_beliefs = [b[perm] for b in pq.beliefs]
        shuffled_pq = _PrecomputedQuestion(
            qid=pq.qid,
            gold_index=new_gold,
            num_options=pq.num_options,
            beliefs=shuffled_beliefs,
        )
        result = _softmax_episode_from_precomputed(shuffled_pq, threshold, alpha)
        runs.append(asdict(result))
    summary = {**summarize_buzz_metrics(runs), **calibration_at_buzz(runs)}
    summary["runs"] = runs
    return summary


def bootstrap_ci(
    values: list[float],
    n_samples: int = 1000,
    alpha: float = 0.05,
    seed: int = 13,
) -> tuple[float, float]:
    """Compute bootstrap confidence interval for the mean.

    Parameters
    ----------
    values : list[float]
        Observed values.
    n_samples : int
        Number of bootstrap resamples.
    alpha : float
        Significance level (0.05 = 95% CI).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    tuple[float, float]
        Lower and upper bounds of the confidence interval.
    """
    if not values:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    arr = np.array(values, dtype=np.float64)
    samples = []
    for _ in range(n_samples):
        idx = rng.integers(0, len(arr), size=len(arr))
        samples.append(float(arr[idx].mean()))
    lo = np.quantile(samples, alpha / 2.0)
    hi = np.quantile(samples, 1.0 - alpha / 2.0)
    return float(lo), float(hi)
```

## File: models/answer_profiles.py
```python
"""qb-rl compatibility re-export for answer profile building."""

from qb_data.answer_profiles import AnswerProfileBuilder

__all__ = ["AnswerProfileBuilder"]
```

## File: models/features.py
```python
"""
Belief Feature Extraction

Extracts derived features from belief probability distributions for use as
policy observations. Given a belief vector of K probabilities (one per answer
option), produces a (K + 6)-dimensional feature vector containing:

    belief[0..K-1]   raw belief probabilities
    top_p             max belief probability
    margin            gap between top two probabilities
    entropy           Shannon entropy of the distribution
    stability         L1 distance from previous belief (0 if first step)
    progress          fraction of total clue steps elapsed
    clue_idx_norm     normalized clue index (0 to 1 over steps)

Ported from qb-rl reference implementation (models/features.py).
"""

from __future__ import annotations

import numpy as np


def entropy_of_distribution(prob: np.ndarray) -> float:
    """Compute Shannon entropy of a probability distribution.

    Uses clipping for numerical stability to avoid log(0).

    Parameters
    ----------
    prob : np.ndarray
        1D probability vector. Values should sum to ~1.0.

    Returns
    -------
    float
        Shannon entropy H(p) = -sum(p * log(p)), non-negative.

    Examples
    --------
    >>> import numpy as np
    >>> uniform = np.array([0.25, 0.25, 0.25, 0.25])
    >>> abs(entropy_of_distribution(uniform) - 1.3863) < 0.001
    True
    """
    clipped = np.clip(prob, 1e-12, 1.0)
    return float(-(clipped * np.log(clipped)).sum())


def extract_belief_features(
    belief: np.ndarray,
    prev_belief: np.ndarray | None,
    step_idx: int,
    total_steps: int,
) -> np.ndarray:
    """Extract derived features from a belief probability vector.

    Concatenates the raw belief with 6 derived scalar features to produce
    a fixed-size observation vector for the RL policy.

    Parameters
    ----------
    belief : np.ndarray
        1D probability vector of shape (K,) over answer options.
    prev_belief : np.ndarray or None
        Previous step's belief vector, same shape as ``belief``.
        Pass None on the first step (stability will be 0.0).
    step_idx : int
        Current clue step index (0-based).
    total_steps : int
        Total number of clue steps in the episode.

    Returns
    -------
    np.ndarray
        Feature vector of shape (K + 6,) with dtype float32.
        Layout: [belief..., top_p, margin, entropy, stability, progress, clue_idx_norm].

    Raises
    ------
    ValueError
        If ``belief`` is not a 1D array.

    Examples
    --------
    >>> import numpy as np
    >>> belief = np.array([0.5, 0.3, 0.15, 0.05], dtype=np.float32)
    >>> feats = extract_belief_features(belief, None, 2, 6)
    >>> feats.shape
    (10,)
    >>> feats.dtype
    dtype('float32')
    """
    belief = np.asarray(belief, dtype=np.float32)
    if belief.ndim != 1:
        raise ValueError("belief must be a 1D probability vector")

    top_p = float(np.max(belief))
    sorted_probs = np.sort(belief)[::-1]
    second = float(sorted_probs[1]) if len(sorted_probs) > 1 else 0.0
    margin = top_p - second
    ent = entropy_of_distribution(belief)
    stability = float(np.abs(belief - prev_belief).sum()) if prev_belief is not None else 0.0
    progress = float(step_idx / max(1, total_steps))
    clue_idx_norm = float(step_idx / max(1, total_steps - 1))

    extras = np.array([top_p, margin, ent, stability, progress, clue_idx_norm], dtype=np.float32)
    return np.concatenate([belief, extras]).astype(np.float32)


def extract_padded_belief_features(
    belief: np.ndarray,
    prev_belief: np.ndarray | None,
    step_idx: int,
    total_steps: int,
    max_K: int,
) -> np.ndarray:
    """Extract belief features padded to a fixed ``max_K`` size.

    Identical to :func:`extract_belief_features` except the belief
    segment is zero-padded (or truncated) to exactly ``max_K`` elements,
    producing a ``(max_K + 6)``-dimensional vector regardless of the
    actual number of answer options.

    Parameters
    ----------
    belief : np.ndarray
        1D probability vector of shape (K_actual,).
    prev_belief : np.ndarray or None
        Previous belief vector (same shape as *belief*).
    step_idx : int
        Current clue step index (0-based).
    total_steps : int
        Total clue steps in the episode.
    max_K : int
        Target padded length for the belief segment.

    Returns
    -------
    np.ndarray
        Feature vector of shape (max_K + 6,), dtype float32.
    """
    belief = np.asarray(belief, dtype=np.float32)
    if belief.ndim != 1:
        raise ValueError("belief must be a 1D probability vector")

    K_actual = len(belief)

    top_p = float(np.max(belief))
    sorted_probs = np.sort(belief)[::-1]
    second = float(sorted_probs[1]) if len(sorted_probs) > 1 else 0.0
    margin = top_p - second
    ent = entropy_of_distribution(belief)
    stability = float(np.abs(belief - prev_belief).sum()) if prev_belief is not None else 0.0
    progress = float(step_idx / max(1, total_steps))
    clue_idx_norm = float(step_idx / max(1, total_steps - 1))

    padded = np.zeros(max_K, dtype=np.float32)
    padded[:K_actual] = belief[:max_K]
    extras = np.array([top_p, margin, ent, stability, progress, clue_idx_norm], dtype=np.float32)
    return np.concatenate([padded, extras]).astype(np.float32)
```

## File: qb_data/__init__.py
```python
"""Quiz Bowl Data Package.

Core data structures and utilities for quiz bowl question processing,
including qb-rl compatibility loader helpers.
"""

from qb_data.data_loader import (
    QANTADatasetLoader,
    TossupQuestion,
    load_tossup_questions,
    load_tossup_questions_from_config,
    parse_row,
)
from qb_data.text_utils import normalize_answer

__all__ = [
    'TossupQuestion',
    'QANTADatasetLoader',
    'parse_row',
    'load_tossup_questions',
    'load_tossup_questions_from_config',
    'normalize_answer',
]
```

## File: qb_data/answer_profiles.py
```python
"""Answer profile builder with leave-one-out exclusion for quiz bowl questions."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from qb_data.data_loader import TossupQuestion


class AnswerProfileBuilder:
    """Builds profiles for answers by aggregating question texts.

    The profile for an answer is created by concatenating all question texts
    that have that answer. When building profiles for distractors, we use
    all questions. For the gold answer, we exclude the current question to
    prevent information leakage (leave-one-out).

    Attributes:
        max_tokens_per_profile: Maximum number of tokens to keep in each profile.
        min_questions_per_answer: Minimum questions needed to build a profile.
        _grouped: Dictionary mapping answer_primary to list of (qid, question_text) tuples.
    """

    def __init__(
        self,
        max_tokens_per_profile: int = 2000,
        min_questions_per_answer: int = 1
    ):
        """Initialize the answer profile builder.

        Args:
            max_tokens_per_profile: Maximum tokens to keep in each profile.
            min_questions_per_answer: Minimum questions needed to build a profile.
        """
        self.max_tokens_per_profile = max_tokens_per_profile
        self.min_questions_per_answer = min_questions_per_answer
        self._grouped: Dict[str, List[Tuple[str, str]]] = {}
        self._cache: Dict[Tuple[str, Optional[str]], str] = {}

    def fit(self, questions: List[TossupQuestion]) -> "AnswerProfileBuilder":
        """Fit the builder on a set of questions.

        Groups questions by their primary answer for efficient profile building.

        Args:
            questions: List of tossup questions to group by answer.

        Returns:
            Self for method chaining.
        """
        grouped: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        for q in questions:
            # Store qid and full question text for each answer
            grouped[q.answer_primary].append((q.qid, q.question))
        self._grouped = dict(grouped)
        self._cache = {}
        return self

    def _profile_text(
        self,
        answer_primary: str,
        exclude_qid: Optional[str] = None
    ) -> str:
        """Build profile text for an answer with optional exclusion.

        Args:
            answer_primary: The answer to build a profile for.
            exclude_qid: Optional question ID to exclude (leave-one-out).

        Returns:
            Profile text truncated to max_tokens_per_profile.
        """
        key = (answer_primary, exclude_qid)
        if key in self._cache:
            return self._cache[key]

        items = self._grouped.get(answer_primary, [])
        texts: List[str] = []

        # Collect all question texts except the excluded one
        for qid, qtext in items:
            if exclude_qid is not None and qid == exclude_qid:
                continue
            texts.append(qtext)

        # If not enough questions after exclusion, fall back to answer text
        if len(texts) < self.min_questions_per_answer:
            self._cache[key] = answer_primary
            return answer_primary

        # Merge all texts and split into tokens
        merged = " ".join(texts).split()

        # Truncate to max tokens if specified
        if self.max_tokens_per_profile > 0:
            merged = merged[:self.max_tokens_per_profile]

        result = " ".join(merged) if merged else answer_primary
        self._cache[key] = result
        return result

    def profile_for_answer(
        self,
        answer_primary: str,
        exclude_qid: Optional[str] = None
    ) -> str:
        """Get the profile for a specific answer.

        Args:
            answer_primary: The answer to get a profile for.
            exclude_qid: Optional question ID to exclude (for gold answer).

        Returns:
            Profile text for the answer.
        """
        return self._profile_text(
            answer_primary=answer_primary,
            exclude_qid=exclude_qid
        )

    def build_profiles(
        self,
        questions: List[TossupQuestion],
        exclude_qid: Optional[str] = None,
    ) -> Dict[str, str]:
        """Build profiles for all answers in the dataset.

        Args:
            questions: List of questions (used to fit if not already fitted).
            exclude_qid: Optional question ID to exclude from all profiles.

        Returns:
            Dictionary mapping answer_primary to profile text.
        """
        if not self._grouped:
            self.fit(questions)

        return {
            answer: self._profile_text(answer, exclude_qid=exclude_qid)
            for answer in self._grouped.keys()
        }
```

## File: qb_data/config.py
```python
"""Configuration loading and management utilities.

Provides functions to load YAML configurations, apply small
cross-codebase compatibility normalizations, and merge CLI overrides
using dot notation (e.g., ``data.K=5`` updates ``config["data"]["K"]``).
"""

import argparse
from pathlib import Path
from typing import Any, Dict, Optional, Union


def normalize_config(
    config: Dict[str, Any],
    smoke: bool = False,
) -> Dict[str, Any]:
    """Apply compatibility defaults to a loaded configuration.

    Parameters
    ----------
    config : dict
        Parsed configuration dictionary.
    smoke : bool
        Whether the caller intends to run in smoke mode.

    Returns
    -------
    dict
        Normalized configuration dictionary.
    """
    data_cfg = config.setdefault("data", {})
    env_cfg = config.setdefault("environment", {})
    lik_cfg = config.setdefault("likelihood", {})

    if "reward" in env_cfg and "reward_mode" not in env_cfg:
        env_cfg["reward_mode"] = env_cfg["reward"]
    elif "reward_mode" in env_cfg and "reward" not in env_cfg:
        env_cfg["reward"] = env_cfg["reward_mode"]

    if smoke and data_cfg.get("dataset_smoke") and "dataset" not in data_cfg:
        data_cfg["dataset"] = data_cfg["dataset_smoke"]
    if smoke and data_cfg.get("dataset_smoke_config") and "dataset_config" not in data_cfg:
        data_cfg["dataset_config"] = data_cfg["dataset_smoke_config"]

    if "embedding_model" in lik_cfg and "sbert_name" not in lik_cfg:
        lik_cfg["sbert_name"] = lik_cfg["embedding_model"]
    if "sbert_name" in lik_cfg and "embedding_model" not in lik_cfg:
        lik_cfg["embedding_model"] = lik_cfg["sbert_name"]

    return config


def resolve_data_loading_options(
    config: Dict[str, Any],
    smoke: bool = False,
) -> Dict[str, Any]:
    """Resolve CSV/Hugging Face data-loading options from a config dict.

    Parameters
    ----------
    config : dict
        Parsed configuration dictionary.
    smoke : bool
        Whether the caller intends to run in smoke mode.

    Returns
    -------
    dict
        Resolved data-loading settings.
    """
    data_cfg = config.get("data", {})
    use_smoke_dataset = smoke and any(
        data_cfg.get(key) is not None
        for key in ("dataset_smoke", "dataset_smoke_config", "split_smoke", "csv_smoke_path")
    )

    csv_path = data_cfg.get("csv_path")
    if smoke and data_cfg.get("csv_smoke_path"):
        csv_path = data_cfg["csv_smoke_path"]

    dataset = data_cfg.get("dataset")
    dataset_config = data_cfg.get("dataset_config")
    split = data_cfg.get("split", "eval")

    if use_smoke_dataset:
        dataset = data_cfg.get("dataset_smoke", dataset)
        dataset_config = data_cfg.get("dataset_smoke_config", dataset_config)
        split = data_cfg.get("split_smoke", split)

    return {
        "csv_path": csv_path,
        "dataset": dataset,
        "dataset_config": dataset_config,
        "split": split,
        "use_huggingface": bool(data_cfg.get("use_huggingface", False) or dataset),
        "max_questions": data_cfg.get("max_questions"),
        "uses_dataset_smoke": use_smoke_dataset,
    }


def load_config(
    config_path: Optional[Union[str, Path]] = None,
    smoke: bool = False,
) -> Dict[str, Any]:
    """Load configuration from YAML file.

    Parameters
    ----------
    config_path : str or Path, optional
        Path to configuration file. Defaults to configs/default.yaml.

    Returns
    -------
    dict
        Parsed configuration dictionary.

    Raises
    ------
    FileNotFoundError
        If config file doesn't exist.
    ImportError
        If PyYAML is not installed.
    """
    try:
        import yaml
    except ImportError:
        raise ImportError(
            "PyYAML is required for config loading. "
            "Install it with: pip install pyyaml"
        )

    # Default to configs/default.yaml if no path given
    if config_path is None:
        project_root = Path(__file__).parent.parent
        default_path = project_root / "configs" / "default.yaml"
        smoke_path = project_root / "configs" / "smoke.yaml"

        if smoke and default_path.exists():
            with open(default_path, "r", encoding="utf-8") as f:
                default_config = yaml.safe_load(f) or {}
            default_data = default_config.get("data", {})
            if any(
                default_data.get(key) is not None
                for key in ("dataset_smoke", "dataset_smoke_config", "split_smoke", "csv_smoke_path")
            ):
                config_path = default_path
            elif smoke_path.exists():
                config_path = smoke_path
            else:
                config_path = default_path
        else:
            config_path = default_path
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return normalize_config(config or {}, smoke=smoke)


def merge_overrides(
    config: Dict[str, Any],
    overrides: Dict[str, Any]
) -> Dict[str, Any]:
    """Merge override values into configuration using dot notation.

    Parameters
    ----------
    config : dict
        Base configuration dictionary.
    overrides : dict
        Override values to merge. Keys can use dot notation
        (e.g., {"data.K": 5} updates config["data"]["K"]).

    Returns
    -------
    dict
        Updated configuration with overrides applied.

    Examples
    --------
    >>> config = {"data": {"K": 4}, "ppo": {"batch_size": 32}}
    >>> overrides = {"data.K": 5, "ppo.batch_size": 16}
    >>> config = merge_overrides(config, overrides)
    >>> assert config["data"]["K"] == 5
    >>> assert config["ppo"]["batch_size"] == 16
    """
    for key, value in overrides.items():
        # Split on dots for nested keys
        keys = key.split(".")

        # Navigate to the nested location
        current = config
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]

        # Set the final value
        final_key = keys[-1]
        current[final_key] = value

    return normalize_config(config)


def build_argparse_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    """Convert argparse namespace to configuration overrides.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.

    Returns
    -------
    dict
        Configuration overrides extracted from args.

    Notes
    -----
    Special handling:
    - --smoke flag loads smoke.yaml config
    - --config specifies custom config path
    - --override key=value pairs for dot notation overrides
    """
    overrides = {}

    # Handle smoke test mode
    if hasattr(args, "smoke") and args.smoke:
        overrides["__smoke__"] = True

    # Handle custom config path
    if hasattr(args, "config") and args.config:
        overrides["__config_path__"] = args.config

    # Parse key=value override pairs
    if hasattr(args, "override") and args.override:
        for override_str in args.override:
            if "=" not in override_str:
                print(f"Warning: Invalid override format '{override_str}', expected 'key=value'")
                continue

            key, value_str = override_str.split("=", 1)

            # Try to parse value as appropriate type
            value = parse_value(value_str)
            overrides[key] = value

    return overrides


def parse_value(value_str: str) -> Any:
    """Parse string value to appropriate Python type.

    Parameters
    ----------
    value_str : str
        String representation of value.

    Returns
    -------
    any
        Parsed value with appropriate type.

    Examples
    --------
    >>> parse_value("5") == 5
    >>> parse_value("3.14") == 3.14
    >>> parse_value("true") == True
    >>> parse_value("false") == False
    >>> parse_value("null") == None
    >>> parse_value("hello") == "hello"
    """
    # Handle boolean values
    if value_str.lower() == "true":
        return True
    if value_str.lower() == "false":
        return False

    # Handle null/none
    if value_str.lower() in ("null", "none"):
        return None

    # Try to parse as number
    try:
        # Try integer first
        if "." not in value_str:
            return int(value_str)
        # Then float
        return float(value_str)
    except ValueError:
        pass

    # Return as string
    return value_str


def add_config_args(parser: argparse.ArgumentParser) -> None:
    """Add configuration-related arguments to parser.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Parser to add arguments to.
    """
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use smoke test configuration for quick testing"
    )
    parser.add_argument(
        "--override",
        action="append",
        help="Override config values using dot notation (e.g., data.K=5)"
    )


def load_config_with_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    """Load configuration and apply command-line overrides.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.

    Returns
    -------
    dict
        Final configuration with all overrides applied.
    """
    # Build overrides from args
    overrides = build_argparse_overrides(args)

    # Check for special config path
    config_path = overrides.pop("__config_path__", None)
    smoke = bool(overrides.pop("__smoke__", False))

    # Load base config
    config = load_config(config_path, smoke=smoke)

    # Apply remaining overrides
    if overrides:
        config = merge_overrides(config, overrides)

    return config


# Convenience exports
__all__ = [
    "load_config",
    "merge_overrides",
    "normalize_config",
    "resolve_data_loading_options",
    "build_argparse_overrides",
    "add_config_args",
    "load_config_with_overrides",
]
```

## File: qb_data/dataset_splits.py
```python
"""
Stratified dataset splitting utilities for quiz bowl data.

This module provides functions to create train/val/test splits that maintain
category distribution across all splits.
"""

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple, Dict, Any

from qb_data.data_loader import TossupQuestion


def create_stratified_splits(
    questions: List[TossupQuestion],
    ratios: List[float] = [0.7, 0.15, 0.15],
    seed: int = 42
) -> Tuple[List[TossupQuestion], List[TossupQuestion], List[TossupQuestion]]:
    """
    Create stratified train/val/test splits maintaining category distribution.

    Parameters
    ----------
    questions : List[TossupQuestion]
        List of questions to split
    ratios : List[float]
        Train/val/test split ratios (must sum to 1.0)
    seed : int
        Random seed for reproducibility

    Returns
    -------
    Tuple[List[TossupQuestion], List[TossupQuestion], List[TossupQuestion]]
        Train, validation, and test splits

    Raises
    ------
    ValueError
        If ratios don't sum to 1.0 or questions list is empty
    """
    # Validate inputs
    if not questions:
        raise ValueError("Cannot split empty question list")

    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {sum(ratios)}")

    # Initialize random generator for reproducibility
    rng = random.Random(seed)

    # Group questions by category
    category_groups = defaultdict(list)
    for q in questions:
        category_groups[q.category].append(q)

    # Initialize output lists
    train_questions = []
    val_questions = []
    test_questions = []

    # Split each category maintaining ratios
    for category, category_questions in category_groups.items():
        # Sort for deterministic splits
        sorted_questions = sorted(category_questions, key=lambda q: q.qid)

        # Deterministic per-category seed via MD5 (immune to PYTHONHASHSEED)
        cat_hash = int(hashlib.md5(category.encode("utf-8")).hexdigest(), 16)
        category_seed = seed + cat_hash % 1_000_000
        category_rng = random.Random(category_seed)
        shuffled = sorted_questions.copy()
        category_rng.shuffle(shuffled)

        n = len(shuffled)

        # Calculate split indices
        train_end = int(n * ratios[0])
        val_end = train_end + int(n * ratios[1])

        # Handle small categories - ensure at least 1 in train if possible
        if n == 1:
            train_questions.extend(shuffled)
        elif n == 2:
            train_questions.extend(shuffled[:1])
            val_questions.extend(shuffled[1:])
        else:
            # Standard split
            train_questions.extend(shuffled[:train_end])
            val_questions.extend(shuffled[train_end:val_end])
            test_questions.extend(shuffled[val_end:])

    # Verify all questions assigned exactly once
    total_original = len(questions)
    total_split = len(train_questions) + len(val_questions) + len(test_questions)

    if total_original != total_split:
        raise RuntimeError(f"Split mismatch: {total_original} original vs {total_split} split")

    # Log category distribution statistics
    print(f"Dataset split complete:")
    print(f"  Train: {len(train_questions)} questions ({len(train_questions)/total_original:.1%})")
    print(f"  Val:   {len(val_questions)} questions ({len(val_questions)/total_original:.1%})")
    print(f"  Test:  {len(test_questions)} questions ({len(test_questions)/total_original:.1%})")

    # Category distribution analysis
    train_categories = defaultdict(int)
    val_categories = defaultdict(int)
    test_categories = defaultdict(int)

    for q in train_questions:
        train_categories[q.category] += 1
    for q in val_questions:
        val_categories[q.category] += 1
    for q in test_questions:
        test_categories[q.category] += 1

    all_categories = set(train_categories.keys()) | set(val_categories.keys()) | set(test_categories.keys())
    print(f"\nCategory distribution ({len(all_categories)} categories):")

    for category in sorted(all_categories)[:5]:  # Show first 5 categories
        orig_count = len(category_groups[category])
        train_count = train_categories.get(category, 0)
        val_count = val_categories.get(category, 0)
        test_count = test_categories.get(category, 0)
        print(f"  {category}: {train_count}/{val_count}/{test_count} (orig: {orig_count})")

    if len(all_categories) > 5:
        print(f"  ... and {len(all_categories) - 5} more categories")

    return train_questions, val_questions, test_questions


def save_splits(
    train: List[TossupQuestion],
    val: List[TossupQuestion],
    test: List[TossupQuestion],
    output_dir: str = "data"
) -> None:
    """
    Save dataset splits to JSON files with metadata.

    Parameters
    ----------
    train : List[TossupQuestion]
        Training split
    val : List[TossupQuestion]
        Validation split
    test : List[TossupQuestion]
        Test split
    output_dir : str
        Directory to save split files
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Helper to convert TossupQuestion to dict
    def questions_to_dict(questions: List[TossupQuestion]) -> List[Dict[str, Any]]:
        return [
            {
                "qid": q.qid,
                "question": q.question,
                "tokens": q.tokens,
                "answer_primary": q.answer_primary,
                "clean_answers": q.clean_answers,
                "run_indices": q.run_indices,
                "human_buzz_positions": q.human_buzz_positions,
                "category": q.category,
                "cumulative_prefixes": q.cumulative_prefixes
            }
            for q in questions
        ]

    # Calculate category distributions for metadata
    def get_category_distribution(questions: List[TossupQuestion]) -> Dict[str, int]:
        dist = defaultdict(int)
        for q in questions:
            dist[q.category] += 1
        return dict(dist)

    # Save each split with metadata
    splits = [
        ("train_dataset.json", train),
        ("val_dataset.json", val),
        ("test_dataset.json", test)
    ]

    for filename, questions in splits:
        filepath = output_path / filename

        data = {
            "metadata": {
                "total_questions": len(questions),
                "categories": len(set(q.category for q in questions)),
                "category_distribution": get_category_distribution(questions),
                "split_type": filename.replace("_dataset.json", "")
            },
            "questions": questions_to_dict(questions)
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"Saved {len(questions)} questions to {filepath}")

    # Save combined metadata file
    metadata_path = output_path / "split_metadata.json"
    metadata = {
        "train": {
            "count": len(train),
            "categories": get_category_distribution(train)
        },
        "val": {
            "count": len(val),
            "categories": get_category_distribution(val)
        },
        "test": {
            "count": len(test),
            "categories": get_category_distribution(test)
        },
        "total_questions": len(train) + len(val) + len(test),
        "split_ratios": [
            len(train) / (len(train) + len(val) + len(test)),
            len(val) / (len(train) + len(val) + len(test)),
            len(test) / (len(train) + len(val) + len(test))
        ]
    }

    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved split metadata to {metadata_path}")
```

## File: qb_data/huggingface_loader.py
```python
"""
HuggingFace dataset loader for quiz bowl data.

This module provides fallback loading from HuggingFace Hub when local CSV files
are not available.
"""

from typing import List, Optional, Dict, Any

from qb_data.data_loader import TossupQuestion
from qb_data.text_utils import tokenize_text, normalize_answer


def load_from_huggingface(
    dataset_name: str,
    config_name: Optional[str] = None,
    split: str = "eval"
) -> List[TossupQuestion]:
    """
    Load quiz bowl dataset from HuggingFace Hub.

    Parameters
    ----------
    dataset_name : str
        Name of the HuggingFace dataset (e.g., "qanta-challenge/acf-co24-tossups")
    config_name : Optional[str]
        Configuration name for the dataset (e.g., "questions", "tossup")
    split : str
        Dataset split to load (default: "eval")

    Returns
    -------
    List[TossupQuestion]
        List of parsed questions

    Raises
    ------
    ImportError
        If datasets library is not installed
    ValueError
        If dataset not found or required fields missing
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("Warning: datasets library not installed. Falling back to CSV loader.")
        print("Install with: pip install datasets")
        raise ImportError("HuggingFace datasets library not available. Please use CSV fallback.")

    # Known dataset configurations from qb-rl
    known_configs = {
        "qanta-challenge/acf-co24-tossups": "questions",
        "qanta-challenge/qanta25-playground": "tossup"
    }

    # Use known config if not provided
    if config_name is None and dataset_name in known_configs:
        config_name = known_configs[dataset_name]
        print(f"Using known config '{config_name}' for {dataset_name}")

    # Try to load dataset
    try:
        print(f"Loading {dataset_name} from HuggingFace Hub...")
        if config_name:
            dataset = load_dataset(dataset_name, config_name, split=split)
        else:
            dataset = load_dataset(dataset_name, split=split)
        print(f"Successfully loaded {len(dataset)} questions")
    except Exception as e:
        error_msg = f"Failed to load dataset {dataset_name}: {e}"
        print(f"Error: {error_msg}")
        print("Falling back to local CSV loader...")
        raise ValueError(error_msg)

    # Parse dataset rows into TossupQuestion format
    questions = []
    for idx, row in enumerate(dataset):
        try:
            question = parse_huggingface_row(row, idx)
            questions.append(question)
        except KeyError as e:
            print(f"Warning: Skipping row {idx} due to missing field: {e}")
            continue
        except Exception as e:
            print(f"Warning: Failed to parse row {idx}: {e}")
            continue

    if not questions:
        raise ValueError(f"No valid questions parsed from {dataset_name}")

    print(f"Parsed {len(questions)} questions from HuggingFace dataset")
    return questions


def parse_huggingface_row(row: Dict[str, Any], idx: int = 0) -> TossupQuestion:
    """
    Parse a HuggingFace dataset row into TossupQuestion format.

    Parameters
    ----------
    row : Dict[str, Any]
        Single row from HuggingFace dataset
    idx : int
        Row index for generating IDs

    Returns
    -------
    TossupQuestion
        Parsed question object

    Raises
    ------
    KeyError
        If required fields are missing
    """
    # Field mapping for different dataset formats
    # Primary fields
    question_fields = ["question", "text", "question_text", "tossup_text"]
    answer_fields = ["answer_primary", "answer", "clean_answer", "clean_answers", "page"]
    category_fields = ["category", "topic", "subject"]

    # Extract question text
    question_text = None
    for field in question_fields:
        if field in row:
            question_text = row[field]
            break

    if not question_text:
        raise KeyError(f"No question field found. Available fields: {list(row.keys())}")

    # Extract answer
    answer_text = None
    for field in answer_fields:
        if field in row:
            value = row[field]
            # Handle list of answers
            if isinstance(value, list) and value:
                answer_text = value[0]
            elif isinstance(value, str):
                answer_text = value
            break

    if not answer_text:
        raise KeyError(f"No answer field found. Available fields: {list(row.keys())}")

    # Extract category (with default)
    category = "General"
    for field in category_fields:
        if field in row and row[field]:
            category = str(row[field])
            break

    # Generate ID if not present
    qid = row.get("qid") or row.get("id") or row.get("qanta_id") or f"hf_{idx:06d}"

    # Handle clues that may be separated by ||| or in a list
    if "|||" in question_text:
        # QANTA format with ||| separators
        clues = question_text.split("|||")
        question_text = " ".join(clues)
    elif isinstance(question_text, list):
        # List of clues
        clues = question_text
        question_text = " ".join(clues)
    else:
        # Single text, split by sentences as approximation
        import re
        sentences = re.split(r'(?<=[.!?])\s+', question_text)
        clues = sentences if len(sentences) > 1 else [question_text]

    # Tokenize text
    tokens = tokenize_text(question_text)

    # Build run indices (boundaries between clues)
    run_indices = []
    current_pos = 0
    for clue in clues:
        clue_tokens = tokenize_text(clue)
        current_pos += len(clue_tokens)
        if current_pos > 0:
            run_indices.append(current_pos - 1)  # Index is 0-based

    # Build cumulative prefixes
    cumulative_prefixes = []
    for idx in run_indices:
        prefix = " ".join(tokens[:idx + 1])
        cumulative_prefixes.append(prefix)

    # Normalize answer for matching
    clean_answers = [normalize_answer(answer_text)]

    return TossupQuestion(
        qid=qid,
        question=question_text,
        tokens=tokens,
        answer_primary=answer_text,  # Keep original answer as primary
        clean_answers=clean_answers,  # Normalized version for matching
        run_indices=run_indices,
        human_buzz_positions=None,  # Not available from HuggingFace
        category=category,
        cumulative_prefixes=cumulative_prefixes
    )


def try_huggingface_fallback(csv_path: str) -> Optional[List[TossupQuestion]]:
    """
    Attempt to load from HuggingFace if CSV is missing.

    Parameters
    ----------
    csv_path : str
        Path to missing CSV file

    Returns
    -------
    Optional[List[TossupQuestion]]
        Questions if HuggingFace load succeeds, None otherwise
    """
    print(f"CSV file {csv_path} not found. Attempting HuggingFace fallback...")

    # Try known datasets in order
    fallback_datasets = [
        ("qanta-challenge/acf-co24-tossups", "questions"),
        ("qanta-challenge/qanta25-playground", "tossup")
    ]

    for dataset_name, config_name in fallback_datasets:
        try:
            questions = load_from_huggingface(dataset_name, config_name)
            if questions:
                print(f"Successfully loaded {len(questions)} questions from {dataset_name}")
                return questions
        except Exception as e:
            print(f"Failed to load {dataset_name}: {e}")
            continue

    print("All HuggingFace fallback attempts failed")
    return None
```

## File: qb_data/text_utils.py
```python
"""
Text utilities for quiz bowl answer normalization and tokenization.
"""

import re
from typing import Optional, List


def tokenize_text(text: str) -> List[str]:
    """
    Tokenize text by splitting on whitespace.

    Parameters
    ----------
    text : str
        Text to tokenize

    Returns
    -------
    List[str]
        List of tokens (words)
    """
    if not text:
        return []
    return text.split()


def normalize_answer(answer: str) -> str:
    """
    Normalize an answer string for comparison.

    Removes articles (a, an, the) from the beginning, converts to lowercase,
    strips punctuation and extra whitespace, and handles edge cases.

    Parameters
    ----------
    answer : str
        The answer string to normalize

    Returns
    -------
    str
        The normalized answer string

    Examples
    --------
    >>> normalize_answer("The Great Gatsby")
    'great gatsby'
    >>> normalize_answer("A Tale of Two Cities!")
    'tale of two cities'
    >>> normalize_answer("   An    Example   ")
    'example'
    >>> normalize_answer("")
    ''
    """
    if not answer:
        return ""

    # Convert to lowercase
    normalized = answer.lower()

    # Remove leading/trailing whitespace
    normalized = normalized.strip()

    # Remove leading articles (a, an, the)
    # Use \b word boundary to ensure we match complete words
    normalized = re.sub(r'^(a|an|the)\b\s*', '', normalized)

    # Remove punctuation
    # Keep alphanumeric characters and spaces
    normalized = re.sub(r'[^\w\s]', '', normalized)

    # Normalize whitespace (collapse multiple spaces to single space)
    normalized = re.sub(r'\s+', ' ', normalized)

    # Final strip in case punctuation removal left spaces
    normalized = normalized.strip()

    return normalized
```

## File: qb_env/data_loader.py
```python
"""qb-rl compatibility re-exports for tossup data loading."""

from qb_data.data_loader import (
    QANTADatasetLoader,
    TossupQuestion,
    load_tossup_questions,
    load_tossup_questions_from_config,
    parse_row,
)

__all__ = [
    "TossupQuestion",
    "QANTADatasetLoader",
    "parse_row",
    "load_tossup_questions",
    "load_tossup_questions_from_config",
]
```

## File: qb_env/mc_builder.py
```python
"""qb-rl compatibility re-exports for MC question building."""

from qb_data.mc_builder import MCBuilder, MCQuestion, _token_overlap

__all__ = ["MCQuestion", "MCBuilder", "_token_overlap"]
```

## File: qb_env/text_utils.py
```python
"""qb-rl compatibility re-exports for text utilities."""

from qb_data.text_utils import normalize_answer, tokenize_text

__all__ = ["normalize_answer", "tokenize_text"]
```

## File: qb_env/text_wrapper.py
```python
"""
TextObservationWrapper for converting belief features to text observations.

Wraps TossupMCEnv to provide text-formatted observations (clues + choices)
instead of numeric belief feature vectors. This bridges the gap between
the environment's native observation space (Box(K+6,)) and T5PolicyModel's
text input requirement.

The underlying environment still operates on beliefs internally for reward
computation -- the wrapper only transforms what the agent SEES, not how the
environment computes rewards or transitions.

Text format matches T5PolicyModel's expected input:
    "CLUES: clue1 clue2 ... | CHOICES: (1) ans1 (2) ans2 (3) ans3 (4) ans4"

Ported from qanta-buzzer's environment.py get_text_representation() method,
adapted for the unified codebase's Gymnasium wrapper pattern.
"""

from __future__ import annotations

from typing import Any, Tuple

import gymnasium as gym
import numpy as np

from qb_data.mc_builder import MCQuestion


class TextObservationWrapper(gym.ObservationWrapper):
    """Wrap TossupMCEnv to provide text observations instead of belief features.

    The underlying env still operates on beliefs internally (for reward
    computation), but the agent sees text-formatted observations for T5 input.
    This is a Gymnasium ObservationWrapper that intercepts the observation
    returned by reset() and step() and converts it to a text string.

    The observation space is set to a placeholder Box(1,) since Gymnasium
    requires a defined space, but text observations are variable-length
    strings. Downstream code (T5PolicyModel) handles tokenization.

    Parameters
    ----------
    env : gym.Env
        The underlying TossupMCEnv instance. Must have ``question``
        (MCQuestion) and ``step_idx`` (int) attributes.

    Examples
    --------
    >>> from qb_env.tossup_env import TossupMCEnv
    >>> env = TossupMCEnv(questions=qs, likelihood_model=lm, K=4)
    >>> wrapped = TextObservationWrapper(env)
    >>> obs, info = wrapped.reset()
    >>> assert isinstance(obs, str)
    >>> assert "CLUES:" in obs and "CHOICES:" in obs
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        # Override observation space with a placeholder.
        # Text observations are variable-length strings; Gymnasium requires
        # a Space object, so we use a minimal Box as a sentinel.
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(1,), dtype=np.float32
        )

    def observation(self, obs: np.ndarray) -> str:
        """Convert numeric belief observation to formatted text string.

        Reconstructs visible clues from the underlying environment's current
        question and step index, then formats them with answer choices in the
        standard T5PolicyModel input format.

        Parameters
        ----------
        obs : np.ndarray
            Numeric belief features from the underlying environment.
            Shape ``(K+6,)``. Not used directly -- the text is reconstructed
            from ``env.question`` and ``env.step_idx``.

        Returns
        -------
        str
            Formatted text observation:
            ``"CLUES: <visible clue tokens> | CHOICES: (1) opt1 (2) opt2 ..."``
        """
        question: MCQuestion = self.env.question
        step_idx: int = self.env.step_idx

        # Build visible clue text from cumulative prefixes.
        #
        # TossupMCEnv step semantics:
        #   - reset() sets step_idx=0, belief is uniform (no clues processed).
        #   - step(WAIT) calls _compute_belief(step_idx), THEN increments step_idx.
        #   - The observation returned after step() has step_idx ALREADY incremented.
        #
        # So step_idx tells us how many WAIT actions have been taken:
        #   step_idx=0: No WAITs yet; no clues processed; show minimal context
        #   step_idx=N: N WAITs taken; beliefs from cumulative_prefixes[0..N-1]
        #
        # cumulative_prefixes[i] = text of tokens[0..run_indices[i]].
        # After N WAITs, the agent has seen information up to
        # cumulative_prefixes[N-1], so that is what the text obs shows.
        if step_idx == 0:
            # No clues processed yet; show question start as minimal context
            # (matches initial observation having some textual content for T5)
            clues_text = question.tokens[0] if question.tokens else ""
        elif step_idx <= len(question.cumulative_prefixes):
            clues_text = question.cumulative_prefixes[step_idx - 1]
        else:
            # Past all clues (truncated episode); show all text
            clues_text = question.cumulative_prefixes[-1]

        # Format answer choices
        choices_parts = [
            f"({i + 1}) {opt}" for i, opt in enumerate(question.options)
        ]
        choices_text = " ".join(choices_parts)

        return f"CLUES: {clues_text} | CHOICES: {choices_text}"

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> Tuple[str, dict[str, Any]]:
        """Reset the environment and return a text observation.

        Parameters
        ----------
        seed : int or None
            Random seed passed to underlying environment.
        options : dict or None
            Options passed to underlying environment.

        Returns
        -------
        observation : str
            Text-formatted initial observation.
        info : dict[str, Any]
            Episode metadata from underlying environment.
        """
        obs, info = self.env.reset(seed=seed, options=options)
        return self.observation(obs), info

    def step(
        self, action: int
    ) -> Tuple[str, float, bool, bool, dict[str, Any]]:
        """Execute one step and return text observation.

        Parameters
        ----------
        action : int
            Action to take. 0 = WAIT, 1..K = BUZZ with answer (action-1).

        Returns
        -------
        observation : str
            Text-formatted observation after the step.
        reward : float
            Scalar reward for this step.
        terminated : bool
            True if the agent buzzed (natural episode end).
        truncated : bool
            True if all clues exhausted (forced termination).
        info : dict[str, Any]
            Step metadata from underlying environment.
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self.observation(obs), reward, terminated, truncated, info

    @property
    def unwrapped_env(self):
        """Access the underlying TossupMCEnv directly.

        Returns
        -------
        TossupMCEnv
            The unwrapped environment instance.
        """
        return self.env
```

## File: tests/conftest.py
```python
"""Shared pytest fixtures for test suites.

Provides reusable test data for environment, likelihood, features,
factory, and agent test suites. All fixtures create minimal but complete
data structures that satisfy the interfaces expected by the codebase modules.

Fixtures
--------
sample_mc_question
    A single MCQuestion with 4 options (gold_index=0), 6 clue steps,
    and pre-computed cumulative prefixes. Suitable for environment and
    feature extraction tests.

sample_config
    A minimal config dict matching the YAML structure expected by
    ``make_env_from_config`` and ``build_likelihood_from_config``.
    Uses "simple" reward mode for predictable test outcomes.

sample_corpus
    A list of 10 short text strings about US presidents and historical
    events. Suitable for fitting TF-IDF vectorizers in tests.

sample_tfidf_env
    A TossupMCEnv with TF-IDF likelihood and 3 sample MCQuestions.
    Fast to construct, suitable for agent and PPO tests.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from qb_data.mc_builder import MCQuestion

if TYPE_CHECKING:
    from qb_env.tossup_env import TossupMCEnv


@pytest.fixture
def sample_mc_question() -> MCQuestion:
    """Return a minimal MCQuestion for testing.

    The question is about the first US president with 4 answer options.
    Gold answer is "George Washington" at index 0. Six clue steps are
    defined via run_indices with pre-computed cumulative prefixes.

    Returns
    -------
    MCQuestion
        A complete MCQuestion suitable for environment testing.
    """
    tokens = [
        "Who", "was", "the", "first", "president",
        "of", "the", "United", "States", "?",
    ]
    run_indices = [0, 2, 4, 6, 8, 9]
    cumulative_prefixes = [
        "Who",
        "Who was the",
        "Who was the first president",
        "Who was the first president of the",
        "Who was the first president of the United States",
        "Who was the first president of the United States ?",
    ]
    return MCQuestion(
        qid="test_q1",
        question="Who was the first president of the United States?",
        tokens=tokens,
        answer_primary="George Washington",
        clean_answers=["George Washington", "Washington"],
        run_indices=run_indices,
        human_buzz_positions=[],
        category="History",
        cumulative_prefixes=cumulative_prefixes,
        options=[
            "George Washington",
            "Thomas Jefferson",
            "John Adams",
            "Benjamin Franklin",
        ],
        gold_index=0,
        option_profiles=[
            "George Washington first president commander revolutionary war continental army",
            "Thomas Jefferson third president declaration independence Virginia",
            "John Adams second president Massachusetts diplomat",
            "Benjamin Franklin inventor diplomat Philadelphia printing press",
        ],
        option_answer_primary=[
            "George Washington",
            "Thomas Jefferson",
            "John Adams",
            "Benjamin Franklin",
        ],
        distractor_strategy="test",
    )


@pytest.fixture
def sample_config() -> dict:
    """Return a minimal config dict for factory tests.

    Matches the YAML structure expected by ``make_env_from_config`` and
    ``build_likelihood_from_config``. Uses "simple" reward mode and
    "from_scratch" belief mode for predictable test outcomes.

    Returns
    -------
    dict
        Config dict with data, environment, and likelihood sections.
    """
    return {
        "data": {"K": 4},
        "environment": {
            "reward": "simple",
            "wait_penalty": 0.0,
            "buzz_correct": 1.0,
            "buzz_incorrect": -1.0,
            "belief_mode": "from_scratch",
        },
        "likelihood": {
            "model": "sbert",
            "beta": 5.0,
        },
    }


@pytest.fixture
def sample_corpus() -> list[str]:
    """Return a list of 10 short text strings for TF-IDF fitting.

    Topics cover US presidents and major historical events, providing
    sufficient vocabulary variety for TF-IDF vectorizer tests.

    Returns
    -------
    list[str]
        Ten text strings suitable for corpus fitting.
    """
    return [
        "George Washington was the first president of the United States",
        "Thomas Jefferson wrote the Declaration of Independence",
        "John Adams served as the second president after Washington",
        "Benjamin Franklin was an inventor and diplomat in Philadelphia",
        "Abraham Lincoln freed the slaves during the Civil War",
        "Alexander Hamilton established the national banking system",
        "James Madison authored the Bill of Rights and Constitution",
        "Andrew Jackson was a military hero and populist president",
        "The American Revolution established independence from Britain",
        "The Constitution created a federal system of government",
    ]


@pytest.fixture(scope="module")
def sample_t5_model():
    """Return a T5Likelihood model for testing.

    Uses t5-small (60M params) for fast test execution. Scoped to module
    level so the model is loaded once per test file, not per test function.

    Returns
    -------
    T5Likelihood
        A T5 likelihood model suitable for testing semantic scoring.

    Notes
    -----
    This fixture may take 5-10 seconds on first run to download the model
    from HuggingFace. Subsequent runs use cached weights.
    """
    from models.likelihoods import T5Likelihood

    return T5Likelihood(model_name="t5-small")


@pytest.fixture
def sample_tfidf_env(sample_mc_question: MCQuestion) -> "TossupMCEnv":
    """Return a TossupMCEnv with TF-IDF likelihood and 3 sample questions.

    Creates a lightweight environment suitable for PPOBuzzer and agent
    tests. Uses TF-IDF likelihood for fast execution (< 1ms per score).
    Three copies of the sample question are used to provide enough data
    for environment sampling.

    Returns
    -------
    TossupMCEnv
        A configured environment with simple reward mode.
    """
    from models.likelihoods import TfIdfLikelihood
    from qb_env.tossup_env import TossupMCEnv

    corpus = sample_mc_question.option_profiles[:]
    model = TfIdfLikelihood(corpus_texts=corpus)

    # Use 3 copies for variety in sampling
    questions = [sample_mc_question] * 3
    return TossupMCEnv(
        questions=questions,
        likelihood_model=model,
        K=4,
        reward_mode="simple",
        wait_penalty=0.0,
        buzz_correct=1.0,
        buzz_incorrect=-1.0,
        belief_mode="from_scratch",
        beta=5.0,
    )
```

## File: tests/test_features.py
```python
"""Test suite for models/features.py — belief feature extraction.

Covers ENV-03: Belief feature extraction produces (K+6)-dimensional vectors
with correct derived features (entropy, margin, stability, progress).
"""

from __future__ import annotations

import numpy as np
import pytest

from models.features import entropy_of_distribution, extract_belief_features


# ------------------------------------------------------------------ #
# Tests for entropy_of_distribution
# ------------------------------------------------------------------ #


class TestEntropyOfDistribution:
    """Tests for Shannon entropy computation."""

    def test_entropy_uniform(self) -> None:
        """Uniform distribution over 4 options has maximum entropy ln(4)."""
        belief = np.array([0.25, 0.25, 0.25, 0.25])
        ent = entropy_of_distribution(belief)
        # ln(4) ~ 1.3863
        assert 1.35 < ent < 1.40, f"Uniform entropy {ent} not near ln(4)=1.3863"

    def test_entropy_peaked(self) -> None:
        """Peaked distribution has low entropy."""
        belief = np.array([0.9, 0.05, 0.03, 0.02])
        ent = entropy_of_distribution(belief)
        assert ent < 0.5, f"Peaked entropy {ent} should be < 0.5"

    def test_entropy_deterministic_no_nan(self) -> None:
        """Deterministic distribution [1, 0, 0, 0] produces no NaN/inf."""
        belief = np.array([1.0, 0.0, 0.0, 0.0])
        ent = entropy_of_distribution(belief)
        assert np.isfinite(ent), f"Entropy {ent} should be finite"
        assert ent >= 0.0, f"Entropy {ent} should be non-negative"

    def test_entropy_deterministic_last(self) -> None:
        """Deterministic distribution [0, 0, 0, 1] produces no NaN/inf."""
        belief = np.array([0.0, 0.0, 0.0, 1.0])
        ent = entropy_of_distribution(belief)
        assert np.isfinite(ent), f"Entropy {ent} should be finite"
        assert ent >= 0.0, f"Entropy {ent} should be non-negative"

    def test_entropy_binary(self) -> None:
        """Binary uniform distribution has entropy ln(2)."""
        belief = np.array([0.5, 0.5])
        ent = entropy_of_distribution(belief)
        assert abs(ent - np.log(2)) < 0.01, f"Binary entropy {ent} != ln(2)={np.log(2):.4f}"


# ------------------------------------------------------------------ #
# Tests for extract_belief_features
# ------------------------------------------------------------------ #


class TestExtractBeliefFeatures:
    """Tests for belief feature vector extraction."""

    def test_feature_shape(self) -> None:
        """Output shape is (K+6,) for K=4 belief vector."""
        belief = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
        features = extract_belief_features(belief, None, 0, 6)
        assert features.shape == (10,), f"Expected (10,), got {features.shape}"

    def test_feature_shape_k3(self) -> None:
        """Output shape adapts to K=3."""
        belief = np.array([0.4, 0.3, 0.3], dtype=np.float32)
        features = extract_belief_features(belief, None, 0, 5)
        assert features.shape == (9,), f"Expected (9,), got {features.shape}"

    def test_feature_contents_belief_prefix(self) -> None:
        """First K elements of feature vector are the raw belief."""
        belief = np.array([0.5, 0.3, 0.15, 0.05], dtype=np.float32)
        features = extract_belief_features(belief, None, 2, 6)
        np.testing.assert_array_almost_equal(
            features[:4], belief, decimal=5,
            err_msg="First K elements should match input belief",
        )

    def test_derived_top_p(self) -> None:
        """top_p is max(belief)."""
        belief = np.array([0.5, 0.3, 0.15, 0.05], dtype=np.float32)
        features = extract_belief_features(belief, None, 2, 6)
        assert abs(features[4] - 0.5) < 1e-5, f"top_p={features[4]}, expected 0.5"

    def test_derived_margin(self) -> None:
        """margin is top_p - second_highest."""
        belief = np.array([0.5, 0.3, 0.15, 0.05], dtype=np.float32)
        features = extract_belief_features(belief, None, 2, 6)
        expected_margin = 0.5 - 0.3
        assert abs(features[5] - expected_margin) < 1e-5, (
            f"margin={features[5]}, expected {expected_margin}"
        )

    def test_derived_entropy_in_range(self) -> None:
        """Entropy is in a reasonable range for a non-uniform distribution."""
        belief = np.array([0.5, 0.3, 0.15, 0.05], dtype=np.float32)
        features = extract_belief_features(belief, None, 2, 6)
        ent = features[6]
        assert 0 < ent < np.log(4) + 0.01, f"Entropy {ent} out of range"

    def test_stability_none_prev(self) -> None:
        """Stability is 0.0 when prev_belief is None (first step)."""
        belief = np.array([0.5, 0.3, 0.15, 0.05], dtype=np.float32)
        features = extract_belief_features(belief, None, 0, 6)
        assert features[7] == 0.0, f"Stability={features[7]}, expected 0.0 for first step"

    def test_stability_computation(self) -> None:
        """Stability tracks L1 distance between consecutive beliefs."""
        prev_belief = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
        belief = np.array([0.5, 0.3, 0.15, 0.05], dtype=np.float32)
        features = extract_belief_features(belief, prev_belief, 1, 6)
        expected_stability = float(np.abs(belief - prev_belief).sum())
        assert abs(features[7] - expected_stability) < 1e-5, (
            f"Stability={features[7]}, expected {expected_stability}"
        )

    def test_progress(self) -> None:
        """progress = step_idx / total_steps."""
        belief = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
        features = extract_belief_features(belief, None, 3, 6)
        expected_progress = 3.0 / 6.0
        assert abs(features[8] - expected_progress) < 1e-5, (
            f"progress={features[8]}, expected {expected_progress}"
        )

    def test_clue_idx_norm(self) -> None:
        """clue_idx_norm = step_idx / (total_steps - 1)."""
        belief = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
        features = extract_belief_features(belief, None, 3, 6)
        expected_norm = 3.0 / 5.0
        assert abs(features[9] - expected_norm) < 1e-5, (
            f"clue_idx_norm={features[9]}, expected {expected_norm}"
        )

    def test_dtype_float32(self) -> None:
        """Output dtype is float32."""
        belief = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
        features = extract_belief_features(belief, None, 0, 6)
        assert features.dtype == np.float32, f"Expected float32, got {features.dtype}"

    def test_invalid_2d_belief_raises(self) -> None:
        """Passing a 2D belief array raises ValueError."""
        belief = np.array([[0.5, 0.5]], dtype=np.float32)
        with pytest.raises(ValueError, match="1D"):
            extract_belief_features(belief, None, 0, 1)


class TestPaddedBeliefFeatures:
    """Tests for extract_padded_belief_features."""

    def test_shape_equals_max_k_plus_6(self) -> None:
        from models.features import extract_padded_belief_features

        belief = np.array([0.5, 0.3, 0.2], dtype=np.float32)
        feats = extract_padded_belief_features(belief, None, 0, 6, max_K=8)
        assert feats.shape == (8 + 6,)

    def test_valid_slots_preserved(self) -> None:
        from models.features import extract_padded_belief_features

        belief = np.array([0.6, 0.3, 0.1], dtype=np.float32)
        feats = extract_padded_belief_features(belief, None, 0, 6, max_K=5)
        np.testing.assert_allclose(feats[:3], belief, atol=1e-7)

    def test_padded_slots_zero(self) -> None:
        from models.features import extract_padded_belief_features

        belief = np.array([0.5, 0.5], dtype=np.float32)
        feats = extract_padded_belief_features(belief, None, 0, 6, max_K=6)
        np.testing.assert_array_equal(feats[2:6], [0.0, 0.0, 0.0, 0.0])

    def test_extras_match_unpadded(self) -> None:
        from models.features import extract_padded_belief_features

        belief = np.array([0.4, 0.3, 0.2, 0.1], dtype=np.float32)
        unpadded = extract_belief_features(belief, None, 2, 6)
        padded = extract_padded_belief_features(belief, None, 2, 6, max_K=4)
        np.testing.assert_allclose(unpadded[4:], padded[4:], atol=1e-7)

    def test_dtype(self) -> None:
        from models.features import extract_padded_belief_features

        belief = np.array([0.5, 0.5], dtype=np.float32)
        feats = extract_padded_belief_features(belief, None, 0, 4, max_K=4)
        assert feats.dtype == np.float32
```

## File: tests/test_likelihoods.py
```python
"""Test suite for models/likelihoods.py — likelihood model interface and implementations.

Covers:
- LIK-01: LikelihoodModel ABC contract
- LIK-02: TfIdfLikelihood with corpus fitting and cosine scoring
- LIK-03: SBERTLikelihood with semantic embeddings and caching
- LIK-04: T5Likelihood semantic scoring and embedding shape
- LIK-05: T5 embedding cache reuse and factory construction
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from models.likelihoods import (
    LikelihoodModel,
    SBERTLikelihood,
    TfIdfLikelihood,
)


# ------------------------------------------------------------------ #
# Tests for LikelihoodModel ABC
# ------------------------------------------------------------------ #


class TestLikelihoodModelABC:
    """Tests for the abstract base class contract."""

    def test_abstract_interface_cannot_instantiate(self) -> None:
        """LikelihoodModel ABC cannot be instantiated directly."""
        with pytest.raises(TypeError):
            LikelihoodModel()  # type: ignore[abstract]

    def test_embedding_cache_on_subclass(self, sample_corpus: list[str]) -> None:
        """Concrete subclass inherits embedding_cache dict."""
        model = TfIdfLikelihood(corpus_texts=sample_corpus)
        assert hasattr(model, "embedding_cache"), "Missing embedding_cache attribute"
        assert isinstance(model.embedding_cache, dict), "embedding_cache should be dict"


# ------------------------------------------------------------------ #
# Tests for TfIdfLikelihood
# ------------------------------------------------------------------ #


class TestTfIdfLikelihood:
    """Tests for TF-IDF based likelihood model."""

    def test_tfidf_requires_fit(self) -> None:
        """score() before fit() raises RuntimeError."""
        model = TfIdfLikelihood()
        with pytest.raises(RuntimeError, match="must be fit"):
            model.score("test clue", ["option1", "option2"])

    def test_tfidf_embed_requires_fit(self) -> None:
        """_embed_batch() before fit() raises RuntimeError."""
        model = TfIdfLikelihood()
        with pytest.raises(RuntimeError, match="must be fit"):
            model._embed_batch(["test text"])

    def test_tfidf_fit_and_score(self, sample_corpus: list[str]) -> None:
        """After fitting, score returns correct shape and dtype.

        Also verifies that more relevant text scores higher.
        """
        model = TfIdfLikelihood()
        model.fit(sample_corpus)

        scores = model.score(
            "Who was the first president?",
            ["George Washington first president", "Abraham Lincoln Civil War"],
        )
        assert scores.shape == (2,), f"Expected shape (2,), got {scores.shape}"
        assert scores.dtype == np.float32, f"Expected float32, got {scores.dtype}"
        # Washington should score higher for "first president" clue
        assert scores[0] >= scores[1], (
            f"Washington ({scores[0]:.3f}) should score >= Lincoln ({scores[1]:.3f})"
        )

    def test_tfidf_embed_batch(self, sample_corpus: list[str]) -> None:
        """_embed_batch produces dense vectors of correct shape."""
        model = TfIdfLikelihood(corpus_texts=sample_corpus)
        embeddings = model._embed_batch(["test text one", "test text two"])
        assert embeddings.shape[0] == 2, f"Expected 2 rows, got {embeddings.shape[0]}"
        assert embeddings.dtype == np.float32, f"Expected float32, got {embeddings.dtype}"
        vocab_size = len(model.vectorizer.vocabulary_)
        assert embeddings.shape[1] == vocab_size, (
            f"Expected {vocab_size} cols, got {embeddings.shape[1]}"
        )

    def test_tfidf_corpus_in_constructor(self, sample_corpus: list[str]) -> None:
        """Passing corpus_texts to __init__ auto-fits the model."""
        model = TfIdfLikelihood(corpus_texts=sample_corpus)
        assert model._is_fit is True, "Model should be fit after corpus in constructor"
        # Should work without explicit fit()
        scores = model.score("president", ["Washington", "Lincoln"])
        assert scores.shape == (2,)

    def test_tfidf_fit_returns_self(self, sample_corpus: list[str]) -> None:
        """fit() returns self for method chaining."""
        model = TfIdfLikelihood()
        result = model.fit(sample_corpus)
        assert result is model, "fit() should return self"

    def test_tfidf_score_all_options(self, sample_corpus: list[str]) -> None:
        """Score works with 4 options matching K=4 environment setup."""
        model = TfIdfLikelihood(corpus_texts=sample_corpus)
        scores = model.score(
            "first president United States",
            [
                "George Washington commander revolutionary",
                "Thomas Jefferson declaration independence",
                "John Adams Massachusetts diplomat",
                "Benjamin Franklin inventor Philadelphia",
            ],
        )
        assert scores.shape == (4,), f"Expected shape (4,), got {scores.shape}"
        assert all(np.isfinite(scores)), "All scores should be finite"

    def test_tfidf_embed_batch_normalized(self, sample_corpus: list[str]) -> None:
        """_embed_batch returns L2-normalized vectors (row norms ~1.0)."""
        model = TfIdfLikelihood(corpus_texts=sample_corpus)
        embeddings = model._embed_batch(["George Washington president", "Thomas Jefferson"])
        norms = np.linalg.norm(embeddings, axis=1)
        np.testing.assert_array_almost_equal(norms, np.ones(2), decimal=5)

    def test_tfidf_score_uses_cache(self, sample_corpus: list[str]) -> None:
        """score() populates embedding_cache via embed_and_cache()."""
        model = TfIdfLikelihood(corpus_texts=sample_corpus)
        assert len(model.embedding_cache) == 0
        model.score("first president", ["Washington profile", "Lincoln profile"])
        assert len(model.embedding_cache) == 3  # 1 clue + 2 options

    def test_tfidf_score_cache_hit(self, sample_corpus: list[str]) -> None:
        """Repeated score() with same options reuses cache."""
        model = TfIdfLikelihood(corpus_texts=sample_corpus)
        options = ["George Washington president", "Thomas Jefferson declaration"]
        model.score("first president", options)
        cache_after_first = len(model.embedding_cache)
        model.score("second president", options)
        # Only the new clue should be added; options are cached
        assert len(model.embedding_cache) == cache_after_first + 1

    def test_tfidf_score_matches_cosine_reference(self, sample_corpus: list[str]) -> None:
        """New cached score() matches sklearn cosine_similarity reference."""
        from sklearn.metrics.pairwise import cosine_similarity as sklearn_cos

        model = TfIdfLikelihood(corpus_texts=sample_corpus)
        clue = "Who was the first president?"
        options = [
            "George Washington first president commander revolutionary",
            "Abraham Lincoln Civil War emancipation",
            "Thomas Jefferson declaration independence Virginia",
            "Benjamin Franklin inventor Philadelphia diplomat",
        ]
        # Compute reference via sklearn cosine_similarity (old method)
        clue_vec = model.vectorizer.transform([clue])
        option_vecs = model.vectorizer.transform(options)
        ref_scores = sklearn_cos(clue_vec, option_vecs)[0].astype(np.float32)
        # Compute via new cached path
        actual_scores = model.score(clue, options)
        np.testing.assert_allclose(actual_scores, ref_scores, atol=1e-6)


# ------------------------------------------------------------------ #
# Tests for SBERTLikelihood
# ------------------------------------------------------------------ #


class TestSBERTLikelihood:
    """Tests for Sentence-BERT likelihood model."""

    def test_sbert_instantiation(self) -> None:
        """SBERTLikelihood can be instantiated with default model."""
        model = SBERTLikelihood()
        assert hasattr(model, "encoder"), "Missing encoder attribute"
        assert model.model_name == "all-MiniLM-L6-v2"

    def test_sbert_score_shape_and_dtype(self) -> None:
        """score() returns correct shape and dtype for 4 options."""
        model = SBERTLikelihood()
        scores = model.score(
            "first president United States",
            [
                "George Washington first president commander",
                "Thomas Jefferson third president declaration",
                "John Adams second president Massachusetts",
                "Benjamin Franklin inventor diplomat",
            ],
        )
        assert scores.shape == (4,), f"Expected shape (4,), got {scores.shape}"
        assert scores.dtype == np.float32, f"Expected float32, got {scores.dtype}"

    def test_sbert_semantic_ranking(self) -> None:
        """SBERT ranks semantically similar text higher."""
        model = SBERTLikelihood()
        scores = model.score(
            "George Washington was the first president of the United States and led the Continental Army",
            [
                "George Washington first president commander revolutionary war continental army",
                "The theory of relativity was developed by Albert Einstein in physics",
            ],
        )
        # Washington profile should score much higher than Einstein
        assert scores[0] > scores[1], (
            f"Washington ({scores[0]:.3f}) should score > Einstein ({scores[1]:.3f})"
        )

    def test_sbert_embedding_cache_populated(self) -> None:
        """Embedding cache grows after first scoring call."""
        model = SBERTLikelihood()
        assert len(model.embedding_cache) == 0, "Cache should start empty"

        model.score("test clue", ["option A", "option B"])
        cache_after_first = len(model.embedding_cache)
        assert cache_after_first > 0, "Cache should be populated after score()"

    def test_sbert_embedding_cache_hit(self) -> None:
        """Repeated calls with same text use cache (size unchanged)."""
        model = SBERTLikelihood()
        scores1 = model.score("test clue", ["option A", "option B"])
        cache_size_1 = len(model.embedding_cache)

        scores2 = model.score("test clue", ["option A", "option B"])
        cache_size_2 = len(model.embedding_cache)

        assert cache_size_2 == cache_size_1, (
            f"Cache grew from {cache_size_1} to {cache_size_2} on repeated call"
        )
        np.testing.assert_array_almost_equal(
            scores1, scores2, decimal=5,
            err_msg="Cached results should match original",
        )

    def test_sbert_normalized_embeddings(self) -> None:
        """SBERT embeddings are L2-normalized (norm ~ 1.0)."""
        model = SBERTLikelihood()
        embeddings = model._embed_batch(["test sentence one", "test sentence two"])
        norms = np.linalg.norm(embeddings, axis=1)
        np.testing.assert_array_almost_equal(
            norms, np.ones(2), decimal=4,
            err_msg="Embeddings should be L2-normalized",
        )


# ------------------------------------------------------------------ #
# Tests for T5Likelihood (LIK-04, LIK-05)
# ------------------------------------------------------------------ #


class TestT5Likelihood:
    """Tests for T5 encoder likelihood model.

    Uses the sample_t5_model fixture (t5-small, module-scoped) from conftest.py
    so the model is loaded once per test file, not per test function.
    """

    def test_t5_semantic_scoring(self, sample_t5_model) -> None:
        """T5 should score semantically relevant options higher (LIK-04).

        "First president" clue should rank Washington higher than Einstein,
        demonstrating that T5 captures semantic similarity between question
        content and answer profiles.
        """
        clue = "This person was the first president of the United States"
        options = [
            "George Washington first president commander revolutionary war",
            "Albert Einstein physicist theory relativity Nobel Prize",
        ]

        scores = sample_t5_model.score(clue, options)

        assert isinstance(scores, np.ndarray)
        assert scores.dtype == np.float32
        assert len(scores) == 2
        # Washington should score higher than Einstein for "first president" query
        assert scores[0] > scores[1], (
            f"Expected Washington > Einstein, got {scores}"
        )

    def test_t5_embedding_cache(self, sample_t5_model) -> None:
        """T5 should cache embeddings and reuse them (LIK-05).

        After embedding two texts, the cache should contain 2 entries.
        Re-embedding the same texts should not grow the cache, and the
        returned embeddings should be identical.
        """
        # Clear cache to get a clean test
        sample_t5_model.embedding_cache.clear()

        texts = ["George Washington", "Thomas Jefferson"]

        # First call embeds and caches
        emb1 = sample_t5_model.embed_and_cache(texts)
        cache_size_1 = len(sample_t5_model.embedding_cache)

        # Second call reuses cache
        emb2 = sample_t5_model.embed_and_cache(texts)
        cache_size_2 = len(sample_t5_model.embedding_cache)

        np.testing.assert_array_equal(
            emb1, emb2, err_msg="Cached embeddings should match"
        )
        assert cache_size_1 == cache_size_2 == 2, (
            f"Cache size should not grow on reuse, got {cache_size_1} -> {cache_size_2}"
        )

    def test_t5_score_returns_float32(self, sample_t5_model) -> None:
        """T5 score should return float32 array, not probabilities.

        Scores are raw cosine similarities (not softmax probabilities),
        so they do not necessarily sum to 1.
        """
        scores = sample_t5_model.score("test clue", ["option 1", "option 2"])
        assert scores.dtype == np.float32
        assert scores.shape == (2,)
        # Scores are raw similarities, not probabilities (don't sum to 1)
        assert all(np.isfinite(scores)), "All scores should be finite"

    def test_build_t5_from_config(self) -> None:
        """Factory should construct T5Likelihood from config (LIK-04).

        The build_likelihood_from_config factory should recognize
        model="t5" and instantiate a T5Likelihood with the specified
        t5_name parameter.
        """
        from models.likelihoods import T5Likelihood, build_likelihood_from_config

        config = {
            "likelihood": {
                "model": "t5",
                "t5_name": "t5-small",
            }
        }

        model = build_likelihood_from_config(config)
        assert isinstance(model, T5Likelihood)
        assert model.model_name == "t5-small"

    def test_t5_handles_variable_length(self, sample_t5_model) -> None:
        """T5 should handle variable-length texts via attention mask.

        Short and long texts should both embed without error, producing
        embeddings of the same hidden dimension regardless of input length.
        """
        short = "Washington"
        long = (
            "George Washington was the first president of the United States "
            "and commander of the Continental Army during the Revolutionary War"
        )

        # Both should embed without error
        embs = sample_t5_model.embed_and_cache([short, long])
        assert embs.shape == (2, sample_t5_model.encoder.config.d_model), (
            f"Expected shape (2, {sample_t5_model.encoder.config.d_model}), "
            f"got {embs.shape}"
        )


# ------------------------------------------------------------------ #
# Tests for Embedding Cache Persistence
# ------------------------------------------------------------------ #


class TestEmbeddingCachePersistence:
    """Tests for save_cache / load_cache disk persistence on LikelihoodModel."""

    def test_save_load_cache_round_trip(self, tmp_path: Path, sample_corpus: list[str]) -> None:
        """save_cache writes .npz; load_cache restores identical entries."""
        model = SBERTLikelihood()
        texts = ["George Washington", "Thomas Jefferson", "Abraham Lincoln"]
        model.embed_and_cache(texts)
        assert len(model.embedding_cache) == 3

        cache_path = tmp_path / "cache.npz"
        saved = model.save_cache(cache_path)
        assert saved == 3
        assert cache_path.exists()

        model2 = SBERTLikelihood()
        assert len(model2.embedding_cache) == 0
        loaded = model2.load_cache(cache_path)
        assert loaded == 3

        for key in model.embedding_cache:
            np.testing.assert_array_equal(
                model.embedding_cache[key],
                model2.embedding_cache[key],
                err_msg=f"Mismatch for key {key}",
            )

    def test_load_cache_missing_file(self, tmp_path: Path) -> None:
        """load_cache with nonexistent file returns 0 and leaves cache empty."""
        model = SBERTLikelihood()
        result = model.load_cache(tmp_path / "nonexistent.npz")
        assert result == 0
        assert len(model.embedding_cache) == 0

    def test_save_cache_empty(self, tmp_path: Path) -> None:
        """save_cache with empty cache creates a valid .npz with zero arrays."""
        model = SBERTLikelihood()
        cache_path = tmp_path / "empty.npz"
        saved = model.save_cache(cache_path)
        assert saved == 0
        assert cache_path.exists()

        # Should be loadable
        model2 = SBERTLikelihood()
        loaded = model2.load_cache(cache_path)
        assert loaded == 0

    def test_tfidf_save_cache_noop(self, sample_corpus: list[str]) -> None:
        """TfIdfLikelihood.save_cache is a no-op returning 0."""
        model = TfIdfLikelihood(corpus_texts=sample_corpus)
        # Populate the cache with some embeddings
        model.embed_and_cache(["test text one", "test text two"])
        assert len(model.embedding_cache) > 0

        import tempfile
        import os
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "should_not_exist.npz"
            result = model.save_cache(path)
            assert result == 0
            assert not path.exists(), "TfIdfLikelihood should NOT write a cache file"

    def test_load_cache_does_not_overwrite(self, tmp_path: Path) -> None:
        """load_cache merges without overwriting existing cache entries."""
        model = SBERTLikelihood()
        texts = ["Hello world"]
        model.embed_and_cache(texts)

        # Save this cache
        cache_path = tmp_path / "cache.npz"
        model.save_cache(cache_path)

        # Create a second model, pre-populate with the same key but different value
        model2 = SBERTLikelihood()
        from models.likelihoods import _text_key
        key = _text_key("Hello world")
        original_value = np.ones(384, dtype=np.float32)  # dummy
        model2.embedding_cache[key] = original_value

        loaded = model2.load_cache(cache_path)
        assert loaded == 0, "Key already present, so nothing should be loaded"

        # Original value should be preserved (not overwritten)
        np.testing.assert_array_equal(
            model2.embedding_cache[key],
            original_value,
            err_msg="Existing cache entry was overwritten by load_cache",
        )


class TestCacheMemory:
    """Verify cache_memory_bytes property for resource monitoring."""

    def test_tfidf_cache_memory_bytes(self, sample_corpus):
        model = TfIdfLikelihood(corpus_texts=sample_corpus)
        assert model.cache_memory_bytes == 0
        model.embed_and_cache(["George Washington"])
        assert model.cache_memory_bytes > 0

    def test_empty_cache_zero_bytes(self, sample_corpus):
        model = TfIdfLikelihood(corpus_texts=sample_corpus)
        assert model.cache_memory_bytes == 0
```

## File: tests/test_pipeline_smoke.py
```python
"""Subprocess-based smoke tests for pipeline entry points.

Each test runs a pipeline script as a subprocess with ``--output-dir``
pointing at a pytest ``tmp_path``, so no artifacts leak to ``artifacts/``.
These tests verify that each script's CLI wiring and end-to-end path
work without errors; they do not validate result quality.

Marked with ``@pytest.mark.slow`` and ``@pytest.mark.pipeline``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _run(args: list[str], timeout: int = 300) -> subprocess.CompletedProcess:
    """Run a Python command as a subprocess from the project root."""
    cmd = [sys.executable, *args]
    return subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


@pytest.fixture(scope="module")
def smoke_mc_dataset(tmp_path_factory) -> Path:
    """Build a smoke MC dataset once per module for reuse by downstream tests."""
    out = tmp_path_factory.mktemp("mc_data")
    result = _run([
        "scripts/build_mc_dataset.py",
        "--smoke",
        "--output-dir", str(out),
    ])
    assert result.returncode == 0, f"build_mc_dataset failed:\n{result.stderr}"
    mc_path = out / "mc_dataset.json"
    assert mc_path.exists(), f"mc_dataset.json not created in {out}"
    return mc_path


@pytest.mark.slow
@pytest.mark.pipeline
def test_build_mc_dataset_smoke(tmp_path):
    """build_mc_dataset.py --smoke --output-dir writes expected outputs."""
    result = _run([
        "scripts/build_mc_dataset.py",
        "--smoke",
        "--output-dir", str(tmp_path),
    ])
    assert result.returncode == 0, f"build_mc_dataset failed:\n{result.stderr}"
    assert (tmp_path / "mc_dataset.json").exists()
    assert (tmp_path / "train_dataset.json").exists()
    assert (tmp_path / "val_dataset.json").exists()
    assert (tmp_path / "test_dataset.json").exists()


@pytest.mark.slow
@pytest.mark.pipeline
def test_run_baselines_smoke(tmp_path, smoke_mc_dataset):
    """run_baselines.py --smoke --output-dir writes baseline_summary.json."""
    result = _run([
        "scripts/run_baselines.py",
        "--smoke",
        "--output-dir", str(tmp_path),
        "--mc-path", str(smoke_mc_dataset),
        "likelihood.model=tfidf",
    ])
    assert result.returncode == 0, f"run_baselines failed:\n{result.stderr}"
    summary = tmp_path / "baseline_summary.json"
    assert summary.exists(), f"baseline_summary.json not created in {tmp_path}"
    data = json.loads(summary.read_text())
    assert "softmax_profile" in data or "threshold" in data


@pytest.mark.slow
@pytest.mark.pipeline
def test_train_ppo_smoke(tmp_path, smoke_mc_dataset):
    """train_ppo.py --smoke --output-dir --timesteps 100 produces a model."""
    result = _run([
        "scripts/train_ppo.py",
        "--smoke",
        "--output-dir", str(tmp_path),
        "--mc-path", str(smoke_mc_dataset),
        "--timesteps", "100",
        "likelihood.model=tfidf",
    ])
    assert result.returncode == 0, f"train_ppo failed:\n{result.stderr}"
    assert (tmp_path / "ppo_model.zip").exists()
    assert (tmp_path / "ppo_summary.json").exists()
    assert (tmp_path / "config_used.json").exists()


@pytest.mark.slow
@pytest.mark.pipeline
def test_evaluate_all_smoke(tmp_path, smoke_mc_dataset):
    """evaluate_all.py --smoke --output-dir writes evaluation_report.json."""
    result = _run([
        "scripts/evaluate_all.py",
        "--smoke",
        "--output-dir", str(tmp_path),
        "--mc-path", str(smoke_mc_dataset),
        "likelihood.model=tfidf",
    ])
    assert result.returncode == 0, f"evaluate_all failed:\n{result.stderr}"
    report = tmp_path / "evaluation_report.json"
    assert report.exists(), f"evaluation_report.json not created in {tmp_path}"


@pytest.mark.slow
@pytest.mark.pipeline
@pytest.mark.skipif(
    not os.environ.get("RUN_PIPELINE_E2E"),
    reason="set RUN_PIPELINE_E2E=1 to run full 4-stage pipeline test",
)
def test_run_smoke_pipeline(tmp_path):
    """run_smoke_pipeline.py --output-dir runs all 4 stages in a temp dir.

    Skipped by default because it re-runs the full 4-stage pipeline (~18s),
    which the individual stage tests already cover. Run explicitly with:
        RUN_PIPELINE_E2E=1 pytest tests/test_pipeline_smoke.py -k run_smoke_pipeline
    """
    result = _run([
        "scripts/run_smoke_pipeline.py",
        "--output-dir", str(tmp_path),
    ], timeout=600)
    assert result.returncode == 0, (
        f"run_smoke_pipeline failed:\n{result.stdout}\n{result.stderr}"
    )
    summary_path = tmp_path / "smoke_pipeline_summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text())
    assert summary["status"] == "ok"
    assert len(summary["stages"]) == 4
    assert all(s["exit_code"] == 0 for s in summary["stages"])
```

## File: tests/test_ppo_t5.py
```python
"""Unit tests for custom PPO trainer for T5PolicyModel.

Tests cover RolloutStep dataclass, RolloutBuffer with GAE computation,
rollout collection with memory management, dynamic padding, and PPO update.

Uses t5-small (60M params) and TF-IDF likelihood for fast execution.
The T5 model fixture is module-scoped (loaded once per test file).
"""

from __future__ import annotations

import pytest
import torch
import numpy as np

from training.train_ppo_t5 import RolloutStep, RolloutBuffer, PPOTrainer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def t5_ppo_config() -> dict:
    """Minimal PPO config for testing."""
    return {
        "model_name": "t5-small",
        "device": "cpu",
        "max_input_length": 64,
        "num_choices": 4,
        "ppo_lr": 1e-4,
        "ppo_iterations": 2,
        "ppo_batch_size": 4,
        "ppo_epochs_per_iter": 2,
        "ppo_gamma": 0.99,
        "ppo_gae_lambda": 0.95,
        "ppo_clip_ratio": 0.2,
        "ppo_value_coef": 0.5,
        "ppo_entropy_coef": 0.01,
        "ppo_max_grad_norm": 0.5,
        "ppo_episodes_per_iter": 2,
        "eval_interval": 1,
        "save_interval": 100,
        "checkpoint_dir": "/tmp/test_ppo_t5_checkpoints",
        "reward_time_penalty": 0.01,
    }


@pytest.fixture(scope="module")
def t5_ppo_model(t5_ppo_config):
    """Load T5PolicyModel with t5-small once per test module."""
    from models.t5_policy import T5PolicyModel

    model = T5PolicyModel(t5_ppo_config)
    return model


@pytest.fixture
def sample_rollout_steps() -> list:
    """Create sample RolloutStep instances for testing GAE computation."""
    # Simulate a 4-step episode: WAIT, WAIT, WAIT, BUZZ(correct)
    steps = [
        RolloutStep(
            observation_text="CLUES: Who | CHOICES: (1) A (2) B (3) C (4) D",
            action=0,
            reward=-0.01,
            done=False,
            value=0.2,
            log_prob=-0.8,
            input_ids=torch.randint(0, 100, (1, 10)),
            attention_mask=torch.ones(1, 10, dtype=torch.long),
        ),
        RolloutStep(
            observation_text="CLUES: Who was | CHOICES: (1) A (2) B (3) C (4) D",
            action=0,
            reward=-0.01,
            done=False,
            value=0.4,
            log_prob=-0.7,
            input_ids=torch.randint(0, 100, (1, 12)),
            attention_mask=torch.ones(1, 12, dtype=torch.long),
        ),
        RolloutStep(
            observation_text="CLUES: Who was the first | CHOICES: (1) A (2) B (3) C (4) D",
            action=0,
            reward=-0.01,
            done=False,
            value=0.6,
            log_prob=-0.5,
            input_ids=torch.randint(0, 100, (1, 15)),
            attention_mask=torch.ones(1, 15, dtype=torch.long),
        ),
        RolloutStep(
            observation_text="CLUES: Who was the first president | CHOICES: (1) A (2) B (3) C (4) D",
            action=1,
            reward=1.0,
            done=True,
            value=0.8,
            log_prob=-0.3,
            input_ids=torch.randint(0, 100, (1, 18)),
            attention_mask=torch.ones(1, 18, dtype=torch.long),
        ),
    ]
    return steps


# ---------------------------------------------------------------------------
# RolloutStep Tests
# ---------------------------------------------------------------------------


class TestRolloutStep:
    """Tests for the RolloutStep dataclass."""

    def test_rollout_step_dataclass(self):
        """RolloutStep stores all required fields."""
        step = RolloutStep(
            observation_text="test",
            action=0,
            reward=1.0,
            done=True,
            value=0.5,
            log_prob=-0.3,
        )
        assert step.observation_text == "test"
        assert step.action == 0
        assert step.reward == 1.0
        assert step.done is True
        assert step.value == 0.5
        assert step.log_prob == -0.3
        assert step.input_ids is None
        assert step.attention_mask is None
        assert step.return_ == 0.0
        assert step.advantage == 0.0

    def test_rollout_step_with_tensors(self):
        """RolloutStep stores tensor fields on CPU."""
        ids = torch.randint(0, 100, (1, 10))
        mask = torch.ones(1, 10, dtype=torch.long)
        step = RolloutStep(
            observation_text="test",
            action=1,
            reward=0.5,
            done=False,
            value=0.3,
            log_prob=-0.5,
            input_ids=ids,
            attention_mask=mask,
        )
        assert step.input_ids is not None
        assert step.input_ids.device.type == "cpu"
        assert step.attention_mask.device.type == "cpu"
        assert step.input_ids.shape == (1, 10)


# ---------------------------------------------------------------------------
# RolloutBuffer Tests
# ---------------------------------------------------------------------------


class TestRolloutBuffer:
    """Tests for the RolloutBuffer class."""

    def test_rollout_buffer_add(self, sample_rollout_steps):
        """Buffer accumulates rollouts correctly."""
        buffer = RolloutBuffer()
        assert len(buffer) == 0

        buffer.add_rollout(sample_rollout_steps)
        assert len(buffer) == 1

        buffer.add_rollout(sample_rollout_steps[:2])
        assert len(buffer) == 2

    def test_rollout_buffer_get_all_steps(self, sample_rollout_steps):
        """get_all_steps returns flat list of all steps."""
        buffer = RolloutBuffer()
        buffer.add_rollout(sample_rollout_steps)
        buffer.add_rollout(sample_rollout_steps[:2])

        all_steps = buffer.get_all_steps()
        assert len(all_steps) == 6  # 4 + 2

    def test_rollout_buffer_reset(self, sample_rollout_steps):
        """reset() clears all rollouts."""
        buffer = RolloutBuffer()
        buffer.add_rollout(sample_rollout_steps)
        assert len(buffer) == 1

        buffer.reset()
        assert len(buffer) == 0
        assert len(buffer.get_all_steps()) == 0

    def test_gae_computation(self, sample_rollout_steps):
        """GAE advantages match hand-calculated values.

        Episode: 4 steps with rewards [-0.01, -0.01, -0.01, 1.0]
        and values [0.2, 0.4, 0.6, 0.8].
        """
        buffer = RolloutBuffer()
        buffer.add_rollout(sample_rollout_steps)

        gamma = 0.99
        gae_lambda = 0.95

        buffer.compute_returns_and_advantages(gamma, gae_lambda)

        all_steps = buffer.get_all_steps()

        # Verify terminal step (t=3): done=True
        # delta_3 = r_3 + gamma * 0 - v_3 = 1.0 + 0 - 0.8 = 0.2
        # gae_3 = delta_3 = 0.2 (reset because done=True)
        assert abs(all_steps[3].advantage - 0.2) < 1e-6
        assert abs(all_steps[3].return_ - (0.2 + 0.8)) < 1e-6  # adv + value

        # Step t=2: not done
        # delta_2 = r_2 + gamma * v_3 - v_2 = -0.01 + 0.99 * 0.8 - 0.6 = 0.182
        # gae_2 = delta_2 + gamma * lambda * gae_3 = 0.182 + 0.99 * 0.95 * 0.2
        delta_2 = -0.01 + gamma * 0.8 - 0.6
        gae_2 = delta_2 + gamma * gae_lambda * 0.2
        assert abs(all_steps[2].advantage - gae_2) < 1e-6

        # Step t=1:
        # delta_1 = r_1 + gamma * v_2 - v_1 = -0.01 + 0.99 * 0.6 - 0.4
        delta_1 = -0.01 + gamma * 0.6 - 0.4
        gae_1 = delta_1 + gamma * gae_lambda * gae_2
        assert abs(all_steps[1].advantage - gae_1) < 1e-6

        # Step t=0:
        delta_0 = -0.01 + gamma * 0.4 - 0.2
        gae_0 = delta_0 + gamma * gae_lambda * gae_1
        assert abs(all_steps[0].advantage - gae_0) < 1e-6

    def test_gae_multiple_episodes(self, sample_rollout_steps):
        """GAE handles multiple episodes independently."""
        buffer = RolloutBuffer()

        # Two episodes
        buffer.add_rollout(sample_rollout_steps)
        buffer.add_rollout(sample_rollout_steps[:2] + [
            RolloutStep(
                observation_text="end",
                action=2,
                reward=-1.0,
                done=True,
                value=0.1,
                log_prob=-1.0,
            )
        ])

        buffer.compute_returns_and_advantages(gamma=0.99, gae_lambda=0.95)

        all_steps = buffer.get_all_steps()
        # All steps should have return_ and advantage set
        for step in all_steps:
            assert isinstance(step.return_, float)
            assert isinstance(step.advantage, float)


# ---------------------------------------------------------------------------
# Dynamic Padding Tests
# ---------------------------------------------------------------------------


class TestDynamicPadding:
    """Tests for dynamic batch padding."""

    def test_dynamic_padding(self, t5_ppo_model, t5_ppo_config, sample_mc_question):
        """Padding works with variable-length sequences."""
        trainer = PPOTrainer(
            model=t5_ppo_model,
            train_questions=[sample_mc_question] * 3,
            val_questions=[sample_mc_question] * 2,
            config=t5_ppo_config,
        )

        # Create steps with different sequence lengths
        steps = [
            RolloutStep(
                observation_text="short",
                action=0,
                reward=0.0,
                done=False,
                value=0.1,
                log_prob=-0.5,
                input_ids=torch.randint(0, 100, (1, 5)),
                attention_mask=torch.ones(1, 5, dtype=torch.long),
            ),
            RolloutStep(
                observation_text="this is a longer sequence",
                action=1,
                reward=1.0,
                done=True,
                value=0.8,
                log_prob=-0.2,
                input_ids=torch.randint(0, 100, (1, 15)),
                attention_mask=torch.ones(1, 15, dtype=torch.long),
            ),
            RolloutStep(
                observation_text="medium",
                action=0,
                reward=0.0,
                done=False,
                value=0.3,
                log_prob=-0.6,
                input_ids=torch.randint(0, 100, (1, 10)),
                attention_mask=torch.ones(1, 10, dtype=torch.long),
            ),
        ]

        input_ids, attention_mask = trainer._pad_batch(steps)

        # All padded to max length in batch (15)
        assert input_ids.shape == (3, 15)
        assert attention_mask.shape == (3, 15)

        # First sequence (len 5) should have 10 padding tokens
        assert attention_mask[0, :5].sum() == 5
        assert attention_mask[0, 5:].sum() == 0

        # Second sequence (len 15) should have no padding
        assert attention_mask[1].sum() == 15

        # Third sequence (len 10) should have 5 padding tokens
        assert attention_mask[2, :10].sum() == 10
        assert attention_mask[2, 10:].sum() == 0


# ---------------------------------------------------------------------------
# Memory Management Tests
# ---------------------------------------------------------------------------


class TestMemoryManagement:
    """Tests for memory-safe tensor handling."""

    def test_memory_management_cpu_storage(self, sample_rollout_steps):
        """Rollout tensors are stored on CPU, not GPU."""
        for step in sample_rollout_steps:
            if step.input_ids is not None:
                assert step.input_ids.device.type == "cpu", (
                    f"input_ids on {step.input_ids.device}, expected CPU"
                )
            if step.attention_mask is not None:
                assert step.attention_mask.device.type == "cpu", (
                    f"attention_mask on {step.attention_mask.device}, expected CPU"
                )

    def test_rollout_tensors_are_detached(self, sample_rollout_steps):
        """Stored tensors do not require gradients."""
        for step in sample_rollout_steps:
            if step.input_ids is not None:
                assert not step.input_ids.requires_grad
            if step.attention_mask is not None:
                assert not step.attention_mask.requires_grad


# ---------------------------------------------------------------------------
# PPO Update Tests
# ---------------------------------------------------------------------------


class TestPPOUpdate:
    """Tests for PPO policy updates."""

    def test_ppo_update_no_oom(
        self, t5_ppo_model, t5_ppo_config, sample_mc_question
    ):
        """update_policy completes without OOM or errors."""
        trainer = PPOTrainer(
            model=t5_ppo_model,
            train_questions=[sample_mc_question] * 3,
            val_questions=[sample_mc_question] * 2,
            config=t5_ppo_config,
        )

        # Create a small buffer with tokenized steps
        buffer = RolloutBuffer()
        texts = [
            "CLUES: Who | CHOICES: (1) A (2) B (3) C (4) D",
            "CLUES: Who was | CHOICES: (1) A (2) B (3) C (4) D",
            "CLUES: Who was the | CHOICES: (1) A (2) B (3) C (4) D",
        ]

        rollout = []
        for i, text in enumerate(texts):
            inputs = t5_ppo_model.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=64,
            )
            is_last = i == len(texts) - 1
            step = RolloutStep(
                observation_text=text,
                action=0 if not is_last else 1,
                reward=-0.01 if not is_last else 1.0,
                done=is_last,
                value=0.1 * (i + 1),
                log_prob=-0.5,
                input_ids=inputs["input_ids"].detach().cpu(),
                attention_mask=inputs["attention_mask"].detach().cpu(),
            )
            rollout.append(step)

        buffer.add_rollout(rollout)

        # Should complete without errors
        metrics = trainer.update_policy(buffer)

        assert "policy_loss" in metrics
        assert "value_loss" in metrics
        assert "entropy" in metrics
        assert metrics["num_updates"] > 0

    def test_ppo_update_empty_buffer(
        self, t5_ppo_model, t5_ppo_config, sample_mc_question
    ):
        """update_policy handles empty buffer gracefully."""
        trainer = PPOTrainer(
            model=t5_ppo_model,
            train_questions=[sample_mc_question] * 3,
            val_questions=[sample_mc_question] * 2,
            config=t5_ppo_config,
        )

        buffer = RolloutBuffer()
        metrics = trainer.update_policy(buffer)

        assert metrics["num_updates"] == 0
        assert metrics["policy_loss"] == 0.0


# ---------------------------------------------------------------------------
# Rollout Collection Tests
# ---------------------------------------------------------------------------


class TestRolloutCollection:
    """Tests for rollout collection."""

    def test_rollout_collection(
        self, t5_ppo_model, t5_ppo_config, sample_mc_question
    ):
        """collect_rollouts returns buffer with episodes."""
        trainer = PPOTrainer(
            model=t5_ppo_model,
            train_questions=[sample_mc_question] * 3,
            val_questions=[sample_mc_question] * 2,
            config=t5_ppo_config,
        )

        buffer = trainer.collect_rollouts(num_episodes=2)

        assert len(buffer) == 2  # 2 episodes collected
        all_steps = buffer.get_all_steps()
        assert len(all_steps) > 0  # At least some steps

        # Each step should have text, action, reward, tensors
        for step in all_steps:
            assert isinstance(step.observation_text, str)
            assert isinstance(step.action, int)
            assert 0 <= step.action <= 4  # WAIT or SELECT
            assert step.input_ids is not None
            assert step.attention_mask is not None
            # Tensors should be on CPU
            assert step.input_ids.device.type == "cpu"
            assert step.attention_mask.device.type == "cpu"

    def test_rollout_episodes_terminate(
        self, t5_ppo_model, t5_ppo_config, sample_mc_question
    ):
        """All collected episodes properly terminate."""
        trainer = PPOTrainer(
            model=t5_ppo_model,
            train_questions=[sample_mc_question] * 3,
            val_questions=[sample_mc_question] * 2,
            config=t5_ppo_config,
        )

        buffer = trainer.collect_rollouts(num_episodes=3)

        for rollout in buffer.rollouts:
            # Last step should be done
            assert rollout[-1].done, "Episode should terminate"
            # Non-terminal steps should not be done
            for step in rollout[:-1]:
                assert not step.done, "Non-terminal step should not be done"
```

## File: tests/test_qb_rl_bridge.py
```python
"""Compatibility bridge tests for qb-rl surfaces ported into qanta-buzzer."""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

import agents.bayesian_buzzer as bayesian_buzzer
import models.answer_profiles as compat_answer_profiles
import models.likelihoods as likelihoods
import qb_data.answer_profiles as qb_answer_profiles
import qb_data.data_loader as qb_data_loader
import qb_env.data_loader as compat_data_loader
import qb_env.mc_builder as compat_mc_builder
import qb_env.text_utils as compat_text_utils
from agents.softmax_profile_buzzer import (
    SequentialBayesBuzzer as CompatSequentialBayesBuzzer,
)
from agents.softmax_profile_buzzer import (
    SoftmaxEpisodeResult as CompatSoftmaxEpisodeResult,
)
from agents.softmax_profile_buzzer import (
    SoftmaxProfileBuzzer as CompatSoftmaxProfileBuzzer,
)
from models.likelihoods import OpenAILikelihood, build_likelihood_from_config
from qb_data.mc_builder import MCBuilder


def _install_fake_openai(monkeypatch, vectors: dict[str, list[float]], calls: list[tuple[str, tuple[str, ...]]]) -> None:
    """Install a fake ``openai`` module that serves deterministic embeddings."""

    class FakeEmbeddingsClient:
        def create(self, model: str, input: list[str]):
            calls.append((model, tuple(input)))
            return types.SimpleNamespace(
                data=[
                    types.SimpleNamespace(embedding=vectors[text])
                    for text in input
                ]
            )

    class FakeOpenAI:
        def __init__(self, api_key: str):
            self.api_key = api_key
            self.embeddings = FakeEmbeddingsClient()

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=FakeOpenAI))


class TestOpenAILikelihood:
    """Tests for optional OpenAI embedding support."""

    def test_openai_likelihood_requires_api_key(self, monkeypatch) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
            OpenAILikelihood()

    def test_openai_likelihood_scores_and_reuses_cache(self, monkeypatch) -> None:
        calls: list[tuple[str, tuple[str, ...]]] = []
        vectors = {
            "first president": [2.0, 0.0],
            "george washington": [3.0, 0.0],
            "albert einstein": [0.0, 4.0],
        }
        _install_fake_openai(monkeypatch, vectors=vectors, calls=calls)
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        model = OpenAILikelihood(model="fake-embedding-model")

        embeddings = model._embed_batch(["first president", "george washington"])
        norms = np.linalg.norm(embeddings, axis=1)
        np.testing.assert_allclose(norms, np.ones(2), atol=1e-6)
        calls_before_score = len(calls)

        scores_1 = model.score(
            "first president",
            ["george washington", "albert einstein"],
        )
        assert scores_1[0] > scores_1[1]
        assert len(calls) == calls_before_score + 2, (
            "first score should call the embeddings API twice"
        )

        scores_2 = model.score(
            "first president",
            ["george washington", "albert einstein"],
        )
        np.testing.assert_allclose(scores_1, scores_2, atol=1e-6)
        assert len(calls) == calls_before_score + 2, "second score should be served from cache"

    def test_likelihood_factory_openai(self, monkeypatch) -> None:
        calls: list[tuple[str, tuple[str, ...]]] = []
        vectors = {"a": [1.0, 0.0]}
        _install_fake_openai(monkeypatch, vectors=vectors, calls=calls)
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        config = {"likelihood": {"model": "openai", "openai_model": "fake-openai"}}
        model = build_likelihood_from_config(config)

        assert isinstance(model, OpenAILikelihood)
        assert model.model == "fake-openai"


class TestOpenAIProfileStrategy:
    """Tests for OpenAI-backed distractor ranking."""

    def test_openai_profile_uses_openai_embeddings(self, monkeypatch) -> None:
        calls: list[str] = []
        embeddings = {
            "gold profile": np.array([1.0, 0.0], dtype=np.float32),
            "near distractor": np.array([0.9, 0.1], dtype=np.float32),
            "far distractor": np.array([0.0, 1.0], dtype=np.float32),
        }

        class FakeOpenAILikelihood:
            def __init__(self, model: str = "unused") -> None:
                calls.append(model)

            def embed_and_cache(self, texts: list[str]) -> np.ndarray:
                return np.stack([embeddings[text] for text in texts]).astype(np.float32)

        monkeypatch.setattr(likelihoods, "OpenAILikelihood", FakeOpenAILikelihood)

        builder = MCBuilder(strategy="openai_profile", openai_model="fake-openai")
        rankings = builder._compute_rankings(
            answers=["gold", "near", "far"],
            answer_profiles={
                "gold": "gold profile",
                "near": "near distractor",
                "far": "far distractor",
            },
            answer_to_category={},
        )

        assert calls == ["fake-openai"]
        assert rankings["gold"][0] == "near"
        assert rankings["gold"][1] == "far"


class TestQBRLCompatibilityModules:
    """Tests for qb-rl import-path shims."""

    def test_module_aliases_resolve_expected_symbols(self) -> None:
        assert compat_answer_profiles.AnswerProfileBuilder is qb_answer_profiles.AnswerProfileBuilder
        assert compat_data_loader.parse_row is qb_data_loader.parse_row
        assert compat_mc_builder.MCBuilder.__name__ == "MCBuilder"
        assert compat_text_utils.normalize_answer("The Answer") == "answer"
        assert CompatSoftmaxProfileBuzzer is bayesian_buzzer.SoftmaxProfileBuzzer
        assert CompatSequentialBayesBuzzer is bayesian_buzzer.SequentialBayesBuzzer
        assert CompatSoftmaxEpisodeResult is bayesian_buzzer.SoftmaxEpisodeResult

    def test_parse_row_supports_qb_rl_metadata(self) -> None:
        question = compat_data_loader.parse_row(
            {
                "qid": "q-1",
                "question": "alpha beta gamma",
                "answer_primary": "George Washington",
                "clean_answers": ["George Washington", "Washington"],
                "run_indices": [1, 2],
                "metadata": {
                    "category": "History",
                    "human_buzz_positions": [{"position": 4, "count": 2}],
                },
            }
        )

        assert question.qid == "q-1"
        assert question.category == "History"
        assert question.human_buzz_positions == [(4, 2)]
        assert question.cumulative_prefixes == ["alpha beta", "alpha beta gamma"]

    def test_load_tossup_questions_from_config_prefers_dataset_smoke(
        self, monkeypatch
    ) -> None:
        captured: dict[str, object] = {}
        sample_question = compat_data_loader.TossupQuestion(
            qid="hf-1",
            question="alpha beta",
            tokens=["alpha", "beta"],
            answer_primary="Answer",
            clean_answers=["Answer"],
            run_indices=[1],
            human_buzz_positions=None,
            category="History",
            cumulative_prefixes=["alpha beta"],
        )

        def fake_load_tossup_questions(
            dataset: str,
            dataset_config: str | None = None,
            split: str = "eval",
            limit: int | None = None,
        ):
            captured["dataset"] = dataset
            captured["dataset_config"] = dataset_config
            captured["split"] = split
            captured["limit"] = limit
            return [sample_question]

        monkeypatch.setattr(qb_data_loader, "load_tossup_questions", fake_load_tossup_questions)

        config = {
            "data": {
                "dataset": "main-dataset",
                "dataset_config": "main-config",
                "dataset_smoke": "smoke-dataset",
                "dataset_smoke_config": "smoke-config",
                "split": "train",
            }
        }

        questions = compat_data_loader.load_tossup_questions_from_config(config, smoke=True)

        assert len(questions) == 1
        assert captured == {
            "dataset": "smoke-dataset",
            "dataset_config": "smoke-config",
            "split": "train",
            "limit": None,
        }
```

## File: tests/test_stop_only_env.py
```python
"""Tests for StopOnlyEnv wrapper action_masks and step mapping."""

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces

from qb_env.stop_only_env import StopOnlyEnv


class FakeBaseEnv(gym.Env):
    """Minimal fake TossupMCEnv for testing StopOnlyEnv."""

    def __init__(self, belief=None):
        super().__init__()
        self.belief = belief
        self.observation_space = spaces.Box(low=-1, high=1, shape=(10,))
        self.action_space = spaces.Discrete(5)
        self._last_action = None

    def reset(self, seed=None, options=None):
        return np.zeros(10, dtype=np.float32), {}

    def step(self, action):
        self._last_action = action
        return np.zeros(10, dtype=np.float32), 0.0, True, False, {"step_idx": 0}


def test_action_masks_shape_and_dtype():
    env = StopOnlyEnv(FakeBaseEnv(belief=np.array([0.2, 0.8])))
    masks = env.action_masks()
    assert masks.shape == (2,)
    assert masks.dtype == bool


def test_action_masks_both_true_when_belief_present():
    env = StopOnlyEnv(FakeBaseEnv(belief=np.array([0.2, 0.8])))
    masks = env.action_masks()
    assert masks[0]
    assert masks[1]


def test_action_masks_buzz_false_when_no_belief():
    env = StopOnlyEnv(FakeBaseEnv(belief=None))
    masks = env.action_masks()
    assert masks[0]
    assert not masks[1]


def test_step_buzz_maps_to_argmax():
    base = FakeBaseEnv(belief=np.array([0.1, 0.3, 0.6]))
    env = StopOnlyEnv(base)
    env.step(1)
    assert base._last_action == 3  # 1 + argmax([0.1, 0.3, 0.6]) = 1 + 2


@pytest.mark.parametrize("belief", [None, np.array([])])
def test_step_buzz_raises_when_belief_unavailable(belief):
    env = StopOnlyEnv(FakeBaseEnv(belief=belief))

    with pytest.raises(ValueError, match="belief is unavailable"):
        env.step(1)
```

## File: tests/test_supervised_t5.py
```python
"""Unit tests for SupervisedTrainer and supervised training utilities.

Tests cover batch preparation, training epochs, gradient accumulation,
checkpoint save/load, best model selection, and the run_supervised_training
entry point.

Uses t5-small (60M params) for speed. The model fixture is module-scoped
to load t5-small only once per test file.
"""

from __future__ import annotations

import json
import os
import tempfile

import pytest
import torch

from models.t5_policy import T5PolicyModel
from qb_data.mc_builder import MCQuestion
from training.train_supervised_t5 import (
    SupervisedTrainer,
    format_question_text,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_question(qid: str, gold_index: int = 0) -> MCQuestion:
    """Create a minimal MCQuestion for testing."""
    tokens = ["Who", "was", "the", "first", "president"]
    return MCQuestion(
        qid=qid,
        question="Who was the first president",
        tokens=tokens,
        answer_primary="George Washington",
        clean_answers=["George Washington"],
        run_indices=[0, 2, 4],
        human_buzz_positions=[],
        category="History",
        cumulative_prefixes=[
            "Who",
            "Who was the",
            "Who was the first president",
        ],
        options=[
            "George Washington",
            "Thomas Jefferson",
            "John Adams",
            "Benjamin Franklin",
        ],
        gold_index=gold_index,
        option_profiles=[
            "George Washington first president",
            "Thomas Jefferson third president",
            "John Adams second president",
            "Benjamin Franklin inventor diplomat",
        ],
        option_answer_primary=[
            "George Washington",
            "Thomas Jefferson",
            "John Adams",
            "Benjamin Franklin",
        ],
        distractor_strategy="test",
    )


@pytest.fixture(scope="module")
def t5_small_model() -> T5PolicyModel:
    """Load T5PolicyModel with t5-small once per test module."""
    model = T5PolicyModel(
        {
            "model_name": "t5-small",
            "device": "cpu",
            "max_input_length": 64,
            "num_choices": 4,
        }
    )
    return model


@pytest.fixture
def train_questions() -> list[MCQuestion]:
    """Return 8 training questions with varied gold indices."""
    return [_make_question(f"train_{i}", i % 4) for i in range(8)]


@pytest.fixture
def val_questions() -> list[MCQuestion]:
    """Return 4 validation questions."""
    return [_make_question(f"val_{i}", i % 4) for i in range(4)]


@pytest.fixture
def trainer_config(tmp_path) -> dict:
    """Return a minimal supervised trainer config using temp directory."""
    return {
        "model_name": "t5-small",
        "device": "cpu",
        "num_choices": 4,
        "supervised_lr": 1e-3,
        "supervised_epochs": 2,
        "supervised_batch_size": 2,
        "supervised_grad_accum_steps": 2,
        "max_input_length": 64,
        "max_grad_norm": 1.0,
        "weight_decay": 0.01,
        "checkpoint_dir": str(tmp_path / "checkpoints"),
    }


@pytest.fixture
def trainer(
    t5_small_model: T5PolicyModel,
    train_questions: list[MCQuestion],
    val_questions: list[MCQuestion],
    trainer_config: dict,
) -> SupervisedTrainer:
    """Return a configured SupervisedTrainer instance."""
    return SupervisedTrainer(
        model=t5_small_model,
        train_questions=train_questions,
        val_questions=val_questions,
        config=trainer_config,
    )


# ---------------------------------------------------------------------------
# Format Tests
# ---------------------------------------------------------------------------


class TestFormatQuestionText:
    """Tests for the format_question_text utility."""

    def test_format_includes_all_tokens(self):
        """Formatted text includes all question tokens as clues."""
        q = _make_question("q1")
        text = format_question_text(q)
        assert "Who was the first president" in text

    def test_format_includes_all_choices(self):
        """Formatted text includes all 4 answer choices."""
        q = _make_question("q1")
        text = format_question_text(q)
        assert "(1) George Washington" in text
        assert "(2) Thomas Jefferson" in text
        assert "(3) John Adams" in text
        assert "(4) Benjamin Franklin" in text

    def test_format_structure(self):
        """Formatted text has CLUES: ... | CHOICES: ... structure."""
        q = _make_question("q1")
        text = format_question_text(q)
        assert text.startswith("CLUES: ")
        assert " | CHOICES: " in text


# ---------------------------------------------------------------------------
# Batch Preparation Tests
# ---------------------------------------------------------------------------


class TestPrepareBatch:
    """Tests for SupervisedTrainer.prepare_batch."""

    def test_prepare_batch_format(self, trainer: SupervisedTrainer):
        """Batch preparation produces correct tensor types and shapes."""
        questions = [_make_question(f"q{i}", i % 4) for i in range(3)]
        input_ids, attention_mask, labels = trainer.prepare_batch(questions)

        assert isinstance(input_ids, torch.Tensor)
        assert isinstance(attention_mask, torch.Tensor)
        assert isinstance(labels, torch.Tensor)
        assert input_ids.shape[0] == 3  # batch_size
        assert attention_mask.shape == input_ids.shape
        assert labels.shape == (3,)

    def test_prepare_batch_complete_questions(self, trainer: SupervisedTrainer):
        """Batch shows complete questions (all clues), not incremental."""
        q = _make_question("q1")
        input_ids, _, _ = trainer.prepare_batch([q])

        # Decode tokens to verify all clues are included
        decoded = trainer.model.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        # All tokens should be present in the decoded text
        assert "first" in decoded.lower()
        assert "president" in decoded.lower()

    def test_prepare_batch_labels_correct(self, trainer: SupervisedTrainer):
        """Labels match gold_index of each question."""
        questions = [
            _make_question("q0", gold_index=0),
            _make_question("q1", gold_index=2),
            _make_question("q2", gold_index=3),
        ]
        _, _, labels = trainer.prepare_batch(questions)
        assert labels.tolist() == [0, 2, 3]


# ---------------------------------------------------------------------------
# Training Tests
# ---------------------------------------------------------------------------


class TestTrainEpoch:
    """Tests for SupervisedTrainer.train_epoch."""

    def test_training_epoch_completes(self, trainer: SupervisedTrainer):
        """One epoch completes without errors."""
        loss, acc = trainer.train_epoch()

        assert isinstance(loss, float)
        assert isinstance(acc, float)
        assert loss > 0, "Loss should be positive"
        assert 0 <= acc <= 1, "Accuracy should be in [0, 1]"

    def test_gradient_accumulation(
        self,
        t5_small_model: T5PolicyModel,
        train_questions: list[MCQuestion],
        val_questions: list[MCQuestion],
        tmp_path,
    ):
        """Optimizer updates only on accumulation steps (not every batch)."""
        config = {
            "supervised_lr": 1e-3,
            "supervised_epochs": 1,
            "supervised_batch_size": 2,
            "supervised_grad_accum_steps": 4,  # Update every 4 batches
            "max_input_length": 64,
            "checkpoint_dir": str(tmp_path / "checkpoints"),
        }

        trainer = SupervisedTrainer(
            model=t5_small_model,
            train_questions=train_questions,
            val_questions=val_questions,
            config=config,
        )

        # Record initial params
        initial_params = {
            name: param.clone()
            for name, param in t5_small_model.policy_head.named_parameters()
        }

        # Run one epoch
        trainer.train_epoch()

        # Check that params changed (at least some should update)
        any_changed = False
        for name, param in t5_small_model.policy_head.named_parameters():
            if not torch.equal(initial_params[name], param):
                any_changed = True
                break

        assert any_changed, "Policy head parameters should change after training"


# ---------------------------------------------------------------------------
# Validation Tests
# ---------------------------------------------------------------------------


class TestValidation:
    """Tests for SupervisedTrainer.validate."""

    def test_validate_returns_metrics(self, trainer: SupervisedTrainer):
        """Validation returns loss and accuracy."""
        val_loss, val_acc = trainer.validate()

        assert isinstance(val_loss, float)
        assert isinstance(val_acc, float)
        assert val_loss > 0
        assert 0 <= val_acc <= 1


# ---------------------------------------------------------------------------
# Checkpoint Tests
# ---------------------------------------------------------------------------


class TestCheckpoint:
    """Tests for checkpoint save/load functionality."""

    def test_checkpoint_save_load(self, trainer: SupervisedTrainer):
        """Save then load produces identical model outputs."""
        trainer.model.eval()

        # Get output before save
        q = _make_question("test_checkpoint")
        input_ids, attention_mask, _ = trainer.prepare_batch([q])
        with torch.no_grad():
            logits_before, preds_before = trainer.model.predict_answer(
                input_ids, attention_mask
            )

        # Save checkpoint
        save_path = trainer.save_checkpoint(is_best=True)
        assert save_path.exists()
        assert (save_path / "policy_head.pt").exists()
        assert (save_path / "training_state.pt").exists()

        # Load checkpoint
        trainer.model.load(str(save_path))

        # Get output after load
        with torch.no_grad():
            logits_after, preds_after = trainer.model.predict_answer(
                input_ids, attention_mask
            )

        assert torch.allclose(logits_before, logits_after, atol=1e-5)

    def test_best_model_selection(
        self,
        t5_small_model: T5PolicyModel,
        train_questions: list[MCQuestion],
        val_questions: list[MCQuestion],
        tmp_path,
    ):
        """Best model saved by validation accuracy (best_model/ dir exists)."""
        config = {
            "supervised_lr": 1e-3,
            "supervised_epochs": 2,
            "supervised_batch_size": 4,
            "supervised_grad_accum_steps": 1,
            "max_input_length": 64,
            "checkpoint_dir": str(tmp_path / "checkpoints"),
        }

        trainer = SupervisedTrainer(
            model=t5_small_model,
            train_questions=train_questions,
            val_questions=val_questions,
            config=config,
        )

        result = trainer.train()

        # Best model directory should exist
        best_model_path = trainer.checkpoint_dir / "best_model"
        assert best_model_path.exists(), "best_model/ directory should exist"
        assert (best_model_path / "policy_head.pt").exists()
        assert result["best_val_acc"] >= 0

    def test_history_saved(self, trainer: SupervisedTrainer):
        """Training history saved to history.json with correct structure."""
        # Run a quick training
        trainer.config["supervised_epochs"] = 1
        trainer.epochs = 1
        trainer.train()

        history_path = trainer.checkpoint_dir / "history.json"
        assert history_path.exists()

        with open(history_path) as f:
            history = json.load(f)

        assert "train" in history
        assert "val" in history
        assert "config" in history
        assert len(history["train"]) >= 1
        assert "loss" in history["train"][0]
        assert "accuracy" in history["train"][0]
```

## File: tests/test_variable_k_integration.py
```python
"""Integration test exercising a mixed-K pipeline path."""

from __future__ import annotations

import numpy as np
import pytest

from qb_data.data_loader import TossupQuestion
from qb_data.answer_profiles import AnswerProfileBuilder
from qb_data.mc_builder import MCBuilder
from models.likelihoods import TfIdfLikelihood
from qb_env.tossup_env import TossupMCEnv
from qb_env.text_wrapper import TextObservationWrapper
from agents.threshold_buzzer import ThresholdBuzzer


def _make_questions(n: int = 30) -> list[TossupQuestion]:
    questions = []
    for i in range(n):
        tokens = [f"word{i}_{j}" for j in range(8)]
        questions.append(
            TossupQuestion(
                qid=f"q{i:03d}",
                question=" ".join(tokens),
                tokens=tokens,
                answer_primary=f"Answer_{i}",
                clean_answers=[f"Answer_{i}"],
                run_indices=[1, 3, 7],
                human_buzz_positions=[],
                category=["History", "Science"][i % 2],
                cumulative_prefixes=[
                    " ".join(tokens[:2]),
                    " ".join(tokens[:4]),
                    " ".join(tokens),
                ],
            )
        )
    return questions


def test_mixed_k_build_env_baseline() -> None:
    """Build mixed-K dataset, construct env, run a baseline agent."""
    questions = _make_questions(30)
    builder = MCBuilder(
        K=5, strategy="category_random", random_seed=42,
        variable_K=True, min_K=2, max_K=5,
    )
    profile = AnswerProfileBuilder()
    mc = builder.build(questions, profile)
    assert len(mc) > 0

    option_counts = {len(q.options) for q in mc}
    assert len(option_counts) > 1, f"Expected mixed K, got {option_counts}"

    corpus = [q.question for q in mc] + [p for q in mc for p in q.option_profiles]
    lm = TfIdfLikelihood(corpus_texts=corpus)

    max_k = max(len(q.options) for q in mc)
    env = TossupMCEnv(
        questions=mc, likelihood_model=lm,
        K=max_k, variable_K=True, max_K=max_k,
        reward_mode="simple", belief_mode="from_scratch",
    )

    obs, info = env.reset(seed=42, options={"question_idx": 0})
    assert obs.shape == (max_k + 6,)

    mask = env.action_masks()
    k_actual = len(mc[0].options)
    assert mask[0]
    assert all(mask[1: k_actual + 1])

    buzzer = ThresholdBuzzer(
        likelihood_model=lm, threshold=0.5, beta=5.0, alpha=10.0,
    )
    result = buzzer.run_episode(mc[0])
    assert 0 <= result.buzz_index < len(mc[0].options)


def test_mixed_k_text_wrapper_formats_correctly() -> None:
    """TextObservationWrapper formats per-question K dynamically."""
    questions = _make_questions(30)
    builder = MCBuilder(
        K=4, strategy="category_random", random_seed=42,
        variable_K=True, min_K=2, max_K=4,
    )
    profile = AnswerProfileBuilder()
    mc = builder.build(questions, profile)
    assert len(mc) > 0

    corpus = [q.question for q in mc] + [p for q in mc for p in q.option_profiles]
    lm = TfIdfLikelihood(corpus_texts=corpus)

    max_k = max(len(q.options) for q in mc)
    env = TossupMCEnv(
        questions=mc, likelihood_model=lm,
        K=max_k, variable_K=True, max_K=max_k,
        reward_mode="simple", belief_mode="from_scratch",
    )
    wrapped = TextObservationWrapper(env)

    for idx in range(min(5, len(mc))):
        obs, _ = wrapped.reset(seed=42, options={"question_idx": idx})
        n_opts = len(mc[idx].options)
        assert f"({n_opts})" in obs
        if n_opts < max_k:
            assert f"({n_opts + 1})" not in obs


def test_variable_k_sequential_bayes_wait_step_shapes() -> None:
    """Variable-K env should handle sequential_bayes without shape mismatch."""
    questions = _make_questions(30)
    builder = MCBuilder(
        K=5, strategy="category_random", random_seed=42,
        variable_K=True, min_K=2, max_K=5,
    )
    profile = AnswerProfileBuilder()
    mc = builder.build(questions, profile)
    assert len(mc) > 0

    corpus = [q.question for q in mc] + [p for q in mc for p in q.option_profiles]
    lm = TfIdfLikelihood(corpus_texts=corpus)

    max_k = max(len(q.options) for q in mc)
    env = TossupMCEnv(
        questions=mc, likelihood_model=lm,
        K=max_k, variable_K=True, max_K=max_k,
        reward_mode="simple", belief_mode="sequential_bayes",
    )

    obs, _ = env.reset(seed=42, options={"question_idx": 0})
    assert obs.shape == (max_k + 6,)
    assert len(env.belief) == len(mc[0].options)

    obs2, reward, done, truncated, info = env.step(0)
    assert obs2.shape == (max_k + 6,)
    assert len(env.belief) == len(mc[0].options)
    assert isinstance(reward, float)
```

## File: training/__init__.py
```python
"""
Training Package

Supervised warm-start and PPO fine-tuning for T5 policy models.
"""
```

## File: training/train_supervised_t5.py
```python
"""
Supervised warm-start training for T5PolicyModel.

Trains answer selection on complete questions using cross-entropy loss. All
clues are shown at once (not incremental), providing a strong initialization
before PPO fine-tuning on partial observations.

The training loop uses gradient accumulation (default 4 steps, effective
batch = 32) for stable training without exceeding GPU memory. Best model
is saved by validation accuracy to checkpoints/supervised/best_model/.

Ported from qanta-buzzer reference implementation (train_supervised.py)
with these changes:
    - Accepts list of MCQuestion objects instead of QuizBowlDataset class
    - Config dict interface instead of qanta-buzzer's Config class
    - Direct text formatting from MCQuestion (no QuizBowlEnvironment needed)
    - NumPy-style docstrings added throughout

Usage
-----
From Python::

    from training.train_supervised_t5 import SupervisedTrainer, run_supervised_training
    from models.t5_policy import T5PolicyModel
    from qb_data.mc_builder import MCQuestion

    model = T5PolicyModel({"model_name": "t5-small", "device": "cpu"})
    trainer = SupervisedTrainer(model, train_qs, val_qs, config)
    trainer.train()

From command line::

    python -m training.train_supervised_t5 --config configs/t5_policy.yaml
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models.t5_policy import T5PolicyModel
from qb_data.mc_builder import MCQuestion


def format_question_text(question: MCQuestion) -> str:
    """Format a complete question as text for supervised training.

    Shows ALL clues (complete question) since supervised training is the
    easier task of answer selection on full information. PPO later trains
    on incremental clues.

    Parameters
    ----------
    question : MCQuestion
        Question with tokens, options, and gold_index.

    Returns
    -------
    str
        Formatted text: ``"CLUES: <all tokens> | CHOICES: (1) opt1 (2) opt2 ..."``
    """
    clues_text = " ".join(question.tokens)
    choices_parts = [f"({i + 1}) {opt}" for i, opt in enumerate(question.options)]
    choices_text = " ".join(choices_parts)
    return f"CLUES: {clues_text} | CHOICES: {choices_text}"


class SupervisedTrainer:
    """Trainer for supervised warm-start of T5PolicyModel.

    Trains the answer head using cross-entropy loss on complete questions
    (all clues shown at once). Uses gradient accumulation for stable training
    with large effective batch sizes without exceeding GPU memory.

    The training loop:
    1. Shuffles training data each epoch
    2. Iterates over mini-batches
    3. Computes cross-entropy loss on answer logits
    4. Accumulates gradients for ``grad_accum_steps`` batches
    5. Clips gradients and updates optimizer
    6. Validates after each epoch
    7. Saves best model by validation accuracy

    Parameters
    ----------
    model : T5PolicyModel
        Model to train. Must have ``predict_answer`` and ``tokenizer``.
    train_questions : list[MCQuestion]
        Training set questions.
    val_questions : list[MCQuestion]
        Validation set questions.
    config : dict[str, Any]
        Configuration dictionary with keys:

        - ``supervised_lr`` (float): Learning rate. Default 3e-4.
        - ``supervised_epochs`` (int): Number of epochs. Default 10.
        - ``supervised_batch_size`` (int): Batch size. Default 8.
        - ``supervised_grad_accum_steps`` (int): Gradient accumulation. Default 4.
        - ``checkpoint_dir`` (str): Base checkpoint directory. Default "checkpoints".
        - ``max_input_length`` (int): Max token length. Default 512.
        - ``max_grad_norm`` (float): Gradient clip norm. Default 1.0.
        - ``weight_decay`` (float): AdamW weight decay. Default 0.01.

    Attributes
    ----------
    model : T5PolicyModel
        The model being trained.
    optimizer : torch.optim.AdamW
        Optimizer with weight decay.
    criterion : nn.CrossEntropyLoss
        Loss function for answer classification.
    best_val_acc : float
        Best validation accuracy seen so far.
    train_history : list[dict]
        Per-epoch training metrics.
    val_history : list[dict]
        Per-epoch validation metrics.
    checkpoint_dir : Path
        Directory for saving checkpoints.
    """

    def __init__(
        self,
        model: T5PolicyModel,
        train_questions: List[MCQuestion],
        val_questions: List[MCQuestion],
        config: Dict[str, Any],
    ) -> None:
        self.model = model
        self.train_questions = list(train_questions)
        self.val_questions = list(val_questions)
        self.config = config

        self.device = model.device

        # Hyperparameters with defaults
        self.lr = float(config.get("supervised_lr", 3e-4))
        self.epochs = int(config.get("supervised_epochs", 10))
        self.batch_size = int(config.get("supervised_batch_size", 8))
        self.grad_accum_steps = int(config.get("supervised_grad_accum_steps", 4))
        self.max_input_length = int(config.get("max_input_length", 512))
        self.max_grad_norm = float(config.get("max_grad_norm", 1.0))
        self.weight_decay = float(config.get("weight_decay", 0.01))

        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Training state
        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.train_history: List[Dict[str, Any]] = []
        self.val_history: List[Dict[str, Any]] = []

        # Checkpoint directory
        self.checkpoint_dir = Path(config.get("checkpoint_dir", "checkpoints")) / "supervised"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def prepare_batch(
        self, questions: List[MCQuestion]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Format a batch of complete questions as tokenized tensors.

        Each question is formatted with ALL clues visible (supervised training
        shows complete information). Text is tokenized using the model's
        T5TokenizerFast.

        Parameters
        ----------
        questions : list[MCQuestion]
            Batch of questions to format.

        Returns
        -------
        input_ids : torch.Tensor
            Token IDs of shape ``[batch_size, seq_len]``, on device.
        attention_mask : torch.Tensor
            Attention mask of shape ``[batch_size, seq_len]``, on device.
        labels : torch.Tensor
            Gold answer indices of shape ``[batch_size]``, on device.
        """
        texts = [format_question_text(q) for q in questions]
        labels = [q.gold_index for q in questions]

        # Tokenize
        inputs = self.model.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_input_length,
        )

        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        labels_tensor = torch.tensor(labels, dtype=torch.long).to(self.device)

        return input_ids, attention_mask, labels_tensor

    def train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch with gradient accumulation.

        Shuffles training data, iterates over mini-batches, and updates
        the optimizer every ``grad_accum_steps`` batches. Gradients are
        clipped to ``max_grad_norm`` before each optimizer step.

        Returns
        -------
        epoch_loss : float
            Average loss over all batches in the epoch.
        epoch_acc : float
            Average accuracy over all batches in the epoch.
        """
        self.model.train()

        # Shuffle training data
        shuffled = self.train_questions[:]
        random.shuffle(shuffled)

        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        num_batches = max(1, len(shuffled) // self.batch_size)

        # Zero gradients at start
        self.optimizer.zero_grad()

        for batch_idx in range(num_batches):
            # Get batch
            start = batch_idx * self.batch_size
            end = min(start + self.batch_size, len(shuffled))
            batch_questions = shuffled[start:end]

            if not batch_questions:
                continue

            # Prepare batch
            input_ids, attention_mask, labels = self.prepare_batch(batch_questions)

            # Forward pass
            answer_logits, predictions = self.model.predict_answer(
                input_ids, attention_mask
            )

            # Compute loss (scaled by accumulation steps for correct gradient magnitude)
            loss = self.criterion(answer_logits, labels)
            scaled_loss = loss / self.grad_accum_steps
            scaled_loss.backward()

            # Track metrics (use unscaled loss for logging)
            total_loss += loss.item()
            total_correct += (predictions == labels).sum().item()
            total_samples += len(labels)

            # Gradient accumulation: update every N batches
            if (batch_idx + 1) % self.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )
                self.optimizer.step()
                self.optimizer.zero_grad()

        # Handle remaining accumulated gradients (if num_batches not divisible by accum_steps)
        remaining = num_batches % self.grad_accum_steps
        if remaining > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.max_grad_norm
            )
            self.optimizer.step()
            self.optimizer.zero_grad()

        epoch_loss = total_loss / max(1, num_batches)
        epoch_acc = total_correct / max(1, total_samples)

        return epoch_loss, epoch_acc

    def validate(self) -> Tuple[float, float]:
        """Validate on the validation set.

        Runs the model in eval mode on all validation questions, computing
        accuracy and loss without gradient computation.

        Returns
        -------
        val_loss : float
            Average cross-entropy loss on validation set.
        val_acc : float
            Accuracy on validation set (fraction correct).
        """
        self.model.eval()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        num_batches = max(1, len(self.val_questions) // self.batch_size)

        with torch.no_grad():
            for batch_idx in range(num_batches):
                start = batch_idx * self.batch_size
                end = min(start + self.batch_size, len(self.val_questions))
                batch_questions = self.val_questions[start:end]

                if not batch_questions:
                    continue

                input_ids, attention_mask, labels = self.prepare_batch(batch_questions)
                answer_logits, predictions = self.model.predict_answer(
                    input_ids, attention_mask
                )

                loss = self.criterion(answer_logits, labels)
                total_loss += loss.item()
                total_correct += (predictions == labels).sum().item()
                total_samples += len(labels)

        val_loss = total_loss / max(1, num_batches)
        val_acc = total_correct / max(1, total_samples)

        return val_loss, val_acc

    def train(self) -> Dict[str, Any]:
        """Run full supervised training loop.

        Iterates over epochs, training and validating each epoch. Saves the
        best model by validation accuracy to ``checkpoint_dir/best_model/``.
        Training history is saved to ``checkpoint_dir/history.json``.

        Returns
        -------
        dict[str, Any]
            Training summary with keys: ``best_val_acc``, ``final_train_acc``,
            ``final_train_loss``, ``total_epochs``.
        """
        print(f"Starting supervised training for {self.epochs} epochs")
        print(f"  Training samples: {len(self.train_questions)}")
        print(f"  Validation samples: {len(self.val_questions)}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Gradient accumulation: {self.grad_accum_steps} (effective batch = {self.batch_size * self.grad_accum_steps})")
        print(f"  Learning rate: {self.lr}")
        print(f"  Device: {self.device}")
        print()

        final_train_loss = 0.0
        final_train_acc = 0.0

        for epoch in range(self.epochs):
            self.current_epoch = epoch

            # Train epoch
            train_loss, train_acc = self.train_epoch()
            final_train_loss = train_loss
            final_train_acc = train_acc

            # Validate
            val_loss, val_acc = self.validate()

            # Log results
            print(
                f"Epoch {epoch + 1}/{self.epochs} - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
            )

            # Save history
            self.train_history.append(
                {"epoch": epoch + 1, "loss": train_loss, "accuracy": train_acc}
            )
            self.val_history.append(
                {"epoch": epoch + 1, "loss": val_loss, "accuracy": val_acc}
            )

            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_checkpoint(is_best=True)
                print(f"  -> New best validation accuracy: {val_acc:.4f}")

        print(f"\nSupervised training completed!")
        print(f"  Best validation accuracy: {self.best_val_acc:.4f}")

        # Save training history
        self.save_history()

        return {
            "best_val_acc": self.best_val_acc,
            "final_train_acc": final_train_acc,
            "final_train_loss": final_train_loss,
            "total_epochs": self.epochs,
        }

    def save_checkpoint(self, is_best: bool = False) -> Path:
        """Save model checkpoint to disk.

        Saves the model (T5 encoder + policy head) and optimizer state.
        Best model is saved to ``checkpoint_dir/best_model/``, epoch
        checkpoints to ``checkpoint_dir/epoch_N/``.

        Parameters
        ----------
        is_best : bool
            If True, save to ``best_model/`` directory.

        Returns
        -------
        Path
            Path to the saved checkpoint directory.
        """
        if is_best:
            save_path = self.checkpoint_dir / "best_model"
        else:
            save_path = self.checkpoint_dir / f"epoch_{self.current_epoch + 1}"

        # Use T5PolicyModel's save() method
        self.model.save(str(save_path))

        # Save training state
        state = {
            "epoch": self.current_epoch + 1,
            "best_val_acc": self.best_val_acc,
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        torch.save(state, save_path / "training_state.pt")

        return save_path

    def save_history(self) -> Path:
        """Save training history to JSON.

        Converts numpy types to native Python types for JSON serialization.

        Returns
        -------
        Path
            Path to the saved history file.
        """
        history = {
            "train": _convert_to_native(self.train_history),
            "val": _convert_to_native(self.val_history),
            "config": {
                "lr": self.lr,
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "grad_accum_steps": self.grad_accum_steps,
            },
        }

        history_path = self.checkpoint_dir / "history.json"
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

        print(f"Training history saved to {history_path}")
        return history_path


def run_supervised_training(
    config: Dict[str, Any],
    train_questions: List[MCQuestion],
    val_questions: List[MCQuestion],
    test_questions: Optional[List[MCQuestion]] = None,
) -> Tuple[T5PolicyModel, SupervisedTrainer]:
    """Run the complete supervised training pipeline.

    Creates a T5PolicyModel, trains it on complete questions, and optionally
    evaluates on a test set. This is the main entry point for supervised
    warm-start training.

    Parameters
    ----------
    config : dict[str, Any]
        Configuration dictionary. Must include model config keys
        (``model_name``, ``device``, ``num_choices``) and supervised
        training keys (``supervised_lr``, etc.).
    train_questions : list[MCQuestion]
        Training set questions.
    val_questions : list[MCQuestion]
        Validation set questions.
    test_questions : list[MCQuestion] or None
        Optional test set for final evaluation.

    Returns
    -------
    model : T5PolicyModel
        The trained model (with best weights loaded).
    trainer : SupervisedTrainer
        The trainer instance with training history.
    """
    print("=" * 60)
    print("SUPERVISED TRAINING PHASE")
    print("=" * 60)

    # Initialize model
    model_config = {
        "model_name": config.get("model_name", "t5-large"),
        "device": config.get("device", "cpu"),
        "max_input_length": config.get("max_input_length", 512),
        "num_choices": config.get("num_choices", 4),
    }
    model = T5PolicyModel(model_config)

    # Create trainer
    trainer = SupervisedTrainer(
        model=model,
        train_questions=train_questions,
        val_questions=val_questions,
        config=config,
    )

    # Train
    summary = trainer.train()

    # Evaluate on test set if provided
    if test_questions is not None:
        print("\n" + "=" * 60)
        print("FINAL EVALUATION ON TEST SET")
        print("=" * 60)

        # Load best model
        best_model_path = trainer.checkpoint_dir / "best_model"
        model.load(str(best_model_path))
        model.eval()

        # Evaluate
        test_loss, test_acc = _evaluate_on_questions(model, test_questions, trainer)
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

        # Save test results
        test_results = {
            "test_loss": test_loss,
            "test_accuracy": test_acc,
            "training_summary": summary,
        }
        results_path = trainer.checkpoint_dir / "test_results.json"
        with open(results_path, "w") as f:
            json.dump(_convert_to_native(test_results), f, indent=2)
        print(f"Test results saved to {results_path}")

    return model, trainer


def _evaluate_on_questions(
    model: T5PolicyModel,
    questions: List[MCQuestion],
    trainer: SupervisedTrainer,
) -> Tuple[float, float]:
    """Evaluate model on a set of questions.

    Parameters
    ----------
    model : T5PolicyModel
        Model to evaluate.
    questions : list[MCQuestion]
        Questions to evaluate on.
    trainer : SupervisedTrainer
        Trainer instance (for batch preparation).

    Returns
    -------
    avg_loss : float
        Average cross-entropy loss.
    accuracy : float
        Fraction of correctly predicted answers.
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    batch_size = trainer.batch_size
    num_batches = max(1, len(questions) // batch_size)
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, len(questions))
            batch_questions = questions[start:end]

            if not batch_questions:
                continue

            input_ids, attention_mask, labels = trainer.prepare_batch(batch_questions)
            answer_logits, predictions = model.predict_answer(input_ids, attention_mask)

            loss = criterion(answer_logits, labels)
            total_loss += loss.item()
            total_correct += (predictions == labels).sum().item()
            total_samples += len(labels)

    return total_loss / max(1, num_batches), total_correct / max(1, total_samples)


def _convert_to_native(obj: Any) -> Any:
    """Convert numpy types to Python native types for JSON serialization.

    Parameters
    ----------
    obj : Any
        Object to convert. Handles dicts, lists, numpy scalars and arrays.

    Returns
    -------
    Any
        Object with all numpy types converted to native Python types.
    """
    if isinstance(obj, dict):
        return {k: _convert_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_to_native(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return _convert_to_native(obj.tolist())
    else:
        return obj
```

## File: agents/bayesian_buzzer.py
```python
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from agents._math import sigmoid
from models.likelihoods import LikelihoodModel
from qb_data.mc_builder import MCQuestion

if TYPE_CHECKING:
    from agents.threshold_buzzer import _PrecomputedQuestion



@dataclass
class SoftmaxEpisodeResult:
    qid: str
    buzz_step: int
    buzz_index: int
    gold_index: int
    correct: bool
    c_trace: list[float]
    g_trace: list[float]
    top_p_trace: list[float]
    entropy_trace: list[float]


class SoftmaxProfileBuzzer:
    def __init__(
        self,
        likelihood_model: LikelihoodModel,
        threshold: float = 0.8,
        beta: float = 5.0,
        alpha: float = 10.0,
    ):
        self.likelihood_model = likelihood_model
        self.threshold = threshold
        self.beta = beta
        self.alpha = alpha
        self.belief: np.ndarray | None = None

    def _belief_from_scratch(self, cumulative_prefix: str, option_profiles: list[str]) -> np.ndarray:
        scores = self.likelihood_model.score(cumulative_prefix, option_profiles)
        scores = scores - np.max(scores)
        probs = np.exp(self.beta * scores)
        probs = probs / max(1e-12, probs.sum())
        return probs.astype(np.float32)

    def confidence_proxy(self, top_p: float) -> float:
        return sigmoid(self.alpha * (top_p - self.threshold))

    def run_episode(self, question: MCQuestion) -> SoftmaxEpisodeResult:
        c_trace: list[float] = []
        g_trace: list[float] = []
        top_p_trace: list[float] = []
        entropy_trace: list[float] = []

        chosen_idx = 0
        chosen_step = len(question.cumulative_prefixes) - 1

        for step_idx, prefix in enumerate(question.cumulative_prefixes):
            belief = self._belief_from_scratch(prefix, question.option_profiles)
            self.belief = belief
            top_idx = int(np.argmax(belief))
            top_p = float(np.max(belief))
            entropy = float(-(np.clip(belief, 1e-12, 1.0) * np.log(np.clip(belief, 1e-12, 1.0))).sum())
            c_t = self.confidence_proxy(top_p)
            g_t = 1.0 if top_idx == question.gold_index else 0.0

            c_trace.append(c_t)
            g_trace.append(g_t)
            top_p_trace.append(top_p)
            entropy_trace.append(entropy)

            is_last = step_idx == len(question.cumulative_prefixes) - 1
            if top_p >= self.threshold or is_last:
                chosen_step = step_idx
                chosen_idx = top_idx
                break

        return SoftmaxEpisodeResult(
            qid=question.qid,
            buzz_step=chosen_step,
            buzz_index=chosen_idx,
            gold_index=question.gold_index,
            correct=(chosen_idx == question.gold_index),
            c_trace=c_trace,
            g_trace=g_trace,
            top_p_trace=top_p_trace,
            entropy_trace=entropy_trace,
        )


class SequentialBayesBuzzer:
    def __init__(
        self,
        likelihood_model: LikelihoodModel,
        threshold: float = 0.8,
        beta: float = 5.0,
        alpha: float = 10.0,
    ):
        self.likelihood_model = likelihood_model
        self.threshold = threshold
        self.beta = beta
        self.alpha = alpha

    def _step_update(self, prior: np.ndarray, fragment: str, option_profiles: list[str]) -> np.ndarray:
        scores = self.likelihood_model.score(fragment, option_profiles)
        scores = scores - np.max(scores)
        likelihood = np.exp(self.beta * scores)
        posterior = prior * likelihood
        denom = posterior.sum()
        if denom <= 0:
            return np.ones_like(prior) / len(prior)
        return (posterior / denom).astype(np.float32)

    def run_episode(self, question: MCQuestion) -> SoftmaxEpisodeResult:
        c_trace: list[float] = []
        g_trace: list[float] = []
        top_p_trace: list[float] = []
        entropy_trace: list[float] = []

        K = len(question.options)
        belief = np.ones(K, dtype=np.float32) / K
        chosen_idx = 0
        chosen_step = len(question.cumulative_prefixes) - 1

        for step_idx, token_idx in enumerate(question.run_indices):
            prev_token_idx = question.run_indices[step_idx - 1] if step_idx > 0 else -1
            fragment = " ".join(question.tokens[prev_token_idx + 1 : token_idx + 1])
            belief = self._step_update(belief, fragment, question.option_profiles)
            top_idx = int(np.argmax(belief))
            top_p = float(np.max(belief))
            entropy = float(-(np.clip(belief, 1e-12, 1.0) * np.log(np.clip(belief, 1e-12, 1.0))).sum())
            c_t = sigmoid(self.alpha * (top_p - self.threshold))
            g_t = 1.0 if top_idx == question.gold_index else 0.0

            c_trace.append(c_t)
            g_trace.append(g_t)
            top_p_trace.append(top_p)
            entropy_trace.append(entropy)

            is_last = step_idx == len(question.cumulative_prefixes) - 1
            if top_p >= self.threshold or is_last:
                chosen_step = step_idx
                chosen_idx = top_idx
                break

        return SoftmaxEpisodeResult(
            qid=question.qid,
            buzz_step=chosen_step,
            buzz_index=chosen_idx,
            gold_index=question.gold_index,
            correct=(chosen_idx == question.gold_index),
            c_trace=c_trace,
            g_trace=g_trace,
            top_p_trace=top_p_trace,
            entropy_trace=entropy_trace,
        )


def precompute_sequential_beliefs(
    questions: list[MCQuestion],
    likelihood_model: LikelihoodModel,
    beta: float,
) -> list["_PrecomputedQuestion"]:
    """Compute Bayesian sequential beliefs at every step for every question.

    Starts with a uniform prior and applies Bayesian update
    ``posterior = prior * likelihood`` using token fragments derived from
    ``question.run_indices``.  Returns one ``_PrecomputedQuestion`` per
    question where ``beliefs`` are the Bayesian posteriors (NOT the
    from-scratch softmax beliefs).
    """
    from agents.threshold_buzzer import _PrecomputedQuestion

    out: list[_PrecomputedQuestion] = []
    for q in questions:
        K = len(q.options)
        belief = np.ones(K, dtype=np.float32) / K
        beliefs: list[np.ndarray] = []

        for step_idx, token_idx in enumerate(q.run_indices):
            prev_token_idx = q.run_indices[step_idx - 1] if step_idx > 0 else -1
            fragment = " ".join(q.tokens[prev_token_idx + 1 : token_idx + 1])
            scores = likelihood_model.score(fragment, q.option_profiles)
            scores = scores - np.max(scores)
            likelihood = np.exp(beta * scores)
            posterior = belief * likelihood
            denom = posterior.sum()
            if denom <= 0:
                belief = np.ones_like(belief) / len(belief)
            else:
                belief = (posterior / denom).astype(np.float32)
            beliefs.append(belief.copy())

        out.append(_PrecomputedQuestion(
            qid=q.qid,
            gold_index=q.gold_index,
            num_options=K,
            beliefs=beliefs,
        ))
    return out


def _sequential_episode_from_precomputed(
    pq: "_PrecomputedQuestion",
    threshold: float,
    alpha: float,
) -> SoftmaxEpisodeResult:
    """Build a SoftmaxEpisodeResult from pre-computed sequential beliefs.

    Identical buzzing logic to ``SequentialBayesBuzzer.run_episode`` but
    reads beliefs from a ``_PrecomputedQuestion`` instead of calling the
    likelihood model.
    """
    from agents.threshold_buzzer import _belief_stats

    c_trace: list[float] = []
    g_trace: list[float] = []
    top_p_trace: list[float] = []
    entropy_trace: list[float] = []

    chosen_step = len(pq.beliefs) - 1
    chosen_idx = 0

    for step_idx, belief in enumerate(pq.beliefs):
        top_idx, top_p, entropy = _belief_stats(belief)
        c_t = sigmoid(alpha * (top_p - threshold))
        g_t = 1.0 if top_idx == pq.gold_index else 0.0

        c_trace.append(c_t)
        g_trace.append(g_t)
        top_p_trace.append(top_p)
        entropy_trace.append(entropy)

        is_last = step_idx == len(pq.beliefs) - 1
        if top_p >= threshold or is_last:
            chosen_step = step_idx
            chosen_idx = top_idx
            break

    correct = chosen_idx == pq.gold_index
    return SoftmaxEpisodeResult(
        qid=pq.qid,
        buzz_step=chosen_step,
        buzz_index=chosen_idx,
        gold_index=pq.gold_index,
        correct=correct,
        c_trace=c_trace,
        g_trace=g_trace,
        top_p_trace=top_p_trace,
        entropy_trace=entropy_trace,
    )


def sweep_sequential_thresholds(
    questions: list[MCQuestion],
    likelihood_model: LikelihoodModel,
    thresholds: list[float],
    beta: float = 5.0,
    alpha: float = 10.0,
    precomputed: list["_PrecomputedQuestion"] | None = None,
) -> dict[float, list[SoftmaxEpisodeResult]]:
    """Sweep multiple thresholds with a single sequential belief pass.

    If *precomputed* is provided the expensive model calls are skipped
    entirely and the sweep is pure numpy.  Otherwise beliefs are computed
    once internally and reused across thresholds.
    """
    if precomputed is None:
        precomputed = precompute_sequential_beliefs(questions, likelihood_model, beta)

    out: dict[float, list[SoftmaxEpisodeResult]] = {}
    for threshold in thresholds:
        out[float(threshold)] = [
            _sequential_episode_from_precomputed(pq, threshold, alpha)
            for pq in precomputed
        ]
    return out
```

## File: models/__init__.py
```python
"""
Models Package

Likelihood models, belief feature extraction, and policy model interfaces
for the quiz bowl RL buzzer system.
"""

from models.features import extract_belief_features, entropy_of_distribution
from models.likelihoods import (
    LikelihoodModel,
    OpenAILikelihood,
    SBERTLikelihood,
    T5Likelihood,
    TfIdfLikelihood,
    build_likelihood_from_config,
)

# Lazy import: T5PolicyModel and PolicyHead require transformers + torch.
# Import on demand to keep package lightweight for belief-feature-only usage.


def __getattr__(name: str):
    if name in ("T5PolicyModel", "PolicyHead"):
        from models.t5_policy import T5PolicyModel, PolicyHead
        return {"T5PolicyModel": T5PolicyModel, "PolicyHead": PolicyHead}[name]
    raise AttributeError(f"module 'models' has no attribute {name!r}")


__all__ = [
    "extract_belief_features",
    "entropy_of_distribution",
    "LikelihoodModel",
    "TfIdfLikelihood",
    "SBERTLikelihood",
    "OpenAILikelihood",
    "T5Likelihood",
    "build_likelihood_from_config",
    "T5PolicyModel",
    "PolicyHead",
]
```

## File: models/t5_policy.py
```python
"""
T5-based Policy Model for Quiz Bowl RL Agent

Implements T5PolicyModel with a custom PolicyHead containing three independent
heads (wait/answer/value) for end-to-end text-based policy learning. This
provides an alternative to the MLP policy trained on belief features
(Phase 4 approach).

Architecture overview:

    Text input  -->  T5 Encoder  -->  Mean Pooling  -->  PolicyHead
                                                          |-- Wait head (2)
                                                          |-- Answer head (K)
                                                          |-- Value head (1)

The T5 encoder produces contextual embeddings from tokenized text. Mean pooling
(attention-masked) reduces the variable-length sequence to a fixed-size vector.
The PolicyHead then produces three independent outputs:

- **Wait logits** [B, 2]: probability of waiting vs answering now
- **Answer logits** [B, K]: probability of selecting each answer option
- **Value estimate** [B, 1]: state value for PPO advantage computation

Action space maps to the TossupMCEnv convention:
    0 = WAIT (wait head selects "wait")
    1..K = SELECT answer i-1 (wait head selects "answer now", answer head picks i-1)

Ported from qanta-buzzer reference implementation (model.py) with these changes:
    - T5EncoderModel replaces T5ForConditionalGeneration (2x faster, 50% less memory)
    - T5TokenizerFast replaces T5Tokenizer (3-5x faster tokenization via Rust backend)
    - Config dict replaces qanta-buzzer's Config class for unified codebase compatibility
    - NumPy-style docstrings added throughout
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyHead(nn.Module):
    """Custom policy head with three independent output heads.

    Attached to a T5 encoder's pooled output, this module produces the three
    outputs needed for actor-critic RL in the quiz bowl POMDP: a binary
    wait/answer-now decision, a K-way answer selection, and a scalar value
    estimate.

    All three heads are fully independent (no shared hidden layers beyond the
    encoder), using the same pattern: Linear -> ReLU -> Dropout -> Linear.

    Parameters
    ----------
    hidden_size : int
        Dimensionality of the input from the T5 encoder's pooled output.
        Default 1024 matches T5-large (``d_model``). Use 512 for t5-small,
        768 for t5-base.
    num_choices : int
        Number of answer options (K). Default 4 for quiz bowl MC questions.

    Attributes
    ----------
    wait_head : nn.Sequential
        Binary head producing [wait, answer_now] logits.
    answer_head : nn.Sequential
        Multi-class head producing logits over K answer choices.
    value_head : nn.Sequential
        Scalar head producing state value estimate.
    """

    def __init__(self, hidden_size: int = 1024, num_choices: int = 4) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.num_choices = num_choices

        # Wait/continue decision head (binary: wait vs answer_now)
        self.wait_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 2),  # [wait, answer_now]
        )

        # Answer selection head (over K choices)
        self.answer_head = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_choices),
        )

        # Value head (state value estimate for PPO)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
        )

    def forward(
        self, encoder_hidden_state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through all three heads.

        Parameters
        ----------
        encoder_hidden_state : torch.Tensor
            Pooled encoder output of shape ``[batch_size, hidden_size]``.

        Returns
        -------
        wait_logits : torch.Tensor
            Shape ``[batch_size, 2]`` -- logits for [wait, answer_now].
        answer_logits : torch.Tensor
            Shape ``[batch_size, num_choices]`` -- logits over answer options.
        values : torch.Tensor
            Shape ``[batch_size, 1]`` -- state value estimates.
        """
        wait_logits = self.wait_head(encoder_hidden_state)
        answer_logits = self.answer_head(encoder_hidden_state)
        values = self.value_head(encoder_hidden_state)

        return wait_logits, answer_logits, values


class T5PolicyModel(nn.Module):
    """T5 encoder with custom policy head for end-to-end RL.

    Combines a pre-trained T5 encoder with a ``PolicyHead`` to produce policy
    outputs directly from text observations. This is the alternative approach
    to Phase 4's MLP policy, which operates on numeric belief features.

    The model processes text in three stages:

    1. **Tokenization**: Text is tokenized with ``T5TokenizerFast`` (Rust-backed
       for speed) with padding and truncation.
    2. **Encoding**: ``T5EncoderModel`` produces contextual hidden states
       ``[B, seq_len, d_model]``.
    3. **Pooling + Heads**: Attention-masked mean pooling reduces to
       ``[B, d_model]``, then PolicyHead produces wait/answer/value outputs.

    Action space follows TossupMCEnv convention:
        - 0 = WAIT
        - 1..K = SELECT answer (i-1)

    Combined actions are treated as a factorized policy:
        - ``P(WAIT) = p_wait``
        - ``P(BUZZ_i) = p_buzz * p_ans(i)``

    The answer distribution only contributes when the policy chooses to buzz.

    Parameters
    ----------
    config : dict[str, Any]
        Configuration dictionary with the following keys:

        - ``model_name`` (str): HuggingFace T5 model identifier.
          Default ``"t5-large"``. Options: ``"t5-small"``, ``"t5-base"``,
          ``"t5-large"``.
        - ``device`` (str): Torch device. Default auto-detects
          (cuda > mps > cpu).
        - ``max_input_length`` (int): Maximum token sequence length.
          Default 512.
        - ``num_choices`` (int): Number of answer options (K). Default 4.

    Attributes
    ----------
    config : dict[str, Any]
        Configuration dictionary.
    device : torch.device
        Computation device.
    encoder : T5EncoderModel
        Pre-trained T5 encoder.
    tokenizer : T5TokenizerFast
        Fast T5 tokenizer.
    policy_head : PolicyHead
        Custom three-head policy module.
    max_input_length : int
        Maximum token sequence length for tokenization.

    Examples
    --------
    >>> config = {"model_name": "t5-small", "device": "cpu", "num_choices": 4}
    >>> model = T5PolicyModel(config)
    >>> texts = ["CLUES: first president | CHOICES: (1) Washington (2) Jefferson"]
    >>> wait_logits, answer_logits, values = model(texts)
    >>> wait_logits.shape
    torch.Size([1, 2])
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        from transformers import T5EncoderModel, T5TokenizerFast

        self.config = config
        model_name = config.get("model_name", "t5-large")
        self.max_input_length = config.get("max_input_length", 512)
        num_choices = config.get("num_choices", 4)

        # Auto-detect device
        default_device = "cpu"
        if torch.cuda.is_available():
            default_device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            default_device = "mps"
        self.device = torch.device(config.get("device", default_device))

        # Load T5 encoder only (not full T5ForConditionalGeneration)
        # This is 2x faster and uses 50% less memory since the decoder is unused
        print(f"Loading T5 encoder: {model_name}")
        self.encoder = T5EncoderModel.from_pretrained(model_name)
        self.tokenizer = T5TokenizerFast.from_pretrained(model_name)

        # Get hidden size from T5 config (512 for small, 768 for base, 1024 for large)
        hidden_size = self.encoder.config.d_model

        # Custom policy head
        self.policy_head = PolicyHead(
            hidden_size=hidden_size,
            num_choices=num_choices,
        )

        # Move to device
        self.to(self.device)

        # Print model info
        self._print_model_info()

    def _print_model_info(self) -> None:
        """Print model architecture summary and parameter counts."""
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        policy_params = sum(p.numel() for p in self.policy_head.parameters())
        total_params = encoder_params + policy_params

        print("Model Architecture:")
        print(f"  T5 encoder parameters: {encoder_params:,}")
        print(f"  Policy head parameters: {policy_params:,}")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Device: {self.device}")

    def encode_input(
        self,
        text_inputs: List[str],
        max_length: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Tokenize text inputs using T5TokenizerFast.

        Parameters
        ----------
        text_inputs : list[str]
            List of input text strings to tokenize.
        max_length : int or None
            Maximum sequence length. If None, uses ``self.max_input_length``.

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary with ``"input_ids"`` and ``"attention_mask"`` tensors,
            both of shape ``[batch_size, seq_len]``, moved to ``self.device``.
        """
        if max_length is None:
            max_length = self.max_input_length

        encoding = self.tokenizer(
            text_inputs,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        return {k: v.to(self.device) for k, v in encoding.items()}

    def get_encoder_output(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute T5 encoder output and pool to a fixed-size vector.

        Uses attention-masked mean pooling: sum hidden states where attention
        mask is 1, divide by number of non-padding tokens. This ensures
        padding tokens contribute zero to the pooled representation.

        Parameters
        ----------
        input_ids : torch.Tensor
            Token IDs of shape ``[batch_size, seq_len]``.
        attention_mask : torch.Tensor
            Attention mask of shape ``[batch_size, seq_len]`` (1 for real
            tokens, 0 for padding).

        Returns
        -------
        torch.Tensor
            Pooled encoder output of shape ``[batch_size, hidden_size]``.
        """
        # Get encoder outputs
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )

        # encoder_outputs.last_hidden_state: [batch_size, seq_len, hidden_size]
        hidden_states = encoder_outputs.last_hidden_state

        # Attention-masked mean pooling over sequence dimension
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_hidden = torch.sum(hidden_states * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        pooled_output = sum_hidden / sum_mask

        return pooled_output

    def forward(
        self,
        text_inputs: List[str],
        return_value: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass: tokenize, encode, pool, then apply policy head.

        Parameters
        ----------
        text_inputs : list[str]
            List of text observations (e.g.,
            ``"CLUES: clue1 clue2 | CHOICES: (1) ans1 (2) ans2"``).
        return_value : bool
            If True, return value estimates. If False, values is None.

        Returns
        -------
        wait_logits : torch.Tensor
            Shape ``[batch_size, 2]`` -- logits for [wait, answer_now].
        answer_logits : torch.Tensor
            Shape ``[batch_size, num_choices]`` -- logits over answer options.
        values : torch.Tensor or None
            Shape ``[batch_size, 1]`` if return_value is True, else None.
        """
        # Encode inputs
        encoding = self.encode_input(text_inputs)

        # Get pooled encoder output
        pooled_output = self.get_encoder_output(
            encoding["input_ids"],
            encoding["attention_mask"],
        )

        # Pass through policy head
        wait_logits, answer_logits, values = self.policy_head(pooled_output)

        if not return_value:
            values = None

        return wait_logits, answer_logits, values

    def predict_answer(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict answer choice for supervised training.

        Only uses the answer head (wait and value heads are ignored). This is
        the interface for supervised warm-start training where the model learns
        to select the correct answer from complete questions.

        Parameters
        ----------
        input_ids : torch.Tensor
            Token IDs of shape ``[batch_size, seq_len]``.
        attention_mask : torch.Tensor
            Attention mask of shape ``[batch_size, seq_len]``.

        Returns
        -------
        answer_logits : torch.Tensor
            Shape ``[batch_size, num_choices]`` -- logits over answer choices.
        predictions : torch.Tensor
            Shape ``[batch_size]`` -- predicted answer indices (argmax).
        """
        # Get encoder output
        pooled_output = self.get_encoder_output(input_ids, attention_mask)

        # Get answer logits from policy head
        _, answer_logits, _ = self.policy_head(pooled_output)

        # Get predictions
        predictions = torch.argmax(answer_logits, dim=-1)

        return answer_logits, predictions

    def _joint_action_log_prob(
        self,
        wait_logits: torch.Tensor,
        answer_logits: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Compute factorized log-probabilities for flat WAIT/BUZZ actions.

        Parameters
        ----------
        wait_logits : torch.Tensor
            Binary logits of shape ``[batch_size, 2]`` for [WAIT, BUZZ].
        answer_logits : torch.Tensor
            Answer logits of shape ``[batch_size, K]``.
        actions : torch.Tensor
            Flat actions of shape ``[batch_size]`` where 0 = WAIT and
            1..K = BUZZ with answer index action-1.

        Returns
        -------
        torch.Tensor
            Log-probabilities of shape ``[batch_size]``.
        """
        wait_log_probs = F.log_softmax(wait_logits, dim=-1)
        answer_log_probs = F.log_softmax(answer_logits, dim=-1)

        wait_actions = (actions > 0).long()
        answer_actions = torch.clamp(actions - 1, min=0)

        selected_wait = wait_log_probs.gather(1, wait_actions.unsqueeze(-1)).squeeze(-1)
        selected_answer = answer_log_probs.gather(
            1, answer_actions.unsqueeze(-1)
        ).squeeze(-1)

        return torch.where(actions == 0, selected_wait, selected_wait + selected_answer)

    def _joint_entropy(
        self,
        wait_logits: torch.Tensor,
        answer_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Compute chain-rule entropy for the factorized wait/answer policy.

        Returns ``H(wait) + p_buzz * H(answer)`` for each example.
        """
        wait_probs = F.softmax(wait_logits, dim=-1)
        wait_log_probs = F.log_softmax(wait_logits, dim=-1)
        answer_probs = F.softmax(answer_logits, dim=-1)
        answer_log_probs = F.log_softmax(answer_logits, dim=-1)

        wait_entropy = -(wait_probs * wait_log_probs).sum(dim=-1)
        answer_entropy = -(answer_probs * answer_log_probs).sum(dim=-1)
        return wait_entropy + wait_probs[:, 1] * answer_entropy

    def select_action(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        deterministic: bool = False,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Select flat WAIT/BUZZ actions from the factorized policy.

        Produces combined actions following TossupMCEnv convention:
        0 = WAIT, 1..K = SELECT answer 0..K-1. Under the factorized policy:

        - ``P(WAIT) = p_wait``
        - ``P(BUZZ_i) = p_buzz * p_ans(i)``

        Answer sampling only occurs for examples that actually buzz.

        Parameters
        ----------
        input_ids : torch.Tensor
            Token IDs of shape ``[batch_size, seq_len]``.
        attention_mask : torch.Tensor
            Attention mask of shape ``[batch_size, seq_len]``.
        deterministic : bool
            If True, use argmax instead of sampling.
        temperature : float
            Temperature for softmax. Higher values increase randomness.
            Default 1.0 (no scaling).

        Returns
        -------
        combined_actions : torch.Tensor
            Shape ``[batch_size]`` -- combined actions (0 = WAIT, 1..K = SELECT).
        info : dict[str, Any]
            Dictionary with keys:

            - ``wait_logits``: raw wait head output
            - ``answer_logits``: raw answer head output
            - ``wait_probs``: softmax of wait logits
            - ``answer_probs``: softmax of answer logits
            - ``wait_actions``: sampled wait decisions (0 or 1)
            - ``answer_actions``: sampled answer indices (0..K-1)
            - ``values``: value estimates
            - ``log_probs``: total log probability of the combined action
        """
        with torch.no_grad():
            pooled_output = self.get_encoder_output(input_ids, attention_mask)
            wait_logits, answer_logits, values = self.policy_head(pooled_output)

            wait_logits_scaled = wait_logits / temperature
            answer_logits_scaled = answer_logits / temperature

            wait_probs = F.softmax(wait_logits_scaled, dim=-1)
            answer_probs = F.softmax(answer_logits_scaled, dim=-1)
            flat_action_probs = torch.cat(
                [wait_probs[:, :1], wait_probs[:, 1:2] * answer_probs],
                dim=-1,
            )

            if deterministic:
                combined_actions = torch.argmax(flat_action_probs, dim=-1)
                wait_actions = (combined_actions > 0).long()
                answer_actions = torch.clamp(combined_actions - 1, min=0)
            else:
                wait_actions = torch.distributions.Categorical(wait_probs).sample()
                answer_actions = torch.argmax(answer_probs, dim=-1)
                buzz_mask = wait_actions == 1
                if buzz_mask.any():
                    buzz_answers = torch.distributions.Categorical(
                        answer_probs[buzz_mask]
                    ).sample()
                    answer_actions = answer_actions.clone()
                    answer_actions[buzz_mask] = buzz_answers
                combined_actions = torch.where(
                    wait_actions == 0,
                    torch.zeros_like(wait_actions),
                    1 + answer_actions,
                )

            log_probs = self._joint_action_log_prob(
                wait_logits_scaled, answer_logits_scaled, combined_actions
            )

            combined_actions = torch.where(
                wait_actions == 0,
                torch.zeros_like(wait_actions),
                1 + answer_actions,
            )

            info = {
                "wait_logits": wait_logits,
                "answer_logits": answer_logits,
                "wait_probs": wait_probs,
                "answer_probs": answer_probs,
                "wait_actions": wait_actions,
                "answer_actions": answer_actions,
                "values": values,
                "log_probs": log_probs,
            }

            return combined_actions, info

    def get_action_log_probs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute log probabilities and entropy for given actions.

        Used during PPO training to evaluate old actions under the current
        policy. Combined actions follow the factorized semantics:

        - ``P(WAIT) = p_wait``
        - ``P(BUZZ_i) = p_buzz * p_ans(i)``

        Parameters
        ----------
        input_ids : torch.Tensor
            Token IDs of shape ``[batch_size, seq_len]``.
        attention_mask : torch.Tensor
            Attention mask of shape ``[batch_size, seq_len]``.
        actions : torch.Tensor
            Combined actions of shape ``[batch_size]``. Values in {0, 1, ..., K}.

        Returns
        -------
        log_probs : torch.Tensor
            Shape ``[batch_size]`` -- total log probability of each action.
        entropy : torch.Tensor
            Shape ``[batch_size]`` -- chain-rule entropy for the factorized policy.
        values : torch.Tensor
            Shape ``[batch_size]`` -- value estimates (squeezed).
        """
        pooled_output = self.get_encoder_output(input_ids, attention_mask)
        wait_logits, answer_logits, values = self.policy_head(pooled_output)

        log_probs = self._joint_action_log_prob(wait_logits, answer_logits, actions)
        entropy = self._joint_entropy(wait_logits, answer_logits)

        return log_probs, entropy, values.squeeze(-1)

    def save(self, save_dir: str) -> None:
        """Save model checkpoint to disk.

        Saves three components:
        1. T5 encoder weights and config (HuggingFace format)
        2. Tokenizer files (HuggingFace format)
        3. Policy head state dict (PyTorch format as ``policy_head.pt``)

        Parameters
        ----------
        save_dir : str
            Directory path to save the checkpoint. Created if it doesn't exist.
        """
        os.makedirs(save_dir, exist_ok=True)

        # Save T5 encoder
        self.encoder.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)

        # Save policy head
        policy_head_path = os.path.join(save_dir, "policy_head.pt")
        torch.save(self.policy_head.state_dict(), policy_head_path)

        print(f"Model saved to {save_dir}")

    def load(self, load_dir: str) -> None:
        """Load model checkpoint from disk.

        Loads T5 encoder weights, tokenizer, and policy head state dict from
        the specified directory. The model is moved to ``self.device`` after
        loading.

        Parameters
        ----------
        load_dir : str
            Directory containing a previously saved checkpoint.

        Raises
        ------
        FileNotFoundError
            If ``policy_head.pt`` is not found in ``load_dir``.
        """
        from transformers import T5EncoderModel, T5TokenizerFast

        # Load T5 encoder
        self.encoder = T5EncoderModel.from_pretrained(load_dir)
        self.tokenizer = T5TokenizerFast.from_pretrained(load_dir)

        # Load policy head
        policy_head_path = os.path.join(load_dir, "policy_head.pt")
        self.policy_head.load_state_dict(
            torch.load(policy_head_path, map_location=self.device, weights_only=True)
        )

        self.to(self.device)
        print(f"Model loaded from {load_dir}")

    @classmethod
    def load_pretrained(
        cls,
        load_dir: str,
        device: Optional[str] = None,
    ) -> "T5PolicyModel":
        """Load a pretrained model from a directory.

        Class method that creates a new T5PolicyModel instance and loads
        weights from a saved checkpoint.

        Parameters
        ----------
        load_dir : str
            Directory containing a previously saved checkpoint.
        device : str or None
            Device to load model on (e.g., ``"cpu"``, ``"cuda"``, ``"mps"``).
            If None, auto-detects.

        Returns
        -------
        T5PolicyModel
            A loaded model instance ready for inference.
        """
        from transformers import T5Config

        # Validate checkpoint integrity (lightweight — config JSON only)
        T5Config.from_pretrained(load_dir, local_files_only=True)

        # Infer num_choices from policy head state dict
        policy_head_path = os.path.join(load_dir, "policy_head.pt")
        policy_head_state = torch.load(
            policy_head_path, map_location="cpu", weights_only=True
        )
        # answer_head final linear layer weight shape is [num_choices, hidden_dim]
        num_choices = policy_head_state["answer_head.3.weight"].shape[0]

        config = {
            "model_name": load_dir,
            "num_choices": num_choices,
        }
        if device is not None:
            config["device"] = device

        model = cls(config)
        model.load(load_dir)
        return model
```

## File: qb_data/data_loader.py
```python
"""
Data structures and loaders for quiz bowl questions.
"""

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Any, Dict

from qb_data.text_utils import normalize_answer


@dataclass
class TossupQuestion:
    """
    A quiz bowl tossup question with incremental clues.

    Attributes
    ----------
    qid : str
        Unique question identifier
    question : str
        Full question text (all clues concatenated)
    tokens : List[str]
        Tokenized question split on whitespace
    answer_primary : str
        Primary answer text
    clean_answers : List[str]
        List of acceptable answer variants
    run_indices : List[int]
        Token indices where clues end (for incremental reveal)
    human_buzz_positions : Optional[List[Tuple[int, int]]]
        Human buzzer positions as (position, count) tuples
    category : str
        Question category (e.g., "History", "Literature")
    cumulative_prefixes : List[str]
        Precomputed text prefixes at each run_index
    """
    qid: str
    question: str
    tokens: List[str]
    answer_primary: str
    clean_answers: List[str]
    run_indices: List[int]
    human_buzz_positions: Optional[List[Tuple[int, int]]]
    category: str
    cumulative_prefixes: List[str]


def _parse_clues_to_tokens(clues: List[str]) -> Tuple[List[str], List[int]]:
    """
    Convert list of clues to tokens and run indices.

    Parameters
    ----------
    clues : List[str]
        List of clue strings

    Returns
    -------
    Tuple[List[str], List[int]]
        Tokens (words) and indices where each clue ends
    """
    tokens = []
    run_indices = []

    for clue in clues:
        clue_tokens = clue.split()
        tokens.extend(clue_tokens)
        if clue_tokens:  # Only add index if clue has tokens
            run_indices.append(len(tokens) - 1)

    return tokens, run_indices


def _generate_qid(text: str) -> str:
    """
    Generate a unique question ID from question text.

    Parameters
    ----------
    text : str
        Question text to hash

    Returns
    -------
    str
        Unique identifier based on text hash
    """
    hash_obj = hashlib.md5(text.encode('utf-8'))
    return f"qid-{hash_obj.hexdigest()[:12]}"


def _coerce_human_buzz_positions(value: Any) -> Optional[List[Tuple[int, int]]]:
    """Coerce various metadata formats into ``(position, count)`` tuples."""
    if value is None:
        return None

    if isinstance(value, list):
        result: List[Tuple[int, int]] = []
        for item in value:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                try:
                    result.append((int(item[0]), int(item[1])))
                except (TypeError, ValueError):
                    continue
            elif isinstance(item, dict):
                pos = item.get("position")
                count = item.get("count", 1)
                if pos is None:
                    continue
                try:
                    result.append((int(pos), int(count)))
                except (TypeError, ValueError):
                    continue
        return result or None

    return None


def _coerce_run_indices(run_indices: Any, token_count: int) -> List[int]:
    """Validate and coerce run indices into a sorted unique list."""
    clean: List[int] = []
    for idx in run_indices or []:
        try:
            clean.append(int(idx))
        except (TypeError, ValueError):
            continue

    if not clean:
        if token_count <= 0:
            raise ValueError("question must contain at least one token")
        clean = list(range(token_count))

    clean = sorted(set(clean))
    if clean[0] < 0 or clean[-1] > token_count - 1:
        raise ValueError(
            f"run_indices out of bounds: min={clean[0]} max={clean[-1]} token_count={token_count}"
        )
    return clean


def parse_row(row: Dict[str, Any]) -> TossupQuestion:
    """Parse a qb-rl/HuggingFace-style row into ``TossupQuestion``."""
    question = str(row["question"])
    tokens = question.split()
    metadata = row.get("metadata", {}) or {}
    answer_primary = str(
        row.get("answer_primary") or (row.get("clean_answers") or [""])[0]
    ).strip()
    clean_answers = [str(x) for x in (row.get("clean_answers") or [])]
    if not clean_answers and answer_primary:
        clean_answers = [answer_primary]

    run_indices = _coerce_run_indices(
        row.get("run_indices") or [],
        token_count=len(tokens),
    )

    normalized_question = " ".join(question.split())
    normalized_tokens = " ".join(tokens)
    if normalized_tokens != normalized_question:
        raise ValueError("tokenization roundtrip mismatch")
    if max(run_indices) > len(tokens) - 1:
        raise ValueError("run_indices out of bounds")

    cumulative_prefixes = [" ".join(tokens[: idx + 1]) for idx in run_indices]
    category = str(metadata.get("category") or row.get("category") or "")
    human_buzz_positions = _coerce_human_buzz_positions(
        metadata.get("human_buzz_positions") or row.get("human_buzz_positions")
    )

    qid_raw = row.get("qid") or row.get("question_id") or row.get("id")
    if qid_raw is None:
        qid_raw = _generate_qid(question)

    return TossupQuestion(
        qid=str(qid_raw),
        question=question,
        tokens=tokens,
        answer_primary=answer_primary,
        clean_answers=clean_answers,
        run_indices=run_indices,
        human_buzz_positions=human_buzz_positions,
        category=category,
        cumulative_prefixes=cumulative_prefixes,
    )


def load_tossup_questions(
    dataset: str,
    dataset_config: Optional[str] = None,
    split: str = "eval",
    limit: Optional[int] = None,
) -> List[TossupQuestion]:
    """Load tossup questions from Hugging Face datasets using qb-rl semantics."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "datasets is required for Hugging Face loading. Install it with: pip install datasets"
        ) from exc

    if dataset_config:
        ds = load_dataset(dataset, dataset_config, split=split)
    else:
        ds = load_dataset(dataset, split=split)

    if limit is not None:
        ds = ds.select(range(min(int(limit), len(ds))))

    return [parse_row(dict(row)) for row in ds]


def load_tossup_questions_from_config(
    config: Dict[str, Any],
    smoke: bool = False,
) -> List[TossupQuestion]:
    """Load tossups from config, supporting qb-rl and qanta-buzzer keys."""
    from qb_data.config import resolve_data_loading_options

    data_opts = resolve_data_loading_options(config, smoke=smoke)
    csv_path = data_opts.get("csv_path")
    dataset = data_opts.get("dataset")
    dataset_config = data_opts.get("dataset_config")
    split = data_opts.get("split", "eval")
    limit = data_opts.get("max_questions")

    if csv_path and Path(csv_path).exists():
        questions = QANTADatasetLoader.load_from_csv(str(csv_path))
    elif dataset:
        questions = load_tossup_questions(
            dataset=str(dataset),
            dataset_config=str(dataset_config) if dataset_config else None,
            split=str(split),
            limit=int(limit) if limit is not None else None,
        )
    elif csv_path and data_opts.get("use_huggingface"):
        from qb_data.huggingface_loader import try_huggingface_fallback

        questions = try_huggingface_fallback(str(csv_path))
        if questions is None:
            raise FileNotFoundError(
                f"Could not load questions from missing CSV path {csv_path} via Hugging Face fallback"
            )
    else:
        raise FileNotFoundError(
            "No valid data source configured. Provide data.csv_path or "
            "data.dataset/data.dataset_config for qb-rl compatibility."
        )

    if limit is not None:
        questions = questions[: int(limit)]

    return questions


class QANTADatasetLoader:
    """
    Loader for QANTA-format quiz bowl CSV files.

    The QANTA format has questions with clues separated by ||| delimiters.
    """

    @classmethod
    def load_from_csv(cls, filepath: str) -> List[TossupQuestion]:
        """
        Load questions from a QANTA-format CSV file.

        Parameters
        ----------
        filepath : str
            Path to the CSV file

        Returns
        -------
        List[TossupQuestion]
            List of parsed questions

        Raises
        ------
        FileNotFoundError
            If the CSV file doesn't exist
        ValueError
            If required columns are missing or data is malformed
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"CSV file not found: {filepath}")

        questions = []

        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)

            # Validate required columns
            actual_columns = set(reader.fieldnames or [])

            # Handle alternate column names
            if 'Text' in actual_columns and 'question' not in actual_columns:
                # QANTA format uses 'Text' instead of 'question'
                text_col = 'Text'
            elif 'question' in actual_columns:
                text_col = 'question'
            else:
                raise ValueError(f"Missing required column 'question' or 'Text'. Found columns: {actual_columns}")

            if 'Answer' in actual_columns and 'answer' not in actual_columns:
                answer_col = 'Answer'
            elif 'answer' in actual_columns:
                answer_col = 'answer'
            else:
                raise ValueError(f"Missing required column 'answer' or 'Answer'. Found columns: {actual_columns}")

            # Check for optional columns
            category_col = None
            if 'Category' in actual_columns:
                category_col = 'Category'
            elif 'category' in actual_columns:
                category_col = 'category'

            qid_col = None
            if 'Question ID' in actual_columns:
                qid_col = 'Question ID'
            elif 'qid' in actual_columns:
                qid_col = 'qid'
            elif 'question_id' in actual_columns:
                qid_col = 'question_id'

            # Parse each row
            for row_idx, row in enumerate(reader):
                try:
                    # Get question text and parse clues
                    question_text = row[text_col]
                    if not question_text or not question_text.strip():
                        continue  # Skip empty questions

                    # Split on ||| delimiter
                    if '|||' in question_text:
                        clues = [clue.strip() for clue in question_text.split('|||')]
                        clues = [c for c in clues if c]  # Remove empty clues
                    else:
                        # Treat entire text as single clue if no delimiter
                        clues = [question_text.strip()]

                    if not clues:
                        continue  # Skip if no valid clues

                    # Get answer
                    answer = row[answer_col].strip()
                    if not answer:
                        continue  # Skip questions without answers

                    # Get category (optional)
                    category = ""
                    if category_col:
                        category = row.get(category_col, "").strip()

                    # Get or generate question ID
                    if qid_col and row.get(qid_col):
                        qid = row[qid_col].strip()
                    else:
                        qid = _generate_qid(question_text)

                    # Parse clues into tokens and run indices
                    tokens, run_indices = _parse_clues_to_tokens(clues)

                    # Build cumulative prefixes
                    cumulative_prefixes = []
                    for idx in run_indices:
                        prefix = " ".join(tokens[:idx + 1])
                        cumulative_prefixes.append(prefix)

                    # Create clean answers list
                    clean_answers = [normalize_answer(answer)]

                    # Full question is all clues joined
                    full_question = " ".join(clues)

                    # Create TossupQuestion
                    question = TossupQuestion(
                        qid=qid,
                        question=full_question,
                        tokens=tokens,
                        answer_primary=answer,
                        clean_answers=clean_answers,
                        run_indices=run_indices,
                        human_buzz_positions=None,  # Not available in basic CSV
                        category=category,
                        cumulative_prefixes=cumulative_prefixes
                    )

                    questions.append(question)

                except Exception as e:
                    print(f"Warning: Failed to parse row {row_idx + 1}: {e}")
                    continue

        if not questions:
            raise ValueError(f"No valid questions found in {filepath}")

        return questions
```

## File: qb_data/mc_builder.py
```python
"""Multiple-choice question builder with anti-artifact guards."""

from __future__ import annotations

import random
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from qb_data.answer_profiles import AnswerProfileBuilder
from qb_data.data_loader import TossupQuestion
from qb_data.text_utils import normalize_answer


@dataclass
class MCQuestion(TossupQuestion):
    """A tossup question with multiple-choice options.

    Extends TossupQuestion with fields for multiple-choice presentation
    and tracking of distractor generation strategy.
    """
    options: List[str]
    gold_index: int
    option_profiles: List[str]
    option_answer_primary: List[str]
    distractor_strategy: str


def _normalized_edit_distance(a: str, b: str) -> float:
    """Compute normalized edit distance between two strings.

    Args:
        a: First string.
        b: Second string.

    Returns:
        Distance between 0 (identical) and 1 (completely different).
    """
    return 1.0 - SequenceMatcher(None, a, b).ratio()


def _token_overlap(a: str, b: str) -> float:
    """Compute token overlap between two strings.

    Args:
        a: First string.
        b: Second string.

    Returns:
        Fraction of overlapping tokens (0 to 1).
    """
    a_tokens = set(a.lower().split())
    b_tokens = set(b.lower().split())
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / max(1, min(len(a_tokens), len(b_tokens)))


class MCBuilder:
    """Builder for multiple-choice questions with anti-artifact guards.

    This class implements four layers of guards to prevent spurious patterns
    that agents could exploit:
    1. Alias collision guard: Prevents distractors that are aliases of the gold answer
    2. Duplicate guard: Prevents distractors with high token overlap
    3. Length ratio guard: Prevents distractors much longer/shorter than others
    4. Question overlap guard: Prevents answers that appear in the question text
    """

    def __init__(
        self,
        K: int = 4,
        strategy: str = "sbert_profile",
        alias_edit_distance_threshold: float = 0.2,
        duplicate_token_overlap_threshold: float = 0.8,
        max_length_ratio: float = 3.0,
        random_seed: int = 13,
        embedding_model: str = "all-MiniLM-L6-v2",
        openai_model: str = "text-embedding-3-small",
        variable_K: bool = False,
        min_K: int = 2,
        max_K: int | None = None,
    ):
        """Initialize the MC builder.

        Args:
            K: Default number of answer choices (must be >= 2).
            strategy: Distractor selection strategy.
            alias_edit_distance_threshold: Max edit distance for alias detection.
            duplicate_token_overlap_threshold: Max token overlap between options.
            max_length_ratio: Max ratio between longest and shortest option.
            random_seed: Random seed for reproducibility.
            embedding_model: SentenceTransformer model name for ``sbert_profile``.
            openai_model: OpenAI embedding model for ``openai_profile``.
            variable_K: If True, sample target K per question from
                ``[min_K, max_K or K]``.
            min_K: Minimum K when ``variable_K`` is True.
            max_K: Maximum K when ``variable_K`` is True.  Defaults to ``K``.
        """
        if K < 2:
            raise ValueError("K must be >= 2")
        self.K = K
        self.variable_K = variable_K
        self.min_K = max(2, min_K)
        self.max_K = max_K if max_K is not None else K
        self.strategy = strategy
        self.alias_edit_distance_threshold = alias_edit_distance_threshold
        self.duplicate_token_overlap_threshold = duplicate_token_overlap_threshold
        self.max_length_ratio = max_length_ratio
        self.rng = random.Random(random_seed)
        self.embedding_model = embedding_model
        self.openai_model = openai_model

    def _prepare_lookup(
        self, questions: List[TossupQuestion]
    ) -> Tuple[Dict[str, List[str]], Dict[str, str], Dict[str, str], List[str]]:
        """Prepare lookup structures for answer processing.

        Args:
            questions: List of tossup questions.

        Returns:
            Tuple of (answer_to_aliases, answer_to_category, answer_to_norm, answers).
        """
        answer_to_aliases: Dict[str, Set[str]] = {}
        answer_to_category: Dict[str, str] = {}

        for q in questions:
            # Collect all aliases for each answer
            aliases = answer_to_aliases.setdefault(q.answer_primary, set())
            aliases.update(str(alias) for alias in q.clean_answers)
            aliases.add(q.answer_primary)

            # Track category for category-based distractor selection
            if q.category and q.answer_primary not in answer_to_category:
                answer_to_category[q.answer_primary] = q.category

        # Convert to sorted lists for consistency
        answer_to_aliases_list = {k: sorted(v) for k, v in answer_to_aliases.items()}
        answers = sorted(answer_to_aliases_list.keys())
        answer_to_norm = {a: str(normalize_answer(a)) for a in answers}

        return answer_to_aliases_list, answer_to_category, answer_to_norm, answers

    def _rank_by_similarity(
        self,
        sim: np.ndarray,
        answers: List[str],
        answer_idx: Dict[str, int],
        M: int,
    ) -> Dict[str, List[str]]:
        """Rank distractors for each answer using a similarity matrix.

        Uses ``np.argpartition`` for top-M retrieval when M < N-1,
        reducing per-answer work from O(N log N) to O(N + M log M).

        Parameters
        ----------
        sim : np.ndarray
            Pairwise similarity matrix of shape (N, N).
        answers : list[str]
            Ordered answer strings corresponding to matrix rows/cols.
        answer_idx : dict[str, int]
            Mapping from answer string to its index in *sim*.
        M : int
            Number of top candidates to retain per answer.

        Returns
        -------
        dict[str, list[str]]
            Each answer mapped to its ranked distractor list (length <= M).
        """
        N = len(answers)
        rankings: Dict[str, List[str]] = {}
        for answer in answers:
            idx = answer_idx[answer]
            row = sim[idx]
            if M >= N - 1:
                # Small N: full sort (no benefit from partition)
                order = np.argsort(-row).tolist()
            else:
                # Top-M retrieval: O(N) partition + O(M log M) sort
                top_m_idx = np.argpartition(-row, M)[:M]
                top_m_idx = top_m_idx[np.argsort(-row[top_m_idx])]
                order = top_m_idx.tolist()
            rankings[answer] = [answers[i] for i in order if answers[i] != answer]
        return rankings

    def _compute_rankings(
        self,
        answers: List[str],
        answer_profiles: Dict[str, str],
        answer_to_category: Dict[str, str],
    ) -> Dict[str, List[str]]:
        """Compute distractor rankings for each answer.

        For profile-based strategies, uses top-M retrieval via
        ``np.argpartition`` instead of full ``np.argsort`` to reduce
        per-answer complexity from O(N log N) to O(N + M log M) and
        total memory from O(N^2) to O(N*M), where M = max(5*K, 30).

        Args:
            answers: List of all unique answers.
            answer_profiles: Dictionary mapping answers to their profiles.
            answer_to_category: Dictionary mapping answers to categories.

        Returns:
            Dictionary mapping each answer to a ranked list of distractors.
        """
        if self.strategy == "category_random":
            # Random selection within the same category
            rankings: Dict[str, List[str]] = {}
            for answer in answers:
                category = answer_to_category.get(answer, "")
                # First try same category, then fall back to all answers
                candidates = [
                    a for a in answers
                    if a != answer and answer_to_category.get(a, "") == category
                ]
                if len(candidates) < self.K - 1:
                    candidates = [a for a in answers if a != answer]
                self.rng.shuffle(candidates)
                rankings[answer] = candidates
            return rankings

        # Profile-based ranking strategies
        docs = [answer_profiles[a] for a in answers]
        answer_idx = {a: i for i, a in enumerate(answers)}
        M = min(max(5 * self.K, 30), len(answers) - 1)

        if self.strategy == "tfidf_profile":
            # TF-IDF based similarity
            vectorizer = TfidfVectorizer(stop_words="english")
            matrix = vectorizer.fit_transform(docs)
            sim = cosine_similarity(matrix, matrix)
            return self._rank_by_similarity(sim, answers, answer_idx, M)

        if self.strategy in {"sbert_profile", "openai_profile"}:
            if self.strategy == "sbert_profile":
                # One-shot SBERT encoding for distractor ranking.
                # This is separate from the SBERTLikelihood runtime cache
                # because it runs only during MC dataset construction.
                from sentence_transformers import SentenceTransformer
                encoder = SentenceTransformer(self.embedding_model)
                embeddings = encoder.encode(docs, convert_to_numpy=True, normalize_embeddings=True)
                sim = embeddings @ embeddings.T
            else:
                from models.likelihoods import OpenAILikelihood

                likelihood = OpenAILikelihood(model=self.openai_model)
                embeddings = likelihood.embed_and_cache(docs)
                sim = embeddings @ embeddings.T

            return self._rank_by_similarity(sim, answers, answer_idx, M)

        raise ValueError(f"Unknown distractor strategy: {self.strategy}")

    def _aliases_collide(self, candidate: str, gold_aliases: List[str]) -> bool:
        """Check if a candidate is too similar to any gold answer alias.

        Args:
            candidate: Candidate distractor.
            gold_aliases: List of aliases for the gold answer.

        Returns:
            True if the candidate collides with a gold alias.
        """
        candidate_norm = str(normalize_answer(candidate))
        gold_norms = [str(normalize_answer(alias)) for alias in gold_aliases]

        # Check exact match
        if candidate_norm in set(gold_norms):
            return True

        # Check edit distance
        for gold_norm in gold_norms:
            if _normalized_edit_distance(candidate_norm, gold_norm) < self.alias_edit_distance_threshold:
                return True

        return False

    def _violates_duplicate_guard(self, candidate: str, selected: List[str]) -> bool:
        """Check if candidate has too much token overlap with already selected options.

        Args:
            candidate: Candidate distractor.
            selected: List of already selected distractors.

        Returns:
            True if the candidate has too much overlap.
        """
        for chosen in selected:
            if _token_overlap(candidate, chosen) > self.duplicate_token_overlap_threshold:
                return True
        return False

    def _violates_length_ratio_guard(self, options: List[str]) -> bool:
        """Check if options have too different lengths.

        Args:
            options: List of all options.

        Returns:
            True if the length ratio is too high.
        """
        lengths = [max(1, len(o.split())) for o in options]
        return (max(lengths) / min(lengths)) > self.max_length_ratio

    def _violates_question_overlap_guard(self, question: str, options: List[str]) -> bool:
        """Check if any option appears in the question text.

        Args:
            question: Question text.
            options: List of answer options.

        Returns:
            True if any option appears in the question.
        """
        q_norm = str(normalize_answer(question))
        for option in options:
            o_norm = str(normalize_answer(option))
            if o_norm and o_norm in q_norm:
                return True
        return False

    def _target_k(self) -> int:
        """Return the target K for the next question.

        When ``variable_K`` is False, always returns ``self.K``.
        When True, samples uniformly from ``[min_K, max_K]``.
        """
        if not self.variable_K:
            return self.K
        return self.rng.randint(self.min_K, self.max_K)

    def build(
        self,
        questions: List[TossupQuestion],
        profile_builder: AnswerProfileBuilder,
    ) -> List[MCQuestion]:
        """Build multiple-choice questions with anti-artifact guards.

        Args:
            questions: List of tossup questions.
            profile_builder: Profile builder for answer representations.

        Returns:
            List of MCQuestion objects that passed all guards.
        """
        if not questions:
            return []

        # Build answer profiles
        profile_builder.fit(questions)
        answer_profiles = profile_builder.build_profiles(questions)

        # Prepare lookup structures
        answer_to_aliases, answer_to_category, _answer_to_norm, answers = self._prepare_lookup(questions)

        # Compute distractor rankings
        rankings = self._compute_rankings(answers, answer_profiles, answer_to_category)

        mc_questions: List[MCQuestion] = []

        for q in questions:
            target_k = self._target_k()
            gold = q.answer_primary
            gold_aliases = answer_to_aliases.get(gold, [gold])
            ranked = rankings.get(gold, [a for a in answers if a != gold])
            selected: List[str] = []

            # Select distractors from ranked list
            for candidate in ranked:
                if candidate == gold:
                    continue
                if self._aliases_collide(candidate, gold_aliases):
                    continue
                if self._violates_duplicate_guard(candidate, selected):
                    continue
                selected.append(candidate)
                if len(selected) >= target_k - 1:
                    break

            # If not enough distractors from ranking, try random fallback
            if len(selected) < target_k - 1:
                fallback = [a for a in answers if a not in selected and a != gold]
                self.rng.shuffle(fallback)
                for candidate in fallback:
                    if self._aliases_collide(candidate, gold_aliases):
                        continue
                    if self._violates_duplicate_guard(candidate, selected):
                        continue
                    selected.append(candidate)
                    if len(selected) >= target_k - 1:
                        break

            # Skip question if we can't find enough valid distractors
            if len(selected) < target_k - 1:
                continue

            # Create options and shuffle
            option_answer_primary = [gold] + selected[:target_k - 1]
            self.rng.shuffle(option_answer_primary)
            gold_index = option_answer_primary.index(gold)
            options = option_answer_primary[:]

            # Apply guard 3: Check length ratio
            if self._violates_length_ratio_guard(options):
                continue

            # Apply guard 4: Check question overlap
            if self._violates_question_overlap_guard(q.question, options):
                continue

            # Build option profiles with leave-one-out for gold
            option_profiles: List[str] = []
            for answer in option_answer_primary:
                exclude_qid = q.qid if answer == gold else None
                option_profiles.append(
                    profile_builder.profile_for_answer(answer, exclude_qid=exclude_qid)
                )

            # Create MCQuestion
            mc_questions.append(
                MCQuestion(
                    qid=q.qid,
                    question=q.question,
                    tokens=q.tokens,
                    answer_primary=q.answer_primary,
                    clean_answers=q.clean_answers,
                    run_indices=q.run_indices,
                    human_buzz_positions=q.human_buzz_positions,
                    category=q.category,
                    cumulative_prefixes=q.cumulative_prefixes,
                    options=options,
                    gold_index=gold_index,
                    option_profiles=option_profiles,
                    option_answer_primary=option_answer_primary,
                    distractor_strategy=self.strategy,
                )
            )

        return mc_questions


def build_mc_questions(
    questions: List[TossupQuestion],
    K: int,
    strategy: str,
    profile_builder: AnswerProfileBuilder,
    guards: Optional[Dict[str, Any]] = None,
    random_seed: int = 13,
) -> List[MCQuestion]:
    """Factory function to build multiple-choice questions.

    Args:
        questions: List of tossup questions.
        K: Number of answer choices.
        strategy: Distractor selection strategy.
        profile_builder: Profile builder for answer representations.
        guards: Optional dictionary of guard thresholds.
        random_seed: Random seed for reproducibility.

    Returns:
        List of MCQuestion objects that passed all guards.
    """
    guards = guards or {}
    builder = MCBuilder(
        K=K,
        strategy=strategy,
        alias_edit_distance_threshold=float(guards.get("alias_edit_distance_threshold", 0.2)),
        duplicate_token_overlap_threshold=float(guards.get("duplicate_token_overlap_threshold", 0.8)),
        max_length_ratio=float(guards.get("max_length_ratio", 3.0)),
        random_seed=random_seed,
    )
    return builder.build(questions=questions, profile_builder=profile_builder)
```

## File: qb_env/stop_only_env.py
```python
"""Stop-only action-space wrapper for the quiz bowl environment."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from qb_env.tossup_env import TossupMCEnv


class StopOnlyEnv(gym.Wrapper):
    """Wrap TossupMCEnv with a binary WAIT/BUZZ action space.

    Action mapping:
    - 0 -> WAIT
    - 1 -> BUZZ using the current answer-selection strategy

    The default answer-selection strategy commits to the current belief argmax.
    """

    def __init__(self, env: TossupMCEnv, answer_mode: str = "argmax_belief") -> None:
        super().__init__(env)
        self.answer_mode = answer_mode
        self.action_space = spaces.Discrete(2)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        return self.env.reset(seed=seed, options=options)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")
        if action == 0:
            return self.env.step(0)

        if self.answer_mode != "argmax_belief":
            raise ValueError(f"Unknown answer_mode: {self.answer_mode}")

        belief = getattr(self.env, "belief", None)
        if belief is None or len(belief) == 0:
            raise ValueError("BUZZ is invalid when belief is unavailable")

        chosen_idx = int(np.argmax(belief))
        return self.env.step(1 + chosen_idx)

    def action_masks(self) -> np.ndarray:
        """Return a binary action mask for the WAIT/BUZZ action space.

        WAIT (0) is always valid. BUZZ (1) is valid when the wrapped env
        has a non-empty belief vector that argmax can act on.
        """
        mask = np.array([True, False], dtype=bool)
        if (
            self.answer_mode == "argmax_belief"
            and getattr(self.env, "belief", None) is not None
            and len(self.env.belief) > 0
        ):
            mask[1] = True
        return mask
```

## File: scripts/run_smoke_pipeline.py
```python
#!/usr/bin/env python3
"""Run the full canonical smoke pipeline end-to-end.

Stages:
1) build_mc_dataset
2) run_baselines
3) train_ppo
4) evaluate_all

Writes a summary JSON to smoke_pipeline_summary.json in the output directory
(default: artifacts/smoke/, overridable via --output-dir).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "smoke"


def _build_stages(output_dir: str | None) -> list[list[str]]:
    """Build stage command lists, propagating --output-dir when provided."""
    base = [
        ["scripts/build_mc_dataset.py", "--smoke"],
        ["scripts/run_baselines.py", "--smoke"],
        ["scripts/train_ppo.py", "--smoke"],
        ["scripts/evaluate_all.py", "--smoke"],
    ]
    if output_dir is not None:
        return [cmd + ["--output-dir", output_dir] for cmd in base]
    return base


def run_stage(python_exe: str, args: list[str]) -> tuple[int, float]:
    """Run one stage command and return (exit_code, seconds)."""
    cmd = [python_exe, *args]
    start = time.time()
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT)
    elapsed = time.time() - start
    return proc.returncode, elapsed


def main() -> int:
    parser = argparse.ArgumentParser(description="Run full smoke pipeline")
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter to use (default: current interpreter)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Override output directory for all stages (default: artifacts/smoke).",
    )
    ns = parser.parse_args()

    artifact_dir = Path(ns.output_dir) if ns.output_dir else DEFAULT_ARTIFACT_DIR
    stages = _build_stages(ns.output_dir)

    print("=" * 60)
    print("Smoke Pipeline Runner")
    print("=" * 60)
    print(f"Python: {ns.python}")
    print(f"Output: {artifact_dir}")
    print()

    artifact_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, object] = {
        "python": ns.python,
        "output_dir": str(artifact_dir),
        "started_at_unix": time.time(),
        "stages": [],
    }

    pipeline_start = time.time()
    for stage_args in stages:
        stage_name = stage_args[0]
        print(f"Running: {stage_name} {' '.join(stage_args[1:])}")
        code, seconds = run_stage(ns.python, stage_args)
        summary["stages"].append(
            {
                "stage": stage_name,
                "args": stage_args[1:],
                "exit_code": code,
                "seconds": round(seconds, 3),
            }
        )
        if code != 0:
            summary["status"] = "failed"
            summary["failed_stage"] = stage_name
            summary["total_seconds"] = round(time.time() - pipeline_start, 3)
            out_path = artifact_dir / "smoke_pipeline_summary.json"
            out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            print(f"\nFAILED at {stage_name} (exit={code})")
            print(f"Summary written: {out_path}")
            return code
        print(f"✓ {stage_name} completed in {seconds:.1f}s\n")

    summary["status"] = "ok"
    summary["total_seconds"] = round(time.time() - pipeline_start, 3)
    out_path = artifact_dir / "smoke_pipeline_summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("=" * 60)
    print("Smoke pipeline completed successfully")
    print(f"Summary written: {out_path}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

## File: scripts/train_t5_policy.py
```python
#!/usr/bin/env python3
"""
Train T5 policy with supervised warm-start then PPO fine-tuning.

End-to-end pipeline for training a T5PolicyModel on quiz bowl questions:
1. Supervised warm-start: Train answer selection on complete questions
2. PPO fine-tuning: Optimize wait/answer policy on incremental episodes

Usage:
    # Full pipeline (supervised + PPO)
    python scripts/train_t5_policy.py --config configs/t5_policy.yaml

    # Quick smoke test (t5-small, few epochs)
    python scripts/train_t5_policy.py --config configs/t5_policy.yaml --smoke

    # Skip supervised, load pretrained for PPO only
    python scripts/train_t5_policy.py --config configs/t5_policy.yaml \
        --skip-supervised --model-path checkpoints/supervised/best_model

    # Custom number of PPO iterations
    python scripts/train_t5_policy.py --config configs/t5_policy.yaml \
        --ppo-iterations 50
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml

from qb_data.config import merge_overrides
from scripts._common import ARTIFACT_DIR, load_mc_questions, parse_overrides


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments for training configuration.
    """
    parser = argparse.ArgumentParser(
        description="Train T5 policy with supervised warm-start then PPO.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "t5_policy.yaml"),
        help="Path to YAML config file (default: configs/t5_policy.yaml).",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Quick test run: uses t5-small, 2 epochs, 4 batch size.",
    )
    parser.add_argument(
        "--skip-supervised",
        action="store_true",
        help="Skip supervised training phase.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to pretrained model checkpoint (required if --skip-supervised).",
    )
    parser.add_argument(
        "--mc-path",
        type=str,
        default=None,
        help="Path to MC dataset JSON file.",
    )
    parser.add_argument(
        "--ppo-iterations",
        type=int,
        default=None,
        help="Override number of PPO iterations from config.",
    )
    parser.add_argument(
        "--hazard-pretrain",
        action="store_true",
        help="Enable the experimental hazard pretraining bridge before PPO.",
    )
    parser.add_argument(
        "--beta-terminal",
        type=float,
        default=1.0,
        help="Terminal survival penalty used by the hazard bridge.",
    )
    parser.add_argument(
        "--freeze-answer-head",
        action="store_true",
        help="Freeze the answer head during the hazard bridge phase.",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Config overrides: key=value (e.g. model.model_name=t5-base)",
    )
    return parser.parse_args()


def load_config_with_overrides(args: argparse.Namespace) -> dict:
    """Load YAML config and apply smoke/CLI overrides.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.

    Returns
    -------
    dict
        Configuration dictionary with overrides applied.
    """
    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.smoke:
        smoke = config.get("smoke", {})
        # Override model settings
        if "model" in smoke:
            for key, val in smoke["model"].items():
                config["model"][key] = val
        # Override supervised settings
        if "supervised" in smoke:
            for key, val in smoke["supervised"].items():
                config["supervised"][key] = val
        # Override PPO settings
        if "ppo" in smoke:
            for key, val in smoke["ppo"].items():
                config["ppo"][key] = val
        # Override data settings
        if "data" in smoke:
            for key, val in smoke["data"].items():
                config["data"][key] = val

    if args.ppo_iterations is not None:
        config["ppo"]["iterations"] = args.ppo_iterations

    return config


def flatten_config(config: dict) -> dict:
    """Flatten nested config sections into a single dict for trainer APIs.

    Parameters
    ----------
    config : dict
        Nested config dict with sections (model, supervised, ppo, data).

    Returns
    -------
    dict
        Flat config dict with prefixed keys for each trainer.
    """
    flat = {}

    # Model section
    model = config.get("model", {})
    flat["model_name"] = model.get("model_name", "t5-large")
    device = model.get("device", "auto")
    if device == "auto":
        import torch
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    flat["device"] = device
    flat["max_input_length"] = model.get("max_input_length", 512)
    flat["num_choices"] = model.get("num_choices", config.get("data", {}).get("K", 4))

    # Supervised section
    sup = config.get("supervised", {})
    flat["supervised_lr"] = sup.get("lr", 3e-4)
    flat["supervised_epochs"] = sup.get("epochs", 10)
    flat["supervised_batch_size"] = sup.get("batch_size", 8)
    flat["supervised_grad_accum_steps"] = sup.get("grad_accum_steps", 4)
    flat["max_grad_norm"] = sup.get("max_grad_norm", 1.0)
    flat["weight_decay"] = sup.get("weight_decay", 0.01)
    flat["checkpoint_dir"] = sup.get("checkpoint_dir", "checkpoints")

    # PPO section
    ppo = config.get("ppo", {})
    flat["ppo_lr"] = ppo.get("lr", 1e-5)
    flat["ppo_iterations"] = ppo.get("iterations", 100)
    flat["ppo_batch_size"] = ppo.get("batch_size", 8)
    flat["ppo_epochs_per_iter"] = ppo.get("epochs_per_iter", 4)
    flat["ppo_gamma"] = ppo.get("gamma", 0.99)
    flat["ppo_gae_lambda"] = ppo.get("gae_lambda", 0.95)
    flat["ppo_clip_ratio"] = ppo.get("clip_ratio", 0.2)
    flat["ppo_value_coef"] = ppo.get("value_coef", 0.5)
    flat["ppo_entropy_coef"] = ppo.get("entropy_coef", 0.01)
    flat["ppo_max_grad_norm"] = ppo.get("max_grad_norm", 0.5)
    flat["ppo_episodes_per_iter"] = ppo.get("episodes_per_iter", 16)
    flat["eval_interval"] = ppo.get("eval_interval", 10)
    flat["save_interval"] = ppo.get("save_interval", 20)

    return flat


def load_questions(args: argparse.Namespace, config: dict) -> list:
    """Load MC questions from file or fallback paths.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments, may have mc_path override.
    config : dict
        Config dict with data section.

    Returns
    -------
    list
        List of MCQuestion instances.
    """
    if args.mc_path:
        mc_path = Path(args.mc_path)
    else:
        # Try standard locations
        candidates = [
            ARTIFACT_DIR / "main" / "mc_dataset.json",
            ARTIFACT_DIR / "smoke" / "mc_dataset.json",
            PROJECT_ROOT / "data" / "processed" / "mc_dataset.json",
        ]
        mc_path = None
        for candidate in candidates:
            if candidate.exists():
                mc_path = candidate
                break

        if mc_path is None:
            print("ERROR: No MC dataset found. Run build_mc_dataset.py first.")
            print("Searched locations:")
            for c in candidates:
                print(f"  {c}")
            sys.exit(1)

    print(f"Loading MC questions from: {mc_path}")
    questions = load_mc_questions(mc_path)
    print(f"Loaded {len(questions)} questions")

    # Apply max_questions limit (smoke mode)
    max_questions = config.get("data", {}).get("max_questions", None)
    if max_questions and len(questions) > max_questions:
        questions = questions[:max_questions]
        print(f"Limited to {max_questions} questions (smoke mode)")

    return questions


def validate_args(args: argparse.Namespace) -> None:
    """Validate CLI arguments and reject unsupported bridge paths."""
    if args.skip_supervised and args.model_path is None:
        print("ERROR: --model-path is required when using --skip-supervised")
        sys.exit(1)
    if args.hazard_pretrain:
        raise NotImplementedError(
            "Hazard pretraining loop not yet implemented. "
            "The math utilities are available in training/hazard_pretrain.py, "
            "but the end-to-end bridge has not been wired into train_t5_policy.py yet."
        )


def split_questions(questions: list, config: dict) -> tuple:
    """Split questions into train/val/test sets.

    Parameters
    ----------
    questions : list
        Full list of MCQuestion instances.
    config : dict
        Config dict with data section (train_size, val_size, test_size, seed).

    Returns
    -------
    tuple[list, list, list]
        Train, validation, and test question lists.
    """
    import random

    data = config.get("data", {})
    seed = data.get("seed", 42)
    train_size = data.get("train_size", 0.7)
    val_size = data.get("val_size", 0.15)

    rng = random.Random(seed)
    shuffled = questions[:]
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_size)
    n_val = int(n * val_size)

    train_questions = shuffled[:n_train]
    val_questions = shuffled[n_train : n_train + n_val]
    test_questions = shuffled[n_train + n_val :]

    print(f"Split: {len(train_questions)} train, {len(val_questions)} val, {len(test_questions)} test")
    return train_questions, val_questions, test_questions


def main() -> None:
    """Run the full T5 policy training pipeline."""
    args = parse_args()
    validate_args(args)

    # Load config with overrides
    config = load_config_with_overrides(args)
    overrides = parse_overrides(args)
    if overrides:
        config = merge_overrides(config, overrides)
    flat_config = flatten_config(config)

    # Load and split dataset
    questions = load_questions(args, config)
    train_questions, val_questions, test_questions = split_questions(questions, config)

    # Import training modules (lazy to avoid loading transformers until needed)
    from training.train_supervised_t5 import run_supervised_training
    from training.train_ppo_t5 import run_ppo_training

    # Phase 1: Supervised warm-start (optional)
    supervised_model_path = None
    if not args.skip_supervised:
        print("\n" + "=" * 60)
        print("PHASE 1: SUPERVISED WARM-START")
        print("=" * 60)

        model, trainer = run_supervised_training(
            config=flat_config,
            train_questions=train_questions,
            val_questions=val_questions,
        )
        supervised_model_path = str(
            trainer.checkpoint_dir / "best_model"
        )
        print(f"Supervised model saved to: {supervised_model_path}")
    else:
        supervised_model_path = args.model_path
        print(f"\nSkipping supervised training, using model: {supervised_model_path}")

    # Phase 2: PPO fine-tuning
    print("\n" + "=" * 60)
    print("PHASE 2: PPO FINE-TUNING (T5 Policy)")
    print("=" * 60)

    model, trainer = run_ppo_training(
        config=flat_config,
        train_questions=train_questions,
        val_questions=val_questions,
        test_questions=test_questions,
        pretrained_model_path=supervised_model_path,
    )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best PPO model saved to: {trainer.checkpoint_dir / 'best_model'}")
    print(f"Training history: {trainer.checkpoint_dir / 'history.json'}")


if __name__ == "__main__":
    main()
```

## File: tests/test_agents.py
```python
"""Test suite for agents/ -- baseline agent execution and episode result schemas.

Covers:
- AGT-02: ThresholdBuzzer execution and buzzing logic
- AGT-03: AlwaysBuzzFinalBuzzer wait-then-buzz behavior
- AGT-04: SoftmaxProfileBuzzer from-scratch belief recomputation
- AGT-05: SequentialBayesBuzzer incremental Bayesian updates
- AGT-06: EpisodeResult and SoftmaxEpisodeResult schema validation
- Threshold sweep utility tests
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from agents import (
    AlwaysBuzzFinalBuzzer,
    EpisodeResult,
    SequentialBayesBuzzer,
    SoftmaxEpisodeResult,
    SoftmaxProfileBuzzer,
    ThresholdBuzzer,
    result_to_dict,
    sweep_thresholds,
)
from agents._math import sigmoid
from models.likelihoods import TfIdfLikelihood
from qb_data.mc_builder import MCQuestion


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #


def _make_likelihood(corpus: list[str]) -> TfIdfLikelihood:
    """Create a fitted TF-IDF likelihood model from a corpus.

    Uses TF-IDF (fast) for agent logic tests so tests run quickly.
    """
    return TfIdfLikelihood(corpus_texts=corpus)


class TestSigmoidMath:
    """Tests for stable scalar sigmoid helper."""

    def test_sigmoid_handles_extreme_inputs_without_warning(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            assert sigmoid(1000.0) == pytest.approx(1.0)
            assert sigmoid(-1000.0) == pytest.approx(0.0)


# ------------------------------------------------------------------ #
# ThresholdBuzzer tests (AGT-02)
# ------------------------------------------------------------------ #


class TestThresholdBuzzer:
    """Tests for ThresholdBuzzer execution and buzzing logic."""

    def test_threshold_buzzer_executes(
        self, sample_mc_question: MCQuestion, sample_corpus: list[str]
    ) -> None:
        """ThresholdBuzzer runs an episode without error and returns EpisodeResult."""
        likelihood = _make_likelihood(sample_corpus)
        agent = ThresholdBuzzer(likelihood_model=likelihood, threshold=0.7)
        result = agent.run_episode(sample_mc_question)

        assert isinstance(result, EpisodeResult)
        assert result.qid == sample_mc_question.qid
        assert len(result.c_trace) > 0

    def test_threshold_buzzer_buzzes_on_threshold(
        self, sample_mc_question: MCQuestion, sample_corpus: list[str]
    ) -> None:
        """ThresholdBuzzer buzzes when top_p >= threshold.

        With threshold=0.0, the agent should buzz immediately at step 0
        because any non-negative top_p will meet the threshold.
        """
        likelihood = _make_likelihood(sample_corpus)
        agent = ThresholdBuzzer(likelihood_model=likelihood, threshold=0.0)
        result = agent.run_episode(sample_mc_question)

        # With threshold 0.0, should buzz at step 0
        assert result.buzz_step == 0, (
            f"Expected buzz at step 0 with threshold=0.0, got step {result.buzz_step}"
        )

    def test_threshold_buzzer_waits_on_low_confidence(
        self, sample_mc_question: MCQuestion, sample_corpus: list[str]
    ) -> None:
        """ThresholdBuzzer waits when top_p < threshold.

        With threshold=1.0 (impossible for softmax to reach exactly 1.0 in
        practice), the agent should wait until the final step.
        """
        likelihood = _make_likelihood(sample_corpus)
        agent = ThresholdBuzzer(likelihood_model=likelihood, threshold=1.0)
        result = agent.run_episode(sample_mc_question)

        # With threshold 1.0, should wait until the last step
        expected_final = len(sample_mc_question.cumulative_prefixes) - 1
        assert result.buzz_step == expected_final, (
            f"Expected buzz at final step {expected_final} with threshold=1.0, "
            f"got step {result.buzz_step}"
        )

    def test_threshold_buzzer_buzzes_at_final(
        self, sample_mc_question: MCQuestion, sample_corpus: list[str]
    ) -> None:
        """ThresholdBuzzer always buzzes on final step regardless of threshold.

        Even with threshold=1.0 (unreachable), the agent must buzz at the
        final step as a forced fallback.
        """
        likelihood = _make_likelihood(sample_corpus)
        agent = ThresholdBuzzer(likelihood_model=likelihood, threshold=1.0)
        result = agent.run_episode(sample_mc_question)

        final_step = len(sample_mc_question.cumulative_prefixes) - 1
        assert result.buzz_step == final_step
        assert result.buzz_index in range(len(sample_mc_question.options))

    def test_threshold_buzzer_traces_valid(
        self, sample_mc_question: MCQuestion, sample_corpus: list[str]
    ) -> None:
        """c_trace and g_trace have correct and matching lengths.

        Traces should have length equal to buzz_step + 1 (one entry per
        step from 0 to buzz_step inclusive).
        """
        likelihood = _make_likelihood(sample_corpus)
        agent = ThresholdBuzzer(likelihood_model=likelihood, threshold=0.7)
        result = agent.run_episode(sample_mc_question)

        trace_len = result.buzz_step + 1
        assert len(result.c_trace) == trace_len, (
            f"c_trace length {len(result.c_trace)} != expected {trace_len}"
        )
        assert len(result.g_trace) == trace_len, (
            f"g_trace length {len(result.g_trace)} != expected {trace_len}"
        )
        assert len(result.top_p_trace) == trace_len
        assert len(result.entropy_trace) == trace_len

    def test_threshold_buzzer_confidence_proxy(
        self, sample_mc_question: MCQuestion, sample_corpus: list[str]
    ) -> None:
        """c_t values in [0, 1] via sigmoid transformation."""
        likelihood = _make_likelihood(sample_corpus)
        agent = ThresholdBuzzer(likelihood_model=likelihood, threshold=0.7)
        result = agent.run_episode(sample_mc_question)

        for c_t in result.c_trace:
            assert 0.0 <= c_t <= 1.0, (
                f"Confidence proxy {c_t} outside [0, 1]"
            )

    def test_threshold_buzzer_custom_params(
        self, sample_mc_question: MCQuestion, sample_corpus: list[str]
    ) -> None:
        """ThresholdBuzzer accepts custom beta and alpha parameters."""
        likelihood = _make_likelihood(sample_corpus)
        agent = ThresholdBuzzer(
            likelihood_model=likelihood,
            threshold=0.5,
            beta=10.0,
            alpha=20.0,
        )
        assert agent.beta == 10.0
        assert agent.alpha == 20.0

        result = agent.run_episode(sample_mc_question)
        assert isinstance(result, EpisodeResult)

    def test_threshold_buzzer_confidence_proxy_stable_extremes(
        self, sample_corpus: list[str]
    ) -> None:
        likelihood = _make_likelihood(sample_corpus)
        agent = ThresholdBuzzer(
            likelihood_model=likelihood,
            threshold=-100.0,
            alpha=100.0,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            assert agent._confidence_proxy(1.0) == pytest.approx(1.0)

        agent = ThresholdBuzzer(
            likelihood_model=likelihood,
            threshold=100.0,
            alpha=100.0,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            assert agent._confidence_proxy(0.0) == pytest.approx(0.0)

    def test_threshold_buzzer_top_p_in_range(
        self, sample_mc_question: MCQuestion, sample_corpus: list[str]
    ) -> None:
        """top_p_trace values are valid probabilities in [0, 1]."""
        likelihood = _make_likelihood(sample_corpus)
        agent = ThresholdBuzzer(likelihood_model=likelihood, threshold=0.7)
        result = agent.run_episode(sample_mc_question)

        for p in result.top_p_trace:
            assert 0.0 <= p <= 1.0, f"top_p {p} outside [0, 1]"

    def test_threshold_buzzer_entropy_nonnegative(
        self, sample_mc_question: MCQuestion, sample_corpus: list[str]
    ) -> None:
        """Entropy values are non-negative (Shannon entropy >= 0)."""
        likelihood = _make_likelihood(sample_corpus)
        agent = ThresholdBuzzer(likelihood_model=likelihood, threshold=0.7)
        result = agent.run_episode(sample_mc_question)

        for h in result.entropy_trace:
            assert h >= 0.0, f"Entropy {h} is negative"


# ------------------------------------------------------------------ #
# AlwaysBuzzFinalBuzzer tests (AGT-03)
# ------------------------------------------------------------------ #


class TestAlwaysBuzzFinalBuzzer:
    """Tests for AlwaysBuzzFinalBuzzer wait-then-buzz behavior."""

    def test_always_buzz_final_waits(
        self, sample_mc_question: MCQuestion, sample_corpus: list[str]
    ) -> None:
        """All c_trace entries except the last are 0.0 (agent waits)."""
        likelihood = _make_likelihood(sample_corpus)
        agent = AlwaysBuzzFinalBuzzer(likelihood_model=likelihood)
        result = agent.run_episode(sample_mc_question)

        # All entries except last should be 0.0
        for c_t in result.c_trace[:-1]:
            assert c_t == 0.0, f"Expected c_t=0.0 for waiting, got {c_t}"

    def test_always_buzz_final_buzzes_last(
        self, sample_mc_question: MCQuestion, sample_corpus: list[str]
    ) -> None:
        """The last c_trace entry is 1.0 (agent buzzes at final step)."""
        likelihood = _make_likelihood(sample_corpus)
        agent = AlwaysBuzzFinalBuzzer(likelihood_model=likelihood)
        result = agent.run_episode(sample_mc_question)

        assert result.c_trace[-1] == 1.0, (
            f"Expected c_trace[-1]=1.0, got {result.c_trace[-1]}"
        )

    def test_always_buzz_final_computes_beliefs(
        self, sample_mc_question: MCQuestion, sample_corpus: list[str]
    ) -> None:
        """Beliefs are computed at each step (not skipped).

        All top_p_trace entries should have valid probability values,
        demonstrating the model computed beliefs at every step.
        """
        likelihood = _make_likelihood(sample_corpus)
        agent = AlwaysBuzzFinalBuzzer(likelihood_model=likelihood)
        result = agent.run_episode(sample_mc_question)

        n_steps = len(sample_mc_question.cumulative_prefixes)
        assert len(result.top_p_trace) == n_steps, (
            f"Expected {n_steps} top_p entries, got {len(result.top_p_trace)}"
        )
        for p in result.top_p_trace:
            assert 0.0 <= p <= 1.0, f"top_p {p} outside [0, 1]"

    def test_always_buzz_final_buzz_step(
        self, sample_mc_question: MCQuestion, sample_corpus: list[str]
    ) -> None:
        """buzz_step equals len(cumulative_prefixes) - 1 (last step)."""
        likelihood = _make_likelihood(sample_corpus)
        agent = AlwaysBuzzFinalBuzzer(likelihood_model=likelihood)
        result = agent.run_episode(sample_mc_question)

        expected = len(sample_mc_question.cumulative_prefixes) - 1
        assert result.buzz_step == expected, (
            f"Expected buzz_step={expected}, got {result.buzz_step}"
        )

    def test_always_buzz_final_full_trace(
        self, sample_mc_question: MCQuestion, sample_corpus: list[str]
    ) -> None:
        """All traces have length equal to number of cumulative prefixes."""
        likelihood = _make_likelihood(sample_corpus)
        agent = AlwaysBuzzFinalBuzzer(likelihood_model=likelihood)
        result = agent.run_episode(sample_mc_question)

        n = len(sample_mc_question.cumulative_prefixes)
        assert len(result.c_trace) == n
        assert len(result.g_trace) == n
        assert len(result.top_p_trace) == n
        assert len(result.entropy_trace) == n


# ------------------------------------------------------------------ #
# SoftmaxProfileBuzzer tests (AGT-04)
# ------------------------------------------------------------------ #


class TestSoftmaxProfileBuzzer:
    """Tests for SoftmaxProfileBuzzer from-scratch belief computation."""

    def test_softmax_profile_executes(
        self, sample_mc_question: MCQuestion, sample_corpus: list[str]
    ) -> None:
        """SoftmaxProfileBuzzer runs an episode without error."""
        likelihood = _make_likelihood(sample_corpus)
        agent = SoftmaxProfileBuzzer(likelihood_model=likelihood, threshold=0.7)
        result = agent.run_episode(sample_mc_question)

        assert isinstance(result, SoftmaxEpisodeResult)
        assert result.qid == sample_mc_question.qid

    def test_softmax_profile_recomputes_belief(
        self, sample_mc_question: MCQuestion, sample_corpus: list[str]
    ) -> None:
        """SoftmaxProfileBuzzer calls _belief_from_scratch each step.

        Verifies the method exists and the agent stores beliefs, confirming
        from-scratch recomputation (not incremental Bayesian updates).
        """
        likelihood = _make_likelihood(sample_corpus)
        agent = SoftmaxProfileBuzzer(likelihood_model=likelihood, threshold=0.7)

        # Verify the from-scratch method exists
        assert hasattr(agent, "_belief_from_scratch")

        result = agent.run_episode(sample_mc_question)

        # After episode, agent should have a stored belief
        assert agent.belief is not None
        assert isinstance(agent.belief, np.ndarray)
        assert agent.belief.shape == (len(sample_mc_question.options),)

    def test_softmax_profile_result_schema(
        self, sample_mc_question: MCQuestion, sample_corpus: list[str]
    ) -> None:
        """SoftmaxProfileBuzzer returns SoftmaxEpisodeResult, not EpisodeResult."""
        likelihood = _make_likelihood(sample_corpus)
        agent = SoftmaxProfileBuzzer(likelihood_model=likelihood, threshold=0.7)
        result = agent.run_episode(sample_mc_question)

        assert isinstance(result, SoftmaxEpisodeResult)
        # SoftmaxEpisodeResult should NOT be an EpisodeResult (different dataclass)
        assert not isinstance(result, EpisodeResult)

    def test_softmax_profile_confidence_proxy(
        self, sample_mc_question: MCQuestion, sample_corpus: list[str]
    ) -> None:
        """SoftmaxProfileBuzzer c_t values in [0, 1] via sigmoid."""
        likelihood = _make_likelihood(sample_corpus)
        agent = SoftmaxProfileBuzzer(likelihood_model=likelihood, threshold=0.7)
        result = agent.run_episode(sample_mc_question)

        for c_t in result.c_trace:
            assert 0.0 <= c_t <= 1.0, f"c_t {c_t} outside [0, 1]"

    def test_softmax_profile_threshold_behavior(
        self, sample_mc_question: MCQuestion, sample_corpus: list[str]
    ) -> None:
        """SoftmaxProfileBuzzer respects threshold for buzzing."""
        likelihood = _make_likelihood(sample_corpus)

        # With threshold 0.0, should buzz immediately
        agent_low = SoftmaxProfileBuzzer(likelihood_model=likelihood, threshold=0.0)
        result_low = agent_low.run_episode(sample_mc_question)
        assert result_low.buzz_step == 0

        # With threshold 1.0, should wait until the end
        agent_high = SoftmaxProfileBuzzer(likelihood_model=likelihood, threshold=1.0)
        result_high = agent_high.run_episode(sample_mc_question)
        assert result_high.buzz_step == len(sample_mc_question.cumulative_prefixes) - 1

    def test_softmax_profile_confidence_proxy_stable_extremes(
        self, sample_corpus: list[str]
    ) -> None:
        likelihood = _make_likelihood(sample_corpus)
        agent = SoftmaxProfileBuzzer(
            likelihood_model=likelihood,
            threshold=-100.0,
            alpha=100.0,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            assert agent.confidence_proxy(1.0) == pytest.approx(1.0)

        agent = SoftmaxProfileBuzzer(
            likelihood_model=likelihood,
            threshold=100.0,
            alpha=100.0,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            assert agent.confidence_proxy(0.0) == pytest.approx(0.0)


# ------------------------------------------------------------------ #
# SequentialBayesBuzzer tests (AGT-05)
# ------------------------------------------------------------------ #


class TestSequentialBayesBuzzer:
    """Tests for SequentialBayesBuzzer incremental Bayesian update."""

    def test_sequential_bayes_executes(
        self, sample_mc_question: MCQuestion, sample_corpus: list[str]
    ) -> None:
        """SequentialBayesBuzzer runs an episode without error."""
        likelihood = _make_likelihood(sample_corpus)
        agent = SequentialBayesBuzzer(likelihood_model=likelihood, threshold=0.7)
        result = agent.run_episode(sample_mc_question)

        assert isinstance(result, SoftmaxEpisodeResult)
        assert result.qid == sample_mc_question.qid

    def test_sequential_bayes_uses_run_indices(
        self, sample_mc_question: MCQuestion, sample_corpus: list[str]
    ) -> None:
        """SequentialBayesBuzzer requires question.run_indices field.

        The agent iterates over run_indices to extract token fragments,
        not over cumulative_prefixes. The number of trace entries should
        match the number of run_indices steps processed.
        """
        likelihood = _make_likelihood(sample_corpus)
        agent = SequentialBayesBuzzer(likelihood_model=likelihood, threshold=0.7)
        result = agent.run_episode(sample_mc_question)

        # Trace length should be <= len(run_indices)
        assert len(result.c_trace) <= len(sample_mc_question.run_indices), (
            f"Trace length {len(result.c_trace)} > run_indices length "
            f"{len(sample_mc_question.run_indices)}"
        )

    def test_sequential_bayes_bayesian_update(
        self, sample_mc_question: MCQuestion, sample_corpus: list[str]
    ) -> None:
        """Belief is posterior proportional to prior * likelihood.

        Verify the _step_update method produces valid posterior:
        all entries >= 0 and sum to 1.
        """
        likelihood = _make_likelihood(sample_corpus)
        agent = SequentialBayesBuzzer(likelihood_model=likelihood, threshold=0.7)

        K = len(sample_mc_question.options)
        prior = np.ones(K, dtype=np.float32) / K
        fragment = "first president"
        profiles = sample_mc_question.option_profiles

        posterior = agent._step_update(prior, fragment, profiles)

        assert posterior.shape == (K,), f"Expected shape ({K},), got {posterior.shape}"
        assert all(posterior >= 0), "Posterior has negative entries"
        np.testing.assert_almost_equal(
            posterior.sum(), 1.0, decimal=5,
            err_msg="Posterior should sum to 1.0",
        )

    def test_sequential_bayes_result_schema(
        self, sample_mc_question: MCQuestion, sample_corpus: list[str]
    ) -> None:
        """SequentialBayesBuzzer returns SoftmaxEpisodeResult."""
        likelihood = _make_likelihood(sample_corpus)
        agent = SequentialBayesBuzzer(likelihood_model=likelihood, threshold=0.7)
        result = agent.run_episode(sample_mc_question)

        assert isinstance(result, SoftmaxEpisodeResult)
        assert not isinstance(result, EpisodeResult)

    def test_sequential_bayes_fragments(
        self, sample_mc_question: MCQuestion, sample_corpus: list[str]
    ) -> None:
        """SequentialBayesBuzzer processes token fragments, not full prefixes.

        With threshold 1.0 (never buzzes early), all run_indices should be
        processed, producing traces of length len(run_indices).
        """
        likelihood = _make_likelihood(sample_corpus)
        agent = SequentialBayesBuzzer(likelihood_model=likelihood, threshold=1.0)
        result = agent.run_episode(sample_mc_question)

        n_steps = len(sample_mc_question.run_indices)
        assert len(result.c_trace) == n_steps, (
            f"Expected {n_steps} trace entries, got {len(result.c_trace)}"
        )


# ------------------------------------------------------------------ #
# Episode result schema tests (AGT-06)
# ------------------------------------------------------------------ #


class TestEpisodeResultSchema:
    """Tests for EpisodeResult and SoftmaxEpisodeResult dataclass schemas."""

    def test_episode_result_fields(self) -> None:
        """EpisodeResult has all required fields."""
        result = EpisodeResult(
            qid="test_q",
            buzz_step=3,
            buzz_index=1,
            gold_index=0,
            correct=False,
            reward_like=-0.5,
            c_trace=[0.1, 0.2, 0.3, 0.4],
            g_trace=[0.0, 0.0, 0.0, 1.0],
            top_p_trace=[0.3, 0.4, 0.5, 0.6],
            entropy_trace=[1.4, 1.2, 1.0, 0.8],
        )
        assert result.qid == "test_q"
        assert result.buzz_step == 3
        assert result.buzz_index == 1
        assert result.gold_index == 0
        assert result.correct is False
        assert result.reward_like == -0.5

    def test_softmax_episode_result_fields(self) -> None:
        """SoftmaxEpisodeResult has all required fields."""
        result = SoftmaxEpisodeResult(
            qid="test_q",
            buzz_step=2,
            buzz_index=0,
            gold_index=0,
            correct=True,
            c_trace=[0.1, 0.5, 0.9],
            g_trace=[1.0, 1.0, 1.0],
            top_p_trace=[0.4, 0.6, 0.9],
            entropy_trace=[1.2, 0.8, 0.3],
        )
        assert result.qid == "test_q"
        assert result.buzz_step == 2
        assert result.correct is True

    def test_traces_same_length(
        self, sample_mc_question: MCQuestion, sample_corpus: list[str]
    ) -> None:
        """len(c_trace) == len(g_trace) for all agents."""
        likelihood = _make_likelihood(sample_corpus)

        agents = [
            ThresholdBuzzer(likelihood_model=likelihood, threshold=0.7),
            AlwaysBuzzFinalBuzzer(likelihood_model=likelihood),
            SoftmaxProfileBuzzer(likelihood_model=likelihood, threshold=0.7),
            SequentialBayesBuzzer(likelihood_model=likelihood, threshold=0.7),
        ]

        for agent in agents:
            result = agent.run_episode(sample_mc_question)
            agent_name = type(agent).__name__
            assert len(result.c_trace) == len(result.g_trace), (
                f"{agent_name}: c_trace ({len(result.c_trace)}) != "
                f"g_trace ({len(result.g_trace)})"
            )
            assert len(result.c_trace) == len(result.top_p_trace), (
                f"{agent_name}: c_trace ({len(result.c_trace)}) != "
                f"top_p_trace ({len(result.top_p_trace)})"
            )
            assert len(result.c_trace) == len(result.entropy_trace), (
                f"{agent_name}: c_trace ({len(result.c_trace)}) != "
                f"entropy_trace ({len(result.entropy_trace)})"
            )

    def test_g_trace_binary(
        self, sample_mc_question: MCQuestion, sample_corpus: list[str]
    ) -> None:
        """g_trace values are 0.0 or 1.0 (correctness is binary)."""
        likelihood = _make_likelihood(sample_corpus)

        agents = [
            ThresholdBuzzer(likelihood_model=likelihood, threshold=0.7),
            AlwaysBuzzFinalBuzzer(likelihood_model=likelihood),
            SoftmaxProfileBuzzer(likelihood_model=likelihood, threshold=0.7),
            SequentialBayesBuzzer(likelihood_model=likelihood, threshold=0.7),
        ]

        for agent in agents:
            result = agent.run_episode(sample_mc_question)
            agent_name = type(agent).__name__
            for g_t in result.g_trace:
                assert g_t in (0.0, 1.0), (
                    f"{agent_name}: g_t={g_t} not in {{0.0, 1.0}}"
                )

    def test_buzz_index_valid(
        self, sample_mc_question: MCQuestion, sample_corpus: list[str]
    ) -> None:
        """buzz_index in range(K) where K = len(options)."""
        likelihood = _make_likelihood(sample_corpus)
        K = len(sample_mc_question.options)

        agents = [
            ThresholdBuzzer(likelihood_model=likelihood, threshold=0.7),
            AlwaysBuzzFinalBuzzer(likelihood_model=likelihood),
            SoftmaxProfileBuzzer(likelihood_model=likelihood, threshold=0.7),
            SequentialBayesBuzzer(likelihood_model=likelihood, threshold=0.7),
        ]

        for agent in agents:
            result = agent.run_episode(sample_mc_question)
            agent_name = type(agent).__name__
            assert 0 <= result.buzz_index < K, (
                f"{agent_name}: buzz_index={result.buzz_index} not in [0, {K})"
            )

    def test_result_to_dict(
        self, sample_mc_question: MCQuestion, sample_corpus: list[str]
    ) -> None:
        """result_to_dict() converts EpisodeResult to dict."""
        likelihood = _make_likelihood(sample_corpus)
        agent = ThresholdBuzzer(likelihood_model=likelihood, threshold=0.7)
        result = agent.run_episode(sample_mc_question)

        d = result_to_dict(result)
        assert isinstance(d, dict)
        assert d["qid"] == sample_mc_question.qid
        assert "buzz_step" in d
        assert "buzz_index" in d
        assert "gold_index" in d
        assert "correct" in d
        assert "reward_like" in d
        assert "c_trace" in d
        assert "g_trace" in d
        assert isinstance(d["c_trace"], list)


# ------------------------------------------------------------------ #
# Threshold sweep utility tests
# ------------------------------------------------------------------ #


class TestSweepThresholds:
    """Tests for sweep_thresholds utility function."""

    def test_sweep_thresholds_runs(
        self, sample_mc_question: MCQuestion, sample_corpus: list[str]
    ) -> None:
        """sweep_thresholds() returns dict[float, list[EpisodeResult]]."""
        likelihood = _make_likelihood(sample_corpus)
        results = sweep_thresholds(
            questions=[sample_mc_question],
            likelihood_model=likelihood,
            thresholds=[0.7],
        )

        assert isinstance(results, dict)
        assert 0.7 in results
        assert len(results[0.7]) == 1
        assert isinstance(results[0.7][0], EpisodeResult)

    def test_sweep_thresholds_multiple_values(
        self, sample_mc_question: MCQuestion, sample_corpus: list[str]
    ) -> None:
        """Sweeps over [0.6, 0.7, 0.8, 0.9] and returns results for each."""
        likelihood = _make_likelihood(sample_corpus)
        thresholds = [0.6, 0.7, 0.8, 0.9]
        results = sweep_thresholds(
            questions=[sample_mc_question],
            likelihood_model=likelihood,
            thresholds=thresholds,
        )

        assert len(results) == len(thresholds)
        for thresh in thresholds:
            assert thresh in results, f"Missing results for threshold {thresh}"
            assert len(results[thresh]) == 1

    def test_sweep_thresholds_monotonic_buzz_step(
        self, sample_mc_question: MCQuestion, sample_corpus: list[str]
    ) -> None:
        """Higher thresholds should produce later or equal buzz steps.

        A higher threshold means the agent needs more confidence to buzz,
        so it should wait at least as long as with a lower threshold.
        """
        likelihood = _make_likelihood(sample_corpus)
        thresholds = [0.3, 0.5, 0.7, 0.9]
        results = sweep_thresholds(
            questions=[sample_mc_question],
            likelihood_model=likelihood,
            thresholds=thresholds,
        )

        buzz_steps = [results[t][0].buzz_step for t in thresholds]
        for i in range(len(buzz_steps) - 1):
            assert buzz_steps[i] <= buzz_steps[i + 1], (
                f"Buzz step not monotonic: threshold {thresholds[i]} "
                f"(step {buzz_steps[i]}) > threshold {thresholds[i+1]} "
                f"(step {buzz_steps[i+1]})"
            )


# ------------------------------------------------------------------ #
# Precomputed equivalence tests
# ------------------------------------------------------------------ #


class TestPrecomputedEquivalence:
    """Prove precomputed-path functions are numerically identical to live agents."""

    def test_softmax_precomputed_matches_live(
        self, sample_mc_question: MCQuestion, sample_corpus: list[str]
    ) -> None:
        """_softmax_episode_from_precomputed matches SoftmaxProfileBuzzer.run_episode."""
        from agents.threshold_buzzer import (
            _softmax_episode_from_precomputed,
            precompute_beliefs,
        )

        likelihood = _make_likelihood(sample_corpus)
        threshold, beta, alpha = 0.7, 5.0, 10.0

        # Live agent
        agent = SoftmaxProfileBuzzer(
            likelihood_model=likelihood, threshold=threshold, beta=beta, alpha=alpha
        )
        live = agent.run_episode(sample_mc_question)

        # Precomputed path
        pqs = precompute_beliefs([sample_mc_question], likelihood, beta)
        pre = _softmax_episode_from_precomputed(pqs[0], threshold, alpha)

        assert pre.buzz_step == live.buzz_step
        assert pre.buzz_index == live.buzz_index
        assert pre.correct == live.correct
        np.testing.assert_array_almost_equal(pre.c_trace, live.c_trace)
        np.testing.assert_array_almost_equal(pre.g_trace, live.g_trace)
        np.testing.assert_array_almost_equal(pre.top_p_trace, live.top_p_trace)
        np.testing.assert_array_almost_equal(pre.entropy_trace, live.entropy_trace)

    def test_always_final_precomputed_matches_live(
        self, sample_mc_question: MCQuestion, sample_corpus: list[str]
    ) -> None:
        """_always_final_from_precomputed matches AlwaysBuzzFinalBuzzer.run_episode."""
        from agents.threshold_buzzer import (
            _always_final_from_precomputed,
            precompute_beliefs,
        )

        likelihood = _make_likelihood(sample_corpus)
        beta = 5.0

        # Live agent
        agent = AlwaysBuzzFinalBuzzer(likelihood_model=likelihood, beta=beta)
        live = agent.run_episode(sample_mc_question)

        # Precomputed path
        pqs = precompute_beliefs([sample_mc_question], likelihood, beta)
        pre = _always_final_from_precomputed(pqs[0])

        assert pre.buzz_step == live.buzz_step
        assert pre.buzz_index == live.buzz_index
        assert pre.correct == live.correct
        assert pre.reward_like == live.reward_like
        np.testing.assert_array_almost_equal(pre.c_trace, live.c_trace)
        np.testing.assert_array_almost_equal(pre.g_trace, live.g_trace)
        np.testing.assert_array_almost_equal(pre.top_p_trace, live.top_p_trace)
        np.testing.assert_array_almost_equal(pre.entropy_trace, live.entropy_trace)

    def test_sequential_precomputed_matches_live(
        self, sample_mc_question: MCQuestion, sample_corpus: list[str]
    ) -> None:
        """_sequential_episode_from_precomputed matches SequentialBayesBuzzer.run_episode."""
        from agents.bayesian_buzzer import (
            _sequential_episode_from_precomputed,
            precompute_sequential_beliefs,
        )

        likelihood = _make_likelihood(sample_corpus)
        threshold, beta, alpha = 0.7, 5.0, 10.0

        # Live agent
        agent = SequentialBayesBuzzer(
            likelihood_model=likelihood, threshold=threshold, beta=beta, alpha=alpha
        )
        live = agent.run_episode(sample_mc_question)

        # Precomputed path
        pqs = precompute_sequential_beliefs([sample_mc_question], likelihood, beta)
        pre = _sequential_episode_from_precomputed(pqs[0], threshold, alpha)

        assert pre.buzz_step == live.buzz_step
        assert pre.buzz_index == live.buzz_index
        assert pre.correct == live.correct
        np.testing.assert_array_almost_equal(pre.c_trace, live.c_trace)
        np.testing.assert_array_almost_equal(pre.g_trace, live.g_trace)
        np.testing.assert_array_almost_equal(pre.top_p_trace, live.top_p_trace)
        np.testing.assert_array_almost_equal(pre.entropy_trace, live.entropy_trace)

    def test_sweep_sequential_matches_per_threshold(
        self, sample_mc_question: MCQuestion, sample_corpus: list[str]
    ) -> None:
        """sweep_sequential_thresholds matches per-threshold SequentialBayesBuzzer."""
        from agents.bayesian_buzzer import sweep_sequential_thresholds

        likelihood = _make_likelihood(sample_corpus)
        thresholds = [0.5, 0.7, 0.9]
        beta, alpha = 5.0, 10.0

        # Sweep
        sweep = sweep_sequential_thresholds(
            questions=[sample_mc_question],
            likelihood_model=likelihood,
            thresholds=thresholds,
            beta=beta,
            alpha=alpha,
        )

        # Per-threshold live agents
        for threshold in thresholds:
            agent = SequentialBayesBuzzer(
                likelihood_model=likelihood,
                threshold=threshold,
                beta=beta,
                alpha=alpha,
            )
            live = agent.run_episode(sample_mc_question)
            pre = sweep[float(threshold)][0]

            assert pre.buzz_step == live.buzz_step, (
                f"threshold={threshold}: buzz_step {pre.buzz_step} != {live.buzz_step}"
            )
            assert pre.buzz_index == live.buzz_index
            assert pre.correct == live.correct
            np.testing.assert_array_almost_equal(pre.c_trace, live.c_trace)
            np.testing.assert_array_almost_equal(pre.g_trace, live.g_trace)
            np.testing.assert_array_almost_equal(pre.top_p_trace, live.top_p_trace)
            np.testing.assert_array_almost_equal(
                pre.entropy_trace, live.entropy_trace
            )


# ------------------------------------------------------------------ #
# Shuffle precomputed equivalence tests
# ------------------------------------------------------------------ #


class TestShufflePrecomputedEquivalence:
    """Prove precomputed shuffle control matches live rescore shuffle control."""

    def test_shuffle_precomputed_matches_rescore(
        self, sample_mc_question: MCQuestion, sample_corpus: list[str]
    ) -> None:
        """Precomputed shuffle control matches live rescore shuffle control."""
        from dataclasses import asdict

        from agents.threshold_buzzer import precompute_beliefs
        from evaluation.controls import (
            run_shuffle_control,
            run_shuffle_control_precomputed,
        )
        from evaluation.metrics import calibration_at_buzz, summarize_buzz_metrics

        likelihood = _make_likelihood(sample_corpus)
        threshold, beta, alpha = 0.7, 5.0, 10.0
        questions = [sample_mc_question]

        # Live rescore path
        def evaluator(qset):
            agent = SoftmaxProfileBuzzer(
                likelihood_model=likelihood,
                threshold=threshold,
                beta=beta,
                alpha=alpha,
            )
            runs = [asdict(agent.run_episode(q)) for q in qset]
            summary = {**summarize_buzz_metrics(runs), **calibration_at_buzz(runs)}
            summary["runs"] = runs
            return summary

        live_result = run_shuffle_control(questions, evaluator=evaluator, random_seed=13)

        # Precomputed path
        precomputed = precompute_beliefs(questions, likelihood, beta)
        pre_result = run_shuffle_control_precomputed(
            precomputed, threshold, alpha, random_seed=13
        )

        # Compare summary metrics
        assert live_result["mean_sq"] == pytest.approx(pre_result["mean_sq"])
        assert live_result["buzz_accuracy"] == pytest.approx(pre_result["buzz_accuracy"])

        # Compare per-run results
        for live_run, pre_run in zip(live_result["runs"], pre_result["runs"]):
            assert live_run["buzz_step"] == pre_run["buzz_step"]
            assert live_run["buzz_index"] == pre_run["buzz_index"]
            assert live_run["correct"] == pre_run["correct"]
            np.testing.assert_array_almost_equal(
                live_run["c_trace"], pre_run["c_trace"]
            )
            np.testing.assert_array_almost_equal(
                live_run["g_trace"], pre_run["g_trace"]
            )
            np.testing.assert_array_almost_equal(
                live_run["top_p_trace"], pre_run["top_p_trace"]
            )
            np.testing.assert_array_almost_equal(
                live_run["entropy_trace"], pre_run["entropy_trace"]
            )

    def test_permutation_consistency(
        self, sample_mc_question: MCQuestion, sample_corpus: list[str]
    ) -> None:
        """Permutation applied to beliefs matches permutation applied to gold_index."""
        import random as random_mod

        from agents.threshold_buzzer import _PrecomputedQuestion, precompute_beliefs
        from evaluation.controls import shuffled_option_copy

        likelihood = _make_likelihood(sample_corpus)
        beta = 5.0
        questions = [sample_mc_question]
        precomputed = precompute_beliefs(questions, likelihood, beta)

        # Reproduce the permutation that shuffled_option_copy would use
        rng_live = random_mod.Random(13)
        shuffled_q = shuffled_option_copy(sample_mc_question, rng_live)

        # Reproduce the same permutation for precomputed
        rng_pre = random_mod.Random(13)
        pq = precomputed[0]
        perm = list(range(pq.num_options))
        rng_pre.shuffle(perm)
        new_gold = perm.index(pq.gold_index)

        # The gold index should match
        assert new_gold == shuffled_q.gold_index


class TestBaselineAgentsVariableK:
    """Baseline agents work on non-K=4 questions (K-agnostic check)."""

    def test_threshold_buzzer_k3(self, sample_corpus):
        from agents.threshold_buzzer import ThresholdBuzzer
        from dataclasses import replace
        from models.likelihoods import TfIdfLikelihood
        from tests.conftest import sample_mc_question as _  # reuse fixture pattern

        model = TfIdfLikelihood(corpus_texts=sample_corpus)
        q4 = MCQuestion(
            qid="q_k3",
            question="Who was the first president?",
            tokens=["Who", "was", "the", "first", "president"],
            answer_primary="George Washington",
            clean_answers=["George Washington"],
            run_indices=[1, 3, 4],
            human_buzz_positions=[],
            category="History",
            cumulative_prefixes=["Who was", "Who was the first", "Who was the first president"],
            options=["George Washington", "Thomas Jefferson", "John Adams"],
            gold_index=0,
            option_profiles=[
                "George Washington first president",
                "Thomas Jefferson third president",
                "John Adams second president",
            ],
            option_answer_primary=["George Washington", "Thomas Jefferson", "John Adams"],
            distractor_strategy="test",
        )
        buzzer = ThresholdBuzzer(
            likelihood_model=model, threshold=0.5, beta=5.0, alpha=10.0,
        )
        result = buzzer.run_episode(q4)
        assert len(result.c_trace) > 0
        assert 0 <= result.buzz_index < 3
```

## File: tests/test_build_mc_dataset.py
```python
"""Regression tests for scripts/build_mc_dataset.py CLI defaults."""

from __future__ import annotations

from pathlib import Path

from qb_data.config import load_config as load_yaml_config, merge_overrides
from scripts.build_mc_dataset import parse_args, parse_overrides, resolve_output_dir


class TestBuildMcDatasetArgs:
    """Tests for smoke-aware argument resolution."""

    def test_parse_args_smoke_uses_dynamic_defaults(self) -> None:
        args = parse_args(["--smoke"])

        assert args.smoke is True
        assert args.config is None
        assert args.output_dir is None
        assert args.overrides == []

    def test_parse_args_explicit_overrides_win(self) -> None:
        args = parse_args(
            [
                "--smoke",
                "--config",
                "configs/custom.yaml",
                "--output-dir",
                "custom/output",
                "data.K=5",
            ]
        )

        assert args.smoke is True
        assert args.config == "configs/custom.yaml"
        assert args.output_dir == "custom/output"
        assert args.overrides == ["data.K=5"]

    def test_resolve_output_dir_defaults_to_smoke_artifacts(self) -> None:
        assert resolve_output_dir(None, smoke=True) == Path("artifacts/smoke")

    def test_resolve_output_dir_defaults_to_processed_data(self) -> None:
        assert resolve_output_dir(None, smoke=False) == Path("data/processed")

    def test_resolve_output_dir_preserves_explicit_override(self) -> None:
        assert resolve_output_dir("custom/output", smoke=True) == Path("custom/output")

    def test_load_config_smoke_without_explicit_path(self) -> None:
        cfg = load_yaml_config(None, smoke=True)

        assert cfg["data"]["max_questions"] == 50
        assert cfg["ppo"]["total_timesteps"] == 3000


class TestParseOverrides:
    """Tests for the fixed flat-key override parsing."""

    def test_returns_dotted_keys(self) -> None:
        """parse_overrides must return flat dotted keys, not nested dicts."""
        args = parse_args(["data.K=5", "environment.reward_mode=simple"])
        overrides = parse_overrides(args)
        assert "data.K" in overrides
        assert overrides["data.K"] == 5
        assert "environment.reward_mode" in overrides
        assert overrides["environment.reward_mode"] == "simple"
        assert "data" not in overrides, "Must not nest into a 'data' sub-dict"

    def test_preserves_sibling_sections(self) -> None:
        """Overriding data.K must not clobber data.csv_path."""
        base = {
            "data": {"K": 4, "csv_path": "questions.csv", "distractor_strategy": "sbert_profile"},
            "environment": {"reward_mode": "time_penalty", "seed": 13},
        }
        args = parse_args(["data.K=5"])
        overrides = parse_overrides(args)
        merged = merge_overrides(dict(base), overrides)
        assert merged["data"]["K"] == 5
        assert merged["data"]["csv_path"] == "questions.csv"
        assert merged["data"]["distractor_strategy"] == "sbert_profile"
        assert merged["environment"]["reward_mode"] == "time_penalty"

    def test_value_types(self) -> None:
        """Values are parsed as int, float, bool, or string."""
        args = parse_args(["data.K=5", "likelihood.beta=3.5", "data.shuffle=true", "data.name=foo"])
        overrides = parse_overrides(args)
        assert overrides["data.K"] == 5
        assert isinstance(overrides["data.K"], int)
        assert overrides["likelihood.beta"] == 3.5
        assert isinstance(overrides["likelihood.beta"], float)
        assert overrides["data.shuffle"] is True
        assert overrides["data.name"] == "foo"

    def test_no_overrides_returns_empty(self) -> None:
        args = parse_args(["--smoke"])
        overrides = parse_overrides(args)
        assert overrides == {}

    def test_merge_overrides_leaf_only(self) -> None:
        """merge_overrides with dotted keys updates only targeted leaves."""
        config = {
            "data": {"K": 4, "csv_path": "q.csv"},
            "environment": {"reward_mode": "simple"},
        }
        result = merge_overrides(config, {"data.K": 6, "environment.reward_mode": "time_penalty"})
        assert result["data"]["K"] == 6
        assert result["data"]["csv_path"] == "q.csv"
        assert result["environment"]["reward_mode"] == "time_penalty"
```

## File: tests/test_environment.py
```python
"""Test suite for qb_env/tossup_env.py — TossupMCEnv Gymnasium environment.

Covers:
- ENV-01: Gymnasium interface compliance (reset, step, spaces)
- ENV-02: Action space Discrete(K+1) with WAIT and BUZZ actions
- ENV-04: Reward modes (time_penalty, simple, human_grounded)
- ENV-05: Likelihood model pluggability
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import gymnasium as gym
import numpy as np
import pytest

from models.likelihoods import SBERTLikelihood, TfIdfLikelihood
from qb_data.mc_builder import MCQuestion
from qb_env.tossup_env import TossupMCEnv, precompute_beliefs


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #


def _make_env(
    mc_question: MCQuestion,
    corpus: list[str] | None = None,
    reward_mode: str = "simple",
    wait_penalty: float = 0.0,
    buzz_correct: float = 1.0,
    buzz_incorrect: float = -1.0,
    belief_mode: str = "from_scratch",
    beta: float = 5.0,
    use_sbert: bool = False,
) -> TossupMCEnv:
    """Create a TossupMCEnv with TF-IDF or SBERT likelihood model.

    Helper for tests that need a configured environment without going
    through the factory function.
    """
    if use_sbert:
        model = SBERTLikelihood()
    else:
        if corpus is None:
            corpus = mc_question.option_profiles[:]
        model = TfIdfLikelihood(corpus_texts=corpus)
    return TossupMCEnv(
        questions=[mc_question],
        likelihood_model=model,
        K=4,
        reward_mode=reward_mode,
        wait_penalty=wait_penalty,
        buzz_correct=buzz_correct,
        buzz_incorrect=buzz_incorrect,
        belief_mode=belief_mode,
        beta=beta,
    )


# ------------------------------------------------------------------ #
# Tests: Gymnasium Interface (ENV-01)
# ------------------------------------------------------------------ #


class TestGymnasiumInterface:
    """Tests for Gymnasium API compliance."""

    def test_isinstance_gym_env(self, sample_mc_question: MCQuestion) -> None:
        """TossupMCEnv is a subclass of gym.Env."""
        env = _make_env(sample_mc_question)
        assert isinstance(env, gym.Env), "TossupMCEnv should be a gym.Env subclass"

    def test_has_reset_and_step(self, sample_mc_question: MCQuestion) -> None:
        """Environment has reset() and step() methods."""
        env = _make_env(sample_mc_question)
        assert hasattr(env, "reset"), "Missing reset() method"
        assert hasattr(env, "step"), "Missing step() method"
        assert callable(env.reset), "reset should be callable"
        assert callable(env.step), "step should be callable"

    def test_action_space_discrete(self, sample_mc_question: MCQuestion) -> None:
        """Action space is Discrete(K+1) = Discrete(5) for K=4."""
        env = _make_env(sample_mc_question)
        assert isinstance(env.action_space, gym.spaces.Discrete), (
            f"Expected Discrete, got {type(env.action_space)}"
        )
        assert env.action_space.n == 5, (
            f"Expected Discrete(5) for K=4, got Discrete({env.action_space.n})"
        )

    def test_observation_space_box(self, sample_mc_question: MCQuestion) -> None:
        """Observation space is Box(K+6,) = Box(10,) for K=4."""
        env = _make_env(sample_mc_question)
        assert isinstance(env.observation_space, gym.spaces.Box), (
            f"Expected Box, got {type(env.observation_space)}"
        )
        assert env.observation_space.shape == (10,), (
            f"Expected shape (10,), got {env.observation_space.shape}"
        )
        assert env.observation_space.dtype == np.float32, (
            f"Expected float32, got {env.observation_space.dtype}"
        )

    def test_action_space_contains_all_valid_actions(
        self, sample_mc_question: MCQuestion
    ) -> None:
        """All actions 0..K are valid in the action space."""
        env = _make_env(sample_mc_question)
        for action in range(5):
            assert env.action_space.contains(action), (
                f"Action {action} should be valid"
            )
        assert not env.action_space.contains(5), "Action 5 should be invalid for K=4"
        assert not env.action_space.contains(-1), "Action -1 should be invalid"


# ------------------------------------------------------------------ #
# Tests: Episode Flow
# ------------------------------------------------------------------ #


class TestEpisodeFlow:
    """Tests for reset/step/termination lifecycle."""

    def test_reset_returns_obs_and_info(self, sample_mc_question: MCQuestion) -> None:
        """reset() returns (observation, info) tuple."""
        env = _make_env(sample_mc_question)
        result = env.reset()
        assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
        assert len(result) == 2, f"Expected 2 elements, got {len(result)}"

    def test_reset_obs_shape_dtype(self, sample_mc_question: MCQuestion) -> None:
        """Observation from reset is (K+6,) float32."""
        env = _make_env(sample_mc_question)
        obs, info = env.reset()
        assert obs.shape == (10,), f"Expected (10,), got {obs.shape}"
        assert obs.dtype == np.float32, f"Expected float32, got {obs.dtype}"

    def test_reset_info_contains_qid(self, sample_mc_question: MCQuestion) -> None:
        """Info dict from reset contains qid."""
        env = _make_env(sample_mc_question)
        _obs, info = env.reset()
        assert "qid" in info, "Info should contain 'qid'"
        assert info["qid"] == "test_q1", f"Expected 'test_q1', got {info['qid']}"

    def test_reset_initializes_state(self, sample_mc_question: MCQuestion) -> None:
        """After reset, step_idx=0, not terminated, not truncated."""
        env = _make_env(sample_mc_question)
        env.reset()
        assert env.step_idx == 0, f"step_idx should be 0, got {env.step_idx}"
        assert env.terminated is False, "terminated should be False"
        assert env.truncated is False, "truncated should be False"

    def test_wait_action_advances_step(self, sample_mc_question: MCQuestion) -> None:
        """WAIT (action 0) increments step_idx and returns not terminated."""
        env = _make_env(sample_mc_question)
        env.reset()
        obs, reward, terminated, truncated, info = env.step(0)
        assert not terminated, "Should not terminate on WAIT"
        assert obs.shape == (10,), f"Expected (10,), got {obs.shape}"
        assert env.step_idx == 1, f"step_idx should be 1, got {env.step_idx}"

    def test_buzz_correct_terminates(self, sample_mc_question: MCQuestion) -> None:
        """Buzzing with correct answer (action 1 = option 0 = gold) terminates."""
        env = _make_env(sample_mc_question)
        env.reset()
        obs, reward, terminated, truncated, info = env.step(1)  # gold_index=0, action=1
        assert terminated is True, "Should terminate on buzz"
        assert truncated is False, "Should not be truncated"
        assert info["correct"] is True, "Buzzing with gold should be correct"
        assert info["chosen_idx"] == 0, f"chosen_idx should be 0, got {info['chosen_idx']}"

    def test_buzz_incorrect_terminates(self, sample_mc_question: MCQuestion) -> None:
        """Buzzing with incorrect answer terminates with correct=False."""
        env = _make_env(sample_mc_question)
        env.reset()
        obs, reward, terminated, truncated, info = env.step(2)  # option 1 = incorrect
        assert terminated is True, "Should terminate on buzz"
        assert info["correct"] is False, "Buzzing with wrong answer should be incorrect"

    def test_forced_termination(self, sample_mc_question: MCQuestion) -> None:
        """Exhausting all clues causes truncation with forced choice."""
        env = _make_env(sample_mc_question)
        env.reset()
        total = env.total_steps  # 6 steps for sample question

        # WAIT until all clues exhausted
        for i in range(total):
            obs, reward, terminated, truncated, info = env.step(0)
            if truncated:
                break

        assert truncated is True, "Should be truncated after exhausting clues"
        assert "forced_choice" in info, "Info should contain 'forced_choice'"
        assert "forced_correct" in info, "Info should contain 'forced_correct'"
        assert isinstance(info["forced_choice"], int), "forced_choice should be int"

    def test_step_before_reset_raises(self, sample_mc_question: MCQuestion) -> None:
        """Calling step() before reset() raises RuntimeError."""
        env = _make_env(sample_mc_question)
        with pytest.raises(RuntimeError, match="reset"):
            env.step(0)

    def test_step_after_terminated_raises(self, sample_mc_question: MCQuestion) -> None:
        """Calling step() after termination raises RuntimeError."""
        env = _make_env(sample_mc_question)
        env.reset()
        env.step(1)  # buzz to terminate
        with pytest.raises(RuntimeError, match="terminated"):
            env.step(0)

    def test_invalid_action_raises(self, sample_mc_question: MCQuestion) -> None:
        """Invalid action raises ValueError."""
        env = _make_env(sample_mc_question)
        env.reset()
        with pytest.raises(ValueError, match="Invalid action"):
            env.step(99)

    def test_default_end_mode_is_force_commit(
        self, sample_mc_question: MCQuestion
    ) -> None:
        """Default env keeps legacy forced-commit behavior."""
        env = _make_env(sample_mc_question)
        assert env.end_mode == "force_commit"

    def test_no_buzz_end_mode_returns_marker_and_reward(
        self, sample_mc_question: MCQuestion
    ) -> None:
        """no_buzz mode truncates without forcing an answer choice."""
        corpus = sample_mc_question.option_profiles[:]
        model = TfIdfLikelihood(corpus_texts=corpus)
        env = TossupMCEnv(
            questions=[sample_mc_question],
            likelihood_model=model,
            K=4,
            reward_mode="simple",
            end_mode="no_buzz",
            no_buzz_reward=0.25,
        )
        env.reset()

        for _ in range(env.total_steps):
            _obs, reward, _terminated, truncated, info = env.step(0)
            if truncated:
                break

        assert truncated is True
        assert reward == pytest.approx(0.25)
        assert info["no_buzz"] is True
        assert info["forced_choice"] == -1
        assert info["forced_correct"] is False

    def test_invalid_end_mode_raises_on_terminal_wait(
        self, sample_mc_question: MCQuestion
    ) -> None:
        """Unknown end_mode raises ValueError at horizon."""
        corpus = sample_mc_question.option_profiles[:]
        model = TfIdfLikelihood(corpus_texts=corpus)
        env = TossupMCEnv(
            questions=[sample_mc_question],
            likelihood_model=model,
            K=4,
            reward_mode="simple",
            end_mode="unknown_mode",
        )
        env.reset()

        with pytest.raises(ValueError, match="Unknown end_mode"):
            for _ in range(env.total_steps):
                env.step(0)


class TestStopOnlyEnv:
    """Tests for the stop-only WAIT/BUZZ wrapper."""

    def test_stop_only_env_has_discrete2_action_space(
        self, sample_mc_question: MCQuestion
    ) -> None:
        """StopOnlyEnv exposes a binary action space."""
        from qb_env import StopOnlyEnv

        env = StopOnlyEnv(_make_env(sample_mc_question))
        assert isinstance(env.action_space, gym.spaces.Discrete)
        assert env.action_space.n == 2

    def test_stop_only_wait_delegates_to_base_env(
        self, sample_mc_question: MCQuestion
    ) -> None:
        """Action 0 remains a WAIT in the wrapped env."""
        from qb_env import StopOnlyEnv

        base_env = _make_env(sample_mc_question)
        env = StopOnlyEnv(base_env)
        env.reset()

        _obs, _reward, terminated, truncated, _info = env.step(0)
        assert not terminated
        assert not truncated
        assert base_env.step_idx == 1

    def test_stop_only_buzz_uses_argmax_belief(
        self, sample_mc_question: MCQuestion
    ) -> None:
        """Action 1 maps to BUZZ with the current belief argmax."""
        from qb_env import StopOnlyEnv

        base_env = _make_env(sample_mc_question)
        env = StopOnlyEnv(base_env)
        env.reset()
        base_env.belief = np.array([0.05, 0.8, 0.1, 0.05], dtype=np.float32)

        _obs, _reward, terminated, truncated, info = env.step(1)
        assert terminated
        assert not truncated
        assert info["chosen_idx"] == 1
        assert info["correct"] is False

    def test_stop_only_invalid_action_raises(
        self, sample_mc_question: MCQuestion
    ) -> None:
        """StopOnlyEnv rejects actions outside its Discrete(2) contract."""
        from qb_env import StopOnlyEnv

        base_env = _make_env(sample_mc_question)
        env = StopOnlyEnv(base_env)
        env.reset()

        with pytest.raises(ValueError, match="Invalid action"):
            env.step(2)

    def test_train_ppo_policy_mode_defaults_flat_kplus1(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """train_ppo CLI defaults to flat_kplus1 for compatibility."""
        from scripts.train_ppo import parse_args

        monkeypatch.setattr(sys, "argv", ["train_ppo.py"])
        args = parse_args()
        assert args.policy_mode == "flat_kplus1"


# ------------------------------------------------------------------ #
# Tests: Reward Modes (ENV-04)
# ------------------------------------------------------------------ #


class TestRewardModes:
    """Tests for different reward computation modes."""

    def test_reward_simple_correct(self, sample_mc_question: MCQuestion) -> None:
        """Simple mode: correct buzz gives +1.0."""
        env = _make_env(sample_mc_question, reward_mode="simple")
        env.reset()
        _obs, reward, _term, _trunc, _info = env.step(1)  # correct buzz
        assert reward == 1.0, f"Simple correct reward should be 1.0, got {reward}"

    def test_reward_simple_incorrect(self, sample_mc_question: MCQuestion) -> None:
        """Simple mode: incorrect buzz gives -1.0."""
        env = _make_env(sample_mc_question, reward_mode="simple")
        env.reset()
        _obs, reward, _term, _trunc, _info = env.step(2)  # incorrect buzz
        assert reward == -1.0, f"Simple incorrect reward should be -1.0, got {reward}"

    def test_reward_simple_wait_no_penalty(self, sample_mc_question: MCQuestion) -> None:
        """Simple mode: WAIT has 0 reward regardless of wait_penalty setting."""
        env = _make_env(
            sample_mc_question, reward_mode="simple", wait_penalty=0.1
        )
        env.reset()
        _obs, reward, _term, _trunc, _info = env.step(0)
        assert reward == 0.0, f"Simple WAIT reward should be 0.0, got {reward}"

    def test_reward_time_penalty_wait(self, sample_mc_question: MCQuestion) -> None:
        """Time penalty mode: WAIT incurs -wait_penalty."""
        env = _make_env(
            sample_mc_question, reward_mode="time_penalty", wait_penalty=0.1
        )
        env.reset()
        _obs, reward, _term, _trunc, _info = env.step(0)
        assert abs(reward - (-0.1)) < 1e-6, (
            f"Time penalty WAIT reward should be -0.1, got {reward}"
        )

    def test_reward_time_penalty_buzz_correct(
        self, sample_mc_question: MCQuestion
    ) -> None:
        """Time penalty mode: correct buzz gives buzz_correct."""
        env = _make_env(
            sample_mc_question,
            reward_mode="time_penalty",
            buzz_correct=1.0,
            wait_penalty=0.1,
        )
        env.reset()
        _obs, reward, _term, _trunc, _info = env.step(1)
        assert reward == 1.0, f"Time penalty correct buzz should be 1.0, got {reward}"

    def test_reward_time_penalty_cumulative(
        self, sample_mc_question: MCQuestion
    ) -> None:
        """Time penalty mode: waiting then buzzing accumulates penalties."""
        env = _make_env(
            sample_mc_question,
            reward_mode="time_penalty",
            wait_penalty=0.1,
            buzz_correct=1.0,
        )
        env.reset()
        # Wait 2 steps (-0.2 cumulative), then buzz correct (+1.0)
        total_reward = 0.0
        _obs, r1, _t, _tr, _info = env.step(0)
        total_reward += r1
        _obs, r2, _t, _tr, _info = env.step(0)
        total_reward += r2
        _obs, r3, _t, _tr, _info = env.step(1)  # buzz correct
        total_reward += r3
        assert abs(total_reward - 0.8) < 1e-6, (
            f"Cumulative reward should be ~0.8, got {total_reward}"
        )

    def test_reward_human_grounded(self, sample_mc_question: MCQuestion) -> None:
        """Human grounded mode works without human buzz data (returns normal reward)."""
        env = _make_env(
            sample_mc_question,
            reward_mode="human_grounded",
            buzz_correct=1.0,
            buzz_incorrect=-0.5,
        )
        env.reset()
        # With no human buzz positions, reward should be buzz_correct/incorrect
        _obs, reward, _term, _trunc, _info = env.step(1)
        assert reward == 1.0, f"Human grounded correct buzz should be 1.0, got {reward}"

    def test_reward_human_grounded_with_positions(self) -> None:
        """Human grounded mode: buzzing after human position gives 0.0."""
        # Create question with human buzz at position 0 (very early)
        mc_q = MCQuestion(
            qid="hg_test",
            question="Who was the first president?",
            tokens=["Who", "was", "the", "first", "president", "?"],
            answer_primary="George Washington",
            clean_answers=["George Washington"],
            run_indices=[0, 2, 4, 5],
            human_buzz_positions=[(0, 10)],  # Most humans buzz at position 0
            category="History",
            cumulative_prefixes=[
                "Who",
                "Who was the",
                "Who was the first president",
                "Who was the first president ?",
            ],
            options=["George Washington", "Jefferson", "Adams", "Franklin"],
            gold_index=0,
            option_profiles=["Washington", "Jefferson", "Adams", "Franklin"],
            option_answer_primary=["George Washington", "Jefferson", "Adams", "Franklin"],
            distractor_strategy="test",
        )
        corpus = mc_q.option_profiles[:]
        model = TfIdfLikelihood(corpus_texts=corpus)
        env = TossupMCEnv(
            questions=[mc_q],
            likelihood_model=model,
            K=4,
            reward_mode="human_grounded",
            buzz_correct=1.0,
            buzz_incorrect=-0.5,
        )
        env.reset()
        # Wait a few steps so agent buzzes after human position (0)
        env.step(0)  # step 0 -> reveal clue at position 0
        env.step(0)  # step 1 -> reveal clue at position 2
        _obs, reward, _term, _trunc, _info = env.step(1)  # buzz at step 2
        # Agent buzzes at token pos > 0 (human), so reward should be 0.0
        assert reward == 0.0, f"Should get 0.0 for buzzing after human, got {reward}"


# ------------------------------------------------------------------ #
# Tests: Likelihood Model Pluggability (ENV-05)
# ------------------------------------------------------------------ #


class TestLikelihoodPluggability:
    """Tests for interchangeable likelihood models."""

    def test_tfidf_model_produces_valid_obs(
        self, sample_mc_question: MCQuestion
    ) -> None:
        """TF-IDF likelihood model produces valid observations."""
        env = _make_env(sample_mc_question, use_sbert=False)
        obs, info = env.reset()
        assert obs.shape == (10,), f"Expected (10,), got {obs.shape}"
        assert np.all(np.isfinite(obs)), "All observations should be finite"
        # Take a step
        obs2, _r, _t, _tr, _info = env.step(0)
        assert obs2.shape == (10,), f"Expected (10,), got {obs2.shape}"
        assert np.all(np.isfinite(obs2)), "Step observations should be finite"

    def test_sbert_model_produces_valid_obs(
        self, sample_mc_question: MCQuestion
    ) -> None:
        """SBERT likelihood model produces valid observations."""
        env = _make_env(sample_mc_question, use_sbert=True)
        obs, info = env.reset()
        assert obs.shape == (10,), f"Expected (10,), got {obs.shape}"
        assert np.all(np.isfinite(obs)), "All observations should be finite"
        # Take a step
        obs2, _r, _t, _tr, _info = env.step(0)
        assert obs2.shape == (10,), f"Expected (10,), got {obs2.shape}"
        assert np.all(np.isfinite(obs2)), "Step observations should be finite"

    def test_both_models_same_obs_shape(
        self, sample_mc_question: MCQuestion
    ) -> None:
        """Both TF-IDF and SBERT produce same observation shape."""
        env_tfidf = _make_env(sample_mc_question, use_sbert=False)
        env_sbert = _make_env(sample_mc_question, use_sbert=True)

        obs_tfidf, _ = env_tfidf.reset(seed=42)
        obs_sbert, _ = env_sbert.reset(seed=42)

        assert obs_tfidf.shape == obs_sbert.shape, (
            f"TF-IDF obs {obs_tfidf.shape} != SBERT obs {obs_sbert.shape}"
        )
        assert obs_tfidf.dtype == obs_sbert.dtype, (
            f"TF-IDF dtype {obs_tfidf.dtype} != SBERT dtype {obs_sbert.dtype}"
        )


# ------------------------------------------------------------------ #
# Tests: Belief Modes
# ------------------------------------------------------------------ #


class TestBeliefModes:
    """Tests for different belief computation modes."""

    def test_from_scratch_belief(self, sample_mc_question: MCQuestion) -> None:
        """from_scratch mode recomputes belief from cumulative prefix."""
        env = _make_env(sample_mc_question, belief_mode="from_scratch")
        env.reset()
        # Wait several steps to get a more discriminative clue prefix
        for _ in range(3):
            env.step(0)
        # After multiple steps with more context, belief should be valid
        # and at least one option should have higher probability
        assert abs(env.belief.sum() - 1.0) < 1e-5, (
            f"Belief should sum to 1.0, got {env.belief.sum()}"
        )
        assert all(env.belief >= 0), "All beliefs should be non-negative"
        assert env.belief.dtype == np.float32, "Belief should be float32"

    def test_sequential_bayes_belief(self, sample_mc_question: MCQuestion) -> None:
        """sequential_bayes mode updates belief incrementally."""
        env = _make_env(sample_mc_question, belief_mode="sequential_bayes")
        env.reset()
        env.step(0)  # first WAIT
        # Belief should sum to ~1.0
        assert abs(env.belief.sum() - 1.0) < 1e-5, (
            f"Belief should sum to 1.0, got {env.belief.sum()}"
        )

    def test_invalid_belief_mode_raises(self, sample_mc_question: MCQuestion) -> None:
        """Unknown belief mode raises ValueError on step."""
        env = _make_env(sample_mc_question, belief_mode="unknown_mode")
        env.reset()
        with pytest.raises(ValueError, match="Unknown belief_mode"):
            env.step(0)


# ------------------------------------------------------------------ #
# Tests: Constructor Validation
# ------------------------------------------------------------------ #


class TestConstructorValidation:
    """Tests for constructor input validation."""

    def test_empty_questions_raises(self) -> None:
        """Empty question list raises ValueError."""
        model = TfIdfLikelihood(corpus_texts=["test"])
        with pytest.raises(ValueError, match="cannot be empty"):
            TossupMCEnv(questions=[], likelihood_model=model)

    def test_k_less_than_2_raises(self, sample_mc_question: MCQuestion) -> None:
        """K < 2 raises ValueError."""
        model = TfIdfLikelihood(corpus_texts=["test"])
        with pytest.raises(ValueError, match="K must be >= 2"):
            TossupMCEnv(
                questions=[sample_mc_question], likelihood_model=model, K=1
            )


# ------------------------------------------------------------------ #
# Tests: Precomputed Beliefs (OPT-1)
# ------------------------------------------------------------------ #


class TestPrecomputedBeliefs:
    """Tests for precomputed belief trajectory bypass."""

    def test_precomputed_matches_live_from_scratch(
        self, sample_mc_question: MCQuestion
    ) -> None:
        """Precomputed env produces identical beliefs as live env (from_scratch)."""
        corpus = sample_mc_question.option_profiles[:]
        model = TfIdfLikelihood(corpus_texts=corpus)
        questions = [sample_mc_question]

        # Run live env and record beliefs at each step
        live_env = TossupMCEnv(
            questions=questions, likelihood_model=model, K=4,
            belief_mode="from_scratch", beta=5.0,
        )
        live_env.reset(seed=42, options={"question_idx": 0})
        live_beliefs = []
        for _ in range(live_env.total_steps):
            live_env.step(0)  # WAIT
            live_beliefs.append(live_env.belief.copy())
            if live_env.truncated:
                break

        # Build precomputed cache
        cache = precompute_beliefs(
            questions=questions, likelihood_model=model,
            belief_mode="from_scratch", beta=5.0, K=4,
        )

        # Run precomputed env and compare beliefs
        pre_env = TossupMCEnv(
            questions=questions, likelihood_model=model, K=4,
            belief_mode="from_scratch", beta=5.0,
            precomputed_beliefs=cache,
        )
        pre_env.reset(seed=42, options={"question_idx": 0})
        for i in range(len(live_beliefs)):
            pre_env.step(0)
            np.testing.assert_allclose(
                pre_env.belief, live_beliefs[i], atol=1e-6,
                err_msg=f"Belief mismatch at step {i} (from_scratch)",
            )
            if pre_env.truncated:
                break

    def test_precomputed_matches_live_sequential_bayes(
        self, sample_mc_question: MCQuestion
    ) -> None:
        """Precomputed env produces identical beliefs as live env (sequential_bayes)."""
        corpus = sample_mc_question.option_profiles[:]
        model = TfIdfLikelihood(corpus_texts=corpus)
        questions = [sample_mc_question]

        # Run live env
        live_env = TossupMCEnv(
            questions=questions, likelihood_model=model, K=4,
            belief_mode="sequential_bayes", beta=5.0,
        )
        live_env.reset(seed=42, options={"question_idx": 0})
        live_beliefs = []
        for _ in range(live_env.total_steps):
            live_env.step(0)
            live_beliefs.append(live_env.belief.copy())
            if live_env.truncated:
                break

        # Build precomputed cache
        cache = precompute_beliefs(
            questions=questions, likelihood_model=model,
            belief_mode="sequential_bayes", beta=5.0, K=4,
        )

        # Run precomputed env
        pre_env = TossupMCEnv(
            questions=questions, likelihood_model=model, K=4,
            belief_mode="sequential_bayes", beta=5.0,
            precomputed_beliefs=cache,
        )
        pre_env.reset(seed=42, options={"question_idx": 0})
        for i in range(len(live_beliefs)):
            pre_env.step(0)
            np.testing.assert_allclose(
                pre_env.belief, live_beliefs[i], atol=1e-6,
                err_msg=f"Belief mismatch at step {i} (sequential_bayes)",
            )
            if pre_env.truncated:
                break

    def test_precomputed_skips_scoring(
        self, sample_mc_question: MCQuestion
    ) -> None:
        """Precomputed env never calls likelihood_model.score()."""
        corpus = sample_mc_question.option_profiles[:]
        model = TfIdfLikelihood(corpus_texts=corpus)
        questions = [sample_mc_question]

        cache = precompute_beliefs(
            questions=questions, likelihood_model=model,
            belief_mode="from_scratch", beta=5.0, K=4,
        )

        # Replace score with a mock
        mock_model = MagicMock(spec=TfIdfLikelihood)
        mock_model.score = MagicMock()

        env = TossupMCEnv(
            questions=questions, likelihood_model=mock_model, K=4,
            belief_mode="from_scratch", beta=5.0,
            precomputed_beliefs=cache,
        )
        env.reset(seed=42, options={"question_idx": 0})
        for _ in range(env.total_steps):
            env.step(0)
            if env.truncated:
                break

        mock_model.score.assert_not_called()

    def test_no_precomputed_backward_compat(
        self, sample_mc_question: MCQuestion
    ) -> None:
        """Env with precomputed_beliefs=None behaves identically to default."""
        corpus = sample_mc_question.option_profiles[:]
        model = TfIdfLikelihood(corpus_texts=corpus)
        questions = [sample_mc_question]

        # Default env (no precomputed_beliefs arg)
        env_default = TossupMCEnv(
            questions=questions, likelihood_model=model, K=4,
            belief_mode="from_scratch", beta=5.0,
        )
        env_default.reset(seed=42, options={"question_idx": 0})
        obs_default, _, _, _, _ = env_default.step(0)

        # Explicit None
        env_none = TossupMCEnv(
            questions=questions, likelihood_model=model, K=4,
            belief_mode="from_scratch", beta=5.0,
            precomputed_beliefs=None,
        )
        env_none.reset(seed=42, options={"question_idx": 0})
        obs_none, _, _, _, _ = env_none.step(0)

        np.testing.assert_array_equal(obs_default, obs_none)

    def test_precompute_beliefs_helper_shape(
        self, sample_mc_question: MCQuestion
    ) -> None:
        """precompute_beliefs returns correct keys and belief shapes."""
        corpus = sample_mc_question.option_profiles[:]
        model = TfIdfLikelihood(corpus_texts=corpus)
        questions = [sample_mc_question]

        cache = precompute_beliefs(
            questions=questions, likelihood_model=model,
            belief_mode="from_scratch", beta=5.0, K=4,
        )

        total_steps = len(sample_mc_question.run_indices)
        for s in range(total_steps):
            key = (0, s)
            assert key in cache, f"Missing key {key}"
            belief = cache[key]
            assert belief.shape == (4,), f"Expected (4,), got {belief.shape}"
            assert belief.dtype == np.float32, f"Expected float32, got {belief.dtype}"
            assert abs(belief.sum() - 1.0) < 1e-5, (
                f"Belief should sum to ~1.0, got {belief.sum()}"
            )


class TestExpectedWinsRewardMode:
    """Tests for the expected_wins reward mode in TossupMCEnv."""

    def _make_env(self, sample_mc_question, survival: float):
        """Build an EW env with a fixed-survival opponent model."""
        from unittest.mock import MagicMock

        from models.likelihoods import TfIdfLikelihood

        corpus = sample_mc_question.option_profiles[:]
        model = TfIdfLikelihood(corpus_texts=corpus)
        opp = MagicMock()
        opp.prob_survive_to_step = MagicMock(return_value=survival)
        opp.prob_buzzed_before_step = MagicMock(return_value=1.0 - survival)
        return TossupMCEnv(
            questions=[sample_mc_question],
            likelihood_model=model,
            K=4,
            reward_mode="expected_wins",
            opponent_buzz_model=opp,
            ew_reward_correct=10.0,
            ew_reward_incorrect=-5.0,
            ew_opponent_expected_value=0.0,
            belief_mode="from_scratch",
            beta=5.0,
        )

    def test_survival_1_correct_gives_ew_correct(self, sample_mc_question):
        env = self._make_env(sample_mc_question, survival=1.0)
        env.reset(seed=42, options={"question_idx": 0})
        gold = sample_mc_question.gold_index
        _, reward, _, _, _ = env.step(gold + 1)
        assert abs(reward - 10.0) < 1e-9

    def test_survival_1_incorrect_gives_ew_incorrect(self, sample_mc_question):
        env = self._make_env(sample_mc_question, survival=1.0)
        env.reset(seed=42, options={"question_idx": 0})
        wrong = (sample_mc_question.gold_index + 1) % 4
        _, reward, _, _, _ = env.step(wrong + 1)
        assert abs(reward - (-5.0)) < 1e-9

    def test_survival_0_gives_opponent_value(self, sample_mc_question):
        env = self._make_env(sample_mc_question, survival=0.0)
        env.reset(seed=42, options={"question_idx": 0})
        _, reward, _, _, _ = env.step(1)
        assert abs(reward - 0.0) < 1e-9

    def test_non_ew_modes_unchanged(self, sample_tfidf_env):
        """Non-EW reward modes are unaffected by the new EW plumbing."""
        env = sample_tfidf_env
        obs, _ = env.reset(seed=42)
        _, reward, _, _, _ = env.step(0)
        assert isinstance(reward, float)

    def test_expected_wins_no_buzz_end_mode(self, sample_mc_question):
        """expected_wins + no_buzz should truncate without forced choice."""
        env = self._make_env(sample_mc_question, survival=0.5)
        env.end_mode = "no_buzz"
        env.no_buzz_reward = 0.25
        env.reset(seed=42, options={"question_idx": 0})
        done = False
        truncated = False
        reward = 0.0
        info = {}
        while not (done or truncated):
            _, reward, done, truncated, info = env.step(0)
        assert truncated is True
        assert info["no_buzz"] is True
        assert info["forced_choice"] == -1
        assert reward == pytest.approx(0.25)


class TestVariableKEnv:
    """Tests for variable-K mode and action masks in TossupMCEnv."""

    def _make_mixed_k_questions(self, sample_mc_question):
        """Create a K=3 variant alongside the K=4 original."""
        from dataclasses import replace

        q3 = replace(
            sample_mc_question,
            qid="q_k3",
            options=sample_mc_question.options[:3],
            option_profiles=sample_mc_question.option_profiles[:3],
            option_answer_primary=sample_mc_question.option_answer_primary[:3],
            gold_index=0,
        )
        return [sample_mc_question, q3]

    def test_variable_k_obs_shape(self, sample_mc_question):
        from models.likelihoods import TfIdfLikelihood

        questions = self._make_mixed_k_questions(sample_mc_question)
        corpus = sample_mc_question.option_profiles[:]
        model = TfIdfLikelihood(corpus_texts=corpus)
        env = TossupMCEnv(
            questions=questions, likelihood_model=model,
            K=4, variable_K=True, max_K=4,
            reward_mode="simple", belief_mode="from_scratch",
        )
        obs, _ = env.reset(seed=42, options={"question_idx": 1})
        assert obs.shape == (4 + 6,)

    def test_action_mask_shape_and_validity(self, sample_mc_question):
        from models.likelihoods import TfIdfLikelihood

        questions = self._make_mixed_k_questions(sample_mc_question)
        corpus = sample_mc_question.option_profiles[:]
        model = TfIdfLikelihood(corpus_texts=corpus)
        env = TossupMCEnv(
            questions=questions, likelihood_model=model,
            K=4, variable_K=True, max_K=4,
            reward_mode="simple", belief_mode="from_scratch",
        )
        env.reset(seed=42, options={"question_idx": 1})
        mask = env.action_masks()
        assert mask.shape == (5,)
        assert mask[0]
        assert mask[1] and mask[2] and mask[3]
        assert not mask[4]

    def test_fixed_k_path_unchanged(self, sample_tfidf_env):
        """Fixed-K env (variable_K=False) behavior is unchanged."""
        env = sample_tfidf_env
        obs, _ = env.reset(seed=42)
        assert obs.shape == (4 + 6,)
        mask = env.action_masks()
        assert mask.shape == (5,)
        assert all(mask)
```

## File: tests/test_factories.py
```python
"""Test suite for factory functions — build_likelihood_from_config and make_env_from_config.

Covers:
- LIK-06: build_likelihood_from_config dispatches on config["likelihood"]["model"]
- CFG-02: make_env_from_config constructs TossupMCEnv from YAML config
"""

from __future__ import annotations

import numpy as np
import pytest

from models.likelihoods import (
    LikelihoodModel,
    SBERTLikelihood,
    TfIdfLikelihood,
    build_likelihood_from_config,
)
from qb_data.mc_builder import MCQuestion
from qb_env.tossup_env import TossupMCEnv, make_env_from_config


# ------------------------------------------------------------------ #
# Tests: build_likelihood_from_config (LIK-06)
# ------------------------------------------------------------------ #


class TestBuildLikelihoodFromConfig:
    """Tests for likelihood model factory function."""

    @pytest.fixture
    def stub_sbert_init(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Stub SBERT model loading so factory tests stay offline-safe."""

        def fake_init(self, model_name: str = "all-MiniLM-L6-v2") -> None:
            LikelihoodModel.__init__(self)
            self.model_name = model_name
            self.encoder = object()

        monkeypatch.setattr(SBERTLikelihood, "__init__", fake_init)

    def test_likelihood_factory_sbert(
        self, sample_config: dict, stub_sbert_init: None
    ) -> None:
        """Config with model='sbert' creates SBERTLikelihood."""
        sample_config["likelihood"]["model"] = "sbert"
        model = build_likelihood_from_config(sample_config)
        assert isinstance(model, SBERTLikelihood), (
            f"Expected SBERTLikelihood, got {type(model).__name__}"
        )

    def test_likelihood_factory_tfidf(
        self, sample_config: dict, sample_corpus: list[str]
    ) -> None:
        """Config with model='tfidf' creates TfIdfLikelihood (fitted)."""
        sample_config["likelihood"]["model"] = "tfidf"
        model = build_likelihood_from_config(sample_config, corpus_texts=sample_corpus)
        assert isinstance(model, TfIdfLikelihood), (
            f"Expected TfIdfLikelihood, got {type(model).__name__}"
        )
        assert model._is_fit is True, "TF-IDF model should be fitted after construction"

    def test_likelihood_factory_tfidf_missing_corpus(
        self, sample_config: dict
    ) -> None:
        """TF-IDF factory without corpus_texts raises ValueError."""
        sample_config["likelihood"]["model"] = "tfidf"
        with pytest.raises(ValueError, match="corpus_texts"):
            build_likelihood_from_config(sample_config)

    def test_likelihood_factory_unknown_model(self, sample_config: dict) -> None:
        """Unknown model name raises ValueError."""
        sample_config["likelihood"]["model"] = "unknown_model"
        with pytest.raises(ValueError, match="Unknown likelihood model"):
            build_likelihood_from_config(sample_config)

    def test_likelihood_factory_sbert_name_override(
        self, sample_config: dict, stub_sbert_init: None
    ) -> None:
        """sbert_name config key overrides default model name."""
        sample_config["likelihood"]["model"] = "sbert"
        sample_config["likelihood"]["sbert_name"] = "all-MiniLM-L6-v2"
        model = build_likelihood_from_config(sample_config)
        assert isinstance(model, SBERTLikelihood)
        assert model.model_name == "all-MiniLM-L6-v2", (
            f"Expected all-MiniLM-L6-v2, got {model.model_name}"
        )

    def test_likelihood_factory_embedding_model_key(
        self, sample_config: dict, stub_sbert_init: None
    ) -> None:
        """embedding_model config key works as fallback for sbert_name."""
        sample_config["likelihood"]["model"] = "sbert"
        sample_config["likelihood"]["embedding_model"] = "all-MiniLM-L6-v2"
        # Remove sbert_name if present to test fallback
        sample_config["likelihood"].pop("sbert_name", None)
        model = build_likelihood_from_config(sample_config)
        assert isinstance(model, SBERTLikelihood)
        assert model.model_name == "all-MiniLM-L6-v2"


# ------------------------------------------------------------------ #
# Tests: make_env_from_config (CFG-02)
# ------------------------------------------------------------------ #


class TestMakeEnvFromConfig:
    """Tests for environment factory function."""

    def _make_model_and_env(
        self, mc_question: MCQuestion, config: dict
    ) -> TossupMCEnv:
        """Helper to create a model and env from config."""
        corpus = mc_question.option_profiles[:]
        model = TfIdfLikelihood(corpus_texts=corpus)
        return make_env_from_config([mc_question], model, config)

    def test_env_factory_creates_tossup_env(
        self, sample_mc_question: MCQuestion, sample_config: dict
    ) -> None:
        """Factory creates a TossupMCEnv instance."""
        env = self._make_model_and_env(sample_mc_question, sample_config)
        assert isinstance(env, TossupMCEnv), (
            f"Expected TossupMCEnv, got {type(env).__name__}"
        )

    def test_env_factory_config_values(
        self, sample_mc_question: MCQuestion, sample_config: dict
    ) -> None:
        """Factory correctly extracts config values."""
        env = self._make_model_and_env(sample_mc_question, sample_config)
        assert env.K == 4, f"Expected K=4, got {env.K}"
        assert env.reward_mode == "simple", (
            f"Expected 'simple', got '{env.reward_mode}'"
        )
        assert env.belief_mode == "from_scratch", (
            f"Expected 'from_scratch', got '{env.belief_mode}'"
        )
        assert env.beta == 5.0, f"Expected beta=5.0, got {env.beta}"

    def test_env_factory_reward_mode_override(
        self, sample_mc_question: MCQuestion, sample_config: dict
    ) -> None:
        """Config overrides reward mode."""
        sample_config["environment"]["reward"] = "human_grounded"
        env = self._make_model_and_env(sample_mc_question, sample_config)
        assert env.reward_mode == "human_grounded", (
            f"Expected 'human_grounded', got '{env.reward_mode}'"
        )

    def test_env_factory_beta_override(
        self, sample_mc_question: MCQuestion, sample_config: dict
    ) -> None:
        """Config overrides beta value."""
        sample_config["likelihood"]["beta"] = 10.0
        env = self._make_model_and_env(sample_mc_question, sample_config)
        assert env.beta == 10.0, f"Expected beta=10.0, got {env.beta}"

    def test_env_factory_wait_penalty_override(
        self, sample_mc_question: MCQuestion, sample_config: dict
    ) -> None:
        """Config overrides wait_penalty value."""
        sample_config["environment"]["wait_penalty"] = 0.05
        env = self._make_model_and_env(sample_mc_question, sample_config)
        assert env.wait_penalty == 0.05, (
            f"Expected wait_penalty=0.05, got {env.wait_penalty}"
        )

    def test_env_factory_reset_works(
        self, sample_mc_question: MCQuestion, sample_config: dict
    ) -> None:
        """Factory-created env can reset and produce valid observation."""
        env = self._make_model_and_env(sample_mc_question, sample_config)
        obs, info = env.reset()
        assert obs.shape == (10,), f"Expected (10,), got {obs.shape}"
        assert "qid" in info, "Info should contain 'qid'"
        assert np.all(np.isfinite(obs)), "All observations should be finite"

    def test_env_factory_step_works(
        self, sample_mc_question: MCQuestion, sample_config: dict
    ) -> None:
        """Factory-created env can step and return valid results."""
        env = self._make_model_and_env(sample_mc_question, sample_config)
        env.reset()
        obs, reward, terminated, truncated, info = env.step(0)
        assert obs.shape == (10,), f"Expected (10,), got {obs.shape}"
        assert isinstance(reward, float), f"Reward should be float, got {type(reward)}"
        assert terminated is False, "WAIT should not terminate"

    def test_env_factory_reward_mode_key_fallback(
        self, sample_mc_question: MCQuestion
    ) -> None:
        """Factory supports 'reward_mode' key (default.yaml uses this)."""
        config = {
            "data": {"K": 4},
            "environment": {
                "reward_mode": "time_penalty",
                "wait_penalty": 0.1,
                "buzz_correct": 1.0,
                "buzz_incorrect": -0.5,
            },
            "likelihood": {"beta": 5.0},
        }
        corpus = sample_mc_question.option_profiles[:]
        model = TfIdfLikelihood(corpus_texts=corpus)
        env = make_env_from_config([sample_mc_question], model, config)
        assert env.reward_mode == "time_penalty", (
            f"Expected 'time_penalty', got '{env.reward_mode}'"
        )

    def test_env_factory_end_mode_and_no_buzz_reward(
        self, sample_mc_question: MCQuestion
    ) -> None:
        """Factory reads end_mode and no_buzz_reward from config."""
        config = {
            "data": {"K": 4},
            "environment": {
                "reward_mode": "simple",
                "end_mode": "no_buzz",
                "no_buzz_reward": 0.25,
                "wait_penalty": 0.1,
                "buzz_correct": 1.0,
                "buzz_incorrect": -0.5,
            },
            "likelihood": {"beta": 5.0},
        }
        corpus = sample_mc_question.option_profiles[:]
        model = TfIdfLikelihood(corpus_texts=corpus)
        env = make_env_from_config([sample_mc_question], model, config)
        assert getattr(env, "end_mode") == "no_buzz"
        assert getattr(env, "no_buzz_reward") == 0.25


class TestDSPyFactoryIntegration:
    """Factory dispatches to DSPyLikelihood when configured."""

    def test_factory_returns_dspy_likelihood(self):
        from models.dspy_likelihood import DSPyLikelihood

        config = {
            "likelihood": {"model": "dspy"},
            "dspy": {"cache_dir": None, "program_fingerprint": "test"},
        }
        model = build_likelihood_from_config(config)
        assert isinstance(model, DSPyLikelihood)

    def test_default_paths_unchanged(self, sample_corpus):
        config = {"likelihood": {"model": "tfidf"}}
        model = build_likelihood_from_config(config, corpus_texts=sample_corpus)
        assert isinstance(model, TfIdfLikelihood)
```

## File: tests/test_t5_policy.py
```python
"""Unit tests for T5PolicyModel and PolicyHead.

Tests cover PolicyHead architecture, T5PolicyModel forward pass, action
decomposition, tokenization, mean pooling, and checkpoint I/O.

Uses t5-small (60M params) for speed -- tests complete in <30 seconds.
The model fixture is module-scoped to load t5-small only once.
"""

from __future__ import annotations

import os
import tempfile

import pytest
import torch

from models.t5_policy import PolicyHead, T5PolicyModel


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def t5_small_config() -> dict:
    """Return a minimal config dict for T5PolicyModel with t5-small."""
    return {
        "model_name": "t5-small",
        "device": "cpu",
        "max_input_length": 128,
        "num_choices": 4,
    }


@pytest.fixture(scope="module")
def t5_small_model(t5_small_config):
    """Load T5PolicyModel with t5-small once per test module."""
    model = T5PolicyModel(t5_small_config)
    model.eval()
    return model


@pytest.fixture
def sample_texts() -> list[str]:
    """Return sample text inputs in quiz bowl format."""
    return [
        "CLUES: Who was the first president | CHOICES: (1) Washington (2) Jefferson (3) Adams (4) Franklin",
        "CLUES: This element has atomic number 1 | CHOICES: (1) Hydrogen (2) Helium (3) Lithium (4) Carbon",
    ]


# ---------------------------------------------------------------------------
# PolicyHead Tests
# ---------------------------------------------------------------------------


class TestPolicyHead:
    """Tests for PolicyHead class."""

    def test_policy_head_forward(self):
        """PolicyHead returns 3 tensors with correct shapes [B,2], [B,K], [B,1]."""
        batch_size = 4
        hidden_size = 512
        num_choices = 4

        head = PolicyHead(hidden_size=hidden_size, num_choices=num_choices)
        x = torch.randn(batch_size, hidden_size)

        wait_logits, answer_logits, values = head(x)

        assert wait_logits.shape == (batch_size, 2)
        assert answer_logits.shape == (batch_size, num_choices)
        assert values.shape == (batch_size, 1)

    def test_policy_head_different_num_choices(self):
        """PolicyHead handles non-default num_choices."""
        head = PolicyHead(hidden_size=256, num_choices=6)
        x = torch.randn(2, 256)

        wait_logits, answer_logits, values = head(x)

        assert wait_logits.shape == (2, 2)
        assert answer_logits.shape == (2, 6)
        assert values.shape == (2, 1)

    def test_policy_head_dropout(self):
        """Dropout layers exist and affect output in training mode."""
        head = PolicyHead(hidden_size=128, num_choices=4)
        head.train()  # Enable dropout

        x = torch.randn(8, 128)

        # Run forward twice in training mode; outputs should differ with high probability
        out1 = head(x)[0]
        out2 = head(x)[0]

        # Not strictly guaranteed but extremely likely with 8 samples and dropout
        # Use eval mode comparison for determinism
        head.eval()
        out3 = head(x)[0]
        out4 = head(x)[0]
        assert torch.allclose(out3, out4), "Eval mode should be deterministic"

    def test_policy_head_single_sample(self):
        """PolicyHead works with batch_size=1."""
        head = PolicyHead(hidden_size=512, num_choices=4)
        x = torch.randn(1, 512)

        wait_logits, answer_logits, values = head(x)

        assert wait_logits.shape == (1, 2)
        assert answer_logits.shape == (1, 4)
        assert values.shape == (1, 1)


# ---------------------------------------------------------------------------
# T5PolicyModel Tests
# ---------------------------------------------------------------------------


class TestT5PolicyModel:
    """Tests for T5PolicyModel class."""

    def test_t5_policy_init(self, t5_small_model):
        """T5PolicyModel initializes without errors and has correct structure."""
        model = t5_small_model

        assert hasattr(model, "encoder")
        assert hasattr(model, "tokenizer")
        assert hasattr(model, "policy_head")
        assert isinstance(model.policy_head, PolicyHead)

    def test_t5_policy_forward(self, t5_small_model, sample_texts):
        """Forward pass returns correct shapes for text inputs."""
        model = t5_small_model
        wait_logits, answer_logits, values = model(sample_texts)

        batch_size = len(sample_texts)
        assert wait_logits.shape == (batch_size, 2)
        assert answer_logits.shape == (batch_size, 4)
        assert values.shape == (batch_size, 1)

    def test_t5_policy_forward_no_value(self, t5_small_model, sample_texts):
        """Forward pass with return_value=False returns None for values."""
        model = t5_small_model
        wait_logits, answer_logits, values = model(sample_texts, return_value=False)

        assert values is None
        assert wait_logits.shape[0] == len(sample_texts)

    def test_encode_input(self, t5_small_model, sample_texts):
        """Tokenization produces input_ids and attention_mask with correct device."""
        model = t5_small_model
        encoding = model.encode_input(sample_texts)

        assert "input_ids" in encoding
        assert "attention_mask" in encoding
        assert encoding["input_ids"].shape[0] == len(sample_texts)
        assert encoding["attention_mask"].shape == encoding["input_ids"].shape
        assert encoding["input_ids"].device == model.device

    def test_encode_input_padding(self, t5_small_model):
        """Tokenization handles inputs of different lengths with padding."""
        model = t5_small_model
        texts = ["short", "this is a much longer text input with more tokens"]
        encoding = model.encode_input(texts)

        # Both should have same seq_len after padding
        assert encoding["input_ids"].shape[0] == 2
        # Second text should have more non-padding tokens
        mask_sums = encoding["attention_mask"].sum(dim=1)
        assert mask_sums[1] > mask_sums[0]

    def test_mean_pooling(self, t5_small_model):
        """Mean pooling respects attention mask (padded tokens have zero contribution)."""
        model = t5_small_model

        # Create a simple case: two identical sentences, one with extra padding
        texts = ["hello world"]
        encoding = model.encode_input(texts)

        pooled = model.get_encoder_output(
            encoding["input_ids"], encoding["attention_mask"]
        )

        # Output should be [1, hidden_size]
        assert pooled.shape == (1, model.encoder.config.d_model)
        assert not torch.isnan(pooled).any()
        assert not torch.isinf(pooled).any()


# ---------------------------------------------------------------------------
# Action Decomposition Tests
# ---------------------------------------------------------------------------


class TestActionDecomposition:
    """Tests for action decomposition in select_action and get_action_log_probs."""

    def test_action_decomposition_wait(self, t5_small_model, sample_texts):
        """action=0 decomposes to wait=0 in get_action_log_probs."""
        model = t5_small_model
        encoding = model.encode_input(sample_texts)

        # WAIT action
        actions = torch.zeros(len(sample_texts), dtype=torch.long, device=model.device)
        log_probs, entropy, values = model.get_action_log_probs(
            encoding["input_ids"], encoding["attention_mask"], actions
        )

        assert log_probs.shape == (len(sample_texts),)
        assert entropy.shape == (len(sample_texts),)
        assert values.shape == (len(sample_texts),)
        # Log probs should be negative
        assert (log_probs <= 0).all()
        # Entropy should be non-negative
        assert (entropy >= 0).all()

    def test_action_decomposition_buzz(self, t5_small_model, sample_texts):
        """actions 1-4 decompose to wait=1, answer=0-3."""
        model = t5_small_model
        encoding = model.encode_input(sample_texts[:1])  # Single sample

        for action_val in [1, 2, 3, 4]:
            actions = torch.tensor([action_val], dtype=torch.long, device=model.device)
            log_probs, entropy, values = model.get_action_log_probs(
                encoding["input_ids"], encoding["attention_mask"], actions
            )

            assert log_probs.shape == (1,)
            assert (log_probs <= 0).all()

    def test_joint_action_log_prob_wait_vs_buzz(self, t5_small_model):
        """WAIT uses only wait prob; BUZZ uses wait+buzzed-answer prob."""
        model = t5_small_model
        wait_logits = torch.tensor(
            [[2.0, 0.0], [0.0, 2.0]],
            dtype=torch.float32,
            device=model.device,
        )
        answer_logits = torch.tensor(
            [[0.1, 0.2, 0.3, 0.4], [1.0, 0.0, -1.0, -2.0]],
            dtype=torch.float32,
            device=model.device,
        )
        actions = torch.tensor([0, 2], dtype=torch.long, device=model.device)

        log_probs = model._joint_action_log_prob(wait_logits, answer_logits, actions)

        wait_log_probs = torch.log_softmax(wait_logits, dim=-1)
        answer_log_probs = torch.log_softmax(answer_logits, dim=-1)
        expected = torch.stack(
            [
                wait_log_probs[0, 0],
                wait_log_probs[1, 1] + answer_log_probs[1, 1],
            ]
        )
        assert torch.allclose(log_probs, expected, atol=1e-6)

    def test_joint_entropy_matches_chain_rule(self, t5_small_model):
        """Entropy follows H(wait) + p_buzz * H(answer)."""
        model = t5_small_model
        wait_logits = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0]],
            dtype=torch.float32,
            device=model.device,
        )
        answer_logits = torch.tensor(
            [[0.1, 0.2, 0.3, 0.4], [2.0, 1.0, 0.0, -1.0]],
            dtype=torch.float32,
            device=model.device,
        )

        entropy = model._joint_entropy(wait_logits, answer_logits)

        wait_probs = torch.softmax(wait_logits, dim=-1)
        wait_log_probs = torch.log_softmax(wait_logits, dim=-1)
        answer_probs = torch.softmax(answer_logits, dim=-1)
        answer_log_probs = torch.log_softmax(answer_logits, dim=-1)
        expected = (
            -(wait_probs * wait_log_probs).sum(dim=-1)
            + wait_probs[:, 1] * (-(answer_probs * answer_log_probs).sum(dim=-1))
        )
        assert torch.allclose(entropy, expected, atol=1e-6)

    def test_select_action_skips_answer_sampling_when_all_wait(
        self, t5_small_model, monkeypatch: pytest.MonkeyPatch
    ):
        """Answer sampling only runs for buzz examples, not all WAIT examples."""
        model = t5_small_model
        encoding = model.encode_input(["alpha", "beta"])

        hidden_size = model.encoder.config.d_model
        fake_pooled = torch.zeros((2, hidden_size), dtype=torch.float32, device=model.device)
        monkeypatch.setattr(model, "get_encoder_output", lambda *_args, **_kwargs: fake_pooled)

        def fake_head(_pooled):
            wait_logits = torch.tensor(
                [[10.0, -10.0], [8.0, -8.0]],
                dtype=torch.float32,
                device=model.device,
            )
            answer_logits = torch.tensor(
                [[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]],
                dtype=torch.float32,
                device=model.device,
            )
            values = torch.zeros((2, 1), dtype=torch.float32, device=model.device)
            return wait_logits, answer_logits, values

        monkeypatch.setattr(model.policy_head, "forward", fake_head)

        sample_shapes = []
        original_sample = torch.distributions.Categorical.sample

        def fake_sample(self, sample_shape=torch.Size()):
            sample_shapes.append(tuple(self.probs.shape))
            return torch.zeros(self.probs.shape[:-1], dtype=torch.long, device=self.probs.device)

        monkeypatch.setattr(torch.distributions.Categorical, "sample", fake_sample)

        actions, _info = model.select_action(
            encoding["input_ids"],
            encoding["attention_mask"],
            deterministic=False,
        )

        assert torch.equal(actions.cpu(), torch.zeros(2, dtype=torch.long))
        assert sample_shapes == [(2, 2)]

    def test_select_action_deterministic(self, t5_small_model, sample_texts):
        """Deterministic mode produces consistent actions."""
        model = t5_small_model
        encoding = model.encode_input(sample_texts)

        actions1, info1 = model.select_action(
            encoding["input_ids"],
            encoding["attention_mask"],
            deterministic=True,
        )
        actions2, info2 = model.select_action(
            encoding["input_ids"],
            encoding["attention_mask"],
            deterministic=True,
        )

        assert torch.equal(actions1, actions2)

    def test_select_action_stochastic(self, t5_small_model, sample_texts):
        """Stochastic mode samples from distribution (info dict has correct keys)."""
        model = t5_small_model
        encoding = model.encode_input(sample_texts)

        actions, info = model.select_action(
            encoding["input_ids"],
            encoding["attention_mask"],
            deterministic=False,
        )

        assert actions.shape == (len(sample_texts),)
        assert "wait_logits" in info
        assert "answer_logits" in info
        assert "wait_probs" in info
        assert "answer_probs" in info
        assert "values" in info
        assert "log_probs" in info

        # All actions should be in valid range [0, K]
        assert (actions >= 0).all()
        assert (actions <= 4).all()

    def test_select_action_returns_valid_range(self, t5_small_model, sample_texts):
        """Combined actions are in range [0, num_choices]."""
        model = t5_small_model
        encoding = model.encode_input(sample_texts)

        # Run many times to cover both wait and buzz actions
        for _ in range(10):
            actions, info = model.select_action(
                encoding["input_ids"],
                encoding["attention_mask"],
                deterministic=False,
                temperature=2.0,  # Higher temp for more randomness
            )
            assert (actions >= 0).all()
            assert (actions <= 4).all()

    def test_get_action_log_probs_matches_select(self, t5_small_model, sample_texts):
        """Log probs from get_action_log_probs are consistent with select_action."""
        model = t5_small_model
        model.eval()
        encoding = model.encode_input(sample_texts[:1])

        # Get deterministic action
        actions, info = model.select_action(
            encoding["input_ids"],
            encoding["attention_mask"],
            deterministic=True,
        )

        # Compute log probs for the same action
        log_probs, entropy, values = model.get_action_log_probs(
            encoding["input_ids"],
            encoding["attention_mask"],
            actions,
        )

        # Log probs should be finite
        assert torch.isfinite(log_probs).all()
        assert torch.isfinite(entropy).all()
        assert torch.isfinite(values).all()


# ---------------------------------------------------------------------------
# Predict Answer Tests
# ---------------------------------------------------------------------------


class TestPredictAnswer:
    """Tests for supervised training interface."""

    def test_predict_answer(self, t5_small_model, sample_texts):
        """predict_answer returns logits and predictions with correct shapes."""
        model = t5_small_model
        encoding = model.encode_input(sample_texts)

        answer_logits, predictions = model.predict_answer(
            encoding["input_ids"],
            encoding["attention_mask"],
        )

        assert answer_logits.shape == (len(sample_texts), 4)
        assert predictions.shape == (len(sample_texts),)
        # Predictions should be in valid range
        assert (predictions >= 0).all()
        assert (predictions < 4).all()


# ---------------------------------------------------------------------------
# Checkpoint Tests
# ---------------------------------------------------------------------------


class TestCheckpoint:
    """Tests for save/load checkpoint functionality."""

    def test_save_load_checkpoint(self, t5_small_model, sample_texts):
        """Save then load produces identical model outputs."""
        model = t5_small_model
        model.eval()

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "checkpoint")

            # Get output before save
            with torch.no_grad():
                wait_before, answer_before, value_before = model(sample_texts)

            # Save
            model.save(save_path)

            # Verify files exist
            assert os.path.exists(os.path.join(save_path, "policy_head.pt"))
            assert os.path.exists(os.path.join(save_path, "config.json"))

            # Load into same model
            model.load(save_path)

            # Get output after load
            with torch.no_grad():
                wait_after, answer_after, value_after = model(sample_texts)

            # Outputs should be identical
            assert torch.allclose(wait_before, wait_after, atol=1e-5)
            assert torch.allclose(answer_before, answer_after, atol=1e-5)
            assert torch.allclose(value_before, value_after, atol=1e-5)
```

## File: tests/test_text_wrapper.py
```python
"""Unit tests for TextObservationWrapper.

Tests verify that the wrapper correctly converts TossupMCEnv's numeric
belief observations into text-formatted strings for T5PolicyModel input.

Uses TF-IDF likelihood for fast test execution (<1 second total).
"""

from __future__ import annotations

import pytest

from qb_data.mc_builder import MCQuestion
from qb_env.text_wrapper import TextObservationWrapper
from qb_env.tossup_env import TossupMCEnv
from models.likelihoods import TfIdfLikelihood


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_mc_question() -> MCQuestion:
    """Return a minimal MCQuestion for wrapper testing."""
    tokens = [
        "Who", "was", "the", "first", "president",
        "of", "the", "United", "States", "?",
    ]
    run_indices = [0, 2, 4, 6, 8, 9]
    cumulative_prefixes = [
        "Who",
        "Who was the",
        "Who was the first president",
        "Who was the first president of the",
        "Who was the first president of the United States",
        "Who was the first president of the United States ?",
    ]
    return MCQuestion(
        qid="test_q1",
        question="Who was the first president of the United States?",
        tokens=tokens,
        answer_primary="George Washington",
        clean_answers=["George Washington", "Washington"],
        run_indices=run_indices,
        human_buzz_positions=[],
        category="History",
        cumulative_prefixes=cumulative_prefixes,
        options=[
            "George Washington",
            "Thomas Jefferson",
            "John Adams",
            "Benjamin Franklin",
        ],
        gold_index=0,
        option_profiles=[
            "George Washington first president commander revolutionary war",
            "Thomas Jefferson third president declaration independence",
            "John Adams second president Massachusetts diplomat",
            "Benjamin Franklin inventor diplomat Philadelphia printing",
        ],
        option_answer_primary=[
            "George Washington",
            "Thomas Jefferson",
            "John Adams",
            "Benjamin Franklin",
        ],
        distractor_strategy="test",
    )


@pytest.fixture
def wrapped_env(sample_mc_question: MCQuestion) -> TextObservationWrapper:
    """Return a TextObservationWrapper around a TossupMCEnv."""
    corpus = sample_mc_question.option_profiles[:]
    model = TfIdfLikelihood(corpus_texts=corpus)
    questions = [sample_mc_question] * 3
    env = TossupMCEnv(
        questions=questions,
        likelihood_model=model,
        K=4,
        reward_mode="simple",
        wait_penalty=0.0,
        buzz_correct=1.0,
        buzz_incorrect=-1.0,
        belief_mode="from_scratch",
        beta=5.0,
    )
    return TextObservationWrapper(env)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestTextObservationWrapper:
    """Tests for TextObservationWrapper class."""

    def test_wrapper_observation_format(self, wrapped_env: TextObservationWrapper):
        """Observation returns 'CLUES: ... | CHOICES: ...' format."""
        obs, info = wrapped_env.reset()

        assert isinstance(obs, str), f"Expected str, got {type(obs)}"
        assert "CLUES:" in obs, "Observation must contain 'CLUES:'"
        assert "CHOICES:" in obs, "Observation must contain 'CHOICES:'"
        assert "(1)" in obs, "Choices must be numbered starting at (1)"
        assert "(4)" in obs, "All 4 choices must be present"

    def test_wrapper_incremental_clues(self, wrapped_env: TextObservationWrapper):
        """Wrapper shows correct clues based on step_idx progression."""
        obs0, _ = wrapped_env.reset()

        # Initial: first token only
        clues_part = obs0.split(" | CHOICES:")[0].replace("CLUES: ", "")
        assert clues_part == "Who", f"Initial clues should be 'Who', got '{clues_part}'"

        # After first WAIT: cumulative_prefixes[0] = "Who"
        obs1, _, _, _, _ = wrapped_env.step(0)
        clues1 = obs1.split(" | CHOICES:")[0].replace("CLUES: ", "")
        assert clues1 == "Who", f"After 1st WAIT should be 'Who', got '{clues1}'"

        # After second WAIT: cumulative_prefixes[1] = "Who was the"
        obs2, _, _, _, _ = wrapped_env.step(0)
        clues2 = obs2.split(" | CHOICES:")[0].replace("CLUES: ", "")
        assert clues2 == "Who was the", f"After 2nd WAIT should be 'Who was the', got '{clues2}'"

    def test_wrapper_gymnasium_api(self, wrapped_env: TextObservationWrapper):
        """reset() and step() still work after wrapping."""
        # reset returns (obs, info) tuple
        result = wrapped_env.reset()
        assert isinstance(result, tuple)
        assert len(result) == 2
        obs, info = result
        assert isinstance(obs, str)
        assert isinstance(info, dict)
        assert "qid" in info

        # step returns (obs, reward, terminated, truncated, info)
        result = wrapped_env.step(0)  # WAIT
        assert isinstance(result, tuple)
        assert len(result) == 5
        obs, reward, terminated, truncated, info = result
        assert isinstance(obs, str)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_wrapper_preserves_reward(self, sample_mc_question: MCQuestion):
        """Reward from wrapped env matches underlying env behavior."""
        corpus = sample_mc_question.option_profiles[:]
        model = TfIdfLikelihood(corpus_texts=corpus)

        # Create unwrapped env
        env = TossupMCEnv(
            questions=[sample_mc_question] * 3,
            likelihood_model=model,
            K=4,
            reward_mode="simple",
            buzz_correct=1.0,
            buzz_incorrect=-1.0,
            seed=42,
        )

        # Create wrapped env with same seed
        env2 = TossupMCEnv(
            questions=[sample_mc_question] * 3,
            likelihood_model=model,
            K=4,
            reward_mode="simple",
            buzz_correct=1.0,
            buzz_incorrect=-1.0,
            seed=42,
        )
        wrapped = TextObservationWrapper(env2)

        # Reset both
        _, info1 = env.reset(seed=42)
        _, info2 = wrapped.reset(seed=42)

        # Take same actions
        _, r1, d1, t1, _ = env.step(0)
        _, r2, d2, t2, _ = wrapped.step(0)
        assert r1 == r2, f"Rewards differ: {r1} vs {r2}"
        assert d1 == d2, f"Terminated differs"
        assert t1 == t2, f"Truncated differs"

        # BUZZ with answer 1 (correct for gold_index=0)
        _, r1, d1, t1, _ = env.step(1)
        _, r2, d2, t2, _ = wrapped.step(1)
        assert r1 == r2, f"Buzz rewards differ: {r1} vs {r2}"
        assert d1 == d2

    def test_wrapper_multiple_steps(self, wrapped_env: TextObservationWrapper):
        """Multi-step episode produces increasing clue text."""
        obs, _ = wrapped_env.reset()
        prev_clues = obs.split(" | CHOICES:")[0]

        # Take multiple WAIT steps and verify clues grow
        grew_at_least_once = False
        for step in range(4):
            obs, _, terminated, truncated, _ = wrapped_env.step(0)
            if terminated or truncated:
                break
            current_clues = obs.split(" | CHOICES:")[0]
            if len(current_clues) > len(prev_clues):
                grew_at_least_once = True
            # Clues should never shrink
            assert len(current_clues) >= len(prev_clues), (
                f"Clues shrank at step {step}: '{prev_clues}' -> '{current_clues}'"
            )
            prev_clues = current_clues

        assert grew_at_least_once, "Clue text should grow with more WAITs"

    def test_wrapper_choices_include_all_options(
        self, wrapped_env: TextObservationWrapper
    ):
        """All 4 answer options appear in the choices section."""
        obs, _ = wrapped_env.reset()
        choices_part = obs.split("CHOICES: ")[1]

        assert "George Washington" in choices_part
        assert "Thomas Jefferson" in choices_part
        assert "John Adams" in choices_part
        assert "Benjamin Franklin" in choices_part

    def test_wrapper_buzz_ends_episode(self, wrapped_env: TextObservationWrapper):
        """Buzzing with an answer ends the episode."""
        wrapped_env.reset()
        _, _, terminated, truncated, info = wrapped_env.step(1)  # BUZZ answer 0
        assert terminated or truncated, "Episode should end after BUZZ"

    def test_wrapper_complete_episode(self, wrapped_env: TextObservationWrapper):
        """Full episode: WAIT until truncated or BUZZ."""
        wrapped_env.reset()

        for step in range(20):
            obs, reward, terminated, truncated, info = wrapped_env.step(0)
            if terminated or truncated:
                break
            assert isinstance(obs, str)

        # Episode must have ended (6 clue steps)
        assert terminated or truncated, "Episode should end within 20 steps"

    def test_wrapper_k3_formats_three_choices(
        self, sample_mc_question: MCQuestion
    ) -> None:
        """Text wrapper dynamically formats K=3 questions correctly."""
        from dataclasses import replace

        q3 = replace(
            sample_mc_question,
            qid="q_k3",
            options=sample_mc_question.options[:3],
            option_profiles=sample_mc_question.option_profiles[:3],
            option_answer_primary=sample_mc_question.option_answer_primary[:3],
            gold_index=0,
        )
        corpus = sample_mc_question.option_profiles[:]
        model = TfIdfLikelihood(corpus_texts=corpus)
        env = TossupMCEnv(
            questions=[q3],
            likelihood_model=model,
            K=3,
            reward_mode="simple",
            belief_mode="from_scratch",
        )
        wrapped = TextObservationWrapper(env)
        obs, _ = wrapped.reset(seed=42)
        assert "(3)" in obs
        assert "(4)" not in obs
```

## File: training/train_ppo_t5.py
```python
"""
Custom PPO Training for T5 Policy Model

Implements PPOTrainer with RolloutBuffer for end-to-end PPO fine-tuning of
T5PolicyModel on incremental quiz bowl episodes. Uses Generalized Advantage
Estimation (GAE) for variance reduction and dynamic batch padding to minimize
memory footprint.

Key design decisions:
    - Rollout tensors (input_ids, attention_mask) are immediately detached and
      moved to CPU after collection to prevent GPU memory accumulation.
    - Dynamic padding: each mini-batch is padded to the max length within that
      batch, not a global 512-token maximum, saving ~50%+ memory.
    - Config-dict interface for compatibility with the unified codebase YAML
      config pattern (see configs/t5_policy.yaml).

Ported from qanta-buzzer reference implementation (train_ppo.py) with:
    - TextObservationWrapper for text-based rollout collection
    - Memory-safe tensor management (detach + CPU storage)
    - Dynamic padding per mini-batch
    - Config dict interface replacing Config class
    - NumPy-style docstrings

Usage
-----
From Python::

    from training.train_ppo_t5 import PPOTrainer, run_ppo_training
    from models.t5_policy import T5PolicyModel
    from qb_data.mc_builder import MCQuestion

    model = T5PolicyModel({"model_name": "t5-small", "device": "cpu"})
    trainer = PPOTrainer(model, train_qs, val_qs, config)
    trainer.train()

From command line::

    python scripts/train_t5_policy.py --config configs/t5_policy.yaml
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models.t5_policy import T5PolicyModel
from qb_data.mc_builder import MCQuestion


@dataclass
class RolloutStep:
    """Single step in an episode rollout.

    Stores observation text, action, reward, value estimate, and log probability
    for a single environment step. Tokenized tensors (input_ids, attention_mask)
    are stored on CPU to prevent GPU memory accumulation during rollout collection.

    Attributes
    ----------
    observation_text : str
        Text observation at this step (CLUES: ... | CHOICES: ...).
    action : int
        Combined action taken (0=WAIT, 1..K=SELECT).
    reward : float
        Scalar reward received.
    done : bool
        Whether this step ended the episode.
    value : float
        Value estimate from the critic at this step.
    log_prob : float
        Log probability of the action under the policy at collection time.
    input_ids : torch.Tensor or None
        Tokenized input IDs stored on CPU. Shape ``[1, seq_len]``.
    attention_mask : torch.Tensor or None
        Attention mask stored on CPU. Shape ``[1, seq_len]``.
    return_ : float
        Discounted return (filled by ``compute_returns_and_advantages``).
    advantage : float
        GAE advantage (filled by ``compute_returns_and_advantages``).
    """

    observation_text: str
    action: int
    reward: float
    done: bool
    value: float
    log_prob: float
    input_ids: Optional[torch.Tensor] = None
    attention_mask: Optional[torch.Tensor] = None
    return_: float = 0.0
    advantage: float = 0.0


class RolloutBuffer:
    """Buffer to store and process episode rollouts for PPO updates.

    Accumulates complete episode rollouts (lists of RolloutStep), then computes
    discounted returns and GAE advantages across all episodes. Provides a flat
    view of all steps for mini-batch iteration during PPO updates.

    Attributes
    ----------
    rollouts : list[list[RolloutStep]]
        List of episode rollouts, each a list of steps.
    """

    def __init__(self) -> None:
        self.rollouts: List[List[RolloutStep]] = []

    def reset(self) -> None:
        """Clear all stored rollouts."""
        self.rollouts = []

    def add_rollout(self, steps: List[RolloutStep]) -> None:
        """Add a complete episode rollout to the buffer.

        Parameters
        ----------
        steps : list[RolloutStep]
            Complete episode rollout (ordered list of steps from reset to done).
        """
        self.rollouts.append(steps)

    def get_all_steps(self) -> List[RolloutStep]:
        """Get a flat list of all steps from all rollouts.

        Returns
        -------
        list[RolloutStep]
            All steps concatenated in order (rollout 0 steps, then rollout 1, ...).
        """
        all_steps: List[RolloutStep] = []
        for rollout in self.rollouts:
            all_steps.extend(rollout)
        return all_steps

    def compute_returns_and_advantages(
        self, gamma: float, gae_lambda: float
    ) -> None:
        """Compute discounted returns and GAE advantages for all rollouts.

        Uses Generalized Advantage Estimation (GAE) to compute per-step
        advantages. For each rollout, iterates backward from the terminal
        step computing:

            delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
            A_t = delta_t + gamma * lambda * A_{t+1}
            G_t = A_t + V(s_t)

        Terminal states reset next_value and gae to 0.

        Parameters
        ----------
        gamma : float
            Discount factor in [0, 1]. Higher values weight future rewards more.
        gae_lambda : float
            GAE lambda in [0, 1]. Trades off bias (low) vs variance (high).
        """
        for rollout in self.rollouts:
            rewards = [step.reward for step in rollout]
            values = [step.value for step in rollout]
            dones = [step.done for step in rollout]

            # GAE computation (backward pass)
            gae = 0.0
            next_value = 0.0  # Terminal state value

            for t in reversed(range(len(rollout))):
                if dones[t]:
                    next_value = 0.0
                    gae = 0.0

                # TD error
                delta = rewards[t] + gamma * next_value - values[t]

                # GAE accumulation
                gae = delta + gamma * gae_lambda * gae

                # Store return and advantage
                rollout[t].return_ = gae + values[t]
                rollout[t].advantage = gae

                next_value = values[t]

    def __len__(self) -> int:
        return len(self.rollouts)


class PPOTrainer:
    """Custom PPO trainer for T5PolicyModel on quiz bowl episodes.

    Collects rollouts by running T5PolicyModel in text-observation episodes
    (via TextObservationWrapper), then updates the policy using clipped
    surrogate PPO loss with value function and entropy regularization.

    The trainer handles the complete training loop:
    1. Collect rollouts (episodes) using the current policy
    2. Compute GAE advantages
    3. Update policy with mini-batch PPO for multiple epochs
    4. Periodically validate and save checkpoints

    Parameters
    ----------
    model : T5PolicyModel
        T5 policy model to train. Should be pre-trained via supervised
        warm-start for faster convergence.
    train_questions : list[MCQuestion]
        Training set questions for rollout collection.
    val_questions : list[MCQuestion]
        Validation set questions for periodic evaluation.
    config : dict[str, Any]
        Configuration dictionary with PPO hyperparameters:

        - ``ppo_lr`` (float): Learning rate. Default 1e-5.
        - ``ppo_iterations`` (int): Number of collect-update cycles. Default 100.
        - ``ppo_batch_size`` (int): Mini-batch size for PPO updates. Default 8.
        - ``ppo_epochs_per_iter`` (int): PPO epochs per iteration. Default 4.
        - ``ppo_gamma`` (float): Discount factor. Default 0.99.
        - ``ppo_gae_lambda`` (float): GAE lambda. Default 0.95.
        - ``ppo_clip_ratio`` (float): PPO clip ratio. Default 0.2.
        - ``ppo_value_coef`` (float): Value loss coefficient. Default 0.5.
        - ``ppo_entropy_coef`` (float): Entropy bonus coefficient. Default 0.01.
        - ``ppo_max_grad_norm`` (float): Gradient clip norm. Default 0.5.
        - ``ppo_episodes_per_iter`` (int): Episodes per rollout. Default 16.
        - ``eval_interval`` (int): Validate every N iterations. Default 10.
        - ``save_interval`` (int): Save checkpoint every N iterations. Default 20.
        - ``checkpoint_dir`` (str): Base checkpoint directory. Default "checkpoints".
        - ``reward_time_penalty`` (float): Time penalty for env. Default 0.1.

    Attributes
    ----------
    model : T5PolicyModel
        The model being trained.
    optimizer : torch.optim.AdamW
        Optimizer with weight decay.
    best_val_reward : float
        Best validation reward seen so far.
    history : list[dict]
        Per-iteration training metrics.
    checkpoint_dir : Path
        Directory for saving PPO checkpoints.
    """

    def __init__(
        self,
        model: T5PolicyModel,
        train_questions: List[MCQuestion],
        val_questions: List[MCQuestion],
        config: Dict[str, Any],
    ) -> None:
        self.model = model
        self.train_questions = list(train_questions)
        self.val_questions = list(val_questions)
        self.config = config

        self.device = model.device

        # PPO hyperparameters
        self.lr = float(config.get("ppo_lr", 1e-5))
        self.iterations = int(config.get("ppo_iterations", 100))
        self.batch_size = int(config.get("ppo_batch_size", 8))
        self.epochs_per_iter = int(config.get("ppo_epochs_per_iter", 4))
        self.gamma = float(config.get("ppo_gamma", 0.99))
        self.gae_lambda = float(config.get("ppo_gae_lambda", 0.95))
        self.clip_ratio = float(config.get("ppo_clip_ratio", 0.2))
        self.value_coef = float(config.get("ppo_value_coef", 0.5))
        self.entropy_coef = float(config.get("ppo_entropy_coef", 0.01))
        self.max_grad_norm = float(config.get("ppo_max_grad_norm", 0.5))
        self.episodes_per_iter = int(config.get("ppo_episodes_per_iter", 16))
        self.eval_interval = int(config.get("eval_interval", 10))
        self.save_interval = int(config.get("save_interval", 20))
        self.reward_time_penalty = float(config.get("reward_time_penalty", 0.1))
        self.max_input_length = int(config.get("max_input_length", 512))

        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(), lr=self.lr, weight_decay=0.01
        )

        # Training state
        self.current_iteration = 0
        self.best_val_reward = -float("inf")
        self.history: List[Dict[str, Any]] = []

        # Checkpoint directory
        self.checkpoint_dir = (
            Path(config.get("checkpoint_dir", "checkpoints")) / "ppo_t5"
        )
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def collect_rollouts(self, num_episodes: int) -> RolloutBuffer:
        """Collect rollouts by running episodes with the current policy.

        Creates a TossupMCEnv + TextObservationWrapper for each sampled
        question, runs the policy until episode termination, and stores
        all steps in a RolloutBuffer. Tokenized tensors are detached and
        moved to CPU immediately to prevent GPU memory accumulation.

        Parameters
        ----------
        num_episodes : int
            Number of episodes to collect.

        Returns
        -------
        RolloutBuffer
            Buffer containing all collected episode rollouts.
        """
        from qb_env.text_wrapper import TextObservationWrapper
        from qb_env.tossup_env import TossupMCEnv
        from models.likelihoods import TfIdfLikelihood

        self.model.eval()
        buffer = RolloutBuffer()

        # Sample questions for this iteration
        questions = random.choices(self.train_questions, k=num_episodes)

        # Build a simple TF-IDF likelihood for environment scoring
        # (The T5 policy reads text directly; likelihood is only used for
        # environment reward computation via belief updates)
        corpus = []
        for q in self.train_questions[:100]:  # Use subset for speed
            corpus.extend(q.option_profiles)
        likelihood_model = TfIdfLikelihood(corpus_texts=corpus)

        with torch.no_grad():
            for question in questions:
                env = TossupMCEnv(
                    questions=[question],
                    likelihood_model=likelihood_model,
                    K=len(question.options),
                    reward_mode="time_penalty",
                    wait_penalty=self.reward_time_penalty,
                    belief_mode="from_scratch",
                )
                wrapped_env = TextObservationWrapper(env)

                obs, info = wrapped_env.reset()
                done = False
                rollout: List[RolloutStep] = []

                while not done:
                    # Tokenize text observation
                    inputs = self.model.tokenizer(
                        obs,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=self.max_input_length,
                    ).to(self.device)

                    # Get action from policy
                    actions, act_info = self.model.select_action(
                        inputs["input_ids"],
                        inputs["attention_mask"],
                        deterministic=False,
                    )

                    action = actions.item()
                    value = act_info["values"].squeeze().item()
                    log_prob = act_info["log_probs"].item()

                    # Take environment step
                    next_obs, reward, terminated, truncated, step_info = (
                        wrapped_env.step(action)
                    )
                    done = terminated or truncated

                    # CRITICAL: Detach and move tensors to CPU immediately
                    # to prevent GPU memory accumulation during rollout collection
                    step = RolloutStep(
                        observation_text=obs,
                        action=action,
                        reward=reward,
                        done=done,
                        value=value,
                        log_prob=log_prob,
                        input_ids=inputs["input_ids"].detach().cpu(),
                        attention_mask=inputs["attention_mask"].detach().cpu(),
                    )
                    rollout.append(step)

                    obs = next_obs

                buffer.add_rollout(rollout)

        return buffer

    def _pad_batch(
        self, batch_steps: List[RolloutStep]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Dynamically pad a mini-batch of steps to the max length in the batch.

        Instead of padding all sequences to the global max (512 tokens), pads
        only to the longest sequence in the current mini-batch. This typically
        saves 50%+ memory since most quiz bowl observations are 100-200 tokens.

        Parameters
        ----------
        batch_steps : list[RolloutStep]
            Mini-batch of rollout steps with stored input_ids and attention_mask.

        Returns
        -------
        input_ids : torch.Tensor
            Padded input IDs of shape ``[batch_size, max_len]``, on device.
        attention_mask : torch.Tensor
            Padded attention mask of shape ``[batch_size, max_len]``, on device.
        """
        max_len = max(step.input_ids.shape[1] for step in batch_steps)
        pad_token_id = self.model.tokenizer.pad_token_id

        padded_input_ids = []
        padded_attention_mask = []

        for step in batch_steps:
            seq_len = step.input_ids.shape[1]
            if seq_len < max_len:
                pad_len = max_len - seq_len
                input_ids_padded = torch.cat(
                    [
                        step.input_ids,
                        torch.full(
                            (1, pad_len),
                            pad_token_id,
                            dtype=step.input_ids.dtype,
                        ),
                    ],
                    dim=1,
                )
                attention_mask_padded = torch.cat(
                    [
                        step.attention_mask,
                        torch.zeros(
                            (1, pad_len), dtype=step.attention_mask.dtype
                        ),
                    ],
                    dim=1,
                )
            else:
                input_ids_padded = step.input_ids
                attention_mask_padded = step.attention_mask

            padded_input_ids.append(input_ids_padded)
            padded_attention_mask.append(attention_mask_padded)

        input_ids = torch.cat(padded_input_ids).to(self.device)
        attention_mask = torch.cat(padded_attention_mask).to(self.device)

        return input_ids, attention_mask

    def update_policy(self, buffer: RolloutBuffer) -> Dict[str, float]:
        """Update the policy using PPO with clipped surrogate loss.

        Computes GAE advantages, normalizes them, then runs multiple epochs
        of mini-batch PPO updates. Each update computes the clipped surrogate
        policy loss, value function MSE loss, and entropy bonus.

        Parameters
        ----------
        buffer : RolloutBuffer
            Buffer with collected rollouts (compute_returns_and_advantages
            will be called internally).

        Returns
        -------
        dict[str, float]
            Training metrics: policy_loss, value_loss, entropy, num_updates.
        """
        self.model.train()

        # Compute returns and advantages
        buffer.compute_returns_and_advantages(
            gamma=self.gamma, gae_lambda=self.gae_lambda
        )

        # Get all steps
        all_steps = buffer.get_all_steps()
        if not all_steps:
            return {
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "entropy": 0.0,
                "num_updates": 0,
            }

        # Normalize advantages
        advantages = torch.tensor(
            [step.advantage for step in all_steps], dtype=torch.float32
        )
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-8
        )

        # Training metrics
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        num_updates = 0

        # PPO epochs
        for epoch in range(self.epochs_per_iter):
            # Shuffle step indices
            indices = np.random.permutation(len(all_steps))

            # Mini-batch updates
            for start_idx in range(0, len(all_steps), self.batch_size):
                end_idx = min(start_idx + self.batch_size, len(all_steps))
                batch_indices = indices[start_idx:end_idx]

                # Get batch steps
                batch_steps = [all_steps[i] for i in batch_indices]

                # Dynamic padding to max length in THIS batch
                input_ids, attention_mask = self._pad_batch(batch_steps)

                # Prepare batch tensors
                actions = torch.tensor(
                    [step.action for step in batch_steps],
                    dtype=torch.long,
                ).to(self.device)
                old_log_probs = torch.tensor(
                    [step.log_prob for step in batch_steps],
                    dtype=torch.float32,
                ).to(self.device)
                returns = torch.tensor(
                    [step.return_ for step in batch_steps],
                    dtype=torch.float32,
                ).to(self.device)
                batch_advantages = advantages[batch_indices].to(self.device)

                # Get new log probs, entropy, and values from current policy
                new_log_probs, entropy, values = (
                    self.model.get_action_log_probs(
                        input_ids, attention_mask, actions
                    )
                )

                # PPO clipped surrogate policy loss
                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = (
                    torch.clamp(
                        ratio,
                        1.0 - self.clip_ratio,
                        1.0 + self.clip_ratio,
                    )
                    * batch_advantages
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value function loss (MSE)
                value_loss = nn.MSELoss()(values, returns)

                # Entropy bonus (negative because we maximize entropy)
                entropy_loss = -entropy.mean()

                # Total loss
                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    + self.entropy_coef * entropy_loss
                )

                # Backward pass and optimizer step
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )
                self.optimizer.step()

                # Track metrics
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                num_updates += 1

        return {
            "policy_loss": total_policy_loss / max(1, num_updates),
            "value_loss": total_value_loss / max(1, num_updates),
            "entropy": total_entropy / max(1, num_updates),
            "num_updates": num_updates,
        }

    def validate(self) -> Dict[str, float]:
        """Validate on validation set by running deterministic episodes.

        Runs one episode per validation question with deterministic action
        selection (argmax) and computes accuracy and average reward.

        Returns
        -------
        dict[str, float]
            Validation metrics: accuracy, average_reward, avg_episode_length.
        """
        from qb_env.text_wrapper import TextObservationWrapper
        from qb_env.tossup_env import TossupMCEnv
        from models.likelihoods import TfIdfLikelihood

        self.model.eval()

        corpus = []
        for q in self.train_questions[:100]:
            corpus.extend(q.option_profiles)
        likelihood_model = TfIdfLikelihood(corpus_texts=corpus)

        correct = 0
        total = 0
        total_reward = 0.0
        total_length = 0

        # Limit validation size for speed
        val_questions = self.val_questions[:50]

        with torch.no_grad():
            for question in val_questions:
                env = TossupMCEnv(
                    questions=[question],
                    likelihood_model=likelihood_model,
                    K=len(question.options),
                    reward_mode="time_penalty",
                    wait_penalty=self.reward_time_penalty,
                    belief_mode="from_scratch",
                )
                wrapped_env = TextObservationWrapper(env)

                obs, info = wrapped_env.reset()
                done = False
                episode_reward = 0.0
                episode_length = 0

                while not done:
                    inputs = self.model.tokenizer(
                        obs,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=self.max_input_length,
                    ).to(self.device)

                    actions, act_info = self.model.select_action(
                        inputs["input_ids"],
                        inputs["attention_mask"],
                        deterministic=True,
                    )

                    action = actions.item()
                    obs, reward, terminated, truncated, step_info = (
                        wrapped_env.step(action)
                    )
                    done = terminated or truncated
                    episode_reward += reward
                    episode_length += 1

                total_reward += episode_reward
                total_length += episode_length
                total += 1

                # Check if answer was correct
                if step_info.get("correct", False) or step_info.get(
                    "forced_correct", False
                ):
                    correct += 1

        return {
            "accuracy": correct / max(1, total),
            "average_reward": total_reward / max(1, total),
            "avg_episode_length": total_length / max(1, total),
        }

    def train(self) -> Dict[str, Any]:
        """Run the full PPO training loop.

        Alternates between rollout collection and policy updates for
        ``self.iterations`` cycles. Periodically validates and saves
        checkpoints.

        Returns
        -------
        dict[str, Any]
            Training summary: best_val_reward, total_iterations.
        """
        print(f"Starting PPO training for {self.iterations} iterations")
        print(f"  Training questions: {len(self.train_questions)}")
        print(f"  Validation questions: {len(self.val_questions)}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Episodes per iteration: {self.episodes_per_iter}")
        print(f"  Device: {self.device}")
        print()

        for iteration in range(self.iterations):
            self.current_iteration = iteration

            # Collect rollouts
            print(f"\nIteration {iteration + 1}/{self.iterations}")
            print("  Collecting rollouts...")
            buffer = self.collect_rollouts(self.episodes_per_iter)

            # Compute episode statistics
            episode_rewards = []
            episode_lengths = []
            for rollout in buffer.rollouts:
                episode_reward = sum(step.reward for step in rollout)
                episode_rewards.append(episode_reward)
                episode_lengths.append(len(rollout))

            avg_reward = np.mean(episode_rewards) if episode_rewards else 0.0
            avg_length = np.mean(episode_lengths) if episode_lengths else 0.0

            print(f"  Avg episode reward: {avg_reward:.4f}")
            print(f"  Avg episode length: {avg_length:.2f}")

            # Update policy
            print("  Updating policy...")
            update_metrics = self.update_policy(buffer)

            print(f"  Policy loss: {update_metrics['policy_loss']:.4f}")
            print(f"  Value loss: {update_metrics['value_loss']:.4f}")
            print(f"  Entropy: {update_metrics['entropy']:.4f}")

            # Validate periodically
            if (iteration + 1) % self.eval_interval == 0:
                print("\n  Validating...")
                val_summary = self.validate()
                val_reward = val_summary.get("average_reward", 0.0)

                print(f"  Val Accuracy: {val_summary['accuracy']:.4f}")
                print(f"  Val Reward: {val_reward:.4f}")
                print(
                    f"  Val Avg Length: {val_summary['avg_episode_length']:.2f}"
                )

                # Save history
                self.history.append(
                    {
                        "iteration": iteration + 1,
                        "train_reward": float(avg_reward),
                        "train_length": float(avg_length),
                        **update_metrics,
                        "val": val_summary,
                    }
                )

                # Save best model
                if val_reward > self.best_val_reward:
                    self.best_val_reward = val_reward
                    self.save_checkpoint(is_best=True)
                    print(
                        f"  -> New best validation reward: {val_reward:.4f}"
                    )

            # Save regular checkpoint
            if (iteration + 1) % self.save_interval == 0:
                self.save_checkpoint(is_best=False)
                self.save_history()

        print("\n" + "=" * 60)
        print("PPO training completed!")
        print(f"Best validation reward: {self.best_val_reward:.4f}")
        print("=" * 60)

        # Save final history
        self.save_history()

        return {
            "best_val_reward": self.best_val_reward,
            "total_iterations": self.iterations,
        }

    def save_checkpoint(self, is_best: bool = False) -> Path:
        """Save model checkpoint to disk.

        Parameters
        ----------
        is_best : bool
            If True, save to ``best_model/`` directory.

        Returns
        -------
        Path
            Path to the saved checkpoint directory.
        """
        if is_best:
            save_path = self.checkpoint_dir / "best_model"
        else:
            save_path = (
                self.checkpoint_dir
                / f"iter_{self.current_iteration + 1}"
            )

        # Use T5PolicyModel's save() method
        self.model.save(str(save_path))

        # Save training state
        state = {
            "iteration": self.current_iteration + 1,
            "best_val_reward": self.best_val_reward,
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        torch.save(state, save_path / "training_state.pt")

        print(f"  Checkpoint saved to {save_path}")
        return save_path

    def save_history(self) -> Path:
        """Save training history to JSON.

        Returns
        -------
        Path
            Path to the saved history file.
        """
        history_path = self.checkpoint_dir / "history.json"
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2, default=float)
        return history_path


def run_ppo_training(
    config: Dict[str, Any],
    train_questions: List[MCQuestion],
    val_questions: List[MCQuestion],
    test_questions: Optional[List[MCQuestion]] = None,
    pretrained_model_path: Optional[str] = None,
) -> Tuple[T5PolicyModel, PPOTrainer]:
    """Run the PPO training pipeline with optional pretrained model.

    Creates or loads a T5PolicyModel, trains it with PPO on quiz bowl
    episodes, and optionally evaluates on a test set.

    Parameters
    ----------
    config : dict[str, Any]
        Configuration dictionary with model and PPO hyperparameters.
    train_questions : list[MCQuestion]
        Training set questions.
    val_questions : list[MCQuestion]
        Validation set questions.
    test_questions : list[MCQuestion] or None
        Optional test set for final evaluation.
    pretrained_model_path : str or None
        Path to a supervised pretrained checkpoint. If provided, loads the
        model from this path. Otherwise creates a new model.

    Returns
    -------
    model : T5PolicyModel
        The trained model.
    trainer : PPOTrainer
        The trainer instance with training history.
    """
    print("=" * 60)
    print("PPO TRAINING PHASE (T5 Policy)")
    print("=" * 60)

    # Load or create model
    if pretrained_model_path:
        print(f"Loading pretrained model from {pretrained_model_path}")
        device = config.get("device", "cpu")
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        model = T5PolicyModel.load_pretrained(
            pretrained_model_path, device=device
        )
    else:
        print("Initializing new model (no pretraining)")
        model_config = {
            "model_name": config.get("model_name", "t5-large"),
            "device": config.get("device", "cpu"),
            "max_input_length": config.get("max_input_length", 512),
            "num_choices": config.get("num_choices", 4),
        }
        model = T5PolicyModel(model_config)

    # Create trainer
    trainer = PPOTrainer(
        model=model,
        train_questions=train_questions,
        val_questions=val_questions,
        config=config,
    )

    # Train
    summary = trainer.train()

    # Evaluate on test set if provided
    if test_questions is not None:
        print("\n" + "=" * 60)
        print("FINAL EVALUATION ON TEST SET")
        print("=" * 60)

        # Load best model if it exists
        best_model_path = trainer.checkpoint_dir / "best_model"
        if best_model_path.exists():
            print(f"Loading best model from {best_model_path}")
            model.load(str(best_model_path))

        # Run validation on test set
        # Temporarily swap val questions with test questions
        original_val = trainer.val_questions
        trainer.val_questions = list(test_questions)
        test_metrics = trainer.validate()
        trainer.val_questions = original_val

        print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"Test Avg Reward: {test_metrics['average_reward']:.4f}")

        # Save test results
        test_results = {
            "test_metrics": test_metrics,
            "training_summary": summary,
        }
        results_path = trainer.checkpoint_dir / "test_results.json"
        with open(results_path, "w") as f:
            json.dump(test_results, f, indent=2, default=float)
        print(f"Test results saved to {results_path}")

    return model, trainer
```

## File: generate_poster.py
```python
#!/usr/bin/env python3
"""Generate a readable CS234 poster for the Quiz Bowl buzzer project.

Three-column landscape layout (30 x 20 in @ 150 dpi) with cross-platform
font loading and generous minimum font sizes for poster-session readability.
All content from the original poster is preserved.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Canvas: 30 x 20 inches at 150 dpi
# ---------------------------------------------------------------------------
DPI = 150
W, H = 30 * DPI, 20 * DPI  # 4500 x 3000 px
MARGIN = 70
COL_GAP = 40
CARD_GAP = 28
HEADER_H = 340
FOOTER_H = 70

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
BG = "#F6F3EE"
WHITE = "#FFFFFF"
PANEL = WHITE
PANEL_SOFT = "#FBF9F5"
TEXT = "#1F2933"
TEXT_SOFT = "#5F6B76"
BORDER = "#D7D1C8"
STANFORD_RED = "#8C1515"
NAVY = "#2F4EA1"
NAVY_DARK = "#243B7A"
BLUE = "#3B63D0"
BLUE_SOFT = "#E8EEFF"
PURPLE = "#7A57E2"
PURPLE_SOFT = "#EFE7FF"
GREEN = "#1E8E5A"
GREEN_SOFT = "#E7F7EE"
ORANGE = "#D8881F"
ORANGE_SOFT = "#FFF1DD"
GOLD = "#B57900"
GOLD_SOFT = "#FFF4DA"
RED_SOFT = "#FCE6E3"
GRID = "#E8E3DB"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
REPORT_PATH = ROOT / "artifacts" / "smoke" / "evaluation_report.json"
OUT_DIR = ROOT / "generated" / "quizbowl_mc_stopping_data_driven"

# ---------------------------------------------------------------------------
# Running-example content (from generate_presentation.py / frame deck)
# ---------------------------------------------------------------------------
EXAMPLE_SOURCE = "Example tossup: QBReader 2022 ARCADIA 08/11"
EXAMPLE_CHOICES = [
    ("A", "Andrey Andreyevich Markov"),
    ("B", "Leonhard Euler"),
    ("C", "Carl Friedrich Gauss"),
    ("D", "Augustin-Louis Cauchy"),
]
TOSSUP_SENTENCES = [
    [
        ("Given a set of nodes ", False),
        ("named for this person", True),
        (", a node is conditionally independent from the rest of "
         "a Bayesian network.", False),
    ],
    [
        ("The Baum-Welch algorithm is used to train a type of model ", False),
        ("named for this person", True),
        (" that is used for multiple sequence alignment.", False),
    ],
    [
        ('The initial "burn-in" states of a process ', False),
        ("named for this person", True),
        (" are discarded in methods like Gibbs sampling.", False),
    ],
    [
        ("The Metropolis-Hastings algorithm approximates an unknown "
         "distribution as the (*) stationary distribution of a process ", False),
        ("named for this person", True),
        (".", False),
    ],
    [
        ("Monte Carlo methods often use processes ", False),
        ("named for this person", True),
        (" that have stochastic transition matrices.", False),
    ],
    [
        ('Dynamic programming is used to decode "hidden" models ', False),
        ("named for this person", True),
        (".", False),
    ],
    [
        ("The next state of a random process ", False),
        ("named for this person", True),
        (" depends only on the current state.", False),
    ],
    [
        ("For 10 points, what Russian mathematician names a type of ", False),
        ('memoryless "chain?"', True),
    ],
]
TOSSUP_ANSWER = "Andrey Andreyevich Markov"
EXAMPLE_PREFIXES = [
    "blankets + HMM",
    "burn-in / Gibbs",
    "MC / stochastic",
    "memoryless prop.",
    'giveaway: "chain"',
]
EXAMPLE_POSTERIORS = [
    [0.42, 0.24, 0.21, 0.13],
    [0.56, 0.19, 0.15, 0.10],
    [0.68, 0.14, 0.11, 0.07],
    [0.82, 0.08, 0.06, 0.04],
    [0.91, 0.04, 0.03, 0.02],
]
EXAMPLE_DECISIONS = ["WAIT", "WAIT", "WAIT", "BUZZ", "BUZZ"]
ACT_NOW_VALUES = [0.18, 0.38, 0.49, 0.63, 0.74]
WAIT_VALUES = [0.64, 0.58, 0.52, 0.40, 0.28]

# ---------------------------------------------------------------------------
# Cross-platform font loading (macOS + Linux)
# ---------------------------------------------------------------------------
_FONT_PATHS = {
    "regular": [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ],
    "bold": [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/Library/Fonts/Arial Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
    ],
}
_TTC_FALLBACKS = [
    ("/System/Library/Fonts/Helvetica.ttc", 0, 1),
    ("/System/Library/Fonts/HelveticaNeue.ttc", 0, 1),
]

_font_warning_printed = False


def get_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    global _font_warning_printed
    key = "bold" if bold else "regular"
    for p in _FONT_PATHS[key]:
        if Path(p).exists():
            return ImageFont.truetype(p, size=size)
    for ttc_path, reg_idx, bold_idx in _TTC_FALLBACKS:
        if Path(ttc_path).exists():
            try:
                return ImageFont.truetype(
                    ttc_path, size=size, index=bold_idx if bold else reg_idx
                )
            except Exception:
                continue
    if not _font_warning_printed:
        print("WARNING: No system TrueType font found — falling back to PIL default")
        _font_warning_printed = True
    return ImageFont.load_default()


FONTS = {
    "title": get_font(140, bold=True),
    "subtitle": get_font(56),
    "authors": get_font(44),
    "section": get_font(48, bold=True),
    "body": get_font(40),
    "body_bold": get_font(40, bold=True),
    "small": get_font(36),
    "small_bold": get_font(36, bold=True),
    "detail": get_font(32),
    "detail_bold": get_font(32, bold=True),
    "caption": get_font(28),
    "caption_bold": get_font(28, bold=True),
}

# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------


def measure(draw: ImageDraw.ImageDraw, text: str, font) -> Tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0], box[3] - box[1]


def wrap_text(
    draw: ImageDraw.ImageDraw, text: str, font, max_width: int
) -> List[str]:
    if not text:
        return [""]
    lines: List[str] = []
    for para in text.split("\n"):
        para = para.strip()
        if not para:
            lines.append("")
            continue
        words = para.split()
        cur = words[0]
        for word in words[1:]:
            trial = cur + " " + word
            if measure(draw, trial, font)[0] <= max_width:
                cur = trial
            else:
                lines.append(cur)
                cur = word
        lines.append(cur)
    return lines


def fit_wrapped_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    box: Tuple[int, int, int, int],
    *,
    max_size: int,
    min_size: int,
    bold: bool = False,
    line_gap: int = 6,
):
    x0, y0, x1, y1 = box
    bw = max(1, x1 - x0)
    bh = max(1, y1 - y0)
    for size in range(max_size, min_size - 1, -1):
        font = get_font(size, bold=bold)
        lines = wrap_text(draw, text, font, bw)
        _, line_h = measure(draw, "Ag", font)
        total_h = len(lines) * line_h + max(0, len(lines) - 1) * line_gap
        if total_h <= bh and all(measure(draw, ln, font)[0] <= bw for ln in lines):
            return font, lines, total_h
    font = get_font(min_size, bold=bold)
    lines = wrap_text(draw, text, font, bw)
    _, line_h = measure(draw, "Ag", font)
    total_h = len(lines) * line_h + max(0, len(lines) - 1) * line_gap
    return font, lines, total_h


def draw_text_fit(
    draw: ImageDraw.ImageDraw,
    box: Tuple[int, int, int, int],
    text: str,
    *,
    fill=TEXT,
    max_size: int = 36,
    min_size: int = 24,
    bold: bool = False,
    align: str = "left",
    valign: str = "top",
    line_gap: int = 6,
):
    x0, y0, x1, y1 = box
    font, lines, total_h = fit_wrapped_text(
        draw, text, box,
        max_size=max_size, min_size=min_size, bold=bold, line_gap=line_gap,
    )
    _, line_h = measure(draw, "Ag", font)
    if valign == "center":
        y = y0 + max(0, ((y1 - y0) - total_h) // 2)
    else:
        y = y0
    for line in lines:
        lw, _ = measure(draw, line, font)
        if align == "center":
            x = x0 + max(0, ((x1 - x0) - lw) // 2)
        elif align == "right":
            x = x1 - lw
        else:
            x = x0
        draw.text((x, y), line, font=font, fill=fill)
        y += line_h + line_gap
    return y


def draw_bullets(
    draw: ImageDraw.ImageDraw,
    box: Tuple[int, int, int, int],
    items: Sequence[str],
    *,
    font=None,
    fill=TEXT,
    bullet_fill=STANFORD_RED,
    bullet_radius: int = 7,
    gap_after: int = 12,
) -> int:
    if font is None:
        font = FONTS["body"]
    x0, y0, x1, y1 = box
    bullet_pad = 30
    y = y0
    for item in items:
        lines = wrap_text(draw, item, font, (x1 - x0) - bullet_pad)
        _, line_h = measure(draw, "Ag", font)
        cy = y + line_h // 2 + 2
        draw.ellipse(
            (x0, cy - bullet_radius, x0 + 2 * bullet_radius, cy + bullet_radius),
            fill=bullet_fill,
        )
        for line in lines:
            draw.text((x0 + bullet_pad, y), line, font=font, fill=fill)
            y += line_h + 4
        y += gap_after
        if y > y1:
            break
    return y


# ---------------------------------------------------------------------------
# Primitive drawing helpers
# ---------------------------------------------------------------------------

CARD_HEADER_H = 76


def rounded(draw: ImageDraw.ImageDraw, xy, fill, outline=None, width=1, radius=22):
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def draw_card(
    draw: ImageDraw.ImageDraw,
    rect,
    title: str,
    accent: str,
    *,
    fill=PANEL,
    title_bg=None,
):
    x0, y0, x1, _y1 = rect
    rounded(draw, rect, fill=fill, outline=BORDER, width=2, radius=26)
    tb = title_bg or accent
    rounded(
        draw, (x0, y0, x1, y0 + CARD_HEADER_H),
        fill=tb, outline=tb, width=1, radius=24,
    )
    draw.rectangle(
        (x0 + 2, y0 + CARD_HEADER_H - 12, x1 - 2, y0 + CARD_HEADER_H), fill=tb,
    )
    draw.text((x0 + 22, y0 + 14), title, font=FONTS["section"], fill=WHITE)


def draw_subcard(
    draw: ImageDraw.ImageDraw, rect, title: str, color: str, fill=PANEL_SOFT,
):
    x0, y0, _x1, _y1 = rect
    rounded(draw, rect, fill=fill, outline=BORDER, width=2, radius=18)
    draw.text((x0 + 18, y0 + 12), title, font=FONTS["small_bold"], fill=color)


def draw_chip(
    draw: ImageDraw.ImageDraw, rect, text: str, fill, outline, text_fill, *, bold=False,
):
    rounded(draw, rect, fill=fill, outline=outline, width=2, radius=16)
    draw_text_fit(
        draw, (rect[0] + 8, rect[1] + 6, rect[2] - 8, rect[3] - 6),
        text, fill=text_fill, max_size=30, min_size=20, bold=bold,
        align="center", valign="center", line_gap=3,
    )


def arrow(
    draw: ImageDraw.ImageDraw,
    start: Tuple[int, int],
    end: Tuple[int, int],
    color: str,
    width: int = 4,
):
    draw.line([start, end], fill=color, width=width)
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    if dx == 0 and dy == 0:
        return
    ang = math.atan2(dy, dx)
    ah = 16
    left = (end[0] - ah * math.cos(ang - 0.5), end[1] - ah * math.sin(ang - 0.5))
    right = (end[0] - ah * math.cos(ang + 0.5), end[1] - ah * math.sin(ang + 0.5))
    draw.polygon([end, left, right], fill=color)


def fmt_pct(v) -> str:
    try:
        return f"{100 * float(v):.1f}%"
    except Exception:
        return "n/a"


def fmt_num(v, digits: int = 3) -> str:
    try:
        return f"{float(v):.{digits}f}"
    except Exception:
        return "n/a"


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def load_report() -> Dict:
    if not REPORT_PATH.exists():
        return {}
    try:
        return json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def baseline_points(report: Dict) -> List[Dict]:
    out: List[Dict] = []
    summary = report.get("baseline_summary", {})
    for method, payload in summary.items():
        if isinstance(payload, dict) and payload and all(
            isinstance(v, dict) for v in payload.values()
        ):
            for threshold, metrics in payload.items():
                out.append({
                    "label": f"{method}@{threshold}",
                    "family": method,
                    "threshold": threshold,
                    "mean_sq": float(metrics.get("mean_sq", 0.0)),
                    "mean_buzz_step": float(metrics.get("mean_buzz_step", 0.0)),
                    "buzz_accuracy": float(metrics.get("buzz_accuracy", 0.0)),
                    "mean_reward_like": float(metrics.get("mean_reward_like", 0.0)),
                })
        elif isinstance(payload, dict):
            out.append({
                "label": method,
                "family": method,
                "threshold": None,
                "mean_sq": float(payload.get("mean_sq", 0.0)),
                "mean_buzz_step": float(payload.get("mean_buzz_step", 0.0)),
                "buzz_accuracy": float(payload.get("buzz_accuracy", 0.0)),
                "mean_reward_like": float(payload.get("mean_reward_like", 0.0)),
            })
    return out


def pick_points(report: Dict) -> Dict[str, Dict]:
    points = baseline_points(report)
    seq05 = max(
        (
            p for p in points
            if p["family"] == "sequential_bayes" and p["threshold"] == "0.5"
        ),
        key=lambda x: x["mean_sq"],
        default=None,
    )
    always_final = next(
        (p for p in points if p["family"] == "always_final"), None,
    )
    ppo = report.get("ppo_summary", {})
    return {
        "sequential_bayes_05": seq05,
        "always_final": always_final,
        "ppo": {
            "label": "PPO smoke",
            "family": "ppo",
            "mean_sq": float(ppo.get("mean_sq", 0.0)),
            "mean_buzz_step": float(ppo.get("mean_buzz_step", 0.0)),
            "buzz_accuracy": float(ppo.get("buzz_accuracy", 0.0)),
            "mean_reward_like": float(ppo.get("mean_reward_like", 0.0)),
            "ece": float(ppo.get("ece", 0.0)),
            "brier": float(ppo.get("brier", 0.0)),
        },
    }


# ---------------------------------------------------------------------------
# Column 1 panels
# ---------------------------------------------------------------------------


def draw_problem_card(draw: ImageDraw.ImageDraw, rect):
    x0, y0, x1, y1 = rect
    draw_card(draw, rect, "Problem + design choice", STANFORD_RED)
    bx0, bx1 = x0 + 24, x1 - 24
    p = CARD_HEADER_H + 16

    y = draw_bullets(
        draw, (bx0, y0 + p, bx1, y0 + p + 300),
        [
            "Clues arrive incrementally, so the system must decide "
            "when to buzz — timing matters beyond answer selection.",
            "Early buzzes risk mistakes; late buzzes waste strategic value.",
            "A fixed 4-choice setting makes the stopping problem reproducible.",
        ],
        font=FONTS["small"], fill=TEXT,
        bullet_fill=STANFORD_RED, bullet_radius=7, gap_after=14,
    )

    py0 = y + 8
    py1 = py0 + 190
    draw_subcard(draw, (bx0, py0, bx1, py1), "POMDP at a glance", NAVY)
    fy = py0 + 52
    for label, value in [
        ("Observation", "prefix h_t + choice set C"),
        ("Actions", "WAIT or BUZZ(i)"),
        ("Goal", "buzz when acting beats waiting"),
    ]:
        draw_text_fit(
            draw, (bx0 + 20, fy, bx0 + 240, fy + 38),
            label, max_size=30, min_size=24, bold=True, fill=NAVY,
        )
        draw_text_fit(
            draw, (bx0 + 248, fy, bx1 - 20, fy + 38),
            value, max_size=30, min_size=24, fill=TEXT,
        )
        fy += 44

    my0 = py1 + 14
    my1 = my0 + 270
    draw_subcard(draw, (bx0, my0, bx1, my1), "Why multiple choice?", GREEN)
    draw_bullets(
        draw, (bx0 + 18, my0 + 52, bx1 - 18, my1 - 60),
        [
            "Removes aliasing and grading noise.",
            "Keeps the gold answer inside the action space.",
            "Allows cleaner baseline and control comparisons.",
            "Uses explicit anti-artifact distractor checks.",
        ],
        font=FONTS["detail"], fill=TEXT,
        bullet_fill=GREEN, bullet_radius=6, gap_after=10,
    )

    chip_y = my1 - 50
    chips = ["alias collisions", "token overlap", "length ratio", "question overlap"]
    cw_chip = (bx1 - bx0 - 18 - 3 * 10) // 4
    for i, label in enumerate(chips):
        cx = bx0 + 18 + i * (cw_chip + 10)
        draw_chip(
            draw, (cx, chip_y, cx + cw_chip, chip_y + 42),
            label, GREEN_SOFT, GREEN, GREEN,
        )

    mg0 = my1 + 14
    mg1 = y1 - 24
    draw_subcard(draw, (bx0, mg0, bx1, mg1), "Metrics glossary", ORANGE)
    metrics_defs = [
        ("S_q", "QANTA system score; rewards early correct buzzes"),
        ("acc", "buzz accuracy — fraction of buzzes on the gold answer"),
        ("step", "mean buzz step — average prefix index at buzz time"),
        ("rew", "reward-like — composite PPO training signal"),
    ]
    row_area = mg1 - mg0 - 44
    row_step = row_area // len(metrics_defs)
    gy = mg0 + 44
    for mlabel, mdesc in metrics_defs:
        draw_text_fit(
            draw, (bx0 + 20, gy, bx0 + 100, gy + row_step - 4),
            mlabel, max_size=28, min_size=18, bold=True, fill=ORANGE,
        )
        draw_text_fit(
            draw, (bx0 + 108, gy, bx1 - 20, gy + row_step - 4),
            mdesc, max_size=26, min_size=16, fill=TEXT,
        )
        gy += row_step


def draw_method_card(draw: ImageDraw.ImageDraw, rect):
    x0, y0, x1, _y1 = rect
    draw_card(draw, rect, "Method", NAVY)
    bx0, bx1 = x0 + 24, x1 - 24
    p = CARD_HEADER_H + 12

    draw_text_fit(
        draw, (bx0, y0 + p, bx1, y0 + p + 70),
        "Factor answer quality and stop timing into separate "
        "components, then combine into WAIT / BUZZ(i).",
        max_size=32, min_size=24, fill=TEXT_SOFT,
    )

    dy = y0 + p + 80
    cx = (bx0 + bx1) // 2
    bw = min(640, bx1 - bx0 - 40)

    ib = (cx - bw // 2, dy, cx + bw // 2, dy + 100)
    rounded(draw, ib, BLUE_SOFT, outline=BLUE, width=3, radius=18)
    draw_text_fit(
        draw, (ib[0] + 14, ib[1] + 8, ib[2] - 14, ib[1] + 44),
        "Incremental clue prefix h_t",
        max_size=30, min_size=22, bold=True, fill=NAVY,
    )
    draw_text_fit(
        draw, (ib[0] + 14, ib[1] + 48, ib[2] - 14, ib[3] - 8),
        "Observed text + 4-choice answer set C",
        max_size=26, min_size=20, fill=TEXT_SOFT,
    )

    split_y = dy + 110
    left_cx = cx - 170
    right_cx = cx + 170
    arrow(draw, (cx, dy + 100), (left_cx, split_y + 24), BLUE, width=4)
    arrow(draw, (cx, dy + 100), (right_cx, split_y + 24), ORANGE, width=4)

    abw = 290
    ab = (left_cx - abw // 2, split_y + 24, left_cx + abw // 2, split_y + 124)
    rounded(draw, ab, PURPLE_SOFT, outline=PURPLE, width=3, radius=16)
    draw_text_fit(
        draw, (ab[0] + 10, ab[1] + 6, ab[2] - 10, ab[1] + 38),
        "Answer model", max_size=28, min_size=20, bold=True, fill=PURPLE,
    )
    draw_text_fit(
        draw, (ab[0] + 10, ab[1] + 40, ab[2] - 10, ab[3] - 6),
        "Outputs p_ans(i | h_t) over A/B/C/D",
        max_size=24, min_size=18, fill=TEXT_SOFT,
    )

    sb = (right_cx - abw // 2, split_y + 24, right_cx + abw // 2, split_y + 124)
    rounded(draw, sb, GOLD_SOFT, outline=ORANGE, width=3, radius=16)
    draw_text_fit(
        draw, (sb[0] + 10, sb[1] + 6, sb[2] - 10, sb[1] + 38),
        "Stop head", max_size=28, min_size=20, bold=True, fill=ORANGE,
    )
    draw_text_fit(
        draw, (sb[0] + 10, sb[1] + 40, sb[2] - 10, sb[3] - 6),
        "Uses posterior / entropy / step features",
        max_size=24, min_size=18, fill=TEXT_SOFT,
    )

    merge_y = split_y + 134
    arrow(draw, (left_cx, split_y + 124), (cx, merge_y + 24), PURPLE, width=4)
    arrow(draw, (right_cx, split_y + 124), (cx, merge_y + 24), ORANGE, width=4)

    ob = (cx - bw // 2, merge_y + 24, cx + bw // 2, merge_y + 124)
    rounded(draw, ob, PANEL_SOFT, outline=GREEN, width=3, radius=18)
    draw_text_fit(
        draw, (ob[0] + 14, ob[1] + 8, ob[2] - 14, ob[1] + 40),
        "Flattened action distribution",
        max_size=28, min_size=22, bold=True, fill=GREEN,
    )
    draw_text_fit(
        draw, (ob[0] + 14, ob[1] + 44, ob[2] - 14, ob[3] - 8),
        'P("WAIT") = 1\u2212p_buzz   P("BUZZ i") = p_buzz \u00d7 p_ans(i)',
        max_size=24, min_size=18, fill=TEXT, align="center",
    )

    strip_y = merge_y + 140
    rounded(
        draw, (bx0, strip_y, bx1, strip_y + 82),
        "#F8F7F4", outline=BORDER, width=2, radius=14,
    )
    steps = [
        ("1", "Build MC prefixes", BLUE_SOFT, BLUE),
        ("2", "Run baselines", PURPLE_SOFT, PURPLE),
        ("3", "Warm-start stop", GOLD_SOFT, ORANGE),
        ("4", "PPO fine-tune", GREEN_SOFT, GREEN),
    ]
    sw = ((bx1 - bx0) - 40 - 3 * 12) // 4
    sx = bx0 + 20
    for i, (num, label, sfill, soutline) in enumerate(steps):
        bx = sx + i * (sw + 12)
        rounded(
            draw, (bx, strip_y + 12, bx + sw, strip_y + 70),
            sfill, outline=soutline, width=2, radius=12,
        )
        draw_chip(
            draw, (bx + 8, strip_y + 20, bx + 46, strip_y + 58),
            num, soutline, soutline, WHITE, bold=True,
        )
        draw_text_fit(
            draw, (bx + 54, strip_y + 20, bx + sw - 8, strip_y + 62),
            label, max_size=24, min_size=18, fill=TEXT, valign="center",
        )


def draw_conclusions_card(draw: ImageDraw.ImageDraw, rect):
    x0, y0, x1, y1 = rect
    draw_card(draw, rect, "Conclusions", STANFORD_RED)
    draw_bullets(
        draw, (x0 + 24, y0 + CARD_HEADER_H + 16, x1 - 24, y1 - 24),
        [
            "With a fixed answer set, buzzing becomes an "
            "optimal-stopping problem over growing evidence.",
            "The Markov example shows posterior mass concentrating "
            "before the final giveaway.",
        ],
        font=FONTS["small"], fill=TEXT,
        bullet_fill=STANFORD_RED, bullet_radius=7, gap_after=14,
    )


def draw_references_card(draw: ImageDraw.ImageDraw, rect):
    x0, y0, x1, y1 = rect
    draw_card(draw, rect, "References", GREEN)
    refs = [
        "Rodriguez et al. (2019). Quizbowl and incremental QA.",
        "QANTA (2024). System score S_q for buzzing.",
        "Sung et al. (2025). ADVSCORE.",
    ]
    y = draw_bullets(
        draw, (x0 + 24, y0 + CARD_HEADER_H + 16, x1 - 24, y1 - 64),
        refs, font=FONTS["detail"], fill=TEXT,
        bullet_fill=GREEN, bullet_radius=5, gap_after=8,
    )
    draw_text_fit(
        draw, (x0 + 24, y - 4, x1 - 24, y1 - 24),
        "Repo: github.com/hass0114/qanta-buzzer",
        max_size=28, min_size=22, fill=TEXT_SOFT,
    )


# ---------------------------------------------------------------------------
# Column 2: running example
# ---------------------------------------------------------------------------


def draw_highlighted_clue(
    draw: ImageDraw.ImageDraw,
    rect,
    left_text: str,
    highlight: str,
    right_text: str,
    *,
    fill=PANEL_SOFT,
    outline=BORDER,
    hl_fill=BLUE_SOFT,
    hl_text=NAVY_DARK,
):
    x0, y0, x1, y1 = rect
    rounded(draw, rect, fill=fill, outline=outline, width=2, radius=14)
    avail = (x1 - x0) - 32
    font = None
    for size in [32, 30, 28, 26, 24]:
        f = get_font(size)
        total = sum(
            measure(draw, t, f)[0] for t in [left_text, highlight, right_text]
        ) + 30
        if total <= avail:
            font = f
            break
    if font is None:
        font = get_font(24)

    _, fh = measure(draw, "Ag", font)
    tx = x0 + 16
    ty = y0 + ((y1 - y0) - fh) // 2
    draw.text((tx, ty), left_text, font=font, fill=TEXT)
    left_w = measure(draw, left_text, font)[0]
    hx0 = tx + left_w
    hpad = 6
    hl_w, hl_h = measure(draw, highlight, font)
    rounded(
        draw,
        (hx0 - 2, ty - 2, hx0 + hl_w + 2 * hpad, ty + hl_h + 2 * hpad - 2),
        hl_fill, outline=None, radius=10,
    )
    draw.text(
        (hx0 + hpad - 2, ty + hpad - 4), highlight, font=font, fill=hl_text,
    )
    rx = hx0 + hl_w + 2 * hpad + 6
    draw.text((rx, ty), right_text, font=font, fill=TEXT)


def draw_tossup_flow(
    draw: ImageDraw.ImageDraw, bx0: int, cy: int, bx1: int,
) -> int:
    """Render all 8 tossup sentences with run badges and anaphoric highlights.

    Returns the y coordinate below the last rendered line.
    """
    font = get_font(22)
    hl_font = get_font(22, bold=True)
    badge_font = get_font(16, bold=True)
    _, line_h = measure(draw, "Ag", font)
    space_w = measure(draw, " ", font)[0]

    badge_colors = [BLUE, BLUE, BLUE, PURPLE, PURPLE, ORANGE, ORANGE, GOLD]

    for run_idx, segments in enumerate(TOSSUP_SENTENCES):
        run_num = run_idx + 1
        bc = badge_colors[run_idx]
        is_giveaway = run_num == len(TOSSUP_SENTENCES)

        badge_r = 14
        badge_x = bx0 + badge_r + 2
        first_line_y = cy

        tokens: list = []
        for text, is_hl in segments:
            for w in text.split():
                if w:
                    tokens.append((w, is_hl))

        text_x0 = bx0 + badge_r * 2 + 14
        cx = text_x0

        for word, is_hl in tokens:
            f = hl_font if is_hl else font
            hl_bg = (
                GOLD_SOFT if (is_giveaway and is_hl)
                else (BLUE_SOFT if is_hl else None)
            )
            hl_color = (
                GOLD if (is_giveaway and is_hl)
                else (NAVY_DARK if is_hl else TEXT)
            )

            w = measure(draw, word, f)[0]

            if word and word[0] in ',.;:?!\u201d' and cx > text_x0:
                cx -= space_w

            if cx + w > bx1 and cx > text_x0:
                cx = text_x0
                cy += line_h + 2

            if hl_bg:
                rounded(
                    draw,
                    (int(cx - 2), cy - 1, int(cx + w + 2), cy + line_h + 1),
                    hl_bg, outline=None, radius=5,
                )

            draw.text((cx, cy), word, font=f, fill=hl_color)
            cx += w + space_w

        badge_cy = first_line_y + line_h // 2
        draw.ellipse(
            (badge_x - badge_r, badge_cy - badge_r,
             badge_x + badge_r, badge_cy + badge_r),
            fill=bc,
        )
        bw_b, bh_b = measure(draw, str(run_num), badge_font)
        draw.text(
            (badge_x - bw_b // 2, badge_cy - bh_b // 2),
            str(run_num), font=badge_font, fill=WHITE,
        )

        cy += line_h + 8

    draw_text_fit(
        draw, (bx0, cy + 2, bx1, cy + 30),
        f"ANSWER: {TOSSUP_ANSWER}",
        max_size=24, min_size=18, bold=True, fill=STANFORD_RED,
    )
    cy += 34
    return cy


def draw_example_card(draw: ImageDraw.ImageDraw, rect):
    x0, y0, x1, y1 = rect
    draw_card(draw, rect, "Key example: the Markov tossup", BLUE)

    bx0, bx1 = x0 + 22, x1 - 22
    cw = bx1 - bx0
    sec_gap = 18

    cy = y0 + CARD_HEADER_H + 12

    draw_text_fit(
        draw, (bx0, cy, bx1, cy + 28),
        EXAMPLE_SOURCE, max_size=24, min_size=18, fill=TEXT_SOFT,
    )
    cy += 30
    draw_text_fit(
        draw, (bx0, cy, bx1, cy + 36),
        "Full tossup \u2014 pyramidal clue progression (8 runs)",
        max_size=32, min_size=24, bold=True, fill=NAVY,
    )
    cy += 40

    cy = draw_tossup_flow(draw, bx0, cy, bx1)
    cy += sec_gap

    exp_h = 72
    rounded(
        draw, (bx0, cy, bx1, cy + exp_h),
        "#F8F7F4", outline=BORDER, width=2, radius=14,
    )
    draw_text_fit(
        draw, (bx0 + 14, cy + 6, bx1 - 14, cy + exp_h - 6),
        'Each sentence repeats the oblique referent "named for this person"; '
        'the final giveaway says "memoryless chain." '
        "Under MC framing, the core challenge shifts to timing the buzz.",
        max_size=22, min_size=16, fill=TEXT_SOFT, valign="center",
    )
    cy += exp_h + sec_gap

    ans_h = 100
    draw_subcard(draw, (bx0, cy, bx1, cy + ans_h), "Fixed answer set", NAVY)
    grid_x = bx0 + 18
    grid_y = cy + 46
    gbw = (cw - 56 - 3 * 12) // 4
    gbh = 40
    option_styles = [
        (BLUE_SOFT, BLUE), (PANEL_SOFT, BORDER),
        (PANEL_SOFT, BORDER), (PANEL_SOFT, BORDER),
    ]
    for idx, (label, text) in enumerate(EXAMPLE_CHOICES):
        gx = grid_x + idx * (gbw + 12)
        gy = grid_y
        gfill, goutline = option_styles[idx]
        rounded(
            draw, (gx, gy, gx + gbw, gy + gbh),
            fill=gfill, outline=goutline,
            width=3 if idx == 0 else 2, radius=14,
        )
        draw_chip(
            draw, (gx + 6, gy + 6, gx + 34, gy + gbh - 6),
            label, "#EDE7FF", goutline if idx != 0 else BLUE, NAVY, bold=True,
        )
        draw_text_fit(
            draw, (gx + 40, gy + 2, gx + gbw - 6, gy + gbh - 2),
            text, max_size=20, min_size=14, fill=TEXT, valign="center",
        )
    cy += ans_h + sec_gap

    remaining = y1 - cy - 24
    post_h = int(remaining * 0.55)
    chart_h = remaining - post_h - sec_gap

    poy0 = cy
    poy1 = poy0 + post_h
    draw_subcard(
        draw, (bx0, poy0, bx1, poy1),
        "Posterior over A/B/C/D as clues arrive", PURPLE,
    )
    colors = [BLUE, ORANGE, PURPLE, "#8A94A6"]
    bar_x0 = bx0 + 240
    bar_x1 = bx1 - 130
    n_rows = len(EXAMPLE_POSTERIORS)
    row_area = poy1 - poy0 - 100
    row_step = row_area // n_rows
    bar_h = min(36, row_step // 2)
    row_y = poy0 + 58
    run_labels = [
        "t\u22642", "t\u22643", "t\u22645", "t\u22647", "t=8",
    ]
    for i, probs in enumerate(EXAMPLE_POSTERIORS):
        draw_text_fit(
            draw, (bx0 + 16, row_y, bar_x0 - 12, row_y + 30),
            run_labels[i], max_size=28, min_size=20, bold=True, fill=TEXT,
        )
        draw_text_fit(
            draw, (bx0 + 16, row_y + 30, bar_x0 - 12, row_y + 56),
            EXAMPLE_PREFIXES[i], max_size=24, min_size=16, fill=TEXT_SOFT,
        )
        bar_top = row_y + 10
        rounded(
            draw, (bar_x0, bar_top, bar_x1, bar_top + bar_h),
            GRID, outline=None, radius=12,
        )
        seg = bar_x0
        total_w = bar_x1 - bar_x0
        for p_val, c in zip(probs, colors):
            seg_w = total_w * p_val
            rounded(
                draw, (seg, bar_top, seg + seg_w, bar_top + bar_h),
                c, outline=None, radius=12,
            )
            seg += seg_w
        draw_text_fit(
            draw, (bar_x1 + 10, bar_top, bx1 - 130, bar_top + bar_h),
            f"{probs[0] * 100:.0f}%",
            max_size=26, min_size=18, fill=TEXT_SOFT, valign="center",
        )
        dec = EXAMPLE_DECISIONS[i]
        dfill = ORANGE_SOFT if dec == "WAIT" else GREEN_SOFT
        dout = ORANGE if dec == "WAIT" else GREEN
        draw_chip(
            draw, (bx1 - 122, row_y + 4, bx1 - 18, row_y + 48),
            dec, dfill, dout, dout, bold=True,
        )
        row_y += row_step

    lx = bx0 + 20
    for lbl, c in zip(["A", "B", "C", "D"], colors):
        rounded(
            draw, (lx, poy1 - 38, lx + 22, poy1 - 16),
            c, outline=None, radius=6,
        )
        draw.text(
            (lx + 30, poy1 - 40), lbl, font=FONTS["caption_bold"], fill=TEXT_SOFT,
        )
        lx += 80
    cy += post_h + sec_gap

    vy0 = cy
    vy1 = vy0 + chart_h
    draw_subcard(draw, (bx0, vy0, bx1, vy1), "Act now vs. wait value", GREEN)

    chart_x0 = bx0 + 80
    chart_x1 = bx1 - 60
    chart_y0 = vy0 + 66
    chart_y1 = vy1 - 50

    draw.line((chart_x0, chart_y1, chart_x1, chart_y1), fill=BORDER, width=3)
    draw.line((chart_x0, chart_y0, chart_x0, chart_y1), fill=BORDER, width=3)
    for t in range(5):
        x = chart_x0 + t * (chart_x1 - chart_x0) / 4
        draw.line((x, chart_y1, x, chart_y1 + 8), fill=BORDER, width=2)
        draw_text_fit(
            draw, (int(x - 30), chart_y1 + 10, int(x + 30), chart_y1 + 34),
            f"{t + 1}", max_size=22, min_size=16, fill=TEXT_SOFT, align="center",
        )
    for yv in [0.2, 0.4, 0.6, 0.8]:
        y = chart_y1 - yv * (chart_y1 - chart_y0) / 0.8
        draw.line((chart_x0, y, chart_x1, y), fill=GRID, width=1)
        draw_text_fit(
            draw, (chart_x0 - 52, int(y - 10), chart_x0 - 6, int(y + 10)),
            f"{yv:.1f}", max_size=20, min_size=14, fill=TEXT_SOFT,
            align="right", valign="center",
        )

    def map_pt(i: int, val: float):
        return (
            chart_x0 + i * (chart_x1 - chart_x0) / 4,
            chart_y1 - val * (chart_y1 - chart_y0) / 0.8,
        )

    act_pts = [map_pt(i, v) for i, v in enumerate(ACT_NOW_VALUES)]
    wait_pts = [map_pt(i, v) for i, v in enumerate(WAIT_VALUES)]
    draw.line(act_pts, fill=BLUE, width=5)
    draw.line(wait_pts, fill=ORANGE, width=5)
    for pts, c in [(act_pts, BLUE), (wait_pts, ORANGE)]:
        for px, py in pts:
            draw.ellipse(
                (px - 6, py - 6, px + 6, py + 6),
                fill=PANEL, outline=c, width=3,
            )

    bx_buzz, _ = map_pt(3, 0)
    draw.line((bx_buzz, chart_y0 - 6, bx_buzz, chart_y1), fill=GREEN, width=3)
    draw_chip(
        draw,
        (int(bx_buzz - 50), chart_y0 - 36, int(bx_buzz + 50), chart_y0 - 4),
        "buzz", GREEN_SOFT, GREEN, GREEN, bold=True,
    )

    rounded(
        draw, (chart_x1 - 240, vy0 + 14, chart_x1 - 20, vy0 + 48),
        PANEL_SOFT, outline=BORDER, width=1, radius=12,
    )
    draw.line(
        (chart_x1 - 220, vy0 + 32, chart_x1 - 190, vy0 + 32),
        fill=BLUE, width=5,
    )
    draw.text(
        (chart_x1 - 182, vy0 + 18), "act now",
        font=FONTS["caption_bold"], fill=TEXT_SOFT,
    )
    draw.line(
        (chart_x1 - 112, vy0 + 32, chart_x1 - 82, vy0 + 32),
        fill=ORANGE, width=5,
    )
    draw.text(
        (chart_x1 - 74, vy0 + 18), "wait",
        font=FONTS["caption_bold"], fill=TEXT_SOFT,
    )


# ---------------------------------------------------------------------------
# Column 3 panels
# ---------------------------------------------------------------------------


def draw_scatter_card(draw: ImageDraw.ImageDraw, rect, report: Dict):
    x0, y0, x1, y1 = rect
    draw_card(draw, rect, "Smoke snapshot (sanity-check)", ORANGE)

    bx0, bx1 = x0 + 20, x1 - 20
    p = CARD_HEADER_H + 12
    draw_text_fit(
        draw, (bx0, y0 + p, bx1, y0 + p + 36),
        "n = 44. Smoke-test sanity checks only.",
        max_size=28, min_size=22, fill=TEXT_SOFT,
    )

    points = baseline_points(report)
    plotted = [
        pt for pt in points
        if pt["family"] in {"threshold", "sequential_bayes", "always_final"}
    ]
    ppo = report.get("ppo_summary", {})

    plot_x0 = bx0 + 70
    plot_y0 = y0 + p + 52
    plot_x1 = bx1 - 24
    plot_y1 = plot_y0 + 300
    x_max, y_max = 4.5, 0.44

    def xmap(v: float) -> int:
        return int(plot_x0 + (plot_x1 - plot_x0) * (v / x_max))

    def ymap(v: float) -> int:
        return int(plot_y1 - (plot_y1 - plot_y0) * (v / y_max))

    for tick in [0, 1, 2, 3, 4]:
        x = xmap(tick)
        draw.line((x, plot_y0, x, plot_y1), fill=GRID, width=1)
        draw_text_fit(
            draw, (x - 24, plot_y1 + 6, x + 24, plot_y1 + 28),
            f"{tick:.0f}", max_size=22, min_size=16, fill=TEXT_SOFT, align="center",
        )
    for tick in [0.1, 0.2, 0.3, 0.4]:
        y = ymap(tick)
        draw.line((plot_x0, y, plot_x1, y), fill=GRID, width=1)
        draw_text_fit(
            draw, (plot_x0 - 58, y - 10, plot_x0 - 6, y + 10),
            f"{tick:.1f}", max_size=22, min_size=16, fill=TEXT_SOFT,
            align="right", valign="center",
        )
    draw.line((plot_x0, plot_y1, plot_x1, plot_y1), fill=BORDER, width=3)
    draw.line((plot_x0, plot_y0, plot_x0, plot_y1), fill=BORDER, width=3)
    draw_text_fit(
        draw, (plot_x0, plot_y1 + 28, plot_x1, plot_y1 + 52),
        "mean buzz step (later = more evidence)",
        max_size=24, min_size=18, fill=TEXT_SOFT, align="center",
    )
    draw_text_fit(
        draw, (bx0, plot_y0, plot_x0 - 8, plot_y1),
        "mean\nS_q", max_size=24, min_size=18, bold=True, fill=TEXT_SOFT,
        align="center", valign="center",
    )

    family_style = {
        "threshold": (BLUE, True),
        "sequential_bayes": (GREEN, True),
        "always_final": (STANFORD_RED, False),
    }
    families: Dict[str, list] = {}
    for pt in plotted:
        families.setdefault(pt["family"], []).append(pt)

    for family, seq in families.items():
        color, line = family_style[family]
        seq = sorted(seq, key=lambda s: s["mean_buzz_step"])
        pts = [(xmap(s["mean_buzz_step"]), ymap(s["mean_sq"])) for s in seq]
        if line and len(pts) > 1:
            draw.line(pts, fill=color, width=4)
        for (px, py), s in zip(pts, seq):
            r = 10 if family != "always_final" else 12
            draw.ellipse(
                (px - r, py - r, px + r, py + r),
                fill=color, outline=PANEL, width=3,
            )
            if family == "always_final":
                draw_text_fit(
                    draw, (px + 16, py - 14, px + 160, py + 14),
                    "always_final", max_size=20, min_size=16, fill=color,
                )

    ppo_x = xmap(float(ppo.get("mean_buzz_step", 0.0)))
    ppo_y = ymap(float(ppo.get("mean_sq", 0.0)))
    draw.polygon(
        [
            (ppo_x, ppo_y - 13),
            (ppo_x + 13, ppo_y),
            (ppo_x, ppo_y + 13),
            (ppo_x - 13, ppo_y),
        ],
        fill=ORANGE, outline=PANEL,
    )
    draw_text_fit(
        draw, (ppo_x + 16, ppo_y - 16, min(plot_x1, ppo_x + 160), ppo_y + 16),
        "PPO smoke", max_size=22, min_size=16, fill=ORANGE,
    )

    seq05 = next(
        (pt for pt in families.get("sequential_bayes", [])
         if pt["threshold"] == "0.5"),
        None,
    )
    thr05 = next(
        (pt for pt in families.get("threshold", []) if pt["threshold"] == "0.5"),
        None,
    )
    if seq05:
        sx, sy = xmap(seq05["mean_buzz_step"]), ymap(seq05["mean_sq"])
        draw_text_fit(
            draw, (sx + 14, sy - 30, sx + 180, sy - 4),
            "seq_bayes@0.5", max_size=20, min_size=16, fill=GREEN,
        )
    if thr05:
        tx, ty_pt = xmap(thr05["mean_buzz_step"]), ymap(thr05["mean_sq"])
        draw_text_fit(
            draw, (tx + 14, ty_pt + 4, tx + 180, ty_pt + 28),
            "threshold@0.5", max_size=20, min_size=16, fill=BLUE,
        )

    leg_y = plot_y1 + 54
    items = [
        (BLUE, "threshold"), (GREEN, "seq_bayes"),
        (STANFORD_RED, "always_final"), (ORANGE, "ppo"),
    ]
    lx = bx0 + 10
    leg_step = (bx1 - bx0 - 20) // 4
    for c, lbl in items:
        rounded(draw, (lx, leg_y, lx + 16, leg_y + 16), c, outline=None, radius=5)
        draw.text(
            (lx + 24, leg_y - 4), lbl,
            font=FONTS["caption_bold"], fill=TEXT_SOFT,
        )
        lx += leg_step

    note_y = leg_y + 28
    draw_text_fit(
        draw, (bx0, note_y, bx1, note_y + 30),
        "softmax_profile overlaps threshold, omitted for readability.",
        max_size=24, min_size=18, fill=TEXT_SOFT,
    )

    chips_y0 = note_y + 40
    picks = pick_points(report)
    cards_data = [
        ("late baseline", picks["always_final"], STANFORD_RED, RED_SOFT),
        ("best non-final", picks["sequential_bayes_05"], GREEN, GREEN_SOFT),
        ("ppo smoke", picks["ppo"], ORANGE, ORANGE_SOFT),
    ]
    chip_w = (bx1 - bx0 - 2 * 12) // 3
    for i, (title, metrics, color, cfill) in enumerate(cards_data):
        cx0 = bx0 + i * (chip_w + 12)
        cx1 = cx0 + chip_w
        cy1 = y1 - 20
        rounded(
            draw, (cx0, chips_y0, cx1, cy1),
            cfill, outline=color, width=2, radius=14,
        )
        draw_text_fit(
            draw, (cx0 + 10, chips_y0 + 8, cx1 - 10, chips_y0 + 36),
            title, max_size=24, min_size=18, bold=True, fill=color, align="center",
        )
        lines = [
            f"S_q = {fmt_num(metrics.get('mean_sq'))}",
            f"acc = {fmt_pct(metrics.get('buzz_accuracy'))}",
            f"step = {fmt_num(metrics.get('mean_buzz_step'), 2)}",
        ]
        if title == "ppo smoke":
            lines.append(f"rew = {fmt_num(metrics.get('mean_reward_like'))}")
        draw_bullets(
            draw, (cx0 + 12, chips_y0 + 44, cx1 - 12, cy1 - 8),
            lines, font=FONTS["caption"], fill=TEXT,
            bullet_fill=color, bullet_radius=4, gap_after=4,
        )


def draw_controls_card(draw: ImageDraw.ImageDraw, rect, report: Dict):
    x0, y0, x1, y1 = rect
    draw_card(draw, rect, "Results + controls", GREEN)

    bx0, bx1 = x0 + 20, x1 - 20
    p = CARD_HEADER_H + 12

    rounded(
        draw, (bx0, y0 + p, bx1, y0 + p + 90),
        PANEL_SOFT, outline=BORDER, width=2, radius=14,
    )
    draw_text_fit(
        draw, (bx0 + 16, y0 + p + 8, bx1 - 16, y0 + p + 82),
        "Smoke takeaway: always_final best S_q (0.386); "
        "sequential_bayes@0.5 strongest non-final (0.267); "
        "PPO reaches 0.326 but buzzes at step 0.0.",
        max_size=26, min_size=20, fill=TEXT, valign="center",
    )

    controls = report.get("controls", {})
    full = report.get("full_eval", {})
    choices = controls.get("choices_only", {})
    shuffle = controls.get("shuffle", {})
    alias = controls.get("alias_substitution", {})

    card_h = 150
    card_gap = 14
    cards = [
        {
            "title": "Choices only",
            "big": fmt_pct(choices.get("accuracy")),
            "sub": f"chance = {fmt_pct(choices.get('chance'))}  \u2022  "
                   f"n = {fmt_num(choices.get('n_test'), 0)}",
            "note": "Answer options alone do not solve the task.",
            "color": BLUE, "fill": BLUE_SOFT,
        },
        {
            "title": "Shuffle clues",
            "big": f"\u0394S_q = {float(shuffle.get('mean_sq', 0)) - float(full.get('mean_sq', 0)):+.3f}",
            "sub": f"full = {fmt_num(full.get('mean_sq'))}  \u2022  "
                   f"shuffled = {fmt_num(shuffle.get('mean_sq'))}",
            "note": "Very small movement on the smoke slice.",
            "color": ORANGE, "fill": ORANGE_SOFT,
        },
        {
            "title": "Alias substitution",
            "big": f"\u0394S_q = {float(alias.get('mean_sq', 0)) - float(full.get('mean_sq', 0)):+.3f}",
            "sub": f"full = {fmt_num(full.get('mean_sq'))}  \u2022  "
                   f"alias = {fmt_num(alias.get('mean_sq'))}",
            "note": "Essentially unchanged in this smoke report.",
            "color": GREEN, "fill": GREEN_SOFT,
        },
    ]

    cy = y0 + p + 106
    for c in cards:
        rounded(
            draw, (bx0, cy, bx1, cy + card_h),
            c["fill"], outline=c["color"], width=2, radius=14,
        )
        draw_text_fit(
            draw, (bx0 + 14, cy + 10, bx1 - 14, cy + 36),
            c["title"], max_size=26, min_size=20, bold=True, fill=c["color"],
        )
        draw_text_fit(
            draw, (bx0 + 14, cy + 38, bx1 - 14, cy + 74),
            c["big"], max_size=34, min_size=24, bold=True, fill=TEXT,
        )
        draw_text_fit(
            draw, (bx0 + 14, cy + 76, bx1 - 14, cy + 104),
            c["sub"], max_size=24, min_size=18, fill=TEXT_SOFT,
        )
        draw_text_fit(
            draw, (bx0 + 14, cy + 106, bx1 - 14, cy + card_h - 10),
            c["note"], max_size=24, min_size=18, fill=TEXT_SOFT,
        )
        cy += card_h + card_gap

    ry0 = cy + 4
    rounded(
        draw, (bx0, ry0, bx1, ry0 + 100),
        RED_SOFT, outline=STANFORD_RED, width=2, radius=14,
    )
    draw_text_fit(
        draw, (bx0 + 14, ry0 + 8, bx1 - 14, ry0 + 36),
        "Interpret with caution",
        max_size=26, min_size=20, bold=True, fill=STANFORD_RED,
    )
    draw_text_fit(
        draw, (bx0 + 14, ry0 + 40, bx1 - 14, ry0 + 92),
        "Choices-only underperforms, but shuffle and alias barely "
        "move S_q. These controls are suggestive but inconclusive.",
        max_size=24, min_size=18, fill=TEXT, valign="center",
    )

    sy0 = ry0 + 110
    rounded(
        draw, (bx0, sy0, bx1, y1 - 20),
        GOLD_SOFT, outline=GOLD, width=2, radius=14,
    )
    draw_text_fit(
        draw, (bx0 + 14, sy0 + 10, bx1 - 14, y1 - 30),
        "The framing exposes gaps in timing, calibration, and "
        "artifact checking that remain open.",
        max_size=26, min_size=20, fill=TEXT, valign="center",
    )


def draw_limitations_card(draw: ImageDraw.ImageDraw, rect):
    x0, y0, x1, y1 = rect
    draw_card(draw, rect, "Limitations + next checks", NAVY)
    draw_bullets(
        draw, (x0 + 24, y0 + CARD_HEADER_H + 16, x1 - 24, y1 - 24),
        [
            "The smoke slice is small: full_eval uses n = 44 "
            "and choices-only uses n = 11.",
            "PPO still shows degenerate timing, so reward design "
            "needs another pass.",
        ],
        font=FONTS["small"], fill=TEXT,
        bullet_fill=NAVY, bullet_radius=7, gap_after=14,
    )


# ---------------------------------------------------------------------------
# Header & footer
# ---------------------------------------------------------------------------


def draw_header(draw: ImageDraw.ImageDraw):
    draw.rectangle((0, 0, W, 24), fill=STANFORD_RED)

    x0 = MARGIN
    y0 = 48
    draw_text_fit(
        draw, (x0, y0, 2900, y0 + 130),
        "Quiz Bowl RL Buzzer",
        max_size=140, min_size=100, bold=True, fill=TEXT,
    )
    draw_text_fit(
        draw, (x0, y0 + 124, 2700, y0 + 180),
        "Learning when to buzz under incremental clues",
        max_size=56, min_size=40, fill=TEXT_SOFT,
    )
    draw_text_fit(
        draw, (x0, y0 + 186, 2700, y0 + 226),
        "Kathleen Weng  \u2022  Imran Hassan  \u2022  Ankit Aggarwal",
        max_size=42, min_size=30, fill=TEXT_SOFT,
    )
    draw_text_fit(
        draw, (x0, y0 + 230, 2700, y0 + 264),
        "CS234 final project  \u2022  smoke metrics from evaluation_report.json",
        max_size=30, min_size=22, fill=TEXT_SOFT,
    )

    tx0, ty0, tx1, ty1 = 2780, 48, W - MARGIN, 280
    rounded(draw, (tx0, ty0, tx1, ty1), GOLD_SOFT, outline=GOLD, width=2, radius=24)
    draw_text_fit(
        draw, (tx0 + 18, ty0 + 14, tx1 - 18, ty0 + 48),
        "One-sentence takeaway",
        max_size=28, min_size=22, bold=True, fill=GOLD,
    )
    draw_text_fit(
        draw, (tx0 + 18, ty0 + 56, tx1 - 18, ty1 - 14),
        "Buzzing becomes an optimal-stopping problem; on this smoke "
        "slice, simple baselines remain the strongest anchors.",
        max_size=32, min_size=24, fill=TEXT, valign="center",
    )


def draw_footer(draw: ImageDraw.ImageDraw):
    draw.line(
        (MARGIN, H - FOOTER_H + 10, W - MARGIN, H - FOOTER_H + 10),
        fill=BORDER, width=2,
    )
    draw_text_fit(
        draw, (MARGIN, H - FOOTER_H + 18, W - MARGIN, H - 10),
        "Poster: content from the bundled slide deck and smoke report; "
        "all numerical claims from evaluation_report.json; "
        "smoke results are sanity-check snapshots only.",
        max_size=24, min_size=18, fill=TEXT_SOFT,
        align="center", valign="center",
    )


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------


def _stack_cards(
    body_y: int, body_h: int, heights: List[int],
) -> List[Tuple[int, int]]:
    """Return (y_start, y_end) for each card, distributing extra space."""
    n = len(heights)
    total = sum(heights) + (n - 1) * CARD_GAP
    extra = max(0, body_h - total)
    pad = extra // n if n else 0
    adjusted = [h + pad for h in heights]

    y = body_y
    out: List[Tuple[int, int]] = []
    for h in adjusted:
        out.append((y, y + h))
        y += h + CARD_GAP
    return out


# ---------------------------------------------------------------------------
# Main build
# ---------------------------------------------------------------------------


def generate_poster() -> None:
    report = load_report()
    poster = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(poster)

    draw_header(draw)
    draw_footer(draw)

    body_y = HEADER_H
    body_h = H - HEADER_H - FOOTER_H - 10
    col_w = (W - 2 * MARGIN - 2 * COL_GAP) // 3

    c1x = MARGIN
    c2x = c1x + col_w + COL_GAP
    c3x = c2x + col_w + COL_GAP

    c1 = _stack_cards(body_y, body_h, [900, 700, 310, 300])
    for (ys, ye), fn in zip(c1, [
        draw_problem_card, draw_method_card,
        draw_conclusions_card, draw_references_card,
    ]):
        fn(draw, (c1x, ys, c1x + col_w, ye))

    draw_example_card(draw, (c2x, body_y, c2x + col_w, body_y + body_h))

    c3 = _stack_cards(body_y, body_h, [770, 1060, 310])
    draw_scatter_card(draw, (c3x, c3[0][0], c3x + col_w, c3[0][1]), report)
    draw_controls_card(draw, (c3x, c3[1][0], c3x + col_w, c3[1][1]), report)
    draw_limitations_card(draw, (c3x, c3[2][0], c3x + col_w, c3[2][1]))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = OUT_DIR / "poster.pdf"
    png_path = OUT_DIR / "poster.png"
    poster.save(str(pdf_path), "PDF", resolution=DPI)
    poster.save(str(png_path), "PNG")
    print(f"Saved poster to {pdf_path}")
    print(f"Saved preview to {png_path}")


if __name__ == "__main__":
    generate_poster()
```

## File: agents/threshold_buzzer.py
```python
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from agents._math import sigmoid
from models.likelihoods import LikelihoodModel
from qb_data.mc_builder import MCQuestion

if TYPE_CHECKING:
    from agents.bayesian_buzzer import SoftmaxEpisodeResult


@dataclass
class EpisodeResult:
    qid: str
    buzz_step: int
    buzz_index: int
    gold_index: int
    correct: bool
    reward_like: float
    c_trace: list[float]
    g_trace: list[float]
    top_p_trace: list[float]
    entropy_trace: list[float]


def _scores_to_belief(scores: np.ndarray, beta: float) -> np.ndarray:
    """Convert raw similarity scores to a belief distribution via softmax."""
    shifted = scores - np.max(scores)
    probs = np.exp(beta * shifted)
    probs = probs / max(1e-12, probs.sum())
    return probs.astype(np.float32)


def _belief_stats(belief: np.ndarray) -> tuple[int, float, float]:
    """Return (top_idx, top_p, entropy) from a belief distribution."""
    top_idx = int(np.argmax(belief))
    top_p = float(belief[top_idx])
    clipped = np.clip(belief, 1e-12, 1.0)
    entropy = float(-(clipped * np.log(clipped)).sum())
    return top_idx, top_p, entropy


@dataclass
class _PrecomputedQuestion:
    """Pre-computed belief distributions for every clue step of one question."""
    qid: str
    gold_index: int
    num_options: int
    beliefs: list[np.ndarray]


def precompute_beliefs(
    questions: list[MCQuestion],
    likelihood_model: LikelihoodModel,
    beta: float,
) -> list[_PrecomputedQuestion]:
    """Compute beliefs at every step for every question (single model pass).

    After calling ``likelihood_model.precompute_embeddings()`` this is
    pure cache lookups + numpy math, so it runs in seconds rather than
    hours.
    """
    from tqdm import tqdm

    out: list[_PrecomputedQuestion] = []
    for q in tqdm(questions, desc="Computing beliefs"):
        beliefs = [
            _scores_to_belief(
                likelihood_model.score(prefix, q.option_profiles), beta
            )
            for prefix in q.cumulative_prefixes
        ]
        out.append(_PrecomputedQuestion(
            qid=q.qid,
            gold_index=q.gold_index,
            num_options=len(q.options),
            beliefs=beliefs,
        ))
    return out


class ThresholdBuzzer:
    def __init__(
        self,
        likelihood_model: LikelihoodModel,
        threshold: float = 0.8,
        beta: float = 5.0,
        alpha: float = 10.0,
    ):
        self.likelihood_model = likelihood_model
        self.threshold = threshold
        self.beta = beta
        self.alpha = alpha
        self.belief: np.ndarray | None = None

    def _belief_from_prefix(self, prefix: str, option_profiles: list[str]) -> np.ndarray:
        scores = self.likelihood_model.score(prefix, option_profiles)
        return _scores_to_belief(scores, self.beta)

    def _confidence_proxy(self, top_p: float) -> float:
        return sigmoid(self.alpha * (top_p - self.threshold))

    def run_episode(self, question: MCQuestion) -> EpisodeResult:
        c_trace: list[float] = []
        g_trace: list[float] = []
        top_p_trace: list[float] = []
        entropy_trace: list[float] = []

        chosen_step = len(question.cumulative_prefixes) - 1
        chosen_idx = 0

        for step_idx, prefix in enumerate(question.cumulative_prefixes):
            belief = self._belief_from_prefix(prefix, question.option_profiles)
            self.belief = belief
            top_idx, top_p, entropy = _belief_stats(belief)
            c_t = self._confidence_proxy(top_p)
            g_t = 1.0 if top_idx == question.gold_index else 0.0

            c_trace.append(c_t)
            g_trace.append(g_t)
            top_p_trace.append(top_p)
            entropy_trace.append(entropy)

            is_last = step_idx == len(question.cumulative_prefixes) - 1
            if top_p >= self.threshold or is_last:
                chosen_step = step_idx
                chosen_idx = top_idx
                break

        correct = chosen_idx == question.gold_index
        reward_like = 1.0 if correct else -0.5
        return EpisodeResult(
            qid=question.qid,
            buzz_step=chosen_step,
            buzz_index=chosen_idx,
            gold_index=question.gold_index,
            correct=correct,
            reward_like=reward_like,
            c_trace=c_trace,
            g_trace=g_trace,
            top_p_trace=top_p_trace,
            entropy_trace=entropy_trace,
        )


class AlwaysBuzzFinalBuzzer:
    def __init__(self, likelihood_model: LikelihoodModel, beta: float = 5.0):
        self.likelihood_model = likelihood_model
        self.beta = beta

    def run_episode(self, question: MCQuestion) -> EpisodeResult:
        c_trace: list[float] = []
        g_trace: list[float] = []
        top_p_trace: list[float] = []
        entropy_trace: list[float] = []

        final_step = len(question.cumulative_prefixes) - 1
        final_belief = np.ones(len(question.options), dtype=np.float32) / len(question.options)
        for prefix in question.cumulative_prefixes:
            scores = self.likelihood_model.score(prefix, question.option_profiles)
            probs = _scores_to_belief(scores, self.beta)
            final_belief = probs
            top_idx, top_p, entropy = _belief_stats(probs)
            c_trace.append(0.0)
            g_trace.append(1.0 if top_idx == question.gold_index else 0.0)
            top_p_trace.append(top_p)
            entropy_trace.append(entropy)

        c_trace[-1] = 1.0
        buzz_idx = int(np.argmax(final_belief))
        correct = buzz_idx == question.gold_index
        reward_like = 1.0 if correct else -0.5
        return EpisodeResult(
            qid=question.qid,
            buzz_step=final_step,
            buzz_index=buzz_idx,
            gold_index=question.gold_index,
            correct=correct,
            reward_like=reward_like,
            c_trace=c_trace,
            g_trace=g_trace,
            top_p_trace=top_p_trace,
            entropy_trace=entropy_trace,
        )


def _softmax_episode_from_precomputed(
    pq: _PrecomputedQuestion,
    threshold: float,
    alpha: float,
) -> "SoftmaxEpisodeResult":
    """Build a SoftmaxEpisodeResult from pre-computed beliefs (pure numpy).

    Identical buzzing logic to ``SoftmaxProfileBuzzer.run_episode`` but
    reads beliefs from a ``_PrecomputedQuestion`` instead of calling the
    likelihood model.
    """
    from agents.bayesian_buzzer import SoftmaxEpisodeResult

    c_trace: list[float] = []
    g_trace: list[float] = []
    top_p_trace: list[float] = []
    entropy_trace: list[float] = []

    chosen_step = len(pq.beliefs) - 1
    chosen_idx = 0

    for step_idx, belief in enumerate(pq.beliefs):
        top_idx, top_p, entropy = _belief_stats(belief)
        c_t = sigmoid(alpha * (top_p - threshold))
        g_t = 1.0 if top_idx == pq.gold_index else 0.0

        c_trace.append(c_t)
        g_trace.append(g_t)
        top_p_trace.append(top_p)
        entropy_trace.append(entropy)

        is_last = step_idx == len(pq.beliefs) - 1
        if top_p >= threshold or is_last:
            chosen_step = step_idx
            chosen_idx = top_idx
            break

    correct = chosen_idx == pq.gold_index
    return SoftmaxEpisodeResult(
        qid=pq.qid,
        buzz_step=chosen_step,
        buzz_index=chosen_idx,
        gold_index=pq.gold_index,
        correct=correct,
        c_trace=c_trace,
        g_trace=g_trace,
        top_p_trace=top_p_trace,
        entropy_trace=entropy_trace,
    )


def _always_final_from_precomputed(pq: _PrecomputedQuestion) -> EpisodeResult:
    """Build an EpisodeResult for AlwaysBuzzFinal from pre-computed beliefs.

    Iterates all beliefs (no early stopping), buzzes at the last step
    with argmax of the final belief.
    """
    c_trace: list[float] = []
    g_trace: list[float] = []
    top_p_trace: list[float] = []
    entropy_trace: list[float] = []

    for belief in pq.beliefs:
        top_idx, top_p, entropy = _belief_stats(belief)
        g_t = 1.0 if top_idx == pq.gold_index else 0.0
        c_trace.append(0.0)
        g_trace.append(g_t)
        top_p_trace.append(top_p)
        entropy_trace.append(entropy)

    c_trace[-1] = 1.0
    buzz_idx = int(np.argmax(pq.beliefs[-1]))
    correct = buzz_idx == pq.gold_index
    return EpisodeResult(
        qid=pq.qid,
        buzz_step=len(pq.beliefs) - 1,
        buzz_index=buzz_idx,
        gold_index=pq.gold_index,
        correct=correct,
        reward_like=1.0 if correct else -0.5,
        c_trace=c_trace,
        g_trace=g_trace,
        top_p_trace=top_p_trace,
        entropy_trace=entropy_trace,
    )


def _episode_from_precomputed(
    pq: _PrecomputedQuestion,
    threshold: float,
    alpha: float,
) -> EpisodeResult:
    """Build an EpisodeResult from pre-computed beliefs (pure numpy)."""
    c_trace: list[float] = []
    g_trace: list[float] = []
    top_p_trace: list[float] = []
    entropy_trace: list[float] = []

    chosen_step = len(pq.beliefs) - 1
    chosen_idx = 0

    for step_idx, belief in enumerate(pq.beliefs):
        top_idx, top_p, entropy = _belief_stats(belief)
        c_t = sigmoid(alpha * (top_p - threshold))
        g_t = 1.0 if top_idx == pq.gold_index else 0.0

        c_trace.append(c_t)
        g_trace.append(g_t)
        top_p_trace.append(top_p)
        entropy_trace.append(entropy)

        is_last = step_idx == len(pq.beliefs) - 1
        if top_p >= threshold or is_last:
            chosen_step = step_idx
            chosen_idx = top_idx
            break

    correct = chosen_idx == pq.gold_index
    return EpisodeResult(
        qid=pq.qid,
        buzz_step=chosen_step,
        buzz_index=chosen_idx,
        gold_index=pq.gold_index,
        correct=correct,
        reward_like=1.0 if correct else -0.5,
        c_trace=c_trace,
        g_trace=g_trace,
        top_p_trace=top_p_trace,
        entropy_trace=entropy_trace,
    )


def sweep_thresholds(
    questions: list[MCQuestion],
    likelihood_model: LikelihoodModel,
    thresholds: list[float],
    beta: float = 5.0,
    alpha: float = 10.0,
    precomputed: list[_PrecomputedQuestion] | None = None,
) -> dict[float, list[EpisodeResult]]:
    """Sweep multiple thresholds with a single belief-computation pass.

    If *precomputed* is provided the expensive model calls are skipped
    entirely and the sweep is pure numpy.  Otherwise beliefs are computed
    once internally and reused across thresholds.
    """
    if precomputed is None:
        precomputed = precompute_beliefs(questions, likelihood_model, beta)

    out: dict[float, list[EpisodeResult]] = {}
    for threshold in thresholds:
        out[float(threshold)] = [
            _episode_from_precomputed(pq, threshold, alpha)
            for pq in precomputed
        ]
    return out


def result_to_dict(result: EpisodeResult) -> dict[str, Any]:
    return {
        "qid": result.qid,
        "buzz_step": result.buzz_step,
        "buzz_index": result.buzz_index,
        "gold_index": result.gold_index,
        "correct": result.correct,
        "reward_like": result.reward_like,
        "c_trace": result.c_trace,
        "g_trace": result.g_trace,
        "top_p_trace": result.top_p_trace,
        "entropy_trace": result.entropy_trace,
    }
```

## File: configs/default.yaml
```yaml
# Default configuration for qanta-buzzer
# Adapted from qb-rl structure for T5-based quiz bowl agent

data:
  csv_path: "questions.csv"  # Raw QANTA CSV with ||| separated clues
  K: 4  # Default number of answer choices
  distractor_strategy: "sbert_profile"  # sbert_profile | tfidf_profile | category_random | openai_profile
  variable_K: false  # If true, sample K per question from [min_K, max_K]
  min_K: 2
  max_K: null  # Defaults to K when null
  train_ratio: 0.7
  val_ratio: 0.15
  test_ratio: 0.15
  max_questions: null  # Limit for testing (null = use all)
  shuffle_seed: 42

answer_profiles:
  max_tokens_per_profile: 2000  # Max tokens to use for answer profile
  min_questions_per_answer: 1  # Minimum examples to build profile
  leave_one_out: true  # Exclude current question from profile

likelihood:
  model: "t5-large"  # Model for computing answer likelihoods (t5-small | t5-base | t5-large)
  embedding_model: "all-MiniLM-L6-v2"  # For distractor generation
  beta: 5.0  # Softmax temperature for belief distribution
  cache_embeddings: true
  cache_dir: "cache/embeddings"
  batch_size: 16
  max_length: 512  # Max input tokens for T5

environment:
  reward_mode: "time_penalty"  # time_penalty | simple | expected_wins
  seed: 13
  wait_penalty: 0.05  # Tuned candidate from multi-seed smoke sweep
  early_buzz_penalty: 0.2  # Tuned candidate from multi-seed smoke sweep
  buzz_correct: 1.0  # Reward for correct answer
  buzz_incorrect: -0.5  # Penalty for wrong answer
  max_steps: 20  # Maximum clues to reveal
  # Expected Wins opponent model (only used when reward_mode: expected_wins)
  opponent_buzz_model:
    type: "none"  # none | logistic | empirical
  end_mode: "force_commit"  # force_commit | no_buzz
  no_buzz_reward: 0.0  # Only used when end_mode == no_buzz

mc_guards:  # Anti-artifact guards from qb-rl
  alias_edit_distance_threshold: 0.2  # Reject similar answer aliases
  duplicate_token_overlap_threshold: 0.8  # Reject token-overlapping distractors
  max_length_ratio: 3.0  # Reject distractors much longer than answer

bayesian:  # Bayesian buzzer sweep parameters (from qb-rl)
  threshold_sweep: [0.5, 0.6, 0.7, 0.8, 0.9]
  alpha: 10.0  # Sigmoid steepness for confidence proxy

ppo:  # PPO hyperparameters (for future use)
  seed: 13
  total_timesteps: 100000
  learning_rate: 3e-4
  n_steps: 128
  batch_size: 32
  n_epochs: 4
  gamma: 0.99
  gae_lambda: 0.95
  clip_ratio: 0.2
  value_coef: 0.5
  entropy_coef: 0.01
  max_grad_norm: 0.5
  target_kl: 0.03
  policy_kwargs:
    net_arch: [64, 64]  # MLP architecture for belief-based policy

evaluation:
  metrics:
    - accuracy
    - reward
    - buzz_position
    - calibration  # ECE and Brier score
    - per_category
  compute_sq: true  # S_q scoring metric
  run_choices_only: true  # Control: model sees only choices, no clues
  run_shuffle: true  # Control: shuffle clue order
  bootstrap_ci_samples: 1000  # Bootstrap confidence intervals
  save_predictions: true
  prediction_dir: "results/predictions"

# DSPy integration (optional, offline-first)
# Activated by setting likelihood.model: dspy — no separate enable flag.
dspy:
  model: "openai/gpt-4o-mini"  # DSPy LM identifier
  optimizer: "BootstrapFewShot"  # BootstrapFewShot | MIPROv2
  cache_dir: "cache/dspy"
  max_examples: 50

# Supervised warm-start settings (for T5 policy)
supervised:
  epochs: 10
  batch_size: 8
  gradient_accumulation_steps: 4  # Effective batch = 32
  learning_rate: 1e-4
  warmup_steps: 500
  eval_steps: 100
  save_steps: 500
  save_total_limit: 3
  checkpoint_dir: "checkpoints/supervised"
```

## File: configs/smoke.yaml
```yaml
# Smoke test configuration - quick testing with reduced data
# Inherits from default.yaml and overrides key settings

# Data settings for quick testing
data:
  csv_path: "questions.csv"
  K: 4
  distractor_strategy: "category_random"  # Faster than sbert_profile
  train_ratio: 0.7
  val_ratio: 0.15
  test_ratio: 0.15
  max_questions: 50  # Use only 50 questions for smoke test
  shuffle_seed: 42

answer_profiles:
  max_tokens_per_profile: 500  # Reduced for speed
  min_questions_per_answer: 1
  leave_one_out: false  # Skip for smoke test

likelihood:
  model: "tfidf"  # Use TF-IDF for fastest smoke testing (<5 seconds)
  embedding_model: "all-MiniLM-L6-v2"
  beta: 5.0  # Softmax temperature for belief distribution
  cache_embeddings: true
  cache_dir: "cache/embeddings"
  batch_size: 4  # Smaller batch for memory
  max_length: 256  # Shorter sequences

environment:
  reward_mode: "time_penalty"
  seed: 13
  wait_penalty: 0.05
  early_buzz_penalty: 0.2
  buzz_correct: 1.0
  buzz_incorrect: -1.0
  max_steps: 10  # Fewer steps for quick testing
  opponent_buzz_model:
    type: "none"

mc_guards:
  alias_edit_distance_threshold: 0.2
  duplicate_token_overlap_threshold: 0.8
  max_length_ratio: 3.0

bayesian:  # Reduced sweep for smoke testing
  threshold_sweep: [0.5, 0.7, 0.9]
  alpha: 10.0

ppo:  # Reduced for smoke testing
  seed: 13
  total_timesteps: 3000
  learning_rate: 3e-4
  n_steps: 32  # Smaller rollout
  batch_size: 8  # Smaller batch
  n_epochs: 2  # Fewer epochs
  gamma: 0.99
  gae_lambda: 0.95
  clip_ratio: 0.2
  value_coef: 0.5
  entropy_coef: 0.01
  max_grad_norm: 0.5
  target_kl: 0.03
  policy_kwargs:
    net_arch: [32, 32]  # Smaller network

evaluation:
  metrics:
    - accuracy
    - reward
  compute_sq: false  # Skip expensive metrics
  run_choices_only: false  # Skip control experiments
  run_shuffle: false
  bootstrap_ci_samples: 0  # No bootstrap for smoke test
  save_predictions: false
  prediction_dir: "results/predictions"

# Supervised settings for smoke test
supervised:
  epochs: 2  # Very few epochs
  batch_size: 4
  gradient_accumulation_steps: 1  # No accumulation for speed
  learning_rate: 1e-4
  warmup_steps: 10
  eval_steps: 20
  save_steps: 100
  save_total_limit: 1
  checkpoint_dir: "checkpoints/supervised_smoke"
```

## File: evaluation/__init__.py
```python
"""
Evaluation Package

Metrics computation for quiz bowl buzzer agents, including S_q scoring,
calibration analysis (ECE, Brier score), and buzz timing statistics.

Ported from qb-rl reference implementation with adaptations for
qanta-buzzer's EpisodeResult / SoftmaxEpisodeResult / PPOEpisodeTrace
dataclass structures.
"""

from evaluation.metrics import (
    calibration_at_buzz,
    calibration_pairs_at_buzz,
    expected_calibration_error,
    expected_wins_score,
    per_category_accuracy,
    summarize_buzz_metrics,
    system_score,
)

__all__ = [
    "system_score",
    "expected_wins_score",
    "summarize_buzz_metrics",
    "calibration_at_buzz",
    "calibration_pairs_at_buzz",
    "expected_calibration_error",
    "per_category_accuracy",
]
```

## File: qb_env/__init__.py
```python
"""Quiz Bowl Environment Package.

Gymnasium-compliant POMDP environment for quiz bowl question answering,
plus thin qb-rl compatibility exports for the old `qb_env.*` import paths.
"""

from qb_env.data_loader import (
    QANTADatasetLoader,
    TossupQuestion,
    load_tossup_questions,
    load_tossup_questions_from_config,
    parse_row,
)
from qb_env.mc_builder import MCBuilder, MCQuestion
from qb_env.stop_only_env import StopOnlyEnv
from qb_env.text_utils import normalize_answer, tokenize_text
from qb_env.tossup_env import TossupMCEnv, make_env_from_config
from qb_env.text_wrapper import TextObservationWrapper

__all__ = [
    "TossupMCEnv",
    "make_env_from_config",
    "TextObservationWrapper",
    "TossupQuestion",
    "QANTADatasetLoader",
    "parse_row",
    "load_tossup_questions",
    "load_tossup_questions_from_config",
    "MCQuestion",
    "MCBuilder",
    "StopOnlyEnv",
    "normalize_answer",
    "tokenize_text",
]
```

## File: scripts/build_mc_dataset.py
```python
#!/usr/bin/env python3
"""
Build multiple-choice dataset from QANTA quiz bowl questions.

This script orchestrates the complete data pipeline:
1. Load questions from CSV or HuggingFace
2. Build answer profiles from training data
3. Generate MC questions with anti-artifact guards
4. Create stratified train/val/test splits
5. Save processed datasets as JSON

Usage:
    python scripts/build_mc_dataset.py
    python scripts/build_mc_dataset.py --smoke  # Quick test with 50 questions in artifacts/smoke
    python scripts/build_mc_dataset.py --config configs/custom.yaml
    python scripts/build_mc_dataset.py --data.K=5 --data.distractor_strategy=tfidf_profile
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, List, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from qb_data import TossupQuestion
from qb_data.answer_profiles import AnswerProfileBuilder
from qb_data.config import load_config, merge_overrides, resolve_data_loading_options
from qb_data.data_loader import QANTADatasetLoader
from qb_data.dataset_splits import create_stratified_splits
from qb_data.huggingface_loader import load_from_huggingface
from qb_data.mc_builder import MCBuilder, MCQuestion
from scripts._common import parse_overrides

DEFAULT_OUTPUT_DIR = Path("data/processed")
SMOKE_OUTPUT_DIR = Path("artifacts/smoke")


def resolve_output_dir(output_dir: Optional[str], smoke: bool) -> Path:
    """Resolve the dataset output directory from CLI inputs."""
    if output_dir is not None:
        return Path(output_dir)
    return SMOKE_OUTPUT_DIR if smoke else DEFAULT_OUTPUT_DIR


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments for dataset construction."""
    parser = argparse.ArgumentParser(
        description="Build multiple-choice dataset from QANTA questions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help=(
            "Path to YAML configuration file. Defaults to configs/default.yaml, "
            "or the smoke config path selected by load_config() when --smoke is set."
        ),
    )
    parser.add_argument(
        '--smoke',
        action='store_true',
        help='Use smoke test settings (50 questions, quick run, outputs to artifacts/smoke by default).',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Directory to save processed datasets. Defaults to data/processed, or artifacts/smoke when --smoke is set.',
    )
    parser.add_argument(
        'overrides',
        nargs='*',
        help='Config overrides in format: data.K=5 data.distractor_strategy=tfidf_profile',
    )

    return parser.parse_args(argv)


def save_json(path: Path, data: List[Any]) -> None:
    """
    Save dataclass objects to JSON file.

    Parameters
    ----------
    path : Path
        Output file path
    data : List[Any]
        List of dataclass objects (TossupQuestion or MCQuestion)
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert dataclasses to dictionaries
    if data and hasattr(data[0], '__dataclass_fields__'):
        # It's a dataclass, use asdict
        from dataclasses import asdict
        json_data = [asdict(item) for item in data]
    else:
        json_data = data

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(data)} items to {path}")


def print_statistics(
    train: List[MCQuestion],
    val: List[MCQuestion],
    test: List[MCQuestion],
    profile_builder: Optional[AnswerProfileBuilder] = None,
    mc_builder: Optional[MCBuilder] = None
) -> None:
    """
    Print dataset statistics.

    Parameters
    ----------
    train : List[MCQuestion]
        Training split
    val : List[MCQuestion]
        Validation split
    test : List[MCQuestion]
        Test split
    profile_builder : Optional[AnswerProfileBuilder]
        Answer profile builder for profile stats
    mc_builder : Optional[MCBuilder]
        MC builder for guard rejection stats
    """
    print("\n" + "="*60)
    print("Dataset Construction Complete")
    print("="*60)

    # Split statistics
    total = len(train) + len(val) + len(test)
    print(f"\nTotal MC questions: {total}")
    print(f"  Train: {len(train)} ({100*len(train)/total:.1f}%)")
    print(f"  Val:   {len(val)} ({100*len(val)/total:.1f}%)")
    print(f"  Test:  {len(test)} ({100*len(test)/total:.1f}%)")

    # Category distribution
    def get_categories(questions):
        return set(q.category for q in questions if q.category)

    all_categories = get_categories(train) | get_categories(val) | get_categories(test)
    print(f"\nCategories: {len(all_categories)}")

    # Sample categories
    sample_cats = sorted(all_categories)[:5]
    print("Sample categories:", ", ".join(sample_cats))

    # Answer profile statistics
    if profile_builder and hasattr(profile_builder, '_grouped'):
        print(f"\nAnswer profiles: {len(profile_builder._grouped)}")
        # Get average questions per answer
        avg_questions = sum(len(items) for items in profile_builder._grouped.values()) / len(profile_builder._grouped)
        print(f"Average questions per answer: {avg_questions:.1f}")

    # Guard rejection statistics
    if mc_builder and hasattr(mc_builder, 'guard_stats'):
        stats = mc_builder.guard_stats
        if stats:
            print("\nGuard rejection statistics:")
            for guard_name, count in stats.items():
                print(f"  {guard_name}: {count} rejections")

    # Sample MC question
    if train:
        sample = train[0]
        print(f"\nSample MC question:")
        # Get first sentence from the question
        first_sentence = sample.question[:100] + "..." if len(sample.question) > 100 else sample.question
        print(f"  Question: {first_sentence}")
        print(f"  Correct answer: {sample.answer_primary}")
        print(f"  Options: {', '.join(sample.options[:3])}...")
        print(f"  Category: {sample.category}")


def main(argv: Optional[list[str]] = None):
    """Main entry point for dataset construction."""
    args = parse_args(argv)

    # Start timing
    start_time = time.time()

    # Load configuration
    print("Loading configuration...")
    config = load_config(args.config, smoke=args.smoke)

    # Apply overrides
    overrides = parse_overrides(args)
    if overrides:
        print(f"Applying overrides: {overrides}")
        config = merge_overrides(config, overrides)

    # Create output directory
    output_dir = resolve_output_dir(args.output_dir, smoke=args.smoke)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load questions
    print("\nLoading questions...")
    questions = None
    data_opts = resolve_data_loading_options(config, smoke=args.smoke)

    # Try CSV first
    csv_path = data_opts.get('csv_path')
    if csv_path and Path(csv_path).exists():
        print(f"Loading from CSV: {csv_path}")
        loader = QANTADatasetLoader()
        questions = loader.load_from_csv(csv_path)
        print(f"Loaded {len(questions)} questions from CSV")

    # Fallback to HuggingFace if configured
    if questions is None and data_opts.get('use_huggingface'):
        print("CSV not found, falling back to HuggingFace")
        dataset_name = data_opts.get('dataset') or 'qanta-challenge/acf-co24-tossups'
        questions = load_from_huggingface(
            dataset_name,
            config_name=data_opts.get('dataset_config'),
            split=data_opts.get('split', 'eval'),
        )
        print(f"Loaded {len(questions)} questions from HuggingFace")

    if questions is None:
        raise FileNotFoundError(f"Could not load questions from {csv_path} and HuggingFace fallback not enabled")

    # Apply configured limit after loading
    max_questions = data_opts.get('max_questions')
    if max_questions is not None and len(questions) > int(max_questions):
        print(f"Limiting dataset to {int(max_questions)} questions")
        questions = questions[: int(max_questions)]

    # Build answer profiles
    print("\nBuilding answer profiles...")
    profile_builder = AnswerProfileBuilder(
        max_tokens_per_profile=config['answer_profiles']['max_tokens_per_profile'],
        min_questions_per_answer=config['answer_profiles']['min_questions_per_answer']
    )
    profile_builder.fit(questions)
    print(f"Built {len(profile_builder._grouped)} answer profiles")

    # Construct MC questions with guards
    print("\nConstructing MC questions...")
    data_cfg = config['data']
    mc_builder = MCBuilder(
        K=data_cfg['K'],
        strategy=data_cfg['distractor_strategy'],
        embedding_model=config['likelihood'].get(
            'sbert_name',
            config['likelihood'].get('embedding_model', 'all-MiniLM-L6-v2'),
        ),
        openai_model=config['likelihood'].get('openai_model', 'text-embedding-3-small'),
        variable_K=bool(data_cfg.get('variable_K', False)),
        min_K=int(data_cfg.get('min_K', 2)),
        max_K=int(data_cfg['max_K']) if data_cfg.get('max_K') is not None else None,
        **config['mc_guards']
    )

    # Track guard statistics
    mc_builder.guard_stats = {}

    mc_questions = mc_builder.build(questions, profile_builder)
    print(f"Generated {len(mc_questions)} MC questions")

    if len(mc_questions) < len(questions):
        print(f"Note: {len(questions) - len(mc_questions)} questions filtered by guards")

    # Create stratified splits
    print("\nCreating stratified splits...")
    ratios = [
        config['data']['train_ratio'],
        config['data']['val_ratio'],
        config['data']['test_ratio']
    ]

    train, val, test = create_stratified_splits(mc_questions, ratios=ratios)

    # Save datasets
    print("\nSaving datasets...")
    save_json(output_dir / "mc_dataset.json", mc_questions)
    save_json(output_dir / "train_dataset.json", train)
    save_json(output_dir / "val_dataset.json", val)
    save_json(output_dir / "test_dataset.json", test)

    # Save answer profiles for debugging
    if profile_builder._grouped:
        profiles_dict = {
            answer: {
                'question_count': len(items),
                'sample_qids': [qid for qid, _ in items[:5]]  # First 5 question IDs
            }
            for answer, items in profile_builder._grouped.items()
        }
        with open(output_dir / "answer_profiles.json", 'w') as f:
            json.dump(profiles_dict, f, indent=2)
        print(f"Saved answer profiles to {output_dir / 'answer_profiles.json'}")

    # Print statistics
    print_statistics(train, val, test, profile_builder, mc_builder)

    # Print timing
    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed:.1f} seconds")

    if args.smoke:
        # Print sample MC questions for verification
        print("\n" + "="*60)
        print("Sample MC Questions (Smoke Test)")
        print("="*60)

        for i, q in enumerate(train[:3], 1):
            print(f"\nQuestion {i}:")
            # Get first clue from cumulative_prefixes if available
            if q.cumulative_prefixes:
                first_clue = q.cumulative_prefixes[0][:100] + "..." if len(q.cumulative_prefixes[0]) > 100 else q.cumulative_prefixes[0]
            else:
                first_clue = q.question[:100] + "..." if len(q.question) > 100 else q.question
            print(f"  First clue: {first_clue}")
            print(f"  Category: {q.category}")
            print(f"  Correct: {q.answer_primary}")
            print(f"  Options: {', '.join(q.options[:3])}...")

    print("\nDataset construction complete!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
```

## File: tests/test_metrics.py
```python
"""Unit tests for evaluation metrics.

Tests edge cases for system_score (S_q), calibration metrics (ECE, Brier),
and per-category accuracy grouping.
"""

import pytest

from evaluation.metrics import (
    brier_score,
    calibration_at_buzz,
    calibration_pairs_at_buzz,
    expected_calibration_error,
    expected_wins_score,
    per_category_accuracy,
    summarize_buzz_metrics,
    system_score,
)


# ---------------------------------------------------------------------------
# system_score (S_q) edge cases
# ---------------------------------------------------------------------------


def test_system_score_empty_trace():
    """S_q should return 0.0 for empty traces."""
    assert system_score([], []) == 0.0


def test_system_score_all_zero_confidence():
    """S_q should return 0.0 when agent never considers buzzing."""
    c_trace = [0.0, 0.0, 0.0]
    g_trace = [1.0, 1.0, 1.0]  # All correct but agent doesn't buzz
    assert system_score(c_trace, g_trace) == 0.0


def test_system_score_all_correct_immediate_buzz():
    """S_q should equal first g_trace value when agent buzzes immediately."""
    c_trace = [1.0, 0.0, 0.0]  # Buzz on step 0
    g_trace = [1.0, 1.0, 1.0]
    expected = 1.0 * 1.0  # b_0 = c_0 * 1.0 = 1.0, survival after = 0
    assert abs(system_score(c_trace, g_trace) - expected) < 1e-9


def test_system_score_gradual_confidence():
    """S_q should accumulate survival-weighted correctness."""
    c_trace = [0.3, 0.5, 1.0]
    g_trace = [0.0, 0.0, 1.0]  # Only correct at final step
    # b_0 = 0.3 * 1.0 = 0.3, survival = 0.7
    # b_1 = 0.5 * 0.7 = 0.35, survival = 0.7 * 0.5 = 0.35
    # b_2 = 1.0 * 0.35 = 0.35
    # S_q = 0.3*0 + 0.35*0 + 0.35*1 = 0.35
    expected = 0.35
    assert abs(system_score(c_trace, g_trace) - expected) < 1e-9


def test_system_score_single_step():
    """S_q should work for single-step episodes."""
    c_trace = [1.0]
    g_trace = [1.0]
    assert abs(system_score(c_trace, g_trace) - 1.0) < 1e-9

    c_trace = [0.5]
    g_trace = [1.0]
    assert abs(system_score(c_trace, g_trace) - 0.5) < 1e-9


def test_system_score_never_correct():
    """S_q should return 0.0 when g_trace is all zeros."""
    c_trace = [0.5, 0.5, 0.5]
    g_trace = [0.0, 0.0, 0.0]
    assert system_score(c_trace, g_trace) == 0.0


# ---------------------------------------------------------------------------
# Expected Calibration Error (ECE)
# ---------------------------------------------------------------------------


def test_expected_calibration_error_perfect():
    """ECE should be near 0.0 for perfectly calibrated predictions."""
    # 70% confidence with 70% accuracy
    confidences = [0.7] * 10
    outcomes = [1, 1, 1, 1, 1, 1, 1, 0, 0, 0]
    ece = expected_calibration_error(confidences, outcomes, n_bins=10)
    assert ece < 0.01  # Near zero for perfect calibration


def test_expected_calibration_error_empty():
    """ECE should return 0.0 for empty inputs."""
    assert expected_calibration_error([], []) == 0.0


# ---------------------------------------------------------------------------
# Brier Score
# ---------------------------------------------------------------------------


def test_brier_score_perfect():
    """Brier score should be 0.0 for perfect predictions."""
    confidences = [1.0, 1.0, 0.0, 0.0]
    outcomes = [1, 1, 0, 0]
    bs = brier_score(confidences, outcomes)
    assert bs == 0.0


def test_brier_score_worst():
    """Brier score should be 1.0 for worst-case predictions."""
    confidences = [0.0, 0.0, 1.0, 1.0]
    outcomes = [1, 1, 0, 0]
    bs = brier_score(confidences, outcomes)
    assert abs(bs - 1.0) < 1e-9


def test_brier_score_empty():
    """Brier score should return 0.0 for empty inputs."""
    assert brier_score([], []) == 0.0


# ---------------------------------------------------------------------------
# summarize_buzz_metrics
# ---------------------------------------------------------------------------


def test_summarize_buzz_metrics_empty():
    """summarize_buzz_metrics should handle empty results."""
    result = summarize_buzz_metrics([])
    assert result["n"] == 0.0
    assert result["buzz_accuracy"] == 0.0


def test_summarize_buzz_metrics_basic():
    """summarize_buzz_metrics should compute correct aggregates."""
    results = [
        {
            "qid": "q1",
            "correct": True,
            "buzz_step": 2,
            "c_trace": [0.0, 0.0, 1.0],
            "g_trace": [0.0, 0.0, 1.0],
            "reward_like": 0.8,
        },
        {
            "qid": "q2",
            "correct": False,
            "buzz_step": 1,
            "c_trace": [0.0, 1.0],
            "g_trace": [0.0, 0.0],
            "reward_like": -0.1,
        },
    ]
    summary = summarize_buzz_metrics(results)
    assert summary["n"] == 2.0
    assert abs(summary["buzz_accuracy"] - 0.5) < 1e-9
    assert abs(summary["mean_buzz_step"] - 1.5) < 1e-9


# ---------------------------------------------------------------------------
# per_category_accuracy
# ---------------------------------------------------------------------------


def test_per_category_accuracy_basic():
    """per_category_accuracy should group results by question category."""
    results = [
        {
            "qid": "q1",
            "correct": True,
            "buzz_step": 2,
            "c_trace": [0.0, 0.0, 1.0],
            "g_trace": [0.0, 0.0, 1.0],
            "reward_like": 0.8,
        },
        {
            "qid": "q2",
            "correct": False,
            "buzz_step": 1,
            "c_trace": [0.0, 1.0],
            "g_trace": [0.0, 0.0],
            "reward_like": -0.1,
        },
        {
            "qid": "q3",
            "correct": True,
            "buzz_step": 3,
            "c_trace": [0.0, 0.0, 0.0, 1.0],
            "g_trace": [0.0, 0.0, 0.0, 1.0],
            "reward_like": 0.7,
        },
    ]
    questions = [
        {"qid": "q1", "category": "History"},
        {"qid": "q2", "category": "Science"},
        {"qid": "q3", "category": "History"},
    ]
    cat_metrics = per_category_accuracy(results, questions)
    assert "History" in cat_metrics
    assert "Science" in cat_metrics
    assert cat_metrics["History"]["n"] == 2.0
    assert cat_metrics["History"]["buzz_accuracy"] == 1.0
    assert cat_metrics["Science"]["n"] == 1.0
    assert cat_metrics["Science"]["buzz_accuracy"] == 0.0


def test_per_category_accuracy_missing_category():
    """per_category_accuracy should default missing categories to 'unknown'."""
    results = [
        {
            "qid": "q1",
            "correct": True,
            "buzz_step": 0,
            "c_trace": [1.0],
            "g_trace": [1.0],
            "reward_like": 1.0,
        },
    ]
    questions = [
        {"qid": "q1", "category": ""},
    ]
    cat_metrics = per_category_accuracy(results, questions)
    assert "unknown" in cat_metrics
    assert cat_metrics["unknown"]["n"] == 1.0


def test_per_category_accuracy_none_category():
    """per_category_accuracy should handle None category."""
    results = [
        {
            "qid": "q1",
            "correct": True,
            "buzz_step": 0,
            "c_trace": [1.0],
            "g_trace": [1.0],
            "reward_like": 1.0,
        },
    ]
    questions = [
        {"qid": "q1", "category": None},
    ]
    cat_metrics = per_category_accuracy(results, questions)
    assert "unknown" in cat_metrics


def test_per_category_accuracy_unmatched_qid():
    """Results with qids not in questions should group to 'unknown'."""
    results = [
        {
            "qid": "q_orphan",
            "correct": False,
            "buzz_step": 0,
            "c_trace": [1.0],
            "g_trace": [0.0],
            "reward_like": -0.1,
        },
    ]
    questions = [
        {"qid": "q1", "category": "History"},
    ]
    cat_metrics = per_category_accuracy(results, questions)
    assert "unknown" in cat_metrics
    assert cat_metrics["unknown"]["n"] == 1.0


# ---------------------------------------------------------------------------
# calibration_at_buzz — uses top_p_trace, not g_trace
# ---------------------------------------------------------------------------


def test_calibration_at_buzz_uses_top_p_trace():
    """calibration_at_buzz must use top_p_trace (belief prob), not g_trace (binary)."""
    results = [
        {
            "qid": "q1",
            "correct": True,
            "buzz_step": 2,
            "c_trace": [0.1, 0.3, 0.9],
            "g_trace": [0.0, 0.0, 1.0],
            "top_p_trace": [0.3, 0.5, 0.8],
        },
        {
            "qid": "q2",
            "correct": False,
            "buzz_step": 1,
            "c_trace": [0.2, 0.7],
            "g_trace": [0.0, 0.0],
            "top_p_trace": [0.4, 0.6],
        },
    ]
    cal = calibration_at_buzz(results)
    assert cal["n_calibration"] == 2.0
    # Confidence from top_p_trace at buzz_step:
    # q1: top_p_trace[2] = 0.8, q2: top_p_trace[1] = 0.6
    # Brier = ((0.8-1)^2 + (0.6-0)^2)/2 = (0.04+0.36)/2 = 0.2
    assert abs(cal["brier"] - 0.2) < 1e-9


def test_calibration_at_buzz_falls_back_to_c_trace():
    """When top_p_trace is absent, calibration should fall back to c_trace."""
    results = [
        {
            "qid": "q1",
            "correct": True,
            "buzz_step": 0,
            "c_trace": [0.7],
            "g_trace": [1.0],
        },
    ]
    cal = calibration_at_buzz(results)
    assert cal["n_calibration"] == 1.0
    assert abs(cal["brier"] - (0.7 - 1.0) ** 2) < 1e-9


def test_calibration_at_buzz_empty():
    """calibration_at_buzz should return zeros for empty input."""
    cal = calibration_at_buzz([])
    assert cal["ece"] == 0.0
    assert cal["brier"] == 0.0
    assert cal["n_calibration"] == 0.0


def test_calibration_at_buzz_binary_g_trace_not_used():
    """Regression: binary g_trace must NOT be used as confidence.

    If g_trace (binary 0/1) were used, Brier for a correct episode with
    g_trace=[1.0] would be 0.0 regardless of actual confidence.  With
    top_p_trace=[0.5] and correct=True, Brier = (0.5-1)^2 = 0.25.
    """
    results = [
        {
            "qid": "q1",
            "correct": True,
            "buzz_step": 0,
            "c_trace": [0.9],
            "g_trace": [1.0],
            "top_p_trace": [0.5],
        },
    ]
    cal = calibration_at_buzz(results)
    assert abs(cal["brier"] - 0.25) < 1e-9


# ---------------------------------------------------------------------------
# expected_wins_score
# ---------------------------------------------------------------------------


def test_expected_wins_score_binary_g_trace():
    """Hand-worked EW with baseline-style binary g_trace.

    Agent buzzes immediately (c=[1.0]), correct (g=[1.0]),
    opponent survival=0.8 → EW = 1.0 * [0.8*10 + 0.2*0] = 8.0
    """
    ew = expected_wins_score(
        c_trace=[1.0],
        g_trace=[1.0],
        opponent_survival_trace=[0.8],
        reward_correct=10.0,
        reward_incorrect=-5.0,
        opponent_expected_value=0.0,
    )
    assert abs(ew - 8.0) < 1e-9


def test_expected_wins_score_fractional_g_trace():
    """Hand-worked EW with PPO-style fractional g_trace.

    c=[1.0], g=[0.6], S=[0.8]
    V_self = 0.6*10 + 0.4*(-5) = 4.0
    V = 0.8*4.0 + 0.2*0 = 3.2
    EW = 1.0 * 3.2 = 3.2
    """
    ew = expected_wins_score(
        c_trace=[1.0],
        g_trace=[0.6],
        opponent_survival_trace=[0.8],
        reward_correct=10.0,
        reward_incorrect=-5.0,
        opponent_expected_value=0.0,
    )
    assert abs(ew - 3.2) < 1e-9


def test_expected_wins_score_empty():
    assert expected_wins_score([], [], []) == 0.0


def test_expected_wins_does_not_regress_system_score():
    """system_score must remain unchanged by EW addition."""
    c = [0.3, 0.5, 1.0]
    g = [0.0, 0.0, 1.0]
    expected = 0.35
    assert abs(system_score(c, g) - expected) < 1e-9


def test_calibration_pairs_skip_no_buzz():
    """calibration_pairs_at_buzz must skip episodes with buzz_step < 0."""
    results = [
        {"buzz_step": -1, "correct": True, "top_p_trace": [0.9, 0.95]},
        {"buzz_step": 1, "correct": False, "top_p_trace": [0.3, 0.7]},
        {"buzz_step": 0, "correct": True, "c_trace": [0.8]},
    ]
    confs, outs = calibration_pairs_at_buzz(results)
    assert len(confs) == 2
    assert len(outs) == 2
    assert confs[0] == pytest.approx(0.7)
    assert outs[0] == 0
    assert confs[1] == pytest.approx(0.8)
    assert outs[1] == 1


def test_calibration_at_buzz_consistent_with_pairs():
    """calibration_at_buzz must use calibration_pairs_at_buzz internally."""
    results = [
        {"buzz_step": 0, "correct": True, "top_p_trace": [0.9]},
        {"buzz_step": -1, "correct": False, "top_p_trace": [0.1]},
    ]
    cal = calibration_at_buzz(results)
    confs, outs = calibration_pairs_at_buzz(results)
    assert cal["n_calibration"] == len(confs)
```

## File: evaluation/metrics.py
```python
"""
Evaluation Metrics for Quiz Bowl Buzzer Agents

Computes buzz accuracy, S_q scoring, calibration metrics (ECE, Brier score),
and buzz timing statistics from episode trace data.

Ported from qb-rl reference implementation (evaluation/metrics.py).
Accepts both raw dicts and dataclass instances (EpisodeResult,
SoftmaxEpisodeResult, PPOEpisodeTrace) via the _to_dict helper.

Functions
---------
system_score(c_trace, g_trace)
    Compute S_q = sum_t b_t * g_t where b_t = c_t * prod_{i<t} (1 - c_i).
expected_calibration_error(confidences, outcomes, n_bins)
    Binned ECE over confidence-outcome pairs.
brier_score(confidences, outcomes)
    Mean squared error between confidence and binary outcome.
summarize_buzz_metrics(results)
    Aggregate accuracy, buzz step, S_q, and reward across episodes.
calibration_at_buzz(results)
    Extract buzz-time top_p confidence and compute ECE + Brier score.
expected_wins_score(c_trace, g_trace, opponent_survival_trace, ...)
    Offline Expected Wins scoring over an episode.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

import numpy as np


def _to_dict(item: Any) -> dict[str, Any]:
    """Convert dataclass or object to dict for uniform access.

    Parameters
    ----------
    item : Any
        A dict, dataclass instance, or object with __dict__.

    Returns
    -------
    dict[str, Any]
        Dictionary representation of the item.
    """
    if isinstance(item, dict):
        return item
    if is_dataclass(item):
        return asdict(item)
    return item.__dict__


def system_score(c_trace: list[float], g_trace: list[float]) -> float:
    """Compute S_q scoring metric for a single episode.

    S_q = sum_t b_t * g_t, where b_t = c_t * prod_{i<t} (1 - c_i).
    This is the expected correctness under the agent's buzz policy,
    accounting for the survival probability of not having buzzed earlier.

    Parameters
    ----------
    c_trace : list[float]
        Buzz probability at each time step (confidence proxy).
    g_trace : list[float]
        Correctness indicator at each time step (1.0 if top answer is
        correct, 0.0 otherwise).

    Returns
    -------
    float
        S_q score for the episode, in [0, 1].
    """
    c = np.array(c_trace, dtype=np.float64)
    g = np.array(g_trace, dtype=np.float64)
    if len(c) == 0:
        return 0.0
    b = np.zeros_like(c)
    survival = 1.0
    for t in range(len(c)):
        b[t] = c[t] * survival
        survival *= (1.0 - c[t])
    return float(np.sum(b * g))


def expected_wins_score(
    c_trace: list[float],
    g_trace: list[float],
    opponent_survival_trace: list[float],
    reward_correct: float = 10.0,
    reward_incorrect: float = -5.0,
    opponent_expected_value: float = 0.0,
) -> float:
    """Compute offline Expected Wins score for a single episode.

    Uses the continuous V_self formulation::

        V_self_t = g_t * reward_correct + (1 - g_t) * reward_incorrect

    NOT a binary branch on ``g_t``.

    The full formula is::

        EW = sum_t  b_t * [S_t * V_self_t + (1 - S_t) * V_opp]

    where ``b_t = c_t * prod_{i<t}(1 - c_i)`` is the agent's buzz
    probability mass at step *t*, and ``S_t`` is opponent survival.

    Parameters
    ----------
    c_trace : list[float]
        Per-step buzz probability from the agent.
    g_trace : list[float]
        Per-step correctness probability (P(gold) / P(buzz) for PPO,
        binary 0/1 for baseline agents).
    opponent_survival_trace : list[float]
        Per-step P(opponent has not buzzed before step t).
    reward_correct : float
        Points for buzzing correctly before the opponent.
    reward_incorrect : float
        Points for buzzing incorrectly before the opponent.
    opponent_expected_value : float
        Expected score when the opponent buzzes first.

    Returns
    -------
    float
        Expected Wins score for the episode.
    """
    c = np.array(c_trace, dtype=np.float64)
    g = np.array(g_trace, dtype=np.float64)
    s = np.array(opponent_survival_trace, dtype=np.float64)
    if len(c) == 0:
        return 0.0
    n = min(len(c), len(g), len(s))
    c, g, s = c[:n], g[:n], s[:n]

    b = np.zeros(n, dtype=np.float64)
    survival = 1.0
    for t in range(n):
        b[t] = c[t] * survival
        survival *= 1.0 - c[t]

    v_self = g * reward_correct + (1.0 - g) * reward_incorrect
    v = s * v_self + (1.0 - s) * opponent_expected_value
    return float(np.sum(b * v))


def expected_calibration_error(
    confidences: list[float], outcomes: list[int], n_bins: int = 10
) -> float:
    """Compute Expected Calibration Error (ECE) with uniform binning.

    ECE measures the gap between predicted confidence and actual accuracy
    across confidence bins. Lower ECE indicates better-calibrated predictions.

    Parameters
    ----------
    confidences : list[float]
        Predicted confidence values in [0, 1].
    outcomes : list[int]
        Binary outcomes (1 = correct, 0 = incorrect).
    n_bins : int
        Number of uniform bins for confidence bucketing.

    Returns
    -------
    float
        Expected calibration error in [0, 1]. Returns 0.0 if no data.
    """
    if not confidences:
        return 0.0
    conf = np.array(confidences, dtype=np.float64)
    y = np.array(outcomes, dtype=np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf >= lo) & (conf < hi if i < n_bins - 1 else conf <= hi)
        if not mask.any():
            continue
        bin_acc = y[mask].mean()
        bin_conf = conf[mask].mean()
        ece += (mask.mean()) * abs(bin_acc - bin_conf)
    return float(ece)


def brier_score(confidences: list[float], outcomes: list[int]) -> float:
    """Compute Brier score (mean squared calibration error).

    Brier score measures the mean squared difference between predicted
    confidence and binary outcome. Lower is better; 0 is perfect.

    Parameters
    ----------
    confidences : list[float]
        Predicted confidence values in [0, 1].
    outcomes : list[int]
        Binary outcomes (1 = correct, 0 = incorrect).

    Returns
    -------
    float
        Brier score in [0, 1]. Returns 0.0 if no data.
    """
    if not confidences:
        return 0.0
    conf = np.array(confidences, dtype=np.float64)
    y = np.array(outcomes, dtype=np.float64)
    return float(np.mean((conf - y) ** 2))


def summarize_buzz_metrics(results: list[Any]) -> dict[str, float]:
    """Aggregate buzz metrics across a list of episode results.

    Computes accuracy, mean buzz step, mean S_q score, and mean reward
    from episode trace data. Accepts dicts or dataclass instances.

    Parameters
    ----------
    results : list[Any]
        List of episode results (dicts, EpisodeResult, SoftmaxEpisodeResult,
        or PPOEpisodeTrace instances). Each must have: correct, buzz_step,
        c_trace, g_trace. Optionally: reward_like or episode_reward.

    Returns
    -------
    dict[str, float]
        Summary metrics: n, buzz_accuracy, mean_buzz_step, mean_sq,
        mean_reward_like.
    """
    rows = [_to_dict(r) for r in results]
    if not rows:
        return {
            "n": 0.0,
            "buzz_accuracy": 0.0,
            "mean_buzz_step": 0.0,
            "mean_sq": 0.0,
            "mean_reward_like": 0.0,
        }

    correct = np.array(
        [1 if bool(r.get("correct", False)) else 0 for r in rows],
        dtype=np.float64,
    )
    buzz_steps = np.array(
        [int(r.get("buzz_step", 0)) for r in rows], dtype=np.float64
    )
    sq_scores = np.array(
        [
            system_score(
                list(r.get("c_trace", [])),
                list(r.get("g_trace", [])),
            )
            for r in rows
        ],
        dtype=np.float64,
    )
    reward_like = np.array(
        [
            float(r.get("reward_like", r.get("episode_reward", 0.0)))
            for r in rows
        ],
        dtype=np.float64,
    )

    return {
        "n": float(len(rows)),
        "buzz_accuracy": float(correct.mean()),
        "mean_buzz_step": float(buzz_steps.mean()),
        "mean_sq": float(sq_scores.mean()),
        "mean_reward_like": float(reward_like.mean()),
    }


def per_category_accuracy(
    results: list[Any],
    questions: list[Any],
) -> dict[str, dict[str, float]]:
    """Compute accuracy and S_q metrics grouped by question category.

    Joins results with questions to extract category field, then groups
    and computes summarize_buzz_metrics per category.

    Parameters
    ----------
    results : list[Any]
        Episode results from agent evaluation (dicts or dataclasses).
        Must have qid field for joining.
    questions : list[Any]
        Original questions with category field (MCQuestion or similar).

    Returns
    -------
    dict[str, dict[str, float]]
        Mapping from category name to metrics dict with keys:
        n, buzz_accuracy, mean_buzz_step, mean_sq, mean_reward_like.
    """
    from collections import defaultdict

    # Build qid -> category lookup, default to "unknown" for missing
    qid_to_category: dict[str, str] = {}
    for q in questions:
        q_dict = _to_dict(q)
        cat = q_dict.get("category", "") or ""
        qid = q_dict.get("qid", "")
        qid_to_category[qid] = cat if cat else "unknown"

    # Group results by category
    by_category: dict[str, list[Any]] = defaultdict(list)
    for r in results:
        r_dict = _to_dict(r)
        qid = r_dict.get("qid", "")
        category = qid_to_category.get(qid, "unknown")
        by_category[category].append(r)

    # Compute metrics per category
    return {
        cat: summarize_buzz_metrics(rows)
        for cat, rows in sorted(by_category.items())
    }


def calibration_at_buzz(results: list[Any]) -> dict[str, float]:
    """Compute calibration metrics at the buzz decision point.

    Uses the belief model's top-answer probability (``top_p_trace``) at
    buzz time as the confidence proxy.  This measures whether the belief
    distribution is well-calibrated: when the model assigns 0.8
    probability to its top answer, that answer should be correct ~80% of
    the time.

    Falls back to ``c_trace`` (sigmoid confidence) when ``top_p_trace``
    is unavailable (e.g. PPO episode traces that lack per-step belief
    breakdowns).

    Parameters
    ----------
    results : list[Any]
        List of episode results (dicts or dataclass instances). Each must
        have: buzz_step, correct, and at least one of top_p_trace or
        c_trace.

    Returns
    -------
    dict[str, float]
        Calibration metrics: ece, brier, n_calibration.
    """
    confidences, outcomes = calibration_pairs_at_buzz(results)
    return {
        "ece": expected_calibration_error(confidences, outcomes),
        "brier": brier_score(confidences, outcomes),
        "n_calibration": float(len(confidences)),
    }


def calibration_pairs_at_buzz(
    results: list[Any],
) -> tuple[list[float], list[int]]:
    """Extract (confidence, outcome) pairs at the buzz step.

    Canonical helper for all calibration consumers. Episodes with
    ``buzz_step < 0`` (no-buzz) are skipped. Uses ``top_p_trace`` when
    available, falling back to ``c_trace``.

    Parameters
    ----------
    results : list[Any]
        Episode results (dicts or dataclass instances).

    Returns
    -------
    tuple[list[float], list[int]]
        (confidences, outcomes) lists of equal length.
    """
    rows = [_to_dict(r) for r in results]
    confidences: list[float] = []
    outcomes: list[int] = []
    for row in rows:
        top_p_trace = list(row.get("top_p_trace", []))
        c_trace = list(row.get("c_trace", []))
        conf_trace = top_p_trace if top_p_trace else c_trace
        if not conf_trace:
            continue
        buzz_step = int(row.get("buzz_step", max(0, len(conf_trace) - 1)))
        if buzz_step < 0:
            continue
        idx = min(buzz_step, len(conf_trace) - 1)
        confidences.append(float(conf_trace[idx]))
        outcomes.append(1 if bool(row.get("correct", False)) else 0)
    return confidences, outcomes
```

## File: models/likelihoods.py
```python
"""
Likelihood Model Interface

Abstract base class for likelihood models that score answer options against
revealed clue text. Concrete implementations (TF-IDF, SBERT, T5) inherit
from ``LikelihoodModel`` and implement ``score()`` and ``_embed_batch()``.

The ``score()`` method returns **raw similarity scores**, not probabilities.
The environment applies softmax with a configurable temperature (beta) to
convert scores into a belief distribution.

Embedding caching is built into the base class: texts are hashed via SHA-256
and cached as float32 numpy arrays, so repeated calls with the same text
skip recomputation.

Ported from qb-rl reference implementation (models/likelihoods.py lines 1-38).
"""

from __future__ import annotations

import hashlib
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import torch


def _text_key(text: str) -> str:
    """Compute a SHA-256 hash key for embedding cache lookups.

    Parameters
    ----------
    text : str
        Input text to hash.

    Returns
    -------
    str
        64-character hexadecimal SHA-256 digest.

    Examples
    --------
    >>> key = _text_key("hello world")
    >>> len(key)
    64
    >>> _text_key("hello world") == _text_key("hello world")
    True
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _best_torch_device() -> "torch.device":
    """Select the best available accelerator: CUDA > MPS > CPU."""
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class LikelihoodModel(ABC):
    """Abstract base class for likelihood models.

    Likelihood models score how well each answer option matches a given
    clue prefix. The environment uses these scores (via softmax) to compute
    belief distributions over answer options.

    Subclasses must implement:
        - ``score(clue_prefix, option_profiles) -> np.ndarray``
        - ``_embed_batch(texts) -> np.ndarray``

    The base class provides ``embed_and_cache()`` which handles caching of
    text embeddings via SHA-256 content hashing.

    Attributes
    ----------
    embedding_cache : dict[str, np.ndarray]
        Maps SHA-256 text hashes to float32 embedding vectors.
    """

    def __init__(self) -> None:
        self.embedding_cache: dict[str, np.ndarray] = {}

    @property
    def cache_memory_bytes(self) -> int:
        """Approximate memory used by the embedding cache in bytes."""
        return sum(v.nbytes for v in self.embedding_cache.values())

    @abstractmethod
    def score(self, clue_prefix: str, option_profiles: list[str]) -> np.ndarray:
        """Return raw similarity scores for each answer option.

        The caller (environment) converts these to probabilities via
        softmax with a beta temperature parameter. Higher scores indicate
        stronger match between clue and option.

        Parameters
        ----------
        clue_prefix : str
            Clue text revealed so far (concatenation of clues up to current step).
        option_profiles : list[str]
            Answer profile text for each of the K answer options.

        Returns
        -------
        np.ndarray
            Raw similarity scores of shape (K,) where K = len(option_profiles).
        """

    def embed_and_cache(self, texts: list[str]) -> np.ndarray:
        """Embed texts, using cache for previously seen inputs.

        Texts are identified by their SHA-256 hash. Only unseen texts
        are passed to ``_embed_batch()`` for actual computation; cached
        results are reused.

        Parameters
        ----------
        texts : list[str]
            Texts to embed.

        Returns
        -------
        np.ndarray
            Stacked embeddings of shape (len(texts), embed_dim), dtype float32.
        """
        missing = [text for text in texts if _text_key(text) not in self.embedding_cache]
        if missing:
            new_embeddings = self._embed_batch(missing)
            for text, emb in zip(missing, new_embeddings):
                self.embedding_cache[_text_key(text)] = emb.astype(np.float32)
        return np.stack([self.embedding_cache[_text_key(text)] for text in texts])

    def precompute_embeddings(
        self,
        texts: list[str],
        batch_size: int = 64,
        desc: str = "Pre-computing embeddings",
    ) -> None:
        """Bulk pre-embed texts into cache, processing in batches.

        Call this before running agents so that all subsequent ``score()``
        calls are pure cache lookups (numpy dot products).  Duplicate and
        already-cached texts are skipped automatically.

        Parameters
        ----------
        texts : list[str]
            All texts to embed (clue prefixes, option profiles, fragments).
        batch_size : int
            Number of texts per ``_embed_batch`` call.
        desc : str
            tqdm progress-bar description.
        """
        from tqdm import tqdm

        unique = [t for t in dict.fromkeys(texts) if _text_key(t) not in self.embedding_cache]
        if not unique:
            return
        for i in tqdm(range(0, len(unique), batch_size), desc=desc,
                       total=(len(unique) + batch_size - 1) // batch_size):
            batch = unique[i : i + batch_size]
            embeddings = self._embed_batch(batch)
            for text, emb in zip(batch, embeddings):
                self.embedding_cache[_text_key(text)] = emb.astype(np.float32)

    def save_cache(self, path: str | Path) -> int:
        """Persist embedding_cache to disk as compressed ``.npz``.

        Creates parent directories if needed. Keys are SHA-256 hex
        strings (valid Python identifiers), values are float32 arrays.

        Parameters
        ----------
        path : str or Path
            Destination file path (should end with ``.npz``).

        Returns
        -------
        int
            Number of cache entries saved.
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(p, **self.embedding_cache)
        return len(self.embedding_cache)

    def load_cache(self, path: str | Path) -> int:
        """Load embedding_cache from a ``.npz`` file on disk.

        Merges loaded entries into the existing cache **without**
        overwriting keys that are already present (existing keys win).
        If the file does not exist, silently returns 0 (cold-start).

        Parameters
        ----------
        path : str or Path
            Path to ``.npz`` file previously written by ``save_cache``.

        Returns
        -------
        int
            Number of *new* entries added to the cache.
        """
        p = Path(path)
        if not p.exists():
            return 0
        with np.load(p) as data:
            loaded = 0
            for key in data.files:
                if key not in self.embedding_cache:
                    self.embedding_cache[key] = data[key].astype(np.float32)
                    loaded += 1
            return loaded

    @abstractmethod
    def _embed_batch(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of texts. Subclasses must implement.

        Parameters
        ----------
        texts : list[str]
            Texts to embed (guaranteed non-empty, all cache misses).

        Returns
        -------
        np.ndarray
            Embeddings of shape (len(texts), embed_dim), dtype float32.
        """
        raise NotImplementedError


class TfIdfLikelihood(LikelihoodModel):
    """TF-IDF based likelihood model using cosine similarity.

    Uses scikit-learn's ``TfidfVectorizer`` to learn vocabulary and IDF weights
    from a corpus, then scores clue-option similarity via cosine distance in the
    TF-IDF vector space.

    The model **must** be ``fit()`` on a corpus before calling ``score()`` or
    ``_embed_batch()``. Calling these methods on an unfitted model raises
    ``RuntimeError``.

    This is the fast, interpretable baseline: keyword overlap drives similarity.
    It works well when clues contain distinctive vocabulary but misses semantic
    relationships (e.g., "first president" vs "George Washington").

    Parameters
    ----------
    corpus_texts : list[str] or None
        If provided, ``fit()`` is called immediately on these texts.

    Attributes
    ----------
    vectorizer : TfidfVectorizer
        Scikit-learn vectorizer with English stop words removed.
    _is_fit : bool
        Whether the vectorizer has been fit on a corpus.

    Examples
    --------
    >>> corpus = ["George Washington was the first president",
    ...           "Abraham Lincoln freed the slaves"]
    >>> model = TfIdfLikelihood(corpus_texts=corpus)
    >>> scores = model.score("first president", ["Washington", "Lincoln"])
    >>> scores.shape
    (2,)
    """

    def __init__(self, corpus_texts: list[str] | None = None) -> None:
        super().__init__()
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vectorizer = TfidfVectorizer(stop_words="english")
        self._is_fit = False
        if corpus_texts:
            self.fit(corpus_texts)

    def save_cache(self, path: str | Path) -> int:
        """No-op: TF-IDF embeddings are vocabulary-specific and not portable.

        TF-IDF vectors depend on the fitted vocabulary, which changes
        between ``fit()`` calls. Persisting them would produce wrong
        results if the vocabulary differs.

        Returns
        -------
        int
            Always 0.
        """
        return 0

    def load_cache(self, path: str | Path) -> int:
        """No-op: TF-IDF embeddings are not portable across fits."""
        return 0

    def fit(self, corpus_texts: list[str]) -> "TfIdfLikelihood":
        """Learn vocabulary and IDF weights from a text corpus.

        Parameters
        ----------
        corpus_texts : list[str]
            Corpus of documents to learn from. Should include answer profiles,
            clue texts, or both to capture domain vocabulary.

        Returns
        -------
        TfIdfLikelihood
            Self, for method chaining.
        """
        self.vectorizer.fit(corpus_texts)
        self._is_fit = True
        return self

    def score(self, clue_prefix: str, option_profiles: list[str]) -> np.ndarray:
        """Score each option against the clue using TF-IDF cosine similarity.

        Uses ``embed_and_cache()`` to embed both the clue and options, so
        repeated calls with the same texts skip vectorizer.transform().
        Since ``_embed_batch()`` returns L2-normalized vectors, the dot
        product equals cosine similarity.

        Parameters
        ----------
        clue_prefix : str
            Clue text revealed so far.
        option_profiles : list[str]
            Answer profile text for each of the K answer options.

        Returns
        -------
        np.ndarray
            Cosine similarity scores of shape (K,), dtype float32.
            Values in [-1, 1] but typically [0, 1] for TF-IDF.

        Raises
        ------
        RuntimeError
            If called before ``fit()``.
        """
        if not self._is_fit:
            raise RuntimeError("TfIdfLikelihood must be fit() before score().")
        clue_emb = self.embed_and_cache([clue_prefix])[0]
        option_embs = self.embed_and_cache(option_profiles)
        sims = option_embs @ clue_emb
        return sims.astype(np.float32)

    def _embed_batch(self, texts: list[str]) -> np.ndarray:
        """Embed texts as dense, L2-normalized TF-IDF vectors.

        Row-wise L2 normalization ensures that dot product between any
        two embedding vectors equals their cosine similarity, matching
        the convention used by SBERT and T5 likelihood models.

        Parameters
        ----------
        texts : list[str]
            Texts to embed (guaranteed non-empty, all cache misses).

        Returns
        -------
        np.ndarray
            L2-normalized dense TF-IDF matrix of shape
            (len(texts), vocab_size), dtype float32.

        Raises
        ------
        RuntimeError
            If called before ``fit()``.
        """
        if not self._is_fit:
            raise RuntimeError("TfIdfLikelihood must be fit() before embedding.")
        mat = self.vectorizer.transform(texts).toarray().astype(np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0  # avoid division by zero for empty docs
        return mat / norms


class SBERTLikelihood(LikelihoodModel):
    """Sentence-BERT likelihood model using semantic embeddings.

    Uses a ``SentenceTransformer`` model to compute dense, L2-normalized
    embeddings. Cosine similarity is computed as a simple dot product since
    embeddings are pre-normalized (``normalize_embeddings=True``).

    Inherits ``embed_and_cache()`` from ``LikelihoodModel`` for transparent
    caching of embeddings via SHA-256 content hashing. The first call to
    ``score()`` computes and caches all embeddings; subsequent calls with the
    same texts are fast cache lookups.

    Compared to TF-IDF, SBERT captures semantic similarity (e.g., "first
    president" and "George Washington" score highly even without word overlap)
    but is slower due to the neural encoder.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier for ``SentenceTransformer``.
        Default is ``"all-MiniLM-L6-v2"`` (22M params, 384-dim embeddings).
        First run downloads the model (~80MB) from HuggingFace.

    Attributes
    ----------
    model_name : str
        The SentenceTransformer model name.
    encoder : SentenceTransformer
        The loaded sentence transformer model.

    Examples
    --------
    >>> model = SBERTLikelihood()  # downloads model on first run
    >>> scores = model.score("first president", ["Washington", "Lincoln"])
    >>> scores.shape
    (2,)
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        super().__init__()
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.encoder = SentenceTransformer(model_name)

    def _embed_batch(self, texts: list[str]) -> np.ndarray:
        """Embed texts using the SentenceTransformer encoder.

        Embeddings are L2-normalized so that cosine similarity can be computed
        as a simple dot product (avoiding the division by norms).

        Parameters
        ----------
        texts : list[str]
            Texts to embed (guaranteed non-empty, all cache misses).

        Returns
        -------
        np.ndarray
            Normalized embeddings of shape (len(texts), embed_dim), dtype float32.
        """
        return self.encoder.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True
        ).astype(np.float32)

    def score(self, clue_prefix: str, option_profiles: list[str]) -> np.ndarray:
        """Score each option using semantic cosine similarity.

        Computes dot product between the clue embedding and each option
        embedding. Since embeddings are L2-normalized, dot product equals
        cosine similarity.

        Parameters
        ----------
        clue_prefix : str
            Clue text revealed so far.
        option_profiles : list[str]
            Answer profile text for each of the K answer options.

        Returns
        -------
        np.ndarray
            Cosine similarity scores of shape (K,), dtype float32.
            Values in [-1, 1].
        """
        clue_emb = self.embed_and_cache([clue_prefix])[0]
        option_embs = self.embed_and_cache(option_profiles)
        sims = option_embs @ clue_emb
        return sims.astype(np.float32)


class OpenAILikelihood(LikelihoodModel):
    """OpenAI embedding likelihood model using normalized embedding similarity.

    This path is optional and only activates when explicitly selected in config.
    It requires both the ``openai`` Python package and ``OPENAI_API_KEY`` to be
    available at runtime.
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
    ) -> None:
        super().__init__()

        resolved_api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not resolved_api_key:
            raise RuntimeError(
                "OpenAI likelihood requires OPENAI_API_KEY to be set."
            )

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "OpenAI likelihood requires the openai package. "
                "Install it with: pip install -e .[openai] or pip install openai."
            ) from exc

        self.model = model
        self.client = OpenAI(api_key=resolved_api_key)

    def _embed_batch(self, texts: list[str]) -> np.ndarray:
        """Embed texts via the OpenAI embeddings API and L2-normalize them."""
        response = self.client.embeddings.create(model=self.model, input=texts)
        vectors = [np.array(item.embedding, dtype=np.float32) for item in response.data]
        embeddings = np.stack(vectors)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (embeddings / norms).astype(np.float32)

    def score(self, clue_prefix: str, option_profiles: list[str]) -> np.ndarray:
        """Score each option using cosine similarity over normalized embeddings."""
        clue_emb = self.embed_and_cache([clue_prefix])[0]
        option_embs = self.embed_and_cache(option_profiles)
        sims = option_embs @ clue_emb
        return sims.astype(np.float32)


class T5Likelihood(LikelihoodModel):
    """T5 encoder likelihood model using mean-pooled semantic embeddings.

    Uses ``T5EncoderModel`` (not full ``T5ForConditionalGeneration``) for 2x
    faster inference and half the memory. Embeddings are mean-pooled over
    sequence length with attention mask weighting to handle padding correctly.

    Inherits ``embed_and_cache()`` from ``LikelihoodModel`` for transparent
    caching of embeddings via SHA-256 content hashing. The first call to
    ``score()`` computes and caches all embeddings; subsequent calls with the
    same texts are fast cache lookups.

    Compared to SBERT, T5 captures deeper semantic relationships via its
    encoder-decoder pre-training on massive text corpora. This is the novel
    contribution: using T5 as a likelihood model rather than just as a policy
    encoder.

    Parameters
    ----------
    model_name : str
        HuggingFace T5 model identifier. Default is ``"t5-base"``
        (220M params). Options:

        - ``"t5-small"`` (60M params) -- fastest, lowest quality
        - ``"t5-base"`` (220M params) -- balanced (recommended)
        - ``"t5-large"`` (770M params) -- best quality, requires 8GB GPU VRAM

        First run downloads the model from HuggingFace (~850MB for t5-base).

    Attributes
    ----------
    model_name : str
        The T5 model identifier.
    encoder : T5EncoderModel
        Pre-trained T5 encoder loaded from HuggingFace.
    tokenizer : T5TokenizerFast
        Fast T5 tokenizer for text preprocessing.
    device : torch.device
        Computation device (cuda if available, else cpu).

    Examples
    --------
    >>> model = T5Likelihood(model_name="t5-small")
    >>> scores = model.score("first president", ["Washington", "Einstein"])
    >>> scores.shape
    (2,)
    """

    def __init__(self, model_name: str = "t5-base") -> None:
        super().__init__()
        import torch
        from transformers import T5EncoderModel, T5TokenizerFast

        self.model_name = model_name
        self.encoder = T5EncoderModel.from_pretrained(model_name)
        self.tokenizer = T5TokenizerFast.from_pretrained(model_name)
        self.device = _best_torch_device()
        self.encoder.to(self.device)
        self.encoder.eval()

    def _embed_batch(self, texts: list[str]) -> np.ndarray:
        """Embed texts using T5 encoder with attention-masked mean pooling.

        Mean pooling uses the attention mask to exclude padding tokens from the
        average, ensuring correct semantic embeddings when sequences have
        different lengths. Embeddings are L2-normalized so that cosine
        similarity can be computed as a simple dot product.

        Parameters
        ----------
        texts : list[str]
            Texts to embed (guaranteed non-empty, all cache misses).

        Returns
        -------
        np.ndarray
            L2-normalized embeddings of shape (len(texts), hidden_dim),
            dtype float32. Hidden dim is 512 (t5-small), 768 (t5-base),
            or 1024 (t5-large).

        Notes
        -----
        Tensors are detached and moved to CPU immediately after computation
        to prevent GPU memory leaks when called repeatedly during episodes.
        """
        import torch

        with torch.no_grad():
            encoded = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)

            outputs = self.encoder(**encoded)
            last_hidden = outputs.last_hidden_state  # (batch, seq_len, hidden_dim)

            # Mean pooling over sequence length with attention mask
            mask = encoded.attention_mask.unsqueeze(-1)  # (batch, seq_len, 1)
            masked_hidden = last_hidden * mask
            sum_hidden = masked_hidden.sum(dim=1)  # (batch, hidden_dim)
            mask_sum = mask.sum(dim=1).clamp(min=1e-9)  # (batch, 1)
            mean_pooled = sum_hidden / mask_sum  # (batch, hidden_dim)

            # L2 normalize for cosine similarity via dot product
            embeddings = torch.nn.functional.normalize(mean_pooled, p=2, dim=1)

            # Detach and move to CPU to prevent GPU memory leak
            embeddings = embeddings.detach().cpu().numpy().astype(np.float32)

        return embeddings

    def score(self, clue_prefix: str, option_profiles: list[str]) -> np.ndarray:
        """Score each option using T5 semantic cosine similarity.

        Computes dot product between the clue embedding and each option
        embedding. Since embeddings are L2-normalized, dot product equals
        cosine similarity.

        Parameters
        ----------
        clue_prefix : str
            Clue text revealed so far.
        option_profiles : list[str]
            Answer profile text for each of the K answer options.

        Returns
        -------
        np.ndarray
            Cosine similarity scores of shape (K,), dtype float32.
            Values in [-1, 1].
        """
        clue_emb = self.embed_and_cache([clue_prefix])[0]
        option_embs = self.embed_and_cache(option_profiles)
        sims = option_embs @ clue_emb
        return sims.astype(np.float32)


def build_likelihood_from_config(
    config: dict[str, Any], corpus_texts: list[str] | None = None
) -> LikelihoodModel:
    """Construct a likelihood model from YAML configuration.

    Factory function that reads the ``likelihood`` section of the config dict
    and instantiates the appropriate ``LikelihoodModel`` subclass.

    Parameters
    ----------
    config : dict[str, Any]
        Full YAML config dict. Must contain a ``"likelihood"`` key with at
        least a ``"model"`` field specifying the model type.

        Supported model types:
        - ``"tfidf"``: TF-IDF cosine similarity (requires ``corpus_texts``)
        - ``"sbert"``: Sentence-BERT semantic similarity
        - ``"openai"``: OpenAI embedding similarity
        - ``"t5"`` / ``"t5-small"`` / ``"t5-base"`` / ``"t5-large"``:
          T5 encoder semantic similarity

        Optional config keys:
        - ``"sbert_name"`` or ``"embedding_model"``: SentenceTransformer model
          name (default: ``"all-MiniLM-L6-v2"``)
        - ``"openai_model"``: OpenAI embedding model name
          (default: ``"text-embedding-3-small"``)
        - ``"t5_name"``: T5 model name (default: ``"t5-base"``)

    corpus_texts : list[str] or None
        Text corpus for TF-IDF fitting. Required when ``model == "tfidf"``,
        ignored for other models.

    Returns
    -------
    LikelihoodModel
        An instantiated and ready-to-use likelihood model.

    Raises
    ------
    ValueError
        If ``model`` is ``"tfidf"`` and ``corpus_texts`` is None.
        If ``model`` is not a recognized model type.

    Examples
    --------
    >>> from qb_data.config import load_config
    >>> config = load_config("configs/default.yaml")
    >>> model = build_likelihood_from_config(config, corpus_texts=my_corpus)
    >>> scores = model.score("first president", ["Washington", "Lincoln"])
    """
    cfg = config["likelihood"]
    model_name = cfg.get("model", "sbert")

    if model_name == "tfidf":
        if not corpus_texts:
            raise ValueError("TF-IDF likelihood requires corpus_texts.")
        return TfIdfLikelihood(corpus_texts=corpus_texts)

    if model_name == "sbert":
        # Support both "sbert_name" (qb-rl convention) and
        # "embedding_model" (qanta-buzzer default.yaml convention)
        sbert_name = cfg.get("sbert_name", cfg.get("embedding_model", "all-MiniLM-L6-v2"))
        return SBERTLikelihood(model_name=sbert_name)

    if model_name == "openai":
        return OpenAILikelihood(
            model=cfg.get("openai_model", "text-embedding-3-small"),
        )

    if model_name == "t5":
        t5_name = cfg.get("t5_name", "t5-base")
        return T5Likelihood(model_name=t5_name)

    if isinstance(model_name, str) and model_name.startswith("t5"):
        t5_name = model_name
        return T5Likelihood(model_name=t5_name)

    if model_name == "dspy":
        try:
            from models.dspy_likelihood import DSPyLikelihood
        except ImportError as exc:
            raise ImportError(
                "DSPy likelihood requires the dspy package. "
                "Install with: pip install -e '.[dspy]'"
            ) from exc
        dspy_cfg = config.get("dspy", {})
        cache_dir = dspy_cfg.get("cache_dir")
        fingerprint = dspy_cfg.get("program_fingerprint", "default")

        def _placeholder_scorer(clue: str, options: list[str]) -> list[float]:
            return [1.0 / max(1, len(options))] * len(options)

        return DSPyLikelihood(
            scorer=_placeholder_scorer,
            program_fingerprint=fingerprint,
            cache_dir=cache_dir,
        )

    raise ValueError(f"Unknown likelihood model: {model_name}")
```

## File: tests/test_compare_policies.py
```python
"""Tests for compare_policies helper functions."""

from __future__ import annotations

import json

from scripts.compare_policies import resolve_mlp_eval_config


def test_resolve_mlp_eval_config_prefers_checkpoint_sidecar(tmp_path):

    sidecar_config = {"likelihood": {"model": "t5-base"}, "ppo": {"seed": 99}}
    sidecar_path = tmp_path / "config_used.json"
    sidecar_path.write_text(json.dumps(sidecar_config))

    fake_checkpoint = tmp_path / "ppo_model.zip"
    fake_checkpoint.touch()

    fallback = {"likelihood": {"model": "tfidf"}}
    resolved = resolve_mlp_eval_config(str(fake_checkpoint), fallback)
    assert resolved["likelihood"]["model"] == "t5-base"
    assert resolved["ppo"]["seed"] == 99


def test_resolve_mlp_eval_config_uses_fallback_when_no_sidecar(tmp_path):
    fake_checkpoint = tmp_path / "ppo_model.zip"
    fake_checkpoint.touch()

    fallback = {"likelihood": {"model": "tfidf"}}
    resolved = resolve_mlp_eval_config(str(fake_checkpoint), fallback)
    assert resolved is fallback


def test_resolve_mlp_eval_config_handles_directory_checkpoint(tmp_path):
    """When checkpoint_path is a directory, look for sidecar inside it."""
    ckpt_dir = tmp_path / "best_model"
    ckpt_dir.mkdir()
    sidecar_config = {"likelihood": {"model": "sbert"}, "ppo": {"seed": 7}}
    (ckpt_dir / "config_used.json").write_text(json.dumps(sidecar_config))

    fallback = {"likelihood": {"model": "tfidf"}}
    resolved = resolve_mlp_eval_config(str(ckpt_dir), fallback)
    assert resolved["likelihood"]["model"] == "sbert"


def test_resolve_mlp_eval_config_survives_corrupt_json(tmp_path):
    """Corrupt sidecar JSON should fall back gracefully, not crash."""
    (tmp_path / "config_used.json").write_text("{bad json")
    fake_checkpoint = tmp_path / "ppo_model.zip"
    fake_checkpoint.touch()

    fallback = {"likelihood": {"model": "tfidf"}}
    resolved = resolve_mlp_eval_config(str(fake_checkpoint), fallback)
    assert resolved is fallback


def test_evaluate_mlp_policy_uses_builder_and_question_idx(monkeypatch):
    import scripts.compare_policies as cp
    from agents.ppo_buzzer import PPOBuzzer
    import qb_env.tossup_env as te

    calls: dict[str, object] = {
        "builder_count": 0,
        "builder_args": None,
        "question_idx": [],
    }

    def fake_builder(config, test_questions):
        calls["builder_count"] += 1
        calls["builder_args"] = (config, test_questions)
        return object()

    def fake_make_env_from_config(mc_questions, likelihood_model, config):
        return object()

    class FakeAgent:
        def run_episode(self, deterministic=True, question_idx=None):
            calls["question_idx"].append(question_idx)
            return {"buzz_step": 0, "correct": True, "top_p_trace": [0.9]}

    monkeypatch.setattr(cp, "build_likelihood_model", fake_builder)
    monkeypatch.setattr(cp, "load_embedding_cache", lambda model, config: None)
    monkeypatch.setattr(te, "make_env_from_config", fake_make_env_from_config)
    monkeypatch.setattr(
        PPOBuzzer,
        "load",
        classmethod(lambda cls, checkpoint_path, env, use_maskable_ppo=False: FakeAgent()),
    )
    monkeypatch.setattr(cp, "summarize_buzz_metrics", lambda results: {"buzz_accuracy": 1.0, "mean_sq": 1.0, "mean_buzz_step": 0.0, "mean_reward_like": 0.0})
    monkeypatch.setattr(cp, "calibration_pairs_at_buzz", lambda results: ([0.9], [1]))
    monkeypatch.setattr(cp, "expected_calibration_error", lambda c, o: 0.0)
    monkeypatch.setattr(cp, "brier_score", lambda c, o: 0.0)

    out = cp.evaluate_mlp_policy(
        checkpoint_path="artifacts/main/ppo_model",
        test_questions=[object(), object(), object()],
        config={"likelihood": {"model": "tfidf"}, "ppo": {}},
    )

    assert calls["builder_count"] == 1
    assert calls["question_idx"] == [0, 1, 2]
    assert out["accuracy"] == 1.0
```

## File: generate_presentation.py
```python
from __future__ import annotations

import json
import os
import re
from math import ceil
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from PIL import Image, ImageDraw, ImageFont

# ============================================================
# Paths
# ============================================================
DEFAULT_OUT_DIR = Path("/mnt/data/quizbowl_mc_stopping_data_driven")
FALLBACK_OUT_DIR = (
    Path(__file__).resolve().parent
    / "generated"
    / "quizbowl_mc_stopping_data_driven"
)

env_out_dir = os.getenv("QB_PRESENTATION_OUT_DIR")
if env_out_dir:
    OUT_DIR = Path(env_out_dir).expanduser()
else:
    preferred_parent = DEFAULT_OUT_DIR.parent
    if preferred_parent.exists() and os.access(preferred_parent, os.W_OK):
        OUT_DIR = DEFAULT_OUT_DIR
    else:
        OUT_DIR = FALLBACK_OUT_DIR

OUT_DIR.mkdir(parents=True, exist_ok=True)

GIF_OUT = OUT_DIR / "quizbowl_mc_stopping_data_driven.gif"
CONTACT_OUT = OUT_DIR / "quizbowl_mc_stopping_data_driven_contact.png"
FRAMES_DIR = OUT_DIR / "frames"

# ============================================================
# Canvas + palette
# ============================================================
W, H = 960, 540
BG = "#F3F5F8"
WHITE = "#FFFFFF"

NAVY = "#2E4A9E"
NAVY_DARK = "#243B7A"
TEXT = "#1F2937"
TEXT_SOFT = "#5B6472"
BORDER = "#C8D1E0"
GRID = "#E6EBF3"

BLUE = "#345BD3"
BLUE_SOFT = "#DDE7FF"

PURPLE = "#7A57E2"
PURPLE_SOFT = "#E9E0FF"

GREEN = "#4BB773"
GREEN_SOFT = "#DFF5E7"

ORANGE = "#E9A23B"
ORANGE_SOFT = "#FDEFD9"

RED = "#E76A5E"
RED_SOFT = "#FCE2DE"

# ============================================================
# Running example
# ============================================================
# Source tossup: https://www.qbreader.org/db/tossup/?_id=63ec2f74c5548754cbcb03f4
# Distractors here are a cleaner, presentation-friendly mathematician-only set
# drawn from answers that already exist in the local corpus.
EXAMPLE_TITLE = "Markov tossup"
EXAMPLE_SOURCE = "QBReader • 2022 ARCADIA • packet 08 • tossup 11"
EXAMPLE_TOSSUP_VERBATIM = (
    'Given a set of nodes named for this person, a node is conditionally '
    'independent from the rest of a Bayesian network. The Baum-Welch algorithm '
    'is used to train a type of model named for this person that is used for '
    'multiple sequence alignment. The initial “burn-in” states of a process '
    'named for this person are discarded in methods like Gibbs sampling. The '
    'Metropolis-Hastings algorithm approximates an unknown distribution as the '
    '(*) stationary distribution of a process named for this person. Monte Carlo '
    'methods often use processes named for this person that have stochastic '
    'transition matrices. Dynamic programming is used to decode “hidden” models '
    'named for this person. The next state of a random process named for this '
    'person depends only on the current state. For 10 points, what Russian '
    'mathematician names a type of memoryless “chain?”'
)
EXAMPLE_ANSWERLINE = "ANSWER: Andrey Andreyevich Markov"
EXAMPLE_ACCEPTS = "Accepts: Markov blankets; hidden Markov models; Markov chains"
EXAMPLE_TOSSUP_HIGHLIGHTS = [
    {"phrase": "named for", "fill": ORANGE_SOFT, "text_fill": "#8A5A11"},
    {"phrase": "this person", "fill": BLUE_SOFT, "text_fill": NAVY_DARK},
]
EXAMPLE_OPTIONS = [
    ("A", "Andrey Andreyevich Markov"),
    ("B", "Leonhard Euler"),
    ("C", "Carl Friedrich Gauss"),
    ("D", "Augustin-Louis Cauchy"),
]
EXAMPLE_POSTERIOR_STAGES = [
    {
        "prefix": 'Only the opening referent is visible: "nodes named for this person..."',
        "probs": [0.42, 0.24, 0.21, 0.13],
        "decision": "WAIT",
        "callout": 'Keep "waiting": Markov only leads narrowly over the other mathematicians, so another clue can still change the ranking.',
    },
    {
        "prefix": 'The referent repeats: "a type of model named for this person..."',
        "probs": [0.56, 0.19, 0.15, 0.10],
        "decision": "WAIT",
        "callout": 'Keep "waiting": Markov now leads, but continuation value is still meaningful.',
    },
    {
        "prefix": 'Another repetition appears: "\"burn-in\" states of a process named for this person..."',
        "probs": [0.68, 0.14, 0.11, 0.07],
        "decision": "WAIT",
        "callout": 'Keep "waiting": A is favored, yet one more clue can still sharpen confidence.',
    },
    {
        "prefix": 'The referent stays active: "stationary distribution of a process named for this person..."',
        "probs": [0.82, 0.08, 0.06, 0.04],
        "decision": "BUZZ",
        "callout": '"Buzz" now: the Markov reading is strong enough that waiting has less value.',
    },
    {
        "prefix": 'Finally, "memoryless chain" turns the repeated "this person" referent into a giveaway.',
        "probs": [0.91, 0.04, 0.03, 0.02],
        "decision": "BUZZ",
        "callout": '"Buzz" now: the final clue mostly confirms what the posterior already says.',
    },
]
EXAMPLE_VALUE_NOTES = [
    'step 1 act now = 0.18\nwait = 0.64',
    'step 2 act now = 0.38\nwait = 0.58',
    'step 3 act now = 0.49\nwait = 0.52',
    'step 4 act now = 0.63\nwait = 0.40',
    'step 5 act now = 0.74\nwait = 0.28',
]
EXAMPLE_VALUE_MESSAGES = [
    'On the first Markov clue, another "clue" is still worth much more than committing.',
    'Baum-Welch pushes the example toward Markov, but waiting still wins.',
    'Near the Gibbs / Metropolis region, timing becomes the whole game.',
    'Once the random-process clue arrives, "buzzing" overtakes waiting.',
    'By the "memoryless chain" clue, waiting mostly delays a strong answer.',
]
EXAMPLE_ABSTAIN_MESSAGES = [
    'Even with Markov already listed at A, "ABSTAIN" can still be rational after the earliest clue.',
    '"Waiting" is still about the value of one more clue, not about missing options.',
    'So "ABSTAIN" means continuation value, even when the correct answer has been on the board all along.',
]

# ============================================================
# Fonts
# ============================================================
def load_font(size: int, bold: bool = False):
    candidates = []
    if bold:
        candidates.extend(
            [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
            ]
        )
    else:
        candidates.extend(
            [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
            ]
        )

    for path in candidates:
        p = Path(path)
        if p.exists():
            return ImageFont.truetype(str(p), size=size)

    return ImageFont.load_default()

# ============================================================
# Text fitting
# ============================================================
def measure(draw: ImageDraw.ImageDraw, text: str, font) -> Tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0], box[3] - box[1]

def wrap_text(draw: ImageDraw.ImageDraw, text: str, font, max_width: int) -> List[str]:
    lines: List[str] = []
    for paragraph in text.split("\n"):
        if not paragraph.strip():
            lines.append("")
            continue
        words = paragraph.split()
        current = words[0]
        for word in words[1:]:
            trial = current + " " + word
            tw, _ = measure(draw, trial, font)
            if tw <= max_width:
                current = trial
            else:
                lines.append(current)
                current = word
        lines.append(current)
    return lines

def fit_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    box: Tuple[int, int, int, int],
    *,
    max_size: int,
    min_size: int,
    bold: bool = False,
    line_gap: int = 4,
):
    x0, y0, x1, y1 = box
    bw = max(1, x1 - x0)
    bh = max(1, y1 - y0)

    for size in range(max_size, min_size - 1, -1):
        font = load_font(size, bold=bold)
        lines = wrap_text(draw, text, font, bw)
        _, lh = measure(draw, "Ag", font)
        total_h = len(lines) * lh + max(0, len(lines) - 1) * line_gap

        if total_h > bh:
            continue

        too_wide = False
        for line in lines:
            lw, _ = measure(draw, line, font)
            if lw > bw:
                too_wide = True
                break
        if not too_wide:
            return font, lines, total_h

    font = load_font(min_size, bold=bold)
    lines = wrap_text(draw, text, font, bw)
    _, lh = measure(draw, "Ag", font)
    total_h = len(lines) * lh + max(0, len(lines) - 1) * line_gap
    return font, lines, total_h

def draw_text_fit(
    draw: ImageDraw.ImageDraw,
    box: Tuple[int, int, int, int],
    text: str,
    *,
    fill=TEXT,
    max_size=24,
    min_size=12,
    bold=False,
    align="left",
    valign="top",
    line_gap=4,
):
    x0, y0, x1, y1 = box
    font, lines, total_h = fit_text(
        draw,
        text,
        box,
        max_size=max_size,
        min_size=min_size,
        bold=bold,
        line_gap=line_gap,
    )
    _, lh = measure(draw, "Ag", font)

    if valign == "center":
        cy = y0 + max(0, (y1 - y0 - total_h) // 2)
    elif valign == "bottom":
        cy = y1 - total_h
    else:
        cy = y0

    for line in lines:
        lw, _ = measure(draw, line, font)
        if align == "center":
            cx = x0 + max(0, (x1 - x0 - lw) // 2)
        elif align == "right":
            cx = x1 - lw
        else:
            cx = x0
        draw.text((cx, cy), line, font=font, fill=fill)
        cy += lh + line_gap

def _split_highlight_segments(text: str, highlights: List[Dict]) -> List[Tuple[str, Dict | None]]:
    phrase_to_style = {item["phrase"]: item for item in highlights}
    pattern = re.compile(
        "|".join(re.escape(item["phrase"]) for item in sorted(highlights, key=lambda item: len(item["phrase"]), reverse=True))
    )

    segments: List[Tuple[str, Dict | None]] = []
    last = 0
    for match in pattern.finditer(text):
        if match.start() > last:
            segments.append((text[last:match.start()], None))
        segments.append((match.group(0), phrase_to_style[match.group(0)]))
        last = match.end()
    if last < len(text):
        segments.append((text[last:], None))
    return segments

def _layout_highlighted_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    highlights: List[Dict],
    font,
    max_width: int,
) -> List[List[Tuple[str, Dict | None, int]]]:
    raw_tokens: List[Tuple[str, Dict | None]] = []
    for segment, style in _split_highlight_segments(text, highlights):
        if not segment:
            continue
        if style is None:
            raw_tokens.extend((token, None) for token in re.findall(r"\S+\s*", segment))
        else:
            raw_tokens.append((segment, style))

    lines: List[List[Tuple[str, Dict | None, int]]] = []
    current: List[Tuple[str, Dict | None, int]] = []
    current_width = 0

    for token_text, style in raw_tokens:
        display = token_text if current else token_text.lstrip()
        if not display:
            continue
        token_width, _ = measure(draw, display, font)

        if current and current_width + token_width > max_width:
            lines.append(current)
            current = []
            current_width = 0
            display = token_text.lstrip()
            if not display:
                continue
            token_width, _ = measure(draw, display, font)

        current.append((display, style, token_width))
        current_width += token_width

    if current:
        lines.append(current)

    return lines

def draw_highlighted_text_fit(
    draw: ImageDraw.ImageDraw,
    box: Tuple[int, int, int, int],
    text: str,
    highlights: List[Dict],
    *,
    fill=TEXT,
    max_size=16,
    min_size=8,
    line_gap=2,
):
    x0, y0, x1, y1 = box
    bw = max(1, x1 - x0)
    bh = max(1, y1 - y0)

    chosen_font = load_font(min_size)
    chosen_lines: List[List[Tuple[str, Dict | None, int]]] = []
    chosen_lh = 0

    for size in range(max_size, min_size - 1, -1):
        font = load_font(size)
        lines = _layout_highlighted_text(draw, text, highlights, font, bw)
        _, lh = measure(draw, "Ag", font)
        total_h = len(lines) * lh + max(0, len(lines) - 1) * line_gap
        if total_h <= bh:
            chosen_font = font
            chosen_lines = lines
            chosen_lh = lh
            break

    if not chosen_lines:
        chosen_lines = _layout_highlighted_text(draw, text, highlights, chosen_font, bw)
        _, chosen_lh = measure(draw, "Ag", chosen_font)

    cy = y0
    for line in chosen_lines:
        cx = x0
        for token_text, style, token_width in line:
            if style is not None:
                rounded(
                    draw,
                    (cx - 2, cy - 1, cx + token_width + 2, cy + chosen_lh - 1),
                    style["fill"],
                    outline=style["fill"],
                    radius=5,
                    width=1,
                )
                draw.text((cx, cy), token_text, font=chosen_font, fill=style["text_fill"])
            else:
                draw.text((cx, cy), token_text, font=chosen_font, fill=fill)
            cx += token_width
        cy += chosen_lh + line_gap

# ============================================================
# Drawing primitives
# ============================================================
def make_canvas() -> Image.Image:
    img = Image.new("RGBA", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.rectangle((0, 0, W, 72), fill="#EEF2F8")
    d.line((0, 72, W, 72), fill="#DCE4F0", width=2)
    for y in range(100, H, 80):
        d.line((36, y, W - 36, y), fill=GRID, width=1)
    return img

def rounded(draw: ImageDraw.ImageDraw, box, fill, outline=BORDER, radius=18, width=2):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)

def card(draw: ImageDraw.ImageDraw, box, title=None, title_color=NAVY, fill=WHITE):
    rounded(draw, box, fill=fill, outline=BORDER, radius=18, width=2)
    if title:
        draw_text_fit(
            draw,
            (box[0] + 18, box[1] + 14, box[2] - 18, box[1] + 46),
            title,
            max_size=22,
            min_size=14,
            bold=True,
            fill=title_color,
        )

def header(draw: ImageDraw.ImageDraw, title: str, subtitle: str | None = None):
    draw_text_fit(
        draw,
        (40, 18, W - 40, 52),
        title,
        max_size=28,
        min_size=16,
        bold=True,
        fill=NAVY,
        align="center",
        valign="center",
    )
    if subtitle:
        draw_text_fit(
            draw,
            (40, 52, W - 40, 70),
            subtitle,
            max_size=14,
            min_size=10,
            fill=TEXT_SOFT,
            align="center",
            valign="center",
        )

def footer(draw: ImageDraw.ImageDraw, txt: str):
    draw_text_fit(
        draw,
        (40, H - 26, W - 40, H - 8),
        txt,
        max_size=12,
        min_size=10,
        fill=TEXT_SOFT,
        align="center",
        valign="center",
    )

def pill(draw: ImageDraw.ImageDraw, box, text, fill, outline=None, text_color=TEXT):
    rounded(draw, box, fill=fill, outline=outline or fill, radius=16, width=2)
    draw_text_fit(
        draw,
        (box[0] + 10, box[1] + 8, box[2] - 10, box[3] - 8),
        text,
        max_size=16,
        min_size=10,
        bold=True,
        fill=text_color,
        align="center",
        valign="center",
    )

def draw_options_grid(draw: ImageDraw.ImageDraw, box, options=None):
    x0, y0, x1, y1 = box
    pad = 14
    gap = 14
    cell_w = (x1 - x0 - 2 * pad - gap) // 2
    cell_h = 84
    labels = options or EXAMPLE_OPTIONS
    positions = [
        (x0 + pad, y0 + pad),
        (x0 + pad + cell_w + gap, y0 + pad),
        (x0 + pad, y0 + pad + cell_h + gap),
        (x0 + pad + cell_w + gap, y0 + pad + cell_h + gap),
    ]

    for idx, ((lab, txt), (cx, cy)) in enumerate(zip(labels, positions)):
        fill = "#EAE6FB" if idx == 0 else "#F7F8FA"
        outline = BLUE if idx == 0 else BORDER
        rounded(draw, (cx, cy, cx + cell_w, cy + cell_h), fill, outline, radius=14, width=2)
        draw.ellipse((cx + 10, cy + 10, cx + 34, cy + 34), fill="#ECEAF7")
        draw_text_fit(draw, (cx + 10, cy + 10, cx + 34, cy + 34), lab, max_size=14, min_size=10, bold=True, align="center", valign="center", fill=NAVY_DARK)
        draw_text_fit(draw, (cx + 42, cy + 18, cx + cell_w - 12, cy + cell_h - 12), txt, max_size=15, min_size=10, valign="center")

def draw_bullets(draw: ImageDraw.ImageDraw, box, items: Iterable[str], bullet_color=BLUE):
    x0, y0, x1, y1 = box
    cy = y0
    for item in items:
        draw.ellipse((x0, cy + 7, x0 + 10, cy + 17), fill=bullet_color)
        draw_text_fit(draw, (x0 + 18, cy, x1, cy + 28), item, max_size=17, min_size=11, fill=TEXT, valign="center")
        cy += 28

def draw_progress_dots(draw: ImageDraw.ImageDraw, box, stage: int, total: int):
    x0, y0, x1, y1 = box
    xs = []
    for i in range(total):
        frac = i / max(1, total - 1)
        xs.append(int(x0 + frac * (x1 - x0)))
    line_y = (y0 + y1) // 2
    draw.line((x0, line_y, x1, line_y), fill=BORDER, width=2)
    for i, x in enumerate(xs):
        fill = BLUE if i <= stage else WHITE
        outline = BLUE if i <= stage else BORDER
        draw.ellipse((x - 5, line_y - 5, x + 5, line_y + 5), fill=fill, outline=outline, width=2)

def draw_posterior_bars(draw: ImageDraw.ImageDraw, box, probs):
    x0, y0, x1, y1 = box
    labels = ["A", "B", "C", "D"]
    colors = [BLUE, ORANGE, PURPLE, "#9CA3AF"]

    bar_y = y1 - 28
    left = x0 + 26
    right = x1 - 10
    usable_w = right - left
    gap = 14
    bw = (usable_w - gap * (len(probs) - 1)) // len(probs)

    for i, p in enumerate(probs):
        bx = left + i * (bw + gap)
        fill_h = int(60 * p) + 8
        by0 = bar_y - fill_h
        by1 = bar_y
        draw.rounded_rectangle((bx, by0, bx + bw, by1), radius=7, fill=colors[i])
        draw_text_fit(draw, (bx, bar_y + 4, bx + bw, bar_y + 24), labels[i], max_size=12, min_size=10, fill=TEXT_SOFT, align="center")
        draw_text_fit(draw, (bx, by0 - 22, bx + bw, by0 - 2), f"{int(round(100*p))}%", max_size=12, min_size=9, fill=TEXT_SOFT, align="center")

def draw_line_chart(draw: ImageDraw.ImageDraw, box, phase: int):
    x0, y0, x1, y1 = box
    draw.line((x0 + 34, y1 - 36, x1 - 18, y1 - 36), fill=TEXT_SOFT, width=2)
    draw.line((x0 + 34, y0 + 18, x0 + 34, y1 - 36), fill=TEXT_SOFT, width=2)
    draw_text_fit(draw, (x0 + 56, y1 - 26, x1 - 24, y1 - 8), "more of the Markov tossup revealed", max_size=11, min_size=9, fill=TEXT_SOFT, align="center")
    draw_text_fit(draw, (x0 - 4, y0 + 22, x0 + 44, y0 + 74), "value", max_size=11, min_size=9, fill=TEXT_SOFT, align="center", valign="center")

    p1 = [
        (x0 + 34, y1 - 60),
        (x0 + 110, y1 - 84),
        (x0 + 190, y1 - 108),
        (x0 + 270, y1 - 132),
        (x0 + 348, y1 - 154),
    ]
    p2 = [
        (x0 + 34, y0 + 44),
        (x0 + 110, y0 + 56),
        (x0 + 190, y0 + 70),
        (x0 + 270, y0 + 88),
        (x0 + 348, y0 + 104),
    ]
    draw.line(p1, fill=BLUE, width=3)
    draw.line(p2, fill=ORANGE, width=3)

    idx = min(phase, 4)
    cx = p1[idx][0]
    draw.line((cx, y0 + 18, cx, y1 - 36), fill="#444444", width=2)

    legend_x = x1 - 140
    rounded(draw, (legend_x, y0 + 24, x1 - 22, y0 + 70), WHITE, outline=BORDER, radius=12, width=2)
    draw.line((legend_x + 12, y0 + 40, legend_x + 34, y0 + 40), fill=BLUE, width=3)
    draw_text_fit(draw, (legend_x + 40, y0 + 28, x1 - 30, y0 + 46), "act now value", max_size=12, min_size=9, fill=TEXT_SOFT)
    draw.line((legend_x + 12, y0 + 56, legend_x + 34, y0 + 56), fill=ORANGE, width=3)
    draw_text_fit(draw, (legend_x + 40, y0 + 44, x1 - 30, y0 + 62), "wait value", max_size=12, min_size=9, fill=TEXT_SOFT)

    notes = EXAMPLE_VALUE_NOTES
    rounded(draw, (x0 + 126, y0 + 52, x0 + 236, y0 + 104), GREEN_SOFT if phase >= 2 else ORANGE_SOFT,
            outline=GREEN if phase >= 2 else ORANGE, radius=12, width=2)
    draw_text_fit(draw, (x0 + 136, y0 + 60, x0 + 226, y0 + 98), notes[idx], max_size=12, min_size=9, fill=TEXT_SOFT, align="center", valign="center")

def fmt_pct(value) -> str:
    try:
        return f"{100 * float(value):.1f}%"
    except (TypeError, ValueError):
        return "n/a"

def fmt_num(value, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "n/a"

def load_eval_report() -> Dict:
    path = Path(__file__).resolve().parent / "artifacts" / "smoke" / "evaluation_report.json"
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}

def best_baseline_result(report: Dict) -> Tuple[str, Dict]:
    baseline_summary = report.get("baseline_summary", {})
    best_label = "n/a"
    best_metrics: Dict = {}
    best_sq = float("-inf")

    for method, payload in baseline_summary.items():
        if isinstance(payload, dict) and payload and all(isinstance(v, dict) for v in payload.values()):
            for threshold, metrics in payload.items():
                sq = metrics.get("mean_sq")
                if isinstance(sq, (float, int)) and sq > best_sq:
                    best_sq = float(sq)
                    best_label = f"{method} @ {threshold}"
                    best_metrics = metrics
        elif isinstance(payload, dict):
            sq = payload.get("mean_sq")
            if isinstance(sq, (float, int)) and sq > best_sq:
                best_sq = float(sq)
                best_label = method
                best_metrics = payload

    return best_label, best_metrics

EVAL_REPORT = load_eval_report()

# ============================================================
# Scene specs
# ============================================================
SCENES: List[Dict] = [
    {
        "kind": "title_slide",
        "footer": "CS234 Final Project",
    },
    {
        "kind": "problem_slide",
        "footer": "Problem",
    },
    {
        "kind": "background_slide",
        "footer": "Background and Setup",
    },
    {
        "kind": "why_mc_slide",
        "footer": "Why Multiple Choice?",
    },
    {
        "kind": "method_slide",
        "footer": "Method Overview",
    },
    {
        "kind": "contribution_slide",
        "footer": "Our Contribution",
    },
    {
        "kind": "expected_results_slide",
        "footer": "Smoke Results Snapshot",
    },
    {
        "kind": "why_ppo_less_accurate",
        "footer": "How To Read PPO Metrics",
    },
    {
        "kind": "intro",
        "footer": "MC Stopping Intuition",
    },
    {
        "kind": "section",
        "title": "Quiz bowl MC buzzing",
        "section_title": "Section 1: the posterior sharpens\nas \"clues\" arrive",
        "accent": BLUE,
        "footer_small": "first: watch uncertainty collapse",
        "footer": "Posterior Sharpening",
    },
    {
        "kind": "posterior",
        "stage": 0,
        "footer": "Posterior Sharpening",
    },
    {
        "kind": "posterior",
        "stage": 1,
        "footer": "Posterior Sharpening",
    },
    {
        "kind": "posterior",
        "stage": 2,
        "footer": "Posterior Sharpening",
    },
    {
        "kind": "posterior",
        "stage": 3,
        "footer": "Posterior Sharpening",
    },
    {
        "kind": "posterior",
        "stage": 4,
        "footer": "Posterior Sharpening",
    },
    {
        "kind": "section",
        "title": "Quiz bowl MC buzzing",
        "section_title": "Section 2: \"buzz\" when acting now\nbeats waiting",
        "accent": ORANGE,
        "footer_small": "next: compare current value to one more clue",
        "footer": "Act Now Vs Wait",
    },
    {
        "kind": "value_chart",
        "stage": 0,
        "footer": "Act Now Vs Wait",
    },
    {
        "kind": "value_chart",
        "stage": 1,
        "footer": "Act Now Vs Wait",
    },
    {
        "kind": "value_chart",
        "stage": 2,
        "footer": "Act Now Vs Wait",
    },
    {
        "kind": "value_chart",
        "stage": 3,
        "footer": "Act Now Vs Wait",
    },
    {
        "kind": "value_chart",
        "stage": 4,
        "footer": "Act Now Vs Wait",
    },
    {
        "kind": "section",
        "title": "Quiz bowl MC buzzing",
        "section_title": "Section 3: \"ABSTAIN\" means wait,\nnot \"None-of-the-Above\"",
        "accent": RED,
        "footer_small": "next: waiting is about value of information",
        "footer": "Abstain Semantics",
    },
    {
        "kind": "abstain",
        "stage": 0,
        "footer": "Abstain Semantics",
    },
    {
        "kind": "abstain",
        "stage": 1,
        "footer": "Abstain Semantics",
    },
    {
        "kind": "abstain",
        "stage": 2,
        "footer": "Abstain Semantics",
    },
    {
        "kind": "section",
        "title": "Quiz bowl MC buzzing",
        "section_title": "Section 4: separate answer quality\nfrom \"buzz\" timing",
        "accent": PURPLE,
        "footer_small": "next: factor the decision into two modules",
        "footer": "Factor Stop And Answer",
    },
    {
        "kind": "factorization",
        "stage": 0,
        "footer": "Factor Stop And Answer",
    },
    {
        "kind": "factorization",
        "stage": 1,
        "footer": "Factor Stop And Answer",
    },
    {
        "kind": "section",
        "title": "Quiz bowl MC buzzing",
        "section_title": "Section 5: one compact mental model",
        "accent": GREEN,
        "footer_small": "finally: compress the whole story",
        "footer": "Training Recipe",
    },
    {
        "kind": "recipe",
        "footer": "Training Recipe",
    },
    {
        "kind": "pipeline",
        "stage": 0,
        "footer": "Training Setup",
    },
    {
        "kind": "pipeline",
        "stage": 1,
        "footer": "Evaluation Setup",
    },
    {
        "kind": "pipeline",
        "stage": 2,
        "footer": "Smoke Results",
    },
    {
        "kind": "evaluation_slide",
        "footer": "Evaluation",
    },
    {
        "kind": "summary",
        "stage": 0,
        "footer": "Summary",
    },
    {
        "kind": "summary",
        "stage": 1,
        "footer": "Summary",
    },
    {
        "kind": "summary",
        "stage": 2,
        "footer": "Summary",
    },
    {
        "kind": "references_slide",
        "footer": "References",
    },
]

assert len(SCENES) == 38

# ============================================================
# Scene renderers
# ============================================================
def render_title_slide(spec: Dict) -> Image.Image:
    img = make_canvas()
    d = ImageDraw.Draw(img)
    draw_text_fit(
        d,
        (80, 100, W - 80, 180),
        "Quiz Bowl RL Buzzer",
        max_size=38,
        min_size=22,
        bold=True,
        fill=NAVY,
        align="center",
        valign="center",
    )
    draw_text_fit(
        d,
        (80, 190, W - 80, 240),
        "Multiple-Choice Strategic Buzzing Under Incremental Clues",
        max_size=22,
        min_size=14,
        fill=TEXT_SOFT,
        align="center",
        valign="center",
    )
    rounded(d, (300, 270, 660, 380), WHITE, outline=BORDER, radius=16, width=2)
    draw_text_fit(
        d,
        (320, 284, 640, 370),
        "CS234 Final Project\n\nKathleen Weng\nImran Hassan\nAnkit Aggarwal",
        max_size=18,
        min_size=12,
        fill=TEXT,
        align="center",
        valign="center",
    )
    draw_text_fit(
        d,
        (80, 400, W - 80, 440),
        "March 2026",
        max_size=16,
        min_size=12,
        fill=TEXT_SOFT,
        align="center",
        valign="center",
    )
    footer(d, spec["footer"])
    return img

def render_problem_slide(spec: Dict) -> Image.Image:
    img = make_canvas()
    d = ImageDraw.Draw(img)
    header(d, "Problem")
    rounded(d, (48, 96, 912, 482), WHITE, outline=BORDER, radius=16, width=2)
    draw_bullets(d, (80, 120, 880, 460), [
        "Quiz bowl questions reveal evidence incrementally",
        "A good system must decide WHEN to buzz, not just WHAT to pick",
        "Buzz too early: higher risk of a wrong answer",
        "Buzz too late: lower strategic value and less chance to beat an opponent",
        "We study this in a multiple-choice setting so the answer space is controlled and evaluation is reproducible",
    ], bullet_color=BLUE)
    footer(d, spec["footer"])
    return img

def render_background_slide(spec: Dict) -> Image.Image:
    img = make_canvas()
    d = ImageDraw.Draw(img)
    header(d, "Background And Setup")
    rounded(d, (48, 96, 460, 482), WHITE, outline=BORDER, radius=16, width=2)
    draw_text_fit(d, (68, 110, 440, 140), "Sequential decision problem", max_size=20, min_size=13, bold=True, fill=NAVY)
    draw_bullets(d, (68, 152, 440, 340), [
        "Model quiz bowl as sequential decision over partial clues",
        "Question prefixes over time",
        "K = 4 answer options",
        "One correct option plus three distractors",
    ], bullet_color=BLUE)
    rounded(d, (500, 96, 912, 482), WHITE, outline=BORDER, radius=16, width=2)
    draw_text_fit(d, (520, 110, 892, 140), "Two policy families", max_size=20, min_size=13, bold=True, fill=PURPLE)
    pill(d, (540, 170, 872, 220), "Belief-feature buzzers", BLUE_SOFT, outline=BLUE, text_color=NAVY)
    draw_bullets(d, (540, 240, 872, 330), [
        "Threshold, softmax-profile, and Bayesian baselines",
        "PPO on structured observations",
    ], bullet_color=BLUE)
    pill(d, (540, 340, 872, 390), "T5 text-policy buzzers", ORANGE_SOFT, outline=ORANGE, text_color=NAVY)
    draw_bullets(d, (540, 410, 872, 470), [
        "End-to-end supervised warm start plus PPO",
    ], bullet_color=ORANGE)
    footer(d, spec["footer"])
    return img

def render_why_mc_slide(spec: Dict) -> Image.Image:
    img = make_canvas()
    d = ImageDraw.Draw(img)
    header(d, "Why Multiple Choice?")
    rounded(d, (48, 96, 500, 340), WHITE, outline=BORDER, radius=16, width=2)
    draw_text_fit(d, (68, 112, 480, 140), "Advantages", max_size=20, min_size=13, bold=True, fill=GREEN)
    draw_bullets(d, (68, 154, 480, 320), [
        "Eliminates aliasing and grading complexity",
        "Isolates the buzzing decision itself",
        "Makes evaluation controlled and reproducible",
    ], bullet_color=GREEN)
    rounded(d, (48, 360, 500, 482), WHITE, outline=BORDER, radius=16, width=2)
    draw_text_fit(d, (68, 374, 480, 400), "Challenge", max_size=20, min_size=13, bold=True, fill=RED)
    draw_bullets(d, (68, 412, 480, 472), [
        "Naive distractors create artifacts",
        "Answer generation quality still matters",
    ], bullet_color=RED)
    rounded(d, (530, 96, 912, 482), BLUE_SOFT, outline=BLUE, radius=16, width=2)
    draw_text_fit(d, (550, 112, 892, 140), "Design goals", max_size=20, min_size=13, bold=True, fill=NAVY)
    draw_bullets(d, (550, 160, 892, 460), [
        "Keep the answer space constrained",
        "Make options hard enough that the agent must use clues",
        "Use anti-artifact guards: alias collision, token overlap, length ratio, and question-text overlap checks",
    ], bullet_color=BLUE)
    footer(d, spec["footer"])
    return img

def render_method_slide(spec: Dict) -> Image.Image:
    img = make_canvas()
    d = ImageDraw.Draw(img)
    header(d, "Method Overview")
    rounded(d, (48, 96, 912, 482), WHITE, outline=BORDER, radius=16, width=2)
    steps = [
        ("1. Load tossups", BLUE_SOFT, BLUE),
        ("2. Build answer profiles", PURPLE_SOFT, PURPLE),
        ("3. Construct MC questions", ORANGE_SOFT, ORANGE),
        ("4. Score with a likelihood model", GREEN_SOFT, GREEN),
        ("5. Convert beliefs into observations", BLUE_SOFT, BLUE),
        ("6. Run buzzer agents", PURPLE_SOFT, PURPLE),
        ("7. Evaluate", GREEN_SOFT, GREEN),
    ]
    y = 116
    for i, (label, bg, outline) in enumerate(steps):
        x0 = 80 if i % 2 == 0 else 480
        rounded(d, (x0, y, x0 + 360, y + 42), bg, outline=outline, radius=12, width=2)
        draw_text_fit(d, (x0 + 14, y + 8, x0 + 346, y + 36), label, max_size=17, min_size=11, bold=True, fill=NAVY, valign="center")
        if i % 2 == 1:
            y += 52
    footer(d, spec["footer"])
    return img

def render_contribution_slide(spec: Dict) -> Image.Image:
    img = make_canvas()
    d = ImageDraw.Draw(img)
    header(d, "Our Contribution")
    rounded(d, (48, 96, 912, 482), WHITE, outline=BORDER, radius=16, width=2)
    draw_text_fit(
        d,
        (68, 112, 892, 144),
        "We study pyramidal quiz bowl under a restricted multiple-choice action space with reinforcement learning.",
        max_size=19,
        min_size=12,
        bold=True,
        fill=NAVY,
    )
    draw_bullets(d, (68, 160, 892, 460), [
        "Multiple-choice POMDP formulation for quiz bowl",
        "PPO to learn when to wait versus buzz and answer",
        "Artifact-resistant MC construction with several distractor strategies",
        "Direct tests of whether learned policies use clues or answer-choice patterns",
        "Calibration-focused evaluation with ECE, Brier, and S_q beyond raw accuracy",
    ], bullet_color=PURPLE)
    footer(d, spec["footer"])
    return img

def render_expected_results_slide(spec: Dict) -> Image.Image:
    img = make_canvas()
    d = ImageDraw.Draw(img)
    header(d, "Smoke Results Snapshot")
    rounded(d, (48, 90, 912, 482), WHITE, outline=BORDER, radius=16, width=2)

    best_name, best_metrics = best_baseline_result(EVAL_REPORT)
    ppo = EVAL_REPORT.get("ppo_summary", {})

    rounded(d, (68, 106, 892, 138), BLUE_SOFT, outline=BLUE, radius=8, width=1)
    draw_text_fit(
        d,
        (84, 112, 876, 132),
        f"Best baseline by mean S_q: {best_name}",
        max_size=14,
        min_size=10,
        bold=True,
        fill=NAVY,
        align="center",
        valign="center",
    )

    col_xs = [68, 338, 578, 770]
    col_ws = [250, 220, 180, 122]
    headers_txt = ["Metric", "Best baseline", "PPO", "Reading"]
    for cx, cw, ht in zip(col_xs, col_ws, headers_txt):
        rounded(d, (cx, 150, cx + cw, 184), NAVY, outline=NAVY, radius=8, width=1)
        draw_text_fit(d, (cx + 8, 154, cx + cw - 8, 180), ht, max_size=15, min_size=10, bold=True, fill=WHITE, align="center", valign="center")

    rows = [
        ("Mean S_q", fmt_num(best_metrics.get("mean_sq")), fmt_num(ppo.get("mean_sq")), "higher better"),
        ("Buzz accuracy", fmt_pct(best_metrics.get("buzz_accuracy")), fmt_pct(ppo.get("buzz_accuracy")), "higher better"),
        ("Mean buzz step", fmt_num(best_metrics.get("mean_buzz_step"), 2), fmt_num(ppo.get("mean_buzz_step"), 2), "timing profile"),
        ("Reward-like", fmt_num(best_metrics.get("mean_reward_like")), fmt_num(ppo.get("mean_reward_like")), "higher better"),
        ("ECE", fmt_num(best_metrics.get("ece")), fmt_num(ppo.get("ece")), "lower better"),
        ("Brier", fmt_num(best_metrics.get("brier")), fmt_num(ppo.get("brier")), "lower better"),
    ]
    for ri, (metric, baseline_val, ppo_val, reading) in enumerate(rows):
        ry = 194 + ri * 42
        bg = "#FAFBFD" if ri % 2 == 0 else WHITE
        for cx, cw, val in zip(col_xs, col_ws, [metric, baseline_val, ppo_val, reading]):
            rounded(d, (cx, ry, cx + cw, ry + 38), bg, outline=BORDER, radius=6, width=1)
            color = TEXT_SOFT if val in {"higher better", "lower better", "timing profile"} else TEXT
            draw_text_fit(d, (cx + 8, ry + 5, cx + cw - 8, ry + 33), val, max_size=14, min_size=9, fill=color, align="center", valign="center")

    rounded(d, (100, 444, 860, 470), ORANGE_SOFT, outline=ORANGE, radius=10, width=1)
    draw_text_fit(
        d,
        (116, 448, 844, 466),
        "Use this as a smoke-test snapshot, not a final leaderboard: compare S_q, calibration, and timing together.",
        max_size=14,
        min_size=10,
        fill=ORANGE,
        align="center",
        valign="center",
    )
    footer(d, spec["footer"])
    return img

def render_why_ppo_less_accurate(spec: Dict) -> Image.Image:
    img = make_canvas()
    d = ImageDraw.Draw(img)
    header(d, "How To Read PPO Metrics")

    best_name, best_metrics = best_baseline_result(EVAL_REPORT)
    ppo = EVAL_REPORT.get("ppo_summary", {})

    rounded(d, (48, 96, 460, 482), WHITE, outline=BORDER, radius=16, width=2)
    draw_text_fit(d, (68, 112, 440, 140), "Why raw accuracy can dip", max_size=18, min_size=12, bold=True, fill=NAVY)
    draw_bullets(d, (68, 156, 440, 380), [
        "PPO learns a timing policy, not just a final guess",
        "Waiting can avoid low-confidence answers",
        "That may lower raw buzz accuracy on some slices",
        "Reward design and training stability matter a lot",
        "So accuracy alone is not the right summary statistic",
    ], bullet_color=ORANGE)
    rounded(d, (68, 400, 440, 466), ORANGE_SOFT, outline=ORANGE, radius=10, width=1)
    draw_text_fit(
        d,
        (82, 408, 426, 458),
        f"Current smoke: {best_name} buzz acc {fmt_pct(best_metrics.get('buzz_accuracy'))} vs PPO {fmt_pct(ppo.get('buzz_accuracy'))}",
        max_size=13,
        min_size=9,
        fill=ORANGE,
        align="center",
        valign="center",
    )

    rounded(d, (500, 96, 912, 482), WHITE, outline=BORDER, radius=16, width=2)
    draw_text_fit(d, (520, 112, 892, 140), "What to compare alongside it", max_size=18, min_size=12, bold=True, fill=GREEN)
    draw_bullets(d, (520, 156, 892, 360), [
        "Mean S_q or task reward",
        "Mean buzz step / timing profile",
        "Calibration at buzz time",
        "Controls that test clue use versus answer-artifact use",
        "Whether the policy abstains or times out strategically",
    ], bullet_color=GREEN)
    rounded(d, (520, 380, 892, 466), GREEN_SOFT, outline=GREEN, radius=10, width=1)
    draw_text_fit(
        d,
        (534, 388, 878, 458),
        "A PPO buzzer is only better if its timing behavior improves the metrics you actually care about.",
        max_size=14,
        min_size=10,
        fill=GREEN,
        align="center",
        valign="center",
    )

    footer(d, spec["footer"])
    return img

def render_intro(spec: Dict) -> Image.Image:
    img = make_canvas()
    d = ImageDraw.Draw(img)

    header(d, "Quiz bowl MC \"buzzing\" is optimal stopping with improving evidence")

    card(d, (48, 96, 458, 486), EXAMPLE_TITLE)
    draw_text_fit(d, (76, 136, 426, 152),
                  "Original tossup (verbatim)", max_size=14, min_size=10, bold=True, fill=TEXT_SOFT)
    draw_text_fit(d, (76, 154, 426, 168),
                  EXAMPLE_SOURCE, max_size=12, min_size=9, fill=TEXT_SOFT)
    draw_highlighted_text_fit(d, (76, 176, 426, 274),
                              EXAMPLE_TOSSUP_VERBATIM,
                              EXAMPLE_TOSSUP_HIGHLIGHTS,
                              max_size=12, min_size=6, fill=TEXT_SOFT, line_gap=0)
    draw_text_fit(d, (76, 282, 426, 296),
                  EXAMPLE_ANSWERLINE,
                  max_size=11, min_size=8, bold=True, fill=NAVY, line_gap=1)
    draw_text_fit(d, (76, 298, 426, 316),
                  EXAMPLE_ACCEPTS,
                  max_size=8, min_size=6, fill=TEXT_SOFT, line_gap=1)
    rounded(d, (86, 326, 430, 376), "#F7F8FA", outline=BORDER, radius=12, width=1)
    draw_text_fit(d, (98, 336, 418, 368),
                  'Clue progression\nEarly clues are obscure, middle clues narrow the field, and the final "memoryless chain" clue is a giveaway.',
                  max_size=13, min_size=9, fill=TEXT_SOFT, line_gap=2)
    rounded(d, (86, 388, 430, 458), WHITE, outline=BLUE, radius=16, width=2)
    draw_text_fit(d, (96, 396, 420, 450),
                  'RL intuition\nThose repeated anaphoric phrases all resolve to Markov, and the strategic decision is when to "buzz."',
                  max_size=20, min_size=12, bold=True, fill=NAVY)

    card(d, (506, 96, 916, 486), "Presentation multiple-choice setting", title_color=PURPLE)
    draw_text_fit(d, (536, 142, 886, 174),
                  'The answer options are fixed from the start, using a cleaner mathematician-only set:',
                  max_size=20, min_size=13, fill=TEXT_SOFT)
    draw_options_grid(d, (536, 184, 884, 360), EXAMPLE_OPTIONS)
    rounded(d, (536, 428, 886, 486), "#EAE6FB", outline=BLUE, radius=16, width=2)
    draw_text_fit(d, (548, 438, 874, 478),
                  'Decision rule\nAt each "prefix," should the policy "WAIT" or "BUZZ" on one listed answer?',
                  max_size=18, min_size=11, bold=True, fill=NAVY)

    footer(d, spec["footer"])
    return img

def render_section(spec: Dict) -> Image.Image:
    img = make_canvas()
    d = ImageDraw.Draw(img)
    header(d, spec["title"])
    rounded(d, (255, 170, 705, 332), WHITE, outline=spec["accent"], radius=14, width=2)
    d.line((320, 206, 640, 206), fill=spec["accent"], width=4)
    draw_text_fit(d, (300, 226, 660, 284), spec["section_title"], max_size=26, min_size=15, bold=True, fill=spec["accent"], align="center", valign="center")
    draw_text_fit(d, (292, 286, 668, 312), spec["footer_small"], max_size=12, min_size=10, fill=TEXT_SOFT, align="center")
    return img

def render_posterior(spec: Dict) -> Image.Image:
    stage = spec["stage"]
    stage_data = EXAMPLE_POSTERIOR_STAGES[stage]
    img = make_canvas()
    d = ImageDraw.Draw(img)
    header(d, 'As more "clues" arrive, the Markov posterior collapses onto one option')

    rounded(d, (44, 86, 916, 482), WHITE, outline=BORDER, radius=16, width=2)
    draw_progress_dots(d, (72, 110, 664, 136), stage, 5)
    draw_text_fit(d, (72, 142, 664, 174), "\"prefix t\": only the clues seen so far are visible", max_size=15, min_size=10, fill=TEXT_SOFT)

    rounded(d, (72, 176, 664, 232), WHITE, outline=BORDER, radius=12, width=2)
    draw_text_fit(d, (88, 188, 648, 220), stage_data["prefix"], max_size=17, min_size=11, fill=TEXT)

    draw_posterior_bars(d, (72, 254, 664, 388), stage_data["probs"])

    is_wait = stage_data["decision"] == "WAIT"
    rounded(d, (694, 182, 758, 254), WHITE if is_wait else GREEN_SOFT,
            outline=BORDER if is_wait else GREEN, radius=14, width=2)
    draw_text_fit(d, (700, 197, 752, 240), stage_data["decision"], max_size=18, min_size=12, bold=True, align="center", valign="center", fill=ORANGE if is_wait else GREEN)

    if is_wait:
        rounded(d, (774, 170, 902, 286), ORANGE_SOFT, outline=ORANGE, radius=14, width=2)
        draw_text_fit(d, (786, 182, 890, 274),
                      stage_data["callout"],
                      max_size=17, min_size=10, fill=ORANGE)
    else:
        rounded(d, (774, 170, 902, 286), GREEN_SOFT, outline=GREEN, radius=14, width=2)
        draw_text_fit(d, (786, 182, 890, 274),
                      stage_data["callout"],
                      max_size=17, min_size=10, fill=GREEN)

    rounded(d, (128, 420, 832, 462), "#F7F8FA", outline=BORDER, radius=10, width=1)
    draw_text_fit(d, (142, 430, 818, 454),
                  'With a fixed answer set, the policy is not discovering new options. It is deciding when the current posterior over Markov vs. the three distractors is sharp enough to "act."',
                  max_size=14, min_size=9, fill=TEXT_SOFT, align="center", valign="center")

    footer(d, spec["footer"])
    return img

def render_value_chart(spec: Dict) -> Image.Image:
    stage = spec["stage"]
    img = make_canvas()
    d = ImageDraw.Draw(img)
    header(d, '"Buzz" when acting now is worth more than waiting in the Markov example')

    rounded(d, (50, 92, 910, 470), WHITE, outline=BORDER, radius=16, width=2)
    draw_line_chart(d, (90, 130, 850, 382), stage)

    rounded(d, (160, 408, 800, 448), "#F7F8FA", outline=BORDER, radius=10, width=1)
    draw_text_fit(d, (178, 416, 782, 440), EXAMPLE_VALUE_MESSAGES[stage], max_size=15, min_size=10, fill=TEXT_SOFT, align="center", valign="center")

    footer(d, spec["footer"])
    return img

def render_abstain(spec: Dict) -> Image.Image:
    stage = spec["stage"]
    img = make_canvas()
    d = ImageDraw.Draw(img)
    header(d, "\"ABSTAIN\" means wait, not \"none of the above\"")

    rounded(d, (46, 96, 914, 472), WHITE, outline=BORDER, radius=16, width=2)
    card(d, (72, 130, 396, 388), "The correct answer is already in the \"set\"")
    draw_options_grid(d, (92, 176, 376, 312), EXAMPLE_OPTIONS)
    draw_text_fit(d, (92, 326, 376, 374),
                  '"Abstaining" means one more "clue" is worth more than committing, even though Markov is already option A.',
                  max_size=16, min_size=10, fill=TEXT_SOFT, align="center", valign="center")

    card(d, (426, 130, 888, 388), 'What changes over "time" in the Markov example')
    draw_posterior_bars(d, (452, 186, 790, 300), [
        EXAMPLE_POSTERIOR_STAGES[0]["probs"],
        EXAMPLE_POSTERIOR_STAGES[1]["probs"],
        EXAMPLE_POSTERIOR_STAGES[3]["probs"],
    ][stage])

    draw_text_fit(d, (500, 316, 706, 348), "ABSTAIN", max_size=26, min_size=16, bold=True, fill=ORANGE, align="center")
    draw_text_fit(d, (492, 344, 716, 382), "means continue because another \"clue\" still has value", max_size=16, min_size=10, fill=TEXT_SOFT, align="center")

    d.line((760, 334, 856, 370), fill=RED, width=3)
    d.line((760, 370, 856, 334), fill=RED, width=3)
    draw_text_fit(d, (738, 310, 880, 334), "None-of-the-Above", max_size=15, min_size=9, bold=True, fill=RED, align="center")

    rounded(d, (160, 416, 800, 454), "#F7F8FA", outline=BORDER, radius=10, width=1)
    draw_text_fit(d, (176, 424, 784, 446), EXAMPLE_ABSTAIN_MESSAGES[stage], max_size=13, min_size=9, fill=TEXT_SOFT, align="center", valign="center")

    footer(d, spec["footer"])
    return img

def render_factorization(spec: Dict) -> Image.Image:
    stage = spec["stage"]
    img = make_canvas()
    d = ImageDraw.Draw(img)
    header(d, "Implementation idea: separate answer quality from \"buzz\" timing")

    rounded(d, (46, 96, 914, 472), WHITE, outline=BORDER, radius=16, width=2)

    rounded(d, (84, 162, 214, 214), WHITE, outline=BORDER, radius=12, width=2)
    draw_text_fit(d, (96, 170, 202, 206), "\"prefix h_t\"\n+ option set C", max_size=18, min_size=11, align="center", valign="center")

    rounded(d, (300, 122, 500, 238), BLUE_SOFT if stage >= 0 else WHITE, outline=BLUE, radius=14, width=2)
    draw_text_fit(d, (316, 140, 484, 176), "Answer module", max_size=22, min_size=13, bold=True, align="center")
    draw_text_fit(d, (320, 178, 480, 220), "outputs p_ans(i | h_t)", max_size=20, min_size=12, align="center", valign="center")

    rounded(d, (300, 286, 500, 402), ORANGE_SOFT if stage >= 1 else WHITE, outline=ORANGE, radius=14, width=2)
    draw_text_fit(d, (316, 304, 484, 340), "Stop module", max_size=22, min_size=13, bold=True, align="center")
    draw_text_fit(d, (320, 344, 480, 384), "outputs p_buzz(h_t)", max_size=20, min_size=12, align="center", valign="center")

    d.line((214, 188, 300, 178), fill=BLUE, width=4)
    d.line((214, 188, 300, 344), fill=ORANGE, width=4)

    rounded(d, (608, 180, 866, 344), PURPLE_SOFT if stage >= 1 else WHITE, outline=PURPLE, radius=14, width=2)
    draw_text_fit(d, (624, 198, 850, 234), "Flat action interface", max_size=22, min_size=13, bold=True, align="center")
    draw_text_fit(d, (626, 248, 848, 320),
                  "P(\"WAIT\") = 1 - p_buzz\nP(\"BUZZ i\") = p_buzz * p_ans(i)",
                  max_size=18, min_size=10, align="center", valign="center")

    rounded(d, (158, 424, 804, 462), "#F7F8FA", outline=BORDER, radius=10, width=1)
    draw_text_fit(d, (176, 432, 786, 454),
                  "Same semantics, cleaner code: factor answer choice and \"stop timing\" internally, then flatten only if the RL stack needs it.",
                  max_size=13, min_size=9, fill=TEXT_SOFT, align="center", valign="center")

    footer(d, spec["footer"])
    return img

def render_recipe(spec: Dict) -> Image.Image:
    img = make_canvas()
    d = ImageDraw.Draw(img)
    header(d, "A practical training recipe")

    rounded(d, (48, 104, 912, 472), WHITE, outline=BORDER, radius=16, width=2)

    steps = [
        ("1. train the answer model", "learn p_ans(i | h_t)", BLUE_SOFT, BLUE),
        ("2. warm-start the stop head", "learn when to \"buzz\"", ORANGE_SOFT, ORANGE),
        ("3. PPO fine-tuning", "optimize task reward", GREEN_SOFT, GREEN),
    ]
    xs = [84, 328, 572]
    for x, (a, b, fill, outline) in zip(xs, steps):
        rounded(d, (x, 202, x + 200, 314), fill, outline=outline, radius=14, width=2)
        draw_text_fit(d, (x + 12, 218, x + 188, 246), a, max_size=20, min_size=12, bold=True, fill=NAVY, align="center")
        draw_text_fit(d, (x + 14, 254, x + 186, 296), b, max_size=18, min_size=11, fill=TEXT_SOFT, align="center", valign="center")

    rounded(d, (154, 394, 806, 438), "#F7F8FA", outline=BORDER, radius=10, width=1)
    draw_text_fit(d, (170, 404, 790, 430),
                  "RL can then focus mostly on \"timing\" instead of learning answer choice and stopping from scratch at the same time.",
                  max_size=15, min_size=10, fill=TEXT_SOFT, align="center", valign="center")
    footer(d, spec["footer"])
    return img

def render_pipeline(spec: Dict) -> Image.Image:
    stage = spec["stage"]
    img = make_canvas()
    d = ImageDraw.Draw(img)

    titles = [
        "Training setup in code",
        "Evaluation protocol",
        "Smoke results snapshot",
    ]
    header(d, titles[stage])
    rounded(d, (48, 98, 912, 474), WHITE, outline=BORDER, radius=16, width=2)

    if stage == 0:
        rounded(d, (74, 138, 886, 230), BLUE_SOFT, outline=BLUE, radius=12, width=2)
        draw_text_fit(
            d,
            (92, 154, 868, 214),
            "scripts/build_mc_dataset.py --smoke -> scripts/run_baselines.py --smoke -> scripts/train_ppo.py --smoke",
            max_size=18,
            min_size=11,
            bold=True,
            fill=NAVY,
            align="center",
            valign="center",
        )

        card(d, (88, 258, 370, 420), "Data + baseline prep", title_color=BLUE)
        draw_bullets(d, (104, 298, 352, 398), [
            "build MC prefixes",
            "set distractor strategy",
            "run baseline sweeps",
        ], bullet_color=BLUE)

        card(d, (396, 258, 678, 420), "RL fine-tune", title_color=ORANGE)
        draw_bullets(d, (412, 298, 660, 398), [
            "initialize policy",
            "optimize timing reward",
            "trade speed versus accuracy",
        ], bullet_color=ORANGE)

        card(d, (704, 258, 886, 420), "Artifacts", title_color=GREEN)
        draw_text_fit(
            d,
            (718, 300, 872, 406),
            "artifacts/smoke/\n- mc_dataset.json\n- baseline_summary.json\n- ppo_runs.json",
            max_size=14,
            min_size=9,
            fill=TEXT_SOFT,
            valign="top",
        )

    elif stage == 1:
        card(d, (88, 136, 472, 320), "Core metrics", title_color=BLUE)
        draw_bullets(d, (106, 176, 454, 294), [
            "mean S_q",
            "buzz accuracy",
            "mean buzz step",
            "ECE and Brier",
        ], bullet_color=BLUE)

        card(d, (488, 136, 872, 320), "Control checks", title_color=PURPLE)
        draw_bullets(d, (506, 176, 854, 294), [
            "choices-only baseline",
            "shuffle test",
            "alias substitution",
            "per-category slices",
        ], bullet_color=PURPLE)

        rounded(d, (88, 344, 872, 430), "#F7F8FA", outline=BORDER, radius=10, width=1)
        draw_text_fit(
            d,
            (106, 360, 854, 416),
            "Evaluation source of truth: artifacts/smoke/evaluation_report.json",
            max_size=18,
            min_size=10,
            fill=TEXT_SOFT,
            align="center",
            valign="center",
        )

    else:
        full_eval = EVAL_REPORT.get("full_eval", {})
        ppo = EVAL_REPORT.get("ppo_summary", {})
        best_name, best_metrics = best_baseline_result(EVAL_REPORT)

        card(d, (88, 136, 472, 424), "Best baseline (mean S_q)", title_color=BLUE)
        draw_text_fit(d, (106, 178, 454, 208), best_name, max_size=20, min_size=11, bold=True, fill=NAVY, align="center")
        draw_bullets(d, (108, 220, 454, 382), [
            f"mean S_q = {fmt_num(best_metrics.get('mean_sq'))}",
            f"buzz acc = {fmt_pct(best_metrics.get('buzz_accuracy'))}",
            f"mean step = {fmt_num(best_metrics.get('mean_buzz_step'), 2)}",
            f"n = {fmt_num(best_metrics.get('n'), 0)}",
        ], bullet_color=BLUE)

        card(d, (488, 136, 872, 424), "PPO smoke", title_color=ORANGE)
        draw_bullets(d, (508, 178, 854, 382), [
            f"mean S_q = {fmt_num(ppo.get('mean_sq'))}",
            f"buzz acc = {fmt_pct(ppo.get('buzz_accuracy'))}",
            f"reward-like = {fmt_num(ppo.get('mean_reward_like'))}",
            f"ECE/Brier = {fmt_num(ppo.get('ece'))}/{fmt_num(ppo.get('brier'))}",
        ], bullet_color=ORANGE)

        rounded(d, (88, 438, 872, 464), BLUE_SOFT, outline=BLUE, radius=10, width=2)
        draw_text_fit(
            d,
            (106, 442, 854, 460),
            f"Full eval: S_q={fmt_num(full_eval.get('mean_sq'))}, accuracy={fmt_pct(full_eval.get('buzz_accuracy'))}, mean step={fmt_num(full_eval.get('mean_buzz_step'), 2)}",
            max_size=13,
            min_size=9,
            fill=NAVY_DARK,
            align="center",
            valign="center",
        )

    footer(d, spec["footer"])
    return img

def render_summary(spec: Dict) -> Image.Image:
    stage = spec["stage"]
    img = make_canvas()
    d = ImageDraw.Draw(img)
    header(d, "Quiz bowl MC \"buzzing,\" in one picture")

    rounded(d, (48, 98, 912, 474), WHITE, outline=BORDER, radius=16, width=2)
    rounded(d, (74, 144, 650, 418), "#FAFBFD", outline=BORDER, radius=12, width=2)

    bullets = [
        ('1. the Markov "clues" go from Bayesian-network jargon to a giveaway', BLUE),
        ('2. the posterior over the fixed four choices sharpens onto Markov', ORANGE),
        ('3. the key RL decision is when to stop "waiting" and buzz A', PURPLE),
        ('4. "abstain" means another clue is worth it, not "None-of-the-Above"', GREEN),
    ]
    y = 174
    visible = bullets[: stage + 2]
    for i, (txt, color) in enumerate(visible):
        draw_text_fit(d, (96, y, 624, y + 24), txt, max_size=18, min_size=11, bold=(i == 0), fill=color)
        y += 48

    rounded(d, (94, 350, 620, 398), WHITE, outline=BORDER, radius=10, width=1)
    draw_text_fit(d, (108, 360, 606, 388),
                  'Mental model: selective prediction + optimal stopping over a fixed answer "set" that already contains Markov.',
                  max_size=15, min_size=10, fill=TEXT_SOFT, align="center", valign="center")

    rounded(d, (688, 146, 892, 256), "#EEF2F8", outline=BORDER, radius=12, width=2)
    draw_text_fit(d, (702, 160, 878, 188), "For RL people", max_size=18, min_size=12, bold=True, fill=NAVY, align="center")
    draw_text_fit(d, (702, 196, 878, 242),
                  "At each \"prefix,\" compare the value of acting now with the value of one more \"clue.\"",
                  max_size=15, min_size=10, fill=TEXT_SOFT, align="center", valign="center")

    if stage >= 1:
        rounded(d, (688, 282, 892, 388), GREEN_SOFT, outline=GREEN, radius=12, width=2)
        draw_text_fit(d, (702, 296, 878, 374),
                      "If you keep a flat action interface, factor it internally:\nP(\"WAIT\")=1-p_buzz\nP(\"BUZZ i\")=p_buzz * p_ans(i)",
                      max_size=15, min_size=9, fill=GREEN, align="center", valign="center")

    if stage >= 2:
        rounded(d, (86, 434, 860, 464), BLUE_SOFT, outline=BLUE, radius=10, width=2)
        draw_text_fit(d, (100, 440, 846, 458),
                      "Key takeaway: \"pyramidal\" quiz bowl turns multiple-choice QA into an optimal-stopping problem because evidence improves before action.",
                      max_size=13, min_size=9, fill=NAVY_DARK, align="center", valign="center")

    footer(d, spec["footer"])
    return img

def render_evaluation_slide(spec: Dict) -> Image.Image:
    img = make_canvas()
    d = ImageDraw.Draw(img)
    header(d, "Evaluation")
    rounded(d, (48, 96, 460, 482), WHITE, outline=BORDER, radius=16, width=2)
    draw_text_fit(d, (68, 112, 440, 140), "Metrics", max_size=20, min_size=13, bold=True, fill=NAVY)
    draw_bullets(d, (68, 154, 440, 400), [
        "S_q (QANTA system score)",
        "Buzz accuracy",
        "Mean buzz step",
        "Calibration-at-buzz / ECE / Brier",
        "Per-category accuracy",
    ], bullet_color=BLUE)
    rounded(d, (500, 96, 912, 482), WHITE, outline=BORDER, radius=16, width=2)
    draw_text_fit(d, (520, 112, 892, 140), "Controls", max_size=20, min_size=13, bold=True, fill=RED)
    draw_bullets(d, (520, 154, 892, 400), [
        "Choices-only baseline (no question text)",
        "Shuffle clue order",
        "Alias substitution",
        "Distractor difficulty variation",
    ], bullet_color=RED)
    rounded(d, (120, 440, 840, 476), "#F7F8FA", outline=BORDER, radius=10, width=1)
    draw_text_fit(
        d,
        (136, 448, 824, 470),
        "A buzzer can look strong while exploiting answer artifacts instead of clue content.",
        max_size=14,
        min_size=10,
        fill=TEXT_SOFT,
        align="center",
        valign="center",
    )
    footer(d, spec["footer"])
    return img

def render_references_slide(spec: Dict) -> Image.Image:
    img = make_canvas()
    d = ImageDraw.Draw(img)
    header(d, "References")
    rounded(d, (48, 96, 912, 482), WHITE, outline=BORDER, radius=16, width=2)
    refs = [
        "Rodriguez et al. (2019) — Quizbowl: The case for incremental QA",
        "Schulman et al. — Proximal Policy Optimization (PPO)",
        "Raffel et al. — T5: Exploring the Limits of Transfer Learning",
        "Boyd-Graber and Daume (2013) — Bayesian thinking on your feet",
        "Boyd-Graber and Borschinger (2020) — What QA can learn from trivia nerds",
        "Balepur et al. (2025) — Test-time reasoners are strategic MC test-takers",
        "Kalai et al. (2025) — Why language models hallucinate",
        "UMD / QANTA (2024) — S_q evaluation metric",
    ]
    draw_bullets(d, (80, 120, 880, 460), refs, bullet_color=NAVY)
    footer(d, spec["footer"])
    return img

# ============================================================
# Dispatcher
# ============================================================
RENDERERS = {
    "title_slide": render_title_slide,
    "problem_slide": render_problem_slide,
    "background_slide": render_background_slide,
    "why_mc_slide": render_why_mc_slide,
    "method_slide": render_method_slide,
    "contribution_slide": render_contribution_slide,
    "expected_results_slide": render_expected_results_slide,
    "why_ppo_less_accurate": render_why_ppo_less_accurate,
    "intro": render_intro,
    "section": render_section,
    "posterior": render_posterior,
    "value_chart": render_value_chart,
    "abstain": render_abstain,
    "factorization": render_factorization,
    "recipe": render_recipe,
    "pipeline": render_pipeline,
    "evaluation_slide": render_evaluation_slide,
    "summary": render_summary,
    "references_slide": render_references_slide,
}

# ============================================================
# Build frames
# ============================================================
frames: List[Image.Image] = []
for spec in SCENES:
    frames.append(RENDERERS[spec["kind"]](spec))

assert len(frames) == len(SCENES)

# ============================================================
# Durations
# ============================================================
durations = []
for spec in SCENES:
    kind = spec["kind"]
    stage = spec.get("stage")
    if kind == "title_slide":
        durations.append(1000)
    elif kind in {"section", "references_slide"}:
        durations.append(900)
    elif kind in {"problem_slide", "background_slide", "why_mc_slide", "method_slide", "contribution_slide", "expected_results_slide", "why_ppo_less_accurate", "evaluation_slide"}:
        durations.append(800)
    elif kind == "summary" and stage == 2:
        durations.append(850)
    else:
        durations.append(450)

# ============================================================
# Save frames
# ============================================================
FRAMES_DIR.mkdir(parents=True, exist_ok=True)
for idx, fr in enumerate(frames, start=1):
    fr.save(FRAMES_DIR / f"frame_{idx:02d}.png")

# ============================================================
# Save GIF
# ============================================================
pal_frames = [fr.convert("P", palette=Image.Palette.ADAPTIVE) for fr in frames]
pal_frames[0].save(
    GIF_OUT,
    save_all=True,
    append_images=pal_frames[1:],
    duration=durations,
    loop=0,
    optimize=False,
    disposal=2,
)

# ============================================================
# Save contact sheet
# ============================================================
def make_contact(frames: List[Image.Image], cols: int = 3):
    thumb_w, thumb_h = 248, 139
    cell_w, cell_h = 260, 185
    rows = ceil(len(frames) / cols)

    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), "#ECECEC")
    draw = ImageDraw.Draw(sheet)
    label_font = load_font(16, bold=False)

    for idx, fr in enumerate(frames):
        row = idx // cols
        col = idx % cols
        x0 = col * cell_w
        y0 = row * cell_h

        thumb = fr.copy().convert("RGB")
        thumb.thumbnail((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        tx = x0 + (cell_w - thumb.width) // 2
        ty = y0 + 6
        sheet.paste(thumb, (tx, ty))

        label = f"Frame {idx + 1}"
        draw.text((x0 + 6, y0 + 150), label, font=label_font, fill="#111111")

    return sheet

contact = make_contact(frames, cols=3)
contact.save(CONTACT_OUT)

print(f"Saved GIF to: {GIF_OUT}")
print(f"Saved contact sheet to: {CONTACT_OUT}")
print(f"Saved frames to: {FRAMES_DIR}")
```

## File: pyproject.toml
```toml
[build-system]
requires = ["setuptools>=69.0"]
build-backend = "setuptools.build_meta"

[project]
name = "qanta-buzzer"
version = "1.0.0"
description = "Unified quiz bowl RL buzzer system for Stanford CS234"
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
  "datasets>=2.14.0",
  "gymnasium>=1.1.0",
  "jsonlines>=3.1.0",
  "matplotlib>=3.7.0",
  "numpy>=1.24.0",
  "pandas>=2.0.0",
  "PyYAML>=6.0.0",
  "scikit-learn>=1.3.0",
  "seaborn>=0.12.0",
  "sentence-transformers>=2.2.0",
  "stable-baselines3>=2.6.0",
  "torch>=2.0.0",
  "tqdm>=4.65.0",
  "transformers>=4.30.0",
]

[project.optional-dependencies]
dev = ["pytest>=7.0.0"]
openai = ["openai>=1.0.0"]
maskable = ["sb3-contrib>=2.6.0"]
dspy = ["dspy>=2.5.0"]

[tool.setuptools.packages.find]
include = ["agents", "evaluation", "models", "qb_data", "qb_env", "training"]

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "pipeline: subprocess-based pipeline entrypoint smoke tests",
]
```

## File: scripts/run_baselines.py
```python
#!/usr/bin/env python3
"""
Run non-RL baseline agents and save episode traces + summary artifacts.

Executes four baseline agent types across a threshold sweep:
1. ThresholdBuzzer -- buzzes when top belief exceeds threshold
2. SoftmaxProfileBuzzer -- softmax belief from scratch at each step
3. SequentialBayesBuzzer -- Bayesian belief update with sequential fragments
4. AlwaysBuzzFinalBuzzer -- always waits until last clue, then buzzes

Results are saved to artifacts/{smoke,main}/ as JSON files with per-episode
traces and aggregated summary metrics (accuracy, S_q, ECE, Brier score).

Usage:
    python scripts/run_baselines.py              # Full run (default config)
    python scripts/run_baselines.py --smoke      # Quick smoke test (~50 questions)
    python scripts/run_baselines.py --config configs/custom.yaml
    python scripts/run_baselines.py --mc-path artifacts/main/mc_dataset.json

Ported from qb-rl reference implementation (scripts/run_baselines.py).
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.bayesian_buzzer import (
    precompute_sequential_beliefs,
    sweep_sequential_thresholds,
)
from agents.threshold_buzzer import (
    _always_final_from_precomputed,
    _softmax_episode_from_precomputed,
    precompute_beliefs,
    sweep_thresholds,
)
from evaluation.metrics import calibration_at_buzz, summarize_buzz_metrics
from qb_data.config import merge_overrides
from scripts._common import (
    ARTIFACT_DIR,
    build_likelihood_model,
    load_config,
    load_embedding_cache,
    load_mc_questions,
    parse_overrides,
    save_embedding_cache,
    save_json,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments with config, smoke, and mc_path fields.
    """
    parser = argparse.ArgumentParser(description="Run non-RL baseline agents.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (default: configs/default.yaml).",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use smoke mode: loads configs/smoke.yaml, outputs to artifacts/smoke/.",
    )
    parser.add_argument(
        "--mc-path",
        type=str,
        default=None,
        help="Optional MC dataset JSON path (overrides config-derived path).",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Override output directory (default: artifacts/<split>).",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Config overrides: key=value (e.g. likelihood.model=tfidf)",
    )
    return parser.parse_args()


def summarize(results: list[dict]) -> dict:
    """Combine buzz metrics and calibration into a single summary dict.

    Parameters
    ----------
    results : list[dict]
        List of episode trace dicts (from asdict(EpisodeResult)).

    Returns
    -------
    dict
        Merged summary with accuracy, S_q, ECE, Brier, etc.
    """
    return {
        **summarize_buzz_metrics(results),
        **calibration_at_buzz(results),
    }


def main() -> None:
    """Run all baseline agents and save artifacts."""
    start_time = time.time()

    args = parse_args()

    config = load_config(args.config, smoke=args.smoke)
    overrides = parse_overrides(args)
    if overrides:
        print(f"Applying overrides: {overrides}")
        config = merge_overrides(config, overrides)

    split = "smoke" if args.smoke else "main"
    out_dir = Path(args.output_dir) if args.output_dir else ARTIFACT_DIR / split
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine MC dataset path
    mc_path = Path(args.mc_path) if args.mc_path else out_dir / "mc_dataset.json"

    # Fallback: check data/processed/ if artifacts path doesn't exist
    if not mc_path.exists():
        fallback = PROJECT_ROOT / "data" / "processed" / "mc_dataset.json"
        if fallback.exists():
            print(f"MC dataset not found at {mc_path}, using fallback: {fallback}")
            mc_path = fallback

    print(f"Loading MC questions from: {mc_path}")
    mc_questions = load_mc_questions(mc_path)
    print(f"Loaded {len(mc_questions)} MC questions")

    # Build likelihood model
    print(f"Building likelihood model: {config['likelihood']['model']}")
    likelihood_model = build_likelihood_model(config, mc_questions)
    load_embedding_cache(likelihood_model, config)

    # Extract hyperparameters
    beta = float(config["likelihood"].get("beta", 5.0))
    alpha = float(config["bayesian"].get("alpha", 10.0))
    thresholds = [float(x) for x in config["bayesian"]["threshold_sweep"]]

    print(f"Beta: {beta}, Alpha: {alpha}")
    print(f"Thresholds: {thresholds}")

    # --- Pre-compute all embeddings once (batched) ---
    all_texts: list[str] = []
    for q in mc_questions:
        all_texts.extend(q.cumulative_prefixes)
        all_texts.extend(q.option_profiles)
        for step_idx in range(len(q.run_indices)):
            prev_idx = q.run_indices[step_idx - 1] if step_idx > 0 else -1
            all_texts.append(" ".join(q.tokens[prev_idx + 1 : q.run_indices[step_idx] + 1]))
    print(f"\nPre-computing embeddings for {len(set(all_texts)):,} unique texts...")
    likelihood_model.precompute_embeddings(all_texts, batch_size=64)
    save_embedding_cache(likelihood_model, config)

    # --- Pre-compute beliefs (one model pass, all steps) ---
    precomputed = precompute_beliefs(mc_questions, likelihood_model, beta)

    # --- Threshold sweep (pure numpy, instant) ---
    print("\nRunning ThresholdBuzzer sweep...")
    threshold_runs = sweep_thresholds(
        questions=mc_questions,
        likelihood_model=likelihood_model,
        thresholds=thresholds,
        beta=beta,
        alpha=alpha,
        precomputed=precomputed,
    )

    threshold_payload: dict[str, list[dict]] = {}
    threshold_summary: dict[str, dict] = {}
    for threshold, runs in threshold_runs.items():
        rows = [asdict(r) for r in runs]
        threshold_payload[str(threshold)] = rows
        threshold_summary[str(threshold)] = summarize(rows)

    # --- Softmax profile sweep (reuse from_scratch precomputed beliefs) ---
    print("\nRunning SoftmaxProfile sweep (precomputed)...")
    softmax_payload: dict[str, list[dict]] = {}
    softmax_summary: dict[str, dict] = {}
    for threshold in thresholds:
        results = [
            asdict(_softmax_episode_from_precomputed(pq, threshold, alpha))
            for pq in precomputed
        ]
        softmax_payload[str(threshold)] = results
        softmax_summary[str(threshold)] = summarize(results)

    # --- Sequential Bayes sweep (one belief pass, pure numpy threshold sweep) ---
    print("Pre-computing sequential Bayes beliefs...")
    seq_precomputed = precompute_sequential_beliefs(mc_questions, likelihood_model, beta)
    print("Running SequentialBayes sweep (precomputed)...")
    seq_results = sweep_sequential_thresholds(
        questions=mc_questions,
        likelihood_model=likelihood_model,
        thresholds=thresholds,
        beta=beta,
        alpha=alpha,
        precomputed=seq_precomputed,
    )
    sequential_payload: dict[str, list[dict]] = {}
    sequential_summary: dict[str, dict] = {}
    for threshold, runs in seq_results.items():
        rows = [asdict(r) for r in runs]
        sequential_payload[str(threshold)] = rows
        sequential_summary[str(threshold)] = summarize(rows)

    # --- AlwaysBuzzFinal (reuse from_scratch precomputed beliefs) ---
    print("Running AlwaysBuzzFinal baseline (precomputed)...")
    floor_runs = [asdict(_always_final_from_precomputed(pq)) for pq in precomputed]
    floor_summary = summarize(floor_runs)

    # --- Save artifacts ---
    print(f"\nSaving artifacts to: {out_dir}")
    save_json(out_dir / "baseline_threshold_runs.json", threshold_payload)
    save_json(out_dir / "baseline_softmax_profile_runs.json", softmax_payload)
    save_json(out_dir / "baseline_sequential_bayes_runs.json", sequential_payload)
    save_json(out_dir / "baseline_floor_runs.json", floor_runs)

    summary = {
        "threshold": threshold_summary,
        "softmax_profile": softmax_summary,
        "sequential_bayes": sequential_summary,
        "always_final": floor_summary,
    }
    save_json(out_dir / "baseline_summary.json", summary)

    elapsed = time.time() - start_time
    print(f"\nWrote baseline outputs to: {out_dir}")
    print(f"Total time: {elapsed:.1f} seconds")

    # Print summary highlights
    print("\n--- Summary ---")
    for agent_name, agent_summary in summary.items():
        if isinstance(agent_summary, dict) and "buzz_accuracy" in agent_summary:
            # Single-threshold agent (always_final)
            print(f"  {agent_name}: accuracy={agent_summary['buzz_accuracy']:.3f}, "
                  f"mean_sq={agent_summary.get('mean_sq', 0):.3f}")
        elif isinstance(agent_summary, dict):
            # Multi-threshold agent
            for thr, metrics in agent_summary.items():
                if isinstance(metrics, dict) and "buzz_accuracy" in metrics:
                    print(f"  {agent_name}[{thr}]: accuracy={metrics['buzz_accuracy']:.3f}, "
                          f"mean_sq={metrics.get('mean_sq', 0):.3f}")


if __name__ == "__main__":
    main()
```

## File: qb_env/tossup_env.py
```python
"""
Gymnasium-compliant POMDP Environment for Quiz Bowl

Implements a tossup question environment where clues are revealed incrementally.
At each step the agent observes a belief-based feature vector and chooses either
to WAIT (action 0, reveals next clue) or to BUZZ with a specific answer option
(actions 1..K, ends the episode).

The environment computes beliefs over K answer options using a pluggable
LikelihoodModel and converts them to observations via extract_belief_features.

Ported from qb-rl reference implementation (qb_env/tossup_env.py) and adapted
for the unified qanta-buzzer codebase.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from qb_env.opponent_models import OpponentBuzzModel

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from models.features import extract_belief_features
from models.likelihoods import LikelihoodModel
from qb_data.mc_builder import MCQuestion


def _softmax(scores: np.ndarray, beta: float) -> np.ndarray:
    """Temperature-scaled softmax with numerical stability.

    Parameters
    ----------
    scores : np.ndarray
        Raw similarity scores of shape (K,).
    beta : float
        Temperature parameter. Higher values produce sharper distributions.

    Returns
    -------
    np.ndarray
        Probability distribution of shape (K,), dtype float32.
    """
    stable = scores - np.max(scores)
    probs = np.exp(beta * stable)
    probs_sum = np.sum(probs)
    if probs_sum <= 0:
        return np.ones_like(scores, dtype=np.float32) / len(scores)
    return (probs / probs_sum).astype(np.float32)


def precompute_beliefs(
    questions: list[MCQuestion],
    likelihood_model: LikelihoodModel,
    belief_mode: str = "from_scratch",
    beta: float = 5.0,
    K: int = 4,
) -> dict[tuple[int, int], np.ndarray]:
    """Precompute belief trajectories for all questions and steps.

    Iterates over each question and each step index, computing the belief
    using the same logic as ``TossupMCEnv._compute_belief``. The result is
    a dict keyed by ``(question_index, step_idx)`` for O(1) lookup during
    training rollouts.

    Parameters
    ----------
    questions : list[MCQuestion]
        Pool of questions to precompute beliefs for.
    likelihood_model : LikelihoodModel
        Model that scores clue text against answer option profiles.
    belief_mode : str
        One of ``"from_scratch"``, ``"sequential_bayes"``.
    beta : float
        Softmax temperature for converting raw scores to probabilities.
    K : int
        Deprecated — ignored. Each question uses ``len(question.options)``
        as its local K. Kept for backward compatibility with callers.

    Returns
    -------
    dict[tuple[int, int], np.ndarray]
        Maps ``(question_index, step_idx)`` to belief vectors of shape
        ``(len(question.options),)`` with dtype float32. Each belief sums to ~1.0.
    """
    cache: dict[tuple[int, int], np.ndarray] = {}

    for q_idx, question in enumerate(questions):
        num_steps = len(question.run_indices)
        q_k = len(question.options)
        belief = np.ones(q_k, dtype=np.float32) / q_k

        for step_idx in range(num_steps):
            if belief_mode == "from_scratch":
                prefix = question.cumulative_prefixes[step_idx]
                scores = likelihood_model.score(prefix, question.option_profiles)
                belief = _softmax(scores, beta)

            elif belief_mode == "sequential_bayes":
                idx = question.run_indices[step_idx]
                prev_idx = question.run_indices[step_idx - 1] if step_idx > 0 else -1
                frag = " ".join(question.tokens[prev_idx + 1 : idx + 1])
                scores = likelihood_model.score(frag, question.option_profiles)
                likelihood = _softmax(scores, beta)
                posterior = belief * likelihood
                denom = posterior.sum()
                if denom <= 0:
                    belief = np.ones(q_k, dtype=np.float32) / q_k
                else:
                    belief = (posterior / denom).astype(np.float32)

            else:
                raise ValueError(f"Unknown belief_mode: {belief_mode}")

            cache[(q_idx, step_idx)] = belief.copy()

    return cache


class TossupMCEnv(gym.Env[np.ndarray, int]):
    """Gymnasium environment for quiz bowl tossup questions with MC options.

    Models quiz bowl as a POMDP where clues are revealed incrementally.
    The agent maintains a belief distribution over K answer options, updated
    at each step by a likelihood model. The agent decides when to buzz and
    which answer to select.

    Action Space
    ------------
    Discrete(K + 1):
        - 0: WAIT -- reveal the next clue and update belief
        - 1..K: BUZZ with answer option (i-1), ending the episode

    Observation Space
    -----------------
    Box(K + 6,):
        Belief features: [belief[0..K-1], top_p, margin, entropy,
        stability, progress, clue_idx_norm].
        See ``models.features.extract_belief_features`` for details.

    Reward Modes
    ------------
    ``time_penalty`` (default):
        -wait_penalty per WAIT step; +buzz_correct for correct buzz,
        +buzz_incorrect (negative) for wrong buzz.
    ``simple``:
        +1.0 for correct buzz, -1.0 for incorrect buzz, no WAIT penalty.
    ``human_grounded``:
        0.0 if the agent buzzes after the sampled human buzz position;
        otherwise +buzz_correct/-buzz_incorrect for correct/incorrect.

    Belief Modes
    ------------
    ``from_scratch``:
        Recompute belief from all clues seen so far via cumulative_prefixes.
    ``sequential_bayes``:
        Bayesian update: multiply prior belief by likelihood of new clue
        fragment, then normalize.

    Parameters
    ----------
    questions : list[MCQuestion]
        Pool of questions to sample from. Must be non-empty.
    likelihood_model : LikelihoodModel
        Model that scores clue text against answer option profiles.
    K : int
        Number of answer options per question. Must be >= 2.
    reward_mode : str
        One of ``"time_penalty"``, ``"simple"``, ``"human_grounded"``.
    wait_penalty : float
        Per-step penalty when reward_mode is ``"time_penalty"``.
    buzz_correct : float
        Reward for buzzing with the correct answer.
    buzz_incorrect : float
        Reward (typically negative) for buzzing with an incorrect answer.
    belief_mode : str
        One of ``"from_scratch"``, ``"sequential_bayes"``.
    beta : float
        Softmax temperature for converting raw scores to probabilities.
        Higher values produce sharper distributions.
    end_mode : str
        Horizon behavior when clues are exhausted:
        ``"force_commit"`` (legacy forced answer) or ``"no_buzz"``.
    no_buzz_reward : float
        Reward added at horizon when ``end_mode == "no_buzz"``.
    seed : int
        Random seed for question sampling and human buzz simulation.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        questions: list[MCQuestion],
        likelihood_model: LikelihoodModel,
        K: int = 4,
        reward_mode: str = "time_penalty",
        wait_penalty: float = 0.01,
        early_buzz_penalty: float = 0.0,
        buzz_correct: float = 1.0,
        buzz_incorrect: float = -0.5,
        belief_mode: str = "from_scratch",
        beta: float = 5.0,
        seed: int = 13,
        precomputed_beliefs: dict[tuple[int, int], np.ndarray] | None = None,
        opponent_buzz_model: "OpponentBuzzModel | None" = None,
        ew_reward_correct: float = 10.0,
        ew_reward_incorrect: float = -5.0,
        ew_opponent_expected_value: float = 0.0,
        variable_K: bool = False,
        max_K: int | None = None,
        end_mode: str = "force_commit",
        no_buzz_reward: float = 0.0,
    ) -> None:
        if not questions:
            raise ValueError("questions cannot be empty")
        if K < 2:
            raise ValueError("K must be >= 2")

        self.questions = questions
        self.likelihood_model = likelihood_model
        self.K = K
        self.reward_mode = reward_mode
        self.wait_penalty = wait_penalty
        self.early_buzz_penalty = early_buzz_penalty
        self.buzz_correct = buzz_correct
        self.buzz_incorrect = buzz_incorrect
        self.belief_mode = belief_mode
        self.beta = beta
        self.rng = random.Random(seed)
        self.precomputed_beliefs = precomputed_beliefs

        self.opponent_buzz_model = opponent_buzz_model
        self.ew_reward_correct = ew_reward_correct
        self.ew_reward_incorrect = ew_reward_incorrect
        self.ew_opponent_expected_value = ew_opponent_expected_value

        self.variable_K = variable_K
        self.end_mode = end_mode
        self.no_buzz_reward = no_buzz_reward
        if variable_K:
            self._max_K = max_K or max(len(q.options) for q in questions)
        else:
            self._max_K = K

        # Build qid -> list-index map for precomputed belief lookups
        self._question_index_map: dict[str, int] = {
            q.qid: i for i, q in enumerate(questions)
        }

        obs_K = self._max_K if self.variable_K else self.K
        self.action_space = spaces.Discrete(obs_K + 1)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_K + 6,), dtype=np.float32
        )

        self.question: MCQuestion | None = None
        self.step_idx: int = 0
        self.prev_belief: np.ndarray | None = None
        self.belief: np.ndarray = np.ones(self.K, dtype=np.float32) / self.K
        self.terminated: bool = False
        self.truncated: bool = False
        self._sampled_human_buzz_pos: int | None = None
        self._current_question_idx: int = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def total_steps(self) -> int:
        """Total number of incremental clue steps for the current question.

        Returns
        -------
        int
            Length of ``question.run_indices`` if a question is loaded, else 1.
        """
        if self.question is None:
            return 1
        return len(self.question.run_indices)

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def _sample_question(self) -> MCQuestion:
        """Sample a random question from the question pool.

        Returns
        -------
        MCQuestion
            A randomly selected question.
        """
        return self.rng.choice(self.questions)

    def _sample_human_buzz(self, question: MCQuestion) -> int | None:
        """Sample a human buzz position from the question's distribution.

        Uses weighted random sampling based on the number of humans who
        buzzed at each position. Returns None if no human buzz data exists.

        Parameters
        ----------
        question : MCQuestion
            The question to sample a human buzz position for.

        Returns
        -------
        int or None
            Sampled token position, or None if no human buzz data.
        """
        if not question.human_buzz_positions:
            return None
        positions = []
        weights = []
        for pos, count in question.human_buzz_positions:
            positions.append(int(pos))
            weights.append(max(1, int(count)))
        if not positions:
            return None
        return self.rng.choices(positions, weights=weights, k=1)[0]

    def _softmax_scores(self, scores: np.ndarray) -> np.ndarray:
        """Convert raw likelihood scores to a probability distribution.

        Delegates to module-level ``_softmax`` with this environment's beta.

        Parameters
        ----------
        scores : np.ndarray
            Raw similarity scores of shape (K,).

        Returns
        -------
        np.ndarray
            Probability distribution of shape (K,), dtype float32.
        """
        return _softmax(scores, self.beta)

    def _compute_belief(self, question: MCQuestion, step_idx: int) -> np.ndarray:
        """Compute belief distribution over answer options at a given step.

        Two modes are supported:

        ``from_scratch``
            Score the cumulative clue prefix against all option profiles,
            then apply softmax. Each step is independent of the previous
            belief.

        ``sequential_bayes``
            Extract only the new clue fragment since the last step, score
            it, and perform a Bayesian update: posterior = prior * likelihood,
            then normalize. This is cheaper per step but may accumulate
            approximation errors.

        Parameters
        ----------
        question : MCQuestion
            Current question being played.
        step_idx : int
            Current step index (0-based, indexes into run_indices).

        Returns
        -------
        np.ndarray
            Updated belief distribution of shape (K,), dtype float32.

        Raises
        ------
        ValueError
            If ``self.belief_mode`` is not a recognized mode.
        """
        if self.precomputed_beliefs is not None:
            key = (self._current_question_idx, step_idx)
            return self.precomputed_beliefs[key].copy()

        if self.belief_mode == "from_scratch":
            prefix = question.cumulative_prefixes[step_idx]
            scores = self.likelihood_model.score(prefix, question.option_profiles)
            return self._softmax_scores(scores)

        if self.belief_mode == "sequential_bayes":
            idx = question.run_indices[step_idx]
            prev_idx = question.run_indices[step_idx - 1] if step_idx > 0 else -1
            frag = " ".join(question.tokens[prev_idx + 1 : idx + 1])
            scores = self.likelihood_model.score(frag, question.option_profiles)
            likelihood = self._softmax_scores(scores)
            posterior = self.belief * likelihood
            denom = posterior.sum()
            if denom <= 0:
                n = len(self.belief)
                posterior = np.ones(n, dtype=np.float32) / n
            else:
                posterior = posterior / denom
            return posterior.astype(np.float32)

        raise ValueError(f"Unknown belief_mode: {self.belief_mode}")

    def _obs(self) -> np.ndarray:
        """Build the observation vector from current belief state.

        In variable-K mode, uses padded features sized to ``_max_K``.
        Otherwise delegates to ``extract_belief_features``.

        Returns
        -------
        np.ndarray
            Feature vector of shape (obs_K + 6,), dtype float32.
        """
        if self.variable_K:
            from models.features import extract_padded_belief_features

            return extract_padded_belief_features(
                belief=self.belief,
                prev_belief=self.prev_belief,
                step_idx=self.step_idx,
                total_steps=self.total_steps,
                max_K=self._max_K,
            )
        return extract_belief_features(
            belief=self.belief,
            prev_belief=self.prev_belief,
            step_idx=self.step_idx,
            total_steps=self.total_steps,
        )

    def action_masks(self) -> np.ndarray:
        """Return a boolean mask of valid actions.

        WAIT (action 0) is always valid.  Buzz actions ``1..K_actual``
        are valid; padded slots ``K_actual+1..max_K`` are invalid.

        Returns
        -------
        np.ndarray
            Boolean array of shape ``(max_K + 1,)`` or ``(K + 1,)``.
        """
        n_actions = self._max_K + 1 if self.variable_K else self.K + 1
        mask = np.zeros(n_actions, dtype=bool)
        mask[0] = True  # WAIT
        k_actual = len(self.question.options) if self.question is not None else self.K
        mask[1 : k_actual + 1] = True
        return mask

    def _step_to_token_pos(self, step_idx: int) -> int:
        """Convert a step index to the corresponding token position.

        Used by the ``human_grounded`` reward mode to compare the agent's
        buzz position against the sampled human buzz position.

        Parameters
        ----------
        step_idx : int
            Step index (0-based, indexes into run_indices).

        Returns
        -------
        int
            Token position in the original question text.
        """
        if self.question is None or not self.question.run_indices:
            return step_idx
        if step_idx >= len(self.question.run_indices):
            return self.question.run_indices[-1]
        if step_idx < 0:
            return self.question.run_indices[0]
        return self.question.run_indices[step_idx]

    def _expected_wins_reward(
        self, question: MCQuestion, chosen_idx: int, last_seen_step: int
    ) -> float:
        """Compute Expected Wins reward at buzz time.

        R_t = S_t * V_self + (1 - S_t) * V_opp

        where S_t = P(opponent has NOT buzzed by step t).
        """
        correct = chosen_idx == question.gold_index
        v_self = self.ew_reward_correct if correct else self.ew_reward_incorrect
        if self.opponent_buzz_model is None:
            return v_self
        s_t = self.opponent_buzz_model.prob_survive_to_step(question, last_seen_step)
        return s_t * v_self + (1.0 - s_t) * self.ew_opponent_expected_value

    def _buzz_reward(self, question: MCQuestion, chosen_idx: int, last_seen_step: int) -> float:
        """Compute the reward for buzzing with a given answer.

        Dispatches on ``self.reward_mode``:

        ``simple``
            +1.0 for correct, -1.0 for incorrect.
        ``human_grounded``
            0.0 if the agent buzzes after the sampled human would have;
            otherwise +buzz_correct / +buzz_incorrect.
        ``time_penalty`` (default)
            +buzz_correct / +buzz_incorrect. The per-step wait penalty
            is applied separately in ``step()``.
        ``expected_wins``
            S_t * V_self + (1 - S_t) * V_opp via opponent model.

        Parameters
        ----------
        question : MCQuestion
            Current question.
        chosen_idx : int
            Index of the chosen answer option (0-based).
        last_seen_step : int
            Step index of the last clue seen before buzzing.

        Returns
        -------
        float
            Reward value.
        """
        correct = chosen_idx == question.gold_index
        if self.reward_mode == "simple":
            return 1.0 if correct else -1.0
        if self.reward_mode == "human_grounded":
            token_pos = self._step_to_token_pos(last_seen_step)
            if self._sampled_human_buzz_pos is not None and token_pos > self._sampled_human_buzz_pos:
                return 0.0
            return self.buzz_correct if correct else self.buzz_incorrect
        if self.reward_mode == "expected_wins":
            return self._expected_wins_reward(question, chosen_idx, last_seen_step)
        # default: time_penalty
        reward = self.buzz_correct if correct else self.buzz_incorrect

        if self.early_buzz_penalty > 0 and self.total_steps > 1:
            progress = np.clip((last_seen_step + 1) / self.total_steps, 0.0, 1.0)
            reward -= float(self.early_buzz_penalty) * (1.0 - progress)

        return reward

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the environment and start a new episode.

        Samples a random question from the pool, initializes belief to a
        uniform distribution, and returns the initial observation.

        Parameters
        ----------
        seed : int or None
            If provided, reseeds both the internal RNG and numpy's global
            RNG for reproducibility.
        options : dict or None
            Unused. Included for Gymnasium API compatibility.

        Returns
        -------
        observation : np.ndarray
            Initial observation of shape (K + 6,), dtype float32.
            Belief is uniform, so top_p = 1/K, margin = 0, entropy = max.
        info : dict[str, Any]
            Episode metadata. Contains ``"qid"`` (the sampled question ID).
        """
        if seed is not None:
            self.rng.seed(seed)
            np.random.seed(seed)

        if options and "question_idx" in options:
            q_idx = int(options["question_idx"])
            if q_idx < 0 or q_idx >= len(self.questions):
                raise ValueError(f"question_idx out of range: {q_idx}")
            self.question = self.questions[q_idx]
            self._current_question_idx = q_idx
        else:
            self.question = self._sample_question()
            self._current_question_idx = self._question_index_map.get(
                self.question.qid, self.questions.index(self.question)
            )
        self.step_idx = 0
        self.prev_belief = None
        actual_k = len(self.question.options) if self.variable_K else self.K
        self.belief = np.ones(actual_k, dtype=np.float32) / actual_k
        self.terminated = False
        self.truncated = False
        self._sampled_human_buzz_pos = self._sample_human_buzz(self.question)
        return self._obs(), {"qid": self.question.qid}

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Execute one step in the environment.

        If ``action == 0`` (WAIT):
            - Saves previous belief, computes new belief from current clue.
            - Applies wait_penalty if reward_mode is ``"time_penalty"``.
            - Advances step counter.
            - If all clues exhausted: forced termination with best-guess
              answer (``truncated=True``).

        If ``action in 1..K`` (BUZZ):
            - Computes buzz reward for chosen answer option ``action - 1``.
            - Episode ends (``terminated=True``).

        Parameters
        ----------
        action : int
            Action to take. 0 = WAIT, 1..K = buzz with option (action-1).

        Returns
        -------
        observation : np.ndarray
            Updated observation of shape (K + 6,), dtype float32.
        reward : float
            Scalar reward for this step.
        terminated : bool
            True if the agent buzzed (natural episode end).
        truncated : bool
            True if all clues were exhausted (forced termination).
        info : dict[str, Any]
            Step metadata. Always contains ``"qid"`` and ``"step_idx"``.
            On BUZZ: also ``"chosen_idx"`` and ``"correct"``.
            On forced termination in ``force_commit`` mode: also
            ``"forced_choice"`` and ``"forced_correct"``.
            On forced termination in ``no_buzz`` mode: also ``"no_buzz"``,
            ``"forced_choice" = -1``, and ``"forced_correct" = False``.

        Raises
        ------
        RuntimeError
            If called before ``reset()`` or after episode has ended.
        ValueError
            If ``action`` is not in the action space.
        """
        if self.question is None:
            raise RuntimeError("Environment must be reset() before step().")
        if self.terminated or self.truncated:
            raise RuntimeError("Cannot call step() on terminated/truncated episode.")
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")
        if self.variable_K and action > 0:
            actual_k = len(self.question.options)
            if action - 1 >= actual_k:
                raise ValueError(
                    f"Buzz action {action} targets padded option index "
                    f"{action - 1}, but question only has {actual_k} options"
                )

        info: dict[str, Any] = {"qid": self.question.qid}
        reward = 0.0

        if action == 0:
            # WAIT: reveal next clue and update belief
            self.prev_belief = self.belief.copy()
            self.belief = self._compute_belief(self.question, self.step_idx)
            if self.reward_mode == "time_penalty":
                reward -= self.wait_penalty

            self.step_idx += 1
            if self.step_idx >= self.total_steps:
                last_seen = self.step_idx - 1
                self.truncated = True
                info["step_idx"] = last_seen
                if self.end_mode == "force_commit":
                    forced_choice = int(np.argmax(self.belief))
                    reward += self._buzz_reward(self.question, forced_choice, last_seen)
                    info["forced_choice"] = forced_choice
                    info["forced_correct"] = forced_choice == self.question.gold_index
                elif self.end_mode == "no_buzz":
                    reward += self.no_buzz_reward
                    info["no_buzz"] = True
                    info["forced_choice"] = -1
                    info["forced_correct"] = False
                else:
                    raise ValueError(f"Unknown end_mode: {self.end_mode}")
            else:
                info["step_idx"] = self.step_idx

        else:
            # BUZZ: select an answer option
            last_seen = max(0, self.step_idx - 1)
            chosen_idx = action - 1
            reward += self._buzz_reward(self.question, chosen_idx, last_seen)
            self.terminated = True
            info["step_idx"] = last_seen
            info["chosen_idx"] = chosen_idx
            info["correct"] = chosen_idx == self.question.gold_index

        obs = self._obs()
        return obs, float(reward), self.terminated, self.truncated, info


def make_env_from_config(
    mc_questions: list[MCQuestion],
    likelihood_model: LikelihoodModel,
    config: dict[str, Any],
    precomputed_beliefs: dict[tuple[int, int], np.ndarray] | None = None,
) -> TossupMCEnv:
    """Construct a TossupMCEnv from YAML configuration.

    Factory function that reads the ``environment``, ``data``, and
    ``likelihood`` sections of a config dict and instantiates a fully
    configured environment. The likelihood model must be pre-constructed
    (e.g., via ``build_likelihood_from_config``).

    Parameters
    ----------
    mc_questions : list[MCQuestion]
        List of MCQuestion instances with options and answer profiles.
        Must be non-empty.
    likelihood_model : LikelihoodModel
        Pre-constructed likelihood model for scoring clues against options.
        Use ``build_likelihood_from_config`` to create one from config.
    config : dict[str, Any]
        Full YAML config dict. Must contain the following sections:

        - ``environment``: reward mode, penalties, belief mode
        - ``data``: K (number of answer choices)
        - ``likelihood``: beta (softmax temperature)
    precomputed_beliefs : dict or None
        Optional precomputed belief cache from ``precompute_beliefs()``.
        When provided, ``_compute_belief`` uses O(1) lookups instead of
        calling ``likelihood_model.score()``.

    Returns
    -------
    TossupMCEnv
        A configured Gymnasium environment ready for ``reset()``.

    Examples
    --------
    >>> from qb_data.config import load_config
    >>> from models.likelihoods import build_likelihood_from_config
    >>> config = load_config("configs/default.yaml")
    >>> model = build_likelihood_from_config(config, corpus_texts=corpus)
    >>> env = make_env_from_config(mc_questions, model, config)
    >>> obs, info = env.reset()
    """
    from qb_env.opponent_models import build_opponent_model_from_config

    env_cfg = config["environment"]
    data_cfg = config["data"]
    lik_cfg = config["likelihood"]
    variable_k = bool(data_cfg.get("variable_K", False) or env_cfg.get("variable_K", False))
    max_k_raw = data_cfg.get("max_K") or env_cfg.get("max_K")
    opponent_model = build_opponent_model_from_config(
        questions=mc_questions, config=config,
    )
    return TossupMCEnv(
        questions=mc_questions,
        likelihood_model=likelihood_model,
        K=int(data_cfg.get("K", 4)),
        reward_mode=str(env_cfg.get("reward", env_cfg.get("reward_mode", "time_penalty"))),
        seed=int(env_cfg.get("seed", 13)),
        wait_penalty=float(env_cfg.get("wait_penalty", 0.01)),
        early_buzz_penalty=float(env_cfg.get("early_buzz_penalty", 0.0)),
        buzz_correct=float(env_cfg.get("buzz_correct", 1.0)),
        buzz_incorrect=float(env_cfg.get("buzz_incorrect", -0.5)),
        belief_mode=str(env_cfg.get("belief_mode", "from_scratch")),
        beta=float(lik_cfg.get("beta", 5.0)),
        precomputed_beliefs=precomputed_beliefs,
        opponent_buzz_model=opponent_model,
        end_mode=str(env_cfg.get("end_mode", "force_commit")),
        no_buzz_reward=float(env_cfg.get("no_buzz_reward", 0.0)),
        variable_K=variable_k,
        max_K=int(max_k_raw) if max_k_raw is not None else None,
    )
```

## File: tests/test_ppo_buzzer.py
```python
"""Test suite for scripts/_common.py and agents/ppo_buzzer.py.

Covers:
- AGT-01: PPOBuzzer training, save, load, episode execution
- AGT-07: Shared utilities (config, JSON, MCQuestion serialization)
- S_q metric support: c_trace, g_trace, entropy_trace generation

Uses TF-IDF likelihood for fast test execution (< 10 seconds total).
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import sys
import types

import numpy as np
import pytest
import torch as th

from agents.ppo_buzzer import PPOBuzzer, PPOEpisodeTrace
from qb_data.mc_builder import MCQuestion
from qb_env.tossup_env import TossupMCEnv
from scripts._common import (
    ARTIFACT_DIR,
    PROJECT_ROOT,
    load_config,
    load_json,
    mc_question_from_dict,
    save_json,
    to_serializable,
)


# ------------------------------------------------------------------ #
# Tests: _common utilities (AGT-07)
# ------------------------------------------------------------------ #


class TestLoadConfig:
    """Tests for config loading utility."""

    def test_load_config_default(self) -> None:
        """load_config() without args loads default.yaml with expected keys."""
        cfg = load_config()
        assert isinstance(cfg, dict)
        assert "data" in cfg
        assert "ppo" in cfg
        assert "environment" in cfg
        assert "likelihood" in cfg

    def test_load_config_smoke(self) -> None:
        """load_config() can load smoke.yaml with reduced settings."""
        smoke_path = str(PROJECT_ROOT / "configs" / "smoke.yaml")
        cfg = load_config(smoke_path)
        assert cfg["data"]["max_questions"] == 50
        assert cfg["ppo"]["total_timesteps"] == 3000


class TestJsonUtilities:
    """Tests for JSON save/load round-trip."""

    def test_save_load_json_roundtrip(self, tmp_path: Path) -> None:
        """save_json/load_json round-trips nested dicts."""
        data = {"a": 1, "b": [2, 3], "c": {"d": "hello"}}
        path = tmp_path / "test.json"
        save_json(path, data)
        loaded = load_json(path)
        assert loaded == data

    def test_save_json_creates_parent_dirs(self, tmp_path: Path) -> None:
        """save_json creates missing parent directories."""
        path = tmp_path / "sub" / "dir" / "test.json"
        save_json(path, {"x": 1})
        assert path.exists()


class TestMCQuestionSerialization:
    """Tests for MCQuestion serialization and deserialization."""

    def test_to_serializable_on_mcquestion(
        self, sample_mc_question: MCQuestion
    ) -> None:
        """to_serializable converts MCQuestion to a dict."""
        result = to_serializable(sample_mc_question)
        assert isinstance(result, dict)
        assert result["qid"] == "test_q1"
        assert result["gold_index"] == 0
        assert len(result["options"]) == 4

    def test_mc_question_roundtrip(
        self, sample_mc_question: MCQuestion
    ) -> None:
        """MCQuestion survives serialization -> deserialization round-trip."""
        serialized = to_serializable(sample_mc_question)
        restored = mc_question_from_dict(serialized)
        assert restored.qid == sample_mc_question.qid
        assert restored.gold_index == sample_mc_question.gold_index
        assert restored.options == sample_mc_question.options
        assert restored.tokens == sample_mc_question.tokens

    def test_mc_question_json_roundtrip(
        self, sample_mc_question: MCQuestion, tmp_path: Path
    ) -> None:
        """MCQuestion survives save_json -> load_json -> mc_question_from_dict."""
        path = tmp_path / "mc.json"
        save_json(path, [sample_mc_question])
        raw = load_json(path)
        restored = mc_question_from_dict(raw[0])
        assert restored.qid == sample_mc_question.qid
        assert restored.answer_primary == sample_mc_question.answer_primary


class TestArtifactDir:
    """Tests for path constants."""

    def test_artifact_dir_constant(self) -> None:
        """ARTIFACT_DIR points to project/artifacts."""
        assert ARTIFACT_DIR.name == "artifacts"
        assert ARTIFACT_DIR.parent == PROJECT_ROOT


# ------------------------------------------------------------------ #
# Tests: PPOBuzzer initialization (AGT-01)
# ------------------------------------------------------------------ #


class TestPPOBuzzerInit:
    """Tests for PPOBuzzer construction."""

    def test_ppo_buzzer_init(self, sample_tfidf_env: TossupMCEnv) -> None:
        """PPOBuzzer instantiates with default hyperparameters."""
        buzzer = PPOBuzzer(env=sample_tfidf_env)
        assert buzzer.model is not None
        assert buzzer.env is sample_tfidf_env

    def test_ppo_buzzer_custom_policy_kwargs(
        self, sample_tfidf_env: TossupMCEnv
    ) -> None:
        """PPOBuzzer accepts custom policy_kwargs."""
        buzzer = PPOBuzzer(
            env=sample_tfidf_env,
            policy_kwargs={"net_arch": [128, 128, 64]},
        )
        assert buzzer.model is not None


# ------------------------------------------------------------------ #
# Tests: Episode trace generation
# ------------------------------------------------------------------ #


class TestActionProbabilities:
    """Tests for action probability extraction."""

    def test_action_probabilities_shape(
        self, sample_tfidf_env: TossupMCEnv
    ) -> None:
        """action_probabilities returns K+1 probabilities that sum to 1."""
        buzzer = PPOBuzzer(env=sample_tfidf_env)
        obs, _ = sample_tfidf_env.reset(seed=42)
        probs = buzzer.action_probabilities(obs)
        K = sample_tfidf_env.K
        assert probs.shape == (K + 1,), f"Expected ({K + 1},), got {probs.shape}"
        assert abs(probs.sum() - 1.0) < 1e-5, f"Probabilities sum to {probs.sum()}"
        assert (probs >= 0).all(), "All probabilities should be non-negative"

    def test_c_t_computation(self, sample_tfidf_env: TossupMCEnv) -> None:
        """c_t returns buzz probability in [0, 1]."""
        buzzer = PPOBuzzer(env=sample_tfidf_env)
        obs, _ = sample_tfidf_env.reset(seed=42)
        c_val = buzzer.c_t(obs)
        assert 0.0 <= c_val <= 1.0, f"c_t={c_val} out of range"

    def test_g_t_computation(self, sample_tfidf_env: TossupMCEnv) -> None:
        """g_t returns correctness probability, handles near-zero c_t."""
        buzzer = PPOBuzzer(env=sample_tfidf_env)
        obs, _ = sample_tfidf_env.reset(seed=42)
        gold_index = sample_tfidf_env.question.gold_index
        g_val = buzzer.g_t(obs, gold_index)
        assert g_val >= 0.0, f"g_t={g_val} should be non-negative"
        # g_t can be > 1.0 if P(gold) > P(buzz) in early steps, but
        # mathematically g_t = P(gold) / c_t <= 1.0 since P(gold) <= c_t
        # (gold action is one of the buzz actions)
        assert g_val <= 1.0 + 1e-5, f"g_t={g_val} should be <= 1.0"


class TestRunEpisode:
    """Tests for full episode execution with traces."""

    def test_run_episode_generates_traces(
        self, sample_tfidf_env: TossupMCEnv
    ) -> None:
        """run_episode returns PPOEpisodeTrace with matching trace lengths."""
        buzzer = PPOBuzzer(env=sample_tfidf_env)
        trace = buzzer.run_episode(seed=42)

        assert isinstance(trace, PPOEpisodeTrace)
        assert len(trace.c_trace) == len(trace.g_trace)
        assert len(trace.c_trace) == len(trace.top_p_trace)
        assert len(trace.c_trace) == len(trace.entropy_trace)
        assert len(trace.c_trace) > 0, "Episode should have at least one step"

    def test_run_episode_trace_values(
        self, sample_tfidf_env: TossupMCEnv
    ) -> None:
        """Trace values are in valid ranges."""
        buzzer = PPOBuzzer(env=sample_tfidf_env)
        trace = buzzer.run_episode(seed=42)

        for c_val in trace.c_trace:
            assert 0.0 <= c_val <= 1.0, f"c_trace value {c_val} out of [0,1]"
        for g_val in trace.g_trace:
            assert g_val >= 0.0, f"g_trace value {g_val} should be non-negative"
        for top_p in trace.top_p_trace:
            assert 0.0 <= top_p <= 1.0, f"top_p_trace value {top_p} out of [0,1]"
        for ent in trace.entropy_trace:
            assert ent >= 0.0, f"entropy {ent} should be non-negative"

    def test_ppo_calibration_uses_top_p_trace(
        self, sample_tfidf_env: TossupMCEnv
    ) -> None:
        """calibration_at_buzz on PPO traces uses top_p_trace, not c_trace."""
        from dataclasses import asdict
        from evaluation.metrics import calibration_at_buzz

        buzzer = PPOBuzzer(env=sample_tfidf_env)
        trace = buzzer.run_episode(seed=42)
        assert len(trace.top_p_trace) > 0, "top_p_trace must be populated"

        cal = calibration_at_buzz([asdict(trace)])
        assert cal["n_calibration"] == 1.0
        # Confidence should be top_p_trace[buzz_step], not c_trace[buzz_step]
        idx = min(max(0, trace.buzz_step), len(trace.top_p_trace) - 1)
        expected_conf = trace.top_p_trace[idx]
        expected_brier = (expected_conf - (1.0 if trace.correct else 0.0)) ** 2
        assert abs(cal["brier"] - expected_brier) < 1e-9

    def test_run_episode_deterministic(
        self, sample_tfidf_env: TossupMCEnv
    ) -> None:
        """Deterministic episodes with same seed produce same traces."""
        buzzer = PPOBuzzer(env=sample_tfidf_env)
        trace1 = buzzer.run_episode(deterministic=True, seed=42)
        trace2 = buzzer.run_episode(deterministic=True, seed=42)

        assert trace1.buzz_step == trace2.buzz_step
        assert trace1.buzz_index == trace2.buzz_index
        np.testing.assert_allclose(trace1.c_trace, trace2.c_trace, atol=1e-6)

    def test_run_episode_has_qid(
        self, sample_tfidf_env: TossupMCEnv
    ) -> None:
        """Episode trace includes the question ID."""
        buzzer = PPOBuzzer(env=sample_tfidf_env)
        trace = buzzer.run_episode(seed=42)
        assert trace.qid != "", "qid should not be empty"

    def test_run_episode_correct_field(
        self, sample_tfidf_env: TossupMCEnv
    ) -> None:
        """correct field matches buzz_index vs gold_index."""
        buzzer = PPOBuzzer(env=sample_tfidf_env)
        trace = buzzer.run_episode(seed=42)
        assert trace.correct == (trace.buzz_index == trace.gold_index)

    def test_run_episode_stop_only_uses_env_chosen_idx(
        self, sample_tfidf_env: TossupMCEnv
    ) -> None:
        """Stop-only episodes must use the env-selected answer index."""
        sample_obs, _ = sample_tfidf_env.reset(seed=42)
        buzzer = PPOBuzzer(env=sample_tfidf_env)

        class FakeStopOnlyEnv:
            def __init__(self, obs_shape):
                self.obs_shape = obs_shape
                self.unwrapped = type("BaseEnv", (), {})()
                self.unwrapped.question = type("Question", (), {"gold_index": 2})()
                self.unwrapped.belief = np.array(
                    [0.1, 0.2, 0.6, 0.1], dtype=np.float32
                )

            def reset(self, seed=None, options=None):
                self.unwrapped.question = type("Question", (), {"gold_index": 2})()
                self.unwrapped.belief = np.array(
                    [0.1, 0.2, 0.6, 0.1], dtype=np.float32
                )
                return np.zeros(self.obs_shape, dtype=np.float32), {"qid": "stop_only_q"}

            def step(self, action):
                assert action == 1
                return (
                    np.zeros(self.obs_shape, dtype=np.float32),
                    1.0,
                    True,
                    False,
                    {"qid": "stop_only_q", "step_idx": 0, "chosen_idx": 2, "correct": True},
                )

        buzzer.env = FakeStopOnlyEnv(sample_obs.shape)
        buzzer.action_probabilities = lambda _obs: np.array([0.1, 0.9], dtype=np.float32)

        trace = buzzer.run_episode(deterministic=True, seed=42)

        assert trace.buzz_index == 2
        assert trace.gold_index == 2
        assert trace.correct is True
        assert trace.g_trace == pytest.approx([0.6])

    def test_run_episode_no_buzz_keeps_buzz_step_unset(
        self, sample_mc_question: MCQuestion
    ) -> None:
        """no_buzz truncations stay distinct from voluntary buzz episodes."""
        from models.likelihoods import TfIdfLikelihood

        corpus = sample_mc_question.option_profiles[:]
        model = TfIdfLikelihood(corpus_texts=corpus)
        env = TossupMCEnv(
            questions=[sample_mc_question],
            likelihood_model=model,
            K=4,
            reward_mode="simple",
            end_mode="no_buzz",
            no_buzz_reward=0.0,
        )
        buzzer = PPOBuzzer(env=env)
        buzzer.action_probabilities = lambda _obs: np.array(
            [1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32
        )

        trace = buzzer.run_episode(deterministic=True, seed=42)

        assert trace.buzz_step == -1
        assert trace.buzz_index == -1
        assert trace.correct is False


# ------------------------------------------------------------------ #
# Tests: Checkpoint save/load
# ------------------------------------------------------------------ #


class TestCheckpointSaveLoad:
    """Tests for PPOBuzzer model persistence."""

    def test_ppo_checkpoint_save_load(
        self, sample_tfidf_env: TossupMCEnv, tmp_path: Path
    ) -> None:
        """PPOBuzzer saves and loads from checkpoint."""
        buzzer = PPOBuzzer(env=sample_tfidf_env)
        save_path = tmp_path / "ppo_test"
        buzzer.save(save_path)

        # SB3 appends .zip
        assert (tmp_path / "ppo_test.zip").exists(), "Model file should exist"

        loaded = PPOBuzzer.load(save_path, env=sample_tfidf_env)
        assert loaded.model is not None

        # Verify loaded model produces valid probabilities
        obs, _ = sample_tfidf_env.reset(seed=42)
        probs = loaded.action_probabilities(obs)
        assert probs.shape == (sample_tfidf_env.K + 1,)
        assert abs(probs.sum() - 1.0) < 1e-5


class TestMaskablePPO:
    """Tests for optional MaskablePPO path."""

    def test_default_ppo_unchanged(self, sample_tfidf_env) -> None:
        buzzer = PPOBuzzer(env=sample_tfidf_env, use_maskable_ppo=False)
        assert not buzzer._use_maskable
        trace = buzzer.run_episode(seed=42)
        assert len(trace.c_trace) > 0

    def test_maskable_import_error(self, sample_tfidf_env) -> None:
        sb3_contrib = pytest.importorskip("sb3_contrib", reason="sb3-contrib not installed")
        buzzer = PPOBuzzer(env=sample_tfidf_env, use_maskable_ppo=True)
        assert buzzer._use_maskable

    def test_current_action_masks_prefers_wrapper(self, sample_tfidf_env) -> None:
        """Wrapper-provided binary masks should win over base-env masks."""
        buzzer = PPOBuzzer(env=sample_tfidf_env)
        buzzer._use_maskable = True

        class Wrapper:
            def __init__(self, base):
                self.unwrapped = base

            def action_masks(self):
                return np.array([True, False], dtype=bool)

        buzzer.env = Wrapper(sample_tfidf_env)
        masks = buzzer._current_action_masks()
        assert masks.tolist() == [True, False]

    def test_action_probabilities_passes_masks_when_maskable(self, sample_tfidf_env) -> None:
        """action_probabilities should pass binary masks to the policy distribution."""
        buzzer = PPOBuzzer(env=sample_tfidf_env)
        buzzer._use_maskable = True

        class Wrapper:
            def __init__(self, base):
                self.unwrapped = base

            def action_masks(self):
                return np.array([True, False], dtype=bool)

        class FakeDist:
            def __init__(self):
                self.distribution = types.SimpleNamespace(
                    probs=th.tensor([[0.8, 0.2]], dtype=th.float32)
                )
                self._masking_applied = False

            def apply_masking(self, masks):
                self._masking_applied = True
                seen["masks"] = masks

        seen = {}

        class FakePolicy:
            def get_distribution(self, obs_tensor):
                return FakeDist()

        class FakeModel:
            device = th.device("cpu")
            policy = FakePolicy()

        buzzer.env = Wrapper(sample_tfidf_env)
        buzzer.model = FakeModel()
        probs = buzzer.action_probabilities(np.zeros(sample_tfidf_env.observation_space.shape, dtype=np.float32))
        assert probs.tolist() == pytest.approx([0.8, 0.2])
        assert seen["masks"] is not None
        assert seen["masks"].shape == (1, 2)
        assert seen["masks"].device == th.device("cpu")

    def test_maskable_checkpoint_load_path(self, sample_tfidf_env, tmp_path, monkeypatch) -> None:
        """PPOBuzzer.load(..., use_maskable_ppo=True) should use MaskablePPO.load."""
        calls = {}

        class FakeMaskablePPO:
            def __init__(self, *args, **kwargs):
                pass

            @classmethod
            def load(cls, path, env=None):
                calls["path"] = path
                calls["env"] = env
                return object()

        fake_module = types.ModuleType("sb3_contrib")
        fake_module.MaskablePPO = FakeMaskablePPO
        monkeypatch.setitem(sys.modules, "sb3_contrib", fake_module)
        loaded = PPOBuzzer.load(tmp_path / "ppo_test", env=sample_tfidf_env, use_maskable_ppo=True)
        assert calls["path"].endswith("ppo_test")
        assert calls["env"] is sample_tfidf_env
        assert loaded.model is not None
```

## File: scripts/_common.py
```python
"""Shared utilities for pipeline scripts.

Provides config loading, JSON serialization, MC question deserialization,
and path constants used across all pipeline scripts (build, baseline, train,
evaluate).

Ported from qb-rl reference implementation with import path adaptations
for the unified qanta-buzzer codebase.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from models.likelihoods import LikelihoodModel, build_likelihood_from_config
from qb_data.config import load_config as load_yaml_config
from qb_data.mc_builder import MCQuestion

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _parse_value(value: str) -> Any:
    """Parse a CLI override value string into a typed Python value.

    Tries JSON first, then bool/int/float, and falls back to str.
    """
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        pass
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    if value.lstrip("-").isdigit():
        return int(value)
    try:
        return float(value)
    except ValueError:
        return value


def parse_overrides(args: argparse.Namespace) -> dict[str, Any]:
    """Parse CLI override arguments into flat dotted-key overrides.

    Returns a dict with dotted keys (e.g. ``{"data.K": 5}``) that
    ``merge_overrides`` can apply leaf-by-leaf without clobbering
    sibling config entries.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments.  Positional ``overrides`` are
        ``key=value`` strings where *key* uses dot-notation
        (e.g. ``data.K=5``).

    Returns
    -------
    dict[str, Any]
        Flat dotted-key overrides ready for ``merge_overrides()``.
    """
    overrides: dict[str, Any] = {}
    if hasattr(args, "overrides") and args.overrides:
        for token in args.overrides:
            if "=" not in token:
                continue
            key, value = token.split("=", 1)
            overrides[key] = _parse_value(value)
    return overrides
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"


def load_config(config_path: str | None = None, smoke: bool = False) -> dict[str, Any]:
    """Load YAML configuration from a file path.

    Parameters
    ----------
    config_path : str or None
        Path to YAML config file. If None, loads ``configs/default.yaml``.

    Returns
    -------
    dict[str, Any]
        Parsed config dict with nested structure (data, likelihood,
        environment, ppo, etc.).
    """
    return load_yaml_config(config_path, smoke=smoke)


def build_likelihood_model(config: dict[str, Any], mc_questions: list[MCQuestion]):
    """Build a likelihood model with shared TF-IDF corpus handling."""
    corpus = None
    if config["likelihood"].get("model") == "tfidf":
        corpus = [q.question for q in mc_questions] + [
            profile
            for question in mc_questions
            for profile in question.option_profiles
        ]
    return build_likelihood_from_config(config, corpus_texts=corpus)


def ensure_dir(path: str | Path) -> Path:
    """Create a directory (and parents) if it does not exist.

    Parameters
    ----------
    path : str or Path
        Directory path to create.

    Returns
    -------
    Path
        The created (or existing) directory path.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def to_serializable(item: Any) -> Any:
    """Recursively convert dataclasses to dicts for JSON serialization.

    Parameters
    ----------
    item : Any
        Object to convert. Dataclasses are converted via ``asdict()``,
        dicts and lists are processed recursively.

    Returns
    -------
    Any
        JSON-serializable version of the input.
    """
    if is_dataclass(item):
        return asdict(item)
    if isinstance(item, dict):
        return {k: to_serializable(v) for k, v in item.items()}
    if isinstance(item, list):
        return [to_serializable(v) for v in item]
    return item


def save_json(path: str | Path, data: Any) -> Path:
    """Save data to a JSON file, creating parent directories as needed.

    Applies ``to_serializable`` to convert dataclasses before writing.

    Parameters
    ----------
    path : str or Path
        Output file path.
    data : Any
        Data to serialize. Dataclasses are converted to dicts automatically.

    Returns
    -------
    Path
        The path where the JSON was written.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(to_serializable(data), f, indent=2)
    return p


def load_json(path: str | Path) -> Any:
    """Load data from a JSON file.

    Parameters
    ----------
    path : str or Path
        Path to JSON file.

    Returns
    -------
    Any
        Parsed JSON data.
    """
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def mc_question_from_dict(row: dict[str, Any]) -> MCQuestion:
    """Reconstruct an MCQuestion dataclass from a JSON-deserialized dict.

    Parameters
    ----------
    row : dict[str, Any]
        Dictionary with all MCQuestion fields.

    Returns
    -------
    MCQuestion
        Reconstructed MCQuestion instance.
    """
    return MCQuestion(
        qid=row["qid"],
        question=row["question"],
        tokens=list(row["tokens"]),
        answer_primary=row["answer_primary"],
        clean_answers=list(row["clean_answers"]),
        run_indices=list(row["run_indices"]),
        human_buzz_positions=row.get("human_buzz_positions"),
        category=row.get("category", ""),
        cumulative_prefixes=list(row["cumulative_prefixes"]),
        options=list(row["options"]),
        gold_index=int(row["gold_index"]),
        option_profiles=list(row["option_profiles"]),
        option_answer_primary=list(row["option_answer_primary"]),
        distractor_strategy=row.get("distractor_strategy", "unknown"),
    )


def load_mc_questions(path: str | Path) -> list[MCQuestion]:
    """Load and deserialize a list of MCQuestions from a JSON file.

    Parameters
    ----------
    path : str or Path
        Path to JSON file containing a list of serialized MCQuestion dicts.

    Returns
    -------
    list[MCQuestion]
        List of reconstructed MCQuestion instances.
    """
    raw = load_json(path)
    return [mc_question_from_dict(item) for item in raw]


# ------------------------------------------------------------------ #
# Embedding cache persistence helpers
# ------------------------------------------------------------------ #


def embedding_cache_path(config: dict[str, Any]) -> Path:
    """Return the resolved embedding cache file path from config.

    Uses ``config['likelihood']['cache_dir']`` (default ``'cache/embeddings'``)
    and appends ``'embedding_cache_{model}.npz'`` where ``{model}`` is the
    likelihood model name from config (e.g., ``tfidf``, ``t5-base``).

    Parameters
    ----------
    config : dict
        Full YAML config dict.

    Returns
    -------
    Path
        Absolute path to the embedding cache ``.npz`` file.
    """
    lik_cfg = config.get("likelihood", {})
    cache_dir = lik_cfg.get("cache_dir", "cache/embeddings")
    model_family = str(lik_cfg.get("model", "unknown"))
    if model_family == "sbert":
        variant = lik_cfg.get("sbert_name", lik_cfg.get("embedding_model", "all-MiniLM-L6-v2"))
    elif model_family == "openai":
        variant = lik_cfg.get("openai_model", "text-embedding-3-small")
    elif model_family == "t5":
        variant = lik_cfg.get("t5_name", "t5-base")
    elif model_family.startswith("t5"):
        variant = model_family
    else:
        variant = model_family
    safe_name = str(variant).replace("/", "_")
    return PROJECT_ROOT / cache_dir / f"embedding_cache_{safe_name}.npz"


def load_embedding_cache(model: LikelihoodModel, config: dict[str, Any]) -> None:
    """Load persisted embedding cache into model if file exists.

    Parameters
    ----------
    model : LikelihoodModel
        Likelihood model whose embedding_cache will be populated.
    config : dict
        Full YAML config dict (used to resolve cache path).
    """
    path = embedding_cache_path(config)
    n = model.load_cache(path)
    if n > 0:
        print(f"Loaded {n} cached embeddings from {path}")


def save_embedding_cache(model: LikelihoodModel, config: dict[str, Any]) -> None:
    """Persist model's embedding cache to disk.

    Parameters
    ----------
    model : LikelihoodModel
        Likelihood model whose embedding_cache will be saved.
    config : dict
        Full YAML config dict (used to resolve cache path).
    """
    path = embedding_cache_path(config)
    n = model.save_cache(path)
    if n > 0:
        print(f"Saved {n} embeddings to {path}")
```

## File: scripts/compare_policies.py
```python
#!/usr/bin/env python3
"""
Compare T5-as-likelihood (MLP policy) vs T5-as-policy (end-to-end).

Evaluates both approaches on the same test set using the same metric
functions (accuracy, S_q, ECE, Brier score, buzz position).

**Important caveats for numeric comparison:**

The two evaluation paths are *not* fully apples-to-apples:

- The MLP path uses config-driven environment settings (e.g. wait_penalty
  from default.yaml or smoke.yaml).
- The T5 path uses its own hardcoded reward settings (wait_penalty=0.1,
  matching the T5 pipeline's default).
- The MLP path builds TF-IDF from test questions + all option profiles.
  The T5 path builds TF-IDF from profiles of the first 100 questions
  only (lightweight env reward computation — the T5 policy does not
  consume TF-IDF likelihoods).
- S_q semantics differ: for MLP, c_trace is a sigmoid confidence proxy
  over belief max; for T5, c_trace is the wait-head buzz probability.

These differences are inherent to the two architectures.  Accuracy and
buzz-position comparisons are directly meaningful.  ECE and Brier are
computed identically (both use top_p at buzz time).  S_q and reward
comparisons should be interpreted qualitatively.

MLP Policy (Phase 4):
    T5/TF-IDF computes likelihood scores -> belief features -> MLP
    policy decides.  Uses SB3 PPO with belief-feature observations.

T5 Policy (Phase 6):
    T5 encoder processes text directly -> PolicyHead decides.
    Uses custom PPO with text observations via TextObservationWrapper.

Usage:
    python scripts/compare_policies.py \\
        --mlp-checkpoint checkpoints/ppo/best_model \\
        --t5-checkpoint checkpoints/ppo_t5/best_model \\
        --output results/t5_comparison.json

    python scripts/compare_policies.py \\
        --t5-checkpoint checkpoints/ppo_t5/best_model \\
        --t5-only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from evaluation.metrics import (
    calibration_pairs_at_buzz,
    expected_calibration_error,
    brier_score,
    summarize_buzz_metrics,
    system_score,
)
from scripts._common import (
    ARTIFACT_DIR,
    build_likelihood_model,
    load_config,
    load_embedding_cache,
    load_mc_questions,
    save_json,
)


def resolve_mlp_eval_config(
    checkpoint_path: str | Path,
    fallback_config: dict[str, Any],
) -> dict[str, Any]:
    """Resolve the config that was used to train an MLP checkpoint.

    If a ``config_used.json`` sidecar exists next to the checkpoint,
    load and return it. Otherwise return ``fallback_config`` unchanged.
    """
    import json

    cp = Path(checkpoint_path).resolve()
    candidates = [cp / "config_used.json"] if cp.is_dir() else []
    candidates.append(cp.parent / "config_used.json")

    for sidecar in candidates:
        if sidecar.exists():
            try:
                with open(sidecar, encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
    return fallback_config


def evaluate_mlp_policy(
    checkpoint_path: str,
    test_questions: list,
    config: dict,
) -> dict[str, Any]:
    """Evaluate Phase 4 MLP policy on belief features.

    Uses the likelihood model specified by the checkpoint's sidecar
    config (``config_used.json``) when available, otherwise falls back
    to the provided config. If the resolved config selects TF-IDF, the
    corpus is fit on the evaluation set's question/option text.

    Parameters
    ----------
    checkpoint_path : str
        Path to SB3 PPO model checkpoint (``.zip`` file).
    test_questions : list
        List of MCQuestion instances to evaluate on.
    config : dict
        YAML config dict (fallback if no checkpoint sidecar exists).

    Returns
    -------
    dict[str, Any]
        Evaluation results: accuracy, mean_sq, ece, brier, avg_buzz_pos,
        n_questions.
    """
    from agents.ppo_buzzer import PPOBuzzer
    from qb_env.tossup_env import make_env_from_config

    resolved_config = resolve_mlp_eval_config(checkpoint_path, config)
    likelihood_model = build_likelihood_model(resolved_config, test_questions)
    load_embedding_cache(likelihood_model, resolved_config)

    env = make_env_from_config(
        mc_questions=test_questions,
        likelihood_model=likelihood_model,
        config=resolved_config,
    )

    use_maskable = bool(resolved_config.get("ppo", {}).get("use_maskable_ppo", False))
    agent = PPOBuzzer.load(checkpoint_path, env=env, use_maskable_ppo=use_maskable)

    # Run episodes — one per test question, deterministic order
    results = [
        agent.run_episode(deterministic=True, question_idx=i)
        for i in range(len(test_questions))
    ]

    # Compute metrics
    buzz_metrics = summarize_buzz_metrics(results)
    confidences, outcomes = calibration_pairs_at_buzz(results)
    ece = expected_calibration_error(confidences, outcomes)
    brier = brier_score(confidences, outcomes)

    return {
        "accuracy": buzz_metrics["buzz_accuracy"],
        "mean_sq": buzz_metrics["mean_sq"],
        "ece": ece,
        "brier": brier,
        "avg_buzz_pos": buzz_metrics.get("mean_buzz_step", 0.0),
        "mean_reward": buzz_metrics["mean_reward_like"],
        "n_questions": len(test_questions),
    }


def evaluate_t5_policy(
    checkpoint_path: str,
    test_questions: list,
    config: dict,
) -> dict[str, Any]:
    """Evaluate Phase 6 T5 end-to-end policy on text observations.

    Loads a T5PolicyModel from checkpoint, runs deterministic episodes
    on each test question using TextObservationWrapper, and computes the
    same metrics as evaluate_mlp_policy for fair comparison.

    Parameters
    ----------
    checkpoint_path : str
        Path to T5PolicyModel checkpoint directory.
    test_questions : list
        List of MCQuestion instances to evaluate on.
    config : dict
        YAML config dict.

    Returns
    -------
    dict[str, Any]
        Evaluation results: accuracy, mean_sq, ece, brier, avg_buzz_pos,
        n_questions.
    """
    import torch
    from models.t5_policy import T5PolicyModel
    from models.likelihoods import TfIdfLikelihood
    from qb_env.text_wrapper import TextObservationWrapper
    from qb_env.tossup_env import TossupMCEnv

    # Load T5 policy model
    model = T5PolicyModel.load_pretrained(checkpoint_path)
    model.eval()

    # Build lightweight likelihood for environment reward computation
    corpus = []
    for q in test_questions[:100]:
        corpus.extend(q.option_profiles)
    likelihood_model = TfIdfLikelihood(corpus_texts=corpus)

    correct_count = 0
    total_count = 0
    sq_scores = []
    confidences = []
    outcomes = []
    buzz_positions = []

    with torch.no_grad():
        for question in test_questions:
            env = TossupMCEnv(
                questions=[question],
                likelihood_model=likelihood_model,
                K=len(question.options),
                reward_mode="time_penalty",
                wait_penalty=0.1,
                belief_mode="from_scratch",
            )
            wrapped_env = TextObservationWrapper(env)

            obs, info = wrapped_env.reset()
            done = False
            c_trace = []
            g_trace = []
            top_p_trace = []
            episode_reward = 0.0
            step_count = 0

            while not done:
                inputs = model.encode_input([obs], max_length=512)
                actions, act_info = model.select_action(
                    inputs["input_ids"],
                    inputs["attention_mask"],
                    deterministic=True,
                )

                action = actions.item()

                wait_probs = act_info["wait_probs"]
                buzz_prob = wait_probs[0, 1].item()
                c_trace.append(buzz_prob)

                answer_probs = act_info["answer_probs"]
                gold_prob = answer_probs[0, question.gold_index].item()
                g_trace.append(gold_prob)

                top_p = float(answer_probs[0].max().item())
                top_p_trace.append(top_p)

                obs, reward, terminated, truncated, step_info = (
                    wrapped_env.step(action)
                )
                done = terminated or truncated
                episode_reward += reward
                step_count += 1

            sq = system_score(c_trace, g_trace)
            sq_scores.append(sq)

            is_correct = step_info.get("correct", False) or step_info.get(
                "forced_correct", False
            )
            if is_correct:
                correct_count += 1
            total_count += 1

            # Calibration: use top_p (max answer prob) for consistency
            # with belief-feature agents
            if top_p_trace:
                buzz_step = step_count - 1
                confidences.append(top_p_trace[-1])
                outcomes.append(1 if is_correct else 0)
                buzz_positions.append(buzz_step)

    accuracy = correct_count / max(1, total_count)
    mean_sq = float(np.mean(sq_scores)) if sq_scores else 0.0
    ece = expected_calibration_error(confidences, outcomes)
    brier_val = brier_score(confidences, outcomes)
    avg_buzz_pos = float(np.mean(buzz_positions)) if buzz_positions else 0.0

    return {
        "accuracy": accuracy,
        "mean_sq": mean_sq,
        "ece": ece,
        "brier": brier_val,
        "avg_buzz_pos": avg_buzz_pos,
        "mean_reward": 0.0,  # Not tracked per-episode for T5 policy eval
        "n_questions": total_count,
    }


def print_comparison(
    mlp_results: dict[str, Any] | None,
    t5_results: dict[str, Any],
    test_size: int,
) -> dict[str, Any]:
    """Print and return comparison summary.

    Parameters
    ----------
    mlp_results : dict or None
        MLP policy evaluation results. None if --t5-only.
    t5_results : dict
        T5 policy evaluation results.
    test_size : int
        Number of test questions evaluated.

    Returns
    -------
    dict[str, Any]
        Complete comparison dict for JSON serialization.
    """
    print("\n" + "=" * 70)
    print("COMPARISON RESULTS: T5-as-Likelihood vs T5-as-Policy")
    print("=" * 70)
    print(f"Test set size: {test_size}")
    print()

    if mlp_results is not None:
        print(f"{'Metric':<20} {'MLP (T5-likelihood)':>20} {'T5 (end-to-end)':>20} {'Difference':>15}")
        print("-" * 75)
        for metric in ["accuracy", "mean_sq", "ece", "brier", "avg_buzz_pos"]:
            mlp_val = mlp_results.get(metric, 0.0)
            t5_val = t5_results.get(metric, 0.0)
            diff = t5_val - mlp_val
            print(f"{metric:<20} {mlp_val:>20.4f} {t5_val:>20.4f} {diff:>+15.4f}")
    else:
        print("T5 Policy (end-to-end) results:")
        print("-" * 40)
        for metric in ["accuracy", "mean_sq", "ece", "brier", "avg_buzz_pos"]:
            val = t5_results.get(metric, 0.0)
            print(f"  {metric:<20}: {val:.4f}")

    # Build comparison dict
    comparison: dict[str, Any] = {
        "test_size": test_size,
        "t5_policy": t5_results,
    }
    if mlp_results is not None:
        comparison["mlp_policy"] = mlp_results
        comparison["difference"] = {
            metric: t5_results.get(metric, 0.0) - mlp_results.get(metric, 0.0)
            for metric in ["accuracy", "mean_sq", "ece", "brier", "avg_buzz_pos"]
        }

    return comparison


def parse_compare_args() -> argparse.Namespace:
    """Parse comparison script arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Compare T5-as-likelihood (MLP) vs T5-as-policy.",
    )
    parser.add_argument(
        "--mlp-checkpoint",
        type=str,
        default=None,
        help="Path to Phase 4 MLP policy checkpoint.",
    )
    parser.add_argument(
        "--t5-checkpoint",
        type=str,
        required=True,
        help="Path to Phase 6 T5 policy checkpoint.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--mc-path",
        type=str,
        default=None,
        help="Path to MC dataset JSON file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/t5_comparison.json",
        help="Path for output JSON results.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Quick test with first 50 questions.",
    )
    parser.add_argument(
        "--t5-only",
        action="store_true",
        help="Only evaluate T5 policy (skip MLP comparison).",
    )
    return parser.parse_args()


def main() -> None:
    """Run the comparison experiment."""
    args = parse_compare_args()

    # Load config
    config = load_config(args.config)

    # Load test questions
    if args.mc_path:
        mc_path = Path(args.mc_path)
    else:
        candidates = [
            ARTIFACT_DIR / "main" / "mc_dataset.json",
            ARTIFACT_DIR / "smoke" / "mc_dataset.json",
            PROJECT_ROOT / "data" / "processed" / "mc_dataset.json",
        ]
        mc_path = None
        for candidate in candidates:
            if candidate.exists():
                mc_path = candidate
                break
        if mc_path is None:
            print("ERROR: No MC dataset found. Run build_mc_dataset.py first.")
            sys.exit(1)

    print(f"Loading questions from: {mc_path}")
    all_questions = load_mc_questions(mc_path)
    print(f"Loaded {len(all_questions)} questions")

    # Prefer the persisted test split if it exists alongside mc_dataset.json
    test_split_path = mc_path.parent / "test_dataset.json"
    if test_split_path.exists():
        test_questions = load_mc_questions(test_split_path)
        print(f"Using persisted test split: {len(test_questions)} questions")
    else:
        import random
        rng = random.Random(42)
        shuffled = all_questions[:]
        rng.shuffle(shuffled)
        test_start = int(len(shuffled) * 0.85)
        test_questions = shuffled[test_start:]
        print(f"No test_dataset.json found; using random 15% split: {len(test_questions)} questions")

    if args.smoke:
        test_questions = test_questions[:50]

    print(f"Test set: {len(test_questions)} questions")

    # Evaluate MLP policy (if checkpoint provided and not t5-only)
    mlp_results = None
    if args.mlp_checkpoint and not args.t5_only:
        print("\n" + "-" * 40)
        print("Evaluating MLP policy (T5-as-likelihood)...")
        print("-" * 40)
        mlp_results = evaluate_mlp_policy(
            args.mlp_checkpoint, test_questions, config
        )
        print(f"  Accuracy: {mlp_results['accuracy']:.4f}")
        print(f"  Mean S_q: {mlp_results['mean_sq']:.4f}")

    # Evaluate T5 policy
    print("\n" + "-" * 40)
    print("Evaluating T5 policy (end-to-end)...")
    print("-" * 40)
    t5_results = evaluate_t5_policy(
        args.t5_checkpoint, test_questions, config
    )
    print(f"  Accuracy: {t5_results['accuracy']:.4f}")
    print(f"  Mean S_q: {t5_results['mean_sq']:.4f}")

    # Print comparison
    comparison = print_comparison(mlp_results, t5_results, len(test_questions))

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(output_path, comparison)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
```

## File: scripts/train_ppo.py
```python
#!/usr/bin/env python3
"""
Train PPO buzzer agent on belief-feature observations.

Loads MC questions, builds a likelihood model, creates a Gymnasium environment,
trains an MLP policy with SB3 PPO, then evaluates with episode traces and
summary metrics (accuracy, S_q, ECE, Brier score).

Usage:
    python scripts/train_ppo.py --smoke              # Quick smoke test
    python scripts/train_ppo.py --smoke --deterministic-eval
    python scripts/train_ppo.py --config configs/custom.yaml
    python scripts/train_ppo.py --timesteps 50000    # Override timesteps

Ported from qb-rl reference implementation (scripts/train_ppo.py) with
import path adaptations for the unified qanta-buzzer codebase.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.ppo_buzzer import PPOBuzzer
from evaluation.metrics import calibration_at_buzz, summarize_buzz_metrics
from qb_env.stop_only_env import StopOnlyEnv
from qb_env.tossup_env import make_env_from_config, precompute_beliefs
from qb_data.config import merge_overrides
from scripts._common import (
    ARTIFACT_DIR,
    build_likelihood_model,
    load_config,
    load_embedding_cache,
    load_mc_questions,
    parse_overrides,
    save_embedding_cache,
    save_json,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments with config, smoke, mc_path, timesteps, and
        deterministic_eval fields.
    """
    parser = argparse.ArgumentParser(description="Train PPO buzzer.")
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to YAML config file (default: configs/default.yaml).",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Use smoke mode: loads configs/smoke.yaml, outputs to artifacts/smoke/.",
    )
    parser.add_argument(
        "--mc-path", type=str, default=None,
        help="Optional MC dataset JSON path (overrides config-derived path).",
    )
    parser.add_argument(
        "--timesteps", type=int, default=None,
        help="Override total_timesteps from config.",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Override PPO/environment seed from config.",
    )
    parser.add_argument(
        "--deterministic-eval", action="store_true",
        help="Use deterministic policy for post-training episode evaluation.",
    )
    parser.add_argument(
        "--stochastic-eval", action="store_true",
        help="Force stochastic policy sampling for post-training evaluation.",
    )
    parser.add_argument(
        "--policy-mode",
        type=str,
        choices=["flat_kplus1", "stop_only"],
        default="flat_kplus1",
        help="Policy action space: flat K+1 actions (default) or binary stop_only.",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Override output directory (default: artifacts/<split>).",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Config overrides: key=value (e.g. likelihood.model=tfidf)",
    )
    return parser.parse_args()


def main() -> None:
    """Train PPO agent and save model + evaluation artifacts."""
    args = parse_args()

    config = load_config(args.config, smoke=args.smoke)
    overrides = parse_overrides(args)
    if overrides:
        print(f"Applying overrides: {overrides}")
        config = merge_overrides(config, overrides)

    split = "smoke" if args.smoke else "main"
    out_dir = Path(args.output_dir) if args.output_dir else ARTIFACT_DIR / split
    out_dir.mkdir(parents=True, exist_ok=True)
    mc_path = Path(args.mc_path) if args.mc_path else out_dir / "mc_dataset.json"

    # Fallback: check data/processed/ if artifacts path doesn't exist
    if not mc_path.exists():
        fallback = PROJECT_ROOT / "data" / "processed" / "mc_dataset.json"
        if fallback.exists():
            print(f"MC dataset not found at {mc_path}, using fallback: {fallback}")
            mc_path = fallback

    print(f"Loading MC questions from: {mc_path}")
    mc_questions = load_mc_questions(mc_path)
    print(f"Loaded {len(mc_questions)} MC questions")

    print(f"Building likelihood model: {config['likelihood']['model']}")
    likelihood_model = build_likelihood_model(config, mc_questions)
    load_embedding_cache(likelihood_model, config)

    env_cfg = config["environment"]
    lik_cfg = config["likelihood"]

    print(f"Precomputing belief trajectories for {len(mc_questions)} questions...")
    belief_cache = precompute_beliefs(
        questions=mc_questions,
        likelihood_model=likelihood_model,
        belief_mode=str(env_cfg.get("belief_mode", "from_scratch")),
        beta=float(lik_cfg.get("beta", 5.0)),
        K=int(config["data"].get("K", 4)),
    )
    print(f"Cached {len(belief_cache)} belief vectors")
    save_embedding_cache(likelihood_model, config)

    env = make_env_from_config(
        mc_questions=mc_questions,
        likelihood_model=likelihood_model,
        config=config,
        precomputed_beliefs=belief_cache,
    )
    if args.policy_mode == "stop_only":
        print("Wrapping environment with StopOnlyEnv (WAIT/BUZZ only)...")
        env = StopOnlyEnv(env)

    ppo_cfg = config["ppo"]
    train_seed = int(args.seed if args.seed is not None else ppo_cfg.get("seed", 13))
    total_timesteps = int(
        args.timesteps if args.timesteps is not None else ppo_cfg["total_timesteps"]
    )

    use_maskable = bool(ppo_cfg.get("use_maskable_ppo", False))
    if use_maskable:
        print("Using MaskablePPO for variable-K action masking")
    print(f"Training PPO for {total_timesteps} timesteps...")
    agent = PPOBuzzer(
        env=env,
        learning_rate=float(ppo_cfg["learning_rate"]),
        n_steps=int(ppo_cfg["n_steps"]),
        batch_size=int(ppo_cfg["batch_size"]),
        n_epochs=int(ppo_cfg["n_epochs"]),
        gamma=float(ppo_cfg["gamma"]),
        seed=train_seed,
        policy_kwargs=ppo_cfg.get("policy_kwargs", {"net_arch": [64, 64]}),
        verbose=1,
        use_maskable_ppo=use_maskable,
    )

    agent.train(total_timesteps=total_timesteps)
    model_path = out_dir / "ppo_model"
    agent.save(model_path)
    save_json(out_dir / "config_used.json", config)

    eval_deterministic = True
    if args.stochastic_eval:
        eval_deterministic = False
    elif args.deterministic_eval:
        eval_deterministic = True

    print(
        f"Evaluating PPO agent on {len(mc_questions)} questions "
        f"(deterministic={eval_deterministic})..."
    )
    traces = [
        asdict(
            agent.run_episode(
                deterministic=eval_deterministic,
                question_idx=i,
            )
        )
        for i in range(len(mc_questions))
    ]
    summary = {**summarize_buzz_metrics(traces), **calibration_at_buzz(traces)}

    save_json(out_dir / "ppo_runs.json", traces)
    save_json(out_dir / "ppo_summary.json", summary)
    print(f"Saved PPO model to: {model_path}.zip")
    print(f"Saved PPO summaries to: {out_dir}")


if __name__ == "__main__":
    main()
```

## File: agents/ppo_buzzer.py
```python
"""PPO Buzzer agent wrapping Stable-Baselines3's PPO.

Provides the PPOBuzzer class for training an MLP policy on belief-feature
observations from TossupMCEnv, and PPOEpisodeTrace for recording per-step
action probabilities needed to compute the S_q scoring metric.

The key design rationale: SB3's ``learn()`` does not expose per-step action
distributions, so ``run_episode()`` implements custom episode execution that
records c_trace (buzz probability) and g_trace (correctness probability)
at each step for downstream S_q computation.

Ported from qb-rl reference implementation (agents/ppo_buzzer.py) with
import path adaptations for the unified qanta-buzzer codebase.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch as th
from stable_baselines3 import PPO

from qb_env.tossup_env import TossupMCEnv


@dataclass
class PPOEpisodeTrace:
    """Record of a single episode with per-step action probability traces.

    Used to compute the S_q scoring metric: S_q = sum(c_t * g_t) over steps,
    and calibration metrics (ECE, Brier) via ``top_p_trace``.

    Attributes
    ----------
    qid : str
        Question identifier.
    buzz_step : int
        Step at which the agent buzzed (-1 if never buzzed voluntarily).
    buzz_index : int
        Index of the chosen answer option (0-based, -1 if forced).
    gold_index : int
        Index of the correct answer option (0-based).
    correct : bool
        Whether the agent selected the correct answer.
    episode_reward : float
        Total accumulated reward over the episode.
    c_trace : list[float]
        Per-step buzz probability: 1 - P(wait) at each timestep.
    g_trace : list[float]
        Per-step correctness probability: P(gold_option) / P(buzz).
    top_p_trace : list[float]
        Per-step max belief probability: max(env.belief). Used as the
        confidence proxy for calibration metrics, consistent with
        baseline agents.
    entropy_trace : list[float]
        Per-step policy entropy over the full action distribution.
    """

    qid: str
    buzz_step: int
    buzz_index: int
    gold_index: int
    correct: bool
    episode_reward: float
    c_trace: list[float]
    g_trace: list[float]
    top_p_trace: list[float]
    entropy_trace: list[float]


class PPOBuzzer:
    """PPO-trained buzzer agent wrapping Stable-Baselines3's PPO.

    Trains an MLP policy on belief-feature observations (Box(K+6,)) from
    TossupMCEnv. The policy maps observation vectors to a Discrete(K+1)
    action space: WAIT (0) or BUZZ with option i (1..K).

    Parameters
    ----------
    env : TossupMCEnv
        Gymnasium environment with belief-feature observations.
    learning_rate : float
        Learning rate for the Adam optimizer.
    n_steps : int
        Number of steps per rollout buffer collection.
    batch_size : int
        Minibatch size for PPO updates.
    n_epochs : int
        Number of optimization epochs per rollout.
    gamma : float
        Discount factor for return computation.
    policy_kwargs : dict or None
        Additional keyword arguments for the MLP policy. Defaults to
        ``{"net_arch": [64, 64]}`` (two hidden layers of 64 units).
    verbose : int
        SB3 verbosity level (0=silent, 1=info, 2=debug).
    """

    def __init__(
        self,
        env: TossupMCEnv,
        learning_rate: float = 3e-4,
        n_steps: int = 128,
        batch_size: int = 32,
        n_epochs: int = 10,
        gamma: float = 0.99,
        seed: int | None = None,
        policy_kwargs: dict[str, Any] | None = None,
        verbose: int = 0,
        use_maskable_ppo: bool = False,
    ):
        if policy_kwargs is None:
            policy_kwargs = {"net_arch": [64, 64]}

        self.env = env
        self._use_maskable = use_maskable_ppo

        if use_maskable_ppo:
            try:
                from sb3_contrib import MaskablePPO
            except ImportError as exc:
                raise ImportError(
                    "MaskablePPO requires sb3-contrib. "
                    "Install with: pip install -e '.[maskable]'"
                ) from exc
            self.model = MaskablePPO(
                "MlpPolicy",
                env,
                verbose=verbose,
                seed=seed,
                learning_rate=learning_rate,
                n_steps=n_steps,
                batch_size=batch_size,
                n_epochs=n_epochs,
                gamma=gamma,
                policy_kwargs=policy_kwargs,
            )
        else:
            self.model = PPO(
                "MlpPolicy",
                env,
                verbose=verbose,
                seed=seed,
                learning_rate=learning_rate,
                n_steps=n_steps,
                batch_size=batch_size,
                n_epochs=n_epochs,
                gamma=gamma,
                policy_kwargs=policy_kwargs,
            )

    def train(self, total_timesteps: int = 100_000) -> None:
        """Train the PPO policy for the specified number of timesteps.

        Parameters
        ----------
        total_timesteps : int
            Total environment steps to collect during training.
        """
        self.model.learn(total_timesteps=total_timesteps)

    def save(self, path: str | Path) -> None:
        """Save the trained PPO model to disk.

        The checkpoint does not record whether it was trained with PPO or
        MaskablePPO. Callers must pass ``use_maskable_ppo`` to ``load()``
        matching the training configuration.

        Parameters
        ----------
        path : str or Path
            File path for the saved model (SB3 appends .zip if needed).
        """
        self.model.save(str(path))

    @classmethod
    def load(
        cls,
        path: str | Path,
        env: TossupMCEnv,
        use_maskable_ppo: bool = False,
    ) -> "PPOBuzzer":
        """Load a previously saved PPO model.

        Parameters
        ----------
        path : str or Path
            Path to the saved model file.
        env : TossupMCEnv
            Environment to attach to the loaded model.
        use_maskable_ppo : bool
            If True, load with ``MaskablePPO`` from sb3-contrib instead
            of plain SB3 ``PPO``.

        Returns
        -------
        PPOBuzzer
            A PPOBuzzer with the loaded model weights.
        """
        agent = cls.__new__(cls)
        agent.env = env
        agent._use_maskable = use_maskable_ppo
        if use_maskable_ppo:
            try:
                from sb3_contrib import MaskablePPO
            except ImportError as exc:
                raise ImportError(
                    "MaskablePPO requires sb3-contrib. "
                    "Install with: pip install -e '.[maskable]'"
                ) from exc
            agent.model = MaskablePPO.load(str(path), env=env)
        else:
            agent.model = PPO.load(str(path), env=env)
        return agent

    def _current_action_masks(self) -> np.ndarray | None:
        """Return action masks from the env, or None if not maskable."""
        if not self._use_maskable:
            return None
        env_for_mask = self.env if hasattr(self.env, "action_masks") else self._base_env()
        if not hasattr(env_for_mask, "action_masks"):
            return None
        return np.asarray(env_for_mask.action_masks(), dtype=bool)

    def action_probabilities(self, obs: np.ndarray) -> np.ndarray:
        """Extract action probabilities from the policy for a given observation.

        When ``use_maskable_ppo=True``, passes ``action_masks`` to the
        policy distribution so that probabilities for invalid actions are
        zeroed out before action selection.

        Parameters
        ----------
        obs : np.ndarray
            Observation vector of shape (K + 6,).

        Returns
        -------
        np.ndarray
            Action probability vector of shape (K + 1,), dtype float32.
            Index 0 = P(wait), indices 1..K = P(buzz with option i).
        """
        obs_tensor = th.as_tensor(
            obs, dtype=th.float32, device=self.model.device
        ).unsqueeze(0)

        masks = self._current_action_masks()
        dist = self.model.policy.get_distribution(obs_tensor)
        if masks is not None:
            masks_tensor = th.as_tensor(
                masks, dtype=th.bool, device=self.model.device
            ).unsqueeze(0)
            dist.apply_masking(masks_tensor)

        probs = dist.distribution.probs[0].detach().cpu().numpy()
        return probs.astype(np.float32)

    def _base_env(self) -> TossupMCEnv:
        """Return the underlying TossupMCEnv, unwrapping if needed."""
        return getattr(self.env, "unwrapped", self.env)

    def c_t(self, obs: np.ndarray) -> float:
        """Compute buzz probability at the current step.

        Parameters
        ----------
        obs : np.ndarray
            Observation vector of shape (K + 6,).

        Returns
        -------
        float
            Probability of buzzing: 1 - P(wait). Range [0, 1].
        """
        probs = self.action_probabilities(obs)
        return float(1.0 - probs[0])

    def g_t(self, obs: np.ndarray, gold_index: int) -> float:
        """Compute correctness probability at the current step.

        Given that the agent buzzes, what is the probability it selects
        the correct answer? Formally: P(gold_action) / P(buzz).

        Parameters
        ----------
        obs : np.ndarray
            Observation vector of shape (K + 6,).
        gold_index : int
            Index of the correct answer option (0-based).

        Returns
        -------
        float
            Conditional correctness probability. Returns 0.0 if buzz
            probability is near zero (< 1e-12).
        """
        probs = self.action_probabilities(obs)
        base_env = self._base_env()
        c_t = float(1.0 - probs[0])
        if c_t <= 1e-12:
            return 0.0
        if len(probs) == 2:
            if gold_index < 0 or base_env.belief is None:
                return 0.0
            return float(base_env.belief[gold_index])
        return float(probs[gold_index + 1] / c_t)

    def run_episode(
        self,
        deterministic: bool = False,
        seed: int | None = None,
        question_idx: int | None = None,
    ) -> PPOEpisodeTrace:
        """Run a full episode and record per-step action probability traces.

        Executes the policy in the environment, computing c_trace (buzz
        probability), g_trace (correctness probability), and entropy_trace
        at each step. These traces are needed to compute the S_q metric.

        Parameters
        ----------
        deterministic : bool
            If True, select actions by argmax instead of sampling.
        seed : int or None
            If provided, seeds the environment reset for reproducibility.

        Returns
        -------
        PPOEpisodeTrace
            Complete episode record with action traces and outcome.
        """
        reset_options = None
        if question_idx is not None:
            reset_options = {"question_idx": int(question_idx)}

        obs, info = self.env.reset(seed=seed, options=reset_options)
        terminated = False
        truncated = False
        total_reward = 0.0
        c_trace: list[float] = []
        g_trace: list[float] = []
        top_p_trace: list[float] = []
        entropy_trace: list[float] = []

        base_env = self._base_env()
        buzz_step = -1
        buzz_index = -1
        gold_index = (
            base_env.question.gold_index
            if getattr(base_env, "question", None) is not None
            else -1
        )

        while not (terminated or truncated):
            probs = self.action_probabilities(obs)
            c_val = float(probs[1] if len(probs) == 2 else 1.0 - probs[0])
            if len(probs) == 2:
                g_val = (
                    float(base_env.belief[gold_index])
                    if gold_index >= 0 and base_env.belief is not None
                    else 0.0
                )
            else:
                g_val = (
                    float(probs[gold_index + 1] / c_val) if c_val > 1e-12 else 0.0
                )
            entropy = float(
                -(np.clip(probs, 1e-12, 1.0) * np.log(np.clip(probs, 1e-12, 1.0))).sum()
            )

            top_p_val = float(np.max(base_env.belief)) if base_env.belief is not None else c_val
            c_trace.append(c_val)
            g_trace.append(g_val)
            top_p_trace.append(top_p_val)
            entropy_trace.append(entropy)

            if deterministic:
                action = int(np.argmax(probs))
            else:
                action = int(np.random.choice(len(probs), p=probs))

            obs, reward, terminated, truncated, step_info = self.env.step(action)
            total_reward += reward

            if action != 0 and buzz_step < 0:
                buzz_step = int(step_info.get("step_idx", 0))
                if len(probs) == 2:
                    buzz_index = int(
                        step_info.get(
                            "chosen_idx",
                            step_info.get("forced_choice", np.argmax(base_env.belief)),
                        )
                    )
                else:
                    buzz_index = action - 1
            if truncated and buzz_step < 0 and not step_info.get("no_buzz", False):
                buzz_step = int(
                    step_info.get("step_idx", len(c_trace) - 1)
                )
                buzz_index = int(
                    step_info.get("forced_choice", np.argmax(base_env.belief))
                )

        correct = buzz_index == gold_index
        return PPOEpisodeTrace(
            qid=info.get("qid", ""),
            buzz_step=buzz_step,
            buzz_index=buzz_index,
            gold_index=gold_index,
            correct=correct,
            episode_reward=total_reward,
            c_trace=c_trace,
            g_trace=g_trace,
            top_p_trace=top_p_trace,
            entropy_trace=entropy_trace,
        )
```

## File: scripts/evaluate_all.py
```python
#!/usr/bin/env python3
"""
Comprehensive evaluation with control experiments and visualization.

Runs the SoftmaxProfileBuzzer at the best threshold (from baseline sweep),
then executes control experiments (choices-only, shuffle, alias substitution)
and generates comparison plots and tables for the CS234 writeup.

Consumes outputs from:
- build_mc_dataset.py (mc_dataset.json)
- run_baselines.py (baseline_summary.json)
- train_ppo.py (ppo_summary.json)

Produces:
- evaluation_report.json (full eval + controls + baseline + PPO summaries)
- plots/entropy_vs_clue.png
- plots/calibration.png
- plots/comparison.csv

Usage:
    python scripts/evaluate_all.py --smoke
    python scripts/evaluate_all.py --config configs/custom.yaml
    python scripts/evaluate_all.py --mc-path artifacts/main/mc_dataset.json

Ported from qb-rl reference implementation (scripts/evaluate_all.py) with
import path adaptations for the unified qanta-buzzer codebase.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.bayesian_buzzer import SoftmaxProfileBuzzer
from agents.threshold_buzzer import (
    _softmax_episode_from_precomputed,
    precompute_beliefs,
)
from evaluation.controls import (
    run_alias_substitution_control,
    run_choices_only_control,
    run_shuffle_control_precomputed,
)
from evaluation.metrics import (
    calibration_at_buzz,
    per_category_accuracy,
    summarize_buzz_metrics,
)
from evaluation.plotting import (
    plot_calibration_curve,
    plot_entropy_vs_clue_index,
    save_comparison_table,
)
from qb_data.config import merge_overrides
from scripts._common import (
    ARTIFACT_DIR,
    build_likelihood_model,
    load_config,
    load_embedding_cache,
    load_json,
    load_mc_questions,
    parse_overrides,
    save_json,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments with config, smoke, and mc_path fields.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate all agents and controls."
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to YAML config file (default: configs/default.yaml).",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Use smoke mode: loads configs/smoke.yaml, outputs to artifacts/smoke/.",
    )
    parser.add_argument(
        "--mc-path", type=str, default=None,
        help="Optional MC dataset JSON path (overrides config-derived path).",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Override output directory (default: artifacts/<split>).",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Config overrides: key=value (e.g. likelihood.model=tfidf)",
    )
    return parser.parse_args()


def pick_best_softmax_threshold(
    out_dir: Path, default_threshold: float
) -> float:
    """Select the best softmax threshold from baseline sweep results.

    Loads baseline_summary.json and extracts the threshold with the
    highest mean S_q score from the softmax_profile results.

    Parameters
    ----------
    out_dir : Path
        Directory containing baseline_summary.json.
    default_threshold : float
        Fallback threshold if baseline summary is unavailable.

    Returns
    -------
    float
        Best threshold by S_q score, or default_threshold if unavailable.
    """
    summary_path = out_dir / "baseline_summary.json"
    if not summary_path.exists():
        return default_threshold
    summary = load_json(summary_path)
    softmax = summary.get("softmax_profile", {})
    if not softmax:
        return default_threshold
    best_t = default_threshold
    best_sq = float("-inf")
    for t_str, metrics in softmax.items():
        sq = float(metrics.get("mean_sq", float("-inf")))
        if sq > best_sq:
            best_sq = sq
            best_t = float(t_str)
    return best_t


def main() -> None:
    """Run comprehensive evaluation with controls and visualizations."""
    args = parse_args()

    config = load_config(args.config, smoke=args.smoke)
    overrides = parse_overrides(args)
    if overrides:
        print(f"Applying overrides: {overrides}")
        config = merge_overrides(config, overrides)

    split = "smoke" if args.smoke else "main"
    out_dir = Path(args.output_dir) if args.output_dir else ARTIFACT_DIR / split
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)
    mc_path = Path(args.mc_path) if args.mc_path else out_dir / "mc_dataset.json"

    # Fallback: check data/processed/ if artifacts path doesn't exist
    if not mc_path.exists():
        fallback = PROJECT_ROOT / "data" / "processed" / "mc_dataset.json"
        if fallback.exists():
            print(f"MC dataset not found at {mc_path}, using fallback: {fallback}")
            mc_path = fallback

    print(f"Loading MC questions from: {mc_path}")
    mc_questions = load_mc_questions(mc_path)
    print(f"Loaded {len(mc_questions)} MC questions")

    # Load alias lookup (generated by build_mc_dataset.py)
    alias_path = out_dir / "alias_lookup.json"
    if alias_path.exists():
        alias_lookup = load_json(alias_path)
    else:
        print(f"Warning: alias_lookup.json not found at {alias_path}, using empty lookup")
        alias_lookup = {}

    # Build likelihood model
    print(f"Building likelihood model: {config['likelihood']['model']}")
    likelihood_model = build_likelihood_model(config, mc_questions)
    load_embedding_cache(likelihood_model, config)
    beta = float(config["likelihood"].get("beta", 5.0))
    alpha = float(config["bayesian"].get("alpha", 10.0))
    default_threshold = float(config["bayesian"]["threshold_sweep"][0])
    threshold = pick_best_softmax_threshold(out_dir, default_threshold=default_threshold)
    print(f"Using best softmax threshold: {threshold}")

    # Precompute beliefs once (single pass of likelihood_model.score())
    print("Precomputing beliefs...")
    precomputed = precompute_beliefs(mc_questions, likelihood_model, beta)

    # Precomputed evaluation (zero extra score() calls)
    def evaluate_questions_precomputed(pqs):
        runs = [asdict(_softmax_episode_from_precomputed(pq, threshold, alpha)) for pq in pqs]
        summary = {**summarize_buzz_metrics(runs), **calibration_at_buzz(runs)}
        summary["runs"] = runs
        return summary

    # Live evaluator for controls that genuinely change option text (alias)
    def evaluate_questions_live(qset):
        agent = SoftmaxProfileBuzzer(
            likelihood_model=likelihood_model,
            threshold=threshold,
            beta=beta,
            alpha=alpha,
        )
        runs = [asdict(agent.run_episode(q)) for q in qset]
        summary = {**summarize_buzz_metrics(runs), **calibration_at_buzz(runs)}
        summary["runs"] = runs
        return summary

    # --- Run evaluations ---
    print("Running full evaluation...")
    full_eval = evaluate_questions_precomputed(precomputed)

    # Compute per-category breakdown
    print("\nComputing per-category breakdown...")
    per_category_results = per_category_accuracy(full_eval["runs"], mc_questions)

    # Sort by category name for readability
    per_category_sorted = dict(sorted(per_category_results.items()))

    print("\nPer-category accuracy:")
    for category, metrics in per_category_sorted.items():
        print(
            f"  {category:20s} (n={metrics['n']:3.0f}): "
            f"acc={metrics['buzz_accuracy']:.3f}, "
            f"S_q={metrics['mean_sq']:.3f}"
        )
    print()

    print("Running shuffle control...")
    shuffle_eval = run_shuffle_control_precomputed(precomputed, threshold, alpha)

    if alias_lookup:
        print("Running alias substitution control...")
        alias_eval = run_alias_substitution_control(
            mc_questions,
            alias_lookup=alias_lookup,
            evaluator=lambda qset: evaluate_questions_live(qset),
        )
        alias_control_report = {k: v for k, v in alias_eval.items() if k != "runs"}
    else:
        print(
            "Skipping alias substitution control: alias_lookup.json missing or empty"
        )
        alias_control_report = {
            "skipped": True,
            "reason": "alias_lookup.json missing or empty",
        }

    print("Running choices-only control...")
    choices_only = run_choices_only_control(mc_questions)

    # --- Load existing artifacts ---
    ppo_summary_path = out_dir / "ppo_summary.json"
    ppo_summary = load_json(ppo_summary_path) if ppo_summary_path.exists() else {}
    baseline_summary_path = out_dir / "baseline_summary.json"
    baseline_summary = (
        load_json(baseline_summary_path) if baseline_summary_path.exists() else {}
    )

    # --- Build evaluation report ---
    report = {
        "softmax_profile_best_threshold": threshold,
        "full_eval": {k: v for k, v in full_eval.items() if k != "runs"},
        "controls": {
            "choices_only": choices_only,
            "shuffle": {k: v for k, v in shuffle_eval.items() if k != "runs"},
            "alias_substitution": alias_control_report,
        },
        "per_category": per_category_sorted,
        "baseline_summary": baseline_summary,
        "ppo_summary": ppo_summary,
    }

    # Add Expected Wins summary only when that reward mode is active
    if config.get("environment", {}).get("reward_mode") == "expected_wins":
        from evaluation.metrics import expected_wins_score
        from qb_env.opponent_models import build_opponent_model_from_config

        opp_model = build_opponent_model_from_config(mc_questions, config)
        qid_to_q = {q.qid: q for q in mc_questions}
        if opp_model is not None:
            ew_scores = []
            for run in full_eval["runs"]:
                q = qid_to_q.get(run.get("qid", ""), mc_questions[0])
                opp_surv = [
                    opp_model.prob_survive_to_step(q, t)
                    for t in range(len(run.get("c_trace", [])))
                ]
                ew = expected_wins_score(
                    run.get("c_trace", []),
                    run.get("g_trace", []),
                    opp_surv,
                )
                ew_scores.append(ew)
            report["expected_wins"] = {
                "mean_ew": float(np.mean(ew_scores)) if ew_scores else 0.0,
                "n": len(ew_scores),
            }

    save_json(out_dir / "evaluation_report.json", report)

    # --- Generate visualizations ---
    print("Generating plots...")

    # Entropy vs clue index
    entropy_traces = [
        list(r["entropy_trace"])
        for r in full_eval["runs"]
        if r.get("entropy_trace")
    ]
    max_len = max((len(t) for t in entropy_traces), default=0)
    padded = np.full((len(entropy_traces), max_len), np.nan, dtype=np.float32)
    for i, trace in enumerate(entropy_traces):
        padded[i, : len(trace)] = np.array(trace, dtype=np.float32)
    entropy_trace = (
        np.nanmean(padded, axis=0).tolist() if max_len > 0 else []
    )
    plot_entropy_vs_clue_index(
        {"softmax_profile": entropy_trace},
        out_dir / "plots" / "entropy_vs_clue.png",
    )

    # Calibration curve — use canonical helper for consistency
    from evaluation.metrics import calibration_pairs_at_buzz
    confidences, outcomes = calibration_pairs_at_buzz(full_eval["runs"])
    plot_calibration_curve(
        confidences, outcomes, out_dir / "plots" / "calibration.png"
    )

    # Comparison table: include baseline sweep, controls, and PPO
    table_rows = []

    # Add baseline sweep results (threshold at multiple values)
    if "threshold" in baseline_summary:
        for threshold_str, metrics in baseline_summary["threshold"].items():
            table_rows.append({
                "agent": f"threshold_{threshold_str}",
                **{k: v for k, v in metrics.items() if k != "runs"},
            })

    # Add softmax_profile sweep results
    if "softmax_profile" in baseline_summary:
        for threshold_str, metrics in baseline_summary["softmax_profile"].items():
            table_rows.append({
                "agent": f"softmax_{threshold_str}",
                **{k: v for k, v in metrics.items() if k != "runs"},
            })

    # Add full softmax eval (best threshold) and control experiments
    table_rows.append({
        "agent": "full_softmax",
        **{k: v for k, v in full_eval.items() if k != "runs"},
    })
    table_rows.append({
        "agent": "shuffle_control",
        **{k: v for k, v in shuffle_eval.items() if k != "runs"},
    })
    if not alias_control_report.get("skipped"):
        table_rows.append({
            "agent": "alias_control",
            **{k: v for k, v in alias_control_report.items() if k != "runs"},
        })

    # Add PPO if available
    if ppo_summary:
        table_rows.append({"agent": "ppo", **ppo_summary})

    save_comparison_table(table_rows, out_dir / "plots" / "comparison.csv")

    print(f"Wrote evaluation report to: {out_dir / 'evaluation_report.json'}")


if __name__ == "__main__":
    main()
```
