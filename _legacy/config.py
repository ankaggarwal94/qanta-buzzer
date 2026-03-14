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
