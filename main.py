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
