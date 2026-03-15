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
