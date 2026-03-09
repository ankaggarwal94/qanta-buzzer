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
