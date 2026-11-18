"""
Supervised warm-start training for T5PolicyModel.

Trains answer selection using cross-entropy loss. The default shows complete
questions. Prefix mode trains and validates once per scheduled cumulative
reveal. Overlength rows retain the complete menu and the newest visible clue
tokens. These are answer-supervision examples even when the buzzer cannot act
at a scheduled reveal.

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

    python scripts/train_t5_policy.py --supervised-only supervised.input_mode=prefixes
"""

from __future__ import annotations

import json
import random
from collections.abc import Callable
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models.t5_policy import T5PolicyModel
from qb_data.data_audit import validate_question_splits
from qb_data.mc_builder import MCQuestion, validate_reveal_grid
from training.checkpoints import exclusive_training_run, save_training_checkpoint, training_lock


AnswerExample = tuple[MCQuestion, str | None]


def _menu_text(question: MCQuestion) -> str:
    choices = " ".join(f"({i}) {option}" for i, option in enumerate(question.options, 1))
    return f"| CHOICES: {choices}"


def format_question_text(question: MCQuestion, clues: str | None = None) -> str:
    """Format visible clues and the original ordered menu for answer selection.

    Parameters
    ----------
    question : MCQuestion
        Question with tokens, options, and gold_index.
    clues : str, optional
        A scheduled clue prefix. Defaults to the complete token sequence.

    Returns
    -------
    str
        Formatted text: ``"CLUES: <visible tokens> | CHOICES: (1) opt1 (2) opt2 ..."``
    """
    if clues is None:
        clues = " ".join(question.tokens)
    return f"CLUES: {clues} {_menu_text(question)}"


class SupervisedTrainer:
    """Trainer for supervised warm-start of T5PolicyModel.

    Trains the answer head using cross-entropy loss. Uses gradient accumulation for stable training
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
        - ``supervised_input_mode`` (str): "full_question" or "prefixes".
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
        self.write_checkpoint_sidecars: Callable[[Path], None] | None = None

        self.device = model.device
        self.input_mode = config.get("supervised_input_mode", "full_question")
        if self.input_mode not in ("full_question", "prefixes"):
            raise ValueError("supervised.input_mode must be 'full_question' or 'prefixes'")
        self.shuffle = random.shuffle
        if self.input_mode == "prefixes":
            validate_question_splits(train=self.train_questions, val=self.val_questions)
            self.shuffle = random.Random(config.get("supervised_seed", 42)).shuffle
        self.train_examples = self._examples(self.train_questions)
        self.val_examples = self._examples(self.val_questions)

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
        self.best_val_acc = -float("inf")
        self.best_checkpoint_path: Path | None = None
        self.train_history: List[Dict[str, Any]] = []
        self.val_history: List[Dict[str, Any]] = []

        # Checkpoint directory
        self.checkpoint_dir = Path(config.get("checkpoint_dir", "checkpoints")) / "supervised"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _examples(self, questions: List[MCQuestion]) -> list[AnswerExample]:
        """Keep references to each unique reveal within its already-assigned source split."""
        if self.input_mode == "full_question":
            return [(question, None) for question in questions]
        examples = []
        head = getattr(self.model, "policy_head", None)
        width = getattr(head, "num_choices", self.config.get("num_choices", 4))
        for question in questions:
            count = len(question.options)
            if not 1 <= count <= width:
                raise ValueError(f"{question.qid}: menu size {count} exceeds the answer head or is empty")
            index = question.gold_index
            if isinstance(index, bool) or not isinstance(index, (int, np.integer)) or not 0 <= index < count:
                raise ValueError(f"{question.qid}: gold_index must identify a displayed menu option")
            validate_reveal_grid(question)
            # Repeated boundaries can differ in whitespace while exposing identical tokens.
            by_end = dict(zip(question.run_indices, question.cumulative_prefixes, strict=True))
            examples.extend((question, prefix) for prefix in by_end.values())
        return examples

    def prepare_batch(
        self, questions: List[MCQuestion], clues: list[str | None] | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Format a batch of answer examples as tokenized tensors.

        Tokenize either complete questions or the supplied scheduled reveals.

        Parameters
        ----------
        questions : list[MCQuestion]
            Batch of questions to format.
        clues : list[str or None], optional
            Visible clue text for each row. None selects the full question.

        Returns
        -------
        input_ids : torch.Tensor
            Token IDs of shape ``[batch_size, seq_len]``, on device.
        attention_mask : torch.Tensor
            Attention mask of shape ``[batch_size, seq_len]``, on device.
        labels : torch.Tensor
            Gold answer indices of shape ``[batch_size]``, on device.
        """
        if clues is None:
            clues = [None] * len(questions)
        texts = [format_question_text(q, p) for q, p in zip(questions, clues, strict=True)]
        labels = [q.gold_index for q in questions]

        tokenizer = self.model.tokenizer
        if self.input_mode == "prefixes":
            rows = tokenizer(texts, truncation=False, return_attention_mask=False, verbose=False)["input_ids"]
            for i, row in enumerate(rows):
                if len(row) > self.max_input_length:
                    # Preserve the new reveal and every displayed answer in a long prompt.
                    header = tokenizer("CLUES:", add_special_tokens=False)["input_ids"]
                    menu = tokenizer(_menu_text(questions[i]))["input_ids"]
                    available = self.max_input_length - len(header) - len(menu)
                    if available < 1:
                        raise ValueError(f"{questions[i].qid}: menu leaves no context for clues")
                    visible = clues[i] if clues[i] is not None else " ".join(questions[i].tokens)
                    clue_ids = tokenizer(visible, add_special_tokens=False)["input_ids"]
                    rows[i] = header + clue_ids[-available:] + menu
            inputs = tokenizer.pad({"input_ids": rows}, padding=True, return_tensors="pt")
        else:
            inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True,
                               max_length=self.max_input_length)

        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        labels_tensor = torch.tensor(labels, dtype=torch.long).to(self.device)

        return input_ids, attention_mask, labels_tensor

    def _predict_batch(self, examples: list[AnswerExample]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict active menu slots for both optimization and validation."""
        questions = [q for q, _ in examples]
        ids, attention, labels = self.prepare_batch(questions, [p for _, p in examples])
        logits, predictions = self.model.predict_answer(ids, attention)
        if self.input_mode == "prefixes":
            counts = torch.tensor([len(q.options) for q in questions], device=logits.device)
            slots = torch.arange(logits.shape[-1], device=logits.device)
            logits = logits.masked_fill(slots >= counts[:, None], -torch.inf)
            predictions = logits.argmax(dim=-1)
        return logits, predictions, labels

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
        shuffled = self.train_examples[:]
        self.shuffle(shuffled)

        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        num_batches = max(1, (len(shuffled) + self.batch_size - 1) // self.batch_size)

        # Zero gradients at start
        self.optimizer.zero_grad()

        for batch_idx in range(num_batches):
            start = batch_idx * self.batch_size
            end = min(start + self.batch_size, len(shuffled))
            batch_questions = shuffled[start:end]

            if not batch_questions:
                continue

            answer_logits, predictions, labels = self._predict_batch(batch_questions)

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
        """Compute loss and answer accuracy over the configured validation examples."""
        return self._evaluate(self.val_examples)

    def _evaluate(self, examples: list[AnswerExample]) -> Tuple[float, float]:
        """Compute mean batch loss and sample accuracy for the requested reveals."""
        self.model.eval()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        num_batches = max(1, (len(examples) + self.batch_size - 1) // self.batch_size)

        with torch.no_grad():
            for batch_idx in range(num_batches):
                start = batch_idx * self.batch_size
                end = min(start + self.batch_size, len(examples))
                batch_questions = examples[start:end]

                if not batch_questions:
                    continue

                answer_logits, predictions, labels = self._predict_batch(batch_questions)

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
        # Existing files are never evidence that this invocation saved a model.
        with training_lock(self.checkpoint_dir.parent):
            return self._train()

    def _train(self) -> Dict[str, Any]:
        self.best_checkpoint_path = None
        self.train_history.clear()
        self.val_history.clear()
        self.best_val_acc = -float("inf")
        if self.epochs <= 0:
            raise ValueError("supervised_epochs must be positive to produce a fresh checkpoint")
        if not self.val_questions:
            raise ValueError("T5 checkpoint selection requires validation questions")
        print(f"Starting supervised training for {self.epochs} epochs")
        print(f"  Training samples: {len(self.train_examples)} from {len(self.train_questions)} questions")
        print(f"  Validation samples: {len(self.val_examples)} from {len(self.val_questions)} questions")
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
            if not np.isfinite(val_loss) or not np.isfinite(val_acc):
                raise ValueError("Supervised validation loss and accuracy must be finite")

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
                self.save_checkpoint(is_best=True, validation_score=val_acc)
                print(f"  -> New best validation accuracy: {val_acc:.4f}")

        print("\nSupervised training completed!")
        print(f"  Best validation accuracy: {self.best_val_acc:.4f}")

        # Save training history
        self.save_history()

        return {
            "best_val_acc": self.best_val_acc,
            "final_train_acc": final_train_acc,
            "final_train_loss": final_train_loss,
            "total_epochs": self.epochs,
        }

    def save_checkpoint(self, is_best: bool = False, *, validation_score: float | None = None) -> Path:
        """Save model checkpoint to disk.

        Saves the model (T5 encoder + policy head) and optimizer state.
        Best model is saved to ``checkpoint_dir/best_model/``, epoch
        checkpoints to ``checkpoint_dir/epoch_N/``.

        Parameters
        ----------
        is_best : bool
            If True, save to ``best_model/`` directory.
        validation_score : float, optional
            Updated best score, committed with the model after a successful save.

        Returns
        -------
        Path
            Path to the saved checkpoint directory.
        """
        if is_best:
            save_path = self.checkpoint_dir / "best_model"
        else:
            save_path = self.checkpoint_dir / f"epoch_{self.current_epoch + 1}"

        score = self.best_val_acc if validation_score is None else validation_score
        state = {
            "epoch": self.current_epoch + 1,
            "best_val_acc": score,
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        with training_lock(self.checkpoint_dir.parent):
            save_training_checkpoint(self.model, save_path, state, self.write_checkpoint_sidecars)
            if is_best:
                self.best_checkpoint_path = save_path
                self.best_val_acc = score

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
                "input_mode": self.input_mode,
                "seed": self.config.get("supervised_seed", 42) if self.input_mode == "prefixes" else None,
                "train_examples": len(self.train_examples),
                "val_examples": len(self.val_examples),
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


@exclusive_training_run
def run_supervised_training(
    config: Dict[str, Any],
    train_questions: List[MCQuestion],
    val_questions: List[MCQuestion],
    test_questions: Optional[List[MCQuestion]] = None,
    *,
    write_checkpoint_sidecars: Callable[[Path], None] | None = None,
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

    write_checkpoint_sidecars : callable, optional
        Write run metadata into each staged checkpoint before publication.

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

    if int(config.get("supervised_epochs", 10)) <= 0:
        raise ValueError("supervised_epochs must be positive to produce a fresh checkpoint")
    if config.get("supervised_input_mode") == "prefixes" and test_questions is not None:
        validate_question_splits(train=train_questions, val=val_questions, test=test_questions)

    # Initialize model
    if config.get("supervised_input_mode") == "prefixes":
        torch.manual_seed(config.get("supervised_seed", 42))
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

    trainer.write_checkpoint_sidecars = write_checkpoint_sidecars

    # Train
    summary = trainer.train()
    if trainer.best_checkpoint_path is None:
        raise RuntimeError("Supervised training did not save a best checkpoint in this run")

    # Evaluate on test set if provided
    if test_questions is not None:
        print("\n" + "=" * 60)
        print("FINAL EVALUATION ON TEST SET")
        print("=" * 60)

        model.load(str(trainer.best_checkpoint_path))
        model.eval()

        # Evaluate
        test_loss, test_acc = trainer._evaluate(trainer._examples(test_questions))
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
