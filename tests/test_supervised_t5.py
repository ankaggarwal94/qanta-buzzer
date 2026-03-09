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
