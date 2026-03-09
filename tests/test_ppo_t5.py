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
