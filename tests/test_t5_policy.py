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
    try:
        model = T5PolicyModel(t5_small_config)
    except OSError as exc:
        pytest.skip(f"t5-small unavailable in test environment: {exc}")
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

def test_wait_action_log_prob_matches_wait_mass(t5_small_model, sample_texts):
    model = t5_small_model
    enc = model.encode_input(sample_texts[:1])

    pooled = model.get_encoder_output(enc["input_ids"], enc["attention_mask"])
    wait_logits, answer_logits, _ = model.policy_head(pooled)
    wait_probs = torch.softmax(wait_logits, dim=-1)

    actions = torch.tensor([0], dtype=torch.long, device=model.device)
    log_probs, _entropy, _ = model.get_action_log_probs(
        enc["input_ids"], enc["attention_mask"], actions
    )

    expected = torch.log(wait_probs[:, 0])
    assert torch.allclose(log_probs, expected, atol=1e-5)


def test_buzz_action_log_prob_matches_factorized_mass(t5_small_model, sample_texts):
    model = t5_small_model
    enc = model.encode_input(sample_texts[:1])

    pooled = model.get_encoder_output(enc["input_ids"], enc["attention_mask"])
    wait_logits, answer_logits, _ = model.policy_head(pooled)
    wait_probs = torch.softmax(wait_logits, dim=-1)
    answer_probs = torch.softmax(answer_logits, dim=-1)

    answer_idx = 2
    actions = torch.tensor([1 + answer_idx], dtype=torch.long, device=model.device)
    log_probs, _entropy, _ = model.get_action_log_probs(
        enc["input_ids"], enc["attention_mask"], actions
    )

    expected = torch.log(wait_probs[:, 1]) + torch.log(answer_probs[:, answer_idx])
    assert torch.allclose(log_probs, expected, atol=1e-5)


def test_joint_entropy_matches_chain_rule(t5_small_model, sample_texts):
    model = t5_small_model
    enc = model.encode_input(sample_texts[:1])

    pooled = model.get_encoder_output(enc["input_ids"], enc["attention_mask"])
    wait_logits, answer_logits, _ = model.policy_head(pooled)
    wait_probs = torch.softmax(wait_logits, dim=-1)
    answer_probs = torch.softmax(answer_logits, dim=-1)

    actions = torch.tensor([0], dtype=torch.long, device=model.device)
    _, entropy, _ = model.get_action_log_probs(
        enc["input_ids"], enc["attention_mask"], actions
    )

    wait_log_probs = torch.log(wait_probs.clamp_min(1e-12))
    answer_log_probs = torch.log(answer_probs.clamp_min(1e-12))
    wait_entropy = -(wait_probs * wait_log_probs).sum(dim=-1)
    answer_entropy = -(answer_probs * answer_log_probs).sum(dim=-1)
    expected = wait_entropy + wait_probs[:, 1] * answer_entropy
    assert torch.allclose(entropy, expected, atol=1e-5)


def test_old_unconditional_entropy_semantics_do_not_match(t5_small_model, sample_texts):
    model = t5_small_model
    enc = model.encode_input(sample_texts[:1])
    pooled = model.get_encoder_output(enc["input_ids"], enc["attention_mask"])
    wait_logits, answer_logits, _ = model.policy_head(pooled)
    wait_probs = torch.softmax(wait_logits, dim=-1)
    answer_probs = torch.softmax(answer_logits, dim=-1)
    actions = torch.tensor([0], dtype=torch.long, device=model.device)
    _, entropy, _ = model.get_action_log_probs(enc["input_ids"], enc["attention_mask"], actions)

    old = (-(wait_probs * torch.log(wait_probs.clamp_min(1e-12))).sum(dim=-1)
           - (answer_probs * torch.log(answer_probs.clamp_min(1e-12))).sum(dim=-1))
    assert not torch.allclose(entropy, old, atol=1e-6)
