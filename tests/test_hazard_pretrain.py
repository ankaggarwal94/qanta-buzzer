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
