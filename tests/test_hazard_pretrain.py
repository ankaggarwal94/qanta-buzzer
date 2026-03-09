import torch

from training.hazard_pretrain import compute_survival_terms, hazard_expected_nll_loss


def test_survival_terms_sum_to_one():
    stop_probs = torch.tensor([[0.2, 0.3, 0.5]], dtype=torch.float32)
    survival, stop_mass = compute_survival_terms(stop_probs)
    total = stop_mass.sum(dim=1) + survival[:, -1]
    assert torch.allclose(total, torch.ones_like(total), atol=1e-6)


def test_hazard_loss_penalizes_never_commit():
    stop_probs = torch.zeros((1, 3), dtype=torch.float32)
    nll = torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32)
    loss = hazard_expected_nll_loss(stop_probs, nll, beta_terminal=2.0)
    assert loss.item() >= 2.0


def test_hazard_loss_reduces_to_prefix_nll_when_commit_certain_immediately():
    stop_probs = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
    nll = torch.tensor([[0.7, 9.0, 9.0]], dtype=torch.float32)
    loss = hazard_expected_nll_loss(stop_probs, nll, beta_terminal=2.0)
    assert abs(loss.item() - 0.7) < 1e-5
