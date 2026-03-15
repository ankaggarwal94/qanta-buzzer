"""Hazard pretraining bridge utilities for stopping-aware warm starts."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class HazardBatchOutput:
    """Container for hazard-bridge intermediate tensors."""

    stop_probs: torch.Tensor
    survival: torch.Tensor
    stop_mass: torch.Tensor
    nll_per_prefix: torch.Tensor
    loss: torch.Tensor


def compute_survival_terms(stop_probs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute survival and stop-mass terms from per-prefix stop probabilities."""
    stay_probs = 1.0 - stop_probs
    batch_size, steps = stop_probs.shape
    survival = torch.ones(
        (batch_size, steps + 1), dtype=stop_probs.dtype, device=stop_probs.device
    )
    if steps > 0:
        survival[:, 1:] = torch.cumprod(stay_probs, dim=1)
    stop_mass = survival[:, :-1] * stop_probs
    return survival, stop_mass


def hazard_expected_nll_loss(
    stop_probs: torch.Tensor,
    nll_per_prefix: torch.Tensor,
    beta_terminal: float = 1.0,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute the hazard-bridge expected NLL loss prior to PPO."""
    survival, stop_mass = compute_survival_terms(stop_probs)
    weighted_nll = stop_mass * nll_per_prefix
    if mask is not None:
        weighted_nll = weighted_nll * mask
    seq_loss = weighted_nll.sum(dim=1) + beta_terminal * survival[:, -1]
    return seq_loss.mean()
