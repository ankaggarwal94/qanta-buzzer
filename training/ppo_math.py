"""Numerical operations shared by the PPO policy trainers."""

import torch


def normalize_advantages(advantages: torch.Tensor) -> torch.Tensor:
    """Standardize PPO advantages while preserving singleton policy gradients.

    Parameters
    ----------
    advantages : torch.Tensor
        Advantage estimates for the rollout.

    Returns
    -------
    torch.Tensor
        Empty and singleton inputs are returned unchanged. For larger inputs,
        subtract the mean and divide by the sample standard deviation plus 1e-8.
    """
    if advantages.numel() < 2:
        return advantages
    return (advantages - advantages.mean()) / (advantages.std() + 1e-8)
