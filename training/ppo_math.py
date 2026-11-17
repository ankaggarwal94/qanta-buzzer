"""Numerical operations shared by the PPO policy trainers."""

import torch


def normalize_advantages(advantages: torch.Tensor) -> torch.Tensor:
    """Use sample standardization when defined, retaining a singleton's policy gradient."""
    if advantages.numel() < 2:
        return advantages
    return (advantages - advantages.mean()) / (advantages.std() + 1e-8)
