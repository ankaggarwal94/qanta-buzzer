"""MLP value model for learned StopDFF continuation values."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn


DEFAULT_HIDDEN_SIZES = (128, 128)


class StopDFFValueModel(nn.Module):
    """Estimate scalar continuation value from fixed tabular features.

    Parameters
    ----------
    input_dim:
        Number of tabular input features.
    hidden_sizes:
        Hidden ReLU widths, defaulting to ``(128, 128)``.
    dropout:
        Dropout after hidden activations.
    feature_schema:
        Optional plain-dict feature metadata.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_sizes: Sequence[int] = DEFAULT_HIDDEN_SIZES,
        dropout: float = 0.0,
        feature_schema: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.input_dim = _validate_input_dim(input_dim)
        self.hidden_sizes = _validate_hidden_sizes(hidden_sizes)
        self.dropout = _validate_dropout(dropout)
        self.feature_schema = dict(feature_schema or {})

        layers: list[nn.Module] = []
        prev_dim = self.input_dim
        for hidden_dim in self.hidden_sizes:
            layers.extend((nn.Linear(prev_dim, hidden_dim), nn.ReLU()))
            if self.dropout > 0.0:
                layers.append(nn.Dropout(self.dropout))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Predict values for a batch or a single feature vector.

        Parameters
        ----------
        features:
            Tensor with shape ``[batch, n_features]`` or ``[n_features]``.

        Returns
        -------
        torch.Tensor
            Scalar predictions with shape ``[batch]``.
        """

        if not isinstance(features, torch.Tensor):
            raise TypeError("features must be a torch.Tensor")
        if features.ndim == 1:
            features = features.unsqueeze(0)
        if features.ndim != 2:
            raise ValueError("features must have shape [batch, n_features] or [n_features]")
        if features.shape[-1] != self.input_dim:
            raise ValueError(
                f"expected {self.input_dim} features, got {features.shape[-1]}"
            )
        return self.network(features).squeeze(-1)

    def to_config(self) -> dict[str, Any]:
        """Return checkpoint-ready config and feature schema."""
        return {
            "input_dim": self.input_dim,
            "hidden_sizes": list(self.hidden_sizes),
            "dropout": self.dropout,
            "feature_schema": dict(self.feature_schema),
        }

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "StopDFFValueModel":
        """Instantiate a model from serialized configuration."""
        return cls(
            input_dim=int(config["input_dim"]),
            hidden_sizes=tuple(config.get("hidden_sizes", DEFAULT_HIDDEN_SIZES)),
            dropout=float(config.get("dropout", 0.0)),
            feature_schema=config.get("feature_schema"),
        )

    @classmethod
    def load_from_state_dict(
        cls,
        state_dict: Mapping[str, torch.Tensor],
        config: Mapping[str, Any],
    ) -> "StopDFFValueModel":
        """Load weights into a fresh model created from config."""
        model = cls.from_config(config)
        model.load_state_dict(dict(state_dict))
        return model


def _validate_input_dim(input_dim: int) -> int:
    input_dim = int(input_dim)
    if input_dim <= 0:
        raise ValueError("input_dim must be > 0")
    return input_dim


def _validate_hidden_sizes(hidden_sizes: Sequence[int]) -> tuple[int, ...]:
    sizes = tuple(int(size) for size in hidden_sizes)
    if any(size <= 0 for size in sizes):
        raise ValueError("hidden sizes must all be > 0")
    return sizes


def _validate_dropout(dropout: float) -> float:
    dropout = float(dropout)
    if dropout < 0.0 or dropout >= 1.0:
        raise ValueError("dropout must be in [0.0, 1.0)")
    return dropout
