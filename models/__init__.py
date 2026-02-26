"""
Models Package

Likelihood models, belief feature extraction, and policy model interfaces
for the quiz bowl RL buzzer system.
"""

from models.features import extract_belief_features, entropy_of_distribution
from models.likelihoods import (
    LikelihoodModel,
    SBERTLikelihood,
    T5Likelihood,
    TfIdfLikelihood,
    build_likelihood_from_config,
)

# Lazy import: T5PolicyModel and PolicyHead require transformers + torch.
# Import on demand to keep package lightweight for belief-feature-only usage.


def __getattr__(name: str):
    if name in ("T5PolicyModel", "PolicyHead"):
        from models.t5_policy import T5PolicyModel, PolicyHead
        return {"T5PolicyModel": T5PolicyModel, "PolicyHead": PolicyHead}[name]
    raise AttributeError(f"module 'models' has no attribute {name!r}")


__all__ = [
    "extract_belief_features",
    "entropy_of_distribution",
    "LikelihoodModel",
    "TfIdfLikelihood",
    "SBERTLikelihood",
    "T5Likelihood",
    "build_likelihood_from_config",
    "T5PolicyModel",
    "PolicyHead",
]
