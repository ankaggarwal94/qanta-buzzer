"""Test suite for models/features.py — belief feature extraction.

Covers ENV-03: Belief feature extraction produces (K+6)-dimensional vectors
with correct derived features (entropy, margin, stability, progress).
"""

from __future__ import annotations

import numpy as np
import pytest

from models.features import entropy_of_distribution, extract_belief_features


# ------------------------------------------------------------------ #
# Tests for entropy_of_distribution
# ------------------------------------------------------------------ #


class TestEntropyOfDistribution:
    """Tests for Shannon entropy computation."""

    def test_entropy_uniform(self) -> None:
        """Uniform distribution over 4 options has maximum entropy ln(4)."""
        belief = np.array([0.25, 0.25, 0.25, 0.25])
        ent = entropy_of_distribution(belief)
        # ln(4) ~ 1.3863
        assert 1.35 < ent < 1.40, f"Uniform entropy {ent} not near ln(4)=1.3863"

    def test_entropy_peaked(self) -> None:
        """Peaked distribution has low entropy."""
        belief = np.array([0.9, 0.05, 0.03, 0.02])
        ent = entropy_of_distribution(belief)
        assert ent < 0.5, f"Peaked entropy {ent} should be < 0.5"

    def test_entropy_deterministic_no_nan(self) -> None:
        """Deterministic distribution [1, 0, 0, 0] produces no NaN/inf."""
        belief = np.array([1.0, 0.0, 0.0, 0.0])
        ent = entropy_of_distribution(belief)
        assert np.isfinite(ent), f"Entropy {ent} should be finite"
        assert ent >= 0.0, f"Entropy {ent} should be non-negative"

    def test_entropy_deterministic_last(self) -> None:
        """Deterministic distribution [0, 0, 0, 1] produces no NaN/inf."""
        belief = np.array([0.0, 0.0, 0.0, 1.0])
        ent = entropy_of_distribution(belief)
        assert np.isfinite(ent), f"Entropy {ent} should be finite"
        assert ent >= 0.0, f"Entropy {ent} should be non-negative"

    def test_entropy_binary(self) -> None:
        """Binary uniform distribution has entropy ln(2)."""
        belief = np.array([0.5, 0.5])
        ent = entropy_of_distribution(belief)
        assert abs(ent - np.log(2)) < 0.01, f"Binary entropy {ent} != ln(2)={np.log(2):.4f}"


# ------------------------------------------------------------------ #
# Tests for extract_belief_features
# ------------------------------------------------------------------ #


class TestExtractBeliefFeatures:
    """Tests for belief feature vector extraction."""

    def test_feature_shape(self) -> None:
        """Output shape is (K+6,) for K=4 belief vector."""
        belief = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
        features = extract_belief_features(belief, None, 0, 6)
        assert features.shape == (10,), f"Expected (10,), got {features.shape}"

    def test_feature_shape_k3(self) -> None:
        """Output shape adapts to K=3."""
        belief = np.array([0.4, 0.3, 0.3], dtype=np.float32)
        features = extract_belief_features(belief, None, 0, 5)
        assert features.shape == (9,), f"Expected (9,), got {features.shape}"

    def test_feature_contents_belief_prefix(self) -> None:
        """First K elements of feature vector are the raw belief."""
        belief = np.array([0.5, 0.3, 0.15, 0.05], dtype=np.float32)
        features = extract_belief_features(belief, None, 2, 6)
        np.testing.assert_array_almost_equal(
            features[:4], belief, decimal=5,
            err_msg="First K elements should match input belief",
        )

    def test_derived_top_p(self) -> None:
        """top_p is max(belief)."""
        belief = np.array([0.5, 0.3, 0.15, 0.05], dtype=np.float32)
        features = extract_belief_features(belief, None, 2, 6)
        assert abs(features[4] - 0.5) < 1e-5, f"top_p={features[4]}, expected 0.5"

    def test_derived_margin(self) -> None:
        """margin is top_p - second_highest."""
        belief = np.array([0.5, 0.3, 0.15, 0.05], dtype=np.float32)
        features = extract_belief_features(belief, None, 2, 6)
        expected_margin = 0.5 - 0.3
        assert abs(features[5] - expected_margin) < 1e-5, (
            f"margin={features[5]}, expected {expected_margin}"
        )

    def test_derived_entropy_in_range(self) -> None:
        """Entropy is in a reasonable range for a non-uniform distribution."""
        belief = np.array([0.5, 0.3, 0.15, 0.05], dtype=np.float32)
        features = extract_belief_features(belief, None, 2, 6)
        ent = features[6]
        assert 0 < ent < np.log(4) + 0.01, f"Entropy {ent} out of range"

    def test_stability_none_prev(self) -> None:
        """Stability is 0.0 when prev_belief is None (first step)."""
        belief = np.array([0.5, 0.3, 0.15, 0.05], dtype=np.float32)
        features = extract_belief_features(belief, None, 0, 6)
        assert features[7] == 0.0, f"Stability={features[7]}, expected 0.0 for first step"

    def test_stability_computation(self) -> None:
        """Stability tracks L1 distance between consecutive beliefs."""
        prev_belief = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
        belief = np.array([0.5, 0.3, 0.15, 0.05], dtype=np.float32)
        features = extract_belief_features(belief, prev_belief, 1, 6)
        expected_stability = float(np.abs(belief - prev_belief).sum())
        assert abs(features[7] - expected_stability) < 1e-5, (
            f"Stability={features[7]}, expected {expected_stability}"
        )

    def test_progress(self) -> None:
        """progress = step_idx / total_steps."""
        belief = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
        features = extract_belief_features(belief, None, 3, 6)
        expected_progress = 3.0 / 6.0
        assert abs(features[8] - expected_progress) < 1e-5, (
            f"progress={features[8]}, expected {expected_progress}"
        )

    def test_clue_idx_norm(self) -> None:
        """clue_idx_norm = step_idx / (total_steps - 1)."""
        belief = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
        features = extract_belief_features(belief, None, 3, 6)
        expected_norm = 3.0 / 5.0
        assert abs(features[9] - expected_norm) < 1e-5, (
            f"clue_idx_norm={features[9]}, expected {expected_norm}"
        )

    def test_dtype_float32(self) -> None:
        """Output dtype is float32."""
        belief = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
        features = extract_belief_features(belief, None, 0, 6)
        assert features.dtype == np.float32, f"Expected float32, got {features.dtype}"

    def test_invalid_2d_belief_raises(self) -> None:
        """Passing a 2D belief array raises ValueError."""
        belief = np.array([[0.5, 0.5]], dtype=np.float32)
        with pytest.raises(ValueError, match="1D"):
            extract_belief_features(belief, None, 0, 1)
