"""Tests for qb_data/dspy_answer_profiles.py."""

from __future__ import annotations

import pytest


class TestBuildDspyProfiles:
    def test_requires_dspy_import(self) -> None:
        dspy = pytest.importorskip("dspy", reason="dspy not installed")
        from qb_data.dspy_answer_profiles import build_dspy_profiles
        assert callable(build_dspy_profiles)

    def test_fallback_to_existing(self) -> None:
        """When DSPy is not available, importing raises ImportError."""
        try:
            import dspy
            pytest.skip("dspy is installed; cannot test import failure")
        except ImportError:
            with pytest.raises(ImportError, match="dspy"):
                from qb_data.dspy_answer_profiles import build_dspy_profiles
                build_dspy_profiles(
                    answers=["A"],
                    existing_profiles={"A": "existing"},
                    dspy_config={"model": "test"},
                )
