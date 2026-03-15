"""Tests for qb_data/dspy_answer_profiles.py."""

from __future__ import annotations

import pytest


class TestBuildDspyProfiles:
    def test_module_importable_without_dspy(self) -> None:
        """The module imports cleanly even when dspy is not installed."""
        from qb_data.dspy_answer_profiles import build_dspy_profiles
        assert callable(build_dspy_profiles)

    def test_runtime_call_without_dspy_raises(self) -> None:
        """Calling build_dspy_profiles without dspy raises ImportError."""
        try:
            import dspy
            pytest.skip("dspy is installed; cannot test import failure")
        except ImportError:
            from qb_data.dspy_answer_profiles import build_dspy_profiles
            with pytest.raises(ImportError, match="dspy"):
                build_dspy_profiles(
                    answers=["A"],
                    existing_profiles={"A": "existing"},
                    dspy_config={"model": "test"},
                )

    def test_with_dspy_installed(self) -> None:
        """When dspy IS installed, the function is callable."""
        dspy = pytest.importorskip("dspy", reason="dspy not installed")
        from qb_data.dspy_answer_profiles import build_dspy_profiles
        assert callable(build_dspy_profiles)
