"""Guard: AGENTS.md must not pin test-suite counts.

PR #30 round 2 (ps#M-V2-1 / testing#L-V2-01): pinned counts like
"~1204 test functions across 87 test files" go stale within the very
commit that writes them -- every merged test re-arms the regression.
The durable remedy is a count-free Testing section that points at the
live command (`pytest tests/ --collect-only -q | tail -1`); this test
machine-blocks the regression class.
"""

from __future__ import annotations

import re
from pathlib import Path

PINNED_TEST_COUNT_PATTERN = re.compile(r"\d+ test functions|\d+ test files")

AGENTS_MD = Path(__file__).resolve().parents[1] / "AGENTS.md"


class TestAgentsDocCounts:
    """AGENTS.md stays count-free so it can never go stale."""

    def test_agents_md_exists(self):
        assert AGENTS_MD.is_file(), f"expected repo contract doc at {AGENTS_MD}"

    def test_agents_md_has_no_pinned_test_counts(self):
        text = AGENTS_MD.read_text(encoding="utf-8")
        matches = PINNED_TEST_COUNT_PATTERN.findall(text)
        assert not matches, (
            f"AGENTS.md pins test-suite counts {matches!r}; pinned counts go "
            "stale immediately -- keep the doc count-free and point readers at "
            "the live command (pytest tests/ --collect-only -q | tail -1)."
        )

    def test_agents_md_keeps_live_count_command(self):
        text = AGENTS_MD.read_text(encoding="utf-8")
        assert "pytest tests/ --collect-only -q | tail -1" in text, (
            "AGENTS.md must keep the live test-count command so readers can "
            "measure the suite instead of trusting a pinned number."
        )
