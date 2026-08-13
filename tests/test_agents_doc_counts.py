"""Guard: contract docs must not pin test-suite counts.

PR #30 round 2 (ps#M-V2-1 / testing#L-V2-01) and round 3 (ps#FYI-V3-1 /
testing#FYI-V3-01 / adversarial#L-V3-01): pinned counts like "~1204 test
functions across 87 test files" go stale within the very commit that writes
them -- every merged test re-arms the regression. The durable remedy is a
count-free Testing section that points at the live command
(``pytest tests/ --collect-only -q | tail -1``); this test machine-blocks the
regression class across BOTH contract docs (AGENTS.md and README.md) and a
wider set of stale-count phrasings than the two originally observed.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

# Match any pinned test-suite count phrasing: "1252 tests", "429 test",
# "~1204 test functions", "87 test files", "N test cases/modules/suites". The
# bare ``\d+ tests?\b`` alternate also fires on the "<n> test" prefix of the
# longer phrasings, so the class is caught regardless of the trailing noun.
PINNED_TEST_COUNT_PATTERN = re.compile(
    r"\d+\s+tests?\b|\d+\s+test\s+(?:functions?|files?|cases?|modules?|suites?)\b"
)

LIVE_COUNT_COMMAND = "pytest tests/ --collect-only -q | tail -1"

REPO_ROOT = Path(__file__).resolve().parents[1]
COUNT_FREE_DOCS = {
    "AGENTS.md": REPO_ROOT / "AGENTS.md",
    "README.md": REPO_ROOT / "README.md",
}

_DOC_PATHS = list(COUNT_FREE_DOCS.values())
_DOC_IDS = list(COUNT_FREE_DOCS)


@pytest.mark.parametrize("doc_path", _DOC_PATHS, ids=_DOC_IDS)
def test_doc_exists(doc_path):
    assert doc_path.is_file(), f"expected contract doc at {doc_path}"


@pytest.mark.parametrize("doc_path", _DOC_PATHS, ids=_DOC_IDS)
def test_doc_has_no_pinned_test_counts(doc_path):
    matches = PINNED_TEST_COUNT_PATTERN.findall(doc_path.read_text(encoding="utf-8"))
    assert not matches, (
        f"{doc_path.name} pins test-suite counts {matches!r}; pinned counts go "
        "stale immediately -- keep the doc count-free and point readers at the "
        f"live command ({LIVE_COUNT_COMMAND})."
    )


@pytest.mark.parametrize("doc_path", _DOC_PATHS, ids=_DOC_IDS)
def test_doc_keeps_live_count_command(doc_path):
    assert LIVE_COUNT_COMMAND in doc_path.read_text(encoding="utf-8"), (
        f"{doc_path.name} must keep the live test-count command so readers can "
        "measure the suite instead of trusting a pinned number."
    )
