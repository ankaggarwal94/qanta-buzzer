"""Cross-consumer dual-shape acceptance for {val,test}_dataset.json (Iter2 IN-01).

The on-disk shape of split JSON files depends on which producer ran
last:

- ``qb_data.dataset_splits.save_splits`` (called by
  ``scripts/fresh_split.py``) writes the wrapped form
  ``{"metadata": {...}, "questions": [...]}``.
- ``scripts/build_mc_dataset.py`` writes the plain-list form via
  ``save_json``.

Three downstream consumers read these files:

- ``scripts/compute_csli.py``
- ``scripts/compute_stopdff.py``
- ``scripts/compute_prefix_calibration.py``

Iter1 WR-05 fixed ``compute_csli.py`` only. Iter2 IN-01 surfaced the
cross-consumer gap. The Phase 02 iteration-3 deferrals fix extracts a
shared helper ``scripts._common.iter_split_questions`` and wires all
three consumers through it. These tests pin the contract.

The tests target the helper directly (round-trip both shapes) and
also assert via AST that each consumer uses the helper at the
expected call site. The AST check is the regression guard: a future
edit that re-inlines ``test_data["questions"]`` (the symptom of the
original gap) will fail the test before reaching the runtime.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from scripts._common import iter_split_questions


_SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
_CONSUMERS = (
    "compute_csli.py",
    "compute_stopdff.py",
    "compute_prefix_calibration.py",
)


# ---------------------------------------------------------------------------
# Helper behavior
# ---------------------------------------------------------------------------


def test_iter_split_questions_accepts_wrapped_form() -> None:
    """The wrapped form (fresh_split.py producer) is accepted."""
    payload = {
        "metadata": {"split": "test", "n": 2},
        "questions": [
            {"qid": "q1", "options": ["a", "b", "c", "d"], "gold_index": 0},
            {"qid": "q2", "options": ["e", "f", "g", "h"], "gold_index": 2},
        ],
    }
    out = iter_split_questions(payload, source_path="test_dataset.json")
    assert len(out) == 2
    assert {str(q["qid"]) for q in out} == {"q1", "q2"}


def test_iter_split_questions_accepts_plain_list_form() -> None:
    """The plain-list form (build_mc_dataset.py producer) is accepted."""
    payload = [
        {"qid": "q1", "options": ["a", "b", "c", "d"], "gold_index": 0},
        {"qid": "q2", "options": ["e", "f", "g", "h"], "gold_index": 2},
    ]
    out = iter_split_questions(payload, source_path="test_dataset.json")
    assert len(out) == 2
    assert {str(q["qid"]) for q in out} == {"q1", "q2"}


def test_iter_split_questions_rejects_dict_without_questions() -> None:
    """A dict without a ``"questions"`` key is rejected loudly."""
    payload = {"metadata": {"split": "test"}, "items": [{"qid": "q1"}]}
    with pytest.raises(RuntimeError, match="Unrecognized shape"):
        iter_split_questions(payload, source_path="bogus.json")


def test_iter_split_questions_rejects_scalar() -> None:
    """A scalar/None payload is rejected loudly."""
    with pytest.raises(RuntimeError, match="Unrecognized shape"):
        iter_split_questions(None, source_path="bogus.json")
    with pytest.raises(RuntimeError, match="Unrecognized shape"):
        iter_split_questions(42, source_path="bogus.json")


def test_iter_split_questions_returns_list_reusable_across_consumers() -> None:
    """Return type is a concrete list so callers can iterate multiple times.

    The compute scripts do ``set(...)`` followed by ``len(...)`` on the
    result. A generator would have been single-pass and silently empty
    on the second touch -- that bug class is excluded by returning a
    list.
    """
    payload = {"questions": [{"qid": "q1"}, {"qid": "q2"}]}
    out = iter_split_questions(payload, source_path=None)
    qids_first = {str(q["qid"]) for q in out}
    qids_second = {str(q["qid"]) for q in out}  # second iteration
    assert qids_first == qids_second == {"q1", "q2"}
    assert len(out) == 2


# ---------------------------------------------------------------------------
# AST guard: every consumer routes through iter_split_questions
# ---------------------------------------------------------------------------


def _consumer_source(name: str) -> str:
    """Return the source text of a consumer script."""
    path = _SCRIPTS_DIR / name
    assert path.exists(), f"Consumer script not found: {path}"
    return path.read_text(encoding="utf-8")


@pytest.mark.parametrize("consumer", _CONSUMERS)
def test_consumer_imports_iter_split_questions(consumer: str) -> None:
    """Each consumer must reference ``iter_split_questions``.

    The import can be top-level OR inside ``main()`` (the latter is
    the established pattern in this codebase because of sys.path
    setup ordering). What matters is the symbol name appears
    somewhere in the source.
    """
    src = _consumer_source(consumer)
    assert "iter_split_questions" in src, (
        f"{consumer} does not reference iter_split_questions. "
        "The Iter2 IN-01 fix requires every test_dataset.json consumer "
        "to route through scripts._common.iter_split_questions so the "
        "wrapped + plain-list producer shapes are both accepted."
    )


@pytest.mark.parametrize("consumer", _CONSUMERS)
def test_consumer_does_not_index_test_data_questions_directly(consumer: str) -> None:
    """No consumer should index ``test_data["questions"]`` directly.

    This is the symptom of the original Iter2 IN-01 gap: the consumer
    that indexes ``data["questions"]`` crashes when the producer wrote
    the plain-list form. The AST guard catches a regression where a
    future maintainer re-inlines the bad pattern.

    Walks ``ast.Subscript`` nodes whose slice is the string literal
    ``"questions"`` and whose value is a ``Name`` matching
    ``test_data`` or ``val_data``. The helper call
    ``iter_split_questions(test_data, ...)`` does NOT match because
    it is an ``ast.Call``, not an ``ast.Subscript`` on
    ``test_data``.
    """
    src = _consumer_source(consumer)
    tree = ast.parse(src)
    bad_subscripts: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Subscript):
            continue
        if not isinstance(node.value, ast.Name):
            continue
        if node.value.id not in ("test_data", "val_data"):
            continue
        slice_node = node.slice
        if isinstance(slice_node, ast.Constant) and slice_node.value == "questions":
            bad_subscripts.append(
                f"  {consumer}:{node.lineno} -- "
                f"`{node.value.id}[\"questions\"]` "
                "(use iter_split_questions instead)"
            )
    assert not bad_subscripts, (
        "Iter2 IN-01 regression: direct indexing of "
        "test_data/val_data with the literal 'questions' key found. "
        "These call sites must route through "
        "scripts._common.iter_split_questions so both producer shapes "
        "are accepted.\n" + "\n".join(bad_subscripts)
    )
