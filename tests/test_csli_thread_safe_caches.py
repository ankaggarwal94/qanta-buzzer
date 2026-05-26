"""Thread-safety tests for scripts.compute_csli lazy-load caches (Iter1 IN-03).

The lazy-load model caches (_SBERT_MODEL, _T5_MODEL/_T5_TOKENIZER) and
answer-prior caches (_TFIDF_ANSWER_PRIOR, _SBERT_ANSWER_PRIOR) are
module-level globals used as test-and-set caches. The non-atomic
``if _CACHE is None`` check is the classic race-condition entry: two
threads can both observe None and both construct, producing
duplicated work and (for SBERT/T5) significant memory bloat from
holding two independent model copies.

Iter1 IN-03 wraps each cache in a dedicated ``threading.Lock`` and
uses the double-checked locking pattern. These tests verify the
contract by spinning 4 threads against ``_get_sbert_model()`` (the
specific case the dispatch instructions called out) and asserting
all four return the same identity-equal object with exactly one
constructor call observed.

The test uses a fake SentenceTransformer injected into
``sys.modules`` so we do not load the real ~90MB model in CI -- the
race semantics are independent of the model's identity, only the
constructor's call count matters. A ``threading.Barrier`` forces all
4 worker threads to enter the getter simultaneously, so the lock
contention is real (not racing a single thread to completion before
others start).
"""

from __future__ import annotations

import sys

# DATA-05 guard interaction (see WR-01):
# scripts.compute_csli's module-level _assert_no_controls_import()
# fires whenever ``evaluation.controls`` is already in
# ``sys.modules`` at our import time. Other test files transitively
# load evaluation.controls via ``scripts/evaluate_all.py`` (line 49)
# during pytest collection. Drop the offending module so the next
# import sees a clean state. Local to this test file; does not
# modify shared conftest.py. See test_bootstrap_ci_validation.py
# for the same pattern with a longer note.
sys.modules.pop("evaluation.controls", None)
sys.modules.pop("scripts.compute_csli", None)

import ast
import threading
import types
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pytest

import scripts.compute_csli as csli


_COMPUTE_CSLI_PATH = Path(__file__).resolve().parents[1] / "scripts" / "compute_csli.py"


# ---------------------------------------------------------------------------
# The dispatch-required test: SBERT lazy-load is thread-safe
# ---------------------------------------------------------------------------


def test_get_sbert_model_thread_safe_with_4_workers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """4 concurrent _get_sbert_model() calls return the SAME instance.

    Iter1 IN-03 wrapped the getter in a threading.Lock + double-
    checked locking. Without the lock, two threads racing the
    `if _SBERT_MODEL is None` check would BOTH construct, duplicating
    the model in memory and returning two different `id()`s.

    Test mechanics:
      1. Reset the module-level cache to None (the prior test order
         may have populated it).
      2. Inject a fake ``sentence_transformers`` module so we do not
         load the real ~90MB model. The fake constructor increments a
         call counter so we can verify exactly one construction.
      3. Wrap each worker in a ``Barrier(4)`` so all four threads
         enter the getter simultaneously (otherwise the test might
         race a single thread to completion before the others start,
         and the test would pass even on broken code).
      4. Assert all four returned values are ``is`` the same object
         AND the constructor was called exactly once.
    """
    monkeypatch.setattr(csli, "_SBERT_MODEL", None, raising=False)
    monkeypatch.setattr(csli, "_SBERT_LOCK", threading.Lock(), raising=False)

    call_count = [0]
    counter_lock = threading.Lock()

    class _FakeSentenceTransformer:
        """Stand-in for sentence_transformers.SentenceTransformer."""

        def __init__(self, name: str) -> None:
            with counter_lock:
                call_count[0] += 1
            self.name = name
            self._instance_id = id(self)

        def encode(self, *args, **kwargs):  # pragma: no cover - not exercised
            import numpy as np
            return np.zeros((1, 8), dtype=np.float32)

    fake_module = types.ModuleType("sentence_transformers")
    fake_module.SentenceTransformer = _FakeSentenceTransformer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)

    barrier = threading.Barrier(4, timeout=10.0)

    def worker() -> object:
        barrier.wait(timeout=10.0)
        return csli._get_sbert_model()

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(worker) for _ in range(4)]
        results = [f.result(timeout=15.0) for f in as_completed(futures)]

    assert len(results) == 4
    first = results[0]
    for i, r in enumerate(results[1:], start=1):
        assert r is first, (
            f"Iter1 IN-03 regression: thread {i} got a different "
            f"SBERT instance (id={id(r)}) than thread 0 (id={id(first)}). "
            "The lazy-load cache is racing -- two threads constructed "
            "independently. Wrap _get_sbert_model in _SBERT_LOCK."
        )
    assert call_count[0] == 1, (
        f"Iter1 IN-03 regression: SentenceTransformer constructor was "
        f"called {call_count[0]} times for 4 concurrent _get_sbert_model() "
        "calls. Expected exactly 1. The double-checked-lock re-check "
        "inside _SBERT_LOCK is missing or broken."
    )


# ---------------------------------------------------------------------------
# AST guard: the lock wiring cannot silently regress
# ---------------------------------------------------------------------------


def _compute_csli_source() -> str:
    return _COMPUTE_CSLI_PATH.read_text(encoding="utf-8")


def test_compute_csli_imports_threading() -> None:
    """The threading module must be imported at module scope.

    Without this import, ``threading.Lock()`` in the global
    initializers would NameError at import time. Catches a stray
    edit that strips the import on the assumption it is unused.
    """
    src = _compute_csli_source()
    tree = ast.parse(src)
    has_threading = any(
        (isinstance(node, ast.Import) and any(a.name == "threading" for a in node.names))
        for node in ast.walk(tree)
    )
    assert has_threading, (
        "Iter1 IN-03: scripts/compute_csli.py must `import threading` "
        "for the lazy-load cache locks to exist."
    )


def test_compute_csli_declares_all_expected_locks() -> None:
    """All four cache locks must exist at module scope.

    The four caches are: SBERT model, T5 model (+ tokenizer together
    under one lock), TF-IDF answer prior, SBERT answer prior. Each
    has a dedicated lock so independent caches don't pointlessly
    serialize. Catches a regression where one of the locks gets
    deleted under the (mistaken) assumption it is unused.
    """
    src = _compute_csli_source()
    tree = ast.parse(src)
    locks_assigned: set[str] = set()
    expected = {"_SBERT_LOCK", "_T5_LOCK", "_TFIDF_PRIOR_LOCK", "_SBERT_PRIOR_LOCK"}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in expected:
                locks_assigned.add(target.id)
    missing = expected - locks_assigned
    assert not missing, (
        f"Iter1 IN-03: scripts/compute_csli.py is missing lock "
        f"declarations for {sorted(missing)}. Each lazy-load cache "
        "must have its own threading.Lock for thread-safety."
    )
