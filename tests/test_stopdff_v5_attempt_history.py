from __future__ import annotations

import json

import pytest

from scripts.stopdff_v5 import sweep
from scripts.stopdff_v5.attempt_history import canonical_attempt_line


def _attempt(number: int) -> dict:
    resume = number > 1
    return {
        "attempt": number,
        "mode": "resume" if resume else "fresh",
        "command": ["dp_sweep"] + (["--resume"] if resume else []),
        "run_spec_id": "a" * 64,
        "adapter_id": "b" * 64,
        "bootstrap_plan_id": "c" * 64,
        "state": "started",
    }


def test_attempt_history_append_is_atomic_and_canonical(tmp_path):
    path = tmp_path / "attempts.jsonl"
    first = _attempt(1)
    second = _attempt(2)

    sweep._append_attempt(path, first)
    sweep._append_attempt(path, second)

    data = path.read_bytes()
    assert data.endswith(b"\n")
    assert [json.loads(line) for line in data.splitlines()] == [first, second]


def test_attempt_history_rejects_torn_tail_without_rewrite(tmp_path):
    path = tmp_path / "attempts.jsonl"
    torn = canonical_attempt_line(_attempt(1))[:-1]
    path.write_bytes(torn)

    with pytest.raises(ValueError, match="unterminated tail"):
        sweep._append_attempt(path, _attempt(2))

    assert path.read_bytes() == torn


def test_attempt_history_replace_failure_preserves_old_bytes(
    tmp_path, monkeypatch
):
    path = tmp_path / "attempts.jsonl"
    sweep._append_attempt(path, _attempt(1))
    before = path.read_bytes()

    def fail_replace(_source, _destination):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(sweep.os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated replace failure"):
        sweep._append_attempt(path, _attempt(2))

    assert path.read_bytes() == before
