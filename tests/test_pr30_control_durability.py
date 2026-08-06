from __future__ import annotations

import json

import pytest

from scripts.stopdff_v5 import sweep


def test_attempt_history_append_is_atomic_and_canonical(tmp_path):
    path = tmp_path / "attempts.jsonl"
    first = {"attempt": 1, "state": "started"}
    second = {"attempt": 2, "state": "started"}

    sweep._append_attempt(path, first)
    sweep._append_attempt(path, second)

    data = path.read_bytes()
    assert data.endswith(b"\n")
    assert [json.loads(line) for line in data.splitlines()] == [first, second]


def test_attempt_history_rejects_torn_tail_without_rewrite(tmp_path):
    path = tmp_path / "attempts.jsonl"
    torn = json.dumps({"attempt": 1, "state": "started"}).encode("utf-8")
    path.write_bytes(torn)

    with pytest.raises(ValueError, match="unterminated tail"):
        sweep._append_attempt(path, {"attempt": 2, "state": "started"})

    assert path.read_bytes() == torn


def test_attempt_history_replace_failure_preserves_old_bytes(
    tmp_path, monkeypatch
):
    path = tmp_path / "attempts.jsonl"
    sweep._append_attempt(path, {"attempt": 1, "state": "started"})
    before = path.read_bytes()

    def fail_replace(_source, _destination):
        raise OSError("simulated replace failure")

    monkeypatch.setattr(sweep.os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated replace failure"):
        sweep._append_attempt(path, {"attempt": 2, "state": "started"})

    assert path.read_bytes() == before
