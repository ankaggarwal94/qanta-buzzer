"""Canonical append-only attempt-history encoding and parsing."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .identity import loads_no_duplicate_keys


ATTEMPT_FIELDS = frozenset(
    {
        "attempt",
        "mode",
        "command",
        "run_spec_id",
        "adapter_id",
        "bootstrap_plan_id",
        "state",
    }
)


def canonical_attempt_line(record: Any) -> bytes:
    """Validate and encode one canonical ``state=started`` record."""
    if not isinstance(record, dict) or set(record) != ATTEMPT_FIELDS:
        raise ValueError("attempt record fields do not match the canonical contract")
    attempt = record.get("attempt")
    if (
        not isinstance(attempt, int)
        or isinstance(attempt, bool)
        or attempt < 1
    ):
        raise ValueError("attempt number must be a positive integer")
    if record.get("mode") not in {"fresh", "resume"}:
        raise ValueError("attempt mode must be 'fresh' or 'resume'")
    command = record.get("command")
    if not isinstance(command, list) or not all(
        isinstance(part, str) for part in command
    ):
        raise ValueError("attempt command must be a string list")
    for field in ("run_spec_id", "adapter_id", "bootstrap_plan_id"):
        value = record.get(field)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"attempt {field} must be lowercase SHA-256")
    if record.get("state") != "started":
        raise ValueError("attempt record state must be 'started'")
    return (json.dumps(record, sort_keys=True) + "\n").encode("utf-8")


def load_attempt_history(path: Path) -> tuple[bytes, list[dict[str, Any]]]:
    """Load a regular, newline-terminated history of exact canonical lines."""
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("attempt history is missing or noncanonical")
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise ValueError("attempt history cannot be read") from exc
    if not data or not data.endswith(b"\n"):
        raise ValueError("attempt history has an unterminated tail")

    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(data[:-1].split(b"\n"), start=1):
        if not line:
            raise ValueError(f"attempt history line {line_number} is empty")
        try:
            record = loads_no_duplicate_keys(line.decode("utf-8"))
            canonical = canonical_attempt_line(record)
        except (UnicodeDecodeError, ValueError, TypeError) as exc:
            raise ValueError(
                f"attempt history line {line_number} is invalid"
            ) from exc
        if canonical != line + b"\n":
            raise ValueError(
                f"attempt history line {line_number} is not canonical JSON"
            )
        records.append(record)
    return data, records
