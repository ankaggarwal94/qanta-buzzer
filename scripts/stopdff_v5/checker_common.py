"""Shared primitives for the checker module family (ACCEPTANCE_CONTRACT.md).

Core value type, canonical-path classification, strict scalar predicates, and
the strict JSON/row loaders shared by ``checker`` and its extracted sections
(``checker_runspec``, ``checker_png``, ``checker_attempts``). Behavior here is
part of the standalone-checker acceptance surface; ``checker`` re-exports the
historical names so external callers keep importing from ``checker``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .identity import is_sha256_hex, loads_no_duplicate_keys
from .rowio import read_jsonl_gz

_FLOAT_TOL = 1e-9
_INTERRUPTED_REASON = "terminal_result_missing_at_resume"


@dataclass
class CheckResult:
    passed: bool
    errors: list[str] = field(default_factory=list)
    recomputed: dict[str, Any] = field(default_factory=dict)


def _canonical_path_issue(
    path: Path,
    *,
    expect_directory: bool,
) -> str | None:
    """Classify a path that must not follow symlinks."""
    if path.is_symlink():
        return "symlink"
    if not path.exists():
        return "missing"
    if expect_directory:
        return None if path.is_dir() else "wrong_type"
    return None if path.is_file() else "wrong_type"


def _is_strict_int(value: Any, *, minimum: int | None = None) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and (minimum is None or value >= minimum)
    )


def _is_finite_number(
    value: Any,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        number = float(value)
    except (OverflowError, ValueError):
        return False
    return (
        math.isfinite(number)
        and (minimum is None or number >= minimum)
        and (maximum is None or number <= maximum)
    )


def _scientific_equal(actual: Any, expected: Any) -> bool:
    """Compare recomputable JSON claims without Python's coercive equality."""
    if expected is None:
        return actual is None
    if isinstance(expected, bool):
        return actual is expected
    if isinstance(expected, int):
        return _is_strict_int(actual) and actual == expected
    if isinstance(expected, float):
        return _is_finite_number(actual) and math.isclose(
            float(actual),
            expected,
            rel_tol=0.0,
            abs_tol=_FLOAT_TOL,
        )
    if isinstance(expected, str):
        return isinstance(actual, str) and actual == expected
    if isinstance(expected, dict):
        return (
            isinstance(actual, dict)
            and set(actual) == set(expected)
            and all(
                _scientific_equal(actual[key], value)
                for key, value in expected.items()
            )
        )
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(
                _scientific_equal(actual_value, expected_value)
                for actual_value, expected_value in zip(actual, expected)
            )
        )
    return type(actual) is type(expected) and actual == expected


def _is_quantized_number(value: Any, *, decimal_places: int) -> bool:
    """Return whether a finite number is unchanged by producer rounding."""
    if not _is_finite_number(value):
        return False
    number = float(value)
    return number == round(number, decimal_places)


def _producer_hash_errors(
    value: Any,
    *,
    label: str,
    required_keys: set[str] | None = None,
) -> list[str]:
    errors: list[str] = []
    if not isinstance(value, dict) or not value:
        return [f"{label} must be a nonempty object"]
    if required_keys is not None and set(value) != required_keys:
        errors.append(
            f"{label} keys do not match the canonical producer set"
        )
    for key, digest in value.items():
        if (
            not isinstance(key, str)
            or not key
            or key.startswith("/")
            or ".." in Path(key).parts
        ):
            errors.append(f"{label} contains an invalid producer path")
        if not is_sha256_hex(digest):
            errors.append(f"{label} hash for {key!r} must be 64-hex")
    return errors


def load_json(path: Path) -> Any:
    return loads_no_duplicate_keys(Path(path).read_text(encoding="utf-8"))


# The strict adapter-row reader is shared with the producer side; the checker
# keeps its historical name as the module-level entry point.
load_jsonl_gz = read_jsonl_gz


def load_adapter_rows(bundle_dir: Path) -> list[dict]:
    rows = load_jsonl_gz(bundle_dir / "fit_rows.jsonl.gz")
    rows += load_jsonl_gz(bundle_dir / "eval_rows.jsonl.gz")
    return rows
