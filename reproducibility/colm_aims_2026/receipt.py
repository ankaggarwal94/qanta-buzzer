"""Verification + suite-evidence receipts (schema-versioned, create-once).

Spec rules owned here: R-036 (per-run verification receipts), R-070
(machine-readable suite receipts; A'-F4 defect shape rejected).
Spec: .correctless/specs/camera-ready-aims-evidence-2.md
"""
from __future__ import annotations

import time
import uuid
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from scripts.stopdff_v5 import fileio

from .schema import (
    SCHEMA_VERSION,
    ColmAimsError,
    check_schema_version,
    encode_json,
    is_git_object_id,
    is_path_component,
    is_real_int,
    is_sha256_hex,
    resolves_inside,
)


def _unique_run_id() -> str:
    # Fixed-width nanosecond prefix so run-scoped receipt names sort in
    # emission order; uuid suffix guarantees cross-process uniqueness.
    return f"{time.time_ns():020d}-{uuid.uuid4().hex[:12]}"


def emit_receipt(
    report_payload: dict[str, Any],
    *,
    receipts_dir: Path,
    verified_tree: Path,
    run_id: str | None = None,
) -> Path:
    """Emit one schema-versioned JSON receipt outside the verified tree.

    ``run_id`` pins the run-scoped receipt name (auto-generated unique when
    None); an existing receipt at the resulting path fails closed
    (create-once, R-016 primitives).
    """
    receipts_dir = Path(receipts_dir)
    verified_tree = Path(verified_tree)
    if run_id is None:
        run_id = _unique_run_id()
    if not is_path_component(run_id):
        raise ColmAimsError(
            f"receipt run_id {run_id!r} must be a single path component"
            " (R-036)"
        )
    path = receipts_dir / f"receipt-{run_id}.json"
    if resolves_inside(path, verified_tree):
        raise ColmAimsError(
            "receipt path must be outside the verified artifact tree (R-036)"
        )
    payload = dict(report_payload)
    payload.setdefault("schema_version", SCHEMA_VERSION)
    fileio.create_once_bytes(
        path, encode_json(payload), exists_label="verification receipt"
    )
    return path


# ---------------------------------------------------------------------------
# R-070: machine-readable suite-evidence receipts
# ---------------------------------------------------------------------------

SUITE_RECEIPT_REQUIRED_FIELDS = (
    "environment_lock_sha256",
    "workflow_sha256",
    "interpreter_realpath",
    "commit",
    "tree_sha256",
    "dirty",
    "command",
    "exit_code",
    "junit_sha256",
    "transcript_sha256",
    "counts",
    "skip_identities",
)


class SuiteReceiptError(ColmAimsError):
    """Suite-evidence receipt violates the R-070 binding contract."""


def validate_suite_receipt(receipt: dict[str, Any]) -> None:
    """Validate one machine-readable suite-evidence receipt (R-070).

    Every binding is a HASH or an exact machine-readable value; the A'-F4
    defect shape — ``environment_lock_sha256`` as a metadata object instead of a
    lockfile/environment-export hash — is rejected by name.
    """
    if not isinstance(receipt, dict):
        raise SuiteReceiptError("suite receipt must be an object (R-070)")
    if set(receipt) != {"schema_version", *SUITE_RECEIPT_REQUIRED_FIELDS}:
        raise SuiteReceiptError(
            "suite receipt must have the exact closed R-070 shape"
        )
    check_schema_version(receipt, "suite receipt")
    missing = sorted(
        set(SUITE_RECEIPT_REQUIRED_FIELDS) - set(receipt)
    )
    if missing:
        raise SuiteReceiptError(
            f"suite receipt missing required field(s) {missing} (R-070)"
        )
    if not is_sha256_hex(receipt["environment_lock_sha256"]):
        raise SuiteReceiptError(
            "suite receipt environment_lock_sha256 must be a dependency-lockfile"
            " or environment-export SHA-256 HASH, not a metadata object"
            " (A'-F4); got type"
            f" {type(receipt['environment_lock_sha256']).__name__}"
            " (R-070)"
        )
    for field in (
        "workflow_sha256",
        "junit_sha256",
        "transcript_sha256",
    ):
        if not is_sha256_hex(receipt[field]):
            raise SuiteReceiptError(
                f"suite receipt {field} must be a SHA-256 hash (R-070)"
            )
    interpreter = receipt["interpreter_realpath"]
    if not isinstance(interpreter, str) or not interpreter:
        raise SuiteReceiptError(
            "suite receipt interpreter_realpath must be a non-empty string"
            " (R-070)"
        )
    if not (
        PurePosixPath(interpreter).is_absolute()
        or PureWindowsPath(interpreter).is_absolute()
    ):
        raise SuiteReceiptError(
            "suite receipt interpreter_realpath must be a syntactically"
            " absolute POSIX or Windows path (R-070)"
        )
    for field in ("commit", "tree_sha256"):
        if not is_git_object_id(receipt[field]):
            raise SuiteReceiptError(
                f"suite receipt {field} must be a full-length native Git"
                " object id (40- or 64-hex; R-070)"
            )
    if receipt["dirty"] is not False:
        raise SuiteReceiptError(
            "suite receipt dirty state must be identically false (R-070/R-082)"
        )
    command = receipt["command"]
    if not isinstance(command, list) or not command or not all(
        isinstance(part, str) for part in command
    ):
        raise SuiteReceiptError(
            "suite receipt command must be the exact argv list, not a shell"
            " string (R-070)"
        )
    if not is_real_int(receipt["exit_code"]):
        raise SuiteReceiptError(
            "suite receipt exit_code must be a real integer (R-070)"
        )
    counts = receipt["counts"]
    count_keys = {"tests", "failures", "errors", "skipped"}
    if (
        not isinstance(counts, dict)
        or set(counts) != count_keys
        or any(
            not is_real_int(counts[key]) or counts[key] < 0
            for key in count_keys
        )
        or counts["tests"] <= counts["skipped"]
        or counts["failures"] != 0
        or counts["errors"] != 0
        or receipt["exit_code"] != 0
    ):
        raise SuiteReceiptError(
            "suite receipt counts/exit must describe a nonempty green suite"
            " with exact tests/failures/errors/skipped fields (R-070)"
        )
    if not isinstance(receipt["skip_identities"], list):
        raise SuiteReceiptError(
            "suite receipt skip_identities must be a list (R-070)"
        )
    if any(
        not isinstance(identity, str) or not identity
        for identity in receipt["skip_identities"]
    ):
        raise SuiteReceiptError(
            "suite receipt skip identities must be non-empty strings (R-070)"
        )
    if len(receipt["skip_identities"]) != counts["skipped"] or len(
        set(receipt["skip_identities"])
    ) != len(receipt["skip_identities"]):
        raise SuiteReceiptError(
            "suite receipt skip identities must be duplicate-free and match"
            " counts.skipped (R-070)"
        )
