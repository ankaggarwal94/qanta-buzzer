"""Verification + suite-evidence receipts (schema-versioned, create-once).

Spec rules owned here: R-036 (per-run verification receipts), R-070
(machine-readable suite receipts; A'-F4 defect shape rejected).
Spec: .correctless/specs/camera-ready-aims-evidence-2.md
"""
from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any

from scripts.stopdff_v5 import fileio

from .schema import (
    SCHEMA_VERSION,
    ColmAimsError,
    check_schema_version,
    encode_json,
    is_commit_sha,
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
    "environment_digest",
    "workflow_file_sha256",
    "interpreter_realpath",
    "commit",
    "tree",
    "dirty",
    "command",
    "exit_code",
    "junit_report_sha256",
    "skip_identities",
    "artifact_hashes",
)


class SuiteReceiptError(ColmAimsError):
    """Suite-evidence receipt violates the R-070 binding contract."""


def validate_suite_receipt(receipt: dict[str, Any]) -> None:
    """Validate one machine-readable suite-evidence receipt (R-070).

    Every binding is a HASH or an exact machine-readable value; the A'-F4
    defect shape — ``environment_digest`` as a metadata object instead of a
    lockfile/environment-export hash — is rejected by name.
    """
    if not isinstance(receipt, dict):
        raise SuiteReceiptError("suite receipt must be an object (R-070)")
    if "schema_version" in receipt:
        check_schema_version(receipt, "suite receipt")
    missing = sorted(
        set(SUITE_RECEIPT_REQUIRED_FIELDS) - set(receipt)
    )
    if missing:
        raise SuiteReceiptError(
            f"suite receipt missing required field(s) {missing} (R-070)"
        )
    if not is_sha256_hex(receipt["environment_digest"]):
        raise SuiteReceiptError(
            "suite receipt environment_digest must be a dependency-lockfile"
            " or environment-export SHA-256 HASH, not a metadata object"
            f" (A'-F4); got type {type(receipt['environment_digest']).__name__}"
            " (R-070)"
        )
    for field in ("workflow_file_sha256", "junit_report_sha256"):
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
    for field in ("commit", "tree"):
        if not is_commit_sha(receipt[field]):
            raise SuiteReceiptError(
                f"suite receipt {field} must be a full-length 40-hex object"
                " id (R-070)"
            )
    if not isinstance(receipt["dirty"], bool):
        raise SuiteReceiptError(
            "suite receipt dirty state must be a boolean (R-070)"
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
    if not isinstance(receipt["skip_identities"], list):
        raise SuiteReceiptError(
            "suite receipt skip_identities must be a list (R-070)"
        )
    hashes = receipt["artifact_hashes"]
    if not isinstance(hashes, dict) or not all(
        isinstance(name, str) and name and is_sha256_hex(value)
        for name, value in hashes.items()
    ):
        raise SuiteReceiptError(
            "suite receipt artifact_hashes must map artifact names to"
            " SHA-256 hashes (R-070)"
        )
