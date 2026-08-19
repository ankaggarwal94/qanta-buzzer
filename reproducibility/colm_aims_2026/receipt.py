"""Verification receipt emission (schema-versioned, create-once, run-scoped).

Spec rule owned here: R-036.
Spec: .correctless/specs/camera-ready-aims-evidence.md
"""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any

from scripts.stopdff_v5 import fileio

from .schema import ColmAimsError

RECEIPT_SCHEMA_VERSION = 1


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
    if not isinstance(run_id, str) or not run_id or Path(run_id).name != run_id:
        raise ColmAimsError(
            f"receipt run_id {run_id!r} must be a single path component"
            " (R-036)"
        )
    path = receipts_dir / f"receipt-{run_id}.json"
    resolved = path.resolve()
    tree_resolved = verified_tree.resolve()
    if resolved == tree_resolved or tree_resolved in resolved.parents:
        raise ColmAimsError(
            "receipt path must be outside the verified artifact tree (R-036)"
        )
    payload = dict(report_payload)
    payload.setdefault("schema_version", RECEIPT_SCHEMA_VERSION)
    data = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    fileio.create_once_bytes(path, data, exists_label="verification receipt")
    return path
