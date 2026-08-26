"""CAMERA_READY_CLOSURE gate (R-071; D4; handoff L526).

A DISTINCT gate token from ``PASS_RELEASE`` — neither implies the other, and
source mode can emit neither. The expected-claim inventory is FROZEN from
the D6 baseline's COMPLETE checksum closure (never ``main.tex`` alone); the
Holm/inference row is satisfiable ONLY by the D7(b) regenerated outputs;
QA-012 (R-072) is blocking for closure.
Spec: .correctless/specs/camera-ready-aims-evidence-2.md
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

from .schema import (
    ANALYSIS_PROVENANCE_D7B,
    SCHEMA_VERSION,
    SchemaValidationError,
    check_schema_version,
    is_sha256_hex,
)

CAMERA_READY_CLOSURE = "CAMERA_READY_CLOSURE"

# D6 designated manuscript baseline (two-party hash-verified).
D6_MAIN_TEX_SHA256 = (
    "79dccfb3fbdfafbd566a3fb239755ab35142bac510d629d513ed8b3c2c4cdd2f"
)
D6_MAIN_PDF_SHA256 = (
    "6de23119df59679befc356e3c916bc5a498b2cc2015b6cd8a516a5181dabf10a"
)

# Row statuses that do not block closure: SATISFIED rows map to clean bound
# evidence; EXTERNAL rows stay EXTERNAL per R-024 and never block.
_NON_BLOCKING_ROW_STATUSES = frozenset({"SATISFIED", "EXTERNAL"})

# QA-012 statuses that satisfy the R-072 blocking gate (with an inventory
# hash); the fail-closed default state UNVERIFIED blocks.
_QA012_SATISFIED_STATUSES = frozenset(
    {"VERIFIED_VACUOUS", "VERIFIED_WITH_FIXTURES"}
)

_CLOSURE_TOP_LEVEL_KEYS = frozenset(
    {"schema_version", "d6_baseline", "rows", "holm_row", "qa012"}
)
_D6_BASELINE_KEYS = frozenset(
    {
        "main_tex_sha256",
        "main_pdf_sha256",
        "final_checksums_sha256",
        "final_checksums_entries",
        "final_checksums_entries_sha256",
    }
)
_ROW_KEYS = frozenset({"item", "status", "evidence"})
_HOLM_KEYS = frozenset({"satisfied_by"})
_QA012_KEYS = frozenset({"status", "inventory_sha256"})
EXPECTED_CLAIM_ITEMS = frozenset(
    {"table-1-headline-shifts", "manuscript-identity"}
)


def checksum_entries_sha256(entries: dict[str, str]) -> str:
    """Canonical digest of the complete FINAL_CHECKSUMS entry map."""
    payload = json.dumps(
        entries, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _as_dict(value: Any) -> dict[str, Any]:
    """Shape coercion for an inventory-controlled block: ``{}`` for ANY
    non-dict, so a malformed block fails its duties instead of crashing."""
    return value if isinstance(value, dict) else {}


def _require_exact_keys(
    value: Any, expected: frozenset[str], where: str
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SchemaValidationError(f"{where} must be an object")
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected)
    if missing or unknown:
        raise SchemaValidationError(
            f"{where} key set is not closed; missing={missing},"
            f" unknown={unknown}"
        )
    return value


def validate_closure_inventory(inventory: Any) -> None:
    """Validate the closed, versioned closure container before semantics."""
    if not isinstance(inventory, dict):
        raise SchemaValidationError("closure inventory must be an object")
    check_schema_version(inventory, "closure inventory")
    if inventory["schema_version"] != SCHEMA_VERSION:
        raise SchemaValidationError(
            "closure inventory schema version does not match this verifier"
        )
    _require_exact_keys(inventory, _CLOSURE_TOP_LEVEL_KEYS, "closure inventory")
    baseline = _require_exact_keys(
        inventory["d6_baseline"], _D6_BASELINE_KEYS, "closure d6_baseline"
    )
    for field in (
        "main_tex_sha256",
        "main_pdf_sha256",
        "final_checksums_sha256",
        "final_checksums_entries_sha256",
    ):
        if not is_sha256_hex(baseline[field]):
            raise SchemaValidationError(
                f"closure d6_baseline.{field} must be a SHA-256 value"
            )
    entries = baseline["final_checksums_entries"]
    if not isinstance(entries, dict) or not entries:
        raise SchemaValidationError(
            "closure d6_baseline.final_checksums_entries must be a non-empty"
            " path-to-SHA-256 map"
        )
    for rel, digest in entries.items():
        if (
            not isinstance(rel, str)
            or not rel
            or "\\" in rel
            or rel.startswith("/")
            or any(part in ("", ".", "..") for part in rel.split("/"))
            or not is_sha256_hex(digest)
        ):
            raise SchemaValidationError(
                "closure FINAL_CHECKSUMS entries must use safe POSIX-relative"
                " paths and SHA-256 values"
            )
    rows = inventory["rows"]
    if not isinstance(rows, list) or not rows:
        raise SchemaValidationError(
            "closure inventory rows must be a non-empty list"
        )
    seen_items: set[str] = set()
    for index, row_value in enumerate(rows):
        row = _require_exact_keys(
            row_value, _ROW_KEYS, f"closure rows[{index}]"
        )
        item = row["item"]
        if not isinstance(item, str) or not item:
            raise SchemaValidationError(
                f"closure rows[{index}].item must be a non-empty string"
            )
        if item in seen_items:
            raise SchemaValidationError(
                f"closure inventory carries duplicate row item {item!r}"
            )
        seen_items.add(item)
        status = row["status"]
        evidence = row["evidence"]
        if not isinstance(status, str) or not status:
            raise SchemaValidationError(
                f"closure rows[{index}].status must be a non-empty string"
            )
        if evidence is not None and not isinstance(evidence, str):
            raise SchemaValidationError(
                f"closure rows[{index}].evidence must be a string or null"
            )
        if status in _NON_BLOCKING_ROW_STATUSES and not evidence:
            raise SchemaValidationError(
                f"closure rows[{index}] with status {status!r} must bind"
                " non-empty evidence"
            )
    if seen_items != EXPECTED_CLAIM_ITEMS:
        raise SchemaValidationError(
            "closure expected-claim row set is not frozen;"
            f" missing={sorted(EXPECTED_CLAIM_ITEMS - seen_items)},"
            f" unexpected={sorted(seen_items - EXPECTED_CLAIM_ITEMS)}"
        )
    holm_row = _require_exact_keys(
        inventory["holm_row"], _HOLM_KEYS, "closure holm_row"
    )
    satisfied_by = holm_row["satisfied_by"]
    if satisfied_by is not None and (
        not isinstance(satisfied_by, str) or not satisfied_by
    ):
        raise SchemaValidationError(
            "closure holm_row.satisfied_by must be a non-empty string or null"
        )
    qa012 = _require_exact_keys(
        inventory["qa012"], _QA012_KEYS, "closure qa012"
    )
    if not isinstance(qa012["status"], str) or not qa012["status"]:
        raise SchemaValidationError(
            "closure qa012.status must be a non-empty string"
        )
    inventory_sha256 = qa012["inventory_sha256"]
    if inventory_sha256 is not None and not is_sha256_hex(inventory_sha256):
        raise SchemaValidationError(
            "closure qa012.inventory_sha256 must be a SHA-256 value or null"
        )


def evaluate_closure(
    inventory: dict[str, Any],
    *,
    expected_final_checksums_entries_sha256: str | None = None,
) -> dict[str, Any]:
    """Evaluate the frozen CAMERA_READY_CLOSURE inventory, fail-closed.

    Returns ``{"gate": CAMERA_READY_CLOSURE, "satisfied": bool,
    "failing_rows": [...]}`` — every unsatisfied duty appears as a failing
    row; the gate is satisfied only when none remain.
    """
    validate_closure_inventory(inventory)
    failing: list[str] = []

    baseline = _as_dict(inventory.get("d6_baseline"))
    if baseline.get("main_tex_sha256") != D6_MAIN_TEX_SHA256:
        failing.append(
            "d6 baseline main.tex hash drifted from the two-party-verified"
            f" designated baseline {D6_MAIN_TEX_SHA256}"
        )
    if baseline.get("main_pdf_sha256") != D6_MAIN_PDF_SHA256:
        failing.append(
            "d6 baseline main.pdf hash drifted from the two-party-verified"
            f" designated baseline {D6_MAIN_PDF_SHA256}"
        )
    if not is_sha256_hex(baseline.get("final_checksums_sha256")):
        failing.append(
            "d6 baseline missing the FINAL_CHECKSUMS.sha256 manifest hash —"
            " closure binds the COMPLETE checksum closure"
        )
    entries = _as_dict(baseline.get("final_checksums_entries"))
    observed_entries_sha256 = checksum_entries_sha256(entries)
    if baseline.get("final_checksums_entries_sha256") != observed_entries_sha256:
        failing.append(
            "FINAL_CHECKSUMS entry-map digest does not match the complete"
            " canonical path-to-digest map"
        )
    if not is_sha256_hex(expected_final_checksums_entries_sha256):
        failing.append(
            "complete FINAL_CHECKSUMS entry-map authority is absent; closure"
            " requires an independently supplied canonical entry-map digest"
        )
    elif observed_entries_sha256 != expected_final_checksums_entries_sha256:
        failing.append(
            "FINAL_CHECKSUMS entry set or a figure/bibliography digest differs"
            " from the independently supplied complete-manifest authority"
        )
    if entries.get("main.tex") != D6_MAIN_TEX_SHA256:
        failing.append(
            "FINAL_CHECKSUMS entries do not pin main.tex at the designated"
            " baseline hash"
        )
    if entries.get("main.pdf") != D6_MAIN_PDF_SHA256:
        failing.append(
            "FINAL_CHECKSUMS entries do not pin main.pdf — closure duties"
            " bind the COMPLETE manifest (figures, bibliography), never"
            " main.tex alone"
        )

    rows = inventory.get("rows")
    rows = rows if isinstance(rows, list) else []
    if not rows:
        failing.append("closure inventory carries no expected-claim rows")
    for row in rows:
        if not isinstance(row, dict):
            failing.append("malformed closure inventory row")
            continue
        status = row.get("status")
        if status not in _NON_BLOCKING_ROW_STATUSES:
            failing.append(
                f"inventory row {row.get('item')!r} status {status!r}: every"
                " displayed number, count, table, and figure maps to clean"
                " bound evidence or is removed/downgraded (handoff L526)"
            )

    # The Holm/inference row is satisfied ONLY by the D7(b) regenerated
    # outputs — until they exist the gate fails on that row by construction.
    holm_row = _as_dict(inventory.get("holm_row"))
    if holm_row.get("satisfied_by") != ANALYSIS_PROVENANCE_D7B:
        failing.append(
            "holm/inference row unsatisfied: only the D7(b) regenerated"
            f" outputs ({ANALYSIS_PROVENANCE_D7B!r}) satisfy it; observed"
            f" {holm_row.get('satisfied_by')!r}"
        )

    # QA-012 (R-072) is blocking for closure.
    qa012 = _as_dict(inventory.get("qa012"))
    qa012_status = qa012.get("status")
    if qa012_status not in _QA012_SATISFIED_STATUSES or not is_sha256_hex(
        qa012.get("inventory_sha256")
    ):
        failing.append(
            f"QA-012 status {qa012_status!r}: UNVERIFIED is blocking for"
            " final CAMERA_READY_CLOSURE; closure requires the executed"
            " inventory procedure and its manifest hash (R-072)"
        )

    return {
        "gate": CAMERA_READY_CLOSURE,
        "satisfied": not failing,
        "failing_rows": failing,
    }
