"""CAMERA_READY_CLOSURE gate (R-071; D4; handoff L526).

A DISTINCT gate token from ``PASS_RELEASE`` — neither implies the other, and
source mode can emit neither. The expected-claim inventory is FROZEN from
the D6 baseline's COMPLETE checksum closure (never ``main.tex`` alone); the
Holm/inference row is satisfiable ONLY by the D7(b) regenerated outputs;
QA-012 (R-072) is blocking for closure.
Spec: .correctless/specs/camera-ready-aims-evidence-2.md
"""
from __future__ import annotations

from typing import Any

from .schema import (
    ANALYSIS_PROVENANCE_D7B,
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


def _as_dict(value: Any) -> dict[str, Any]:
    """Shape coercion for an inventory-controlled block: ``{}`` for ANY
    non-dict, so a malformed block fails its duties instead of crashing."""
    return value if isinstance(value, dict) else {}


def evaluate_closure(inventory: dict[str, Any]) -> dict[str, Any]:
    """Evaluate the frozen CAMERA_READY_CLOSURE inventory, fail-closed.

    Returns ``{"gate": CAMERA_READY_CLOSURE, "satisfied": bool,
    "failing_rows": [...]}`` — every unsatisfied duty appears as a failing
    row; the gate is satisfied only when none remain.
    """
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
