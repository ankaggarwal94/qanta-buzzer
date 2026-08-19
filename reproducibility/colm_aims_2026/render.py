"""Rendered human-readable summaries for verifier reports.

Spec rules owned here: R-017 (source-mode summary content), R-027
(vocabulary discipline over every renderer output), R-033 (validated
artifact enumeration in PASS-class summaries).
Spec: .correctless/specs/camera-ready-aims-evidence.md
"""
from __future__ import annotations

import json
from typing import Any

# QA-014: the rendered "checks performed" list is DERIVED from the legs
# actually present in the report (plus the emitted receipt) — a hardcoded
# constant could claim checks that never ran. Order: (check name, predicate
# over the leg-id list).
_CHECK_DERIVATIONS: tuple[tuple[str, Any], ...] = (
    (
        "typed ingress (R-020)",
        lambda ids: "typed_ingress" in ids,
    ),
    (
        "profile validation (R-001..R-011)",
        lambda ids: "profile_validation" in ids,
    ),
    (
        "pair/censoring identity recomputation (R-005..R-010)",
        lambda ids: any(
            i.startswith(("cell_", "records_")) for i in ids
        ),
    ),
    (
        "anchored expectation bindings (R-012/R-013)",
        lambda ids: any(
            i.startswith(("binding_", "anchor", "tree_files")) for i in ids
        ),
    ),
    (
        "rights inventory (R-026)",
        lambda ids: any(i.startswith("rights") for i in ids),
    ),
    (
        "presentation manifest reconciliation (R-035)",
        lambda ids: any(
            i.startswith(("manifest", "presentation_manifest")) for i in ids
        ),
    ),
    (
        "claim-ledger status recomputation (R-012)",
        lambda ids: any(i.startswith("ledger") for i in ids),
    ),
)


def _derived_checks(report: Any, legs: list[dict[str, Any]]) -> list[str]:
    ids = [str(leg.get("leg_id", "")) for leg in legs]
    checks = [name for name, present in _CHECK_DERIVATIONS if present(ids)]
    if getattr(report, "receipt_path", None) is not None:
        checks.append("receipt emission (R-036)")
    return checks


def render_summary(report: Any) -> str:
    """Render one verifier report to its human-readable summary text.

    Vocabulary discipline (R-027): every line stays within the enumerated
    fixture vocabulary — the constructed QA reference qualification is
    always present, and no line asserts observed decision outcomes.
    """
    mode = getattr(report, "mode", "source")
    verdict = getattr(report, "verdict", "FAIL")
    legs = list(getattr(report, "legs", []) or [])
    validated = list(getattr(report, "validated_artifacts", []) or [])
    classifications = dict(getattr(report, "classifications", {}) or {})

    lines: list[str] = [
        "COLM AIMS 2026 evidence verifier — constructed QA reference"
        " sensitivity diagnostic",
        f"mode: {mode}",
        f"verdict: {verdict}",
        "scope: constructed QA reference sensitivity only (insensitivity to"
        " sub-threshold shifts within the constructed reference grid); no"
        " observed open-ended stopping policy was measured.",
    ]
    checks = _derived_checks(report, legs)
    if checks:
        lines.append("checks performed: " + "; ".join(checks) + ".")
    if mode == "source":
        lines.append(
            "source-only verification does NOT certify: release bindings,"
            " anchored expectations, rights clearance, archival identity, or"
            " any observed-decision claim. Its ceiling is PASS_SOURCE_ONLY."
        )
    if validated:
        lines.append("validated artifacts:")
        for name in validated:
            lines.append(f"  - {name}")
    if classifications:
        lines.append("artifact classifications:")
        for name in sorted(classifications):
            lines.append(f"  - {name}: {classifications[name]}")
    failing = [leg for leg in legs if leg.get("outcome") == "FAIL"]
    if failing:
        lines.append("failing legs:")
        for leg in failing:
            expected = json.dumps(
                leg.get("expected"), sort_keys=True, default=str
            )
            observed = json.dumps(
                leg.get("observed"), sort_keys=True, default=str
            )
            lines.append(
                f"  - {leg.get('leg_id')}: expected={expected}"
                f" observed={observed}"
                f" remediation={leg.get('remediation_class')}"
            )
    # MA-CC-5: SKIPPED legs (a required capability was unavailable, e.g. no
    # source git for the object-existence check) are surfaced with their
    # reason so the gap is on the record, not silently passed.
    skipped = [leg for leg in legs if leg.get("outcome") == "SKIPPED"]
    if skipped:
        lines.append("skipped legs:")
        for leg in skipped:
            lines.append(
                f"  - {leg.get('leg_id')}: {leg.get('reason')}"
            )
    passed = sum(1 for leg in legs if leg.get("outcome") == "PASS")
    lines.append(
        f"legs: {passed} passed, {len(failing)} failed,"
        f" {len(skipped)} skipped."
    )
    return "\n".join(lines)
