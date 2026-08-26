"""Rendered human-readable summaries for verifier reports.

Spec rules owned here: R-017 (source-mode summary content), R-027/R-057
(vocabulary discipline over every renderer output), R-033 (validated
artifact enumeration in PASS-class summaries).

Vocabulary discipline: every line stays inside the enumerated fixture
vocabulary — the constructed QA reference qualification is always present;
no line asserts observed decision outcomes, decision preservation, format
effects, or any recovery of the historical inference; display text never
carries the closure gate token.
Spec: .correctless/specs/camera-ready-aims-evidence-2.md
"""
from __future__ import annotations

import json
from typing import Any

# The rendered "checks performed" list is DERIVED from the legs actually
# present in the report — a hardcoded constant could claim checks that never
# ran. Order: (check name, predicate over the leg-id list).
_CHECK_DERIVATIONS: tuple[tuple[str, Any], ...] = (
    ("typed ingress (R-020)", lambda ids: "typed_ingress" in ids),
    (
        "strict profile validation (R-001..R-011)",
        lambda ids: "profile_validation" in ids,
    ),
    (
        "grid identity checks (R-040..R-043)",
        lambda ids: any(i.startswith("grid_") for i in ids),
    ),
    (
        "event representation checks (R-045..R-047)",
        lambda ids: "event_representation" in ids,
    ),
    (
        "pair/censoring identity recomputation (R-005..R-010)",
        lambda ids: "counts_identities" in ids or "rates" in ids,
    ),
    (
        "estimand label binding (R-048/R-049/R-068)",
        lambda ids: "estimand_label_binding" in ids,
    ),
    (
        "in-package inference recompute (R-050..R-056)",
        lambda ids: any(i.startswith("inference_") for i in ids),
    ),
    (
        "anchored expectation bindings (R-012/R-013/R-044)",
        lambda ids: any(
            i.startswith(("binding_", "anchor", "anchored_", "tree_files"))
            for i in ids
        ),
    ),
    (
        "rights inventory (R-026)",
        lambda ids: any(i.startswith("rights") for i in ids),
    ),
    (
        "presentation manifest reconciliation (R-035)",
        lambda ids: any(i.startswith("manifest") for i in ids),
    ),
    (
        "claim-ledger status recomputation (R-012)",
        lambda ids: any(i.startswith("ledger") for i in ids),
    ),
    (
        "canonical run selection (R-069)",
        lambda ids: "canonical_selection" in ids,
    ),
)


def _derived_checks(report: Any, legs: list[dict[str, Any]]) -> list[str]:
    ids = [str(leg.get("leg_id", "")) for leg in legs]
    checks = [name for name, present in _CHECK_DERIVATIONS if present(ids)]
    if getattr(report, "receipt_path", None) is not None:
        checks.append("receipt emission (R-036)")
    return checks


def render_summary(report: Any) -> str:
    """Render one verifier report to its human-readable summary text."""
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
    failing = [leg for leg in legs if leg.get("status") == "FAIL"]
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
                f" remediation={leg.get('remediation')}"
            )
    passed = sum(1 for leg in legs if leg.get("status") == "PASS")
    lines.append(f"legs: {passed} passed, {len(failing)} failed.")
    return "\n".join(lines)
