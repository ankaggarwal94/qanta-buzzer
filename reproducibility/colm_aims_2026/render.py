"""Rendered human-readable summaries for verifier reports.

Spec rules owned here: R-017 (source-mode summary content), R-027
(vocabulary discipline over every renderer output), R-033 (validated
artifact enumeration in PASS-class summaries).
Spec: .correctless/specs/camera-ready-aims-evidence.md
"""
from __future__ import annotations

import json
from typing import Any

_SOURCE_CHECKS = (
    "typed ingress (R-020)",
    "profile validation (R-001..R-011)",
    "pair/censoring identity recomputation (R-005..R-010)",
    "receipt emission (R-036)",
)

_RELEASE_CHECKS = _SOURCE_CHECKS + (
    "anchored expectation bindings (R-012/R-013)",
    "rights inventory (R-026)",
    "presentation manifest reconciliation (R-035)",
    "claim-ledger status recomputation (R-012)",
)


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

    lines: list[str] = []
    lines.append(
        "COLM AIMS 2026 evidence verifier — constructed QA reference"
        " sensitivity diagnostic"
    )
    lines.append(f"mode: {mode}")
    lines.append(f"verdict: {verdict}")
    lines.append(
        "scope: constructed QA reference sensitivity only (insensitivity to"
        " sub-threshold shifts within the constructed reference grid); no"
        " observed open-ended stopping policy was measured."
    )
    checks = _RELEASE_CHECKS if mode == "release" else _SOURCE_CHECKS
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
    passed = sum(1 for leg in legs if leg.get("outcome") == "PASS")
    lines.append(f"legs: {passed} passed, {len(failing)} failed.")
    return "\n".join(lines)
