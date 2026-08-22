"""The ONE named legacy loader (R-060/R-014; OQ-V2-002).

Historical v1/legacy artifacts enter the namespace ONLY through
``load_legacy_v1_document`` and remain historical/noncertifying unless
independently migrated AND regenerated. Strict v2 loaders never import this
module; nothing here feeds a certification leg.

Enumerated legacy families (defined against the repo's actual historical
artifact formats, captured bytes): the ``paper_exports/csli.json``,
``paper_exports/calibration.json``, and ``paper_exports/audit_card.json``
aggregate families, plus v1-versioned flat profile documents.
Spec: .correctless/specs/camera-ready-aims-evidence-2.md
"""
from __future__ import annotations

import json
from typing import Any

from . import schema

LEGACY_FAMILIES = ("csli", "calibration", "audit_card", "v1_profile")

# MA2-003: per-family certify table. ONLY the three captured paper_exports
# aggregate families may back an aggregate claim row. A v1-versioned flat
# profile is a strict-schema PREDECESSOR, not a captured aggregate export —
# it parses here (tolerated, classified) but certifies NOTHING; a recorded
# PASS on a row backed by it recomputes to UNVERIFIED.
AGGREGATE_CERTIFYING_FAMILIES = frozenset({"csli", "calibration", "audit_card"})


def classify_legacy_family(obj: dict[str, Any]) -> str | None:
    """Match one well-formed JSON object against the enumerated legacy
    families; ``None`` for an unknown family (the VERIFIER tolerates unknown
    OBJECTS as historical sidecars per R-064 — the LOADER refuses them)."""
    if "panel_csli" in obj:
        return "csli"
    if "per_bucket" in obj and "max_ece" in obj:
        return "calibration"
    if "metrics" in obj and "overall_verdict" in obj:
        return "audit_card"
    version = obj.get("schema_version")
    if type(version) is int and version == 1 and "profile_id" in obj:
        return "v1_profile"
    return None


def load_legacy_v1_document(data: bytes) -> dict[str, Any]:
    """Parse a known legacy/historical document from captured bytes (R-060).

    Output is barred from certification legs: ``certifying`` is always
    ``False``; ``aggregate_only`` marks the aggregate families. Malformed
    JSON, invalid UTF-8, non-finite tokens, and overlong integer tokens are
    typed ingress errors even inside a known family (R-064/R-067).
    """
    try:
        obj = schema.parse_json_bytes_strict(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise schema.TypedIngressError(
            f"legacy artifact bytes are not valid JSON: {exc} (R-020)"
        ) from exc
    if not isinstance(obj, dict):
        raise schema.TypedIngressError(
            "legacy artifact must be a JSON object (R-064)"
        )
    family = classify_legacy_family(obj)
    if family is None:
        raise schema.SchemaValidationError(
            "unknown legacy artifact family — not one of the enumerated"
            " csli/calibration/audit_card/v1-profile historical formats"
            " (R-014/R-060)"
        )
    if family in ("csli", "calibration", "audit_card") and "metadata" not in obj:
        raise schema.SchemaValidationError(
            f"legacy {family} artifact is missing the named invariant"
            " 'metadata' (generation/provenance block) — refusal requires a"
            " demonstrably missing named invariant (R-014)"
        )
    return {
        "legacy_family": family,
        "aggregate_only": True,
        "certifying": False,
        "payload": obj,
    }


def legacy_certifies(legacy: dict[str, Any], claim_kind: str) -> bool:
    """Whether a legacy artifact can back a claim kind (R-014/MA2-003).

    Per-family privilege: only the three captured paper_exports aggregate
    families (csli, calibration, audit_card) may back an ``aggregate`` row;
    aggregate-only files never certify per-item paired claims; the
    ``v1_profile`` family (and any other family) certifies nothing.
    """
    family = legacy.get("legacy_family")
    if family not in AGGREGATE_CERTIFYING_FAMILIES:
        return False
    if legacy.get("aggregate_only", True):
        return claim_kind == "aggregate"
    return True
