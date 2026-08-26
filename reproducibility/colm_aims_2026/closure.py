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
    ColmAimsError,
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
D6_FINAL_CHECKSUMS_SHA256 = (
    "7d1ee36df4dc884a0cb1dad8ee638d1150c2f2a9f46416adb8b7f81c7b6b7e6d"
)
D6_FINAL_CHECKSUMS_ENTRIES_SHA256 = (
    "a1bfeb89796b933d38c91d12eed3b652397ecd7fd3a7c55ae73a141fb776b455"
)

# Human-designated D6 complete checksum entry map. The map was independently
# read from the designated manuscript bundle on 2026-08-26; external bundle
# bytes remain outside the repository. Keeping the full allowlist here makes
# the closure authority reviewable and clean-checkout reproducible.
D6_FINAL_CHECKSUM_ENTRIES = {
    "ABSTRACT_AND_TLDR.txt": "2b5d808bc73842876c44b0edb248027679c2db613916c6584b8e46c1dc3e1863",
    "CHANGES_VS_ORIGINAL.md": "0d3b5e93865644ee944e1ec23cfc3be14bcf3ef95bd6f42be336d81faf5fb457",
    "MAINTAINER_NOTES.md": "7d892974f8d55771f7aba24f1c5318111fce3d55b670344567df1709d5e733b0",
    "OPENREVIEW_FORM_FIELDS.md": "c7a0eb7096427f122cf36d29e7bd135e8f924f3c888d3ceadc6bdc5495f2dff5",
    "OPENREVIEW_FORM_FIELDS.txt": "6e5296e1f258b0feb1e970ff8bab9dc95fd8239e36ba17300dbf2fd243b8459c",
    "PDF_PREFLIGHT.json": "30be113ce2b1f991294b745d9cb542865edd275c28850c46749fb62738addc83",
    "PROVENANCE_README.md": "740148617949f0c96efec55eacfc41c8843ca86e3f12ef890692e00c2d36bd56",
    "README.md": "435ce0cb06d311e17fec61e79eb81dcfd9da5ac2bba62ef3a3ee7fd25fea5f83",
    "REVISION_NOTES.md": "a745bd863052a93302f557636128bf307a740a575cb74033ceecb830a20dc8d6",
    "SUBMISSION_NOTES.md": "5ae6d7baca5dcb70c4c36748366ae0ec6a051319e29aae13ee0c86433b3e49ec",
    "colm2026_conference.bst": "2d67552db7ed38ccfccb5957b52f95656e25c249724761d3cf5f7922ad1844c5",
    "colm2026_conference.sty": "55962ae80c25a50335825c85d23eb5f1cd9015aa8e77f7af32b483b646c7483e",
    "compile_audit.json": "fcf90487d55ca663cd924f18e14d230ddcd6c058d22dc28219a84dff4dff4820",
    "figures/calibration_ece.pdf": "9c7fd92ca0fbf045d4faee241197bde8f1e7a81de1afb9463745ac81e4178eca",
    "figures/calibration_ece.png": "497f9ab01b9802ff69e3279c429b74406d6ce26e2c7e3108961a594b6ac51e0e",
    "figures/learned_direction.pdf": "d91e132ddeeb018b540cac2f1ecc41c3ea7f04cee49563104fb1f55512769443",
    "figures/learned_direction.png": "0e2fa1e6631d2328197e4d46358eaeb0b5b1659c1915ea79586d65a7d401f5c4",
    "figures/reachable_comparator.tex": "a36a122f242f315404c0412d848425ccaa315ae96048803eeff6cbb0205c995d",
    "figures/reference_sensitivity.pdf": "192435a360630ee8ef72713b7fdecbb3a3658ada0cd59ec21c656fbe1b5a42fd",
    "figures/reference_sensitivity.png": "acb7abc34e9b94bcbfe2bc1dfc40f40374d5b7b31086643a2d7b81c3783e26cc",
    "main.bbl": "115c420ad9e64c4bdd2383789a164b5bb422f0e3fe86d988b15c80fb6727b910",
    "main.pdf": "6de23119df59679befc356e3c916bc5a498b2cc2015b6cd8a516a5181dabf10a",
    "main.tex": "79dccfb3fbdfafbd566a3fb239755ab35142bac510d629d513ed8b3c2c4cdd2f",
    "main_linenumbered.txt": "7f1f24f53c3ea6876ac805d86a8d4d6b993b41e492d3d3d2d9b8aa94aafd8a7c",
    "make_figures.py": "4579890c296d6c5eed8408494b73df70ed652a67a137f6fbd0f0b01bade42f7b",
    "openreview_metadata.txt": "6e5296e1f258b0feb1e970ff8bab9dc95fd8239e36ba17300dbf2fd243b8459c",
    "references.bib": "49341bfa1ad9be3ccb8ec1f8646fc721ada3e64b11c207f5c15a07276673750f",
    "reported_results.json": "d9eb39de1f3239aca8c4e16f1e1de0a3db3f07b4ef1963cb01b618f95136a333",
    "reported_results_provenance.json": "29b414f0b6831d1f4044bc1f1f5b5f5d7bac59845a9f11d5bb7ca28855107390",
    "verify_provenance.py": "ec3e3228507c8e0554a0b53c163e890927fc374b96cc9fa7186bcf93ad747d97",
}
D6_FINAL_CHECKSUMS_BYTES = "".join(
    f"{digest}  ./{path}\n"
    for path, digest in D6_FINAL_CHECKSUM_ENTRIES.items()
).encode("utf-8")


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
_QA012_KEYS = frozenset(
    {"status", "inventory_sha256", "authority_sha256", "manifest"}
)
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
    qa012 = inventory["qa012"]
    if (
        not isinstance(qa012, dict)
        or not {"status", "inventory_sha256"} <= set(qa012)
        or not set(qa012) <= _QA012_KEYS
    ):
        raise SchemaValidationError(
            "closure qa012 must have status/inventory_sha256 and only the"
            " optional authority digest or diagnostic manifest"
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
    authority_sha256 = qa012.get("authority_sha256")
    if authority_sha256 is not None and not is_sha256_hex(authority_sha256):
        raise SchemaValidationError(
            "closure qa012.authority_sha256 must be a SHA-256 value or null"
        )
    if qa012.get("manifest") is not None:
        from . import qa012 as qa012_module

        qa012_module.validate_inventory_manifest(qa012["manifest"])


def evaluate_closure(
    inventory: dict[str, Any],
    *,
    profile_bytes: bytes | None = None,
) -> dict[str, Any]:
    """Evaluate the frozen CAMERA_READY_CLOSURE inventory, fail-closed.

    Returns ``{"gate": CAMERA_READY_CLOSURE, "satisfied": bool,
    "failing_rows": [...]}`` — every unsatisfied duty appears as a failing
    row; the gate is satisfied only when none remain.

    This evidence-level evaluator validates supplied bytes and the pinned
    QA-012 authority; optional caller-root scans are diagnostic only. It does
    not itself recompute record semantics. Only
    ``phase4_assemble_d7b.build_evidence_package`` is an authoritative
    publication path: that producer additionally runs the full source
    verifier over its staged, byte-verified closed envelope before create-once
    promotion. Create-once is a protocol property under the R-081 process/host
    trust boundary, not an OS-level immutability or hostile-process claim.
    """
    validate_closure_inventory(inventory)
    failing: list[str] = []

    bound_profile: dict[str, Any] | None = None
    if profile_bytes is None:
        failing.append(
            "closure evaluation requires the actual bound profile.json bytes"
        )
    else:
        try:
            from . import schema as schema_module

            parsed_profile = schema_module.parse_json_bytes_strict(
                profile_bytes
            )
            if not isinstance(parsed_profile, dict):
                raise SchemaValidationError("profile.json must be an object")
            schema_module.validate_profile(parsed_profile)
            bound_profile = parsed_profile
        except (UnicodeDecodeError, json.JSONDecodeError, ColmAimsError) as exc:
            failing.append(f"bound profile.json is invalid: {exc}")

    if bound_profile is not None:
        observed_profile_sha256 = hashlib.sha256(profile_bytes).hexdigest()
        expected_evidence = f"profile.json sha256:{observed_profile_sha256}"
        table_rows = {
            row.get("item"): row
            for row in inventory.get("rows", [])
            if isinstance(row, dict)
        }
        table_row = table_rows.get("table-1-headline-shifts", {})
        if table_row.get("evidence") != expected_evidence:
            failing.append(
                "table evidence does not bind the supplied profile.json bytes"
            )
        observed_analysis = bound_profile["inference"]["analysis_provenance"]
        if observed_analysis != ANALYSIS_PROVENANCE_D7B:
            failing.append(
                "bound profile does not carry the D7(b) analysis provenance"
            )

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
    elif baseline.get("final_checksums_sha256") != D6_FINAL_CHECKSUMS_SHA256:
        failing.append(
            "D6 FINAL_CHECKSUMS raw bytes differ from the independently"
            " pinned designated-manuscript authority"
        )
    entries = _as_dict(baseline.get("final_checksums_entries"))
    observed_entries_sha256 = checksum_entries_sha256(entries)
    if baseline.get("final_checksums_entries_sha256") != observed_entries_sha256:
        failing.append(
            "FINAL_CHECKSUMS entry-map digest does not match the complete"
            " canonical path-to-digest map"
        )
    if (
        entries != D6_FINAL_CHECKSUM_ENTRIES
        or observed_entries_sha256 != D6_FINAL_CHECKSUMS_ENTRIES_SHA256
    ):
        failing.append(
            "FINAL_CHECKSUMS entry map differs from the independently pinned"
            " designated-manuscript authority"
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
    if (
        holm_row.get("satisfied_by") != ANALYSIS_PROVENANCE_D7B
        or bound_profile is None
        or bound_profile["inference"]["analysis_provenance"]
        != ANALYSIS_PROVENANCE_D7B
    ):
        failing.append(
            "holm/inference row unsatisfied: only the D7(b) regenerated"
            f" outputs ({ANALYSIS_PROVENANCE_D7B!r}) satisfy it; observed"
            f" {holm_row.get('satisfied_by')!r}"
        )

    # QA-012 (R-072) is blocking for closure.
    qa012 = _as_dict(inventory.get("qa012"))
    qa012_status = qa012.get("status")
    qa012_manifest = qa012.get("manifest")
    if qa012_status in _QA012_SATISFIED_STATUSES:
        try:
            from . import qa012 as qa012_module

            authority_sha256 = qa012.get("authority_sha256")
            if (
                authority_sha256 != qa012_module.CANONICAL_AUTHORITY_SHA256
                or qa012.get("inventory_sha256") != authority_sha256
                or qa012_manifest is not None
            ):
                raise SchemaValidationError(
                    "QA-012 satisfying closure must bind only the exact pinned"
                    " rev3 authority SHA-256"
                )
            authority = qa012_module.load_authority_manifest()
            derived_status = (
                "VERIFIED_WITH_FIXTURES"
                if qa012_module.authority_hit_fixtures_verified(authority)
                else "HITS_PRESENT"
            )
            if derived_status != qa012_status:
                raise SchemaValidationError(
                    "QA-012 closure status is not derived from the pinned rev3"
                    " authority and committed fixtures"
                )
        except (KeyError, TypeError, OSError, ColmAimsError) as exc:
            failing.append(f"QA-012 embedded inventory invalid: {exc}")
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
