"""Claim ledger + rights inventory validation (v2).

Spec rules owned here: R-023..R-026, R-030, R-056 (ledger-side rejected-ID
storage), D3 blocking-task discipline, R-025 Random-K joint gates.
Spec: .correctless/specs/camera-ready-aims-evidence-2.md
"""
from __future__ import annotations

import re
from typing import Any

from .schema import (
    ANALYSIS_PROVENANCE_D7B,
    ColmAimsError,
    check_schema_version,
    is_commit_sha,
    is_sha256_hex,
)


class LedgerValidationError(ColmAimsError):
    """Claim-ledger row/field/status violation (R-023..R-025, R-030)."""


class RightsError(ColmAimsError):
    """Rights inventory violation (R-026)."""


# R-023: closed status enum.
LEDGER_STATUSES = frozenset({"PASS", "FAIL", "UNVERIFIED", "EXTERNAL"})

# R-023: the ledger distinguishes manuscript identity, historical submission
# artifacts, historical Random-K/v5 results, current source, and future
# clean evidence.
PROVENANCE_CLASSES = frozenset(
    {
        "manuscript_identity",
        "historical_submission_artifact",
        "historical_randomk_v5",
        "current_source",
        "future_evidence",
    }
)

# R-026: the rights inventory enum, exactly four values.
RIGHTS_STATUSES = frozenset(
    {
        "VERIFIED_ALLOWED",
        "VERIFIED_RESTRICTED",
        "UNVERIFIED",
        "AUTHOR_DECISION_REQUIRED",
    }
)

# R-025: exactly two sanctioned Random-K dispositions.
RANDOM_K_DISPOSITIONS = frozenset(
    {"historical_nonconfirmatory", "predeclared_multidraw_family"}
)

# R-014/R-023: closed claim-kind enum — the discriminant the recompute gate
# reads; free-text estimand string-matching is banned for gate decisions.
CLAIM_KINDS = frozenset(
    {"aggregate", "per_item_paired", "venue_rule", "external_fact"}
)

# R-023: artifact_family is a validated closed discriminant too.
ARTIFACT_FAMILIES = frozenset(
    {
        "constructed_reference_profile",
        "manuscript",
        "venue_rule",
        "random_k",
        "inference_block",
        "pr41_hazard_report",
        "legacy_aggregate",
    }
)

REQUIRED_ROW_FIELDS = (
    "claim_id",
    "claim_kind",
    "manuscript_location",
    "manuscript_wording",
    "estimand",
    "allowed_scope",
    "producer_entrypoint",
    "dependency_closure",
    "input_identity",
    "split_identity",
    "model_identity",
    "calibration_identity",
    "artifact_id",
    "artifact_family",
    "renderer_id",
    "verifier_oracle",
    "rights_status",
    "status",
    "blocking_task",
    "provenance_class",
    "headline_eligible",
)

REQUIRED_LEDGER_FIELDS = (
    "schema_version",
    "ledger_id",
    "anchored_source_commit",
    "manuscript",
    "documents",
    "rows",
)

LEDGER_TOP_LEVEL_KEYS = frozenset(
    set(REQUIRED_LEDGER_FIELDS) | {"canonical_run_id", "availability_assertions"}
)

# R-025: the full predeclared multi-draw protocol block (discriminated,
# INACTIVE contract — metadata/validation support only).
MULTIDRAW_PROTOCOL_FIELDS = (
    "frozen_protocol",
    "seed_draw_registry",
    "no_selective_omission",
    "aggregation_rule",
    "sensitivity_analysis",
    "multiplicity_treatment",
)

_ROW_ENUM_FIELDS = (
    ("status", LEDGER_STATUSES, "R-023"),
    ("provenance_class", PROVENANCE_CLASSES, "R-023"),
    ("rights_status", RIGHTS_STATUSES, "R-026"),
    ("claim_kind", CLAIM_KINDS, "R-014"),
    ("artifact_family", ARTIFACT_FAMILIES, "R-023"),
)

# R-030: DOI-class archival identifiers — bare DOI or a doi.org URL. A
# GitHub (or any other) URL does not qualify (ACM v1.1).
_DOI_RE = re.compile(r"^(?:https?://(?:dx\.)?doi\.org/)?10\.\d{4,9}/\S+$")


def _row_is_external_typed(row: dict[str, Any]) -> bool:
    """A row owned by a human/external process, immune to repo tooling
    (R-024)."""
    producer = row.get("producer_entrypoint")
    external_producer = isinstance(producer, str) and producer.startswith(
        "external:"
    )
    return row.get("verifier_oracle") == "human" or external_producer


_EXTERNAL_KIND_CLAIM_KINDS = frozenset({"venue_rule", "external_fact"})


def _row_is_external_kind(row: dict[str, Any]) -> bool:
    return (
        _row_is_external_typed(row)
        or row.get("provenance_class") == "manuscript_identity"
        or row.get("claim_kind") in _EXTERNAL_KIND_CLAIM_KINDS
    )


def _has_human_attribution(row: dict[str, Any]) -> bool:
    attribution = row.get("human_attribution")
    return (
        isinstance(attribution, dict)
        and bool(attribution.get("attributed_to"))
        and bool(attribution.get("as_of"))
    )


def _validate_random_k_row(row: dict[str, Any], label: str) -> None:
    """R-025: the Random-K binding discipline (F1 joint validation)."""
    decision = row.get("author_decision")
    if not isinstance(decision, str) or decision not in RANDOM_K_DISPOSITIONS:
        raise LedgerValidationError(
            f"ledger row {label!r} author_decision {decision!r} is not one of"
            f" the two sanctioned dispositions"
            f" {sorted(RANDOM_K_DISPOSITIONS)} (R-025)"
        )
    if not isinstance(row.get("rng_pinned"), bool):
        raise LedgerValidationError(
            f"ledger row {label!r} must record rng_pinned explicitly as a"
            " boolean (rng_pinned = false is recorded, never implied)"
            " (R-025)"
        )
    for field in ("archived_draw_id", "fresh_draw_id"):
        value = row.get(field)
        if not isinstance(value, str) or not value:
            raise LedgerValidationError(
                f"ledger row {label!r} must bind {field!r} — the archived AND"
                " fresh-run draw identities are both bound (R-025)"
            )
    marker = row.get("disclosure_marker")
    if not isinstance(marker, str) or not marker:
        raise LedgerValidationError(
            f"ledger row {label!r} must retain its disclosure marker"
            " (dagger disclosure) (R-025)"
        )
    headline = row.get("headline_eligible")
    if headline:
        # F1 consistency: the PAIR is validated jointly — headline
        # eligibility is consistent ONLY with the predeclared multi-draw
        # family disposition.
        if decision != "predeclared_multidraw_family":
            raise LedgerValidationError(
                f"ledger row {label!r} is headline-eligible with"
                f" author_decision {decision!r} — headline_eligible: true"
                " requires exactly 'predeclared_multidraw_family' (R-025)"
            )
        protocol = row.get("multidraw_protocol")
        if not isinstance(protocol, dict):
            raise LedgerValidationError(
                f"ledger row {label!r} declares the predeclared multi-draw"
                " family without its full protocol block (R-025)"
            )
        for field in MULTIDRAW_PROTOCOL_FIELDS:
            if field not in protocol:
                raise LedgerValidationError(
                    f"ledger row {label!r} multidraw_protocol missing"
                    f" required field {field!r} — the discriminated contract"
                    " requires frozen protocol, complete seed/draw registry,"
                    " no selective omission, aggregation rule, sensitivity"
                    " analysis, and multiplicity treatment (R-025)"
                )
    elif decision == "predeclared_multidraw_family":
        # Structurally valid but INACTIVE for this paper — nothing further.
        pass


def _validate_inference_row(row: dict[str, Any], label: str) -> None:
    """R-023/R-056: the Holm/inference row names the exact rejected cell IDs
    and carries the new-analysis discriminator."""
    rejected = row.get("rejected_cell_ids")
    if not isinstance(rejected, list) or not all(
        isinstance(v, str) for v in rejected
    ):
        raise LedgerValidationError(
            f"ledger row {label!r} is an inference-block row without the"
            " exact rejected_cell_ids list from the Holm family"
            " (R-023/R-056)"
        )
    if row.get("analysis_provenance") != ANALYSIS_PROVENANCE_D7B:
        raise LedgerValidationError(
            f"ledger row {label!r} inference-block row must carry the"
            f" {ANALYSIS_PROVENANCE_D7B!r} new-analysis discriminator —"
            " D7(b) outputs never claim to recover the historical inference"
            " (R-057)"
        )


def _validate_row(
    row: Any, index: int, external_claim_ids: list[str] | None = None
) -> None:
    if not isinstance(row, dict):
        raise LedgerValidationError(f"rows[{index}] must be an object (R-023)")
    label = row.get("claim_id", f"rows[{index}]")
    for field in REQUIRED_ROW_FIELDS:
        if field not in row:
            raise LedgerValidationError(
                f"ledger row {label!r} missing required field {field!r}"
                " (R-023)"
            )
    for field, allowed, rule in _ROW_ENUM_FIELDS:
        # MA2-002a: guard the type BEFORE frozenset membership — an
        # unhashable recorded value is the same closed-enum violation,
        # never a TypeError crash.
        value = row[field]
        if not isinstance(value, str) or value not in allowed:
            raise LedgerValidationError(
                f"ledger row {label!r} {field} {value!r} outside the"
                f" closed enum {sorted(allowed)} ({rule})"
            )
    if not isinstance(row["headline_eligible"], bool):
        raise LedgerValidationError(
            f"ledger row {label!r} headline_eligible must be a boolean"
            " discriminant (R-025)"
        )
    status = row["status"]

    # D3 (R-023): blocking_task names a genuine actionable remaining blocker.
    blocking_task = row["blocking_task"]
    if status == "UNVERIFIED":
        if not isinstance(blocking_task, str) or not blocking_task.strip():
            raise LedgerValidationError(
                f"ledger row {label!r} is UNVERIFIED without a genuine"
                " actionable blocking_task — a genuinely-open row must name"
                " a real remaining task (R-023/D3)"
            )
    elif status in ("PASS", "EXTERNAL"):
        if blocking_task is not None:
            raise LedgerValidationError(
                f"ledger row {label!r} with status {status!r} carries a"
                " blocking_task — resolved rows never restate resolved"
                " decisions as blockers (R-023/D3)"
            )

    # R-023: PR #41 hazard reports may appear only if the exact manuscript
    # cites them.
    if row.get("artifact_family") == "pr41_hazard_report":
        if row.get("manuscript_cites") is not True:
            raise LedgerValidationError(
                f"ledger row {label!r} references a PR #41 hazard report the"
                " exact manuscript does not cite (R-023)"
            )

    # R-024: EXTERNAL -> PASS requires a human-attribution field; repository
    # green tests never substitute for an EXTERNAL item.
    if _row_is_external_typed(row) and status == "PASS":
        if not _has_human_attribution(row):
            raise LedgerValidationError(
                f"ledger row {label!r} moves an EXTERNAL item to PASS without"
                " a human_attribution field (attributed_to + as_of) (R-024)"
            )

    anchored_external = (
        external_claim_ids is not None
        and row.get("claim_id") in external_claim_ids
    )
    if (anchored_external or _row_is_external_kind(row)) and (
        status != "EXTERNAL"
    ):
        if not _has_human_attribution(row):
            raise LedgerValidationError(
                f"ledger row {label!r} is an EXTERNAL-kind claim recorded"
                " with a non-EXTERNAL status and no human_attribution —"
                " laundering an EXTERNAL row fails ledger validation (R-024)"
            )

    # R-024: venue-rule rows record only officially published facts with
    # source and as-of date.
    if row.get("claim_kind") == "venue_rule":
        if not row.get("source") or not row.get("as_of"):
            raise LedgerValidationError(
                f"ledger row {label!r} is a venue-rule row without a"
                " published source and as_of date (R-024)"
            )

    if row.get("artifact_family") == "random_k":
        _validate_random_k_row(row, label)
    if row.get("artifact_family") == "inference_block":
        _validate_inference_row(row, label)


def _validate_availability_assertion(assertion: Any, index: int) -> None:
    """R-030: an Available-grade assertion needs a DOI-class identifier."""
    if not isinstance(assertion, dict):
        raise LedgerValidationError(
            f"availability_assertions[{index}] must be an object (R-030)"
        )
    grade = assertion.get("grade")
    if not isinstance(grade, str) or "available" not in grade.lower():
        return
    identifier = assertion.get("archival_identifier")
    if not isinstance(identifier, str) or not _DOI_RE.fullmatch(identifier):
        raise LedgerValidationError(
            f"availability_assertions[{index}] grade {grade!r} requires a"
            f" DOI-class archival identifier; {identifier!r} does not"
            " qualify (a GitHub URL is not archival, ACM v1.1) (R-030)"
        )


def validate_ledger(
    ledger: dict[str, Any],
    *,
    external_claim_ids: list[str] | None = None,
) -> None:
    """Validate the full claim ledger document; raise on any defect.

    Validation order (R-059): container shape → schema_version via the ONE
    shared bool-safe checker → all other key and semantic checks.
    """
    if not isinstance(ledger, dict):
        raise LedgerValidationError("ledger must be an object (R-023)")
    check_schema_version(ledger, "ledger")
    for field in REQUIRED_LEDGER_FIELDS:
        if field not in ledger:
            raise LedgerValidationError(
                f"ledger missing required field {field!r} (R-023)"
            )
    unknown = sorted(set(ledger) - LEDGER_TOP_LEVEL_KEYS)
    if unknown:
        raise LedgerValidationError(
            f"ledger carries unknown top-level field(s) {unknown} (R-023)"
        )
    manuscript = ledger["manuscript"]
    if not isinstance(manuscript, dict):
        raise LedgerValidationError(
            "ledger manuscript block must be an object (R-023)"
        )
    for field in ("main_tex_sha256", "main_pdf_sha256"):
        if not is_sha256_hex(manuscript.get(field)):
            raise LedgerValidationError(
                f"ledger manuscript block must pin {field!r} as a SHA-256"
                " digest — the ledger pins the D6 manuscript identity"
                " (R-023)"
            )
    commit = ledger["anchored_source_commit"]
    if not is_commit_sha(commit):
        raise LedgerValidationError(
            f"ledger anchored_source_commit {commit!r} must be a full-length"
            " commit SHA (R-023/R-065)"
        )
    if not isinstance(ledger["documents"], list):
        raise LedgerValidationError("ledger documents must be a list (R-023)")
    rows = ledger["rows"]
    if not isinstance(rows, list):
        raise LedgerValidationError("ledger rows must be a list (R-023)")
    for index, row in enumerate(rows):
        _validate_row(row, index, external_claim_ids)

    assertions = ledger.get("availability_assertions")
    if assertions is not None:
        if not isinstance(assertions, list):
            raise LedgerValidationError(
                "ledger availability_assertions must be a list (R-030)"
            )
        for index, assertion in enumerate(assertions):
            _validate_availability_assertion(assertion, index)


def validate_rights_inventory(rights: dict[str, Any]) -> None:
    """Validate the rights inventory document (enum + terms basis, R-026).

    Duplicate ``path`` rows fail closed — a later row must never shadow an
    earlier (e.g. RESTRICTED-then-ALLOWED) row for the same file.
    """
    if not isinstance(rights, dict):
        raise RightsError("rights inventory must be an object (R-026)")
    check_schema_version(rights, "rights inventory")
    paths = rights.get("paths")
    if not isinstance(paths, list):
        raise RightsError("rights inventory missing paths list (R-026)")
    seen_paths: set[str] = set()
    for index, row in enumerate(paths):
        if not isinstance(row, dict) or not isinstance(row.get("path"), str):
            raise RightsError(
                f"rights inventory row {index} missing an opaque path key"
                " (R-026)"
            )
        label = row["path"]
        if label in seen_paths:
            raise RightsError(
                f"rights inventory carries duplicate rows for path"
                f" {label!r} — duplicate path rows fail closed (R-026)"
            )
        seen_paths.add(label)
        status = row.get("status")
        if not isinstance(status, str) or status not in RIGHTS_STATUSES:
            raise RightsError(
                f"rights inventory row {label!r} status {status!r} outside"
                f" the enum {sorted(RIGHTS_STATUSES)} (R-026)"
            )
        basis = row.get("upstream_terms_basis")
        if not isinstance(basis, str) or not basis.strip():
            raise RightsError(
                f"rights inventory row {label!r} must name its upstream"
                " terms basis rather than a bare enum tick (R-026)"
            )


def check_rights_release(
    rights: dict[str, Any], included_paths: list[str]
) -> None:
    """Release gate: every included path VERIFIED_ALLOWED and inventoried."""
    validate_rights_inventory(rights)
    status_by_path = {row["path"]: row["status"] for row in rights["paths"]}
    for path in included_paths:
        if path not in status_by_path:
            raise RightsError(
                f"included path {path!r} is not inventoried in the rights"
                " inventory — release fails on any uninventoried included"
                " path (R-026)"
            )
        status = status_by_path[path]
        if status != "VERIFIED_ALLOWED":
            raise RightsError(
                f"included path {path!r} carries rights status {status!r};"
                " release mode requires every included path"
                " VERIFIED_ALLOWED (R-026)"
            )
