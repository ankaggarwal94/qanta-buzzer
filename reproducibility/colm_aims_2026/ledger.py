"""Claim ledger + rights inventory validation.

Spec rules owned here: R-023..R-026, R-030.
Spec: .correctless/specs/camera-ready-aims-evidence.md
"""
from __future__ import annotations

import re
from typing import Any

from .schema import ColmAimsError, is_commit_sha, is_sha256_hex


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

# QA-004 (R-014/R-023): closed claim-kind enum — the discriminant the
# recompute gate reads. Free-text estimand string-matching is banned for
# semantic gate decisions; `aggregate` vs `per_item_paired` decides what a
# legacy aggregate-only artifact may certify; `venue_rule`/`external_fact`
# mark EXTERNAL fact rows.
CLAIM_KINDS = frozenset(
    {"aggregate", "per_item_paired", "venue_rule", "external_fact"}
)

# QA-004 class fix: artifact_family is a validated closed discriminant too —
# a row omitting or renaming it is rejected, never routed to a permissive
# branch (the hazard-citation and Random-K gates key off it).
ARTIFACT_FAMILIES = frozenset(
    {
        "constructed_reference_profile",
        "manuscript",
        "venue_rule",
        "random_k",
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

# R-023/R-026 (+ QA-004): the enumerated per-row fields whose values are
# closed sets — every semantic gate decision keys off one of these validated
# discriminants, never off free-text string equality.
_ROW_ENUM_FIELDS = (
    ("status", LEDGER_STATUSES, "R-023"),
    ("provenance_class", PROVENANCE_CLASSES, "R-023"),
    ("rights_status", RIGHTS_STATUSES, "R-026"),
    ("claim_kind", CLAIM_KINDS, "R-014/QA-004"),
    ("artifact_family", ARTIFACT_FAMILIES, "R-023/QA-004"),
)

# R-030: DOI-class archival identifiers — bare DOI or a doi.org URL. A
# GitHub (or any other) URL does not qualify (ACM v1.1).
_DOI_RE = re.compile(r"^(?:https?://(?:dx\.)?doi\.org/)?10\.\d{4,9}/\S+$")


def _row_is_external_typed(row: dict[str, Any]) -> bool:
    """A row owned by a human/external process, immune to repo tooling (R-024)."""
    producer = row.get("producer_entrypoint")
    external_producer = isinstance(producer, str) and producer.startswith(
        "external:"
    )
    return row.get("verifier_oracle") == "human" or external_producer


# QA-015: EXTERNAL-kind markers a laundering edit is unlikely to flip
# consistently — the widened fallback heuristic when no anchored
# external_claim_ids list is supplied.
_EXTERNAL_KIND_CLAIM_KINDS = frozenset({"venue_rule", "external_fact"})


def _row_is_external_kind(row: dict[str, Any]) -> bool:
    """Widened EXTERNAL-kind heuristic (QA-015, R-024): a manuscript-identity
    or venue/external-fact row is external-kind regardless of the flippable
    oracle/producer fields."""
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
        and bool(attribution.get("date"))
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
        if row[field] not in allowed:
            raise LedgerValidationError(
                f"ledger row {label!r} {field} {row[field]!r} outside the"
                f" closed enum {sorted(allowed)} ({rule})"
            )
    if not isinstance(row["headline_eligible"], bool):
        raise LedgerValidationError(
            f"ledger row {label!r} headline_eligible must be a boolean"
            " discriminant (R-025/QA-004)"
        )
    status = row["status"]

    # R-023: PR #41 hazard reports may appear only if the exact manuscript
    # cites them.
    if row.get("artifact_family") == "pr41_hazard_report":
        if row.get("manuscript_cites") is not True:
            raise LedgerValidationError(
                f"ledger row {label!r} references a PR #41 hazard report the"
                " exact manuscript does not cite (manuscript_cites must be"
                " true) (R-023)"
            )

    # R-024: EXTERNAL -> PASS requires a human-attribution field; repository
    # green tests never substitute for an EXTERNAL item.
    if _row_is_external_typed(row) and status == "PASS":
        if not _has_human_attribution(row):
            raise LedgerValidationError(
                f"ledger row {label!r} moves an EXTERNAL item to PASS without"
                " a human_attribution field (attributed_to + date) (R-024)"
            )

    # QA-015 (R-024): the EXTERNAL predicate at the surface the rule names.
    # With an anchored external_claim_ids list supplied, MEMBERSHIP is the
    # predicate — a listed row not recorded EXTERNAL requires human
    # attribution regardless of what its flippable fields now say. Without
    # the list, the widened kind heuristic (manuscript-identity provenance,
    # venue/external-fact claim kinds) fires, so the three-field laundering
    # (status + verifier_oracle + producer_entrypoint) still fails standalone.
    anchored_external = (
        external_claim_ids is not None and row.get("claim_id") in (
            external_claim_ids
        )
    )
    if (anchored_external or _row_is_external_kind(row)) and (
        status != "EXTERNAL"
    ):
        if not _has_human_attribution(row):
            raise LedgerValidationError(
                f"ledger row {label!r} is an EXTERNAL-kind claim"
                f" ({'anchored' if anchored_external else 'kind-typed'})"
                " recorded with a non-EXTERNAL status and no"
                " human_attribution — laundering an EXTERNAL row fails"
                " ledger validation (R-024/QA-015)"
            )

    # R-024: venue-rule rows record only officially published facts with
    # source and as-of date.
    if row.get("claim_kind") == "venue_rule":
        if not row.get("source") or not row.get("as_of_date"):
            raise LedgerValidationError(
                f"ledger row {label!r} is a venue-rule row without a"
                " published source and as_of_date (R-024)"
            )

    # R-025: Random-K disposition gate on headline-eligible rows.
    if row.get("artifact_family") == "random_k" and row.get(
        "headline_eligible"
    ):
        decision = row.get("author_decision")
        if decision is None:
            raise LedgerValidationError(
                f"ledger row {label!r} is a headline-eligible Random-K row"
                " without an explicit author_decision (R-025)"
            )
        if decision not in RANDOM_K_DISPOSITIONS:
            raise LedgerValidationError(
                f"ledger row {label!r} author_decision {decision!r} is not"
                f" one of the two sanctioned dispositions"
                f" {sorted(RANDOM_K_DISPOSITIONS)} (R-025)"
            )


def _validate_availability_assertion(assertion: Any) -> None:
    """R-030: an Available-grade assertion needs a DOI-class identifier."""
    if not isinstance(assertion, dict):
        raise LedgerValidationError(
            "availability_assertion must be an object (R-030)"
        )
    grade = assertion.get("grade")
    if not isinstance(grade, str) or "available" not in grade.lower():
        return
    doi = assertion.get("archival_doi")
    if not isinstance(doi, str) or not _DOI_RE.fullmatch(doi):
        raise LedgerValidationError(
            f"availability_assertion grade {grade!r} requires a DOI-class"
            f" archival identifier; {doi!r} does not qualify (a GitHub URL is"
            " not archival, ACM v1.1) (R-030)"
        )


def validate_ledger(
    ledger: dict[str, Any],
    *,
    external_claim_ids: list[str] | None = None,
) -> None:
    """Validate the full claim ledger document; raise on any defect.

    ``external_claim_ids`` (QA-015): the independently anchored EXTERNAL
    claim-id list from the expectations anchor. When supplied, membership is
    the EXTERNAL predicate for the R-024 laundering gate; when omitted, the
    widened kind heuristic applies.
    """
    if not isinstance(ledger, dict):
        raise LedgerValidationError("ledger must be an object (R-023)")
    for field in REQUIRED_LEDGER_FIELDS:
        if field not in ledger:
            raise LedgerValidationError(
                f"ledger missing required field {field!r} (R-023)"
            )
    manuscript = ledger["manuscript"]
    if not isinstance(manuscript, dict) or "submission_pdf_sha256" not in (
        manuscript
    ):
        raise LedgerValidationError(
            "ledger manuscript block missing submission_pdf_sha256 — the"
            " ledger must pin manuscript identity (R-023)"
        )
    pdf_sha = manuscript["submission_pdf_sha256"]
    if not is_sha256_hex(pdf_sha):
        raise LedgerValidationError(
            f"ledger manuscript submission_pdf_sha256 {pdf_sha!r} is not a"
            " SHA-256 hex digest (R-023)"
        )
    commit = ledger["anchored_source_commit"]
    if not is_commit_sha(commit):
        raise LedgerValidationError(
            f"ledger anchored_source_commit {commit!r} must be a full-length"
            " commit SHA (R-013)"
        )
    rows = ledger["rows"]
    if not isinstance(rows, list):
        raise LedgerValidationError("ledger rows must be a list (R-023)")
    for index, row in enumerate(rows):
        _validate_row(row, index, external_claim_ids)

    # R-030: Available-grade assertions require a DOI-class identifier.
    assertion = ledger.get("availability_assertion")
    if assertion is not None:
        _validate_availability_assertion(assertion)


def validate_rights_inventory(rights: dict[str, Any]) -> None:
    """Validate the rights inventory document (enum + terms basis, R-026)."""
    if not isinstance(rights, dict):
        raise RightsError("rights inventory must be an object (R-026)")
    paths = rights.get("paths")
    if not isinstance(paths, list):
        raise RightsError("rights inventory missing paths list (R-026)")
    for index, row in enumerate(paths):
        if not isinstance(row, dict) or not isinstance(row.get("path"), str):
            raise RightsError(
                f"rights inventory row {index} missing an opaque path key"
                " (R-026)"
            )
        label = row["path"]
        status = row.get("status")
        if status not in RIGHTS_STATUSES:
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
