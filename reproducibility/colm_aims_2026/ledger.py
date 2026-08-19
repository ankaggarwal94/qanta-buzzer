"""Claim ledger + rights inventory validation.

Spec rules owned here: R-023..R-026, R-030.
Spec: .correctless/specs/camera-ready-aims-evidence.md
"""
from __future__ import annotations

import re
from typing import Any

from .schema import ColmAimsError


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

REQUIRED_ROW_FIELDS = (
    "claim_id",
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
    "renderer_id",
    "verifier_oracle",
    "rights_status",
    "status",
    "blocking_task",
    "provenance_class",
)

REQUIRED_LEDGER_FIELDS = (
    "schema_version",
    "ledger_id",
    "anchored_source_commit",
    "manuscript",
    "documents",
    "rows",
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
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


def _validate_row(row: Any, index: int) -> None:
    if not isinstance(row, dict):
        raise LedgerValidationError(f"rows[{index}] must be an object (R-023)")
    label = row.get("claim_id", f"rows[{index}]")
    for field in REQUIRED_ROW_FIELDS:
        if field not in row:
            raise LedgerValidationError(
                f"ledger row {label!r} missing required field {field!r}"
                " (R-023)"
            )
    status = row["status"]
    if status not in LEDGER_STATUSES:
        raise LedgerValidationError(
            f"ledger row {label!r} status {status!r} outside the closed enum"
            f" {sorted(LEDGER_STATUSES)} (R-023)"
        )
    if row["provenance_class"] not in PROVENANCE_CLASSES:
        raise LedgerValidationError(
            f"ledger row {label!r} provenance_class"
            f" {row['provenance_class']!r} outside the closed set"
            f" {sorted(PROVENANCE_CLASSES)} (R-023)"
        )
    if row["rights_status"] not in RIGHTS_STATUSES:
        raise LedgerValidationError(
            f"ledger row {label!r} rights_status {row['rights_status']!r}"
            f" outside the enum {sorted(RIGHTS_STATUSES)} (R-026)"
        )

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
        attribution = row.get("human_attribution")
        if not isinstance(attribution, dict) or not attribution.get(
            "attributed_to"
        ) or not attribution.get("date"):
            raise LedgerValidationError(
                f"ledger row {label!r} moves an EXTERNAL item to PASS without"
                " a human_attribution field (attributed_to + date) (R-024)"
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


def validate_ledger(ledger: dict[str, Any]) -> None:
    """Validate the full claim ledger document; raise on any defect."""
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
    if not isinstance(pdf_sha, str) or not _SHA256_RE.fullmatch(pdf_sha):
        raise LedgerValidationError(
            f"ledger manuscript submission_pdf_sha256 {pdf_sha!r} is not a"
            " SHA-256 hex digest (R-023)"
        )
    commit = ledger["anchored_source_commit"]
    if not isinstance(commit, str) or not _COMMIT_RE.fullmatch(commit):
        raise LedgerValidationError(
            f"ledger anchored_source_commit {commit!r} must be a full-length"
            " commit SHA (R-013)"
        )
    rows = ledger["rows"]
    if not isinstance(rows, list):
        raise LedgerValidationError("ledger rows must be a list (R-023)")
    for index, row in enumerate(rows):
        _validate_row(row, index)

    # R-030: Available-grade assertions require a DOI-class identifier.
    assertion = ledger.get("availability_assertion")
    if assertion is not None:
        if not isinstance(assertion, dict):
            raise LedgerValidationError(
                "availability_assertion must be an object (R-030)"
            )
        grade = assertion.get("grade")
        if isinstance(grade, str) and "available" in grade.lower():
            doi = assertion.get("archival_doi")
            if not isinstance(doi, str) or not _DOI_RE.fullmatch(doi):
                raise LedgerValidationError(
                    f"availability_assertion grade {grade!r} requires a"
                    f" DOI-class archival identifier; {doi!r} does not"
                    " qualify (a GitHub URL is not archival, ACM v1.1)"
                    " (R-030)"
                )


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
