"""RED suite — documentation contract.

Covers: R-038 (plus the repo-ledger documentation row it requires).
Spec: .correctless/specs/camera-ready-aims-evidence.md
"""
from __future__ import annotations

import json
import re

from tests._colm_aims_helpers import (  # noqa: F401  (colm_no_network is an autouse fixture)
    MANUSCRIPT_PDF_SHA256,
    REPO_ROOT,
    colm_no_network,
)

README_PATH = REPO_ROOT / "reproducibility" / "colm_aims_2026" / "README.md"
SOURCE_TO_CLAIM = REPO_ROOT / "reproducibility" / "source_to_claim.md"
REPO_LEDGER_PATH = REPO_ROOT / "reproducibility" / "colm_aims_2026" / "ledger.json"


def _readme() -> str:
    assert README_PATH.exists(), (
        "reproducibility/colm_aims_2026/README.md must exist (R-038)"
    )
    return README_PATH.read_text("utf-8")


def test_readme_pins_both_modes_exact_invocation():
    # Tests R-038 [integration]: README pins both modes' exact invocation
    # (the documented module-run form with both mode values and all flags).
    text = _readme()
    assert "python -m reproducibility.colm_aims_2026.verify" in text
    assert "--mode source" in text
    assert "--mode release" in text
    assert "--tree" in text
    assert "--expectations" in text
    assert "--receipts-dir" in text


def test_readme_pins_input_layout_including_expectations_location():
    # Tests R-038 [integration]: input layout including the expectations-file
    # location (outside the verified artifact tree).
    text = _readme().lower()
    assert "expectations" in text
    assert "outside" in text
    assert "profile.json" in text
    assert "records.jsonl" in text
    assert "presentation_manifest.json" in text


def test_readme_pins_verdict_semantics_exit_codes_and_receipts():
    # Tests R-038 [integration]: verdict enum semantics, exit codes, and the
    # receipt location are documented.
    text = _readme()
    assert "PASS_SOURCE_ONLY" in text
    assert "PASS_RELEASE" in text
    assert "FAIL" in text
    low = text.lower()
    assert "exit code" in low
    for token in ("0", "1", "2", "3"):
        assert token in text
    assert "usage" in low
    assert "ingress" in low
    assert "receipt" in low


def test_readme_disambiguates_from_legacy_verifier():
    # Tests R-038 [integration]: a one-line disambiguation from
    # scripts/verify_audit_release.py — and no doc redefines the legacy
    # verifier as camera-ready certification.
    text = _readme()
    assert "scripts/verify_audit_release.py" in text
    line = next(
        ln for ln in text.splitlines() if "verify_audit_release.py" in ln
    )
    assert re.search(r"legacy|not|separate|distinct", line, re.IGNORECASE), (
        "disambiguation line must distinguish the legacy verifier"
    )


def test_source_to_claim_gains_historical_scope_header():
    # Tests R-038 [integration]: reproducibility/source_to_claim.md gains a
    # historical-scope header naming the manuscript it maps and pointing to
    # the new ledger.
    head = "\n".join(SOURCE_TO_CLAIM.read_text("utf-8").splitlines()[:40])
    assert re.search(r"historical", head, re.IGNORECASE), (
        "source_to_claim.md must carry a historical-scope header"
    )
    assert "final_project.tex" in head  # names the manuscript it maps
    assert "colm_aims_2026" in head  # points to the new ledger namespace


def test_repo_ledger_records_source_to_claim_as_historical_document():
    # Tests R-038 [integration]: the ledger records source_to_claim.md as a
    # historical-submission-artifact document.
    # DECISION: the feature's real claim ledger lives at
    # reproducibility/colm_aims_2026/ledger.json.
    assert REPO_LEDGER_PATH.exists(), "feature claim ledger must exist"
    doc = json.loads(REPO_LEDGER_PATH.read_text("utf-8"))
    entries = [
        d
        for d in doc.get("documents", [])
        if d.get("path") == "reproducibility/source_to_claim.md"
    ]
    assert entries, "ledger must record reproducibility/source_to_claim.md"
    assert entries[0]["provenance_class"] == "historical_submission_artifact"


def test_repo_ledger_pins_manuscript_identity():
    # Tests R-023/R-038 [integration]: the real ledger distinguishes
    # manuscript identity via the pinned submission PDF SHA-256.
    # Source: handoff_prompt_camera_ready_2026-08-18.md (full SHA of the
    # spec's abbreviated 6de23119…dabf10a).
    assert REPO_LEDGER_PATH.exists(), "feature claim ledger must exist"
    doc = json.loads(REPO_LEDGER_PATH.read_text("utf-8"))
    assert doc["manuscript"]["submission_pdf_sha256"] == MANUSCRIPT_PDF_SHA256
