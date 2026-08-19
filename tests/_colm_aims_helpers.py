"""Shared helpers + fixture builders for the colm_aims_2026 RED test suite.

Everything here is synthetic and tiny: opaque item keys (``itm-0001``), no raw
quizbowl text, no network, no absolute paths inside artifacts.

The package builder pins the on-disk contract of the tiny evidence package
that the two-mode verifier consumes (spec R-012/R-013/R-019/R-021). Layout:

    pkg/
      tree/                        <- the verified artifact tree
        profile.json
        records.jsonl
        presentation_manifest.json
      expectations.json            <- OUTSIDE the tree (R-013)
      ledger.json
      rights.json
      receipts/                    <- receipt output dir (outside tree, R-036)

DECISION: the independently anchored expectations file carries (a) an anchor
block (reviewed source commit + frozen ledger hash), (b) tree byte hashes,
(c) one ``bindings`` sub-block per R-012 release leg. Leg ids in verifier
reports must contain their binding key name so tests can match them without
over-pinning exact id spellings.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures" / "colm_aims"
LEGACY_DIR = FIXTURES_DIR / "legacy"

# R-026: sentinel planted in restricted-content positions of adversarial
# fixtures; must NEVER appear in any CLI output, error message, or receipt.
SENTINEL = "RESTRICTED_SENTINEL_DO_NOT_EMIT"

# R-023: manuscript identity — submission PDF SHA-256.
# Source: handoff_prompt_camera_ready_2026-08-18.md (full value of the spec's
# abbreviated "6de23119…dabf10a").
MANUSCRIPT_PDF_SHA256 = (
    "6de23119df59679befc356e3c916bc5a498b2cc2015b6cd8a516a5181dabf10a"
)

# R-001: the eight pinned semantic fields, verbatim from handoff §8 / spec.
SEMANTIC_BLOCK: dict[str, Any] = {
    "trajectory_source": "constructed_reference",
    "observed_open_ended": False,
    "observed_open_ended_answers": False,
    "observed_open_ended_stopping_actions": False,
    "pairing_unit": "matched_item_prefix_grid",
    "pairing_is_observed_sessions": False,
    "supports": "reference_sensitivity_diagnostic",
    "does_not_support": "actual_decision_preservation_or_format_effect",
}

# DECISION: strict profile identifier + reserved future observed identifier
# (R-002). The constructed-reference validator must never accept the latter.
STRICT_PROFILE_ID = "colm_aims_constructed_reference_v1"
OBSERVED_PROFILE_ID = "colm_aims_observed_paired_v1"

# DECISION (R-037): pinned exit codes.
EXIT_PASS = 0
EXIT_GATE_FAIL = 1
EXIT_USAGE_ERROR = 2
EXIT_INGRESS_ERROR = 3

# DECISION (R-017): pinned verdict tokens.
VERDICT_SOURCE_PASS = "PASS_SOURCE_ONLY"
VERDICT_RELEASE_PASS = "PASS_RELEASE"
VERDICT_FAIL = "FAIL"

REMEDIATION_CLASSES = {
    "ARTIFACT_DEFECT",
    "MISSING_EXPECTATION",
    "AUTHOR_DECISION_REQUIRED",
    "EXTERNAL",
}

TRAJECTORY_HORIZON = 6

FAKE_SHA_A = "a" * 64
FAKE_SHA_B = "b" * 64
FAKE_SHA_C = "c" * 64
FAKE_COMMIT = "f" * 40


@pytest.fixture(autouse=True)
def colm_no_network(monkeypatch: pytest.MonkeyPatch):
    """R-028 primary gate: no-network guard, scoped to colm_aims test modules.

    Import this fixture into each tests/test_colm_aims_*.py module
    (``from tests._colm_aims_helpers import colm_no_network``); pytest
    registers imported autouse fixtures per-module, so the other ~1938 tests
    in the flat suite are untouched. AF_UNIX (local IPC) stays allowed.
    """
    real_connect = socket.socket.connect

    def guarded_connect(self, address):  # type: ignore[no-untyped-def]
        if self.family == socket.AF_UNIX:  # local IPC is not network
            return real_connect(self, address)
        raise RuntimeError(
            "network disabled in colm_aims tests (R-028 no-network guard)"
        )

    def guarded_create_connection(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError(
            "network disabled in colm_aims tests (R-028 no-network guard)"
        )

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket, "create_connection", guarded_create_connection)
    yield


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def keyset_sha256(keys: list[str]) -> str:
    """Pinned key-set hash: sha256 over sorted keys joined by newlines.

    Same convention as the historical scripts/stopdff_v5/bootstrap.py
    item-id list hash, so a third party can re-derive it.
    """
    return sha256_bytes("\n".join(sorted(keys)).encode("utf-8"))


def expected_item_key(source_text: str) -> str:
    """DECISION (R-008): pinned item-key derivation.

    itm-<first 16 hex of sha256(NFC-normalized text, utf-8)>. Byte-exact
    comparison after derivation; NFC normalization makes Unicode
    normalization-variant near-duplicates collide (and thus fail closed as
    duplicates).
    """
    normalized = unicodedata.normalize("NFC", source_text)
    return "itm-" + hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def expected_estimand_digest(estimand: dict[str, Any]) -> str:
    """DECISION (R-011): pinned digest = sha256 of canonical compact JSON.

    json.dumps(estimand, sort_keys=True, separators=(",", ":")) encoded utf-8.
    """
    payload = json.dumps(estimand, sort_keys=True, separators=(",", ":"))
    return sha256_bytes(payload.encode("utf-8"))


def tree_hashes(root: Path) -> dict[str, str]:
    """Byte hash of every file under root, keyed by posix relpath (R-014)."""
    out: dict[str, str] = {}
    for path in sorted(Path(root).rglob("*")):
        if path.is_file():
            out[path.relative_to(root).as_posix()] = sha256_file(path)
    return out


def expected_tree_sha256(root: Path) -> str:
    """DECISION (R-036): pinned input-tree digest = sha256 over utf-8 lines
    ``<posix relpath>:<file sha256>`` sorted by relpath, newline-joined with
    a trailing newline. Computed test-side so the receipt value is anchored
    to an independent digest, not to whatever the implementation emits."""
    lines = [f"{rel}:{sha}" for rel, sha in sorted(tree_hashes(root).items())]
    return sha256_bytes(("\n".join(lines) + "\n").encode("utf-8"))


def expected_code_sha256() -> str:
    """DECISION (R-036): pinned verifier-code digest = the expected_tree_sha256
    line scheme over the namespace's ``*.py`` files (relpath within
    reproducibility/colm_aims_2026/, sorted)."""
    namespace = REPO_ROOT / "reproducibility" / "colm_aims_2026"
    lines = [
        f"{p.relative_to(namespace).as_posix()}:{sha256_file(p)}"
        for p in sorted(namespace.glob("**/*.py"))
    ]
    return sha256_bytes(("\n".join(lines) + "\n").encode("utf-8"))


def repo_head_commit() -> str:
    """Real HEAD commit when available (subprocess CLI runs happen from the
    repo root, where the verifier's optional object-existence check can see
    the repository), else a fixed fake 40-hex commit."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        commit = res.stdout.strip()
        if res.returncode == 0 and len(commit) == 40:
            return commit
    except OSError:
        pass
    return FAKE_COMMIT


def load_vocabulary() -> dict[str, Any]:
    return json.loads((FIXTURES_DIR / "vocabulary.json").read_text("utf-8"))


def load_parity_golden() -> dict[str, Any]:
    return json.loads((FIXTURES_DIR / "parity_golden.json").read_text("utf-8"))


# ---------------------------------------------------------------------------
# Record + profile builders
# ---------------------------------------------------------------------------


def make_record(
    item_key: str,
    mc_stop_step: Any,
    ref_stop_step: Any,
    *,
    trajectory_horizon: int = TRAJECTORY_HORIZON,
    **extra: Any,
) -> dict[str, Any]:
    rec: dict[str, Any] = {
        "item_key": item_key,
        "trajectory_horizon": trajectory_horizon,
        "mc_stop_step": mc_stop_step,
        "ref_stop_step": ref_stop_step,
    }
    rec.update(extra)
    return rec


def standard_records() -> list[dict[str, Any]]:
    """Six complete pairs, horizon 6: 3 both-finite, 1 mc-finite/ref-timeout,
    1 mc-timeout/ref-finite, 1 both-timeout. Timeout encoding: stop_step ==
    horizon (zero-indexed rule: stop >= horizon is timeout, R-007).

    The three both-finite stops match tests/fixtures/colm_aims/
    parity_golden.json ``package_cell_3`` so the recorded timing summary is
    the historical estimator's output.
    """
    return [
        make_record("itm-0001", 1, 3),
        make_record("itm-0002", 2, 2),
        make_record("itm-0003", 5, 1),
        make_record("itm-0004", 2, 6),
        make_record("itm-0005", 6, 3),
        make_record("itm-0006", 6, 6),
    ]


# Hand-computed count block for standard_records() — the independent oracle
# for R-005 (never derived by calling the implementation under test).
STANDARD_COUNTS: dict[str, Any] = {
    "n_both_finite": 3,
    "n_mc_finite_ref_timeout": 1,
    "n_mc_timeout_ref_finite": 1,
    "n_both_timeout": 1,
    "n_complete": 6,
    "n_excluded_or_unpaired": 0,
    "exclusion_reason_counts": {},
    "n_pairing_population": 6,
    "n_mc_timeout": 2,
    "n_ref_timeout": 2,
}

STANDARD_RATES: dict[str, Any] = {
    "rate_both_finite": 0.5,
    "rate_mc_finite_ref_timeout": 1.0 / 6.0,
    "rate_mc_timeout_ref_finite": 1.0 / 6.0,
    "rate_both_timeout": 1.0 / 6.0,
}

# Historical-estimator values for the 3 both-finite items (shifts -2, 0, +4).
# Source: tests/fixtures/colm_aims/parity_golden.json (package_cell_3),
# generated by scripts/stopdff_v5/bootstrap.py::cell_bootstrap_stats.
STANDARD_TIMING_SUMMARY: dict[str, Any] = {
    "conditional_on": "n_both_finite",
    "estimand": "signed_index_shift_mc_minus_ref",
    "n": 3,
    "signed_index_mean": 0.6666666666666666,
    "signed_index_median": 0.0,
    "absolute_index_mean": 2.0,
    "absolute_index_median": 2.0,
}

STANDARD_INTERVAL: dict[str, Any] = {
    "procedure": "percentile_bootstrap",
    "draw_count": 100,
    "resampling_seeds": [1],
    "statistic": "signed_index_mean",
    "ci": [-1.6833333333333331, 4.0],
}

# R-006: retained sentinel-coded historical summary — separately named,
# never pooled. Timeouts coded as horizon; over all 6 items the signed
# shifts are (-2, 0, 4, -4, 3, 0).
STANDARD_SENTINEL_CODED_SUMMARY: dict[str, Any] = {
    "convention": "timeout_coded_as_horizon",
    "n": 6,
    "signed_index_mean": 0.16666666666666666,
    "signed_index_median": 0.0,
}


def make_arm(arm_id: str, *, cardinality: str = "k_way", construction: str = "mc_grid") -> dict[str, Any]:
    """One arm identity block (R-003)."""
    return {
        "arm_id": arm_id,
        "construction": construction,
        "cardinality": cardinality,  # "scalar" | "k_way"
        "selector": "argmax_calibrated_score",
        "scorer": "tiny-scorer",
        "candidate_pool_role": "distractor_pool" if cardinality == "k_way" else "none",
        "correctness_assignment": (
            "oracle_gold" if construction == "idealized" else "option_match"
        ),
        "calibration_role": "calibrated",
        "continuation_role": "dp_continuation",
        "seed_contract": {"seeds": [1, 2, 3]},
        "reporting_eligibility": "headline_eligible",
    }


def make_idealized_arm(arm_id: str = "arm-ref") -> dict[str, Any]:
    """The idealized arm: scalar prefix-to-gold cosine, oracle correctness."""
    arm = make_arm(arm_id, cardinality="scalar", construction="idealized")
    arm["scorer"] = "prefix_to_gold_cosine"
    arm["selector"] = "threshold_on_scalar"
    return arm


def make_estimand(**overrides: Any) -> dict[str, Any]:
    est: dict[str, Any] = {
        "arm_mc": "arm-mc",
        "arm_ref": "arm-ref",
        "pairing_definition": "matched_item_prefix_grid",
        "timeout_parameters": {
            "trajectory_horizon": TRAJECTORY_HORIZON,
            "rule": "zero_indexed_stop_ge_horizon_is_timeout",
        },
        "denominator_policy": "n_complete",
        "calibration_identity": "cal-0001",
        "continuation_identity": "cont-0001",
        "random_k_draw_id": "draw-none",
        "numerical_tolerance": 1e-9,
    }
    est.update(overrides)
    return est


def make_cell(records: list[dict[str, Any]] | None = None, **overrides: Any) -> dict[str, Any]:
    if records is None:
        records = standard_records()
    keys = [r["item_key"] for r in records]
    estimand = overrides.pop("estimand", make_estimand())
    cell: dict[str, Any] = {
        "cell_id": "cell-0001",
        "estimand": estimand,
        "estimand_digest": expected_estimand_digest(estimand),
        "counts": dict(STANDARD_COUNTS),
        "rates": dict(STANDARD_RATES),
        "timing_summary_finite_only": dict(STANDARD_TIMING_SUMMARY),
        "timing_summary_sentinel_coded_historical": dict(
            STANDARD_SENTINEL_CODED_SUMMARY
        ),
        "interval": dict(STANDARD_INTERVAL),
        "complete_pair_keys": sorted(keys),
        "excluded_keys": [],
        "pairing_population_keyset_sha256": keyset_sha256(keys),
    }
    cell.update(overrides)
    return cell


def make_llm_involvement(**overrides: Any) -> dict[str, Any]:
    block: dict[str, Any] = {
        "reference_construction": "none",
        "data_plot_creation": "none",
        "evaluation": "none",
    }
    block.update(overrides)
    return block


def make_provenance(
    records_sha: str = FAKE_SHA_A,
    *,
    source_commit: str | None = None,
    eval_keys: list[str] | None = None,
) -> dict[str, Any]:
    if eval_keys is None:
        eval_keys = [r["item_key"] for r in standard_records()]
    fit_keys = [f"fit-{i:04d}" for i in range(1, 7)]
    return {
        "producer_entrypoint": "scripts/fake_producer.py",
        "producer_sha256": FAKE_SHA_B,
        "helper_sha256s": {"scripts/fake_helper.py": FAKE_SHA_C},
        "semantic_command": ["python", "scripts/fake_producer.py", "--seed", "1"],
        "seeds": [1, 2, 3],
        "dirty_state": {
            "git_dirty": False,
            "source_commit": source_commit or repo_head_commit(),
        },
        "splits": {
            "fit": {
                "name": "fit-v1",
                "count": len(fit_keys),
                "keyset_sha256": keyset_sha256(fit_keys),
            },
            "eval": {
                "name": "eval-v1",
                "count": len(eval_keys),
                "keyset_sha256": keyset_sha256(eval_keys),
            },
            "zero_overlap": True,
        },
        "calibration_identity": "cal-0001",
        "continuation_identity": "cont-0001",
        "input_sha256": {"records.jsonl": records_sha},
        "split_metadata_sha256": FAKE_SHA_C,
        "mc_build": {
            "built_after_split": True,
            "coverage_rate": 1.0,
            "retention_policy": "retain_all",
            "retained_count": len(eval_keys),
        },
        "model": {
            "repository_namespace": "example-org/tiny-scorer",
            "revision": "1234567890abcdef1234567890abcdef12345678",
            "weights_sha256": FAKE_SHA_A,
            "tokenizer_config_sha256": FAKE_SHA_B,
            "dtype": "float32",
            "device_class": "cpu",
            "numerical_settings": {"deterministic": True},
        },
        "runtime_packages": {"python": "3.12", "numpy": "2.4.6"},
    }


def make_profile(
    records: list[dict[str, Any]] | None = None,
    *,
    records_sha: str = FAKE_SHA_A,
    source_commit: str | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    """A complete, internally consistent strict profile (R-001)."""
    if records is None:
        records = standard_records()
    profile: dict[str, Any] = {
        "schema_version": 1,
        "profile_id": STRICT_PROFILE_ID,
        "semantic": dict(SEMANTIC_BLOCK),
        "llm_involvement": make_llm_involvement(),
        "numerical_tolerance": 1e-9,
        "item_key_derivation": {
            "hash": "sha256",
            "text_normalization": "NFC",
            "prefix": "itm-",
            "hex_digits": 16,
        },
        "arms": [make_arm("arm-mc"), make_idealized_arm("arm-ref")],
        "provenance": make_provenance(
            records_sha,
            source_commit=source_commit,
            eval_keys=[r["item_key"] for r in records],
        ),
        "cells": [make_cell(records)],
    }
    profile.update(overrides)
    return profile


# ---------------------------------------------------------------------------
# Package builder (R-012/R-013/R-019/R-021 tiny evidence package)
# ---------------------------------------------------------------------------


@dataclass
class Package:
    root: Path
    tree: Path
    profile_path: Path
    records_path: Path
    manifest_path: Path
    expectations_path: Path
    ledger_path: Path
    rights_path: Path
    receipts_dir: Path
    profile: dict[str, Any]
    ledger: dict[str, Any]
    rights: dict[str, Any]
    manifest: dict[str, Any]
    expectations: dict[str, Any]
    source_commit: str


def _dump(obj: Any) -> bytes:
    return (json.dumps(obj, indent=2, sort_keys=True) + "\n").encode("utf-8")


def make_ledger_row(**overrides: Any) -> dict[str, Any]:
    # QA-004 (fix round 1): rows carry an explicit closed-enum claim_kind
    # discriminant; the recompute gate reads it instead of free-text
    # estimand string-matching.
    row: dict[str, Any] = {
        "claim_id": "clm-0001",
        "claim_kind": "per_item_paired",
        "manuscript_location": "Section 4.1, paragraph 2",
        "manuscript_wording": (
            "Constructed QA reference sensitivity diagnostic over matched"
            " item-prefix grids."
        ),
        "estimand": "signed_index_shift_mc_minus_ref",
        "allowed_scope": "reference_sensitivity_diagnostic",
        "producer_entrypoint": "scripts/fake_producer.py",
        "dependency_closure": ["scripts/fake_producer.py", "scripts/fake_helper.py"],
        "input_identity": FAKE_SHA_A,
        "split_identity": "eval-v1",
        "model_identity": "example-org/tiny-scorer@1234567890abcdef1234567890abcdef12345678",
        "calibration_identity": "cal-0001",
        "artifact_id": "profile.json",
        "renderer_id": "colm_aims_render_v1",
        "verifier_oracle": "reproducibility/colm_aims_2026/verifier.py",
        "rights_status": "VERIFIED_ALLOWED",
        "status": "PASS",
        "blocking_task": None,
        "provenance_class": "current_source",
        "artifact_family": "constructed_reference_profile",
        "headline_eligible": True,
    }
    row.update(overrides)
    return row


def make_external_row(**overrides: Any) -> dict[str, Any]:
    row = make_ledger_row(
        claim_id="clm-ext-0001",
        claim_kind="external_fact",
        manuscript_location="Title page",
        manuscript_wording="Manuscript identity (submission PDF).",
        estimand="manuscript_identity",
        allowed_scope="manuscript_identity",
        producer_entrypoint="external:manuscript",
        dependency_closure=[],
        artifact_id="manuscript-pdf",
        artifact_family="manuscript",
        verifier_oracle="human",
        status="EXTERNAL",
        provenance_class="manuscript_identity",
        headline_eligible=False,
    )
    row.update(overrides)
    return row


def make_ledger(*, source_commit: str, rows: list[dict[str, Any]] | None = None, **overrides: Any) -> dict[str, Any]:
    ledger: dict[str, Any] = {
        "schema_version": 1,
        "ledger_id": "colm-aims-2026-ledger",
        "anchored_source_commit": source_commit,
        "manuscript": {"submission_pdf_sha256": MANUSCRIPT_PDF_SHA256},
        "documents": [
            {
                "path": "reproducibility/source_to_claim.md",
                "provenance_class": "historical_submission_artifact",
            }
        ],
        "canonical_run_id": "run-0001",
        "rows": rows if rows is not None else [make_ledger_row(), make_external_row()],
    }
    ledger.update(overrides)
    return ledger


def make_rights(paths: list[str] | None = None, **overrides: Any) -> dict[str, Any]:
    if paths is None:
        paths = ["profile.json", "records.jsonl", "presentation_manifest.json"]
    rights: dict[str, Any] = {
        "schema_version": 1,
        "paths": [
            {
                "path": p,
                "status": "VERIFIED_ALLOWED",
                "upstream_terms_basis": "synthetic test fixture generated in-repo",
            }
            for p in paths
        ],
    }
    rights.update(overrides)
    return rights


def make_manifest(**overrides: Any) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "artifacts": [
            {"path": "profile.json", "role": "strict_profile"},
            {"path": "records.jsonl", "role": "per_item_records"},
        ],
        "allowlist_undeclared": [],
    }
    manifest.update(overrides)
    return manifest


def build_package(
    base: Path,
    *,
    records: list[dict[str, Any]] | None = None,
    profile_mutator: Callable[[dict[str, Any]], None] | None = None,
    binding_mutator: Callable[[dict[str, Any]], None] | None = None,
    ledger_mutator: Callable[[dict[str, Any]], None] | None = None,
    rights_mutator: Callable[[dict[str, Any]], None] | None = None,
    manifest_mutator: Callable[[dict[str, Any]], None] | None = None,
    expectations_mutator: Callable[[dict[str, Any]], None] | None = None,
    extra_tree_files: dict[str, bytes] | None = None,
    source_commit: str | None = None,
    include_sentinel_payload: bool = True,
) -> Package:
    """Build the tiny evidence package.

    Pre-hash mutators (profile/ledger/rights/manifest) keep expectations
    consistent with the mutated bytes so a semantic defect is the ONLY
    defect; ``expectations_mutator`` runs last, post-hash, to deliberately
    break a binding (R-021 verdict-flip mutations).

    ``include_sentinel_payload`` (default True) plants ``sealed-notes.bin``
    — a declared, rights-covered tree file whose CONTENT is the R-026
    sentinel — so EVERY end-to-end verifier run over the package exercises
    the content-leak surface: an implementation that echoes file contents
    trips the sentinel-free assertions (audit ADV-2).
    """
    if records is None:
        records = standard_records()
    commit = source_commit or repo_head_commit()

    root = Path(base) / "pkg"
    tree = root / "tree"
    receipts = root / "receipts"
    tree.mkdir(parents=True, exist_ok=True)
    receipts.mkdir(parents=True, exist_ok=True)

    records_path = tree / "records.jsonl"
    records_bytes = (
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in records)
    ).encode("utf-8")
    records_path.write_bytes(records_bytes)
    records_sha = sha256_bytes(records_bytes)

    profile = make_profile(records, records_sha=records_sha, source_commit=commit)
    if profile_mutator is not None:
        profile_mutator(profile)
    if binding_mutator is not None:
        # QA-001 (fix round 1): artifact-side binding mutation hook — runs
        # pre-hash, so the expectations mirror the mutated provenance and
        # ONLY a value-admissibility predicate can catch the defect.
        binding_mutator(profile["provenance"])
    profile_path = tree / "profile.json"
    profile_path.write_bytes(_dump(profile))

    manifest = make_manifest()
    rights = make_rights()
    if include_sentinel_payload:
        (tree / "sealed-notes.bin").write_bytes((SENTINEL + "\n").encode("utf-8"))
        manifest["artifacts"].append(
            {"path": "sealed-notes.bin", "role": "sealed_payload"}
        )
        rights["paths"].append(
            {
                "path": "sealed-notes.bin",
                "status": "VERIFIED_ALLOWED",
                "upstream_terms_basis": (
                    "synthetic sentinel canary — content must never be emitted"
                ),
            }
        )
    if manifest_mutator is not None:
        manifest_mutator(manifest)
    manifest_path = tree / "presentation_manifest.json"
    manifest_path.write_bytes(_dump(manifest))

    if extra_tree_files:
        for rel, data in extra_tree_files.items():
            p = tree / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)

    if rights_mutator is not None:
        rights_mutator(rights)
    rights_path = root / "rights.json"
    rights_path.write_bytes(_dump(rights))

    ledger = make_ledger(source_commit=commit)
    if ledger_mutator is not None:
        ledger_mutator(ledger)
    ledger_path = root / "ledger.json"
    ledger_path.write_bytes(_dump(ledger))

    prov = profile["provenance"]
    expectations: dict[str, Any] = {
        "schema_version": 1,
        "anchor": {
            "source_commit": commit,
            "ledger_path": "ledger.json",
            "ledger_sha256": sha256_file(ledger_path),
            # QA-005 (fix round 1): the EXTERNAL predicate lives in the
            # independently anchored expectations file, not in row fields the
            # ledger editor can flip in the same document.
            "external_claim_ids": sorted(
                r["claim_id"]
                for r in ledger.get("rows", [])
                if isinstance(r, dict) and r.get("status") == "EXTERNAL"
            ),
        },
        "rights_inventory": {
            "path": "rights.json",
            "sha256": sha256_file(rights_path),
        },
        "tree_files": {
            rel: sha for rel, sha in sorted(tree_hashes(tree).items())
        },
        "bindings": {
            "schema_profile": {
                "profile_id": profile.get("profile_id"),
                "schema_version": profile.get("schema_version"),
                "profile_sha256": sha256_file(profile_path),
            },
            "producer": {
                "entrypoint": prov["producer_entrypoint"],
                "sha256": prov["producer_sha256"],
                "helper_sha256s": dict(prov["helper_sha256s"]),
            },
            "semantic_command": list(prov["semantic_command"]),
            "seeds": list(prov["seeds"]),
            "dirty_state": dict(prov["dirty_state"]),
            "splits": json.loads(json.dumps(prov["splits"])),
            "calibration_identity": prov["calibration_identity"],
            "continuation_identity": prov["continuation_identity"],
            "input_hashes": dict(prov["input_sha256"]),
            "split_metadata_sha256": prov["split_metadata_sha256"],
            "mc_build": dict(prov["mc_build"]),
            "model": json.loads(json.dumps(prov["model"])),
            "runtime_packages": dict(prov["runtime_packages"]),
        },
    }
    if expectations_mutator is not None:
        expectations_mutator(expectations)
    expectations_path = root / "expectations.json"
    expectations_path.write_bytes(_dump(expectations))

    return Package(
        root=root,
        tree=tree,
        profile_path=profile_path,
        records_path=records_path,
        manifest_path=manifest_path,
        expectations_path=expectations_path,
        ledger_path=ledger_path,
        rights_path=rights_path,
        receipts_dir=receipts,
        profile=profile,
        ledger=ledger,
        rights=rights,
        manifest=manifest,
        expectations=expectations,
        source_commit=commit,
    )


def rewrite_json(path: Path, mutate: Callable[[dict[str, Any]], None]) -> None:
    obj = json.loads(Path(path).read_text("utf-8"))
    mutate(obj)
    Path(path).write_bytes(_dump(obj))


def wipe(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)


# ---------------------------------------------------------------------------
# CLI runner (R-021/R-037: documented invocation, via subprocess, repo root)
# ---------------------------------------------------------------------------

CLI_MODULE = "reproducibility.colm_aims_2026.verify"

# QA-011 (fix round 1): env-triggered socket guard for subprocess CLI runs —
# the shim's sitecustomize.py installs the R-028 no-network guard in every
# child interpreter when COLM_AIMS_TEST_NO_NET=1, so the primary gate covers
# the ~16 subprocess CLI runs, not just the parent pytest process.
NO_NET_SHIM_DIR = FIXTURES_DIR / "no_net_shim"


def cli_subprocess_env() -> dict[str, str]:
    env = dict(os.environ)
    env["COLM_AIMS_TEST_NO_NET"] = "1"
    # Keep children from writing __pycache__ bytecode into the fixtures tree
    # (the shim lives under tests/fixtures/, which the tiny-and-synthetic
    # scan sweeps byte-for-byte).
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        f"{NO_NET_SHIM_DIR}{os.pathsep}{existing}"
        if existing
        else str(NO_NET_SHIM_DIR)
    )
    return env


def run_cli(*args: str, cwd: Path = REPO_ROOT) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", CLI_MODULE, *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
        env=cli_subprocess_env(),
    )


def cli_args_for(pkg: Package, mode: str) -> list[str]:
    args = [
        "--mode",
        mode,
        "--tree",
        str(pkg.tree),
        "--receipts-dir",
        str(pkg.receipts_dir),
    ]
    if mode == "release":
        args += ["--expectations", str(pkg.expectations_path)]
    return args
