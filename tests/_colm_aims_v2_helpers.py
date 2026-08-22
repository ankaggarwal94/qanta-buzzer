"""Shared helpers + fixture builders for the colm_aims_2026 v2 RED suite.

GREENFIELD successor (spec `.correctless/specs/camera-ready-aims-evidence-2.md`,
71 active rules, governed by `contract_freeze_signoff_2026-08-20.md`). The
production namespace `reproducibility/colm_aims_2026/` does NOT exist on this
branch; this module never imports it at module scope, so it is importable at
head. Test modules other than `test_colm_aims_v2_inference_d7b.py` import the
namespace at module scope and are EXPECTED to fail collection until GREEN.

Everything here is synthetic and tiny in *content* (opaque `itm-<hex>` keys,
no raw quizbowl text, no network, no absolute paths inside artifacts) while
FULL-SIZE in *shape*: the frozen v2 contract pins exactly 5 references x 2
calibrations = 10 cells x 2,249 complete pairs (R-040/R-042/R-061), so the
canonical valid package really carries 2,249 records per cell.

On-disk package contract (D1 + sign-off SS2.1; carries the v1 layout):

    pkg/
      tree/                          <- the verified artifact tree
        profile.json                 <- strict v2 profile (grid + inference in-profile)
        records/<cell_id>.jsonl      <- exactly one per declared cell (R-041)
        presentation_manifest.json
        sealed-notes.bin             <- R-026 sentinel canary (content never emitted)
      expectations.json              <- OUTSIDE the tree (R-013)
      ledger.json
      rights.json
      receipts/                      <- receipt output dir (outside tree, R-036)

Runs-site contract for the R-069 release path (canonical selection):

    site/
      runs/<run_id>/tree/...         <- run-scoped published package trees
      ledger.json                    <- carries canonical_run_id pointer
      expectations.json / rights.json / receipts/

DECISION (leg ids): the v2 verifier reports carry EXACT leg ids pinned by the
LEG_* constants below; failing legs carry expected/observed and a remediation
class from REMEDIATION_CLASSES. Sidecar ingress legs are
``sidecar_ingress:<tree-relative path>``; anchored (release) grid/inference
legs are ``anchored_grid_<field>`` / ``anchored_inference_<field>``.

DECISION (exceptions): the v1 exception taxonomy carries, plus
``SchemaVersionError(TypedIngressError)`` as the one bool-safe version-defect
type raised by the single shared checker on EVERY versioned surface (R-059).
"""
from __future__ import annotations

import functools
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
NAMESPACE_DIR = REPO_ROOT / "reproducibility" / "colm_aims_2026"

# ---------------------------------------------------------------------------
# Pinned tokens and constants (the de-facto v2 interface contract)
# ---------------------------------------------------------------------------

# R-026 sentinel planted in restricted-content positions; must NEVER appear in
# any CLI output, error message, or receipt.
SENTINEL = "RESTRICTED_SENTINEL_DO_NOT_EMIT"

# R-058 canonical revision constant set (sign-off SS4.1, exact).
SCHEMA_VERSION = 2
VERIFIER_REVISION = "reproducibility.colm_aims_2026:r2"

# R-001: v2 strict profile id (spec-pinned successor of the v1 id) + the
# reserved observed-study id (R-002, carried unchanged from v1).
STRICT_PROFILE_ID = "colm_aims_constructed_reference_v2"
OBSERVED_PROFILE_ID = "colm_aims_observed_paired_v1"
V1_PROFILE_ID = "colm_aims_constructed_reference_v1"

# R-001: the eight pinned semantic fields, verbatim from handoff SS8 / spec.
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

# R-037 pinned exit codes (v1 QA-019 promoted to contract).
EXIT_PASS = 0
EXIT_GATE_FAIL = 1
EXIT_USAGE_ERROR = 2
EXIT_INGRESS_ERROR = 3
EXIT_INTERNAL_ERROR = 4

# R-017 pinned verdict tokens; R-071 closure token is DISTINCT from both.
VERDICT_SOURCE_PASS = "PASS_SOURCE_ONLY"
VERDICT_RELEASE_PASS = "PASS_RELEASE"
VERDICT_FAIL = "FAIL"
CLOSURE_GATE_TOKEN = "CAMERA_READY_CLOSURE"

REMEDIATION_CLASSES = {
    "ARTIFACT_DEFECT",
    "MISSING_EXPECTATION",
    "AUTHOR_DECISION_REQUIRED",
    "EXTERNAL",
}

# R-045 closed event vocabulary + terminal-imputation enum (DECISION: the
# imputation field is ALWAYS present per arm; FINITE_STOP pairs with "NONE",
# NEVER_STOPPED pairs with "FINAL_PREFIX_IF_NEVER").
EVENT_FINITE = "FINITE_STOP"
EVENT_NEVER = "NEVER_STOPPED"
IMPUTATION_NONE = "NONE"
IMPUTATION_FINAL_PREFIX = "FINAL_PREFIX_IF_NEVER"

# R-046/R-010: the preserved fair-QA producer's derived-scalar convention.
SENTINEL_CONVENTION = "timeout_coded_as_horizon"

# R-047: spec-pinned new exclusion-reason enum member.
AMBIGUOUS_TERMINAL_SENTINEL = "AMBIGUOUS_TERMINAL_SENTINEL"

# R-054 spec-pinned population enum.
POPULATION_ALL = "all_complete_pairs_terminal_imputed"
POPULATION_FINITE = "both_finite_only"

# R-048/R-049 DECISION: closed estimand labels for the two named estimands.
HEADLINE_ESTIMAND_LABEL = (
    "mean_signed_shift_mc_minus_ref_all_complete_pairs_terminal_imputed"
)
FINITE_ONLY_ESTIMAND_LABEL = "mean_signed_shift_mc_minus_ref_both_finite_only"

# R-057: exact analysis-provenance discriminator token (spec-pinned).
ANALYSIS_PROVENANCE_D7B = "d7b_regenerated_2026"

# R-052: recorded seed-derivation string (DECISION: this exact sentence is the
# recorded derivation string; the derived integer is recorded beside it).
SEED_DERIVATION_STRING = (
    'sha256(b"colm_aims_2026/v2/bootstrap_holm\\0"'
    " + bytes.fromhex(pairing_population_keyset_sha256)).digest()[:8]"
    " big-endian unsigned"
)

# Grid identity (sign-off SS2.1 + OQ-V2-001 adopted proposal): producer
# reference spellings; cell_id = "<reference_id>__<calibration_id>".
REFERENCE_IDS = ("idealized", "kdisjoint", "khard", "klex", "krandom")
CALIBRATION_IDS = ("format_specific", "shared")
CELL_IDS = tuple(
    sorted(f"{r}__{c}" for r in REFERENCE_IDS for c in CALIBRATION_IDS)
)
N_ITEMS = 2249
TRAJECTORY_HORIZON = 6

# R-003 DECISION: closed per-family stop-semantics vocabularies (no
# overloaded global stop integer across families).
FAMILY_STOP_VOCAB: dict[str, str] = {
    "constructed_reference": "reference_threshold_crossing",
    "fixed_threshold": "fixed_threshold_crossing",
    "myopic": "myopic_one_step_stop",
    "learned_continuation": "learned_value_stop",
}

# R-071: D6 designated manuscript baseline (two-party hash-verified).
D6_MAIN_TEX_SHA256 = (
    "79dccfb3fbdfafbd566a3fb239755ab35142bac510d629d513ed8b3c2c4cdd2f"
)
D6_MAIN_PDF_SHA256 = (
    "6de23119df59679befc356e3c916bc5a498b2cc2015b6cd8a516a5181dabf10a"
)

# R-023/D3: golden Random-K blocking task (exact spec text).
RANDOM_K_BLOCKING_TASK = (
    "bind archived + fresh-run identities with rng_pinned=false; publish the"
    " first pinned evidence run under the v2 data model"
)

# R-018: scripts/verify_audit_release.py stays byte-identical (sha256 pinned
# at the v2 branch head 689aecf3 at RED authoring time).
VERIFY_AUDIT_RELEASE_SHA256 = (
    "8d4e76c5e183e6efb96844ac13b55dd3fbaa1eab64b9da74fe611466f456513a"
)

# R-027 vocabulary core (v1 fixture carried + R-057 inference additions).
BANNED_PHRASES = (
    "qa effect",
    "preserves the observed",
    "preserves observed",
    "observed stopping policy is preserved",
    "observed decision preservation",
    "decision preservation established",
    "format effect established",
    "would hide real shifts",
    "camera-ready certified",
    "certifies camera-ready",
    # R-057: the D7(b) outputs are a NEW analysis.
    "recovers the historical",
    "recovered the historical",
    "original analysis",
    "authenticates the historical",
)
BANNED_PHRASES_CASE_SENSITIVE = ("Reproduced", "Replicated")
REQUIRED_QUALIFIER = "constructed qa reference"
SANCTIONED_OBSERVED_CLAIM_OUTPUT = (
    "observed_paired_claim=OBSERVED_PAIRED_STUDY_REQUIRED"
)

# ---------------------------------------------------------------------------
# Exact leg ids (DECISION: the v2 leg-id contract)
# ---------------------------------------------------------------------------

LEG_TYPED_INGRESS = "typed_ingress"
LEG_PROFILE_VALIDATION = "profile_validation"
LEG_GRID_COMPLETENESS = "grid_completeness"
LEG_RECORD_FILE_BIJECTION = "grid_record_file_bijection"
LEG_ITEM_KEY_SET = "grid_item_key_set_equality"
LEG_HELD_FIXED = "grid_held_fixed_identities"
LEG_MC_STOP_WITHIN_CAL = "grid_mc_stop_within_calibration"
LEG_EVENT_REPRESENTATION = "event_representation"
LEG_COUNTS = "counts_identities"
LEG_RATES = "rates"
LEG_ESTIMAND_LABELS = "estimand_label_binding"
LEG_CELL_COMPARABILITY = "cell_comparability"
LEG_INFERENCE_SEED = "inference_seed_derivation"
LEG_INFERENCE_MATRIX = "inference_resample_matrix_digest"
LEG_INFERENCE_RECOMPUTE = "inference_recompute"
LEG_INFERENCE_HOLM = "inference_holm_family"
LEG_LEDGER_ANCHOR_EQ = "ledger_anchor_commit_equality"
LEG_GIT_OBJECT = "anchor_source_commit_object"
LEG_ANCHOR_COMMIT = "anchor_source_commit"
LEG_CANONICAL_SELECTION = "canonical_selection"
SIDECAR_LEG_PREFIX = "sidecar_ingress:"
ANCHORED_GRID_PREFIX = "anchored_grid_"
ANCHORED_INFERENCE_PREFIX = "anchored_inference_"

FAKE_SHA_A = "a" * 64
FAKE_SHA_B = "b" * 64
FAKE_SHA_C = "c" * 64
FAKE_COMMIT = "f" * 40

# ---------------------------------------------------------------------------
# Authoring-time golden literals (computed once with the repo venv:
# Python 3.11.15 / NumPy 2.4.6 on little-endian arm64; see the d7b module).
# ---------------------------------------------------------------------------

# sha256("\n".join(sorted canonical item keys)) for the 2,249 synthetic keys.
CANONICAL_KEYSET_SHA256 = (
    "25b43e24896fa321bad445f8ae7d8559d68c4cd9dbd479b8a4e02ffbc9c12f68"
)
CANONICAL_SEED = 3300285925885496919
CANONICAL_MATRIX_SHA256 = (
    "e73836c6bdbb1247b539d350a3af8d96e96c76417a9c395fe31d63881f817abb"
)
# Independent pure fixture for the d7b early-signal tests.
FIXTURE_KEYSET_SHA256 = (
    "101a6c3b9f1602d1215fe02f5173c7dba11519177ca67aaebb0e7744a9ef91da"
)
FIXTURE_SEED = 4049282694496731189
FIXTURE_MATRIX_SHA256 = (
    "544e78a05da629ce8594421ec97bf23e07fe161cc44b82cd35e630db301c650b"
)


# ---------------------------------------------------------------------------
# R-028 no-network guard (autouse; import into every test module)
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def colm_no_network(monkeypatch: pytest.MonkeyPatch):
    """R-028 primary gate, scoped to colm_aims v2 test modules.

    AF_UNIX (local IPC) stays allowed. Subprocess CLI runs get the same guard
    via the sitecustomize shim in ``cli_subprocess_env`` below.
    """
    import socket

    real_connect = socket.socket.connect

    def guarded_connect(self, address):  # type: ignore[no-untyped-def]
        if self.family == socket.AF_UNIX:
            return real_connect(self, address)
        raise RuntimeError(
            "network disabled in colm_aims v2 tests (R-028 no-network guard)"
        )

    def guarded_create_connection(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError(
            "network disabled in colm_aims v2 tests (R-028 no-network guard)"
        )

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket, "create_connection", guarded_create_connection)
    yield


# ---------------------------------------------------------------------------
# Hash utilities (independent test-side digest conventions, carried from v1)
# ---------------------------------------------------------------------------


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def keyset_sha256(keys: list[str]) -> str:
    """Pinned key-set hash: sha256 over SORTED keys joined by newlines."""
    return sha256_bytes("\n".join(sorted(keys)).encode("utf-8"))


def item_order_sha256(ordered_keys: list[str]) -> str:
    """R-050 DECISION: digest over keys IN THE ORDER USED for the vectors
    (newline-joined, no trailing newline). For the canonical package the
    order is ascending UTF-8, so this equals ``keyset_sha256`` there — they
    remain distinct FIELDS with distinct duties."""
    return sha256_bytes("\n".join(ordered_keys).encode("utf-8"))


def expected_item_key(source_text: str) -> str:
    """R-008 pinned item-key derivation: itm-<first 16 hex of sha256(NFC)>."""
    normalized = unicodedata.normalize("NFC", source_text)
    return "itm-" + hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def expected_estimand_digest(estimand: dict[str, Any]) -> str:
    """R-011 pinned digest = sha256 of canonical compact sorted JSON."""
    payload = json.dumps(estimand, sort_keys=True, separators=(",", ":"))
    return sha256_bytes(payload.encode("utf-8"))


def tree_hashes(root: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for path in sorted(Path(root).rglob("*")):
        if path.is_file():
            out[path.relative_to(root).as_posix()] = sha256_file(path)
    return out


def expected_tree_sha256(root: Path) -> str:
    """R-036 pinned input-tree digest: sha256 over utf-8 lines
    ``<posix relpath>:<file sha256>`` sorted by relpath, newline-joined with
    a trailing newline."""
    lines = [f"{rel}:{sha}" for rel, sha in sorted(tree_hashes(root).items())]
    return sha256_bytes(("\n".join(lines) + "\n").encode("utf-8"))


def expected_code_sha256() -> str:
    """R-036 pinned verifier-code digest (same line scheme over the
    namespace's ``*.py`` files, relpath within the namespace, sorted)."""
    lines = [
        f"{p.relative_to(NAMESPACE_DIR).as_posix()}:{sha256_file(p)}"
        for p in sorted(NAMESPACE_DIR.glob("**/*.py"))
    ]
    return sha256_bytes(("\n".join(lines) + "\n").encode("utf-8"))


def repo_head_commit() -> str:
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


def namespace_py_files() -> list[Path]:
    """Every production .py file in the namespace (AST-scan surface)."""
    return sorted(NAMESPACE_DIR.glob("**/*.py"))


# ---------------------------------------------------------------------------
# D7(b) pure reference procedure (sign-off SS3, exact; NO namespace imports)
# ---------------------------------------------------------------------------


def d7b_seed(pairing_population_keyset_sha256: str) -> int:
    """R-052 exact seed derivation."""
    seed_material = b"colm_aims_2026/v2/bootstrap_holm\0" + bytes.fromhex(
        pairing_population_keyset_sha256
    )
    return int.from_bytes(
        hashlib.sha256(seed_material).digest()[:8],
        byteorder="big",
        signed=False,
    )


def d7b_resample_matrix(seed: int, *, n: int = N_ITEMS, b: int = 1000) -> np.ndarray:
    """R-051 exact shared resample-index plan (NumPy 2.4.6 / PCG64)."""
    rng = np.random.Generator(np.random.PCG64(seed))
    return rng.integers(0, n, size=(b, n), dtype=np.int64, endpoint=False)


def d7b_matrix_digest_record(
    indices: np.ndarray, canonical_item_order_digest: str
) -> dict[str, Any]:
    """R-053: digest over the exact resample-index bytes + the four covering
    fields (dtype, shape, byte order, item-order digest)."""
    return {
        "sha256": sha256_bytes(indices.tobytes()),
        "dtype": str(indices.dtype),
        "shape": list(indices.shape),
        "byte_order": sys.byteorder,
        "canonical_item_order_digest": canonical_item_order_digest,
    }


def d7b_interval(d: np.ndarray, indices: np.ndarray) -> tuple[float, float]:
    """R-054: uncentered percentile interval (2.5/97.5, method='linear')."""
    boot_means = d[indices].mean(axis=1)
    lo, hi = np.quantile(boot_means, [0.025, 0.975], method="linear")
    return float(lo), float(hi)


def d7b_p_value_from_null_means(null_means: np.ndarray, mu_hat: float) -> float:
    """R-055: p = (1 + #{|mu0_b| >= |mu_hat|}) / 1001 (the +1 is mandatory)."""
    exceed = int(np.sum(np.abs(np.asarray(null_means)) >= abs(mu_hat)))
    return (1 + exceed) / 1001


def d7b_p_value(d: np.ndarray, indices: np.ndarray) -> float:
    """R-055: null-centered paired bootstrap p over the SAME index matrix."""
    mu_hat = float(np.mean(d))
    z = d - mu_hat
    null_means = z[indices].mean(axis=1)
    return d7b_p_value_from_null_means(null_means, mu_hat)


def d7b_holm(raw_p_by_cell: dict[str, float]) -> dict[str, Any]:
    """R-056: Holm step-down, m=10, alpha 0.05, ascending raw p, ties by
    ascending UTF-8 byte order of cell_id; adjusted p capped at 1 with
    step-down monotonicity."""
    if len(raw_p_by_cell) != 10:
        raise ValueError(
            f"Holm family must be exactly the ten-cell 5x2 grid (m=10);"
            f" got {len(raw_p_by_cell)} cells (R-056)"
        )
    m = 10
    alpha = 0.05
    ordered = sorted(
        raw_p_by_cell.items(),
        key=lambda kv: (kv[1], kv[0].encode("utf-8")),
    )
    per_cell: dict[str, dict[str, Any]] = {}
    adjusted_running = 0.0
    still_rejecting = True
    rejected: list[str] = []
    for rank0, (cell_id, p) in enumerate(ordered):
        mult = m - rank0
        adjusted_running = max(adjusted_running, min(1.0, p * mult))
        if still_rejecting and p <= alpha / mult:
            rejected.append(cell_id)
        else:
            still_rejecting = False
        per_cell[cell_id] = {
            "holm_rank": rank0 + 1,
            "holm_adjusted_p_value": adjusted_running,
            "holm_rejected": cell_id in rejected,
        }
    return {
        "ordered_family": [cell_id for cell_id, _ in ordered],
        "rejected_cell_ids": sorted(rejected),
        "per_cell": per_cell,
        "familywise_alpha": alpha,
        "family_size": m,
    }


# ---------------------------------------------------------------------------
# Canonical ten-cell synthetic data (deterministic generative arithmetic)
# ---------------------------------------------------------------------------
#
# Item index i = rank of the item key in ascending UTF-8 order (0..2248).
#   mc_raw(i, cal)       = (7*i + 3 + 11*cal_idx) % 9   -> >= horizon => NEVER
#   ref_raw(i, ref, cal) = (5*i + 1 + 3*ref_idx + 13*cal_idx) % 8
# MC stops depend only on (item, calibration): identical across the five
# references WITHIN a calibration (R-043) and differing BETWEEN calibrations
# (the mandatory nearest-true control for the retracted global MC-stop leg).


def canonical_item_keys() -> list[str]:
    return sorted(
        expected_item_key(f"synthetic-item-{i:04d}") for i in range(N_ITEMS)
    )


def _raw_stops(cell_id: str) -> tuple[np.ndarray, np.ndarray]:
    ref_id, cal_id = cell_id.split("__", 1)
    r_i = REFERENCE_IDS.index(ref_id)
    c_i = CALIBRATION_IDS.index(cal_id)
    i = np.arange(N_ITEMS, dtype=np.int64)
    mc_raw = (7 * i + 3 + 11 * c_i) % 9
    ref_raw = (5 * i + 1 + 3 * r_i + 13 * c_i) % 8
    return mc_raw, ref_raw


@dataclass
class CellData:
    cell_id: str
    reference_id: str
    calibration_id: str
    mc_raw: np.ndarray
    ref_raw: np.ndarray
    d: np.ndarray  # derived signed shifts (sentinel-coded, int64)
    counts: dict[str, Any]
    rates: dict[str, float]
    headline_mean: float
    finite_only: dict[str, Any]
    ci: tuple[float, float]
    raw_p_value: float
    records_bytes: bytes


@functools.lru_cache(maxsize=1)
def canonical_data() -> dict[str, Any]:
    """Compute the full canonical ten-cell dataset once (test-side oracle)."""
    keys = canonical_item_keys()
    keyset_digest = keyset_sha256(keys)
    order_digest = item_order_sha256(keys)
    seed = d7b_seed(keyset_digest)
    indices = d7b_resample_matrix(seed)
    matrix_digest = d7b_matrix_digest_record(indices, order_digest)

    cells: dict[str, CellData] = {}
    raw_p: dict[str, float] = {}
    h = TRAJECTORY_HORIZON
    for cell_id in CELL_IDS:
        mc_raw, ref_raw = _raw_stops(cell_id)
        mc_fin = mc_raw < h
        ref_fin = ref_raw < h
        s_mc = np.where(mc_fin, mc_raw, h)
        s_ref = np.where(ref_fin, ref_raw, h)
        d = (s_mc - s_ref).astype(np.int64)
        n_bf = int(np.sum(mc_fin & ref_fin))
        counts = {
            "n_both_finite": n_bf,
            "n_mc_finite_ref_timeout": int(np.sum(mc_fin & ~ref_fin)),
            "n_mc_timeout_ref_finite": int(np.sum(~mc_fin & ref_fin)),
            "n_both_timeout": int(np.sum(~mc_fin & ~ref_fin)),
            "n_complete": N_ITEMS,
            "n_excluded_or_unpaired": 0,
            "exclusion_reason_counts": {},
            "n_pairing_population": N_ITEMS,
            "n_mc_timeout": int(np.sum(~mc_fin)),
            "n_ref_timeout": int(np.sum(~ref_fin)),
        }
        rates = {
            "rate_both_finite": counts["n_both_finite"] / N_ITEMS,
            "rate_mc_finite_ref_timeout": counts["n_mc_finite_ref_timeout"]
            / N_ITEMS,
            "rate_mc_timeout_ref_finite": counts["n_mc_timeout_ref_finite"]
            / N_ITEMS,
            "rate_both_timeout": counts["n_both_timeout"] / N_ITEMS,
        }
        d_bf = (s_mc - s_ref)[mc_fin & ref_fin].astype(np.float64)
        finite_only = {
            "n": n_bf,
            "signed_index_mean": float(np.mean(d_bf)),
            "signed_index_median": float(np.median(d_bf)),
            "absolute_index_mean": float(np.mean(np.abs(d_bf))),
            "absolute_index_median": float(np.median(np.abs(d_bf))),
        }
        df = d.astype(np.float64)
        ci = d7b_interval(df, indices)
        p = d7b_p_value(df, indices)
        raw_p[cell_id] = p

        records_lines: list[str] = []
        for idx, key in enumerate(keys):
            rec = make_record_v2(
                key,
                int(mc_raw[idx]) if mc_fin[idx] else None,
                int(ref_raw[idx]) if ref_fin[idx] else None,
            )
            records_lines.append(json.dumps(rec, sort_keys=True))
        records_bytes = ("\n".join(records_lines) + "\n").encode("utf-8")

        ref_id, cal_id = cell_id.split("__", 1)
        cells[cell_id] = CellData(
            cell_id=cell_id,
            reference_id=ref_id,
            calibration_id=cal_id,
            mc_raw=mc_raw,
            ref_raw=ref_raw,
            d=d,
            counts=counts,
            rates=rates,
            headline_mean=float(np.mean(df)),
            finite_only=finite_only,
            ci=ci,
            raw_p_value=p,
            records_bytes=records_bytes,
        )

    holm = d7b_holm(raw_p)
    return {
        "keys": keys,
        "keyset_digest": keyset_digest,
        "order_digest": order_digest,
        "seed": seed,
        "matrix_digest": matrix_digest,
        "cells": cells,
        "holm": holm,
    }


# ---------------------------------------------------------------------------
# Record / block / document builders
# ---------------------------------------------------------------------------


def make_record_v2(
    item_key: str,
    mc_stop: int | None,
    ref_stop: int | None,
    *,
    trajectory_horizon: int = TRAJECTORY_HORIZON,
    **extra: Any,
) -> dict[str, Any]:
    """One canonical v2 event record (R-045). ``None`` stop => NEVER_STOPPED."""
    rec: dict[str, Any] = {
        "item_key": item_key,
        "trajectory_horizon": trajectory_horizon,
        "mc_event_status": EVENT_FINITE if mc_stop is not None else EVENT_NEVER,
        "mc_stop_step": mc_stop,
        "mc_terminal_imputation": (
            IMPUTATION_NONE if mc_stop is not None else IMPUTATION_FINAL_PREFIX
        ),
        "ref_event_status": (
            EVENT_FINITE if ref_stop is not None else EVENT_NEVER
        ),
        "ref_stop_step": ref_stop,
        "ref_terminal_imputation": (
            IMPUTATION_NONE if ref_stop is not None else IMPUTATION_FINAL_PREFIX
        ),
    }
    rec.update(extra)
    return rec


def make_event_representation(**overrides: Any) -> dict[str, Any]:
    """R-045 record-set bindings (cell-level, inside the estimand)."""
    block: dict[str, Any] = {
        "index_base": 0,
        "horizon_identity": "hz-0006",
        "mc_trajectory_identity": "traj-mc-v2-0001",
        "historical_sentinel_convention": SENTINEL_CONVENTION,
        "terminal_imputation_policy": IMPUTATION_FINAL_PREFIX,
        "producer_profile_identity": f"{STRICT_PROFILE_ID}:producer-0001",
    }
    block.update(overrides)
    return block


def calibration_identity_map() -> dict[str, str]:
    """R-001 D1: calibration identity is a MAP, one entry per calibration ID."""
    return {"shared": "cal-shared-0001", "format_specific": "cal-fmt-0001"}


def make_estimand(cell_id: str, **overrides: Any) -> dict[str, Any]:
    ref_id, cal_id = cell_id.split("__", 1)
    est: dict[str, Any] = {
        "arm_mc": "mc_trajectory",
        "arm_ref": ref_id,
        "reference_id": ref_id,
        "calibration_id": cal_id,
        "pairing_definition": "matched_item_prefix_grid",
        "timeout_parameters": {
            "trajectory_horizon": TRAJECTORY_HORIZON,
            "rule": "zero_indexed_stop_ge_horizon_is_timeout",
        },
        "event_representation": make_event_representation(),
        "population": POPULATION_ALL,
        "denominator_policy": "n_complete",
        "numerical_tolerance": 1e-9,
        "calibration_identity": calibration_identity_map()[cal_id],
        "continuation_identity": "cont-0001",
        "random_k_draw_id": (
            "draw-archived-0001" if ref_id == "krandom" else "draw-none"
        ),
    }
    est.update(overrides)
    return est


def make_arm(
    arm_id: str,
    *,
    family: str = "constructed_reference",
    cardinality: str = "k_way",
    construction: str = "mc_grid",
    reporting_eligibility: str = "headline_eligible",
) -> dict[str, Any]:
    """One arm identity block (R-003, v2 per-family closed vocabulary)."""
    return {
        "arm_id": arm_id,
        "family": family,
        "stop_semantics": FAMILY_STOP_VOCAB[family],
        "construction": construction,
        "cardinality": cardinality,
        "selector": "argmax_calibrated_score",
        "scorer": "tiny-scorer",
        "candidate_pool_role": (
            "distractor_pool" if cardinality == "k_way" else "none"
        ),
        "correctness_assignment": (
            "oracle_gold" if construction == "idealized" else "option_match"
        ),
        "calibration_role": "calibrated",
        "continuation_role": "dp_continuation",
        "seed_contract": {"seeds": [1, 2, 3]},
        "reporting_eligibility": reporting_eligibility,
    }


def make_idealized_arm(arm_id: str = "idealized") -> dict[str, Any]:
    """R-003: scalar prefix-to-gold cosine with oracle-assigned correctness."""
    arm = make_arm(arm_id, cardinality="scalar", construction="idealized")
    arm["scorer"] = "prefix_to_gold_cosine"
    arm["selector"] = "threshold_on_scalar"
    return arm


def make_arms() -> list[dict[str, Any]]:
    arms = [
        make_arm("mc_trajectory", family="learned_continuation"),
        make_idealized_arm("idealized"),
    ]
    for ref in ("kdisjoint", "khard", "klex"):
        arms.append(make_arm(ref))
    arms.append(
        make_arm("krandom", reporting_eligibility="non_headline_disclosure_only")
    )
    return arms


def make_cell_v2(cell_id: str, **overrides: Any) -> dict[str, Any]:
    data = canonical_data()
    cd: CellData = data["cells"][cell_id]
    holm_cell = data["holm"]["per_cell"][cell_id]
    estimand = overrides.pop("estimand", make_estimand(cell_id))
    cell: dict[str, Any] = {
        "cell_id": cell_id,
        "reference_id": cd.reference_id,
        "calibration_id": cd.calibration_id,
        "estimand": estimand,
        "estimand_digest": expected_estimand_digest(estimand),
        "records_file": f"records/{cell_id}.jsonl",
        "counts": dict(cd.counts),
        "rates": dict(cd.rates),
        "headline_summary": {
            "estimand_label": HEADLINE_ESTIMAND_LABEL,
            "population": POPULATION_ALL,
            "n": N_ITEMS,
            "mean_signed_shift": cd.headline_mean,
            "convention": SENTINEL_CONVENTION,
        },
        "finite_only_summary": {
            "estimand_label": FINITE_ONLY_ESTIMAND_LABEL,
            "population": POPULATION_FINITE,
            **cd.finite_only,
        },
        "interval": {
            "procedure": "d7b_shared_percentile_bootstrap",
            "draw_count": 1000,
            "seed": data["seed"],
            "seed_derivation": SEED_DERIVATION_STRING,
            "statistic": "mean_signed_shift",
            "population": POPULATION_ALL,
            "quantile_method": "linear",
            "ci": [cd.ci[0], cd.ci[1]],
        },
        "raw_p_value": cd.raw_p_value,
        "holm_rank": holm_cell["holm_rank"],
        "holm_adjusted_p_value": holm_cell["holm_adjusted_p_value"],
        "holm_rejected": holm_cell["holm_rejected"],
        "excluded_keys": [],
        "pairing_population_keyset_sha256": data["keyset_digest"],
    }
    cell.update(overrides)
    return cell


def make_grid_block(**overrides: Any) -> dict[str, Any]:
    data = canonical_data()
    block: dict[str, Any] = {
        "reference_ids": sorted(REFERENCE_IDS),
        "calibration_ids": sorted(CALIBRATION_IDS),
        "cell_ids": list(CELL_IDS),
        "record_files": {c: f"records/{c}.jsonl" for c in CELL_IDS},
        "item_keys_sha256": data["keyset_digest"],
        "held_fixed": {
            "mc_trajectory_identity": "traj-mc-v2-0001",
            "horizon_identity": "hz-0006",
        },
    }
    block.update(overrides)
    return block


def make_inference_block(**overrides: Any) -> dict[str, Any]:
    data = canonical_data()
    block: dict[str, Any] = {
        "analysis_provenance": ANALYSIS_PROVENANCE_D7B,
        "numpy_version": "2.4.6",
        "bit_generator": "PCG64",
        "generator_construction": (
            "numpy.random.Generator(numpy.random.PCG64(seed))"
        ),
        "draw_count": 1000,
        "sample_size": N_ITEMS,
        "resampling_unit": "item_tossup_clustered_all_prefixes_both_arms",
        "with_replacement": True,
        "dtype": "int64",
        "endpoint": False,
        "seed": data["seed"],
        "seed_derivation": SEED_DERIVATION_STRING,
        "pairing_population_keyset_sha256": data["keyset_digest"],
        "canonical_item_order_digest": data["order_digest"],
        "resample_matrix_digest": dict(data["matrix_digest"]),
        "familywise_alpha": 0.05,
        "family_size": 10,
        "ordered_family": list(data["holm"]["ordered_family"]),
        "rejected_cell_ids": list(data["holm"]["rejected_cell_ids"]),
    }
    block.update(overrides)
    return block


def make_llm_involvement(**overrides: Any) -> dict[str, Any]:
    block: dict[str, Any] = {
        "reference_construction": "none",
        "data_plot_creation": "none",
        "evaluation": "none",
    }
    block.update(overrides)
    return block


def make_provenance(
    *,
    source_commit: str | None = None,
    input_sha256: dict[str, str] | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    keys = canonical_item_keys()
    fit_keys = [f"fit-{i:04d}" for i in range(1, 7)]
    prov: dict[str, Any] = {
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
                "name": "fit-v2",
                "count": len(fit_keys),
                "keyset_sha256": keyset_sha256(fit_keys),
            },
            "eval": {
                "name": "eval-v2",
                "count": len(keys),
                "keyset_sha256": keyset_sha256(keys),
            },
            "zero_overlap": True,
        },
        "calibration_identity": calibration_identity_map(),
        "continuation_identity": "cont-0001",
        "input_sha256": dict(input_sha256 or {}),
        "split_metadata_sha256": FAKE_SHA_C,
        "mc_build": {
            "built_after_split": True,
            "coverage_rate": 1.0,
            "retention_policy": "retain_all",
            "retained_count": len(keys),
        },
        # R-052(a): the 9 upstream-unpaired items are pre-package retention
        # documentation in provenance, never in-package excluded_keys.
        "pre_package_retention": {
            "retained_count": 2258,
            "paired_count": 2249,
            "upstream_unpaired_count": 9,
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
        "runtime_packages": {"python": "3.11", "numpy": "2.4.6"},
    }
    prov.update(overrides)
    return prov


def make_profile_v2(
    *,
    source_commit: str | None = None,
    input_sha256: dict[str, str] | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    """A complete, internally consistent strict v2 ten-cell profile (R-001)."""
    profile: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
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
        "arms": make_arms(),
        "provenance": make_provenance(
            source_commit=source_commit, input_sha256=input_sha256
        ),
        "grid": make_grid_block(),
        "inference": make_inference_block(),
        "cells": [make_cell_v2(cell_id) for cell_id in CELL_IDS],
    }
    profile.update(overrides)
    return profile


# ---------------------------------------------------------------------------
# Ledger / rights / manifest / expectations builders
# ---------------------------------------------------------------------------


def make_ledger_row(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "claim_id": "clm-0001",
        "claim_kind": "per_item_paired",
        "manuscript_location": "Section 4.1, paragraph 2",
        "manuscript_wording": (
            "Constructed QA reference sensitivity diagnostic over matched"
            " item-prefix grids."
        ),
        "estimand": HEADLINE_ESTIMAND_LABEL,
        "allowed_scope": "reference_sensitivity_diagnostic",
        "producer_entrypoint": "scripts/fake_producer.py",
        "dependency_closure": [
            "scripts/fake_producer.py",
            "scripts/fake_helper.py",
        ],
        "input_identity": FAKE_SHA_A,
        "split_identity": "eval-v2",
        "model_identity": (
            "example-org/tiny-scorer@1234567890abcdef1234567890abcdef12345678"
        ),
        "calibration_identity": "cal-shared-0001",
        "artifact_id": "profile.json",
        "renderer_id": "colm_aims_render_v2",
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


def make_holm_row(**overrides: Any) -> dict[str, Any]:
    data = canonical_data()
    row = make_ledger_row(
        claim_id="clm-holm-inference",
        claim_kind="aggregate",
        manuscript_location="Section 5 (ten-cell Holm family)",
        manuscript_wording=(
            "D7(b) regenerated ten-cell Holm family over the constructed QA"
            " reference grid (new analysis; not the historical inference)."
        ),
        estimand="d7b_holm_family_m10",
        artifact_id="profile.json",
        artifact_family="inference_block",
        headline_eligible=False,
    )
    row["analysis_provenance"] = ANALYSIS_PROVENANCE_D7B
    row["rejected_cell_ids"] = list(data["holm"]["rejected_cell_ids"])
    row.update(overrides)
    return row


def make_random_k_row(**overrides: Any) -> dict[str, Any]:
    row = make_ledger_row(
        claim_id="clm-randomk-disposition",
        claim_kind="per_item_paired",
        manuscript_location="Section 5 (Random-K analysis, submitted version)",
        manuscript_wording=(
            "Historical Random-K/v5 result; single unpinned archived"
            " realization; non-headline dagger disclosure."
        ),
        estimand="random_k_draw_sensitivity",
        artifact_id="historical-random-k-results",
        artifact_family="random_k",
        rights_status="UNVERIFIED",
        status="UNVERIFIED",
        blocking_task=RANDOM_K_BLOCKING_TASK,
        provenance_class="historical_randomk_v5",
        headline_eligible=False,
    )
    row["author_decision"] = "historical_nonconfirmatory"
    row["rng_pinned"] = False
    row["archived_draw_id"] = "draw-archived-0001"
    row["fresh_draw_id"] = "draw-fresh-0001"
    row["disclosure_marker"] = "dagger"
    row.update(overrides)
    return row


def make_external_row(**overrides: Any) -> dict[str, Any]:
    row = make_ledger_row(
        claim_id="clm-ext-0001",
        claim_kind="external_fact",
        manuscript_location="Title page",
        manuscript_wording="Manuscript identity (D6 baseline).",
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


def make_ledger(
    *,
    source_commit: str,
    rows: list[dict[str, Any]] | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    ledger: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "ledger_id": "colm-aims-2026-ledger-v2",
        "anchored_source_commit": source_commit,
        "manuscript": {
            "main_tex_sha256": D6_MAIN_TEX_SHA256,
            "main_pdf_sha256": D6_MAIN_PDF_SHA256,
        },
        "documents": [
            {
                "path": "reproducibility/source_to_claim.md",
                "provenance_class": "historical_submission_artifact",
            }
        ],
        "canonical_run_id": "run-0001",
        "rows": (
            rows
            if rows is not None
            else [
                make_ledger_row(),
                make_holm_row(),
                make_random_k_row(),
                make_external_row(),
            ]
        ),
    }
    ledger.update(overrides)
    return ledger


def make_rights(paths: list[str], **overrides: Any) -> dict[str, Any]:
    rights: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "paths": [
            {
                "path": p,
                "status": "VERIFIED_ALLOWED",
                "upstream_terms_basis": (
                    "synthetic test fixture generated in-repo"
                ),
            }
            for p in paths
        ],
    }
    rights.update(overrides)
    return rights


def make_manifest(artifact_paths: list[str], **overrides: Any) -> dict[str, Any]:
    roles = {"profile.json": "strict_profile"}
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifacts": [
            {
                "path": p,
                "role": roles.get(
                    p,
                    "per_item_records" if p.startswith("records/") else "other",
                ),
            }
            for p in artifact_paths
        ],
        "allowlist_undeclared": [],
    }
    manifest.update(overrides)
    return manifest


def make_suite_receipt(**overrides: Any) -> dict[str, Any]:
    """R-070 canonical CI suite-evidence receipt (all-hash bindings)."""
    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "environment_digest": FAKE_SHA_A,
        "workflow_file_sha256": FAKE_SHA_B,
        "interpreter_realpath": "/opt/python/3.11.15/bin/python3.11",
        "commit": FAKE_COMMIT,
        "tree": "e" * 40,
        "dirty": False,
        "command": ["python", "-m", "pytest", "tests/", "-q"],
        "exit_code": 0,
        "junit_report_sha256": FAKE_SHA_C,
        "skip_identities": [],
        "artifact_hashes": {"results/suite.xml": FAKE_SHA_C},
    }
    receipt.update(overrides)
    return receipt


def make_closure_inventory(**overrides: Any) -> dict[str, Any]:
    """R-071/R-072 canonical CAMERA_READY_CLOSURE inventory (satisfied form)."""
    inventory: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "d6_baseline": {
            "main_tex_sha256": D6_MAIN_TEX_SHA256,
            "main_pdf_sha256": D6_MAIN_PDF_SHA256,
            "final_checksums_sha256": FAKE_SHA_A,
            "final_checksums_entries": {
                "main.tex": D6_MAIN_TEX_SHA256,
                "main.pdf": D6_MAIN_PDF_SHA256,
                "figures/fig1.pdf": FAKE_SHA_B,
                "references.bib": FAKE_SHA_C,
            },
        },
        "rows": [
            {
                "item": "table-1-headline-shifts",
                "status": "SATISFIED",
                "evidence": "profile.json ten-cell package",
            },
            {
                "item": "manuscript-identity",
                "status": "EXTERNAL",
                "evidence": "D6 two-party hash verification",
            },
        ],
        "holm_row": {"satisfied_by": ANALYSIS_PROVENANCE_D7B},
        "qa012": {
            "status": "VERIFIED_VACUOUS",
            "inventory_sha256": FAKE_SHA_B,
        },
    }
    inventory.update(overrides)
    return inventory


# ---------------------------------------------------------------------------
# Package builder
# ---------------------------------------------------------------------------


@dataclass
class Package:
    root: Path
    tree: Path
    profile_path: Path
    records_dir: Path
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


def build_package_v2(
    base: Path,
    *,
    profile_mutator: Callable[[dict[str, Any]], None] | None = None,
    binding_mutator: Callable[[dict[str, Any]], None] | None = None,
    ledger_mutator: Callable[[dict[str, Any]], None] | None = None,
    rights_mutator: Callable[[dict[str, Any]], None] | None = None,
    manifest_mutator: Callable[[dict[str, Any]], None] | None = None,
    expectations_mutator: Callable[[dict[str, Any]], None] | None = None,
    records_mutator: (
        Callable[[str, list[dict[str, Any]]], list[dict[str, Any]] | None] | None
    ) = None,
    raw_records_bytes: dict[str, bytes] | None = None,
    omit_record_files: tuple[str, ...] = (),
    extra_tree_files: dict[str, bytes] | None = None,
    extra_manifest_allowlist: tuple[str, ...] = (),
    extra_rights_paths: tuple[str, ...] = (),
    source_commit: str | None = None,
    include_sentinel_payload: bool = True,
    tree_dirname: str = "tree",
) -> Package:
    """Build the canonical ten-cell evidence package, optionally mutated.

    Pre-hash mutators (profile/records/ledger/rights/manifest) keep the
    expectations consistent with the mutated bytes so a semantic defect is
    the ONLY defect; ``expectations_mutator`` runs last, post-hash, to
    deliberately break a binding (R-021 verdict-flip mutations).
    ``binding_mutator`` runs on the profile's provenance pre-hash (the
    artifact-side mutation hook).
    """
    data = canonical_data()
    commit = source_commit or repo_head_commit()

    root = Path(base) / "pkg"
    tree = root / tree_dirname
    receipts = root / "receipts"
    records_dir = tree / "records"
    records_dir.mkdir(parents=True, exist_ok=True)
    receipts.mkdir(parents=True, exist_ok=True)

    # --- per-cell record files (R-041) ---------------------------------
    input_hashes: dict[str, str] = {}
    for cell_id in CELL_IDS:
        rel = f"records/{cell_id}.jsonl"
        if cell_id in omit_record_files:
            continue
        if raw_records_bytes and cell_id in raw_records_bytes:
            blob = raw_records_bytes[cell_id]
        elif records_mutator is not None:
            records = [
                json.loads(line)
                for line in data["cells"][cell_id]
                .records_bytes.decode("utf-8")
                .splitlines()
                if line.strip()
            ]
            mutated = records_mutator(cell_id, records)
            if mutated is None:
                mutated = records
            blob = (
                "\n".join(json.dumps(r, sort_keys=True) for r in mutated) + "\n"
            ).encode("utf-8")
        else:
            blob = data["cells"][cell_id].records_bytes
        (tree / rel).write_bytes(blob)
        input_hashes[rel] = sha256_bytes(blob)

    profile = make_profile_v2(source_commit=commit, input_sha256=input_hashes)
    if profile_mutator is not None:
        profile_mutator(profile)
    if binding_mutator is not None:
        binding_mutator(profile["provenance"])
    profile_path = tree / "profile.json"
    profile_path.write_bytes(_dump(profile))

    artifact_paths = ["profile.json"] + sorted(input_hashes)
    rights_paths = list(artifact_paths)
    manifest = make_manifest(artifact_paths)
    if include_sentinel_payload:
        (tree / "sealed-notes.bin").write_bytes(
            (SENTINEL + "\n").encode("utf-8")
        )
        manifest["artifacts"].append(
            {"path": "sealed-notes.bin", "role": "sealed_payload"}
        )
        rights_paths.append("sealed-notes.bin")

    if extra_tree_files:
        for rel, blob in extra_tree_files.items():
            p = tree / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(blob)
    for rel in extra_manifest_allowlist:
        manifest["allowlist_undeclared"].append(
            {"path": rel, "reason": "tolerated historical sidecar (test)"}
        )
    if manifest_mutator is not None:
        manifest_mutator(manifest)
    manifest_path = tree / "presentation_manifest.json"
    manifest_path.write_bytes(_dump(manifest))

    rights = make_rights(rights_paths + list(extra_rights_paths))
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
        "schema_version": SCHEMA_VERSION,
        "anchor": {
            "source_commit": commit,
            "ledger_path": "ledger.json",
            "ledger_sha256": sha256_file(ledger_path),
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
        "tree_files": {rel: sha for rel, sha in sorted(tree_hashes(tree).items())},
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
            "calibration_identity": dict(prov["calibration_identity"]),
            "continuation_identity": prov["continuation_identity"],
            "input_hashes": dict(prov["input_sha256"]),
            "split_metadata_sha256": prov["split_metadata_sha256"],
            "mc_build": dict(prov["mc_build"]),
            "model": json.loads(json.dumps(prov["model"])),
            "runtime_packages": dict(prov["runtime_packages"]),
            # R-044: the grid identity pinned SEMANTICALLY out-of-tree.
            "grid": {
                "reference_ids": sorted(REFERENCE_IDS),
                "calibration_ids": sorted(CALIBRATION_IDS),
                "cell_ids": list(CELL_IDS),
                "record_files": {c: f"records/{c}.jsonl" for c in CELL_IDS},
                "item_keys_sha256": data["keyset_digest"],
                "held_fixed": {
                    "mc_trajectory_identity": "traj-mc-v2-0001",
                    "horizon_identity": "hz-0006",
                },
            },
            # R-052(b)/R-053: inference identities pinned out-of-tree.
            "inference": {
                "seed": data["seed"],
                "seed_derivation": SEED_DERIVATION_STRING,
                "pairing_population_keyset_sha256": data["keyset_digest"],
                "canonical_item_order_digest": data["order_digest"],
                "resample_matrix_digest": dict(data["matrix_digest"]),
                "draw_count": 1000,
                "numpy_version": "2.4.6",
                "bit_generator": "PCG64",
            },
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
        records_dir=records_dir,
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


@dataclass
class RunsSite:
    root: Path
    runs_root: Path
    run_tree: Path
    ledger_path: Path
    expectations_path: Path
    rights_path: Path
    receipts_dir: Path


def build_runs_site(
    base: Path,
    *,
    run_id: str = "run-0001",
    canonical_pointer: str | None = None,
    extra_runs: tuple[str, ...] = (),
    **package_kwargs: Any,
) -> RunsSite:
    """Build the R-069 runs-site layout: canonical selection via the ledger
    pointer, package tree at ``runs/<run_id>/tree``. ``canonical_pointer``
    overrides the ledger's ``canonical_run_id`` (defaults to ``run_id``)."""
    pointer = canonical_pointer if canonical_pointer is not None else run_id
    site = Path(base) / "site"
    runs_root = site / "runs"
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    for extra in extra_runs:
        (runs_root / extra / "tree").mkdir(parents=True, exist_ok=True)

    prior_ledger_mutator = package_kwargs.pop("ledger_mutator", None)

    def _ledger_mutator(ledger: dict[str, Any]) -> None:
        ledger["canonical_run_id"] = pointer
        if prior_ledger_mutator is not None:
            prior_ledger_mutator(ledger)

    pkg = build_package_v2(
        run_dir, ledger_mutator=_ledger_mutator, **package_kwargs
    )
    # Relocate: the run dir keeps ONLY the published tree (at
    # ``runs/<run_id>/tree``); the ledger, expectations, rights, and
    # receipts live at the SITE root.
    ledger_path = site / "ledger.json"
    expectations_path = site / "expectations.json"
    rights_path = site / "rights.json"
    receipts = site / "receipts"
    receipts.mkdir(parents=True, exist_ok=True)
    shutil.move(str(pkg.ledger_path), str(ledger_path))
    shutil.move(str(pkg.rights_path), str(rights_path))
    run_tree = run_dir / "tree"
    shutil.move(str(pkg.tree), str(run_tree))

    expectations = pkg.expectations
    expectations["anchor"]["ledger_sha256"] = sha256_file(ledger_path)
    expectations_path.write_bytes(_dump(expectations))
    pkg.expectations_path.unlink()
    shutil.rmtree(pkg.receipts_dir, ignore_errors=True)
    pkg.root.rmdir()  # now empty: everything relocated

    return RunsSite(
        root=site,
        runs_root=runs_root,
        run_tree=run_tree,
        ledger_path=ledger_path,
        expectations_path=expectations_path,
        rights_path=rights_path,
        receipts_dir=receipts,
    )


def rewrite_json(path: Path, mutate: Callable[[dict[str, Any]], None]) -> None:
    obj = json.loads(Path(path).read_text("utf-8"))
    mutate(obj)
    Path(path).write_bytes(_dump(obj))


# ---------------------------------------------------------------------------
# Report / leg assertion helpers
# ---------------------------------------------------------------------------


def leg_by_id(report: Any, leg_id: str) -> dict[str, Any] | None:
    for leg in report.legs:
        if leg.get("leg_id") == leg_id:
            return leg
    return None


def failing_leg_ids(report: Any) -> list[str]:
    return [
        leg.get("leg_id")
        for leg in report.legs
        if leg.get("status") == "FAIL"
    ]


def assert_failing_leg(report: Any, leg_id: str) -> dict[str, Any]:
    """Exact-leg-id oracle: the run FAILs and names this exact leg."""
    assert report.verdict == VERDICT_FAIL, (
        f"expected verdict FAIL, got {report.verdict!r}"
    )
    failing = failing_leg_ids(report)
    assert leg_id in failing, (
        f"expected failing leg {leg_id!r}; failing legs: {failing}"
    )
    leg = leg_by_id(report, leg_id)
    assert leg is not None
    return leg


def assert_failing_leg_prefix(report: Any, prefix: str) -> list[dict[str, Any]]:
    assert report.verdict == VERDICT_FAIL
    hits = [
        leg
        for leg in report.legs
        if leg.get("status") == "FAIL"
        and str(leg.get("leg_id", "")).startswith(prefix)
    ]
    assert hits, (
        f"expected a failing leg with prefix {prefix!r};"
        f" failing legs: {failing_leg_ids(report)}"
    )
    return hits


def assert_passing_report(report: Any, verdict: str) -> None:
    failing = failing_leg_ids(report)
    assert report.verdict == verdict, (
        f"expected {verdict}, got {report.verdict!r}; failing legs: {failing}"
    )
    assert not failing, f"unexpected failing legs on a passing run: {failing}"


def latest_receipt(receipts_dir: Path) -> dict[str, Any]:
    paths = sorted(Path(receipts_dir).glob("receipt-*.json"))
    assert paths, f"no receipt emitted under {receipts_dir}"
    return json.loads(paths[-1].read_text("utf-8"))


# ---------------------------------------------------------------------------
# CLI runner (R-021/R-037: documented invocation, via subprocess, repo root)
# ---------------------------------------------------------------------------

CLI_MODULE = "reproducibility.colm_aims_2026.verify"

_SHIM_DIR: Path | None = None


def _no_net_shim_dir() -> Path:
    """Session-scoped sitecustomize shim: installs the R-028 socket guard in
    every child interpreter when COLM_AIMS_TEST_NO_NET=1 (created at runtime
    in a temp dir; no repo fixture files)."""
    global _SHIM_DIR
    if _SHIM_DIR is None:
        _SHIM_DIR = Path(tempfile.mkdtemp(prefix="colm_v2_no_net_shim_"))
        (_SHIM_DIR / "sitecustomize.py").write_text(
            "import os, socket\n"
            "if os.environ.get('COLM_AIMS_TEST_NO_NET') == '1':\n"
            "    _real_connect = socket.socket.connect\n"
            "    def _guarded_connect(self, address):\n"
            "        if self.family == socket.AF_UNIX:\n"
            "            return _real_connect(self, address)\n"
            "        raise RuntimeError('network disabled (R-028)')\n"
            "    def _guarded_create_connection(*a, **k):\n"
            "        raise RuntimeError('network disabled (R-028)')\n"
            "    socket.socket.connect = _guarded_connect\n"
            "    socket.create_connection = _guarded_create_connection\n",
            encoding="utf-8",
        )
    return _SHIM_DIR


def cli_subprocess_env(
    overrides: dict[str, str] | None = None,
) -> dict[str, str]:
    env = dict(os.environ)
    env["COLM_AIMS_TEST_NO_NET"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    shim = str(_no_net_shim_dir())
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = f"{shim}{os.pathsep}{existing}" if existing else shim
    if overrides:
        env.update(overrides)
    return env


def run_cli(
    *args: str,
    cwd: Path = REPO_ROOT,
    env_overrides: dict[str, str] | None = None,
    argv0: list[str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run the documented CLI form via subprocess; asserts the R-026 sentinel
    never leaks into stdout/stderr on EVERY run."""
    base = argv0 if argv0 is not None else [sys.executable, "-m", CLI_MODULE]
    proc = subprocess.run(
        [*base, *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
        env=cli_subprocess_env(env_overrides),
    )
    assert SENTINEL not in proc.stdout, "sentinel leaked to stdout (R-026)"
    assert SENTINEL not in proc.stderr, "sentinel leaked to stderr (R-026)"
    return proc


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


def cli_args_for_runs_site(site: RunsSite) -> list[str]:
    """R-069 release entry: canonical selection through the ledger pointer."""
    return [
        "--mode",
        "release",
        "--runs-root",
        str(site.runs_root),
        "--expectations",
        str(site.expectations_path),
        "--receipts-dir",
        str(site.receipts_dir),
    ]


def gitless_path_dir() -> str:
    """A PATH with no git: contains only a symlink to the python interpreter
    (R-066 git-disappeared fixture)."""
    d = Path(tempfile.mkdtemp(prefix="colm_v2_gitless_"))
    target = Path(sys.executable)
    try:
        (d / target.name).symlink_to(target)
    except OSError:
        pass
    return str(d)


# ---------------------------------------------------------------------------
# In-process verifier convenience (imports the namespace lazily)
# ---------------------------------------------------------------------------


def run_verifier_on(pkg: Package, mode: str) -> Any:
    """Lazy-import run_verifier and run it over a built package."""
    from reproducibility.colm_aims_2026 import verifier as verifier_mod

    return verifier_mod.run_verifier(
        pkg.tree,
        mode=mode,
        receipts_dir=pkg.receipts_dir,
        expectations=pkg.expectations_path if mode == "release" else None,
    )


def source_report(pkg: Package) -> Any:
    return run_verifier_on(pkg, "source")


def release_report(pkg: Package) -> Any:
    return run_verifier_on(pkg, "release")
