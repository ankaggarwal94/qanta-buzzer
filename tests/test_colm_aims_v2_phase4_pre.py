"""Phase-4 PRE-run repairs: R-073..R-080 plus the amended R-043/R-045/R-072.

RED-phase contract module (2026-08-22). Spec:
`.correctless/specs/camera-ready-aims-evidence-2.md` section "Phase-4 PRE-run
repairs"; intent: `phase4_pre_run_reconciliation_2026-08-22.md` sections 4-5.

Source-only discipline: NO model load, NO network (autouse R-028 guard), NO
reads of `data/processed/` — only committed frozen artifacts, committed
excerpt fixtures, and tiny synthetic trees under tmp_path.

API CONTRACT PINNED FOR GREEN
=============================
``reproducibility.colm_aims_2026.phase4`` (new module):
  - ``load_pairing_eligibility(path) -> dict``: strict-parse + closed-key
    schema validation of the frozen eligibility artifact; version-first
    (bool-safe schema_version check precedes everything else); RECOMPUTES
    both digests via ``pairing.keyset_sha256(eligible_keys)`` and
    ``schema.horizon_map_sha256(horizon_map)`` and compares to the declared
    values; raises a ``schema.TypedIngressError`` subclass on ANY
    mismatch/malformation (unsorted keys, count drift, non-enum reason,
    bool/sub-2 horizons, unknown/missing keys, substituted digests).
  - ``staged_input_gate(staged: list[dict]) -> list[dict]``: items are
    ``{"path": Path, "expected_sha256": str, "label": str}``; hashes EVERY
    file and returns entries carrying ``observed_sha256`` == expected only
    when ALL match; on the FIRST (list-order) mismatch or missing file raises
    a ``schema.TypedIngressError`` subclass naming the file, the expected and
    the observed sha256; an EMPTY staged list raises (vacuously-empty
    authoritative sets are a defect); a malformed expected digest raises.
  - ``load_model_snapshot_manifest(path) -> dict``: strict load of the
    frozen role-keyed manifest; roles exactly
    ``{"primary_scorer", "disjoint_selector"}``; typed error otherwise.
  - ``verify_snapshot_dir(manifest_role_entry: dict, snapshot_dir: Path)
    -> None``: per-file sha256 AND size, no extra + no missing files,
    file_count consistency; raises a ``schema.ColmAimsError`` subclass
    naming the offending relative path on any deviation.
  - ``compare_parity(anchor: dict, regenerated_export: dict) -> dict``:
    regenerated export is producer-payload-shaped (identity fields at
    ``regenerated["metadata"]["n_eval"/"n_fit"]``; per-cell values at
    ``regenerated["results"][<historical cell label>][<policy>][<field>]``).
    Exact parsed-JSON-value equality AT THE SAME JSON TYPE (amended R-077:
    int drifting to float, bool, string-encoded number, non-finite — all
    FAIL; no cross-type numeric laundering) over the anchor allowlist: 160
    point fields + 32 CI arrays (every element) + the 2 identity fields
    (n_eval, n_fit) => ``checked == 194``. A TRUNCATED anchor (fewer than
    8 nonrandom cells x 2 policies x 10 point fields x 2 CI fields) is
    refused or FAILs — a comparison checking fewer than the full allowlist
    can never emit PASS, so ``verdict == "PASS"`` implies
    ``checked == 194``. Returns
    ``{"verdict": "PASS"|"FAIL", "checked": int, "failures": [
    {"cell","policy","field","expected","observed"}...],
    "random_k_informational": {...}}``. Identity-field failure rows carry
    ``cell=None, policy=None``. Bool observed vs numeric expected is a
    MISMATCH (no True==1/False==0 laundering). Missing cells/policies/fields
    become failure rows, never exceptions (guarded builder). The two
    Random-K cells are NEVER blocking and are reported informationally.
  - ``required_staged_coverage(consumed, staged_entries) -> list[dict]``
    (F-1): ``consumed`` is the producer's ordered enumeration of every
    fit/eval input as ``{"label": str, "path": Path, "frozen_sha256":
    str | None}``; ``staged_entries`` are the operator's ``--staged-input``
    triples ``{"label", "path", "expected_sha256"}``. Returns the fully
    resolved gate plan: one entry per consumed input, in consumed order,
    each ``{"label", "path", "expected_sha256"}`` with the expected digest
    filled from the frozen pin when present, else from the operator entry
    covering the same path. Raises a ``schema.TypedIngressError`` subclass
    when (a) any consumed input has neither a frozen pin nor an operator
    digest (uncovered input, named), (b) an operator digest CONTRADICTS a
    frozen pin (error names the file and BOTH digests), or (c) an operator
    entry names a path outside the consumed set (unknown staged input).
    The eval-split frozen pin is wired from
    ``eligibility["derived_from"]["test_dataset_sha256"]``.
  - ``gather_certificate_components(config, run=None) -> dict`` (F-4):
    pure gatherer feeding ``assemble_certificate``. ``run`` is an
    injectable command-runner ``run(cmd: list[str]) -> str`` (stdout;
    defaults to subprocess). Repo dirty state comes from the RUNNER's
    ``git status --porcelain --untracked-files=no`` output (empty ==
    tracked-clean, anything else == dirty — never a caller assertion;
    untracked evidence artifacts are disclosed by list, adjudicated
    amendment 2026-08-22); commit from ``git rev-parse HEAD``;
    tree from ``git rev-parse HEAD^{tree}``. Every staged-plan input is
    REHASHED from file bytes into ``observed_sha256`` (never copied from
    the expectation). Content hashes are computed by hashing the files at
    ``config["content_hash_paths"]``; the parity anchor and qa012 manifest
    hashes are recomputed from their files. Snapshot dirs are verified via
    ``verify_snapshot_dir`` and recorded as ``verified`` True/False (check
    failures are RECORDED, not raised — ``assemble_certificate`` decides).
    Suite receipts are ingested from the receipt FILES at
    ``config["suite_receipt_paths"]`` and must carry the R-070 fields
    (see below). config keys: ``repo_root``, ``eligibility_path``,
    ``snapshot_manifest_path``, ``snapshot_dirs`` (role -> Path),
    ``parity_anchor_path``, ``qa012_manifest_path``, ``staged_plan``,
    ``suite_receipt_paths`` ({"focused","full"} -> Path),
    ``content_hash_paths``, ``environment``, ``offline_flags``.
  - ``assemble_certificate(components: dict) -> dict``: pure core of the
    PRE_RUN_READY generator. Required component keys:
    {"repo","content_hashes","eligibility","snapshots","offline_flags",
    "staged_inputs","suite_receipts","parity","qa012","environment"}.
    Emits ``{"schema_version": 2, "ready": <bool>, "failing_checks":
    [<str>...], "components": <the components>}`` (extra keys allowed).
    ``ready`` is True (identity) ONLY when every check passes:
    repo.dirty is exactly False; every staged input observed==expected
    (present, hex-equal); every snapshot entry verified is exactly True;
    both suite receipts exit_code exactly int 0 (False/True rejected);
    offline_flags == the two required flags; every required component and
    every required environment field present. Any defect => ``ready`` is
    False and ``failing_checks`` names EVERY failing component (substring:
    the component key appears in at least one failing-check string) —
    never a partial pass, never an exception.

``reproducibility.colm_aims_2026.phase4_records`` (new module):
  - ``map_calibration_label(label) -> str``: "performat"->"format_specific";
    "shared"->"shared"; "format_specific"->"format_specific"; anything else
    raises.
  - ``export_records(scored_items: list[dict], cell_id: str, out_dir: Path)
    -> Path``: writes ``out_dir / "records" / f"{cell_id}.jsonl"``. Input
    items are EXACTLY ``{"item_key", "horizon", "mc_stop", "ref_stop"}``
    (unknown/missing keys refuse). ``stop == horizon`` (exactly) is the DP
    sentinel (``timeout_coded_as_horizon``): emit ``NEVER_STOPPED`` with
    ``stop_step=None`` and ``terminal_imputation="FINAL_PREFIX_IF_NEVER"``;
    a stop < horizon emits ``FINITE_STOP`` with the integer stop and
    imputation ``NONE`` (R-046 keeps the derived scalar distinct — it is
    recomputable via ``pairing.sentinel_coded_stop`` and never stored);
    ``stop > horizon`` is UNREACHABLE from the DP and is REFUSED as frame
    corruption (amended R-080 — never absorbed into the weaker
    NEVER_STOPPED bucket). Refusals (typed error): any cell_id containing
    the legacy "performat" label or the legacy "+" separator (the error
    message names "format_specific"); duplicate item keys;
    bool/float/negative stops; bool horizons; horizon < 2; stop > horizon.
    Output rows are sorted ascending by UTF-8 item_key and byte-identical
    under input permutation.

``scripts.stopdff_fair_qa_retest`` (producer, F-2/F-6 seams):
  - ``run_phase4_gates(args_like, sentinels=None)``: the gate-ordering
    seam. With ``sentinels`` given (a dict keyed EXACTLY
    {"staged_gate", "eligibility_load", "snapshot_verify", "dataset_load",
    "model_construct"}), the injected callables REPLACE the stage
    implementations and are invoked in exactly that order; a stage callable
    raising aborts the run BEFORE any later stage fires (fail-closed gate
    ordering, R-076).
  - ``phase4_metadata_block(...)``: pure builder for the phase4 metadata
    block; keyword args ``interpreter_realpath``, ``os_name``, ``arch``,
    ``device``, ``pythonhashseed``, ``seeds``, ``offline_flags_set``,
    ``fitted_platt_digests``, ``continuation_estimator_digests``, optional
    ``staged_receipt`` and ``eligibility``. Output carries verbatim
    ``fitted_platt_digests`` + ``continuation_estimator_digests`` keys plus
    the environment/rng fields (``archived_rng_pinned`` is False,
    ``fresh_rng_pinned`` is True), ``staged_inputs`` when a receipt is
    given, and the two eligibility digests when the artifact is given.
  - ``--records-out`` REQUIRES ``--eligibility`` at ARGUMENT validation
    (SystemExit 2 before any gate or load) — records regenerated outside
    the frozen paired population are unusable (F-6 flag coupling).

R-070/R-082 suite receipts (assemble_certificate + gatherer): each receipt
must carry ``exit_code`` (exact int 0), ``command``,
``environment_lock_sha256`` (64-hex), ``workflow_sha256``,
``interpreter_realpath``, ``counts``, ``skip_identities`` — and, per the
R-082 operational-rejection repair, HEAD BINDINGS: ``commit`` ==
repo.commit, ``tree_sha256`` == repo.tree_sha256, ``dirty`` identically
False, plus ``counts.failures`` and ``counts.errors`` each exactly int 0
(bools rejected). Any missing or mismatching binding is a named failing
suite_receipts component (receipts must come from EXECUTING the suites at
the certified head, never from pre-existing files — the head binding is
what makes stale ingestion detectable). The gatherer records staged-input
paths as ABSOLUTE strings (out-of-repo staging, R-082).

Amended R-077 (Random-K STRUCTURE, operational-rejection repair): both
krandom cells must be PRESENT in the regenerated export with the full
10-point + 2-CI field set. A missing krandom cell (failure row field
``"<cell>"``) or a missing krandom field is a blocking STRUCTURAL failure
row; only the numeric VALUES stay exempt (never compared, reported
informationally). Structural checks do NOT increment ``checked`` — a PASS
still carries exactly ``checked == 194``.

Amended R-080 path discipline: ``export_records``'s ``out_dir`` is the
PARENT directory — the exporter owns the ``records/`` segment. An
``out_dir`` whose final path component is exactly ``records`` is REFUSED
(fail-loud; the P1-6 doubled-segment argv class), naming the doubled
segment; names merely containing "records" (e.g. ``records_v2``) are fine.

R-072 rev3: rev2 (`52ac2902…`) was found defective at readback (sha256-only
entries; 0-based hit pointers). The rev3 manifest at repo root
``qa012_inventory_2026-08-22_rev3.json`` (landed mid-round; the REAL
artifact's field names govern) corrects both over the same 67-file scope:
every entry carries BOTH ``dropbox_content_hash`` (4 MiB-block convention)
AND ``sha256``; ``conventions.jsonl_line_numbers == "1-based"`` declared
top-level (no ``line 0:`` pointer anywhere); ``revision == 3``; the
in-manifest ``supersession_chain`` names BOTH rev1's and rev2's SHA-256
with defect notes; totals unchanged (67 scanned / 4,556 hits).

``reproducibility.colm_aims_2026.phase4_launcher`` (new module, R-081):
  - ``LaunchRefusal`` / ``RunFailed``: ``schema.ColmAimsError`` subclasses
    (pre-launch refusals vs post-launch failures).
  - ``validate_and_launch(config, *, run_git=None, launch=None,
    compare=None, now=None)``: single-use launcher. ``run_git(cmd)->stdout``
    and ``launch(argv, env)->exit_code`` are injectable; ``compare`` is the
    injectable comparator hook ``compare(quarantine_dir) -> dict``
    returning a compare_parity-shaped result (the default loads the anchor
    from ``config["anchor_path"]`` and the regenerated export from the
    quarantine). config keys: ``certificate_path``, ``activation_digest``,
    ``quarantine_dir``, ``promote_to``, ``ledger_path``,
    ``snapshot_manifest_path``, ``snapshot_dirs``, ``anchor_path``.
    Pre-launch refusal classes (each a ``LaunchRefusal`` whose message
    names the class; ``launch`` is NEVER invoked and the ledger is NEVER
    newly created on any refusal — refusals are side-effect-free):
    (1) certificate bytes sha256 != activation_digest — checked FIRST,
    before parsing/ready; (2) ``ready`` not identically True (bool-safe:
    ``1`` refuses); (3) live ``git rev-parse HEAD`` != certificate commit;
    (4) live ``rev-parse HEAD^{tree}`` != certificate tree; (5) live
    TRACKED-dirty status (untracked-only ``?? ...`` porcelain lines do NOT
    refuse — the tracked-clean + untracked-disclosure convention);
    (6) ``MODAL_HOST_GIT_STATUS`` or ``MODAL_HOST_GIT_COMMIT`` present in
    os.environ AT ALL (even empty) — ambient provenance laundering;
    (7) snapshot re-verification failure against the frozen manifest;
    (R-082/F-1, PRE-LEDGER) any staged path — from BOTH the certificate's
    ``staged_inputs`` component AND the composed argv (``--staged-input``
    values, the ``--calibration`` value; relative forms resolved against
    the repo root) — that ``schema.resolves_inside`` the repository tree
    refuses, naming R-082/P0-1 (an in-repo untracked staged file would
    burn the ledger post-scoring — rejected certificate 8731ad00's
    defect); (9-then-8, F-2 ordering) the WORKSPACE is materialized before
    the ledger: pre-existing ``promote_to``, missing promote parent,
    pre-existing ``quarantine_dir`` (mkdir exist_ok=False doubles as the
    staleness check), unwritable quarantine, and cross-device
    quarantine/promote (os.rename cannot be atomic) all refuse
    PRE-LEDGER; only then is the ledger consumed via O_CREAT|O_EXCL
    (recording the activation digest BEFORE launch), and a
    ledger-already-exists refusal rmdirs the just-created quarantine so
    the refusal leaves no workspace behind. The three workspace config
    paths (``quarantine_dir``, ``promote_to``, ``ledger_path``) are
    ``.resolve()``d at entry (F-4: a relative path must not split between
    the launcher cwd and the child cwd). Success path: ``launch`` called
    EXACTLY once; env carries ``PYTHONHASHSEED=0``, ``OMP_NUM_THREADS=1``,
    ``VECLIB_MAXIMUM_THREADS=1``, ``HF_HUB_OFFLINE=1``,
    ``TRANSFORMERS_OFFLINE=1`` and NO ``MODAL_HOST_*`` key; argv is
    composed FROM the certificate's recorded command
    (components.environment.command) with ONLY output paths remapped —
    ``--out`` value to ``quarantine_dir/<basename>``, ``--records-out``
    value to ``quarantine_dir`` itself (the R-080 parent) — plus an
    appended ``--certificate-digest <activation_digest>``; all other
    tokens (including ``--calibration``/``--staged-input`` values, which
    must point OUTSIDE the repo) are preserved verbatim. Post-launch
    (F-3: the ledger is consumed, so the triage artifact must exist on
    the messiest failures): nonzero exit -> ``RunFailed`` + STOP report
    at ``quarantine_dir / "STOP_REPORT.json"`` (reason ``nonzero_exit``,
    contains the activation digest), quarantine intact, ``promote_to``
    absent; a CRASH inside ``launch``/``compare`` -> ``RunFailed`` + STOP
    report with reason ``launch_crash``/``comparator_crash``; zero exit
    -> comparator invoked MANDATORILY; PASS -> single atomic rename
    quarantine -> promote_to (quarantine gone, contents preserved);
    comparator FAIL -> ``RunFailed`` + STOP report (reason
    ``parity_comparator_fail``) + quarantine intact + ``promote_to``
    absent.

Verifier-side (R-073, existing modules):
  - ``schema.TIMEOUT_PARAMETER_KEYS == {"horizon_map_sha256", "rule"}``
    (scalar ``trajectory_horizon`` RETIRED).
  - ``schema.horizon_map_sha256`` gains domain guards: non-string keys,
    bool/non-int/non-positive values, and the empty map all raise.
  - ``"SINGLE_PREFIX_TRAJECTORY" in schema.EXCLUSION_REASONS`` (R-074).
  - New verifier legs (ids pinned here): ``horizon_map_declaration``
    (recomputed per-cell records digest == timeout_parameters pin) and
    ``horizon_map_cross_cell`` (recomputed digest equal across all ten
    cells); the recompute-vs-held-fixed comparison stays on the existing
    ``grid_held_fixed_identities`` leg. In release mode the held-fixed
    horizon_identity is additionally pinned by expectations
    (``anchored_grid_held_fixed``, existing R-044 leg).
"""
from __future__ import annotations

import functools
import hashlib
import json
import os
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from reproducibility.colm_aims_2026 import pairing, schema

from tests._colm_aims_v2_helpers import (
    CALIBRATION_IDS,
    CELL_IDS,
    EVENT_FINITE,
    EVENT_NEVER,
    FINITE_ONLY_ESTIMAND_LABEL,
    IMPUTATION_FINAL_PREFIX,
    IMPUTATION_NONE,
    N_ITEMS,
    POPULATION_FINITE,
    REFERENCE_IDS,
    REPO_ROOT,
    TRAJECTORY_HORIZON,
    VERDICT_FAIL,
    VERDICT_RELEASE_PASS,
    VERDICT_SOURCE_PASS,
    assert_failing_leg,
    assert_passing_report,
    build_package_v2,
    canonical_data,
    colm_no_network,  # noqa: F401 - autouse fixture
    d7b_holm,
    d7b_interval,
    d7b_p_value,
    d7b_resample_matrix,
    expected_estimand_digest,
    failing_leg_ids,
    leg_by_id,
    make_record_v2,
    release_report,
    sha256_bytes,
    sha256_file,
    source_report,
)

# ---------------------------------------------------------------------------
# Pinned constants (frozen artifacts + adjudicated hashes)
# ---------------------------------------------------------------------------

FROZEN_DIR = REPO_ROOT / "reproducibility" / "colm_aims_2026" / "frozen"
ELIGIBILITY_PATH = FROZEN_DIR / "pairing_eligibility_v2.json"
MODEL_MANIFEST_PATH = FROZEN_DIR / "model_snapshot_manifests.json"
PARITY_ANCHOR_PATH = FROZEN_DIR / "parity_anchor_export_a.json"
QA012_FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "qa012_item10"
QA012_BINDINGS_PATH = QA012_FIXTURE_DIR / "bindings.json"
QA012_REV2_MANIFEST_PATH = REPO_ROOT / "qa012_inventory_2026-08-22_rev2.json"
QA012_REV3_MANIFEST_PATH = REPO_ROOT / "qa012_inventory_2026-08-22_rev3.json"

# R-074 (PRE-1): two-party pinned test-dataset digest.
TEST_DATASET_SHA = (
    "638a4df978b77a12655ea72d56daad7fa70851ae486ddb4365d9b060549e34f1"
)
# R-076 (PRE-4): archival calibration_train.json digest.
CALIB_TRAIN_SHA = (
    "745bd67597278bd9d24d41c1dea53bf3a7c56cd6334cfc07ea62bccbdcf44259"
)
# R-077 (PRE-6): Export A anchor / Export B corroborative digests.
EXPORT_A_SHA = (
    "59e1c1a74e5fc0cf4f09f8befca87cfc81516684dca2e88dd275c952b28893ff"
)
EXPORT_B_SHA = (
    "ba784741ea5f472db50bea7cf24de5ee8eb567e4690c0f73a5e056fb0691a5f9"
)
# R-072 (amended): QA-012 rev2 manifest + superseded rev1 digests.
QA012_REV2_SHA = (
    "52ac29026beb77a93aae3ce7694c2f8ae0b60bd8a3ad2f97aa505f167e28e06c"
)
QA012_REV1_SHA = (
    "149fe39cfe99a0ee69ea844ca2712bb79069bb65215ec7faca240fff41240187"
)
# R-078 (PRE-7): the four full hit files, bound by SHA-256.
QA012_FULL_FILE_SHAS = frozenset(
    {
        "32ecda092990c8672ee31ebcc743af446486fc58a2d8679bee38d76a0a99c8da",
        "8f38ef3f93f9caaa6889bdb1b247594bad7570e60bfbb6e60007998a70fef7f8",
        "c3aa63085ad991bfd243a240f0255737cec213d99b1afc9652bd02e96da896ea",
        "f7dcb43bd1a3599062d9ad05cfe0c0d4b5d2745b4b3807fe14c2932aa85b07a3",
    }
)
# R-074: the 9 excluded qids, each SINGLE_PREFIX_TRAJECTORY.
EXCLUDED_QIDS = (
    "103295",
    "119618",
    "190798",
    "191687",
    "196619",
    "197040",
    "206660",
    "207981",
    "209745",
)
# R-075 (PRE-3): role-keyed model identities.
PRIMARY_SCORER_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DISJOINT_SELECTOR_NAME = "sentence-transformers/all-mpnet-base-v2"
OFFLINE_FLAGS = ["HF_HUB_OFFLINE=1", "TRANSFORMERS_OFFLINE=1"]
TFIDF_CONFIG = {
    "analyzer": "char_wb",
    "ngram_range": [2, 4],
    "fit_corpus": "answer pool",
}

RULE_TOKEN = "zero_indexed_stop_ge_horizon_is_timeout"

# New/existing verifier leg ids exercised here (see module docstring).
LEG_HORIZON_DECLARATION = "horizon_map_declaration"
LEG_HORIZON_CROSS_CELL = "horizon_map_cross_cell"
LEG_HELD_FIXED = "grid_held_fixed_identities"
LEG_RECORD_VALIDATION = "record_validation"
ANCHORED_GRID_HELD_FIXED = "anchored_grid_held_fixed"

# assemble_certificate required component keys (contract, module docstring).
CERT_COMPONENT_KEYS = frozenset(
    {
        "repo",
        "content_hashes",
        "eligibility",
        "snapshots",
        "offline_flags",
        "staged_inputs",
        "suite_receipts",
        "parity",
        "qa012",
        "environment",
    }
)
CERT_ENVIRONMENT_KEYS = frozenset(
    {
        "interpreter_realpath",
        "os",
        "arch",
        "cpu",
        "blas",
        "thread_settings",
        "environment_lock_sha256",
        "command",
        "seeds",
        "pythonhashseed",
        "archived_rng_pinned",
        "fresh_rng_pinned",
        "quarantine_dir",
        "promote_to",
        "exception_ledger_path",
    }
)

# Digest-function guard failures may be any typed family member (GREEN picks
# the concrete class; a silent coerced digest is the defect under test).
DIGEST_GUARD_ERRORS = (schema.ColmAimsError, TypeError, ValueError)


# ---------------------------------------------------------------------------
# Lazy imports for the GREEN modules (ImportError == correct RED failure;
# module-level import would wrongly kill the artifact-pin tests too).
# ---------------------------------------------------------------------------


def _phase4():
    from reproducibility.colm_aims_2026 import phase4

    return phase4


def _phase4_records():
    from reproducibility.colm_aims_2026 import phase4_records

    return phase4_records


def _load_json(path):
    return json.loads(path.read_text("utf-8"))


def _copy(obj):
    return json.loads(json.dumps(obj))


# ===========================================================================
# R-073: canonical horizon-map digest + retired scalar horizon
# ===========================================================================


class TestR073DigestFunction:
    def test_known_answer_digest(self):
        # Tests R-073 [unit]: hand-computed KAT — sorted keys, compact
        # separators, UTF-8, lowercase-hex sha256.
        expected = hashlib.sha256(b'{"a":2,"b":10}').hexdigest()
        assert schema.horizon_map_sha256({"a": 2, "b": 10}) == expected

    def test_key_insertion_order_is_irrelevant(self):
        # Tests R-073 [unit]: serialization sorts keys ascending by UTF-8
        # byte order — insertion order must not leak into the digest.
        assert schema.horizon_map_sha256(
            {"b": 10, "a": 2}
        ) == schema.horizon_map_sha256({"a": 2, "b": 10})

    def test_frozen_artifact_digest_reproduces(self):
        # Tests R-073/R-074 [unit]: the committed artifact's declared digest
        # is exactly the canonical function over its own horizon_map.
        # Source: reproducibility/colm_aims_2026/frozen/pairing_eligibility_v2.json
        art = _load_json(ELIGIBILITY_PATH)
        assert (
            schema.horizon_map_sha256(art["horizon_map"])
            == art["horizon_map_sha256"]
        )

    def test_bool_horizon_value_raises_not_laundered(self):
        # Tests R-073 [unit]: True must never be digested as 1 (bool
        # laundering — seed catalog).
        with pytest.raises(DIGEST_GUARD_ERRORS):
            schema.horizon_map_sha256({"a": True})

    def test_non_integer_horizon_value_raises_not_truncated(self):
        # Tests R-073 [unit]: 2.5 must never be coerced/truncated to 2
        # (coercion-inside-comparison laundering — seed catalog).
        with pytest.raises(DIGEST_GUARD_ERRORS):
            schema.horizon_map_sha256({"a": 2.5})

    def test_integer_valued_float_raises(self):
        # Tests R-073 [unit]: the domain is positive INT — 2.0 is not in it.
        with pytest.raises(DIGEST_GUARD_ERRORS):
            schema.horizon_map_sha256({"a": 2.0})

    def test_non_positive_horizon_raises(self):
        # Tests R-073 [unit]: horizons are positive integers.
        with pytest.raises(DIGEST_GUARD_ERRORS):
            schema.horizon_map_sha256({"a": 0})

    def test_non_string_key_raises_not_stringified(self):
        # Tests R-073 [unit]: item keys are strings; int keys must not be
        # silently str()-ed into the digest domain.
        with pytest.raises(DIGEST_GUARD_ERRORS):
            schema.horizon_map_sha256({1: 2})

    def test_empty_map_raises(self):
        # Tests R-073 [unit]: a vacuously-empty horizon map is a defect,
        # never a valid digestible identity (seed catalog: vacuously-empty
        # authoritative sets).
        with pytest.raises(DIGEST_GUARD_ERRORS):
            schema.horizon_map_sha256({})

    def test_timeout_parameter_keys_closed_set(self):
        # Tests R-073 [unit]: scalar trajectory_horizon is RETIRED from the
        # closed estimand.timeout_parameters key set.
        assert schema.TIMEOUT_PARAMETER_KEYS == frozenset(
            {"horizon_map_sha256", "rule"}
        )


# ---------------------------------------------------------------------------
# R-073 heterogeneous-horizon package fixtures (local builders ONLY — the
# shared helpers keep their uniform-scalar builder; we override via the
# documented build_package_v2 mutator hooks).
#
# Generative arithmetic (i = ascending-UTF-8 rank of the item key):
#   h(i)                 = 2 + ((i * 7) % 9)             -> spans 2..10
#   item 0 (h = 2)       : mc FINITE stop 1 (== h-1, genuine final-prefix
#                          crossing), ref FINITE stop 0, in EVERY cell
#   mc_never(i, c)       = ((i + 11c) % 17) == 5          (i >= 1)
#   ref_never(i, r, c)   = ((i + 3r + 13c) % 19) == 7     (i >= 1)
#   mc_stop(i, c)        = (7i + 3 + 11c) % h(i)          < h(i)
#   ref_stop(i, r, c)    = (5i + 1 + 3r + 13c) % h(i)     < h(i)
# MC events depend only on (item, calibration): the R-043 within-calibration
# equality holds while calibrations differ (nearest-true control carries).
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def _hetero_data():
    base = canonical_data()
    keys = base["keys"]
    indices = d7b_resample_matrix(base["seed"])
    idx = np.arange(N_ITEMS, dtype=np.int64)
    h = 2 + ((idx * 7) % 9)
    horizon_of = {key: int(h[i]) for i, key in enumerate(keys)}
    # Fixture-integrity guards: full 2..10 span, item 0 at the minimum.
    assert set(horizon_of.values()) == set(range(2, 11))
    assert horizon_of[keys[0]] == 2
    digest = schema.horizon_map_sha256(horizon_of)

    cells: dict[str, dict] = {}
    raw_p: dict[str, float] = {}
    for cell_id in CELL_IDS:
        ref_id, cal_id = cell_id.split("__", 1)
        r = REFERENCE_IDS.index(ref_id)
        c = CALIBRATION_IDS.index(cal_id)
        mc_never = ((idx + 11 * c) % 17) == 5
        ref_never = ((idx + 3 * r + 13 * c) % 19) == 7
        mc_stop = (7 * idx + 3 + 11 * c) % h
        ref_stop = (5 * idx + 1 + 3 * r + 13 * c) % h
        # Item 0: deterministic final-prefix crossing at horizon 2.
        mc_never[0] = False
        ref_never[0] = False
        mc_stop[0] = 1
        ref_stop[0] = 0
        mc_fin = ~mc_never
        ref_fin = ~ref_never
        assert bool(np.any(~mc_fin)) and bool(np.any(~ref_fin)), (
            "hetero fixture must exercise NEVER_STOPPED sentinel coding"
        )
        s_mc = np.where(mc_fin, mc_stop, h)
        s_ref = np.where(ref_fin, ref_stop, h)
        d = (s_mc - s_ref).astype(np.int64)
        n_bf = int(np.sum(mc_fin & ref_fin))
        assert n_bf > 0
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
            "rate_mc_finite_ref_timeout": (
                counts["n_mc_finite_ref_timeout"] / N_ITEMS
            ),
            "rate_mc_timeout_ref_finite": (
                counts["n_mc_timeout_ref_finite"] / N_ITEMS
            ),
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
        records = [
            make_record_v2(
                key,
                int(mc_stop[i]) if mc_fin[i] else None,
                int(ref_stop[i]) if ref_fin[i] else None,
                trajectory_horizon=int(h[i]),
            )
            for i, key in enumerate(keys)
        ]
        raw_p[cell_id] = d7b_p_value(df, indices)
        cells[cell_id] = {
            "records": records,
            "counts": counts,
            "rates": rates,
            "headline_mean": float(np.mean(df)),
            "finite_only": finite_only,
            "ci": d7b_interval(df, indices),
            "raw_p": raw_p[cell_id],
        }
    return {
        "keys": keys,
        "horizon_of": horizon_of,
        "digest": digest,
        "cells": cells,
        "holm": d7b_holm(raw_p),
    }


def _records_blob(records) -> bytes:
    return (
        "\n".join(json.dumps(r, sort_keys=True) for r in records) + "\n"
    ).encode("utf-8")


def _horizon_profile_mutator(digest, cells_numeric=None, holm=None):
    def mutate(profile):
        profile["grid"]["held_fixed"]["horizon_identity"] = digest
        if holm is not None:
            profile["inference"]["ordered_family"] = list(
                holm["ordered_family"]
            )
            profile["inference"]["rejected_cell_ids"] = list(
                holm["rejected_cell_ids"]
            )
        for cell in profile["cells"]:
            cid = cell["cell_id"]
            est = cell["estimand"]
            est["timeout_parameters"] = {
                "horizon_map_sha256": digest,
                "rule": RULE_TOKEN,
            }
            est["event_representation"]["horizon_identity"] = digest
            cell["estimand_digest"] = expected_estimand_digest(est)
            if cells_numeric is None:
                continue
            num = cells_numeric[cid]
            cell["counts"] = dict(num["counts"])
            cell["rates"] = dict(num["rates"])
            cell["headline_summary"]["n"] = num["counts"]["n_complete"]
            cell["headline_summary"]["mean_signed_shift"] = num[
                "headline_mean"
            ]
            cell["finite_only_summary"] = {
                "estimand_label": FINITE_ONLY_ESTIMAND_LABEL,
                "population": POPULATION_FINITE,
                **num["finite_only"],
            }
            cell["interval"]["ci"] = [num["ci"][0], num["ci"][1]]
            cell["raw_p_value"] = num["raw_p"]
            hc = holm["per_cell"][cid]
            cell["holm_rank"] = hc["holm_rank"]
            cell["holm_adjusted_p_value"] = hc["holm_adjusted_p_value"]
            cell["holm_rejected"] = hc["holm_rejected"]

    return mutate


def _build_hetero_package(tmp_path, *, override=None, expectations_pin=None):
    """Full-size hetero-horizon package; ``override`` mutates ONE record in
    ONE cell pre-serialization: (cell_id, item_index, field, value)."""
    data = _hetero_data()
    raw: dict[str, bytes] = {}
    for cid in CELL_IDS:
        records = [dict(r) for r in data["cells"][cid]["records"]]
        if override is not None and override[0] == cid:
            _, item_index, field, value = override
            records[item_index][field] = value
        raw[cid] = _records_blob(records)
    digest = data["digest"]
    pin = expectations_pin if expectations_pin is not None else digest

    def expectations_mutator(exp):
        exp["bindings"]["grid"]["held_fixed"]["horizon_identity"] = pin

    return build_package_v2(
        tmp_path,
        raw_records_bytes=raw,
        profile_mutator=_horizon_profile_mutator(
            digest, data["cells"], data["holm"]
        ),
        expectations_mutator=expectations_mutator,
    )


def _build_uniform_map_package(tmp_path):
    """Nearest-true: canonical uniform horizons (all 6) declared through the
    NEW horizon-map contract — numeric blocks stay canonical."""
    keys = canonical_data()["keys"]
    digest = schema.horizon_map_sha256(
        {key: TRAJECTORY_HORIZON for key in keys}
    )

    def expectations_mutator(exp):
        exp["bindings"]["grid"]["held_fixed"]["horizon_identity"] = digest

    return build_package_v2(
        tmp_path,
        profile_mutator=_horizon_profile_mutator(digest),
        expectations_mutator=expectations_mutator,
    )


class TestR073HorizonLegs:
    def test_heterogeneous_horizons_pass_source_mode(self, tmp_path):
        # Tests R-073/R-043/R-045 [integration]: horizons spanning 2..10,
        # per-record ownership, digest declared in timeout_parameters and
        # held fixed as horizon_identity => PASS.
        pkg = _build_hetero_package(tmp_path)
        report = source_report(pkg)
        assert_passing_report(report, VERDICT_SOURCE_PASS)

    def test_uniform_horizons_nearest_true_passes(self, tmp_path):
        # Tests R-073 [integration]: an all-equal-horizons package under the
        # SAME map contract passes — heterogeneity is legal, not mandatory.
        pkg = _build_uniform_map_package(tmp_path)
        report = source_report(pkg)
        assert_passing_report(report, VERDICT_SOURCE_PASS)

    def test_retired_scalar_horizon_declaration_fails(self, tmp_path):
        # Tests R-073 [integration]: the OLD scalar timeout_parameters shape
        # ({trajectory_horizon, rule}) is no longer a valid declaration —
        # forced explicitly here so the fixture stays a scalar-shape package
        # even after the shared helpers migrate to the map contract.
        def mutate(profile):
            for cell in profile["cells"]:
                est = cell["estimand"]
                est["timeout_parameters"] = {
                    "trajectory_horizon": TRAJECTORY_HORIZON,
                    "rule": RULE_TOKEN,
                }
                cell["estimand_digest"] = expected_estimand_digest(est)

        pkg = build_package_v2(tmp_path, profile_mutator=mutate)
        report = source_report(pkg)
        assert report.verdict == VERDICT_FAIL

    def test_single_item_horizon_mutation_fails_all_three_legs(
        self, tmp_path
    ):
        # Tests R-073 [integration] (substitution-negative): ONE item's
        # horizon in ONE cell changed to a DIFFERENT VALID value that leaves
        # classification and every derived number unchanged (both arms
        # finite, stops < both horizons) — the recomputed digest is the ONLY
        # discriminant, and all three comparison legs must fire.
        data = _hetero_data()
        keys = data["keys"]
        victim_cell = "khard__shared"
        rec = data["cells"][victim_cell]["records"][1]
        assert rec["mc_event_status"] == EVENT_FINITE
        assert rec["ref_event_status"] == EVENT_FINITE
        assert rec["trajectory_horizon"] == 9
        assert rec["mc_stop_step"] < 10 and rec["ref_stop_step"] < 10
        pkg = _build_hetero_package(
            tmp_path, override=(victim_cell, 1, "trajectory_horizon", 10)
        )
        report = source_report(pkg)
        assert report.verdict == VERDICT_FAIL
        assert set(failing_leg_ids(report)) == {
            LEG_HORIZON_DECLARATION,
            LEG_HELD_FIXED,
            LEG_HORIZON_CROSS_CELL,
        }
        # Receipt discipline: the declaration leg carries the RECOMPUTED
        # digest of the mutated map (recompute-from-records, never an echo
        # of the declaration — mirror-equality catcher).
        mutated_digest = schema.horizon_map_sha256(
            {**data["horizon_of"], keys[1]: 10}
        )
        leg = leg_by_id(report, LEG_HORIZON_DECLARATION)
        assert mutated_digest in json.dumps(leg)

    def test_bool_horizon_record_fails_int_domain_never_one(self, tmp_path):
        # Tests R-073/R-061 [integration]: trajectory_horizon=True in one
        # record is a typed int-domain rejection — never laundered as 1,
        # never an unhandled crash (guarded legs).
        pkg = _build_hetero_package(
            tmp_path,
            override=("khard__shared", 1, "trajectory_horizon", True),
        )
        report = source_report(pkg)
        assert report.verdict == VERDICT_FAIL
        leg = assert_failing_leg(report, LEG_RECORD_VALIDATION)
        assert "R-061" in json.dumps(leg)
        decl = leg_by_id(report, LEG_HORIZON_DECLARATION)
        assert decl is None or decl.get("status") != "PASS"

    def test_release_mode_pins_horizon_identity(self, tmp_path):
        # Tests R-073/R-044 [integration]: the hetero package passes release
        # mode when the expectations pin equals the canonical digest.
        pkg = _build_hetero_package(tmp_path)
        report = release_report(pkg)
        assert_passing_report(report, VERDICT_RELEASE_PASS)

    def test_release_expectations_pin_substitution_fails(self, tmp_path):
        # Tests R-073/R-044 [integration] (substitution-negative): the
        # expectations horizon pin replaced by a DIFFERENT VALID digest must
        # fail the anchored held-fixed leg.
        data = _hetero_data()
        other = schema.horizon_map_sha256(
            {**data["horizon_of"], data["keys"][0]: 3}
        )
        assert other != data["digest"]
        pkg = _build_hetero_package(tmp_path, expectations_pin=other)
        report = release_report(pkg)
        assert_failing_leg(report, ANCHORED_GRID_HELD_FIXED)


class TestR046FinalPrefixShortHorizon:
    # R-046 carried behavior at the NEW short horizons (unit pins — these
    # pass today and must keep passing after the R-073 rewiring).

    def test_finite_stop_at_final_prefix_of_horizon_two_validates(self):
        # Tests R-046 [unit]: a genuine crossing at stop == horizon-1 == 1
        # is FINITE_STOP, legal, and classifies as a complete both-finite
        # pair.
        rec = make_record_v2("itm-final", 1, 0, trajectory_horizon=2)
        schema.validate_record(rec)
        outcome = pairing.classify_record(rec)
        assert outcome["status"] == "complete"
        assert outcome["joint_class"] == "both_finite"
        assert pairing.sentinel_coded_stop(rec, "mc") == 1

    def test_never_stopped_at_horizon_two_sentinel_codes_to_two(self):
        # Tests R-046 [unit]: the DERIVED scalar for NEVER at horizon 2 is
        # the horizon (2) while the canonical stop stays null — distinct
        # encodings.
        rec = make_record_v2("itm-nvr", None, 0, trajectory_horizon=2)
        schema.validate_record(rec)
        assert rec["mc_stop_step"] is None
        assert pairing.sentinel_coded_stop(rec, "mc") == 2

    def test_finite_stop_at_horizon_is_still_rejected(self):
        # Tests R-045/R-061 [unit]: stop == horizon remains the old sentinel
        # coding and is illegal in the canonical representation.
        rec = make_record_v2("itm-bad", 2, 0, trajectory_horizon=2)
        with pytest.raises(schema.RecordValidationError):
            schema.validate_record(rec)


# ===========================================================================
# R-074: frozen pairing eligibility — artifact pins + typed loader
# ===========================================================================


class TestR074EligibilityArtifact:
    # Direct pins over the COMMITTED artifact (these must pass at RED time).
    # Source: reproducibility/colm_aims_2026/frozen/pairing_eligibility_v2.json

    @pytest.fixture(scope="class")
    def art(self):
        return _load_json(ELIGIBILITY_PATH)

    def test_counts(self, art):
        assert art["eligible_count"] == 2249
        assert art["excluded_count"] == 9
        assert len(art["eligible_keys"]) == 2249
        assert len(art["excluded"]) == 9

    def test_exact_excluded_qids_and_reason(self, art):
        assert [e["item_key"] for e in art["excluded"]] == list(EXCLUDED_QIDS)
        assert all(
            e["reason"] == "SINGLE_PREFIX_TRAJECTORY" for e in art["excluded"]
        )

    def test_single_prefix_trajectory_is_enum_member(self):
        # Tests R-074 [unit]: the reason is a spec-pinned NEW member of the
        # closed exclusion-reason enum.
        assert "SINGLE_PREFIX_TRAJECTORY" in schema.EXCLUSION_REASONS

    def test_keyset_digest_recomputes_via_pairing_keyset_sha256(self, art):
        # Tests R-074 [unit]: the D7(b) seed input digest is EXACTLY
        # pairing.keyset_sha256 over the eligible keys (sorted,
        # newline-joined) — recompute, never trust.
        assert (
            pairing.keyset_sha256(art["eligible_keys"])
            == art["pairing_population_keyset_sha256"]
        )

    def test_eligible_keys_sorted_and_duplicate_free(self, art):
        keys = art["eligible_keys"]
        assert keys == sorted(keys)
        assert len(set(keys)) == len(keys)
        assert not (set(keys) & {e["item_key"] for e in art["excluded"]})

    def test_horizon_map_covers_exactly_the_eligible_keys(self, art):
        assert set(art["horizon_map"]) == set(art["eligible_keys"])

    def test_horizon_values_are_real_ints_in_2_to_10(self, art):
        for value in art["horizon_map"].values():
            assert isinstance(value, int) and not isinstance(value, bool)
            assert 2 <= value <= 10

    def test_horizon_map_digest_recomputes(self, art):
        assert (
            schema.horizon_map_sha256(art["horizon_map"])
            == art["horizon_map_sha256"]
        )

    def test_derivation_provenance_pins_test_dataset(self, art):
        assert (
            art["derived_from"]["test_dataset_sha256"] == TEST_DATASET_SHA
        )


class TestR074LoadPairingEligibility:
    # Typed loader contract (GREEN: phase4.load_pairing_eligibility).

    def _tampered(self, tmp_path, mutate):
        art = _load_json(ELIGIBILITY_PATH)
        mutate(art)
        path = tmp_path / "pairing_eligibility_v2.json"
        path.write_text(json.dumps(art), encoding="utf-8")
        return path

    def test_happy_path_returns_validated_artifact(self):
        loaded = _phase4().load_pairing_eligibility(ELIGIBILITY_PATH)
        assert loaded["eligible_count"] == 2249
        assert loaded["horizon_map_sha256"] == schema.horizon_map_sha256(
            loaded["horizon_map"]
        )

    def test_dropped_eligible_key_raises(self, tmp_path):
        # Digest recompute over records-side truth: removing one key (counts
        # left stale) must raise.
        def mutate(art):
            art["eligible_keys"] = art["eligible_keys"][:-1]

        path = self._tampered(tmp_path, mutate)
        with pytest.raises(schema.TypedIngressError):
            _phase4().load_pairing_eligibility(path)

    def test_mutated_horizon_value_raises(self, tmp_path):
        # Substitution-negative: 5 -> 6 is a VALID horizon; the recomputed
        # map digest no longer matches the declaration => typed error.
        def mutate(art):
            key = art["eligible_keys"][0]
            art["horizon_map"][key] = (
                6 if art["horizon_map"][key] != 6 else 5
            )

        path = self._tampered(tmp_path, mutate)
        with pytest.raises(schema.TypedIngressError):
            _phase4().load_pairing_eligibility(path)

    def test_substituted_horizon_digest_raises(self, tmp_path):
        # Substitution-negative on the declaration side: a DIFFERENT valid
        # 64-hex digest must not pass shape checks.
        def mutate(art):
            art["horizon_map_sha256"] = "0" * 64

        path = self._tampered(tmp_path, mutate)
        with pytest.raises(schema.TypedIngressError):
            _phase4().load_pairing_eligibility(path)

    def test_substituted_keyset_digest_raises(self, tmp_path):
        def mutate(art):
            art["pairing_population_keyset_sha256"] = "1" * 64

        path = self._tampered(tmp_path, mutate)
        with pytest.raises(schema.TypedIngressError):
            _phase4().load_pairing_eligibility(path)

    def test_unsorted_eligible_keys_raise(self, tmp_path):
        # keyset_sha256 sorts internally, so an out-of-order artifact would
        # still digest-match — sortedness needs its OWN check.
        def mutate(art):
            keys = art["eligible_keys"]
            keys[0], keys[1] = keys[1], keys[0]

        path = self._tampered(tmp_path, mutate)
        with pytest.raises(schema.TypedIngressError):
            _phase4().load_pairing_eligibility(path)

    def test_non_enum_exclusion_reason_raises(self, tmp_path):
        def mutate(art):
            art["excluded"][0]["reason"] = "OTHER"

        path = self._tampered(tmp_path, mutate)
        with pytest.raises(schema.TypedIngressError):
            _phase4().load_pairing_eligibility(path)

    def test_bool_horizon_value_raises(self, tmp_path):
        def mutate(art):
            art["horizon_map"][art["eligible_keys"][0]] = True

        path = self._tampered(tmp_path, mutate)
        with pytest.raises(schema.TypedIngressError):
            _phase4().load_pairing_eligibility(path)

    def test_horizon_below_two_raises(self, tmp_path):
        # DECISION: horizon 1 contradicts the SINGLE_PREFIX_TRAJECTORY
        # exclusion rule that produced this artifact — loader refuses.
        def mutate(art):
            key = art["eligible_keys"][0]
            art["horizon_map"][key] = 1
            art["horizon_map_sha256"] = schema.horizon_map_sha256(
                {**art["horizon_map"], key: 1}
            )

        path = self._tampered(tmp_path, mutate)
        with pytest.raises(schema.TypedIngressError):
            _phase4().load_pairing_eligibility(path)

    def test_count_drift_raises(self, tmp_path):
        def mutate(art):
            art["eligible_count"] = 2250

        path = self._tampered(tmp_path, mutate)
        with pytest.raises(schema.TypedIngressError):
            _phase4().load_pairing_eligibility(path)

    def test_unknown_top_level_key_raises(self, tmp_path):
        def mutate(art):
            art["extra_block"] = {"smuggled": 1}

        path = self._tampered(tmp_path, mutate)
        with pytest.raises(schema.TypedIngressError):
            _phase4().load_pairing_eligibility(path)

    def test_missing_required_key_raises(self, tmp_path):
        def mutate(art):
            del art["excluded"]

        path = self._tampered(tmp_path, mutate)
        with pytest.raises(schema.TypedIngressError):
            _phase4().load_pairing_eligibility(path)

    def test_version_first_bool_schema_version(self, tmp_path):
        # Version-first + bool-safe (seed catalog): schema_version=True must
        # fail AS a version error even with other fields intact.
        def mutate(art):
            art["schema_version"] = True

        path = self._tampered(tmp_path, mutate)
        with pytest.raises(schema.TypedIngressError) as excinfo:
            _phase4().load_pairing_eligibility(path)
        assert "version" in str(excinfo.value).lower()

    def test_version_checked_before_digests(self, tmp_path):
        # Version-first ordering: wrong version + corrupted digest must
        # surface the VERSION error, not the digest error.
        def mutate(art):
            art["schema_version"] = 1
            art["horizon_map_sha256"] = "f" * 64

        path = self._tampered(tmp_path, mutate)
        with pytest.raises(schema.TypedIngressError) as excinfo:
            _phase4().load_pairing_eligibility(path)
        assert "version" in str(excinfo.value).lower()

    def test_non_object_artifact_raises(self, tmp_path):
        path = tmp_path / "pairing_eligibility_v2.json"
        path.write_text("[1, 2, 3]", encoding="utf-8")
        with pytest.raises(schema.TypedIngressError):
            _phase4().load_pairing_eligibility(path)


# ===========================================================================
# R-075: role-keyed model identity pins + snapshot verification gate
# ===========================================================================


class TestR075ManifestArtifact:
    # Direct pins over the COMMITTED manifest (pass at RED time).
    # Source: reproducibility/colm_aims_2026/frozen/model_snapshot_manifests.json

    @pytest.fixture(scope="class")
    def art(self):
        return _load_json(MODEL_MANIFEST_PATH)

    def test_roles_exactly_primary_and_disjoint(self, art):
        assert set(art["roles"]) == {"primary_scorer", "disjoint_selector"}

    def test_model_names(self, art):
        assert (
            art["roles"]["primary_scorer"]["model_name"]
            == PRIMARY_SCORER_NAME
        )
        assert (
            art["roles"]["disjoint_selector"]["model_name"]
            == DISJOINT_SELECTOR_NAME
        )

    def test_per_file_manifests_nonempty_and_well_formed(self, art):
        for role, entry in art["roles"].items():
            files = entry["files"]
            assert files, f"role {role} has an empty file manifest"
            assert entry["file_count"] == len(files)
            assert schema.is_sha256_hex(entry["hf_revision"]) or (
                isinstance(entry["hf_revision"], str)
                and len(entry["hf_revision"]) == 40
            )
            for rel, meta in files.items():
                assert schema.is_sha256_hex(meta["sha256"]), (role, rel)
                assert isinstance(meta["size"], int) and not isinstance(
                    meta["size"], bool
                )
                assert meta["size"] > 0

    def test_offline_flags_exact(self, art):
        assert art["offline_flags_required"] == OFFLINE_FLAGS

    def test_tfidf_config_exact(self, art):
        assert art["tfidf_config"] == TFIDF_CONFIG


class TestR075SnapshotGate:
    # GREEN: phase4.load_model_snapshot_manifest / phase4.verify_snapshot_dir.

    def _make_snapshot(self, tmp_path):
        snap = tmp_path / "snapshot"
        (snap / "1_Pooling").mkdir(parents=True)
        blobs = {
            "config.json": b'{"hidden_size": 384}\n',
            "1_Pooling/config.json": b'{"pooling_mode_mean_tokens": true}\n',
            "vocab.txt": b"alpha\nbeta\ngamma\n",
        }
        for rel, blob in blobs.items():
            (snap / rel).write_bytes(blob)
        entry = {
            "model_name": PRIMARY_SCORER_NAME,
            "hf_revision": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
            "file_count": len(blobs),
            "files": {
                rel: {"sha256": sha256_bytes(blob), "size": len(blob)}
                for rel, blob in blobs.items()
            },
        }
        return snap, entry

    def test_load_manifest_happy_path(self):
        loaded = _phase4().load_model_snapshot_manifest(MODEL_MANIFEST_PATH)
        assert set(loaded["roles"]) == {
            "primary_scorer",
            "disjoint_selector",
        }

    def test_load_manifest_rejects_extra_role(self, tmp_path):
        art = _load_json(MODEL_MANIFEST_PATH)
        art["roles"]["shadow_scorer"] = _copy(
            art["roles"]["primary_scorer"]
        )
        path = tmp_path / "model_snapshot_manifests.json"
        path.write_text(json.dumps(art), encoding="utf-8")
        with pytest.raises(schema.ColmAimsError):
            _phase4().load_model_snapshot_manifest(path)

    def test_load_manifest_rejects_missing_offline_flags(self, tmp_path):
        art = _load_json(MODEL_MANIFEST_PATH)
        art["offline_flags_required"] = ["HF_HUB_OFFLINE=1"]
        path = tmp_path / "model_snapshot_manifests.json"
        path.write_text(json.dumps(art), encoding="utf-8")
        with pytest.raises(schema.ColmAimsError):
            _phase4().load_model_snapshot_manifest(path)

    def test_verify_snapshot_dir_match_is_ok(self, tmp_path):
        snap, entry = self._make_snapshot(tmp_path)
        assert _phase4().verify_snapshot_dir(entry, snap) is None

    def test_mutated_file_bytes_fail_naming_the_file(self, tmp_path):
        snap, entry = self._make_snapshot(tmp_path)
        (snap / "vocab.txt").write_bytes(b"alpha\nbeta\ngamma\ndelta\n")
        with pytest.raises(schema.ColmAimsError) as excinfo:
            _phase4().verify_snapshot_dir(entry, snap)
        assert "vocab.txt" in str(excinfo.value)

    def test_extra_file_fails(self, tmp_path):
        # Artifact-side mutation (seed catalog): an UNDECLARED extra file in
        # the snapshot is a mismatch even though every declared file checks.
        snap, entry = self._make_snapshot(tmp_path)
        (snap / "smuggled.bin").write_bytes(b"\x00\x01")
        with pytest.raises(schema.ColmAimsError) as excinfo:
            _phase4().verify_snapshot_dir(entry, snap)
        assert "smuggled.bin" in str(excinfo.value)

    def test_missing_file_fails(self, tmp_path):
        snap, entry = self._make_snapshot(tmp_path)
        (snap / "1_Pooling/config.json").unlink()
        with pytest.raises(schema.ColmAimsError) as excinfo:
            _phase4().verify_snapshot_dir(entry, snap)
        assert "1_Pooling/config.json" in str(excinfo.value)

    def test_size_is_independently_checked(self, tmp_path):
        # A manifest entry with the CORRECT sha but a wrong declared size
        # must still fail — size is a real check, not sha-shadowed.
        snap, entry = self._make_snapshot(tmp_path)
        entry["files"]["config.json"]["size"] += 1
        with pytest.raises(schema.ColmAimsError):
            _phase4().verify_snapshot_dir(entry, snap)

    def test_file_count_consistency_checked(self, tmp_path):
        snap, entry = self._make_snapshot(tmp_path)
        entry["file_count"] = 2
        with pytest.raises(schema.ColmAimsError):
            _phase4().verify_snapshot_dir(entry, snap)

    @pytest.mark.parametrize("reparse_at", ["root", "child", "file"])
    def test_reparse_entry_is_rejected_before_walk_or_open(
        self, tmp_path, monkeypatch, reparse_at
    ):
        phase4 = _phase4()
        snap, entry = self._make_snapshot(tmp_path)
        child = snap / "1_Pooling"
        file_entry = snap / "config.json"
        target = {"root": snap, "child": child, "file": file_entry}[
            reparse_at
        ]
        real_stat = os.stat
        nofollow_targets = []

        class StatWithFileAttributes:
            def __init__(self, original):
                self._original = original
                self.st_file_attributes = getattr(
                    original, "st_file_attributes", 0
                ) | getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)

            def __getattr__(self, name):
                return getattr(self._original, name)

        def mark_child(path, *args, **kwargs):
            observed = real_stat(path, *args, **kwargs)
            if Path(path) == target and kwargs.get("follow_symlinks") is False:
                nofollow_targets.append(Path(path))
                return StatWithFileAttributes(observed)
            return observed

        monkeypatch.setattr(phase4.os, "stat", mark_child)
        if reparse_at == "root":

            def reject_walk(*args, **kwargs):
                pytest.fail("snapshot verifier walked a reparse root")

            monkeypatch.setattr(phase4.os, "walk", reject_walk)
        else:

            def guarded_walk(path, *, topdown, onerror, followlinks):
                assert Path(path) == snap
                assert topdown is True
                assert followlinks is False
                yield str(snap), [child.name], ["config.json", "vocab.txt"]
                pytest.fail("snapshot verifier resumed after a reparse entry")

            monkeypatch.setattr(phase4.os, "walk", guarded_walk)
        with pytest.raises(schema.ColmAimsError, match="reparse"):
            phase4.verify_snapshot_dir(entry, snap)
        assert nofollow_targets == [target]

    @pytest.mark.skipif(os.name != "nt", reason="requires a real NTFS junction")
    @pytest.mark.parametrize("junction_at", ["root", "child"])
    def test_real_windows_junction_is_never_scanned(
        self, tmp_path, monkeypatch, junction_at
    ):
        phase4 = _phase4()
        snap, entry = self._make_snapshot(tmp_path)
        external = tmp_path / "external"
        if junction_at == "root":
            snap.rename(external)
            junction = snap
        else:
            junction = snap / "1_Pooling"
            external.mkdir()
            (external / "config.json").write_bytes(
                (junction / "config.json").read_bytes()
            )
            (junction / "config.json").unlink()
            junction.rmdir()

        completed = subprocess.run(
            [
                os.environ.get("COMSPEC", "cmd.exe"),
                "/d",
                "/c",
                "mklink",
                "/J",
                str(junction),
                str(external),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            pytest.skip(f"junction creation unavailable: {completed.stderr}")

        real_scandir = os.scandir
        scanned = []

        def record_scandir(path):
            scanned.append(Path(path))
            return real_scandir(path)

        monkeypatch.setattr(phase4.os, "scandir", record_scandir)
        try:
            with pytest.raises(schema.ColmAimsError, match="reparse"):
                phase4.verify_snapshot_dir(entry, snap)
            assert junction not in scanned
        finally:
            os.rmdir(junction)


# ===========================================================================
# R-076: staged-input hash gates (fail-closed, before any loader)
# ===========================================================================


class TestR076StagedInputGate:
    def _staged(self, tmp_path):
        a = tmp_path / "calibration_train.json"
        b = tmp_path / "test_dataset.json"
        a.write_bytes(b'{"rows": [1, 2]}\n')
        b.write_bytes(b'{"rows": [3]}\n')
        return [
            {
                "path": a,
                "expected_sha256": sha256_file(a),
                "label": "calibration_train",
            },
            {
                "path": b,
                "expected_sha256": sha256_file(b),
                "label": "test_dataset",
            },
        ]

    def test_all_match_returns_observed_hashes_in_order(self, tmp_path):
        staged = self._staged(tmp_path)
        out = _phase4().staged_input_gate(staged)
        assert [e["label"] for e in out] == [
            "calibration_train",
            "test_dataset",
        ]
        for given, got in zip(staged, out):
            assert got["expected_sha256"] == given["expected_sha256"]
            assert got["observed_sha256"] == given["expected_sha256"]

    def test_mismatch_raises_naming_file_expected_and_observed(
        self, tmp_path
    ):
        staged = self._staged(tmp_path)
        real_bytes = staged[1]["path"].read_bytes()
        staged[1]["path"].write_bytes(b'{"rows": [3, 4]}\n')
        observed = sha256_bytes(staged[1]["path"].read_bytes())
        with pytest.raises(schema.TypedIngressError) as excinfo:
            _phase4().staged_input_gate(staged)
        message = str(excinfo.value)
        assert "test_dataset.json" in message
        assert staged[1]["expected_sha256"] in message
        assert observed in message
        assert observed != sha256_bytes(real_bytes)

    def test_missing_file_raises_naming_it(self, tmp_path):
        staged = self._staged(tmp_path)
        staged[0]["path"].unlink()
        with pytest.raises(schema.TypedIngressError) as excinfo:
            _phase4().staged_input_gate(staged)
        assert "calibration_train.json" in str(excinfo.value)

    def test_first_mismatch_in_list_order_is_named(self, tmp_path):
        staged = self._staged(tmp_path)
        staged[0]["path"].write_bytes(b"tampered-a")
        staged[1]["path"].write_bytes(b"tampered-b")
        with pytest.raises(schema.TypedIngressError) as excinfo:
            _phase4().staged_input_gate(staged)
        assert "calibration_train.json" in str(excinfo.value)

    def test_empty_staged_list_raises(self):
        # DECISION: a gate over ZERO inputs is a defect (vacuously-empty
        # authoritative set), not a trivially-passing gate.
        with pytest.raises(schema.TypedIngressError):
            _phase4().staged_input_gate([])

    def test_malformed_expected_digest_raises(self, tmp_path):
        staged = self._staged(tmp_path)
        staged[0]["expected_sha256"] = "not-a-sha"
        with pytest.raises(schema.TypedIngressError):
            _phase4().staged_input_gate(staged)


# ===========================================================================
# R-077: materialized parity comparator
# ===========================================================================


def _anchor():
    # Source: reproducibility/colm_aims_2026/frozen/parity_anchor_export_a.json
    return _load_json(PARITY_ANCHOR_PATH)


def _regen_from_anchor(anchor):
    """A producer-shaped export whose values come FROM the anchor (exact
    match), with the two Random-K cells filled from the informational
    archived values."""
    results = {}
    for cell, policies in anchor["expected"].items():
        results[cell] = {p: _copy(v) for p, v in policies.items()}
    informational = anchor["random_k"]["informational_archived_values"]
    for cell, policies in informational.items():
        results[cell] = {p: _copy(v) for p, v in policies.items()}
    return {
        "metadata": {
            "n_eval": anchor["identity_fields"]["n_eval"],
            "n_fit": anchor["identity_fields"]["n_fit"],
        },
        "results": results,
    }


class TestR077AnchorArtifact:
    # Direct pins over the COMMITTED anchor (pass at RED time).

    @pytest.fixture(scope="class")
    def art(self):
        return _anchor()

    def test_field_counts(self, art):
        assert art["field_count"] == {"point": 160, "ci_arrays": 32}

    def test_allowlist_completeness_is_arithmetically_exact(self, art):
        # 8 nonrandom cells x 2 policies x 10 point fields == 160;
        # 8 x 2 x 2 CI arrays == 32 — no more, no fewer.
        assert len(art["point_fields"]) == 10
        assert len(art["ci_fields"]) == 2
        assert len(art["nonrandom_cells"]) == 8
        assert len(art["policies"]) == 2
        assert 8 * 2 * 10 == art["field_count"]["point"]
        assert 8 * 2 * 2 == art["field_count"]["ci_arrays"]
        expected_keys = sorted(art["point_fields"] + art["ci_fields"])
        for cell in art["nonrandom_cells"]:
            for policy in art["policies"]:
                assert (
                    sorted(art["expected"][cell][policy]) == expected_keys
                ), (cell, policy)

    def test_nonrandom_cells_exact_set(self, art):
        assert set(art["nonrandom_cells"]) == {
            f"{ref}+{cal}"
            for ref in ("idealized", "khard", "kdisjoint", "klex")
            for cal in ("shared", "performat")
        }

    def test_source_is_export_a(self, art):
        assert art["source"]["sha256"] == EXPORT_A_SHA
        assert art["source"]["basename"] == "stopdff_fair_qa.json"

    def test_export_b_is_corroborative_never_anchor(self, art):
        assert art["corroborative"]["sha256"] == EXPORT_B_SHA
        assert "corroborative" in art["corroborative"]["role"]

    def test_random_k_flags(self, art):
        rk = art["random_k"]
        assert rk["exempt_from_historical_parity"] is True
        assert rk["archived_rng_pinned"] is False
        assert rk["fresh_rng_pinned"] is True
        assert set(rk["cells"]) == {"krandom+shared", "krandom+performat"}

    def test_identity_fields(self, art):
        assert art["identity_fields"] == {
            "n_eval": 2258,
            "n_fit": 2142,
            "per_cell_n": 2249,
        }

    def test_per_cell_n_uniformly_2249(self, art):
        for cell in art["nonrandom_cells"]:
            for policy in art["policies"]:
                assert art["expected"][cell][policy]["n"] == 2249

    def test_label_mapping_performat_to_format_specific(self, art):
        assert art["label_mapping"] == {"performat": "format_specific"}

    def test_comparison_rules_blocking(self, art):
        assert art["comparison_rules"]["any_mismatch_blocking"] is True
        assert art["comparison_rules"]["ci_arrays_blocking"] is True


class TestR077CompareParity:
    # GREEN: phase4.compare_parity.

    def test_exact_match_passes_with_full_checked_count(self):
        anchor = _anchor()
        regen = _regen_from_anchor(anchor)
        result = _phase4().compare_parity(anchor, regen)
        assert result["verdict"] == "PASS"
        assert result["failures"] == []
        # DECISION: checked == 160 point + 32 CI arrays + 2 identity fields
        # (n_eval, n_fit); the per-cell n sits INSIDE the 160.
        assert result["checked"] == 160 + 32 + 2

    def test_single_point_field_mutation_fails_naming_it(self):
        anchor = _anchor()
        regen = _regen_from_anchor(anchor)
        regen["results"]["khard+shared"]["myopic"]["abs_mean"] += 0.0001
        result = _phase4().compare_parity(anchor, regen)
        assert result["verdict"] == "FAIL"
        rows = [
            f
            for f in result["failures"]
            if f["cell"] == "khard+shared"
            and f["policy"] == "myopic"
            and f["field"] == "abs_mean"
        ]
        assert len(rows) == 1
        assert set(rows[0]) == {
            "cell",
            "policy",
            "field",
            "expected",
            "observed",
        }

    def test_single_ci_element_mutation_is_blocking(self):
        anchor = _anchor()
        regen = _regen_from_anchor(anchor)
        regen["results"]["klex+performat"]["dp"]["signed_mean_ci"][1] += (
            0.0001
        )
        result = _phase4().compare_parity(anchor, regen)
        assert result["verdict"] == "FAIL"
        assert any(
            f["cell"] == "klex+performat"
            and f["policy"] == "dp"
            and f["field"] == "signed_mean_ci"
            for f in result["failures"]
        )

    def test_identity_field_mutation_fails(self):
        anchor = _anchor()
        regen = _regen_from_anchor(anchor)
        regen["metadata"]["n_eval"] = 2257
        result = _phase4().compare_parity(anchor, regen)
        assert result["verdict"] == "FAIL"
        rows = [f for f in result["failures"] if f["field"] == "n_eval"]
        assert rows and rows[0]["cell"] is None
        assert rows[0]["policy"] is None

    def test_random_k_divergence_is_informational_never_blocking(self):
        anchor = _anchor()
        regen = _regen_from_anchor(anchor)
        for policy in ("dp", "myopic"):
            regen["results"]["krandom+shared"][policy]["signed_mean"] = 99.0
            regen["results"]["krandom+performat"][policy]["abs_mean"] = 42.0
        result = _phase4().compare_parity(anchor, regen)
        assert result["verdict"] == "PASS"
        assert result["failures"] == []
        info = result["random_k_informational"]
        assert "krandom+shared" in json.dumps(info)

    def test_bool_observed_is_never_numeric_equal(self):
        # Bool-laundering (seed catalog): expected 1.0 vs observed True and
        # expected 0.0 vs observed False are Python-== equal — the
        # comparator must still FAIL both.
        anchor = _anchor()
        assert anchor["expected"]["idealized+shared"]["dp"][
            "abs_median"
        ] == 1.0
        assert anchor["expected"]["idealized+shared"]["dp"][
            "signed_median"
        ] == 0.0
        regen = _regen_from_anchor(anchor)
        regen["results"]["idealized+shared"]["dp"]["abs_median"] = True
        regen["results"]["idealized+shared"]["dp"]["signed_median"] = False
        result = _phase4().compare_parity(anchor, regen)
        assert result["verdict"] == "FAIL"
        failed_fields = {
            f["field"]
            for f in result["failures"]
            if f["cell"] == "idealized+shared" and f["policy"] == "dp"
        }
        assert {"abs_median", "signed_median"} <= failed_fields

    def test_missing_cell_becomes_failures_not_exception(self):
        # Guarded builder (seed catalog): absence is a FAIL with receipts,
        # never a KeyError escape.
        anchor = _anchor()
        regen = _regen_from_anchor(anchor)
        del regen["results"]["khard+shared"]
        result = _phase4().compare_parity(anchor, regen)
        assert result["verdict"] == "FAIL"
        assert any(
            f["cell"] == "khard+shared" for f in result["failures"]
        )

    def test_missing_point_field_becomes_failure_row(self):
        anchor = _anchor()
        regen = _regen_from_anchor(anchor)
        del regen["results"]["idealized+performat"]["myopic"]["same_step"]
        result = _phase4().compare_parity(anchor, regen)
        assert result["verdict"] == "FAIL"
        assert any(
            f["cell"] == "idealized+performat"
            and f["policy"] == "myopic"
            and f["field"] == "same_step"
            for f in result["failures"]
        )


def _perturb(value):
    """Return a value of the SAME JSON type as ``value`` that is guaranteed
    to be ``!= value`` under ``phase4._values_equal`` (no cross-type
    laundering). Used to force a VALUE mismatch on the regenerated side while
    keeping the JSON type identical, so the divergence exercises the
    value-routing path rather than the structural (missing/type) path."""
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return value + 1
    if isinstance(value, float):
        return value + 1.0
    if isinstance(value, list):
        if not value:
            return [0]
        return [_perturb(value[0])] + [_copy(v) for v in value[1:]]
    if isinstance(value, str):
        return value + "_x"
    return value


class TestR077KnownDivergenceCarveOut:
    """Phase-4 Option-A scoped known-divergence carve-out (amended R-077,
    2026-08-24). Locks the six-field carve-out: exactly the six fields
    ``{signed_mean, abs_mean, mc_earlier, qa_earlier, same_step,
    signed_mean_ci}`` of the SINGLE cell/policy ``idealized+performat``·``dp``
    route a VALUE mismatch to the non-blocking
    ``known_divergence_informational`` bucket, while (a) STRUCTURE stays
    blocking, (b) ``checked`` stays 194, and (c) any VALUE mismatch outside
    that {cell x six-field} set stays a blocking FAILURE. Mirrors the
    pre-existing R-077 Random-K informational-only treatment, narrowed from
    whole-cell to a six-field allowlist.

    Source: reproducibility/colm_aims_2026/phase4.py — KNOWN_DIVERGENCE_CELL/
    _POLICY/_FIELDS/KNOWN_DIVERGENCES (L203-216), compare_parity value routing
    (L2635-2662), known_divergence_informational bucket (L2742-2755);
    phase4_reconciliation_amendment_proposal_A_2026-08-24.md.
    Reuses the module-level ``_anchor`` / ``_regen_from_anchor`` / ``_phase4``
    parity fixtures (frozen anchor loaded read-only; the regenerated side is
    synthetic — no model load, no trajectory records).
    """

    CELL = "idealized+performat"
    POLICY = "dp"
    SIX_FIELDS = (
        "signed_mean",
        "abs_mean",
        "mc_earlier",
        "qa_earlier",
        "same_step",
        "signed_mean_ci",
    )

    def test_carveout_constants_match_spec(self):
        # Pin the carve-out identity so a silent widening/renaming of the
        # allowlist is caught here rather than only via behavior.
        phase4 = _phase4()
        assert phase4.KNOWN_DIVERGENCE_CELL == self.CELL
        assert phase4.KNOWN_DIVERGENCE_POLICY == self.POLICY
        assert tuple(phase4.KNOWN_DIVERGENCE_FIELDS) == self.SIX_FIELDS
        assert phase4.KNOWN_DIVERGENCES == frozenset(
            (self.CELL, self.POLICY, f) for f in self.SIX_FIELDS
        )
        assert phase4.EXPECTED_PARITY_CHECKED == 194

    def test_s1_six_field_value_divergence_is_informational_pass(self):
        # S1 (carve-out active): diverge ONLY the six fields for the
        # idealized+performat·dp cell => PASS, exactly 6 informational
        # entries, checked == 194, blocking failures empty.
        anchor = _anchor()
        regen = _regen_from_anchor(anchor)
        cell = regen["results"][self.CELL][self.POLICY]
        for field in self.SIX_FIELDS:
            perturbed = _perturb(cell[field])
            # guard against a vacuous test: the mismatch must be real
            # (``_perturb`` preserves JSON type, so a plain ``!=`` suffices).
            assert perturbed != cell[field]
            cell[field] = perturbed
        result = _phase4().compare_parity(anchor, regen)

        assert result["verdict"] == "PASS"
        assert result["failures"] == []
        assert result["checked"] == 194

        info = result["known_divergence_informational"]
        assert info["cells"] == [self.CELL]
        assert info["policies"] == [self.POLICY]
        assert info["exempt_from_historical_parity"] is True
        assert info["compared"] == 6
        assert len(info["divergences"]) == 6
        diverged = {
            (d["cell"], d["policy"], d["field"]) for d in info["divergences"]
        }
        assert diverged == {
            (self.CELL, self.POLICY, f) for f in self.SIX_FIELDS
        }
        # none of the six leaked into the blocking failures list
        assert not any(
            f["cell"] == self.CELL and f["policy"] == self.POLICY
            for f in result["failures"]
        )

    def test_s2a_seventh_field_in_carveout_cell_stays_blocking(self):
        # S2 (must stay blocking): a 7th, NON-carve-out field VALUE mismatch
        # in the SAME idealized+performat·dp cell => FAIL, not routed to
        # informational.
        anchor = _anchor()
        regen = _regen_from_anchor(anchor)
        seventh = next(
            f for f in anchor["point_fields"] if f not in self.SIX_FIELDS
        )
        cell = regen["results"][self.CELL][self.POLICY]
        cell[seventh] = _perturb(cell[seventh])
        result = _phase4().compare_parity(anchor, regen)

        assert result["verdict"] == "FAIL"
        assert result["checked"] == 194
        assert any(
            f["cell"] == self.CELL
            and f["policy"] == self.POLICY
            and f["field"] == seventh
            for f in result["failures"]
        )
        assert result["known_divergence_informational"]["divergences"] == []

    def test_s2b_carveout_fieldname_other_cell_stays_blocking(self):
        # S2 (must stay blocking): one of the six field-names mismatched in a
        # DIFFERENT cell => FAIL (carve-out is cell/policy-scoped).
        anchor = _anchor()
        regen = _regen_from_anchor(anchor)
        other_cell = next(
            c for c in anchor["nonrandom_cells"] if c != self.CELL
        )
        target = regen["results"][other_cell][self.POLICY]
        target["signed_mean"] = _perturb(target["signed_mean"])
        result = _phase4().compare_parity(anchor, regen)

        assert result["verdict"] == "FAIL"
        assert result["checked"] == 194
        assert any(
            f["cell"] == other_cell
            and f["policy"] == self.POLICY
            and f["field"] == "signed_mean"
            for f in result["failures"]
        )
        # the other cell's divergence is NOT laundered into informational
        assert result["known_divergence_informational"]["divergences"] == []

    def test_s2c_carveout_fieldname_other_policy_stays_blocking(self):
        # S2 (must stay blocking): a carve-out field-name mismatched under the
        # non-carve-out policy (myopic) of the SAME cell => FAIL.
        anchor = _anchor()
        regen = _regen_from_anchor(anchor)
        other_policy = next(
            p for p in anchor["policies"] if p != self.POLICY
        )
        target = regen["results"][self.CELL][other_policy]
        target["signed_mean"] = _perturb(target["signed_mean"])
        result = _phase4().compare_parity(anchor, regen)

        assert result["verdict"] == "FAIL"
        assert result["checked"] == 194
        assert any(
            f["cell"] == self.CELL
            and f["policy"] == other_policy
            and f["field"] == "signed_mean"
            for f in result["failures"]
        )
        assert result["known_divergence_informational"]["divergences"] == []

    def test_s3_missing_carveout_field_stays_structural_blocking(self):
        # S3 (structural stays blocking): a MISSING (removed) carve-out field
        # is structural, not a value divergence => blocking FAIL, and it is
        # NOT laundered into the informational bucket (routing is gated on
        # ``observed_value is not _MISSING``).
        anchor = _anchor()
        regen = _regen_from_anchor(anchor)
        del regen["results"][self.CELL][self.POLICY]["signed_mean"]
        result = _phase4().compare_parity(anchor, regen)

        assert result["verdict"] == "FAIL"
        assert result["checked"] == 194
        rows = [
            f
            for f in result["failures"]
            if f["cell"] == self.CELL
            and f["policy"] == self.POLICY
            and f["field"] == "signed_mean"
        ]
        assert len(rows) == 1
        assert rows[0]["observed"] is None
        assert all(
            d["field"] != "signed_mean"
            for d in result["known_divergence_informational"]["divergences"]
        )


# ===========================================================================
# R-078: QA-012 compatibility fixtures — bytes-bound, loader-rejected
# ===========================================================================


class TestR078Qa012Fixtures:
    # Source: tests/fixtures/qa012_item10/bindings.json (+ four committed
    # exact full-file fixtures and first-2-line excerpts beside it), binding
    # the item-10 bundle per_prefix_scores*.jsonl hit files by SHA-256.

    @pytest.fixture(scope="class")
    def bindings(self):
        return _load_json(QA012_BINDINGS_PATH)

    def test_bindings_cover_exactly_four_files(self, bindings):
        assert len(bindings["files"]) == 4
        assert {
            meta["full_file_sha256"] for meta in bindings["files"].values()
        } == set(QA012_FULL_FILE_SHAS)

    def test_excerpt_fixture_bytes_hash_to_bindings(self, bindings):
        for name, meta in bindings["files"].items():
            full_blob = (
                QA012_FIXTURE_DIR / meta["full_fixture"]
            ).read_bytes()
            fixture = QA012_FIXTURE_DIR / meta["excerpt_fixture"]
            blob = fixture.read_bytes()
            assert len(full_blob) == meta["full_file_size"], name
            assert sha256_bytes(full_blob) == meta["full_file_sha256"], name
            assert sha256_bytes(blob) == meta["excerpt_sha256"], name
            assert b"\n".join(full_blob.splitlines()[:2]) + b"\n" == blob
            lines = [
                line
                for line in blob.decode("utf-8").splitlines()
                if line.strip()
            ]
            assert len(lines) == meta["excerpt_lines"] == 2, name

    def test_first_record_shape_is_the_incompatible_quad(self, bindings):
        for name, meta in bindings["files"].items():
            fixture = QA012_FIXTURE_DIR / meta["excerpt_fixture"]
            first = json.loads(
                fixture.read_text("utf-8").splitlines()[0]
            )
            assert set(first) == {
                "item_id",
                "format",
                "prefix_fractions",
                "p_calibrated",
            }, name
            assert first["format"] in {"MC", "QA"}

    def test_v2_ingestion_rejects_the_prototype_shape(self, bindings):
        # The cannot-substitute demonstration: these rows carry no item_key,
        # no event statuses, no stop steps, no horizon — the v2 record
        # loader must refuse each one, naming a missing required field.
        for name, meta in bindings["files"].items():
            fixture = QA012_FIXTURE_DIR / meta["full_fixture"]
            loaded = schema.load_records_bytes(
                fixture.read_bytes(), f"records/{name}"
            )
            assert loaded["kind"] == "records"
            for record in loaded["records"]:
                with pytest.raises(
                    schema.RecordValidationError
                ) as excinfo:
                    schema.validate_record(record)
                assert "item_key" in str(excinfo.value)


# ===========================================================================
# R-079: PRE_RUN_READY certificate assembly (pure core)
# ===========================================================================


# R-070/R-082 (F-4 + operational rejection): the required suite-receipt
# field set. A receipt missing environment_lock_sha256 — or, per R-082, the
# head bindings commit/tree_sha256/dirty — is a failing suite_receipts
# component.
R070_RECEIPT_KEYS = frozenset(
    {
        "exit_code",
        "command",
        "environment_lock_sha256",
        "workflow_sha256",
        "interpreter_realpath",
        "counts",
        "skip_identities",
        "commit",
        "tree_sha256",
        "dirty",
    }
)

_PHASE4_STAGED_FILENAMES = {
    "calibration_train": "calibration_train.json",
    "eval_split": "test_dataset.json",
    "fit_split": "val_dataset.json",
    "mc_dataset": "mc_dataset.json",
    "answer_profiles": "answer_profiles.json",
    "build_metadata": "build_metadata.json",
}
_PHASE4_OPERATOR_DIGEST_LABELS = (
    "fit_split",
    "mc_dataset",
    "answer_profiles",
    "build_metadata",
)


def _staging_command_args(
    staged_dir, digest_by_label, eligibility_path=ELIGIBILITY_PATH
):
    staged_dir = Path(staged_dir)
    args = [
        "--data-dir",
        str(staged_dir),
        "--calibration",
        str(staged_dir / _PHASE4_STAGED_FILENAMES["calibration_train"]),
        "--eligibility",
        str(Path(eligibility_path).resolve()),
        "--fit-split",
        "val",
        "--eval-split",
        "test",
    ]
    for label in _PHASE4_OPERATOR_DIGEST_LABELS:
        path = staged_dir / _PHASE4_STAGED_FILENAMES[label]
        args.extend(
            ["--staged-input", f"{label}={path}:{digest_by_label[label]}"]
        )
    return args


def _experiment_command_args():
    return [
        "--reward-schedule",
        "power_mark",
        "--qa-arms",
        "idealized,krandom,khard,kdisjoint,klex",
        "--calibrations",
        "shared,performat",
        "--num-bootstrap",
        "1000",
        "--n-test",
        "0",
        "--n-val",
        "0",
        "--seed",
        "1",
    ]


def _snapshot_output_command_args(manifest_path, snapshot_dirs):
    return [
        "--snapshot-manifest",
        str(manifest_path),
        "--primary-model-path",
        str(snapshot_dirs["primary_scorer"]),
        "--disjoint-model-path",
        str(snapshot_dirs["disjoint_selector"]),
        "--records-out",
        "phase4_run_output",
        "--out",
        "phase4_run_output/stopdff_fair_qa_regenerated.json",
    ]


def _r070_receipt(
    command,
    *,
    exit_code=0,
    commit="f" * 40,
    tree_sha256="1" * 64,
    dirty=False,
    interpreter_realpath="/repo/.venv/bin/python3.11",
    environment_lock_sha256="6" * 64,
):
    # AMENDED (R-082 forward-compat, same trick as the F-4 round): receipts
    # carry the head bindings; defaults match _good_components()["repo"] so
    # the original R-079 suite stays green once the checker tightens.
    if isinstance(command, list):
        rendered_command = list(command)
    else:
        suite_name = "full" if "full" in str(command).lower() else "focused"
        phase4 = _phase4()
        selection = (
            phase4.FULL_SUITE_SELECTION
            if suite_name == "full"
            else phase4.FOCUSED_SUITE_SELECTION
        )
        rendered_command = [
            interpreter_realpath,
            "-m",
            "pytest",
            *selection,
            "-q",
            "-p",
            "no:cacheprovider",
            f"--junitxml={Path(tempfile.gettempdir()).resolve() / (suite_name + '_suite.xml')}",
        ]
    return {
        "schema_version": schema.SCHEMA_VERSION,
        "exit_code": exit_code,
        "command": rendered_command,
        "environment_lock_sha256": environment_lock_sha256,
        "workflow_sha256": "7" * 64,
        "interpreter_realpath": interpreter_realpath,
        "counts": {
            "tests": 125,
            "failures": 0,
            "errors": 0,
            "skipped": 0,
        },
        "skip_identities": [],
        "junit_sha256": "c" * 64,
        "transcript_sha256": "d" * 64,
        "commit": commit,
        "tree_sha256": tree_sha256,
        "dirty": dirty,
    }


def _good_components():
    eligibility = _load_json(ELIGIBILITY_PATH)
    phase4 = _phase4()
    fixture_interpreter = str(Path(sys.executable).resolve())
    fixture_base = Path(tempfile.gettempdir()).resolve()
    fixture_repo = REPO_ROOT.resolve()
    fixture_staging = (
        fixture_base / "qanta_phase4_certificate_staging_fixture"
    )
    fixture_launch = fixture_base / "qanta_phase4_certificate_launch_fixture"
    staged_digests = dict(phase4.R082_STAGED_INPUT_SHA256)
    return {
        "repo": {
            "commit": "f" * 40,
            "tree_sha256": "1" * 64,
            "dirty": False,
            "untracked_disclosure": [],
            "root_realpath": str(fixture_repo),
        },
        "content_hashes": {
            key: {
                "artifact_path": str((fixture_repo / relpath).resolve()),
                "sha256": sha256_file(fixture_repo / relpath),
            }
            for key, relpath in phase4.CONTENT_HASH_RELPATHS.items()
        },
        "eligibility": {
            "digest": eligibility["pairing_population_keyset_sha256"],
            "horizon_map_sha256": eligibility["horizon_map_sha256"],
            "artifact_path": str(ELIGIBILITY_PATH.resolve()),
            "artifact_sha256": sha256_file(ELIGIBILITY_PATH),
            "test_dataset_sha256": eligibility["derived_from"][
                "test_dataset_sha256"
            ],
        },
        "snapshots": {
            "artifact_path": str(MODEL_MANIFEST_PATH.resolve()),
            "artifact_sha256": sha256_file(MODEL_MANIFEST_PATH),
            "primary_scorer": {
                "verified": True,
                "model_name": PRIMARY_SCORER_NAME,
                "hf_revision": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
            },
            "disjoint_selector": {
                "verified": True,
                "model_name": DISJOINT_SELECTOR_NAME,
                "hf_revision": "e8c3b32edf5434bc2275fc9bab85f82640a19130",
            },
        },
        "offline_flags": list(OFFLINE_FLAGS),
        "staged_inputs": [
            {
                "path": str(fixture_staging / filename),
                "label": label,
                "expected_sha256": staged_digests[label],
                "observed_sha256": staged_digests[label],
            }
            for label, filename in _PHASE4_STAGED_FILENAMES.items()
        ],
        # AMENDED (F-4/R-070 forward-compat): receipts carry the full R-070
        # field set; the current checker ignores the extras, the F-4 round
        # makes them REQUIRED (see TestR070ReceiptFieldsInAssemble).
        "suite_receipts": {
            "focused": _r070_receipt(
                "focused",
                interpreter_realpath=fixture_interpreter,
            ),
            "full": _r070_receipt(
                "full",
                interpreter_realpath=fixture_interpreter,
            ),
        },
        "parity": {
            "comparator_identity": (
                "reproducibility.colm_aims_2026.phase4.compare_parity"
            ),
            "artifact_path": str(PARITY_ANCHOR_PATH.resolve()),
            "anchor_sha256": phase4.PARITY_ANCHOR_SHA256,
            "source_export_a_sha256": phase4.PARITY_SOURCE_EXPORT_A_SHA256,
        },
        "qa012": {
            "artifact_path": str(QA012_REV3_MANIFEST_PATH.resolve()),
            "manifest_sha256": phase4.QA012_MANIFEST_SHA256,
            "manifest_type": "qa012_format_qa_inventory",
            "revision": 3,
            "conventions": {
                "content_hash": (
                    "Dropbox content hash: sha256 over concatenated"
                    " per-4MiB-block sha256 digests, hex"
                ),
                "sha256": "sha256 over the raw file bytes, hex",
                "jsonl_line_numbers": "1-based",
            },
        },
        "environment": {
            "interpreter_realpath": fixture_interpreter,
            "os": "Darwin",
            "arch": "arm64",
            "cpu": "Apple M3 Max",
            "blas": "accelerate",
            "thread_settings": dict(phase4.PHASE4_THREAD_SETTINGS),
            "environment_lock_sha256": "6" * 64,
            "command": (
                [fixture_interpreter, "scripts/stopdff_fair_qa_retest.py"]
                + _staging_command_args(fixture_staging, staged_digests)
                + _experiment_command_args()
                + _snapshot_output_command_args(
                    MODEL_MANIFEST_PATH.resolve(),
                    {
                        "primary_scorer": fixture_staging
                        / "snap_primary_scorer",
                        "disjoint_selector": fixture_staging
                        / "snap_disjoint_selector",
                    },
                )
            ),
            "seeds": [1],
            "pythonhashseed": "0",
            "archived_rng_pinned": False,
            "fresh_rng_pinned": True,
            "quarantine_dir": str(fixture_launch / "quarantine"),
            "promote_to": str(fixture_launch / "promoted"),
            "exception_ledger_path": str(
                fixture_launch / "exception-ledger.json"
            ),
        },
    }


class TestR079AssembleCertificate:
    def test_all_good_is_ready_true(self):
        cert = _phase4().assemble_certificate(_good_components())
        assert cert["ready"] is True
        assert cert["failing_checks"] == []
        assert cert["schema_version"] == 2

    def test_certificate_embeds_the_component_bindings(self):
        components = _good_components()
        cert = _phase4().assemble_certificate(components)
        blob = json.dumps(cert)
        assert components["eligibility"]["digest"] in blob
        assert components["eligibility"]["horizon_map_sha256"] in blob
        assert _phase4().QA012_MANIFEST_SHA256 in blob
        assert EXPORT_A_SHA in blob
        assert CALIB_TRAIN_SHA in blob

    def test_dirty_tree_fails_closed(self):
        components = _good_components()
        components["repo"]["dirty"] = True
        cert = _phase4().assemble_certificate(components)
        assert cert["ready"] is False
        assert any("repo" in check for check in cert["failing_checks"])

    def test_stringly_typed_dirty_flag_is_not_clean(self):
        # Bool-guard: only the exact False means clean; "false" is a defect.
        components = _good_components()
        components["repo"]["dirty"] = "false"
        cert = _phase4().assemble_certificate(components)
        assert cert["ready"] is False

    def test_staged_input_hash_mismatch_fails(self):
        components = _good_components()
        components["staged_inputs"][1]["observed_sha256"] = "9" * 64
        cert = _phase4().assemble_certificate(components)
        assert cert["ready"] is False
        assert any(
            "staged_inputs" in check for check in cert["failing_checks"]
        )

    def test_missing_observation_is_not_a_pass(self):
        components = _good_components()
        components["staged_inputs"][0]["observed_sha256"] = None
        cert = _phase4().assemble_certificate(components)
        assert cert["ready"] is False

    def test_snapshot_mismatch_fails(self):
        components = _good_components()
        components["snapshots"]["primary_scorer"]["verified"] = False
        cert = _phase4().assemble_certificate(components)
        assert cert["ready"] is False
        assert any(
            "snapshot" in check for check in cert["failing_checks"]
        )

    def test_suite_failure_fails(self):
        components = _good_components()
        components["suite_receipts"]["full"]["exit_code"] = 1
        cert = _phase4().assemble_certificate(components)
        assert cert["ready"] is False
        assert any("suite" in check for check in cert["failing_checks"])

    def test_bool_exit_code_is_rejected_not_laundered(self):
        # False == 0 in Python; a bool exit code must NOT read as success.
        components = _good_components()
        components["suite_receipts"]["focused"]["exit_code"] = False
        cert = _phase4().assemble_certificate(components)
        assert cert["ready"] is False

    def test_missing_component_fails_and_is_named(self):
        components = _good_components()
        del components["environment"]
        cert = _phase4().assemble_certificate(components)
        assert cert["ready"] is False
        assert any(
            "environment" in check for check in cert["failing_checks"]
        )

    def test_missing_environment_field_fails(self):
        components = _good_components()
        del components["environment"]["pythonhashseed"]
        cert = _phase4().assemble_certificate(components)
        assert cert["ready"] is False

    def test_wrong_offline_flags_fail(self):
        components = _good_components()
        components["offline_flags"] = ["HF_HUB_OFFLINE=1"]
        cert = _phase4().assemble_certificate(components)
        assert cert["ready"] is False
        assert any(
            "offline" in check for check in cert["failing_checks"]
        )

    def test_two_defects_are_both_named_never_partial(self):
        components = _good_components()
        components["repo"]["dirty"] = True
        components["staged_inputs"][0]["observed_sha256"] = "8" * 64
        cert = _phase4().assemble_certificate(components)
        assert cert["ready"] is False
        assert any("repo" in check for check in cert["failing_checks"])
        assert any(
            "staged_inputs" in check for check in cert["failing_checks"]
        )

    def test_required_component_key_universe_is_pinned(self):
        # Contract guard: the good fixture covers exactly the required set.
        components = _good_components()
        assert set(components) == set(CERT_COMPONENT_KEYS)
        assert set(components["environment"]) == set(CERT_ENVIRONMENT_KEYS)


# ===========================================================================
# R-080: records export (producer boundary, model-free)
# ===========================================================================


def _scored_items():
    # AMENDED (F-7, adjudicated R-080 tightening 2026-08-22): the DP sentinel
    # is EXACTLY stop == horizon; stop > horizon is frame corruption and is
    # refused (see TestF7OvershootRefusal). QID 1's ref overshoot (5 > 2)
    # became ref_stop == 2 and QID 4's mc overshoot (12 > 7) became
    # mc_stop == 7 — both previously asserted the absorbed-overshoot
    # behavior this round retires.
    return [
        # mc at the sentinel (stop == horizon) => NEVER; ref finite.
        {"item_key": "2", "horizon": 4, "mc_stop": 4, "ref_stop": 1},
        # mc genuine final-prefix crossing at horizon 2; ref sentinel (==).
        {"item_key": "1", "horizon": 2, "mc_stop": 1, "ref_stop": 2},
        # both finite; ref crosses at the final prefix of horizon 10.
        {"item_key": "3", "horizon": 10, "mc_stop": 0, "ref_stop": 9},
        # both at the exact sentinel (7 == 7).
        {"item_key": "4", "horizon": 7, "mc_stop": 7, "ref_stop": 7},
    ]


class TestR080MapCalibrationLabel:
    def test_performat_maps_to_format_specific(self):
        mod = _phase4_records()
        assert mod.map_calibration_label("performat") == "format_specific"

    def test_shared_is_identity(self):
        assert _phase4_records().map_calibration_label("shared") == "shared"

    def test_format_specific_is_idempotent(self):
        assert (
            _phase4_records().map_calibration_label("format_specific")
            == "format_specific"
        )

    def test_unknown_label_raises(self):
        with pytest.raises((schema.ColmAimsError, ValueError)):
            _phase4_records().map_calibration_label("perfmt")


class TestR080ExportRecords:
    def test_writes_records_path_under_out_dir(self, tmp_path):
        out = _phase4_records().export_records(
            _scored_items(), "khard__format_specific", tmp_path
        )
        assert out == tmp_path / "records" / "khard__format_specific.jsonl"
        assert out.is_file()

    def test_exported_rows_load_under_the_v2_oracle(self, tmp_path):
        # The EXISTING v2 ingestion is the oracle: typed jsonl ingress +
        # per-record validation + complete classification.
        out = _phase4_records().export_records(
            _scored_items(), "khard__format_specific", tmp_path
        )
        loaded = schema.load_records_bytes(
            out.read_bytes(), "records/khard__format_specific.jsonl"
        )
        records = loaded["records"]
        assert len(records) == 4
        for record in records:
            schema.validate_record(record)
            assert pairing.classify_record(record)["status"] == "complete"

    def test_rows_sorted_by_item_key(self, tmp_path):
        out = _phase4_records().export_records(
            _scored_items(), "khard__format_specific", tmp_path
        )
        keys = [
            json.loads(line)["item_key"]
            for line in out.read_text("utf-8").splitlines()
            if line.strip()
        ]
        assert keys == ["1", "2", "3", "4"]

    def test_output_keys_follow_the_profile_declared_derivation(self, tmp_path):
        out = _phase4_records().export_records(
            _scored_items(), "khard__format_specific", tmp_path
        )
        observed = {
            json.loads(line)["item_key"]
            for line in out.read_text("utf-8").splitlines()
            if line.strip()
        }
        assert observed == {item["item_key"] for item in _scored_items()}
        assert all(
            schema.item_key_conforms_to_derivation(
                key, schema.PHASE4_ITEM_KEY_DERIVATION
            )
            for key in observed
        )

    def test_canonical_events_and_field_set(self, tmp_path):
        out = _phase4_records().export_records(
            _scored_items(), "khard__format_specific", tmp_path
        )
        by_key = {
            r["item_key"]: r
            for r in (
                json.loads(line)
                for line in out.read_text("utf-8").splitlines()
                if line.strip()
            )
        }
        expected_fields = {
            "item_key",
            "trajectory_horizon",
            "mc_event_status",
            "mc_stop_step",
            "mc_terminal_imputation",
            "ref_event_status",
            "ref_stop_step",
            "ref_terminal_imputation",
        }
        for record in by_key.values():
            assert set(record) == expected_fields
        # Sentinel (stop == horizon) => NEVER_STOPPED, null stop, imputed.
        b = by_key["2"]
        assert b["mc_event_status"] == EVENT_NEVER
        assert b["mc_stop_step"] is None
        assert b["mc_terminal_imputation"] == IMPUTATION_FINAL_PREFIX
        assert b["ref_event_status"] == EVENT_FINITE
        assert b["ref_stop_step"] == 1
        assert b["ref_terminal_imputation"] == IMPUTATION_NONE
        assert b["trajectory_horizon"] == 4
        # Final-prefix crossing at horizon 2 stays FINITE (R-046).
        a = by_key["1"]
        assert a["mc_event_status"] == EVENT_FINITE
        assert a["mc_stop_step"] == 1
        assert a["trajectory_horizon"] == 2
        assert a["ref_event_status"] == EVENT_NEVER
        assert a["ref_stop_step"] is None
        # Exact sentinel (7 == 7) on both arms codes as NEVER (F-7: the
        # former overshoot leg of this assertion moved to refusal tests).
        d = by_key["4"]
        assert d["mc_event_status"] == EVENT_NEVER
        assert d["ref_event_status"] == EVENT_NEVER
        # Final-prefix crossing at horizon 10.
        c = by_key["3"]
        assert c["ref_event_status"] == EVENT_FINITE
        assert c["ref_stop_step"] == 9

    def test_derived_reporting_encoding_stays_distinct(self, tmp_path):
        # R-046: the DP scalar is recomputable from the canonical record via
        # pairing.sentinel_coded_stop — never stored into stop_step.
        out = _phase4_records().export_records(
            _scored_items(), "khard__format_specific", tmp_path
        )
        by_key = {
            r["item_key"]: r
            for r in (
                json.loads(line)
                for line in out.read_text("utf-8").splitlines()
                if line.strip()
            )
        }
        assert pairing.sentinel_coded_stop(
            by_key["2"], "mc"
        ) == 4
        assert pairing.sentinel_coded_stop(
            by_key["4"], "mc"
        ) == 7
        assert pairing.sentinel_coded_stop(
            by_key["4"], "ref"
        ) == 7
        assert pairing.sentinel_coded_stop(
            by_key["3"], "ref"
        ) == 9

    def test_export_is_deterministic_under_input_permutation(self, tmp_path):
        mod = _phase4_records()
        out1 = mod.export_records(
            _scored_items(), "khard__format_specific", tmp_path / "one"
        )
        shuffled = list(reversed(_scored_items()))
        out2 = mod.export_records(
            shuffled, "khard__format_specific", tmp_path / "two"
        )
        assert out1.read_bytes() == out2.read_bytes()

    def test_legacy_performat_cell_id_is_refused(self, tmp_path):
        # The boundary translation is explicit: a performat-labelled cell id
        # is refused with guidance toward format_specific, in BOTH the
        # legacy "+" spelling and a smuggled "__" spelling.
        mod = _phase4_records()
        for bad in ("khard+performat", "khard__performat"):
            with pytest.raises(
                (schema.ColmAimsError, ValueError)
            ) as excinfo:
                mod.export_records(_scored_items(), bad, tmp_path)
            assert "format_specific" in str(excinfo.value)

    def test_duplicate_item_key_is_refused(self, tmp_path):
        items = _scored_items()
        items.append(dict(items[0]))
        with pytest.raises((schema.ColmAimsError, ValueError)):
            _phase4_records().export_records(
                items, "khard__format_specific", tmp_path
            )

    @pytest.mark.parametrize("bad_key", ["01", "-1", "1.0", "caf\u00e9"])
    def test_noncanonical_qid_is_refused(self, tmp_path, bad_key):
        items = _scored_items()
        items[0]["item_key"] = bad_key
        with pytest.raises(schema.ColmAimsError, match="canonical unsigned"):
            _phase4_records().export_records(
                items, "khard__format_specific", tmp_path
            )

    def test_bool_stop_is_refused_not_laundered(self, tmp_path):
        items = _scored_items()
        items[0]["mc_stop"] = True
        with pytest.raises((schema.ColmAimsError, ValueError)):
            _phase4_records().export_records(
                items, "khard__format_specific", tmp_path
            )

    def test_bool_horizon_is_refused(self, tmp_path):
        items = _scored_items()
        items[0]["horizon"] = True
        with pytest.raises((schema.ColmAimsError, ValueError)):
            _phase4_records().export_records(
                items, "khard__format_specific", tmp_path
            )

    def test_float_stop_is_refused(self, tmp_path):
        items = _scored_items()
        items[0]["mc_stop"] = 2.5
        with pytest.raises((schema.ColmAimsError, ValueError)):
            _phase4_records().export_records(
                items, "khard__format_specific", tmp_path
            )

    def test_negative_stop_is_refused(self, tmp_path):
        items = _scored_items()
        items[0]["ref_stop"] = -1
        with pytest.raises((schema.ColmAimsError, ValueError)):
            _phase4_records().export_records(
                items, "khard__format_specific", tmp_path
            )

    def test_horizon_below_two_is_refused(self, tmp_path):
        # DECISION: eligible horizons are >= 2 (single-prefix items are
        # excluded upstream by the frozen eligibility artifact).
        items = _scored_items()
        items[0]["horizon"] = 1
        items[0]["mc_stop"] = 0
        items[0]["ref_stop"] = 0
        with pytest.raises((schema.ColmAimsError, ValueError)):
            _phase4_records().export_records(
                items, "khard__format_specific", tmp_path
            )

    def test_unknown_item_field_is_refused(self, tmp_path):
        # Closed input contract: callers project scored frames down to
        # exactly {item_key, horizon, mc_stop, ref_stop}.
        items = _scored_items()
        items[0]["p_calibrated"] = [0.5]
        with pytest.raises((schema.ColmAimsError, ValueError)):
            _phase4_records().export_records(
                items, "khard__format_specific", tmp_path
            )

    def test_missing_item_field_is_refused(self, tmp_path):
        items = _scored_items()
        del items[0]["ref_stop"]
        with pytest.raises((schema.ColmAimsError, ValueError)):
            _phase4_records().export_records(
                items, "khard__format_specific", tmp_path
            )


# ===========================================================================
# R-072 (amended): QA-012 rev2 manifest pins
# ===========================================================================


class TestR072Rev2Manifest:
    # Source: qa012_inventory_2026-08-22_rev2.json (repo root; adjudicated
    # 2026-08-22, supersedes the VOID rev1).

    @pytest.fixture(scope="class")
    def manifest(self):
        return _load_json(QA012_REV2_MANIFEST_PATH)

    def test_manifest_bytes_hash_to_the_adjudicated_sha(self):
        assert sha256_file(QA012_REV2_MANIFEST_PATH) == QA012_REV2_SHA

    def test_verdict_hits_present_not_vacuous(self, manifest):
        assert manifest["verdict"] == "HITS_PRESENT_NOT_VACUOUS"

    def test_total_hits_4556(self, manifest):
        assert manifest["total_format_qa_hits"] == 4556

    def test_hits_by_file_sums_to_total_over_four_files(self, manifest):
        hits = manifest["hits_by_file"]
        assert len(hits) == 4
        assert sum(hits.values()) == 4556

    def test_supersedes_names_rev1_by_sha(self, manifest):
        supersedes = manifest["supersedes"]
        assert supersedes["sha256"] == QA012_REV1_SHA
        assert supersedes["file"] == "qa012_inventory_2026-08-21.json"

    def test_corrected_scope_and_revision(self, manifest):
        assert manifest["revision"] == 2
        assert manifest["files_scanned"] == 67
        assert manifest["parse_failures"] == []


# ===========================================================================
# FIX ROUND (adversarial critic, 2026-08-22): F-1..F-7 exploit tests.
# Written RED-first against the GREEN implementation; current-behavior
# probes for F-3/F-5/F-7 confirmed the defects live before pinning.
# ===========================================================================


def _producer():
    from scripts import stopdff_fair_qa_retest as producer

    return producer


GATE_SENTINEL_NAMES = (
    "staged_gate",
    "eligibility_load",
    "snapshot_verify",
    "dataset_load",
    "model_construct",
)


def _gate_args(**overrides):
    """Minimal argparse-shaped namespace for the run_phase4_gates seam."""
    base = {
        "eligibility": str(ELIGIBILITY_PATH),
        "records_out": None,
        "staged_input": [],
        "primary_model_path": None,
        "disjoint_model_path": None,
        "snapshot_manifest": None,
        "calibration": "paper_exports/calibration_train.json",
    }
    base.update(overrides)
    return SimpleNamespace(**base)


# ---------------------------------------------------------------------------
# F-1: staged-input sibling coverage (required_staged_coverage)
# ---------------------------------------------------------------------------


class TestF1RequiredStagedCoverage:
    def _consumed(self, tmp_path):
        return [
            {
                "label": "calibration_train",
                "path": tmp_path / "calibration_train.json",
                "frozen_sha256": CALIB_TRAIN_SHA,
            },
            {
                "label": "eval_split",
                "path": tmp_path / "test_dataset.json",
                "frozen_sha256": TEST_DATASET_SHA,
            },
            {
                "label": "fit_split",
                "path": tmp_path / "val_dataset.json",
                "frozen_sha256": None,
            },
            {
                "label": "mc_dataset",
                "path": tmp_path / "mc_dataset.json",
                "frozen_sha256": None,
            },
        ]

    def _operator(self, consumed, index, digest):
        return {
            "label": f"op_{consumed[index]['label']}",
            "path": consumed[index]["path"],
            "expected_sha256": digest,
        }

    def test_full_coverage_resolves_in_consumed_order(self, tmp_path):
        consumed = self._consumed(tmp_path)
        staged = [
            self._operator(consumed, 2, "a" * 64),
            self._operator(consumed, 3, "b" * 64),
        ]
        plan = _phase4().required_staged_coverage(consumed, staged)
        assert [entry["label"] for entry in plan] == [
            "calibration_train",
            "eval_split",
            "fit_split",
            "mc_dataset",
        ]
        expected = [CALIB_TRAIN_SHA, TEST_DATASET_SHA, "a" * 64, "b" * 64]
        assert [entry["expected_sha256"] for entry in plan] == expected

    def test_eval_split_frozen_pin_comes_from_eligibility_artifact(
        self, tmp_path
    ):
        # The wiring pin: the eval-split frozen digest IS the eligibility
        # artifact's derived_from.test_dataset_sha256 (two-party pinned).
        eligibility = _load_json(ELIGIBILITY_PATH)
        consumed = [
            {
                "label": "eval_split",
                "path": tmp_path / "test_dataset.json",
                "frozen_sha256": eligibility["derived_from"][
                    "test_dataset_sha256"
                ],
            }
        ]
        plan = _phase4().required_staged_coverage(consumed, [])
        assert plan[0]["expected_sha256"] == TEST_DATASET_SHA

    def test_operator_entry_agreeing_with_frozen_pin_is_fine(self, tmp_path):
        consumed = self._consumed(tmp_path)[:2]
        staged = [self._operator(consumed, 0, CALIB_TRAIN_SHA)]
        plan = _phase4().required_staged_coverage(consumed, staged)
        assert plan[0]["expected_sha256"] == CALIB_TRAIN_SHA

    def test_uncovered_input_raises_naming_it(self, tmp_path):
        # (a) fit_split has no frozen pin and no operator digest.
        consumed = self._consumed(tmp_path)
        staged = [self._operator(consumed, 3, "b" * 64)]
        with pytest.raises(schema.TypedIngressError) as excinfo:
            _phase4().required_staged_coverage(consumed, staged)
        assert "val_dataset.json" in str(excinfo.value)

    def test_operator_digest_contradicting_frozen_pin_raises_both(
        self, tmp_path
    ):
        # (b) substitution-negative: a DIFFERENT valid digest for a
        # frozen-pinned file must refuse, naming the file and BOTH digests.
        consumed = self._consumed(tmp_path)[:2]
        contradicting = "9" * 64
        staged = [self._operator(consumed, 0, contradicting)]
        with pytest.raises(schema.TypedIngressError) as excinfo:
            _phase4().required_staged_coverage(consumed, staged)
        message = str(excinfo.value)
        assert "calibration_train.json" in message
        assert CALIB_TRAIN_SHA in message
        assert contradicting in message

    def test_unknown_operator_path_raises(self, tmp_path):
        # (c) an operator entry outside the consumed set is a defect, not
        # silently ignored coverage.
        consumed = self._consumed(tmp_path)[:2]
        staged = [
            {
                "label": "mystery",
                "path": tmp_path / "not_consumed.json",
                "expected_sha256": "c" * 64,
            }
        ]
        with pytest.raises(schema.TypedIngressError) as excinfo:
            _phase4().required_staged_coverage(consumed, staged)
        assert "not_consumed.json" in str(excinfo.value)


# ---------------------------------------------------------------------------
# F-2a: gate-ordering seam (run_phase4_gates)
# ---------------------------------------------------------------------------


class TestF2GateOrdering:
    def _recording_sentinels(self, calls, raising=None):
        sentinels = {}
        for name in GATE_SENTINEL_NAMES:
            if raising is not None and name == raising[0]:
                exc = raising[1]

                def _raiser(*args, _name=name, _exc=exc, **kwargs):
                    calls.append(_name)
                    raise _exc

                sentinels[name] = _raiser
            else:

                def _recorder(*args, _name=name, **kwargs):
                    calls.append(_name)

                sentinels[name] = _recorder
        return sentinels

    def test_gates_run_strictly_before_dataset_and_model(self):
        producer = _producer()
        calls: list[str] = []
        producer.run_phase4_gates(
            _gate_args(), sentinels=self._recording_sentinels(calls)
        )
        assert calls == list(GATE_SENTINEL_NAMES)

    def test_staged_gate_failure_aborts_before_any_later_stage(self):
        producer = _producer()
        phase4 = _phase4()
        calls: list[str] = []
        sentinels = self._recording_sentinels(
            calls,
            raising=(
                "staged_gate",
                phase4.StagedInputError("staged gate failed (test)"),
            ),
        )
        with pytest.raises(schema.TypedIngressError):
            producer.run_phase4_gates(_gate_args(), sentinels=sentinels)
        assert calls == ["staged_gate"]


# ---------------------------------------------------------------------------
# F-2b: producer fatality helpers (exist since GREEN; coverage was missing)
# ---------------------------------------------------------------------------


class TestF2ProducerFatalityHelpers:
    def test_paired_population_equal_is_silent(self):
        _producer()._require_frozen_paired_population(
            cell="khard+shared",
            paired_ids={"1", "2", "3"},
            eligible=frozenset({"1", "2", "3"}),
        )

    def test_silent_exclusion_is_refused(self):
        with pytest.raises(ValueError) as excinfo:
            _producer()._require_frozen_paired_population(
                cell="khard+shared",
                paired_ids={"1", "2"},
                eligible=frozenset({"1", "2", "3"}),
            )
        assert "3" in str(excinfo.value)

    def test_silent_inclusion_is_refused(self):
        # The other direction: an item OUTSIDE the frozen set sneaking into
        # the paired population is equally fatal.
        with pytest.raises(ValueError) as excinfo:
            _producer()._require_frozen_paired_population(
                cell="khard+shared",
                paired_ids={"1", "2", "3", "99"},
                eligible=frozenset({"1", "2", "3"}),
            )
        assert "99" in str(excinfo.value)

    def test_consistent_horizons_are_silent(self):
        _producer()._require_consistent_item_horizons(
            cell="khard+shared",
            stops={"7": {"mc_horizon": 4, "ref_horizon": 4}},
            horizon_map={"7": 4},
        )

    def test_mc_vs_qa_horizon_mismatch_is_refused(self):
        with pytest.raises(ValueError) as excinfo:
            _producer()._require_consistent_item_horizons(
                cell="khard+shared",
                stops={"7": {"mc_horizon": 4, "ref_horizon": 5}},
                horizon_map=None,
            )
        assert "7" in str(excinfo.value)

    def test_observed_horizon_differing_from_frozen_is_refused(self):
        # Substitution-negative: 5 is a VALID horizon, just not the frozen
        # one for this qid.
        with pytest.raises(ValueError) as excinfo:
            _producer()._require_consistent_item_horizons(
                cell="khard+shared",
                stops={"7": {"mc_horizon": 5, "ref_horizon": 5}},
                horizon_map={"7": 4},
            )
        assert "R-073" in str(excinfo.value)


# ---------------------------------------------------------------------------
# F-2c: phase4 metadata block (pure builder)
# ---------------------------------------------------------------------------


class TestF2Phase4MetadataBlock:
    def _block(self):
        return _producer().phase4_metadata_block(
            interpreter_realpath="/repo/.venv/bin/python3.11",
            os_name="Darwin",
            arch="arm64",
            device="cpu",
            pythonhashseed="0",
            seeds=[1],
            offline_flags_set=True,
            fitted_platt_digests={"shared": "a" * 64},
            continuation_estimator_digests={"idealized+shared": "b" * 64},
            staged_receipt=[
                {
                    "label": "calibration_train",
                    "path": "paper_exports/calibration_train.json",
                    "expected_sha256": CALIB_TRAIN_SHA,
                    "observed_sha256": CALIB_TRAIN_SHA,
                }
            ],
            eligibility={
                "pairing_population_keyset_sha256": "d" * 64,
                "horizon_map_sha256": "e" * 64,
            },
        )

    def test_receipt_carries_fit_digests_verbatim(self):
        block = self._block()
        assert block["fitted_platt_digests"] == {"shared": "a" * 64}
        assert block["continuation_estimator_digests"] == {
            "idealized+shared": "b" * 64
        }

    def test_receipt_field_presence_and_rng_flags(self):
        block = self._block()
        for key in (
            "interpreter_realpath",
            "os",
            "arch",
            "device",
            "pythonhashseed",
            "seeds",
            "offline_flags_set",
            "fitted_platt_digests",
            "continuation_estimator_digests",
            "staged_inputs",
            "eligibility_keyset_sha256",
            "eligibility_horizon_map_sha256",
        ):
            assert key in block, key
        assert block["archived_rng_pinned"] is False
        assert block["fresh_rng_pinned"] is True
        assert block["staged_inputs"][0]["observed_sha256"] == CALIB_TRAIN_SHA
        assert block["eligibility_horizon_map_sha256"] == "e" * 64


# ---------------------------------------------------------------------------
# F-3: comparator must refuse a truncated anchor (never a vacuous PASS)
# ---------------------------------------------------------------------------


class TestF3TruncatedAnchor:
    def _not_pass(self, anchor, regen):
        """Contract freedom: typed refusal at anchor validation OR a FAIL
        verdict — but NEVER PASS over a sub-allowlist comparison."""
        try:
            result = _phase4().compare_parity(anchor, regen)
        except schema.ColmAimsError:
            return
        assert result["verdict"] != "PASS", (
            f"truncated anchor produced PASS with checked="
            f"{result.get('checked')!r} — a comparison over fewer than the"
            " full 194-field allowlist can never PASS (amended R-077)"
        )

    def test_cell_truncated_anchor_cannot_pass(self):
        anchor = _anchor()
        regen = _regen_from_anchor(anchor)
        truncated = _copy(anchor)
        for cell in ("klex+shared", "klex+performat"):
            truncated["nonrandom_cells"].remove(cell)
            del truncated["expected"][cell]
        self._not_pass(truncated, regen)

    def test_point_field_truncated_anchor_cannot_pass(self):
        # Sibling site: same exploit through the FIELD axis instead of the
        # cell axis.
        anchor = _anchor()
        regen = _regen_from_anchor(anchor)
        truncated = _copy(anchor)
        dropped = truncated["point_fields"].pop()
        for cell in truncated["nonrandom_cells"]:
            for policy in truncated["policies"]:
                truncated["expected"][cell][policy].pop(dropped, None)
        self._not_pass(truncated, regen)

    def test_ci_field_truncated_anchor_cannot_pass(self):
        anchor = _anchor()
        regen = _regen_from_anchor(anchor)
        truncated = _copy(anchor)
        dropped = truncated["ci_fields"].pop()
        for cell in truncated["nonrandom_cells"]:
            for policy in truncated["policies"]:
                truncated["expected"][cell][policy].pop(dropped, None)
        self._not_pass(truncated, regen)


# ---------------------------------------------------------------------------
# F-5: cross-type numeric laundering in the comparator
# ---------------------------------------------------------------------------


class TestF5CrossTypeParity:
    def test_int_field_drifted_to_float_fails(self):
        # 2249 == 2249.0 in Python — the amended R-077 same-JSON-type rule
        # must still flag the drift (int is not float).
        anchor = _anchor()
        regen = _regen_from_anchor(anchor)
        assert isinstance(
            anchor["expected"]["idealized+shared"]["dp"]["n"], int
        )
        regen["results"]["idealized+shared"]["dp"]["n"] = 2249.0
        result = _phase4().compare_parity(anchor, regen)
        assert result["verdict"] == "FAIL"
        assert any(
            f["cell"] == "idealized+shared"
            and f["policy"] == "dp"
            and f["field"] == "n"
            for f in result["failures"]
        )

    def test_identity_field_drifted_to_float_fails(self):
        # Sibling site: the identity leg must apply the same type rule.
        anchor = _anchor()
        regen = _regen_from_anchor(anchor)
        regen["metadata"]["n_eval"] = 2258.0
        result = _phase4().compare_parity(anchor, regen)
        assert result["verdict"] == "FAIL"
        assert any(f["field"] == "n_eval" for f in result["failures"])

    def test_string_encoded_number_fails(self):
        # Regression pin (already correct in GREEN): "0.8528" is a string,
        # never numeric-equal.
        anchor = _anchor()
        regen = _regen_from_anchor(anchor)
        expected = anchor["expected"]["idealized+shared"]["dp"]["signed_mean"]
        regen["results"]["idealized+shared"]["dp"]["signed_mean"] = str(
            expected
        )
        result = _phase4().compare_parity(anchor, regen)
        assert result["verdict"] == "FAIL"
        assert any(
            f["field"] == "signed_mean" for f in result["failures"]
        )


# ---------------------------------------------------------------------------
# F-6: flag coupling — --records-out requires --eligibility at parse time
# ---------------------------------------------------------------------------


class TestF6FlagCoupling:
    def test_records_out_without_eligibility_exits_2_before_gates(
        self, tmp_path, monkeypatch
    ):
        # Records regenerated outside the frozen paired population are
        # unusable — refuse at ARGUMENT validation (SystemExit 2), before
        # any staged gate, artifact load, or dataset read fires.
        producer = _producer()
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "stopdff_fair_qa_retest.py",
                "--records-out",
                str(tmp_path / "records"),
                "--calibration",
                str(tmp_path / "missing_calibration.json"),
                "--data-dir",
                str(tmp_path / "nodata"),
                "--out",
                str(tmp_path / "out.json"),
            ],
        )
        with pytest.raises(SystemExit) as excinfo:
            producer.main()
        assert excinfo.value.code == 2

    def test_records_out_with_eligibility_but_no_snapshots_exits_2(
        self, tmp_path, monkeypatch
    ):
        # Eligibility alone cannot authorize identity-bearing record bytes:
        # both models must come from verified local snapshots.
        producer = _producer()
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "stopdff_fair_qa_retest.py",
                "--records-out",
                str(tmp_path / "records"),
                "--eligibility",
                str(ELIGIBILITY_PATH),
                "--calibration",
                str(tmp_path / "missing_calibration.json"),
                "--data-dir",
                str(tmp_path / "nodata"),
                "--out",
                str(tmp_path / "out.json"),
            ],
        )
        with pytest.raises(SystemExit) as excinfo:
            producer.main()
        assert excinfo.value.code == 2

    def test_records_out_with_complete_snapshot_flags_passes_validation(
        self, tmp_path, monkeypatch
    ):
        # Control: once all authority flags are present, argument validation
        # succeeds and the missing staged calibration is a typed gate error.
        producer = _producer()
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "stopdff_fair_qa_retest.py",
                "--records-out",
                str(tmp_path / "records"),
                "--eligibility",
                str(ELIGIBILITY_PATH),
                "--snapshot-manifest",
                str(tmp_path / "model_snapshot_manifests.json"),
                "--primary-model-path",
                str(tmp_path / "primary"),
                "--disjoint-model-path",
                str(tmp_path / "disjoint"),
                "--calibration",
                str(tmp_path / "missing_calibration.json"),
                "--data-dir",
                str(tmp_path / "nodata"),
                "--out",
                str(tmp_path / "out.json"),
            ],
        )
        with pytest.raises(schema.TypedIngressError):
            producer.main()

    @pytest.mark.parametrize(
        "present_flags",
        [
            ("--snapshot-manifest",),
            ("--primary-model-path",),
            ("--disjoint-model-path",),
            ("--snapshot-manifest", "--primary-model-path"),
            ("--snapshot-manifest", "--disjoint-model-path"),
            ("--primary-model-path", "--disjoint-model-path"),
        ],
    )
    def test_records_out_refuses_each_incomplete_snapshot_binding(
        self, tmp_path, monkeypatch, present_flags
    ):
        argv = [
            "stopdff_fair_qa_retest.py",
            "--records-out",
            str(tmp_path / "records"),
            "--eligibility",
            str(ELIGIBILITY_PATH),
        ]
        values = {
            "--snapshot-manifest": tmp_path / "manifest.json",
            "--primary-model-path": tmp_path / "primary",
            "--disjoint-model-path": tmp_path / "disjoint",
        }
        for flag in present_flags:
            argv.extend((flag, str(values[flag])))
        monkeypatch.setattr(sys, "argv", argv)

        with pytest.raises(SystemExit) as excinfo:
            _producer().main()
        assert excinfo.value.code == 2


# ---------------------------------------------------------------------------
# F-7: stop > horizon is frame corruption, never absorbed (amended R-080)
# ---------------------------------------------------------------------------


class TestF7OvershootRefusal:
    def test_mc_stop_one_past_horizon_is_refused_as_corruption(
        self, tmp_path
    ):
        items = _scored_items()
        items[0]["mc_stop"] = items[0]["horizon"] + 1
        with pytest.raises(
            (schema.ColmAimsError, ValueError)
        ) as excinfo:
            _phase4_records().export_records(
                items, "khard__format_specific", tmp_path
            )
        assert "corrupt" in str(excinfo.value).lower()

    def test_ref_overshoot_is_refused_on_the_sibling_arm(self, tmp_path):
        items = _scored_items()
        items[2]["ref_stop"] = items[2]["horizon"] + 3
        with pytest.raises(
            (schema.ColmAimsError, ValueError)
        ) as excinfo:
            _phase4_records().export_records(
                items, "khard__format_specific", tmp_path
            )
        assert "corrupt" in str(excinfo.value).lower()

    def test_exact_sentinel_still_exports_never_stopped(self, tmp_path):
        # Nearest-true control: the tightening must not break the exact
        # sentinel (stop == horizon).
        out = _phase4_records().export_records(
            _scored_items(), "khard__format_specific", tmp_path
        )
        rows = [
            json.loads(line)
            for line in out.read_text("utf-8").splitlines()
            if line.strip()
        ]
        d = next(r for r in rows if r["item_key"] == "4")
        assert d["mc_event_status"] == EVENT_NEVER
        assert d["mc_stop_step"] is None


# ---------------------------------------------------------------------------
# F-4 companion: assemble_certificate requires the R-070 receipt fields
# ---------------------------------------------------------------------------


class TestR070ReceiptFieldsInAssemble:
    def test_receipt_missing_environment_lock_hash_fails(self):
        components = _good_components()
        del components["suite_receipts"]["focused"]["environment_lock_sha256"]
        cert = _phase4().assemble_certificate(components)
        assert cert["ready"] is False
        assert any("suite" in check for check in cert["failing_checks"])

    def test_receipt_missing_counts_fails(self):
        components = _good_components()
        del components["suite_receipts"]["full"]["counts"]
        cert = _phase4().assemble_certificate(components)
        assert cert["ready"] is False
        assert any("suite" in check for check in cert["failing_checks"])


# ---------------------------------------------------------------------------
# F-4: gather_certificate_components (runner-injectable gatherer)
# ---------------------------------------------------------------------------

FAKE_COMMIT = "a1" * 20
FAKE_TREE = "b2" * 32


def _fake_git_runner(
    *,
    commit=FAKE_COMMIT,
    tree=FAKE_TREE,
    status="",
    untracked_status="",
):
    """run(cmd) -> stdout. Git identities come from HERE, never from caller
    assertions. The fake tree id is 64-hex (sha256-object-format repos); see
    the CONTRACT_DEFECT note in TestF4GatherCertificateComponents."""
    calls: list[list[str]] = []

    def run(cmd):
        cmd_list = [str(part) for part in cmd]
        calls.append(cmd_list)
        joined = " ".join(cmd_list)
        assert "git" in joined, f"unexpected non-git command: {cmd_list}"
        if "status" in joined and "--untracked-files=all" in cmd_list:
            return untracked_status
        if "status" in joined:
            return status
        if "HEAD^{tree}" in joined or "tree" in joined:
            return tree + "\n"
        if "rev-parse" in joined:
            return commit + "\n"
        raise AssertionError(f"unexpected git command: {cmd_list}")

    run.calls = calls
    return run


class TestF4GatherCertificateComponents:
    # CONTRACT_DEFECT R-079 (escalate, do not silently absorb): the existing
    # assemble_certificate._check_repo demands a 64-hex repo tree_sha256,
    # but `git rev-parse HEAD^{tree}` on a SHA-1-format repository (this
    # one) emits a 40-hex object id — the two constraints cannot both be
    # satisfied verbatim on this repo. These tests inject a 64-hex tree id
    # (valid for sha256-object-format git) and additionally pin only that
    # the gathered tree value is RUNNER-SOURCED (changing the runner output
    # changes the component). GREEN/spec must reconcile the real-repo form
    # (e.g. a declared tree_object_id -> tree_sha256 derivation) — flagged
    # in the RED report as SPEC_ISSUE-1.

    @pytest.fixture(autouse=True)
    def _synthetic_calibration_pin(self, monkeypatch):
        # This gatherer unit uses tiny synthetic bytes. Pin those bytes only
        # within this class; dedicated R-082 tests retain the archival pin.
        blob = json.dumps({"calibration_train": [1]}).encode("utf-8") + b"\n"
        monkeypatch.setattr(
            _phase4(), "CALIBRATION_TRAIN_SHA256", sha256_bytes(blob)
        )
        self._phase4_monkeypatch = monkeypatch

    def _config(self, tmp_path):
        fixture_interpreter = str(Path(sys.executable).resolve())
        phase4 = _phase4()
        repo_root = REPO_ROOT.resolve()
        staged_dir = tmp_path / "staged"
        staged_dir.mkdir()
        staged_blobs = {
            label: json.dumps({label: [index]}).encode("utf-8") + b"\n"
            for index, label in enumerate(_PHASE4_STAGED_FILENAMES, start=1)
        }
        for label, filename in _PHASE4_STAGED_FILENAMES.items():
            (staged_dir / filename).write_bytes(staged_blobs[label])
        staged_digests = {
            label: sha256_file(staged_dir / filename)
            for label, filename in _PHASE4_STAGED_FILENAMES.items()
        }
        self._phase4_monkeypatch.setattr(
            phase4, "R082_STAGED_INPUT_SHA256", dict(staged_digests)
        )
        eligibility = _load_json(ELIGIBILITY_PATH)
        eligibility["derived_from"]["test_dataset_sha256"] = staged_digests[
            "eval_split"
        ]
        eligibility_path = tmp_path / "pairing_eligibility_v2.json"
        eligibility_path.write_text(
            json.dumps(eligibility), encoding="utf-8"
        )
        self._phase4_monkeypatch.setattr(
            phase4,
            "ELIGIBILITY_ARTIFACT_RELPATH",
            str(eligibility_path.resolve()),
        )
        self._phase4_monkeypatch.setattr(
            phase4,
            "ELIGIBILITY_ARTIFACT_SHA256",
            sha256_file(eligibility_path),
        )
        self._phase4_monkeypatch.setattr(
            phase4,
            "ELIGIBILITY_TEST_DATASET_SHA256",
            staged_digests["eval_split"],
        )
        staged_plan = [
            {
                "label": label,
                "path": staged_dir / filename,
                "expected_sha256": staged_digests[label],
            }
            for label, filename in _PHASE4_STAGED_FILENAMES.items()
        ]
        manifest = _load_json(MODEL_MANIFEST_PATH)
        snapshot_dirs = {}
        for role in ("primary_scorer", "disjoint_selector"):
            snap = tmp_path / f"snap_{role}"
            (snap / "1_Pooling").mkdir(parents=True)
            blobs = {
                "config.json": json.dumps({"role": role}).encode() + b"\n",
                "1_Pooling/config.json": b'{"pooling": "mean"}\n',
            }
            for rel, blob in blobs.items():
                (snap / rel).write_bytes(blob)
            manifest["roles"][role]["files"] = {
                rel: {"sha256": sha256_bytes(blob), "size": len(blob)}
                for rel, blob in blobs.items()
            }
            manifest["roles"][role]["file_count"] = len(blobs)
            snapshot_dirs[role] = snap
        manifest_path = tmp_path / "model_snapshot_manifests.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        self._phase4_monkeypatch.setattr(
            phase4,
            "SNAPSHOT_MANIFEST_RELPATH",
            str(manifest_path.resolve()),
        )
        self._phase4_monkeypatch.setattr(
            phase4,
            "SNAPSHOT_MANIFEST_SHA256",
            sha256_file(manifest_path),
        )
        receipt_paths = {}
        for name in ("focused", "full"):
            path = tmp_path / f"suite_receipt_{name}.json"
            # R-082: receipt head bindings must match the RUNNER-sourced
            # repo component (FAKE_COMMIT/FAKE_TREE), or assemble refuses.
            path.write_text(
                json.dumps(
                    _r070_receipt(
                        f"pytest --suite {name}",
                        commit=FAKE_COMMIT,
                        tree_sha256=FAKE_TREE,
                        interpreter_realpath=fixture_interpreter,
                    )
                ),
                encoding="utf-8",
            )
            receipt_paths[name] = path
        content_paths = {
            key: repo_root / relpath
            for key, relpath in phase4.CONTENT_HASH_RELPATHS.items()
        }
        environment = _copy(_good_components()["environment"])
        environment["command"] = (
            [fixture_interpreter, "scripts/stopdff_fair_qa_retest.py"]
            + _staging_command_args(
                staged_dir, staged_digests, eligibility_path
            )
            + _experiment_command_args()
            + _snapshot_output_command_args(manifest_path, snapshot_dirs)
        )
        return {
            "repo_root": repo_root,
            "eligibility_path": eligibility_path,
            "snapshot_manifest_path": manifest_path,
            "snapshot_dirs": snapshot_dirs,
            "parity_anchor_path": PARITY_ANCHOR_PATH,
            "qa012_manifest_path": QA012_REV3_MANIFEST_PATH,
            "staged_plan": staged_plan,
            "suite_receipt_paths": receipt_paths,
            "content_hash_paths": content_paths,
            "environment": environment,
            "offline_flags": list(OFFLINE_FLAGS),
        }

    def test_good_config_gathers_ready_true(self, tmp_path):
        phase4 = _phase4()
        config = self._config(tmp_path)
        run = _fake_git_runner()
        components = phase4.gather_certificate_components(config, run=run)
        cert = phase4.assemble_certificate(components)
        assert cert["ready"] is True, cert["failing_checks"]

    def test_repo_identity_is_runner_sourced(self, tmp_path):
        phase4 = _phase4()
        config = self._config(tmp_path)
        components = phase4.gather_certificate_components(
            config, run=_fake_git_runner()
        )
        repo = components["repo"]
        assert repo["commit"] == FAKE_COMMIT
        assert repo["dirty"] is False
        # Runner-sourcing pin without over-pinning the derivation: the same
        # tree output reproduces the same component; a DIFFERENT tree output
        # must change it (substitution-negative).
        other = phase4.gather_certificate_components(
            config, run=_fake_git_runner(tree="c3" * 32)
        )
        again = phase4.gather_certificate_components(
            config, run=_fake_git_runner()
        )
        assert repo["tree_sha256"] == again["repo"]["tree_sha256"]
        assert repo["tree_sha256"] != other["repo"]["tree_sha256"]

    def test_git_commands_actually_flow_through_the_runner(self, tmp_path):
        phase4 = _phase4()
        run = _fake_git_runner()
        phase4.gather_certificate_components(self._config(tmp_path), run=run)
        joined = [" ".join(call) for call in run.calls]
        assert any("status" in call for call in joined)
        assert any("rev-parse" in call for call in joined)

    def test_dirty_git_status_output_flips_ready_false(self, tmp_path):
        phase4 = _phase4()
        components = phase4.gather_certificate_components(
            self._config(tmp_path),
            run=_fake_git_runner(status=" M scripts/stopdff_fair_qa_retest.py\n"),
        )
        assert components["repo"]["dirty"] is True
        cert = phase4.assemble_certificate(components)
        assert cert["ready"] is False
        assert any("repo" in check for check in cert["failing_checks"])

    def test_staged_inputs_are_rehashed_from_bytes(self, tmp_path):
        # Mirror-equality catcher: observed must be recomputed from the
        # file bytes, never copied from the expectation.
        phase4 = _phase4()
        config = self._config(tmp_path)
        real_hash = config["staged_plan"][1]["expected_sha256"]
        config["staged_plan"][1]["expected_sha256"] = "9" * 64
        components = phase4.gather_certificate_components(
            config, run=_fake_git_runner()
        )
        entry = components["staged_inputs"][1]
        assert entry["observed_sha256"] == real_hash
        assert entry["expected_sha256"] == "9" * 64
        cert = phase4.assemble_certificate(components)
        assert cert["ready"] is False
        assert any(
            "staged_inputs" in check for check in cert["failing_checks"]
        )

    def test_content_and_artifact_hashes_are_recomputed(self, tmp_path):
        phase4 = _phase4()
        config = self._config(tmp_path)
        components = phase4.gather_certificate_components(
            config, run=_fake_git_runner()
        )
        for key, path in config["content_hash_paths"].items():
            assert components["content_hashes"][key] == {
                "artifact_path": str(path.resolve()),
                "sha256": sha256_file(path),
            }
        assert components["parity"]["anchor_sha256"] == sha256_file(
            PARITY_ANCHOR_PATH
        )
        assert components["qa012"]["manifest_sha256"] == (
            phase4.QA012_MANIFEST_SHA256
        )
        eligibility = _load_json(ELIGIBILITY_PATH)
        assert (
            components["eligibility"]["digest"]
            == eligibility["pairing_population_keyset_sha256"]
        )
        assert (
            components["eligibility"]["horizon_map_sha256"]
            == eligibility["horizon_map_sha256"]
        )
        # R-082: staged paths recorded ABSOLUTE (out-of-repo staging —
        # identity via hash gates, location outside the tracked tree).
        for entry in components["staged_inputs"]:
            assert Path(str(entry["path"])).is_absolute()

    def test_mutated_snapshot_is_recorded_not_raised(self, tmp_path):
        # Gather RECORDS check failures; assemble decides (guarded builder).
        phase4 = _phase4()
        config = self._config(tmp_path)
        victim = config["snapshot_dirs"]["primary_scorer"] / "config.json"
        victim.write_bytes(victim.read_bytes() + b"tamper\n")
        components = phase4.gather_certificate_components(
            config, run=_fake_git_runner()
        )
        assert components["snapshots"]["primary_scorer"]["verified"] is False
        cert = phase4.assemble_certificate(components)
        assert cert["ready"] is False
        assert any("snapshot" in check for check in cert["failing_checks"])

    def test_receipt_file_missing_environment_lock_fails(self, tmp_path):
        phase4 = _phase4()
        config = self._config(tmp_path)
        receipt = _r070_receipt(
            "pytest --suite focused",
            commit=FAKE_COMMIT,
            tree_sha256=FAKE_TREE,
        )
        del receipt["environment_lock_sha256"]
        config["suite_receipt_paths"]["focused"].write_text(
            json.dumps(receipt), encoding="utf-8"
        )
        components = phase4.gather_certificate_components(
            config, run=_fake_git_runner()
        )
        cert = phase4.assemble_certificate(components)
        assert cert["ready"] is False
        assert any("suite" in check for check in cert["failing_checks"])


# ===========================================================================
# OPERATIONAL-REJECTION ROUND (ChatGPT FALSE_READY readback, 2026-08-22):
# R-081 launcher, R-082 head-bound receipts, amended R-077 Random-K
# structure, amended R-080 parent-dir, R-072 rev3. RED-first; current
# behavior probed before pinning (missing-krandom => PASS, doubled
# records/records segment written silently, receipts head-blind).
# ===========================================================================


def _phase4_launcher():
    from reproducibility.colm_aims_2026 import phase4_launcher

    return phase4_launcher


# ---------------------------------------------------------------------------
# Amended R-077: Random-K STRUCTURE is required (values stay exempt)
# ---------------------------------------------------------------------------


class TestR077RandomKStructure:
    def test_missing_both_krandom_cells_is_a_blocking_structural_fail(self):
        # Probe-confirmed current defect: deleting BOTH krandom cells still
        # returns PASS. Structure is now blocking; whole-cell rows carry
        # field "<cell>".
        anchor = _anchor()
        regen = _regen_from_anchor(anchor)
        del regen["results"]["krandom+shared"]
        del regen["results"]["krandom+performat"]
        result = _phase4().compare_parity(anchor, regen)
        assert result["verdict"] == "FAIL"
        for cell in ("krandom+shared", "krandom+performat"):
            rows = [f for f in result["failures"] if f["cell"] == cell]
            assert rows, f"no structural failure row for missing {cell}"
            assert any(f["field"] == "<cell>" for f in rows)

    def test_missing_krandom_point_field_is_structural_fail(self):
        anchor = _anchor()
        regen = _regen_from_anchor(anchor)
        del regen["results"]["krandom+performat"]["dp"]["n"]
        result = _phase4().compare_parity(anchor, regen)
        assert result["verdict"] == "FAIL"
        assert any(
            f["cell"] == "krandom+performat" and f["field"] == "n"
            for f in result["failures"]
        )

    def test_missing_krandom_ci_field_is_structural_fail(self):
        # Sibling site: the CI-field axis of the same structural check.
        anchor = _anchor()
        regen = _regen_from_anchor(anchor)
        del regen["results"]["krandom+shared"]["myopic"]["signed_mean_ci"]
        result = _phase4().compare_parity(anchor, regen)
        assert result["verdict"] == "FAIL"
        assert any(
            f["cell"] == "krandom+shared"
            and f["field"] == "signed_mean_ci"
            for f in result["failures"]
        )

    def test_structural_checks_do_not_inflate_checked(self):
        # The 194 blocking VALUE comparisons stay the PASS invariant;
        # structure rides in failures only. (Complete-structure wild-value
        # exemption is pinned by test_random_k_divergence_is_informational
        # _never_blocking above.)
        anchor = _anchor()
        regen = _regen_from_anchor(anchor)
        result = _phase4().compare_parity(anchor, regen)
        assert result["verdict"] == "PASS"
        assert result["checked"] == 194


# ---------------------------------------------------------------------------
# R-082: head-bound suite receipts (assemble path)
# ---------------------------------------------------------------------------


class TestR082ReceiptHeadBinding:
    def _cert(self, mutate):
        components = _good_components()
        mutate(components["suite_receipts"])
        cert = _phase4().assemble_certificate(components)
        return cert

    def _assert_suite_fail(self, mutate):
        cert = self._cert(mutate)
        assert cert["ready"] is False
        assert any("suite" in check for check in cert["failing_checks"])

    def test_missing_commit_binding_fails(self):
        self._assert_suite_fail(lambda r: r["focused"].pop("commit"))

    def test_missing_tree_binding_fails(self):
        self._assert_suite_fail(lambda r: r["full"].pop("tree_sha256"))

    def test_missing_dirty_binding_fails(self):
        self._assert_suite_fail(lambda r: r["focused"].pop("dirty"))

    def test_head_mismatched_commit_fails(self):
        # Substitution-negative: a DIFFERENT valid 40-hex commit is exactly
        # the stale-receipt-ingestion signature.
        def mutate(receipts):
            receipts["full"]["commit"] = "d4" * 20

        self._assert_suite_fail(mutate)

    def test_head_mismatched_tree_fails(self):
        def mutate(receipts):
            receipts["focused"]["tree_sha256"] = "e5" * 32

        self._assert_suite_fail(mutate)

    def test_dirty_receipt_fails_bool_exact(self):
        self._assert_suite_fail(
            lambda r: r["focused"].__setitem__("dirty", True)
        )
        # Stringly-typed "false" is not clean either.
        self._assert_suite_fail(
            lambda r: r["full"].__setitem__("dirty", "false")
        )

    def test_nonzero_failures_or_errors_fail(self):
        self._assert_suite_fail(
            lambda r: r["focused"]["counts"].__setitem__("failures", 2)
        )
        self._assert_suite_fail(
            lambda r: r["full"]["counts"].__setitem__("errors", 1)
        )

    def test_bool_zero_failures_is_rejected_not_laundered(self):
        # False == 0 in Python; counts must be exact ints (seed catalog).
        self._assert_suite_fail(
            lambda r: r["focused"]["counts"].__setitem__("failures", False)
        )


# ---------------------------------------------------------------------------
# Amended R-080: --records-out/out_dir is the PARENT; doubled segment refused
# ---------------------------------------------------------------------------


class TestR080ParentDirRefusal:
    def test_out_dir_ending_in_records_is_refused(self, tmp_path):
        # Probe-confirmed current defect: out_dir=".../records" silently
        # writes ".../records/records/<cell>.jsonl" (the P1-6 argv class).
        # Fail-loud refusal adjudicated over silent dedup.
        with pytest.raises(
            (schema.ColmAimsError, ValueError)
        ) as excinfo:
            _phase4_records().export_records(
                _scored_items(),
                "khard__format_specific",
                tmp_path / "records",
            )
        message = str(excinfo.value)
        assert "records" in message
        assert "parent" in message.lower() or "doubled" in message.lower()

    def test_out_dir_merely_containing_records_is_fine(self, tmp_path):
        # Nearest-true control: only a FINAL component exactly "records"
        # is the doubled-segment class.
        out = _phase4_records().export_records(
            _scored_items(),
            "khard__format_specific",
            tmp_path / "records_v2",
        )
        assert out == (
            tmp_path / "records_v2" / "records"
            / "khard__format_specific.jsonl"
        )
        assert out.is_file()


# ---------------------------------------------------------------------------
# R-072 rev3: dual-hash entries + 1-based pointers over the same scope
# ---------------------------------------------------------------------------


class TestR072Rev3Manifest:
    # Source: qa012_inventory_2026-08-22_rev3.json (repo root; landed
    # 2026-08-22 21:53 — a real artifact, so its field names govern over
    # the shapes this round initially sketched).

    def test_rev3_exists_and_corrects_the_rev2_defects(self):
        manifest = _load_json(QA012_REV3_MANIFEST_PATH)
        assert manifest["revision"] == 3
        assert manifest["verdict"] == "HITS_PRESENT_NOT_VACUOUS"
        assert manifest["total_format_qa_hits"] == 4556
        assert manifest["files_scanned"] == 67
        assert manifest["parse_failures"] == []
        # Correction 1 (0-based pointers): 1-based declared, and no
        # pointer anywhere says "line 0:".
        assert manifest["conventions"]["jsonl_line_numbers"] == "1-based"
        pointers = json.dumps(
            [entry["format_qa_hits"] for entry in manifest["entries"]]
        )
        assert "line 0:" not in pointers
        # Correction 2 (sha256-only entries): dual hash on EVERY entry.
        for entry in manifest["entries"]:
            assert schema.is_sha256_hex(entry["sha256"]), entry["path"]
            assert (
                isinstance(entry["dropbox_content_hash"], str)
                and len(entry["dropbox_content_hash"]) == 64
            ), entry["path"]
        # Per-entry hit pointers recount to the manifest total.
        assert (
            sum(len(entry["format_qa_hits"]) for entry in manifest["entries"])
            == 4556
        )

    def test_rev3_supersession_chain_names_rev1_and_rev2(self):
        manifest = _load_json(QA012_REV3_MANIFEST_PATH)
        chain = manifest["supersession_chain"]
        shas = [link["sha256"] for link in chain]
        assert QA012_REV1_SHA in shas
        assert QA012_REV2_SHA in shas
        assert all(link.get("defect") for link in chain)


# ---------------------------------------------------------------------------
# R-081: single-use launcher
# ---------------------------------------------------------------------------


LAUNCHER_ENV_PINS = {
    "PYTHONHASHSEED": "0",
    "OMP_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
}
# MIGRATED (R-082/F-1, 2026-08-22): staged inputs live OUTSIDE the repo —
# the certificate command and staged_inputs component now carry ABSOLUTE
# tmp_path locations (the old repo-relative `staged/...` forms are the
# refusal class the launcher's new pre-ledger gate exists to catch).


def _staged_outside_dir(tmp_path):
    staged = tmp_path / "staged_outside"
    staged.mkdir(parents=True, exist_ok=True)
    for label, filename in _PHASE4_STAGED_FILENAMES.items():
        path = staged / filename
        if not path.exists():
            path.write_bytes(
                json.dumps({label: "staged-out-of-repo"}).encode("utf-8")
                + b"\n"
            )
    return staged


def _cert_command(
    staged_dir,
    manifest_path,
    snapshot_dirs,
    eligibility_path=ELIGIBILITY_PATH,
):
    """The certificate-recorded producer command: outputs repo-relative
    (remapped into quarantine by the launcher), staged inputs ABSOLUTE and
    out-of-repo (R-082)."""
    staged_digests = {
        label: sha256_file(staged_dir / filename)
        for label, filename in _PHASE4_STAGED_FILENAMES.items()
    }
    return [
        str(Path(sys.executable).resolve()),
        "scripts/stopdff_fair_qa_retest.py",
    ] + _staging_command_args(
        staged_dir, staged_digests, eligibility_path
    ) + _experiment_command_args() + _snapshot_output_command_args(
        manifest_path, snapshot_dirs
    )


def _staged_component_entries(staged_dir):
    return [
        {
            "path": str(staged_dir / filename),
            "label": label,
            "expected_sha256": sha256_file(staged_dir / filename),
            "observed_sha256": sha256_file(staged_dir / filename),
        }
        for label, filename in _PHASE4_STAGED_FILENAMES.items()
    ]


def _launcher_snapshots(tmp_path):
    """Synthetic role snapshots + a matching manifest file (frozen shape,
    tmp-file digests)."""
    manifest = _load_json(MODEL_MANIFEST_PATH)
    snapshot_dirs = {}
    for role in ("primary_scorer", "disjoint_selector"):
        snap = tmp_path / f"launcher_snap_{role}"
        (snap / "1_Pooling").mkdir(parents=True)
        blobs = {
            "config.json": json.dumps({"role": role}).encode() + b"\n",
            "1_Pooling/config.json": b'{"pooling": "mean"}\n',
        }
        for rel, blob in blobs.items():
            (snap / rel).write_bytes(blob)
        manifest["roles"][role]["files"] = {
            rel: {"sha256": sha256_bytes(blob), "size": len(blob)}
            for rel, blob in blobs.items()
        }
        manifest["roles"][role]["file_count"] = len(blobs)
        snapshot_dirs[role] = snap
    manifest_path = tmp_path / "launcher_snapshot_manifests.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path, snapshot_dirs


class _LaunchRecorder:
    """Injectable launch(argv, env) -> exit_code with full capture; also
    proves the ledger is written BEFORE launch and drops an unrelated marker
    so the private promotion tree can prove producer junk is excluded."""

    def __init__(self, config, digest, exit_code=0):
        self.config = config
        self.digest = digest
        self.exit_code = exit_code
        self.calls = []

    def __call__(self, argv, env):
        ledger = Path(self.config["ledger_path"])
        assert ledger.is_file(), "ledger must be written BEFORE launch"
        assert self.digest in ledger.read_text("utf-8")
        quarantine = Path(self.config["quarantine_dir"])
        assert quarantine.is_dir(), "quarantine must exist at launch time"
        (quarantine / "marker.txt").write_text("ran", encoding="utf-8")
        if self.exit_code == 0:
            eligibility = json.loads(
                Path(argv[argv.index("--eligibility") + 1]).read_text("utf-8")
            )
            rows = [
                {
                    "item_key": item_key,
                    "trajectory_horizon": eligibility["horizon_map"][item_key],
                    "mc_event_status": "FINITE_STOP",
                    "mc_stop_step": 0,
                    "mc_terminal_imputation": "NONE",
                    "ref_event_status": "FINITE_STOP",
                    "ref_stop_step": 0,
                    "ref_terminal_imputation": "NONE",
                }
                for item_key in eligibility["eligible_keys"]
            ]
            record_bytes = b"".join(
                (
                    json.dumps(row, sort_keys=True, separators=(",", ":"))
                    + "\n"
                ).encode("utf-8")
                for row in rows
            )
            record_sha256 = sha256_bytes(record_bytes)
            records_root = (
                Path(argv[argv.index("--records-out") + 1]) / "records"
            )
            records_root.mkdir()
            exported_records = {}
            for cell_id in schema.CELL_IDS:
                (records_root / f"{cell_id}.jsonl").write_bytes(record_bytes)
                reference, calibration = cell_id.rsplit("__", 1)
                historical_calibration = (
                    "performat"
                    if calibration == "format_specific"
                    else calibration
                )
                exported_records[cell_id] = {
                    "path": f"records/{cell_id}.jsonl",
                    "sha256": record_sha256,
                    "n_items": schema.EXPECTED_COMPLETE_PAIRS,
                    "historical_cell": (
                        f"{reference}+{historical_calibration}"
                    ),
                    "policy": "dp",
                }
            Path(argv[argv.index("--out") + 1]).write_text(
                json.dumps(
                    {
                        "metadata": {
                            "phase4": {
                                "certificate_digest": self.digest,
                                "exported_records": exported_records,
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
        self.calls.append((list(argv), dict(env)))
        return self.exit_code


def _launcher_fixture(
    tmp_path,
    monkeypatch,
    *,
    mutate_components=None,
    force_ready_after_mutation=False,
):
    """Build a READY certificate + launcher config with injectable fakes.

    Returns (launcher_module, config, digest). MODAL_HOST_* ambient
    overrides are cleared (the ambient-refusal test sets them back).
    """
    monkeypatch.delenv("MODAL_HOST_GIT_STATUS", raising=False)
    monkeypatch.delenv("MODAL_HOST_GIT_COMMIT", raising=False)
    phase4 = _phase4()
    components = _good_components()
    components["repo"] = {
        "commit": FAKE_COMMIT,
        "tree_sha256": FAKE_TREE,
        "dirty": False,
        "untracked_disclosure": [],
        "root_realpath": str(REPO_ROOT.resolve()),
    }
    components["suite_receipts"] = {
        "focused": _r070_receipt(
            "pytest --suite focused",
            commit=FAKE_COMMIT,
            tree_sha256=FAKE_TREE,
        ),
        "full": _r070_receipt(
            "pytest --suite full",
            commit=FAKE_COMMIT,
            tree_sha256=FAKE_TREE,
        ),
    }
    # R-082: absolute out-of-repo staged paths in BOTH launcher-checked
    # sources (the recorded command and the staged_inputs component).
    staged_dir = _staged_outside_dir(tmp_path)
    staged_digests = {
        label: sha256_file(staged_dir / filename)
        for label, filename in _PHASE4_STAGED_FILENAMES.items()
    }
    monkeypatch.setattr(
        phase4, "R082_STAGED_INPUT_SHA256", dict(staged_digests)
    )
    monkeypatch.setattr(
        phase4,
        "CALIBRATION_TRAIN_SHA256",
        staged_digests["calibration_train"],
    )
    synthetic_eligibility = _load_json(ELIGIBILITY_PATH)
    synthetic_eligibility["derived_from"]["test_dataset_sha256"] = (
        staged_digests["eval_split"]
    )
    eligibility_path = tmp_path / "pairing_eligibility_v2.json"
    eligibility_path.write_text(
        json.dumps(synthetic_eligibility), encoding="utf-8"
    )
    monkeypatch.setattr(
        phase4,
        "ELIGIBILITY_ARTIFACT_RELPATH",
        str(eligibility_path.resolve()),
    )
    monkeypatch.setattr(
        phase4,
        "ELIGIBILITY_ARTIFACT_SHA256",
        sha256_file(eligibility_path),
    )
    monkeypatch.setattr(
        phase4,
        "ELIGIBILITY_TEST_DATASET_SHA256",
        staged_digests["eval_split"],
    )
    components["eligibility"].update(
        {
            "artifact_path": str(eligibility_path.resolve()),
            "artifact_sha256": sha256_file(eligibility_path),
            "test_dataset_sha256": staged_digests["eval_split"],
        }
    )
    manifest_path, snapshot_dirs = _launcher_snapshots(tmp_path)
    monkeypatch.setattr(
        phase4,
        "SNAPSHOT_MANIFEST_RELPATH",
        str(manifest_path.resolve()),
    )
    monkeypatch.setattr(
        phase4,
        "SNAPSHOT_MANIFEST_SHA256",
        sha256_file(manifest_path),
    )
    components["snapshots"]["artifact_path"] = str(manifest_path.resolve())
    components["snapshots"]["artifact_sha256"] = sha256_file(manifest_path)
    fixture_lock = b"fixture-package==1\n"
    fixture_interpreter = Path(sys.executable).resolve()
    fixture_quarantine = tmp_path / "quarantine"
    fixture_promote = tmp_path / "promoted"
    fixture_ledger = tmp_path / "exception_ledger.json"
    components["environment"].update(
        {
            "command": _cert_command(
                staged_dir,
                manifest_path,
                snapshot_dirs,
                eligibility_path,
            ),
            "interpreter_realpath": str(fixture_interpreter),
            "os": "FixtureOS 1 ()",
            "arch": "fixture-arch",
            "environment_lock_sha256": sha256_bytes(fixture_lock),
            "quarantine_dir": str(fixture_quarantine),
            "promote_to": str(fixture_promote),
            "exception_ledger_path": str(fixture_ledger),
        }
    )
    for name, receipt in components["suite_receipts"].items():
        receipt["interpreter_realpath"] = str(fixture_interpreter)
        receipt["environment_lock_sha256"] = sha256_bytes(fixture_lock)
        selection = (
            phase4.FOCUSED_SUITE_SELECTION
            if name == "focused"
            else phase4.FULL_SUITE_SELECTION
        )
        receipt["command"] = [
            str(fixture_interpreter),
            "-m",
            "pytest",
            *selection,
            "-q",
            "-p",
            "no:cacheprovider",
            f"--junitxml={tmp_path / (name + '_suite.xml')}",
        ]
    components["parity"]["anchor_sha256"] = sha256_file(
        PARITY_ANCHOR_PATH
    )
    components["staged_inputs"] = _staged_component_entries(staged_dir)
    if mutate_components is not None:
        mutate_components(components)
    cert = phase4.assemble_certificate(components)
    if force_ready_after_mutation:
        # Defense-in-depth launcher tests deliberately hand-craft a
        # malicious ready:true certificate that the repaired assembler
        # itself would now reject. The launcher must still refuse it.
        cert["ready"] = True
        cert["failing_checks"] = []
    cert_bytes = (json.dumps(cert, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    cert_path = tmp_path / "pre_run_ready_certificate.json"
    cert_path.write_bytes(cert_bytes)
    digest = sha256_bytes(cert_bytes)
    config = {
        "certificate_path": cert_path,
        "activation_digest": digest,
        "quarantine_dir": fixture_quarantine,
        "promote_to": fixture_promote,
        "ledger_path": fixture_ledger,
        "snapshot_manifest_path": manifest_path,
        "snapshot_dirs": snapshot_dirs,
        "anchor_path": PARITY_ANCHOR_PATH,
    }
    launcher = _phase4_launcher()
    monkeypatch.setattr(
        launcher,
        "_default_resolve_executable",
        lambda _token: fixture_interpreter,
    )
    monkeypatch.setattr(
        launcher,
        "_default_host_identity",
        lambda: {"os": "FixtureOS 1 ()", "arch": "fixture-arch"},
    )
    monkeypatch.setattr(
        launcher,
        "_default_probe_environment_lock",
        lambda _interpreter: fixture_lock,
    )
    # These legacy launcher lifecycle tests use a marker-only launch double.
    # Receipt payload semantics are exercised by the focused launcher tests;
    # keep this fixture scoped to ordering, single-use, and promotion.
    monkeypatch.setattr(
        launcher,
        "_write_launch_receipt",
        lambda quarantine, **_kwargs: (
            Path(quarantine) / launcher.LAUNCH_RECEIPT_NAME
        ).write_text("{}\n", encoding="utf-8"),
    )
    return launcher, config, digest


def _pass_compare(_quarantine):
    return {"verdict": "PASS", "checked": 194, "failures": []}


def _fail_compare(_quarantine):
    return {
        "verdict": "FAIL",
        "checked": 194,
        "failures": [
            {
                "cell": "khard+shared",
                "policy": "dp",
                "field": "signed_mean",
                "expected": 0.1,
                "observed": 0.2,
            }
        ],
    }


class TestR081LauncherRefusals:
    """Every pre-launch refusal class: typed LaunchRefusal naming the class,
    and launch NEVER invoked."""

    def _refusal(
        self,
        tmp_path,
        monkeypatch,
        *,
        token,
        config_mutate=None,
        mutate_components=None,
        run_git_kwargs=None,
        pre=None,
        expect_ledger_absent=True,
        force_ready_after_mutation=False,
    ):
        launcher, config, digest = _launcher_fixture(
            tmp_path,
            monkeypatch,
            mutate_components=mutate_components,
            force_ready_after_mutation=force_ready_after_mutation,
        )
        if config_mutate is not None:
            config_mutate(config)
        if pre is not None:
            pre(config)
        launch = _LaunchRecorder(config, config["activation_digest"])
        run_git = _fake_git_runner(**(run_git_kwargs or {}))
        with pytest.raises(launcher.LaunchRefusal) as excinfo:
            launcher.validate_and_launch(
                config,
                run_git=run_git,
                launch=launch,
                compare=_pass_compare,
            )
        assert token in str(excinfo.value), (
            f"refusal must name its class; wanted {token!r} in"
            f" {excinfo.value}"
        )
        assert launch.calls == [], "launch must NEVER fire on a refusal"
        if expect_ledger_absent:
            # F-2: refusals never consume the single-use exception.
            assert not Path(config["ledger_path"]).exists(), (
                "a refusal must not create the exception ledger"
            )
        return excinfo

    def test_1_certificate_digest_mismatch_refuses(
        self, tmp_path, monkeypatch
    ):
        self._refusal(
            tmp_path,
            monkeypatch,
            token="digest",
            config_mutate=lambda c: c.__setitem__(
                "activation_digest", "0" * 64
            ),
        )

    def test_1b_digest_is_checked_before_ready(self, tmp_path, monkeypatch):
        # Ordering pin: a cert that is BOTH not-ready and digest-mismatched
        # must refuse on the DIGEST (bytes first, semantics second).
        excinfo = self._refusal(
            tmp_path,
            monkeypatch,
            token="digest",
            mutate_components=lambda comps: comps["repo"].__setitem__(
                "dirty", True
            ),
            config_mutate=lambda c: c.__setitem__(
                "activation_digest", "0" * 64
            ),
        )
        assert "ready" not in str(excinfo.value).lower()

    def test_2_not_ready_certificate_refuses(self, tmp_path, monkeypatch):
        # ready:false cert, CORRECT digest (fixture recomputes it).
        self._refusal(
            tmp_path,
            monkeypatch,
            token="ready",
            mutate_components=lambda comps: comps["repo"].__setitem__(
                "dirty", True
            ),
        )

    def test_2b_ready_laundered_as_one_refuses(self, tmp_path, monkeypatch):
        # Bool-safe: tamper the WRITTEN cert to ready=1 and re-point the
        # activation digest at the tampered bytes — class (2) must fire.
        def tamper(config):
            cert = json.loads(
                Path(config["certificate_path"]).read_text("utf-8")
            )
            cert["ready"] = 1
            blob = (json.dumps(cert, indent=2, sort_keys=True) + "\n").encode(
                "utf-8"
            )
            Path(config["certificate_path"]).write_bytes(blob)
            config["activation_digest"] = sha256_bytes(blob)

        self._refusal(tmp_path, monkeypatch, token="ready", pre=tamper)

    def test_3_live_commit_mismatch_refuses(self, tmp_path, monkeypatch):
        self._refusal(
            tmp_path,
            monkeypatch,
            token="commit",
            run_git_kwargs={"commit": "d4" * 20},
        )

    def test_4_live_tree_mismatch_refuses(self, tmp_path, monkeypatch):
        self._refusal(
            tmp_path,
            monkeypatch,
            token="tree",
            run_git_kwargs={"tree": "c3" * 32},
        )

    def test_5_live_tracked_dirty_refuses(self, tmp_path, monkeypatch):
        self._refusal(
            tmp_path,
            monkeypatch,
            token="dirty",
            run_git_kwargs={"status": " M scripts/stopdff_fair_qa_retest.py\n"},
        )

    def test_6_ambient_modal_host_overrides_refuse(
        self, tmp_path, monkeypatch
    ):
        # The laundering trap: an ambient EMPTY status would fake-clean the
        # committed-writer guard — PRESENCE at all refuses, value ignored.
        launcher, config, _ = _launcher_fixture(tmp_path, monkeypatch)
        for var in ("MODAL_HOST_GIT_STATUS", "MODAL_HOST_GIT_COMMIT"):
            monkeypatch.setenv(var, "")
            launch = _LaunchRecorder(config, config["activation_digest"])
            with pytest.raises(launcher.LaunchRefusal) as excinfo:
                launcher.validate_and_launch(
                    config,
                    run_git=_fake_git_runner(),
                    launch=launch,
                    compare=_pass_compare,
                )
            assert "MODAL_HOST" in str(excinfo.value)
            assert launch.calls == []
            monkeypatch.delenv(var)

    def test_7_snapshot_mismatch_refuses(self, tmp_path, monkeypatch):
        def tamper(config):
            victim = config["snapshot_dirs"]["primary_scorer"] / "config.json"
            victim.write_bytes(victim.read_bytes() + b"tamper\n")

        self._refusal(tmp_path, monkeypatch, token="snapshot", pre=tamper)

    def test_8_preexisting_ledger_refuses(self, tmp_path, monkeypatch):
        def pre(config):
            Path(config["ledger_path"]).write_text(
                '{"consumed": true}', encoding="utf-8"
            )

        self._refusal(
            tmp_path,
            monkeypatch,
            token="ledger",
            pre=pre,
            expect_ledger_absent=False,
        )
        # The pre-existing ledger is untouched (create-once, never rewrite)
        # and the refusal leaves no quarantine behind (F-2 rmdir).
        assert (
            (tmp_path / "exception_ledger.json").read_text("utf-8")
            == '{"consumed": true}'
        )
        assert not (tmp_path / "quarantine").exists()

    def test_9_preexisting_quarantine_refuses(self, tmp_path, monkeypatch):
        def pre(config):
            Path(config["quarantine_dir"]).mkdir(parents=True)

        # F-2 strengthening: workspace refusals fire PRE-LEDGER.
        self._refusal(tmp_path, monkeypatch, token="quarantine", pre=pre)

    def test_9b_preexisting_promote_destination_refuses(
        self, tmp_path, monkeypatch
    ):
        def pre(config):
            Path(config["promote_to"]).mkdir(parents=True)

        self._refusal(tmp_path, monkeypatch, token="promote", pre=pre)
        # Promote staleness is checked before the quarantine mkdir: the
        # refusal materializes NO workspace at all.
        assert not (tmp_path / "quarantine").exists()

    def test_9c_missing_promote_parent_refuses_pre_ledger(
        self, tmp_path, monkeypatch
    ):
        # F-2 pin: a promote destination whose PARENT is absent refuses
        # up front (a PASS must always be able to promote), pre-ledger,
        # leaving no workspace.
        def mutate(config):
            config["promote_to"] = tmp_path / "no_such_parent" / "promoted"

        self._refusal(
            tmp_path, monkeypatch, token="promote", config_mutate=mutate
        )
        assert not (tmp_path / "quarantine").exists()

    # -- R-082/F-1 staged-location refusals (regression pins; the exact
    # rejected-certificate-8731ad00 class) --------------------------------

    def test_r082_relative_in_repo_staged_command_path_refuses(
        self, tmp_path, monkeypatch
    ):
        # A repo-RELATIVE --calibration value resolves against the child's
        # cwd (the repo root) => inside the tree => pre-ledger refusal.
        def mutate(components):
            command = list(components["environment"]["command"])
            command[command.index("--calibration") + 1] = (
                "data/processed/val_dataset.json"
            )
            components["environment"]["command"] = command

        excinfo = self._refusal(
            tmp_path,
            monkeypatch,
            token="R-082",
            mutate_components=mutate,
            force_ready_after_mutation=True,
        )
        assert "val_dataset.json" in str(excinfo.value)

    def test_r082_absolute_in_repo_staged_command_path_refuses(
        self, tmp_path, monkeypatch
    ):
        # Sibling form: an ABSOLUTE path under the repo root is the same
        # class (location, not spelling).
        in_repo = str(REPO_ROOT / "staged" / "calibration_train.json")

        def mutate(components):
            command = list(components["environment"]["command"])
            command[command.index("--calibration") + 1] = in_repo
            components["environment"]["command"] = command

        self._refusal(
            tmp_path,
            monkeypatch,
            token="R-082",
            mutate_components=mutate,
            force_ready_after_mutation=True,
        )

    def test_r082_component_only_in_repo_staged_path_refuses(
        self, tmp_path, monkeypatch
    ):
        # Command clean; the certificate's staged_inputs COMPONENT alone
        # carries an in-repo path (the second checked source).
        def mutate(components):
            components["staged_inputs"][1]["path"] = (
                "data/processed/test_dataset.json"
            )

        self._refusal(
            tmp_path,
            monkeypatch,
            token="R-082",
            mutate_components=mutate,
            force_ready_after_mutation=True,
        )

    def test_r082_staged_input_flag_in_repo_path_refuses(
        self, tmp_path, monkeypatch
    ):
        # The --staged-input LABEL=PATH:SHA argv form is the third checked
        # source; an in-repo PATH inside the triple refuses.
        def mutate(components):
            components["environment"]["command"] = list(
                components["environment"]["command"]
            ) + [
                "--staged-input",
                f"calibration_train=staged/calibration_train.json:"
                f"{CALIB_TRAIN_SHA}",
            ]

        self._refusal(
            tmp_path,
            monkeypatch,
            token="R-082",
            mutate_components=mutate,
            force_ready_after_mutation=True,
        )


class TestR081LauncherRun:
    def test_success_path_launch_env_argv_ledger_promote(
        self, tmp_path, monkeypatch
    ):
        launcher, config, digest = _launcher_fixture(tmp_path, monkeypatch)
        launch = _LaunchRecorder(config, digest, exit_code=0)
        launcher.validate_and_launch(
            config,
            run_git=_fake_git_runner(),
            launch=launch,
            compare=_pass_compare,
        )
        assert len(launch.calls) == 1, "launch fires EXACTLY once"
        argv, env = launch.calls[0]
        # Env pins + no ambient-override keys survive into the child.
        for key, value in LAUNCHER_ENV_PINS.items():
            assert env.get(key) == value, key
        assert not any(k.startswith("MODAL_HOST") for k in env)
        # Argv is composed FROM the certificate command, then every mutable
        # input path is rebound to its authenticated private quarantine copy.
        quarantine = Path(config["quarantine_dir"])
        staged_dir = _staged_outside_dir(tmp_path)
        expected_argv = _cert_command(
            staged_dir,
            config["snapshot_manifest_path"],
            config["snapshot_dirs"],
            tmp_path / "pairing_eligibility_v2.json",
        )
        expected_argv[expected_argv.index("--out") + 1] = str(
            quarantine / "stopdff_fair_qa_regenerated.json"
        )
        expected_argv[expected_argv.index("--records-out") + 1] = str(
            quarantine
        )
        expected_argv.extend(["--certificate-digest", digest])
        capture_root = quarantine / launcher.CAPTURED_INPUTS_DIRNAME
        expected_argv = launcher._rewrite_argv_for_captured_inputs(
            expected_argv,
            {
                "data_dir": capture_root / "data",
                "staged": {
                    "calibration_train": (
                        capture_root / "calibration_train.json"
                    ),
                    **{
                        label: (
                            capture_root
                            / "data"
                            / launcher.phase4.R082_DATA_FILENAMES[label]
                        )
                        for label in launcher.phase4.R082_DATA_FILENAMES
                    },
                },
                "manifest": capture_root / "model_snapshot_manifest.json",
                "snapshots": {
                    role: capture_root / "models" / role
                    for role in launcher.phase4.SNAPSHOT_ROLES
                },
                "eligibility": capture_root / "pairing_eligibility.json",
            },
        )
        assert argv == expected_argv
        # Ledger consumed (content binds the digest; written pre-launch —
        # asserted inside the recorder).
        assert digest in Path(config["ledger_path"]).read_text("utf-8")
        # Atomic private promote: quarantine is gone, while only the closed
        # accepted tree (not arbitrary producer files) reaches promote_to.
        promote_to = Path(config["promote_to"])
        assert promote_to.is_dir()
        assert not (promote_to / "marker.txt").exists()
        assert (
            promote_to / "stopdff_fair_qa_regenerated.json"
        ).is_file()
        assert {
            path.name for path in (promote_to / "records").iterdir()
        } == {f"{cell_id}.jsonl" for cell_id in schema.CELL_IDS}
        assert (
            promote_to / launcher.CAPTURED_INPUTS_DIRNAME
        ).is_dir()
        assert (promote_to / launcher.LAUNCH_RECEIPT_NAME).is_file()
        assert (promote_to / launcher.ACCEPTANCE_MARKER_NAME).is_file()
        assert not quarantine.exists()

    def test_untracked_only_status_is_not_a_refusal(
        self, tmp_path, monkeypatch
    ):
        # Nearest-true control (tracked-clean + untracked-disclosure, the
        # signed PRE convention): "?? ..." porcelain lines alone must not
        # block the launch.
        disclosed = [
            "phase4_pre_receipts/focused.xml",
            "untracked_note.md",
        ]
        launcher, config, digest = _launcher_fixture(
            tmp_path,
            monkeypatch,
            mutate_components=lambda components: components["repo"].update(
                {"untracked_disclosure": disclosed}
            ),
        )
        launch = _LaunchRecorder(config, digest, exit_code=0)
        launcher.validate_and_launch(
            config,
            run_git=_fake_git_runner(
                untracked_status=(
                    "?? phase4_pre_receipts/focused.xml\x00"
                    "?? untracked_note.md\x00"
                )
            ),
            launch=launch,
            compare=_pass_compare,
        )
        assert len(launch.calls) == 1

    def test_single_use_second_call_with_same_config_refuses(
        self, tmp_path, monkeypatch
    ):
        launcher, config, digest = _launcher_fixture(tmp_path, monkeypatch)
        launch = _LaunchRecorder(config, digest, exit_code=0)
        launcher.validate_and_launch(
            config,
            run_git=_fake_git_runner(),
            launch=launch,
            compare=_pass_compare,
        )
        with pytest.raises(schema.ColmAimsError):
            launcher.validate_and_launch(
                config,
                run_git=_fake_git_runner(),
                launch=launch,
                compare=_pass_compare,
            )
        assert len(launch.calls) == 1, "at most ONE launch across both calls"

    def test_single_use_fresh_workspace_same_ledger_refuses_on_ledger(
        self, tmp_path, monkeypatch
    ):
        # Isolate the ledger as the single-use mechanism while retaining the
        # exact certificate-owned workspace strings.
        launcher, config, digest = _launcher_fixture(tmp_path, monkeypatch)
        launch = _LaunchRecorder(config, digest, exit_code=0)
        launcher.validate_and_launch(
            config,
            run_git=_fake_git_runner(),
            launch=launch,
            compare=_pass_compare,
        )
        Path(config["promote_to"]).rename(tmp_path / "first_promoted")
        launch2 = _LaunchRecorder(config, digest, exit_code=0)
        with pytest.raises(launcher.LaunchRefusal) as excinfo:
            launcher.validate_and_launch(
                config,
                run_git=_fake_git_runner(),
                launch=launch2,
                compare=_pass_compare,
            )
        assert "ledger" in str(excinfo.value).lower()
        assert launch2.calls == []
        # F-2: the consumed-ledger refusal is side-effect-free — the
        # quarantine it just created for this attempt is removed again.
        assert not Path(config["quarantine_dir"]).exists()

    def test_nonzero_exit_leaves_quarantine_and_stop_report(
        self, tmp_path, monkeypatch
    ):
        launcher, config, digest = _launcher_fixture(tmp_path, monkeypatch)
        launch = _LaunchRecorder(config, digest, exit_code=3)
        with pytest.raises(launcher.RunFailed):
            launcher.validate_and_launch(
                config,
                run_git=_fake_git_runner(),
                launch=launch,
                compare=_pass_compare,
            )
        quarantine = Path(config["quarantine_dir"])
        assert (quarantine / "marker.txt").is_file(), "quarantine intact"
        assert not Path(config["promote_to"]).exists()
        stop = quarantine / "STOP_REPORT.json"
        assert stop.is_file()
        assert digest in stop.read_text("utf-8")

    def test_comparator_fail_blocks_promotion_with_stop_report(
        self, tmp_path, monkeypatch
    ):
        launcher, config, digest = _launcher_fixture(tmp_path, monkeypatch)
        launch = _LaunchRecorder(config, digest, exit_code=0)
        with pytest.raises(launcher.RunFailed):
            launcher.validate_and_launch(
                config,
                run_git=_fake_git_runner(),
                launch=launch,
                compare=_fail_compare,
            )
        assert len(launch.calls) == 1
        quarantine = Path(config["quarantine_dir"])
        assert (quarantine / "marker.txt").is_file()
        assert not Path(config["promote_to"]).exists()
        assert (quarantine / "STOP_REPORT.json").is_file()

    # -- F-3 crash pins: the ledger is consumed, so the triage artifact
    # must exist on the messiest failures -------------------------------

    def test_launch_crash_writes_stop_report_and_keeps_ledger(
        self, tmp_path, monkeypatch
    ):
        launcher, config, digest = _launcher_fixture(tmp_path, monkeypatch)

        def crashing_launch(argv, env):
            raise RuntimeError("child never started")

        with pytest.raises(launcher.RunFailed):
            launcher.validate_and_launch(
                config,
                run_git=_fake_git_runner(),
                launch=crashing_launch,
                compare=_pass_compare,
            )
        quarantine = Path(config["quarantine_dir"])
        assert quarantine.is_dir(), "quarantine intact after a crash"
        report = json.loads(
            (quarantine / "STOP_REPORT.json").read_text("utf-8")
        )
        assert report["reason"] == "launch_crash"
        assert report["activation_digest"] == digest
        # Honest accounting: the exception WAS consumed (ledger present).
        assert digest in Path(config["ledger_path"]).read_text("utf-8")
        assert not Path(config["promote_to"]).exists()

    def test_comparator_crash_writes_stop_report(
        self, tmp_path, monkeypatch
    ):
        launcher, config, digest = _launcher_fixture(tmp_path, monkeypatch)
        launch = _LaunchRecorder(config, digest, exit_code=0)

        def crashing_compare(quarantine_dir):
            raise RuntimeError("comparator exploded")

        with pytest.raises(launcher.RunFailed):
            launcher.validate_and_launch(
                config,
                run_git=_fake_git_runner(),
                launch=launch,
                compare=crashing_compare,
            )
        assert len(launch.calls) == 1
        quarantine = Path(config["quarantine_dir"])
        report = json.loads(
            (quarantine / "STOP_REPORT.json").read_text("utf-8")
        )
        assert report["reason"] == "comparator_crash"
        assert report["activation_digest"] == digest
        assert (quarantine / "marker.txt").is_file(), "quarantine intact"
        assert not Path(config["promote_to"]).exists()

    # -- F-4 pin: workspace paths are exact certificate bindings ----------

    def test_relative_workspace_paths_refuse_as_unsigned_substitutions(
        self, tmp_path, monkeypatch
    ):
        # Relative replacements cannot redirect the one-shot run.
        monkeypatch.chdir(tmp_path)
        launcher, config, digest = _launcher_fixture(tmp_path, monkeypatch)
        config["quarantine_dir"] = Path("quarantine_rel")
        config["promote_to"] = Path("promoted_rel")
        config["ledger_path"] = Path("ledger_rel.json")
        launch = _LaunchRecorder(config, digest, exit_code=0)
        with pytest.raises(launcher.LaunchRefusal, match="exactly match"):
            launcher.validate_and_launch(
                config,
                run_git=_fake_git_runner(),
                launch=launch,
                compare=_pass_compare,
            )
        assert launch.calls == []
        assert not (tmp_path / "ledger_rel.json").exists()
