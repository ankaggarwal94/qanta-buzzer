"""Phase-4 PRE-run gates: eligibility, staged inputs, snapshots, parity, cert.

Spec rules owned here: R-074 (frozen pairing eligibility loader), R-075
(role-keyed model snapshot manifests + snapshot-directory verification),
R-076 (staged-input hash gates), R-077 (materialized parity comparator),
R-079 (PRE_RUN_READY certificate assembly/generation), R-081 (external,
disjoint launch-workspace bindings), and R-082 (certificate-side
external-staging containment and exact coverage binding).
Spec: .correctless/specs/camera-ready-aims-evidence-2.md
("Phase-4 PRE-run repairs"); intent:
phase4_pre_run_reconciliation_2026-08-22.md sections 4-5.

Error taxonomy: loaders and gates raise ``schema.TypedIngressError``
subclasses; snapshot-directory refusals raise a ``schema.ColmAimsError``
subclass; ``compare_parity`` NEVER raises on missing regenerated fields
(failure rows, guarded builder); ``assemble_certificate`` never raises and
never emits a partial pass.
"""
from __future__ import annotations

import errno
import hashlib
import json
import math
import os
import stat
import subprocess
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from . import pairing, schema


class EligibilityArtifactError(schema.TypedIngressError):
    """Frozen pairing-eligibility artifact failed typed validation (R-074)."""


class StagedInputError(schema.TypedIngressError):
    """A staged fit/eval input failed its fail-closed hash gate (R-076)."""


class StagedCoverageError(schema.TypedIngressError):
    """The consumed-input coverage plan could not be resolved: an uncovered
    input, an operator digest contradicting a frozen pin, or an operator
    entry outside the consumed set (F-1, R-076)."""


class SnapshotManifestError(schema.TypedIngressError):
    """Model snapshot manifest bytes failed typed validation (R-075)."""


class SnapshotMismatchError(schema.ColmAimsError):
    """A local snapshot directory deviates from its pinned manifest (R-075)."""


class ParityAnchorError(schema.TypedIngressError):
    """The committed parity anchor is malformed — fail closed, never a
    vacuous comparison over a truncated allowlist (R-077)."""


# ---------------------------------------------------------------------------
# Pinned constants (R-074/R-075/R-079)
# ---------------------------------------------------------------------------

ELIGIBILITY_ARTIFACT_TYPE = "pairing_eligibility"
ELIGIBILITY_ARTIFACT_RELPATH = (
    "reproducibility/colm_aims_2026/frozen/pairing_eligibility_v2.json"
)
ELIGIBILITY_ARTIFACT_SHA256 = (
    "3f5e042e592276f8f0f3810180ce71eecdd0c1bd4dd1d6565b1ffa869e39e3c2"
)
ELIGIBILITY_KEYSET_SHA256 = (
    "d0ebac8f300f936f10298e2186532dfc1efd0fee6f400c1a1d8696cf86dd00f1"
)
ELIGIBILITY_HORIZON_MAP_SHA256 = (
    "b0514b6cbe6dfffad0ce225869d20b306377d5baff1e1aca4b9cc9904a95486d"
)
ELIGIBILITY_TEST_DATASET_SHA256 = (
    "638a4df978b77a12655ea72d56daad7fa70851ae486ddb4365d9b060549e34f1"
)
ELIGIBILITY_KEYS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "derived_from",
        "eligible_count",
        "eligible_keys",
        "excluded",
        "excluded_count",
        "horizon_map",
        "horizon_map_sha256",
        "pairing_population_keyset_sha256",
    }
)
DERIVED_FROM_KEYS = frozenset(
    {
        "derivation",
        "test_dataset_basename",
        "test_dataset_sha256",
        "two_party_pin",
    }
)
EXCLUDED_ENTRY_KEYS = frozenset({"item_key", "reason"})
# R-074: the frozen artifact's cardinalities are spec-pinned (2,249 eligible
# sorted keys; 9 SINGLE_PREFIX_TRAJECTORY exclusions).
EXPECTED_ELIGIBLE_COUNT = schema.EXPECTED_COMPLETE_PAIRS
EXPECTED_EXCLUDED_COUNT = 9
# DECISION: horizon 1 contradicts the SINGLE_PREFIX_TRAJECTORY exclusion rule
# that produced the artifact — eligible horizons start at 2.
MIN_ELIGIBLE_HORIZON = 2
# R-076: two-party frozen archival digest of calibration_train.json.  The
# certificate owns the same pin as the producer so a caller cannot substitute
# a self-consistent calibration path+digest pair.
CALIBRATION_TRAIN_SHA256 = (
    "745bd67597278bd9d24d41c1dea53bf3a7c56cd6334cfc07ea62bccbdcf44259"
)

MANIFEST_ARTIFACT_TYPE = "model_snapshot_manifests"
MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "note",
        "offline_flags_required",
        "roles",
        "tfidf_config",
    }
)
SNAPSHOT_ROLES = frozenset({"primary_scorer", "disjoint_selector"})
SNAPSHOT_MANIFEST_RELPATH = (
    "reproducibility/colm_aims_2026/frozen/model_snapshot_manifests.json"
)
SNAPSHOT_MANIFEST_SHA256 = (
    "49ad59e61025f1ee2e0dad12356cd1251ecb13c89c11016fb9d34358e2bfb23a"
)
SNAPSHOT_COMPONENT_METADATA_KEYS = frozenset(
    {"artifact_path", "artifact_sha256"}
)
EXPECTED_SNAPSHOT_IDENTITIES = {
    "primary_scorer": {
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "hf_revision": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
    },
    "disjoint_selector": {
        "model_name": "sentence-transformers/all-mpnet-base-v2",
        "hf_revision": "e8c3b32edf5434bc2275fc9bab85f82640a19130",
    },
}
ROLE_ENTRY_KEYS = frozenset({"model_name", "hf_revision", "file_count", "files"})
FILE_ENTRY_KEYS = frozenset({"sha256", "size"})
REQUIRED_OFFLINE_FLAGS = ("HF_HUB_OFFLINE=1", "TRANSFORMERS_OFFLINE=1")
TFIDF_CONFIG_KEYS = frozenset({"analyzer", "ngram_range", "fit_corpus"})

STAGED_ENTRY_KEYS = frozenset({"path", "expected_sha256", "label"})

PARITY_ANCHOR_ARTIFACT_TYPE = "parity_anchor"
PARITY_ANCHOR_RELPATH = (
    "reproducibility/colm_aims_2026/frozen/parity_anchor_export_a.json"
)
PARITY_ANCHOR_SHA256 = (
    "2efff65778f0d65676adb924aeb787b44f2ff7888f3cc8792d663151ef973eee"
)
PARITY_SOURCE_EXPORT_A_SHA256 = (
    "59e1c1a74e5fc0cf4f09f8befca87cfc81516684dca2e88dd275c952b28893ff"
)
QA012_MANIFEST_RELPATH = "qa012_inventory_2026-08-22_rev3.json"
QA012_MANIFEST_SHA256 = (
    "bb692446ad07bea63b5fc6799d4c0b6474cc084076c87b2db7c2c2a9b7334303"
)
QA012_MANIFEST_TYPE = "qa012_format_qa_inventory"
QA012_MANIFEST_REVISION = 3
QA012_CONVENTIONS = {
    "content_hash": (
        "Dropbox content hash: sha256 over concatenated per-4MiB-block"
        " sha256 digests, hex"
    ),
    "sha256": "sha256 over the raw file bytes, hex",
    "jsonl_line_numbers": "1-based",
}
IDENTITY_FIELDS = ("n_eval", "n_fit")
# Amended R-077 (F-3): the frozen anchor allowlist cardinalities are pinned —
# a truncated anchor must refuse, never produce a vacuous sub-194 PASS.
PARITY_ANCHOR_CARDINALITIES = (
    ("nonrandom_cells", 8),
    ("policies", 2),
    ("point_fields", 10),
    ("ci_fields", 2),
)
# 8 cells x 2 policies x (10 point + 2 CI) fields + 2 identity fields.
EXPECTED_PARITY_CHECKED = 8 * 2 * (10 + 2) + len(IDENTITY_FIELDS)

CERT_SCHEMA_VERSION = 2
CERT_COMPONENT_KEYS = (
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
)
CONTENT_HASH_RELPATHS = {
    "producer_sha256": "scripts/stopdff_fair_qa_retest.py",
    "verifier_sha256": "reproducibility/colm_aims_2026/verifier.py",
    "spec_sha256": ".correctless/specs/camera-ready-aims-evidence-2.md",
    "schema_py_sha256": "reproducibility/colm_aims_2026/schema.py",
    "pairing_py_sha256": "reproducibility/colm_aims_2026/pairing.py",
    "phase4_py_sha256": "reproducibility/colm_aims_2026/phase4.py",
    "phase4_records_py_sha256": (
        "reproducibility/colm_aims_2026/phase4_records.py"
    ),
    "phase4_launcher_py_sha256": (
        "reproducibility/colm_aims_2026/phase4_launcher.py"
    ),
    "fileio_py_sha256": "scripts/stopdff_v5/fileio.py",
    "locking_py_sha256": "scripts/stopdff_v5/locking.py",
    "orchestration_sha256": "phase4_pre_run_ready_orchestration.py",
}
CONTENT_HASH_KEYS = tuple(CONTENT_HASH_RELPATHS)
CONTENT_HASH_ENTRY_KEYS = frozenset({"artifact_path", "sha256"})
SUITE_RECEIPT_NAMES = ("focused", "full")
# R-070/R-082: every suite receipt must carry the full machine-readable
# binding — including the R-082 head bindings (commit/tree_sha256/dirty) —
# a receipt missing any of these is a failing suite_receipts component.
R070_RECEIPT_FIELDS = (
    "exit_code",
    "command",
    "environment_lock_sha256",
    "workflow_sha256",
    "interpreter_realpath",
    "counts",
    "skip_identities",
    "junit_sha256",
    "transcript_sha256",
    "commit",
    "tree_sha256",
    "dirty",
)
FOCUSED_SUITE_SELECTION = (
    "tests/test_colm_aims_v2_phase4_pre.py",
    "tests/test_colm_aims_v2_schema_raw_bytes.py",
    "tests/test_phase4_build_metadata_staging.py",
    "tests/test_phase4_certificate_external_staging.py",
    "tests/test_phase4_launcher_cli.py",
    "tests/test_stopdff_v5_fileio_windows.py",
    "tests/test_stopdff_v5_windows_control_plane.py",
)
FULL_SUITE_SELECTION = ("tests/",)
CERT_ENVIRONMENT_KEYS = (
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
)
R082_REPO_ROOT_FIELD = "root_realpath"
# The producer argparse surface accepts only these hyphenated names.  Both
# ``--flag value`` and ``--flag=value`` value spellings remain valid.
_R082_COMMAND_PATH_FLAGS = {
    "--data-dir": "data_dir",
    "--calibration": "calibration",
    "--eligibility": "eligibility",
    "--staged-input": "staged_input",
}
_R082_COMMAND_SPLIT_FLAGS = {
    "--fit-split": "fit_split_name",
    "--eval-split": "eval_split_name",
}
_R082_UNSUPPORTED_INPUT_FLAG_ALIASES = {
    "--data_dir": "--data-dir",
    "--staged_input": "--staged-input",
    "--fit_split": "--fit-split",
    "--eval_split": "--eval-split",
}
_R082_KNOWN_INPUT_FLAGS = (
    *_R082_COMMAND_PATH_FLAGS,
    *_R082_COMMAND_SPLIT_FLAGS,
)
R081_LAUNCH_PATH_FIELDS = (
    "quarantine_dir",
    "promote_to",
    "exception_ledger_path",
)
PHASE4_QA_ARMS = (
    "idealized",
    "krandom",
    "khard",
    "kdisjoint",
    "klex",
)
PHASE4_CALIBRATIONS = ("shared", "performat")
PHASE4_PRODUCER_SCRIPT = "scripts/stopdff_fair_qa_retest.py"
PHASE4_THREAD_SETTINGS = {
    "PYTHONHASHSEED": "0",
    "PYTHONNOUSERSITE": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}
_PHASE4_REQUIRED_COMMAND_VALUES = {
    "--reward-schedule": "power_mark",
    "--num-bootstrap": "1000",
    "--n-test": "0",
    "--n-val": "0",
    "--seed": "1",
    "--records-out": "phase4_run_output",
    "--out": "phase4_run_output/stopdff_fair_qa_regenerated.json",
}
_PHASE4_REQUIRED_COMMAND_PATHS = (
    "--snapshot-manifest",
    "--primary-model-path",
    "--disjoint-model-path",
)
_PHASE4_COMMAND_FLAGS = (
    *_PHASE4_REQUIRED_COMMAND_VALUES,
    "--qa-arms",
    "--calibrations",
    *_PHASE4_REQUIRED_COMMAND_PATHS,
)
_PHASE4_UNSUPPORTED_FLAG_ALIASES = {
    "--reward_schedule": "--reward-schedule",
    "--qa_arms": "--qa-arms",
    "--num_bootstrap": "--num-bootstrap",
    "--n_test": "--n-test",
    "--n_val": "--n-val",
    "--records_out": "--records-out",
    "--snapshot_manifest": "--snapshot-manifest",
    "--primary_model_path": "--primary-model-path",
    "--disjoint_model_path": "--disjoint-model-path",
    "--certificate_digest": "--certificate-digest",
}
R082_STAGED_INPUT_LABELS = (
    "calibration_train",
    "eval_split",
    "fit_split",
    "mc_dataset",
    "answer_profiles",
    "build_metadata",
)
R082_STAGED_INPUT_SHA256 = {
    "calibration_train": CALIBRATION_TRAIN_SHA256,
    "eval_split": ELIGIBILITY_TEST_DATASET_SHA256,
    "fit_split": (
        "9b7a131b6c94c446e6b40b95559cb62aeee63f6e6f29ddd1d7ed3fb19cc72c65"
    ),
    "mc_dataset": (
        "3dbebf8e4d690da41a15e3cf467e57fdbe69af420ed831d56b61160af8bf7946"
    ),
    "answer_profiles": (
        "635586393ad36cf7e0726066bc242d97d0f982abd6108e4d8b87a3cf4598fc75"
    ),
    "build_metadata": (
        "70871984390f252c0a06a5a2c9a2d3b4337f10ad48c87583ebec215d5c0c9c6e"
    ),
}
R082_OPERATOR_DIGEST_LABELS = (
    "fit_split",
    "mc_dataset",
    "answer_profiles",
    "build_metadata",
)
R082_DATA_FILENAMES = {
    "eval_split": "test_dataset.json",
    "fit_split": "val_dataset.json",
    "mc_dataset": "mc_dataset.json",
    "answer_profiles": "answer_profiles.json",
    "build_metadata": "build_metadata.json",
}

_MISSING = object()


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _r082_command_bindings(
    command: Any,
) -> tuple[
    dict[str, list[tuple[str, str]]],
    list[dict[str, str]],
    list[str],
]:
    """Parse certificate-owned input bindings without executing the command.

    Both ``--flag VALUE`` and ``--flag=VALUE`` forms are recognized.  A
    malformed recognized flag is itself a failure: silently dropping a path
    that cannot be parsed would turn the certificate check into a bypass.
    """
    options: dict[str, list[tuple[str, str]]] = {
        "data_dir": [],
        "calibration": [],
        "eligibility": [],
        "fit_split_name": [],
        "eval_split_name": [],
    }
    staged: list[dict[str, str]] = []
    failures: list[str] = []
    if not isinstance(command, list) or not command or not all(
        isinstance(part, str) for part in command
    ):
        return options, staged, [
            "exact command must be a non-empty list of strings"
        ]

    # Do not silently accept aliases that argparse itself rejects.  Scan the
    # complete argv (including anything following ``--``) so a certificate
    # cannot be ready for a command that is guaranteed to die in argument
    # parsing after the one-shot activation ledger has been consumed.
    for token in command:
        option_name = token.partition("=")[0]
        canonical = _R082_UNSUPPORTED_INPUT_FLAG_ALIASES.get(option_name)
        if canonical is not None:
            failures.append(
                f"exact command {option_name}: unsupported flag spelling;"
                f" producer argparse requires {canonical}"
            )

    index = 0
    while index < len(command):
        token = command[index]
        if token == "--":
            break

        flag: str | None = None
        value: str | None = None
        known_flags = {
            **_R082_COMMAND_PATH_FLAGS,
            **_R082_COMMAND_SPLIT_FLAGS,
        }
        for known_flag in known_flags:
            if token == known_flag:
                flag = known_flag
                if index + 1 >= len(command):
                    failures.append(
                        f"exact command {known_flag}: missing path value"
                    )
                    break
                following = command[index + 1]
                if not following or following.startswith("--"):
                    failures.append(
                        f"exact command {known_flag}: missing path value"
                    )
                    break
                value = following
                index += 1
                break
            prefix = f"{known_flag}="
            if token.startswith(prefix):
                flag = known_flag
                value = token[len(prefix) :]
                if not value:
                    failures.append(
                        f"exact command {known_flag}: empty path value"
                    )
                break

        if flag is None:
            option_name = token.partition("=")[0]
            if (
                option_name.startswith("--")
                and len(option_name) > 2
                and any(
                    known.startswith(option_name)
                    for known in _R082_KNOWN_INPUT_FLAGS
                )
            ):
                failures.append(
                    f"exact command {option_name}: abbreviated input flags"
                    " are unsupported"
                )
            index += 1
            continue
        if not value:
            index += 1
            continue

        source = f"exact command {flag}"
        kind = known_flags[flag]
        if kind == "staged_input":
            label, has_label, remainder = value.partition("=")
            staged_path, has_digest, digest = remainder.rpartition(":")
            if (
                not has_label
                or not label
                or not has_digest
                or not staged_path
                or not schema.is_sha256_hex(digest)
            ):
                failures.append(
                    f"{source}: value must be LABEL=PATH:SHA256"
                )
            if has_label and has_digest and staged_path:
                staged.append(
                    {
                        "source": source,
                        "label": label,
                        "path": staged_path,
                        "expected_sha256": digest,
                    }
                )
        else:
            options[kind].append((source, value))
        index += 1

    return options, staged, failures


def phase4_command_failures(command: Any) -> list[str]:
    """Validate the explicit canonical producer run shape, never execute it.

    These values eliminate argument-parser refusals and deliberately partial
    runs that would otherwise be discovered only after consuming the
    single-use activation.  Only argparse's real hyphenated option names are
    accepted; every required option supports split and equals value forms.
    """
    if not isinstance(command, list) or not command or not all(
        isinstance(part, str) for part in command
    ):
        return ["exact command must be a non-empty list of strings"]

    failures: list[str] = []
    if not command[0]:
        failures.append("exact command command[0] interpreter is empty")
    if len(command) < 2 or command[1] != PHASE4_PRODUCER_SCRIPT:
        observed_script = command[1] if len(command) > 1 else None
        failures.append(
            "exact command command[1] must be exactly"
            f" {PHASE4_PRODUCER_SCRIPT!r}; found {observed_script!r}"
        )

    values: dict[str, list[str]] = {
        flag: [] for flag in _PHASE4_COMMAND_FLAGS
    }
    unsupported_aliases = {
        **_R082_UNSUPPORTED_INPUT_FLAG_ALIASES,
        **_PHASE4_UNSUPPORTED_FLAG_ALIASES,
    }
    for token in command:
        option_name = token.partition("=")[0]
        canonical = unsupported_aliases.get(option_name)
        if canonical is not None:
            failures.append(
                f"exact command {option_name}: unsupported flag spelling;"
                f" producer argparse requires {canonical}"
            )

    allowed_options = frozenset(
        {
            *_PHASE4_COMMAND_FLAGS,
            *_R082_COMMAND_PATH_FLAGS,
            *_R082_COMMAND_SPLIT_FLAGS,
        }
    )
    index = 2
    while index < len(command):
        token = command[index]
        if token == "--":
            failures.append(
                "exact command '--' end-of-options marker is unsupported;"
                " the producer accepts no positional run arguments"
            )
            index += 1
            continue
        if not token.startswith("--"):
            failures.append(
                f"exact command stray positional token {token!r} at"
                f" argv[{index}]"
            )
            index += 1
            continue
        option_name, separator, joined_value = token.partition("=")
        if option_name in _R082_UNSUPPORTED_INPUT_FLAG_ALIASES or (
            option_name in _PHASE4_UNSUPPORTED_FLAG_ALIASES
        ):
            if not separator and index + 1 < len(command) and not command[
                index + 1
            ].startswith("--"):
                index += 1
            index += 1
            continue
        if option_name == "--certificate-digest":
            if not separator and index + 1 < len(command) and not command[
                index + 1
            ].startswith("--"):
                index += 1
            index += 1
            continue
        if option_name not in allowed_options:
            failures.append(
                f"exact command option {option_name!r} is non-allowlisted"
                " or an argparse abbreviation"
            )
            index += 1
            continue
        if separator:
            if not joined_value:
                failures.append(
                    f"exact command {option_name}: empty value"
                )
            index += 1
            continue
        if index + 1 >= len(command) or not command[index + 1] or command[
            index + 1
        ].startswith("--"):
            failures.append(f"exact command {option_name}: missing value")
            index += 1
            continue
        index += 2

    index = 0
    while index < len(command):
        token = command[index]
        if token == "--certificate-digest" or token.startswith(
            "--certificate-digest="
        ):
            failures.append(
                "exact command must not contain --certificate-digest;"
                " the launcher alone appends the activation digest"
            )
            if token == "--certificate-digest" and index + 1 < len(command):
                index += 1
            index += 1
            continue

        matched_flag: str | None = None
        matched_value: str | None = None
        for flag in _PHASE4_COMMAND_FLAGS:
            if token == flag:
                matched_flag = flag
                if index + 1 >= len(command):
                    failures.append(
                        f"exact command {flag}: missing value"
                    )
                    break
                candidate = command[index + 1]
                if not candidate or candidate.startswith("--"):
                    failures.append(
                        f"exact command {flag}: missing value"
                    )
                    break
                matched_value = candidate
                index += 1
                break
            prefix = f"{flag}="
            if token.startswith(prefix):
                matched_flag = flag
                matched_value = token[len(prefix) :]
                if not matched_value:
                    failures.append(f"exact command {flag}: empty value")
                break
        if matched_flag is not None and matched_value:
            values[matched_flag].append(matched_value)
        index += 1

    unique: dict[str, str] = {}
    for flag in _PHASE4_COMMAND_FLAGS:
        observed = values[flag]
        if len(observed) != 1:
            failures.append(
                f"exact command {flag} must appear exactly once;"
                f" found {len(observed)}"
            )
        else:
            unique[flag] = observed[0]

    for flag, expected in _PHASE4_REQUIRED_COMMAND_VALUES.items():
        observed = unique.get(flag)
        if observed is not None and observed != expected:
            failures.append(
                f"exact command {flag} must equal {expected!r};"
                f" found {observed!r}"
            )

    for flag, expected_values in (
        ("--qa-arms", PHASE4_QA_ARMS),
        ("--calibrations", PHASE4_CALIBRATIONS),
    ):
        observed = unique.get(flag)
        if observed is None:
            continue
        entries = observed.split(",")
        expected = frozenset(expected_values)
        if (
            any(not entry or entry != entry.strip() for entry in entries)
            or len(entries) != len(set(entries))
            or frozenset(entries) != expected
        ):
            failures.append(
                f"exact command {flag} must contain exactly once each of"
                f" {list(expected_values)!r}; found {observed!r}"
            )

    return failures


def _r082_command_consumed_paths(
    command: Any,
) -> tuple[list[tuple[str, str]], list[str]]:
    """Extract every command path consumed by the external-staging gate."""
    options, staged, failures = _r082_command_bindings(command)
    candidates = [
        binding
        for kind in ("data_dir", "calibration")
        for binding in options[kind]
    ]
    candidates.extend(
        (entry["source"], entry["path"]) for entry in staged
    )
    return candidates, failures


def external_staging_failures(
    repo_root: Any, staged_inputs: Any, command: Any
) -> list[str]:
    """Return every R-082 external-staging defect, never raise.

    Relative command paths resolve from the producer subprocess's working
    directory, which is the configured repository root.  Staged-component
    paths must additionally be absolute because the certificate is the
    portable binding of the already-staged inputs.  Resolution follows
    existing symlinks, so an out-of-tree spelling cannot hide an in-tree
    target.
    """
    failures: list[str] = []
    try:
        root = Path(repo_root)
    except (TypeError, ValueError, OSError):
        return [
            f"{R082_REPO_ROOT_FIELD} is missing or is not a valid path"
        ]
    if not root.is_absolute():
        return [f"{R082_REPO_ROOT_FIELD} must be an absolute path"]
    try:
        root = root.resolve()
    except (OSError, RuntimeError, ValueError) as exc:
        return [
            f"{R082_REPO_ROOT_FIELD} could not be resolved"
            f" ({exc.__class__.__name__})"
        ]

    candidates: list[tuple[str, str, bool]] = []
    if not isinstance(staged_inputs, list):
        failures.append("staged_inputs component is not a list")
    else:
        for index, entry in enumerate(staged_inputs):
            source = f"staged_inputs[{index}].path"
            if not isinstance(entry, dict):
                failures.append(f"{source}: entry is not an object")
                continue
            raw_path = entry.get("path")
            if not isinstance(raw_path, str) or not raw_path:
                failures.append(f"{source}: path is missing")
                continue
            candidates.append((source, raw_path, True))

    command_candidates, command_failures = _r082_command_consumed_paths(
        command
    )
    failures.extend(command_failures)
    candidates.extend(
        (source, raw_path, False)
        for source, raw_path in command_candidates
    )

    for source, raw_path, require_absolute in candidates:
        try:
            candidate = Path(raw_path)
        except (TypeError, ValueError, OSError) as exc:
            failures.append(
                f"{source} {raw_path!r} is not a resolvable path"
                f" ({exc.__class__.__name__})"
            )
            continue
        if require_absolute and not candidate.is_absolute():
            failures.append(
                f"{source} {raw_path!r} must be absolute in the certificate"
            )
        if not candidate.is_absolute():
            candidate = root / candidate
        try:
            inside = schema.resolves_inside(candidate, root)
        except (OSError, RuntimeError, ValueError) as exc:
            failures.append(
                f"{source} {raw_path!r} could not be resolved"
                f" ({exc.__class__.__name__})"
            )
            continue
        if inside:
            failures.append(
                f"{source} {raw_path!r} resolves inside the repository root"
            )
    return failures


def launch_path_failures(repo_root: Any, environment: Any) -> list[str]:
    """Return every certificate launch-workspace path defect, never raise.

    The activation ledger is single use, so the certificate must bind a
    workspace topology that the launcher can safely materialize *before*
    activation.  All three paths are absolute and outside the operational
    repository root; quarantine and promotion trees are disjoint; and the
    exception ledger is outside both trees.  Comparisons use resolved paths
    so aliases and existing symlinks cannot defeat containment checks.
    """
    failures: list[str] = []
    try:
        root = Path(repo_root)
    except (OSError, TypeError, ValueError):
        return [
            f"{R082_REPO_ROOT_FIELD} is missing or is not a valid path"
        ]
    if not root.is_absolute():
        return [f"{R082_REPO_ROOT_FIELD} must be an absolute path"]
    try:
        root = root.resolve()
    except (OSError, RuntimeError, ValueError) as exc:
        return [
            f"{R082_REPO_ROOT_FIELD} could not be resolved"
            f" ({exc.__class__.__name__})"
        ]

    if not isinstance(environment, dict):
        return ["environment component is not an object"]

    resolved: dict[str, Path] = {}
    for field in R081_LAUNCH_PATH_FIELDS:
        raw_path = environment.get(field)
        if not isinstance(raw_path, str) or not raw_path:
            failures.append(
                f"environment {field} is missing or is not a non-empty string"
            )
            continue
        try:
            path = Path(raw_path)
        except (OSError, TypeError, ValueError) as exc:
            failures.append(
                f"environment {field} is not a valid path"
                f" ({exc.__class__.__name__})"
            )
            continue
        if not path.is_absolute():
            failures.append(
                f"environment {field} must be an absolute external path"
            )
            continue
        try:
            path = path.resolve()
        except (OSError, RuntimeError, ValueError) as exc:
            failures.append(
                f"environment {field} could not be resolved"
                f" ({exc.__class__.__name__})"
            )
            continue
        if os.path.normcase(str(Path(raw_path))) != os.path.normcase(
            str(path)
        ):
            failures.append(
                f"environment {field} must be recorded as its canonical"
                " resolved path (symlink/junction and dot-segment aliases"
                " are forbidden)"
            )
        resolved[field] = path
        if path == root or root in path.parents:
            failures.append(
                f"environment {field} {str(path)!r} resolves inside the"
                " repository root"
            )

    for index, left_name in enumerate(R081_LAUNCH_PATH_FIELDS):
        left = resolved.get(left_name)
        if left is None:
            continue
        for right_name in R081_LAUNCH_PATH_FIELDS[index + 1 :]:
            right = resolved.get(right_name)
            if right is not None and left == right:
                failures.append(
                    "launch paths must be all distinct:"
                    f" {left_name} equals {right_name}"
                )

    quarantine = resolved.get("quarantine_dir")
    promote = resolved.get("promote_to")
    if quarantine is not None and promote is not None and (
        quarantine == promote
        or quarantine in promote.parents
        or promote in quarantine.parents
    ):
        failures.append(
            "environment quarantine_dir and promote_to must be disjoint"
            " (neither equal nor nested)"
        )

    ledger = resolved.get("exception_ledger_path")
    if ledger is not None:
        for workspace_name, workspace in (
            ("quarantine_dir", quarantine),
            ("promote_to", promote),
        ):
            if workspace is not None and (
                ledger == workspace
                or workspace in ledger.parents
                or ledger in workspace.parents
            ):
                failures.append(
                    "environment exception_ledger_path must be outside"
                    f" and disjoint from {workspace_name}"
                )

    return failures


def suite_command_failures(name: Any, command: Any) -> list[str]:
    """Validate one R-070 pytest receipt's exact non-vacuous selection."""
    if name not in SUITE_RECEIPT_NAMES:
        return [f"unknown suite identity {name!r}"]
    if not isinstance(command, list) or not all(
        isinstance(part, str) for part in command
    ):
        return [f"{name} receipt command must be an argv list of strings"]
    failures: list[str] = []
    if len(command) < 3 or command[1:3] != ["-m", "pytest"]:
        failures.append(
            f"{name} receipt command must begin interpreter -m pytest"
        )
        return failures

    selections: list[str] = []
    quiet_count = 0
    no_cache_count = 0
    junit_values: list[str] = []
    index = 3
    while index < len(command):
        token = command[index]
        if token == "-q":
            quiet_count += 1
            index += 1
            continue
        if token == "-p":
            if index + 1 >= len(command):
                failures.append(f"{name} receipt command has dangling -p")
                index += 1
                continue
            plugin = command[index + 1]
            if plugin != "no:cacheprovider":
                failures.append(
                    f"{name} receipt command -p must select"
                    f" 'no:cacheprovider'; found {plugin!r}"
                )
            else:
                no_cache_count += 1
            index += 2
            continue
        if token == "--junitxml":
            if index + 1 >= len(command) or not command[index + 1]:
                failures.append(
                    f"{name} receipt command has empty --junitxml path"
                )
                index += 1
                continue
            junit_values.append(command[index + 1])
            index += 2
            continue
        if token.startswith("--junitxml="):
            value = token.partition("=")[2]
            if not value:
                failures.append(
                    f"{name} receipt command has empty --junitxml path"
                )
            else:
                junit_values.append(value)
            index += 1
            continue
        if token.startswith("-"):
            failures.append(
                f"{name} receipt command carries non-allowlisted pytest"
                f" option {token!r}"
            )
        else:
            selections.append(token)
        index += 1

    if quiet_count != 1:
        failures.append(
            f"{name} receipt command must carry -q exactly once;"
            f" found {quiet_count}"
        )
    if no_cache_count != 1:
        failures.append(
            f"{name} receipt command must carry -p no:cacheprovider"
            f" exactly once; found {no_cache_count}"
        )
    if len(junit_values) != 1:
        failures.append(
            f"{name} receipt command must carry --junitxml exactly once;"
            f" found {len(junit_values)}"
        )
    else:
        try:
            if not Path(junit_values[0]).is_absolute():
                failures.append(
                    f"{name} receipt --junitxml path must be absolute"
                )
        except (OSError, TypeError, ValueError):
            failures.append(
                f"{name} receipt --junitxml path is malformed"
            )

    expected_selection = (
        FOCUSED_SUITE_SELECTION if name == "focused" else FULL_SUITE_SELECTION
    )
    if name == "focused":
        if (
            len(selections) != len(expected_selection)
            or len(set(selections)) != len(selections)
            or frozenset(selections) != frozenset(expected_selection)
        ):
            failures.append(
                "focused receipt command must select exactly the canonical"
                f" focused file set {list(expected_selection)!r}; found"
                f" {selections!r}"
            )
    elif selections != list(expected_selection):
        failures.append(
            "full receipt command must select exactly ['tests/'];"
            f" found {selections!r}"
        )
    return failures


def receipt_environment_failures(
    repo_root: Any, receipts: Any, environment: Any
) -> list[str]:
    """Cross-bind both suite receipts to the certified runtime and lock."""
    failures: list[str] = []
    try:
        root = Path(repo_root)
    except (OSError, TypeError, ValueError):
        return [
            f"{R082_REPO_ROOT_FIELD} is missing or is not a valid path"
        ]
    if not root.is_absolute():
        return [f"{R082_REPO_ROOT_FIELD} must be an absolute path"]
    try:
        root = root.resolve()
    except (OSError, RuntimeError, ValueError) as exc:
        return [
            f"{R082_REPO_ROOT_FIELD} could not be resolved"
            f" ({exc.__class__.__name__})"
        ]
    if not isinstance(receipts, dict):
        return ["suite_receipts component is not an object"]
    if not isinstance(environment, dict):
        return ["environment component is not an object"]

    interpreter = environment.get("interpreter_realpath")
    lock_digest = environment.get("environment_lock_sha256")
    interpreter_path: Path | None = None
    if not isinstance(interpreter, str) or not interpreter:
        failures.append(
            "environment interpreter_realpath is missing or malformed"
        )
    else:
        try:
            interpreter_path = Path(interpreter)
            if not interpreter_path.is_absolute():
                failures.append(
                    "environment interpreter_realpath must be absolute"
                )
                interpreter_path = None
            else:
                interpreter_path = interpreter_path.resolve()
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            failures.append(
                "environment interpreter_realpath could not be resolved"
                f" ({exc.__class__.__name__})"
            )
            interpreter_path = None

    for name in SUITE_RECEIPT_NAMES:
        receipt = receipts.get(name)
        if not isinstance(receipt, dict):
            failures.append(f"{name} receipt is missing or malformed")
            continue
        receipt_interpreter = receipt.get("interpreter_realpath")
        if receipt_interpreter != interpreter:
            failures.append(
                f"{name} receipt interpreter_realpath"
                f" {receipt_interpreter!r} does not equal environment"
                f" interpreter_realpath {interpreter!r}"
            )
        receipt_lock = receipt.get("environment_lock_sha256")
        if receipt_lock != lock_digest:
            failures.append(
                f"{name} receipt environment_lock_sha256"
                f" {receipt_lock!r} does not equal environment"
                f" environment_lock_sha256 {lock_digest!r}"
            )

        command = receipt.get("command")
        failures.extend(suite_command_failures(name, command))
        if (
            not isinstance(command, list)
            or not command
            or not isinstance(command[0], str)
            or not command[0]
        ):
            failures.append(
                f"{name} receipt command must be a non-empty argv list"
                " whose command[0] is the suite interpreter"
            )
            continue
        if command[0] == interpreter:
            continue
        try:
            command_interpreter = Path(command[0])
            if not command_interpreter.is_absolute():
                command_interpreter = root / command_interpreter
            command_interpreter = command_interpreter.resolve()
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            failures.append(
                f"{name} receipt command[0] could not be resolved"
                f" ({exc.__class__.__name__})"
            )
            continue
        if (
            interpreter_path is None
            or command_interpreter != interpreter_path
        ):
            failures.append(
                f"{name} receipt command[0] {command[0]!r} does not resolve"
                f" to environment interpreter {interpreter!r}"
            )

    return failures


def command_environment_failures(
    repo_root: Any, environment: Any
) -> list[str]:
    """Bind producer command[0] to the certified interpreter realpath."""
    try:
        root = Path(repo_root)
    except (OSError, TypeError, ValueError):
        return [
            f"{R082_REPO_ROOT_FIELD} is missing or is not a valid path"
        ]
    if not root.is_absolute():
        return [f"{R082_REPO_ROOT_FIELD} must be an absolute path"]
    try:
        root = root.resolve()
    except (OSError, RuntimeError, ValueError) as exc:
        return [
            f"{R082_REPO_ROOT_FIELD} could not be resolved"
            f" ({exc.__class__.__name__})"
        ]
    if not isinstance(environment, dict):
        return ["environment component is not an object"]

    interpreter = environment.get("interpreter_realpath")
    command = environment.get("command")
    if not isinstance(interpreter, str) or not interpreter:
        return ["environment interpreter_realpath is missing or malformed"]
    if (
        not isinstance(command, list)
        or not command
        or not isinstance(command[0], str)
        or not command[0]
    ):
        return [
            "environment command must carry a non-empty command[0]"
            " interpreter"
        ]
    try:
        interpreter_path = Path(interpreter)
        if not interpreter_path.is_absolute():
            return ["environment interpreter_realpath must be absolute"]
        interpreter_path = interpreter_path.resolve()
        command_interpreter = Path(command[0])
        if not command_interpreter.is_absolute():
            command_interpreter = root / command_interpreter
        command_interpreter = command_interpreter.resolve()
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        return [
            "producer command interpreter binding could not be resolved"
            f" ({exc.__class__.__name__})"
        ]
    if command_interpreter != interpreter_path:
        return [
            f"environment command[0] {command[0]!r} does not resolve to"
            f" certified interpreter_realpath {interpreter!r}"
        ]
    return []


def snapshot_manifest_failures(
    repo_root: Any, snapshots: Any, command: Any
) -> list[str]:
    """Bind the command manifest path to the raw-hashed snapshot artifact."""
    try:
        root = Path(repo_root)
    except (OSError, TypeError, ValueError):
        return [
            f"{R082_REPO_ROOT_FIELD} is missing or is not a valid path"
        ]
    if not root.is_absolute():
        return [f"{R082_REPO_ROOT_FIELD} must be an absolute path"]
    try:
        root = root.resolve()
    except (OSError, RuntimeError, ValueError) as exc:
        return [
            f"{R082_REPO_ROOT_FIELD} could not be resolved"
            f" ({exc.__class__.__name__})"
        ]
    if not isinstance(snapshots, dict):
        return ["snapshots component is not an object"]
    artifact_path = snapshots.get("artifact_path")
    if not isinstance(artifact_path, str) or not artifact_path:
        return ["snapshots artifact_path is missing or malformed"]
    try:
        certified_path = Path(artifact_path)
        if not certified_path.is_absolute():
            return ["snapshots artifact_path must be absolute"]
        certified_path = certified_path.resolve()
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        return [
            "snapshots artifact_path could not be resolved"
            f" ({exc.__class__.__name__})"
        ]
    canonical_path = (root / SNAPSHOT_MANIFEST_RELPATH).resolve()
    if certified_path != canonical_path:
        return [
            "snapshots artifact_path"
            f" {str(certified_path)!r} is not the canonical repo manifest"
            f" {str(canonical_path)!r}"
        ]

    values: list[str] = []
    if isinstance(command, list):
        index = 0
        while index < len(command):
            token = command[index]
            if token == "--snapshot-manifest":
                if index + 1 < len(command) and isinstance(
                    command[index + 1], str
                ):
                    values.append(command[index + 1])
                    index += 1
            elif isinstance(token, str) and token.startswith(
                "--snapshot-manifest="
            ):
                values.append(token.partition("=")[2])
            index += 1
    if len(values) != 1:
        return [
            "exact command --snapshot-manifest must appear exactly once;"
            f" found {len(values)}"
        ]
    try:
        command_path = Path(values[0])
        if not command_path.is_absolute():
            command_path = root / command_path
        command_path = command_path.resolve()
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        return [
            "exact command --snapshot-manifest could not be resolved"
            f" ({exc.__class__.__name__})"
        ]
    if command_path != certified_path:
        return [
            "exact command --snapshot-manifest path"
            f" {str(command_path)!r} does not match snapshots artifact_path"
            f" {str(certified_path)!r}"
        ]
    return []


def content_hash_failures(
    repo_root: Any, content_hashes: Any
) -> list[str]:
    """Validate the closed, canonical audited-source hash component.

    Source digests are intentionally not compiled into this module: the
    certificate may cover this file and the governing spec in the repair
    commit itself.  Instead, each digest is bound to one canonical repo file;
    gathering computes its raw-byte hash and the launcher rehashes it before
    activation.
    """
    failures: list[str] = []
    try:
        root = Path(repo_root)
    except (OSError, TypeError, ValueError):
        return [
            f"{R082_REPO_ROOT_FIELD} is missing or is not a valid path"
        ]
    if not root.is_absolute():
        return [f"{R082_REPO_ROOT_FIELD} must be an absolute path"]
    try:
        root = root.resolve()
    except (OSError, RuntimeError, ValueError) as exc:
        return [
            f"{R082_REPO_ROOT_FIELD} could not be resolved"
            f" ({exc.__class__.__name__})"
        ]
    if not isinstance(content_hashes, dict):
        return ["component must be an object"]

    expected_keys = set(CONTENT_HASH_RELPATHS)
    observed_keys = set(content_hashes)
    missing = sorted(expected_keys - observed_keys, key=repr)
    unexpected = sorted(observed_keys - expected_keys, key=repr)
    if missing:
        failures.append(f"missing required key(s) {missing!r}")
    if unexpected:
        failures.append(f"unexpected key(s) {unexpected!r}")

    for key, relpath in CONTENT_HASH_RELPATHS.items():
        if key not in content_hashes:
            continue
        entry = content_hashes[key]
        if not isinstance(entry, dict):
            failures.append(f"{key} entry must be an object")
            continue
        entry_keys = set(entry)
        missing_fields = sorted(
            CONTENT_HASH_ENTRY_KEYS - entry_keys, key=repr
        )
        unexpected_fields = sorted(
            entry_keys - CONTENT_HASH_ENTRY_KEYS, key=repr
        )
        if missing_fields:
            failures.append(
                f"{key} missing field(s) {missing_fields!r}"
            )
        if unexpected_fields:
            failures.append(
                f"{key} unexpected field(s) {unexpected_fields!r}"
            )
        if not schema.is_sha256_hex(entry.get("sha256")):
            failures.append(f"{key} sha256 is not a sha256 hex digest")

        raw_path = entry.get("artifact_path")
        if not isinstance(raw_path, str) or not raw_path:
            failures.append(
                f"{key} artifact_path must be a non-empty absolute path"
            )
            continue
        try:
            observed_path = Path(raw_path)
            if not observed_path.is_absolute():
                failures.append(f"{key} artifact_path must be absolute")
                continue
            resolved_path = observed_path.resolve()
            expected_path = (root / relpath).resolve()
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            failures.append(
                f"{key} artifact_path could not be resolved"
                f" ({exc.__class__.__name__})"
            )
            continue
        if resolved_path != expected_path:
            failures.append(
                f"{key} artifact_path {str(resolved_path)!r} is not the"
                f" canonical repo artifact {str(expected_path)!r}"
            )
        if os.path.normcase(str(observed_path)) != os.path.normcase(
            str(resolved_path)
        ):
            failures.append(
                f"{key} artifact_path must be recorded as its canonical"
                " resolved path"
            )
    return failures


def canonical_artifact_path_failures(
    repo_root: Any,
    component: Any,
    *,
    artifact_relpath: str,
    component_name: str,
) -> list[str]:
    """Require one certificate artifact path to name its canonical repo file."""
    try:
        root = Path(repo_root)
    except (OSError, TypeError, ValueError):
        return [
            f"{R082_REPO_ROOT_FIELD} is missing or is not a valid path"
        ]
    if not root.is_absolute():
        return [f"{R082_REPO_ROOT_FIELD} must be an absolute path"]
    if not isinstance(component, dict):
        return [f"{component_name} component is not an object"]
    raw_path = component.get("artifact_path")
    if not isinstance(raw_path, str) or not raw_path:
        return [f"{component_name} artifact_path is missing or malformed"]
    try:
        root = root.resolve()
        observed = Path(raw_path)
        if not observed.is_absolute():
            return [f"{component_name} artifact_path must be absolute"]
        observed = observed.resolve()
        expected = (root / artifact_relpath).resolve()
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        return [
            f"{component_name} artifact path binding could not be resolved"
            f" ({exc.__class__.__name__})"
        ]
    if observed != expected:
        return [
            f"{component_name} artifact_path {str(observed)!r} is not the"
            f" canonical repo artifact {str(expected)!r}"
        ]
    return []


def _r082_resolve_bound_path(
    raw_path: Any, root: Path
) -> tuple[Path | None, str | None]:
    """Resolve one child-cwd-relative path into a comparable canonical path."""
    if not isinstance(raw_path, str) or not raw_path:
        return None, "path is missing or is not a string"
    try:
        path = Path(raw_path)
        if not path.is_absolute():
            path = root / path
        return path.resolve(), None
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        return None, f"path could not be resolved ({exc.__class__.__name__})"


def staged_coverage_failures(
    repo_root: Any, staged_inputs: Any, eligibility: Any, command: Any
) -> list[str]:
    """Validate the exact six-input certificate/command coverage contract.

    The component owns one entry for each consumed input.  The exact command
    independently owns the calibration path, data directory, and four
    operator-supplied path+digest bindings.  Comparing both representations
    here prevents a syntactically ready certificate from deferring an
    uncovered-input refusal until after the one-shot ledger is consumed.
    """
    failures: list[str] = []
    try:
        root = Path(repo_root)
    except (OSError, TypeError, ValueError):
        return [
            f"{R082_REPO_ROOT_FIELD} is missing or is not a valid path"
        ]
    if not root.is_absolute():
        return [f"{R082_REPO_ROOT_FIELD} must be an absolute path"]
    try:
        root = root.resolve()
    except (OSError, RuntimeError, ValueError) as exc:
        return [
            f"{R082_REPO_ROOT_FIELD} could not be resolved"
            f" ({exc.__class__.__name__})"
        ]

    required_labels = frozenset(R082_STAGED_INPUT_LABELS)
    component_by_label: dict[str, dict[str, Any]] = {}
    if not isinstance(staged_inputs, list):
        failures.append("staged_inputs component is not a list")
    else:
        for index, entry in enumerate(staged_inputs):
            if not isinstance(entry, dict):
                failures.append(
                    f"staged_inputs[{index}] is not an object"
                )
                continue
            label = entry.get("label")
            if not isinstance(label, str) or not label:
                failures.append(
                    f"staged_inputs[{index}] label is missing or malformed"
                )
                continue
            if label not in required_labels:
                failures.append(f"unknown component label {label!r}")
                continue
            if label in component_by_label:
                failures.append(f"duplicate component label {label!r}")
                continue
            component_by_label[label] = entry

    for label in R082_STAGED_INPUT_LABELS:
        if label not in component_by_label:
            failures.append(f"missing required label {label!r}")

    eligibility_component = (
        eligibility if isinstance(eligibility, dict) else {}
    )

    options, operator_entries, parse_failures = _r082_command_bindings(
        command
    )
    failures.extend(parse_failures)

    unique_options: dict[str, str] = {}
    for kind, display in (
        ("data_dir", "--data-dir"),
        ("calibration", "--calibration"),
        ("eligibility", "--eligibility"),
        ("fit_split_name", "--fit-split"),
        ("eval_split_name", "--eval-split"),
    ):
        values = options[kind]
        if len(values) != 1:
            failures.append(
                f"exact command {display} must appear exactly once;"
                f" found {len(values)}"
            )
        else:
            unique_options[kind] = values[0][1]

    split_names: dict[str, str] = {}
    for kind, display, label in (
        ("fit_split_name", "--fit-split", "fit_split"),
        ("eval_split_name", "--eval-split", "eval_split"),
    ):
        value = unique_options.get(kind)
        if value is None:
            continue
        if not schema.is_path_component(value):
            failures.append(
                f"exact command {display} value {value!r} must be a simple"
                " non-empty split token"
            )
            continue
        required_split = "val" if label == "fit_split" else "test"
        if value != required_split:
            failures.append(
                f"exact command {display} must equal {required_split!r};"
                f" found {value!r}"
            )
        split_names[label] = value

    expected_paths: dict[str, Path] = {}
    calibration = unique_options.get("calibration")
    if calibration is not None:
        resolved, error = _r082_resolve_bound_path(calibration, root)
        if error is not None:
            failures.append(f"exact command --calibration {error}")
        elif resolved is not None:
            expected_paths["calibration_train"] = resolved

    eligibility_path = unique_options.get("eligibility")
    certified_eligibility_path, eligibility_path_error = (
        _r082_resolve_bound_path(
            eligibility_component.get("artifact_path"), root
        )
    )
    if eligibility_path_error is not None:
        failures.append(
            f"eligibility artifact_path {eligibility_path_error}"
        )
    elif certified_eligibility_path is not None:
        canonical_eligibility_path = (
            root / ELIGIBILITY_ARTIFACT_RELPATH
        ).resolve()
        if certified_eligibility_path != canonical_eligibility_path:
            failures.append(
                "eligibility artifact_path"
                f" {str(certified_eligibility_path)!r} is not the canonical"
                f" repo artifact {str(canonical_eligibility_path)!r}"
            )
    if eligibility_path is not None:
        command_eligibility_path, error = _r082_resolve_bound_path(
            eligibility_path, root
        )
        if error is not None:
            failures.append(f"exact command --eligibility {error}")
        elif (
            command_eligibility_path is not None
            and certified_eligibility_path is not None
            and command_eligibility_path != certified_eligibility_path
        ):
            failures.append(
                "exact command --eligibility path"
                f" {str(command_eligibility_path)!r} does not match"
                " eligibility artifact_path"
                f" {str(certified_eligibility_path)!r}"
            )

    data_dir = unique_options.get("data_dir")
    if data_dir is not None:
        resolved, error = _r082_resolve_bound_path(data_dir, root)
        if error is not None:
            failures.append(f"exact command --data-dir {error}")
        elif resolved is not None:
            data_filenames = {
                label: filename
                for label, filename in R082_DATA_FILENAMES.items()
                if label not in {"fit_split", "eval_split"}
            }
            data_filenames.update(
                {
                    label: f"{split_name}_dataset.json"
                    for label, split_name in split_names.items()
                }
            )
            expected_paths.update(
                {
                    label: (resolved / filename).resolve()
                    for label, filename in data_filenames.items()
                }
            )

    component_paths: dict[str, Path] = {}
    for label, entry in component_by_label.items():
        resolved, error = _r082_resolve_bound_path(entry.get("path"), root)
        if error is not None:
            failures.append(f"component {label!r} {error}")
            continue
        if resolved is None:
            continue
        component_paths[label] = resolved
        expected = expected_paths.get(label)
        if expected is not None and resolved != expected:
            failures.append(
                f"component {label!r} path {str(resolved)!r} does not match"
                f" exact command binding {str(expected)!r}"
            )

    for label, frozen_pin in R082_STAGED_INPUT_SHA256.items():
        component = component_by_label.get(label)
        if component is not None and (
            component.get("expected_sha256") != frozen_pin
        ):
            failures.append(
                f"component {label!r} expected_sha256"
                f" {component.get('expected_sha256')!r} does not equal"
                f" frozen pin {frozen_pin!r}"
            )

    eval_component = component_by_label.get("eval_split")
    eligibility_eval_digest = eligibility_component.get(
        "test_dataset_sha256"
    )
    if not schema.is_sha256_hex(eligibility_eval_digest):
        failures.append(
            "eligibility test_dataset_sha256 is not a sha256 hex digest"
        )
    elif eval_component is not None and (
        eval_component.get("expected_sha256") != eligibility_eval_digest
    ):
        failures.append(
            "component 'eval_split' expected_sha256"
            f" {eval_component.get('expected_sha256')!r} does not match"
            " eligibility test_dataset_sha256"
            f" {eligibility_eval_digest!r}"
        )

    operator_labels = frozenset(R082_OPERATOR_DIGEST_LABELS)
    operator_by_label: dict[str, dict[str, str]] = {}
    for entry in operator_entries:
        label = entry["label"]
        if label not in operator_labels:
            failures.append(f"unknown operator label {label!r}")
            continue
        if label in operator_by_label:
            failures.append(f"duplicate operator label {label!r}")
            continue
        operator_by_label[label] = entry

    for label in R082_OPERATOR_DIGEST_LABELS:
        if label not in operator_by_label:
            failures.append(f"missing operator label {label!r}")

    for label, operator in operator_by_label.items():
        operator_path, error = _r082_resolve_bound_path(
            operator["path"], root
        )
        if error is not None:
            failures.append(f"operator {label!r} {error}")
        else:
            component_path = component_paths.get(label)
            if (
                operator_path is not None
                and component_path is not None
                and operator_path != component_path
            ):
                failures.append(
                    f"operator {label!r} path {str(operator_path)!r} does"
                    f" not match component path {str(component_path)!r}"
                )
        component = component_by_label.get(label)
        if component is not None and (
            operator["expected_sha256"] != component.get("expected_sha256")
        ):
            failures.append(
                f"operator {label!r} digest"
                f" {operator['expected_sha256']!r} contradicts component"
                f" expected_sha256 {component.get('expected_sha256')!r}"
            )

    return failures


def _load_json_object(
    path: Path, error_cls: type[schema.TypedIngressError]
) -> tuple[dict[str, Any], str]:
    """Bounded strict-parse of one frozen JSON artifact, version-first."""
    path = Path(path)
    rel = path.name
    data = schema.read_regular_file_bytes(path)
    try:
        obj = schema.parse_json_bytes_strict(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise error_cls(f"{rel}: malformed JSON: {exc} (R-020)") from exc
    if not isinstance(obj, dict):
        raise error_cls(f"{rel}: artifact must be a JSON object (R-020)")
    # Version-first (R-059): the shared bool-safe checker runs before any
    # other key validation so mixed-invalid artifacts surface the VERSION
    # error. SchemaVersionError is itself a TypedIngressError subclass.
    schema.check_schema_version(obj, rel)
    return obj, rel


def _check_closed_keys(
    obj: dict[str, Any],
    allowed: frozenset[str],
    required: frozenset[str],
    where: str,
    error_cls: type[schema.TypedIngressError],
) -> None:
    unknown = sorted(set(obj) - allowed)
    if unknown:
        raise error_cls(
            f"{where}: unknown field(s) {unknown} — no silent key-dropping"
            " (R-020/R-063)"
        )
    missing = sorted(required - set(obj))
    if missing:
        raise error_cls(f"{where}: missing required field(s) {missing}")


def _sha256_regular_file(
    path: Path, *, error_cls: type[schema.ColmAimsError], label: str
) -> str:
    """Streaming SHA-256 of one regular file, symlink-free and FIFO-safe.

    DECISION: staged inputs include multi-hundred-MB data files
    (mc_dataset.json is ~330 MB), so this deliberately does NOT reuse
    ``schema.read_regular_file_bytes`` and its 64 MB parse-artifact cap —
    the gate only hashes, never parses. The O_NOFOLLOW + S_ISREG
    discipline is identical.
    """
    path = Path(path)
    name = path.name
    if path.is_symlink():
        raise error_cls(f"{label} ({name}): refusing to hash a symlink (R-020)")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        if exc.errno in (errno.ELOOP, errno.EMLINK):
            raise error_cls(
                f"{label} ({name}): refusing to hash a symlink (R-020)"
            ) from exc
        raise error_cls(
            f"{label} ({name}): missing or unreadable"
            f" ({exc.__class__.__name__}) (R-076)"
        ) from exc
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise error_cls(
                f"{label} ({name}): not a regular file — refusing to hash a"
                " FIFO, device, or socket (R-020)"
            )
        if hasattr(os, "O_NONBLOCK") and hasattr(os, "get_blocking"):
            os.set_blocking(fd, True)
        digest = hashlib.sha256()
        while True:
            chunk = os.read(fd, 1 << 20)
            if not chunk:
                break
            digest.update(chunk)
        return digest.hexdigest()
    finally:
        os.close(fd)


# ---------------------------------------------------------------------------
# R-074: frozen pairing eligibility loader
# ---------------------------------------------------------------------------


def load_pairing_eligibility(path: Path) -> dict[str, Any]:
    """Strict, closed-key, digest-recomputing load of the frozen pairing
    eligibility artifact (R-074).

    Recomputes BOTH digests — ``pairing.keyset_sha256(eligible_keys)`` and
    ``schema.horizon_map_sha256(horizon_map)`` — and compares them to the
    declared values (recompute-from-source, never mirror-equality). ANY
    mismatch or malformation raises ``EligibilityArtifactError``.
    """
    obj, rel = _load_json_object(path, EligibilityArtifactError)
    _check_closed_keys(
        obj, ELIGIBILITY_KEYS, ELIGIBILITY_KEYS, rel, EligibilityArtifactError
    )
    if obj["artifact_type"] != ELIGIBILITY_ARTIFACT_TYPE:
        raise EligibilityArtifactError(
            f"{rel}: artifact_type {obj['artifact_type']!r} is not"
            f" {ELIGIBILITY_ARTIFACT_TYPE!r} (R-074)"
        )

    keys = obj["eligible_keys"]
    if not isinstance(keys, list) or not all(
        isinstance(k, str) and k for k in keys
    ):
        raise EligibilityArtifactError(
            f"{rel}: eligible_keys must be a list of non-empty strings"
            " (R-074)"
        )
    if len(set(keys)) != len(keys):
        raise EligibilityArtifactError(
            f"{rel}: eligible_keys carries duplicate item keys (R-074)"
        )
    # keyset_sha256 sorts internally, so an out-of-order artifact would still
    # digest-match — sortedness needs its OWN check (RED contract).
    if keys != pairing.canonical_item_order(keys):
        raise EligibilityArtifactError(
            f"{rel}: eligible_keys are not sorted ascending by UTF-8 byte"
            " order (R-074)"
        )
    count = obj["eligible_count"]
    if not schema.is_real_int(count) or count != len(keys):
        raise EligibilityArtifactError(
            f"{rel}: eligible_count {count!r} does not equal the actual"
            f" eligible key count {len(keys)} — count drift (R-074)"
        )
    if count != EXPECTED_ELIGIBLE_COUNT:
        raise EligibilityArtifactError(
            f"{rel}: eligible_count {count!r} is not the frozen pinned"
            f" population {EXPECTED_ELIGIBLE_COUNT} (R-074/R-042)"
        )

    excluded = obj["excluded"]
    if not isinstance(excluded, list):
        raise EligibilityArtifactError(
            f"{rel}: excluded must be a list of exclusion entries (R-074)"
        )
    eligible_set = set(keys)
    seen_excluded: set[str] = set()
    for index, entry in enumerate(excluded):
        where = f"{rel}: excluded[{index}]"
        if not isinstance(entry, dict):
            raise EligibilityArtifactError(f"{where}: must be an object")
        _check_closed_keys(
            entry,
            EXCLUDED_ENTRY_KEYS,
            EXCLUDED_ENTRY_KEYS,
            where,
            EligibilityArtifactError,
        )
        item_key = entry["item_key"]
        if not isinstance(item_key, str) or not item_key:
            raise EligibilityArtifactError(
                f"{where}: item_key must be a non-empty string (R-074)"
            )
        if item_key in eligible_set:
            raise EligibilityArtifactError(
                f"{where}: excluded item also appears in eligible_keys"
                " (R-074/R-008)"
            )
        if item_key in seen_excluded:
            raise EligibilityArtifactError(
                f"{where}: duplicate excluded item key (R-074/R-008)"
            )
        seen_excluded.add(item_key)
        reason = entry["reason"]
        if (
            reason not in schema.EXCLUSION_REASONS
            or reason != schema.SINGLE_PREFIX_TRAJECTORY
        ):
            raise EligibilityArtifactError(
                f"{where}: reason {reason!r} is not the enumerated frozen"
                f" derivation reason {schema.SINGLE_PREFIX_TRAJECTORY!r}"
                " (R-074)"
            )
    excluded_count = obj["excluded_count"]
    if not schema.is_real_int(excluded_count) or excluded_count != len(excluded):
        raise EligibilityArtifactError(
            f"{rel}: excluded_count {excluded_count!r} does not equal the"
            f" actual excluded entry count {len(excluded)} — count drift"
            " (R-074)"
        )
    if excluded_count != EXPECTED_EXCLUDED_COUNT:
        raise EligibilityArtifactError(
            f"{rel}: excluded_count {excluded_count!r} is not the frozen"
            f" pinned exclusion count {EXPECTED_EXCLUDED_COUNT} (R-074)"
        )

    horizon_map = obj["horizon_map"]
    if not isinstance(horizon_map, dict):
        raise EligibilityArtifactError(
            f"{rel}: horizon_map must be an object (R-073/R-074)"
        )
    if set(horizon_map) != eligible_set:
        raise EligibilityArtifactError(
            f"{rel}: horizon_map keys do not cover exactly the eligible"
            " keys (R-073/R-074)"
        )
    for value in horizon_map.values():
        if not schema.is_real_int(value):
            raise EligibilityArtifactError(
                f"{rel}: horizon_map value {value!r} is outside the"
                " positive-int domain — bools never satisfy an integer"
                " domain (R-061/R-073)"
            )
        if value < MIN_ELIGIBLE_HORIZON:
            raise EligibilityArtifactError(
                f"{rel}: horizon_map value {value!r} is below the minimum"
                f" eligible horizon {MIN_ELIGIBLE_HORIZON} — a sub-2 horizon"
                " contradicts the SINGLE_PREFIX_TRAJECTORY exclusion rule"
                " that produced this artifact (R-074)"
            )

    declared_horizon_digest = obj["horizon_map_sha256"]
    if not schema.is_sha256_hex(declared_horizon_digest):
        raise EligibilityArtifactError(
            f"{rel}: horizon_map_sha256 is not a lowercase sha256 hex digest"
            " (R-073)"
        )
    try:
        recomputed_horizon = schema.horizon_map_sha256(horizon_map)
    except schema.ColmAimsError as exc:
        raise EligibilityArtifactError(f"{rel}: {exc}") from exc
    if recomputed_horizon != declared_horizon_digest:
        raise EligibilityArtifactError(
            f"{rel}: recomputed horizon-map digest {recomputed_horizon} !="
            f" declared horizon_map_sha256 {declared_horizon_digest}"
            " (R-073/R-074)"
        )
    if declared_horizon_digest != ELIGIBILITY_HORIZON_MAP_SHA256:
        raise EligibilityArtifactError(
            f"{rel}: horizon_map_sha256 {declared_horizon_digest} is not"
            f" the frozen canonical pin {ELIGIBILITY_HORIZON_MAP_SHA256}"
            " (R-074)"
        )

    declared_keyset_digest = obj["pairing_population_keyset_sha256"]
    if not schema.is_sha256_hex(declared_keyset_digest):
        raise EligibilityArtifactError(
            f"{rel}: pairing_population_keyset_sha256 is not a lowercase"
            " sha256 hex digest (R-074)"
        )
    recomputed_keyset = pairing.keyset_sha256(keys)
    if recomputed_keyset != declared_keyset_digest:
        raise EligibilityArtifactError(
            f"{rel}: recomputed eligible-keyset digest {recomputed_keyset}"
            f" != declared pairing_population_keyset_sha256"
            f" {declared_keyset_digest} (R-074/R-052)"
        )
    if declared_keyset_digest != ELIGIBILITY_KEYSET_SHA256:
        raise EligibilityArtifactError(
            f"{rel}: pairing_population_keyset_sha256"
            f" {declared_keyset_digest} is not the frozen canonical pin"
            f" {ELIGIBILITY_KEYSET_SHA256} (R-074)"
        )

    derived = obj["derived_from"]
    if not isinstance(derived, dict):
        raise EligibilityArtifactError(
            f"{rel}: derived_from must be an object (R-074)"
        )
    _check_closed_keys(
        derived,
        DERIVED_FROM_KEYS,
        DERIVED_FROM_KEYS,
        f"{rel}: derived_from",
        EligibilityArtifactError,
    )
    if not schema.is_sha256_hex(derived["test_dataset_sha256"]):
        raise EligibilityArtifactError(
            f"{rel}: derived_from.test_dataset_sha256 is not a sha256 digest"
            " (R-074)"
        )
    if derived["test_dataset_sha256"] != ELIGIBILITY_TEST_DATASET_SHA256:
        raise EligibilityArtifactError(
            f"{rel}: derived_from.test_dataset_sha256"
            f" {derived['test_dataset_sha256']} is not the frozen canonical"
            f" pin {ELIGIBILITY_TEST_DATASET_SHA256} (R-074)"
        )
    for field in ("derivation", "test_dataset_basename", "two_party_pin"):
        if not isinstance(derived[field], str) or not derived[field]:
            raise EligibilityArtifactError(
                f"{rel}: derived_from.{field} must be a non-empty string"
                " (R-074)"
            )
    return obj


# ---------------------------------------------------------------------------
# R-076: staged-input hash gate (fail-closed, before any loader)
# ---------------------------------------------------------------------------


def staged_input_gate(staged: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Hash-verify EVERY staged fit/eval input fail-closed (R-076).

    Raises ``StagedInputError`` on the FIRST (list-order) missing file,
    hash mismatch, or malformed entry, naming the file plus the expected
    and observed digests. An EMPTY staged list raises — a gate over zero
    inputs is a vacuously-empty authoritative set, not a trivially-passing
    gate. Returns fresh entries carrying ``observed_sha256``.
    """
    if not isinstance(staged, list):
        raise StagedInputError(
            "staged inputs must be a list of {path, expected_sha256, label}"
            " entries (R-076)"
        )
    if not staged:
        raise StagedInputError(
            "staged-input gate invoked over ZERO inputs — an empty"
            " authoritative input set is a defect, never a trivially"
            " passing gate (R-076)"
        )
    verified: list[dict[str, Any]] = []
    for index, entry in enumerate(staged):
        where = f"staged input [{index}]"
        if not isinstance(entry, dict):
            raise StagedInputError(f"{where}: entry must be an object (R-076)")
        _check_closed_keys(
            entry, STAGED_ENTRY_KEYS, STAGED_ENTRY_KEYS, where, StagedInputError
        )
        label = entry["label"]
        if not isinstance(label, str) or not label:
            raise StagedInputError(
                f"{where}: label must be a non-empty string (R-076)"
            )
        expected = entry["expected_sha256"]
        path = Path(entry["path"])
        if not schema.is_sha256_hex(expected):
            raise StagedInputError(
                f"staged input {label!r} ({path.name}): expected_sha256"
                f" {expected!r} is not a lowercase 64-hex sha256 digest"
                " (R-076)"
            )
        observed = _sha256_regular_file(
            path, error_cls=StagedInputError, label=f"staged input {label!r}"
        )
        if observed != expected:
            raise StagedInputError(
                f"staged input {label!r} ({path.name}): observed sha256"
                f" {observed} != expected {expected} — refusing to proceed"
                " to any loader or model construction (R-076)"
            )
        verified.append(
            {
                "label": label,
                "path": str(path),
                "expected_sha256": expected,
                "observed_sha256": observed,
            }
        )
    return verified


def required_staged_coverage(
    consumed: list[dict[str, Any]], staged_entries: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Resolve the producer's consumed-input enumeration into the full
    staged-gate plan (F-1, R-076).

    ``consumed`` is the ordered enumeration of every fit/eval input as
    ``{"label", "path", "frozen_sha256": str | None}``; ``staged_entries``
    are the operator's ``--staged-input`` triples
    ``{"label", "path", "expected_sha256"}``. Returns one plan entry per
    consumed input, IN CONSUMED ORDER, each ``{"label", "path",
    "expected_sha256"}`` with the expected digest filled from the frozen
    pin when present (frozen-pin-wins), else from the operator entry
    covering the same path.

    Typed refusals (``StagedCoverageError``): (a) a consumed input with
    neither a frozen pin nor an operator digest (uncovered — named); (b) an
    operator digest CONTRADICTING a frozen pin (names the file and BOTH
    digests); (c) an operator entry naming a path outside the consumed set
    (unknown staged input — silently-ignored coverage is a defect); plus
    malformed shapes, duplicate contradictory operator entries, and the
    vacuously-empty consumed enumeration.
    """
    if not isinstance(consumed, list) or not consumed:
        raise StagedCoverageError(
            "consumed-input enumeration must be a non-empty list — a gate"
            " plan over ZERO consumed inputs is a vacuously-empty"
            " authoritative set (F-1/R-076)"
        )
    if not isinstance(staged_entries, list):
        raise StagedCoverageError(
            "staged entries must be a list of {label, path, expected_sha256}"
            " triples (F-1/R-076)"
        )

    # Operator entries keyed by resolved path; duplicate entries for one
    # path must agree (duplicate-key last-wins is a laundering vector).
    operator_by_path: dict[str, dict[str, Any]] = {}
    for index, entry in enumerate(staged_entries):
        where = f"staged entry [{index}]"
        if not isinstance(entry, dict):
            raise StagedCoverageError(f"{where}: entry must be an object (F-1)")
        _check_closed_keys(
            entry,
            STAGED_ENTRY_KEYS,
            STAGED_ENTRY_KEYS,
            where,
            StagedCoverageError,
        )
        label = entry["label"]
        if not isinstance(label, str) or not label:
            raise StagedCoverageError(
                f"{where}: label must be a non-empty string (F-1)"
            )
        digest = entry["expected_sha256"]
        if not schema.is_sha256_hex(digest):
            raise StagedCoverageError(
                f"staged entry {label!r}: expected_sha256 {digest!r} is not"
                " a lowercase 64-hex sha256 digest (F-1/R-076)"
            )
        resolved = str(Path(entry["path"]).resolve())
        prior = operator_by_path.get(resolved)
        if prior is not None and prior["expected_sha256"] != digest:
            raise StagedCoverageError(
                f"staged entries {prior['label']!r} and {label!r} both cover"
                f" {Path(resolved).name} with contradictory digests"
                f" {prior['expected_sha256']} != {digest} (F-1/R-076)"
            )
        operator_by_path[resolved] = {
            "label": label,
            "expected_sha256": digest,
        }

    plan: list[dict[str, Any]] = []
    consumed_paths: set[str] = set()
    for index, entry in enumerate(consumed):
        where = f"consumed input [{index}]"
        if not isinstance(entry, dict):
            raise StagedCoverageError(f"{where}: entry must be an object (F-1)")
        unknown = sorted(set(entry) - {"label", "path", "frozen_sha256"})
        if unknown:
            raise StagedCoverageError(
                f"{where}: unknown field(s) {unknown} — consumed entries are"
                " exactly {{label, path, frozen_sha256}} (F-1)"
            )
        missing = sorted({"label", "path", "frozen_sha256"} - set(entry))
        if missing:
            raise StagedCoverageError(
                f"{where}: missing required field(s) {missing} (F-1)"
            )
        label = entry["label"]
        if not isinstance(label, str) or not label:
            raise StagedCoverageError(
                f"{where}: label must be a non-empty string (F-1)"
            )
        path = Path(entry["path"])
        resolved = str(path.resolve())
        if resolved in consumed_paths:
            raise StagedCoverageError(
                f"consumed input {label!r} ({path.name}): duplicate consumed"
                " path — one gate entry per input (F-1)"
            )
        consumed_paths.add(resolved)
        frozen = entry["frozen_sha256"]
        if frozen is not None and not schema.is_sha256_hex(frozen):
            raise StagedCoverageError(
                f"consumed input {label!r} ({path.name}): frozen_sha256"
                f" {frozen!r} is neither None nor a lowercase 64-hex sha256"
                " digest (F-1)"
            )
        operator = operator_by_path.get(resolved)
        if frozen is not None:
            # Frozen-pin-wins: an agreeing operator entry is redundant; a
            # contradicting one is a laundering attempt and refuses loudly.
            if operator is not None and operator["expected_sha256"] != frozen:
                raise StagedCoverageError(
                    f"consumed input {label!r} ({path.name}): operator digest"
                    f" {operator['expected_sha256']} CONTRADICTS the frozen"
                    f" pin {frozen} — frozen pins are two-party and never"
                    " overridable (F-1/R-076)"
                )
            expected = frozen
        elif operator is not None:
            expected = operator["expected_sha256"]
        else:
            raise StagedCoverageError(
                f"consumed input {label!r} ({path.name}): UNCOVERED — no"
                " frozen pin and no --staged-input digest; every consumed"
                " fit/eval input must be hash-gated (F-1/R-076)"
            )
        plan.append(
            {"label": label, "path": path, "expected_sha256": expected}
        )

    uncovered_operators = sorted(
        set(operator_by_path) - consumed_paths
    )
    if uncovered_operators:
        names = [Path(p).name for p in uncovered_operators]
        raise StagedCoverageError(
            f"staged entr{'y' if len(names) == 1 else 'ies'} for"
            f" {names} name path(s) OUTSIDE the consumed-input set —"
            " unknown staged inputs are a defect, never silently-ignored"
            " coverage (F-1/R-076)"
        )
    return plan


# ---------------------------------------------------------------------------
# R-075: role-keyed model snapshot manifests + snapshot verification
# ---------------------------------------------------------------------------


def _is_hf_revision(value: Any) -> bool:
    """HF snapshot revisions are 40-hex git SHAs; 64-hex content digests are
    also admissible pins."""
    return schema.is_commit_sha(value) or schema.is_sha256_hex(value)


def _validate_role_entry(
    entry: Any,
    where: str,
    error_cls: type[schema.ColmAimsError],
) -> dict[str, Any]:
    if not isinstance(entry, dict):
        raise error_cls(f"{where}: role entry must be an object (R-075)")
    unknown = sorted(set(entry) - ROLE_ENTRY_KEYS)
    if unknown:
        raise error_cls(
            f"{where}: unknown role-entry field(s) {unknown} (R-075)"
        )
    missing = sorted(ROLE_ENTRY_KEYS - set(entry))
    if missing:
        raise error_cls(
            f"{where}: missing role-entry field(s) {missing} (R-075)"
        )
    if not isinstance(entry["model_name"], str) or not entry["model_name"]:
        raise error_cls(f"{where}: model_name must be a non-empty string")
    if not _is_hf_revision(entry["hf_revision"]):
        raise error_cls(
            f"{where}: hf_revision is not an immutable 40-hex git SHA or"
            " 64-hex content digest (R-075)"
        )
    files = entry["files"]
    if not isinstance(files, dict) or not files:
        raise error_cls(
            f"{where}: files must be a NON-EMPTY per-file manifest — an"
            " empty manifest is a vacuously-empty authoritative set (R-075)"
        )
    for rel_name, meta in files.items():
        file_where = f"{where}: files[{rel_name!r}]"
        if not isinstance(rel_name, str) or not rel_name:
            raise error_cls(f"{file_where}: file name must be a non-empty string")
        parts = rel_name.split("/")
        if not all(schema.is_path_component(part) for part in parts):
            raise error_cls(
                f"{file_where}: file name must be a relative path of plain"
                " components (no traversal, no absolute paths) (R-020)"
            )
        if not isinstance(meta, dict):
            raise error_cls(f"{file_where}: entry must be an object")
        unknown_meta = sorted(set(meta) - FILE_ENTRY_KEYS)
        missing_meta = sorted(FILE_ENTRY_KEYS - set(meta))
        if unknown_meta or missing_meta:
            raise error_cls(
                f"{file_where}: entry keys must be exactly"
                f" {sorted(FILE_ENTRY_KEYS)}"
            )
        if not schema.is_sha256_hex(meta["sha256"]):
            raise error_cls(f"{file_where}: sha256 is not a sha256 hex digest")
        if not schema.is_real_int(meta["size"]) or meta["size"] <= 0:
            raise error_cls(
                f"{file_where}: size must be a positive real integer —"
                " bools never satisfy an integer domain (R-061)"
            )
    file_count = entry["file_count"]
    if not schema.is_real_int(file_count) or file_count != len(files):
        raise error_cls(
            f"{where}: file_count {file_count!r} does not equal the actual"
            f" per-file manifest size {len(files)} (R-075)"
        )
    return entry


def load_model_snapshot_manifest(path: Path) -> dict[str, Any]:
    """Strict load of the frozen role-keyed model snapshot manifest (R-075)."""
    obj, rel = _load_json_object(path, SnapshotManifestError)
    _check_closed_keys(
        obj, MANIFEST_KEYS, MANIFEST_KEYS, rel, SnapshotManifestError
    )
    if obj["artifact_type"] != MANIFEST_ARTIFACT_TYPE:
        raise SnapshotManifestError(
            f"{rel}: artifact_type {obj['artifact_type']!r} is not"
            f" {MANIFEST_ARTIFACT_TYPE!r} (R-075)"
        )
    if not isinstance(obj["note"], str):
        raise SnapshotManifestError(f"{rel}: note must be a string (R-075)")
    roles = obj["roles"]
    if not isinstance(roles, dict) or set(roles) != SNAPSHOT_ROLES:
        raise SnapshotManifestError(
            f"{rel}: roles must be exactly {sorted(SNAPSHOT_ROLES)} — a"
            " missing or extra role fails closed (R-075)"
        )
    for role in sorted(SNAPSHOT_ROLES):
        _validate_role_entry(
            roles[role], f"{rel}: roles[{role!r}]", SnapshotManifestError
        )
    if obj["offline_flags_required"] != list(REQUIRED_OFFLINE_FLAGS):
        raise SnapshotManifestError(
            f"{rel}: offline_flags_required must be exactly"
            f" {list(REQUIRED_OFFLINE_FLAGS)} (R-075)"
        )
    tfidf = obj["tfidf_config"]
    if not isinstance(tfidf, dict) or set(tfidf) != TFIDF_CONFIG_KEYS:
        raise SnapshotManifestError(
            f"{rel}: tfidf_config keys must be exactly"
            f" {sorted(TFIDF_CONFIG_KEYS)} (R-075)"
        )
    if not isinstance(tfidf["analyzer"], str) or not tfidf["analyzer"]:
        raise SnapshotManifestError(
            f"{rel}: tfidf_config.analyzer must be a non-empty string"
        )
    ngram = tfidf["ngram_range"]
    if (
        not isinstance(ngram, list)
        or len(ngram) != 2
        or not all(schema.is_real_int(v) and v > 0 for v in ngram)
    ):
        raise SnapshotManifestError(
            f"{rel}: tfidf_config.ngram_range must be a 2-list of positive"
            " integers (R-075)"
        )
    if not isinstance(tfidf["fit_corpus"], str) or not tfidf["fit_corpus"]:
        raise SnapshotManifestError(
            f"{rel}: tfidf_config.fit_corpus must be a non-empty string"
        )
    return obj


def verify_snapshot_dir(
    manifest_role_entry: dict[str, Any], snapshot_dir: Path
) -> None:
    """Verify one local snapshot directory byte-for-byte against its pinned
    manifest entry (R-075): per-file sha256 AND size, no extra files, no
    missing files, file_count consistency. Raises ``SnapshotMismatchError``
    naming the offending relative path on any deviation.
    """
    entry = _validate_role_entry(
        manifest_role_entry, "snapshot manifest entry", SnapshotManifestError
    )
    snapshot_dir = Path(snapshot_dir)
    if snapshot_dir.is_symlink():
        raise SnapshotMismatchError(
            f"snapshot directory {snapshot_dir.name!r} is a symlink —"
            " refusing (R-075/R-013)"
        )
    if not snapshot_dir.is_dir():
        raise SnapshotMismatchError(
            f"snapshot directory {snapshot_dir.name!r} does not exist or is"
            " not a directory (R-075)"
        )
    declared: dict[str, dict[str, Any]] = entry["files"]
    observed: dict[str, Path] = {}
    for member in sorted(snapshot_dir.rglob("*")):
        rel_name = member.relative_to(snapshot_dir).as_posix()
        if member.is_symlink():
            # Symlinked tree members hash bytes from OUTSIDE the tree —
            # refuse (seed catalog: DoS/containment trio).
            raise SnapshotMismatchError(
                f"snapshot member {rel_name!r} is a symlink — refusing"
                " (R-075/R-013)"
            )
        if member.is_dir():
            continue
        if not member.is_file():
            raise SnapshotMismatchError(
                f"snapshot member {rel_name!r} is not a regular file (R-020)"
            )
        observed[rel_name] = member
    extra = sorted(set(observed) - set(declared))
    if extra:
        raise SnapshotMismatchError(
            f"snapshot carries undeclared file {extra[0]!r} — an extra file"
            " is a mismatch even when every declared file checks (R-075)"
        )
    missing = sorted(set(declared) - set(observed))
    if missing:
        raise SnapshotMismatchError(
            f"snapshot is missing declared file {missing[0]!r} (R-075)"
        )
    for rel_name in sorted(declared):
        meta = declared[rel_name]
        member = observed[rel_name]
        actual_size = os.lstat(member).st_size
        if actual_size != meta["size"]:
            # Size is a REAL check, independently of the content hash — a
            # manifest with the correct sha but wrong declared size fails.
            raise SnapshotMismatchError(
                f"snapshot file {rel_name!r} size {actual_size} != declared"
                f" size {meta['size']} (R-075)"
            )
        actual_sha = _sha256_regular_file(
            member,
            error_cls=SnapshotMismatchError,
            label=f"snapshot file {rel_name!r}",
        )
        if actual_sha != meta["sha256"]:
            raise SnapshotMismatchError(
                f"snapshot file {rel_name!r} sha256 {actual_sha} != declared"
                f" {meta['sha256']} (R-075)"
            )
    return None


# ---------------------------------------------------------------------------
# R-077: materialized parity comparator
# ---------------------------------------------------------------------------


def _values_equal(expected: Any, observed: Any) -> bool:
    """Exact parsed-JSON-value equality AT THE SAME JSON TYPE (amended R-077).

    ``True == 1`` / ``False == 0`` are Python-equal — a bool on either side
    matches ONLY a bool of the same value (seed catalog: bool laundering).
    ``2249 == 2249.0`` is Python-equal — an int drifting to float (or vice
    versa) is a serialization-identity change and MUST fail: no cross-type
    numeric laundering. Non-finite floats never compare equal. Lists compare
    element-wise under the same rules.
    """
    if isinstance(expected, bool) or isinstance(observed, bool):
        return (
            isinstance(expected, bool)
            and isinstance(observed, bool)
            and expected == observed
        )
    if isinstance(expected, (int, float)) and isinstance(observed, (int, float)):
        if type(expected) is not type(observed):
            return False
        if isinstance(expected, float) and (
            not math.isfinite(expected) or not math.isfinite(observed)
        ):
            return False
        return expected == observed
    if isinstance(expected, list) or isinstance(observed, list):
        if not (isinstance(expected, list) and isinstance(observed, list)):
            return False
        if len(expected) != len(observed):
            return False
        return all(_values_equal(e, o) for e, o in zip(expected, observed))
    return type(expected) is type(observed) and expected == observed


def _validate_parity_anchor(anchor: Any) -> dict[str, Any]:
    """Fail closed on a malformed ANCHOR: a truncated allowlist must never
    produce a vacuous comparison. (The REGENERATED side, by contrast, never
    raises — absences become failure rows.)"""
    anchor = _as_dict(anchor)
    required = (
        "nonrandom_cells",
        "policies",
        "point_fields",
        "ci_fields",
        "expected",
        "identity_fields",
        "random_k",
    )
    missing = sorted(k for k in required if k not in anchor)
    if missing:
        raise ParityAnchorError(
            f"parity anchor missing required field(s) {missing} (R-077)"
        )
    for list_field in ("nonrandom_cells", "policies", "point_fields", "ci_fields"):
        value = anchor[list_field]
        if not isinstance(value, list) or not value or not all(
            isinstance(v, str) and v for v in value
        ):
            raise ParityAnchorError(
                f"parity anchor {list_field} must be a non-empty list of"
                " strings — an empty allowlist axis is a vacuously-empty"
                " authoritative set (R-077)"
            )
    # Amended R-077 (F-3): the allowlist cardinalities are HARD-PINNED —
    # exactly 8 nonrandom cells x 2 policies x 10 point fields x 2 CI
    # fields. A truncated (or padded, or duplicate-carrying) anchor must
    # refuse: a comparison over fewer than the full 194-field allowlist can
    # never produce a vacuous PASS.
    for field_name, expected_count in PARITY_ANCHOR_CARDINALITIES:
        entries = anchor[field_name]
        if len(entries) != expected_count:
            raise ParityAnchorError(
                f"parity anchor {field_name} has {len(entries)} entries —"
                f" the frozen allowlist pins exactly {expected_count};"
                " a truncated anchor never yields a vacuous PASS"
                " (amended R-077/F-3)"
            )
        if len(set(entries)) != len(entries):
            raise ParityAnchorError(
                f"parity anchor {field_name} carries duplicate entries —"
                " duplicates shrink the effective allowlist under a"
                " full-looking count (amended R-077/F-3)"
            )
    if not isinstance(anchor["expected"], dict):
        raise ParityAnchorError("parity anchor expected block must be an object")
    identity = anchor["identity_fields"]
    if not isinstance(identity, dict) or any(
        f not in identity for f in IDENTITY_FIELDS
    ):
        raise ParityAnchorError(
            f"parity anchor identity_fields must carry {list(IDENTITY_FIELDS)}"
            " (R-077)"
        )
    rk = anchor["random_k"]
    if not isinstance(rk, dict) or not isinstance(rk.get("cells"), list):
        raise ParityAnchorError(
            "parity anchor random_k block must carry the informational cell"
            " list (R-077)"
        )
    return anchor


def compare_parity(
    anchor: dict[str, Any], regenerated_export: dict[str, Any]
) -> dict[str, Any]:
    """R-077 materialized parity comparison: anchor allowlist vs a
    producer-payload-shaped regenerated export.

    Checks all 160 nonrandom point fields + all 32 nonrandom CI arrays
    (every element) + the 2 population identity fields (n_eval, n_fit) —
    ``checked == 194`` against the frozen anchor. ANY mismatch, including
    any CI-array element, is a blocking FAIL. Missing cells/policies/fields
    become failure rows, never exceptions (guarded builder). The two
    Random-K cells are exempt from historical parity and reported
    informationally.
    """
    anchor = _validate_parity_anchor(anchor)
    regenerated = _as_dict(regenerated_export)
    metadata = _as_dict(regenerated.get("metadata"))
    results = _as_dict(regenerated.get("results"))

    failures: list[dict[str, Any]] = []
    checked = 0

    # Identity fields (failure rows carry cell=None, policy=None).
    identity = _as_dict(anchor["identity_fields"])
    for field in IDENTITY_FIELDS:
        checked += 1
        expected_value = identity.get(field)
        observed_value = metadata.get(field, _MISSING)
        if observed_value is _MISSING or not _values_equal(
            expected_value, observed_value
        ):
            failures.append(
                {
                    "cell": None,
                    "policy": None,
                    "field": field,
                    "expected": expected_value,
                    "observed": (
                        None if observed_value is _MISSING else observed_value
                    ),
                }
            )

    expected_block = _as_dict(anchor["expected"])
    point_fields = list(anchor["point_fields"])
    ci_fields = list(anchor["ci_fields"])
    for cell in anchor["nonrandom_cells"]:
        expected_policies = _as_dict(expected_block.get(cell))
        observed_policies = _as_dict(results.get(cell))
        for policy in anchor["policies"]:
            expected_values = _as_dict(expected_policies.get(policy))
            observed_values = _as_dict(observed_policies.get(policy))
            for field in point_fields + ci_fields:
                checked += 1
                expected_value = expected_values.get(field, _MISSING)
                if expected_value is _MISSING:
                    raise ParityAnchorError(
                        f"parity anchor expected[{cell!r}][{policy!r}] is"
                        f" missing allowlisted field {field!r} — the anchor"
                        " allowlist must be complete (R-077)"
                    )
                observed_value = observed_values.get(field, _MISSING)
                if observed_value is _MISSING or not _values_equal(
                    expected_value, observed_value
                ):
                    failures.append(
                        {
                            "cell": cell,
                            "policy": policy,
                            "field": field,
                            "expected": expected_value,
                            "observed": (
                                None
                                if observed_value is _MISSING
                                else observed_value
                            ),
                        }
                    )

    # Random-K cells (amended R-077, operational-rejection repair): the
    # STRUCTURE is blocking — both krandom cells must be present with the
    # full point + CI field set (a missing cell/field is a structural
    # failure row; a whole missing cell reports field "<cell>") — while the
    # numeric VALUES stay exempt (never compared, informational only).
    # Structural rows do NOT increment ``checked``: PASS still implies
    # exactly the 194 blocking value comparisons.
    rk = _as_dict(anchor["random_k"])
    for cell in rk.get("cells", []):
        observed_cell = results.get(cell)
        if not isinstance(observed_cell, dict):
            failures.append(
                {
                    "cell": cell,
                    "policy": None,
                    "field": "<cell>",
                    "expected": "present",
                    "observed": None,
                }
            )
            continue
        for policy in anchor["policies"]:
            observed_values = observed_cell.get(policy)
            if not isinstance(observed_values, dict):
                observed_values = {}
            for field in point_fields + ci_fields:
                if field not in observed_values:
                    failures.append(
                        {
                            "cell": cell,
                            "policy": policy,
                            "field": field,
                            "expected": "present",
                            "observed": None,
                        }
                    )

    # Random-K VALUES: NEVER blocking; informational report only (R-077).
    archived = _as_dict(rk.get("informational_archived_values"))
    divergences: list[dict[str, Any]] = []
    rk_compared = 0
    for cell in rk.get("cells", []):
        archived_policies = _as_dict(archived.get(cell))
        observed_policies = _as_dict(results.get(cell))
        for policy in anchor["policies"]:
            archived_values = _as_dict(archived_policies.get(policy))
            observed_values = _as_dict(observed_policies.get(policy))
            for field in point_fields + ci_fields:
                if field not in archived_values:
                    continue
                rk_compared += 1
                archived_value = archived_values.get(field)
                observed_value = observed_values.get(field, _MISSING)
                if observed_value is _MISSING or not _values_equal(
                    archived_value, observed_value
                ):
                    divergences.append(
                        {
                            "cell": cell,
                            "policy": policy,
                            "field": field,
                            "archived": archived_value,
                            "regenerated": (
                                None
                                if observed_value is _MISSING
                                else observed_value
                            ),
                        }
                    )
    random_k_informational = {
        "cells": [str(c) for c in rk.get("cells", [])],
        "exempt_from_historical_parity": True,
        "archived_rng_pinned": rk.get("archived_rng_pinned"),
        "fresh_rng_pinned": rk.get("fresh_rng_pinned"),
        "compared": rk_compared,
        "divergences": divergences,
    }

    # Amended R-077 (F-3): PASS additionally requires the full 194-field
    # allowlist to have been checked — belt-and-braces behind the anchor
    # cardinality pins; a sub-allowlist comparison can never PASS.
    verdict = (
        "PASS"
        if not failures and checked == EXPECTED_PARITY_CHECKED
        else "FAIL"
    )
    return {
        "verdict": verdict,
        "checked": checked,
        "failures": failures,
        "random_k_informational": random_k_informational,
    }


# ---------------------------------------------------------------------------
# R-079/R-082: PRE_RUN_READY certificate (pure core + thin generator)
# ---------------------------------------------------------------------------


def _is_git_object_id(value: Any) -> bool:
    """R-079 (SPEC_ISSUE-1 adjudication 2026-08-22): repo commit and tree
    bind the repository's NATIVE git object ids — 40-hex SHA-1 and 64-hex
    SHA-256 object formats are both admissible, lowercase hex, fixed
    length."""
    return (
        isinstance(value, str)
        and len(value) in (40, 64)
        and all(char in "0123456789abcdef" for char in value)
    )


def untracked_disclosure_failures(value: Any) -> list[str]:
    """Validate the exact portable representation of untracked repo paths."""
    if not isinstance(value, list):
        return ["must be a list"]

    failures: list[str] = []
    all_strings = True
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item:
            failures.append(f"entry {index} must be a non-empty string")
            all_strings = False
            continue
        try:
            posix_path = PurePosixPath(item)
            windows_path = PureWindowsPath(item)
        except (OSError, TypeError, ValueError) as exc:
            failures.append(
                f"entry {index} is not a valid repo-relative path"
                f" ({exc.__class__.__name__})"
            )
            continue
        if (
            "\x00" in item
            or "\\" in item
            or posix_path.is_absolute()
            or windows_path.is_absolute()
            or bool(windows_path.drive)
            or ".." in posix_path.parts
            or item in (".", "..")
            or posix_path.as_posix() != item
        ):
            failures.append(
                f"entry {index} {item!r} must be a normalized forward-slash"
                " repo-relative path without dot segments"
            )

    if all_strings:
        if value != sorted(value):
            failures.append("entries must be sorted")
        if len(value) != len(set(value)):
            failures.append("entries must be unique")
    return failures


def parse_untracked_porcelain_v1_z(raw_status: Any) -> list[str]:
    """Parse machine-safe full-status output into canonical untracked paths."""
    if isinstance(raw_status, bytes):
        try:
            status = raw_status.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise schema.ColmAimsError(
                "git untracked status is not valid UTF-8"
            ) from exc
    elif isinstance(raw_status, str):
        status = raw_status
    else:
        raise schema.ColmAimsError(
            "git untracked status must be text or bytes"
        )
    if not status:
        return []
    if not status.endswith("\x00"):
        raise schema.ColmAimsError(
            "git untracked status is not NUL-terminated porcelain-v1 output"
        )

    paths: list[str] = []
    for record in status[:-1].split("\x00"):
        if not record.startswith("?? "):
            raise schema.ColmAimsError(
                "git full status contains a non-untracked or malformed"
                f" record {record[:80]!r}"
            )
        paths.append(record[3:])
    paths.sort()
    failures = untracked_disclosure_failures(paths)
    if failures:
        raise schema.ColmAimsError(
            f"git untracked paths are not canonical: {failures!r}"
        )
    return paths


def _check_repo(repo: Any, fail: Any) -> None:
    if not isinstance(repo, dict):
        fail("repo: component must be an object")
        return
    if repo.get("dirty") is not False:
        fail(
            "repo: dirty must be exactly False (clean-state proof); found"
            f" {repo.get('dirty')!r}"
        )
    if not _is_git_object_id(repo.get("commit")):
        fail(
            "repo: commit is not a native git object id (40- or 64-hex"
            " lowercase)"
        )
    if not _is_git_object_id(repo.get("tree_sha256")):
        fail(
            "repo: tree_sha256 is not a native git object id (40- or 64-hex"
            " lowercase)"
        )
    root_realpath = repo.get(R082_REPO_ROOT_FIELD)
    try:
        root_is_absolute = (
            isinstance(root_realpath, str)
            and bool(root_realpath)
            and Path(root_realpath).is_absolute()
        )
    except (OSError, ValueError):
        root_is_absolute = False
    if not root_is_absolute:
        fail(
            f"repo: {R082_REPO_ROOT_FIELD} must be a non-empty absolute"
            " path (R-082 external staging binding)"
        )
    if "untracked_disclosure" not in repo:
        fail("repo: required field 'untracked_disclosure' missing")
    else:
        for failure in untracked_disclosure_failures(
            repo.get("untracked_disclosure")
        ):
            fail(f"repo: untracked_disclosure {failure}")


def _check_content_hashes(
    hashes: Any, fail: Any, *, repo: Any = None
) -> None:
    root = _as_dict(repo).get(R082_REPO_ROOT_FIELD)
    for failure in content_hash_failures(root, hashes):
        fail(f"content_hashes: {failure}")


def _check_eligibility(eligibility: Any, fail: Any) -> None:
    if not isinstance(eligibility, dict):
        fail("eligibility: component must be an object")
        return
    if not schema.is_sha256_hex(eligibility.get("digest")):
        fail("eligibility: digest is not a sha256 hex digest")
    elif eligibility.get("digest") != ELIGIBILITY_KEYSET_SHA256:
        fail(
            "eligibility: digest does not equal the canonical keyset pin"
            f" {ELIGIBILITY_KEYSET_SHA256}"
        )
    if not schema.is_sha256_hex(eligibility.get("horizon_map_sha256")):
        fail("eligibility: horizon_map_sha256 is not a sha256 hex digest")
    elif (
        eligibility.get("horizon_map_sha256")
        != ELIGIBILITY_HORIZON_MAP_SHA256
    ):
        fail(
            "eligibility: horizon_map_sha256 does not equal the canonical"
            f" pin {ELIGIBILITY_HORIZON_MAP_SHA256}"
        )
    artifact_path = eligibility.get("artifact_path")
    if not isinstance(artifact_path, str) or not artifact_path:
        fail("eligibility: artifact_path must be a non-empty absolute path")
    else:
        try:
            if not Path(artifact_path).is_absolute():
                fail(
                    "eligibility: artifact_path must be a non-empty"
                    " absolute path"
                )
        except (OSError, TypeError, ValueError):
            fail("eligibility: artifact_path is not a valid absolute path")
    if not schema.is_sha256_hex(eligibility.get("artifact_sha256")):
        fail("eligibility: artifact_sha256 is not a sha256 hex digest")
    elif eligibility.get("artifact_sha256") != ELIGIBILITY_ARTIFACT_SHA256:
        fail(
            "eligibility: artifact_sha256 does not equal the canonical raw"
            f" artifact pin {ELIGIBILITY_ARTIFACT_SHA256}"
        )
    if not schema.is_sha256_hex(eligibility.get("test_dataset_sha256")):
        fail("eligibility: test_dataset_sha256 is not a sha256 hex digest")
    elif (
        eligibility.get("test_dataset_sha256")
        != ELIGIBILITY_TEST_DATASET_SHA256
    ):
        fail(
            "eligibility: test_dataset_sha256 does not equal the canonical"
            f" test split pin {ELIGIBILITY_TEST_DATASET_SHA256}"
        )


def _check_snapshots(snapshots: Any, fail: Any) -> None:
    if not isinstance(snapshots, dict):
        fail("snapshots: component must be an object")
        return
    expected_keys = SNAPSHOT_ROLES | SNAPSHOT_COMPONENT_METADATA_KEYS
    if set(snapshots) != expected_keys:
        fail(
            f"snapshots: keys must be exactly {sorted(expected_keys)};"
            f" found {sorted(map(str, snapshots))}"
        )
    artifact_path = snapshots.get("artifact_path")
    if not isinstance(artifact_path, str) or not artifact_path:
        fail("snapshots: artifact_path must be a non-empty absolute path")
    else:
        try:
            if not Path(artifact_path).is_absolute():
                fail(
                    "snapshots: artifact_path must be a non-empty"
                    " absolute path"
                )
        except (OSError, TypeError, ValueError):
            fail("snapshots: artifact_path is not a valid absolute path")
    if not schema.is_sha256_hex(snapshots.get("artifact_sha256")):
        fail("snapshots: artifact_sha256 is not a sha256 hex digest")
    elif snapshots.get("artifact_sha256") != SNAPSHOT_MANIFEST_SHA256:
        fail(
            "snapshots: artifact_sha256 does not equal the canonical raw"
            f" manifest pin {SNAPSHOT_MANIFEST_SHA256}"
        )
    for role in sorted(SNAPSHOT_ROLES):
        entry = snapshots.get(role)
        if not isinstance(entry, dict):
            fail(f"snapshots: {role} snapshot entry missing or malformed")
            continue
        if entry.get("verified") is not True:
            fail(
                f"snapshots: {role} snapshot verified must be exactly True;"
                f" found {entry.get('verified')!r}"
            )
        if not isinstance(entry.get("model_name"), str) or not entry.get(
            "model_name"
        ):
            fail(f"snapshots: {role} snapshot model_name missing")
        if not _is_hf_revision(entry.get("hf_revision")):
            fail(f"snapshots: {role} snapshot hf_revision is not a valid pin")
        expected_identity = EXPECTED_SNAPSHOT_IDENTITIES[role]
        for field in ("model_name", "hf_revision"):
            if entry.get(field) != expected_identity[field]:
                fail(
                    f"snapshots: {role} {field} {entry.get(field)!r} does"
                    " not match the canonical loaded-manifest identity"
                    f" {expected_identity[field]!r}"
                )


def _check_offline_flags(flags: Any, fail: Any) -> None:
    if flags != list(REQUIRED_OFFLINE_FLAGS):
        fail(
            "offline_flags: must be exactly"
            f" {list(REQUIRED_OFFLINE_FLAGS)}; found {flags!r}"
        )


def _check_staged_inputs(staged: Any, fail: Any) -> None:
    if not isinstance(staged, list):
        fail("staged_inputs: component must be a list")
        return
    if not staged:
        fail(
            "staged_inputs: empty staged-input set — a vacuously-empty"
            " authoritative set is a defect"
        )
        return
    for index, entry in enumerate(staged):
        if not isinstance(entry, dict):
            fail(f"staged_inputs: entry [{index}] must be an object")
            continue
        label = entry.get("label")
        name = label if isinstance(label, str) and label else f"[{index}]"
        expected = entry.get("expected_sha256")
        observed = entry.get("observed_sha256")
        if not schema.is_sha256_hex(expected):
            fail(f"staged_inputs: {name}: expected_sha256 is not a sha256 digest")
        if not schema.is_sha256_hex(observed):
            fail(
                f"staged_inputs: {name}: observed_sha256 missing or not a"
                " sha256 digest — a missing observation is never a pass"
            )
        elif schema.is_sha256_hex(expected) and observed != expected:
            fail(
                f"staged_inputs: {name}: observed sha256 {observed} !="
                f" expected {expected}"
            )
        if not isinstance(entry.get("path"), str) or not entry.get("path"):
            fail(f"staged_inputs: {name}: path missing")


def _check_suite_receipts(receipts: Any, fail: Any, repo: Any = None) -> None:
    repo = repo if isinstance(repo, dict) else {}
    if not isinstance(receipts, dict):
        fail("suite_receipts: component must be an object")
        return
    for name in SUITE_RECEIPT_NAMES:
        receipt = receipts.get(name)
        if not isinstance(receipt, dict):
            fail(f"suite_receipts: {name} receipt missing or malformed")
            continue
        # R-070/R-082: the full machine-readable receipt binding is REQUIRED
        # — a receipt missing any field is a failing suite_receipts
        # component.
        for field in R070_RECEIPT_FIELDS:
            if field not in receipt:
                fail(
                    f"suite_receipts: {name} receipt is missing the"
                    f" R-070/R-082 field {field!r}"
                )
        exit_code = receipt.get("exit_code")
        # Bool-guard: False == 0 in Python; only the exact int 0 is success.
        if type(exit_code) is not int or exit_code != 0:
            fail(
                f"suite_receipts: {name} exit_code must be exactly int 0;"
                f" found {exit_code!r}"
            )
        command = receipt.get("command")
        if (
            not isinstance(command, list)
            or not command
            or not all(isinstance(part, str) for part in command)
        ):
            fail(
                f"suite_receipts: {name} command must be a non-empty argv"
                " list of strings"
            )
        if "environment_lock_sha256" in receipt and not schema.is_sha256_hex(
            receipt["environment_lock_sha256"]
        ):
            fail(
                f"suite_receipts: {name} environment_lock_sha256 is not a"
                " sha256 hex digest (R-070: a HASH, not a metadata object)"
            )
        if "workflow_sha256" in receipt and not schema.is_sha256_hex(
            receipt["workflow_sha256"]
        ):
            fail(
                f"suite_receipts: {name} workflow_sha256 is not a sha256"
                " hex digest (R-070)"
            )
        if "interpreter_realpath" in receipt and (
            not isinstance(receipt["interpreter_realpath"], str)
            or not receipt["interpreter_realpath"]
        ):
            fail(
                f"suite_receipts: {name} interpreter_realpath must be a"
                " non-empty string (R-070)"
            )
        for digest_field in ("junit_sha256", "transcript_sha256"):
            if digest_field in receipt and not schema.is_sha256_hex(
                receipt[digest_field]
            ):
                fail(
                    f"suite_receipts: {name} {digest_field} is not a"
                    " sha256 hex digest (R-070)"
                )
        counts = receipt.get("counts")
        if "counts" in receipt and not isinstance(counts, dict):
            fail(
                f"suite_receipts: {name} counts must be a machine-readable"
                " object (R-070)"
            )
        elif isinstance(counts, dict):
            count_fields = {"tests", "failures", "errors", "skipped"}
            if set(counts) != count_fields:
                fail(
                    f"suite_receipts: {name} counts keys must be exactly"
                    f" {sorted(count_fields)}; found"
                    f" {sorted(map(str, counts))} (R-070/R-082)"
                )
            for tally in sorted(count_fields):
                value = counts.get(tally, _MISSING)
                if not schema.is_real_int(value) or value < 0:
                    fail(
                        f"suite_receipts: {name} counts.{tally} must be a"
                        f" real nonnegative int; found {value!r}"
                        " (R-070/R-082)"
                    )
            if counts.get("failures") != 0:
                fail(
                    f"suite_receipts: {name} counts.failures must be"
                    f" exactly int 0; found {counts.get('failures')!r}"
                    " (R-082)"
                )
            if counts.get("errors") != 0:
                fail(
                    f"suite_receipts: {name} counts.errors must be exactly"
                    f" int 0; found {counts.get('errors')!r} (R-082)"
                )
            tests = counts.get("tests")
            skipped = counts.get("skipped")
            if (
                schema.is_real_int(tests)
                and schema.is_real_int(skipped)
                and (tests <= 0 or tests <= skipped)
            ):
                fail(
                    f"suite_receipts: {name} counts.tests must be positive"
                    " and greater than counts.skipped; found"
                    f" tests={tests!r}, skipped={skipped!r} (R-070)"
                )
        if "skip_identities" in receipt:
            skips = receipt["skip_identities"]
            if (
                not isinstance(skips, list)
                or not all(isinstance(item, str) and item for item in skips)
                or len(set(skips)) != len(skips)
            ):
                fail(
                    f"suite_receipts: {name} skip_identities must be a"
                    " duplicate-free list of non-empty strings (R-070)"
                )
            elif isinstance(counts, dict) and schema.is_real_int(
                counts.get("skipped")
            ) and len(skips) != counts.get("skipped"):
                fail(
                    f"suite_receipts: {name} skip_identities length"
                    f" {len(skips)} != counts.skipped"
                    f" {counts.get('skipped')} (R-070)"
                )
        # R-082 head bindings: receipts must come from EXECUTING the suites
        # at the certified head — commit/tree must EQUAL the runner-sourced
        # repo component and dirty must be identically False. A stale
        # receipt ingested from an earlier head mismatches here.
        if "commit" in receipt and receipt["commit"] != repo.get("commit"):
            fail(
                f"suite_receipts: {name} receipt commit"
                f" {receipt['commit']!r} != certified repo commit"
                f" {repo.get('commit')!r} (R-082 head binding)"
            )
        if "tree_sha256" in receipt and receipt["tree_sha256"] != repo.get(
            "tree_sha256"
        ):
            fail(
                f"suite_receipts: {name} receipt tree_sha256"
                f" {receipt['tree_sha256']!r} != certified repo tree"
                f" {repo.get('tree_sha256')!r} (R-082 head binding)"
            )
        if "dirty" in receipt and receipt["dirty"] is not False:
            fail(
                f"suite_receipts: {name} receipt dirty must be identically"
                f" False; found {receipt['dirty']!r} (R-082 head binding)"
            )


def _check_parity(parity: Any, fail: Any) -> None:
    if not isinstance(parity, dict):
        fail("parity: component must be an object")
        return
    identity = parity.get("comparator_identity")
    if not isinstance(identity, str) or not identity:
        fail("parity: comparator_identity must be a non-empty string")
    elif identity != PARITY_COMPARATOR_IDENTITY:
        fail(
            "parity: comparator_identity must be exactly"
            f" {PARITY_COMPARATOR_IDENTITY!r}"
        )
    if not schema.is_sha256_hex(parity.get("anchor_sha256")):
        fail("parity: anchor_sha256 is not a sha256 hex digest")
    elif parity.get("anchor_sha256") != PARITY_ANCHOR_SHA256:
        fail(
            "parity: anchor_sha256 does not equal the canonical raw anchor"
            f" pin {PARITY_ANCHOR_SHA256}"
        )
    artifact_path = parity.get("artifact_path")
    if not isinstance(artifact_path, str) or not artifact_path:
        fail("parity: artifact_path must be a non-empty absolute path")
    else:
        try:
            if not Path(artifact_path).is_absolute():
                fail("parity: artifact_path must be a non-empty absolute path")
        except (OSError, TypeError, ValueError):
            fail("parity: artifact_path is not a valid absolute path")
    if (
        parity.get("source_export_a_sha256")
        != PARITY_SOURCE_EXPORT_A_SHA256
    ):
        fail(
            "parity: source_export_a_sha256 does not equal the canonical"
            f" Export-A pin {PARITY_SOURCE_EXPORT_A_SHA256}"
        )


def _check_qa012(qa012: Any, fail: Any) -> None:
    if not isinstance(qa012, dict):
        fail("qa012: component must be an object")
        return
    if not schema.is_sha256_hex(qa012.get("manifest_sha256")):
        fail("qa012: manifest_sha256 is not a sha256 hex digest")
    elif qa012.get("manifest_sha256") != QA012_MANIFEST_SHA256:
        fail(
            "qa012: manifest_sha256 does not equal the canonical rev3 raw"
            f" manifest pin {QA012_MANIFEST_SHA256}"
        )
    artifact_path = qa012.get("artifact_path")
    if not isinstance(artifact_path, str) or not artifact_path:
        fail("qa012: artifact_path must be a non-empty absolute path")
    else:
        try:
            if not Path(artifact_path).is_absolute():
                fail("qa012: artifact_path must be a non-empty absolute path")
        except (OSError, TypeError, ValueError):
            fail("qa012: artifact_path is not a valid absolute path")
    if qa012.get("manifest_type") != QA012_MANIFEST_TYPE:
        fail(
            f"qa012: manifest_type must be exactly {QA012_MANIFEST_TYPE!r}"
        )
    if (
        type(qa012.get("revision")) is not int
        or qa012.get("revision") != QA012_MANIFEST_REVISION
    ):
        fail(
            f"qa012: revision must be exactly int {QA012_MANIFEST_REVISION}"
        )
    if qa012.get("conventions") != QA012_CONVENTIONS:
        fail("qa012: conventions do not equal the canonical rev3 conventions")


def _check_environment(env: Any, fail: Any) -> None:
    if not isinstance(env, dict):
        fail("environment: component must be an object")
        return
    for key in CERT_ENVIRONMENT_KEYS:
        if key not in env:
            fail(f"environment: required field {key!r} missing")
    for key in (
        "interpreter_realpath",
        "os",
        "arch",
        "cpu",
        "blas",
        *R081_LAUNCH_PATH_FIELDS,
    ):
        if key in env and (not isinstance(env[key], str) or not env[key]):
            fail(f"environment: {key} must be a non-empty string")
    if "thread_settings" in env and env["thread_settings"] != (
        PHASE4_THREAD_SETTINGS
    ):
        fail(
            "environment: thread_settings must equal the exact Phase-4 pin"
            f" map {PHASE4_THREAD_SETTINGS!r}"
        )
    if "environment_lock_sha256" in env and not schema.is_sha256_hex(
        env["environment_lock_sha256"]
    ):
        fail("environment: environment_lock_sha256 is not a sha256 digest")
    if "command" in env and (
        not isinstance(env["command"], list)
        or not env["command"]
        or not all(isinstance(part, str) for part in env["command"])
    ):
        fail("environment: command must be a non-empty list of strings")
    if "seeds" in env and env["seeds"] != [1]:
        fail("environment: seeds must be exactly [1]")
    if "pythonhashseed" in env and env["pythonhashseed"] != "0":
        fail("environment: pythonhashseed must be exactly '0'")
    if "archived_rng_pinned" in env and env["archived_rng_pinned"] is not False:
        fail(
            "environment: archived_rng_pinned must be exactly False"
            " (R-077 flags)"
        )
    if "fresh_rng_pinned" in env and env["fresh_rng_pinned"] is not True:
        fail("environment: fresh_rng_pinned must be exactly True (R-077 flags)")


def _check_external_staging(components: dict[str, Any], fail: Any) -> None:
    """Cross-component certificate gate for the R-082 ledger-burn trap."""
    repo = _as_dict(components.get("repo"))
    environment = _as_dict(components.get("environment"))
    for failure in external_staging_failures(
        repo.get(R082_REPO_ROOT_FIELD),
        components.get("staged_inputs"),
        environment.get("command"),
    ):
        fail(f"R-082 external_staging: {failure}")


def _check_staged_coverage(components: dict[str, Any], fail: Any) -> None:
    """Cross-bind the exact command to the six staged certificate entries."""
    repo = _as_dict(components.get("repo"))
    environment = _as_dict(components.get("environment"))
    for failure in staged_coverage_failures(
        repo.get(R082_REPO_ROOT_FIELD),
        components.get("staged_inputs"),
        components.get("eligibility"),
        environment.get("command"),
    ):
        fail(f"R-082 staged_coverage: {failure}")


def _check_launch_paths(components: dict[str, Any], fail: Any) -> None:
    """Cross-bind the certified launch workspace to the operational root."""
    repo = _as_dict(components.get("repo"))
    for failure in launch_path_failures(
        repo.get(R082_REPO_ROOT_FIELD),
        components.get("environment"),
    ):
        fail(f"R-081 launch_paths: {failure}")


def _check_receipt_environment(
    components: dict[str, Any], fail: Any
) -> None:
    """Cross-bind suite evidence to the certificate runtime and lock."""
    repo = _as_dict(components.get("repo"))
    for failure in receipt_environment_failures(
        repo.get(R082_REPO_ROOT_FIELD),
        components.get("suite_receipts"),
        components.get("environment"),
    ):
        fail(f"R-070 receipt_environment: {failure}")


def _check_phase4_command(components: dict[str, Any], fail: Any) -> None:
    """Require the explicit canonical producer run shape pre-activation."""
    repo = _as_dict(components.get("repo"))
    environment = _as_dict(components.get("environment"))
    for failure in phase4_command_failures(environment.get("command")):
        fail(f"R-081 phase4_command: {failure}")
    for failure in command_environment_failures(
        repo.get(R082_REPO_ROOT_FIELD), environment
    ):
        fail(f"R-081 phase4_command: {failure}")


def _check_snapshot_manifest_binding(
    components: dict[str, Any], fail: Any
) -> None:
    """Cross-bind raw manifest identity to the producer command path."""
    repo = _as_dict(components.get("repo"))
    environment = _as_dict(components.get("environment"))
    for failure in snapshot_manifest_failures(
        repo.get(R082_REPO_ROOT_FIELD),
        components.get("snapshots"),
        environment.get("command"),
    ):
        fail(f"R-075 snapshot_manifest: {failure}")


def _check_canonical_artifact_paths(
    components: dict[str, Any], fail: Any
) -> None:
    """Bind parity and QA evidence to their tracked canonical repo files."""
    repo = _as_dict(components.get("repo"))
    root = repo.get(R082_REPO_ROOT_FIELD)
    for failure in canonical_artifact_path_failures(
        root,
        components.get("parity"),
        artifact_relpath=PARITY_ANCHOR_RELPATH,
        component_name="parity",
    ):
        fail(f"R-077 parity_anchor: {failure}")
    for failure in canonical_artifact_path_failures(
        root,
        components.get("qa012"),
        artifact_relpath=QA012_MANIFEST_RELPATH,
        component_name="qa012",
    ):
        fail(f"R-072 qa012_manifest: {failure}")


_COMPONENT_CHECKERS = {
    "repo": _check_repo,
    "content_hashes": _check_content_hashes,
    "eligibility": _check_eligibility,
    "snapshots": _check_snapshots,
    "offline_flags": _check_offline_flags,
    "staged_inputs": _check_staged_inputs,
    "suite_receipts": _check_suite_receipts,
    "parity": _check_parity,
    "qa012": _check_qa012,
    "environment": _check_environment,
}


def assemble_certificate(components: dict[str, Any]) -> dict[str, Any]:
    """Pure core of the PRE_RUN_READY generator (R-079).

    ``ready`` is True ONLY when every check passes; ANY defect yields
    ``ready: False`` with EVERY failing component named in
    ``failing_checks`` — never a partial pass, never an exception.
    """
    failing_checks: list[str] = []
    fail = failing_checks.append
    if not isinstance(components, dict):
        return {
            "schema_version": CERT_SCHEMA_VERSION,
            "ready": False,
            "failing_checks": [
                f"{key}: required component missing"
                for key in CERT_COMPONENT_KEYS
            ],
            "components": {},
        }
    for key in CERT_COMPONENT_KEYS:
        if key not in components:
            fail(f"{key}: required component missing")
            continue
        checker = _COMPONENT_CHECKERS[key]
        try:
            if key == "suite_receipts":
                # R-082: the receipt head bindings compare against the
                # runner-sourced repo component (cross-component check).
                checker(
                    components[key], fail, repo=components.get("repo")
                )
            elif key == "content_hashes":
                # R-079: audited source digests are meaningful only when
                # bound to their canonical files under the operational root.
                checker(
                    components[key], fail, repo=components.get("repo")
                )
            else:
                checker(components[key], fail)
        except Exception as exc:  # noqa: BLE001 - the never-raise contract
            fail(
                f"{key}: check evaluation failed"
                f" ({exc.__class__.__name__}) — fail closed"
            )
    try:
        _check_external_staging(components, fail)
    except Exception as exc:  # noqa: BLE001 - the never-raise contract
        fail(
            "R-082 external_staging: check evaluation failed"
            f" ({exc.__class__.__name__}) — fail closed"
        )
    try:
        _check_staged_coverage(components, fail)
    except Exception as exc:  # noqa: BLE001 - the never-raise contract
        fail(
            "R-082 staged_coverage: check evaluation failed"
            f" ({exc.__class__.__name__}) — fail closed"
        )
    try:
        _check_launch_paths(components, fail)
    except Exception as exc:  # noqa: BLE001 - the never-raise contract
        fail(
            "R-081 launch_paths: check evaluation failed"
            f" ({exc.__class__.__name__}) — fail closed"
        )
    try:
        _check_receipt_environment(components, fail)
    except Exception as exc:  # noqa: BLE001 - the never-raise contract
        fail(
            "R-070 receipt_environment: check evaluation failed"
            f" ({exc.__class__.__name__}) — fail closed"
        )
    try:
        _check_phase4_command(components, fail)
    except Exception as exc:  # noqa: BLE001 - the never-raise contract
        fail(
            "R-081 phase4_command: check evaluation failed"
            f" ({exc.__class__.__name__}) — fail closed"
        )
    try:
        _check_snapshot_manifest_binding(components, fail)
    except Exception as exc:  # noqa: BLE001 - the never-raise contract
        fail(
            "R-075 snapshot_manifest: check evaluation failed"
            f" ({exc.__class__.__name__}) — fail closed"
        )
    try:
        _check_canonical_artifact_paths(components, fail)
    except Exception as exc:  # noqa: BLE001 - the never-raise contract
        fail(
            "canonical_artifact_paths: check evaluation failed"
            f" ({exc.__class__.__name__}) — fail closed"
        )
    return {
        "schema_version": CERT_SCHEMA_VERSION,
        "ready": not failing_checks,
        "failing_checks": failing_checks,
        "components": components,
    }


PARITY_COMPARATOR_IDENTITY = (
    "reproducibility.colm_aims_2026.phase4.compare_parity"
)
CERT_CONFIG_KEYS = (
    "repo_root",
    "eligibility_path",
    "snapshot_manifest_path",
    "snapshot_dirs",
    "parity_anchor_path",
    "qa012_manifest_path",
    "staged_plan",
    "suite_receipt_paths",
    "content_hash_paths",
    "environment",
    "offline_flags",
)


def _default_command_runner(repo_root: Path):
    """Subprocess-backed ``run(cmd) -> stdout`` for production gathering."""

    def run(cmd: list[str]) -> str:
        completed = subprocess.run(
            [str(part) for part in cmd],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=True,
        )
        return completed.stdout

    return run


def _recompute_file_sha256(path: Any) -> str | None:
    """Record-not-raise recompute: None (a failing component under
    ``assemble_certificate``) when the file is missing/irregular."""
    try:
        return _sha256_regular_file(
            Path(path), error_cls=schema.ColmAimsError, label="certificate input"
        )
    except schema.ColmAimsError:
        return None


def _gather_content_hashes(
    repo_root: Path, configured_paths: Any
) -> dict[str, dict[str, str | None]]:
    """Refuse caller-selected audited sources, then hash canonical bytes."""
    if not isinstance(configured_paths, dict):
        raise schema.ColmAimsError(
            "content_hash_paths must be an object with the exact canonical"
            " audited-source keys"
        )
    expected_keys = set(CONTENT_HASH_RELPATHS)
    observed_keys = set(configured_paths)
    missing = sorted(expected_keys - observed_keys, key=repr)
    unexpected = sorted(observed_keys - expected_keys, key=repr)
    if missing or unexpected:
        raise schema.ColmAimsError(
            "content_hash_paths must equal the exact canonical key set;"
            f" missing={missing!r}, unexpected={unexpected!r}"
        )

    canonical_paths: dict[str, Path] = {}
    for key, relpath in CONTENT_HASH_RELPATHS.items():
        raw_path = configured_paths[key]
        try:
            configured = Path(raw_path)
            if not configured.is_absolute():
                configured = repo_root / configured
            resolved = configured.resolve()
            expected = (repo_root / relpath).resolve()
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            raise schema.ColmAimsError(
                f"content_hash_paths[{key!r}] could not be resolved"
            ) from exc
        if resolved != expected or os.path.normcase(
            str(configured)
        ) != os.path.normcase(str(resolved)):
            raise schema.ColmAimsError(
                f"content_hash_paths[{key!r}] must name the canonical repo"
                f" artifact {str(expected)!r}; found {str(configured)!r}"
            )
        canonical_paths[key] = expected

    return {
        key: {
            "artifact_path": str(path),
            "sha256": _recompute_file_sha256(path),
        }
        for key, path in canonical_paths.items()
    }


def gather_certificate_components(
    config: dict[str, Any], run: Any = None
) -> dict[str, Any]:
    """Gather every PRE_RUN_READY component for ``assemble_certificate``
    (F-4, R-079).

    ``run`` is an injectable command-runner ``run(cmd: list[str]) -> str``
    (stdout; defaults to subprocess in ``config["repo_root"]``). Repo
    identity is RUNNER-SOURCED: dirty from ``git status --porcelain``
    (empty == clean), commit from ``git rev-parse HEAD``, tree from
    ``git rev-parse HEAD^{tree}`` — never a caller assertion. Every staged
    input is REHASHED from file bytes (observed is never copied from the
    expectation); content/anchor/qa012 hashes are recomputed from their
    files; the eligibility digests come through the typed loader; snapshot
    verification failures are RECORDED as ``verified: False`` (assemble
    decides); suite receipts are ingested from the receipt FILES.
    """
    if not isinstance(config, dict):
        raise schema.ColmAimsError(
            "gather_certificate_components config must be an object (F-4)"
        )
    missing = sorted(k for k in CERT_CONFIG_KEYS if k not in config)
    if missing:
        raise schema.ColmAimsError(
            f"gather_certificate_components config missing key(s) {missing}"
            " (F-4)"
        )
    # R-082: this is both the git-command cwd and the child producer cwd.
    # Resolve it once so every relative staged/command path is judged against
    # the same operational root, independent of the orchestrator's own cwd.
    repo_root = Path(config["repo_root"]).resolve()
    if run is None:
        run = _default_command_runner(repo_root)

    # Repo identity: runner-sourced, verbatim native git object ids
    # (40-hex SHA-1 or 64-hex SHA-256 — SPEC_ISSUE-1 adjudication).
    # Clean-state proof is the TRACKED tree (untracked evidence artifacts —
    # the certificate itself, suite receipts, staged inputs, shuttle
    # documents — are unavoidable by construction and are DISCLOSED by
    # list instead; adjudicated amendment 2026-08-22). Code identity is
    # already bound by commit+tree, which untracked files cannot alter.
    tracked_status = run(
        [
            "git",
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=no",
        ]
    )
    dirty = bool(str(tracked_status).strip())
    full_status = run(
        [
            "git",
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=all",
        ]
    )
    untracked = parse_untracked_porcelain_v1_z(full_status)
    commit = str(run(["git", "rev-parse", "HEAD"])).strip()
    tree = str(run(["git", "rev-parse", "HEAD^{tree}"])).strip()
    repo = {
        "commit": commit,
        "tree_sha256": tree,
        "dirty": dirty,
        "untracked_disclosure": untracked,
        R082_REPO_ROOT_FIELD: str(repo_root),
    }

    content_hashes = _gather_content_hashes(
        repo_root, config["content_hash_paths"]
    )

    eligibility_path = Path(config["eligibility_path"])
    if not eligibility_path.is_absolute():
        eligibility_path = repo_root / eligibility_path
    eligibility_path = eligibility_path.resolve()
    eligibility_artifact_sha256 = _recompute_file_sha256(eligibility_path)
    try:
        art = load_pairing_eligibility(eligibility_path)
        eligibility = {
            "digest": art["pairing_population_keyset_sha256"],
            "horizon_map_sha256": art["horizon_map_sha256"],
            "artifact_path": str(eligibility_path),
            "artifact_sha256": eligibility_artifact_sha256,
            "test_dataset_sha256": _as_dict(
                art.get("derived_from")
            ).get("test_dataset_sha256"),
        }
    except schema.ColmAimsError as exc:
        eligibility = {
            "digest": None,
            "horizon_map_sha256": None,
            "artifact_path": str(eligibility_path),
            "artifact_sha256": eligibility_artifact_sha256,
            "test_dataset_sha256": None,
            "error": str(exc),
        }

    snapshot_manifest_path = Path(config["snapshot_manifest_path"])
    if not snapshot_manifest_path.is_absolute():
        snapshot_manifest_path = repo_root / snapshot_manifest_path
    snapshot_manifest_path = snapshot_manifest_path.resolve()
    snapshots: dict[str, Any] = {
        "artifact_path": str(snapshot_manifest_path),
        "artifact_sha256": _recompute_file_sha256(snapshot_manifest_path),
    }
    manifest_roles: dict[str, Any] | None
    manifest_error = None
    try:
        manifest = load_model_snapshot_manifest(snapshot_manifest_path)
        manifest_roles = manifest["roles"]
    except schema.ColmAimsError as exc:
        manifest_roles = None
        manifest_error = str(exc)
    snapshot_dirs = dict(config["snapshot_dirs"])
    for role in sorted(SNAPSHOT_ROLES):
        if manifest_roles is None:
            snapshots[role] = {
                "verified": False,
                "model_name": None,
                "hf_revision": None,
                "error": manifest_error,
            }
            continue
        role_entry = manifest_roles[role]
        record: dict[str, Any] = {
            "verified": False,
            "model_name": role_entry["model_name"],
            "hf_revision": role_entry["hf_revision"],
        }
        snap_dir = snapshot_dirs.get(role)
        if snap_dir is None:
            record["error"] = "no snapshot directory configured for this role"
        else:
            try:
                verify_snapshot_dir(role_entry, Path(snap_dir))
                record["verified"] = True
            except schema.ColmAimsError as exc:
                record["error"] = str(exc)
        snapshots[role] = record

    staged_inputs: list[dict[str, Any]] = []
    for entry in list(config["staged_plan"]):
        entry = _as_dict(entry)
        label = entry.get("label")
        raw_path = entry.get("path", "")
        recorded_path = ""
        try:
            path = Path(raw_path)
            if not path.is_absolute():
                path = repo_root / path
            path = path.resolve()
            recorded_path = str(path)
            observed = _sha256_regular_file(
                path,
                error_cls=StagedInputError,
                label=f"staged input {label!r}",
            )
        except (schema.ColmAimsError, OSError, RuntimeError, TypeError, ValueError):
            observed = None
            if not recorded_path and isinstance(raw_path, (str, Path)):
                recorded_path = str(raw_path)
        staged_inputs.append(
            {
                # R-082: staged inputs live OUTSIDE the repository tree —
                # the certificate records their ABSOLUTE paths (identity is
                # carried by the hash gates, never by location).
                "label": label,
                "path": recorded_path,
                "expected_sha256": entry.get("expected_sha256"),
                "observed_sha256": observed,
            }
        )

    suite_receipts: dict[str, Any] = {}
    receipt_paths = dict(config["suite_receipt_paths"])
    for name in SUITE_RECEIPT_NAMES:
        path = receipt_paths.get(name)
        try:
            data = schema.read_regular_file_bytes(Path(path))
            receipt = schema.parse_json_bytes_strict(data)
        except (schema.ColmAimsError, OSError, TypeError, ValueError) as exc:
            suite_receipts[name] = {"ingest_error": str(exc)}
            continue
        if not isinstance(receipt, dict):
            suite_receipts[name] = {
                "ingest_error": "receipt file is not a JSON object"
            }
            continue
        suite_receipts[name] = receipt

    parity_anchor_path = Path(config["parity_anchor_path"])
    if not parity_anchor_path.is_absolute():
        parity_anchor_path = repo_root / parity_anchor_path
    parity_anchor_path = parity_anchor_path.resolve()
    parity_source_sha256 = None
    try:
        parity_bytes = schema.read_regular_file_bytes(parity_anchor_path)
        parity_obj = schema.parse_json_bytes_strict(parity_bytes)
        if isinstance(parity_obj, dict):
            parity_source_sha256 = _as_dict(parity_obj.get("source")).get(
                "sha256"
            )
    except (
        schema.ColmAimsError,
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
    ):
        pass

    qa012_manifest_path = Path(config["qa012_manifest_path"])
    if not qa012_manifest_path.is_absolute():
        qa012_manifest_path = repo_root / qa012_manifest_path
    qa012_manifest_path = qa012_manifest_path.resolve()
    qa012_obj: dict[str, Any] = {}
    try:
        qa012_bytes = schema.read_regular_file_bytes(qa012_manifest_path)
        parsed_qa012 = schema.parse_json_bytes_strict(qa012_bytes)
        if isinstance(parsed_qa012, dict):
            qa012_obj = parsed_qa012
    except (
        schema.ColmAimsError,
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
    ):
        pass

    return {
        "repo": repo,
        "content_hashes": content_hashes,
        "eligibility": eligibility,
        "snapshots": snapshots,
        "offline_flags": list(config["offline_flags"]),
        "staged_inputs": staged_inputs,
        "suite_receipts": suite_receipts,
        "parity": {
            "comparator_identity": PARITY_COMPARATOR_IDENTITY,
            "artifact_path": str(parity_anchor_path),
            "anchor_sha256": _recompute_file_sha256(parity_anchor_path),
            "source_export_a_sha256": parity_source_sha256,
        },
        "qa012": {
            "artifact_path": str(qa012_manifest_path),
            "manifest_sha256": _recompute_file_sha256(qa012_manifest_path),
            "manifest_type": qa012_obj.get("manifest_type"),
            "revision": qa012_obj.get("revision"),
            "conventions": qa012_obj.get("conventions"),
        },
        "environment": dict(_as_dict(config["environment"])),
    }


def generate_pre_run_ready(
    components: dict[str, Any], out_path: Path
) -> dict[str, Any]:
    """Assemble the PRE_RUN_READY certificate and write it as canonical JSON.

    DECISION: component GATHERING (git state, content hashes, suite
    receipts, environment capture) is the orchestrating runner's job — this
    generator stays a thin, deterministic assemble-serialize-bind step so
    the pure core (``assemble_certificate``) carries every check. Returns
    the certificate plus the written file's SHA-256; the author's
    single-run exception activation references that digest (R-079).
    """
    certificate = assemble_certificate(components)
    payload = schema.encode_json(certificate)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(payload)
    return {
        "certificate": certificate,
        "ready": certificate["ready"],
        "path": str(out_path),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
