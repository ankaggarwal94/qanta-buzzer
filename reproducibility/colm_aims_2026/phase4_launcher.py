"""R-081 single-use launcher: the ONLY sanctioned path to the ONE run.

Executable interface::

    python -m reproducibility.colm_aims_2026.phase4_launcher --config PATH

Validates the PRE_RUN_READY certificate byte-for-byte against the author's
activation digest, re-proves the live repository state against the
certified head, refuses ambient provenance overrides, re-verifies both
model snapshots, consumes the single-use exception via a CREATE-ONCE
ledger, launches the producer EXACTLY once into a fresh quarantine
directory with a pinned environment, and promotes a detached candidate built
from retained comparator-approved bytes only after a mandatory comparator
PASS.

This is an integrity and reproducibility workflow, not a sandbox or hostile-
process containment boundary. R-081 assumes the certified producer,
dependencies, host, filesystem, and processes sharing the launcher's OS
identity are cooperative, with no surviving producer descendants. See the
governing R-081 process/host trust boundary for the exact limitation.

Spec: .correctless/specs/camera-ready-aims-evidence-2.md R-081/R-082
(operational-rejection repair, 2026-08-22).

Error taxonomy: every PRE-launch defect raises ``LaunchRefusal`` (the run
never started; the message names the refusal class); POST-claim defects
raise ``RunFailed`` and attempt a truthful diagnostic STOP report beside the
retained output state. STOP publication failure never supplies acceptance;
only the terminal positive marker does. Both typed outcomes are
``schema.ColmAimsError`` subclasses.
"""
from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
import platform
import shutil
import stat
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from scripts.stopdff_v5 import fileio
from scripts.stopdff_v5.fileio import (
    create_once_bytes,
    publish_dir_create_once,
    reclaim_empty_relic,
)

from . import pairing, phase4, phase4_finalize_release, schema

# reproducibility/colm_aims_2026/phase4_launcher.py -> repo root.
_REPO_ROOT = Path(__file__).resolve().parents[2]

# R-081 class (3)/(6): ambient provenance overrides — PRESENCE refuses,
# value ignored (an ambient EMPTY status would fake-clean the guard).
AMBIENT_OVERRIDE_VARS = ("MODAL_HOST_GIT_STATUS", "MODAL_HOST_GIT_COMMIT")
AMBIENT_ENV_PREFIX = "MODAL_HOST"
AMBIENT_PYTHON_INJECTION_VARS = (
    "PYTHONPATH",
    "PYTHONHOME",
    "PYTHONUSERBASE",
    "PYTHONWARNINGS",
)

# R-081 (3): deterministic, offline, single-threaded child environment.
LAUNCH_ENV_PINS = {
    "PYTHONHASHSEED": "0",
    "PYTHONNOUSERSITE": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
}

# Closed host-compatibility surface.  The certified interpreter is invoked by
# absolute path, so PATH and loader-control variables are unnecessary and are
# deliberately absent.  These keys are the minimum cross-platform process
# plumbing needed for temporary files, Windows runtime discovery, and stable
# locale handling.
RUNTIME_ENV_ALLOWLIST = frozenset(
    {
        "COMSPEC",
        "HOME",
        "LANG",
        "LC_ALL",
        "SYSTEMROOT",
        "TEMP",
        "TMP",
        "USERPROFILE",
        "WINDIR",
    }
)

LAUNCHER_CONFIG_KEYS = (
    "certificate_path",
    "activation_digest",
    "quarantine_dir",
    "promote_to",
    "ledger_path",
    "snapshot_manifest_path",
    "snapshot_dirs",
    "anchor_path",
)

STOP_REPORT_NAME = "STOP_REPORT.json"
LAUNCH_RECEIPT_NAME = "LAUNCH_RECEIPT.json"
ACCEPTANCE_MARKER_NAME = "LAUNCH_ACCEPTED.json"
ACCEPTANCE_PENDING_NAME = "LAUNCH_ACCEPTANCE_PENDING.json"
CAPTURED_INPUTS_DIRNAME = ".certified_inputs"
PRIVATE_PROMOTION_PREFIX = ".phase4-accepted-"
CERTIFICATE_GENERATION_SUMMARY_NAME = "certificate_generation_summary.json"

# CLI exit-code contract: a pre-launch refusal never consumes the exception;
# a run failure occurs after consumption and preserves the quarantine.  Usage
# errors retain argparse's conventional exit 2 and unexpected defects are
# separated from both typed outcomes.
EXIT_PASS = 0
EXIT_RUN_FAILED = 1
EXIT_LAUNCH_REFUSAL = 3
EXIT_INTERNAL_ERROR = 4

_PATH_CONFIG_KEYS = frozenset(
    {
        "certificate_path",
        "quarantine_dir",
        "promote_to",
        "ledger_path",
        "snapshot_manifest_path",
        "anchor_path",
    }
)

_SNAPSHOT_COMMAND_FLAGS = {
    "primary_scorer": "--primary-model-path",
    "disjoint_selector": "--disjoint-model-path",
}

_PHASE4_PRODUCER_FLAGS = frozenset(
    {
        "--data-dir",
        "--calibration",
        "--fit-split",
        "--eval-split",
        "--reward-schedule",
        "--qa-arms",
        "--calibrations",
        "--num-bootstrap",
        "--n-test",
        "--n-val",
        "--seed",
        "--eligibility",
        "--snapshot-manifest",
        "--primary-model-path",
        "--disjoint-model-path",
        "--records-out",
        "--staged-input",
        "--out",
    }
)

_PHASE4_RECORDS_OUT = "phase4_run_output"
_PHASE4_OUTPUT = "phase4_run_output/stopdff_fair_qa_regenerated.json"
_QA012_RELPATH = phase4.QA012_MANIFEST_RELPATH
_QA012_SHA256 = phase4.QA012_MANIFEST_SHA256
_GIT_UNTRACKED_STATUS_COMMAND = (
    "git",
    "status",
    "--porcelain=v1",
    "-z",
    "--untracked-files=all",
)
_PYTHON_IMPORTABLE_SUFFIXES = frozenset(
    {".py", ".pyc", ".pyo", ".pyd", ".so"}
)


class LaunchRefusal(schema.ColmAimsError):
    """Pre-launch refusal (R-081): the producer was never started."""


class RunFailed(schema.ColmAimsError):
    """Post-ledger failure (R-081), with retained state and a STOP report."""


class ComparatorValidationError(schema.ColmAimsError):
    """A zero-exit producer emitted an inadmissible Phase-4 export."""


def _validate_certificate_envelope_and_semantics(
    certificate: dict[str, Any], components: Any
) -> dict[str, Any]:
    """Reject hand-crafted ``ready: true`` envelopes that bypass assembly."""
    expected_keys = {
        "schema_version",
        "ready",
        "failing_checks",
        "components",
    }
    observed_keys = set(certificate)
    if observed_keys != expected_keys:
        missing = sorted(expected_keys - observed_keys)
        unknown = sorted(observed_keys - expected_keys)
        raise LaunchRefusal(
            "certificate envelope is not closed"
            f" (missing={missing}, unknown={unknown}) (R-079/R-081)"
        )
    schema_version = certificate.get("schema_version")
    if (
        type(schema_version) is not int
        or schema_version != phase4.CERT_SCHEMA_VERSION
    ):
        raise LaunchRefusal(
            "certificate schema_version does not equal the Phase-4"
            " certificate schema (R-079/R-081)"
        )
    if certificate.get("failing_checks") != []:
        raise LaunchRefusal(
            "ready certificate failing_checks must be exactly []"
            " (R-079/R-081)"
        )
    if not isinstance(components, dict):
        raise LaunchRefusal(
            "certificate components must be an object (R-079/R-081)"
        )
    try:
        reassembled = phase4.assemble_certificate(components)
    except Exception as exc:
        raise LaunchRefusal(
            "certificate semantic reassembly crashed"
            f" ({exc.__class__.__name__}) — launch refused pre-ledger"
        ) from exc
    if not isinstance(reassembled, dict):
        raise LaunchRefusal(
            "certificate semantic reassembly returned a malformed result"
            " (R-079/R-081)"
        )
    if reassembled.get("ready") is not True or reassembled.get(
        "failing_checks"
    ) != []:
        failures = reassembled.get("failing_checks")
        shown = failures[:8] if isinstance(failures, list) else failures
        raise LaunchRefusal(
            "certificate semantic reassembly is not ready:true with zero"
            f" failures: {shown!r} (R-079/R-081)"
        )
    return components


def _validate_config_shape(
    config: Any, *, require_json_values: bool = False
) -> dict[str, Any]:
    """Validate the launcher's closed configuration surface.

    ``validate_and_launch`` intentionally continues to accept ``Path``
    objects for programmatic callers.  The file-backed CLI additionally
    requires JSON strings so accidental coercions (``null``, booleans,
    arrays, or objects) never become filesystem paths.
    """
    if not isinstance(config, dict):
        raise LaunchRefusal("launcher config must be a JSON object (R-081)")

    expected = set(LAUNCHER_CONFIG_KEYS)
    observed = set(config)
    missing = sorted(expected - observed)
    unknown = sorted(observed - expected)
    if missing:
        raise LaunchRefusal(
            f"launcher config missing key(s) {missing} (R-081)"
        )
    if unknown:
        raise LaunchRefusal(
            f"launcher config carries unknown key(s) {unknown} — the"
            " config surface is closed (R-081)"
        )

    activation_digest = config["activation_digest"]
    if not schema.is_sha256_hex(activation_digest):
        raise LaunchRefusal(
            "launcher config activation_digest must be a lowercase SHA-256"
            " digest (R-081)"
        )

    snapshot_dirs = config["snapshot_dirs"]
    if not isinstance(snapshot_dirs, dict):
        raise LaunchRefusal(
            "launcher config snapshot_dirs must be an object (R-081)"
        )
    expected_roles = set(phase4.SNAPSHOT_ROLES)
    observed_roles = set(snapshot_dirs)
    missing_roles = sorted(expected_roles - observed_roles)
    unknown_roles = sorted(observed_roles - expected_roles)
    if missing_roles:
        raise LaunchRefusal(
            "launcher config snapshot_dirs missing role(s)"
            f" {missing_roles} (R-081)"
        )
    if unknown_roles:
        raise LaunchRefusal(
            "launcher config snapshot_dirs carries unknown role(s)"
            f" {unknown_roles} — the role map is closed (R-081)"
        )

    path_values = {
        key: config[key] for key in _PATH_CONFIG_KEYS
    } | {
        f"snapshot_dirs.{role}": snapshot_dirs[role]
        for role in sorted(phase4.SNAPSHOT_ROLES)
    }
    for label, value in path_values.items():
        if require_json_values:
            valid = isinstance(value, str) and bool(value)
        else:
            valid = isinstance(value, (str, os.PathLike)) and bool(
                os.fspath(value)
            )
        if not valid:
            value_kind = "a non-empty JSON string" if require_json_values else (
                "a non-empty path"
            )
            raise LaunchRefusal(
                f"launcher config {label} must be {value_kind} (R-081)"
            )
    return config


def _load_launcher_config(config_path: Path) -> dict[str, Any]:
    """Load one bounded, regular JSON file through the namespace parser."""
    config_path = Path(config_path)
    try:
        raw = schema.read_regular_file_bytes(config_path)
    except schema.ColmAimsError as exc:
        raise LaunchRefusal(
            f"launcher config {config_path.name!r} is unreadable: {exc}"
            " (R-081)"
        ) from exc
    try:
        parsed = schema.parse_json_bytes_strict(raw)
    except (UnicodeDecodeError, json.JSONDecodeError, schema.ColmAimsError) as exc:
        raise LaunchRefusal(
            f"launcher config {config_path.name!r} is malformed JSON: {exc}"
            " (R-081)"
        ) from exc
    return _validate_config_shape(parsed, require_json_values=True)


def _resolve_child_path(raw_path: str) -> Path:
    """Resolve a certificate-command path using the child's repo-root cwd."""
    path = Path(raw_path)
    if not path.is_absolute():
        path = _REPO_ROOT / path
    return path.resolve()


def _same_resolved_path(left: Path, right: Path) -> bool:
    """OS-aware equality for already resolved filesystem identities."""
    return os.path.normcase(str(Path(left).resolve())) == os.path.normcase(
        str(Path(right).resolve())
    )


def _path_lexists(path: Path) -> bool:
    """Existence without following a dangling final symlink."""
    return os.path.lexists(os.fspath(path))


def _require_command_option(command: list[str], flag: str) -> str:
    """Return the unique non-empty value for ``flag`` in either spelling."""
    values: list[str] = []
    index = 0
    while index < len(command):
        token = command[index]
        if token == flag:
            if index + 1 >= len(command) or not command[index + 1]:
                raise LaunchRefusal(
                    f"certificate command carries a dangling {flag} flag"
                    " (R-081)"
                )
            values.append(command[index + 1])
            index += 2
            continue
        if token.startswith(f"{flag}="):
            value = token.partition("=")[2]
            if not value:
                raise LaunchRefusal(
                    f"certificate command carries an empty {flag} value"
                    " (R-081)"
                )
            values.append(value)
        index += 1
    if not values:
        raise LaunchRefusal(
            f"certificate command is missing required {flag} binding"
            " (R-081)"
        )
    if len(values) != 1:
        raise LaunchRefusal(
            f"certificate command carries duplicate {flag} bindings"
            " (R-081)"
        )
    return values[0]


def _staged_input_path(spec: str) -> str:
    """Extract PATH from one strict ``LABEL=PATH:SHA256`` command value."""
    label, separator, path_and_digest = spec.partition("=")
    path, digest_separator, digest = path_and_digest.rpartition(":")
    if (
        not separator
        or not digest_separator
        or not label
        or not path
        or not schema.is_sha256_hex(digest)
    ):
        raise LaunchRefusal(
            "certificate command --staged-input must be"
            f" LABEL=PATH:SHA256; found {spec!r} (R-081/R-082)"
        )
    return path


def _default_resolve_executable(command_token: str) -> Path:
    """Resolve and require the actual child executable selected by argv[0]."""
    token_path = Path(command_token)
    if token_path.is_absolute():
        candidate = token_path
    elif token_path.parent != Path("."):
        candidate = _REPO_ROOT / token_path
    else:
        found = shutil.which(command_token)
        if found is None:
            raise LaunchRefusal(
                f"certificate command interpreter {command_token!r} cannot"
                " be resolved on PATH (R-081)"
            )
        candidate = Path(found)
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise LaunchRefusal(
            f"certificate command interpreter {command_token!r} does not"
            " resolve to an existing executable (R-081)"
        ) from exc
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise LaunchRefusal(
            f"certificate command interpreter {command_token!r} is not an"
            " executable regular file (R-081)"
        )
    return resolved


def _default_host_identity() -> dict[str, str]:
    """Capture host fields using the certificate orchestration's spelling."""
    return {
        "os": (
            f"{platform.system()} {platform.release()}"
            f" ({platform.mac_ver()[0]})"
        ),
        "arch": platform.machine(),
    }


def _sanitized_runtime_environment() -> dict[str, str]:
    """Return the exact non-injectable environment used by every child."""
    sanitized = {
        key: value
        for key, value in os.environ.items()
        if key.upper() in RUNTIME_ENV_ALLOWLIST
    }
    sanitized.update(LAUNCH_ENV_PINS)
    return sanitized


def _validate_ambient_environment() -> None:
    """Refuse ambient provenance and Python-import control surfaces."""
    for var in AMBIENT_OVERRIDE_VARS:
        if var in os.environ:
            raise LaunchRefusal(
                f"ambient provenance override {var} is present in the"
                " environment (MODAL_HOST overrides are refused even when"
                " empty — provenance laundering) (R-081)"
            )
    for var in AMBIENT_PYTHON_INJECTION_VARS:
        if var in os.environ:
            raise LaunchRefusal(
                f"ambient Python import injection variable {var} is"
                " present in the environment (presence refuses even when"
                " empty; R-081)"
            )


def _default_probe_environment_lock(interpreter: Path) -> bytes:
    """Return LF-normalized UTF-8 ``pip freeze`` stdout bytes."""
    try:
        completed = subprocess.run(
            [str(interpreter), "-m", "pip", "freeze"],
            cwd=str(_REPO_ROOT),
            capture_output=True,
            check=True,
            env=_sanitized_runtime_environment(),
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise LaunchRefusal(
            "certified interpreter environment-lock probe failed"
            f" ({exc.__class__.__name__}) (R-081)"
        ) from exc
    # Work directly from subprocess bytes: pip emits UTF-8, and explicit
    # newline normalization makes the certificate lock portable across
    # Windows CRLF and POSIX LF without locale-dependent decode/re-encode.
    return completed.stdout.replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def _validate_runtime_binding(
    environment: dict[str, Any],
    command: list[str],
    *,
    resolve_executable: Any,
    host_identity: Any,
) -> Path:
    """Bind argv[0] and the current host to the certified environment."""
    certified_interpreter = environment.get("interpreter_realpath")
    certified_os = environment.get("os")
    certified_arch = environment.get("arch")
    if not all(
        isinstance(value, str) and value
        for value in (certified_interpreter, certified_os, certified_arch)
    ):
        raise LaunchRefusal(
            "certificate environment lacks non-empty interpreter_realpath,"
            " os, or arch runtime bindings (R-081)"
        )
    try:
        command_interpreter = Path(resolve_executable(command[0])).resolve()
    except LaunchRefusal:
        raise
    except Exception as exc:
        raise LaunchRefusal(
            "certificate command interpreter resolution failed"
            f" ({exc.__class__.__name__}) (R-081)"
        ) from exc
    certified_path = Path(certified_interpreter).resolve()
    if not _same_resolved_path(command_interpreter, certified_path):
        raise LaunchRefusal(
            "certificate command interpreter realpath"
            f" {str(command_interpreter)!r} != certified interpreter_realpath"
            f" {str(certified_path)!r} (R-081)"
        )

    try:
        live_host = host_identity()
    except Exception as exc:
        raise LaunchRefusal(
            f"live host identity probe failed ({exc.__class__.__name__})"
            " (R-081)"
        ) from exc
    if not isinstance(live_host, dict):
        raise LaunchRefusal(
            "live host identity probe did not return an object (R-081)"
        )
    for key, certified in (("os", certified_os), ("arch", certified_arch)):
        observed = live_host.get(key)
        if observed != certified:
            raise LaunchRefusal(
                f"live host {key} {observed!r} != certificate environment"
                f" {key} {certified!r} (R-081)"
            )
    return command_interpreter


def _validate_environment_lock(
    environment: dict[str, Any],
    interpreter: Path,
    *,
    probe_environment_lock: Any,
) -> None:
    """Recompute the dependency lock with the exact certified interpreter."""
    certified_digest = environment.get("environment_lock_sha256")
    if not schema.is_sha256_hex(certified_digest):
        raise LaunchRefusal(
            "certificate environment.environment_lock_sha256 is missing or"
            " malformed (R-081)"
        )
    try:
        stdout = probe_environment_lock(interpreter)
    except LaunchRefusal:
        raise
    except Exception as exc:
        raise LaunchRefusal(
            "certified interpreter environment-lock probe failed"
            f" ({exc.__class__.__name__}) (R-081)"
        ) from exc
    if isinstance(stdout, str):
        lock_bytes = stdout.encode("utf-8")
    elif isinstance(stdout, bytes):
        lock_bytes = stdout
    else:
        raise LaunchRefusal(
            "environment-lock probe returned neither text nor bytes"
            " (R-081)"
        )
    observed_digest = hashlib.sha256(lock_bytes).hexdigest()
    if observed_digest != certified_digest:
        raise LaunchRefusal(
            f"live pip-freeze sha256 {observed_digest} != certificate"
            " environment.environment_lock_sha256"
            f" {certified_digest} (R-081)"
        )


def _validate_external_config_bindings(
    config: dict[str, Any], command: list[str]
) -> None:
    """Prevent external config from substituting command-owned artifacts."""
    # Omitting either option would silently select the producer's in-repo
    # default, bypassing the R-082 location check.  Requiring the unique
    # explicit value also rejects duplicates and supports both spellings.
    _require_command_option(command, "--data-dir")
    _require_command_option(command, "--calibration")
    _require_command_option(command, "--eligibility")
    _require_command_option(command, "--out")
    _require_command_option(command, "--records-out")

    manifest_command = _require_command_option(command, "--snapshot-manifest")
    manifest_config = Path(config["snapshot_manifest_path"]).resolve()
    if not _same_resolved_path(
        manifest_config, _resolve_child_path(manifest_command)
    ):
        raise LaunchRefusal(
            "launcher config snapshot_manifest_path does not resolve to the"
            " certificate command's --snapshot-manifest value (R-081)"
        )

    snapshot_dirs = config["snapshot_dirs"]
    for role, flag in _SNAPSHOT_COMMAND_FLAGS.items():
        command_value = _require_command_option(command, flag)
        config_value = Path(snapshot_dirs[role]).resolve()
        if not _same_resolved_path(
            config_value, _resolve_child_path(command_value)
        ):
            raise LaunchRefusal(
                f"launcher config snapshot_dirs[{role!r}] does not resolve"
                f" to the certificate command's {flag} value (R-081)"
            )


def _validate_launch_workspace_bindings(
    config: dict[str, Any],
    repo: dict[str, Any],
    environment: dict[str, Any],
) -> tuple[Path, Path, Path]:
    """Bind all mutable launch paths to the certified external topology.

    The ledger path is certificate-owned just like the quarantine and
    promotion paths: accepting an arbitrary fresh ledger from launcher JSON
    would turn the one-use exception into a reusable switch.  Lexical
    equality is deliberate here (after the certificate itself has required
    absolute paths); canonical topology checks then close alias/nesting
    escapes.
    """
    repo_root = repo.get(phase4.R082_REPO_ROOT_FIELD)
    try:
        failures = phase4.launch_path_failures(repo_root, environment)
    except Exception as exc:
        raise LaunchRefusal(
            "certificate launch-workspace checker crashed"
            f" ({exc.__class__.__name__}) — launch refused pre-ledger"
        ) from exc
    if not isinstance(failures, list):
        raise LaunchRefusal(
            "certificate launch-workspace checker returned a malformed"
            " result — launch refused pre-ledger"
        )
    if failures:
        shown = "; ".join(str(failure) for failure in failures[:8])
        remainder = len(failures) - 8
        suffix = f"; and {remainder} more" if remainder > 0 else ""
        raise LaunchRefusal(
            f"certificate launch-workspace refusal: {shown}{suffix} (R-081)"
        )

    try:
        certified_root_matches = _same_resolved_path(
            Path(repo_root), _REPO_ROOT
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise LaunchRefusal(
            "certificate repository root cannot be resolved for launch-path"
            f" validation ({exc.__class__.__name__}) (R-081)"
        ) from exc
    if not certified_root_matches:
        raise LaunchRefusal(
            "certificate repository root does not match the launcher's"
            " actual repository root (R-081)"
        )

    bindings = (
        ("quarantine_dir", "quarantine_dir"),
        ("promote_to", "promote_to"),
        ("ledger_path", "exception_ledger_path"),
    )
    resolved: dict[str, Path] = {}
    for config_key, environment_key in bindings:
        raw_config = os.fspath(config[config_key])
        raw_certified = environment.get(environment_key)
        if raw_config != raw_certified:
            raise LaunchRefusal(
                f"launcher config {config_key} {raw_config!r} does not"
                f" exactly match certificate environment.{environment_key}"
                f" {raw_certified!r} (R-081)"
            )
        # ``launch_path_failures`` has already proved that the certified
        # value is a non-empty absolute path and canonicalized its topology.
        try:
            raw_path = Path(raw_config)
            resolved_path = raw_path.resolve()
            lexical_absolute = Path(os.path.abspath(os.fspath(raw_path)))
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            raise LaunchRefusal(
                f"certificate environment.{environment_key} cannot be"
                f" resolved ({exc.__class__.__name__}) (R-081)"
            ) from exc
        if os.path.normcase(str(lexical_absolute)) != os.path.normcase(
            str(resolved_path)
        ):
            raise LaunchRefusal(
                f"certificate environment.{environment_key} is not a"
                " canonical resolved path — symlink/junction/alias redirects"
                " are refused (R-081)"
            )
        resolved[config_key] = resolved_path

    # Every create/rename parent is checked before the fresh quarantine is
    # materialized, and therefore before O_EXCL can consume the exception.
    access_mode = os.W_OK | (os.X_OK if os.name != "nt" else 0)
    for config_key in ("quarantine_dir", "promote_to", "ledger_path"):
        parent = resolved[config_key].parent
        if not parent.is_dir():
            raise LaunchRefusal(
                f"launcher config {config_key} parent {parent} does not"
                " exist or is not a directory (R-081)"
            )
        if not os.access(parent, access_mode):
            raise LaunchRefusal(
                f"launcher config {config_key} parent {parent} is not"
                " writable (R-081)"
            )

    return (
        resolved["quarantine_dir"],
        resolved["promote_to"],
        resolved["ledger_path"],
    )


def _validate_staged_coverage(
    repo: dict[str, Any],
    components: dict[str, Any],
    command: list[str],
) -> None:
    """Re-prove the canonical seven-input command/component coverage contract."""
    repo_root = repo.get(phase4.R082_REPO_ROOT_FIELD)
    try:
        failures = phase4.staged_coverage_failures(
            repo_root,
            components.get("staged_inputs"),
            components.get("eligibility"),
            command,
        )
    except Exception as exc:
        raise LaunchRefusal(
            "R-082 staged coverage checker crashed"
            f" ({exc.__class__.__name__}) — launch refused pre-ledger"
        ) from exc
    if not isinstance(failures, list):
        raise LaunchRefusal(
            "R-082 staged coverage checker returned a malformed result —"
            " launch refused pre-ledger"
        )
    if failures:
        shown = "; ".join(str(failure) for failure in failures[:8])
        remainder = len(failures) - 8
        suffix = f"; and {remainder} more" if remainder > 0 else ""
        raise LaunchRefusal(
            f"R-082 staged coverage refusal: {shown}{suffix}"
        )

    # The helper resolves relative command paths from the certified root;
    # bind that root to this launcher's actual child cwd so a hand-crafted
    # ready:true certificate cannot make the coverage proof about another
    # checkout.
    try:
        roots_match = _same_resolved_path(Path(repo_root), _REPO_ROOT)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise LaunchRefusal(
            "R-082 certificate repository root cannot be resolved"
            f" ({exc.__class__.__name__})"
        ) from exc
    if not roots_match:
        raise LaunchRefusal(
            "R-082 certificate repository root does not match the launcher's"
            " repository root — command path bindings target another checkout"
        )


def _validate_phase4_command(
    repo: dict[str, Any], environment: dict[str, Any], command: list[str]
) -> None:
    """Re-prove the exact full-run producer/argparse contract pre-ledger."""
    try:
        failures = phase4.phase4_command_failures(command)
        failures.extend(
            phase4.command_environment_failures(
                repo.get(phase4.R082_REPO_ROOT_FIELD), environment
            )
        )
    except Exception as exc:
        raise LaunchRefusal(
            "Phase-4 command checker crashed"
            f" ({exc.__class__.__name__}) — launch refused pre-ledger"
        ) from exc
    if not isinstance(failures, list):
        raise LaunchRefusal(
            "Phase-4 command checker returned a malformed result — launch"
            " refused pre-ledger"
        )

    # The pure checker owns the experimental values.  The launcher also
    # closes the executable surface itself: exactly the audited producer
    # script, followed solely by recognized value-taking options.
    if len(command) < 2:
        failures.append("exact command is missing the producer script")
    else:
        try:
            observed_script = _resolve_child_path(command[1])
            expected_script = (
                _REPO_ROOT / "scripts" / "stopdff_fair_qa_retest.py"
            ).resolve()
            if not _same_resolved_path(observed_script, expected_script):
                failures.append(
                    "exact command script does not resolve to"
                    " scripts/stopdff_fair_qa_retest.py"
                )
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            failures.append(
                "exact command producer script could not be resolved"
                f" ({exc.__class__.__name__})"
            )

    index = 2
    while index < len(command):
        token = command[index]
        flag, equals, equals_value = token.partition("=")
        if flag not in _PHASE4_PRODUCER_FLAGS:
            failures.append(
                f"exact command carries unknown option/positional token"
                f" {token!r}"
            )
            index += 1
            continue
        if equals:
            if not equals_value:
                failures.append(f"exact command {flag} has an empty value")
            index += 1
            continue
        if index + 1 >= len(command) or command[index + 1].startswith("--"):
            failures.append(f"exact command {flag} has no value")
            index += 1
            continue
        index += 2

    if environment.get("seeds") != [1]:
        failures.append(
            "certificate environment.seeds must equal [1] and bind the"
            " exact command --seed 1"
        )
    if environment.get("pythonhashseed") != LAUNCH_ENV_PINS["PYTHONHASHSEED"]:
        failures.append(
            "certificate environment.pythonhashseed must equal the launcher"
            " PYTHONHASHSEED pin '0'"
        )
    expected_thread_settings = dict(phase4.PHASE4_THREAD_SETTINGS)
    for key, value in expected_thread_settings.items():
        if LAUNCH_ENV_PINS.get(key) != value:
            failures.append(
                f"launcher runtime pin {key} does not equal the certificate"
                f" contract value {value!r}"
            )
    if environment.get("thread_settings") != expected_thread_settings:
        failures.append(
            "certificate environment.thread_settings does not exactly match"
            " the launcher's deterministic thread pins"
        )
    for flag, expected in (("--fit-split", "val"), ("--eval-split", "test")):
        try:
            observed = _require_command_option(command, flag)
        except LaunchRefusal as exc:
            failures.append(str(exc))
        else:
            if observed != expected:
                failures.append(
                    f"exact command {flag} must equal {expected!r};"
                    f" found {observed!r}"
                )
    for flag, expected in (
        ("--records-out", _PHASE4_RECORDS_OUT),
        ("--out", _PHASE4_OUTPUT),
    ):
        try:
            observed = _require_command_option(command, flag)
        except LaunchRefusal as exc:
            failures.append(str(exc))
        else:
            if observed != expected:
                failures.append(
                    f"exact command {flag} must equal {expected!r};"
                    f" found {observed!r}"
                )
    if failures:
        shown = "; ".join(str(failure) for failure in failures[:10])
        remainder = len(failures) - 10
        suffix = f"; and {remainder} more" if remainder > 0 else ""
        raise LaunchRefusal(
            f"Phase-4 exact-command refusal: {shown}{suffix} (R-081)"
        )


def _load_verified_eligibility(
    components: dict[str, Any],
) -> dict[str, Any]:
    """Rehash, strict-load, and pin the pairing eligibility pre-ledger."""
    eligibility = components.get("eligibility")
    if not isinstance(eligibility, dict):
        raise LaunchRefusal(
            "certificate eligibility component is missing or malformed"
            " (R-074/R-081)"
        )

    artifact_path = eligibility.get("artifact_path")
    artifact_sha256 = eligibility.get("artifact_sha256")
    certified_keyset = eligibility.get("digest")
    certified_horizons = eligibility.get("horizon_map_sha256")
    certified_test = eligibility.get("test_dataset_sha256")
    if not isinstance(artifact_path, str) or not artifact_path:
        raise LaunchRefusal(
            "certificate eligibility.artifact_path is missing or malformed"
            " (R-074/R-081)"
        )
    if not Path(artifact_path).is_absolute():
        raise LaunchRefusal(
            "certificate eligibility.artifact_path must be absolute"
            " (R-074/R-081)"
        )
    canonical_path = (_REPO_ROOT / phase4.ELIGIBILITY_ARTIFACT_RELPATH).resolve()
    if not _same_resolved_path(Path(artifact_path), canonical_path):
        raise LaunchRefusal(
            "certificate eligibility.artifact_path does not resolve to the"
            " canonical frozen pairing eligibility artifact"
            " (R-074/R-081)"
        )
    for field, value in (
        ("artifact_sha256", artifact_sha256),
        ("digest", certified_keyset),
        ("horizon_map_sha256", certified_horizons),
        ("test_dataset_sha256", certified_test),
    ):
        if not schema.is_sha256_hex(value):
            raise LaunchRefusal(
                f"certificate eligibility.{field} is not a lowercase"
                " SHA-256 digest (R-074/R-081)"
            )
    canonical_pins = (
        ("artifact_sha256", artifact_sha256, phase4.ELIGIBILITY_ARTIFACT_SHA256),
        ("digest", certified_keyset, phase4.ELIGIBILITY_KEYSET_SHA256),
        (
            "horizon_map_sha256",
            certified_horizons,
            phase4.ELIGIBILITY_HORIZON_MAP_SHA256,
        ),
        (
            "test_dataset_sha256",
            certified_test,
            phase4.ELIGIBILITY_TEST_DATASET_SHA256,
        ),
    )
    for field, observed, expected in canonical_pins:
        if observed != expected:
            raise LaunchRefusal(
                f"certificate eligibility.{field} {observed!r} != canonical"
                f" frozen pin {expected!r} (R-074/R-081)"
            )

    live_digest = _sha256_regular_file(
        Path(artifact_path), label="pairing eligibility artifact"
    )
    if live_digest != artifact_sha256:
        raise LaunchRefusal(
            f"pairing eligibility artifact live SHA-256 {live_digest} !="
            " certificate eligibility.artifact_sha256"
            f" {artifact_sha256} (R-074/R-081)"
        )
    try:
        loaded = phase4.load_pairing_eligibility(Path(artifact_path))
    except schema.ColmAimsError as exc:
        raise LaunchRefusal(
            f"pairing eligibility artifact failed strict load: {exc}"
            " (R-074/R-081)"
        ) from exc

    loaded_bindings = (
        (
            "pairing_population_keyset_sha256",
            certified_keyset,
            "digest",
        ),
        ("horizon_map_sha256", certified_horizons, "horizon_map_sha256"),
    )
    for artifact_field, certified, certificate_field in loaded_bindings:
        observed = loaded.get(artifact_field)
        if observed != certified:
            raise LaunchRefusal(
                f"pairing eligibility artifact {artifact_field}"
                f" {observed!r} != certificate eligibility."
                f"{certificate_field} {certified!r} (R-074/R-081)"
            )
    derived = loaded.get("derived_from")
    derived = derived if isinstance(derived, dict) else {}
    observed_test = derived.get("test_dataset_sha256")
    if observed_test != certified_test:
        raise LaunchRefusal(
            "pairing eligibility artifact derived_from.test_dataset_sha256"
            f" {observed_test!r} != certificate eligibility."
            f"test_dataset_sha256 {certified_test!r} (R-074/R-081)"
        )

    # A loader path is necessarily reopened; rehash after the strict load as
    # well, so ordinary replace/mutation races cannot swap the parsed bytes
    # away from the certified raw-byte identity unnoticed.
    post_load_digest = _sha256_regular_file(
        Path(artifact_path), label="pairing eligibility artifact"
    )
    if post_load_digest != artifact_sha256:
        raise LaunchRefusal(
            "pairing eligibility artifact changed while it was being"
            " verified (R-074/R-081)"
        )
    return loaded


def _load_verified_snapshot_manifest(
    config: dict[str, Any],
    repo: dict[str, Any],
    components: dict[str, Any],
    command: list[str],
) -> dict[str, Any]:
    """Bind, raw-hash, and strict-load the certified snapshot manifest."""
    snapshots = components.get("snapshots")
    if not isinstance(snapshots, dict):
        raise LaunchRefusal(
            "certificate snapshots component is missing or malformed"
            " (R-075/R-081)"
        )
    repo_root = repo.get(phase4.R082_REPO_ROOT_FIELD)
    try:
        failures = phase4.snapshot_manifest_failures(
            repo_root, snapshots, command
        )
    except Exception as exc:
        raise LaunchRefusal(
            "snapshot-manifest binding checker crashed"
            f" ({exc.__class__.__name__}) — launch refused pre-ledger"
        ) from exc
    if not isinstance(failures, list):
        raise LaunchRefusal(
            "snapshot-manifest binding checker returned a malformed result"
            " — launch refused pre-ledger"
        )
    if failures:
        shown = "; ".join(str(failure) for failure in failures[:8])
        remainder = len(failures) - 8
        suffix = f"; and {remainder} more" if remainder > 0 else ""
        raise LaunchRefusal(
            f"snapshot-manifest binding refusal: {shown}{suffix}"
            " (R-075/R-081)"
        )

    artifact_path = snapshots.get("artifact_path")
    artifact_sha256 = snapshots.get("artifact_sha256")
    if not isinstance(artifact_path, str) or not artifact_path:
        raise LaunchRefusal(
            "certificate snapshots.artifact_path is missing or malformed"
            " (R-075/R-081)"
        )
    if not Path(artifact_path).is_absolute():
        raise LaunchRefusal(
            "certificate snapshots.artifact_path must be absolute"
            " (R-075/R-081)"
        )
    canonical_path = (_REPO_ROOT / phase4.SNAPSHOT_MANIFEST_RELPATH).resolve()
    if not _same_resolved_path(Path(artifact_path), canonical_path):
        raise LaunchRefusal(
            "certificate snapshots.artifact_path does not resolve to the"
            " canonical frozen model snapshot manifest (R-075/R-081)"
        )
    if not schema.is_sha256_hex(artifact_sha256):
        raise LaunchRefusal(
            "certificate snapshots.artifact_sha256 is malformed"
            " (R-075/R-081)"
        )
    if artifact_sha256 != phase4.SNAPSHOT_MANIFEST_SHA256:
        raise LaunchRefusal(
            "certificate snapshots.artifact_sha256 does not equal the"
            " canonical frozen snapshot-manifest pin (R-075/R-081)"
        )
    if os.fspath(config["snapshot_manifest_path"]) != artifact_path:
        raise LaunchRefusal(
            "launcher config snapshot_manifest_path does not exactly match"
            " certificate snapshots.artifact_path (R-075/R-081)"
        )
    observed_sha256 = _sha256_regular_file(
        Path(artifact_path), label="model snapshot manifest"
    )
    if observed_sha256 != artifact_sha256:
        raise LaunchRefusal(
            f"model snapshot manifest live SHA-256 {observed_sha256} !="
            " certificate snapshots.artifact_sha256"
            f" {artifact_sha256} (R-075/R-081)"
        )
    try:
        manifest = phase4.load_model_snapshot_manifest(Path(artifact_path))
    except schema.ColmAimsError as exc:
        raise LaunchRefusal(
            f"snapshot manifest failed strict load: {exc} (R-075/R-081)"
        ) from exc
    manifest_roles = manifest.get("roles") if isinstance(manifest, dict) else None
    if not isinstance(manifest_roles, dict):
        raise LaunchRefusal(
            "loaded snapshot manifest roles map is missing or malformed"
            " (R-075/R-081)"
        )
    for role in phase4.SNAPSHOT_ROLES:
        certified_role = snapshots.get(role)
        loaded_role = manifest_roles.get(role)
        if not isinstance(certified_role, dict) or not isinstance(
            loaded_role, dict
        ):
            raise LaunchRefusal(
                f"snapshot role {role!r} is missing or malformed in the"
                " certificate or loaded manifest (R-075/R-081)"
            )
        for field in ("model_name", "hf_revision"):
            certified_value = certified_role.get(field)
            loaded_value = loaded_role.get(field)
            if not isinstance(certified_value, str) or not certified_value:
                raise LaunchRefusal(
                    f"certificate snapshots[{role!r}].{field} is missing or"
                    " malformed (R-075/R-081)"
                )
            if loaded_value != certified_value:
                raise LaunchRefusal(
                    f"loaded snapshot manifest role {role!r} {field}"
                    f" {loaded_value!r} != certificate snapshots role"
                    f" {certified_value!r} (R-075/R-081)"
                )
    post_load_sha256 = _sha256_regular_file(
        Path(artifact_path), label="model snapshot manifest"
    )
    if post_load_sha256 != artifact_sha256:
        raise LaunchRefusal(
            "model snapshot manifest changed while it was being verified"
            " (R-075/R-081)"
        )
    return manifest


def _verify_canonical_qa012(components: dict[str, Any]) -> None:
    """Rehash the exact tracked QA-012 rev3 inventory pre-ledger."""
    qa012 = components.get("qa012")
    qa012 = qa012 if isinstance(qa012, dict) else {}
    certified_digest = qa012.get("manifest_sha256")
    if certified_digest != _QA012_SHA256:
        raise LaunchRefusal(
            "certificate qa012.manifest_sha256 does not equal the canonical"
            " rev3 inventory pin (R-079/R-081)"
        )
    manifest_path = (_REPO_ROOT / _QA012_RELPATH).resolve()
    observed_digest = _sha256_regular_file(
        manifest_path, label="QA-012 rev3 inventory"
    )
    if observed_digest != _QA012_SHA256:
        raise LaunchRefusal(
            f"canonical QA-012 rev3 inventory live SHA-256 {observed_digest}"
            f" != frozen pin {_QA012_SHA256} (R-079/R-081)"
        )


def _verify_content_hashes(
    repo: dict[str, Any], components: dict[str, Any]
) -> frozenset[str]:
    """Re-prove every canonical audited source from its live raw bytes."""
    content_hashes = components.get("content_hashes")
    try:
        failures = phase4.content_hash_failures(
            repo.get(phase4.R082_REPO_ROOT_FIELD), content_hashes
        )
    except Exception as exc:
        raise LaunchRefusal(
            "certificate content_hashes validation crashed"
            f" ({exc.__class__.__name__}) — launch refused pre-ledger"
        ) from exc
    if failures:
        shown = "; ".join(str(failure) for failure in failures[:10])
        raise LaunchRefusal(
            f"certificate content_hashes refusal: {shown}"
            " (R-079/R-081)"
        )

    assert isinstance(content_hashes, dict)
    bound_relpaths: set[str] = set()
    for key, relpath in phase4.CONTENT_HASH_RELPATHS.items():
        entry = content_hashes[key]
        assert isinstance(entry, dict)
        artifact_path = Path(entry["artifact_path"])
        expected = entry["sha256"]
        observed = _sha256_regular_file(
            artifact_path, label=f"canonical audited source {key!r}"
        )
        if observed != expected:
            raise LaunchRefusal(
                f"canonical audited source {key!r} live SHA-256"
                f" {observed} != certificate content_hashes SHA-256"
                f" {expected} — source bytes changed after certification"
                " (R-079/R-081)"
            )
        bound_relpaths.add(PurePosixPath(relpath).as_posix())
    return frozenset(bound_relpaths)


def _parse_live_untracked_status(raw_status: Any) -> list[str]:
    """Parse only NUL-delimited porcelain-v1 untracked records."""
    if isinstance(raw_status, bytes):
        try:
            status = raw_status.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise LaunchRefusal(
                "live untracked git status is not valid UTF-8 (R-081)"
            ) from exc
    elif isinstance(raw_status, str):
        status = raw_status
    else:
        raise LaunchRefusal(
            "live untracked git status did not return text or bytes"
            " (R-081)"
        )
    if not status:
        return []
    if not status.endswith("\x00"):
        raise LaunchRefusal(
            "live untracked git status is not NUL-terminated porcelain-v1"
            " output (R-081)"
        )

    paths: list[str] = []
    for record in status[:-1].split("\x00"):
        if not record.startswith("?? "):
            raise LaunchRefusal(
                "live full git status contains a non-untracked or malformed"
                f" record {record[:80]!r} (R-081)"
            )
        paths.append(record[3:])
    paths.sort()
    failures = phase4.untracked_disclosure_failures(paths)
    if failures:
        raise LaunchRefusal(
            "live untracked git paths are not canonical:"
            f" {failures[:8]!r} (R-081)"
        )
    return paths


def _is_import_capable_untracked(relpath: str) -> bool:
    """Whether a disclosed repo path can participate in normal imports."""
    path = PurePosixPath(relpath)
    # A leading-dot component cannot be named by a normal dotted Python
    # import.  All other Python source/bytecode/native-module suffixes are
    # conservatively treated as executable import surfaces at any depth.
    if any(part.startswith(".") for part in path.parts):
        return False
    return path.suffix.lower() in _PYTHON_IMPORTABLE_SUFFIXES


def _validate_untracked_disclosure(
    repo: dict[str, Any],
    bound_content_relpaths: frozenset[str],
    *,
    run_git: Any,
) -> None:
    """Match the signed disclosure and reject unbound import surfaces."""
    certified = repo.get("untracked_disclosure")
    failures = phase4.untracked_disclosure_failures(certified)
    if failures:
        raise LaunchRefusal(
            "certificate repo.untracked_disclosure is malformed:"
            f" {failures[:8]!r} (R-079/R-081)"
        )
    assert isinstance(certified, list)
    live = _parse_live_untracked_status(
        run_git(list(_GIT_UNTRACKED_STATUS_COMMAND))
    )
    if live != certified:
        added = sorted(set(live) - set(certified))
        missing = sorted(set(certified) - set(live))
        raise LaunchRefusal(
            "live untracked path disclosure does not exactly equal the"
            " certificate"
            f" (added={added[:8]!r}, missing={missing[:8]!r}) (R-081)"
        )
    unbound_imports = [
        path
        for path in live
        if _is_import_capable_untracked(path)
        and path not in bound_content_relpaths
    ]
    if unbound_imports:
        raise LaunchRefusal(
            "untracked import-capable path(s) are not canonical"
            " live-hash-bound content entries:"
            f" {unbound_imports[:8]!r} (R-081)"
        )


def _load_verified_anchor(
    config: dict[str, Any], components: dict[str, Any]
) -> dict[str, Any]:
    """Rehash and parse the comparator anchor exactly once, pre-ledger."""
    parity = components.get("parity")
    parity = parity if isinstance(parity, dict) else {}
    certified_digest = parity.get("anchor_sha256")
    if not schema.is_sha256_hex(certified_digest):
        raise LaunchRefusal(
            "certificate parity.anchor_sha256 is missing or malformed"
            " (R-081)"
        )
    if certified_digest != phase4.PARITY_ANCHOR_SHA256:
        raise LaunchRefusal(
            "certificate parity.anchor_sha256 does not equal the canonical"
            " frozen parity-anchor pin (R-077/R-081)"
        )
    anchor_path = Path(config["anchor_path"])
    canonical_path = (_REPO_ROOT / phase4.PARITY_ANCHOR_RELPATH).resolve()
    if not _same_resolved_path(anchor_path, canonical_path):
        raise LaunchRefusal(
            "launcher config anchor_path does not resolve to the canonical"
            " frozen parity anchor (R-077/R-081)"
        )
    try:
        anchor_bytes = schema.read_regular_file_bytes(anchor_path)
    except schema.ColmAimsError as exc:
        raise LaunchRefusal(
            f"parity anchor {anchor_path.name!r} is unreadable: {exc}"
            " (R-081)"
        ) from exc
    observed_digest = hashlib.sha256(anchor_bytes).hexdigest()
    if observed_digest != certified_digest:
        raise LaunchRefusal(
            f"parity anchor sha256 {observed_digest} != certificate"
            f" parity.anchor_sha256 {certified_digest} (R-081)"
        )
    try:
        anchor = schema.parse_json_bytes_strict(anchor_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError, schema.ColmAimsError) as exc:
        raise LaunchRefusal(
            f"parity anchor {anchor_path.name!r} is malformed JSON: {exc}"
            " (R-081)"
        ) from exc
    if not isinstance(anchor, dict):
        raise LaunchRefusal(
            f"parity anchor {anchor_path.name!r} must be a JSON object"
            " (R-081)"
        )
    source = anchor.get("source")
    source = source if isinstance(source, dict) else {}
    if source.get("sha256") != phase4.PARITY_SOURCE_EXPORT_A_SHA256:
        raise LaunchRefusal(
            "canonical parity anchor source.sha256 does not equal the"
            " frozen Export-A source pin (R-077/R-081)"
        )
    return anchor


def _sha256_regular_file(path: Path, *, label: str) -> str:
    """Stream-hash a regular, symlink-free staged input of any size."""
    path = Path(path)
    if path.is_symlink():
        raise LaunchRefusal(
            f"{label} ({path.name}): refusing to hash a symlink (R-082)"
        )
    flags = (
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
        | getattr(os, "O_BINARY", 0)
    )
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        if exc.errno in (errno.ELOOP, errno.EMLINK):
            raise LaunchRefusal(
                f"{label} ({path.name}): refusing to hash a symlink (R-082)"
            ) from exc
        raise LaunchRefusal(
            f"{label} ({path.name}): missing or unreadable"
            f" ({exc.__class__.__name__}) (R-082)"
        ) from exc
    try:
        file_stat = os.fstat(fd)
        if not stat.S_ISREG(file_stat.st_mode):
            raise LaunchRefusal(
                f"{label} ({path.name}): not a regular file — refusing to"
                " hash a FIFO, device, or socket (R-082)"
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


def _verify_staged_input_hashes(components: dict[str, Any]) -> None:
    """Require certificate and live byte identity for every staged input."""
    entries = components.get("staged_inputs")
    if not isinstance(entries, list) or not entries:
        raise LaunchRefusal(
            "certificate components.staged_inputs must be a non-empty list"
            " (R-082)"
        )
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise LaunchRefusal(
                f"certificate staged_inputs[{index}] must be an object"
                " (R-082)"
            )
        label = entry.get("label")
        raw_path = entry.get("path")
        expected = entry.get("expected_sha256")
        observed = entry.get("observed_sha256")
        if not isinstance(label, str) or not label:
            raise LaunchRefusal(
                f"certificate staged_inputs[{index}].label is malformed"
                " (R-082)"
            )
        if not isinstance(raw_path, str) or not raw_path:
            raise LaunchRefusal(
                f"certificate staged input {label!r} path is malformed"
                " (R-082)"
            )
        if not schema.is_sha256_hex(expected) or not schema.is_sha256_hex(
            observed
        ):
            raise LaunchRefusal(
                f"certificate staged input {label!r} carries a malformed"
                " expected/observed SHA-256 (R-082)"
            )
        if expected != observed:
            raise LaunchRefusal(
                f"certificate staged input {label!r} expected SHA-256"
                f" {expected} != certified observed SHA-256 {observed}"
                " (R-082)"
            )
        live = _sha256_regular_file(Path(raw_path), label=f"staged input {label!r}")
        if live != expected:
            raise LaunchRefusal(
                f"staged input {label!r} live SHA-256 {live} != certified"
                f" SHA-256 {expected} (R-082)"
            )


def _write_all(fd: int, data: bytes) -> None:
    """Write all bytes or fail instead of accepting a short write."""
    offset = 0
    while offset < len(data):
        written = os.write(fd, data[offset:])
        if type(written) is not int or written <= 0:
            raise OSError(errno.EIO, "captured-input write made no progress")
        offset += written


def _capture_regular_file(
    source: Path,
    destination: Path,
    *,
    expected_sha256: str,
    label: str,
    expected_size: int | None = None,
) -> None:
    """Copy and authenticate one source through the same held read handle.

    The child receives only ``destination``.  A replacement of ``source``
    after this function returns therefore cannot change the bytes consumed by
    the producer.  ``O_NOFOLLOW`` is used where available and both the digest
    and optional declared size are checked from the held descriptor.
    """
    source = Path(source)
    destination = Path(destination)
    if source.is_symlink():
        raise LaunchRefusal(f"{label}: source is a symlink (R-075/R-082)")
    source_flags = (
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
        | getattr(os, "O_BINARY", 0)
    )
    try:
        source_fd = os.open(source, source_flags)
    except OSError as exc:
        raise LaunchRefusal(
            f"{label}: source is missing or unreadable"
            f" ({exc.__class__.__name__}) (R-075/R-082)"
        ) from exc
    destination_fd: int | None = None
    try:
        source_stat = os.fstat(source_fd)
        if not stat.S_ISREG(source_stat.st_mode):
            raise LaunchRefusal(
                f"{label}: source is not a regular file (R-075/R-082)"
            )
        if expected_size is not None and source_stat.st_size != expected_size:
            raise LaunchRefusal(
                f"{label}: source size {source_stat.st_size} != declared"
                f" size {expected_size} (R-075)"
            )
        if hasattr(os, "O_NONBLOCK") and hasattr(os, "get_blocking"):
            os.set_blocking(source_fd, True)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination_fd = os.open(
            destination,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_BINARY", 0),
            0o400,
        )
        digest = hashlib.sha256()
        copied = 0
        while True:
            chunk = os.read(source_fd, 1 << 20)
            if not chunk:
                break
            digest.update(chunk)
            copied += len(chunk)
            _write_all(destination_fd, chunk)
        if expected_size is not None and copied != expected_size:
            raise LaunchRefusal(
                f"{label}: copied size {copied} != declared size"
                f" {expected_size} (R-075)"
            )
        observed_sha256 = digest.hexdigest()
        if observed_sha256 != expected_sha256:
            raise LaunchRefusal(
                f"{label}: copied SHA-256 {observed_sha256} != certified"
                f" SHA-256 {expected_sha256} (R-075/R-082)"
            )
        os.fsync(destination_fd)
    except OSError as exc:
        raise LaunchRefusal(
            f"{label}: could not materialize authenticated private copy"
            f" ({exc.__class__.__name__}) (R-075/R-082)"
        ) from exc
    finally:
        if destination_fd is not None:
            os.close(destination_fd)
        os.close(source_fd)
    try:
        destination.chmod(stat.S_IREAD)
    except OSError as exc:
        raise LaunchRefusal(
            f"{label}: authenticated private copy cannot be made read-only"
            f" ({exc.__class__.__name__}) (R-075/R-082)"
        ) from exc


def _capture_snapshot_role(
    role_entry: dict[str, Any], source_dir: Path, destination_dir: Path
) -> None:
    """Capture every declared model file into one private role directory."""
    try:
        destination_dir.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        raise LaunchRefusal(
            "private model-snapshot directory cannot be created"
            f" ({exc.__class__.__name__}) (R-075/R-082)"
        ) from exc
    files = role_entry["files"]
    for rel_name in sorted(files):
        metadata = files[rel_name]
        _capture_regular_file(
            Path(source_dir) / PurePosixPath(rel_name),
            destination_dir / PurePosixPath(rel_name),
            expected_sha256=metadata["sha256"],
            expected_size=metadata["size"],
            label=f"snapshot role {role_entry['model_name']!r} file {rel_name!r}",
        )


def _capture_verified_inputs(
    quarantine_dir: Path,
    config: dict[str, Any],
    components: dict[str, Any],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """Materialize all child-consumed mutable inputs under quarantine."""
    capture_root = quarantine_dir / CAPTURED_INPUTS_DIRNAME
    try:
        capture_root.mkdir(parents=False, exist_ok=False)
    except OSError as exc:
        raise LaunchRefusal(
            "private captured-input directory cannot be created"
            f" ({exc.__class__.__name__}) (R-075/R-082)"
        ) from exc

    entries = components["staged_inputs"]
    by_label = {entry["label"]: entry for entry in entries}
    captured_staged: dict[str, Path] = {}
    data_dir = capture_root / "data"
    for label in phase4.R082_STAGED_INPUT_LABELS:
        entry = by_label[label]
        if label == "calibration_train":
            destination = capture_root / "calibration_train.json"
        else:
            destination = data_dir / phase4.R082_DATA_FILENAMES[label]
        _capture_regular_file(
            Path(entry["path"]),
            destination,
            expected_sha256=entry["expected_sha256"],
            label=f"staged input {label!r}",
        )
        captured_staged[label] = destination

    snapshots = components["snapshots"]
    captured_manifest = capture_root / "model_snapshot_manifest.json"
    _capture_regular_file(
        Path(snapshots["artifact_path"]),
        captured_manifest,
        expected_sha256=snapshots["artifact_sha256"],
        label="model snapshot manifest",
    )
    captured_snapshots: dict[str, Path] = {}
    for role in sorted(phase4.SNAPSHOT_ROLES):
        destination = capture_root / "models" / role
        _capture_snapshot_role(
            manifest["roles"][role],
            Path(config["snapshot_dirs"][role]),
            destination,
        )
        captured_snapshots[role] = destination

    eligibility = components["eligibility"]
    captured_eligibility = capture_root / "pairing_eligibility.json"
    _capture_regular_file(
        Path(eligibility["artifact_path"]),
        captured_eligibility,
        expected_sha256=eligibility["artifact_sha256"],
        label="pairing eligibility artifact",
    )
    captured = {
        "root": capture_root,
        "data_dir": data_dir,
        "staged": captured_staged,
        "manifest": captured_manifest,
        "snapshots": captured_snapshots,
        "eligibility": captured_eligibility,
    }
    captured["snapshot"] = _captured_input_snapshot(capture_root)
    return captured


def _captured_input_snapshot(root: Path) -> dict[str, dict[str, Any]]:
    """Hash the complete private input tree without following aliases."""
    root = Path(root)
    snapshot: dict[str, dict[str, Any]] = {}
    for path in _closed_tree_regular_files(
        root,
        context="captured input",
        error_cls=LaunchRefusal,
    ):
        rel = path.relative_to(root).as_posix()
        snapshot[rel] = {
            "size": os.stat(path, follow_symlinks=False).st_size,
            "sha256": _sha256_regular_file(
                path, label=f"captured input {rel!r}"
            ),
        }
    if not snapshot:
        raise LaunchRefusal("captured input tree is empty (R-075/R-082)")
    return snapshot


def _closed_tree_regular_files(
    root: Path,
    *,
    context: str,
    error_cls: type[schema.ColmAimsError],
) -> list[Path]:
    """Return regular files after rejecting aliases before tree descent."""
    root = Path(root)

    def nofollow_info(path: Path, label: str) -> os.stat_result:
        try:
            info = os.stat(path, follow_symlinks=False)
        except OSError as exc:
            raise error_cls(
                f"{label} is missing or unreadable"
                f" ({exc.__class__.__name__}) (R-075/R-082)"
            ) from exc
        if stat.S_ISLNK(info.st_mode) or bool(
            getattr(info, "st_file_attributes", 0)
            & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
        ):
            raise error_cls(
                f"{label} is a symlink or reparse point (R-075/R-082)"
            )
        return info

    root_info = nofollow_info(root, f"{context} root")
    if not stat.S_ISDIR(root_info.st_mode):
        raise error_cls(f"{context} root is not a directory (R-075/R-082)")

    files: list[Path] = []

    def raise_walk_error(error: OSError) -> None:
        raise error_cls(
            f"{context} tree traversal failed"
            f" ({error.__class__.__name__}) (R-075/R-082)"
        ) from error

    for directory_name, child_directories, filenames in os.walk(
        root,
        topdown=True,
        onerror=raise_walk_error,
        followlinks=False,
    ):
        directory = Path(directory_name)
        directory_rel = directory.relative_to(root).as_posix()
        directory_label = (
            f"{context} root"
            if directory == root
            else f"{context} {directory_rel!r}"
        )
        directory_info = nofollow_info(directory, directory_label)
        if not stat.S_ISDIR(directory_info.st_mode):
            raise error_cls(
                f"{directory_label} is not a directory (R-075/R-082)"
            )
        for name in list(child_directories):
            child = directory / name
            rel = child.relative_to(root).as_posix()
            try:
                child_info = nofollow_info(child, f"{context} {rel!r}")
            except schema.ColmAimsError:
                child_directories.remove(name)
                raise
            if not stat.S_ISDIR(child_info.st_mode):
                child_directories.remove(name)
                raise error_cls(
                    f"{context} {rel!r} is not a directory (R-075/R-082)"
                )
        for name in filenames:
            path = directory / name
            rel = path.relative_to(root).as_posix()
            path_info = nofollow_info(path, f"{context} {rel!r}")
            if not stat.S_ISREG(path_info.st_mode):
                raise error_cls(
                    f"{context} {rel!r} is not a regular file"
                    " (R-075/R-082)"
                )
            files.append(path)
    return sorted(files, key=lambda path: path.relative_to(root).as_posix())


def _verify_captured_inputs(captured: dict[str, Any]) -> None:
    """Reverify the complete private input tree after producer exit."""
    observed = _captured_input_snapshot(Path(captured["root"]))
    if observed != captured["snapshot"]:
        raise LaunchRefusal(
            "captured input bytes or membership changed during producer"
            " execution (R-075/R-082)"
        )


def _release_captured_inputs_for_durable_sync(captured: dict[str, Any]) -> None:
    """Make verified private copies syncable by the shared Windows barrier.

    Captured inputs stay read-only throughout child execution.  After their
    post-child byte/membership verification, that permission is no longer an
    integrity boundary.  The shared ``fileio.fsync_tree`` implementation must
    open regular files read/write on Windows because the CRT rejects fsync on
    read-only descriptors, so release only the already-verified captured
    files immediately before the complete-tree durability barrier.
    """
    if os.name != "nt":
        return
    root = Path(captured["root"])
    for relative in sorted(captured["snapshot"]):
        (root / PurePosixPath(relative)).chmod(stat.S_IREAD | stat.S_IWRITE)


def _rewrite_argv_for_captured_inputs(
    argv: list[str], captured: dict[str, Any]
) -> list[str]:
    """Replace every mutable input binding with its private captured path."""
    replacements = {
        "--data-dir": captured["data_dir"],
        "--calibration": captured["staged"]["calibration_train"],
        "--eligibility": captured["eligibility"],
        "--snapshot-manifest": captured["manifest"],
        "--primary-model-path": captured["snapshots"]["primary_scorer"],
        "--disjoint-model-path": captured["snapshots"]["disjoint_selector"],
    }
    rewritten: list[str] = []
    index = 0
    while index < len(argv):
        token = argv[index]
        if token in replacements:
            rewritten.extend([token, str(replacements[token])])
            index += 2
            continue
        matched = next(
            (flag for flag in replacements if token.startswith(f"{flag}=")),
            None,
        )
        if matched is not None:
            rewritten.append(f"{matched}={replacements[matched]}")
            index += 1
            continue
        if token == "--staged-input":
            spec = argv[index + 1]
            label, _, remainder = spec.partition("=")
            _path, separator, digest = remainder.rpartition(":")
            if not separator or label not in captured["staged"]:
                raise LaunchRefusal(
                    f"cannot rewrite malformed staged-input {spec!r}"
                    " (R-081/R-082)"
                )
            rewritten.extend(
                [token, f"{label}={captured['staged'][label]}:{digest}"]
            )
            index += 2
            continue
        if token.startswith("--staged-input="):
            spec = token.partition("=")[2]
            label, _, remainder = spec.partition("=")
            _path, separator, digest = remainder.rpartition(":")
            if not separator or label not in captured["staged"]:
                raise LaunchRefusal(
                    f"cannot rewrite malformed staged-input {spec!r}"
                    " (R-081/R-082)"
                )
            rewritten.append(
                f"--staged-input={label}={captured['staged'][label]}:{digest}"
            )
            index += 1
            continue
        rewritten.append(token)
        index += 1
    return rewritten


def _remove_preledger_quarantine(quarantine_dir: Path) -> None:
    """Remove only this launcher's fresh, not-yet-claimed workspace."""
    if _path_lexists(quarantine_dir):
        shutil.rmtree(quarantine_dir)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_run_git(cmd: list[str]) -> str:
    """Production git runner: subprocess in the repository root."""
    if not cmd or str(cmd[0]) != "git":
        raise LaunchRefusal("git probe command must begin with 'git' (R-081)")
    candidates = (
        (
            Path("C:/Program Files/Git/cmd/git.exe"),
            Path("C:/Program Files/Git/bin/git.exe"),
            Path("C:/Program Files (x86)/Git/cmd/git.exe"),
        )
        if os.name == "nt"
        else (
            Path("/usr/bin/git"),
            Path("/opt/homebrew/bin/git"),
            Path("/usr/local/bin/git"),
        )
    )
    git_executable = next(
        (
            candidate
            for candidate in candidates
            if candidate.is_file() and not schema.is_filesystem_link(candidate)
        ),
        None,
    )
    if git_executable is None:
        raise LaunchRefusal("git executable cannot be resolved (R-081)")
    try:
        git_executable = str(git_executable.resolve(strict=True))
    except OSError as exc:
        raise LaunchRefusal("git executable cannot be resolved (R-081)") from exc
    completed = subprocess.run(
        [git_executable, *(str(part) for part in cmd[1:])],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        check=True,
        env=_sanitized_runtime_environment(),
    )
    return completed.stdout


def _default_launch(argv: list[str], env: dict[str, str]) -> int:
    """Production launcher: run the producer subprocess from the repo root
    (the certificate command carries repo-relative paths) and return its
    exit code."""
    completed = subprocess.run(
        [str(part) for part in argv], env=dict(env), cwd=str(_REPO_ROOT)
    )
    return completed.returncode


def _compose_argv(
    command: list[str], quarantine_dir: Path, activation_digest: str
) -> tuple[list[str], str | None]:
    """Compose the child argv FROM the certificate's recorded command.

    ONLY output paths are remapped: the ``--out`` value to
    ``quarantine_dir/<basename>`` and the ``--records-out`` value to the
    quarantine directory ITSELF (amended R-080: --records-out is the
    PARENT; the producer's exporter owns the ``records/`` segment).
    ``--certificate-digest <activation_digest>`` is appended. Every other
    token is preserved verbatim. Returns ``(argv, out_basename)``.
    """
    argv: list[str] = []
    out_basename: str | None = None
    index = 0
    total = len(command)
    while index < total:
        token = command[index]
        if token in ("--out", "--records-out"):
            if index + 1 >= total:
                raise LaunchRefusal(
                    f"certificate command carries a dangling {token} flag —"
                    " argv composition refused (R-081)"
                )
            value = command[index + 1]
            if token == "--out":
                out_basename = Path(value).name
                if not schema.is_path_component(out_basename):
                    raise LaunchRefusal(
                        "certificate command --out must name a regular"
                        " output file (R-081)"
                    )
                argv.extend([token, str(quarantine_dir / out_basename)])
            else:
                argv.extend([token, str(quarantine_dir)])
            index += 2
            continue
        if token.startswith("--out=") or token.startswith("--records-out="):
            flag, _, value = token.partition("=")
            if flag == "--out":
                out_basename = Path(value).name
                if not schema.is_path_component(out_basename):
                    raise LaunchRefusal(
                        "certificate command --out must name a regular"
                        " output file (R-081)"
                    )
                argv.append(f"--out={quarantine_dir / out_basename}")
            else:
                argv.append(f"--records-out={quarantine_dir}")
            index += 1
            continue
        argv.append(token)
        index += 1
    argv.extend(["--certificate-digest", activation_digest])
    return argv, out_basename


_EXPORTED_RECORD_ENTRY_KEYS = frozenset(
    {"path", "sha256", "n_items", "historical_cell", "policy"}
)


def _historical_cell_id(cell_id: str) -> str:
    reference_id, separator, calibration_id = cell_id.rpartition("__")
    if not separator:
        raise ComparatorValidationError(
            f"exported record cell id {cell_id!r} is malformed"
        )
    historical_calibration = (
        "performat" if calibration_id == "format_specific" else calibration_id
    )
    return f"{reference_id}+{historical_calibration}"


def _validate_exported_records(
    quarantine_dir: Path,
    regenerated: dict[str, Any],
    activation_digest: str,
    verified_eligibility: dict[str, Any],
) -> None:
    """Validate the complete zero-exit Phase-4 record export before parity."""
    metadata = regenerated.get("metadata")
    if not isinstance(metadata, dict):
        raise ComparatorValidationError(
            "regenerated export metadata is missing or malformed"
        )
    phase4_metadata = metadata.get("phase4")
    if not isinstance(phase4_metadata, dict):
        raise ComparatorValidationError(
            "regenerated export metadata.phase4 is missing or malformed"
        )
    observed_activation = phase4_metadata.get("certificate_digest")
    if observed_activation != activation_digest:
        raise ComparatorValidationError(
            "regenerated export metadata.phase4.certificate_digest"
            f" {observed_activation!r} != activation digest"
            f" {activation_digest!r}"
        )

    exported = phase4_metadata.get("exported_records")
    if not isinstance(exported, dict):
        raise ComparatorValidationError(
            "regenerated export metadata.phase4.exported_records is missing"
            " or malformed"
        )
    expected_cells = set(schema.CELL_IDS)
    observed_cells = set(exported)
    if observed_cells != expected_cells:
        missing = sorted(expected_cells - observed_cells)
        extra = sorted(observed_cells - expected_cells)
        raise ComparatorValidationError(
            "regenerated exported_records keyset is not the exact ten-cell"
            f" grid (missing={missing}, extra={extra})"
        )

    eligible_keys = verified_eligibility.get("eligible_keys")
    horizon_map = verified_eligibility.get("horizon_map")
    if (
        not isinstance(eligible_keys, list)
        or not isinstance(horizon_map, dict)
        or len(eligible_keys) != schema.EXPECTED_COMPLETE_PAIRS
    ):
        raise ComparatorValidationError(
            "pre-ledger verified eligibility object is malformed"
        )
    expected_item_keys = set(eligible_keys)
    quarantine_dir = Path(quarantine_dir).resolve()
    records_root = quarantine_dir / "records"
    if not records_root.is_dir() or records_root.is_symlink():
        raise ComparatorValidationError(
            "regenerated records directory is missing, malformed, or a"
            " symlink"
        )
    try:
        if os.path.normcase(str(records_root.resolve(strict=True))) != (
            os.path.normcase(str(records_root))
        ):
            raise ComparatorValidationError(
                "regenerated records directory resolves outside the"
                " quarantine records path"
            )
    except OSError as exc:
        raise ComparatorValidationError(
            "regenerated records directory cannot be resolved"
        ) from exc
    expected_record_names = {
        f"{cell_id}.jsonl" for cell_id in schema.CELL_IDS
    }
    try:
        observed_record_names = {child.name for child in records_root.iterdir()}
    except OSError as exc:
        raise ComparatorValidationError(
            "regenerated records directory cannot be enumerated"
        ) from exc
    if observed_record_names != expected_record_names:
        missing = sorted(expected_record_names - observed_record_names)
        extra = sorted(observed_record_names - expected_record_names)
        raise ComparatorValidationError(
            "regenerated records directory file set is not the exact"
            f" ten-cell export (missing={missing}, extra={extra})"
        )

    for cell_id in schema.CELL_IDS:
        entry = exported[cell_id]
        if not isinstance(entry, dict):
            raise ComparatorValidationError(
                f"exported_records[{cell_id!r}] must be an object"
            )
        unknown = sorted(set(entry) - _EXPORTED_RECORD_ENTRY_KEYS)
        missing = sorted(_EXPORTED_RECORD_ENTRY_KEYS - set(entry))
        if unknown or missing:
            raise ComparatorValidationError(
                f"exported_records[{cell_id!r}] has a non-closed shape"
                f" (missing={missing}, unknown={unknown})"
            )

        expected_record_path = records_root / f"{cell_id}.jsonl"
        raw_record_path = entry["path"]
        expected_metadata_path = f"records/{cell_id}.jsonl"
        if raw_record_path != expected_metadata_path:
            raise ComparatorValidationError(
                f"exported_records[{cell_id!r}].path must be the exact"
                f" artifact-relative path {expected_metadata_path!r}; found"
                f" {raw_record_path!r}"
            )

        declared_sha256 = entry["sha256"]
        if not schema.is_sha256_hex(declared_sha256):
            raise ComparatorValidationError(
                f"exported_records[{cell_id!r}].sha256 is malformed"
            )
        try:
            record_bytes = schema.read_regular_file_bytes(
                expected_record_path, tree_root=quarantine_dir
            )
        except schema.ColmAimsError as exc:
            raise ComparatorValidationError(
                f"record file {cell_id}.jsonl is unreadable: {exc}"
            ) from exc
        observed_sha256 = hashlib.sha256(record_bytes).hexdigest()
        if observed_sha256 != declared_sha256:
            raise ComparatorValidationError(
                f"record file {cell_id}.jsonl SHA-256 {observed_sha256} !="
                f" exported metadata {declared_sha256}"
            )

        n_items = entry["n_items"]
        if (
            not schema.is_real_int(n_items)
            or n_items != schema.EXPECTED_COMPLETE_PAIRS
        ):
            raise ComparatorValidationError(
                f"exported_records[{cell_id!r}].n_items {n_items!r} !="
                f" {schema.EXPECTED_COMPLETE_PAIRS}"
            )
        expected_historical = _historical_cell_id(cell_id)
        if entry["historical_cell"] != expected_historical:
            raise ComparatorValidationError(
                f"exported_records[{cell_id!r}].historical_cell"
                f" {entry['historical_cell']!r} != {expected_historical!r}"
            )
        if entry["policy"] != "dp":
            raise ComparatorValidationError(
                f"exported_records[{cell_id!r}].policy must be 'dp'"
            )

        try:
            loaded = schema.load_records_bytes(
                record_bytes, f"records/{cell_id}.jsonl"
            )
        except schema.ColmAimsError as exc:
            raise ComparatorValidationError(
                f"record file {cell_id}.jsonl failed strict load: {exc}"
            ) from exc
        records = loaded.get("records")
        if (
            not isinstance(records, list)
            or len(records) != schema.EXPECTED_COMPLETE_PAIRS
        ):
            count = len(records) if isinstance(records, list) else None
            raise ComparatorValidationError(
                f"record file {cell_id}.jsonl row count {count!r} !="
                f" {schema.EXPECTED_COMPLETE_PAIRS}"
            )
        observed_keys: set[str] = set()
        for row_index, record in enumerate(records):
            try:
                schema.validate_record(record)
            except schema.ColmAimsError as exc:
                raise ComparatorValidationError(
                    f"record file {cell_id}.jsonl row {row_index + 1}"
                    f" violates the canonical schema: {exc}"
                ) from exc
            outcome = pairing.classify_record(record)
            if outcome["status"] != "complete":
                raise ComparatorValidationError(
                    f"record file {cell_id}.jsonl row {row_index + 1}"
                    " is excluded rather than a complete pair"
                )
            item_key = record["item_key"]
            if item_key in observed_keys:
                raise ComparatorValidationError(
                    f"record file {cell_id}.jsonl duplicates an item_key"
                )
            observed_keys.add(item_key)
            if item_key not in expected_item_keys:
                raise ComparatorValidationError(
                    f"record file {cell_id}.jsonl carries an ineligible"
                    " item_key"
                )
            if record["trajectory_horizon"] != horizon_map.get(item_key):
                raise ComparatorValidationError(
                    f"record file {cell_id}.jsonl carries a trajectory"
                    " horizon that contradicts the verified eligibility"
                    " artifact"
                )
        if observed_keys != expected_item_keys:
            raise ComparatorValidationError(
                f"record file {cell_id}.jsonl item-key set is not exactly"
                " the verified eligible population"
            )


def _build_default_compare(
    verified_anchor: dict[str, Any],
    out_basename: str | None,
    activation_digest: str,
    verified_eligibility: dict[str, Any],
):
    """Compare regenerated output to the pre-ledger cached anchor.

    Only the regenerated export is read after the run.  The anchor object
    came from bytes whose digest was certificate-verified before the ledger,
    closing the prior post-launch anchor reread/TOCTOU gap.
    """
    if out_basename is None:
        raise LaunchRefusal(
            "certificate command carries no --out flag — the default"
            " comparator cannot locate the regenerated export (R-081)"
        )

    def compare(quarantine_dir: Path) -> dict[str, Any]:
        try:
            regenerated = schema.parse_json_bytes_strict(
                schema.read_regular_file_bytes(
                    Path(quarantine_dir) / out_basename,
                    tree_root=Path(quarantine_dir),
                )
            )
        except (schema.ColmAimsError, UnicodeDecodeError, ValueError) as exc:
            raise ComparatorValidationError(
                "regenerated export failed strict JSON ingress"
            ) from exc
        if not isinstance(regenerated, dict):
            raise ComparatorValidationError(
                "regenerated export must be a JSON object"
            )
        _validate_exported_records(
            Path(quarantine_dir),
            regenerated,
            activation_digest,
            verified_eligibility,
        )
        return phase4.compare_parity(verified_anchor, regenerated)

    return compare


def _snapshot_comparator_outputs(
    quarantine_dir: Path, out_basename: str
) -> dict[str, bytes]:
    """Retain the exact export/record bytes presented to the comparator."""
    quarantine_dir = Path(quarantine_dir)
    records_root = quarantine_dir / "records"
    expected_names = {f"{cell_id}.jsonl" for cell_id in schema.CELL_IDS}
    try:
        observed_names = {child.name for child in records_root.iterdir()}
    except OSError as exc:
        raise ComparatorValidationError(
            "comparator output records cannot be enumerated"
        ) from exc
    if observed_names != expected_names:
        raise ComparatorValidationError(
            "comparator output record file set or membership changed"
        )
    paths = {out_basename: quarantine_dir / out_basename}
    paths.update(
        {
            f"records/{cell_id}.jsonl": records_root / f"{cell_id}.jsonl"
            for cell_id in schema.CELL_IDS
        }
    )
    try:
        return {
            relative: schema.read_regular_file_bytes(
                path, tree_root=quarantine_dir
            )
            for relative, path in sorted(paths.items())
        }
    except schema.ColmAimsError as exc:
        raise ComparatorValidationError(
            "comparator output snapshot is unreadable"
        ) from exc


def _require_comparator_outputs_unchanged(
    quarantine_dir: Path,
    out_basename: str,
    expected: dict[str, bytes],
) -> None:
    """Reject any byte or membership drift from the compared snapshot."""
    if _snapshot_comparator_outputs(quarantine_dir, out_basename) != expected:
        raise ComparatorValidationError(
            "regenerated export or record bytes changed after the comparator"
        )


def _remove_private_promotion_tree(
    candidate: Path, quarantine_dir: Path
) -> None:
    """Best-effort cleanup of exactly one launcher-created private tree."""
    candidate = Path(candidate)
    quarantine_dir = Path(quarantine_dir)
    if not _path_lexists(candidate):
        return
    try:
        if (
            candidate.parent.resolve(strict=True)
            != quarantine_dir.resolve(strict=True)
            or candidate.is_symlink()
            or schema.is_filesystem_link(candidate)
            or not candidate.is_dir()
        ):
            return
        files = _closed_tree_regular_files(
            candidate,
            context="private promotion cleanup",
            error_cls=LaunchRefusal,
        )
        if os.name == "nt":
            for path in files:
                path.chmod(stat.S_IREAD | stat.S_IWRITE)
        shutil.rmtree(candidate)
    except BaseException:
        # Never mask the primary post-ledger failure. Any residue remains
        # inside the already-stale quarantine and is covered by its STOP.
        return


def _materialize_private_promotion_tree(
    quarantine_dir: Path,
    *,
    captured_inputs: dict[str, Any],
    activation_digest: str,
    ledger_path: Path,
    out_basename: str,
    comparator_result: dict[str, Any],
    comparator_output_snapshot: dict[str, bytes],
) -> Path:
    """Build the promoted tree without reusing producer-owned output paths.

    The producer is given only paths in ``quarantine_dir``. A descendant that
    survives the direct child can therefore retain handles to those files.
    Promotion instead uses a fresh, path-detached launcher-owned child whose
    export and records are created solely from the bytes retained before
    comparator execution. "Private" describes byte provenance and path
    ownership, not ACL, principal, or process isolation. The authenticated
    captured inputs are copied through held descriptors and checked against
    their original size/digest inventory.
    """
    quarantine_dir = Path(quarantine_dir)
    expected_outputs = {out_basename} | {
        f"records/{cell_id}.jsonl" for cell_id in schema.CELL_IDS
    }
    if set(comparator_output_snapshot) != expected_outputs:
        raise ComparatorValidationError(
            "retained comparator output snapshot has unexpected membership"
        )
    candidate = Path(
        tempfile.mkdtemp(
            prefix=PRIVATE_PROMOTION_PREFIX,
            dir=str(quarantine_dir),
        )
    )
    try:
        source_capture = Path(captured_inputs["root"])
        captured_snapshot = captured_inputs["snapshot"]
        candidate_capture = candidate / CAPTURED_INPUTS_DIRNAME
        for relative, metadata in sorted(captured_snapshot.items()):
            _capture_regular_file(
                source_capture / PurePosixPath(relative),
                candidate_capture / PurePosixPath(relative),
                expected_sha256=metadata["sha256"],
                expected_size=metadata["size"],
                label=f"private promotion input {relative!r}",
            )
        for relative, data in sorted(comparator_output_snapshot.items()):
            create_once_bytes(
                candidate / PurePosixPath(relative),
                data,
                exists_label="private promotion output",
            )
        _write_launch_receipt(
            candidate,
            activation_digest=activation_digest,
            ledger_path=ledger_path,
            out_basename=out_basename,
            comparator_result=comparator_result,
            comparator_output_snapshot=comparator_output_snapshot,
        )
        _release_captured_inputs_for_durable_sync(
            {"root": candidate_capture, "snapshot": captured_snapshot}
        )
        fileio.fsync_tree(candidate)
        _require_comparator_outputs_unchanged(
            candidate, out_basename, comparator_output_snapshot
        )
        if _captured_input_snapshot(candidate_capture) != captured_snapshot:
            raise ComparatorValidationError(
                "private promotion captured-input bytes or membership drifted"
            )
        expected_top_level = {
            CAPTURED_INPUTS_DIRNAME,
            LAUNCH_RECEIPT_NAME,
            "records",
            out_basename,
        }
        if {child.name for child in candidate.iterdir()} != expected_top_level:
            raise ComparatorValidationError(
                "private promotion tree has unexpected top-level membership"
            )
        return candidate
    except BaseException:
        _remove_private_promotion_tree(candidate, quarantine_dir)
        raise


def _write_stop_report(quarantine_dir: Path, payload: dict[str, Any]) -> None:
    quarantine_dir = Path(quarantine_dir)
    schema.stable_directory_chain(quarantine_dir, quarantine_dir)
    report_path = quarantine_dir / STOP_REPORT_NAME
    create_once_bytes(
        report_path,
        (
            json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
        ).encode("utf-8"),
        exists_label="Phase-4 STOP report",
    )


def _attempt_stop_report(
    retained_dir: Path, payload: dict[str, Any]
) -> str:
    """Best-effort diagnostic that never masks the primary run failure."""
    try:
        _write_stop_report(retained_dir, payload)
    except BaseException as exc:
        return f"STOP report publication failed ({exc.__class__.__name__})"
    return "STOP report written"


def _write_launch_receipt(
    quarantine_dir: Path,
    *,
    activation_digest: str,
    ledger_path: Path,
    out_basename: str,
    comparator_result: dict[str, Any],
    comparator_output_snapshot: dict[str, bytes],
) -> None:
    """Bind comparator-approved bytes before promotion, without accepting."""
    _require_comparator_outputs_unchanged(
        quarantine_dir, out_basename, comparator_output_snapshot
    )
    records_sha256 = {
        cell_id: hashlib.sha256(
            comparator_output_snapshot[f"records/{cell_id}.jsonl"]
        ).hexdigest()
        for cell_id in schema.CELL_IDS
    }
    export_sha256 = hashlib.sha256(
        comparator_output_snapshot[out_basename]
    ).hexdigest()
    ledger_sha256 = hashlib.sha256(
        schema.read_regular_file_bytes(ledger_path)
    ).hexdigest()
    payload = {
        "schema_version": schema.SCHEMA_VERSION,
        "receipt_type": "phase4_launch",
        "process_trust_model": phase4.PHASE4_PROCESS_TRUST_MODEL_ID,
        "activation_digest": activation_digest,
        "ledger_sha256": ledger_sha256,
        "producer_exit_code": 0,
        "comparator_verdict": "PASS",
        "comparator_checked": comparator_result.get("checked"),
        "export_basename": out_basename,
        "export_sha256": export_sha256,
        "records_sha256": records_sha256,
    }
    create_once_bytes(
        Path(quarantine_dir) / LAUNCH_RECEIPT_NAME,
        schema.encode_json(payload),
        exists_label="Phase-4 launch receipt",
    )


def _write_acceptance_marker(
    promote_to: Path, *, activation_digest: str
) -> None:
    """Make post-cleanup acceptance explicit at one terminal commit point.

    A durable negative guard precedes marker publication. The driver rejects
    any tree where that guard exists, even if exact marker bytes are visible.
    The guard is removed and its parent synced while no positive marker exists.
    The positive marker is then published by ``_commit_acceptance_marker`` as
    the final no-replace commit. A pre-existing destination is never adopted.
    """
    promote_to = Path(promote_to)
    receipt_bytes = schema.read_regular_file_bytes(
        promote_to / LAUNCH_RECEIPT_NAME, tree_root=promote_to
    )
    marker_bytes = schema.encode_json(
        {
            "schema_version": schema.SCHEMA_VERSION,
            "marker_type": "phase4_launch_accepted",
            "activation_digest": activation_digest,
            "launch_receipt_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
        }
    )
    pending_path = promote_to / ACCEPTANCE_PENDING_NAME
    marker_path = promote_to / ACCEPTANCE_MARKER_NAME
    pending_bytes = schema.encode_json(
        {
            "schema_version": schema.SCHEMA_VERSION,
            "marker_type": "phase4_launch_acceptance_pending",
            "activation_digest": activation_digest,
        }
    )
    create_once_bytes(
        pending_path,
        pending_bytes,
        exists_label="Phase-4 acceptance pending guard",
    )
    os.unlink(pending_path)
    fileio.fsync_directory(promote_to)
    _commit_acceptance_marker(marker_path, marker_bytes)


def _commit_acceptance_marker(marker_path: Path, marker_bytes: bytes) -> None:
    """Make marker visibility the final, content-durable acceptance commit.

    The temporary inode lives outside the promoted tree and is fully fsynced
    before the no-replace hard link. Before that link, marker absence is the
    rejection oracle. After it, the launcher has accepted the transaction;
    temporary cleanup and a best-effort directory sync cannot reverse that
    outcome. A crash may lose an unsynced directory entry, which is a safe
    false negative because the driver then observes no marker.
    """
    marker_path = Path(marker_path)
    if os.path.lexists(marker_path):
        raise FileExistsError(
            f"Phase-4 acceptance marker already exists: {marker_path}"
        )
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{marker_path.name}.acceptance-",
        dir=str(marker_path.parent.parent),
    )
    temporary = Path(temporary_name)
    committed = False
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(marker_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, marker_path)
        except FileExistsError as exc:
            raise FileExistsError(
                f"Phase-4 acceptance marker already exists: {marker_path}"
            ) from exc
        committed = True
    finally:
        try:
            os.unlink(temporary)
        except BaseException:
            # The temporary lives outside the promoted tree. Once the marker
            # link commits, cleanup cannot downgrade acceptance; before commit,
            # marker absence remains fail-closed.
            pass
    if committed:
        try:
            fileio.fsync_directory(marker_path.parent)
        except BaseException:
            # The fully-fsynced inode is already visible at the final path.
            # Treat the no-replace link as the acceptance commit; a crash may
            # lose the entry only as a safe false negative.
            pass


def _default_sync_parent_directory(parent: Path) -> None:
    """Durably publish a newly-created directory entry where supported."""
    # CPython exposes FlushFileBuffers for file descriptors on Windows but
    # has no portable way to open/fsync a directory handle.  The ledger file
    # itself is still fsynced there.  POSIX directory fsync is the strongest
    # supported create-once publication barrier.
    if os.name == "nt":
        return
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    directory_fd = os.open(str(Path(parent)), flags)
    try:
        try:
            os.fsync(directory_fd)
        except OSError as exc:
            unsupported = {
                errno.EINVAL,
                getattr(errno, "ENOTSUP", errno.EINVAL),
                getattr(errno, "EOPNOTSUPP", errno.EINVAL),
            }
            if exc.errno not in unsupported:
                raise
    finally:
        os.close(directory_fd)


def _durably_write_claimed_ledger(
    ledger_fd: int,
    payload: bytes,
    ledger_path: Path,
    *,
    write: Any,
    fsync: Any,
    sync_parent: Any,
) -> None:
    """Fully write/fsync an already O_EXCL-claimed ledger descriptor."""
    failure: BaseException | None = None
    try:
        view = memoryview(payload)
        offset = 0
        while offset < len(view):
            written = write(ledger_fd, view[offset:])
            remaining = len(view) - offset
            if (
                type(written) is not int
                or written <= 0
                or written > remaining
            ):
                raise OSError(
                    errno.EIO,
                    "exception-ledger write made invalid/no progress",
                )
            offset += written
        fsync(ledger_fd)
    except BaseException as exc:  # noqa: BLE001 - durability boundary
        failure = exc
    try:
        os.close(ledger_fd)
    except BaseException as exc:  # noqa: BLE001 - durability boundary
        if failure is None:
            failure = exc
    if failure is not None:
        raise failure
    sync_parent(Path(ledger_path).parent)


def validate_and_launch(
    config: dict[str, Any],
    *,
    run_git: Any = None,
    launch: Any = None,
    compare: Any = None,
    now: Any = None,
    resolve_executable: Any = None,
    host_identity: Any = None,
    probe_environment_lock: Any = None,
    ledger_write: Any = None,
    ledger_fsync: Any = None,
    ledger_parent_sync: Any = None,
) -> dict[str, Any]:
    """Single-use, fail-closed launch of the ONE authorized run (R-081).

    Pre-launch refusal classes, in execution order (``launch`` NEVER fires
    on any refusal): closed config shape; missing/pending/unbound certificate
    publication; accepted certificate bytes sha256 != activation digest
    (bytes before parsing); ``ready`` not identically True (bool-safe);
    command interpreter, host, or dependency lock drift; external config !=
    certificate-command artifact bindings;
    ambient provenance overrides; live commit/tree/tracked-clean mismatch;
    snapshot mismatch; comparator-anchor hash/parse failure; staged path
    containment or live-hash mismatch; stale/unusable workspace; and the
    create-once ledger via ``os.open(O_CREAT|O_EXCL)``.  Every check and all
    workspace materialization precede ledger consumption.

    Post-launch: nonzero exit -> ``RunFailed`` + STOP report in the intact
    quarantine, nothing promoted; zero exit -> comparator invoked
    MANDATORILY; PASS -> single atomic create-once quarantine promotion;
    comparator/promotion failure -> ``RunFailed`` + truthful STOP report at
    the retained pre- or post-commit output location.
    """
    _validate_config_shape(config)
    if run_git is None:
        run_git = _default_run_git
    if launch is None:
        launch = _default_launch
    if now is None:
        now = _utc_now_iso
    if resolve_executable is None:
        resolve_executable = _default_resolve_executable
    if host_identity is None:
        host_identity = _default_host_identity
    if probe_environment_lock is None:
        probe_environment_lock = _default_probe_environment_lock
    if ledger_write is None:
        ledger_write = os.write
    if ledger_fsync is None:
        ledger_fsync = os.fsync
    if ledger_parent_sync is None:
        ledger_parent_sync = _default_sync_parent_directory

    certificate_path = Path(config["certificate_path"])
    activation_digest = str(config["activation_digest"])

    # (1) Capture the certificate from a positively accepted, non-pending
    # publication.  The returned bytes are the authoritative snapshot bound
    # by the marker; never reopen the live certificate path after this gate.
    try:
        certificate_name = certificate_path.name
        if certificate_name == CERTIFICATE_GENERATION_SUMMARY_NAME:
            raise schema.TypedIngressError(
                "certificate path collides with the generation summary"
            )
        _, certificate_snapshot = (
            phase4_finalize_release._read_accepted_directory_snapshot(
                certificate_path.parent,
                "PRE_RUN_READY certificate publication",
                expected_names=(
                    certificate_name,
                    CERTIFICATE_GENERATION_SUMMARY_NAME,
                ),
            )
        )
        certificate_bytes = certificate_snapshot[certificate_name]
    except (OSError, schema.ColmAimsError) as exc:
        raise LaunchRefusal(
            "certificate publication acceptance check failed — exact"
            f" certificate bytes unavailable: {exc} (R-081)"
        ) from exc

    # Activation digest over the accepted RAW BYTES — before any parse,
    # before any semantic check (bytes first, semantics second).
    observed_digest = hashlib.sha256(certificate_bytes).hexdigest()
    if observed_digest != activation_digest:
        raise LaunchRefusal(
            f"certificate digest mismatch: certificate bytes sha256"
            f" {observed_digest} != activation digest {activation_digest}"
            " (R-081)"
        )

    # (2) Hardened parse + ready identity (bool-safe: 1 is not True).
    try:
        certificate = schema.parse_json_bytes_strict(certificate_bytes)
    except (schema.ColmAimsError, ValueError) as exc:
        raise LaunchRefusal(
            f"certificate parse failed: {exc} (R-081)"
        ) from exc
    if not isinstance(certificate, dict):
        raise LaunchRefusal(
            "certificate must be a JSON object (R-081)"
        )
    if certificate.get("ready") is not True:
        raise LaunchRefusal(
            "certificate ready is not identically True (found"
            f" {certificate.get('ready')!r}) — a non-ready certificate"
            " never launches (R-081)"
        )

    components = _validate_certificate_envelope_and_semantics(
        certificate, certificate.get("components")
    )
    _verify_canonical_qa012(components)
    repo = components.get("repo")
    repo = repo if isinstance(repo, dict) else {}
    certificate_commit = repo.get("commit")
    certificate_tree = repo.get("tree_sha256")
    bound_content_relpaths = _verify_content_hashes(repo, components)
    environment = components.get("environment")
    environment = environment if isinstance(environment, dict) else {}
    command = environment.get("command")
    if (
        not isinstance(command, list)
        or not command
        or not all(isinstance(token, str) and token for token in command)
    ):
        raise LaunchRefusal(
            "certificate components.environment.command missing or"
            " malformed — argv must be composed FROM the certificate"
            " (R-081)"
        )

    # These mutable paths are certificate-owned.  Re-prove their external,
    # disjoint topology and exact launcher-config equality before any
    # workspace materialization; in particular, callers cannot substitute a
    # fresh ledger to reuse an already-consumed activation.
    quarantine_dir, promote_to, ledger_path = (
        _validate_launch_workspace_bindings(config, repo, environment)
    )

    # Defense in depth for a hand-crafted ``ready: true`` certificate: the
    # pure certificate assembler normally enforces this exact seven-input
    # contract, but the single-use launcher re-proves it independently so an
    # omitted label/operator binding or split-selector drift cannot burn the
    # ledger in the producer's own pre-model gate.
    _validate_staged_coverage(repo, components, command)
    _validate_phase4_command(repo, environment, command)
    # Refuse interpreter-control variables before invoking even the
    # certified interpreter for its dependency-lock probe.  The process
    # running this module has already started, but no launcher subprocess
    # is permitted to inherit import-shadowing controls.
    _validate_ambient_environment()
    _validate_untracked_disclosure(
        repo, bound_content_relpaths, run_git=run_git
    )
    # Prove the signed commit/tree before invoking the certified Python for
    # its dependency probe: a different clean checkout could otherwise put
    # an import shadow such as ``pip.py`` at the repository-root cwd.
    live_commit = str(run_git(["git", "rev-parse", "HEAD"])).strip()
    if live_commit != certificate_commit:
        raise LaunchRefusal(
            f"live repository commit {live_commit!r} != certificate commit"
            f" {certificate_commit!r} (R-081)"
        )
    live_tree = str(run_git(["git", "rev-parse", "HEAD^{tree}"])).strip()
    if live_tree != certificate_tree:
        raise LaunchRefusal(
            f"live repository tree {live_tree!r} != certificate tree"
            f" {certificate_tree!r} (R-081)"
        )

    # The command-owned eligibility artifact is itself a frozen pin surface,
    # not merely another path argument.  Verify its raw bytes, strict schema,
    # keyset/horizon/test pins, and cache the result for post-run record
    # validation before the one-use ledger can be claimed.
    verified_eligibility = _load_verified_eligibility(components)

    # The executable and platform are certificate-owned runtime identity,
    # not ambient choices.  This check precedes git, snapshot loads, every
    # workspace effect, and the single-use ledger.
    command_interpreter = _validate_runtime_binding(
        environment,
        command,
        resolve_executable=resolve_executable,
        host_identity=host_identity,
    )
    _validate_environment_lock(
        environment,
        command_interpreter,
        probe_environment_lock=probe_environment_lock,
    )

    # External launcher config may locate certificate-owned artifacts but
    # may not substitute them.  All command-relative paths resolve using
    # the producer's actual repo-root cwd.
    _validate_external_config_bindings(config, command)

    # (5) Live TRACKED-dirty state; untracked-only "??" porcelain lines are
    # the signed tracked-clean + untracked-disclosure convention.
    status_output = str(
        run_git(["git", "status", "--porcelain", "--untracked-files=no"])
    )
    tracked_lines = [
        line
        for line in status_output.splitlines()
        if line.strip() and not line.startswith("??")
    ]
    if tracked_lines:
        raise LaunchRefusal(
            "live repository is tracked-dirty"
            f" ({tracked_lines[:5]!r}) — the certified clean-state proof no"
            " longer holds (R-081)"
        )

    # (7) Snapshot-manifest bytes and path are certificate-owned.  Load the
    # already raw-hash-verified object, then re-verify both external snapshot
    # trees against its per-file identities.
    manifest = _load_verified_snapshot_manifest(
        config, repo, components, command
    )
    snapshot_dirs = config["snapshot_dirs"]
    snapshot_dirs = snapshot_dirs if isinstance(snapshot_dirs, dict) else {}
    for role in sorted(phase4.SNAPSHOT_ROLES):
        snapshot_dir = snapshot_dirs.get(role)
        if snapshot_dir is None:
            raise LaunchRefusal(
                f"snapshot directory for role {role!r} is not configured"
                " (R-081)"
            )
        try:
            phase4.verify_snapshot_dir(
                manifest["roles"][role], Path(snapshot_dir)
            )
        except schema.ColmAimsError as exc:
            raise LaunchRefusal(
                f"snapshot re-verification failed for role {role!r}: {exc}"
                " (R-081)"
            ) from exc

    # The comparator anchor is an external config path, so rehash its bytes
    # against the certificate and parse/cache it before it can influence the
    # parity decision.  It is never reread after ledger consumption.
    verified_anchor = _load_verified_anchor(config, components)

    # Argv/env composition + default comparator resolution happen BEFORE
    # the ledger so every composition defect refuses without consuming the
    # single-use exception.
    argv, out_basename = _compose_argv(
        command, quarantine_dir, activation_digest
    )

    # (F-1, R-082) Staged inputs must live OUTSIDE the repository tree —
    # an in-repo untracked staged file passes every other gate, then trips
    # the producer's committed-writer git-pathspec guard AFTER scoring and
    # burns the single-use exception (the P0-1 trap, the exact defect of
    # rejected certificate 8731ad00). Refused here, pre-ledger, from BOTH
    # sources: the certificate's staged_inputs component and every staged
    # path the composed argv would hand the child (relative forms resolve
    # against the child's cwd, the repo root).
    staged_candidates: list[str] = []
    for entry in components.get("staged_inputs") or []:
        if isinstance(entry, dict) and entry.get("path"):
            staged_candidates.append(str(entry["path"]))
    for i, token in enumerate(argv):
        if token == "--staged-input":
            if i + 1 >= len(argv):
                raise LaunchRefusal(
                    "certificate command carries a dangling --staged-input"
                    " flag (R-081/R-082)"
                )
            staged_candidates.append(_staged_input_path(str(argv[i + 1])))
        elif token.startswith("--staged-input="):
            staged_candidates.append(
                _staged_input_path(token.partition("=")[2])
            )
        elif token in ("--calibration", "--data-dir"):
            if i + 1 >= len(argv) or not argv[i + 1]:
                raise LaunchRefusal(
                    f"certificate command carries a dangling {token} flag"
                    " (R-081/R-082)"
                )
            staged_candidates.append(str(argv[i + 1]))
        elif token.startswith("--calibration=") or token.startswith(
            "--data-dir="
        ):
            value = token.partition("=")[2]
            if not value:
                raise LaunchRefusal(
                    f"certificate command carries an empty {token} value"
                    " (R-081/R-082)"
                )
            staged_candidates.append(value)
    for raw_path in staged_candidates:
        if not raw_path:
            continue
        resolved = Path(raw_path)
        if not resolved.is_absolute():
            resolved = _REPO_ROOT / resolved
        if schema.resolves_inside(resolved, _REPO_ROOT):
            raise LaunchRefusal(
                f"staged input {raw_path!r} resolves inside the repository"
                " tree — staged inputs live OUTSIDE the repo (identity by"
                " hash, never location; R-082, the P0-1 ledger-burn trap)"
            )
    _verify_staged_input_hashes(components)
    if compare is None:
        compare = _build_default_compare(
            verified_anchor,
            out_basename,
            activation_digest,
            verified_eligibility,
        )
    child_env = _sanitized_runtime_environment()

    # (9) Workspace — fully materialized BEFORE the ledger (F-2: no
    # workspace defect may consume the single-use exception; the mkdir's
    # exist_ok=False doubles as the staleness check, and an unwritable
    # parent refuses here instead of burning the ledger).
    if _path_lexists(promote_to):
        raise LaunchRefusal(
            f"promote destination {promote_to} already exists — stale"
            " workspace; promotion is a single atomic rename (R-081)"
        )
    if not promote_to.parent.is_dir():
        raise LaunchRefusal(
            f"promote destination parent {promote_to.parent} does not exist"
            " — pre-flight refused so a PASS can always promote (R-081)"
        )
    if _path_lexists(ledger_path):
        raise LaunchRefusal(
            f"exception ledger {ledger_path} already exists — the"
            " single-use exception was already consumed; no second run"
            " without a new recorded amendment (R-081)"
        )
    if _path_lexists(quarantine_dir):
        raise LaunchRefusal(
            f"quarantine directory {quarantine_dir} already exists — stale"
            " workspace; the run writes into a FRESH quarantine (R-081)"
        )
    try:
        quarantine_dir.mkdir(parents=False, exist_ok=False)
        _created_quarantine = True
    except FileExistsError as exc:
        raise LaunchRefusal(
            f"quarantine directory {quarantine_dir} already exists — stale"
            " workspace; the run writes into a FRESH quarantine (R-081)"
        ) from exc
    except OSError as exc:
        raise LaunchRefusal(
            f"quarantine directory {quarantine_dir} cannot be created"
            f" ({exc.__class__.__name__}) — workspace refused pre-ledger"
            " (R-081/F-2)"
        ) from exc
    if quarantine_dir.stat().st_dev != promote_to.parent.stat().st_dev:
        quarantine_dir.rmdir()
        raise LaunchRefusal(
            "quarantine and promote destination live on different devices —"
            " os.rename cannot promote atomically across devices (R-081)"
        )

    # Capture every path-based child input before claiming the one-use
    # ledger.  Each private copy is hashed while read through the same held
    # source descriptor, and argv is then rewritten to address only those
    # copies.  Mutating or replacing the original staged/model paths after
    # this point cannot alter the producer's input bytes.
    try:
        captured_inputs = _capture_verified_inputs(
            quarantine_dir, config, components, manifest
        )
        argv = _rewrite_argv_for_captured_inputs(argv, captured_inputs)
    except BaseException:
        _remove_preledger_quarantine(quarantine_dir)
        raise

    # (8) Single-use consumption: CREATE-ONCE ledger via O_CREAT|O_EXCL,
    # recording the activation digest BEFORE launch.
    ledger_payload = (
        json.dumps(
            {
                "activation_digest": activation_digest,
                "certificate_path": str(certificate_path),
                "certificate_commit": certificate_commit,
                "certificate_tree": certificate_tree,
                "argv": argv,
                "consumed_at": now(),
            },
            indent=2,
            sort_keys=True,
            default=str,
        )
        + "\n"
    )
    try:
        ledger_fd = os.open(
            str(ledger_path),
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_BINARY", 0),
            0o644,
        )
    except FileExistsError as exc:
        # Ledger refusals stay side-effect-free: remove the private input
        # envelope this call just created.
        if _created_quarantine:
            _remove_preledger_quarantine(quarantine_dir)
        raise LaunchRefusal(
            f"exception ledger {ledger_path} already exists — the"
            " single-use exception was already consumed; no second run"
            " without a new recorded amendment (R-081)"
        ) from exc
    except OSError as exc:
        if _created_quarantine:
            _remove_preledger_quarantine(quarantine_dir)
        raise LaunchRefusal(
            f"exception ledger {ledger_path} is unwritable"
            f" ({exc.__class__.__name__}) (R-081)"
        ) from exc
    try:
        _durably_write_claimed_ledger(
            ledger_fd,
            ledger_payload.encode("utf-8"),
            ledger_path,
            write=ledger_write,
            fsync=ledger_fsync,
            sync_parent=ledger_parent_sync,
        )
    except BaseException as exc:
        # O_EXCL succeeded, so the exception is consumed even if only a
        # prefix (or zero bytes) reached disk.  Never unlink/retry: preserve
        # the claimed ledger and quarantine as forensic evidence and stop
        # before launch.
        stop_detail = _attempt_stop_report(
            quarantine_dir,
            {
                "reason": "ledger_write_failure",
                "error": f"{exc.__class__.__name__}: {exc}",
                "activation_digest": activation_digest,
                "ledger_path": str(ledger_path),
                "stopped_at": now(),
            },
        )
        raise RunFailed(
            "exception ledger was claimed but could not be durably written"
            f" ({exc.__class__.__name__}) — launch blocked; partial ledger"
            f" preserved at {ledger_path}, quarantine left intact at"
            f" {quarantine_dir}; {stop_detail} (R-081)"
        ) from exc

    # Launch EXACTLY once. A crash inside the launch callable still gets a
    # STOP report (F-3): the ledger is consumed, so the triage artifact
    # must exist precisely on the messiest failures.
    try:
        exit_code = launch(list(argv), dict(child_env))
    except BaseException as exc:
        stop_detail = _attempt_stop_report(
            quarantine_dir,
            {
                "reason": "launch_crash",
                "error": f"{exc.__class__.__name__}: {exc}",
                "activation_digest": activation_digest,
                "argv": argv,
                "stopped_at": now(),
            },
        )
        raise RunFailed(
            f"producer launch crashed ({exc.__class__.__name__}) —"
            f" quarantine left intact at {quarantine_dir}; {stop_detail};"
            " nothing promoted (R-081)"
        ) from exc
    if type(exit_code) is not int or exit_code != 0:
        stop_detail = _attempt_stop_report(
            quarantine_dir,
            {
                "reason": "nonzero_exit",
                "exit_code": exit_code,
                "activation_digest": activation_digest,
                "argv": argv,
                "stopped_at": now(),
            },
        )
        raise RunFailed(
            f"producer run exited nonzero ({exit_code!r}) — quarantine left"
            f" intact at {quarantine_dir}; {stop_detail}; nothing"
            " promoted (R-081)"
        )

    # The child runs as the same user on supported local platforms, so chmod
    # is not an immutability boundary. Catch every persistent mutation of the
    # private captured inputs before comparator execution or promotion.
    try:
        _verify_captured_inputs(captured_inputs)
    except BaseException as exc:
        stop_detail = _attempt_stop_report(
            quarantine_dir,
            {
                "reason": "captured_input_drift",
                "error": f"{exc.__class__.__name__}: {exc}",
                "activation_digest": activation_digest,
                "stopped_at": now(),
            },
        )
        raise RunFailed(
            "captured producer inputs changed during execution — promotion"
            f" blocked; quarantine left intact at {quarantine_dir};"
            f" {stop_detail} (R-075/R-081/R-082)"
        ) from exc

    # Mandatory comparator on a zero exit. A comparator crash gets a STOP
    # report too (F-3) — fail-closed with the triage artifact present.
    try:
        if out_basename is None:
            raise ComparatorValidationError(
                "accepted launch has no bound export basename"
            )
        comparator_output_snapshot = _snapshot_comparator_outputs(
            quarantine_dir, out_basename
        )
        result = compare(quarantine_dir)
        _require_comparator_outputs_unchanged(
            quarantine_dir, out_basename, comparator_output_snapshot
        )
    except BaseException as exc:
        stop_detail = _attempt_stop_report(
            quarantine_dir,
            {
                "reason": "comparator_crash",
                "error": f"{exc.__class__.__name__}: {exc}",
                "activation_digest": activation_digest,
                "stopped_at": now(),
            },
        )
        raise RunFailed(
            f"parity comparator crashed ({exc.__class__.__name__}) —"
            f" promotion blocked; quarantine left intact at"
            f" {quarantine_dir}; {stop_detail} (R-081)"
        ) from exc
    result = result if isinstance(result, dict) else {}
    if result.get("verdict") == "PASS":
        # Single create-once atomic promotion.  Bare POSIX ``os.rename`` may
        # silently replace a destination directory that appears empty in the
        # exists-check race window.  The shared primitive first atomically
        # claims that slot on POSIX and uses Windows' direct no-replace rename,
        # so every incumbent destination fails closed.
        promotion_claim_owned = False
        promotion_candidate: Path | None = None
        publication_returned = False
        cleanup_completed = False
        pass_result = {
            "promoted_to": str(promote_to),
            "activation_digest": activation_digest,
            "exit_code": 0,
            "verdict": "PASS",
            "argv": argv,
        }

        def _mark_promotion_claim_owned() -> None:
            nonlocal promotion_claim_owned
            promotion_claim_owned = True

        try:
            promotion_candidate = _materialize_private_promotion_tree(
                quarantine_dir,
                captured_inputs=captured_inputs,
                activation_digest=activation_digest,
                ledger_path=ledger_path,
                out_basename=out_basename,
                comparator_result=result,
                comparator_output_snapshot=comparator_output_snapshot,
            )
            publish_dir_create_once(
                promotion_candidate,
                promote_to,
                exists_label="Phase-4 promotion destination",
                claim_created=_mark_promotion_claim_owned,
            )
            publication_returned = True
            # Only the detached, fully-synced candidate is accepted. Remove
            # the producer-owned quarantine before reporting PASS; on Windows
            # an unquiesced descendant handle can make this fail, which is a
            # truthful post-commit STOP rather than a false acceptance.
            _release_captured_inputs_for_durable_sync(captured_inputs)
            shutil.rmtree(quarantine_dir)
            fileio.fsync_directory(quarantine_dir.parent)
            cleanup_completed = True
            # LAUNCH_RECEIPT.json is a pre-acceptance byte binding.  Only this
            # terminal, post-cleanup positive marker makes the launch usable.
            # No scientific or filesystem prerequisite may follow it.
            _write_acceptance_marker(
                promote_to, activation_digest=activation_digest
            )
        except BaseException as exc:
            # The shared primitive's commit point is the directory rename;
            # parent-directory fsync happens afterward.  If that durability
            # barrier fails, the complete output already belongs to the
            # destination.  Record that truthful post-commit state *there*;
            # never recreate an empty quarantine or claim nothing promoted.
            promotion_committed = (
                promotion_candidate is not None
                and not _path_lexists(promotion_candidate)
                and _path_lexists(promote_to)
            )
            report_dir = promote_to if promotion_committed else quarantine_dir
            stop_report_error: BaseException | None = None
            stop_report_written = False
            if report_dir.is_dir() and not report_dir.is_symlink():
                try:
                    _write_stop_report(
                        report_dir,
                        {
                            "reason": (
                                "acceptance_marker_failure"
                                if promotion_committed and cleanup_completed
                                else (
                                    "post_promotion_cleanup_failure"
                                    if promotion_committed
                                    and publication_returned
                                    else (
                                        "promotion_durability_failure"
                                        if promotion_committed
                                        else "promotion_crash"
                                    )
                                )
                            ),
                            "error": f"{exc.__class__.__name__}: {exc}",
                            "activation_digest": activation_digest,
                            "promotion_committed": promotion_committed,
                            "stopped_at": now(),
                        },
                    )
                    stop_report_written = True
                except BaseException as report_exc:
                    stop_report_error = report_exc
            stop_detail = (
                "; STOP report publication also failed"
                f" ({stop_report_error.__class__.__name__})"
                if stop_report_error is not None
                else (
                    "; STOP report written"
                    if stop_report_written
                    else "; STOP report directory unavailable"
                )
            )
            if promotion_committed:
                raise RunFailed(
                    "atomic private promotion committed, but destination"
                    " durability or producer-quarantine cleanup failed, or"
                    " terminal acceptance-marker publication failed"
                    f" ({exc.__class__.__name__}) —"
                    f" output remains at {promote_to}{stop_detail} and"
                    " is not an accepted PASS (R-081)"
                ) from exc
            if promotion_candidate is not None:
                _remove_private_promotion_tree(
                    promotion_candidate, quarantine_dir
                )
            # Nothing was promoted and the quarantine is intact. When OUR own
            # create-once claim (POSIX ``os.mkdir``) succeeded but the ensuing
            # rename failed, an EMPTY ``promote_to`` relic is left behind that
            # contradicts this path's "nothing promoted" contract — reclaim it.
            # A ``FileExistsError`` instead means the destination pre-existed (a
            # peer's incumbent claim we do not own): never remove that.
            # ``reclaim_empty_relic`` is itself empty-only and best-effort safe.
            if (
                promotion_claim_owned
                and isinstance(exc, OSError)
                and not isinstance(exc, FileExistsError)
            ):
                reclaim_empty_relic(promote_to)
            raise RunFailed(
                f"atomic promotion failed ({exc.__class__.__name__}) —"
                f" quarantine left intact at {quarantine_dir}{stop_detail};"
                " nothing promoted (R-081)"
            ) from exc
        return pass_result
    stop_detail = _attempt_stop_report(
        quarantine_dir,
        {
            "reason": "parity_comparator_fail",
            "verdict": result.get("verdict"),
            "checked": result.get("checked"),
            "failures": result.get("failures"),
            "activation_digest": activation_digest,
            "stopped_at": now(),
        },
    )
    raise RunFailed(
        f"parity comparator verdict {result.get('verdict')!r} — promotion"
        f" blocked; quarantine left intact at {quarantine_dir}; {stop_detail}"
        " (R-081)"
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m reproducibility.colm_aims_2026.phase4_launcher",
        description=(
            "Certificate-bound, single-use Phase-4 launcher. The JSON"
            " config surface is closed and every refusal is fail-closed."
        ),
        epilog=(
            "Threat boundary: this integrity workflow is not a sandbox; it"
            " requires a cooperative same-identity environment and no"
            " surviving producer descendants."
        ),
        allow_abbrev=False,
    )
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="path to the strict launcher JSON configuration",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the strict CLI and return its explicit exit-code outcome."""
    args = _build_parser().parse_args(argv)
    try:
        config = _load_launcher_config(args.config)
        result = validate_and_launch(config)
    except RunFailed as exc:
        print(f"RunFailed: {exc}", file=sys.stderr)
        return EXIT_RUN_FAILED
    except LaunchRefusal as exc:
        print(f"LaunchRefusal: {exc}", file=sys.stderr)
        return EXIT_LAUNCH_REFUSAL
    except schema.ColmAimsError as exc:
        # Any other namespace-typed failure is still a pre-launch refusal at
        # this boundary; name its type and never emit a traceback.
        print(f"{exc.__class__.__name__}: {exc}", file=sys.stderr)
        return EXIT_LAUNCH_REFUSAL
    except Exception as exc:  # noqa: BLE001 - explicit internal outcome
        print(
            f"InternalError: unexpected {exc.__class__.__name__}: {exc}",
            file=sys.stderr,
        )
        return EXIT_INTERNAL_ERROR

    try:
        rendered = json.dumps(
            result, indent=2, sort_keys=True, allow_nan=False
        )
    except (TypeError, ValueError) as exc:
        print(
            f"InternalError: PASS result was not strict JSON: {exc}",
            file=sys.stderr,
        )
        return EXIT_INTERNAL_ERROR
    print(rendered)
    return EXIT_PASS


if __name__ == "__main__":  # pragma: no cover - exercised by subprocess
    raise SystemExit(main())
