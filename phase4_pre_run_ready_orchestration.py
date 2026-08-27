"""Generate a Windows-host Phase-4 PRE_RUN_READY certificate.

This checked-in Windows-host orchestrator is source-only: it verifies the seven
external inputs and two materialized snapshot trees, executes the focused and
full test suites at one clean committed HEAD, writes head-bound receipts, and
assembles a replacement certificate. It never constructs or runs a model.

Usage::

    python phase4_pre_run_ready_orchestration.py \
        --run-root PATH --run-id RUN_ID [--asset-root PATH]

The run root is an operator-prepared, empty external workspace.  A run ID is
used only as a portable filename component; it cannot select a parent path.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from reproducibility.colm_aims_2026 import (  # noqa: E402
    phase4,
    phase4_launcher,
    receipt as receipt_module,
    schema,
)
from scripts.stopdff_v5 import fileio  # noqa: E402

DEFAULT_ASSET_ROOT = Path.home() / "qanta_phase4_windows_v3"
_RUN_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")

MAX_STAGED_INPUT_BYTES = 512 * 1024 * 1024
MAX_STAGED_TOTAL_BYTES = 2 * 1024 * 1024 * 1024
MAX_SNAPSHOT_FILE_BYTES = 2 * 1024 * 1024 * 1024
MAX_SNAPSHOT_TOTAL_BYTES = 4 * 1024 * 1024 * 1024
MAX_SUITE_TRANSCRIPT_BYTES = 64 * 1024 * 1024
MAX_JUNIT_BYTES = 64 * 1024 * 1024
SUITE_TIMEOUT_SECONDS = {
    "focused": 2 * 60 * 60,
    "full": 6 * 60 * 60,
}
SUITE_READER_JOIN_TIMEOUT_SECONDS = 30.0

_WINDOWS_SUITE_SUPERVISOR = """\
import json
import subprocess
import sys

if sys.stdin.buffer.read(1) != b"R":
    raise SystemExit(125)
command = json.loads(sys.argv[1])
raise SystemExit(
    subprocess.run(command, stdin=subprocess.DEVNULL, check=False).returncode
)
"""

_UNTRACKED_EXECUTABLE_SUFFIXES = frozenset(
    {
        ".py",
        ".pyi",
        ".pyc",
        ".pth",
        ".pyd",
        ".so",
        ".dll",
        ".exe",
        ".sh",
        ".bat",
        ".cmd",
        ".ps1",
    }
)
_UNTRACKED_PYTEST_CONFIGS = frozenset(
    {"pytest.ini", "pyproject.toml", "setup.cfg", "tox.ini"}
)

FROZEN = REPO / "reproducibility" / "colm_aims_2026" / "frozen"
ELIGIBILITY_PATH = FROZEN / "pairing_eligibility_v2.json"
SNAPSHOT_MANIFEST_PATH = FROZEN / "model_snapshot_manifests.json"
PARITY_ANCHOR_PATH = FROZEN / "parity_anchor_export_a.json"
QA012_REV3_PATH = REPO / "qa012_inventory_2026-08-22_rev3.json"
WORKFLOW_PATH = REPO / ".github" / "workflows" / "python-app.yml"

EXPECTED_STAGED = {
    "calibration_train": (
        "calibration_train.json",
        "745bd67597278bd9d24d41c1dea53bf3a7c56cd6334cfc07ea62bccbdcf44259",
    ),
    "eval_split": (
        "test_dataset.json",
        "638a4df978b77a12655ea72d56daad7fa70851ae486ddb4365d9b060549e34f1",
    ),
    "fit_split": (
        "val_dataset.json",
        "9b7a131b6c94c446e6b40b95559cb62aeee63f6e6f29ddd1d7ed3fb19cc72c65",
    ),
    "mc_dataset": (
        "mc_dataset.json",
        "3dbebf8e4d690da41a15e3cf467e57fdbe69af420ed831d56b61160af8bf7946",
    ),
    "answer_profiles": (
        "answer_profiles.json",
        "635586393ad36cf7e0726066bc242d97d0f982abd6108e4d8b87a3cf4598fc75",
    ),
    "build_metadata": (
        "build_metadata.json",
        "70871984390f252c0a06a5a2c9a2d3b4337f10ad48c87583ebec215d5c0c9c6e",
    ),
    "split_metadata": (
        "split_metadata.json",
        "b67bcdbb937411c7e14f9a3bfa9fd1ab0f7ed6956458978da0d148e65c246b39",
    ),
}

FOCUSED_TESTS = [
    "tests/test_colm_aims_v2_phase4_pre.py",
    "tests/test_colm_aims_v2_schema_raw_bytes.py",
    "tests/test_phase4_build_metadata_staging.py",
    "tests/test_phase4_certificate_external_staging.py",
    "tests/test_phase4_launcher_cli.py",
    "tests/test_stopdff_v5_fileio_windows.py",
    "tests/test_stopdff_v5_windows_control_plane.py",
]


@dataclass(frozen=True)
class OrchestrationPaths:
    """Canonical external paths for one PRE_RUN_READY attempt."""

    asset_root: Path
    run_root: Path
    staged_data: Path
    snapshot_root: Path
    receipts_dir: Path
    certificate_dir: Path
    certificate_path: Path
    quarantine_dir: Path
    promote_to: Path
    exception_ledger_path: Path


def _canonical_absolute(path: Path, label: str) -> Path:
    """Resolve an absolute operator path without accepting CWD ambiguity."""

    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        raise RuntimeError(f"{label} must be an absolute path: {candidate}")
    try:
        resolved = candidate.resolve()
    except (OSError, RuntimeError, ValueError) as exc:
        raise RuntimeError(f"{label} cannot be resolved: {candidate}") from exc
    if os.path.normcase(str(candidate)) != os.path.normcase(str(resolved)):
        raise RuntimeError(
            f"{label} must be a canonical path without links or dot segments:"
            f" {candidate}"
        )
    return resolved


def derive_paths(
    *, asset_root: Path, run_root: Path, run_id: str
) -> OrchestrationPaths:
    """Derive the closed operational path surface for one fresh run."""

    if (
        not schema.is_path_component(run_id)
        or not _RUN_ID_RE.fullmatch(run_id)
        or run_id.endswith(".")
    ):
        raise RuntimeError(
            "run ID must be a portable 1-128 character component beginning"
            " with an alphanumeric and containing only alphanumerics, '.',"
            " '_', or '-'"
        )
    asset = _canonical_absolute(asset_root, "asset root")
    run = _canonical_absolute(run_root, "run root")
    return OrchestrationPaths(
        asset_root=asset,
        run_root=run,
        staged_data=asset / "staging" / "processed",
        snapshot_root=asset / "snapshots",
        receipts_dir=run / "receipts",
        certificate_dir=run / "certificate",
        certificate_path=(
            run / "certificate" / f"pre_run_ready_certificate_{run_id}.json"
        ),
        quarantine_dir=run / "output" / f"phase4_quarantine_{run_id}",
        promote_to=run / "output" / f"phase4_run_output_{run_id}",
        exception_ledger_path=(
            run / "launch" / f"phase4_{run_id}_single_use_ledger.json"
        ),
    )


def _paths_overlap(left: Path, right: Path) -> bool:
    """Return whether two canonical paths are equal or nested."""

    left = Path(left).resolve()
    right = Path(right).resolve()
    return left == right or left in right.parents or right in left.parents


def _lexists(path: Path) -> bool:
    """Check path presence without following a dangling final link."""

    return os.path.lexists(path)


def _bounded_child_names(
    directory: Path,
    *,
    maximum: int,
    failure_message: str,
) -> list[str]:
    """Read at most ``maximum + 1`` directory entries, never an unbounded list."""

    if maximum < 0:
        raise ValueError("bounded directory maximum must be nonnegative")
    names: list[str] = []
    try:
        with os.scandir(directory) as entries:
            for entry in entries:
                if len(names) == maximum:
                    raise RuntimeError(
                        f"{failure_message}; more than {maximum} entries observed"
                    )
                names.append(entry.name)
    except RuntimeError:
        raise
    except OSError as exc:
        raise RuntimeError(
            f"{failure_message}; directory enumeration failed"
            f" ({exc.__class__.__name__})"
        ) from exc
    return names


def _identity(info: os.stat_result) -> tuple[int, int, int]:
    return int(info.st_dev), int(info.st_ino), int(info.st_mode)


def _is_link_info(info: os.stat_result) -> bool:
    return stat.S_ISLNK(info.st_mode) or bool(
        getattr(info, "st_file_attributes", 0)
        & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    )


def sha256_file(
    path: Path,
    *,
    tree_root: Path | None = None,
    max_bytes: int = MAX_STAGED_INPUT_BYTES,
) -> str:
    """Hash one bounded ordinary file through a stable lexical path."""

    candidate = Path(os.path.abspath(path))
    root = Path(os.path.abspath(tree_root)) if tree_root is not None else None
    before_chain = (
        schema.stable_directory_chain(candidate.parent, root)
        if root is not None
        else schema.stable_directory_chain(candidate.parent, candidate.parent)
    )
    try:
        before_info = os.stat(candidate, follow_symlinks=False)
    except OSError as exc:
        raise RuntimeError(
            f"bounded input is missing or unreadable: {candidate.name}"
        ) from exc
    if _is_link_info(before_info) or not stat.S_ISREG(before_info.st_mode):
        raise RuntimeError(
            f"bounded input is not an ordinary regular file: {candidate.name}"
        )
    if before_info.st_size > max_bytes:
        raise RuntimeError(
            f"bounded input {candidate.name!r} exceeds {max_bytes} bytes"
        )

    flags = (
        os.O_RDONLY
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        descriptor = os.open(candidate, flags)
    except OSError as exc:
        raise RuntimeError(
            f"bounded input cannot be opened without following links:"
            f" {candidate.name}"
        ) from exc
    digest = hashlib.sha256()
    observed_bytes = 0
    try:
        opened_info = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened_info.st_mode)
            or _identity(opened_info) != _identity(before_info)
        ):
            raise RuntimeError(
                f"bounded input identity changed before open: {candidate.name}"
            )
        if hasattr(os, "O_NONBLOCK") and hasattr(os, "get_blocking"):
            os.set_blocking(descriptor, True)
        while True:
            chunk = os.read(descriptor, 1 << 20)
            if not chunk:
                break
            observed_bytes += len(chunk)
            if observed_bytes > max_bytes:
                raise RuntimeError(
                    f"bounded input {candidate.name!r} exceeds {max_bytes} bytes"
                )
            digest.update(chunk)
    finally:
        os.close(descriptor)

    after_chain = (
        schema.stable_directory_chain(candidate.parent, root)
        if root is not None
        else schema.stable_directory_chain(candidate.parent, candidate.parent)
    )
    try:
        after_info = os.stat(candidate, follow_symlinks=False)
    except OSError as exc:
        raise RuntimeError(
            f"bounded input disappeared during hashing: {candidate.name}"
        ) from exc
    if (
        before_chain != after_chain
        or _identity(after_info) != _identity(opened_info)
        or int(after_info.st_size) != int(opened_info.st_size)
        or observed_bytes != int(opened_info.st_size)
    ):
        raise RuntimeError(
            f"bounded input path identity changed during hashing: {candidate.name}"
        )
    return digest.hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def publish_bytes_create_once(path: Path, data: bytes, *, label: str) -> None:
    """Publish one evidence file without any exists-check/write race."""

    fileio.create_once_bytes(path, data, exists_label=label)


def write_json_create_once(path: Path, value: Any, *, label: str) -> None:
    publish_bytes_create_once(
        path,
        _json_bytes(value),
        label=label,
    )


def _resolve_git_executable() -> Path:
    """Resolve Git from the same closed install locations as the launcher."""

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
    for candidate in candidates:
        if candidate.is_file() and not schema.is_filesystem_link(candidate):
            try:
                return candidate.resolve(strict=True)
            except OSError:
                continue
    raise RuntimeError("Git executable cannot be resolved from trusted locations")


def producer_environment() -> dict[str, str]:
    """Return the launcher's exact PATH-free producer environment."""

    environment = phase4_launcher._sanitized_runtime_environment()
    environment.pop("PATH", None)
    return environment


def suite_environment(git_executable: Path | None = None) -> dict[str, str]:
    """Add only the trusted Git directory to the suite child environment.

    Several integration tests invoke ``git`` by name.  The certified producer
    must remain PATH-free, but pytest needs this one executable lookup surface.
    No ambient PATH entry is inherited.
    """

    if git_executable is not None:
        supplied_git = Path(git_executable)
        if schema.is_filesystem_link(supplied_git):
            raise RuntimeError(
                f"suite Git executable must not be linked: {supplied_git}"
            )
        git_path = supplied_git.resolve(strict=True)
    else:
        git_path = _resolve_git_executable()
    if not git_path.is_file() or schema.is_filesystem_link(git_path):
        raise RuntimeError(f"suite Git executable is missing or linked: {git_path}")
    environment = producer_environment()
    environment["PATH"] = str(git_path.parent)
    return environment


def git(*args: str) -> str:
    return subprocess.run(
        [str(_resolve_git_executable()), *args],
        cwd=REPO,
        env=producer_environment(),
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def clean_head() -> dict[str, Any]:
    tracked = [
        line
        for line in git(
            "status", "--porcelain", "--untracked-files=no"
        ).splitlines()
        if line
    ]
    if tracked:
        raise RuntimeError(f"tracked tree is dirty: {tracked[:20]}")
    full_status = git(
        "status", "--porcelain=v1", "-z", "--untracked-files=all"
    )
    untracked = phase4.parse_untracked_porcelain_v1_z(full_status)
    unsafe_untracked = []
    for relative in untracked:
        portable = relative.replace("\\", "/")
        candidate = Path(portable)
        lexical = Path(os.path.abspath(REPO / candidate))
        try:
            lexical.relative_to(REPO)
        except ValueError as exc:
            raise RuntimeError(
                f"untracked Git entry escapes the repository: {portable!r}"
            ) from exc
        try:
            info = os.stat(lexical, follow_symlinks=False)
        except OSError as exc:
            raise RuntimeError(
                f"untracked Git entry changed or is unreadable: {portable!r}"
            ) from exc
        if _is_link_info(info):
            raise RuntimeError(
                "untracked symlink/reparse entries can shadow certified"
                f" imports or tests: {portable!r}"
            )
        try:
            schema.stable_directory_chain(lexical.parent, REPO)
        except schema.ColmAimsError as exc:
            raise RuntimeError(
                "untracked Git entry has an aliased parent chain:"
                f" {portable!r}"
            ) from exc
        if (
            portable == "tests"
            or portable.startswith("tests/")
            or candidate.suffix.lower() in _UNTRACKED_EXECUTABLE_SUFFIXES
            or candidate.name.lower() in _UNTRACKED_PYTEST_CONFIGS
            or candidate.name.lower()
            in {"conftest.py", "sitecustomize.py", "usercustomize.py"}
        ):
            unsafe_untracked.append(portable)
    if unsafe_untracked:
        raise RuntimeError(
            "untracked executable/import/test surfaces could influence the"
            f" certified suites: {unsafe_untracked[:20]}"
        )
    return {
        "commit": git("rev-parse", "HEAD").strip(),
        "tree": git("rev-parse", "HEAD^{tree}").strip(),
        "dirty": False,
        "untracked": untracked,
    }


def require_fresh_operational_paths(paths: OrchestrationPaths) -> None:
    """Fail closed unless roots and every derived target are safely fresh."""

    roots = {
        "repository root": REPO.resolve(),
        "asset root": paths.asset_root,
        "run root": paths.run_root,
    }
    for label, root in roots.items():
        try:
            schema.stable_directory_chain(root, root)
        except schema.ColmAimsError as exc:
            raise RuntimeError(f"{label} has an unstable directory chain") from exc
        if not root.is_dir() or schema.is_filesystem_link(root):
            raise RuntimeError(
                f"{label} is missing, not a directory, or linked: {root}"
            )
    root_items = list(roots.items())
    for index, (left_label, left) in enumerate(root_items):
        for right_label, right in root_items[index + 1 :]:
            if _paths_overlap(left, right):
                raise RuntimeError(
                    f"{left_label} and {right_label} must be disjoint:"
                    f" {left} vs {right}"
                )

    for directory in (
        paths.staged_data,
        paths.snapshot_root,
        paths.receipts_dir,
        paths.certificate_dir,
        paths.run_root / "output",
        paths.run_root / "launch",
    ):
        try:
            schema.stable_directory_chain(directory, directory)
        except schema.ColmAimsError as exc:
            raise RuntimeError(
                f"required prepared directory has an unstable chain: {directory}"
            ) from exc
        if not directory.is_dir() or schema.is_filesystem_link(directory):
            raise RuntimeError(
                f"required prepared directory is missing or linked: {directory}"
            )

    expected_run_children = {"certificate", "launch", "output", "receipts"}
    observed_run_children = set(
        _bounded_child_names(
            paths.run_root,
            maximum=len(expected_run_children),
            failure_message=(
                "fresh run root membership must be exactly"
                f" {sorted(expected_run_children)}"
            ),
        )
    )
    if observed_run_children != expected_run_children:
        raise RuntimeError(
            "fresh run root membership must be exactly"
            f" {sorted(expected_run_children)}; observed"
            f" {sorted(observed_run_children)}"
        )
    targets = {
        "certificate": paths.certificate_path,
        "quarantine": paths.quarantine_dir,
        "promotion": paths.promote_to,
        "exception ledger": paths.exception_ledger_path,
    }
    target_items = list(targets.items())
    for index, (left_label, left) in enumerate(target_items):
        for right_label, right in target_items[index + 1 :]:
            if _paths_overlap(left, right):
                raise RuntimeError(
                    f"derived {left_label} and {right_label} targets overlap:"
                    f" {left} vs {right}"
                )
    for label, path in targets.items():
        if _lexists(path):
            raise RuntimeError(f"derived {label} target already exists: {path}")
        if not path.parent.is_dir():
            raise RuntimeError(f"launch-path parent is missing: {path.parent}")
    for directory in (
        paths.receipts_dir,
        paths.certificate_dir,
        paths.run_root / "output",
        paths.run_root / "launch",
    ):
        _bounded_child_names(
            directory,
            maximum=0,
            failure_message=f"operational directory is not fresh: {directory}",
        )


def verify_prepared_inputs(paths: OrchestrationPaths) -> list[dict[str, str]]:
    plan: list[dict[str, str]] = []
    total_bytes = 0
    root = Path(os.path.abspath(paths.staged_data))
    root_chain = schema.stable_directory_chain(root, root)
    for label, (filename, expected) in EXPECTED_STAGED.items():
        path = root / filename
        if path.parent != root:
            raise RuntimeError(
                f"staged input path escapes its lexical root: {filename}"
            )
        try:
            info = os.stat(path, follow_symlinks=False)
        except OSError as exc:
            raise RuntimeError(f"staged input is unreadable: {filename}") from exc
        if _is_link_info(info) or not stat.S_ISREG(info.st_mode):
            raise RuntimeError(f"staged input is not a regular file: {filename}")
        if schema.resolves_inside(path, REPO):
            raise RuntimeError(f"staged input resolves inside the repo: {path}")
        total_bytes += int(info.st_size)
        if total_bytes > MAX_STAGED_TOTAL_BYTES:
            raise RuntimeError(
                "staged inputs exceed the aggregate byte limit"
                f" {MAX_STAGED_TOTAL_BYTES}"
            )
        observed = sha256_file(
            path,
            tree_root=root,
            max_bytes=MAX_STAGED_INPUT_BYTES,
        )
        if observed != expected:
            raise RuntimeError(
                f"{label} SHA-256 {observed} != expected {expected}"
            )
        plan.append(
            {
                "label": label,
                "path": str(path),
                "expected_sha256": expected,
            }
        )
    if schema.stable_directory_chain(root, root) != root_chain:
        raise RuntimeError("staged input root identity changed during verification")
    return plan


def verify_prepared_snapshots(paths: OrchestrationPaths) -> dict[str, Path]:
    manifest = phase4.load_model_snapshot_manifest(SNAPSHOT_MANIFEST_PATH)
    snapshot_root = Path(os.path.abspath(paths.snapshot_root))
    snapshot_root_chain = schema.stable_directory_chain(
        snapshot_root, snapshot_root
    )
    snapshot_dirs = {
        "primary_scorer": snapshot_root / "primary_scorer",
        "disjoint_selector": snapshot_root / "disjoint_selector",
    }
    declared_total = 0
    for role, directory in snapshot_dirs.items():
        before_chain = schema.stable_directory_chain(directory, directory)
        if schema.resolves_inside(directory, REPO):
            raise RuntimeError(f"snapshot resolves inside the repo: {directory}")
        declared_files = manifest["roles"][role]["files"]
        role_total = 0
        for relative, metadata in declared_files.items():
            size = metadata["size"]
            if size > MAX_SNAPSHOT_FILE_BYTES:
                raise RuntimeError(
                    f"snapshot member {relative!r} exceeds the per-file byte limit"
                )
            role_total += size
        declared_total += role_total
        if declared_total > MAX_SNAPSHOT_TOTAL_BYTES:
            raise RuntimeError(
                "snapshot roles exceed the aggregate byte limit"
                f" {MAX_SNAPSHOT_TOTAL_BYTES}"
            )
        phase4.verify_snapshot_dir(manifest["roles"][role], directory)
        if schema.stable_directory_chain(directory, directory) != before_chain:
            raise RuntimeError(
                f"snapshot role root changed during verification: {role}"
            )
    if (
        schema.stable_directory_chain(snapshot_root, snapshot_root)
        != snapshot_root_chain
    ):
        raise RuntimeError("snapshot root identity changed during verification")
    return snapshot_dirs


class _WindowsJobMemberListTooSmall(RuntimeError):
    """A bounded Job PID query needs a larger caller-owned buffer."""

    def __init__(self, assigned: int, listed: int) -> None:
        super().__init__("Windows Job Object member list buffer is too small")
        self.assigned = assigned
        self.listed = listed


class _WindowsJobObject:
    """Own a fail-closed Windows process job for one suite invocation."""

    _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
    _JOB_OBJECT_BASIC_ACCOUNTING_INFORMATION = 1
    _JOB_OBJECT_BASIC_PROCESS_ID_LIST = 3
    _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9

    def __init__(self) -> None:
        if os.name != "nt":  # pragma: no cover - construction is Windows-only
            raise RuntimeError("Windows Job Objects are unavailable on this host")

        import ctypes
        from ctypes import wintypes

        class IO_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("ReadOperationCount", ctypes.c_ulonglong),
                ("WriteOperationCount", ctypes.c_ulonglong),
                ("OtherOperationCount", ctypes.c_ulonglong),
                ("ReadTransferCount", ctypes.c_ulonglong),
                ("WriteTransferCount", ctypes.c_ulonglong),
                ("OtherTransferCount", ctypes.c_ulonglong),
            ]

        class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_longlong),
                ("PerJobUserTimeLimit", ctypes.c_longlong),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
                ("IoInfo", IO_COUNTERS),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        class JOBOBJECT_BASIC_ACCOUNTING_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("TotalUserTime", ctypes.c_longlong),
                ("TotalKernelTime", ctypes.c_longlong),
                ("ThisPeriodTotalUserTime", ctypes.c_longlong),
                ("ThisPeriodTotalKernelTime", ctypes.c_longlong),
                ("TotalPageFaultCount", wintypes.DWORD),
                ("TotalProcesses", wintypes.DWORD),
                ("ActiveProcesses", wintypes.DWORD),
                ("TotalTerminatedProcesses", wintypes.DWORD),
            ]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.SetInformationJobObject.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
        ]
        kernel32.SetInformationJobObject.restype = wintypes.BOOL
        kernel32.AssignProcessToJobObject.argtypes = [
            wintypes.HANDLE,
            wintypes.HANDLE,
        ]
        kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
        kernel32.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
        kernel32.TerminateJobObject.restype = wintypes.BOOL
        kernel32.QueryInformationJobObject.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
            ctypes.POINTER(wintypes.DWORD),
        ]
        kernel32.QueryInformationJobObject.restype = wintypes.BOOL
        kernel32.OpenProcess.argtypes = [
            wintypes.DWORD,
            wintypes.BOOL,
            wintypes.DWORD,
        ]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.IsProcessInJob.argtypes = [
            wintypes.HANDLE,
            wintypes.HANDLE,
            ctypes.POINTER(wintypes.BOOL),
        ]
        kernel32.IsProcessInJob.restype = wintypes.BOOL
        kernel32.WaitForSingleObject.argtypes = [
            wintypes.HANDLE,
            wintypes.DWORD,
        ]
        kernel32.WaitForSingleObject.restype = wintypes.DWORD
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL

        handle = kernel32.CreateJobObjectW(None, None)
        if not handle:
            error = ctypes.get_last_error()
            raise OSError(error, ctypes.FormatError(error))
        self._ctypes = ctypes
        self._kernel32 = kernel32
        self._wintypes = wintypes
        self._accounting_type = JOBOBJECT_BASIC_ACCOUNTING_INFORMATION
        self._handle: Any | None = handle

        limits = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        limits.BasicLimitInformation.LimitFlags = (
            self._JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        )
        if not kernel32.SetInformationJobObject(
            handle,
            self._JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
            ctypes.byref(limits),
            ctypes.sizeof(limits),
        ):
            error = ctypes.get_last_error()
            kernel32.CloseHandle(handle)
            self._handle = None
            raise OSError(error, ctypes.FormatError(error))

    def assign(self, process: subprocess.Popen[bytes]) -> None:
        """Assign the suite root before it can intentionally spawn work."""

        if self._handle is None:
            raise RuntimeError("Windows Job Object is already closed")
        process_handle = getattr(process, "_handle", None)
        if process_handle is None:  # pragma: no cover - CPython contract guard
            raise RuntimeError("Windows suite process handle is unavailable")
        if not self._kernel32.AssignProcessToJobObject(
            self._handle, self._ctypes.c_void_p(int(process_handle))
        ):
            error = self._ctypes.get_last_error()
            raise OSError(error, self._ctypes.FormatError(error))

    def terminate(self) -> None:
        """Terminate every process still associated with this job."""

        if self._handle is None:
            return
        if not self._kernel32.TerminateJobObject(self._handle, 1):
            error = self._ctypes.get_last_error()
            raise OSError(error, self._ctypes.FormatError(error))
        deadline = time.monotonic() + 30
        member_handles = self._open_member_process_handles(deadline=deadline)
        try:
            for member_handle in member_handles:
                remaining_ms = max(0, int((deadline - time.monotonic()) * 1000))
                wait_result = self._kernel32.WaitForSingleObject(
                    member_handle, remaining_ms
                )
                if wait_result != 0:
                    raise RuntimeError(
                        "Windows Job Object member did not terminate within 30 seconds"
                    )
            while self.active_processes() != 0:
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        "Windows Job Object members did not terminate within 30 seconds"
                    )
                time.sleep(0.01)
        finally:
            close_error: OSError | None = None
            for member_handle in member_handles:
                if not self._kernel32.CloseHandle(member_handle):
                    error = self._ctypes.get_last_error()
                    close_error = close_error or OSError(
                        error, self._ctypes.FormatError(error)
                    )
            if close_error is not None:
                raise close_error

    def _open_member_process_handles(self, *, deadline: float) -> list[Any]:
        """Hold waitable handles for Job members visible after termination begins."""

        if self._handle is None:
            raise RuntimeError("Windows Job Object is already closed")
        maximum_capacity = 4096
        capacity = max(1, min(self.active_processes(), maximum_capacity))
        for _attempt in range(8):
            try:
                process_ids = self._query_member_process_ids(capacity)
                break
            except _WindowsJobMemberListTooSmall as exc:
                assigned = exc.assigned
                listed = exc.listed
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    "Windows Job Object member enumeration exceeded its deadline"
                )
            requested_capacity = max(
                capacity * 2,
                assigned,
                listed,
            )
            if requested_capacity > maximum_capacity:
                raise RuntimeError(
                    "Windows Job Object exceeds the 4096-member cleanup bound"
                )
            capacity = requested_capacity
        else:
            raise RuntimeError(
                "Windows Job Object member enumeration did not stabilize"
            )

        handles: list[Any] = []
        try:
            for process_id in process_ids:
                handle = self._kernel32.OpenProcess(0x00101000, False, process_id)
                if handle:
                    is_member = self._wintypes.BOOL()
                    if not self._kernel32.IsProcessInJob(
                        handle, self._handle, self._ctypes.byref(is_member)
                    ):
                        error = self._ctypes.get_last_error()
                        self._kernel32.CloseHandle(handle)
                        raise OSError(error, self._ctypes.FormatError(error))
                    if is_member.value:
                        handles.append(handle)
                    elif not self._kernel32.CloseHandle(handle):
                        error = self._ctypes.get_last_error()
                        raise OSError(error, self._ctypes.FormatError(error))
                    continue
                error = self._ctypes.get_last_error()
                if error != 87:
                    raise OSError(error, self._ctypes.FormatError(error))
        except BaseException:
            for handle in handles:
                self._kernel32.CloseHandle(handle)
            raise
        return handles

    def _query_member_process_ids(self, capacity: int) -> tuple[int, ...]:
        """Perform one fixed-capacity Job member query."""

        class JOBOBJECT_BASIC_PROCESS_ID_LIST(self._ctypes.Structure):
            _fields_ = [
                ("NumberOfAssignedProcesses", self._wintypes.DWORD),
                ("NumberOfProcessIdsInList", self._wintypes.DWORD),
                ("ProcessIdList", self._ctypes.c_size_t * capacity),
            ]

        process_ids = JOBOBJECT_BASIC_PROCESS_ID_LIST()
        if not self._kernel32.QueryInformationJobObject(
            self._handle,
            self._JOB_OBJECT_BASIC_PROCESS_ID_LIST,
            self._ctypes.byref(process_ids),
            self._ctypes.sizeof(process_ids),
            None,
        ):
            error = self._ctypes.get_last_error()
            if error == 234:
                raise _WindowsJobMemberListTooSmall(
                    int(process_ids.NumberOfAssignedProcesses),
                    int(process_ids.NumberOfProcessIdsInList),
                )
            raise OSError(error, self._ctypes.FormatError(error))
        return tuple(
            int(process_ids.ProcessIdList[index])
            for index in range(int(process_ids.NumberOfProcessIdsInList))
        )

    def active_processes(self) -> int:
        """Return the live member count while the job handle remains owned."""

        if self._handle is None:
            raise RuntimeError("Windows Job Object is already closed")
        accounting = self._accounting_type()
        if not self._kernel32.QueryInformationJobObject(
            self._handle,
            self._JOB_OBJECT_BASIC_ACCOUNTING_INFORMATION,
            self._ctypes.byref(accounting),
            self._ctypes.sizeof(accounting),
            None,
        ):
            error = self._ctypes.get_last_error()
            raise OSError(error, self._ctypes.FormatError(error))
        return int(accounting.ActiveProcesses)

    def close(self) -> None:
        """Close the job; the configured limit kills any remaining members."""

        handle = self._handle
        if handle is None:
            return
        self._handle = None
        if not self._kernel32.CloseHandle(handle):
            error = self._ctypes.get_last_error()
            raise OSError(error, self._ctypes.FormatError(error))


def _terminate_process_tree(
    process: subprocess.Popen[bytes],
    windows_job: _WindowsJobObject | None = None,
) -> None:
    """Terminate the suite process and its descendants, then reap it."""

    if os.name == "nt":
        if windows_job is None:
            raise RuntimeError("Windows suite process has no owning Job Object")
        windows_job.terminate()
    else:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        grace_deadline = time.monotonic() + 5
        while time.monotonic() < grace_deadline:
            process.poll()
            try:
                os.killpg(process.pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.05)
        else:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=30)


def _run_bounded_suite_process(
    command: list[str],
    *,
    environment: dict[str, str],
    transcript_stage: Path,
    timeout_seconds: float,
    max_transcript_bytes: int,
) -> int:
    """Run pytest with bounded streaming capture and process-tree cleanup."""

    windows_job: _WindowsJobObject | None = None
    if os.name == "nt":
        windows_job = _WindowsJobObject()
        launch_command = [
            sys.executable,
            "-I",
            "-S",
            "-c",
            _WINDOWS_SUITE_SUPERVISOR,
            json.dumps(command, ensure_ascii=True, separators=(",", ":")),
        ]
    else:
        launch_command = command
    popen_kwargs: dict[str, Any] = {
        "cwd": REPO,
        "env": environment,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.STDOUT,
        "stdin": subprocess.PIPE if os.name == "nt" else subprocess.DEVNULL,
    }
    if os.name == "nt":
        popen_kwargs["creationflags"] = getattr(
            subprocess, "CREATE_NEW_PROCESS_GROUP", 0
        )
    else:
        popen_kwargs["start_new_session"] = True
    process: subprocess.Popen[bytes] | None = None
    reader: threading.Thread | None = None
    try:
        process = subprocess.Popen(launch_command, **popen_kwargs)
        if windows_job is not None:
            try:
                windows_job.assign(process)
            except BaseException:
                process.kill()
                process.wait(timeout=30)
                raise
        if process.stdout is None:  # pragma: no cover - Popen contract guard
            _terminate_process_tree(process, windows_job)
            raise RuntimeError("suite stdout pipe was not created")

        overflow = threading.Event()
        reader_error: list[BaseException] = []

        def copy_output() -> None:
            written = 0
            try:
                with transcript_stage.open("xb") as transcript:
                    while True:
                        chunk = process.stdout.read(1 << 16)
                        if not chunk:
                            break
                        if written + len(chunk) > max_transcript_bytes:
                            allowed = max(0, max_transcript_bytes - written)
                            if allowed:
                                transcript.write(chunk[:allowed])
                            transcript.flush()
                            overflow.set()
                            break
                        transcript.write(chunk)
                        written += len(chunk)
            except BaseException as exc:  # noqa: BLE001 - relayed to owner thread
                reader_error.append(exc)
                overflow.set()

        reader = threading.Thread(target=copy_output, daemon=True)
        reader.start()
        if windows_job is not None:
            if process.stdin is None:  # pragma: no cover - Popen contract guard
                raise RuntimeError("Windows suite supervisor gate was not created")
            process.stdin.write(b"R")
            process.stdin.flush()
            process.stdin.close()
        deadline = time.monotonic() + timeout_seconds
        timed_out = False
        while process.poll() is None:
            if overflow.is_set():
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
                break
            try:
                process.wait(timeout=min(0.25, remaining))
            except subprocess.TimeoutExpired:
                continue
        lingering_windows_descendants = False
        if windows_job is not None:
            if not timed_out and not overflow.is_set():
                process.wait(timeout=30)
                lingering_windows_descendants = windows_job.active_processes() > 0
            _terminate_process_tree(process, windows_job)
        elif timed_out or overflow.is_set():
            _terminate_process_tree(process, windows_job)
        reader.join(timeout=SUITE_READER_JOIN_TIMEOUT_SECONDS)
        if reader.is_alive():
            _terminate_process_tree(process, windows_job)
            reader.join(timeout=SUITE_READER_JOIN_TIMEOUT_SECONDS)
            raise RuntimeError("suite transcript reader did not terminate")
        if reader_error:
            raise RuntimeError("suite transcript streaming failed") from reader_error[0]
        if lingering_windows_descendants:
            raise RuntimeError("suite left running descendants after its root exited")
        if timed_out:
            raise RuntimeError(
                f"suite exceeded its timeout of {timeout_seconds:g} seconds"
            )
        if overflow.is_set():
            raise RuntimeError(
                "suite transcript exceeded its byte limit"
                f" {max_transcript_bytes}"
            )
        return int(process.returncode)
    finally:
        active_error = sys.exc_info()[1]
        cleanup_error: BaseException | None = None
        if process is not None and (
            process.poll() is None or (reader is not None and reader.is_alive())
        ):
            try:
                _terminate_process_tree(process, windows_job)
            except BaseException as exc:  # noqa: BLE001 - preserve cleanup failure
                cleanup_error = exc
        if windows_job is not None:
            try:
                windows_job.close()
            except BaseException as exc:  # noqa: BLE001 - preserve cleanup failure
                cleanup_error = cleanup_error or exc
        if process is not None and process.stdin is not None:
            try:
                process.stdin.close()
            except BaseException as exc:  # noqa: BLE001 - preserve cleanup failure
                cleanup_error = cleanup_error or exc
        if process is not None and process.stdout is not None:
            try:
                process.stdout.close()
            except BaseException as exc:  # noqa: BLE001 - preserve cleanup failure
                cleanup_error = cleanup_error or exc
        if reader is not None and reader.is_alive():
            reader.join(timeout=SUITE_READER_JOIN_TIMEOUT_SECONDS)
            if reader.is_alive() and cleanup_error is None:
                cleanup_error = RuntimeError(
                    "suite transcript reader survived final cleanup"
                )
        if cleanup_error is not None:
            if active_error is None:
                raise cleanup_error
            active_error.add_note(f"additional cleanup failure: {cleanup_error!r}")


def run_suite(
    paths: OrchestrationPaths,
    name: str,
    pytest_args: list[str],
    *,
    timeout_seconds: float | None = None,
    max_transcript_bytes: int = MAX_SUITE_TRANSCRIPT_BYTES,
) -> dict[str, Any]:
    if name not in SUITE_TIMEOUT_SECONDS:
        raise RuntimeError(f"unknown suite identity: {name!r}")
    if timeout_seconds is None:
        timeout_seconds = SUITE_TIMEOUT_SECONDS[name]
    if timeout_seconds <= 0 or max_transcript_bytes <= 0:
        raise RuntimeError("suite timeout and transcript byte limit must be positive")
    junit_path = paths.receipts_dir / f"{name}_suite.xml"
    transcript_path = paths.receipts_dir / f"{name}_suite_transcript.txt"
    if _lexists(junit_path) or _lexists(transcript_path):
        raise RuntimeError(f"suite {name!r} output slot is already claimed")
    with tempfile.TemporaryDirectory(
        prefix=f".{name}-suite-staged-", dir=paths.run_root
    ) as temporary:
        stage_root = Path(temporary)
        junit_stage = stage_root / "suite.xml"
        transcript_stage = stage_root / "transcript.txt"
        command = [
            str(Path(sys.executable).resolve()),
            "-m",
            "pytest",
            *pytest_args,
            "-q",
            "-p",
            "no:cacheprovider",
            f"--junitxml={junit_stage}",
        ]
        print(f"[suite:{name}] starting", flush=True)
        return_code = _run_bounded_suite_process(
            command,
            environment=suite_environment(),
            transcript_stage=transcript_stage,
            timeout_seconds=timeout_seconds,
            max_transcript_bytes=max_transcript_bytes,
        )
        if not junit_stage.is_file() or schema.is_filesystem_link(junit_stage):
            raise RuntimeError(f"suite {name!r} did not produce ordinary JUnit XML")
        junit_bytes = schema.read_regular_file_bytes(
            junit_stage,
            tree_root=stage_root,
            max_bytes=MAX_JUNIT_BYTES,
        )
        transcript_bytes = schema.read_regular_file_bytes(
            transcript_stage,
            tree_root=stage_root,
            max_bytes=max_transcript_bytes,
        )
        publish_bytes_create_once(
            transcript_path,
            transcript_bytes,
            label=f"{name} suite transcript",
        )
        publish_bytes_create_once(
            junit_path,
            junit_bytes,
            label=f"{name} suite JUnit XML",
        )
    tail = transcript_bytes.decode("utf-8", errors="replace").splitlines()[-3:]
    print(
        f"[suite:{name}] exit={return_code} tail={tail}",
        flush=True,
    )
    return {
        "command": command,
        "exit_code": return_code,
        "junit_path": junit_path,
        "transcript_path": transcript_path,
    }


def junit_counts(junit_path: Path) -> tuple[dict[str, int], list[str]]:
    data = schema.read_regular_file_bytes(
        junit_path,
        tree_root=Path(junit_path).parent,
        max_bytes=MAX_JUNIT_BYTES,
    )
    root = ET.fromstring(data)
    suites = [root] if root.tag == "testsuite" else root.findall("testsuite")
    if not suites:
        raise RuntimeError(f"no testsuite element in {junit_path}")
    counts = {
        key: sum(int(suite.get(key, "0")) for suite in suites)
        for key in ("tests", "failures", "errors", "skipped")
    }
    skips = [
        f"{case.get('classname')}::{case.get('name')}"
        for case in root.iter("testcase")
        if case.find("skipped") is not None
    ]
    return counts, skips


def build_receipt(
    run_info: dict[str, Any],
    environment_lock_sha256: str,
    workflow_sha256: str,
    interpreter_realpath: str,
    head: dict[str, Any],
) -> dict[str, Any]:
    counts, skips = junit_counts(run_info["junit_path"])
    return {
        "schema_version": schema.SCHEMA_VERSION,
        "exit_code": run_info["exit_code"],
        "command": run_info["command"],
        "environment_lock_sha256": environment_lock_sha256,
        "workflow_sha256": workflow_sha256,
        "interpreter_realpath": interpreter_realpath,
        "counts": counts,
        "skip_identities": skips,
        "junit_sha256": sha256_file(run_info["junit_path"]),
        "transcript_sha256": sha256_file(run_info["transcript_path"]),
        "commit": head["commit"],
        "tree_sha256": head["tree"],
        "dirty": head["dirty"],
    }


def producer_command(
    paths: OrchestrationPaths,
    interpreter: Path,
    snapshot_dirs: dict[str, Path],
    staged_plan: list[dict[str, str]],
) -> list[str]:
    staged = {entry["label"]: entry for entry in staged_plan}
    return [
        str(interpreter),
        "scripts/stopdff_fair_qa_retest.py",
        "--data-dir",
        str(paths.staged_data.resolve()),
        "--calibration",
        staged["calibration_train"]["path"],
        "--fit-split",
        "val",
        "--eval-split",
        "test",
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
        "--eligibility",
        "reproducibility/colm_aims_2026/frozen/pairing_eligibility_v2.json",
        "--snapshot-manifest",
        "reproducibility/colm_aims_2026/frozen/model_snapshot_manifests.json",
        "--primary-model-path",
        str(snapshot_dirs["primary_scorer"]),
        "--disjoint-model-path",
        str(snapshot_dirs["disjoint_selector"]),
        "--records-out",
        "phase4_run_output",
        "--staged-input",
        (
            f"fit_split={staged['fit_split']['path']}:"
            f"{staged['fit_split']['expected_sha256']}"
        ),
        "--staged-input",
        (
            f"mc_dataset={staged['mc_dataset']['path']}:"
            f"{staged['mc_dataset']['expected_sha256']}"
        ),
        "--staged-input",
        (
            f"answer_profiles={staged['answer_profiles']['path']}:"
            f"{staged['answer_profiles']['expected_sha256']}"
        ),
        "--staged-input",
        (
            f"build_metadata={staged['build_metadata']['path']}:"
            f"{staged['build_metadata']['expected_sha256']}"
        ),
        "--staged-input",
        (
            f"split_metadata={staged['split_metadata']['path']}:"
            f"{staged['split_metadata']['expected_sha256']}"
        ),
        "--out",
        "phase4_run_output/stopdff_fair_qa_regenerated.json",
    ]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python phase4_pre_run_ready_orchestration.py",
        description=(
            "Generate a Phase-4 PRE_RUN_READY certificate in a fresh,"
            " external, operator-prepared run root."
        ),
        allow_abbrev=False,
    )
    parser.add_argument(
        "--run-root",
        required=True,
        type=Path,
        help=(
            "absolute fresh external root containing four empty run"
            " directories"
        ),
    )
    parser.add_argument(
        "--run-id",
        required=True,
        help="portable identifier used in create-once certificate/launch paths",
    )
    parser.add_argument(
        "--asset-root",
        type=Path,
        default=DEFAULT_ASSET_ROOT,
        help=(
            "absolute external root containing staging/processed and snapshots"
            f" (default: {DEFAULT_ASSET_ROOT})"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    paths = derive_paths(
        asset_root=args.asset_root,
        run_root=args.run_root,
        run_id=args.run_id,
    )
    phase4_launcher._validate_ambient_environment()
    require_fresh_operational_paths(paths)
    staged_plan = verify_prepared_inputs(paths)
    snapshot_dirs = verify_prepared_snapshots(paths)

    head = clean_head()
    interpreter = Path(sys.executable).resolve()
    lock_bytes = phase4_launcher._default_probe_environment_lock(interpreter)
    publish_bytes_create_once(
        paths.receipts_dir / "environment_lock_pip_freeze.txt",
        lock_bytes,
        label="certified environment lock",
    )
    environment_lock_sha256 = hashlib.sha256(lock_bytes).hexdigest()
    workflow_sha256 = sha256_file(WORKFLOW_PATH, tree_root=REPO)

    suite_runs = {
        "focused": run_suite(paths, "focused", FOCUSED_TESTS),
        "full": run_suite(paths, "full", ["tests/"]),
    }
    post_suite_head = clean_head()
    if post_suite_head != head:
        raise RuntimeError(
            f"repository identity changed during suites: {head} -> {post_suite_head}"
        )
    post_suite_lock = phase4_launcher._default_probe_environment_lock(
        interpreter
    )
    if post_suite_lock != lock_bytes:
        raise RuntimeError(
            "certified interpreter dependency lock changed during suites"
        )

    receipt_paths: dict[str, Path] = {}
    receipts: dict[str, dict[str, Any]] = {}
    for name, run_info in suite_runs.items():
        receipt = build_receipt(
            run_info,
            environment_lock_sha256,
            workflow_sha256,
            str(interpreter),
            head,
        )
        receipt_path = paths.receipts_dir / f"suite_receipt_{name}.json"
        if receipt["exit_code"] == 0:
            receipt_module.validate_suite_receipt(receipt)
        write_json_create_once(
            receipt_path,
            receipt,
            label=f"{name} suite receipt",
        )
        receipt_paths[name] = receipt_path
        receipts[name] = receipt

    failed = [
        name
        for name, receipt in receipts.items()
        if receipt["exit_code"] != 0
        or receipt["counts"]["failures"] != 0
        or receipt["counts"]["errors"] != 0
    ]
    if failed:
        print(f"CERTIFICATE_NOT_GENERATED: suite failure(s) {failed}")
        return 1

    host = phase4_launcher._default_host_identity()
    try:
        import numpy as np

        blas = json.dumps(
            np.show_config(mode="dicts"), sort_keys=True, default=str
        )
    except Exception as exc:  # pragma: no cover - fail closed in operation
        raise RuntimeError(f"NumPy BLAS capture failed: {exc}") from exc
    cpu = (
        platform.processor()
        or os.environ.get("PROCESSOR_IDENTIFIER")
        or platform.uname().processor
        or "unknown"
    )
    environment = {
        "interpreter_realpath": str(interpreter),
        "os": host["os"],
        "arch": host["arch"],
        "cpu": cpu,
        "blas": blas,
        "thread_settings": dict(phase4.PHASE4_THREAD_SETTINGS),
        "environment_lock_sha256": environment_lock_sha256,
        "command": producer_command(
            paths, interpreter, snapshot_dirs, staged_plan
        ),
        "seeds": [1],
        "pythonhashseed": "0",
        "archived_rng_pinned": False,
        "fresh_rng_pinned": True,
        "quarantine_dir": str(paths.quarantine_dir.resolve()),
        "promote_to": str(paths.promote_to.resolve()),
        "exception_ledger_path": str(paths.exception_ledger_path.resolve()),
    }
    config = {
        "repo_root": str(REPO),
        "eligibility_path": str(ELIGIBILITY_PATH),
        "snapshot_manifest_path": str(SNAPSHOT_MANIFEST_PATH),
        "snapshot_dirs": {
            role: str(path) for role, path in snapshot_dirs.items()
        },
        "parity_anchor_path": str(PARITY_ANCHOR_PATH),
        "qa012_manifest_path": str(QA012_REV3_PATH),
        "staged_plan": staged_plan,
        "suite_receipt_paths": {
            name: str(path) for name, path in receipt_paths.items()
        },
        "content_hash_paths": {
            "spec_sha256": str(
                REPO / ".correctless/specs/camera-ready-aims-evidence-2.md"
            ),
            "verifier_sha256": str(
                REPO / "reproducibility/colm_aims_2026/verifier.py"
            ),
            "producer_sha256": str(
                REPO / "scripts/stopdff_fair_qa_retest.py"
            ),
            "schema_py_sha256": str(
                REPO / "reproducibility/colm_aims_2026/schema.py"
            ),
            "pairing_py_sha256": str(
                REPO / "reproducibility/colm_aims_2026/pairing.py"
            ),
            "phase4_py_sha256": str(
                REPO / "reproducibility/colm_aims_2026/phase4.py"
            ),
            "phase4_records_py_sha256": str(
                REPO / "reproducibility/colm_aims_2026/phase4_records.py"
            ),
            "phase4_launcher_py_sha256": str(
                REPO / "reproducibility/colm_aims_2026/phase4_launcher.py"
            ),
            "fileio_py_sha256": str(
                REPO / "scripts/stopdff_v5/fileio.py"
            ),
            "locking_py_sha256": str(
                REPO / "scripts/stopdff_v5/locking.py"
            ),
            "orchestration_sha256": str(Path(__file__).resolve()),
        },
        "environment": environment,
        "offline_flags": [
            "HF_HUB_OFFLINE=1",
            "TRANSFORMERS_OFFLINE=1",
        ],
    }
    components = phase4.gather_certificate_components(config)
    certificate = phase4.assemble_certificate(components)
    certificate_bytes = schema.encode_json(certificate)
    publish_bytes_create_once(
        paths.certificate_path,
        certificate_bytes,
        label="PRE_RUN_READY certificate",
    )
    result = {
        "certificate": certificate,
        "ready": certificate["ready"],
        "path": str(paths.certificate_path),
        "sha256": hashlib.sha256(certificate_bytes).hexdigest(),
    }
    summary = {
        "ready": result["ready"],
        "path": result["path"],
        "sha256": result["sha256"],
        "commit": head["commit"],
        "tree": head["tree"],
        "focused_counts": receipts["focused"]["counts"],
        "full_counts": receipts["full"]["counts"],
        "failing_checks": result["certificate"].get("failing_checks", []),
    }
    write_json_create_once(
        paths.certificate_dir / "certificate_generation_summary.json",
        summary,
        label="certificate generation summary",
    )
    print("CERTIFICATE_RESULT=" + json.dumps(summary, sort_keys=True))
    return 0 if result["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
