"""Publish externally supplied, pre-authored Phase-4 release sidecars.

This module deliberately does not construct a claim ledger, rights inventory,
or expectations anchor.  Those three authority documents are pre-authored
external inputs; this command makes no claim about their signer or authorship.
It validates and copies their exact bytes into one create-once sidecar bundle,
then exercises the existing canonical runs-root release verifier against the
closed staged bytes before the terminal atomic publication step.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import os
import re
import secrets
import stat
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scripts.stopdff_v5 import fileio

from . import ledger as ledger_module
from . import schema, verifier


EXIT_PASS = 0
EXIT_VERIFY_FAIL = 1
EXIT_USAGE_ERROR = 2
EXIT_INGRESS_ERROR = 3
EXIT_INTERNAL_ERROR = 4

_LEDGER_NAME = "ledger.json"
_RIGHTS_NAME = "rights.json"
_EXPECTATIONS_NAME = "expectations.json"
_SIDECAR_NAMES = (_LEDGER_NAME, _RIGHTS_NAME, _EXPECTATIONS_NAME)
_PENDING_GUARD_SUFFIX = ".pending"
_ACCEPTED_MARKER_SUFFIX = ".accepted"
_PORTABLE_ID_MAX_BYTES = 64
_PORTABLE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")
_WINDOWS_VOLUME_GUID_PATH_RE = re.compile(
    r"^\\\\\?\\Volume\{[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-"
    r"[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}\}"
    r"\\.*\Z"
)
_WINDOWS_RESERVED_BASENAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{number}" for number in range(1, 10)),
    *(f"LPT{number}" for number in range(1, 10)),
}
_POSIX_DIR_FD_OPERATIONS_SUPPORTED = os.name != "posix" or all(
    function in os.supports_dir_fd
    for function in (os.open, os.stat, os.mkdir, os.rename, os.unlink)
)


class ReleaseVerificationFailed(schema.ColmAimsError):
    """A staged release candidate did not reach ``PASS_RELEASE``."""


class _GuardRetirementCommittedError(OSError):
    """Guard unlink committed after acceptance, but its sync retries failed."""


_LexicalChain = tuple[tuple[str, tuple[int, int, int]], ...]


@dataclass(frozen=True)
class _WindowsComponentIdentity:
    """Version-independent native identity captured for one chain component."""

    component: str
    volume_serial: int
    file_id: bytes
    final_volume_path: str


@dataclass(frozen=True)
class _CapturedDirectoryChain:
    """Lexical chain plus the original full native Windows identity snapshot."""

    lexical: _LexicalChain
    windows: tuple[_WindowsComponentIdentity, ...] | None = None


def _windows_stat_pair_matches(
    lexical_identity: tuple[int, int, int],
    full_native_pair: tuple[int, int],
    legacy_pair: tuple[int, int],
) -> bool:
    """Accept one complete supported Windows stat identity profile."""
    return lexical_identity[:2] in (full_native_pair, legacy_pair)


class _DirectoryAnchor:
    """Hold one validated publication parent generation for a transaction.

    POSIX child operations are descriptor-relative.  Windows holds a native
    directory handle without ``FILE_SHARE_DELETE`` across every lexical child
    operation, which prevents the anchored parent from being renamed or
    replaced while the transaction is live.  Other hosts fail closed.
    """

    def __init__(
        self,
        path: Path,
        captured: _CapturedDirectoryChain | _LexicalChain,
        label: str,
        *,
        delete_access: bool = False,
        share_delete: bool = False,
    ) -> None:
        self.path = Path(os.path.abspath(path))
        if isinstance(captured, _CapturedDirectoryChain):
            self.snapshot = captured
        else:
            if os.name == "nt":
                raise schema.TypedIngressError(
                    "Windows directory anchors require an original full-native"
                    " chain snapshot"
                )
            self.snapshot = _CapturedDirectoryChain(lexical=captured)
        self.captured = self.snapshot.lexical
        self.label = label
        self._delete_access = delete_access
        self._share_delete = share_delete
        self._fd: int | None = None
        self._win_handles: list[int] = []
        self._operation_path: Path | None = None

    def __enter__(self) -> "_DirectoryAnchor":
        _require_unchanged_directory(self.path, self.captured, self.label)
        try:
            if os.name == "posix":
                if not _POSIX_DIR_FD_OPERATIONS_SUPPORTED:
                    raise schema.TypedIngressError(
                        "host does not support the required descriptor-relative"
                        " publication operations"
                    )
                flags = (
                    os.O_RDONLY
                    | getattr(os, "O_DIRECTORY", 0)
                    | getattr(os, "O_NOFOLLOW", 0)
                    | getattr(os, "O_CLOEXEC", 0)
                )
                if not getattr(os, "O_DIRECTORY", 0) or not getattr(
                    os, "O_NOFOLLOW", 0
                ):
                    raise schema.TypedIngressError(
                        "host cannot open a no-follow directory anchor"
                    )
                self._fd = os.open(self.path, flags)
                self._verify_posix_handle()
            elif os.name == "nt":
                self._open_windows_handle()
            else:  # pragma: no cover - closed unsupported-host branch
                raise schema.TypedIngressError(
                    "host has no supported directory-anchor implementation"
                )
            self.revalidate("anchor acquisition")
        except BaseException:
            self.__exit__()
            raise
        return self

    def __exit__(self, *_exc: object) -> None:
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
        if self._win_handles:
            import ctypes
            from ctypes import wintypes

            close_handle = ctypes.WinDLL(
                "kernel32", use_last_error=True
            ).CloseHandle
            close_handle.argtypes = [wintypes.HANDLE]
            close_handle.restype = wintypes.BOOL
            while self._win_handles:
                close_handle(wintypes.HANDLE(self._win_handles.pop()))
        self._operation_path = None

    def _verify_posix_handle(self) -> None:
        if self._fd is None:  # pragma: no cover - internal invariant
            raise RuntimeError("directory anchor is not open")
        info = os.fstat(self._fd)
        if not stat.S_ISDIR(info.st_mode) or schema._identity_tuple(info) != (
            self.captured[-1][1]
        ):
            raise schema.TypedIngressError(
                f"{self.label} handle does not identify the captured directory"
            )

    @staticmethod
    def _windows_native_file_id(file_id_information: Any) -> tuple[int, bytes]:
        """Return the complete ReFS-safe native volume and 128-bit file ID."""
        return (
            int(file_id_information.volume_serial_number),
            bytes(file_id_information.file_id.identifier),
        )

    @staticmethod
    def _windows_final_volume_path(handle: int) -> str:
        """Resolve one handle to a normalized local Volume-GUID path."""
        import ctypes
        from ctypes import wintypes

        get_final_path = ctypes.WinDLL(
            "kernel32", use_last_error=True
        ).GetFinalPathNameByHandleW
        get_final_path.argtypes = [
            wintypes.HANDLE,
            wintypes.LPWSTR,
            wintypes.DWORD,
            wintypes.DWORD,
        ]
        get_final_path.restype = wintypes.DWORD
        capacity = 32768
        buffer = ctypes.create_unicode_buffer(capacity)
        length = get_final_path(
            wintypes.HANDLE(handle),
            buffer,
            capacity,
            0x00000001,  # FILE_NAME_NORMALIZED | VOLUME_NAME_GUID
        )
        if length == 0:
            error = ctypes.get_last_error()
            raise schema.TypedIngressError(
                "cannot resolve a local Volume-GUID directory anchor"
                f" (WindowsError {error})"
            )
        if length >= capacity:
            raise schema.TypedIngressError(
                "local Volume-GUID directory anchor path is truncated"
            )
        return buffer.value

    @staticmethod
    def _require_local_volume_guid_path(path: str) -> str:
        """Reject UNC, DOS, device, malformed, or non-normalized final names."""
        if not isinstance(path, str) or _WINDOWS_VOLUME_GUID_PATH_RE.fullmatch(
            path
        ) is None or "/" in path or "\x00" in path:
            raise schema.TypedIngressError(
                "Windows publication requires a verified local"
                " \\\\?\\Volume{GUID}\\... operation path; UNC, mapped-network,"
                " DOS-device, and unverifiable paths are refused"
            )
        suffix = path[path.index("}") + 1:]
        if any(part in {".", "..", ""} for part in suffix.split("\\")[1:]):
            # One terminal empty part is the canonical volume-root slash.
            if suffix != "\\":
                raise schema.TypedIngressError(
                    "Windows Volume-GUID operation path is not normalized"
                )
        return path

    def _open_windows_handle(self) -> None:
        import ctypes
        from ctypes import wintypes

        class _ByHandleFileInformation(ctypes.Structure):
            _fields_ = [
                ("dwFileAttributes", wintypes.DWORD),
                ("ftCreationTime", wintypes.FILETIME),
                ("ftLastAccessTime", wintypes.FILETIME),
                ("ftLastWriteTime", wintypes.FILETIME),
                ("dwVolumeSerialNumber", wintypes.DWORD),
                ("nFileSizeHigh", wintypes.DWORD),
                ("nFileSizeLow", wintypes.DWORD),
                ("nNumberOfLinks", wintypes.DWORD),
                ("nFileIndexHigh", wintypes.DWORD),
                ("nFileIndexLow", wintypes.DWORD),
            ]

        class _FileId128(ctypes.Structure):
            _fields_ = [("identifier", ctypes.c_ubyte * 16)]

        class _FileIdInformation(ctypes.Structure):
            _fields_ = [
                ("volume_serial_number", ctypes.c_ulonglong),
                ("file_id", _FileId128),
            ]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        close_handle = kernel32.CloseHandle
        close_handle.argtypes = [wintypes.HANDLE]
        close_handle.restype = wintypes.BOOL
        get_information = kernel32.GetFileInformationByHandle
        get_information.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(_ByHandleFileInformation),
        ]
        get_information.restype = wintypes.BOOL
        get_information_ex = kernel32.GetFileInformationByHandleEx
        get_information_ex.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            wintypes.LPVOID,
            wintypes.DWORD,
        ]
        get_information_ex.restype = wintypes.BOOL
        create_file = kernel32.CreateFileW
        create_file.argtypes = [
            wintypes.LPCWSTR,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.LPVOID,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.HANDLE,
        ]
        create_file.restype = wintypes.HANDLE
        invalid = ctypes.c_void_p(-1).value
        # DOS drive mappings (including SUBST) are mutable. The root and every
        # descendant therefore receive held handles, and all later I/O uses a
        # Volume-GUID path derived from those handles instead of the DOS name.
        components = self.captured
        original_native = self.snapshot.windows
        if original_native is None or len(original_native) != len(components):
            raise schema.TypedIngressError(
                "Windows directory anchor lacks a complete original"
                " full-native chain snapshot"
            )

        def metadata(handle: Any) -> tuple[int, int, tuple[int, bytes]]:
            information = _ByHandleFileInformation()
            file_id_information = _FileIdInformation()
            if not get_information(handle, ctypes.byref(information)) or not (
                get_information_ex(
                    handle,
                    18,  # FileIdInfo
                    ctypes.byref(file_id_information),
                    ctypes.sizeof(file_id_information),
                )
            ):
                error = ctypes.get_last_error()
                raise schema.TypedIngressError(
                    f"cannot verify {self.label} directory-chain anchor"
                    f" (WindowsError {error})"
                )
            attributes = int(information.dwFileAttributes)
            legacy_file_index = (int(information.nFileIndexHigh) << 32) | int(
                information.nFileIndexLow
            )
            return (
                attributes,
                legacy_file_index,
                self._windows_native_file_id(file_id_information),
            )

        def validate_metadata(
            observed: tuple[int, int, tuple[int, bytes]],
        ) -> None:
            attributes, _legacy_file_index, _native_file_id = observed
            if (
                not attributes
                & getattr(stat, "FILE_ATTRIBUTE_DIRECTORY", 0x10)
                or attributes
                & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
            ):
                raise schema.TypedIngressError(
                    f"{self.label} chain handle does not identify its captured"
                    " ordinary directory"
                )

        final_paths: list[str] = []
        for index, ((component, _captured_identity), original) in enumerate(
            zip(components, original_native)
        ):
            if os.path.normcase(component) != original.component:
                raise schema.TypedIngressError(
                    f"{self.label} original native chain component order changed"
                )
            # Capture the complete native identity without blocking a peer's
            # delete request, then immediately reacquire with the requested
            # share policy and require byte-for-byte identity equality.
            temporary = create_file(
                component,
                0x80000000,
                0x00000001 | 0x00000002 | 0x00000004,
                None,
                3,
                0x02000000 | 0x00200000,
                None,
            )
            if temporary == invalid:
                error = ctypes.get_last_error()
                raise schema.TypedIngressError(
                    f"cannot pre-capture {self.label} directory-chain identity"
                    f" (WindowsError {error})"
                )
            try:
                precaptured = metadata(temporary)
            finally:
                close_handle(temporary)
            handle = create_file(
                component,
                0x80000000
                | (
                    0x00010000
                    if self._delete_access and index == len(components) - 1
                    else 0
                ),  # GENERIC_READ plus optional DELETE on the final component
                0x00000001
                | 0x00000002
                | (0x00000004 if self._share_delete else 0),
                None,
                3,  # OPEN_EXISTING
                0x02000000 | 0x00200000,  # BACKUP_SEMANTICS | OPEN_REPARSE_POINT
                None,
            )
            if handle == invalid:
                error = ctypes.get_last_error()
                raise schema.TypedIngressError(
                    f"cannot acquire {self.label} directory-chain anchor"
                    f" (WindowsError {error})"
                )
            self._win_handles.append(int(handle))
            anchored = metadata(handle)
            validate_metadata(precaptured)
            validate_metadata(anchored)
            expected_native = (original.volume_serial, original.file_id)
            if precaptured[2] != expected_native or anchored[2] != expected_native:
                raise schema.TypedIngressError(
                    f"{self.label} native volume/file identity does not match"
                    " its original full-native chain snapshot"
                )
            if anchored[2] != precaptured[2]:
                raise schema.TypedIngressError(
                    f"{self.label} native volume/file identity changed before"
                    " its no-delete-share anchor was acquired"
                )
            final_path = (
                self._require_local_volume_guid_path(
                    self._windows_final_volume_path(int(handle))
                )
            )
            if final_path.rstrip("\\").casefold() != (
                original.final_volume_path.rstrip("\\").casefold()
            ):
                raise schema.TypedIngressError(
                    f"{self.label} Volume-GUID component path changed from its"
                    " original native snapshot"
                )
            final_paths.append(final_path)
        normalized_paths = [path.rstrip("\\").casefold() for path in final_paths]
        for parent, child in zip(normalized_paths, normalized_paths[1:]):
            if not child.startswith(parent + "\\"):
                raise schema.TypedIngressError(
                    f"{self.label} Volume-GUID chain has mixed or"
                    " non-descendant components"
                )
        self._operation_path = Path(final_paths[-1])

    def revalidate(self, stage: str) -> None:
        if os.name == "posix":
            self._verify_posix_handle()
        elif not self._win_handles:  # pragma: no cover - internal invariant
            raise RuntimeError("Windows directory anchor is not open")
        _require_unchanged_directory(
            self.path, self.captured, f"{self.label} during {stage}"
        )

    @staticmethod
    def _name(name: str) -> str:
        if (
            not name
            or name in {".", ".."}
            or Path(name).name != name
            or "/" in name
            or "\\" in name
        ):
            raise schema.ConfigSurfaceError(
                "anchored child name must be one lexical path component"
            )
        return name

    def _path(self, name: str) -> Path:
        parent = self._operation_path if self._operation_path is not None else self.path
        return parent / self._name(name)

    def stat(self, name: str) -> os.stat_result:
        name = self._name(name)
        if self._fd is not None:
            return os.stat(name, dir_fd=self._fd, follow_symlinks=False)
        return os.stat(self._path(name), follow_symlinks=False)

    def exists(self, name: str) -> bool:
        try:
            self.stat(name)
        except FileNotFoundError:
            return False
        return True

    @staticmethod
    def _require_regular(info: os.stat_result, label: str) -> None:
        if (
            stat.S_ISLNK(info.st_mode)
            or getattr(info, "st_file_attributes", 0)
            & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
            or not stat.S_ISREG(info.st_mode)
        ):
            raise schema.TypedIngressError(
                f"{label} is not an ordinary regular file"
            )

    def read_regular(self, name: str, *, max_bytes: int) -> bytes:
        name = self._name(name)
        self.revalidate(f"read {name} precheck")
        before = self.stat(name)
        self._require_regular(before, name)
        flags = (
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0)
            | getattr(os, "O_BINARY", 0)
        )
        if self._fd is not None:
            fd = os.open(name, flags, dir_fd=self._fd)
        else:
            fd = os.open(self._path(name), flags)
        try:
            opened = os.fstat(fd)
            self._require_regular(opened, name)
            if schema._identity_tuple(opened) != schema._identity_tuple(before):
                raise schema.TypedIngressError(
                    f"{name} changed identity during anchored read"
                )
            if opened.st_size > max_bytes:
                raise schema.TypedIngressError(
                    f"{name} exceeds the maximum admissible byte count"
                )
            if os.name != "nt" and hasattr(os, "set_blocking"):
                os.set_blocking(fd, True)
            chunks: list[bytes] = []
            remaining = max_bytes + 1
            while remaining:
                chunk = os.read(fd, min(1 << 20, remaining))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            data = b"".join(chunks)
            if len(data) > max_bytes:
                raise schema.TypedIngressError(
                    f"{name} exceeds the maximum admissible byte count"
                )
        finally:
            os.close(fd)
        self.revalidate(f"read {name} completion")
        return data

    def create_once(
        self,
        name: str,
        data: bytes,
        *,
        exists_label: str,
        mode: int = 0o600,
    ) -> None:
        name = self._name(name)
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_BINARY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            if self._fd is not None:
                fd = os.open(name, flags, mode, dir_fd=self._fd)
            else:
                fd = os.open(self._path(name), flags, mode)
        except FileExistsError as exc:
            raise FileExistsError(
                f"{exists_label} already exists: {self._path(name)}"
            ) from exc
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        self.sync()
        self.revalidate(f"create {name} completion")

    def create_directory_once(
        self, name: str, *, exists_label: str, mode: int = 0o700
    ) -> tuple[int, int, int]:
        """Create and immediately claim one directory generation.

        Portable directory creation APIs do not return a generation handle.
        The unpredictable 128-bit child name therefore assumes cooperative
        same-account peers only until the immediately following no-follow
        claim succeeds.  After that claim, replacement is excluded or
        detected by exact identity.  If claiming itself fails, this method
        leaves at most that one unresolved empty child in place: deleting a
        lexical name whose generation was never claimed would be unsafe.
        """
        name = self._name(name)
        try:
            if self._fd is not None:
                os.mkdir(name, mode, dir_fd=self._fd)
            else:
                os.mkdir(self._path(name), mode)
        except FileExistsError as exc:
            raise FileExistsError(
                f"{exists_label} already exists: {self._path(name)}"
            ) from exc
        child_fd: int | None = None
        child_handle: int | None = None
        try:
            if self._fd is not None:
                child_fd = os.open(
                    name,
                    os.O_RDONLY
                    | os.O_DIRECTORY
                    | os.O_NOFOLLOW
                    | getattr(os, "O_CLOEXEC", 0),
                    dir_fd=self._fd,
                )
                created = os.fstat(child_fd)
            else:
                child_handle = self._open_created_windows_directory(name)
                # The no-delete-share native handle keeps this exact child
                # generation stable while its Python identity is captured.
                created = self.stat(name)
            if (
                stat.S_ISLNK(created.st_mode)
                or getattr(created, "st_file_attributes", 0)
                & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
                or not stat.S_ISDIR(created.st_mode)
            ):
                raise schema.TypedIngressError(
                    "created staging child is not an ordinary directory"
                )
            return schema._identity_tuple(created)
        except BaseException:
            if child_fd is not None:
                with contextlib.suppress(BaseException):
                    opened = os.fstat(child_fd)
                    current = os.stat(
                        name, dir_fd=self._fd, follow_symlinks=False
                    )
                    with os.scandir(child_fd) as entries:
                        empty = not self._bounded_names(entries, 0)
                    if (
                        empty
                        and stat.S_ISDIR(opened.st_mode)
                        and schema._identity_tuple(current)
                        == schema._identity_tuple(opened)
                    ):
                        os.rmdir(name, dir_fd=self._fd)
            elif child_handle is not None:
                with contextlib.suppress(BaseException):
                    self._delete_empty_windows_directory_handle(child_handle)
            raise
        finally:
            if child_fd is not None:
                os.close(child_fd)
            if child_handle is not None:
                self._close_windows_handle(child_handle)

    def _open_created_windows_directory(self, name: str) -> int:
        """Acquire one no-delete-share handle to a newly created child."""
        if os.name != "nt" or self._operation_path is None:
            raise RuntimeError("Windows created-directory claim is unavailable")
        import ctypes
        from ctypes import wintypes

        create_file = ctypes.WinDLL(
            "kernel32", use_last_error=True
        ).CreateFileW
        create_file.argtypes = [
            wintypes.LPCWSTR,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.LPVOID,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.HANDLE,
        ]
        create_file.restype = wintypes.HANDLE
        handle = create_file(
            str(self._path(name)),
            0x80000000 | 0x00010000,  # GENERIC_READ | DELETE
            0x00000001 | 0x00000002,  # share read/write, not delete
            None,
            3,  # OPEN_EXISTING
            0x02000000 | 0x00200000,  # BACKUP_SEMANTICS | OPEN_REPARSE_POINT
            None,
        )
        if handle == ctypes.c_void_p(-1).value:
            error = ctypes.get_last_error()
            raise OSError(error, "cannot claim created staging directory")
        return int(handle)

    @staticmethod
    def _close_windows_handle(handle: int) -> None:
        import ctypes
        from ctypes import wintypes

        close_handle = ctypes.WinDLL(
            "kernel32", use_last_error=True
        ).CloseHandle
        close_handle.argtypes = [wintypes.HANDLE]
        close_handle.restype = wintypes.BOOL
        close_handle(wintypes.HANDLE(handle))

    def _delete_empty_windows_directory_handle(self, handle: int) -> None:
        """Mark the exact claimed empty Windows directory for deletion."""
        import ctypes
        from ctypes import wintypes

        operation_path = Path(
            self._require_local_volume_guid_path(
                self._windows_final_volume_path(handle)
            )
        )
        with os.scandir(operation_path) as entries:
            if self._bounded_names(entries, 0):  # pragma: no cover - invariant
                raise schema.TypedIngressError(
                    "created staging directory is unexpectedly nonempty"
                )

        class _FileDispositionInformation(ctypes.Structure):
            _fields_ = [("delete_file", wintypes.BOOLEAN)]

        set_information = ctypes.WinDLL(
            "kernel32", use_last_error=True
        ).SetFileInformationByHandle
        set_information.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            wintypes.LPVOID,
            wintypes.DWORD,
        ]
        set_information.restype = wintypes.BOOL
        disposition = _FileDispositionInformation(True)
        if not set_information(
            wintypes.HANDLE(handle),
            4,  # FileDispositionInfo
            ctypes.byref(disposition),
            ctypes.sizeof(disposition),
        ):
            error = ctypes.get_last_error()
            raise OSError(error, "cannot delete claimed staging directory")

    def unlink(self, name: str) -> None:
        name = self._name(name)
        if self._fd is not None:
            os.unlink(name, dir_fd=self._fd)
        else:
            os.unlink(self._path(name))

    def sync(self) -> None:
        if self._fd is not None:
            os.fsync(self._fd)
        else:
            if self._operation_path is None:  # pragma: no cover - invariant
                raise RuntimeError("Windows operation path is not pinned")
            fileio.fsync_directory(self._operation_path)

    def delete_on_close(self) -> None:
        """Mark this exact empty Windows directory handle for deletion."""
        if self._fd is not None or os.name != "nt":
            raise RuntimeError("handle-bound deletion is Windows-only")
        if not self._delete_access or not self._win_handles:
            raise RuntimeError("directory anchor lacks delete access")
        import ctypes
        from ctypes import wintypes

        class _FileDispositionInformation(ctypes.Structure):
            _fields_ = [("delete_file", wintypes.BOOLEAN)]

        set_information = ctypes.WinDLL(
            "kernel32", use_last_error=True
        ).SetFileInformationByHandle
        set_information.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            wintypes.LPVOID,
            wintypes.DWORD,
        ]
        set_information.restype = wintypes.BOOL
        disposition = _FileDispositionInformation(True)
        if not set_information(
            wintypes.HANDLE(self._win_handles[-1]),
            4,  # FileDispositionInfo
            ctypes.byref(disposition),
            ctypes.sizeof(disposition),
        ):
            error = ctypes.get_last_error()
            raise OSError(error, "cannot delete exact staging directory handle")

    def sync_directory(
        self,
        name: str,
        expected_names: tuple[str, ...],
        *,
        child_snapshot: _CapturedDirectoryChain | None = None,
    ) -> None:
        """Sync one exact child tree without leaving the anchored generation."""
        name = self._name(name)
        if self._fd is None:
            child = self.path / name if child_snapshot is not None else self._path(name)
            if child_snapshot is None:
                fileio.fsync_tree(child)
            else:
                with _DirectoryAnchor(
                    child,
                    child_snapshot,
                    "anchored sync directory",
                    share_delete=True,
                ) as child_anchor:
                    operation_path = child_anchor._operation_path
                    if operation_path is None:  # pragma: no cover - invariant
                        raise RuntimeError("Windows sync path is not pinned")
                    fileio.fsync_tree(operation_path)
            self.revalidate(f"sync {name} completion")
            return
        child_fd = os.open(
            name,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
            dir_fd=self._fd,
        )
        try:
            opened = os.fstat(child_fd)
            if (
                child_snapshot is not None
                and schema._identity_tuple(opened)
                != child_snapshot.lexical[-1][1]
            ):
                raise schema.TypedIngressError(
                    "sync directory does not match its captured identity"
                )
            with os.scandir(child_fd) as entries:
                names = self._bounded_names(entries, len(expected_names))
            if set(names) != set(expected_names) or len(names) != len(
                expected_names
            ):
                raise schema.TypedIngressError(
                    "accepted output must contain exactly three ordinary files"
                )
            for child in sorted(expected_names):
                info = os.stat(child, dir_fd=child_fd, follow_symlinks=False)
                self._require_regular(info, child)
                file_fd = os.open(
                    child,
                    os.O_RDONLY | os.O_NOFOLLOW,
                    dir_fd=child_fd,
                )
                try:
                    os.fsync(file_fd)
                finally:
                    os.close(file_fd)
            os.fsync(child_fd)
        finally:
            os.close(child_fd)
        self.revalidate(f"sync {name} completion")

    def remove_exact_directory(
        self,
        name: str,
        expected_names: tuple[str, ...],
        child_snapshot: _CapturedDirectoryChain,
        *,
        allow_subset: bool = False,
    ) -> bool:
        """Remove one known flat child tree without recursive path traversal.

        Cleanup is deliberately narrower than publication validation.  A
        missing, incomplete, contaminated, aliased, or non-ordinary staging
        tree is left in place for explicit recovery rather than guessed at.
        """
        name = self._name(name)
        expected = set(expected_names)
        if (
            len(expected) != len(expected_names)
            or any(self._name(item) != item for item in expected)
        ):
            raise schema.ConfigSurfaceError(
                "expected staging membership is invalid"
            )
        try:
            child_info = self.stat(name)
        except FileNotFoundError:
            return False
        if schema._identity_tuple(child_info) != child_snapshot.lexical[-1][1]:
            return False
        if (
            stat.S_ISLNK(child_info.st_mode)
            or getattr(child_info, "st_file_attributes", 0)
            & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
            or not stat.S_ISDIR(child_info.st_mode)
        ):
            return False

        if self._fd is not None:
            flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(
                os, "O_CLOEXEC", 0
            )
            try:
                child_fd = os.open(name, flags, dir_fd=self._fd)
            except OSError:
                return False
            try:
                opened = os.fstat(child_fd)
                if (
                    not stat.S_ISDIR(opened.st_mode)
                    or schema._identity_tuple(opened)
                    != schema._identity_tuple(child_info)
                ):
                    return False
                with os.scandir(child_fd) as entries:
                    names = self._bounded_names(entries, len(expected))
                observed = set(names)
                if (
                    len(observed) != len(names)
                    or (not allow_subset and observed != expected)
                    or (allow_subset and not observed.issubset(expected))
                ):
                    return False
                for child in sorted(observed):
                    info = os.stat(
                        child, dir_fd=child_fd, follow_symlinks=False
                    )
                    self._require_regular(info, child)
                removed = False
                try:
                    for child in sorted(observed):
                        os.unlink(child, dir_fd=child_fd)
                        removed = True
                finally:
                    if removed:
                        with contextlib.suppress(OSError):
                            os.fsync(child_fd)
                current = os.stat(
                    name, dir_fd=self._fd, follow_symlinks=False
                )
                if schema._identity_tuple(current) != schema._identity_tuple(
                    opened
                ):
                    raise schema.TypedIngressError(
                        "staging directory changed identity during cleanup"
                    )
                os.rmdir(name, dir_fd=self._fd)
            finally:
                os.close(child_fd)
        else:
            # Reacquire against the original lexical snapshot; the nested
            # anchor then moves all operations onto its verified GUID path.
            child = self.path / name
            try:
                with _DirectoryAnchor(
                    child,
                    child_snapshot,
                    "staging cleanup directory",
                    delete_access=True,
                ) as child_anchor:
                    operation_path = child_anchor._operation_path
                    if operation_path is None:  # pragma: no cover - invariant
                        raise RuntimeError(
                            "Windows staging path is not pinned"
                        )
                    with os.scandir(operation_path) as entries:
                        names = self._bounded_names(entries, len(expected))
                    observed = set(names)
                    if (
                        len(observed) != len(names)
                        or (not allow_subset and observed != expected)
                        or (allow_subset and not observed.issubset(expected))
                    ):
                        return False
                    for item in sorted(observed):
                        self._require_regular(child_anchor.stat(item), item)
                    removed = False
                    try:
                        for item in sorted(observed):
                            child_anchor.unlink(item)
                            removed = True
                    finally:
                        if removed:
                            with contextlib.suppress(OSError):
                                child_anchor.sync()
                    child_anchor.delete_on_close()
            except (OSError, schema.ColmAimsError):
                return False
        self.sync()
        self.revalidate(f"remove {name} completion")
        return True

    def snapshot_directory(
        self,
        name: str,
        expected_names: tuple[str, ...],
        *,
        child_snapshot: _CapturedDirectoryChain | None = None,
    ) -> dict[str, bytes]:
        """Capture one flat, exact, ordinary-file directory under the anchor."""
        name = self._name(name)
        expected = set(expected_names)
        if not expected or any(self._name(item) != item for item in expected):
            raise schema.ConfigSurfaceError("expected output membership is invalid")
        if self._fd is not None:
            flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(
                os, "O_CLOEXEC", 0
            )
            try:
                child_fd = os.open(name, flags, dir_fd=self._fd)
            except OSError as exc:
                raise schema.TypedIngressError(
                    "anchored output cannot be opened as an ordinary directory"
                ) from exc
            try:
                info = os.fstat(child_fd)
                if (
                    not stat.S_ISDIR(info.st_mode)
                    or (
                        child_snapshot is not None
                        and schema._identity_tuple(info)
                        != child_snapshot.lexical[-1][1]
                    )
                ):
                    raise schema.TypedIngressError(
                        "anchored output identity is not the captured ordinary directory"
                    )
                with os.scandir(child_fd) as entries:
                    names = self._bounded_names(entries, len(expected))
                if set(names) != expected or len(names) != len(expected):
                    raise schema.TypedIngressError(
                        "accepted output must contain exactly three ordinary files"
                    )
                snapshot = {
                    child: self._read_regular_at(child_fd, child)
                    for child in sorted(expected)
                }
            finally:
                os.close(child_fd)
        else:
            child = self.path / name if child_snapshot is not None else self._path(name)
            chain = child_snapshot or _capture_directory_chain(child)
            with _DirectoryAnchor(
                child,
                chain,
                "anchored output directory",
                share_delete=child_snapshot is not None,
            ) as anchor:
                operation_path = anchor._operation_path
                if operation_path is None:  # pragma: no cover - invariant
                    raise RuntimeError("Windows output path is not pinned")
                with os.scandir(operation_path) as entries:
                    names = self._bounded_names(entries, len(expected))
                if set(names) != expected or len(names) != len(expected):
                    raise schema.TypedIngressError(
                        "accepted output must contain exactly three ordinary files"
                    )
                snapshot = {
                    item: anchor.read_regular(
                        item, max_bytes=schema.MAX_ARTIFACT_BYTES
                    )
                    for item in sorted(expected)
                }
        self.revalidate(f"snapshot {name} completion")
        return snapshot

    def snapshot_self(
        self, expected_names: tuple[str, ...]
    ) -> dict[str, bytes]:
        """Capture this held exact flat directory generation."""
        expected = set(expected_names)
        if (
            not expected
            or len(expected) != len(expected_names)
            or any(self._name(item) != item for item in expected)
        ):
            raise schema.ConfigSurfaceError("expected output membership is invalid")
        if self._fd is not None:
            with os.scandir(self._fd) as entries:
                names = self._bounded_names(entries, len(expected))
            if set(names) != expected or len(names) != len(expected):
                raise schema.TypedIngressError(
                    "accepted output must contain exactly three ordinary files"
                )
            return {
                item: self._read_regular_at(self._fd, item)
                for item in sorted(expected)
            }
        operation_path = self._operation_path
        if operation_path is None:  # pragma: no cover - invariant
            raise RuntimeError("Windows output path is not pinned")
        with os.scandir(operation_path) as entries:
            names = self._bounded_names(entries, len(expected))
        if set(names) != expected or len(names) != len(expected):
            raise schema.TypedIngressError(
                "accepted output must contain exactly three ordinary files"
            )
        return {
            item: self.read_regular(item, max_bytes=schema.MAX_ARTIFACT_BYTES)
            for item in sorted(expected)
        }

    def sync_self(self, expected_names: tuple[str, ...]) -> dict[str, bytes]:
        """Bound, validate, and durably sync this exact held flat tree."""
        before = self.snapshot_self(expected_names)
        access = os.O_RDWR if os.name == "nt" else os.O_RDONLY
        flags = (
            access
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0)
            | getattr(os, "O_BINARY", 0)
        )
        for name in sorted(expected_names):
            expected = self.stat(name)
            self._require_regular(expected, name)
            if expected.st_size > schema.MAX_ARTIFACT_BYTES:
                raise schema.TypedIngressError(
                    f"{name} exceeds the maximum admissible byte count"
                )
            if self._fd is not None:
                fd = os.open(name, flags, dir_fd=self._fd)
            else:
                fd = os.open(self._path(name), flags)
            try:
                opened = os.fstat(fd)
                self._require_regular(opened, name)
                if (
                    schema._identity_tuple(opened)
                    != schema._identity_tuple(expected)
                    or opened.st_size > schema.MAX_ARTIFACT_BYTES
                ):
                    raise schema.TypedIngressError(
                        f"{name} changed identity or size before durable sync"
                    )
                chunks: list[bytes] = []
                remaining = schema.MAX_ARTIFACT_BYTES + 1
                while remaining:
                    chunk = os.read(fd, min(1 << 20, remaining))
                    if not chunk:
                        break
                    chunks.append(chunk)
                    remaining -= len(chunk)
                data = b"".join(chunks)
                if len(data) > schema.MAX_ARTIFACT_BYTES:
                    raise schema.TypedIngressError(
                        f"{name} exceeds the maximum admissible byte count"
                    )
                if data != before[name]:
                    raise schema.TypedIngressError(
                        f"{name} changed bytes before durable sync"
                    )
                os.fsync(fd)
            finally:
                os.close(fd)
        self.sync()
        self.revalidate("staging tree sync completion")
        after = self.snapshot_self(expected_names)
        if after != before:
            raise schema.TypedIngressError(
                "staged release sidecars changed during durable sync"
            )
        return after

    def require_child_identity(
        self, name: str, child_snapshot: _CapturedDirectoryChain
    ) -> None:
        """Require a child name to identify one previously captured directory."""
        name = self._name(name)
        try:
            info = self.stat(name)
        except OSError as exc:
            raise schema.TypedIngressError(
                "captured publication directory is missing"
            ) from exc
        if (
            not stat.S_ISDIR(info.st_mode)
            or stat.S_ISLNK(info.st_mode)
            or getattr(info, "st_file_attributes", 0)
            & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
            or schema._identity_tuple(info) != child_snapshot.lexical[-1][1]
        ):
            raise schema.TypedIngressError(
                "publication directory does not match its captured identity"
            )
        if self._fd is None:
            child = self.path / name
            with _DirectoryAnchor(
                child,
                child_snapshot,
                "captured publication directory",
                share_delete=True,
            ):
                pass

    def _rename_windows_child_handle(
        self,
        source_anchor: "_DirectoryAnchor",
        destination_name: str,
    ) -> None:
        """Rename the exact held Windows source handle under this parent."""
        if (
            os.name != "nt"
            or self._fd is not None
            or not self._win_handles
            or not source_anchor._win_handles
            or not source_anchor._delete_access
        ):
            raise RuntimeError("exact Windows rename anchors are unavailable")
        import ctypes
        from ctypes import wintypes

        destination_path = str(self._path(destination_name))

        class _FileRenameInformation(ctypes.Structure):
            _fields_ = [
                ("replace_if_exists", wintypes.BOOLEAN),
                ("root_directory", wintypes.HANDLE),
                ("file_name_length", wintypes.DWORD),
                ("file_name", wintypes.WCHAR * len(destination_path)),
            ]

        encoded = destination_path.encode("utf-16-le")
        information = _FileRenameInformation()
        information.replace_if_exists = False
        information.root_directory = None
        information.file_name_length = len(encoded)
        information.file_name = destination_path
        set_information = ctypes.WinDLL(
            "kernel32", use_last_error=True
        ).SetFileInformationByHandle
        set_information.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            wintypes.LPVOID,
            wintypes.DWORD,
        ]
        set_information.restype = wintypes.BOOL
        if not set_information(
            wintypes.HANDLE(source_anchor._win_handles[-1]),
            3,  # FileRenameInfo
            ctypes.byref(information),
            ctypes.sizeof(information),
        ):
            error = ctypes.get_last_error()
            if error in {80, 183}:
                raise FileExistsError(
                    f"publication destination already exists: {destination_name}"
                )
            raise OSError(error, "cannot rename exact staging directory handle")

    @staticmethod
    def _bounded_names(entries: Any, expected_count: int) -> list[str]:
        """Enumerate at most expected_count+1 entries, rejecting overflow."""
        names: list[str] = []
        for entry in entries:
            names.append(entry.name)
            if len(names) > expected_count:
                raise schema.TypedIngressError(
                    "accepted output must contain exactly three ordinary files"
                )
        return names

    @staticmethod
    def _read_regular_at(directory_fd: int, name: str) -> bytes:
        info = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        _DirectoryAnchor._require_regular(info, name)
        flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_NONBLOCK", 0)
        fd = os.open(name, flags, dir_fd=directory_fd)
        try:
            opened = os.fstat(fd)
            _DirectoryAnchor._require_regular(opened, name)
            if schema._identity_tuple(opened) != schema._identity_tuple(info):
                raise schema.TypedIngressError(
                    f"{name} changed identity during anchored read"
                )
            if opened.st_size > schema.MAX_ARTIFACT_BYTES:
                raise schema.TypedIngressError(
                    f"{name} exceeds the maximum admissible byte count"
                )
            chunks: list[bytes] = []
            remaining = schema.MAX_ARTIFACT_BYTES + 1
            while remaining:
                chunk = os.read(fd, min(1 << 20, remaining))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            data = b"".join(chunks)
            if len(data) > schema.MAX_ARTIFACT_BYTES:
                raise schema.TypedIngressError(
                    f"{name} exceeds the maximum admissible byte count"
                )
            return data
        finally:
            os.close(fd)

    def publish_directory(
        self,
        staged_name: str,
        destination_name: str,
        *,
        exists_label: str,
        source_anchor: "_DirectoryAnchor",
        source_snapshot: _CapturedDirectoryChain,
        destination_snapshot: _CapturedDirectoryChain,
    ) -> None:
        staged_name = self._name(staged_name)
        destination_name = self._name(destination_name)
        if self._fd is not None:
            self.require_child_identity(staged_name, source_snapshot)
        committed = False
        try:
            if self._fd is None:
                self._rename_windows_child_handle(
                    source_anchor, destination_name
                )
            else:
                try:
                    os.mkdir(destination_name, dir_fd=self._fd)
                except FileExistsError as exc:
                    raise FileExistsError(
                        f"{exists_label} already exists:"
                        f" {self._path(destination_name)}"
                    ) from exc
                os.rename(
                    staged_name,
                    destination_name,
                    src_dir_fd=self._fd,
                    dst_dir_fd=self._fd,
                )
            committed = True
            self.require_child_identity(destination_name, destination_snapshot)
            self.sync()
        except BaseException as exc:
            if committed:
                cause = exc if isinstance(exc, OSError) else OSError(str(exc))
                raise fileio.DirectoryPublicationCommittedError(
                    self._path(destination_name), cause
                ) from exc
            raise


def _capture_directory_chain(path: Path) -> _CapturedDirectoryChain:
    """Capture one coherent lexical and full-native directory generation."""
    path = Path(os.path.abspath(path))
    lexical = schema.stable_directory_chain(path, path)
    if os.name != "nt":
        return _CapturedDirectoryChain(lexical=lexical)

    import ctypes
    from ctypes import wintypes

    class _ByHandleFileInformation(ctypes.Structure):
        _fields_ = [
            ("dwFileAttributes", wintypes.DWORD),
            ("ftCreationTime", wintypes.FILETIME),
            ("ftLastAccessTime", wintypes.FILETIME),
            ("ftLastWriteTime", wintypes.FILETIME),
            ("dwVolumeSerialNumber", wintypes.DWORD),
            ("nFileSizeHigh", wintypes.DWORD),
            ("nFileSizeLow", wintypes.DWORD),
            ("nNumberOfLinks", wintypes.DWORD),
            ("nFileIndexHigh", wintypes.DWORD),
            ("nFileIndexLow", wintypes.DWORD),
        ]

    class _FileId128(ctypes.Structure):
        _fields_ = [("identifier", ctypes.c_ubyte * 16)]

    class _FileIdInformation(ctypes.Structure):
        _fields_ = [
            ("volume_serial_number", ctypes.c_ulonglong),
            ("file_id", _FileId128),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = [wintypes.HANDLE]
    close_handle.restype = wintypes.BOOL
    create_file = kernel32.CreateFileW
    create_file.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    create_file.restype = wintypes.HANDLE
    get_information = kernel32.GetFileInformationByHandle
    get_information.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(_ByHandleFileInformation),
    ]
    get_information.restype = wintypes.BOOL
    get_information_ex = kernel32.GetFileInformationByHandleEx
    get_information_ex.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        wintypes.LPVOID,
        wintypes.DWORD,
    ]
    get_information_ex.restype = wintypes.BOOL
    invalid = ctypes.c_void_p(-1).value
    handles: list[int] = []
    captured: list[_WindowsComponentIdentity] = []
    try:
        for component, lexical_identity in lexical:
            handle = create_file(
                component,
                0x80000000,
                0x00000001 | 0x00000002,
                None,
                3,
                0x02000000 | 0x00200000,
                None,
            )
            if handle == invalid:
                error = ctypes.get_last_error()
                raise schema.TypedIngressError(
                    "cannot acquire original full-native directory snapshot"
                    f" (WindowsError {error})"
                )
            handles.append(int(handle))
            information = _ByHandleFileInformation()
            file_id_information = _FileIdInformation()
            if not get_information(handle, ctypes.byref(information)) or not (
                get_information_ex(
                    handle,
                    18,
                    ctypes.byref(file_id_information),
                    ctypes.sizeof(file_id_information),
                )
            ):
                error = ctypes.get_last_error()
                raise schema.TypedIngressError(
                    "cannot read original full-native directory identity"
                    f" (WindowsError {error})"
                )
            attributes = int(information.dwFileAttributes)
            if (
                not attributes
                & getattr(stat, "FILE_ATTRIBUTE_DIRECTORY", 0x10)
                or attributes
                & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
            ):
                raise schema.TypedIngressError(
                    "original native chain member is not an ordinary directory"
                )
            volume, file_id = _DirectoryAnchor._windows_native_file_id(
                file_id_information
            )
            if len(file_id) != 16:
                raise schema.TypedIngressError(
                    "original native chain member lacks a 128-bit file ID"
                )
            legacy_pair = (
                int(information.dwVolumeSerialNumber),
                (int(information.nFileIndexHigh) << 32)
                | int(information.nFileIndexLow),
            )
            full_native_pair = (volume, int.from_bytes(file_id, "little"))
            lexical_pair = lexical_identity[:2]
            if not _windows_stat_pair_matches(
                lexical_identity, full_native_pair, legacy_pair
            ):
                raise schema.TypedIngressError(
                    "lexical stat and native directory identity do not match"
                    " an exact supported Windows pair profile"
                )
            final_path = _DirectoryAnchor._require_local_volume_guid_path(
                _DirectoryAnchor._windows_final_volume_path(int(handle))
            )
            captured.append(
                _WindowsComponentIdentity(
                    component=os.path.normcase(component),
                    volume_serial=volume,
                    file_id=file_id,
                    final_volume_path=final_path,
                )
            )
        if schema.stable_directory_chain(path, path) != lexical:
            raise schema.TypedIngressError(
                "directory identity changed during original full-native"
                " chain snapshot"
            )
        normalized = [
            item.final_volume_path.rstrip("\\").casefold() for item in captured
        ]
        for parent, child in zip(normalized, normalized[1:]):
            if not child.startswith(parent + "\\"):
                raise schema.TypedIngressError(
                    "original full-native Volume-GUID chain has mixed or"
                    " non-descendant components"
                )
        return _CapturedDirectoryChain(
            lexical=lexical,
            windows=tuple(captured),
        )
    finally:
        while handles:
            close_handle(wintypes.HANDLE(handles.pop()))


@dataclass(frozen=True)
class ReleaseFinalizationResult:
    """The immutable published bundle and its successful verifier report."""

    published_dir: Path
    ledger_path: Path
    rights_path: Path
    expectations_path: Path
    report: verifier.VerificationReport


def _canonical_existing_directory(path: Path, label: str) -> Path:
    """Require an existing ordinary directory with no aliased ancestor."""
    candidate = Path(os.path.abspath(path))
    try:
        schema.stable_directory_chain(candidate, candidate)
    except schema.ColmAimsError as exc:
        raise schema.ConfigSurfaceError(
            f"{label} must be an existing alias-free directory: {exc}"
        ) from exc
    return candidate


def _require_unchanged_directory(
    path: Path,
    captured: _CapturedDirectoryChain | _LexicalChain,
    label: str,
) -> None:
    """Reject an alias or identity swap of a previously trusted directory."""
    try:
        observed = schema.stable_directory_chain(path, path)
    except schema.ColmAimsError as exc:
        raise schema.TypedIngressError(
            f"{label} changed or became aliased during the operation"
        ) from exc
    lexical = captured.lexical if isinstance(captured, _CapturedDirectoryChain) else captured
    if observed != lexical:
        raise schema.TypedIngressError(
            f"{label} identity changed during the operation"
        )


def _relocated_child_snapshot(
    parent_snapshot: _CapturedDirectoryChain,
    child_snapshot: _CapturedDirectoryChain,
    destination: Path,
) -> _CapturedDirectoryChain:
    """Bind a captured child identity to its post-rename sibling spelling."""
    if child_snapshot.lexical[:-1] != parent_snapshot.lexical:
        raise schema.TypedIngressError(
            "staging identity is not a child of the publication parent"
        )
    destination = Path(os.path.abspath(destination))
    component = os.path.normcase(str(destination))
    lexical = parent_snapshot.lexical + (
        (component, child_snapshot.lexical[-1][1]),
    )
    parent_windows = parent_snapshot.windows
    child_windows = child_snapshot.windows
    if parent_windows is None and child_windows is None:
        return _CapturedDirectoryChain(lexical=lexical)
    if (
        parent_windows is None
        or child_windows is None
        or child_windows[:-1] != parent_windows
    ):
        raise schema.TypedIngressError(
            "staging native identity is not a child of the publication parent"
        )
    leaf = child_windows[-1]
    parent_volume_path = parent_windows[-1].final_volume_path.rstrip("\\")
    moved_leaf = _WindowsComponentIdentity(
        component=component,
        volume_serial=leaf.volume_serial,
        file_id=leaf.file_id,
        final_volume_path=f"{parent_volume_path}\\{destination.name}",
    )
    return _CapturedDirectoryChain(
        lexical=lexical,
        windows=parent_windows + (moved_leaf,),
    )


def _remove_exact_staged_directory(
    *,
    parent: Path,
    parent_snapshot: _CapturedDirectoryChain,
    staged_name: str,
    staged_snapshot: _CapturedDirectoryChain,
    expected_names: tuple[str, ...],
    allow_subset: bool = False,
) -> bool:
    """Best-effort cleanup confined to the captured parent generation."""
    try:
        with _DirectoryAnchor(
            parent, parent_snapshot, "staging cleanup parent"
        ) as anchor:
            return anchor.remove_exact_directory(
                staged_name,
                expected_names,
                staged_snapshot,
                allow_subset=allow_subset,
            )
    except (OSError, schema.ColmAimsError):
        # Safe orphaning is preferable to resolving a mutable lexical path.
        return False


def _materialize_staged_directory(
    staged: Path,
    parent_snapshot: _CapturedDirectoryChain,
    staged_snapshot: _CapturedDirectoryChain,
    expected_snapshot: dict[str, bytes],
    expected_names: tuple[str, ...],
    *,
    label: str,
) -> dict[str, bytes]:
    """Create a flat staging tree through its captured directory generation."""
    if (
        not expected_names
        or len(set(expected_names)) != len(expected_names)
        or set(expected_snapshot) != set(expected_names)
    ):
        raise schema.ConfigSurfaceError("staging membership is invalid")
    with (
        _DirectoryAnchor(
            Path(staged).parent,
            parent_snapshot,
            f"{label} parent",
        ) as parent_anchor,
        _DirectoryAnchor(staged, staged_snapshot, label) as staged_anchor,
    ):
        parent_anchor.require_child_identity(Path(staged).name, staged_snapshot)
        for name in expected_names:
            staged_anchor.create_once(
                name,
                expected_snapshot[name],
                exists_label=f"{label} member",
                mode=0o666,
            )
        observed = staged_anchor.snapshot_self(expected_names)
        parent_anchor.require_child_identity(Path(staged).name, staged_snapshot)
    if observed != expected_snapshot:
        raise schema.TypedIngressError(
            "captured staging bytes differ from the requested bytes"
        )
    return observed


def _create_staged_directory(
    parent: Path,
    parent_snapshot: _CapturedDirectoryChain,
    *,
    prefix: str,
    label: str,
) -> tuple[Path, _CapturedDirectoryChain]:
    """Create and capture one unpredictable child through a held parent."""
    with _DirectoryAnchor(parent, parent_snapshot, f"{label} parent") as anchor:
        for _attempt in range(32):
            name = f"{prefix}{secrets.token_hex(16)}"
            try:
                created_identity = anchor.create_directory_once(
                    name, exists_label=label
                )
            except FileExistsError:
                continue
            staged = Path(parent) / name
            staged_snapshot: _CapturedDirectoryChain | None = None
            try:
                candidate = _capture_directory_chain(staged)
                if candidate.lexical[-1][1] != created_identity:
                    raise schema.TypedIngressError(
                        "created staging directory changed identity before capture"
                    )
                staged_snapshot = candidate
                anchor.require_child_identity(name, staged_snapshot)
                anchor.sync()
                anchor.revalidate(f"capture {label} completion")
                return staged, staged_snapshot
            except BaseException:
                cleanup_snapshot = staged_snapshot
                if cleanup_snapshot is None:
                    with contextlib.suppress(BaseException):
                        candidate = _capture_directory_chain(staged)
                        if candidate.lexical[-1][1] == created_identity:
                            cleanup_snapshot = candidate
                if cleanup_snapshot is not None:
                    with contextlib.suppress(BaseException):
                        anchor.remove_exact_directory(
                            name, (), cleanup_snapshot
                        )
                raise
    raise schema.TypedIngressError(
        f"cannot allocate a fresh {label} after 32 bounded attempts"
    )


def _require_publication_parent(
    destination: Path,
    parent_chain: _CapturedDirectoryChain,
    stage: str,
) -> None:
    """Bind every publication syscall to the originally validated parent."""
    _require_unchanged_directory(
        Path(destination).parent,
        parent_chain,
        f"publication parent during {stage}",
    )


def _require_disjoint(path: Path, root: Path, message: str) -> None:
    if schema.resolves_inside(path, root):
        raise schema.ConfigSurfaceError(message)


def _require_mutually_disjoint(left: Path, right: Path, message: str) -> None:
    """Reject either containment direction between two filesystem regions."""
    if schema.resolves_inside(left, right) or schema.resolves_inside(right, left):
        raise schema.ConfigSurfaceError(message)


def _require_portable_id(value: Any, label: str) -> str:
    """Require one bounded ASCII component portable across supported hosts."""
    if (
        not isinstance(value, str)
        or len(value.encode("utf-8")) > _PORTABLE_ID_MAX_BYTES
        or _PORTABLE_ID_RE.fullmatch(value) is None
        or value.endswith((".", " "))
        or value.split(".", 1)[0].upper() in _WINDOWS_RESERVED_BASENAMES
    ):
        raise schema.ConfigSurfaceError(
            f"{label} must be a portable 1-{_PORTABLE_ID_MAX_BYTES}-byte ASCII"
            " path component using letters, digits, dot, underscore, or"
            " hyphen; separators, colons/ADS spellings, trailing dot/space,"
            " and Windows reserved basenames are forbidden"
        )
    return value


def _require_unclaimed(path: Path, label: str) -> None:
    """Fail early on any existing destination entry; publication rechecks."""
    try:
        os.stat(Path(path), follow_symlinks=False)
    except FileNotFoundError:
        return
    except OSError as exc:
        raise schema.ConfigSurfaceError(
            f"cannot establish that {label} is unclaimed"
            f" ({exc.__class__.__name__})"
        ) from exc
    raise schema.ConfigSurfaceError(f"{label} already exists: {path}")


def _pending_guard_path(destination: Path) -> Path:
    """Return the durable sibling guard for one terminal publication slot."""
    destination = Path(destination)
    return destination.with_name(
        f".{destination.name}{_PENDING_GUARD_SUFFIX}"
    )


def _accepted_marker_path(destination: Path) -> Path:
    """Return the durable positive-acceptance marker for one final slot."""
    destination = Path(destination)
    return destination.with_name(
        f".{destination.name}{_ACCEPTED_MARKER_SUFFIX}"
    )


def _pending_guard_bytes(destination: Path) -> bytes:
    return schema.encode_json(
        {
            "schema_version": schema.SCHEMA_VERSION,
            "artifact_type": "colm_aims_2026_pending_publication_guard",
            "state": "PENDING",
            "target_name": Path(destination).name,
        }
    )


def _directory_tree_sha256(directory: Path) -> str:
    snapshot = verifier._read_tree_snapshot(directory)
    return verifier._tree_digest_from_shas(
        {
            rel: hashlib.sha256(data).hexdigest()
            for rel, data in snapshot.items()
        }
    )


def _accepted_marker_bytes(destination: Path, tree_sha256: str) -> bytes:
    return schema.encode_json(
        {
            "schema_version": schema.SCHEMA_VERSION,
            "artifact_type": "colm_aims_2026_accepted_publication_marker",
            "state": "ACCEPTED",
            "target_name": Path(destination).name,
            "tree_sha256": tree_sha256,
        }
    )


def _create_once_in_bound_parent(
    path: Path,
    data: bytes,
    *,
    destination: Path,
    parent_chain: _CapturedDirectoryChain,
    exists_label: str,
    anchor: _DirectoryAnchor | None = None,
) -> None:
    """Durably create one sibling without ever creating its parent."""
    if anchor is None:
        with _DirectoryAnchor(
            Path(destination).parent, parent_chain, "publication parent"
        ) as opened:
            _create_once_in_bound_parent(
                path,
                data,
                destination=destination,
                parent_chain=parent_chain,
                exists_label=exists_label,
                anchor=opened,
            )
        return
    _require_publication_parent(destination, parent_chain, f"{exists_label} open")
    anchor.create_once(Path(path).name, data, exists_label=exists_label)
    _require_publication_parent(
        destination, parent_chain, f"{exists_label} parent sync"
    )


def _require_published_tree(
    destination: Path,
    expected_tree_sha256: str,
    parent_chain: _CapturedDirectoryChain,
    stage: str,
    expected_names: tuple[str, ...],
    anchor: _DirectoryAnchor,
    destination_snapshot: _CapturedDirectoryChain,
) -> None:
    """Rebind the committed directory bytes to the staged precommit digest."""
    _require_publication_parent(destination, parent_chain, f"{stage} precheck")
    observed = verifier._tree_digest_from_shas(
        {
            rel: hashlib.sha256(data).hexdigest()
            for rel, data in anchor.snapshot_directory(
                Path(destination).name,
                expected_names,
                child_snapshot=destination_snapshot,
            ).items()
        }
    )
    _require_publication_parent(destination, parent_chain, f"{stage} postcheck")
    if observed != expected_tree_sha256:
        raise schema.TypedIngressError(
            "published directory tree differs from the verified staged tree"
        )


def _read_accepted_directory_snapshot(
    path: Path,
    label: str,
    *,
    expected_names: tuple[str, ...] = _SIDECAR_NAMES,
) -> tuple[Path, dict[str, bytes]]:
    """Return the exact snapshot bound by a stable positive marker.

    The first snapshot is the sole authoritative byte capture returned to the
    consumer. A second capture is only a concurrency postcheck; its bytes are
    never substituted into the return value.
    """
    directory = Path(os.path.abspath(path))
    directory = _canonical_existing_directory(directory, label)
    parent = directory.parent
    parent_chain = _capture_directory_chain(parent)
    guard_name = _pending_guard_path(directory).name
    marker_name = _accepted_marker_path(directory).name
    with _DirectoryAnchor(parent, parent_chain, f"{label} parent") as anchor:
        if anchor.exists(guard_name):
            raise schema.TypedIngressError(
                f"{label} has an unresolved pending publication guard"
            )
        snapshot = anchor.snapshot_directory(directory.name, expected_names)
        tree_sha256 = verifier._tree_digest_from_shas(
            {
                rel: hashlib.sha256(data).hexdigest()
                for rel, data in snapshot.items()
            }
        )
        expected = _accepted_marker_bytes(directory, tree_sha256)
        try:
            observed = anchor.read_regular(marker_name, max_bytes=len(expected))
        except (OSError, schema.ColmAimsError) as exc:
            raise schema.TypedIngressError(
                f"{label} has no valid positive acceptance marker"
            ) from exc
        if observed != expected:
            raise schema.TypedIngressError(
                f"{label} positive acceptance marker does not bind its exact tree"
            )
        if anchor.exists(guard_name):
            raise schema.TypedIngressError(
                f"{label} became pending during accepted snapshot capture"
            )
        if anchor.snapshot_directory(directory.name, expected_names) != snapshot:
            raise schema.TypedIngressError(
                f"{label} tree changed during accepted snapshot capture"
            )
    return directory, snapshot


def _require_accepted_directory(
    path: Path,
    label: str,
    *,
    expected_names: tuple[str, ...] = _SIDECAR_NAMES,
) -> Path:
    """Require exact positive acceptance and no sibling pending-state guard."""
    directory, _ = _read_accepted_directory_snapshot(
        path, label, expected_names=expected_names
    )
    return directory


def _create_pending_guard(
    destination: Path,
    *,
    parent_chain: _CapturedDirectoryChain,
    anchor: _DirectoryAnchor | None = None,
) -> tuple[Path, bytes]:
    if anchor is None:
        with _DirectoryAnchor(
            Path(destination).parent, parent_chain, "publication parent"
        ) as opened:
            return _create_pending_guard(
                destination, parent_chain=parent_chain, anchor=opened
            )
    _require_publication_parent(destination, parent_chain, "guard precheck")
    guard = _pending_guard_path(destination)
    if anchor.exists(guard.name):
        raise schema.ConfigSurfaceError(
            f"pending publication guard already exists: {guard}"
        )
    marker = _accepted_marker_path(destination)
    if anchor.exists(marker.name):
        raise schema.ConfigSurfaceError(
            f"positive acceptance marker already exists: {marker}"
        )
    encoded = _pending_guard_bytes(destination)
    _require_publication_parent(destination, parent_chain, "guard creation")
    _create_once_in_bound_parent(
        guard,
        encoded,
        destination=destination,
        parent_chain=parent_chain,
        exists_label="pending publication guard",
        anchor=anchor,
    )
    _require_publication_parent(destination, parent_chain, "guard readback")
    observed = anchor.read_regular(guard.name, max_bytes=len(encoded))
    if observed != encoded:
        raise schema.TypedIngressError(
            "pending publication guard differs from its deterministic bytes"
        )
    _require_publication_parent(destination, parent_chain, "guard completion")
    return guard, encoded


def _create_accepted_marker(
    destination: Path,
    tree_sha256: str,
    *,
    parent_chain: _CapturedDirectoryChain,
    anchor: _DirectoryAnchor | None = None,
) -> None:
    if anchor is None:
        with _DirectoryAnchor(
            Path(destination).parent, parent_chain, "publication parent"
        ) as opened:
            _create_accepted_marker(
                destination,
                tree_sha256,
                parent_chain=parent_chain,
                anchor=opened,
            )
        return
    _require_publication_parent(destination, parent_chain, "marker creation")
    marker = _accepted_marker_path(destination)
    encoded = _accepted_marker_bytes(destination, tree_sha256)
    _create_once_in_bound_parent(
        marker,
        encoded,
        destination=destination,
        parent_chain=parent_chain,
        exists_label="positive acceptance marker",
        anchor=anchor,
    )
    _require_publication_parent(destination, parent_chain, "marker readback")
    observed = anchor.read_regular(marker.name, max_bytes=len(encoded))
    if observed != encoded:
        raise schema.TypedIngressError(
            "positive acceptance marker differs from its deterministic bytes"
        )
    _require_publication_parent(destination, parent_chain, "marker completion")


def _require_positive_acceptance_state(
    destination: Path,
    tree_sha256: str,
    parent_chain: _CapturedDirectoryChain,
    expected_names: tuple[str, ...],
    anchor: _DirectoryAnchor,
    destination_snapshot: _CapturedDirectoryChain,
) -> None:
    """Validate committed tree and marker while the pending guard is live."""
    _require_published_tree(
        destination,
        tree_sha256,
        parent_chain,
        "acceptance tree validation",
        expected_names,
        anchor,
        destination_snapshot,
    )
    marker = _accepted_marker_path(destination)
    expected = _accepted_marker_bytes(destination, tree_sha256)
    observed = anchor.read_regular(marker.name, max_bytes=len(expected))
    _require_publication_parent(
        destination, parent_chain, "acceptance marker validation"
    )
    if observed != expected:
        raise schema.TypedIngressError(
            "positive acceptance marker changed before guard retirement"
        )


def _retire_pending_guard(
    guard: Path,
    encoded: bytes,
    *,
    destination: Path,
    parent_chain: _CapturedDirectoryChain,
    anchor: _DirectoryAnchor | None = None,
    destination_snapshot: _CapturedDirectoryChain | None = None,
) -> None:
    """Remove the guard only after the final directory is independently durable."""
    if anchor is None:
        with _DirectoryAnchor(
            Path(destination).parent, parent_chain, "publication parent"
        ) as opened:
            _retire_pending_guard(
                guard,
                encoded,
                destination=destination,
                parent_chain=parent_chain,
                anchor=opened,
                destination_snapshot=destination_snapshot,
            )
        return
    _require_publication_parent(destination, parent_chain, "guard retirement")
    if destination_snapshot is not None:
        anchor.require_child_identity(destination.name, destination_snapshot)
    observed = anchor.read_regular(guard.name, max_bytes=len(encoded))
    if observed != encoded:
        raise schema.TypedIngressError(
            "pending publication guard changed before acceptance"
        )
    _require_publication_parent(destination, parent_chain, "guard unlink")
    if destination_snapshot is not None:
        anchor.require_child_identity(destination.name, destination_snapshot)
    anchor.unlink(guard.name)
    _require_publication_parent(destination, parent_chain, "guard unlink commit")
    try:
        anchor.sync()
        _require_publication_parent(
            destination, parent_chain, "guard retirement completion"
        )
    except OSError:
        # The directory artifact was already made durable before guard
        # retirement. Retry the acceptance-state durability barrier once;
        # exhaustion is classified explicitly by the publishing protocol.
        try:
            _require_publication_parent(
                destination, parent_chain, "guard retirement retry"
            )
            anchor.sync()
            _require_publication_parent(
                destination, parent_chain, "guard retirement retry completion"
            )
        except OSError as second_exc:
            raise _GuardRetirementCommittedError(
                "pending guard removal committed after durable positive"
                " acceptance, but both parent sync attempts failed"
            ) from second_exc


def _read_external(path: Path, *, runs_root: Path, output_root: Path) -> bytes:
    """Read one external input through the no-follow, stable-chain boundary."""
    source = Path(os.path.abspath(path))
    parent = _canonical_existing_directory(source.parent, "external input parent")
    _require_disjoint(
        source,
        runs_root,
        "release authority inputs must remain outside the runs root",
    )
    _require_disjoint(
        source,
        output_root,
        "release authority inputs must remain outside the publication root",
    )
    return schema.read_regular_file_bytes(source, tree_root=parent)


def _validated_authority_input_paths(
    paths: tuple[Path, Path, Path],
    *,
    destination: Path,
    receipts_dir: Path,
) -> tuple[Path, Path, Path]:
    """Normalize distinct inputs and isolate their authority directories."""
    normalized = tuple(Path(os.path.abspath(path)) for path in paths)
    spellings = [os.path.normcase(str(path)) for path in normalized]
    if len(set(spellings)) != len(spellings):
        raise schema.ConfigSurfaceError(
            "ledger, rights, and expectations must be three distinct inputs"
        )
    authority_bases = {
        _canonical_existing_directory(path.parent, "authority input parent")
        for path in normalized
    }
    for authority_base in authority_bases:
        _require_mutually_disjoint(
            destination,
            authority_base,
            "release destination and authority input base must be disjoint",
        )
        _require_mutually_disjoint(
            receipts_dir,
            authority_base,
            "release receipts and authority input base must be disjoint",
        )
    return normalized


def _parse_object(data: bytes, label: str) -> dict[str, Any]:
    try:
        value = schema.parse_json_bytes_strict(data)
    except (UnicodeDecodeError, ValueError) as exc:
        raise schema.TypedIngressError(
            f"{label}: malformed JSON ({exc.__class__.__name__})"
        ) from exc
    if not isinstance(value, dict):
        raise schema.TypedIngressError(f"{label}: top-level value must be an object")
    schema.check_schema_version(value, label)
    return value


def _validate_authority_documents(
    *,
    run_id: str,
    ledger_bytes: bytes,
    rights_bytes: bytes,
    expectations_bytes: bytes,
) -> None:
    """Validate exact external bytes without adding or changing any field."""
    ledger_doc = _parse_object(ledger_bytes, _LEDGER_NAME)
    rights_doc = _parse_object(rights_bytes, _RIGHTS_NAME)
    expectations_doc = _parse_object(expectations_bytes, _EXPECTATIONS_NAME)

    anchor = expectations_doc.get("anchor")
    rights_decl = expectations_doc.get("rights_inventory")
    if not isinstance(anchor, dict) or not isinstance(rights_decl, dict):
        raise schema.ConfigSurfaceError(
            "expectations must carry anchor and rights_inventory objects"
        )
    external_claim_ids = anchor.get("external_claim_ids")
    if not isinstance(external_claim_ids, list) or not all(
        isinstance(value, str) for value in external_claim_ids
    ):
        raise schema.ConfigSurfaceError(
            "expectations anchor.external_claim_ids must be a list of strings"
        )

    ledger_module.validate_ledger(
        ledger_doc, external_claim_ids=external_claim_ids
    )
    ledger_module.validate_rights_inventory(rights_doc)

    if ledger_doc.get("canonical_run_id") != run_id:
        raise schema.ConfigSurfaceError(
            "external ledger canonical_run_id does not equal the explicitly"
            f" selected run_id {run_id!r}"
        )
    if anchor.get("ledger_path") != _LEDGER_NAME:
        raise schema.ConfigSurfaceError(
            "external expectations anchor.ledger_path must be exactly"
            f" {_LEDGER_NAME!r} for the closed release bundle"
        )
    if rights_decl.get("path") != _RIGHTS_NAME:
        raise schema.ConfigSurfaceError(
            "external expectations rights_inventory.path must be exactly"
            f" {_RIGHTS_NAME!r} for the closed release bundle"
        )
    expected_ledger_hash = hashlib.sha256(ledger_bytes).hexdigest()
    if anchor.get("ledger_sha256") != expected_ledger_hash:
        raise schema.ConfigSurfaceError(
            "external expectations do not bind the exact supplied ledger bytes"
        )
    expected_rights_hash = hashlib.sha256(rights_bytes).hexdigest()
    if rights_decl.get("sha256") != expected_rights_hash:
        raise schema.ConfigSurfaceError(
            "external expectations do not bind the exact supplied rights bytes"
        )


def _read_bundle(directory: Path) -> dict[str, bytes]:
    directory = _canonical_existing_directory(directory, "release bundle")
    snapshot = _capture_directory_chain(directory)
    with _DirectoryAnchor(directory, snapshot, "staged release bundle") as anchor:
        return anchor.snapshot_self(_SIDECAR_NAMES)


def _sync_captured_bundle(
    directory: Path, directory_snapshot: _CapturedDirectoryChain
) -> dict[str, bytes]:
    """Durably sync one exact staged release bundle with bounded membership."""
    with _DirectoryAnchor(
        directory, directory_snapshot, "staged release bundle sync"
    ) as anchor:
        return anchor.sync_self(_SIDECAR_NAMES)


def _require_release_pass(
    *, runs_root: Path, expectations: Path, receipts_dir: Path, stage: str
) -> verifier.VerificationReport:
    report = verifier.run_release_over_runs_root(
        runs_root,
        expectations=expectations,
        receipts_dir=receipts_dir,
    )
    if report.verdict != verifier.VERDICT_RELEASE_PASS:
        failing = sorted(
            str(leg.get("leg_id"))
            for leg in report.legs
            if leg.get("status") == "FAIL"
        )
        raise ReleaseVerificationFailed(
            f"{stage} release verification did not reach PASS_RELEASE;"
            f" failing legs: {failing}"
        )
    return report


def _require_report_bindings(
    report: verifier.VerificationReport,
    *,
    tree_snapshot: dict[str, bytes],
    expectations_bytes: bytes,
    receipts_dir: Path,
) -> dict[str, str]:
    """Bind a PASS report receipt to the exact captured inputs and code."""
    if report.receipt_path is None:
        raise schema.TypedIngressError(
            "PASS_RELEASE verifier report did not provide a receipt"
        )
    receipt_path = Path(report.receipt_path)
    if not schema.resolves_inside(receipt_path, receipts_dir):
        raise schema.TypedIngressError(
            "PASS_RELEASE verifier receipt escaped the selected receipt directory"
        )
    receipt_bytes = schema.read_regular_file_bytes(
        receipt_path, tree_root=receipts_dir
    )
    receipt = _parse_object(receipt_bytes, receipt_path.name)
    expected_tree = verifier._tree_digest_from_shas(
        {
            rel: hashlib.sha256(data).hexdigest()
            for rel, data in tree_snapshot.items()
        }
    )
    expected_expectations = hashlib.sha256(expectations_bytes).hexdigest()
    expected_code = verifier._code_digest()
    required = {
        "input_tree_sha256": expected_tree,
        "expectations_anchor_sha256": expected_expectations,
        "verifier_code_sha256": expected_code,
    }
    for field, expected in required.items():
        if receipt.get(field) != expected:
            raise schema.TypedIngressError(
                f"PASS_RELEASE verifier receipt {field} does not bind the exact"
                " captured input"
            )
    if receipt.get("mode") != "release" or receipt.get("verdict") != (
        verifier.VERDICT_RELEASE_PASS
    ):
        raise schema.TypedIngressError(
            "PASS_RELEASE verifier receipt does not carry the release verdict"
        )
    return {
        **required,
        "verifier_revision": schema.VERIFIER_REVISION,
    }


def _publish_verified_directory(
    staged: Path,
    destination: Path,
    *,
    exists_label: str,
    parent_chain: _CapturedDirectoryChain,
    staged_chain: _CapturedDirectoryChain,
    expected_snapshot: dict[str, bytes],
    expected_names: tuple[str, ...] = _SIDECAR_NAMES,
) -> None:
    """Create-once publish guarded until the final directory is durable."""
    if Path(os.path.abspath(staged)).parent != Path(
        os.path.abspath(destination)
    ).parent:
        raise schema.ConfigSurfaceError(
            "staged and destination directories must be siblings under the"
            " validated publication parent"
        )
    destination_chain = _relocated_child_snapshot(
        parent_chain, staged_chain, destination
    )
    with (
        _DirectoryAnchor(
            Path(destination).parent, parent_chain, "publication parent"
        ) as anchor,
        _DirectoryAnchor(
            staged,
            staged_chain,
            "verified staging source",
            delete_access=os.name == "nt",
        ) as source_anchor,
    ):
        _require_publication_parent(destination, parent_chain, "transaction start")
        staged_bytes = source_anchor.snapshot_self(expected_names)
        if staged_bytes != expected_snapshot:
            raise schema.TypedIngressError(
                "captured staging bytes differ from the caller-verified bytes"
            )
        tree_sha256 = verifier._tree_digest_from_shas(
            {
                rel: hashlib.sha256(data).hexdigest()
                for rel, data in staged_bytes.items()
            }
        )
        _require_publication_parent(destination, parent_chain, "guard dispatch")
        guard, guard_bytes = _create_pending_guard(
            destination, parent_chain=parent_chain, anchor=anchor
        )
        published = False
        try:
            try:
                _require_publication_parent(
                    destination, parent_chain, "directory rename"
                )
                anchor.publish_directory(
                    Path(staged).name,
                    Path(destination).name,
                    exists_label=exists_label,
                    source_anchor=source_anchor,
                    source_snapshot=staged_chain,
                    destination_snapshot=destination_chain,
                )
                published = True
                _require_publication_parent(
                    destination, parent_chain, "directory rename completion"
                )
            except fileio.DirectoryPublicationCommittedError as exc:
                committed_destination = Path(exc.destination).absolute()
                allowed_destinations = {
                    Path(destination).absolute(),
                    anchor._path(Path(destination).name).absolute(),
                }
                if committed_destination not in allowed_destinations:
                    raise
                published = True
                _require_publication_parent(
                    destination, parent_chain, "committed rename recovery"
                )
                # Re-open and sync only through the held parent generation.
                anchor.snapshot_directory(
                    Path(destination).name,
                    expected_names,
                    child_snapshot=destination_chain,
                )
                anchor.sync_directory(
                    Path(destination).name,
                    expected_names,
                    child_snapshot=destination_chain,
                )
                anchor.sync()
                _require_publication_parent(
                    destination, parent_chain, "committed parent sync"
                )
            except FileExistsError as exc:
                raise schema.ConfigSurfaceError(str(exc)) from exc
        except BaseException:
            if not published:
                with contextlib.suppress(BaseException):
                    _retire_pending_guard(
                        guard,
                        guard_bytes,
                        destination=destination,
                        parent_chain=parent_chain,
                        anchor=anchor,
                    )
            raise
        _require_published_tree(
            destination,
            tree_sha256,
            parent_chain,
            "postcommit tree validation",
            expected_names,
            anchor,
            destination_chain,
        )
        anchor.require_child_identity(
            Path(destination).name, destination_chain
        )
        try:
            _create_accepted_marker(
                destination,
                tree_sha256,
                parent_chain=parent_chain,
                anchor=anchor,
            )
        except BaseException:
            raise
        anchor.require_child_identity(
            Path(destination).name, destination_chain
        )
        _require_positive_acceptance_state(
            destination,
            tree_sha256,
            parent_chain,
            expected_names,
            anchor,
            destination_chain,
        )
        anchor.require_child_identity(
            Path(destination).name, destination_chain
        )
        try:
            _retire_pending_guard(
                guard,
                guard_bytes,
                destination=destination,
                parent_chain=parent_chain,
                anchor=anchor,
                destination_snapshot=destination_chain,
            )
        except _GuardRetirementCommittedError:
            return


def finalize_release(
    *,
    runs_root: Path,
    run_id: str,
    ledger_input: Path,
    rights_input: Path,
    expectations_input: Path,
    output_root: Path,
    release_id: str,
    receipts_dir: Path,
) -> ReleaseFinalizationResult:
    """Verify exact external sidecars, then publish them atomically.

    ``output_root`` and ``receipts_dir`` must already exist.  This prevents a
    caller typo from silently creating authority directories through an alias.
    The release bundle is a single create-once directory, avoiding a partially
    published three-file authority set.
    """
    _require_portable_id(run_id, "run_id")
    _require_portable_id(release_id, "release_id")

    runs_root = _canonical_existing_directory(runs_root, "runs root")
    output_root = _canonical_existing_directory(output_root, "output root")
    receipts_dir = _canonical_existing_directory(receipts_dir, "receipts root")
    output_root_chain = _capture_directory_chain(output_root)
    receipts_chain = _capture_directory_chain(receipts_dir)
    destination = output_root / release_id
    _require_unclaimed(destination, "release sidecar bundle")

    _require_disjoint(
        destination,
        runs_root,
        "release bundle destination must be outside the runs root",
    )
    _require_disjoint(
        receipts_dir,
        runs_root,
        "release receipts must be outside the runs root",
    )
    if schema.resolves_inside(receipts_dir, destination) or schema.resolves_inside(
        destination, receipts_dir
    ):
        raise schema.ConfigSurfaceError(
            "release bundle and receipt directory must be disjoint"
        )

    input_paths = _validated_authority_input_paths(
        (Path(ledger_input), Path(rights_input), Path(expectations_input)),
        destination=destination,
        receipts_dir=receipts_dir,
    )

    supplied = {
        _LEDGER_NAME: _read_external(
            input_paths[0], runs_root=runs_root, output_root=output_root
        ),
        _RIGHTS_NAME: _read_external(
            input_paths[1], runs_root=runs_root, output_root=output_root
        ),
        _EXPECTATIONS_NAME: _read_external(
            input_paths[2], runs_root=runs_root, output_root=output_root
        ),
    }
    _validate_authority_documents(
        run_id=run_id,
        ledger_bytes=supplied[_LEDGER_NAME],
        rights_bytes=supplied[_RIGHTS_NAME],
        expectations_bytes=supplied[_EXPECTATIONS_NAME],
    )

    ledger_doc = _parse_object(supplied[_LEDGER_NAME], _LEDGER_NAME)
    selected = verifier.resolve_canonical_package(runs_root, ledger_doc)
    if selected.name != run_id:
        raise schema.ConfigSurfaceError(
            "canonical release selection does not equal the requested run_id"
        )
    selected_tree = selected / "tree"
    tree_snapshot = verifier._read_tree_snapshot(selected_tree)

    _require_unchanged_directory(
        output_root, output_root_chain, "release output root"
    )
    staged, staged_snapshot = _create_staged_directory(
        output_root,
        output_root_chain,
        prefix=".release-staged-",
        label="release staging directory",
    )
    if (
        staged_snapshot.lexical[:-1] != output_root_chain.lexical
        or (
            output_root_chain.windows is not None
            and staged_snapshot.windows is not None
            and staged_snapshot.windows[:-1] != output_root_chain.windows
        )
        or (output_root_chain.windows is None) != (staged_snapshot.windows is None)
    ):
        raise schema.TypedIngressError(
            "staging directory is not a child of the captured output root"
        )
    try:
        _materialize_staged_directory(
            staged,
            output_root_chain,
            staged_snapshot,
            supplied,
            _SIDECAR_NAMES,
            label="release staging directory",
        )
        if _read_bundle(staged) != supplied:
            raise schema.TypedIngressError(
                "staged release sidecar bytes differ from the external inputs"
            )
        report = _require_release_pass(
            runs_root=runs_root,
            expectations=staged / _EXPECTATIONS_NAME,
            receipts_dir=receipts_dir,
            stage="prepublication",
        )
        if verifier._read_tree_snapshot(selected_tree) != tree_snapshot:
            raise schema.TypedIngressError(
                "selected release tree changed during verification"
            )
        _require_report_bindings(
            report,
            tree_snapshot=tree_snapshot,
            expectations_bytes=supplied[_EXPECTATIONS_NAME],
            receipts_dir=receipts_dir,
        )
        if _read_bundle(staged) != supplied:
            raise schema.TypedIngressError(
                "staged release sidecars changed during verification"
            )
        if _sync_captured_bundle(staged, staged_snapshot) != supplied:
            raise schema.TypedIngressError(
                "staged release sidecars changed before durable sync"
            )
        if verifier._read_tree_snapshot(selected_tree) != tree_snapshot:
            raise schema.TypedIngressError(
                "selected release tree changed before publication"
            )
        _require_unchanged_directory(
            output_root, output_root_chain, "release output root"
        )
        _require_unchanged_directory(
            receipts_dir, receipts_chain, "release receipts directory"
        )
        _publish_verified_directory(
            staged,
            destination,
            exists_label="release sidecar bundle",
            parent_chain=output_root_chain,
            staged_chain=staged_snapshot,
            expected_snapshot=supplied,
            expected_names=_SIDECAR_NAMES,
        )
        # From this point the complete public name is terminal: no fallible
        # verification, readback, cleanup stat, or other filesystem operation.
        staged = None
        return ReleaseFinalizationResult(
            published_dir=destination,
            ledger_path=destination / _LEDGER_NAME,
            rights_path=destination / _RIGHTS_NAME,
            expectations_path=destination / _EXPECTATIONS_NAME,
            report=report,
        )
    finally:
        if staged is not None:
            with contextlib.suppress(BaseException):
                _remove_exact_staged_directory(
                    parent=output_root,
                    parent_snapshot=output_root_chain,
                    staged_name=staged.name,
                    staged_snapshot=staged_snapshot,
                    expected_names=_SIDECAR_NAMES,
                )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m reproducibility.colm_aims_2026.phase4_finalize_release",
        description=(
            "Publish exact externally supplied, pre-authored release authority"
            " bytes into one create-once bundle and require canonical"
            " PASS_RELEASE. This command makes no signer/authorship claim."
        ),
        allow_abbrev=False,
    )
    parser.add_argument("--runs-root", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("--rights", required=True, type=Path)
    parser.add_argument("--expectations", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--receipts-dir", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else EXIT_USAGE_ERROR
    try:
        result = finalize_release(
            runs_root=args.runs_root,
            run_id=args.run_id,
            ledger_input=args.ledger,
            rights_input=args.rights,
            expectations_input=args.expectations,
            output_root=args.output_root,
            release_id=args.release_id,
            receipts_dir=args.receipts_dir,
        )
    except ReleaseVerificationFailed as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_VERIFY_FAIL
    except schema.TypedIngressError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_INGRESS_ERROR
    except schema.ColmAimsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE_ERROR
    except Exception as exc:  # noqa: BLE001 - pinned CLI internal-error class
        print(
            f"error: unexpected {exc.__class__.__name__} during release finalization",
            file=sys.stderr,
        )
        return EXIT_INTERNAL_ERROR
    print(
        "[release] PASS_RELEASE: exact external sidecars published at"
        f" {result.published_dir}"
    )
    return EXIT_PASS


if __name__ == "__main__":  # pragma: no cover - subprocess tests own CLI
    raise SystemExit(main())
