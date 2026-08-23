"""Windows-specific directory-fsync compatibility for durable publishers."""
from __future__ import annotations

import errno
import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.stopdff_v5 import fileio  # noqa: E402


def _deny_directory_open(monkeypatch, directory: Path, error: OSError):
    """Make only the target directory's ``os.open`` fail."""
    real_open = os.open
    opened: list[Path] = []

    def guarded_open(path, flags, *args, **kwargs):
        candidate = Path(path)
        opened.append(candidate)
        if candidate == directory:
            raise error
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(fileio.os, "open", guarded_open)
    return opened


def test_publish_bytes_windows_falls_back_to_published_file_fsync(
    tmp_path, monkeypatch
):
    """Windows' unsupported directory open does not make publication fail."""
    destination = tmp_path / "artifact.bin"
    opened = _deny_directory_open(
        monkeypatch,
        tmp_path,
        PermissionError(errno.EACCES, "directory handles are unsupported"),
    )
    monkeypatch.setattr(fileio, "_IS_WINDOWS", True, raising=False)

    real_fsync = os.fsync
    fsync_calls: list[int] = []

    def recording_fsync(fd: int):
        fsync_calls.append(fd)
        return real_fsync(fd)

    monkeypatch.setattr(fileio.os, "fsync", recording_fsync)

    fileio.publish_bytes(destination, b"durable")

    assert destination.read_bytes() == b"durable"
    assert tmp_path in opened
    assert destination in opened
    assert len(fsync_calls) == 2  # temp file before publish, live file after


def test_create_once_bytes_windows_uses_the_same_fallback(tmp_path, monkeypatch):
    destination = tmp_path / "create-once.bin"
    opened = _deny_directory_open(
        monkeypatch,
        tmp_path,
        PermissionError(errno.EACCES, "directory handles are unsupported"),
    )
    monkeypatch.setattr(fileio, "_IS_WINDOWS", True, raising=False)

    fileio.create_once_bytes(destination, b"create once")

    assert destination.read_bytes() == b"create once"
    assert destination in opened


def test_directory_open_permission_error_remains_fatal_off_windows(
    tmp_path, monkeypatch
):
    destination = tmp_path / "artifact.bin"
    _deny_directory_open(
        monkeypatch,
        tmp_path,
        PermissionError(errno.EACCES, "real permission failure"),
    )
    monkeypatch.setattr(fileio, "_IS_WINDOWS", False, raising=False)

    with pytest.raises(PermissionError, match="real permission failure"):
        fileio.publish_bytes(destination, b"payload")


def test_windows_directory_open_non_permission_error_remains_fatal(
    tmp_path, monkeypatch
):
    destination = tmp_path / "artifact.bin"
    _deny_directory_open(
        monkeypatch,
        tmp_path,
        OSError(errno.EIO, "directory I/O failure"),
    )
    monkeypatch.setattr(fileio, "_IS_WINDOWS", True, raising=False)

    with pytest.raises(OSError, match="directory I/O failure"):
        fileio.publish_bytes(destination, b"payload")


def test_file_fsync_error_remains_fatal_and_prevents_publication(
    tmp_path, monkeypatch
):
    destination = tmp_path / "artifact.bin"

    def fail_file_fsync(_fd: int):
        raise OSError(errno.EIO, "file fsync failed")

    monkeypatch.setattr(fileio.os, "fsync", fail_file_fsync)

    with pytest.raises(OSError, match="file fsync failed"):
        fileio.publish_bytes(destination, b"payload")

    assert not destination.exists()


def test_windows_directory_publish_renames_without_empty_claim(
    tmp_path, monkeypatch
):
    staged = tmp_path / "staged"
    staged.mkdir()
    (staged / "payload.bin").write_bytes(b"payload")
    destination = tmp_path / "published"
    monkeypatch.setattr(fileio, "_IS_WINDOWS", True)

    real_mkdir = os.mkdir
    mkdir_paths: list[Path] = []

    def reject_destination_claim(path, *args, **kwargs):
        candidate = Path(path)
        mkdir_paths.append(candidate)
        if candidate == destination:
            raise AssertionError("Windows must not pre-create the destination")
        return real_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(fileio.os, "mkdir", reject_destination_claim)

    fileio.publish_dir_create_once(staged, destination)

    assert destination not in mkdir_paths
    assert (destination / "payload.bin").read_bytes() == b"payload"
    assert not staged.exists()


def test_windows_directory_publish_translates_atomic_collision(
    tmp_path, monkeypatch
):
    staged = tmp_path / "staged"
    staged.mkdir()
    (staged / "payload.bin").write_bytes(b"new")
    destination = tmp_path / "published"
    destination.mkdir()
    (destination / "incumbent.bin").write_bytes(b"keep")
    monkeypatch.setattr(fileio, "_IS_WINDOWS", True)

    with pytest.raises(
        FileExistsError, match="Windows slot already exists"
    ):
        fileio.publish_dir_create_once(
            staged,
            destination,
            exists_label="Windows slot",
        )

    assert (destination / "incumbent.bin").read_bytes() == b"keep"
    assert (staged / "payload.bin").read_bytes() == b"new"


def test_posix_directory_publish_keeps_mkdir_claim_before_rename(
    tmp_path, monkeypatch
):
    staged = tmp_path / "staged"
    staged.mkdir()
    destination = tmp_path / "published"
    monkeypatch.setattr(fileio, "_IS_WINDOWS", False)

    real_mkdir = os.mkdir
    events: list[str] = []

    def recording_mkdir(path, *args, **kwargs):
        if Path(path) == destination:
            events.append("claim")
        return real_mkdir(path, *args, **kwargs)

    def emulate_posix_rename(source, target):
        assert Path(source) == staged
        assert Path(target) == destination
        assert destination.is_dir()
        events.append("rename")

    monkeypatch.setattr(fileio.os, "mkdir", recording_mkdir)
    monkeypatch.setattr(fileio.os, "rename", emulate_posix_rename)
    monkeypatch.setattr(fileio, "_fsync_directory", lambda _path: None)

    fileio.publish_dir_create_once(staged, destination)

    assert events == ["claim", "rename"]
