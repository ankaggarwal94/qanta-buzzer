"""Windows-specific directory-fsync compatibility for durable publishers."""
from __future__ import annotations

import errno
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.stopdff_v5 import fileio  # noqa: E402


class _StatWithFileAttributes:
    """Proxy a real stat result while injecting Windows file attributes."""

    def __init__(self, original, file_attributes):
        self._original = original
        self.st_file_attributes = file_attributes

    def __getattr__(self, name):
        return getattr(self._original, name)


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


def test_create_once_commit_callback_precedes_post_link_sync_failure(
    tmp_path, monkeypatch
):
    destination = tmp_path / "committed-before-sync.bin"
    committed = []

    def fail_directory_sync(_directory, published_file=None):
        assert published_file == destination
        raise OSError(errno.EIO, "synthetic post-link sync failure")

    monkeypatch.setattr(fileio, "_fsync_directory", fail_directory_sync)
    with pytest.raises(OSError, match="post-link sync failure"):
        fileio.create_once_bytes(
            destination,
            b"committed",
            commit_created=lambda: committed.append(True),
        )

    assert committed == [True]
    assert destination.read_bytes() == b"committed"


@pytest.mark.parametrize("reparse_at", ["root", "child", "file"])
def test_fsync_tree_rejects_windows_directory_reparse_points_before_descent(
    tmp_path, monkeypatch, reparse_at
):
    root = tmp_path / "staged"
    child = root / "junction"
    child.mkdir(parents=True)
    file_entry = root / "external-like.bin"
    file_entry.write_bytes(b"must not be traversed")
    target = {
        "root": root,
        "child": child,
        "file": file_entry,
    }[reparse_at]
    real_stat = os.stat
    nofollow_targets = []

    def mark_target_as_reparse(path, *args, **kwargs):
        observed = real_stat(path, *args, **kwargs)
        if Path(path) == target and kwargs.get("follow_symlinks") is False:
            nofollow_targets.append(Path(path))
            attributes = getattr(observed, "st_file_attributes", 0)
            return _StatWithFileAttributes(
                observed,
                attributes | stat.FILE_ATTRIBUTE_REPARSE_POINT,
            )
        return observed

    monkeypatch.setattr(fileio.os, "stat", mark_target_as_reparse)
    real_open = os.open

    def reject_reparse_file_open(path, flags, *args, **kwargs):
        if Path(path) == file_entry and reparse_at == "file":
            pytest.fail("reparse-point file was opened before rejection")
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(fileio.os, "open", reject_reparse_file_open)

    if reparse_at == "root":

        def reject_walk(*args, **kwargs):
            pytest.fail("root reparse point was walked before rejection")

        monkeypatch.setattr(fileio.os, "walk", reject_walk)
    else:

        def guarded_walk(path, *, topdown, onerror, followlinks):
            assert Path(path) == root
            assert topdown is True
            assert followlinks is False
            yield str(root), [child.name], [file_entry.name]
            pytest.fail("reparse-point child was descended into")

        monkeypatch.setattr(fileio.os, "walk", guarded_walk)

    with pytest.raises(ValueError, match="reparse|canonical"):
        fileio.fsync_tree(root)
    assert target in nofollow_targets


@pytest.mark.skipif(os.name != "nt", reason="requires a real NTFS junction")
@pytest.mark.parametrize("junction_at", ["root", "child"])
def test_fsync_tree_never_descends_into_real_windows_junction(
    tmp_path, monkeypatch, junction_at
):
    external = tmp_path / "external"
    external.mkdir()
    external_file = external / "must-not-open.bin"
    external_file.write_bytes(b"external")
    staged = tmp_path / "staged"
    staged.mkdir()
    junction = (
        tmp_path / "staged-junction"
        if junction_at == "root"
        else staged / "junction"
    )
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

    monkeypatch.setattr(fileio.os, "scandir", record_scandir)
    try:
        with pytest.raises(ValueError, match="reparse|canonical"):
            fileio.fsync_tree(junction if junction_at == "root" else staged)
        assert junction not in scanned
    finally:
        os.rmdir(junction)


def test_fsync_tree_preserves_bottom_up_directory_sync_order(tmp_path, monkeypatch):
    root = tmp_path / "staged"
    child = root / "child"
    child.mkdir(parents=True)
    (child / "payload.bin").write_bytes(b"payload")
    synced = []

    monkeypatch.setattr(fileio, "fsync_directory", lambda path: synced.append(path))

    fileio.fsync_tree(root)

    assert synced == [child, root]


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

    # The production Windows branch depends on ``os.rename`` refusing to
    # replace an existing destination (Win32 raises ``FileExistsError``). On
    # POSIX the real ``os.rename`` instead raises ``OSError(ENOTEMPTY)`` for a
    # non-empty directory, which is NOT a ``FileExistsError`` — so emulate the
    # Windows no-replace contract to exercise the branch faithfully on POSIX CI.
    real_rename = os.rename

    def emulate_windows_rename(source, target):
        assert Path(source) == staged
        assert Path(target) == destination
        if Path(target).exists():
            raise FileExistsError(
                "emulated Windows no-replace rename onto existing"
                f" destination: {target}"
            )
        real_rename(source, target)

    monkeypatch.setattr(fileio.os, "rename", emulate_windows_rename)

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
