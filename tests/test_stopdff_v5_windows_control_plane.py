"""Cross-platform regressions for StopDFF host control-plane mechanics."""
from __future__ import annotations

import os
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import run_stopdff_v5_local as local_runner
from scripts import stopdff_v5_control_plane as control_plane
from scripts.stopdff_v5 import locking, sweep


def _close_all(held_fds: dict[str, int]) -> None:
    for fd in held_fds.values():
        os.close(fd)
    held_fds.clear()


def test_process_lock_reenters_one_owner_and_rejects_a_second(tmp_path):
    lock_path = tmp_path / "driver.lock"
    owner: dict[str, int] = {}
    contender: dict[str, int] = {}
    try:
        locking.acquire_process_lock(
            lock_path,
            owner,
            busy_label="owner collision",
        )
        held_fd = owner[os.path.realpath(lock_path)]

        locking.acquire_process_lock(
            lock_path,
            owner,
            busy_label="owner collision",
        )
        assert owner == {os.path.realpath(lock_path): held_fd}

        with pytest.raises(RuntimeError, match="owner collision"):
            locking.acquire_process_lock(
                lock_path,
                contender,
                busy_label="owner collision",
            )
        assert contender == {}
    finally:
        _close_all(owner)
        _close_all(contender)


def test_process_lock_rejects_another_thread_sharing_owner_map(tmp_path):
    lock_path = tmp_path / "shared-driver.lock"
    owner: dict[str, int] = {}
    failures: list[BaseException] = []
    try:
        locking.acquire_process_lock(
            lock_path,
            owner,
            busy_label="thread owner collision",
        )

        def contend() -> None:
            try:
                locking.acquire_process_lock(
                    lock_path,
                    owner,
                    busy_label="thread owner collision",
                )
            except BaseException as exc:
                failures.append(exc)

        contender = threading.Thread(target=contend)
        contender.start()
        contender.join()

        assert len(failures) == 1
        assert isinstance(failures[0], RuntimeError)
        assert "thread owner collision" in str(failures[0])
        assert list(owner) == [os.path.realpath(lock_path)]
    finally:
        _close_all(owner)


@pytest.mark.parametrize(
    ("lock_path", "acquire", "map_name", "error"),
    [
        (
            lambda root: root / "local_lifecycle.json.lock",
            lambda root: local_runner._acquire_lifecycle_lock(root),
            "_LIFECYCLE_LOCK_FDS",
            "another local StopDFF driver holds",
        ),
        (
            lambda root: root / "control.json.lock",
            lambda root: control_plane._acquire_control_plane_lock(
                root / "control.json"
            ),
            "_CONTROL_LOCK_FDS",
            "another control-plane driver holds",
        ),
    ],
)
def test_driver_lock_entrypoints_fail_fast_for_a_foreign_owner(
    tmp_path,
    monkeypatch,
    lock_path,
    acquire,
    map_name,
    error,
):
    foreign: dict[str, int] = {}
    module = local_runner if map_name == "_LIFECYCLE_LOCK_FDS" else control_plane
    monkeypatch.setattr(module, map_name, {})
    path = lock_path(tmp_path)
    try:
        locking.acquire_process_lock(
            path,
            foreign,
            busy_label="foreign owner",
        )
        with pytest.raises(RuntimeError, match=error):
            acquire(tmp_path)
        assert getattr(module, map_name) == {}
    finally:
        _close_all(foreign)


def test_local_staged_tree_sync_succeeds_on_this_host(tmp_path):
    staged = tmp_path / "staged"
    nested = staged / "nested"
    nested.mkdir(parents=True)
    (staged / "root.bin").write_bytes(b"root")
    (nested / "child.bin").write_bytes(b"child")

    local_runner._fsync_staged_tree(staged)


def test_fresh_sweep_initialization_uses_platform_directory_sync(
    tmp_path, monkeypatch
):
    destination = tmp_path / "run"
    synced: list[Path] = []

    monkeypatch.setattr(sweep, "_run_identity_files", lambda _ctx: ())
    monkeypatch.setattr(
        sweep,
        "_append_attempt",
        lambda path, _attempt: Path(path).write_bytes(b"attempt\n"),
    )
    monkeypatch.setattr(
        sweep,
        "fsync_directory",
        lambda path: synced.append(Path(path)),
    )
    monkeypatch.setattr(
        sweep,
        "publish_dir_create_once",
        lambda staged, dest, **_kwargs: os.rename(staged, dest),
    )

    sweep._publish_fresh_initialization(
        SimpleNamespace(output_dir=destination),
        started_attempt={},
    )

    assert (destination / "attempts.jsonl").read_bytes() == b"attempt\n"
    assert len(synced) == 1
    assert synced[0].name.startswith(".stopdff_run_initializing_")
