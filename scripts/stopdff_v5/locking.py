"""Cross-platform advisory locks for StopDFF host-side drivers.

The control-plane journals are replace-written files, so each driver must have
exactly one process owner for a given lock path.  POSIX keeps the historical
``flock(LOCK_EX | LOCK_NB)`` behavior.  Windows uses a non-blocking one-byte
``msvcrt.locking`` region held by the same open descriptor for the process
lifetime.
"""
from __future__ import annotations

import os
from collections.abc import MutableMapping
from pathlib import Path

if os.name == "nt":  # pragma: no cover - selected by the host at import time
    import msvcrt
else:  # pragma: no cover - selected by the host at import time
    import fcntl


def _lock_descriptor(fd: int) -> None:
    """Take one non-blocking exclusive lock on ``fd``."""
    if os.name == "nt":
        # msvcrt.locking locks bytes from the descriptor's current position.
        # Give every lock file one stable byte and always lock byte zero.
        if os.fstat(fd).st_size == 0:
            os.lseek(fd, 0, os.SEEK_SET)
            os.write(fd, b"\0")
            os.fsync(fd)
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        return
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)


def acquire_process_lock(
    lock_path: Path,
    held_fds: MutableMapping[str, int],
    *,
    busy_label: str,
) -> None:
    """Acquire or re-enter a process-lifetime advisory lock.

    Parameters
    ----------
    lock_path
        File whose advisory lock identifies the protected journal.
    held_fds
        Caller-owned process-lifetime descriptor map.  A realpath already in
        this map is a same-process re-entry and reuses the existing lock.
    busy_label
        Prefix for the fail-fast ``RuntimeError`` raised when another owner
        holds the lock.
    """
    lock_path = Path(lock_path)
    key = os.path.realpath(lock_path)
    if key in held_fds:
        return
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_BINARY", 0)
    fd = os.open(lock_path, flags, 0o644)
    try:
        _lock_descriptor(fd)
    except OSError as exc:
        os.close(fd)
        raise RuntimeError(f"{busy_label} {lock_path}") from exc
    held_fds[key] = fd
