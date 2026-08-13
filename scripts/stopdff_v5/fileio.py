"""Canonical durable-write primitives for StopDFF v5 artifacts.

Single package-wide implementation of the atomic publish discipline: write to a
same-directory temp file, flush + fsync the file, ``os.replace`` onto the
destination, then fsync the directory so the published name survives a crash.
Callers own their byte encodings (adapter rows, control records, manifests);
this module owns only the write mechanics, so durability fixes land in exactly
one place. Artifact bytes are hash-attested downstream — routing a writer
through here must never change the bytes it publishes.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def publish_bytes(path: Path, data: bytes) -> None:
    """Atomically replace a regular file and durably publish its directory entry."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if (path.exists() or path.is_symlink()) and (
        path.is_symlink() or not path.is_file()
    ):
        raise ValueError(f"atomic-write destination is noncanonical: {path}")
    fd, tmp = tempfile.mkstemp(dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def dumps_json_bytes(obj: Any) -> bytes:
    """Encode ``obj`` with the package's artifact JSON convention.

    Sorted keys, two-space indent, trailing newline — byte-identical to what
    every v5 JSON artifact writer has always published (these bytes are
    hash-attested, so the encoding must not drift).
    """
    return (json.dumps(obj, indent=2, sort_keys=True) + "\n").encode("utf-8")
