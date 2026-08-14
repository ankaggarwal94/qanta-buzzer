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


def create_once_bytes(
    path: Path,
    data: bytes,
    *,
    exists_label: str = "create-once artifact",
) -> None:
    """Durably publish a new regular file, failing closed if ``path`` exists.

    Same fsync discipline as ``publish_bytes`` (same-directory temp, flush +
    file fsync before publication, directory fsync after) but with create-once
    semantics: the temp file is ``os.link``-ed onto the destination, so an
    existing ``path`` raises ``FileExistsError`` instead of being replaced.

    Parameters
    ----------
    path
        Destination that must not already exist.
    data
        Bytes to publish.
    exists_label
        Artifact description used in the ``FileExistsError`` message, so
        callers keep their historical error wording.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise FileExistsError(
                f"{exists_label} already exists: {path}"
            ) from exc
        os.unlink(temporary)
        temporary = ""
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def publish_dir_create_once(
    staged: Path,
    dest: Path,
    *,
    exists_label: str = "create-once directory",
) -> None:
    """Atomically publish a staged directory into a create-once slot.

    Directory analogue of ``create_once_bytes`` (``os.link`` has no directory
    form): publication happens in two steps that fail closed on ANY
    pre-existing ``dest`` — empty directory, non-empty directory, file, or
    symlink alike.

    1. Claim ``dest`` with ``os.mkdir``. ``mkdir`` is atomic and raises
       ``FileExistsError`` whenever the name already exists, so exactly one
       concurrent caller can create it. A pre-existing *empty* directory
       therefore fails closed here instead of being silently replaced — the
       hole a bare ``os.rename`` leaves, since ``rename`` onto an empty
       directory replaces it.
    2. Fill the freshly-claimed empty slot with ``os.rename(staged, dest)``.
       The claimant is the sole actor past the ``mkdir`` gate, so ``dest`` is
       still the empty directory it just created and the rename installs
       ``staged``'s complete contents under the live name in one step; no peer
       can rename into ``dest`` because no peer won the claim.

    The published directory entry is made durable by fsync-ing ``dest``'s
    parent after the rename, so the live name survives a crash. This changes
    only publish mechanics: the bytes inside ``staged`` are published
    unchanged.

    ``staged`` is owned by the caller: on ``FileExistsError`` (the slot was
    already claimed) ``staged`` is left untouched for the caller's existing
    cleanup path, and no exception is masked.

    Parameters
    ----------
    staged
        Fully-materialized source directory to publish. It should be a sibling
        of ``dest`` (same parent, hence same filesystem) so the rename is an
        atomic same-filesystem move.
    dest
        Create-once destination slot that must not already exist.
    exists_label
        Description used in the ``FileExistsError`` message so callers keep
        their historical error wording.
    """
    staged = Path(staged)
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.mkdir(dest)
    except FileExistsError as exc:
        raise FileExistsError(f"{exists_label} already exists: {dest}") from exc
    os.rename(staged, dest)
    directory_fd = os.open(dest.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def dumps_json_bytes(obj: Any) -> bytes:
    """Encode ``obj`` with the package's artifact JSON convention.

    Sorted keys, two-space indent, trailing newline — byte-identical to what
    every v5 JSON artifact writer has always published (these bytes are
    hash-attested, so the encoding must not drift). ``allow_nan=False`` makes a
    non-finite float fail loudly at write time instead of emitting non-JSON
    bytes the strict readers reject anyway; valid artifacts are unaffected.
    """
    return (
        json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
