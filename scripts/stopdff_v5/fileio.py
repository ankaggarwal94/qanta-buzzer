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


def reclaim_empty_relic(dest: Path) -> bool:
    """Reclaim an empty create-once directory relic left by a crashed publish.

    Best-effort *recovery* companion to ``publish_dir_create_once``. That
    publisher claims ``dest`` with ``os.mkdir`` and then fills it with
    ``os.rename``; a crash in the window between those two steps leaves an
    EMPTY ``dest`` directory that thereafter fails closed on every retry
    (``os.mkdir`` cannot re-claim a name that already exists). This helper
    removes ONLY that empty relic so a deliberate recovery/resume can re-claim
    the slot.

    It can never destroy a real artifact: it refuses a symlink, a regular file,
    and any non-empty directory, and it reclaims solely via ``os.rmdir`` — which
    itself fails on a non-empty directory. It is therefore safe to call even
    when the "relic" turns out to be a live artifact; such a call is a no-op
    that returns ``False``.

    Single-owner safety does NOT come from this reclaim, and — importantly — is
    NOT fully guaranteed by the subsequent ``os.mkdir`` claim alone. This reclaim
    cannot distinguish an empty crash relic from an empty *in-flight* ``os.mkdir``
    claim made by a concurrent ``publish_dir_create_once``; a reclaimer running
    alongside a fresh publisher of the same slot could remove that live claim and
    let both callers pass the mkdir gate (only ``os.rename``'s ENOTEMPTY then
    salvages single *content*, not single ownership). The real guarantor is the
    CALLER's context: invoke this ONLY on a genuine recovery/resume path AND ONLY
    where the caller already excludes any concurrent publisher of the same slot
    (e.g. under the ``run_sweep`` ``max_containers=1`` singleton, or an equivalent
    lifecycle lock). Under that exclusion a stale relic has no concurrent
    claimant, so reclaim→mkdir→rename is race-free and concurrent *reclaimers*
    only churn (the loser gets ``False``). Never invoke on a fresh publish — a
    fresh collision with a peer's in-progress claim must fail closed.
    WARNING: if a caller's exclusion guarantee is ever weakened (e.g. ``run_sweep``
    stops being a singleton), this reclaim reopens a two-owner window and every
    call site must be re-audited.

    Parameters
    ----------
    dest
        Create-once destination that may hold an empty crash relic.

    Returns
    -------
    bool
        ``True`` only when an empty relic directory was actually removed;
        ``False`` when there is nothing to reclaim (``dest`` is absent), when
        ``dest`` is a symlink, a regular file, or a non-empty directory, or
        when the ``rmdir`` lost a race.
    """
    dest = Path(dest)
    # A symlink is never a create-once relic. Never follow it and never remove
    # it: leave it for the later os.mkdir claim to fail closed on. Checked first
    # so we never stat through the link via exists()/is_dir().
    if dest.is_symlink():
        return False
    if not dest.exists():
        return False
    if not dest.is_dir():
        # A regular file (or other non-directory) is a real artifact or peer
        # state, not an empty crash relic — refuse to touch it.
        return False
    try:
        os.rmdir(dest)
    except OSError:
        # Lost a benign race: either FileNotFoundError (a concurrent reclaimer
        # already removed it) or ENOTEMPTY (a peer filled it between our checks
        # and the rmdir). Either way it is no longer an empty relic we may
        # reclaim; leave it for the subsequent os.mkdir claim to adjudicate and
        # never raise from this best-effort recovery step.
        return False
    return True


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
