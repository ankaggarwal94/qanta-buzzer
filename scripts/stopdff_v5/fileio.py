"""Canonical durable-write primitives for StopDFF v5 artifacts.

Single package-wide implementation of the atomic publish discipline: write to a
same-directory temp file, flush + fsync the file, ``os.replace`` onto the
destination, then fsync the directory so the published name survives a crash
where the host exposes directory descriptors. On Windows, where Python rejects
directory ``os.open``, regular-file publishers re-sync the live file instead.
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


_IS_WINDOWS = os.name == "nt"


def _fsync_directory(
    directory: Path,
    *,
    published_file: Path | None = None,
) -> None:
    """Sync a published directory entry where the host supports doing so.

    POSIX permits opening a directory read-only and passing that descriptor to
    ``fsync``. Python's Windows ``os.open`` rejects directory paths with
    ``PermissionError`` even when the caller has access to the directory. In
    that one platform-specific case, re-open and sync a published regular file
    when one is available; this preserves a post-publication flush without
    pretending that Windows exposed a directory descriptor. Directory
    publications have no regular-file fallback and therefore return after the
    unsupported operation.

    Every other open or sync error propagates. In particular, POSIX permission
    failures, non-permission Windows failures, and regular-file sync failures
    remain fail-closed.
    """
    directory = Path(directory)
    try:
        directory_fd = os.open(directory, os.O_RDONLY)
    except PermissionError:
        if not _IS_WINDOWS or not directory.is_dir():
            raise
        if published_file is None:
            return
        # Windows requires a writable CRT descriptor for ``os.fsync`` even
        # though no bytes are changed by the flush.
        published_fd = os.open(Path(published_file), os.O_RDWR)
        try:
            os.fsync(published_fd)
        finally:
            os.close(published_fd)
        return
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def fsync_directory(directory: Path) -> None:
    """Apply the host's strongest supported sync to an existing directory.

    This is the public entry point for staged-tree callers.  Unsupported
    Windows directory descriptors use the narrowly-scoped platform fallback
    in ``_fsync_directory``; all other open and sync failures still propagate.
    """
    _fsync_directory(Path(directory))


def publish_bytes(path: Path, data: bytes) -> None:
    """Atomically replace a file and apply the host's strongest publish sync."""
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
        _fsync_directory(path.parent, published_file=path)
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
    file fsync before publication, host publish sync after) but with create-once
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
        _fsync_directory(path.parent, published_file=path)
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
    form). Both platform paths fail closed on ANY pre-existing ``dest`` — empty
    directory, non-empty directory, file, or symlink alike:

    * POSIX ``rename`` may replace an existing empty directory. Preserve the
      original two-step discipline there: atomically claim ``dest`` with
      ``os.mkdir``, then rename the fully-materialized ``staged`` directory
      into that exclusively-owned empty slot.
    * Windows ``rename`` already refuses every existing destination, while it
      cannot rename onto the empty directory created by the POSIX claim. Use
      its direct no-replace rename as the single atomic create-once operation.

    After publication, sync the parent directory where the host exposes a
    directory descriptor. Windows does not expose one through ``os.open``;
    the direct same-volume rename is its best available atomic publication.
    The bytes inside ``staged`` are unchanged on both paths.

    ``staged`` is owned by the caller: on ``FileExistsError`` (the slot was
    already claimed on POSIX or the Windows rename lost the destination race)
    ``staged`` is left untouched for the caller's existing cleanup path, and
    no exception is masked.

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
    if _IS_WINDOWS:
        try:
            os.rename(staged, dest)
        except FileExistsError as exc:
            raise FileExistsError(
                f"{exists_label} already exists: {dest}"
            ) from exc
    else:
        try:
            os.mkdir(dest)
        except FileExistsError as exc:
            raise FileExistsError(
                f"{exists_label} already exists: {dest}"
            ) from exc
        os.rename(staged, dest)
    _fsync_directory(dest.parent)


def reclaim_empty_relic(dest: Path) -> bool:
    """Reclaim an empty create-once directory relic left by a crashed publish.

    Best-effort *recovery* companion to ``publish_dir_create_once``. Its POSIX
    path claims ``dest`` with ``os.mkdir`` and then fills it with ``os.rename``;
    a crash in the window between those two steps leaves an EMPTY ``dest``
    directory that thereafter fails closed on every retry. This helper removes
    ONLY that empty relic so a deliberate recovery/resume can re-claim the
    slot. The Windows path is one direct no-replace rename and creates no such
    claim relic, though recovery may still encounter one from an older writer.

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
