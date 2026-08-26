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
import shutil
import sys
import tempfile
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
_WINDOWS_RESERVED_BASENAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{number}" for number in range(1, 10)),
    *(f"LPT{number}" for number in range(1, 10)),
}


class ReleaseVerificationFailed(schema.ColmAimsError):
    """A staged release candidate did not reach ``PASS_RELEASE``."""


class _GuardRetirementCommittedError(OSError):
    """Guard unlink committed after acceptance, but its sync retries failed."""


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
    captured: tuple[tuple[str, tuple[int, int, int]], ...],
    label: str,
) -> None:
    """Reject an alias or identity swap of a previously trusted directory."""
    try:
        observed = schema.stable_directory_chain(path, path)
    except schema.ColmAimsError as exc:
        raise schema.TypedIngressError(
            f"{label} changed or became aliased during the operation"
        ) from exc
    if observed != captured:
        raise schema.TypedIngressError(
            f"{label} identity changed during the operation"
        )


def _require_publication_parent(
    destination: Path,
    parent_chain: tuple[tuple[str, tuple[int, int, int]], ...],
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
    parent_chain: tuple[tuple[str, tuple[int, int, int]], ...],
    exists_label: str,
) -> None:
    """Durably create one sibling without ever creating its parent."""
    _require_publication_parent(destination, parent_chain, f"{exists_label} open")
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        fd = os.open(path, flags, 0o600)
    except FileExistsError as exc:
        raise FileExistsError(f"{exists_label} already exists: {path}") from exc
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        _require_publication_parent(
            destination, parent_chain, f"{exists_label} file sync"
        )
        fileio.fsync_directory(path.parent)
        _require_publication_parent(
            destination, parent_chain, f"{exists_label} parent sync"
        )
    except BaseException:
        # A partial sibling remains fail-closed: pending is rejection, while a
        # marker cannot be accepted unless its exact bytes validate.
        raise


def _require_published_tree(
    destination: Path,
    expected_tree_sha256: str,
    parent_chain: tuple[tuple[str, tuple[int, int, int]], ...],
    stage: str,
) -> None:
    """Rebind the committed directory bytes to the staged precommit digest."""
    _require_publication_parent(destination, parent_chain, f"{stage} precheck")
    observed = _directory_tree_sha256(destination)
    _require_publication_parent(destination, parent_chain, f"{stage} postcheck")
    if observed != expected_tree_sha256:
        raise schema.TypedIngressError(
            "published directory tree differs from the verified staged tree"
        )


def _read_accepted_directory_snapshot(
    path: Path, label: str
) -> tuple[Path, dict[str, bytes]]:
    """Return the exact snapshot bound by a stable positive marker.

    The first snapshot is the sole authoritative byte capture returned to the
    consumer. A second capture is only a concurrency postcheck; its bytes are
    never substituted into the return value.
    """
    directory = Path(os.path.abspath(path))
    guard = _pending_guard_path(directory)
    if os.path.lexists(guard):
        raise schema.TypedIngressError(
            f"{label} has an unresolved pending publication guard"
        )
    directory = _canonical_existing_directory(directory, label)
    snapshot = verifier._read_tree_snapshot(directory)
    marker = _accepted_marker_path(directory)
    tree_sha256 = verifier._tree_digest_from_shas(
        {
            rel: hashlib.sha256(data).hexdigest()
            for rel, data in snapshot.items()
        }
    )
    expected = _accepted_marker_bytes(directory, tree_sha256)
    try:
        observed = schema.read_regular_file_bytes(
            marker,
            tree_root=marker.parent,
            max_bytes=len(expected),
        )
    except (OSError, schema.ColmAimsError) as exc:
        raise schema.TypedIngressError(
            f"{label} has no valid positive acceptance marker"
        ) from exc
    if observed != expected:
        raise schema.TypedIngressError(
            f"{label} positive acceptance marker does not bind its exact tree"
        )
    if os.path.lexists(guard):
        raise schema.TypedIngressError(
            f"{label} became pending during accepted snapshot capture"
        )
    if verifier._read_tree_snapshot(directory) != snapshot:
        raise schema.TypedIngressError(
            f"{label} tree changed during accepted snapshot capture"
        )
    return directory, snapshot


def _require_accepted_directory(path: Path, label: str) -> Path:
    """Require exact positive acceptance and no sibling pending-state guard."""
    directory, _ = _read_accepted_directory_snapshot(path, label)
    return directory


def _create_pending_guard(
    destination: Path,
    *,
    parent_chain: tuple[tuple[str, tuple[int, int, int]], ...],
) -> tuple[Path, bytes]:
    _require_publication_parent(destination, parent_chain, "guard precheck")
    guard = _pending_guard_path(destination)
    _require_unclaimed(guard, "pending publication guard")
    _require_unclaimed(
        _accepted_marker_path(destination), "positive acceptance marker"
    )
    encoded = _pending_guard_bytes(destination)
    _require_publication_parent(destination, parent_chain, "guard creation")
    _create_once_in_bound_parent(
        guard,
        encoded,
        destination=destination,
        parent_chain=parent_chain,
        exists_label="pending publication guard",
    )
    _require_publication_parent(destination, parent_chain, "guard readback")
    observed = schema.read_regular_file_bytes(
        guard,
        tree_root=guard.parent,
        max_bytes=len(encoded),
    )
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
    parent_chain: tuple[tuple[str, tuple[int, int, int]], ...],
) -> None:
    _require_publication_parent(destination, parent_chain, "marker creation")
    marker = _accepted_marker_path(destination)
    encoded = _accepted_marker_bytes(destination, tree_sha256)
    _create_once_in_bound_parent(
        marker,
        encoded,
        destination=destination,
        parent_chain=parent_chain,
        exists_label="positive acceptance marker",
    )
    _require_publication_parent(destination, parent_chain, "marker readback")
    observed = schema.read_regular_file_bytes(
        marker,
        tree_root=marker.parent,
        max_bytes=len(encoded),
    )
    if observed != encoded:
        raise schema.TypedIngressError(
            "positive acceptance marker differs from its deterministic bytes"
        )
    _require_publication_parent(destination, parent_chain, "marker completion")


def _require_positive_acceptance_state(
    destination: Path,
    tree_sha256: str,
    parent_chain: tuple[tuple[str, tuple[int, int, int]], ...],
) -> None:
    """Validate committed tree and marker while the pending guard is live."""
    _require_published_tree(
        destination,
        tree_sha256,
        parent_chain,
        "acceptance tree validation",
    )
    marker = _accepted_marker_path(destination)
    expected = _accepted_marker_bytes(destination, tree_sha256)
    observed = schema.read_regular_file_bytes(
        marker,
        tree_root=marker.parent,
        max_bytes=len(expected),
    )
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
    parent_chain: tuple[tuple[str, tuple[int, int, int]], ...],
) -> None:
    """Remove the guard only after the final directory is independently durable."""
    _require_publication_parent(destination, parent_chain, "guard retirement")
    observed = schema.read_regular_file_bytes(
        guard,
        tree_root=guard.parent,
        max_bytes=len(encoded),
    )
    if observed != encoded:
        raise schema.TypedIngressError(
            "pending publication guard changed before acceptance"
        )
    _require_publication_parent(destination, parent_chain, "guard unlink")
    os.unlink(guard)
    _require_publication_parent(destination, parent_chain, "guard unlink commit")
    try:
        fileio.fsync_directory(guard.parent)
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
            fileio.fsync_directory(guard.parent)
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
    observed = {entry.name for entry in directory.iterdir()}
    if observed != set(_SIDECAR_NAMES):
        raise schema.TypedIngressError(
            "release bundle must contain exactly ledger.json, rights.json, and"
            f" expectations.json; observed {sorted(observed)}"
        )
    return {
        name: schema.read_regular_file_bytes(
            directory / name, tree_root=directory
        )
        for name in _SIDECAR_NAMES
    }


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
    parent_chain: tuple[tuple[str, tuple[int, int, int]], ...],
) -> None:
    """Create-once publish guarded until the final directory is durable."""
    if Path(os.path.abspath(staged)).parent != Path(
        os.path.abspath(destination)
    ).parent:
        raise schema.ConfigSurfaceError(
            "staged and destination directories must be siblings under the"
            " validated publication parent"
        )
    _require_publication_parent(destination, parent_chain, "transaction start")
    tree_sha256 = _directory_tree_sha256(staged)
    _require_publication_parent(destination, parent_chain, "guard dispatch")
    guard, guard_bytes = _create_pending_guard(
        destination, parent_chain=parent_chain
    )
    published = False
    try:
        try:
            _require_publication_parent(destination, parent_chain, "directory rename")
            fileio.publish_dir_create_once(
                staged,
                destination,
                exists_label=exists_label,
            )
            published = True
            _require_publication_parent(
                destination, parent_chain, "directory rename completion"
            )
        except fileio.DirectoryPublicationCommittedError as exc:
            if Path(exc.destination).absolute() != Path(destination).absolute():
                raise
            published = True
            _require_publication_parent(
                destination, parent_chain, "committed rename recovery"
            )
            # The atomic rename committed, but acceptance stays guarded until
            # an independent full-tree + parent durability barrier succeeds.
            fileio.fsync_tree(destination)
            _require_publication_parent(
                destination, parent_chain, "committed tree sync"
            )
            fileio.fsync_directory(destination.parent)
            _require_publication_parent(
                destination, parent_chain, "committed parent sync"
            )
        except FileExistsError as exc:
            raise schema.ConfigSurfaceError(str(exc)) from exc
    except BaseException:
        if not published:
            # No final directory was committed. Best-effort removal avoids a
            # stale guard; a cleanup failure safely leaves the slot guarded.
            with contextlib.suppress(BaseException):
                _retire_pending_guard(
                    guard,
                    guard_bytes,
                    destination=destination,
                    parent_chain=parent_chain,
                )
        raise
    _require_published_tree(
        destination,
        tree_sha256,
        parent_chain,
        "postcommit tree validation",
    )
    try:
        _create_accepted_marker(
            destination,
            tree_sha256,
            parent_chain=parent_chain,
        )
    except BaseException:
        # The final tree is durable, but without a durable exact marker it is
        # not accepted. Keep the pending guard so every protocol-aware reader
        # mechanically rejects this ambiguous state.
        raise
    _require_positive_acceptance_state(
        destination,
        tree_sha256,
        parent_chain,
    )
    try:
        _retire_pending_guard(
            guard,
            guard_bytes,
            destination=destination,
            parent_chain=parent_chain,
        )
    except _GuardRetirementCommittedError:
        # Destination and positive marker were each durably published before
        # guard retirement. The live state (exact marker, no guard) is accepted;
        # a crash may conservatively restore the guard, yielding safe rejection.
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
    output_root_chain = schema.stable_directory_chain(output_root, output_root)
    receipts_chain = schema.stable_directory_chain(receipts_dir, receipts_dir)
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
    staged = Path(tempfile.mkdtemp(prefix=".release-staged-", dir=output_root))
    try:
        for name in _SIDECAR_NAMES:
            (staged / name).write_bytes(supplied[name])
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
        fileio.fsync_tree(staged)
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
        if staged is not None and staged.exists():
            with contextlib.suppress(BaseException):
                shutil.rmtree(staged)


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
