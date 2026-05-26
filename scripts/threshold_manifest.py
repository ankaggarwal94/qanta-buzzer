"""Frozen threshold manifest loader with sha256 sidecar integrity check.

Implements the load-time integrity contract for ``threshold_manifest.json``.
Closes the DATA-03 gap surfaced by Phase 02 review CR-01: the manifest
sha256 hash is documented in ``PROJECT_WIKI/SPLIT_PROVENANCE.md`` and
copied to a sidecar file, but until this module existed no consumer
verified that the on-disk manifest still hashed to the recorded value
at load time. Without this check, a future edit (or filesystem
corruption) could silently change threshold values used by
``compute_csli.py``, ``compute_stopdff.py``, ``compute_prefix_calibration.py``,
and ``make_audit_card.py`` while the audit card kept citing the frozen
sha256 string -- documentation, not enforcement.

The canonical sidecar filename is ``<manifest>.sha256`` (e.g.,
``threshold_manifest.json.sha256``), matching the output of
``sha256sum threshold_manifest.json``. The legacy
``threshold_manifest.sha256`` (no ``.json`` segment) was deleted in
the same commit as this module to remove the dual-canonical
ambiguity flagged by CR-01.

Importing scripts should use ``strict=True`` -- the only callers that
should accept ``strict=False`` are introspection/diagnostics tools
explicitly designed to tolerate a missing manifest.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any


def _expected_sidecar(manifest_path: Path) -> Path:
    """Return the canonical sidecar path for a manifest.

    Uses ``<manifest>.sha256`` (e.g., ``threshold_manifest.json.sha256``)
    so the filename matches the default output of
    ``sha256sum threshold_manifest.json``. Constructed via
    ``parent / (name + ".sha256")`` rather than ``with_suffix`` because
    ``with_suffix`` only replaces the LAST extension segment, which
    would map ``threshold_manifest.json`` to ``threshold_manifest.sha256``
    -- the legacy ambiguous filename CR-01 deleted.
    """
    return manifest_path.parent / (manifest_path.name + ".sha256")


def _read_sidecar_hash(sidecar_path: Path) -> str:
    """Extract the hex hash from a ``sha256sum``-format sidecar line.

    Accepts both ``<hash>  <filename>`` (two-space separator, the
    BSD/GNU default) and bare ``<hash>`` files. Whitespace-tolerant.
    """
    raw = sidecar_path.read_text(encoding="utf-8").strip()
    if not raw:
        raise RuntimeError(
            f"Threshold manifest sidecar is empty: {sidecar_path}"
        )
    return raw.split()[0].strip()


def load_frozen_threshold_manifest(
    manifest_path: Path,
    sidecar_path: Path | None = None,
    *,
    strict: bool = True,
) -> dict[str, Any]:
    """Load ``threshold_manifest.json`` after verifying its sha256 sidecar.

    Parameters
    ----------
    manifest_path : Path
        Path to ``threshold_manifest.json`` (or compatible JSON).
    sidecar_path : Path or None
        Path to the sha256 sidecar. Defaults to
        ``<manifest_path>.sha256``.
    strict : bool, keyword-only
        When True (default), a missing manifest, missing sidecar, or
        hash mismatch raises ``RuntimeError``. Consumers in this repo
        MUST use ``strict=True`` so DATA-03 ("thresholds frozen before
        test inspection") becomes an enforced check rather than a
        documentation claim.

    Returns
    -------
    dict
        Parsed manifest JSON.

    Raises
    ------
    RuntimeError
        If ``strict=True`` and any of: manifest is missing, sidecar is
        missing, sidecar is empty, on-disk hash does not match sidecar.
        Hash mismatch is ALWAYS fatal regardless of ``strict`` -- if the
        sidecar exists but disagrees with the manifest, the integrity
        contract is already violated and the caller MUST stop.
    """
    if sidecar_path is None:
        sidecar_path = _expected_sidecar(manifest_path)

    if not manifest_path.exists():
        msg = f"Threshold manifest missing: {manifest_path}"
        if strict:
            raise RuntimeError(msg)
        print(
            f"WARNING: {msg}; falling back to hardcoded defaults",
            file=sys.stderr,
        )
        return {}

    if not sidecar_path.exists():
        msg = f"Threshold manifest sidecar missing: {sidecar_path}"
        if strict:
            raise RuntimeError(msg)
        print(
            f"WARNING: {msg}; skipping integrity check",
            file=sys.stderr,
        )
    else:
        # SECURITY-REVIEW: sha256 of a small JSON file; this is an
        # integrity check (not a secret), so SHA-256 is appropriate
        # per the enterprise prodsec baseline (no MD5/SHA-1).
        actual = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        expected = _read_sidecar_hash(sidecar_path)
        if actual != expected:
            raise RuntimeError(
                "Threshold manifest sha256 mismatch:\n"
                f"  expected (from {sidecar_path.name}): {expected}\n"
                f"  actual   ({manifest_path.name}):     {actual}\n"
                "DATA-03 integrity contract violated. Refusing to load.\n"
                "If the manifest was intentionally updated, regenerate "
                "the sidecar with `shasum -a 256 threshold_manifest.json "
                f"> {sidecar_path.name}` AND update "
                "PROJECT_WIKI/SPLIT_PROVENANCE.md "
                "THRESHOLD_MANIFEST_SHA256."
            )

    return json.loads(manifest_path.read_text(encoding="utf-8"))


def threshold_value(
    manifest: dict[str, Any],
    metric: str,
    *,
    key: str = "threshold",
) -> Any:
    """Look up a single threshold entry by metric name.

    Raises ``RuntimeError`` when the metric is absent from the manifest's
    ``thresholds`` list. This is intentional: silently falling back to a
    hardcoded default was the WR-02 weakness this helper closes.

    Parameters
    ----------
    manifest : dict
        Parsed manifest payload (as returned by
        ``load_frozen_threshold_manifest``).
    metric : str
        Value of the ``metric`` field to look up.
    key : str, keyword-only
        Which numeric field to return. ``"threshold"`` for
        ``prefix_ece``/``stopdff_median_abs_prefix``; pass
        ``"numeric_value_K4"`` for ``choices_only_accuracy``.

    Returns
    -------
    Any
        The looked-up threshold value (typically int or float).
    """
    for entry in manifest.get("thresholds", []):
        if entry.get("metric") == metric:
            if key not in entry:
                raise RuntimeError(
                    f"DATA-03 violation: threshold entry for metric "
                    f"'{metric}' is missing field '{key}'. "
                    "Refusing to fabricate a default."
                )
            return entry[key]
    raise RuntimeError(
        f"DATA-03 violation: threshold_manifest.json does not declare "
        f"metric '{metric}'. Refusing to fabricate a default."
    )
