"""Shared helper-hash provenance for the DP StopDFF pipeline.

The DP producer (``scripts/compute_stopdff_dp.py``) and the sweep
(``scripts/sweep_stopdff_dp.py``) both depend on every ``.py`` module
under ``scripts/stopdff_dp/`` plus the shared ``scripts/_audit_gates.py``
and ``scripts/_common.py``. Their cache fingerprint (sweep) and artifact
provenance (producer) both need a single source of truth for those
hashes so the audit-card consumer can cross-check them.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _file_sha256(path: Path) -> str | None:
    """Return the SHA-256 digest of a local file, or None if missing."""
    if path is None or not path.exists() or not path.is_file():
        return None
    from scripts._common import sha256_file

    return sha256_file(path)


def helper_paths() -> list[Path]:
    """Return the canonical list of helper module paths the DP pipeline imports.

    Order: ``scripts/stopdff_dp/*.py`` (sorted), then the shared
    ``scripts/_audit_gates.py`` and ``scripts/_common.py``.
    Used by:
      - ``helper_sha256s`` (for the cache fingerprint + producer provenance)
      - ``scripts/sweep_stopdff_dp.py:_git_metadata`` (for the dirty-status
        pathspec — keeps the dirty flag consistent with what the fingerprint
        actually hashes)
    """
    helper_dir = PROJECT_ROOT / "scripts" / "stopdff_dp"
    paths: list[Path] = sorted(helper_dir.glob("*.py"))
    for shared in ("_audit_gates.py", "_common.py"):
        candidate = PROJECT_ROOT / "scripts" / shared
        if candidate.exists():
            paths.append(candidate)
    return paths


def helper_sha256s() -> dict[str, str]:
    """Hash every .py file the DP pipeline imports beyond its producer scripts.

    See ``helper_paths`` for the file set. Returned as a dict keyed by
    repo-relative POSIX path so it serializes deterministically into the
    fingerprint and folds into cell_id via json.dumps(..., sort_keys=True).
    The same dict is embedded in the DP producer's generation block so
    the audit card consumer can cross-check it.
    """
    out: dict[str, str] = {}
    for path in helper_paths():
        digest = _file_sha256(path)
        if digest is None:
            continue
        try:
            rel = path.resolve().relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            rel = str(path)
        out[rel] = digest
    return out
