"""Shared helper-hash provenance for the DP StopDFF pipeline.

The DP producer (``scripts/compute_stopdff_dp.py``) and the sweep
(``scripts/sweep_stopdff_dp.py``) both depend on every ``.py`` module
under ``scripts/stopdff_dp/`` plus the shared ``scripts/_audit_gates.py``
and ``scripts/_common.py``. Their cache fingerprint (sweep) and artifact
provenance (producer) both need a single source of truth for those
hashes so the audit-card consumer can cross-check them.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _file_sha256(path: Path) -> str | None:
    """Return the SHA-256 digest of a local file, or None if missing."""
    if path is None or not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def helper_sha256s() -> dict[str, str]:
    """Hash every .py file the DP pipeline imports beyond its producer scripts.

    Includes:
      - scripts/stopdff_dp/*.py (rewards, dp_solver, adapter, continuation,
        diagnostics, types, writers, __init__, this _provenance module)
      - scripts/_audit_gates.py (MC coverage + retention gate helpers)
      - scripts/_common.py (project_relative + serializer + provenance)

    Returned as a dict keyed by repo-relative POSIX path so it serializes
    deterministically into the fingerprint and folds into cell_id via
    json.dumps(..., sort_keys=True). The same dict is embedded in the
    DP producer's generation block so the audit card consumer can
    cross-check it.
    """
    helper_dir = PROJECT_ROOT / "scripts" / "stopdff_dp"
    paths: list[Path] = sorted(helper_dir.glob("*.py"))
    for shared in ("_audit_gates.py", "_common.py"):
        candidate = PROJECT_ROOT / "scripts" / shared
        if candidate.exists():
            paths.append(candidate)
    out: dict[str, str] = {}
    for path in paths:
        digest = _file_sha256(path)
        if digest is None:
            continue
        try:
            rel = path.resolve().relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            rel = str(path)
        out[rel] = digest
    return out
