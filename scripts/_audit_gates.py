"""Shared MC coverage and retention gate helpers for audit metrics.

PR #14 review (Blocker 3) flagged that ``compute_csli.py`` enforced
a unique-qid MC coverage gate and a raw-test MC retention gate
(``--allow-incomplete-mc-coverage`` / ``--allow-low-mc-retention``
overrides), but ``compute_prefix_calibration.py`` and
``compute_stopdff.py`` did not. Missing MC rows are not random
(items where good distractors could not be built — typically the
hardest or most ambiguous answer texts), so calibration and StopDFF
silently consumed a biased subset of the test split.

This module exposes the same coverage and retention primitives the
CSLI script already enforces, so all three audit metrics
(``compute_csli.py`` + ``compute_prefix_calibration.py`` +
``compute_stopdff.py``) and the aggregator (``make_audit_card.py``)
agree on what counts as a defensible retained-subset audit and how
the operator opts into it.

Design intent:

* No script-level state. Every function is pure (filesystem reads
  are confined to ``load_mc_build_metadata``, which fails closed on
  malformed input).
* Default thresholds match the CSLI script's defaults
  (``min_mc_coverage = 0.98``, retention default from
  ``build_metadata.retention_thresholds`` or 0.98) so the three
  audit metrics behave identically on the same data unless the
  operator passes an explicit override flag.
* No exception is raised on a gate violation -- the helpers return
  structured dicts and let each caller decide whether to ``return
  1`` from ``main()`` or to record the override decision in the
  artifact's metadata.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256_file(path: Path) -> str:
    """Return the SHA-256 digest for a local artifact.

    Duplicated from ``compute_csli._sha256_file`` to keep this
    module free of cross-script imports. The function is small
    enough that the duplication is cheaper than the extra coupling.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def filter_mc_questions_to_split(
    mc_questions: list[dict],
    target_qids: set[str],
) -> tuple[list[dict], dict[str, Any]]:
    """Filter MC rows to ``target_qids`` and report unique-qid coverage.

    Mirrors ``compute_csli._filter_test_mc_questions`` (which the
    PR-14 babysit fixed to use ``matched_test_mc_qids`` rather than
    raw row count, so a duplicate-qid MC row does not hide a missing
    qid).

    Parameters
    ----------
    mc_questions : list[dict]
        MC question rows (must carry a ``"qid"`` field).
    target_qids : set[str]
        Set of qids the consumer wants to evaluate against (usually
        the test split qids).

    Returns
    -------
    (list[dict], dict[str, Any])
        Filtered MC rows AND a coverage dict with the same shape
        the audit card expects:

        * ``target_qids`` -- count of qids in the target split
        * ``mc_questions_total`` -- count of all MC rows in the pool
        * ``matched_questions`` -- count of MC rows whose qid is in
          ``target_qids``
        * ``matched_qids`` -- count of *unique* qids that survived
          the filter
        * ``missing_qids`` -- count of qids in ``target_qids`` with
          no MC row at all
        * ``missing_qids_set`` -- the set of missing qids (kept
          separately so callers can print or persist them)
        * ``coverage_rate`` -- ``matched_qids / max(1, len(target_qids))``
    """
    questions = [q for q in mc_questions if str(q["qid"]) in target_qids]
    matched_qids = {str(q["qid"]) for q in questions}
    missing_qids = target_qids - matched_qids
    coverage = len(matched_qids) / max(1, len(target_qids))
    return questions, {
        "target_qids": len(target_qids),
        "mc_questions_total": len(mc_questions),
        "matched_questions": len(questions),
        "matched_qids": len(matched_qids),
        "missing_qids": len(missing_qids),
        "missing_qids_set": missing_qids,
        "coverage_rate": coverage,
    }


def load_mc_build_metadata(data_dir: Path) -> dict[str, Any]:
    """Load MC-construction retention metadata when present.

    Mirrors ``compute_csli._load_mc_build_metadata``. Returns a
    summary dict with ``status`` in ``{"missing", "loaded"}``; on
    ``"loaded"`` the dict includes per-split retention counts and
    any ``retention_thresholds`` block from the file.

    Raises
    ------
    RuntimeError
        If the file exists but is malformed. Failing closed here is
        preferred over silently degrading to ``status="missing"``
        because the operator is asking us to trust this artifact for
        gate enforcement.
    """
    path = data_dir / "build_metadata.json"
    summary: dict[str, Any] = {
        "status": "missing",
        "source_path": str(path),
        "source_sha256": None,
        "splits": None,
        "retention_thresholds": None,
    }
    if not path.exists():
        return summary

    try:
        with open(path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        splits = metadata["splits"]
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Invalid MC build metadata at {path}: {exc}") from exc

    summary["status"] = "loaded"
    summary["source_sha256"] = _sha256_file(path)
    summary["retention_thresholds"] = metadata.get("retention_thresholds")
    summary["splits"] = {
        split_name: {
            "raw_count": int(split_data["raw_count"]),
            "retained_count": int(split_data["retained_count"]),
            "dropped_count": int(split_data["dropped_count"]),
            "retention_rate": float(split_data["retention_rate"]),
            "drop_reasons": split_data.get("drop_reasons", {}),
        }
        for split_name, split_data in splits.items()
        if isinstance(split_data, dict)
    }
    return summary


def metadata_retention_threshold(
    build_metadata: dict[str, Any],
    *,
    smoke: bool,
    explicit_threshold: float | None,
) -> float:
    """Resolve the effective MC retention threshold.

    Precedence: explicit CLI override > ``retention_thresholds.smoke``
    / ``retention_thresholds.full`` from ``build_metadata.json`` >
    0.98 default. Matches ``compute_csli._metadata_retention_threshold``.
    """
    if explicit_threshold is not None:
        return explicit_threshold

    thresholds = build_metadata.get("retention_thresholds")
    if isinstance(thresholds, dict):
        key = "smoke" if smoke else "full"
        try:
            return float(thresholds[key])
        except (KeyError, TypeError, ValueError):
            pass

    return 0.98


def coverage_gate_decision(
    coverage_rate: float,
    *,
    threshold: float,
    override: bool,
    override_flag: str = "--allow-incomplete-mc-coverage",
) -> dict[str, Any]:
    """Return a structured coverage-gate decision dict.

    ``passed`` is the pure threshold check (independent of override).
    ``overridden`` is True only when the gate failed AND override was
    set. ``effective_pass`` is the operator-visible outcome the
    audit card surfaces.
    """
    passed = coverage_rate >= threshold
    overridden = (not passed) and override
    return {
        "coverage_rate": float(coverage_rate),
        "threshold": float(threshold),
        "passed": bool(passed),
        "overridden": bool(overridden),
        "effective_pass": bool(passed or overridden),
        "override_flag": override_flag,
    }


def retention_gate_decision(
    retention_rate: float,
    *,
    threshold: float,
    override: bool,
    override_flag: str = "--allow-low-mc-retention",
) -> dict[str, Any]:
    """Return a structured retention-gate decision dict.

    Mirrors ``coverage_gate_decision`` so the audit card can render
    coverage and retention with the same shape.
    """
    passed = retention_rate >= threshold
    overridden = (not passed) and override
    return {
        "retention_rate": float(retention_rate),
        "threshold": float(threshold),
        "passed": bool(passed),
        "overridden": bool(overridden),
        "effective_pass": bool(passed or overridden),
        "override_flag": override_flag,
    }


def build_coverage_metadata(
    coverage: dict[str, Any],
    *,
    threshold: float,
    override: bool,
    override_flag: str = "--allow-incomplete-mc-coverage",
) -> dict[str, Any]:
    """Return the ``mc_coverage`` metadata block consumers serialize.

    Compatible with the existing CSLI ``metadata.mc_coverage``
    schema so ``make_audit_card.py`` can read both old and new
    artifacts.
    """
    decision = coverage_gate_decision(
        coverage["coverage_rate"],
        threshold=threshold,
        override=override,
        override_flag=override_flag,
    )
    return {
        "test_dataset_qids": coverage["target_qids"],
        "mc_questions_total": coverage["mc_questions_total"],
        "matched_test_mc_questions": coverage["matched_questions"],
        "matched_test_mc_qids": coverage["matched_qids"],
        "missing_test_qids": coverage["missing_qids"],
        "coverage_rate": decision["coverage_rate"],
        "threshold": decision["threshold"],
        "passed": decision["passed"],
        "overridden": decision["overridden"],
        "override_flag": decision["override_flag"],
    }


def build_retention_metadata(
    build_metadata: dict[str, Any],
    *,
    split: str,
    smoke: bool,
    explicit_threshold: float | None,
    override: bool,
    override_flag: str = "--allow-low-mc-retention",
) -> dict[str, Any]:
    """Return the ``mc_retention_gate`` metadata block consumers serialize.

    Returns ``{"applies": False, ...}`` when ``build_metadata`` is
    not loaded so the audit card can render a clear "not available"
    state without raising.
    """
    block: dict[str, Any] = {
        "applies": build_metadata["status"] == "loaded",
        "split": split,
        "threshold": None,
        "retention_rate": None,
        "raw_count": None,
        "retained_count": None,
        "dropped_count": None,
        "passed": None,
        "overridden": False,
        "override_flag": override_flag,
    }
    if build_metadata["status"] != "loaded":
        return block

    splits = build_metadata.get("splits") or {}
    split_block = splits.get(split)
    if not isinstance(split_block, dict):
        return block

    threshold = metadata_retention_threshold(
        build_metadata,
        smoke=smoke,
        explicit_threshold=explicit_threshold,
    )
    decision = retention_gate_decision(
        split_block["retention_rate"],
        threshold=threshold,
        override=override,
        override_flag=override_flag,
    )
    block.update(
        {
            "threshold": decision["threshold"],
            "retention_rate": decision["retention_rate"],
            "raw_count": int(split_block["raw_count"]),
            "retained_count": int(split_block["retained_count"]),
            "dropped_count": int(split_block["dropped_count"]),
            "passed": decision["passed"],
            "overridden": decision["overridden"],
        }
    )
    return block
