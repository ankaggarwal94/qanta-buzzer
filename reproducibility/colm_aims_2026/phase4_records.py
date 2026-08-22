"""R-080 records export: producer boundary, model-free, oracle-validated.

Translates scored per-item DP outputs (raw ``stop`` integers under the
preserved fair-QA ``timeout_coded_as_horizon`` sentinel convention) into
canonical v2 event records ``records/<cell_id>.jsonl``, with the historical
``performat`` calibration label mapped to ``format_specific`` at this
boundary (R-080). The existing v2 ingestion is the oracle: every exported
row loads under ``schema.load_records_bytes``, passes
``schema.validate_record``, and classifies complete under
``pairing.classify_record``.

R-046 discipline: a DP stop EXACTLY AT the horizon is the historical
sentinel (NEVER_STOPPED, ``stop_step = null``,
``terminal_imputation = FINAL_PREFIX_IF_NEVER``); a stop BEYOND the horizon
is unreachable from the DP and is refused as frame corruption (amended
R-080, F-7); the derived reporting scalar stays recomputable via
``pairing.sentinel_coded_stop`` and is never stored into the canonical
record.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import pairing, schema


class RecordsExportError(schema.ColmAimsError):
    """Producer-boundary records export refused (R-080)."""


# Closed input contract: callers project scored frames down to EXACTLY this
# key set before export (R-080).
SCORED_ITEM_KEYS = frozenset({"item_key", "horizon", "mc_stop", "ref_stop"})

# DECISION: eligible horizons are >= 2 — single-prefix items are excluded
# upstream by the frozen eligibility artifact (R-074).
MIN_EXPORT_HORIZON = 2

_LABEL_MAP = {
    "performat": "format_specific",
    "shared": "shared",
    "format_specific": "format_specific",
}


def map_calibration_label(label: Any) -> str:
    """Map the historical ``performat`` calibration label to the v2
    ``format_specific`` spelling; ``shared``/``format_specific`` are
    identity. Anything else refuses (R-080)."""
    if isinstance(label, str) and label in _LABEL_MAP:
        return _LABEL_MAP[label]
    raise RecordsExportError(
        f"unknown calibration label {label!r} — expected one of"
        f" {sorted(_LABEL_MAP)} (R-080)"
    )


def _validate_cell_id(cell_id: Any) -> str:
    if not isinstance(cell_id, str) or not cell_id:
        raise RecordsExportError(
            "cell_id must be a non-empty string (R-080)"
        )
    # The boundary translation is EXPLICIT: legacy spellings are refused
    # with guidance toward the v2 label, in both the legacy "+" separator
    # and any smuggled "__performat" spelling.
    if "performat" in cell_id:
        raise RecordsExportError(
            f"cell_id {cell_id!r} carries the legacy 'performat' label —"
            " map it to 'format_specific' via map_calibration_label before"
            " export (R-080)"
        )
    if "+" in cell_id:
        raise RecordsExportError(
            f"cell_id {cell_id!r} uses the legacy '+' separator — v2 cell"
            " ids are '<reference_id>__<calibration_id>' with calibration"
            " in {'shared', 'format_specific'} (R-080)"
        )
    if cell_id not in schema.CELL_IDS:
        raise RecordsExportError(
            f"cell_id {cell_id!r} is not one of the ten frozen v2 grid cell"
            " ids (R-040/R-080)"
        )
    return cell_id


def _validate_stop(value: Any, field: str, item_key: str) -> int:
    if not schema.is_real_int(value):
        raise RecordsExportError(
            f"item {item_key!r}: {field} {value!r} is outside the"
            " non-negative integer stop domain — bools and floats are never"
            " coerced (R-061/R-080)"
        )
    if value < 0:
        raise RecordsExportError(
            f"item {item_key!r}: {field} {value!r} is negative — raw DP"
            " stops are zero-based non-negative integers (R-061/R-080)"
        )
    return int(value)


def _arm_fields(prefix: str, stop: int, horizon: int) -> dict[str, Any]:
    """Translate one arm's raw DP stop into the canonical event (R-046).

    ``stop == horizon`` (EXACTLY) is the preserved producer sentinel
    (``timeout_coded_as_horizon``): the threshold never crossed within the
    horizon. A stop strictly below the horizon — including the final prefix
    ``horizon - 1`` — is a genuine ``FINITE_STOP`` crossing. ``stop >
    horizon`` is unreachable from the DP and is refused upstream as frame
    corruption (amended R-080); the guard here is defense in depth so the
    weaker NEVER_STOPPED bucket can never absorb an overshoot.
    """
    if stop > horizon:
        raise RecordsExportError(
            f"{prefix}_stop {stop} > horizon {horizon} — an overshoot is"
            " unreachable from the DP and is refused as frame corruption,"
            " never absorbed into NEVER_STOPPED (amended R-080)"
        )
    if stop == horizon:
        return {
            f"{prefix}_event_status": schema.EVENT_NEVER,
            f"{prefix}_stop_step": None,
            f"{prefix}_terminal_imputation": schema.IMPUTATION_FINAL_PREFIX,
        }
    return {
        f"{prefix}_event_status": schema.EVENT_FINITE,
        f"{prefix}_stop_step": stop,
        f"{prefix}_terminal_imputation": schema.IMPUTATION_NONE,
    }


def export_records(
    scored_items: list[dict[str, Any]], cell_id: str, out_dir: Path
) -> Path:
    """Write ``out_dir / "records" / f"{cell_id}.jsonl"`` in the canonical
    v2 record schema (R-080).

    Input items are EXACTLY ``{item_key, horizon, mc_stop, ref_stop}``;
    unknown or missing keys refuse. Output rows are sorted ascending by
    UTF-8 ``item_key`` and are byte-identical under input permutation.
    """
    cell_id = _validate_cell_id(cell_id)
    if not isinstance(scored_items, list) or not scored_items:
        raise RecordsExportError(
            f"cell {cell_id!r}: scored_items must be a non-empty list —"
            " an empty export is a vacuously-empty record set (R-080)"
        )
    records: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    for index, item in enumerate(scored_items):
        where = f"cell {cell_id!r}: scored item [{index}]"
        if not isinstance(item, dict):
            raise RecordsExportError(f"{where}: must be an object (R-080)")
        unknown = sorted(set(item) - SCORED_ITEM_KEYS)
        if unknown:
            raise RecordsExportError(
                f"{where}: unknown field(s) {unknown} — the closed input"
                f" contract is exactly {sorted(SCORED_ITEM_KEYS)} (R-080)"
            )
        missing = sorted(SCORED_ITEM_KEYS - set(item))
        if missing:
            raise RecordsExportError(
                f"{where}: missing required field(s) {missing} (R-080)"
            )
        item_key = item["item_key"]
        if not isinstance(item_key, str) or not item_key:
            raise RecordsExportError(
                f"{where}: item_key must be a non-empty opaque string"
                " (R-031/R-080)"
            )
        if item_key in seen_keys:
            raise RecordsExportError(
                f"{where}: duplicate item key {item_key!r} — duplicate pair"
                " keys fail closed (R-008/R-080)"
            )
        seen_keys.add(item_key)
        horizon = item["horizon"]
        if not schema.is_real_int(horizon):
            raise RecordsExportError(
                f"item {item_key!r}: horizon {horizon!r} is outside the"
                " positive-int domain — bools never satisfy an integer"
                " domain (R-061/R-080)"
            )
        if horizon < MIN_EXPORT_HORIZON:
            raise RecordsExportError(
                f"item {item_key!r}: horizon {horizon!r} is below the"
                f" minimum eligible horizon {MIN_EXPORT_HORIZON} —"
                " single-prefix items are excluded upstream by the frozen"
                " eligibility artifact (R-074/R-080)"
            )
        horizon = int(horizon)
        mc_stop = _validate_stop(item["mc_stop"], "mc_stop", item_key)
        ref_stop = _validate_stop(item["ref_stop"], "ref_stop", item_key)
        # Amended R-080 (F-7): the DP sentinel is EXACTLY stop == horizon;
        # a stop BEYOND the horizon is unreachable from the DP solver and
        # is refused as frame corruption — never silently absorbed into the
        # weaker NEVER_STOPPED bucket (seed catalog: weaker-bucket
        # absorption is not fail-closed).
        for stop_field, stop_value in (
            ("mc_stop", mc_stop),
            ("ref_stop", ref_stop),
        ):
            if stop_value > horizon:
                raise RecordsExportError(
                    f"item {item_key!r}: {stop_field} {stop_value} overshoots"
                    f" the horizon {horizon} — unreachable from the DP,"
                    " refused as frame corruption (amended R-080)"
                )
        record: dict[str, Any] = {
            "item_key": item_key,
            "trajectory_horizon": horizon,
        }
        record.update(_arm_fields("mc", mc_stop, horizon))
        record.update(_arm_fields("ref", ref_stop, horizon))
        # Oracle self-check (R-080): the EXISTING v2 ingestion is the
        # contract — every exported record must validate and classify
        # complete before any byte reaches disk.
        try:
            schema.validate_record(record)
        except schema.RecordValidationError as exc:
            raise RecordsExportError(
                f"item {item_key!r}: constructed record failed the v2"
                f" oracle: {exc} (R-080)"
            ) from exc
        outcome = pairing.classify_record(record)
        if outcome.get("status") != "complete":
            raise RecordsExportError(
                f"item {item_key!r}: constructed record classifies as"
                f" {outcome.get('status')!r}, not 'complete' (R-080)"
            )
        records.append(record)

    records.sort(key=lambda rec: rec["item_key"].encode("utf-8"))
    blob = (
        "\n".join(
            json.dumps(
                rec, sort_keys=True, separators=(",", ":"), ensure_ascii=True
            )
            for rec in records
        )
        + "\n"
    ).encode("utf-8")
    records_dir = Path(out_dir) / "records"
    records_dir.mkdir(parents=True, exist_ok=True)
    out_path = records_dir / f"{cell_id}.jsonl"
    out_path.write_bytes(blob)
    return out_path
