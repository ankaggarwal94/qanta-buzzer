#!/usr/bin/env python3
"""Durable host-side control plane for the StopDFF v5 Modal pipeline.

This module owns the local controller state: the fsynced checkpoint file, the
hash-linked event journal, plan validation, attempt-scoped adapter naming, the
per-stage result validators, and the ``run_control_plane`` driver that walks
the canonical Modal stage order. Everything here runs on the host — no code in
this module executes inside a Modal container and nothing here imports
``modal``.

The Modal stage functions themselves (and the app/image/volume registration)
live in ``scripts/modal_stopdff_v5_runner.py``, which re-exports these names
so existing imports and ``modal run …::control_main`` keep working. The
runner's facade binds the deployment-specific values (validated image source
ID, staged source dir, default stage API) that ``run_control_plane`` receives
here as explicit parameters.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

from scripts.stopdff_v5.fileio import publish_bytes

_ADAPTER_COMPONENT_MAX_BYTES = 255


def _canonical_adapter_subdir(value: object) -> str:
    """Return one canonical adapter path component or fail closed."""
    if not isinstance(value, str) or not value:
        raise ValueError("adapter subdir must be a nonempty string")
    if "\0" in value:
        raise ValueError("adapter subdir must not contain NUL")
    from pathlib import PurePosixPath

    parsed = PurePosixPath(value)
    if (
        parsed.is_absolute()
        or ".." in parsed.parts
        or len(parsed.parts) != 1
        or str(parsed) != value
    ):
        raise ValueError(f"unsafe or noncanonical adapter subdir: {value!r}")
    if len(value.encode("utf-8")) > _ADAPTER_COMPONENT_MAX_BYTES:
        raise ValueError("adapter subdir must be at most 255 UTF-8 bytes")
    return value


def _retry_adapter_subdir(base: str, attempt: int) -> str:
    """Derive a stable retry component within the internal byte contract."""
    attempt_text = str(attempt)
    readable_suffix = f"__attempt_{attempt_text}"
    candidate = f"{base}{readable_suffix}"
    if len(candidate.encode("utf-8")) <= _ADAPTER_COMPONENT_MAX_BYTES:
        return candidate

    digest = hashlib.sha256(
        f"{base}\0{attempt_text}".encode("utf-8")
    ).hexdigest()[:16]
    if len(readable_suffix.encode("utf-8")) <= 48:
        suffix = f"{readable_suffix}_{digest}"
    else:
        suffix = f"__attempt_{digest}"
    prefix_budget = _ADAPTER_COMPONENT_MAX_BYTES - len(suffix.encode("utf-8"))
    prefix = base.encode("utf-8")[:prefix_budget].decode(
        "utf-8",
        errors="ignore",
    )
    return f"{prefix}{suffix}"


def _control_plan_digest(plan: dict) -> str:
    return hashlib.sha256(
        json.dumps(
            plan,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


def _load_control_json(path: Path) -> dict:
    from scripts.stopdff_v5.identity import loads_no_duplicate_keys

    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"control JSON is missing or noncanonical: {path}")
    value = loads_no_duplicate_keys(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"control JSON must contain an object: {path}")
    return value


def _write_control_state(path: Path, state: dict) -> None:
    """Atomically replace and fsync the local control-plane checkpoint."""
    data = (json.dumps(state, indent=2, sort_keys=True) + "\n").encode("utf-8")
    _atomic_replace_control_bytes(path, data)


def _atomic_replace_control_bytes(path: Path, data: bytes) -> None:
    """Replace one local control artifact without exposing partial bytes.

    Delegates to the package-wide durable-write primitive so the atomic
    publish discipline (same-dir temp, fsync file, ``os.replace``, fsync
    directory) is implemented in exactly one place.
    """
    publish_bytes(Path(path), data)


def _atomic_create_control_bytes(path: Path, data: bytes) -> None:
    """Publish one fsynced control artifact without replacing any path."""
    import tempfile

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
                f"create-once control artifact already exists: {path}"
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


def _append_control_event(path: Path, event: dict) -> None:
    """Atomically append one canonical event to the local journal."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = b""
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"control journal is noncanonical: {path}")
        existing = path.read_bytes()
        if existing and not existing.endswith(b"\n"):
            raise ValueError("control journal has an unterminated tail")
    line = (json.dumps(event, sort_keys=True) + "\n").encode("utf-8")
    _atomic_replace_control_bytes(path, existing + line)


_CONTROL_EVENT_NAMES = {
    "control_completed",
    "control_initialized",
    "control_recovery_required",
    "control_revalidated",
    "stage_checkpoint_invalid",
    "stage_checkpoint_refresh_required",
    "stage_completed",
    "stage_failed",
    "stage_started",
}
_CONTROL_STAGE_ORDER = (
    "verify_source",
    "verify_raw",
    "environment_probe",
    "freeze_model",
    "adapter_determinism",
    "promote_adapter",
    "fvi_study",
    "smoke_bootstrap",
    "smoke_sweep",
    "mutation_gate",
    "final_bootstrap",
    "final_sweep",
    "package",
    "validate_package",
)
_CONTROL_STAGE_NAMES = set(_CONTROL_STAGE_ORDER)
_CONTROL_RESULT_FIELDS = {
    "run_id",
    "run_spec_id",
    "adapter_id",
    "receipt_ids",
    "validation",
}


def _control_event_sha256(record: dict) -> str:
    return hashlib.sha256(
        json.dumps(
            record,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


def _control_payload_sha256(payload: object) -> str:
    """Hash finite canonical JSON used in a durable controller checkpoint."""
    try:
        data = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("control payload is not finite canonical JSON") from exc
    return hashlib.sha256(data).hexdigest()


def _validate_control_event_record(
    record: object,
    *,
    expected_sequence: int,
    previous_record: dict | None,
) -> dict:
    """Validate one canonical, hash-linked control-journal record."""
    required = {
        "sequence",
        "event",
        "stage",
        "utc_epoch_seconds",
        "detail",
        "previous_event_sha256",
    }
    if not isinstance(record, dict) or set(record) != required:
        raise ValueError("control journal record schema is invalid")
    if record.get("sequence") != expected_sequence:
        raise ValueError("control journal sequence is not contiguous")
    event = record.get("event")
    if not isinstance(event, str) or event not in _CONTROL_EVENT_NAMES:
        raise ValueError("control journal event is unknown")
    timestamp = record.get("utc_epoch_seconds")
    if (
        not isinstance(timestamp, int)
        or isinstance(timestamp, bool)
        or timestamp < 0
    ):
        raise ValueError("control journal timestamp is invalid")
    detail = record.get("detail")
    if not isinstance(detail, dict):
        raise ValueError("control journal detail must be an object")

    expected_previous = (
        _control_event_sha256(previous_record)
        if previous_record is not None
        else None
    )
    if record.get("previous_event_sha256") != expected_previous:
        raise ValueError("control journal hash chain is invalid")
    if (
        previous_record is not None
        and timestamp < previous_record["utc_epoch_seconds"]
    ):
        raise ValueError("control journal timestamps are not monotonic")

    stage = record.get("stage")
    stage_event = event.startswith("stage_") or event in {
        "control_recovery_required",
        "control_revalidated",
    }
    if stage_event:
        if not isinstance(stage, str) or stage not in _CONTROL_STAGE_NAMES:
            raise ValueError("control journal stage is invalid")
    elif stage is not None:
        raise ValueError("control journal event must not name a stage")

    if expected_sequence == 1 and event != "control_initialized":
        raise ValueError("control journal must begin with initialization")
    if event == "control_initialized" and (
        expected_sequence != 1 or detail
    ):
        raise ValueError("control initialization event is invalid")
    detail_fields = {
        "control_completed": {"run_id", "run_spec_id", "result_sha256"},
        "control_initialized": set(),
        "control_recovery_required": {"stage", "type", "message"},
        "control_revalidated": {"run_id", "result_sha256"},
        "stage_checkpoint_invalid": {"attempt", "stage", "type", "message"},
        "stage_checkpoint_refresh_required": {"reason"},
        "stage_completed": {"attempt", "result_sha256"},
        "stage_failed": {"attempt", "stage", "type", "message"},
        "stage_started": {"attempt"},
    }
    if set(detail) != detail_fields[event]:
        raise ValueError("control journal event detail schema is invalid")
    if event in {
        "stage_checkpoint_invalid",
        "stage_completed",
        "stage_failed",
        "stage_started",
    }:
        attempt = detail.get("attempt")
        if (
            not isinstance(attempt, int)
            or isinstance(attempt, bool)
            or attempt < 1
        ):
            raise ValueError("control journal stage attempt is invalid")
    if event in {"stage_checkpoint_invalid", "stage_failed"}:
        if (
            detail.get("stage") != stage
            or not isinstance(detail.get("type"), str)
            or not detail.get("type")
            or not isinstance(detail.get("message"), str)
        ):
            raise ValueError("control journal stage failure detail is invalid")
    if event == "stage_checkpoint_refresh_required" and not isinstance(
        detail.get("reason"), str
    ):
        raise ValueError("control journal refresh detail is invalid")
    if event == "control_recovery_required" and (
        not isinstance(detail.get("stage"), str)
        or not isinstance(detail.get("type"), str)
        or not isinstance(detail.get("message"), str)
    ):
        raise ValueError("control journal recovery detail is invalid")
    if event == "control_revalidated" and not _is_final_control_run_id(
        detail.get("run_id")
    ):
        raise ValueError("control journal revalidation detail is invalid")
    if event == "control_completed" and (
        not _is_final_control_run_id(detail.get("run_id"))
        or not _is_control_sha(detail.get("run_spec_id"))
    ):
        raise ValueError("control journal completion detail is invalid")
    if event in {
        "stage_completed",
        "control_completed",
        "control_revalidated",
    } and not _is_control_sha(detail.get("result_sha256")):
        raise ValueError("control journal result digest is invalid")
    return record


def _validate_control_journal_projection(
    records: list[dict],
    state: dict,
) -> None:
    """Replay the journal's stage projection and bind it to the checkpoint."""
    attempts: dict[str, int] = {}
    active: tuple[str, int] | None = None
    completed: set[str] = set()
    completed_digests: dict[str, str] = {}
    terminal_result_digest: str | None = None
    terminal_run_id: str | None = None
    terminal_run_spec_id: str | None = None
    terminal_seen = False

    def require_completed_prefix() -> None:
        expected = set(_CONTROL_STAGE_ORDER[: len(completed)])
        if completed != expected:
            raise ValueError(
                "control journal completed stages are not a canonical prefix"
            )

    for record in records:
        event = record["event"]
        stage = record["stage"]
        detail = record["detail"]
        if terminal_seen and event.startswith("stage_"):
            raise ValueError(
                "control journal has stage activity after a terminal event"
            )
        if event == "stage_started":
            attempt = detail["attempt"]
            if attempt != attempts.get(stage, 0) + 1:
                raise ValueError("control journal stage attempts are inconsistent")
            if active is not None:
                raise ValueError("control journal has overlapping active stages")
            stage_index = _CONTROL_STAGE_ORDER.index(stage)
            if completed != set(_CONTROL_STAGE_ORDER[:stage_index]):
                raise ValueError(
                    "control journal stage start lacks its completed predecessors"
                )
            attempts[stage] = attempt
            active = (stage, attempt)
        elif event in {"stage_completed", "stage_failed"}:
            if active != (stage, detail["attempt"]):
                raise ValueError(
                    "control journal stage terminal event lacks its start"
                )
            active = None
            if event == "stage_completed":
                completed.add(stage)
                completed_digests[stage] = detail["result_sha256"]
                require_completed_prefix()
            else:
                completed.discard(stage)
                completed_digests.pop(stage, None)
        elif event == "stage_checkpoint_invalid":
            if active is not None:
                raise ValueError(
                    "control journal invalidated a checkpoint with an active stage"
                )
            if (
                stage not in completed
                or detail["attempt"] != attempts.get(stage)
            ):
                raise ValueError(
                    "control journal invalidation lacks a completed checkpoint"
                )
            completed.remove(stage)
            completed_digests.pop(stage, None)
            require_completed_prefix()
        elif event == "stage_checkpoint_refresh_required":
            if active is not None:
                raise ValueError(
                    "control journal refreshed a checkpoint with an active stage"
                )
            if stage not in completed:
                raise ValueError(
                    "control journal refresh lacks a completed checkpoint"
                )
            completed.remove(stage)
            completed_digests.pop(stage, None)
            require_completed_prefix()
        elif event in {
            "control_completed",
            "control_recovery_required",
            "control_revalidated",
        }:
            if active is not None:
                raise ValueError(
                    "control journal terminal event has an active stage"
                )
            if completed != _CONTROL_STAGE_NAMES:
                raise ValueError(
                    "control journal terminal event lacks completed stages"
                )
            if event == "control_completed" and terminal_seen:
                raise ValueError(
                    "control journal has a duplicate completion event"
                )
            if event in {
                "control_recovery_required",
                "control_revalidated",
            } and not terminal_seen:
                raise ValueError(
                    "control journal recovery event lacks prior completion"
                )
            terminal_seen = True
            if event in {"control_completed", "control_revalidated"}:
                terminal_result_digest = detail["result_sha256"]
                terminal_run_id = detail["run_id"]
                terminal_run_spec_id = (
                    detail["run_spec_id"]
                    if event == "control_completed"
                    else None
                )

    strict_state = state.get("schema_version") == 4
    state_attempts = state.get("stage_attempts")
    if strict_state and not isinstance(state_attempts, dict):
        raise ValueError("control state stage attempts must be an object")
    if isinstance(state_attempts, dict) and state_attempts != attempts:
        raise ValueError("control state stage attempts disagree with journal")
    state_completed = state.get("completed")
    if strict_state and not isinstance(state_completed, dict):
        raise ValueError("control state completed stages must be an object")
    if isinstance(state_completed, dict) and set(state_completed) != completed:
        raise ValueError("control state completed stages disagree with journal")
    if isinstance(state_completed, dict):
        for stage in sorted(completed):
            if (
                _control_payload_sha256(state_completed[stage])
                != completed_digests.get(stage)
            ):
                raise ValueError(
                    "control state completed payload disagrees with journal"
                )

    if strict_state:
        if not records:
            raise ValueError("schema-v4 control journal cannot be empty")
        last = records[-1]
        expected_status = {
            "control_completed": "completed",
            "control_initialized": "initialized",
            "control_recovery_required": "recovery_required",
            "control_revalidated": "completed",
            "stage_checkpoint_invalid": "running",
            "stage_checkpoint_refresh_required": "running",
            "stage_completed": "running",
            "stage_failed": "failed",
            "stage_started": "running",
        }[last["event"]]
        if state.get("status") != expected_status:
            raise ValueError("control state status disagrees with journal")
        if terminal_result_digest is None:
            if "result" in state:
                raise ValueError("control state has an unbound terminal result")
        else:
            result = state.get("result")
            if (
                not isinstance(result, dict)
                or set(result) != _CONTROL_RESULT_FIELDS
                or result.get("run_id") != terminal_run_id
                or _control_payload_sha256(result)
                != terminal_result_digest
                or (
                    terminal_run_spec_id is not None
                    and result.get("run_spec_id")
                    != terminal_run_spec_id
                )
            ):
                raise ValueError("control state result disagrees with journal")
        if last["event"] in {"control_recovery_required", "stage_failed"}:
            last_error = state.get("last_error")
            expected_error = {
                key: last["detail"][key]
                for key in ("stage", "type", "message")
            }
            if last_error != expected_error:
                raise ValueError("control state error disagrees with journal")


def _reconcile_control_journal(state_path: Path, state: dict) -> None:
    """Repair one provable final-record gap or reject journal drift."""
    from scripts.stopdff_v5.identity import loads_no_duplicate_keys

    journal_path = state_path.with_name(state_path.name + ".jsonl")
    journal_bytes = b""
    if journal_path.exists() or journal_path.is_symlink():
        if journal_path.is_symlink() or not journal_path.is_file():
            raise ValueError("control journal is noncanonical")
        journal_bytes = journal_path.read_bytes()

    complete_lines: list[bytes]
    torn_tail: bytes | None
    if journal_bytes and not journal_bytes.endswith(b"\n"):
        parts = journal_bytes.split(b"\n")
        complete_lines = parts[:-1]
        torn_tail = parts[-1]
    else:
        complete_lines = (
            journal_bytes[:-1].split(b"\n") if journal_bytes else []
        )
        torn_tail = None

    records: list[dict] = []
    for line_number, line in enumerate(complete_lines, start=1):
        if not line:
            raise ValueError(
                f"control journal line {line_number} is empty"
            )
        try:
            record = loads_no_duplicate_keys(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise ValueError(
                f"control journal line {line_number} is invalid JSON"
            ) from exc
        canonical_line = json.dumps(record, sort_keys=True).encode("utf-8")
        if line != canonical_line:
            raise ValueError(
                f"control journal line {line_number} is not canonical JSON"
            )
        records.append(_validate_control_event_record(
            record,
            expected_sequence=line_number,
            previous_record=records[-1] if records else None,
        ))

    sequence = state.get("sequence", 0)
    if (
        not isinstance(sequence, int)
        or isinstance(sequence, bool)
        or sequence < 0
    ):
        raise ValueError("control state sequence is invalid")
    last_event = state.get("last_event")
    if sequence == 0 and last_event is not None:
        raise ValueError("empty control state must not contain a last event")
    if torn_tail is not None:
        canonical_last = (
            json.dumps(last_event, sort_keys=True) + "\n"
        ).encode("utf-8") if isinstance(last_event, dict) else b""
        if (
            not torn_tail
            or len(records) != sequence - 1
            or not isinstance(last_event, dict)
            or last_event.get("sequence") != sequence
            or not canonical_last[:-1].startswith(torn_tail)
        ):
            raise ValueError("control journal has an unprovable torn tail")
        validated_last = _validate_control_event_record(
            last_event,
            expected_sequence=sequence,
            previous_record=records[-1] if records else None,
        )
        _validate_control_journal_projection(records + [validated_last], state)
        complete_prefix = journal_bytes[:-len(torn_tail)]
        _atomic_replace_control_bytes(
            journal_path,
            complete_prefix + canonical_last,
        )
        records.append(validated_last)

    if len(records) == sequence:
        if sequence and records[-1] != last_event:
            raise ValueError("control state and journal last event disagree")
        _validate_control_journal_projection(records, state)
        return
    if (
        len(records) == sequence - 1
        and isinstance(last_event, dict)
        and last_event.get("sequence") == sequence
    ):
        validated_last = _validate_control_event_record(
            last_event,
            expected_sequence=sequence,
            previous_record=records[-1] if records else None,
        )
        _validate_control_journal_projection(records + [validated_last], state)
        _append_control_event(journal_path, last_event)
        return
    raise ValueError("control state and journal sequence disagree")


def _record_control_event(
    state_path: Path,
    state: dict,
    *,
    event: str,
    stage: str | None = None,
    detail: dict | None = None,
) -> None:
    event_detail = dict(detail or {})
    if event == "stage_completed":
        completed = state.get("completed")
        if not isinstance(completed, dict) or stage not in completed:
            raise ValueError("completed stage event lacks checkpoint payload")
        event_detail["result_sha256"] = _control_payload_sha256(
            completed[stage]
        )
    elif event in {"control_completed", "control_revalidated"}:
        event_detail["result_sha256"] = _control_payload_sha256(
            state.get("result")
        )
    state["sequence"] = int(state.get("sequence", 0)) + 1
    previous_event = state.get("last_event")
    record = {
        "sequence": state["sequence"],
        "event": event,
        "stage": stage,
        "utc_epoch_seconds": int(time.time()),
        "detail": event_detail,
        "previous_event_sha256": (
            _control_event_sha256(previous_event)
            if isinstance(previous_event, dict)
            else None
        ),
    }
    _validate_control_event_record(
        record,
        expected_sequence=state["sequence"],
        previous_record=(
            previous_event if isinstance(previous_event, dict) else None
        ),
    )
    state["last_event"] = record
    _write_control_state(state_path, state)
    _append_control_event(
        state_path.with_name(state_path.name + ".jsonl"),
        record,
    )


def _close_interrupted_control_attempt(
    state_path: Path,
    state: dict,
) -> bool:
    """Close a host-abandoned stage before a resumed controller does work."""
    last_event = state.get("last_event")
    if (
        not isinstance(last_event, dict)
        or last_event.get("event") != "stage_started"
    ):
        return False
    stage = last_event.get("stage")
    detail = last_event.get("detail")
    attempt = detail.get("attempt") if isinstance(detail, dict) else None
    if stage not in _CONTROL_STAGE_NAMES or not isinstance(attempt, int):
        raise ValueError("active control attempt is noncanonical")
    error = {
        "stage": stage,
        "type": "HostControllerInterrupted",
        "message": (
            "controller resumed after a stage start without a terminal event"
        ),
    }
    state["status"] = "failed"
    state["last_error"] = error
    _record_control_event(
        state_path,
        state,
        event="stage_failed",
        stage=stage,
        detail={"attempt": attempt, **error},
    )
    return True


def _validate_control_plan(plan: dict) -> dict:
    from scripts.stopdff_v5.identity import compute_id

    allowed = {
        "source_id",
        "raw_id",
        "adapter_subdirs",
        "gate_overrides",
        "resource_summary",
        "resource_summary_id",
    }
    if set(plan) - allowed:
        raise ValueError(
            f"unknown control-plan fields: {sorted(set(plan) - allowed)}"
        )

    def require_sha(name: str) -> str:
        value = plan.get(name)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(ch not in "0123456789abcdef" for ch in value)
        ):
            raise ValueError(f"control plan {name} must be canonical 64-hex")
        return value

    source_id = require_sha("source_id")
    raw_id = require_sha("raw_id")
    subdirs = plan.get("adapter_subdirs")
    if (
        not isinstance(subdirs, list)
        or len(subdirs) != 2
        or not all(isinstance(value, str) and value for value in subdirs)
    ):
        raise ValueError("control plan requires two distinct adapter_subdirs")
    canonical_subdirs = [_canonical_adapter_subdir(value) for value in subdirs]
    if len(set(canonical_subdirs)) != 2:
        raise ValueError("control plan requires two distinct adapter_subdirs")
    if any("__attempt_" in value for value in canonical_subdirs):
        raise ValueError(
            "control plan adapter_subdirs use a reserved retry namespace"
        )
    gate_overrides = plan.get("gate_overrides", {})
    if (
        not isinstance(gate_overrides, dict)
        or set(gate_overrides)
        - {"allow_low_mc_retention", "allow_incomplete_mc_coverage"}
        or not all(isinstance(value, bool) for value in gate_overrides.values())
    ):
        raise ValueError("control plan contains invalid gate_overrides")
    resource_summary = plan.get("resource_summary", {})
    if not isinstance(resource_summary, dict):
        raise ValueError("control plan resource_summary must be an object")
    try:
        resource_summary_id = compute_id(resource_summary)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "control plan resource_summary is not canonically identity-safe"
        ) from exc
    supplied_resource_id = plan.get("resource_summary_id")
    if (
        supplied_resource_id is not None
        and supplied_resource_id != resource_summary_id
    ):
        raise ValueError("control plan resource_summary_id mismatch")
    return {
        "source_id": source_id,
        "raw_id": raw_id,
        "adapter_subdirs": canonical_subdirs,
        "gate_overrides": dict(gate_overrides),
        "resource_summary": dict(resource_summary),
        "resource_summary_id": resource_summary_id,
    }


def _adapter_attempt_subdirs(
    base_subdirs: list[str],
    attempt: int,
) -> tuple[str, str]:
    """Derive fresh adapter destinations for one controller attempt."""
    if (
        not isinstance(attempt, int)
        or isinstance(attempt, bool)
        or attempt < 1
    ):
        raise ValueError("adapter attempt must be a positive integer")
    if len(base_subdirs) != 2:
        raise ValueError("adapter attempt requires two base subdirs")
    canonical_bases = tuple(
        _canonical_adapter_subdir(base) for base in base_subdirs
    )
    if attempt == 1:
        candidates = canonical_bases
    else:
        candidates = tuple(
            _retry_adapter_subdir(base, attempt) for base in canonical_bases
        )
    first = _canonical_adapter_subdir(candidates[0])
    second = _canonical_adapter_subdir(candidates[1])
    if first == second:
        raise ValueError("adapter attempt requires distinct subdirs")
    return first, second


def _invalidate_control_dependents(
    state_path: Path,
    state: dict,
    *,
    upstream: str,
    reason: str,
) -> None:
    """Remove and journal every completed suffix dependent of ``upstream``."""
    if upstream not in _CONTROL_STAGE_NAMES:
        raise ValueError("cannot invalidate dependents of an unknown stage")
    completed = state.setdefault("completed", {})
    attempts = state.setdefault("stage_attempts", {})
    upstream_index = _CONTROL_STAGE_ORDER.index(upstream)
    for dependent in reversed(_CONTROL_STAGE_ORDER[upstream_index + 1 :]):
        if dependent not in completed:
            continue
        attempt = attempts.get(dependent)
        if (
            not isinstance(attempt, int)
            or isinstance(attempt, bool)
            or attempt < 1
        ):
            raise ValueError(
                f"completed dependent {dependent} lacks a canonical attempt"
            )
        completed.pop(dependent)
        state["status"] = "running"
        _record_control_event(
            state_path,
            state,
            event="stage_checkpoint_invalid",
            stage=dependent,
            detail={
                "attempt": attempt,
                "stage": dependent,
                "type": "DependencyInvalidated",
                "message": (
                    f"upstream stage {upstream} requires refresh: {reason}"
                ),
            },
        )


def _refresh_control_stage(
    state_path: Path,
    state: dict,
    *,
    stage: str,
    reason: str,
) -> None:
    """Explicitly refresh a completed stage and all transitive dependents."""
    completed = state.setdefault("completed", {})
    if stage not in completed:
        return
    _invalidate_control_dependents(
        state_path,
        state,
        upstream=stage,
        reason=reason,
    )
    completed.pop(stage)
    state["status"] = "running"
    _record_control_event(
        state_path,
        state,
        event="stage_checkpoint_refresh_required",
        stage=stage,
        detail={"reason": reason},
    )


def _run_control_stage(
    state_path: Path,
    state: dict,
    *,
    name: str,
    invoke,
    validate_result,
) -> dict:
    if not callable(validate_result):
        raise TypeError(f"control stage {name} requires a result validator")
    completed = state.setdefault("completed", {})
    attempts = state.setdefault("stage_attempts", {})
    if name in completed:
        result = completed[name]
        try:
            _validate_control_stage_result(name, result, validate_result)
        except Exception as exc:
            _invalidate_control_dependents(
                state_path,
                state,
                upstream=name,
                reason=f"{type(exc).__name__}: {exc}",
            )
            completed.pop(name, None)
            prior_attempt = attempts.get(name, 0)
            if (
                not isinstance(prior_attempt, int)
                or isinstance(prior_attempt, bool)
                or prior_attempt < 1
            ):
                prior_attempt = 1
                attempts[name] = prior_attempt
            state["status"] = "running"
            state["last_error"] = {
                "stage": name,
                "type": type(exc).__name__,
                "message": str(exc),
            }
            _record_control_event(
                state_path,
                state,
                event="stage_checkpoint_invalid",
                stage=name,
                detail={"attempt": prior_attempt, **state["last_error"]},
            )
        else:
            return result
    attempt = int(attempts.get(name, 0)) + 1
    attempts[name] = attempt
    state["status"] = "running"
    state.pop("last_error", None)
    _record_control_event(
        state_path,
        state,
        event="stage_started",
        stage=name,
        detail={"attempt": attempt},
    )
    try:
        result = invoke(attempt)
        _validate_control_stage_result(name, result, validate_result)
    except BaseException as exc:
        state["status"] = "failed"
        state["last_error"] = {
            "stage": name,
            "type": type(exc).__name__,
            "message": str(exc),
        }
        _record_control_event(
            state_path,
            state,
            event="stage_failed",
            stage=name,
            detail={"attempt": attempt, **state["last_error"]},
        )
        raise
    completed[name] = result
    state.pop("last_error", None)
    _record_control_event(
        state_path,
        state,
        event="stage_completed",
        stage=name,
        detail={"attempt": attempt},
    )
    return result


def _validate_control_stage_result(
    stage: str,
    result,
    validate_result,
) -> None:
    """Apply generic and stage-specific success checks before checkpointing."""
    if not isinstance(result, dict):
        raise TypeError(f"control stage {stage} returned a non-object")
    if result.get("ok") is False:
        raise ValueError(
            f"control stage {stage} returned ok=false: "
            f"{result.get('error') or result.get('errors')}"
        )
    if result.get("passed") is False:
        raise ValueError(
            f"control stage {stage} returned passed=false: "
            f"{result.get('error') or result.get('errors')}"
        )
    validate_result(result)


def _require_control_sha(
    stage: str,
    result: dict,
    *,
    field: str,
    expected: str | None = None,
) -> str:
    value = result.get(field)
    if not _is_control_sha(value):
        raise ValueError(
            f"control stage {stage} returned a noncanonical {field}"
        )
    if expected is not None and value != expected:
        raise ValueError(
            f"control stage {stage} returned {field}={value!r}, "
            f"expected {expected!r}"
        )
    return value


def _is_control_sha(value) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(ch in "0123456789abcdef" for ch in value)
    )


def _is_final_control_run_id(value) -> bool:
    prefix = "final_modal_"
    return (
        isinstance(value, str)
        and value.startswith(prefix)
        and len(value) == len(prefix) + 12
        and all(ch in "0123456789abcdef" for ch in value[len(prefix):])
    )


def _require_control_bool(stage: str, result: dict, field: str) -> bool:
    value = result.get(field)
    if not isinstance(value, bool):
        raise ValueError(f"control stage {stage} returned an invalid {field}")
    return value


def _require_control_count(
    stage: str,
    result: dict,
    field: str,
    *,
    positive: bool = False,
) -> int:
    value = result.get(field)
    lower_bound = 1 if positive else 0
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < lower_bound
    ):
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(
            f"control stage {stage} returned a non-{qualifier} {field}"
        )
    return value


def _validate_verified_artifact_result(
    stage: str,
    result: dict,
    *,
    expected_id: str,
    require_myopic: bool,
) -> None:
    if result.get("ok") is not True or result.get("mismatches") != []:
        raise ValueError(f"control stage {stage} did not verify cleanly")
    _require_control_sha(stage, result, field="id", expected=expected_id)
    _require_control_count(stage, result, "n_files", positive=True)
    if require_myopic:
        _require_control_sha(stage, result, field="myopic_artifact_sha256")


def _validate_probe_result(result: dict, package_names: tuple[str, ...]) -> None:
    stage = "environment_probe"
    python_version = result.get("python")
    versions = result.get("package_versions")
    if not isinstance(python_version, str) or not python_version:
        raise ValueError("environment probe returned an invalid Python version")
    if not isinstance(versions, dict) or set(versions) != set(package_names):
        raise ValueError("environment probe returned an incomplete package set")
    if not all(isinstance(value, str) and value for value in versions.values()):
        raise ValueError(f"control stage {stage} returned an invalid package version")


def _validate_model_result(result: dict) -> None:
    _require_control_sha("freeze_model", result, field="model_id")
    _require_control_bool("freeze_model", result, "cached")


def _validate_adapter_result(
    stage: str,
    result: dict,
    *,
    expected_subdir: str,
    expected_id: str | None = None,
    expected_source_id: str | None = None,
    expected_raw_id: str | None = None,
    expected_model_id: str | None = None,
    require_fresh: bool = False,
) -> None:
    _require_control_sha(
        stage,
        result,
        field="adapter_id",
        expected=expected_id,
    )
    _require_control_sha(stage, result, field="fit_rows_sha256")
    _require_control_sha(stage, result, field="eval_rows_sha256")
    if result.get("subdir") != expected_subdir:
        raise ValueError(f"control stage {stage} returned the wrong subdir")
    _require_control_bool(stage, result, "cached")
    expected_bindings = {
        "source_manifest_id": expected_source_id,
        "raw_input_bundle_id": expected_raw_id,
        "model_snapshot_id": expected_model_id,
    }
    for field, expected in expected_bindings.items():
        if expected is not None and result.get(field) != expected:
            raise ValueError(f"control stage {stage} returned the wrong {field}")
    if require_fresh and result.get("cached") is not False:
        raise ValueError(f"control stage {stage} did not execute a fresh build")


def _validate_determinism_result(result: dict, source_id: str) -> None:
    stage = "adapter_determinism"
    if result.get("ok") is not True:
        raise ValueError("adapter determinism did not pass")
    _require_control_sha(stage, result, field="adapter_id")
    _require_control_sha(
        stage, result, field="source_manifest_id", expected=source_id
    )
    first_execution = result.get("first_build_execution_id")
    second_execution = result.get("second_build_execution_id")
    if (
        not isinstance(first_execution, str)
        or not first_execution
        or not isinstance(second_execution, str)
        or not second_execution
        or first_execution == second_execution
    ):
        raise ValueError("adapter determinism returned invalid build executions")
    _require_control_sha(stage, result, field="prerequisite_receipt_id")


def _validate_promotion_result(result: dict, adapter_id: str) -> None:
    stage = "promote_adapter"
    if result.get("canonical_subdir") != f"canonical_{adapter_id}":
        raise ValueError("adapter promotion returned the wrong destination")
    _require_control_bool(stage, result, "cached")


def _validate_fvi_result(result: dict) -> None:
    stage = "fvi_study"
    _require_control_sha(stage, result, field="fvi_study_id")
    selected = result.get("selected")
    if not isinstance(selected, dict) or set(selected) != {
        "tolerance",
        "max_iterations",
    }:
        raise ValueError("FVI stage returned an incomplete selection")
    if not isinstance(selected["tolerance"], str) or not selected["tolerance"]:
        raise ValueError("FVI stage returned an invalid tolerance")
    if (
        not isinstance(selected["max_iterations"], int)
        or isinstance(selected["max_iterations"], bool)
        or selected["max_iterations"] < 1
    ):
        raise ValueError("FVI stage returned invalid max_iterations")
    _require_control_bool(stage, result, "cached")


def _validate_bootstrap_result(
    stage: str,
    result: dict,
    replicates: int,
) -> None:
    _require_control_sha(stage, result, field="bootstrap_plan_id")
    if result.get("replicates") != replicates:
        raise ValueError(f"control stage {stage} returned the wrong replicate count")
    _require_control_count(stage, result, "n_items", positive=True)
    _require_control_bool(stage, result, "cached")


def _validate_sweep_result(
    stage: str,
    result: dict,
    *,
    run_id: str,
    require_receipt: bool,
) -> None:
    if result.get("run_id") != run_id:
        raise ValueError(f"control stage {stage} returned the wrong run_id")
    requested = _require_control_count(stage, result, "requested", positive=True)
    completed = _require_control_count(stage, result, "completed")
    skipped = _require_control_count(stage, result, "skipped")
    failed = _require_control_count(stage, result, "failed")
    if completed != requested or skipped != 0 or failed != 0:
        raise ValueError(f"control stage {stage} did not complete every cell")
    if result.get("release_status") != "VALID":
        raise ValueError(f"control stage {stage} did not produce a valid release")
    if not isinstance(result.get("family"), dict):
        raise ValueError(f"control stage {stage} returned no family result")
    if require_receipt:
        _require_control_sha(stage, result, field="prerequisite_receipt_id")


def _validate_mutation_result(result: dict, source_id: str) -> None:
    stage = "mutation_gate"
    if result.get("ok") is not True or result.get("unexpected") != []:
        raise ValueError("mutation gate did not pass cleanly")
    _require_control_count(stage, result, "n", positive=True)
    _require_control_sha(
        stage, result, field="source_manifest_id", expected=source_id
    )
    _require_control_sha(stage, result, field="prerequisite_receipt_id")


def _validate_checker_result(
    stage: str,
    result: dict,
    *,
    expected_adapter_id: str | None = None,
) -> None:
    if result.get("passed") is not True or result.get("errors") != []:
        raise ValueError(f"control stage {stage} did not pass cleanly")
    recomputed = result.get("recomputed")
    if (
        not isinstance(recomputed, dict)
        or recomputed.get("release_status") != "VALID"
    ):
        raise ValueError(
            f"control stage {stage} did not recompute a valid release"
        )
    if (
        expected_adapter_id is not None
        and recomputed.get("adapter_bundle_id") != expected_adapter_id
    ):
        raise ValueError(
            f"control stage {stage} recomputed the wrong adapter identity"
        )


def _validate_package_result(result: dict, run_id: str) -> None:
    if result.get("run_id") != run_id or result.get("packaged") is not True:
        raise ValueError("package stage did not package the expected run")


def run_control_plane(
    plan: dict,
    state_path: Path,
    *,
    resume: bool,
    stage_api: dict[str, object],
    image_source_id: str,
    source_dir: str,
) -> dict:
    """Run the canonical Modal stage order with a durable local checkpoint.

    ``stage_api``, ``image_source_id``, and ``source_dir`` are the
    deployment-specific bindings the runner facade supplies: the callable per
    stage, the validated image source-manifest ID, and the staged source tree
    whose producer files are hashed into every run spec.
    """
    from scripts.stopdff_v5.identity import compute_id, sha256_file
    from scripts.stopdff_v5.manifests import (
        ENVIRONMENT_PACKAGES,
        environment_contract_identity,
        run_spec_identity,
    )

    plan = _validate_control_plan(plan)
    if plan["source_id"] != image_source_id:
        raise ValueError(
            "control plan source_id does not match the validated Modal image source"
        )
    state_path = Path(state_path)
    digest = _control_plan_digest(plan)
    api = stage_api
    if resume:
        state = _load_control_json(state_path)
        if state.get("schema_version") != 4:
            raise ValueError("unsupported control-state schema")
        _reconcile_control_journal(state_path, state)
        if state.get("plan_digest") != digest or state.get("plan") != plan:
            raise ValueError("resume control plan does not match durable state")
        _close_interrupted_control_attempt(state_path, state)
        if (
            state.get("status") not in {"completed", "recovery_required"}
            and "validate_package" in state.get("completed", {})
        ):
            _refresh_control_stage(
                state_path,
                state,
                stage="validate_package",
                reason="nonterminal resume must re-read packaged bytes",
            )
        if state.get("status") in {"completed", "recovery_required"}:
            stored_result = state.get("result")
            validator = api.get("validate") if isinstance(api, dict) else None
            if (
                not isinstance(stored_result, dict)
                or not _is_final_control_run_id(stored_result.get("run_id"))
                or not _is_control_sha(stored_result.get("adapter_id"))
                or not callable(validator)
            ):
                state["status"] = "recovery_required"
                state["last_error"] = {
                    "stage": "completed_resume_revalidation",
                    "type": "RecoveryRequired",
                    "message": "completed state cannot re-prove the final package",
                }
                _record_control_event(
                    state_path,
                    state,
                    event="control_recovery_required",
                    stage="validate_package",
                    detail=state["last_error"],
                )
                return state
            try:
                current_validation = validator(
                    stored_result["run_id"],
                    stored_result["adapter_id"],
                    True,
                    True,
                )
                _validate_control_stage_result(
                    "completed_resume_revalidation",
                    current_validation,
                    lambda result: _validate_checker_result(
                        "completed_resume_revalidation",
                        result,
                        expected_adapter_id=stored_result["adapter_id"],
                    ),
                )
            except BaseException as exc:
                state["status"] = "recovery_required"
                state["last_error"] = {
                    "stage": "completed_resume_revalidation",
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
                _record_control_event(
                    state_path,
                    state,
                    event="control_recovery_required",
                    stage="validate_package",
                    detail=state["last_error"],
                )
                return state
            stored_result["validation"] = current_validation
            state["status"] = "completed"
            state.pop("last_error", None)
            _record_control_event(
                state_path,
                state,
                event="control_revalidated",
                stage="validate_package",
                detail={"run_id": stored_result["run_id"]},
            )
            return state
    else:
        journal_path = state_path.with_name(state_path.name + ".jsonl")
        if (
            state_path.exists()
            or state_path.is_symlink()
            or journal_path.exists()
            or journal_path.is_symlink()
        ):
            raise FileExistsError("fresh control state or journal already exists")
        state = {
            "schema_version": 4,
            "plan": plan,
            "plan_digest": digest,
            "status": "initialized",
            "sequence": 0,
            "stage_attempts": {},
            "completed": {},
        }
        _record_control_event(
            state_path,
            state,
            event="control_initialized",
        )

    required_api = {
        "probe",
        "verify_volume_artifact",
        "freeze_model",
        "adapter_determinism_receipt",
        "promote_adapter",
        "fvi_study",
        "bootstrap_plan",
        "run_sweep",
        "mutation_gate",
        "validate",
        "package",
    }
    if set(api) != required_api or not all(callable(api[name]) for name in api):
        raise ValueError("control stage API does not match the canonical stage set")

    source_id = plan["source_id"]
    raw_id = plan["raw_id"]
    _run_control_stage(
        state_path,
        state,
        name="verify_source",
        invoke=lambda _: api["verify_volume_artifact"](
            f"inputs/source_{source_id}",
            "source",
        ),
        validate_result=lambda result: _validate_verified_artifact_result(
            "verify_source",
            result,
            expected_id=source_id,
            require_myopic=False,
        ),
    )
    raw_check = _run_control_stage(
        state_path,
        state,
        name="verify_raw",
        invoke=lambda _: api["verify_volume_artifact"](
            f"inputs/raw_{raw_id}",
            "raw",
        ),
        validate_result=lambda result: _validate_verified_artifact_result(
            "verify_raw",
            result,
            expected_id=raw_id,
            require_myopic=True,
        ),
    )
    myopic_sha256 = raw_check["myopic_artifact_sha256"]

    resumed_probe_result = None
    if resume and "environment_probe" in state["completed"]:
        current_probe = api["probe"]()
        _validate_control_stage_result(
            "environment_probe",
            current_probe,
            lambda result: _validate_probe_result(
                result,
                ENVIRONMENT_PACKAGES,
            ),
        )
        current_environment_id = compute_id(
            environment_contract_identity(
                python_version=current_probe["python"],
                package_versions=current_probe["package_versions"],
            )
        )
        cached_probe = state["completed"]["environment_probe"]
        refresh_reason = None
        try:
            _validate_control_stage_result(
                "environment_probe",
                cached_probe,
                lambda result: _validate_probe_result(
                    result,
                    ENVIRONMENT_PACKAGES,
                ),
            )
            cached_environment_id = compute_id(
                environment_contract_identity(
                    python_version=cached_probe["python"],
                    package_versions=cached_probe["package_versions"],
                )
            )
        except Exception as exc:
            refresh_reason = (
                "cached environment probe is invalid: "
                f"{type(exc).__name__}: {exc}"
            )
        else:
            if cached_environment_id != current_environment_id:
                refresh_reason = (
                    "nonterminal resume observed a different Modal "
                    f"environment contract: {cached_environment_id} -> "
                    f"{current_environment_id}"
                )
        if refresh_reason is not None:
            _refresh_control_stage(
                state_path,
                state,
                stage="environment_probe",
                reason=refresh_reason,
            )
            # The live probe is a read-only resume preflight. Reuse its already
            # validated payload when checkpointing the refreshed stage so one
            # resume performs exactly one remote environment probe.
            resumed_probe_result = current_probe

    probe_result = _run_control_stage(
        state_path,
        state,
        name="environment_probe",
        invoke=lambda _: (
            resumed_probe_result
            if resumed_probe_result is not None
            else api["probe"]()
        ),
        validate_result=lambda result: _validate_probe_result(
            result,
            ENVIRONMENT_PACKAGES,
        ),
    )
    environment_identity = environment_contract_identity(
        python_version=probe_result["python"],
        package_versions=probe_result["package_versions"],
    )
    environment_contract_id = compute_id(environment_identity)
    producer_hashes = {
        name: sha256_file(Path(source_dir) / "scripts" / "stopdff_v5" / name)
        for name in ("checker.py", "sweep.py")
    }

    model_result = _run_control_stage(
        state_path,
        state,
        name="freeze_model",
        invoke=lambda _: api["freeze_model"](),
        validate_result=_validate_model_result,
    )
    model_id = model_result["model_id"]

    def invoke_adapter_determinism(attempt: int) -> dict:
        attempt_first, attempt_second = _adapter_attempt_subdirs(
            plan["adapter_subdirs"],
            attempt,
        )
        return api["adapter_determinism_receipt"](
            attempt_first,
            attempt_second,
            source_id,
            raw_id,
            model_id,
            bool(
                plan["gate_overrides"].get(
                    "allow_low_mc_retention",
                    False,
                )
            ),
        )

    determinism = _run_control_stage(
        state_path,
        state,
        name="adapter_determinism",
        invoke=invoke_adapter_determinism,
        validate_result=lambda result: _validate_determinism_result(
            result,
            source_id,
        ),
    )
    first_subdir, second_subdir = _adapter_attempt_subdirs(
        plan["adapter_subdirs"],
        state["stage_attempts"]["adapter_determinism"],
    )
    adapter_id = determinism["adapter_id"]
    determinism_receipt_id = determinism["prerequisite_receipt_id"]
    determinism_bindings = {
        "source_manifest_id": source_id,
        "raw_input_bundle_id": raw_id,
        "model_snapshot_id": model_id,
        "adapter_bundle_id": adapter_id,
    }

    _run_control_stage(
        state_path,
        state,
        name="promote_adapter",
        invoke=lambda _: api["promote_adapter"](first_subdir, adapter_id),
        validate_result=lambda result: _validate_promotion_result(
            result,
            adapter_id,
        ),
    )
    fvi_result = _run_control_stage(
        state_path,
        state,
        name="fvi_study",
        invoke=lambda _: api["fvi_study"](adapter_id),
        validate_result=_validate_fvi_result,
    )
    fvi_id = fvi_result["fvi_study_id"]
    selected = fvi_result["selected"]

    common_bindings = {
        **determinism_bindings,
        "fvi_study_id": fvi_id,
        "environment_contract_id": environment_contract_id,
    }
    smoke_bootstrap = _run_control_stage(
        state_path,
        state,
        name="smoke_bootstrap",
        invoke=lambda _: api["bootstrap_plan"](adapter_id, 100),
        validate_result=lambda result: _validate_bootstrap_result(
            "smoke_bootstrap",
            result,
            100,
        ),
    )
    smoke_bootstrap_id = smoke_bootstrap["bootstrap_plan_id"]
    smoke_spec = run_spec_identity(
        source_manifest_id=source_id,
        raw_input_bundle_id=raw_id,
        model_snapshot_id=model_id,
        adapter_bundle_id=adapter_id,
        fvi_study_id=fvi_id,
        bootstrap_plan_id=smoke_bootstrap_id,
        environment_contract_id=environment_contract_id,
        resource_summary_id=plan["resource_summary_id"],
        fvi_selected=selected,
        replicate_count=100,
        profile_variant="smoke",
        myopic_artifact_sha256=myopic_sha256,
        producer_hashes=producer_hashes,
        prerequisite_receipts={},
        gate_overrides=plan["gate_overrides"],
    )
    smoke_spec_id = compute_id(smoke_spec)
    smoke_run_id = f"smoke_modal_{smoke_spec_id[:12]}"

    def invoke_smoke(attempt: int) -> dict:
        wrapper = {
            "run_id": smoke_run_id,
            "run_spec_id": smoke_spec_id,
            "run_spec_identity": smoke_spec,
            "resource_summary": plan["resource_summary"],
        }
        return api["run_sweep"](
            json.dumps(wrapper, sort_keys=True),
            adapter_id,
            smoke_bootstrap_id,
            attempt > 1,
        )

    smoke_result = _run_control_stage(
        state_path,
        state,
        name="smoke_sweep",
        invoke=invoke_smoke,
        validate_result=lambda result: _validate_sweep_result(
            "smoke_sweep",
            result,
            run_id=smoke_run_id,
            require_receipt=True,
        ),
    )
    smoke_receipt_id = smoke_result["prerequisite_receipt_id"]

    mutation = _run_control_stage(
        state_path,
        state,
        name="mutation_gate",
        invoke=lambda _: api["mutation_gate"](
            json.dumps(common_bindings, sort_keys=True)
        ),
        validate_result=lambda result: _validate_mutation_result(
            result,
            source_id,
        ),
    )
    mutation_receipt_id = mutation["prerequisite_receipt_id"]

    final_bootstrap = _run_control_stage(
        state_path,
        state,
        name="final_bootstrap",
        invoke=lambda _: api["bootstrap_plan"](adapter_id, 1000),
        validate_result=lambda result: _validate_bootstrap_result(
            "final_bootstrap",
            result,
            1000,
        ),
    )
    final_bootstrap_id = final_bootstrap["bootstrap_plan_id"]
    receipt_ids = {
        "determinism": determinism_receipt_id,
        "mutation": mutation_receipt_id,
        "smoke": smoke_receipt_id,
    }
    final_spec = run_spec_identity(
        source_manifest_id=source_id,
        raw_input_bundle_id=raw_id,
        model_snapshot_id=model_id,
        adapter_bundle_id=adapter_id,
        fvi_study_id=fvi_id,
        bootstrap_plan_id=final_bootstrap_id,
        environment_contract_id=environment_contract_id,
        resource_summary_id=plan["resource_summary_id"],
        fvi_selected=selected,
        replicate_count=1000,
        profile_variant="final",
        myopic_artifact_sha256=myopic_sha256,
        producer_hashes=producer_hashes,
        prerequisite_receipts=receipt_ids,
        gate_overrides=plan["gate_overrides"],
    )
    final_spec_id = compute_id(final_spec)
    final_run_id = f"final_modal_{final_spec_id[:12]}"

    def invoke_final(attempt: int) -> dict:
        wrapper = {
            "run_id": final_run_id,
            "run_spec_id": final_spec_id,
            "run_spec_identity": final_spec,
            "resource_summary": plan["resource_summary"],
        }
        return api["run_sweep"](
            json.dumps(wrapper, sort_keys=True),
            adapter_id,
            final_bootstrap_id,
            attempt > 1,
        )

    _run_control_stage(
        state_path,
        state,
        name="final_sweep",
        invoke=invoke_final,
        validate_result=lambda result: _validate_sweep_result(
            "final_sweep",
            result,
            run_id=final_run_id,
            require_receipt=False,
        ),
    )
    # package() performs its own fail-closed unpacked validation.  A separate
    # controller validation here would repeat the complete 96-cell computation
    # without adding a trust boundary; keep the independent packaged validation
    # below, after publication has changed the evidence surface.
    _run_control_stage(
        state_path,
        state,
        name="package",
        invoke=lambda _: api["package"](final_run_id),
        validate_result=lambda result: _validate_package_result(
            result,
            final_run_id,
        ),
    )
    final_validation = _run_control_stage(
        state_path,
        state,
        name="validate_package",
        invoke=lambda _: api["validate"](
            final_run_id,
            adapter_id,
            True,
            True,
        ),
        validate_result=lambda result: _validate_checker_result(
            "validate_package",
            result,
            expected_adapter_id=adapter_id,
        ),
    )
    state["status"] = "completed"
    state["result"] = {
        "run_id": final_run_id,
        "run_spec_id": final_spec_id,
        "adapter_id": adapter_id,
        "receipt_ids": receipt_ids,
        "validation": final_validation,
    }
    _record_control_event(
        state_path,
        state,
        event="control_completed",
        detail={"run_id": final_run_id, "run_spec_id": final_spec_id},
    )
    return state
