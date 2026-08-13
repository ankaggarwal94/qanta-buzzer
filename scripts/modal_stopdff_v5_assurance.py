#!/usr/bin/env python3
"""Cross-process host driver for the bounded Modal recovery assurance canary."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import modal

_REPO = Path(__file__).resolve().parents[1]
_REPO_IMPORT_ROOT = str(_REPO)
# ``scripts`` is excluded from packaging (pyproject) and the documented form is
# ``python scripts/modal_stopdff_v5_assurance.py <subcmd>`` (REPRODUCTION.md),
# so the script's own dir — not the repo root — is on sys.path[0]. Bootstrap
# the repo root before importing ``scripts.*`` below. Membership is not
# precedence: make this entrypoint's checkout the authoritative import root so
# a second checkout cannot shadow the evidentiary producer code loaded here.
sys.path[:] = [entry for entry in sys.path if entry != _REPO_IMPORT_ROOT]
sys.path.insert(0, _REPO_IMPORT_ROOT)

from scripts.stopdff_v5.fileio import (  # noqa: E402
    create_once_bytes,
    dumps_json_bytes,
)
from scripts.stopdff_v5.identity import loads_no_duplicate_keys  # noqa: E402


def _require_receipt_absent(path: Path) -> None:
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"assurance receipt already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)


def _write_once(path: Path, value: object) -> None:
    data = dumps_json_bytes(value)
    create_once_bytes(Path(path), data, exists_label="assurance receipt")


def _load_object(path: Path) -> dict:
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"assurance receipt must be a regular file: {path}")
    data = path.read_bytes()
    try:
        value = loads_no_duplicate_keys(data.decode("utf-8"))
        canonical = dumps_json_bytes(value)
    except (UnicodeDecodeError, ValueError, TypeError) as exc:
        raise ValueError(f"assurance receipt is invalid JSON: {path}") from exc
    if not isinstance(value, dict) or data != canonical:
        raise ValueError(
            f"assurance receipt is not a canonical object: {path}"
        )
    return value


def _validated_submission(path: Path) -> dict:
    value = _load_object(path)
    expected = {
        "schema_version",
        "deployment",
        "tag",
        "phase",
        "function_call_id",
    }
    if (
        set(value) != expected
        or type(value.get("schema_version")) is not int
        or value.get("schema_version") != 1
    ):
        raise ValueError("submission receipt does not match the exact schema")
    for field in ("deployment", "tag", "function_call_id"):
        if not isinstance(value.get(field), str) or not value[field]:
            raise ValueError(f"submission receipt lacks a nonempty {field}")
    if value.get("phase") != "crash":
        raise ValueError("submission receipt does not describe the crash phase")
    return value


def _function(deployment: str):
    return modal.Function.from_name(deployment, "recovery_assurance")


def _validated_timeout(value: float) -> float:
    if not math.isfinite(value) or not 0.0 < value <= 300.0:
        raise ValueError("timeout-seconds must be finite and in (0, 300]")
    return value


def _required_call_id(call) -> str:
    call_id = getattr(call, "object_id", None)
    if not isinstance(call_id, str) or not call_id:
        raise RuntimeError("Modal did not return a nonempty FunctionCall ID")
    return call_id


def _bounded_phase_call(
    deployment: str,
    tag: str,
    phase: str,
    *,
    timeout_seconds: float,
) -> tuple[str, object]:
    timeout_seconds = _validated_timeout(timeout_seconds)
    call = _function(deployment).spawn(tag, phase)
    call_id = _required_call_id(call)
    try:
        result = call.get(timeout=timeout_seconds)
    except TimeoutError:
        call.cancel(terminate_containers=True)
        raise
    return call_id, result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    submit = subparsers.add_parser("submit")
    submit.add_argument("--deployment", required=True)
    submit.add_argument("--tag", required=True)
    submit.add_argument("--receipt", type=Path, required=True)

    recover = subparsers.add_parser("recover")
    recover.add_argument("--call-receipt", type=Path, required=True)
    recover.add_argument("--timeout-seconds", type=float, default=300.0)
    recover.add_argument("--receipt", type=Path, required=True)

    for phase in ("classify", "finish", "verify"):
        command = subparsers.add_parser(phase)
        command.add_argument("--deployment", required=True)
        command.add_argument("--tag", required=True)
        command.add_argument("--timeout-seconds", type=float, default=300.0)
        command.add_argument("--receipt", type=Path, required=True)

    args = parser.parse_args(argv)
    if args.command == "submit":
        _require_receipt_absent(args.receipt)
        call = _function(args.deployment).spawn(args.tag, "crash")
        call_id = _required_call_id(call)
        result = {
            "schema_version": 1,
            "deployment": args.deployment,
            "tag": args.tag,
            "phase": "crash",
            "function_call_id": call_id,
        }
        _write_once(args.receipt, result)
    elif args.command == "recover":
        _require_receipt_absent(args.receipt)
        submitted = _validated_submission(args.call_receipt)
        call_id = submitted["function_call_id"]
        timeout_seconds = _validated_timeout(args.timeout_seconds)
        call = modal.FunctionCall.from_id(call_id)
        try:
            remote_result = call.get(timeout=timeout_seconds)
        except TimeoutError:
            call.cancel(terminate_containers=True)
            raise
        result = {
            "schema_version": 1,
            "function_call_id": call_id,
            "result": remote_result,
        }
        _write_once(args.receipt, result)
    else:
        _require_receipt_absent(args.receipt)
        call_id, remote_result = _bounded_phase_call(
            args.deployment,
            args.tag,
            args.command,
            timeout_seconds=args.timeout_seconds,
        )
        result = {
            "schema_version": 1,
            "deployment": args.deployment,
            "tag": args.tag,
            "phase": args.command,
            "function_call_id": call_id,
            "result": remote_result,
        }
        _write_once(args.receipt, result)

    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
