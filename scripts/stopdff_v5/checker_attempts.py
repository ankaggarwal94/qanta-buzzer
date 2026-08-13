"""Attempt-history, package-namespace, and report checks (ACCEPTANCE_CONTRACT.md).

Package-structure lanes of the standalone checker: the append-only attempt
history and its terminal results, the DISALLOW-unknown package path policy,
and byte-exact regeneration of every displayed report/figure. Extracted
verbatim from ``checker``, which re-exports the historical names.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from .attempt_history import load_attempt_history
from .checker_common import _INTERRUPTED_REASON, _is_strict_int, load_json
from .checker_package import _err
from .checker_png import _check_png

def _check_attempts(
    run_root: Path,
    errors: list[str],
    *,
    run_spec_id: str,
    adapter_bundle_id: str,
    bootstrap_plan_id: str,
    aggregate: dict[str, Any],
) -> bool:
    error_count = len(errors)
    path = run_root / "attempts.jsonl"
    try:
        _, attempts = load_attempt_history(path)
    except (OSError, UnicodeError, TypeError, ValueError) as exc:
        errors.append(f"attempts.jsonl is noncanonical: {exc}")
        return False
    if not attempts:
        errors.append("attempts.jsonl contains no attempts")
        return False
    last_num = 0
    attempt_numbers: list[int] = []
    for index, a in enumerate(attempts):
        if not isinstance(a, dict):
            errors.append("attempt record is not an object")
            continue
        raw_num = a.get("attempt")
        if not _is_strict_int(raw_num, minimum=1):
            errors.append("attempt number must be a positive integer")
            continue
        num = raw_num
        attempt_numbers.append(num)
        mode = a.get("mode")
        cmd = a.get("command", [])
        if not isinstance(mode, str) or mode not in {"fresh", "resume"}:
            errors.append(f"unknown attempt mode {mode!r}")
        if a.get("state") != "started":
            errors.append("attempt record state must be 'started'")
        if (
            not isinstance(cmd, list)
            or not all(isinstance(part, str) for part in cmd)
        ):
            errors.append("attempt command must be a string list")
            cmd = []
        if num != index + 1:
            errors.append("attempt numbers must be consecutive starting at 1")
        elif num <= last_num:
            errors.append("attempt numbers not monotonic")
        last_num = num
        if "--overwrite" in cmd:
            errors.append("--overwrite present in an evidence attempt")
        resume_count = list(cmd).count("--resume")
        if index == 0:
            if num != 1 or mode != "fresh":
                errors.append("first attempt must be attempt 1 in fresh mode")
            if resume_count != 0:
                errors.append("fresh attempt must omit --resume")
        elif mode == "resume":
            if resume_count != 1:
                errors.append("resume attempt must contain exactly one bare --resume")
        elif mode == "fresh":
            errors.append("only the first attempt may use fresh mode")
        _err(
            errors,
            a.get("run_spec_id") == run_spec_id,
            "attempt run_spec_id does not match run spec",
        )
        _err(
            errors,
            a.get("adapter_id") == adapter_bundle_id,
            "attempt adapter_id does not match run spec",
        )
        _err(
            errors,
            a.get("bootstrap_plan_id") == bootstrap_plan_id,
            "attempt bootstrap_plan_id does not match run spec",
        )

    results_dir = run_root / "attempt_results"
    if results_dir.is_symlink() or not results_dir.is_dir():
        errors.append("missing attempt_results directory")
        return False
    result_paths = sorted(results_dir.iterdir())
    result_numbers: set[int] = set()
    results: dict[int, dict[str, Any]] = {}
    for result_path in result_paths:
        try:
            result_number = int(result_path.stem)
        except ValueError:
            errors.append(f"invalid attempt result filename {result_path.name!r}")
            continue
        if (
            result_number < 1
            or result_path.name != f"{result_number}.json"
            or result_number in result_numbers
            or result_path.is_symlink()
            or not result_path.is_file()
        ):
            errors.append(f"invalid attempt result path {result_path.name!r}")
            continue
        try:
            result = load_json(result_path)
        except (
            OSError,
            UnicodeError,
            json.JSONDecodeError,
            TypeError,
            ValueError,
        ) as exc:
            errors.append(
                f"attempt result {result_path.name!r} cannot be decoded: {exc}"
            )
            continue
        if not isinstance(result, dict):
            errors.append(f"attempt result {result_path.name!r} is not an object")
            continue
        result_numbers.add(result_number)
        results[result_number] = result
        _err(
            errors,
            result.get("attempt") == result_number,
            f"attempt result {result_number} number mismatch",
        )
        _err(
            errors,
            result.get("run_spec_id") == run_spec_id,
            f"attempt result {result_number} run_spec_id mismatch",
        )
        state = result.get("state")
        state_fields = {
            "completed": {"completed", "failed"},
            "failed": {"error_type", "error_message"},
            "interrupted": {"reason"},
        }
        valid_state = isinstance(state, str) and state in state_fields
        _err(
            errors,
            valid_state,
            f"attempt result {result_number} has invalid state",
        )
        if valid_state:
            _err(
                errors,
                set(result)
                == {"attempt", "state", "run_spec_id"} | state_fields[state],
                f"attempt result {result_number} fields do not match its state",
            )
        if state == "completed":
            _err(
                errors,
                _is_strict_int(result.get("completed"), minimum=0)
                and _is_strict_int(result.get("failed"), minimum=0),
                f"attempt result {result_number} has invalid counts",
            )
        elif state == "failed":
            _err(
                errors,
                isinstance(result.get("error_type"), str)
                and bool(result.get("error_type"))
                and isinstance(result.get("error_message"), str),
                f"attempt result {result_number} has invalid failure evidence",
            )
        elif state == "interrupted":
            _err(
                errors,
                result.get("reason") == _INTERRUPTED_REASON,
                f"attempt result {result_number} has invalid interruption evidence",
            )
    _err(
        errors,
        result_numbers == set(attempt_numbers),
        "attempt results do not match attempt history",
    )
    if attempt_numbers and attempt_numbers[-1] in results:
        final_result = results[attempt_numbers[-1]]
        _err(
            errors,
            final_result.get("state") == "completed",
            "latest attempt did not complete",
        )
        _err(
            errors,
            final_result.get("completed") == aggregate.get("completed")
            and final_result.get("failed") == aggregate.get("failed"),
            "latest attempt counts do not match aggregate",
        )
    return len(errors) == error_count


def _check_package_path_policy(run_root: Path, errors: list[str]) -> None:
    """Reject package entries no validation lane audits (DISALLOW-unknown).

    ``check_complete_checksums`` proves SHA256SUMS↔inventory bijection but not
    that every inventoried path is one the checker actually audits: an
    orphaned atomic-write temp file present at package time is hashed into
    SHA256SUMS and stays self-consistent forever. Enforce the same explicit
    namespace the packager enforces (writers path-policy constants), plus the
    two package-level files and the three package-managed roots whose
    contents are audited by the checksum bijection, report regeneration, and
    evidence lanes. Symlinks and special entries are already rejected by
    ``check_complete_checksums``.
    """
    from .writers import (
        PACKAGE_LEVEL_FILES,
        PACKAGE_MANAGED_ROOTS,
        RUN_JSON_ONLY_DIRS,
        RUN_LEVEL_FILES,
    )

    audited_files = RUN_LEVEL_FILES | PACKAGE_LEVEL_FILES
    audited_dirs = RUN_JSON_ONLY_DIRS | PACKAGE_MANAGED_ROOTS
    for path in sorted(run_root.iterdir()):
        name = path.name
        if path.is_symlink():
            continue
        if path.is_dir():
            if name not in audited_dirs:
                errors.append(f"unaudited package directory: {name!r}")
        elif path.is_file():
            if name not in audited_files:
                errors.append(f"unaudited package file: {name!r}")
    for dir_name in sorted(RUN_JSON_ONLY_DIRS):
        directory = run_root / dir_name
        if directory.is_symlink() or not directory.is_dir():
            continue
        for path in sorted(directory.iterdir()):
            if (
                not path.is_symlink()
                and path.is_file()
                and path.suffix == ".json"
            ):
                continue
            errors.append(
                f"unaudited entry in {dir_name}/: {path.name!r}"
            )


def _check_reports(
    run_root: Path,
    aggregate: dict[str, Any],
    resource_summary: dict[str, Any],
    errors: list[str],
) -> None:
    """Bind every displayed package byte to the validated scientific inputs."""
    from . import writers

    try:
        expected: dict[str, bytes] = {
            "reports/report.md": writers.render_markdown(
                aggregate,
                resource_summary=resource_summary,
            ).encode("utf-8"),
            "reports/report.tex": writers.render_latex(aggregate).encode(
                "utf-8"
            ),
        }
        with tempfile.TemporaryDirectory(prefix="stopdff_v5_check_figures_") as td:
            figure_root = Path(td)
            figure_paths = writers.write_figures(
                figure_root,
                aggregate,
                profile_variant=aggregate.get("profile_variant"),
            )
            for relative in figure_paths:
                expected[relative] = (figure_root / relative).read_bytes()
    except (
        AttributeError,
        KeyError,
        OSError,
        OverflowError,
        RecursionError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        errors.append(
            "canonical reports/figures cannot be regenerated: "
            f"{type(exc).__name__}: {exc}"
        )
        return

    for directory_name in ("reports", "figures"):
        directory = run_root / directory_name
        expected_names = {
            Path(relative).name
            for relative in expected
            if Path(relative).parts[0] == directory_name
        }
        if directory.is_symlink() or not directory.is_dir():
            errors.append(f"missing or noncanonical {directory_name} directory")
            continue
        actual_names: set[str] = set()
        try:
            entries = list(directory.iterdir())
        except OSError as exc:
            errors.append(f"{directory_name} directory cannot be read: {exc}")
            continue
        for path in entries:
            if path.is_symlink() or not path.is_file():
                errors.append(
                    f"unexpected non-file package evidence: "
                    f"{directory_name}/{path.name}"
                )
                continue
            actual_names.add(path.name)
        for name in sorted(expected_names - actual_names):
            errors.append(f"missing {directory_name}/{name}")
        for name in sorted(actual_names - expected_names):
            errors.append(f"unexpected {directory_name}/{name}")

    for relative, expected_bytes in sorted(expected.items()):
        path = run_root / relative
        if path.is_symlink() or not path.is_file():
            continue
        try:
            actual_size = path.stat().st_size
        except OSError as exc:
            errors.append(f"{relative} cannot be inspected: {exc}")
            continue
        if actual_size != len(expected_bytes):
            errors.append(f"{relative} does not match canonical rendered content")
            continue
        try:
            actual_bytes = path.read_bytes()
        except OSError as exc:
            errors.append(f"{relative} cannot be read: {exc}")
            continue
        if actual_bytes != expected_bytes:
            errors.append(f"{relative} does not match canonical rendered content")
        if path.suffix == ".png":
            _check_png(path, errors)
