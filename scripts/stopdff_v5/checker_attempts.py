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

from . import writers
from .attempt_history import load_attempt_history
from .checker_common import _INTERRUPTED_REASON, _err, _is_strict_int, load_json
from .checker_png import _check_png
from .writers import (
    BOUND_CONTENT_LAYOUTS,
    MANIFEST_EVIDENCE_PATHS,
    PACKAGE_LEVEL_FILES,
    PACKAGE_MANAGED_ROOTS,
    RECEIPT_GATES,
    RUN_JSON_ONLY_DIRS,
    RUN_LEVEL_FILES,
)

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
    two package-level files and the three package-managed roots.
    ``reports``/``figures`` contents are exact-membership checked by the
    report-regeneration lane; the ``evidence`` namespace is recursed here for
    exact membership (:func:`_check_evidence_namespace`) because the checksum
    bijection constrains hashes, never which paths may appear. Symlinks and
    special entries are already rejected by ``check_complete_checksums``.
    """
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
    _check_evidence_namespace(run_root, errors)


def _packaged_profile_variant(run_root: Path) -> Any:
    """Best-effort read of the packaged run spec's declared profile variant.

    The identity lanes validate ``run_spec.json`` (and bind its declared
    variant) independently, so tampering with the declaration already fails
    validation there; the value is only used here for namespace membership.
    An unreadable or malformed spec returns ``None``, which fails closed —
    the receipts directory is then treated as unaudited.
    """
    try:
        manifest = load_json(run_root / "run_spec.json")
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
        return None
    if not isinstance(manifest, dict):
        return None
    identity = manifest.get("identity")
    if not isinstance(identity, dict):
        return None
    return identity.get("profile_variant")


def _check_evidence_namespace(run_root: Path, errors: list[str]) -> None:
    """Recurse ``evidence/`` enforcing exact membership (DISALLOW-unknown).

    The checksum bijection constrains hashes, the external-artifact ledger
    audits the five manifest files at fixed paths, the bound-content lane
    exhaustively inventories the files under the three packaged content
    subtrees, and the receipt lane audits the canonical receipt files — but
    none of them enumerates ``evidence/`` itself, so an extra file (or an
    empty directory, which SHA256SUMS and the content inventories never see)
    would otherwise ride inside an accepted package. Membership is
    single-sourced from the packager's layout tables in ``writers``:

    - top level: the manifest evidence files, the three bound content roots,
      and (for the final profile only) ``prerequisite_receipts/``;
    - each bound content root: exactly its packaged content subdir, whose
      file inventory the bound-content lane audits exhaustively;
    - ``prerequisite_receipts/``: only canonical receipt file names;
    - everywhere: directories with no entries are rejected — no lane audits
      a bare directory name.
    """
    evidence_root = run_root / "evidence"
    if evidence_root.is_symlink() or not evidence_root.is_dir():
        return  # a missing evidence tree is the external-artifact lane's error

    audited_files = {
        Path(packaged_path).name
        for packaged_path in MANIFEST_EVIDENCE_PATHS.values()
    }
    bound_roots: dict[str, str] = {}
    for layout in BOUND_CONTENT_LAYOUTS.values():
        root_name, subdir_name = Path(layout["packaged_subdir"]).parts
        bound_roots[root_name] = subdir_name
    receipt_files = {
        f"{gate}{suffix}"
        for gate in RECEIPT_GATES
        for suffix in (".json", ".evidence.json")
    }
    receipts_expected = _packaged_profile_variant(run_root) == "final"

    def _rel(path: Path) -> str:
        return path.relative_to(run_root).as_posix()

    for path in sorted([evidence_root, *evidence_root.rglob("*")]):
        if path.is_symlink() or not path.is_dir():
            continue
        if not any(path.iterdir()):
            errors.append(
                f"unaudited empty package directory: {_rel(path)!r}"
            )

    for path in sorted(evidence_root.iterdir()):
        name = path.name
        if path.is_symlink():
            continue
        if path.is_dir():
            if name in bound_roots:
                _check_bound_content_root(
                    run_root, path, bound_roots[name], errors
                )
            elif name == "prerequisite_receipts" and receipts_expected:
                for entry in sorted(path.iterdir()):
                    if entry.is_symlink():
                        continue
                    if entry.is_dir():
                        errors.append(
                            f"unaudited package directory: {_rel(entry)!r}"
                        )
                    elif entry.is_file() and entry.name not in receipt_files:
                        errors.append(
                            f"unaudited package file: {_rel(entry)!r}"
                        )
            else:
                errors.append(f"unaudited package directory: {_rel(path)!r}")
        elif path.is_file():
            if name not in audited_files:
                errors.append(f"unaudited package file: {_rel(path)!r}")


def _check_bound_content_root(
    run_root: Path,
    content_root: Path,
    subdir_name: str,
    errors: list[str],
) -> None:
    """A bound content root may contain only its packaged content subdir."""
    for path in sorted(content_root.iterdir()):
        if path.is_symlink():
            continue
        if path.is_dir() and path.name == subdir_name:
            # The subtree's file inventory is audited exhaustively by the
            # bound-content manifest lane; entry-free directories inside it
            # are rejected by the namespace walk above.
            continue
        rel = path.relative_to(run_root).as_posix()
        if path.is_dir():
            errors.append(f"unaudited package directory: {rel!r}")
        elif path.is_file():
            errors.append(f"unaudited package file: {rel!r}")


def _check_reports(
    run_root: Path,
    aggregate: dict[str, Any],
    resource_summary: dict[str, Any],
    errors: list[str],
) -> None:
    """Bind every displayed package byte to the validated scientific inputs."""
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
