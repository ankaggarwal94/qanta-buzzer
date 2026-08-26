"""Run one pytest suite and emit its closed R-070 evidence receipt."""
from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from reproducibility.colm_aims_2026 import phase4_launcher, receipt, schema
from scripts.stopdff_v5 import fileio


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _junit_counts(path: Path) -> tuple[dict[str, int], list[str]]:
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    counts = {
        key: sum(int(suite.attrib.get(key, "0")) for suite in suites)
        for key in ("tests", "failures", "errors", "skipped")
    }
    skipped = []
    for case in root.iter("testcase"):
        if case.find("skipped") is not None:
            skipped.append(
                f"{case.attrib.get('classname', '')}::{case.attrib.get('name', '')}"
            )
    return counts, sorted(skipped)


def _pytest_command(command: list[str], junit_path: Path) -> list[str]:
    """Return the exact argv recorded in an R-070 receipt."""
    if command[:1] != ["pytest"]:
        raise ValueError("R-070 suite command must start with pytest")
    return [
        sys.executable,
        "-m",
        "pytest",
        *command[1:],
        f"--junitxml={junit_path.resolve()}",
    ]


def _environment_lock_bytes() -> bytes:
    """Use the exact environment-lock definition consumed by Phase 4."""
    return phase4_launcher._default_probe_environment_lock(
        Path(sys.executable).resolve()
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--workflow", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = list(args.command)
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        parser.error("a pytest command is required after --")
    if not schema.is_path_component(args.name):
        parser.error("--name must be one safe path component")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    junit_path = output_dir / f"{args.name}.junit.xml"
    transcript_path = output_dir / f"{args.name}.transcript.txt"
    environment_path = output_dir / f"{args.name}.environment.txt"
    receipt_path = output_dir / f"{args.name}.receipt.json"
    for path in (junit_path, transcript_path, environment_path, receipt_path):
        if path.exists():
            parser.error(f"refusing to overwrite existing evidence path: {path}")

    try:
        actual_command = _pytest_command(command, junit_path)
    except ValueError as exc:
        parser.error(str(exc))
    workflow_bytes_before = args.workflow.read_bytes()
    commit_before = _git("rev-parse", "HEAD")
    tree_before = _git("rev-parse", "HEAD^{tree}")
    dirty_before = bool(
        _git("status", "--porcelain", "--untracked-files=no")
    )
    environment = _environment_lock_bytes()
    fileio.create_once_bytes(
        environment_path, environment, exists_label="suite environment export"
    )
    completed = subprocess.run(
        actual_command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    fileio.create_once_bytes(
        transcript_path, completed.stdout, exists_label="suite transcript"
    )
    sys.stdout.buffer.write(completed.stdout)

    junit_valid = junit_path.is_file()
    if junit_valid:
        try:
            counts, skipped = _junit_counts(junit_path)
            junit_bytes = junit_path.read_bytes()
        except (ET.ParseError, OSError, ValueError):
            junit_valid = False
    if not junit_valid:
        counts = {key: 0 for key in ("tests", "failures", "errors", "skipped")}
        skipped = []
        junit_bytes = b""
    transcript_bytes = transcript_path.read_bytes()
    environment_bytes = environment_path.read_bytes()
    commit_after = _git("rev-parse", "HEAD")
    tree_after = _git("rev-parse", "HEAD^{tree}")
    workflow_bytes_after = args.workflow.read_bytes()
    dirty_after = bool(_git("status", "--porcelain", "--untracked-files=no"))
    head_drift = (
        commit_after != commit_before
        or tree_after != tree_before
        or workflow_bytes_after != workflow_bytes_before
    )
    payload = {
        "schema_version": schema.SCHEMA_VERSION,
        "environment_lock_sha256": _sha256(environment_bytes),
        "workflow_sha256": _sha256(workflow_bytes_before),
        "interpreter_realpath": str(Path(sys.executable).resolve()),
        "commit": commit_before,
        "tree_sha256": tree_before,
        "dirty": dirty_before or dirty_after or head_drift,
        "command": actual_command,
        "exit_code": completed.returncode,
        "junit_sha256": _sha256(junit_bytes),
        "transcript_sha256": _sha256(transcript_bytes),
        "counts": counts,
        "skip_identities": skipped,
    }
    fileio.create_once_bytes(
        receipt_path, schema.encode_json(payload), exists_label="suite receipt"
    )
    evidence_ok = junit_valid
    if completed.returncode == 0 and evidence_ok:
        receipt.validate_suite_receipt(payload)
    if completed.returncode != 0:
        return completed.returncode
    return 0 if evidence_ok else 4


if __name__ == "__main__":
    raise SystemExit(main())
