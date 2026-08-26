"""Generate Windows-host Phase-4 retry certificate #1.

This checked-in Windows-host orchestrator is source-only: it verifies the six
external inputs and two materialized snapshot trees, executes the focused and
full test suites at one clean committed HEAD, writes head-bound receipts, and
assembles a replacement certificate. It never constructs or runs a model.

Usage: ``python phase4_pre_run_ready_orchestration.py`` from the repo root.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from reproducibility.colm_aims_2026 import (  # noqa: E402
    phase4,
    phase4_launcher,
    schema,
)

ASSET_ROOT = Path.home() / "qanta_phase4_windows_v3"
RUN_ROOT = Path.home() / "qanta_phase4_windows_retry1"
STAGED_DATA = ASSET_ROOT / "staging" / "processed"
SNAPSHOT_ROOT = ASSET_ROOT / "snapshots"
RECEIPTS_DIR = RUN_ROOT / "receipts"
CERTIFICATE_DIR = RUN_ROOT / "certificate"
CERTIFICATE_PATH = (
    CERTIFICATE_DIR / "pre_run_ready_certificate_2026-08-23_retry1.json"
)
QUARANTINE_DIR = RUN_ROOT / "output" / "phase4_quarantine_retry1"
PROMOTE_TO = RUN_ROOT / "output" / "phase4_run_output_retry1"
EXCEPTION_LEDGER_PATH = (
    RUN_ROOT / "launch" / "phase4_retry1_single_use_ledger.json"
)

FROZEN = REPO / "reproducibility" / "colm_aims_2026" / "frozen"
ELIGIBILITY_PATH = FROZEN / "pairing_eligibility_v2.json"
SNAPSHOT_MANIFEST_PATH = FROZEN / "model_snapshot_manifests.json"
PARITY_ANCHOR_PATH = FROZEN / "parity_anchor_export_a.json"
QA012_REV3_PATH = REPO / "qa012_inventory_2026-08-22_rev3.json"
WORKFLOW_PATH = REPO / ".github" / "workflows" / "python-app.yml"

EXPECTED_STAGED = {
    "calibration_train": (
        "calibration_train.json",
        "745bd67597278bd9d24d41c1dea53bf3a7c56cd6334cfc07ea62bccbdcf44259",
    ),
    "eval_split": (
        "test_dataset.json",
        "638a4df978b77a12655ea72d56daad7fa70851ae486ddb4365d9b060549e34f1",
    ),
    "fit_split": (
        "val_dataset.json",
        "9b7a131b6c94c446e6b40b95559cb62aeee63f6e6f29ddd1d7ed3fb19cc72c65",
    ),
    "mc_dataset": (
        "mc_dataset.json",
        "3dbebf8e4d690da41a15e3cf467e57fdbe69af420ed831d56b61160af8bf7946",
    ),
    "answer_profiles": (
        "answer_profiles.json",
        "635586393ad36cf7e0726066bc242d97d0f982abd6108e4d8b87a3cf4598fc75",
    ),
    "build_metadata": (
        "build_metadata.json",
        "70871984390f252c0a06a5a2c9a2d3b4337f10ad48c87583ebec215d5c0c9c6e",
    ),
}

FOCUSED_TESTS = [
    "tests/test_colm_aims_v2_phase4_pre.py",
    "tests/test_colm_aims_v2_schema_raw_bytes.py",
    "tests/test_phase4_build_metadata_staging.py",
    "tests/test_phase4_certificate_external_staging.py",
    "tests/test_phase4_launcher_cli.py",
    "tests/test_stopdff_v5_fileio_windows.py",
    "tests/test_stopdff_v5_windows_control_plane.py",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def clean_head() -> dict[str, Any]:
    tracked = [
        line
        for line in git(
            "status", "--porcelain", "--untracked-files=no"
        ).splitlines()
        if line
    ]
    if tracked:
        raise RuntimeError(f"tracked tree is dirty: {tracked[:20]}")
    return {
        "commit": git("rev-parse", "HEAD").strip(),
        "tree": git("rev-parse", "HEAD^{tree}").strip(),
        "dirty": False,
    }


def require_fresh_operational_paths() -> None:
    if schema.resolves_inside(RUN_ROOT, REPO):
        raise RuntimeError("operational run root must live outside the repo")
    for directory in (
        RUN_ROOT,
        STAGED_DATA,
        SNAPSHOT_ROOT,
        RECEIPTS_DIR,
        CERTIFICATE_DIR,
    ):
        if not directory.is_dir():
            raise RuntimeError(f"required prepared directory is missing: {directory}")
    stale_receipts = list(RECEIPTS_DIR.iterdir())
    if stale_receipts:
        raise RuntimeError(
            f"receipt directory is not fresh: {stale_receipts[:10]}"
        )
    if CERTIFICATE_PATH.exists() or CERTIFICATE_PATH.is_symlink():
        raise RuntimeError(f"certificate path already exists: {CERTIFICATE_PATH}")
    for path in (QUARANTINE_DIR, PROMOTE_TO, EXCEPTION_LEDGER_PATH):
        if not path.parent.is_dir():
            raise RuntimeError(f"launch-path parent is missing: {path.parent}")
        if path.exists() or path.is_symlink():
            raise RuntimeError(f"launch path is not fresh: {path}")


def verify_prepared_inputs() -> list[dict[str, str]]:
    plan: list[dict[str, str]] = []
    for label, (filename, expected) in EXPECTED_STAGED.items():
        path = (STAGED_DATA / filename).resolve()
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"staged input is not a regular file: {path}")
        if schema.resolves_inside(path, REPO):
            raise RuntimeError(f"staged input resolves inside the repo: {path}")
        observed = sha256_file(path)
        if observed != expected:
            raise RuntimeError(
                f"{label} SHA-256 {observed} != expected {expected}"
            )
        plan.append(
            {
                "label": label,
                "path": str(path),
                "expected_sha256": expected,
            }
        )
    return plan


def verify_prepared_snapshots() -> dict[str, Path]:
    manifest = phase4.load_model_snapshot_manifest(SNAPSHOT_MANIFEST_PATH)
    snapshot_dirs = {
        "primary_scorer": (SNAPSHOT_ROOT / "primary_scorer").resolve(),
        "disjoint_selector": (
            SNAPSHOT_ROOT / "disjoint_selector"
        ).resolve(),
    }
    for role, directory in snapshot_dirs.items():
        if schema.resolves_inside(directory, REPO):
            raise RuntimeError(f"snapshot resolves inside the repo: {directory}")
        phase4.verify_snapshot_dir(manifest["roles"][role], directory)
    return snapshot_dirs


def suite_environment() -> dict[str, str]:
    return phase4_launcher._sanitized_runtime_environment()


def run_suite(name: str, pytest_args: list[str]) -> dict[str, Any]:
    junit_path = RECEIPTS_DIR / f"{name}_suite.xml"
    transcript_path = RECEIPTS_DIR / f"{name}_suite_transcript.txt"
    command = [
        str(Path(sys.executable).resolve()),
        "-m",
        "pytest",
        *pytest_args,
        "-q",
        "-p",
        "no:cacheprovider",
        f"--junitxml={junit_path}",
    ]
    print(f"[suite:{name}] starting", flush=True)
    completed = subprocess.run(
        command,
        cwd=REPO,
        env=suite_environment(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    transcript_path.write_bytes(completed.stdout)
    tail = completed.stdout.decode("utf-8", errors="replace").splitlines()[-3:]
    print(
        f"[suite:{name}] exit={completed.returncode} tail={tail}",
        flush=True,
    )
    if not junit_path.is_file():
        raise RuntimeError(f"suite {name!r} did not produce JUnit XML")
    return {
        "command": command,
        "exit_code": completed.returncode,
        "junit_path": junit_path,
        "transcript_path": transcript_path,
    }


def junit_counts(junit_path: Path) -> tuple[dict[str, int], list[str]]:
    root = ET.parse(junit_path).getroot()
    suites = [root] if root.tag == "testsuite" else root.findall("testsuite")
    if not suites:
        raise RuntimeError(f"no testsuite element in {junit_path}")
    counts = {
        key: sum(int(suite.get(key, "0")) for suite in suites)
        for key in ("tests", "failures", "errors", "skipped")
    }
    skips = [
        f"{case.get('classname')}::{case.get('name')}"
        for case in root.iter("testcase")
        if case.find("skipped") is not None
    ]
    return counts, skips


def build_receipt(
    run_info: dict[str, Any],
    environment_lock_sha256: str,
    workflow_sha256: str,
    interpreter_realpath: str,
    head: dict[str, Any],
) -> dict[str, Any]:
    counts, skips = junit_counts(run_info["junit_path"])
    return {
        "exit_code": run_info["exit_code"],
        "command": run_info["command"],
        "environment_lock_sha256": environment_lock_sha256,
        "workflow_sha256": workflow_sha256,
        "interpreter_realpath": interpreter_realpath,
        "counts": counts,
        "skip_identities": skips,
        "junit_sha256": sha256_file(run_info["junit_path"]),
        "transcript_sha256": sha256_file(run_info["transcript_path"]),
        "commit": head["commit"],
        "tree_sha256": head["tree"],
        "dirty": head["dirty"],
    }


def producer_command(
    interpreter: Path,
    snapshot_dirs: dict[str, Path],
    staged_plan: list[dict[str, str]],
) -> list[str]:
    staged = {entry["label"]: entry for entry in staged_plan}
    return [
        str(interpreter),
        "scripts/stopdff_fair_qa_retest.py",
        "--data-dir",
        str(STAGED_DATA.resolve()),
        "--calibration",
        staged["calibration_train"]["path"],
        "--fit-split",
        "val",
        "--eval-split",
        "test",
        "--reward-schedule",
        "power_mark",
        "--qa-arms",
        "idealized,krandom,khard,kdisjoint,klex",
        "--calibrations",
        "shared,performat",
        "--num-bootstrap",
        "1000",
        "--n-test",
        "0",
        "--n-val",
        "0",
        "--seed",
        "1",
        "--eligibility",
        "reproducibility/colm_aims_2026/frozen/pairing_eligibility_v2.json",
        "--snapshot-manifest",
        "reproducibility/colm_aims_2026/frozen/model_snapshot_manifests.json",
        "--primary-model-path",
        str(snapshot_dirs["primary_scorer"]),
        "--disjoint-model-path",
        str(snapshot_dirs["disjoint_selector"]),
        "--records-out",
        "phase4_run_output",
        "--staged-input",
        (
            f"fit_split={staged['fit_split']['path']}:"
            f"{staged['fit_split']['expected_sha256']}"
        ),
        "--staged-input",
        (
            f"mc_dataset={staged['mc_dataset']['path']}:"
            f"{staged['mc_dataset']['expected_sha256']}"
        ),
        "--staged-input",
        (
            f"answer_profiles={staged['answer_profiles']['path']}:"
            f"{staged['answer_profiles']['expected_sha256']}"
        ),
        "--staged-input",
        (
            f"build_metadata={staged['build_metadata']['path']}:"
            f"{staged['build_metadata']['expected_sha256']}"
        ),
        "--out",
        "phase4_run_output/stopdff_fair_qa_regenerated.json",
    ]


def main() -> int:
    phase4_launcher._validate_ambient_environment()
    require_fresh_operational_paths()
    staged_plan = verify_prepared_inputs()
    snapshot_dirs = verify_prepared_snapshots()

    head = clean_head()
    interpreter = Path(sys.executable).resolve()
    lock_bytes = phase4_launcher._default_probe_environment_lock(interpreter)
    (RECEIPTS_DIR / "environment_lock_pip_freeze.txt").write_bytes(lock_bytes)
    environment_lock_sha256 = hashlib.sha256(lock_bytes).hexdigest()
    workflow_sha256 = sha256_file(WORKFLOW_PATH)

    suite_runs = {
        "focused": run_suite("focused", FOCUSED_TESTS),
        "full": run_suite("full", ["tests/"]),
    }
    post_suite_head = clean_head()
    if post_suite_head != head:
        raise RuntimeError(
            f"repository identity changed during suites: {head} -> {post_suite_head}"
        )
    post_suite_lock = phase4_launcher._default_probe_environment_lock(
        interpreter
    )
    if post_suite_lock != lock_bytes:
        raise RuntimeError(
            "certified interpreter dependency lock changed during suites"
        )

    receipt_paths: dict[str, Path] = {}
    receipts: dict[str, dict[str, Any]] = {}
    for name, run_info in suite_runs.items():
        receipt = build_receipt(
            run_info,
            environment_lock_sha256,
            workflow_sha256,
            str(interpreter),
            head,
        )
        receipt_path = RECEIPTS_DIR / f"suite_receipt_{name}.json"
        write_json(receipt_path, receipt)
        receipt_paths[name] = receipt_path
        receipts[name] = receipt

    failed = [
        name
        for name, receipt in receipts.items()
        if receipt["exit_code"] != 0
        or receipt["counts"]["failures"] != 0
        or receipt["counts"]["errors"] != 0
    ]
    if failed:
        print(f"CERTIFICATE_NOT_GENERATED: suite failure(s) {failed}")
        return 1

    host = phase4_launcher._default_host_identity()
    try:
        import numpy as np

        blas = json.dumps(
            np.show_config(mode="dicts"), sort_keys=True, default=str
        )
    except Exception as exc:  # pragma: no cover - fail closed in operation
        raise RuntimeError(f"NumPy BLAS capture failed: {exc}") from exc
    cpu = (
        platform.processor()
        or os.environ.get("PROCESSOR_IDENTIFIER")
        or platform.uname().processor
        or "unknown"
    )
    environment = {
        "interpreter_realpath": str(interpreter),
        "os": host["os"],
        "arch": host["arch"],
        "cpu": cpu,
        "blas": blas,
        "thread_settings": dict(phase4.PHASE4_THREAD_SETTINGS),
        "environment_lock_sha256": environment_lock_sha256,
        "command": producer_command(interpreter, snapshot_dirs, staged_plan),
        "seeds": [1],
        "pythonhashseed": "0",
        "archived_rng_pinned": False,
        "fresh_rng_pinned": True,
        "quarantine_dir": str(QUARANTINE_DIR.resolve()),
        "promote_to": str(PROMOTE_TO.resolve()),
        "exception_ledger_path": str(EXCEPTION_LEDGER_PATH.resolve()),
    }
    config = {
        "repo_root": str(REPO),
        "eligibility_path": str(ELIGIBILITY_PATH),
        "snapshot_manifest_path": str(SNAPSHOT_MANIFEST_PATH),
        "snapshot_dirs": {
            role: str(path) for role, path in snapshot_dirs.items()
        },
        "parity_anchor_path": str(PARITY_ANCHOR_PATH),
        "qa012_manifest_path": str(QA012_REV3_PATH),
        "staged_plan": staged_plan,
        "suite_receipt_paths": {
            name: str(path) for name, path in receipt_paths.items()
        },
        "content_hash_paths": {
            "spec_sha256": str(
                REPO / ".correctless/specs/camera-ready-aims-evidence-2.md"
            ),
            "verifier_sha256": str(
                REPO / "reproducibility/colm_aims_2026/verifier.py"
            ),
            "producer_sha256": str(
                REPO / "scripts/stopdff_fair_qa_retest.py"
            ),
            "schema_py_sha256": str(
                REPO / "reproducibility/colm_aims_2026/schema.py"
            ),
            "pairing_py_sha256": str(
                REPO / "reproducibility/colm_aims_2026/pairing.py"
            ),
            "phase4_py_sha256": str(
                REPO / "reproducibility/colm_aims_2026/phase4.py"
            ),
            "phase4_records_py_sha256": str(
                REPO / "reproducibility/colm_aims_2026/phase4_records.py"
            ),
            "phase4_launcher_py_sha256": str(
                REPO / "reproducibility/colm_aims_2026/phase4_launcher.py"
            ),
            "fileio_py_sha256": str(
                REPO / "scripts/stopdff_v5/fileio.py"
            ),
            "locking_py_sha256": str(
                REPO / "scripts/stopdff_v5/locking.py"
            ),
            "orchestration_sha256": str(Path(__file__).resolve()),
        },
        "environment": environment,
        "offline_flags": [
            "HF_HUB_OFFLINE=1",
            "TRANSFORMERS_OFFLINE=1",
        ],
    }
    components = phase4.gather_certificate_components(config)
    result = phase4.generate_pre_run_ready(components, CERTIFICATE_PATH)
    summary = {
        "ready": result["ready"],
        "path": result["path"],
        "sha256": result["sha256"],
        "commit": head["commit"],
        "tree": head["tree"],
        "focused_counts": receipts["focused"]["counts"],
        "full_counts": receipts["full"]["counts"],
        "failing_checks": result["certificate"].get("failing_checks", []),
    }
    write_json(CERTIFICATE_DIR / "certificate_generation_summary.json", summary)
    print("CERTIFICATE_RESULT=" + json.dumps(summary, sort_keys=True))
    return 0 if result["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
