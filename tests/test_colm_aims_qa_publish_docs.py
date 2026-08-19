"""QA fix-round-1 regression suite — crash-relic recovery + doc conformance
+ subprocess guard (QA-008, QA-011).

QA-008: a mid-publish crash of the run-directory publisher leaves an empty
relic; recovery is the explicit single-owner ``reclaim_crashed_relic`` path;
canonical selection rejects empty relics; every README runtime claim is
backed by an executable assertion.
QA-011: the R-028 no-network guard covers subprocess CLI runs via the
env-triggered sitecustomize shim.
Spec: .correctless/specs/camera-ready-aims-evidence.md (R-016/R-028/R-037/R-039)
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from reproducibility.colm_aims_2026 import schema, verifier, verify
from tests._colm_aims_helpers import (  # noqa: F401  (colm_no_network is an autouse fixture)
    NO_NET_SHIM_DIR,
    REPO_ROOT,
    build_package,
    cli_args_for,
    cli_subprocess_env,
    colm_no_network,
    make_ledger,
    repo_head_commit,
    run_cli,
    sha256_file,
)

README_PATH = REPO_ROOT / "reproducibility" / "colm_aims_2026" / "README.md"


def _stage(base: Path, name: str, content: str = '{"v": 1}\n') -> Path:
    staged = base / f"staged-{name}"
    staged.mkdir(parents=True)
    (staged / "profile.json").write_text(content, encoding="utf-8")
    return staged


# ---------------------------------------------------------------------------
# QA-008: kill-mid-publish / poisoned retry / explicit recovery
# ---------------------------------------------------------------------------


def test_dir_publish_crash_poisons_plain_retry_and_flagged_retry_recovers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # QA-008 [R-016]: crash between the mkdir claim and the filling rename
    # leaves an EMPTY run-slot relic; a plain retry fails closed forever;
    # the explicit single-owner recovery reclaims and republishes exactly
    # one artifact with no debris.
    runs_root = tmp_path / "runs"

    def crash(*args, **kwargs):
        raise OSError("simulated kill mid-publish")

    monkeypatch.setattr(os, "rename", crash)
    with pytest.raises(OSError):
        schema.publish_evidence_package(
            _stage(tmp_path, "a"), runs_root, "run-0001"
        )
    monkeypatch.undo()

    slot = runs_root / "run-0001"
    assert slot.is_dir() and list(slot.iterdir()) == [], (
        "crash must leave the empty mkdir-claimed relic"
    )

    # Plain retry: poisoned — fails closed exactly as before the fix.
    with pytest.raises(FileExistsError):
        schema.publish_evidence_package(
            _stage(tmp_path, "b"), runs_root, "run-0001"
        )

    # Explicit single-owner recovery: reclaims the relic, publishes once.
    published = schema.publish_evidence_package(
        _stage(tmp_path, "c"),
        runs_root,
        "run-0001",
        reclaim_crashed_relic=True,
    )
    assert published == slot
    assert [p.name for p in sorted(slot.iterdir())] == ["profile.json"]
    assert (slot / "profile.json").read_text("utf-8") == '{"v": 1}\n'
    assert sorted(p.name for p in runs_root.iterdir()) == ["run-0001"], (
        "no staging debris or duplicate run slots"
    )


def test_reclaim_flag_never_clobbers_published_bytes(tmp_path: Path):
    # QA-008 [R-016/R-039]: the recovery flag reclaims ONLY an empty relic —
    # a genuinely published run stays byte-identical and the retry fails
    # closed even with the flag set.
    runs_root = tmp_path / "runs"
    published = schema.publish_evidence_package(
        _stage(tmp_path, "a"), runs_root, "run-0001"
    )
    original = sha256_file(published / "profile.json")
    with pytest.raises(FileExistsError):
        schema.publish_evidence_package(
            _stage(tmp_path, "b", '{"v": 2}\n'),
            runs_root,
            "run-0001",
            reclaim_crashed_relic=True,
        )
    assert sha256_file(published / "profile.json") == original


def test_plain_publish_still_fails_closed_on_preclaimed_empty_slot(
    tmp_path: Path,
):
    # QA-008 guard: without the recovery flag, the pre-claimed empty slot
    # keeps failing closed (the fresh-publish collision semantics of
    # publish_dir_create_once are untouched).
    runs_root = tmp_path / "runs"
    (runs_root / "run-0001").mkdir(parents=True)
    with pytest.raises(FileExistsError):
        schema.publish_evidence_package(
            _stage(tmp_path, "a"), runs_root, "run-0001"
        )
    assert list((runs_root / "run-0001").iterdir()) == []


def test_resolve_canonical_rejects_empty_relic_as_dangling(tmp_path: Path):
    # QA-008 [R-039]: an empty run directory is a crash relic, not a
    # published evidence package — the canonical pointer is dangling.
    runs_root = tmp_path / "runs"
    (runs_root / "run-0001").mkdir(parents=True)
    ledger = make_ledger(
        source_commit=repo_head_commit(), canonical_run_id="run-0001"
    )
    with pytest.raises(schema.ColmAimsError) as exc:
        verifier.resolve_canonical_package(runs_root, ledger)
    msg = str(exc.value).lower()
    assert "empty" in msg and "run-0001" in str(exc.value)


def test_resolve_canonical_still_returns_published_run(tmp_path: Path):
    # QA-008 guard: a genuinely published run still resolves.
    runs_root = tmp_path / "runs"
    published = schema.publish_evidence_package(
        _stage(tmp_path, "a"), runs_root, "run-0001"
    )
    ledger = make_ledger(
        source_commit=repo_head_commit(), canonical_run_id="run-0001"
    )
    assert verifier.resolve_canonical_package(runs_root, ledger) == published


# ---------------------------------------------------------------------------
# QA-008 class fix: README runtime claims backed by executable assertions
# ---------------------------------------------------------------------------


def _readme_exit_code_rows() -> dict[int, str]:
    text = README_PATH.read_text("utf-8")
    rows = re.findall(r"^\|\s*(\d+)\s*\|\s*([^|]+)\|", text, re.MULTILINE)
    return {int(code): meaning.strip() for code, meaning in rows}


def test_readme_exit_code_table_matches_pinned_constants():
    # QA-008 class fix [R-037/R-038]: the documented table is parsed and
    # checked against the real constants — the doc cannot drift silently.
    codes = _readme_exit_code_rows()
    assert set(codes) == {
        verify.EXIT_PASS,
        verify.EXIT_GATE_FAIL,
        verify.EXIT_USAGE_ERROR,
        verify.EXIT_INGRESS_ERROR,
    }
    assert "pass" in codes[verify.EXIT_PASS].lower()
    assert "fail" in codes[verify.EXIT_GATE_FAIL].lower()
    assert "usage" in codes[verify.EXIT_USAGE_ERROR].lower()
    assert "ingress" in codes[verify.EXIT_INGRESS_ERROR].lower()


def test_readme_pass_row_drives_the_real_cli(tmp_path: Path):
    # QA-008 class fix: the documented pass code is what the real CLI exits
    # with on a pristine source run.
    pkg = build_package(tmp_path)
    proc = run_cli(*cli_args_for(pkg, "source"))
    codes = _readme_exit_code_rows()
    pass_code = next(
        code for code, meaning in codes.items() if "pass" in meaning.lower()
    )
    assert proc.returncode == pass_code == verify.EXIT_PASS


def test_readme_reclaim_claims_match_publisher_behavior():
    # QA-008 [R-038]: the README documents the per-shape reclaim policy the
    # publisher actually implements — the explicit dir-publish recovery flag
    # by name, the file-publish auto-reclaim primitive by name, and the
    # dangling-pointer rejection of empty relics.
    text = README_PATH.read_text("utf-8")
    assert "reclaim_crashed_relic" in text
    assert "create_once_bytes" in text
    low = text.lower()
    assert "empty run" in low or "empty relic" in low or "relic" in low
    assert "dangling" in low


# ---------------------------------------------------------------------------
# QA-011: subprocess no-network guard
# ---------------------------------------------------------------------------


def test_cli_subprocess_env_carries_guard_shim():
    # QA-011 [R-028]: run_cli children inherit the env-triggered guard.
    env = cli_subprocess_env()
    assert env["COLM_AIMS_TEST_NO_NET"] == "1"
    assert str(NO_NET_SHIM_DIR) in env["PYTHONPATH"]
    assert (NO_NET_SHIM_DIR / "sitecustomize.py").is_file()


def test_guard_blocks_network_in_child_interpreters():
    # QA-011 [R-028]: a child interpreter under the run_cli env raises the
    # guard error on any INET connect attempt.
    probe = (
        "import socket\n"
        "try:\n"
        "    socket.create_connection(('127.0.0.1', 9), timeout=0.1)\n"
        "except RuntimeError as exc:\n"
        "    print('GUARDED' if 'network disabled' in str(exc) else 'WRONG')\n"
        "else:\n"
        "    print('UNGUARDED')\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", probe],
        env=cli_subprocess_env(),
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=False,
    )
    assert "GUARDED" in proc.stdout, (proc.stdout, proc.stderr[-300:])


def test_guard_is_inert_without_the_env_trigger():
    # QA-011: the shim is opt-in — without the env var a child interpreter
    # keeps its normal socket module (no cross-suite interference).
    probe = (
        "import socket\n"
        "patched = socket.create_connection.__module__ == 'sitecustomize'\n"
        "print('PATCHED' if patched else 'CLEAN')\n"
    )
    env = cli_subprocess_env()
    env.pop("COLM_AIMS_TEST_NO_NET")
    proc = subprocess.run(
        [sys.executable, "-c", probe],
        env=env,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=False,
    )
    assert "CLEAN" in proc.stdout, (proc.stdout, proc.stderr[-300:])
