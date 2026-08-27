"""Focused tests for the checked-in Phase-4 certificate orchestrator."""

from __future__ import annotations

import hashlib
import os
import stat
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import phase4_pre_run_ready_orchestration as orchestration


def _prepared_roots(tmp_path: Path) -> tuple[Path, Path]:
    asset_root = tmp_path / "assets"
    (asset_root / "staging" / "processed").mkdir(parents=True)
    (asset_root / "snapshots").mkdir()
    run_root = tmp_path / "run"
    for name in ("certificate", "launch", "output", "receipts"):
        (run_root / name).mkdir(parents=True, exist_ok=True)
    return asset_root.resolve(), run_root.resolve()


class _GuardedScandir:
    """Iterator that records reads and explodes past its supplied sentinel."""

    def __init__(self, names: list[str]) -> None:
        self._names = names
        self.next_calls = 0

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def __iter__(self):
        return self

    def __next__(self):
        self.next_calls += 1
        index = self.next_calls - 1
        if index >= len(self._names):
            raise AssertionError("directory iterator resumed past overflow sentinel")
        return SimpleNamespace(name=self._names[index])


def test_cli_derives_closed_paths_independently_of_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    asset_root, run_root = _prepared_roots(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    monkeypatch.chdir(outside)

    args = orchestration._build_parser().parse_args(
        [
            "--run-root",
            str(run_root),
            "--run-id",
            "camera-ready_007",
            "--asset-root",
            str(asset_root),
        ]
    )
    paths = orchestration.derive_paths(
        asset_root=args.asset_root,
        run_root=args.run_root,
        run_id=args.run_id,
    )

    assert paths.asset_root == asset_root
    assert paths.run_root == run_root
    assert paths.certificate_path == (
        run_root
        / "certificate"
        / "pre_run_ready_certificate_camera-ready_007.json"
    )
    assert paths.quarantine_dir == (
        run_root / "output" / "phase4_quarantine_camera-ready_007"
    )
    assert paths.promote_to == (
        run_root / "output" / "phase4_run_output_camera-ready_007"
    )
    assert paths.exception_ledger_path == (
        run_root
        / "launch"
        / "phase4_camera-ready_007_single_use_ledger.json"
    )


@pytest.mark.parametrize(
    "run_id",
    [
        "",
        ".",
        "..",
        "../escape",
        "nested/name",
        r"nested\name",
        " space",
        "trailing.",
    ],
)
def test_derive_paths_refuses_unsafe_run_id(
    tmp_path: Path, run_id: str
) -> None:
    asset_root, run_root = _prepared_roots(tmp_path)
    with pytest.raises(RuntimeError, match="run ID must be a portable"):
        orchestration.derive_paths(
            asset_root=asset_root, run_root=run_root, run_id=run_id
        )


def test_derive_paths_refuses_relative_operator_roots(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="asset root must be an absolute"):
        orchestration.derive_paths(
            asset_root=Path("relative-assets"),
            run_root=tmp_path.resolve(),
            run_id="run-1",
        )
    with pytest.raises(RuntimeError, match="run root must be an absolute"):
        orchestration.derive_paths(
            asset_root=tmp_path.resolve(),
            run_root=Path("relative-run"),
            run_id="run-1",
        )


def test_derive_paths_refuses_noncanonical_absolute_root(tmp_path: Path) -> None:
    asset_root, run_root = _prepared_roots(tmp_path)
    noncanonical_asset = asset_root / "child" / ".."
    with pytest.raises(RuntimeError, match="must be a canonical path"):
        orchestration.derive_paths(
            asset_root=noncanonical_asset,
            run_root=run_root,
            run_id="run-1",
        )


def test_freshness_refuses_overlapping_asset_and_run_roots(
    tmp_path: Path,
) -> None:
    asset_root = tmp_path / "assets"
    (asset_root / "staging" / "processed").mkdir(parents=True)
    (asset_root / "snapshots").mkdir()
    run_root = asset_root / "run"
    for name in ("certificate", "launch", "output", "receipts"):
        (run_root / name).mkdir(parents=True, exist_ok=True)
    paths = orchestration.derive_paths(
        asset_root=asset_root.resolve(),
        run_root=run_root.resolve(),
        run_id="run-1",
    )

    with pytest.raises(RuntimeError, match="asset root and run root.*disjoint"):
        orchestration.require_fresh_operational_paths(paths)


@pytest.mark.parametrize(
    "target_name",
    ["certificate_path", "quarantine_dir", "promote_to", "exception_ledger_path"],
)
def test_freshness_refuses_every_existing_derived_target(
    tmp_path: Path, target_name: str
) -> None:
    asset_root, run_root = _prepared_roots(tmp_path)
    paths = orchestration.derive_paths(
        asset_root=asset_root, run_root=run_root, run_id="run-1"
    )
    target = getattr(paths, target_name)
    if target.suffix:
        target.write_bytes(b"incumbent")
    else:
        target.mkdir()

    with pytest.raises(RuntimeError, match="target already exists"):
        orchestration.require_fresh_operational_paths(paths)


def test_freshness_accepts_only_exact_empty_operational_shape(
    tmp_path: Path,
) -> None:
    asset_root, run_root = _prepared_roots(tmp_path)
    paths = orchestration.derive_paths(
        asset_root=asset_root, run_root=run_root, run_id="run-1"
    )

    orchestration.require_fresh_operational_paths(paths)

    (run_root / "unexpected.txt").write_text("stale", encoding="utf-8")
    with pytest.raises(RuntimeError, match="membership must be exactly"):
        orchestration.require_fresh_operational_paths(paths)


def test_freshness_stops_run_root_scan_at_expected_plus_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    asset_root, run_root = _prepared_roots(tmp_path)
    paths = orchestration.derive_paths(
        asset_root=asset_root, run_root=run_root, run_id="run-1"
    )
    guarded = _GuardedScandir(
        ["certificate", "launch", "output", "receipts", "overflow", "forbidden"]
    )
    original_scandir = os.scandir

    def bounded_root(path):
        return guarded if Path(path) == run_root else original_scandir(path)

    monkeypatch.setattr(orchestration.os, "scandir", bounded_root)

    with pytest.raises(RuntimeError, match="more than 4 entries"):
        orchestration.require_fresh_operational_paths(paths)

    assert guarded.next_calls == 5


def test_freshness_stops_empty_directory_scan_at_first_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    asset_root, run_root = _prepared_roots(tmp_path)
    paths = orchestration.derive_paths(
        asset_root=asset_root, run_root=run_root, run_id="run-1"
    )
    receipts = run_root / "receipts"
    guarded = _GuardedScandir(["unexpected", "forbidden"])
    original_scandir = os.scandir

    def bounded_receipts(path):
        return guarded if Path(path) == receipts else original_scandir(path)

    monkeypatch.setattr(orchestration.os, "scandir", bounded_receipts)

    with pytest.raises(RuntimeError, match="operational directory is not fresh"):
        orchestration.require_fresh_operational_paths(paths)

    assert guarded.next_calls == 1


def test_suite_environment_adds_only_trusted_git_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    git_dir = tmp_path / "trusted-git"
    git_dir.mkdir()
    git_executable = git_dir / ("git.exe" if os.name == "nt" else "git")
    git_executable.write_bytes(b"placeholder")
    monkeypatch.setattr(
        orchestration.phase4_launcher,
        "_sanitized_runtime_environment",
        lambda: {
            "SYSTEMROOT": "system",
            "PATH": "ambient-untrusted-path",
            "HF_HUB_OFFLINE": "1",
        },
    )

    producer = orchestration.producer_environment()
    suite = orchestration.suite_environment(git_executable)

    assert "PATH" not in producer
    assert suite["PATH"] == str(git_dir.resolve())
    assert "ambient-untrusted-path" not in suite.values()
    assert suite["HF_HUB_OFFLINE"] == "1"


def test_run_suite_uses_repo_cwd_and_suite_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    asset_root, run_root = _prepared_roots(tmp_path)
    paths = orchestration.derive_paths(
        asset_root=asset_root, run_root=run_root, run_id="run-1"
    )
    outside = tmp_path / "operator-cwd"
    outside.mkdir()
    monkeypatch.chdir(outside)
    expected_environment = {"PATH": "trusted-git-only", "MARKER": "suite"}
    monkeypatch.setattr(
        orchestration, "suite_environment", lambda: expected_environment
    )
    observed: dict[str, object] = {}

    def fake_run(
        command,
        *,
        environment,
        transcript_stage,
        timeout_seconds,
        max_transcript_bytes,
    ):
        observed.update(
            command=command,
            env=environment,
            timeout_seconds=timeout_seconds,
            max_transcript_bytes=max_transcript_bytes,
        )
        junit_arg = next(
            token for token in command if token.startswith("--junitxml=")
        )
        Path(junit_arg.partition("=")[2]).write_text(
            '<testsuite tests="1" failures="0" errors="0" skipped="0"/>',
            encoding="utf-8",
        )
        transcript_stage.write_bytes(b"one passed\n")
        return 0

    monkeypatch.setattr(orchestration, "_run_bounded_suite_process", fake_run)
    result = orchestration.run_suite(paths, "focused", ["tests/example.py"])

    assert observed["env"] is expected_environment
    assert Path(result["command"][-1].partition("=")[2]).is_absolute()
    assert result["exit_code"] == 0
    assert result["junit_path"].parent == paths.receipts_dir
    assert result["transcript_path"].read_bytes() == b"one passed\n"
    assert Path.cwd() == outside


def test_producer_command_binds_parameterized_asset_root(tmp_path: Path) -> None:
    asset_root, run_root = _prepared_roots(tmp_path)
    paths = orchestration.derive_paths(
        asset_root=asset_root, run_root=run_root, run_id="run-1"
    )
    staged_plan = [
        {
            "label": label,
            "path": str(asset_root / "staging" / "processed" / filename),
            "expected_sha256": digest,
        }
        for label, (filename, digest) in orchestration.EXPECTED_STAGED.items()
    ]
    snapshots = {
        "primary_scorer": asset_root / "snapshots" / "primary_scorer",
        "disjoint_selector": asset_root / "snapshots" / "disjoint_selector",
    }
    command = orchestration.producer_command(
        paths, Path("C:/Python/python.exe"), snapshots, staged_plan
    )

    data_index = command.index("--data-dir")
    assert command[data_index + 1] == str(paths.staged_data)
    assert "PATH" not in orchestration.producer_environment()


def test_clean_head_rejects_untracked_executable_or_test_surface(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "harmless-note.txt").write_text("note", encoding="utf-8")
    (tmp_path / "injected.py").write_text("raise SystemExit", encoding="utf-8")

    def fake_git(*args: str) -> str:
        if args[:3] == ("status", "--porcelain", "--untracked-files=no"):
            return ""
        if args[:2] == ("status", "--porcelain=v1"):
            return "?? harmless-note.txt\0?? injected.py\0"
        if args == ("rev-parse", "HEAD"):
            return "a" * 40 + "\n"
        if args == ("rev-parse", "HEAD^{tree}"):
            return "b" * 40 + "\n"
        raise AssertionError(args)

    monkeypatch.setattr(orchestration, "REPO", tmp_path)
    monkeypatch.setattr(orchestration, "git", fake_git)

    with pytest.raises(RuntimeError, match="untracked executable/import/test"):
        orchestration.clean_head()


def test_clean_head_preserves_inert_ordinary_untracked_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "operator-notes").write_text("inert", encoding="utf-8")

    def fake_git(*args: str) -> str:
        if args[:3] == ("status", "--porcelain", "--untracked-files=no"):
            return ""
        if args[:2] == ("status", "--porcelain=v1"):
            return "?? operator-notes\0"
        if args == ("rev-parse", "HEAD"):
            return "a" * 40 + "\n"
        if args == ("rev-parse", "HEAD^{tree}"):
            return "b" * 40 + "\n"
        raise AssertionError(args)

    monkeypatch.setattr(orchestration, "REPO", tmp_path)
    monkeypatch.setattr(orchestration, "git", fake_git)

    head = orchestration.clean_head()

    assert head["untracked"] == ["operator-notes"]


def test_clean_head_rejects_suffixless_untracked_directory_reparse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate = tmp_path / "numpy"
    candidate.mkdir()

    def fake_git(*args: str) -> str:
        if args[:3] == ("status", "--porcelain", "--untracked-files=no"):
            return ""
        if args[:2] == ("status", "--porcelain=v1"):
            return "?? numpy\0"
        raise AssertionError(args)

    original_stat = os.stat

    def reparse_stat(path, *args, **kwargs):
        if Path(path) == candidate and kwargs.get("follow_symlinks") is False:
            return SimpleNamespace(
                st_mode=stat.S_IFDIR,
                st_dev=1,
                st_ino=2,
                st_file_attributes=0x400,
            )
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(
        orchestration.stat,
        "FILE_ATTRIBUTE_REPARSE_POINT",
        0x400,
        raising=False,
    )
    monkeypatch.setattr(orchestration, "REPO", tmp_path)
    monkeypatch.setattr(orchestration, "git", fake_git)
    monkeypatch.setattr(orchestration.os, "stat", reparse_stat)

    with pytest.raises(RuntimeError, match="symlink/reparse"):
        orchestration.clean_head()


def _small_staged_inputs(
    asset_root: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, tuple[str, str]]:
    prepared: dict[str, tuple[str, str]] = {}
    for index, label in enumerate(orchestration.EXPECTED_STAGED):
        filename = f"input-{index}.json"
        data = f'{{"index":{index}}}\n'.encode()
        (asset_root / "staging" / "processed" / filename).write_bytes(data)
        prepared[label] = (filename, hashlib.sha256(data).hexdigest())
    monkeypatch.setattr(orchestration, "EXPECTED_STAGED", prepared)
    return prepared


def test_staged_inputs_are_hashed_no_follow_under_stable_lexical_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    asset_root, run_root = _prepared_roots(tmp_path)
    prepared = _small_staged_inputs(asset_root, monkeypatch)
    paths = orchestration.derive_paths(
        asset_root=asset_root, run_root=run_root, run_id="run-1"
    )

    plan = orchestration.verify_prepared_inputs(paths)

    assert [entry["label"] for entry in plan] == list(prepared)
    assert all(
        Path(entry["path"]).parent == paths.staged_data for entry in plan
    )


def test_staged_input_final_symlink_is_rejected_without_following(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    asset_root, run_root = _prepared_roots(tmp_path)
    prepared = _small_staged_inputs(asset_root, monkeypatch)
    first_label = next(iter(prepared))
    filename, digest = prepared[first_label]
    source = asset_root / "outside-source.json"
    source.write_text("{}", encoding="utf-8")
    candidate = asset_root / "staging" / "processed" / filename
    candidate.unlink()
    try:
        candidate.symlink_to(source)
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")
    prepared[first_label] = (filename, digest)
    paths = orchestration.derive_paths(
        asset_root=asset_root, run_root=run_root, run_id="run-1"
    )

    with pytest.raises(RuntimeError, match="not a regular file"):
        orchestration.verify_prepared_inputs(paths)


def test_snapshot_role_root_alias_is_rejected_before_verifier_descent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    asset_root, run_root = _prepared_roots(tmp_path)
    outside = tmp_path / "outside-snapshot"
    outside.mkdir()
    role_link = asset_root / "snapshots" / "primary_scorer"
    try:
        role_link.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlink creation unavailable: {exc}")
    (asset_root / "snapshots" / "disjoint_selector").mkdir()
    manifest = {
        "roles": {
            role: {"files": {}}
            for role in ("primary_scorer", "disjoint_selector")
        }
    }
    monkeypatch.setattr(
        orchestration.phase4,
        "load_model_snapshot_manifest",
        lambda path: manifest,
    )
    monkeypatch.setattr(
        orchestration.phase4,
        "verify_snapshot_dir",
        lambda *args: pytest.fail("aliased role reached snapshot verifier"),
    )
    paths = orchestration.derive_paths(
        asset_root=asset_root, run_root=run_root, run_id="run-1"
    )

    with pytest.raises(orchestration.schema.TypedIngressError, match="symlink|reparse"):
        orchestration.verify_prepared_snapshots(paths)


def test_snapshot_declared_file_cap_refuses_before_content_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    asset_root, run_root = _prepared_roots(tmp_path)
    for role in ("primary_scorer", "disjoint_selector"):
        (asset_root / "snapshots" / role).mkdir()
    manifest = {
        "roles": {
            "primary_scorer": {
                "files": {
                    "weights.bin": {
                        "size": orchestration.MAX_SNAPSHOT_FILE_BYTES + 1,
                        "sha256": "a" * 64,
                    }
                }
            },
            "disjoint_selector": {"files": {}},
        }
    }
    monkeypatch.setattr(
        orchestration.phase4,
        "load_model_snapshot_manifest",
        lambda path: manifest,
    )
    monkeypatch.setattr(
        orchestration.phase4,
        "verify_snapshot_dir",
        lambda *args: pytest.fail("oversize role reached snapshot verifier"),
    )
    paths = orchestration.derive_paths(
        asset_root=asset_root, run_root=run_root, run_id="run-1"
    )

    with pytest.raises(RuntimeError, match="per-file byte limit"):
        orchestration.verify_prepared_snapshots(paths)


@pytest.mark.parametrize("collision_kind", ["transcript", "junit"])
def test_suite_late_collision_preserves_incumbent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    collision_kind: str,
) -> None:
    asset_root, run_root = _prepared_roots(tmp_path)
    paths = orchestration.derive_paths(
        asset_root=asset_root, run_root=run_root, run_id="run-1"
    )
    incumbent = b"peer-owned"

    def collide_after_suite(
        command,
        *,
        environment,
        transcript_stage,
        timeout_seconds,
        max_transcript_bytes,
    ):
        junit_arg = next(
            token for token in command if token.startswith("--junitxml=")
        )
        Path(junit_arg.partition("=")[2]).write_text(
            '<testsuite tests="1" failures="0" errors="0" skipped="0"/>',
            encoding="utf-8",
        )
        transcript_stage.write_bytes(b"one passed\n")
        collision_path = (
            paths.receipts_dir / "focused_suite_transcript.txt"
            if collision_kind == "transcript"
            else paths.receipts_dir / "focused_suite.xml"
        )
        collision_path.write_bytes(incumbent)
        return 0

    monkeypatch.setattr(
        orchestration, "_run_bounded_suite_process", collide_after_suite
    )
    monkeypatch.setattr(orchestration, "suite_environment", lambda: {})

    with pytest.raises(FileExistsError, match="already exists"):
        orchestration.run_suite(paths, "focused", ["tests/example.py"])
    collision_path = (
        paths.receipts_dir / "focused_suite_transcript.txt"
        if collision_kind == "transcript"
        else paths.receipts_dir / "focused_suite.xml"
    )
    assert collision_path.read_bytes() == incumbent
    if collision_kind == "transcript":
        assert not (paths.receipts_dir / "focused_suite.xml").exists()
    else:
        assert (
            paths.receipts_dir / "focused_suite_transcript.txt"
        ).read_bytes() == b"one passed\n"


def test_create_once_publication_rejects_late_claim_without_overwrite(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "suite_receipt_focused.json"
    destination.write_bytes(b"incumbent")

    with pytest.raises(FileExistsError, match="already exists"):
        orchestration.publish_bytes_create_once(
            destination, b"replacement", label="focused suite receipt"
        )

    assert destination.read_bytes() == b"incumbent"


def test_bounded_suite_process_times_out_and_reaps_process_tree(
    tmp_path: Path,
) -> None:
    transcript = tmp_path / "timeout.txt"
    command = [sys.executable, "-c", "import time; time.sleep(30)"]

    with pytest.raises(RuntimeError, match="exceeded its timeout"):
        orchestration._run_bounded_suite_process(
            command,
            environment=os.environ.copy(),
            transcript_stage=transcript,
            timeout_seconds=0.05,
            max_transcript_bytes=1024,
        )

    assert transcript.exists()


def test_bounded_suite_process_stops_at_transcript_cap(tmp_path: Path) -> None:
    transcript = tmp_path / "bounded.txt"
    command = [
        sys.executable,
        "-c",
        "import sys; sys.stdout.write('x' * 1000000); sys.stdout.flush()",
    ]

    with pytest.raises(RuntimeError, match="transcript exceeded"):
        orchestration._run_bounded_suite_process(
            command,
            environment=os.environ.copy(),
            transcript_stage=transcript,
            timeout_seconds=10,
            max_transcript_bytes=127,
        )

    assert transcript.stat().st_size == 127


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object regression")
def test_bounded_suite_process_reaps_descendant_after_root_exits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A dead root PID must not strand a pipe-owning Windows descendant."""

    transcript = tmp_path / "descendant.txt"
    pid_path = tmp_path / "descendant.pid"
    child_code = (
        "import os,time; "
        f"open({str(pid_path)!r}, 'w').write(str(os.getpid())); "
        "time.sleep(30)"
    )
    root_code = (
        "import pathlib,subprocess,sys,time; "
        "time.sleep(0.2); "
        f"p=subprocess.Popen([sys.executable, '-c', {child_code!r}], "
        "stdout=sys.stdout, stderr=sys.stderr); "
        f"target=pathlib.Path({str(pid_path)!r}); "
        "deadline=time.monotonic()+5; "
        "exec('while not target.exists() and time.monotonic() < deadline:\\n "
        "   time.sleep(0.01)')"
    )
    monkeypatch.setattr(
        orchestration, "SUITE_READER_JOIN_TIMEOUT_SECONDS", 0.2
    )

    started = time.monotonic()
    with pytest.raises(RuntimeError, match="left running descendants"):
        orchestration._run_bounded_suite_process(
            [sys.executable, "-c", root_code],
            environment=os.environ.copy(),
            transcript_stage=transcript,
            timeout_seconds=10,
            max_transcript_bytes=1024,
        )
    assert time.monotonic() - started < 5

    _assert_windows_process_exited(int(pid_path.read_text(encoding="utf-8")))

    moved = transcript.with_suffix(".closed")
    transcript.rename(moved)
    moved.unlink()


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object regression")
def test_bounded_suite_process_timeout_reaps_all_job_members(
    tmp_path: Path,
) -> None:
    """Timeout cleanup must kill both the gated root and its descendant."""

    transcript = tmp_path / "timeout-descendant.txt"
    pid_path = tmp_path / "timeout-descendant.pid"
    child_code = (
        "import os,time; "
        f"open({str(pid_path)!r}, 'w').write(str(os.getpid())); "
        "time.sleep(30)"
    )
    root_code = (
        "import pathlib,subprocess,sys,time; "
        f"subprocess.Popen([sys.executable, '-c', {child_code!r}], "
        "stdout=sys.stdout, stderr=sys.stderr); "
        f"target=pathlib.Path({str(pid_path)!r}); "
        "deadline=time.monotonic()+5; "
        "exec('while not target.exists() and time.monotonic() < deadline:\\n "
        "   time.sleep(0.01)'); "
        "time.sleep(30)"
    )

    with pytest.raises(RuntimeError, match="exceeded its timeout"):
        orchestration._run_bounded_suite_process(
            [sys.executable, "-c", root_code],
            environment=os.environ.copy(),
            transcript_stage=transcript,
            timeout_seconds=1,
            max_transcript_bytes=1024,
        )

    _assert_windows_process_exited(int(pid_path.read_text(encoding="utf-8")))
    moved = transcript.with_suffix(".closed")
    transcript.rename(moved)
    moved.unlink()


def _assert_windows_process_exited(process_id: int) -> None:
    """Assert that a Windows PID is absent or its retained object is signaled."""

    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel32.WaitForSingleObject.restype = wintypes.DWORD
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    handle = kernel32.OpenProcess(0x00100000, False, process_id)
    if handle:
        try:
            assert kernel32.WaitForSingleObject(handle, 5000) == 0
        finally:
            assert kernel32.CloseHandle(handle)
    else:
        assert ctypes.get_last_error() == 87


def test_windows_job_termination_precedes_member_enumeration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cleanup must stop a growing tree before any bounded PID inventory."""

    events: list[str] = []
    job = object.__new__(orchestration._WindowsJobObject)
    job._handle = 1
    job._ctypes = SimpleNamespace(
        get_last_error=lambda: 0,
        FormatError=lambda _error: "",
    )
    job._kernel32 = SimpleNamespace(
        TerminateJobObject=lambda _handle, _code: events.append("terminate") or 1,
        CloseHandle=lambda _handle: 1,
    )
    monkeypatch.setattr(
        job,
        "_open_member_process_handles",
        lambda *, deadline: events.append("enumerate") or [],
    )
    monkeypatch.setattr(job, "active_processes", lambda: 0)

    job.terminate()

    assert events == ["terminate", "enumerate"]


def test_windows_job_member_enumeration_has_finite_attempt_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Perpetual ERROR_MORE_DATA cannot delay terminal cleanup indefinitely."""

    job = object.__new__(orchestration._WindowsJobObject)
    job._handle = 1
    attempts = 0
    monkeypatch.setattr(job, "active_processes", lambda: 1)

    def always_too_small(_capacity: int) -> tuple[int, ...]:
        nonlocal attempts
        attempts += 1
        raise orchestration._WindowsJobMemberListTooSmall(1, 1)

    monkeypatch.setattr(job, "_query_member_process_ids", always_too_small)

    with pytest.raises(RuntimeError, match="did not stabilize"):
        job._open_member_process_handles(deadline=time.monotonic() + 30)

    assert attempts == 8


def test_windows_job_rejects_reused_pid_handle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An opened PID is retained only if its process still belongs to the Job."""

    import ctypes
    from ctypes import wintypes

    events: list[tuple[object, ...]] = []

    class FakeKernel32:
        def OpenProcess(self, access, inherit, process_id):
            events.append(("open", access, inherit, process_id))
            return 99

        def IsProcessInJob(self, handle, job_handle, result):
            events.append(("member", handle, job_handle))
            result._obj.value = 0
            return 1

        def CloseHandle(self, handle):
            events.append(("close", handle))
            return 1

    job = object.__new__(orchestration._WindowsJobObject)
    job._handle = 1
    job._ctypes = ctypes
    job._wintypes = wintypes
    job._kernel32 = FakeKernel32()
    monkeypatch.setattr(job, "active_processes", lambda: 1)
    monkeypatch.setattr(job, "_query_member_process_ids", lambda _capacity: (123,))

    handles = job._open_member_process_handles(deadline=time.monotonic() + 30)

    assert handles == []
    assert events == [
        ("open", 0x00101000, False, 123),
        ("member", 99, 1),
        ("close", 99),
    ]


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object regression")
def test_bounded_suite_process_closes_job_after_termination_fault(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """KILL_ON_CLOSE ownership is retained when explicit termination fails."""

    events: list[str] = []

    class FaultedJob:
        def assign(self, _process) -> None:
            events.append("assign")

        def active_processes(self) -> int:
            events.append("active")
            return 0

        def terminate(self) -> None:
            events.append("terminate")
            raise RuntimeError("injected job termination failure")

        def close(self) -> None:
            events.append("close")

    monkeypatch.setattr(orchestration, "_WindowsJobObject", FaultedJob)
    transcript = tmp_path / "termination-fault.txt"

    with pytest.raises(RuntimeError, match="injected job termination failure"):
        orchestration._run_bounded_suite_process(
            [sys.executable, "-c", "print('finished')"],
            environment=os.environ.copy(),
            transcript_stage=transcript,
            timeout_seconds=10,
            max_transcript_bytes=1024,
        )

    assert events == ["assign", "active", "terminate", "close"]
    moved = transcript.with_suffix(".closed")
    transcript.rename(moved)
    moved.unlink()
