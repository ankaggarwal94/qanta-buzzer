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
        transcript_fd,
        inherited_fds,
        timeout_seconds,
        max_transcript_bytes,
    ):
        observed.update(
            command=command,
            env=environment,
            transcript_fd=transcript_fd,
            inherited_fds=inherited_fds,
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
        os.write(transcript_fd, b"one passed\n")
        return 0

    monkeypatch.setattr(orchestration, "_run_bounded_suite_process", fake_run)
    result = orchestration.run_suite(paths, "focused", ["tests/example.py"])

    assert observed["env"] is expected_environment
    assert Path(result["command"][-1].partition("=")[2]).is_absolute()
    assert result["exit_code"] == 0
    assert result["junit_path"].parent == paths.receipts_dir
    assert result["transcript_path"].read_bytes() == b"one passed\n"
    assert Path.cwd() == outside
    assert list(run_root.glob(".focused-suite-staged-*")) == []


def test_run_suite_real_process_uses_generation_bound_stage_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    asset_root, run_root = _prepared_roots(tmp_path)
    paths = orchestration.derive_paths(
        asset_root=asset_root, run_root=run_root, run_id="run-1"
    )
    monkeypatch.setattr(
        orchestration, "suite_environment", lambda: os.environ.copy()
    )

    result = orchestration.run_suite(
        paths,
        "focused",
        [
            "tests/test_phase4_pre_run_ready_orchestration.py::"
            "test_producer_command_binds_parameterized_asset_root"
        ],
        timeout_seconds=60,
    )

    assert result["exit_code"] == 0
    assert b"1 passed" in result["transcript_bytes"]
    assert result["junit_bytes"].startswith(b"<?xml")
    assert list(run_root.glob(".focused-suite-staged-*")) == []


def test_run_suite_cleanup_removes_transcript_only_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    asset_root, run_root = _prepared_roots(tmp_path)
    paths = orchestration.derive_paths(
        asset_root=asset_root, run_root=run_root, run_id="run-1"
    )

    def write_transcript_then_fail(
        _command,
        *,
        environment,
        transcript_stage,
        transcript_fd,
        inherited_fds,
        timeout_seconds,
        max_transcript_bytes,
    ):
        del environment, inherited_fds, timeout_seconds, max_transcript_bytes
        os.write(transcript_fd, b"partial\n")
        raise RuntimeError("injected partial suite failure")

    monkeypatch.setattr(
        orchestration,
        "_run_bounded_suite_process",
        write_transcript_then_fail,
    )
    monkeypatch.setattr(orchestration, "suite_environment", lambda: {})

    with pytest.raises(RuntimeError, match="injected partial"):
        orchestration.run_suite(paths, "focused", ["tests/example.py"])

    assert list(run_root.glob(".focused-suite-staged-*")) == []


def test_run_suite_cleanup_removes_zero_output_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    asset_root, run_root = _prepared_roots(tmp_path)
    paths = orchestration.derive_paths(
        asset_root=asset_root, run_root=run_root, run_id="run-1"
    )

    def fail_before_outputs(*_args, **_kwargs):
        raise OSError("injected process creation failure")

    monkeypatch.setattr(
        orchestration, "_run_bounded_suite_process", fail_before_outputs
    )
    monkeypatch.setattr(orchestration, "suite_environment", lambda: {})

    with pytest.raises(OSError, match="injected process creation failure"):
        orchestration.run_suite(paths, "focused", ["tests/example.py"])

    assert list(run_root.glob(".focused-suite-staged-*")) == []


def test_run_suite_cleanup_never_follows_replacement_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    asset_root, run_root = _prepared_roots(tmp_path)
    paths = orchestration.derive_paths(
        asset_root=asset_root, run_root=run_root, run_id="run-1"
    )
    displaced = run_root / "captured-suite-stage"
    replacement: Path | None = None
    rename_blocked = False

    def replace_stage_then_fail(
        command,
        *,
        environment,
        transcript_stage,
        transcript_fd,
        inherited_fds,
        timeout_seconds,
        max_transcript_bytes,
    ):
        del environment, inherited_fds, timeout_seconds, max_transcript_bytes
        nonlocal replacement, rename_blocked
        [stage] = run_root.glob(".focused-suite-staged-*")
        if os.name == "nt":
            with pytest.raises(OSError):
                stage.rename(displaced)
            rename_blocked = True
        else:
            stage.rename(displaced)
            stage.mkdir()
            replacement = stage
            (stage / "sentinel.txt").write_bytes(b"replacement\n")
        junit_arg = next(
            token for token in command if token.startswith("--junitxml=")
        )
        Path(junit_arg.partition("=")[2]).write_text(
            '<testsuite tests="1" failures="0" errors="0" skipped="0"/>',
            encoding="utf-8",
        )
        os.write(transcript_fd, b"one passed\n")
        return 0

    monkeypatch.setattr(
        orchestration, "_run_bounded_suite_process", replace_stage_then_fail
    )
    monkeypatch.setattr(orchestration, "suite_environment", lambda: {})

    if os.name == "nt":
        result = orchestration.run_suite(
            paths, "focused", ["tests/example.py"]
        )
        assert result["exit_code"] == 0
        assert rename_blocked is True
        assert not displaced.exists()
    else:
        with pytest.raises(
            orchestration.schema.TypedIngressError, match="identity"
        ):
            orchestration.run_suite(paths, "focused", ["tests/example.py"])
        assert replacement is not None and replacement.is_dir()
        assert (replacement / "sentinel.txt").read_bytes() == b"replacement\n"
        assert (displaced / "transcript.txt").read_bytes() == b"one passed\n"
        assert (displaced / "suite.xml").is_file()


def test_run_suite_never_publishes_through_replacement_receipts_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    asset_root, run_root = _prepared_roots(tmp_path)
    paths = orchestration.derive_paths(
        asset_root=asset_root, run_root=run_root, run_id="run-1"
    )
    displaced = run_root / "captured-receipts"
    decoy = run_root / "replacement-receipts"
    rename_blocked = False

    def replace_receipts_after_claim(
        command,
        *,
        environment,
        transcript_stage,
        transcript_fd,
        inherited_fds,
        timeout_seconds,
        max_transcript_bytes,
    ):
        del environment, inherited_fds, timeout_seconds, max_transcript_bytes
        nonlocal rename_blocked
        if os.name == "nt":
            with pytest.raises(OSError):
                paths.receipts_dir.rename(displaced)
            rename_blocked = True
        else:
            paths.receipts_dir.rename(displaced)
            paths.receipts_dir.mkdir()
            (paths.receipts_dir / "sentinel.txt").write_bytes(b"decoy\n")
        junit_arg = next(
            token for token in command if token.startswith("--junitxml=")
        )
        Path(junit_arg.partition("=")[2]).write_text(
            '<testsuite tests="1" failures="0" errors="0" skipped="0"/>',
            encoding="utf-8",
        )
        os.write(transcript_fd, b"one passed\n")
        return 0

    monkeypatch.setattr(
        orchestration,
        "_run_bounded_suite_process",
        replace_receipts_after_claim,
    )
    monkeypatch.setattr(orchestration, "suite_environment", lambda: {})

    if os.name == "nt":
        result = orchestration.run_suite(
            paths, "focused", ["tests/example.py"]
        )
        assert result["exit_code"] == 0
        assert rename_blocked is True
        assert not displaced.exists()
    else:
        with pytest.raises(
            orchestration.schema.TypedIngressError, match="identity"
        ):
            orchestration.run_suite(paths, "focused", ["tests/example.py"])
        paths.receipts_dir.rename(decoy)
        displaced.rename(paths.receipts_dir)
        assert (decoy / "sentinel.txt").read_bytes() == b"decoy\n"
        assert list(paths.receipts_dir.iterdir()) == []


def test_build_receipt_uses_captured_suite_bytes_not_mutable_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    asset_root, run_root = _prepared_roots(tmp_path)
    paths = orchestration.derive_paths(
        asset_root=asset_root, run_root=run_root, run_id="run-1"
    )

    def fake_run(
        command,
        *,
        environment,
        transcript_stage,
        transcript_fd,
        inherited_fds,
        timeout_seconds,
        max_transcript_bytes,
    ):
        del environment, inherited_fds, timeout_seconds, max_transcript_bytes
        junit_arg = next(
            token for token in command if token.startswith("--junitxml=")
        )
        Path(junit_arg.partition("=")[2]).write_text(
            '<testsuite tests="1" failures="0" errors="0" skipped="0"/>',
            encoding="utf-8",
        )
        os.write(transcript_fd, b"one passed\n")
        return 0

    monkeypatch.setattr(orchestration, "_run_bounded_suite_process", fake_run)
    monkeypatch.setattr(orchestration, "suite_environment", lambda: {})
    run_info = orchestration.run_suite(
        paths, "focused", ["tests/example.py"]
    )
    original_junit = run_info["junit_bytes"]
    original_transcript = run_info["transcript_bytes"]
    run_info["junit_path"].write_bytes(b"substituted\n")
    run_info["transcript_path"].write_bytes(b"substituted\n")

    receipt = orchestration.build_receipt(
        run_info,
        "a" * 64,
        "b" * 64,
        str(Path(sys.executable).resolve()),
        {"commit": "c" * 40, "tree": "d" * 40, "dirty": False},
    )

    assert receipt["counts"]["tests"] == 1
    assert receipt["junit_sha256"] == hashlib.sha256(original_junit).hexdigest()
    assert receipt["transcript_sha256"] == hashlib.sha256(
        original_transcript
    ).hexdigest()


@pytest.mark.parametrize("replaced_name", ("suite.xml", "transcript.txt"))
def test_run_suite_never_reads_replacement_child_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    replaced_name: str,
) -> None:
    asset_root, run_root = _prepared_roots(tmp_path)
    paths = orchestration.derive_paths(
        asset_root=asset_root, run_root=run_root, run_id="run-1"
    )
    replacement_blocked = False

    def replace_claimed_child(
        command,
        *,
        environment,
        transcript_stage,
        transcript_fd,
        inherited_fds,
        timeout_seconds,
        max_transcript_bytes,
    ):
        del environment, inherited_fds, timeout_seconds, max_transcript_bytes
        nonlocal replacement_blocked
        junit_arg = next(
            token for token in command if token.startswith("--junitxml=")
        )
        Path(junit_arg.partition("=")[2]).write_text(
            '<testsuite tests="1" failures="0" errors="0" skipped="0"/>',
            encoding="utf-8",
        )
        os.write(transcript_fd, b"one passed\n")
        [stage] = run_root.glob(".focused-suite-staged-*")
        target = stage / replaced_name
        try:
            target.unlink()
        except OSError:
            replacement_blocked = True
        else:
            target.write_bytes(b"replacement bytes\n")
        return 0

    monkeypatch.setattr(
        orchestration, "_run_bounded_suite_process", replace_claimed_child
    )
    monkeypatch.setattr(orchestration, "suite_environment", lambda: {})

    if os.name == "nt":
        result = orchestration.run_suite(
            paths, "focused", ["tests/example.py"]
        )
        assert replacement_blocked is True
        assert result["exit_code"] == 0
    else:
        with pytest.raises(
            orchestration.schema.TypedIngressError, match="claimed"
        ):
            orchestration.run_suite(paths, "focused", ["tests/example.py"])
        assert not (paths.receipts_dir / "focused_suite.xml").exists()
        assert not (
            paths.receipts_dir / "focused_suite_transcript.txt"
        ).exists()


@pytest.mark.skipif(os.name != "posix", reason="POSIX symlink substitution")
def test_run_suite_junit_symlink_never_redirects_pytest_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    asset_root, run_root = _prepared_roots(tmp_path)
    paths = orchestration.derive_paths(
        asset_root=asset_root, run_root=run_root, run_id="run-1"
    )
    outside = tmp_path / "outside.xml"
    outside.write_bytes(b"outside sentinel\n")

    def insert_symlink_before_write(
        command,
        *,
        environment,
        transcript_stage,
        transcript_fd,
        inherited_fds,
        timeout_seconds,
        max_transcript_bytes,
    ):
        del environment, inherited_fds, timeout_seconds, max_transcript_bytes
        [stage] = run_root.glob(".focused-suite-staged-*")
        (stage / "suite.xml").unlink()
        (stage / "suite.xml").symlink_to(outside)
        junit_arg = next(
            token for token in command if token.startswith("--junitxml=")
        )
        Path(junit_arg.partition("=")[2]).write_text(
            '<testsuite tests="1" failures="0" errors="0" skipped="0"/>',
            encoding="utf-8",
        )
        os.write(transcript_fd, b"one passed\n")
        return 0

    monkeypatch.setattr(
        orchestration,
        "_run_bounded_suite_process",
        insert_symlink_before_write,
    )
    monkeypatch.setattr(orchestration, "suite_environment", lambda: {})

    with pytest.raises(orchestration.schema.TypedIngressError):
        orchestration.run_suite(paths, "focused", ["tests/example.py"])

    assert outside.read_bytes() == b"outside sentinel\n"


def test_run_suite_rejects_replaced_frozen_receipts_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    asset_root, run_root = _prepared_roots(tmp_path)
    paths = orchestration.derive_paths(
        asset_root=asset_root, run_root=run_root, run_id="run-1"
    )
    run_snapshot = orchestration.phase4_finalize_release._capture_directory_chain(
        paths.run_root
    )
    receipts_snapshot = (
        orchestration.phase4_finalize_release._capture_directory_chain(
            paths.receipts_dir
        )
    )
    displaced = run_root / "original-receipts"
    paths.receipts_dir.rename(displaced)
    paths.receipts_dir.mkdir()
    sentinel = paths.receipts_dir / "sentinel.txt"
    sentinel.write_bytes(b"replacement\n")
    monkeypatch.setattr(orchestration, "suite_environment", lambda: {})

    with pytest.raises(orchestration.schema.TypedIngressError):
        orchestration.run_suite(
            paths,
            "focused",
            ["tests/example.py"],
            run_root_snapshot=run_snapshot,
            receipts_snapshot=receipts_snapshot,
        )

    assert sentinel.read_bytes() == b"replacement\n"
    assert list(displaced.iterdir()) == []


def test_captured_receipt_publication_rejects_replacement_parent(
    tmp_path: Path
) -> None:
    parent = tmp_path / "receipts"
    parent.mkdir()
    snapshot = orchestration.phase4_finalize_release._capture_directory_chain(
        parent
    )
    displaced = tmp_path / "original-receipts"
    parent.rename(displaced)
    parent.mkdir()
    sentinel = parent / "sentinel.txt"
    sentinel.write_bytes(b"replacement\n")

    with pytest.raises(orchestration.schema.TypedIngressError):
        orchestration.publish_bytes_to_captured_directory(
            parent,
            snapshot,
            parent / "suite_receipt_focused.json",
            b"{}\n",
            label="focused suite receipt",
        )

    assert sentinel.read_bytes() == b"replacement\n"
    assert list(displaced.iterdir()) == []


def test_certificate_publication_rejects_replaced_frozen_parent(
    tmp_path: Path,
) -> None:
    asset_root, run_root = _prepared_roots(tmp_path)
    paths = orchestration.derive_paths(
        asset_root=asset_root, run_root=run_root, run_id="run-1"
    )
    run_snapshot = orchestration.phase4_finalize_release._capture_directory_chain(
        paths.run_root
    )
    certificate_snapshot = (
        orchestration.phase4_finalize_release._capture_directory_chain(
            paths.certificate_dir
        )
    )
    displaced = run_root / "original-certificate"
    paths.certificate_dir.rename(displaced)
    paths.certificate_dir.mkdir()
    sentinel = paths.certificate_dir / "sentinel.txt"
    sentinel.write_bytes(b"replacement\n")

    with pytest.raises(orchestration.schema.TypedIngressError):
        orchestration.publish_certificate_bundle(
            paths,
            run_snapshot,
            certificate_snapshot,
            b"certificate\n",
            {"ready": True},
        )

    assert sentinel.read_bytes() == b"replacement\n"
    assert list(displaced.iterdir()) == []


@pytest.mark.skipif(os.name != "posix", reason="POSIX dir-fd regression")
@pytest.mark.parametrize(
    "swap_name",
    (
        "pre_run_ready_certificate_run-1.json",
        "certificate_generation_summary.json",
    ),
)
def test_certificate_pair_uses_one_held_generation_during_transient_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    swap_name: str,
) -> None:
    asset_root, run_root = _prepared_roots(tmp_path)
    paths = orchestration.derive_paths(
        asset_root=asset_root, run_root=run_root, run_id="run-1"
    )
    run_snapshot = orchestration.phase4_finalize_release._capture_directory_chain(
        paths.run_root
    )
    certificate_snapshot = (
        orchestration.phase4_finalize_release._capture_directory_chain(
            paths.certificate_dir
        )
    )
    displaced = run_root / "held-certificate"
    replacement = run_root / "replacement-certificate"
    real_open = orchestration.phase4_finalize_release.os.open
    swapped = False

    def swap_restore_around_claim(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal swapped
        if path == swap_name and dir_fd is not None and not swapped:
            swapped = True
            paths.certificate_dir.rename(displaced)
            paths.certificate_dir.mkdir()
            (paths.certificate_dir / "sentinel.txt").write_bytes(b"decoy\n")
            try:
                descriptor = real_open(
                    path, flags, mode, dir_fd=dir_fd
                )
            finally:
                paths.certificate_dir.rename(replacement)
                displaced.rename(paths.certificate_dir)
            return descriptor
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(
        orchestration.phase4_finalize_release.os,
        "open",
        swap_restore_around_claim,
    )
    certificate_bytes = b"certificate\n"
    summary = {"ready": True}

    orchestration.publish_certificate_bundle(
        paths,
        run_snapshot,
        certificate_snapshot,
        certificate_bytes,
        summary,
    )

    assert swapped is True
    assert paths.certificate_path.read_bytes() == certificate_bytes
    assert (
        paths.certificate_dir / "certificate_generation_summary.json"
    ).read_bytes() == orchestration._json_bytes(summary)
    assert [item.name for item in replacement.iterdir()] == ["sentinel.txt"]
    assert (replacement / "sentinel.txt").read_bytes() == b"decoy\n"


@pytest.mark.skipif(os.name != "nt", reason="Windows locking regression")
def test_windows_certificate_anchor_blocks_parent_rename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    asset_root, run_root = _prepared_roots(tmp_path)
    paths = orchestration.derive_paths(
        asset_root=asset_root, run_root=run_root, run_id="run-1"
    )
    run_snapshot = orchestration.phase4_finalize_release._capture_directory_chain(
        paths.run_root
    )
    certificate_snapshot = (
        orchestration.phase4_finalize_release._capture_directory_chain(
            paths.certificate_dir
        )
    )
    original = (
        orchestration.phase4_finalize_release._DirectoryAnchor.create_open_once
    )
    attempted = False

    def attempt_rename(anchor, name, **kwargs):
        nonlocal attempted
        if anchor.label == "frozen certificate directory" and not attempted:
            attempted = True
            with pytest.raises(OSError):
                paths.certificate_dir.rename(run_root / "displaced-certificate")
        return original(anchor, name, **kwargs)

    monkeypatch.setattr(
        orchestration.phase4_finalize_release._DirectoryAnchor,
        "create_open_once",
        attempt_rename,
    )
    orchestration.publish_certificate_bundle(
        paths,
        run_snapshot,
        certificate_snapshot,
        b"certificate\n",
        {"ready": True},
    )

    assert attempted is True
    assert paths.certificate_path.is_file()


def test_certificate_gather_reads_only_frozen_receipts_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipts = tmp_path / "receipts"
    receipts.mkdir()
    paths = {
        name: receipts / f"suite_receipt_{name}.json"
        for name in ("focused", "full")
    }
    expected = {name: {"source": name} for name in paths}
    for name, path in paths.items():
        path.write_bytes(orchestration._json_bytes(expected[name]))
    snapshot = orchestration.phase4_finalize_release._capture_directory_chain(
        receipts
    )
    displaced = tmp_path / "original-receipts"
    replacement = tmp_path / "replacement-receipts"
    rename_blocked = False

    def replace_then_gather(config):
        nonlocal rename_blocked
        if os.name == "nt":
            with pytest.raises(OSError):
                receipts.rename(displaced)
            rename_blocked = True
        else:
            receipts.rename(displaced)
            receipts.mkdir()
            for name in paths:
                (receipts / f"suite_receipt_{name}.json").write_bytes(
                    orchestration._json_bytes({"source": "substituted"})
                )
        return {
            "suite_receipts": {
                name: orchestration.schema.parse_json_bytes_strict(
                    orchestration.schema.read_regular_file_bytes(Path(path))
                )
                for name, path in config["suite_receipt_paths"].items()
            }
        }

    monkeypatch.setattr(
        orchestration.phase4,
        "gather_certificate_components",
        replace_then_gather,
    )

    if os.name == "nt":
        gathered = orchestration.gather_certificate_from_captured_receipts(
            {"suite_receipt_paths": paths}, receipts, snapshot, expected
        )
        assert gathered["suite_receipts"] == {
            "focused": {"source": "focused"},
            "full": {"source": "full"},
        }
        assert rename_blocked is True
        assert not displaced.exists()
    else:
        with pytest.raises(
            orchestration.schema.TypedIngressError, match="identity"
        ):
            orchestration.gather_certificate_from_captured_receipts(
                {"suite_receipt_paths": paths}, receipts, snapshot, expected
            )
        receipts.rename(replacement)
        displaced.rename(receipts)
        assert {
            orchestration.schema.parse_json_bytes_strict(path.read_bytes())[
                "source"
            ]
            for path in replacement.iterdir()
        } == {"substituted"}


def test_certificate_gather_rejects_replaced_receipt_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipts = tmp_path / "receipts"
    receipts.mkdir()
    paths = {
        name: receipts / f"suite_receipt_{name}.json"
        for name in ("focused", "full")
    }
    expected = {name: {"source": name} for name in paths}
    for name, path in paths.items():
        path.write_bytes(orchestration._json_bytes(expected[name]))
    snapshot = orchestration.phase4_finalize_release._capture_directory_chain(
        receipts
    )

    def replace_child_then_gather(config):
        paths["focused"].unlink()
        paths["focused"].write_bytes(
            orchestration._json_bytes({"source": "substituted"})
        )
        return {
            "suite_receipts": {
                name: orchestration.schema.parse_json_bytes_strict(
                    orchestration.schema.read_regular_file_bytes(Path(path))
                )
                for name, path in config["suite_receipt_paths"].items()
            }
        }

    monkeypatch.setattr(
        orchestration.phase4,
        "gather_certificate_components",
        replace_child_then_gather,
    )

    with pytest.raises(RuntimeError, match="substituted|changed"):
        orchestration.gather_certificate_from_captured_receipts(
            {"suite_receipt_paths": paths},
            receipts,
            snapshot,
            expected,
        )


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
        transcript_fd,
        inherited_fds,
        timeout_seconds,
        max_transcript_bytes,
    ):
        del environment, inherited_fds, timeout_seconds, max_transcript_bytes
        junit_arg = next(
            token for token in command if token.startswith("--junitxml=")
        )
        Path(junit_arg.partition("=")[2]).write_text(
            '<testsuite tests="1" failures="0" errors="0" skipped="0"/>',
            encoding="utf-8",
        )
        os.write(transcript_fd, b"one passed\n")
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
