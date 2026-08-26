"""Regression tests for pull-request checkout identity in CI."""
from __future__ import annotations

import shlex
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from reproducibility.colm_aims_2026 import phase4, receipt
from scripts import run_ci_suite_with_receipt


REPO = Path(__file__).resolve().parents[1]
WORKFLOW_CASES = (
    (
        "python-app.yml",
        "build",
        (
            "pip install -e '.[dev]'",
            "flake8 . --count --select=E9,F63,F7,F82",
            "--name full -- pytest tests/ -q -p no:cacheprovider",
        ),
    ),
    (
        "audit-artifacts.yml",
        "audit-artifacts",
        (
            "pip install -e '.[dev]'",
            "python scripts/verify_audit_release.py",
            "--name audit -- pytest tests/test_verify_audit_release.py -q",
        ),
    ),
)


def _load_workflow(name: str) -> dict[str, Any]:
    path = REPO / ".github" / "workflows" / name
    loaded = yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    assert isinstance(loaded, dict)
    return loaded


def _step_named(job: dict[str, Any], name: str) -> dict[str, Any]:
    matches = [step for step in job["steps"] if step.get("name") == name]
    assert len(matches) == 1
    return matches[0]


def _checkout_step(job: dict[str, Any]) -> dict[str, Any]:
    matches = [
        step
        for step in job["steps"]
        if step.get("uses", "").startswith("actions/checkout@")
    ]
    assert len(matches) == 1
    return matches[0]


def _run_commands(job: dict[str, Any]) -> str:
    return "\n".join(step.get("run", "") for step in job["steps"])


def _receipt_pytest_command(step: dict[str, Any]) -> tuple[str, list[str]]:
    """Extract the suite name and pytest argv from one workflow producer step."""
    argv = shlex.split(step["run"])
    separator = argv.index("--")
    name = argv[argv.index("--name") + 1]
    return name, argv[separator + 1 :]


@pytest.mark.parametrize(
    ("workflow_name", "merge_job_name", "required_commands"),
    WORKFLOW_CASES,
)
def test_pr_workflow_checks_merge_ref_and_literal_head(
    workflow_name: str,
    merge_job_name: str,
    required_commands: tuple[str, ...],
) -> None:
    """Keep integration coverage and exact-head evidence as separate lanes."""
    workflow = _load_workflow(workflow_name)
    triggers = workflow["on"]
    assert triggers["pull_request"]["branches"] == ["main"]
    assert "pull_request_target" not in triggers
    assert workflow["permissions"] == {"contents": "read"}

    merge_job = workflow["jobs"][merge_job_name]
    merge_checkout = _checkout_step(merge_job)
    assert merge_checkout.get("with", {}).get("ref") is None
    assert merge_checkout["with"]["persist-credentials"] == "false"
    merge_identity = _step_named(merge_job, "Verify merge-ref checkout")
    assert merge_identity["if"] == "github.event_name == 'pull_request'"
    assert merge_identity["env"]["EXPECTED_SHA"] == "${{ github.sha }}"
    assert "git rev-parse HEAD" in merge_identity["run"]
    assert "refs/pull/*/merge" in merge_identity["run"]

    head_job = workflow["jobs"]["literal-head"]
    assert head_job["if"] == "github.event_name == 'pull_request'"
    head_checkout = _checkout_step(head_job)
    assert head_checkout["with"]["ref"] == (
        "${{ github.event.pull_request.head.sha }}"
    )
    assert head_checkout["with"]["persist-credentials"] == "false"
    head_identity = _step_named(head_job, "Verify literal-head checkout")
    assert head_identity["env"]["EXPECTED_SHA"] == (
        "${{ github.event.pull_request.head.sha }}"
    )
    assert "git rev-parse HEAD" in head_identity["run"]

    for command in required_commands:
        assert command in _run_commands(merge_job)
        assert command in _run_commands(head_job)
    if workflow_name == "python-app.yml":
        for job in (merge_job, head_job):
            for step_name, expected_name in (
                ("Focused Phase-4 suite with R-070 receipt", "focused"),
                ("Full suite with R-070 receipt", "full"),
            ):
                suite_name, pytest_argv = _receipt_pytest_command(
                    _step_named(job, step_name)
                )
                assert suite_name == expected_name
                command = run_ci_suite_with_receipt._pytest_command(
                    pytest_argv, REPO / "ci-evidence" / f"{suite_name}.xml"
                )
                assert phase4.suite_command_failures(suite_name, command) == []

    merge_upload = _step_named(
        merge_job,
        "Retain suite evidence"
        if workflow_name == "python-app.yml"
        else "Retain audit evidence",
    )
    direct_upload = _step_named(
        merge_job,
        "Retain direct-head suite evidence"
        if workflow_name == "python-app.yml"
        else "Retain direct-head audit evidence",
    )
    assert merge_upload["uses"] == "actions/upload-artifact@v4"
    assert merge_upload["if"] == (
        "always() && github.event_name == 'pull_request'"
    )
    assert "merge-ref-evidence" in merge_upload["with"]["name"]
    assert direct_upload["uses"] == "actions/upload-artifact@v4"
    assert direct_upload["if"] == (
        "always() && github.event_name != 'pull_request'"
    )
    assert "direct-head-evidence" in direct_upload["with"]["name"]

    head_uploads = [
        step
        for step in head_job["steps"]
        if step.get("uses") == "actions/upload-artifact@v4"
    ]
    assert len(head_uploads) == 1
    assert head_uploads[0]["if"] == "always()"
    assert "literal-head-evidence" in head_uploads[0]["with"]["name"]


def test_r070_producer_argv_matches_certificate_contract(tmp_path) -> None:
    assert set(receipt.SUITE_RECEIPT_REQUIRED_FIELDS) == set(
        phase4.R070_RECEIPT_FIELDS
    )
    interpreter = str(
        Path(run_ci_suite_with_receipt.sys.executable).resolve()
    )
    for name, selection in (
        ("focused", phase4.FOCUSED_SUITE_SELECTION),
        ("full", phase4.FULL_SUITE_SELECTION),
    ):
        command = run_ci_suite_with_receipt._pytest_command(
            ["pytest", *selection, "-q", "-p", "no:cacheprovider"],
            tmp_path / f"{name}.xml",
        )
        assert command[0] == interpreter
        assert phase4.suite_command_failures(name, command) == []


def test_r070_producer_reuses_canonical_environment_lock(monkeypatch) -> None:
    expected = b"numpy==2.4.6\n"
    observed = []

    def canonical_probe(interpreter):
        observed.append(Path(interpreter))
        return expected

    monkeypatch.setattr(
        run_ci_suite_with_receipt.phase4_launcher,
        "_default_probe_environment_lock",
        canonical_probe,
    )
    assert run_ci_suite_with_receipt._environment_lock_bytes() == expected
    assert observed == [
        Path(run_ci_suite_with_receipt.sys.executable).resolve()
    ]


def test_r070_git_identity_is_bound_to_repository_root(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    observed = Path(
        run_ci_suite_with_receipt._git("rev-parse", "--show-toplevel")
    ).resolve()
    assert observed == REPO.resolve()


@pytest.mark.parametrize("junit_bytes", [None, b"<not-xml"])
def test_r070_main_rejects_missing_or_malformed_junit(
    tmp_path, monkeypatch, junit_bytes
) -> None:
    monkeypatch.chdir(tmp_path)
    workflow = tmp_path / "workflow.yml"
    workflow.write_text("name: test\n", encoding="utf-8")
    output_dir = tmp_path / "evidence"

    def fake_git(*args):
        if args[:2] == ("rev-parse", "HEAD"):
            return "a" * 40
        if args[:2] == ("rev-parse", "HEAD^{tree}"):
            return "b" * 40
        if args[:1] == ("status",):
            return ""
        raise AssertionError(args)

    def fake_run(command, **kwargs):
        assert Path(kwargs["cwd"]).resolve() == REPO.resolve()
        junit_arg = next(
            part for part in command if part.startswith("--junitxml=")
        )
        if junit_bytes is not None:
            Path(junit_arg.partition("=")[2]).write_bytes(junit_bytes)
        return SimpleNamespace(returncode=0, stdout=b"pytest output\n")

    monkeypatch.setattr(run_ci_suite_with_receipt, "_git", fake_git)
    monkeypatch.setattr(
        run_ci_suite_with_receipt, "_environment_lock_bytes", lambda: b"lock\n"
    )
    monkeypatch.setattr(run_ci_suite_with_receipt.subprocess, "run", fake_run)
    rc = run_ci_suite_with_receipt.main(
        [
            "--workflow",
            str(workflow),
            "--output-dir",
            str(output_dir),
            "--name",
            "focused",
            "--",
            "pytest",
            *phase4.FOCUSED_SUITE_SELECTION,
            "-q",
            "-p",
            "no:cacheprovider",
        ]
    )
    assert rc == 4
    assert (output_dir / "focused.receipt.json").is_file()


@pytest.mark.parametrize("drift", ["dirty", "head"])
def test_r070_main_rejects_dirty_or_head_drift(
    tmp_path, monkeypatch, drift
) -> None:
    workflow = tmp_path / "workflow.yml"
    workflow.write_text("name: test\n", encoding="utf-8")
    calls = {"head": 0, "status": 0}

    def fake_git(*args):
        if args[:2] == ("rev-parse", "HEAD"):
            calls["head"] += 1
            if drift == "head" and calls["head"] == 2:
                return "c" * 40
            return "a" * 40
        if args[:2] == ("rev-parse", "HEAD^{tree}"):
            return "b" * 40
        if args[:1] == ("status",):
            calls["status"] += 1
            return " M tracked.py" if drift == "dirty" else ""
        raise AssertionError(args)

    def fake_run(command, **_kwargs):
        junit_arg = next(
            part for part in command if part.startswith("--junitxml=")
        )
        Path(junit_arg.partition("=")[2]).write_bytes(
            b'<testsuite tests="1" failures="0" errors="0" skipped="0">'
            b'<testcase classname="tests.test_x" name="test_ok"/>'
            b"</testsuite>"
        )
        return SimpleNamespace(returncode=0, stdout=b"pytest output\n")

    monkeypatch.setattr(run_ci_suite_with_receipt, "_git", fake_git)
    monkeypatch.setattr(
        run_ci_suite_with_receipt, "_environment_lock_bytes", lambda: b"lock\n"
    )
    monkeypatch.setattr(run_ci_suite_with_receipt.subprocess, "run", fake_run)
    with pytest.raises(receipt.SuiteReceiptError, match="dirty"):
        run_ci_suite_with_receipt.main(
            [
                "--workflow",
                str(workflow),
                "--output-dir",
                str(tmp_path / "evidence"),
                "--name",
                "focused",
                "--",
                "pytest",
                *phase4.FOCUSED_SUITE_SELECTION,
                "-q",
                "-p",
                "no:cacheprovider",
            ]
        )
