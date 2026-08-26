"""Regression tests for pull-request checkout identity in CI."""
from __future__ import annotations

from pathlib import Path
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
            focused = _step_named(
                job, "Focused Phase-4 suite with R-070 receipt"
            )["run"]
            for selection in phase4.FOCUSED_SUITE_SELECTION:
                assert selection in focused
    for job in (merge_job, head_job):
        assert any(
            step.get("uses") == "actions/upload-artifact@v4"
            and step.get("if") == "always()"
            for step in job["steps"]
        )


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
