"""Regression tests for pull-request checkout identity in CI."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml


REPO = Path(__file__).resolve().parents[1]
WORKFLOW_CASES = (
    (
        "python-app.yml",
        "build",
        (
            "pip install -e '.[dev]'",
            "flake8 . --count --select=E9,F63,F7,F82",
            "pytest tests/ -q --tb=short",
        ),
    ),
    (
        "audit-artifacts.yml",
        "audit-artifacts",
        (
            "pip install -e '.[dev]'",
            "python scripts/verify_audit_release.py",
            "pytest tests/test_verify_audit_release.py -q",
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
