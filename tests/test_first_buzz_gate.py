"""Verify the first-buzz comparator stages exactly the reviewed Git objects."""

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml


FORMAL = Path(__file__).resolve().parents[1] / "formal"
GATE = Path("formal/first_buzz")
FROZEN = tuple(json.loads((FORMAL / "first_buzz/freeze.json").read_bytes()))


def test_workflow_checksums_match_shipped_inputs():
    workflow = FORMAL.parent / ".github/workflows/first-buzz.yml"
    steps = yaml.safe_load(workflow.read_text(encoding="utf-8"))["jobs"]["compare"]["steps"]
    commands = [line.strip() for step in steps for line in step.get("run", "").splitlines()]
    verifier = hashlib.sha256((FORMAL / "first_buzz/verify.py").read_bytes()).hexdigest()
    freeze = hashlib.sha256((FORMAL / "first_buzz/freeze.json").read_bytes()).hexdigest()

    assert f'echo "{verifier}  $RUNNER_TEMP/first-buzz.py" | sha256sum -c -' in commands
    assert f"--freeze-sha256 {freeze}" in commands


def git(repo, *args):
    return subprocess.check_output(
        [
            "git", "-c", "core.autocrlf=false", "-c", "commit.gpgSign=false",
            "-c", "user.name=test", "-c", "user.email=test@example.org", *args,
        ],
        cwd=repo,
        text=True,
    ).strip()


@pytest.fixture
def reviewed(tmp_path):
    repo = tmp_path / "repo"
    (repo / GATE).mkdir(parents=True)
    for name in (*FROZEN, "freeze.json"):
        shutil.copyfile(FORMAL / "first_buzz" / name, repo / GATE / name)
    shutil.copyfile(FORMAL / "FirstBuzz.lean", repo / "formal/FirstBuzz.lean")
    git(repo, "init", "-q")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "reviewed inputs")
    revision = git(repo, "rev-parse", "HEAD")
    digest = hashlib.sha256((repo / GATE / "freeze.json").read_bytes()).hexdigest()
    return repo, revision, digest


@pytest.fixture
def reject_at_cli(monkeypatch, capsys):
    from formal.first_buzz import verify

    # A premature Lake lookup or comparator subprocess fails the test on any OS.
    popen = subprocess.Popen

    def git_only(args, *positional, **kwargs):
        assert Path(args[0]).stem.lower() == "git", args
        return popen(args, *positional, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", git_only)

    def reject(repo, revision, digest, reason):
        monkeypatch.setattr(
            sys, "argv", ["verify.py", "--repo", str(repo), "--revision", revision,
                          "--freeze-sha256", digest],
        )
        with pytest.raises(SystemExit) as raised:
            verify.main()
        assert raised.value.code == 1
        assert reason in capsys.readouterr().err

    return reject


@pytest.mark.parametrize("name", FROZEN)
def test_changed_frozen_blob_is_rejected(reviewed, reject_at_cli, name):
    repo, _, digest = reviewed
    with (repo / GATE / name).open("ab") as blob:
        blob.write(b"\n")
    git(repo, "commit", "-qam", "modified frozen input")
    reject_at_cli(repo, git(repo, "rev-parse", "HEAD"), digest, name)


def test_wrong_external_pin_is_rejected(reviewed, reject_at_cli):
    repo, revision, _ = reviewed
    reject_at_cli(repo, revision, "0" * 64, "freeze")


def test_missing_solution_is_rejected(reviewed, reject_at_cli):
    repo, _, digest = reviewed
    (repo / "formal/FirstBuzz.lean").unlink()
    git(repo, "commit", "-qam", "removed solution")
    reject_at_cli(repo, git(repo, "rev-parse", "HEAD"), digest, "formal/FirstBuzz.lean")


def test_mutable_revision_is_rejected(reviewed, reject_at_cli):
    repo, _, digest = reviewed
    reject_at_cli(repo, "HEAD", digest, "commit")


def test_stage_uses_commit_despite_dirty_worktree(reviewed, tmp_path):
    from formal.first_buzz.verify import stage_project

    repo, revision, digest = reviewed
    solution = repo / "formal/FirstBuzz.lean"
    proof = solution.read_bytes()
    solution.write_bytes(b"uncommitted solution")
    (repo / GATE / "freeze.json").unlink()
    for name in FROZEN:
        (repo / GATE / name).write_bytes(b"uncommitted input")
    (repo / GATE / "Unexpected.lean").write_bytes(b"untracked file")

    project = tmp_path / "project"
    stage_project(repo, revision, digest, project)

    assert (project / "FirstBuzz.lean").read_bytes() == proof
    for name in FROZEN:
        assert (project / name).read_bytes() == (FORMAL / "first_buzz" / name).read_bytes()
    assert not (project / "Unexpected.lean").exists()
