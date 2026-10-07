"""Read-only checks for the gated two-checkout Mistral evaluation workflow."""
from pathlib import Path
import re
import subprocess

import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/imcqa-mistral-evaluation.yml"


def parsed():
    return yaml.safe_load(WORKFLOW.read_text())


def test_evaluation_requires_exact_one_shot_trigger():
    workflow = parsed()
    trigger = workflow.get("on", workflow.get(True))
    assert set(trigger) == {"push"}
    assert trigger["push"]["branches"] == ["feat/imcqa-mistral-replication-20261007"]
    assert trigger["push"]["paths"] == [".github/workflows/imcqa-mistral-evaluation.yml"]
    gate = workflow["jobs"]["evaluation"]["if"]
    for required in ("github.event_name == 'push'", "github.actor == 'ankaggarwal94'",
                     "github.run_attempt == 1",
                     "github.event.head_commit.message == 'ops: run approved eight-dollar frozen Mistral evaluation once'"):
        assert required in gate
    assert workflow["permissions"] == {"contents": "read"}
    assert workflow["concurrency"] == {"group": "imcqa-mistral-eight-usd-v1", "cancel-in-progress": False}


def test_frozen_source_and_launch_inputs_use_separate_checkouts():
    workflow = parsed()
    assert re.fullmatch(r"[0-9a-f]{40}", workflow["env"]["MISTRAL_SOURCE_COMMIT"])
    job = workflow["jobs"]["evaluation"]
    assert job["defaults"]["run"]["working-directory"] == "source"
    checkouts = [s for s in job["steps"] if s.get("uses", "").startswith("actions/checkout@")]
    assert len(checkouts) == 2
    assert checkouts[0]["with"] == {"ref": "${{ github.sha }}", "path": "launch", "persist-credentials": False}
    assert checkouts[1]["with"] == {"ref": "${{ env.MISTRAL_SOURCE_COMMIT }}", "path": "source", "persist-credentials": False}
    for step in job["steps"]:
        if "uses" in step:
            assert re.fullmatch(r"actions/[a-z-]+@[0-9a-f]{40}", step["uses"])


def test_one_evaluation_allocation_no_cache_creation_or_retry():
    job = parsed()["jobs"]["evaluation"]
    assert job["timeout-minutes"] == 170
    paid = [s for s in job["steps"] if "MODAL_TOKEN_SECRET" in s.get("env", {})]
    assert len(paid) == 2
    assert paid[0]["id"] == "score"
    assert paid[0]["timeout-minutes"] == 150
    assert paid[0]["run"].count("score-stage --stage evaluation") == 1
    assert "--dry-run" not in paid[0]["run"]
    assert "cleanup-cache" in paid[1]["run"]
    assert paid[1]["timeout-minutes"] == 3
    assert "always()" in paid[1]["if"]
    assert "steps.score.outcome" not in paid[1]["if"]
    assert "prepare-cache" not in WORKFLOW.read_text()
    assert job["steps"][-1]["if"] == "always()"
    assert job["steps"][-1]["with"]["path"] == "imcqa_mistral_eval_artifacts/"


def test_manifest_lock_and_source_validation_precede_credentials():
    steps = parsed()["jobs"]["evaluation"]["steps"]
    binding = next(s for s in steps if s.get("id") == "bind")
    program = binding["run"].split("python - <<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]
    compile(program, "evaluation-workflow-binding", "exec")
    for required in ("runner.verify_sources(Path.cwd(), cache, committed=True)",
                     "runner.validate_prepare", "evaluation_template_sha256",
                     "design.validate_public_package(public)",
                     "validate_policy_lock(lock, evaluation_package=public)",
                     "design.validate_stage_manifest", "development_receipt=development",
                     "design.file_hash(path) != manifest['public_input_sha256']"):
        assert required in program
    first_paid = next(i for i, s in enumerate(steps) if "MODAL_TOKEN_SECRET" in s.get("env", {}))
    assert steps.index(binding) < first_paid
    assert any("--dry-run" in s.get("run", "") for s in steps[steps.index(binding) + 1:first_paid])
    assert all("MODAL_TOKEN_ID" not in s.get("env", {}) for s in steps[:first_paid])


def test_all_five_launch_files_are_bound_and_public_transport_is_frozen():
    steps = parsed()["jobs"]["evaluation"]["steps"]
    binding = next(s for s in steps if s.get("id") == "bind")["run"]
    for name in ("cache_control.json", "prepare_receipt.json", "policy_lock.json",
                 "development_receipt.json", "evaluation_stage_manifest.json"):
        assert name in binding
    reconstruct = next(s["run"] for s in steps if "--transport-root" in s.get("run", ""))
    assert "--transport-root ." in reconstruct
    assert "--public-bindings imcqa_mistral_public/public_bindings.json" in reconstruct
    assert "../launch" not in reconstruct
    assert not any("evaluator.json" in s.get("run", "") for s in steps)


def test_frozen_and_launch_test_packages_run_in_separate_processes():
    job = parsed()["jobs"]["evaluation"]
    steps = job["steps"]
    test_steps = [s for s in steps if "python -m pytest" in s.get("run", "")]
    assert len(test_steps) == 2
    frozen, launch = test_steps
    assert frozen.get("working-directory", job["defaults"]["run"]["working-directory"]) == "source"
    assert launch["working-directory"] == "launch"
    assert "tests/test_imcqa_mistral_evaluation_workflow.py" not in frozen["run"]
    assert "../launch" not in frozen["run"]
    assert launch["run"].split() == ["python", "-m", "pytest", "--noconftest",
                                     "tests/test_imcqa_mistral_evaluation_workflow.py", "-q"]
    first_paid = next(i for i, s in enumerate(steps) if "MODAL_TOKEN_SECRET" in s.get("env", {}))
    assert all(steps.index(step) < first_paid for step in test_steps)


def test_every_workflow_shell_block_parses_without_execution():
    for step in parsed()["jobs"]["evaluation"]["steps"]:
        if "run" in step:
            result = subprocess.run(["bash", "-n"], input=step["run"], text=True, capture_output=True)
            assert result.returncode == 0, (step["name"], result.stderr)
