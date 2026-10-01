"""Pure guards: no Modal import, GPU model load, or paid provider calls."""
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import modal_jane_pilot as pilot


def job(fmt="oe", *, qid="dev", split="calibration"):
    prompt = f"Question prefix for {qid}"
    return {"job_id": f"{qid}-{fmt}", "qid": qid, "group_id": qid, "split": split,
            "format": fmt, "condition": "oe" if fmt == "oe" else "sampled_pool",
            "menu_id": None if fmt == "oe" else "fixed_1", "prefix_id": "p1", "fraction": 0.2,
            "prompt": prompt, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()}


def package(jobs):
    return {"schema_version": "jane-public-jobs-v1", "evidence_scope": "engineering_smoke", "jobs": jobs}


def trace(jobs, answer="France"):
    rows = []
    for j in jobs:
        value = {"answer": "A" if j["format"] == "mc" else answer, "confidence": 0.75, "status": "answer"}
        rows.append({"job_id": j["job_id"], "prompt_sha256": j["prompt_sha256"],
                     **value, "raw_response": json.dumps(value)})
    return {"predictions": rows}


@pytest.mark.parametrize("value", ["nan", "NaN", "Infinity", "-1", "0", "2", "10.01", "100"])
def test_budget_rejects_before_provider_import(value):
    with pytest.raises(ValueError):
        pilot.budget_plan(value)


def test_ten_dollar_allocation_includes_non_gpu_resources_and_reserve():
    plan = pilot.budget_plan("10")
    assert plan["max_session_seconds"] == 12000
    assert Decimal(plan["max_allocation_plus_reserve_usd"]) == Decimal("9.67088")
    assert Decimal(plan["allocation_rate_usd_per_second"]) > pilot.GPU_RATE
    assert plan["cpu_request_and_limit"] == [2, 2]
    assert plan["memory_request_and_limit_mib"] == [32768, 32768]


@pytest.mark.parametrize("key", ["answerline", "gold_option_id", "accepted_answers", "future_question", "labels"])
def test_unexpected_evaluator_data_rejected(key):
    p = package([job()])
    p["jobs"][0][key] = "forbidden"
    with pytest.raises(ValueError, match="unexpected"):
        pilot.validate_public_package(p)


def test_public_prompt_hash_must_match():
    p = package([job()])
    p["jobs"][0]["prompt"] += " modified"
    with pytest.raises(ValueError, match="hash"):
        pilot.validate_public_package(p)


def test_interface_gate_validates_both_formats_without_gold():
    jobs = [job("oe"), job("mc")]
    gate = pilot.interface_gate(jobs, trace(jobs))
    assert gate["passed"] and not gate["uses_gold_accuracy"]


@pytest.mark.parametrize("answer", ["...", "…", "answer here", "your answer"])
def test_placeholder_copies_abort_main(answer):
    jobs = [job("oe"), job("mc")]
    assert not pilot.interface_gate(jobs, trace(jobs, answer))["passed"]


def test_one_illegal_mc_id_aborts_even_with_perfect_schema():
    jobs = [job("oe"), job("mc")]
    t = trace(jobs)
    t["predictions"][1]["answer"] = "France"
    raw = {k: t["predictions"][1][k] for k in ("answer", "confidence", "status")}
    t["predictions"][1]["raw_response"] = json.dumps(raw)
    assert not pilot.interface_gate(jobs, t)["passed"]


def test_missing_development_format_fails_gate():
    jobs = [job()]
    assert not pilot.interface_gate(jobs, trace(jobs))["passed"]


def test_valid_abstention_counts_toward_schema_gate():
    jobs = [job("oe"), job("mc")]
    t = trace(jobs)
    for row in t["predictions"]:
        row.update(answer=None, confidence=None, status="abstain")
        row["raw_response"] = json.dumps({k: row[k] for k in ("answer", "confidence", "status")})
    assert pilot.interface_gate(jobs, t)["passed"]


def test_confidence_nan_does_not_pass_exact_parsing():
    jobs = [job("oe"), job("mc")]
    t = trace(jobs)
    t["predictions"][0]["raw_response"] = '{"answer":"France","confidence":NaN,"status":"answer"}'
    t["predictions"][0].update(answer=None, confidence=None, status="invalid")
    assert not pilot.interface_gate(jobs, t)["passed"]


def test_raw_and_parsed_mismatch_is_integrity_failure():
    jobs = [job("oe"), job("mc")]
    t = trace(jobs)
    t["predictions"][0]["answer"] = "altered"
    with pytest.raises(ValueError, match="raw parsing"):
        pilot.interface_gate(jobs, t)


def test_throughput_reserves_both_models_and_includes_choices_controls():
    packages = {"main_jobs.json": {"jobs": [{}] * 100}, "main_choices_only.json": {"jobs": [{}] * 20}}
    measured = {tag: {"inference_seconds": 10.0, "jobs": 10} for tag, _, _ in pilot.MODELS}
    gate = pilot.throughput_gate(measured, packages, 3000)
    assert gate["predicted_seconds_with_load_reserve"] == 2520
    assert gate["passed"]
    assert not pilot.throughput_gate(measured, packages, 2500)["passed"]


def test_throughput_does_not_scale_one_time_model_download_with_job_count():
    timing = pilot.phase_timing({"metadata": {"total_seconds": 130, "model_load_seconds": 100}}, 131)
    assert timing["inference_seconds"] == 30
    measured = {tag: {**timing, "jobs": 177} for tag, _, _ in pilot.MODELS}
    packages = {"main_jobs.json": {"jobs": [{}] * 2856}, "main_choices_only.json": {"jobs": [{}] * 400}}
    gate = pilot.throughput_gate(measured, packages, 12000)
    assert gate["passed"]
    assert gate["predicted_seconds_with_load_reserve"] == pytest.approx(5111.186440677966)


def test_checkpoint_closes_before_commit_and_reopens_append(tmp_path):
    path = tmp_path / "predictions.jsonl"
    with pilot.BatchCheckpoint(path) as checkpoint:
        checkpoint.write('first\n')
        checkpoint.flush()
        assert checkpoint.fileno() >= 0
        checkpoint.close_batch()
        assert checkpoint.stream is None
        checkpoint.write('second\n')
        checkpoint.close_batch()
        assert checkpoint.stream is None
    assert path.read_text() == "first\nsecond\n"
    with pytest.raises(FileExistsError):
        pilot.BatchCheckpoint(path)


def test_workspace_receipt_excludes_token_and_user_identity():
    response = SimpleNamespace(workspace_name="ankaggarwal94", workspace_id="workspace-test",
                               token_id="DO_NOT_EMIT", user_identity="DO_NOT_EMIT")
    assert pilot.workspace_identity(response) == {"workspace_name": "ankaggarwal94", "workspace_id": "workspace-test"}
    response.workspace_name = "different-account"
    with pytest.raises(ValueError, match="unexpected workspace"):
        pilot.workspace_identity(response)


def test_local_receipt_is_create_once(tmp_path):
    path = tmp_path / "proof.json"
    pilot._write_once(path, {"first": True})
    with pytest.raises(FileExistsError):
        pilot._write_once(path, {"first": False})
    assert json.loads(path.read_text()) == {"first": True}


def test_frozen_workflow_limits_trigger_secrets_and_attempts():
    source = (Path(__file__).resolve().parents[1] / ".github/workflows/jane-modal-pilot.yml").read_text()
    assert f"branches: [{pilot.BRANCH}]" in source
    assert "paths: [.github/workflows/jane-modal-pilot.yml]" in source
    assert "github.run_attempt == 1" in source
    assert "github.actor == 'ankaggarwal94'" in source
    assert pilot.LAUNCH_MESSAGE in source
    assert "ref: ${{ github.sha }}" in source
    assert "persist-credentials: false" in source
    assert "contents: read" in source
    assert "pull_request" not in source and "workflow_dispatch" not in source
    assert "python -m scripts.modal_jane_pilot launch" in source
    assert "python scripts/modal_jane_pilot.py launch" not in source
    assert source.count("secrets.MODAL_TOKEN_ID") == 1
    assert source.count("secrets.MODAL_TOKEN_SECRET") == 1
    assert "--budget-usd 10" in source and "--detach" not in source


def test_pinned_sdk_remote_import_matches_the_only_image_module():
    function_utils = pytest.importorskip("modal._utils.function_utils")
    assert function_utils.FunctionSourceInfo(pilot.remote_pilot).module_name == "scripts.modal_jane_pilot"


def test_image_source_allowlist_excludes_evaluator_and_data():
    assert set(pilot.SOURCE_FILES) == {"scripts/__init__.py", "scripts/modal_jane_pilot.py",
                                    "scripts/jane_gpu_backend.py", "scripts/jane_qwen_backend.py"}
    assert all("data" not in path and "evaluation" not in path for path in pilot.SOURCE_FILES)
