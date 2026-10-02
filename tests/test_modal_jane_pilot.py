"""Pure guards: no Modal import, GPU model load, or paid provider calls."""
from copy import deepcopy
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


@pytest.mark.parametrize("value", ["nan", "NaN", "Infinity", "-1", "0", "2", "9.35", "10", "10.01", "100"])
def test_budget_rejects_before_provider_import(value):
    with pytest.raises(ValueError):
        pilot.budget_plan(value)


def test_remaining_allocation_includes_three_prior_attempts_and_one_global_reserve():
    plan = pilot.budget_plan("9.34")
    assert plan["max_session_seconds"] == 11482
    assert Decimal(plan["cumulative_max_estimate_plus_reserves_usd"]) <= Decimal("10")
    assert Decimal(plan["max_allocation_plus_reserve_usd"]) <= Decimal("9.34")
    measured_prior = sum((pilot.ALLOCATION_RATE * Decimal(attempt["host_session_seconds"])
                          for attempt in pilot.PRIOR_ATTEMPTS), Decimal(0))
    assert measured_prior == Decimal("0.6490735711451468080380")
    assert Decimal(plan["prior_whole_host_allocation_estimate_usd"]) == measured_prior
    assert measured_prior <= Decimal(plan["prior_allocation_debit_usd"])
    assert Decimal(plan["reserve_usd"]) == Decimal("2")
    assert "single cumulative" in plan["reserve_scope"]
    assert all("reserve_usd" not in attempt for attempt in plan["prior_attempts"])
    assert {a["run_id"] for a in plan["prior_attempts"]} == {36918829142, 36921167573, 36942800586}
    assert all(len(a["source_commit"]) == 40 and len(a["host_receipt_sha256"]) == 64
               for a in plan["prior_attempts"])
    assert plan["prior_attempts_sha256"] == pilot._sha(pilot._canonical(plan["prior_attempts"]))
    assert Decimal(plan["allocation_rate_usd_per_second"]) > pilot.GPU_RATE
    assert plan["cpu_request_and_limit"] == [2, 2]
    assert plan["memory_request_and_limit_mib"] == [32768, 32768]


def test_prior_debit_cannot_underfund_any_completed_attempt(monkeypatch):
    monkeypatch.setattr(pilot, "PRIOR_DEBIT_USD", Decimal("0.64"))
    with pytest.raises(ValueError, match="all three completed host sessions"):
        pilot.budget_plan("9.34")


@pytest.mark.parametrize("mutation", ["missing_attempt", "duplicate_attempt", "missing_receipt", "missing_stop_proof"])
def test_prior_ledger_rejects_missing_or_unbound_attempts(monkeypatch, mutation):
    attempts = deepcopy(list(pilot.PRIOR_ATTEMPTS))
    if mutation == "missing_attempt":
        attempts.pop()
    elif mutation == "duplicate_attempt":
        attempts[1] = dict(attempts[0])
    elif mutation == "missing_receipt":
        attempts[1]["host_receipt_sha256"] = ""
    else:
        attempts[1]["stop_log_sha256"] = ""
    monkeypatch.setattr(pilot, "PRIOR_ATTEMPTS", tuple(attempts))
    with pytest.raises(ValueError, match="prior"):
        pilot.budget_plan()


@pytest.mark.parametrize("mutation", ["ceiling", "session", "source", "receipt", "ledger"])
def test_runtime_budget_cannot_override_committed_prior_evidence(mutation, tmp_path):
    control = {"budget": pilot.budget_plan()}
    budget = control["budget"]
    if mutation == "ceiling":
        budget["ceiling_usd"] = "10"
    elif mutation == "session":
        budget["max_session_seconds"] += 1
    elif mutation == "source":
        budget["prior_attempts"][1]["source_commit"] = "0" * 40
    elif mutation == "receipt":
        budget["prior_attempts"][1]["host_receipt_sha256"] = "0" * 64
    else:
        budget["prior_attempts_sha256"] = "0" * 64
    # Rejection happens before git/source lookup, output creation, or Modal import.
    with pytest.raises(ValueError):
        pilot.launch({}, control, tmp_path / "must_not_exist", tmp_path)
    with pytest.raises(ValueError):
        pilot.remote_pilot({}, control)
    assert not (tmp_path / "must_not_exist").exists()


def test_valid_runtime_budget_retains_exact_ledger():
    pilot.validate_budget_control({"budget": pilot.budget_plan()})


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
    assert "--budget-usd 9.34" in source and "--budget-usd 10" not in source and "--detach" not in source


def test_pinned_sdk_remote_import_matches_the_only_image_module():
    function_utils = pytest.importorskip("modal._utils.function_utils")
    assert function_utils.FunctionSourceInfo(pilot.remote_pilot).module_name == "scripts.modal_jane_pilot"


def test_image_source_allowlist_excludes_evaluator_and_data():
    assert set(pilot.SOURCE_FILES) == {"scripts/__init__.py", "scripts/modal_jane_pilot.py",
                                    "scripts/jane_gpu_backend.py", "scripts/jane_qwen_backend.py",
                                    "scripts/jane_output_constraints.py"}
    assert all("data" not in path and "evaluation" not in path for path in pilot.SOURCE_FILES)


def test_fourth_attempt_names_never_reuse_prior_allocation_claim():
    assert pilot.VOLUME_NAME == "jane-mcq-pilot-20261001-v6"
    assert pilot.APP_NAME == "jane-mcq-pilot-20261001-v6"
    assert "v6 global-budget" in pilot.LAUNCH_MESSAGE


def test_small_remaining_budget_fails_before_model_or_provider_loading():
    with pytest.raises(ValueError, match="safety reserves"):
        pilot.budget_plan("3.22")


@pytest.mark.parametrize("valid_oe", [18, 19])
def test_interface_threshold_is_per_format_and_preserves_95_percent(valid_oe):
    jobs = [job("oe", qid=f"oe-{i}") for i in range(20)]
    jobs += [job("mc", qid=f"mc-{i}") for i in range(100)]
    result = trace(jobs)
    for row in result["predictions"][valid_oe:20]:
        row.update(answer=None, confidence=None, status="invalid", raw_response="not JSON")
    assert pilot.interface_gate(jobs, result)["passed"] is (valid_oe == 19)


@pytest.mark.parametrize("fail_second_model", [False, True])
def test_no_main_inference_until_both_development_gates_pass(monkeypatch, tmp_path, fail_second_model, capsys):
    from scripts import jane_gpu_backend as gpu
    root = tmp_path / "output"
    volume = SimpleNamespace(reload=lambda: None, commit=lambda: None)
    monkeypatch.setitem(sys.modules, "modal", SimpleNamespace(
        Volume=SimpleNamespace(from_name=lambda *_args, **_kwargs: volume)))
    real_path = Path
    monkeypatch.setattr(pilot, "Path", lambda value: root if value == "/jane/output" else real_path(value))
    monkeypatch.setattr(pilot, "source_identity", lambda _repo: {})
    calls = []
    def backend_run(public, config, **_kwargs):
        _kwargs["checkpoint"].write("checkpoint row\n")
        _kwargs["progress"]({"completed_jobs": len(public["jobs"]), "total_jobs": len(public["jobs"]),
                             "elapsed_seconds": 0.1, "prompt": "must never log"})
        assert _kwargs["checkpoint"].stream is None
        calls.append((config.model, public["jobs"][0]["qid"]))
        answer = "..." if fail_second_model and config.model == pilot.MODELS[1][1] else "France"
        result = trace(public["jobs"], answer)
        result["metadata"] = {"total_seconds": 0.1, "model_load_seconds": 0.05}
        return result
    monkeypatch.setattr(gpu, "run", backend_run)
    dev = [job("oe", qid="dev"), job("mc", qid="dev")]
    main = [job("oe", qid="main"), job("mc", qid="main")]
    ctrl = {k: v for k, v in job("mc", qid="control").items() if k not in {"prefix_id", "fraction"}}
    ctrl["options"] = [{"id": letter, "text": letter} for letter in "ABCD"]
    packages = {"dev_jobs.json": package(dev), "main_jobs.json": package(main),
                "main_choices_only.json": {"schema_version": "jane-choice-controls-v1",
                "evidence_scope": "engineering_smoke", "jobs": [ctrl]}}
    control = {"budget": pilot.budget_plan(), "source_commit": "a" * 40,
               "source_files_sha256": {}, "public_input_id": "b" * 64,
               "absolute_deadline_unix": pilot.time.time() + 5500}
    result = pilot.remote_pilot(packages, control)
    assert calls[:2] == [(model, "dev") for _tag, model, _rev in pilot.MODELS]
    if fail_second_model:
        assert len(calls) == 2
        assert result["status"] == "DEVELOPMENT_INTERFACE_GATE_FAILED"
        assert not (root / "throughput_gate.json").exists()
    else:
        assert calls[2:] == [(model, qid) for _tag, model, _rev in pilot.MODELS
                             for qid in ("main", "control")]
        assert result["status"] == "COMPLETED"
    assert (root / "execution_receipt.json").exists()
    summaries = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [s["passed"] for s in summaries if s.get("event") == "development_gate"] == [True, not fail_second_model]
    progress = [s for s in summaries if "completed_jobs" in s]
    assert len(progress) == len(calls)
    assert all(set(s) == {"model", "phase", "completed_jobs", "total_jobs", "elapsed_seconds"} for s in progress)
    throughput = [s for s in summaries if s.get("event") == "throughput_gate"]
    assert len(throughput) == (0 if fail_second_model else 1)


@pytest.mark.parametrize("completed,previous,expected", [
    (8, 0, False), (120, 0, False), (128, 0, True), (136, 128, False),
    (256, 128, True), (264, 0, True), (280, 256, True), (280, 280, False),
])
def test_progress_logs_only_milestones_or_completion_and_drops_extra_fields(completed, previous, expected):
    update = {"completed_jobs": completed, "total_jobs": 280, "elapsed_seconds": 14.5,
              "raw_response": "must never log", "prompt": "must never log", "secret": "must never log"}
    summary = pilot.progress_summary("qwen3b", "main", update, previous)
    if expected:
        assert summary == {"model": "qwen3b", "phase": "main", "completed_jobs": completed,
                           "total_jobs": 280, "elapsed_seconds": 14.5}
    else:
        assert summary is None
