"""CPU evidence gates: real physical batches, exact replay, and bounded accounting."""
import json
import math
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
from scripts import imcqa_tuned_scoring as scoring


def job(index=0, **changes):
    return {"score_id": str(index), "allowed_actions": "ABCDE", "option_source_ids": dict(zip("ABCD", "ABCD")), **changes}


def test_diagnostic_feature_batch_is_not_the_physical_forward_batch(monkeypatch):
    observed = []
    def fake_forward(torch, model, tokenizer, contexts, *, cached, batch_size):
        observed.append(([c["i"] for c in contexts], cached, batch_size))
        return [{"i": c["i"]} for c in contexts]
    monkeypatch.setattr(scoring, "forward", fake_forward)
    contexts = [{"i": i} for i in range(30)]
    result = scoring.forward_batches(None, None, None, contexts, cached=True, permute=True)
    assert result == contexts
    assert [len(batch[0]) for batch in observed] == [8, 8, 8, 6]
    assert all(batch[1:] == (True, 8) for batch in observed)
    assert scoring.original.BATCH_SIZE == 32


@pytest.mark.parametrize("contexts", [[], [{}], [{}]*3])
def test_incomplete_cache_pairs_rejected(contexts):
    with pytest.raises(ValueError, match="complete context pairs"):
        scoring.forward_batches(None, None, None, contexts, cached=True)


def test_diagnostics_include_true_individual_minimum_not_only_shortest_pair(monkeypatch):
    contexts = [{"scored_input_token_ids": [0]*length} for length in (1, 100, 10, 10, 200, 200)]
    jobs = [job(i) for i in range(6)]
    monkeypatch.setattr(scoring.original, "diagnostic_indices", lambda jobs, contexts: [2, 3, 4, 5])
    assert scoring.diagnostic_indices(jobs, contexts) == list(range(6))


def test_production_checks_cover_actual_first_middle_last_batches():
    assert scoring.production_offsets(8000) == [0, 4000, 7992]
    with pytest.raises(ValueError):
        scoring.production_offsets(7999)


def test_failure_preserves_raw_numeric_sides_before_rejection(tmp_path):
    path = tmp_path/"failure_raw.json"
    left, right = [{"logits": [0, 1, 2, 3, 4]}], [{"logits": [0, 1, 2, 3, 4.01]}]
    with pytest.raises(ValueError, match="logits exceed"):
        scoring.record_numeric_gate(path, left, right, [job()])
    saved = json.loads(path.read_text())
    assert saved["production"] == left and saved["reference"] == right
    assert saved["score_ids"] == ["0"]


def test_unchanged_numeric_gate_catches_argmax_even_within_raw_tolerance():
    with pytest.raises(ValueError, match="argmax"):
        scoring.numeric_agreement([{"logits": [1.0001, 1, 0, 0, 0]}],
            [{"logits": [1, 1.0001, 0, 0, 0]}], [job()])
    gate = scoring.numeric_agreement([{"logits": [1, 0, 0, 0, 0]}],
            [{"logits": [1, 0, 0, 0, 0]}], [job()])
    assert (gate["atol"], gate["rtol"], gate["probability_atol"]) == (.001, .00001, .001)


def replay_evidence():
    rows = []
    for qid in ("q0", "q1", "q2", "q3"):
        for condition in scoring.CONDITIONS:
            for rotation in (0, 2):
                for r in range(1, 6):
                    rows.append({"score_id": f"{qid}|{condition}|{rotation}|{r}", "qid": qid,
                        "condition": condition, "rotation": rotation, "round": r,
                        "agreement": {"passed": True}})
    return {"qids": [f"q{i}" for i in range(4)], "rows": rows}


def test_complete_full_trajectory_replay_accepted():
    scoring.validate_replay_evidence(replay_evidence())


@pytest.mark.parametrize("mutation", ["empty", "duplicate", "failed_gate", "round_gap", "missing_state"])
def test_false_replay_completion_rejected(mutation):
    evidence = replay_evidence()
    if mutation == "empty":
        evidence["rows"] = []
    elif mutation == "duplicate":
        evidence["rows"][1] = evidence["rows"][0]
    elif mutation == "failed_gate":
        evidence["rows"][0]["agreement"]["passed"] = False
    elif mutation == "round_gap":
        evidence["rows"][0]["round"] = 0
    else:
        evidence["rows"] = evidence["rows"][1:]
    with pytest.raises(ValueError):
        scoring.validate_replay_evidence(evidence)


def test_projection_reserves_active_and_actual_production_validation():
    projected = scoring.projection_with_validation(128, 8000, 12.8, 2000, [0.8]*16, [0.1, 0.5])
    assert projected["forecast_method"] == "max(mean_times_1_2,nearest_rank_p95_batch_projection)"
    assert projected["total_validation_reserve_seconds"] == pytest.approx((80+24+48)*.5*1.2+45)
    assert projected["proceed"] is True
    stopped = scoring.projection_with_validation(128, 8000, 12.8, 900, [0.8]*16, [.5])
    assert stopped["proceed"] is False
    with pytest.raises(ValueError):
        scoring.projection_with_validation(128, 8000, 12.8, 2000, [.8]*16, [math.nan])


def test_one_off_diagnostics_do_not_distort_recurring_throughput():
    common = dict(processed=128, total=8000, batch_times=[.8]*16, single_seconds=[.1],
                  checkpoint_seconds=[.05, .1])
    ordinary = scoring.projection_with_validation(seconds=14., remaining=2000., **common)
    expensive = scoring.projection_with_validation(seconds=114., remaining=1900.,
        production_diagnostic_seconds=100., **common)
    assert expensive["recurring_production_seconds"] == ordinary["recurring_production_seconds"] == 14.
    assert expensive["projected_remaining_seconds"] == ordinary["projected_remaining_seconds"]
    # The elapsed one-off work still consumes the actual remaining deadline.
    assert ordinary["remaining_seconds"] - expensive["remaining_seconds"] == pytest.approx(100.)
    assert expensive["benchmark_wall_seconds"] == expensive["recurring_production_seconds"] + expensive["nonrecurring_production_diagnostic_seconds"]
    assert expensive["proceed"] is True


def test_recurring_serialization_cost_and_low_time_still_stop():
    common = dict(processed=128, total=8000, batch_times=[.8]*16, single_seconds=[.1],
                  production_diagnostic_seconds=100., checkpoint_seconds=[.1])
    normal = scoring.projection_with_validation(seconds=114., remaining=2000., **common)
    slow_io = scoring.projection_with_validation(seconds=140., remaining=2000., **common)
    low_time = scoring.projection_with_validation(seconds=114., remaining=800., **common)
    assert normal["proceed"] is True
    assert slow_io["recurring_production_seconds"] == 40.
    assert slow_io["proceed"] is False
    assert low_time["proceed"] is False


def test_progress_commits_are_counted_and_expensive_callbacks_stop():
    common = dict(processed=128, total=34000, seconds=24., remaining=8000.,
                  batch_times=[1.4]*16, single_seconds=[.2])
    fast = scoring.projection_with_validation(checkpoint_seconds=[.1, .2], **common)
    slow = scoring.projection_with_validation(checkpoint_seconds=[.1, 8.], **common)
    assert fast["remaining_checkpoint_counts"] == {"periodic_progress": 132,
        "future_production_diagnostics": 2, "benchmark": 1, "final_status": 1,
        "allocation_final_commit": 1}
    assert fast["checkpoint_reserve_seconds"] == pytest.approx(137 * .2 * 1.2)
    assert fast["proceed"] is True
    assert slow["proceed"] is False


def test_p95_batch_guard_still_rejects_a_slow_tail():
    projected = scoring.projection_with_validation(128, 8000, 13., 2000.,
        [.4] * 15 + [6.], [.1], checkpoint_seconds=[.1])
    assert projected["mean_projected_remaining_seconds"] * 1.2 < projected["remaining_seconds"]
    assert projected["p95_projected_remaining_seconds"] > projected["remaining_seconds"]
    assert projected["proceed"] is False


@pytest.mark.parametrize("diagnostic,callbacks", [(-1., []), (13., []), (math.nan, []),
                                                   (0., [-.1]), (0., [math.inf])])
def test_invalid_timing_accounting_fails_closed(diagnostic, callbacks):
    with pytest.raises(ValueError):
        scoring.projection_with_validation(128, 8000, 13., 2000., [.8]*16, [.1],
            production_diagnostic_seconds=diagnostic, checkpoint_seconds=callbacks)


def test_diagnostic_subtraction_cannot_remove_measured_production_work():
    with pytest.raises(ValueError, match="smaller than its forward calls"):
        scoring.projection_with_validation(128, 8000, 13., 2000., [.8]*16, [.1],
            production_diagnostic_seconds=1.)


def test_only_new_7b_allocation_accepted_before_file_access(tmp_path):
    with pytest.raises(ValueError, match="invalid pinned model"):
        scoring.run_scoring("qwen3b", tmp_path/"missing", "0"*64, tmp_path, tmp_path/"out",
                            source_commit="1"*40, max_seconds=2880)
    with pytest.raises(ValueError, match="invalid pinned model"):
        scoring.run_scoring("qwen7b", tmp_path/"missing", "0"*64, tmp_path, tmp_path/"out",
                            source_commit="1"*40, max_seconds=float("nan"))


@pytest.mark.parametrize("extra_diagnostic_seconds", [0., 20.])
def test_mocked_complete_worker_exercises_runtime_and_live_replay(tmp_path, monkeypatch, extra_diagnostic_seconds):
    """Fake GPU orchestration through real public validation, numerical joins and analysis.

    Model/tokenizer and trusted cache receipt are synthetic. This tests software
    compatibility and rejection gates; it is not GPU numerical validation.
    """
    tag = "qwen7b"
    revision = scoring.base.PINNED_MODELS[scoring.base.MODELS[tag]]
    snapshot = tmp_path/revision
    snapshot.mkdir()
    weight = snapshot/"model.safetensors"
    weight.write_bytes(b"fake pinned weight")
    (tmp_path/f"{tag}_expected_model_hashes.json").write_text(json.dumps({
        "model": scoring.base.MODELS[tag], "revision": revision,
        "model_files_sha256": {weight.name: scoring.base.file_hash(weight)}}))
    tokenizer = SimpleNamespace(padding_side="left", pad_token_id=0, chat_template="test")
    model = SimpleNamespace()
    model.to = lambda *args: model
    model.eval = lambda: model
    model.float = lambda: model
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=lambda **kwargs: str(snapshot)))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *args, **kwargs: tokenizer),
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=lambda *args, **kwargs: model)))
    nothing = lambda *args, **kwargs: None
    cuda = SimpleNamespace(is_available=lambda: True, device_count=lambda: 1, is_bf16_supported=lambda: True,
        manual_seed_all=nothing, synchronize=nothing, empty_cache=nothing, max_memory_allocated=lambda: 123)
    torch = SimpleNamespace(cuda=cuda, set_num_threads=nothing, manual_seed=nothing, use_deterministic_algorithms=nothing,
        set_float32_matmul_precision=nothing, bfloat16="bfloat16",
        backends=SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace()), cudnn=SimpleNamespace()))
    monkeypatch.setitem(sys.modules, "torch", torch)
    versions = {"torch": "2.6.0", "transformers": "4.51.3", "tokenizers": "0.21.1", "safetensors": "0.5.3", "huggingface-hub": "0.30.2"}
    monkeypatch.setattr(scoring.importlib.metadata, "version", versions.__getitem__)
    monkeypatch.setattr(scoring.base, "snapshot_tensor_state", lambda model: {})
    monkeypatch.setattr(scoring.paired, "validate_promotion", lambda model, state: {"all_checks_passed": True})
    from tests.test_imcqa_tuned_design import build_fixture
    from scripts import analyze_imcqa_tuned as analysis
    from scripts import modal_acl_paired_prompt_scores as cached_runner
    package, dataset = build_fixture()
    lock = {"schema_version": "imcqa-independently-tuned-policies-v1", "status": "frozen_for_fresh_evaluation",
        "selection_qids": [], "calibration_qids": [], "selection_group_ids": [], "calibration_group_ids": [],
        "calibrators": {m: {"intercept": 0., "slope": 1., "feature_clip": [1e-6, .999999]}
                        for m in scoring.CONDITIONS},
        "policies": [{"condition":m, "family":f, "always_pass":False, "threshold":.6,
                      "fixed_round":1 if f == "fixed_selective" else None}
                     for m in scoring.CONDITIONS for f in ("adaptive_selective", "fixed_selective")]}
    lock_path = tmp_path/"policies.json"
    lock_path.write_bytes(scoring.base.canonical(lock))
    evaluator = tmp_path/"main_dataset.json"
    evaluator.write_bytes(scoring.base.canonical(dataset))
    package["policy_lock_sha256"] = scoring.base.file_hash(lock_path)
    package["main_dataset_sha256"] = scoring.base.file_hash(evaluator)
    jobs = package["jobs"]
    (tmp_path/"cache_prepare_receipt.json").write_text("{}")
    monkeypatch.setattr(cached_runner, "verify_prepare", lambda raw: {"model_receipts": {
        "qwen7b": {"model_files_sha256": {weight.name: scoring.base.file_hash(weight)}}}})
    monkeypatch.setattr(scoring, "prepare_context", lambda tokenizer, job: {
        "option_token_ids": dict(zip("ABCDE", range(5))), "scored_context_sha256": "a"*64,
        "scored_input_token_ids": [1, 2, job["score_index"]+3]})
    monkeypatch.setattr(scoring, "paired_order", lambda contexts: list(range(len(contexts))))
    clock = {"now": 1000., "production": False, "extra_added": False}
    monkeypatch.setattr(scoring.time, "monotonic", lambda: clock["now"])
    real_fsync = scoring.os.fsync
    def timed_fsync(fd):
        clock["now"] += .003
        return real_fsync(fd)
    monkeypatch.setattr(scoring.os, "fsync", timed_fsync)
    def progress(update):
        clock["now"] += .05
        if update["phase"] == "diagnostics_passed":
            clock["production"] = True
    def fake_forward(torch, model, tokenizer, contexts, **kwargs):
        clock["now"] += .005 if len(contexts) == 1 else .01
        if clock["production"] and len(contexts) == 1 and not clock["extra_added"]:
            clock["now"] += extra_diagnostic_seconds
            clock["extra_added"] = True
        return [{"logits": [2., 0., 0., 0., -2.], "vocabulary_logsumexp": 3.,
                 "unconstrained_top_token_id": 0, "unconstrained_top_logit": 2.} for _ in contexts]
    monkeypatch.setattr(scoring, "forward", fake_forward)
    public = tmp_path/"public.json"
    public.write_bytes(scoring.base.canonical(package))
    receipt = scoring.run_scoring(tag, public, scoring.base.file_hash(public), tmp_path, tmp_path/"out",
                                  source_commit="1"*40, max_seconds=scoring.worker_seconds(4)-120, progress=progress)
    assert receipt["status"] == "complete", receipt
    benchmark = receipt["benchmark"]
    assert benchmark["recurring_production_seconds"] == pytest.approx(16 * (.01 + .003))
    # The four-question mock checks first and middle production batches before
    # the benchmark. Only their singles/replay/evidence/checkpoint blocks leave
    # the recurring timer; the ordinary score fsync above remains included.
    assert benchmark["nonrecurring_production_diagnostic_seconds"] == pytest.approx(extra_diagnostic_seconds + 2 * (.005 * 8 + .01 + .003 + .05))
    assert benchmark["maximum_checkpoint_callback_seconds"] == pytest.approx(.05)
    assert benchmark["remaining_checkpoint_counts"]["future_production_diagnostics"] == 1
    assert benchmark["checkpoint_reserve_seconds"] == pytest.approx(4 * .05 * 1.2)
    assert receipt["completed_rows"] == 160
    assert receipt["production_single_gate"]["rows"] == 24
    live = json.loads((tmp_path/"out/attempts/000_live_trajectories.json").read_text())
    assert live["states"] == 80 and len(live["rows"]) == 80
    assert (tmp_path/"out/attempts/000_live_000_raw.json").is_file()
    assert receipt["scores_sha256"] == scoring.base.file_hash(tmp_path/"out/scores.jsonl")
    views = analysis.load_views(public, evaluator, tmp_path/"out", lock_path)
    assert len(views) == 160
    assert len(analysis.build_episodes(views, lock)) == 128
    diagnostic_path = tmp_path/"out/attempts/000_production_diagnostics.json"
    corrupted = json.loads(diagnostic_path.read_text())
    corrupted["single_reference"][0]["logits"][0] += .1
    diagnostic_path.write_text(json.dumps(corrupted))
    with pytest.raises(ValueError, match="numerical tolerance"):
        analysis.load_views(public, evaluator, tmp_path/"out", lock_path)


@pytest.mark.parametrize("n", [False, 0, 3, 5001, 4.0])
def test_question_count_outside_frozen_range_rejected(n):
    with pytest.raises(ValueError):
        scoring.worker_seconds(n)


def test_context_runtime_bound_and_terminal_plain_coverage():
    assert scoring.worker_seconds(600) == 6000
    assert scoring.production_offsets(24000) == [0, 12000, 23992]


@pytest.mark.parametrize("case", ["wait_arm", "wrong_count", "wrong_deadline"])
def test_invalid_plain_scoring_design_rejected_before_model_load(tmp_path, monkeypatch, case):
    jobs=[{"qid": f"q{i//40}", "arm":"plain", "execution":"new"} for i in range(160)]
    seconds=scoring.worker_seconds(4)-120
    if case == "wait_arm":
        jobs[0]["arm"]="wait"
    elif case == "wrong_count":
        jobs.pop()
    else:
        seconds+=1
    monkeypatch.setattr(scoring.design,"validate_public_package",lambda package:jobs)
    public=tmp_path/"public.json"
    public.write_text("{}")
    with pytest.raises(ValueError,match="plain-only"):
        scoring.run_scoring("qwen7b",public,scoring.base.file_hash(public),tmp_path,tmp_path/"out",
            source_commit="1"*40,max_seconds=seconds)
    assert not (tmp_path/"out").exists()
