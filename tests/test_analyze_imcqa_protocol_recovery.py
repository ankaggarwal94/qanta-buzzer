"""Recovery must retain scientific identities and all frozen numerical gates."""
import copy
import json

import pytest

from scripts import analyze_imcqa_protocol_recovery as a


NUMERICAL = {"raw_logit_atol": .001, "raw_logit_rtol": .00001, "probability_atol": .001}


def modes_fixture():
    jobs = [{"allowed_actions": "ABCDE", "option_source_ids": dict(zip("ABCD", "ABCD"))}]
    reference = [{"logits": [1., .5, 0., -1., 3.]}]
    candidates = []
    for mode in ("uncached", "cached"):
        for size in (2, 4, 8):
            elapsed = .1 if mode == "cached" and size == 2 else .2
            candidates.append({"name": f"{mode}_{size}", "batch_size": size, "cached": mode == "cached",
                "outputs": copy.deepcopy(reference), "batch_seconds": [elapsed], "elapsed_forward_seconds": elapsed,
                "diagnostic_compatible": True, "production_approved": False})
    return candidates, jobs, reference


def test_trial_selection_recomputes_raw_scores_not_passed_flags():
    candidates, jobs, reference = modes_fixture()
    name, checks = a.selected_trial_mode(candidates, jobs, reference, NUMERICAL)
    assert name == "cached_2" and len(checks) == 6
    candidates[0]["outputs"][0]["logits"][0] += .1
    with pytest.raises(ValueError, match="numerical logit"):
        a.selected_trial_mode(candidates, jobs, reference, NUMERICAL)


def test_hidden_candidate_flip_under_wait_is_rejected():
    candidates, jobs, reference = modes_fixture()
    reference[0]["logits"] = [1.00001, 1., 0., -1., 3.]
    for candidate in candidates:
        candidate["outputs"] = copy.deepcopy(reference)
    candidates[3]["outputs"][0]["logits"] = [1., 1.00001, 0., -1., 3.]
    with pytest.raises(ValueError, match="candidate numerical"):
        a.selected_trial_mode(candidates, jobs, reference, NUMERICAL)


def test_trial_fastest_rule_and_timing_identity_are_enforced():
    candidates, jobs, reference = modes_fixture()
    candidates[0].update(batch_seconds=[.05], elapsed_forward_seconds=.05)
    with pytest.raises(ValueError, match="fastest"):
        a.selected_trial_mode(candidates, jobs, reference, NUMERICAL)
    candidates[0]["elapsed_forward_seconds"] = .2
    with pytest.raises(ValueError, match="timing"):
        a.selected_trial_mode(candidates, jobs, reference, NUMERICAL)


def test_source_allowlist_cannot_omit_helpers_or_accept_changed_bytes(tmp_path):
    (tmp_path / "helper.py").write_text("original")
    expected = {"helper.py": a.old.sha256(tmp_path / "helper.py")}
    assert a.validate_source_hashes({"source_files_sha256": expected}, ["helper.py"], tmp_path) == expected
    with pytest.raises(ValueError, match="source allowlist"):
        a.validate_source_hashes({"source_files_sha256": {}}, ["helper.py"], tmp_path)
    (tmp_path / "helper.py").write_text("changed")
    with pytest.raises(ValueError, match="source allowlist"):
        a.validate_source_hashes({"source_files_sha256": expected}, ["helper.py"], tmp_path)


def test_diagnostics_use_actual_production_pair_companions():
    assert a.production_pair_ids(["a", "b", "c", "d", "e", "f"], ["b", "c"]) == ["a", "b", "c", "d"]
    with pytest.raises(ValueError, match="missing"):
        a.production_pair_ids(["a", "b"], ["c"])
    with pytest.raises(ValueError, match="duplicated"):
        a.production_pair_ids(["a", "a"], ["a"])


def contract_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(a, "RECOVERY_LAUNCH_COMMIT", "a" * 40)
    run = tmp_path / "run"
    directory = run / "output/qwen3b"
    predecessor, diagnostic = tmp_path / "predecessor", tmp_path / "diagnostic"
    for path in (directory, predecessor, diagnostic):
        path.mkdir(parents=True)
    (predecessor / "receipt.json").write_text('{"status":"failed"}')
    original = {"input_sha256": "public", "ordered_score_ids": ["a", "b"], "context_sha256": ["x", "y"], "token_counts": [3, 4]}
    (predecessor / "plan.json").write_text(json.dumps(original))
    (diagnostic / "receipt.json").write_text('{"status":"complete"}')
    for name, path in (("PREDECESSOR_RECEIPT_SHA", predecessor / "receipt.json"),
                       ("PREDECESSOR_PLAN_SHA", predecessor / "plan.json"),
                       ("DIAGNOSTIC_RECEIPT_SHA", diagnostic / "receipt.json")):
        monkeypatch.setattr(a, name, a.old.sha256(path))
    shared = {"execution_protocol": a.EXECUTION_PROTOCOL, "selected_mode": "cached_2", "cached": True,
              "batch_size": 2, "diagnostic_receipt_sha256": a.DIAGNOSTIC_RECEIPT_SHA,
              "public_input_sha256": "public", "source_commit": "a" * 40}
    (directory / "receipt.json").write_text(json.dumps({**shared, "predecessor_run_id": "imcqa-protocol-dev-20261004"}))
    (directory / "metadata.json").write_text(json.dumps(shared))
    (run / "control.json").write_text(json.dumps({"source_commit": "a" * 40}))
    (directory / "plan.json").write_text(json.dumps({**original, "execution_protocol": a.EXECUTION_PROTOCOL,
        "batch_size": 2, "cached": True, "original_plan_sha256": a.PREDECESSOR_PLAN_SHA}))
    return directory, predecessor, diagnostic


@pytest.mark.parametrize("mutation", ["batch", "runtime_mode", "context", "launch"])
def test_recovery_contract_rejects_batch_context_or_attribution_change(tmp_path, monkeypatch, mutation):
    directory, predecessor, diagnostic = contract_fixture(tmp_path, monkeypatch)
    assert a.validate_recovery_contract(directory, "public", {}, predecessor, diagnostic)["scientific_inputs_changed"] is False
    filename = "plan.json" if mutation == "context" else "metadata.json"
    path = directory / filename
    data = json.loads(path.read_text())
    if mutation == "batch": data["batch_size"] = 32
    if mutation == "runtime_mode": data["cached"] = False
    if mutation == "context": data["context_sha256"] = ["changed", "y"]
    if mutation == "launch": data["source_commit"] = "b" * 40
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="recovery"):
        a.validate_recovery_contract(directory, "public", {}, predecessor, diagnostic)


def test_production_pre_gate_raw_must_bind_all_three_forward_paths(tmp_path):
    (tmp_path / "attempts").mkdir()
    expected = {"score_ids": ["a"], "production": [{"logits": [1., 2., 3., 4., 5.]}],
        "diagnostic": [{"logits": [1., 2., 3., 4., 5.]}], "single_reference": [{"logits": [1., 2., 3., 4., 5.]}]}
    path = tmp_path / "attempts/000_production_diagnostics_raw.json"
    path.write_text(json.dumps(expected))
    arguments = (tmp_path, {**expected, "gate": {"passed": True}}, expected["score_ids"],
        expected["production"], expected["diagnostic"], expected["single_reference"])
    a.validate_production_raw(*arguments)
    corrupt = copy.deepcopy(expected)
    corrupt["single_reference"][0]["logits"][0] += .1
    path.write_text(json.dumps(corrupt))
    with pytest.raises(ValueError, match="pre-gate production"):
        a.validate_production_raw(*arguments)


@pytest.mark.parametrize("mutation", ["hash", "sequence", "job", "live", "production", "missing", "extra", "filename"])
def test_active_pre_gate_raw_is_exact_and_one_to_one(tmp_path, mutation):
    (tmp_path / "attempts").mkdir()
    job = {"qid": "q", "condition": "independent", "block": "factorial", "rotation": 2, "round": 1}
    production = {"raw_action_logits": dict(zip("ABCDE", [1., 2., 3., 4., 5.]))}
    row = {"score_id": "a", "live": {"logits": [1., 2., 3., 4., 5.]}, "raw_file": "000_active_000_raw.json"}
    expected = {"score_id": "a", "sequence_index": 0, **job, "live": row["live"], "production": {"logits": [1., 2., 3., 4., 5.]}}
    path = tmp_path / "attempts" / row["raw_file"]
    path.write_text(json.dumps(expected))
    row["raw_file_sha256"] = a.old.sha256(path)
    a.validate_active_raw(tmp_path, [row], {"a": production}, {"a": job})
    if mutation == "hash": row["raw_file_sha256"] = "changed"
    elif mutation == "filename": row["raw_file"] = "../000_active_000_raw.json"
    elif mutation == "missing": path.unlink()
    elif mutation == "extra": (tmp_path / "attempts/000_active_001_raw.json").write_text("{}")
    else:
        changed = copy.deepcopy(expected)
        if mutation == "sequence": changed["sequence_index"] = 1
        if mutation == "job": changed["qid"] = "wrong"
        if mutation == "live": changed["live"]["logits"][0] += .1
        if mutation == "production": changed["production"]["logits"][0] += .1
        path.write_text(json.dumps(changed))
        row["raw_file_sha256"] = a.old.sha256(path)
    with pytest.raises(ValueError, match="active pre-gate"):
        a.validate_active_raw(tmp_path, [row], {"a": production}, {"a": job})


def test_recovery_cannot_relax_original_numerical_contract():
    root = a.Path(a.__file__).resolve().parents[1]
    recovery = a.old.load_json(root / "configs/imcqa_3b_protocol_recovery.json")
    original = a.old.load_json(root / "configs/imcqa_protocol_pilot.json")
    a.validate_recovery_config(recovery, original)
    recovery["numerical_checks"]["raw_logit_atol"] *= 2
    with pytest.raises(ValueError, match="numerical tolerance"):
        a.validate_recovery_config(recovery, original)
