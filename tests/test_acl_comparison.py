import pytest


def test_pairing_uses_ids_and_common_payoff():
    from scripts.compare_acl_arms import compare_episodes
    # Arbitrary arm-specific payoffs must not affect the common accuracy measure.
    def case(qid, answer, base):
        return {"qid": qid, "group_id": qid, "correct": answer, "abstained": False,
                "steps": [{"transition": {"base_reward": base}, "reward": base}]}
    reference = [case("A", False, 0), case("B", True, 1)]
    other = [case("B", False, 8), case("A", True, 4)]
    result = compare_episodes(reference, other, gamma=.99)
    assert result["accuracy_delta"] == 0
    assert result["correct_only_in_arm"] == result["correct_only_in_reference"] == 1
    assert result["matched_questions"] == 2
    with pytest.raises(ValueError, match="match"):
        compare_episodes(reference, [other[0]], gamma=.99)
    with pytest.raises(ValueError, match="duplicate"):
        compare_episodes(reference, other + [other[0]], gamma=.99)


@pytest.mark.parametrize("changed", ["wait_cost", "class_review_sha256", "selection_sha256"])
def test_matched_runs_require_common_costs_and_oracle_reference(monkeypatch, changed):
    from copy import deepcopy
    from types import SimpleNamespace
    from scripts import compare_acl_arms as analysis

    spec = {"version": 1, "menu": "baseline", "seed": 1, "k": 4, "wait_cost": -.01,
            "wrong_reward": -.5, "correct_reward": 1., "abstain_reward": 0.,
            "baseline_condition": "independent_pool", "reward": {"gamma": .99}}
    inputs = {"spec": spec, "dataset_sha256": "d", "oracle_sha256": "o", "reservoir_sha256": "r",
              "split_roles": {}, "core": {}, "source_review": {changed: "same"} if changed != "wait_cost" else {}}
    reference = {"runtime": {}, "policy": "ppo", "training": {}, "device": "cpu", "assets": {}, "input": inputs}
    arm = deepcopy(reference)
    arm["input"]["spec"]["menu"] = "uniform_global"
    if changed == "wait_cost":
        arm["input"]["spec"][changed] = -.5
    else:
        arm["input"]["source_review"][changed] = "different"
    episode = {"qid": "q", "group_id": "g", "correct": True, "abstained": False,
               "steps": [{"transition": {"base_reward": 1.}, "reward": 1.}]}
    monkeypatch.setattr(analysis, "_read_run", lambda path, split: (
        reference if path.name == "ref" else arm, [episode]))
    with pytest.raises(ValueError, match="matched arms"):
        analysis.compare(SimpleNamespace(name="ref"), SimpleNamespace(name="arm"), "test")

@pytest.mark.parametrize("field", ["start","end","shape","threshold","knots","gamma",
                                    "menus_sha256","review_sha256","review_presence","encoder",
                                    "retriever","interval_arithmetic","class_review_code_sha256"])
def test_reward_only_match_checks_nuisance_dimensions(monkeypatch, field):
    from copy import deepcopy
    from pathlib import Path
    from scripts import compare_acl_arms as analysis
    reward = {"arm":"linear","potential_scale":0.,"start":2.,"end":.5,"shape":2.,
              "threshold":.4,"knots":[[0,1],[1,0]],"gamma":.99}
    inputs = {"spec":{"menu":"sbert_hard","reward":reward},"menus_sha256":"menus",
              "source_review":{"review_sha256":"fixed-menu"}}
    left = {"input":inputs,"runtime":{},"training":{},"policy":"t5","device":"cpu",
            "assets":dict.fromkeys(("encoder","retriever","interval_arithmetic","class_review_code_sha256"),"a")}
    right=deepcopy(left)
    right["input"]["spec"]["reward"].update(arm="cosine",potential_scale=.5)
    if field in reward:
        right["input"]["spec"]["reward"][field] = "changed"
    elif field == "menus_sha256":
        right["input"][field]="changed"
    elif field == "review_sha256":
        right["input"]["source_review"][field]="changed"
    elif field == "review_presence":
        del right["input"]["source_review"]
    else:
        right["assets"][field]="changed"
    row={"qid":"q","group_id":"g","correct":True,"abstained":False,
         "steps":[{"reward":1,"transition":{"base_reward":1}}]}
    monkeypatch.setattr(analysis,"_read_run",lambda folder, split:
                        (left if folder.name=="left" else right,[row]))
    with pytest.raises(ValueError, match="matched arms"):
        analysis.compare(Path("left"),Path("right"),"test")


def test_checkpoint_parent_and_payload_required(tmp_path):
    from scripts.acl_checkpoints import verify_completed_lineage, hash_tree
    from qb_data.acl_protocol import publish_json, digest
    settings={"updates":2,"checkpoint_updates":1}
    parent=None
    for i in range(1,3):
        block=tmp_path/f"train-{i:06d}-abc"
        block.mkdir()
        (block/"policy.zip").write_bytes(b"fixture")
        manifest={"run_id":tmp_path.name,"parent":parent,"first_update":i-1,"end":i,"files":hash_tree(block)}
        publish_json(block/"files.json",manifest)
        parent=digest(manifest)
    receipt={"checkpoint_sha256":parent}
    verify_completed_lineage(tmp_path,settings,receipt)
    first=tmp_path/"train-000001-abc"
    (first/"policy.zip").write_bytes(b"altered")
    with pytest.raises(ValueError, match="checkpoint"):
        verify_completed_lineage(tmp_path,settings,receipt)
    (first/"files.json").unlink()
    with pytest.raises(ValueError, match="checkpoint"):
        verify_completed_lineage(tmp_path,settings,receipt)
