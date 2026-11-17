import pytest


def test_pairing_uses_ids_and_common_payoff():
    from scripts.compare_acl_arms import compare_episodes
    # Arbitrary arm-specific payoffs must not affect the common accuracy measure.
    def case(qid, answer, base):
        return {"qid": qid, "group_id": qid, "correct": answer, "abstained": False,
                "steps": [{"transition": {"base_reward": base}, "reward": base}]}
    reference = [case("A", False, 0), case("B", True, 1)]
    other = [case("B", True, 8), case("A", True, 4)]
    result = compare_episodes(reference, other, gamma=.99)
    assert result["accuracy_delta"] == .5
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
