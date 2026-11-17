"""Context-dependent answerlines must gate the production scientific configuration."""
import json
from pathlib import Path
from copy import deepcopy

import pytest
import yaml

from scripts.acl_expansion import load_inputs
from scripts.acl_smoke_data import inputs
from qb_data.acl_protocol import digest
from qb_data.acl_menus import MenuOracle
from qb_data.acl_scientific_review import compile_review, extend_template


def config_for_counterexample(tmp_path):
    cfg = yaml.safe_load(Path("configs/acl_expansion/full.yaml").read_text(encoding="utf-8"))
    data = inputs()
    data["evidence_scope"] = "scientific"
    target = data["questions"][-1]
    alternative = target["menus"][0]["options"][0]["text"]
    assert alternative != target["source"]["canonical_answer"]
    target["answer"]["raw"] = target["source"]["canonical_answer"] + " [accept " + alternative + "]"
    # Like the actual freeze, the extract omits the explicitly acceptable alternative.
    assert alternative not in target["answer"]["accepted"]
    aliases = {"version": "fixture", "reviewer": "fixture", "classes": {
        q["source"]["canonical_answer"]: [] for q in data["questions"]}}
    for name, value in (("dataset", data), ("oracle", aliases)):
        path = tmp_path / (name + ".json")
        path.write_text(json.dumps(value), encoding="utf-8")
        cfg["inputs"][name] = str(path)
    cfg["inputs"].pop("dataset_file_sha256")
    cfg["inputs"].pop("reservoir")
    cfg["inputs"]["class_review"] = str(tmp_path / "missing-review.json")
    cfg["inputs"]["split_roles"] = {"train": "train", "val": "val", "test": "test"}
    return cfg, data, aliases, alternative


def test_full_profile_never_prepares_without_source_adjudication(tmp_path):
    cfg, _, _, _ = config_for_counterexample(tmp_path)
    with pytest.raises((ValueError, FileNotFoundError), match="review|No such file"):
        load_inputs(cfg, None, None, None, None)
    cfg["inputs"].pop("class_review")
    with pytest.raises(ValueError, match="scientific inputs require"):
        load_inputs(cfg, None, None, None, None)


def test_reviewed_second_correct_rejected_and_decisions_transfer_across_menus(tmp_path):
    cfg, data, aliases, alternative = config_for_counterexample(tmp_path)
    baseline = load_inputs(cfg, None, None, None, None, review_draft=True)
    other = load_inputs(cfg, "uniform_global", None, 4, None, review_draft=True)
    decisions = extend_template(baseline, data, aliases)
    original = deepcopy(decisions)
    decisions = extend_template(other, data, aliases, existing=decisions)
    targets = {q["qid"]: q["source"]["canonical_answer"] for q in data["questions"]}
    for qid, source in decisions["questions"].items():
        for key, decision in source["classes"].items():
            assert decision["judgment"] == "unresolved"
            decision.update(judgment="accepted" if key == targets[qid] else "rejected",
                            reviewer="fixture adjudicator", rationale="Fixture source-specific judgment.")
    qid = data["questions"][-1]["qid"]
    decision = decisions["questions"][qid]["classes"][alternative]
    decision["judgment"] = "accepted"
    with pytest.raises(ValueError, match="contradictory"):
        compile_review(baseline, data, aliases, decisions)
    for blocked in ("prompt", "conditional", "unresolved"):
        decision["judgment"] = blocked
        with pytest.raises(ValueError, match="conditional"):
            compile_review(baseline, data, aliases, decisions)
    del decisions["questions"][qid]["classes"][alternative]
    with pytest.raises(ValueError, match="no reviewed"):
        compile_review(baseline, data, aliases, decisions)
    assert original == extend_template(baseline, data, aliases)


def test_compiler_reuses_review_on_changed_order_and_standard_runner(tmp_path):
    cfg, data, aliases, _ = config_for_counterexample(tmp_path)
    # Use a separate fully reviewed case for the valid path.
    data["questions"][-1]["answer"]["raw"] = data["questions"][-1]["source"]["canonical_answer"]
    Path(cfg["inputs"]["dataset"]).write_text(json.dumps(data), encoding="utf-8")
    runs = [load_inputs(cfg, "uniform_global", None, seed, None, review_draft=True) for seed in (1, 2)]
    assert runs[0].menus != runs[1].menus
    decisions = None
    for run in runs:
        decisions = extend_template(run, data, aliases, existing=decisions)
    for q in data["questions"]:
        for key, decision in decisions["questions"][q["qid"]]["classes"].items():
            decision.update(judgment="accepted" if key == q["source"]["canonical_answer"] else "rejected",
                            reviewer="fixture adjudicator", rationale="Reviewed for this source row.")
    sidecars = [compile_review(run, data, aliases, decisions)[0] for run in runs]
    assert sidecars[0]["menus_sha256"] != sidecars[1]["menus_sha256"]
    Path(cfg["inputs"]["class_review"]).write_text(json.dumps(decisions), encoding="utf-8")
    assert all(load_inputs(cfg, "uniform_global", None, seed, None).identity["source_review"][
               "class_review_sha256"] for seed in (1, 2))
    path = tmp_path / "review.json"
    cfg["inputs"].pop("class_review")
    cfg["inputs"]["review"] = str(path)
    path.write_text(json.dumps(sidecars[0]), encoding="utf-8")
    assert load_inputs(cfg, "uniform_global", None, 1, None).identity["source_review"]
    with pytest.raises(ValueError, match="digest"):
        load_inputs(cfg, "uniform_global", None, 2, None)


def test_source_accepted_classes_remain_distinct_and_cannot_be_certified_as_rejected(tmp_path):
    cfg, data, aliases, alternative = config_for_counterexample(tmp_path)
    target = data["questions"][-1]
    target["answer"]["accepted"].append(alternative)
    Path(cfg["inputs"]["dataset"]).write_text(json.dumps(data), encoding="utf-8")
    draft = load_inputs(cfg, None, None, None, None, review_draft=True)
    decisions = extend_template(draft, data, aliases)
    for q in data["questions"]:
        for key, decision in decisions["questions"][q["qid"]]["classes"].items():
            decision.update(judgment="accepted" if key == q["source"]["canonical_answer"] else "rejected",
                            reviewer="fixture adjudicator", rationale="Negative control.")
    with pytest.raises(ValueError, match="conflicts with the source answer rules"):
        compile_review(draft, data, aliases, decisions)


def test_reviewed_local_only_spelling_without_an_eligible_global_class(tmp_path):
    cfg, data, registry, _ = config_for_counterexample(tmp_path)
    q = data["questions"][-1]
    q["answer"]["raw"] = q["source"]["canonical_answer"] + " [accept a context-dependent description]"
    term = "a context-dependent description"
    q["answer"]["accepted"].append(term)
    Path(cfg["inputs"]["dataset"]).write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="unreviewed source rule"):
        load_inputs(cfg, None, None, None, None, review_draft=True)
    registry["source_review"] = {"dataset_sha256": digest(data), "context_only_by_qid": {q["qid"]: [term]}}
    Path(cfg["inputs"]["oracle"]).write_text(json.dumps(registry), encoding="utf-8")
    run = load_inputs(cfg, None, None, None, None, review_draft=True)
    decisions = extend_template(run, data, registry)
    for source in data["questions"]:
        for label, decision in decisions["questions"][source["qid"]]["classes"].items():
            decision.update(judgment="accepted" if label == source["source"]["canonical_answer"] else "rejected",
                            reviewer="fixture adjudicator", rationale="Reviewed for the specific source.")
    Path(cfg["inputs"]["class_review"]).write_text(json.dumps(decisions), encoding="utf-8")
    assert load_inputs(cfg, None, None, None, None).identity["source_review"]["reviewed_questions"] == 10
    with pytest.raises(ValueError, match="unreviewed answer"):
        MenuOracle([], aliases=registry).class_id(term)
    registry["source_review"]["context_only_by_qid"] = {data["questions"][0]["qid"]: [term]}
    Path(cfg["inputs"]["oracle"]).write_text(json.dumps(registry), encoding="utf-8")
    with pytest.raises(ValueError, match="unreviewed source rule"):
        load_inputs(cfg, None, None, None, None, review_draft=True)

@pytest.mark.parametrize("scope",[None,"scentific"])
def test_external_unclassified_data_is_rejected(tmp_path, scope):
    cfg,dataset,_,_=config_for_counterexample(tmp_path)
    cfg["inputs"].pop("class_review",None)
    if scope is None:
        dataset.pop("evidence_scope")
    else:
        dataset["evidence_scope"]=scope
    Path(cfg["inputs"]["dataset"]).write_text(json.dumps(dataset),encoding="utf-8")
    with pytest.raises(ValueError,match="evidence_scope"):
        load_inputs(cfg,None,None,None,None)
