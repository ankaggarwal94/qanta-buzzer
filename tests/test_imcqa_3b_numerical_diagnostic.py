"""Numerical misses remain findings; semantic or reference failures stay visible."""
from copy import deepcopy
from decimal import Decimal
from pathlib import Path
import json

import pytest

from scripts import imcqa_3b_numerical_diagnostic as diagnostic
from scripts import imcqa_protocol_design as design
from scripts import modal_imcqa_3b_numerical_diagnostic as runner


def job(mapping=None, actions="ABCDE"):
    return {"score_id": "example", "allowed_actions": actions,
            "option_source_ids": mapping or design.mapping(0)}


def test_logit_failure_does_not_raise_or_hide_stable_choices():
    reference = [{"logits": [50, 30, 20, 10, 0]}]
    candidate = [{"logits": [50, 30.01, 20, 10, 0]}]
    report = diagnostic.compare(candidate, reference, [job()])
    assert not report["passed"]
    assert report["rows"][0]["logit_violation_labels"] == ["B"]
    assert report["action_argmax_changes"] == report["candidate_argmax_changes"] == 0
    assert report["max_action_probability_difference"] < .001


def test_candidate_flip_is_detected_when_wait_action_stays_fixed():
    report = diagnostic.compare([{"logits": [0, .00001, -1, -2, 10]}],
                                [{"logits": [.00001, 0, -1, -2, 10]}], [job()])
    assert not report["passed"]
    assert report["candidate_argmax_changes"] == 1
    assert report["action_argmax_changes"] == 0
    assert report["rows"][0]["logit_violation_labels"] == []


def test_relabelled_wait_does_not_enter_candidate_probability_normalizer():
    report = diagnostic.compare([{"logits": [10, 0, 0, 0, .00001]}],
                                [{"logits": [10, .00001, 0, 0, 0]}], [job(design.mapping(0, "A"))])
    assert not report["passed"] and report["candidate_argmax_changes"] == 1
    assert report["action_argmax_changes"] == 0


def test_exact_reference_and_same_values_pass_unchanged_tolerances():
    values = [{"logits": [1, 2, 3, 4, 5]}]
    report = diagnostic.compare(values, deepcopy(values), [job()])
    assert report["passed"] and report["max_logit_difference"] == 0
    assert report["atol"] == .001 and report["rtol"] == .00001 and report["probability_atol"] == .001


@pytest.mark.parametrize("left,right,jobs", [([], [], []), ([{"logits": [1]*5}], [], [job()]),
                                           ([{"logits": [1, 2, 3, 4, float('nan')]}], [{"logits": [1]*5}], [job()])])
def test_missing_or_nonfinite_evidence_is_an_error(left, right, jobs):
    with pytest.raises(ValueError):
        diagnostic.compare(left, right, jobs)


def test_volume_and_collected_artifact_paths_resolve_same_public_location(tmp_path):
    (tmp_path/"pilot.json").write_text("collected")
    assert diagnostic.locate(tmp_path, "public/pilot.json") == tmp_path/"pilot.json"
    (tmp_path/"public").mkdir()
    (tmp_path/"public/pilot.json").write_text("volume")
    assert diagnostic.locate(tmp_path, "public/pilot.json") == tmp_path/"public/pilot.json"
    assert diagnostic.locate(tmp_path, "output/receipt.json") == tmp_path/"output/receipt.json"


def test_fifty_cent_reservation_includes_startup_staging_and_contingency():
    plan = runner.budget_plan()
    expected = Decimal(".00063924")*(360+90+2)+Decimal(".01723232")+Decimal(".10")
    assert expected == Decimal(".40616880") == Decimal(plan["reserved_estimate_usd"])
    assert expected < Decimal(plan["ceiling_usd"])
    assert plan["gpu_calls"] == 1 and plan["automatic_retries"] == 0
    assert plan["internal_deadline_seconds"] < plan["worker_timeout_seconds"]


def test_exact_manifest_hash_is_checked_before_provider_or_parsing():
    with pytest.raises(ValueError, match="hash differs"):
        runner.verify_manifest(b"{}")


def test_frozen_config_has_no_automatic_production_or_tolerance_relaxation():
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root/"configs/imcqa_3b_numerical_diagnostic.json").read_text())
    assert config["budget"] == runner.budget_plan()
    assert config["manifest_sha256"] == runner.MANIFEST_SHA256
    assert config["production_approved"] is config["automatic_production_launch"] is False
    assert config["numerical_checks"]["raw_logit_atol"] == .001
    assert config["numerical_checks"]["raw_logit_rtol"] == .00001
    assert config["contexts"] == 20 and config["logical_evaluations"] == 200 and config["model_forward_calls"] == 117


def test_frozen_sources_and_budget_verify_before_allocation():
    root = Path(__file__).resolve().parents[1]
    control = {"budget": runner.budget_plan(), "run_id": runner.RUN_ID, "cache_run": runner.CACHE_RUN,
               "protocol_run": runner.PROTOCOL_RUN, "prior_run": runner.PRIOR_RUN,
               "manifest_sha256": runner.MANIFEST_SHA256,
               "source_files_sha256": {name: runner.digest(root/name) for name in runner.SOURCES}}
    runner.verify_sources(root, control)
    control["budget"]["automatic_retries"] = 1
    with pytest.raises(ValueError, match="budget"):
        runner.verify_sources(root, control)
