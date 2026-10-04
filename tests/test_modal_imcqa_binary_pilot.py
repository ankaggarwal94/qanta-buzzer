"""No provider calls: one create-once worker and the exact eighty-cent bound."""
from copy import deepcopy
from decimal import Decimal
from pathlib import Path

import pytest
from scripts import modal_imcqa_binary_pilot as runner


def test_timeout_reservation_includes_startup_cpu_and_contingency():
    plan = runner.budget_plan()
    computed = Decimal(plan["allocation_rate_usd_per_second"])*(900+92)+Decimal(".10")+Decimal(".01723232")
    assert computed == Decimal(plan["reserved_estimate_usd"]) == Decimal(".75135840")
    assert computed < Decimal(plan["ceiling_usd"])
    assert plan["automatic_retries"] == 0 and plan["gpu_calls"] == 1
    assert runner.MODEL_LIMITS == {"qwen7b":{"timeout":900,"deadline":780}}
    runner.validate_plan(plan)


@pytest.mark.parametrize("field,value", [("ceiling_usd","1"),("automatic_retries",1),("gpu_calls",2)])
def test_budget_mutation_fails_closed(field,value):
    plan=runner.budget_plan();plan[field]=value
    with pytest.raises(ValueError):runner.validate_plan(plan)


def test_public_hash_checked_before_provider_access():
    with pytest.raises(ValueError,match="hash differs"):
        runner.verify_public(b"{}","0"*64)


def test_new_volume_and_all_transitive_project_modules_are_packaged():
    assert runner.RUN_ID == "imcqa-binary-dev-20261004"
    assert runner.PRIOR_RUN == "imcqa-protocol-dev-20261004"
    assert runner.RUN_ID != runner.PRIOR_RUN != runner.CACHE_RUN
    for name in ("scripts/imcqa_binary_design.py","scripts/imcqa_protocol_design.py","scripts/imcqa_wait_scoring.py"):
        assert name in runner.SOURCES


def test_frozen_config_and_source_manifest_match_local_files():
    root=Path(__file__).resolve().parents[1]
    control={"budget":runner.budget_plan(),"run_id":runner.RUN_ID,"cache_run":runner.CACHE_RUN,"prior_run":runner.PRIOR_RUN,
             "source_files_sha256":{name:runner.digest(root/name) for name in runner.SOURCES}}
    runner.verify_sources(root,control)
    changed=deepcopy(control);changed["source_files_sha256"][runner.SOURCES[0]]="0"*64
    with pytest.raises(ValueError,match="sources differ"):
        runner.verify_sources(root,changed)
