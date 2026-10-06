"""No-provider tests for bounded launch gates and source/input identity."""
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from pathlib import Path
import pytest
from scripts import modal_imcqa_tuned as runner

NOW = datetime(2026, 10, 6, 1, 0, tzinfo=timezone.utc)


def plan(**changes):
    args = dict(max_cost_usd="5", rate_usd_per_second="0.00063924",
                rate_verified_utc=NOW.isoformat(), rate_source_url="https://modal.com/pricing", now=NOW)
    args.update(changes)
    return runner.budget_plan(600, **args)


def test_one_allocation_bounded_for_predeclared_question_count():
    value = plan()
    assert Decimal(value["reserved_estimate_usd"]) == Decimal("0.00063924")*6092+Decimal(".20")
    assert value["worker_timeout_seconds"] == 6000
    assert value["internal_deadline_seconds"] == 5880
    assert value["automatic_retries"] == 0 and value["gpu_calls"] == 1
    runner.validate_plan(value, now=NOW)


@pytest.mark.parametrize("changes", [
    {"max_cost_usd":"3"}, {"max_cost_usd":"NaN"}, {"rate_usd_per_second":"0"},
    {"rate_usd_per_second":"Infinity"}, {"rate_source_url":"https://example.com"},
    {"rate_verified_utc":(NOW-timedelta(days=2)).isoformat()},
    {"rate_verified_utc":(NOW+timedelta(seconds=1)).isoformat()},
    {"rate_verified_utc":"2026-10-06T01:00:00"},
])
def test_unbounded_or_unverified_pricing_rejected(changes):
    with pytest.raises(ValueError):
        plan(**changes)


@pytest.mark.parametrize("key,value", [("automatic_retries",1),("gpu_calls",2),
    ("worker_timeout_seconds",6001),("reserved_estimate_usd","0")])
def test_budget_field_tampering_rejected(key,value):
    result = plan()
    result[key]=value
    with pytest.raises(ValueError):
        runner.validate_plan(result, now=NOW)


def test_hash_mismatch_fails_before_parse_or_provider():
    with pytest.raises(ValueError,match="hash differs"):
        runner.verify_public(b"{}","0"*64)


def test_invalid_commit_fails_before_input_or_provider(tmp_path):
    with pytest.raises(ValueError,match="committed source"):
        runner.launch(tmp_path,tmp_path/"missing",tmp_path/"out","HEAD",run_id="imcqa-tuned-test",budget=plan())


def test_source_control_rejects_modified_source_or_count(monkeypatch):
    root=Path(__file__).resolve().parents[1]
    monkeypatch.setattr(runner,"validate_plan",lambda value:None)
    control={"budget":plan(),"run_id":"imcqa-tuned-test","cache_run":runner.CACHE_RUN,
        "prepare_receipt_sha256":runner.PREPARE_SHA256,"production_contexts_per_model":24000,
        "source_files_sha256":{name:runner.digest(root/name) for name in runner.SOURCES}}
    runner.verify_sources(root,control)
    control["production_contexts_per_model"]=23999
    with pytest.raises(ValueError,match="question count"):
        runner.verify_sources(root,control)
    control["production_contexts_per_model"]=24000
    control["source_files_sha256"]["scripts/imcqa_tuned_scoring.py"]="0"*64
    with pytest.raises(ValueError,match="sources differ"):
        runner.verify_sources(root,control)


def test_dry_run_checks_commit_and_never_connects_or_allocates(tmp_path, monkeypatch):
    from scripts import modal_acl_expansion as expansion
    root = Path(__file__).resolve().parents[1]
    public = tmp_path/"public.json"
    public.write_text("{}")
    monkeypatch.setattr(runner,"verify_public",lambda *args:{"jobs":[{}]*24000})
    monkeypatch.setattr(runner,"validate_plan",lambda value:None)
    observed=[]
    monkeypatch.setattr(expansion,"verify_source_commit",lambda repo,commit,hashes:observed.append(commit))
    def forbidden():
        raise AssertionError("provider access is forbidden in dry run")
    monkeypatch.setattr(expansion,"connect",forbidden)
    result=runner.launch(root,public,tmp_path/"out","1"*40,run_id="imcqa-tuned-test",budget=plan(),dry_run=True)
    assert result["status"] == "preflight_only" and result["gpu_executed"] is False
    assert observed == ["1"*40]
    assert not (tmp_path/"out").exists()


def test_collection_cap_scales_with_frozen_context_count():
    assert runner.collection_limit(160) == 160*1024**2
    assert runner.collection_limit(34000) == 34000*32768+16*1024**2
    assert runner.collection_limit(200000) == 200000*32768+16*1024**2
    for bad in (159, 200040, 0, True):
        with pytest.raises(ValueError):
            runner.collection_limit(bad)


def fake_volume(files):
    from types import SimpleNamespace
    return SimpleNamespace(iterdir=lambda *args,**kwargs:[SimpleNamespace(path=name) for name in files],
        read_file=lambda name:iter([files[name]]))


def test_collect_create_once_hashes_and_cumulative_bound(tmp_path,monkeypatch):
    volume=fake_volume({'output/qwen7b/receipt.json':b'{}','output/qwen7b/scores.jsonl':b'{}\n'})
    result=runner.collect(volume,tmp_path,expected_contexts=160)
    assert result['total_bytes']==5
    assert result['files'][1]['sha256']==runner.digest(tmp_path/'output/qwen7b/scores.jsonl')
    with pytest.raises(FileExistsError):
        runner.collect(volume,tmp_path,expected_contexts=160)
    limited=tmp_path/'limited'
    limited.mkdir()
    monkeypatch.setattr(runner,'collection_limit',lambda count:4)
    with pytest.raises(ValueError,match='byte limit'):
        runner.collect(volume,limited,expected_contexts=160)
    assert not (limited/'download_manifest.json').exists()


def test_collect_rejects_traversal_and_symlink_escape(tmp_path):
    with pytest.raises(ValueError,match='safe output'):
        runner.collect(fake_volume({'output/../../bad.json':b'{}'}),tmp_path,expected_contexts=160)
    outside=tmp_path/'outside'
    outside.mkdir()
    root=tmp_path/'collect'
    root.mkdir()
    (root/'output').symlink_to(outside,target_is_directory=True)
    with pytest.raises(ValueError,match='symlink'):
        runner.collect(fake_volume({'output/bad.json':b'{}'}),root,expected_contexts=160)
    assert not (outside/'bad.json').exists()


def test_collect_only_reads_existing_volume_without_gpu(tmp_path,monkeypatch):
    import json
    from types import SimpleNamespace
    from scripts import modal_acl_expansion as expansion
    control={'run_id':'imcqa-tuned-recovery','cache_run':runner.CACHE_RUN,
        'prepare_receipt_sha256':runner.PREPARE_SHA256,'production_contexts_per_model':160}
    control_path=tmp_path/'control.json'
    control_path.write_text(json.dumps(control))
    volume=fake_volume({'output/qwen7b/receipt.json':b'{}'})
    volume.read_file=lambda name:iter([json.dumps(control).encode() if name=='control.json' else b'{}'])
    calls=[]
    def from_name(name,**kwargs):
        calls.append((name,kwargs))
        return volume
    modal=SimpleNamespace(Volume=SimpleNamespace(from_name=from_name))
    monkeypatch.setattr(expansion,'connect',lambda:(modal,{'workspace_name':'fixture'}))
    result=runner.collect_only(control['run_id'],tmp_path/'recovered',control_path)
    assert result['status']=='collected_only' and result['gpu_executed'] is False
    assert calls==[('imcqa-tuned-recovery',{'create_if_missing':False})]
    volume.read_file=lambda name:iter([b'{}'])
    with pytest.raises(ValueError,match='remote allocation control'):
        runner.collect_only(control['run_id'],tmp_path/'wrong',control_path)
    assert not (tmp_path/'wrong').exists()
