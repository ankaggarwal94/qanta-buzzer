"""CI launch guards and frozen public transport, without provider calls."""
import ast
import gzip
import hashlib
import json
from pathlib import Path

import pytest
import yaml

WORKFLOW = Path(__file__).resolve().parents[1]/'.github/workflows/imcqa-tuned-fresh.yml'


def workflow():
    return yaml.load(WORKFLOW.read_text(), Loader=yaml.BaseLoader)


def restoration_source():
    step = next(s for s in workflow()['jobs']['score']['steps'] if s.get('name') == 'Restore and verify frozen public-only transport')
    return step['run'].split("python - <<'PY'\n",1)[1].rsplit('\nPY',1)[0]


def test_one_scoped_credential_step_after_all_cpu_gates():
    data = workflow()
    assert data['on']=={'push':{'branches':['feat/imcqa-tuned-policy-comparison-20261005'],
        'paths':['.github/workflows/imcqa-tuned-fresh.yml']}}
    assert data['permissions']=={'contents':'read'}
    job=data['jobs']['score']
    assert int(job['timeout-minutes'])==170
    for gate in ("github.event_name == 'push'", "github.actor == 'ankaggarwal94'",
                 'github.run_attempt == 1', "'ops: run frozen tuned IMCQA comparison within six dollars'"):
        assert gate in job['if']
    steps=job['steps']
    secret_indices=[i for i,s in enumerate(steps) if 'MODAL_TOKEN_SECRET' in s.get('env',{})]
    assert len(secret_indices)==1
    i=secret_indices[0]
    assert steps[i-1]['run'].endswith('--dry-run')
    assert '--dry-run' not in steps[i]['run']
    assert steps[i]['run']==steps[i-1]['run'].removesuffix(' --dry-run')
    assert int(steps[i]['timeout-minutes'])==160
    assert all('--run-id imcqa-tuned-fresh-20261006' in steps[k]['run'] for k in (i-1,i))
    assert '--max-cost-usd 6.00' in steps[i]['run']
    assert '2026-10-06T01:41:54Z' in steps[i]['run']
    assert 'evaluator' not in steps[i]['run'] and 'frozen_policies' not in steps[i]['run']
    assert steps[-1]['if']=='always()' and steps[-1]['uses']=='actions/upload-artifact@v4'
    assert steps[-1]['with']['path']=='imcqa_tuned_run_artifacts/'
    assert data['concurrency']['cancel-in-progress']=='false'
    tree=ast.parse(restoration_source())
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and
                   n.func.attr in ('spawn','remote','connect','function') for n in ast.walk(tree))


@pytest.fixture
def transport(tmp_path,monkeypatch):
    """Mock only scientific parsing and pricing, leaving byte/identity gates real."""
    from scripts import imcqa_tuned_design as design
    from scripts import modal_imcqa_tuned as runner
    monkeypatch.chdir(tmp_path)
    root=tmp_path/'imcqa_tuned_public'
    root.mkdir()
    (tmp_path/'source.py').write_bytes(b'pinned source\n')
    monkeypatch.setattr(runner,'SOURCES',('source.py',))
    package={'n_questions':850,'model':design.MODEL,'jobs':[],
        'policy_lock_sha256':'a'*64,'sample_size_planning_sha256':'b'*64,
        'selection_id':'c'*64,'main_dataset_sha256':'d'*64,'source_input_sha256':'e'*64}
    raw=json.dumps(package).encode()
    packed=gzip.compress(raw,mtime=0)
    (root/'public.json.gz').write_bytes(packed)
    cfg={'policy_lock_sha256':package['policy_lock_sha256'],
         'sample_size_planning_sha256':package['sample_size_planning_sha256'],
         'fresh_questions':850,'new_plain_score_contexts':34000,'new_wait_score_contexts':0,
         'post_result_changes_allowed':False}
    (tmp_path/'configs').mkdir()
    config=tmp_path/'configs/imcqa_tuned_fresh.json'
    config.write_text(json.dumps(cfg))
    sha=lambda content:hashlib.sha256(content).hexdigest()
    manifest={'schema_version':'imcqa-tuned-transport-v1','protocol':design.PROTOCOL,
        'run_id':'imcqa-tuned-fresh-20261006','public_file':'public.json.gz',
        'compressed_sha256':sha(packed),'compressed_bytes':len(packed),
        'public_sha256':sha(raw),'public_bytes':len(raw),'n_questions':850,
        'contexts_per_model':34000,'model':design.MODEL,'config_sha256':sha(config.read_bytes()),
        'source_files_sha256':{'source.py':sha((tmp_path/'source.py').read_bytes())},
        'pricing':{'max_cost_usd':'6.00','rate_usd_per_second':'0.00063924',
                   'rate_verified_utc':'2026-10-06T01:41:54Z','rate_source_url':'https://modal.com/pricing'},
        'real_gold_labels_included':False,'new_fits':0}
    manifest.update({k:package[k] for k in ('policy_lock_sha256','sample_size_planning_sha256',
                                           'selection_id','main_dataset_sha256','source_input_sha256')})
    (root/'transport_manifest.json').write_text(json.dumps(manifest))
    monkeypatch.setattr(design,'validate_public_package',lambda value:[None]*34000)
    monkeypatch.setattr(runner,'budget_plan',lambda n,**kwargs:{'worker_timeout_seconds':8250,
        'internal_deadline_seconds':8130,'reserved_estimate_usd':'5.53254008'})
    return root,manifest,raw


def test_transport_restores_exact_public_bytes_without_provider(transport):
    root,manifest,raw=transport
    exec(compile(restoration_source(),'workflow-transport','exec'),{})
    out=root.parent/'imcqa_tuned_run_artifacts'
    assert (out/'public.json').read_bytes()==raw
    assert json.loads((out/'transport_manifest.json').read_text())==manifest
    assert not (out/'run').exists()


@pytest.mark.parametrize('mutation',['gzip_bytes','raw_checksum','path','ceiling','policy_binding','source_bytes','config_bytes','too_large','extra_field'])
def test_transport_corruption_cannot_reach_launch(transport,mutation):
    root,manifest,_=transport
    if mutation=='gzip_bytes':
        (root/'public.json.gz').write_bytes(b'changed')
    elif mutation=='raw_checksum':
        manifest['public_sha256']='0'*64
    elif mutation=='path':
        manifest['public_file']='../other.gz'
    elif mutation=='ceiling':
        manifest['pricing']['max_cost_usd']='60.00'
    elif mutation=='policy_binding':
        manifest['policy_lock_sha256']='0'*64
    elif mutation=='source_bytes':
        (root.parent/'source.py').write_bytes(b'changed')
    elif mutation=='config_bytes':
        (root.parent/'configs/imcqa_tuned_fresh.json').write_text('{}')
    elif mutation=='too_large':
        manifest['public_bytes']=128*1024**2+1
    else:
        manifest['extra']='unreviewed'
    (root/'transport_manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        exec(compile(restoration_source(),'workflow-transport','exec'),{})
    assert not (root.parent/'imcqa_tuned_run_artifacts').exists()
