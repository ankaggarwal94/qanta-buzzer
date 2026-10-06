"""Bounded credential-local CI inspection with no remote compute APIs."""
import ast
from pathlib import Path
import json
from types import SimpleNamespace

import pytest
import yaml

WORKFLOW = Path(__file__).resolve().parents[1]/'.github/workflows/imcqa-tuned-preflight.yml'


def workflow_and_script():
    data = yaml.load(WORKFLOW.read_text(), Loader=yaml.BaseLoader)
    step = next(s for s in data['jobs']['preflight']['steps'] if 'env' in s)
    source = step['run'].split("python - <<'PY'\n",1)[1].rsplit('\nPY',1)[0]
    return data, step, source


def test_workflow_trigger_is_exact_read_only_and_secret_scoped():
    data, step, source = workflow_and_script()
    assert data['on'] == {'push': {'branches': ['feat/imcqa-tuned-policy-comparison-20261005'],
                                  'paths': ['.github/workflows/imcqa-tuned-preflight.yml']}}
    assert data['permissions'] == {'contents':'read'}
    guard = data['jobs']['preflight']['if']
    assert "github.run_attempt == 1" in guard
    assert "github.actor == 'ankaggarwal94'" in guard
    assert "'ops: verify tuned IMCQA access without inference'" in guard
    tree = ast.parse(source)
    calls = [n for n in ast.walk(tree) if isinstance(n,ast.Call)]
    assert not any(isinstance(n.func,ast.Attribute) and n.func.attr in
                   ('function','spawn','remote','run','batch_upload','create','put_file','commit') for n in calls)
    for n in calls:
        if isinstance(n.func,ast.Attribute) and n.func.attr == 'from_name':
            assert any(k.arg=='create_if_missing' and isinstance(k.value,ast.Constant)
                       and k.value.value is False for k in n.keywords)
    assert 'MODAL_TOKEN' not in source
    assert step['env']['MODAL_TOKEN_ID'] == '${{ secrets.MODAL_TOKEN_ID }}'
    assert step['env']['MODAL_TOKEN_SECRET'] == '${{ secrets.MODAL_TOKEN_SECRET }}'


@pytest.mark.parametrize('missing_model_file',[False,True])
def test_complete_inspection_uses_only_existing_metadata(tmp_path,monkeypatch,capsys,missing_model_file):
    from scripts import modal_acl_expansion as expansion
    from scripts import modal_acl_paired_prompt_scores as scoring
    import subprocess
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('SOURCE_COMMIT','a'*40)
    monkeypatch.setattr(subprocess,'check_output',lambda *a,**kw:'a'*40+'\n')
    revision='a09a35458c702b33eeacc393d103063234e8bc28'
    model={'model':'Qwen/Qwen2.5-7B-Instruct','revision':revision,
           'model_files_sha256':{'config.json':'c'*64,'model.safetensors':'d'*64}}
    prep={'model_receipts':{'qwen7b':model}}
    raw=json.dumps(prep).encode()
    monkeypatch.setattr(scoring,'verify_prepare',lambda data:prep)
    class Volume:
        def __init__(self,name): self.name=name
        def iterdir(self,path,recursive=True):
            if path.startswith('/models/'):
                files=['config.json'] if missing_model_file else ['config.json','model.safetensors']
                return [SimpleNamespace(path=path+'/'+f) for f in files]
            return [SimpleNamespace(path='public/main_choices_only.json')]
    observed=[]
    def from_name(name,create_if_missing):
        assert create_if_missing is False
        observed.append(name)
        return Volume(name)
    modal=SimpleNamespace(Volume=SimpleNamespace(from_name=from_name))
    monkeypatch.setattr(expansion,'connect',lambda:(modal,{'workspace_name':'ankaggarwal94','workspace_id':'fixture'}))
    monkeypatch.setattr(expansion,'remote_read',lambda volume,name,maximum:
        raw if name=='output/prepare_receipt.json' else json.dumps(model).encode())
    _,_,source=workflow_and_script()
    if missing_model_file:
        with pytest.raises(SystemExit): exec(compile(source,'workflow-test','exec'),{})
    else:
        exec(compile(source,'workflow-test','exec'),{})
    result=json.loads((tmp_path/'imcqa_tuned_access_preflight.json').read_text())
    assert result['gpu_calls']==result['cpu_function_calls']==0
    assert result['status']==('failed' if missing_model_file else 'passed')
    assert observed[0]==scoring.CACHE_RUN
    if not missing_model_file:
        assert result['cache']['passed'] is True
        assert all(v['contents_read'] is False for v in result['known_public_inputs'])
