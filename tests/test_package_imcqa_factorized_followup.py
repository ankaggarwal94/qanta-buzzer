"""Archive safety and complete member hash/CRC validation without experiment data."""
import hashlib
import json
from pathlib import Path
import zipfile

import pytest
from scripts import package_imcqa_factorized_followup as p


@pytest.mark.parametrize("name",["../escape","/absolute","a/../b","a//b","a\\b","./a",""])
def test_archive_rejects_unsafe_member_paths(name):
    with pytest.raises(ValueError): p.valid_name(name)


def test_zip_hashes_generated_readme_and_exact_input_bytes(tmp_path):
    source=tmp_path/'data.json';source.write_bytes(b'{"value":17}\n')
    out=tmp_path/'evidence.zip'
    result=p.write_archive(out,[p.Entry('inputs/data.json',path=source),p.Entry('README.txt',data=b'Useful instructions\n')],{'scope':'test'})
    assert result['files']==3 and result['sha256']==p.digest(out)
    with zipfile.ZipFile(out) as archive:
        manifest=json.loads(archive.read('artifact_manifest.json'))
        assert {row['path'] for row in manifest['files']}=={'inputs/data.json','README.txt'}
        for row in manifest['files']:
            assert hashlib.sha256(archive.read(row['path'])).hexdigest()==row['sha256']
    with pytest.raises(FileExistsError):
        p.write_archive(out,[],{})


def test_duplicate_member_and_symlink_rejected_before_creation(tmp_path):
    path=tmp_path/'x';path.write_text('x')
    link=tmp_path/'link';link.symlink_to(path)
    with pytest.raises(ValueError,match='symlink'):p.Entry('link',path=link)
    with pytest.raises(ValueError,match='duplicate'):
        p.write_archive(tmp_path/'bad.zip',[p.Entry('x',data=b'a'),p.Entry('x',data=b'b')],{})
    assert not (tmp_path/'bad.zip').exists()


def test_weight_and_credential_inputs_rejected_provider_workspace_omitted(tmp_path):
    (tmp_path/'workspace.json').write_text('{}')
    (tmp_path/'receipt.json').write_text('{}')
    assert [entry.name for entry in p.collect_directory(tmp_path,'run')]==['run/receipt.json']
    (tmp_path/'weights.safetensors').write_bytes(b'x')
    with pytest.raises(ValueError,match='weight'):p.collect_directory(tmp_path,'run')


def test_reproduction_commands_are_cpu_only_and_recovery_is_explicit():
    basic=p.reproduction_script(False); recovery=p.reproduction_script(True)
    assert 'modal_imcqa' not in basic and '--recovery-outputs' not in basic
    assert '--recovery-outputs recovery_run/output' in recovery
    assert 'record["outputs"], reference, plan["jobs"]' in basic


def execution_fixture():
    return {"schema_version":"imcqa-factorized-execution-v1","analysis_source_commit":"a"*40,
            "launch_commit":"b"*40,"workflow_id":17,
            "workflow_url":"https://github.com/ankaggarwal94/qanta-buzzer/actions/runs/17",
            "recovery_launch_commit":"c"*40,"recovery_workflow_id":19,
            "recovery_workflow_url":"https://github.com/ankaggarwal94/qanta-buzzer/actions/runs/19"}


@pytest.mark.parametrize("field,value",[("analysis_source_commit","0"*40),("launch_commit","0"*40),
    ("workflow_id",18),("workflow_url","https://github.com/ankaggarwal94/qanta-buzzer/actions/runs/20"),
    ("recovery_launch_commit","0"*40),("recovery_workflow_id",20)])
def test_execution_ledger_rejects_mismatched_identity(field,value):
    record=execution_fixture();record[field]=value
    with pytest.raises(ValueError):
        p.validate_execution(record,source_commit="a"*40,binary_launch_commit="b"*40,numerics_launch_commit="b"*40,
            workflow_urls=[execution_fixture()["workflow_url"],execution_fixture()["recovery_workflow_url"]],recovery_launch_commit="c"*40)


def test_execution_accepts_bound_recovery_and_rejects_unrelated_declared_workflow():
    record=execution_fixture()
    keywords=dict(source_commit="a"*40,binary_launch_commit="b"*40,numerics_launch_commit="b"*40,
                  workflow_urls=[record['workflow_url'],record['recovery_workflow_url']],recovery_launch_commit="c"*40)
    p.validate_execution(record,**keywords)
    keywords['workflow_urls']=[record['workflow_url']]
    with pytest.raises(ValueError,match='workflow URLs'):p.validate_execution(record,**keywords)


def test_recovery_analyzer_is_hash_bound_to_completed_receipt(tmp_path):
    content=b'original recovery analyzer'
    (tmp_path/'analysis_receipt.json').write_text(json.dumps({'analyzer_sha256':hashlib.sha256(content).hexdigest()}))
    filename='analyze_imcqa_protocol_recovery.py'
    p.validate_analyzer_sources({'source/scripts/'+filename:content},[(tmp_path,filename)])
    with pytest.raises(ValueError,match='completed analyzer'):
        p.validate_analyzer_sources({'source/scripts/'+filename:b'changed'},[(tmp_path,filename)])
