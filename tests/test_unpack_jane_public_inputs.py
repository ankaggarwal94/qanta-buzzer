"""Public transport must preserve exact frozen bytes and exclude evaluator data."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import pytest

from scripts.unpack_jane_public_inputs import inspect_archive, unpack


def content_fixture():
    job = {"job_id": "job", "qid": "question", "group_id": "group", "split": "test",
           "format": "oe", "condition": "oe", "menu_id": None, "prefix_id": "p1",
           "fraction": 1, "prompt": "public clue", "prompt_sha256": hashlib.sha256(b"public clue").hexdigest()}
    from scripts.jane_gpu_backend import build_choice_control_prompt
    options = [{"id": letter, "text": letter + " option"} for letter in "ABCD"]
    prompt = build_choice_control_prompt(options)
    control = {key: value for key, value in job.items() if key not in {"fraction", "prefix_id"}}
    control.update(format="mc", condition="pool", menu_id="fixed", options=options,
                   prompt=prompt, prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest())
    content = {}
    for phase in ("dev", "main"):
        for suffix, schema, row in (("jobs", "jane-public-jobs-v1", job),
                                   ("choices_only", "jane-choice-controls-v1", control)):
            content[f"{phase}_{suffix}.json"] = json.dumps({"schema_version": schema,
                "evidence_scope": "engineering_smoke", "jobs": [row]}).encode()
    return with_manifest(content)


def with_manifest(content):
    result = deepcopy(content)
    result["manifest.json"] = json.dumps({"schema_version": "jane-public-input-manifest-v1",
        "files": {name: {"sha256": hashlib.sha256(raw).hexdigest(), "byte_count": len(raw),
                         "job_count": 1} for name, raw in result.items() if name != "manifest.json"}}).encode()
    return result


def make_archive(path, content):
    with ZipFile(path, "x") as bundle:
        for name, raw in content.items():
            bundle.writestr(name, raw)


def test_frozen_bytes_are_restored_once(tmp_path):
    content = content_fixture()
    archive, out = tmp_path / "inputs.zip", tmp_path / "public"
    make_archive(archive, content)
    assert inspect_archive(archive) == content
    receipt = unpack(archive, out)
    assert receipt["archive_sha256"] == hashlib.sha256(archive.read_bytes()).hexdigest()
    assert {path.name: path.read_bytes() for path in out.iterdir()} == content
    with pytest.raises(FileExistsError):
        unpack(archive, out)


@pytest.mark.parametrize("name", ["../outside", "evaluator/gold.json", "extra.json"])
def test_extra_private_or_unsafe_members_fail_before_output(tmp_path, name):
    content = content_fixture()
    content[name] = b"private"
    archive, out = tmp_path / "inputs.zip", tmp_path / "public"
    make_archive(archive, content)
    with pytest.raises(ValueError, match="five approved"):
        unpack(archive, out)
    assert not out.exists()


def test_tampered_public_bytes_fail_before_output(tmp_path):
    content = content_fixture()
    content["main_jobs.json"] += b" "
    archive = tmp_path / "inputs.zip"
    make_archive(archive, content)
    with pytest.raises(ValueError, match="identity mismatch"):
        inspect_archive(archive)


def test_gold_rejected_even_if_manifest_hashes_agree(tmp_path):
    content = content_fixture()
    package = json.loads(content["main_jobs.json"])
    package["jobs"][0]["gold_answer"] = "private"
    content["main_jobs.json"] = json.dumps(package).encode()
    content = with_manifest(content)
    archive = tmp_path / "inputs.zip"
    make_archive(archive, content)
    with pytest.raises(ValueError, match="unexpected public job keys"):
        inspect_archive(archive)
