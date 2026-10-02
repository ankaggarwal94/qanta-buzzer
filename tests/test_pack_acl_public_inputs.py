"""Lossless transport and fail-closed public input boundary tests."""
import copy
import json
from pathlib import Path

import pytest

from qb_data.jane_paired import build_jobs
from scripts.prepare_acl_expansion import word_prefixes
from scripts.prepare_jane_gpu_pilot import build_choice_controls
from scripts.pack_acl_public_inputs import compact_inputs, digest, json_bytes, pack, restore, NAMES


@pytest.fixture
def inputs(tmp_path):
    questions=[]
    for i,split in enumerate(("calibration","selection","test")):
        question="  "+"  ".join(f"word{i}_{j}" for j in range(51))+" café — Noël  "
        options=[{"id":letter,"text":f"{letter} café {i}"} for letter in "ABCD"]
        questions.append({"qid":f"qbreader:{i:024x}","group_id":"component:"+digest(str(i)),"split":split,
            "question":question,"prefixes":word_prefixes(question),
            "answer":{"raw":"fixture", "accepted":[options[0]["text"]],"rejected":[],"prompt":[]},
            "menus":[{"condition":condition,"menu_id":"fixed_1","gold_option_id":"A","options":options,
                      "provenance":{"fixture":True}} for condition in ("independent_pool","same_category_pool")]})
    dataset={"schema_version":"jane-paired-v1","evidence_scope":"scientific","prompt_template":"concise_json_v4",
        "source":{"origin":"synthetic fixture","provenance":"synthetic fixture","full_answerlines_available":True},"questions":questions}
    main={"schema_version":"jane-public-jobs-v1","evidence_scope":"scientific","jobs":build_jobs(dataset)}
    controls=build_choice_controls(dataset); controls["evidence_scope"]="scientific"
    public=tmp_path/"public";public.mkdir()
    raws={NAMES[0]:json_bytes(main),NAMES[1]:json_bytes(controls)}
    for name,raw in raws.items():(public/name).write_bytes(raw)
    return public,{name:digest(raw) for name,raw in raws.items()},main,controls


def test_exact_roundtrip_unicode_whitespace_and_create_once(inputs,tmp_path):
    public,hashes,_,_=inputs
    transport=tmp_path/"transport"
    summary=pack(public,transport,hashes)
    assert summary["compressed_bytes"] < sum((public/name).stat().st_size for name in NAMES)
    out=tmp_path/"restored"
    receipt=restore(transport,out,hashes)
    assert set(p.name for p in out.iterdir()) == set(NAMES)
    assert all((out/name).read_bytes()==(public/name).read_bytes() for name in NAMES)
    assert receipt["transport_manifest_sha256"]==digest((transport/"transport_manifest.json").read_bytes())
    with pytest.raises(FileExistsError):restore(transport,out,hashes)


def test_private_metadata_is_rejected_not_dropped(inputs):
    _,_,main,controls=inputs
    bad=copy.deepcopy(main)
    bad["jobs"][0]["gold_option_id"]="A"
    with pytest.raises(ValueError,match="unexpected public job fields"):
        compact_inputs(bad,controls)


def test_external_frozen_identity_required(inputs,tmp_path):
    public,hashes,_,_=inputs
    with pytest.raises(ValueError,match="externally expected"):
        pack(public,tmp_path/"transport",{name:"0"*64 for name in NAMES})
    assert not (tmp_path/"transport").exists()


@pytest.mark.parametrize("mutation",["path","bitflip","oversize","extra_file","symlink"])
def test_corrupt_or_unsafe_transport_never_publishes(inputs,tmp_path,mutation):
    public,hashes,_,_=inputs
    transport=tmp_path/"transport";pack(public,transport,hashes)
    manifest_path=transport/"transport_manifest.json"
    manifest=json.loads(manifest_path.read_text())
    chunk=transport/manifest["chunks"][0]["name"]
    if mutation=="path":manifest["chunks"][0]["name"]="../outside.b64"
    elif mutation=="bitflip":
        raw=chunk.read_bytes();chunk.write_bytes((b"A" if raw[:1]!=b"A" else b"B")+raw[1:])
    elif mutation=="oversize":manifest["compact_bytes"]=10**12
    elif mutation=="extra_file":(transport/"answers.json").write_text("{}")
    elif mutation=="symlink":
        copied=tmp_path/"outside.b64";copied.write_bytes(chunk.read_bytes());chunk.unlink();chunk.symlink_to(copied)
    manifest_path.write_bytes(json_bytes(manifest))
    with pytest.raises(ValueError):restore(transport,tmp_path/"restored",hashes)
    assert not (tmp_path/"restored").exists()
