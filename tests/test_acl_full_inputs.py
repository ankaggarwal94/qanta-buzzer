"""A private review draft cannot be mistaken for a reviewed equivalence registry."""
import gzip,json
import pytest

from scripts.acl_full_inputs import build_draft, compile_decisions, _create_or_match
from scripts.acl_upstream_conformance import FIXTURE
from qb_data.acl_menus import MenuOracle


def inputs():
    with gzip.open(FIXTURE,"rb") as f:
        data = json.load(f)
    return data["dataset"],data["reservoir"]


def completed(sources=None):
    dataset,donors = sources if sources is not None else inputs()
    rows = build_draft(dataset,donors)
    rows["status"] = "reviewed"
    rows["reviewer"] = "Synthetic test adjudicator"
    for row in rows["records"].values():
        row["status"] = "reviewed"
        row["canonical_class"] = row["source_canonical"]
        row["global_aliases"] = row["source_aliases"]
        row["rationale"] = "Synthetic fixture identity explicitly established for unit tests."
    return dataset,donors,rows


def test_draft_rejected_by_loader_and_compiler():
    d,r=inputs()
    draft=build_draft(d,r)
    with pytest.raises(ValueError):
        MenuOracle([],aliases=draft)
    with pytest.raises(ValueError,match="reviewed"):
        compile_decisions(d,r,draft)


def test_completed_synthetic_registry_preserves_context_only_distinction():
    d,r,rows=completed()
    source=next(iter(rows["records"].values()))
    source["context_only"]=["conditional nickname"]
    registry=compile_decisions(d,r,rows)
    oracle=MenuOracle([],aliases=registry)
    assert oracle.class_id(source["source_aliases"][0]) == source["canonical_class"]
    with pytest.raises(ValueError,match="unreviewed"):
        oracle.class_id("conditional nickname")


def test_source_only_acceptance_keeps_an_independent_global_class():
    d,r=inputs()
    target=d["questions"][0]
    alternative=d["questions"][1]["source"]["canonical_answer"]
    target["answer"]["accepted"].append(alternative)
    _,_,rows=completed((d,r))
    row=rows["records"][target["qid"]]
    row["global_aliases"]=[a for a in row["global_aliases"] if a!=alternative]
    row["context_only"]=[alternative]
    registry=compile_decisions(d,r,rows)
    oracle=MenuOracle([],aliases=registry)
    assert oracle.class_id(alternative)==alternative
    assert alternative not in registry["classes"][row["canonical_class"]]
    assert registry["source_review"]["context_only_by_qid"][target["qid"]]==[alternative]


def test_unregistered_source_only_acceptance_remains_private_to_its_qid():
    d,r=inputs()
    target=d["questions"][0]
    alternative="fixture shorthand valid for this question"
    target["answer"]["accepted"].append(alternative)
    _,_,rows=completed((d,r))
    row=rows["records"][target["qid"]]
    row["global_aliases"]=[a for a in row["global_aliases"] if a!=alternative]
    row["context_only"]=[alternative]
    registry=compile_decisions(d,r,rows)
    with pytest.raises(ValueError,match="unreviewed"):
        MenuOracle([],aliases=registry).class_id(alternative)
    assert registry["source_review"]["context_only_by_qid"]=={target["qid"]:[alternative]}


@pytest.mark.parametrize("change",["stale","coverage","status","reason","bad_alias","scoped_overlap"])
def test_invalid_decisions(change):
    d,r,rows=completed()
    rec=next(iter(rows["records"].values()))
    if change=="stale":
        rec["source_sha256"]="f"*64
    if change=="coverage":
        rows["records"].popitem()
    if change=="status":
        rec["status"]="pending"
    if change=="reason":
        rec["rationale"]=" "
    if change=="bad_alias":
        rec["global_aliases"]=["unrelated"]
    if change=="scoped_overlap":
        rec["context_only"]=[rec["source_canonical"]]
    with pytest.raises(ValueError):
        compile_decisions(d,r,rows)


def test_repeated_publication_preserves_existing_bytes(tmp_path):
    path=tmp_path/"private.json"
    _create_or_match(path,b"one")
    _create_or_match(path,b"one")
    with pytest.raises(ValueError,match="differs"):
        _create_or_match(path,b"two")
    assert path.read_bytes()==b"one"

def test_optional_menu_draft_command(tmp_path,monkeypatch):
    import sys
    import yaml
    from scripts.acl_full_inputs import main
    from scripts.acl_archive_compatibility import CROSSWALK

    d,r,rows=completed()
    registry=compile_decisions(d,r,rows)
    for name,value in (("dataset",d),("reservoir",r),("oracle",registry)):
        (tmp_path/name).write_text(json.dumps(value),encoding="utf-8")
    config={"inputs":{**{name:str(tmp_path/name) for name in ("dataset","reservoir","oracle")},
                      "split_roles":CROSSWALK},
            "arm":{"version":1,"k":4,"seed":1,"menu":"baseline","reward":{"arm":"baseline"}}}
    (tmp_path/"config").write_text(yaml.safe_dump(config),encoding="utf-8")
    output=tmp_path/"menu.json"
    monkeypatch.setattr(sys,"argv",["acl_full_inputs","menu-draft","--config",str(tmp_path/"config"),
                                    "--out",str(output)])
    main()
    draft=json.loads(output.read_text(encoding="utf-8"))
    assert len(draft["questions"])==len(d["questions"])
    assert all(q["status"]=="unresolved" for q in draft["questions"].values())
