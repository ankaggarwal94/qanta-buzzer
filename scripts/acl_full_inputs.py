"""Materialize hash-pinned private full inputs and compile completed class decisions."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

from qb_data.acl_menus import MenuOracle, _label_key
from qb_data.acl_protocol import ArmSpec, digest, prepare, publish_json
from scripts.acl_archive_compatibility import ARCHIVE_SHA256, CROSSWALK, verify_manifest
from scripts.stopdff_v5.fileio import create_once_bytes
from scripts.stopdff_v5.identity import load_json_strict, sha256_file


PRIVATE = Path("data/acl_expansion")
FILES = {
    "evaluator/main_dataset.json":"d16d8e611965fba3829f01cda936145743b7d46187030e2138caf52068aa9b62",
    "evaluator/reservoir.json":"db37b0322ef0846d19567d287dc9e48652a26ec24dfb94ebf544f054ce8926d3",
}
DECISIONS = {"status","canonical_class","global_aliases","context_only","rationale"}


def _create_or_match(path, data):
    """Publish exact source bytes or accept an identical prior publication."""
    if not path.exists():
        try:
            create_once_bytes(path,data)
            return
        except FileExistsError:
            pass
    if path.is_symlink() or sha256_file(path) != hashlib.sha256(data).hexdigest():
        raise ValueError(f"existing private file differs: {path.name}")


def build_draft(dataset, donors):
    """Bind every answerline decision to immutable private source evidence."""
    records = {}
    def add(key, row, source, raw, canonical, aliases):
        if key in records:
            raise ValueError("duplicate class-review source key")
        records[key] = {
            "source_sha256":digest(row), "source_id":source["_id"], "group_id":row["group_id"],
            "raw_answerline":raw, "source_canonical":canonical, "source_aliases":aliases,
            "status":"unresolved", "canonical_class":"", "global_aliases":[],
            "context_only":[], "rationale":"",
        }
    for q in dataset["questions"]:
        source=q["source"]
        add(q["qid"],q,source,q["answer"]["raw"],source["canonical_answer"],q["answer"]["accepted"])
    for d in donors:
        source=d["source"]
        add("reservoir:"+source["_id"],d,source,source["answer"],d["canonical_answer"],d["aliases"])
    return {"schema_version":"acl-private-class-decisions-v1", "status":"draft", "reviewer":"",
            "dataset_sha256":digest(dataset), "reservoir_sha256":digest(donors), "records":records}


def _text(value):
    return isinstance(value,str) and bool(value.strip())


def compile_decisions(dataset, donors, decisions):
    template = build_draft(dataset,donors)
    if (not isinstance(decisions,dict) or decisions.keys()!=template.keys()
            or decisions["status"]!="reviewed" or not _text(decisions["reviewer"])):
        raise ValueError("a completed reviewed decisions document and named reviewer are required")
    if any(decisions[k]!=template[k] for k in ("schema_version","dataset_sha256","reservoir_sha256")):
        raise ValueError("decisions belong to different source inputs or schema")
    rows=decisions["records"]
    if not isinstance(rows,dict) or rows.keys()!=template["records"].keys():
        raise ValueError("review coverage must match every target and reservoir source")
    classes=defaultdict(set)
    target_ids={q["qid"] for q in dataset["questions"]}
    scoped_by_qid={}
    for key,original in template["records"].items():
        record=rows[key]
        if (not isinstance(record,dict) or record.keys()!=original.keys()
                or any(record[k]!=original[k] for k in original if k not in DECISIONS)):
            raise ValueError(f"source identity differs for {key}")
        if record["status"]!="reviewed" or not all(_text(record[k]) for k in ("canonical_class","rationale")):
            raise ValueError(f"source still needs adjudication: {key}")
        for field in ("global_aliases","context_only"):
            if not isinstance(record[field],list) or not all(map(_text,record[field])):
                raise ValueError(f"invalid {field} for {key}")
        canonical=record["canonical_class"]
        classes[canonical].update(record["global_aliases"])
        if key in target_ids and record["context_only"]:
            scoped_by_qid[key]=list(record["context_only"])
    review_id=digest(decisions)
    result={"version":"acl-source-class-v1-"+review_id, "reviewer":decisions["reviewer"],
            "classes":{name:sorted(values-{name}) for name,values in classes.items()},
            "source_review":{"decisions_sha256":review_id,"dataset_sha256":template["dataset_sha256"],
                             "reservoir_sha256":template["reservoir_sha256"],"reviewed_sources":len(rows),
                             "context_only_by_qid":scoped_by_qid}}
    oracle=MenuOracle([],aliases=result)
    for rec in rows.values():
        canonical=rec["canonical_class"]
        scoped={_label_key(label) for label in rec["context_only"]}
        equivalents={_label_key(label) for label in {canonical,*classes[canonical]}}
        if scoped & equivalents:
            raise ValueError("a source-local form is also declared globally interchangeable with its source gold")
        required={rec["source_canonical"],*(label for label in rec["source_aliases"]
                                            if _label_key(label) not in scoped)}
        if any(oracle.class_id(label)!=rec["canonical_class"] for label in required):
            raise ValueError("reviewed classes conflict with the frozen accepted labels; revisit the source freeze")
    for condition in ("independent_pool","same_category_pool"):
        spec=ArmSpec.from_dict({"version":1,"k":4,"menu":"baseline","seed":1,
                                "baseline_condition":condition,"reward":{"arm":"baseline"}})
        prepare(spec,dataset,result,reservoir=donors,split_roles=CROSSWALK,contextual_review=True)
    return result


def _inputs():
    for name,expected in FILES.items():
        if sha256_file(PRIVATE/name)!=expected:
            raise ValueError(f"private input differs from pinned member: {name}")
    return tuple(load_json_strict(PRIVATE/name) for name in FILES)


def materialize(archive):
    if sha256_file(archive)!=ARCHIVE_SHA256:
        raise ValueError("unexpected historical archive SHA-256")
    with ZipFile(archive) as z:
        verify_manifest(z)
        for name,expected in FILES.items():
            data=z.read("frozen/"+name)
            if hashlib.sha256(data).hexdigest()!=expected:
                raise ValueError("historical member differs from full configuration")
            _create_or_match(PRIVATE/name,data)
    dataset,donors=_inputs()
    draft=build_draft(dataset,donors)
    readable=json.dumps(draft,ensure_ascii=False,indent=2,allow_nan=False)+"\n"
    _create_or_match(PRIVATE/"review/draft.json",readable.encode("utf-8"))
    return len(draft["records"])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="command",required=True)
    sub.add_parser("materialize").add_argument("--archive",required=True,type=Path)
    sub.add_parser("compile").add_argument("--decisions",required=True,type=Path)
    per_menu=sub.add_parser("menu-draft",help="prepare an optional source-bound menu review")
    per_menu.add_argument("--config",type=Path,default=Path("configs/acl_expansion/full.yaml"))
    per_menu.add_argument("--menu")
    per_menu.add_argument("--seed",type=int)
    per_menu.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    if args.command=="materialize":
        print(f"{materialize(args.archive)} source records in {PRIVATE.as_posix()}/review/draft.json")
    elif args.command=="compile":
        dataset,donors=_inputs()
        result=compile_decisions(dataset,donors,load_json_strict(args.decisions))
        publish_json(PRIVATE/"reviewed_classes.json",result)
        print("Published reviewed registry",result["source_review"]["decisions_sha256"])
    else:
        import yaml
        from qb_data.acl_oracle_review import review_template
        from scripts.acl_expansion import load_inputs

        config=yaml.safe_load(args.config.read_text(encoding="utf-8"))
        prepared=load_inputs(config,args.menu,None,args.seed,None,review_draft=True)
        dataset=load_json_strict(config["inputs"]["dataset"])
        draft=review_template(prepared,dataset)
        readable=json.dumps(draft,ensure_ascii=False,indent=2,allow_nan=False)+"\n"
        _create_or_match(args.out,readable.encode("utf-8"))
        print("Unresolved source-bound menu review records:",len(draft["questions"]))


if __name__=="__main__":
    main()
