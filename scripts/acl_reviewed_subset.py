"""Materialize the privately held, source-reviewed seven-question engineering subset."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from qb_data.acl_menus import MenuOracle
from qb_data.acl_oracle_review import review_template, verify_review
from qb_data.acl_protocol import ArmSpec, digest, prepare, publish_json
from qb_data.acl_scientific_review import compile_review, extend_template
from scripts.stopdff_v5.fileio import create_once_bytes
from scripts.stopdff_v5.identity import sha256_file


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/acl_expansion/real_inputs"
SELECTION = OUTPUT / "selection.json"
REGISTRY = OUTPUT / "registry.json"
ENVELOPE = ROOT / "configs/acl_expansion/reviewed_geography.siv"
ARCHIVE_DIGEST = "cd2f72b9a1fd774ccd1da30aefca418b60c5c8c35ffaf88b625773dd13f82cc3"
ARCHIVE_KEY_DOMAIN = b"qanta-buzzer/acl-reviewed-subset/archive-key/v1\0"
REVIEW_DOMAIN = b"qanta-buzzer/acl-reviewed-subset/selection-and-registry/v1"
DATASET = "frozen/evaluator/main_dataset.json"
RESERVOIR = "frozen/evaluator/reservoir.json"
SELECTION_DIGEST = "3f2cedf4ba5c368550a532e716bb3442e5415ac80e9f7c10cc61b13952b4c576"


def require_selection(selection):
    if digest(selection) != SELECTION_DIGEST:
        raise ValueError("selection differs from the independently reviewed seven-question scope")


def recover_review(archive: Path, envelope: Path = ENVELOPE):
    """Recover the original reviewer files using the exact private source archive."""
    archive_hash = hashlib.sha256()
    key_hash = hashlib.sha512(ARCHIVE_KEY_DOMAIN)
    with archive.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            archive_hash.update(chunk)
            key_hash.update(chunk)
    if archive_hash.hexdigest() != ARCHIVE_DIGEST:
        raise ValueError("private source archive digest mismatch")

    from cryptography.exceptions import InvalidTag
    from cryptography.hazmat.primitives.ciphers.aead import AESSIV

    try:
        raw = AESSIV(key_hash.digest()).decrypt(envelope.read_bytes(), [REVIEW_DOMAIN])
    except InvalidTag as exc:
        raise ValueError("private review envelope authentication failed") from exc
    selection_bytes, registry_bytes = (s.encode("utf-8") for s in json.loads(raw))
    selection = json.loads(selection_bytes)
    require_selection(selection)
    if digest(json.loads(registry_bytes)) != selection["oracle_sha256"]:
        raise ValueError("reviewed answer registry digest mismatch")
    return selection_bytes, registry_bytes


def _publish_review(path: Path, data: bytes):
    try:
        create_once_bytes(path, data)
    except FileExistsError:
        if path.is_symlink() or path.read_bytes() != data:
            raise ValueError(f"existing private review differs: {path.name}") from None


def _source_rows(records, wanted, *, donor):
    key = "id" if donor else "qid"
    by_id = {entry[key]: entry for entry in wanted}
    selected = []
    for row in records:
        source = row["source"]
        identifier = source["_id"] if donor else row["qid"]
        entry = by_id.get(identifier)
        if entry is None:
            continue
        members = row["component_source_ids"] if donor else source["component_source_ids"]
        label = row["canonical_answer"] if donor else source["canonical_answer"]
        if (digest(row) != entry["source_row_sha256"] or row["group_id"] != entry["group_id"]
                or members != entry["component_source_ids"] or
                label != entry["display_class" if donor else "accepted_class"]):
            raise ValueError("reviewed source metadata, members, label or row digest changed")
        selected.append(row)
    ids = [r["source"]["_id"] if donor else r["qid"] for r in selected]
    if len(by_id) != len(wanted) or ids != [entry[key] for entry in wanted]:
        raise ValueError("source order or coverage differs from the reviewed selection")
    return selected


def materialize(archive: Path, selection: dict, registry: dict):
    require_selection(selection)
    if sha256_file(archive) != selection["archive_sha256"]:
        raise ValueError("private source archive digest mismatch")
    if digest(registry) != selection["oracle_sha256"]:
        raise ValueError("reviewed answer registry digest mismatch")
    with ZipFile(archive) as bundle:
        contents = {}
        for name in (DATASET, RESERVOIR):
            if bundle.namelist().count(name) != 1:
                raise ValueError("archive member missing or repeated")
            raw = bundle.read(name)
            if hashlib.sha256(raw).hexdigest() != selection["members"][name]:
                raise ValueError("private archive member digest mismatch")
            contents[name] = json.loads(raw)
    original = contents[DATASET]
    questions = _source_rows(original["questions"], selection["questions"], donor=False)
    donors = _source_rows(contents[RESERVOIR], selection["donors"], donor=True)
    # A reference source must be outside all original target splits, not just the seven rows.
    groups = {q["group_id"] for q in original["questions"]}
    members = {i for q in original["questions"] for i in (q["source"]["id"],
                                                             *q["source"]["component_source_ids"])}
    selected_groups = {q["group_id"]: q["split"] for q in questions}
    selected_members = {i: q["split"] for q in questions for i in
                        [q["source"]["id"], *q["source"]["component_source_ids"]]}
    for q in original["questions"]:
        s, split = q["source"], q["split"]
        if (selected_groups.get(q["group_id"], split) != split or
                any(selected_members.get(i, split) != split for i in
                    [s["id"], *s["component_source_ids"]])):
            raise ValueError("reviewed question component crosses an original source split")
    for r in donors:
        if (r["group_id"] in groups or
                members.intersection([r["source"]["_id"], *r["component_source_ids"]])):
            raise ValueError("reviewed donor overlaps original target groups or component members")
    frozen_sources = {i for q in questions for m in q["menus"] if m["condition"] == "independent_pool"
                      for i in m["provenance"]["candidate_source_ids"]}
    if frozen_sources != {r["source"]["_id"] for r in donors}:
        raise ValueError("independent baseline differs from reviewed reference pool")
    classes = {r["canonical_answer"] for r in donors}
    golds = {q["source"]["canonical_answer"] for q in questions}
    if classes & golds or classes | golds != set(registry["classes"]):
        raise ValueError("reviewed class universe differs")
    dataset = {**original, "evidence_scope": "engineering_smoke", "questions": questions,
               "subset_version": selection["version"]}
    return dataset, donors


def bind_review(prepared, dataset: dict, selection: dict, registry: dict) -> dict:
    """Bind already recorded full-question judgments to the actual displayed options."""
    require_selection(selection)
    if (digest(dataset) != selection["materialized_dataset_sha256"] or
            prepared.identity["reservoir_sha256"] != selection["materialized_reservoir_sha256"] or
            digest(registry) != selection["oracle_sha256"] or
            prepared.identity["oracle_sha256"] != selection["oracle_sha256"]):
        raise ValueError("source review differs from pinned data, donor pool or registry")
    reviewed = {q["qid"]: q for q in selection["questions"]}
    roles = {"calibration": "train", "selection": "val", "test": "test"}
    if prepared.identity["split_roles"] != roles:
        raise ValueError("reviewed training, validation or test split mapping changed")
    donor_reasons = {r["display_class"]: r["referent"] for r in selection["donors"]}
    lookup = MenuOracle([], aliases=registry)
    review = review_template(prepared, dataset)
    for qid, q in review["questions"].items():
        source = reviewed[qid]
        if q["split"] != roles[source["split"]] or prepared.groups[qid] != source["group_id"]:
            raise ValueError("source review group or runtime split differs")
        if q["source_sha256"] != source["source_row_sha256"]:
            raise ValueError("source review refers to a different question")
        q.update(status="reviewed", reviewer=selection["reviewer"])
        for option in q["options"]:
            label = lookup.class_id(option["label"])
            if label == source["accepted_class"]:
                option.update(judgment="accepted", rationale=source["referent"])
            elif label in donor_reasons:
                option.update(judgment="rejected", rationale=source["referent"] + " Donor: " + donor_reasons[label])
            else:
                raise ValueError("displayed class outside recorded question-specific review")
    return {**verify_review(prepared, dataset, review), "selection_sha256": digest(selection),
            "reviewer": selection["reviewer"]}


def class_review(dataset, donors, selection, registry):
    """Project the independently reviewed target and donor referents onto the scientific schema."""
    spec = ArmSpec.from_dict({"version": 1, "menu": "baseline", "k": 4, "seed": 1,
                              "reward": {"arm": "baseline"}})
    prepared = prepare(spec, dataset, registry, reservoir=donors,
                       split_roles={"calibration": "train", "selection": "val", "test": "test"})
    baseline = bind_review(prepared, dataset, selection, registry)
    decisions = extend_template(prepared, dataset, registry)
    for target in selection["questions"]:
        classes = decisions["questions"][target["qid"]]["classes"]
        classes[target["accepted_class"]] = {
            "judgment": "accepted", "reviewer": selection["reviewer"], "rationale": target["referent"]}
        for donor in selection["donors"]:
            classes[donor["display_class"]] = {
                "judgment": "rejected", "reviewer": selection["reviewer"],
                "rationale": target["referent"] + " Donor: " + donor["referent"]}
    _, projected = compile_review(prepared, dataset, registry, decisions)
    if projected["review_sha256"] != baseline["review_sha256"]:
        raise ValueError("scientific projection disagrees with the reviewed baseline")
    return decisions


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args(argv)
    selection_bytes, registry_bytes = recover_review(args.archive)
    selection, registry = json.loads(selection_bytes), json.loads(registry_bytes)
    dataset, donors = materialize(args.archive, selection, registry)
    if (digest(dataset) != selection["materialized_dataset_sha256"] or
            digest(donors) != selection["materialized_reservoir_sha256"]):
        raise ValueError("derived data differ from reviewed version")
    decisions = class_review(dataset, donors, selection, registry)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=OUTPUT.parent, prefix=".review-") as temp:
        stage = Path(temp)/"complete"
        publish_json(stage/"dataset.json", dataset)
        publish_json(stage/"reservoir.json", donors)
        publish_json(stage/"class_review.json", decisions)
        _publish_review(stage/SELECTION.name, selection_bytes)
        _publish_review(stage/REGISTRY.name, registry_bytes)
        receipt = {"scope": dataset["evidence_scope"], "archive_sha256": selection["archive_sha256"],
               "members": selection["members"], "selection_sha256": digest(selection),
               "oracle_sha256": selection["oracle_sha256"],
               "dataset_file_sha256": sha256_file(stage/"dataset.json"),
               "reservoir_file_sha256": sha256_file(stage/"reservoir.json"),
               "splits": dict(Counter(q["split"] for q in dataset["questions"])),
               "donor_classes": len(donors),
               "donor_categories": dict(Counter(r["source"]["category"] for r in donors))}
        publish_json(stage/"receipt.json", receipt)
        from filelock import FileLock
        with FileLock(str(OUTPUT.parent/".materialize-review.lock")):
            if OUTPUT.exists():
                files = list(stage.iterdir())
                if any((OUTPUT/f.name).exists() and (OUTPUT/f.name).read_bytes() != f.read_bytes()
                       for f in files):
                    raise ValueError("existing private materialization differs from reviewed bytes")
                for f in sorted(files, key=lambda path: path.name == "receipt.json"):
                    _publish_review(OUTPUT/f.name, f.read_bytes())
            else:
                stage.rename(OUTPUT)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
