"""Inspect immutable smoke runs and retain a small portable evidence ledger."""
from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
from types import SimpleNamespace

from qb_data.acl_protocol import digest, publish_json
from scripts.acl_expansion import REWARDS, STRATEGIES
from scripts.compare_acl_arms import compare
from scripts.stopdff_v5.identity import load_json_strict, sha256_file


def checked_review(inputs, scope, menus=None):
    """Bind the recorded engineering scope to the actual source and realized menu."""
    if scope == "synthetic held-out engineering smoke":
        from scripts.acl_smoke_data import inputs as fixture
        if inputs["dataset_sha256"] != digest(fixture()):
            raise ValueError("synthetic scope requires the smoke fixture")
        return {}
    from scripts.acl_reviewed_subset import OUTPUT, REGISTRY, SELECTION, class_review, require_selection
    from qb_data.acl_scientific_review import compile_review
    if scope != "reviewed ACL subset engineering smoke":
        raise ValueError("unknown reviewed smoke scope")
    if not SELECTION.is_file():
        raise ValueError("reviewed scope requires the materialized private selection")
    selection = load_json_strict(SELECTION)
    require_selection(selection)
    review = inputs.get("source_review", {})
    expected = {"dataset_sha256": selection["materialized_dataset_sha256"],
                "menus_sha256": inputs["menus_sha256"], "evidence_scope": "engineering_smoke",
                "reviewed_questions": len(selection["questions"])}
    if (scope != "reviewed ACL subset engineering smoke" or
            inputs["dataset_sha256"] != expected["dataset_sha256"] or
            inputs.get("oracle_sha256") != selection["oracle_sha256"] or
            inputs.get("reservoir_sha256") != selection["materialized_reservoir_sha256"] or
            any(review.get(k) != v for k, v in expected.items()) or
            not review.get("class_review_sha256") or not review.get("review_sha256")):
        raise ValueError("reviewed scope requires the source-bound ACL subset and realized menu")
    dataset = load_json_strict(OUTPUT/"dataset.json")
    donors = load_json_strict(OUTPUT/"reservoir.json")
    registry = load_json_strict(REGISTRY)
    decisions = class_review(dataset, donors, selection, registry)
    if digest(decisions) != review["class_review_sha256"] or menus is None or digest(menus) != inputs["menus_sha256"]:
        raise ValueError("reviewed class or realized menu differs")
    questions = defaultdict(list)
    groups = {}
    for row in dataset["questions"]:
        qid = row["qid"]
        menu = menus[qid]
        questions[inputs["split_roles"][row["split"]]].append(SimpleNamespace(
            qid=qid, question=row["question"], options=list(menu["options"]), gold_index=menu["gold_index"]))
        groups[qid] = row["group_id"]
    prepared = SimpleNamespace(identity=inputs, questions=questions, menus=menus, groups=groups,
                               registry=registry)
    _, authentic = compile_review(prepared, dataset, registry, decisions)
    if review != authentic:
        raise ValueError("reviewed menu adjudication digest or fields differ")
    return {"source_review": authentic}


def _run_path(parent, run_id):
    if not isinstance(run_id, str) or re.fullmatch(r"[0-9a-f]{64}", run_id) is None:
        raise ValueError("expected a content-addressed run ID")
    return _member(parent, run_id)


def _member(parent, name):
    if (not isinstance(name, str) or not name or "\\" in name or PureWindowsPath(name).drive
            or PurePosixPath(name).is_absolute() or any(p in ("", ".", "..") for p in name.split("/"))):
        raise ValueError("invalid receipt member path")
    path = (parent/name).resolve()
    if not path.is_relative_to(parent.resolve()):
        raise ValueError("receipt member escapes its directory")
    return path


def summarize(receipts, *, scope="synthetic held-out engineering smoke"):
    output = {"scope": scope, "learners": {}}
    shared = {}
    cases = ({(m, "baseline", 0.) for m in STRATEGIES}
             | {("baseline", r, 0.) for r in REWARDS} | {("baseline", "baseline", .5)})
    for receipt in receipts:
        source = load_json_strict(receipt)
        if source["command"] != "smoke" or len(source["runs"]) != len(cases):
            raise ValueError("incomplete smoke receipt")
        policy = source["policy"]
        if policy not in ("ppo", "t5") or policy in output["learners"]:
            raise ValueError("expected one receipt per learner")
        runs = source["runs"]
        if {(r["menu"], r["reward"], r["potential_scale"]) for r in runs} != cases:
            raise ValueError("smoke receipt lacks required conditions")
        reference = next(r for r in runs if (r["menu"], r["reward"], r["potential_scale"])
                         == ("baseline", "baseline", 0.))
        reference_path = _run_path(receipt.parent, reference["run_id"])
        ref_manifest = load_json_strict(_member(reference_path, "manifest.json"))
        result = {"runtime_sha256": digest(ref_manifest["runtime"]), "core_sha256": digest(ref_manifest["input"]["core"]),
                  "training_sha256": digest(ref_manifest["training"]), "dataset": ref_manifest["input"]["dataset_sha256"],
                  "oracle": ref_manifest["input"]["oracle_sha256"], "source_receipt": sha256_file(receipt),
                  "runs": []}
        for row in runs:
            folder = _run_path(receipt.parent, row["run_id"])
            for name in ("completed.json", "manifest.json", "menus.json", "val_episodes.json", "test_episodes.json"):
                _member(folder, name)
            completed = load_json_strict(folder/"completed.json")
            if sha256_file(folder/"completed.json") != row["completion_sha256"]:
                raise ValueError("completion digest differs")
            manifest = load_json_strict(folder/"manifest.json")
            inputs = manifest["input"]
            if any(not isinstance(inputs[k], str) or not re.fullmatch(r"[0-9a-f]{64}", inputs[k])
                   for k in ("dataset_sha256", "oracle_sha256", "menus_sha256")):
                raise ValueError("invalid input digest")
            condition = (row["menu"], row["reward"], row["potential_scale"])
            if shared.setdefault(condition, inputs) != inputs:
                raise ValueError("paired learners differ in source, arms, review or realized menus")
            spec = manifest["input"]["spec"]
            if (manifest["policy"], spec["menu"], spec["reward"]["arm"], spec["reward"]["potential_scale"]) != (
                    policy, row["menu"], row["reward"], row["potential_scale"]):
                raise ValueError("condition differs from realized manifest")
            checkpoints = list(folder.glob("train-*/files.json"))
            if len(checkpoints) != 1:
                raise ValueError("expected a single bounded smoke update")
            checkpoint_file = _member(folder, checkpoints[0].relative_to(folder).as_posix())
            ckpt = load_json_strict(checkpoint_file)
            if digest(ckpt) != completed["checkpoint_sha256"] or ckpt["run_id"] != folder.name:
                raise ValueError("checkpoint identity differs")
            for name, expected in ckpt["files"].items():
                if sha256_file(_member(checkpoint_file.parent, name)) != expected:
                    raise ValueError("checkpoint payload differs")
            steps = load_json_strict(_member(checkpoint_file.parent, "train_steps.json"))
            if not steps or any("curve_certificate" not in s["transition"] for s in steps):
                raise ValueError("missing production curve certificates")
            for split in ("val", "test"):
                compare(reference_path, folder, split)
            elapsed = row["elapsed_s"]
            if type(elapsed) not in (int,float) or not 0 <= elapsed < math.inf:
                raise ValueError("invalid elapsed time")
            summary = {key: row[key] for key in ("menu","reward","potential_scale","run_id","completion_sha256","elapsed_s")}
            review = checked_review(inputs, scope, load_json_strict(folder/"menus.json"))
            result["runs"].append({**summary, **review, "training_steps": len(steps),
                                   "episodes": {s: sha256_file(folder/f"{s}_episodes.json") for s in ("val","test")}})
        output["learners"][policy] = result
    if set(output["learners"]) != {"ppo", "t5"}:
        raise ValueError("need PPO and T5 receipts")
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("receipts", type=Path, nargs=2)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--scope", choices=("synthetic held-out engineering smoke",
                                      "reviewed ACL subset engineering smoke"),
                   default="synthetic held-out engineering smoke")
    args = p.parse_args()
    publish_json(args.out, summarize(args.receipts, scope=args.scope))


if __name__ == "__main__":
    main()
