"""Inspect immutable smoke runs and retain a small portable evidence ledger."""
from __future__ import annotations

import argparse
from pathlib import Path

from qb_data.acl_protocol import digest, publish_json
from scripts.acl_expansion import REWARDS, STRATEGIES
from scripts.compare_acl_arms import compare
from scripts.stopdff_v5.identity import load_json_strict, sha256_file


def checked_review(inputs, scope):
    """Bind the recorded engineering scope to the actual source and realized menu."""
    if scope == "synthetic held-out engineering smoke":
        from scripts.acl_smoke_data import inputs as fixture
        if inputs["dataset_sha256"] != digest(fixture()):
            raise ValueError("synthetic scope requires the smoke fixture")
        return {}
    from scripts.acl_reviewed_subset import SELECTION, require_selection
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
    return {"source_review": review}


def summarize(receipts, *, scope="synthetic held-out engineering smoke"):
    output = {"scope": scope, "learners": {}}
    cases = ({(m, "baseline", 0.) for m in STRATEGIES}
             | {("baseline", r, 0.) for r in REWARDS} | {("baseline", "baseline", .5)})
    for receipt in receipts:
        source = load_json_strict(receipt)
        if source["command"] != "smoke" or len(source["runs"]) != len(cases):
            raise ValueError("incomplete smoke receipt")
        policy = source["policy"]
        runs = source["runs"]
        if {(r["menu"], r["reward"], r["potential_scale"]) for r in runs} != cases:
            raise ValueError("smoke receipt lacks required conditions")
        reference = next(r for r in runs if (r["menu"], r["reward"], r["potential_scale"])
                         == ("baseline", "baseline", 0.))
        reference_path = receipt.parent/reference["run_id"]
        ref_manifest = load_json_strict(reference_path/"manifest.json")
        result = {"runtime": ref_manifest["runtime"], "lean": ref_manifest["input"]["core"],
                  "training": ref_manifest["training"], "dataset": ref_manifest["input"]["dataset_sha256"],
                  "oracle": ref_manifest["input"]["oracle_sha256"], "source_receipt": sha256_file(receipt),
                  "runs": []}
        for row in runs:
            folder = receipt.parent/row["run_id"]
            completed = load_json_strict(folder/"completed.json")
            if sha256_file(folder/"completed.json") != row["completion_sha256"]:
                raise ValueError("completion digest differs")
            manifest = load_json_strict(folder/"manifest.json")
            spec = manifest["input"]["spec"]
            if (manifest["policy"], spec["menu"], spec["reward"]["arm"], spec["reward"]["potential_scale"]) != (
                    policy, row["menu"], row["reward"], row["potential_scale"]):
                raise ValueError("condition differs from realized manifest")
            checkpoints = list(folder.glob("train-*/files.json"))
            if len(checkpoints) != 1:
                raise ValueError("expected a single bounded smoke update")
            ckpt = load_json_strict(checkpoints[0])
            if digest(ckpt) != completed["checkpoint_sha256"] or ckpt["run_id"] != folder.name:
                raise ValueError("checkpoint identity differs")
            for name, expected in ckpt["files"].items():
                if sha256_file(checkpoints[0].parent/name) != expected:
                    raise ValueError("checkpoint payload differs")
            steps = load_json_strict(checkpoints[0].parent/"train_steps.json")
            if not steps or any("curve_certificate" not in s["transition"] for s in steps):
                raise ValueError("missing production curve certificates")
            for split in ("val", "test"):
                compare(reference_path, folder, split)
            result["runs"].append({**row, **checked_review(manifest["input"], scope),
                                   "training_steps": len(steps), "episodes": completed["episodes"]})
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
