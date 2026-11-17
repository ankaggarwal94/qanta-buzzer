"""Matched held-out comparison of immutable ACL expansion runs."""
from __future__ import annotations

import argparse
from pathlib import Path
from statistics import fmean

from qb_data.acl_protocol import digest, publish_json
from scripts.stopdff_v5.identity import load_json_strict, sha256_file


def _indexed(episodes):
    index = {e["qid"]: e for e in episodes}
    if len(index) != len(episodes):
        raise ValueError("duplicate qid in paired comparison")
    return index


def _stats(episodes, gamma):
    def discounted(e, key):
        return sum(gamma**i * step["transition"][key] for i, step in enumerate(e["steps"]))
    return {"accuracy": fmean(e["correct"] for e in episodes),
            "abstention_rate": fmean(e["abstained"] for e in episodes),
            "mean_prefixes": fmean(len(e["steps"]) for e in episodes),
            "mean_discounted_base": fmean(discounted(e, "base_reward") for e in episodes),
            "mean_discounted_shaped": fmean(sum(gamma**i * s["reward"] for i, s in enumerate(e["steps"]))
                                              for e in episodes)}


def compare_episodes(reference, other, *, gamma):
    left, right = _indexed(reference), _indexed(other)
    if (not left or left.keys() != right.keys() or
            any(left[q]["group_id"] != right[q]["group_id"] for q in left)):
        raise ValueError("held-out qids and source groups must match")
    return {"matched_questions": len(left), "matched_groups": len({q["group_id"] for q in reference}),
            "reference": _stats(reference, gamma), "arm": _stats(other, gamma),
            "accuracy_delta": fmean(int(right[q]["correct"])-int(left[q]["correct"]) for q in left),
            "correct_only_in_arm": sum(right[q]["correct"] and not left[q]["correct"] for q in left),
            "correct_only_in_reference": sum(left[q]["correct"] and not right[q]["correct"] for q in left)}


def _read_run(folder, split):
    manifest = load_json_strict(folder/"manifest.json")
    receipt = load_json_strict(folder/"completed.json")
    if digest(manifest) != folder.name or receipt["run_id"] != folder.name:
        raise ValueError("run identity mismatch")
    path = folder/f"{split}_episodes.json"
    if sha256_file(path) != receipt["episodes"][split]:
        raise ValueError("episode identity mismatch")
    return manifest, load_json_strict(path)


def compare(reference, other, split):
    left, lrows = _read_run(reference, split)
    right, rrows = _read_run(other, split)
    def conditions(run):
        inputs = run["input"]
        arm = inputs["spec"]
        review = inputs.get("source_review", {})
        fixed_spec = {key: value for key, value in arm.items() if key not in {"menu", "reward"}}
        return (run["runtime"], run["policy"], run["training"], run["device"],
                run["assets"].get("encoder"), inputs["dataset_sha256"], inputs["oracle_sha256"],
                inputs["reservoir_sha256"], inputs["split_roles"], inputs["core"],
                fixed_spec, arm["reward"]["gamma"], review.get("class_review_sha256"),
                review.get("selection_sha256"))
    if conditions(left) != conditions(right):
        raise ValueError("source review, dataset, training, fixed payoff settings or runtime differ across matched arms")
    return {"reference_run": reference.name, "arm_run": other.name, "split": split,
            **compare_episodes(lrows, rrows, gamma=left["input"]["spec"]["reward"]["gamma"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--arm", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    args = parser.parse_args()
    publish_json(args.out, compare(args.reference, args.arm, args.split))


if __name__ == "__main__":
    main()
