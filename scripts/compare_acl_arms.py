"""Matched held-out comparison of immutable ACL expansion runs."""
from __future__ import annotations

import argparse
from pathlib import Path
from statistics import fmean

from qb_data.acl_protocol import digest, publish_json
from scripts.acl_checkpoints import verify_completed_lineage
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
    verify_completed_lineage(folder, manifest["training"], receipt)
    path = folder/f"{split}_episodes.json"
    if sha256_file(path) != receipt["episodes"][split]:
        raise ValueError("episode identity mismatch")
    return manifest, load_json_strict(path)


def compare(reference, other, split):
    """Compare two completed arms for the same learner and seed on a common held-out split.

    Accuracy, abstention and prefix counts have common units. Discounted returns
    describe each arm under its own reward schedule.
    """
    left, lrows = _read_run(reference, split)
    right, rrows = _read_run(other, split)
    left_menu = left["input"]["spec"]["menu"]
    right_menu = right["input"]["spec"]["menu"]
    same_menu = left_menu == right_menu

    def conditions(run):
        inputs = run["input"]
        arm = inputs["spec"]
        fixed_spec = {k: v for k, v in arm.items() if k not in {"menu", "reward"}}
        fixed_reward = {k: v for k, v in arm["reward"].items() if k not in {"arm", "potential_scale"}}
        assets = dict(run["assets"])
        if not same_menu and ("sbert" in left_menu) != ("sbert" in right_menu):
            # Only the SBERT menu consumes retrieval weights in this comparison.
            if "sbert" in arm["menu"]:
                if not assets.pop("retriever", None):
                    raise ValueError("missing retriever identity for matched arms")
        review = inputs.get("source_review")
        if review is not None and not same_menu and (
                review.get("class_review_sha256") or review.get("selection_sha256")):
            # The same question-level decisions can project to distinct ordered menus.
            review = {k: v for k, v in review.items() if k not in {"review_sha256", "menus_sha256"}}
        fixed_inputs = {k: v for k, v in inputs.items()
                        if k not in {"spec", "source_review", "menus_sha256"}}
        return (run["runtime"], run["policy"], run["training"], run["device"], assets,
                fixed_inputs, fixed_spec, fixed_reward, review, inputs["menus_sha256"] if same_menu else None)

    if conditions(left) != conditions(right):
        raise ValueError("source review, menus, fixed payoff, model or runtime differ across matched arms")
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
