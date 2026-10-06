#!/usr/bin/env python3
"""Prepare a fresh historical test from the original hash-pinned source snapshot.

This CPU-only builder needs the complete normalized source, original exclusion
file and curated identity overrides. It refuses substituted source snapshots or
a 2026 reserve cohort. It performs no model inference and reads no test outcomes.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qb_data.jane_paired import build_jobs
from scripts import imcqa_tuned_design as design
from scripts import prepare_acl_expansion as original

INPUT_PINS = {
    "source_jsonl": "29c0add66d5bb837dd374335603fcc37269c604655e93105893f13770df7819c",
    "exclusions": "3f2460f55a978e5f2459f85007579ce743b28b9a711b93f01d46f759ea83f2c1",
    "identity_overrides": "956180ce8ee490c677111eb109861e67a66fedb8118bd30a7c647c3c8707437f",
    "prior_dataset": "d16d8e611965fba3829f01cda936145743b7d46187030e2138caf52068aa9b62",
    "reservoir": "db37b0322ef0846d19567d287dc9e48652a26ec24dfb94ebf544f054ce8926d3",
}
LOCK_INPUTS = ("policy_lock", "sample_size_planning", "selection_receipt")
RULE = ("Reconstruct original alias/0.85-text components including original prior exclusions and curated overrides; "
        "retain 2010-2025 standard academic components unused by the original 5000 or reservoir; "
        "reject exact normalized word-5gram Jaccard >=0.8 against original exclusion texts, original 5000 and reservoir texts; "
        "greedily deduplicate the complete candidate pool in SHA256(salt|qid) order with one question/group; "
        "allocate locked N using original raw-source category/difficulty weights with capacity capping and largest remainders; "
        "take the lowest hash-ranked candidates per stratum. Reuse the original distractor reservoir and menu renderer.")


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")


def _identity(record):
    return record["source"]["_id"], record["group_id"], tuple(record["component_source_ids"]), tuple(record["component_alias_keys"])


def _prior_identity(question):
    source = question["source"]
    return source["_id"], question["group_id"], tuple(source["component_source_ids"]), tuple(source["component_alias_keys"])


def verify_original_partition(reconstructed_main, reconstructed_reservoir, prior_dataset, reservoir):
    """Require exact retained identities when replaying the original selection."""
    if ({_identity(r) for r in reconstructed_main} != {_prior_identity(q) for q in prior_dataset["questions"]}
            or {_identity(r) for r in reconstructed_reservoir} != {_identity(r) for r in reservoir}):
        raise ValueError("reconstructed original main or reservoir identities differ")


def fresh_candidates(representatives, prior_dataset, reservoir, exclusions):
    """Build a reproducible outcome-blind pool with complete prior exclusion.

    Parameters
    ----------
    representatives : list of dict
        Original grouped source records, before main/reservoir selection.
    prior_dataset, exclusions : dict
        Retained original gold dataset and original preproject exclusions.
    reservoir : list of dict
        Original distractor records; these components cannot become fresh gold.

    Returns
    -------
    tuple
        Deduplicated historical records and the complete selection audit.
    """
    prior_questions = prior_dataset["questions"]
    groups = {q["group_id"] for q in prior_questions} | {r["group_id"] for r in reservoir}
    source_ids = {str(x).removeprefix("qbreader:") for x in exclusions.get("source_ids", [])}
    aliases = {original.entity_key(x) for x in exclusions.get("aliases", [])}
    for q in prior_questions:
        source_ids.update(q["source"]["component_source_ids"])
        aliases.update(q["source"]["component_alias_keys"])
    for r in reservoir:
        source_ids.update(r["component_source_ids"])
        aliases.update(r["component_alias_keys"])
    eligible, rejected = [], []
    for record in representatives:
        qid = "qbreader:" + record["source"]["_id"]
        reason = None
        if not 2010 <= record["source"]["year"] <= 2025:
            reason = "outside_historical_year_frame"
        elif record["group_id"] in groups:
            reason = "prior_or_reservoir_group"
        elif source_ids.intersection(record["component_source_ids"]):
            reason = "prior_component_source_id"
        elif aliases.intersection(record["component_alias_keys"]):
            reason = "prior_component_alias"
        if reason:
            rejected.append({"qid": qid, "reason": reason})
        else:
            eligible.append(record)
    eligible.sort(key=lambda r: (design.rank("qbreader:" + r["source"]["_id"]), r["source"]["_id"]))
    references = [(q["qid"], q["question"]) for q in prior_questions]
    references += [("reservoir:" + r["source"]["_id"], r["source"]["question_sanitized"]) for r in reservoir]
    references += [(f"original_exclusion:{i}", t) for i, t in enumerate(exclusions.get("questions", [])) if isinstance(t, str) and t.strip()]
    values = [design.shingles(r["source"]["question_sanitized"]) for r in eligible]
    values.extend(design.shingles(t) for _, t in references)
    neighbors = [set() for _ in eligible]
    prior_collisions = {}
    for a, b, _ in original.near_duplicate_edges(values, .8):
        if not design.near_duplicate(values[a], values[b]):
            continue
        if a > b:
            a, b = b, a
        if a >= len(eligible):
            continue
        if b >= len(eligible):
            ref = references[b - len(eligible)][0]
            prior_collisions[a] = min(prior_collisions.get(a, ref), ref)
        else:
            neighbors[a].add(b)
            neighbors[b].add(a)
    accepted, accepted_indices, accepted_groups = [], set(), set()
    for i, record in enumerate(eligible):
        qid = "qbreader:" + record["source"]["_id"]
        collisions = neighbors[i] & accepted_indices
        if i in prior_collisions:
            rejected.append({"qid": qid, "reason": "prior_text_jaccard_ge_0.8", "reference_qid": prior_collisions[i]})
        elif record["group_id"] in accepted_groups:
            rejected.append({"qid": qid, "reason": "accepted_group"})
        elif collisions:
            other = eligible[min(collisions)]
            rejected.append({"qid": qid, "reason": "fresh_text_jaccard_ge_0.8", "reference_qid": "qbreader:" + other["source"]["_id"]})
        else:
            accepted.append(record)
            accepted_indices.add(i)
            accepted_groups.add(record["group_id"])
    return accepted, {"rule": RULE, "salt": design.SALT, "outcomes_used_for_selection": False,
                      "original_component_count": len(representatives), "prior_dataset_questions": len(prior_questions),
                      "reservoir_components": len(reservoir), "prior_reference_texts": len(references),
                      "historical_before_text_screen": len(eligible), "historical_after_text_screen": len(accepted),
                      "excluded_prior_and_reservoir_groups": sorted(groups), "rejected": rejected,
                      "rejection_counts": dict(Counter(r["reason"] for r in rejected))}


def select_fresh(candidates, n, weights):
    """Apply raw-source category/difficulty targets after fresh-pool screening."""
    capacities = dict(Counter(original.stratum(r) for r in candidates))
    quotas = original.allocate_bounded(n, weights, capacities)
    pools = {}
    for record in candidates:
        pools.setdefault(original.stratum(record), []).append(record)
    selected = []
    for key in sorted(pools):
        ordered = sorted(pools[key], key=lambda r: (design.rank("qbreader:" + r["source"]["_id"]), r["source"]["_id"]))
        selected.extend(ordered[:quotas.get(key, 0)])
    if len(selected) != n:
        raise ValueError("fresh cohort count differs")
    for record in selected:
        record["split"] = "test"
    audit = [{"category": key[0], "difficulty": key[1], "source_weight": weights.get(key, 0),
              "fresh_capacity": capacities.get(key, 0), "selected": quotas.get(key, 0)} for key in sorted(set(weights) | set(capacities))]
    return selected, audit


def prepare(paths, out_dir):
    """Create a complete, hash-bound package only from the original source bytes."""
    if out_dir.exists():
        raise ValueError("output directory already exists")
    hashes = {}
    for name, path in paths.items():
        hashes[name] = original.file_sha(path)
        if name in INPUT_PINS and hashes[name] != INPUT_PINS[name]:
            raise ValueError("hash-pinned input differs: " + name)
    policy, planning = read_json(paths["policy_lock"]), read_json(paths["sample_size_planning"])
    receipt = read_json(paths["selection_receipt"])
    if (receipt.get("status") != "complete" or receipt.get("validation_passed") is not True
            or receipt.get("outputs_sha256", {}).get("frozen_policies.json") != hashes["policy_lock"]
            or receipt.get("outputs_sha256", {}).get("sample_size_planning.json") != hashes["sample_size_planning"]):
        raise ValueError("policy and sample-size planning do not share a validated selection receipt")
    if (policy.get("schema_version") != "imcqa-independently-tuned-policies-v1"
            or policy.get("status") != "frozen_for_fresh_evaluation" or policy.get("new_calibration_fits") != 0):
        raise ValueError("fresh evaluation requires the frozen independently selected policy artifact")
    n = planning.get("planned_questions")
    if type(n) is not int or not 4 <= n <= 5000:
        raise ValueError("sample size is not locked within supported bounds")
    exclusions, overrides = read_json(paths["exclusions"]), read_json(paths["identity_overrides"])
    prior, reservoir = read_json(paths["prior_dataset"]), read_json(paths["reservoir"])
    config = original.Config()
    records, loading = original.load_candidates(paths["source_jsonl"], exclusions, config)
    representatives, grouping = original.group_candidates(records, exclusions, config, overrides)
    weights = {(k.rsplit("|", 1)[0], int(k.rsplit("|", 1)[1])): v for k, v in loading["source_population_weights"].items()}
    old_main, old_reservoir, _, old_selection = original.select_records(representatives, config, weights)
    verify_original_partition(old_main, old_reservoir, prior, reservoir)
    candidates, audit = fresh_candidates(representatives, prior, reservoir, exclusions)
    selected, allocation = select_fresh(candidates, n, weights)
    audit.update(n_questions=n, stratum_allocation=allocation, input_sha256=hashes,
                 selected_qids=["qbreader:" + r["source"]["_id"] for r in selected],
                 original_selection_replayed=True, original_unused_historical_components=old_selection["unused_historical_components"],
                 original_loading=loading, original_grouping_summary={k: v for k, v in grouping.items() if k != "near_duplicate_edges"})
    dataset = original.build_dataset(selected, reservoir, config, {"input_sha256": hashes, "rule": RULE,
                "preparer_sha256": original.file_sha(Path(__file__)), "original_preparer_sha256": original.file_sha(Path(original.__file__))}, required_splits={"test"})
    dataset["source"]["origin"] = "frozen QBReader 2026-07-18 snapshot; unused historical IMCQA test components"
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / "evaluator").mkdir()
    (out_dir / "public").mkdir()
    dataset_path = out_dir / "evaluator/main_dataset.json"
    write_json(dataset_path, dataset)
    write_json(out_dir / "evaluator/selection_manifest.json", audit)
    source_jobs = build_jobs(dataset, required_splits={"test"})
    source_public = {"schema_version": "jane-public-jobs-v1", "evidence_scope": "scientific", "jobs": source_jobs}
    write_json(out_dir / "public/source_jobs.json", source_public)
    selection = {"salt": design.SALT, "manifest_sha256": original.file_sha(out_dir / "evaluator/selection_manifest.json"),
                 "selected_qids": audit["selected_qids"], "outcomes_used_for_selection": False,
                 "selected_group_ids": {q["qid"]: q["group_id"] for q in dataset["questions"]},
                 "qid_category": {q["qid"]: q["source"]["category"] for q in dataset["questions"]},
                 "selected_normalized_text_sha256": {q["qid"]: design.base.sha(design.base.canonical(design.normalized_tokens(q["question"]))) for q in dataset["questions"]}}
    public = design.build_public_package(source_jobs, selection,
                source_input_sha256=original.file_sha(out_dir / "public/source_jobs.json"),
                main_dataset_sha256=original.file_sha(dataset_path), policy_lock_sha256=hashes["policy_lock"],
                sample_size_planning_sha256=hashes["sample_size_planning"])
    write_json(out_dir / "public/pilot.json", public)
    for name in LOCK_INPUTS:
        shutil.copyfile(paths[name], out_dir / (name + ".json"))
    summary = {"status": "prepared_not_scored", "protocol": design.PROTOCOL, "n_questions": n,
               "n_jobs": len(public["jobs"]), "new_model_forward_passes": 0,
               "selection_id": public["selection_id"], "counts": original.counts(selected),
               "inference_public_file": "public/pilot.json", "evaluator_file": "evaluator/main_dataset.json",
               "population": "2010-2025 standard academic eligible components, raw-source category/difficulty targets with capacity-driven deviations",
               "original_semantic_setup_assumption": "retained as authorized; no new human validity certification"}
    write_json(out_dir / "preparation_receipt.json", summary)
    files = {str(p.relative_to(out_dir)): {"sha256": original.file_sha(p), "bytes": p.stat().st_size}
             for p in sorted(out_dir.rglob("*")) if p.is_file()}
    write_json(out_dir / "manifest.json", {"schema_version": "imcqa-tuned-input-manifest-v1", "files": files})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (*INPUT_PINS, *LOCK_INPUTS):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare({name: getattr(args, name) for name in (*INPUT_PINS, *LOCK_INPUTS)}, args.out_dir), sort_keys=True))


if __name__ == "__main__":
    main()
