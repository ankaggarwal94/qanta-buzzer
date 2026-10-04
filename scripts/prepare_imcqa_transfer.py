"""Freeze transfer inputs, policies, and analysis before any new inference."""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

from scripts import acl_option_scoring as base
from scripts import imcqa_transfer_design as design


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "evaluator", "prior-public", "fitted", "config", "out", "selected-evaluator"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    inputs = {}
    for name, path in (("main_jobs", args.source), ("main_dataset", args.evaluator),
                       ("prior_public", args.prior_public), ("fitted_parameters", args.fitted)):
        raw = path.read_bytes()
        if base.sha(raw) != design.SOURCE_SHA256[name]:
            raise ValueError(f"frozen input checksum differs: {name}")
        inputs[name] = base.load_json(raw)
    config_raw = args.config.read_bytes(); config = base.load_json(config_raw)
    if (config["protocol"] != design.PROTOCOL or config["selection"]["n_questions"] != design.N_QUESTIONS
            or config["selection"]["salt"] != design.SALT or config["contexts_per_model"] != 8000
            or config["frozen_source_sha256"] != design.SOURCE_SHA256):
        raise ValueError("configuration design or source identity differs")
    package = design.build_public_package(inputs["main_jobs"], inputs["main_dataset"],
                                          inputs["prior_public"], inputs["fitted_parameters"])
    encoded = base.canonical(package); compressed = gzip.compress(encoded, mtime=0)
    selected = set(package["selection"]["selected_qids"]["selection"])
    evaluator = {**inputs["main_dataset"], "questions": [q for q in inputs["main_dataset"]["questions"] if q["qid"] in selected],
                 "transfer_source_dataset_sha256": design.SOURCE_SHA256["main_dataset"],
                 "transfer_public_sha256": base.sha(encoded)}
    if args.selected_evaluator.exists():
        raise FileExistsError("selected evaluator is create-once")
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "public.json").write_bytes(encoded)
    (args.out / "public.json.gz").write_bytes(compressed)
    (args.out / "selection_manifest.json").write_bytes(base.canonical(package["selection"]["manifest"]))
    (args.out / "frozen_policy.json").write_bytes(base.canonical(package["frozen_policy"]))
    (args.out / "analysis_plan.json").write_bytes(base.canonical(config["analysis"]))
    args.selected_evaluator.parent.mkdir(parents=True, exist_ok=True)
    evaluator_raw = base.canonical(evaluator)
    args.selected_evaluator.write_bytes(evaluator_raw)
    manifest = {
        "schema_version": "imcqa-transfer-transport-v1", "public_sha256": base.sha(encoded),
        "compressed_sha256": base.sha(compressed), "public_bytes": len(encoded), "compressed_bytes": len(compressed),
        "config_sha256": base.sha(config_raw), "selection_manifest_sha256": package["selection"]["manifest_sha256"],
        "frozen_policy_sha256": base.sha(base.canonical(package["frozen_policy"])),
        "analysis_plan_sha256": base.sha(base.canonical(config["analysis"])),
        "selected_evaluator_sha256": base.sha(evaluator_raw), "source_sha256": dict(design.SOURCE_SHA256),
        "real_questions": len(selected), "contexts_per_model": len(package["jobs"]),
        "real_gold_labels_included": False, "inference_files": ["public.json"],
        "all_contexts_new": True, "new_fits": 0,
    }
    (args.out / "transport_manifest.json").write_bytes(base.canonical(manifest))
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
