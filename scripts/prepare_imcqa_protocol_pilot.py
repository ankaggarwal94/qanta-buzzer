"""Prepare the frozen protocol pilot without transferring real answer labels."""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path

from scripts.acl_option_scoring import canonical, file_hash, load_json, sha
from scripts import imcqa_protocol_design as design


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prior-public", "prior-scores-root", "config", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    prior_raw, config_raw = args.prior_public.read_bytes(), args.config.read_bytes()
    config = load_json(config_raw)
    expected_counts = {"contexts_per_model": 5232, "new_contexts_per_model": 4032,
                       "reused_contexts_per_model": 1200}
    if config.get("protocol") != design.PROTOCOL or any(config.get("execution", {}).get(key) != value for key, value in expected_counts.items()):
        raise ValueError("config protocol or execution counts differ")
    scores_sha = {tag: file_hash(args.prior_scores_root / tag / "scores.jsonl")
                  for tag in design.PRIOR_SCORES_SHA256}
    package = design.build_public_package(load_json(prior_raw), prior_public_sha256=sha(prior_raw),
                                          prior_scores_sha256=scores_sha)
    encoded = canonical(package)
    compressed = gzip.compress(encoded, mtime=0)
    _, evaluator = design.comprehension_cases()
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "public.json").write_bytes(encoded)
    (args.out / "public.json.gz").write_bytes(compressed)
    (args.out / "comprehension_expected.json").write_bytes(canonical(evaluator))
    manifest = {"schema_version": "imcqa-protocol-transport-v1", "public_sha256": sha(encoded),
                "compressed_sha256": sha(compressed), "public_bytes": len(encoded),
                "compressed_bytes": len(compressed), "config_sha256": sha(config_raw),
                "prior_public_sha256": sha(prior_raw), "prior_scores_sha256": scores_sha,
                "real_questions": 40, "contexts_per_model": len(package["jobs"]),
                "blocks": dict(Counter(job["block"] for job in package["jobs"])),
                "execution": dict(Counter(job["execution"] for job in package["jobs"])),
                "real_gold_labels_included": False,
                "inference_files": ["public.json"],
                "private_evaluator_file": "comprehension_expected.json"}
    (args.out / "transport_manifest.json").write_bytes(canonical(manifest))
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
