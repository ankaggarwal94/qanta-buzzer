"""Freeze public fixed-proposal binary decisions and a separate evaluator fixture."""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

from scripts import imcqa_binary_design as design
from scripts.acl_option_scoring import canonical, load_json, sha


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prior-public", "prior-scores", "config", "out"):
        parser.add_argument("--"+name, required=True, type=Path)
    args = parser.parse_args()
    public_raw, scores_raw, config_raw = args.prior_public.read_bytes(), args.prior_scores.read_bytes(), args.config.read_bytes()
    if sha(public_raw) != design.PRIOR_PUBLIC_SHA256 or sha(scores_raw) != design.PRIOR_QWEN7B_SCORES_SHA256:
        raise ValueError("source artifact hashes differ")
    config = load_json(config_raw)
    counts = {"contexts_per_model": 832, "real_contexts_per_model": 800, "synthetic_contexts_per_model": 32}
    if config.get("protocol") != design.PROTOCOL or any(config.get("execution", {}).get(key) != value for key, value in counts.items()):
        raise ValueError("binary config or counts differ")
    scores = [load_json(line) for line in scores_raw.splitlines()]
    public = design.build_public_package(load_json(public_raw), scores)
    encoded = canonical(public)
    compressed = gzip.compress(encoded, mtime=0)
    _, evaluator = design.comprehension_cases()
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out/"public.json").write_bytes(encoded)
    (args.out/"public.json.gz").write_bytes(compressed)
    (args.out/"comprehension_expected.json").write_bytes(canonical(evaluator))
    manifest = {"schema_version": "imcqa-binary-transport-v1", "public_sha256": sha(encoded),
                "compressed_sha256": sha(compressed), "public_bytes": len(encoded), "compressed_bytes": len(compressed),
                "config_sha256": sha(config_raw), "prior_public_sha256": sha(public_raw),
                "prior_qwen7b_scores_sha256": sha(scores_raw), "n_questions": 40, **counts,
                "real_gold_or_confidence_included": False, "inference_files": ["public.json"],
                "private_evaluator_file": "comprehension_expected.json"}
    (args.out/"transport_manifest.json").write_bytes(canonical(manifest))
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
