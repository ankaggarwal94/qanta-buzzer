"""Prepare and hash the development-only public pilot from frozen CPU inputs."""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
from pathlib import Path

from scripts.acl_option_scoring import canonical, load_json
from scripts.imcqa_wait_scoring import SOURCE_INPUT_SHA256, build_public_package


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("public-jobs", "category-map", "config", "out"):
        p.add_argument("--"+name, required=True, type=Path)
    a=p.parse_args()
    raw, categories_raw, config_raw = a.public_jobs.read_bytes(), a.category_map.read_bytes(), a.config.read_bytes()
    config=load_json(config_raw)
    if sha(raw) != SOURCE_INPUT_SHA256 or sha(raw) != config["frozen_source_sha256"]["public/main_jobs.json"]:
        raise ValueError("public source hash mismatch")
    if sha(categories_raw) != config["frozen_source_sha256"]["qid_category.json"]:
        raise ValueError("public category-map hash mismatch")
    package=build_public_package(load_json(raw), source_input_sha256=sha(raw), category_map=load_json(categories_raw),
                                 n_per_split=config["n_per_split"], n_diagnostic_per_split=config["n_diagnostic_per_split"], seed=config["seed"])
    encoded=canonical(package)
    if len(package["jobs"]) != config["execution"]["production_contexts_per_model"]:
        raise ValueError("context count differs from frozen design")
    a.out.mkdir(parents=True,exist_ok=False)
    (a.out/"pilot.json").write_bytes(encoded)
    compressed=gzip.compress(encoded,mtime=0)
    (a.out/"pilot.json.gz").write_bytes(compressed)
    manifest={"schema":"imcqa-public-transport-v1", "public_sha256":sha(encoded),
              "compressed_sha256":sha(compressed), "public_bytes":len(encoded), "compressed_bytes":len(compressed),
              "config_sha256":sha(config_raw), "source_input_sha256":sha(raw), "category_map_sha256":sha(categories_raw),
              "questions":sum(map(len,package["selection"]["selected_qids"].values())), "contexts":len(package["jobs"]),
              "gold_labels_included":False}
    (a.out/"transport_manifest.json").write_bytes(canonical(manifest))
    print(json.dumps(manifest,sort_keys=True))


if __name__ == "__main__":
    main()
