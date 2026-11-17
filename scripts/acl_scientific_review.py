"""Collect or compile source-specific decisions for scientific menu/seed combinations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from qb_data.acl_menus import STRATEGIES
from qb_data.acl_protocol import digest, publish_json
from qb_data.acl_scientific_review import compile_review, extend_template
from scripts.acl_expansion import load_inputs
from scripts.stopdff_v5.identity import load_json_strict


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["template", "compile"])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--menu", action="append", choices=sorted(STRATEGIES))
    parser.add_argument("--seed", action="append", type=int)
    parser.add_argument("--decisions", type=Path, help="existing template or reviewed class decisions")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "compile" and not args.decisions:
        parser.error("compile requires --decisions")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    decisions = load_json_strict(args.decisions) if args.decisions else None
    dataset = load_json_strict(config["inputs"]["dataset"])
    registry = load_json_strict(config["inputs"]["oracle"])
    compiled = []
    for menu in dict.fromkeys(args.menu or [config["arm"]["menu"]]):
        for seed in dict.fromkeys(args.seed or [config["arm"]["seed"]]):
            # This private preparation only discovers candidates for source review.
            # No learner, model-policy prediction, score, or inference export is run.
            prepared = load_inputs(config, menu, None, seed, None, review_draft=True)
            if args.command == "template":
                decisions = extend_template(prepared, dataset, registry, existing=decisions)
                continue
            _, receipt = compile_review(prepared, dataset, registry, decisions)
            compiled.append({"menu": menu, "seed": seed, **receipt})
    if args.command == "template":
        publish_json(args.out, decisions)
        print(json.dumps({"source_questions": len(decisions["questions"]), "template": str(args.out)}))
    else:
        publish_json(args.out, {"class_review_sha256": digest(decisions), "verified": compiled})


if __name__ == "__main__":
    main()
