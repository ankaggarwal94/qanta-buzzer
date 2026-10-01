"""Create a prospective interface candidate from the unchanged frozen sample.

No retrieval, inference, grading, or accuracy-based selection occurs here.
The original development gate failure and input archive remain immutable.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

from qb_data.jane_paired import build_jobs
from scripts.modal_jane_pilot import INPUT_FILES, PRIOR_HOST_RECEIPT_SHA256, budget_plan, load_public_inputs


def freeze(original: Path, out: Path, prior_host_receipt: Path) -> dict:
    if hashlib.sha256(prior_host_receipt.read_bytes()).hexdigest() != PRIOR_HOST_RECEIPT_SHA256:
        raise ValueError("prior host receipt differs from the frozen budget debit")
    prior = json.loads(prior_host_receipt.read_text())
    if prior["status"] != "DEVELOPMENT_INTERFACE_GATE_FAILED":
        raise ValueError("candidate only follows the recorded development interface stop")
    original_packages, original_identity = load_public_inputs(original / "public")
    out.mkdir(parents=True, exist_ok=False)
    (out / "public").mkdir()
    (out / "evaluator").mkdir()
    changes = []
    for phase in ("dev", "main"):
        dataset = json.loads((original / "evaluator" / f"{phase}_dataset.json").read_text())
        if dataset["prompt_template"] != "concise_json_v3":
            raise ValueError("original dataset prompt version mismatch")
        old_jobs = build_jobs(dataset)
        assert old_jobs == original_packages[f"{phase}_jobs.json"]["jobs"]
        dataset["prompt_template"] = "concise_json_v4"
        new_jobs = build_jobs(dataset)
        for old, new in zip(old_jobs, new_jobs, strict=True):
            excluded = {"job_id", "prompt", "prompt_sha256"}
            assert {k: v for k, v in old.items() if k not in excluded} == {
                k: v for k, v in new.items() if k not in excluded}
            assert json.loads(old["prompt"].split("\n\n", 1)[0]) == json.loads(new["prompt"].split("\n\n", 1)[1])
        for path, value in ((out / "evaluator" / f"{phase}_dataset.json", dataset),
                            (out / "public" / f"{phase}_jobs.json",
                             {**original_packages[f"{phase}_jobs.json"], "jobs": new_jobs})):
            with path.open("x") as stream:
                json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
                stream.write("\n")
        for subdir, name in (("public", f"{phase}_choices_only.json"),
                             ("evaluator", f"{phase}_choices_only_gold.json")):
            shutil.copyfile(original / subdir / name, out / subdir / name)
        changes.append({"phase": phase, "questions": len(dataset["questions"]), "jobs": len(new_jobs),
                        "unchanged": ["questions", "prefixes", "menus", "gold", "splits", "choice_controls"],
                        "changed": ["prompt_template", "prompt", "prompt_sha256", "job_id"]})
    manifest = {"schema_version": "jane-public-input-manifest-v1", "files": {}}
    for name in INPUT_FILES:
        raw = (out / "public" / name).read_bytes()
        manifest["files"][name] = {"sha256": hashlib.sha256(raw).hexdigest(),
                                   "byte_count": len(raw), "job_count": len(json.loads(raw)["jobs"])}
    (out / "public" / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    _, identity = load_public_inputs(out / "public")
    record = {"schema_version": "jane-prospective-interface-candidate-v1",
              "frozen_utc": datetime.now(timezone.utc).isoformat(), "candidate": "concise_json_v4",
              "original_public_input_id": original_identity["public_input_id"],
              "candidate_public_input_id": identity["public_input_id"], "changes": changes,
              "basis": "Only strict-schema failures on the 12 development questions; no main inference or accuracy selection.",
              "correction": "Instructions precede unchanged question JSON; short answer, low confidence allowed, exact null/null abstention object, no explanations.",
              "preserved": ["models", "revisions", "greedy", "seed", "batching", "token caps", "parser", "95% per-format gate"],
              "original_status": prior["status"], "original_all_3b_oe_abstain": True,
              "coverage_gate_added": False, "automatic_reruns": 0, "budget": budget_plan("7.74")}
    (out / "prompt_candidate_freeze.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    with zipfile.ZipFile(out / "public_inputs.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in (*INPUT_FILES, "manifest.json"):
            archive.write(out / "public" / name, name)
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prior-host-receipt", type=Path, required=True)
    args = parser.parse_args()
    record = freeze(args.original, args.out, args.prior_host_receipt)
    print(json.dumps({"candidate_public_input_id": record["candidate_public_input_id"], "budget": record["budget"]}))


if __name__ == "__main__":
    main()
