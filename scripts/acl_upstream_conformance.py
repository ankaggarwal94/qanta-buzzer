"""Regenerate a synthetic interoperability fixture with the unmodified historical ACL preparer."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

from qb_data.acl_protocol import digest, encode
from scripts.acl_paired_inference import JANE_REVISION

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/acl_upstream_freeze.json.gz"
BLOBS = ("qb_data/jane_paired.py", "scripts/prepare_jane_gpu_pilot.py",
         "scripts/prepare_acl_expansion.py", "scripts/jane_gpu_backend.py",
         "scripts/jane_qwen_backend.py", "scripts/jane_output_constraints.py")


def synthetic_rows():
    rows = []
    for index in range(12):
        tag = chr(ord("a") + index)
        category = "History" if index % 2 == 0 else "Literature"
        words = [category, "synthetic", tag, *[f"{tag}clue{j}" for j in range(37)]]
        rows.append({"_id": f"fixture-{tag}", "year":2018, "difficulty":3,
                     "standard":True, "category":category,
                     "question_sanitized":" ".join(words),
                     "answer":f"<u>Fixture {tag}</u> [accept Synthetic {tag}]"})
    # One duplicate joins a nontrivial component; the historical representative
    # and full membership are chosen by its own grouping code.
    rows.append({**rows[0], "_id":"fixture-a-duplicate"})
    return rows


def regenerate():
    with TemporaryDirectory() as tmp:
        stage = Path(tmp)
        code = {}
        for name in BLOBS:
            data = subprocess.check_output(["git","show",f"{JANE_REVISION}:{name}"], cwd=ROOT)
            if name == "scripts/jane_output_constraints.py" and (ROOT/name).read_bytes() != data:
                raise ValueError("production grammar differs from the pinned upstream blob")
            path = stage / name
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(data)
            (path.parent / "__init__.py").touch()
            code[name] = hashlib.sha256(data).hexdigest()
        (stage/"source.jsonl").write_bytes(b"".join(encode(row)+b"\n" for row in synthetic_rows()))
        (stage/"exclusions.json").write_bytes(b"{}\n")
        # Isolate the historical imports from the installed/current qb_data package.
        driver = """
import json, sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
from scripts.prepare_acl_expansion import Config, prepare
from qb_data.jane_paired import validate_dataset
prepare(Path("source.jsonl"), Path("exclusions.json"), Path("freeze"),
        Config(seed=1, main_count=4, calibration_count=1, selection_count=1,
               reservoir_per_category=3, holdout_count=0))
validate_dataset(json.loads(Path("freeze/evaluator/main_dataset.json").read_bytes()))
"""
        subprocess.run([sys.executable,"-I","-c",driver],cwd=stage,check=True)
        private = stage/"freeze/evaluator"
        return {"classification":"synthetic schema conformance",
                "upstream_revision":JANE_REVISION, "upstream_code_sha256":code,
                "dataset":json.loads((private/"main_dataset.json").read_bytes()),
                "reservoir":json.loads((private/"reservoir.json").read_bytes())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="create the missing golden fixture")
    args = parser.parse_args()
    generated = regenerate()
    if args.write:
        FIXTURE.parent.mkdir(parents=True,exist_ok=True)
        with FIXTURE.open("xb") as fp:
            with gzip.GzipFile(filename="", mode="wb", fileobj=fp, mtime=0) as gz:
                gz.write(encode(generated))
    with gzip.open(FIXTURE,"rb") as fp:
        if json.load(fp) != generated:
            raise ValueError("pinned upstream output differs from the retained fixture")
    print("Synthetic upstream fixture verified", digest(generated))


if __name__ == "__main__":
    main()
