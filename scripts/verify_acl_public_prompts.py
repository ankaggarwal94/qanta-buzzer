"""Replay all frozen historical public prompts without loading evaluator data or models."""
from __future__ import annotations

import argparse
import base64
from collections import Counter
import hashlib
import json
import lzma
from pathlib import Path
import subprocess
from types import SimpleNamespace

from qb_data.acl_protocol import digest, encode
from scripts.acl_paired_inference import JANE_REVISION, paired_jobs


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_SHA256 = "594b961d22cc7db2278315bf8c1d17e0599cdab8f0fe6780a2bee5bc3ba4c6ec"


def blob(name):
    return subprocess.check_output(["git", "-C", str(ROOT), "show",
                                    f"{JANE_REVISION}:acl_expansion_public/{name}"])


def check_blob(raw, size, expected_hash):
    if len(raw) != size or hashlib.sha256(raw).hexdigest() != expected_hash:
        raise ValueError("public transport digest or size mismatch")


def load_compact():
    raw_manifest = blob("transport_manifest.json")
    if hashlib.sha256(raw_manifest).hexdigest() != MANIFEST_SHA256:
        raise ValueError("public manifest differs from independently pinned bytes")
    manifest = json.loads(raw_manifest)
    encoded = bytearray()
    for i, entry in enumerate(manifest["chunks"]):
        name = f"chunk_{i:03d}.b64"
        if entry["name"] != name:
            raise ValueError("unexpected transport chunk order")
        chunk = blob(name)
        check_blob(chunk, entry["byte_count"], entry["sha256"])
        encoded.extend(chunk)
    compressed = base64.b64decode(encoded, validate=True)
    check_blob(compressed, manifest["compressed_bytes"], manifest["compressed_sha256"])
    data = lzma.decompress(compressed, format=lzma.FORMAT_XZ)
    check_blob(data, manifest["compact_bytes"], manifest["compact_sha256"])
    return manifest, json.loads(data)


class FrozenPackageHash:
    """Stream original two-space JSON file serialization, including all job IDs and order."""
    def __init__(self, schema):
        self.hash = hashlib.sha256()
        self.size = self.jobs = 0
        self.update((f'{{\n  "schema_version": "{schema}",\n  "evidence_scope": "scientific",'
                     '\n  "jobs": [\n').encode())

    def update(self, raw):
        self.hash.update(raw)
        self.size += len(raw)

    def append(self, identity):
        prompt_sha = hashlib.sha256(identity["prompt"].encode()).hexdigest()
        job = {"job_id": digest(identity), **identity, "prompt_sha256": prompt_sha}
        raw = json.dumps(job, indent=2, ensure_ascii=False, allow_nan=False).encode()
        self.update((b",\n" if self.jobs else b"") + b"    " + raw.replace(b"\n", b"\n    "))
        self.jobs += 1

    def finish(self, expected):
        self.update(b"\n  ]\n}\n")
        actual = {"sha256": self.hash.hexdigest(), "byte_count": self.size, "job_count": self.jobs}
        if actual != expected:
            raise ValueError(f"ordered historical file replay differs: {actual}")
        return actual


def verify():
    manifest, compact = load_compact()
    templates = compact["templates"]
    main = FrozenPackageHash("jane-public-jobs-v1")
    controls = FrozenPackageHash("jane-choice-controls-v1")
    questions_by_split, cells, control_cells = Counter(), Counter(), Counter()
    for row in compact["questions"]:
        qid, split = row["qid"], row["split"]
        questions_by_split[split] += 1
        base = {"qid": qid, "group_id": row["group_id"], "split": split}
        prefixes = [row["question"][:p["end"]] for p in row["prefixes"]]
        exported = []
        for menu in row["menus"]:
            # These are the only question attributes the production paired exporter
            # is allowed to access. Private answer/slot metadata is unavailable.
            question = SimpleNamespace(qid=qid, cumulative_prefixes=prefixes,
                                       options=[o["text"] for o in menu["options"]])
            projection = SimpleNamespace(questions={split: [question]}, groups={qid: row["group_id"]})
            jobs = paired_jobs(projection, split)
            if len(jobs) != 2 * len(prefixes):
                raise ValueError("incomplete paired exporter coverage")
            exported.append(jobs)
        if exported[0][::2] != exported[1][::2]:
            raise ValueError("OE jobs differ across public menus")
        for i, prefix in enumerate(row["prefixes"], 1):
            text = row["question"][:prefix["end"]]
            offset = 2 * (i - 1)
            observed = [exported[0][offset], *(batch[offset+1] for batch in exported)]
            for menu, job in zip((None, *row["menus"]), observed, strict=True):
                arm = "oe" if menu is None else "mc"
                condition = "oe" if menu is None else menu["condition"]
                if (job["arm"], job["prefix_index"], job["round"], job["qid"], job["group_id"],
                    job["split"]) != (arm, i-1, i, qid, row["group_id"], split):
                    raise ValueError("paired exporter changed the historical public alignment")
                actual = job["prompt"]
                if job["prompt_sha256"] != hashlib.sha256(actual.encode()).hexdigest():
                    raise ValueError("paired exporter prompt hash mismatch")
                payload = {"question_prefix": text}
                if menu is not None:
                    payload["options"] = menu["options"]
                if actual != templates[arm] + "\n\n" + encode(payload).decode():
                    raise ValueError(f"renderer differs at {qid}, {prefix['prefix_id']}, {condition}")
                main.append({**base, "format": arm, "condition": condition,
                             "menu_id": None if menu is None else menu["menu_id"],
                             "prefix_id": prefix["prefix_id"], "fraction": prefix["fraction"],
                             "prompt": actual})
                cells[split, prefix["prefix_id"], condition] += 1
        for menu in row["menus"]:
            options = menu["options"]
            text = "\n".join(f"{opt['id']}. {opt['text']}" for opt in options)
            prompt = templates["control_prefix"] + text + templates["control_suffix"]
            controls.append({**base, "format": "mc", "condition": menu["condition"],
                             "menu_id": menu["menu_id"], "options": options, "prompt": prompt})
            control_cells[split, menu["condition"]] += 1
    files = {name: builder.finish(manifest["files"][name]) for name, builder in (
        ("main_jobs.json", main), ("main_choices_only.json", controls))}
    return {
        "source_revision": JANE_REVISION, "transport_manifest_sha256": MANIFEST_SHA256,
        "renderer_sha256": hashlib.sha256((ROOT/"scripts/acl_paired_inference.py").read_bytes()).hexdigest(),
        "verifier_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "transport_chunks": len(manifest["chunks"]), "compact_sha256": manifest["compact_sha256"],
        "compressed_sha256": manifest["compressed_sha256"], "verified_files": files,
        "scope": "Public prompt bytes, frozen job IDs and original file order. No model or evaluator inputs.",
        "matched_renderer_jobs": sum(cells.values()),
        "shared_oe_cross_menu_checks": sum(n for (_, _, c), n in cells.items() if c == "oe"),
        "transport_only_choices_only_controls": sum(control_cells.values()),
        "questions_by_split": dict(sorted(questions_by_split.items())),
        "matched_jobs_by_split_prefix_condition": [
            {"split": s, "prefix_id": p, "condition": c, "jobs": n}
            for (s, p, c), n in sorted(cells.items(), key=lambda kv: (
                kv[0][0], int(kv[0][1][1:]), kv[0][2]))],
        "choices_only_by_split_condition": [
            {"split": s, "condition": c, "jobs": n}
            for (s, c), n in sorted(control_cells.items())],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = json.dumps(verify(), indent=2, ensure_ascii=False) + "\n"
    if args.out is None:
        print(result, end="")
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(result, encoding="utf-8", newline="\n")
