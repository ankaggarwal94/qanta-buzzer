"""Restore the exact frozen public input bundle before a guarded Modal launch."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import stat
import sys
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.jane_gpu_backend import validate_package
from scripts.jane_qwen_backend import _unique_object, _reject_constant

PUBLIC_FILES = {"dev_jobs.json", "main_jobs.json", "dev_choices_only.json",
                "main_choices_only.json"}
ARCHIVE_FILES = PUBLIC_FILES | {"manifest.json"}
MAX_BYTES = 64 * 1024 * 1024


def inspect_archive(archive: Path) -> dict[str, bytes]:
    """Validate names, sizes, bytes, hashes, and public schemas before any write."""
    with ZipFile(archive) as bundle:
        members = bundle.infolist()
        if len(members) != len(ARCHIVE_FILES) or {m.filename for m in members} != ARCHIVE_FILES:
            raise ValueError("archive must contain only the five approved public files")
        if (sum(m.file_size for m in members) > MAX_BYTES
                or any(m.is_dir() or stat.S_ISLNK(m.external_attr >> 16) for m in members)):
            raise ValueError("archive exceeds byte limit or contains a nonregular member")
        content = {m.filename: bundle.read(m) for m in members}
    def load(raw: bytes):
        return json.loads(raw, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    manifest = load(content["manifest.json"])
    if (manifest.get("schema_version") != "jane-public-input-manifest-v1"
            or not isinstance(manifest.get("files"), dict)
            or set(manifest["files"]) != PUBLIC_FILES):
        raise ValueError("unexpected frozen public manifest")
    for name in sorted(PUBLIC_FILES):
        raw, row = content[name], manifest["files"][name]
        if (hashlib.sha256(raw).hexdigest() != row["sha256"]
                or len(raw) != row["byte_count"]):
            raise ValueError("public byte identity mismatch")
        package = load(raw)
        jobs = validate_package(package, max_jobs=10000)
        if (package["schema_version"] != ("jane-choice-controls-v1" if "choices_only" in name
                                          else "jane-public-jobs-v1")
                or len(jobs) != row["job_count"]):
            raise ValueError("public schema or job count mismatch")
    return content


def unpack(archive: Path, out_dir: Path) -> dict:
    """Publish validated public bytes once; existing outputs are never replaced."""
    content = inspect_archive(archive)
    out_dir.mkdir(parents=True, exist_ok=False)
    for name in sorted(content):
        with (out_dir / name).open("xb") as stream:
            stream.write(content[name])
    return {"schema_version": "jane-public-unpack-v1",
            "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
            "files": {name: hashlib.sha256(raw).hexdigest()
                      for name, raw in sorted(content.items())}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(unpack(args.archive, args.out_dir), sort_keys=True))


if __name__ == "__main__":
    main()
