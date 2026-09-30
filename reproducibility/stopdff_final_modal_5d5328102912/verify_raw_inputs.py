#!/usr/bin/env python3
"""Verify the ten exact pre-v5 raw inputs bound to the certified StopDFF run."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="repository root containing data/processed and paper_exports",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).with_name("run_identity.json"),
    )
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows = manifest.get("raw_inputs")
    if not isinstance(rows, list):
        raise ValueError("run_identity.json lacks a raw_inputs array")

    failures: list[str] = []
    for row in rows:
        rel = Path(row["path"])
        path = args.repo_root / rel
        if not path.is_file():
            failures.append(f"MISSING {rel}")
            continue
        size = path.stat().st_size
        digest = sha256_file(path)
        if size != int(row["size"]) or digest != str(row["sha256"]):
            failures.append(
                f"MISMATCH {rel}: size={size}, sha256={digest}; "
                f"expected size={row['size']}, sha256={row['sha256']}"
            )
        else:
            print(f"OK {rel}")

    if failures:
        for failure in failures:
            print(failure, file=sys.stderr)
        return 1

    print(f"PASS: verified {len(rows)} raw inputs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
