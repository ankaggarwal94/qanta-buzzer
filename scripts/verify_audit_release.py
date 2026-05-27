#!/usr/bin/env python3
"""Verify that committed CS321M audit artifacts are release-consistent."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PAPER_EXPORTS = ROOT / "paper_exports"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-exports", type=Path, default=PAPER_EXPORTS)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help=(
            "Repo root containing threshold_manifest.json. Defaults to the "
            "real repo root; tests can override for isolation."
        ),
    )
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args(argv)

    paper_exports = args.paper_exports
    repo_root = args.repo_root if args.repo_root is not None else ROOT
    errors: list[str] = []

    required = [
        "csli.json",
        "calibration.json",
        "stopdff.json",
        "audit_card.json",
        "audit_card.md",
        "audit_table.tex",
    ]
    for name in required:
        require(
            (paper_exports / name).exists(),
            f"missing {paper_exports / name}",
            errors,
        )

    if errors:
        for err in errors:
            print(f"ERROR: {err}", file=sys.stderr)
        return 1

    audit = load_json(paper_exports / "audit_card.json")
    generation = audit.get("metadata", {}).get("generation", {})
    if not args.allow_dirty:
        require(
            generation.get("git_dirty") is False,
            "audit_card.json was generated from a dirty tree; rerun from "
            "clean tree or pass --allow-dirty",
            errors,
        )

    artifact_provenance = audit.get("artifact_provenance", {})
    expected_provenance_keys = {"csli.json", "calibration.json", "stopdff.json"}
    missing_provenance = expected_provenance_keys - set(artifact_provenance)
    for missing in sorted(missing_provenance):
        errors.append(
            f"audit_card.artifact_provenance is missing required entry "
            f"for {missing}"
        )
    for artifact_name, block in artifact_provenance.items():
        require(
            block.get("sha_matches") is True,
            f"{artifact_name} producer script SHA mismatch: {block}",
            errors,
        )

    metrics = {m.get("name", ""): m for m in audit.get("metrics", [])}
    stopdff = next((m for n, m in metrics.items() if "StopDFF" in n), None)
    require(stopdff is not None, "StopDFF metric row missing", errors)
    if stopdff:
        details = stopdff.get("details", {})
        if details.get("ceiling_effect_detected") or details.get(
            "unreachable_buckets"
        ):
            require(
                stopdff.get("verdict") == "warn",
                "StopDFF with ceiling/unreachable buckets must be WARN",
                errors,
            )

    provenance = audit.get("data_provenance", {})
    overridden = []
    for metric_name, block in provenance.items():
        if not isinstance(block, dict):
            continue
        for gate_name in ("coverage", "retention"):
            gate = block.get(gate_name)
            if not isinstance(gate, dict):
                continue
            for split_name, split_block in gate.items():
                if (
                    isinstance(split_block, dict)
                    and split_block.get("overridden") is True
                ):
                    overridden.append(
                        f"{metric_name}/{split_name} {gate_name}"
                    )

    md_text = (paper_exports / "audit_card.md").read_text(encoding="utf-8")
    if overridden:
        require(
            "retained MC subset" in md_text,
            "audit_card.md must surface retained MC subset note when gates "
            "are overridden",
            errors,
        )

    threshold_manifest = repo_root / "threshold_manifest.json"
    threshold_sidecar = repo_root / "threshold_manifest.json.sha256"
    if threshold_manifest.exists() and threshold_sidecar.exists():
        sidecar_text = threshold_sidecar.read_text(encoding="utf-8").strip()
        # Sidecar may follow the `sha256sum` format ("<hash>  <filename>") or
        # contain only the hash. Take the leading whitespace-separated token.
        expected = sidecar_text.split()[0] if sidecar_text else ""
        actual = sha256_file(threshold_manifest)
        require(
            actual == expected,
            f"threshold_manifest.json SHA mismatch: expected {expected}, "
            f"got {actual}",
            errors,
        )

    if errors:
        for err in errors:
            print(f"ERROR: {err}", file=sys.stderr)
        return 1

    print("audit release verification passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
