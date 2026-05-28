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

# Canonical producer scripts for each source-metric artifact, per ARTIFACTS.md.
# Pinning these prevents a tampered audit card from redirecting `script_path`
# to an unchanged helper file and bypassing the producer-drift check.
EXPECTED_PRODUCERS: dict[str, str] = {
    "csli.json": "scripts/compute_csli.py",
    "calibration.json": "scripts/compute_prefix_calibration.py",
    "stopdff.json": "scripts/compute_stopdff.py",
}

# Canonical generator for the audit card itself, per ARTIFACTS.md.
EXPECTED_AUDIT_CARD_GENERATOR = "scripts/make_audit_card.py"


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
        "csli_panel.png",
        "reliability_early.png",
        "reliability_mid.png",
        "reliability_late.png",
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

    # Recompute the audit-card generator's SHA against the live tree. The
    # audit card and Markdown are themselves canonical generated artifacts;
    # if make_audit_card.py is edited after the card was generated, stale
    # aggregation/rendering logic could ship without any source-metric SHA
    # mismatch.
    generator_script_path = generation.get("script_path")
    generator_recorded_sha = generation.get("script_sha256")
    if not isinstance(generator_script_path, str) or not generator_script_path:
        errors.append(
            "audit_card.metadata.generation is missing script_path"
        )
    elif not isinstance(generator_recorded_sha, str) or not generator_recorded_sha:
        errors.append(
            "audit_card.metadata.generation is missing script_sha256"
        )
    elif generator_script_path != EXPECTED_AUDIT_CARD_GENERATOR:
        errors.append(
            f"audit_card.metadata.generation.script_path is "
            f"{generator_script_path!r}, expected canonical generator "
            f"{EXPECTED_AUDIT_CARD_GENERATOR!r} (see ARTIFACTS.md)"
        )
    else:
        gen_script = repo_root / generator_script_path
        if not gen_script.exists():
            errors.append(
                f"audit-card generator script not found at {gen_script}"
            )
        else:
            live_gen_sha = sha256_file(gen_script)
            require(
                live_gen_sha == generator_recorded_sha,
                f"audit-card generator SHA drift: "
                f"recorded={generator_recorded_sha}, live={live_gen_sha} "
                f"(script_path={generator_script_path})",
                errors,
            )

    # Bind audit_card.md to the same generation as audit_card.json.
    # Without this, the Markdown could be left from an older run or be
    # hand-edited and the verifier would still pass because it only
    # reads the Markdown to look for the retained-subset phrase.
    recorded_md_sha = generation.get("markdown_sha256")
    if not isinstance(recorded_md_sha, str) or not recorded_md_sha:
        errors.append(
            "audit_card.metadata.generation is missing markdown_sha256; "
            "rerun make_audit_card.py to bind audit_card.md"
        )
    else:
        live_md_sha = sha256_file(paper_exports / "audit_card.md")
        require(
            live_md_sha == recorded_md_sha,
            f"audit_card.md content SHA drift: "
            f"recorded={recorded_md_sha}, live={live_md_sha} "
            f"(audit_card.md does not match the audit_card.json generation; "
            f"rerun make_audit_card.py)",
            errors,
        )

    artifact_provenance = audit.get("artifact_provenance", {})
    expected_provenance_keys = set(EXPECTED_PRODUCERS)
    missing_provenance = expected_provenance_keys - set(artifact_provenance)
    for missing in sorted(missing_provenance):
        errors.append(
            f"audit_card.artifact_provenance is missing required entry "
            f"for {missing}"
        )
    # Recompute producer-script SHAs against the live tree rather than trust
    # the cached `sha_matches` flag in audit_card.json. The cached flag goes
    # stale if a producer script is edited after `make_audit_card.py` ran.
    for artifact_name, block in artifact_provenance.items():
        if not isinstance(block, dict):
            errors.append(
                f"artifact_provenance[{artifact_name}] is not a dict: {block}"
            )
            continue

        recorded_sha = block.get("recorded_sha256")
        script_path_str = block.get("script_path")

        if not isinstance(recorded_sha, str) or not recorded_sha:
            errors.append(
                f"artifact_provenance[{artifact_name}] missing "
                f"recorded_sha256"
            )
            continue
        if not isinstance(script_path_str, str) or not script_path_str:
            errors.append(
                f"artifact_provenance[{artifact_name}] missing script_path"
            )
            continue

        # Pin known artifacts to their canonical producer (per ARTIFACTS.md)
        # so a tampered card cannot redirect script_path to an unchanged
        # helper file and bypass the producer-drift check.
        expected_producer = EXPECTED_PRODUCERS.get(artifact_name)
        if expected_producer is not None and script_path_str != expected_producer:
            errors.append(
                f"artifact_provenance[{artifact_name}].script_path is "
                f"{script_path_str!r}, expected canonical producer "
                f"{expected_producer!r} (see ARTIFACTS.md)"
            )
            continue

        script_path = repo_root / script_path_str
        if not script_path.exists():
            errors.append(
                f"{artifact_name} producer script not found at "
                f"{script_path}"
            )
            continue

        live_sha = sha256_file(script_path)
        require(
            live_sha == recorded_sha,
            f"{artifact_name} producer script SHA drift: "
            f"recorded={recorded_sha}, live={live_sha} "
            f"(script_path={script_path_str})",
            errors,
        )

        # Bind the audit card to the exact bytes of the source JSON it
        # aggregated. Without this, csli.json/calibration.json/stopdff.json
        # could be regenerated or hand-edited after make_audit_card.py ran,
        # and the audit card's verdicts would be stale.
        recorded_content_sha = block.get("content_sha256")
        if not isinstance(recorded_content_sha, str) or not recorded_content_sha:
            errors.append(
                f"artifact_provenance[{artifact_name}] missing "
                f"content_sha256; rerun make_audit_card.py to record it"
            )
            continue
        source_path = paper_exports / artifact_name
        # Existence was already required above, but guard anyway for
        # robustness when this branch is reached via tests.
        if not source_path.exists():
            errors.append(f"missing source artifact {source_path}")
            continue
        live_content_sha = sha256_file(source_path)
        require(
            live_content_sha == recorded_content_sha,
            f"{artifact_name} content SHA drift: "
            f"recorded={recorded_content_sha}, live={live_content_sha} "
            f"(audit_card.json was generated against an earlier revision; "
            f"rerun make_audit_card.py)",
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
    require(
        threshold_manifest.exists(),
        f"missing {threshold_manifest}",
        errors,
    )
    require(
        threshold_sidecar.exists(),
        f"missing {threshold_sidecar}",
        errors,
    )
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
