"""Phase-4 PRE freeze generator (source-only; loads no models, fits nothing).

Generates the committed frozen artifacts required by the PRE-run contract
(ChatGPT Phase-4 adjudication, accepted 2026-08-22; reconciliation document
`phase4_pre_run_reconciliation_2026-08-22.md`):

  * ``frozen/pairing_eligibility_v2.json``  — R-074 (eligible keys, the nine
    ``SINGLE_PREFIX_TRAJECTORY`` exclusions, keyset digest = the D7(b) seed
    input, per-item horizon map + R-073 digest, derivation provenance).
  * ``frozen/parity_anchor_export_a.json``  — R-077 (Export-A anchor: all
    160 nonrandom point fields + 32 nonrandom CI arrays + identity fields,
    stored-precision comparison rules, Random-K exemption metadata).
  * ``frozen/model_snapshot_manifests.json`` — R-075 (role-keyed snapshot
    manifests for primary_scorer / disjoint_selector resolved via the HF
    cache's ``refs/main``, per-file SHA-256; TF-IDF configuration pin), and
    the immutable snapshot copies themselves (symlinks resolved) under an
    OUT-OF-REPO directory supplied by ``--snapshots-out``.
  * ``tests/fixtures/qa012_item10/``        — R-078 (exact-byte full-file
    and excerpt fixtures for the four hit files + authority bindings).

Every input is taken from an explicit CLI path (no hard-coded absolute
paths; committed artifacts record basenames and hashes only) and every
load-bearing source is hash-verified fail-closed before use. Artifacts are
generated, never hand-edited (sign-off Phase 4 item 5); regeneration is
byte-deterministic given identical inputs.

Spec rules owned here: R-073/R-074/R-075/R-077/R-078 artifact production.
Spec: .correctless/specs/camera-ready-aims-evidence-2.md
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from reproducibility.colm_aims_2026 import pairing, qa012, schema  # noqa: E402

EXPORT_A_SHA256 = (
    "59e1c1a74e5fc0cf4f09f8befca87cfc81516684dca2e88dd275c952b28893ff"
)
EXPORT_B_SHA256 = (
    "ba784741ea5f472db50bea7cf24de5ee8eb567e4690c0f73a5e056fb0691a5f9"
)
TEST_DATASET_SHA256 = (
    "638a4df978b77a12655ea72d56daad7fa70851ae486ddb4365d9b060549e34f1"
)
# R-077 frozen allowlist: the export's per-(cell, policy) aggregate fields.
POINT_FIELDS = (
    "n",
    "signed_mean",
    "signed_median",
    "abs_mean",
    "abs_median",
    "mc_earlier",
    "qa_earlier",
    "same_step",
    "mc_never_buzz",
    "qa_never_buzz",
)
CI_FIELDS = ("signed_mean_ci", "signed_median_ci")
POLICIES = ("dp", "myopic")
NONRANDOM_CELLS = (
    "idealized+shared",
    "idealized+performat",
    "khard+shared",
    "khard+performat",
    "kdisjoint+shared",
    "kdisjoint+performat",
    "klex+shared",
    "klex+performat",
)
RANDOM_K_CELLS = ("krandom+shared", "krandom+performat")
# R-075 role-keyed model identities (names as loaded by the producer today)
MODEL_ROLES = {
    "primary_scorer": "sentence-transformers/all-MiniLM-L6-v2",
    "disjoint_selector": "sentence-transformers/all-mpnet-base-v2",
}
# R-075 TF-IDF pin: the producer's exact vectorizer construction
# (scripts/stopdff_fair_qa_retest.py klex selector); all other parameters are
# sklearn defaults and the sklearn version is recorded at certificate time.
TFIDF_CONFIG = {"analyzer": "char_wb", "ngram_range": [2, 4], "fit_corpus": "answer pool"}
QA012_HIT_FILES = {
    "per_prefix_scores_test.jsonl": (
        "32ecda092990c8672ee31ebcc743af446486fc58a2d8679bee38d76a0a99c8da"
    ),
    "per_prefix_scores_test_limit20.jsonl": (
        "8f38ef3f93f9caaa6889bdb1b247594bad7570e60bfbb6e60007998a70fef7f8"
    ),
    "per_prefix_scores_test_sentence-transformers_all-mpnet-base-v2.jsonl": (
        "c3aa63085ad991bfd243a240f0255737cec213d99b1afc9652bd02e96da896ea"
    ),
    "per_prefix_scores_test_limit20_sentence-transformers_all-mpnet-base-v2.jsonl": (
        "f7dcb43bd1a3599062d9ad05cfe0c0d4b5d2745b4b3807fe14c2932aa85b07a3"
    ),
}
EXCERPT_LINES = 2


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_hash(path: Path, expected: str, label: str) -> bytes:
    raw = path.read_bytes()
    observed = hashlib.sha256(raw).hexdigest()
    if observed != expected:
        raise SystemExit(
            f"freeze: {label} hash mismatch: expected {expected}, observed"
            f" {observed} ({path.name})"
        )
    return raw


def _write_artifact(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"[freeze] wrote {path} ({path.stat().st_size} bytes)")


def generate_eligibility(test_dataset: Path, out: Path) -> None:
    raw = _require_hash(test_dataset, TEST_DATASET_SHA256, "test_dataset")
    # R-067: every namespace parse site routes through the hardened loader
    # (non-finite constants, huge exponents, overlong ints, deep nesting).
    doc = schema.parse_json_bytes_strict(raw)
    questions = doc["questions"] if isinstance(doc, dict) else doc
    eligible: dict[str, int] = {}
    excluded = []
    for q in questions:
        qid = str(q["qid"])
        horizon = len(q["cumulative_prefixes"])
        if horizon >= 2:
            eligible[qid] = horizon
        else:
            excluded.append(
                {"item_key": qid, "reason": "SINGLE_PREFIX_TRAJECTORY"}
            )
    excluded.sort(key=lambda e: e["item_key"])
    keys = sorted(eligible)
    payload = {
        "schema_version": schema.SCHEMA_VERSION,
        "artifact_type": "pairing_eligibility",
        "derived_from": {
            "test_dataset_basename": test_dataset.name,
            "test_dataset_sha256": TEST_DATASET_SHA256,
            "derivation": (
                "retained test-split items with >= 2 cumulative prefixes;"
                " horizon = the item's prefix count (identical for both"
                " arms and all ten cells)"
            ),
            "two_party_pin": (
                "test_dataset_sha256 independently recorded by the item-10"
                " Appendix-B balance file (retained_dropped_balance.json)"
            ),
        },
        "eligible_count": len(keys),
        "eligible_keys": keys,
        "pairing_population_keyset_sha256": pairing.keyset_sha256(keys),
        "excluded_count": len(excluded),
        "excluded": excluded,
        "horizon_map": {k: eligible[k] for k in keys},
        "horizon_map_sha256": schema.horizon_map_sha256(eligible),
    }
    if payload["eligible_count"] != 2249 or payload["excluded_count"] != 9:
        raise SystemExit(
            "freeze: eligibility cardinality mismatch:"
            f" {payload['eligible_count']} eligible /"
            f" {payload['excluded_count']} excluded (expected 2249/9)"
        )
    _write_artifact(out, payload)


def generate_parity_anchor(export_a: Path, out: Path) -> None:
    raw = _require_hash(export_a, EXPORT_A_SHA256, "export A")
    # R-067: hardened loader (see generate_eligibility).
    doc = schema.parse_json_bytes_strict(raw)
    results = doc["results"]
    missing = [
        c for c in NONRANDOM_CELLS + RANDOM_K_CELLS if c not in results
    ]
    if missing:
        raise SystemExit(f"freeze: export A missing cells: {missing}")
    expected: dict[str, dict] = {}
    point_count = ci_count = 0
    for cell in NONRANDOM_CELLS:
        expected[cell] = {}
        for pol in POLICIES:
            block = results[cell][pol]
            entry = {}
            for f in POINT_FIELDS:
                entry[f] = block[f]
                point_count += 1
            for f in CI_FIELDS:
                entry[f] = block[f]
                ci_count += 1
            expected[cell][pol] = entry
    if point_count != 160 or ci_count != 32:
        raise SystemExit(
            f"freeze: allowlist cardinality {point_count}/{ci_count}"
            " (expected 160 point fields / 32 CI arrays)"
        )
    payload = {
        "schema_version": schema.SCHEMA_VERSION,
        "artifact_type": "parity_anchor",
        "source": {"basename": export_a.name, "sha256": EXPORT_A_SHA256},
        "corroborative": {
            "basename": "stopdff_fair_qa_with_coverage.json",
            "sha256": EXPORT_B_SHA256,
            "role": "corroborative only, never the anchor",
        },
        "comparison_rules": {
            "rounding": "the producer's own round(x, 4) write path",
            "equality": "parsed JSON value exact equality per field",
            "ci_arrays_blocking": True,
            "any_mismatch_blocking": True,
        },
        "identity_fields": {
            "n_eval": doc["metadata"]["n_eval"],
            "n_fit": doc["metadata"]["n_fit"],
            "per_cell_n": 2249,
        },
        "policies": list(POLICIES),
        "point_fields": list(POINT_FIELDS),
        "ci_fields": list(CI_FIELDS),
        "field_count": {"point": point_count, "ci_arrays": ci_count},
        "nonrandom_cells": list(NONRANDOM_CELLS),
        "expected": expected,
        "random_k": {
            "cells": list(RANDOM_K_CELLS),
            "exempt_from_historical_parity": True,
            "archived_rng_pinned": False,
            "fresh_rng_pinned": True,
            "informational_archived_values": {
                cell: results[cell] for cell in RANDOM_K_CELLS
            },
        },
        "label_mapping": {"performat": "format_specific"},
    }
    _write_artifact(out, payload)


def _resolve_hf_snapshot(hf_cache: Path, model_name: str) -> Path:
    slug = "models--" + model_name.replace("/", "--")
    ref = hf_cache / slug / "refs" / "main"
    if not ref.is_file():
        raise SystemExit(f"freeze: no refs/main for {model_name} in {hf_cache}")
    revision = ref.read_text(encoding="utf-8").strip()
    snap = hf_cache / slug / "snapshots" / revision
    if not snap.is_dir():
        raise SystemExit(f"freeze: snapshot dir missing for {model_name}@{revision}")
    return snap


def generate_model_snapshots(
    hf_cache: Path, snapshots_out: Path, out: Path
) -> None:
    manifests: dict[str, dict] = {}
    for role, model_name in MODEL_ROLES.items():
        snap = _resolve_hf_snapshot(hf_cache, model_name)
        dest = snapshots_out / role
        if dest.exists():
            shutil.rmtree(dest)
        # copy resolving symlinks so the pinned snapshot is self-contained
        shutil.copytree(snap, dest, symlinks=False)
        files = {}
        for f in sorted(p for p in dest.rglob("*") if p.is_file()):
            files[str(f.relative_to(dest))] = {
                "sha256": _sha256_file(f),
                "size": f.stat().st_size,
            }
        if not files:
            raise SystemExit(f"freeze: empty snapshot copy for {role}")
        manifests[role] = {
            "model_name": model_name,
            "hf_revision": snap.name,
            "file_count": len(files),
            "files": files,
        }
    payload = {
        "schema_version": schema.SCHEMA_VERSION,
        "artifact_type": "model_snapshot_manifests",
        "offline_flags_required": ["HF_HUB_OFFLINE=1", "TRANSFORMERS_OFFLINE=1"],
        "roles": manifests,
        "tfidf_config": TFIDF_CONFIG,
        "note": (
            "snapshot copies live OUTSIDE the repository; the certificate"
            " verifies a supplied snapshot directory against these per-file"
            " digests before the run"
        ),
    }
    _write_artifact(out, payload)


def generate_qa012_fixtures(item10_dir: Path, fixtures_out: Path) -> None:
    fixtures_out.mkdir(parents=True, exist_ok=True)
    authority = qa012.load_authority_manifest()
    hit_entries = [entry for entry in authority["entries"] if entry["format_qa_hits"]]
    bindings: dict[str, dict] = {}
    for basename, expected in sorted(QA012_HIT_FILES.items()):
        matches = [p for p in item10_dir.rglob(basename) if p.is_file()]
        if len(matches) != 1:
            raise SystemExit(
                f"freeze: expected exactly one {basename} under {item10_dir},"
                f" found {len(matches)}"
            )
        raw = _require_hash(matches[0], expected, basename)
        lines = raw.split(b"\n")
        excerpt = b"\n".join(lines[:EXCERPT_LINES]) + b"\n"
        full_fixture = fixtures_out / basename
        fixture = fixtures_out / f"{basename}.first{EXCERPT_LINES}.jsonl"
        full_fixture.write_bytes(raw)
        fixture.write_bytes(excerpt)

        authority_matches = [
            entry
            for entry in hit_entries
            if entry["sha256"] == expected
            and Path(entry["path"].partition(":")[2].strip()).name == basename
        ]
        if len(authority_matches) != 1:
            raise SystemExit(
                f"freeze: expected exactly one authority entry for {basename},"
                f" found {len(authority_matches)}"
            )
        authority_entry = authority_matches[0]
        scope_prong, relative_path = qa012._authority_prong_relative_path(
            authority_entry["path"]
        )
        normalized_hits = []
        for hit in authority_entry["format_qa_hits"]:
            match = re.fullmatch(r"line ([1-9][0-9]*): (/.*/?format|/format)", hit)
            if match is None:
                raise SystemExit(
                    f"freeze: malformed authority hit for {basename}: {hit}"
                )
            normalized_hits.append(
                {"line": int(match.group(1)), "pointer": match.group(2)}
            )
        hits_sha256 = hashlib.sha256(
            json.dumps(
                normalized_hits, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()
        content_hash = qa012.dropbox_content_hash(raw)
        if (
            authority_entry["size"] != len(raw)
            or authority_entry["dropbox_content_hash"] != content_hash
        ):
            raise SystemExit(
                f"freeze: authority size/content hash mismatch for {basename}"
            )
        bindings[basename] = {
            "scope_prong": scope_prong,
            "relative_path": relative_path,
            "full_fixture": full_fixture.name,
            "excerpt_fixture": fixture.name,
            "excerpt_lines": EXCERPT_LINES,
            "excerpt_sha256": hashlib.sha256(excerpt).hexdigest(),
            "dropbox_content_hash": content_hash,
            "hit_count": len(normalized_hits),
            "hits_sha256": hits_sha256,
            "full_file_sha256": expected,
            "full_file_size": len(raw),
        }
        print(f"[freeze] wrote {full_fixture} and {fixture}")
    # Preserve the deliberately pinned insertion order as well as the values:
    # qa012 verifies the complete raw bindings artifact, not merely parsed JSON.
    payload = {
        "artifact_type": "qa012_compatibility_fixture_bindings",
        "files": bindings,
        "schema_version": schema.SCHEMA_VERSION,
        "source_bundle": "item10_reachable_comparator_prototype",
    }
    bindings_path = fixtures_out / "bindings.json"
    bindings_path.write_bytes((json.dumps(payload, indent=1) + "\n").encode("utf-8"))
    print(f"[freeze] wrote {bindings_path} ({bindings_path.stat().st_size} bytes)")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--test-dataset", type=Path, required=True)
    ap.add_argument("--export-a", type=Path, required=True)
    ap.add_argument("--hf-cache", type=Path, required=True)
    ap.add_argument("--snapshots-out", type=Path, required=True)
    ap.add_argument("--item10-dir", type=Path, required=True)
    args = ap.parse_args(argv)
    frozen = _REPO_ROOT / "reproducibility/colm_aims_2026/frozen"
    generate_eligibility(args.test_dataset, frozen / "pairing_eligibility_v2.json")
    generate_parity_anchor(args.export_a, frozen / "parity_anchor_export_a.json")
    generate_model_snapshots(
        args.hf_cache, args.snapshots_out, frozen / "model_snapshot_manifests.json"
    )
    generate_qa012_fixtures(
        args.item10_dir, _REPO_ROOT / "tests/fixtures/qa012_item10"
    )
    print("[freeze] all artifacts generated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
