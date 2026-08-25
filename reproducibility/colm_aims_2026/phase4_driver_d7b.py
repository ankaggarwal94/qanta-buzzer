"""Phase-4 D7(b) real-input driver (source-only; loads no models).

The FIRST production reader that constructs the caller-supplied identity
blocks the ``phase4_assemble_d7b`` orchestrator needs from the REAL frozen
inputs + a records root + a (parameterized) D6 baseline + a QA-012 manifest,
then calls ``build_evidence_package`` to emit the real ``profile.json`` +
evidence package + closure inventory. Before this module the identity blocks
existed only as ``tests/_colm_aims_v2_helpers.make_*`` synthesizers.

Identity-block provenance (constants vs. input-derived):

  * arms / llm_involvement / grid.held_fixed identities / estimand study-design
    fields  -> FIXED study-design constants. The per-family stop vocabulary,
    populations, sentinel/imputation conventions, and the 5x2 grid axes are
    sourced from ``schema`` (never re-hardcoded); the remaining study-design
    scalars (arm selector/scorer, MC-trajectory / calibration / continuation /
    producer-profile identities) are documented constants below, pinned by the
    independent oracle in ``tests/_colm_aims_v2_helpers`` at test time.
  * grid.item_keys_sha256 / grid.held_fixed.horizon_identity / estimand horizon
    digests  -> RECORD-DERIVED (recomputed from ``records_root`` fail-closed;
    they must equal the assembler's own recompute or the verifier refuses).
  * provenance.model  -> derived from ``frozen/model_snapshot_manifests.json``
    (the ``primary_scorer`` role). provenance.pre_package_retention /
    splits.eval  -> derived from ``frozen/pairing_eligibility_v2.json`` + the
    record keyset. provenance.input_sha256 + dirty_state.source_commit are
    bound by ``build_evidence_package`` from the staged record bytes.
  * d6_baseline  -> PARAMETERIZED: the pinned two-party ``main.tex``/``main.pdf``
    hashes default from ``closure`` while the COMPLETE FINAL_CHECKSUMS closure
    (post-de-anonymization) is bound from ``--d6-checksums`` / explicit args.
  * qa012  -> the executed detector manifest (``qa012.build_inventory_manifest``
    over ``--qa012-root``) or an explicit status/inventory-hash pair.

Exit codes mirror ``verify.py`` / ``phase4_assemble_d7b`` (0 pass, 2 usage,
3 ingress, 4 internal). Spec rules: R-001..R-011 (profile shape), R-040/R-043
(grid/held-fixed), R-052 (retention), R-071/R-072 (closure), R-074/R-075
(frozen loaders), R-016/R-039 (create-once publish).
Spec: .correctless/specs/camera-ready-aims-evidence-2.md
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from reproducibility.colm_aims_2026 import (  # noqa: E402
    closure,
    pairing,
    phase4,
    phase4_assemble_d7b as assembler,
    qa012,
    schema,
)

# Exit-code contract (mirrors ``verify.py`` / the assembler).
EXIT_PASS = 0
EXIT_USAGE_ERROR = 2
EXIT_INGRESS_ERROR = 3
EXIT_INTERNAL_ERROR = 4

# ---------------------------------------------------------------------------
# Study-design identity constants (documented; ceremony-overridable).
#
# These are the fixed study-design scalars that have no dedicated ``schema``
# constant. Each is pinned by the independent ``make_*`` oracle in
# ``tests/_colm_aims_v2_helpers`` (the driver's builders are asserted equal to
# that oracle), so a drift here is caught by the contract suite rather than by
# re-hardcoding a schema-owned value.
# ---------------------------------------------------------------------------
MC_ARM_ID = "mc_trajectory"
MC_TRAJECTORY_IDENTITY = "traj-mc-v2-0001"
CONTINUATION_IDENTITY = "cont-0001"
CALIBRATION_IDENTITY: dict[str, str] = {
    "shared": "cal-shared-0001",
    "format_specific": "cal-fmt-0001",
}
PRODUCER_PROFILE_IDENTITY = f"{schema.STRICT_PROFILE_ID}:producer-0001"
PAIRING_DEFINITION = "matched_item_prefix_grid"
DENOMINATOR_POLICY = "n_complete"
TIMEOUT_RULE = "zero_indexed_stop_ge_horizon_is_timeout"
RANDOM_K_REFERENCE = "krandom"
RANDOM_K_DRAW_ID = "draw-archived-0001"
NO_DRAW_ID = "draw-none"
PRODUCER_ENTRYPOINT = "reproducibility/colm_aims_2026/phase4_driver_d7b.py"

# The frozen model manifest role/file map bound into provenance.model.
_PRIMARY_SCORER_ROLE = "primary_scorer"
_MODEL_WEIGHTS_FILE = "model.safetensors"
_MODEL_TOKENIZER_CONFIG_FILE = "tokenizer_config.json"
_MODEL_DTYPE = "float32"
_MODEL_DEVICE_CLASS = "cpu"

# dirty_state.source_commit placeholder for a standalone provenance build;
# ``build_evidence_package`` overwrites it with the bound ``source_commit``.
_UNBOUND_SOURCE_COMMIT = "0" * 40

_FROZEN_ELIGIBILITY_BASENAME = "pairing_eligibility_v2.json"
_FROZEN_MANIFEST_BASENAME = "model_snapshot_manifests.json"

_SHA256SUM_LINE = re.compile(r"^([0-9a-f]{64})[ \t]+(.+)$")


# ---------------------------------------------------------------------------
# Record-derived identities (fail-closed; recomputed from records_root)
# ---------------------------------------------------------------------------


def record_keyset_digest(
    complete_by_cell: dict[str, dict[str, dict[str, Any]]],
) -> str:
    """Recompute the shared pairing-population keyset digest from records.

    All ten cells share ONE item key set (R-050); a cell that disagrees is a
    fail-closed defect, never a silently-taken first cell.
    """
    reference: frozenset[str] | None = None
    for cell_id in schema.CELL_IDS:
        keys = frozenset(complete_by_cell[cell_id])
        if reference is None:
            reference = keys
        elif keys != reference:
            raise schema.ColmAimsError(
                f"cell {cell_id!r} item key set differs from the shared"
                " pairing population; all ten cells share ONE key set (R-050)"
            )
    assert reference is not None
    ordered = pairing.canonical_item_order(list(reference))
    return pairing.keyset_sha256(ordered)


def record_horizon_identity(
    complete_by_cell: dict[str, dict[str, dict[str, Any]]],
) -> str:
    """Recompute the held-fixed horizon identity (the per-item horizon-map
    digest, R-073/R-043) from records, fail-closed across cells.

    Each cell's ``{item_key: trajectory_horizon}`` map is digested with the
    shared ``schema.horizon_map_sha256``; a cell whose horizon map digests
    differently breaks the held-fixed invariant and is refused.
    """
    per_cell: dict[str, str] = {}
    for cell_id in schema.CELL_IDS:
        records = complete_by_cell[cell_id]
        horizon_map = {
            item_key: record["trajectory_horizon"]
            for item_key, record in records.items()
        }
        per_cell[cell_id] = schema.horizon_map_sha256(horizon_map)
    distinct = set(per_cell.values())
    if len(distinct) != 1:
        raise schema.ColmAimsError(
            "per-cell horizon-map digests disagree across the ten cells —"
            " the held-fixed horizon identity is not shared (R-043/R-073)"
        )
    return next(iter(distinct))


# ---------------------------------------------------------------------------
# Fixed study-design identity blocks (arms / llm_involvement / grid / estimands)
# ---------------------------------------------------------------------------


def _build_arm(
    arm_id: str,
    *,
    family: str = "constructed_reference",
    cardinality: str = "k_way",
    construction: str = "mc_grid",
    reporting_eligibility: str = "headline_eligible",
) -> dict[str, Any]:
    """One arm identity block (R-003); stop_semantics sourced from schema."""
    return {
        "arm_id": arm_id,
        "family": family,
        "stop_semantics": schema.FAMILY_STOP_VOCAB[family],
        "construction": construction,
        "cardinality": cardinality,
        "selector": "argmax_calibrated_score",
        "scorer": "tiny-scorer",
        "candidate_pool_role": (
            "distractor_pool" if cardinality == "k_way" else "none"
        ),
        "correctness_assignment": (
            "oracle_gold" if construction == "idealized" else "option_match"
        ),
        "calibration_role": "calibrated",
        "continuation_role": "dp_continuation",
        "seed_contract": {"seeds": [1, 2, 3]},
        "reporting_eligibility": reporting_eligibility,
    }


def _build_idealized_arm(arm_id: str = "idealized") -> dict[str, Any]:
    arm = _build_arm(arm_id, cardinality="scalar", construction="idealized")
    arm["scorer"] = "prefix_to_gold_cosine"
    arm["selector"] = "threshold_on_scalar"
    return arm


def build_arms() -> list[dict[str, Any]]:
    """The frozen six-arm study design (R-003): the MC trajectory arm, the
    idealized scalar reference, three k-way constructed references, and the
    non-headline random-K disclosure arm."""
    arms = [
        _build_arm("mc_trajectory", family="learned_continuation"),
        _build_idealized_arm("idealized"),
    ]
    for reference_id in ("kdisjoint", "khard", "klex"):
        arms.append(_build_arm(reference_id))
    arms.append(
        _build_arm(
            "krandom", reporting_eligibility="non_headline_disclosure_only"
        )
    )
    return arms


def build_llm_involvement() -> dict[str, Any]:
    """The all-``none`` LLM-involvement disclosure block (no LLM in the loop)."""
    return {
        "reference_construction": "none",
        "data_plot_creation": "none",
        "evaluation": "none",
    }


def build_grid_block(
    *,
    keyset_digest: str,
    horizon_identity: str,
    mc_trajectory_identity: str = MC_TRAJECTORY_IDENTITY,
) -> dict[str, Any]:
    """The in-profile grid identity block (GRID_KEYS, R-040/R-044).

    Mirrors ``assemble_grid_block`` but is built from the record-derived
    ``keyset_digest``/``horizon_identity`` (rather than an ``InferenceResult``)
    so the driver can construct the grid before the assembler recomputes the
    inference; the two must agree or the verifier's grid legs refuse.
    """
    return {
        "reference_ids": sorted(schema.REFERENCE_IDS),
        "calibration_ids": sorted(schema.CALIBRATION_IDS),
        "cell_ids": list(schema.CELL_IDS),
        "record_files": {
            cell_id: f"records/{cell_id}.jsonl" for cell_id in schema.CELL_IDS
        },
        "item_keys_sha256": keyset_digest,
        "held_fixed": {
            "mc_trajectory_identity": mc_trajectory_identity,
            "horizon_identity": horizon_identity,
        },
    }


def _build_estimand(
    cell_id: str, *, horizon_identity: str, numerical_tolerance: float = 1e-9
) -> dict[str, Any]:
    reference_id, calibration_id = cell_id.split("__", 1)
    return {
        "arm_mc": MC_ARM_ID,
        "arm_ref": reference_id,
        "reference_id": reference_id,
        "calibration_id": calibration_id,
        "pairing_definition": PAIRING_DEFINITION,
        "timeout_parameters": {
            "horizon_map_sha256": horizon_identity,
            "rule": TIMEOUT_RULE,
        },
        "event_representation": {
            "index_base": 0,
            "horizon_identity": horizon_identity,
            "mc_trajectory_identity": MC_TRAJECTORY_IDENTITY,
            "historical_sentinel_convention": schema.SENTINEL_CONVENTION,
            "terminal_imputation_policy": schema.IMPUTATION_FINAL_PREFIX,
            "producer_profile_identity": PRODUCER_PROFILE_IDENTITY,
        },
        "population": schema.POPULATION_ALL,
        "denominator_policy": DENOMINATOR_POLICY,
        "numerical_tolerance": numerical_tolerance,
        "calibration_identity": CALIBRATION_IDENTITY[calibration_id],
        "continuation_identity": CONTINUATION_IDENTITY,
        "random_k_draw_id": (
            RANDOM_K_DRAW_ID if reference_id == RANDOM_K_REFERENCE else NO_DRAW_ID
        ),
    }


def build_estimands(
    *, horizon_identity: str, numerical_tolerance: float = 1e-9
) -> dict[str, dict[str, Any]]:
    """The per-cell estimand identity blocks (ESTIMAND_KEYS, R-005/R-073).

    Every estimand's ``timeout_parameters``/``event_representation`` horizon is
    the record-derived ``horizon_identity`` so the assembler's estimand digest
    and the verifier's horizon legs agree.
    """
    return {
        cell_id: _build_estimand(
            cell_id,
            horizon_identity=horizon_identity,
            numerical_tolerance=numerical_tolerance,
        )
        for cell_id in schema.CELL_IDS
    }


# ---------------------------------------------------------------------------
# provenance (frozen model manifest + eligibility retention + record keyset)
# ---------------------------------------------------------------------------


def _runtime_packages() -> dict[str, str]:
    import numpy as np

    return {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "numpy": np.__version__,
    }


def build_provenance_from_frozen(
    frozen_dir: Path,
    *,
    keyset_digest: str,
    source_commit: str | None = None,
    git_dirty: bool = False,
) -> dict[str, Any]:
    """Assemble the provenance identity block from the frozen inputs.

    ``model`` is bound from the ``primary_scorer`` role of the frozen snapshot
    manifest; ``pre_package_retention`` and the eval split count come from the
    frozen pairing-eligibility artifact; the eval keyset is the record-derived
    ``keyset_digest``. Both frozen artifacts are consumed through the strict,
    digest-recomputing ``phase4`` loaders (R-074/R-075), so a malformed or
    missing frozen input is a typed ingress refusal. ``input_sha256`` and
    ``dirty_state.source_commit`` are (re)bound by ``build_evidence_package``.
    """
    frozen_dir = Path(frozen_dir)
    eligibility = phase4.load_pairing_eligibility(
        frozen_dir / _FROZEN_ELIGIBILITY_BASENAME
    )
    manifest = phase4.load_model_snapshot_manifest(
        frozen_dir / _FROZEN_MANIFEST_BASENAME
    )

    role = manifest["roles"][_PRIMARY_SCORER_ROLE]
    files = role["files"]
    model = {
        "repository_namespace": role["model_name"],
        "revision": role["hf_revision"],
        "weights_sha256": files[_MODEL_WEIGHTS_FILE]["sha256"],
        "tokenizer_config_sha256": files[_MODEL_TOKENIZER_CONFIG_FILE]["sha256"],
        "dtype": _MODEL_DTYPE,
        "device_class": _MODEL_DEVICE_CLASS,
        "numerical_settings": {"deterministic": True},
    }

    eligible_count = eligibility["eligible_count"]
    excluded_count = eligibility["excluded_count"]

    return {
        "producer_entrypoint": PRODUCER_ENTRYPOINT,
        "dirty_state": {
            "git_dirty": bool(git_dirty),
            "source_commit": source_commit or _UNBOUND_SOURCE_COMMIT,
        },
        "calibration_identity": dict(CALIBRATION_IDENTITY),
        "continuation_identity": CONTINUATION_IDENTITY,
        # R-052(a): the upstream-unpaired items are pre-package retention
        # documentation here, never in-package excluded_keys.
        "pre_package_retention": {
            "retained_count": eligible_count + excluded_count,
            "paired_count": eligible_count,
            "upstream_unpaired_count": excluded_count,
        },
        "splits": {
            "eval": {
                "name": "eval-v2",
                "count": eligible_count,
                "keyset_sha256": keyset_digest,
            }
        },
        "model": model,
        "runtime_packages": _runtime_packages(),
        "input_sha256": {},
    }


# ---------------------------------------------------------------------------
# D6 baseline (parameterized FINAL_CHECKSUMS closure) + QA-012 block
# ---------------------------------------------------------------------------


def _parse_final_checksums(raw: bytes, rel: str) -> dict[str, str]:
    """Parse a FINAL_CHECKSUMS manifest into a ``{relpath: sha256}`` map.

    Accepts a JSON object (a flat ``path -> hash`` map, a ``path -> {sha256}``
    map, or one nested under ``entries``/``files``) and falls back to the
    ``sha256sum`` text form (``<64-hex>  <relpath>`` per line).
    """

    def _coerce_map(obj: dict[str, Any]) -> dict[str, str]:
        entries: dict[str, str] = {}
        for key, value in obj.items():
            if isinstance(value, str):
                entries[str(key)] = value
            elif isinstance(value, dict) and isinstance(
                value.get("sha256"), str
            ):
                entries[str(key)] = value["sha256"]
        return entries

    try:
        # R-067: route through the hardened loader, never raw ``json.loads``.
        obj = schema.parse_json_bytes_strict(raw)
    except (UnicodeDecodeError, json.JSONDecodeError, schema.TypedIngressError):
        obj = None
    if isinstance(obj, dict):
        for nested_key in ("entries", "files"):
            nested = obj.get(nested_key)
            if isinstance(nested, dict):
                entries = _coerce_map(nested)
                if entries:
                    return entries
        entries = _coerce_map(obj)
        if entries:
            return entries

    entries = {}
    for line in raw.decode("utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        match = _SHA256SUM_LINE.match(line)
        if match is None:
            continue
        entries[match.group(2).strip()] = match.group(1)
    if not entries:
        raise schema.ConfigSurfaceError(
            f"{rel}: FINAL_CHECKSUMS carries no parseable path->sha256"
            " entries (JSON map or sha256sum text)"
        )
    return entries


def build_d6_baseline(
    *,
    checksums_path: Path | None = None,
    main_tex_sha256: str | None = None,
    main_pdf_sha256: str | None = None,
    final_checksums_sha256: str | None = None,
    final_checksums_entries: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build the PARAMETERIZED D6 manuscript baseline block.

    ``main.tex``/``main.pdf`` default to the pinned two-party ``closure``
    constants (never re-hardcoded) and are overridable. The COMPLETE
    FINAL_CHECKSUMS closure is bound from ``checksums_path`` (its file digest +
    parsed entries) and/or explicit overrides; absent it, the block omits the
    closure hash so ``evaluate_closure`` fails closed (the final
    post-de-anonymization checksums are unknown until the ceremony).
    """
    baseline: dict[str, Any] = {
        "main_tex_sha256": main_tex_sha256 or closure.D6_MAIN_TEX_SHA256,
        "main_pdf_sha256": main_pdf_sha256 or closure.D6_MAIN_PDF_SHA256,
    }
    if checksums_path is not None:
        path = Path(checksums_path)
        rel = path.name
        try:
            raw = path.read_bytes()
        except OSError as exc:
            raise schema.TypedIngressError(
                f"{rel}: FINAL_CHECKSUMS file is missing or unreadable"
                f" ({exc.__class__.__name__})"
            ) from exc
        baseline["final_checksums_sha256"] = hashlib.sha256(raw).hexdigest()
        baseline["final_checksums_entries"] = _parse_final_checksums(raw, rel)
    if final_checksums_sha256 is not None:
        baseline["final_checksums_sha256"] = final_checksums_sha256
    if final_checksums_entries is not None:
        baseline["final_checksums_entries"] = dict(final_checksums_entries)
    return baseline


def build_qa012_block(
    *,
    roots: list[Path] | None = None,
    status: str | None = None,
    inventory_sha256: str | None = None,
) -> dict[str, Any]:
    """Build the closure ``qa012`` block from an executed detector manifest or
    an explicit status/inventory-hash pair (R-072).

    ``roots`` runs ``qa012.build_inventory_manifest`` over the given corpora and
    maps the zero-hit/hits result to the closure status. An explicit
    ``status``+``inventory_sha256`` short-circuits the scan (for a manifest
    executed elsewhere). One of the two inputs is required — a QA-012 block
    fabricated from nothing is refused.
    """
    if status is not None or inventory_sha256 is not None:
        if status is None or not schema.is_sha256_hex(inventory_sha256):
            raise schema.ConfigSurfaceError(
                "explicit QA-012 requires both a status and a sha256"
                " inventory hash (R-072)"
            )
        return {"status": status, "inventory_sha256": inventory_sha256}
    if roots:
        manifest = qa012.build_inventory_manifest([Path(r) for r in roots])
        return assembler.qa012_status_block(manifest)
    raise schema.ConfigSurfaceError(
        "QA-012 block requires either a corpus root to scan or an explicit"
        " status + inventory sha256 (R-072)"
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_driver(
    records_root: Path,
    out_dir: Path,
    frozen_dir: Path,
    *,
    source_commit: str,
    d6_checksums: Path | None = None,
    d6_main_tex_sha256: str | None = None,
    d6_main_pdf_sha256: str | None = None,
    qa012_roots: list[Path] | None = None,
    qa012_status: str | None = None,
    qa012_inventory_sha256: str | None = None,
    run_id: str = "run-0001",
    reclaim_crashed_relic: bool = False,
) -> assembler.BuildResult:
    """Read the frozen inputs + records, construct the real identity blocks,
    and build/create-once-publish the D7(b) evidence package.

    Ordering fixes the exit-code taxonomy: record ingress first (typed ingress
    refusals -> exit 3), then the frozen loaders (ingress -> 3), then the QA-012
    surface (config -> 2), then the create-once publish.
    """
    records_root = Path(records_root)
    complete_by_cell = assembler.load_complete_by_cell(records_root)
    horizon_identity = record_horizon_identity(complete_by_cell)
    keyset_digest = record_keyset_digest(complete_by_cell)

    arms = build_arms()
    llm_involvement = build_llm_involvement()
    grid = build_grid_block(
        keyset_digest=keyset_digest, horizon_identity=horizon_identity
    )
    estimands = build_estimands(horizon_identity=horizon_identity)
    provenance = build_provenance_from_frozen(
        frozen_dir, keyset_digest=keyset_digest, source_commit=source_commit
    )
    d6_baseline = build_d6_baseline(
        checksums_path=d6_checksums,
        main_tex_sha256=d6_main_tex_sha256,
        main_pdf_sha256=d6_main_pdf_sha256,
    )
    qa012_block = build_qa012_block(
        roots=qa012_roots,
        status=qa012_status,
        inventory_sha256=qa012_inventory_sha256,
    )

    return assembler.build_evidence_package(
        records_root,
        out_dir,
        source_commit=source_commit,
        arms=arms,
        provenance=provenance,
        grid=grid,
        llm_involvement=llm_involvement,
        estimands=estimands,
        d6_baseline=d6_baseline,
        qa012=qa012_block,
        run_id=run_id,
        reclaim_crashed_relic=reclaim_crashed_relic,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m reproducibility.colm_aims_2026.phase4_driver_d7b",
        description=(
            "Construct the real D7(b) identity blocks from the frozen inputs"
            " + a records root and build/publish the evidence package"
            " (source-only)."
        ),
        allow_abbrev=False,
    )
    parser.add_argument("--records-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--frozen-dir", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--d6-checksums", default=None)
    parser.add_argument("--d6-main-tex-sha256", default=None)
    parser.add_argument("--d6-main-pdf-sha256", default=None)
    parser.add_argument(
        "--qa012-root", action="append", default=None, dest="qa012_roots"
    )
    parser.add_argument("--qa012-status", default=None)
    parser.add_argument("--qa012-inventory-sha256", default=None)
    parser.add_argument("--run-id", default="run-0001")
    parser.add_argument(
        "--reclaim-crashed-relic", action="store_true", default=False
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; returns the pinned exit code (0/2/3/4)."""
    parser = _build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:  # argparse usage error -> pinned exit 2
        code = exc.code
        return code if isinstance(code, int) else EXIT_USAGE_ERROR

    try:
        result = run_driver(
            Path(args.records_root),
            Path(args.out_dir),
            Path(args.frozen_dir),
            source_commit=args.source_commit,
            d6_checksums=(
                Path(args.d6_checksums) if args.d6_checksums else None
            ),
            d6_main_tex_sha256=args.d6_main_tex_sha256,
            d6_main_pdf_sha256=args.d6_main_pdf_sha256,
            qa012_roots=args.qa012_roots,
            qa012_status=args.qa012_status,
            qa012_inventory_sha256=args.qa012_inventory_sha256,
            run_id=args.run_id,
            reclaim_crashed_relic=args.reclaim_crashed_relic,
        )
    except (schema.TypedIngressError, schema.EmptyEvaluationError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_INGRESS_ERROR
    except (schema.ConfigSurfaceError, schema.ColmAimsError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE_ERROR
    except Exception as exc:  # noqa: BLE001 - pinned internal-error code
        print(
            f"error: unexpected {exc.__class__.__name__} during driver run",
            file=sys.stderr,
        )
        return EXIT_INTERNAL_ERROR

    inventory_result = closure.evaluate_closure(result.closure_inventory)
    closure_state = (
        "SATISFIED" if inventory_result["satisfied"] else "UNSATISFIED"
    )
    print(
        f"[driver] published {result.published_tree.name}: closure"
        f" {closure_state} ({len(inventory_result['failing_rows'])}"
        " failing rows)"
    )
    return EXIT_PASS


if __name__ == "__main__":  # pragma: no cover - exercised via subprocess
    raise SystemExit(main())
