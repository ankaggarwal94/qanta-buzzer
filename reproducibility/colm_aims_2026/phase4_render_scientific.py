"""Deterministically render the release-verified Phase-4 scientific table."""
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import io
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scripts.stopdff_v5 import fileio

from . import schema, verifier
from .phase4_finalize_release import (
    EXIT_INGRESS_ERROR,
    EXIT_INTERNAL_ERROR,
    EXIT_PASS,
    EXIT_USAGE_ERROR,
    EXIT_VERIFY_FAIL,
    ReleaseVerificationFailed,
    _capture_directory_chain,
    _canonical_existing_directory,
    _parse_object,
    _publish_verified_directory,
    _read_accepted_directory_snapshot,
    _remove_exact_staged_directory,
    _require_accepted_directory,
    _require_disjoint,
    _require_mutually_disjoint,
    _require_unchanged_directory,
    _require_unclaimed,
    _require_portable_id,
    _require_report_bindings,
)


JSON_NAME = "scientific_results.json"
CSV_NAME = "scientific_results.csv"
TEX_NAME = "scientific_results_table.tex"
OUTPUT_NAMES = (JSON_NAME, CSV_NAME, TEX_NAME)

KHARD_CAVEAT_ID = "same_space_nearest_neighbor_selection_circularity"
KHARD_CAVEAT_TEXT = (
    "khard distractors are nearest neighbors selected in the same"
    " representation space used to score them; this same-space selection is"
    " circular. khard is a constructed QA reference, not observed open-ended"
    " response evidence."
)
KRANDOM_DISCLOSURE_TEXT = (
    "Random-K is retained in the ten-cell Holm family for complete disclosure"
    " but is non-headline and historical-nonconfirmatory."
)
QUALIFIER_TEXT = (
    "Constructed QA reference sensitivity only; no observed open-ended"
    " stopping policy was measured."
)
DIRECTION_TEXT = (
    "Mean signed shift equals MC stop index minus constructed QA reference"
    " stop index; positive values mean the constructed QA reference stops"
    " earlier."
)


@dataclass(frozen=True)
class ScientificRenderResult:
    """The create-once scientific output bundle."""

    published_dir: Path
    json_path: Path
    csv_path: Path
    tex_path: Path
    report: verifier.VerificationReport


def _external_reference(base: Path, rel: Any, label: str) -> Path:
    if not isinstance(rel, str) or not rel or Path(rel).is_absolute():
        raise schema.ConfigSurfaceError(f"{label} must be a relative path")
    candidate = base / rel
    if not schema.resolves_inside(candidate, base):
        raise schema.ConfigSurfaceError(f"{label} escapes the expectations base")
    return candidate


def _capture_release_inputs(
    runs_root: Path, expectations_path: Path
) -> tuple[Path, dict[str, bytes]]:
    """Capture the three release sidecars and canonical tree without aliases."""
    expectations_path = Path(os.path.abspath(expectations_path))
    base, accepted_snapshot = _read_accepted_directory_snapshot(
        expectations_path.parent,
        "expectations parent",
        expected_names=("ledger.json", "rights.json", "expectations.json"),
    )
    _require_disjoint(
        expectations_path,
        runs_root,
        "expectations must remain outside the runs root",
    )
    try:
        expectations_rel = expectations_path.relative_to(base).as_posix()
        expectations_bytes = accepted_snapshot[expectations_rel]
    except (KeyError, ValueError) as exc:
        raise schema.TypedIngressError(
            "expectations are absent from the accepted authority snapshot"
        ) from exc
    expectations = _parse_object(expectations_bytes, expectations_path.name)
    anchor = expectations.get("anchor")
    rights_decl = expectations.get("rights_inventory")
    if not isinstance(anchor, dict) or not isinstance(rights_decl, dict):
        raise schema.ConfigSurfaceError(
            "expectations must carry anchor and rights_inventory objects"
        )
    ledger_path = _external_reference(base, anchor.get("ledger_path"), "ledger path")
    rights_path = _external_reference(
        base, rights_decl.get("path"), "rights inventory path"
    )
    try:
        ledger_bytes = accepted_snapshot[ledger_path.relative_to(base).as_posix()]
        rights_bytes = accepted_snapshot[rights_path.relative_to(base).as_posix()]
    except (KeyError, ValueError) as exc:
        raise schema.TypedIngressError(
            "ledger or rights bytes are absent from the accepted authority"
            " snapshot"
        ) from exc
    ledger_doc = _parse_object(ledger_bytes, ledger_path.name)
    run_dir = verifier.resolve_canonical_package(runs_root, ledger_doc)
    return run_dir / "tree", {
        "expectations": expectations_bytes,
        "ledger": ledger_bytes,
        "rights": rights_bytes,
    }


def _scientific_rows(profile: dict[str, Any]) -> list[dict[str, Any]]:
    arms = {
        arm.get("arm_id"): arm
        for arm in profile.get("arms", [])
        if isinstance(arm, dict)
    }
    random_arm = arms.get("krandom")
    if not isinstance(random_arm, dict) or random_arm.get(
        "reporting_eligibility"
    ) != "non_headline_disclosure_only":
        raise schema.SchemaValidationError(
            "verified profile does not preserve Random-K non-headline eligibility"
        )
    if "khard" not in arms:
        raise schema.SchemaValidationError(
            "verified profile does not contain the khard reference"
        )

    cells = {
        cell.get("cell_id"): cell
        for cell in profile.get("cells", [])
        if isinstance(cell, dict)
    }
    rows: list[dict[str, Any]] = []
    for cell_id in schema.CELL_IDS:
        cell = cells[cell_id]
        reference_id = cell["reference_id"]
        eligibility = arms[reference_id]["reporting_eligibility"]
        caveat_id = KHARD_CAVEAT_ID if reference_id == "khard" else None
        rows.append(
            {
                "cell_id": cell_id,
                "reference_id": reference_id,
                "calibration_id": cell["calibration_id"],
                "reporting_eligibility": eligibility,
                "headline_eligible": eligibility == "headline_eligible",
                "caveat_id": caveat_id,
                "n": cell["headline_summary"]["n"],
                "mean_signed_shift": cell["headline_summary"][
                    "mean_signed_shift"
                ],
                "ci_lower": cell["interval"]["ci"][0],
                "ci_upper": cell["interval"]["ci"][1],
                "raw_p_value": cell["raw_p_value"],
                "holm_rank": cell["holm_rank"],
                "holm_adjusted_p_value": cell["holm_adjusted_p_value"],
                "holm_rejected": cell["holm_rejected"],
            }
        )
    return rows


def _machine_bytes(
    profile: dict[str, Any],
    profile_bytes: bytes,
    *,
    sidecars: dict[str, bytes],
    tree_snapshot: dict[str, bytes],
    verifier_bindings: dict[str, str],
) -> bytes:
    tree_shas = {
        rel: hashlib.sha256(data).hexdigest()
        for rel, data in tree_snapshot.items()
    }
    tree_digest = verifier._tree_digest_from_shas(tree_shas)
    if verifier_bindings["input_tree_sha256"] != tree_digest:
        raise schema.TypedIngressError(
            "verifier report tree binding differs from the captured render input"
        )
    payload = {
        "schema_version": schema.SCHEMA_VERSION,
        "artifact_type": "colm_aims_2026_scientific_results",
        "analysis_provenance": profile["inference"]["analysis_provenance"],
        "source_profile_sha256": hashlib.sha256(profile_bytes).hexdigest(),
        "release_bindings": {
            "verdict": verifier.VERDICT_RELEASE_PASS,
            "input_tree_sha256": tree_digest,
            "expectations_sha256": hashlib.sha256(
                sidecars["expectations"]
            ).hexdigest(),
            "ledger_sha256": hashlib.sha256(sidecars["ledger"]).hexdigest(),
            "rights_sha256": hashlib.sha256(sidecars["rights"]).hexdigest(),
            "expectations_anchor_sha256": verifier_bindings[
                "expectations_anchor_sha256"
            ],
            "verifier_revision": verifier_bindings["verifier_revision"],
            "verifier_code_sha256": verifier_bindings[
                "verifier_code_sha256"
            ],
        },
        "semantic": dict(profile["semantic"]),
        "qualifier": QUALIFIER_TEXT,
        "direction": DIRECTION_TEXT,
        "display_rounding": {
            "json": "unrounded_binary64_round_trip",
            "csv": "unrounded_binary64_round_trip",
            "tex": "four_decimal_places_display_only",
        },
        "reference_disclosures": {
            "khard": {
                "caveat_id": KHARD_CAVEAT_ID,
                "text": KHARD_CAVEAT_TEXT,
            },
            "krandom": {
                "reporting_eligibility": "non_headline_disclosure_only",
                "headline_eligible": False,
                "text": KRANDOM_DISCLOSURE_TEXT,
            },
        },
        "rejected_cell_ids": list(profile["inference"]["rejected_cell_ids"]),
        "cells": _scientific_rows(profile),
    }
    return schema.encode_json(payload)


def _csv_bytes(
    rows: list[dict[str, Any]], release_identity: dict[str, str]
) -> bytes:
    columns = (
        "source_profile_sha256",
        "input_tree_sha256",
        "expectations_anchor_sha256",
        "verifier_revision",
        "verifier_code_sha256",
        "qualifier",
        "direction",
        "cell_id",
        "reference_id",
        "calibration_id",
        "reporting_eligibility",
        "headline_eligible",
        "caveat_id",
        "caveat_text",
        "reference_disclosure_text",
        "n",
        "mean_signed_shift",
        "ci_lower",
        "ci_upper",
        "raw_p_value",
        "holm_rank",
        "holm_adjusted_p_value",
        "holm_rejected",
    )
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(
        {
            **release_identity,
            "qualifier": QUALIFIER_TEXT,
            "direction": DIRECTION_TEXT,
            **row,
            "caveat_text": (
                KHARD_CAVEAT_TEXT if row["reference_id"] == "khard" else ""
            ),
            "reference_disclosure_text": (
                KRANDOM_DISCLOSURE_TEXT
                if row["reference_id"] == "krandom"
                else KHARD_CAVEAT_TEXT
                if row["reference_id"] == "khard"
                else ""
            ),
        }
        for row in rows
    )
    return stream.getvalue().encode("utf-8")


def _tex_escape(value: str) -> str:
    return value.replace("\\", r"\textbackslash{}").replace("_", r"\_")


def _tex_float(value: Any) -> str:
    return f"{float(value):.4f}"


def _tex_bytes(
    rows: list[dict[str, Any]], release_identity: dict[str, str]
) -> bytes:
    lines = [
        "% Deterministic display-only rendering; JSON/CSV retain round-trip values.",
        *(
            f"% {field}={release_identity[field]}"
            for field in (
                "source_profile_sha256",
                "input_tree_sha256",
                "expectations_anchor_sha256",
                "verifier_revision",
                "verifier_code_sha256",
            )
        ),
        r"\begin{tabular}{llrlllll}",
        r"Reference & Calibration & $n$ & Mean shift & 95\% CI & Raw $p$ & Holm $p$ & Rejected \\",
        r"\hline",
    ]
    for row in rows:
        reference = _tex_escape(str(row["reference_id"]))
        if row["reference_id"] == "krandom":
            reference += r"$^{\dagger}$"
        elif row["reference_id"] == "khard":
            reference += r"$^{\ddagger}$"
        calibration = _tex_escape(str(row["calibration_id"]))
        interval = (
            f"[{_tex_float(row['ci_lower'])}, {_tex_float(row['ci_upper'])}]"
        )
        rejected = "yes" if row["holm_rejected"] else "no"
        lines.append(
            " & ".join(
                (
                    reference,
                    calibration,
                    str(row["n"]),
                    _tex_float(row["mean_signed_shift"]),
                    interval,
                    _tex_float(row["raw_p_value"]),
                    _tex_float(row["holm_adjusted_p_value"]),
                    rejected,
                )
            )
            + r" \\"
        )
    lines.extend(
        (
            r"\hline",
            "\\multicolumn{8}{l}{\\footnotesize "
            + _tex_escape(QUALIFIER_TEXT)
            + r"} \\",
            "\\multicolumn{8}{l}{\\footnotesize Direction: "
            + _tex_escape(DIRECTION_TEXT)
            + r"} \\",
            r"\multicolumn{8}{l}{\footnotesize $^{\dagger}$ Random-K is in the"
            r" ten-cell Holm family but is non-headline and"
            r" historical-nonconfirmatory.} \\",
            "\\multicolumn{8}{l}{\\footnotesize $^{\\ddagger}$ "
            + _tex_escape(KHARD_CAVEAT_TEXT)
            + r"} \\",
            r"\end{tabular}",
            "",
        )
    )
    return "\n".join(lines).encode("utf-8")


def _read_outputs(
    directory: Path, *, require_acceptance: bool = True
) -> dict[str, bytes]:
    if require_acceptance:
        directory, snapshot = _read_accepted_directory_snapshot(
            directory,
            "scientific output",
            expected_names=OUTPUT_NAMES,
        )
    else:
        directory = _canonical_existing_directory(
            directory, "staged scientific output"
        )
        snapshot = verifier._read_tree_snapshot(directory)
    observed = set(snapshot)
    if observed != set(OUTPUT_NAMES):
        raise schema.TypedIngressError(
            "scientific output must contain exactly the JSON, CSV, and TeX"
            f" products; observed {sorted(observed)}"
        )
    return {name: snapshot[name] for name in OUTPUT_NAMES}


def _require_render_boundaries(
    destination: Path, receipts_dir: Path, authority_base: Path
) -> None:
    """Reassert every mutable-output/authority containment boundary."""
    _require_mutually_disjoint(
        destination,
        receipts_dir,
        "scientific output and receipt directory must be disjoint",
    )
    _require_mutually_disjoint(
        destination,
        authority_base,
        "scientific output destination and expectations authority base must"
        " be disjoint",
    )
    _require_mutually_disjoint(
        receipts_dir,
        authority_base,
        "scientific receipts and expectations authority base must be disjoint",
    )


def render_scientific_release(
    *,
    runs_root: Path,
    expectations: Path,
    output_root: Path,
    render_id: str,
    receipts_dir: Path,
) -> ScientificRenderResult:
    """Require PASS_RELEASE, then create-once render its verified profile."""
    _require_portable_id(render_id, "render_id")
    runs_root = _canonical_existing_directory(runs_root, "runs root")
    output_root = _canonical_existing_directory(output_root, "output root")
    receipts_dir = _canonical_existing_directory(receipts_dir, "receipts root")
    expectations_path = Path(os.path.abspath(expectations))
    authority_base = _require_accepted_directory(
        expectations_path.parent,
        "expectations authority base",
        expected_names=("ledger.json", "rights.json", "expectations.json"),
    )
    output_root_chain = _capture_directory_chain(output_root)
    receipts_chain = _capture_directory_chain(receipts_dir)
    authority_chain = _capture_directory_chain(authority_base)
    destination = output_root / render_id
    _require_unclaimed(destination, "scientific output bundle")
    _require_disjoint(
        destination,
        runs_root,
        "scientific output destination must be outside the runs root",
    )
    _require_disjoint(
        receipts_dir,
        runs_root,
        "scientific-render receipts must be outside the runs root",
    )
    _require_render_boundaries(destination, receipts_dir, authority_base)

    tree, sidecars_before = _capture_release_inputs(
        runs_root, expectations_path
    )
    tree_snapshot_before = verifier._read_tree_snapshot(tree)
    report = verifier.run_release_over_runs_root(
        runs_root,
        expectations=expectations_path,
        receipts_dir=receipts_dir,
    )
    if report.verdict != verifier.VERDICT_RELEASE_PASS:
        raise ReleaseVerificationFailed(
            "scientific rendering requires canonical PASS_RELEASE"
        )
    tree_after, sidecars_after = _capture_release_inputs(
        runs_root, expectations_path
    )
    tree_snapshot_after = verifier._read_tree_snapshot(tree_after)
    if (
        tree_after != tree
        or sidecars_after != sidecars_before
        or tree_snapshot_after != tree_snapshot_before
    ):
        raise schema.TypedIngressError(
            "release inputs changed during verification; refusing to render"
        )
    report_bindings = _require_report_bindings(
        report,
        tree_snapshot=tree_snapshot_before,
        expectations_bytes=sidecars_before["expectations"],
        receipts_dir=receipts_dir,
    )
    _require_unchanged_directory(
        output_root, output_root_chain, "scientific output root"
    )
    _require_unchanged_directory(
        receipts_dir, receipts_chain, "scientific receipts directory"
    )
    _require_unchanged_directory(
        authority_base, authority_chain, "expectations authority base"
    )
    _require_render_boundaries(destination, receipts_dir, authority_base)
    profile_bytes = tree_snapshot_before["profile.json"]
    profile = schema.load_artifact_bytes(profile_bytes, "profile.json")
    schema.validate_profile(profile)
    rows = _scientific_rows(profile)
    release_identity = {
        "source_profile_sha256": hashlib.sha256(profile_bytes).hexdigest(),
        "input_tree_sha256": report_bindings["input_tree_sha256"],
        "expectations_anchor_sha256": report_bindings[
            "expectations_anchor_sha256"
        ],
        "verifier_revision": report_bindings["verifier_revision"],
        "verifier_code_sha256": report_bindings["verifier_code_sha256"],
    }
    generated = {
        JSON_NAME: _machine_bytes(
            profile,
            profile_bytes,
            sidecars=sidecars_before,
            tree_snapshot=tree_snapshot_before,
            verifier_bindings=report_bindings,
        ),
        CSV_NAME: _csv_bytes(rows, release_identity),
        TEX_NAME: _tex_bytes(rows, release_identity),
    }

    staged = Path(tempfile.mkdtemp(prefix=".scientific-staged-", dir=output_root))
    staged_snapshot = _capture_directory_chain(staged)
    if (
        staged_snapshot.lexical[:-1] != output_root_chain.lexical
        or (
            output_root_chain.windows is not None
            and staged_snapshot.windows is not None
            and staged_snapshot.windows[:-1] != output_root_chain.windows
        )
        or (output_root_chain.windows is None) != (staged_snapshot.windows is None)
    ):
        raise schema.TypedIngressError(
            "staging directory is not a child of the captured output root"
        )
    try:
        for name in OUTPUT_NAMES:
            (staged / name).write_bytes(generated[name])
        if _read_outputs(staged, require_acceptance=False) != generated:
            raise schema.TypedIngressError(
                "staged scientific outputs differ from the deterministic bytes"
            )
        fileio.fsync_tree(staged)
        tree_final, sidecars_final = _capture_release_inputs(
            runs_root, expectations_path
        )
        if (
            tree_final != tree
            or sidecars_final != sidecars_before
            or verifier._read_tree_snapshot(tree_final) != tree_snapshot_before
        ):
            raise schema.TypedIngressError(
                "release inputs changed before publication; refusing to render"
            )
        _require_unchanged_directory(
            output_root, output_root_chain, "scientific output root"
        )
        _require_unchanged_directory(
            receipts_dir, receipts_chain, "scientific receipts directory"
        )
        _require_unchanged_directory(
            authority_base, authority_chain, "expectations authority base"
        )
        _require_render_boundaries(destination, receipts_dir, authority_base)
        _publish_verified_directory(
            staged,
            destination,
            exists_label="scientific output bundle",
            parent_chain=output_root_chain,
            expected_names=OUTPUT_NAMES,
        )
        # The final name is the terminal operation; avoid fallible I/O after
        # the complete-looking public directory exists.
        staged = None
        return ScientificRenderResult(
            published_dir=destination,
            json_path=destination / JSON_NAME,
            csv_path=destination / CSV_NAME,
            tex_path=destination / TEX_NAME,
            report=report,
        )
    finally:
        if staged is not None:
            with contextlib.suppress(BaseException):
                _remove_exact_staged_directory(
                    parent=output_root,
                    parent_snapshot=output_root_chain,
                    staged_name=staged.name,
                    staged_snapshot=staged_snapshot,
                    expected_names=OUTPUT_NAMES,
                )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m reproducibility.colm_aims_2026.phase4_render_scientific",
        description=(
            "Render deterministic JSON, CSV, and TeX results from a canonical"
            " PASS_RELEASE Phase-4 profile."
        ),
        allow_abbrev=False,
    )
    parser.add_argument("--runs-root", required=True, type=Path)
    parser.add_argument("--expectations", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--render-id", required=True)
    parser.add_argument("--receipts-dir", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else EXIT_USAGE_ERROR
    try:
        result = render_scientific_release(
            runs_root=args.runs_root,
            expectations=args.expectations,
            output_root=args.output_root,
            render_id=args.render_id,
            receipts_dir=args.receipts_dir,
        )
    except ReleaseVerificationFailed as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_VERIFY_FAIL
    except schema.TypedIngressError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_INGRESS_ERROR
    except schema.ColmAimsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE_ERROR
    except Exception as exc:  # noqa: BLE001 - pinned CLI internal-error class
        print(
            f"error: unexpected {exc.__class__.__name__} during scientific render",
            file=sys.stderr,
        )
        return EXIT_INTERNAL_ERROR
    print(f"[scientific] rendered create-once bundle at {result.published_dir}")
    return EXIT_PASS


if __name__ == "__main__":  # pragma: no cover - subprocess tests own CLI
    raise SystemExit(main())
