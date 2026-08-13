"""Report / figure / package writers (see ACCEPTANCE_CONTRACT.md).

Markdown + LaTeX include every normative definition and the resource/cost block; figures
are valid PNGs with positive dimensions; the package carries a safe complete SHA256SUMS
and an external_artifacts.json retrieval index.
"""
from __future__ import annotations

import json
import hashlib
import math
import struct
import tempfile
import zlib
from pathlib import Path
from typing import Any

from . import PROFILE_NAME, PROTOCOL_VERSION
from .fileio import publish_bytes
from .identity import build_manifest, compute_id, loads_no_duplicate_keys, sha256_file
from .profile import CALIBRATION, REWARD_SCHEDULES
from .receipt_evidence import (
    DETERMINISM_BINDINGS,
    FULL_RECEIPT_BINDINGS,
    build_prerequisite_evidence,
    prerequisite_evidence_sha256,
    validate_prerequisite_evidence,
    validate_prerequisite_receipts,
    validate_receipt_evidence_digest,
    verify_prerequisite_evidence_bytes,
)
from .rewards import REWARD_SCHEDULE_STRINGS
from .verdicts import MATERIAL_THRESHOLD

_RECEIPT_GATES = {"smoke", "mutation", "determinism"}
_FULL_RECEIPT_BINDINGS = FULL_RECEIPT_BINDINGS
_DETERMINISM_BINDINGS = DETERMINISM_BINDINGS


def build_prerequisite_receipt(
    *,
    gate: str,
    bindings: dict[str, str],
    evidence: dict[str, Any],
) -> dict[str, Any]:
    """Build a content-addressed successful prerequisite receipt."""
    if gate not in _RECEIPT_GATES:
        raise ValueError(f"unknown prerequisite gate {gate!r}")
    required = (
        _DETERMINISM_BINDINGS
        if gate == "determinism"
        else _FULL_RECEIPT_BINDINGS
    )
    if set(bindings) != required:
        raise ValueError(f"{gate} receipt bindings mismatch")
    validate_receipt_evidence_digest(gate, evidence)
    return build_manifest({
        "kind": "prerequisite_receipt",
        "gate": gate,
        "status": "successful",
        "bindings": {key: bindings[key] for key in sorted(bindings)},
        "evidence": evidence,
    })


def build_evidenced_prerequisite_receipt(
    *,
    gate: str,
    bindings: dict[str, str],
    evidence: dict[str, Any],
) -> dict[str, Any]:
    """Issue a receipt only after validating and hashing its full evidence object."""
    validate_prerequisite_evidence(
        gate=gate,
        bindings=bindings,
        evidence=evidence,
    )
    return build_prerequisite_receipt(
        gate=gate,
        bindings=bindings,
        evidence={"evidence_sha256": prerequisite_evidence_sha256(evidence)},
    )


def _gate_override_active(aggregate: dict[str, Any]) -> bool:
    ov = aggregate.get("gate_overrides") or {}
    return bool(
        ov.get("allow_low_mc_retention") or ov.get("allow_incomplete_mc_coverage")
    )


def _rendered_cell_verdict(cell: dict[str, Any], *, override_active: bool) -> str:
    """Render a per-cell verdict with the qualifier reasons the JSON carries.

    A bare WARN is not actionable; the reader must see whether it came from a
    ceiling flag, dirty coverage, an active MC gate override, or a CI upper
    bound above the material threshold (mirrors verdicts.cell_verdict).
    """
    verdict = str(cell.get("verdict"))
    if cell.get("status") != "completed":
        return verdict
    reasons: list[str] = []
    if cell.get("ceiling_any"):
        reasons.append("ceiling")
    if cell.get("coverage_clean") is False:
        reasons.append("coverage")
    if override_active:
        reasons.append("override")
    ci = cell.get("abs_median_ci")
    if (
        verdict == "WARN"
        and isinstance(ci, (list, tuple))
        and len(ci) == 2
        and not isinstance(ci[1], bool)
        and isinstance(ci[1], (int, float))
        and float(ci[1]) > MATERIAL_THRESHOLD
    ):
        reasons.append("ci_above_threshold")
    if reasons:
        return f"{verdict} ({', '.join(reasons)})"
    return verdict


def render_markdown(aggregate: dict[str, Any], *, resource_summary: dict[str, Any]) -> str:
    fam = aggregate.get("family") or {}
    any_override = _gate_override_active(aggregate)
    lines: list[str] = []
    lines.append(f"# StopDFF bucketed-DP paired audit ({PROFILE_NAME}, protocol {PROTOCOL_VERSION})")
    lines.append("")
    lines.append(f"- Profile name / version: `{PROFILE_NAME}` schema 2")
    lines.append(f"- Profile variant: {aggregate.get('profile_variant')}")
    lines.append(f"- Backend: {aggregate.get('backend')}")
    lines.append("")
    lines.append("## Paired-format definition")
    lines.append("paired_qa_prefix_vs_mc_fixed: each item contributes a QA prefix trajectory and an "
                 "MC fixed-option trajectory; the signed index metric is tau_MC - tau_QA.")
    lines.append("")
    lines.append("## Reward table")
    lines.append("| schedule | correct_early | correct_late | wrong | split | wait_cost |")
    lines.append("|---|---|---|---|---|---|")
    for name in REWARD_SCHEDULES:
        s = REWARD_SCHEDULE_STRINGS[name]
        lines.append(f"| {name} | {s['correct_early']} | {s['correct_late']} | {s['wrong']} | {s['split']} | {s['wait_cost']} |")
    lines.append("")
    lines.append("## Calibrator definition")
    lines.append("Calibrators (platt-logistic, similarity-temperature, isotonic) are fit on "
                 "validation MC rows only and the shared phase map is applied to MC and QA. "
                 f"Phase boundaries: {CALIBRATION['phase_boundaries']}.")
    lines.append("")
    lines.append("## Continuation definition")
    lines.append("Empirical-bucket / pooled-empirical continuation with the canonical fallback "
                 "ladders and coverage tags (primary/fallback/missing). FVI is damped (0.5), "
                 "float64, with two-consecutive-iteration convergence and cycle detection.")
    lines.append("")
    lines.append("## FVI settings")
    sel = aggregate.get("fvi_selected", {})
    lines.append(f"- damping: 0.5")
    lines.append(f"- tolerance: {sel.get('tolerance')}")
    lines.append(f"- max_iterations: {sel.get('max_iterations')}")
    lines.append("")
    lines.append("## Cell counts")
    lines.append(f"- requested: {aggregate.get('requested')}")
    lines.append(f"- completed: {aggregate.get('completed')}")
    lines.append(f"- skipped: {aggregate.get('skipped')}")
    lines.append(f"- failed: {aggregate.get('failed')}")
    counts = {"PASS": 0, "WARN": 0, "FAIL": 0}
    for c in aggregate.get("cells", {}).values():
        v = c.get("verdict")
        if v in counts:
            counts[v] += 1
    lines.append(f"- cell verdicts: PASS={counts['PASS']} WARN={counts['WARN']} FAIL={counts['FAIL']}")
    for key, c in sorted(aggregate.get("cells", {}).items()):
        if c.get("verdict") in ("WARN", "FAIL"):
            lines.append(
                f"- {key}: "
                f"{_rendered_cell_verdict(c, override_active=any_override)}"
            )
    lines.append("")
    lines.append("## Family maximum statistic and CI")
    lines.append(f"- family statistic M (max cell median |index shift|): {fam.get('M')}")
    lines.append(f"- family 95% CI: {fam.get('ci')}")
    lines.append(f"- family verdict: {fam.get('verdict')}")
    lines.append("")
    lines.append("## MC gate evidence and overrides")
    ov = aggregate.get("gate_overrides", {})
    lines.append(f"- allow_low_mc_retention: {ov.get('allow_low_mc_retention')}")
    lines.append(f"- allow_incomplete_mc_coverage: {ov.get('allow_incomplete_mc_coverage')}")
    if any_override:
        lines.append("- NOTE: an override is active; family PASS is prevented (retained MC subset).")
    lines.append("")
    lines.append("## Never-buzz asymmetry")
    lines.append("Per-cell never_buzz_MC and never_buzz_QA are preserved in each cell record.")
    lines.append("")
    lines.append("## Release validity")
    lines.append(f"- release_status: {aggregate.get('release_status')}")
    lines.append("")
    lines.append("## Resource and cost summary")
    lines.append(f"```json\n{json.dumps(resource_summary, indent=2, sort_keys=True)}\n```")
    lines.append("")
    return "\n".join(lines)


def render_latex(aggregate: dict[str, Any]) -> str:
    fam = aggregate.get("family") or {}
    any_override = _gate_override_active(aggregate)
    rows = []
    for key, c in sorted(aggregate.get("cells", {}).items()):
        verdict = _rendered_cell_verdict(c, override_active=any_override)
        rows.append(f"{key.replace('_', ' ')} & {verdict} \\\\")
    body = "\n".join(rows) if rows else "none & none \\\\"
    return (
        "% StopDFF bucketed-DP paired audit table\n"
        "\\begin{table}[h]\n\\centering\n\\begin{tabular}{ll}\n\\hline\n"
        "cell & verdict \\\\\n\\hline\n"
        f"{body}\n\\hline\n\\end{{tabular}}\n"
        f"\\caption{{Family statistic M={fam.get('M')} CI={fam.get('ci')} "
        f"verdict={fam.get('verdict')} profile={PROFILE_NAME}}}\n"
        "\\end{table}\n"
    )


def _png_chunk(tag: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)


def write_min_png(path: Path, width: int = 16, height: int = 16, rgb=(40, 80, 160)) -> None:
    """Write a valid solid-color PNG (used when matplotlib is unavailable)."""
    raw = bytearray()
    row = b"\x00" + bytes(rgb) * width
    for _ in range(height):
        raw += row
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    png = b"\x89PNG\r\n\x1a\n" + _png_chunk(b"IHDR", ihdr) + _png_chunk(b"IDAT", zlib.compress(bytes(raw), 9)) + _png_chunk(b"IEND", b"")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png)


def _stored_zlib_stream(data: bytes) -> bytes:
    """Return a deterministic zlib stream made only of stored DEFLATE blocks."""
    stream = bytearray(b"\x78\x01")
    offset = 0
    while offset < len(data):
        block = data[offset:offset + 65535]
        offset += len(block)
        stream.append(1 if offset == len(data) else 0)
        stream.extend(struct.pack("<H", len(block)))
        stream.extend(struct.pack("<H", 0xFFFF - len(block)))
        stream.extend(block)
    stream.extend(struct.pack(">I", zlib.adler32(data) & 0xFFFFFFFF))
    return bytes(stream)


def _canonical_figure_points(
    aggregate: dict[str, Any],
) -> list[tuple[str, float]]:
    cells = aggregate.get("cells")
    if not isinstance(cells, dict):
        raise ValueError("aggregate cells must be an object for figure rendering")
    points: list[tuple[str, float]] = []
    for cell_key, cell in sorted(cells.items()):
        if not isinstance(cell_key, str) or not isinstance(cell, dict):
            raise ValueError("aggregate cell summaries are invalid for rendering")
        if "abs_median_point" not in cell:
            continue
        value = cell["abs_median_point"]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise ValueError("figure points must be finite numeric values")
        points.append((cell_key, float(value)))
    return points


def _write_canonical_histogram(
    path: Path,
    points: list[tuple[str, float]],
) -> None:
    """Render the canonical histogram without fonts or platform libraries."""
    width, height = 600, 400
    left, right, top, bottom = 55, 580, 25, 350
    pixels = bytearray([0]) * (width * height)

    def fill(x0: int, y0: int, x1: int, y1: int, color: int) -> None:
        x0, x1 = max(0, x0), min(width, x1)
        y0, y1 = max(0, y0), min(height, y1)
        for y in range(y0, y1):
            start = y * width + x0
            pixels[start:start + max(0, x1 - x0)] = bytes([color]) * max(
                0, x1 - x0
            )

    values = [value for _key, value in points]
    bins = min(20, max(3, len(values)))
    low = min([0.0, 1.0, *values])
    high = max([0.0, 1.0, *values])
    if high == low:
        high = low + 1.0
    counts = [0] * bins
    for value in values:
        index = int((value - low) / (high - low) * bins)
        counts[min(bins - 1, max(0, index))] += 1
    maximum = max(counts, default=0)
    plot_width = right - left
    plot_height = bottom - top
    for index, count in enumerate(counts):
        x0 = left + index * plot_width // bins + 1
        x1 = left + (index + 1) * plot_width // bins - 1
        bar_height = 0 if maximum == 0 else count * (plot_height - 5) // maximum
        fill(x0, bottom - bar_height, max(x0 + 1, x1), bottom, 2)
    fill(left - 2, top, left, bottom + 2, 1)
    fill(left - 2, bottom, right, bottom + 2, 1)
    threshold_x = left + round((1.0 - low) / (high - low) * plot_width)
    for y in range(top, bottom, 10):
        fill(threshold_x, y, threshold_x + 2, min(y + 6, bottom), 3)

    raw = bytearray()
    for y in range(height):
        raw.append(0)
        raw.extend(pixels[y * width:(y + 1) * width])
    metadata = json.dumps(
        {
            "bins": bins,
            "kind": "cell_median_absolute_index_shift_histogram",
            "points": points,
            "schema_version": 1,
            "threshold": 1.0,
            "title": "StopDFF cell median absolute index shift",
            "x_label": "cell median |index shift|",
            "y_label": "count",
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 3, 0, 0, 0)
    palette = bytes((255, 255, 255, 30, 30, 30, 40, 80, 160, 220, 30, 30))
    png = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"PLTE", palette)
        + _png_chunk(b"tEXt", b"stopdff_v5\x00" + metadata)
        + _png_chunk(b"IDAT", _stored_zlib_stream(bytes(raw)))
        + _png_chunk(b"IEND", b"")
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png)


def write_figures(
    output_dir: Path,
    aggregate: dict[str, Any],
    *,
    profile_variant: str | None = None,
) -> list[str]:
    figs_dir = Path(output_dir) / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)
    relative = "figures/cell_median_index_shift.png"
    _write_canonical_histogram(
        Path(output_dir) / relative,
        _canonical_figure_points(aggregate),
    )
    return [relative]


# Every character str.splitlines() treats as a line boundary: a name carrying
# one would tear the line-oriented SHA256SUMS format at parse time, so the
# writer fails closed instead of emitting an inventory its checker rejects.
_CHECKSUM_LINE_BREAKS = "\n\r\v\f\x1c\x1d\x1e\x85\u2028\u2029"


def _checksum_line(digest: str, rel: str) -> str:
    """Format one SHA256SUMS entry, rejecting names the format cannot carry."""
    if any(ch in _CHECKSUM_LINE_BREAKS for ch in rel):
        raise ValueError(f"checksum path contains a line break: {rel!r}")
    return f"{digest}  {rel}"


def write_sha256sums(output_dir: Path) -> None:
    root = Path(output_dir)
    lines: list[str] = []
    for p in sorted(root.rglob("*")):
        if p.is_symlink() or not p.is_file():
            continue
        rel = p.relative_to(root).as_posix()
        if rel == "SHA256SUMS":
            continue
        lines.append(_checksum_line(sha256_file(p), rel))
    (root / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


# Package path policy (in-toto DISALLOW-unknown norm): the complete namespace
# a packaged run root may contain. Sweep publishes the run-level files and the
# two json-only evidence directories; package_run itself manages the three
# package roots and the two package-level files. Anything else — typically an
# orphaned ``tmpXXXXXX`` left by a hard kill between mkstemp and rename — is
# partial-failure residue that no validation lane audits, so both the packager
# (here) and the checker (checker._check_package_path_policy) reject it
# instead of byte-attesting it into SHA256SUMS.
RUN_LEVEL_FILES = frozenset({
    "run_spec.json",
    "bootstrap_plan.json",
    "environment.json",
    "resource_summary.json",
    "aggregate.json",
    "attempts.jsonl",
    "run_manifest.json",
    "command_manifest.json",
})
RUN_JSON_ONLY_DIRS = frozenset({"cells", "attempt_results"})
PACKAGE_MANAGED_ROOTS = frozenset({"reports", "figures", "evidence"})
PACKAGE_LEVEL_FILES = frozenset({"external_artifacts.json", "SHA256SUMS"})


def _packageable_path_violation(rel: str) -> str | None:
    """Return why a non-candidate regular file may not be packaged, if so."""
    parts = Path(rel).parts
    if len(parts) == 1:
        if parts[0] not in RUN_LEVEL_FILES:
            return f"unaudited run-level file cannot be packaged: {rel!r}"
        return None
    if parts[0] in RUN_JSON_ONLY_DIRS:
        if len(parts) != 2 or Path(parts[1]).suffix != ".json":
            return f"unaudited file in {parts[0]}/ cannot be packaged: {rel!r}"
        return None
    return f"unaudited file cannot be packaged: {rel!r}"


_MANIFEST_EVIDENCE_PATHS = {
    "source_manifest": "evidence/source_manifest.json",
    "raw_input_manifest": "evidence/raw_input_manifest.json",
    "model_snapshot_manifest": "evidence/model_snapshot_manifest.json",
    "fvi_study": "evidence/fvi_study.json",
    "environment_contract": "evidence/environment_contract.json",
}
_MANIFEST_KINDS = {
    "source_manifest": {"source_snapshot"},
    "raw_input_manifest": {"raw_input_bundle"},
    "model_snapshot_manifest": {"model_snapshot"},
    "fvi_study": {"fvi_study", "fvi_study_fixed"},
    "environment_contract": {"environment_contract"},
}

# External content is validated in its staged layout and copied into a
# normalized, self-contained package layout.  Keeping the manifest paths
# stable preserves the external-artifact ledger contract; these subtrees bind
# every byte named by those manifests.
_BOUND_CONTENT_LAYOUTS = {
    "source_manifest": {
        "kind": "source_snapshot",
        "file_key": "files",
        "name_key": "path",
        "staged_subdir": "source",
        "packaged_subdir": "source_snapshot/source",
    },
    "raw_input_manifest": {
        "kind": "raw_input_bundle",
        "file_key": "files",
        "name_key": "role",
        "staged_subdir": "",
        "packaged_subdir": "raw_inputs/raw",
    },
    "model_snapshot_manifest": {
        "kind": "model_snapshot",
        "file_key": "files",
        "name_key": "path",
        "staged_subdir": "snapshot",
        "packaged_subdir": "model_snapshot/snapshot",
    },
}


def _manifest_from_bytes(data: bytes, *, role: str) -> dict[str, Any]:
    try:
        manifest = loads_no_duplicate_keys(data.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ValueError(f"{role} evidence is not canonical JSON: {exc}") from exc
    identity = manifest.get("identity") if isinstance(manifest, dict) else None
    if (
        not isinstance(identity, dict)
        or compute_id(identity) != manifest.get("id")
        or identity.get("kind") not in _MANIFEST_KINDS[role]
    ):
        raise ValueError(f"{role} evidence has an invalid manifest identity")
    content_kinds = {
        "source_manifest": "source_snapshot",
        "raw_input_manifest": "raw_input_bundle",
        "model_snapshot_manifest": "model_snapshot",
    }
    expected_kind = content_kinds.get(role)
    if expected_kind is not None:
        from .content_manifest import validate_content_manifest_document

        validate_content_manifest_document(
            manifest,
            manifest_name=role,
            expected_id=manifest["id"],
            expected_kind=expected_kind,
            require_semantic_pass=role == "raw_input_manifest",
        )
    return manifest


def _resolve_retrieval_path(root: Path, retrieval: str) -> Path:
    path = Path(retrieval)
    if path.is_absolute():
        candidates = (path,)
    else:
        if ".." in path.parts:
            raise ValueError(f"unsafe external-artifact retrieval path: {retrieval!r}")
        candidates = (root / path, root.parents[1] / path)
    for candidate in candidates:
        if candidate.is_symlink():
            raise ValueError(f"external-artifact retrieval path is a symlink: {candidate}")
        if candidate.is_file():
            return candidate
    raise ValueError(f"external-artifact retrieval path is missing: {retrieval!r}")


def _prepare_package_evidence(
    root: Path,
    *,
    external_artifacts: list[dict[str, Any]],
    evidence_files: dict[str, bytes],
) -> tuple[list[dict[str, Any]], dict[str, bytes], str | None]:
    # Byte-verified packaging needs the run spec; a root without one cannot
    # be verified, so it fails closed instead of taking a lenient path.
    run_spec_path = root / "run_spec.json"
    if run_spec_path.is_symlink() or not run_spec_path.is_file():
        raise ValueError("package requires run_spec.json before packaging")

    candidates: dict[str, bytes] = {}
    for name, data in evidence_files.items():
        path = Path(name)
        if (
            path.is_absolute()
            or ".." in path.parts
            or not path.parts
            or path.parts[0] != "evidence"
            or not isinstance(data, bytes)
        ):
            raise ValueError(f"unsafe packaged evidence path: {name!r}")
        candidates[path.as_posix()] = data

    normalized: dict[str, dict[str, Any]] = {}
    for artifact in external_artifacts:
        if not isinstance(artifact, dict):
            raise ValueError("invalid external-artifact ledger entry")
        if set(artifact) != {
            "role",
            "content_id",
            "sha256",
            "byte_size",
            "retrieval_path",
        }:
            raise ValueError("invalid external-artifact ledger entry")
        role = artifact.get("role")
        content_id = artifact.get("content_id")
        digest = artifact.get("sha256")
        byte_size = artifact.get("byte_size")
        retrieval = artifact.get("retrieval_path")
        if (
            role not in _MANIFEST_EVIDENCE_PATHS
            or role in normalized
            or not isinstance(content_id, str)
            or len(content_id) != 64
            or any(ch not in "0123456789abcdef" for ch in content_id)
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(ch not in "0123456789abcdef" for ch in digest)
            or isinstance(byte_size, bool)
            or not isinstance(byte_size, int)
            or byte_size <= 0
            or not isinstance(retrieval, str)
            or not retrieval
        ):
            raise ValueError("invalid external-artifact ledger entry")

        packaged_path = _MANIFEST_EVIDENCE_PATHS[role]
        if role in {
            "source_manifest",
            "raw_input_manifest",
            "model_snapshot_manifest",
        }:
            manifest_path = _resolve_retrieval_path(root, retrieval)
            data = manifest_path.read_bytes()
            candidates[packaged_path] = data
        else:
            data = candidates.get(packaged_path)
            if data is None:
                raise ValueError(f"missing packaged evidence for {role}")
        manifest = _manifest_from_bytes(data, role=role)
        content_layout = _BOUND_CONTENT_LAYOUTS.get(role)
        if content_layout is not None:
            from .content_manifest import validate_bound_content_manifest

            staged_subdir = content_layout["staged_subdir"]
            packaged_subdir = content_layout["packaged_subdir"]
            if (manifest_path.parent / packaged_subdir).is_dir():
                staged_subdir = packaged_subdir
            # Local staging keeps raw roles under ``raw/`` beside its operator
            # record; the Modal Volume stores the same canonical role set next
            # to the manifest.  Normalize both verified layouts into one
            # packaged subtree.
            elif role == "raw_input_manifest" and (
                manifest_path.parent / "raw"
            ).is_dir():
                staged_subdir = "raw"
            validate_bound_content_manifest(
                manifest_path.parent,
                manifest_name=manifest_path.name,
                expected_id=manifest["id"],
                expected_kind=content_layout["kind"],
                file_key=content_layout["file_key"],
                name_key=content_layout["name_key"],
                content_subdir=staged_subdir,
                require_semantic_pass=role == "raw_input_manifest",
            )
            staged_root = manifest_path.parent
            if staged_subdir:
                staged_root /= staged_subdir
            for entry in manifest["identity"][content_layout["file_key"]]:
                relative = entry[content_layout["name_key"]]
                source = staged_root / relative
                content = source.read_bytes()
                if (
                    len(content) != entry["size"]
                    or hashlib.sha256(content).hexdigest() != entry["sha256"]
                ):
                    raise ValueError(
                        f"{role} content changed during packaging: {relative}"
                    )
                destination = Path("evidence") / content_layout[
                    "packaged_subdir"
                ] / relative
                packaged_name = destination.as_posix()
                if (
                    packaged_name in candidates
                    and candidates[packaged_name] != content
                ):
                    raise ValueError(
                        f"conflicting packaged evidence path: {packaged_name}"
                    )
                candidates[packaged_name] = content
        actual_digest = hashlib.sha256(data).hexdigest()
        if (
            manifest["id"] != content_id
            or actual_digest != digest
            or len(data) != byte_size
        ):
            raise ValueError(f"{role} ledger entry does not match evidence bytes")
        normalized[role] = {
            "role": role,
            "content_id": manifest["id"],
            "sha256": actual_digest,
            "byte_size": len(data),
            "retrieval_path": packaged_path,
        }

    if set(normalized) != set(_MANIFEST_EVIDENCE_PATHS):
        raise ValueError("external-artifact ledger roles are incomplete")

    run_spec_manifest = loads_no_duplicate_keys(
        run_spec_path.read_text(encoding="utf-8")
    )
    run_spec_identity = (
        run_spec_manifest.get("identity")
        if isinstance(run_spec_manifest, dict)
        else None
    )
    if (
        not isinstance(run_spec_identity, dict)
        or compute_id(run_spec_identity) != run_spec_manifest.get("id")
    ):
        raise ValueError("package run_spec.json has an invalid manifest identity")
    profile_variant = run_spec_identity.get("profile_variant")
    receipt_ids = (
        run_spec_identity.get("evidence_roots", {}).get("prerequisite_receipts")
        if isinstance(run_spec_identity.get("evidence_roots"), dict)
        else None
    )
    if not isinstance(receipt_ids, dict):
        raise ValueError("package run spec prerequisite_receipts must be an object")
    if profile_variant == "final":
        if set(receipt_ids) != _RECEIPT_GATES:
            raise ValueError("final package requires all prerequisite receipt IDs")
        receipts: dict[str, dict[str, Any]] = {}
        receipt_evidence_bytes: dict[str, bytes] = {}
        for gate in sorted(_RECEIPT_GATES):
            receipt_id = receipt_ids[gate]
            receipt_path = (
                root.parents[1]
                / "receipts"
                / gate
                / f"{receipt_id}.json"
            )
            if receipt_path.is_symlink() or not receipt_path.is_file():
                raise ValueError(f"missing {gate} prerequisite receipt")
            data = receipt_path.read_bytes()
            manifest = loads_no_duplicate_keys(data.decode("utf-8"))
            identity = manifest.get("identity") if isinstance(manifest, dict) else None
            if (
                not isinstance(identity, dict)
                or compute_id(identity) != manifest.get("id")
                or manifest.get("id") != receipt_id
            ):
                raise ValueError(f"{gate} prerequisite receipt id mismatch")
            receipts[gate] = manifest
            packaged_path = f"evidence/prerequisite_receipts/{gate}.json"
            candidates[packaged_path] = data
            evidence_path = receipt_path.with_suffix(".evidence.json")
            if evidence_path.is_symlink() or not evidence_path.is_file():
                raise ValueError(f"missing {gate} prerequisite evidence")
            evidence_data = evidence_path.read_bytes()
            receipt_evidence_bytes[gate] = evidence_data
            candidates[
                f"evidence/prerequisite_receipts/{gate}.evidence.json"
            ] = evidence_data
            role = f"prerequisite_receipt_{gate}"
            normalized[role] = {
                "role": role,
                "content_id": receipt_id,
                "sha256": hashlib.sha256(data).hexdigest(),
                "byte_size": len(data),
                "retrieval_path": packaged_path,
            }
        identity_bindings = {
            key: run_spec_identity.get("identity", {}).get(key)
            for key in _FULL_RECEIPT_BINDINGS
        }
        validate_prerequisite_receipts(
            profile_variant="final",
            identity_bindings=identity_bindings,
            receipt_ids=receipt_ids,
            receipts=receipts,
        )
        for gate in sorted(_RECEIPT_GATES):
            identity = receipts[gate]["identity"]
            verify_prerequisite_evidence_bytes(
                gate=gate,
                bindings=identity["bindings"],
                receipt_evidence=identity["evidence"],
                data=receipt_evidence_bytes[gate],
            )
    elif profile_variant == "smoke":
        validate_prerequisite_receipts(
            profile_variant="smoke",
            identity_bindings={},
            receipt_ids=receipt_ids,
            receipts={},
        )
    else:
        raise ValueError(f"unknown package profile variant {profile_variant!r}")
    return (
        [normalized[role] for role in sorted(normalized)],
        candidates,
        profile_variant,
    )


def package_run(
    output_dir: Path,
    aggregate: dict[str, Any],
    *,
    resource_summary: dict[str, Any],
    external_artifacts: list[dict[str, Any]] | None = None,
    evidence_files: dict[str, bytes] | None = None,
) -> None:
    """Create a package once, or accept only byte-identical cached content."""
    if not external_artifacts:
        raise ValueError("package requires a nonempty external-artifact ledger")
    root = Path(output_dir)
    normalized_artifacts, packaged_evidence, package_profile = (
        _prepare_package_evidence(
        root,
        external_artifacts=external_artifacts,
        evidence_files=dict(evidence_files or {}),
        )
    )
    candidates: dict[str, bytes] = {
        "reports/report.md": render_markdown(
            aggregate,
            resource_summary=resource_summary,
        ).encode("utf-8"),
        "reports/report.tex": render_latex(aggregate).encode("utf-8"),
        "external_artifacts.json": (
            json.dumps(
                {"artifacts": normalized_artifacts},
                indent=2,
                sort_keys=True,
            ) + "\n"
        ).encode("utf-8"),
    }
    candidates.update(packaged_evidence)

    with tempfile.TemporaryDirectory(prefix="stopdff_v5_package_") as td:
        figure_root = Path(td)
        for name in write_figures(
            figure_root,
            aggregate,
            profile_variant=package_profile or aggregate.get("profile_variant"),
        ):
            candidates[name] = (figure_root / name).read_bytes()

    for path in root.rglob("*"):
        if not path.is_file() and not path.is_symlink():
            continue
        rel = path.relative_to(root).as_posix()
        if Path(rel).parts[0] in PACKAGE_MANAGED_ROOTS and rel not in candidates:
            raise ValueError(f"unexpected cached package evidence at {path}")

    checksum_lines: list[str] = []
    scientific_paths: dict[str, bytes] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"symlink cannot be packaged: {path}")
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        if rel == "SHA256SUMS" or rel in candidates:
            continue
        violation = _packageable_path_violation(rel)
        if violation is not None:
            raise ValueError(violation)
        scientific_paths[rel] = path.read_bytes()
    for rel, data in {**scientific_paths, **candidates}.items():
        checksum_lines.append(
            _checksum_line(hashlib.sha256(data).hexdigest(), rel)
        )
    candidates["SHA256SUMS"] = (
        "\n".join(sorted(checksum_lines)) + "\n"
    ).encode("utf-8")

    # Check every existing managed byte before filling any missing path.
    for rel, data in candidates.items():
        path = root / rel
        if path.exists() and (
            path.is_symlink()
            or not path.is_file()
            or path.read_bytes() != data
        ):
            raise ValueError(f"package evidence mismatch at {path}")
    for rel, data in candidates.items():
        path = root / rel
        if not path.exists():
            publish_bytes(path, data)
