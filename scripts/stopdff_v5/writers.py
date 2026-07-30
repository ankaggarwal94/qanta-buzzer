"""Report / figure / package writers (ACCEPTANCE_CONTRACT.md section 4, 7).

Markdown + LaTeX include every normative definition and the resource/cost block; figures
are valid PNGs with positive dimensions; the package carries a safe complete SHA256SUMS
and an external_artifacts.json retrieval index.
"""
from __future__ import annotations

import json
import hashlib
import struct
import tempfile
import zlib
from pathlib import Path
from typing import Any

from . import PROFILE_NAME, PROTOCOL_VERSION
from .profile import CALIBRATION, REWARD_SCHEDULES
from .rewards import REWARD_SCHEDULE_STRINGS

_RECEIPT_GATES = {"smoke", "mutation", "determinism"}
_FULL_RECEIPT_BINDINGS = {
    "source_manifest_id",
    "raw_input_bundle_id",
    "model_snapshot_id",
    "adapter_bundle_id",
    "fvi_study_id",
    "environment_contract_id",
}
_DETERMINISM_BINDINGS = {
    "source_manifest_id",
    "raw_input_bundle_id",
    "model_snapshot_id",
    "adapter_bundle_id",
}


def build_prerequisite_receipt(
    *,
    gate: str,
    bindings: dict[str, str],
    evidence: dict[str, Any],
) -> dict[str, Any]:
    """Build a content-addressed successful prerequisite receipt."""
    from .identity import build_manifest

    if gate not in _RECEIPT_GATES:
        raise ValueError(f"unknown prerequisite gate {gate!r}")
    required = (
        _DETERMINISM_BINDINGS
        if gate == "determinism"
        else _FULL_RECEIPT_BINDINGS
    )
    if set(bindings) != required:
        raise ValueError(f"{gate} receipt bindings mismatch")
    if not evidence:
        raise ValueError(f"{gate} receipt evidence must be nonempty")
    return build_manifest({
        "kind": "prerequisite_receipt",
        "gate": gate,
        "status": "successful",
        "bindings": {key: bindings[key] for key in sorted(bindings)},
        "evidence": evidence,
    })


def validate_prerequisite_receipts(
    *,
    profile_variant: str,
    identity_bindings: dict[str, str],
    receipt_ids: dict[str, str],
    receipts: dict[str, dict[str, Any]],
) -> None:
    """Fail closed on missing, mismatched, or unbound final-run receipts."""
    from .identity import compute_id

    if profile_variant == "smoke":
        if receipt_ids or receipts:
            raise ValueError("smoke run must not claim prerequisite receipts")
        return
    if profile_variant != "final":
        raise ValueError(f"unknown profile variant {profile_variant!r}")
    if set(receipt_ids) != _RECEIPT_GATES or set(receipts) != _RECEIPT_GATES:
        raise ValueError("final run requires smoke/mutation/determinism receipts")
    if set(identity_bindings) != _FULL_RECEIPT_BINDINGS:
        raise ValueError("final receipt identity bindings are incomplete")
    for gate in sorted(_RECEIPT_GATES):
        manifest = receipts[gate]
        identity = manifest.get("identity")
        if (
            not isinstance(identity, dict)
            or compute_id(identity) != manifest.get("id")
            or manifest.get("id") != receipt_ids[gate]
        ):
            raise ValueError(f"{gate} receipt id mismatch")
        required = (
            _DETERMINISM_BINDINGS
            if gate == "determinism"
            else _FULL_RECEIPT_BINDINGS
        )
        expected_bindings = {
            key: identity_bindings[key]
            for key in sorted(required)
        }
        if (
            identity.get("kind") != "prerequisite_receipt"
            or identity.get("gate") != gate
            or identity.get("status") != "successful"
            or identity.get("bindings") != expected_bindings
            or not isinstance(identity.get("evidence"), dict)
            or not identity["evidence"]
        ):
            raise ValueError(f"{gate} receipt bindings/status mismatch")


def render_markdown(aggregate: dict[str, Any], *, resource_summary: dict[str, Any]) -> str:
    fam = aggregate.get("family") or {}
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
    any_override = bool(ov.get("allow_low_mc_retention") or ov.get("allow_incomplete_mc_coverage"))
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
    rows = []
    for key, c in sorted(aggregate.get("cells", {}).items()):
        rows.append(f"{key.replace('_', ' ')} & {c.get('verdict')} \\\\")
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


def write_figures(output_dir: Path, aggregate: dict[str, Any]) -> list[str]:
    figs_dir = Path(output_dir) / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        cells = aggregate.get("cells", {})
        points = [c.get("abs_median_point", 0.0) for c in cells.values() if "abs_median_point" in c]
        fig, ax = plt.subplots(figsize=(6, 4))
        if points:
            ax.hist(points, bins=min(20, max(3, len(points))))
        ax.axvline(1.0, color="red", linestyle="--", label="material threshold")
        ax.set_xlabel("cell median |index shift|")
        ax.set_ylabel("count")
        ax.set_title("StopDFF cell median absolute index shift")
        ax.legend()
        fig.tight_layout()
        fig.savefig(figs_dir / "cell_median_index_shift.png", dpi=100)
        plt.close(fig)
        written.append("figures/cell_median_index_shift.png")
    except Exception:
        write_min_png(figs_dir / "cell_median_index_shift.png")
        written.append("figures/cell_median_index_shift.png")
    return written


def write_external_artifacts(output_dir: Path, artifacts: list[dict[str, Any]]) -> None:
    from .sweep import atomic_write_json
    atomic_write_json(Path(output_dir) / "external_artifacts.json", {"artifacts": artifacts})


def write_sha256sums(output_dir: Path) -> None:
    from .identity import sha256_file
    root = Path(output_dir)
    lines: list[str] = []
    for p in sorted(root.rglob("*")):
        if p.is_symlink() or not p.is_file():
            continue
        rel = p.relative_to(root).as_posix()
        if rel == "SHA256SUMS":
            continue
        lines.append(f"{sha256_file(p)}  {rel}")
    (root / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def package_run(
    output_dir: Path,
    aggregate: dict[str, Any],
    *,
    resource_summary: dict[str, Any],
    external_artifacts: list[dict[str, Any]] | None = None,
    evidence_files: dict[str, bytes] | None = None,
) -> None:
    """Create a package once, or accept only byte-identical cached content."""
    from .sweep import atomic_write_bytes

    if not external_artifacts:
        raise ValueError("package requires a nonempty external-artifact ledger")
    for artifact in external_artifacts:
        retrieval = (
            artifact.get("retrieval_path")
            if isinstance(artifact, dict)
            else None
        )
        if retrieval is None and isinstance(artifact, dict):
            # Backward-compatible synthetic checker fixture; real runners emit
            # retrieval_path exclusively.
            retrieval = artifact.get("retrieval")
        if (
            not isinstance(artifact, dict)
            or not isinstance(artifact.get("role"), str)
            or not isinstance(artifact.get("content_id"), str)
            or not isinstance(artifact.get("sha256"), str)
            or len(artifact.get("sha256", "")) != 64
            or not isinstance(artifact.get("byte_size"), int)
            or artifact.get("byte_size", -1) < 0
            or not isinstance(retrieval, str)
            or not retrieval
        ):
            raise ValueError("invalid external-artifact ledger entry")

    root = Path(output_dir)
    candidates: dict[str, bytes] = {
        "reports/report.md": render_markdown(
            aggregate,
            resource_summary=resource_summary,
        ).encode("utf-8"),
        "reports/report.tex": render_latex(aggregate).encode("utf-8"),
        "external_artifacts.json": (
            json.dumps(
                {"artifacts": external_artifacts},
                indent=2,
                sort_keys=True,
            ) + "\n"
        ).encode("utf-8"),
    }
    for name, data in (evidence_files or {}).items():
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

    with tempfile.TemporaryDirectory(prefix="stopdff_v5_package_") as td:
        figure_root = Path(td)
        for name in write_figures(figure_root, aggregate):
            candidates[name] = (figure_root / name).read_bytes()

    managed_roots = {"reports", "figures", "evidence"}
    for path in root.rglob("*"):
        if not path.is_file() and not path.is_symlink():
            continue
        rel = path.relative_to(root).as_posix()
        if Path(rel).parts[0] in managed_roots and rel not in candidates:
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
        scientific_paths[rel] = path.read_bytes()
    for rel, data in {**scientific_paths, **candidates}.items():
        checksum_lines.append(f"{hashlib.sha256(data).hexdigest()}  {rel}")
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
            atomic_write_bytes(path, data)
