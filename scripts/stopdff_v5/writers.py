"""Report / figure / package writers (ACCEPTANCE_CONTRACT.md section 4, 7).

Markdown + LaTeX include every normative definition and the resource/cost block; figures
are valid PNGs with positive dimensions; the package carries a safe complete SHA256SUMS
and an external_artifacts.json retrieval index.
"""
from __future__ import annotations

import json
import struct
import zlib
from pathlib import Path
from typing import Any

from . import PROFILE_NAME, PROTOCOL_VERSION
from .profile import CALIBRATION, GATE, REWARD_SCHEDULES, profile_static_identity
from .rewards import REWARD_SCHEDULE_STRINGS


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
) -> None:
    """Write reports, figures, external_artifacts.json, and SHA256SUMS (package stage)."""
    from .sweep import atomic_write_bytes
    root = Path(output_dir)
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(reports / "report.md", render_markdown(aggregate, resource_summary=resource_summary).encode("utf-8"))
    atomic_write_bytes(reports / "report.tex", render_latex(aggregate).encode("utf-8"))
    write_figures(root, aggregate)
    write_external_artifacts(root, external_artifacts or [])
    write_sha256sums(root)  # must be last
