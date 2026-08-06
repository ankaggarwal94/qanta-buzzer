"""Report / figure / package writers (see ACCEPTANCE_CONTRACT.md).

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
from .identity import compute_id, loads_no_duplicate_keys
from .profile import CALIBRATION, REWARD_SCHEDULES
from .receipt_evidence import (
    DETERMINISM_BINDINGS,
    FULL_RECEIPT_BINDINGS,
    build_prerequisite_evidence,
    prerequisite_evidence_sha256,
    verify_prerequisite_evidence_bytes,
)
from .rewards import REWARD_SCHEDULE_STRINGS

_RECEIPT_GATES = {"smoke", "mutation", "determinism"}
_FULL_RECEIPT_BINDINGS = FULL_RECEIPT_BINDINGS
_DETERMINISM_BINDINGS = DETERMINISM_BINDINGS


def _validate_receipt_evidence(gate: str, evidence: Any) -> None:
    """Validate the digest that binds a receipt to its packaged evidence bytes."""
    if not isinstance(evidence, dict) or set(evidence) != {"evidence_sha256"}:
        raise ValueError(f"{gate} receipt evidence fields mismatch")
    value = evidence.get("evidence_sha256")
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(ch not in "0123456789abcdef" for ch in value)
    ):
        raise ValueError(f"{gate} receipt evidence_sha256 must be lowercase SHA-256")


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
    _validate_receipt_evidence(gate, evidence)
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
    from .receipt_evidence import validate_prerequisite_evidence

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
            set(identity) != {"kind", "gate", "status", "bindings", "evidence"}
            or
            identity.get("kind") != "prerequisite_receipt"
            or identity.get("gate") != gate
            or identity.get("status") != "successful"
            or identity.get("bindings") != expected_bindings
            or not isinstance(identity.get("evidence"), dict)
            or not identity["evidence"]
        ):
            raise ValueError(f"{gate} receipt bindings/status mismatch")
        _validate_receipt_evidence(gate, identity["evidence"])


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


def write_figures(
    output_dir: Path,
    aggregate: dict[str, Any],
    *,
    profile_variant: str | None = None,
) -> list[str]:
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
    except Exception as exc:
        if profile_variant != "smoke":
            raise RuntimeError(
                "figure generation failed for a non-smoke package"
            ) from exc
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
    if role == "raw_input_manifest":
        semantic_checks = identity.get("semantic_checks")
        if (
            not isinstance(semantic_checks, dict)
            or semantic_checks.get("all_semantic_checks_pass") is not True
        ):
            raise ValueError("raw-input semantic checks did not pass")
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

    run_spec_path = root / "run_spec.json"
    if not run_spec_path.exists():
        normalized_legacy: list[dict[str, Any]] = []
        for artifact in external_artifacts:
            if not isinstance(artifact, dict):
                raise ValueError("invalid external-artifact ledger entry")
            retrieval = artifact.get("retrieval_path", artifact.get("retrieval"))
            if (
                not isinstance(artifact.get("role"), str)
                or not isinstance(artifact.get("content_id"), str)
                or not isinstance(artifact.get("sha256"), str)
                or len(artifact.get("sha256", "")) != 64
                or isinstance(artifact.get("byte_size"), bool)
                or not isinstance(artifact.get("byte_size"), int)
                or artifact.get("byte_size", 0) <= 0
                or not isinstance(retrieval, str)
                or not retrieval
            ):
                raise ValueError("invalid external-artifact ledger entry")
            normalized_legacy.append(
                {
                    "role": artifact["role"],
                    "content_id": artifact["content_id"],
                    "sha256": artifact["sha256"],
                    "byte_size": artifact["byte_size"],
                    "retrieval_path": retrieval,
                }
            )
        return normalized_legacy, candidates, None

    normalized: dict[str, dict[str, Any]] = {}
    for artifact in external_artifacts:
        if not isinstance(artifact, dict):
            raise ValueError("invalid external-artifact ledger entry")
        fields = set(artifact)
        retrieval_fields = fields & {"retrieval", "retrieval_path"}
        if (
            fields
            not in (
                {"role", "content_id", "sha256", "byte_size", "retrieval"},
                {"role", "content_id", "sha256", "byte_size", "retrieval_path"},
            )
            or len(retrieval_fields) != 1
        ):
            raise ValueError("invalid external-artifact ledger entry")
        role = artifact.get("role")
        content_id = artifact.get("content_id")
        digest = artifact.get("sha256")
        byte_size = artifact.get("byte_size")
        retrieval = artifact[next(iter(retrieval_fields))]
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
            data = _resolve_retrieval_path(root, retrieval).read_bytes()
            candidates[packaged_path] = data
        else:
            data = candidates.get(packaged_path)
            if data is None:
                raise ValueError(f"missing packaged evidence for {role}")
        manifest = _manifest_from_bytes(data, role=role)
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

    if run_spec_path.is_symlink() or not run_spec_path.is_file():
        raise ValueError("package requires run_spec.json before packaging")
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
    from .sweep import atomic_write_bytes

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
