from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.verify_audit_release import main


REQUIRED_EXPORT_FILES = [
    "csli.json",
    "calibration.json",
    "stopdff.json",
    "audit_table.tex",
    "csli_panel.png",
    "reliability_early.png",
    "reliability_mid.png",
    "reliability_late.png",
]


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _populate_required_exports(exports: Path) -> None:
    """Create minimal placeholder files for every required export.

    The verifier checks existence only, not content, so empty/JSON-stub
    payloads are fine.
    """
    exports.mkdir(parents=True, exist_ok=True)
    for name in REQUIRED_EXPORT_FILES:
        (exports / name).write_text("{}", encoding="utf-8")


def _write_threshold_manifest(repo_root: Path, payload: str = "payload") -> None:
    manifest = repo_root / "threshold_manifest.json"
    sidecar = repo_root / "threshold_manifest.json.sha256"
    manifest.write_text(payload, encoding="utf-8")
    sidecar.write_text(
        f"{_sha(payload)}  threshold_manifest.json\n", encoding="utf-8"
    )


def _write_producer_scripts(repo_root: Path) -> dict[str, dict[str, str]]:
    """Write the three producer scripts under repo_root and return the
    `artifact_provenance` block that matches their live SHAs."""
    scripts_dir = repo_root / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)

    provenance: dict[str, dict[str, str]] = {}
    for artifact_name, script_basename in (
        ("csli.json", "compute_csli.py"),
        ("calibration.json", "compute_prefix_calibration.py"),
        ("stopdff.json", "compute_stopdff.py"),
    ):
        body = f"# fake producer for {artifact_name}\n"
        script = scripts_dir / script_basename
        script.write_text(body, encoding="utf-8")
        # Match the canonical content the test populator wrote in
        # _populate_required_exports ("{}").
        content_sha = _sha("{}")
        provenance[artifact_name] = {
            "recorded_sha256": _sha_file(script),
            "script_path": f"scripts/{script_basename}",
            "content_sha256": content_sha,
        }
    return provenance


def _write_audit_generator(
    repo_root: Path, md_path: Path | None = None
) -> dict[str, str]:
    """Write a fake make_audit_card.py under repo_root and return the
    `metadata.generation` snippet matching its live SHA.

    If ``md_path`` is provided, also record the live SHA of that file as
    ``markdown_sha256`` so the generation block satisfies the verifier's
    audit_card.md content-binding check.
    """
    scripts_dir = repo_root / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    body = "# fake audit-card generator\n"
    script = scripts_dir / "make_audit_card.py"
    script.write_text(body, encoding="utf-8")
    block: dict[str, str] = {
        "git_dirty": False,
        "script_path": "scripts/make_audit_card.py",
        "script_sha256": _sha_file(script),
    }
    if md_path is not None:
        block["markdown_sha256"] = _sha_file(md_path)
    return block


def _bind_markdown(generation_block: dict, md_path: Path) -> None:
    """Update an existing generation block to record the live SHA of
    ``md_path`` as ``markdown_sha256``, mirroring what
    ``scripts/make_audit_card.py`` does after rendering audit_card.md.
    """
    generation_block["markdown_sha256"] = _sha_file(md_path)


def test_verify_audit_release_flags_stale_artifact(tmp_path: Path) -> None:
    exports = tmp_path / "paper_exports"
    _populate_required_exports(exports)
    _write_threshold_manifest(tmp_path)

    # Producer for csli was edited after audit_card.json was generated, so the
    # live SHA no longer matches the recorded SHA.
    provenance = _write_producer_scripts(tmp_path)
    generation_block = _write_audit_generator(tmp_path)
    (tmp_path / "scripts/compute_csli.py").write_text(
        "# tampered after audit_card was generated\n", encoding="utf-8"
    )

    audit = {
        "metrics": [
            {
                "name": "Diagnostic StopDFF (Median Abs Prefix Shift)",
                "verdict": "pass",
                "details": {
                    "ceiling_effect_detected": True,
                    "unreachable_buckets": ["early"],
                },
            }
        ],
        "metadata": {"generation": generation_block},
        "artifact_provenance": provenance,
        "data_provenance": {},
    }
    (exports / "audit_card.md").write_text("Overall WARN\n", encoding="utf-8")
    _bind_markdown(generation_block, exports / "audit_card.md")
    (exports / "audit_card.json").write_text(
        json.dumps(audit), encoding="utf-8"
    )

    assert (
        main(
            [
                "--paper-exports",
                str(exports),
                "--repo-root",
                str(tmp_path),
            ]
        )
        == 1
    )


def test_verify_audit_release_accepts_clean_warn(tmp_path: Path) -> None:
    exports = tmp_path / "paper_exports"
    _populate_required_exports(exports)
    _write_threshold_manifest(tmp_path)
    provenance = _write_producer_scripts(tmp_path)
    generation_block = _write_audit_generator(tmp_path)

    (exports / "audit_card.md").write_text(
        "Overall WARN\n\nretained MC subset\n", encoding="utf-8"
    )
    _bind_markdown(generation_block, exports / "audit_card.md")
    audit = {
        "metrics": [
            {
                "name": "Diagnostic StopDFF (Median Abs Prefix Shift)",
                "verdict": "warn",
                "details": {
                    "ceiling_effect_detected": True,
                    "unreachable_buckets": ["early"],
                },
            }
        ],
        "metadata": {"generation": generation_block},
        "artifact_provenance": provenance,
        "data_provenance": {
            "csli": {
                "retention": {
                    "test": {
                        "overridden": True,
                        "passed": False,
                        "applies": True,
                    }
                }
            }
        },
    }
    (exports / "audit_card.json").write_text(
        json.dumps(audit), encoding="utf-8"
    )

    assert (
        main(
            [
                "--paper-exports",
                str(exports),
                "--repo-root",
                str(tmp_path),
            ]
        )
        == 0
    )


def test_verify_audit_release_flags_threshold_sha_mismatch(tmp_path: Path) -> None:
    exports = tmp_path / "paper_exports"
    _populate_required_exports(exports)
    provenance = _write_producer_scripts(tmp_path)
    generation_block = _write_audit_generator(tmp_path)

    (exports / "audit_card.md").write_text("Overall WARN\n", encoding="utf-8")
    _bind_markdown(generation_block, exports / "audit_card.md")
    audit = {
        "metrics": [
            {
                "name": "Diagnostic StopDFF (Median Abs Prefix Shift)",
                "verdict": "warn",
                "details": {},
            }
        ],
        "metadata": {"generation": generation_block},
        "artifact_provenance": provenance,
        "data_provenance": {},
    }
    (exports / "audit_card.json").write_text(
        json.dumps(audit), encoding="utf-8"
    )

    # Threshold manifest content does not match the recorded SHA.
    (tmp_path / "threshold_manifest.json").write_text("payload", encoding="utf-8")
    (tmp_path / "threshold_manifest.json.sha256").write_text(
        "deadbeef  threshold_manifest.json\n", encoding="utf-8"
    )

    assert (
        main(
            [
                "--paper-exports",
                str(exports),
                "--repo-root",
                str(tmp_path),
            ]
        )
        == 1
    )


def test_verify_audit_release_flags_missing_provenance_entry(tmp_path: Path) -> None:
    exports = tmp_path / "paper_exports"
    _populate_required_exports(exports)
    _write_threshold_manifest(tmp_path)
    full_provenance = _write_producer_scripts(tmp_path)
    generation_block = _write_audit_generator(tmp_path)

    (exports / "audit_card.md").write_text("Overall WARN\n", encoding="utf-8")
    _bind_markdown(generation_block, exports / "audit_card.md")
    audit = {
        "metrics": [
            {
                "name": "Diagnostic StopDFF (Median Abs Prefix Shift)",
                "verdict": "warn",
                "details": {},
            }
        ],
        "metadata": {"generation": generation_block},
        # calibration.json + stopdff.json provenance entries are missing.
        "artifact_provenance": {"csli.json": full_provenance["csli.json"]},
        "data_provenance": {},
    }
    (exports / "audit_card.json").write_text(
        json.dumps(audit), encoding="utf-8"
    )

    assert (
        main(
            [
                "--paper-exports",
                str(exports),
                "--repo-root",
                str(tmp_path),
            ]
        )
        == 1
    )


def test_verify_audit_release_flags_missing_figure(tmp_path: Path) -> None:
    exports = tmp_path / "paper_exports"
    _populate_required_exports(exports)
    _write_threshold_manifest(tmp_path)
    provenance = _write_producer_scripts(tmp_path)
    generation_block = _write_audit_generator(tmp_path)

    # Remove a required figure to simulate a release that dropped a paper
    # asset declared canonical by ARTIFACTS.md.
    (exports / "csli_panel.png").unlink()

    (exports / "audit_card.md").write_text("Overall WARN\n", encoding="utf-8")
    _bind_markdown(generation_block, exports / "audit_card.md")
    audit = {
        "metrics": [
            {
                "name": "Diagnostic StopDFF (Median Abs Prefix Shift)",
                "verdict": "warn",
                "details": {},
            }
        ],
        "metadata": {"generation": generation_block},
        "artifact_provenance": provenance,
        "data_provenance": {},
    }
    (exports / "audit_card.json").write_text(
        json.dumps(audit), encoding="utf-8"
    )

    assert (
        main(
            [
                "--paper-exports",
                str(exports),
                "--repo-root",
                str(tmp_path),
            ]
        )
        == 1
    )


def test_verify_audit_release_flags_missing_threshold_manifest(
    tmp_path: Path,
) -> None:
    exports = tmp_path / "paper_exports"
    _populate_required_exports(exports)
    # Intentionally do NOT write threshold_manifest.json or its sidecar.
    provenance = _write_producer_scripts(tmp_path)
    generation_block = _write_audit_generator(tmp_path)

    (exports / "audit_card.md").write_text("Overall WARN\n", encoding="utf-8")
    _bind_markdown(generation_block, exports / "audit_card.md")
    audit = {
        "metrics": [
            {
                "name": "Diagnostic StopDFF (Median Abs Prefix Shift)",
                "verdict": "warn",
                "details": {},
            }
        ],
        "metadata": {"generation": generation_block},
        "artifact_provenance": provenance,
        "data_provenance": {},
    }
    (exports / "audit_card.json").write_text(
        json.dumps(audit), encoding="utf-8"
    )

    assert (
        main(
            [
                "--paper-exports",
                str(exports),
                "--repo-root",
                str(tmp_path),
            ]
        )
        == 1
    )



def test_verify_audit_release_flags_generator_sha_drift(tmp_path: Path) -> None:
    exports = tmp_path / "paper_exports"
    _populate_required_exports(exports)
    _write_threshold_manifest(tmp_path)
    provenance = _write_producer_scripts(tmp_path)
    generation_block = _write_audit_generator(tmp_path)

    # Tamper with make_audit_card.py after audit_card.json was generated.
    (tmp_path / "scripts/make_audit_card.py").write_text(
        "# tampered audit-card generator\n", encoding="utf-8"
    )

    (exports / "audit_card.md").write_text("Overall WARN\n", encoding="utf-8")
    _bind_markdown(generation_block, exports / "audit_card.md")
    audit = {
        "metrics": [
            {
                "name": "Diagnostic StopDFF (Median Abs Prefix Shift)",
                "verdict": "warn",
                "details": {},
            }
        ],
        "metadata": {"generation": generation_block},
        "artifact_provenance": provenance,
        "data_provenance": {},
    }
    (exports / "audit_card.json").write_text(
        json.dumps(audit), encoding="utf-8"
    )

    assert (
        main(
            [
                "--paper-exports",
                str(exports),
                "--repo-root",
                str(tmp_path),
            ]
        )
        == 1
    )



def test_verify_audit_release_flags_non_canonical_producer(tmp_path: Path) -> None:
    exports = tmp_path / "paper_exports"
    _populate_required_exports(exports)
    _write_threshold_manifest(tmp_path)
    provenance = _write_producer_scripts(tmp_path)
    generation_block = _write_audit_generator(tmp_path)

    # Plant a non-canonical helper script whose SHA matches a manipulated
    # provenance entry, simulating a tampered audit card that redirects
    # csli.json's script_path away from scripts/compute_csli.py.
    helper_body = "# non-canonical helper that happens to match recorded sha\n"
    helper_path = tmp_path / "scripts/helper.py"
    helper_path.write_text(helper_body, encoding="utf-8")
    provenance["csli.json"] = {
        "recorded_sha256": _sha(helper_body),
        "script_path": "scripts/helper.py",
    }

    (exports / "audit_card.md").write_text("Overall WARN\n", encoding="utf-8")
    _bind_markdown(generation_block, exports / "audit_card.md")
    audit = {
        "metrics": [
            {
                "name": "Diagnostic StopDFF (Median Abs Prefix Shift)",
                "verdict": "warn",
                "details": {},
            }
        ],
        "metadata": {"generation": generation_block},
        "artifact_provenance": provenance,
        "data_provenance": {},
    }
    (exports / "audit_card.json").write_text(
        json.dumps(audit), encoding="utf-8"
    )

    assert (
        main(
            [
                "--paper-exports",
                str(exports),
                "--repo-root",
                str(tmp_path),
            ]
        )
        == 1
    )



def test_verify_audit_release_flags_source_content_drift(tmp_path: Path) -> None:
    exports = tmp_path / "paper_exports"
    _populate_required_exports(exports)
    _write_threshold_manifest(tmp_path)
    provenance = _write_producer_scripts(tmp_path)
    generation_block = _write_audit_generator(tmp_path)

    # Edit csli.json after audit_card.json was generated. The producer
    # script SHA still matches, but the source content has drifted.
    (exports / "csli.json").write_text(
        '{"tampered": true}', encoding="utf-8"
    )

    (exports / "audit_card.md").write_text("Overall WARN\n", encoding="utf-8")
    _bind_markdown(generation_block, exports / "audit_card.md")
    audit = {
        "metrics": [
            {
                "name": "Diagnostic StopDFF (Median Abs Prefix Shift)",
                "verdict": "warn",
                "details": {},
            }
        ],
        "metadata": {"generation": generation_block},
        "artifact_provenance": provenance,
        "data_provenance": {},
    }
    (exports / "audit_card.json").write_text(
        json.dumps(audit), encoding="utf-8"
    )

    assert (
        main(
            [
                "--paper-exports",
                str(exports),
                "--repo-root",
                str(tmp_path),
            ]
        )
        == 1
    )


def test_verify_audit_release_flags_non_canonical_generator(tmp_path: Path) -> None:
    exports = tmp_path / "paper_exports"
    _populate_required_exports(exports)
    _write_threshold_manifest(tmp_path)
    provenance = _write_producer_scripts(tmp_path)

    # Generator script_path points to a non-canonical helper rather than
    # scripts/make_audit_card.py.
    helper_body = "# non-canonical generator helper\n"
    helper = tmp_path / "scripts/audit_helper.py"
    helper.parent.mkdir(parents=True, exist_ok=True)
    helper.write_text(helper_body, encoding="utf-8")
    generation_block = {
        "git_dirty": False,
        "script_path": "scripts/audit_helper.py",
        "script_sha256": _sha(helper_body),
    }

    (exports / "audit_card.md").write_text("Overall WARN\n", encoding="utf-8")
    _bind_markdown(generation_block, exports / "audit_card.md")
    audit = {
        "metrics": [
            {
                "name": "Diagnostic StopDFF (Median Abs Prefix Shift)",
                "verdict": "warn",
                "details": {},
            }
        ],
        "metadata": {"generation": generation_block},
        "artifact_provenance": provenance,
        "data_provenance": {},
    }
    (exports / "audit_card.json").write_text(
        json.dumps(audit), encoding="utf-8"
    )

    assert (
        main(
            [
                "--paper-exports",
                str(exports),
                "--repo-root",
                str(tmp_path),
            ]
        )
        == 1
    )


def test_verify_audit_release_flags_stale_audit_card_md(tmp_path: Path) -> None:
    exports = tmp_path / "paper_exports"
    _populate_required_exports(exports)
    _write_threshold_manifest(tmp_path)
    provenance = _write_producer_scripts(tmp_path)

    # Generation block records the SHA of the original audit_card.md.
    (exports / "audit_card.md").write_text(
        "Overall WARN\n", encoding="utf-8"
    )
    generation_block = _write_audit_generator(
        tmp_path, md_path=exports / "audit_card.md"
    )

    # Now hand-edit audit_card.md after the generation was recorded. This
    # simulates a stale or hand-edited Markdown that should be rejected.
    (exports / "audit_card.md").write_text(
        "Overall PASS (tampered)\n", encoding="utf-8"
    )

    audit = {
        "metrics": [
            {
                "name": "Diagnostic StopDFF (Median Abs Prefix Shift)",
                "verdict": "warn",
                "details": {},
            }
        ],
        "metadata": {"generation": generation_block},
        "artifact_provenance": provenance,
        "data_provenance": {},
    }
    (exports / "audit_card.json").write_text(
        json.dumps(audit), encoding="utf-8"
    )

    assert (
        main(
            [
                "--paper-exports",
                str(exports),
                "--repo-root",
                str(tmp_path),
            ]
        )
        == 1
    )
