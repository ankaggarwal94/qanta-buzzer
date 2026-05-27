from __future__ import annotations

import json
from pathlib import Path

from scripts.verify_audit_release import main


def test_verify_audit_release_flags_stale_artifact(tmp_path: Path) -> None:
    exports = tmp_path / "paper_exports"
    exports.mkdir()

    for name in [
        "csli.json",
        "calibration.json",
        "stopdff.json",
        "audit_card.md",
        "audit_table.tex",
    ]:
        (exports / name).write_text("{}", encoding="utf-8")

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
        "metadata": {"generation": {"git_dirty": False}},
        "artifact_provenance": {
            "csli.json": {"sha_matches": False},
        },
        "data_provenance": {},
    }
    (exports / "audit_card.json").write_text(
        json.dumps(audit), encoding="utf-8"
    )

    assert main(["--paper-exports", str(exports), "--repo-root", str(tmp_path)]) == 1


def test_verify_audit_release_accepts_clean_warn(tmp_path: Path) -> None:
    exports = tmp_path / "paper_exports"
    exports.mkdir()

    for name in [
        "csli.json",
        "calibration.json",
        "stopdff.json",
        "audit_table.tex",
    ]:
        (exports / name).write_text("{}", encoding="utf-8")

    (exports / "audit_card.md").write_text(
        "Overall WARN\n\nretained MC subset\n", encoding="utf-8"
    )
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
        "metadata": {"generation": {"git_dirty": False}},
        "artifact_provenance": {
            "csli.json": {"sha_matches": True},
            "calibration.json": {"sha_matches": True},
            "stopdff.json": {"sha_matches": True},
        },
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

    assert main(["--paper-exports", str(exports), "--repo-root", str(tmp_path)]) == 0



def test_verify_audit_release_flags_threshold_sha_mismatch(tmp_path: Path) -> None:
    exports = tmp_path / "paper_exports"
    exports.mkdir()

    for name in [
        "csli.json",
        "calibration.json",
        "stopdff.json",
        "audit_table.tex",
    ]:
        (exports / name).write_text("{}", encoding="utf-8")

    (exports / "audit_card.md").write_text("Overall WARN\n", encoding="utf-8")
    audit = {
        "metrics": [
            {
                "name": "Diagnostic StopDFF (Median Abs Prefix Shift)",
                "verdict": "warn",
                "details": {},
            }
        ],
        "metadata": {"generation": {"git_dirty": False}},
        "artifact_provenance": {
            "csli.json": {"sha_matches": True},
        },
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
