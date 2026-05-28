"""Tests for the --fit-split flag in scripts/compute_prefix_calibration.py.

The flag lets a learned-value StopDFF trainer (Prompt 5) consume a
train-fit calibration artifact without overwriting the val-fit artifact
that backs the audit card. Default --fit-split=val preserves byte-for-
byte backward compatibility with the existing paper_exports/
calibration.json (modulo two additive metadata fields: ``fit_split``
and ``n_fit``).

These tests exercise the argparse surface and the --dry-run integration
path. A full SBERT-driven integration run is deliberately out of scope
(it would add minutes per pytest invocation); tests/test_pr14_review_
regressions.py already pins the calibrator-internal behavior via
``_fit_bucket_calibrator`` / ``_calibrate_bucket_scores``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from scripts import compute_prefix_calibration

_parse_args = getattr(compute_prefix_calibration, "_parse_args")


# ---------------------------------------------------------------------------
# Argparse surface
# ---------------------------------------------------------------------------


def test_fit_split_flag_defaults_to_val() -> None:
    """No flag → args.fit_split == 'val' (backward-compatible default)."""
    args = _parse_args([])
    assert args.fit_split == "val"


def test_fit_split_train_is_accepted() -> None:
    """--fit-split train parses cleanly and resolves to 'train'."""
    args = _parse_args(["--fit-split", "train"])
    assert args.fit_split == "train"


def test_fit_split_val_is_accepted_explicitly() -> None:
    """Explicit --fit-split val parses cleanly (matches default)."""
    args = _parse_args(["--fit-split", "val"])
    assert args.fit_split == "val"


def test_fit_split_invalid_value_rejected() -> None:
    """argparse choices validation must reject unknown values."""
    with pytest.raises(SystemExit) as exc_info:
        _parse_args(["--fit-split", "unknown"])
    # argparse exits with code 2 on argument-parse failure.
    assert exc_info.value.code == 2


# ---------------------------------------------------------------------------
# --dry-run integration: the dry-run path enumerates required files based on
# args.fit_split and exits 0 without loading SBERT.
# ---------------------------------------------------------------------------


def _write_minimal_split_files(data_dir: Path) -> None:
    """Write empty-list dataset files sufficient for --dry-run gating.

    The dry-run check only stats file existence; content is not parsed.
    """
    for fname in ("mc_dataset.json", "val_dataset.json",
                  "train_dataset.json", "test_dataset.json"):
        (data_dir / fname).write_text("[]\n", encoding="utf-8")


def test_dry_run_default_requires_val_dataset(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Default --fit-split=val: dry-run requires mc + val + test, not train."""
    _write_minimal_split_files(tmp_path)
    # Remove train_dataset.json so we can prove val (not train) is required.
    (tmp_path / "train_dataset.json").unlink()

    rc = compute_prefix_calibration.main([
        "--dry-run",
        "--data-dir", str(tmp_path),
        "--output", str(tmp_path / "calibration_out.json"),
    ])

    assert rc == 0
    captured = capsys.readouterr()
    out = captured.out + captured.err
    assert "Fit split: val" in out
    assert "val_dataset.json" in out
    assert "FOUND" in out  # at least one FOUND from val/mc/test


def test_dry_run_fit_split_train_requires_train_dataset(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """--fit-split train: dry-run requires train_dataset.json instead of val."""
    _write_minimal_split_files(tmp_path)
    # Remove val to prove the dry-run no longer requires it.
    (tmp_path / "val_dataset.json").unlink()

    rc = compute_prefix_calibration.main([
        "--dry-run",
        "--fit-split", "train",
        "--data-dir", str(tmp_path),
        "--output", str(tmp_path / "calibration_train_out.json"),
    ])

    assert rc == 0
    captured = capsys.readouterr()
    out = captured.out + captured.err
    assert "Fit split: train" in out
    assert "train_dataset.json" in out
    # val_dataset.json was deleted but dry-run must NOT have flagged it MISSING
    # (it isn't required when --fit-split=train).
    assert "val_dataset.json: MISSING" not in out


def test_dry_run_fit_split_train_fails_when_train_dataset_missing(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """--fit-split train without train_dataset.json must fail dry-run."""
    _write_minimal_split_files(tmp_path)
    (tmp_path / "train_dataset.json").unlink()

    rc = compute_prefix_calibration.main([
        "--dry-run",
        "--fit-split", "train",
        "--data-dir", str(tmp_path),
        "--output", str(tmp_path / "calibration_train_out.json"),
    ])

    assert rc == 1
    captured = capsys.readouterr()
    out = captured.out + captured.err
    assert "train_dataset.json: MISSING" in out


def test_default_invocation_warns_only_for_fit_split_train_at_default_output(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Warning fires only on --fit-split=train + default --output.

    The warning is emitted before --dry-run runs, so a dry-run is a
    cheap way to exercise the branch. With an explicit non-default
    --output (a tmp_path), the warning must NOT fire.
    """
    _write_minimal_split_files(tmp_path)

    # Non-default --output: no warning even with --fit-split=train.
    rc = compute_prefix_calibration.main([
        "--dry-run",
        "--fit-split", "train",
        "--data-dir", str(tmp_path),
        "--output", str(tmp_path / "calibration_train_out.json"),
    ])
    assert rc == 0
    captured = capsys.readouterr()
    assert "WARNING: --fit-split=train" not in (captured.out + captured.err)


# ---------------------------------------------------------------------------
# Argparse-level guarantee that --fit-split shows up in --help output.
# Acts as a discoverability regression guard for operators.
# ---------------------------------------------------------------------------


def test_fit_split_flag_documented_in_help(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """--help output mentions the --fit-split flag and both choices."""
    with pytest.raises(SystemExit):
        _parse_args(["--help"])
    captured = capsys.readouterr()
    help_text = captured.out + captured.err
    assert "--fit-split" in help_text
    assert "val" in help_text
    assert "train" in help_text
