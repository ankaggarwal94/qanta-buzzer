#!/usr/bin/env python3
"""Audit raw quiz bowl questions or persisted train, validation and test splits."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qb_data.raw_audit import audit_raw_file


def main(argv: list[str] | None = None) -> int:
    """Write a JSON report and return 0 for clean, 1 for findings or 2 for an input error."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Raw question source")
    parser.add_argument("--format", choices=("csv", "qanta", "qbreader"), default="csv")
    parser.add_argument(
        "--qanta-answer-field", choices=("auto", "page", "answer"), default="auto"
    )
    parser.add_argument(
        "--splits-dir",
        type=Path,
        help="Directory containing train/val/test_dataset.json",
    )
    parser.add_argument("--report", type=Path, required=True, help="Output JSON report")
    parser.add_argument(
        "--quarantine", type=Path, help="Optional JSONL copy of rejected raw records"
    )
    args = parser.parse_args(argv)
    if args.input is None and args.splits_dir is None:
        parser.error("provide --input, --splits-dir, or both")
    if args.quarantine and args.input is None:
        parser.error("--quarantine requires --input")
    if args.qanta_answer_field != "auto" and (
        args.format != "qanta" or args.input is None
    ):
        parser.error("--qanta-answer-field requires a QANTA input")

    paths = {}
    protected = [args.input] if args.input else []
    if args.splits_dir:
        paths = {
            name: args.splits_dir / f"{name}_dataset.json"
            for name in ("train", "val", "test")
        }
        protected.extend(paths.values())
        protected.extend(
            args.splits_dir / name
            for name in (
                "mc_dataset.json",
                "split_metadata.json",
                "build_status.json",
                "build_metadata.json",
            )
        )
    for output in (args.report, args.quarantine):
        if output is None:
            continue
        if any(
            output.resolve() == p.resolve()
            or (output.exists() and p.exists() and output.samefile(p))
            for p in protected
        ):
            parser.error(
                "audit outputs must be distinct from input files and each other"
            )
        protected.append(output)

    report = {"schema_version": 1}
    errors = {}
    findings = False
    if args.input:
        try:
            raw, rejected = audit_raw_file(
                args.input, args.format, args.qanta_answer_field
            )
            report["raw_data"] = raw
            findings = bool(rejected)
            if args.quarantine:
                args.quarantine.parent.mkdir(parents=True, exist_ok=True)
                with args.quarantine.open(
                    "w", encoding="utf-8", newline="\n"
                ) as stream:
                    for row in rejected:
                        stream.write(json.dumps(row) + "\n")
        except (OSError, TypeError, ValueError, csv.Error) as exc:
            errors["raw_data"] = str(exc)
    if paths:
        from qb_data.data_audit import audit_split_files

        try:
            splits = audit_split_files(paths)
            report["split_integrity"] = splits
            findings |= splits["status"] != "PASS"
        except (OSError, TypeError, ValueError) as exc:
            errors["split_integrity"] = str(exc)
    status = "FAIL" if findings else "PASS"
    if errors:
        report["errors"] = errors
        status = "ERROR"
    report["status"] = status
    try:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
        )
    except OSError as exc:
        print(f"Could not write report: {exc}", file=sys.stderr)
        return 2
    print(f"{status}: {args.report}")
    return {"PASS": 0, "FAIL": 1, "ERROR": 2}[status]


if __name__ == "__main__":
    raise SystemExit(main())
