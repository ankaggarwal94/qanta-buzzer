"""Verifier CLI. Documented invocation (R-037):

    python -m reproducibility.colm_aims_2026.verify --mode {source,release} \
        --tree PATH [--expectations PATH] --receipts-dir PATH

Spec rules owned here: R-021 (E2E surface), R-022 (fail-closed flags,
allow_abbrev=False), R-033 (vacuous-input typed errors), R-037 (exit-code
contract).
Spec: .correctless/specs/camera-ready-aims-evidence.md
"""
from __future__ import annotations

import sys
from pathlib import Path

# Direct-path invocation bootstraps the repo root onto sys.path (repo
# convention: standalone entrypoints force the repo root to sys.path[0]),
# so `python reproducibility/colm_aims_2026/verify.py` behaves identically
# to the documented module-run form (R-037).
if __package__ in (None, ""):  # pragma: no cover - exercised via subprocess
    _REPO_ROOT = str(Path(__file__).resolve().parents[2])
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)

import argparse

from reproducibility.colm_aims_2026 import render, schema
from reproducibility.colm_aims_2026 import verifier as verifier_mod

# R-037 pinned exit codes (DECISION: 0/1/2/3).
EXIT_PASS = 0
EXIT_GATE_FAIL = 1
EXIT_USAGE_ERROR = 2
EXIT_INGRESS_ERROR = 3

_PASS_VERDICTS = (
    verifier_mod.VERDICT_SOURCE_PASS,
    verifier_mod.VERDICT_RELEASE_PASS,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m reproducibility.colm_aims_2026.verify",
        description=(
            "Two-mode fail-closed verifier for the constructed QA reference"
            " evidence package (source-contract ceiling PASS_SOURCE_ONLY;"
            " release mode requires anchored expectations)."
        ),
        # R-022: abbreviated flag forms cannot smuggle past the
        # unknown-flag check; unknown flags are a usage error, never a no-op.
        allow_abbrev=False,
    )
    parser.add_argument(
        "--mode", required=True, choices=("source", "release"),
        help="verifier mode",
    )
    parser.add_argument(
        "--tree", required=True, help="verified artifact tree root"
    )
    parser.add_argument(
        "--receipts-dir",
        required=True,
        help="receipt output directory (outside the verified tree)",
    )
    parser.add_argument(
        "--expectations",
        default=None,
        help=(
            "independently anchored expectations file located outside the"
            " verified tree (required in release mode)"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; returns the pinned exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)  # usage errors exit EXIT_USAGE_ERROR (2)
    try:
        report = verifier_mod.run_verifier(
            Path(args.tree),
            mode=args.mode,
            receipts_dir=Path(args.receipts_dir),
            expectations=(
                Path(args.expectations) if args.expectations else None
            ),
        )
    except (
        verifier_mod.VacuousInputError,
        schema.TypedIngressError,
        schema.EmptyEvaluationError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_INGRESS_ERROR
    except schema.ColmAimsError as exc:
        # Containment/config violations (R-013/R-022) are usage errors.
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE_ERROR
    print(render.render_summary(report))
    return EXIT_PASS if report.verdict in _PASS_VERDICTS else EXIT_GATE_FAIL


if __name__ == "__main__":  # pragma: no cover - exercised via subprocess tests
    raise SystemExit(main())
