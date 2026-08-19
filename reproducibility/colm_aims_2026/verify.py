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
import re

from reproducibility.colm_aims_2026 import render, schema
from reproducibility.colm_aims_2026 import verifier as verifier_mod

# R-037 pinned exit codes (DECISION: 0/1/2/3).
EXIT_PASS = 0
EXIT_GATE_FAIL = 1
EXIT_USAGE_ERROR = 2
EXIT_INGRESS_ERROR = 3
# QA-019: internal (non-ingress) defects get their own pinned code so they
# can never be mistaken for a typed-ingress refusal or a gate FAIL.
EXIT_INTERNAL_ERROR = 4

_ABSOLUTE_PATH_TOKEN = re.compile(r"(?:/[^\s'\":,;]+)+")

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
    supplied_paths = [
        p for p in (args.tree, args.receipts_dir, args.expectations) if p
    ]
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
    except (schema.ConfigSurfaceError, schema.ColmAimsError) as exc:
        # Containment/config violations (R-013/R-022, QA-009) are usage
        # errors (exit 2) — unknown config keys included.
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE_ERROR
    except Exception as exc:  # noqa: BLE001 - QA-006/QA-019 last resort
        # QA-006/QA-019: an internal (non-ingress) defect gets its own
        # pinned exit code, a traceback-free line, and the exception message
        # scrubbed of every supplied/resolved path form (tree-relative
        # basenames only) — diagnostics without leakage.
        print(
            f"error: unexpected {exc.__class__.__name__} during"
            " verification; no verdict was reached:"
            f" {_scrub_paths(str(exc), supplied_paths)}",
            file=sys.stderr,
        )
        return EXIT_INTERNAL_ERROR

    # QA-019: the verdict is reached — rendering happens OUTSIDE the
    # verification try so a render defect can never convert a gate result
    # into another exit code.
    exit_code = (
        EXIT_PASS if report.verdict in _PASS_VERDICTS else EXIT_GATE_FAIL
    )
    try:
        print(render.render_summary(report))
    except Exception:  # noqa: BLE001 - render must never mask the verdict
        print(f"verdict: {report.verdict}")
    return exit_code


def _scrub_paths(message: str, supplied: list[str]) -> str:
    """QA-019: scrub every supplied path (and its resolved form) down to its
    basename, then collapse any residual absolute-path token (R-026)."""
    for raw in supplied:
        basename = Path(raw).name or "<path>"
        forms = {raw}
        try:
            forms.add(str(Path(raw).resolve()))
        except OSError:  # pragma: no cover - resolution never load-bearing
            pass
        for form in sorted(forms, key=len, reverse=True):
            if form and form in message:
                message = message.replace(form, basename)
    return _ABSOLUTE_PATH_TOKEN.sub(
        lambda match: Path(match.group(0)).name, message
    )


if __name__ == "__main__":  # pragma: no cover - exercised via subprocess tests
    raise SystemExit(main())
