"""Verifier CLI. Documented invocation (R-037):

    python -m reproducibility.colm_aims_2026.verify --mode {source,release} \
        (--tree PATH | --runs-root PATH) [--expectations PATH] \
        --receipts-dir PATH

Spec rules owned here: R-021 (E2E surface), R-022 (fail-closed flags,
allow_abbrev=False), R-033 (vacuous-input typed errors), R-037 (exit-code
contract 0/1/2/3/4), R-069 (the --runs-root release entry).
Spec: .correctless/specs/camera-ready-aims-evidence-2.md
"""
from __future__ import annotations

import sys
from pathlib import Path

# Direct-path invocation force-fronts the repo root on sys.path so the gate
# code (which the receipt SHA-stamps) always resolves from THIS repository.
# A membership guard would let a stale checkout earlier on PYTHONPATH supply
# the imported modules: dedupe then insert-at-front makes the repo root
# sys.path[0] exactly once, unconditionally (R-037).
if __package__ in (None, ""):  # pragma: no cover - exercised via subprocess
    _REPO_ROOT = str(Path(__file__).resolve().parents[2])
    sys.path[:] = [entry for entry in sys.path if entry != _REPO_ROOT]
    sys.path.insert(0, _REPO_ROOT)

import argparse
import re

from reproducibility.colm_aims_2026 import render, schema
from reproducibility.colm_aims_2026 import verifier as verifier_mod

# R-037 pinned exit codes (contract): 0 mode-ceiling pass, 1 gate FAIL,
# 2 usage/config error, 3 typed ingress error, 4 internal error.
EXIT_PASS = 0
EXIT_GATE_FAIL = 1
EXIT_USAGE_ERROR = 2
EXIT_INGRESS_ERROR = 3
EXIT_INTERNAL_ERROR = 4

# MA2-001: the residual scrub matches ONLY tokens whose leading "/" sits at
# start-of-string or after whitespace/quote/equals — a genuinely absolute
# path position. Interior slashes (tree-relative paths like
# records/<cell_id>.jsonl, rule citations like R-062/R-020) survive verbatim.
_ABSOLUTE_PATH_TOKEN = re.compile(
    r"(?:(?<=^)|(?<=[\s'\"=]))(?:/[^\s'\":,;]+)"
)

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
        "--mode",
        required=True,
        choices=("source", "release"),
        help="verifier mode",
    )
    parser.add_argument(
        "--tree", default=None, help="verified artifact tree root"
    )
    parser.add_argument(
        "--runs-root",
        default=None,
        help=(
            "runs site root for release-mode canonical selection via the"
            " ledger pointer (mutually exclusive with --tree)"
        ),
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
    if args.tree is not None and args.runs_root is not None:
        print(
            "error: --tree and --runs-root are mutually exclusive (R-069)",
            file=sys.stderr,
        )
        return EXIT_USAGE_ERROR
    if args.tree is None and args.runs_root is None:
        print(
            "error: one of --tree or --runs-root is required",
            file=sys.stderr,
        )
        return EXIT_USAGE_ERROR
    if args.mode == "release" and args.expectations is None:
        print(
            "error: release mode requires an independently anchored"
            " --expectations file (R-013)",
            file=sys.stderr,
        )
        return EXIT_USAGE_ERROR
    if args.runs_root is not None and args.mode != "release":
        print(
            "error: --runs-root is a release-mode canonical-selection entry"
            " (R-069)",
            file=sys.stderr,
        )
        return EXIT_USAGE_ERROR
    supplied_paths = [
        p
        for p in (args.tree, args.runs_root, args.receipts_dir, args.expectations)
        if p
    ]
    try:
        if args.runs_root is not None:
            report = verifier_mod.run_release_over_runs_root(
                Path(args.runs_root),
                expectations=Path(args.expectations),
                receipts_dir=Path(args.receipts_dir),
            )
        else:
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
        # QA2-002: every user-facing error branch is path-scrubbed (R-026),
        # not just the internal-error branch.
        print(
            f"error: {_scrub_paths(str(exc), supplied_paths)}",
            file=sys.stderr,
        )
        return EXIT_INGRESS_ERROR
    except (schema.ConfigSurfaceError, schema.ColmAimsError) as exc:
        # Containment/config violations are usage errors (exit 2) — unknown
        # config keys included (R-022). Path-scrubbed per QA2-002.
        print(
            f"error: {_scrub_paths(str(exc), supplied_paths)}",
            file=sys.stderr,
        )
        return EXIT_USAGE_ERROR
    except Exception as exc:  # noqa: BLE001 - pinned internal-error code
        # An internal (non-ingress) defect gets its own pinned exit code, a
        # traceback-free line, and a message scrubbed of every supplied or
        # resolved path form (R-026/R-037).
        print(
            f"error: unexpected {exc.__class__.__name__} during"
            " verification; no verdict was reached:"
            f" {_scrub_paths(str(exc), supplied_paths)}",
            file=sys.stderr,
        )
        return EXIT_INTERNAL_ERROR

    exit_code = (
        EXIT_PASS if report.verdict in _PASS_VERDICTS else EXIT_GATE_FAIL
    )
    # The verdict is reached — rendering happens OUTSIDE the verification
    # try so a render defect can never convert a gate result into another
    # exit code.
    try:
        print(render.render_summary(report))
    except Exception:  # noqa: BLE001 - render must never mask the verdict
        print(f"verdict: {report.verdict}")
    return exit_code


def _scrub_paths(message: str, supplied: list[str]) -> str:
    """Scrub every supplied path (and its resolved form) down to its
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


if __name__ == "__main__":  # pragma: no cover - exercised via subprocess
    raise SystemExit(main())
