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
from datetime import datetime, timezone

from reproducibility.colm_aims_2026 import receipt as receipt_mod
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


def _argv_option(argv: list[str], name: str) -> str | None:
    """Return the last exact option value without accepting abbreviations."""
    value = None
    for index, token in enumerate(argv):
        if token == name and index + 1 < len(argv):
            value = argv[index + 1]
        elif token.startswith(f"{name}="):
            value = token.partition("=")[2]
    return value


def _emit_cli_failure(
    *, mode: str | None, tree: str | None, receipts_dir: str | None, observed: str
) -> None:
    """Best-effort create-once failure evidence when a receipt path is known."""
    if not receipts_dir:
        return
    receipts = Path(receipts_dir)
    verified_tree = Path(tree) if tree else receipts.parent / ".no-verified-tree"
    payload = {
        "schema_version": schema.SCHEMA_VERSION,
        "mode": mode if mode in {"source", "release"} else "unknown",
        "verdict": verifier_mod.VERDICT_FAIL,
        "legs": [
            verifier_mod._fail(
                verifier_mod.LEG_TYPED_INGRESS,
                expected="valid CLI configuration and completed verification",
                observed=observed,
            )
        ],
        "validated_artifacts": [],
        "classifications": {},
        "input_tree_sha256": None,
        "expectations_anchor_sha256": None,
        "verifier_code_sha256": verifier_mod._code_digest(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    try:
        receipt_mod.emit_receipt(
            payload, receipts_dir=receipts, verified_tree=verified_tree
        )
    except (OSError, schema.ColmAimsError):
        # An unusable or in-tree receipt destination cannot safely receive
        # evidence; retain the pinned exit code without masking the cause.
        return


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
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = _build_parser()
    try:
        args = parser.parse_args(raw_argv)
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else EXIT_USAGE_ERROR
        if code != 0:
            _emit_cli_failure(
                mode=_argv_option(raw_argv, "--mode"),
                tree=_argv_option(raw_argv, "--tree"),
                receipts_dir=_argv_option(raw_argv, "--receipts-dir"),
                observed="CLI argument parsing failed",
            )
        return code
    receipts_path = Path(args.receipts_dir)
    receipts_before = (
        set(receipts_path.glob("receipt-*.json"))
        if receipts_path.is_dir()
        else set()
    )

    def emit_if_needed(observed: str) -> None:
        current = (
            set(receipts_path.glob("receipt-*.json"))
            if receipts_path.is_dir()
            else set()
        )
        if current == receipts_before:
            _emit_cli_failure(
                mode=args.mode,
                tree=args.tree,
                receipts_dir=args.receipts_dir,
                observed=observed,
            )

    if args.tree is not None and args.runs_root is not None:
        print(
            "error: --tree and --runs-root are mutually exclusive (R-069)",
            file=sys.stderr,
        )
        emit_if_needed("--tree and --runs-root are mutually exclusive")
        return EXIT_USAGE_ERROR
    if args.tree is None and args.runs_root is None:
        print(
            "error: one of --tree or --runs-root is required",
            file=sys.stderr,
        )
        emit_if_needed("one of --tree or --runs-root is required")
        return EXIT_USAGE_ERROR
    if args.mode == "release" and args.expectations is None:
        print(
            "error: release mode requires an independently anchored"
            " --expectations file (R-013)",
            file=sys.stderr,
        )
        emit_if_needed("release mode requires --expectations")
        return EXIT_USAGE_ERROR
    if args.runs_root is not None and args.mode != "release":
        print(
            "error: --runs-root is a release-mode canonical-selection entry"
            " (R-069)",
            file=sys.stderr,
        )
        emit_if_needed("--runs-root requires release mode")
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
        emit_if_needed(f"{exc.__class__.__name__}: {exc}")
        return EXIT_INGRESS_ERROR
    except (schema.ConfigSurfaceError, schema.ColmAimsError) as exc:
        # Containment/config violations are usage errors (exit 2) — unknown
        # config keys included (R-022). Path-scrubbed per QA2-002.
        print(
            f"error: {_scrub_paths(str(exc), supplied_paths)}",
            file=sys.stderr,
        )
        emit_if_needed(f"{exc.__class__.__name__}: {exc}")
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
        emit_if_needed(f"unexpected {exc.__class__.__name__}: {exc}")
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
