"""Create a non-authorizing QA-012 diagnostic inventory.

This command executes the frozen five-prong R-072 scan and writes its closed
diagnostic manifest create-once.  Its output is never QA-012 closure authority;
only the independently reviewed authority pinned by :mod:`qa012` can satisfy
``CAMERA_READY_CLOSURE``.

Example
-------
``python -m reproducibility.colm_aims_2026.qa012_inventory`` followed by one
``--prong NAME=PATH`` argument for each name in ``qa012.REQUIRED_SCOPE_PRONGS``
and ``--out qa012-diagnostic.json``.
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from . import qa012, schema
from .phase4_finalize_release import (
    _CapturedDirectoryChain,
    _capture_directory_chain,
    _DirectoryAnchor,
)

EXIT_OK = 0
EXIT_USAGE_ERROR = 2
EXIT_INGRESS_ERROR = 3
EXIT_INTERNAL_ERROR = 4
NON_AUTHORIZING_LABEL = "NON_AUTHORIZING_DIAGNOSTIC"
_DirectoryChain = tuple[tuple[str, tuple[int, int, int]], ...]


@dataclass(frozen=True)
class _OutputPlan:
    """Initial canonical identities for one create-once output slot."""

    destination: Path
    parent: Path
    parent_chain: _CapturedDirectoryChain
    resolved_parent: Path
    resolved_roots: tuple[tuple[str, Path], ...]
    root_chains: tuple[tuple[str, _DirectoryChain], ...]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m reproducibility.colm_aims_2026.qa012_inventory",
        description=(
            "Create a non-authorizing QA-012 diagnostic manifest. Supply"
            " exactly one NAME=PATH value for every frozen scope prong."
        ),
        allow_abbrev=False,
    )
    parser.add_argument(
        "--prong",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help=(
            "frozen scope root; repeat exactly once for each of: "
            + ", ".join(qa012.REQUIRED_SCOPE_PRONGS)
        ),
    )
    parser.add_argument(
        "--out",
        required=True,
        help="new JSON destination; existing paths are never overwritten",
    )
    return parser


def _parse_prong_specs(specs: list[str]) -> dict[str, Path]:
    roots: dict[str, Path] = {}
    allowed = set(qa012.REQUIRED_SCOPE_PRONGS)
    for spec in specs:
        name, separator, raw_path = spec.partition("=")
        if not separator or not name or not raw_path:
            raise schema.ConfigSurfaceError(
                "--prong values must use the exact NAME=PATH form"
            )
        if name not in allowed:
            raise schema.ConfigSurfaceError(
                f"unknown QA-012 scope prong {name!r}; expected exactly"
                f" {list(qa012.REQUIRED_SCOPE_PRONGS)!r}"
            )
        if name in roots:
            raise schema.ConfigSurfaceError(
                f"duplicate QA-012 scope prong {name!r}"
            )
        lexical = Path(raw_path)
        if any(part in {".", ".."} for part in lexical.parts):
            raise schema.ConfigSurfaceError(
                f"QA-012 scope prong {name!r} uses an aliased path spelling"
            )
        roots[name] = lexical
    missing = [name for name in qa012.REQUIRED_SCOPE_PRONGS if name not in roots]
    if missing:
        raise schema.ConfigSurfaceError(
            f"missing QA-012 scope prong(s) {missing!r}"
        )
    return roots


def _validate_roots(roots: dict[str, Path]) -> dict[str, Path]:
    """Require five located, ordinary, distinct, non-overlapping roots."""
    lexical_roots: dict[str, Path] = {}
    resolved_roots: dict[str, Path] = {}
    for name in qa012.REQUIRED_SCOPE_PRONGS:
        lexical = Path(os.path.abspath(roots[name]))
        schema.stable_directory_chain(lexical, lexical)
        if not lexical.is_dir() or schema.is_filesystem_link(lexical):
            raise schema.TypedIngressError(
                f"QA-012 scope prong {name!r} is not an ordinary directory"
            )
        resolved = lexical.resolve(strict=True)
        for other_name, other in resolved_roots.items():
            if (
                resolved == other
                or resolved in other.parents
                or other in resolved.parents
            ):
                raise schema.ConfigSurfaceError(
                    "QA-012 scope roots must be distinct and non-overlapping;"
                    f" {name!r} overlaps {other_name!r}"
                )
        lexical_roots[name] = lexical
        resolved_roots[name] = resolved
    return lexical_roots


def _validate_output(output: Path, roots: dict[str, Path]) -> _OutputPlan:
    """Validate a new destination before any potentially expensive scan."""
    raw = Path(output)
    if any(part in {".", ".."} for part in raw.parts):
        raise schema.ConfigSurfaceError(
            "QA-012 diagnostic output uses an aliased path spelling"
        )
    destination = Path(os.path.abspath(raw))
    if os.path.lexists(destination):
        raise FileExistsError("QA-012 diagnostic output already exists")
    parent = destination.parent
    schema.stable_directory_chain(parent, parent)
    if not parent.is_dir() or schema.is_filesystem_link(parent):
        raise schema.TypedIngressError(
            "QA-012 diagnostic output parent is not an ordinary directory"
        )
    parent_chain = _capture_directory_chain(parent)
    resolved_parent = parent.resolve(strict=True)
    resolved_destination = resolved_parent / destination.name
    resolved_roots: list[tuple[str, Path]] = []
    root_chains: list[tuple[str, _DirectoryChain]] = []
    for name, root in roots.items():
        root_chains.append((name, schema.stable_directory_chain(root, root)))
        resolved_root = root.resolve(strict=True)
        resolved_roots.append((name, resolved_root))
        if (
            resolved_destination == resolved_root
            or resolved_root in resolved_destination.parents
        ):
            raise schema.ConfigSurfaceError(
                "QA-012 diagnostic output must be outside every scanned root;"
                f" destination is inside {name!r}"
            )
    return _OutputPlan(
        destination=destination,
        parent=parent,
        parent_chain=parent_chain,
        resolved_parent=resolved_parent,
        resolved_roots=tuple(resolved_roots),
        root_chains=tuple(root_chains),
    )


def _revalidate_output(
    plan: _OutputPlan,
    roots: dict[str, Path],
    *,
    require_unclaimed: bool,
) -> None:
    """Recheck output ancestry and root containment around publication."""
    current_chain = _capture_directory_chain(plan.parent)
    try:
        current_parent = plan.parent.resolve(strict=True)
    except OSError as exc:
        raise schema.TypedIngressError(
            "QA-012 diagnostic output parent changed during inventory"
        ) from exc
    if current_chain != plan.parent_chain or current_parent != plan.resolved_parent:
        raise schema.TypedIngressError(
            "QA-012 diagnostic output parent changed during inventory"
            " (R-072/R-013/R-020)"
        )
    resolved_destination = current_parent / plan.destination.name
    expected_roots = dict(plan.resolved_roots)
    expected_root_chains = dict(plan.root_chains)
    for name, root in roots.items():
        current_root_chain = schema.stable_directory_chain(root, root)
        try:
            resolved_root = root.resolve(strict=True)
        except OSError as exc:
            raise schema.TypedIngressError(
                f"QA-012 scope prong {name!r} changed during inventory"
            ) from exc
        if (
            resolved_root != expected_roots[name]
            or current_root_chain != expected_root_chains[name]
        ):
            raise schema.TypedIngressError(
                f"QA-012 scope prong {name!r} changed during inventory"
                " (R-072/R-013/R-020)"
            )
        if (
            resolved_destination == resolved_root
            or resolved_root in resolved_destination.parents
        ):
            raise schema.ConfigSurfaceError(
                "QA-012 diagnostic output moved inside a scanned root"
            )
    if require_unclaimed and os.path.lexists(plan.destination):
        raise FileExistsError("QA-012 diagnostic output already exists")


def main(argv: list[str] | None = None) -> int:
    """Run the bounded inventory and create one diagnostic JSON manifest."""
    parser = _build_parser()
    try:
        args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else EXIT_USAGE_ERROR

    try:
        roots = _validate_roots(_parse_prong_specs(args.prong))
        output_plan = _validate_output(Path(args.out), roots)
        manifest = qa012.build_inventory_manifest(roots)
        qa012.validate_inventory_manifest(manifest)
        encoded = schema.encode_json(manifest)
        with _DirectoryAnchor(
            output_plan.parent,
            output_plan.parent_chain,
            "QA-012 diagnostic output parent",
        ) as output_anchor:
            _revalidate_output(output_plan, roots, require_unclaimed=True)
            output_anchor.create_once(
                output_plan.destination.name,
                encoded,
                exists_label="QA-012 diagnostic manifest",
                mode=0o666,
            )
            _revalidate_output(output_plan, roots, require_unclaimed=False)
            observed = output_anchor.read_regular(
                output_plan.destination.name,
                max_bytes=qa012.MAX_QA_TOTAL_BYTES,
            )
            _revalidate_output(output_plan, roots, require_unclaimed=False)
        if observed != encoded:
            raise schema.TypedIngressError(
                "QA-012 diagnostic output differs from the validated bytes"
            )
    except (schema.ConfigSurfaceError, FileExistsError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE_ERROR
    except schema.TypedIngressError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_INGRESS_ERROR
    except (schema.ColmAimsError, OSError, ValueError) as exc:
        print(
            f"error: QA-012 diagnostic inventory failed closed"
            f" ({exc.__class__.__name__})",
            file=sys.stderr,
        )
        return EXIT_INGRESS_ERROR
    except Exception as exc:  # noqa: BLE001 - stable internal-error boundary
        print(
            "error: unexpected"
            f" {exc.__class__.__name__} during QA-012 diagnostic inventory",
            file=sys.stderr,
        )
        return EXIT_INTERNAL_ERROR

    print(
        f"{NON_AUTHORIZING_LABEL}: result={manifest['result']}"
        f" inventory_sha256={manifest['inventory_sha256']}"
        f" output={output_plan.destination.name}"
    )
    print(
        "This diagnostic does not satisfy CAMERA_READY_CLOSURE; only the"
        " independently pinned QA-012 authority can authorize closure."
    )
    return EXIT_OK


if __name__ == "__main__":  # pragma: no cover - exercised through -m
    raise SystemExit(main())
