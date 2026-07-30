#!/usr/bin/env python3
"""Standalone StopDFF bucketed-sweep validator (ACCEPTANCE_CONTRACT.md section 1).

Subcommands:
  validate-spec   SPEC [--require-final-profile]
  validate-adapter ADAPTER_BUNDLE
  validate        RUN_ROOT --backend {local,modal} --adapter-bundle BUNDLE
                          [--require-final-profile] [--require-package]
  self-test       [--work-dir DIR]     # negative mutation suite (synthetic fixtures)

Exit code 0 on success, 1 on failure. No standalone command requires another backend or
a comparison policy.
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from scripts.stopdff_v5 import checker, selftest  # noqa: E402


def _print_result(label: str, result: "checker.CheckResult") -> int:
    if result.passed:
        print(f"{label}: PASS")
        return 0
    print(f"{label}: FAIL", file=sys.stderr)
    for err in result.errors:
        print(f"  - {err}", file=sys.stderr)
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_spec = sub.add_parser("validate-spec")
    p_spec.add_argument("spec", type=Path)
    p_spec.add_argument("--require-final-profile", action="store_true")

    p_adapter = sub.add_parser("validate-adapter")
    p_adapter.add_argument("bundle", type=Path)

    p_val = sub.add_parser("validate")
    p_val.add_argument("run_root", type=Path)
    p_val.add_argument("--backend", choices=["local", "modal"], required=True)
    p_val.add_argument("--adapter-bundle", type=Path, required=True)
    p_val.add_argument("--require-final-profile", action="store_true")
    p_val.add_argument("--require-package", action="store_true")

    p_self = sub.add_parser("self-test")
    p_self.add_argument("--work-dir", type=Path, default=None)

    args = parser.parse_args(argv)

    if args.command == "validate-spec":
        return _print_result("validate-spec",
                             checker.validate_spec(args.spec, require_final_profile=args.require_final_profile))
    if args.command == "validate-adapter":
        return _print_result("validate-adapter", checker.validate_adapter(args.bundle))
    if args.command == "validate":
        return _print_result(
            "validate",
            checker.validate_run(
                args.run_root, backend=args.backend, adapter_bundle=args.adapter_bundle,
                require_final_profile=args.require_final_profile, require_package=args.require_package,
            ),
        )
    if args.command == "self-test":
        work = args.work_dir
        cleanup = False
        if work is None:
            work = Path(tempfile.mkdtemp(prefix="stopdff_selftest_"))
            cleanup = True
        ok, results = selftest.run_self_test(work)
        for r in results:
            status = "OK" if r["ok"] else "UNEXPECTED"
            print(f"[{status}] {r['mutation']} (expected {r['expected']}, checker_passed={r['passed_check']})")
        if cleanup:
            import shutil
            shutil.rmtree(work, ignore_errors=True)
        if ok:
            print("SELF_TEST=PASS (all mutations rejected; baseline valid)")
            return 0
        print("SELF_TEST=FAIL", file=sys.stderr)
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
