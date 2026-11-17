"""Rebuild reviewed statements and check all solutions inside Lake's Linux sandbox."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


PROJECT = Path(__file__).resolve().parent
CONFIGS = ("comparator.json", "order-comparator.json", "transition-comparator.json")
SOLUTIONS = ("Solution.lean", "OrderSolution.lean", "TransitionSolution.lean")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def comparator(project, config):
    return subprocess.run(
        ["lake", "comparator", "--config", config, "--paranoid"],
        cwd=project, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        check=False,
    )

def check_supplement(folder, expected_freeze, label, solutions):
    freeze = PROJECT / f"{label}-freeze.json"
    if sha256(freeze) != expected_freeze.lower():
        raise SystemExit(f"reviewed {label} freeze digest mismatch")
    folder.mkdir()
    for name, digest in json.loads(freeze.read_text()).items():
        dest = folder / ("lakefile.toml" if name == f"{label}-lakefile.toml" else name)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PROJECT / name, dest)
        if sha256(dest) != digest:
            raise SystemExit(f"reviewed {label} input changed: {name}")
    for name in solutions:
        shutil.copyfile(PROJECT / name, folder / name)
    result = comparator(folder, f"{label}-comparator.json")
    print(result.stdout, flush=True)
    if result.returncode or "Your solution is okay!" not in result.stdout:
        raise SystemExit(f"{label} comparator failed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-sha256", required=True,
                        help="independently retained SHA-256 of freeze.json")
    parser.add_argument("--negative-control", action="store_true",
                        help="also test rejection of an altered solution-side definition")
    parser.add_argument("--law-freeze-sha256", help="also check the reviewed finite sampling law")
    parser.add_argument("--wire-freeze-sha256", help="also check the reviewed opcode serialization")
    args = parser.parse_args()
    freeze = PROJECT / "freeze.json"
    if sha256(freeze) != args.freeze_sha256.lower():
        raise SystemExit("reviewed freeze digest mismatch")
    if "COMPARATOR_BWRAP" in os.environ:
        raise SystemExit("run with the pinned Lake default bubblewrap invocation")
    expected = json.loads(freeze.read_text())
    with tempfile.TemporaryDirectory(prefix="acl-proof-") as directory:
        clean = Path(directory)
        for name in (*expected, *SOLUTIONS):
            shutil.copyfile(PROJECT / name, clean / name)
        for name, digest in expected.items():
            if sha256(clean / name) != digest:
                raise SystemExit(f"reviewed input changed: {name}")
        version = subprocess.check_output(["lake", "--version"], text=True)
        if "(Lean version 4.35.0)" not in version:
            raise SystemExit(f"unexpected toolchain: {version.strip()}")
        for config in CONFIGS:
            run = comparator(clean, config)
            print(run.stdout, flush=True)
            if run.returncode or "Your solution is okay!" not in run.stdout:
                raise SystemExit(f"comparator failed: {config}")
        if args.law_freeze_sha256:
            check_supplement(clean / "sampler", args.law_freeze_sha256, "law",
                             ("Solution.lean", "LawSolution.lean"))
        if args.wire_freeze_sha256:
            check_supplement(clean / "wire", args.wire_freeze_sha256, "wire",
                             ("WireSolution.lean",))
        if args.negative_control:
            # Preserve the reviewed challenge. Only the solution uses this definition.
            source = (clean / "Core.lean").read_text()
            altered = source.replace("lo + (hi-lo)*weight", "lo + weight*(hi-lo)")
            if altered == source:
                raise SystemExit("negative control did not alter its target")
            (clean / "AlternateCore.lean").write_text(altered)
            solution = clean / "Solution.lean"
            solution.write_text(solution.read_text().replace("import Core", "import AlternateCore"))
            with (clean / "lakefile.toml").open("a") as handle:
                handle.write('\n[[lean_lib]]\nname = "AlternateCore"\n')
            run = comparator(clean, CONFIGS[0])
            if run.returncode == 0 or "Const does not match between challenge and target 'ACL.prize'" not in run.stdout:
                raise SystemExit(f"unexpected negative-control result:\n{run.stdout}")
            print("Rejected altered ACL.prize definition")


if __name__ == "__main__":
    main()
