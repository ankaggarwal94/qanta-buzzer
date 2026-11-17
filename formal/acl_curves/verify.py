"""Verify the analytic curve and opcode bridge projects in a fresh Linux workspace."""

import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

PROJECT = Path(__file__).resolve().parent
MODULES = (
    "Mathlib.Analysis.SpecialFunctions.Pow.Real",
    "Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic",
    "Mathlib.Analysis.SpecialFunctions.ExpDeriv",
    "Mathlib.Tactic.Linarith", "Mathlib.Tactic.Positivity", "Mathlib.Tactic.Ring",
    "Mathlib.Tactic.FieldSimp",
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stage(project, manifest, expected):
    file = PROJECT / manifest
    if digest(file) != expected.lower():
        raise SystemExit(f"reviewed freeze digest mismatch: {manifest}")
    table = json.loads(file.read_text())
    for name, hash_value in table.items():
        if name == "../acl_expansion/Core/Expr.lean":
            dest = project / "Core" / "Expr.lean"
        elif name == "bridge-lakefile.toml":
            dest = project / "lakefile.toml"
        else:
            dest = project / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PROJECT / name, dest)
        if digest(dest) != hash_value:
            raise SystemExit(f"reviewed file differs: {name}")


def fetch_packages(project):
    packages = json.loads((project / "lake-manifest.json").read_text())["packages"]

    def checkout(package):
        target = project / ".lake" / "packages" / package["name"]
        target.mkdir(parents=True)
        def git(*args):
            subprocess.run(["git", *args], cwd=target, check=True)
        git("init", "-q")
        git("remote", "add", "origin", package["url"])
        git("fetch", "-q", "--depth=1", "origin", package["rev"])
        git("checkout", "-q", "--detach", "FETCH_HEAD")
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=target, text=True)
        if head.strip() != package["rev"]:
            raise SystemExit(f"dependency pin mismatch: {package['name']}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(checkout, packages))


def audit(project, lean_prefix, config):
    mod = config["solution_module"]
    names = config["theorem_names"]
    source = [f"import {mod}", *(f"#print axioms {name}" for name in names)]
    (project / "Audit.lean").write_text("\n".join(source) + "\n")
    # The compiler must only load untrusted solution declarations behind the sandbox.
    lean = lean_prefix / "bin" / "lean"
    # Inspect the trusted toolchain executable only. Bind its system dependencies
    # at their loader paths; the remaining libraries live under the frozen prefix.
    dependencies = subprocess.check_output(["ldd", lean], text=True)
    command = ["bwrap", "--die-with-parent", "--unshare-all", "--clearenv",
               "--new-session", "--tmpfs", "/tmp", "--dev", "/dev", "--proc", "/proc",
               "--ro-bind", str(lean_prefix), str(lean_prefix),
               "--ro-bind", str(project), str(project)]
    for name in re.findall(r"(?:=>\s+|^\s*)(/[^\s]+)\s+\(", dependencies, re.MULTILINE):
        library = Path(name)
        if not library.resolve().is_relative_to(lean_prefix):
            command.extend(["--ro-bind", str(library.resolve()), name])
    search = [project / ".lake/build/lib/lean"]
    search.extend(sorted((project / ".lake/packages").glob("*/.lake/build/lib/lean")))
    command.extend(["--chdir", str(project), "--setenv", "HOME", "/tmp",
                    "--setenv", "LEAN_PATH", os.pathsep.join(map(str, search)),
                    str(lean), "Audit.lean"])
    output = subprocess.check_output(command, text=True, cwd=project)
    print(output, flush=True)
    rows = dict(re.findall(r"'([^']+)' depends on axioms: \[([^\]]*)\]", output))
    if set(rows) != set(names) or any(set(text.split(", ")) - set(config["permitted_axioms"]) for text in rows.values()):
        raise SystemExit("missing theorem or unexpected transitive axiom")


def verify(project, prefix, config_name):
    run = subprocess.run(
        [str(prefix / "bin" / "lake"), "comparator", "--config", config_name, "--paranoid"],
        cwd=project, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    print(run.stdout, flush=True)
    if run.returncode or "Your solution is okay!" not in run.stdout:
        raise SystemExit(f"sandboxed comparator failed: {config_name}")
    audit(project, prefix, json.loads((project / config_name).read_text()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-sha256", required=True)
    parser.add_argument("--bridge-freeze-sha256", required=True)
    args = parser.parse_args()
    if "COMPARATOR_BWRAP" in os.environ:
        raise SystemExit("use the default Lake bubblewrap invocation")
    with tempfile.TemporaryDirectory(prefix="acl-analytic-") as folder:
        first = Path(folder) / "curves"
        second = Path(folder) / "bridge"
        stage(first, "freeze.json", args.freeze_sha256)
        stage(second, "bridge-freeze.json", args.bridge_freeze_sha256)
        raw_prefix = subprocess.check_output(["lean", "--print-prefix"], cwd=first, text=True)
        prefix = Path(raw_prefix.strip()).resolve()
        lake = str(prefix / "bin" / "lake")
        version = subprocess.check_output([lake, "--version"], cwd=first, text=True)
        if "(Lean version 4.35.0)" not in version:
            raise SystemExit(f"unexpected Lean toolchain: {version.strip()}")
        fetch_packages(first)
        # Only reviewed configs and pinned upstream dependency code exist in this workspace.
        subprocess.run([lake, "exe", "cache", "get", *MODULES], cwd=first, check=True)
        shutil.copyfile(PROJECT / "Solution.lean", first / "Solution.lean")
        # Snapshot external dependencies before any solution has executed.
        # Hard links are avoided so the two workspaces cannot change each other's artifacts.
        shutil.copytree(first / ".lake" / "packages", second / ".lake" / "packages")
        shutil.copyfile(PROJECT / "BridgeSolution.lean", second / "BridgeSolution.lean")
        verify(first, prefix, "comparator.json")
        verify(second, prefix, "bridge-comparator.json")


if __name__ == "__main__":
    main()
