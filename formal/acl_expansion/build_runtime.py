"""Build and exercise the pinned production Lean executable inside offline bubblewrap."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

PROJECT = Path(__file__).resolve().parent


def sha256(path):
    with path.open("rb") as src:
        return hashlib.file_digest(src, "sha256").hexdigest()


def sandbox(project, prefix, command, *, build=False, input_text=None):
    args = [
        "bwrap", "--die-with-parent", "--unshare-all", "--clearenv",
        "--ro-bind", "/", "/", "--tmpfs", "/tmp", "--tmpfs", "/home",
        "--tmpfs", "/root", "--tmpfs", "/mnt", "--tmpfs", "/run", "--tmpfs", "/var",
        "--proc", "/proc", "--dev", "/dev", "--ro-bind", str(prefix), str(prefix),
        "--ro-bind", str(project), str(project),
    ]
    if build:
        args += ["--bind", str(project / ".lake"), str(project / ".lake")]
    args += [
        "--chdir", str(project), "--setenv", "HOME", "/tmp",
        "--setenv", "PATH", f"{prefix}/bin:{os.defpath}", "--setenv", "LC_ALL", "C",
        *map(str, command),
    ]
    result = subprocess.run(args, cwd=project, input=input_text, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(result.stderr + result.stdout)
    return result.stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-sha256", required=True)
    parser.add_argument("--install", action="store_true",
                        help="install the verified ELF in the local Lake build directory")
    args = parser.parse_args()
    freeze = PROJECT / "runtime-freeze.json"
    if sha256(freeze) != args.freeze_sha256.lower():
        raise SystemExit("reviewed runtime freeze digest mismatch")
    expected = json.loads(freeze.read_bytes())
    with tempfile.TemporaryDirectory(prefix="acl-runtime-build-") as tmp:
        clean = Path(tmp)
        for name, source_hash in expected.items():
            dest = clean / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(PROJECT / name, dest)
            if sha256(dest) != source_hash:
                raise SystemExit(f"runtime input differs: {name}")
        (clean / ".lake").mkdir()
        prefix = Path(subprocess.check_output(["lean", "--print-prefix"], cwd=clean, text=True).strip())
        lake = prefix / "bin" / "lake"
        version = sandbox(clean, prefix, [lake, "--version"]).strip()
        if "(Lean version 4.35.0)" not in version:
            raise SystemExit(f"unexpected version: {version}")
        sandbox(clean, prefix, [lake, "build", "acl_runtime"], build=True)
        binary = clean / ".lake" / "build" / "bin" / "acl_runtime"
        with binary.open("rb") as exe:
            if exe.read(4) != b"\x7fELF":
                raise SystemExit("expected Linux ELF binary")

        checks = [
            ({"op": "draw", "n": 4, "digits": [1, 2]}, {"value": [1, 3]}),
            ({"op": "draw", "n": 2, "digits": [1, 1]}, {"error": "invalid draw"}),
            ({"op": "insert", "xs": [1, 3], "gold": 4, "pos": 1}, {"value": [1, 4, 3]}),
            ({"op": "schedule", "values": [[1, 1], [1, 2]], "first": [1, 1], "last": [1, 2]},
             {"value": True}),
            ({"op": "formula", "arm": "baseline", "knots": [], "start": [1, 1], "end": [0, 1],
              "shape": [1, 1], "threshold": [1, 2], "baseline": [-1, 4]}, {"value": ["q", -1, 4]}),
            ({"op": "transition", "base": [1, 4], "gamma": [1, 2], "current": [1, 4],
              "successor": [2, 1], "prev": [(2**53)-1, 2**53], "rounded": [1, 1],
              "next": [(2**52)+1, 2**52], "even": True}, {"value": True}),
        ]
        payload = "".join(json.dumps(request) + "\n" for request, _ in checks)
        response = sandbox(clean, prefix, [binary], input_text=payload)
        observed = [json.loads(line) for line in response.splitlines()]
        if observed != [expected_value for _, expected_value in checks]:
            raise SystemExit(f"unexpected production probe responses: {response}")
        binary_hash = sha256(binary)
        if args.install:
            target = PROJECT / ".lake" / "build" / "bin" / "acl_runtime"
            target.parent.mkdir(parents=True, exist_ok=True)
            fd, stage_name = tempfile.mkstemp(prefix=".acl-runtime-", dir=target.parent)
            os.close(fd)
            stage = Path(stage_name)
            try:
                shutil.copy2(binary, stage)
                if sha256(stage) != binary_hash:
                    raise SystemExit("binary changed during installation")
                stage.replace(target)
            finally:
                stage.unlink(missing_ok=True)
        print(json.dumps({
            "version": version, "format": "ELF", "sandbox": "offline bubblewrap",
            "freeze_sha256": args.freeze_sha256, "binary_sha256": binary_hash,
            "installed": args.install, "probe_responses": observed,
        }, indent=2))


if __name__ == "__main__":
    main()
