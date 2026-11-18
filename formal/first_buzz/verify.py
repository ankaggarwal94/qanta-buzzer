"""Compare the proof at a specific Git commit with the reviewed FirstBuzz challenge."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile


def git(repo: Path, *args: str) -> bytes:
    """Run Git with replacement objects disabled.

    Parameters
    ----------
    repo : Path
        Repository in which to run Git.
    *args : str
        Git subcommand and its arguments.

    Returns
    -------
    bytes
        Standard output from the completed command, including any trailing newline.
    """
    return subprocess.check_output(["git", "--no-replace-objects", "-C", str(repo), *args])


def stage_project(repo: Path, revision: str, freeze_sha256: str, project: Path) -> None:
    """Stage the committed proof and its reviewed comparison project.

    Parameters
    ----------
    repo : Path
        Git repository containing the challenge and submitted proof.
    revision : str
        Full Git commit object ID from which to read every input.
    freeze_sha256 : str
        Independently reviewed SHA-256 digest of the challenge manifest.
    project : Path
        Empty or new directory to receive the verified inputs and proof.

    Raises
    ------
    ValueError
        If the commit, manifest, frozen files, or destination fail validation.
    subprocess.CalledProcessError
        If Git cannot read the requested commit or one of its blobs.
    OSError
        If Git cannot start or the project directory cannot be prepared or written.
    """
    if not re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", revision):
        raise ValueError("supply a full commit object ID")
    if git(repo, "cat-file", "-t", revision).strip() != b"commit":
        raise ValueError("revision must identify a commit")

    def blob(path: str) -> bytes:
        return git(repo, "cat-file", "blob", f"{revision}:{path}")

    root = "formal/first_buzz"
    manifest = blob(f"{root}/freeze.json")
    if hashlib.sha256(manifest).hexdigest() != freeze_sha256:
        raise ValueError("reviewed freeze digest mismatch")
    files = json.loads(manifest)
    if set(files) != {
        "Challenge.lean", "comparator.json", "lake-manifest.json", "lakefile.toml", "lean-toolchain",
    }:
        raise ValueError("unexpected frozen project layout")
    project.mkdir(parents=True, exist_ok=True)
    if any(project.iterdir()):
        raise ValueError("the comparison directory must be empty")
    for name, expected in files.items():
        data = blob(f"{root}/{name}")
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f"reviewed file differs: {name}")
        (project / name).write_bytes(data)
    proof = blob("formal/FirstBuzz.lean")
    (project / "FirstBuzz.lean").write_bytes(proof)
    print(f"commit={revision} freeze_sha256={freeze_sha256} "
          f"solution_sha256={hashlib.sha256(proof).hexdigest()}", flush=True)


def compare(project: Path) -> None:
    """Run the frozen FirstBuzz comparison using the pinned Lean toolchain.

    Parameters
    ----------
    project : Path
        Directory populated by `stage_project`, including the comparator config.

    Raises
    ------
    ValueError
        If the sandbox is overridden or Lake reports an unexpected Lean version.
    subprocess.CalledProcessError
        If Lake fails to report its version or the proof comparison fails.
    """
    if "COMPARATOR_BWRAP" in os.environ:
        raise ValueError("the default Lake bubblewrap sandbox is required")
    version = subprocess.check_output(["lake", "--version"], cwd=project, text=True).strip()
    if "(Lean version 4.35.0)" not in version:
        raise ValueError(f"unexpected toolchain: {version}")
    print(version, flush=True)
    # Lake checks the frozen declarations and their transitive assumptions.
    # The candidate module is only loaded by the comparator inside its sandbox.
    subprocess.run(["lake", "comparator", "--config", "comparator.json", "--paranoid"],
                   cwd=project, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--revision", required=True)
    parser.add_argument("--freeze-sha256", required=True)
    args = parser.parse_args()
    try:
        with tempfile.TemporaryDirectory(prefix="first-buzz-") as directory:
            project = Path(directory)
            stage_project(args.repo, args.revision, args.freeze_sha256, project)
            compare(project)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"FirstBuzz verification failed: {error}\n")


if __name__ == "__main__":
    main()
