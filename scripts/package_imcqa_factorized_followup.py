#!/usr/bin/env python3
"""Archive a self-contained, hash-verified factorized stopping evidence package."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import zipfile


CHUNK = 1 << 20
EXCLUDED_NAMES = {"workspace.json", ".env", ".env.local", "credentials.json", "token.json"}
FORBIDDEN_SUFFIXES = {".safetensors", ".bin", ".pt", ".pth", ".pkl", ".pem", ".key"}


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK), b""):
            result.update(chunk)
    return result.hexdigest()


def valid_name(name: str) -> str:
    if (not name or "\\" in name or PurePosixPath(name).is_absolute()
            or any(part in {"", ".", ".."} for part in name.split("/"))):
        raise ValueError("unsafe relative archive path")
    return name


@dataclass(frozen=True)
class Entry:
    name: str
    path: Path | None = None
    data: bytes | None = None

    def __post_init__(self):
        valid_name(self.name)
        if (self.path is None) == (self.data is None):
            raise ValueError("entry needs exactly one content source")
        if self.path is not None and (not self.path.is_file() or self.path.is_symlink()):
            raise ValueError("archive input must be a regular, non-symlink file")

    def manifest_row(self) -> dict:
        return {"path": self.name,
                "bytes": self.path.stat().st_size if self.path else len(self.data),
                "sha256": digest(self.path) if self.path else hashlib.sha256(self.data).hexdigest()}


def collect_directory(root: Path, label: str) -> list[Entry]:
    if not root.is_dir() or root.is_symlink():
        raise ValueError("missing or symlinked evidence directory: " + str(root))
    result = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("symlink in evidence: " + str(path))
        if not path.is_file():
            continue
        if path.suffix.lower() in FORBIDDEN_SUFFIXES or path.name.startswith(".env"):
            raise ValueError("weight, credential, or opaque binary file in evidence: " + path.name)
        if path.name in EXCLUDED_NAMES or "__pycache__" in path.parts:
            continue
        result.append(Entry(label + "/" + path.relative_to(root).as_posix(), path=path))
    if not result:
        raise ValueError("empty evidence directory: " + str(root))
    return result


def verify_receipt(directory: Path, *, statuses: set[str], schema: str | None = None) -> dict:
    receipt = json.loads((directory / "analysis_receipt.json").read_bytes())
    if receipt.get("status") not in statuses:
        raise ValueError("analysis completion status differs")
    for name, expected in receipt["output_sha256"].items():
        valid_name(name)
        if digest(directory / name) != expected:
            raise ValueError("analysis output differs from receipt: " + name)
    report = json.loads((directory / "report.json").read_bytes())
    if "report.json" not in receipt["output_sha256"] or (schema and report.get("schema_version") != schema):
        raise ValueError("analysis schema or report binding differs")
    return report


def git(repo: Path, *arguments: str) -> bytes:
    return subprocess.run(["git", "-C", str(repo), *arguments], check=True, capture_output=True).stdout


def validate_commit(value: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", value):
        raise ValueError("an exact committed source SHA is required")
    return value


def source_entries(repo: Path, commit: str, run_roots: list[Path]) -> list[Entry]:
    """Include committed IMCQA modules plus every hash-bound worker dependency."""
    validate_commit(commit)
    tracked = git(repo, "ls-tree", "-r", "--name-only", commit).decode().splitlines()
    paths = {name for name in tracked if "imcqa" in name and name.startswith(
        ("scripts/", "tests/", "configs/", "docs/", ".github/workflows/"))
        and name.endswith((".py", ".json", ".md", ".yml"))}
    paths.add("scripts/__init__.py")
    bindings = {}
    for root in run_roots:
        control = json.loads((root / "control.json").read_bytes())
        for name, expected in control["source_files_sha256"].items():
            valid_name(name)
            if name in bindings and bindings[name] != expected:
                raise ValueError("worker dependencies changed between included runs: " + name)
            bindings[name] = expected
            paths.add(name)
    result = []
    for name in sorted(paths):
        valid_name(name)
        content = git(repo, "show", f"{commit}:{name}")
        if name in bindings and hashlib.sha256(content).hexdigest() != bindings[name]:
            raise ValueError("analysis source commit differs from inference-bound source: " + name)
        result.append(Entry("source/" + name, data=content))
    return result


def validate_execution(execution: dict, *, source_commit: str, binary_launch_commit: str,
                       numerics_launch_commit: str, workflow_urls: list[str],
                       recovery_launch_commit: str | None = None) -> None:
    """Bind the timing/cost ledger to the exact source and inference workflows."""
    if (execution.get("schema_version") != "imcqa-factorized-execution-v1"
            or execution.get("analysis_source_commit") != source_commit
            or execution.get("launch_commit") != binary_launch_commit
            or execution.get("launch_commit") != numerics_launch_commit):
        raise ValueError("execution ledger source or launch identity differs")
    expected_urls = []
    for prefix in ("", "recovery_") if recovery_launch_commit else ("",):
        value = execution.get(prefix + "workflow_url")
        workflow_id = execution.get(prefix + "workflow_id")
        if (type(workflow_id) is not int or workflow_id <= 0
                or value != "https://github.com/ankaggarwal94/qanta-buzzer/actions/runs/" + str(workflow_id)):
            raise ValueError("execution workflow URL or identifier differs")
        expected_urls.append(value)
    if len(workflow_urls) != len(set(workflow_urls)) or set(workflow_urls) != set(expected_urls):
        raise ValueError("declared workflow URLs differ from execution ledger")
    if recovery_launch_commit and execution.get("recovery_launch_commit") != recovery_launch_commit:
        raise ValueError("execution recovery launch identity differs")


def validate_analyzer_sources(source_by_name: dict[str, bytes], analyses: list[tuple[Path, str]]) -> None:
    for directory, filename in analyses:
        expected = json.loads((directory / "analysis_receipt.json").read_bytes())["analyzer_sha256"]
        if hashlib.sha256(source_by_name["source/scripts/" + filename]).hexdigest() != expected:
            raise ValueError("analysis source commit differs from completed analyzer: " + filename)


def reproduction_script(recovery: bool) -> str:
    text = '''#!/usr/bin/env bash
set -euo pipefail
# Run from the extracted archive root using Python 3.12.
# CPU dependencies used in the recorded run: numpy==2.3.5 scipy==1.17.0.
# These commands do not allocate a model worker or perform model inference.
export PYTHONPATH="$PWD/source${PYTHONPATH:+:$PYTHONPATH}"
python -m scripts.analyze_imcqa_factorized_cpu \\
  --plan inputs/factorized_cpu_plan.json \\
  --public inputs/protocol/public.json --prior-public inputs/prior_wait_public.json \\
  --frozen-source inputs/frozen/main_jobs.json --gold inputs/frozen/main_dataset.json \\
  --protocol-config source/configs/imcqa_protocol_pilot.json \\
  --protocol-analysis prior_protocol_analysis \\
  --prior-outputs prior_wait_run/output --outputs prior_protocol_run/output \\
  --out recomputed/cpu
python -m scripts.analyze_imcqa_binary_pilot \\
  --public inputs/binary/public.json --proposal-public inputs/protocol/public.json \\
  --earlier-public inputs/prior_wait_public.json \\
  --frozen-source inputs/frozen/main_jobs.json --gold inputs/frozen/main_dataset.json \\
  --config source/configs/imcqa_binary_pilot.json \\
  --proposal-outputs prior_protocol_run/output --earlier-outputs prior_wait_run/output \\
  --outputs binary_run/output --out recomputed/binary
python -m scripts.imcqa_3b_numerical_diagnostic \\
  --protocol-run-root prior_protocol_run --prior-run-root prior_wait_run \\
  --out recomputed/numerical_manifest.json
python - <<'PYNUM'
import json
from pathlib import Path
from scripts.imcqa_3b_numerical_diagnostic import compare
root = Path("numerics_run/output/qwen3b")
read = lambda name: json.loads((root / name).read_bytes())
plan, reference = read("plan.json"), read("single_reference.json")["outputs"]
for mode in ("cached_2", "cached_4", "cached_8", "uncached_2", "uncached_4", "uncached_8"):
    record = read(mode + ".json")
    assert compare(record["outputs"], reference, plan["jobs"]) == record["comparison"], mode
print("All six recorded numerical mode comparisons recomputed exactly on CPU.")
PYNUM
'''
    if recovery:
        text += '''python -m scripts.analyze_imcqa_protocol_recovery \\
  --public inputs/protocol/public.json --prior-public inputs/prior_wait_public.json \\
  --frozen-source inputs/frozen/main_jobs.json --gold inputs/frozen/main_dataset.json \\
  --config source/configs/imcqa_protocol_pilot.json \\
  --recovery-config source/configs/imcqa_3b_protocol_recovery.json \\
  --prior-outputs prior_wait_run/output --original-outputs prior_protocol_run/output \\
  --recovery-outputs recovery_run/output --diagnostic-run numerics_run \\
  --out recomputed/recovery
'''
    return text


def write_archive(out: Path, entries: list[Entry], metadata: dict) -> dict:
    names = [entry.name for entry in entries]
    if len(set(names)) != len(names) or "artifact_manifest.json" in names:
        raise ValueError("duplicate or reserved archive path")
    manifest = {"schema_version": "imcqa-factorized-followup-package-v1", **metadata,
                "files": [entry.manifest_row() for entry in entries]}
    with zipfile.ZipFile(out, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for entry in entries:
            if entry.path:
                archive.write(entry.path, entry.name)
            else:
                archive.writestr(entry.name, entry.data)
        archive.writestr("artifact_manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    with zipfile.ZipFile(out) as archive:
        if len(archive.namelist()) != len(set(archive.namelist())) or archive.testzip() is not None:
            raise ValueError("ZIP paths or CRC integrity failed")
        for name in archive.namelist():
            valid_name(name)
        for row in manifest["files"]:
            value = hashlib.sha256()
            with archive.open(row["path"]) as stream:
                for chunk in iter(lambda: stream.read(CHUNK), b""):
                    value.update(chunk)
            if archive.getinfo(row["path"]).file_size != row["bytes"] or value.hexdigest() != row["sha256"]:
                raise ValueError("archived bytes differ from manifest: " + row["path"])
    return {"path": str(out.resolve()), "bytes": out.stat().st_size, "sha256": digest(out),
            "files": len(entries) + 1}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cpu-analysis", "binary-analysis", "binary-run", "numerics-run", "prior-protocol-run",
                 "prior-protocol-analysis", "prior-wait-run", "binary-inputs", "protocol-inputs",
                 "prior-public", "frozen-source", "gold", "plan", "report", "execution", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("source-commit", "binary-launch-commit", "numerics-launch-commit"):
        parser.add_argument("--" + name, required=True, type=validate_commit)
    parser.add_argument("--workflow-url", action="append", required=True)
    parser.add_argument("--recovery-run", type=Path)
    parser.add_argument("--recovery-analysis", type=Path)
    parser.add_argument("--recovery-launch-commit", type=validate_commit)
    parser.add_argument("--review-log", "--numerical-review", dest="review_log", type=Path)
    args = parser.parse_args()
    recovery = args.recovery_run is not None
    if recovery != (args.recovery_analysis is not None) or recovery != (args.recovery_launch_commit is not None):
        raise ValueError("recovery run, analysis, and launch commit must be supplied together")
    if any(not url.startswith("https://github.com/ankaggarwal94/qanta-buzzer/actions/runs/")
           or not url.rsplit("/", 1)[-1].isdigit() for url in args.workflow_url):
        raise ValueError("unexpected workflow evidence URL")
    cpu = verify_receipt(args.cpu_analysis, statuses={"complete"}, schema="imcqa-factorized-cpu-analysis-v1")
    binary = verify_receipt(args.binary_analysis, statuses={"complete"}, schema="imcqa-binary-analysis-v1")
    verify_receipt(args.prior_protocol_analysis, statuses={"partial"}, schema="imcqa-protocol-partial-analysis-v1")
    if cpu["n_model_forward_passes"] != 0 or binary["n_score_rows"] != 832:
        raise ValueError("CPU or binary analysis scope differs")
    if digest(args.plan) != cpu["plan_sha256"]:
        raise ValueError("frozen CPU plan differs from analysis")
    binary_inputs = {"public": args.binary_inputs / "public.json",
        "proposal_public": args.protocol_inputs / "public.json", "earlier_public": args.prior_public,
        "frozen_source": args.frozen_source, "gold": args.gold}
    if any(digest(path) != binary["input_sha256"][name] for name, path in binary_inputs.items()):
        raise ValueError("self-contained input differs from analyzed input")
    for run, launch in ((args.binary_run, args.binary_launch_commit), (args.numerics_run, args.numerics_launch_commit)):
        if json.loads((run / "control.json").read_bytes())["source_commit"] != launch:
            raise ValueError("declared launch commit differs from raw control")
    numerical_root = args.numerics_run / "output/qwen3b"
    numerical_receipt = json.loads((numerical_root / "receipt.json").read_bytes())
    if numerical_receipt.get("status") != "complete" or not numerical_receipt.get("reference_valid"):
        raise ValueError("numerical diagnostic is incomplete or reference-invalid")
    for name, expected in numerical_receipt["output_sha256"].items():
        valid_name(name)
        if digest(numerical_root / name) != expected:
            raise ValueError("numerical diagnostic differs from receipt")
    roots = [("cpu_analysis", args.cpu_analysis), ("binary_analysis", args.binary_analysis),
        ("binary_run", args.binary_run), ("numerics_run", args.numerics_run),
        ("prior_protocol_run", args.prior_protocol_run), ("prior_protocol_analysis", args.prior_protocol_analysis),
        ("prior_wait_run", args.prior_wait_run), ("inputs/binary", args.binary_inputs),
        ("inputs/protocol", args.protocol_inputs)]
    runs = [args.binary_run, args.numerics_run, args.prior_protocol_run, args.prior_wait_run]
    if recovery:
        verify_receipt(args.recovery_analysis, statuses={"complete"}, schema="imcqa-protocol-recovered-analysis-v1")
        if json.loads((args.recovery_run / "control.json").read_bytes())["source_commit"] != args.recovery_launch_commit:
            raise ValueError("declared recovery launch differs from raw control")
        roots += [("recovery_run", args.recovery_run), ("recovery_analysis", args.recovery_analysis)]
        runs.append(args.recovery_run)
    entries = [entry for label, root in roots for entry in collect_directory(root, label)]
    for label, path in (("inputs/prior_wait_public.json", args.prior_public),
        ("inputs/frozen/main_jobs.json", args.frozen_source), ("inputs/frozen/main_dataset.json", args.gold),
        ("inputs/factorized_cpu_plan.json", args.plan), ("execution_summary.json", args.execution),
        (args.report.name, args.report)):
        entries.append(Entry(label, path=path))
    if args.review_log:
        entries.append(Entry("independent_review.json", path=args.review_log))
    execution = json.loads(args.execution.read_bytes())
    validate_execution(execution, source_commit=args.source_commit, binary_launch_commit=args.binary_launch_commit,
        numerics_launch_commit=args.numerics_launch_commit, workflow_urls=args.workflow_url,
        recovery_launch_commit=args.recovery_launch_commit)
    repo = Path(__file__).resolve().parents[1]
    entries += source_entries(repo, args.source_commit, runs)
    source_by_name = {entry.name: entry.data for entry in entries if entry.data is not None}
    analyses = [(args.cpu_analysis, "analyze_imcqa_factorized_cpu.py"),
                (args.binary_analysis, "analyze_imcqa_binary_pilot.py")]
    if recovery:
        analyses.append((args.recovery_analysis, "analyze_imcqa_protocol_recovery.py"))
    validate_analyzer_sources(source_by_name, analyses)
    commands = reproduction_script(recovery)
    entries.append(Entry("reproduce_cpu.sh", data=commands.encode()))
    readme = f'''IMCQA factorized stopping follow-up, 2026-10-04 UTC

Open {args.report.name} for the findings and limitations.

This archive contains the exact frozen public jobs and evaluator dataset, old WAIT
and protocol evidence, the zero-inference CPU analysis, the binary pilot, and the
3B numerical study. The failed original 3B attempt remains preserved and distinct.
Recovery included: {recovery}. All scientific estimates concern already inspected
development questions; no fresh confirmatory evaluation or response sampling occurred.

Source snapshot: {args.source_commit}
Binary inference launch: {args.binary_launch_commit}
Numerical study launch: {args.numerics_launch_commit}
Recovery launch: {args.recovery_launch_commit or "not included"}
Repository: https://github.com/ankaggarwal94/qanta-buzzer/tree/{args.source_commit}
Workflows:
''' + "".join("  " + url + "\n" for url in args.workflow_url) + '''
Reproduction
------------
Extract to a new directory. Using Python 3.12, install the CPU dependencies:
  python -m pip install numpy==2.3.5 scipy==1.17.0
Then run from the extracted archive root:
  bash reproduce_cpu.sh

The script only recomputes validations, analyses, and numerical comparisons from
saved logits. It never launches inference. Each analysis output is create-once;
use a fresh extraction or move recomputed/ before rerunning. Scientific tables
should reproduce; analysis timestamps and fitted-parameter timestamp hashes change.
The archived source/ tree includes exact inference-bound dependencies and configs,
so the independent analyzers can check historical source hashes without a checkout.

Integrity
---------
artifact_manifest.json binds every other member, including this README and the
reproduction script, with its byte count and SHA-256. The packager verified all
member paths, ZIP CRCs, and archived hashes after writing. The enclosing ZIP SHA
is reported separately because a ZIP cannot contain its own final checksum.
No model weights, credentials, helper transfer files, or provider workspace.json
are included. Inference receipts distinguish new runs from preserved earlier runs.
'''
    entries.append(Entry("README.txt", data=readme.encode()))
    metadata = {"analysis_source_commit": args.source_commit,
        "binary_launch_commit": args.binary_launch_commit, "numerics_launch_commit": args.numerics_launch_commit,
        "recovery_launch_commit": args.recovery_launch_commit, "recovery_included": recovery,
        "workflow_urls": args.workflow_url, "binary_contexts": 832, "cpu_model_forward_passes": 0,
        "self_contained_cpu_inputs": True, "source_snapshot_included": True}
    print(json.dumps(write_archive(args.out, entries, metadata), sort_keys=True))


if __name__ == "__main__":
    main()
