#!/usr/bin/env python3
"""Package completed frozen-transfer data without duplicating git-backed source."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import zipfile


ANALYSIS_COMMIT = "291253fb1d04645da4c8dd60bbb2d6e1451dc7ca"
INFERENCE_COMMIT = "c037f01fa66770dd6d12b1011207fe644b9b72e0"
REPOSITORY = "https://github.com/ankaggarwal94/qanta-buzzer.git"
CHUNK = 1 << 20
SKIP_NAMES = {"workspace.json", "credentials.json", "token.json", "auth.json"}
FORBIDDEN_SUFFIXES = {".safetensors", ".bin", ".pt", ".pth", ".pkl", ".pem", ".key", ".py", ".sh", ".zip"}


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(CHUNK), b""):
            result.update(block)
    return result.hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_bytes())


def safe_name(name: str) -> str:
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
        safe_name(self.name)
        if (self.path is None) == (self.data is None):
            raise ValueError("entry requires exactly one content source")
        if self.path is not None and (not self.path.is_file() or self.path.is_symlink()):
            raise ValueError("entry must be a regular non-symlink file")

    def metadata(self) -> dict:
        return {"path": self.name, "bytes": self.path.stat().st_size if self.path else len(self.data),
                "sha256": digest(self.path) if self.path else hashlib.sha256(self.data).hexdigest()}


def collect(root: Path, name: str) -> list[Entry]:
    if not root.is_dir() or root.is_symlink():
        raise ValueError("missing evidence directory: " + str(root))
    entries = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("symlink in evidence directory")
        if not path.is_file():
            continue
        if path.name in SKIP_NAMES or path.name.startswith(".env") or "__pycache__" in path.parts or ".git" in path.parts:
            continue
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            raise ValueError("source code, weights, credentials, or nested archive in evidence: " + str(path))
        entries.append(Entry(name + "/" + path.relative_to(root).as_posix(), path=path))
    if not entries:
        raise ValueError("empty evidence directory")
    return entries


def committed_bytes(name: str) -> bytes:
    safe_name(name)
    repo = Path(__file__).resolve().parents[1]
    return subprocess.run(["git", "-C", str(repo), "show", ANALYSIS_COMMIT + ":" + name],
                          check=True, capture_output=True).stdout


def validate(args) -> tuple[dict, dict, dict, Path | None]:
    """Require completed, mutually hash-bound analysis and independent validation."""
    receipt = load(args.analysis / "analysis_receipt.json")
    validation = load(args.analysis / "validation.json")
    summary = load(args.analysis / "summary.json")
    audit = load(args.audit)
    provenance = load(args.provenance)
    worker = args.run / "output" / "qwen7b"
    worker_receipt = load(worker / "receipt.json")
    control = load(args.run / "control.json")
    if receipt.get("status") != "complete" or receipt.get("fitting_performed") is not False or validation.get("passed") is not True:
        raise ValueError("analysis or validation is incomplete")
    if summary.get("n_questions") != 100 or summary.get("n_score_rows") != 8000 or summary.get("model") != "qwen7b":
        raise ValueError("completed analysis scope differs")
    if (worker_receipt.get("status") != "complete" or worker_receipt.get("completed_rows") != 8000
            or worker_receipt.get("expected_rows") != 8000 or worker_receipt.get("source_commit") != INFERENCE_COMMIT
            or control.get("source_commit") != INFERENCE_COMMIT or validation.get("source_commit") != INFERENCE_COMMIT):
        raise ValueError("worker or launch completion differs")
    if (digest(worker / "scores.jsonl") != worker_receipt["scores_sha256"]
            or digest(worker / "scores.jsonl") != validation["scores_sha256"]
            or digest(worker / "metadata.json") != validation["metadata_sha256"]
            or worker_receipt.get("public_input_sha256") != receipt.get("public_sha256")):
        raise ValueError("saved worker data changed after validation")
    expected_outputs = receipt.get("outputs_sha256", {})
    if not {"summary.json", "validation.json", "episodes.csv", "per_question.csv"} <= set(expected_outputs):
        raise ValueError("analysis receipt does not bind required outputs")
    for name, expected in expected_outputs.items():
        safe_name(name)
        if digest(args.analysis / name) != expected:
            raise ValueError("analysis output changed after completion: " + name)
    source_paths = {
        "main_jobs": args.prior / "inputs/frozen/main_jobs.json",
        "main_dataset": args.prior / "inputs/frozen/main_dataset.json",
        "prior_public": args.prior / "inputs/prior_wait_public.json",
        "fitted_parameters": args.prior / "cpu_analysis/fitted_parameters.json",
    }
    if set(receipt.get("input_sha256", {})) != set(source_paths):
        raise ValueError("analysis source-input hash map differs")
    for name, path in source_paths.items():
        if digest(path) != receipt["input_sha256"][name]:
            raise ValueError("original source bytes differ: " + name)
    if (hashlib.sha256(committed_bytes("scripts/analyze_imcqa_transfer.py")).hexdigest() != receipt["analyzer_sha256"]
            or hashlib.sha256(committed_bytes("configs/imcqa_frozen_transfer.json")).hexdigest() != receipt["plan_sha256"]):
        raise ValueError("analysis/config bytes differ from pinned analysis commit")
    if (audit.get("schema_version") != "imcqa-frozen-transfer-independent-audit-v1"
            or audit.get("status") != "passed" or audit.get("imports_project_analysis_code") is not False
            or audit.get("public_sha256") != receipt["public_sha256"]
            or audit.get("result_audit", {}).get("all_policy_points_and_intervals_reproduced") is not True
            or audit.get("result_audit", {}).get("raw_rows_checked") != 8000
            or audit.get("result_audit", {}).get("paired_contrasts_reproduced") != 4
            or audit.get("result_audit", {}).get("episodes_independently_reconstructed") != 4000):
        raise ValueError("full independent result audit is missing or did not pass")
    audited = audit.get("audited_files_sha256", {})
    audited_paths = {"gold": source_paths["main_dataset"], "fits": source_paths["fitted_parameters"],
                     "prior_public": source_paths["prior_public"], "scores": worker/"scores.jsonl",
                     "receipt": worker/"receipt.json", "summary": args.analysis/"summary.json",
                     "episodes": args.analysis/"episodes.csv", "per_question": args.analysis/"per_question.csv"}
    if audited.get("public") != receipt["public_sha256"] or any(audited.get(k) != digest(p) for k,p in audited_paths.items()):
        raise ValueError("independent audit is not bound to current input/result bytes")
    if audit.get("scores_sha256") != worker_receipt["scores_sha256"] or audit.get("analysis_receipt_sha256") != digest(args.analysis/"analysis_receipt.json"):
        raise ValueError("independent audit completion binding differs")
    script = args.independent_audit_script or args.audit.parent / "independent_transfer_audit.py"
    if script.is_file():
        if digest(script) != audit.get("auditor_sha256"):
            raise ValueError("independent auditor script changed after audit")
    elif args.independent_audit_script:
        raise FileNotFoundError(script)
    else:
        script = None
    if (provenance.get("source_commit") != INFERENCE_COMMIT or provenance.get("analysis_commit") != ANALYSIS_COMMIT
            or provenance.get("workflow_conclusion") != "success"
            or type(provenance.get("workflow_run_id")) is not int
            or provenance.get("workflow_url") != "https://github.com/ankaggarwal94/qanta-buzzer/actions/runs/" + str(provenance["workflow_run_id"])):
        raise ValueError("workflow provenance is incomplete or mismatched")
    compute = provenance.get("compute", {})
    if compute.get("invoice_verified") is not False:
        raise ValueError("compute ledger must preserve unverified invoice status")
    elapsed = compute.get("actual_elapsed_seconds")
    if elapsed is not None and abs(float(elapsed) - float(worker_receipt["elapsed_seconds"])) > 1e-6:
        raise ValueError("compute elapsed time differs from worker receipt")
    # Preserve exactly the hash-bound original cache/model metadata, not old scores.
    cache = args.prior / "prior_protocol_run/output/cache_prepare_receipt.json"
    if digest(cache) != "95cc6dcd2e99e74597c95c1bf4580457dd843a01053267cf3888b21461a7a1e0":
        raise ValueError("original cache receipt hash differs")
    # The public object is already in the pinned source; include it only when it
    # is part of the downloaded new run evidence. Reproduction can recover it
    # directly from the committed compressed input otherwise.
    candidates = [args.run / "pilot.json", args.run / "public.json"]
    for candidate in candidates:
        if candidate.is_file() and digest(candidate) != receipt["public_sha256"]:
            raise ValueError("run public input differs from completed analysis")
    committed_public = gzip.decompress(committed_bytes("imcqa_transfer_public/public.json.gz"))
    if hashlib.sha256(committed_public).hexdigest() != receipt["public_sha256"]:
        raise ValueError("committed public input differs from run")
    return receipt, summary, provenance, script


def reproduction_script(public_hash: str, with_auditor: bool) -> str:
    if not re.fullmatch(r"[0-9a-f]{64}", public_hash):
        raise ValueError("public checksum must be exact")
    text = '''#!/usr/bin/env bash
set -euo pipefail
# CPU-only reproduction. Python 3.12 + numpy==2.3.5 are required.
# This script performs no model inference or cloud allocation.
IMCQA_ARTIFACT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
IMCQA_ANALYSIS_COMMIT="__COMMIT__"
if [[ -n "${IMCQA_CODE_REPO:-}" ]]; then
  IMCQA_CODE_REPO="$(cd -- "$IMCQA_CODE_REPO" && pwd)"
else
  IMCQA_CODE_REPO="$IMCQA_ARTIFACT_ROOT/reproduction_code"
  if [[ ! -d "$IMCQA_CODE_REPO/.git" ]]; then
    git init -q "$IMCQA_CODE_REPO"
    git -C "$IMCQA_CODE_REPO" remote add origin __REPOSITORY__
    git -C "$IMCQA_CODE_REPO" fetch --depth=1 origin "$IMCQA_ANALYSIS_COMMIT"
    git -C "$IMCQA_CODE_REPO" checkout --detach "$IMCQA_ANALYSIS_COMMIT"
  fi
fi
if [[ "$(git -C "$IMCQA_CODE_REPO" rev-parse HEAD)" != "$IMCQA_ANALYSIS_COMMIT" ]]; then
  echo "IMCQA_CODE_REPO must be checked out at $IMCQA_ANALYSIS_COMMIT" >&2
  exit 1
fi
git -C "$IMCQA_CODE_REPO" diff --quiet HEAD -- scripts configs imcqa_transfer_public
export IMCQA_ARTIFACT_ROOT IMCQA_CODE_REPO
python - <<'PYINPUT'
import gzip, hashlib, json, os
from pathlib import Path
root=Path(os.environ['IMCQA_ARTIFACT_ROOT']);code=Path(os.environ['IMCQA_CODE_REPO'])
manifest=json.loads((root/'artifact_manifest.json').read_bytes())
for row in manifest['files']:
    path=root/row['path']; h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1<<20),b''):h.update(chunk)
    if path.stat().st_size != row['bytes'] or h.hexdigest() != row['sha256']:
        raise RuntimeError('Artifact file hash mismatch: '+row['path'])
out=root/'recomputed';out.mkdir(exist_ok=False)
public=next((p for p in (root/'run/pilot.json',root/'run/public.json') if p.is_file()),None)
raw=public.read_bytes() if public else gzip.decompress((code/'imcqa_transfer_public/public.json.gz').read_bytes())
if hashlib.sha256(raw).hexdigest() != '__PUBLIC_HASH__':
    raise RuntimeError('Reproduction public input hash differs')
(out/'public.json').write_bytes(raw)
PYINPUT
cd -- "$IMCQA_CODE_REPO"
export PYTHONPATH="$IMCQA_CODE_REPO"
python -m scripts.analyze_imcqa_transfer \\
  --public "$IMCQA_ARTIFACT_ROOT/recomputed/public.json" \\
  --source "$IMCQA_ARTIFACT_ROOT/inputs/main_jobs.json" \\
  --evaluator "$IMCQA_ARTIFACT_ROOT/inputs/main_dataset.json" \\
  --fitted "$IMCQA_ARTIFACT_ROOT/inputs/fitted_parameters.json" \\
  --prior-public "$IMCQA_ARTIFACT_ROOT/inputs/prior_wait_public.json" \\
  --prior-model-dir "$IMCQA_ARTIFACT_ROOT/inputs/prior_qwen7b" \\
  --cache-receipt "$IMCQA_ARTIFACT_ROOT/inputs/cache_prepare_receipt.json" \\
  --plan "$IMCQA_CODE_REPO/configs/imcqa_frozen_transfer.json" \\
  --run-dir "$IMCQA_ARTIFACT_ROOT/run/output/qwen7b" \\
  --out "$IMCQA_ARTIFACT_ROOT/recomputed/analysis"
'''.replace("__COMMIT__", ANALYSIS_COMMIT).replace("__REPOSITORY__", REPOSITORY).replace("__PUBLIC_HASH__", public_hash)
    if with_auditor:
        text += '''python "$IMCQA_ARTIFACT_ROOT/audit/independent_transfer_audit.py" \\
  --public "$IMCQA_ARTIFACT_ROOT/recomputed/public.json" \\
  --gold "$IMCQA_ARTIFACT_ROOT/inputs/main_dataset.json" \\
  --fits "$IMCQA_ARTIFACT_ROOT/inputs/fitted_parameters.json" \\
  --prior-public "$IMCQA_ARTIFACT_ROOT/inputs/prior_wait_public.json" \\
  --worker "$IMCQA_ARTIFACT_ROOT/run/output/qwen7b" \\
  --analysis "$IMCQA_ARTIFACT_ROOT/recomputed/analysis" \\
  --out "$IMCQA_ARTIFACT_ROOT/recomputed/independent_audit.json"
'''
    text += '''python - <<'PYCOMPARE'
import hashlib, json, os
from pathlib import Path
root=Path(os.environ['IMCQA_ARTIFACT_ROOT'])
original=json.loads((root/'analysis/analysis_receipt.json').read_bytes())
for name,expected in original['outputs_sha256'].items():
    actual=hashlib.sha256((root/'recomputed/analysis'/name).read_bytes()).hexdigest()
    if actual != expected:raise RuntimeError('CPU reproduction differs: '+name)
print('All scientific and validation output bytes reproduced exactly; no model inference ran.')
PYCOMPARE
'''
    return text


def write_archive(out: Path, entries: list[Entry], metadata: dict) -> dict:
    names = [e.name for e in entries]
    if len(names) != len(set(names)) or "artifact_manifest.json" in names:
        raise ValueError("duplicate or reserved archive path")
    manifest = {"schema_version": "imcqa-frozen-transfer-package-v1", **metadata,
                "files": [e.metadata() for e in entries]}
    with zipfile.ZipFile(out, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for entry in entries:
            if entry.path:
                archive.write(entry.path, entry.name)
            else:
                archive.writestr(entry.name, entry.data)
        archive.writestr("artifact_manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    with zipfile.ZipFile(out) as archive:
        if len(archive.namelist()) != len(set(archive.namelist())) or archive.testzip() is not None:
            raise ValueError("ZIP duplicate-path or CRC validation failed")
        for name in archive.namelist():
            safe_name(name)
        for row in manifest["files"]:
            hashed = hashlib.sha256()
            with archive.open(row["path"]) as stream:
                for block in iter(lambda: stream.read(CHUNK), b""):
                    hashed.update(block)
            if archive.getinfo(row["path"]).file_size != row["bytes"] or hashed.hexdigest() != row["sha256"]:
                raise ValueError("archived bytes differ: " + row["path"])
    return {"path": str(out.resolve()), "bytes": out.stat().st_size, "sha256": digest(out), "files": len(entries)+1}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prior", "run", "analysis", "audit", "report", "provenance", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--independent-audit-script", type=Path)
    args = parser.parse_args()
    receipt, summary, provenance, script = validate(args)
    entries = [
        Entry("inputs/main_jobs.json", path=args.prior/"inputs/frozen/main_jobs.json"),
        Entry("inputs/main_dataset.json", path=args.prior/"inputs/frozen/main_dataset.json"),
        Entry("inputs/prior_wait_public.json", path=args.prior/"inputs/prior_wait_public.json"),
        Entry("inputs/fitted_parameters.json", path=args.prior/"cpu_analysis/fitted_parameters.json"),
        Entry("inputs/prior_qwen7b/metadata.json", path=args.prior/"prior_protocol_run/output/qwen7b/metadata.json"),
        Entry("inputs/cache_prepare_receipt.json", path=args.prior/"prior_protocol_run/output/cache_prepare_receipt.json"),
        Entry("audit/independent_audit.json", path=args.audit),
        Entry("IMCQA_Frozen_Transfer_2026-10-04.html", path=args.report),
        Entry("provenance.json", path=args.provenance),
    ]
    entries += collect(args.run, "run") + collect(args.analysis, "analysis")
    if script:
        entries.append(Entry("audit/independent_transfer_audit.py", path=script))
    entries.append(Entry("reproduce_cpu.sh", data=reproduction_script(receipt["public_sha256"], script is not None).encode()))
    readme = f'''IMCQA frozen-policy transfer, 2026-10-04 UTC

Open IMCQA_Frozen_Transfer_2026-10-04.html for findings and limitations.
This package contains completed data, analysis, and independent audit evidence:
100 protocol-unexposed selection questions, 8,000 Qwen 7B scoring contexts,
five saved policies, and four prespecified paired reward contrasts. Policies
were frozen before inference. The corpus was already used in earlier analyses;
this is development transfer, not a pristine confirmatory evaluation.

Source code is preserved in its original git repository rather than duplicated:
Repository: {REPOSITORY.removesuffix('.git')}
Inference commit: {INFERENCE_COMMIT}
Analysis commit: {ANALYSIS_COMMIT}
Workflow: {provenance['workflow_url']}

CPU reproduction
----------------
Use Python 3.12 and numpy==2.3.5. Extract to a new directory, then run:
  bash reproduce_cpu.sh
The script clones the public repository and checks out the exact analysis
commit. To use an existing clean checkout at that exact commit instead:
  IMCQA_CODE_REPO=/absolute/path/to/exact-checkout bash reproduce_cpu.sh
No model inference or cloud compute is performed. The optional git clone uses
network access. Reproduction creates recomputed/ once, validates every packaged
file, reruns the analyzer, and compares all scientific output bytes. The included
standalone independent auditor reruns when available. Use a new extraction for
another reproduction attempt.

Contents and integrity
----------------------
inputs/ retains the four complete original files needed for source hash checks,
plus only the prior 7B model metadata and cache receipt. These older corpora
contain 5,000 questions; they do not expand the transfer sample beyond 100.
run/ contains only the new run evidence. analysis/ contains its completed outputs.
No prior full score sets, model weights, credentials, provider workspace files,
or git-backed project source code are included. The standalone auditor and
generated reproduction helper are included for independent verification.
artifact_manifest.json binds every other member by path, size, and SHA-256.
ZIP paths, CRCs, and each archived member hash were verified during packaging.
The enclosing ZIP checksum is reported separately, since a ZIP cannot include
its own final checksum. Resource cost is an estimate; the invoice is unverified.
'''
    entries.append(Entry("README.txt", data=readme.encode()))
    result = write_archive(args.out, entries, {"inference_commit": INFERENCE_COMMIT,
        "analysis_commit": ANALYSIS_COMMIT, "repository": REPOSITORY.removesuffix(".git"),
        "n_questions": summary["n_questions"], "n_score_rows": summary["n_score_rows"],
        "public_sha256": receipt["public_sha256"], "independent_audit_passed": True,
        "git_backed_source_code_included": False})
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
