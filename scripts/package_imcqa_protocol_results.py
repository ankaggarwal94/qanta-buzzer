"""Package protocol diagnosis evidence, preserving prior/new inference identity."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

from scripts.package_imcqa_results import digest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("decomposition", "analysis", "run", "prior-run", "inputs", "report", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--launch-commit", required=True)
    parser.add_argument("--workflow-url", required=True)
    args = parser.parse_args()
    files = []
    for label, root in (("decomposition", args.decomposition), ("protocol_analysis", args.analysis),
                        ("protocol_run", args.run), ("prior_wait_run", args.prior_run)):
        if not root.is_dir():
            raise ValueError("missing evidence directory")
        for path in sorted(root.rglob("*")):
            if path.is_symlink():
                raise ValueError("symlink in evidence")
            if path.is_file():
                files.append((path, label + "/" + path.relative_to(root).as_posix()))
    for name in ("transport_manifest.json", "comprehension_expected.json"):
        files.append((args.inputs / name, "protocol_inputs/" + name))
    files.append((args.report, args.report.name))
    if len({name for _, name in files}) != len(files):
        raise ValueError("duplicate archive path")
    manifest = {"schema": "imcqa-protocol-results-package-v1",
                "analysis_source_commit": args.source_commit, "launch_commit": args.launch_commit,
                "workflow_url": args.workflow_url,
                "files": [{"path": name, "bytes": path.stat().st_size, "sha256": digest(path)}
                          for path, name in files]}
    readme = (
        "IMCQA protocol diagnosis, 2026-10-04 UTC\n\n"
        "Open the HTML report for the CPU decomposition and matched development pilot findings.\n"
        "decomposition/: 800 original-pilot trajectories, own-candidate hindsight, and calibration references.\n"
        "protocol_analysis/: validated merged contexts, paired summaries, and execution evidence.\n"
        "protocol_run/: 8,064 newly scored contexts, numerical checks, exact inputs, and reuse manifests.\n"
        "prior_wait_run/: unchanged prior inputs and scores supplying 2,400 reused contexts.\n"
        "protocol_inputs/: transport manifest and synthetic evaluator fixture; fixture was not uploaded to inference.\n"
        "artifact_manifest.json hashes every included file. No model weights or credentials are included.\n\n"
        "Code is versioned in the repository and is not duplicated here:\n"
        f"https://github.com/ankaggarwal94/qanta-buzzer/tree/{args.source_commit}\n"
        f"Inference launch commit: {args.launch_commit}\n"
        f"Workflow: {args.workflow_url}\n\n"
        "The complete frozen evaluator, original public jobs, and original graded generations remain in\n"
        "ACL_5000_Analysis_2026-10-03.zip; exact input hashes appear in the analysis receipts.\n"
        "All findings are exploratory. The matched pilot has 40 previously inspected real questions,\n"
        "20 per development split, plus 32 synthetic contexts per model. No test question was used.\n"
    )
    with zipfile.ZipFile(args.out, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path, name in files:
            archive.write(path, name)
        archive.writestr("artifact_manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        archive.writestr("README.txt", readme)
    with zipfile.ZipFile(args.out) as archive:
        if archive.testzip() is not None:
            raise ValueError("ZIP integrity failure")
        for row in manifest["files"]:
            if hashlib.sha256(archive.read(row["path"])).hexdigest() != row["sha256"]:
                raise ValueError("archived content hash differs")
    print(json.dumps({"path": str(args.out.resolve()), "bytes": args.out.stat().st_size,
                      "sha256": digest(args.out), "files": len(files) + 2}))


if __name__ == "__main__":
    main()
