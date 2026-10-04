"""Package reviewed IMCQA results with hashes and exact source references."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def digest(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''): h.update(block)
    return h.hexdigest()


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('retrospective','menu-controls','pilot-analysis','pilot-run','report','out'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--source-commit',required=True)
    p.add_argument('--workflow-url',required=True)
    a=p.parse_args()
    files=[]
    for key,root in [('retrospective',a.retrospective),('pilot_analysis',a.pilot_analysis),('pilot_run',a.pilot_run)]:
        for path in sorted(root.rglob('*')):
            if path.is_file():
                if path.is_symlink():raise ValueError('symlink in evidence')
                files.append((path,key+'/'+path.relative_to(root).as_posix()))
    for name in ('menu_control_report.json','menu_control_rows.csv'):
        files.append((a.menu_controls/name,'questionless_controls/'+name))
    files.append((a.report,a.report.name))
    if len({name for _,name in files})!=len(files):raise ValueError('duplicate archive paths')
    manifest={'schema':'imcqa-results-package-v1','analysis_source_commit':a.source_commit,
              'workflow_url':a.workflow_url,'files':[{'path':name,'bytes':path.stat().st_size,'sha256':digest(path)} for path,name in files]}
    readme=(
        'IMCQA retrospective quizbowl analysis and development WAIT pilot\n\n'
        'Open the HTML report for findings, definitions, denominators, and limitations.\n'
        'retrospective/: authoritative 200,000-row MC analysis and per-question policy decisions.\n'
        'questionless_controls/: gold joins for the previous 40,000 menu-only score rows.\n'
        'pilot_run/: complete inference outputs, frozen public inputs, numerical checks, and execution receipts.\n'
        'pilot_analysis/: independently validated development-only metrics and paired comparisons.\n'
        'artifact_manifest.json binds every included file by SHA-256.\n\n'
        'Code and tests are versioned in the repository, not duplicated in this data package:\n'
        f'https://github.com/ankaggarwal94/qanta-buzzer/tree/{a.source_commit}\n'
        f'Pilot execution: {a.workflow_url}\n\n'
        'Original raw generations and full frozen evaluator remain in ACL_5000_Raw_Evidence_2026-10-03.tar.gz\n'
        'and ACL_5000_Analysis_2026-10-03.zip. Their exact hashes are recorded in retrospective/evidence_audit.json.\n'
        'This package does not contain model weights or credentials.\n'
    )
    with zipfile.ZipFile(a.out,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for path,name in files:z.write(path,name)
        z.writestr('artifact_manifest.json',json.dumps(manifest,indent=2,sort_keys=True)+'\n')
        z.writestr('README.txt',readme)
    with zipfile.ZipFile(a.out) as z:
        if z.testzip() is not None:raise ValueError('archive CRC failed')
        for row in manifest['files']:
            if hashlib.sha256(z.read(row['path'])).hexdigest()!=row['sha256']:raise ValueError('archive content hash failed')
    print(json.dumps({'path':str(a.out.resolve()),'bytes':a.out.stat().st_size,'sha256':digest(a.out),'files':len(files)+2}))


if __name__=='__main__':main()
