"""Losslessly transport the ACL frozen PUBLIC inputs as small text chunks.

Only two public inference files are accepted. Repeated prompts are factored into
question/prefix/menu rows; job and prompt digests are recomputed on restoration.
Exact original byte hashes are required before publishing restored files. There
is no pickle, eval, archive extraction, network request, or evaluator metadata.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import lzma
from pathlib import Path
import re
import tempfile

NAMES = ("main_jobs.json", "main_choices_only.json")
FROZEN_HASHES = {
    "main_jobs.json": "9bfeaf2d86116ced0e8c55c3c390d3be12050ad38820b4e02c6ed684cc8786bf",
    "main_choices_only.json": "9db13301d928cd31dc54c97f0c5cfd88b9bc25774ceb56027c032c52e4afb043",
}
MAIN_FIELDS = {"job_id", "qid", "group_id", "split", "format", "condition", "menu_id", "prefix_id", "fraction", "prompt", "prompt_sha256"}
CONTROL_FIELDS = {"job_id", "qid", "group_id", "split", "format", "condition", "menu_id", "options", "prompt", "prompt_sha256"}
CONDITIONS = ("independent_pool", "same_category_pool")
MAX_MAIN_BYTES = 384 * 1024 * 1024
MAX_CONTROL_BYTES = 32 * 1024 * 1024
MAX_COMPACT_BYTES = 64 * 1024 * 1024
MAX_COMPRESSED_BYTES = 32 * 1024 * 1024
MAX_MANIFEST_BYTES = 256 * 1024
CHUNK_BYTES = 65_536
MAX_CHUNKS = 768


def canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def digest(value: bytes | str) -> str:
    return hashlib.sha256(value.encode("utf-8") if isinstance(value, str) else value).hexdigest()


def _unique(pairs: list[tuple]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def load(raw: bytes) -> dict:
    def reject(value: str):
        raise ValueError(f"nonfinite JSON number: {value}")
    return json.loads(raw, object_pairs_hook=_unique, parse_constant=reject)


def keys(value: object, expected: set[str], label: str) -> None:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"unexpected {label} fields")


def text(value: object, label: str, maximum: int = 1024) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum:
        raise ValueError(f"invalid {label} text")
    return value


def read_regular(path: Path, maximum: int) -> bytes:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > maximum:
        raise ValueError(f"nonregular or oversized transport input: {path.name}")
    with path.open("rb") as stream:
        raw = stream.read(maximum + 1)
    if len(raw) > maximum:
        raise ValueError("input grew beyond byte bound")
    return raw


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def with_identity(identity: dict) -> dict:
    return {"job_id": digest(canonical(identity)), **identity, "prompt_sha256": digest(identity["prompt"])}


def check_job(job: dict, fields: set[str]) -> None:
    keys(job, fields, "public job")
    identity = {key: value for key, value in job.items() if key not in {"job_id", "prompt_sha256"}}
    if digest(canonical(identity)) != job["job_id"] or digest(job["prompt"]) != job["prompt_sha256"]:
        raise ValueError("public job digest mismatch")


def check_options(options: object) -> None:
    if not isinstance(options, list) or len(options) != 4:
        raise ValueError("exactly four public options required")
    for letter, option in zip("ABCD", options):
        keys(option, {"id", "text"}, "public option")
        if option["id"] != letter:
            raise ValueError("option IDs must be ordered A/B/C/D")
        text(option["text"], "option", 1024)
    if len({option["text"] for option in options}) != 4:
        raise ValueError("duplicate option text")


def compact_inputs(main: dict, controls: dict) -> dict:
    for package, schema in ((main,"jane-public-jobs-v1"),(controls,"jane-choice-controls-v1")):
        keys(package, {"schema_version", "evidence_scope", "jobs"}, "public package")
        if package["schema_version"] != schema or package["evidence_scope"] != "scientific" or not isinstance(package["jobs"], list):
            raise ValueError("unexpected public package schema/scope/jobs")
    if not 0 < len(main["jobs"]) <= 150000 or not 0 < len(controls["jobs"]) <= 10000:
        raise ValueError("public job count outside ACL expansion bound")
    templates, questions, seen_ids = {}, {}, set()
    for job in main["jobs"]:
        check_job(job, MAIN_FIELDS)
        if job["job_id"] in seen_ids:
            raise ValueError("duplicate public job ID")
        seen_ids.add(job["job_id"])
        instruction, separator, payload_raw = job["prompt"].rpartition("\n\n")
        if not separator:
            raise ValueError("missing v4 prompt delimiter")
        payload = load(payload_raw.encode())
        fmt = job["format"]
        if fmt not in {"oe", "mc"}:
            raise ValueError("unknown public format")
        keys(payload, {"question_prefix"} if fmt == "oe" else {"question_prefix", "options"}, "prompt payload")
        if fmt in templates and templates[fmt] != instruction:
            raise ValueError("prompt instruction varies within format")
        templates[fmt] = instruction
        qid = job["qid"]
        question = questions.setdefault(qid, {"qid": qid, "group_id": job["group_id"], "split": job["split"], "prefixes": {}, "menus": {}})
        if question["group_id"] != job["group_id"] or question["split"] != job["split"]:
            raise ValueError("question metadata varies across public jobs")
        prefix = {"prefix_id": job["prefix_id"], "text": payload["question_prefix"], "fraction": job["fraction"]}
        previous = question["prefixes"].setdefault(job["prefix_id"], prefix)
        if previous != prefix:
            raise ValueError("paired prefix differs across arms")
        if fmt == "oe":
            if job["condition"] != "oe" or job["menu_id"] is not None:
                raise ValueError("unexpected OE design")
        else:
            check_options(payload["options"])
            if job["condition"] not in CONDITIONS or job["menu_id"] != "fixed_1":
                raise ValueError("unexpected MC design")
            menu = {"condition": job["condition"], "menu_id": job["menu_id"], "options": payload["options"]}
            previous_menu = question["menus"].setdefault(job["condition"], menu)
            if previous_menu != menu:
                raise ValueError("menu differs across prefixes")
    rows = []
    for qid, question in sorted(questions.items()):
        prefixes = list(question["prefixes"].values())
        if len(prefixes) != 10 or [p["prefix_id"] for p in prefixes] != [f"p{k}" for k in range(1,11)]:
            raise ValueError("ten ordered public prefixes required")
        full = prefixes[-1]["text"]
        if not all(full.startswith(prefix["text"]) for prefix in prefixes):
            raise ValueError("public prefix is not an exact leading substring")
        if set(question["menus"]) != set(CONDITIONS):
            raise ValueError("both frozen MC conditions required")
        rows.append({"qid":qid,"group_id":question["group_id"],"split":question["split"], "question":full,
            "prefixes":[{"prefix_id":p["prefix_id"],"end":len(p["text"]),"fraction":p["fraction"]} for p in prefixes],
            "menus":[question["menus"][condition] for condition in CONDITIONS]})
    seen_controls = set()
    for job in controls["jobs"]:
        check_job(job, CONTROL_FIELDS)
        key = (job["qid"],job["condition"])
        if key in seen_controls or job["qid"] not in questions:
            raise ValueError("duplicate or unknown control job")
        seen_controls.add(key)
        question=questions[job["qid"]]
        if (job["group_id"] != question["group_id"] or job["split"] != question["split"] or job["format"] != "mc"
                or job["condition"] not in CONDITIONS or job["menu_id"] != "fixed_1"
                or job["options"] != question["menus"][job["condition"]]["options"]):
            raise ValueError("control does not match public main menu")
        rendered = "\n".join(f"{o['id']}. {o['text']}" for o in job["options"])
        before, separator, after = job["prompt"].partition(rendered)
        if not separator:
            raise ValueError("control option rendering missing")
        for name, value in (("control_prefix",before),("control_suffix",after)):
            if name in templates and templates[name] != value:
                raise ValueError("choice-only instructions vary")
            templates[name]=value
    if len(seen_controls) != 2*len(rows):
        raise ValueError("control coverage incomplete")
    return {"schema_version":"acl-public-compact-v1","evidence_scope":"scientific","templates":templates,"questions":rows}


def restore_packages(compact: dict) -> dict[str, bytes]:
    keys(compact,{"schema_version","evidence_scope","templates","questions"},"compact public envelope")
    if compact["schema_version"] != "acl-public-compact-v1" or compact["evidence_scope"] != "scientific":
        raise ValueError("unexpected compact schema/scope")
    templates = compact["templates"]
    keys(templates,{"oe","mc","control_prefix","control_suffix"},"prompt templates")
    for name, value in templates.items(): text(value,name,8192)
    rows = compact["questions"]
    if not isinstance(rows,list) or not 0 < len(rows) <= 5000:
        raise ValueError("compact question count outside bound")
    main_jobs, control_jobs, qids = [], [], set()
    for row in rows:
        keys(row,{"qid","group_id","split","question","prefixes","menus"},"compact question")
        qid=text(row["qid"],"qid"); text(row["group_id"],"group_id")
        full=text(row["question"],"question",131072)
        if qid in qids or row["split"] not in {"calibration","selection","test"}:
            raise ValueError("duplicate qid or invalid split")
        qids.add(qid)
        menus=row["menus"]
        if not isinstance(menus,list) or len(menus)!=2:
            raise ValueError("two public menus required")
        for condition,menu in zip(CONDITIONS,menus):
            keys(menu,{"condition","menu_id","options"},"compact menu")
            if menu["condition"]!=condition or menu["menu_id"]!="fixed_1":raise ValueError("invalid menu identity")
            check_options(menu["options"])
        prefixes=row["prefixes"]
        if not isinstance(prefixes,list) or len(prefixes)!=10:raise ValueError("ten prefixes required")
        previous=0
        for k,prefix in enumerate(prefixes,1):
            keys(prefix,{"prefix_id","end","fraction"},"compact prefix")
            end=prefix["end"]
            if type(end) is not int or not previous < end <= len(full) or prefix["prefix_id"]!=f"p{k}":
                raise ValueError("invalid prefix boundary or order")
            if end<len(full) and not full[end].isspace() and not full[end-1].isspace():raise ValueError("prefix splits a word")
            fraction=prefix["fraction"]
            if type(fraction) is not float or not 0 < fraction <= 1 or fraction!=len(full[:end].split())/len(full.split()):
                raise ValueError("invalid actual prefix fraction")
            previous=end
            for menu in [None,*menus]:
                fmt="oe" if menu is None else "mc"
                payload={"question_prefix":full[:end]}
                if menu is not None:payload["options"]=menu["options"]
                prompt=templates[fmt]+"\n\n"+canonical(payload)
                identity={"qid":qid,"group_id":row["group_id"],"split":row["split"],"format":fmt,
                    "condition":"oe" if menu is None else menu["condition"],"menu_id":None if menu is None else menu["menu_id"],
                    "prefix_id":prefix["prefix_id"],"fraction":fraction,"prompt":prompt}
                main_jobs.append(with_identity(identity))
        if previous!=len(full):raise ValueError("terminal prefix is incomplete")
        for menu in menus:
            options=menu["options"]
            rendered="\n".join(f"{o['id']}. {o['text']}" for o in options)
            prompt=templates["control_prefix"]+rendered+templates["control_suffix"]
            identity={"qid":qid,"group_id":row["group_id"],"split":row["split"],"format":"mc",
                "condition":menu["condition"],"menu_id":menu["menu_id"],"options":options,"prompt":prompt}
            control_jobs.append(with_identity(identity))
    return {"main_jobs.json":json_bytes({"schema_version":"jane-public-jobs-v1","evidence_scope":"scientific","jobs":main_jobs}),
            "main_choices_only.json":json_bytes({"schema_version":"jane-choice-controls-v1","evidence_scope":"scientific","jobs":control_jobs})}


def pack(public_dir: Path, out_dir: Path, expected_hashes: dict[str,str] | None = None) -> dict:
    expected_hashes=FROZEN_HASHES if expected_hashes is None else expected_hashes
    keys(expected_hashes,set(NAMES),"expected file digests")
    raw={name:read_regular(public_dir/name,MAX_MAIN_BYTES if name==NAMES[0] else MAX_CONTROL_BYTES) for name in NAMES}
    if any(digest(raw[name])!=expected_hashes[name] for name in NAMES):raise ValueError("public source differs from externally expected frozen hash")
    packages={name:load(raw[name]) for name in NAMES}
    compact=compact_inputs(packages[NAMES[0]],packages[NAMES[1]])
    restored=restore_packages(compact)
    if restored!=raw:raise ValueError("factored public inputs do not restore exact original bytes")
    compact_bytes=canonical(compact).encode()
    if len(compact_bytes)>MAX_COMPACT_BYTES:raise ValueError("compact byte limit exceeded")
    compressed=lzma.compress(compact_bytes,format=lzma.FORMAT_XZ,preset=6)
    if len(compressed)>MAX_COMPRESSED_BYTES:raise ValueError("compressed byte limit exceeded")
    encoded=base64.b64encode(compressed)
    chunks=[encoded[i:i+CHUNK_BYTES] for i in range(0,len(encoded),CHUNK_BYTES)]
    if not 0<len(chunks)<=MAX_CHUNKS:raise ValueError("chunk count limit exceeded")
    manifest={"schema_version":"acl-public-transport-v1","compression":"xz+base64","compact_sha256":digest(compact_bytes),
        "compact_bytes":len(compact_bytes),"compressed_sha256":digest(compressed),"compressed_bytes":len(compressed),
        "chunks":[{"name":f"chunk_{i:03d}.b64","sha256":digest(chunk),"byte_count":len(chunk)} for i,chunk in enumerate(chunks)],
        "files":{name:{"sha256":digest(raw[name]),"byte_count":len(raw[name]),"job_count":len(packages[name]["jobs"])} for name in NAMES}}
    out_dir.mkdir(parents=True,exist_ok=False)
    for entry,chunk in zip(manifest["chunks"],chunks):
        with (out_dir/entry["name"]).open("xb") as stream:stream.write(chunk)
    with (out_dir/"transport_manifest.json").open("xb") as stream:stream.write(json_bytes(manifest))
    return {"compact_bytes":len(compact_bytes),"compressed_bytes":len(compressed),"base64_bytes":len(encoded),
            "chunks":len(chunks),"files":manifest["files"]}


def restore(transport_dir: Path, out_dir: Path, expected_hashes: dict[str,str] | None = None) -> dict:
    expected_hashes=FROZEN_HASHES if expected_hashes is None else expected_hashes
    keys(expected_hashes,set(NAMES),"expected file digests")
    manifest_raw=read_regular(transport_dir/"transport_manifest.json",MAX_MANIFEST_BYTES)
    manifest=load(manifest_raw)
    keys(manifest,{"schema_version","compression","compact_sha256","compact_bytes","compressed_sha256","compressed_bytes","chunks","files"},"transport manifest")
    if manifest["schema_version"]!="acl-public-transport-v1" or manifest["compression"]!="xz+base64":raise ValueError("unsupported transport schema/compression")
    if (type(manifest["compact_bytes"]) is not int or not 0<manifest["compact_bytes"]<=MAX_COMPACT_BYTES
            or type(manifest["compressed_bytes"]) is not int or not 0<manifest["compressed_bytes"]<=MAX_COMPRESSED_BYTES):raise ValueError("transport size bound exceeded")
    keys(manifest["files"],set(NAMES),"restored file allowlist")
    for name in NAMES:
        entry=manifest["files"][name]
        keys(entry,{"sha256","byte_count","job_count"},"restored file entry")
        if (entry["sha256"]!=expected_hashes[name] or type(entry["byte_count"]) is not int
                or not 0<entry["byte_count"]<=(MAX_MAIN_BYTES if name==NAMES[0] else MAX_CONTROL_BYTES)
                or type(entry["job_count"]) is not int or not 0<entry["job_count"]<=(150000 if name==NAMES[0] else 10000)):
            raise ValueError("restored file identity/count/size differs from expected bounds")
    chunks=manifest["chunks"]
    if not isinstance(chunks,list) or not 0<len(chunks)<=MAX_CHUNKS:raise ValueError("invalid chunk count")
    encoded=[]
    for i,entry in enumerate(chunks):
        keys(entry,{"name","sha256","byte_count"},"chunk manifest entry")
        if entry["name"]!=f"chunk_{i:03d}.b64" or type(entry["byte_count"]) is not int or not 0<entry["byte_count"]<=CHUNK_BYTES:
            raise ValueError("unsafe or unexpected chunk name/size")
        chunk=read_regular(transport_dir/entry["name"],CHUNK_BYTES)
        if len(chunk)!=entry["byte_count"] or digest(chunk)!=entry["sha256"]:raise ValueError("chunk digest/size mismatch")
        encoded.append(chunk)
    if {p.name for p in transport_dir.iterdir()}!={"transport_manifest.json",*(entry["name"] for entry in chunks)}:
        raise ValueError("unexpected transport file outside allowlist")
    compressed=base64.b64decode(b"".join(encoded),validate=True)
    if len(compressed)!=manifest["compressed_bytes"] or digest(compressed)!=manifest["compressed_sha256"]:raise ValueError("compressed digest/size mismatch")
    decoder=lzma.LZMADecompressor(format=lzma.FORMAT_XZ,memlimit=128*1024*1024)
    compact_raw=decoder.decompress(compressed,max_length=MAX_COMPACT_BYTES+1)
    if len(compact_raw)!=manifest["compact_bytes"] or not decoder.eof or decoder.unused_data or digest(compact_raw)!=manifest["compact_sha256"]:
        raise ValueError("compact digest/size/trailing-stream mismatch")
    restored=restore_packages(load(compact_raw))
    for name,raw in restored.items():
        if len(raw)!=manifest["files"][name]["byte_count"] or digest(raw)!=expected_hashes[name]:raise ValueError("restored byte identity mismatch")
        if len(load(raw)["jobs"])!=manifest["files"][name]["job_count"]:raise ValueError("restored job count mismatch")
    if out_dir.exists():raise FileExistsError(out_dir)
    out_dir.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="acl-public-restore-",dir=out_dir.parent) as temporary:
        staging=Path(temporary)/"inputs"; staging.mkdir()
        for name,raw in restored.items():
            with (staging/name).open("xb") as stream:stream.write(raw)
        staging.rename(out_dir)
    return {"schema_version":"acl-public-restore-receipt-v1","files":manifest["files"],"transport_manifest_sha256":digest(manifest_raw)}


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("pack","restore","unpack"))
    input_args=parser.add_mutually_exclusive_group(required=True)
    input_args.add_argument("--input-dir",type=Path)
    input_args.add_argument("--transport-dir",type=Path,dest="input_dir")
    parser.add_argument("--out-dir",type=Path,required=True)
    parser.add_argument("--expected-main-sha256",default=FROZEN_HASHES[NAMES[0]])
    parser.add_argument("--expected-controls-sha256",default=FROZEN_HASHES[NAMES[1]])
    args=parser.parse_args()
    expected={NAMES[0]:args.expected_main_sha256,NAMES[1]:args.expected_controls_sha256}
    print(json.dumps((pack if args.action=="pack" else restore)(args.input_dir,args.out_dir,expected),sort_keys=True))


if __name__=="__main__":main()
