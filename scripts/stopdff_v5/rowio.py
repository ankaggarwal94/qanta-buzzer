"""Deterministic adapter-row I/O.

Adapter row files must be byte-identical across two builds from identical inputs
(adapter determinism pilot). We serialize each row canonically (sorted keys, no extra
whitespace) and gzip with a fixed mtime and compression level so the bytes are stable.
"""
from __future__ import annotations

import gzip
import io
import json
import os
import tempfile
from pathlib import Path
from typing import Iterable

_COMPRESSLEVEL = 6


def _canonical_row(row: dict) -> str:
    return json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def dumps_rows(rows: Iterable[dict]) -> bytes:
    payload = "".join(_canonical_row(r) + "\n" for r in rows).encode("utf-8")
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0, compresslevel=_COMPRESSLEVEL) as gz:
        gz.write(payload)
    return buf.getvalue()


def write_jsonl_gz(path: str | Path, rows: Iterable[dict]) -> None:
    data = dumps_rows(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def read_jsonl_gz(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
