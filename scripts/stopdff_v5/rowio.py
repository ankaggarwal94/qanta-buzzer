"""Deterministic adapter-row I/O.

Adapter row files must be byte-identical across two builds from identical inputs
(adapter determinism pilot). We serialize each row canonically (sorted keys, no extra
whitespace) and gzip with a fixed mtime and compression level so the bytes are stable.
"""
from __future__ import annotations

import gzip
import io
import json
from pathlib import Path
from typing import Iterable

from .fileio import publish_bytes
from .identity import loads_strict

_COMPRESSLEVEL = 6


def _canonical_row(row: dict) -> str:
    # allow_nan=False: a non-finite value fails loudly at write time instead of
    # emitting a row that ``read_jsonl_gz``'s strict parse rejects anyway.
    return json.dumps(
        row,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


def dumps_rows(rows: Iterable[dict]) -> bytes:
    payload = "".join(_canonical_row(r) + "\n" for r in rows).encode("utf-8")
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0, compresslevel=_COMPRESSLEVEL) as gz:
        gz.write(payload)
    return buf.getvalue()


def write_jsonl_gz(path: str | Path, rows: Iterable[dict]) -> None:
    """Atomically and durably publish one adapter row file.

    Row serialization stays canonical (``dumps_rows``); the crash-durable
    publish mechanics (flush + file fsync before the rename, directory fsync
    after) are the package-wide primitive ``fileio.publish_bytes``.
    """
    publish_bytes(Path(path), dumps_rows(rows))


def read_jsonl_gz(path: str | Path) -> list[dict]:
    """Strictly read adapter rows.

    Each line is parsed under the canonical strict discipline (duplicate keys
    and non-finite constants rejected) and must decode to a JSON object; the
    checker's row reader delegates here so producer and checker reads share
    one fail-closed loader.
    """
    rows: list[dict] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                row = loads_strict(line)
                if not isinstance(row, dict):
                    raise ValueError("adapter JSONL row must be an object")
                rows.append(row)
    return rows
