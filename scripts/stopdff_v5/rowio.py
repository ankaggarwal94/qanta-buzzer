"""Deterministic adapter-row I/O.

Adapter row files must be byte-identical across two builds from identical inputs
(adapter determinism pilot). We serialize each row canonically (sorted keys, no extra
whitespace) and gzip with a fixed mtime and compression level so the bytes are stable.
"""
from __future__ import annotations

import gzip
import io
import json
from collections.abc import Iterable
from pathlib import Path
from typing import BinaryIO

from .fileio import atomic_binary_writer
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


def _write_rows(handle: BinaryIO, rows: Iterable[dict]) -> None:
    # The private staging filename must never enter the hash-attested gzip header.
    with gzip.GzipFile(
        filename="", fileobj=handle, mode="wb", mtime=0, compresslevel=_COMPRESSLEVEL
    ) as archive:
        for row in rows:
            archive.write((_canonical_row(row) + "\n").encode("utf-8"))


def dumps_rows(rows: Iterable[dict]) -> bytes:
    """Return the deterministic gzip representation of a single pass over rows."""
    with io.BytesIO() as buf:
        _write_rows(buf, rows)
        return buf.getvalue()


def write_jsonl_gz(path: str | Path, rows: Iterable[dict]) -> None:
    """Stream canonical rows into a durably published gzip file."""
    with atomic_binary_writer(Path(path)) as handle:
        _write_rows(handle, rows)


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
