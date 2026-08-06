"""Focused regressions for complete package-PNG validation."""
from __future__ import annotations

import struct
import zlib
from pathlib import Path

import pytest

from scripts.stopdff_v5 import checker, writers


_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _chunk(chunk_type: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + chunk_type
        + payload
        + struct.pack(">I", zlib.crc32(chunk_type + payload) & 0xFFFFFFFF)
    )


def _chunks(data: bytes) -> list[tuple[bytes, bytes]]:
    chunks: list[tuple[bytes, bytes]] = []
    offset = len(_SIGNATURE)
    while offset < len(data):
        length = struct.unpack(">I", data[offset:offset + 4])[0]
        chunk_type = data[offset + 4:offset + 8]
        payload = data[offset + 8:offset + 8 + length]
        chunks.append((chunk_type, payload))
        offset += 12 + length
    return chunks


def _png(chunks: list[tuple[bytes, bytes]]) -> bytes:
    return _SIGNATURE + b"".join(
        _chunk(chunk_type, payload) for chunk_type, payload in chunks
    )


def _errors(path: Path) -> list[str]:
    errors: list[str] = []
    checker._check_png(path, errors)
    return errors


def _minimal_png(tmp_path: Path) -> tuple[Path, bytes]:
    path = tmp_path / "minimal.png"
    writers.write_min_png(path, width=7, height=5)
    return path, path.read_bytes()


def test_png_validator_accepts_both_package_writer_outputs(tmp_path):
    minimal_path, _ = _minimal_png(tmp_path)
    assert _errors(minimal_path) == []

    written = writers.write_figures(
        tmp_path / "matplotlib",
        {
            "cells": {
                "one": {"abs_median_point": 0.5},
                "two": {"abs_median_point": 1.5},
            }
        },
        profile_variant="final",
    )
    matplotlib_path = tmp_path / "matplotlib" / written[0]
    assert _errors(matplotlib_path) == []


def test_png_validator_accepts_ancillary_chunk_and_split_idat(tmp_path):
    path, data = _minimal_png(tmp_path)
    chunks = _chunks(data)
    ihdr = next(payload for kind, payload in chunks if kind == b"IHDR")
    idat = next(payload for kind, payload in chunks if kind == b"IDAT")
    midpoint = len(idat) // 2
    path.write_bytes(
        _png(
            [
                (b"IHDR", ihdr),
                (b"tEXt", b"generator\x00focused-test"),
                (b"IDAT", idat[:midpoint]),
                (b"IDAT", idat[midpoint:]),
                (b"IEND", b""),
            ]
        )
    )
    assert _errors(path) == []


def _corruptions(data: bytes) -> dict[str, bytes]:
    chunks = _chunks(data)
    ihdr = next(payload for kind, payload in chunks if kind == b"IHDR")
    idat = next(payload for kind, payload in chunks if kind == b"IDAT")

    stale_crc = bytearray(data)
    offset = len(_SIGNATURE)
    while data[offset + 4:offset + 8] != b"IDAT":
        offset += 12 + struct.unpack(">I", data[offset:offset + 4])[0]
    stale_crc[offset + 8] ^= 0xFF

    corrupt_stream = bytearray(idat)
    corrupt_stream[0] ^= 0xFF

    invalid_ihdr = bytearray(ihdr)
    invalid_ihdr[10] = 1

    raw = bytearray(zlib.decompress(idat))
    raw[0] = 5

    midpoint = len(idat) // 2
    return {
        "historical_24_byte_prefix": data[:24],
        "truncated_iend": data[:-1],
        "trailing_bytes": data + b"trailing",
        "oversized_chunk_length": (
            data[:8] + struct.pack(">I", 0x7FFFFFFF) + data[12:]
        ),
        "stale_chunk_crc": bytes(stale_crc),
        "corrupt_zlib_with_fresh_crc": _png(
            [(b"IHDR", ihdr), (b"IDAT", bytes(corrupt_stream)), (b"IEND", b"")]
        ),
        "missing_idat": _png([(b"IHDR", ihdr), (b"IEND", b"")]),
        "nonconsecutive_idat": _png(
            [
                (b"IHDR", ihdr),
                (b"IDAT", idat[:midpoint]),
                (b"tEXt", b"gap\x00between-idat"),
                (b"IDAT", idat[midpoint:]),
                (b"IEND", b""),
            ]
        ),
        "illegal_ihdr_method": _png(
            [(b"IHDR", bytes(invalid_ihdr)), (b"IDAT", idat), (b"IEND", b"")]
        ),
        "invalid_scanline_filter": _png(
            [
                (b"IHDR", ihdr),
                (b"IDAT", zlib.compress(bytes(raw), 9)),
                (b"IEND", b""),
            ]
        ),
        "missing_iend": _png([(b"IHDR", ihdr), (b"IDAT", idat)]),
    }


@pytest.mark.parametrize(
    "case",
    [
        "historical_24_byte_prefix",
        "truncated_iend",
        "trailing_bytes",
        "oversized_chunk_length",
        "stale_chunk_crc",
        "corrupt_zlib_with_fresh_crc",
        "missing_idat",
        "nonconsecutive_idat",
        "illegal_ihdr_method",
        "invalid_scanline_filter",
        "missing_iend",
    ],
)
def test_png_validator_rejects_structural_and_stream_corruption(tmp_path, case):
    path, data = _minimal_png(tmp_path)
    path.write_bytes(_corruptions(data)[case])
    assert _errors(path), case
