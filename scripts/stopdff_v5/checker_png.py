"""Pure-PNG format validation for packaged figures (ACCEPTANCE_CONTRACT.md).

Self-contained byte-level PNG validator (chunk framing, CRCs, IHDR legality,
zlib image stream, scanline layout) so packaged figures are proven complete
without relying on an optional image library. Extracted verbatim from
``checker``; a concern separate from run semantics.
"""
from __future__ import annotations

import struct
import zlib
from pathlib import Path

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_PNG_MAX_DECODED_BYTES = 256 * 1024 * 1024
_PNG_ALLOWED_DEPTHS = {
    0: {1, 2, 4, 8, 16},
    2: {8, 16},
    3: {1, 2, 4, 8},
    4: {8, 16},
    6: {8, 16},
}
_PNG_CHANNELS = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}
_PNG_ADAM7_PASSES = (
    (0, 0, 8, 8),
    (4, 0, 8, 8),
    (0, 4, 4, 8),
    (2, 0, 4, 4),
    (0, 2, 2, 4),
    (1, 0, 2, 2),
    (0, 1, 1, 2),
)


class _PNGError(ValueError):
    """A complete PNG structural or image-stream check failed."""


def _png_pass_geometry(
    width: int,
    height: int,
    interlace: int,
) -> list[tuple[int, int]]:
    if interlace == 0:
        return [(width, height)]
    passes: list[tuple[int, int]] = []
    for x_start, y_start, x_step, y_step in _PNG_ADAM7_PASSES:
        pass_width = (
            0
            if width <= x_start
            else (width - x_start + x_step - 1) // x_step
        )
        pass_height = (
            0
            if height <= y_start
            else (height - y_start + y_step - 1) // y_step
        )
        if pass_width and pass_height:
            passes.append((pass_width, pass_height))
    return passes


def _png_scanline_layout(
    *,
    width: int,
    height: int,
    bit_depth: int,
    color_type: int,
    interlace: int,
) -> tuple[list[tuple[int, int]], int]:
    bits_per_pixel = bit_depth * _PNG_CHANNELS[color_type]
    layout: list[tuple[int, int]] = []
    expected_size = 0
    for pass_width, pass_height in _png_pass_geometry(
        width,
        height,
        interlace,
    ):
        row_bytes = (pass_width * bits_per_pixel + 7) // 8
        layout.append((pass_height, row_bytes))
        expected_size += pass_height * (1 + row_bytes)
    return layout, expected_size


def _validate_png_bytes(data: bytes) -> None:
    """Validate a complete PNG without relying on an optional image library."""
    if data[:8] != _PNG_SIGNATURE:
        raise _PNGError("invalid signature")

    offset = len(_PNG_SIGNATURE)
    chunk_index = 0
    ihdr: tuple[int, int, int, int, int] | None = None
    saw_palette = False
    saw_idat = False
    idat_closed = False
    saw_iend = False
    idat_parts: list[bytes] = []

    while offset < len(data):
        if saw_iend:
            raise _PNGError("trailing bytes after IEND")
        if len(data) - offset < 12:
            raise _PNGError("truncated chunk framing")

        chunk_length = struct.unpack(">I", data[offset:offset + 4])[0]
        if chunk_length > 0x7FFFFFFF:
            raise _PNGError("chunk length exceeds the PNG limit")
        chunk_end = offset + 12 + chunk_length
        if chunk_end > len(data):
            raise _PNGError("truncated chunk payload")

        chunk_type = data[offset + 4:offset + 8]
        payload = data[offset + 8:offset + 8 + chunk_length]
        stored_crc = struct.unpack(">I", data[chunk_end - 4:chunk_end])[0]
        if not all(
            ord("A") <= byte <= ord("Z")
            or ord("a") <= byte <= ord("z")
            for byte in chunk_type
        ):
            raise _PNGError("invalid chunk type")
        if chunk_type[2] & 0x20:
            raise _PNGError("invalid reserved bit in chunk type")
        actual_crc = zlib.crc32(chunk_type + payload) & 0xFFFFFFFF
        if stored_crc != actual_crc:
            raise _PNGError(f"CRC mismatch in {chunk_type.decode('ascii')}")

        if chunk_index == 0 and chunk_type != b"IHDR":
            raise _PNGError("IHDR is not the first chunk")
        if chunk_type == b"IHDR":
            if chunk_index != 0 or ihdr is not None or chunk_length != 13:
                raise _PNGError("invalid IHDR placement or length")
            (
                width,
                height,
                bit_depth,
                color_type,
                compression,
                filter_method,
                interlace,
            ) = struct.unpack(">IIBBBBB", payload)
            if (
                width == 0
                or height == 0
                or width > 0x7FFFFFFF
                or height > 0x7FFFFFFF
            ):
                raise _PNGError("invalid image dimensions")
            if bit_depth not in _PNG_ALLOWED_DEPTHS.get(color_type, set()):
                raise _PNGError("illegal bit-depth/color-type combination")
            if compression != 0 or filter_method != 0 or interlace not in {0, 1}:
                raise _PNGError("unsupported IHDR method")
            ihdr = (width, height, bit_depth, color_type, interlace)
        elif ihdr is None:
            raise _PNGError("chunk appears before IHDR")
        elif chunk_type == b"PLTE":
            if saw_palette or saw_idat:
                raise _PNGError("invalid PLTE placement")
            color_type = ihdr[3]
            entries = chunk_length // 3
            if (
                color_type in {0, 4}
                or chunk_length == 0
                or chunk_length % 3
                or entries > 256
                or (color_type == 3 and entries > 2 ** ihdr[2])
            ):
                raise _PNGError("invalid PLTE payload")
            saw_palette = True
        elif chunk_type == b"IDAT":
            if idat_closed:
                raise _PNGError("nonconsecutive IDAT chunks")
            if ihdr[3] == 3 and not saw_palette:
                raise _PNGError("indexed PNG is missing PLTE before IDAT")
            saw_idat = True
            idat_parts.append(payload)
        elif chunk_type == b"IEND":
            if chunk_length != 0 or not saw_idat:
                raise _PNGError("invalid IEND or missing IDAT")
            saw_iend = True
            if chunk_end != len(data):
                raise _PNGError("trailing bytes after IEND")
        else:
            if not (chunk_type[0] & 0x20):
                raise _PNGError("unknown critical chunk")
            if saw_idat:
                idat_closed = True

        offset = chunk_end
        chunk_index += 1

    if ihdr is None or not saw_idat or not saw_iend:
        raise _PNGError("missing required PNG chunk")
    if ihdr[3] == 3 and not saw_palette:
        raise _PNGError("indexed PNG is missing PLTE")

    layout, expected_size = _png_scanline_layout(
        width=ihdr[0],
        height=ihdr[1],
        bit_depth=ihdr[2],
        color_type=ihdr[3],
        interlace=ihdr[4],
    )
    if expected_size > _PNG_MAX_DECODED_BYTES:
        raise _PNGError("decoded image exceeds the package limit")

    compressed = b"".join(idat_parts)
    if not compressed:
        raise _PNGError("empty IDAT stream")
    try:
        decompressor = zlib.decompressobj()
        raw = decompressor.decompress(compressed, expected_size + 1)
        if decompressor.unconsumed_tail or len(raw) > expected_size:
            raise _PNGError("inflated image exceeds its IHDR dimensions")
        if not decompressor.eof:
            raise _PNGError("truncated zlib image stream")
        if decompressor.unused_data:
            raise _PNGError("trailing data in zlib image stream")
        raw += decompressor.flush()
    except zlib.error as exc:
        raise _PNGError(f"invalid zlib image stream: {exc}") from exc
    if len(raw) != expected_size:
        raise _PNGError("inflated image size does not match IHDR")

    cursor = 0
    for row_count, row_bytes in layout:
        stride = 1 + row_bytes
        for _ in range(row_count):
            if raw[cursor] > 4:
                raise _PNGError("invalid scanline filter")
            cursor += stride
    if cursor != len(raw):
        raise _PNGError("scanline layout does not consume the image stream")


def _check_png(path: Path, errors: list[str]) -> None:
    try:
        _validate_png_bytes(path.read_bytes())
    except (OSError, _PNGError) as exc:
        errors.append(f"invalid PNG {path.name}: {exc}")
