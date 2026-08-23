"""Raw-byte ingress regression for the shared hardened file reader."""
from __future__ import annotations

import hashlib

from reproducibility.colm_aims_2026 import schema


def test_regular_file_reader_preserves_windows_text_sentinels(tmp_path):
    payload = b"alpha\r\nbeta\x1agamma\r\n\x00\xff\x80"
    path = tmp_path / "raw-byte-vector.bin"
    path.write_bytes(payload)

    observed = schema.read_regular_file_bytes(path)

    assert observed == payload
    assert hashlib.sha256(observed).digest() == hashlib.sha256(payload).digest()
