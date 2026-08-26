"""Raw-byte ingress regression for the shared hardened file reader."""
from __future__ import annotations

import hashlib

import pytest

from reproducibility.colm_aims_2026 import schema


def test_regular_file_reader_preserves_windows_text_sentinels(tmp_path):
    payload = b"alpha\r\nbeta\x1agamma\r\n\x00\xff\x80"
    path = tmp_path / "raw-byte-vector.bin"
    path.write_bytes(payload)

    observed = schema.read_regular_file_bytes(path)

    assert observed == payload
    assert hashlib.sha256(observed).digest() == hashlib.sha256(payload).digest()


@pytest.mark.parametrize(
    "payload",
    [
        b'{"identity":"first","identity":"second"}',
        b'{"outer":{"mode":"first","mode":"second"}}',
    ],
)
def test_strict_json_bytes_reject_duplicate_object_members(payload):
    with pytest.raises(schema.TypedIngressError, match="duplicate JSON object"):
        schema.parse_json_bytes_strict(payload)


def test_strict_json_text_rejects_duplicate_record_members():
    with pytest.raises(schema.TypedIngressError, match="duplicate JSON object"):
        schema.parse_json_text_strict(
            '{"item_key":"first","item_key":"second"}',
            "records/cell.jsonl: line 1",
        )


@pytest.mark.parametrize("length", [40, 64])
def test_native_git_object_ids_accept_sha1_and_sha256(length):
    assert schema.is_git_object_id("a" * length)


@pytest.mark.parametrize(
    "value", ["a" * 39, "a" * 41, "a" * 63, "a" * 65, "A" * 40]
)
def test_native_git_object_ids_reject_noncanonical_values(value):
    assert not schema.is_git_object_id(value)


def test_model_revision_commit_predicate_remains_sha1_only():
    assert schema.is_commit_sha("a" * 40)
    assert not schema.is_commit_sha("a" * 64)
