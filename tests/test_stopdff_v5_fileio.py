"""Write-side canonical-encoder tests for ``fileio.dumps_json_bytes``.

L-V3-02 (testing, PR #30 round 3): round 2 added ``allow_nan=False`` to the
single artifact JSON encoder so a non-finite float fails loudly at write time
instead of emitting non-JSON bytes the strict readers reject anyway. The
read-side rejection twins are already tested; this pins the write-side
ValueError (and the canonical byte layout) so removing the flag cannot ship
green.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.stopdff_v5.fileio import (  # noqa: E402
    dumps_json_bytes,
    publish_dir_create_once,
    reclaim_empty_relic,
)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_dumps_json_bytes_rejects_nonfinite(value):
    with pytest.raises(ValueError):
        dumps_json_bytes({"x": value})


def test_dumps_json_bytes_emits_canonical_layout():
    # Sorted keys, two-space indent, trailing newline -- the hash-attested
    # artifact convention. Finite payloads are unaffected by allow_nan=False.
    assert dumps_json_bytes({"b": 1, "a": 2}) == b'{\n  "a": 2,\n  "b": 1\n}\n'


# --- publish_dir_create_once (PR #30: no-replace directory publication) -------
#
# The primitive both codex P2 threads route through: publishing a staged
# directory into a create-once slot must fail closed on ANY pre-existing
# destination -- crucially including an *empty* one, which a bare os.rename
# silently replaces. The empty-dest tests below are red against os.rename.


def _make_staged(tmp_path, name="staged", payload=b"payload"):
    staged = tmp_path / name
    staged.mkdir()
    (staged / "artifact.bin").write_bytes(payload)
    return staged


def test_publish_dir_create_once_publishes_into_absent_slot(tmp_path):
    staged = _make_staged(tmp_path)
    dest = tmp_path / "dest"
    publish_dir_create_once(staged, dest)
    assert (dest / "artifact.bin").read_bytes() == b"payload"
    assert not staged.exists()  # staged consumed by the rename


def test_publish_dir_create_once_fails_closed_on_existing_empty_dest(tmp_path):
    # A bare os.rename onto an empty directory silently replaces it; the
    # mkdir-claim must make it fail closed instead. Mutation-discriminating:
    # against os.rename this raises nothing and the assertion below fails.
    staged = _make_staged(tmp_path)
    dest = tmp_path / "dest"
    dest.mkdir()  # pre-existing EMPTY slot
    with pytest.raises(FileExistsError, match="already exists"):
        publish_dir_create_once(staged, dest)
    assert list(dest.iterdir()) == []  # never replaced
    assert (staged / "artifact.bin").exists()  # staged left for caller cleanup


def test_publish_dir_create_once_fails_closed_on_existing_nonempty_dest(tmp_path):
    staged = _make_staged(tmp_path)
    dest = tmp_path / "dest"
    dest.mkdir()
    (dest / "incumbent").write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError, match="already exists"):
        publish_dir_create_once(staged, dest)
    assert (dest / "incumbent").read_text(encoding="utf-8") == "keep"


def test_publish_dir_create_once_fails_closed_on_symlink_dest(tmp_path):
    # os.mkdir never follows a symlink: the claim fails closed and the link's
    # target is never published into.
    staged = _make_staged(tmp_path)
    other = tmp_path / "other"
    other.mkdir()
    dest = tmp_path / "dest"
    dest.symlink_to(other, target_is_directory=True)
    with pytest.raises(FileExistsError, match="already exists"):
        publish_dir_create_once(staged, dest)
    assert list(other.iterdir()) == []  # symlink target untouched
    assert dest.is_symlink()


def test_publish_dir_create_once_uses_custom_exists_label(tmp_path):
    staged = _make_staged(tmp_path)
    dest = tmp_path / "dest"
    dest.mkdir()
    with pytest.raises(
        FileExistsError, match="fresh run destination already exists"
    ):
        publish_dir_create_once(staged, dest, exists_label="fresh run destination")


# --- reclaim_empty_relic (PR #30: create-once crash-recovery step) -----------
#
# Best-effort recovery companion to publish_dir_create_once: it removes ONLY an
# empty crash relic (the mkdir-succeeded/rename-crashed shape) so a deliberate
# recovery can re-claim the slot, and it must NEVER touch a real artifact (a
# non-empty directory, a file, or a symlink) nor raise.


def test_reclaim_empty_relic_removes_empty_directory(tmp_path):
    dest = tmp_path / "relic"
    dest.mkdir()  # the empty mkdir->rename crash relic
    assert reclaim_empty_relic(dest) is True
    assert not dest.exists()  # relic reclaimed


def test_reclaim_empty_relic_refuses_nonempty_directory(tmp_path):
    dest = tmp_path / "artifact"
    dest.mkdir()
    (dest / "keep.bin").write_bytes(b"real")
    assert reclaim_empty_relic(dest) is False
    assert dest.is_dir()
    assert (dest / "keep.bin").read_bytes() == b"real"  # artifact intact


def test_reclaim_empty_relic_refuses_regular_file(tmp_path):
    dest = tmp_path / "file"
    dest.write_bytes(b"real")
    assert reclaim_empty_relic(dest) is False
    assert dest.is_file()
    assert dest.read_bytes() == b"real"


def test_reclaim_empty_relic_refuses_symlink(tmp_path):
    # Never follow or remove a symlink; leave it for the later mkdir gate.
    target = tmp_path / "target"
    target.mkdir()
    (target / "keep.bin").write_bytes(b"real")
    dest = tmp_path / "dest"
    dest.symlink_to(target, target_is_directory=True)
    assert reclaim_empty_relic(dest) is False
    assert dest.is_symlink()  # link untouched
    assert (target / "keep.bin").read_bytes() == b"real"  # target untouched


def test_reclaim_empty_relic_absent_dest_is_noop(tmp_path):
    dest = tmp_path / "absent"
    assert reclaim_empty_relic(dest) is False
    assert not dest.exists()


def test_reclaim_empty_relic_then_publish_recovers_the_slot(tmp_path):
    # End-to-end recovery: an empty crash relic is reclaimed, after which the
    # create-once publish succeeds via reclaim -> mkdir -> rename.
    dest = tmp_path / "dest"
    dest.mkdir()  # crash relic blocking the create-once mkdir claim
    staged = _make_staged(tmp_path)
    assert reclaim_empty_relic(dest) is True
    publish_dir_create_once(staged, dest)  # mkdir claim now succeeds
    assert (dest / "artifact.bin").read_bytes() == b"payload"
