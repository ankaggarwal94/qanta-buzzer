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

from scripts.stopdff_v5.fileio import dumps_json_bytes  # noqa: E402


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_dumps_json_bytes_rejects_nonfinite(value):
    with pytest.raises(ValueError):
        dumps_json_bytes({"x": value})


def test_dumps_json_bytes_emits_canonical_layout():
    # Sorted keys, two-space indent, trailing newline -- the hash-attested
    # artifact convention. Finite payloads are unaffected by allow_nan=False.
    assert dumps_json_bytes({"b": 1, "a": 2}) == b'{\n  "a": 2,\n  "b": 1\n}\n'
