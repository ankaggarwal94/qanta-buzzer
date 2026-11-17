"""Persistent interface to the pinned Lean menu decoder and reward schedule checker."""

from __future__ import annotations

import json
import hashlib
import math
import os
import struct
import subprocess
from collections.abc import Sequence
from fractions import Fraction
from pathlib import Path

PROJECT = Path(__file__).resolve().parent


class LeanRuntime:
    def __init__(self) -> None:
        binary = PROJECT / ".lake" / "build" / "bin" / (
            "acl_runtime.exe" if os.name == "nt" else "acl_runtime"
        )
        if not binary.is_file():
            raise FileNotFoundError("Build formal/acl_expansion with 'lake build acl_runtime'")
        def digest(path):
            with path.open("rb") as handle:
                return hashlib.file_digest(handle, "sha256").hexdigest()
        self.identity = {
            "binary_sha256": digest(binary),
            "sources": {f: digest(PROJECT / f) for f in (
                "Core.lean", "Core/Expr.lean", "Core/Wire.lean", "TransitionCore.lean", "Runtime.lean",
                "lean-toolchain", "runtime.py"
            )},
        }
        self.process = subprocess.Popen(
            [str(binary)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, encoding="utf-8", cwd=PROJECT,
        )

    def _request(self, **fields):
        assert self.process.stdin and self.process.stdout
        self.process.stdin.write(json.dumps(fields, allow_nan=False) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("Lean runtime stopped before answering")
        response = json.loads(line)
        if "error" in response:
            raise ValueError(response["error"])
        return response["value"]

    def draw(self, n: int, digits: Sequence[int]) -> tuple[int, ...]:
        return tuple(self._request(op="draw", n=n, digits=list(digits)))

    def insert(self, xs: Sequence[int], gold: int, pos: int) -> tuple[int, ...]:
        return tuple(self._request(op="insert", xs=list(xs), gold=gold, pos=pos))

    def formula(self, arm, start, end, shape, threshold, knots, baseline):
        """Emit the complete payout expression with its exact configured parameters."""
        params = dict(start=start, end=end, shape=shape, threshold=threshold, baseline=baseline)
        return self._request(
            op="formula", arm=arm,
            knots=[[x.as_integer_ratio(), y.as_integer_ratio()] for x, y in knots],
            **{key: value.as_integer_ratio() for key, value in params.items()},
        )

    def validate_transition(self, base, gamma, current, successor, rounded_total) -> None:
        """Check single-rounding via exact dyadics including subnormal and overflow edges."""
        rounded = float(rounded_total)
        if not math.isfinite(rounded):
            raise ValueError("rounded reward must be finite")
        prev = math.nextafter(rounded, -math.inf)
        after = math.nextafter(rounded, math.inf)
        # The virtual neighbor at either overflow edge has magnitude 2**1024.
        prev = Fraction(prev) if math.isfinite(prev) else Fraction(-(2**1024))
        after = Fraction(after) if math.isfinite(after) else Fraction(2**1024)
        bits = struct.unpack(">Q", struct.pack(">d", rounded))[0]
        exact = dict(base=base, gamma=gamma, current=current, successor=successor,
                     prev=prev, rounded=rounded, next=after)
        if self._request(op="transition", even=(bits & 1) == 0, **{
            name: value.as_integer_ratio() for name, value in exact.items()
        }) is not True:
            raise ValueError("unexpected transition verification result")

    def validate_schedule(self, values: Sequence[float], first: float, last: float) -> None:
        def ratio(x):
            if not math.isfinite(x):
                raise ValueError("reward value must be finite")
            return float(x).as_integer_ratio()
        if self._request(op="schedule", values=[ratio(x) for x in values],
                         first=ratio(first), last=ratio(last)) is not True:
            raise ValueError("unexpected schedule verification result")

    def close(self) -> None:
        if self.process.stdin and not self.process.stdin.closed:
            self.process.stdin.close()
        try:
            self.process.wait(timeout=10)
        finally:
            if self.process.poll() is None:
                self.process.kill()
                self.process.wait()
            if self.process.stdout:
                self.process.stdout.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


def make_decoder() -> LeanRuntime:
    return LeanRuntime()
