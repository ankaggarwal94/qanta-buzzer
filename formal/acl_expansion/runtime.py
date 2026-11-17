"""Persistent interface to the pinned Lean menu decoder and reward schedule checker."""

from __future__ import annotations

import json
import hashlib
import math
import os
import struct
import subprocess
import shutil
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from collections.abc import Sequence
from fractions import Fraction
from pathlib import Path

PROJECT = Path(__file__).resolve().parent


class MissingRuntimeError(FileNotFoundError):
    """The optional native executable has not been built."""


def _wire_json(value):
    """Encode exact integers, independently of CPython's decimal conversion limit."""
    if isinstance(value, dict):
        return "{" + ",".join(json.dumps(k) + ":" + _wire_json(v) for k, v in value.items()) + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(map(_wire_json, value)) + "]"
    if type(value) is int and value.bit_length() > 2000:
        return format(Decimal(value), "f")
    return json.dumps(value, allow_nan=False)


class LeanRuntime:
    def __init__(self, timeout: float = 30.0) -> None:
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be positive and finite")
        binary = PROJECT / ".lake" / "build" / "bin" / (
            "acl_runtime.exe" if os.name == "nt" else "acl_runtime"
        )
        if not binary.is_file():
            raise MissingRuntimeError("Build formal/acl_expansion with 'lake build acl_runtime'")

        def digest(path):
            with path.open("rb") as handle:
                return hashlib.file_digest(handle, "sha256").hexdigest()

        self.timeout = timeout
        self._lock = threading.Lock()
        self._closed = False
        self._generation = tempfile.TemporaryDirectory(prefix="acl-runtime-")
        try:
            # A private executable snapshot keeps its identity through an atomic install.
            pinned = Path(self._generation.name) / binary.name
            shutil.copy2(binary, pinned)
            self.identity = {
                "binary_sha256": digest(pinned),
                "sources": {f: digest(PROJECT / f) for f in (
                    "Core.lean", "Core/Expr.lean", "Core/Wire.lean", "TransitionCore.lean",
                    "Runtime.lean", "lean-toolchain", "runtime.py",
                )},
            }
            self.process = subprocess.Popen(
                [str(pinned)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                text=True, encoding="utf-8", cwd=PROJECT,
            )
        except BaseException:
            self._generation.cleanup()
            raise
        self._worker = ThreadPoolExecutor(max_workers=1)

    def _exchange(self, payload):
        self.process.stdin.write(payload + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("Lean runtime stopped before answering")
        return json.loads(line, parse_int=lambda digits: int(Decimal(digits)))

    def _request(self, **fields):
        payload = _wire_json(fields)
        with self._lock:
            if self._closed:
                raise RuntimeError("Lean runtime is closed")
            pending = self._worker.submit(self._exchange, payload)
            try:
                response = pending.result(timeout=self.timeout)
            except BaseException:
                self.process.kill()
                self._shutdown()
                raise
        if "error" in response:
            raise ValueError(response["error"])
        return response["value"]

    def draw(self, n: int, digits: Sequence[int]) -> tuple[int, ...]:
        return tuple(self._request(op="draw", n=n, digits=list(digits)))

    def insert(self, xs: Sequence[int], gold: int, pos: int) -> tuple[int, ...]:
        return tuple(self._request(op="insert", xs=list(xs), gold=gold, pos=pos))

    def formula(self, arm, start, end, shape, threshold, knots, baseline):
        """Emit the complete payout expression for one configured arm.

        Parameters
        ----------
        arm : str
            Reward arm name accepted by the native dispatcher.
        start, end, shape, threshold, baseline : int, float or Fraction
            Finite parameters exposing an exact integer ratio.
        knots : sequence of pairs
            Progress and payout coordinates exposing exact integer ratios.

        Returns
        -------
        list
            The nested opcode array with signed integer numerator and denominator literals.
        """
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

    def _shutdown(self):
        self._closed = True
        try:
            self._worker.shutdown(wait=True)
            try:
                self.process.stdin.close()
            except BrokenPipeError:
                pass  # The child may already have exited or been killed on timeout.
            self.process.wait(timeout=self.timeout)
        finally:
            if self.process.poll() is None:
                self.process.kill()
                self.process.wait()
            self.process.stdout.close()
            self._generation.cleanup()

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._shutdown()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
