"""Exercise the compiled Lean bridge. Build acl_runtime to enable this optional suite."""

import itertools
import math
import sys
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock
from fractions import Fraction

import pytest

from formal.acl_expansion.runtime import LeanRuntime, MissingRuntimeError


@pytest.fixture(scope="module")
def runtime():
    try:
        client = LeanRuntime()
    except MissingRuntimeError:
        pytest.skip("build formal/acl_expansion acl_runtime first")
    with client:
        yield client


def test_draw_and_gold(runtime):
    for digits in itertools.product(range(4), range(3)):
        pool = list(range(4))
        selected = tuple(pool.pop(i) for i in digits)
        assert runtime.draw(4, digits) == selected
        for pos in range(3):
            expected = list(selected)
            expected.insert(pos, 4)
            assert runtime.insert(selected, 4, pos) == tuple(expected)
    with pytest.raises(ValueError):
        runtime.draw(4, [0, 3])
    with pytest.raises(ValueError):
        runtime.insert([1, 1], 4, 1)
    with pytest.raises(ValueError):
        runtime.insert([0, 1], 1, 1)
    with pytest.raises(ValueError):
        runtime.insert([0, 1], 2, 3)


def test_schedule(runtime):
    runtime.validate_schedule([0.7, 0.7, 0.25], 0.7, 0.25)
    runtime.validate_schedule([0.0], 0.0, 0.0)
    for schedule in ([1.0, 0.5, 0.75, 0.0], [0.75, 0.0], [1.0, 0.5], []):
        with pytest.raises(ValueError):
            runtime.validate_schedule(schedule, 1.0, 0.0)


def test_binary64_boundaries(runtime):
    tiny = math.ulp(0.0)
    for value in (0.0, -0.0, tiny, -tiny, sys.float_info.min,
                  sys.float_info.max, -sys.float_info.max):
        runtime.validate_transition(Fraction(value), 0.0, 0.0, 0.0, value)
    for a, b in ((1.0, math.nextafter(1.0, 2.0)), (0.0, tiny), (0.0, -tiny)):
        midpoint = (Fraction(a) + Fraction(b)) / 2
        runtime.validate_transition(midpoint, 0.0, 0.0, 0.0, a)
        with pytest.raises(ValueError):
            runtime.validate_transition(midpoint, 0.0, 0.0, 0.0, b)
    with pytest.raises(ValueError):
        runtime.validate_transition(Fraction(2**1024), 0.0, 0.0, 0.0, sys.float_info.max)


def test_transition_includes_shaping(runtime):
    runtime.validate_transition(Fraction(1, 4), 0.5, 0.125, 2.0, 1.125)
    for wrong in (1.0, 0.875):
        with pytest.raises(ValueError):
            runtime.validate_transition(Fraction(1, 4), 0.5, 0.125, 2.0, wrong)


def test_formula_wire_roundtrip(runtime):
    def emit(arm, shape=0.5):
        return runtime.formula(arm, 1.0, 0.0, shape, 0.5, [(0.0, 1.0), (1.0, 0.0)], -0.25)

    assert emit("baseline") == ["q", -1, 4]
    assert emit("constant") == ["q", 1, 1]
    q = lambda n, d=1: ["q", n, d]
    p = ["p"]
    linear = ["sub", q(1), p]
    def wrap(decay):
        return ["add", q(0), ["mul", ["sub", q(1), q(0)], decay]]
    assert emit("linear") == wrap(linear)
    assert emit("exponential", 0.0) == wrap(linear)
    e = ["exp", q(-3, 4)]
    assert emit("exponential", 0.75) == wrap(
        ["div", ["sub", ["exp", ["mul", q(-3, 4), p]], e], ["sub", q(1), e]])
    assert emit("power", 0.75) == wrap(["pow", linear, q(3, 4)])
    def segment(x0, x1, y0, y1):
        return ["add", y1, ["mul", ["sub", y0, y1],
                            ["div", ["sub", x1, p], ["sub", x1, x0]]]]
    last = ["if_le", p, q(1, 4), q(1, 2),
            ["if_lt", p, q(1), segment(q(1, 4), q(1), q(1, 2), q(0)), q(0)]]
    expected = ["if_le", p, q(0), q(1),
                ["if_lt", p, q(1, 4), segment(q(0), q(1, 4), q(1), q(1, 2)), last]]
    assert runtime.formula("piecewise", 1., 0., .75, .5, [(0., 1.), (.25, .5), (1., 0.)], 0.) == wrap(expected)
    assert emit("cosine")[2][2] == ["div", ["add", ["q", 1, 1], ["cospi", ["p"]]], ["q", 2, 1]]
    assert emit("step")[2][2] == ["if_lt", ["p"], ["q", 1, 2], ["q", 1, 1], ["q", 0, 1]]
    with pytest.raises(ValueError):
        emit("unrecognized")


def test_large_exact_rational(runtime):
    runtime.validate_transition(Fraction(1, 10**5000), 0., 0., 0., 0.)


def test_parallel_requests_share_one_stream(runtime):
    def check(index):
        if index % 2:
            with pytest.raises(ValueError):
                runtime.validate_schedule([1, 2], 1, 2)
        else:
            assert runtime.draw(3, [index % 3]) == (index % 3,)
    with ThreadPoolExecutor(max_workers=8) as workers:
        list(workers.map(check, range(100)))


def test_probe_compares_json_types():
    from formal.acl_expansion.build_runtime import probe_responses_match
    assert not probe_responses_match([{"value": 1}], [{"value": True}])
    assert not probe_responses_match([{"value": [True]}], [{"value": [1]}])
    assert probe_responses_match([{"value": True}], [{"value": True}])


def simulated_client(process):
    client = LeanRuntime.__new__(LeanRuntime)
    client.timeout = 0.1
    client._lock = threading.Lock()
    client._closed = False
    client._generation = Mock()
    client._worker = ThreadPoolExecutor(max_workers=1)
    client.process = process
    return client


def test_shutdown_reaps_stalled_child():
    process = Mock()
    process.wait.side_effect = [subprocess.TimeoutExpired("acl_runtime", 0.1), 0]
    process.poll.return_value = None
    client = simulated_client(process)
    with pytest.raises(subprocess.TimeoutExpired):
        client.close()
    process.stdin.close.assert_called_once()
    process.kill.assert_called_once()
    assert process.wait.call_count == 2
    process.stdout.close.assert_called_once()


def test_request_timeout_discards_stream():
    released = threading.Event()
    process = Mock()
    process.stdout.readline.side_effect = lambda: (released.wait(5), "")[1]
    process.kill.side_effect = released.set
    process.poll.return_value = -1
    client = simulated_client(process)
    with pytest.raises(TimeoutError):
        client.draw(1, [0])
    process.kill.assert_called_once()
    with pytest.raises(RuntimeError, match="closed"):
        client.draw(1, [0])


def test_missing_identity_file_is_an_error(tmp_path, monkeypatch):
    from formal.acl_expansion import runtime as bridge
    monkeypatch.setattr(bridge, "PROJECT", tmp_path)
    with pytest.raises(MissingRuntimeError):
        LeanRuntime()
    binary = tmp_path / ".lake/build/bin" / ("acl_runtime.exe" if bridge.os.name == "nt" else "acl_runtime")
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"candidate")
    with pytest.raises(FileNotFoundError) as failure:
        LeanRuntime()
    assert not isinstance(failure.value, MissingRuntimeError)
    assert failure.value.filename.endswith("Core.lean")


def test_client_pins_its_launched_generation(tmp_path, monkeypatch):
    from formal.acl_expansion import runtime as bridge

    monkeypatch.setattr(bridge, "PROJECT", tmp_path)
    binary = tmp_path / ".lake/build/bin" / ("acl_runtime.exe" if bridge.os.name == "nt" else "acl_runtime")
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"first published generation")
    for name in ("Core.lean", "Core/Expr.lean", "Core/Wire.lean", "TransitionCore.lean",
                 "Runtime.lean", "lean-toolchain", "runtime.py"):
        source = tmp_path / name
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text("fixture")
    launched = []
    def launch(command, **kwargs):
        replacement = binary.with_suffix(".next")
        replacement.write_bytes(b"new published generation")
        replacement.replace(binary)
        launched.append(bridge.Path(command[0]).read_bytes())
        return Mock()
    monkeypatch.setattr(bridge.subprocess, "Popen", launch)
    with LeanRuntime() as client:
        assert launched == [b"first published generation"]
        assert bridge.hashlib.sha256(launched[0]).hexdigest() == client.identity["binary_sha256"]


def test_axiom_audit_uses_frozen_policy_and_minimal_mounts(tmp_path, monkeypatch):
    from formal.acl_curves import verify
    prefix = tmp_path / "toolchain"
    config = {"solution_module": "Solution", "theorem_names": ["Example"],
              "permitted_axioms": ["Quot.sound"]}
    allowed = "/lib/test-loader/libc.so.6"
    def checked(command, **kwargs):
        if command[0] == "ldd":
            return f"libc.so.6 => {allowed} (0x01)\n"
        assert "--clearenv" in command and "--unshare-all" in command
        assert "/" not in command
        assert "--bind" not in command
        assert str(allowed) in command
        return "'Example' depends on axioms: [Classical.choice]\n"
    monkeypatch.setattr(verify.subprocess, "check_output", checked)
    with pytest.raises(SystemExit, match="unexpected transitive axiom"):
        verify.audit(tmp_path, prefix, config)
