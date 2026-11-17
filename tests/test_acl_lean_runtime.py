"""Exercise the compiled Lean bridge. Build acl_runtime to enable this optional suite."""

import itertools
import math
import sys
from fractions import Fraction

import pytest

from formal.acl_expansion.runtime import make_decoder


@pytest.fixture(scope="module")
def runtime():
    try:
        client = make_decoder()
    except FileNotFoundError:
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
    runtime.validate_transition(Fraction(1, 4), 0.5, 0.25, 2.0, 1.0)
    with pytest.raises(ValueError):
        runtime.validate_transition(Fraction(1, 4), 0.5, 0.25, 2.0, 0.75)


def test_formula_wire_roundtrip(runtime):
    def emit(arm, shape=0.5):
        return runtime.formula(arm, 1.0, 0.0, shape, 0.5, [(0.0, 1.0), (1.0, 0.0)], -0.25)

    assert emit("baseline") == ["q", -1, 4]
    assert emit("constant") == ["q", 1, 1]
    assert emit("exponential", 0.0) == emit("linear")
    assert emit("cosine")[2][2] == ["div", ["add", ["q", 1, 1], ["cospi", ["p"]]], ["q", 2, 1]]
    assert emit("step")[2][2] == ["if_lt", ["p"], ["q", 1, 2], ["q", 1, 1], ["q", 0, 1]]
    with pytest.raises(ValueError):
        emit("unrecognized")
