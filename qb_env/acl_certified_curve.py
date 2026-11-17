"""Evaluate the proof-side schedule expression on an exact prefix grid."""

from __future__ import annotations

from fractions import Fraction
from operator import add, mul, sub, truediv
from threading import RLock

from qb_env.acl_settlement import TransitionValidator


_FLINT_LOCK = RLock()  # flint.ctx.prec is shared process-wide.
_BINARY = {"add": add, "sub": sub, "mul": mul, "div": truediv}


def _ball(flint, value):
    if isinstance(value, Fraction):
        return flint.arb(flint.fmpq(value.numerator, value.denominator))
    return value


def _power(flint, base, exponent):
    if isinstance(base, Fraction) and isinstance(exponent, Fraction):
        if base < 0 or exponent < 0:
            raise ValueError("ACL schedule powers require nonnegative operands")
        if exponent == 0:
            return Fraction(1)
        if base in (0, 1):
            return base
        # Exact perfect rational roots avoid undecidable midpoint ties when a
        # noninteger power nevertheless has a rational value. Use ball arithmetic
        # for large exponents; constructing huge intermediate integers is unnecessary.
        bits = max(base.numerator.bit_length(), base.denominator.bit_length())
        if exponent.denominator <= bits and exponent.numerator * bits <= 65536:
            n = exponent.denominator
            a = int(flint.fmpz(base.numerator).root(n))
            b = int(flint.fmpz(base.denominator).root(n))
            if a**n == base.numerator and b**n == base.denominator:
                return Fraction(a, b) ** exponent.numerator
    return _ball(flint, base) ** _ball(flint, exponent)


def _evaluate(flint, tree, progress):
    if not isinstance(tree, list) or not tree or not isinstance(tree[0], str):
        raise ValueError("invalid ACL curve opcode")
    op, *args = tree
    if op == "p" and not args:
        return progress
    if op == "q" and len(args) == 2:
        n, d = args
        if type(n) is int and type(d) is int and d > 0:
            return Fraction(n, d)
        raise ValueError("invalid rational curve literal")
    if op in {"if_lt", "if_le"} and len(args) == 4:
        lhs, rhs = (_evaluate(flint, expr, progress) for expr in args[:2])
        if not isinstance(lhs, Fraction) or not isinstance(rhs, Fraction):
            raise ValueError("curve selection requires exact rational comparisons")
        branch = lhs < rhs if op == "if_lt" else lhs <= rhs
        return _evaluate(flint, args[2 if branch else 3], progress)
    if op in {"cospi", "exp"} and len(args) == 1:
        value = _evaluate(flint, args[0], progress)
        if op == "exp":
            return Fraction(1) if isinstance(value, Fraction) and value == 0 else _ball(flint, value).exp()
        if isinstance(value, Fraction):
            return flint.arb.cos_pi_fmpq(flint.fmpq(value.numerator, value.denominator))
        return value.cos_pi()
    if (op in _BINARY or op == "pow") and len(args) == 2:
        lhs, rhs = (_evaluate(flint, expr, progress) for expr in args)
        if op == "pow":
            return _power(flint, lhs, rhs)
        if isinstance(lhs, Fraction) and isinstance(rhs, Fraction):
            return _BINARY[op](lhs, rhs)
        return _BINARY[op](_ball(flint, lhs), _ball(flint, rhs))
    raise ValueError(f"invalid ACL curve opcode or arity: {op}")


def _dyadic_bound(endpoint, *, lower: bool, precision: int) -> Fraction:
    """Serialize an outward dyadic even for exponents beyond feasible integer storage."""
    mantissa, exponent = map(int, endpoint.man_exp())
    if not mantissa:
        return Fraction(0)
    # Quantize outward far below the smallest binary64 (2**-1074). FLINT
    # can encode arbitrarily large negative exponents, e.g. exp(-1e308).
    floor_exponent = -max(1100, precision + 64)
    if exponent < floor_exponent:
        shift = floor_exponent - exponent
        if shift > abs(mantissa).bit_length():
            mantissa = -int(mantissa < 0) if lower else int(mantissa > 0)
        else:
            mantissa = (mantissa >> shift) if lower else -((-mantissa) >> shift)
        exponent = floor_exponent
    if exponent + abs(mantissa).bit_length() > 2048:
        raise ValueError("curve output exceeds the finite payout domain")
    return Fraction(mantissa << exponent) if exponent >= 0 else Fraction(mantissa, 1 << -exponent)


def certify_payout(program, progress: Fraction, validate: TransitionValidator, *,
                   claimed: float | None = None) -> tuple[float, dict]:
    """Enclose the emitted expression and certify both endpoints in one rounding cell.

    The progress argument is the mathematical prefix fraction, before conversion
    to binary64. The trusted Arb ball implementation includes transcendental errors
    in its enclosures. Unresolved ties exhaust the precision budget and fail closed.
    """
    import flint

    if flint.__version__ != "0.8.0":
        raise ValueError("verified ACL schedules require python-flint==0.8.0")
    if not isinstance(progress, Fraction) or not 0 <= progress <= 1:
        raise ValueError("certified progress must be an exact rational in [0,1]")
    with _FLINT_LOCK:
        for precision in (96, 192, 384, 768, 1536, 3072, 6144, 12288):
            with flint.ctx.workprec(precision):
                result = _evaluate(flint, program, progress)
                if isinstance(result, Fraction):
                    lo = hi = result
                    if max(result.numerator.bit_length(), result.denominator.bit_length()) > 12000:
                        # Keep JSON integer receipts within Python's default digit
                        # limit, even for e.g. exact (1/3)^30000.
                        scale = 1 << max(1100, precision + 64)
                        scaled = result.numerator * scale
                        lo = Fraction(scaled // result.denominator, scale)
                        hi = Fraction(-((-scaled) // result.denominator), scale)
                else:
                    if not result.is_finite():
                        continue
                    lo = _dyadic_bound(result.lower(), lower=True, precision=precision)
                    hi = _dyadic_bound(result.upper(), lower=False, precision=precision)
                try:
                    lower_float, upper_float = float(lo), float(hi)
                except OverflowError:
                    # An enclosure can cross the overflow edge at low precision.
                    continue
                if lower_float != upper_float:
                    continue
                value = lower_float if lower_float else 0.
                if claimed is not None and claimed != value:
                    raise ValueError("claimed schedule value differs from certified payout")
                validate(lo, 0., 0., 0., value)
                if hi != lo:
                    validate(hi, 0., 0., 0., value)
                return value, {
                    "progress": list(progress.as_integer_ratio()),
                    "lower": list(lo.as_integer_ratio()),
                    "upper": list(hi.as_integer_ratio()),
                    "precision_bits": 0 if isinstance(result, Fraction) and lo == hi else precision,
                }
    raise ValueError("could not certify a unique binary64 curve payout")
