"""Configuration-bound finite payouts and optional certified evidence."""
from dataclasses import dataclass
from fractions import Fraction
from math import isfinite


@dataclass(frozen=True)
class Schedule:
    values: tuple[float, ...]
    certificates: tuple[dict, ...]


class ScheduleCache:
    """Keep values and their certificates together across repeated episode horizons."""

    def __init__(self):
        self.config = None
        self.schedules = {}
        self.program = None
        self.program_sha256 = None

    def matches(self, spec, baseline, formula, validate_schedule, validate_transition):
        return self.config == (spec, baseline, formula, validate_schedule, validate_transition)

    def prepare(self, h, spec, baseline, formula, validate_schedule, validate_transition):
        if not self.matches(spec, baseline, formula, validate_schedule, validate_transition):
            self.schedules.clear()
            self.program = self.program_sha256 = None
            self.config = (spec, baseline, formula, validate_schedule, validate_transition)
        if h in self.schedules:
            return
        certificates = ()
        if formula is None:
            values = tuple(spec.correct_reward(Fraction(i, h - 1) if h > 1 else Fraction(1), baseline)
                           for i in range(h))
        else:
            from qb_env.acl_certified_curve import certify_payout

            if self.program is None:
                from qb_data.acl_protocol import digest

                self.program = formula(spec.arm, spec.start, spec.end, spec.shape,
                                       spec.threshold, spec.knots, baseline)
                self.program_sha256 = digest(self.program)
            checked = [certify_payout(self.program,
                                      Fraction(i, h - 1) if h > 1 else Fraction(1),
                                      validate_transition) for i in range(h)]
            values, certificates = zip(*checked)
        first = baseline if spec.arm == "baseline" else spec.start
        last = first if spec.arm in ("baseline", "constant") else spec.end
        if h == 1:
            first = last
        if (values[0] != first or values[-1] != last
                or not all(isfinite(v) for v in values)
                or any(a < b for a, b in zip(values, values[1:]))):
            raise ValueError(f"Invalid ACL reward schedule on horizon {h}")
        if validate_schedule is not None:
            validate_schedule(values, first, last)
        self.schedules[h] = Schedule(values, certificates)
