"""Scalar validators shared by the pure-Python layers.

Only checks that are identical, message for message, in every module that
uses them belong here. The differently worded variants in mc, life and
reserving stay local: merging them would change user-visible errors.
"""

from __future__ import annotations

import math

from .errors import InputError

# Parameters that are times in years: a bad value is RP106 (time out of range), with a hint about units.
_TIMES = frozenset({"T", "t", "t_1", "t_2", "maturity", "horizon"})
_TIME_HINT = "Times are in years (6 months is 0.5); a maturity or time to expiry must be positive."


def _range_error(name: str, message: str) -> InputError:
    if name in _TIMES:
        return InputError(message, code="RP106", param=name, hint=_TIME_HINT)
    return InputError(message, code="RP102", param=name)


def _check_finite(name: str, value: float) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise InputError(f"{name} must be a number, got {value!r}", code="RP101", param=name) from None
    if not math.isfinite(value):
        raise InputError(f"{name} must be finite, got {value!r}", code="RP101", param=name)
    return value


def _check_positive(name: str, value: float) -> float:
    value = _check_finite(name, value)
    if not value > 0.0:
        raise _range_error(name, f"{name} must be > 0, got {value!r}")
    return value


def _check_non_negative(name: str, value: float) -> float:
    value = _check_finite(name, value)
    if not value >= 0.0:
        raise _range_error(name, f"{name} must be >= 0, got {value!r}")
    return value
