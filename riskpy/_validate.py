"""Scalar validators shared by the pure-Python layers.

Only checks that are identical, message for message, in every module that
uses them belong here. The differently worded variants in mc, life and
reserving stay local: merging them would change user-visible errors.
"""

from __future__ import annotations

import math


def _check_finite(name: str, value: float) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number, got {value!r}") from None
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return value


def _check_positive(name: str, value: float) -> float:
    value = _check_finite(name, value)
    if not value > 0.0:
        raise ValueError(f"{name} must be > 0, got {value!r}")
    return value


def _check_non_negative(name: str, value: float) -> float:
    value = _check_finite(name, value)
    if not value >= 0.0:
        raise ValueError(f"{name} must be >= 0, got {value!r}")
    return value
