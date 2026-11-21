"""Scalar validators shared by the pure-Python layers.

Only checks that are identical, message for message, in every module that
uses them belong here. The differently worded variants in mc, life and
reserving stay local: merging them would change user-visible errors.
"""

from __future__ import annotations

import math
import numbers

from .errors import InputError

# Parameters that are times in years: a bad value is RP106 (time out of range), with a hint about units.
_TIMES = frozenset({"T", "t", "t_1", "t_2", "maturity", "horizon"})
_TIME_HINT = "Times are in years (6 months is 0.5); a maturity or time to expiry must be positive."

_NOT_A_NUMBER_HINT = ("Pass a number. Numeric strings such as \"100\" and booleans are refused, not coerced: "
                      "True is not 1.0 here.")
_SEED_HINT = ("A seed is a whole number >= 0 (42, say) that makes the draws repeatable; None draws fresh "
              "entropy. A numpy.random.Generator or SeedSequence is accepted too.")


def _range_error(name: str, message: str) -> InputError:
    if name in _TIMES:
        return InputError(message, code="RP106", param=name, hint=_TIME_HINT)
    return InputError(message, code="RP102", param=name)


def _is_text_or_flag(value: object) -> bool:
    """True for values ``float()`` would happily coerce but that are not numbers.

    ``float("100")``, ``float(b"1")`` and ``float(True)`` all succeed, so a typo'd
    string or a flag passed in the wrong position used to be priced as if it were a
    number. NumPy values are recognised by their dtype, without importing NumPy: a
    ``numpy.bool_``, a 0-d array of text or booleans, and any array with a dimension
    (which is not a scalar at all) are refused; NumPy integers and floats, and 0-d
    numeric arrays, are numbers.
    """
    if isinstance(value, (bool, str, bytes, bytearray)):
        return True
    dtype = getattr(value, "dtype", None)
    if dtype is not None and not isinstance(value, numbers.Real):
        return getattr(dtype, "kind", None) not in ("i", "u", "f") or getattr(value, "ndim", 0) != 0
    if dtype is not None:  # a NumPy scalar registered as Real; numpy.bool_ is not, and is caught above
        return getattr(dtype, "kind", None) not in ("i", "u", "f")
    return False


def _check_number(name: str, value: object) -> float:
    """``value`` as a float, refusing what is not a number. NaN and infinity pass."""
    if _is_text_or_flag(value):
        raise InputError(f"{name} must be a number, got {value!r}", code="RP101", param=name,
                         hint=_NOT_A_NUMBER_HINT)
    try:
        return float(value)  # type: ignore[arg-type]
    except OverflowError:
        # An int past ~1.8e308 has no float; repr() of it could be thousands of digits.
        raise InputError(f"{name} is too large to be a floating-point number (the limit is about 1.8e308)",
                         code="RP101", param=name,
                         hint="Pass a finite number of a realistic size.") from None
    except (TypeError, ValueError):
        raise InputError(f"{name} must be a number, got {value!r}", code="RP101", param=name) from None


def _check_finite(name: str, value: float) -> float:
    value = _check_number(name, value)
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


def _is_whole(value: object) -> bool:
    """An integer, Python or NumPy, but not a bool (``True`` is an ``Integral``)."""
    return isinstance(value, numbers.Integral) and not isinstance(value, bool) and not _is_text_or_flag(value)


def _check_trials(name: str, value: object, minimum: int = 1) -> int:
    """A whole count such as ``trials`` or ``steps``: an integer >= ``minimum``.

    ``2.5`` trials, ``True`` trials and ``None`` are refused with RP107 instead of
    surfacing as NumPy's ``TypeError`` (or, for ``True``, quietly meaning one trial).
    """
    if not _is_whole(value) or value < minimum:  # type: ignore[operator]
        raise InputError(f"{name} must be a positive integer, got {value!r}" if minimum == 1 else
                         f"{name} must be an integer >= {minimum}, got {value!r}", code="RP107", param=name)
    return int(value)  # type: ignore[arg-type]


def _check_seed(seed: object, name: str = "seed") -> object:
    """``None``, a non-negative integer, or a NumPy ``Generator``/``BitGenerator``/``SeedSequence``.

    NumPy refuses ``-1`` and ``1.5`` itself, but with a bare ``ValueError`` or
    ``TypeError`` from deep inside ``default_rng``; this names the parameter. A list of
    non-negative integers (NumPy's entropy pool form) is accepted as well.
    """
    if seed is None:
        return None
    if _is_whole(seed):
        if seed < 0:  # type: ignore[operator]
            raise InputError(f"{name} must be >= 0, got {seed!r}", code="RP102", param=name, hint=_SEED_HINT)
        return int(seed)  # type: ignore[arg-type]
    module = type(seed).__module__ or ""
    if module.startswith("numpy.random"):
        return seed
    if isinstance(seed, (list, tuple)) and seed and all(_is_whole(s) for s in seed):
        if any(s < 0 for s in seed):
            raise InputError(f"{name} entries must be >= 0, got {seed!r}", code="RP102", param=name, hint=_SEED_HINT)
        return seed
    raise InputError(f"{name} must be a non-negative integer or None, got {seed!r}", code="RP107", param=name,
                     hint=_SEED_HINT)
