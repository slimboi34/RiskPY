"""Life contingencies — life tables, actuarial present values, premiums, reserves.

The life-and-pensions half of actuarial science: a survival model (a life
table), an interest rate, and the handful of present-value formulas that turn
the two into insurance and annuity prices, premiums and reserves.

    from riskpy import life

    sult = life.LifeTable.sult()                       # AMLCR standard table
    A40 = life.whole_life_insurance(sult, 40, i=0.05)  # 0.121059
    a40 = life.whole_life_annuity_due(sult, 40, i=0.05)  # 18.457757
    P = life.whole_life_premium(sult, 40, i=0.05)      # A40 / a40 = 0.006559

This module is **pure Python** — ``math`` only, no NumPy — so it works in the
zero-dependency core install. Everything here is scalar arithmetic over a table
of at most a couple of hundred ages; NumPy would buy nothing.

Conventions
-----------
* Ages and durations are **integers**. A ``LifeTable`` is a discrete survival
  model: ``q_x`` is the probability that a life aged exactly ``x`` dies before
  age ``x + 1``. Fractional ages are not modelled.
* ``i`` is the effective annual interest rate as a decimal (``0.05``, not
  ``5``). ``v = 1/(1+i)`` is the discount factor, ``d = iv`` the discount rate
  and ``δ = ln(1+i)`` the force of interest.
* Insurances pay 1 at the **end of the year of death** (``A_x``); annuities-due
  pay 1 at the **start** of each year while alive (``ä_x``). The
  ``continuous=True`` flag on each present-value function applies the uniform
  distribution of deaths (UDD) adjustment ``Ā = (i/δ)·A`` to insurances and the
  two-term Woolhouse approximation ``ā ≈ ä − ½`` to annuities. Both are
  approximations; each docstring says how good.
* A table must **close**: its final ``q_x`` is 1, so ``l_ω = 0`` at the
  limiting age ``ω``. Every identity in this module (``A_x = 1 − d·ä_x`` and
  friends) depends on it, which is why a table that does not close is rejected
  rather than quietly truncated.
* Tables are **aggregate / ultimate** only. Select tables (mortality that
  depends on time since underwriting as well as on age) and multiple-decrement
  tables are out of scope.

References: Bowers, Gerber, Hickman, Jones & Nesbitt, *Actuarial Mathematics*
(2nd ed., 1997); Dickson, Hardy & Waters, *Actuarial Mathematics for Life
Contingent Risks* (AMLCR, 2nd ed., 2013), whose Standard Ultimate Life Table
is :meth:`LifeTable.sult`.
"""

from __future__ import annotations

import math
import numbers
import warnings
from typing import Callable, List, Optional, Sequence, Tuple

__all__ = [
    "LifeTable",
    # interest
    "discount_factor",
    "discount_rate",
    "force_of_interest",
    "nominal_rate",
    "effective_from_nominal",
    "annuity_certain",
    "accumulated_annuity",
    "perpetuity",
    # insurances
    "whole_life_insurance",
    "term_insurance",
    "pure_endowment",
    "endowment_insurance",
    "deferred_whole_life_insurance",
    "increasing_insurance",
    # annuities
    "whole_life_annuity_due",
    "whole_life_annuity_immediate",
    "temporary_annuity_due",
    "temporary_annuity_immediate",
    "deferred_annuity_due",
    "increasing_annuity_due",
    "mthly_annuity_due",
    # commutation
    "Commutation",
    # premiums
    "net_premium",
    "whole_life_premium",
    "term_premium",
    "endowment_premium",
    "gross_premium",
    # reserves
    "net_premium_reserve",
    "reserve_profile",
    # joint life
    "joint_life_annuity_due",
    "last_survivor_annuity_due",
    "joint_life_insurance",
    "last_survivor_insurance",
]


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _check_finite(name: str, value: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number, got {value!r}") from None
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return out


def _check_int(name: str, value: int, minimum: Optional[int] = None) -> int:
    """Accept ints (and integral floats such as ``40.0``), reject everything else.

    Ages and durations are whole years in a discrete table; ``40.5`` is not an
    off-by-one, it is a request for fractional-age mortality this module does
    not model, so it is refused rather than rounded.
    """
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    if isinstance(value, numbers.Integral):
        out = int(value)
    elif isinstance(value, float) and math.isfinite(value) and value.is_integer():
        out = int(value)
    else:
        raise ValueError(
            f"{name} must be a whole number of years, got {value!r} — this is a "
            f"discrete (integer-age) table"
        )
    if minimum is not None and out < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value!r}")
    return out


def _check_rate(i: float) -> float:
    i = _check_finite("i", i)
    if not (i > -1.0):
        raise ValueError(
            f"i must be an effective annual rate > -1, as a decimal (0.05 for 5%), got {i!r}"
        )
    return i


def _check_table(table: "LifeTable") -> "LifeTable":
    if not isinstance(table, LifeTable):
        raise TypeError(f"table must be a LifeTable, got {type(table).__name__}")
    return table


# ---------------------------------------------------------------------------
# Life table
# ---------------------------------------------------------------------------


def _close_generated(rates: List[float], start_age: int, max_age: int) -> List[float]:
    """Close a table built from a mortality law at the first age where death is
    certain, warning that it stops short of the age that was asked for.

    A law with a steep ``c`` reaches ``q_x = 1`` in floating point well before
    any plausible limiting age — Gompertz with ``c = 1.15`` is certain by the
    mid-nineties — and ages past that point describe lives that cannot exist.
    Trimming them is right for a generated table, but silently returning a
    shorter table than the caller asked for is the kind of surprise that shows
    up much later as an off-by-thirty-years in a reserve, so it warns.
    """
    certain = next((k for k, q in enumerate(rates) if q >= 1.0), None)
    if certain is None:
        rates = list(rates)
        rates.append(1.0)
        return rates
    if certain == len(rates) - 1:
        return list(rates)
    omega = start_age + certain + 1
    warnings.warn(
        f"mortality reaches certainty at age {start_age + certain}: q_"
        f"{start_age + certain} = 1 under these parameters, so the table closes "
        f"at omega = {omega} rather than at the requested max_age {max_age}. "
        f"The dropped ages describe lives no one can reach.",
        RuntimeWarning,
        stacklevel=3,
    )
    return list(rates[: certain + 1])


class LifeTable:
    """A discrete survival model: one-year death probabilities ``q_x`` by age.

    Construct from mortality rates (``LifeTable(qx, start_age)``), from
    survivors (:meth:`from_lx`), from a parametric law (:meth:`gompertz`,
    :meth:`makeham`, :meth:`from_force`) or take the textbook standard
    (:meth:`sult`).

    ``qx[k]`` is ``q_{start_age + k}``. The table runs from ``start_age`` to
    ``omega - 1`` and must close: the last ``q_x`` is 1.0, so nobody survives to
    ``omega``. Survivors ``l_x`` are built from the radix downwards, so
    ``l_omega == 0`` exactly and ``d_x = l_x - l_{x+1} = l_x · q_x``.

    A ``q_x`` of 1 anywhere but the last position is rejected: it would close
    the table early and leave unreachable ages behind it, which is nearly always
    a data error (an age column misaligned by one, say) rather than intent.

    The thing people get wrong: a table copied from a book that stops at age
    100 with ``l_100 > 0`` is *not* a survival model — six percent of the SULT
    is still alive at 100 — and pricing a whole-life contract on it drops that
    benefit silently. Append ``q = 1`` (everyone alive at the last age dies in
    the year) or extend the table; the constructor tells you which.

    Numerical note: if ``l_x`` underflows to zero (survival below about 1e-308
    of the radix, a cumulative hazard above ~700) before the supplied rates run
    out, the table closes at that age with a ``RuntimeWarning``. Nothing
    representable is lost, but you are told.
    """

    def __init__(self, qx: Sequence[float], start_age: int = 0, radix: float = 100_000.0):
        start_age = _check_int("start_age", start_age, minimum=0)
        radix = _check_finite("radix", radix)
        if not (radix > 0.0):
            raise ValueError(f"radix must be > 0, got {radix!r}")

        rates = [_check_finite(f"q_{start_age + k}", q) for k, q in enumerate(qx)]
        if not rates:
            raise ValueError("qx is empty — a life table needs at least one age")
        for k, q in enumerate(rates):
            if not (0.0 <= q <= 1.0):
                raise ValueError(
                    f"q_{start_age + k} = {q!r} is not a probability in [0, 1]"
                )
        first_one = next((k for k, q in enumerate(rates) if q == 1.0), None)
        if first_one is None:
            last = start_age + len(rates) - 1
            raise ValueError(
                f"the table does not close: q_{last} = {rates[-1]!r} but the last "
                f"q_x must be 1.0 (l_omega = 0). Append 1.0 so that everyone alive "
                f"at age {last + 1} dies within the year, or extend the table."
            )
        if first_one != len(rates) - 1:
            raise ValueError(
                f"q_{start_age + first_one} = 1.0 closes the table at age "
                f"{start_age + first_one + 1}, but {len(rates) - first_one - 1} more "
                f"age(s) follow it. Drop the unreachable ages, or fix the rate."
            )

        survivors = [radix]
        for k, q in enumerate(rates):
            nxt = survivors[-1] * (1.0 - q)
            survivors.append(nxt)
            if nxt == 0.0 and k != len(rates) - 1:
                closing_age = start_age + k + 1
                warnings.warn(
                    f"l_x underflowed to zero at age {closing_age}, before the table's "
                    f"last age {start_age + len(rates) - 1}; closing the table at "
                    f"omega = {closing_age} (q_{closing_age - 1} set to 1) and dropping "
                    f"the later ages, which no representable life could reach.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                rates = rates[: k + 1]
                rates[-1] = 1.0
                break

        self._start = start_age
        self._radix = float(radix)
        self._q: List[float] = rates
        self._l: List[float] = survivors
        self._omega = start_age + len(rates)

    # -- constructors -------------------------------------------------------

    @classmethod
    def from_lx(cls, lx: Sequence[float], start_age: int = 0) -> "LifeTable":
        """Build from a survivor column ``l_x``; ``lx[0]`` becomes the radix.

        ``l_x`` must be positive, non-increasing, and reach zero: the first zero
        is ``l_omega``. Zeros after it carry no information and are dropped.
        The supplied survivors are stored exactly (not re-derived from the
        implied ``q_x``), so ``lx(x)`` returns what you passed in.
        """
        start_age = _check_int("start_age", start_age, minimum=0)
        values = [_check_finite(f"l_{start_age + k}", v) for k, v in enumerate(lx)]
        if len(values) < 2:
            raise ValueError("lx needs at least two entries (a radix and a later age)")
        if not (values[0] > 0.0):
            raise ValueError(f"l_{start_age} (the radix) must be > 0, got {values[0]!r}")
        for k in range(1, len(values)):
            if values[k] < 0.0:
                raise ValueError(f"l_{start_age + k} = {values[k]!r} is negative")
            if values[k] > values[k - 1]:
                raise ValueError(
                    f"l_x must be non-increasing, but l_{start_age + k} = {values[k]!r} "
                    f"> l_{start_age + k - 1} = {values[k - 1]!r}"
                )
        first_zero = next((k for k, v in enumerate(values) if v == 0.0), None)
        if first_zero is None:
            last = start_age + len(values) - 1
            raise ValueError(
                f"the table does not close: l_{last} = {values[-1]!r} > 0. Append 0.0 "
                f"(nobody survives past age {last}) or extend the table."
            )
        values = values[: first_zero + 1]

        rates = [1.0 - values[k + 1] / values[k] for k in range(len(values) - 1)]
        table = cls(rates, start_age=start_age, radix=values[0])
        table._l = values  # keep the caller's survivors exactly
        return table

    @classmethod
    def gompertz(
        cls,
        B: float,
        c: float,
        start_age: int = 0,
        max_age: int = 130,
        radix: float = 100_000.0,
    ) -> "LifeTable":
        """Gompertz law, force of mortality ``μ_x = B·c^x``.

        Exponentially rising hazard — a good fit to adult human mortality from
        about age 30 to 90 with ``c`` near 1.1. Uses the closed form
        ``t_p_x = exp(−B·c^x·(c^t − 1)/ln c)``, so ``q_x`` is exact.

        The law is applied for ages ``start_age`` to ``max_age``, and the table
        is closed by setting ``q_{max_age} = 1``. With realistic parameters the
        survivors that closure kills off are a vanishing fraction of the radix;
        check ``lx(max_age)`` if in doubt.
        """
        return cls.makeham(0.0, B, c, start_age=start_age, max_age=max_age, radix=radix)

    @classmethod
    def makeham(
        cls,
        A: float,
        B: float,
        c: float,
        start_age: int = 0,
        max_age: int = 130,
        radix: float = 100_000.0,
    ) -> "LifeTable":
        """Makeham's law, force of mortality ``μ_x = A + B·c^x``.

        Gompertz plus a constant ``A`` for age-independent hazards (accidents).
        Uses the closed form ``t_p_x = exp(−A·t − B·c^x·(c^t − 1)/ln c)``, so
        ``q_x`` is exact — :meth:`from_force` integrates numerically and is the
        general tool; this is the one to use when the law is Makeham.

        The law is applied for ages ``start_age`` to ``max_age``, and the table
        is closed by ``q_{max_age} = 1``. See :meth:`gompertz`.
        """
        A = _check_finite("A", A)
        B = _check_finite("B", B)
        c = _check_finite("c", c)
        if A < 0.0:
            raise ValueError(f"A must be >= 0, got {A!r}")
        if not (B > 0.0):
            raise ValueError(f"B must be > 0, got {B!r}")
        if not (c > 1.0):
            raise ValueError(
                f"c must be > 1 (mortality rising with age), got {c!r}. For a "
                f"constant force use from_force(lambda x: A)."
            )
        start_age, max_age = _check_age_span(start_age, max_age)

        log_c = math.log(c)
        rates = [
            1.0 - math.exp(-A - B * c ** x * (c - 1.0) / log_c)
            for x in range(start_age, max_age)
        ]
        rates = _close_generated(rates, start_age, max_age)
        return cls(rates, start_age=start_age, radix=radix)

    @classmethod
    def sult(cls) -> "LifeTable":
        """The Standard Ultimate Life Table of Dickson, Hardy & Waters (AMLCR).

        Makeham with ``A = 0.00022``, ``B = 2.7e-6``, ``c = 1.124``, ages 20 to
        130, radix ``l_20 = 100 000``. Reproduces the book's Appendix D to the
        printed precision: ``l_100 = 6248.17``, and at 5% ``ä_40 = 18.4578``,
        ``A_40 = 0.12106``. Handy as a test bed because so many worked answers
        exist for it; not a table to price real business on.
        """
        return cls.makeham(0.00022, 2.7e-6, 1.124, start_age=20, max_age=130)

    @classmethod
    def from_force(
        cls,
        mu: Callable[[float], float],
        start_age: int = 0,
        max_age: int = 130,
        radix: float = 100_000.0,
        steps: int = 64,
    ) -> "LifeTable":
        """Build from a force of mortality ``μ(x)`` by numerical integration.

        ``q_x = 1 − exp(−∫_x^{x+1} μ(s) ds)``, the integral by composite
        Simpson's rule with ``steps`` (even) sub-intervals per year. Simpson is
        exact for cubics, so for any smooth hazard the error is of order
        ``(1/steps)^4 · μ''''`` — invisible at the default 64 for Gompertz-like
        laws (about 1e-12 in ``q_x`` even where ``μ`` is 10 per year).

        ``μ`` must be finite and non-negative at every node; a negative force
        would let survivors *increase*. The table is closed by ``q_{max_age} = 1``.
        """
        if not callable(mu):
            raise TypeError("mu must be callable, e.g. lambda x: 0.0002 + 2.7e-6 * 1.124 ** x")
        start_age, max_age = _check_age_span(start_age, max_age)
        steps = _check_int("steps", steps, minimum=2)
        if steps % 2:
            raise ValueError(f"steps must be even (Simpson's rule pairs intervals), got {steps}")

        h = 1.0 / steps

        def force(s: float) -> float:
            value = _check_finite(f"mu({s:g})", mu(s))
            if value < 0.0:
                raise ValueError(f"mu({s:g}) = {value!r} is negative; a force of mortality must be >= 0")
            return value

        rates = []
        for x in range(start_age, max_age):
            total = force(float(x)) + force(float(x + 1))
            for j in range(1, steps):
                total += (4.0 if j % 2 else 2.0) * force(x + j * h)
            rates.append(1.0 - math.exp(-total * h / 3.0))
        rates = _close_generated(rates, start_age, max_age)
        return cls(rates, start_age=start_age, radix=radix)

    # -- basic properties ---------------------------------------------------

    @property
    def start_age(self) -> int:
        return self._start

    @property
    def omega(self) -> int:
        """The limiting age: the first age with ``l_x = 0`` (``last age + 1``)."""
        return self._omega

    @property
    def radix(self) -> float:
        return self._radix

    @property
    def ages(self) -> List[int]:
        """Every age that carries a ``q_x``: ``start_age`` to ``omega - 1``."""
        return list(range(self._start, self._omega))

    def __len__(self) -> int:
        return self._omega - self._start

    def __repr__(self) -> str:
        return f"LifeTable(ages {self._start}..{self._omega - 1}, radix {self._radix:g})"

    def _index(self, x: int, allow_omega: bool = False) -> int:
        x = _check_int("x", x)
        top = self._omega if allow_omega else self._omega - 1
        if not (self._start <= x <= top):
            raise ValueError(
                f"age {x} is outside the table, which runs from {self._start} to "
                f"{self._omega - 1} (omega = {self._omega})"
            )
        return x - self._start

    # -- single-age quantities ---------------------------------------------

    def qx(self, x: int) -> float:
        """``q_x``: probability that ``(x)`` dies before age ``x + 1``."""
        return self._q[self._index(x)]

    def px(self, x: int) -> float:
        """``p_x = 1 − q_x``."""
        return 1.0 - self._q[self._index(x)]

    def lx(self, x: int) -> float:
        """``l_x``: expected survivors to age ``x`` out of the radix. Valid up to ``omega`` (where it is 0)."""
        return self._l[self._index(x, allow_omega=True)]

    def dx(self, x: int) -> float:
        """``d_x = l_x − l_{x+1}``: expected deaths between ages ``x`` and ``x + 1``."""
        k = self._index(x)
        return self._l[k] - self._l[k + 1]

    # -- multi-year probabilities ------------------------------------------

    def tpx(self, t: int, x: int) -> float:
        """``t_p_x``: probability that ``(x)`` survives ``t`` whole years.

        Argument order follows the actuarial notation (``t`` first, then ``x``),
        which is the opposite of how you would say it — watch for it. Beyond
        ``omega`` the answer is 0, not an error, so sums can run to any length.
        """
        k = self._index(x)
        t = _check_int("t", t, minimum=0)
        if k + t >= len(self._l) - 1:
            return 0.0
        return self._l[k + t] / self._l[k]

    def tqx(self, t: int, x: int) -> float:
        """``t_q_x = 1 − t_p_x``: probability that ``(x)`` dies within ``t`` years."""
        return 1.0 - self.tpx(t, x)

    def deferred_qx(self, t: int, x: int) -> float:
        """``t|q_x = t_p_x · q_{x+t}``: ``(x)`` dies in the year starting at age ``x + t``.

        These sum to 1 over ``t = 0, 1, …`` for a closed table — the discrete
        distribution of the curtate future lifetime ``K_x``.
        """
        k = self._index(x)
        t = _check_int("t", t, minimum=0)
        if k + t >= len(self._q):
            return 0.0
        return self._l[k + t] * self._q[k + t] / self._l[k]

    # -- life expectancy -----------------------------------------------------

    def curtate_expectation(self, x: int) -> float:
        """``e_x = Σ_{k≥1} k_p_x``: expected number of *whole* future years lived by ``(x)``."""
        k = self._index(x)
        l_x = self._l[k]
        return sum(self._l[k + 1:]) / l_x

    def complete_expectation(self, x: int) -> float:
        """``e°_x ≈ e_x + ½``: expected future lifetime including the fraction of the final year.

        The ``+½`` assumes deaths are uniformly distributed within each year
        (UDD). It is exact under that assumption and within a few hundredths of
        a year otherwise; a discrete table cannot do better on its own.
        """
        return self.curtate_expectation(x) + 0.5

    # -- data for charts -----------------------------------------------------

    def survival_curve(self) -> Tuple[List[int], List[float]]:
        """``(ages, l_x)`` from ``start_age`` to ``omega`` inclusive — the survival curve for :mod:`riskpy.viz`."""
        return list(range(self._start, self._omega + 1)), list(self._l)

    def mortality_curve(self) -> Tuple[List[int], List[float]]:
        """``(ages, q_x)`` for every table age — plot on a log axis to see the Gompertz straight line."""
        return self.ages, list(self._q)


def _check_age_span(start_age: int, max_age: int) -> Tuple[int, int]:
    start_age = _check_int("start_age", start_age, minimum=0)
    max_age = _check_int("max_age", max_age)
    if max_age <= start_age:
        raise ValueError(f"max_age must exceed start_age, got start_age={start_age} max_age={max_age}")
    return start_age, max_age


# ---------------------------------------------------------------------------
# Interest
# ---------------------------------------------------------------------------


def discount_factor(i: float) -> float:
    """``v = 1/(1+i)``: value now of 1 due in a year."""
    return 1.0 / (1.0 + _check_rate(i))


def discount_rate(i: float) -> float:
    """``d = i/(1+i) = 1 − v``: interest paid *in advance*.

    The rate that appears in every annuity-due formula (``ä = (1 − vⁿ)/d``) and
    in ``A_x = 1 − d·ä_x``. Mixing ``d`` up with ``i`` is the classic exam slip.
    """
    i = _check_rate(i)
    return i / (1.0 + i)


def force_of_interest(i: float) -> float:
    """``δ = ln(1+i)``: the continuously compounded equivalent of ``i``."""
    return math.log1p(_check_rate(i))


def nominal_rate(i: float, m: int) -> float:
    """``i^(m) = m·((1+i)^{1/m} − 1)``: nominal annual rate convertible ``m`` times a year."""
    i = _check_rate(i)
    m = _check_int("m", m, minimum=1)
    return m * ((1.0 + i) ** (1.0 / m) - 1.0)


def effective_from_nominal(i_m: float, m: int) -> float:
    """``i = (1 + i^(m)/m)^m − 1``: the inverse of :func:`nominal_rate`."""
    i_m = _check_finite("i_m", i_m)
    m = _check_int("m", m, minimum=1)
    base = 1.0 + i_m / m
    if not (base > 0.0):
        raise ValueError(f"i_m/m must exceed -1, got i_m={i_m!r} m={m}")
    return base ** m - 1.0


def annuity_certain(i: float, n: int, due: bool = False) -> float:
    """Present value of ``n`` level payments of 1: ``a_n = (1 − vⁿ)/i`` or, if ``due``, ``ä_n = (1 − vⁿ)/d``.

    Immediate (the default) pays at the *end* of each year, due at the start;
    ``ä_n = (1+i)·a_n``. At ``i = 0`` both are simply ``n``.
    """
    i = _check_rate(i)
    n = _check_int("n", n, minimum=0)
    if i == 0.0:
        return float(n)
    v_n = (1.0 + i) ** (-n)
    return (1.0 - v_n) / (i / (1.0 + i) if due else i)


def accumulated_annuity(i: float, n: int, due: bool = False) -> float:
    """Value at time ``n`` of ``n`` payments of 1: ``s_n = ((1+i)ⁿ − 1)/i``, or ``s̈_n`` with ``d`` if ``due``."""
    i = _check_rate(i)
    n = _check_int("n", n, minimum=0)
    if i == 0.0:
        return float(n)
    growth = (1.0 + i) ** n
    return (growth - 1.0) / (i / (1.0 + i) if due else i)


def perpetuity(i: float, due: bool = False) -> float:
    """``a_∞ = 1/i`` (or ``ä_∞ = 1/d``). Needs ``i > 0``: at zero interest the sum diverges."""
    i = _check_rate(i)
    if not (i > 0.0):
        raise ValueError(f"a perpetuity needs i > 0 (the sum diverges otherwise), got {i!r}")
    return 1.0 / (i / (1.0 + i) if due else i)


def _udd_factor(i: float) -> float:
    """``i/δ``: the UDD claims-acceleration factor, 1 at ``i = 0``."""
    if i == 0.0:
        return 1.0
    return i / math.log1p(i)


# ---------------------------------------------------------------------------
# Present-value engines (private)
# ---------------------------------------------------------------------------


def _insurance_sum(table: LifeTable, x: int, n: Optional[int], i: float) -> float:
    """``Σ_{k=0}^{n-1} v^{k+1} · k|q_x``, capped at the table's end."""
    k0 = table._index(x)
    v = 1.0 / (1.0 + i)
    limit = len(table._q) - k0 if n is None else min(n, len(table._q) - k0)
    l_x = table._l[k0]
    total = 0.0
    disc = v
    for k in range(limit):
        total += disc * (table._l[k0 + k] - table._l[k0 + k + 1])
        disc *= v
    return total / l_x


def _annuity_due_sum(table: LifeTable, x: int, n: Optional[int], i: float) -> float:
    """``Σ_{k=0}^{n-1} v^k · k_p_x``, capped at the table's end."""
    k0 = table._index(x)
    v = 1.0 / (1.0 + i)
    limit = len(table._q) - k0 if n is None else min(n, len(table._q) - k0)
    l_x = table._l[k0]
    total = 0.0
    disc = 1.0
    for k in range(limit):
        total += disc * table._l[k0 + k]
        disc *= v
    return total / l_x


def _pure_endowment(table: LifeTable, x: int, n: int, i: float) -> float:
    return (1.0 + i) ** (-n) * table.tpx(n, x)


# ---------------------------------------------------------------------------
# Insurances
# ---------------------------------------------------------------------------


def whole_life_insurance(table: LifeTable, x: int, i: float, continuous: bool = False) -> float:
    """``A_x``: present value of 1 paid at the end of the year of death of ``(x)``.

    ``A_x = Σ_k v^{k+1} · k|q_x``. Check it against ``1 − d·ä_x``, which holds
    exactly for a closed table.

    ``continuous=True`` returns ``Ā_x ≈ (i/δ)·A_x``, the payment-at-the-moment-
    of-death value under the uniform distribution of deaths. The factor is
    exact under UDD; under any other fractional-age assumption it is off by a
    few parts in ten thousand at ordinary interest rates.
    """
    _check_table(table)
    i = _check_rate(i)
    value = _insurance_sum(table, x, None, i)
    return _udd_factor(i) * value if continuous else value


def term_insurance(table: LifeTable, x: int, n: int, i: float, continuous: bool = False) -> float:
    """``A¹_{x:n}``: 1 at the end of the year of death, only if death is within ``n`` years.

    ``n`` may exceed the years left in the table, in which case the cover is
    effectively whole life. ``n = 0`` is worth nothing. ``continuous`` applies
    the UDD factor ``i/δ`` as in :func:`whole_life_insurance`.
    """
    _check_table(table)
    n = _check_int("n", n, minimum=0)
    i = _check_rate(i)
    value = _insurance_sum(table, x, n, i)
    return _udd_factor(i) * value if continuous else value


def pure_endowment(table: LifeTable, x: int, n: int, i: float) -> float:
    """``nE_x = vⁿ · n_p_x``: 1 paid at time ``n`` if ``(x)`` is then alive.

    Also the actuarial discount factor: ``nE_x · ä_{x+n}`` is the value now of
    an annuity that starts in ``n`` years, and ``nE_x`` at ``n = 0`` is 1.
    """
    _check_table(table)
    n = _check_int("n", n, minimum=0)
    i = _check_rate(i)
    return _pure_endowment(table, x, n, i)


def endowment_insurance(table: LifeTable, x: int, n: int, i: float, continuous: bool = False) -> float:
    """``A_{x:n} = A¹_{x:n} + nE_x``: 1 on death within ``n`` years, or at time ``n`` on survival.

    ``continuous`` accelerates only the death benefit (``(i/δ)·A¹_{x:n}``);
    the survival benefit is paid at a fixed date and is not adjusted.
    """
    return term_insurance(table, x, n, i, continuous) + pure_endowment(table, x, n, i)


def deferred_whole_life_insurance(
    table: LifeTable, x: int, m: int, i: float, continuous: bool = False
) -> float:
    """``m|A_x = mE_x · A_{x+m}``: whole life cover that only starts after ``m`` years.

    Equals ``A_x − A¹_{x:m}``. Zero if ``(x)`` cannot survive the deferral.
    """
    _check_table(table)
    m = _check_int("m", m, minimum=0)
    i = _check_rate(i)
    k0 = table._index(x)
    if k0 + m >= len(table._q):
        return 0.0
    value = _pure_endowment(table, x, m, i) * _insurance_sum(table, x + m, None, i)
    return _udd_factor(i) * value if continuous else value


def increasing_insurance(table: LifeTable, x: int, i: float, continuous: bool = False) -> float:
    """``(IA)_x = Σ_k (k+1) v^{k+1} · k|q_x``: benefit of ``k + 1`` for death in year ``k + 1``.

    Satisfies ``(IA)_x = A_x + v·p_x·(IA)_{x+1}``. ``continuous`` gives
    ``(IĀ)_x ≈ (i/δ)·(IA)_x`` — the benefit still steps up annually, only the
    payment moves to the moment of death, so the UDD factor applies unchanged.
    """
    _check_table(table)
    i = _check_rate(i)
    k0 = table._index(x)
    v = 1.0 / (1.0 + i)
    total = 0.0
    disc = v
    for k in range(len(table._q) - k0):
        total += (k + 1) * disc * (table._l[k0 + k] - table._l[k0 + k + 1])
        disc *= v
    value = total / table._l[k0]
    return _udd_factor(i) * value if continuous else value


# ---------------------------------------------------------------------------
# Annuities
# ---------------------------------------------------------------------------


def whole_life_annuity_due(table: LifeTable, x: int, i: float, continuous: bool = False) -> float:
    """``ä_x = Σ_k v^k · k_p_x``: 1 at the start of each year ``(x)`` is alive.

    The first payment is certain (``k = 0``), which is why ``ä_x ≥ 1`` and why
    the annuity-immediate is one payment less: ``a_x = ä_x − 1``.

    ``continuous=True`` returns the two-term Woolhouse approximation
    ``ā_x ≈ ä_x − ½``. The omitted third term is ``−(δ + μ_x)/12``, about
    −0.004 at 5% for a young life and larger at old ages where ``μ_x`` grows;
    the same ``½`` is what ``complete_expectation`` adds, and for the same
    reason. Note this is *not* algebraically the UDD-consistent value
    ``(1 − Ā_x)/δ``; the two differ at the third decimal place.
    """
    _check_table(table)
    i = _check_rate(i)
    value = _annuity_due_sum(table, x, None, i)
    return value - 0.5 if continuous else value


def whole_life_annuity_immediate(table: LifeTable, x: int, i: float, continuous: bool = False) -> float:
    """``a_x = ä_x − 1``: 1 at the *end* of each year ``(x)`` survives.

    ``continuous=True`` returns ``ā_x ≈ ä_x − ½`` — the continuous annuity is
    the same quantity whichever discrete version you start from.
    """
    _check_table(table)
    i = _check_rate(i)
    value = _annuity_due_sum(table, x, None, i)
    return value - 0.5 if continuous else value - 1.0


def temporary_annuity_due(table: LifeTable, x: int, n: int, i: float, continuous: bool = False) -> float:
    """``ä_{x:n} = Σ_{k=0}^{n-1} v^k · k_p_x``: at most ``n`` payments, in advance.

    ``ä_{x:n} = ä_x − nE_x · ä_{x+n}``. ``continuous`` uses two-term Woolhouse
    for a temporary annuity, ``ā_{x:n} ≈ ä_{x:n} − ½(1 − nE_x)``.
    """
    _check_table(table)
    n = _check_int("n", n, minimum=0)
    i = _check_rate(i)
    value = _annuity_due_sum(table, x, n, i)
    if continuous:
        return value - 0.5 * (1.0 - _pure_endowment(table, x, n, i))
    return value


def temporary_annuity_immediate(
    table: LifeTable, x: int, n: int, i: float, continuous: bool = False
) -> float:
    """``a_{x:n} = Σ_{k=1}^{n} v^k · k_p_x = ä_{x:n} − 1 + nE_x``: at most ``n`` payments, in arrears."""
    _check_table(table)
    n = _check_int("n", n, minimum=0)
    i = _check_rate(i)
    due = _annuity_due_sum(table, x, n, i)
    endowment = _pure_endowment(table, x, n, i)
    if continuous:
        return due - 0.5 * (1.0 - endowment)
    return due - 1.0 + endowment


def deferred_annuity_due(table: LifeTable, x: int, m: int, i: float, continuous: bool = False) -> float:
    """``m|ä_x = mE_x · ä_{x+m}``: payments in advance, starting in ``m`` years if alive.

    Equals ``ä_x − ä_{x:m}``. The typical pension: bought at ``x``, paid from
    ``x + m``. ``continuous`` uses Woolhouse, ``m|ā_x ≈ m|ä_x − ½·mE_x``.
    """
    _check_table(table)
    m = _check_int("m", m, minimum=0)
    i = _check_rate(i)
    k0 = table._index(x)
    if k0 + m >= len(table._q):
        return 0.0
    endowment = _pure_endowment(table, x, m, i)
    value = endowment * _annuity_due_sum(table, x + m, None, i)
    return value - 0.5 * endowment if continuous else value


def increasing_annuity_due(table: LifeTable, x: int, i: float) -> float:
    """``(Iä)_x = Σ_k (k+1) v^k · k_p_x``: payments of 1, 2, 3, … in advance while alive.

    Satisfies ``(Iä)_x = ä_x + v·p_x·(Iä)_{x+1}``.
    """
    _check_table(table)
    i = _check_rate(i)
    k0 = table._index(x)
    v = 1.0 / (1.0 + i)
    total = 0.0
    disc = 1.0
    for k in range(len(table._q) - k0):
        total += (k + 1) * disc * table._l[k0 + k]
        disc *= v
    return total / table._l[k0]


def mthly_annuity_due(table: LifeTable, x: int, i: float, m: int, n: Optional[int] = None) -> float:
    """``ä_x^{(m)}``: 1 a year paid in ``m`` instalments of ``1/m`` in advance — Woolhouse, two terms.

    ``ä_x^{(m)} ≈ ä_x − (m − 1)/(2m)``, or with a term ``n``,
    ``ä_{x:n}^{(m)} ≈ ä_{x:n} − (m − 1)/(2m)·(1 − nE_x)``. Monthly (``m = 12``)
    therefore takes off ``11/24``.

    This is an **approximation**. The dropped third Woolhouse term is
    ``−(m² − 1)/(12m²)·(δ + μ_x)`` — around 0.004 at 5% interest for a life in
    middle age, growing with ``μ_x`` — so quote the result to two decimals, not
    four. ``m = 1`` returns the annual annuity exactly. As ``m → ∞`` it tends to
    the ``continuous=True`` value of :func:`whole_life_annuity_due`.
    """
    _check_table(table)
    i = _check_rate(i)
    m = _check_int("m", m, minimum=1)
    adjust = (m - 1.0) / (2.0 * m)
    if n is None:
        return _annuity_due_sum(table, x, None, i) - adjust
    n = _check_int("n", n, minimum=0)
    return _annuity_due_sum(table, x, n, i) - adjust * (1.0 - _pure_endowment(table, x, n, i))


# ---------------------------------------------------------------------------
# Commutation functions
# ---------------------------------------------------------------------------


class Commutation:
    """Commutation functions ``D, N, C, M, S, R`` at rate ``i`` on a table.

    ``D_x = v^x·l_x``, ``N_x = Σ_{y≥x} D_y``, ``C_x = v^{x+1}·d_x``,
    ``M_x = Σ_{y≥x} C_y``, ``S_x = Σ_{y≥x} N_y``, ``R_x = Σ_{y≥x} M_y``. Then
    ``ä_x = N_x/D_x``, ``A_x = M_x/D_x``, ``nE_x = D_{x+n}/D_x``,
    ``A¹_{x:n} = (M_x − M_{x+n})/D_x``, ``(Iä)_x = S_x/D_x``, ``(IA)_x = R_x/D_x``.

    The present-value functions in this module are computed **directly** from
    the survivors — commutation functions are a pre-computer device for doing
    those sums once per table and are numerically no better. They are provided
    so that results can be checked against older tables and textbooks that
    print ``D_x`` and ``N_x`` columns. All six are defined for every table age
    and are 0 at ``omega``, so ``D(x + n)`` works right up to the table's end.
    """

    def __init__(self, table: LifeTable, i: float):
        self.table = _check_table(table)
        self.i = _check_rate(i)
        v = 1.0 / (1.0 + self.i)
        n = len(table._q)
        # v^x with x the actual age, as in every printed table.
        d_col = [v ** (table._start + k) * table._l[k] for k in range(n + 1)]
        c_col = [v ** (table._start + k + 1) * (table._l[k] - table._l[k + 1]) for k in range(n)]
        c_col.append(0.0)
        self._D = d_col
        self._C = c_col
        self._N = _suffix_sums(d_col)
        self._M = _suffix_sums(c_col)
        self._S = _suffix_sums(self._N)
        self._R = _suffix_sums(self._M)

    def _k(self, x: int) -> int:
        return self.table._index(x, allow_omega=True)

    def D(self, x: int) -> float:
        return self._D[self._k(x)]

    def N(self, x: int) -> float:
        return self._N[self._k(x)]

    def C(self, x: int) -> float:
        return self._C[self._k(x)]

    def M(self, x: int) -> float:
        return self._M[self._k(x)]

    def S(self, x: int) -> float:
        return self._S[self._k(x)]

    def R(self, x: int) -> float:
        return self._R[self._k(x)]

    def __repr__(self) -> str:
        return f"Commutation({self.table!r}, i={self.i:g})"


def _suffix_sums(values: Sequence[float]) -> List[float]:
    out = [0.0] * len(values)
    running = 0.0
    for k in range(len(values) - 1, -1, -1):
        running += values[k]
        out[k] = running
    return out


# ---------------------------------------------------------------------------
# Premiums
# ---------------------------------------------------------------------------


def net_premium(benefit_apv: float, annuity_apv: float) -> float:
    """Level premium by the equivalence principle: ``P = APV(benefits) / APV(1 per premium date)``.

    No expenses, no margin — the premium at which the insurer expects to break
    even. ``annuity_apv`` is the value of 1 paid on each premium date while the
    policy is in force, e.g. ``ä_x`` for whole-life premiums or ``ä_{x:n}`` for
    ``n`` years of them.
    """
    benefit_apv = _check_finite("benefit_apv", benefit_apv)
    annuity_apv = _check_finite("annuity_apv", annuity_apv)
    if not (annuity_apv > 0.0):
        raise ValueError(
            f"annuity_apv must be > 0 (the value of the premium stream), got {annuity_apv!r}"
        )
    return benefit_apv / annuity_apv


def whole_life_premium(table: LifeTable, x: int, i: float) -> float:
    """``P_x = A_x / ä_x``: annual net premium for unit whole-life cover, payable for life."""
    return net_premium(whole_life_insurance(table, x, i), whole_life_annuity_due(table, x, i))


def term_premium(table: LifeTable, x: int, n: int, i: float) -> float:
    """``P¹_{x:n} = A¹_{x:n} / ä_{x:n}``: annual net premium for ``n``-year term, payable ``n`` years."""
    n = _check_int("n", n, minimum=1)
    return net_premium(term_insurance(table, x, n, i), temporary_annuity_due(table, x, n, i))


def endowment_premium(table: LifeTable, x: int, n: int, i: float) -> float:
    """``P_{x:n} = A_{x:n} / ä_{x:n}``: annual net premium for an ``n``-year endowment, payable ``n`` years."""
    n = _check_int("n", n, minimum=1)
    return net_premium(endowment_insurance(table, x, n, i), temporary_annuity_due(table, x, n, i))


def gross_premium(
    table: LifeTable,
    x: int,
    i: float,
    benefit: float = 1.0,
    initial_expense: float = 0.0,
    renewal_expense: float = 0.0,
    premium_loading: float = 0.0,
    initial_loading: float = 0.0,
    settlement_expense: float = 0.0,
) -> float:
    """Gross annual premium ``G`` for whole-life cover, premiums for life, by the equivalence principle.

    Solves ``G·ä_x = (benefit + settlement_expense)·A_x + initial_expense +
    renewal_expense·(ä_x − 1) + premium_loading·G·ä_x + initial_loading·G``,
    i.e. expected income equals expected outgo, with

    * ``benefit`` — sum assured, paid at the end of the year of death;
    * ``initial_expense`` — a fixed amount at issue (currency, not a fraction);
    * ``renewal_expense`` — a fixed amount at the start of every policy year
      *after the first* (the first year's costs live in ``initial_expense``);
    * ``premium_loading`` — a fraction of *every* premium, ``0.05`` for 5%;
    * ``initial_loading`` — an extra fraction of the first premium only, the
      usual way to model commission;
    * ``settlement_expense`` — a fixed amount paid with the death benefit.

    Each expense raises ``G``; with all of them zero it is ``benefit · P_x``.
    Expenses are assumed level in money terms (no inflation) and to stop when
    the policy does. That is the textbook model, not a pricing basis.
    """
    _check_table(table)
    i = _check_rate(i)
    benefit = _check_finite("benefit", benefit)
    if not (benefit > 0.0):
        raise ValueError(f"benefit must be > 0, got {benefit!r}")
    for name, value in (
        ("initial_expense", initial_expense),
        ("renewal_expense", renewal_expense),
        ("settlement_expense", settlement_expense),
        ("initial_loading", initial_loading),
    ):
        if _check_finite(name, value) < 0.0:
            raise ValueError(f"{name} must be >= 0, got {value!r}")
    premium_loading = _check_finite("premium_loading", premium_loading)
    if not (0.0 <= premium_loading < 1.0):
        raise ValueError(
            f"premium_loading is a fraction of each premium and must be in [0, 1), got {premium_loading!r}"
        )

    A = whole_life_insurance(table, x, i)
    a = whole_life_annuity_due(table, x, i)
    outgo = (benefit + settlement_expense) * A + initial_expense + renewal_expense * (a - 1.0)
    income_per_unit = (1.0 - premium_loading) * a - initial_loading
    if not (income_per_unit > 0.0):
        raise ValueError(
            f"the loadings swallow the whole premium: (1 - premium_loading)·ä_x - "
            f"initial_loading = {income_per_unit:.4f} <= 0"
        )
    return outgo / income_per_unit


# ---------------------------------------------------------------------------
# Reserves
# ---------------------------------------------------------------------------


_KINDS = ("whole_life", "term", "endowment")


def _check_contract(table: LifeTable, x: int, kind: str, n: Optional[int]) -> Tuple[int, str, Optional[int]]:
    _check_table(table)
    x = table._start + table._index(x)
    if kind not in _KINDS:
        raise ValueError(f"kind must be one of {list(_KINDS)}, got {kind!r}")
    if kind == "whole_life":
        if n is not None:
            raise ValueError("whole_life has no term — leave n as None (use kind='term' or 'endowment' for an n-year contract)")
        return x, kind, None
    if n is None:
        raise ValueError(f"kind={kind!r} needs a term: pass n (the number of years)")
    n = _check_int("n", n, minimum=1)
    if x + n > table._omega:
        raise ValueError(
            f"an {n}-year contract on ({x}) runs past omega = {table._omega}; "
            f"this table supports n <= {table._omega - x}"
        )
    return x, kind, n


def _premium_for(table: LifeTable, x: int, i: float, kind: str, n: Optional[int]) -> float:
    if kind == "whole_life":
        return whole_life_premium(table, x, i)
    if kind == "term":
        return term_premium(table, x, n, i)
    return endowment_premium(table, x, n, i)


def _reserve_at(table: LifeTable, x: int, t: int, i: float, kind: str, n: Optional[int], P: float) -> float:
    if kind == "whole_life":
        return whole_life_insurance(table, x + t, i) - P * whole_life_annuity_due(table, x + t, i)
    if t == n:
        # The instant before maturity: the endowment is about to pay 1, term cover has run out.
        return 1.0 if kind == "endowment" else 0.0
    remaining = n - t
    annuity = temporary_annuity_due(table, x + t, remaining, i)
    if kind == "term":
        return term_insurance(table, x + t, remaining, i) - P * annuity
    return endowment_insurance(table, x + t, remaining, i) - P * annuity


def net_premium_reserve(
    table: LifeTable, x: int, t: int, i: float, kind: str = "whole_life", n: Optional[int] = None
) -> float:
    """``tV``: prospective net premium reserve at duration ``t`` per unit sum assured.

    Future benefits less future net premiums, valued at age ``x + t``, for a
    policy issued at ``x`` with the net premium of :func:`whole_life_premium`,
    :func:`term_premium` or :func:`endowment_premium`. Level annual premiums
    for the whole contract, benefit at the end of the year of death, and for
    ``endowment`` the survival benefit at ``t = n``.

    ``0V = 0`` by construction (that is what the equivalence principle means),
    an endowment's ``nV = 1`` and a term policy's ``nV = 0``. Between, the
    values satisfy Fackler's recursion ``(tV + P)(1 + i) = q_{x+t} + p_{x+t}·(t+1)V``,
    which is how a reserve is actually rolled forward on a valuation system.

    The reserve is on the *valuation basis* ``(table, i)``, which here is also
    the premium basis. Multiply by the sum assured. ``t`` runs from 0 to ``n``
    for term and endowment contracts and to ``omega − x − 1`` for whole life.
    """
    x, kind, n = _check_contract(table, x, kind, n)
    t = _check_int("t", t, minimum=0)
    i = _check_rate(i)
    if kind == "whole_life":
        if x + t >= table._omega:
            raise ValueError(
                f"t = {t} is past the table: ({x}) cannot be alive at age {x + t} >= omega = {table._omega}; "
                f"t must be in 0..{table._omega - x - 1}"
            )
    elif t > n:
        raise ValueError(f"t = {t} is after the {n}-year contract has expired; t must be in 0..{n}")
    P = _premium_for(table, x, i, kind, n)
    return _reserve_at(table, x, t, i, kind, n, P)


def reserve_profile(
    table: LifeTable, x: int, i: float, kind: str = "whole_life", n: Optional[int] = None
) -> List[Tuple[int, float]]:
    """``[(t, tV), …]`` from issue to expiry — the reserve build-up for :mod:`riskpy.viz`.

    ``t`` runs 0 to ``n`` for term and endowment policies, 0 to ``omega − x − 1``
    for whole life. Same basis and conventions as :func:`net_premium_reserve`;
    the premium is computed once and reused.
    """
    x, kind, n = _check_contract(table, x, kind, n)
    i = _check_rate(i)
    P = _premium_for(table, x, i, kind, n)
    last = table._omega - x - 1 if kind == "whole_life" else n
    return [(t, _reserve_at(table, x, t, i, kind, n, P)) for t in range(last + 1)]


# ---------------------------------------------------------------------------
# Joint life
# ---------------------------------------------------------------------------


def _joint_setup(table: LifeTable, x: int, y: int, i: float, table_y: Optional[LifeTable]):
    _check_table(table)
    ty = table if table_y is None else _check_table(table_y)
    kx = table._index(x)
    ky = ty._index(y)
    i = _check_rate(i)
    horizon = min(len(table._q) - kx, len(ty._q) - ky)
    return ty, kx, ky, i, horizon


def joint_life_annuity_due(
    table: LifeTable, x: int, y: int, i: float, table_y: Optional[LifeTable] = None
) -> float:
    """``ä_xy = Σ_k v^k · k_p_x · k_p_y``: 1 a year in advance while **both** are alive.

    Lives are assumed independent, so the joint survival probability is the
    product. Pass ``table_y`` to give the second life its own mortality (a
    female table for a spouse, say); otherwise both use ``table``. Pays until
    the *first* death, so it is the smaller of the two single-life annuities.
    """
    ty, kx, ky, i, horizon = _joint_setup(table, x, y, i, table_y)
    v = 1.0 / (1.0 + i)
    total, disc = 0.0, 1.0
    for k in range(horizon):
        total += disc * table._l[kx + k] * ty._l[ky + k]
        disc *= v
    return total / (table._l[kx] * ty._l[ky])


def last_survivor_annuity_due(
    table: LifeTable, x: int, y: int, i: float, table_y: Optional[LifeTable] = None
) -> float:
    """``ä_{x̄ȳ} = ä_x + ä_y − ä_xy``: 1 a year while **at least one** is alive.

    Follows from ``P(either alive) = P(x) + P(y) − P(both)``, so it holds with
    or without independence — only ``ä_xy`` itself needs that assumption.
    """
    ty = table if table_y is None else table_y
    return (
        whole_life_annuity_due(table, x, i)
        + whole_life_annuity_due(ty, y, i)
        - joint_life_annuity_due(table, x, y, i, table_y)
    )


def joint_life_insurance(
    table: LifeTable, x: int, y: int, i: float, table_y: Optional[LifeTable] = None
) -> float:
    """``A_xy = Σ_k v^{k+1} (k_p_xy − (k+1)_p_xy)``: 1 at the end of the year of the **first** death.

    Independent lives. Satisfies ``A_xy = 1 − d·ä_xy`` exactly, the joint-life
    version of the single-life identity.
    """
    ty, kx, ky, i, horizon = _joint_setup(table, x, y, i, table_y)
    v = 1.0 / (1.0 + i)
    scale = table._l[kx] * ty._l[ky]
    total, disc = 0.0, v
    alive = 1.0
    for k in range(horizon):
        nxt = table._l[kx + k + 1] * ty._l[ky + k + 1] / scale
        total += disc * (alive - nxt)
        alive = nxt
        disc *= v
    return total


def last_survivor_insurance(
    table: LifeTable, x: int, y: int, i: float, table_y: Optional[LifeTable] = None
) -> float:
    """``A_{x̄ȳ} = A_x + A_y − A_xy``: 1 at the end of the year of the **second** death."""
    ty = table if table_y is None else table_y
    return (
        whole_life_insurance(table, x, i)
        + whole_life_insurance(ty, y, i)
        - joint_life_insurance(table, x, y, i, table_y)
    )


# ---------------------------------------------------------------------------
# Verification hook
# ---------------------------------------------------------------------------


def _verification_checks():
    """Return a list of ``(name, value, reference, tolerance)`` tuples.

    Tolerances are absolute. The published references are AMLCR 2nd ed.,
    Appendix D (Table D.1 for ``l_x``, Table D.3 for ``ä_x`` and ``A_x`` at 5%),
    to the book's printed precision; the rest are identities that hold exactly.
    """
    i = 0.05
    d = discount_rate(i)
    sult = LifeTable.sult()
    A40 = whole_life_insurance(sult, 40, i)
    a40 = whole_life_annuity_due(sult, 40, i)

    makeham = LifeTable.makeham(0.00022, 2.7e-6, 1.124, start_age=20, max_age=100)
    integrated = LifeTable.from_force(
        lambda s: 0.00022 + 2.7e-6 * 1.124 ** s, start_age=20, max_age=100
    )
    comm = Commutation(sult, i)

    return [
        ("SULT l_100 vs AMLCR Table D.1", sult.lx(100), 6248.17, 0.01),
        ("SULT ä_40 at 5% vs AMLCR Table D.3", a40, 18.4578, 5e-5),
        ("SULT A_40 at 5% vs AMLCR Table D.3", A40, 0.12106, 5e-6),
        ("whole life A_x = 1 - d * ä_x", A40, 1.0 - d * a40, 1e-12),
        (
            "endowment A_{x:n} = A¹_{x:n} + nE_x",
            endowment_insurance(sult, 40, 20, i),
            term_insurance(sult, 40, 20, i) + pure_endowment(sult, 40, 20, i),
            1e-12,
        ),
        (
            "Makeham closed form 30_p_40 vs Simpson-integrated force",
            makeham.tpx(30, 40),
            integrated.tpx(30, 40),
            1e-9,
        ),
        ("commutation M_60 / D_60 = A_60", comm.M(60) / comm.D(60), whole_life_insurance(sult, 60, i), 1e-12),
        (
            "joint life ä_xy + ä_x̄ȳ = ä_x + ä_y",
            joint_life_annuity_due(sult, 60, 55, i) + last_survivor_annuity_due(sult, 60, 55, i),
            whole_life_annuity_due(sult, 60, i) + whole_life_annuity_due(sult, 55, i),
            1e-12,
        ),
    ]
