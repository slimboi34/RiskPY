"""Interest rates — compounding, yield curves, bonds and short-rate models.

Two layers, split by what they cost to install:

* **Scalar maths** — compounding conventions, :class:`YieldCurve`,
  :class:`Bond`, the Nelson–Siegel and Svensson families and the closed-form
  Vasicek and CIR bond prices — is pure Python and needs nothing beyond the
  standard library, so it works in the same lean install as the C++ core.
* **Path simulation** (:func:`vasicek_paths`, :func:`cir_paths`) needs NumPy
  (``pip install open-riskpy[sim]``), imported lazily on first use.

Conventions throughout: times are in **years**, rates are annualised
**decimals** (``0.05``, not ``5``) and ``compounding`` is ``"continuous"``,
``"annual"`` or a positive integer ``m`` of periods per year (``"semiannual"``,
``"quarterly"`` and ``"monthly"`` are accepted as aliases for 2, 4 and 12). A
:class:`YieldCurve` stores zero rates in one convention and converts for you;
a :class:`Bond` quotes its yield compounded at its own coupon frequency,
because that is how bond yields are quoted.

Nothing here plots. :meth:`YieldCurve.points` returns the data a curve chart
needs, and the path functions return arrays shaped for :func:`riskpy.viz.fan`.
"""

from __future__ import annotations

import bisect
import math
import numbers
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

from .mc import _numpy

__all__ = [
    "discount_factor",
    "zero_rate",
    "forward_rate",
    "convert_rate",
    "YieldCurve",
    "nelson_siegel_zero",
    "svensson_zero",
    "Bond",
    "bond_price",
    "duration_price_change",
    "vasicek_zero_bond",
    "cir_zero_bond",
    "vasicek_curve",
    "cir_curve",
    "vasicek_paths",
    "cir_paths",
]

# Monthly knots out to thirty years: fine enough that linear interpolation of a
# smooth parametric curve (Nelson–Siegel, Vasicek) is exact to well under a
# hundredth of a basis point.
_DEFAULT_GRID: Tuple[float, ...] = tuple(month / 12.0 for month in range(1, 361))

_NAMED_COMPOUNDING = {
    "continuous": None,
    "annual": 1,
    "semiannual": 2,
    "semi-annual": 2,
    "quarterly": 4,
    "monthly": 12,
}


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


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


def _check_frequency(value: object, name: str = "frequency") -> int:
    """A positive integer number of periods per year. Rejects bools and 2.5."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    if isinstance(value, numbers.Integral):
        m = int(value)
    elif isinstance(value, numbers.Real) and float(value).is_integer():
        m = int(value)
    else:
        raise ValueError(f"{name} must be a positive integer (periods per year), got {value!r}")
    if m < 1:
        raise ValueError(f"{name} must be >= 1, got {value!r}")
    return m


def _periods_per_year(compounding: object) -> Optional[int]:
    """``None`` for continuous compounding, else the integer periods per year."""
    if isinstance(compounding, str):
        key = compounding.strip().lower()
        if key in _NAMED_COMPOUNDING:
            return _NAMED_COMPOUNDING[key]
        raise ValueError(
            f"compounding must be 'continuous', 'annual' or a positive integer of "
            f"periods per year (or 'semiannual', 'quarterly', 'monthly'), got {compounding!r}"
        )
    return _check_frequency(compounding, "compounding")


def _normalise_compounding(compounding: object):
    """Canonical form: the string ``"continuous"`` or an ``int``."""
    m = _periods_per_year(compounding)
    return "continuous" if m is None else m


def _whole_periods(maturity: float, frequency: int, what: str = "maturity") -> int:
    """Number of coupon periods in ``maturity`` years, which must be whole."""
    periods = maturity * frequency
    if abs(periods - round(periods)) > 1e-8:
        raise ValueError(
            f"{what} {maturity!r} is not a whole number of periods at {frequency} per "
            f"year — use a multiple of {1.0 / frequency:g}"
        )
    return int(round(periods))


def _bisect_root(
    fn: Callable[[float], float],
    low: float,
    high: float,
    tolerance: float = 1e-14,
    max_iterations: int = 200,
) -> float:
    """Root of ``fn`` on ``[low, high]``, which must bracket a sign change.

    Bisection rather than Newton, for the same reason as
    :func:`riskpy.quant.implied_vol`: it cannot diverge, and two hundred
    iterations of a bond price is still instant.
    """
    f_low, f_high = fn(low), fn(high)
    if f_low == 0.0:
        return low
    if f_high == 0.0:
        return high
    if (f_low > 0.0) == (f_high > 0.0):
        raise ValueError(f"no sign change on [{low!r}, {high!r}]")
    for _ in range(max_iterations):
        mid = 0.5 * (low + high)
        f_mid = fn(mid)
        if f_mid == 0.0 or (high - low) < tolerance:
            return mid
        if (f_mid > 0.0) == (f_low > 0.0):
            low, f_low = mid, f_mid
        else:
            high = mid
    return 0.5 * (low + high)


# ---------------------------------------------------------------------------
# Compounding conventions
# ---------------------------------------------------------------------------


def discount_factor(rate: float, t: float, compounding: object = "continuous") -> float:
    """Present value of 1 paid at time ``t`` (years) under a flat ``rate``.

    ``compounding`` is ``"continuous"`` (``e^{-rt}``), ``"annual"``
    (``(1+r)^{-t}``) or an integer ``m`` (``(1+r/m)^{-mt}``). Negative rates
    are fine; a rate so negative that ``1 + r/m <= 0`` is not, and raises.

    The one thing people get wrong: the same number means different things
    under different conventions. 5% semi-annual is 4.94% continuous — use
    :func:`convert_rate` rather than passing the label you happen to have.
    """
    rate = _check_finite("rate", rate)
    t = _check_non_negative("t", t)
    m = _periods_per_year(compounding)
    if m is None:
        return math.exp(-rate * t)
    base = 1.0 + rate / m
    if base <= 0.0:
        raise ValueError(
            f"rate {rate!r} compounded {m} times a year gives 1 + rate/m <= 0; "
            f"no discount factor exists — the rate is below -{m * 100:.0f}%"
        )
    return base ** (-m * t)


def zero_rate(df: float, t: float, compounding: object = "continuous") -> float:
    """The flat rate that turns discount factor ``df`` at ``t`` back into 1.

    The inverse of :func:`discount_factor`. ``t`` must be positive — the zero
    rate at ``t = 0`` is a limit, not a number; ask a :class:`YieldCurve` for
    it instead.
    """
    df = _check_positive("df", df)
    t = _check_positive("t", t)
    m = _periods_per_year(compounding)
    if m is None:
        return -math.log(df) / t
    return m * (df ** (-1.0 / (m * t)) - 1.0)


def forward_rate(
    df_1: float, t_1: float, df_2: float, t_2: float, compounding: object = "continuous"
) -> float:
    """The rate for lending from ``t_1`` to ``t_2`` implied by two discount factors.

    Quoted in ``compounding`` over the period ``t_2 - t_1``, so a
    continuously compounded forward ``f`` satisfies
    ``df_2 = df_1 · exp(-f · (t_2 - t_1))`` exactly. Requires
    ``0 <= t_1 < t_2``.

    The one thing people get wrong: a forward is a *ratio* of discount
    factors, not a difference of zero rates. Averaging zero rates gives a
    number that is close and wrong.
    """
    t_1 = _check_non_negative("t_1", t_1)
    t_2 = _check_finite("t_2", t_2)
    if not t_2 > t_1:
        raise ValueError(f"need t_1 < t_2, got t_1={t_1!r} t_2={t_2!r}")
    df_1 = _check_positive("df_1", df_1)
    df_2 = _check_positive("df_2", df_2)
    return zero_rate(df_2 / df_1, t_2 - t_1, compounding)


def convert_rate(rate: float, from_compounding: object, to_compounding: object) -> float:
    """Re-express an annualised rate in another compounding convention.

    Goes through the one-year discount factor, so the two rates agree on the
    value of money exactly — ``convert_rate(0.05, 2, "continuous")`` is
    ``2·ln(1.025) = 0.049385``.
    """
    return zero_rate(discount_factor(rate, 1.0, from_compounding), 1.0, to_compounding)


# ---------------------------------------------------------------------------
# Parametric zero-curve families
# ---------------------------------------------------------------------------


def nelson_siegel_zero(t: float, beta0: float, beta1: float, beta2: float, tau: float) -> float:
    """Nelson–Siegel (1987) zero rate at maturity ``t``.

    ``z(t) = β0 + β1·L(t) + β2·(L(t) − e^{−t/τ})`` with
    ``L(t) = (1 − e^{−t/τ}) / (t/τ)``. The three betas are the level, slope and
    curvature factors: ``z(0) = β0 + β1`` (the short rate), ``z(∞) = β0``
    (the long rate) and ``β2`` puts a hump — or a dip — at maturities around
    ``τ``. ``t = 0`` returns the limit ``β0 + β1`` rather than dividing by
    zero.

    The one thing people get wrong: ``β1`` is the *negative* of the slope.
    A normal upward-sloping curve has ``β1 < 0``.
    """
    t = _check_non_negative("t", t)
    tau = _check_positive("tau", tau)
    beta0, beta1, beta2 = (_check_finite(n, v) for n, v in (("beta0", beta0), ("beta1", beta1), ("beta2", beta2)))
    if t == 0.0:
        return beta0 + beta1
    x = t / tau
    decay = math.exp(-x)
    loading = -math.expm1(-x) / x
    return beta0 + beta1 * loading + beta2 * (loading - decay)


def svensson_zero(
    t: float, beta0: float, beta1: float, beta2: float, beta3: float, tau1: float, tau2: float
) -> float:
    """Svensson (1994) extension of Nelson–Siegel: a second hump at scale ``tau2``.

    ``z(t) = NS(t; β0, β1, β2, τ1) + β3·(L2(t) − e^{−t/τ2})``. Same limits as
    Nelson–Siegel (``β0 + β1`` at zero, ``β0`` at infinity); the extra term
    is what central banks use to fit both a short-end kink and a long-end
    hump at once. Nearly collinear when ``tau1 ≈ tau2`` — fitting it is
    ill-conditioned, evaluating it is not.
    """
    base = nelson_siegel_zero(t, beta0, beta1, beta2, tau1)
    tau2 = _check_positive("tau2", tau2)
    beta3 = _check_finite("beta3", beta3)
    if t == 0.0:
        return base
    x = float(t) / tau2
    decay = math.exp(-x)
    loading = -math.expm1(-x) / x
    return base + beta3 * (loading - decay)


# ---------------------------------------------------------------------------
# Yield curve
# ---------------------------------------------------------------------------


class YieldCurve:
    """A zero-coupon curve: zero rates at knot maturities plus an interpolation rule.

    ``times`` are knot maturities in years (strictly increasing, all positive)
    and ``zero_rates`` the annualised zero rates at those knots, quoted in
    ``compounding``. Everything else — discount factors, forwards, par rates —
    is derived, so the curve is internally consistent by construction.

    ``interpolation``:

    * ``"linear"`` — linear in the zero rate between knots. Simple, the
      market's default, and its instantaneous forward curve has small jumps
      at the knots.
    * ``"log_linear"`` — linear in the *logarithm of the discount factor*,
      i.e. a piecewise-constant continuously compounded forward rate. The
      forward curve is a staircase, but the forward is positive wherever the
      discount factors decrease, which linear-in-zero does not guarantee.

    **Extrapolation is flat in the zero rate**: below the first knot the
    curve returns the first knot's rate, beyond the last knot the last one.
    This is stated because it is the source of most surprises with curves —
    a 30-year curve asked for a 40-year discount factor answers confidently.

    The one thing people get wrong: mixing conventions. A curve built from
    semi-annual bond yields is *not* the same curve as one built from the
    same numbers labelled continuous. :func:`convert_rate` makes the
    conversion explicit.
    """

    def __init__(
        self,
        times: Sequence[float],
        zero_rates: Sequence[float],
        interpolation: str = "linear",
        compounding: object = "continuous",
    ):
        times = [_check_positive(f"times[{i}]", t) for i, t in enumerate(times)]
        zeros = [_check_finite(f"zero_rates[{i}]", z) for i, z in enumerate(zero_rates)]
        if not times:
            raise ValueError("a YieldCurve needs at least one knot")
        if len(times) != len(zeros):
            raise ValueError(
                f"times and zero_rates must have the same length, got {len(times)} and {len(zeros)}"
            )
        for earlier, later in zip(times, times[1:]):
            if not later > earlier:
                raise ValueError(f"times must be strictly increasing, got {earlier!r} then {later!r}")
        if interpolation not in ("linear", "log_linear"):
            raise ValueError(
                f"interpolation must be 'linear' (in zero rates) or 'log_linear' (in "
                f"discount factors), got {interpolation!r}"
            )
        self.times: Tuple[float, ...] = tuple(times)
        self.zero_rates: Tuple[float, ...] = tuple(zeros)
        self.interpolation = interpolation
        self.compounding = _normalise_compounding(compounding)
        # Raises here, not later, if a rate is below -m·100%.
        self._log_dfs = tuple(
            math.log(discount_factor(z, t, self.compounding)) for t, z in zip(times, zeros)
        )
        # Set by the parametric constructors. When a curve has a formula there
        # is no reason to answer from an interpolation of samples of it: doing
        # so gets the t -> 0 and t -> infinity limits wrong, which are exactly
        # the values people check a Nelson-Siegel fit against.
        self._analytic: Optional[Callable[[float], float]] = None

    # -- constructors -------------------------------------------------------

    @classmethod
    def flat(cls, rate: float, compounding: object = "continuous") -> "YieldCurve":
        """A flat curve at ``rate``; ``df(t) = exp(-rate·t)`` when continuous."""
        rate = _check_finite("rate", rate)
        return cls((1.0, 30.0), (rate, rate), "linear", compounding)

    @classmethod
    def nelson_siegel(
        cls,
        beta0: float,
        beta1: float,
        beta2: float,
        tau: float,
        times: Optional[Sequence[float]] = None,
    ) -> "YieldCurve":
        """Sample :func:`nelson_siegel_zero` on ``times`` (monthly to 30y by default).

        The curve keeps the formula and evaluates it at any maturity, so the
        limits hold exactly: ``zero(0) = beta0 + beta1`` and
        ``zero(inf) = beta0``. The knots are there so the curve prints and
        plots like any other. Continuous compounding.
        """
        grid = _DEFAULT_GRID if times is None else tuple(times)
        zeros = [nelson_siegel_zero(t, beta0, beta1, beta2, tau) for t in grid]
        curve = cls(grid, zeros, "linear", "continuous")
        curve._analytic = lambda t: nelson_siegel_zero(t, beta0, beta1, beta2, tau)
        return curve

    @classmethod
    def svensson(
        cls,
        beta0: float,
        beta1: float,
        beta2: float,
        beta3: float,
        tau1: float,
        tau2: float,
        times: Optional[Sequence[float]] = None,
    ) -> "YieldCurve":
        """Sample :func:`svensson_zero` on ``times`` (monthly to 30y by default)."""
        grid = _DEFAULT_GRID if times is None else tuple(times)
        zeros = [svensson_zero(t, beta0, beta1, beta2, beta3, tau1, tau2) for t in grid]
        curve = cls(grid, zeros, "linear", "continuous")
        curve._analytic = lambda t: svensson_zero(t, beta0, beta1, beta2, beta3, tau1, tau2)
        return curve

    @classmethod
    def from_discount_factors(
        cls,
        times: Sequence[float],
        dfs: Sequence[float],
        interpolation: str = "linear",
        compounding: object = "continuous",
    ) -> "YieldCurve":
        """Build from discount factors instead of zero rates.

        Discount factors must be positive; they may exceed 1 (negative
        rates) but a factor of zero or below has no zero rate.
        """
        times = list(times)
        dfs = list(dfs)
        if len(times) != len(dfs):
            raise ValueError(f"times and dfs must have the same length, got {len(times)} and {len(dfs)}")
        zeros = [
            zero_rate(_check_positive(f"dfs[{i}]", df), _check_positive(f"times[{i}]", t), compounding)
            for i, (t, df) in enumerate(zip(times, dfs))
        ]
        return cls(times, zeros, interpolation, compounding)

    @classmethod
    def bootstrap(
        cls,
        par_rates: Sequence[float],
        maturities: Sequence[float],
        frequency: int = 1,
        interpolation: str = "linear",
    ) -> "YieldCurve":
        """Zero curve from par swap (or par bond) rates, by sequential bootstrap.

        ``par_rates[k]`` is the annual coupon, paid ``frequency`` times a year,
        that prices a bond maturing at ``maturities[k]`` exactly at par.
        Maturities must be increasing and whole numbers of periods. They need
        not be *consecutive*: for a gap (say 5y then 7y) the 6y discount factor
        is whatever the curve's own interpolation says it is, and the 7y zero
        rate is solved so that the 7y bond reprices at par. That is the
        standard bootstrap and it reproduces every input par rate to machine
        precision — :meth:`par_rate` is the round trip.

        The result is continuously compounded. The one thing people get
        wrong: feeding in *yields to maturity* of coupon bonds trading away
        from par. Those are not par rates, and the curve will be off by the
        coupon effect.
        """
        m = _check_frequency(frequency)
        rates = [_check_finite(f"par_rates[{i}]", r) for i, r in enumerate(par_rates)]
        mats = [_check_positive(f"maturities[{i}]", t) for i, t in enumerate(maturities)]
        if not mats:
            raise ValueError("bootstrap needs at least one par rate")
        if len(rates) != len(mats):
            raise ValueError(
                f"par_rates and maturities must have the same length, got {len(rates)} and {len(mats)}"
            )
        for earlier, later in zip(mats, mats[1:]):
            if not later > earlier:
                raise ValueError(f"maturities must be strictly increasing, got {earlier!r} then {later!r}")

        times: List[float] = []
        zeros: List[float] = []
        for maturity, par in zip(mats, rates):
            periods = _whole_periods(maturity, m)
            coupon = par / m
            dates = [j / m for j in range(1, periods)] + [maturity]

            def mispricing(z: float) -> float:
                trial = cls(times + [maturity], zeros + [z], interpolation, "continuous")
                return coupon * sum(trial.df(d) for d in dates) + trial.df(maturity) - 1.0

            try:
                z = _bisect_root(mispricing, -0.5, 2.0)
            except ValueError:
                raise ValueError(
                    f"cannot bootstrap the {maturity:g}y par rate {par!r}: no continuous zero "
                    f"rate between -50% and +200% reprices it at par. Check the earlier "
                    f"rates — an implausible short rate poisons every later maturity."
                ) from None
            times.append(maturity)
            zeros.append(z)
        return cls(times, zeros, interpolation, "continuous")

    # -- interpolation core -------------------------------------------------

    def _interp(self, t: float, values: Sequence[float]) -> float:
        times = self.times
        if t <= times[0]:
            return values[0]
        if t >= times[-1]:
            return values[-1]
        i = bisect.bisect_right(times, t) - 1
        w = (t - times[i]) / (times[i + 1] - times[i])
        return values[i] + w * (values[i + 1] - values[i])

    def _outside(self, t: float) -> bool:
        return t <= self.times[0] or t >= self.times[-1]

    # -- queries ------------------------------------------------------------

    def zero(self, t: float) -> float:
        """Zero rate at ``t`` in the curve's compounding.

        A curve built from a formula (Nelson–Siegel, Svensson, Vasicek, CIR)
        evaluates that formula at any ``t``, including outside its knots. A
        curve built from data interpolates between its knots and extrapolates
        flat beyond them.
        """
        t = _check_non_negative("t", t)
        if self._analytic is not None:
            return self._analytic(t)
        if self.interpolation == "linear" or self._outside(t):
            return self._interp(t, self.zero_rates)
        return zero_rate(math.exp(self._interp(t, self._log_dfs)), t, self.compounding)

    def df(self, t: float) -> float:
        """Discount factor for a payment at ``t`` years; ``df(0) = 1``."""
        t = _check_non_negative("t", t)
        if t == 0.0:
            return 1.0
        if self._analytic is not None:
            return discount_factor(self._analytic(t), t, self.compounding)
        if self.interpolation == "linear" or self._outside(t):
            return discount_factor(self._interp(t, self.zero_rates), t, self.compounding)
        return math.exp(self._interp(t, self._log_dfs))

    def forward(self, t1: float, t2: float) -> float:
        """Forward rate over ``[t1, t2]`` in the curve's compounding.

        Defined through discount factors, so ``df(t2) = df(t1)·discount(forward)``
        holds exactly — see :func:`forward_rate`.
        """
        return forward_rate(self.df(t1), t1, self.df(t2), t2, self.compounding)

    def instantaneous_forward(self, t: float) -> float:
        """The continuously compounded forward for an instant at ``t``: ``-d ln df / dt``.

        Computed by a central difference with step ``1e-5`` years (one-sided at
        ``t = 0``), so it is accurate to about ``1e-9`` where the curve is
        smooth. Under ``"linear"`` interpolation the forward jumps at each
        knot and the difference quotient returns the average of the two
        sides there; under ``"log_linear"`` it is exactly the flat forward of
        the segment. Always continuous, whatever the curve's compounding.
        """
        t = _check_non_negative("t", t)
        h = 1e-5
        lo, hi = (0.0, 2.0 * h) if t < h else (t - h, t + h)
        return (math.log(self.df(lo)) - math.log(self.df(hi))) / (hi - lo)

    def annuity(self, maturity: float, frequency: int = 1) -> float:
        """PV of 1 per year paid ``frequency`` times a year until ``maturity``.

        The swap "annuity factor" ``Σ df(t_j) / frequency`` over coupon dates
        ``t_j = j / frequency``. Also the DV01 of a par swap per unit notional
        per unit rate.
        """
        maturity = _check_positive("maturity", maturity)
        m = _check_frequency(frequency)
        periods = _whole_periods(maturity, m)
        return sum(self.df(j / m) for j in range(1, periods + 1)) / m

    def par_rate(self, maturity: float, frequency: int = 1) -> float:
        """The coupon (annual rate, paid ``frequency`` times a year) that prices at par.

        ``(1 − df(T)) / annuity(T)`` — the fixed rate of a par swap. This is
        the inverse of :meth:`bootstrap`.
        """
        maturity = _check_positive("maturity", maturity)
        return (1.0 - self.df(maturity)) / self.annuity(maturity, frequency)

    def shift(self, bp: float) -> "YieldCurve":
        """A new curve with every zero rate moved by ``bp`` basis points (parallel)."""
        bp = _check_finite("bp", bp)
        delta = bp / 10_000.0
        return YieldCurve(
            self.times, [z + delta for z in self.zero_rates], self.interpolation, self.compounding
        )

    def points(
        self, grid: Optional[Sequence[float]] = None
    ) -> Tuple[List[float], List[float], List[float]]:
        """``(times, zero_rates, instantaneous_forwards)`` for a curve chart.

        Default grid: 121 evenly spaced points from 0 to the last knot. Zero
        rates are in the curve's compounding, forwards are continuous — plot
        them on the same axes anyway, that is what everyone does.
        """
        if grid is None:
            end = self.times[-1]
            grid = [end * i / 120.0 for i in range(121)]
        times = [_check_non_negative(f"grid[{i}]", t) for i, t in enumerate(grid)]
        zeros = [self.zero(t) for t in times]
        forwards = [self.instantaneous_forward(t) for t in times]
        return times, zeros, forwards

    def __repr__(self) -> str:
        lo, hi = min(self.zero_rates), max(self.zero_rates)
        return (
            f"<YieldCurve {len(self.times)} knots {self.times[0]:g}y–{self.times[-1]:g}y, "
            f"zero {lo:.3%}–{hi:.3%}, {self.interpolation}, {self.compounding}>"
        )


# ---------------------------------------------------------------------------
# Bonds
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Bond:
    """A fixed-coupon bullet bond.

    ``face`` is the redemption amount, ``coupon`` the **annual** coupon rate
    as a decimal (``0.05`` for a 5% bond), ``maturity`` in years and
    ``frequency`` the coupons per year (2 for the US/UK convention). The
    yield in every method is compounded ``frequency`` times a year — the
    bond-equivalent yield — because that is how bond yields are quoted.

    Coupon dates are counted back from maturity. If ``maturity`` is not a
    whole number of periods the first period is a short stub that still pays
    a full coupon, and :meth:`price` is the **dirty** price: no accrued
    interest is subtracted. That is the textbook convention for "a 2.5-year
    bond paying annual coupons"; day-count conventions are out of scope.

    The one thing people get wrong: ``coupon`` is a rate, not an amount. A
    bond paying 2.50 twice a year on 100 face has ``coupon=0.05``.
    """

    face: float
    coupon: float
    maturity: float
    frequency: int = 2

    def __post_init__(self) -> None:
        object.__setattr__(self, "face", _check_positive("face", self.face))
        object.__setattr__(self, "coupon", _check_non_negative("coupon", self.coupon))
        object.__setattr__(self, "maturity", _check_positive("maturity", self.maturity))
        object.__setattr__(self, "frequency", _check_frequency(self.frequency))

    def cashflows(self) -> List[Tuple[float, float]]:
        """``[(t, amount), …]`` in time order; the last entry includes the face."""
        m = self.frequency
        if self.coupon == 0.0:
            return [(self.maturity, self.face)]
        periods = int(math.ceil(self.maturity * m - 1e-9))
        amount = self.face * self.coupon / m
        flows = [(self.maturity - (periods - k) / m, amount) for k in range(1, periods)]
        flows.append((self.maturity, amount + self.face))
        return flows

    def _period_base(self, ytm: float) -> float:
        ytm = _check_finite("ytm", ytm)
        base = 1.0 + ytm / self.frequency
        if base <= 0.0:
            raise ValueError(
                f"ytm {ytm!r} is below -{self.frequency * 100:.0f}%, where "
                f"(1 + ytm/{self.frequency}) is not positive and the price is undefined"
            )
        return base

    def _present_values(self, ytm: float) -> List[Tuple[float, float]]:
        base = self._period_base(ytm)
        m = self.frequency
        return [(t, amount * base ** (-m * t)) for t, amount in self.cashflows()]

    def price(self, ytm: float) -> float:
        """Dirty price at yield ``ytm`` (compounded ``frequency`` times a year)."""
        return sum(pv for _, pv in self._present_values(ytm))

    def yield_to_maturity(self, price: float) -> float:
        """The yield that reproduces ``price``, by bisection on ``[-99%, +100%]``.

        The upper bracket is widened automatically for a distressed price;
        a price above what ``-99%`` produces is rejected as not a price.
        Accurate to ``1e-14`` in yield.
        """
        price = _check_positive("price", price)
        low, high = -0.99, 1.0
        if self.price(low) < price:
            raise ValueError(
                f"price {price!r} exceeds {self.price(low):.4f}, the price at a -99% yield — "
                f"check the face and the units (the price is in the same units as face)"
            )
        for _ in range(6):
            if self.price(high) <= price:
                break
            high *= 2.0
        else:
            raise ValueError(
                f"price {price!r} is below what a {high:.0%} yield produces — check the inputs"
            )
        return _bisect_root(lambda y: self.price(y) - price, low, high)

    def macaulay_duration(self, ytm: float) -> float:
        """PV-weighted average time to cash flow, in years.

        Equals the maturity for a zero-coupon bond, and is always shorter for
        a coupon bond. This is the *time* measure; :meth:`modified_duration`
        is the *price sensitivity*, and the two differ by ``1 + y/m``.
        """
        pvs = self._present_values(ytm)
        price = sum(pv for _, pv in pvs)
        return sum(t * pv for t, pv in pvs) / price


    def modified_duration(self, ytm: float) -> float:
        """``-(1/P)·dP/dy`` — the percentage price change per unit yield change.

        ``Macaulay / (1 + y/m)``. A modified duration of 7 means a 1bp rise in
        yield costs about 0.07% of price.
        """
        return self.macaulay_duration(ytm) / self._period_base(ytm)

    def convexity(self, ytm: float) -> float:
        """``(1/P)·d²P/dy²`` in years², the second-order term in :func:`duration_price_change`.

        ``Σ t(t + 1/m)·PV_t / (P·(1 + y/m)²)``. Positive for any option-free
        bond, which is why duration alone *overstates* the loss from a rate
        rise and understates the gain from a fall.
        """
        pvs = self._present_values(ytm)
        price = sum(pv for _, pv in pvs)
        base = self._period_base(ytm)
        m = self.frequency
        return sum(t * (t + 1.0 / m) * pv for t, pv in pvs) / (price * base * base)

    def dv01(self, ytm: float) -> float:
        """Price change for a one basis point move in yield, as a positive number.

        The analytic first-order sensitivity ``modified_duration · price ·
        0.0001`` — a bumped price differs from it only by the convexity term,
        of order ``1e-8 · price``. In the same units as ``face``.
        """
        return self.modified_duration(ytm) * self.price(ytm) * 1e-4

    def price_from_curve(self, curve: YieldCurve) -> float:
        """Discount each cash flow off ``curve`` — the arbitrage-free price."""
        return sum(amount * curve.df(t) for t, amount in self.cashflows())

    def z_spread(self, price: float, curve: YieldCurve) -> float:
        """The constant spread over every zero rate on ``curve`` that reproduces ``price``.

        Quoted in the curve's own compounding (a continuous curve gives a
        continuous spread), solved by bisection to ``1e-14``. Zero when the
        bond is priced off the curve itself; ``+0.0050`` when it trades 50bp
        cheap to it. Sign convention: positive spread means a *lower* price.
        """
        price = _check_positive("price", price)
        flows = self.cashflows()
        comp = curve.compounding

        def spread_price(s: float) -> float:
            return sum(amount * discount_factor(curve.zero(t) + s, t, comp) for t, amount in flows)

        low, high = -0.5, 1.0
        if spread_price(low) < price:
            raise ValueError(
                f"price {price!r} exceeds {spread_price(low):.4f}, the price at a -50% spread — "
                f"check the face and the units"
            )
        for _ in range(6):
            if spread_price(high) <= price:
                break
            high *= 2.0
        else:
            raise ValueError(f"price {price!r} is below what a {high:.0%} spread produces")
        return _bisect_root(lambda s: spread_price(s) - price, low, high)


def bond_price(face: float, coupon: float, maturity: float, ytm: float, frequency: int = 2) -> float:
    """Functional form of :meth:`Bond.price` — same conventions."""
    return Bond(face, coupon, maturity, frequency).price(ytm)


def duration_price_change(price: float, modified_duration: float, convexity: float, dy: float) -> float:
    """Second-order estimate of the price change for a yield move ``dy``.

    ``ΔP ≈ −D·P·dy + ½·C·P·dy²``, signed (negative for a rate rise). ``dy`` is
    a decimal, ``0.0025`` for 25bp. The error is third order in ``dy``: for a
    10-year bond and a 100bp move the estimate is within a few hundredths of
    a percent; for a 300bp move, re-price instead.
    """
    price = _check_finite("price", price)
    modified_duration = _check_finite("modified_duration", modified_duration)
    convexity = _check_finite("convexity", convexity)
    dy = _check_finite("dy", dy)
    return -modified_duration * price * dy + 0.5 * convexity * price * dy * dy


# ---------------------------------------------------------------------------
# Short-rate models — closed forms
# ---------------------------------------------------------------------------


def _check_short_rate(kappa: float, theta: float, sigma: float) -> Tuple[float, float, float]:
    kappa = _check_positive("kappa", kappa)
    theta = _check_finite("theta", theta)
    sigma = _check_non_negative("sigma", sigma)
    return kappa, theta, sigma


def _check_horizon(t: float, T: float) -> float:
    t = _check_non_negative("t", t)
    T = _check_finite("T", T)
    if T < t:
        raise ValueError(f"need t <= T, got t={t!r} T={T!r}")
    return T - t


def vasicek_zero_bond(r: float, t: float, T: float, kappa: float, theta: float, sigma: float) -> float:
    """Price at ``t`` of a zero-coupon bond paying 1 at ``T`` under Vasicek (1977).

    ``dr = κ(θ − r)dt + σ dW``. The price is affine in the current short rate
    ``r``: ``P = A(τ)·exp(−B(τ)·r)`` with ``τ = T − t``,
    ``B = (1 − e^{−κτ})/κ`` and
    ``ln A = (B − τ)(θ − σ²/(2κ²)) − σ²B²/(4κ)``. ``θ`` is the
    **risk-neutral** long-run mean — if you estimated it from history you
    need the market price of risk to get here, and the difference is the
    term premium.

    The one thing people get wrong: the ``σ²`` terms mean the yield curve is
    *not* ``θ`` at long maturities. It converges to ``θ − σ²/(2κ²)``.
    """
    r = _check_finite("r", r)
    kappa, theta, sigma = _check_short_rate(kappa, theta, sigma)
    tau = _check_horizon(t, T)
    if tau == 0.0:
        return 1.0
    B = -math.expm1(-kappa * tau) / kappa
    log_A = (B - tau) * (theta - sigma * sigma / (2.0 * kappa * kappa)) - sigma * sigma * B * B / (4.0 * kappa)
    return math.exp(log_A - B * r)


def cir_zero_bond(r: float, t: float, T: float, kappa: float, theta: float, sigma: float) -> float:
    """Price at ``t`` of a zero-coupon bond paying 1 at ``T`` under Cox–Ingersoll–Ross (1985).

    ``dr = κ(θ − r)dt + σ√r dW``. Affine again, ``P = A(τ)·exp(−B(τ)·r)``,
    with ``γ = √(κ² + 2σ²)``,
    ``B = 2(e^{γτ} − 1) / ((γ + κ)(e^{γτ} − 1) + 2γ)`` and
    ``A = [2γ e^{(κ+γ)τ/2} / ((γ + κ)(e^{γτ} − 1) + 2γ)]^{2κθ/σ²}``, written
    here in terms of ``e^{−γτ}`` so long maturities cannot overflow. At
    ``σ = 0`` the exponent ``2κθ/σ²`` blows up while its base tends to 1; the
    limit is taken analytically and equals the deterministic
    ``exp(−∫ r dt)``, the same as Vasicek's.

    Unlike Vasicek, rates stay non-negative when the Feller condition
    ``2κθ > σ²`` holds — and the *price* is valid either way, it is the
    simulation that misbehaves (see :func:`cir_paths`).
    """
    r = _check_non_negative("r", r)
    kappa, theta, sigma = _check_short_rate(kappa, theta, sigma)
    theta = _check_non_negative("theta", theta)
    tau = _check_horizon(t, T)
    if tau == 0.0:
        return 1.0
    gamma = math.sqrt(kappa * kappa + 2.0 * sigma * sigma)
    total = gamma + kappa
    # gamma - kappa, computed without cancellation: it is O(sigma²), and the
    # exponent 2κθ/σ² multiplies whatever rounding error it carries.
    gap = 2.0 * sigma * sigma / total
    decay = math.exp(-gamma * tau)
    denominator = total * (1.0 - decay) + 2.0 * gamma * decay
    B = 2.0 * (1.0 - decay) / denominator
    if sigma == 0.0:
        log_A = -theta * (tau - B)
    else:
        # ln A = (2κθ/σ²)·ln[2γ·e^{(κ−γ)τ/2} / denominator]. Written as a
        # difference of two O(1) logarithms the bracket cancels to O(σ²) and
        # the prefactor blows the rounding error up to O(1) for small σ.
        # With 2γ = total + gap and denominator = total + gap·decay, each
        # term is a log1p of something small and the precision survives.
        log_base = math.log1p(gap / total) - 0.5 * gap * tau - math.log1p(gap * decay / total)
        log_A = (2.0 * kappa * theta / (sigma * sigma)) * log_base
    return math.exp(log_A - B * r)


def vasicek_curve(
    r0: float, kappa: float, theta: float, sigma: float, times: Optional[Sequence[float]] = None
) -> YieldCurve:
    """The zero curve implied by Vasicek parameters, as a :class:`YieldCurve`.

    Continuously compounded, sampled monthly to 30 years by default. Useful
    for seeing what the parameters *mean*: raise ``σ`` and watch the long end
    fall (the convexity effect), raise ``κ`` and watch the curve flatten
    towards ``θ`` sooner.
    """
    r0 = _check_finite("r0", r0)
    grid = _DEFAULT_GRID if times is None else tuple(times)
    zeros = [-math.log(vasicek_zero_bond(r0, 0.0, T, kappa, theta, sigma)) / _check_positive("times", T) for T in grid]
    return YieldCurve(grid, zeros, "linear", "continuous")


def cir_curve(
    r0: float, kappa: float, theta: float, sigma: float, times: Optional[Sequence[float]] = None
) -> YieldCurve:
    """The zero curve implied by CIR parameters; see :func:`vasicek_curve`."""
    r0 = _check_non_negative("r0", r0)
    grid = _DEFAULT_GRID if times is None else tuple(times)
    zeros = [-math.log(cir_zero_bond(r0, 0.0, T, kappa, theta, sigma)) / _check_positive("times", T) for T in grid]
    return YieldCurve(grid, zeros, "linear", "continuous")


# ---------------------------------------------------------------------------
# Short-rate models — simulation (NumPy)
# ---------------------------------------------------------------------------


def _numpy():
    from .mc import _numpy as _np

    return _np()


def _check_simulation(T: float, steps: int, trials: int) -> Tuple[float, int, int]:
    T = _check_positive("T", T)
    if isinstance(steps, bool) or not isinstance(steps, numbers.Integral) or steps < 1:
        raise ValueError(f"steps must be a positive integer, got {steps!r}")
    if isinstance(trials, bool) or not isinstance(trials, numbers.Integral) or trials < 1:
        raise ValueError(f"trials must be a positive integer, got {trials!r}")
    return T, int(steps), int(trials)


def vasicek_paths(
    r0: float,
    kappa: float,
    theta: float,
    sigma: float,
    T: float,
    steps: int = 252,
    trials: int = 10_000,
    seed: Optional[int] = None,
):
    """Vasicek short-rate paths, shape ``(trials, steps + 1)``.

    Uses the **exact** Gaussian transition rather than an Euler step: over
    ``dt`` the rate is normal with mean ``θ + (r − θ)e^{−κ dt}`` and variance
    ``σ²(1 − e^{−2κ dt})/(2κ)``. Vasicek is one of the few SDEs with a
    closed-form transition, so there is no discretisation error to accept
    and no reason to accept one — the simulated moments match
    ``vasicek_zero_bond`` to sampling error at any ``steps``.

    Rates can and do go negative; that is the model, not a bug. Feed the
    output to :func:`riskpy.viz.fan`, or integrate it with the trapezium rule
    for ``exp(−∫r dt)`` pricing.
    """
    np = _numpy()
    r0 = _check_finite("r0", r0)
    kappa, theta, sigma = _check_short_rate(kappa, theta, sigma)
    T, steps, trials = _check_simulation(T, steps, trials)

    dt = T / steps
    decay = math.exp(-kappa * dt)
    sd = sigma * math.sqrt(-math.expm1(-2.0 * kappa * dt) / (2.0 * kappa))

    rng = np.random.default_rng(seed)
    shocks = rng.standard_normal((trials, steps))
    paths = np.empty((trials, steps + 1))
    paths[:, 0] = r0
    for i in range(steps):
        paths[:, i + 1] = theta + (paths[:, i] - theta) * decay + sd * shocks[:, i]
    return paths


def cir_paths(
    r0: float,
    kappa: float,
    theta: float,
    sigma: float,
    T: float,
    steps: int = 252,
    trials: int = 10_000,
    seed: Optional[int] = None,
):
    """Cox–Ingersoll–Ross short-rate paths, shape ``(trials, steps + 1)``.

    The CIR transition is non-central chi-squared, so this uses the
    **full-truncation Euler** scheme of Lord, Koekkoek and van Dijk (2010):
    an auxiliary process ``x`` steps as
    ``x += κ(θ − x⁺)dt + σ√(x⁺) √dt·Z`` and the reported rate is ``x⁺ =
    max(x, 0)``. It is the least biased of the simple fixes for the square
    root going imaginary, with weak error of order ``dt`` — use at least 100
    steps a year when pricing off the paths.

    The Feller condition ``2κθ > σ²`` keeps the true process away from zero.
    It is warned about, not enforced, as in :func:`riskpy.quant.heston_price`:
    below it the truncation activates often and the paths spend time pinned
    at zero, which the exact process does not.
    """
    np = _numpy()
    r0 = _check_non_negative("r0", r0)
    kappa, theta, sigma = _check_short_rate(kappa, theta, sigma)
    theta = _check_non_negative("theta", theta)
    T, steps, trials = _check_simulation(T, steps, trials)

    if 2.0 * kappa * theta <= sigma * sigma:
        import warnings

        warnings.warn(
            f"Feller condition violated (2·kappa·theta = {2 * kappa * theta:.4f} <= "
            f"sigma² = {sigma * sigma:.4f}). Rates will be truncated at zero more often "
            f"than the exact process reaches it; the closed form cir_zero_bond is unaffected.",
            RuntimeWarning,
            stacklevel=2,
        )

    dt = T / steps
    sqrt_dt = math.sqrt(dt)
    rng = np.random.default_rng(seed)
    shocks = rng.standard_normal((trials, steps))
    paths = np.empty((trials, steps + 1))
    paths[:, 0] = r0
    x = np.full(trials, r0)
    for i in range(steps):
        positive = np.maximum(x, 0.0)
        x = x + kappa * (theta - positive) * dt + sigma * np.sqrt(positive) * sqrt_dt * shocks[:, i]
        paths[:, i + 1] = np.maximum(x, 0.0)
    return paths


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def _verification_checks():
    """Return a list of (name, value, reference, tolerance) tuples."""
    checks = []

    bond = Bond(face=100.0, coupon=0.05, maturity=10.0, frequency=2)
    checks.append(("par bond prices at face when ytm = coupon", bond.price(0.05), 100.0, 1e-10))
    checks.append(("ytm(price(y)) = y", bond.yield_to_maturity(bond.price(0.043)), 0.043, 1e-10))
    zero = Bond(face=100.0, coupon=0.0, maturity=7.0, frequency=2)
    checks.append(("zero-coupon Macaulay duration = maturity", zero.macaulay_duration(0.04), 7.0, 1e-12))

    par = [0.02, 0.025, 0.03, 0.035, 0.0375, 0.04]
    mats = [1.0, 2.0, 3.0, 5.0, 7.0, 10.0]
    curve = YieldCurve.bootstrap(par, mats, frequency=2)
    checks.append(("bootstrap round trip: 10y par rate", curve.par_rate(10.0, 2), 0.04, 1e-10))

    flat = YieldCurve.flat(0.03)
    checks.append(("flat curve df(5) = exp(-0.15)", flat.df(5.0), math.exp(-0.15), 1e-14))
    ns = YieldCurve.nelson_siegel(0.04, -0.02, 0.01, 2.0)
    f = ns.forward(2.0, 5.0)
    checks.append(("forward consistency df(5) = df(2)·exp(-3f)", ns.df(2.0) * math.exp(-3.0 * f), ns.df(5.0), 1e-12))
    checks.append(("Nelson–Siegel z(0) = β0 + β1", nelson_siegel_zero(0.0, 0.04, -0.02, 0.01, 2.0), 0.02, 1e-15))

    # CIR at sigma -> 0 is the deterministic integral, which Vasicek also gives.
    r0, kappa, theta, T = 0.03, 0.5, 0.05, 5.0
    deterministic = math.exp(-(theta * T + (r0 - theta) * -math.expm1(-kappa * T) / kappa))
    checks.append(("CIR bond at sigma = 0 = exp(-∫r dt)", cir_zero_bond(r0, 0.0, T, kappa, theta, 0.0), deterministic, 1e-12))
    checks.append(("Vasicek bond at sigma = 0 = exp(-∫r dt)", vasicek_zero_bond(r0, 0.0, T, kappa, theta, 0.0), deterministic, 1e-12))

    try:
        np = _numpy()
    except ImportError:  # pragma: no cover - environment dependent
        return checks
    paths = vasicek_paths(0.03, 0.5, 0.05, 0.02, T=2.0, steps=100, trials=20_000, seed=2024)
    dt = 2.0 / 100
    integral = dt * (paths.sum(axis=1) - 0.5 * (paths[:, 0] + paths[:, -1]))
    mc_price = float(np.mean(np.exp(-integral)))
    checks.append(("Vasicek bond vs Monte Carlo exp(-∫r dt)", mc_price, vasicek_zero_bond(0.03, 0.0, 2.0, 0.5, 0.05, 0.02), 3e-3))
    return checks
